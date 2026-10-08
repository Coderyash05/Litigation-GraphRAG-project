"""
Command-line entry point for the embedding layer.

The actual logic lives in embedding/store.py. Chunks and their
embeddings are stored in Neo4j, next to the knowledge graph.

Examples (PowerShell; NEO4J_PASSWORD is read from the .env file):

    # Ingest documents/ with the baseline model and settings
    python embedding_layer.py ingest

    # Try a larger model with smaller chunks (new vector size,
    # so the index must be recreated)
    python embedding_layer.py ingest --model bge-base --chunk-size 500 --overlap 50 --recreate

    # Search (same model as the ingest)
    python embedding_layer.py search "What evidence did ABC Ltd provide?"

    # Search only within one case
    python embedding_layer.py search "court reasoning" --case-id CASE001

    # Find the vector dimension / chunk size with the highest
    # similarity for a query (in memory, does not touch Neo4j)
    python embedding_layer.py compare "What evidence did ABC Ltd provide?"
"""

import argparse
from pathlib import Path

import numpy as np

from embedding.store import (
    DOCUMENTS_DIR,
    MODEL_PRESETS,
    config_from_preset,
    count_chunks,
    document_metadata,
    embed_passages,
    embed_query,
    extract_pdf_chunks,
    get_model,
    ingest_documents,
    load_document_metadata,
    passage_text,
    semantic_search
)
from graph_layer import get_driver


# Settings tried by the compare command
COMPARE_MODELS = list(MODEL_PRESETS)

COMPARE_CHUNKS = ["1200:200", "600:100", "300:50"]


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Litigation document embedding layer"
    )

    common = argparse.ArgumentParser(add_help=False)

    common.add_argument(
        "--model",
        default="bge-small",
        help=(
            f"Preset ({', '.join(MODEL_PRESETS)}) or any "
            f"sentence-transformers model name. Default: bge-small"
        )
    )

    common.add_argument("--chunk-size", type=int)
    common.add_argument("--overlap", type=int)

    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser(
        "ingest",
        parents=[common],
        help="Embed PDFs and store the chunks in Neo4j"
    )

    ingest.add_argument("--documents-dir", default=str(DOCUMENTS_DIR))

    ingest.add_argument(
        "--recreate",
        action="store_true",
        help="Drop the vector index and all embedded chunks first"
    )

    ingest.add_argument(
        "--force",
        action="store_true",
        help="Re-embed documents even if unchanged"
    )

    search = commands.add_parser(
        "search",
        parents=[common],
        help="Semantic search over ingested chunks"
    )

    search.add_argument(
        "query",
        nargs="?",
        help="Query text. Prompted for if omitted"
    )

    search.add_argument("--case-id")
    search.add_argument("--document-id")
    search.add_argument("--limit", type=int, default=5)

    compare = commands.add_parser(
        "compare",
        help=(
            "Try several models (vector dimensions) and chunk sizes "
            "and show the top similarity score for each"
        )
    )

    compare.add_argument(
        "query",
        nargs="?",
        help="Query text. Prompted for if omitted"
    )

    compare.add_argument(
        "--models",
        nargs="+",
        default=COMPARE_MODELS,
        help=f"Default: {' '.join(COMPARE_MODELS)}"
    )

    compare.add_argument(
        "--chunks",
        nargs="+",
        default=COMPARE_CHUNKS,
        help=f"SIZE:OVERLAP values. Default: {' '.join(COMPARE_CHUNKS)}"
    )

    compare.add_argument("--case-id")
    compare.add_argument("--documents-dir", default=str(DOCUMENTS_DIR))

    return parser.parse_args()


# ============================================================
# COMMANDS
# ============================================================

def run_ingest(args, driver, config):

    summary = ingest_documents(
        driver,
        config,
        documents_dir=args.documents_dir,
        recreate=args.recreate,
        force=args.force
    )

    print(
        f"\nDocuments: {summary['documents']} | "
        f"chunks written: {summary['written']} | "
        f"skipped: {summary['skipped']} | "
        f"chunks in Neo4j: {count_chunks(driver, config)}"
    )


def run_search(args, driver, config):

    query = args.query or input(
        "\nEnter your litigation query: "
    ).strip()

    if not query:
        print("No query entered.")
        return

    results = semantic_search(
        driver,
        config,
        query,
        limit=args.limit,
        case_id=args.case_id,
        document_id=args.document_id
    )

    print("\nTop matching results")
    print("====================")

    if not results:
        print("No results found.")
        return

    for index, result in enumerate(results, start=1):

        print(f"\nResult {index}")
        print("--------------------")
        print(f"Similarity Score: {result['score']:.4f}")
        print(f"Case: {result['case_id']}")
        print(f"Document: {result['document_id']}")
        print(
            f"Page: {result['page_number']} | "
            f"Chunk: {result['chunk_number']} | "
            f"Section: {result['section']}"
        )
        print(f"Text:\n{result['text']}")


def load_chunks(config, documents_dir, case_id=None):
    """
    Chunks of every PDF (optionally one case) with the text that
    gets embedded, without storing anything.
    """

    all_metadata = load_document_metadata(documents_dir)

    chunks = []

    for pdf_path in sorted(documents_dir.glob("*.pdf")):

        metadata = document_metadata(pdf_path, all_metadata)

        if case_id and metadata["case_id"] != case_id:
            continue

        for chunk in extract_pdf_chunks(pdf_path, config):
            chunks.append(
                {
                    **chunk,
                    "case_id": metadata["case_id"],
                    "embedded_text": passage_text(metadata, chunk)
                }
            )

    return chunks


def run_compare(args):
    """
    For each model x chunk setting: chunk and embed the PDFs in
    memory, run the query and report the top result.
    """

    query = args.query or input(
        "\nEnter your litigation query: "
    ).strip()

    if not query:
        print("No query entered.")
        return

    documents_dir = Path(args.documents_dir)

    rows = []

    for model in args.models:

        for setting in args.chunks:

            chunk_size, overlap = (
                int(value) for value in setting.split(":")
            )

            config = config_from_preset(
                model,
                chunk_size=chunk_size,
                chunk_overlap=overlap
            )

            print(f"Testing {model} {chunk_size}/{overlap} ...")

            chunks = load_chunks(config, documents_dir, args.case_id)

            if not chunks:
                continue

            vectors = np.array(
                embed_passages(
                    config,
                    [chunk["embedded_text"] for chunk in chunks]
                )
            )

            # Vectors are normalized, so the dot product is the
            # cosine similarity
            scores = vectors @ np.array(embed_query(config, query))

            order = np.argsort(scores)[::-1]

            best = scores[order[0]]

            top = chunks[order[0]]

            rows.append(
                {
                    "model": model,
                    "dims": get_model(
                        config.model_name
                    ).get_embedding_dimension(),
                    "chunk": f"{chunk_size}/{overlap}",
                    "chunks": len(chunks),
                    "score": best,
                    # How far the best chunk is ahead of the next one
                    "margin": (
                        best - scores[order[1]]
                        if len(chunks) > 1 else 0.0
                    ),
                    "where": (
                        f"{top['case_id']} p{top['page_number']} "
                        f"{top['section']}"
                    )
                }
            )

    rows.sort(key=lambda row: row["score"], reverse=True)

    print(f"\nQuery: {query}\n")

    print(
        f"{'Model':<10} {'Dims':>5} {'Chunk':>9} {'Chunks':>6} "
        f"{'Score':>6} {'Margin':>6}  Top result"
    )

    print("-" * 90)

    for row in rows:
        print(
            f"{row['model']:<10} {row['dims']:>5} {row['chunk']:>9} "
            f"{row['chunks']:>6} {row['score']:>6.3f} "
            f"{row['margin']:>6.3f}  {row['where']}"
        )

    print(
        "\nScore  = cosine similarity of the best chunk"
        "\nMargin = lead over the 2nd-best chunk (larger = clearer match)"
        "\nCheck that 'Top result' is the page you expected."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    if args.command == "compare":
        run_compare(args)
        return

    config = config_from_preset(
        args.model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap
    )

    print(
        f"Model: {config.model_name} | "
        f"chunk size: {config.chunk_size} | "
        f"overlap: {config.chunk_overlap}"
    )

    driver = get_driver()

    try:

        if args.command == "ingest":
            run_ingest(args, driver, config)

        else:
            run_search(args, driver, config)

    finally:
        driver.close()


if __name__ == "__main__":
    main()
