"""
Embedding layer: PDF -> chunks -> embeddings -> Neo4j -> semantic search.

Chunks are stored in the same Neo4j database as the knowledge graph:

    (:DocumentChunk {text, embedding, ...})-[:PART_OF]->(:Document)
    (:Document)-[:RELATES_TO]->(:Case)

and searched through a Neo4j vector index. Requires Neo4j 5.18+.

Every function takes a Neo4j driver (graph_layer.get_driver()), so
ingestion scripts and LangGraph nodes can all import the same code.

Nothing runs on import.
"""

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_DEVICE = os.environ.get("EMBEDDING_DEVICE", "cpu")

DOCUMENTS_DIR = Path("documents")

# Per-document metadata (case_id, document_id, ...) lives here,
# keyed by PDF file name.
METADATA_FILE_NAME = "metadata.json"

# Neo4j vector index over DocumentChunk.embedding
VECTOR_INDEX = "chunk_embeddings"

# Bump when chunking or embedded-text logic changes, so every
# document is re-embedded on the next ingest.
CHUNKER_VERSION = 2

# Heading detection (see is_heading)
MAX_HEADING_WORDS = 8
MAX_HEADING_CHARS = 70

# Shorter final chunks are merged into the previous chunk
MIN_CHUNK_CHARS = 120

# End of sentence followed by the start of a new one
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")

BGE_QUERY_PREFIX = (
    "Represent this sentence for searching relevant passages: "
)

# Candidate embedding models for experiments.
# Dimensions are read from the model itself, not hard-coded.
MODEL_PRESETS = {
    # 384 dimensions (current baseline)
    "minilm": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2"
    },
    # 384 dimensions, newer retrieval model of the same size
    "bge-small": {
        "model_name": "BAAI/bge-small-en-v1.5",
        "query_prefix": BGE_QUERY_PREFIX
    },
    # 768 dimensions
    "mpnet": {
        "model_name": "sentence-transformers/all-mpnet-base-v2"
    },
    # 768 dimensions
    "bge-base": {
        "model_name": "BAAI/bge-base-en-v1.5",
        "query_prefix": BGE_QUERY_PREFIX
    }
}


@dataclass
class EmbeddingConfig:
    """
    One embedding experiment = one configuration.

    The vector index holds one vector size, so switching to a
    model with a different size needs `ingest --recreate`.
    """

    model_name: str = MODEL_PRESETS["bge-small"]["model_name"]
    query_prefix: str = MODEL_PRESETS["bge-small"]["query_prefix"]

    # Characters per chunk / characters repeated between chunks
    chunk_size: int = 1200
    chunk_overlap: int = 200

    # Some models (e.g. BGE) expect an instruction before the query
    # (query_prefix, set above)
    passage_prefix: str = ""

    def __post_init__(self):

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )


def config_from_preset(preset_or_model_name, **overrides):
    """
    Build a config from a preset name ("bge-base") or any
    Hugging Face sentence-transformers model name.
    """

    settings = {
        # Only models that need a query prefix define one
        "query_prefix": "",
        **MODEL_PRESETS.get(
            preset_or_model_name,
            {"model_name": preset_or_model_name}
        )
    }

    settings.update(
        {
            key: value
            for key, value in overrides.items()
            if value is not None
        }
    )

    return EmbeddingConfig(**settings)


# ============================================================
# EMBEDDING MODEL
# ============================================================

_loaded_models = {}


def get_model(model_name):
    """
    Load each embedding model once per process.

    Runs on the CPU by default: these models are small and the
    document set is small, so the GPU is not needed. Set
    EMBEDDING_DEVICE=cuda to use the GPU.
    """

    if model_name not in _loaded_models:

        # Use the cached copy without contacting huggingface.co
        # (a stale saved login there fails even for public models);
        # download only when the model is not cached yet
        try:
            model = SentenceTransformer(
                model_name,
                device=EMBEDDING_DEVICE,
                local_files_only=True
            )
        except OSError:
            model = SentenceTransformer(
                model_name,
                device=EMBEDDING_DEVICE
            )

        _loaded_models[model_name] = model

    return _loaded_models[model_name]


def embed_passages(config, texts):

    model = get_model(config.model_name)

    return model.encode(
        [config.passage_prefix + text for text in texts],
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False
    ).tolist()


def embed_query(config, query):

    model = get_model(config.model_name)

    return model.encode(
        config.query_prefix + query,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).tolist()


# ============================================================
# VECTOR INDEX SETUP
# ============================================================

def ensure_vector_index(driver, config, recreate=False):
    """
    Create the vector index if needed. With recreate=True, drop
    the index and every embedded chunk first.
    """

    vector_size = get_model(
        config.model_name
    ).get_embedding_dimension()

    if recreate:
        driver.execute_query(f"DROP INDEX {VECTOR_INDEX} IF EXISTS")
        driver.execute_query(
            "MATCH (c:DocumentChunk) WHERE c.embedding IS NOT NULL "
            "DETACH DELETE c"
        )

    records, _, _ = driver.execute_query(
        "SHOW INDEXES YIELD name, options WHERE name = $name "
        "RETURN options",
        name=VECTOR_INDEX,
        routing_="r"
    )

    if records:

        existing_size = records[0]["options"]["indexConfig"][
            "vector.dimensions"
        ]

        if existing_size != vector_size:
            raise ValueError(
                f"Vector index '{VECTOR_INDEX}' stores "
                f"{existing_size}-dimensional vectors, but "
                f"{config.model_name} produces {vector_size}. "
                f"Re-run ingest with --recreate."
            )

        return vector_size

    # vector_size is an int, so formatting it in is safe
    driver.execute_query(
        f"""
        CREATE VECTOR INDEX {VECTOR_INDEX} IF NOT EXISTS
        FOR (c:DocumentChunk) ON c.embedding
        OPTIONS {{indexConfig: {{
            `vector.dimensions`: {int(vector_size)},
            `vector.similarity_function`: 'cosine'
        }}}}
        """
    )

    driver.execute_query("CALL db.awaitIndexes(300)")

    return vector_size




# ============================================================
# DOCUMENT METADATA
# ============================================================

def load_document_metadata(documents_dir=DOCUMENTS_DIR):

    metadata_path = Path(documents_dir) / METADATA_FILE_NAME

    if not metadata_path.exists():
        return {}

    return json.loads(
        metadata_path.read_text(encoding="utf-8")
    )


def document_metadata(pdf_path, all_metadata):
    """
    Metadata for one PDF.

    case_id comes from metadata.json, or failing that from a
    CASE### prefix in the file name.
    """

    metadata = dict(all_metadata.get(pdf_path.name, {}))

    if not metadata.get("case_id"):

        match = re.search(r"CASE\d+", pdf_path.stem, re.IGNORECASE)

        metadata["case_id"] = match.group(0).upper() if match else None

    metadata.setdefault("document_id", pdf_path.stem)

    metadata["file_name"] = pdf_path.name

    return metadata


# ============================================================
# CHUNKING
# ============================================================

def create_chunks(text, chunk_size, overlap):
    """
    Split text into overlapping chunks of about chunk_size
    characters.

    Chunks end and start on word boundaries where possible,
    so words are not cut in half.
    """

    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    chunks = []

    start = 0

    text_length = len(text)

    while start < text_length:

        end = min(start + chunk_size, text_length)

        if end < text_length:

            # Search only past start + overlap, so the next chunk
            # always moves forward.
            last_space = text.rfind(" ", start + overlap + 1, end)

            if last_space != -1:
                end = last_space

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        next_start = end - overlap

        # Move forward to the start of the next word
        if text[next_start - 1] != " ":

            next_space = text.find(" ", next_start, end)

            if next_space != -1:
                next_start = next_space + 1

        start = next_start

    return chunks


def is_heading(line):
    """
    A heading is a short line that starts with a capital letter
    and does not end like a sentence, e.g. "Arguments of the
    Taxpayer" or "Court's Consideration".
    """

    words = line.split()

    return (
        0 < len(words) <= MAX_HEADING_WORDS
        and len(line) <= MAX_HEADING_CHARS
        and line[0].isupper()
        and line[-1] not in ".,;:"
    )


def split_sections(page_text):
    """
    Split one page into (heading, lead_lines, body) sections.

    With consecutive heading lines (e.g. document title, case
    title, "Background"), the last one is the section heading and
    the earlier ones are kept as lead lines of that section, so
    they appear once instead of being repeated on every chunk.
    """

    # Re-join words hyphenated across a line break
    page_text = re.sub(r"(\w)-\n(\w)", r"\1\2", page_text)

    sections = []

    heading_lines = []
    body_lines = []

    for line in page_text.splitlines():

        line = re.sub(r"\s+", " ", line).strip()

        if not line:
            continue

        if is_heading(line):

            # A heading after body text starts a new section
            if body_lines:
                sections.append((heading_lines, body_lines))
                heading_lines, body_lines = [], []

            heading_lines.append(line)

        else:
            body_lines.append(line)

    if heading_lines or body_lines:
        sections.append((heading_lines, body_lines))

    return [
        (
            heading[-1] if heading else "",
            heading[:-1],
            " ".join(body)
        )
        for heading, body in sections
    ]


def split_sentences(text):

    return [
        sentence.strip()
        for sentence in SENTENCE_BOUNDARY.split(text)
        if sentence.strip()
    ]


def pack_sentences(sentences, chunk_size, overlap):
    """
    Group whole sentences into pieces of at most chunk_size
    characters.

    Overlap repeats the last sentence(s) of the previous piece,
    but only whole sentences that fit within `overlap`
    characters.

    Sentences are never cut: a sentence longer than chunk_size
    becomes its own piece. Only a sentence more than twice
    chunk_size falls back to a character split.
    """

    pieces = []

    current = []

    for sentence in sentences:

        if len(sentence) > 2 * chunk_size:

            if current:
                pieces.append(" ".join(current))
                current = []

            pieces.extend(create_chunks(sentence, chunk_size, overlap))

            continue

        if current and len(" ".join(current + [sentence])) > chunk_size:

            pieces.append(" ".join(current))

            carry = []

            for previous in reversed(current):

                if len(" ".join([previous] + carry)) > overlap:
                    break

                carry.insert(0, previous)

            current = carry

        current.append(sentence)

    if current:
        pieces.append(" ".join(current))

    return pieces


def chunk_page(page_text, chunk_size, overlap):
    """
    Section-aware chunking for one page:

    1. Split the page into sections at headings.
    2. Split long sections into sentence-aligned pieces.
    3. Merge neighbouring small sections while they fit in
       chunk_size (large chunk_size -> whole page, small
       chunk_size -> one section per chunk).
    4. Merge a very short last piece into the previous one.

    Returns [{"section": ..., "text": ...}, ...]
    """

    # Each piece: (heading, body text)
    pieces = []

    for heading, lead_lines, body in split_sections(page_text):

        section_pieces = pack_sentences(
            lead_lines + split_sentences(body),
            # Leave room for the heading in the chunk text
            max(chunk_size - len(heading) - 2, MIN_CHUNK_CHARS),
            overlap
        ) or [""]

        pieces.extend((heading, piece) for piece in section_pieces)

    # Each chunk: list of pieces
    chunks = []

    for piece in pieces:

        if chunks and rendered_length(chunks[-1] + [piece]) <= chunk_size:
            chunks[-1].append(piece)

        else:
            chunks.append([piece])

    if len(chunks) > 1 and rendered_length(chunks[-1]) < MIN_CHUNK_CHARS:
        chunks[-2].extend(chunks.pop())

    return [
        {
            "section": " / ".join(
                dict.fromkeys(heading for heading, _ in chunk if heading)
            ),
            "text": render_pieces(chunk)
        }
        for chunk in chunks
    ]


def render_pieces(pieces):
    """
    "Heading: body" lines; consecutive pieces of the same
    section share one heading.
    """

    lines = []

    previous_heading = None

    for heading, body in pieces:

        if lines and heading == previous_heading:
            lines[-1] = f"{lines[-1]} {body}".strip()

        elif heading and body:
            lines.append(f"{heading}: {body}")

        else:
            lines.append(heading or body)

        previous_heading = heading

    return "\n".join(lines)


def rendered_length(pieces):

    return len(render_pieces(pieces))


def extract_pdf_chunks(pdf_path, config):
    """
    Chunk each PDF page separately, so every chunk keeps
    its page number.
    """

    reader = PdfReader(str(pdf_path))

    chunks = []

    for page_number, page in enumerate(reader.pages, start=1):

        page_chunks = chunk_page(
            page.extract_text() or "",
            config.chunk_size,
            config.chunk_overlap
        )

        for chunk_number, chunk in enumerate(page_chunks, start=1):

            chunks.append(
                {
                    "page_number": page_number,
                    "chunk_number": chunk_number,
                    "section": chunk["section"],
                    "text": chunk["text"]
                }
            )

    return chunks


# ============================================================
# INGESTION
# ============================================================

def passage_text(metadata, chunk):
    """
    Text that is embedded for a chunk: the chunk plus the case
    it belongs to. A chunk on its own often never names the case
    or company ("The court examined the invoices..."), so this
    context raises similarity for queries that mention them.

    The stored chunk "text" stays the plain chunk.
    """

    case_line = " — ".join(
        str(value)
        for value in (metadata.get("case_id"), metadata.get("title"))
        if value
    )

    if not case_line:
        return chunk["text"]

    return f"{case_line}\n{chunk['text']}"


def chunk_id(document_id, page_number, chunk_number):

    return f"{document_id}_P{page_number}_C{chunk_number}"


def ingest_signature(pdf_path, metadata, config):
    """
    Changes whenever the file, its metadata, the model or the
    chunk settings change -> the document must be re-embedded.
    """

    digest = hashlib.sha256(pdf_path.read_bytes())

    digest.update(
        json.dumps(
            [
                CHUNKER_VERSION,
                metadata,
                config.model_name,
                config.passage_prefix,
                config.chunk_size,
                config.chunk_overlap
            ],
            sort_keys=True
        ).encode("utf-8")
    )

    return digest.hexdigest()


def stored_signature(driver, document_id):

    records, _, _ = driver.execute_query(
        """
        MATCH (c:DocumentChunk {document_id: $document_id})
        RETURN c.ingest_signature AS signature
        LIMIT 1
        """,
        document_id=document_id,
        routing_="r"
    )

    return records[0]["signature"] if records else None


# Links the document to its case, if the case is in the graph
UPSERT_DOCUMENT = """
MERGE (d:Document {document_id: $document_id})
SET d.name = coalesce(d.name, $file_name),
    d.document_type = coalesce(d.document_type, $document_type)
WITH d
OPTIONAL MATCH (c:Case {case_id: $case_id})
FOREACH (_ IN CASE WHEN c IS NULL THEN [] ELSE [1] END |
    MERGE (d)-[:RELATES_TO]->(c)
)
RETURN c IS NOT NULL AS case_found
"""

# Also removes chunks created by hand (no document_id property,
# only a PART_OF relationship)
DELETE_CHUNKS = """
MATCH (c:DocumentChunk)
WHERE c.document_id = $document_id
   OR EXISTS { (c)-[:PART_OF]->(:Document {document_id: $document_id}) }
DETACH DELETE c
"""

CREATE_CHUNKS = """
MATCH (d:Document {document_id: $document_id})
WITH d LIMIT 1
UNWIND $chunks AS chunk
CREATE (c:DocumentChunk)
SET c = chunk
CREATE (c)-[:PART_OF]->(d)
"""


def ingest_pdf(driver, config, pdf_path, metadata, force=False, log=print):
    """
    Embed one PDF and write its chunks to Neo4j.

    Returns the number of chunks written, or None if the
    document was unchanged and skipped.
    """

    signature = ingest_signature(pdf_path, metadata, config)

    document_id = metadata["document_id"]

    if not force and stored_signature(driver, document_id) == signature:
        return None

    records, _, _ = driver.execute_query(
        UPSERT_DOCUMENT,
        document_id=document_id,
        file_name=metadata["file_name"],
        document_type=metadata.get("document_type"),
        case_id=metadata["case_id"]
    )

    if not any(record["case_found"] for record in records):
        log(
            f"WARNING: case {metadata['case_id']} is not in the "
            f"graph; {pdf_path.name} is not linked to a Case node."
        )

    # Remove this document's old chunks first. With different
    # chunk settings the old run may have had more chunks.
    driver.execute_query(DELETE_CHUNKS, document_id=document_id)

    chunks = extract_pdf_chunks(pdf_path, config)

    if not chunks:
        return 0

    vectors = embed_passages(
        config,
        [passage_text(metadata, chunk) for chunk in chunks]
    )

    driver.execute_query(
        CREATE_CHUNKS,
        document_id=document_id,
        chunks=[
            {
                **chunk,
                "chunk_id": chunk_id(
                    document_id,
                    chunk["page_number"],
                    chunk["chunk_number"]
                ),
                "case_id": metadata["case_id"],
                "document_id": document_id,
                "embedding": vector,
                "model_name": config.model_name,
                "chunk_size": config.chunk_size,
                "chunk_overlap": config.chunk_overlap,
                "ingest_signature": signature
            }
            for chunk, vector in zip(chunks, vectors)
        ]
    )

    return len(chunks)


def ingest_documents(
    driver,
    config,
    documents_dir=DOCUMENTS_DIR,
    recreate=False,
    force=False,
    log=print
):
    """
    Ingest every PDF in documents_dir. Unchanged documents
    are skipped, so this is safe to run repeatedly.
    """

    documents_dir = Path(documents_dir)

    pdf_files = sorted(documents_dir.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {documents_dir.resolve()}"
        )

    ensure_vector_index(driver, config, recreate=recreate)

    all_metadata = load_document_metadata(documents_dir)

    summary = {"written": 0, "skipped": 0, "documents": 0}

    for pdf_path in pdf_files:

        metadata = document_metadata(pdf_path, all_metadata)

        if not metadata["case_id"]:
            log(
                f"WARNING: {pdf_path.name} has no case_id; add it "
                f"to {METADATA_FILE_NAME}. Skipped."
            )
            continue

        written = ingest_pdf(
            driver,
            config,
            pdf_path,
            metadata,
            force=force,
            log=log
        )

        summary["documents"] += 1

        if written is None:
            summary["skipped"] += 1
            log(f"{pdf_path.name}: unchanged, skipped")

        else:
            summary["written"] += written
            log(
                f"{pdf_path.name}: {written} chunk(s) "
                f"[case_id={metadata['case_id']}]"
            )

    return summary


def count_chunks(driver, config):

    records, _, _ = driver.execute_query(
        "MATCH (c:DocumentChunk {model_name: $model_name}) "
        "RETURN count(c) AS chunks",
        model_name=config.model_name,
        routing_="r"
    )

    return records[0]["chunks"]


# ============================================================
# SEMANTIC SEARCH
# ============================================================

CHUNK_FIELDS = """
    c.case_id AS case_id,
    c.document_id AS document_id,
    c.page_number AS page_number,
    c.chunk_number AS chunk_number,
    c.section AS section,
    c.text AS text
"""

# No filter: approximate search over all chunks through the vector index
INDEX_SEARCH = f"""
CALL db.index.vector.queryNodes($index, $limit, $vector)
YIELD node AS c, score
WHERE c.model_name = $model_name
RETURN {CHUNK_FIELDS}, score
ORDER BY score DESC
"""

# Within given cases/document: exact similarity over just those
# chunks, so the filter never cuts the result list short
FILTERED_SEARCH = f"""
MATCH (c:DocumentChunk {{model_name: $model_name}})
WHERE ($case_ids IS NULL OR c.case_id IN $case_ids)
  AND ($document_id IS NULL OR c.document_id = $document_id)
WITH c, vector.similarity.cosine(c.embedding, $vector) AS score
RETURN {CHUNK_FIELDS}, score
ORDER BY score DESC
LIMIT $limit
"""


def semantic_search(
    driver,
    config,
    query,
    limit=5,
    case_id=None,
    document_id=None,
    case_ids=None
):
    """
    Vector similarity search, optionally restricted to one
    case (case_id), several cases (case_ids) and/or one document
    (Step 2.2: search within the nodes identified in the graph).

    Returns [{"case_id", "document_id", "page_number",
    "chunk_number", "section", "text", "score"}, ...], best
    first. Scores are cosine similarity scaled to 0..1.
    """

    if case_id:
        case_ids = [case_id]

    filtered = bool(case_ids or document_id)

    records, _, _ = driver.execute_query(
        FILTERED_SEARCH if filtered else INDEX_SEARCH,
        index=VECTOR_INDEX,
        vector=embed_query(config, query),
        model_name=config.model_name,
        case_ids=list(case_ids) if case_ids else None,
        document_id=document_id,
        limit=limit,
        routing_="r"
    )

    return [record.data() for record in records]
