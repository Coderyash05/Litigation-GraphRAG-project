import os
from pathlib import Path

from pypdf import PdfReader

from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams
)


# ============================================================
# CONFIGURATION
# ============================================================

QDRANT_URL = "http://127.0.0.1:6333"

COLLECTION_NAME = "litigation_documents"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

VECTOR_SIZE = 384

# Number of characters in each text chunk
CHUNK_SIZE = 200

# Number of characters repeated between consecutive chunks
CHUNK_OVERLAP = 20

# Folder containing PDF documents
DOCUMENTS_DIR = Path("documents")


# ============================================================
# CONNECT TO QDRANT
# ============================================================

print("\nConnecting to Qdrant...")

client = QdrantClient(
    url=QDRANT_URL,
    timeout=30,
    check_compatibility=False
)

print("Qdrant connection successful.")


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

model = SentenceTransformer(
    MODEL_NAME,
    device="cpu"
)

print("Embedding model loaded.")
print("Embedding dimensions:", model.get_embedding_dimension())


# ============================================================
# CREATE / RESET QDRANT COLLECTION
# ============================================================

print("\nPreparing Qdrant collection...")

existing_collections = [
    collection.name
    for collection in client.get_collections().collections
]

if COLLECTION_NAME in existing_collections:

    print(
        f"Deleting existing collection: "
        f"{COLLECTION_NAME}"
    )

    client.delete_collection(
        collection_name=COLLECTION_NAME
    )

    print("Old collection deleted.")


print(
    f"Creating collection: "
    f"{COLLECTION_NAME}"
)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=VECTOR_SIZE,
        distance=Distance.COSINE
    )
)

print("Collection created.")


# ============================================================
# FIND PDF DOCUMENTS
# ============================================================

if not DOCUMENTS_DIR.exists():

    print(
        f"\nERROR: Documents folder does not exist: "
        f"{DOCUMENTS_DIR.resolve()}"
    )

    raise SystemExit(1)


pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

if not pdf_files:

    print(
        f"\nERROR: No PDF files found in "
        f"{DOCUMENTS_DIR.resolve()}"
    )

    raise SystemExit(1)


print("\nPDF documents found:")

for pdf_file in pdf_files:
    print("-", pdf_file.name)


# ============================================================
# TEXT CHUNKING FUNCTION
# ============================================================

def create_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split text into overlapping chunks.

    Example:

        Chunk 1: characters 0 - 1200
        Chunk 2: characters 1000 - 2200
        Chunk 3: characters 2000 - 3200

    The overlap helps preserve context between chunks.
    """

    if not text:
        return []

    if overlap >= chunk_size:
        raise ValueError(
            "CHUNK_OVERLAP must be smaller than CHUNK_SIZE."
        )

    chunks = []

    start = 0

    text_length = len(text)

    while start < text_length:

        end = min(
            start + chunk_size,
            text_length
        )

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - overlap

    return chunks


# ============================================================
# EXTRACT PDF TEXT
# ============================================================

def extract_pdf_chunks(pdf_path):
    """
    Extract text from every PDF page and create chunks.

    Each chunk keeps its page number.
    """

    print(f"\nReading PDF: {pdf_path.name}")

    reader = PdfReader(str(pdf_path))

    print(f"Number of pages: {len(reader.pages)}")

    chunks = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        page_text = page.extract_text() or ""

        page_text = page_text.strip()

        if not page_text:

            print(
                f"Page {page_number}: "
                f"No extractable text"
            )

            continue

        page_chunks = create_chunks(page_text)

        print(
            f"Page {page_number}: "
            f"{len(page_chunks)} chunk(s)"
        )

        for chunk_number, chunk_text in enumerate(
            page_chunks,
            start=1
        ):

            chunks.append(
                {
                    "page_number": page_number,
                    "chunk_number": chunk_number,
                    "text": chunk_text
                }
            )

    return chunks


# ============================================================
# PROCESS DOCUMENTS
# ============================================================

all_chunks = []

print("\nExtracting text from PDFs...")

for pdf_path in pdf_files:

    document_id = pdf_path.stem

    chunks = extract_pdf_chunks(pdf_path)

    for chunk in chunks:

        chunk["document_id"] = document_id
        chunk["file_name"] = pdf_path.name

        all_chunks.append(chunk)


if not all_chunks:

    print(
        "\nERROR: No text could be extracted "
        "from the PDF documents."
    )

    print(
        "If your PDF contains scanned images rather than "
        "selectable text, OCR will be required."
    )

    raise SystemExit(1)


print(
    f"\nTotal chunks extracted: "
    f"{len(all_chunks)}"
)


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

print("\nGenerating embeddings...")

points = []

for index, chunk in enumerate(
    all_chunks,
    start=1
):

    print(
        f"Embedding chunk {index}/"
        f"{len(all_chunks)}..."
    )

    embedding = model.encode(
        chunk["text"],
        convert_to_numpy=True
    ).tolist()

    # Use a deterministic integer ID for this run.
    point_id = index

    point = PointStruct(
        id=point_id,

        vector=embedding,

        payload={
            "text": chunk["text"],
            "document_id": chunk["document_id"],
            "file_name": chunk["file_name"],
            "page_number": chunk["page_number"],
            "chunk_number": chunk["chunk_number"]
        }
    )

    points.append(point)


# ============================================================
# STORE EMBEDDINGS IN QDRANT
# ============================================================

print("\nUploading embeddings to Qdrant...")

client.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)

print(
    f"Successfully uploaded "
    f"{len(points)} vector(s)."
)


# ============================================================
# VERIFY COLLECTION
# ============================================================

collection_info = client.get_collection(
    COLLECTION_NAME
)

print("\nQdrant collection status")
print("========================")

print(
    "Collection:",
    COLLECTION_NAME
)

print(
    "Points:",
    collection_info.points_count
)

print(
    "Status:",
    collection_info.status
)


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def semantic_search(query, limit=5):

    print("\nGenerating query embedding...")

    query_vector = model.encode(
        query,
        convert_to_numpy=True
    ).tolist()

    print("Searching Qdrant...")

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit
    ).points

    return results


# ============================================================
# USER QUERY
# ============================================================

query = input(
    "\nEnter your litigation query: "
).strip()


if not query:

    print("\nNo query entered.")

    raise SystemExit(0)


# ============================================================
# SEARCH
# ============================================================

results = semantic_search(
    query,
    limit=5
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\nTop matching results")
print("====================")


if not results:

    print("No results found.")

else:

    for index, result in enumerate(
        results,
        start=1
    ):

        payload = result.payload

        print(
            f"\nResult {index}"
        )

        print("--------------------")

        print(
            f"Similarity Score: "
            f"{result.score:.4f}"
        )

        print(
            f"Document: "
            f"{payload['file_name']}"
        )

        print(
            f"Page: "
            f"{payload['page_number']}"
        )

        print(
            f"Chunk: "
            f"{payload['chunk_number']}"
        )

        print(
            f"Text:\n"
            f"{payload['text']}"
        )


print("\nEmbedding layer completed.")