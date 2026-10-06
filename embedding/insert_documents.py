from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "litigation_documents"

# Connect to Qdrant
client = QdrantClient(host="localhost", port=6333)

# Load embedding model
print("Loading embedding model...")
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

print("Embedding model loaded.")
print("Embedding dimensions:", model.get_embedding_dimension())

# Dummy litigation document chunks
documents = [
    {
        "id": 1,
        "text": (
            "ABC Ltd challenged the disallowance of business expenditure "
            "in its income tax assessment for the financial year 2024."
        ),
        "case_id": "CASE001",
        "document_id": "DOC001",
        "year": 2024,
        "document_type": "Judgment"
    },
    {
        "id": 2,
        "text": (
            "The dispute concerned whether the expenditure claimed by "
            "the assessee was allowable under Section 37 of the Income Tax Act."
        ),
        "case_id": "CASE001",
        "document_id": "DOC001",
        "year": 2024,
        "document_type": "Judgment"
    },
    {
        "id": 3,
        "text": (
            "The court considered the evidence submitted by the taxpayer "
            "and examined the applicability of the relevant provisions "
            "of the Income Tax Act."
        ),
        "case_id": "CASE001",
        "document_id": "DOC001",
        "year": 2024,
        "document_type": "Judgment"
    },
    {
        "id": 4,
        "text": (
            "XYZ Pvt Ltd disputed the depreciation claimed on certain "
            "business assets under Section 32 of the Income Tax Act."
        ),
        "case_id": "CASE002",
        "document_id": "DOC002",
        "year": 2023,
        "document_type": "Judgment"
    },
    {
        "id": 5,
        "text": (
            "The appellate proceedings examined whether the taxpayer "
            "was entitled to claim depreciation on the disputed assets."
        ),
        "case_id": "CASE002",
        "document_id": "DOC002",
        "year": 2023,
        "document_type": "Judgment"
    }
]

print("\nGenerating embeddings...")

points = []

for document in documents:

    embedding = model.encode(
        document["text"],
        convert_to_numpy=True
    ).tolist()

    point = PointStruct(
        id=document["id"],
        vector=embedding,
        payload={
            "text": document["text"],
            "case_id": document["case_id"],
            "document_id": document["document_id"],
            "year": document["year"],
            "document_type": document["document_type"]
        }
    )

    points.append(point)

    print(
        f"Embedded document {document['id']} "
        f"({document['case_id']})"
    )

# Upload to Qdrant
client.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)

print("\nDocuments successfully stored in Qdrant.")

# Verify
info = client.get_collection(COLLECTION_NAME)

print("\nCollection information:")
print("Points:", info.points_count)
print("Status:", info.status)