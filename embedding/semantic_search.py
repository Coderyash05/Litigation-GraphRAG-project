from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "litigation_documents"

# Connect to Qdrant
client = QdrantClient(
    url="http://127.0.0.1:6333",
    timeout=30,
    check_compatibility=False
)

# Load embedding model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

# User's natural-language query
query = "Find cases involving disputed business expenses"

print(f"\nQuery: {query}")

# Convert query into embedding
query_vector = model.encode(
    query,
    convert_to_numpy=True
).tolist()

# Search Qdrant
results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    limit=3
).points

print("\nTop matching results:")
print("----------------------")

for i, result in enumerate(results, start=1):
    payload = result.payload

    print(f"\nResult {i}")
    print(f"Score: {result.score:.4f}")
    print(f"Case: {payload['case_id']}")
    print(f"Document: {payload['document_id']}")
    print(f"Year: {payload['year']}")
    print(f"Text: {payload['text']}")
