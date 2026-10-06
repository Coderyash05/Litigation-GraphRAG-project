from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Connect to local Qdrant
client = QdrantClient(host="localhost", port=6333)

COLLECTION_NAME = "litigation_documents"

# Create collection
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)

print(f"Collection '{COLLECTION_NAME}' created successfully.")

# Verify
collections = client.get_collections()

print("\nAvailable collections:")
for collection in collections.collections:
    print("-", collection.name)