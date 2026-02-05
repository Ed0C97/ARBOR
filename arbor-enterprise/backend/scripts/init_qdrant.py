"""Initialize Qdrant collections for local development."""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

def main():
    print("Connecting to Qdrant at http://localhost:6333...")
    client = QdrantClient(url="http://localhost:6333")

    # Check existing collections
    collections = [c.name for c in client.get_collections().collections]
    print(f"Existing collections: {collections}")

    # Create entities_vectors collection
    if "entities_vectors" not in collections:
        print("Creating entities_vectors collection...")
        client.create_collection(
            collection_name="entities_vectors",
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),  # Cohere embed-v4.0 = 1024 dims
        )
        print("OK entities_vectors collection created")
    else:
        print("OK entities_vectors collection already exists")

    # Create semantic_cache collection
    if "semantic_cache" not in collections:
        print("Creating semantic_cache collection...")
        client.create_collection(
            collection_name="semantic_cache",
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )
        print("OK semantic_cache collection created")
    else:
        print("OK semantic_cache collection already exists")

    print("\nQdrant initialization complete!")

if __name__ == "__main__":
    main()
