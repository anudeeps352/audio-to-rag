"""Create the local Qdrant collection used for podcast chunks."""

from pathlib import Path

from qdrant_client import QdrantClient, models


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_PATH = PROJECT_ROOT / "data" / "qdrant"

COLLECTION_NAME = "podcast_chunks"
VECTOR_SIZE = 384


client = QdrantClient(path=str(DATABASE_PATH))

if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=VECTOR_SIZE,
            distance=models.Distance.COSINE,
        ),
    )
    print(f"Created collection: {COLLECTION_NAME}")
else:
    print(f"Collection already exists: {COLLECTION_NAME}")

print(client.get_collections())

client.close()
