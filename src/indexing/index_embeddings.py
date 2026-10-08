"""Load and validate transcript embeddings before indexing them in Qdrant."""

import json
import uuid
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIRECTORY = PROJECT_ROOT / "src" / "output"
DATABASE_PATH = PROJECT_ROOT / "data" / "qdrant"

VECTORS_PATH = OUTPUT_DIRECTORY / "sample_embeddings.npy"
METADATA_PATH = OUTPUT_DIRECTORY / "sample_embedding_metadata.json"
COLLECTION_NAME = "podcast_chunks"


vectors = np.load(VECTORS_PATH)

metadata = json.loads(
    METADATA_PATH.read_text(encoding="utf-8")
)

chunks = metadata["chunks"]


print(f"Vector matrix shape: {vectors.shape}")
print(f"Metadata chunks: {len(chunks)}")
print(f"Embedding model: {metadata['model']}")
print(f"Expected dimensions: {metadata['dimensions']}")


if vectors.ndim != 2:
    raise ValueError(
        f"Expected a two-dimensional vector matrix, got {vectors.shape}"
    )

if len(vectors) != len(chunks):
    raise ValueError(
        "The number of vectors does not match the number of metadata chunks: "
        f"{len(vectors)} vectors versus {len(chunks)} chunks"
    )

if vectors.shape[1] != metadata["dimensions"]:
    raise ValueError(
        "The vector dimensions do not match the metadata: "
        f"{vectors.shape[1]} versus {metadata['dimensions']}"
    )


first_chunk = chunks[0]
first_vector = vectors[0]

print()
print("First vector-to-chunk correlation")
print(f"Vector index: {first_chunk['vector_index']}")
print(f"Chunk ID: {first_chunk['chunk_id']}")
print(f"Text: {first_chunk['text']}")
print(f"Vector dimensions: {len(first_vector)}")
print(f"First five vector values: {first_vector[:5]}")


source_metadata = metadata.get("source_metadata", {})

source_identifier = (
    source_metadata.get("source_audio")
    or source_metadata.get("source_transcript")
    or "unknown-source"
)

points = []

for chunk, vector in zip(chunks, vectors):
    point_identifier = (
        f"{source_identifier}:"
        f"{chunk['chunk_id']}:"
        f"{chunk['start']}:"
        f"{chunk['end']}"
    )

    point_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            point_identifier,
        )
    )

    point = models.PointStruct(
        id=point_id,
        vector=vector.tolist(),
        payload={
            **chunk,
            "source_audio": source_metadata.get("source_audio"),
            "embedding_model": metadata["model"],
        },
    )

    points.append(point)


print()
print(f"Prepared {len(points)} Qdrant points")

first_point = points[0]

print("First Qdrant point")
print(f"Point ID: {first_point.id}")
print(f"Vector dimensions: {len(first_point.vector)}")
print(f"Payload: {first_point.payload}")


client = QdrantClient(path=str(DATABASE_PATH))

try:
    if not client.collection_exists(COLLECTION_NAME):
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' does not exist. "
            "Run qdrant_setup.py first."
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True,
    )

    collection_count = client.count(
        collection_name=COLLECTION_NAME,
        exact=True,
    ).count

    print()
    print(f"Indexed {len(points)} points into: {COLLECTION_NAME}")
    print(f"Collection now contains {collection_count} points")

finally:
    client.close()
