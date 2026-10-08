"""Search transcript chunks stored in the local Qdrant collection."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_PATH = PROJECT_ROOT / "data" / "qdrant"

COLLECTION_NAME = "podcast_chunks"
MODEL_NAME = "BAAI/bge-small-en-v1.5"


QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search indexed podcast transcript chunks"
    )
    parser.add_argument(
        "question",
        help="Natural-language question to search for",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of transcript chunks to return (default: 5)",
    )
    return parser.parse_args()


def embed_question(
    model: SentenceTransformer,
    question: str,
) -> np.ndarray:
    """Turn a question into a normalized vector using the BGE query instruction."""
    query_text = QUERY_INSTRUCTION + question.strip()

    return model.encode(
        query_text,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )


def search_qdrant(
    client: QdrantClient,
    query_vector: np.ndarray,
    limit: int,
) -> list[ScoredPoint]:
    """Return the transcript chunks most similar to the query vector."""
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector.tolist(),
        limit=limit,
        with_payload=True,
    )
    return response.points


def format_timestamp(seconds: object) -> str:
    """Format a timestamp such as 75.2 seconds as 01:15.2."""
    try:
        total_seconds = float(seconds)
    except (TypeError, ValueError):
        return "unknown"

    minutes, remaining_seconds = divmod(total_seconds, 60)
    return f"{int(minutes):02d}:{remaining_seconds:04.1f}"


def print_results(results: list[ScoredPoint]) -> None:
    if not results:
        print("No matching transcript chunks were found.")
        return

    print("\nMost relevant transcript chunks")
    for rank, result in enumerate(results, start=1):
        payload = result.payload or {}
        start = format_timestamp(payload.get("start"))
        end = format_timestamp(payload.get("end"))
        text = payload.get("text", "<text unavailable>")

        print(f"\n{rank}. Score: {result.score:.4f} | Time: {start} - {end}")
        print(text)


def main() -> None:
    args = parse_args()
    question = args.question.strip()

    if not question:
        raise ValueError("The question cannot be empty")
    if args.limit < 1:
        raise ValueError("--limit must be at least 1")
    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            f"Qdrant database not found at {DATABASE_PATH}. Run the indexing pipeline first."
        )

    client = QdrantClient(path=str(DATABASE_PATH))
    try:
        if not client.collection_exists(COLLECTION_NAME):
            raise RuntimeError(
                f"Qdrant collection '{COLLECTION_NAME}' does not exist. "
                "Run the indexing pipeline first."
            )

        print(f"Loading embedding model: {MODEL_NAME}")
        model = SentenceTransformer(MODEL_NAME)
        query_vector = embed_question(model, question)

        print(f"Searching for: {question}")
        results = search_qdrant(client, query_vector, args.limit)
        print_results(results)
    finally:
        client.close()


if __name__ == "__main__":
    main()
