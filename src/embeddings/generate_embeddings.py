"""Generate dense embeddings for aggregated transcript chunks.

The vector at row ``i`` in the generated NumPy file corresponds to the
metadata record whose ``vector_index`` is ``i``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

SRC_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = SRC_DIR / "output" / "sample_chunks.json"
DEFAULT_VECTORS = SRC_DIR / "output" / "sample_embeddings.npy"
DEFAULT_METADATA = SRC_DIR / "output" / "sample_embedding_metadata.json"


def load_chunks(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load chunks and retain any metadata stored beside the chunk list."""
    payload = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(payload, list):
        chunks = payload
        source_metadata: dict[str, Any] = {}
    elif isinstance(payload, dict) and isinstance(payload.get("chunks"), list):
        chunks = payload["chunks"]
        source_metadata = {key: value for key, value in payload.items() if key != "chunks"}
    else:
        raise ValueError(
            f"{path} must contain a JSON list or an object with a 'chunks' list"
        )

    if not chunks:
        raise ValueError(f"No chunks were found in {path}")

    validated: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise ValueError(f"Chunk {index} must be a JSON object")

        text = chunk.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Chunk {index} must have a non-empty 'text' field")

        record = dict(chunk)
        record["text"] = text.strip()
        record.setdefault("chunk_id", index)
        validated.append(record)

    return validated, source_metadata


def embed_chunks(
    chunks: list[dict[str, Any]],
    model_name: str,
    batch_size: int,
) -> np.ndarray:
    """Return one normalized dense vector per chunk."""
    model = SentenceTransformer(model_name)
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
        raise RuntimeError(
            "The embedding model did not return exactly one vector per chunk"
        )

    return embeddings.astype(np.float32, copy=False)


def save_outputs(
    chunks: list[dict[str, Any]],
    source_metadata: dict[str, Any],
    embeddings: np.ndarray,
    model_name: str,
    vectors_path: Path,
    metadata_path: Path,
) -> None:
    vectors_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(vectors_path, embeddings)

    manifest = {
        "model": model_name,
        "dimensions": int(embeddings.shape[1]),
        "normalized": True,
        "query_instruction": QUERY_INSTRUCTION,
        "count": len(chunks),
        "source_metadata": source_metadata,
        "chunks": [
            {"vector_index": index, **chunk}
            for index, chunk in enumerate(chunks)
        ],
    }
    metadata_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate normalized BGE embeddings for transcript chunks"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--vectors-output", type=Path, default=DEFAULT_VECTORS)
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    chunks, source_metadata = load_chunks(args.input)
    print(f"Loaded {len(chunks)} chunks from {args.input}")
    print(f"Loading embedding model: {args.model}")

    embeddings = embed_chunks(chunks, args.model, args.batch_size)
    save_outputs(
        chunks=chunks,
        source_metadata=source_metadata,
        embeddings=embeddings,
        model_name=args.model,
        vectors_path=args.vectors_output,
        metadata_path=args.metadata_output,
    )

    print(f"Embedding matrix shape: {embeddings.shape}")
    print(f"Vectors saved to: {args.vectors_output}")
    print(f"Metadata saved to: {args.metadata_output}")


if __name__ == "__main__":
    main()
