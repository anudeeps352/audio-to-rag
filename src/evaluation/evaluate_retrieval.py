"""Evaluate dense transcript retrieval against a labeled benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from src.retrieval.search import (
    COLLECTION_NAME,
    DATABASE_PATH,
    MODEL_NAME,
    embed_question,
    search_qdrant,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK = PROJECT_ROOT / "benchmarks" / "benchmark.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "benchmarks" / "results" / "dense_baseline.json"


def report_path(path: Path) -> str:
    """Return a repository-relative path when the file is inside the project."""
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate dense Qdrant retrieval against labeled questions"
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=DEFAULT_BENCHMARK,
        help="Path to the benchmark JSON file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of chunks to retrieve for each question (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path where the JSON evaluation report will be saved",
    )
    return parser.parse_args()


def load_queries(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    queries = payload.get("queries")

    if not isinstance(queries, list) or not queries:
        raise ValueError(f"{path} must contain a non-empty 'queries' list")

    seen_ids: set[str] = set()
    for position, query in enumerate(queries, start=1):
        if not isinstance(query, dict):
            raise ValueError(f"Benchmark query {position} must be an object")

        query_id = query.get("id")
        question = query.get("question")
        answerable = query.get("answerable")
        relevant_ids = query.get("relevant_chunk_ids")

        if not isinstance(query_id, str) or not query_id.strip():
            raise ValueError(f"Benchmark query {position} has an invalid 'id'")
        if query_id in seen_ids:
            raise ValueError(f"Duplicate benchmark query ID: {query_id}")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Benchmark query {query_id} has an invalid question")
        if not isinstance(answerable, bool):
            raise ValueError(f"Benchmark query {query_id} needs a boolean 'answerable'")
        if not isinstance(relevant_ids, list) or not all(
            isinstance(chunk_id, int) for chunk_id in relevant_ids
        ):
            raise ValueError(
                f"Benchmark query {query_id} needs integer 'relevant_chunk_ids'"
            )
        if answerable and not relevant_ids:
            raise ValueError(
                f"Answerable benchmark query {query_id} needs a relevant chunk"
            )
        if not answerable and relevant_ids:
            raise ValueError(
                f"Unanswerable benchmark query {query_id} cannot have relevant chunks"
            )

        seen_ids.add(query_id)

    return queries


def retrieved_chunk_ids(results: list[Any]) -> list[int]:
    chunk_ids: list[int] = []

    for result in results:
        payload = result.payload or {}
        chunk_id = payload.get("chunk_id")
        if isinstance(chunk_id, int):
            chunk_ids.append(chunk_id)

    return chunk_ids


def serialize_results(results: list[Any]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []

    for rank, result in enumerate(results, start=1):
        payload = result.payload or {}
        serialized.append(
            {
                "rank": rank,
                "chunk_id": payload.get("chunk_id"),
                "score": round(float(result.score), 6),
            }
        )

    return serialized


def first_relevant_rank(
    retrieved_ids: list[int],
    relevant_ids: set[int],
) -> int | None:
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return rank
    return None


def recall_at_k(
    retrieved_ids: list[int],
    relevant_ids: set[int],
    k: int,
) -> float:
    retrieved_at_k = set(retrieved_ids[:k])
    return len(retrieved_at_k & relevant_ids) / len(relevant_ids)


def main() -> None:
    args = parse_args()

    if args.limit < 1:
        raise ValueError("--limit must be at least 1")
    if not args.benchmark.exists():
        raise FileNotFoundError(f"Benchmark file not found: {args.benchmark}")
    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            f"Qdrant database not found at {DATABASE_PATH}. "
            "Run the indexing pipeline first."
        )

    queries = load_queries(args.benchmark)
    print(f"Loaded {len(queries)} benchmark queries from {args.benchmark}")
    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    client = QdrantClient(path=str(DATABASE_PATH))
    query_reports: list[dict[str, Any]] = []
    try:
        if not client.collection_exists(COLLECTION_NAME):
            raise RuntimeError(
                f"Collection '{COLLECTION_NAME}' does not exist. "
                "Run the indexing pipeline first."
            )

        evaluated_count = 0
        recall_at_1_total = 0.0
        recall_at_limit_total = 0.0
        reciprocal_rank_total = 0.0

        print("\nPer-query results")

        for query in queries:
            query_id = query["id"]
            question = query["question"]
            query_vector = embed_question(model, question)
            results = search_qdrant(client, query_vector, args.limit)
            retrieved_ids = retrieved_chunk_ids(results)
            serialized_results = serialize_results(results)

            if not query["answerable"]:
                query_reports.append(
                    {
                        "query_id": query_id,
                        "question": question,
                        "query_type": query.get("query_type"),
                        "answerable": False,
                        "relevant_chunk_ids": [],
                        "retrieved": serialized_results,
                        "first_relevant_rank": None,
                        "included_in_metrics": False,
                    }
                )
                print(
                    f"{query_id}: UNANSWERABLE | retrieved={retrieved_ids} "
                    "| excluded from Recall/MRR"
                )
                continue

            relevant_ids = set(query["relevant_chunk_ids"])
            rank = first_relevant_rank(retrieved_ids, relevant_ids)
            recall_1 = recall_at_k(retrieved_ids, relevant_ids, 1)
            recall_limit = recall_at_k(
                retrieved_ids,
                relevant_ids,
                args.limit,
            )
            reciprocal_rank = 0.0 if rank is None else 1.0 / rank

            evaluated_count += 1
            recall_at_1_total += recall_1
            recall_at_limit_total += recall_limit
            reciprocal_rank_total += reciprocal_rank

            query_reports.append(
                {
                    "query_id": query_id,
                    "question": question,
                    "query_type": query.get("query_type"),
                    "answerable": True,
                    "relevant_chunk_ids": sorted(relevant_ids),
                    "retrieved": serialized_results,
                    "first_relevant_rank": rank,
                    "recall_at_1": round(recall_1, 6),
                    f"recall_at_{args.limit}": round(recall_limit, 6),
                    "reciprocal_rank": round(reciprocal_rank, 6),
                    "included_in_metrics": True,
                }
            )

            rank_text = "not found" if rank is None else str(rank)
            print(
                f"{query_id}: first_relevant_rank={rank_text} "
                f"| relevant={sorted(relevant_ids)} "
                f"| retrieved={retrieved_ids}"
            )
    finally:
        client.close()

    if evaluated_count == 0:
        raise ValueError("The benchmark contains no answerable queries")

    recall_at_1_mean = recall_at_1_total / evaluated_count
    recall_at_limit_mean = recall_at_limit_total / evaluated_count
    mrr = reciprocal_rank_total / evaluated_count

    report = {
        "retrieval_method": "dense",
        "embedding_model": MODEL_NAME,
        "collection": COLLECTION_NAME,
        "retrieval_limit": args.limit,
        "benchmark": report_path(args.benchmark),
        "total_queries": len(queries),
        "answerable_queries": evaluated_count,
        "unanswerable_queries": len(queries) - evaluated_count,
        "metrics": {
            "recall_at_1": round(recall_at_1_mean, 6),
            f"recall_at_{args.limit}": round(recall_at_limit_mean, 6),
            "mrr": round(mrr, 6),
        },
        "query_results": query_reports,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nDense retrieval metrics")
    print(f"Answerable queries: {evaluated_count}")
    print(f"Recall@1: {recall_at_1_mean:.4f}")
    print(f"Recall@{args.limit}: {recall_at_limit_mean:.4f}")
    print(f"MRR: {mrr:.4f}")
    print(f"Report saved to: {args.output}")


if __name__ == "__main__":
    main()
