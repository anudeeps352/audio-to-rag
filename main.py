"""Run the podcast RAG ingestion pipeline end to end."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_AUDIO = SRC_DIR / "audio_samples" / "sample.mp3"
DEFAULT_OUTPUT_DIR = SRC_DIR / "output"
STAGES = ("transcription", "chunking", "embeddings")


def build_stage_commands(args: argparse.Namespace) -> dict[str, list[str]]:
    """Build stage commands with the active Python interpreter."""
    transcript_path = args.output_dir / "transcript.json"
    chunks_path = args.output_dir / "sample_chunks.json"
    return {
        "transcription": [
            sys.executable,
            str(SRC_DIR / "extract" / "extract.py"),
            "--audio",
            str(args.audio),
            "--output-dir",
            str(args.output_dir),
            "--model",
            args.whisper_model,
            "--device",
            args.device,
            "--compute-type",
            args.compute_type,
        ],
        "chunking": [
            sys.executable,
            str(SRC_DIR / "chunking" / "aggregation.py"),
            "--input",
            str(transcript_path),
            "--output",
            str(chunks_path),
        ],
        "embeddings": [
            sys.executable,
            str(SRC_DIR / "embeddings" / "generate_embeddings.py"),
            "--input",
            str(chunks_path),
            "--vectors-output",
            str(args.output_dir / "sample_embeddings.npy"),
            "--metadata-output",
            str(args.output_dir / "sample_embedding_metadata.json"),
            "--model",
            args.embedding_model,
            "--batch-size",
            str(args.batch_size),
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the podcast RAG pipeline")
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--whisper-model", default="small")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="int8_float16")
    parser.add_argument(
        "--embedding-model",
        default="BAAI/bge-small-en-v1.5",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--start-at", choices=STAGES, default=STAGES[0])
    parser.add_argument("--stop-after", choices=STAGES, default=STAGES[-1])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_index = STAGES.index(args.start_at)
    stop_index = STAGES.index(args.stop_after)

    if start_index > stop_index:
        raise ValueError("--start-at must not come after --stop-after")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if start_index == 0 and not args.audio.is_file():
        raise FileNotFoundError(f"Audio file not found: {args.audio}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    commands = build_stage_commands(args)

    for stage in STAGES[start_index : stop_index + 1]:
        print(f"\n=== Running {stage} ===", flush=True)
        subprocess.run(commands[stage], cwd=PROJECT_ROOT, check=True)

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
