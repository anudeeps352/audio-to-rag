"""Aggregate adjacent transcript segments into retrieval-sized chunks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SRC_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = SRC_DIR / "output" / "transcript.json"
DEFAULT_OUTPUT = SRC_DIR / "output" / "sample_chunks.json"

MAX_WORDS = 180
MAX_DURATION_SECONDS = 75
MAX_GAP_SECONDS = 3
OVERLAP_SEGMENTS = 1


def build_chunk(chunk_id: int, source_segments: list[dict[str, Any]]) -> dict[str, Any]:
    """Create one chunk while preserving source segment and time metadata."""
    combined_text = " ".join(
        segment["text"].strip()
        for segment in source_segments
        if segment["text"].strip()
    )
    return {
        "chunk_id": chunk_id,
        "start": source_segments[0]["start"],
        "end": source_segments[-1]["end"],
        "duration_seconds": round(
            source_segments[-1]["end"] - source_segments[0]["start"], 3
        ),
        "text": combined_text,
        "word_count": len(combined_text.split()),
        "source_segment_ids": [segment["id"] for segment in source_segments],
    }


def aggregate_segments(
    segments: list[dict[str, Any]],
    max_words: int = MAX_WORDS,
    max_duration_seconds: float = MAX_DURATION_SECONDS,
    max_gap_seconds: float = MAX_GAP_SECONDS,
    overlap_segments: int = OVERLAP_SEGMENTS,
) -> list[dict[str, Any]]:
    """Group adjacent transcript segments using size, duration, and gap limits."""
    chunks: list[dict[str, Any]] = []
    current_segments: list[dict[str, Any]] = []
    current_word_count = 0

    for segment in segments:
        segment_text = segment["text"].strip()
        if not segment_text:
            continue

        segment_word_count = len(segment_text.split())
        if current_segments:
            proposed_word_count = current_word_count + segment_word_count
            proposed_duration = segment["end"] - current_segments[0]["start"]
            gap_from_previous = max(
                0,
                segment["start"] - current_segments[-1]["end"],
            )

            exceeds_word_limit = proposed_word_count > max_words
            exceeds_duration_limit = proposed_duration > max_duration_seconds
            exceeds_gap_limit = gap_from_previous > max_gap_seconds

            if exceeds_word_limit or exceeds_duration_limit or exceeds_gap_limit:
                chunks.append(build_chunk(len(chunks), current_segments))

                # Silence is a hard boundary. Size boundaries retain a small
                # amount of context through segment overlap.
                if exceeds_gap_limit or overlap_segments == 0:
                    current_segments = []
                else:
                    current_segments = current_segments[-overlap_segments:]

                current_word_count = sum(
                    len(item["text"].split()) for item in current_segments
                )

        current_segments.append(segment)
        current_word_count += segment_word_count

    if current_segments:
        chunks.append(build_chunk(len(chunks), current_segments))

    return chunks


def aggregate_transcript(input_path: Path, output_path: Path) -> list[dict[str, Any]]:
    """Read a transcript, aggregate its segments, and write chunk JSON."""
    transcript = json.loads(input_path.read_text(encoding="utf-8"))
    segments = transcript["segments"]
    chunks = aggregate_segments(segments)

    output_data = {
        "schema_version": 1,
        "source_transcript": str(input_path),
        "source_audio": transcript.get("source_audio"),
        "transcription": transcript.get("transcription", {}),
        "chunking": {
            "strategy": "adjacent_segment_aggregation",
            "speaker_aware": False,
            "max_words": MAX_WORDS,
            "max_duration_seconds": MAX_DURATION_SECONDS,
            "max_gap_seconds": MAX_GAP_SECONDS,
            "overlap_segments": OVERLAP_SEGMENTS,
        },
        "chunks": chunks,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Input segments: {len(segments)}")
    print(f"Output chunks: {len(chunks)}")
    print(f"Saved chunks to: {output_path}")
    for chunk in chunks:
        print(
            f"Chunk {chunk['chunk_id']}: "
            f"{chunk['start']:.2f}s -> {chunk['end']:.2f}s, "
            f"{chunk['word_count']} words, "
            f"segments {chunk['source_segment_ids']}"
        )
    return chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate transcript segments")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aggregate_transcript(args.input, args.output)


if __name__ == "__main__":
    main()
