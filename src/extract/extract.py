"""Transcribe a local audio file with Faster Whisper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SRC_DIR = Path(__file__).resolve().parents[1]
DEFAULT_AUDIO = SRC_DIR / "audio_samples" / "sample.mp3"
DEFAULT_OUTPUT_DIR = SRC_DIR / "output"


def transcribe_audio(
    audio_path: Path,
    output_directory: Path,
    model_size: str = "small",
    device: str = "cuda",
    compute_type: str = "int8_float16",
) -> tuple[Path, Path]:
    """Transcribe audio and write machine-readable and readable outputs."""
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Delay the expensive optional import so lightweight tooling can import
    # this module without loading the inference stack.
    from faster_whisper import WhisperModel

    output_directory.mkdir(parents=True, exist_ok=True)
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, info = model.transcribe(
        str(audio_path),
        beam_size=5,
        vad_filter=True,
        word_timestamps=True,
        log_progress=True,
    )

    print(
        "Detected language '%s' with probability %f"
        % (info.language, info.language_probability)
    )
    print("Duration:", info.duration)
    print("Duration after VAD:", info.duration_after_vad)

    formatted_segments: list[dict[str, Any]] = []
    text_lines: list[str] = []

    for segment in segments:
        formatted_words = [
            {
                "word": word.word,
                "start": round(word.start, 3) if word.start is not None else None,
                "end": round(word.end, 3) if word.end is not None else None,
                "probability": round(word.probability, 4),
            }
            for word in segment.words or []
        ]
        formatted_segments.append(
            {
                "id": segment.id,
                "seek": segment.seek,
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "text": segment.text.strip(),
                "no_speech_probability": round(segment.no_speech_prob, 4),
                "compression_ratio": round(segment.compression_ratio, 4),
                "words": formatted_words,
            }
        )

        readable_line = (
            f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text.strip()}"
        )
        text_lines.append(readable_line)
        print(readable_line)

    transcript_data = {
        "schema_version": "1.0",
        "source_audio": str(audio_path),
        "transcription": {
            "model": model_size,
            "device": device,
            "compute_type": compute_type,
            "language": info.language,
            "language_probability": round(info.language_probability, 4),
            "duration_seconds": round(info.duration, 3),
            "duration_after_vad_seconds": round(info.duration_after_vad, 3),
        },
        "segments": formatted_segments,
    }

    json_output_path = output_directory / "transcript.json"
    text_output_path = output_directory / "transcript.txt"
    json_output_path.write_text(
        json.dumps(transcript_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    text_output_path.write_text("\n".join(text_lines), encoding="utf-8")

    print(f"JSON saved to: {json_output_path}")
    print(f"Text saved to: {text_output_path}")
    return json_output_path, text_output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe audio with Faster Whisper")
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default="small")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="int8_float16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transcribe_audio(
        audio_path=args.audio,
        output_directory=args.output_dir,
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
    )


if __name__ == "__main__":
    main()
