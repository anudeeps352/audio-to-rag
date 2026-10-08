# Podcast Intelligence RAG

A local pipeline for turning podcast audio into searchable transcript chunks.

The current pipeline supports:

1. Audio transcription with Faster Whisper
2. Adjacent transcript-segment aggregation
3. Dense chunk embeddings with `BAAI/bge-small-en-v1.5`
4. Persistent local vector indexing with Qdrant

## Project layout

```text
src/
  audio_samples/            local audio inputs (not committed)
  extract/extract.py        audio-to-transcript stage
  chunking/aggregation.py   adjacent chunking stage
  embeddings/               dense embedding stage
  indexing/                 Qdrant setup and indexing stage
  output/                   generated artifacts (not committed)
requirements.txt            Python dependencies
main.py                     end-to-end pipeline runner
.github/workflows/ci.yml    GitHub Actions checks
docs/
  podcast-rag-vision.md     longer product and architecture notes
  rag-project-guideline.md  development notes
```

## Setup

Python 3.12 is currently used for development.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`av==18.1.0` is intentionally pinned because the extraction stage had
compatibility problems with PyAV 19 and later.

## Run the pipeline

Place an audio file at `src/audio_samples/sample.mp3`, then run:

```powershell
python .\main.py
```

The default configuration uses CUDA with `int8_float16`. For CPU execution:

```powershell
python .\main.py --device cpu --compute-type int8
```

To resume without rerunning transcription, select a later starting stage:

```powershell
python .\main.py --start-at chunking
python .\main.py --start-at embeddings
python .\main.py --start-at qdrant_setup
python .\main.py --start-at indexing
```

The stages can also be run individually:

```powershell
python .\src\extract\extract.py
python .\src\chunking\aggregation.py
python .\src\embeddings\generate_embeddings.py
python .\src\indexing\qdrant_setup.py
python .\src\indexing\index_embeddings.py
```

Generated transcripts, chunks, and embedding files are written to
`src/output/`. These artifacts and local audio files are excluded from Git.
The local Qdrant database is written to `data/qdrant/` and is also excluded
from Git.

The first embedding run downloads the BGE model from Hugging Face.

## Continuous integration

GitHub Actions runs on every push and pull request. It compiles the Python
sources and checks the pipeline CLI. Model inference is intentionally excluded
because CI has no committed audio fixture and should not download large model
weights.

## Generated embedding files

- `sample_embeddings.npy` contains one normalized 384-dimensional vector per
  transcript chunk.
- `sample_embedding_metadata.json` maps each `vector_index` back to its chunk
  ID, transcript text, timestamps, and source metadata.

## Next stage

The next planned step is to implement query-time semantic retrieval over the
indexed transcript chunks.
