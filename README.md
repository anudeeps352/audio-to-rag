# Podcast Intelligence RAG

A local pipeline for turning podcast audio into searchable, answerable transcript
knowledge.

The current pipeline supports:

1. Audio transcription with Faster Whisper
2. Adjacent transcript-segment aggregation
3. Dense chunk embeddings with `BAAI/bge-small-en-v1.5`
4. Persistent local vector indexing with Qdrant
5. Semantic search over timestamped transcript chunks
6. Local answer generation with Ollama and `qwen3:4b-instruct`
7. A labeled retrieval benchmark with repeatable dense-search evaluation

## Project layout

```text
src/
  audio_samples/            local audio inputs (not committed)
  extract/extract.py        audio-to-transcript stage
  chunking/aggregation.py   adjacent chunking stage
  embeddings/               dense embedding stage
  indexing/                 Qdrant setup and indexing stage
  retrieval/search.py       semantic transcript search
  generation/               local RAG answer generation
  evaluation/               evaluation code
  output/                   generated artifacts (not committed)
benchmarks/
  benchmark.json            labeled retrieval questions
  results/                  saved experiment reports
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

## Ask questions

After running the ingestion pipeline, ask a question with:

```powershell
python -m src.generation.generate_answer "Who are the podcast hosts?" --limit 3
```

This embeds the question, searches Qdrant, builds context from the retrieved
chunks, and asks the local Ollama model to answer from that context.

To inspect retrieval results without generation:

```powershell
python -m src.retrieval.search "What's the plot of Man Quest?" --limit 3
```

## Evaluate retrieval

Run the labeled dense-retrieval benchmark with:

```powershell
python -m src.evaluation.evaluate_retrieval
```

The benchmark lives in `benchmarks/benchmark.json`, and the report is saved to
`benchmarks/results/dense_baseline.json`.

Current dense baseline over 10 answerable questions and one unanswerable
question:

| Metric | Score |
|---|---:|
| Recall@1 | 0.60 |
| Recall@5 | 0.95 |
| MRR | 0.753 |

This small single-episode benchmark is a learning and regression baseline, not
a production-quality accuracy claim.

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

## Current stage and next step

The dense RAG baseline is complete through generation and retrieval evaluation.
The next planned experiment is BM25 lexical retrieval. It will be evaluated
against the same benchmark before adding hybrid retrieval with Reciprocal Rank
Fusion (RRF), followed by reranking and generation evaluation.
