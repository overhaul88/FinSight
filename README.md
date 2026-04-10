# FinSight

FinSight is a production-oriented Retrieval-Augmented Generation system for Indian financial regulatory documents. The target scope is defined in [FinSight_Tutorial.md](./FinSight_Tutorial.md): ingest RBI/SEBI PDFs, chunk and embed them, retrieve relevant passages with reranking, optionally fine-tune a local instruct model with QLoRA, evaluate with Ragas, and serve answers through FastAPI.

## Scope

- Document ingestion from PDF sources.
- Local FAISS-based retrieval first, with Pinecone as an optional production path.
- Multi-query retrieval plus cross-encoder reranking.
- QLoRA fine-tuning for domain adaptation.
- Ragas evaluation and MLflow tracking.
- FastAPI serving with streaming support.
- Docker and EC2 deployment readiness.

## Repo State

The repository now includes these implemented slices:

- Central settings loading in `src/config.py`.
- Document loading, chunking, embeddings, and local ingestion orchestration in `src/ingestion/` and `scripts/ingest.py`.
- Local retrieval abstractions, query expansion, reranking hooks, and RAG chain assembly in `src/retrieval/`.
- Model loading with a deterministic development fallback in `src/llm/model.py`.
- Fine-tuning dataset generation and a dry-run-safe QLoRA training path in `data/eval/create_finetune_data.py`, `src/llm/finetune.py`, and `scripts/finetune.py`.
- FastAPI schemas and service endpoints in `src/serving/`.
- Dry-run evaluation scaffolding and optional MLflow integration in `src/evaluation/` and `scripts/evaluate.py`.
- Containerization and local ops assets in `Dockerfile`, `docker-compose.yml`, and `.dockerignore`.
- Deployment automation in `.github/workflows/deploy.yml` and `scripts/run_on_ec2.sh`.
- Offline fixtures and test coverage across ingestion, retrieval, chain, API, and evaluation in `tests/`.

Still pending from the tutorial: full GPU-backed QLoRA execution, full Ragas-backed evaluation with judge models, and a real Docker validation pass in an environment where Docker is installed.

## Local Model Recommendation

For this repository on a `GTX 1650 4 GB` class GPU, the most practical local pair is:

- Embedding model: `BAAI/bge-small-en-v1.5`
- LLM: `Qwen/Qwen2.5-1.5B-Instruct` loaded in 4-bit mode

Reason: the embedding model remains strong for retrieval while staying lightweight, and the `1.5B` Qwen instruct model is materially more usable on 4 GB VRAM than the tutorial's `Mistral-7B` target. The repository defaults now reflect that local-device choice, and `scripts/model_probe.py` now reports the detected CUDA device and the recommended local profile before running a real embedding and generation pass.

## Getting Started

1. Create and activate a Python 3.10+ virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a local environment file:

```bash
cp .env.example .env
```

4. Run the environment doctor first:

```bash
python3 scripts/doctor.py
```

4.5. Preload and verify the configured local models:

```bash
HF_HUB_DISABLE_XET=1 python3 scripts/model_probe.py
```

4.6. Run the end-to-end local smoke test with the real embedding model and local LLM:

```bash
HF_HUB_DISABLE_XET=1 python3 scripts/e2e_smoke.py
```

4.7. Download the official RBI and SEBI core corpus directly from the web:

```bash
python3 scripts/scrape_regulatory_docs.py --profile core
```

5. Add PDFs under `data/raw/` or use the text fixtures under `tests/fixtures/`.
6. Run the current validation suite:

```bash
python3 -m compileall src scripts tests
.venv/bin/pytest tests/test_ingestion.py -q
.venv/bin/pytest tests/test_retriever.py tests/test_chain.py tests/test_api.py tests/test_evaluation.py -q
.venv/bin/pytest tests/test_finetune.py -q
.venv/bin/pytest tests/test_doctor.py -q
```

7. Generate the fine-tuning dataset and validate the dry-run trainer contract:

```bash
python3 data/eval/create_finetune_data.py
python3 scripts/finetune.py --dry-run
```

8. Use the `Makefile` shortcuts if you prefer:

```bash
make doctor
make compile
make test-fast
make test
make finetune-data
make finetune-dry
make scrape
```

## First Run Flow

Use this order for a clean local setup:

1. `python3 scripts/doctor.py`
2. `python3 data/eval/create_finetune_data.py`
3. `python3 scripts/finetune.py --dry-run`
4. `.venv/bin/pytest -q`
5. `PYTHONPATH=. .venv/bin/python -m uvicorn src.serving.api:app --reload`

This sequence validates the current repository state without requiring Docker, a GPU, or external judge models.

## Current Acceptance Baseline

- `python3 -m compileall src scripts data/eval tests`
- `.venv/bin/pytest -q`
- `python3 data/eval/create_finetune_data.py`
- `python3 scripts/finetune.py --dry-run`
- `HF_HUB_DISABLE_XET=1 python3 scripts/e2e_smoke.py`
- `bash -n scripts/run_on_ec2.sh`

The repository is considered healthy when those checks pass locally.

## Implementation Order

1. Replace dry-run evaluation with full Ragas-backed scoring when the heavier dependencies are installed.
2. Run a real fine-tuning job on a GPU-backed environment.
3. Validate `docker compose config` and container startup on a machine with Docker installed.
4. Add any missing production hardening after the first full containerized run.

## Notes

- Keep all external services optional behind environment variables.
- Prefer FAISS for local development and testing.
- `ENABLE_LLM_QUERY_EXPANSION=false` is the local default so the 4 GB GPU is spent on answer generation rather than a second LLM pass for retrieval expansion.
- Preserve citation grounding in all answers.
- Treat Ragas metrics as acceptance gates for later iterations.
- The current local test baseline is `25 passed`.
- `docker compose config` could not be validated in this workspace because `docker` is not installed.
