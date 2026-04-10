PYTHON ?= python3
VENV_PYTHON ?= .venv/bin/python
VENV_PYTEST ?= .venv/bin/pytest

.PHONY: doctor test test-fast compile finetune-data finetune-dry api smoke scrape

doctor:
	$(PYTHON) scripts/doctor.py

compile:
	$(PYTHON) -m compileall src scripts data/eval tests

test-fast:
	$(VENV_PYTEST) tests/test_ingestion.py tests/test_retriever.py tests/test_chain.py -q

test:
	$(VENV_PYTEST) -q

finetune-data:
	$(PYTHON) data/eval/create_finetune_data.py

finetune-dry:
	$(PYTHON) scripts/finetune.py --dry-run

api:
	PYTHONPATH=. $(VENV_PYTHON) -m uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload

smoke:
	PYTHONPATH=. $(VENV_PYTHON) scripts/e2e_smoke.py

scrape:
	PYTHONPATH=. $(VENV_PYTHON) scripts/scrape_regulatory_docs.py --profile core
