PYTHON ?= python
VENV ?= .venv

.PHONY: install test train api

install:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt

test:
	pytest -q

train:
	$(PYTHON) scripts/train_local.py --config configs/default.yaml

api:
	$(PYTHON) scripts/run_api.py --host 0.0.0.0 --port 8000
