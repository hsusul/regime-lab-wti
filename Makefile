PYTHON ?= python
VENV ?= .venv

.PHONY: install test train plot api ui

install:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt

test:
	pytest -q

train:
	$(PYTHON) -m scripts.train_local --config configs/default.yaml

plot:
	$(PYTHON) -m scripts.make_plots

api:
	uvicorn app.main:app --host 0.0.0.0 --port 8000

ui:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
