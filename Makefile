PYTHON ?= python
VENV ?= .venv
RUN ?= run_YYYYMMDDTHHMMSSZ_example

.PHONY: install test train plot api ui integrity bundle

install:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt

test:
	pytest -q

train:
	$(PYTHON) -m scripts.train_local --config configs/default.yaml --force-refresh

plot:
	$(PYTHON) -m scripts.make_plots

api:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

ui:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

integrity:
	curl -sS "http://127.0.0.1:8000/runs/$(RUN)/integrity"

bundle:
	curl -sS -o /tmp/$(RUN)_bundle.zip "http://127.0.0.1:8000/runs/$(RUN)/bundle.zip"
