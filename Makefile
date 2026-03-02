PYTHON ?= python
VENV ?= .venv
RUN ?= run_YYYYMMDDTHHMMSSZ_example
A ?= run_A
B ?= run_B
TRASH ?= run_YYYYMMDDTHHMMSSZ_example_YYYYMMDDTHHMMSSZ
NOTES ?= "run notes"
EXTRAS ?=

.PHONY: install test train plot api ui integrity bundle version drift report delete restore trash_list trash_get trash_delete trash_purge notes_get notes_put alerts_evaluate compare_get forecast_v3

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

version:
	curl -sS "http://127.0.0.1:8000/version"

drift:
	curl -sS -X POST "http://127.0.0.1:8000/runs/drift" -H "Content-Type: application/json" -d '{"run_a":"$(A)","run_b":"$(B)"}'

report:
	curl -sS "http://127.0.0.1:8000/runs/$(RUN)/report.md"

delete:
	curl -sS -X DELETE "http://127.0.0.1:8000/runs/$(RUN)"

restore:
	curl -sS -X POST "http://127.0.0.1:8000/runs/trash/$(TRASH)/restore"

trash_list:
	curl -sS "http://127.0.0.1:8000/runs/trash"

trash_get:
	curl -sS "http://127.0.0.1:8000/runs/trash/$(TRASH)"

trash_delete:
	curl -sS -X DELETE "http://127.0.0.1:8000/runs/trash/$(TRASH)"

trash_purge:
	curl -sS -X DELETE "http://127.0.0.1:8000/runs/trash/$(TRASH)"

notes_get:
	curl -sS "http://127.0.0.1:8000/runs/$(RUN)/notes"

notes_put:
	curl -sS -X PUT "http://127.0.0.1:8000/runs/$(RUN)/notes" -H "Content-Type: application/json" -d '{"content":$(NOTES)}'

alerts_evaluate:
	curl -sS -X POST "http://127.0.0.1:8000/alerts/evaluate" -H "Content-Type: application/json" -d '{"run_id":"$(RUN)"}'

compare_get:
	curl -sS "http://127.0.0.1:8000/runs/$(A)/compare/$(B)"

forecast_v3:
	curl -sS "http://127.0.0.1:8000/forecast_v3?run_id=$(RUN)&horizon=10&interval=0.95"
