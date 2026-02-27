# wti-regime-monitor

Production-oriented probabilistic regime monitoring for **WTI Cushing daily spot prices**.

This project trains a 3-state Gaussian Hidden Markov Model (HMM) on daily **log returns** of WTI spot prices and serves model outputs through a FastAPI service.

## What regimes mean

Each hidden regime represents a distinct return/volatility profile:

- `Regime 0`: lower-vol / near-flat return behavior
- `Regime 1`: medium-vol behavior
- `Regime 2`: higher-vol / stressed behavior

Regime labels are learned statistically and may swap ordering between runs.

## Repository structure

```text
wti-regime-monitor/
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── routes.py
├── configs/
│   └── default.yaml
├── data/
│   └── raw/
├── energy_data/
│   ├── __init__.py
│   ├── eia_client.py
│   └── features.py
├── models/
│   ├── __init__.py
│   ├── hmm_tfp.py
│   ├── infer.py
│   └── train.py
├── runs/
│   └── .gitkeep
├── scripts/
│   ├── run_api.py
│   ├── make_plots.py
│   ├── print_latest.py
│   └── train_local.py
├── tests/
│   ├── test_data_fetcher.py
│   ├── test_features.py
│   └── test_hmm_shapes.py
├── .gitignore
├── Makefile
├── README.md
└── requirements.txt
```

## Model

Observation model on returns:

- Hidden state: `z_t ∈ {0,1,2}`
- Transition: `P(z_t | z_{t-1}) = A` (trainable row-stochastic matrix)
- Emission: `r_t | z_t = k ~ Normal(mu_k, sigma_k)`

Training objective:

- Maximize `log_prob(r_1:T)` using Adam
- Early stopping on validation log-likelihood
- Time-based split (no shuffling)

## Data

Source: EIA Open Data API v2 (`petroleum/pri/spt`, product `EPCWTI`).

Caching behavior:

- Raw prices cached to `data/raw/wti_cushing_daily.csv`
- After first successful fetch, project runs offline from cache unless `--force-refresh` is used

You can set `EIA_API_KEY`; if omitted, the client uses `DEMO_KEY`.

## Quickstart

```bash
cd wti-regime-monitor
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Train one run locally:

```bash
python scripts/train_local.py --config configs/default.yaml
```

Start API:

```bash
python scripts/run_api.py --host 0.0.0.0 --port 8000
```

## Generate plots

Create an interactive Plotly artifact for a run:

```bash
python -m scripts.make_plots --run-id <RUN_ID>
```

By default, the HTML is saved to `runs/<run_id>/regimes.html`.

## API endpoints

- `POST /fit` -> train a model run, return `run_id`
- `GET /health` -> health check
- `GET /runs` -> list run ids (newest first)
- `GET /runs/latest/summary` -> latest run summary
- `GET /predict_proba` -> dates, returns, per-regime probabilities
- `GET /transition_matrix` -> learned transition matrix
- `GET /regime_summary` -> regime stats, durations, transition counts
- `GET /forecast` -> multi-step predictive mean + uncertainty interval
- `POST /predict_current` -> latest/current regime label and state

## How to use the API

1. Start the server:

```bash
python scripts/run_api.py --host 0.0.0.0 --port 8000
```

2. Check health:

```bash
curl "http://127.0.0.1:8000/health"
```

3. List runs and inspect the latest summary:

```bash
curl "http://127.0.0.1:8000/runs"
curl "http://127.0.0.1:8000/runs/latest/summary"
```

4. Get current regime:

```bash
curl -X POST "http://127.0.0.1:8000/predict_current" -H 'Content-Type: application/json' -d '{}'
```

## Demo workflow

```bash
make test
make train
make plot
make ui
```

Open [http://localhost:8000/ui](http://localhost:8000/ui).

### Example `curl` calls

Fit:

```bash
curl -X POST http://127.0.0.1:8000/fit \
  -H 'Content-Type: application/json' \
  -d '{"config_path":"configs/default.yaml","force_refresh":false}'
```

Posterior probabilities:

```bash
curl "http://127.0.0.1:8000/predict_proba"
```

Transition matrix:

```bash
curl "http://127.0.0.1:8000/transition_matrix"
```

Regime summary:

```bash
curl "http://127.0.0.1:8000/regime_summary"
```

Forecast (20-step, 95% band):

```bash
curl "http://127.0.0.1:8000/forecast?horizon=20&interval=0.95"
```

## Saved artifacts per run

Each run is persisted under `runs/<run_id>/`:

- `config.json`
- `model_params.npz`
- `metrics.json`
- `transition_matrix.json`
- `regime_summary.json`
- `predict_proba.json`
- `forecast_default.json`

`runs/latest_run.txt` tracks the most recent run.

## Sample output JSON (`/forecast`)

```json
{
  "run_id": "run_20260224T231122Z_a1b2c3d4",
  "interval": 0.95,
  "horizon": 3,
  "forecast_dates": ["2026-02-25", "2026-02-26", "2026-02-27"],
  "forecast": [
    {
      "horizon": 1,
      "mean": 0.0004,
      "std": 0.0139,
      "lower": -0.0268,
      "upper": 0.0276,
      "state_probabilities": [0.61, 0.27, 0.12]
    }
  ]
}
```

## Running tests

```bash
pytest -q
```

Notes:

- `test_hmm_shapes.py` auto-skips if TensorFlow / TensorFlow Probability are not installed.
- Other tests validate cache fallback and feature engineering correctness.
