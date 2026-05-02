# causal-uplift-transformer-beam

A production-ready repository foundation for deterministic causal uplift smoke training and serving.

## What is included

- Criteo uplift smoke data loading
- Two-model logistic-regression uplift baseline
- MLflow experiment tracking and artifact logging
- FastAPI smoke service with deterministic fallback behavior
- CI, linting, formatting, and tests

## Setup

```bash
uv sync --all-groups
```

Optional environment file:

```bash
cp .env.example .env
```

## Train baseline smoke model

```bash
uv run python -m causal_uplift.train --sample-size 10000
```

or

```bash
uv run uplift-train --sample-size 10000
```

Expected outputs:

- `models/smoke_uplift_baseline.joblib`
- `artifacts/smoke_metrics.json`
- `mlruns/`

## MLflow UI

```bash
uv run mlflow ui
```

Open `http://127.0.0.1:5000`.

## API

Run the service:

```bash
uv run uvicorn causal_uplift.serve:app --reload --port 8000
```

Endpoints:

- `GET /health`
- `GET /version`
- `POST /predict_uplift`

Example request:

```json
{
  "rows": [
    {"features": [0.1, 0.2, 0.3, 0.4]}
  ]
}
```

## Quality checks

```bash
uv run ruff check src tests
uv run black --check src tests
uv run pytest
```
