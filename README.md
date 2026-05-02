# causal-uplift-transformer-beam

A production-grade causal uplift repository with smoke and full training workflows, MLflow tracking, and FastAPI serving.

## What is included

- Deterministic Criteo uplift loading and stratified train/validation/test splits
- Baselines: two-model logistic, S-learner logistic, T-learner logistic
- FT-Transformer style uplift model with PyTorch and early stopping
- Unified training CLI for smoke and full runs
- SQLite-backed MLflow tracking with per-model runs and exported artifacts
- FastAPI inference API with deterministic fallback responses
- Ruff, Black, and Pytest validation

## Setup

```bash
uv sync --all-groups
```

Optional environment file:

```bash
copy .env.example .env
```

## Verify environment

```bash
uv run python scripts/verify_environment.py
```

## Smoke training

```bash
uv run python -m causal_uplift.train smoke --sample-size 10000
```

Smoke outputs:

- models/smoke_uplift_baseline.joblib
- artifacts/smoke_metrics.json

## Full training

```bash
uv run python -m causal_uplift.train full --config configs/training.yaml
```

Full training models:

- two_model_logistic
- s_learner_logistic
- t_learner_logistic
- ft_transformer

Primary outputs:

- mlflow.db
- models/best_transformer_uplift.pt
- artifacts/evaluation/full_training_metrics.json
- artifacts/evaluation/test_predictions.csv

## MLflow UI

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open <http://127.0.0.1:5000>.

## API

Run the service:

```bash
uv run uvicorn causal_uplift.serve:app --reload --port 8000
```

Endpoints:

- GET /health
- GET /version
- POST /predict_uplift

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
uv run python scripts/verify_environment.py
uv run ruff check src tests
uv run black --check src tests
uv run pytest
uv run python -m causal_uplift.train smoke --sample-size 10000
uv run python -m causal_uplift.train full --config configs/training.yaml
```
