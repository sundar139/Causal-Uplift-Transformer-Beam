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

## Data

Dataset used:

- Criteo Uplift Prediction via `sklift.datasets.fetch_criteo(percent10=True)`

Why `data/raw` and `data/processed` may be empty initially:

- The dataset is downloaded/materialized locally on demand.
- Raw and processed parquet files are git-ignored to keep the repository lightweight.

Materialize local parquet data:

```bash
uv run python -m causal_uplift.data materialize --config configs/training.yaml
```

Generate data lineage/profile artifacts:

```bash
uv run python -m causal_uplift.data profile --config configs/training.yaml
```

Ignored local data files:

- data/raw/*.parquet
- data/processed/*.parquet

Tracked lightweight data artifacts:

- artifacts/data/criteo_data_profile.json
- artifacts/data/criteo_schema.json
- artifacts/data/criteo_sample_preview.csv
- artifacts/data/data_manifest.json
- docs/data_card.md

## Reporting artifacts

Generate consolidated report and plot artifacts from the latest full run:

```bash
uv run python -m causal_uplift.train report --config configs/training.yaml
```

Expected reporting outputs:

- artifacts/reports/model_ranking.csv
- artifacts/reports/model_ranking.json
- artifacts/reports/best_model_summary.json
- artifacts/reports/experiment_manifest.json
- artifacts/plots/qini_curve.png
- artifacts/plots/uplift_curve.png
- artifacts/plots/policy_gain_curve.png

Current model ranking (example):

| model_name | qini_auc | policy_gain_top20 | uplift_auc | treatment_response_auc |
| --- | ---: | ---: | ---: | ---: |
| ft_transformer | 0.2140 | 0.0380 | 0.1820 | 0.6260 |
| two_model_logistic | 0.1910 | 0.0320 | 0.1710 | 0.6150 |
| t_learner_logistic | 0.1780 | 0.0290 | 0.1640 | 0.6090 |
| s_learner_logistic | 0.1650 | 0.0240 | 0.1580 | 0.6030 |

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
uv run python -m causal_uplift.data materialize --config configs/training.yaml
uv run python -m causal_uplift.data profile --config configs/training.yaml
uv run python -m causal_uplift.train smoke --sample-size 10000
uv run python -m causal_uplift.train full --config configs/training.yaml
uv run python -m causal_uplift.train report --config configs/training.yaml
```
