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
- artifacts/evaluation/percent10/full_training_metrics.json
- artifacts/evaluation/percent10/test_predictions.csv

## Optuna tuning

Run Optuna after the percent10 and full dataset workflows are validated. That keeps tuning focused on improving known-good model paths instead of debugging data loading, splitting, or reporting while trials are running.

Tune percent10:

```bash
uv run python -m causal_uplift.tuning --config configs/tuning.yaml
uv run python -m causal_uplift.train full --config configs/training.yaml --use-best-params
uv run python -m causal_uplift.train report --config configs/training.yaml
```

Tune full:

```bash
uv run python -m causal_uplift.tuning --config configs/tuning_full.yaml
uv run python -m causal_uplift.train full --config configs/training_full.yaml --use-best-params
uv run python -m causal_uplift.train report --config configs/training_full.yaml
```

Tuning artifacts:

- artifacts/tuning/percent10/optuna_trials.csv
- artifacts/tuning/percent10/best_params.json
- artifacts/tuning/percent10/tuning_summary.json
- artifacts/tuning/full/optuna_trials.csv
- artifacts/tuning/full/best_params.json
- artifacts/tuning/full/tuning_summary.json

Optuna uses local SQLite storage by default at `optuna_studies.db`; MLflow remains SQLite-backed at `mlflow.db`.

## Causal FT-Transformer Challenger

The original FT-Transformer is an S-learner style model: treatment is appended as an input feature
and the model learns one factual outcome surface. The causal FT-Transformer challenger uses a shared
feature encoder with separate control and treatment heads, so it directly estimates `mu0(x)` and
`mu1(x)`. It also supports a propensity head, group-balanced factual loss, positive-class weighting,
Qini-based checkpointing, and multi-seed ensembling.

Validate on percent10:

```bash
uv run python -m causal_uplift.train causal-ft --config configs/training_causal_ft.yaml
uv run python -m causal_uplift.train report --config configs/training.yaml
```

Train the full challenger:

```bash
uv run python -m causal_uplift.train causal-ft --config configs/training_causal_ft_full.yaml
uv run python -m causal_uplift.train report --config configs/training_full.yaml
```

Champion selection remains honest: models are ranked by `qini_auc`, with `policy_gain_top20` as the
tie-breaker. If logistic still wins, deployment should keep the logistic champion and treat causal FT
as a challenger. Deployment resumes only after the champion/challenger report supports the selected
production model.

Challenger outputs:

- artifacts/evaluation/percent10/causal_ft_metrics.json
- artifacts/evaluation/full/causal_ft_metrics.json
- artifacts/evaluation/full/causal_ft_predictions.csv
- artifacts/reports/full/champion_challenger_summary.json

## Dataset modes

This project supports two dataset modes with separate local parquet, profile, and evaluation paths:

- `percent10` mode (`configs/training.yaml`) for fast iteration
- `full` mode (`configs/training_full.yaml`) for final model training

Dataset source for both modes:

- Criteo Uplift Prediction via `sklift.datasets.fetch_criteo(...)`

Materialize and profile `percent10` mode:

```bash
uv run python -m causal_uplift.data materialize --config configs/training.yaml
uv run python -m causal_uplift.data profile --config configs/training.yaml
```

Materialize and profile `full` mode:

```bash
uv run python -m causal_uplift.data materialize --config configs/training_full.yaml
uv run python -m causal_uplift.data profile --config configs/training_full.yaml
```

Train and report `percent10` mode:

```bash
uv run python -m causal_uplift.train full --config configs/training.yaml
uv run python -m causal_uplift.train report --config configs/training.yaml
```

Train and report `full` mode:

```bash
uv run python -m causal_uplift.train full --config configs/training_full.yaml
uv run python -m causal_uplift.train report --config configs/training_full.yaml
```

Why `data/raw` and `data/processed` are empty in Git:

- The dataset is downloaded/materialized locally on demand.
- Parquet files are git-ignored and must be regenerated locally.

Ignored local data files:

- data/raw/*.parquet
- data/processed/**/*.parquet

Tracked lightweight lineage artifacts:

- artifacts/data/percent10/*.json
- artifacts/data/percent10/*.csv
- artifacts/data/full/*.json
- artifacts/data/full/*.csv

Row-count check:

- Inspect `row_counts` in:
  - `artifacts/data/percent10/data_manifest.json`
  - `artifacts/data/full/data_manifest.json`
- Full mode should have a larger total row count than percent10 mode.

## Data card

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
uv run python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open <http://127.0.0.1:5000>.

## API

Build the production inference bundle first. The bundle uses the champion selected in
`artifacts/reports/full/best_model_summary.json`; it does not assume the transformer is the
production model.

```bash
uv run python scripts/build_inference_bundle.py --config configs/training_full.yaml
```

Run the service locally:

```bash
uv run uvicorn causal_uplift.serve:app --host 127.0.0.1 --port 8080
```

Endpoints:

- GET /
- GET /health
- GET /version
- GET /model-info
- POST /predict_uplift
- POST /predict_batch

Example request:

```json
{
  "features": {
    "f0": 0.1,
    "f1": 0.2
  }
}
```

Health check:

```bash
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/model-info
```

Docker build and run:

```bash
docker build -t causal-uplift-api:local .
docker run --rm -p 8080:8080 causal-uplift-api:local
```

Cloud Run deployment instructions are in `docs/gcp_deployment.md`.

Model binaries in `models/production` are ignored by default unless explicitly added. Regenerate
the bundle locally before building a container.

## Quality checks

```bash
uv run python scripts/verify_environment.py
uv run ruff check src tests
uv run black --check src tests
uv run pytest
uv run python -m causal_uplift.data materialize --config configs/training.yaml
uv run python -m causal_uplift.data profile --config configs/training.yaml
uv run python -m causal_uplift.data materialize --config configs/training_full.yaml
uv run python -m causal_uplift.data profile --config configs/training_full.yaml
uv run python -m causal_uplift.train smoke --sample-size 10000
uv run python -m causal_uplift.train full --config configs/training.yaml
uv run python -m causal_uplift.tuning --config configs/tuning.yaml
uv run python -m causal_uplift.train full --config configs/training.yaml --use-best-params
uv run python -m causal_uplift.train report --config configs/training.yaml
uv run python -m causal_uplift.train full --config configs/training_full.yaml
uv run python -m causal_uplift.tuning --config configs/tuning_full.yaml
uv run python -m causal_uplift.train full --config configs/training_full.yaml --use-best-params
uv run python -m causal_uplift.train report --config configs/training_full.yaml
uv run python scripts/build_inference_bundle.py --config configs/training_full.yaml
```
