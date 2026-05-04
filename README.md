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
uv run python scripts/check_production_bundle.py
```

Run the service locally:

```bash
uv run python -m uvicorn causal_uplift.serve:app --host 127.0.0.1 --port 8080
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

Slim serving Docker build and run:

```bash
docker build -t causal-uplift-api:local .
docker run --rm -p 8091:8080 causal-uplift-api:local
curl http://127.0.0.1:8091/health
curl http://127.0.0.1:8091/model-info
```

Prediction check in PowerShell:

```powershell
$body = Get-Content models/production/example_request.json -Raw
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8091/predict_uplift" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

GCP build and deploy:

```powershell
$PROJECT_ID="causal-uplift-transformer"
$REGION="us-central1"
$REPO="causal-uplift"
$SERVICE="causal-uplift-api"

gcloud config set project $PROJECT_ID

gcloud builds submit `
  --tag "$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/causal-uplift-api:latest" .

gcloud run deploy $SERVICE `
  --image "$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/causal-uplift-api:latest" `
  --region $REGION `
  --platform managed `
  --allow-unauthenticated `
  --memory 2Gi `
  --cpu 2 `
  --min-instances 0 `
  --max-instances 2 `
  --port 8080

$URL = gcloud run services describe $SERVICE --region $REGION --format "value(status.url)"
curl "$URL/health"
curl "$URL/model-info"
```

Cloud Run deployment instructions are in `docs/gcp_deployment.md`.

Model binaries in `models/production` are ignored by default unless explicitly added. Regenerate
the bundle locally before building a container.

## Streamlit dashboard

Run the dashboard locally:

```bash
uv run streamlit run app/streamlit_app.py
```

Set API URL in PowerShell (Cloud Run):

```powershell
$env:CAUSAL_UPLIFT_API_URL="https://causal-uplift-api-sn6k6nocwq-uc.a.run.app"
uv run streamlit run app/streamlit_app.py
```

Dashboard capabilities:

- Overview with deployed champion model and key uplift metrics
- Model performance section with ranking artifacts and policy gain charts when available
- Single prediction interface for `f0` through `f11`
- Batch prediction CSV upload with validation and downloadable predictions
- About section documenting dataset, model families, and deployment architecture

Batch CSV input format:

- Required columns: `f0`, `f1`, `f2`, `f3`, `f4`, `f5`, `f6`, `f7`, `f8`, `f9`, `f10`, `f11`
- Additional columns are ignored by the API client payload builder

Screenshot placeholder:

- Add dashboard screenshots to `assets/` when available.

## CI/CD

GitHub Actions workflows in this repository:

- `ci.yml`: quality checks on push/PR to `main` plus manual dispatch
- `docker.yml`: slim image build + container smoke test on push/PR to `main` plus manual dispatch
- `cloud-run-deploy.yml`: manual-only Cloud Run deployment via workflow dispatch

Required GitHub secrets for manual Cloud Run deployment:

- `GCP_PROJECT_ID`
- `GCP_REGION`
- `GCP_ARTIFACT_REPO`
- `GCP_CLOUD_RUN_SERVICE`
- `GCP_WORKLOAD_IDENTITY_PROVIDER`
- `GCP_SERVICE_ACCOUNT`

Why full training is not run in CI:

- Full training, tuning, and reporting are compute-heavy and dataset-heavy workflows.
- CI is scoped to fast, deterministic quality and serving-path validation.
- This keeps pull request feedback quick and does not require live cloud credentials.

Run the same CI checks locally:

```bash
uv sync --all-groups
uv run ruff check src tests scripts app
uv run python -m black --check src tests scripts app
uv run python -m pytest
uv run python scripts/check_production_bundle.py
docker build -t causal-uplift-api:ci .
docker run -d --name causal-uplift-api-ci -p 8092:8080 causal-uplift-api:ci
Start-Sleep -Seconds 8
docker logs causal-uplift-api-ci
Invoke-RestMethod http://127.0.0.1:8092/health
Invoke-RestMethod http://127.0.0.1:8092/model-info
docker stop causal-uplift-api-ci
docker rm causal-uplift-api-ci
```

If port `8092` is occupied, use another host port such as `8093`.
If health checks fail, inspect `docker logs causal-uplift-api-ci` before removing the container.

## Quality checks

```bash
uv run python scripts/verify_environment.py
uv run ruff check src tests scripts app
uv run python -m black --check src tests scripts app
uv run python -m pytest
uv run python scripts/check_production_bundle.py
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
