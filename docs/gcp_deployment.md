# GCP Cloud Run Deployment

This guide deploys the production FastAPI service to Cloud Run. Replace `PROJECT_ID`,
`REGION`, `REPO`, and `SERVICE` with your values.

## 1. Create A Project

Create or select a GCP project in the Google Cloud Console. Keep the project ID handy.

## 2. Install And Authenticate

Install the Google Cloud CLI, then authenticate:

```bash
gcloud auth login
gcloud config set project PROJECT_ID
```

## 3. Enable Services

```bash
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable cloudbuild.googleapis.com
```

## 4. Create Artifact Registry

```bash
gcloud artifacts repositories create REPO \
  --repository-format=docker \
  --location=REGION \
  --description="Causal uplift API images"
```

## 5. Build The Production Bundle

Build the bundle before building the container. The API serves the champion selected by
`artifacts/reports/full/best_model_summary.json`.

```bash
uv run python scripts/build_inference_bundle.py --config configs/training_full.yaml
```

## 6. Build And Push Image

```bash
gcloud builds submit \
  --tag REGION-docker.pkg.dev/PROJECT_ID/REPO/causal-uplift-api:latest .
```

## 7. Deploy To Cloud Run

```bash
gcloud run deploy SERVICE \
  --image REGION-docker.pkg.dev/PROJECT_ID/REPO/causal-uplift-api:latest \
  --region REGION \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --min-instances 0 \
  --max-instances 2 \
  --concurrency 20 \
  --timeout 300 \
  --port 8080
```

You can also adapt `cloudrun/service.yaml` and deploy it with:

```bash
gcloud run services replace cloudrun/service.yaml --region REGION
```

## 8. Test The Service

After deployment, Cloud Run prints a service URL.

```bash
curl https://SERVICE_URL/health
curl https://SERVICE_URL/model-info
```

## 9. Streamlit Dashboard Client

Use the deployed Cloud Run API as the backend for the local Streamlit client:

```bash
$env:CAUSAL_UPLIFT_API_URL="https://causal-uplift-api-sn6k6nocwq-uc.a.run.app"
uv run streamlit run app/streamlit_app.py
```

If you want to target a local API instead, set:

```bash
$env:CAUSAL_UPLIFT_API_URL="http://127.0.0.1:8080"
```

## Cost Controls

- Use `--min-instances 0` so the service can scale to zero when idle.
- Use `--max-instances 2` to cap runaway scale during testing.
- Choose a nearby `REGION` to reduce latency and avoid unnecessary egress.
- Delete the service when done:

```bash
gcloud run services delete SERVICE --region REGION
```

## Troubleshooting

### Port Already Allocated Locally

If `docker run` fails because a host port is already in use:

```bash
docker ps
docker stop <container_id>
```

Or map a different host port:

```bash
docker run --rm -p 8091:8080 causal-uplift-api:local
```

### Cloud Run Degraded Or `model_loaded=false`

Check that the production bundle files are present in the upload context:

```bash
gcloud meta list-files-for-upload | Select-String "models/production|champion_model|preprocessor"
```

Confirm heavy local artifacts are excluded:

```bash
gcloud meta list-files-for-upload | Select-String "data/|artifacts/data|mlflow.db|optuna_studies.db|parquet"
```

Then inspect Cloud Run logs for startup errors:

```bash
gcloud run services logs read SERVICE --region REGION --limit 100
```

### Slow Docker Build

Production serving images should install only `requirements-serving.txt`.
Do not run `uv sync` in the serving Docker image because it installs the full training stack.
