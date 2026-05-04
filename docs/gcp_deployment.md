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

## Cost Controls

- Use `--min-instances 0` so the service can scale to zero when idle.
- Use `--max-instances 2` to cap runaway scale during testing.
- Choose a nearby `REGION` to reduce latency and avoid unnecessary egress.
- Delete the service when done:

```bash
gcloud run services delete SERVICE --region REGION
```
