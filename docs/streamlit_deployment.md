# Streamlit Dashboard Deployment

## Purpose

The Streamlit dashboard is a public-facing client for the deployed Cloud Run FastAPI service.
It visualizes champion metrics and sends single or batch prediction requests to the API.

## Deployment Target

- Streamlit Community Cloud

## Repository Settings

- Repository: sundar139/Causal-Uplift-Transformer-Beam
- Branch: main
- Main file path: app/streamlit_app.py
- Python version: 3.12

## Required App Config

Set the API URL as a Streamlit secret (preferred) or environment value:

```toml
CAUSAL_UPLIFT_API_URL = "https://causal-uplift-api-sn6k6nocwq-uc.a.run.app"
```

## Manual Deployment Instructions

1. Go to Streamlit Community Cloud.
2. Sign in with GitHub.
3. Click **New app**.
4. Select repository: `sundar139/Causal-Uplift-Transformer-Beam`.
5. Select branch: `main`.
6. Set main file path: `app/streamlit_app.py`.
7. Add secret: `CAUSAL_UPLIFT_API_URL=https://causal-uplift-api-sn6k6nocwq-uc.a.run.app`.
8. Deploy.
9. Open the app URL.
10. Confirm the dashboard health panel shows the API as healthy.

## Validation Checklist

- [ ] App builds successfully
- [ ] API health check succeeds
- [ ] Overview page loads
- [ ] Model Performance page loads ranking artifacts if available
- [ ] Single Prediction returns uplift output
- [ ] Batch Prediction accepts CSV with f0 through f11
- [ ] About page explains champion selection

## Troubleshooting

| Issue | Likely Cause | Fix |
|---|---|---|
| App cannot reach API | API URL missing or incorrect | Set `CAUSAL_UPLIFT_API_URL` |
| Module import error | Missing dependency in `requirements.txt` | Add lightweight dependency |
| App installs heavy training stack | Wrong requirements file | Keep `requirements.txt` app-only |
| API health degraded | Cloud Run model bundle issue | Recheck Cloud Run `/health` |
| Batch upload fails | Missing f0 through f11 columns | Use `examples/sample_batch.csv` |
