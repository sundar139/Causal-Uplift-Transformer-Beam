## Featured Project: Causal Uplift Transformer Beam

Production-grade causal uplift modeling system for estimating incremental treatment effect, not just conversion probability. The project benchmarks classical uplift baselines against FT-Transformer and causal FT-Transformer challengers, tracks experiments with MLflow, tunes models with Optuna, selects the production champion by full-test Qini AUC, and deploys the inference API on Google Cloud Run with a Streamlit dashboard.

**Highlights**

- Full Criteo Uplift dataset workflow with 13,979,592 rows
- Data lineage, profiling, schema, and manifest tracking
- Baseline uplift models, FT-Transformer challenger, and causal FT-Transformer challenger
- MLflow tracking and Optuna tuning
- Champion/challenger governance using Qini AUC and Policy Gain@20
- FastAPI inference service deployed on Google Cloud Run
- Streamlit dashboard for model performance, single prediction, and batch prediction
- Dockerized serving and GitHub Actions CI/CD

**Production Champion**

`S-learner logistic` was selected as the deployed champion by full-test Qini AUC. Transformer models remain documented challengers, which keeps the system honest instead of forcing a neural model into production just because it sounds expensive.

**Links**

- Repository: https://github.com/sundar139/Causal-Uplift-Transformer-Beam
- Streamlit Dashboard: STREAMLIT_APP_URL_TO_BE_ADDED_AFTER_DEPLOYMENT
- Cloud Run API: https://causal-uplift-api-sn6k6nocwq-uc.a.run.app
