# Data Card: Criteo Uplift Prediction

## Dataset name

Criteo Uplift Prediction dataset, accessed through scikit-uplift in two modes:

- percent10 mode for fast iteration
- full mode for final training

## Source

- Provider: Criteo
- Access path: `sklift.datasets.fetch_criteo(percent10=True|False)`
- Loader in this project: `causal_uplift.data.load_criteo_dataset`

## What the dataset represents

This dataset is used for uplift modeling: estimating the incremental effect of treatment on conversion probability. Each row represents an impression/user context with anonymized numeric features and observed conversion outcome under treatment or control.

## Variables

- Target variable: `conversion`
- Treatment variable: `treatment`
- Features: anonymized numeric features (schema is documented per mode)

## Split policy

The project uses a deterministic stratified split by `(conversion, treatment)`:

- Train: `1 - validation_size - test_size`
- Validation: `validation_size`
- Test: `test_size`

Default values come from each config file (`configs/training.yaml` and `configs/training_full.yaml`) and randomness is controlled by `random_state` in config.

## Why full raw data is not committed

Raw and processed parquet files can be large and environment-dependent. To keep repository size and diffs manageable, the repository tracks only lightweight lineage/profile artifacts under `artifacts/data/`, while parquet files remain local and git-ignored.

## Reproducibility

Materialize local data files:

```bash
uv run python -m causal_uplift.data materialize --config configs/training.yaml
uv run python -m causal_uplift.data materialize --config configs/training_full.yaml
```

Generate profile and lineage artifacts:

```bash
uv run python -m causal_uplift.data profile --config configs/training.yaml
uv run python -m causal_uplift.data profile --config configs/training_full.yaml
```

Expected local parquet outputs:

- `data/raw/criteo_percent10.parquet`
- `data/processed/percent10/train.parquet`
- `data/processed/percent10/validation.parquet`
- `data/processed/percent10/test.parquet`
- `data/raw/criteo_full.parquet`
- `data/processed/full/train.parquet`
- `data/processed/full/validation.parquet`
- `data/processed/full/test.parquet`

Expected lightweight tracked artifacts:

- `artifacts/data/percent10/criteo_data_profile.json`
- `artifacts/data/percent10/criteo_schema.json`
- `artifacts/data/percent10/criteo_sample_preview.csv`
- `artifacts/data/percent10/data_manifest.json`
- `artifacts/data/full/criteo_data_profile.json`
- `artifacts/data/full/criteo_schema.json`
- `artifacts/data/full/criteo_sample_preview.csv`
- `artifacts/data/full/data_manifest.json`

## Limitations and ethical notes

- Features are anonymized, which limits direct interpretability.
- Uplift estimates can amplify historical bias if treatment assignment or outcomes reflect inequities.
- Model outputs should support decision-making, not replace human oversight in high-impact settings.
- External validity may degrade across domains, campaigns, and time windows; monitor drift and recalibrate.
