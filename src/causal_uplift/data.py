from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split
from sklift.datasets import fetch_criteo


@dataclass(slots=True)
class CriteoDataset:
    features: pd.DataFrame
    outcomes: pd.Series
    treatment: pd.Series


@dataclass(slots=True)
class DatasetSplit:
    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series
    treatment_train: pd.Series
    treatment_validation: pd.Series
    treatment_test: pd.Series


@dataclass(slots=True)
class CriteoSample:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    treatment_train: pd.Series
    treatment_test: pd.Series


def _clean_numeric_features(features: pd.DataFrame) -> pd.DataFrame:
    clean = features.copy()
    numeric_cols = clean.select_dtypes(include=["number", "bool"]).columns
    clean[numeric_cols] = clean[numeric_cols].apply(pd.to_numeric, errors="coerce")
    medians = clean[numeric_cols].median(numeric_only=True)
    clean[numeric_cols] = clean[numeric_cols].fillna(medians)
    return clean


def load_criteo_dataset(sample_size: int = 0) -> CriteoDataset:
    X, y, treatment = fetch_criteo(
        target_col="conversion",
        treatment_col="treatment",
        return_X_y_t=True,
        percent10=True,
    )

    X_df = _clean_numeric_features(pd.DataFrame(X).reset_index(drop=True))
    y_series = pd.Series(y).reset_index(drop=True).astype(int)
    treatment_series = pd.Series(treatment).reset_index(drop=True).astype(int)

    if sample_size > 0:
        max_rows = min(sample_size, len(X_df))
        X_df = X_df.iloc[:max_rows].copy()
        y_series = y_series.iloc[:max_rows].copy()
        treatment_series = treatment_series.iloc[:max_rows].copy()

    return CriteoDataset(features=X_df, outcomes=y_series, treatment=treatment_series)


def create_train_validation_test_split(
    dataset: CriteoDataset,
    validation_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> DatasetSplit:
    if validation_size <= 0 or test_size <= 0 or (validation_size + test_size) >= 1:
        raise ValueError("validation_size and test_size must be positive and sum to < 1")

    strata = dataset.outcomes.astype(str) + "_" + dataset.treatment.astype(str)
    holdout_size = validation_size + test_size

    X_train, X_holdout, y_train, y_holdout, t_train, t_holdout = train_test_split(
        dataset.features,
        dataset.outcomes,
        dataset.treatment,
        test_size=holdout_size,
        random_state=random_state,
        stratify=strata,
    )

    holdout_strata = y_holdout.astype(str) + "_" + t_holdout.astype(str)
    relative_test_size = test_size / holdout_size

    X_validation, X_test, y_validation, y_test, t_validation, t_test = train_test_split(
        X_holdout,
        y_holdout,
        t_holdout,
        test_size=relative_test_size,
        random_state=random_state,
        stratify=holdout_strata,
    )

    return DatasetSplit(
        X_train=X_train.reset_index(drop=True),
        X_validation=X_validation.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_validation=y_validation.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        treatment_train=t_train.reset_index(drop=True),
        treatment_validation=t_validation.reset_index(drop=True),
        treatment_test=t_test.reset_index(drop=True),
    )


def load_criteo_sample(
    sample_size: int = 10_000,
    test_size: float = 0.2,
    random_state: int = 42,
) -> CriteoSample:
    dataset = load_criteo_dataset(sample_size=sample_size)
    X_df = dataset.features
    y_series = dataset.outcomes
    treatment_series = dataset.treatment

    strata = y_series.astype(str) + "_" + treatment_series.astype(str)

    X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
        X_df,
        y_series,
        treatment_series,
        test_size=test_size,
        random_state=random_state,
        stratify=strata,
    )

    return CriteoSample(
        X_train=X_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        treatment_train=t_train.reset_index(drop=True),
        treatment_test=t_test.reset_index(drop=True),
    )
