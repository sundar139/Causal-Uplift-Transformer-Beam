from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split
from sklift.datasets import fetch_criteo


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


def load_criteo_sample(
    sample_size: int = 10_000,
    test_size: float = 0.2,
    random_state: int = 42,
) -> CriteoSample:
    X, y, treatment = fetch_criteo(
        target_col="conversion",
        treatment_col="treatment",
        return_X_y_t=True,
        percent10=True,
    )

    X_df = pd.DataFrame(X).reset_index(drop=True)
    y_series = pd.Series(y).reset_index(drop=True).astype(int)
    treatment_series = pd.Series(treatment).reset_index(drop=True).astype(int)

    max_rows = min(sample_size, len(X_df))
    X_df = X_df.iloc[:max_rows].copy()
    y_series = y_series.iloc[:max_rows].copy()
    treatment_series = treatment_series.iloc[:max_rows].copy()

    X_df = _clean_numeric_features(X_df)

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
