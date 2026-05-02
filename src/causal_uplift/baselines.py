from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(slots=True)
class UpliftPrediction:
    treatment_proba: np.ndarray
    control_proba: np.ndarray
    uplift: np.ndarray


class TwoModelUpliftBaseline:
    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.treatment_model: Pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(max_iter=500, random_state=random_state),
                ),
            ]
        )
        self.control_model: Pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(max_iter=500, random_state=random_state),
                ),
            ]
        )
        self.feature_columns: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series, treatment: pd.Series) -> TwoModelUpliftBaseline:
        self.feature_columns = list(X.columns)

        treated_mask = treatment == 1
        control_mask = treatment == 0

        self.treatment_model.fit(X.loc[treated_mask, self.feature_columns], y.loc[treated_mask])
        self.control_model.fit(X.loc[control_mask, self.feature_columns], y.loc[control_mask])

        return self

    def predict_uplift(self, X: pd.DataFrame) -> UpliftPrediction:
        aligned = X.reindex(columns=self.feature_columns, fill_value=0.0)
        treatment_proba = self.treatment_model.predict_proba(aligned)[:, 1]
        control_proba = self.control_model.predict_proba(aligned)[:, 1]
        uplift = treatment_proba - control_proba
        return UpliftPrediction(
            treatment_proba=treatment_proba,
            control_proba=control_proba,
            uplift=uplift,
        )
