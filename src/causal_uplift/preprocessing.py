from __future__ import annotations

import numpy as np
import pandas as pd


class NumericFeaturePreprocessor:
    def __init__(self) -> None:
        self.feature_columns: list[str] = []
        self.medians: pd.Series | None = None

    def fit(self, frame: pd.DataFrame) -> NumericFeaturePreprocessor:
        numeric = frame.copy()
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        self.feature_columns = list(numeric.columns)
        numeric = numeric.apply(pd.to_numeric, errors="coerce")
        self.medians = numeric.median(numeric_only=True)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.medians is None:
            raise RuntimeError("NumericFeaturePreprocessor must be fitted before transform")

        aligned = frame.reindex(columns=self.feature_columns, fill_value=np.nan).copy()
        aligned = aligned.replace([np.inf, -np.inf], np.nan)
        aligned = aligned.apply(pd.to_numeric, errors="coerce")
        aligned = aligned.fillna(self.medians)
        return aligned.to_numpy(dtype=np.float32)

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)
