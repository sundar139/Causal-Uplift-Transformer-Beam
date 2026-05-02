from __future__ import annotations

import numpy as np
import pandas as pd

from causal_uplift.data import CriteoDataset, create_train_validation_test_split


def test_train_validation_test_split_contract() -> None:
    rows = 120
    rng = np.random.default_rng(7)
    features = pd.DataFrame(
        {
            "f0": rng.normal(size=rows),
            "f1": rng.normal(size=rows),
            "f2": rng.normal(size=rows),
        }
    )

    outcomes = pd.Series(np.tile([0, 1, 0, 1], rows // 4), dtype=int)
    treatment = pd.Series(np.tile([0, 0, 1, 1], rows // 4), dtype=int)

    dataset = CriteoDataset(features=features, outcomes=outcomes, treatment=treatment)
    split = create_train_validation_test_split(
        dataset,
        validation_size=0.2,
        test_size=0.2,
        random_state=42,
    )

    total = len(split.X_train) + len(split.X_validation) + len(split.X_test)
    assert total == rows

    assert list(split.X_train.columns) == ["f0", "f1", "f2"]
    assert split.X_train.index.min() == 0
    assert split.X_validation.index.min() == 0
    assert split.X_test.index.min() == 0

    for y_part, t_part in [
        (split.y_train, split.treatment_train),
        (split.y_validation, split.treatment_validation),
        (split.y_test, split.treatment_test),
    ]:
        combos = set((y_part.astype(str) + "_" + t_part.astype(str)).unique())
        assert combos == {"0_0", "1_0", "0_1", "1_1"}
