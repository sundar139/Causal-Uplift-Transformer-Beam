from __future__ import annotations

import numpy as np
import pandas as pd

from causal_uplift.data import (
    CriteoDataset,
    build_processed_split_frames,
    create_train_validation_test_split,
)


def test_materialized_split_frames_include_required_columns_and_labels() -> None:
    rows = 120
    rng = np.random.default_rng(11)
    features = pd.DataFrame(
        {
            "f0": rng.normal(size=rows),
            "f1": rng.normal(size=rows),
            "f2": rng.normal(size=rows),
        }
    )
    outcomes = pd.Series(np.tile([0, 1, 0, 1], rows // 4), dtype=int)
    treatment = pd.Series(np.tile([0, 0, 1, 1], rows // 4), dtype=int)

    split = create_train_validation_test_split(
        CriteoDataset(features=features, outcomes=outcomes, treatment=treatment),
        validation_size=0.2,
        test_size=0.2,
        random_state=42,
    )
    split_frames = build_processed_split_frames(split)

    assert set(split_frames.keys()) == {"train", "validation", "test"}
    for split_name, frame in split_frames.items():
        assert "conversion" in frame.columns
        assert "treatment" in frame.columns
        assert "split" in frame.columns
        assert set(frame["split"].unique()) == {split_name}

    total = sum(len(frame) for frame in split_frames.values())
    assert total == rows
