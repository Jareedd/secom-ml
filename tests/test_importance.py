import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from secom_ml.importance import (
    pairwise_stability,
    score_importance_folds,
    sensor_correlations,
    summarise_importance,
)


class HeldOutSignalClassifier(ClassifierMixin, BaseEstimator):
    """Fail if scoring sees a fitting row; predict directly from a known signal."""

    def fit(self, X, y):
        self.fit_indices_ = set(X.index)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        assert not self.fit_indices_.intersection(X.index)
        probability = 0.1 + 0.8 * X["signal"].to_numpy()
        return np.column_stack([1 - probability, probability])


def test_importance_scores_only_unseen_rows_and_finds_signal_without_mutating_data():
    rng = np.random.default_rng(0)
    y = pd.Series(np.tile([0, 1, 0, 0], 30))
    X = pd.DataFrame({"signal": y, "irrelevant": rng.normal(size=len(y))})
    original = X.copy(deep=True)
    folds = [(np.arange(60), np.arange(60, 120)), (np.arange(60, 120), np.arange(60))]
    per_fold, scores = score_importance_folds(
        HeldOutSignalClassifier(), X, y, folds, n_repeats=3,
    )
    pd.testing.assert_frame_equal(X, original)
    assert (scores["pr_auc"] == 1).all()
    assert (per_fold.loc[per_fold["sensor"] == "signal", "importance_mean"] > 0.5).all()
    assert (per_fold.loc[per_fold["sensor"] == "irrelevant", "importance_mean"] == 0).all()
    # Zeros must not earn top-10 membership just because there are few features.
    assert not per_fold.loc[per_fold["sensor"] == "irrelevant", "top_10"].any()
    summary = summarise_importance(per_fold).set_index("sensor")
    assert summary.loc["signal", "positive_folds"] == 2
    assert summary.loc["signal", "top_10_folds"] == 2
    assert summary.loc["irrelevant", "top_10_folds"] == 0
    stability = pairwise_stability(per_fold).iloc[0]
    assert stability["rank_spearman"] == pytest.approx(1)
    assert stability["top_10_jaccard"] == 1


def test_importance_rejects_overlap_before_fitting():
    X = pd.DataFrame({"signal": [0, 1, 0, 1]})
    y = X["signal"]
    with pytest.raises(ValueError, match="disjoint"):
        score_importance_folds(
            HeldOutSignalClassifier(), X, y, [(np.array([0, 1]), np.array([1, 2]))],
        )


def test_stability_retains_negative_importances_and_detects_changed_top_sets():
    table = pd.DataFrame({
        "view": ["time"] * 4, "sensor": ["a", "b", "a", "b"], "fold": [1, 1, 2, 2],
        "importance_mean": [0.2, -0.1, -0.2, 0.3], "shuffle_sd": [0.01] * 4,
        "rank": [1, 2, 2, 1], "top_10": [True, False, False, True],
    })
    summary = summarise_importance(table).set_index("sensor")
    assert summary.loc["a", "mean_drop"] == pytest.approx(0)
    assert summary.loc["a", "positive_folds"] == 1
    assert summary.loc["b", "min_drop"] < 0
    pair = pairwise_stability(table).iloc[0]
    assert pair["rank_spearman"] == pytest.approx(-1)
    assert pair["top_10_overlap"] == 0
    assert pair["top_10_jaccard"] == 0


def test_correlations_exclude_self_and_find_an_inverse_twin():
    X = pd.DataFrame({"a": [1, 2, 3, 4], "twin": [-1, -2, -3, -4],
                      "other": [1, 0, 1, 0]})
    table = sensor_correlations(X).set_index("sensor")
    assert table.loc["a", "nearest_sensor"] == "twin"
    assert table.loc["a", "nearest_abs_correlation"] == pytest.approx(1)
    assert table.loc["a", "correlated_partners_ge_0_95"] == 1
