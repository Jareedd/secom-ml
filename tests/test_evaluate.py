import numpy as np
import pandas as pd
import pytest

from secom_ml.evaluate import score_folds, stratified_folds, time_ordered_folds
from secom_ml.models import build_models, failure_scores


@pytest.fixture
def runs() -> tuple[pd.DataFrame, pd.Series]:
    """120 made-up runs with 5 sensors; failing runs read higher on sensor a."""
    rng = np.random.default_rng(0)
    y = pd.Series(np.r_[np.ones(24), np.zeros(96)].astype(int)).sample(frac=1, random_state=0)
    y = y.reset_index(drop=True)
    X = pd.DataFrame(rng.normal(size=(120, 5)), columns=list("abcde"))
    X["a"] += 2 * y
    X.iloc[::7, 1] = np.nan  # some gaps for the imputer
    return X, y


def test_models_come_in_the_fixed_order() -> None:
    assert list(build_models()) == [
        "logistic regression",
        "random forest",
        "gradient boosting",
        "isolation forest",
        "xgboost",
    ]


def test_no_model_reweights_the_classes() -> None:
    for name, pipeline in build_models().items():
        settings = pipeline["model"].get_params()
        assert settings.get("class_weight") is None, name
        assert settings.get("scale_pos_weight") is None, name  # XGBoost's name for the same thing


def test_only_logistic_regression_is_scaled() -> None:
    scaled = [name for name, pipeline in build_models().items() if "scale" in pipeline.named_steps]
    assert scaled == ["logistic regression"]


def test_every_model_scores_failing_runs_higher(runs) -> None:
    X, y = runs
    for name, model in build_models().items():
        scores = failure_scores(model.fit(X, y), X)
        assert scores[y == 1].mean() > scores[y == 0].mean(), name


def test_stratified_folds_hold_out_every_run_exactly_once(runs) -> None:
    X, y = runs
    held_out = np.concatenate([held_out_rows for _, held_out_rows in stratified_folds(X, y)])
    assert sorted(held_out) == list(range(len(X)))


def test_time_ordered_folds_never_fit_on_the_future(runs) -> None:
    X, _ = runs
    for fit_rows, held_out_rows in time_ordered_folds(X):
        assert fit_rows.max() < held_out_rows.min()


def test_score_folds_reports_one_row_per_fold_and_a_score_per_run(runs) -> None:
    X, y = runs
    model = build_models()["logistic regression"]
    per_fold, held_out_scores = score_folds(model, X, y, stratified_folds(X, y))

    assert len(per_fold) == 5
    assert per_fold["held_out_fails"].sum() == y.sum()
    assert held_out_scores.notna().all()
    assert per_fold["roc_auc"].mean() > 0.8  # the signal on sensor a is strong
