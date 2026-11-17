import numpy as np
import pandas as pd
import pytest

from secom_ml.calibrate import N_BINS, brier_report, reliability_table, wilson_interval
from secom_ml.models import (
    PRIMARY_MODEL,
    SECONDARY_MODEL,
    build_calibrated_model,
    build_final_models,
)


def test_wilson_interval_contains_the_observed_rate_and_stays_between_0_and_1() -> None:
    low, high = wilson_interval(fails=5, runs=157)
    assert 0 <= low < 5 / 157 < high <= 1


def test_wilson_interval_is_not_zero_width_when_nothing_failed() -> None:
    low, high = wilson_interval(fails=0, runs=157)
    assert low == pytest.approx(0)
    assert high > 0.01  # zero failures seen does not prove the rate is zero


def test_wilson_interval_narrows_with_more_runs() -> None:
    small = wilson_interval(fails=5, runs=50)
    large = wilson_interval(fails=50, runs=500)
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_reliability_table_accounts_for_every_run() -> None:
    rng = np.random.default_rng(0)
    probability = pd.Series(rng.uniform(0, 0.3, size=800))
    y = pd.Series((rng.uniform(size=800) < probability).astype(int))

    table = reliability_table(y, probability)

    assert len(table) == N_BINS
    assert table["runs"].sum() == 800
    assert table["fails"].sum() == y.sum()
    assert table["predicted"].is_monotonic_increasing


def test_brier_skill_is_zero_for_the_base_rate_and_positive_for_a_perfect_forecast() -> None:
    y = pd.Series([0] * 90 + [1] * 10)
    base_rate_forecast = pd.Series(np.full(100, 0.1))
    perfect_forecast = y.astype(float)

    assert brier_report(y, base_rate_forecast)["skill"] == pytest.approx(0)
    assert brier_report(y, perfect_forecast)["skill"] == pytest.approx(1)


def test_primary_model_is_calibrated_xgboost_and_secondary_is_the_plain_forest() -> None:
    final = build_final_models()
    assert list(final) == [PRIMARY_MODEL, SECONDARY_MODEL] == ["calibrated xgboost", "random forest"]

    primary, secondary = final[PRIMARY_MODEL], final[SECONDARY_MODEL]
    assert type(primary).__name__ == "CalibratedClassifierCV"
    assert primary.method == "sigmoid"
    assert type(primary.estimator["model"]).__name__ == "XGBClassifier"
    assert type(secondary["model"]).__name__ == "RandomForestClassifier"


def test_calibrated_model_returns_probabilities() -> None:
    rng = np.random.default_rng(0)
    y = pd.Series(np.r_[np.ones(30), np.zeros(120)].astype(int))
    X = pd.DataFrame(rng.normal(size=(150, 4)), columns=list("abcd"))
    X["a"] += 2 * y

    probability = build_calibrated_model("random forest").fit(X, y).predict_proba(X)[:, 1]

    assert ((probability >= 0) & (probability <= 1)).all()
    assert probability[y == 1].mean() > probability[y == 0].mean()
