import numpy as np
import pytest

from secom_ml.cost import (
    alarm_threshold,
    bootstrap_interval,
    cost_curve,
    cost_per_1000_runs,
    policy_summary,
)


@pytest.mark.parametrize(
    ("ratio", "expected"), [(5, 1 / 6), (10, 1 / 11), (20, 1 / 21), (50, 1 / 51)]
)
def test_threshold_is_false_alarm_cost_over_total_cost(ratio: float, expected: float) -> None:
    assert alarm_threshold(cost_false_alarm=1, cost_miss=ratio) == pytest.approx(expected)


def test_threshold_depends_only_on_the_ratio_of_the_costs() -> None:
    assert alarm_threshold(1, 10) == pytest.approx(alarm_threshold(250, 2500))


def test_only_false_alarms_and_misses_cost_anything() -> None:
    y = np.array([1, 1, 0, 0, 0])
    alarm = np.array([1, 0, 1, 0, 0])  # one caught, one missed, one false alarm, two quiet
    # 1 false alarm * 1 + 1 miss * 10 = 11 over 5 runs = 2,200 per 1,000 runs
    assert cost_per_1000_runs(y, alarm, cost_false_alarm=1, cost_miss=10) == pytest.approx(2200)


def test_baselines_are_priced_correctly() -> None:
    y = np.array([1] * 10 + [0] * 90)
    probability = np.full(100, 0.2)  # every run is over the 10:1 threshold, none over 0.5
    summary = policy_summary(y, probability, cost_false_alarm=1, cost_miss=10)

    assert summary["cost_never_alarm"] == pytest.approx(1000)  # 10 misses * 10, per 100 runs
    assert summary["cost_inspect_all"] == pytest.approx(900)  # 90 false alarms * 1
    assert summary["cost_default_cutoff"] == summary["cost_never_alarm"]
    assert summary["recall"] == 1 and summary["false_alarm_rate"] == 1
    assert summary["saving_vs_never_alarm"] == pytest.approx(0.10)
    assert summary["saving_vs_best_naive"] == pytest.approx(0)


def test_formula_threshold_is_the_cheapest_when_probabilities_are_calibrated() -> None:
    """The reason to trust the formula: simulate honest probabilities and check it wins."""
    rng = np.random.default_rng(0)
    probability = rng.beta(1, 12, size=200_000)  # mostly low, like a rare failure
    y = rng.uniform(size=200_000) < probability  # outcomes that match the probabilities

    thresholds = np.linspace(0.01, 0.40, 40)
    curve = cost_curve(y, probability, thresholds, cost_false_alarm=1, cost_miss=10)

    assert thresholds[curve.argmin()] == pytest.approx(alarm_threshold(1, 10), abs=0.02)


def test_bootstrap_interval_brackets_the_estimate() -> None:
    rng = np.random.default_rng(0)
    y = (rng.uniform(size=500) < 0.1).astype(int)
    low, high = bootstrap_interval(lambda y_, p_: y_.mean(), y, np.zeros(500))

    assert low < y.mean() < high
    assert high - low < 0.08  # roughly +/- 2 standard errors of a 10% rate on 500 runs
