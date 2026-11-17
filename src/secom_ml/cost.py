"""Turning a failure probability into an alarm decision using costs.

The cost table (an assumption, stated so it can be challenged):

                          run actually fails      run actually passes
    alarm (inspect)       0   (caught it)         C_FA   (false alarm)
    no alarm              C_miss (missed it)      0

For one run with failure probability p:
    expected cost of alarming     = (1 - p) * C_FA
    expected cost of staying quiet = p * C_miss
Alarming is the cheaper choice when (1 - p) * C_FA < p * C_miss, which
rearranges to
    p > C_FA / (C_FA + C_miss).

So the threshold depends only on the two costs. It is not fitted to data, which
means it cannot overfit; it does rely on p being a calibrated probability.
"""

import numpy as np

from secom_ml.config import N_BOOTSTRAP, RANDOM_SEED


def alarm_threshold(cost_false_alarm: float, cost_miss: float) -> float:
    """Return the probability above which alarming is cheaper than staying quiet."""
    return cost_false_alarm / (cost_false_alarm + cost_miss)


def cost_per_1000_runs(
    y: np.ndarray, alarm: np.ndarray, cost_false_alarm: float, cost_miss: float
) -> float:
    """Total cost of a set of alarm decisions, scaled to 1,000 runs."""
    y, alarm = np.asarray(y, dtype=bool), np.asarray(alarm, dtype=bool)
    false_alarms = (alarm & ~y).sum()
    misses = (~alarm & y).sum()
    return 1000 * (false_alarms * cost_false_alarm + misses * cost_miss) / len(y)


def alarm_policies(probability: np.ndarray, threshold: float) -> dict[str, np.ndarray]:
    """Return the alarm decisions of each policy being compared."""
    probability = np.asarray(probability)
    return {
        "cost-based threshold": probability > threshold,
        "never alarm": np.zeros(len(probability), dtype=bool),
        "default 0.5 cutoff": probability > 0.5,
        "inspect every run": np.ones(len(probability), dtype=bool),
    }


def policy_summary(
    y: np.ndarray, probability: np.ndarray, cost_false_alarm: float, cost_miss: float
) -> dict:
    """Describe the cost-based policy and price it against the baselines."""
    y = np.asarray(y, dtype=bool)
    threshold = alarm_threshold(cost_false_alarm, cost_miss)
    policies = alarm_policies(probability, threshold)
    costs = {
        name: cost_per_1000_runs(y, alarm, cost_false_alarm, cost_miss)
        for name, alarm in policies.items()
    }
    alarm = policies["cost-based threshold"]
    best_naive = min(costs["never alarm"], costs["inspect every run"])
    return {
        "threshold": threshold,
        "alarms": int(alarm.sum()),
        "alarm_rate": alarm.mean(),
        "caught": int((alarm & y).sum()),
        "recall": (alarm & y).sum() / y.sum(),
        "false_alarms": int((alarm & ~y).sum()),
        "false_alarm_rate": (alarm & ~y).sum() / (~y).sum(),
        "cost": costs["cost-based threshold"],
        "cost_never_alarm": costs["never alarm"],
        "cost_default_cutoff": costs["default 0.5 cutoff"],
        "cost_inspect_all": costs["inspect every run"],
        "saving_vs_never_alarm": 1 - costs["cost-based threshold"] / costs["never alarm"],
        "saving_vs_best_naive": 1 - costs["cost-based threshold"] / best_naive,
    }


def cost_curve(
    y: np.ndarray, probability: np.ndarray, thresholds: np.ndarray,
    cost_false_alarm: float, cost_miss: float,
) -> np.ndarray:
    """Cost per 1,000 runs at each candidate threshold."""
    probability = np.asarray(probability)
    return np.array(
        [cost_per_1000_runs(y, probability > t, cost_false_alarm, cost_miss) for t in thresholds]
    )


def bootstrap_interval(statistic, y: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    """95% confidence interval for statistic(y, probability).

    The runs we have are one sample of the runs we could have had. Resampling
    them with replacement, many times, shows how much the statistic would move
    with a different sample. The interval is the middle 95% of those values.
    """
    y, probability = np.asarray(y), np.asarray(probability)
    rng = np.random.default_rng(RANDOM_SEED)
    values = []
    for _ in range(N_BOOTSTRAP):
        rows = rng.integers(0, len(y), size=len(y))
        if y[rows].sum() == 0:
            continue  # a resample with no failing runs cannot be scored
        values.append(statistic(y[rows], probability[rows]))
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)
