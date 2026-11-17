import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from secom_ml.cleaning import choose_sensors, preprocessing_steps, sensor_report


def example_runs() -> pd.DataFrame:
    """Ten runs with one sensor of each kind the rules care about."""
    nan = np.nan
    return pd.DataFrame(
        {
            "healthy": [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "constant": [5.0] * 10,
            "constant_with_gaps": [5.0, nan, 5, 5, nan, 5, 5, 5, 5, 5],
            "exactly_half_missing": [1.0, 2, 3, 4, 5, nan, nan, nan, nan, nan],
            "mostly_missing": [1.0, 2, 3, 4, nan, nan, nan, nan, nan, nan],
        }
    )


def test_drops_constant_and_mostly_missing_sensors() -> None:
    assert choose_sensors(example_runs()) == ["healthy", "exactly_half_missing"]


def test_exactly_half_missing_is_kept_because_the_rule_is_more_than_half() -> None:
    report = sensor_report(example_runs())
    assert not report.loc["exactly_half_missing", "too_empty"]
    assert report.loc["mostly_missing", "too_empty"]


def test_preprocessing_learns_only_from_the_rows_it_is_fit_on() -> None:
    """The leakage check: held-out rows must not move the median, mean or spread."""
    fit_rows = pd.DataFrame({"sensor": [1.0, 2.0, np.nan, 3.0]})
    held_out_rows = pd.DataFrame({"sensor": [1000.0, np.nan]})

    preprocess = Pipeline(preprocessing_steps(scale=True)).fit(fit_rows)

    assert preprocess["impute"].statistics_[0] == 2.0  # median of 1, 2, 3
    assert preprocess["scale"].mean_[0] == 2.0  # mean of 1, 2, 2 (gap filled), 3

    transformed = preprocess.transform(held_out_rows)
    assert transformed[0, 0] > 100  # the extreme held-out value was not used for scaling
    assert transformed[1, 0] == 0.0  # the held-out gap got the fit rows' median


def test_tree_models_get_imputation_without_scaling() -> None:
    assert [name for name, _ in preprocessing_steps(scale=False)] == ["impute"]
