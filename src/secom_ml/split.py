"""The one train/test split: earliest 80% of runs to train, latest 20% to test.

load_train() is what every phase uses. load_test() exists for the final
evaluation script only; tests/test_split.py fails if any other module in this
package mentions it.
"""

import pandas as pd

from secom_ml.config import TEST_FRACTION
from secom_ml.data import load_secom


def cut_index(timestamp: pd.Series, test_fraction: float = TEST_FRACTION) -> int:
    """Return how many of the earliest runs go to training.

    The runs must already be sorted oldest to newest.
    """
    if not timestamp.is_monotonic_increasing:
        raise ValueError("runs must be in time order before splitting")

    cut = int(len(timestamp) * (1 - test_fraction))

    # Two runs with the same timestamp must land on the same side, or "train is
    # strictly earlier than test" would not be true.
    if timestamp.iloc[cut - 1] == timestamp.iloc[cut]:
        raise ValueError("the cut falls between two runs that share a timestamp")
    return cut


def load_train() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return (X, y, timestamp) for the earliest 80% of runs."""
    X, y, timestamp = load_secom()
    cut = cut_index(timestamp)
    return X.iloc[:cut], y.iloc[:cut], timestamp.iloc[:cut]


def load_test() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return (X, y, timestamp) for the latest 20% of runs.

    Call this exactly once, from the final evaluation script, after every
    modelling decision has been made.
    """
    X, y, timestamp = load_secom()
    cut = cut_index(timestamp)
    return X.iloc[cut:], y.iloc[cut:], timestamp.iloc[cut:]
