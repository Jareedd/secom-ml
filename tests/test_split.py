from pathlib import Path

import pandas as pd
import pytest

from secom_ml.split import cut_index

PACKAGE_DIR = Path(__file__).resolve().parents[1] / "src" / "secom_ml"

# The only modules allowed to mention the test-set loader: the one that defines
# it, and the final evaluation script that calls it once.
ALLOWED_TO_MENTION_LOAD_TEST = {"split.py", "final_test.py"}


def hourly(n: int) -> pd.Series:
    return pd.Series(pd.date_range("2008-07-19", periods=n, freq="h"))


def test_cut_puts_the_earliest_80_percent_in_training() -> None:
    assert cut_index(hourly(10), test_fraction=0.2) == 8


def test_cut_refuses_unsorted_runs() -> None:
    backwards = hourly(10).iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="time order"):
        cut_index(backwards, test_fraction=0.2)


def test_cut_refuses_to_separate_simultaneous_runs() -> None:
    timestamp = hourly(10)
    timestamp.iloc[8] = timestamp.iloc[7]  # runs 7 and 8 straddle the cut
    with pytest.raises(ValueError, match="share a timestamp"):
        cut_index(timestamp, test_fraction=0.2)


def test_only_the_final_script_touches_the_test_set() -> None:
    offenders = [
        path.name
        for path in PACKAGE_DIR.glob("*.py")
        if "load_test" in path.read_text(encoding="utf-8")
        and path.name not in ALLOWED_TO_MENTION_LOAD_TEST
    ]
    assert offenders == []
