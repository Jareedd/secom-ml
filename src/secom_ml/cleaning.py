"""Cleaning rules, and the preprocessing steps every model shares.

Two different kinds of cleaning live here, and the difference matters:

1. Which sensors to keep. Decided ONCE, from the training runs, with rules that
   never look at the pass/fail label. The result is a fixed list of sensor
   names, so every model and every fold sees the same features.

2. Filling gaps and scaling. These learn numbers from data (a median, a mean, a
   standard deviation), so they are Pipeline steps. scikit-learn then re-fits
   them on the training part of each cross-validation fold, and the held-out
   part of the fold never influences them.
"""

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from secom_ml.config import MAX_MISSING_SHARE


def sensor_report(X_train: pd.DataFrame) -> pd.DataFrame:
    """One row per sensor: how empty it is, whether it is constant, whether we keep it."""
    report = pd.DataFrame(
        {
            "missing_share": X_train.isna().mean(),
            # "Only one distinct value" rather than std == 0: the standard
            # deviation of a constant float column is not always exactly zero.
            "constant": X_train.nunique(dropna=True) <= 1,
        }
    )
    report["too_empty"] = report["missing_share"] > MAX_MISSING_SHARE
    report["keep"] = ~report["too_empty"] & ~report["constant"]
    return report


def choose_sensors(X_train: pd.DataFrame) -> list[str]:
    """Return the names of the sensors to keep. Pass training runs only."""
    report = sensor_report(X_train)
    return report.index[report["keep"]].tolist()


def preprocessing_steps(scale: bool) -> list[tuple]:
    """Return the Pipeline steps that go in front of a model.

    Median imputation: a sensor's median is not dragged around by the extreme
    readings these sensors have, unlike its mean.

    scale=True adds standardisation. Logistic regression needs it (sensor
    spreads range from 0.0001 to 6,400); tree models do not, because a tree
    only asks "is this reading above or below a cut point?".
    """
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return steps
