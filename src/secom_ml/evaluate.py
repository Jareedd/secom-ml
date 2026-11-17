"""Cross-validation: how the folds are made and how a model is scored on them."""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from secom_ml.config import N_FOLDS, N_TIME_FOLDS, RANDOM_SEED
from secom_ml.models import failure_scores

Folds = list[tuple[np.ndarray, np.ndarray]]


def stratified_folds(X: pd.DataFrame, y: pd.Series) -> Folds:
    """Shuffled folds that each hold the same share of failing runs.

    This is the ranking scheme. Every model gets the same folds.
    """
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    return list(splitter.split(X, y))


def time_ordered_folds(X: pd.DataFrame) -> Folds:
    """Folds that respect time: fit on everything earlier, score the next block.

    This is a check, not the ranking scheme. It shows what happens when a model
    must predict runs that come after everything it learned from.
    """
    return list(TimeSeriesSplit(n_splits=N_TIME_FOLDS).split(X))


def score_folds(
    model: Pipeline, X: pd.DataFrame, y: pd.Series, folds: Folds
) -> tuple[pd.DataFrame, pd.Series]:
    """Fit a fresh copy of the model on each fold and score the runs it did not see.

    Returns:
      per_fold: one row per fold with its size, base rate, PR AUC and ROC AUC.
      held_out_scores: each run's score from the fold where it was held out
        (empty for runs that were never held out, which happens with time folds).
    """
    rows = []
    held_out_scores = pd.Series(np.nan, index=X.index, dtype=float)

    for number, (fit_rows, held_out_rows) in enumerate(folds, start=1):
        fitted = clone(model).fit(X.iloc[fit_rows], y.iloc[fit_rows])
        scores = failure_scores(fitted, X.iloc[held_out_rows])
        y_held_out = y.iloc[held_out_rows]

        held_out_scores.iloc[held_out_rows] = scores
        rows.append(
            {
                "fold": number,
                "fit_runs": len(fit_rows),
                "held_out_runs": len(held_out_rows),
                "held_out_fails": int(y_held_out.sum()),
                "base_rate": y_held_out.mean(),
                # Average precision is the step-wise area under the
                # precision-recall curve. Joining the points with straight
                # lines instead would overstate it.
                "pr_auc": average_precision_score(y_held_out, scores),
                "roc_auc": roc_auc_score(y_held_out, scores),
            }
        )
    return pd.DataFrame(rows), held_out_scores
