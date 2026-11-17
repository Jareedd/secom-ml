"""The models being compared, each as a full Pipeline.

Four were planned from the start; XGBoost was added later as a fifth. Every
setting below was fixed before that model's score was looked at, and none was
tuned afterwards. With 87 failing runs to learn from, a tuning search would
mostly fit noise.

No model uses class weights or resampling. Those are indirect ways of moving
the alarm threshold, and they distort predicted probabilities. This project
sets the threshold directly from costs instead, which needs honest
probabilities.
"""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    GradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from secom_ml.cleaning import preprocessing_steps
from secom_ml.config import N_FOLDS, RANDOM_SEED


def build_models() -> dict[str, Pipeline]:
    """Return the models by name, in the fixed order used in every table and chart."""
    return {
        # Linear baseline. C=0.01 is heavy shrinkage: there are about five times
        # as many sensors as failing runs, so unshrunk weights would memorise.
        # The default L2 penalty spreads weight across near-duplicate sensors
        # (most sensors have one); an L1 penalty would pick one of each pair
        # more or less at random.
        "logistic regression": Pipeline(
            preprocessing_steps(scale=True)
            + [("model", LogisticRegression(C=0.01, max_iter=5000))]
        ),
        # Many deep trees averaged. Leaves must hold at least 5 runs, so a leaf
        # cannot be built around a single failing run.
        "random forest": Pipeline(
            preprocessing_steps(scale=False)
            + [
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        min_samples_leaf=5,
                        n_jobs=-1,
                        random_state=RANDOM_SEED,
                    ),
                )
            ]
        ),
        # Shallow trees added slowly, each fit on a random 80% of the runs. All
        # three choices are the conservative ones for a small data set.
        "gradient boosting": Pipeline(
            preprocessing_steps(scale=False)
            + [
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=3,
                        subsample=0.8,
                        random_state=RANDOM_SEED,
                    ),
                )
            ]
        ),
        # Unsupervised baseline: never sees a pass/fail label. It scores how
        # unusual a run's sensor readings are. If the supervised models cannot
        # beat it, the labels are not adding anything.
        "isolation forest": Pipeline(
            preprocessing_steps(scale=False)
            + [
                (
                    "model",
                    IsolationForest(n_estimators=300, n_jobs=-1, random_state=RANDOM_SEED),
                )
            ]
        ),
        # Added after the first four had been scored, to answer "did you try
        # XGBoost?". Settings were fixed before it was run and not changed
        # after. The first four match the gradient boosting above on purpose,
        # so any difference comes from what XGBoost adds: each tree sees a
        # random half of the sensors (useful when most sensors have a
        # near-duplicate) and leaf values carry an L2 penalty. No search, no
        # early stopping, no scale_pos_weight.
        "xgboost": Pipeline(
            preprocessing_steps(scale=False)
            + [
                (
                    "model",
                    XGBClassifier(
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=3,
                        subsample=0.8,
                        colsample_bytree=0.5,
                        reg_lambda=1.0,
                        n_jobs=-1,
                        random_state=RANDOM_SEED,
                    ),
                )
            ]
        ),
    }


def build_calibrated_model(model_name: str) -> CalibratedClassifierCV:
    """Return one of the models above with a sigmoid calibrator on top.

    How it is fit: the training runs are cut into 5 stratified folds, and each
    run gets a score from a model that did not see it. One S-curve (two
    numbers: a slope and a shift) is fit to map those scores to probabilities.
    Then a single model is refit on all the training runs.

    Sigmoid, not isotonic: isotonic fits a free-form staircase, which overfits
    with only 87 failing runs and leaves just a few distinct probability values.
    """
    return CalibratedClassifierCV(
        build_models()[model_name],
        method="sigmoid",
        cv=StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED),
        ensemble=False,
    )


# Two models go to the final test, and both are reported whatever they score.
# The headline result comes from the primary only. This was fixed before the
# test set was opened, so the better-looking one cannot be promoted afterwards.
PRIMARY_MODEL = "calibrated xgboost"
SECONDARY_MODEL = "random forest"


def build_final_models() -> dict:
    """Return the two models that phases 5 and 6 use, primary first.

    Calibrated XGBoost (primary). XGBoost ranked first in phase 3 on the rule
    set at the start: highest mean PR AUC over the stratified folds, 0.238
    against the random forest's 0.223. That gap is well inside the fold-to-fold
    noise. It is used WITH the sigmoid calibrator, because on held-out training
    runs the uncalibrated model predicted an average failure probability of
    2.9% when 6.9% of runs failed, with every reliability bin below what
    actually happened. The cost threshold needs probabilities at the right
    level. A rule set before that run said "keep calibration only if the Brier
    score is lower"; the calibrated Brier score is marginally higher (0.0632
    against 0.0629), and the rule is set aside here, openly, because the
    reliability table answers the question the rule was a stand-in for.

    Random forest (secondary). Statistically tied with XGBoost on ranking,
    steadier forward in time, and well calibrated as trained (average
    prediction 7.6% against 6.9%, Brier 0.0616), so it is used WITHOUT a
    calibrator.
    """
    return {
        PRIMARY_MODEL: build_calibrated_model("xgboost"),
        SECONDARY_MODEL: build_models()["random forest"],
    }


def failure_scores(fitted_model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Return one number per run where higher means "more likely to fail".

    Classifiers give a probability. The isolation forest gives an anomaly score
    instead (not a probability), which is enough for ranking runs.
    """
    if hasattr(fitted_model, "predict_proba"):
        return fitted_model.predict_proba(X)[:, 1]
    return -fitted_model.score_samples(X)  # score_samples is low for unusual runs
