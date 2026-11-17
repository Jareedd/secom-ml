"""Phase 3 follow-up: is the cross-validation score inflated by drift over time?

Run with:  python -m secom_ml.check_drift   (after compare_models)

Early training runs failed about four times as often as late ones. Shuffled
folds put early and late runs on both sides of every fold, so a model can score
well just by recognising which period a run came from. That skill is useless on
future runs. Three checks measure how much of the score it explains:

1. How well does a model that knows ONLY the date do?
2. Can the sensors tell which period a run came from?
3. Inside a single period, where the date cannot help, how well do the models rank?
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from secom_ml.cleaning import choose_sensors, preprocessing_steps
from secom_ml.config import RANDOM_SEED, RESULTS_DIR
from secom_ml.evaluate import score_folds, stratified_folds
from secom_ml.models import build_models
from secom_ml.split import load_train

MODEL_NAMES = list(build_models())
N_BLOCKS = 5


def date_only_model_scores(y: pd.Series) -> pd.DataFrame:
    """Cross-validate a model whose single input is the run's position in time."""
    run_order = pd.DataFrame({"run_order": np.arange(len(y))}, index=y.index)
    model = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression())])
    per_fold, _ = score_folds(model, run_order, y, stratified_folds(run_order, y))
    return per_fold


def period_detector_scores(X: pd.DataFrame) -> pd.DataFrame:
    """Cross-validate a model that predicts "early run or late run?" from the sensors.

    Early means the first 40% of training runs, the stretch with the high fail rate.
    """
    is_early = pd.Series((np.arange(len(X)) < 0.4 * len(X)).astype(int), index=X.index)
    model = Pipeline(
        preprocessing_steps(scale=False)
        + [("model", RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_SEED))]
    )
    per_fold, _ = score_folds(model, X, is_early, stratified_folds(X, is_early))
    return per_fold


def within_block_scores(held_out: pd.DataFrame) -> pd.DataFrame:
    """Score each model inside each consecutive block of runs."""
    rows = []
    for number, block_rows in enumerate(np.array_split(np.arange(len(held_out)), N_BLOCKS), 1):
        block = held_out.iloc[block_rows]
        for name in MODEL_NAMES:
            rows.append(
                {
                    "block": number,
                    "fails": int(block["fail"].sum()),
                    "base_rate": block["fail"].mean(),
                    "model": name,
                    "mean_score": block[name].mean(),
                    "roc_auc": roc_auc_score(block["fail"], block[name]),
                    "pr_auc": average_precision_score(block["fail"], block[name]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    X_train, y, _ = load_train()
    X = X_train[choose_sensors(X_train)]
    held_out = pd.read_csv(RESULTS_DIR / "cv_held_out_scores.csv", index_col="run")
    fold_scores = pd.read_csv(RESULTS_DIR / "cv_fold_scores.csv")
    stratified = fold_scores[fold_scores["scheme"] == "stratified"].groupby("model", sort=False)

    print("\n=== 1. A model that knows only the date (stratified 5-fold) ===")
    date_only = date_only_model_scores(y)
    print(f"  date only            PR AUC {date_only['pr_auc'].mean():.3f}   "
          f"ROC AUC {date_only['roc_auc'].mean():.3f}")
    for name in MODEL_NAMES:
        scores = stratified.get_group(name)
        print(f"  {name:<20} PR AUC {scores['pr_auc'].mean():.3f}   "
              f"ROC AUC {scores['roc_auc'].mean():.3f}")

    print("\n=== 2. Can the sensors tell an early run from a late one? ===")
    detector = period_detector_scores(X)
    print(f"  ROC AUC {detector['roc_auc'].mean():.3f}  (1.0 = the sensors give the period away)")

    print("\n=== 3. Ranking skill inside each block of runs (stratified held-out scores) ===")
    within = within_block_scores(held_out)
    blocks = within.drop_duplicates("block").set_index("block")
    header = "".join(f"{f'block {b} ({blocks.fails[b]} fails)':>19}" for b in blocks.index)
    print(f"  ROC AUC             {header}{'mean':>8}{'all runs':>10}")
    for name in MODEL_NAMES:
        row = within[within["model"] == name]["roc_auc"]
        pooled = roc_auc_score(held_out["fail"], held_out[name])
        print(f"  {name:<20}" + "".join(f"{value:>19.3f}" for value in row)
              + f"{row.mean():>8.3f}{pooled:>10.3f}")

    for name in ["xgboost", "random forest"]:
        print(f"\n  {name}: average predicted probability vs actual fail rate, by block")
        for row in within[within["model"] == name].itertuples():
            print(f"    block {row.block}: predicted {row.mean_score:.1%}   "
                  f"actual {row.base_rate:.1%}")


if __name__ == "__main__":
    main()
