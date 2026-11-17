"""Phase 2: make the split, decide the cleaning rules, and print what happened.

Run with:  python -m secom_ml.prepare

This script never reads a test-period sensor value. What it says about the test
period comes from timestamps and from subtracting label counts.
"""

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from secom_ml.cleaning import choose_sensors, preprocessing_steps, sensor_report
from secom_ml.config import MAX_MISSING_SHARE, N_FOLDS, RANDOM_SEED
from secom_ml.data import load_secom
from secom_ml.split import cut_index, load_train


def describe_split(y_train: pd.Series) -> None:
    _, y_all, timestamp = load_secom()
    cut = cut_index(timestamp)
    train_time, test_time = timestamp.iloc[:cut], timestamp.iloc[cut:]
    test_fails = int(y_all.sum() - y_train.sum())  # by subtraction, not by reading test labels

    print("\n=== SPLIT (by time, no shuffling) ===")
    print(f"  train: {len(train_time)} runs, {train_time.iloc[0]} to {train_time.iloc[-1]}")
    print(f"         {y_train.sum()} failing runs = {y_train.mean():.1%}")
    print(f"  test:  {len(test_time)} runs, {test_time.iloc[0]} to {test_time.iloc[-1]}")
    print(f"         {test_fails} failing runs = {test_fails / len(test_time):.1%}")
    print(f"  every training run is earlier than every test run: "
          f"{train_time.iloc[-1] < test_time.iloc[0]}")
    print(f"  gap between last training run and first test run: "
          f"{test_time.iloc[0] - train_time.iloc[-1]}")


def describe_cleaning(X_train: pd.DataFrame) -> list[str]:
    report = sensor_report(X_train)
    kept = choose_sensors(X_train)

    print("\n=== CLEANING RULES (decided on training runs only) ===")
    print(f"  sensors to start with: {len(report)}")
    print(f"  dropped, more than {MAX_MISSING_SHARE:.0%} missing: {report['too_empty'].sum()}")
    print(f"  dropped, never changes: {report['constant'].sum()}")
    print(f"  dropped for both reasons: {(report['too_empty'] & report['constant']).sum()}")
    print(f"  sensors kept: {len(kept)}")

    X_kept = X_train[kept]
    print(f"\n  Missing cells left for the imputer to fill: {X_kept.isna().mean().mean():.1%}")
    print(f"  Emptiest kept sensor: {X_kept.isna().mean().max():.1%} missing")

    # The zero-variance rule only catches sensors with exactly one value. Count
    # any that are almost constant and slip through, so the rule's limit is on record.
    top_value_share = X_kept.apply(lambda column: column.value_counts(normalize=True).iloc[0])
    print(f"  Kept sensors that are nearly constant (one value in over 99% of readings): "
          f"{(top_value_share > 0.99).sum()}")
    return kept


def check_folds(X_train: pd.DataFrame, y_train: pd.Series) -> None:
    """Fit the preprocessing separately on each fold and confirm that it behaves."""
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    preprocess = Pipeline(preprocessing_steps(scale=True))
    all_train_medians = X_train.median()

    print(f"\n=== PREPROCESSING, FIT PER FOLD ({N_FOLDS} stratified folds) ===")
    for number, (fit_rows, held_out_rows) in enumerate(folds.split(X_train, y_train), start=1):
        X_fit = X_train.iloc[fit_rows]
        preprocess.fit(X_fit)

        fold_medians = pd.Series(preprocess["impute"].statistics_, index=X_train.columns)
        differ = (fold_medians != all_train_medians).sum()
        constant_here = (X_fit.nunique(dropna=True) <= 1).sum()
        print(
            f"  fold {number}: fit on {len(fit_rows)} runs, hold out {len(held_out_rows)} "
            f"({y_train.iloc[held_out_rows].sum()} failing) | "
            f"{differ} sensor medians differ from the all-training medians | "
            f"{constant_here} kept sensors constant in this fold"
        )


def main() -> None:
    X_train, y_train, _ = load_train()
    describe_split(y_train)
    kept = describe_cleaning(X_train)
    check_folds(X_train[kept], y_train)


if __name__ == "__main__":
    main()
