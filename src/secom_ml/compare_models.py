"""Phase 3: compare the models on the training runs.

Run with:  python -m secom_ml.compare_models

Ranking: mean precision-recall AUC over stratified 5-fold cross-validation.
Beside it: ROC AUC, a date-only reference model, a time-ordered check, and a
shuffled-label control. Writes results/cv_fold_scores.csv,
results/cv_held_out_scores.csv and figures/pr_curves.png.

Takes about 8 minutes; gradient boosting is nearly all of it.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from secom_ml.check_drift import date_only_model_scores
from secom_ml.cleaning import choose_sensors
from secom_ml.config import FIGURES_DIR, RANDOM_SEED, RESULTS_DIR
from secom_ml.evaluate import Folds, score_folds, stratified_folds, time_ordered_folds
from secom_ml.models import build_models
from secom_ml.plotting import (
    AQUA,
    BLUE,
    INK_SECONDARY,
    MAGENTA,
    MUTED,
    ORANGE,
    YELLOW,
    apply_style,
)
from secom_ml.split import load_train

MODEL_COLORS = {
    "logistic regression": BLUE,
    "random forest": ORANGE,
    "gradient boosting": AQUA,
    "isolation forest": YELLOW,
    "xgboost": MAGENTA,
}

# A PR AUC above this many times the base rate is better than SECOM usually
# allows, so it is treated as something to explain, not as good news.
LEAK_ALARM_LIFT = 3.0


def score_all_models(
    X: pd.DataFrame, y: pd.Series, folds: Folds
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score every model on the same folds.

    Returns (fold_scores, held_out_scores): one row per model and fold, and one
    column of held-out scores per model.
    """
    fold_tables, held_out = [], {}
    for name, model in build_models().items():
        per_fold, held_out[name] = score_folds(model, X, y, folds)
        per_fold.insert(0, "model", name)
        fold_tables.append(per_fold)
    return pd.concat(fold_tables, ignore_index=True), pd.DataFrame(held_out)


def summarise(fold_scores: pd.DataFrame) -> pd.DataFrame:
    """Average the folds for each model, keeping the fold-to-fold spread."""
    lift = fold_scores["pr_auc"] / fold_scores["base_rate"]
    grouped = fold_scores.assign(lift=lift).groupby("model", sort=False)
    return pd.DataFrame(
        {
            "pr_auc": grouped["pr_auc"].mean(),
            "pr_auc_sd": grouped["pr_auc"].std(),
            "times_base_rate": grouped["lift"].mean(),
            "roc_auc": grouped["roc_auc"].mean(),
            "roc_auc_sd": grouped["roc_auc"].std(),
        }
    )


def print_summary(title: str, summary: pd.DataFrame, show_lift: bool = True) -> None:
    """Print mean +/- standard deviation over folds for each model."""
    print(f"\n{title}")
    lift_header = f"{'x base rate':>14}" if show_lift else ""
    print(f"  {'model':<22}{'PR AUC':>18}{lift_header}{'ROC AUC':>18}")
    for name, row in summary.iterrows():
        lift = f"{row.times_base_rate:>13.1f}x" if show_lift else ""
        print(
            f"  {name:<22}"
            f"{row.pr_auc:>9.3f} +/- {row.pr_auc_sd:.3f}"
            f"{lift}"
            f"{row.roc_auc:>9.3f} +/- {row.roc_auc_sd:.3f}"
        )


def print_folds(fold_scores: pd.DataFrame, metric: str) -> None:
    """Show every fold, so the averages cannot hide a single lucky or unlucky fold."""
    table = fold_scores.pivot(index="model", columns="fold", values=metric)
    table = table.loc[list(MODEL_COLORS)]
    fails = fold_scores.drop_duplicates("fold").set_index("fold")["held_out_fails"]
    header = "".join(f"{f'fold {fold} ({fails[fold]} fails)':>19}" for fold in table.columns)
    print(f"  {'':<22}{header}")
    for name, row in table.iterrows():
        print(f"  {name:<22}" + "".join(f"{value:>19.3f}" for value in row))


def shuffled_label_control(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Re-run the comparison with the labels shuffled.

    Shuffling breaks any real link between sensors and failures, so an honest
    pipeline must fall to about the base rate (PR AUC) and 0.5 (ROC AUC). A
    model that still scores well here is learning from something it should not see.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    y_shuffled = pd.Series(rng.permutation(y.to_numpy()), index=y.index, name=y.name)
    fold_scores, _ = score_all_models(X, y_shuffled, stratified_folds(X, y_shuffled))
    return summarise(fold_scores)


def plot_pr_curves(
    y: pd.Series,
    held_out_scores: pd.DataFrame,
    fold_scores: pd.DataFrame,
    summary: pd.DataFrame,
    date_only_pr_auc: float,
) -> None:
    apply_style()
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.25, 1]}
    )
    base_rate = y.mean()

    # Left: one curve per model, from the scores each run got while held out.
    for name, color in MODEL_COLORS.items():
        precision, recall, _ = precision_recall_curve(y, held_out_scores[name])
        # The last point scikit-learn returns is a made-up (recall 0, precision 1)
        # anchor, not a real threshold, so it is left off the chart.
        left.plot(
            recall[:-1], precision[:-1], color=color, drawstyle="steps-post", linewidth=1.6,
            linestyle="--" if name == "isolation forest" else "-",
            label=f"{name}  (PR AUC {summary.loc[name, 'pr_auc']:.2f})",
        )
    left.axhline(base_rate, color=MUTED, linewidth=1, linestyle=":")
    left.text(0.01, base_rate - 0.008, f"no skill = fail rate {base_rate:.1%}", va="top",
              ha="left", color=INK_SECONDARY, fontsize=9)
    left.set_xlim(0, 1)
    left.set_ylim(0, 0.8)
    left.set_xlabel("recall: share of failing runs caught")
    left.set_ylabel("precision: share of alarms that are real failures")
    left.set_title("Precision-recall curves, held-out training runs", pad=18)
    left.legend(frameon=False, loc="upper right", fontsize=9)

    # Right: every fold as a dot, so the noise in each average is visible.
    names = list(MODEL_COLORS)
    for position, name in enumerate(names):
        values = fold_scores.loc[fold_scores["model"] == name, "pr_auc"]
        right.scatter(values, [position] * len(values), color=MODEL_COLORS[name], s=45,
                      edgecolor=plt.rcParams["axes.facecolor"], linewidth=1, zorder=3)
        right.plot([values.mean()] * 2, [position - 0.28, position + 0.28], color=INK_SECONDARY,
                   linewidth=2, zorder=2)
    # Two reference lines: knowing nothing, and knowing only the date of the run.
    for value, style, label in [(base_rate, ":", "no skill"), (date_only_pr_auc, "--", "date only")]:
        right.axvline(value, color=MUTED, linewidth=1, linestyle=style)
        right.text(value, 1.01, label, transform=right.get_xaxis_transform(), ha="center",
                   va="bottom", color=INK_SECONDARY, fontsize=8)
    right.set_yticks(range(len(names)), names)
    right.invert_yaxis()
    right.set_xlim(0, None)
    right.grid(axis="y", visible=False)
    right.set_xlabel("PR AUC (dot = one fold, bar = mean of the five folds)")
    right.set_title("Folds disagree as much as models do", pad=18)

    fig.suptitle(
        "Stratified 5-fold cross-validation on the training runs. The test period is not used.",
        x=0.01, ha="left", fontsize=9, color=INK_SECONDARY,
    )
    fig.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / "pr_curves.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"\nSaved {path.relative_to(FIGURES_DIR.parent)}")


def main() -> None:
    X_train, y, timestamp = load_train()
    X = X_train[choose_sensors(X_train)]
    print(f"Training runs: {len(X)}, sensors: {X.shape[1]}, "
          f"failing runs: {y.sum()} ({y.mean():.1%})")

    # 1. The ranking: stratified 5-fold.
    stratified_scores, held_out_scores = score_all_models(X, y, stratified_folds(X, y))
    summary = summarise(stratified_scores)
    print_summary("=== RANKING: stratified 5-fold (mean +/- standard deviation over folds) ===",
                  summary)
    date_only = date_only_model_scores(y)
    print(f"  {'(date only, reference)':<22}{date_only['pr_auc'].mean():>9.3f}"
          f"{'':>23}{date_only['roc_auc'].mean():>9.3f}")
    print("  The reference model sees no sensors, only where a run falls in time.")
    print("\n  PR AUC in each fold:")
    print_folds(stratified_scores, "pr_auc")

    ranking = summary["pr_auc"].sort_values(ascending=False)
    print(f"\n  Ranked by PR AUC: {' > '.join(ranking.index)}")
    worst_case = summary["times_base_rate"].max()
    if worst_case > LEAK_ALARM_LIFT:
        print(f"  WARNING: a model reaches {worst_case:.1f}x the base rate, over the "
              f"{LEAK_ALARM_LIFT:.0f}x alarm level. Do not trust this number until "
              f"check_drift.py and the checks below explain it.")
    else:
        print(f"  Leak alarm: best model is {worst_case:.1f}x the base rate, "
              f"under the {LEAK_ALARM_LIFT:.0f}x alarm level.")

    # 2. The check: same models, time-ordered folds.
    time_scores, _ = score_all_models(X, y, time_ordered_folds(X))
    print_summary("=== CHECK: time-ordered folds (fit on the past, score the next block) ===",
                  summarise(time_scores), show_lift=False)
    print("\n  PR AUC in each time fold (compare with that fold's base rate):")
    print_folds(time_scores, "pr_auc")
    base_rates = time_scores.drop_duplicates("fold").set_index("fold")["base_rate"]
    print(f"  {'fold base rate':<22}" + "".join(f"{rate:>19.3f}" for rate in base_rates))
    print("\n  ROC AUC in each time fold (0.5 = no skill, whatever the base rate):")
    print_folds(time_scores, "roc_auc")

    # 3. The control: shuffled labels.
    print_summary("=== CONTROL: shuffled labels (expect no skill) ===",
                  shuffled_label_control(X, y))
    print("  No skill here means ROC AUC near 0.5 and PR AUC a little ABOVE the base rate:")
    print("  with only 17 failing runs in a fold, a random ranking averages about 1.2x.")

    RESULTS_DIR.mkdir(exist_ok=True)
    all_fold_scores = pd.concat(
        [stratified_scores.assign(scheme="stratified"), time_scores.assign(scheme="time_ordered")],
        ignore_index=True,
    )
    all_fold_scores.to_csv(RESULTS_DIR / "cv_fold_scores.csv", index=False)
    held_out_scores.assign(fail=y, timestamp=timestamp).to_csv(
        RESULTS_DIR / "cv_held_out_scores.csv", index_label="run"
    )
    print(f"\nSaved {RESULTS_DIR.name}/cv_fold_scores.csv and "
          f"{RESULTS_DIR.name}/cv_held_out_scores.csv")

    plot_pr_curves(y, held_out_scores, stratified_scores, summary, date_only["pr_auc"].mean())


if __name__ == "__main__":
    main()
