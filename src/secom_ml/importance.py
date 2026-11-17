"""Phase 6: primary-model permutation importance and fold stability, training only.

Run with: python -m secom_ml.importance

Protocol fixed before running: average-precision loss after shuffling each of the
kept sensors, five shuffles per sensor, in both the five shuffled folds and the
four forward folds. Each model, including its calibrator and imputer, is fitted
only on the fitting rows of that fold. Nothing here selects features or changes
the final model. The fixed sensor list follows the earlier training-wide cleaning
rule; it is not re-selected using any importance results.

Positive importance means the held-out score worsened after shuffling. Negative
values are retained, since they show noise or a feature hurting generalisation.
These are predictive associations, not causes. Correlated sensors can substitute
for each other, and shuffling one breaks their normal relationships. Five repeats
measure shuffle variability; folds measure sensitivity to samples and time. They
are neither independent studies nor confidence intervals.
"""

from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from secom_ml.cleaning import choose_sensors
from secom_ml.config import FIGURES_DIR, RANDOM_SEED, RESULTS_DIR
from secom_ml.evaluate import Folds, stratified_folds, time_ordered_folds
from secom_ml.models import PRIMARY_MODEL, build_final_models, failure_scores
from secom_ml.plotting import BLUE, INK_SECONDARY, MUTED, ORANGE, apply_style
from secom_ml.split import load_train

N_REPEATS = 5
TOP_K = 10


def score_importance_folds(
    model, X: pd.DataFrame, y: pd.Series, folds: Folds, *,
    n_repeats: int = N_REPEATS, view: str = "shuffled folds",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one importance row per sensor/fold and the held-out fold scores.

    One job and one OpenMP thread during permutation scoring avoid launching a
    large thread pool for every small prediction. Fitting uses the existing
    model configuration unchanged. Permutations act on raw held-out columns,
    including missing entries, before the already-fitted pipeline imputes them.
    """
    importance_tables, fold_rows = [], []
    for number, (fit_rows, held_out_rows) in enumerate(folds, start=1):
        if np.intersect1d(fit_rows, held_out_rows).size:
            raise ValueError("fitting and held-out rows must be disjoint")
        X_held_out, y_held_out = X.iloc[held_out_rows], y.iloc[held_out_rows]
        if y_held_out.nunique() != 2:
            raise ValueError("importance requires both classes in each held-out fold")
        print(f"{view}: fitting fold {number}/{len(folds)}", flush=True)
        fitted = clone(model).fit(X.iloc[fit_rows], y.iloc[fit_rows])
        # XGBoost's explicit n_jobs=-1 can override the OpenMP limit. Change
        # only prediction threading on its already-fitted trees; the training
        # settings, learned model, calibrator, and predictions stay the same.
        for calibrated in getattr(fitted, "calibrated_classifiers_", []):
            estimator = calibrated.estimator
            if hasattr(estimator, "named_steps") and "model" in estimator.named_steps:
                predictor = estimator.named_steps["model"]
                if hasattr(predictor, "get_booster"):
                    predictor.set_params(n_jobs=1)
        baseline = average_precision_score(y_held_out, failure_scores(fitted, X_held_out))
        with threadpool_limits(limits=1):
            importance = permutation_importance(
                fitted, X_held_out, y_held_out, scoring="average_precision",
                n_repeats=n_repeats, random_state=RANDOM_SEED + number, n_jobs=1,
            )
        table = pd.DataFrame({
            "sensor": X.columns,
            "importance_mean": importance.importances_mean,
            "shuffle_sd": importance.importances_std,
        })
        for repeat in range(n_repeats):
            table[f"drop_repeat_{repeat + 1}"] = importance.importances[:, repeat]
        # Average ranks keep ties honest. Top-k uses a deterministic sensor-name
        # tie break, and excludes nonpositive scores instead of promoting zeros.
        table["rank"] = table["importance_mean"].rank(ascending=False, method="average")
        positive = table[table["importance_mean"] > 0].sort_values(
            ["importance_mean", "sensor"], ascending=[False, True]
        )
        table["top_10"] = table["sensor"].isin(positive.head(TOP_K)["sensor"])
        importance_tables.append(table.assign(view=view, fold=number))
        fold_rows.append({
            "view": view, "fold": number, "fit_runs": len(fit_rows),
            "held_out_runs": len(held_out_rows), "held_out_fails": int(y_held_out.sum()),
            "base_rate": y_held_out.mean(), "pr_auc": baseline, "repeats": n_repeats,
        })
        print(f"{view}: fold {number} AP {baseline:.4f}; permutations complete", flush=True)
    return pd.concat(importance_tables, ignore_index=True), pd.DataFrame(fold_rows)


def summarise_importance(per_fold: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight fold summary; fold SD is descriptive, not a confidence interval."""
    return per_fold.groupby(["view", "sensor"], sort=False).agg(
        mean_drop=("importance_mean", "mean"),
        fold_sd=("importance_mean", "std"),
        min_drop=("importance_mean", "min"),
        max_drop=("importance_mean", "max"),
        mean_shuffle_sd=("shuffle_sd", "mean"),
        positive_folds=("importance_mean", lambda values: int((values > 0).sum())),
        top_10_folds=("top_10", "sum"),
        mean_rank=("rank", "mean"),
        rank_sd=("rank", "std"),
        folds=("fold", "size"),
    ).reset_index().sort_values(["view", "mean_drop"], ascending=[True, False])


def pairwise_stability(per_fold: pd.DataFrame) -> pd.DataFrame:
    """Compare all sensor ranks and positive top-10 sets across every pair of folds."""
    rows = []
    for view, table in per_fold.groupby("view", sort=False):
        for left, right in combinations(table["fold"].unique(), 2):
            a = table[table["fold"] == left].set_index("sensor")
            b = table[table["fold"] == right].set_index("sensor")
            a_top, b_top = set(a.index[a["top_10"]]), set(b.index[b["top_10"]])
            union = a_top | b_top
            rows.append({
                "view": view, "fold_a": left, "fold_b": right,
                "rank_spearman": a["rank"].corr(b["rank"]),
                "top_10_overlap": len(a_top & b_top),
                "top_10_jaccard": len(a_top & b_top) / len(union) if union else np.nan,
            })
    return pd.DataFrame(rows)


def sensor_correlations(X: pd.DataFrame) -> pd.DataFrame:
    """Training-only pairwise Pearson context, using observed pairs (no imputation)."""
    correlation = X.corr().abs().mask(np.eye(X.shape[1], dtype=bool))
    return pd.DataFrame({
        "sensor": X.columns,
        "nearest_sensor": correlation.idxmax().values,
        "nearest_abs_correlation": correlation.max().values,
        "correlated_partners_ge_0_95": (correlation >= 0.95).sum().values,
    })


def plot_importance(summary: pd.DataFrame, per_fold: pd.DataFrame) -> None:
    """The top training signals, with every fold visible rather than only a mean."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 6.2))
    for ax, (view, color) in zip(axes, [("shuffled folds", BLUE), ("forward in time", ORANGE)]):
        leaders = summary[summary["view"] == view].nlargest(TOP_K, "mean_drop").iloc[::-1]
        positions = np.arange(len(leaders))
        ax.errorbar(
            leaders["mean_drop"], positions, xerr=leaders["fold_sd"], fmt="o",
            color=color, capsize=3, elinewidth=1.4, markersize=6, label="mean +/- fold SD",
        )
        for pos, sensor in zip(positions, leaders["sensor"]):
            points = per_fold[(per_fold["view"] == view) & (per_fold["sensor"] == sensor)]
            ax.scatter(points["importance_mean"], np.full(len(points), pos),
                       s=20, color=MUTED, alpha=0.75, marker="|", zorder=3)
        labels = [f"{row.sensor}   ({row.top_10_folds}/{row.folds})"
                  for row in leaders.itertuples()]
        ax.set_yticks(positions, labels)
        ax.set_ylim(-0.75, len(leaders) - 0.25)
        ax.axvline(0, linewidth=1, color=MUTED, linestyle=":")
        ax.grid(axis="y", visible=False)
        ax.set_title(view)
        ax.set_xlabel("Drop in held-out average precision after shuffling")
    axes[0].set_ylabel("Sensor (number of folds in the positive top 10)")
    fig.suptitle("Calibrated XGBoost: permutation importance on training folds",
                 x=0.02, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.02, 0.02,
             "Five shuffles per sensor per fold. Dots: fold means averaged; bars: fold SD; "
             "ticks: individual folds. These are not confidence intervals.\n"
             "Each panel selects its own top 10. Correlated sensors share signal; "
             "importance is predictive association, not a cause of failures.",
             fontsize=9, color=INK_SECONDARY)
    fig.tight_layout(rect=(0, 0.10, 1, 0.95))
    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / "importance_stability.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}", flush=True)


def main() -> None:
    X_train, y, _ = load_train()
    X = X_train[choose_sensors(X_train)]
    model = build_final_models()[PRIMARY_MODEL]
    print(f"{PRIMARY_MODEL}; {len(X)} training runs, {int(y.sum())} failures, "
          f"{X.shape[1]} sensors, {N_REPEATS} permutations per sensor/fold.", flush=True)
    tables, scores = [], []
    for view, folds in [("shuffled folds", stratified_folds(X, y)),
                        ("forward in time", time_ordered_folds(X))]:
        table, score = score_importance_folds(model, X, y, folds, view=view)
        tables.append(table)
        scores.append(score)
    per_fold = pd.concat(tables, ignore_index=True)
    summary = summarise_importance(per_fold).merge(sensor_correlations(X), on="sensor")
    stability = pairwise_stability(per_fold)
    RESULTS_DIR.mkdir(exist_ok=True)
    for name, table in {
        "importance_per_fold": per_fold,
        "importance_summary": summary,
        "importance_fold_scores": pd.concat(scores, ignore_index=True),
        "importance_pairwise_stability": stability,
    }.items():
        table.to_csv(RESULTS_DIR / f"{name}.csv", index=False)
    for view in per_fold["view"].unique():
        print(f"\n{view}: top sensors (AP drop; positive folds; top-10 folds)")
        for row in summary[summary["view"] == view].nlargest(TOP_K, "mean_drop").itertuples():
            print(f"  {row.sensor}: {row.mean_drop:+.4f}; "
                  f"{row.positive_folds}/{row.folds}; {row.top_10_folds}/{row.folds}")
        pairs = stability[stability["view"] == view]
        print(f"  Mean pairwise rank correlation: {pairs['rank_spearman'].mean():.3f}; "
              f"mean top-10 overlap: {pairs['top_10_overlap'].mean():.1f}")
    plot_importance(summary, per_fold)
    print("No sensors were removed or added based on these diagnostics.")


if __name__ == "__main__":
    main()
