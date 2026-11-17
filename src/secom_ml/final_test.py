"""Phase 6: one frozen evaluation of both declared models on the latest runs.

Run once: python -m secom_ml.final_test
Rebuild tables/figure from saved predictions: python -m secom_ml.final_test --report-only

The manifest is created before the holdout is opened. An existing manifest blocks
another fit/evaluation, including after an interrupted run. Reporting never refits,
selects a threshold, or reads raw data. Bootstrap intervals condition on fitted
models; a circular block bootstrap is a sensitivity check for serial dependence.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)

from secom_ml.calibrate import wilson_interval
from secom_ml.cleaning import choose_sensors
from secom_ml.config import (
    COST_FALSE_ALARM,
    COST_RATIOS,
    FIGURES_DIR,
    HEADLINE_COST_RATIO,
    N_BOOTSTRAP,
    RANDOM_SEED,
    RESULTS_DIR,
)
from secom_ml.cost import policy_summary
from secom_ml.models import PRIMARY_MODEL, SECONDARY_MODEL, build_final_models, failure_scores
from secom_ml.plotting import BLUE, INK_SECONDARY, MUTED, ORANGE, apply_style
from secom_ml.split import load_test, load_train

# Fixed before the first final evaluation, never chosen by the results.
BLOCK_LENGTH = 20
RELIABILITY_BINS = 5
MANIFEST_FILE = "final_test_manifest.json"
PREDICTIONS_FILE = "final_test_predictions.csv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_scores(y: np.ndarray, probability: np.ndarray) -> None:
    if y.ndim != 1 or probability.shape != y.shape or len(y) == 0:
        raise ValueError("outcomes and probabilities must be matching nonempty vectors")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("evaluation requires both pass and fail outcomes")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("probabilities must be finite and between 0 and 1")


def bootstrap_rows(
    y: np.ndarray,
    *,
    repeats: int,
    seed: int,
    block_length: int = 1,
) -> np.ndarray:
    """Paired row resampling, or circular blocks of consecutive time-ordered rows.

    Drop resamples missing either class (AP/ROC and relative costs need both).
    A circular block may join the end to the beginning of the observed period.
    Its length is a sensitivity assumption, not an estimated dependence horizon.
    """
    y = np.asarray(y)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("bootstrap requires both classes")
    if repeats < 1 or not 1 <= block_length <= len(y):
        raise ValueError("invalid bootstrap repeats or block length")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(y), size=(repeats, int(np.ceil(len(y) / block_length))))
    rows = ((starts[:, :, None] + np.arange(block_length)) % len(y)).reshape(repeats, -1)
    rows = rows[:, : len(y)]
    fails = y[rows].sum(axis=1)
    rows = rows[(fails > 0) & (fails < len(y))]
    if len(rows) == 0:
        raise ValueError("no bootstrap sample contains both classes")
    return rows


def metrics(y: np.ndarray, probability: np.ndarray, train_rate: float) -> dict:
    validate_scores(y, probability)
    return {
        "pr_auc": average_precision_score(y, probability),
        "roc_auc": roc_auc_score(y, probability),
        "brier": brier_score_loss(y, probability),
        "mean_probability": probability.mean(),
        "actual_rate": y.mean(),
        "brier_train_rate": brier_score_loss(y, np.full(len(y), train_rate)),
        # Retrospective reference only: test prevalence is not a deployed forecast.
        "brier_test_rate_reference": y.mean() * (1 - y.mean()),
    }


def interval(values: np.ndarray) -> tuple[float, float]:
    return tuple(float(v) for v in np.percentile(values, [2.5, 97.5]))


def evaluate_predictions(predictions: pd.DataFrame, manifest: dict) -> tuple[pd.DataFrame, ...]:
    """All estimates use saved probabilities and the saved pre-test assumptions."""
    metric_rows, cost_rows, reliability_rows = [], [], []
    for model_name in manifest["models"]:
        view = predictions.loc[predictions["model"] == model_name].sort_values("run")
        y, p = view["fail"].to_numpy(), view["probability"].to_numpy()
        validate_scores(y, p)
        role = "primary" if model_name == manifest["primary_model"] else "secondary"
        resamples = {
            "iid": bootstrap_rows(y, repeats=manifest["bootstrap_repeats"], seed=manifest["seed"]),
            "block": bootstrap_rows(
                y,
                repeats=manifest["bootstrap_repeats"],
                seed=manifest["seed"],
                block_length=manifest["block_length"],
            ),
        }
        row = {
            "model": model_name,
            "role": role,
            "runs": len(y),
            "fails": int(y.sum()),
            **metrics(y, p, manifest["train_rate"]),
        }
        for method, rows in resamples.items():
            row[f"{method}_valid_resamples"] = len(rows)
            for name, scorer in [
                ("pr_auc", average_precision_score),
                ("roc_auc", roc_auc_score),
                ("brier", brier_score_loss),
            ]:
                low, high = interval(np.array([scorer(y[idx], p[idx]) for idx in rows]))
                row[f"{name}_{method}_low"], row[f"{name}_{method}_high"] = low, high
        metric_rows.append(row)

        for ratio in manifest["cost_ratios"]:
            cfa = manifest["cost_false_alarm"]
            summary = policy_summary(y, p, cfa, ratio * cfa)
            alarm = p > summary["threshold"]
            summary.update(
                {
                    "model": model_name,
                    "role": role,
                    "ratio": ratio,
                    "runs": len(y),
                    "fails": int(y.sum()),
                    "misses": int((~alarm & (y == 1)).sum()),
                    "true_negatives": int((~alarm & (y == 0)).sum()),
                    "precision": summary["caught"] / summary["alarms"] if alarm.any() else np.nan,
                    "best_naive": (
                        "never alarm"
                        if summary["cost_never_alarm"] <= summary["cost_inspect_all"]
                        else "inspect every run"
                    ),
                }
            )
            for method, rows in resamples.items():
                yy, aa = y[rows].astype(bool), alarm[rows]
                policy = cfa * ((aa & ~yy).sum(axis=1) + ratio * (~aa & yy).sum(axis=1))
                never = cfa * ratio * yy.sum(axis=1)
                inspect = cfa * (~yy).sum(axis=1)
                values = {
                    "saving_vs_best_naive": 1 - policy / np.minimum(never, inspect),
                    "saving_vs_never_alarm": 1 - policy / never,
                    "cost": 1000 * policy / len(y),
                }
                summary[f"{method}_valid_resamples"] = len(rows)
                for key, value in values.items():
                    low, high = interval(value)
                    summary[f"{key}_{method}_low"], summary[f"{key}_{method}_high"] = low, high
            cost_rows.append(summary)

        # qcut keeps tied probabilities together, so fewer than five bins is possible.
        bins = pd.qcut(view["probability"], q=manifest["reliability_bins"], duplicates="drop")
        if bins.isna().all():
            bins = pd.Series(0, index=view.index)  # a constant forecast is one reliability bin
        for number, (_, group) in enumerate(view.groupby(bins, observed=True), start=1):
            low, high = wilson_interval(int(group["fail"].sum()), len(group))
            reliability_rows.append(
                {
                    "model": model_name,
                    "bin": number,
                    "runs": len(group),
                    "fails": int(group["fail"].sum()),
                    "predicted": group["probability"].mean(),
                    "actual": group["fail"].mean(),
                    "actual_low": low,
                    "actual_high": high,
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(cost_rows), pd.DataFrame(reliability_rows)


def run_evaluation(results_dir: Path = RESULTS_DIR) -> None:
    """Fit on training runs, freeze the protocol, then make one prediction per model."""
    results_dir.mkdir(parents=True, exist_ok=True)
    manifest_path, predictions_path = results_dir / MANIFEST_FILE, results_dir / PREDICTIONS_FILE
    if manifest_path.exists() or predictions_path.exists():
        raise FileExistsError(
            "Final evaluation already started. Use --report-only for saved results."
        )
    X_train, y_train, train_time = load_train()
    sensors = choose_sensors(X_train)
    models = build_final_models()
    if list(models) != [PRIMARY_MODEL, SECONDARY_MODEL]:
        raise ValueError("final model roles/order changed")
    manifest = {
        "status": "frozen_before_test",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary_model": PRIMARY_MODEL,
        "secondary_model": SECONDARY_MODEL,
        "models": list(models),
        "model_parameters": {name: repr(m) for name, m in models.items()},
        "headline_cost_ratio": HEADLINE_COST_RATIO,
        "cost_ratios": list(COST_RATIOS),
        "cost_false_alarm": COST_FALSE_ALARM,
        "baseline": "cheaper of never alarm and inspect all",
        "threshold_rule": "p > C_FA / (C_FA + C_miss)",
        "drift_correction": "none",
        "seed": RANDOM_SEED,
        "bootstrap_repeats": N_BOOTSTRAP,
        "block_length": BLOCK_LENGTH,
        "reliability_bins": RELIABILITY_BINS,
        "train_runs": len(y_train),
        "train_fails": int(y_train.sum()),
        "train_rate": float(y_train.mean()),
        "train_start": str(train_time.iloc[0]),
        "train_end": str(train_time.iloc[-1]),
        "sensors": sensors,
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "scikit-learn", "xgboost", "matplotlib"]
        },
        "source_sha256": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
        "data_access_note": "Existing loaders parse raw files before slicing; earlier phases "
        "reported aggregate test counts. No test predictions selected the setup.",
    }
    # Exclusive creation prevents silently overwriting a previous frozen evaluation.
    with manifest_path.open("x", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2)
    fitted = {}
    for name, model in models.items():
        print(f"Fitting {name} on {len(y_train)} training runs...", flush=True)
        fitted[name] = model.fit(X_train[sensors], y_train)

    X_test, y_test, test_time = load_test()  # the single final-evaluation access
    if not (
        X_test.index.equals(y_test.index)
        and X_test.index.equals(test_time.index)
        and X_test.index.is_unique
        and test_time.is_monotonic_increasing
        and train_time.iloc[-1] < test_time.iloc[0]
        and X_train.index.intersection(X_test.index).empty
    ):
        raise ValueError("train/test alignment or chronological separation is invalid")
    frames = []
    for name, model in fitted.items():
        p = failure_scores(model, X_test[sensors])
        validate_scores(y_test.to_numpy(), p)
        frames.append(
            pd.DataFrame(
                {
                    "run": X_test.index,
                    "timestamp": test_time.to_numpy(),
                    "fail": y_test.to_numpy(),
                    "model": name,
                    "role": "primary" if name == PRIMARY_MODEL else "secondary",
                    "probability": p,
                }
            )
        )
    predictions = pd.concat(frames, ignore_index=True)
    temporary = predictions_path.with_suffix(".csv.tmp")
    predictions.to_csv(temporary, index=False)
    temporary.replace(predictions_path)
    manifest.update(
        {
            "status": "predictions_saved",
            "predictions_sha256": sha256(predictions_path),
            "test_runs": len(y_test),
            "test_fails": int(y_test.sum()),
            "test_start": str(test_time.iloc[0]),
            "test_end": str(test_time.iloc[-1]),
            "scored_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(manifest_path)
    print("Saved the frozen manifest and both models' final probabilities.", flush=True)


def read_saved(results_dir: Path) -> tuple[pd.DataFrame, dict]:
    manifest = json.loads((results_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
    path = results_dir / PREDICTIONS_FILE
    if manifest["status"] != "predictions_saved" or not path.exists():
        raise ValueError(
            "final evaluation is incomplete; do not silently refit after opening holdout"
        )
    if sha256(path) != manifest["predictions_sha256"]:
        raise ValueError("saved predictions do not match the frozen checksum")
    predictions = pd.read_csv(path, float_precision="round_trip")
    if predictions.duplicated(["model", "run"]).any():
        raise ValueError("duplicate saved predictions")
    if list(predictions["model"].unique()) != manifest["models"]:
        raise ValueError("saved model names/order do not match the manifest")
    reference = None
    for name in manifest["models"]:
        view = predictions.loc[predictions["model"] == name].sort_values("run")
        if len(view) != manifest["test_runs"] or view["fail"].sum() != manifest["test_fails"]:
            raise ValueError("saved test counts do not match the manifest")
        identity = view[["run", "timestamp", "fail"]].reset_index(drop=True)
        if reference is not None and not identity.equals(reference):
            raise ValueError("models must be evaluated on the same runs and outcomes")
        reference = identity
    return predictions, manifest


def plot_results(predictions, costs, reliability, manifest, figures_dir: Path) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    colors = [BLUE, ORANGE]
    for offset, (name, color) in enumerate(zip(manifest["models"], colors)):
        view = predictions[predictions["model"] == name]
        precision, recall, _ = precision_recall_curve(view["fail"], view["probability"])
        ap = average_precision_score(view["fail"], view["probability"])
        axes[0].step(recall, precision, where="post", color=color, label=f"{name} (AP {ap:.3f})")
        table = reliability[reliability["model"] == name]
        axes[1].errorbar(
            table["predicted"],
            table["actual"],
            yerr=[table["actual"] - table["actual_low"], table["actual_high"] - table["actual"]],
            marker="o",
            color=color,
            capsize=2,
            linewidth=1.5,
            label=name,
        )
        table = costs[costs["model"] == name]
        position = np.arange(len(table)) + (offset - 0.5) * 0.18
        value = table["saving_vs_best_naive"]
        low, high = table["saving_vs_best_naive_iid_low"], table["saving_vs_best_naive_iid_high"]
        # Draw interval and point separately: percentile intervals need not bracket a point estimate.
        axes[2].vlines(position, low, high, color=color, linewidth=1.8)
        axes[2].scatter(position, value, color=color, label=name, zorder=3)
    base_rate = manifest["test_fails"] / manifest["test_runs"]
    axes[0].axhline(base_rate, color=MUTED, linestyle=":", label=f"base rate {base_rate:.1%}")
    axes[0].set(
        xlabel="recall",
        ylabel="precision",
        title="Final holdout: ranking",
        xlim=(0, 1),
        ylim=(0, 1.03),
    )
    axes[0].legend(frameon=False, fontsize=8)
    limit = max(reliability["predicted"].max(), reliability["actual_high"].max()) * 1.08
    axes[1].plot([0, limit], [0, limit], color=MUTED, linestyle=":")
    axes[1].set(
        xlabel="mean predicted probability",
        ylabel="observed fail rate",
        title="Reliability (95% Wilson intervals)",
        xlim=(0, limit),
        ylim=(0, limit),
    )
    axes[2].axhline(0, color=MUTED, linewidth=1)
    axes[2].set_xticks(
        np.arange(len(manifest["cost_ratios"])), [f"{r}:1" for r in manifest["cost_ratios"]]
    )
    axes[2].set(
        xlabel="miss : false-alarm cost",
        ylabel="saving vs cheaper naive policy",
        title="Fixed policies (95% paired IID intervals)",
    )
    axes[2].legend(frameon=False, fontsize=8)
    for ax in axes[:2]:
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    for ax in axes:
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    fig.suptitle(
        f"{manifest['test_runs']} latest runs, {manifest['test_fails']} failures. "
        f"Primary: {manifest['primary_model']}; headline: {manifest['headline_cost_ratio']}:1. "
        "All choices fixed before final predictions.",
        x=0.01,
        ha="left",
        fontsize=10,
        color=INK_SECONDARY,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "final_test.png")
    plt.close(fig)


def write_report(metric_table, costs, manifest, results_dir: Path) -> None:
    headline = costs[
        (costs["model"] == manifest["primary_model"])
        & (costs["ratio"] == manifest["headline_cost_ratio"])
    ].iloc[0]
    lines = [
        "# Phase 6: final chronological holdout",
        "",
        (
            f"Training: {manifest['train_runs']} runs, {manifest['train_fails']} failures "
            f"({manifest['train_start']} through {manifest['train_end']})."
        ),
        (
            f"Test: {manifest['test_runs']} runs, {manifest['test_fails']} failures "
            f"({manifest['test_start']} through {manifest['test_end']})."
        ),
        "",
        (
            f"**Primary {manifest['primary_model']}, {manifest['headline_cost_ratio']}:1:** "
            f"saving {headline.saving_vs_best_naive:+.1%} versus {headline.best_naive}. "
            f"95% paired IID interval {headline.saving_vs_best_naive_iid_low:+.1%} to "
            f"{headline.saving_vs_best_naive_iid_high:+.1%}; "
            f"{manifest['block_length']}-run circular-block sensitivity interval "
            f"{headline.saving_vs_best_naive_block_low:+.1%} to "
            f"{headline.saving_vs_best_naive_block_high:+.1%}."
        ),
        "",
        "## Ranking and probabilities",
        "",
        "| Model | AP (PR AUC) | ROC AUC | Brier | Mean prediction | Actual failure rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in metric_table.itertuples():
        lines.append(
            f"| {r.model} ({r.role}) | {r.pr_auc:.4f} | {r.roc_auc:.4f} | "
            f"{r.brier:.4f} | {r.mean_probability:.1%} | {r.actual_rate:.1%} |"
        )
    lines += [
        "",
        "## Fixed cost policies",
        "",
        (
            "Costs per 1,000 runs, in units of one false-alarm inspection. "
            "Correct decisions cost zero; these costs are assumptions, not fab measurements."
        ),
        "",
        (
            "| Model | Ratio | Threshold | Caught / failures | False alarms | Misses | "
            "Policy cost | Never alarm | 0.5 cutoff | Inspect all | Saving vs best naive (95% IID) |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in costs.itertuples():
        lines.append(
            f"| {r.model} | {r.ratio}:1 | {r.threshold:.2%} | {r.caught}/{r.fails} | "
            f"{r.false_alarms} | {r.misses} | {r.cost:.1f} | {r.cost_never_alarm:.1f} | "
            f"{r.cost_default_cutoff:.1f} | {r.cost_inspect_all:.1f} | "
            f"{r.saving_vs_best_naive:+.1%} ({r.saving_vs_best_naive_iid_low:+.1%} to "
            f"{r.saving_vs_best_naive_iid_high:+.1%}) |"
        )
    lines += [
        "",
        "## Interpretation limits",
        "",
        (
            "- Model roles, costs, thresholds and no drift correction were fixed before final "
            "predictions. No threshold search or promotion of the secondary model is allowed."
        ),
        (
            "- The cheaper naive comparator is recalculated on each bootstrap sample. It is a "
            "retrospective benchmark using observed prevalence, not a deployable oracle."
        ),
        (
            f"- {manifest['bootstrap_repeats']} paired resamples were requested per interval. "
            "Counts retained after excluding single-class resamples are in the CSVs. "
            "Intervals condition on fitted models and this period; they omit training/selection "
            "uncertainty and cannot guarantee performance under future drift."
        ),
        (
            f"- IID resampling assumes independent runs. Circular blocks of "
            f"{manifest['block_length']} consecutive runs give a predeclared dependence sensitivity "
            "check, with an artificial end-to-start join; block length is not estimated."
        ),
        (
            "- Brier references in the metrics CSV distinguish the deployable training "
            "prevalence forecast from the retrospective test-prevalence reference."
        ),
        (
            "- Earlier loaders parsed the complete files before taking the training slice; "
            "aggregate test counts were already reported. Test predictions were first evaluated "
            "in this phase. The shared sensor list uses all training covariates, so earlier CV "
            "is not completely fold-isolated; the chronological final holdout remains excluded "
            "from every fit and sensor choice."
        ),
        "",
        "Regenerate this report without refitting: `python -m secom_ml.final_test --report-only`.",
    ]
    (results_dir / "final_test_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(lines[5], flush=True)


def report_saved(results_dir: Path = RESULTS_DIR, figures_dir: Path = FIGURES_DIR) -> None:
    predictions, manifest = read_saved(results_dir)
    metric_table, costs, reliability = evaluate_predictions(predictions, manifest)
    metric_table.to_csv(results_dir / "final_test_metrics.csv", index=False)
    costs.to_csv(results_dir / "final_test_costs.csv", index=False)
    reliability.to_csv(results_dir / "final_test_reliability.csv", index=False)
    plot_results(predictions, costs, reliability, manifest, figures_dir)
    write_report(metric_table, costs, manifest, results_dir)
    print(metric_table[["model", "pr_auc", "roc_auc", "brier"]].to_string(index=False))
    print("Saved final test tables, report and figures/final_test.png.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-only", action="store_true", help="use saved predictions only")
    args = parser.parse_args()
    if not args.report_only:
        run_evaluation()
    report_saved()


if __name__ == "__main__":
    main()
