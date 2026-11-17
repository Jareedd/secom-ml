"""Phase 4: are the probabilities honest, and does calibration help?

Run with:  python -m secom_ml.calibrate   (about 5 minutes)

"Calibrated" means: of the runs given a 10% chance of failing, about 10% fail.
The cost-based alarm in phase 5 compares a probability with a threshold, so it
only works if the probabilities mean what they say.

Both models that go to the final test are checked. For each, two versions (the
model as trained, and the same model plus a sigmoid calibrator) are scored on
training runs they did not see. Writes, for each model, a held-out table and a
forward-check table in results/ and a chart in figures/.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import brier_score_loss

from secom_ml.cleaning import choose_sensors
from secom_ml.config import FIGURES_DIR, RESULTS_DIR
from secom_ml.evaluate import score_folds, stratified_folds, time_ordered_folds
from secom_ml.models import build_calibrated_model, build_models
from secom_ml.plotting import BLUE, INK_SECONDARY, MUTED, ORANGE, apply_style
from secom_ml.split import load_train

MODELS_TO_CHECK = ["xgboost", "random forest"]
VERSIONS = {"uncalibrated": BLUE, "calibrated": ORANGE}
N_BINS = 8


def wilson_interval(fails: int, runs: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a fail rate measured on a small group of runs."""
    rate = fails / runs
    centre = (rate + z**2 / (2 * runs)) / (1 + z**2 / runs)
    half_width = z * np.sqrt(rate * (1 - rate) / runs + z**2 / (4 * runs**2)) / (1 + z**2 / runs)
    return centre - half_width, centre + half_width


def reliability_table(y: pd.Series, probability: pd.Series) -> pd.DataFrame:
    """Group runs into equal-sized bins by predicted probability and compare with reality."""
    runs_table = pd.DataFrame({"fail": y, "probability": probability})
    bins = pd.qcut(probability, q=N_BINS, duplicates="drop")
    rows = []
    for _, in_bin in runs_table.groupby(bins, observed=True):
        fails, runs = int(in_bin["fail"].sum()), len(in_bin)
        low, high = wilson_interval(fails, runs)
        rows.append(
            {
                "runs": runs,
                "fails": fails,
                "predicted": in_bin["probability"].mean(),
                "actual": fails / runs,
                "actual_low": low,
                "actual_high": high,
            }
        )
    return pd.DataFrame(rows)


def bins_consistent(table: pd.DataFrame) -> int:
    """Count the bins whose prediction lies inside the 95% interval of what happened."""
    inside = (table["predicted"] >= table["actual_low"]) & (
        table["predicted"] <= table["actual_high"]
    )
    return int(inside.sum())


def brier_report(y: pd.Series, probability: pd.Series) -> dict:
    """Brier score = average squared gap between probability and outcome (lower is better)."""
    brier = brier_score_loss(y, probability)
    # The reference: say "the base rate" for every run. A model has to beat this
    # to show that its probabilities carry information about individual runs.
    reference = brier_score_loss(y, np.full(len(y), y.mean()))
    return {
        "brier": brier,
        "reference": reference,
        "skill": 1 - brier / reference,
        "mean_predicted": probability.mean(),
        "actual_rate": y.mean(),
    }


def both_versions(model_name: str) -> dict:
    """Return the model as trained and the same model with a sigmoid calibrator."""
    return {
        "uncalibrated": build_models()[model_name],
        "calibrated": build_calibrated_model(model_name),
    }


def forward_check(X: pd.DataFrame, y: pd.Series, model_name: str) -> pd.DataFrame:
    """Fit on the past, predict the next block, and compare predicted with actual fail rates.

    This asks the question the cross-validated numbers cannot: when the fail
    rate drifts, do probabilities learned from the past still fit the future?
    """
    folds = time_ordered_folds(X)
    models = both_versions(model_name)
    scores = {name: score_folds(model, X, y, folds)[1] for name, model in models.items()}

    rows = []
    for number, (fit_rows, held_out_rows) in enumerate(folds, start=1):
        y_next = y.iloc[held_out_rows]
        past_rate = y.iloc[fit_rows].mean()
        row = {
            "block": number,
            "fails": int(y_next.sum()),
            "past_rate": past_rate,
            "actual_rate": y_next.mean(),
            "brier_past_rate": brier_score_loss(y_next, np.full(len(y_next), past_rate)),
        }
        for name in models:
            predicted = scores[name].iloc[held_out_rows]
            row[f"predicted, {name}"] = predicted.mean()
            row[f"brier, {name}"] = brier_score_loss(y_next, predicted)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_calibration(
    tables: dict[str, pd.DataFrame], forward: pd.DataFrame, model_name: str
) -> None:
    apply_style()
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # Left: reliability. Points on the diagonal are perfectly calibrated.
    limit = max(table[["predicted", "actual_high"]].max().max() for table in tables.values()) * 1.05
    left.plot([0, limit], [0, limit], color=MUTED, linewidth=1, linestyle=":")
    left.text(limit * 0.97, limit * 0.93, "perfect", color=INK_SECONDARY, fontsize=9,
              ha="right", va="top")
    for name, color in VERSIONS.items():
        table = tables[name]
        left.errorbar(
            table["predicted"], table["actual"],
            yerr=[table["actual"] - table["actual_low"], table["actual_high"] - table["actual"]],
            color=color, marker="o", markersize=6, linewidth=1.6, elinewidth=1, capsize=2,
            label=name,
        )
    left.set_xlim(0, limit)
    left.set_ylim(0, limit)
    left.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    left.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    left.set_xlabel("predicted probability of failing (bin average)")
    left.set_ylabel("share of runs that actually failed")
    left.set_title(f"{model_name}: reliability on shuffled folds", pad=18)
    left.legend(frameon=False, loc="upper left", fontsize=9)

    # Right: forward in time. Bars are what happened; markers are what was predicted.
    positions = np.arange(len(forward))
    right.bar(positions, forward["actual_rate"] * 100, color="#c3c2b7", width=0.55,
              label="actual fail rate")
    for offset, (name, color) in zip([-0.12, 0.12], VERSIONS.items()):
        right.scatter(positions + offset, forward[f"predicted, {name}"] * 100, color=color,
                      s=55, zorder=3, edgecolor=plt.rcParams["axes.facecolor"], linewidth=1,
                      label=f"predicted, {name}")
    right.set_xticks(positions, [f"block {row.block}\n({row.fails} fails)"
                                 for row in forward.itertuples()])
    right.set_ylabel("failing runs (%)")
    right.set_xlabel("next block of runs, predicted by a model fit on all earlier runs")
    right.set_title(f"{model_name}: predicted vs actual, forward in time", pad=18)
    right.grid(axis="x", visible=False)
    right.legend(frameon=False, fontsize=9)

    fig.suptitle(
        "Training runs only. Bars on the left chart are 95% intervals: each point is about "
        "157 runs.",
        x=0.01, ha="left", fontsize=9, color=INK_SECONDARY,
    )
    fig.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / f"calibration_{model_name.replace(' ', '_')}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path.relative_to(FIGURES_DIR.parent)}")


def check_model(X: pd.DataFrame, y: pd.Series, timestamp: pd.Series, model_name: str) -> None:
    """Run every calibration check for one model and save its tables and chart."""
    folds = stratified_folds(X, y)
    models = both_versions(model_name)
    held_out, per_fold_brier = {}, {}
    for name, model in models.items():
        _, held_out[name] = score_folds(model, X, y, folds)
        per_fold_brier[name] = [
            brier_score_loss(y.iloc[rows], held_out[name].iloc[rows]) for _, rows in folds
        ]

    print(f"\n\n########## {model_name.upper()} ##########")
    print("\n=== HELD-OUT PROBABILITIES (stratified 5-fold, training runs) ===")
    reports = {name: brier_report(y, held_out[name]) for name in models}
    tables = {name: reliability_table(y, held_out[name]) for name in models}
    print(f"  Brier score if every run is given the base rate: "
          f"{reports['uncalibrated']['reference']:.4f}")
    for name, report in reports.items():
        print(f"  {name:<14} Brier {report['brier']:.4f}  "
              f"(skill {report['skill']:+.1%} vs base rate)   "
              f"average predicted {report['mean_predicted']:.1%} "
              f"vs actual {report['actual_rate']:.1%}   "
              f"bins consistent with reality: {bins_consistent(tables[name])} of {N_BINS}")
    for name in models:
        print(f"  {name:<14} Brier by fold: "
              + "  ".join(f"{value:.4f}" for value in per_fold_brier[name]))
    print("  Probability range:  "
          + "   ".join(f"{name} {held_out[name].min():.3f} to {held_out[name].max():.3f}"
                       for name in models))

    for name, table in tables.items():
        print(f"\n  Reliability, {name} ({N_BINS} equal-sized bins, lowest to highest score)")
        print(f"    {'runs':>5}{'fails':>7}{'predicted':>11}{'actual':>9}   95% interval")
        for row in table.itertuples():
            print(f"    {row.runs:>5}{row.fails:>7}{row.predicted:>11.1%}{row.actual:>9.1%}"
                  f"   {row.actual_low:.1%} to {row.actual_high:.1%}")

    print("\n=== FORWARD IN TIME (fit on the past, predict the next block) ===")
    forward = forward_check(X, y, model_name)
    print(f"  {'block':<8}{'fails':>6}{'past rate':>11}{'actual':>9}"
          f"{'uncalib.':>9}{'calibrated':>12}   Brier: past rate / uncalibrated / calibrated")
    for _, row in forward.iterrows():
        print(f"  {row['block']:<8.0f}{row['fails']:>6.0f}{row['past_rate']:>11.1%}"
              f"{row['actual_rate']:>9.1%}{row['predicted, uncalibrated']:>9.1%}"
              f"{row['predicted, calibrated']:>12.1%}   "
              f"{row['brier_past_rate']:.4f} / {row['brier, uncalibrated']:.4f} / "
              f"{row['brier, calibrated']:.4f}")

    slug = model_name.replace(" ", "_")
    RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame(held_out).assign(fail=y, timestamp=timestamp).to_csv(
        RESULTS_DIR / f"calibration_held_out_{slug}.csv", index_label="run"
    )
    forward.to_csv(RESULTS_DIR / f"calibration_forward_check_{slug}.csv", index=False)
    print(f"\nSaved {RESULTS_DIR.name}/calibration_held_out_{slug}.csv and "
          f"{RESULTS_DIR.name}/calibration_forward_check_{slug}.csv")
    plot_calibration(tables, forward, model_name)


def main() -> None:
    X_train, y, timestamp = load_train()
    X = X_train[choose_sensors(X_train)]
    for model_name in MODELS_TO_CHECK:
        check_model(X, y, timestamp, model_name)
    print("\nWhich version of each model goes to the final test is recorded, with the")
    print("reasons, in build_final_models() in models.py.")


if __name__ == "__main__":
    main()
