"""Phase 5: turn each final model's probabilities into an alarm policy, and price it.

Run with:  python -m secom_ml.cost_policy   (about 3 minutes)

Training runs only, for both final models. Two views of the same policy:
  shuffled folds   - every training run, scored by a model that did not see it.
                     Optimistic, because shuffled folds benefit from the drift.
  forward in time  - each block scored by a model fit on earlier runs only.
                     Closer to how the policy would really be used.
Writes results/cost_policy_training.csv and one cost-versus-threshold chart per
model in figures/.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from secom_ml.cleaning import choose_sensors
from secom_ml.config import COST_FALSE_ALARM, COST_RATIOS, FIGURES_DIR, RESULTS_DIR
from secom_ml.cost import alarm_threshold, bootstrap_interval, cost_curve, policy_summary
from secom_ml.evaluate import score_folds, stratified_folds, time_ordered_folds
from secom_ml.models import PRIMARY_MODEL, build_final_models
from secom_ml.plotting import BLUE, INK, INK_SECONDARY, MUTED, ORANGE, apply_style
from secom_ml.split import load_train

THRESHOLD_GRID = np.linspace(0, 0.40, 401)


def held_out_probabilities(model, X: pd.DataFrame, y: pd.Series) -> dict[str, pd.DataFrame]:
    """Score the training runs two ways and return (fail, probability) tables."""
    views = {}
    for name, folds in [("shuffled folds", stratified_folds(X, y)),
                        ("forward in time", time_ordered_folds(X))]:
        _, probability = score_folds(model, X, y, folds)
        scored = probability.notna()  # the earliest block is never held out in time folds
        views[name] = pd.DataFrame({"fail": y[scored], "probability": probability[scored]})
    return views


def evaluate_ratio(view: pd.DataFrame, ratio: float) -> dict:
    """Everything reported for one cost ratio on one set of scored runs."""
    y, probability = view["fail"].to_numpy(), view["probability"].to_numpy()
    cost_miss = ratio * COST_FALSE_ALARM
    summary = policy_summary(y, probability, COST_FALSE_ALARM, cost_miss)

    for baseline in ["saving_vs_never_alarm", "saving_vs_best_naive"]:
        low, high = bootstrap_interval(
            lambda y_, p_, key=baseline: policy_summary(y_, p_, COST_FALSE_ALARM, cost_miss)[key],
            y, probability,
        )
        summary[f"{baseline}_low"], summary[f"{baseline}_high"] = low, high

    # For comparison only: the threshold that would have been cheapest on these
    # runs. It is never used to make decisions.
    curve = cost_curve(y, probability, THRESHOLD_GRID, COST_FALSE_ALARM, cost_miss)
    summary["data_best_threshold"] = THRESHOLD_GRID[curve.argmin()]
    summary["data_best_cost"] = curve.min()
    return {"ratio": ratio, **summary}


def print_view(name: str, view: pd.DataFrame, table: pd.DataFrame) -> None:
    fails, runs = int(view["fail"].sum()), len(view)
    print(f"\n=== {name.upper()}: {runs} runs, {fails} failing ({fails / runs:.1%}) ===")

    print("  What the policy does")
    print(f"  {'miss:FA':>8}{'threshold':>11}{'alarms':>14}{'failures caught':>19}"
          f"{'false alarms':>18}")
    for row in table.itertuples():
        print(f"  {f'{row.ratio:.0f}:1':>8}{row.threshold:>11.1%}"
              f"{f'{row.alarms} ({row.alarm_rate:.0%})':>14}"
              f"{f'{row.caught} of {fails} ({row.recall:.0%})':>19}"
              f"{f'{row.false_alarms} ({row.false_alarm_rate:.0%})':>18}")

    print("\n  What it costs per 1,000 runs (1 unit = one false-alarm inspection)")
    print(f"  {'miss:FA':>8}{'policy':>9}{'never alarm':>13}{'0.5 cutoff':>12}"
          f"{'inspect all':>13}{'saving vs never alarm':>29}{'saving vs best naive':>29}")
    for row in table.itertuples():
        never = (f"{row.saving_vs_never_alarm:+.0%} "
                 f"({row.saving_vs_never_alarm_low:+.0%} to {row.saving_vs_never_alarm_high:+.0%})")
        naive = (f"{row.saving_vs_best_naive:+.0%} "
                 f"({row.saving_vs_best_naive_low:+.0%} to {row.saving_vs_best_naive_high:+.0%})")
        print(f"  {f'{row.ratio:.0f}:1':>8}{row.cost:>9.0f}{row.cost_never_alarm:>13.0f}"
              f"{row.cost_default_cutoff:>12.0f}{row.cost_inspect_all:>13.0f}"
              f"{never:>29}{naive:>29}")

    print("\n  Check on the formula: the threshold that happened to be cheapest on these runs")
    for row in table.itertuples():
        print(f"  {f'{row.ratio:.0f}:1':>8}  formula {row.threshold:.1%} (cost {row.cost:.0f})   "
              f"cheapest {row.data_best_threshold:.1%} (cost {row.data_best_cost:.0f})")


def plot_cost_curves(
    views: dict[str, pd.DataFrame], tables: dict[str, pd.DataFrame], model_name: str
) -> None:
    """A grid of small charts for one model: one row per view, one column per cost ratio.

    Each chart shows cost against threshold, the two naive policies as flat
    lines, and the formula's threshold as a marked point. The policy is worth
    having only where the marked point sits below both flat lines.
    """
    apply_style()
    fig, axes = plt.subplots(len(views), len(COST_RATIOS), figsize=(14, 6.6), sharex=True)

    for row_axes, (view_name, view) in zip(axes, views.items()):
        y, probability = view["fail"].to_numpy(), view["probability"].to_numpy()
        for ax, row in zip(row_axes, tables[view_name].itertuples()):
            cost_miss = row.ratio * COST_FALSE_ALARM
            curve = cost_curve(y, probability, THRESHOLD_GRID, COST_FALSE_ALARM, cost_miss)
            ax.plot(THRESHOLD_GRID, curve, color=BLUE)

            ax.axhline(row.cost_never_alarm, color=ORANGE, linewidth=1.2, linestyle="--")
            ax.axhline(row.cost_inspect_all, color=INK_SECONDARY, linewidth=1.2, linestyle=":")
            ax.axvline(row.threshold, color=MUTED, linewidth=1)
            ax.scatter([row.threshold], [row.cost], color=BLUE, s=60, zorder=3,
                       edgecolor=plt.rcParams["axes.facecolor"], linewidth=1.2)

            best_naive = min(row.cost_never_alarm, row.cost_inspect_all)
            verdict = f"{row.saving_vs_best_naive:+.0%} vs best naive policy"
            ax.text(0.97, 0.05, f"cost {row.cost:.0f}\n{verdict}", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=8.5, color=INK,
                    bbox={"facecolor": plt.rcParams["axes.facecolor"], "edgecolor": "none",
                          "alpha": 0.85, "pad": 2})
            ax.set_xlim(0, 0.40)
            ax.set_ylim(0, max(curve.max(), best_naive) * 1.08)
            ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
            if ax in axes[0]:
                ax.set_title(f"A miss costs {row.ratio:.0f}x a false alarm")
            if ax in axes[-1]:
                ax.set_xlabel("alarm threshold")
        row_axes[0].set_ylabel(f"{view_name}\ncost per 1,000 runs")

    handles = [
        plt.Line2D([], [], color=BLUE, label="alarm when probability > threshold"),
        plt.Line2D([], [], color=BLUE, marker="o", linestyle="", markersize=7,
                   label="threshold from the cost formula"),
        plt.Line2D([], [], color=ORANGE, linestyle="--", linewidth=1.2, label="never alarm"),
        plt.Line2D([], [], color=INK_SECONDARY, linestyle=":", linewidth=1.2,
                   label="inspect every run"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle(
        f"{model_name}, training runs only. Cost is in units of one false-alarm inspection. "
        "Top row: shuffled folds (optimistic). Bottom row: each block scored by a model "
        "fit on earlier runs.",
        x=0.01, ha="left", fontsize=9, color=INK_SECONDARY,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / f"cost_vs_threshold_{model_name.replace(' ', '_')}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path.relative_to(FIGURES_DIR.parent)}")


def main() -> None:
    X_train, y, _ = load_train()
    X = X_train[choose_sensors(X_train)]

    print("Cost assumptions: a false alarm costs 1 unit; a missed failing run costs")
    print(f"{', '.join(f'{ratio}' for ratio in COST_RATIOS)} units (one result per ratio). "
          "Alarm when p > C_FA / (C_FA + C_miss):")
    for ratio in COST_RATIOS:
        threshold = alarm_threshold(COST_FALSE_ALARM, ratio * COST_FALSE_ALARM)
        print(f"  {ratio}:1 -> alarm when p > {threshold:.1%}")

    all_tables = []
    for model_name, model in build_final_models().items():
        role = "PRIMARY" if model_name == PRIMARY_MODEL else "SECONDARY"
        print(f"\n\n########## {model_name.upper()} ({role}) ##########")
        views = held_out_probabilities(model, X, y)
        tables = {}
        for view_name, view in views.items():
            tables[view_name] = pd.DataFrame(
                [evaluate_ratio(view, ratio) for ratio in COST_RATIOS]
            )
            print_view(view_name, view, tables[view_name])
            all_tables.append(tables[view_name].assign(view=view_name, model=model_name))
        print()
        plot_cost_curves(views, tables, model_name)

    RESULTS_DIR.mkdir(exist_ok=True)
    pd.concat(all_tables).to_csv(RESULTS_DIR / "cost_policy_training.csv", index=False)
    print(f"\nSaved {RESULTS_DIR.name}/cost_policy_training.csv")


if __name__ == "__main__":
    main()
