"""Phase 1: load SECOM and print the facts the later phases depend on.

Run with:  python -m secom_ml.explore

Leakage rule for this script: only structural facts (shape, overall fail rate,
date range) are computed on all runs. Everything about individual sensors, and
everything about how the fail rate moves over time, is computed on the earliest
80% of runs only, because the latest 20% will be the final test set.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from secom_ml.config import FIGURES_DIR, TEST_FRACTION
from secom_ml.data import load_secom, read_raw
from secom_ml.plotting import BLUE, INK_SECONDARY, MUTED, apply_style
from secom_ml.split import load_train

# What I expected before looking, so any difference is called out.
EXPECTED_RUNS = 1567
EXPECTED_SENSORS = 590
EXPECTED_FAIL_RATE = 0.066

MISSING_BUCKETS = ["none", "up to 5%", "5% to 50%", "over 50%"]


def report(name: str, actual: float, expected: float, tolerance: float = 0, fmt: str = "{}") -> None:
    """Print a number next to the value I expected, and say if they differ."""
    verdict = "as expected" if abs(actual - expected) <= tolerance else "DIFFERS from expected"
    print(f"  {name}: {fmt.format(actual)}  ({verdict}: {fmt.format(expected)})")


def describe_all_runs(X: pd.DataFrame, y: pd.Series, timestamp: pd.Series) -> None:
    """Structural facts only. Safe to compute on every run."""
    _, raw_labels = read_raw()

    print("\n=== ALL RUNS (structural facts only) ===")
    report("runs", len(X), EXPECTED_RUNS)
    report("sensors", X.shape[1], EXPECTED_SENSORS)
    report("fail rate", y.mean(), EXPECTED_FAIL_RATE, tolerance=0.0005, fmt="{:.1%}")
    print(f"  failing runs: {y.sum()} of {len(y)}")
    print(f"  accuracy of predicting 'pass' every time: {1 - y.mean():.1%}  (why accuracy is useless here)")
    print(f"  every sensor column is numeric: {all(pd.api.types.is_numeric_dtype(t) for t in X.dtypes)}")
    print(f"  first run: {timestamp.iloc[0]}   last run: {timestamp.iloc[-1]}")
    print(f"  span: {(timestamp.iloc[-1] - timestamp.iloc[0]).days} days")
    print(f"  file was already in time order: {raw_labels['timestamp'].is_monotonic_increasing}")
    print(f"  runs sharing a timestamp with another run: {timestamp.duplicated(keep=False).sum()}")


def missing_bucket_counts(missing_share: pd.Series) -> pd.Series:
    """Count sensors by how much of their data is missing."""
    buckets = pd.cut(
        missing_share,
        bins=[-np.inf, 0, 0.05, 0.50, np.inf],
        labels=MISSING_BUCKETS,
    )
    return buckets.value_counts().reindex(MISSING_BUCKETS)


def fail_rate_by_block(y: pd.Series, timestamp: pd.Series, n_blocks: int = 5) -> pd.DataFrame:
    """Cut the runs into consecutive equal-sized blocks and get each block's fail rate."""
    rows = []
    for block in np.array_split(np.arange(len(y)), n_blocks):
        rows.append(
            {
                "from": timestamp.iloc[block[0]].strftime("%d %b"),
                "to": timestamp.iloc[block[-1]].strftime("%d %b"),
                "runs": len(block),
                "fails": int(y.iloc[block].sum()),
                "fail_rate": y.iloc[block].mean(),
            }
        )
    return pd.DataFrame(rows)


def describe_early_runs(X: pd.DataFrame, y: pd.Series, timestamp: pd.Series) -> tuple[pd.Series, pd.DataFrame]:
    """Sensor-level facts, computed on the earliest 80% of runs only."""
    print(f"\n=== EARLIEST {1 - TEST_FRACTION:.0%} OF RUNS (the future training set) ===")
    print(f"  runs: {len(X)}   from {timestamp.iloc[0]} to {timestamp.iloc[-1]}")
    print(f"  failing runs: {y.sum()}   fail rate: {y.mean():.1%}")

    blocks = fail_rate_by_block(y, timestamp)
    print("\n  Fail rate in 5 consecutive blocks (is the process stable over time?)")
    for row in blocks.itertuples():
        print(f"    {row[1]} to {row.to}: {row.fails:>2} of {row.runs} runs failed = {row.fail_rate:.1%}")

    missing_share = X.isna().mean()
    bucket_counts = missing_bucket_counts(missing_share)
    print(f"\n  Missing values: {X.isna().mean().mean():.1%} of all cells are empty")
    for bucket, count in bucket_counts.items():
        print(f"    sensors with {bucket} missing: {count}")
    missing_per_run = X.isna().sum(axis=1)
    print(f"    empty sensors per run: median {missing_per_run.median():.0f}, max {missing_per_run.max()}")

    is_constant = X.nunique(dropna=True) <= 1
    is_sparse = missing_share > 0.50
    usable = X.loc[:, ~is_constant & ~is_sparse]
    print(f"\n  Sensors that never change (zero variance): {is_constant.sum()}")
    print(f"  Sensors that are both constant and over 50% missing: {(is_constant & is_sparse).sum()}")
    print(f"  Sensors left after dropping both kinds: {usable.shape[1]}")

    spread = usable.std()
    print("\n  Scale of the remaining sensors (standard deviation):")
    print(
        f"    smallest {spread.min():.2g}, median {spread.median():.2g}, "
        f"largest {spread.max():.2g}"
    )

    correlation = usable.corr().abs().to_numpy()
    not_itself = ~np.eye(len(correlation), dtype=bool)  # a sensor always matches itself
    is_twin = (correlation > 0.95) & not_itself
    twin_pairs = int(is_twin.sum() // 2)  # each pair appears twice in the matrix
    sensors_with_twin = int(is_twin.any(axis=0).sum())
    print("\n  Near-duplicate sensors (correlation above 0.95 in absolute value):")
    print(
        f"    {twin_pairs} pairs, involving {sensors_with_twin} "
        f"of the {usable.shape[1]} remaining sensors"
    )

    return bucket_counts, blocks


def plot_overview(bucket_counts: pd.Series, blocks: pd.DataFrame, overall_rate: float) -> None:
    """Two small charts: how much data is missing, and whether the fail rate drifts."""
    apply_style()
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 3.8))

    bars = left.bar(bucket_counts.index, bucket_counts.values, color=BLUE, width=0.6)
    left.bar_label(bars, padding=3, color=INK_SECONDARY)
    left.set_title("Most sensors are nearly complete; a few are mostly empty")
    left.set_xlabel("share of a sensor's readings that are missing")
    left.set_ylabel("number of sensors")
    left.grid(axis="x", visible=False)
    left.margins(y=0.12)

    labels = [f"{row[1]} to\n{row.to}" for row in blocks.itertuples()]
    bars = right.bar(labels, blocks["fail_rate"] * 100, color=BLUE, width=0.6)
    right.bar_label(bars, labels=[f"{r.fails} of {r.runs}" for r in blocks.itertuples()], padding=3, color=INK_SECONDARY)
    right.axhline(overall_rate * 100, color=MUTED, linewidth=1, linestyle="--")
    right.text(
        len(blocks) - 0.5, overall_rate * 100, f" average {overall_rate:.1%}",
        va="bottom", ha="right", color=INK_SECONDARY, fontsize=9,
    )
    right.set_title("The fail rate drifts: early runs fail far more often")
    right.set_xlabel("consecutive blocks of runs, oldest to newest")
    right.set_ylabel("failing runs (%)")
    right.grid(axis="x", visible=False)
    right.margins(y=0.15)

    fig.suptitle(
        f"Earliest {1 - TEST_FRACTION:.0%} of runs only (the test period is not shown)",
        x=0.01, ha="left", fontsize=9, color=INK_SECONDARY,
    )
    fig.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / "explore_overview.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"\nSaved {path.relative_to(FIGURES_DIR.parent)}")


def main() -> None:
    X, y, timestamp = load_secom()
    describe_all_runs(X, y, timestamp)

    X_early, y_early, timestamp_early = load_train()
    bucket_counts, blocks = describe_early_runs(X_early, y_early, timestamp_early)
    plot_overview(bucket_counts, blocks, y_early.mean())


if __name__ == "__main__":
    main()
