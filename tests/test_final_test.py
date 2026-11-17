"""Final-evaluation safeguards exercised only with synthetic runs and fake models."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from secom_ml import final_test
from secom_ml.models import PRIMARY_MODEL, SECONDARY_MODEL


def forbidden(*args, **kwargs):
    raise AssertionError("reporting must not load data, construct models, or refit")


@pytest.fixture
def synthetic_run(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    X_train = pd.DataFrame({"s000": np.arange(20.0), "s001": 7.0})
    y_train = pd.Series([1, 0, 0, 0, 0] * 4, name="fail")
    train_time = pd.Series(pd.date_range("2000-01-01", periods=20, freq="h"))
    test_index = pd.RangeIndex(20, 28)
    X_test = pd.DataFrame({"s000": np.arange(20.0, 28.0), "s001": -100.0}, index=test_index)
    y_test = pd.Series([1, 0, 1, 0, 0, 1, 0, 0], index=test_index, name="fail")
    test_time = pd.Series(pd.date_range("2000-01-02", periods=8, freq="h"), index=test_index)
    calls = {"train": 0, "test": 0, "fit": [], "predict": []}

    class SyntheticModel:
        def __init__(self, name, probabilities):
            self.name = name
            self.probabilities = np.asarray(probabilities)

        def __repr__(self):
            return f"SyntheticModel({self.name!r})"

        def fit(self, X, y):
            # The constant training sensor must stay excluded even though it
            # takes a different value in the future period.
            assert X.columns.tolist() == ["s000"]
            assert X.index.equals(X_train.index)
            assert y.equals(y_train)
            calls["fit"].append(self.name)
            return self

        def predict_proba(self, X):
            assert X.index.equals(X_test.index)
            assert X.columns.tolist() == ["s000"]
            calls["predict"].append(self.name)
            return np.column_stack([1 - self.probabilities, self.probabilities])

    models = {
        PRIMARY_MODEL: SyntheticModel(PRIMARY_MODEL, [0.1, 0.2, 0.8, 0.04, 0.05, 0.03, 0.07, 0.06]),
        SECONDARY_MODEL: SyntheticModel(
            SECONDARY_MODEL, [0.3, 0.1, 0.7, 0.03, 0.04, 0.05, 0.06, 0.02]
        ),
    }

    def training():
        calls["train"] += 1
        return X_train, y_train, train_time

    def holdout():
        calls["test"] += 1
        manifest = json.loads((results_dir / final_test.MANIFEST_FILE).read_text())
        assert manifest["status"] == "frozen_before_test"
        assert manifest["primary_model"] == PRIMARY_MODEL
        assert manifest["headline_cost_ratio"] == 10
        assert manifest["drift_correction"] == "none"
        assert manifest["sensors"] == ["s000"]
        assert calls["fit"] == [PRIMARY_MODEL, SECONDARY_MODEL]
        assert not (results_dir / final_test.PREDICTIONS_FILE).exists()
        return X_test, y_test, test_time

    monkeypatch.setattr(final_test, "load_train", training)
    monkeypatch.setattr(final_test, "load_test", holdout)
    monkeypatch.setattr(final_test, "build_final_models", lambda: models)
    monkeypatch.setattr(final_test, "N_BOOTSTRAP", 30)
    monkeypatch.setattr(final_test, "BLOCK_LENGTH", 4)
    monkeypatch.setattr(final_test, "COST_RATIOS", [5, 10, 20, 50])
    return SimpleNamespace(
        results_dir=results_dir,
        figures_dir=tmp_path / "figures",
        calls=calls,
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        train_time=train_time,
        test_time=test_time,
        models=models,
    )


def test_protocol_is_frozen_before_single_holdout_access_and_both_models_are_saved(synthetic_run):
    run = synthetic_run
    final_test.run_evaluation(run.results_dir)
    predictions, manifest = final_test.read_saved(run.results_dir)

    assert run.calls == {
        "train": 1,
        "test": 1,
        "fit": [PRIMARY_MODEL, SECONDARY_MODEL],
        "predict": [PRIMARY_MODEL, SECONDARY_MODEL],
    }
    assert manifest["status"] == "predictions_saved"
    assert manifest["test_runs"] == 8
    assert manifest["test_fails"] == 3
    assert len(predictions) == 16
    assert predictions.groupby("model")["run"].nunique().eq(8).all()
    assert set(predictions["role"]) == {"primary", "secondary"}
    assert set(manifest["source_sha256"]) >= {"models.py", "config.py", "final_test.py"}

    with pytest.raises(FileExistsError, match="report-only"):
        final_test.run_evaluation(run.results_dir)
    assert run.calls["train"] == run.calls["test"] == 1


@pytest.mark.parametrize("existing_file", [final_test.MANIFEST_FILE, final_test.PREDICTIONS_FILE])
def test_even_incomplete_existing_evaluation_blocks_any_data_access(
    tmp_path, monkeypatch, existing_file
):
    (tmp_path / existing_file).write_text("incomplete")
    monkeypatch.setattr(final_test, "load_train", forbidden)
    monkeypatch.setattr(final_test, "load_test", forbidden)
    monkeypatch.setattr(final_test, "build_final_models", forbidden)
    with pytest.raises(FileExistsError, match="already started"):
        final_test.run_evaluation(tmp_path)


@pytest.mark.parametrize(
    "bad_split", ["same_time", "overlapping_rows", "label_alignment", "time_order"]
)
def test_invalid_chronology_or_row_alignment_stops_before_predictions(synthetic_run, bad_split):
    run = synthetic_run
    if bad_split == "same_time":
        run.test_time.iloc[0] = run.train_time.iloc[-1]
    elif bad_split == "overlapping_rows":
        overlapping = pd.RangeIndex(19, 27)
        run.X_test.index = run.y_test.index = run.test_time.index = overlapping
    elif bad_split == "label_alignment":
        run.y_test.index = pd.RangeIndex(21, 29)
    else:
        run.test_time.iloc[0] = run.test_time.iloc[-1] + pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="alignment or chronological separation"):
        final_test.run_evaluation(run.results_dir)
    assert run.calls["test"] == 1
    assert run.calls["predict"] == []
    assert not (run.results_dir / final_test.PREDICTIONS_FILE).exists()
    with pytest.raises(FileExistsError):
        final_test.run_evaluation(run.results_dir)


def test_report_regeneration_uses_only_unchanged_saved_predictions(synthetic_run, monkeypatch):
    run = synthetic_run
    final_test.run_evaluation(run.results_dir)
    cache = run.results_dir / final_test.PREDICTIONS_FILE
    manifest = run.results_dir / final_test.MANIFEST_FILE
    before = (cache.read_bytes(), manifest.read_bytes())
    for name in ["load_train", "load_test", "build_final_models", "run_evaluation"]:
        monkeypatch.setattr(final_test, name, forbidden)
    for model in run.models.values():
        monkeypatch.setattr(model, "fit", forbidden)
        monkeypatch.setattr(model, "predict_proba", forbidden)

    final_test.report_saved(run.results_dir, run.figures_dir)

    assert (cache.read_bytes(), manifest.read_bytes()) == before
    assert (run.figures_dir / "final_test.png").is_file()
    assert (run.results_dir / "final_test_report.md").is_file()
    metrics = pd.read_csv(run.results_dir / "final_test_metrics.csv")
    costs = pd.read_csv(run.results_dir / "final_test_costs.csv")
    assert metrics["model"].tolist() == [PRIMARY_MODEL, SECONDARY_MODEL]
    assert len(costs) == 8
    assert set(costs["ratio"]) == {5, 10, 20, 50}
    assert metrics["iid_valid_resamples"].between(1, 30).all()


def test_cache_checksum_rejects_changed_predictions(synthetic_run):
    run = synthetic_run
    final_test.run_evaluation(run.results_dir)
    path = run.results_dir / final_test.PREDICTIONS_FILE
    with path.open("a", encoding="utf-8") as output:
        output.write("\n")
    with pytest.raises(ValueError, match="checksum"):
        final_test.read_saved(run.results_dir)


def test_probability_cache_preserves_decisions_one_float_from_threshold(synthetic_run):
    run = synthetic_run
    threshold = 1 / 11
    original = np.array(
        [
            np.nextafter(threshold, 0.0),
            threshold,
            np.nextafter(threshold, 1.0),
            0.01,
            0.02,
            0.03,
            0.04,
            0.05,
        ]
    )
    run.models[PRIMARY_MODEL].probabilities = original
    final_test.run_evaluation(run.results_dir)
    predictions, _ = final_test.read_saved(run.results_dir)
    saved = predictions.loc[predictions["model"] == PRIMARY_MODEL, "probability"].to_numpy()
    np.testing.assert_array_equal(saved, original)
    np.testing.assert_array_equal(saved > threshold, original > threshold)


def test_cache_rejects_models_evaluated_on_different_runs_even_with_valid_checksum(synthetic_run):
    run = synthetic_run
    final_test.run_evaluation(run.results_dir)
    path = run.results_dir / final_test.PREDICTIONS_FILE
    predictions = pd.read_csv(path)
    row = predictions.index[predictions["model"] == SECONDARY_MODEL][0]
    predictions.loc[row, "run"] = 999
    predictions.to_csv(path, index=False)
    manifest_path = run.results_dir / final_test.MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text())
    manifest["predictions_sha256"] = final_test.sha256(path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="same runs and outcomes"):
        final_test.read_saved(run.results_dir)


def minimal_manifest():
    return {
        "models": [PRIMARY_MODEL],
        "primary_model": PRIMARY_MODEL,
        "bootstrap_repeats": 30,
        "seed": 42,
        "block_length": 2,
        "train_rate": 0.1,
        "cost_ratios": [10],
        "cost_false_alarm": 1,
        "reliability_bins": 5,
    }


def test_fixed_cost_counts_use_strict_threshold_and_paired_uncertainty():
    y = np.array([1, 1, 0, 0])
    p = np.array([1 / 11, 0.7, 0.2, 0.05])
    predictions = pd.DataFrame(
        {"run": np.arange(4), "model": PRIMARY_MODEL, "fail": y, "probability": p}
    )
    manifest = minimal_manifest()
    _, costs, _ = final_test.evaluate_predictions(predictions, manifest)
    row = costs.iloc[0]
    assert row["threshold"] == pytest.approx(1 / 11)
    assert (row["caught"], row["misses"], row["false_alarms"], row["true_negatives"]) == (
        1,
        1,
        1,
        1,
    )
    assert row["alarms"] == 2  # exactly-on-threshold failure stays quiet
    assert row["cost"] == pytest.approx(2750)
    assert row["cost_never_alarm"] == pytest.approx(5000)
    assert row["cost_inspect_all"] == pytest.approx(500)
    assert row["saving_vs_best_naive"] == pytest.approx(-4.5)

    indices = final_test.bootstrap_rows(y, repeats=30, seed=42)
    alarm = p > 1 / 11
    yy, aa = y[indices].astype(bool), alarm[indices]
    loss = (aa & ~yy).sum(axis=1) + 10 * (~aa & yy).sum(axis=1)
    best = np.minimum(10 * yy.sum(axis=1), (~yy).sum(axis=1))
    expected = np.percentile(1 - loss / best, [2.5, 97.5])
    assert row["saving_vs_best_naive_iid_low"] == pytest.approx(expected[0])
    assert row["saving_vs_best_naive_iid_high"] == pytest.approx(expected[1])
    assert row["iid_valid_resamples"] == len(indices)


def test_constant_forecasts_still_account_for_every_run_in_reliability():
    predictions = pd.DataFrame(
        {"run": np.arange(8), "model": PRIMARY_MODEL, "fail": [0, 1] * 4, "probability": 0.1}
    )
    metrics, costs, reliability = final_test.evaluate_predictions(predictions, minimal_manifest())
    assert len(reliability) == 1
    assert reliability.iloc[0]["runs"] == 8
    assert reliability.iloc[0]["fails"] == 4
    assert metrics.iloc[0]["roc_auc"] == pytest.approx(0.5)
    assert costs.iloc[0]["alarms"] == 8


def test_circular_block_resamples_preserve_contiguous_blocks_and_exact_sample_size():
    y = np.array([0, 1] * 5)
    rows = final_test.bootstrap_rows(y, repeats=25, seed=7, block_length=3)
    assert rows.shape == (25, len(y))
    assert np.array_equal(rows, final_test.bootstrap_rows(y, repeats=25, seed=7, block_length=3))
    for offset in [0, 3, 6]:
        assert np.all((rows[:, offset + 1] - rows[:, offset]) % len(y) == 1)
        assert np.all((rows[:, offset + 2] - rows[:, offset + 1]) % len(y) == 1)
    assert ((rows >= 0) & (rows < len(y))).all()


def test_iid_resampling_filters_single_class_samples_and_reports_retained_count():
    y = np.array([0, 1])
    rows = final_test.bootstrap_rows(y, repeats=100, seed=42)
    assert 0 < len(rows) < 100
    assert np.all(y[rows].sum(axis=1) == 1)


@pytest.mark.parametrize("probability", [[0.1, np.nan], [-0.1, 0.9], [0.2, 1.1], [0.2]])
def test_invalid_probabilities_are_rejected(probability):
    with pytest.raises(ValueError):
        final_test.validate_scores(np.array([0, 1]), np.asarray(probability))


def test_single_class_evaluation_is_rejected():
    with pytest.raises(ValueError, match="both pass and fail"):
        final_test.validate_scores(np.array([0, 0]), np.array([0.1, 0.2]))
