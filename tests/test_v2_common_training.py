"""Purged, immutable and offline-testable V2 feature-ablation training contract."""
import builtins
import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import v2_common_training as training


@pytest.fixture(scope="module")
def config():
    return copy.deepcopy(training.DEFAULT_CONFIG)


@pytest.fixture(scope="module")
def data(config):
    return training.prepare_data(training.ROOT, config)


def test_fixed_splits_use_full_six_session_purge_and_identical_coverage(data, config):
    expected = [
        ("2022-10-31", "2025-12-11", "2025-12-22", "2026-03-27", "2026-04-09", "2026-06-18"),
        ("2023-01-10", "2026-03-05", "2026-03-16", "2026-06-10", "2026-06-22", "2026-08-31"),
        ("2023-04-07", "2026-05-19", "2026-05-28", "2026-08-21", "2026-09-01", "2026-11-16"),
    ]
    for fold, dates in zip(data["folds"], expected):
        assert [len(fold[k]) for k in ("train", "validation", "test")] == [756, 60, 50]
        assert fold["train"][-1] + 6 < fold["validation"][0]
        assert fold["validation"][-1] + 6 < fold["test"][0]
        actual = tuple(str(data["calendar"][fold[k][p]].date())
                       for k in ("train", "validation", "test") for p in (0, -1))
        assert actual == dates
    np.testing.assert_array_equal(np.concatenate([f["test"] for f in data["folds"]]), data["test_positions"])
    assert str(data["first_retained_input"].date()) == "2022-10-18"


def test_scaler_fits_only_union_of_training_input_rows(data, config):
    fold, ticker = data["folds"][0], training.TICKERS[0]
    frame = data["frames"][ticker]
    windows, scaler, rows = training.scaled_windows(frame, training.FEATURES, fold)
    expected_rows = np.unique(np.concatenate([np.arange(p-9, p+1) for p in fold["train"]]))
    np.testing.assert_array_equal(rows, expected_rows)
    assert len(rows) == 765
    assert rows[-1] == fold["train"][-1] < fold["validation"][0]
    future = frame.copy()
    future.loc[future.index[fold["validation"][0]:], training.FEATURES] *= 1e6
    changed, new_scaler, new_rows = training.scaled_windows(future, training.FEATURES, fold)
    np.testing.assert_array_equal(new_rows, rows)
    np.testing.assert_array_equal(new_scaler.scale_, scaler.scale_)
    np.testing.assert_array_equal(changed["train"], windows["train"])
    np.testing.assert_array_equal(scaler.data_min_, frame.iloc[rows][training.FEATURES].min())


@pytest.mark.parametrize("remove_entire_session", [False, True])
def test_missing_training_bar_never_shortens_calendar(tmp_path, config, data, remove_entire_session):
    raw = pd.read_csv(training.ROOT / training.PRICE_PATH)
    date = "2024-06-03"
    missing = raw.Date.eq(date)
    if not remove_entire_session:
        missing &= raw.ticker.eq(training.TICKERS[0])
    path = tmp_path / "prices.csv"
    raw.loc[~missing].to_csv(path, index=False)
    with pytest.raises(ValueError, match="exchange-session prices"):
        training.prepare_data(training.ROOT, config, price_path=path)


def test_retained_coverage_rejects_non_session_extra_date(data):
    panel = data["panel"].copy()
    panel.loc[pd.Timestamp("2026-11-15")] = 100
    panel.sort_index(inplace=True)
    with pytest.raises(ValueError, match="Non-XTAI"):
        training._assert_coverage(panel, data["first_retained_input"], data["as_of"])


def test_warmup_missing_rows_are_preserved_and_risk_matches_paired(data):
    from scripts.paired_research import market_risk, price_panel

    original = price_panel(training.ROOT / training.PRICE_PATH)
    pd.testing.assert_frame_equal(data["panel"], original)
    assert len(original) == 3135
    assert original.isna().any(axis=1).sum() == 4
    pd.testing.assert_frame_equal(data["risk"], market_risk(original))


def test_pending_test_labels_keep_all_150_signals_without_future_prices(data):
    rows = [row for fold in data["folds"] for ticker in training.TICKERS
            for row in training.prediction_rows("news_lstm", fold, ticker, np.zeros(50), data, "test")]
    predictions = pd.DataFrame(rows)
    assert len(predictions) == 750
    assert predictions.Signal_Date.nunique() == 150
    assert predictions.Label_State.eq("MATURED").sum() == 725
    pending = predictions[predictions.Label_State.eq("PENDING")]
    assert len(pending) == 25
    assert pending.Actual_5d_Pct.isna().all()
    assert (pd.to_datetime(pending.Label_End) > data["as_of"]).all()
    assert predictions.Signal_Date.max() == "2026-11-16"
    assert predictions.Execution_Date.max() == "2026-11-17"
    assert predictions.Label_End.max() == "2026-11-24"
    assert data["panel"].index.max() == data["as_of"]


def test_test_label_values_cannot_change_folds_or_scaler(data, config):
    # This fold's test labels are mutable only in this local copy. No fitting
    # argument or calibration input is obtained from them.
    fold = data["folds"][0]
    changed = dict(data)
    changed["target"] = data["target"].copy()
    changed["target"].iloc[fold["test"]] = -1e9
    for split in ("train", "validation"):
        pd.testing.assert_frame_equal(changed["target"].iloc[fold[split]], data["target"].iloc[fold[split]])
    a = training.scaled_windows(data["frames"][training.TICKERS[0]], ["Close"], fold)
    b = training.scaled_windows(changed["frames"][training.TICKERS[0]], ["Close"], fold)
    np.testing.assert_array_equal(a[0]["train"], b[0]["train"])
    np.testing.assert_array_equal(a[1].scale_, b[1].scale_)


@pytest.mark.parametrize("relative", ["data/run", "models/run", "research/v211/run", "experiments/results/v2/v211/run"])
def test_protected_output_roots_are_rejected_before_tensorflow(relative, monkeypatch):
    monkeypatch.setattr(training, "_tensorflow", lambda: pytest.fail("TensorFlow imported"))
    with pytest.raises(ValueError, match="protected"):
        training.validate_output_dir(training.ROOT, training.ROOT / relative)


def test_existing_output_cannot_be_replaced(tmp_path):
    with pytest.raises(FileExistsError, match="immutable"):
        training.validate_output_dir(training.ROOT, tmp_path)


@pytest.mark.parametrize("key,value", [("allow_test_selection", True), ("study_status", "out_of_sample"),
                                       ("layers", 1), ("batch_size", 16), ("train_samples", 252)])
def test_protocol_settings_cannot_be_silently_changed(tmp_path, config, key, value):
    altered = copy.deepcopy(config)
    altered[key] = value
    path = tmp_path / "config.json"
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match=key):
        training.load_config(path)


def test_registered_input_hash_mismatch_is_rejected(config):
    altered = copy.deepcopy(config)
    altered["source_sha256"] = {"prices": "0" * 64}
    with pytest.raises(ValueError, match="source hash differs: prices"):
        training.prepare_data(training.ROOT, altered)


def test_default_check_has_no_tensorflow_download_or_writes(tmp_path, monkeypatch, capsys, config):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    before = sorted(tmp_path.rglob("*"))
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"tensorflow", "yfinance", "requests"}:
            pytest.fail(f"Default check imported network/training dependency: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(training, "_tensorflow", lambda: pytest.fail("TensorFlow initialized"))
    monkeypatch.setattr(training, "_write_json", lambda *args: pytest.fail("Default check wrote files"))
    assert training.main(["--config", str(config_path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["read_only"] and not report["tensorflow_imported"]
    assert report["matured_test_signal_days"] == 145
    assert sorted(tmp_path.rglob("*")) == before


def test_training_requires_explicit_output_before_input_loading(monkeypatch):
    monkeypatch.setattr(training, "load_config", lambda *args: pytest.fail("Inputs read before CLI validation"))
    with pytest.raises(SystemExit) as exc:
        training.main(["--mode", "train"])
    assert exc.value.code == 2


def test_network_has_registered_two_lstm_layers_and_dropouts(config):
    def layer(kind):
        return lambda *args, **kwargs: (kind, args, kwargs)

    tf = SimpleNamespace(keras=SimpleNamespace(
        Sequential=lambda items: items,
        layers=SimpleNamespace(**{name: layer(name) for name in ("Input", "LSTM", "Dropout", "Dense")})))
    network = training._network(tf, (10, 6), config)
    assert [item[0] for item in network] == ["Input", "LSTM", "Dropout", "LSTM", "Dropout", "Dense"]
    assert network[1][1] == network[3][1] == (50,)
    assert network[1][2] == {"return_sequences": True}
    assert network[2][1] == network[4][1] == (0.2,)


def test_fake_fit_proves_validation_is_sealed_before_any_test_inference(tmp_path, monkeypatch, data, config):
    output = tmp_path / "run"
    fits, test_calls = [], []

    class FakeModel:
        def __call__(self, values, training=False):
            return np.asarray(values).mean(axis=(1, 2))[:, None]

        def get_weights(self):
            return [np.asarray([42], dtype=np.float32)]

    def fake_fit(tf, windows, targets, settings):
        assert set(targets) == {"train", "validation"}
        assert len(targets["train"]) == 756 and len(targets["validation"]) == 60
        assert all(np.max(np.abs(values)) < 1e8 for values in targets.values())
        fits.append(id(targets))
        return FakeModel(), {"epochs_run": 1, "previous_optimizer_carried": False}

    def fake_load(tf, directory, state, settings):
        assert len(fits) == 45
        assert (directory / "validation_predictions.csv").is_file()
        seal = json.loads((directory / "validation_tau.json").read_text())
        assert seal["test_predictions_computed"] is False
        assert len(seal["thresholds"]) == 9
        assert seal["validation_sha256"] == training.digest(directory / "validation_predictions.csv")
        assert all(not t["test_used"] for t in seal["thresholds"])
        test_calls.append(state["weights_path"])
        return FakeModel()

    fake_tf = SimpleNamespace(__version__="test-only", keras=SimpleNamespace(
        __version__="test-only", backend=SimpleNamespace(clear_session=lambda: None)))
    monkeypatch.setattr(training, "_tensorflow", lambda: fake_tf)
    monkeypatch.setattr(training, "_fit_one", fake_fit)
    monkeypatch.setattr(training, "_load_weights_model", fake_load)
    # Poison every current test label in a copy of fold 3. The fitter must still
    # get exactly the earlier train/validation labels, and pending stays NaN.
    local = dict(data)
    local["target"] = data["target"].copy()
    last = data["folds"][-1]
    mature_last = last["test"][:-5]
    local["target"].iloc[mature_last] = -1e9
    summary = training.train(local, config, output)
    assert summary["fresh_model_fits"] == len(fits) == len(test_calls) == 45
    assert summary["selected_candidate"] is None
    assert summary["saved_weights_validation_maximum_delta"] == 0
    assert all(v == {"test_rows": 750, "matured_rows": 725, "pending_rows": 25, "test_signal_days": 150}
               for v in summary["candidate_counts"].values())
    assert len(list(output.rglob("*.weights.npz"))) == 45
    with np.load(next(output.rglob("*.weights.npz")), allow_pickle=False) as archive:
        assert archive.files == ["weight_00"]  # Neural weights only, no optimizer.
    assert not (output / "inputs").exists()
    manifest = training.verify_run(output)
    assert all(not path.startswith("/") for path in (v["path"] for v in manifest["external_sources"].values()))
    artifact = output / "test_predictions.csv"
    artifact.write_text("tampered")
    with pytest.raises(ValueError, match="hash differs"):
        training.verify_run(output)


def test_manifest_cannot_traverse_outside_run(tmp_path):
    outside = tmp_path / "secret.txt"
    outside.write_text("outside")
    output = tmp_path / "run"
    output.mkdir()
    (output / "reproduction_manifest.json").write_text(json.dumps({
        "schema": "v2-common-reproduction-v1", "raw_inputs_copied": False,
        "files_sha256": {"../secret.txt": training.digest(outside)},
    }))
    with pytest.raises(ValueError, match="Artifact"):
        training.verify_run(output)
