"""Causal split, scaler and bounded validation selection for the extension."""
import numpy as np
import pandas as pd
import pytest

from scripts import research_extension as extension


def test_six_session_matured_labels_do_not_cross_splits():
    config = extension.load_config()
    split = extension.split_positions(np.arange(30, 1400), config)
    assert split["train"][-1] + 6 < split["validation"][0]
    assert split["validation"][-1] + 6 < split["test"][0]
    assert len(split["train"]) == 756
    assert len(split["validation"]) == 60
    assert len(split["test"]) == 150


def test_future_extreme_features_do_not_change_scaler_or_training():
    config = extension.load_config()
    split = extension.split_positions(np.arange(30, 1400), config)
    frame = pd.DataFrame({name: np.arange(1500, dtype=float) + i for i, name in enumerate(extension.FEATURES)})
    windows, scaler, rows = extension.scaled_windows(frame, extension.FEATURES, split, 10)
    future = frame.copy()
    future.iloc[split["validation"][0]:] *= 100000
    other, other_scaler, other_rows = extension.scaled_windows(future, extension.FEATURES, split, 10)
    np.testing.assert_array_equal(rows, other_rows)
    np.testing.assert_array_equal(windows["train"], other["train"])
    np.testing.assert_array_equal(scaler.scale_, other_scaler.scale_)
    assert rows.max() < split["validation"][0]


def validation_frame():
    rows = []
    for model, predictions in (("price_lstm", [0, 1, 3, 2, 4]), ("sentiment_risk_lstm", [0, 1, 2, 3, 4])):
        for ticker, actual, predicted in zip(extension.TICKERS, range(5), predictions):
            rows.append({"Model": model, "Split": "validation", "Signal_Date": "2025-01-01",
                         "Ticker": ticker, "Predicted": predicted, "Actual_5d_Pct": actual})
    return pd.DataFrame(rows)


def test_test_values_cannot_change_registered_candidate_selection():
    config = extension.load_config()
    validation = validation_frame()
    selected = extension.choose_candidate(validation, config)
    test = validation.copy()
    test["Split"] = "test"
    test["Actual_5d_Pct"] = -test.Predicted * 10000
    assert extension.choose_candidate(pd.concat([validation, test]), config) == selected
    assert selected["selected"] == "sentiment_risk_lstm"
    assert selected["test_used_for_selection"] is False


def test_selection_tie_keeps_price_control():
    frame = validation_frame()
    frame["Predicted"] = frame.Actual_5d_Pct
    assert extension.choose_candidate(frame, extension.load_config())["selected"] == "price_lstm"


def test_production_replacement_and_test_selection_are_rejected(tmp_path):
    config = extension.load_config()
    config["allow_test_selection"] = True
    path = tmp_path / "config.json"
    path.write_text(__import__("json").dumps(config))
    with pytest.raises(ValueError, match="prohibited"):
        extension.load_config(path)


def test_existing_output_cannot_be_overwritten(tmp_path):
    with pytest.raises(FileExistsError, match="immutable"):
        extension.train(None, None, tmp_path, None, None, None)


def test_changed_frozen_artifacts_are_rejected_without_tensorflow(tmp_path):
    weights = tmp_path / "weights.npz"
    weights.write_bytes(b"frozen research artifact")
    (tmp_path / "reproduction_manifest.json").write_text(__import__("json").dumps({
        "files_sha256": {"weights.npz": extension.digest(weights)},
    }))
    extension.verify_run(tmp_path)
    weights.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        extension.verify_run(tmp_path)


def test_book_ends_after_final_scheduled_five_day_hold_not_final_daily_label():
    index = pd.bdate_range("2025-01-01", periods=30)
    panel = pd.DataFrame(index=index)
    test = pd.DataFrame([{"Signal_Date": index[i], "Execution_Date": index[i+1], "Label_End": index[i+6]}
                         for i in range(2, 12)])
    start, end = extension.fixed_holding_endpoint(panel, test)
    assert start == index[3]
    assert end == index[13]  # Last eligible signal=7, entry=8, exit=13.
    assert end < test.Label_End.max()  # Final daily forecast was not a new trade.


def test_unimplemented_cost_override_is_rejected_instead_of_ignored(tmp_path):
    config = extension.load_config()
    config["slippage_bps_each_side"] = 10
    path = tmp_path / "config.json"
    path.write_text(__import__("json").dumps(config))
    with pytest.raises(ValueError, match="shared book"):
        extension.load_config(path)


def test_missing_session_is_rejected_before_a_label_can_be_compressed(monkeypatch):
    from scripts import common

    expected = pd.bdate_range("2026-06-01", periods=10)
    monkeypatch.setattr(common, "sessions", lambda start, end: expected)
    panel = pd.DataFrame(100., index=expected, columns=extension.TICKERS + [extension.MARKET])
    extension._assert_trading_coverage(panel, expected[0], expected[-1])
    with pytest.raises(ValueError, match="Missing exchange-session"):
        extension._assert_trading_coverage(panel.drop(expected[3]), expected[0], expected[-1])


def test_nonfinite_session_is_not_a_neutral_or_dropped_price(monkeypatch):
    from scripts import common

    expected = pd.bdate_range("2026-06-01", periods=10)
    monkeypatch.setattr(common, "sessions", lambda start, end: expected)
    panel = pd.DataFrame(100., index=expected, columns=extension.TICKERS + [extension.MARKET])
    panel.loc[expected[4], extension.TICKERS[0]] = np.nan
    with pytest.raises(ValueError, match="Non-finite"):
        extension._assert_trading_coverage(panel, expected[0], expected[-1])


def test_new_run_cannot_be_written_inside_production_signal_or_market_roots():
    for relative in ("data/new-extension-run", "models/new-extension-run",
                     "research/v211/new-extension-run", "experiments/results/v2/v211/daily/new-extension-run"):
        with pytest.raises(ValueError, match="production inputs"):
            extension.train(None, None, extension.ROOT / relative, None, None, None)


def test_replay_rejects_changed_runtime_source_before_loading_models():
    with pytest.raises(ValueError, match="source differs"):
        extension._verify_runtime_sources({"replay_code_sha256": {"scripts/research_extension.py": "changed"}})
