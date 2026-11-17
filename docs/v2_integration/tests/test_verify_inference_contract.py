"""Frozen inference comparison rejects partial, changed or non-finite evidence."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import pytest


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("v2_inference_contract", ROOT / "docs/v2_integration/verify_v211_inference.py")
verify = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify)


@pytest.fixture
def signal(tmp_path):
    source = ROOT / "experiments/results/v2/v211"
    daily = tmp_path / "daily"
    daily.mkdir()
    path = daily / "2026-11-05.csv"
    shutil.copyfile(source / "daily" / path.name, path)
    registry = json.loads((source / "signal_registry.json").read_text())
    registry = {path.name: registry[path.name]}
    (tmp_path / "signal_registry.json").write_text(json.dumps(registry))
    return daily, path, registry


def rewrite(path, rows, registry, *, update_hash=True):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if update_hash:
        registry[path.name]["file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        (path.parent.parent / "signal_registry.json").write_text(json.dumps(registry))


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_canonical_five_stock_signal_matches_registry_without_rewriting(signal):
    daily, path, _registry = signal
    original = path.read_bytes()
    result = verify.load_published_signal("2026-11-05", daily)
    assert set(result["predictions"]) == verify.TICKERS
    assert result["file_sha256"] == hashlib.sha256(original).hexdigest()
    assert result["historical_availability_certified"] is False
    assert path.read_bytes() == original


@pytest.mark.parametrize("kind", ["partial", "duplicate", "wrong_ticker", "nan", "infinity", "date", "model", "version"])
def test_matching_local_hash_cannot_bypass_row_contract(signal, kind):
    daily, path, registry = signal
    data = rows(path)
    if kind == "partial":
        data = data[:1]
    elif kind == "duplicate":
        data[-1]["Ticker"] = data[0]["Ticker"]
    elif kind == "wrong_ticker":
        data[-1]["Ticker"] = "9999.TW"
    elif kind in ("nan", "infinity"):
        data[0]["Predicted_Return_Pct"] = "nan" if kind == "nan" else "inf"
    elif kind == "date":
        data[0]["Signal_Date"] = "2026-11-06"
    elif kind == "model":
        data[0]["Model_Sha256"] = "0" * 64
    else:
        data[0]["Model_Version"] = "unreviewed-model"
    rewrite(path, data, registry)
    with pytest.raises(ValueError):
        verify.load_published_signal("2026-11-05", daily)


def test_changed_bytes_are_rejected_against_published_registry(signal):
    daily, path, registry = signal
    data = rows(path)
    data[0]["Predicted_Return_Pct"] = "123.45"
    rewrite(path, data, registry, update_hash=False)
    with pytest.raises(ValueError, match="registry/hash"):
        verify.load_published_signal("2026-11-05", daily)


def test_absent_registry_cannot_turn_an_unregistered_copy_into_evidence(signal):
    daily, _path, _registry = signal
    (daily.parent / "signal_registry.json").write_text("{}")
    with pytest.raises(ValueError, match="registry entry"):
        verify.load_published_signal("2026-11-05", daily)


@pytest.mark.parametrize("day", ["20261105", "../market/prices", "2026-02-30"])
def test_requested_date_cannot_escape_or_change_the_saved_signal(day, signal):
    daily, _path, _registry = signal
    with pytest.raises(ValueError):
        verify.load_published_signal(day, daily)


def test_actual_predictions_require_the_same_complete_finite_five_stocks():
    pairs = [(ticker, 0.1) for ticker in sorted(verify.TICKERS)]
    assert len(verify.validate_prediction_rows(pairs)) == 5
    for invalid in (pairs[:1], [*pairs, pairs[0]], [*pairs[:-1], (pairs[-1][0], float("nan"))]):
        with pytest.raises(ValueError):
            verify.validate_prediction_rows(invalid)
