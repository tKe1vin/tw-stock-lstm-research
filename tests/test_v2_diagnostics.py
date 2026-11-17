"""V2 optimization diagnostics use identical mature rows and guarded saved inputs."""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from dashboard.v2_diagnostics import BOOK, ROOT, V2DiagnosticError, load_v2_diagnostics


def test_saved_model_and_zero_share_complete_mature_rows_and_units():
    result = load_v2_diagnostics()
    frame = pd.read_csv(ROOT / BOOK / "matured_scores.csv")
    summary = result["summary"]
    assert summary["rows"] == len(frame)
    assert summary["signal_days"] == frame.Signal_Date.nunique()
    assert summary["official_claim_blocked"] is True
    assert summary["historical_availability_certified"] is False
    assert set(result["rows"].Signal_Date) == set(frame.Signal_Date)
    error = frame.Predicted - frame.Actual_5d_Pct
    assert summary["model_mae"] == pytest.approx(error.abs().mean())
    assert summary["zero_mae"] == pytest.approx(frame.Actual_5d_Pct.abs().mean())
    assert summary["mae_improvement_pp"] == pytest.approx(summary["zero_mae"] - summary["model_mae"])
    assert result["by_day"].Rows.eq(5).all()
    assert all(Path(name).suffix != ".npz" for name in result["sources"])


def copy_book(tmp_path):
    shutil.copytree(ROOT / BOOK, tmp_path / BOOK)
    return tmp_path / BOOK


def reseal(book, name):
    path = book / name
    checkpoint = book / "ledger_integrity.json"
    data = json.loads(checkpoint.read_text())
    raw = path.read_bytes()
    data["prefixes"][name].update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    checkpoint.write_text(json.dumps(data))


def test_changed_source_is_rejected_before_scoring(tmp_path):
    book = copy_book(tmp_path)
    path = book / "matured_scores.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(V2DiagnosticError, match="完整性"):
        load_v2_diagnostics(tmp_path)


@pytest.mark.parametrize("change", ["missing-stock", "changed-prediction", "wrong-label"])
def test_complete_stock_set_original_prediction_and_maturity_are_independent_checks(tmp_path, change):
    book = copy_book(tmp_path)
    path = book / "matured_scores.csv"
    frame = pd.read_csv(path)
    if change == "missing-stock":
        frame = frame.iloc[1:]
    elif change == "changed-prediction":
        frame.loc[0, "Predicted"] += 1
    else:
        frame.loc[0, "Actual_5d_Pct"] += 1
    frame.to_csv(path, index=False)
    reseal(book, "matured_scores.csv")
    with pytest.raises(V2DiagnosticError):
        load_v2_diagnostics(tmp_path)


def test_diagnosis_does_not_write_or_retrain(monkeypatch):
    import requests
    import sys
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in (ROOT / BOOK).rglob("*") if path.is_file()}
    monkeypatch.setattr(requests.sessions.Session, "request", lambda *a, **k: pytest.fail("no network"))
    monkeypatch.setitem(sys.modules, "tensorflow", None)
    result = load_v2_diagnostics()
    assert result["available"]
    assert before == {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in (ROOT / BOOK).rglob("*") if path.is_file()}
