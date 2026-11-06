"""555 stays independent of the V2 weights."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_notebook_keeps_its_feature_input():
    text = (ROOT / "models" / "model_555" / "555.ipynb").read_text(encoding="utf-8")
    assert "LSTM_FEATURE_CSV" in text
    assert "LSTM_PRICE_CSV" in text


def test_check_waits_when_sentiment_snapshot_is_absent():
    from scripts.run_model import run

    code = run("555", "check")
    assert code == 2
    day = sorted((ROOT / "outputs" / "555").glob("*"))[-1]
    status = json.loads((day / "status.json").read_text(encoding="utf-8"))
    assert status["training_executed"] is False
    assert status["status"] == "waiting"
