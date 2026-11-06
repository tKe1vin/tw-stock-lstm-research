"""444 stays on its own notebook and refuses an older price file as today."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_notebook_is_the_original_entry():
    text = (ROOT / "models" / "model_444" / "444.ipynb").read_text(encoding="utf-8")
    assert "LSTM_AS_OF_DATE" in text
    assert "LSTM_PRICE_CSV" in text
    assert "LOCAL_PRICE_CSV" in text


def test_check_does_not_train_without_today_snapshot():
    from scripts.run_model import run

    code = run("444", "check")
    assert code == 2
    day = sorted((ROOT / "outputs" / "444").glob("*"))[-1]
    status = json.loads((day / "status.json").read_text(encoding="utf-8"))
    assert status["training_executed"] is False
    assert status["forecast_produced"] is False
    assert status["status"] == "waiting"
