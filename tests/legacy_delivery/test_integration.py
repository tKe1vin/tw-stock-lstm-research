"""The four output trees stay separate, and a second run does not replace the signal."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = "37591d21300977b9486212fc86fe257bb050db47177fa2f41bd7a8904919f659"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_run_all_check_keeps_separate_outputs_and_the_signal():
    signal = ROOT / "experiments" / "results" / "v2" / "v211" / "daily" / "2026-11-05.csv"
    before = _sha256(signal)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    command = [sys.executable, str(ROOT / "scripts" / "run_all_daily.py"), "--mode", "check", "--skip-download"]
    first = subprocess.run(command, cwd=str(ROOT), env=env, text=True, encoding="utf-8", errors="replace", capture_output=True)
    second = subprocess.run(command, cwd=str(ROOT), env=env, text=True, encoding="utf-8", errors="replace", capture_output=True)
    assert _sha256(signal) == before == EXPECTED, first.stdout + first.stderr
    assert first.returncode == 2, first.stdout + first.stderr
    assert second.returncode == 2, second.stdout + second.stderr
    summary_dirs = sorted((ROOT / "outputs" / "daily_summary").glob("*"))
    summary = json.loads((summary_dirs[-1] / "summary.json").read_text(encoding="utf-8"))
    assert set(summary["steps"]) == {"444", "555", "555_v2", "v211"}
    assert summary["steps"]["444"]["status"] == "waiting"
    assert summary["steps"]["555"]["status"] == "waiting"
    assert summary["steps"]["v211"]["status"] == "waiting"
    import pandas as pd
    from experiments.v2.v211_market import PRICES, SENTIMENT, prediction_gate

    ready, _reason = prediction_gate(pd.read_csv(PRICES), pd.read_csv(SENTIMENT), summary["completed_session"])
    expected = "success" if ready else "waiting"
    assert summary["steps"]["555_v2"]["status"] == expected
