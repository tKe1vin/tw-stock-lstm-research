"""Scheduled state writers must use the current repaired and published state."""
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_daily_writer_checks_out_current_main_after_waiting_for_concurrency():
    workflow = yaml.safe_load((ROOT / ".github/workflows/daily-lstm.yml").read_text())
    job = workflow["jobs"]["daily"]
    # Both a manual retry and a delayed scheduled job can have an old trigger
    # SHA. Starting from it could omit a repair or an earlier published signal.
    checkout, = [step for step in job["steps"]
                 if str(step.get("uses", "")).startswith("actions/checkout@")]
    assert job["if"] == "github.ref == 'refs/heads/main'"
    assert checkout["with"]["ref"] == "main"
    assert checkout["with"]["fetch-depth"] == 0
    assert workflow["permissions"]["contents"] == "write"
    assert workflow["concurrency"]["cancel-in-progress"] is False
