"""Refresh the V2.11 snapshot, then publish today's signal if the data is complete.

`python -m experiments.v2.v211` does not download prices or news. A second run
on the same day keeps the original signal file and its publish time.
Exit 0 means the signal was published or already stored. Exit 2 means the
source is not ready yet. Exit 1 is a real failure and must be reported.
"""
from __future__ import annotations

import json
import os
from datetime import datetime

import pandas as pd

from experiments.v2.guard import sha256, snapshot, assert_unchanged
from experiments.v2.v211 import LIVE, WEIGHT_DIR, score_matured, seal_signals, update_book, write_status
from experiments.v2.v211_market import update_market
from scripts.common import TZ


def classify_exit(status, report, signal_exists, now):
    """Exit 0 only when today's own session was published. An older file does not count."""
    today = str(pd.Timestamp(now.astimezone(TZ).date()).date())
    signal = status.get("signal") or {}
    reason = signal.get("reason")
    completed = str(signal.get("completed_session") or (report or {}).get("completed_session") or "")
    if completed == today and (status.get("signal_published") or reason == "already_written"):
        return 0
    if reason == "would_backfill_an_earlier_session":
        if completed and not signal_exists(completed):
            return 1
        return 2
    error = str(status.get("error") or (report or {}).get("error") or "")
    if any(token in error for token in ("改寫", "不一致", "停止發布", "被改過")):
        return 1
    if reason == "prediction_failed" or "prediction_failed" in error or "預測失敗" in error:
        return 1
    gate = (report or {}).get("gate")
    if (
        not (report or {}).get("ready_to_predict", True)
        or gate in ("prices_missing", "sentiment_waiting")
        or reason == "completed_session_missing_from_prices"
        or "特徵不完整" in error
        or "情緒未到齊" in error
    ):
        return 2
    if status.get("ok") is False:
        return 1
    return 1


def waiting_status(now, report):
    live = sorted(path.name for path in LIVE.glob("*.csv")) if LIVE.exists() else []
    return {
        "ok": False,
        "error": report.get("error") or report.get("gate"),
        "clock": now.astimezone(TZ).isoformat(),
        "data_updated_through_latest_completed_session": False,
        "signal_published": False,
        "live_files": live,
        "rebalance_today": False,
        "labels_matured_today": [],
        "scorable_signal_days": 0 if not live else None,
        "immature_predictions": 0 if not live else None,
        "fetch": report,
        "weights_reused": (WEIGHT_DIR / "model_card.json").exists(),
        "official_555_replacement": False,
        "tracker_downloads_data": False,
        "daily_command_downloads_data": True,
    }


def update_recorded_results(now):
    """Refresh fills and matured labels from prices already stored. Never publishes a signal."""
    from experiments.horizon_experiment import executable_labels
    from experiments.risk_experiment import load_panel
    from experiments.v2.v211_market import PRICES

    if not PRICES.exists():
        return {"book": {"status": "no_prices", "first_fill_completed": False}, "scoring": {"matured_signal_days": 0, "pending_predictions": 0, "matured_today": [], "next_label_end": None}}
    panel = load_panel(PRICES)
    _one, five, _excess = executable_labels(panel)
    context = (panel, None, None, _one, five, None)
    scored = score_matured(context)
    book = update_book(context)
    return {"scoring": scored, "book": book, "as_of": str(now.astimezone(TZ).date())}


def main(clock=None):
    clock = clock or (lambda: datetime.now(TZ))
    try:
        seal_signals()
        before = snapshot()
        weights = WEIGHT_DIR / "weights.npz"
        before_hash = sha256(weights)
        now = clock()
        if os.environ.get("TW_LSTM_SKIP_MARKET_UPDATE") == "1":
            from experiments.v2.v211_market import inspect_market
            report = inspect_market(now)
            report["skipped_download"] = True
        else:
            report = update_market(now, clock=clock)
        # A completion/decision time must follow fetching and scoring, rather than
        # being the run-start timestamp carried through a potentially long scan.
        now = clock()
        recorded = update_recorded_results(now)
        if report.get("ready_to_predict"):
            from experiments.v2.v211 import run

            status = run(now, fetch=report, publication_clock=clock)
        else:
            status = waiting_status(now, report)
            status["scoring"] = recorded["scoring"]
            status["book"] = recorded["book"]
            status["first_fill_completed"] = bool(recorded["book"].get("first_fill_completed"))
            status["next_label_date"] = recorded["scoring"].get("next_label_end")
            status["labels_matured_today"] = recorded["scoring"].get("matured_today", [])
            status["scorable_signal_days"] = recorded["scoring"].get("matured_signal_days")
            status["immature_predictions"] = recorded["scoring"].get("pending_predictions")
            status["signal_published"] = False
            write_status(status)
        assert_unchanged(before)
        if sha256(weights) != before_hash:
            raise RuntimeError("凍結權重被改寫")
        card = json.loads((WEIGHT_DIR / "model_card.json").read_text(encoding="utf-8"))
        if card.get("weight_sha256") != before_hash:
            raise RuntimeError("模型卡與權重不一致")
        seal_signals()
    except Exception as exc:
        print(json.dumps({"published": False, "reason": "error", "error": str(exc)}, ensure_ascii=False), flush=True)
        return 1
    published = bool(status.get("signal_published"))
    reason = (status.get("signal") or {}).get("reason") or status.get("error")
    code = classify_exit(status, report, lambda day: (LIVE / f"{day}.csv").exists(), now)
    print(json.dumps({
        "published": published,
        "reason": reason,
        "exit_code": code,
        "weights_reused": card.get("weight_sha256") == before_hash,
        "first_fill_completed": bool(status.get("first_fill_completed")),
        "next_label_date": status.get("next_label_date"),
    }, ensure_ascii=False), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
