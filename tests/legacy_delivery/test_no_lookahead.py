"""Time order for news, fills, and five-day labels. Missing today is not yesterday."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_news_after_1330_moves_to_the_next_session():
    from scripts.news_update import to_signal_date

    late = to_signal_date("2026-11-05 19:41:18+08:00")
    previous_evening = to_signal_date("2026-11-04 20:12:44+08:00")
    before_cutoff = to_signal_date("2026-11-05 13:30:00+08:00")
    assert str(pd.Timestamp(late).date()) == "2026-11-06"
    assert str(pd.Timestamp(previous_evening).date()) == "2026-11-05"
    assert str(pd.Timestamp(before_cutoff).date()) == "2026-11-05"


def test_nov5_label_is_not_mature_without_later_closes():
    from experiments.horizon_experiment import LABEL_GAP
    from experiments.risk_experiment import load_panel
    from experiments.v2.v211 import label_maturity
    from experiments.v2.v211_market import PRICES

    panel = load_panel(PRICES)
    result = label_maturity("2026-11-05", panel.index)
    assert LABEL_GAP == 6
    assert result["mature"] is False
    assert result["reason"] == "waiting_for_future_sessions"


def test_missing_today_price_is_not_replaced_by_yesterday():
    from experiments.v2.v211_market import NAMES, PRICES, prediction_gate, select_new_price_rows

    prices = pd.read_csv(PRICES)
    prices["Date"] = pd.to_datetime(prices["Date"]).dt.normalize()
    latest = prices.Date.max()
    older = prices[prices.Date < latest].copy()
    sentiment = pd.read_csv(ROOT / "experiments" / "results" / "v2" / "v211" / "market" / "sentiment.csv")
    ready, reason = prediction_gate(older, sentiment, latest)
    assert ready is False
    assert reason == "prices_missing"
    _new, missing = select_new_price_rows(older, older, latest, [latest], NAMES)
    assert _new is None
    assert str(latest.date()) in missing
    assert str(older.Date.max().date()) not in missing
