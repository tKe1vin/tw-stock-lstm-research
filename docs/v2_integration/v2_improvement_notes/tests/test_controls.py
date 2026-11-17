import json
from pathlib import Path

from docs.v2_integration.v2_improvement_notes.src.checks import (
    five_day_label,
    same_order,
    scaler_stays_inside_train,
    split_is_causal,
    trade_cost,
)
from docs.v2_integration.v2_improvement_notes.src.saved_books import cost_gap, load_books

ROOT = Path(__file__).resolve().parents[4]
FROZEN = ROOT / "models" / "model_555_v2" / "weights" / "weights.npz"
V211 = ROOT / "experiments" / "results" / "v2" / "v211"


def test_buy_and_sell_costs_are_not_doubled():
    assert trade_cost(100000, "buy") == 142.5
    assert trade_cost(100000, "sell") == 442.5
    assert trade_cost(0, "sell") == 0


def test_five_day_label_waits_until_maturity():
    closes = [100, 101, 102, 103, 104, 105, 110, 120]
    assert five_day_label(closes, 0) == 110 / 101 - 1
    assert five_day_label(closes, 2) is None


def test_split_and_scaler_reject_future_rows():
    assert split_is_causal("2025-12-17", "2025-12-18", "2026-03-25", "2026-03-26")
    assert not split_is_causal("2026-03-26", "2025-12-18", "2026-03-25", "2026-03-26")
    assert scaler_stays_inside_train("2025-12-17", "2025-12-17")
    assert not scaler_stays_inside_train("2026-11-04", "2025-12-17")


def test_prediction_order_must_match_execution_order():
    order = ["2330.TW", "2317.TW", "2603.TW", "2002.TW", "2891.TW"]
    assert same_order(order, list(order))
    assert not same_order(order, sorted(order))


def test_saved_v2_book_does_not_beat_equal_weight():
    books = load_books()
    selected = next(row for row in books if row["name"] == "rolling_selected_threshold_weekly")
    blocked = next(row for row in books if row["name"] == "buy_hold")
    assert selected["status"] == "EXPLORATORY_RESULT"
    assert selected["beats_equal"] is False
    assert blocked["status"] == "BLOCKED"
    gap = cost_gap(books)
    assert gap["status"] == "EXPLORATORY_RESULT"
    assert gap["cost_explains_entire_gap"] is False


def test_audit_does_not_write_frozen_research_files():
    before = {path: path.stat().st_mtime_ns for path in (FROZEN, V211 / "status.json") if path.exists()}
    load_books()
    after = {path: path.stat().st_mtime_ns for path in before}
    assert after == before
    assert json.loads((ROOT / "reports" / "v210_comparison" / "comparison.json").read_text(encoding="utf-8"))["candidate"] == "rolling_selected"
