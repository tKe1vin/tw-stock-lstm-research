"""Isolated checks for the V2 research. They do not train or publish signals."""
from __future__ import annotations

FEE_RATE = 0.001425
SELL_TAX_RATE = 0.003


def trade_cost(notional, side):
    amount = float(notional)
    if amount < 0:
        raise ValueError("名目金額不可為負")
    if side == "buy":
        return amount * FEE_RATE
    if side == "sell":
        return amount * (FEE_RATE + SELL_TAX_RATE)
    raise ValueError("只接受 buy 或 sell")


def five_day_label(closes, signal_index):
    needed = signal_index + 6
    if needed >= len(closes) or signal_index + 1 >= len(closes):
        return None
    entry = float(closes[signal_index + 1])
    exit_price = float(closes[needed])
    if entry <= 0 or exit_price <= 0:
        return None
    return exit_price / entry - 1.0


def split_is_causal(train_label_end, validation_signal_start, validation_label_end, test_signal_start):
    return train_label_end < validation_signal_start and validation_label_end < test_signal_start


def scaler_stays_inside_train(fit_end, train_end):
    return fit_end <= train_end


def same_order(left, right):
    return list(left) == list(right)


def comparable(row):
    return bool(row.get("ROI_Comparable_With_Equal")) and row.get("Execution") == "next_session_close"


def classify_saved_book(row, equal_roi):
    """Label a saved V2.10 book. This does not recompute its NAV."""
    if not comparable(row):
        return {
            "name": row["Name"],
            "status": "BLOCKED",
            "reason": "曝險或成交規則與五日等權帳本不同，不能直接排名",
            "roi_pct": row.get("ROI_Pct"),
        }
    gap = float(row["ROI_Pct"]) - float(equal_roi)
    return {
        "name": row["Name"],
        "status": "EXPLORATORY_RESULT",
        "reason": "同一套下一交易日收盤、五日再平衡與風險上限帳本",
        "roi_pct": row["ROI_Pct"],
        "gap_vs_equal_pct": gap,
        "beats_equal": gap > 0,
    }
