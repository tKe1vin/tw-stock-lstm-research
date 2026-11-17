"""Read the already published V2.10 comparison. No files outside the clone are touched."""
from __future__ import annotations

import json
from pathlib import Path

from .checks import classify_saved_book, comparable

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "reports" / "v210_comparison" / "comparison.json"


def load_books(path=None):
    payload = json.loads(Path(path or SOURCE).read_text(encoding="utf-8"))
    books = payload["books"]
    equal = next(row for row in books if row["Name"] == "equal_weekly")
    if not comparable(equal):
        raise RuntimeError("固定等權列本身不可比，停止解讀")
    return [
        classify_saved_book(row, equal["ROI_Pct"]) | {
            "mdd_pct": row.get("MDD_Pct"),
            "sharpe": row.get("Sharpe"),
            "cost": row.get("Total_Transaction_Cost"),
            "cash_pct": row.get("Average_Cash_Weight_Pct"),
            "sessions": row.get("Sessions"),
            "execution": row.get("Execution"),
            "exposure": row.get("Exposure"),
        }
        for row in books
    ]


def cost_gap(books, candidate="rolling_selected_threshold_weekly"):
    by_name = {row["name"]: row for row in books}
    model = by_name[candidate]
    baseline = by_name["equal_weekly"]
    if model["status"] != "EXPLORATORY_RESULT" or baseline["status"] != "EXPLORATORY_RESULT":
        return {"status": "BLOCKED", "reason": "缺少同一協議的兩本帳"}
    extra_cost = float(model["cost"]) - float(baseline["cost"])
    return {
        "status": "EXPLORATORY_RESULT",
        "extra_cost": extra_cost,
        "extra_cost_pct_of_100000": extra_cost / 1000.0,
        "roi_gap_pct": model["gap_vs_equal_pct"],
        "cost_explains_entire_gap": extra_cost / 1000.0 >= abs(model["gap_vs_equal_pct"]),
    }
