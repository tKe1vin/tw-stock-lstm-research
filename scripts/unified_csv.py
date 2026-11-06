"""Shared exchange CSV. Missing values stay marked and are never invented."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

COLUMNS = [
    "model_id",
    "model_version",
    "signal_date",
    "run_date",
    "generated_at",
    "data_asof",
    "stock_id",
    "stock_name",
    "predicted_return_pct",
    "prediction_horizon_days",
    "rank",
    "target_weight",
    "cash_weight",
    "risk_status",
    "intended_execution_date",
    "execution_price",
    "execution_status",
    "model_hash",
    "data_hash",
]

NOT_APPLICABLE = "不適用"
NAMES = {
    "2330.TW": "台積電",
    "2317.TW": "鴻海",
    "2603.TW": "長榮",
    "2002.TW": "中鋼",
    "2891.TW": "中信金",
}


def blank_row() -> dict:
    return {column: NOT_APPLICABLE for column in COLUMNS}


def write_table(path: Path, rows: list[dict], metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    meta = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "predicted_return_pct_unit": "percent_points",
        "percent_points_meaning": "0.677330 means +0.677330 percent, not 67.7330 percent.",
        "target_weight_unit": "fraction_of_capital",
        "cash_weight_unit": "fraction_of_capital",
        "not_applicable_token": NOT_APPLICABLE,
        "comparison_allowed": False,
    }
    meta.update(metadata)
    path.with_suffix(".metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
