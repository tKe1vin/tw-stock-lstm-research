"""Publish current four-system status and the separate 444/555 comparison."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pandas as pd

from scripts.common import ROOT, atomic_json

MODELS = ("444", "555", "555_v2", "v211")
REPORT_SCHEMA = "four-systems-v1"
COMPACT_FILES = (
    "strategy_metrics.csv", "evaluation_protocol.json", "next_day_recommendation.csv",
    "model_metrics.csv", "backtest_nav.csv", "test_predictions.csv", "market_metrics.csv",
    "daily_portfolio_review.csv",
)
REQUIRED_COMPARISON_FILES = (
    "strategy_metrics.csv", "evaluation_protocol.json", "next_day_recommendation.csv",
    "model_metrics.csv", "backtest_nav.csv",
)


def _read(path: Path) -> dict:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def _inside_root(value: str) -> Path:
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError(f"Output path leaves the repository: {value}")
    return path


def _native(model: str, payload: dict, day: str) -> Path:
    """Prefer durable compact native files when full runs lived in an artifact."""
    # 444 exports the benchmark NAV; compare() takes benchmark Sharpe/MDD from
    # 555 only after verifying that the two benchmark NAV series agree.
    required = REQUIRED_COMPARISON_FILES + (("market_metrics.csv",) if model == "555" else ())
    choices = [payload.get("daily_native_dir"), payload.get("native_output_dir")]
    choices.append(f"reports/latest/{model}")
    for value in choices:
        if not value:
            continue
        folder = _inside_root(str(value))
        if not all((folder / name).is_file() for name in required):
            continue
        protocol = _read(folder / "evaluation_protocol.json")
        if str(protocol.get("as_of_date", ""))[:10] != day:
            continue
        metrics = pd.read_csv(folder / "strategy_metrics.csv")
        rec = pd.read_csv(folder / "next_day_recommendation.csv")
        if len(metrics) != 1 or "As_Of_Date" not in metrics:
            continue
        if str(metrics.As_Of_Date.iloc[0])[:10] != day:
            continue
        if "Signal_Date" not in rec or not rec.Signal_Date.astype(str).str[:10].eq(day).all():
            continue
        from scripts.common import TICKERS
        if len(rec) != len(TICKERS) or "Ticker" not in rec or set(rec.Ticker) != set(TICKERS):
            continue
        return folder
    raise ValueError(f"{model} reported success without complete outputs for {day}")


def _last_success(previous: dict) -> str | None:
    models = previous.get("models") or {}
    currentness = previous.get("model_currentness") or {
        model: record.get("current") for model, record in models.items()
        if isinstance(record, dict)
    }
    four_systems = all(model in currentness and isinstance(currentness[model], bool) for model in MODELS)
    if previous.get("status") == "complete" and four_systems and all(currentness[model] for model in MODELS):
        return previous.get("as_of_date")
    # Legacy metadata's "complete" covered only 444/555, not all four systems.
    if four_systems or previous.get("report_schema") == REPORT_SCHEMA:
        return previous.get("last_successful_as_of_date")
    return None


def _validate_forecast(path: Path, day: str) -> None:
    from scripts.common import TICKERS
    rows = pd.read_csv(path)
    stock = "stock_id" if "stock_id" in rows else "Ticker"
    signal = "signal_date" if "signal_date" in rows else "Signal_Date"
    if (stock not in rows or signal not in rows or len(rows) != len(TICKERS)
            or set(rows[stock].astype(str)) != set(TICKERS)
            or not rows[signal].astype(str).str[:10].eq(day).all()):
        raise ValueError(f"Incomplete or stale five-stock forecast: {path}")


def _markdown(report: dict, destination: Path) -> str:
    lines = [
        "# 四套系統每日狀態", "",
        f"資料截止交易日：**{report['as_of_date']}**；執行日：{report['run_date']}。",
        f"整體狀態：**{report['status']}**。", "",
        "| 系統 | 本次狀態 | 當日結果 |", "| --- | --- | --- |",
    ]
    for model in MODELS:
        record = report["models"][model]
        output = record.get("current_output")
        link = "尚未完成"
        if record["current"] and output:
            target = os.path.relpath(ROOT / output, destination).replace(os.sep, "/")
            link = f"[查看結果]({target})"
        elif record["current"]:
            link = "已完成"
        lines.append(f"| {model} | {record['status']} | {link} |")
    lines += [
        "", "444 與 555 的績效只在相同測試日期、價格快照及評估規格下比較。",
        "555 V2 是凍結候選的預測輸出；V2.11 使用同一模型保存研究訊號、成交與成熟標籤。",
        "四套系統的 ROI 不合併排名；未完成的當日結果不以舊結果代替。", "",
    ]
    if report["comparison_status"] == "matched_test_dates_and_price_panel":
        comparison = ROOT / "reports" / "daily" / report["as_of_date"] / "comparison.md"
        target = os.path.relpath(comparison, destination).replace(os.sep, "/")
        lines += [f"[查看 444／555 同協議比較]({target})", ""]
    else:
        lines += ["當日 444／555 比較尚未完成。", ""]
    if report.get("last_successful_as_of_date") and report["status"] != "complete":
        lines += [f"最近四套系統均完成的資料日：{report['last_successful_as_of_date']}。", ""]
    return "\n".join(lines)


def build_reports(summary: dict, completed_session) -> dict:
    """Write status for partial runs, and validate all current comparison inputs."""
    day = str(pd.Timestamp(completed_session).date())
    run_date = str(summary.get("run_date") or day)
    latest = ROOT / "reports" / "latest"
    previous = _read(latest / "four_systems.json") or _read(latest / "metadata.json")
    records = {}
    native = {}
    for model in MODELS:
        step = summary.get("steps", {}).get(model, {})
        payload_path = ROOT / "outputs" / model / run_date / "status.json"
        payload = _read(payload_path)
        observed = str(payload.get("completed_session", ""))[:10]
        success = step.get("status") == "success"
        current = success and payload.get("status") == "success" and observed == day
        if success and not current:
            raise ValueError(f"{model} success is missing its current {day} status record")
        record = {
            "status": step.get("status", "not_run"),
            "exit_code": step.get("exit_code"),
            "current": current,
            "observed_as_of_date": observed or None,
            "reason": payload.get("reason") or step.get("reason"),
            "current_output": None,
        }
        if current and model in ("444", "555"):
            native[model] = _native(model, payload, day)
            record["current_output"] = f"reports/latest/{model}/next_day_recommendation.csv"
        elif current and model == "555_v2":
            signal = _inside_root(payload.get("unified_csv") or f"outputs/555_v2/{run_date}/prediction.csv")
            if not signal.is_file():
                raise ValueError("555_v2 success is missing its current prediction file")
            _validate_forecast(signal, day)
            record["current_output"] = str(signal.relative_to(ROOT.resolve()))
        elif current:
            ledger = ("research/v211/independent_ledger" if payload.get("ledger_mode") == "new_independent"
                      else "experiments/results/v2/v211")
            signal = ROOT / ledger / "daily" / f"{day}.csv"
            if not signal.is_file():
                raise ValueError("v211 success is missing its current research signal")
            _validate_forecast(signal, day)
            record["current_output"] = signal.relative_to(ROOT).as_posix()
        records[model] = record

    comparison_status = "waiting_for_both_current_notebooks"
    if set(native) == {"444", "555"}:
        from scripts.compare import compare
        from scripts.journal import write_journals
        comparison_dir = ROOT / "reports" / "daily" / day / "comparison_data"
        compare(native["444"], native["555"], comparison_dir)
        prices = ROOT / "experiments/results/v2/v211/market/prices.csv"
        write_journals(day, native["444"], native["555"], comparison_dir, price_path=prices)
        comparison_status = "matched_test_dates_and_price_panel"
        for name in ("strategy_comparison.csv", "per_stock_comparison.csv", "comparison.md", "comparison_readme.json"):
            latest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(comparison_dir / name, latest / name)

    # Publish each current notebook independently, even when its peer is waiting.
    for model, source in native.items():
        target = latest / model
        target.mkdir(parents=True, exist_ok=True)
        for name in COMPACT_FILES:
            source_file = source / name
            if source_file.is_file() and source_file.resolve() != (target / name).resolve():
                shutil.copy2(source_file, target / name)

    complete = all(record["current"] for record in records.values())
    report = {
        "report_schema": REPORT_SCHEMA,
        "as_of_date": day, "run_date": run_date,
        "started": summary.get("started"), "finished": summary.get("finished"),
        "status": "complete" if complete else "partial",
        "last_successful_as_of_date": day if complete else _last_success(previous),
        "models": records, "comparison_status": comparison_status,
        "market": summary.get("market", {}),
        "performance_protocols_combined": False,
    }
    daily = ROOT / "reports" / "daily" / day
    atomic_json(daily / "status.json", report)
    atomic_json(latest / "four_systems.json", report)
    atomic_json(latest / "metadata.json", {
        "report_schema": REPORT_SCHEMA,
        "as_of_date": day, "status": report["status"],
        "last_successful_as_of_date": report["last_successful_as_of_date"],
        "comparison_status": comparison_status,
        "model_currentness": {model: record["current"] for model, record in records.items()},
        "note": "Read four_systems.json before treating any older result file as current.",
    })
    (daily / "summary.md").write_text(_markdown(report, daily), encoding="utf-8")
    (latest / "summary.md").write_text(_markdown(report, latest), encoding="utf-8")
    return report
