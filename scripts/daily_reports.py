"""Publish current four-system status and the separate 444/555 comparison."""
from __future__ import annotations

import json
import hashlib
import os
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from scripts.common import ROOT, TZ, atomic_json

MODELS = ("444", "555", "555_v2", "v211")
REPORT_SCHEMA = "four-systems-v1"
ATTEMPT_SCHEMA = "daily-research-attempt-v1"
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


def _news_summary(day: str) -> dict:
    """Summarize observed scan coverage without publishing article content."""
    market = ROOT / "experiments/results/v2/v211/market"
    coverage = _read(market / "news_scan_coverage.json")
    calendar = coverage.get("calendar_days") or {}
    sessions = coverage.get("signal_sessions") or {}
    observed = sessions.get(day) or {}
    latest = max(calendar, default=None)
    return {
        "provider": "cnyes",
        "calendar_scan_through": latest,
        "signal_session": day,
        "signal_session_scan_complete": day in sessions,
        "source_start": observed.get("source_start"),
        "source_end": observed.get("source_end"),
        "completed_at": observed.get("completed_at"),
        "articles_on_signal_calendar_day": (calendar.get(day) or {}).get("articles"),
        "article_text_included": False,
        "historical_availability_retroactively_certified": False,
    }


def _attempt_path(summary: dict, day: str) -> Path:
    run_date = str(pd.Timestamp(summary.get("run_date") or day).date())
    started = pd.Timestamp(summary.get("started") or datetime.now(TZ))
    if started.tzinfo is None:
        raise ValueError("A daily attempt timestamp must contain a timezone")
    # The same invocation is idempotent; a later invocation has its own actual
    # start time. Do not use wall-clock report rendering time as a fake signal.
    core = {"as_of_date": day, "summary": summary}
    raw = json.dumps(core, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode()
    identity = hashlib.sha256(raw).hexdigest()[:16]
    name = started.tz_convert(TZ).strftime("%Y%m%dT%H%M%S%f%z") + "_" + identity + ".json"
    return ROOT / "reports/attempts" / run_date / name


def _write_attempt(summary: dict, day: str, report: dict, path: Path) -> None:
    """Append one immutable actual-run record, then refresh its derived index."""
    payload = {
        "schema": ATTEMPT_SCHEMA,
        "as_of_date": day,
        "run_date": str(pd.Timestamp(summary.get("run_date") or day).date()),
        "output_day": str(summary.get("output_day") or day),
        "started": summary.get("started"),
        "finished": summary.get("finished"),
        "delayed_run": bool(summary.get("delayed_run") or str(summary.get("run_date") or day) != day),
        "attempt_status": summary.get("status") or report.get("status"),
        "reason": summary.get("reason"),
        "steps": summary.get("steps") or {},
        "report": report,
        "market": summary.get("market") or {},
        "historical_signal_backfill_performed": False,
    }
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError("A recorded daily attempt cannot be overwritten")
    else:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
    lines = [f"# {payload['run_date']} 實際執行紀錄", "",
             "每筆記錄保留實際執行時間、資料日與完成／等待原因；延後重建不冒充資料日已發布。", "",
             "| 啟動時間 | 資料日 | 嘗試狀態 | 444 | 555 | 紀錄 |",
             "| --- | --- | --- | --- | --- | --- |"]
    for item in sorted(path.parent.glob("*.json"), reverse=True):
        value = _read(item)
        steps = value.get("steps") or {}
        values = [str(value.get("started") or "未記錄"), value["as_of_date"],
                  str(value.get("attempt_status") or "unknown"),
                  str((steps.get("444") or {}).get("status") or "not_run"),
                  str((steps.get("555") or {}).get("status") or "not_run")]
        lines.append("| " + " | ".join(text.replace("|", "\\|") for text in values)
                     + f" | [JSON]({item.name}) |")
    (path.parent / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _status_path(summary: dict, model: str, day: str, run_date: str) -> Path:
    step = (summary.get("steps") or {}).get(model) or {}
    if step.get("status_path"):
        return _inside_root(str(step["status_path"]))
    if step.get("output_dir"):
        return _inside_root(str(step["output_dir"])) / "status.json"
    output_day = str(summary.get("output_day") or day)
    choices = list(dict.fromkeys((output_day, run_date)))
    paths = [ROOT / "outputs" / model / key / "status.json" for key in choices]
    # Older versions stored by calendar run date. Only fall back when the
    # as-of keyed status is absent, never to an older/different session record.
    return next((path for path in paths if path.is_file()), paths[0])


def _preserve_reason(summary: dict, day: str, previous: dict, daily_previous: dict, records: dict) -> str | None:
    if not summary.get("steps") or summary.get("status") == "skipped":
        return "empty_or_skipped_attempt_preserves_published_results"
    if previous.get("as_of_date") and day < str(previous["as_of_date"]):
        return "older_session_cannot_replace_latest_report"
    old = daily_previous or (previous if previous.get("as_of_date") == day else {})
    old_models = old.get("models") or {}
    if old.get("status") == "complete" and not all(record["current"] for record in records.values()):
        return "retry_cannot_replace_completed_session_report"
    if old.get("status") == "complete" and old.get("as_of_date") == day and _last_success(old) == day:
        return "already_published_completed_session_report"
    if all((old_models.get(model) or {}).get("current") for model in ("444", "555")) and not all(
            records[model]["current"] for model in ("444", "555")):
        return "retry_cannot_erase_completed_444_555_report"
    currentness_improved = any(records[model]["current"] and not (old_models.get(model) or {}).get("current")
                               for model in MODELS)
    if not currentness_improved and all((old_models.get(model) or {}).get("current") for model in ("444", "555")) and all(
            ((summary.get("steps") or {}).get(model) or {}).get("reason") == "already_completed_for_this_session"
            for model in ("444", "555")):
        return "already_published_444_555_for_this_session"
    return None


def _annotate_delayed_journals(day: str, summary: dict, records: dict) -> None:
    """Keep next-day research tables honest when the scheduled run arrived late."""
    if str(summary.get("run_date") or day) == day and not summary.get("delayed_run"):
        return
    marker = "<!-- delayed-run-timing -->"
    end_marker = "<!-- /delayed-run-timing -->"
    for model in ("444", "555"):
        if not records[model]["current"]:
            continue
        record = records[model]
        generated = record.get("generated_at") or summary.get("finished") or summary.get("started") or summary.get("run_date")
        if record.get("forecast_timing") == "expired_entry":
            note = "原預定進場時點已過，下表只作延後重建的研究預測，不計為可成交的新訊號或前瞻命中。"
        elif record.get("forecast_timing") == "future_entry":
            note = "生成時原預定進場尚未到，仍僅為研究輸出，未下單或成交。"
        else:
            note = "須依實際生成時間核對原目標時點，未認證為資料日已存在的前瞻訊號。"
        block = (marker + "\n\n" + f"**延後執行註記：**資料日{day}；實際生成／執行時間{generated}。"
                 + note + "\n\n" + end_marker + "\n")
        daily = ROOT / "reports/daily" / day / f"{model}.md"
        journal = ROOT / "reports/journals" / f"{model}.md"
        for path in (daily, journal):
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            section = text.find(f"## {day} — {model}")
            if section < 0:
                continue
            start = text.find("\n", section)
            finish = text.find("\n## ", start + 1)
            finish = len(text) if finish < 0 else finish
            body = text[start + 1:finish]
            if marker in body:
                before, existing = body.split(marker, 1)
                if end_marker not in existing:
                    raise ValueError("Incomplete delayed journal timing annotation")
                body = before + existing.split(end_marker, 1)[1].lstrip("\n")
            updated = text[:start + 1] + "\n" + block + "\n" + body.lstrip("\n") + text[finish:]
            path.write_text(updated, encoding="utf-8")


def _write_independent_journal(model: str, source: Path, day: str, summary: dict) -> None:
    """A successful model still has a journal when its comparison peer waits."""
    destination = ROOT / "reports/daily" / day / f"{model}.md"
    if destination.exists():
        return  # A validated paired/previous report is not replaced by a fallback.
    metrics = pd.read_csv(source / "strategy_metrics.csv").iloc[0]
    recommendation = pd.read_csv(source / "next_day_recommendation.csv")
    import math
    values = {name: float(metrics[name]) for name in ("ROI_Pct", "Sharpe", "MDD_Pct")}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError(f"{model} has a non-finite independent journal metric")
    marker = f"## {day} — {model} 獨立完成紀錄"
    lines = [marker, "", f"資料日：{day}；實際執行日：{summary.get('run_date') or day}。",
             "另一模型或共同核心比較尚未完成；以下為此模型原生研究輸出，不與不同協議合併排名。", "",
             f"原生研究ROI {values['ROI_Pct']:+.2f}%；Sharpe {values['Sharpe']:.3f}；MDD {values['MDD_Pct']:.2f}%。", "",
             "### 保存的下一期研究目標（不代表可成交委託）", "",
             "| 股票 | 預測報酬 | 目標權重 |", "| --- | ---: | ---: |"]
    for row in recommendation.itertuples():
        prediction, weight = float(row.Predicted_Return_Pct), float(row.Target_Weight)
        if not math.isfinite(prediction) or not math.isfinite(weight):
            raise ValueError(f"{model} has a non-finite independent recommendation")
        lines.append(f"| {row.Ticker} | {prediction:+.3f}% | {weight:.2%} |")
    lines += ["", "研究輸出未下單或成交；方向準確率不當作逐筆交易勝率。", ""]
    text = "\n".join(lines)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    journal = ROOT / "reports/journals" / f"{model}.md"
    journal.parent.mkdir(parents=True, exist_ok=True)
    old = journal.read_text(encoding="utf-8") if journal.exists() else f"# {model} 研究日誌\n"
    if marker not in old:
        title, _, body = old.partition("\n")
        journal.write_text(title + "\n\n" + text + "\n" + body, encoding="utf-8")


def _markdown(report: dict, destination: Path) -> str:
    lines = [
        "# 四套系統每日狀態", "",
        f"資料截止交易日：**{report['as_of_date']}**；執行日：{report['run_date']}。",
        f"整體狀態：**{report['status']}**；444／555 主研究：**{report.get('research_status', 'partial')}**。", "",
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
    if report.get("delayed_run"):
        lines += ["", f"延後執行：資料日為{report['as_of_date']}，實際啟動為{report.get('started') or report['run_date']}。",
                  "本次輸出不冒充於資料日當晚發布；若預定進場已過，預測僅供研究，不列為可成交的新訊號。"]
    if any(record.get("maintenance_only") for record in report["models"].values()):
        lines += ["", "V2.11 本次只維護已保存訊號的成交／成熟標籤，沒有補寫過去訊號；不影響444／555主研究完成狀態。"]
    expired = [model for model, record in report["models"].items() if record.get("forecast_timing") == "expired_entry"]
    if expired:
        lines += ["", "已過原預定進場時點的研究輸出：" + "、".join(expired) + "；不標示為可執行的新建議。"]
    news = report.get("news") or {}
    lines += ["", "新聞掃描：最近來源日 " + str(news.get("calendar_scan_through") or "未完成／未記錄")
              + "；本資料日覆蓋 " + ("已完成" if news.get("signal_session_scan_complete") else "未完成／未記錄")
              + "。來源失敗與確實無匹配文章分開，不用昨日分數冒充今日新聞。"]
    lines += [
        "", "444 與 555 的績效只在相同測試日期、價格快照及評估規格下比較。",
        "555 V2 是凍結候選的預測輸出；V2.11 使用同一模型保存研究訊號、成交與成熟標籤。",
        "四套系統的 ROI 不合併排名；未完成的當日結果不以舊結果代替。", "",
    ]
    paired = report.get("primary_research") or {}
    if paired.get("status") == "complete":
        target = os.path.relpath(ROOT / paired["report"], destination).replace(os.sep, "/")
        lines += [f"**主研究：[444 共同核心＋555 動態權重／新聞／風控消融]({target})**。原六特徵 555 比較為延伸診斷。", ""]
    else:
        lines += [f"共同核心主研究尚未完成：{paired.get('reason', 'waiting_for_444_and_shared_market')}。", ""]
    if report["comparison_status"] in {"matched_test_dates_and_price_panel", "paired_444_core_next_close"}:
        comparison = ROOT / "reports" / "daily" / report["as_of_date"] / "comparison.md"
        target = os.path.relpath(comparison, destination).replace(os.sep, "/")
        label = "共同核心、次日收盤比較" if report["comparison_status"] == "paired_444_core_next_close" else "舊六特徵模型之描述性比較"
        lines += [f"[查看 444／555 {label}]({target})", ""]
    else:
        lines += ["當日 444／555 比較尚未完成。", ""]
    if report.get("last_successful_as_of_date") and report["status"] != "complete":
        lines += [f"最近四套系統均完成的資料日：{report['last_successful_as_of_date']}。", ""]
    if report.get("attempt_journal"):
        target = os.path.relpath(ROOT / report["attempt_journal"], destination).replace(os.sep, "/")
        lines += [f"[本次實際執行紀錄]({target})", ""]
    return "\n".join(lines)


def _build_reports(summary: dict, completed_session, *, attempt_path: Path) -> dict:
    """Write status for partial runs, and validate all current comparison inputs."""
    day = str(pd.Timestamp(completed_session).date())
    run_date = str(summary.get("run_date") or day)
    latest = ROOT / "reports" / "latest"
    previous = _read(latest / "four_systems.json") or _read(latest / "metadata.json")
    daily = ROOT / "reports" / "daily" / day
    daily_previous = _read(daily / "status.json")
    records = {}
    native = {}
    for model in MODELS:
        step = summary.get("steps", {}).get(model, {})
        payload_path = _status_path(summary, model, day, run_date)
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
            "maintenance_only": bool(step.get("maintenance_only") or payload.get("maintenance_only")
                                     or step.get("status") == "maintenance_only" or payload.get("status") == "maintenance_only"),
            "forecast_timing": payload.get("forecast_timing") or step.get("forecast_timing"),
            "recommendation_execution_status": payload.get("recommendation_execution_status") or step.get("recommendation_execution_status"),
            "generated_at": payload.get("generated_at"),
        }
        if current and model in ("444", "555"):
            native[model] = _native(model, payload, day)
            record["current_output"] = f"reports/latest/{model}/next_day_recommendation.csv"
        elif current and model == "555_v2":
            signal = _inside_root(payload.get("unified_csv") or str(payload_path.parent.relative_to(ROOT) / "prediction.csv"))
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

    preserved = _preserve_reason(summary, day, previous, daily_previous, records)
    if preserved:
        # Every later retry is journaled without rebuilding an already complete
        # session from changed inputs or moving the latest report backwards.
        return {
            "report_schema": REPORT_SCHEMA, "as_of_date": day, "run_date": run_date,
            "output_day": str(summary.get("output_day") or day),
            "started": summary.get("started"), "finished": summary.get("finished"),
            "delayed_run": bool(summary.get("delayed_run") or day != run_date),
            "status": ("skipped" if summary.get("status") == "skipped" or not summary.get("steps")
                       else "complete" if all(record["current"] for record in records.values()) else "partial"),
            "research_status": "complete" if all(records[model]["current"] for model in ("444", "555")) else "partial",
            "models": records, "publication_status": "attempt_only", "preserved_reason": preserved,
            "last_successful_as_of_date": _last_success(previous),
            "comparison_status": "not_recomputed_for_preserved_attempt",
            "market": summary.get("market") or {}, "news": _news_summary(day),
            "attempt_journal": attempt_path.relative_to(ROOT).as_posix(),
            "performance_protocols_combined": False,
        }

    comparison_status = "waiting_for_both_current_notebooks"
    primary_research = None
    if set(native) == {"444", "555"}:
        comparison_dir = ROOT / "reports" / "daily" / day / "comparison_data"
        if _read(native["555"] / "evaluation_protocol.json").get("protocol_version") == "paired-444-core-allocation-v1":
            from scripts.paired_delivery import build_paired_report, publish_comparison_contract

            market = ROOT / "experiments/results/v2/v211/market"
            paired_dir = ROOT / "reports/daily" / day / "paired"
            primary_research = build_paired_report(native["444"], market / "prices.csv", market / "sentiment.csv", paired_dir, root=ROOT)
            publish_comparison_contract(paired_dir, native["444"], comparison_dir, root=ROOT)
            comparison_status = "paired_444_core_next_close"
        else:
            from scripts.compare import compare
            from scripts.journal import write_journals

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

    if primary_research is None:
        primary_research = {"status": "waiting", "reason": "waiting_for_444_and_shared_market"}
    if primary_research.get("status") != "complete" and "444" in native:
        market = ROOT / "experiments/results/v2/v211/market"
        needed = (native["444"] / "test_predictions.csv", market / "prices.csv", market / "sentiment.csv")
        if all(path.is_file() for path in needed):
            from scripts.paired_delivery import build_paired_report

            primary_research = build_paired_report(native["444"], market / "prices.csv", market / "sentiment.csv",
                                                   ROOT / "reports/daily" / day / "paired", root=ROOT)
            atomic_json(latest / "paired_research.json", primary_research)
        else:
            primary_research["reason"] = "paired_source_predictions_or_market_missing"
    if primary_research.get("status") == "complete":
        atomic_json(latest / "paired_research.json", primary_research)

    for model, source in native.items():
        _write_independent_journal(model, source, day, summary)
    _annotate_delayed_journals(day, summary, records)

    complete = all(record["current"] for record in records.values())
    report = {
        "report_schema": REPORT_SCHEMA,
        "as_of_date": day, "run_date": run_date,
        "output_day": str(summary.get("output_day") or day),
        "started": summary.get("started"), "finished": summary.get("finished"),
        "delayed_run": bool(summary.get("delayed_run") or day != run_date),
        "status": "complete" if complete else "partial",
        "research_status": "complete" if all(records[model]["current"] for model in ("444", "555")) else "partial",
        "last_successful_as_of_date": day if complete else _last_success(previous),
        "models": records, "comparison_status": comparison_status,
        "market": summary.get("market", {}),
        "news": _news_summary(day),
        "attempt_journal": attempt_path.relative_to(ROOT).as_posix(),
        "publication_status": "published",
        "performance_protocols_combined": False,
        "primary_research": primary_research,
    }
    atomic_json(daily / "status.json", report)
    atomic_json(latest / "four_systems.json", report)
    atomic_json(latest / "metadata.json", {
        "report_schema": REPORT_SCHEMA,
        "as_of_date": day, "status": report["status"],
        "last_successful_as_of_date": report["last_successful_as_of_date"],
        "comparison_status": comparison_status,
        "research_status": report["research_status"],
        "delayed_run": report["delayed_run"],
        "model_currentness": {model: record["current"] for model, record in records.items()},
        "note": "Read four_systems.json before treating any older result file as current.",
    })
    (daily / "summary.md").write_text(_markdown(report, daily), encoding="utf-8")
    (latest / "summary.md").write_text(_markdown(report, latest), encoding="utf-8")
    return report


def build_reports(summary: dict, completed_session) -> dict:
    """Publish validated session outputs and always journal the actual attempt."""
    summary = dict(summary)
    if not summary.get("started"):
        summary["started"] = datetime.now(TZ).isoformat()
    day = str(pd.Timestamp(completed_session).date())
    path = _attempt_path(summary, day)
    if path.exists():
        recorded = _read(path).get("report")
        if not isinstance(recorded, dict) or recorded.get("publication_status") not in {"published", "attempt_only"}:
            raise ValueError("A recorded daily attempt cannot be overwritten; start a new attempt")
        # Replaying the same invocation must not regenerate reports from newer
        # native files, nor turn its original publication into an attempt-only
        # result. Verify the complete saved payload before returning it.
        _write_attempt(summary, day, recorded, path)
        return recorded
    try:
        report = _build_reports(summary, day, attempt_path=path)
    except Exception as exc:
        failed = {"status": "failed", "reason": "report_validation_failed", "error": str(exc),
                  "as_of_date": day, "run_date": str(summary.get("run_date") or day),
                  "publication_status": "rejected"}
        _write_attempt(summary, day, failed, path)
        raise
    _write_attempt(summary, day, report, path)
    return report
