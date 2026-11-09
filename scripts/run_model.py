"""Run one of the four systems. Each system writes only its own output folder."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common import TZ, current_completed_session, sha256  # noqa: E402
from scripts.unified_csv import NAMES, NOT_APPLICABLE, blank_row, write_table  # noqa: E402

EXPECTED_WEIGHTS = "53a1467fdba66e7f22a45e5f2fbc1cb78fc99825b15d516b3e7153d7bfc28144"
EXPECTED_SIGNAL = "37591d21300977b9486212fc86fe257bb050db47177fa2f41bd7a8904919f659"
SIGNAL_CREATED = "2026-11-05T23:23:09.114628+08:00"


def run_day(now=None):
    now = now or datetime.now(TZ)
    return now, pd.Timestamp(now.astimezone(TZ).date()), pd.Timestamp(current_completed_session(now)).normalize()


def ledger_mode() -> str:
    path = ROOT / "research" / "v211" / "config" / "ledger.yaml"
    mode = "imported_research"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("ledger_mode:"):
                mode = stripped.split(":", 1)[1].strip()
    if mode not in {"imported_research", "new_independent"}:
        raise SystemExit(f"ledger_mode 只能是 imported_research 或 new_independent：{mode}")
    return mode


def _market_prefix_unchanged(existing: Path, canonical: Path) -> bool:
    """The independent market must be an unchanged dated prefix of the shared one."""
    left, right = pd.read_csv(existing), pd.read_csv(canonical)
    keys = ["Date", "ticker"]
    if not set(keys).issubset(left.columns) or set(left.columns) != set(right.columns):
        return False
    left["Date"] = pd.to_datetime(left.Date).dt.normalize()
    right["Date"] = pd.to_datetime(right.Date).dt.normalize()
    if left.empty or right.empty or left.duplicated(keys).any() or right.duplicated(keys).any():
        return False
    if left.Date.max() > right.Date.max():
        return False
    prefix = right[right.Date <= left.Date.max()]
    left = left.sort_values(keys).reset_index(drop=True)
    prefix = prefix.sort_values(keys).reset_index(drop=True)[left.columns]
    try:
        pd.testing.assert_frame_equal(left, prefix, check_dtype=False, check_exact=False, rtol=0, atol=1e-8)
    except AssertionError:
        return False
    return True


def prepare_independent_ledger() -> Path:
    """Mirror shared market data without importing or changing independent decisions."""
    dest = ROOT / "research" / "v211" / "independent_ledger"
    market_src = ROOT / "experiments" / "results" / "v2" / "v211" / "market"
    market_dest = dest / "market"
    archived = ROOT / "research" / "v211" / "signal_registry" / "developer_original" / "2026-11-05.csv"
    if sha256(archived) != EXPECTED_SIGNAL:
        raise ValueError("開發者原始訊號封存已被改動，停止更新獨立市場鏡像")
    required = ("prices.csv", "prices_seed.csv", "sentiment.csv", "sentiment_seed.csv", "manifest.json", "state.json")
    optional = ("news_cache.csv", "news_cache_provenance.jsonl", "fetch_audit.jsonl", "news_scan_coverage.json")
    for name in required:
        if not (market_src / name).is_file():
            raise ValueError(f"shared market is incomplete: {name}")
    initialized = (market_dest / "manifest.json").exists()
    if initialized:
        for name in ("prices_seed.csv", "sentiment_seed.csv", "manifest.json"):
            if not (market_dest / name).is_file() or sha256(market_dest / name) != sha256(market_src / name):
                raise ValueError(f"independent market seed or manifest diverged: {name}")
        for name in ("prices.csv", "sentiment.csv"):
            if not (market_dest / name).is_file() or not _market_prefix_unchanged(market_dest / name, market_src / name):
                raise ValueError(f"independent market is newer than or divergent from shared market: {name}")
        for name in optional:
            if (market_dest / name).exists() and not (market_src / name).exists():
                raise ValueError(f"independent market has data absent from shared market: {name}")
    elif dest.exists() and any(dest.rglob("*")):
        raise ValueError("independent ledger exists without a complete market manifest; refusing to replace its history")
    # Validate every prefix before writing any mirror file. Only source/cache files
    # are refreshed; daily signals, registries, fills and scores are never copied.
    market_dest.mkdir(parents=True, exist_ok=True)
    for name in (*required, *optional):
        source = market_src / name
        if not source.exists():
            continue
        target = market_dest / name
        temporary = target.with_name(f".{name}.sync.tmp")
        shutil.copy2(source, temporary)
        temporary.replace(target)
    (dest / "daily").mkdir(parents=True, exist_ok=True)
    note_path = dest / "LEDGER_ORIGIN.json"
    if not note_path.exists():
        note = {
            "origin": "new_independent",
            "imported_developer_signal": False,
            "shared_weight_sha256": EXPECTED_WEIGHTS,
            "created_for": "this package copy only",
            "market_source": "shared canonical market; independent decisions remain separate",
        }
        temporary = note_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(note, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(note_path)
    return dest


def _write_status(model: str, day: pd.Timestamp, payload: dict) -> None:
    folder = ROOT / "outputs" / model / str(day.date())
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "status.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log_dir = ROOT / "logs" / model
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(TZ).strftime("%Y%m%dT%H%M%S%f")
    (log_dir / f"{stamp}.log").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _price_file_ready(path: Path, completed: pd.Timestamp) -> tuple[bool, str]:
    if not path.exists():
        return False, "price_snapshot_missing"
    frame = pd.read_csv(path)
    if not {"Date", "ticker", "Close"}.issubset(frame.columns):
        return False, "price_columns_missing"
    frame["Date"] = pd.to_datetime(frame["Date"]).dt.normalize()
    if frame.empty or frame.Date.max() != completed:
        return False, "prices_not_current"
    day = frame[frame.Date == completed]
    close = pd.to_numeric(day.Close, errors="coerce")
    if (len(day) != 6 or set(day.ticker.astype(str)) != set(NAMES) | {"^TWII"}
            or not np.isfinite(close).all() or not bool((close > 0).all())):
        return False, "prices_incomplete"
    if frame.duplicated(["Date", "ticker"]).any():
        return False, "prices_duplicate"
    return True, "ready"


def _sentiment_file_ready(path: Path, completed: pd.Timestamp) -> tuple[bool, str]:
    if not path.exists():
        return False, "sentiment_snapshot_missing"
    frame = pd.read_csv(path)
    if not {"Date", "ticker", "Daily_Sentiment"}.issubset(frame.columns):
        return False, "sentiment_columns_missing"
    frame["Date"] = pd.to_datetime(frame.Date).dt.normalize()
    day = frame[frame.Date == completed]
    values = pd.to_numeric(day.Daily_Sentiment, errors="coerce")
    if len(day) != 5 or set(day.ticker.astype(str)) != set(NAMES) or not np.isfinite(values).all() or not values.between(-1, 1).all():
        return False, "sentiment_incomplete"
    required = {"News_Count", "Latest_News_Timestamp", "Source_Status", "Scoring_Model"}
    if not required.issubset(day.columns):
        return False, "sentiment_provenance_missing"
    counts = pd.to_numeric(day.News_Count, errors="coerce")
    if not np.isfinite(counts).all() or (counts < 0).any() or (counts % 1 != 0).any():
        return False, "sentiment_counts_invalid"
    cutoff = completed.tz_localize(TZ) + pd.Timedelta(hours=13, minutes=30)
    for _, row in day.iterrows():
        if row.Source_Status == "verified_no_matching_article":
            if row.News_Count != 0 or row.Daily_Sentiment != 0 or row.Scoring_Model != "neutral_no_news":
                return False, "sentiment_provenance_invalid"
        elif row.Source_Status == "observed_article":
            stamp = pd.Timestamp(row.Latest_News_Timestamp)
            if (row.News_Count < 1 or row.Scoring_Model != "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment"
                    or pd.isna(stamp) or stamp.tzinfo is None or stamp.tz_convert(TZ) > cutoff):
                return False, "sentiment_provenance_invalid"
        else:
            return False, "sentiment_provenance_invalid"
    return True, "ready"


def _price_snapshot(completed: pd.Timestamp) -> tuple[Path | None, str]:
    path = ROOT / "data" / "snapshots" / str(completed.date()) / "prices.csv"
    ready, reason = _price_file_ready(path, completed)
    return (path if ready else None), reason


def _sentiment_snapshot(completed: pd.Timestamp) -> tuple[Path | None, str]:
    path = ROOT / "data" / "snapshots" / str(completed.date()) / "sentiment.csv"
    ready, reason = _sentiment_file_ready(path, completed)
    return (path if ready else None), reason


def inspect_model(model: str, now=None, fixture: str = "") -> dict:
    """Inspect inputs and hashes only. Never initialize a ledger or import TensorFlow."""
    if model not in {"444", "555", "555_v2", "v211"}:
        raise ValueError(f"unknown model {model}")
    now, today, completed = run_day(now)
    payload = {"model": model, "mode": "check", "status": "waiting", "ready": False,
               "completed_session": str(completed.date()), "run_date": str(today.date()),
               "training_executed": False, "forecast_produced": False, "signal_published": False,
               "read_only": True, "exit_code": 2}
    try:
        if fixture:
            if model not in {"444", "555"}:
                raise ValueError("reproducible fixture only supports 444 and 555")
            spec = load_fixture(fixture)
            completed = pd.Timestamp(spec["as_of"]).normalize()
            price = ROOT / spec["prices"]
            sentiment = ROOT / spec["sentiment"]
            price_ready, price_reason = _price_file_ready(price, completed)
            # The frozen historical fixture deliberately preserves unaudited legacy provenance.
            sent_ready, sent_reason = True, "frozen_fixture_hash_verified"
            payload["completed_session"] = str(completed.date())
        elif model in {"444", "555"}:
            price, price_reason = _price_snapshot(completed)
            sentiment, sent_reason = _sentiment_snapshot(completed) if model == "555" else (None, "not_used")
            price_ready, sent_ready = price is not None, model == "444" or sentiment is not None
        else:
            ledger = ROOT / "experiments" / "results" / "v2" / "v211"
            if model == "v211" and ledger_mode() == "new_independent":
                ledger = ROOT / "research" / "v211" / "independent_ledger"
            price, sentiment = ledger / "market" / "prices.csv", ledger / "market" / "sentiment.csv"
            price_ready, price_reason = _price_file_ready(price, completed)
            sent_ready, sent_reason = _sentiment_file_ready(sentiment, completed)
            weights = ROOT / "experiments" / "results" / "v2" / "v211" / "weights"
            digest = sha256(weights / "weights.npz")
            card = json.loads((weights / "model_card.json").read_text(encoding="utf-8"))
            payload["weight_sha256"] = digest
            if digest != EXPECTED_WEIGHTS or card.get("weight_sha256") != digest:
                raise ValueError("frozen weight hash mismatch")
            from scripts.model_bundle import verify_bundle

            payload["bundle_sha256"] = verify_bundle(weight_dir=weights, root=ROOT, card=card)["bundle_sha256"]
            if model == "v211":
                signal = ROOT / "experiments" / "results" / "v2" / "v211" / "daily" / "2026-11-05.csv"
                if sha256(signal) != EXPECTED_SIGNAL:
                    raise ValueError("imported signal hash mismatch")
        payload.update({"price_reason": price_reason, "sentiment_reason": sent_reason})
        for key, path in (("price_sha256", price), ("sentiment_sha256", sentiment)):
            if path is not None and path.exists():
                payload[key] = sha256(path)
        ready = bool(price_ready and (model == "444" or sent_ready))
        payload.update({"ready": ready, "status": "ready" if ready else "waiting", "exit_code": 0 if ready else 2,
                        "reason": "inputs_and_hashes_ready" if ready else (price_reason if not price_ready else sent_reason)})
    except (OSError, ValueError, KeyError, TypeError) as exc:
        payload.update({"status": "failed", "exit_code": 1, "reason": str(exc)})
    return payload


def _check(model: str, fixture: str = "") -> int:
    payload = inspect_model(model, fixture=fixture)
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return int(payload["exit_code"])


def load_fixture(name: str) -> dict:
    if name != "reproducible":
        raise SystemExit("目前只有 reproducible 測試資料")
    path = ROOT / "data" / "examples" / "reproducible" / "manifest.json"
    spec = json.loads(path.read_text(encoding="utf-8"))
    for key in ("prices", "sentiment"):
        file_path = ROOT / spec[key]
        digest = sha256(file_path)
        if digest != spec[f"{key}_sha256"]:
            raise SystemExit(f"{key} 雜湊與可重現測試資料不符：{digest}")
    return spec


def run_notebook(kind: str, mode: str, fixture: str = "") -> int:
    if mode == "check":
        return _check(kind, fixture)
    now, today, completed = run_day()
    feature = None
    feature_reason = "not_used"
    if fixture:
        spec = load_fixture(fixture)
        completed = pd.Timestamp(spec["as_of"]).normalize()
        price = ROOT / spec["prices"]
        price_reason = "reproducible_fixture"
        if kind == "555":
            feature = ROOT / spec["sentiment"]
            feature_reason = "reproducible_fixture"
        output_name = f"reproducible-{completed.date()}"
    else:
        price, price_reason = _price_snapshot(completed)
        if kind == "555":
            feature, feature_reason = _sentiment_snapshot(completed)
        output_name = str(today.date())
    ready = price is not None and (kind == "444" or feature is not None)
    payload = {
        "model": kind,
        "mode": mode,
        "status": "waiting",
        "training_executed": False,
        "forecast_produced": False,
        "completed_session": str(completed.date()),
        "run_date": str(today.date()),
        "fixture": fixture or "live_snapshot",
        "price_reason": price_reason,
        "sentiment_reason": feature_reason,
        "execution_assumption": "lstm-evaluation-v1; not the 555 V2 next-close research book",
    }
    if not ready:
        payload["reason"] = price_reason if price is None else feature_reason
        _write_status(kind, today, payload)
        print(json.dumps(payload, ensure_ascii=False), flush=True)
        return 2
    out = ROOT / "outputs" / kind / output_name
    out.mkdir(parents=True, exist_ok=True)
    os.environ["LSTM_AS_OF_DATE"] = str(completed.date())
    os.environ["LSTM_PRICE_CSV"] = str(price)
    os.environ["LSTM_OUTPUT_DIR"] = str(out)
    os.environ.setdefault("LSTM_SAVE_MODELS", "1")
    history = out / "recommendation_history" if fixture else ROOT / "outputs" / kind / "recommendation_history"
    os.environ["LSTM_RECOMMENDATION_HISTORY_DIR"] = str(history)
    if feature is not None:
        os.environ["LSTM_FEATURE_CSV"] = str(feature)
    from scripts.execute_nb import execute

    prior_recommendations = {p: p.stat().st_mtime_ns for p in out.rglob("next_day_recommendation.csv")}
    try:
        execute(kind)
        payload["training_executed"] = True
        _export_notebook_csv(kind, out, completed, price, payload, prior_recommendations)
        payload["status"] = "success"
        payload["forecast_produced"] = True
        code = 0
    except Exception as exc:
        payload.update({"status": "failed", "forecast_produced": False,
                        "reason": f"{type(exc).__name__}: {exc}"})
        code = 1
    (out / "status.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_status(kind, today, payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return code


def _export_notebook_csv(kind: str, out: Path, completed, price: Path, payload: dict, prior_recommendations=None) -> None:
    matches = [p for p in out.rglob("next_day_recommendation.csv") if "native" not in p.relative_to(out).parts]
    if prior_recommendations is not None:
        matches = [p for p in matches if p not in prior_recommendations or p.stat().st_mtime_ns != prior_recommendations[p]]
    if not matches:
        raise ValueError("notebook produced no current next_day_recommendation.csv")
    current = max(matches, key=lambda path: (path.stat().st_mtime_ns, path.as_posix()))
    frame = pd.read_csv(current)
    required = {"Ticker", "Signal_Date", "Target_Date", "Predicted_Return_Pct"}
    if not required.issubset(frame.columns):
        raise ValueError("notebook recommendation columns incomplete")
    if len(frame) != 5 or set(frame.Ticker.astype(str)) != set(NAMES):
        raise ValueError("notebook recommendation must contain exactly five distinct stocks")
    signal_dates = pd.to_datetime(frame.Signal_Date).dt.normalize()
    target_dates = pd.to_datetime(frame.Target_Date).dt.normalize()
    predicted = pd.to_numeric(frame.Predicted_Return_Pct, errors="coerce")
    if not bool((signal_dates == completed).all()) or not bool((target_dates > completed).all()) or not np.isfinite(predicted).all():
        raise ValueError("notebook recommendation is stale or contains invalid predictions")
    native = current.parent
    for name in ("strategy_metrics.csv", "evaluation_protocol.json"):
        if not (native / name).exists():
            raise ValueError(f"notebook native output missing {name}")
    compact = out / "native"
    compact.mkdir(parents=True, exist_ok=True)
    for name in ("strategy_metrics.csv", "evaluation_protocol.json", "next_day_recommendation.csv",
                 "model_metrics.csv", "backtest_nav.csv", "test_predictions.csv", "market_metrics.csv",
                 "daily_portfolio_review.csv"):
        source = native / name
        if source.exists():
            shutil.copy2(source, compact / name)
    payload["native_output_dir"] = native.relative_to(ROOT).as_posix()
    payload["daily_native_dir"] = compact.relative_to(ROOT).as_posix()
    rows = []
    for _, item in frame.iterrows():
        row = blank_row()
        row.update({
            "model_id": kind,
            "model_version": "notebook_daily_retrain",
            "signal_date": str(item.get("Signal_Date", completed.date())),
            "run_date": payload["run_date"],
            "generated_at": str(item.get("Created_At", NOT_APPLICABLE)),
            "data_asof": str(completed.date()),
            "stock_id": item["Ticker"],
            "stock_name": NAMES.get(str(item["Ticker"]), NOT_APPLICABLE),
            "predicted_return_pct": item["Predicted_Return_Pct"],
            "prediction_horizon_days": 1,
            "rank": NOT_APPLICABLE,
            "target_weight": item.get("Target_Weight", NOT_APPLICABLE),
            "cash_weight": item.get("Cash_Weight", NOT_APPLICABLE),
            "risk_status": item["Risk_Off"] if "Risk_Off" in frame.columns else NOT_APPLICABLE,
            "intended_execution_date": item.get("Target_Date", NOT_APPLICABLE),
            "execution_price": NOT_APPLICABLE,
            "execution_status": NOT_APPLICABLE,
            "model_hash": NOT_APPLICABLE,
            "data_hash": sha256(price),
        })
        rows.append(row)
    destination = out / "unified_signal.csv"
    write_table(destination, rows, {
        "model_id": kind,
        "execution_assumption": payload["execution_assumption"],
        "native_column_copied": "Predicted_Return_Pct",
        "values_were_rescaled": False,
        "research_sample": False,
    })
    payload["unified_csv"] = str(destination.relative_to(ROOT)).replace("\\", "/")


def v2_date_record(signal_date: str, run_date: str, execution_date: str) -> dict:
    """Keep the four dates in separate fields. Data cutoff stays on the signal date."""
    signal_date = str(pd.Timestamp(signal_date).date())
    run_date = str(pd.Timestamp(run_date).date())
    execution_date = str(pd.Timestamp(execution_date).date())
    if signal_date == execution_date:
        raise AssertionError("預定成交日不能等於訊號日")
    return {
        "signal_date": signal_date,
        "run_date": run_date,
        "data_asof": signal_date,
        "intended_execution_date": execution_date,
        "signal_date_meaning": "訊號日：已完成交易日，特徵與收盤只到這一天",
        "run_date_meaning": "執行日：程式寫出這份檔案的台北日曆日",
        "data_asof_meaning": "資料截止日：允許進入模型的最後資料日，等於訊號日",
        "intended_execution_date_meaning": "預定成交日：下一個交易日收盤，這份預測尚未成交",
    }


def _frozen_card() -> tuple[str, dict]:
    folder = ROOT / "experiments" / "results" / "v2" / "v211" / "weights"
    digest = sha256(folder / "weights.npz")
    card = json.loads((folder / "model_card.json").read_text(encoding="utf-8"))
    if digest != EXPECTED_WEIGHTS or card.get("weight_sha256") != digest:
        raise ValueError("frozen weight hash mismatch; training is prohibited in the daily runner")
    from scripts.model_bundle import verify_bundle

    verify_bundle(weight_dir=folder, root=ROOT, card=card)
    return digest, card


def run_555_v2(mode: str) -> int:
    """Predict with the frozen weights. This does not publish a V2.11 signal."""
    if mode == "check":
        return _check("555_v2")
    os.environ.pop("TW_LSTM_V211_DIR", None)
    now, today, completed = run_day()
    import experiments.v2.v211 as tracker
    from experiments.v2.v211_market import PRICES, SENTIMENT, market_files_ready, prediction_gate

    payload = {
        "model": "555_v2",
        "mode": mode,
        "status": "waiting",
        "research_candidate": True,
        "official_555_replacement": False,
        "signal_published": False,
        "completed_session": str(completed.date()),
        "run_date": str(today.date()),
    }
    if not market_files_ready():
        payload["reason"] = "v211_market_not_ready"
        _write_status("555_v2", today, payload)
        print(json.dumps(payload, ensure_ascii=False), flush=True)
        return 2
    ready, reason = prediction_gate(pd.read_csv(PRICES), pd.read_csv(SENTIMENT), completed)
    if not ready:
        payload["reason"] = reason
        _write_status("555_v2", today, payload)
        print(json.dumps(payload, ensure_ascii=False), flush=True)
        return 2
    weight_path = tracker.WEIGHT_DIR / "weights.npz"
    digest = sha256(weight_path)
    if digest != EXPECTED_WEIGHTS:
        payload["status"] = "failed"
        payload["reason"] = "weight hash mismatch"
        payload["weight_sha256"] = digest
        _write_status("555_v2", today, payload)
        print(json.dumps(payload, ensure_ascii=False), flush=True)
        return 1
    from experiments.v2.v211_market import load_market_context

    context = load_market_context()
    _digest, card = _frozen_card()
    from scripts.model_bundle import verify_bundle

    bundle = verify_bundle(weight_dir=tracker.WEIGHT_DIR, root=ROOT, card=card)
    if not (tracker.WEIGHT_DIR / "model_card.json").exists() or sha256(weight_path) != digest:
        payload["status"] = "failed"
        payload["reason"] = "weights changed during reuse"
        _write_status("555_v2", today, payload)
        return 1
    rows, execution, _five = tracker.predict_rows(card, context, completed)
    dates = v2_date_record(str(completed.date()), str(today.date()), execution)
    dates.update(tracker.execution_eligibility(completed))
    dates["intended_execution_date_meaning"] = "相容舊欄位：預測標籤的假設進場日；實際交易須另看 trade_eligible 與 scheduled_rebalance_date"
    out = ROOT / "outputs" / "555_v2" / str(today.date())
    out.mkdir(parents=True, exist_ok=True)
    export = rows.copy()
    export.insert(0, "Signal_Date", dates["signal_date"])
    export.insert(1, "Run_Date", dates["run_date"])
    export.insert(2, "Data_AsOf", dates["data_asof"])
    export.insert(3, "Planned_Execution_Date", dates["intended_execution_date"])
    export.to_csv(out / "prediction.csv", index=False, encoding="utf-8-sig")
    (out / "dates.json").write_text(json.dumps(dates, ensure_ascii=False, indent=2), encoding="utf-8")
    unified = []
    for _, item in rows.iterrows():
        row = blank_row()
        row.update({
            "model_id": "555_v2",
            "model_version": tracker.MODEL_VERSION,
            "signal_date": dates["signal_date"],
            "run_date": dates["run_date"],
            "generated_at": now.astimezone(TZ).isoformat(),
            "data_asof": dates["data_asof"],
            "stock_id": item["Ticker"],
            "stock_name": NAMES[item["Ticker"]],
            "predicted_return_pct": item["Predicted_Return_Pct"],
            "prediction_horizon_days": 5,
            "rank": int(item["Rank"]),
            "target_weight": item["Target_Weight"],
            "cash_weight": item["Cash_Weight"],
            "risk_status": "Risk_Off" if bool(item["Risk_Off"]) else "Risk_On",
            "intended_execution_date": dates["forecast_entry_date"] if dates["trade_eligible"] else NOT_APPLICABLE,
            "forecast_entry_date": dates["forecast_entry_date"],
            "scheduled_rebalance_date": dates["scheduled_rebalance_date"],
            "trade_eligible": dates["trade_eligible"],
            "execution_price": NOT_APPLICABLE,
            "execution_status": "eligible_unfilled" if dates["trade_eligible"] else "forecast_only",
            "model_hash": digest,
            "data_hash": sha256(PRICES),
        })
        unified.append(row)
    write_table(out / "unified_signal.csv", unified, {
        "model_id": "555_v2",
        "execution_assumption": "next_session_close; rebalance every 5 sessions; risk-off gross cap 0.50",
        "prediction_horizon_days_meaning": "model label is a five-session return; the book fills at the next close",
        "research_sample": False,
        "v211_signal_written": False,
        "dates": dates,
        "model_bundle_sha256": bundle["bundle_sha256"],
        "note": "This CSV is a prediction export. Only the V2.11 ledger file is a research signal.",
    })
    payload.update({
        "status": "success",
        "reason": "prediction_written_not_a_research_signal",
        "weight_sha256": digest,
        "weights_reused": True,
        "dates": dates,
        "planned_execution": execution,
        "unified_csv": str((out / "unified_signal.csv").relative_to(ROOT)).replace("\\", "/"),
    })
    _write_status("555_v2", today, payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return 0


def run_v211(mode: str) -> int:
    if mode == "check":
        return _check("v211")
    now, today, completed = run_day()
    _frozen_card()  # Refuse missing or inconsistent artifacts before launching the tracker.
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    mode_name = ledger_mode()
    if mode_name == "new_independent":
        dest = prepare_independent_ledger()
        env["TW_LSTM_V211_DIR"] = str(dest)
        env["TW_LSTM_WEIGHT_DIR"] = str(ROOT / "experiments" / "results" / "v2" / "v211" / "weights")
    else:
        env.pop("TW_LSTM_V211_DIR", None)
        env.pop("TW_LSTM_WEIGHT_DIR", None)
    # A parent orchestrator may already have updated the shared market once.
    # Preserve its skip flag so this child does not download sources again.
    signal = ROOT / "experiments" / "results" / "v2" / "v211" / "daily" / "2026-11-05.csv"
    before = sha256(signal)
    if before != EXPECTED_SIGNAL:
        payload = {"model": "v211", "status": "failed", "reason": "imported signal hash mismatch", "sha256": before}
        _write_status("v211", today, payload)
        print(json.dumps(payload, ensure_ascii=False), flush=True)
        return 1
    completed_run = subprocess.run(
        [sys.executable, "-m", "experiments.v2.v211_daily"],
        cwd=str(ROOT),
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    after = sha256(signal)
    text = (completed_run.stdout or "") + ("\n" + completed_run.stderr if completed_run.stderr else "")
    log_dir = ROOT / "logs" / "v211"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(TZ).strftime("%Y%m%dT%H%M%S%f")
    (log_dir / f"{stamp}.log").write_text(text + f"\nexit {completed_run.returncode}\n", encoding="utf-8")
    payload = {
        "model": "v211",
        "mode": mode,
        "ledger_mode": mode_name,
        "status": "success" if completed_run.returncode == 0 else ("waiting" if completed_run.returncode == 2 else "failed"),
        "exit_code": int(completed_run.returncode),
        "completed_session": str(completed.date()),
        "run_date": str(today.date()),
        "imported_signal_sha256_before": before,
        "imported_signal_sha256_after": after,
        "imported_signal_unchanged": before == after,
        "imported_signal_created_at": SIGNAL_CREATED,
        "signal_origin": "developer_original_2026-11-05",
        "official_555_replacement": False,
        "stdout_tail": text[-2000:],
    }
    _write_status("v211", today, payload)
    print(json.dumps({key: value for key, value in payload.items() if key != "stdout_tail"}, ensure_ascii=False), flush=True)
    if after != before:
        return 1
    return int(completed_run.returncode)


def run_paired_555(mode: str) -> int:
    """Use this session's saved 444 core; do not fit a different comparison model."""
    if mode == "check":
        return _check("555")
    from scripts.paired_delivery import build_paired_report
    from scripts.paired_native import export_paired_native
    from scripts.paired_research import paired_daily_recommendation, run_paired_research

    now, today, completed = run_day()
    price, price_reason = _price_snapshot(completed)
    sentiment, sentiment_reason = _sentiment_snapshot(completed)
    baseline = ROOT / "outputs/444" / str(today.date()) / "native"
    baseline_status = ROOT / "outputs/444" / str(today.date()) / "status.json"
    status444 = json.loads(baseline_status.read_text(encoding="utf-8")) if baseline_status.is_file() else {}
    payload = {"model": "555", "mode": mode, "status": "waiting", "core_model": "444",
               "training_executed": False, "forecast_produced": False,
               "completed_session": str(completed.date()), "run_date": str(today.date()),
               "research_protocol": "paired-444-core-allocation-v1",
               "execution_assumption": "signal-day allocation; next-session close; common costs",
               "research_status": "exploratory", "historical_news_point_in_time_proven": False}
    if (price is None or sentiment is None or status444.get("status") != "success"
            or status444.get("completed_session") != str(completed.date())):
        payload["reason"] = price_reason if price is None else sentiment_reason if sentiment is None else "waiting_for_current_444_core"
        _write_status("555", today, payload)
        print(json.dumps(payload, ensure_ascii=False), flush=True)
        return 2
    protocol = json.loads((baseline / "evaluation_protocol.json").read_text(encoding="utf-8"))
    result = run_paired_research(baseline / "test_predictions.csv", price, sentiment, protocol)
    recommendation = paired_daily_recommendation(baseline / "next_day_recommendation.csv", price, sentiment, protocol)
    out = ROOT / "outputs/555" / str(today.date())
    native = out / "native"
    export = export_paired_native(result, recommendation, baseline, native, completed, datetime.now(TZ))
    research = build_paired_report(baseline, price, sentiment, ROOT / "reports/daily" / str(completed.date()) / "paired", root=ROOT)
    rec = pd.read_csv(native / "next_day_recommendation.csv")
    # Separate method histories: never blend these with the old six-feature 555.
    history = ROOT / "outputs/555/paired_recommendation_history"
    history.mkdir(parents=True, exist_ok=True)
    history_path = history / f"{export['run_id']}.csv"
    if history_path.exists():
        raise FileExistsError(history_path)
    shutil.copy2(native / "next_day_recommendation.csv", history_path)
    rows = []
    for item in rec.to_dict("records"):
        row = blank_row()
        row.update({"model_id": "555", "model_version": payload["research_protocol"],
                    "signal_date": item["Signal_Date"], "run_date": str(today.date()),
                    "generated_at": item["Created_At"], "data_asof": str(completed.date()),
                    "stock_id": item["Ticker"], "stock_name": NAMES[item["Ticker"]],
                    "predicted_return_pct": item["Predicted_Return_Pct"], "prediction_horizon_days": 1,
                    "target_weight": item["Target_Weight"], "cash_weight": item["Cash_Weight"],
                    "risk_status": "Risk_Off" if bool(item["Risk_Off"]) else "Risk_On",
                    "forecast_entry_date": item["Target_Date"], "scheduled_rebalance_date": item["Target_Date"],
                    "trade_eligible": True, "intended_execution_date": item["Target_Date"],
                    "execution_status": "research_target_only_no_order", "data_hash": sha256(price)})
        rows.append(row)
    unified = out / "unified_signal.csv"
    write_table(unified, rows, {"model_id": "555", "core_model": "444", "research_sample": False,
                              "execution_assumption": payload["execution_assumption"],
                              "prediction_target": "444 next-day model label; not a forecast of post-fill holding profit",
                              "sentiment_sha256": sha256(sentiment), "core_source_sha256": export["core_source_sha256"],
                              "original_six_feature_555": "available through --legacy-555; a separate secondary study"})
    payload.update({"status": "success", "forecast_produced": True,
                    "reason": "shared_444_core_dynamic_news_risk_exported", "primary_research": research,
                    "native_output_dir": native.relative_to(ROOT).as_posix(),
                    "daily_native_dir": native.relative_to(ROOT).as_posix(),
                    "unified_csv": unified.relative_to(ROOT).as_posix()})
    _write_status("555", today, payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return 0


def run(model: str, mode: str, fixture: str = "", legacy_555: bool = False) -> int:
    if mode == "check":
        return _check(model, fixture)
    os.chdir(ROOT)
    if fixture and model not in {"444", "555"}:
        raise SystemExit("reproducible 測試資料只供 444 與 555 使用")
    try:
        if model == "555" and not fixture and not legacy_555:
            return run_paired_555(mode)
        if model in {"444", "555"}:
            return run_notebook(model, mode, fixture)
        if model == "555_v2":
            return run_555_v2(mode)
        if model == "v211":
            return run_v211(mode)
        raise ValueError(f"unknown model {model}")
    except Exception as exc:
        _now, today, completed = run_day()
        payload = {"model": model, "mode": mode, "status": "failed", "exit_code": 1,
                   "reason": f"{type(exc).__name__}: {exc}", "forecast_produced": False,
                   "completed_session": str(completed.date()), "run_date": str(today.date())}
        _write_status(model, today, payload)
        print(json.dumps(payload, ensure_ascii=False), flush=True)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["444", "555", "555_v2", "v211"])
    parser.add_argument("--mode", choices=["check", "full"], default="full")
    parser.add_argument("--fixture", default="", choices=["", "reproducible"])
    parser.add_argument("--legacy-555", action="store_true", help="Run the preserved six-feature 555 as a separate historical/extension diagnostic")
    args = parser.parse_args()
    return run(args.model, args.mode, args.fixture, legacy_555=args.legacy_555)


if __name__ == "__main__":
    raise SystemExit(main())
