"""Read-only prospective display layer for local V2.11 and Ridge snapshots."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import os

from dashboard_v2.data_access.load import NAMES, csv_rows, json_file, stable_text

TAIPEI = ZoneInfo("Asia/Taipei")
ROOT = Path(__file__).resolve().parents[2]
PKG = Path(__file__).resolve().parents[1]
SNAP = PKG / "fixtures" / "display_snapshots"
# Prefer in-repo frozen V2.11; optional live overlays via env (never required for CI).
LOCAL_V211 = Path(
    os.environ.get(
        "TW_V211_LOCAL",
        str(ROOT / "experiments" / "results" / "v2" / "v211"),
    )
)
RIDGE = Path(os.environ["TW_RIDGE_ROOT"]) if os.environ.get("TW_RIDGE_ROOT") else None
GITHUB = Path(os.environ["TW_GITHUB_REVIEW_ROOT"]) if os.environ.get("TW_GITHUB_REVIEW_ROOT") else None
SOURCE_CONFLICT_DIR = ROOT / "docs" / "v2_integration" / "evidence" / "source_conflict_1110"
EXPECTED_WEIGHT = "53a1467fdba66e7f22a45e5f2fbc1cb78fc99825b15d516b3e7153d7bfc28144"
FROZEN_WEIGHT = EXPECTED_WEIGHT
RIDGE_FIRST_DAY_LIVE = "de1aa410b7378cf5c52f760fc77e700c4d6b4ba27e8f4f96e0ba3e7756275254"
DEMO_META_PATH = SNAP / "DEMO_META.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _prefer(*candidates: Path) -> Path | None:
    for path in candidates:
        if path and Path(path).exists():
            return Path(path)
    return None


def _join(root: Path | None, *parts: str) -> Path | None:
    if root is None:
        return None
    return Path(root).joinpath(*parts)


def _github_root() -> Path | None:
    explicit = os.environ.get("TW_GITHUB_REVIEW_ROOT")
    if explicit:
        return Path(explicit)
    if GITHUB is not None:
        return GITHUB
    # A live local overlay compares with this checkout, never a DEMO source.
    return ROOT if not is_synthetic_demo_path(v211_root()) else None


def _github_source(parts: tuple[str, ...], fixture: str) -> Path | None:
    root = _github_root()
    if root is not None:
        return _prefer(root.joinpath(*parts))
    return _prefer(SNAP / "github_latest" / fixture)


def demo_meta() -> dict:
    if not DEMO_META_PATH.exists():
        return {}
    return json_file(DEMO_META_PATH)


def is_synthetic_demo_path(path: Path | None) -> bool:
    if path is None:
        return False
    try:
        Path(path).resolve().relative_to(SNAP.resolve())
        return True
    except ValueError:
        return False


def fixture_data_kind(path: Path | None = None) -> str:
    """SYNTHETIC_DEMO for in-repo fixtures; LIVE_READONLY for env/local research roots."""
    if path is not None and is_synthetic_demo_path(path):
        return "SYNTHETIC_DEMO"
    if path is not None:
        return "LIVE_READONLY"
    meta = demo_meta()
    if meta.get("fixture_kind") == "SYNTHETIC_DEMO" and not os.environ.get("TW_V211_LOCAL") and not os.environ.get(
        "TW_RIDGE_ROOT"
    ):
        return "SYNTHETIC_DEMO"
    return "LIVE_READONLY"


def v211_root() -> Path:
    # Explicit live env always wins so DEMO fixtures never mask real research.
    if os.environ.get("TW_V211_LOCAL"):
        return Path(os.environ["TW_V211_LOCAL"])
    snap = SNAP / "v211_local"
    if (snap / "status.json").exists() and (snap / "daily").exists():
        return snap
    return LOCAL_V211


def _rel_or_abs(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def sync_manifest():
    root = v211_root()
    synthetic = is_synthetic_demo_path(root)
    path = SNAP / "matrix" / "SYNC_MANIFEST.json" if synthetic else (
        _prefer(root / "SYNC_MANIFEST.json", root / "matrix" / "SYNC_MANIFEST.json")
        or root / "SYNC_MANIFEST.json"
    )
    if not path.exists():
        return {"status": "MISSING", "source": _rel_or_abs(path), "synced_at": None,
                "data_kind": fixture_data_kind(path), "not_official_performance": True}
    payload = json_file(path)
    demo = payload.get("demo_flag") == "SYNTHETIC_DEMO" or payload.get("public_note", "").startswith("SYNTHETIC_DEMO")
    if not synthetic and demo:
        return {"status": "MISSING", "source": _rel_or_abs(path), "synced_at": None,
                "reason": "synthetic_sync_is_not_live_evidence", "data_kind": "LIVE_READONLY",
                "not_official_performance": True}
    return {
        "status": payload.get("status", "UNKNOWN"),
        "synced_at": payload.get("synced_at"),
        "conflicts": payload.get("conflicts", []),
        "items": len(payload.get("items", [])),
        "source": _rel_or_abs(path),
        "demo_flag": payload.get("demo_flag"),
        "data_kind": fixture_data_kind(path),
        "not_official_performance": demo,
    }


def github_research_status():
    """Separate GitHub paired-research currency from frozen V2.11 prospective publish state."""
    meta_path = _github_source(("reports", "latest", "metadata.json"), "metadata.json")
    github_root = _github_root()
    summary_path = None
    if github_root is None:
        summary_path = _github_source((), "daily_summary.json")
    else:
        root = github_root / "outputs" / "daily_summary"
        dated = []
        if root.is_dir():
            for candidate in root.glob("*/summary.json"):
                day = candidate.parent.name
                try:
                    valid = len(day) == 10 and datetime.fromisoformat(day).date().isoformat() == day
                except ValueError:
                    valid = False
                if valid and candidate.is_file():
                    dated.append(candidate)
        summary_path = max(dated, key=lambda candidate: candidate.parent.name, default=None)
    meta = json_file(meta_path) if meta_path else {}
    summary = json_file(summary_path) if summary_path else {}
    v211_step = (summary.get("steps") or {}).get("v211") or (summary.get("reports") or {}).get("models", {}).get("v211") or {}
    currentness = meta.get("model_currentness") or {}
    return {
        "source_meta": _rel_or_abs(meta_path) if meta_path else None,
        "source_summary": _rel_or_abs(summary_path) if summary_path else None,
        "research_as_of_date": meta.get("as_of_date") or summary.get("output_day") or summary.get("completed_session"),
        "v211_last_successful_as_of_date": meta.get("last_successful_as_of_date"),
        "delayed_run": bool(meta.get("delayed_run", summary.get("delayed_run"))),
        "orchestrator_status": summary.get("status") or meta.get("status"),
        "research_status": meta.get("research_status") or (summary.get("reports") or {}).get("research_status"),
        "v211_status": v211_step.get("status") or ("current" if currentness.get("v211") else "not_current"),
        "v211_maintenance_only": bool(
            v211_step.get("maintenance_only")
            or v211_step.get("status") == "maintenance_only"
            or v211_step.get("reason") == "delayed_session_maintenance_only"
        ),
        "v211_signal_published": v211_step.get("signal_published"),
        "model_currentness": currentness,
        "note": "GitHub 444/555 research_as_of_date ≠ V2.11 前瞻發布；delayed research 不可冒充準時可交易訊號。",
    }


def source_status(now=None):
    now = now or datetime.now(TAIPEI)
    local = v211_status_local()
    github_registry = _github_source(("experiments", "results", "v2", "v211", "signal_registry.json"), "signal_registry.json")
    github = github_research_status()
    github_signals = sorted(json_file(github_registry)) if github_registry else []
    conflicts = compare_source_conflicts()
    sync = sync_manifest()
    freshness = _freshness(now, local, sync)
    return {
        "checked_at": now.isoformat(),
        "local_v211_through": local["prices_through"],
        "local_latest_signal": local["latest_signal_date"],
        "local_matured_signal_days": local["matured_signal_days"],
        "local_weight_ok": local["weight_ok"],
        "local_model_version": local["model_version"],
        "local_source_commit": local.get("source_commit", "local-prospective-snapshot"),
        "local_latest_signal_at": local.get("latest_signal_at"),
        # Keep legacy key as V2.11 tracker publish cutoff (not paired-research as_of).
        "github_latest_success": github.get("v211_last_successful_as_of_date"),
        "github_research_as_of_date": github.get("research_as_of_date"),
        "github_delayed_run": github.get("delayed_run"),
        "github_orchestrator_status": github.get("orchestrator_status"),
        "github_research_status": github.get("research_status"),
        "github_v211_status": github.get("v211_status"),
        "github_v211_maintenance_only": github.get("v211_maintenance_only"),
        "github_v211_signal_published": github.get("v211_signal_published"),
        "github_model_currentness": github.get("model_currentness"),
        "github_status_note": github.get("note"),
        "github_signal_days": [name.replace(".csv", "") for name in github_signals],
        "source_conflict": bool(conflicts),
        "conflicts": conflicts,
        "sync": sync,
        "freshness": freshness,
        "display_root": _rel_or_abs(v211_root()),
        "label_maturity_gate": _label_maturity_gate(now, local),
        "data_kind": local.get("data_kind") or fixture_data_kind(v211_root()),
        "demo_flag": local.get("demo_flag"),
        "not_official_performance": local.get("not_official_performance", False),
    }


def _freshness(now, local, sync) -> dict:
    status = sync.get("status") or "MISSING"
    synced_at = sync.get("synced_at")
    evidence = "CURRENT"
    if status == "MISSING":
        evidence = "UNKNOWN"
    elif status in {"SYNC_FAILED", "failed", "error"}:
        evidence = "SYNC_FAILED"
    elif status in {"UNKNOWN", "unknown"} or not synced_at:
        evidence = "UNKNOWN"
    elif synced_at:
        try:
            synced = datetime.fromisoformat(synced_at)
            age_hours = (now - synced).total_seconds() / 3600.0
            # After protocol close, if prices still stuck on prior session, mark STALE for display.
            if now.hour >= 15 and local.get("prices_through") and local["prices_through"] < now.date().isoformat():
                evidence = "STALE"
            elif age_hours > 36:
                evidence = "STALE"
        except (TypeError, ValueError):
            evidence = "STALE"
    return {
        "evidence_status": evidence,
        "synced_at": synced_at,
        "sync_status": status,
        "prices_through": local.get("prices_through"),
        "checked_at": now.isoformat(),
    }


def _label_maturity_gate(now, local) -> dict:
    """11/09 five-day label ends 11/17; do not declare mature before Taipei 15:00 with prices."""
    today = now.astimezone(TAIPEI).date().isoformat()
    after_close = now.astimezone(TAIPEI).hour >= 15
    prices_through = local.get("prices_through")
    matured = set(v211_matured_local().get("signal_days") or [])
    pending = local.get("payload", {}).get("scoring", {}).get("pending_labels", [])
    target = next((row for row in pending if row.get("signal") == "2026-11-09"), {"signal": "2026-11-09", "label_end": "2026-11-17"})
    ready = (
        after_close
        and prices_through is not None
        and prices_through >= target.get("label_end", "2026-11-17")
        and "2026-11-09" in matured
    )
    return {
        "signal": "2026-11-09",
        "label_end": target.get("label_end", "2026-11-17"),
        "protocol_close_passed": after_close,
        "prices_through": prices_through,
        "in_matured_scores": "2026-11-09" in matured,
        "state": "MATURED" if ready else "PENDING",
        "checked_at": now.isoformat(),
        "today": today,
    }


def v211_status_local():
    root = v211_root()
    path = root / "status.json"
    payload = json_file(path)
    registry = json_file(root / "signal_registry.json")
    weight_path = _prefer(
        Path(os.environ["TW_V211_LOCAL"]) / "weights" / "weights.npz" if os.environ.get("TW_V211_LOCAL") else None,
        LOCAL_V211 / "weights" / "weights.npz",
        ROOT / "experiments" / "results" / "v2" / "v211" / "weights" / "weights.npz",
    )
    weight_hash = sha256(weight_path) if weight_path else None
    scoring = payload["scoring"]
    metrics = scoring.get("metrics")
    latest = max((meta["signal_date"] for meta in registry.values()), default=None)
    kind = fixture_data_kind(root)
    demo = kind == "SYNTHETIC_DEMO" or payload.get("demo_flag") == "SYNTHETIC_DEMO"
    registry_ok = _registry_matches(root)
    weight_ok = weight_hash == EXPECTED_WEIGHT and payload["model_sha256"] == EXPECTED_WEIGHT
    usable = bool(latest and registry_ok and weight_ok)
    proof_path = root / "ledger_integrity.json"
    proof = json_file(proof_path) if proof_path.is_file() else {}
    availability_certified = proof.get("historical_availability_certified") is True
    official_blocked = demo or not usable or not availability_certified or scoring["matured_signal_days"] < 60
    return {
        "status": "AVAILABLE" if latest else "DATA_UNAVAILABLE",
        "source": _rel_or_abs(path),
        "payload": payload,
        "model_sha256": payload["model_sha256"],
        "model_version": payload["model_version"],
        "pending_predictions": scoring["pending_predictions"],
        "matured_signal_days": scoring["matured_signal_days"],
        "metrics": metrics if latest else None,
        "performance": "DATA_UNAVAILABLE" if not latest else ("PENDING" if official_blocked else ("HAS_METRICS" if metrics else "PENDING")),
        "confirmation": "PENDING",
        "prices_through": payload["data"]["prices_through"],
        "latest_signal_date": latest,
        "latest_signal_at": payload["signal"].get("created_at") if latest else None,
        "open_nav": payload["book"]["open_nav"],
        "weight_path": _rel_or_abs(weight_path) if weight_path else None,
        "weight_sha256": weight_hash,
        "weight_ok": weight_ok,
        "registry_ok": registry_ok,
        "source_label": "LOCAL_PROSPECTIVE",
        "source_commit": "local-prospective-snapshot",
        "data_kind": kind,
        "demo_flag": "SYNTHETIC_DEMO" if demo else None,
        "not_official_performance": official_blocked,
        "official_claim_blocked": official_blocked,
        "historical_availability_certified": availability_certified,
    }


def _registry_matches(root: Path) -> bool:
    registry = json_file(root / "signal_registry.json")
    for name, meta in registry.items():
        path = root / "daily" / name
        if not path.exists() or sha256(path) != meta["file_sha256"]:
            return False
        if meta["model_sha256"] != EXPECTED_WEIGHT:
            return False
    return True


def v211_signals_local():
    root = v211_root()
    registry = json_file(root / "signal_registry.json")
    rows = []
    for path in sorted((root / "daily").glob("*.csv")):
        meta = registry.get(path.name, {})
        file_hash = sha256(path)
        for row in csv_rows(path):
            enriched = dict(row)
            enriched["_source"] = _rel_or_abs(path)
            enriched["_source_label"] = "LOCAL_PROSPECTIVE"
            enriched["_file_sha256"] = file_hash
            enriched["_registry_sha256"] = meta.get("file_sha256", "")
            enriched["_registry_match"] = file_hash == meta.get("file_sha256")
            enriched.setdefault("Trade_Eligible", "")
            enriched.setdefault("Execution_Status", "")
            enriched.setdefault("Forecast_Entry_Date", enriched.get("Planned_Execution_Date", ""))
            enriched.setdefault("Scheduled_Rebalance_Date", "")
            rows.append(enriched)
    return rows


def v211_fills_local():
    path = v211_root() / "live_fills.csv"
    return {"source": _rel_or_abs(path), "rows": csv_rows(path), "source_label": "LOCAL_PROSPECTIVE"}


def v211_nav_local():
    path = v211_root() / "live_nav.csv"
    return {"source": _rel_or_abs(path), "rows": csv_rows(path), "source_label": "LOCAL_PROSPECTIVE"}


def v211_matured_local():
    path = v211_root() / "matured_scores.csv"
    if not path.exists():
        return {"source": _rel_or_abs(path), "rows": [], "signal_days": []}
    rows = csv_rows(path)
    days = sorted({row["Signal_Date"] for row in rows})
    return {"source": _rel_or_abs(path), "rows": rows, "signal_days": days}


def compare_source_conflicts(signal_date: str = "2026-11-10"):
    # A missing explicit live source is unavailable, not permission to use DEMO.
    local = _prefer(v211_root() / "daily" / f"{signal_date}.csv")
    github = _github_source(("experiments", "results", "v2", "v211", "daily", f"{signal_date}.csv"), f"daily/{signal_date}.csv")
    if not local or not github or fixture_data_kind(local) != fixture_data_kind(github):
        return []
    local_hash = sha256(local)
    github_hash = sha256(github)
    if local_hash == github_hash:
        return []
    local_rows = {row["Ticker"]: row for row in csv_rows(local)}
    github_rows = {row["Ticker"]: row for row in csv_rows(github)}
    field_diff = []
    for ticker in sorted(set(local_rows) | set(github_rows)):
        left = local_rows.get(ticker, {})
        right = github_rows.get(ticker, {})
        for field in (
            "Predicted_Return_Pct",
            "Rank",
            "Target_Weight",
            "Cash_Weight",
            "Risk_Off",
            "Close",
            "Daily_Sentiment",
            "Created_At",
            "Fetched_At",
            "Price_Sha256",
            "Sentiment_Sha256",
            "Model_Sha256",
            "Model_Version",
        ):
            if str(left.get(field, "")) != str(right.get(field, "")):
                field_diff.append(
                    {
                        "signal_date": signal_date,
                        "ticker": ticker,
                        "field": field,
                        "local": left.get(field, ""),
                        "github": right.get(field, ""),
                    }
                )
    local_sent = next(iter(local_rows.values())).get("Sentiment_Sha256", "")
    github_sent = next(iter(github_rows.values())).get("Sentiment_Sha256", "")
    analysis = SOURCE_CONFLICT_DIR / "ANALYSIS.json"
    local_input_available = False
    github_input_available = False
    if analysis.exists():
        payload = json_file(analysis)
        local_input_available = bool(payload.get("replay", {}).get("can_replay_local_with_frozen_sentiment"))
        github_input_available = bool(payload.get("replay", {}).get("can_replay_github_with_frozen_sentiment"))
    integrity = {
        "SOURCE_CONFLICT": True,
        "ORIGINAL_INPUT_UNAVAILABLE": not local_input_available,
        "PARTIALLY_REPRODUCIBLE": github_input_available and not local_input_available,
        "local_sentiment_sha_at_signal": local_sent,
        "github_sentiment_sha_at_signal": github_sent,
        "do_not_refetch_news_as_original": True,
        "do_not_overwrite_local_with_github_sentiment": True,
    }
    return [
        {
            "status": "SOURCE_CONFLICT",
            "signal_date": signal_date,
            "local_path": _rel_or_abs(local),
            "github_path": _rel_or_abs(github),
            "local_sha256": local_hash,
            "github_sha256": github_hash,
            "local_created_at": next(iter(local_rows.values())).get("Created_At", ""),
            "github_created_at": next(iter(github_rows.values())).get("Created_At", ""),
            "same_price_sha": next(iter(local_rows.values())).get("Price_Sha256")
            == next(iter(github_rows.values())).get("Price_Sha256"),
            "same_model_sha": next(iter(local_rows.values())).get("Model_Sha256")
            == next(iter(github_rows.values())).get("Model_Sha256"),
            "field_diffs": field_diff,
            "integrity": integrity,
            "data_kind": fixture_data_kind(local),
            "demo_flag": "SYNTHETIC_DEMO" if is_synthetic_demo_path(local) else None,
            "note": (
                "SYNTHETIC_DEMO SOURCE_CONFLICT 展示情境；非正式本機／GitHub 原始快照。不得計入真實績效，不得以 ROI 選邊。"
                if is_synthetic_demo_path(local)
                else (
                    "SOURCE_CONFLICT + ORIGINAL_INPUT_UNAVAILABLE（本機情緒快照遺失）；"
                    "GitHub 側 PARTIALLY_REPRODUCIBLE。不得用現況新聞冒充原輸入，不得以 ROI 選邊。"
                )
            ),
        }
    ]


def ledger_status():
    """Distinguish NAV_Open from unavailable NAV_Close; paper fills only."""
    nav = v211_nav_local()
    fills = v211_fills_local()
    book_path = v211_root() / "live_book.csv"
    equal_path = v211_root() / "equal_nav.csv"
    book_rows = csv_rows(book_path) if book_path.exists() else []
    equal_rows = csv_rows(equal_path) if equal_path.exists() else []
    latest = nav["rows"][-1] if nav["rows"] else {}
    published_fills = [row for row in fills["rows"] if row.get("Book") == "published"]
    has_nav_close = "NAV_Close" in (nav["rows"][0] if nav["rows"] else {})
    return {
        "source": nav["source"],
        "source_label": "LOCAL_PROSPECTIVE",
        "latest_date": latest.get("Date"),
        "nav_open": float(latest["NAV_Open"]) if latest.get("NAV_Open") not in (None, "") else None,
        "nav_close": float(latest["NAV_Close"]) if has_nav_close and latest.get("NAV_Close") not in (None, "") else None,
        "nav_close_status": "AVAILABLE" if has_nav_close else "DATA_UNAVAILABLE",
        "liquidation_if_closed": latest.get("Liquidation_If_Closed"),
        "liquidation_note": "Liquidation_If_Closed 不是日終 NAV_Close，不可替代已實現日終績效",
        "cash_weight": latest.get("Cash_Weight"),
        "equity_weight": latest.get("Equity_Weight"),
        "stock_pnl": latest.get("Stock_PnL"),
        "rebalance_cost_latest": latest.get("Rebalance_Cost"),
        "trade_today": latest.get("Trade"),
        "published_fill_count": len(published_fills),
        "published_fills": published_fills,
        "book_rows": book_rows,
        "equal_latest": equal_rows[-1] if equal_rows else {},
        "execution_kind": "SIMULATED_EXECUTION",
        "checked_at": datetime.now(TAIPEI).isoformat(),
    }


def trading_journal(day: str = "全部", ticker: str = "全部"):
    signals = v211_signals_local()
    fills = v211_fills_local()["rows"]
    nav_rows = {row["Date"]: row for row in v211_nav_local()["rows"]}
    fill_index = {
        (row["Signal_Date"], row["Book"]): row
        for row in fills
        if row.get("Book") == "published"
    }
    matured = {row["Signal_Date"] for row in v211_matured_local()["rows"]}
    out = []
    for row in signals:
        if day != "全部" and row["Signal_Date"] != day:
            continue
        if ticker != "全部" and row["Ticker"] != ticker:
            continue
        fill = fill_index.get((row["Signal_Date"], "published"))
        nav = nav_rows.get(fill["Execution_Date"]) if fill else None
        if fill:
            state = "SIMULATED_EXECUTION"
        elif row["Signal_Date"] in matured:
            state = "MATURED_LABEL"
        elif row.get("Execution_Status") == "forecast_only" or str(row.get("Trade_Eligible")).lower() == "false":
            state = "SIGNAL_ONLY"
        else:
            state = "SIGNAL_ONLY"
        out.append(
            {
                "signal_date": row["Signal_Date"],
                "model_version": row["Model_Version"],
                "ticker": row["Ticker"],
                "name": NAMES.get(row["Ticker"], row["Ticker"]),
                "forecast": row["Predicted_Return_Pct"],
                "direction": "up" if float(row["Predicted_Return_Pct"]) > 0 else "down",
                "target_weight": row["Target_Weight"],
                "cash_weight": row["Cash_Weight"],
                "risk_off": row["Risk_Off"],
                "planned_execution": row.get("Planned_Execution_Date", ""),
                "rebalance_day": bool(fill),
                "simulated_fill": bool(fill),
                "execution_date": fill["Execution_Date"] if fill else "",
                "fill_price": fill.get(f"Fill_{row['Ticker']}", "") if fill else "",
                "buy_fee": fill.get("Buy_Fee", "") if fill else "",
                "sell_tax": fill.get("Sell_Tax", "") if fill else "",
                "cash": fill.get("Cash", "") if fill else (nav or {}).get("Cash_Weight", ""),
                "nav": fill.get("NAV_Open", "") if fill else (nav or {}).get("NAV_Open", ""),
                "source": row["_source_label"],
                "state": state,
                "model_sha": row["Model_Sha256"][:12],
                "file_sha": row["_file_sha256"][:12],
            }
        )
    return out


def day_review(now=None):
    now = now or datetime.now(TAIPEI)
    status = v211_status_local()
    payload = status["payload"]
    scoring = payload["scoring"]
    matured = v211_matured_local()
    fills = [row for row in v211_fills_local()["rows"] if row.get("Book") == "published"]
    today = now.astimezone(TAIPEI).date().isoformat()
    label_ready = today >= str(scoring.get("next_label_end") or "")
    # 11/17 before close: do not claim 11/09 matured unless file already contains it.
    matured_today = list(scoring.get("matured_today") or [])
    if today == "2026-11-17" and now.astimezone(TAIPEI).hour < 15:
        # 收盤前不得把 11/09 當已成熟；其他已寫入檔案的成熟日仍可顯示。
        matured_today = [day for day in matured_today if day != "2026-11-09"]
    if scoring["matured_signal_days"] == 0:
        label_state = "PENDING_LABEL"
    else:
        label_state = "HAS_MATURED_LABELS"
    trade_state = "SIMULATED_EXECUTION" if fills else "NO_TRADE"
    return {
        "checked_at": now.isoformat(),
        "label_state": label_state,
        "trade_state": trade_state,
        "matured_signal_days": scoring["matured_signal_days"],
        "matured_today": matured_today,
        "matured_rows": matured["rows"],
        "metrics": scoring.get("metrics"),
        "flow_ok": payload.get("ok") is True,
        "error": payload.get("error"),
        "prices_through": payload["data"]["prices_through"],
        "rebalance_today": payload.get("rebalance_today"),
        "rebalance_reason": payload.get("rebalance", {}).get("reason"),
        "fills": fills,
        "entry_cost": payload["book"].get("entry_cost"),
        "open_nav": payload["book"].get("open_nav"),
        "next_label_end": scoring.get("next_label_end"),
        "limitations": [
            "成熟樣本遠少於 60 日審查門檻",
            "不得把預測報酬當成實際報酬",
            "新聞標題未保存，不可虛構",
        ],
        "label_ready_by_calendar": label_ready,
    }


def next_day_strategy():
    rows = v211_signals_local()
    if not rows:
        return {"state": "PENDING_SIGNAL", "rows": []}
    latest = max(row["Signal_Date"] for row in rows)
    chosen = [row for row in rows if row["Signal_Date"] == latest]
    sample = chosen[0]
    fills = v211_fills_local()["rows"]
    published_fills = [row for row in fills if row.get("Book") == "published"]
    return {
        "state": "LATEST_PUBLISHED_SIGNAL",
        "signal_date": latest,
        "created_at": sample["Created_At"],
        "planned_execution": sample.get("Planned_Execution_Date", ""),
        "forecast_entry": sample.get("Forecast_Entry_Date") or sample.get("Planned_Execution_Date", ""),
        "scheduled_rebalance": sample.get("Scheduled_Rebalance_Date", ""),
        "cash_weight": sample["Cash_Weight"],
        "risk_off": sample["Risk_Off"],
        "trade_eligible": sample.get("Trade_Eligible", ""),
        "model_version": sample["Model_Version"],
        "model_sha256": sample["Model_Sha256"],
        "file_sha256": sample["_file_sha256"],
        "source_label": "LOCAL_PROSPECTIVE",
        "hold_sessions": 5,
        "rows": chosen,
        "note": "只顯示最新已發布訊號；不是今日新策略的保證，也不是券商下單。",
        "last_simulated_execution": published_fills[-1]["Execution_Date"] if published_fills else None,
    }


def ridge_status(now=None):
    now = now or datetime.now(TAIPEI)
    live_root = Path(os.environ["TW_RIDGE_ROOT"]) if os.environ.get("TW_RIDGE_ROOT") else RIDGE
    source_root = live_root / "ledger" if live_root is not None else SNAP / "ridge"
    signals_path = _prefer(source_root / "signals.jsonl")
    labels_path = _prefer(source_root / "labels.jsonl")
    if not signals_path or not labels_path:
        # Incomplete live evidence must never inherit labels from fixtures.
        return {
            **_score_ridge([]), "status": "DATA_UNAVAILABLE", "first_day_ok": False,
            "protocol": "273-day prospective close-only ridge", "protocol_days": 273,
            "min_scorable_days_for_claim": 252,
            "first_day_prediction_sha256": None, "first_day_expected": None,
            "model_hash": None,
            "source_signals": _rel_or_abs(signals_path) if signals_path else None,
            "source_labels": _rel_or_abs(labels_path) if labels_path else None,
            "published_targets": [], "scored_days": [], "pending_targets": [],
            "data_kind": fixture_data_kind(source_root), "official_claim_blocked": True,
            "integrity_reasons": ["missing_same_source_signals_or_labels"],
            "not_official_performance": True, "not_merged_with_v211": True,
            "today_label_state": "DATA_UNAVAILABLE", "horizon": "1 trading day",
            "checked_at": now.isoformat(),
        }
    kind = fixture_data_kind(signals_path)
    demo = kind == "SYNTHETIC_DEMO"
    published = []
    for line in stable_text(signals_path).splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event_type") == "signal_published":
            published.append(event)
    labels = []
    for line in stable_text(labels_path).splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event_type") == "label_filled":
            labels.append(event)
    if any(event.get("demo_flag") == "SYNTHETIC_DEMO" for event in published + labels):
        demo = True
        kind = "SYNTHETIC_DEMO"
    label_map = {(event["target_date"], event["stock_symbol"]): event for event in labels}
    rows = []
    rejected_timing = 0
    for event in published:
        key = (event["target_date"], event["stock_symbol"])
        if key not in label_map:
            continue
        lab = label_map[key]
        if abs(float(lab["prediction_unchanged"]) - float(event["prediction"])) > 1e-12:
            continue
        # Maturity is relative to the target close and actual label observation,
        # rather than a special case for the current calendar day only.
        target = event["target_date"]
        try:
            close = datetime.fromisoformat(target).replace(hour=15, tzinfo=TAIPEI)
            filled = datetime.fromisoformat(lab["label_filled_timestamp"])
            published_at = datetime.fromisoformat(event["signal_publication_timestamp"])
            maturity = datetime.fromisoformat(event.get("label_maturity_timestamp") or close.isoformat())
            valid = all(stamp.tzinfo is not None for stamp in (filled, published_at, maturity))
            valid = valid and close <= maturity <= filled <= now and published_at <= close
        except (KeyError, TypeError, ValueError):
            valid = False
        if not valid:
            rejected_timing += 1
            continue
        rows.append(
            {
                "target_date": event["target_date"],
                "stock_symbol": event["stock_symbol"],
                "prediction": float(event["prediction"]),
                "actual_return": float(lab["actual_return"]),
                "published_at": event["signal_publication_timestamp"],
                "label_filled_at": lab["label_filled_timestamp"],
            }
        )
    first_day = [event for event in published if event["target_date"] == "2026-11-10"]
    first_hash = _prediction_sha(first_day, signals_path)
    targets = sorted({event["target_date"] for event in published})
    scored_days = sorted({row["target_date"] for row in rows})
    pending_targets = [day for day in targets if day not in scored_days]
    today = now.astimezone(TAIPEI).date().isoformat()
    today_pending = today in pending_targets and now.astimezone(TAIPEI).hour < 15
    meta = demo_meta()
    expected = (
        meta.get("ridge_first_day_prediction_sha256")
        if demo
        else RIDGE_FIRST_DAY_LIVE
    )
    first_day_ok = bool(first_hash and expected and first_hash == expected)
    # An unlocked live history can be inspected, but is not an accepted sample.
    accepted_rows = rows if demo or first_day_ok else []
    summary = _score_ridge(accepted_rows)
    scored_days = sorted({row["target_date"] for row in accepted_rows})
    pending_targets = [day for day in targets if day not in scored_days]
    reasons = []
    if not first_day_ok:
        reasons.append("first_day_prediction_lock_mismatch")
    if rejected_timing:
        reasons.append("immature_or_invalid_label_timing")
    # This display reader has no complete receipt/CHAIN or input-version
    # verifier. Sample count and a first-day hash cannot certify later signals.
    forward_evidence_verified = False
    reasons.append("full_forward_evidence_not_verified")
    official_blocked = demo or not first_day_ok or summary["scorable_days"] < 252 or not forward_evidence_verified
    return {
        "status": "PENDING",
        "protocol": "273-day prospective close-only ridge",
        "min_scorable_days_for_claim": 252,
        "protocol_days": 273,
        "source_signals": _rel_or_abs(signals_path),
        "source_labels": _rel_or_abs(labels_path),
        "published_targets": targets,
        "scored_days": scored_days,
        "pending_targets": pending_targets,
        "scorable_days": summary["scorable_days"],
        "rows": summary["rows"],
        "mae": summary["mae"],
        "rmse": summary["rmse"],
        "direction": summary["direction"],
        "spearman_mean": summary["spearman_mean"],
        "spearman_lower": summary["spearman_lower"],
        "spearman_upper": summary["spearman_upper"],
        "daily_spearman": summary["daily_spearman"],
        "zero_mae": summary["zero_mae"],
        "baseline_note": (
            "SYNTHETIC_DEMO schema metrics only; not formal Ridge performance"
            if demo
            else "zero_mae is naive zero-prediction MAE baseline on the same matured rows"
        ),
        "first_day_prediction_sha256": first_hash,
        "first_day_expected": expected,
        "first_day_ok": first_day_ok,
        "model_hash": next((event.get("model_hash") for event in published), None),
        "horizon": "1 trading day",
        "not_merged_with_v211": True,
        "today_label_state": "PENDING_BEFORE_CLOSE" if today_pending else ("SCORED" if today in scored_days else "NO_OR_PENDING_LABEL"),
        "checked_at": now.isoformat(),
        "data_kind": kind,
        "demo_flag": "SYNTHETIC_DEMO" if demo else None,
        "not_official_performance": official_blocked,
        "official_claim_blocked": official_blocked,
        "forward_evidence_verified": forward_evidence_verified,
        "integrity_reasons": reasons,
        "rejected_timing_rows": rejected_timing,
        "diagnostic_rows": len(rows),
    }


def _prediction_sha(events, path=None):
    if not events:
        return None
    # preserve original JSONL line order from published_lines style
    lines = []
    # reconstruct canonical published lines by reading raw file subset is safer; here use sorted stable keys dump
    # Match evidence.prediction_sha256: original JSONL text of published events for that day only.
    if path is None:
        root = Path(os.environ["TW_RIDGE_ROOT"]) if os.environ.get("TW_RIDGE_ROOT") else RIDGE
        path = _prefer(root / "ledger" / "signals.jsonl" if root is not None else SNAP / "ridge" / "signals.jsonl")
    if path is None:
        return None
    chosen = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event_type") == "signal_published" and event.get("target_date") == "2026-11-10":
            chosen.append(line)
    payload = "\n".join(chosen) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _rankdata(values):
    import numpy as np

    arr = np.asarray(values, dtype=float)
    order = arr.argsort(kind="mergesort")
    ranks = np.empty(len(arr), dtype=float)
    ranks[order] = np.arange(1, len(arr) + 1, dtype=float)
    # average ties
    _, start_idx = np.unique(arr[order], return_index=True)
    for i, start in enumerate(start_idx):
        end = start_idx[i + 1] if i + 1 < len(start_idx) else len(arr)
        if end - start > 1:
            ranks[order[start:end]] = ranks[order[start:end]].mean()
    return ranks


def _spearman(x, y):
    import numpy as np

    rx = _rankdata(x)
    ry = _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if denom == 0:
        return None
    return float((rx * ry).sum() / denom)


def _score_ridge(rows):
    if not rows:
        return {
            "rows": 0,
            "scorable_days": 0,
            "mae": None,
            "rmse": None,
            "direction": None,
            "spearman_mean": None,
            "spearman_lower": None,
            "spearman_upper": None,
            "zero_mae": None,
            "daily_spearman": [],
        }
    import numpy as np
    import pandas as pd

    frame = pd.DataFrame(rows)
    error = frame["prediction"] - frame["actual_return"]
    nonzero = frame["actual_return"] != 0
    direction = None
    if nonzero.any():
        direction = float(
            (np.sign(frame.loc[nonzero, "prediction"]) == np.sign(frame.loc[nonzero, "actual_return"])).mean()
        )
    daily = []
    daily_detail = []
    scorable = 0
    for target_date, day in frame.groupby("target_date"):
        if len(day) != 5:
            continue
        if not np.isfinite(day["prediction"]).all() or not np.isfinite(day["actual_return"]).all():
            continue
        scorable += 1
        value = None
        if day["prediction"].nunique() >= 2 and day["actual_return"].nunique() >= 2:
            value = _spearman(day["prediction"].to_numpy(), day["actual_return"].to_numpy())
            if value is not None:
                daily.append(value)
        daily_detail.append({"target_date": target_date, "spearman": value, "n": int(len(day))})
    lower = upper = None
    if daily:
        rng = np.random.default_rng(42)
        means = np.sort([float(np.mean(rng.choice(daily, size=len(daily), replace=True))) for _ in range(4000)])
        lower = float(means[int(0.025 * (4000 - 1))])
        upper = float(means[int(0.975 * (4000 - 1))])
    return {
        "rows": int(len(frame)),
        "scorable_days": scorable,
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "direction": direction,
        "spearman_mean": float(np.mean(daily)) if daily else None,
        "spearman_lower": lower,
        "spearman_upper": upper,
        "zero_mae": float(np.mean(np.abs(frame["actual_return"]))),
        "daily_spearman": daily_detail,
    }


def signal_dates():
    return sorted({row["Signal_Date"] for row in v211_signals_local()})
