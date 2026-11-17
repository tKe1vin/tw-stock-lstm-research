"""Small, read-only data contract for the Streamlit research presentation.

Loading a page does not download prices/news, train a model, update a ledger or
read private provider caches.  Curves contain derived portfolio wealth only;
their retained NAV inputs were checked during an explicit offline export.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TZ = ZoneInfo("Asia/Taipei")
ASSETS = "dashboard/assets"
SCHEMA = "research-dashboard-curves-v1"
STUDIES = {
    "v3.3": {"label": "V3.3 · 20 交易日", "report_dir": "reports/research_v3_3/2026-11-17",
             "curves_file": "curves_v33.csv", "hold_sessions": 20},
    "v3.2": {"label": "V3.2 · 5 交易日", "report_dir": "reports/research_v3/2026-11-10",
             "curves_file": "curves_v32.csv", "hold_sessions": 5},
}
ARM_NAMES = {"B0_benchmark": "0050 基準", "B1_equal": "0050＋五股等權",
             "ridge_price": "價格 Ridge", "ridge_news": "新聞 Ridge"}
REPORT_FILES = ("holdout_metrics.csv", "validation_metrics.csv", "selection.json",
                "registered_config.json", "limitations.json", "input_provenance.json",
                "reproduction_manifest.json")


class DataIntegrityError(ValueError):
    """The presentation must not silently show missing or changed evidence."""


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _path(root, name):
    root = Path(root).resolve()
    candidate = (root / name).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise DataIntegrityError(f"缺少研究來源或路徑不安全：{name}")
    return candidate


def _json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise DataIntegrityError(f"研究 JSON 無法讀取：{Path(path).name}") from exc
    if not isinstance(value, dict):
        raise DataIntegrityError("研究 JSON 必須是物件")
    return value


def _numeric(frame, columns, context):
    if not set(columns).issubset(frame.columns):
        raise DataIntegrityError(f"{context} 欄位不完整")
    try:
        values = frame[list(columns)].to_numpy(float)
    except (ValueError, TypeError) as exc:
        raise DataIntegrityError(f"{context} 含非數值資料") from exc
    if not np.isfinite(values).all():
        raise DataIntegrityError(f"{context} 含缺值或非有限值")


def validate_metrics(metrics):
    required = ("ROI_Pct", "MDD_Pct", "Total_Transaction_Cost", "Initial_Capital", "Sessions")
    _numeric(metrics, required, "績效表")
    if "Arm" not in metrics or metrics.Arm.duplicated().any() or set(metrics.Arm) != set(ARM_NAMES):
        raise DataIntegrityError("績效表必須包含四個不同的登記比較組")
    if not {"Start", "End"}.issubset(metrics.columns) or metrics.Start.nunique() != 1 or metrics.End.nunique() != 1:
        raise DataIntegrityError("同一研究的比較期間必須一致")
    if metrics.Initial_Capital.nunique() != 1 or (metrics.Initial_Capital <= 0).any():
        raise DataIntegrityError("同一研究的初始資金必須一致且為正")
    if metrics.Sessions.nunique() != 1 or (metrics.Sessions < 2).any() or not np.equal(metrics.Sessions, np.floor(metrics.Sessions)).all():
        raise DataIntegrityError("比較組的交易日期數量不一致")
    if (metrics.ROI_Pct <= -100).any() or (metrics.MDD_Pct > 1e-9).any() or (metrics.MDD_Pct <= -100).any() or (metrics.Total_Transaction_Cost < 0).any():
        raise DataIntegrityError("績效數值範圍不合法")
    start, end = pd.Timestamp(metrics.Start.iloc[0]), pd.Timestamp(metrics.End.iloc[0])
    if pd.isna(start) or pd.isna(end) or start > end:
        raise DataIntegrityError("績效期間不合法")
    return metrics


def validate_curves(curves, metrics):
    """Check full aligned paths and reconcile terminal return and drawdown."""
    validate_metrics(metrics)
    if list(curves.columns) != ["Date", "Arm", "Wealth_Index", "Drawdown_Pct"]:
        raise DataIntegrityError("曲線僅允許日期、組別、衍生財富與回撤四欄")
    work = curves.copy()
    try:
        work["Date"] = pd.to_datetime(work.Date, errors="raise")
    except (ValueError, TypeError) as exc:
        raise DataIntegrityError("曲線日期不合法") from exc
    if work.Date.isna().any() or work.Date.dt.tz is not None or not work.Date.eq(work.Date.dt.normalize()).all():
        raise DataIntegrityError("曲線日期必須是無時區的市場日期")
    if work.duplicated(["Date", "Arm"]).any() or set(work.Arm) != set(ARM_NAMES):
        raise DataIntegrityError("曲線比較組缺漏或日期重複")
    _numeric(work, ("Wealth_Index", "Drawdown_Pct"), "財富曲線")
    if (work.Wealth_Index <= 0).any():
        raise DataIntegrityError("財富指數必須為正")
    expected_dates = None
    for arm in ARM_NAMES:
        part = work[work.Arm.eq(arm)].sort_values("Date")
        dates = part.Date.reset_index(drop=True)
        row = metrics[metrics.Arm.eq(arm)].iloc[0]
        if expected_dates is not None and not dates.equals(expected_dates):
            raise DataIntegrityError("曲線日期未完整配對，不可靜默刪除缺值")
        expected_dates = dates
        if len(part) != int(row.Sessions) or str(part.Date.iloc[0].date()) != str(row.Start)[:10] or str(part.Date.iloc[-1].date()) != str(row.End)[:10]:
            raise DataIntegrityError("曲線期間或列數與公開績效不一致")
        wealth = part.Wealth_Index.to_numpy(float)
        peak = np.maximum.accumulate(np.r_[100.0, wealth])[1:]
        drawdown = (wealth / peak - 1) * 100
        if not np.allclose(part.Drawdown_Pct, drawdown, rtol=0, atol=1e-8):
            raise DataIntegrityError("曲線回撤無法由含進場成本的財富重算")
        if abs((wealth[-1] - 100) - float(row.ROI_Pct)) > 1e-8 or abs(float(drawdown.min()) - float(row.MDD_Pct)) > 1e-8:
            raise DataIntegrityError("曲線期末報酬／最大回撤與公開績效不符")
    return work.sort_values(["Date", "Arm"]).reset_index(drop=True)


def load_study(version="v3.3", root=ROOT):
    """Load one dated study; never rank or merge different study periods."""
    version = str(version).lower()
    if version not in STUDIES:
        raise DataIntegrityError("未知研究版本")
    root = Path(root)
    spec = STUDIES[version]
    manifest = _json(_path(root, ASSETS + "/manifest.json"))
    if manifest.get("schema") != SCHEMA or manifest.get("raw_provider_data_included") is not False:
        raise DataIntegrityError("展示資料契約不符")
    studies = manifest.get("studies")
    if not isinstance(studies, dict):
        raise DataIntegrityError("展示研究清單不合法")
    entry = studies.get(version)
    if not isinstance(entry, dict) or entry.get("curves_file") != spec["curves_file"]:
        raise DataIntegrityError("展示版本與來源不符")
    expected_paths = {spec["report_dir"] + "/" + name for name in REPORT_FILES}
    if not isinstance(entry.get("report_sha256"), dict) or set(entry["report_sha256"]) != expected_paths:
        raise DataIntegrityError("展示來源清單不完整")
    for name, wanted in entry["report_sha256"].items():
        if sha256(_path(root, name)) != wanted:
            raise DataIntegrityError(f"公開研究來源已變動：{name}")
    curve_path = _path(root, ASSETS + "/" + spec["curves_file"])
    if sha256(curve_path) != entry.get("curves_sha256"):
        raise DataIntegrityError("展示財富曲線雜湊不符")
    report = root / spec["report_dir"]
    metrics = validate_metrics(pd.read_csv(report / "holdout_metrics.csv")).copy()
    curves = validate_curves(pd.read_csv(curve_path), metrics)
    validation = pd.read_csv(report / "validation_metrics.csv")
    _numeric(validation, ("Fold", "Relative_Wealth_Excess_Pct", "Stress_Relative_Wealth_Excess_Pct"), "驗證折")
    if not {"Candidate", "Split"}.issubset(validation) or not validation.Split.eq("validation").all() or validation.duplicated(["Candidate", "Fold"]).any():
        raise DataIntegrityError("驗證折的角色或鍵不合法")
    selection, config = _json(report / "selection.json"), _json(report / "registered_config.json")
    if set(validation.Candidate) != set(config["candidate_order"]) or any(set(part.Fold) != set(range(1, config["validation_folds"] + 1)) for _, part in validation.groupby("Candidate")):
        raise DataIntegrityError("驗證折未保留完整登記候選")
    if selection.get("holdout_used_for_selection") is not False or selection.get("historical_alpha_proven") is not False:
        raise DataIntegrityError("展示不能把探索性結果當成獨立前瞻證據")
    metrics["Display_Name"] = metrics.Arm.map(ARM_NAMES)
    benchmark = float(metrics.loc[metrics.Arm.eq("B0_benchmark"), "ROI_Pct"].iloc[0])
    metrics["Gap_Vs_Benchmark_Pp"] = metrics.ROI_Pct - benchmark
    curves["Display_Name"] = curves.Arm.map(ARM_NAMES)
    return {"id": version, "label": spec["label"], "metrics": metrics,
            "validation": validation, "selection": selection, "curves": curves,
            "config": config, "limitations": _json(report / "limitations.json"),
            "period_start": str(metrics.Start.iloc[0])[:10], "period_end": str(metrics.End.iloc[0])[:10],
            "hold_sessions": spec["hold_sessions"], "news_arm": "ridge_news", "price_arm": "ridge_price",
            "benchmark_arm": "B0_benchmark", "report_dir": spec["report_dir"],
            "provenance": {**entry, "curve_definition": manifest["curve_definition"],
                           "raw_provider_data_included": False, "curve_origin": "retained portfolio accounting NAV",
                           "input_provenance": _json(report / "input_provenance.json")}}


def load_daily_status(root=ROOT):
    """Display retained operational state; a deferred tracker is not current."""
    root = Path(root)
    path = root / "reports/latest/four_systems.json"
    if not path.is_file():
        return {"available": False, "message": "此版本尚無每日系統狀態", "models": [], "news": {}, "links": {}}
    value = _json(path)
    try:
        as_of = pd.Timestamp(value["as_of_date"])
        run_date = pd.Timestamp(value["run_date"])
    except (KeyError, ValueError, TypeError) as exc:
        raise DataIntegrityError("每日資料日／執行日不合法") from exc
    if (pd.isna(as_of) or pd.isna(run_date) or as_of.tzinfo is not None or run_date.tzinfo is not None
            or as_of != as_of.normalize() or run_date != run_date.normalize() or as_of > run_date):
        raise DataIntegrityError("每日資料日／執行日不合法")
    models = []
    for model in ("444", "555", "555_v2", "v211"):
        item = (value.get("models") or {}).get(model)
        if not isinstance(item, dict) or not isinstance(item.get("current"), bool):
            raise DataIntegrityError("每日狀態缺少四個明確的時效標記")
        if item.get("maintenance_only") and item["current"]:
            raise DataIntegrityError("僅維護的追蹤器不能冒充已發布新訊號")
        models.append({"model": model, **item})
    by_model = {item["model"]: item for item in models}
    if value.get("status") == "complete" and not all(item["current"] for item in models):
        raise DataIntegrityError("四系統完成狀態與各模型紀錄矛盾")
    if value.get("research_status") == "complete" and not all(by_model[name]["current"] for name in ("444", "555")):
        raise DataIntegrityError("444／555完成狀態與模型紀錄矛盾")
    if not isinstance(value.get("news") or {}, dict):
        raise DataIntegrityError("新聞覆蓋摘要不合法")
    return {"available": True, "as_of_date": str(as_of.date()), "run_date": str(run_date.date()),
            "research_status": value.get("research_status", "unknown"), "status": value.get("status", "unknown"),
            "delayed_run": bool(value.get("delayed_run")), "models": models,
            "news": value.get("news") or {}, "started": value.get("started"), "finished": value.get("finished"),
            "calendar_age_days": (datetime.now(TZ).date() - as_of.date()).days,
            "last_successful_as_of_date": value.get("last_successful_as_of_date"),
            "currentness_scope": "relative to the retained as_of_date; not an independent live exchange-readiness check",
            "links": {"summary": "reports/latest/summary.md", "444": "reports/journals/444.md",
                      "555": "reports/journals/555.md", "attempts": f"reports/attempts/{run_date.date()}/summary.md"}}
