"""Read-only health checks for the saved research and daily pipeline contract.

No downloading, model fitting, signal publication, Git mutation or automatic
manifest repair occurs. Expected data waiting is distinct from corrupt evidence.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TZ = ZoneInfo("Asia/Taipei")
MARKET = "experiments/results/v2/v211/market"


class HealthFailure(ValueError):
    """A saved contract failed verification."""


def _file(root, relative):
    path = Path(root) / relative
    if path.is_symlink() or not path.resolve().is_relative_to(Path(root).resolve()) or not path.is_file():
        raise HealthFailure(f"Missing or unsafe file: {relative}")
    return path


def _json(root, relative):
    value = json.loads(_file(root, relative).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise HealthFailure(f"Expected a JSON object: {relative}")
    return value


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment(profile):
    if sys.version_info[:2] != (3, 12):
        raise HealthFailure("Python 3.12 is required by the deployed research contract")
    packages = ["numpy", "pandas"]
    packages += (["streamlit", "altair"] if profile == "dashboard" else
                 ["exchange-calendars", "requests", "scikit-learn", "tensorflow-cpu", "torch", "transformers", "yfinance"])
    versions = {name: importlib.metadata.version(name) for name in packages}
    if not (1, 26) <= tuple(int(part) for part in versions["numpy"].split(".")[:2]) < (2, 1):
        raise HealthFailure("NumPy differs from the shared supported range >=1.26,<2.1")
    return {"python": sys.version.split()[0], "profile": profile, "versions": versions,
            "heavy_model_imports_performed": False,
            "dependency_installation": "Use separate production and dashboard requirements; this is a presence/range check, not a lock-file reinstall."}


def syntax(root):
    files = sorted({path for folder in ("scripts", "common", "experiments", "dashboard", "dashboard_v2", "research")
                    for path in (Path(root) / folder).rglob("*.py") if "__pycache__" not in path.parts})
    for path in files:
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path.relative_to(root)))
    return {"python_files": len(files), "bytecode_written": False}


def frozen_bundle(root):
    from scripts.model_bundle import verify_bundle

    verified = verify_bundle(root=root)
    return {"bundle_sha256": verified["bundle_sha256"], "artifacts": verified["artifacts"],
            "source_files": len(verified["inference_sources"])}


def retained_state(root):
    from scripts.persist_daily_state import BOOK_ROOTS, signal_hashes, validate_state

    validate_state(Path(root))
    return {"books": len(BOOK_ROOTS), "immutable_signals": len(signal_hashes(Path(root))),
            "git_and_checkpoint_baselines_checked": True}


def market_clock(root, now):
    from scripts.common import TICKERS, current_completed_session
    from experiments.v2.v211_market import sentiment_day_ready

    completed = pd.Timestamp(current_completed_session(now)).normalize()
    details = {"expected_completed_session": str(completed.date()), "waiting": False}
    for filename, values, symbols in (("prices.csv", ["Close"], [*TICKERS, "^TWII"]),
                                      ("sentiment.csv", ["Daily_Sentiment"], TICKERS)):
        frame = pd.read_csv(_file(root, MARKET + "/" + filename))
        if not {"Date", "ticker", *values}.issubset(frame):
            raise HealthFailure(f"Market source schema incomplete: {filename}")
        dates = pd.to_datetime(frame.Date, errors="raise")
        if dates.isna().any() or frame.duplicated(["Date", "ticker"]).any():
            raise HealthFailure(f"Missing/duplicate market date or ticker: {filename}")
        if dates.gt(completed).any():
            raise HealthFailure(f"Future market rows exceed the latest completed session: {filename}")
        latest = dates.max()
        if pd.isna(latest):
            raise HealthFailure(f"Empty market source: {filename}")
        part = frame.loc[dates.eq(latest)]
        if len(part) != len(symbols) or set(part.ticker) != set(symbols):
            raise HealthFailure(f"Latest market date lacks its complete cross-section: {filename}")
        numbers = part[values].to_numpy(float)
        if not np.isfinite(numbers).all():
            raise HealthFailure(f"Latest market values are nonfinite: {filename}")
        if filename == "prices.csv" and (numbers <= 0).any():
            raise HealthFailure("Latest prices must be positive")
        if filename == "sentiment.csv" and (np.abs(numbers) > 1).any():
            raise HealthFailure("Sentiment scores must stay in [-1,1]")
        details[filename] = {"through": str(latest.date()), "rows": len(frame), "latest_rows": len(part)}
        details["waiting"] |= latest < completed
        if filename == "sentiment.csv":
            ready = sentiment_day_ready(frame, latest)
            details[filename]["source_evidence_ready"] = bool(ready)
            details["waiting"] |= not ready
    details["waiting_note"] = "Missing current-session data can be retried; unknown news is not certified as neutral."
    return details


def latest_research(root, now, profile):
    state = _json(root, "reports/latest/four_systems.json")
    day = str(state.get("as_of_date", ""))
    parsed = pd.Timestamp(day)
    if pd.isna(parsed) or day != str(parsed.date()) or parsed > pd.Timestamp(now.date()):
        raise HealthFailure("Saved daily research has an invalid or future data date")
    if profile == "core":
        from scripts.common import current_completed_session
        if parsed > pd.Timestamp(current_completed_session(now)):
            raise HealthFailure("Saved daily research is later than the latest completed session")
    models = state.get("models")
    if not isinstance(models, dict) or not {"444", "555", "555_v2", "v211"}.issubset(models):
        raise HealthFailure("Latest research omits one of the four system statuses")
    for model, record in models.items():
        if not isinstance(record, dict):
            raise HealthFailure(f"Invalid saved model status: {model}")
        if record.get("current") and record.get("current_output"):
            _file(root, record["current_output"])
    manifest_path = f"reports/daily/{day}/paired/reproduction_manifest.json"
    base = f"reports/daily/{day}/paired"
    manifest_file = Path(root) / manifest_path
    primary = state.get("primary_research") or {}
    complete = (state.get("research_status") == "complete" or state.get("status") == "complete"
                or isinstance(primary, dict) and primary.get("status") in {"complete", "success"})
    if not manifest_file.exists() and not manifest_file.is_symlink() and not complete:
        # A source/model failure can legitimately publish a partial session
        # before any paired report exists. Keep the next update attempt runnable.
        return {"as_of_date": day, "saved_run_date": state.get("run_date"), "status": state.get("status"),
                "research_status": state.get("research_status"), "waiting": True,
                "paired_outputs": 0, "retained_input_archives": 0,
                "note": "The partial session has no completed paired report yet; source/model retry is required."}
    manifest = _json(root, manifest_path)
    for name, expected in manifest["outputs_sha256"].items():
        if _digest(_file(root, base + "/" + name)) != expected:
            raise HealthFailure(f"Canonical paired research output hash changed: {name}")
    for name, entry in manifest["inputs"].items():
        raw = gzip.decompress(_file(root, entry["archive"]).read_bytes())
        if len(raw) != entry["bytes"] or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise HealthFailure(f"Retained paired input archive differs: {name}")
    waiting = state.get("research_status") != "complete"
    if profile == "core":
        from scripts.common import current_completed_session
        waiting |= parsed < pd.Timestamp(current_completed_session(now))
    return {"as_of_date": day, "saved_run_date": state.get("run_date"), "status": state.get("status"),
            "research_status": state.get("research_status"), "waiting": bool(waiting),
            "paired_outputs": len(manifest["outputs_sha256"]), "retained_input_archives": len(manifest["inputs"]),
            "note": "The actual retry date may be newer than this preserved completed research report."}


def historical_research(root):
    from dashboard.data import STUDIES, load_study
    from scripts.research_v2_common import verify_report

    history = verify_report(root)
    if not history["available"]:
        raise HealthFailure("Retained V2.12 report/model pack is missing")
    v3 = {key: len(load_study(key, root=root)["metrics"]) for key in STUDIES}
    return {"v2_model_count": history["model_count"], "v2_verified_files": history["verified_files"],
            "v2_data_cutoff": history["summary"]["as_of"], "v3_studies": v3,
            "model_training_performed": False, "historical_availability_certified": False}


def v2_live_presentation(root):
    """Exercise the saved V2 display contract, including semantic cost checks.

    Ledger hashes alone cannot establish that display assumptions still match
    the next rebalance. Use the same read-only loader used by the website.
    """
    from dashboard.v2_presentation import load_v2_presentation

    study = load_v2_presentation(root=root, mode="saved")
    costs = study["costs"]
    return {"period_start": study["period_start"], "period_end": study["period_end"],
            "books": {str(row.Arm): {"execution_count": int(row.Execution_Count),
                                      "actual_execution_cost": float(row.Actual_Execution_Cost)}
                      for row in costs.itertuples(index=False)},
            "costs_reconciled": True, "simulated_execution_only": True,
            "source_files": len(study["sources"]), "model_training_performed": False}


def evaluate(root=ROOT, profile="core", now=None):
    root = Path(root).resolve()
    now = now or datetime.now(TZ)
    if now.tzinfo is None:
        raise ValueError("Health-check time must contain a timezone")
    now = now.astimezone(TZ)
    checks = []
    jobs = [("environment", lambda: environment(profile)), ("python_syntax", lambda: syntax(root)),
            ("frozen_bundle", lambda: frozen_bundle(root)), ("immutable_daily_state", lambda: retained_state(root)),
            ("latest_research", lambda: latest_research(root, now, profile)),
            ("v2_live_presentation", lambda: v2_live_presentation(root)),
            ("retained_research", lambda: historical_research(root))]
    if profile == "core":
        jobs.insert(4, ("market_clock", lambda: market_clock(root, now)))
    for name, check in jobs:
        try:
            details = check()
            status = "waiting" if details.get("waiting") else "pass"
            checks.append({"name": name, "status": status, "details": details})
        except Exception as exc:
            checks.append({"name": name, "status": "fail", "error": f"{type(exc).__name__}: {exc}"})
    failed = [item["name"] for item in checks if item["status"] == "fail"]
    waiting = [item["name"] for item in checks if item["status"] == "waiting"]
    return {"schema": "project-health-v1", "checked_at": now.isoformat(), "profile": profile,
            "read_only": True, "network_access_performed": False, "model_training_performed": False,
            "status": "fail" if failed else "waiting" if waiting else "pass", "checks": checks,
            "failed_checks": failed, "waiting_checks": waiting,
            "exit_code": 1 if failed else 0,
            "limitations": ["Passing retained checks cannot guarantee remote provider availability or GitHub scheduling.",
                            "Expected current-session waiting is reported separately and does not prevent the update attempt.",
                            "File integrity does not certify historical news timing, model superiority or a broker fill."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("core", "dashboard"), default="core")
    args = parser.parse_args(argv)
    result = evaluate(profile=args.profile)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
