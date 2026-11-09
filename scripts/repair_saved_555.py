"""Repair a failed 555 export using saved 444 forecasts, never live backfilling.

The default is read-only. A repair is explicitly labeled historical_replay,
retains the failed status and original run summary, and never calls a model,
market updater, signal publisher, or trade ledger.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd

from scripts.common import ROOT, TZ, atomic_json, sha256
from scripts.paired_native import export_paired_native
from scripts.paired_research import paired_daily_recommendation, run_paired_research
from scripts.unified_csv import NAMES, blank_row, write_table


def inspect_saved_session(as_of, *, root=ROOT):
    root = Path(root)
    day = str(pd.Timestamp(as_of).date())
    baseline = root / "outputs/444" / day / "native"
    failed_status = root / "outputs/555" / day / "status.json"
    manifest_path = root / "reports/daily" / day / "paired/reproduction_manifest.json"
    if not failed_status.is_file() or not manifest_path.is_file():
        raise ValueError("A saved failed session and its original paired inputs are required")
    status = json.loads(failed_status.read_text(encoding="utf-8"))
    if status.get("status") not in {"failed", "waiting"}:
        raise ValueError("Refusing to replace an already successful 555 session")
    if (failed_status.parent / "native/next_day_recommendation.csv").exists():
        raise ValueError("An existing recommendation must be preserved, not retrospectively replaced")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in ("predictions", "baseline_protocol", "core_recommendation"):
        name = {"predictions": "test_predictions.csv", "baseline_protocol": "evaluation_protocol.json",
                "core_recommendation": "next_day_recommendation.csv"}[key]
        if sha256(baseline / name) != manifest["inputs"][key]["sha256"]:
            raise ValueError(f"The saved 444 source changed: {key}")
    inputs = {}
    for key in ("prices", "sentiment"):
        record = manifest["inputs"][key]
        path = (root / record["archive"]).resolve()
        if not path.is_relative_to((root / "data/research_snapshots").resolve()):
            raise ValueError("Snapshot leaves the content-addressed archive")
        data = gzip.decompress(path.read_bytes())
        if hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError(f"Saved input failed verification: {key}")
        from io import BytesIO

        inputs[key] = pd.read_csv(BytesIO(data))
    return {"day": day, "baseline": baseline, "failed_status": failed_status,
            "original_status": status, "inputs": inputs, "manifest": manifest,
            "read_only": True, "training_executed": False, "signal_published": False}


def repair_saved_session(as_of, *, root=ROOT, now=None):
    root = Path(root)
    inspected = inspect_saved_session(as_of, root=root)
    day, baseline, inputs = inspected["day"], inspected["baseline"], inspected["inputs"]
    created = now or datetime.now(TZ)
    if created.tzinfo is None:
        raise ValueError("Repair time requires a timezone")
    protocol = json.loads((baseline / "evaluation_protocol.json").read_text(encoding="utf-8"))
    result = run_paired_research(baseline / "test_predictions.csv", inputs["prices"], inputs["sentiment"], protocol)
    recommendations = paired_daily_recommendation(baseline / "next_day_recommendation.csv", inputs["prices"], inputs["sentiment"], protocol)
    out = root / "outputs/555" / day
    native = out / "native"
    audit = root / "reports/daily" / day / "paired/repair"
    audit.mkdir(parents=True, exist_ok=True)
    (audit / "original_failed_status.json").write_bytes(inspected["failed_status"].read_bytes())
    old_day_status = root / "reports/daily" / day / "status.json"
    if old_day_status.exists():
        (audit / "original_day_status.json").write_bytes(old_day_status.read_bytes())
    export = export_paired_native(result, recommendations, baseline, native, day, created)
    rec_path = native / "next_day_recommendation.csv"
    rec = pd.read_csv(rec_path)
    rec["Run_Mode"] = "historical_replay"
    rec["Repair_Reason"] = "reconstruct_failed_export_from_previously_saved_444_core"
    rec.to_csv(rec_path, index=False, encoding="utf-8-sig", float_format="%.17g")
    rows = []
    for item in rec.to_dict("records"):
        row = blank_row()
        row.update({"model_id": "555", "model_version": export["protocol_version"],
                    "signal_date": day, "run_date": str(created.astimezone(TZ).date()),
                    "generated_at": created.astimezone(TZ).isoformat(), "data_asof": day,
                    "stock_id": item["Ticker"], "stock_name": NAMES[item["Ticker"]],
                    "predicted_return_pct": item["Predicted_Return_Pct"], "prediction_horizon_days": 1,
                    "target_weight": item["Target_Weight"], "cash_weight": item["Cash_Weight"],
                    "forecast_entry_date": item["Target_Date"], "trade_eligible": False,
                    "execution_status": "historical_replay_no_order",
                    "data_hash": inspected["manifest"]["inputs"]["prices"]["sha256"]})
        rows.append(row)
    unified = out / "unified_signal.csv"
    write_table(unified, rows, {"model_id": "555", "core_model": "444", "run_mode": "historical_replay",
                              "research_sample": False, "signal_published": False,
                              "original_failed_status": (audit / "original_failed_status.json").relative_to(root).as_posix(),
                              "source_sha256": export["core_source_sha256"]})
    record = {"as_of_date": day, "repaired_at": created.astimezone(TZ).isoformat(),
              "run_mode": "historical_replay", "original_failure_preserved": True,
              "training_executed": False, "live_signal_published": False,
              "source_sha256": export["core_source_sha256"], "original_run_summary_changed": False}
    atomic_json(audit / "record.json", record)
    payload = {"model": "555", "status": "success", "completed_session": day,
               "run_date": str(created.astimezone(TZ).date()), "core_model": "444",
               "research_protocol": export["protocol_version"], "run_mode": "historical_replay",
               "training_executed": False, "forecast_produced": True, "signal_published": False,
               "reason": "repaired_saved_research_export_not_a_forward_signal",
               "daily_native_dir": native.relative_to(root).as_posix(),
               "native_output_dir": native.relative_to(root).as_posix(),
               "unified_csv": unified.relative_to(root).as_posix(), "repair": record}
    atomic_json(out / "status.json", payload)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--mode", choices=("check", "repair"), default="check")
    args = parser.parse_args(argv)
    if args.mode == "check":
        result = inspect_saved_session(args.as_of)
        print(json.dumps({"ready": True, "as_of_date": result["day"], "read_only": True,
                          "training_executed": False, "signal_published": False}))
    else:
        print(json.dumps(repair_saved_session(args.as_of), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
