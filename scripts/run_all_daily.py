"""One market update, four independent systems, and durable partial results."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common import TZ, current_completed_session

# Publish the frozen-model decision before the slower baseline retraining.
ORDER = ["v211", "555_v2", "444", "555"]
LOCK = ROOT / "logs" / "orchestrator" / "run.lock"
_RUN_CONTEXT = {}


def _alive(pid: int) -> bool:
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True
    import ctypes
    handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
    if not handle:
        return False
    ctypes.windll.kernel32.CloseHandle(handle)
    return True


def _acquire():
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(2):
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode("ascii"))
            return fd
        except FileExistsError:
            try:
                holder = int(LOCK.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                holder = None
            if holder and _alive(holder):
                return None
            if attempt:
                return None
            try:
                LOCK.unlink()
            except OSError:
                return None
    return None


def _release(fd):
    if fd is not None:
        os.close(fd)
        LOCK.unlink(missing_ok=True)


def _run_process(command, name: str, extra_env=None) -> dict:
    log_dir = ROOT / "logs" / "orchestrator"
    log_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(TZ)
    print(f"[{started.isoformat()}] starting {name}", flush=True)
    env = {**os.environ, "PYTHONPATH": str(ROOT), **(extra_env or {})}
    try:
        proc = subprocess.run(command, cwd=ROOT, env=env, text=True,
                              encoding="utf-8", errors="replace", capture_output=True,
                              timeout=14400 if name == "market" else 9000)
        code = proc.returncode
        output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    except subprocess.TimeoutExpired as exc:
        code = 1
        output = f"{name} timed out: {exc}\n"
        for part in (exc.stdout, exc.stderr):
            if part:
                output += part.decode("utf-8", errors="replace") if isinstance(part, bytes) else part
    path = log_dir / f"{started.strftime('%Y%m%dT%H%M%S%f')}_{name}.log"
    path.write_text(output, encoding="utf-8")
    print(f"[{name}] exit={code}; log={path.relative_to(ROOT)}", flush=True)
    return {"status": "success" if code == 0 else "waiting" if code == 2 else "failed",
            "exit_code": code, "log": path.relative_to(ROOT).as_posix(), "output": output}


def _update_market() -> dict:
    result = _run_process([sys.executable, "-m", "scripts.update_market_data"], "market", _RUN_CONTEXT)
    report = {}
    for line in reversed(result.pop("output").splitlines()):
        try:
            value = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(value, dict) and "completed_session" in value:
            report = value
            break
    return {**report, **result}


def _run_one(model: str, mode: str) -> dict:
    result = _run_process(
        [sys.executable, "-m", "scripts.run_model", "--model", model, "--mode", mode],
        model, {"TW_LSTM_SKIP_MARKET_UPDATE": "1", **_RUN_CONTEXT})
    output = result.pop("output", "")
    for line in reversed(output.splitlines()):
        try:
            payload = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(payload, dict) and payload.get("model") == model:
            for key in ("reason", "maintenance_successful", "signal_published", "output_day",
                        "completed_session", "run_date", "unified_csv", "daily_native_dir",
                        "forecast_timing", "generated_at", "recommendation_execution_status"):
                if key in payload:
                    result[key] = payload[key]
            if result["exit_code"] == 2 and payload.get("status") == "maintenance_only":
                result["status"] = "maintenance_only"
            break
    return {"model": model, **result, "outputs_retained": True}


def _completed_model(model: str, session: str, run_date: str) -> dict | None:
    """Only skip successful model exports, never the V2.11 fill/label maintenance."""
    if model == "v211":
        return None
    folder = ROOT / "outputs" / model / session
    if not (folder / "status.json").exists():
        folder = ROOT / "outputs" / model / run_date  # Legacy calendar-keyed exports.
    try:
        status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
        if status.get("status") != "success" or status.get("completed_session") != session:
            return None
        export = status.get("unified_csv")
        if not export or not (ROOT / export).is_file():
            return None
        if model == "555_v2":
            from scripts.run_model import EXPECTED_WEIGHTS
            if status.get("weight_sha256") != EXPECTED_WEIGHTS:
                return None
        if model == "555" and status.get("research_protocol") != "paired-444-core-allocation-v1":
            return None  # Old six-feature exports are not the new shared-core study.
        if model == "444":
            native = ROOT / status["daily_native_dir"] if status.get("daily_native_dir") else folder / "native"
            if not native.resolve().is_relative_to(ROOT.resolve()):
                return None
            if not all((native / name).is_file() for name in (
                    "evaluation_protocol.json", "test_predictions.csv", "next_day_recommendation.csv",
                    "model_metrics.csv", "strategy_metrics.csv", "backtest_nav.csv")):
                return None
            protocol = json.loads((native / "evaluation_protocol.json").read_text(encoding="utf-8"))
            if str(protocol.get("as_of_date", ""))[:10] != session:
                return None
        return {"model": model, "status": "success", "exit_code": 0,
                "reason": "already_completed_for_this_session", "outputs_retained": True}
    except (OSError, ValueError, TypeError):
        return None


def _save_summary(summary):
    destination = ROOT / "outputs" / "daily_summary" / summary["run_date"]
    destination.mkdir(parents=True, exist_ok=True)
    temporary = destination / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(destination / "summary.json")
    latest = ROOT / "logs" / "orchestrator" / "latest_summary.json"
    latest.parent.mkdir(parents=True, exist_ok=True)
    temporary = latest.with_suffix(".json.tmp")
    temporary.write_bytes((destination / "summary.json").read_bytes())
    temporary.replace(latest)


def main(argv=None, now=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["check", "full"], default="full")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--force", action="store_true", help="Rerun model exports; published signals remain immutable")
    args = parser.parse_args(argv)
    if sys.version_info[:2] != (3, 12):
        print(json.dumps({"status": "failed", "reason": "Python 3.12 is required"}))
        return 1
    now = now or datetime.now(TZ)
    run_date = str(now.astimezone(TZ).date())
    session = str(current_completed_session(now).date())
    summary = {"started": now.isoformat(), "run_date": run_date,
               "completed_session": session, "output_day": session,
               "delayed_run": session != run_date, "mode": args.mode, "steps": {}}
    if args.mode == "check":
        from scripts.run_model import inspect_model
        from experiments.v2.v211_market import inspect_market
        summary["market"] = inspect_market(now)
        for model in ORDER:
            summary["steps"][model] = inspect_model(model, now=now)
        codes = [step.get("exit_code", 2) for step in summary["steps"].values()]
        summary["exit_code"] = 1 if 1 in codes else 2 if 2 in codes else 0
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return summary["exit_code"]
    fd = _acquire()
    if fd is None:
        print(json.dumps({"status": "waiting", "reason": "another run is active"}))
        return 2
    try:
        # A delayed scheduled job still owes the newest closed session's research.
        # Freeze that session across children, without changing their real clocks.
        _RUN_CONTEXT.update(TW_LSTM_TARGET_SESSION=session, TW_LSTM_OUTPUT_DAY=session,
                            TW_LSTM_RUN_STARTED_AT=now.isoformat())
        try:
            if args.skip_download:
                from scripts.update_market_data import build_snapshot
                summary["market"] = build_snapshot(now, allow_download=False)
            else:
                summary["market"] = _update_market()
        except Exception as exc:
            summary["market"] = {"status": "failed", "exit_code": 1,
                                 "error": f"{type(exc).__name__}: {exc}"}
        for model in ORDER:
            previous = None if args.force else _completed_model(model, session, run_date)
            try:
                summary["steps"][model] = previous or _run_one(model, "full")
            except Exception as exc:
                summary["steps"][model] = {"model": model, "status": "failed",
                                            "exit_code": 1, "error": str(exc), "outputs_retained": True}
            _save_summary(summary)
        from scripts.daily_reports import build_reports
        try:
            summary["reports"] = build_reports(summary, session)
        except Exception as exc:
            summary["reports"] = {"status": "failed", "error": str(exc)}
        tracker = summary["steps"].get("v211", {})
        deferred_tracker = (summary["delayed_run"] and tracker.get("status") == "maintenance_only"
                            and tracker.get("reason") == "delayed_session_maintenance_only"
                            and tracker.get("maintenance_successful") is True
                            and tracker.get("signal_published") is False)
        codes = [step["exit_code"] for model, step in summary["steps"].items()
                 if not (model == "v211" and deferred_tracker)]
        codes.append(summary["market"].get("exit_code", 0))
        if summary["reports"].get("status") == "failed":
            codes.append(1)
        code = 1 if any(c not in (0, 2) for c in codes) else 2 if 2 in codes else 0
        status = ("research_complete_tracker_deferred" if code == 0 and deferred_tracker
                  else "complete" if code == 0 else "waiting" if code == 2 else "failed")
        summary.update(status=status,
                       exit_code=code, finished=datetime.now(TZ).isoformat())
        _save_summary(summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return code
    finally:
        _RUN_CONTEXT.clear()
        _release(fd)


if __name__ == "__main__":
    raise SystemExit(main())
