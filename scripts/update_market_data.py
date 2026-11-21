"""Update the shared append-only market ledger and materialize model snapshots.

All four systems use this source. Bundled reproducibility fixtures remain
untouched. ``allow_download=False`` only reads existing files.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import signal
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common import TZ, current_completed_session, sha256  # noqa: E402

DEFAULT_TIMEOUT_SECONDS = 1800
TIMEOUT_ENV = "TW_LSTM_MARKET_TIMEOUT_SECONDS"


def snapshot_dir(completed) -> Path:
    return ROOT / "data" / "snapshots" / str(pd.Timestamp(completed).date())


def sentiment_ready(path: Path, completed) -> bool:
    from experiments.v2.v211_market import sentiment_day_ready

    return path.exists() and sentiment_day_ready(pd.read_csv(path), completed)


def _copy_snapshot(source: Path, destination: Path, *, through=None) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if through is None:
        shutil.copyfile(source, temporary)
    else:
        # Keep the original bytes of retained rows, including BOM, quoting and
        # line endings, when a later canonical append exists beyond this run.
        original = source.read_bytes()
        lines = original.splitlines(keepends=True)
        reader = csv.DictReader(io.StringIO(original.decode("utf-8-sig"), newline=""))
        _ = reader.fieldnames
        previous = reader.line_num
        selected = lines[:previous]
        for row in reader:
            if pd.Timestamp(row["Date"]).normalize() <= through:
                selected.extend(lines[previous:reader.line_num])
            previous = reader.line_num
        temporary.write_bytes(b"".join(selected))
    temporary.replace(destination)


def _snapshot_from_report(now, report, *, materialize, completed=None) -> dict:
    from experiments.v2 import v211_market as market

    completed = pd.Timestamp(completed if completed is not None else current_completed_session(now)).normalize()
    folder = snapshot_dir(completed)
    report = dict(report)
    report.update({
        "snapshot_dir": folder.relative_to(ROOT).as_posix(),
        "used_previous_day_as_today": False,
        "read_only": not materialize,
        "shared_source": "v211_market",
    })
    if materialize:
        folder.mkdir(parents=True, exist_ok=True)
        for field, source, filename in (
            ("prices", market.PRICES, "prices.csv"),
            ("sentiment", market.SENTIMENT, "sentiment.csv"),
        ):
            if report.get(f"{field}_ready"):
                _copy_snapshot(source, folder / filename, through=completed)
    for field, filename, validator in (
        ("prices", "prices.csv", market.price_day_ready),
        ("sentiment", "sentiment.csv", market.sentiment_day_ready),
    ):
        path = folder / filename
        try:
            ready = path.exists() and validator(pd.read_csv(path), completed)
        except (OSError, ValueError, KeyError):
            ready = False
        report[f"snapshot_{field}_ready"] = bool(ready)
        if ready:
            report[f"snapshot_{field}_sha256"] = sha256(path)
    if materialize:
        temporary = folder / "manifest.json.tmp"
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(folder / "manifest.json")
    return report


def build_snapshot(now=None, allow_download=True) -> dict:
    """Fetch once, then give each model identical canonical input files.

    This direct API retains its original synchronous behavior. In read-only
    mode readiness describes the canonical source; the ``snapshot_*_ready``
    fields separately report whether model snapshot files exist.
    """
    from experiments.v2 import v211_market as market

    now = now or datetime.now(TZ)
    if allow_download:
        try:
            report = market.update_market(now)
        except Exception as exc:
            report = market.inspect_market(now)
            report["update_error"] = f"{type(exc).__name__}: {exc}"
    else:
        report = market.inspect_market(now)
    return _snapshot_from_report(now, report, materialize=allow_download)


def _worker_command(now, report_path):
    return [sys.executable, "-m", "scripts.update_market_data", "--worker",
            "--started-at", now.isoformat(), "--report-file", str(report_path)]


def _kill_worker_group(process):
    """No descendant may keep changing canonical inputs after inspection begins."""
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        pass


def _run_market_worker(now, timeout_seconds):
    """Return the transport outcome; the worker only updates canonical sources."""
    with tempfile.TemporaryDirectory(prefix="tw-lstm-market-worker-") as temporary:
        report_path = Path(temporary) / "report.json"
        command = _worker_command(now, report_path)
        inherited = {}
        if sys.platform.startswith("linux"):
            from scripts.process_supervisor import inherited_lock_fds, supervised_command
            command, descriptors = supervised_command(command, os.getpid(), inherited_lock_fds())
            inherited["pass_fds"] = descriptors
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", start_new_session=os.name == "posix",
            **inherited,
        )
        timed_out = False
        try:
            output, _ = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_worker_group(process)
            output, _ = process.communicate()
        except BaseException:
            _kill_worker_group(process)
            process.communicate()
            raise
        _kill_worker_group(process)
        if output:
            print(output, end="" if output.endswith("\n") else "\n", flush=True)
        result = {"update_timed_out": timed_out, "update_timeout_seconds": timeout_seconds,
                  "worker_returncode": process.returncode}
        if timed_out:
            result["update_error"] = f"Market source worker exceeded {timeout_seconds:g} seconds"
            return result
        if process.returncode != 0:
            result["update_error"] = f"Market source worker exited with code {process.returncode}"
            return result
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if (not isinstance(report, dict) or not isinstance(report.get("completed_session"), str)
                    or not all(isinstance(report.get(field), bool) for field in
                               ("prices_ready", "sentiment_ready", "ready_to_predict"))):
                raise ValueError("Worker report is missing typed market readiness fields")
            result["report"] = report
        except (OSError, ValueError, TypeError) as exc:
            result["update_error"] = f"Invalid market worker report: {type(exc).__name__}: {exc}"
        return result


def _inspect_saved_session(completed, decision_time):
    """Inspect the run's frozen session while retaining its real decision clock."""
    from experiments.v2 import v211_market as market

    actual = market.inspect_market(decision_time)
    if actual.get("completed_session") == str(completed.date()) or actual.get("gate") == "integrity_error":
        return actual
    # The source inspector selects a session from its clock. A close crossed
    # during acquisition must not retarget the orchestrator's four children.
    reference = completed.tz_localize(TZ) + pd.Timedelta(hours=15)
    report = market.inspect_market(reference.to_pydatetime())
    report["clock"] = decision_time.astimezone(TZ).isoformat()
    report["inspection_reference_clock"] = reference.isoformat()
    report["latest_completed_session"] = actual.get("completed_session")
    timing_gates = {"news_not_yet_available_at_decision", "market_not_yet_available_at_decision"}
    if report.get("gate") in timing_gates:
        available = [report.get("news_available_at")]
        if report.get("fetched_at_kind") == "observed_completion":
            available.append(report.get("fetched_at"))
        if all(not value or pd.Timestamp(value) <= pd.Timestamp(decision_time) for value in available):
            report["ready_to_predict"] = bool(report.get("prices_ready") and report.get("sentiment_ready"))
            report["gate"] = "ready" if report["ready_to_predict"] else ("prices_missing" if not report.get("prices_ready") else "sentiment_waiting")
            report["error"] = None if report["ready_to_predict"] else report.get("price_error") or report.get("sentiment_error")
    return report


def build_supervised_snapshot(now=None, timeout_seconds=DEFAULT_TIMEOUT_SECONDS, *, clock=None):
    """Bound source acquisition, then materialize only validated saved inputs."""
    from scripts.run_model import run_day

    now = now or datetime.now(TZ)
    _actual, _today, completed = run_day(now)
    clock = clock or (lambda: datetime.now(TZ))
    try:
        outcome = _run_market_worker(now, timeout_seconds)
    except Exception as exc:
        outcome = {"update_timed_out": False, "update_timeout_seconds": timeout_seconds,
                   "update_error": f"Market worker could not start: {type(exc).__name__}: {exc}"}
    worker_report = outcome.pop("report", None)
    inspected_at = clock()
    # Reinspect after the process and all its children have stopped. In
    # particular, a price append completed before a blocked news request survives.
    report = _inspect_saved_session(completed, inspected_at)
    report["run_started_at"] = now.astimezone(TZ).isoformat()
    if worker_report is not None:
        for field, value in worker_report.items():
            if field not in report or field.endswith("_fetch_error") or field.startswith("appended_"):
                report[field] = value
        if worker_report.get("error") and not report.get("error"):
            report["error"] = worker_report["error"]
        if worker_report.get("gate") == "integrity_error":
            report["gate"] = "integrity_error"
            report["ready_to_predict"] = False
    report.update(outcome)
    if outcome.get("update_error"):
        report["used_existing_current_data"] = bool(report.get("ready_to_predict"))
        errors = [report.get("error"), outcome["update_error"]]
        report["error"] = " | ".join(error for error in errors if error)
    return _snapshot_from_report(inspected_at, report, materialize=True, completed=completed)


def _exit_code(report):
    if report.get("gate") == "integrity_error":
        return 1
    if report.get("ready_to_predict") and report.get("snapshot_prices_ready") and report.get("snapshot_sentiment_ready"):
        return 0
    if report.get("update_error") and not report.get("update_timed_out"):
        return 1
    return 2


def _timeout_seconds(value):
    try:
        seconds = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Market timeout must be a number from 1 to 3600 seconds") from exc
    if not math.isfinite(seconds) or not 1 <= seconds <= 3600:
        raise ValueError("Market timeout must be from 1 to 3600 seconds")
    return seconds


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-seconds", help=f"Source worker deadline; defaults to {TIMEOUT_ENV} or 1800")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--started-at", help=argparse.SUPPRESS)
    parser.add_argument("--report-file", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        from experiments.v2.v211_market import update_market

        if not args.started_at or not args.report_file:
            return 1
        now = datetime.fromisoformat(args.started_at)
        if now.tzinfo is None:
            return 1
        report = update_market(now)
        temporary = args.report_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
        temporary.replace(args.report_file)
        return 0
    try:
        timeout = _timeout_seconds(args.timeout_seconds if args.timeout_seconds is not None
                                   else os.environ.get(TIMEOUT_ENV, DEFAULT_TIMEOUT_SECONDS))
    except ValueError as exc:
        print(json.dumps({"status": "failed", "exit_code": 1, "error": str(exc)}), flush=True)
        return 1
    report = build_supervised_snapshot(timeout_seconds=timeout)
    code = _exit_code(report)
    report["exit_code"] = code
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
