"""Run the V2.11 daily command once, with a lock and a log that is never overwritten.

The scheduled task must start this module with the project virtualenv, and its
working directory must be the repository root. A second trigger that arrives
while the first is still running writes a skip log and exits 0.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from experiments.v2.guard import ROOT
from experiments.v2.v211 import V211, assert_v211
from scripts.common import TZ

LOG_DIR = V211 / "logs"
LOCK = LOG_DIR / "run.lock"


def pid_alive(pid):
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


def acquire_run_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            holder = int(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            holder = None
        if holder and pid_alive(holder):
            return None
        try:
            path.unlink()
        except OSError:
            return None
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return None
    os.write(fd, str(os.getpid()).encode("ascii"))
    return fd


def release_run_lock(path, fd):
    if fd is None:
        return
    os.close(fd)
    try:
        Path(path).unlink()
    except OSError:
        pass


def wait_for_lock(path, timeout_s, pause_s=1.0):
    """Wait until the other run finishes, then take the lock. Never report success by skipping."""
    deadline = time.monotonic() + timeout_s
    while True:
        fd = acquire_run_lock(path)
        if fd is not None:
            return fd
        if time.monotonic() >= deadline:
            return None
        time.sleep(pause_s)


def append_index(record):
    path = assert_v211(LOG_DIR / "index.jsonl")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    os.chdir(ROOT)
    started = datetime.now(TZ)
    assert_v211(LOG_DIR / "index.jsonl")
    log_path = assert_v211(LOG_DIR / started.strftime("%Y%m%dT%H%M%S%f.log"))
    fd = wait_for_lock(LOCK, timeout_s=15 * 60)
    if fd is None:
        log_path.write_text("another run is still active; this attempt is not success\nexit 2\n", encoding="utf-8")
        append_index({"started": started.isoformat(), "log": log_path.name, "exit_code": 2, "skipped": True})
        print(json.dumps({"exit_code": 2, "skipped": True, "log": str(log_path)}), flush=True)
        return 2
    code = 1
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "experiments.v2.v211_daily"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        code = int(completed.returncode)
        log_path.write_text(
            completed.stdout + ("\n" + completed.stderr if completed.stderr else "") + f"\nexit {code}\n",
            encoding="utf-8",
        )
    except Exception as exc:
        log_path.write_text(f"error: {exc}\nexit 1\n", encoding="utf-8")
        code = 1
    finally:
        release_run_lock(LOCK, fd)
    append_index({"started": started.isoformat(), "log": log_path.name, "exit_code": code, "skipped": False})
    print(json.dumps({"exit_code": code, "log": str(log_path)}, ensure_ascii=False), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
