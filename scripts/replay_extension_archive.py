"""Replay the retained extension using its pinned historical implementation.

No model is fitted and no report is rewritten. The current implementation keeps
its source-drift rejection; this command runs an isolated committed revision.
Requires a Git checkout containing the public historical commit.
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

from scripts.common import ROOT
from scripts.research_extension import verify_run

RETAINED_COMMIT = "b62034e4fa850933a632c8db6321b309ebad85f9"


def replay(run_dir, *, mode="reproduce", as_of=None, prices=None, sentiment=None):
    if mode not in ("reproduce", "forecast") or (mode == "forecast" and as_of is None):
        raise ValueError("Only reproduce or dated frozen forecast is allowed")
    run = Path(run_dir).resolve()
    manifest = verify_run(run)
    expected = manifest.get("replay_code_sha256", {})
    if not expected:
        raise ValueError("The retained run has no reviewed replay source contract")
    try:
        archive = subprocess.run(["git", "archive", RETAINED_COMMIT], cwd=ROOT,
                                 capture_output=True, check=True).stdout
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise ValueError("Historical replay requires a Git checkout containing commit " + RETAINED_COMMIT) from exc
    with tempfile.TemporaryDirectory(prefix="extension-archive-") as folder:
        sandbox = Path(folder)
        with tarfile.open(fileobj=io.BytesIO(archive)) as package:
            package.extractall(sandbox, filter="data")
        import hashlib
        for name, wanted in expected.items():
            source = (sandbox / name).resolve()
            if not source.is_relative_to(sandbox) or not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != wanted:
                raise ValueError("Pinned source does not match this retained run: " + name)
        env = {**os.environ, "PYTHONPATH": str(sandbox)}
        command = [sys.executable, "-m", "scripts.research_extension", "--mode", mode,
                   "--output-dir", str(run)]
        if mode == "forecast":
            command += ["--as-of", as_of]
            for flag, path in (("--prices", prices), ("--sentiment", sentiment)):
                if path is not None:
                    command += [flag, str(Path(path).resolve())]
        child = subprocess.run(command, cwd=sandbox, env=env,
                               capture_output=True, text=True, check=True)
        result = json.loads(child.stdout) if mode == "reproduce" else {"forecast_csv": child.stdout,
                 "fit_used": False, "live_signals_published": False}
    return {**result, "implementation_commit": RETAINED_COMMIT, "isolated_archive": True,
            "current_source_guard_bypassed": False, "retained_files_changed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--mode", choices=("reproduce", "forecast"), default="reproduce")
    parser.add_argument("--as-of")
    parser.add_argument("--prices", type=Path)
    parser.add_argument("--sentiment", type=Path)
    args = parser.parse_args()
    print(json.dumps(replay(args.run, mode=args.mode, as_of=args.as_of,
                            prices=args.prices, sentiment=args.sentiment), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
