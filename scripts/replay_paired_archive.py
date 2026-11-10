"""Verify the original paired report in its isolated, pinned implementation."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

from scripts.common import ROOT
from scripts.replay_extension_archive import RETAINED_COMMIT


def replay(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("source_sha256", {})
    if not expected or not manifest.get("outputs_sha256"):
        raise ValueError("A retained paired source/output contract is required")
    try:
        raw = subprocess.run(["git", "archive", RETAINED_COMMIT], cwd=ROOT,
                             capture_output=True, check=True).stdout
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise ValueError("Paired historical replay requires Git commit " + RETAINED_COMMIT) from exc
    with tempfile.TemporaryDirectory(prefix="paired-archive-") as folder:
        sandbox = Path(folder)
        with tarfile.open(fileobj=io.BytesIO(raw)) as package:
            package.extractall(sandbox, filter="data")
        runtime_sources = ("scripts/paired_research.py", "scripts/common.py",
                           "scripts/reproducibility.py", "scripts/paired_analysis.py",
                           "scripts/baseline_adapter.py")
        for name in runtime_sources:
            wanted = expected.get(name)
            path = (sandbox / name).resolve()
            if not path.is_relative_to(sandbox) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != wanted:
                raise ValueError("Historical paired source differs: " + name)
        child = subprocess.run([sys.executable, "-m", "scripts.paired_delivery", "--replay",
                                str(manifest_path), "--output", str(sandbox / "replayed-book")],
                               cwd=sandbox, env={**os.environ, "PYTHONPATH": str(sandbox)},
                               capture_output=True, text=True, check=True)
        result = json.loads(child.stdout)
    return {"status": result["status"], "implementation_commit": RETAINED_COMMIT,
            "output_csv_hashes_matched": True, "fit_used": False,
            "verified_runtime_sources": list(runtime_sources),
            "isolated_archive": True, "retained_files_changed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(replay(args.manifest), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
