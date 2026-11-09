"""Durable, content-addressed research inputs and exact portfolio replay.

Sources are compressed losslessly; timestamps and file names cannot change their
identity. This reconstructs portfolio calculations from saved predictions, not a
claim that stochastic TensorFlow training is bitwise reproducible.
"""
from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

from scripts.common import ROOT, atomic_json


def preserve_inputs(sources: dict[str, Path], destination: Path, *, root: Path = ROOT) -> dict:
    root, destination = Path(root), Path(destination)
    archive = root / "data" / "research_snapshots"
    inputs = {}
    for label, source in sources.items():
        data = Path(source).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        target = archive / f"{digest}.gz"
        if target.exists():
            if gzip.decompress(target.read_bytes()) != data:
                raise ValueError(f"Research archive hash collision or corruption: {target}")
        else:
            archive.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".gz.tmp")
            temporary.write_bytes(gzip.compress(data, mtime=0))
            temporary.replace(target)
        resolved = Path(source).resolve()
        original = resolved.relative_to(root.resolve()).as_posix() if resolved.is_relative_to(root.resolve()) else resolved.name
        inputs[label] = {"sha256": digest, "archive": target.relative_to(root).as_posix(),
                         "original_path": original, "bytes": len(data)}
    code = {}
    for folder in ("scripts", "experiments"):
        for source in sorted((root / folder).rglob("*.py")):
            code[source.relative_to(root).as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
    dependencies = {}
    for name in ("numpy", "pandas", "scikit-learn", "tensorflow-cpu", "exchange-calendars"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True)
    manifest = {"schema": "research-replay-v2", "inputs": inputs, "source_sha256": code,
                "git_head": revision.stdout.strip() if revision.returncode == 0 else None,
                "dependencies": dependencies, "replay_scope": "exact saved-prediction portfolio replay",
                "training_bitwise_reproduction_claimed": False}
    atomic_json(destination, manifest)
    return manifest


def restore_inputs(manifest_path: Path, destination: Path, *, root: Path = ROOT) -> dict[str, Path]:
    """Verify all sources before materializing an isolated replay directory."""
    root, destination = Path(root).resolve(), Path(destination)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    verified = {}
    for label, record in manifest["inputs"].items():
        if not label.replace("_", "").isalnum():
            raise ValueError("Invalid replay input label")
        archive = (root / record["archive"]).resolve()
        if not archive.is_relative_to(root / "data" / "research_snapshots"):
            raise ValueError("Replay archive leaves research snapshot directory")
        data = gzip.decompress(archive.read_bytes())
        if hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError(f"Replay input failed verification: {label}")
        verified[label] = data
    destination.mkdir(parents=True, exist_ok=True)
    paths = {}
    for label, data in verified.items():
        path = destination / f"{label}.csv"
        path.write_bytes(data)
        paths[label] = path
    return paths
