"""The public research package cannot replace inputs or redistribute raw quotes."""
import gzip
import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from scripts import research_v2_common as publication


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def test_replay_restores_exact_old_inputs_outside_repo_without_replacing_live_files(tmp_path):
    pair = tmp_path / publication.PAIRED
    pair.mkdir(parents=True)
    records = {}
    for name in ("prices", "sentiment"):
        original = tmp_path / f"live/{name}.csv"
        original.parent.mkdir(exist_ok=True)
        original.write_bytes(b"new daily state remains untouched\n")
        raw = f"synthetic archived {name}\n".encode()
        archive = tmp_path / f"archives/{name}.gz"
        archive.parent.mkdir(exist_ok=True)
        archive.write_bytes(gzip.compress(raw))
        records[name] = {"original_path": str(original.relative_to(tmp_path)),
                         "archive": str(archive.relative_to(tmp_path)), "bytes": len(raw), "sha256": sha(raw)}
    (pair / "nav.csv").write_bytes(b"saved canonical navigation\n")
    (pair / "reproduction_manifest.json").write_text(json.dumps({
        "inputs": records, "outputs_sha256": {"nav.csv": sha((pair / "nav.csv").read_bytes())}}))
    before = {str(path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    with publication.retained_inputs(tmp_path) as (paths, _):
        for name, path in paths.items():
            assert not path.is_relative_to(tmp_path)
            assert sha(path.read_bytes()) == records[name]["sha256"]
    assert {str(path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before
    assert all(not path.exists() for path in paths.values())


def synthetic_model_run(folder, filename="model.weights.npz"):
    folder.mkdir()
    raw = b"synthetic inference artifact"
    (folder / filename).write_bytes(raw)
    manifest = {"schema": "v2-common-reproduction-v1", "raw_inputs_copied": False,
                "files_sha256": {filename: sha(raw)}}
    (folder / "reproduction_manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_model_package_rejects_raw_inputs_and_keeps_existing_evidence(tmp_path):
    run = tmp_path / "run"
    synthetic_model_run(run, "prices.csv")
    archive = tmp_path / "public.zip"
    with pytest.raises(ValueError, match="Raw provider"):
        publication.pack_models(run, archive)
    assert not archive.exists()
    other = tmp_path / "clean-run"
    synthetic_model_run(other)
    publication.pack_models(other, archive)
    before = archive.read_bytes()
    with zipfile.ZipFile(archive) as result:
        assert set(result.namelist()) == {"model.weights.npz", "reproduction_manifest.json"}
    with pytest.raises(FileExistsError):
        publication.pack_models(other, archive)
    assert archive.read_bytes() == before
    (other / "model.weights.npz").write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="hash"):
        publication.pack_models(other, tmp_path / "changed.zip")


def test_public_archive_paths_cannot_escape_retained_root(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        publication.inside(tmp_path, "../different-research/model.npz")
    external = tmp_path / "external"
    external.symlink_to(tmp_path.parent)
    with pytest.raises(ValueError, match="escapes"):
        publication.inside(tmp_path, "external/other-file")
