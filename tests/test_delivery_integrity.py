"""Deterministic packaging, restricted-input exclusion, and portable validation."""
import json
from pathlib import Path
import stat
import zipfile

import pytest

from scripts import build_delivery as delivery


def fixture_source(tmp_path):
    root = tmp_path / "source"
    (root / "scripts").mkdir(parents=True)
    (root / "VERSION.json").write_text(json.dumps({"package_version": "2.1.0"}))
    (root / "README.md").write_text("research source\n")
    (root / "scripts/start.sh").write_text("#!/bin/sh\nexit 0\n")
    return root, ["VERSION.json", "README.md", "scripts/start.sh"]


def rewrite_archive(source, target, mutate):
    with zipfile.ZipFile(source) as original, zipfile.ZipFile(target, "w") as changed:
        for item in original.infolist():
            changed.writestr(item, mutate(item.filename, original.read(item.filename)))


def test_deterministic_bytes_ignore_host_mtime_and_archive_filename(tmp_path):
    root, paths = fixture_source(tmp_path)
    first, second = tmp_path / "first.zip", tmp_path / "second.zip"
    a = delivery.build_archive(first, root=root, paths=paths)
    (root / "README.md").touch()
    b = delivery.build_archive(second, root=root, paths=list(reversed(paths)))
    assert first.read_bytes() == second.read_bytes()
    assert a["sha256"] == b["sha256"]
    with zipfile.ZipFile(first) as package:
        assert package.namelist() == sorted(package.namelist())
        for entry in package.infolist():
            assert entry.date_time == delivery.FIXED_TIMESTAMP
        script = package.getinfo(f"{delivery.PREFIX}/scripts/start.sh")
        assert stat.S_IMODE(script.external_attr >> 16) == 0o755
    assert Path(str(first) + ".sha256").read_text().startswith(a["sha256"])


def test_extracted_package_verifies_without_git_and_detects_file_change(tmp_path, monkeypatch):
    root, paths = fixture_source(tmp_path)
    archive = tmp_path / "release.zip"
    delivery.build_archive(archive, root=root, paths=paths)
    target = tmp_path / "clean"
    with zipfile.ZipFile(archive) as package:
        package.extractall(target)
    extracted = target / delivery.PREFIX
    monkeypatch.setattr(delivery.subprocess, "run", lambda *_a, **_k: pytest.fail("verification invoked Git"))
    assert delivery.verify_directory(extracted)["verified"]
    (extracted / "README.md").write_text("changed")
    with pytest.raises(ValueError, match="content hash mismatch"):
        delivery.verify_directory(extracted)


def test_archive_hash_and_manifest_detect_payload_changes(tmp_path):
    root, paths = fixture_source(tmp_path)
    archive, tampered = tmp_path / "release.zip", tmp_path / "tampered.zip"
    evidence = delivery.build_archive(archive, root=root, paths=paths)
    rewrite_archive(archive, tampered, lambda name, raw: b"changed" if name.endswith("README.md") else raw)
    with pytest.raises(ValueError, match="Archive SHA-256 mismatch"):
        delivery.verify_archive(tampered, expected_sha256=evidence["sha256"])
    with pytest.raises(ValueError, match="content hash mismatch"):
        delivery.verify_archive(tampered)


@pytest.mark.parametrize("name", [".env", "credentials.json", "key.pem"])
def test_credential_files_are_rejected_without_leaking_contents(tmp_path, name):
    root, paths = fixture_source(tmp_path)
    (root / name).write_text("private-value")
    with pytest.raises(ValueError, match="Credential-like") as error:
        delivery.build_archive(tmp_path / "release.zip", root=root, paths=paths + [name])
    assert "private-value" not in str(error.value)
    assert not (tmp_path / "release.zip").exists()


def test_high_confidence_credential_content_is_rejected(tmp_path):
    root, paths = fixture_source(tmp_path)
    (root / "config.txt").write_text("ghp_" + "a" * 36)
    with pytest.raises(ValueError, match="Potential credential content"):
        delivery.build_archive(tmp_path / "release.zip", root=root, paths=paths + ["config.txt"])


@pytest.mark.parametrize("name", ["research/v3/inputs/bars.csv", "research/v3/raw/response.json",
                                   "reports/research_v3/run/inputs/bars.csv", "v3-inputs/raw.json",
                                   "data/v3/bars.csv", "data/research_v3/raw/price.json",
                                   "data/finmind_prices.csv"])
def test_restricted_v3_provider_inputs_are_rejected(tmp_path, name):
    root, paths = fixture_source(tmp_path)
    source = root / name
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("restricted research data")
    with pytest.raises(ValueError, match="Restricted provider inputs"):
        delivery.build_archive(tmp_path / "release.zip", root=root, paths=paths + [name])


def test_nested_archives_and_runtime_caches_do_not_enter_package(tmp_path):
    root, paths = fixture_source(tmp_path)
    for name in ("old.zip", "old.zip.sha256", "__pycache__/cache.pyc", ".venv/runtime", ".git/config"):
        source = root / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("excluded")
        paths.append(name)
    archive = tmp_path / "release.zip"
    evidence = delivery.build_archive(archive, root=root, paths=paths)
    assert evidence["files"] == 3
    with zipfile.ZipFile(archive) as package:
        assert all("excluded" not in package.read(name).decode() for name in package.namelist())


@pytest.mark.parametrize("name", ["../outside.txt", "/outside.txt", "C:/outside.txt", "safe/../../outside"])
def test_source_path_traversal_is_rejected_before_writing(tmp_path, name):
    root, paths = fixture_source(tmp_path)
    with pytest.raises(ValueError, match="Unsafe package path"):
        delivery.build_archive(tmp_path / "release.zip", root=root, paths=paths + [name])
    assert not (tmp_path / "release.zip").exists()


def test_symlink_source_cannot_escape_package_root(tmp_path):
    root, paths = fixture_source(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside")
    (root / "linked.txt").symlink_to(outside)
    with pytest.raises(ValueError, match="leaves package root"):
        delivery.build_archive(tmp_path / "release.zip", root=root, paths=paths + ["linked.txt"])


def test_archive_path_traversal_is_rejected_before_extraction(tmp_path):
    source = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(source, "w") as package:
        package.writestr("../outside.txt", "outside")
    with pytest.raises(ValueError, match="Unsafe package path"):
        delivery.verify_archive(source)
    assert not (tmp_path / "outside.txt").exists()


def test_existing_delivery_cannot_be_overwritten(tmp_path):
    root, paths = fixture_source(tmp_path)
    archive = tmp_path / "release.zip"
    delivery.build_archive(archive, root=root, paths=paths)
    wanted = archive.read_bytes()
    with pytest.raises(FileExistsError):
        delivery.build_archive(archive, root=root, paths=paths)
    assert archive.read_bytes() == wanted


def test_zip_modes_are_checked_against_manifest(tmp_path):
    root, paths = fixture_source(tmp_path)
    archive, changed = tmp_path / "release.zip", tmp_path / "changed.zip"
    delivery.build_archive(archive, root=root, paths=paths)
    with zipfile.ZipFile(archive) as original, zipfile.ZipFile(changed, "w") as output:
        for entry in original.infolist():
            if entry.filename.endswith("README.md"):
                entry.external_attr = (stat.S_IFREG | 0o755) << 16
            output.writestr(entry, original.read(entry.filename))
    with pytest.raises(ValueError, match="file mode mismatch"):
        delivery.verify_archive(changed)


@pytest.mark.parametrize("change", ["modify", "delete"])
def test_post_snapshot_source_change_refuses_completed_archive(tmp_path, monkeypatch, change):
    root, paths = fixture_source(tmp_path)
    original_write = delivery.zipfile.ZipFile.writestr
    def writing(package, entry, raw, *args, **kwargs):
        result = original_write(package, entry, raw, *args, **kwargs)
        if entry.filename.endswith("README.md"):
            if change == "modify":
                (root / "README.md").write_text("changed while building")
            else:
                (root / "README.md").unlink()
        return result
    monkeypatch.setattr(delivery.zipfile.ZipFile, "writestr", writing)
    archive = tmp_path / "release.zip"
    with pytest.raises(ValueError, match="source changed"):
        delivery.build_archive(archive, root=root, paths=paths)
    assert not archive.exists()
    assert not Path(str(archive) + ".sha256").exists()


@pytest.mark.parametrize("name", [".env", "data/v3/bars.csv"])
def test_verified_archive_cannot_smuggle_restricted_file_with_updated_manifest(tmp_path, name):
    root, paths = fixture_source(tmp_path)
    archive, changed = tmp_path / "release.zip", tmp_path / "changed.zip"
    delivery.build_archive(archive, root=root, paths=paths)
    raw = b"not publishable"
    with zipfile.ZipFile(archive) as original, zipfile.ZipFile(changed, "w") as output:
        for entry in original.infolist():
            payload = original.read(entry.filename)
            if entry.filename.endswith(delivery.MANIFEST):
                document = json.loads(payload)
                document["files"][name] = {"sha256": delivery.digest(raw), "size": len(raw), "mode": "0644"}
                payload = json.dumps(document).encode()
            output.writestr(entry, payload)
        output.writestr(delivery._entry(f"{delivery.PREFIX}/{name}", 0o644), raw)
    with pytest.raises(ValueError, match="Credential-like|Restricted provider"):
        delivery.verify_archive(changed)


def test_extracted_manifest_symlink_is_rejected(tmp_path):
    root, paths = fixture_source(tmp_path)
    archive = tmp_path / "release.zip"
    delivery.build_archive(archive, root=root, paths=paths)
    with zipfile.ZipFile(archive) as package:
        package.extractall(tmp_path / "clean")
    extracted = tmp_path / "clean" / delivery.PREFIX
    manifest = extracted / delivery.MANIFEST
    outside = tmp_path / "manifest.json"
    manifest.rename(outside)
    manifest.symlink_to(outside)
    with pytest.raises(ValueError, match="Unsafe extracted package manifest"):
        delivery.verify_directory(extracted)
