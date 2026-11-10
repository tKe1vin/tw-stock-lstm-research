"""Build deterministic research ZIPs; verify ZIP/extracted hashes without Git.

Builds use reviewed files from a Git checkout. Restricted new provider inputs,
credentials, caches, previous archives and environment directories are excluded
or rejected. Existing published inputs/models remain in the source package.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[1]
VERSION = "2.1.0"
PREFIX = f"TW_Stock_LSTM_Research_{VERSION}"
MANIFEST = "PACKAGE_MANIFEST.json"
FIXED_TIMESTAMP = (2026, 11, 10, 0, 0, 0)
IGNORED_PARTS = {".git", "__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".cache",
                 ".venv", "venv", "node_modules", "logs", ".codex"}
SECRET_NAMES = {"credentials.json", "credentials", "secrets.json", "secrets.toml",
                "id_rsa", "id_ed25519", "service-account.json", "service_account.json"}
SECRET_PATTERNS = (
    rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
    rb"\bgh[pousr]_[A-Za-z0-9]{20,}\b",
    rb"\bgithub_pat_[A-Za-z0-9_]{30,}\b",
    rb"\bAKIA[0-9A-Z]{16}\b",
    rb"\bsk-(?:proj-)?[A-Za-z0-9_-]{32,}\b",
)
MAX_TOTAL_BYTES = 256 * 1024 * 1024


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _relative_name(value):
    name = str(value)
    path = PurePosixPath(name)
    if (not name or "\\" in name or path.is_absolute() or ":" in name
            or any(part in ("", ".", "..") for part in name.split("/"))):
        raise ValueError("Unsafe package path")
    return path.as_posix()


def _excluded(name):
    path = PurePosixPath(name)
    lower = path.name.lower()
    return (bool(set(path.parts).intersection(IGNORED_PARTS)) or lower.endswith((".zip", ".zip.sha256", ".pyc", ".tmp"))
            or name == MANIFEST)


def _reject_sensitive(name, raw):
    path = PurePosixPath(name)
    lower = path.name.lower()
    sample_env = lower.endswith((".example", ".sample", ".template"))
    if (lower in SECRET_NAMES or (lower.startswith(".env") and not sample_env)
            or lower.endswith((".pem", ".p12", ".pfx", ".key"))):
        raise ValueError(f"Credential-like file cannot be packaged: {name}")
    parts = tuple(part.lower() for part in path.parts)
    restricted_area = parts[:2] in (("research", "v3"), ("reports", "research_v3"),
                                  ("data", "v3"), ("data", "research_v3"))
    provider_named_data = (lower.startswith("finmind_")
                           and lower.endswith((".csv", ".json", ".jsonl", ".parquet", ".gz")))
    if ("v3-inputs" in parts or "finmind" in parts or provider_named_data or (restricted_area and
            ("raw" in parts or "inputs" in parts or lower in ("bars.csv", "raw_manifest.json")))):
        raise ValueError(f"Restricted provider inputs cannot be packaged: {name}")
    if any(re.search(pattern, raw) for pattern in SECRET_PATTERNS):
        raise ValueError(f"Potential credential content cannot be packaged: {name}")


def checkout_paths(root):
    """List versioned plus nonignored new files; never traverse unrelated paths."""
    try:
        result = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                                cwd=root, capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise ValueError("Building a release requires a Git checkout; verification does not") from exc
    return sorted(set(item.decode("utf-8") for item in result.stdout.split(b"\0") if item))


def _source_bytes(root, paths):
    root = Path(root).resolve()
    payloads = {}
    for value in sorted(set(paths)):
        name = _relative_name(value)
        if _excluded(name):
            continue
        source = root / name
        if not source.resolve().is_relative_to(root):
            raise ValueError("Source path leaves package root")
        if any(part.is_symlink() for part in (source, *source.parents) if part != root and part.is_relative_to(root)):
            raise ValueError(f"Symlinks cannot be packaged: {name}")
        if not source.is_file():
            raise ValueError(f"Missing reviewed source file: {name}")
        raw = source.read_bytes()
        _reject_sensitive(name, raw)
        payloads[name] = raw
    if not payloads or sum(map(len, payloads.values())) > MAX_TOTAL_BYTES:
        raise ValueError("Empty or oversized package")
    return payloads


def _entry(name, mode):
    entry = zipfile.ZipInfo(name, FIXED_TIMESTAMP)
    entry.create_system = 3
    entry.compress_type = zipfile.ZIP_DEFLATED
    entry.external_attr = (stat.S_IFREG | mode) << 16
    return entry


def _snapshot_matches(root, payloads):
    try:
        return all((root / name).read_bytes() == raw for name, raw in payloads.items())
    except OSError:
        return False


def build_archive(output_path, *, root=ROOT, paths=None):
    """Snapshot reviewed files and create a NEW deterministic archive + SHA file."""
    root, output = Path(root).resolve(), Path(output_path).resolve()
    checksum = Path(str(output) + ".sha256")
    if output.exists() or checksum.exists():
        raise FileExistsError("Delivery artifacts are immutable; choose a new output path")
    payloads = _source_bytes(root, checkout_paths(root) if paths is None else paths)
    version = json.loads(payloads.get("VERSION.json", b"{}"))
    if version.get("package_version") != VERSION:
        raise ValueError(f"VERSION.json must identify reviewed package {VERSION}")
    files = {name: {"sha256": digest(raw), "size": len(raw),
                    "mode": "0755" if name.endswith(".sh") else "0644"}
             for name, raw in payloads.items()}
    manifest = {"schema": "research-delivery-v1", "package_version": VERSION,
                "archive_prefix": PREFIX, "files": files,
                "provider_raw_data_included": False,
                "all_v3_source_inputs_included": False,
                "historical_archive_replay_requires_git": True,
                "integrity_is_not_authenticity_or_performance_proof": True}
    encoded = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    # Verify the source snapshot stayed stable before creating an artifact.
    if not _snapshot_matches(root, payloads):
        raise ValueError("Reviewed source changed during archive construction")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as package:
        for name, raw in sorted({**payloads, MANIFEST: encoded}.items()):
            mode = int(files[name]["mode"], 8) if name != MANIFEST else 0o644
            package.writestr(_entry(f"{PREFIX}/{name}", mode), raw, compresslevel=9)
    if not _snapshot_matches(root, payloads):
        output.unlink()
        raise ValueError("Reviewed source changed during archive construction")
    archive_hash = digest(output.read_bytes())
    with checksum.open("x", encoding="utf-8") as handle:
        handle.write(f"{archive_hash}  {output.name}\n")
    verify_archive(output, expected_sha256=archive_hash)
    return {"archive": str(output), "sha256": archive_hash, "files": len(files),
            "package_version": VERSION, "provider_raw_data_included": False}


def _manifest(raw):
    document = json.loads(raw)
    if (document.get("schema") != "research-delivery-v1" or document.get("package_version") != VERSION
            or document.get("archive_prefix") != PREFIX or not isinstance(document.get("files"), dict)
            or document.get("provider_raw_data_included") is not False
            or document.get("all_v3_source_inputs_included") is not False):
        raise ValueError("Unsupported package manifest")
    for name, evidence in document["files"].items():
        _relative_name(name)
        if (name == MANIFEST or not isinstance(evidence, dict)
                or not re.fullmatch(r"[0-9a-f]{64}", str(evidence.get("sha256", "")))
                or type(evidence.get("size")) is not int or evidence["size"] < 0
                or evidence.get("mode") not in ("0644", "0755")):
            raise ValueError("Malformed package file evidence")
    return document


def verify_archive(path, *, expected_sha256=None):
    path = Path(path)
    if expected_sha256 is not None and digest(path.read_bytes()) != expected_sha256:
        raise ValueError("Archive SHA-256 mismatch")
    with zipfile.ZipFile(path) as package:
        entries = package.infolist()
        names = [entry.filename for entry in entries]
        if len(set(names)) != len(names) or sum(entry.file_size for entry in entries) > MAX_TOTAL_BYTES:
            raise ValueError("Duplicate or oversized package entries")
        for entry in entries:
            name = _relative_name(entry.filename)
            file_type = stat.S_IFMT(entry.external_attr >> 16)
            if (not name.startswith(PREFIX + "/") or entry.is_dir()
                    or file_type not in (0, stat.S_IFREG)):
                raise ValueError("Unsafe archive entry")
        manifest_name = f"{PREFIX}/{MANIFEST}"
        if manifest_name not in names:
            raise ValueError("Package manifest is missing")
        document = _manifest(package.read(manifest_name))
        if set(names) != {manifest_name, *(f"{PREFIX}/{name}" for name in document["files"])}:
            raise ValueError("Package file membership mismatch")
        for name, evidence in document["files"].items():
            archive_name = f"{PREFIX}/{name}"
            if stat.S_IMODE(package.getinfo(archive_name).external_attr >> 16) != int(evidence["mode"], 8):
                raise ValueError(f"Package file mode mismatch: {name}")
            raw = package.read(archive_name)
            if len(raw) != evidence["size"] or digest(raw) != evidence["sha256"]:
                raise ValueError(f"Package content hash mismatch: {name}")
            _reject_sensitive(name, raw)
    return {"verified": True, "files": len(document["files"]), "package_version": VERSION,
            "all_v3_source_inputs_included": False}


def verify_directory(root):
    """Verify extracted source files. Normal Python/pytest caches are ignored."""
    root = Path(root).resolve()
    manifest_path = root / MANIFEST
    if manifest_path.is_symlink() or not manifest_path.resolve().is_relative_to(root):
        raise ValueError("Unsafe extracted package manifest")
    document = _manifest(manifest_path.read_bytes())
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*")
              if path.is_file() and not set(path.relative_to(root).parts).intersection(IGNORED_PARTS)}
    if actual != set(document["files"]) | {MANIFEST}:
        raise ValueError("Extracted package file membership mismatch")
    for name, evidence in document["files"].items():
        source = root / name
        if source.is_symlink() or not source.resolve().is_relative_to(root):
            raise ValueError("Unsafe extracted package path")
        raw = source.read_bytes()
        if len(raw) != evidence["size"] or digest(raw) != evidence["sha256"]:
            raise ValueError(f"Extracted package content hash mismatch: {name}")
    return {"verified": True, "files": len(document["files"]), "package_version": VERSION}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output", type=Path)
    action.add_argument("--verify", type=Path)
    action.add_argument("--verify-directory", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.output:
        result = build_archive(args.output, root=args.root)
    elif args.verify:
        result = verify_archive(args.verify, expected_sha256=args.sha256)
    else:
        result = verify_directory(args.verify_directory)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
