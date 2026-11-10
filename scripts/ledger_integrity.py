"""Append-only price and realized-ledger guards for the V2.11 tracker.

Git HEAD is the independent committed baseline. A portable checkpoint protects
later appends without Git, but its first observation cannot authenticate older
history. Hashes establish integrity, never historical information availability.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess

import pandas as pd

CHECKPOINT = "ledger_integrity.json"
AUDIT = "ledger_source_audit.jsonl"
SOURCE_FILES = ("market/prices.csv", "market/sentiment.csv")
REALIZED_FILES = ("matured_scores.csv", "live_fills.csv", "live_nav.csv", "equal_nav.csv")


def digest_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def _head(root, path):
    try:
        name = Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return None
    try:
        result = subprocess.run(["git", "show", "HEAD:" + name], cwd=root, capture_output=True)
    except FileNotFoundError:
        # Extracted packages may not have Git installed. A prior portable
        # checkpoint still protects bytes, while first observation is unverified.
        return None
    return result.stdout if result.returncode == 0 else None


def _prefix(path, length, wanted):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Committed ledger input/output missing or unsafe: {path.name}")
    raw = path.read_bytes()
    if len(raw) < length or digest_bytes(raw[:length]) != wanted:
        raise ValueError(f"Committed ledger history changed: {path.name}")


def _later_source_rows(path, old):
    """A byte append must not backfill a prior date under a different row key."""
    before = pd.read_csv(io.BytesIO(old))
    after = pd.read_csv(path)
    if "Date" not in before or "Date" not in after:
        raise ValueError("Market ledger source requires Date")
    if len(after) > len(before) and len(before):
        last = pd.to_datetime(before.Date, errors="raise").max()
        if not pd.to_datetime(after.Date.iloc[len(before):], errors="raise").gt(last).all():
            raise ValueError(f"Market ledger can only append later dates: {path.name}")


def validate_ledger_sources(root, book):
    """Read-only; reject changes to all committed post-seed prices and results."""
    root, book = Path(root), Path(book)
    checked = 0
    for name in (*SOURCE_FILES, *REALIZED_FILES, AUDIT):
        path = book / name
        old = _head(root, path)
        if old is not None:
            _prefix(path, len(old), digest_bytes(old))
            if name in SOURCE_FILES:
                _later_source_rows(path, old)
            checked += 1
    checkpoint = book / CHECKPOINT
    documents = []
    old_checkpoint = _head(root, checkpoint)
    if old_checkpoint is not None:
        documents.append(json.loads(old_checkpoint))
    if checkpoint.exists():
        if checkpoint.is_symlink():
            raise ValueError("Ledger checkpoint cannot be a symlink")
        documents.append(json.loads(checkpoint.read_text(encoding="utf-8")))
    for document in documents:
        if document.get("schema_version") != 1:
            raise ValueError("Unknown ledger integrity checkpoint schema")
        for name, evidence in document["prefixes"].items():
            if name not in (*SOURCE_FILES, *REALIZED_FILES):
                raise ValueError("Unknown ledger checkpoint path")
            _prefix(book / name, evidence["bytes"], evidence["sha256"])
            if name in SOURCE_FILES:
                _later_source_rows(book / name, (book / name).read_bytes()[:evidence["bytes"]])
            checked += 1
    return {"checked_baselines": checked,
            "baseline": "committed_or_prior_checkpoint" if checked else "legacy_first_observation_unverified"}


def _frame(frame, keys):
    result = frame.copy()
    if not set(keys).issubset(result.columns):
        raise ValueError("Realized ledger key columns are missing")
    if result[list(keys)].isna().any().any():
        raise ValueError("Realized ledger keys cannot be missing")
    for key in keys:
        if "Date" in key:
            result[key] = pd.to_datetime(result[key], errors="raise").dt.strftime("%Y-%m-%d")
        else:
            result[key] = result[key].astype(str)
    if result[list(keys)].isna().any().any() or result.duplicated(list(keys)).any():
        raise ValueError("Realized ledger keys must be unique and finite")
    return result


def _same(left, right):
    if pd.isna(left) or pd.isna(right):
        return bool(pd.isna(left) and pd.isna(right))
    try:
        a, b = float(left), float(right)
    except (ValueError, TypeError):
        return str(left) == str(right)
    return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-10)


def append_plan(path, candidate, keys, *, snapshot_columns=()):
    """Validate first, returning only unseen rows; never rewrite realized rows.

    snapshot_columns are informational endpoint estimates, not realized NAV or
    fills. Their old recorded values remain unchanged as the live end advances.
    """
    path = Path(path)
    proposed = _frame(candidate, keys)
    if path.is_symlink():
        raise ValueError("Realized ledger cannot be a symlink")
    if not path.exists():
        return proposed, True
    existing = _frame(pd.read_csv(path), keys)
    if set(existing.columns) != set(proposed.columns):
        raise ValueError(f"Realized ledger schema changed: {path.name}")
    proposed = proposed[list(existing.columns)]
    keyed = proposed.set_index(list(keys), drop=False)
    for _, row in existing.iterrows():
        key = tuple(row[name] for name in keys)
        key = key[0] if len(key) == 1 else key
        if key not in keyed.index:
            raise ValueError(f"Previously realized ledger row is missing: {path.name}: {key}")
        other = keyed.loc[key]
        for column in existing.columns:
            if column not in snapshot_columns and not _same(row[column], other[column]):
                raise ValueError(f"Previously realized ledger row changed: {path.name}: {key}: {column}")
    old_keys = set(map(tuple, existing[list(keys)].itertuples(index=False, name=None)))
    fresh = proposed[[tuple(row) not in old_keys for row in proposed[list(keys)].itertuples(index=False, name=None)]]
    return fresh, False


def apply_append(path, plan):
    fresh, create = plan
    if fresh.empty:
        return 0
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh.to_csv(path, mode="w" if create else "a", header=create, index=False)
    return len(fresh)


def record_ledger_sources(root, book, panel, operation, added_rows):
    """Record provenance only for new outputs, not for unchanged historical rows."""
    if not any(added_rows.values()):
        return
    book = Path(book)
    prefixes = {}
    for name in (*SOURCE_FILES, *REALIZED_FILES):
        path = book / name
        if path.exists():
            raw = path.read_bytes()
            prefixes[name] = {"bytes": len(raw), "sha256": digest_bytes(raw)}
    old_path = book / CHECKPOINT
    prior = json.loads(old_path.read_text()) if old_path.exists() else None
    record = {
        "schema_version": 1, "recorded_at": datetime.now(timezone.utc).isoformat(),
        "operation": operation, "new_rows": added_rows,
        "panel_sha256": digest_bytes(panel.to_csv(date_format="%Y-%m-%d", float_format="%.17g").encode()),
        "file_prefixes": prefixes,
        "historical_availability_certified": False,
        "prior_output_provenance": "earlier_checkpoint" if prior else "legacy_rows_not_retroactively_certified",
        "signals_sha256": {path.name: digest_bytes(path.read_bytes()) for path in sorted((book / "daily").glob("*.csv"))},
    }
    with (book / AUDIT).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    checkpoint = {"schema_version": 1, "prefixes": prefixes,
                  "historical_availability_certified": False,
                  "initial_history_status": prior.get("initial_history_status") if prior else "legacy_first_observation_unverified"}
    temporary = old_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(old_path)
