"""Validate and persist compact daily state without rewriting published signals.

The default command is read-only. --commit stages only the explicit state
allowlist; --push uses an ordinary push and at most one conflict-checked rebase.
Large training runs are retained by the workflow as artifacts instead.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path, PurePosixPath
import re
import subprocess
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
WEIGHT_SHA256 = "53a1467fdba66e7f22a45e5f2fbc1cb78fc99825b15d516b3e7153d7bfc28144"
SIGNAL_SHA256 = "37591d21300977b9486212fc86fe257bb050db47177fa2f41bd7a8904919f659"
IMMUTABLE_SHA256 = {
    "experiments/results/v2/v211/weights/weights.npz": WEIGHT_SHA256,
    "models/model_555_v2/weights/weights.npz": WEIGHT_SHA256,
    "experiments/results/v2/v211/daily/2026-11-05.csv": SIGNAL_SHA256,
    "research/v211/signal_registry/developer_original/2026-11-05.csv": SIGNAL_SHA256,
}
BOOK_ROOTS = (
    "experiments/results/v2/v211",
    "research/v211/independent_ledger",
)
DATA_FILES = {
    "data/prices/historical_prices.csv",
    "data/prices/historical_prices.meta.json",
    "data/sentiment_features_2026.csv",
    "data/news_scored_cache.csv",
    "data/news_update_state.json",
    "data/news_scored_cache_provenance.jsonl",
}
MARKET_FILES = {
    "prices.csv", "sentiment.csv", "news_cache.csv", "manifest.json",
    "state.json", "fetch_audit.jsonl", "news_scan_coverage.json", "news_cache_provenance.jsonl",
}
BOOK_FILES = {
    "signal_registry.json", "status.json", "matured_scores.csv",
    "live_fills.csv", "live_nav.csv", "equal_nav.csv", "live_book.csv",
    "LEDGER_ORIGIN.json", "signal_audit.json",
}
COMPACT_EXPORTS = {
    "status.json", "unified_signal.csv", "unified_signal.metadata.json",
    "prediction.csv", "dates.json",
}
NATIVE_EXPORTS = {
    "strategy_metrics.csv", "evaluation_protocol.json", "next_day_recommendation.csv",
    "model_metrics.csv", "backtest_nav.csv", "test_predictions.csv", "market_metrics.csv",
    "daily_portfolio_review.csv",
}
DATE = re.compile(r"\d{4}-\d{2}-\d{2}\Z")
TICKERS = {"2330.TW", "2317.TW", "2603.TW", "2002.TW", "2891.TW"}


def git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(["git", *args], cwd=root, capture_output=True)
    if check and result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())
    return result


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def allowed_path(name: str) -> bool:
    """Accept only durable state paths; never scripts, credentials or raw runs."""
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        return False
    if name in DATA_FILES:
        return True
    if (len(path.parts) == 3 and path.parts[:2] == ("data", "research_snapshots")
            and bool(re.fullmatch(r"[0-9a-f]{64}\.gz", path.name))):
        return True
    for book in BOOK_ROOTS:
        prefix = book + "/"
        if name.startswith(prefix):
            suffix = name[len(prefix):]
            parts = PurePosixPath(suffix).parts
            return (
                suffix in BOOK_FILES
                or len(parts) == 2 and parts[0] == "market" and parts[1] in MARKET_FILES
                or book == "research/v211/independent_ledger"
                and suffix in {"market/prices_seed.csv", "market/sentiment_seed.csv"}
                or len(parts) == 2 and parts[0] == "daily"
                and parts[1].endswith(".csv") and bool(DATE.fullmatch(parts[1][:-4]))
            )
    parts = path.parts
    if len(parts) == 5 and parts[0] == "outputs":
        return (
            parts[1] in {"444", "555"} and bool(DATE.fullmatch(parts[2]))
            and parts[3] == "native" and parts[4] in NATIVE_EXPORTS
        )
    if len(parts) == 4 and parts[0] == "outputs":
        if parts[1] in {"444", "555"} and parts[2] in {"recommendation_history", "paired_recommendation_history"}:
            return parts[3].endswith(".csv")
        if parts[1] == "daily_summary" and DATE.fullmatch(parts[2]):
            return parts[3] == "summary.json"
        return (
            parts[1] in {"444", "555", "555_v2", "v211"}
            and bool(DATE.fullmatch(parts[2])) and parts[3] in COMPACT_EXPORTS
        )
    if len(parts) >= 3 and parts[0] == "reports" and parts[1] in {"latest", "daily", "journals"}:
        return path.suffix in {".csv", ".json", ".md"} and not any(p.startswith(".") for p in parts)
    return False


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def signal_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): digest(path)
        for book in BOOK_ROOTS for path in sorted((root / book / "daily").glob("*.csv"))
    }


def validate_state(root: Path, immutable: dict[str, str] | None = None) -> None:
    """Check frozen artifacts, append-only committed signals and registry hashes."""
    expected = IMMUTABLE_SHA256 if immutable is None else immutable
    for name, wanted in expected.items():
        path = root / name
        if path.is_symlink() or not path.is_file() or digest(path) != wanted:
            raise ValueError(f"Immutable artifact changed or missing: {name}")
    for book in BOOK_ROOTS:
        base = root / book
        registry_path = base / "signal_registry.json"
        daily = base / "daily"
        paths = sorted(daily.glob("*.csv"))
        tracked = git(root, "ls-tree", "-r", "--name-only", "-z", "HEAD", "--", f"{book}/daily/")
        manifest_path = base / "market" / "manifest.json"
        if manifest_path.exists():
            manifest = _json(manifest_path)
            for filename, key in (("prices_seed.csv", "price_seed_sha256"), ("sentiment_seed.csv", "sentiment_seed_sha256")):
                seed = base / "market" / filename
                if seed.is_symlink() or not seed.is_file() or digest(seed) != manifest.get(key):
                    raise ValueError(f"Market seed hash mismatch: {seed}")
                previous_seed = git(root, "show", f"HEAD:{book}/market/{filename}", check=False)
                if previous_seed.returncode == 0 and seed.read_bytes() != previous_seed.stdout:
                    raise ValueError(f"Committed market seed changed: {seed}")
        if not registry_path.exists() and not paths and not tracked.stdout:
            continue
        registry = _json(registry_path)
        previous = git(root, "show", f"HEAD:{book}/signal_registry.json", check=False)
        if previous.returncode == 0:
            for name, entry in json.loads(previous.stdout).items():
                if registry.get(name) != entry:
                    raise ValueError(f"Published registry entry changed or missing: {book}/{name}")
        for raw in tracked.stdout.split(b"\0"):
            if not raw:
                continue
            name = raw.decode("utf-8")
            old = git(root, "show", f"HEAD:{name}").stdout
            path = root / name
            if not path.is_file() or path.is_symlink() or path.read_bytes() != old:
                raise ValueError(f"Published signal changed or missing: {name}")
        if set(registry) != {path.name for path in paths}:
            raise ValueError(f"Registry and daily signal filenames differ: {book}")
        for path in paths:
            if path.is_symlink() or not DATE.fullmatch(path.stem):
                raise ValueError(f"Invalid signal path: {path}")
            entry = registry[path.name]
            if entry.get("file_sha256") != digest(path) or entry.get("model_sha256") != WEIGHT_SHA256:
                raise ValueError(f"Signal registry hash mismatch: {path}")
            with path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            if len(rows) != 5 or {r.get("Ticker") for r in rows} != TICKERS:
                raise ValueError(f"Signal must contain five unique stock rows: {path}")
            for row in rows:
                if row.get("Signal_Date") != path.stem or row.get("Model_Sha256") != WEIGHT_SHA256:
                    raise ValueError(f"Signal date/model mismatch: {path}")
                if row.get("Execution_Price") or row.get("Actual_5d_Pct"):
                    raise ValueError(f"Published signal contains retrospective results: {path}")


def changed_state(root: Path) -> list[str]:
    modified = git(root, "diff", "--name-only", "-z", "HEAD", "--").stdout
    untracked = git(root, "ls-files", "--others", "--exclude-standard", "-z").stdout
    names = {raw.decode("utf-8") for raw in (modified + untracked).split(b"\0") if raw}
    selected = sorted(name for name in names if allowed_path(name))
    for name in selected:
        path = root / name
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"State must be a regular file inside the repository: {name}")
    return selected


def push_state(root: Path, branch: str, preserved_signals: dict[str, str]) -> None:
    git(root, "check-ref-format", "--branch", branch)
    target = f"HEAD:refs/heads/{branch}"
    if git(root, "push", "origin", target, check=False).returncode == 0:
        return
    if git(root, "diff", "--quiet", check=False).returncode:
        raise RuntimeError("Push failed and unrelated tracked files are dirty; refusing automatic rebase")
    git(root, "fetch", "origin", f"refs/heads/{branch}:refs/remotes/origin/{branch}")
    if git(root, "rebase", f"refs/remotes/origin/{branch}", check=False).returncode:
        git(root, "rebase", "--abort", check=False)
        raise RuntimeError("Concurrent state changes conflict; rebase aborted without overwriting either ledger")
    validate_state(root)
    current = signal_hashes(root)
    if any(current.get(name) != value for name, value in preserved_signals.items()):
        raise RuntimeError("A published signal changed during rebase; refusing push")
    git(root, "push", "origin", target)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", action="store_true", help="stage and commit the validated allowlist")
    parser.add_argument("--push", action="store_true", help="push the state commit; requires --commit")
    args = parser.parse_args(argv)
    if args.push and not args.commit:
        parser.error("--push requires --commit")
    validate_state(ROOT)
    selected = changed_state(ROOT)
    print(json.dumps({"read_only": not args.commit, "state_files": selected}, ensure_ascii=False), flush=True)
    if not args.commit or not selected:
        return 0
    staged = git(ROOT, "diff", "--cached", "--name-only", "-z").stdout
    if staged:
        raise ValueError("The Git index already contains staged changes; refusing to include pre-existing staged content")
    git(ROOT, "add", "--", *selected)
    day = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d")
    git(ROOT, "commit", "-m", f"research: daily state {day} [skip ci]")
    if args.push:
        branch = os.environ.get("GITHUB_REF_NAME") or git(ROOT, "branch", "--show-current").stdout.decode().strip()
        push_state(ROOT, branch, signal_hashes(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
