"""State persistence protects signals and excludes code, credentials and raw runs."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from scripts.persist_daily_state import (
    BOOK_ROOTS, TICKERS, WEIGHT_SHA256, allowed_path, changed_state, validate_state,
)
import scripts.persist_daily_state as persistence


@pytest.fixture
def repo(tmp_path):
    def run(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)
    run("init", "-b", "main")
    run("config", "user.name", "Test")
    run("config", "user.email", "test@example.invalid")
    (tmp_path / "README.md").write_text("fixture\n")
    run("add", ".")
    run("commit", "-m", "fixture")
    return tmp_path, run


def write_signal(root: Path):
    book = root / BOOK_ROOTS[0]
    daily = book / "daily"
    daily.mkdir(parents=True)
    signal = daily / "2026-11-06.csv"
    columns = ["Signal_Date", "Ticker", "Model_Sha256", "Execution_Price", "Actual_5d_Pct"]
    with signal.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for ticker in sorted(TICKERS):
            writer.writerow({"Signal_Date": signal.stem, "Ticker": ticker, "Model_Sha256": WEIGHT_SHA256})
    entry = {"model_sha256": WEIGHT_SHA256, "file_sha256": hashlib.sha256(signal.read_bytes()).hexdigest()}
    registry = book / "signal_registry.json"
    registry.write_text(json.dumps({signal.name: entry}))
    return signal, registry


def test_allowlist_persists_only_compact_durable_state():
    allowed = [
        "data/news_scored_cache.csv", "data/prices/historical_prices.csv",
        "experiments/results/v2/v211/market/state.json",
        "experiments/results/v2/v211/market/news_scan_coverage.json",
        "experiments/results/v2/v211/daily/2026-11-06.csv",
        "experiments/results/v2/v211/live_fills.csv",
        "research/v211/independent_ledger/signal_registry.json",
        "research/v211/independent_ledger/market/prices_seed.csv",
        "research/v211/independent_ledger/market/sentiment_seed.csv",
        "outputs/555/2026-11-06/unified_signal.csv",
        "outputs/555/2026-11-06/native/strategy_metrics.csv",
        "outputs/444/2026-11-06/native/evaluation_protocol.json",
        "outputs/daily_summary/2026-11-06/summary.json",
        "outputs/444/recommendation_history/run.csv", "reports/latest/555/strategy_metrics.csv",
    ]
    rejected = [
        ".env", "scripts/run_all_daily.py", "data/snapshots/2026-11-06/prices.csv",
        "outputs/555/2026-11-06/runs/large.csv", "outputs/555/2026-11-06/chart.png",
        "outputs/555/2026-11-06/native/historical_prices.csv",
        "experiments/results/v2/v211/weights/weights.npz",
        "experiments/results/v2/v211/market/prices_seed.csv",
        "experiments/results/v2/v211/market/yahoo_staging.csv",
        "research/v211/signal_registry/developer_original/2026-11-05.csv",
        "reports/latest/.env", "../data/news_update_state.json", "/data/news_update_state.json",
    ]
    assert all(allowed_path(name) for name in allowed)
    assert not any(allowed_path(name) for name in rejected)


def test_candidate_scan_ignores_code_and_snapshots(repo):
    root, _run = repo
    for name in ["scripts/new.py", "data/news_update_state.json", "data/snapshots/day/prices.csv"]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    assert changed_state(root) == ["data/news_update_state.json"]


def test_symlink_state_is_rejected(repo, tmp_path):
    root, _run = repo
    outside = tmp_path / "outside.txt"
    outside.write_text("private")
    (root / "data").mkdir()
    (root / "data/news_update_state.json").symlink_to(outside)
    with pytest.raises(ValueError, match="regular file"):
        changed_state(root)


def test_immutable_hash_mismatch_is_rejected(repo):
    root, _run = repo
    path = root / "frozen.npz"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Immutable artifact"):
        validate_state(root, {"frozen.npz": hashlib.sha256(b"original").hexdigest()})


def test_new_signal_requires_matching_registry(repo):
    root, _run = repo
    _signal, registry = write_signal(root)
    validate_state(root, {})
    content = json.loads(registry.read_text())
    next(iter(content.values()))["file_sha256"] = "wrong"
    registry.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_state(root, {})


def test_published_signal_cannot_change_even_with_rehashed_registry(repo):
    root, run = repo
    signal, registry = write_signal(root)
    run("add", ".")
    run("commit", "-m", "publish")
    signal.write_bytes(signal.read_bytes() + b"\n")
    content = json.loads(registry.read_text())
    content[signal.name]["file_sha256"] = hashlib.sha256(signal.read_bytes()).hexdigest()
    registry.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="Published registry entry"):
        validate_state(root, {})


def test_published_signal_cannot_be_deleted(repo):
    root, run = repo
    signal, _registry = write_signal(root)
    run("add", ".")
    run("commit", "-m", "publish")
    signal.unlink()
    with pytest.raises(ValueError, match="Published signal changed or missing"):
        validate_state(root, {})


def test_market_seed_must_match_manifest(repo):
    root, _run = repo
    market = root / BOOK_ROOTS[1] / "market"
    market.mkdir(parents=True)
    manifest = {}
    for name, key in [("prices_seed.csv", "price_seed_sha256"), ("sentiment_seed.csv", "sentiment_seed_sha256")]:
        (market / name).write_bytes(b"seed")
        manifest[key] = hashlib.sha256(b"seed").hexdigest()
    (market / "manifest.json").write_text(json.dumps(manifest))
    validate_state(root, {})
    (market / "prices_seed.csv").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Market seed hash mismatch"):
        validate_state(root, {})


def test_default_is_read_only_and_commit_excludes_code(repo, monkeypatch):
    root, _run = repo
    monkeypatch.setattr(persistence, "ROOT", root)
    monkeypatch.setattr(persistence, "IMMUTABLE_SHA256", {})
    (root / "data").mkdir()
    (root / "scripts").mkdir()
    (root / "data/news_update_state.json").write_text("{}")
    (root / "scripts/unrelated.py").write_text("raise Exception('not state')")
    before = persistence.git(root, "rev-parse", "HEAD").stdout
    assert persistence.main([]) == 0
    assert persistence.git(root, "rev-parse", "HEAD").stdout == before
    assert not persistence.git(root, "diff", "--cached", "--name-only").stdout
    assert persistence.main(["--commit"]) == 0
    tracked = persistence.git(root, "ls-tree", "-r", "--name-only", "HEAD").stdout.decode()
    assert "data/news_update_state.json" in tracked
    assert "scripts/unrelated.py" not in tracked
