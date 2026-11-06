"""Paths and write guards for 555 V2. Official artifacts stay read-only."""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "results" / "v2"
PHASE5 = Path(__file__).resolve().parents[1] / "results" / "signal_models"
SENTIMENT = Path(__file__).resolve().parents[1] / "data" / "sentiment_experimental.csv"

REQUIRED_PROTECTED = (
    ROOT / "notebooks" / "444_baseline.ipynb",
    ROOT / "notebooks" / "555_multivariate.ipynb",
    ROOT / "data" / "sentiment_features_2026.csv",
    ROOT / "data" / "prices" / "historical_prices.csv",
    ROOT / ".github" / "workflows" / "daily-lstm.yml",
    ROOT / "experiments" / "results" / "signal_models" / "strategy_metrics.csv",
    ROOT / "reports" / "latest" / "444" / "strategy_metrics.csv",
    ROOT / "reports" / "latest" / "555" / "strategy_metrics.csv",
)

# These historical research exports may remain in the original delivery ZIP.
# They are not inputs to the frozen daily tracker, but any local copy must stay
# unchanged while a research command runs, including its presence or absence.
OPTIONAL_PROTECTED = (
    ROOT / "experiments" / "results" / "walk_forward_predictions.csv",
    ROOT / "experiments" / "results" / "signal_models" / "predictions.csv",
)
PROTECTED = REQUIRED_PROTECTED + OPTIONAL_PROTECTED
MISSING_OPTIONAL = "missing:optional-archive"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot():
    saved = {str(path.relative_to(ROOT)): sha256(path) for path in REQUIRED_PROTECTED}
    saved.update({
        str(path.relative_to(ROOT)): sha256(path) if path.exists() else MISSING_OPTIONAL
        for path in OPTIONAL_PROTECTED
    })
    return saved


def assert_unchanged(before):
    after = snapshot()
    if after != before:
        changed = sorted(key for key in set(before) | set(after) if before.get(key) != after.get(key))
        raise AssertionError(f"V2 改動了正式或既有研究成果：{changed}")


def assert_v2_output(path):
    resolved = Path(path).resolve()
    root = OUT.resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"拒絕寫到 V2 結果目錄以外：{resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved
