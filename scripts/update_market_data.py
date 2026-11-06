"""Update the shared append-only market ledger and materialize model snapshots.

All four systems use this source. Bundled reproducibility fixtures remain
untouched. ``allow_download=False`` only reads existing files.
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common import TZ, current_completed_session, sha256  # noqa: E402


def snapshot_dir(completed) -> Path:
    return ROOT / "data" / "snapshots" / str(pd.Timestamp(completed).date())


def sentiment_ready(path: Path, completed) -> bool:
    from experiments.v2.v211_market import sentiment_day_ready

    return path.exists() and sentiment_day_ready(pd.read_csv(path), completed)


def _copy_snapshot(source: Path, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def build_snapshot(now=None, allow_download=True) -> dict:
    """Fetch once, then give each model identical canonical input files.

    In read-only mode readiness describes the canonical source; the separate
    ``snapshot_*_ready`` fields report whether model snapshot files exist.
    """
    from experiments.v2 import v211_market as market

    now = now or datetime.now(TZ)
    completed = pd.Timestamp(current_completed_session(now)).normalize()
    folder = snapshot_dir(completed)
    if allow_download:
        try:
            report = market.update_market(now)
        except Exception as exc:
            report = market.inspect_market(now)
            report["update_error"] = f"{type(exc).__name__}: {exc}"
    else:
        report = market.inspect_market(now)
    report.update({
        "snapshot_dir": folder.relative_to(ROOT).as_posix(),
        "used_previous_day_as_today": False,
        "read_only": not allow_download,
        "shared_source": "v211_market",
    })
    if allow_download:
        folder.mkdir(parents=True, exist_ok=True)
        for field, source, filename in (
            ("prices", market.PRICES, "prices.csv"),
            ("sentiment", market.SENTIMENT, "sentiment.csv"),
        ):
            if report.get(f"{field}_ready"):
                _copy_snapshot(source, folder / filename)
    for field, filename, validator in (
        ("prices", "prices.csv", market.price_day_ready),
        ("sentiment", "sentiment.csv", market.sentiment_day_ready),
    ):
        path = folder / filename
        try:
            ready = path.exists() and validator(pd.read_csv(path), completed)
        except (OSError, ValueError, KeyError):
            ready = False
        report[f"snapshot_{field}_ready"] = bool(ready)
        if ready:
            report[f"snapshot_{field}_sha256"] = sha256(path)
    if allow_download:
        temporary = folder / "manifest.json.tmp"
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(folder / "manifest.json")
    return report


def main() -> int:
    report = build_snapshot(allow_download=True)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report["prices_ready"] and report["sentiment_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
