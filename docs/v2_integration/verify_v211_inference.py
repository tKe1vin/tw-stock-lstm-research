"""Read-only V2.11 frozen-bundle inference check.

Does not train, does not write ledgers/weights/schedules.
Requires a TF-capable venv (e.g. .venv-tf-verify).
"""
from __future__ import annotations

import argparse
import csv
from datetime import date
import hashlib
import io
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

EXPECTED_WEIGHT = "53a1467fdba66e7f22a45e5f2fbc1cb78fc99825b15d516b3e7153d7bfc28144"
WEIGHT = ROOT / "experiments" / "results" / "v2" / "v211" / "weights" / "weights.npz"
CARD = WEIGHT.parent / "model_card.json"
DAILY = ROOT / "experiments" / "results" / "v2" / "v211" / "daily"
TICKERS = frozenset({"2330.TW", "2317.TW", "2603.TW", "2002.TW", "2891.TW"})
EXPECTED_VERSION = "rolling_selected_fold16_window756_carried"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_prediction_rows(rows) -> dict[str, float]:
    """Keep duplicate keys visible until the complete five-stock check passes."""
    pairs = list(rows)
    identities = [str(ticker) for ticker, _value in pairs]
    if len(pairs) != len(TICKERS) or len(set(identities)) != len(pairs) or set(identities) != TICKERS:
        raise ValueError("predictions require exactly five unique registered tickers")
    output = {str(ticker): float(value) for ticker, value in pairs}
    if not all(math.isfinite(value) for value in output.values()):
        raise ValueError("predictions must be finite for all five tickers")
    return output


def load_published_signal(signal_date: str, daily_dir=None) -> dict:
    """Validate the saved signal against its registry, without certifying PIT."""
    if not isinstance(signal_date, str) or date.fromisoformat(signal_date).isoformat() != signal_date:
        raise ValueError("signal date must use YYYY-MM-DD")
    daily_dir = Path(daily_dir) if daily_dir is not None else DAILY
    published = daily_dir / f"{signal_date}.csv"
    registry = json.loads((daily_dir.parent / "signal_registry.json").read_text(encoding="utf-8"))
    entry = registry.get(published.name) if isinstance(registry, dict) else None
    if not isinstance(entry, dict):
        raise ValueError("published signal has no registry entry")
    raw = published.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if (entry.get("file_sha256") != digest or entry.get("signal_date") != signal_date
            or entry.get("rows") != len(TICKERS) or entry.get("model_sha256") != EXPECTED_WEIGHT):
        raise ValueError("published signal registry/hash/date/model/row-count mismatch")
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    fields = reader.fieldnames or []
    required = {"Signal_Date", "Ticker", "Model_Sha256", "Model_Version", "Predicted_Return_Pct"}
    if not required.issubset(fields) or len(fields) != len(set(fields)):
        raise ValueError("published signal columns are missing or duplicated")
    rows = list(reader)
    if any(row["Signal_Date"] != signal_date or row["Model_Sha256"] != EXPECTED_WEIGHT
           or row["Model_Version"] != EXPECTED_VERSION for row in rows):
        raise ValueError("published signal row date/model mismatch")
    expected = validate_prediction_rows((row["Ticker"], row["Predicted_Return_Pct"]) for row in rows)
    return {"predictions": expected, "file_sha256": digest, "historical_availability_certified": False}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal-date", default="2026-11-05")
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args()
    if not math.isfinite(args.atol) or args.atol < 0:
        parser.error("--atol must be finite and nonnegative")

    report = {
        "ok": False,
        "weight_sha256": None,
        "weight_ok": False,
        "bundle_ok": False,
        "card_ok": False,
        "signal_registry_ok": False,
        "prediction_contract_ok": False,
        "signal_date": args.signal_date,
        "predictions": {},
        "expected": {},
        "max_abs_diff": None,
        "matched": False,
        "blockers": [],
    }

    if not WEIGHT.exists() or not CARD.exists():
        report["blockers"].append("missing weights.npz or model_card.json")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 2

    weight_hash = _sha(WEIGHT)
    report["weight_sha256"] = weight_hash
    report["weight_ok"] = weight_hash == EXPECTED_WEIGHT
    if not report["weight_ok"]:
        report["blockers"].append(f"weight hash mismatch: {weight_hash}")

    card = json.loads(CARD.read_text(encoding="utf-8"))
    report["card_ok"] = bool(
        card.get("weight_sha256") == EXPECTED_WEIGHT
        and card.get("model_version") == EXPECTED_VERSION
        and "scalers" in card
        and isinstance(card.get("features"), list)
        and len(card.get("features") or []) > 0
    )
    if not report["card_ok"]:
        report["blockers"].append("model_card.json contract incomplete or mismatched")

    try:
        from scripts.model_bundle import verify_bundle

        verify_bundle(weight_dir=WEIGHT.parent, card=card)
        report["bundle_ok"] = True
    except Exception as exc:  # noqa: BLE001 — report exact blocker
        report["bundle_ok"] = False
        report["blockers"].append(f"verify_bundle failed: {exc}")

    try:
        signal = load_published_signal(args.signal_date)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report["blockers"].append(f"published signal contract failed: {exc}")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 3
    expected = signal["predictions"]
    report["expected"] = expected
    report["published_signal_sha256"] = signal["file_sha256"]
    report["signal_registry_ok"] = True

    try:
        import tensorflow  # noqa: F401
        from experiments.v2.v211_market import load_market_context, market_files_ready
        from experiments.v2.v211 import predict_rows
    except Exception as exc:  # noqa: BLE001
        report["blockers"].append(f"TF/inference import failed: {exc}")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 4

    try:
        if not market_files_ready():
            raise RuntimeError("v211 market snapshot not ready (prices/sentiment)")
        context = load_market_context()
        report["context_source"] = "v211_market"
        preds, *_ = predict_rows(card, context, args.signal_date)
        import pandas as pd

        if isinstance(preds, pd.DataFrame):
            output = validate_prediction_rows((row["Ticker"], row["Predicted_Return_Pct"])
                                              for _, row in preds.iterrows())
        else:
            output = validate_prediction_rows(preds.items())
        report["predictions"] = output
        report["prediction_contract_ok"] = True
    except Exception as exc:  # noqa: BLE001
        report["blockers"].append(f"predict_rows failed: {exc}")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 5

    diffs = []
    for ticker, exp in expected.items():
        got = output.get(ticker)
        if got is None:
            report["blockers"].append(f"missing prediction for {ticker}")
            continue
        diffs.append(abs(got - exp))
    report["max_abs_diff"] = max(diffs) if diffs else None
    report["matched"] = bool(diffs) and report["max_abs_diff"] is not None and report["max_abs_diff"] <= args.atol
    report["ok"] = (
        report["weight_ok"]
        and report["card_ok"]
        and report["bundle_ok"]
        and report["signal_registry_ok"]
        and report["prediction_contract_ok"]
        and report["matched"]
        and not report["blockers"]
    )
    if not report["matched"] and report["max_abs_diff"] is not None:
        report["blockers"].append(
            f"prediction mismatch max_abs_diff={report['max_abs_diff']} atol={args.atol}"
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
