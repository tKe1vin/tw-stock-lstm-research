"""Frozen rolling_selected weights and the known 2026-11-05 prediction."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = "53a1467fdba66e7f22a45e5f2fbc1cb78fc99825b15d516b3e7153d7bfc28144"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_weight_file_hash_is_computed():
    path = ROOT / "experiments" / "results" / "v2" / "v211" / "weights" / "weights.npz"
    assert _sha256(path) == EXPECTED
    assert _sha256(ROOT / "models" / "model_555_v2" / "weights" / "weights.npz") == EXPECTED


def test_known_signal_day_prediction_matches_published_rows():
    import experiments.v2.v211 as tracker
    from experiments.v2.v211_market import load_market_context

    daily = ROOT / "experiments" / "results" / "v2" / "v211" / "daily"
    before = sorted(path.name for path in daily.glob("*.csv"))
    context = load_market_context()
    card = tracker.fit_and_save(context)
    rows, execution, _five = tracker.predict_rows(card, context, "2026-11-05")
    published = pd.read_csv(daily / "2026-11-05.csv")
    got = rows.set_index("Ticker")
    for _, item in published.iterrows():
        diff = abs(float(got.loc[item.Ticker, "Predicted_Return_Pct"]) - float(item.Predicted_Return_Pct))
        assert diff < 1e-4, f"{item.Ticker} diff {diff}"
        assert abs(float(got.loc[item.Ticker, "Target_Weight"]) - float(item.Target_Weight)) < 1e-12
    assert execution == "2026-11-06"
    assert card["weight_sha256"] == EXPECTED
    assert sorted(path.name for path in daily.glob("*.csv")) == before
