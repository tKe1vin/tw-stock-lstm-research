"""Forecast dates cannot turn an off-cycle signal into a portfolio trade."""
from datetime import datetime
import hashlib
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from experiments.v2 import v211

TZ = ZoneInfo("Asia/Taipei")


def signal(day, entry):
    return pd.DataFrame([{
        "Signal_Date": day, "Planned_Execution_Date": entry, "Ticker": ticker,
        "Target_Weight": 0.1, "Risk_Off": True, "Close": 100.0,
    } for ticker in v211.TICKERS])


def panel():
    index = pd.bdate_range("2026-11-06", "2026-11-13")
    return pd.DataFrame(100.0, index=index, columns=v211.TICKERS)


def test_november_9_is_forecast_entry_but_not_rebalance():
    result = v211.execution_eligibility("2026-11-06", "2026-11-06")
    assert result["forecast_entry_date"] == "2026-11-09"
    assert result["scheduled_rebalance_date"] == "2026-11-13"
    assert result["trade_eligible"] is False
    due = v211.execution_eligibility("2026-11-12", "2026-11-06")
    assert due["forecast_entry_date"] == due["scheduled_rebalance_date"] == "2026-11-13"
    assert due["trade_eligible"] is True


def test_off_cycle_forecast_never_fills_or_carries_to_november_13():
    signals = pd.concat([signal("2026-11-05", "2026-11-06"), signal("2026-11-06", "2026-11-09")])
    result = v211.build_live_book(panel(), signals)
    assert set(result["fills"].Execution_Date) == {"2026-11-06"}
    daily = result["daily"].set_index("Date")
    assert not bool(daily.loc[pd.Timestamp("2026-11-09"), "Trade"])
    assert bool(daily.loc[pd.Timestamp("2026-11-13"), "Skipped"])


def test_november_13_uses_only_november_12_signal():
    signals = pd.concat([signal("2026-11-05", "2026-11-06"), signal("2026-11-06", "2026-11-09"),
                         signal("2026-11-12", "2026-11-13")])
    result = v211.build_live_book(panel(), signals)
    fills = result["fills"]
    assert set(fills.loc[fills.Execution_Date == "2026-11-13", "Signal_Date"]) == {"2026-11-12"}


def test_old_signal_cannot_be_relabelled_as_next_rebalance():
    signals = pd.concat([signal("2026-11-05", "2026-11-06"), signal("2026-11-06", "2026-11-13")])
    with pytest.raises(RuntimeError, match="不能挪用舊訊號"):
        v211.build_live_book(panel(), signals)


def test_signal_published_after_execution_close_cannot_fill():
    signals = signal("2026-11-05", "2026-11-06")
    signals["Created_At"] = "2026-11-06T13:31:00+08:00"
    with pytest.raises(RuntimeError, match="拒絕事後模擬成交"):
        v211.build_live_book(panel(), signals)


def test_legacy_audit_preserves_old_signal_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(v211, "V211", tmp_path)
    monkeypatch.setattr(v211, "LIVE", tmp_path / "daily")
    v211.LIVE.mkdir()
    path = v211.LIVE / "2026-11-06.csv"
    frame = signal("2026-11-06", "2026-11-09")
    frame["Created_At"] = "2026-11-06T22:56:20.508919+08:00"
    frame.to_csv(path, index=False)
    before = path.read_bytes()
    # Imported ledger's anchor is still November 6.
    anchor = signal("2026-11-05", "2026-11-06")
    anchor["Created_At"] = "2026-11-05T23:23:09.114628+08:00"
    anchor.to_csv(v211.LIVE / "2026-11-05.csv", index=False)
    audit = v211.write_signal_audit()["signals"][path.name]
    assert audit["missing_source_fields"] == ["Price_Sha256", "Sentiment_Sha256", "Fetched_At"]
    assert audit["execution"]["trade_eligible"] is False
    assert path.read_bytes() == before


def test_created_at_is_post_prediction_clock(tmp_path, monkeypatch):
    monkeypatch.setattr(v211, "ROOT", tmp_path)
    monkeypatch.setattr(v211, "V211", tmp_path)
    monkeypatch.setattr(v211, "LIVE", tmp_path / "daily")
    monkeypatch.setattr(v211, "WEIGHT_DIR", tmp_path / "weights")
    v211.LIVE.mkdir()
    v211.WEIGHT_DIR.mkdir()
    (v211.WEIGHT_DIR / "weights.npz").write_bytes(b"fixture")
    card = {"weight_sha256": hashlib.sha256(b"fixture").hexdigest()}
    monkeypatch.setattr(v211, "verify_bundle", lambda **kwargs: {"bundle_sha256": "f" * 64})
    rows = pd.DataFrame([{"Ticker": ticker, "Target_Weight": 0.1} for ticker in v211.TICKERS])
    monkeypatch.setattr(v211, "predict_rows", lambda *_args: (rows, "2026-11-09", None))
    decision = datetime(2026, 11, 6, 17, 0, tzinfo=TZ)
    published = datetime(2026, 11, 6, 17, 3, tzinfo=TZ)
    source = {"price_sha256": "a" * 64, "sentiment_sha256": "b" * 64,
              "fetched_at": decision.isoformat()}
    v211.publish_today(decision, (panel(),), card, source, publication_clock=lambda: published)
    saved = pd.read_csv(v211.LIVE / "2026-11-06.csv")
    assert set(saved.Created_At) == {published.isoformat()}
    assert set(saved.Decision_At) == {decision.isoformat()}
    assert set(saved.Data_Through) == {"2026-11-06"}
