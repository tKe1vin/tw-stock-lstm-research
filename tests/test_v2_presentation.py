"""A shared chart layout cannot blur open NAV, execution costs or source modes."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import pytest

from dashboard import v2_presentation as presentation

ROOT = Path(__file__).resolve().parents[1]
BOOK = "experiments/results/v2/v211"


def copy_book(tmp_path):
    shutil.copytree(ROOT / BOOK, tmp_path / BOOK)
    script = tmp_path / "experiments/risk_experiment.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "experiments/risk_experiment.py", script)
    return tmp_path / BOOK


def checkpoint(book, relative):
    path = book / relative
    proof_path = book / "ledger_integrity.json"
    proof = json.loads(proof_path.read_text())
    proof["prefixes"][relative] = {"bytes": len(path.read_bytes()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    proof_path.write_text(json.dumps(proof))


def test_curves_and_metrics_match_both_saved_books_and_keep_the_entry_fee():
    study = presentation.load_v2_presentation()
    assert study["initial_capital"] == 100000
    assert study["model_arm"] == "v211" and study["benchmark_arm"] == "equal_policy"
    assert study["diagnostics"]["summary"]["signal_days"] >= 3
    metric = study["metrics"].set_index("Arm")
    for arm, filename in (("v211", "live_nav.csv"), ("equal_policy", "equal_nav.csv")):
        raw = pd.read_csv(ROOT / BOOK / filename).sort_values("Date")
        curve = study["curves"][study["curves"].Arm.eq(arm)].sort_values("Date")
        expected = raw.NAV_Open.to_numpy() / 1000
        np.testing.assert_allclose(curve.Wealth_Index, expected, rtol=0, atol=1e-12)
        peaks = np.maximum.accumulate(np.r_[100, expected])[1:]
        drawdown = (expected / peaks - 1) * 100
        np.testing.assert_allclose(curve.Drawdown_Pct, drawdown, rtol=0, atol=1e-12)
        assert curve.Wealth_Index.iloc[0] < 100  # The initial fee is not rebased away.
        assert metric.loc[arm, "ROI_Open_Pct"] == pytest.approx(expected[-1] - 100)
        assert metric.loc[arm, "ROI_Pct"] == metric.loc[arm, "ROI_Open_Pct"]
        assert metric.loc[arm, "MDD_Pct"] == pytest.approx(min(drawdown))
        expected_close = (raw.NAV_Open.iloc[-1] - raw.Liquidation_If_Closed.iloc[-1]) / 1000 - 100
        assert metric.loc[arm, "ROI_If_Closed_Pct"] == pytest.approx(expected_close)
        endpoint = study["nav"][study["nav"].Arm.eq(arm)].sort_values("Date")
        assert endpoint.NAV_If_Closed.notna().sum() == 1
        assert endpoint.NAV_If_Closed.iloc[:-1].isna().all()
        assert endpoint.NAV_If_Closed.iloc[-1] == pytest.approx(raw.NAV_Open.iloc[-1] - raw.Liquidation_If_Closed.iloc[-1])
    assert study["curves"].groupby("Arm").Date.nunique().nunique() == 1
    assert study["period_start"] == "2026-11-06"
    assert any("0050" in message for message in study["missing"])


def test_costs_are_actual_fill_sums_and_terminal_liquidation_is_separate():
    study = presentation.load_v2_presentation()
    fills = pd.read_csv(ROOT / BOOK / "live_fills.csv")
    for arm, side, filename in (("v211", "published", "live_nav.csv"), ("equal_policy", "equal", "equal_nav.csv")):
        source = fills[fills.Book.eq(side)]
        cost = study["costs"].set_index("Arm").loc[arm]
        metrics = study["metrics"].set_index("Arm").loc[arm]
        nav = pd.read_csv(ROOT / BOOK / filename)
        assert cost.Actual_Execution_Cost == pytest.approx(source.Cost.sum())
        assert cost.Buy_Fee == pytest.approx(source.Buy_Fee.sum())
        assert cost.Sell_Tax == pytest.approx(source.Sell_Tax.sum())
        assert metrics.Total_Transaction_Cost == pytest.approx(source.Cost.sum())
        assert cost.Hypothetical_Liquidation_Cost == pytest.approx(nav.Liquidation_If_Closed.iloc[-1])
        assert metrics.Total_Transaction_Cost < cost.Actual_Execution_Cost + cost.Hypothetical_Liquidation_Cost
    weights = study["exposure"].groupby(["Date", "Arm"]).Weight_Pct.sum()
    np.testing.assert_allclose(weights, 100)


def test_actual_holdings_reconcile_shares_and_prices_and_are_not_latest_targets():
    study = presentation.load_v2_presentation()
    holdings = study["holdings"].set_index("Ticker")
    targets = study["target_weights"].set_index("Ticker")
    nav = pd.read_csv(ROOT / BOOK / "live_nav.csv").iloc[-1]
    fills = pd.read_csv(ROOT / BOOK / "live_fills.csv")
    execution = fills[fills.Book.eq("published")].sort_values("Execution_Date").iloc[-1]
    prices = pd.read_csv(ROOT / BOOK / "market/prices.csv")
    closes = prices[prices.Date.eq(nav.Date)].set_index("ticker").Close
    for ticker in presentation.TICKERS:
        shares = float(execution["Shares_" + ticker])
        assert holdings.loc[ticker, "Shares"] == shares
        assert holdings.loc[ticker, "Weight_Pct"] == pytest.approx(shares * closes[ticker] / nav.NAV_Open * 100)
        assert holdings.loc[ticker, "Source_Execution_Date"] == execution.Execution_Date
    assert holdings.Weight_Pct.sum() == pytest.approx(100)
    assert holdings.loc["CASH", "Weight_Pct"] == pytest.approx(nav.Cash_Weight * 100)
    assert study["holdings_reconciliation"]["NAV_Recomputed"] == pytest.approx(nav.NAV_Open)
    assert not np.allclose(holdings.loc[list(presentation.TICKERS), "Weight_Pct"], targets.loc[list(presentation.TICKERS), "Weight_Pct"])
    assert any(shares != int(shares) for shares in holdings.Shares.dropna())  # Preserve fractional shares.


def test_timeline_keeps_hypothetical_entry_rebalance_fill_and_maturity_distinct():
    study = presentation.load_v2_presentation()
    timeline = study["timeline"].set_index("Signal_Date")
    assert timeline.loc["2026-11-05", "Forecast_Entry_Date"] == "2026-11-06"
    assert timeline.loc["2026-11-05", "Label_End_Date"] == "2026-11-13"
    assert timeline.loc["2026-11-06", "Scheduled_Rebalance_Date"] == "2026-11-13"
    assert timeline.loc["2026-11-06", "Forecast_Entry_Date"] == "2026-11-09"
    assert not timeline.loc["2026-11-06", "Trade_Eligible"]
    assert pd.isna(timeline.loc["2026-11-06", "Simulated_Execution_Date"])
    assert timeline.loc["2026-11-17", "Forecast_Entry_Date"] == "2026-11-18"
    assert timeline.loc["2026-11-17", "Scheduled_Rebalance_Date"] == "2026-11-20"
    assert not timeline.loc["2026-11-17", "Trade_Eligible"]
    assert pd.isna(timeline.loc["2026-11-17", "Simulated_Execution_Date"])


def test_explicit_demo_has_no_saved_diagnostics_fees_or_holdings_fallback(monkeypatch):
    monkeypatch.setattr(presentation, "load_v2_diagnostics", lambda *_args: pytest.fail("DEMO cannot read saved diagnostics"))
    demo = presentation.load_v2_presentation(mode="demo")
    assert demo["is_demo"] is True and demo["diagnostics"] is None
    assert demo["status"]["data_kind"] == "SYNTHETIC_DEMO"
    assert demo["holdings"].empty and demo["holdings_reconciliation"] is None
    assert demo["costs"].Actual_Execution_Cost.isna().all()
    assert not demo["costs"].Cost_Breakdown_Available.any()
    assert all(name.startswith("dashboard_v2/fixtures/") or name == "experiments/risk_experiment.py" for name in demo["sources"])
    assert any("DEMO" in item for item in demo["missing"])


def test_changed_nav_hash_and_semantically_unpaired_dates_fail_closed(tmp_path):
    book = copy_book(tmp_path)
    path = book / "equal_nav.csv"
    original = path.read_bytes()
    path.write_bytes(original + b"\n")
    with pytest.raises(presentation.V2PresentationError, match="完整性|hash"):
        presentation.load_v2_presentation(tmp_path)
    nav = pd.read_csv(path).iloc[1:]
    nav.to_csv(path, index=False)
    checkpoint(book, "equal_nav.csv")
    with pytest.raises(presentation.V2PresentationError, match="配對"):
        presentation.load_v2_presentation(tmp_path)


def test_nonreconciling_holdings_are_missing_and_never_replaced_by_target_weights(tmp_path):
    book = copy_book(tmp_path)
    path = book / "live_fills.csv"
    fills = pd.read_csv(path)
    fills.loc[fills.Book.eq("published"), "Shares_2330.TW"] *= 2
    fills.to_csv(path, index=False)
    checkpoint(book, "live_fills.csv")
    study = presentation.load_v2_presentation(tmp_path)
    assert study["available"] is True and not study["metrics"].empty
    assert study["holdings"].empty and study["holdings_reconciliation"] is None
    assert not study["target_weights"].empty
    assert any("持股估值加現金" in message for message in study["missing"])


def test_inconsistent_initial_capital_cannot_rebase_away_the_disagreement(tmp_path):
    copy_book(tmp_path)
    source = tmp_path / "experiments/risk_experiment.py"
    text = source.read_text()
    assert "INITIAL_CAPITAL = 100_000.0" in text
    source.write_text(text.replace("INITIAL_CAPITAL = 100_000.0", "INITIAL_CAPITAL = 200_000.0"))
    with pytest.raises(presentation.V2PresentationError, match="初始資金"):
        presentation.load_v2_presentation(tmp_path)


def test_mid_load_update_is_rejected_instead_of_returning_mixed_snapshots(tmp_path, monkeypatch):
    book = copy_book(tmp_path)
    original = presentation.v211_status_local

    def change_after_status():
        result = original()
        path = book / "status.json"
        path.write_bytes(path.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(presentation, "v211_status_local", change_after_status)
    with pytest.raises(presentation.V2PresentationError, match="載入期間變動"):
        presentation.load_v2_presentation(tmp_path)


def test_source_modes_are_concurrent_offline_read_only_and_ignore_private_environment(monkeypatch, tmp_path):
    import requests

    files = [path for path in (ROOT / BOOK).rglob("*") if path.is_file() and "__pycache__" not in path.parts]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    for variable in ("TW_V211_LOCAL", "TW_RIDGE_ROOT", "TW_GITHUB_REVIEW_ROOT"):
        monkeypatch.setenv(variable, str(tmp_path / variable))
    environment = dict(os.environ)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Presentation may not access providers")

    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
    for name in ("tensorflow", "torch", "transformers"):
        monkeypatch.setitem(sys.modules, name, None)
    with ThreadPoolExecutor(2) as workers:
        saved_future = workers.submit(presentation.load_v2_presentation, ROOT, "saved")
        demo_future = workers.submit(presentation.load_v2_presentation, ROOT, "demo")
        saved, demo = saved_future.result(), demo_future.result()
    assert saved["status"]["data_kind"] == "LIVE_READONLY" and not saved["is_demo"]
    assert demo["status"]["data_kind"] == "SYNTHETIC_DEMO" and demo["is_demo"]
    assert saved["ridge"]["status"] == "DATA_UNAVAILABLE"
    assert saved["diagnostics"] is not None and demo["diagnostics"] is None
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in files} == before
    assert dict(os.environ) == environment
