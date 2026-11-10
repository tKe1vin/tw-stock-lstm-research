"""V3 raw-price, share-event, payment-date and retained-source guardrails."""
import json

import numpy as np
import pandas as pd
import pytest

from research.v3 import data


def sample():
    rows = []
    for date, close, split, tradable in (("2025-06-10", 100.0, 1.0, True),
                                        ("2025-06-11", np.nan, 1.0, False),
                                        ("2025-06-18", 25.0, 4.0, True),
                                        ("2025-06-19", 26.0, 1.0, True)):
        rows.append({"Date": date, "Ticker": "0050.TW", "Open": close, "High": close,
                     "Low": close, "Close": close, "Volume": 100 if tradable else 0,
                     "Stock_Splits": split, "Dividends": 0.0, "Return_of_Capital": 0.0,
                     "Tradable": tradable, "Dividend_Payment_Date": "",
                     "Return_of_Capital_Payment_Date": ""})
    metadata = {"close_basis": "truly_raw_unadjusted", "dividend_basis": "cash_per_post_event_share",
                "fetched_at": "2025-06-19T09:00:00Z", "suspensions": data.SUSPENSIONS}
    return pd.DataFrame(rows), metadata


def test_exact_split_does_not_erase_three_quarters_of_wealth():
    bars, metadata = sample()
    wealth = data.total_return_index(bars, metadata=metadata)["0050.TW"]
    np.testing.assert_allclose(wealth, [100, 100, 100, 104])
    assert bars.loc[1, "Tradable"] == False
    assert np.isnan(bars.loc[1, "Close"])


def test_already_split_adjusted_close_cannot_be_multiplied_again():
    bars, metadata = sample()
    bars.loc[0, ["Open", "High", "Low", "Close"]] = 25
    with pytest.raises(ValueError, match="already split-adjusted"):
        data.total_return_index(bars, metadata=metadata)


def test_split_dividend_basis_is_per_resulting_share():
    bars, metadata = sample()
    bars.loc[2, "Dividends"] = 1
    bars.loc[2, "Dividend_Payment_Date"] = "2025-07-01"
    wealth = data.total_return_index(bars, metadata=metadata)["0050.TW"]
    assert wealth.iloc[2] == pytest.approx(104)
    # The TR index measures an ex-date reinvestment convention. The separate
    # payment column must not falsely imply the cash was available at ex-date.
    assert bars.loc[2, "Dividend_Payment_Date"] == "2025-07-01"


def test_capital_return_is_per_original_share():
    bars, metadata = sample()
    bars = bars.iloc[[0, 2]].copy()
    bars.loc[:, "Ticker"] = "2603.TW"
    bars.loc[bars.index[0], ["Open", "High", "Low", "Close"]] = 80.8
    bars.loc[bars.index[1], ["Open", "High", "Low", "Close"]] = 187
    bars.loc[bars.index[1], "Stock_Splits"] = 0.4
    bars.loc[bars.index[1], "Return_of_Capital"] = 6
    bars.loc[bars.index[1], "Return_of_Capital_Payment_Date"] = "2025-06-29"
    wealth = data.total_return_index(bars, metadata=metadata)
    np.testing.assert_allclose(wealth["2603.TW"], [100, 100])


@pytest.mark.parametrize("share_factor,dividend,capital_return", [(0.4, 0.0, 6.0), (1.0, 2.0, 0.0),
                                                                (2.0, 1.0, 3.0)])
def test_suspended_distribution_detaches_cash_from_stale_share_mark(share_factor, dividend, capital_return):
    bars, metadata = sample()
    bars = bars.iloc[:2].copy()
    bars.loc[1, "Stock_Splits"] = share_factor
    bars.loc[1, "Dividends"] = dividend
    bars.loc[1, "Return_of_Capital"] = capital_return
    bars.loc[1, "Dividend_Payment_Date"] = "2025-06-30"
    bars.loc[1, "Return_of_Capital_Payment_Date"] = "2025-06-30"
    wealth = data.total_return_index(bars, metadata=metadata)
    np.testing.assert_allclose(wealth["0050.TW"], [100, 100])


def test_future_share_actions_do_not_change_prior_index_values():
    bars, metadata = sample()
    prefix = data.total_return_index(bars.iloc[:1], metadata=metadata)
    full = data.total_return_index(bars, metadata=metadata)
    np.testing.assert_array_equal(prefix.to_numpy(), full.iloc[:1].to_numpy())


def test_unknown_missing_quote_is_rejected_instead_of_forward_filled():
    bars, metadata = sample()
    metadata["suspensions"] = []
    with pytest.raises(ValueError, match="Unverified suspension"):
        data.validate_data(bars, metadata=metadata)


def test_missing_asset_session_cannot_create_an_intersection_calendar():
    bars, metadata = sample()
    other = bars.copy()
    other["Ticker"] = "2330.TW"
    other = other.loc[other.Tradable].copy()
    # Give the stock a valid quote on the benchmark's halted session, then
    # intentionally omit a different required market session for the ETF.
    stock_halt_day = other.iloc[[0]].copy()
    stock_halt_day["Date"] = "2025-06-11"
    all_rows = pd.concat([bars.iloc[:-1], other, stock_halt_day])
    with pytest.raises(ValueError, match="Incomplete market-session"):
        data.validate_data(all_rows, metadata=metadata)


@pytest.mark.parametrize("column,value", [("Close", 0), ("Open", np.nan),
                                         ("Volume", 0), ("High", -1),
                                         ("Stock_Splits", 0), ("Dividends", np.inf)])
def test_invalid_tradable_values_fail_closed(column, value):
    bars, metadata = sample()
    bars.loc[0, column] = value
    with pytest.raises(ValueError):
        data.validate_data(bars, metadata=metadata)


def test_duplicate_dates_are_rejected():
    bars, metadata = sample()
    with pytest.raises(ValueError, match="duplicate"):
        data.validate_data(pd.concat([bars, bars.iloc[:1]]), metadata=metadata)


def test_asof_and_fetched_at_must_support_reported_close():
    bars, metadata = sample()
    with pytest.raises(ValueError, match="beyond"):
        data.validate_data(bars, metadata=metadata, as_of="2025-06-18")
    metadata["fetched_at"] = "2025-06-19T01:00:00Z"
    with pytest.raises(ValueError, match="precedes"):
        data.validate_data(bars, metadata=metadata)
    metadata["fetched_at"] = "2025-06-19T09:00:00"
    with pytest.raises(ValueError, match="timezone-aware"):
        data.validate_data(bars, metadata=metadata)


def test_adjusted_metadata_is_rejected_even_with_matching_cash_actions():
    bars, metadata = sample()
    metadata["close_basis"] = "yahoo_split_adjusted"
    with pytest.raises(ValueError, match="truly raw"):
        data.validate_data(bars, metadata=metadata)


def test_payment_before_entitlement_is_invalid():
    bars, metadata = sample()
    bars.loc[2, "Dividends"] = 1
    bars.loc[2, "Dividend_Payment_Date"] = "2025-06-17"
    with pytest.raises(ValueError, match="precedes"):
        data.validate_data(bars, metadata=metadata)


def test_input_hash_and_reconstructed_wealth_both_guard_saved_inputs(tmp_path):
    bars, metadata = sample()
    path = tmp_path / "bars.csv"
    bars.to_csv(path, index=False)
    metadata["data_sha256"] = data.digest(path)
    sidecar = path.with_suffix(".meta.json")
    sidecar.write_text(json.dumps(metadata))
    loaded = data.load_dataset(path)
    assert loaded.Total_Return_Index.iloc[-1] == pytest.approx(104)
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        data.load_dataset(path)
    loaded.loc[loaded.index[-1], "Total_Return_Index"] = 100000
    loaded.to_csv(path, index=False)
    metadata["data_sha256"] = data.digest(path)
    sidecar.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="reconstruction"):
        data.load_dataset(path)


def test_existing_download_directory_is_immutable_without_network(tmp_path):
    with pytest.raises(FileExistsError, match="immutable"):
        data.fetch_finmind(tmp_path, end="2026-11-09")


def test_string_false_is_parsed_as_false_not_truthy_text():
    bars, metadata = sample()
    bars["Tradable"] = bars.Tradable.astype(str)
    parsed = data.canonicalize(bars)
    assert not bool(parsed.loc[1, "Tradable"])
    data.validate_data(parsed, metadata=metadata)


def test_expected_exchange_calendar_detects_global_provider_hole():
    bars, metadata = sample()
    expected = pd.DatetimeIndex(["2025-06-10", "2025-06-11", "2025-06-12", "2025-06-18", "2025-06-19"])
    with pytest.raises(ValueError, match="Incomplete market-session"):
        data.validate_data(bars, metadata=metadata, expected_sessions=expected)


def test_finmind_stock_and_cash_statutory_distributions_are_both_included(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    records = {
        "TaiwanStockPrice": [
            {"date": "2024-01-02", "open": 100, "max": 100, "min": 100, "close": 100, "Trading_Volume": 100},
            {"date": "2024-01-03", "open": 98.8 / 1.03, "max": 98.8 / 1.03,
             "min": 98.8 / 1.03, "close": 98.8 / 1.03, "Trading_Volume": 100},
        ],
        "TaiwanStockDividend": [{
            "CashEarningsDistribution": 1.0, "CashStatutorySurplus": 0.2,
            "StockEarningsDistribution": 0.2, "StockStatutorySurplus": 0.1,
            "StockExDividendTradingDate": "2024-01-03", "CashExDividendTradingDate": "2024-01-03",
            "CashDividendPaymentDate": "2024-02-01", "TotalNumberOfCashCapitalIncrease": 0,
        }],
        "TaiwanStockSplitPrice": [], "TaiwanStockCapitalReductionReferencePrice": [],
    }
    manifest = {"provider": "synthetic test fixture", "fetched_at": "2024-01-03T10:00:00Z", "files": []}
    for name, entries in records.items():
        path = raw / (name + ".json")
        path.write_text(json.dumps({"status": 200, "data": entries}))
        manifest["files"].append({"ticker": "2317.TW", "dataset": name,
                                  "path": str(path.relative_to(tmp_path)), "sha256": data.digest(path)})
    (tmp_path / "raw_manifest.json").write_text(json.dumps(manifest))
    path = data.canonicalize_finmind(tmp_path)
    loaded = data.load_dataset(path)
    assert loaded.Stock_Splits.iloc[-1] == pytest.approx(1.03)
    assert loaded.Dividends.iloc[-1] == pytest.approx(1.2 / 1.03)
    np.testing.assert_allclose(loaded.Total_Return_Index, [100, 100])
