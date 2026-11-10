"""Explicit, retrospective V3 raw-price/corporate-action research data.

This module never reads or writes the production price/news stores.  Quotes are
as traded; a missing quote is never silently forward-filled.  A documented
suspension is represented by a non-tradable row, with NaN OHLC and zero volume.
The total-return index is an ex-date reinvestment *research* convention; it is
not a payment-date cash ledger and is not a point-in-time feature certificate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests


TICKERS = ["0050.TW", "2330.TW", "2317.TW", "2603.TW", "2002.TW", "2891.TW"]
QUOTE_COLUMNS = ["Open", "High", "Low", "Close"]
ACTION_COLUMNS = ["Dividends", "Stock_Splits", "Return_of_Capital"]
REQUIRED_COLUMNS = ["Date", "Ticker", *QUOTE_COLUMNS, "Volume", *ACTION_COLUMNS, "Tradable"]
DATASETS = ("TaiwanStockPrice", "TaiwanStockDividend", "TaiwanStockSplitPrice",
            "TaiwanStockCapitalReductionReferencePrice")
FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
SUSPENSIONS = [
    {"ticker": "0050.TW", "start": "2025-06-11", "end": "2025-06-17",
     "reason": "official_split_suspension",
     "source": "https://www.twse.com.tw/staticFiles/news/news/tsecnews/8a8216d696b406fc0196ce27c2e90063.pdf"},
    {"ticker": "2603.TW", "start": "2022-09-07", "end": "2022-09-16",
     "reason": "announced_capital_reduction_suspension",
     "source": "https://www.cna.com.tw/news/afe/202208230303.aspx"},
    {"ticker": "2317.TW", "start": "2025-07-30", "end": "2025-07-30",
     "reason": "official_material_information_suspension",
     "source": "https://investoredu.twse.com.tw/Mobile_Pages/..%2FFileSystem%2FFileUpload%2Fcb744523-2d37-495f-9738-345b882924e7.pdf"},
]
REVIEWED_ACTIONS = [
    {"ticker": "0050.TW", "date": "2025-06-18", "share_factor": 4.0,
     "source": "https://wwwc.twse.com.tw/zh/ETFortune/announcement?company=A00005&date=20250617&fund=0050&seq=1&type=other"},
    {"ticker": "2603.TW", "date": "2022-09-19", "share_factor": 0.4,
     "return_of_capital_per_old_share": 6.0, "payment_date": "2022-09-29",
     "source": "https://www.cna.com.tw/news/afe/202208040331.aspx"},
]
EX_DATE_OVERRIDES = [
    {"ticker": "2891.TW", "original_date": "2026-07-10", "effective_date": "2026-07-13",
     "evidence_file": "2891_postponed_20260713.html", "required_terms": ["2891", "13"],
     "source": "https://m.cnyes.com/news/print/6529878",
     "basis": "Contemporaneous report quoting TWSE closure/postponement; official calculation table retains original date"},
    {"ticker": "2002.TW", "original_date": "2024-07-25", "effective_date": "2024-07-26",
     "evidence_file": "2002_postponed_20240726.html", "required_terms": ["2002", "2024/07/26"],
     "source": "https://superpy.kgieworld.com.tw/kgipythonapi/stockInfo/finance/exDividend?stockCode=2002",
     "basis": "Broker corporate-action history records effective ex-dividend trading on 26 July"},
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _dates(values, name="Date"):
    parsed = pd.to_datetime(values, errors="raise")
    if parsed.isna().any() or getattr(parsed.dt, "tz", None) is not None:
        raise ValueError(f"{name} requires non-missing local date-only values")
    if not parsed.eq(parsed.dt.normalize()).all():
        raise ValueError(f"{name} cannot contain intraday timestamps")
    return parsed


def _bools(values):
    mapping = {True: True, False: False, "True": True, "False": False,
               "true": True, "false": False, 1: True, 0: False}
    if not values.isin(mapping).all():
        raise ValueError("Tradable must contain explicit boolean values")
    return values.map(mapping).astype(bool)


def canonicalize(bars):
    bars = bars.copy()
    missing = set(REQUIRED_COLUMNS).difference(bars.columns)
    if missing:
        raise ValueError(f"Missing V3 data columns: {sorted(missing)}")
    bars["Date"] = _dates(bars.Date)
    if bars.Ticker.isna().any():
        raise ValueError("Missing ticker")
    bars["Ticker"] = bars.Ticker.astype(str)
    bars["Tradable"] = _bools(bars.Tradable)
    for name in QUOTE_COLUMNS + ["Volume"] + ACTION_COLUMNS:
        bars[name] = pd.to_numeric(bars[name], errors="raise")
    return bars.sort_values(["Date", "Ticker"]).reset_index(drop=True)


def validate_data(bars, *, metadata=None, expected_tickers=None, as_of=None,
                  expected_sessions=None):
    """Fail closed on malformed quotes, unknown gaps, or adjusted-price inputs."""
    bars = canonicalize(bars)
    if bars.empty or bars.duplicated(["Date", "Ticker"]).any():
        raise ValueError("Empty or duplicate Date/Ticker data")
    wanted = set(expected_tickers or bars.Ticker.unique())
    if set(bars.Ticker) != wanted:
        raise ValueError("Ticker membership differs from the registered universe")
    if as_of is not None and bars.Date.max() > pd.Timestamp(as_of).normalize():
        raise ValueError("Data extends beyond the requested as-of date")
    if metadata is not None:
        if metadata.get("close_basis") != "truly_raw_unadjusted":
            raise ValueError("V3 action accounting requires truly raw unadjusted OHLC")
        if metadata.get("dividend_basis") != "cash_per_post_event_share":
            raise ValueError("Unknown dividend/share-event basis")
        fetched = pd.Timestamp(metadata.get("fetched_at"))
        if pd.isna(fetched) or fetched.tzinfo is None:
            raise ValueError("Data provenance requires a timezone-aware fetched_at")
        last_close = pd.Timestamp(bars.Date.max()).tz_localize("Asia/Taipei") + pd.Timedelta(hours=13, minutes=30)
        if fetched < last_close:
            raise ValueError("Fetched timestamp precedes the last reported close")
    numeric = bars[["Volume"] + ACTION_COLUMNS].to_numpy(float)
    if not np.isfinite(numeric).all() or (bars.Volume < 0).any():
        raise ValueError("Non-finite actions/volume or negative volume")
    if (bars.Stock_Splits <= 0).any() or (bars.Dividends < 0).any() or (bars.Return_of_Capital < 0).any():
        raise ValueError("Share factors must be positive; cash actions cannot be negative")
    quotes = bars.loc[bars.Tradable, QUOTE_COLUMNS]
    if not np.isfinite(quotes.to_numpy(float)).all() or (quotes <= 0).any().any():
        raise ValueError("Tradable rows require finite positive raw OHLC")
    if (bars.loc[bars.Tradable, "Volume"] <= 0).any():
        raise ValueError("Tradable rows require actual positive trading volume")
    if ((quotes.High < quotes[["Open", "Close", "Low"]].max(axis=1)) |
            (quotes.Low > quotes[["Open", "Close", "High"]].min(axis=1))).any():
        raise ValueError("Inconsistent OHLC high/low")
    suspended = bars.loc[~bars.Tradable]
    if not suspended.empty:
        if not suspended[QUOTE_COLUMNS].isna().all().all() or not suspended.Volume.eq(0).all():
            raise ValueError("Suspensions require missing quotes and zero volume, not invented bars")
        known = (metadata or {}).get("suspensions", [])
        for row in suspended.itertuples():
            if not any(s["ticker"] == row.Ticker and pd.Timestamp(s["start"]) <= row.Date <= pd.Timestamp(s["end"])
                       and bool(s.get("source")) for s in known):
                raise ValueError("Unverified suspension or an unexplained missing quote")
    missing_payments = 0
    for amount, payment in (("Dividends", "Dividend_Payment_Date"),
                            ("Return_of_Capital", "Return_of_Capital_Payment_Date")):
        events = bars.loc[bars[amount].gt(0)]
        if payment not in bars:
            missing_payments += len(events)
            continue
        populated = events[payment].notna() & events[payment].ne("")
        missing_payments += int((~populated).sum())
        dates_paid = _dates(events.loc[populated, payment], payment)
        if (dates_paid < events.loc[populated, "Date"]).any():
            raise ValueError("Payment date precedes the cash entitlement event")
    dates = pd.DatetimeIndex(expected_sessions if expected_sessions is not None else sorted(bars.Date.unique()))
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("Expected market sessions must be unique and increasing")
    for ticker, group in bars.groupby("Ticker"):
        observed = pd.DatetimeIndex(group.Date)
        if not observed.equals(dates):
            raise ValueError(f"Incomplete market-session coverage for {ticker}; no intersection calendar allowed")
    return {"rows": len(bars), "sessions": len(dates), "tickers": sorted(wanted),
            "start": str(dates.min().date()), "end": str(dates.max().date()),
            "explicit_suspension_rows": len(suspended),
            "cash_events_missing_payment_dates": missing_payments,
            "point_in_time_certified": False,
            "calendar_basis": "explicit expected sessions" if expected_sessions is not None else "provider union; not independent exchange certification"}


def total_return_index(bars, *, metadata=None):
    """Return Date×Ticker wealth, with zero mark returns only on known halts.

    Shares multiply by Stock_Splits. Dividends are cash per resulting share;
    Return_of_Capital is cash per previous share. Thus the event gross factor
    is (factor * (raw_close + dividend) + capital_return) / previous_raw_close.
    No future split adjustment is applied to past features.
    """
    bars = canonicalize(bars)
    validate_data(bars, metadata=metadata)
    output = []
    for ticker, part in bars.groupby("Ticker", sort=True):
        previous, wealth = None, 100.0
        for row in part.itertuples():
            factor = float(row.Stock_Splits)
            if previous is None:
                if not row.Tradable:
                    raise ValueError("Cannot initialize wealth from a suspension")
                mark = float(row.Close)
            else:
                # A halted quote is unknown. Detach an event's cash claim
                # from the mechanical stale share mark so cash plus shares
                # cannot create a distribution-day profit out of nothing.
                mark = (float(row.Close) if row.Tradable else
                        (previous - row.Return_of_Capital) / factor - row.Dividends)
                if mark <= 0:
                    raise ValueError("Invalid mechanical suspension ex-rights reference")
                gross = (factor * (mark + row.Dividends) + row.Return_of_Capital) / previous
                if not np.isfinite(gross) or gross <= 0:
                    raise ValueError("Invalid corporate-action return")
                if factor != 1 and (gross < 0.5 or gross > 1.5):
                    raise ValueError("Implausible share-event return: verify raw versus already split-adjusted prices")
                wealth *= gross
            previous = mark
            output.append({"Date": row.Date, "Ticker": ticker, "Total_Return_Index": wealth})
    return pd.DataFrame(output).pivot(index="Date", columns="Ticker", values="Total_Return_Index").sort_index()


def load_dataset(path, *, metadata_path=None, expected_tickers=None, as_of=None,
                 expected_sessions=None):
    path = Path(path)
    meta_path = Path(metadata_path) if metadata_path else path.with_suffix(".meta.json")
    if not meta_path.is_file():
        raise ValueError("V3 data needs a provenance metadata sidecar")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if metadata.get("data_sha256") != digest(path):
        raise ValueError("V3 input hash mismatch")
    bars = canonicalize(pd.read_csv(path))
    validate_data(bars, metadata=metadata, expected_tickers=expected_tickers,
                  as_of=as_of, expected_sessions=expected_sessions)
    tri = total_return_index(bars, metadata=metadata)
    if "Total_Return_Index" in bars:
        retained = bars.pivot(index="Date", columns="Ticker", values="Total_Return_Index")
        if not np.allclose(retained.loc[tri.index, tri.columns], tri, rtol=1e-11, atol=1e-10):
            raise ValueError("Saved total-return index differs from raw-price/action reconstruction")
        bars = bars.drop(columns="Total_Return_Index")
    bars = bars.merge(tri.stack().rename("Total_Return_Index").reset_index(), on=["Date", "Ticker"], validate="one_to_one")
    bars.attrs["metadata"] = metadata
    return bars


def _get_json(params):
    response = requests.get(FINMIND_URL, params=params, timeout=(10, 45), allow_redirects=False)
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") != 200 or not isinstance(payload.get("data"), list):
        raise ValueError("FinMind did not return a complete successful data response")
    return response.content, payload


def fetch_finmind(output_dir, *, start="2021-01-01", end, tickers=None):
    """Save new raw snapshots only; never append to or overwrite existing runs."""
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("V3 input directories are immutable")
    output.mkdir(parents=True)
    raw_dir = output / "raw"
    raw_dir.mkdir()
    files = []
    for ticker in tickers or TICKERS:
        for dataset in DATASETS:
            params = {"dataset": dataset, "data_id": ticker.removesuffix(".TW"),
                      "start_date": start, "end_date": end}
            content, payload = _get_json(params)
            path = raw_dir / f"{params['data_id']}_{dataset}.json"
            path.write_bytes(content)
            files.append({"ticker": ticker, "dataset": dataset,
                          "path": str(path.relative_to(output)), "sha256": digest(path),
                          "rows": len(payload["data"]), "url": FINMIND_URL,
                          "parameters": params, "fetched_at": datetime.now(timezone.utc).isoformat()})
    for override in EX_DATE_OVERRIDES:
        response = requests.get(override["source"], timeout=(10, 45), allow_redirects=False)
        response.raise_for_status()
        if response.is_redirect:
            raise ValueError("Reviewed ex-date source redirected; review new source explicitly")
        (raw_dir / override["evidence_file"]).write_bytes(response.content)
    manifest = {"provider": "FinMind", "fetched_at": datetime.now(timezone.utc).isoformat(), "files": files}
    (output / "raw_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return canonicalize_finmind(output)


def canonicalize_finmind(input_dir):
    """Build bars from retained provider JSON plus explicit reviewed exceptions."""
    root = Path(input_dir)
    manifest = json.loads((root / "raw_manifest.json").read_text(encoding="utf-8"))
    records = {}
    for entry in manifest["files"]:
        path = root / entry["path"]
        if not path.resolve().is_relative_to(root.resolve()) or digest(path) != entry["sha256"]:
            raise ValueError("Raw response path/hash mismatch")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != 200:
            raise ValueError("Unsuccessful raw response")
        records[entry["ticker"], entry["dataset"]] = payload["data"]
    quote_frames = []
    for ticker in sorted({key[0] for key in records}):
        prices = pd.DataFrame(records[ticker, "TaiwanStockPrice"])
        if prices.empty:
            raise ValueError(f"Missing all raw prices: {ticker}")
        quotes = prices.rename(columns={"date": "Date", "open": "Open", "max": "High", "min": "Low",
                                        "close": "Close", "Trading_Volume": "Volume"})
        quotes = quotes[["Date", *QUOTE_COLUMNS, "Volume"]].copy()
        quotes["Ticker"] = ticker
        quote_frames.append(quotes)
    original = pd.concat(quote_frames, ignore_index=True)
    original["Date"] = _dates(original.Date)
    calendar = pd.DatetimeIndex(sorted(original.Date.unique()))
    tickers = sorted(original.Ticker.unique())
    grid = pd.MultiIndex.from_product([calendar, tickers], names=["Date", "Ticker"])
    bars = original.set_index(["Date", "Ticker"]).reindex(grid).reset_index()
    bars["Tradable"] = bars.Close.notna()
    bars["Suspension_Reason"] = ""
    for suspension in SUSPENSIONS:
        mask = bars.Ticker.eq(suspension["ticker"]) & bars.Date.between(suspension["start"], suspension["end"])
        original_quotes = bars.loc[mask, QUOTE_COLUMNS]
        if not (original_quotes.isna() | original_quotes.eq(0)).all().all():
            raise ValueError("A documented suspension unexpectedly contains a quote")
        # FinMind explicitly encodes the 2317 2025-07-30 exchange '--' bar
        # as all zero. Only a reviewed suspension permits this conversion.
        bars.loc[mask, QUOTE_COLUMNS] = np.nan
        bars.loc[mask, "Tradable"] = False
        bars.loc[mask, "Volume"] = 0
        bars.loc[mask, "Suspension_Reason"] = suspension["reason"]
    bars["Dividends"], bars["Stock_Splits"], bars["Return_of_Capital"] = 0.0, 1.0, 0.0
    bars["Dividend_Payment_Date"], bars["Return_of_Capital_Payment_Date"], bars["Action_Announcement_At"] = "", "", ""
    action_overrides = []
    for ticker in tickers:
        for split in records[ticker, "TaiwanStockSplitPrice"]:
            matches = [a for a in REVIEWED_ACTIONS if a["ticker"] == ticker and a["date"] == split["date"]]
            if len(matches) != 1:
                raise ValueError("Split ratio needs independently reviewed exact shares, not rounded price division")
            mask = bars.Ticker.eq(ticker) & bars.Date.eq(pd.Timestamp(split["date"]))
            bars.loc[mask, "Stock_Splits"] = matches[0]["share_factor"]
        for reduction in records[ticker, "TaiwanStockCapitalReductionReferencePrice"]:
            matches = [a for a in REVIEWED_ACTIONS if a["ticker"] == ticker and a["date"] == reduction["date"] and "return_of_capital_per_old_share" in a]
            if len(matches) != 1:
                raise ValueError("Unsupported capital reduction; reviewed cash/share identity required")
            event = matches[0]
            identity = event["share_factor"] * reduction["PostReductionReferencePrice"] + event["return_of_capital_per_old_share"]
            if abs(identity - reduction["ClosingPriceonTheLastTradingDay"]) > 0.011:
                raise ValueError("Capital reduction cash/share identity failed")
            mask = bars.Ticker.eq(ticker) & bars.Date.eq(pd.Timestamp(event["date"]))
            bars.loc[mask, "Stock_Splits"] = event["share_factor"]
            bars.loc[mask, "Return_of_Capital"] = event["return_of_capital_per_old_share"]
            bars.loc[mask, "Return_of_Capital_Payment_Date"] = event["payment_date"]
        for event in records[ticker, "TaiwanStockDividend"]:
            if float(event.get("TotalNumberOfCashCapitalIncrease", 0) or 0) > 0:
                raise ValueError("Rights/cash increase needs a subscription-right ledger")
            cash = float(event.get("CashEarningsDistribution", 0) or 0) + float(event.get("CashStatutorySurplus", 0) or 0)
            stock = float(event.get("StockEarningsDistribution", 0) or 0) + float(event.get("StockStatutorySurplus", 0) or 0)
            for date, amount, kind in ((event.get("StockExDividendTradingDate"), stock, "stock"),
                                       (event.get("CashExDividendTradingDate"), cash, "cash")):
                if not amount:
                    continue
                if not date:
                    raise ValueError("Cash/stock dividend lacks its ex-date")
                # Only individually reviewed cases are moved. A generic
                # next-session mapping would conceal an unexplained action gap.
                matches = [a for a in EX_DATE_OVERRIDES if a["ticker"] == ticker and a["original_date"] == date]
                if matches:
                    override = matches[0]
                    source = root / "raw" / override["evidence_file"]
                    if not source.is_file():
                        raise ValueError("Postponed ex-date requires the retained reviewed evidence")
                    body = source.read_text(encoding="utf-8")
                    if not all(term in body for term in override["required_terms"]):
                        raise ValueError("Postponed ex-date source no longer matches reviewed evidence")
                    action_overrides.append({**{k: v for k, v in override.items() if k != "required_terms"},
                                             "evidence_path": str(source.relative_to(root)), "sha256": digest(source)})
                    date = override["effective_date"]
                mask = bars.Ticker.eq(ticker) & bars.Date.eq(pd.Timestamp(date))
                if mask.sum() != 1 or not bars.loc[mask, "Tradable"].all():
                    raise ValueError(f"Dividend ex-date has no actual quote: {ticker} {date}")
                if kind == "stock":
                    bars.loc[mask, "Stock_Splits"] *= 1 + amount / 10
                else:
                    bars.loc[mask, "Dividends"] += amount
                    bars.loc[mask, "Dividend_Payment_Date"] = event.get("CashDividendPaymentDate", "")
                ann_date = event.get("AnnouncementDate", "")
                ann_time = event.get("AnnouncementTime", "")
                if ann_date and ann_time:
                    bars.loc[mask, "Action_Announcement_At"] = f"{ann_date}T{ann_time}+08:00"
    # Cash dividend source amounts are per original share. Convert once after
    # all same-day share events so the canonical column is per resulting share.
    bars["Dividends"] /= bars.Stock_Splits
    metadata = {"schema": "v3_raw_bars_v1", "provider": manifest["provider"],
                "fetched_at": manifest["fetched_at"], "close_basis": "truly_raw_unadjusted",
                "dividend_basis": "cash_per_post_event_share", "capital_return_basis": "cash_per_pre_event_share",
                "stock_splits_basis": "new_shares_per_old_share; 1 means no event",
                "corporate_actions_basis": "retrospective provider plus explicitly reviewed exceptions",
                "point_in_time_certified": False, "official_exhaustive_action_coverage_certified": False,
                "suspensions": SUSPENSIONS, "reviewed_actions": REVIEWED_ACTIONS,
                "action_date_overrides": action_overrides, "raw_manifest_sha256": digest(root / "raw_manifest.json"),
                "total_return_convention": "ex-date reinvestment research proxy; payment-date cash not equivalent",
                "limitations": ["Retrospective source fetch; historical first availability not certified",
                                "Provider union calendar is not an independent official session audit",
                                "No complete official corporate-action reconciliation",
                                "No bid/ask spread or executable closing-auction guarantee"]}
    bars = canonicalize(bars)
    metadata["quality"] = validate_data(bars, metadata=metadata, expected_tickers=tickers)
    tri = total_return_index(bars, metadata=metadata)
    bars = bars.merge(tri.stack().rename("Total_Return_Index").reset_index(), on=["Date", "Ticker"], validate="one_to_one")
    path = root / "bars.csv"
    if path.exists():
        raise FileExistsError("Canonical V3 inputs are immutable")
    bars.to_csv(path, index=False, date_format="%Y-%m-%d")
    metadata["data_sha256"] = digest(path)
    (root / "bars.meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check", type=Path)
    modes.add_argument("--fetch", type=Path)
    modes.add_argument("--canonicalize", type=Path)
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--as-of", required=True)
    args = parser.parse_args()
    if args.fetch:
        path = fetch_finmind(args.fetch, start=args.start, end=args.as_of)
    elif args.canonicalize:
        path = canonicalize_finmind(args.canonicalize)
    else:
        path = args.check
    bars = load_dataset(path, as_of=args.as_of)
    print(json.dumps(validate_data(bars, metadata=bars.attrs["metadata"], as_of=args.as_of), indent=2))


if __name__ == "__main__":
    main()
