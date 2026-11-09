"""Repair the 2025-08-01 sentiment hole and replay calendar-based weight rules.

Stored walk-forward predictions are read-only inputs. This module does not
train a model and does not write the official notebooks, reports, or the
historical sentiment file. Every comparison uses a path that was already
used to choose scheme G, so the results stay exploratory.
"""
from __future__ import annotations

import html
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from experiments.risk_experiment import (
    BROKER_FEE_RATE,
    INITIAL_CAPITAL,
    MARKET,
    MAX_WEIGHT_LIMIT,
    RESULT_DIR,
    RUN,
    SELECTION_END,
    SELL_TAX_RATE,
    SENTIMENT_BEAR,
    SENTIMENT_BULL,
    TICKERS,
    TIME_STEP,
    WEEKLY_GAP,
    FEATURES,
    allocate,
    build_frames,
    load_panel,
    load_sentiment,
    max_drawdown,
    rebalance,
    sharpe_ratio,
)
from scripts.common import TZ
from scripts.news_update import MODEL_ID, matches, to_signal_date

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_SENTIMENT = ROOT / "data" / "sentiment_features_2026.csv"
DATA_DIR = Path(__file__).resolve().parent / "data"
OUT = RESULT_DIR / "calendar_rules"
API = "https://api.cnyes.com/media/api/v1/newslist/category/tw_stock"
GAP_DAY = pd.Timestamp("2025-08-01")
HOLE_FETCH_DAYS = (pd.Timestamp("2025-07-31"), pd.Timestamp("2025-08-01"))
COVERAGE_CONTROLS = (pd.Timestamp("2021-01-04"), pd.Timestamp("2024-01-02"))
UNVERIFIED = "unverified_historical_coverage"
VERIFIED_NONE = "verified_no_matching_article"
OBSERVED = "observed_article"

# Fixed before any strategy ROI on this path is consulted. Tau is a price-only
# scale from the 60 sessions before each fold, because validation predictions
# were not saved and cannot be used to calibrate a model-confidence cutoff.
BASE_WEIGHT = 0.20
CONFIDENT_WEIGHT = 0.30
UNCONFIDENT_WEIGHT = 0.10
RANK_TILT = 0.05
RANK_FLOOR = 0.05
RISK_EXPOSURE = 0.50
VALIDATION_SESSIONS = 60
RULES = ("A", "B", "C", "D")
NAMES = {
    "2330.TW": "台積電",
    "2317.TW": "鴻海",
    "2603.TW": "長榮",
    "2002.TW": "中鋼",
    "2891.TW": "中信金",
}
# exchange_calendars 把這兩天算成交易日，證交所公告當天沒有交易。
# 再平衡只數有成交價格的交易日，不把這兩天補進排程。
KNOWN_NON_TRADING = {
    "2023-01-18": "農曆年前僅辦理結算交割，集中市場無交易",
    "2024-10-31": "康芮颱風，證交所公告全日休市",
}
NAME_CUES = ("台積電", "2330", "鴻海", "2317", "長榮", "2603", "中鋼", "2002", "中信金", "2891")


def assert_experimental_path(path):
    resolved = Path(path).resolve()
    if resolved == OFFICIAL_SENTIMENT.resolve() or "experiments" not in resolved.parts:
        raise RuntimeError(f"拒絕寫入正式資料：{resolved}")
    return resolved


def fetch_category_day(day, session=None):
    """Read one calendar day until the category total is exhausted.

    The official fetcher stops when a page has fewer than 50 rows, but this
    endpoint returns 30 rows per page. That early stop is not used here.
    """
    day = pd.Timestamp(day).date()
    start = datetime(day.year, day.month, day.day, tzinfo=TZ)
    end = start + timedelta(days=1) - timedelta(seconds=1)
    http = session or requests.Session()
    found = []
    page_rows = []
    page = 1
    total = None
    last_page = 1
    while page <= 40:
        response = http.get(
            API,
            params={
                "limit": 50,
                "page": page,
                "startAt": int(start.timestamp()),
                "endAt": int(end.timestamp()),
            },
            headers={"User-Agent": "Mozilla/5.0 (compatible; university research project)"},
            timeout=40,
        )
        response.raise_for_status()
        obj = response.json().get("items", {})
        data = obj.get("data") if isinstance(obj, dict) else obj
        if not isinstance(data, list):
            raise ValueError(f"鉅亨網回覆格式異常 {day} page={page}")
        total = int(obj.get("total", len(data)))
        last_page = int(obj.get("last_page", page))
        kept_ids = []
        skipped_other_date = 0
        for item in data:
            stamp = item.get("publishAt")
            article_id = item.get("newsId", item.get("id"))
            if stamp is None or article_id is None:
                raise ValueError("新聞缺少發布時間或文章 ID")
            published = datetime.fromtimestamp(int(stamp), TZ)
            if published.date() != day:
                skipped_other_date += 1
                continue
            article = {
                "news_id": str(article_id),
                "publish_time": published.isoformat(),
                "title": str(item.get("title") or ""),
                "summary": html.unescape(str(item.get("summary") or "")),
                "calendar_day": str(day),
                "page": page,
            }
            found.append(article)
            kept_ids.append(article["news_id"])
        page_rows.append({
            "page": page,
            "raw": len(data),
            "kept": len(kept_ids),
            "skipped_other_date": skipped_other_date,
            "from": obj.get("from"),
            "to": obj.get("to"),
            "ids": kept_ids,
        })
        if page >= last_page or not data:
            break
        page += 1
        time.sleep(0.15)
    else:
        raise RuntimeError(f"{day} 超過 40 頁，停止以免把不完整清單當成完整覆蓋")
    articles = list({row["news_id"]: row for row in found}.values())
    audit = page_audit(page_rows, total, last_page, len(articles))
    return {
        "calendar_day": str(day),
        "api_total": total,
        "fetched": len(articles),
        "last_page": last_page,
        "complete": audit["complete"],
        "page_audit": {key: value for key, value in audit.items() if key != "ids"},
        "articles": articles,
    }


def page_audit(page_rows, api_total, last_page, unique_count):
    """A feed is complete only when every page is present once and nothing is dropped."""
    ids = [article_id for row in page_rows for article_id in row["ids"]]
    duplicates = sorted({article_id for article_id in ids if ids.count(article_id) > 1})
    overlaps = []
    for left, right in zip(page_rows, page_rows[1:]):
        shared = sorted(set(left["ids"]) & set(right["ids"]))
        if shared:
            overlaps.append({"pages": [left["page"], right["page"]], "shared_ids": shared})
    skipped = int(sum(row["skipped_other_date"] for row in page_rows))
    raw = int(sum(row["raw"] for row in page_rows))
    complete = (
        api_total is not None
        and unique_count == api_total
        and len(ids) == api_total
        and raw == api_total
        and skipped == 0
        and not duplicates
        and not overlaps
        and len(page_rows) == last_page
    )
    return {
        "api_total": api_total,
        "last_page": last_page,
        "pages_fetched": len(page_rows),
        "raw_rows": raw,
        "unique_ids": unique_count,
        "duplicate_ids": duplicates,
        "page_overlaps": overlaps,
        "skipped_other_date": skipped,
        "complete": complete,
        "pages": [
            {key: value for key, value in row.items() if key != "ids"}
            for row in page_rows
        ],
    }


def review_signal_articles(articles):
    """Compare the project matcher with company-name cues. A cue without a hit is a possible miss."""
    reviewed = []
    cue_without_hit = []
    for article in articles:
        text = f"{article['title']} {article['summary']}"
        hits = [ticker for ticker in TICKERS if matches(text, ticker)]
        cues = [cue for cue in NAME_CUES if cue in text]
        title_only = [
            ticker for ticker in hits
            if NAMES[ticker] not in article["title"] and ticker.split(".")[0] not in article["title"]
        ]
        row = {
            "news_id": article["news_id"],
            "publish_time": article["publish_time"],
            "title": article["title"],
            "matched": hits,
            "cues": cues,
            "name_only_in_summary": title_only,
        }
        if hits or cues:
            reviewed.append(row)
        if cues and not hits:
            cue_without_hit.append(row)
    return {"mentioned": reviewed, "cue_without_match": cue_without_hit}


def articles_for_signal(bundles, signal_day):
    signal_day = pd.Timestamp(signal_day).normalize()
    if any(not bundle["complete"] for bundle in bundles):
        return {"complete": False, "articles": []}
    chosen = []
    for bundle in bundles:
        for article in bundle["articles"]:
            if pd.Timestamp(to_signal_date(article["publish_time"])).normalize() == signal_day:
                chosen.append(article)
    return {"complete": True, "articles": chosen}


def score_gap(signal_day, bundles, scorer_factory):
    """Score matched articles, or refuse to invent a zero when coverage is incomplete."""
    selected = articles_for_signal(bundles, signal_day)
    rows = []
    scored = []
    if not selected["complete"] or scorer_factory is None:
        for ticker in TICKERS:
            rows.append(_gap_row(signal_day, ticker, np.nan, 0, None, UNVERIFIED, "not_scored"))
        return rows, scored, "unverified"
    pending = []
    for article in selected["articles"]:
        text = f"{article['title']} {article['summary']}"
        hit = [ticker for ticker in TICKERS if matches(text, ticker)]
        if hit:
            pending.append((article, text, hit))
    by_ticker = {ticker: [] for ticker in TICKERS}
    cache = {}
    if pending:
        scorer = scorer_factory()
        for article, text, hit in pending:
            if article["news_id"] not in cache:
                cache[article["news_id"]] = float(scorer.score(text[:1600]))
            for ticker in hit:
                by_ticker[ticker].append(cache[article["news_id"]])
                scored.append({
                    **article,
                    "ticker": ticker,
                    "signal_date": str(pd.Timestamp(signal_day).date()),
                    "sentiment": cache[article["news_id"]],
                    "model_id": MODEL_ID,
                })
    for ticker in TICKERS:
        values = by_ticker[ticker]
        if values:
            latest = max(
                item["publish_time"] for item in scored if item["ticker"] == ticker
            )
            rows.append(_gap_row(
                signal_day, ticker, float(np.mean(values)), len(values), latest, OBSERVED, MODEL_ID,
            ))
        else:
            rows.append(_gap_row(
                signal_day, ticker, 0.0, 0, None, VERIFIED_NONE, "neutral_no_news",
            ))
    return rows, scored, "scored"


def _gap_row(day, ticker, sentiment, count, latest, status, model):
    return {
        "Date": pd.Timestamp(day).normalize(),
        "ticker": ticker,
        "Daily_Sentiment": sentiment,
        "News_Count": int(count),
        "Latest_News_Timestamp": latest,
        "Source_Status": status,
        "Scoring_Model": model,
    }


def write_experimental_sentiment(base, gap_rows, destination):
    destination = assert_experimental_path(destination)
    original = pd.read_csv(base)
    original["Date"] = pd.to_datetime(original["Date"]).dt.normalize()
    addition = pd.DataFrame(gap_rows)
    addition["Date"] = pd.to_datetime(addition["Date"]).dt.normalize()
    unverified = addition.Source_Status.eq(UNVERIFIED)
    if unverified.any() and addition.loc[unverified, "Daily_Sentiment"].notna().any():
        raise ValueError("無法驗證的股票不得填入情緒分數")
    if ((addition.Source_Status == VERIFIED_NONE) & (addition.Daily_Sentiment != 0)).any():
        raise ValueError("確認無對應新聞時分數必須是 0")
    combined = pd.concat([original, addition], ignore_index=True)
    combined = combined.sort_values(["Date", "ticker"]).reset_index(drop=True)
    if combined.duplicated(["Date", "ticker"]).any():
        raise ValueError("實驗情緒副本出現重複的日期與股票")
    destination.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(destination, index=False)
    return combined


def integrity_issues(panel, sentiment, start=None, end=None):
    """Every complete price session needs five rows, or an explicit missing status."""
    frame = sentiment.copy()
    frame["Date"] = pd.to_datetime(frame["Date"]).dt.normalize()
    start = pd.Timestamp(start or frame.Date.min()).normalize()
    end = pd.Timestamp(end or frame.Date.max()).normalize()
    sessions = panel.dropna(subset=TICKERS).index
    sessions = sessions[(sessions >= start) & (sessions <= end)]
    issues = []
    for day in sessions:
        for ticker in TICKERS:
            part = frame[(frame.Date == day) & (frame.ticker == ticker)]
            if len(part) != 1:
                issues.append({
                    "Date": str(pd.Timestamp(day).date()),
                    "ticker": ticker,
                    "problem": "missing_row" if len(part) == 0 else "duplicate_row",
                })
                continue
            value = part.Daily_Sentiment.iloc[0]
            status = part.Source_Status.iloc[0]
            status = "" if pd.isna(status) else str(status)
            if pd.isna(value) and status != UNVERIFIED:
                issues.append({
                    "Date": str(pd.Timestamp(day).date()),
                    "ticker": ticker,
                    "problem": "null_without_explicit_status",
                })
            elif status == VERIFIED_NONE and value != 0:
                issues.append({
                    "Date": str(pd.Timestamp(day).date()),
                    "ticker": ticker,
                    "problem": "verified_none_is_not_zero",
                })
            elif status == OBSERVED and pd.isna(value):
                issues.append({
                    "Date": str(pd.Timestamp(day).date()),
                    "ticker": ticker,
                    "problem": "observed_without_score",
                })
    return pd.DataFrame(issues, columns=["Date", "ticker", "problem"])


def validation_tau(panel, oos_start, oos_end):
    """Median absolute next-day move on the 60 sessions before a fold."""
    oos_start = pd.Timestamp(oos_start).normalize()
    oos_end = pd.Timestamp(oos_end).normalize()
    prior = panel.index[panel.index < oos_start]
    if len(prior) < VALIDATION_SESSIONS:
        raise ValueError(f"{oos_start.date()} 之前不足 {VALIDATION_SESSIONS} 個交易日")
    window = prior[-VALIDATION_SESSIONS:]
    if window.max() >= oos_start:
        raise AssertionError("驗證門檻使用了折外日期")
    rows = []
    for ticker in TICKERS:
        returns = panel[ticker].pct_change(fill_method=None).loc[window] * 100
        if not np.isfinite(returns).all():
            raise ValueError(f"{ticker} 的驗證報酬含有缺值")
        rows.append({
            "oos_start": oos_start,
            "oos_end": oos_end,
            "Ticker": ticker,
            "Tau_Pct": float(np.median(np.abs(returns))),
            "Validation_Start": pd.Timestamp(window.min()),
            "Validation_End": pd.Timestamp(window.max()),
        })
    return pd.DataFrame(rows)


def build_tau_table(panel, folds):
    tables = [
        validation_tau(panel, row.oos_start, row.oos_end)
        for row in folds.itertuples(index=False)
    ]
    return pd.concat(tables, ignore_index=True)


def tau_for(table, target, *, as_of=None):
    target = pd.Timestamp(target).normalize()
    hit = table[(table.oos_start <= target) & (table.oos_end >= target)]
    if set(hit.Ticker) != set(TICKERS) or hit.Ticker.duplicated().any():
        raise ValueError(f"{target.date()} 找不到唯一的驗證門檻")
    if as_of is not None:
        if "Validation_End" not in hit or pd.to_datetime(hit.Validation_End).isna().any():
            raise ValueError("驗證門檻缺少可得時間")
        if (pd.to_datetime(hit.Validation_End) > pd.Timestamp(as_of)).any():
            raise ValueError("驗證門檻使用了訊號日之後的價格")
    return hit.set_index("Ticker").loc[TICKERS, "Tau_Pct"].to_numpy(dtype=float)


def raw_weights(rule, predictions, sentiments, tau):
    pred = np.asarray(predictions, dtype=float)
    sent = np.asarray(sentiments, dtype=float)
    cutoff = np.asarray(tau, dtype=float)
    if rule == "A":
        raw = np.full(len(pred), BASE_WEIGHT)
        raw[(pred > 0) & (sent > SENTIMENT_BULL)] = MAX_WEIGHT_LIMIT
        raw[(pred < 0) | (sent < SENTIMENT_BEAR)] = 0.0
        return raw
    if rule == "B":
        return np.full(len(pred), BASE_WEIGHT)
    if rule == "C":
        raw = np.full(len(pred), BASE_WEIGHT)
        raw[pred > cutoff] = CONFIDENT_WEIGHT
        raw[pred < -cutoff] = UNCONFIDENT_WEIGHT
        return raw
    if rule == "D":
        raw = np.full(len(pred), BASE_WEIGHT)
        highest = float(pred.max())
        lowest = float(pred.min())
        if highest > lowest:
            raw[np.isclose(pred, highest)] += RANK_TILT
            raw[np.isclose(pred, lowest)] -= RANK_TILT
        return np.maximum(raw, RANK_FLOOR)
    raise ValueError(rule)


def cap_weights(raw, risk_off, exposure=RISK_EXPOSURE):
    weights = np.asarray(raw, dtype=float).copy()
    cap = exposure if risk_off else 1.0
    if weights.sum() > cap:
        weights *= cap / weights.sum()
    if (weights < -1e-12).any() or weights.sum() > cap + 1e-8 or weights.max() > MAX_WEIGHT_LIMIT + 1e-8:
        raise ValueError("權重超出預先規定的上限")
    return weights


def apply_rule(rule, predictions, sentiments, risk_off, tau, equalize_gross=False):
    raw = raw_weights(rule, predictions, sentiments, tau)
    if equalize_gross:
        raw = np.full(len(raw), float(raw.sum()) / len(raw))
    weights = cap_weights(raw, risk_off)
    if rule == "A" and not equalize_gross:
        official = allocate(predictions, sentiments, risk_off, RISK_EXPOSURE)
        if not np.allclose(weights, official, atol=1e-10):
            raise AssertionError("規則 A 與正式配置公式不一致")
    return weights


def _prediction_map(predictions):
    mapped = {}
    order = []
    for _, day in predictions.groupby("Date", sort=True):
        day = day.set_index("Ticker").loc[TICKERS]
        signal = pd.Timestamp(day.Signal_Date.iloc[0]).normalize()
        if day.Signal_Date.nunique() != 1 or signal in mapped:
            raise ValueError(f"{signal.date()} 的預測不完整或重複")
        mapped[signal] = day
        order.append(signal)
    return mapped, order


def _empty_ledger():
    return {
        "Buy_Notional": 0.0,
        "Sell_Notional": 0.0,
        "Buy_Fee": 0.0,
        "Sell_Fee": 0.0,
        "Sell_Tax": 0.0,
        "Total_Cost": 0.0,
        "Turnover": 0.0,
    }


def run_book(
    predictions,
    panel,
    risk,
    rule,
    tau_table,
    schedule="trading_calendar",
    gap=WEEKLY_GAP,
    liquidate=True,
    equalize_gross=False,
):
    """Fill at the next session close using only the preceding signal's inputs.

    The schedule is anchored on signal sessions; missing predictions cannot move
    it. This corrected execution protocol must be rerun before citing any ROI.
    """
    if schedule not in {"trading_calendar", "prediction_rows"}:
        raise ValueError(schedule)
    mapped, order = _prediction_map(predictions)
    if not order:
        raise ValueError("帳本沒有訊號")
    first = order[0]
    last = pd.Timestamp(predictions.Date.max()).normalize()
    dates = panel.index[(panel.index > first) & (panel.index <= last)]
    if not len(dates):
        raise ValueError("訊號之後沒有下一交易日成交價格")
    if schedule == "prediction_rows":
        due = set(order[::gap])
    else:
        signal_dates = panel.index[(panel.index >= first) & (panel.index < last)]
        due = {pd.Timestamp(signal_dates[i]) for i in range(0, len(signal_dates), gap)}
    shares = np.zeros(len(TICKERS))
    cash = INITIAL_CAPITAL
    previous = None
    rows = []
    for index, day in enumerate(dates):
        day = pd.Timestamp(day)
        price = panel.loc[day, TICKERS].to_numpy(dtype=float)
        held = shares.copy()
        change = np.zeros(len(TICKERS)) if previous is None else held * (price - previous)
        old_values = held * price
        pre_nav = float(old_values.sum() + cash)
        signal = pd.Timestamp(panel.index[panel.index.get_loc(day) - 1])
        flag = bool(risk.loc[signal, "Risk_Off"])
        traded = False
        skipped = False
        ledger = _empty_ledger()
        pred = np.full(len(TICKERS), np.nan)
        sent = np.full(len(TICKERS), np.nan)
        cutoff = np.full(len(TICKERS), np.nan)
        if signal in due and signal in mapped:
            info = mapped[signal]
            pred = info.Predicted_Return_Pct.to_numpy(dtype=float)
            sent = info.Daily_Sentiment.to_numpy(dtype=float)
            target = pd.Timestamp(info.Date.iloc[0]).normalize()
            cutoff = tau_for(tau_table, target, as_of=signal)
            weights = apply_rule(rule, pred, sent, flag, cutoff, equalize_gross=equalize_gross)
            target_values, cash, ledger = rebalance(old_values, cash, weights)
            shares = target_values / price
            traded = True
        elif signal in due:
            skipped = True
            weights = old_values / pre_nav if pre_nav else np.zeros(len(TICKERS))
        else:
            weights = old_values / pre_nav if pre_nav else np.zeros(len(TICKERS))
        post_values = shares * price
        nav = float(post_values.sum() + cash)
        liquidation_cost = 0.0
        if liquidate and index == len(dates) - 1:
            liquidation_cost = float(post_values.sum() * (BROKER_FEE_RATE + SELL_TAX_RATE))
            nav -= liquidation_cost
        row = {
            "Date": day,
            "Signal_Date": signal,
            "Trade": traded,
            "Scheduled": signal in due,
            "Skipped_For_Missing_Prediction": skipped,
            "Risk_Off": flag,
            "NAV_Pre": pre_nav,
            "NAV_Reported": nav,
            "Rebalance_Cost": ledger["Total_Cost"],
            "Liquidation_Cost": liquidation_cost,
            "Equity_Weight": float(post_values.sum() / (post_values.sum() + cash)) if post_values.sum() + cash else 0.0,
            "Cash_Weight": float(cash / (post_values.sum() + cash)) if post_values.sum() + cash else 1.0,
            "Stock_PnL": float(change.sum()),
            "Execution": "next_session_close",
        }
        for ticker, pnl, weight, prediction in zip(TICKERS, change, weights, pred):
            row[f"PnL_{ticker}"] = float(pnl)
            row[f"Target_{ticker}"] = float(weight)
            row[f"Pred_{ticker}"] = float(prediction) if np.isfinite(prediction) else np.nan
            row[f"Share_{ticker}"] = float(shares[TICKERS.index(ticker)])
            row[f"Trade_Px_{ticker}"] = float(price[TICKERS.index(ticker)])
        rows.append(row)
        previous = price
    daily = pd.DataFrame(rows)
    identity = (
        INITIAL_CAPITAL
        + float(daily.Stock_PnL.sum())
        - float(daily.Rebalance_Cost.sum())
        - float(daily.Liquidation_Cost.sum())
        - float(daily.NAV_Reported.iloc[-1])
    )
    if abs(identity) > 1e-4:
        raise AssertionError(f"淨值恆等式不成立，差額 {identity}")
    return daily


def book_metrics(daily, rule, schedule, window):
    values = np.concatenate([[INITIAL_CAPITAL], daily.NAV_Reported.to_numpy(dtype=float)])
    traded = daily[daily.Trade]
    exclusions = 0
    if len(traded):
        exclusions = int(sum((traded[f"Target_{ticker}"] <= 1e-12).sum() for ticker in TICKERS))
    return {
        "Rule": rule,
        "Schedule": schedule,
        "Window": window,
        "Study": "exploratory_not_a_new_test",
        "ROI_Pct": float((values[-1] / INITIAL_CAPITAL - 1) * 100),
        "Sharpe": sharpe_ratio(values),
        "MDD_Pct": max_drawdown(values),
        "Total_Transaction_Cost": float(daily.Rebalance_Cost.sum() + daily.Liquidation_Cost.sum()),
        "Average_Equity_Weight_Pct": float(daily.Equity_Weight.mean() * 100),
        "Average_Cash_Weight_Pct": float(daily.Cash_Weight.mean() * 100),
        "Exclusion_Count": exclusions,
        "Rebalance_Days": int(daily.Trade.sum()),
        "Skipped_Rebalance_Days": int(daily.Skipped_For_Missing_Prediction.sum()),
        "Sessions": int(len(daily)),
        "Same_Close_Execution": False,
        "Execution": "next_session_close",
    }


def stock_contribution(daily, rule):
    rows = []
    traded = daily[daily.Trade]
    for ticker in TICKERS:
        rows.append({
            "Rule": rule,
            "Ticker": ticker,
            "Name": NAMES[ticker],
            "PnL": float(daily[f"PnL_{ticker}"].sum()),
            "PnL_Pct_Of_Initial": float(daily[f"PnL_{ticker}"].sum() / INITIAL_CAPITAL * 100),
            "Exclusion_Count": int((traded[f"Target_{ticker}"] <= 1e-12).sum()) if len(traded) else 0,
        })
    return pd.DataFrame(rows)


def fold_results(daily, folds):
    assigned = []
    for day in daily.Date:
        day = pd.Timestamp(day)
        if day < folds.oos_start.iloc[0]:
            assigned.append(0)
            continue
        hit = np.flatnonzero((folds.oos_start <= day) & (folds.oos_end >= day))
        if len(hit) != 1:
            raise ValueError(f"{day.date()} 沒有落在唯一的 walk-forward 折")
        assigned.append(int(hit[0]))
    frame = daily.copy()
    frame["Fold"] = assigned
    levels = [INITIAL_CAPITAL]
    rows = []
    for fold, part in frame.groupby("Fold", sort=True):
        start = levels[-1]
        end = float(part.NAV_Reported.iloc[-1])
        levels.append(end)
        rows.append({
            "Fold": int(fold) + 1,
            "Start": str(pd.Timestamp(part.Date.iloc[0]).date()),
            "End": str(pd.Timestamp(part.Date.iloc[-1]).date()),
            "ROI_Pct": float((end / start - 1) * 100),
        })
    return pd.DataFrame(rows)


def index_return(panel, start, end):
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    return float(panel.loc[end, MARKET] / panel.loc[start, MARKET] - 1)


def load_study():
    predictions = pd.read_csv(RESULT_DIR / "walk_forward_predictions.csv", parse_dates=["Date", "Signal_Date"])
    predictions["Date"] = pd.to_datetime(predictions.Date).dt.normalize()
    predictions["Signal_Date"] = pd.to_datetime(predictions.Signal_Date).dt.normalize()
    predictions = predictions[predictions.Date <= SELECTION_END].copy()
    panel = load_panel(RUN / "historical_prices.csv")
    _, risk = build_frames(panel, load_sentiment(RUN / "sentiment_snapshot.csv", panel.index))
    folds = pd.read_csv(RESULT_DIR / "walk_forward_folds.csv", parse_dates=["oos_start", "oos_end"])
    folds["oos_start"] = pd.to_datetime(folds.oos_start).dt.normalize()
    folds["oos_end"] = pd.to_datetime(folds.oos_end).dt.normalize()
    folds = folds.drop_duplicates(["oos_start", "oos_end"]).sort_values("oos_start").reset_index(drop=True)
    return predictions, panel, risk, folds


def trading_calendar_check(panel, start, end):
    """Count rebalance days on price sessions. Library-only dates must be known closures."""
    from scripts.common import sessions

    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    library = set(pd.DatetimeIndex(sessions(start, end)).normalize())
    prices = set(panel.index[(panel.index >= start) & (panel.index <= end)])
    extra = sorted(library - prices)
    missing = sorted(prices - library)
    known = {pd.Timestamp(day) for day in KNOWN_NON_TRADING}
    unexplained = [day for day in extra if day not in known]
    return {
        "price_sessions": len(prices),
        "library_sessions": len(library),
        "prices_absent_from_library": [str(day.date()) for day in missing],
        "library_dates_without_prices": [str(day.date()) for day in extra],
        "unexplained_library_dates": [str(day.date()) for day in unexplained],
        "known_closures_not_counted": KNOWN_NON_TRADING,
    }


def feature_targets(frames):
    common = None
    for ticker in TICKERS:
        frame = frames[ticker]
        values = frame[FEATURES].to_numpy(dtype=float)
        found = []
        for end in range(TIME_STEP - 1, len(frame) - 1):
            start = end - TIME_STEP + 1
            if np.isfinite(values[start:end + 1]).all() and np.isfinite(frame.Target_Next_Return.iloc[end]):
                found.append(pd.Timestamp(frame.Target_Date.iloc[end]).normalize())
        common = set(found) if common is None else common & set(found)
    return common


def prediction_gap_status(panel, predictions):
    from experiments.signal_diagnosis import windows_blocked_by

    official = load_sentiment(OFFICIAL_SENTIMENT, panel.index)
    experimental_path = DATA_DIR / "sentiment_experimental.csv"
    experimental = load_sentiment(experimental_path, panel.index) if experimental_path.exists() else official
    official_frames, _ = build_frames(panel, official)
    experimental_frames, _ = build_frames(panel, experimental)
    index = official_frames[TICKERS[0]].index
    blocked = windows_blocked_by(index, GAP_DAY)
    official_targets = feature_targets(official_frames)
    experimental_targets = feature_targets(experimental_frames)
    stored = set(pd.to_datetime(predictions.Date).dt.normalize())
    return {
        "blocked_targets": [str(day.date()) for day in blocked],
        "official_feature_windows_open": [str(day.date()) for day in blocked if day in official_targets],
        "experimental_feature_windows_open": [str(day.date()) for day in blocked if day in experimental_targets],
        "stored_predictions_still_missing": [str(day.date()) for day in blocked if day not in stored],
        "prediction_gap_closed": all(day in stored for day in blocked),
        "reason": "實驗情緒補上後，這十個目標日的特徵窗可以成形；已儲存的預測沒有這十天，而且沒有留下模型權重，不重新訓練就補不出預測。",
    }


def blocked_targets(predictions, panel):
    """Sessions whose return was earned without a stored prediction."""
    first = pd.Timestamp(predictions.Signal_Date.min())
    last = pd.Timestamp(predictions.Date.max())
    dates = list(panel.index[(panel.index >= first) & (panel.index <= last)])
    signals = set(pd.to_datetime(predictions.Signal_Date).dt.normalize())
    missing = [day for day in dates[:-1] if day not in signals]
    blocked = [dates[dates.index(day) + 1] for day in missing]
    return blocked, missing


def repair_sentiment_gap(fetch=fetch_category_day, scorer_factory=None):
    days = list(dict.fromkeys([*COVERAGE_CONTROLS, *HOLE_FETCH_DAYS]))
    bundles = {}
    for day in days:
        bundles[pd.Timestamp(day).normalize()] = fetch(day)
    coverage = [
        {key: value for key, value in bundle.items() if key != "articles"}
        for bundle in bundles.values()
    ]
    hole_bundles = [bundles[pd.Timestamp(day).normalize()] for day in HOLE_FETCH_DAYS]
    source_complete = all(row["complete"] for row in coverage)
    if scorer_factory is None:
        from scripts.news_update import SentimentScorer

        scorer_factory = SentimentScorer
    rows, scored, state = score_gap(
        GAP_DAY,
        hole_bundles,
        scorer_factory if source_complete else None,
    )
    selected = articles_for_signal(hole_bundles, GAP_DAY)
    review = review_signal_articles(selected["articles"]) if selected["complete"] else {
        "mentioned": [],
        "cue_without_match": [],
    }
    if review["cue_without_match"]:
        for row in rows:
            if row["Source_Status"] == VERIFIED_NONE:
                row["Daily_Sentiment"] = np.nan
                row["News_Count"] = 0
                row["Source_Status"] = UNVERIFIED
                row["Scoring_Model"] = "not_scored"
        state = "unverified_zeros"
    return {
        "coverage": coverage,
        "page_audits": [bundle.get("page_audit", {}) for bundle in bundles.values()],
        "source_complete_for_category": source_complete and state != "unverified",
        "match_review": review,
        "state": state,
        "rows": rows,
        "articles": scored,
        "coverage_limit": "僅能確認鉅亨網 tw_stock 分類的 API total 已被完整取回，不是全市場新聞普查",
    }


def build_report(repair=None):
    predictions, panel, risk, folds = load_study()
    tau = build_tau_table(panel, folds)
    if (tau.Validation_End >= tau.oos_start).any():
        raise AssertionError("驗證門檻含有折外日期")
    if repair is None:
        repair = repair_sentiment_gap()
    destination = DATA_DIR / "sentiment_experimental.csv"
    experimental = write_experimental_sentiment(OFFICIAL_SENTIMENT, repair["rows"], destination)
    original = pd.read_csv(OFFICIAL_SENTIMENT)
    original_issues = integrity_issues(panel, original)
    experimental_issues = integrity_issues(panel, experimental)
    article_path = DATA_DIR / "sentiment_gap_20250801_articles.csv"
    pd.DataFrame(repair["articles"]).to_csv(assert_experimental_path(article_path), index=False)
    pd.DataFrame(repair["rows"]).to_csv(assert_experimental_path(DATA_DIR / "sentiment_gap_20250801.csv"), index=False)
    calendar_check = trading_calendar_check(panel, predictions.Signal_Date.min(), predictions.Date.max())
    if calendar_check["unexplained_library_dates"] or calendar_check["prices_absent_from_library"]:
        raise AssertionError(f"交易日曆有未解釋的差異：{calendar_check}")
    gap_status = prediction_gap_status(panel, predictions)

    # The legacy G replay filled on the signal close. Its saved ROI/cost is no
    # longer an expected value after the next-close timing correction.
    window = "2022-05-06_to_2026-03-25"
    books = {}
    metrics = []
    contributions = []
    folds_out = []
    for schedule, rule, name in (
        ("prediction_rows", "A", "G_row"),
        ("trading_calendar", "A", "G_calendar"),
    ):
        daily = run_book(predictions, panel, risk, rule, tau, schedule=schedule)
        books[name] = daily
        row = book_metrics(daily, rule, schedule, window)
        row["Name"] = name
        metrics.append(row)
    equalized = {}
    for rule in RULES:
        daily = books["G_calendar"] if rule == "A" else run_book(
            predictions, panel, risk, rule, tau, schedule="trading_calendar"
        )
        books[rule] = daily
        row = book_metrics(daily, rule, "trading_calendar", window)
        row["Name"] = f"weight_{rule}"
        twin = run_book(
            predictions, panel, risk, rule, tau, schedule="trading_calendar", equalize_gross=True,
        )
        equalized[rule] = book_metrics(twin, rule, "equalized_gross", window)
        row["Equalized_Gross_ROI_Pct"] = equalized[rule]["ROI_Pct"]
        row["Selection_ROI_Gap_pp"] = row["ROI_Pct"] - equalized[rule]["ROI_Pct"]
        metrics.append(row)
        contributions.append(stock_contribution(daily, rule))
        folds_out.append(fold_results(daily, folds).assign(Rule=rule))

    blocked, missing_signals = blocked_targets(predictions, panel)
    blocked = [pd.Timestamp(day) for day in blocked]
    hole_pnl = {
        name: float(books[name].loc[books[name].Date.isin(blocked), "Stock_PnL"].sum())
        for name in ("G_row", "G_calendar")
    }
    row_dates = set(books["G_row"].loc[books["G_row"].Trade, "Date"])
    calendar_dates = set(books["G_calendar"].loc[books["G_calendar"].Trade, "Date"])
    first_gap = min(row_dates.symmetric_difference(calendar_dates), default=None)

    side = _observed_window(panel)
    index_main = index_return(panel, predictions.Signal_Date.min(), predictions.Date.max())
    reference = pd.read_csv(ROOT / "reports" / "latest" / "444" / "strategy_metrics.csv")
    official_555 = pd.read_csv(ROOT / "reports" / "latest" / "555" / "strategy_metrics.csv")
    summary = {
        "study": "exploratory_not_a_new_independent_test",
        "same_close_execution": False,
        "execution_protocol": "next_session_close_signal_day_inputs_v2",
        "same_close_note": "前一訊號日決定權重與風控，下一交易日收盤成交；舊版績效須重新計算",
        "parameters_not_fit_on_backtest_roi": {
            "base": BASE_WEIGHT,
            "confident": CONFIDENT_WEIGHT,
            "reduced": UNCONFIDENT_WEIGHT,
            "rank_tilt": RANK_TILT,
            "rank_floor": RANK_FLOOR,
            "risk_exposure": RISK_EXPOSURE,
            "tau": "各折開始前 60 個交易日的絕對日報酬中位數，只用股價",
            "validation_predictions_saved": False,
        },
        "sentiment_repair": {
            "state": repair["state"],
            "coverage_limit": repair["coverage_limit"],
            "source_complete_for_category": repair["source_complete_for_category"],
            "coverage": repair["coverage"],
            "rows": [
                {**row, "Date": str(pd.Timestamp(row["Date"]).date())}
                for row in repair["rows"]
            ],
            "official_file_unchanged": True,
            "official_issue_count": int(len(original_issues)),
            "experimental_issue_count": int(len(experimental_issues)),
            "stored_predictions_not_rebuilt": True,
            "page_audits": repair["page_audits"],
            "match_review": repair["match_review"],
        },
        "trading_calendar": calendar_check,
        "prediction_gap": gap_status,
        "calendar_gap": {
            "missing_signal_days": [str(pd.Timestamp(day).date()) for day in missing_signals],
            "blocked_target_days": [str(day.date()) for day in blocked],
            "hole_stock_pnl_row_schedule": hole_pnl["G_row"],
            "hole_stock_pnl_calendar": hole_pnl["G_calendar"],
            "row_rebalance_days": len(row_dates),
            "calendar_rebalance_days": len(calendar_dates),
            "rebalance_dates_only_in_row_schedule": sorted(str(day.date()) for day in row_dates - calendar_dates),
            "rebalance_dates_only_in_calendar": sorted(str(day.date()) for day in calendar_dates - row_dates),
            "first_diverging_rebalance": None if first_gap is None else str(pd.Timestamp(first_gap).date()),
            "official_g_roi_pct": None,
            "legacy_g_comparison": "not_comparable_after_next_close_execution_fix",
        },
        "index_price_return_pct": {
            "walk_forward_window": index_main * 100,
            "observed_window_2026_03_26_to_2026_11_04": _index_or_none(panel, side),
        },
        "published_444_different_window": reference.to_dict(orient="records"),
        "published_555_different_window": official_555.to_dict(orient="records"),
        "observed_window_weight_rules": side["metrics"],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(OUT / "strategy_metrics.csv", index=False)
    pd.concat(contributions, ignore_index=True).to_csv(OUT / "stock_contribution.csv", index=False)
    pd.concat(folds_out, ignore_index=True).to_csv(OUT / "fold_roi.csv", index=False)
    tau.to_csv(OUT / "validation_tau.csv", index=False)
    original_issues.to_csv(OUT / "official_sentiment_issues.csv", index=False)
    experimental_issues.to_csv(OUT / "experimental_sentiment_issues.csv", index=False)
    books["G_row"][["Date", "Trade", "NAV_Reported", "Stock_PnL", "Rebalance_Cost"]].to_csv(
        OUT / "g_row_daily.csv", index=False
    )
    books["G_calendar"][["Date", "Trade", "Skipped_For_Missing_Prediction", "NAV_Reported", "Stock_PnL"]].to_csv(
        OUT / "g_calendar_daily.csv", index=False
    )
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _index_or_none(panel, side):
    if side["end"] not in panel.index or side["start"] not in panel.index:
        return None
    return index_return(panel, side["start"], side["end"]) * 100


def _observed_window(panel):
    path = ROOT / "reports" / "latest" / "555" / "test_predictions.csv"
    predictions = pd.read_csv(path, parse_dates=["Date", "Signal_Date"])
    predictions["Date"] = pd.to_datetime(predictions.Date).dt.normalize()
    predictions["Signal_Date"] = pd.to_datetime(predictions.Signal_Date).dt.normalize()
    start = pd.Timestamp(predictions.Date.min())
    end = pd.Timestamp(predictions.Date.max())
    if end not in panel.index or start not in panel.index:
        return {"metrics": [], "start": start, "end": end}
    _, risk = build_frames(panel, load_sentiment(RUN / "sentiment_snapshot.csv", panel.index))
    tau = validation_tau(panel, start, end)
    metrics = []
    for rule in RULES:
        daily = run_book(predictions, panel, risk, rule, tau, schedule="trading_calendar")
        row = book_metrics(daily, rule, "trading_calendar", "2026-03-26_to_2026-11-04")
        row["Used_To_Choose_Rules"] = False
        metrics.append(row)
    return {"metrics": metrics, "start": pd.Timestamp(predictions.Signal_Date.min()), "end": end}


if __name__ == "__main__":
    result = build_report()
    print(json.dumps({
        "repair": result["sentiment_repair"]["state"],
        "official_issues": result["sentiment_repair"]["official_issue_count"],
        "experimental_issues": result["sentiment_repair"]["experimental_issue_count"],
        "first_divergence": result["calendar_gap"]["first_diverging_rebalance"],
    }, ensure_ascii=False))
