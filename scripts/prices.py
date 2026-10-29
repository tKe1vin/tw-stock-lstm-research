"""Download and audit one shared unadjusted-price snapshot for BOTH notebooks.

The raw Yahoo feed sometimes has isolated old rows with missing/zero Close.  We
report and discard only those *old* invalid rows; recent 400 XTAI sessions must
all have real, finite, strictly positive prices.  No forward-fill or fake close
is ever used.  A repair-enabled Yahoo retry is allowed, but the snapshot always
records which download mode succeeded.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.common import MARKET, ROOT, TICKERS, atomic_json, sessions


class PriceValidationError(ValueError):
    """A price frame violates the daily research data contract."""


def _prepare_frame(raw, ticker, as_of, protected_dates):
    """Return cleaned prices and audit notes, or an informative error."""
    if raw is None or raw.empty:
        raise PriceValidationError(f'{ticker}: Yahoo 回傳 0 筆資料')

    frame = raw.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        # yf.download() may return ('Close', ticker) columns even for one ticker.
        if 'Close' not in frame.columns.get_level_values(0):
            raise PriceValidationError(f'{ticker}: 缺少 Close 欄位，現有欄位={frame.columns.tolist()}')
        frame.columns = frame.columns.get_level_values(0)
    if 'Close' not in frame.columns:
        raise PriceValidationError(f'{ticker}: 缺少 Close 欄位，現有欄位={frame.columns.tolist()}')

    frame = frame.reset_index()
    date_column = 'Date' if 'Date' in frame.columns else 'Datetime'
    if date_column not in frame.columns:
        raise PriceValidationError(f'{ticker}: 無法辨識交易日期，欄位={frame.columns.tolist()}')
    frame = frame[[date_column, 'Close']].rename(columns={date_column: 'Date'})
    frame['Date'] = pd.to_datetime(frame['Date'], errors='raise').dt.tz_localize(None).dt.normalize()
    frame['Close'] = pd.to_numeric(frame['Close'], errors='coerce')
    frame = frame.loc[frame.Date <= as_of].copy()

    if frame.empty:
        raise PriceValidationError(f'{ticker}: 指定截止日 {as_of.date()} 以前沒有股價')

    # Duplicate sessions are only safely collapsible when their values agree.
    duplicate_rows = frame.duplicated('Date', keep=False)
    duplicate_days = int(frame.loc[duplicate_rows, 'Date'].nunique())
    if duplicate_days:
        # NaN treated as one distinct value by pandas: two NaNs are equivalent.
        differences = frame.loc[duplicate_rows].groupby('Date')['Close'].nunique(dropna=False)
        conflicts = differences[differences > 1].index
        if len(conflicts):
            examples = frame.loc[frame.Date.isin(conflicts[:3]), ['Date', 'Close']]
            raise PriceValidationError(
                f'{ticker}: 同一交易日出現相互矛盾的收盤價：{examples.to_dict("records")}'
            )
        frame = frame.drop_duplicates('Date', keep='last')

    latest = frame.Date.max()
    if latest != as_of:
        raise PriceValidationError(
            f'{ticker}: 目標 {as_of.date()} 尚無有效當日資料，Yahoo 最新日期={latest.date()}'
        )

    close = frame['Close'].to_numpy(dtype=float)
    invalid = ~np.isfinite(close) | (close <= 0)
    recent = frame.Date.isin(protected_dates).to_numpy(dtype=bool)
    n_recent = int((invalid & recent).sum())
    if n_recent:
        examples = frame.loc[invalid & recent, ['Date', 'Close']].head(8)
        examples = [f'{d.date()}: {v}' for d, v in examples.itertuples(index=False, name=None)]
        raise PriceValidationError(
            f'{ticker}: 最近 400 個交易日有 {n_recent} 筆無效 Close，'
            f'日期/值={examples}（不能使用過期或補造價格）'
        )

    n_old = int((invalid & ~recent).sum())
    if n_old:
        examples = frame.loc[invalid & ~recent, ['Date', 'Close']].head(8)
        examples = [f'{d.date()}: {v}' for d, v in examples.itertuples(index=False, name=None)]
        print(
            f'[PRICE][WARN] {ticker}: 舊歷史 {n_old} 筆 Close 無效，已刪除該列；'
            f'範例={examples}，近期資料仍嚴格檢查。', flush=True
        )
        frame = frame.loc[~invalid].copy()

    available = pd.DatetimeIndex(frame['Date'])
    missing_recent = protected_dates.difference(available)
    if len(missing_recent):
        raise PriceValidationError(
            f'{ticker}: 最近 400 個 XTAI 交易日缺 {len(missing_recent)} 日：'
            f'{missing_recent.strftime("%Y-%m-%d")[:12].tolist()}'
        )

    frame['ticker'] = ticker
    return frame[['Date', 'Close', 'ticker']], {
        'download_rows': int(len(raw)),
        'valid_rows': int(len(frame)),
        'dropped_old_invalid_close': n_old,
        'deduplicated_identical_dates': duplicate_days,
        'last_date': as_of.date().isoformat(),
    }


def refresh_prices(as_of, path=None, start='2014-01-01'):
    import yfinance as yf

    as_of = pd.Timestamp(as_of).normalize()
    end_excl = (as_of + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    # The protected period is deliberately calendar-based, NOT the last 400
    # returned Yahoo rows, to prevent missing sessions from shifting the window.
    protected_dates = sessions(as_of - pd.Timedelta(days=1100), as_of)[-400:]
    if not len(protected_dates) or protected_dates[-1] != as_of:
        raise ValueError(f'截止日期不是交易日：{as_of.date()}')

    pieces = []
    audit = {}
    for ticker in [*TICKERS, MARKET]:
        errors = []
        # First use unchanged raw Close; if Yahoo returns malformed prices,
        # retry once with its documented repair feature. Third try may recover
        # from a temporary upstream response problem.
        strategies = [False, True, True]
        for attempt, repair in enumerate(strategies, start=1):
            try:
                raw = yf.download(
                    ticker, start=start, end=end_excl, auto_adjust=False,
                    repair=repair, keepna=True, progress=False,
                    threads=False, timeout=45,
                )
                frame, stats = _prepare_frame(raw, ticker, as_of, protected_dates)
                stats['yahoo_repair_enabled'] = repair
                audit[ticker] = stats
                pieces.append(frame)
                print(
                    f'[PRICE] {ticker}: {len(frame)} 筆，有效最新日 {as_of.date()}，'
                    f'repair={repair}，舊無效筆數={stats["dropped_old_invalid_close"]}',
                    flush=True,
                )
                break
            except Exception as exc:
                message = f'[PRICE][attempt {attempt}/3] {ticker}: {type(exc).__name__}: {exc}'
                print(message, flush=True)
                errors.append(message)
                if attempt < len(strategies):
                    time.sleep(3 * attempt)
        else:
            raise RuntimeError(
                f'取得 {ticker} 股價失敗，未使用過期資料。最後問題：{errors[-1]}'
            )

    data = pd.concat(pieces, ignore_index=True).sort_values(['Date', 'ticker']).reset_index(drop=True)
    panel = data.pivot(index='Date', columns='ticker', values='Close')[[*TICKERS, MARKET]]
    common = panel.dropna()
    missing = protected_dates.difference(common.index)
    if len(missing):
        raise ValueError(
            f'六組價格最近 400 個交易日未完全對齊：'
            f'{missing.strftime("%Y-%m-%d").tolist()}'
        )

    path = Path(path or ROOT / 'data/prices/historical_prices.csv')
    path.parent.mkdir(parents=True, exist_ok=True)
    # Save only after *all* instruments are validated; failure never commits
    # a new apparently complete snapshot.
    tmp = path.with_suffix(path.suffix + '.tmp')
    data.to_csv(tmp, index=False, encoding='utf-8-sig')
    tmp.replace(path)
    atomic_json(path.with_suffix('.meta.json'), {
        'as_of_date': as_of.date().isoformat(),
        'source': 'yfinance Yahoo unadjusted Close, auto_adjust=False',
        'records': len(data),
        'common_dates': len(common),
        'first_common_date': str(common.index[0].date()),
        'protected_sessions_count': len(protected_dates),
        'price_quality_audit': audit,
        'note': 'Only old invalid rows may be removed; recent 400 sessions must be present. No synthetic prices.',
    })
    return path
