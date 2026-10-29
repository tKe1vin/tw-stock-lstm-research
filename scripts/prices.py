"""Download shared prices for two notebooks; exclude verified typhoon closures."""
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.common import MARKET, ROOT, TICKERS, SPECIAL_MARKET_CLOSURE_DATES, atomic_json, sessions

class PriceValidationError(ValueError):
    pass

def _prepare_frame(raw, ticker, as_of, protected_dates):
    if raw is None or raw.empty:
        raise PriceValidationError(f'{ticker}: Yahoo 回傳 0 筆資料')
    frame = raw.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        if 'Close' not in frame.columns.get_level_values(0):
            raise PriceValidationError(f'{ticker}: 缺少 Close 欄位')
        frame.columns = frame.columns.get_level_values(0)
    if 'Close' not in frame.columns:
        raise PriceValidationError(f'{ticker}: 缺少 Close 欄位')
    frame = frame.reset_index()
    date_column = 'Date' if 'Date' in frame.columns else 'Datetime'
    if date_column not in frame.columns:
        raise PriceValidationError(f'{ticker}: 缺少交易日期欄位')
    frame = frame[[date_column,'Close']].rename(columns={date_column:'Date'})
    frame['Date'] = pd.to_datetime(frame.Date).dt.tz_localize(None).dt.normalize()
    frame['Close'] = pd.to_numeric(frame.Close, errors='coerce')
    frame = frame.loc[frame.Date<=as_of].copy()
    if frame.empty:
        raise PriceValidationError(f'{ticker}: {as_of.date()} 前無股價')
    closure_mask = frame.Date.isin(SPECIAL_MARKET_CLOSURE_DATES)
    removed = int(closure_mask.sum())
    if removed:
        print(f'[PRICE][INFO] {ticker}: 排除已確認休市日 {frame.loc[closure_mask,"Date"].dt.strftime("%Y-%m-%d").tolist()}',flush=True)
        frame = frame.loc[~closure_mask].copy()
    dup = frame.duplicated('Date',keep=False)
    ndup = int(frame.loc[dup,'Date'].nunique())
    if ndup:
        conflicting = frame.loc[dup].groupby('Date')['Close'].nunique(dropna=False)
        bad = conflicting[conflicting>1].index
        if len(bad):
            raise PriceValidationError(f'{ticker}: 同日股價矛盾 {bad.strftime("%Y-%m-%d").tolist()}')
        frame = frame.drop_duplicates('Date',keep='last')
    latest = frame.Date.max()
    if latest != as_of:
        raise PriceValidationError(f'{ticker}: 目標 {as_of.date()}，最新只有 {latest.date()}')
    close = frame.Close.to_numpy(dtype=float)
    invalid = ~np.isfinite(close) | (close<=0)
    recent = frame.Date.isin(protected_dates).to_numpy(dtype=bool)
    if (invalid & recent).any():
        examples = frame.loc[invalid & recent,['Date','Close']].head(8).to_dict('records')
        raise PriceValidationError(f'{ticker}: 最近 400 個交易日有無效 Close：{examples}')
    old_invalid = int((invalid & ~recent).sum())
    if old_invalid:
        print(f'[PRICE][WARN] {ticker}: 排除 {old_invalid} 筆較早期無效股價',flush=True)
        frame = frame.loc[~invalid].copy()
    missing = protected_dates.difference(pd.DatetimeIndex(frame.Date))
    if len(missing):
        raise PriceValidationError(f'{ticker}: 最近 400 個交易日缺 {len(missing)} 日：{missing.strftime("%Y-%m-%d").tolist()}')
    frame['ticker'] = ticker
    return frame[['Date','Close','ticker']], {'valid_rows':len(frame),'dropped_old_invalid_close':old_invalid,
      'deduplicated_identical_dates':ndup,'removed_confirmed_closure_rows':removed,'last_date':str(as_of.date())}

def refresh_prices(as_of, path=None, start='2014-01-01'):
    import yfinance as yf
    as_of = pd.Timestamp(as_of).normalize()
    protected_dates = sessions(as_of-pd.Timedelta(days=1100),as_of)[-400:]
    if not len(protected_dates) or protected_dates[-1]!=as_of:
        raise ValueError(f'截止日期不是交易日：{as_of.date()}')
    pieces=[]; audit={}
    for ticker in [*TICKERS,MARKET]:
        errors=[]
        for attempt,repair in enumerate([False,True,True],1):
            try:
                raw=yf.download(ticker,start=start,end=(as_of+pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
                    auto_adjust=False,repair=repair,keepna=True,progress=False,threads=False,timeout=45)
                frame,stats=_prepare_frame(raw,ticker,as_of,protected_dates)
                stats['yahoo_repair_enabled']=repair
                pieces.append(frame);audit[ticker]=stats
                print(f'[PRICE] {ticker}: {len(frame)} 列，截止 {as_of.date()}，repair={repair}',flush=True)
                break
            except Exception as exc:
                msg=f'{ticker} attempt {attempt}/3: {type(exc).__name__}: {exc}'
                print('[PRICE][ERROR]',msg,flush=True);errors.append(msg)
                if attempt<3:time.sleep(3*attempt)
        else:
            raise RuntimeError(f'取得 {ticker} 股價失敗，未使用過期資料。{errors[-1]}')
    data=pd.concat(pieces,ignore_index=True).sort_values(['Date','ticker']).reset_index(drop=True)
    panel=data.pivot(index='Date',columns='ticker',values='Close')[[*TICKERS,MARKET]]
    common=panel.dropna()
    missing=protected_dates.difference(common.index)
    if len(missing):
        raise ValueError(f'六組股價未對齊：{missing.strftime("%Y-%m-%d").tolist()}')
    path=Path(path or ROOT/'data/prices/historical_prices.csv')
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    data.to_csv(tmp,index=False,encoding='utf-8-sig');tmp.replace(path)
    atomic_json(path.with_suffix('.meta.json'),{'as_of_date':str(as_of.date()),
        'source':'yfinance unadjusted Close','records':len(data),'common_dates':len(common),
        'protected_sessions_count':len(protected_dates),'price_quality_audit':audit,
        'excluded_official_closure_dates':[str(d.date()) for d in SPECIAL_MARKET_CLOSURE_DATES if d<=as_of]})
    return path
