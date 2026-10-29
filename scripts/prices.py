"""Download one shared, validated, unadjusted price snapshot for BOTH notebooks."""
from __future__ import annotations
import time
import pandas as pd
import numpy as np
from pathlib import Path
from scripts.common import TICKERS, MARKET, ROOT, sessions, atomic_json


def refresh_prices(as_of, path=None, start='2014-01-01'):
    import yfinance as yf
    as_of = pd.Timestamp(as_of).normalize()
    end_excl = (as_of+pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    pieces=[]
    for ticker in [*TICKERS,MARKET]:
        last_error=None
        for attempt in range(3):
            try:
                frame=yf.download(ticker,start=start,end=end_excl,auto_adjust=False,
                                  progress=False,threads=False,timeout=45)
                if frame is None or frame.empty:
                    raise ValueError(f'{ticker} returned zero rows')
                if isinstance(frame.columns,pd.MultiIndex):
                    frame.columns=frame.columns.get_level_values(0)
                if 'Close' not in frame.columns:
                    raise ValueError(f'{ticker} no Close field')
                frame=frame.reset_index()[['Date','Close']]
                frame['Date']=pd.to_datetime(frame.Date,utc=False).dt.tz_localize(None).dt.normalize()
                frame['ticker']=ticker
                frame['Close']=pd.to_numeric(frame.Close,errors='raise')
                frame=frame[frame.Date<=as_of]
                if frame.empty or frame.Date.max()!=as_of:
                    raise ValueError(f'{ticker} 股價尚未更新到 {as_of.date()}，目前最大日 {frame.Date.max() if len(frame) else None}')
                if frame.duplicated(['Date']).any() or not np.isfinite(frame.Close).all() or (frame.Close<=0).any():
                    raise ValueError(f'{ticker} prices invalid')
                pieces.append(frame)
                print(f'股價已取得 {ticker} {len(frame)} 列（{frame.Date.max().date()}）',flush=True)
                break
            except Exception as exc:
                last_error=exc
                if attempt < 2:
                    time.sleep(3*(attempt+1))
        else:
            raise RuntimeError(f'取得 {ticker} 股價失敗，不使用過期資料') from last_error
    data=pd.concat(pieces,ignore_index=True)
    data=data.sort_values(['Date','ticker']).reset_index(drop=True)
    panel=data.pivot(index='Date',columns='ticker',values='Close')[[*TICKERS,MARKET]]
    common=panel.dropna()
    expected=sessions(common.index[0],as_of)
    # Do not silently truncate the last 150+60+80+90 observations.
    missing=expected[-400:].difference(common.index)
    if len(missing):
        raise ValueError(f'價格最近缺少 XTAI 交易日：{missing.strftime("%Y-%m-%d").tolist()}')
    path=Path(path or ROOT/'data/prices/historical_prices.csv')
    path.parent.mkdir(parents=True,exist_ok=True)
    data.to_csv(path,index=False,encoding='utf-8-sig')
    atomic_json(path.with_suffix('.meta.json'),{'as_of_date':as_of.date().isoformat(),
        'source':'yfinance auto_adjust=False','records':len(data),'common_dates':len(common),
        'first_common_date':str(common.index[0].date())})
    return path
