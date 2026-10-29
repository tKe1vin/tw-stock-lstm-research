"""Immutable daily analysis records for two independent strategies."""
from __future__ import annotations
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.common import ROOT,TICKERS


def _format_pct(value):
    return '—' if pd.isna(value) else f'{float(value):+.2f}%'


def past_444(as_of):
    history=ROOT/'outputs/444/recommendation_history'
    old=[]
    for path in history.glob('*.csv'):
        rows=pd.read_csv(path)
        if not rows.empty:old.append(rows)
    if not old:return '第一次執行，尚未累積可驗證的逐日建議。'
    all_rows=pd.concat(old,ignore_index=True)
    all_rows['Target_Date']=pd.to_datetime(all_rows.Target_Date)
    all_rows['Created_At']=pd.to_datetime(all_rows.Created_At,utc=True)
    available=all_rows.loc[all_rows.Target_Date<=pd.Timestamp(as_of)].copy()
    if available.empty:return '前次建議尚未到達實現日，暫不評分。'
    available=available.sort_values('Created_At').drop_duplicates(['Target_Date','Ticker'],keep='last')
    latest=available.Target_Date.max()
    part=available.loc[available.Target_Date==latest]
    if len(part)!=5 or set(part.Ticker)!=set(TICKERS):return '前次預測缺少五檔股票，不計算組合方向。'
    prices=pd.read_csv(ROOT/'data/prices/historical_prices.csv')
    prices['Date']=pd.to_datetime(prices.Date)
    panel=prices.pivot(index='Date',columns='ticker',values='Close')
    signal=pd.Timestamp(part.Signal_Date.iloc[0]);target=pd.Timestamp(latest)
    if signal not in panel.index or target not in panel.index:
        return f'{latest.date()} 尚無股價，不應回填預測評分。'
    part=part.set_index('Ticker').loc[TICKERS]
    actual=(panel.loc[target,TICKERS]/panel.loc[signal,TICKERS]-1).to_numpy()*100
    predicted=part.Predicted_Return_Pct.to_numpy()
    # 20% theoretical next-day reference, not actual fixed-share portfolio P&L.
    ap=float(np.dot(actual,np.full(5,.2)));pp=float(np.dot(predicted,np.full(5,.2)))
    correct=np.sign(pp)==np.sign(ap)
    return (f'最近已完成建議：**{latest.date()}**；預測參考組合毛報酬 **{_format_pct(pp)}**，'
            f'實際參考組合毛報酬 **{_format_pct(ap)}**；方向 **{"正確" if correct else "錯誤"}**。'
            '此為獨立保存的當時建議，非真實下單損益。')


def past_555(run_dir):
    path=Path(run_dir)/'daily_portfolio_review.csv'
    if not path.exists():return '尚未有可供驗證的保存建議。'
    data=pd.read_csv(path)
    if data.empty:return '第一次執行或建議尚未實現，沒有歷史檢討。'
    last=data.iloc[-1]
    return (f'已檢討建議 **{last.Target_Date}**：預期理論毛報酬 **{_format_pct(last.Predicted_Gross_Return_Pct)}**，'
            f'實際理論毛報酬 **{_format_pct(last.Actual_Gross_Return_Pct)}**。'
            '此為推薦權重的理論毛回報，並非交易帳戶淨收益。')


def write_journals(asof, bdir, ndir, cdir):
    asof=str(pd.Timestamp(asof).date())
    day_dir=ROOT/'reports/daily'/asof;day_dir.mkdir(parents=True,exist_ok=True)
    for kind,run_dir in [('444',bdir),('555',ndir)]:
        metrics=pd.read_csv(Path(run_dir)/'strategy_metrics.csv').iloc[0]
        rec=pd.read_csv(Path(run_dir)/'next_day_recommendation.csv')
        if rec.empty or set(rec.Ticker)!=set(TICKERS):
            raise ValueError(f'{kind} 未產出完整五檔建議，不記錄為成功日')
        review=past_444(asof) if kind=='444' else past_555(run_dir)
        report=[f'## {asof} — {kind} 每日研究日誌',
            f'- 執行編號：`{metrics.Run_ID}`',
            f'- 測試區間：{metrics.Test_Start} ~ {metrics.Test_End}（{int(metrics.Test_Days)} 個交易日）',
            f'- 回測 ROI：**{_format_pct(metrics.ROI_Pct)}**；Sharpe：**{float(metrics.Sharpe):.3f}**；最大回撤：**{_format_pct(metrics.MDD_Pct)}**',
            f'- 歷史方向準確率：**{float(metrics.Direction_Accuracy_Pct):.2f}%**（不是逐筆交易勝率）',
            '', '### 過去預測的核對',review,'','### 下一個交易日預測（未來、尚無實現損益）',
            f'預測目標日：**{str(rec.Target_Date.iloc[0])[:10]}**','',
            '| 股票 | 預測次日報酬 | 目標權重 |','|---|---:|---:|']
        for _,r in rec.iterrows():
            report.append(f'| {r.Ticker} | {_format_pct(r.Predicted_Return_Pct)} | {float(r.Target_Weight)*100:.2f}% |')
        report+=['',f'現金預留：{float(rec.Cash_Weight.iloc[0])*100:.2f}%','',
             '研究用途：未實際交易、未考慮下單滑價，不能把預測當獲利。','']
        text='\n'.join(report)
        (day_dir/f'{kind}.md').write_text(text,encoding='utf-8')
        journal=ROOT/'reports/journals'/f'{kind}.md'
        journal.parent.mkdir(parents=True,exist_ok=True)
        if journal.exists():
            old=journal.read_text(encoding='utf-8')
            sections=old.split('\n## ')[1:]
            prior=['## '+x for x in sections if not x.startswith(asof+' —')]
            content=f'# {kind} 日誌（最新在前）\n\n'+text+'\n\n'+ '\n\n'.join(prior)
        else:content=f'# {kind} 日誌（最新在前）\n\n'+text+'\n'
        journal.write_text(content,encoding='utf-8')
    shutil.copy2(Path(cdir)/'comparison.md',day_dir/'comparison.md')
    for name in ['strategy_comparison.csv','per_stock_comparison.csv','comparison_readme.json']:
        shutil.copy2(Path(cdir)/name,day_dir/name)
    return day_dir
