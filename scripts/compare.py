"""Cross-notebook validator and neutral metrics comparator (no model training)."""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.common import ROOT,read_json,atomic_json

PROTOCOL_KEYS=['protocol_version','as_of_date','price_auto_adjust','initial_capital',
               'broker_fee_rate','sell_tax_rate','liquidate_at_end','time_step',
               'test_days','validation_days','annual_trading_days','risk_free_rate',
               'direction_zero_tolerance_pct','test_target_dates_sha256','price_panel_sha256']


def compare(baseline_dir, new_dir, destination):
    baseline_dir,new_dir,destination=map(Path,(baseline_dir,new_dir,destination))
    p444=read_json(baseline_dir/'evaluation_protocol.json')
    p555=read_json(new_dir/'evaluation_protocol.json')
    problems=[]
    for key in PROTOCOL_KEYS:
        if p444.get(key)!=p555.get(key):
            problems.append(f'{key}: 444={p444.get(key)!r} 555={p555.get(key)!r}')
    if p444.get('test_target_dates')!=p555.get('test_target_dates'):
        problems.append('測試日期清單不同')
    # Notebook protocols may have different training histories; disclose rather
    # than implying the training data are identical.
    if problems:
        raise ValueError('444 / 555 評估規格不一致，拒絕比較：\n'+'\n'.join(problems))
    m444=pd.read_csv(baseline_dir/'strategy_metrics.csv').iloc[0]
    m555=pd.read_csv(new_dir/'strategy_metrics.csv').iloc[0]
    for name,row in [('444',m444),('555',m555)]:
        for key in ['ROI_Pct','Sharpe','MDD_Pct','Direction_Accuracy_Pct','Test_Days']:
            if not np.isfinite(float(row[key])):raise ValueError(f'{name} {key} 非有效數字')
    nav444=pd.read_csv(baseline_dir/'backtest_nav.csv')
    nav555=pd.read_csv(new_dir/'backtest_nav.csv')
    if nav444.Date.tolist()!=nav555.Date.tolist():
        raise ValueError('兩份淨值表交易日不同')
    b_returns=nav444.Portfolio_Net.pct_change(fill_method=None).iloc[1:]
    n_returns=nav555.NAV.pct_change(fill_method=None).iloc[1:]
    benchmark444=nav444.Market_Index.astype(float)
    benchmark555=nav555.Market_Index.astype(float)
    if not np.allclose(benchmark444,benchmark555,rtol=1e-6,atol=1e-3):
        raise ValueError('兩版大盤基準淨值不一致')
    def metrics(row,ret,uid):
        return {'Strategy_ID':uid,'ROI_Pct':float(row.ROI_Pct),'Sharpe':float(row.Sharpe),
         'MDD_Pct':float(row.MDD_Pct),'Direction_Accuracy_Pct':float(row.Direction_Accuracy_Pct),
         'Profitable_Net_Day_Pct':float((ret>0).mean()*100),
         'Test_Days':int(row.Test_Days),'As_Of_Date':str(row.As_Of_Date),
         'Trade_Win_Rate_Pct':np.nan}
    base,new=metrics(m444,b_returns,'444'),metrics(m555,n_returns,'555')
    rows=[base,new]
    market_roi=float((benchmark444.iloc[-1]/benchmark444.iloc[0]-1)*100)
    market={'Strategy_ID':'^TWII benchmark','ROI_Pct':market_roi,
            'Sharpe':float(pd.read_csv(new_dir/'market_metrics.csv').iloc[0].Sharpe),
            'MDD_Pct':float(pd.read_csv(new_dir/'market_metrics.csv').iloc[0].MDD_Pct),
            'Direction_Accuracy_Pct':np.nan,'Profitable_Net_Day_Pct':float((benchmark444.pct_change().iloc[1:]>0).mean()*100),
            'Test_Days':int(m444.Test_Days),'As_Of_Date':str(m444.As_Of_Date),
            'Trade_Win_Rate_Pct':np.nan}
    rows.append(market)
    destination.mkdir(parents=True,exist_ok=True)
    result=pd.DataFrame(rows)
    result.to_csv(destination/'strategy_comparison.csv',index=False,encoding='utf-8-sig')
    # Different original code may use different metric names; compare by canonical
    # five-ticker subset without changing notebook exports.
    a=pd.read_csv(baseline_dir/'model_metrics.csv')
    b=pd.read_csv(new_dir/'model_metrics.csv')
    fields=['Ticker','Return_MAE_pp','Return_RMSE_pp','Price_MAE','Price_RMSE','Direction_Accuracy_Pct']
    missing=[col for col in fields if col not in a or col not in b]
    if missing:raise ValueError(f'模型指標欄位不全：{missing}')
    p=a[fields].merge(b[fields],on='Ticker',suffixes=('_444','_555'),validate='one_to_one')
    if len(p)!=5:raise ValueError('模型指標應包含五檔股票')
    p.to_csv(destination/'per_stock_comparison.csv',index=False,encoding='utf-8-sig')
    notes={
      'as_of_date':p444['as_of_date'],'comparison_status':'matched_test_dates_and_price_panel',
      'train_start_444':p444.get('training_start_444'),
      'train_start_555':min(str(r.get('Train_Start')) for r in p555.get('actual_split_audit',[])) if p555.get('actual_split_audit') else None,
      'warning':'Training start dates/models/strategy mechanics may differ. This is a descriptive out-of-sample comparison, not causal proof of sentiment contribution.',
      'trade_win_rate': 'undefined: neither notebook tracks completed trade round-trips',
      'direction_accuracy': 'direction match, NOT actual trade win rate',
      'profitable_net_day': 'fraction of positive portfolio net NAV daily returns, NOT trade win rate',
      'historical_sentiment': 'original uploaded feature history is not fully provenance-audited',
      'future_forecast': 'not used in historical ROI, Sharpe, or MDD'}
    atomic_json(destination/'comparison_readme.json',notes)
    report=['# 444 與 555 每日實測績效比較',
      f'資料截止日：**{p444["as_of_date"]}**；相同的 150 日測試區間與價格快照已檢查。','',
      '| 指標 | 444 單變量 + 固定股數 | 555 多變量 + 動態配置 | ^TWII 大盤 |',
      '|---|---:|---:|---:|']
    for label,key,suffix in [('累計 ROI','ROI_Pct','%'),('Sharpe','Sharpe',''),('最大回撤 MDD','MDD_Pct','%'),
                             ('方向準確率','Direction_Accuracy_Pct','%'),('正報酬交易日比例','Profitable_Net_Day_Pct','%')]:
        def v(obj):
            x=obj.get(key,np.nan)
            return '—' if pd.isna(x) else f'{float(x):.2f}{suffix}'
        report.append(f'| {label} | {v(base)} | {v(new)} | {v(market)} |')
    report+=['','**重要：**方向準確率不等於實際交易勝率；正報酬日比例也不等於逐筆成交勝率。',
      f'**訓練樣本起點：**444={notes["train_start_444"]}、555={notes["train_start_555"]}。',
      '兩個模型的特徵、配置與可能不同的訓練樣本長度會影響比較解讀；不可單憑 ROI 宣稱新聞特徵帶來因果改善。',
      '回測不包含真實下單、滑價與現實成交時點；^TWII 是理論價格指數。','']
    (destination/'comparison.md').write_text('\n'.join(report),encoding='utf-8')
    return result
