"""Adapter ONLY; 444 LSTM modeling and model internals stay in its notebook."""
from __future__ import annotations
import hashlib
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.common import ROOT, TICKERS, MARKET, TZ, atomic_json


def finalize(ns):
    b=ns['get_history_bundle']()
    nav=b['nav']
    panel=ns['prepare_prices']()
    asof=pd.Timestamp(panel.index[-1]).normalize()
    expected=pd.Timestamp(__import__('os').environ['LSTM_AS_OF_DATE']).normalize()
    if asof!=expected:
        raise ValueError(f'444 資料截止 {asof.date()} != 共用截止 {expected.date()}')
    out=Path(ns['OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True)
    created=datetime.now(TZ)
    run_id=created.strftime('%Y%m%dT%H%M%S%f')
    test_dates=nav.index[1:]
    if len(test_dates)!=150:
        raise ValueError('444 不是 150 日測試')
    direction=b['portfolio_direction'].Direction_Correct.mean()*100
    metric={'Strategy_ID':'444_univariate_fixed','Strategy':'444 univariate fixed shares net',
      'Run_ID':run_id,'As_Of_Date':str(asof.date()),'Protocol_Version':'lstm-evaluation-v1',
      'ROI_Pct':float((nav.Portfolio_Net.iloc[-1]/ns['INITIAL_CAPITAL']-1)*100),
      'Sharpe':float(ns['calculate_sharpe'](nav.Portfolio_Net)),
      'MDD_Pct':float(ns['calculate_max_drawdown'](nav.Portfolio_Net)),
      'Direction_Accuracy_Pct':float(direction),
      'Test_Days':150,'Test_Start':str(test_dates[0].date()),'Test_End':str(test_dates[-1].date())}
    pd.DataFrame([metric]).to_csv(out/'strategy_metrics.csv',index=False,encoding='utf-8-sig')
    # Match 555 explicit date and price panel hashing, even though 444 internally
    # lists the five stock symbols in a different order.
    ordered_panel=panel[[*TICKERS,MARKET]]
    protocol={'protocol_version':'lstm-evaluation-v1','strategy_id':'444_univariate_fixed',
     'as_of_date':str(asof.date()),'price_auto_adjust':ns['AUTO_ADJUST'],
     'initial_capital':ns['INITIAL_CAPITAL'],'broker_fee_rate':ns['BROKER_FEE_RATE'],
     'sell_tax_rate':ns['SELL_TAX_RATE'],'liquidate_at_end':ns['LIQUIDATE_AT_END'],
     'time_step':ns['SEQ_LENGTH'],'test_days':ns['TEST_DAYS'],
     'validation_days':ns['VALIDATION_DAYS'],'annual_trading_days':ns['ANNUAL_TRADING_DAYS'],
     'risk_free_rate':ns['RISK_FREE_RATE'],'direction_zero_tolerance_pct':ns['DIRECTION_ZERO_TOL_PCT'],
     'test_target_dates':test_dates.strftime('%Y-%m-%d').tolist(),
     'test_target_dates_sha256':hashlib.sha256('\n'.join(test_dates.strftime('%Y-%m-%d')).encode()).hexdigest(),
     'price_panel_sha256':hashlib.sha256(ordered_panel.to_csv(float_format='%.10f').encode()).hexdigest(),
     'validation_target_dates':panel.index[-210:-150].strftime('%Y-%m-%d').tolist(),
     'test_predictions_not_trading_profit':True,
     'training_start_444':str(pd.DataFrame([m['split'] for m in b['models'].values()]).Train_Start.min()),
     'model_target':'next-trading-day close price (not a trade signal)'}
    atomic_json(out/'evaluation_protocol.json',protocol)
    # Next trading day baseline: fixed 20% reference allocation, not an order.
    target=ns['future_trading_dates'](asof,1)[0]
    rows=[]
    for ticker in TICKERS:
        last,vals=ns['predict_price_sequence'](ticker,1)
        if pd.Timestamp(last).normalize()!=asof:raise ValueError('444 model stale')
        prev,pred=vals
        rows.append({'Created_At':created.isoformat(),'Run_ID':run_id,'Run_Mode':'daily',
          'Signal_Date':str(asof.date()),'Target_Date':str(target.date()),'Ticker':ticker,
          'Previous_Close':prev,'Predicted_Close':pred,
          'Predicted_Return_Pct':(pred/prev-1)*100,
          'Target_Weight':0.2,'Cash_Weight':0.0,
          'Note':'fixed-share reference, not actual drifted holdings or trade instruction'})
    pred=pd.DataFrame(rows)
    pred.to_csv(out/'next_day_recommendation.csv',index=False,encoding='utf-8-sig')
    # Save a daily immutable recommendation; never retrospectively create one.
    hist=ROOT/'outputs/444/recommendation_history';hist.mkdir(parents=True,exist_ok=True)
    file=hist/(run_id+'.csv')
    if file.exists():raise FileExistsError(file)
    pred.to_csv(file,index=False,encoding='utf-8-sig')
    # Daily portfolio reference NAV/ benchmark use original notebook values.
    protocol['price_panel_columns']=[*TICKERS,MARKET]
    atomic_json(out/'evaluation_protocol.json',protocol)
    atomic_json(out/'run_manifest.json',{'status':'completed','run_id':run_id,'created_at':created.isoformat(),
          'as_of_date':str(asof.date()),'output_directory':str(out),'strategy_id':'444_univariate_fixed'})
    print(f'[444] 日誌/比較資料準備完成：{out}',flush=True)
