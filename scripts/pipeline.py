"""One daily command orchestrates sources and SEPARATE independent notebooks."""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
from scripts.common import ROOT,TZ,current_completed_session,check_date_in_calendar,read_json,atomic_json


def checked_run(kind, env):
    print(f'\n===== 執行 {kind} Notebook =====',flush=True)
    cmd=[sys.executable,'-m','scripts.execute_nb',kind]
    subprocess.run(cmd,env=env,cwd=ROOT,check=True)


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--as-of',help='指定已完成交易日，省略則按台北時間 15:00 判斷')
    p.add_argument('--force',action='store_true',help='即使同日已完成，也重新執行並更新當日摘要')
    p.add_argument('--no-fetch',action='store_true',help='只使用已備妥的完整股價與新聞（離線驗收）')
    p.add_argument('--check',action='store_true',help='只檢查目前資料、日期、目錄，不訓練模型')
    args=p.parse_args(argv)
    os.chdir(ROOT)
    asof=check_date_in_calendar(args.as_of) if args.as_of else current_completed_session()
    stamp=asof.strftime('%Y-%m-%d')
    print('台北時間：',datetime.now(TZ).isoformat(),'最新完成交易日：',stamp,flush=True)
    if not args.check and asof != current_completed_session():
        raise ValueError('每日正式日誌只接受最新已完成交易日；歷史回播不得冒充每日真實建議。')
    metadata=ROOT/'reports/latest/metadata.json'
    if metadata.exists() and read_json(metadata).get('as_of_date')==stamp and not args.force and not args.check:
        print('已有相同資料截止日的成功報表，避免重複重訓。若需強制重跑請指定 --force。')
        return 0
    if args.check:
        f=ROOT/'data/sentiment_features_2026.csv'
        prices=ROOT/'data/prices/historical_prices.csv'
        print('情緒檔：',f.exists(),'股價檔：',prices.exists())
        if f.exists():
            d=pd.read_csv(f);print('情緒資料起迄：',d.Date.min(),d.Date.max(),'筆數：',len(d))
        if prices.exists():
            d=pd.read_csv(prices);print('股價資料起迄：',d.Date.min(),d.Date.max(),'筆數：',len(d))
        return 0
    price_path=ROOT/'data/prices/historical_prices.csv'
    feature=ROOT/'data/sentiment_features_2026.csv'
    if args.no_fetch:
        if not price_path.exists():raise FileNotFoundError(price_path)
        pdates=pd.read_csv(price_path,usecols=['Date']).Date
        if str(pd.to_datetime(pdates).max().date())!=stamp:raise ValueError('本地股價尚未更新到指定日')
        sent=pd.read_csv(feature)
        if not (pd.to_datetime(sent.Date)==asof).any():raise ValueError('情緒尚未更新到指定日')
    else:
        from scripts.prices import refresh_prices
        from scripts.news_update import update
        refresh_prices(asof,price_path)
        update(asof,feature)
    # Do not share model state/cache between the two process boundaries.
    runid=datetime.now(TZ).strftime('%Y%m%dT%H%M%S%f')
    baseline=ROOT/'outputs/444/runs'/runid
    env=os.environ.copy()
    env.update({'LSTM_AS_OF_DATE':stamp,'LSTM_PRICE_CSV':str(price_path.resolve()),
        'LSTM_FEATURE_CSV':str(feature.resolve()),'LSTM_SAVE_MODELS':env.get('LSTM_SAVE_MODELS','0'),
        'MPLBACKEND':'Agg','TF_CPP_MIN_LOG_LEVEL':'2','OMP_NUM_THREADS':'2','TF_NUM_INTEROP_THREADS':'2',
        'TF_NUM_INTRAOP_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
    env['LSTM_OUTPUT_DIR']=str(baseline.resolve())
    checked_run('444',env)
    env['LSTM_OUTPUT_DIR']=str((ROOT/'outputs/555').resolve())
    checked_run('555',env)
    meta555=read_json(ROOT/'outputs/555/latest.json')
    if meta555.get('as_of_date')!=stamp or meta555.get('run_mode')!='daily':
        raise ValueError('555 回傳非今日成功的 daily 執行紀錄')
    new=ROOT/'outputs/555'/meta555['output_directory']
    from scripts.compare import compare
    from scripts.journal import write_journals
    comparison=ROOT/'reports/daily'/stamp/'comparison_data'
    table=compare(baseline,new,comparison)
    day_dir=write_journals(asof,baseline,new,comparison)
    # Lightweight stable API for a future front end.
    latest=ROOT/'reports/latest'
    latest.mkdir(parents=True,exist_ok=True)
    for kind,run_dir in [('444',baseline),('555',new)]:
        dest=latest/kind;dest.mkdir(exist_ok=True)
        for file in ['strategy_metrics.csv','model_metrics.csv','backtest_nav.csv',
                     'test_predictions.csv','next_day_recommendation.csv','evaluation_protocol.json']:
            if not (run_dir/file).exists():raise FileNotFoundError(run_dir/file)
            shutil.copy2(run_dir/file,dest/file)
    for file in ['strategy_comparison.csv','per_stock_comparison.csv','comparison.md','comparison_readme.json']:
        shutil.copy2(comparison/file,latest/file)
    atomic_json(latest/'metadata.json',{'as_of_date':stamp,'completed_at':datetime.now(TZ).isoformat(),
        '444_source':str(baseline.relative_to(ROOT)),
        '555_source':str(new.relative_to(ROOT)),'status':'complete',
        'note':'No brokerage orders were executed.'})
    print('\n✅ 444 與 555 都已完成，且兩份相同日期、相同價格快照的比較已通過。',flush=True)
    print(table.to_string(index=False),flush=True)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
