"""Fixed descriptive controls on already-observed history.

This does not select a V3 candidate, train a model, or publish a live signal.
Default: read-only stdout. --output-dir optionally retains diagnostic outputs.
Run from a trusted checkout: python research/v3/diagnose.py
"""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import argparse,hashlib,subprocess
parser=argparse.ArgumentParser(description='Fixed descriptive extension diagnostics; no training, tuning or live writes')
parser.add_argument('--output-dir',type=Path)
args=parser.parse_args()
if args.output_dir: args.output_dir.mkdir(parents=True,exist_ok=True)
from scripts.research_extension import prepare_data,load_config,evaluate_books,fixed_holding_endpoint
from experiments.v2.books import run_book,metrics,index_metrics
from experiments.risk_experiment import INITIAL_CAPITAL,TICKERS,MARKET
base=ROOT/'reports/research_extension/2026-11-06'
config=load_config(base/'registered_config.json')
data=prepare_data(base/'inputs/prices.csv',base/'inputs/sentiment.csv','2026-11-06',config)
pred=pd.read_csv(base/'predictions.csv')
test=pred[pred.Split=='test'];start,end=fixed_holding_endpoint(data['panel'],test)
calendar=pd.to_datetime(test.Signal_Date.unique())
old_metrics=pd.read_csv(base/'strategy_metrics.csv').set_index('Name')
accounting={}
for name in ['price_lstm','sentiment_risk_lstm','matched_risk_equal','full_exposure_buy_hold']:
 nav=pd.read_csv(base/f'nav_{name}.csv')
 accounting[name]={'net_roi_pp':float(old_metrics.loc[name,'ROI_Pct']), 'pnl_contribution_pp':float(nav.Stock_PnL.sum()/INITIAL_CAPITAL*100),'cost_contribution_pp':float((nav.Rebalance_Cost.sum()+nav.Liquidation_Cost.sum())/INITIAL_CAPITAL*100),'cost_currency':float(nav.Rebalance_Cost.sum()+nav.Liquidation_Cost.sum()),'average_equity_pct':float(nav.Equity_Weight.mean()*100),'risk_off_flag_days':int(nav.Risk_Off.sum()),'days':len(nav),'risk_off_at_rebalance':int(nav.loc[nav.Trade,'Risk_Off'].sum()),'rebalance_days':int(nav.Trade.sum())}
no_risk=data['risk'].copy();no_risk['Risk_Off']=False
ablation={}
books={}
for name,model in [('full_equal_weekly',None),('full_price_weekly','price_lstm'),('full_news_weekly','sentiment_risk_lstm')]:
 if model is None:
  book=run_book(data['panel'],no_risk,None,data['sentiment'],start,end,5,'equal',signal_calendar=calendar)
 else:
  block=pred[pred.Model==model]
  book=run_book(data['panel'],no_risk,block[block.Split=='test'],data['sentiment'],start,end,5,'threshold',block[block.Split=='validation'],signal_calendar=calendar)
 ablation[name]=metrics(book,name,'every_5_sessions')
 books[name]=book
 if args.output_dir: book.to_csv(args.output_dir/f'{name}.csv',index=False)
stock_returns=((data['panel'].loc[end,TICKERS]/data['panel'].loc[start,TICKERS]-1)*100).to_dict()
quality={}
for model,part in test.groupby('Model'):
 p=part.Predicted.to_numpy(float);a=part.Actual_5d_Pct.to_numpy(float)
 scores=[]
 for d,day in part.groupby('Signal_Date'):
  scores.append(float(day.Predicted.corr(day.Actual_5d_Pct,method='spearman')) if day.Predicted.std()>1e-12 and day.Actual_5d_Pct.std()>1e-12 else 0.0)
 stats=[]
 for ticker,stock in part.groupby('Ticker'):
  stats.append({'ticker':ticker,'pred_mean':float(stock.Predicted.mean()),'pred_std':float(stock.Predicted.std()),'actual_std':float(stock.Actual_5d_Pct.std()),'prediction_positive_rate':float((stock.Predicted>0).mean()),'actual_positive_rate':float((stock.Actual_5d_Pct>0).mean())})
 quality[model]={'mean_cross_section_spearman':float(np.mean(scores)),'positive_prediction_pct':float((p>0).mean()*100),'actual_positive_pct':float((a>0).mean()*100),'direction_accuracy_pct':float((np.sign(p)==np.sign(a)).mean()*100),'mae_pp':float(np.abs(p-a).mean()),'per_ticker':stats}
report={'period':[str(start.date()),str(end.date())],'accounting':accounting,'full_exposure_diagnostic_ablations':ablation,'stock_gross_price_returns_pct':stock_returns,'index_gross_price_return_pct':float((data['panel'].loc[end,MARKET]/data['panel'].loc[start,MARKET]-1)*100),'test_prediction_quality_descriptive_only':quality,'limits':['Accounting cost add-back does not recompute a no-cost path; share counts were affected by costs','Risk filter ablation is descriptive on observed test; it cannot be used as independent validation or a test-selected final strategy','Full equal vs fixed buy-hold separates rebalancing drift; broad index gap also includes stock universe/sector composition','Raw index with stock-cost proxy is not an investable ETF or total-return benchmark']}
index_roi=float(old_metrics.loc['theoretical_twii_with_stock_cost_proxy_not_an_etf','ROI_Pct'])
full_equal=float(ablation['full_equal_weekly']['ROI_Pct'])
risk_equal=float(old_metrics.loc['matched_risk_equal','ROI_Pct'])
news=float(old_metrics.loc['sentiment_risk_lstm','ROI_Pct'])
report['arithmetic_descriptive_decomposition']={
 'total_gap_pp':index_roi-news,
 'index_vs_full_weekly_equal_pp':index_roi-full_equal,
 'full_equal_vs_risk_equal_pp':full_equal-risk_equal,
 'risk_equal_vs_model_pp':risk_equal-news,
 'model_minus_risk_equal_net_pp':news-risk_equal,
 'model_minus_risk_equal_stock_pnl_pp':accounting['sentiment_risk_lstm']['pnl_contribution_pp']-accounting['matched_risk_equal']['pnl_contribution_pp'],
 'model_extra_cost_vs_risk_equal_pp':accounting['sentiment_risk_lstm']['cost_contribution_pp']-accounting['matched_risk_equal']['cost_contribution_pp'],
 'risk_off_rebalance_pct':accounting['sentiment_risk_lstm']['risk_off_at_rebalance']/accounting['sentiment_risk_lstm']['rebalance_days']*100}
assert abs(report['arithmetic_descriptive_decomposition']['total_gap_pp']-sum(report['arithmetic_descriptive_decomposition'][name] for name in ['index_vs_full_weekly_equal_pp','full_equal_vs_risk_equal_pp','risk_equal_vs_model_pp']))<1e-10
report['status']='posthoc_description_not_candidate_selection_or_independent_test'
report['source_base_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
report['source_files_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [base/'predictions.csv',base/'strategy_metrics.csv',base/'inputs/prices.csv',base/'inputs/sentiment.csv',ROOT/'experiments/v2/books.py',ROOT/'scripts/research_extension.py']}
report['diagnostic_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
text=json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
print(text)
if args.output_dir:(args.output_dir/'diagnostics.json').write_text(text,encoding='utf-8')
