import json
import hashlib
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from scripts import news_update
from scripts.news_update import to_signal_date, matches
from scripts.compare import compare
from scripts.common import current_completed_session


def test_signal_cutoff_and_weekend():
    assert str(to_signal_date('2026-10-27 13:30:00'))[:10]=='2026-10-27'
    assert str(to_signal_date('2026-10-27 13:30:01'))[:10]=='2026-10-28'
    assert str(to_signal_date('2026-10-30 16:00:00'))[:10]=='2026-11-02'
    assert str(to_signal_date('2026-10-31 10:00:00'))[:10]=='2026-11-02'


def test_news_company_filter():
    assert matches('台積電法說會','2330.TW')
    assert not matches('長榮航空宣布新航班','2603.TW')
    assert matches('長榮海運 2603 獲利','2603.TW')


def test_news_backfill_and_no_news(tmp_path,monkeypatch):
    monkeypatch.setattr(news_update,'ROOT',tmp_path)
    (tmp_path/'data').mkdir()
    input_rows=[{'Date':'2026-10-27','ticker':t,'Daily_Sentiment':0.3} for t in news_update.TICKERS]
    f=tmp_path/'data/sentiment_features_2026.csv'
    pd.DataFrame(input_rows).to_csv(f,index=False)
    def fake_fetch(day):
        if str(pd.Timestamp(day).date())=='2026-10-27':
            return [{'news_id':'article1','publish_time':'2026-10-27T14:00:00+08:00',
                     'title':'台積電', 'summary':'獲利提升'}]
        return []
    class FakeScore:
        def score(self,text):return .7
    news_update.update('2026-10-29',f,fetch=fake_fetch,scorer_factory=FakeScore)
    result=pd.read_csv(f)
    assert len(result)==15
    x=result[(result.Date=='2026-10-28')&(result.ticker=='2330.TW')].iloc[0]
    assert x.Daily_Sentiment==.7 and x.News_Count==1
    y=result[(result.Date=='2026-10-29')&(result.ticker=='2330.TW')].iloc[0]
    assert y.Daily_Sentiment==0 and y.Source_Status=='verified_no_matching_article'
    news_update.update('2026-10-29',f,fetch=fake_fetch,scorer_factory=FakeScore)
    assert len(pd.read_csv(tmp_path/'data/news_scored_cache.csv'))==1
    assert len(pd.read_csv(f))==15


def sample_result(p, id_, roi, testhash, pricehash):
    p.mkdir(parents=True)
    protocol={'protocol_version':'lstm-evaluation-v1','as_of_date':'2026-10-28','price_auto_adjust':False,
        'initial_capital':100000.,'broker_fee_rate':.001425,'sell_tax_rate':.003,
        'liquidate_at_end':True,'time_step':10,'test_days':2,'validation_days':60,
        'annual_trading_days':252,'risk_free_rate':0.,'direction_zero_tolerance_pct':1e-8,
        'test_target_dates_sha256':testhash,'price_panel_sha256':pricehash,
        'test_target_dates':['2026-10-27','2026-10-28'],
        'actual_split_audit':[{'Train_Start':'2021-04-01'}]}
    (p/'evaluation_protocol.json').write_text(json.dumps(protocol))
    pd.DataFrame([{'ROI_Pct':roi,'Sharpe':0.25,'MDD_Pct':-4.0,
                  'Direction_Accuracy_Pct':56.,'Test_Days':2,'As_Of_Date':'2026-10-28'}]).to_csv(p/'strategy_metrics.csv',index=False)
    dates=['2026-10-26','2026-10-27','2026-10-28']
    row={'Date':dates,'Market_Index':[100000.,101000.,102000.]}
    row['Portfolio_Net' if id_=='444' else 'NAV']=[100000,101000,100000+roi*1000]
    pd.DataFrame(row).to_csv(p/'backtest_nav.csv',index=False)
    pd.DataFrame([{'Sharpe':.5,'MDD_Pct':-2}]).to_csv(p/'market_metrics.csv',index=False)
    pd.DataFrame([{'Ticker':t,'Price_MAE':1.,'Price_RMSE':2.,'Return_MAE_pp':.5,
        'Return_RMSE_pp':.7,'Direction_Accuracy_Pct':55.} for t in news_update.TICKERS]).to_csv(p/'model_metrics.csv',index=False)


def test_compare_accepts_identical_protocol_and_rejects_mismatch(tmp_path):
    p=tmp_path/'444';q=tmp_path/'555'
    sample_result(p,'444',1.,'abc','same')
    sample_result(q,'555',3.,'abc','same')
    result=compare(p,q,tmp_path/'cmp')
    assert len(result)==3
    assert np.isnan(result.loc[0,'Trade_Win_Rate_Pct'])
    assert (tmp_path/'cmp/comparison.md').exists()
    protocol=json.loads((q/'evaluation_protocol.json').read_text())
    protocol['price_panel_sha256']='different'
    (q/'evaluation_protocol.json').write_text(json.dumps(protocol))
    with pytest.raises(ValueError,match='price_panel_sha256'):
        compare(p,q,tmp_path/'cmp')


def test_source_notebooks_separate_and_valid():
    import nbformat
    from scripts.common import ROOT
    for file in ['444_baseline.ipynb','555_multivariate.ipynb']:
        obj=nbformat.read(ROOT/'notebooks'/file,as_version=4)
        assert len(obj.cells)>25
        for c in obj.cells:
            if c.cell_type=='code' and not c.source.lstrip().startswith(('%pip','!pip','!')):
                compile(c.source,file,'exec')
