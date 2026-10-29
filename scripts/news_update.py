"""Verified CNYES -> news-level cache -> incremental sentiment features.

Historical seed rows are preserved. Every new session is complete ONLY after all
source calendar days have been fetched; no silent carry-forward of sentiment.
"""
from __future__ import annotations
import argparse
import html
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from scripts.common import ROOT,TICKERS,TZ,sessions,atomic_json,read_json

API='https://api.cnyes.com/media/api/v1/newslist/category/tw_stock'
MODEL_ID='IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment'
NAMES={'2330.TW':'台積電','2317.TW':'鴻海','2603.TW':'長榮','2002.TW':'中鋼','2891.TW':'中信金'}


def to_signal_date(when):
    t=pd.Timestamp(when)
    if t.tzinfo is None:t=t.tz_localize('Asia/Taipei')
    else:t=t.tz_convert('Asia/Taipei')
    day=pd.Timestamp(t.date())
    # An article after 13:30 is not available for the same day's signal.
    if (t.hour,t.minute,t.second)>(13,30,0):
        day+=pd.Timedelta(days=1)
    matches=sessions(day,day+pd.Timedelta(days=25))
    if not len(matches):raise RuntimeError('找不到後續台股交易日')
    return matches[0]


def matches(content,ticker):
    name=NAMES[ticker]
    if name not in content and ticker.split('.')[0] not in content:return False
    if ticker=='2603.TW':
        if any(s in content for s in ['長榮航','長榮航空','長榮航太','李長榮']):
            # Prevent entire article from being falsely labeled shipping if no
            # explicit stock code. This is conservative and may miss some cases.
            return ticker.split('.')[0] in content
    return True


def fetch_day(day, session=None):
    d=pd.Timestamp(day).date()
    start=datetime(d.year,d.month,d.day,tzinfo=TZ)
    end=start+timedelta(days=1)-timedelta(seconds=1)
    http=session or requests.Session()
    if session is None:
        retries=Retry(total=4,backoff_factor=2,status_forcelist=[429,500,502,503,504],
                      allowed_methods=['GET'])
        http.mount('https://',HTTPAdapter(max_retries=retries))
    articles=[]
    for page in range(1,41):
        res=http.get(API,params={'limit':50,'page':page,'startAt':int(start.timestamp()),
                                 'endAt':int(end.timestamp())},
                     headers={'User-Agent':'Mozilla/5.0 (compatible; university research project)'},timeout=40)
        res.raise_for_status()
        payload=res.json()
        obj=payload.get('items',{})
        data=obj.get('data') if isinstance(obj,dict) else obj
        if not isinstance(data,list):
            raise ValueError(f'鉅亨網回覆格式異常 {d},page={page}: items.data is not list')
        for item in data:
            stamp=item.get('publishAt')
            article_id=item.get('newsId',item.get('id'))
            if stamp is None or article_id is None:
                raise ValueError('新聞缺少可驗證的發布時間或文章 ID')
            published=datetime.fromtimestamp(int(stamp),TZ)
            if published.date()!=d:
                continue
            articles.append({'news_id':str(article_id),'publish_time':published.isoformat(),
                     'title':str(item.get('title') or ''),
                     'summary':html.unescape(str(item.get('summary') or ''))})
        if not data or len(data)<50:break
        if page==40:raise RuntimeError(f'{d} 超過 2000 則新聞，已停止避免不完整資料')
        time.sleep(.15)
    return list({r['news_id']:r for r in articles}.values())

class SentimentScorer:
    def __init__(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.torch=torch
        self.tokenizer=AutoTokenizer.from_pretrained(MODEL_ID)
        self.model=AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval().cpu()
        labels=self.model.config.id2label
        if str(labels.get(0,'')).lower()!='negative' or str(labels.get(1,'')).lower()!='positive':
            raise ValueError(f'模型標籤不可確認，不能假設 0=negative 1=positive: {labels}')
    def score(self,text):
        torch=self.torch
        input_data=self.tokenizer(text,return_tensors='pt',truncation=True,max_length=256)
        with torch.inference_mode():
            probs=torch.softmax(self.model(**input_data).logits,dim=-1)[0]
        return round(float(probs[1]-probs[0]),4)


def update(as_of,feature_path=None,fetch=fetch_day,scorer_factory=SentimentScorer):
    feature_path=Path(feature_path or ROOT/'data/sentiment_features_2026.csv')
    if not feature_path.exists():raise FileNotFoundError(feature_path)
    state_file=ROOT/'data/news_update_state.json'
    cache_file=ROOT/'data/news_scored_cache.csv'
    df=pd.read_csv(feature_path,dtype={'ticker':'string'})
    if not {'Date','ticker','Daily_Sentiment'}.issubset(df):
        raise ValueError('歷史情緒檔欄位不完整')
    df['Date']=pd.to_datetime(df.Date).dt.normalize()
    as_of=pd.Timestamp(as_of).normalize()
    existing_state=read_json(state_file) if state_file.exists() else None
    if existing_state:
        seed_end=pd.Timestamp(existing_state['seed_end_date'])
        previous=pd.Timestamp(existing_state['last_complete_scan_day'])
    else:
        seed_end=df.Date.max()
        previous=seed_end
    if as_of<seed_end:
        raise ValueError('執行日期早於歷史資料截止日，不得覆蓋歷史資料')
    if existing_state and as_of<previous:
        raise ValueError('指定日早於已完成增量掃描日期，拒絕倒寫新聞資料')
    start=previous-pd.Timedelta(days=1)
    if start>as_of:start=as_of
    scanned=[]
    for date in pd.date_range(start,as_of,freq='D'):
        day=date.strftime('%Y-%m-%d')
        entries=fetch(date)
        print(f'[NEWS] {day}: fetched {len(entries)} articles',flush=True)
        scanned.extend(entries)
    old=pd.read_csv(cache_file,dtype={'news_id':str}) if cache_file.exists() else pd.DataFrame()
    already=set(zip(old.news_id.astype(str),old.ticker)) if not old.empty else set()
    new=[]
    for a in scanned:
        signal=to_signal_date(a['publish_time'])
        if signal<=seed_end:continue  # prevent replacing un-audited historical scores
        content=a['title']+' '+a['summary']
        for ticker in TICKERS:
            key=(a['news_id'],ticker)
            if key in already or not matches(content,ticker):continue
            new.append({'news_id':a['news_id'],'ticker':ticker,
                        'publish_time':a['publish_time'],'Date':str(signal.date()),
                        'text':content,'_key':key})
            already.add(key)
    scorer=scorer_factory() if new else None
    if new:
        # score once per article even if it applies to multiple companies
        scores={}
        for r in new:
            if r['news_id'] not in scores:
                scores[r['news_id']]=scorer.score(r['text'][:1600])
            r['Daily_Sentiment']=scores[r['news_id']]
            r['Model_ID']=MODEL_ID
        new_df=pd.DataFrame([{k:v for k,v in r.items() if k not in ('text','_key')} for r in new])
        old=pd.concat([old,new_df],ignore_index=True)
    # The full article score cache is not rewritten until the entire scan succeeds.
    if not old.empty:
        old=old.drop_duplicates(['news_id','ticker']).sort_values(['Date','ticker','news_id'])
    sessions_to_complete=sessions(seed_end+pd.Timedelta(days=1),as_of)
    rowlist=[]
    for d in sessions_to_complete:
        for ticker in TICKERS:
            part=old.loc[(old.Date.astype(str)==str(d.date()))&(old.ticker==ticker)] if not old.empty else pd.DataFrame()
            latest=part['publish_time'].max() if len(part) else None
            rowlist.append({'Date':d,'ticker':ticker,
               'Daily_Sentiment':float(part.Daily_Sentiment.astype(float).mean()) if len(part) else 0.,
               'News_Count':len(part),'Latest_News_Timestamp':latest,
               'Source_Status':'observed_article' if len(part) else 'verified_no_matching_article',
               'Scoring_Model':MODEL_ID if len(part) else 'neutral_no_news'})
    legacy=df[df.Date<=seed_end].copy()
    built=pd.DataFrame(rowlist)
    result=pd.concat([legacy,built],ignore_index=True)
    result=result.sort_values(['Date','ticker']).reset_index(drop=True)
    if result.duplicated(['Date','ticker']).any():raise ValueError('情緒包含重複 Date/ticker')
    for d in sessions_to_complete:
        if set(result.loc[result.Date==d,'ticker'])!=set(TICKERS):raise ValueError('缺少股票訊號日')
    if not result.Daily_Sentiment.between(-1,1).all():raise ValueError('情緒分數超出合法範圍')
    # Save after validation; errors never update state.
    feature_path.parent.mkdir(parents=True,exist_ok=True)
    tmp=feature_path.with_suffix('.tmp')
    result.to_csv(tmp,index=False,encoding='utf-8-sig',date_format='%Y-%m-%d')
    tmp.replace(feature_path)
    if not old.empty:
        cache_file.parent.mkdir(parents=True,exist_ok=True)
        tmp=cache_file.with_suffix('.tmp');old.to_csv(tmp,index=False,encoding='utf-8-sig');tmp.replace(cache_file)
    atomic_json(state_file,{'seed_end_date':str(seed_end.date()),
       'last_complete_scan_day':str(as_of.date()),'articles_cache':len(old),
       'note':'earlier historical sentiment seeds not recalculated; new rows scored by Erlangshen sentiment'})
    print(f'[NEWS] verified sentiment through {as_of.date()}, new articles {len(new)}',flush=True)
    return feature_path

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--as-of',required=True)
    args=p.parse_args();update(args.as_of)
