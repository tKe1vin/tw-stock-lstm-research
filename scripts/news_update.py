"""Verified CNYES -> news-level cache -> incremental sentiment features.

Historical seed rows are preserved. Every new session is complete ONLY after all
source calendar days have been fetched; no silent carry-forward of sentiment.

The adjacent ``*_provenance.jsonl`` is append-only (new records use schema 2).
Schema 1 observations retain their original, weaker evidence. Article observations
record the locally observed API title/summary version (not full article text),
first_seen_at, response fetched_at, content/raw-payload hashes and source revision.
Scored rows additionally record available_at after scoring and cache_row_hash of
the six CACHE_COLUMNS. A later observation never certifies availability for a
legacy score: load_news_cache marks rows lacking matching scored evidence as
retrospective_unverified and leaves their acquisition timestamps empty.
"""
from __future__ import annotations
import argparse
import hashlib
import html
import json
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
from urllib.parse import urlsplit
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from scripts.common import ROOT,TICKERS,TZ,sessions,atomic_json,read_json

API='https://api.cnyes.com/media/api/v1/newslist/category/tw_stock'
MODEL_ID='IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment'
NAMES={'2330.TW':'台積電','2317.TW':'鴻海','2603.TW':'長榮','2002.TW':'中鋼','2891.TW':'中信金'}
CACHE_COLUMNS = ('news_id', 'ticker', 'publish_time', 'Date', 'Daily_Sentiment', 'Model_ID')
PROVENANCE_FIELDS = ('first_seen_at', 'fetched_at', 'available_at', 'content_sha256',
                     'content_scope', 'raw_payload_sha256', 'source_revision',
                     'source_revision_kind', 'cache_row_hash', 'provenance_status',
                     'model_revision', 'tokenizer_revision', 'scoring_config',
                     'runtime_versions', 'scoring_provenance_status')
SCORING_CONFIG = {'text_scope': 'api_title_summary', 'max_characters': 1600,
                  'max_tokens': 256, 'formula': 'P(positive)-P(negative)',
                  'round_decimals': 4}


def scorer_provenance(scorer):
    """Never infer a checkpoint from a mutable model name or a later download."""
    evidence = scorer.provenance() if callable(getattr(scorer, 'provenance', None)) else {}
    if not isinstance(evidence, dict):
        raise ValueError('新聞評分器 provenance 必須是具名欄位')
    evidence = dict(evidence)
    model = evidence.get('model_revision')
    tokenizer = evidence.get('tokenizer_revision')
    versions = evidence.get('runtime_versions')
    resolved = (isinstance(model, str) and bool(re.fullmatch(r'[0-9a-f]{40}', model))
                and tokenizer == model and evidence.get('scoring_config') == SCORING_CONFIG
                and isinstance(versions, dict)
                and all(isinstance(versions.get(name), str) and bool(versions[name].strip())
                        for name in ('torch', 'transformers')))
    evidence['scoring_provenance_status'] = ('resolved_checkpoint_recorded' if resolved
                                             else 'checkpoint_unverified_custom_scorer')
    return evidence


def validate_scored_provenance(record):
    """Fail closed for malformed matching evidence; hashes are integrity, not attestation."""
    if record.get('source') != API or record.get('schema_version') not in (1, 2):
        raise ValueError('新聞來源或 provenance schema 無法確認')
    if (not set(CACHE_COLUMNS).issubset(record) or record.get('ticker') not in TICKERS
            or not str(record.get('news_id') or '').strip() or record.get('Model_ID') != MODEL_ID):
        raise ValueError('新聞 scored provenance 識別欄位不完整或評分模型不一致')
    try:
        score = float(record['Daily_Sentiment'])
    except (TypeError, ValueError) as exc:
        raise ValueError('新聞情緒分數必須是有限數值') from exc
    if not np.isfinite(score) or not -1 <= score <= 1:
        raise ValueError('新聞情緒分數必須是 -1 到 1 的有限數值')
    if record.get('cache_row_hash') != cache_row_hash(record):
        raise ValueError('新聞 scored provenance 與快取內容雜湊不一致')
    for field in ('content_sha256', 'raw_payload_sha256'):
        if not re.fullmatch(r'[0-9a-f]{64}', str(record.get(field, ''))):
            raise ValueError('新聞 provenance 缺少合法內容雜湊')
    if record.get('content_scope') != 'api_title_summary':
        raise ValueError('新聞內容範圍無法確認')
    kind = record.get('source_revision_kind')
    if kind not in ('provider_revision', 'observed_payload_hash') or not record.get('source_revision'):
        raise ValueError('新聞版本證據無法確認')
    if kind == 'observed_payload_hash' and record['source_revision'] != 'sha256:' + record['raw_payload_sha256']:
        raise ValueError('新聞來源版本與 payload 雜湊不一致')
    try:
        times = [pd.Timestamp(record.get(key)) for key in
                 ('publish_time', 'first_seen_at', 'fetched_at', 'available_at')]
    except (TypeError, ValueError) as exc:
        raise ValueError('新聞 provenance 時間格式無法確認') from exc
    if any(pd.isna(value) or value.tzinfo is None for value in times):
        raise ValueError('新聞 provenance 時間缺漏或沒有時區')
    if not all(left <= right for left, right in zip(times, times[1:])):
        raise ValueError('新聞發布、初見、取得及評分完成時間順序不正確')
    if str(pd.Timestamp(record['Date']).date()) != str(to_signal_date(times[0]).date()):
        raise ValueError('新聞訊號交易日與發布時間不一致')
    if record.get('scoring_provenance_status') == 'resolved_checkpoint_recorded':
        if scorer_provenance(type('RecordedScorer', (), {'provenance': lambda self: record})())[
                'scoring_provenance_status'] != 'resolved_checkpoint_recorded':
            raise ValueError('新聞 checkpoint 證據不完整')
    elif record.get('schema_version') == 2 and record.get('scoring_provenance_status') not in (
            'checkpoint_unverified_custom_scorer', 'checkpoint_unverified_legacy'):
        raise ValueError('新聞評分 checkpoint 驗證狀態不明')


def observation_time(clock=None):
    """Read an aware completion clock; a run's start time is not a fetch time."""
    value = pd.Timestamp(clock() if clock else datetime.now(TZ))
    if value.tzinfo is None:
        raise ValueError('新聞觀測時鐘必須包含時區')
    return value.tz_convert(TZ).isoformat()


def _json_hash(value):
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), default=str)
    return hashlib.sha256(body.encode('utf-8')).hexdigest()


def cache_row_hash(row):
    """Hash the six immutable scored-cache fields, independently of CSV formatting."""
    payload = {key: str(row[key]) for key in CACHE_COLUMNS if key != 'Daily_Sentiment'}
    payload['Daily_Sentiment'] = float(row['Daily_Sentiment'])
    return _json_hash(payload)


def provenance_path(cache_file):
    cache_file = Path(cache_file)
    return cache_file.with_name(cache_file.stem + '_provenance.jsonl')


def article_provenance(article, fetched_at):
    """Describe the observed API title/summary version, never an unavailable full text."""
    content_hash = _json_hash({'title': article['title'], 'summary': article['summary']})
    raw_hash = article.get('raw_payload_sha256') or content_hash
    first_seen = article.get('first_seen_at') or fetched_at
    fetched = article.get('fetched_at') or fetched_at
    published = pd.Timestamp(article['publish_time'])
    observed = pd.Timestamp(first_seen)
    acquired = pd.Timestamp(fetched)
    if observed.tzinfo is None or acquired.tzinfo is None or observed > acquired:
        raise ValueError('新聞觀測與取得時間必須帶時區且依時序記錄')
    if published.tzinfo is None or published > observed:
        raise ValueError('新聞發布時間晚於取得時間或缺少時區')
    return {
        'first_seen_at': first_seen, 'fetched_at': fetched,
        'content_sha256': content_hash, 'content_scope': 'api_title_summary',
        'raw_payload_sha256': raw_hash,
        'source_revision': str(article.get('source_revision') or ('sha256:' + raw_hash)),
        'source_revision_kind': article.get('source_revision_kind') or 'observed_payload_hash',
    }


def append_provenance(cache_file, records):
    """Append observed versions; legacy rows acquire no invented historical timestamp."""
    if not records:
        return
    path = provenance_path(cache_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    known = set()
    first_observed = {}
    def record_key(row):
        return (row.get('record_type'), row.get('news_id'), row.get('ticker'),
                row.get('content_sha256'), row.get('source_revision'),
                row.get('raw_payload_sha256'), row.get('cache_row_hash'))
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            known.add(record_key(row))
            version = (row.get('news_id'), row.get('content_sha256'), row.get('source_revision'))
            if row.get('first_seen_at'):
                first_observed.setdefault(version, row['first_seen_at'])
    pending = []
    for record in records:
        record = {'schema_version': 2, 'source': API, **record}
        if record.get('record_type') == 'scored_cache_row':
            if not record.get('scoring_provenance_status'):
                record['scoring_provenance_status'] = 'checkpoint_unverified_custom_scorer'
        version = (record.get('news_id'), record.get('content_sha256'), record.get('source_revision'))
        if record.get('first_seen_at'):
            record['first_seen_at'] = first_observed.setdefault(version, record['first_seen_at'])
        key = record_key(record)
        if key not in known:
            if record.get('record_type') == 'scored_cache_row':
                validate_scored_provenance(record)
            pending.append(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')
            known.add(key)
    # Validate the whole batch before appending any bytes.
    if pending:
        with path.open('a', encoding='utf-8') as handle:
            handle.writelines(pending)


def load_news_cache(cache_file):
    """Join provenance without rewriting legacy cache bytes or claiming old availability."""
    path = Path(cache_file)
    if not path.exists():
        return pd.DataFrame(columns=[*CACHE_COLUMNS, *PROVENANCE_FIELDS])
    frame = pd.read_csv(path, dtype={'news_id': str})
    metadata = {}
    sidecar = provenance_path(path)
    if sidecar.exists():
        for line in sidecar.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            if row.get('record_type') == 'scored_cache_row':
                validate_scored_provenance(row)
                metadata.setdefault(row['cache_row_hash'], row)
    joined = []
    for row in frame.to_dict('records'):
        digest = cache_row_hash(row)
        evidence = metadata.get(digest)
        row.update({field: evidence.get(field) if evidence else None for field in PROVENANCE_FIELDS})
        row['cache_row_hash'] = digest
        row['provenance_status'] = 'observed_at_recorded_time' if evidence else 'retrospective_unverified'
        row['scoring_provenance_status'] = (evidence.get('scoring_provenance_status', 'checkpoint_unverified_legacy')
                                            if evidence else 'checkpoint_unverified_legacy')
        joined.append(row)
    return pd.DataFrame(joined, columns=[*CACHE_COLUMNS, *PROVENANCE_FIELDS])


def append_news_cache(cache_file, new_rows):
    """Preserve every pre-existing byte. Extra evidence lives in the versioned sidecar."""
    if not len(new_rows):
        return
    path = Path(cache_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(pd.read_csv(path, nrows=0).columns) if path.exists() else list(CACHE_COLUMNS)
    original = path.read_bytes() if path.exists() else b''
    body = new_rows.reindex(columns=columns).to_csv(index=False, header=not bool(original), lineterminator='\r\n')
    if original and not original.endswith(b'\n'):
        original += b'\r\n'
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_bytes(original + body.encode('utf-8-sig' if not original else 'utf-8'))
    temporary.replace(path)


def to_signal_date(when):
    t=pd.Timestamp(when)
    if t.tzinfo is None:t=t.tz_localize('Asia/Taipei')
    else:t=t.tz_convert('Asia/Taipei')
    day=pd.Timestamp(t.date())
    # An article after 13:30 is not available for the same day's signal.
    if t > day.tz_localize(TZ) + pd.Timedelta(hours=13, minutes=30):
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


def fetch_day(day, session=None, *, clock=None):
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
                     headers={'User-Agent':'Mozilla/5.0 (compatible; university research project)'},
                     timeout=(10,40), allow_redirects=False)
        # A fixed public endpoint may not redirect credentials, queries or bodies
        # to an unreviewed provider. Do not log provider response bodies/headers.
        if 300 <= getattr(res, 'status_code', 200) < 400:
            raise ValueError('新聞來源要求重新導向，拒絕未審核的來源變更')
        response_origin = urlsplit(getattr(res, 'url', API))
        if response_origin.scheme != 'https' or response_origin.netloc != urlsplit(API).netloc:
            raise ValueError('新聞回覆來源不在允許清單')
        res.raise_for_status()
        payload=res.json()
        fetched_at = observation_time(clock)
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
            revision = next((item[key] for key in ('updatedAt', 'updateAt', 'modifiedAt', 'revision')
                             if item.get(key) is not None), None)
            raw_hash = _json_hash(item)
            articles.append({'news_id':str(article_id),'publish_time':published.isoformat(),
                     'title':str(item.get('title') or ''),
                     'summary':html.unescape(str(item.get('summary') or '')),
                     'first_seen_at':fetched_at, 'fetched_at':fetched_at,
                     'raw_payload_sha256':raw_hash,
                     'source_revision':str(revision) if revision is not None else 'sha256:' + raw_hash,
                     'source_revision_kind':'provider_revision' if revision is not None else 'observed_payload_hash'})
        if not data or len(data)<50:break
        if page==40:raise RuntimeError(f'{d} 超過 2000 則新聞，已停止避免不完整資料')
        time.sleep(.15)
    return list({r['news_id']:r for r in articles}.values())

class SentimentScorer:
    def __init__(self, revision=None):
        import torch
        import transformers
        from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
        self.torch=torch
        config=AutoConfig.from_pretrained(MODEL_ID,revision=revision or 'main',trust_remote_code=False)
        resolved=getattr(config,'_commit_hash',None)
        if not isinstance(resolved,str) or not re.fullmatch(r'[0-9a-f]{40}',resolved):
            raise ValueError('無法確認新聞模型的確切 Hugging Face checkpoint，停止評分')
        if revision and re.fullmatch(r'[0-9a-f]{40}',revision) and revision != resolved:
            raise ValueError('新聞模型版本與指定 checkpoint 不一致')
        # Resolve once; tokenizer and weights must refer to the same immutable revision.
        self.tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,revision=resolved,trust_remote_code=False)
        self.model=AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID,revision=resolved,trust_remote_code=False).eval().cpu()
        if getattr(self.model.config,'_commit_hash',None) != resolved:
            raise ValueError('新聞權重與已解析的 checkpoint 不一致')
        self._provenance={'model_revision':resolved,'tokenizer_revision':resolved,
            'scoring_config':dict(SCORING_CONFIG),
            'runtime_versions':{'torch':torch.__version__,'transformers':transformers.__version__}}
        labels=self.model.config.id2label
        if str(labels.get(0,'')).lower()!='negative' or str(labels.get(1,'')).lower()!='positive':
            raise ValueError(f'模型標籤不可確認，不能假設 0=negative 1=positive: {labels}')
    def provenance(self):
        return {**self._provenance, 'scoring_config': dict(self._provenance['scoring_config']),
                'runtime_versions': dict(self._provenance['runtime_versions'])}
    def score(self,text):
        torch=self.torch
        input_data=self.tokenizer(text,return_tensors='pt',truncation=True,max_length=256)
        with torch.inference_mode():
            probs=torch.softmax(self.model(**input_data).logits,dim=-1)[0]
        return round(float(probs[1]-probs[0]),4)


def update(as_of,feature_path=None,fetch=fetch_day,scorer_factory=SentimentScorer, *, clock=None):
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
        fetched_at = observation_time(clock)
        scanned.extend([{**entry, **article_provenance(entry, fetched_at)} for entry in entries])
    old=load_news_cache(cache_file)
    already=set(zip(old.news_id.astype(str),old.ticker)) if not old.empty else set()
    new=[]
    observations=[]
    for a in scanned:
        signal=to_signal_date(a['publish_time'])
        if signal<=seed_end:continue  # prevent replacing un-audited historical scores
        content=a['title']+' '+a['summary']
        for ticker in TICKERS:
            key=(a['news_id'],ticker)
            if not matches(content,ticker):continue
            observations.append({'record_type':'article_observation','news_id':a['news_id'],
                'ticker':ticker,'Date':str(signal.date()),'publish_time':a['publish_time'],
                **{field:a.get(field) for field in PROVENANCE_FIELDS if field not in ('available_at','cache_row_hash','provenance_status')}})
            if key in already:continue
            new.append({'news_id':a['news_id'],'ticker':ticker,
                        'publish_time':a['publish_time'],'Date':str(signal.date()),
                        'text':content,'_key':key,
                        **{field:a.get(field) for field in PROVENANCE_FIELDS if field not in ('available_at','cache_row_hash','provenance_status')}})
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
            r.update(scorer_provenance(scorer))
            r['available_at']=observation_time(clock)
            r['cache_row_hash']=cache_row_hash(r)
            r['provenance_status']='observed_at_recorded_time'
            observations.append({'record_type':'scored_cache_row',
                **{k:v for k,v in r.items() if k not in ('text','_key')}})
        new_df=pd.DataFrame([{k:v for k,v in r.items() if k not in ('text','_key')} for r in new])
        old=pd.concat([old,new_df],ignore_index=True) if len(old) else new_df.copy()
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
    append_provenance(cache_file, observations)
    if new:
        append_news_cache(cache_file, new_df)
    atomic_json(state_file,{'seed_end_date':str(seed_end.date()),
       'last_complete_scan_day':str(as_of.date()),'articles_cache':len(old),
       'completed_at':observation_time(clock),
       'news_provenance_path':str(provenance_path(cache_file)),
       'historical_provenance_status':'retrospective_unverified',
       'note':'earlier historical sentiment seeds not recalculated; new rows scored by Erlangshen sentiment'})
    print(f'[NEWS] verified sentiment through {as_of.date()}, new articles {len(new)}',flush=True)
    return feature_path

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--as-of',required=True)
    args=p.parse_args();update(args.as_of)
