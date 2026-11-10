"""Shared trading dates and paths for 444 / 555."""
import hashlib
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TICKERS = ['2330.TW','2317.TW','2603.TW','2002.TW','2891.TW']
MARKET = '^TWII'
TZ = ZoneInfo('Asia/Taipei')

# Verified extraordinary exchange-wide closures; not ordinary missing Yahoo rows.
SPECIAL_MARKET_CLOSURES = {
    # TWSE's final pre-Lunar-New-Year session was Jan 26; reopen Feb 7.
    # XTAI's library calendar otherwise incorrectly includes Feb 4.
    # https://www.cna.com.tw/news/afe/202201260247.aspx
    '2022-02-04': 'TWSE Lunar New Year closure; reopen Feb 7',
    # TWSE 2023 final session was Jan 17; reopen Jan 30 (CNA quoting TWSE).
    # https://www.cna.com.tw/news/afe/202211150057.aspx
    '2023-01-18': 'TWSE Lunar New Year settlement closure after Jan 17 final session',
    # TWSE closure notice, reproduced by E.SUN Securities; TAIFEX also closed.
    # https://www.taifex.com.tw/cht/11/newsDetail?idx=15446&newsType=1
    '2024-10-31': 'Typhoon Kong-rey: TWSE closed',
    # https://www.taifex.com.tw/cht/11/newsDetail?newsType=1&idx=17137
    # TWSE notice reproduced by E.SUN: see docs/audit_130_critical.md.
    '2026-07-10': 'Typhoon Bavi: TWSE closed',
    # https://www.cna.com.tw/news/afe/202610220388.aspx (TWSE/TPEx notice)
    '2026-10-23': 'Typhoon Champi: TWSE closed',
}
SPECIAL_MARKET_CLOSURE_DATES = pd.DatetimeIndex(list(SPECIAL_MARKET_CLOSURES))

def sessions(start, end):
    import exchange_calendars as xcals
    a, b = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    cal = xcals.get_calendar('XTAI', start=a-pd.Timedelta(days=7), end=b+pd.Timedelta(days=7))
    days = pd.DatetimeIndex(cal.sessions_in_range(a, b))
    days = days.tz_localize(None).normalize() if days.tz is not None else days.normalize()
    return days.difference(SPECIAL_MARKET_CLOSURE_DATES)

def current_completed_session(now=None):
    now = now or datetime.now(TZ)
    if now.tzinfo is None:
        raise ValueError('clock must contain timezone')
    now = now.astimezone(TZ)
    upper = pd.Timestamp(now.date()) if (now.hour,now.minute) >= (15,0) else pd.Timestamp(now.date())-pd.Timedelta(days=1)
    found = sessions(upper-pd.Timedelta(days=60),upper)
    if not len(found):
        raise ValueError('找不到最近的已完成交易日')
    return found[-1]

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    tmp.replace(path)

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def check_date_in_calendar(value):
    day = pd.Timestamp(value).normalize()
    if day not in sessions(day-pd.Timedelta(days=7),day):
        raise ValueError(f'指定日期不是 XTAI 交易日：{day.date()}')
    if day > current_completed_session():
        raise ValueError(f'尚未收盤：{day.date()}')
    return day
