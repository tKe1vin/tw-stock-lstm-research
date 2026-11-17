"""Read-only news archive: scores, observed versions and source coverage.

A title obtained for display later never authenticates an old scoring input.
No provider client or model runtime is imported by this module.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GITHUB = "https://github.com/tKe1vin/tw-stock-lstm-research/blob/main/"
MARKET = "experiments/results/v2/v211/market"
CACHE_COLUMNS = ("news_id", "ticker", "publish_time", "Date", "Daily_Sentiment", "Model_ID")
NAMES = {"2330.TW": "台積電", "2317.TW": "鴻海", "2603.TW": "長榮", "2002.TW": "中鋼", "2891.TW": "中信金"}
LEGACY = "缺取得／評分完成紀錄"
LATE = "晚於訊號日取得；不可回填當日可用性"
OBSERVED = "有取得紀錄；仍需與實際決策時點比對"


class NewsEvidenceError(ValueError):
    """A retained source is unreadable or its matching score evidence changed."""


def _path(root, relative):
    root = Path(root).resolve()
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise NewsEvidenceError("新聞來源路徑超出repository")
    return result


def _json(root, relative):
    path = _path(root, relative)
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (ValueError, OSError) as exc:
        raise NewsEvidenceError(f"新聞來源JSON無法讀取：{relative}") from exc
    if not isinstance(value, dict):
        raise NewsEvidenceError(f"新聞來源JSON格式不符：{relative}")
    return value


def _jsonl(root, relative):
    path = _path(root, relative)
    if not path.is_file():
        return []
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    except (ValueError, OSError) as exc:
        raise NewsEvidenceError(f"新聞來源JSONL無法讀取：{relative}") from exc
    if any(not isinstance(row, dict) for row in rows):
        raise NewsEvidenceError(f"新聞來源JSONL格式不符：{relative}")
    return rows


def _row_hash(row):
    payload = {key: str(row[key]) for key in CACHE_COLUMNS if key != "Daily_Sentiment"}
    payload["Daily_Sentiment"] = float(row["Daily_Sentiment"])
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def safe_source_url(value):
    """Keep links explicit and exclude scripts, credentials and malformed URLs."""
    if not isinstance(value, str):
        return ""
    try:
        parsed = urlsplit(value)
    except ValueError:
        return ""
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
        return ""
    return value if not any(ord(char) < 32 for char in value) else ""


def _check_evidence(row):
    try:
        valid_hash = row.get("cache_row_hash") == _row_hash(row)
        score = float(row["Daily_Sentiment"])
        times = [pd.Timestamp(row.get(key)) for key in ("publish_time", "first_seen_at", "fetched_at", "available_at")]
        valid_times = all(not pd.isna(time) and time.tzinfo is not None for time in times)
        valid_times = valid_times and all(left <= right for left, right in zip(times, times[1:]))
    except (KeyError, ValueError, TypeError) as exc:
        raise NewsEvidenceError("新聞評分來源欄位不完整") from exc
    if not valid_hash or not np.isfinite(score) or not -1 <= score <= 1 or not valid_times:
        raise NewsEvidenceError("新聞評分來源雜湊、分數或時間順序不符")


def _timing(row):
    available = row.get("available_at")
    if not available:
        return LEGACY
    when = pd.Timestamp(available).tz_convert("Asia/Taipei")
    return LATE if str(when.date()) > str(row["Date"])[:10] else OBSERVED


def load_news_archive(root=ROOT):
    """Load immutable scored rows and matching evidence; no enrichment by network."""
    records, sources, warnings = {}, [], []
    for relative in ("data/news_scored_cache.csv", f"{MARKET}/news_cache.csv"):
        path = _path(root, relative)
        if not path.is_file():
            continue
        sources.append(relative)
        try:
            frame = pd.read_csv(path, dtype={"news_id": str}, keep_default_na=False)
        except (ValueError, OSError) as exc:
            raise NewsEvidenceError(f"新聞快取無法讀取：{relative}") from exc
        if not set(CACHE_COLUMNS).issubset(frame):
            raise NewsEvidenceError(f"新聞快取缺欄位：{relative}")
        provenance_file = str(Path(relative).with_name(path.stem + "_provenance.jsonl"))
        evidence = {}
        if _path(root, provenance_file).is_file():
            sources.append(provenance_file)
        for observed in _jsonl(root, provenance_file):
            if observed.get("record_type") == "scored_cache_row":
                _check_evidence(observed)
                evidence.setdefault(observed["cache_row_hash"], observed)
        for original in frame.to_dict("records"):
            try:
                score = float(original["Daily_Sentiment"]) if str(original["Daily_Sentiment"]).strip() else np.nan
                original["Daily_Sentiment"] = score
                digest = _row_hash(original)
                day = pd.Timestamp(original["Date"])
            except (ValueError, TypeError) as exc:
                raise NewsEvidenceError("新聞快取含不合法的日期或分數") from exc
            if pd.isna(day) or (not np.isnan(score) and (not np.isfinite(score) or not -1 <= score <= 1)):
                raise NewsEvidenceError("新聞快取含不合法的日期或分數")
            observed = evidence.get(digest, {})
            row = {key: original[key] for key in CACHE_COLUMNS}
            row.update({key: observed.get(key) for key in (
                "first_seen_at", "fetched_at", "available_at", "content_sha256", "source_revision",
                "model_revision", "tokenizer_revision", "scoring_provenance_status")})
            row.update(cache_row_hash=digest, provenance_status="observed_at_recorded_time" if observed else "retrospective_unverified",
                       title="", title_observed_at=None, title_scope="原評分快取未保存標題",
                       source_url=f"https://news.cnyes.com/news/id/{original['news_id']}" if str(original["news_id"]).isdigit() else "",
                       cache_sources=relative)
            row["timing_status"] = _timing(row)
            prior = records.get(digest)
            if prior:
                row["cache_sources"] = prior["cache_sources"] + "；" + relative
                if prior["available_at"] and not row["available_at"]:
                    prior["cache_sources"] = row["cache_sources"]
                    continue
            records[digest] = row
    metadata_file = "data/news_display_metadata.jsonl"
    metadata = _jsonl(root, metadata_file)
    if metadata:
        sources.append(metadata_file)
    by_id = {}
    for item in metadata:
        identity = str(item.get("news_id", ""))
        if identity and isinstance(item.get("title"), str) and item["title"].strip():
            by_id[identity] = item
    for row in records.values():
        item = by_id.get(str(row["news_id"]))
        if item:
            row.update(title=item["title"], title_observed_at=item.get("observed_at"),
                       title_scope="另行取得的展示標題；不證明原評分輸入版本")
            row["source_url"] = safe_source_url(item.get("url")) or row["source_url"]
    articles = pd.DataFrame(list(records.values()))
    if not articles.empty:
        articles = articles.sort_values(["Date", "publish_time", "news_id", "ticker"], ascending=False).reset_index(drop=True)
    coverage_file = f"{MARKET}/news_scan_coverage.json"
    coverage = _json(root, coverage_file)
    if coverage:
        sources.append(coverage_file)
    calendar = pd.DataFrame([{"來源日": day, "掃描文章數": info.get("articles"), "真正掃描完成時間": info.get("completed_at"),
                              "紀錄類型": info.get("timestamp_kind", "舊格式紀錄")}
                             for day, info in sorted(coverage.get("calendar_days", {}).items())])
    sessions = pd.DataFrame([{"訊號日": day, "來源起日": info.get("source_start"), "來源迄日": info.get("source_end"),
                              "評分可用時間": info.get("available_at") or info.get("completed_at"),
                              "歷史可用性標記": info.get("provenance_status", "未提供")}
                             for day, info in sorted(coverage.get("signal_sessions", {}).items())])
    daily = pd.DataFrame()
    for relative in (f"{MARKET}/sentiment.csv", "data/sentiment_features_2026.csv"):
        path = _path(root, relative)
        if path.is_file():
            daily = pd.read_csv(path)
            if not {"Date", "ticker", "Daily_Sentiment"}.issubset(daily):
                raise NewsEvidenceError("每日新聞情緒表缺少必要欄位")
            sources.append(relative)
            break
    if len(articles) and not articles.title.astype(bool).any():
        warnings.append("原快取未保存新聞標題或頁面截圖；目前可查文章ID、原文連結、情緒及來源時點。")
    if not coverage:
        warnings.append("沒有完整掃描紀錄，不能把缺資料當成已確認無新聞。")
    return {"articles": articles, "calendar": calendar, "sessions": sessions, "daily": daily,
            "sources": sources, "warnings": warnings, "has_page_screenshots": False}


def filter_articles(frame, *, day=None, tickers=(), keyword="", score_kind="全部"):
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    if day and day != "全部日期":
        result = result[result.Date.astype(str).str[:10].eq(str(day))]
    if tickers:
        result = result[result.ticker.isin(tickers)]
    if keyword.strip():
        search = result[["title", "news_id", "ticker"]].fillna("").astype(str).agg(" ".join, axis=1)
        result = result[search.str.contains(keyword.strip(), case=False, regex=False)]
    score = pd.to_numeric(result.Daily_Sentiment, errors="coerce")
    if score_kind == "正向":
        result = result[score.gt(0)]
    elif score_kind == "負向":
        result = result[score.lt(0)]
    elif score_kind == "中性":
        result = result[score.eq(0)]
    elif score_kind == "缺評分":
        result = result[score.isna()]
    return result.reset_index(drop=True)


def safe_csv(frame):
    """Prevent metadata text from becoming spreadsheet formulas on download."""
    frame = frame.copy()
    for column in frame.select_dtypes(include=["object", "string"]).columns:
        frame[column] = frame[column].map(lambda value: "'" + value if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")) else value)
    return frame.to_csv(index=False).encode("utf-8-sig")


def render_news(root=ROOT):
    import streamlit as st

    st.header("新聞與情緒資料庫")
    st.caption("此頁只讀repository保存的文章評分與來源紀錄；不會重新抓取、重算情緒或回填歷史訊號。")
    try:
        archive = load_news_archive(root)
    except (NewsEvidenceError, OSError, ValueError) as exc:
        st.error("新聞來源檢查未通過，暫停展示新聞。")
        st.code(str(exc), language=None)
        return
    frame = archive["articles"]
    for message in archive["warnings"]:
        st.info(message)
    if frame.empty:
        st.info("此checkout沒有逐篇新聞評分快取。")
    else:
        a, b, c = st.columns(3)
        a.metric("保存文章ID", frame.news_id.nunique())
        b.metric("股別評分列", len(frame))
        c.metric("有取得／評分完成紀錄", int(frame.available_at.notna().sum()))
        st.caption("同一新聞可對應多檔股票。取得紀錄只證明實際觀測時間，不能證明較早決策時已可用；缺時點的舊分數仍為歷史未驗證。")
        dates = sorted(frame.Date.astype(str).str[:10].unique(), reverse=True)
        left, right = st.columns(2)
        day = left.selectbox("新聞訊號日", ["全部日期", *dates], key="news_day")
        tickers = right.multiselect("股票", sorted(frame.ticker.unique()), format_func=lambda x: f"{x} {NAMES.get(x, '')}", key="news_tickers")
        left, right = st.columns([2, 1])
        keyword = left.text_input("搜尋已保存標題／文章ID／股票", key="news_keyword")
        kind = right.selectbox("情緒篩選", ["全部", "正向", "負向", "中性", "缺評分"], key="news_score_kind")
        shown = filter_articles(frame, day=day, tickers=tickers, keyword=keyword, score_kind=kind)
        st.write(f"符合條件：{len(shown)} 筆股別評分列")
        names = {"Date": "訊號日", "ticker": "股票", "news_id": "文章ID", "title": "展示標題", "publish_time": "發布時間",
                 "Daily_Sentiment": "情緒分數", "first_seen_at": "首次觀測時間", "available_at": "評分可用時間",
                 "timing_status": "時點限制", "source_url": "原文", "title_observed_at": "展示標題取得時間"}
        columns = list(names)
        st.dataframe(shown[columns].rename(columns=names), hide_index=True, width="stretch", column_config={
            "原文": st.column_config.LinkColumn("原文", display_text="開啟原文"),
            "情緒分數": st.column_config.NumberColumn(format="%.4f")})
        st.download_button("下載篩選後新聞紀錄", safe_csv(shown), "news_filtered_metadata.csv", "text/csv", key="news_download")
        if not shown.empty:
            selected = st.selectbox("查看逐篇評分與版本", list(range(len(shown))), format_func=lambda i: f"{shown.iloc[i].news_id} · {shown.iloc[i].ticker} · {shown.iloc[i].Date}", key="news_article")
            row = shown.iloc[selected]
            with st.container(border=True):
                st.text(row.title or f"文章 {row.news_id}（原評分快取未保存標題）")
                score_label = f"{row.Daily_Sentiment:.4f}" if pd.notna(row.Daily_Sentiment) else "缺評分"
                st.write(f"情緒分數：**{score_label}**；評分公式：P(positive) − P(negative)。")
                st.caption(row.title_scope)
                if row.source_url:
                    st.link_button("開啟鉅亨原文", row.source_url)
                fields = ["publish_time", "Date", "first_seen_at", "fetched_at", "available_at", "timing_status", "Model_ID",
                          "model_revision", "tokenizer_revision", "scoring_provenance_status", "source_revision", "content_sha256", "cache_row_hash", "cache_sources", "title_observed_at"]
                st.json({key: row[key] if pd.notna(row[key]) else None for key in fields})
    with st.expander("每日股票情緒彙總與來源狀態", expanded=True):
        daily = archive["daily"]
        if daily.empty:
            st.info("沒有每日情緒彙總檔；不以零值替代。")
        else:
            day = st.selectbox("彙總資料日", sorted(daily.Date.astype(str).unique(), reverse=True), key="news_daily_day")
            selected = daily[daily.Date.astype(str).eq(day)].copy()
            if "Source_Status" in selected:
                selected["Source_Status"] = selected.Source_Status.fillna("歷史種子／來源狀態未保存")
            st.dataframe(selected, hide_index=True, width="stretch")
            st.caption("verified_no_matching_article 是完成來源查詢後無匹配文章；空白／缺來源紀錄不是無新聞。歷史種子分數缺逐篇證據。")
    with st.expander("來源掃描與真正可用時間"):
        if not archive["calendar"].empty:
            st.dataframe(archive["calendar"], hide_index=True, width="stretch")
        if not archive["sessions"].empty:
            st.dataframe(archive["sessions"], hide_index=True, width="stretch")
        st.caption("掃描文章數是來源查詢筆數，不是匹配股票的文章數；沒有日期紀錄不能解讀為零篇。")
    with st.expander("原始保存來源"):
        for relative in archive["sources"]:
            st.link_button(relative, GITHUB + relative)
    st.caption("此處展示結構化新聞結果，未保存網頁截圖；原文網站目前內容可能已更新，不能當作當年評分版本的獨立證明。")
