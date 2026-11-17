"""444 / 555 / TWII page. Charts follow the selected protocol only."""
from __future__ import annotations

import streamlit as st

from dashboard_v2.charts.svg import bars, lines
from dashboard_v2.components.table import cards, table
from dashboard_v2.data_access.load import (
    historical_official,
    news_observations,
    paired_metrics,
    paired_nav,
    paired_returns,
    paired_weights,
)

PAIRED_ARMS = (
    "static_equal444",
    "dynamic_news_risk555",
    "risk_matched_equal_reference",
    "market_price_reference",
)


def render(protocol, news_ticker, news_class, keyword):
    if protocol == "原始歷史比較":
        _historical()
    else:
        _paired()
    st.markdown("#### 每日財經新聞")
    _news(news_ticker, news_class, keyword)


def _historical():
    data = historical_official()
    rows = data["rows"]
    st.markdown(
        f"協議 `{data['protocol']}`，狀態 `{data['status']}`，來源 `{data['source']}`。"
        " 這組不能與下方新版配對帳本相減。"
    )
    order = [("444", "444"), ("555", "555"), ("^TWII benchmark", "TWII 價格參考")]
    st.markdown(cards([
        {"label": f"{label} ROI", "value": rows[key]["ROI_Pct"] + "%", "note": data["status"]}
        for key, label in order
    ]), unsafe_allow_html=True)
    st.markdown(cards([
        {"label": "444 Sharpe", "value": rows["444"]["Sharpe"], "note": "strategy_comparison.csv"},
        {"label": "555 Sharpe", "value": rows["555"]["Sharpe"], "note": "當時六特徵 555"},
        {"label": "444 MDD", "value": rows["444"]["MDD_Pct"] + "%", "note": "負值為回撤"},
        {"label": "555 MDD", "value": rows["555"]["MDD_Pct"] + "%", "note": "較低回撤不等於預測變強"},
    ]), unsafe_allow_html=True)
    st.caption(data["twii_note"] + " RESEARCH_HISTORY.md 的 555 ROI 末位是 323，本畫面採用 CSV 原文 325。")
    st.markdown(bars([
        {"label": label, "value": rows[key]["ROI_Pct"]} for key, label in order
    ], "已保存 ROI", "%"), unsafe_allow_html=True)
    st.markdown(bars([
        {"label": label, "value": rows[key]["Sharpe"]} for key, label in order
    ], "已保存 Sharpe"), unsafe_allow_html=True)
    st.markdown(lines([], "累積 NAV"), unsafe_allow_html=True)
    st.markdown(_method("historical"))
    st.markdown(_conclusion("historical", rows))


def _paired():
    data = paired_metrics()
    by_arm = {row["Arm"]: row for row in data["rows"]}
    st.markdown(
        f"協議 `{data['protocol']}`，狀態 `{data['status']}`，因果驗證 `{data['causal_validated']}`。"
        f" 來源 `{data['source']}`。區間至 2026-11-06。"
    )
    picks = [
        ("static_equal444", "444 固定持股"),
        ("dynamic_news_risk555", "555 新聞加風控"),
        ("risk_matched_equal_reference", "同曝險等權"),
        ("market_price_reference", "TWII 價格參考"),
    ]
    st.markdown(cards([
        {"label": label + " ROI", "value": by_arm[key]["ROI_Pct"] + "%", "note": "EXPLORATORY"}
        for key, label in picks
    ]), unsafe_allow_html=True)
    st.caption("市場列是 ^TWII 理論價格指數，成本為 0，不是 ETF 淨報酬。")
    nav = paired_nav()
    series = []
    for key, label in picks:
        points = [
            (row["Date"], float(row["NAV"]))
            for row in nav["rows"] if row["Arm"] == key
        ]
        series.append({"name": label, "points": points})
    st.markdown(lines(series, "同一配對帳本的 NAV"), unsafe_allow_html=True)
    drawdown = []
    returns = []
    for key, label in picks[:3]:
        points = _drawdown([float(row["NAV"]) for row in nav["rows"] if row["Arm"] == key])
        drawdown.append({"name": label, "points": list(enumerate(points))})
        returns.append({
            "name": label,
            "points": [
                (row["Date"], float(row["Net_Return_Pct"]))
                for row in paired_returns(key)
            ],
        })
    st.markdown(lines(drawdown, "由 nav.csv 計算的回撤；KPI 仍以 metrics.csv 為準"), unsafe_allow_html=True)
    st.markdown(lines(returns, "已保存每日淨報酬百分點"), unsafe_allow_html=True)
    st.markdown(bars([
        {"label": label, "value": by_arm[key]["Total_Transaction_Cost"]}
        for key, label in picks
    ], "帳列交易成本", "元"), unsafe_allow_html=True)
    weight_rows = paired_weights("dynamic_news_risk555")
    by_day = {}
    for row in weight_rows:
        by_day.setdefault(row["Target_Date"], []).append(row)
    exposure = []
    for day, group in by_day.items():
        exposure.append((day, sum(float(item["Actual_Weight"]) for item in group)))
    st.markdown(lines([
        {"name": "555 實際權重和", "points": exposure}
    ], "555 當日實際權重合計"), unsafe_allow_html=True)
    st.markdown(table([
        {
            "arm": row["Arm"],
            "roi": row["ROI_Pct"],
            "sharpe": row["Sharpe"],
            "mdd": row["MDD_Pct"],
            "cost": row["Total_Transaction_Cost"],
            "equity": row["Average_Equity_Weight_Pct"],
            "basis": row["Return_Basis"],
        }
        for row in data["rows"]
    ], [
        ("arm", "研究組"), ("roi", "ROI"), ("sharpe", "Sharpe"), ("mdd", "MDD"),
        ("cost", "成本"), ("equity", "平均持股"), ("basis", "報酬口徑"),
    ]), unsafe_allow_html=True)
    st.markdown(_method("paired"))
    st.markdown(_conclusion("paired", by_arm))
    st.download_button(
        "匯出目前配對指標",
        _export(data["rows"], data["source"], data["protocol"], data["status"]),
        file_name="paired_metrics_view.csv",
        mime="text/csv",
    )


def _news(ticker, klass, keyword):
    data = news_observations()
    rows = []
    for row in data["rows"]:
        if ticker != "全部" and row["ticker"] != ticker:
            continue
        if klass != "全部" and row["sentiment_class"] != klass:
            continue
        blob = " ".join([row["news_id"], row["ticker"], row["source"], row["publish_time"]])
        if keyword and keyword not in blob:
            continue
        rows.append(row)
    st.caption(
        f"來源 {data['source']}。快取沒有文章標題，畫面不補標題。"
        " publish_time、first_seen_at、fetched_at 分開保存。單篇是否進入決策維持 UNVERIFIED。"
    )
    st.markdown(cards([
        {"label": "觀察列", "value": len(rows), "note": "篩選後"},
        {"label": "缺標題", "value": sum(item["title_status"] == "TITLE_NOT_STORED" for item in rows), "note": "TITLE_NOT_STORED"},
        {"label": "零分未驗證", "value": sum(item["sentiment_class"] == "ZERO_UNVERIFIED" for item in rows), "note": "不當成中性"},
    ]), unsafe_allow_html=True)
    counts = {}
    for row in rows:
        counts[row["ticker"]] = counts.get(row["ticker"], 0) + 1
    st.markdown(bars(
        [{"label": key, "value": value} for key, value in sorted(counts.items())],
        "篩選後觀察列數",
    ), unsafe_allow_html=True)
    st.markdown(table(rows, [
        ("news_id", "編號"), ("ticker", "股票"), ("title_status", "標題"),
        ("source", "來源"), ("publish_time", "發布"), ("first_seen_at", "首次看到"),
        ("fetched_at", "擷取"), ("sentiment", "情緒分數"), ("sentiment_class", "分類"),
        ("decision_use", "是否進入決策"),
    ]), unsafe_allow_html=True)
    st.download_button(
        "匯出目前新聞篩選",
        _export(rows, data["source"], "v211 news observation", "TITLE_NOT_STORED"),
        file_name="news_observations_view.csv",
        mime="text/csv",
    )


def _drawdown(values):
    peak = values[0]
    output = []
    for value in values:
        peak = max(peak, value)
        output.append((value / peak - 1) * 100)
    return output


def _export(rows, source, protocol, status):
    import csv
    import io
    from datetime import datetime, timezone
    buffer = io.StringIO()
    if not rows:
        return "status,DATA_UNAVAILABLE\n"
    fields = list(rows[0].keys()) + ["source_path", "protocol", "research_status", "exported_at"]
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    stamp = datetime.now(timezone.utc).isoformat()
    for row in rows:
        item = dict(row)
        item.update({
            "source_path": source,
            "protocol": protocol,
            "research_status": status,
            "exported_at": stamp,
        })
        writer.writerow(item)
    return buffer.getvalue()


def _method(kind):
    common = """
#### 實驗方法
五檔股票是專題固定池：2330.TW、2317.TW、2603.TW、2002.TW、2891.TW。
444 使用 Close 單變量 LSTM。原六特徵 555 另外使用 MA20、RSI_14、MACD_Hist、Market_Vol_20d、Daily_Sentiment，因此不能把 ROI 差距單獨稱為新聞效果。
新版配對研究讓 444 與 555 共用已保存的 444 預測，只改變配置、新聞與 Risk-Off。
新聞正向門檻是 0.3、負向是 -0.3。Risk-Off 使用訊號日已知道的 20 日波動，超過先前波動均值加一個標準差時，總曝險上限 0.50。
單一股票權重上限 0.30。手續費 0.001425，賣出交易稅 0.003，初始資金 100000。
成交在下一交易日收盤。Sharpe 與 MDD 以各協議已保存的 metrics 為準。
同曝險等權用來分開「少持股」與「選股」。新聞增量要看有新聞與無新聞的配對組，不看離 444 最遠的那一組。
"""
    extra = "此頁是 2026-11-04 歷史快照，沒有對應的逐日 NAV，所以不畫累積曲線。" if kind == "historical" else "此頁使用 2026-11-06 paired 帳本。歷史情緒的取得證據不完整，因果標記為 false。"
    return common + "\n" + extra


def _conclusion(kind, rows):
    if kind == "historical":
        return (
            "#### 研究結論\n"
            f"已確認的歷史快照：444 ROI {rows['444']['ROI_Pct']}%，"
            f"當時 555 ROI {rows['555']['ROI_Pct']}%。555 較低。\n\n"
            "這是 HISTORICAL_OFFICIAL，不是新版共同核心的結果，也不能解釋成新聞造成的因果。"
            " 555 的 MDD 較小，不能因此說預測能力較高。新聞時間證據仍有缺口。"
        )
    news = float(rows["dynamic_news_risk555"]["ROI_Pct"])
    no_news = float(rows["dynamic_no_news_risk"]["ROI_Pct"])
    matched = float(rows["risk_matched_equal_reference"]["ROI_Pct"])
    return (
        "#### 研究結論\n"
        f"探索結果：完整 555 為 {news}%，同風控但無新聞為 {no_news}%，同曝險等權為 {matched}%。"
        " 新聞組沒有高於無新聞風控組，也沒有高於同曝險等權。\n\n"
        "市場價格參考較高，但它沒有成本與股利，不能當成可交易大盤勝率。"
        " 這組是 EXPLORATORY，causal_validated 為 false。"
    )
