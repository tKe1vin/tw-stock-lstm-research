"""555 V2, journal, V2.11, and Ridge page. Incomparable books stay out of the ranking."""
from __future__ import annotations

import streamlit as st

from dashboard_v2.charts.svg import bars, lines
from dashboard_v2.components.table import cards, table
from dashboard_v2.data_access.load import NAMES, v2_comparison
from dashboard_v2.data_access.prospective import (
    day_review,
    ledger_status,
    next_day_strategy,
    ridge_status,
    source_status,
    trading_journal,
    v211_matured_local,
    v211_nav_local,
    v211_status_local,
)


def render(section, journal_day, ticker):
    if section == "V2 總覽":
        _overview()
    elif section == "原 555 與 V2":
        _design()
    elif section == "歷史實驗":
        _history()
    elif section == "本機前瞻狀態":
        _source_panel()
    elif section == "V2.11":
        _forward()
    elif section == "Ridge 前瞻":
        _ridge()
    elif section == "交易日誌":
        _journal(journal_day, ticker)
    elif section == "當日檢討":
        _review()
    elif section == "隔日策略":
        _next_day()
    else:
        _limits()


def _overview():
    data = v2_comparison()
    selected = _book(data, "rolling_selected_threshold_weekly")
    equal = _book(data, "equal_weekly")
    st.markdown(
        "V2 是因為舊 555 沒有超過 444 才做的五日探索。它改了預測期間、滾動訓練、下一交易日收盤與五日再平衡。"
        " 結果仍是探索，不是已證實的改良。"
    )
    st.markdown(cards([
        {"label": "等權 ROI", "value": f"{equal['ROI_Pct']}%", "note": "可比基準"},
        {"label": "rolling_selected ROI", "value": f"{selected['ROI_Pct']}%", "note": "低於等權"},
        {"label": "差距", "value": f"{selected['Gap_Vs_Equal_ROI_Pct']} pp", "note": "EXPLORATORY_RESULT"},
        {"label": "權重檔", "value": "未保存" if data["weights_saved"] is False else "已保存", "note": "各折權重 UNVERIFIED"},
    ]), unsafe_allow_html=True)
    st.markdown(_flow())


def _design():
    st.markdown(cards([
        {"label": "原六特徵 555", "value": "NOT_DIRECTLY_COMPARABLE", "note": "一日、2026 測試窗"},
        {"label": "V2 rolling_selected", "value": "NOT_DIRECTLY_COMPARABLE", "note": "五日、2022-05-24 至 2026-03-17"},
        {"label": "共同核心 555", "value": "另一協議", "note": "配對至 2026-11-06"},
        {"label": "V3", "value": "0050 探索／B0", "note": "不進 V2 排行"},
    ]), unsafe_allow_html=True)
    st.markdown(table([
        {"item": "預測", "old": "六特徵 LSTM，一日", "new": "六特徵 rolling，五日標籤 Close[t+6]/Close[t+1]-1"},
        {"item": "訓練", "old": "固定切分", "new": "walk-forward；V2.10 各折 .keras 未保存"},
        {"item": "成交", "old": "歷史協議", "new": "下一交易日收盤，五日再平衡"},
        {"item": "風控", "old": "動態配置內", "new": "事前風險上限；V2.11 上限 0.50"},
        {"item": "績效排名", "old": "不能與 V2 相減", "new": "只與 equal_weekly 同一本帳比較"},
    ], [("item", "項目"), ("old", "原 555"), ("new", "V2")]), unsafe_allow_html=True)
    st.caption("上表是研究方法比較。沒有同一價格、同一成交與同一成本的配對帳本，所以不畫勝負排行。")


def _history():
    data = v2_comparison()
    ranked = data["ranked"]
    st.markdown(bars(
        [{"label": row["Name"].replace("_threshold_weekly", ""), "value": row["ROI_Pct"]} for row in ranked],
        "只列入 ROI_Comparable_With_Equal 為 true 的 ROI",
        "%",
    ), unsafe_allow_html=True)
    st.markdown(bars(
        [{"label": row["Name"].replace("_threshold_weekly", ""), "value": row["Total_Transaction_Cost"]} for row in ranked],
        "可比組交易成本",
    ), unsafe_allow_html=True)
    quality = next(row for row in data["quality"] if row["Split"] == "test")
    validation = next(row for row in data["quality"] if row["Split"] == "validation")
    st.markdown(cards([
        {"label": "測試方向準確率", "value": f"{float(quality['Direction_Accuracy']):.3f}", "note": "quality.test"},
        {"label": "測試 Spearman", "value": f"{float(quality['Spearman']):.3f}", "note": "低於驗證"},
        {"label": "驗證 Spearman", "value": f"{float(validation['Spearman']):.3f}", "note": "quality.validation"},
        {"label": "測試 MAE", "value": f"{float(quality['MAE']):.3f}", "note": "百分點任務"},
    ]), unsafe_allow_html=True)
    st.markdown(lines([], "V2 對等權的逐日 NAV"), unsafe_allow_html=True)
    st.caption("comparison.json 只有彙總帳，沒有 rolling_selected 的逐日 NAV，所以不補曲線。")
    st.markdown(table([
        {"name": row["Name"], "roi": row["ROI_Pct"], "reason": "可比旗標不是 true"}
        for row in data["blocked"]
    ], [("name", "未排名"), ("roi", "檔內 ROI"), ("reason", "原因")]), unsafe_allow_html=True)


def _source_panel():
    status = source_status()
    freshness = status.get("freshness") or {}
    gate = status.get("label_maturity_gate") or {}
    st.markdown("##### 本機前瞻追蹤狀態")
    st.markdown(cards([
        {"label": "本機價格截止", "value": status["local_v211_through"], "note": status.get("local_model_version", "LOCAL_PROSPECTIVE")},
        {"label": "本機最新訊號", "value": status["local_latest_signal"], "note": status.get("local_latest_signal_at") or f"成熟日數 {status['local_matured_signal_days']}"},
        {"label": "本機成熟日數", "value": status["local_matured_signal_days"], "note": "LOCAL_PROSPECTIVE 獨立帳本"},
        {"label": "證據狀態", "value": freshness.get("evidence_status", "UNKNOWN"), "note": f"同步 {freshness.get('synced_at') or status['sync'].get('synced_at', '未同步')}"},
    ]), unsafe_allow_html=True)
    st.markdown("##### GitHub 研究狀態（≠ 本機 V2.11 前瞻）")
    st.markdown(cards([
        {"label": "research_as_of", "value": status.get("github_research_as_of_date") or "UNKNOWN", "note": "444/555 資料日；非準時可交易保證"},
        {"label": "V2.11 last_successful", "value": status.get("github_latest_success") or "UNKNOWN", "note": "GitHub 凍結追蹤器最後成功發布"},
        {"label": "delayed_run", "value": str(status.get("github_delayed_run")), "note": status.get("github_orchestrator_status") or ""},
        {"label": "v211 tracker", "value": "maintenance_only" if status.get("github_v211_maintenance_only") else str(status.get("github_v211_status")), "note": f"signal_published={status.get('github_v211_signal_published')}"},
    ]), unsafe_allow_html=True)
    st.caption(status.get("github_status_note") or "")
    st.markdown(cards([
        {"label": "SOURCE_CONFLICT", "value": str(status["source_conflict"]), "note": status.get("local_source_commit", "")},
        {"label": "11/09 標籤閘門", "value": gate.get("state", "PENDING"), "note": f"label_end {gate.get('label_end')}；收盤後={gate.get('protocol_close_passed')}"},
        {"label": "資料來源", "value": "LOCAL_PROSPECTIVE", "note": status["display_root"]},
        {"label": "權重核對", "value": str(status["local_weight_ok"]), "note": f"GitHub 日 {', '.join(status['github_signal_days']) or '無'}"},
    ]), unsafe_allow_html=True)
    st.caption(
        f"顯示根目錄 {status['display_root']}；權重核對 {status['local_weight_ok']}；"
        f"檢查時間 {status['checked_at']}；"
        f"GitHub V2.11 訊號日 {', '.join(status['github_signal_days']) or '無'}；"
        f"research_status={status.get('github_research_status')}。"
    )
    if freshness.get("evidence_status") in {"STALE", "SYNC_FAILED"}:
        st.warning(f"展示快照狀態為 {freshness['evidence_status']}，保留最後完整資料，不自動發明績效。")
    if status["conflicts"]:
        conflict = status["conflicts"][0]
        integrity = conflict.get("integrity") or {}
        st.warning("偵測到 SOURCE_CONFLICT，兩來源分開顯示，不合併 NAV。")
        st.markdown(cards([
            {"label": "SOURCE_CONFLICT", "value": "True", "note": conflict["signal_date"]},
            {"label": "ORIGINAL_INPUT_UNAVAILABLE", "value": str(integrity.get("ORIGINAL_INPUT_UNAVAILABLE", True)), "note": "本機凍結情緒遺失"},
            {"label": "PARTIALLY_REPRODUCIBLE", "value": str(integrity.get("PARTIALLY_REPRODUCIBLE", False)), "note": "僅 GitHub 側可重播"},
            {"label": "禁令", "value": "NO_REFETCH", "note": "不得用現況新聞冒充原輸入"},
        ]), unsafe_allow_html=True)
        st.markdown(table([{
            "date": conflict["signal_date"],
            "local": conflict["local_sha256"][:16],
            "github": conflict["github_sha256"][:16],
            "local_at": conflict["local_created_at"],
            "github_at": conflict["github_created_at"],
            "same_price": conflict["same_price_sha"],
            "same_model": conflict["same_model_sha"],
        }], [
            ("date", "訊號日"), ("local", "本機雜湊"), ("github", "GitHub 雜湊"),
            ("local_at", "本機發布"), ("github_at", "GitHub 發布"),
            ("same_price", "同價格雜湊"), ("same_model", "同模型雜湊"),
        ]), unsafe_allow_html=True)
        st.caption(conflict["note"])
        diffs = conflict.get("field_diffs", [])[:20]
        if diffs:
            st.markdown("##### 欄位差異（節錄）")
            st.markdown(table(diffs, [
                ("ticker", "股票"), ("field", "欄位"), ("local", "本機"), ("github", "GitHub"),
            ]), unsafe_allow_html=True)


def _forward():
    status = v211_status_local()
    payload = status["payload"]
    matured = v211_matured_local()
    ledger = ledger_status()
    st.markdown(cards([
        {"label": "確認狀態", "value": status["confirmation"], "note": f"成熟訊號日 {status['matured_signal_days']}"},
        {"label": "未成熟預測", "value": status["pending_predictions"], "note": "缺值不是 0%"},
        {"label": "權重核對", "value": "OK" if status["weight_ok"] else "FAIL", "note": status["model_sha256"][:12]},
        {"label": "NAV_Open", "value": f"{status['open_nav']:.2f}", "note": f"{ledger.get('latest_date')}；非日終 NAV_Close"},
    ]), unsafe_allow_html=True)
    st.markdown(cards([
        {"label": "NAV_Close", "value": ledger.get("nav_close_status", "DATA_UNAVAILABLE"), "note": ledger.get("liquidation_note", "")},
        {"label": "模擬成交次數", "value": ledger.get("published_fill_count", 0), "note": ledger.get("execution_kind")},
        {"label": "現金比例", "value": ledger.get("cash_weight"), "note": f"Equity {ledger.get('equity_weight')}"},
        {"label": "Equal NAV", "value": (ledger.get("equal_latest") or {}).get("NAV_Open", "DATA_UNAVAILABLE"), "note": "同曝險對照"},
    ]), unsafe_allow_html=True)
    st.caption(
        f"SOURCE {status['source_label']}；ROOT {status['source']}；"
        f"LAST_SIGNAL {status['latest_signal_date']} @ {status['latest_signal_at']}；"
        f"REGISTRY_OK {status['registry_ok']}。"
    )
    nav = v211_nav_local()["rows"]
    st.markdown(lines([{
        "name": "本機模擬 NAV",
        "points": [(row["Date"], float(row["NAV_Open"])) for row in nav],
    }], "只使用本機前瞻帳本，不混 GitHub"), unsafe_allow_html=True)
    st.markdown("##### 已成熟五日標籤")
    if matured["rows"]:
        st.markdown(table(matured["rows"], [
            ("Signal_Date", "訊號日"), ("Ticker", "代號"), ("Predicted", "預測"), ("Actual_5d_Pct", "實際五日％"),
        ]), unsafe_allow_html=True)
    else:
        st.markdown('<div class="gap">PENDING_LABEL</div>', unsafe_allow_html=True)
    st.markdown("##### 未成熟標籤")
    st.markdown(table(payload["scoring"]["pending_labels"], [
        ("signal", "訊號日"), ("label_end", "標籤成熟日"),
    ]), unsafe_allow_html=True)
    metrics = status["metrics"]
    if metrics:
        st.caption(
            f"現有成熟樣本指標（非正式驗證）：MAE {metrics.get('MAE')}；"
            f"Direction {metrics.get('Direction_Accuracy')}；Spearman {metrics.get('Cross_Section_Spearman')}。"
            " 樣本遠少於 60 日，維持 PENDING。"
        )


def _ridge():
    data = ridge_status()
    st.markdown(
        "Ridge 是一日 Close-only 控制組，預測期間與 V2.11 五日標籤不同，禁止合併績效。"
    )
    st.markdown(cards([
        {"label": "研究狀態", "value": data["status"], "note": f"需滿 {data.get('min_scorable_days_for_claim', 252)}／{data.get('protocol_days', 273)} 日"},
        {"label": "可評分交易日", "value": data["scorable_days"], "note": f"列數 {data['rows']}"},
        {"label": "MAE / RMSE", "value": "None" if data["mae"] is None else f"{data['mae']:.4f} / {data.get('rmse'):.4f}", "note": "即時重算，非硬編碼"},
        {"label": "Spearman", "value": "None" if data["spearman_mean"] is None else f"{data['spearman_mean']:.4f}", "note": f"CI [{data.get('spearman_lower')}, {data.get('spearman_upper')}]"},
    ]), unsafe_allow_html=True)
    st.markdown(cards([
        {"label": "方向準確率", "value": "None" if data["direction"] is None else f"{data['direction']:.4f}", "note": "非零報酬列"},
        {"label": "零預測 MAE", "value": "None" if data["zero_mae"] is None else f"{data['zero_mae']:.4f}", "note": "基準"},
        {"label": "首日雜湊", "value": "OK" if data["first_day_ok"] else "FAIL", "note": (data["first_day_prediction_sha256"] or "")[:12]},
        {"label": "今日標籤", "value": data.get("today_label_state", "UNKNOWN"), "note": "不與 V2.11 混算"},
    ]), unsafe_allow_html=True)
    st.caption(
        f"來源 {data.get('source_signals')}；已發布目標日 {', '.join(data.get('published_targets') or [])}；"
        f"已評分 {', '.join(data.get('scored_days') or [])}；"
        f"未成熟 {', '.join(data.get('pending_targets') or [])}；檢查時間 {data.get('checked_at')}。"
    )


def _journal(day, ticker):
    rows = trading_journal(day, ticker)
    st.caption("所有成交都是 SIMULATED_EXECUTION / PAPER_TRADING，沒有券商委託。來源固定為本機前瞻快照。")
    st.markdown(table(rows, [
        ("signal_date", "訊號日"), ("model_version", "模型"), ("ticker", "代號"), ("name", "名稱"),
        ("direction", "方向"), ("forecast", "預測"), ("target_weight", "目標權重"),
        ("cash_weight", "現金"), ("risk_off", "Risk-Off"), ("rebalance_day", "再平衡日"),
        ("simulated_fill", "模擬成交"), ("execution_date", "成交日"), ("fill_price", "成交價"),
        ("buy_fee", "買進手續費"), ("sell_tax", "交易稅"), ("nav", "NAV"),
        ("source", "來源"), ("state", "狀態"), ("file_sha", "訊號雜湊"),
    ]), unsafe_allow_html=True)


def _review():
    review = day_review()
    st.markdown(cards([
        {"label": "標籤狀態", "value": review["label_state"], "note": f"成熟日數 {review['matured_signal_days']}"},
        {"label": "成交狀態", "value": review["trade_state"], "note": f"再平衡 {review['rebalance_today']}"},
        {"label": "流程", "value": "OK" if review["flow_ok"] else "ERROR", "note": review["error"] or "status.ok"},
        {"label": "NAV", "value": f"{float(review['open_nav']):.2f}", "note": f"成本 {review['entry_cost']}"},
    ]), unsafe_allow_html=True)
    if review["matured_rows"]:
        st.markdown("##### 已成熟評分列")
        st.markdown(table(review["matured_rows"], [
            ("Signal_Date", "訊號日"), ("Ticker", "代號"), ("Predicted", "預測"), ("Actual_5d_Pct", "實際五日％"),
        ]), unsafe_allow_html=True)
    else:
        st.markdown('<div class="gap">PENDING_LABEL</div>', unsafe_allow_html=True)
    if review["fills"]:
        st.markdown("##### 已發生模擬成交")
        st.markdown(table(review["fills"], [
            ("Signal_Date", "訊號日"), ("Execution_Date", "成交日"), ("Cost", "成本"),
            ("Buy_Fee", "買進手續費"), ("Sell_Tax", "交易稅"), ("NAV_Open", "NAV"),
        ]), unsafe_allow_html=True)
        st.caption("原始成交檔Buy_Fee僅為買進手續費，Cost另包含賣出手續費與交易稅；完整成本拆分見V2共同展示的成本與風險頁。")
    else:
        st.markdown('<div class="gap">NO_TRADE</div>', unsafe_allow_html=True)
    st.markdown(
        f"價格到 {review['prices_through']}；下一標籤日 {review['next_label_end']}；"
        f"檢查時間 {review['checked_at']}。限制：{'；'.join(review['limitations'])}。"
    )


def _next_day():
    plan = next_day_strategy()
    if plan["state"] != "LATEST_PUBLISHED_SIGNAL":
        st.markdown(f'<div class="gap">{plan["state"]}</div>', unsafe_allow_html=True)
        return
    st.markdown(cards([
        {"label": "訊號日", "value": plan["signal_date"], "note": plan["created_at"]},
        {"label": "預測假設進場", "value": plan["forecast_entry"], "note": "標籤用途，不等於帳本交易日"},
        {"label": "現金／Risk-Off", "value": f"{plan['cash_weight']} / {plan['risk_off']}", "note": plan["model_version"]},
        {"label": "訊號雜湊", "value": plan["file_sha256"][:12], "note": plan["source_label"]},
    ]), unsafe_allow_html=True)
    st.markdown(cards([
        {"label": "再平衡排程", "value": plan.get("scheduled_rebalance") or "未保存", "note": "依保存的帳本排程"},
        {"label": "交易資格", "value": plan.get("trade_eligible") or "未保存", "note": plan.get("execution_status") or "SIGNAL_ONLY"},
    ]), unsafe_allow_html=True)
    if str(plan.get("trade_eligible", "")).lower() != "true":
        st.info("此訊號未具備已保存的再平衡交易資格；預測進場日與目標權重不能當成隔日掛單。")
    st.markdown(table([
        {
            "ticker": row["Ticker"],
            "name": NAMES.get(row["Ticker"], ""),
            "forecast": row["Predicted_Return_Pct"],
            "rank": row["Rank"],
            "weight": row["Target_Weight"],
            "risk": row["Risk_Off"],
            "status": row.get("Execution_Status") or "SIGNAL_ONLY",
        }
        for row in plan["rows"]
    ], [
        ("ticker", "代號"), ("name", "名稱"), ("forecast", "預測百分點"),
        ("rank", "排序"), ("weight", "目標權重"), ("risk", "Risk-Off"), ("status", "狀態"),
    ]), unsafe_allow_html=True)
    st.caption(
        f"{plan['note']} 持有 {plan['hold_sessions']} 個交易日；模型 {plan['model_sha256'][:12]}；"
        f"最近模擬成交日 {plan['last_simulated_execution']}。"
    )


def _limits():
    st.markdown(
        """
V2 歷史帳沒有勝過同協議等權。各折權重未保存，標成 UNVERIFIED。
V2.11 與 Ridge 都是 PENDING 前瞻追蹤，不是已證實超額報酬。
V3 改以 0050 為基準並採 B0_benchmark，不進 V2 排行。
本機與 GitHub 同日訊號若雜湊不同，維持 SOURCE_CONFLICT，不混 NAV。
取消 Risk-Off 的同預測逐日權重不在 comparison.json，不能補造消融。
新聞標題沒有存進快取。歷史情緒多數沒有取得時間。
"""
    )


def _book(data, name):
    return next(row for row in data["books"] if row["Name"] == name)


def _flow():
    return """
#### 流程差異
原 555：股價、技術指標、新聞與風控進入多變量 LSTM，再做歷史動態配置。

V2：rolling / walk-forward、五日預測、事前風控、下一交易日收盤與五日再平衡。

V2.11：凍結權重每日登錄訊號，成熟後才評分。展示層讀本機前瞻快照。

Ridge：獨立一日 Close-only 控制組，不與 V2.11 合併。
"""
