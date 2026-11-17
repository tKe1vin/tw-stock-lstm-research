# SOURCE_CONFLICT 報告：V2.11 2026-11-10（Phase 8 更新）

**分析時間：** 2026-11-17T13:28:42+08:00  
**可重現包：** `integration_audit_20261117/source_conflict_1110/`

| 項目 | 本機 | GitHub latest-review |
| --- | --- | --- |
| 訊號 SHA-256 | `SYNTHETIC_DEMO` local fixture | `SYNTHETIC_DEMO` github fixture |
| Created_At | 2026-11-10T18:30:07+08:00 | 2026-11-10T21:51:42+08:00 |
| Model_Sha256 | `53a1467f…` | 相同 |
| Price_Sha256 | `93215113…` | 相同 |
| Sentiment_Sha256 | `47d3c884…` | `01cabc6a…` **不同** |

## 預測差異

僅 **2330.TW / 2317.TW** 預測不同；Close 與當日 `Daily_Sentiment`（皆 0.0）相同。

## Phase 8 深化結果

1. **新聞／情緒：** 同日五檔列目前檔案上 Daily_Sentiment 皆 0.0、無欄位差；但完整 `sentiment.csv` 在 **2026-11-09** 的 2330／2317 已分叉（情緒值與 News_Count 不同）。LSTM 視窗會吃到歷史情緒，故不能只看訊號 CSV 當日欄位。
2. **本機凍結輸入：** 訊號記錄的 `47d3c884…` **已不在磁碟**（現況 `fd6f2478…`）→ **無法**用原始凍結情緒重播本機預測。
3. **GitHub 凍結輸入：** 現況 sentiment 仍為 `01cabc6a…` → **可以**對 GitHub 側做凍結重播。
4. **推論程式：** `v211.py`／`v211_daily.py`／`v211_market.py`／`v211_schedule.py` 四檔 SHA **皆不同**。
5. **匯出 schema：** GitHub 多 `Model_Bundle_Sha256`、`Trade_Eligible` 等欄。
6. **結論：** 差異與情緒快照＋時點相關，但因本機輸入遺失且程式碼不同，**證據不足時不得把全部差異直接歸因於新聞**。維持 `SOURCE_CONFLICT`，不以 ROI 選邊，NAV 只走本機。
