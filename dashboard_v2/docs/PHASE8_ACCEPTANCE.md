# Phase 8 最終驗收清單

檢查時間：2026-11-17T13:28:49+08:00

| # | 項目 | 結果 |
| --- | --- | --- |
| 1 | 11/17 成熟標籤時間正確 | PASS — 收盤前／無行情 → PENDING |
| 2 | 衝突對 SHA 穩定（現為 SYNTHETIC_DEMO） | PASS — evidence CSV 與 compare_source_conflicts 一致 |
| 3 | 成熟評分不重複追加 | PASS — (Signal_Date,Ticker) 唯一 |
| 4 | Ridge 與 V2.11 分開 | PASS |
| 5 | 8502 顯示最新合格資料 | PASS — 同步 13:28:42；讀 v211_local |
| 6 | 未成熟不進績效 | PASS — performance/confirmation PENDING |
| 7 | SOURCE_CONFLICT 呈現 | PASS |
| 8 | NAV 與帳本一致 | PASS — 100741.19751310395 |
| 9 | 不同協議不混合 | PASS |
| 10 | 8502 不寫正式資料 | PASS |
| 11 | Windows 排程未改 | PASS — `\TwStockLstmV211Daily` 仍 Ready，下次 18:30 |
| 12 | 正式 Repository 本輪未改寫 | PASS — 權重雜湊未變；本輪無寫入動作 |
| 13 | 無 Commit／Push／Merge | PASS — group-clone 僅未追蹤檔 |

**測試：** 27 passed / 0 failed / 0 skipped  
**未執行：** 收盤後 runner 產標籤（需 15:00+ 合法行情與既有排程；本次不重訓、不手動改正式 runner）
