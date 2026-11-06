# 每日四系統架構

`scripts.run_all_daily` 是唯一每日入口，`scripts.pipeline` 保留相容轉接。

1. 以台北時鐘與 XTAI 確認當日已完成交易。
2. 子程序 `scripts.update_market_data` 一次更新 `experiments/results/v2/v211/market`，保護 seed 與已發訊號，複製當日快照。
3. V2.11 使用已更新市場發布當日不可覆寫訊號、處理成交和成熟標籤。
4. 555 V2 使用相同凍結權重輸出預測，不寫第二份研究樣本。
5. 444、555 各自執行 Notebook，讀同一快照、各自保存預測。
6. `scripts.daily_reports` 稽核當日輸出，僅對同協議的 444／555 產生比較。
7. `scripts.persist_daily_state` 驗證不可變資料後保存精簡狀態，完整 runs 作 Actions artifacts。

NLP、市場更新、各模型分程序，避免共用模型狀態與長時間累積記憶體。共享的是來源資料，四系統的輸出／評估協議仍分開。

正式匯入基準保留在 data/，Complete 的 11/04 重播輸入另封存 data/examples/reproducible。每日新資料唯一來源是 V2.11 market。凍結權重、developer_original 與已提交研究訊號不可改寫；獨立帳本只鏡像已驗證的共享市場，不複製其他帳本的訊號或績效。

check 模式只讀檢查，不建立檔案、不發布。完整模式結束碼0／2／1分別代表完成、等待、失敗；即使最後失敗也持久化先前合法的部分成果。
