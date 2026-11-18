# 每日工作流程的圖表依賴修復

日期：2026-11-18（Asia/Taipei）。

## 原因與影響

新增的 `tests/test_research_charts.py` 在測試收集階段直接載入 Altair。每日模型環境依 `requirements-ci.txt` 安裝，原本不含 Altair／Streamlit，因此收到 `ModuleNotFoundError: No module named 'altair'`，整份 pytest 提前中止。

已核對同一個錯誤的執行：

- [首次 push 執行 44052768319](https://github.com/tKe1vin/tw-stock-lstm-research/actions/runs/44052768319)
- [排程執行 44150930254](https://github.com/tKe1vin/tw-stock-lstm-research/actions/runs/44150930254)
- [排程執行 44184408220](https://github.com/tKe1vin/tw-stock-lstm-research/actions/runs/44184408220)

上述執行在 `Run isolated unit checks` 失敗，後續模型驗證、四系統、新聞標題擷取及狀態保存步驟都被略過。這些失敗不代表模型已運算後績效有誤，也沒有生成新的成功日誌。

## 修正

圖表測試在載入展示模組之前，以 `pytest.importorskip("altair")` 檢查可選展示依賴，與現有 Streamlit 測試的依賴處理方式一致。

兩個專用展示工作流程仍安裝 `dashboard/requirements.txt`，並明確執行圖表測試。每日模型環境維持原本套件配置，不因圖表依賴而中止其他測試。

這次不修改模型、Scaler、推論程式、已發布訊號、每日帳本、新聞分數或歷史績效。台北週一至週五 17:37／21:37、每日 09:17 重試排程保持原設定；GitHub 實際啟動時間可能延遲。

## 本機驗證

- 在獨立程序阻擋 Altair／Streamlit 匯入，模擬每日環境：`451 passed, 11 skipped`。略過項目皆是可選展示測試，其他測試完整執行。
- 含展示依賴的完整測試：`564 passed`。
- 凍結權重、模型卡與 bundle 的 hash-only 驗證通過，雜湊保持原值。
- 修正推送至 `main` 後會由 `tests/**` 路徑規則啟動新的每日工作流程。重跑舊失敗 commit 不會取得這次修正，應查核新 commit 的執行結果。

本機驗證不替代遠端完整流程結果；遠端新聞／行情若有獨立來源延遲，仍應照實顯示等待或部分完成，不能補造成功訊號。
