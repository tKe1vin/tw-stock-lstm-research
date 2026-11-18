# 每日資料增長與歷史回測截止日的測試修復

日期：2026-11-19（Asia/Taipei）。

## 故障原因

[執行 44368632866](https://github.com/tKe1vin/tw-stock-lstm-research/actions/runs/44368632866) 在 `Run isolated unit checks` 發生一項失敗：`test_warmup_missing_rows_are_preserved_and_risk_matches_paired`。

V2.12 歷史研究固定截止 2026-11-17，`prepare_data()` 已在計算特徵、風控與標籤之前，將價格和情緒正確截到該日期，因此其歷史面板為 3135 列。每日行情檔已追加 11/18，測試卻以未截斷的 3136 列行情面板作比較，產生 shape mismatch。

該執行實際結果是 `1 failed, 490 passed, 13 skipped`，尚未進入模型驗證、四系統或新聞擷取。先前已保存的 11/18 成功結果不因這次測試失敗而失效。

## 修正與回歸檢查

- 測試的參考面板採用相同 `as_of` 截止日。保留歷史 3135 列、四列暖身缺值、風控逐值相等的斷言。
- 新增未來資料追加測試：在臨時檔加入六個未來交易日的極端價格及情緒，確認歷史價格、情緒、特徵、風控、標籤、交易日曆、三折切分、Scaler 與所有輸入視窗完全一致。
- 保留 145 日成熟與五日 pending 的檢查；未來價格不得提前使固定歷史標籤成熟。
- 來源檔增加資料後，完整檔案 SHA-256 應改變；測試也核對這項行為，沒有把新的來源內容偽稱為原快照。

本次只修改測試及本說明，不修改訓練／推論程式、模型、Scaler、每日訊號、歷史報告、模型包或 frozen manifest。每日排程保持原設定。

## 驗證方法

先執行相關歷史訓練、帳本與發布契約測試，再執行完整主專案／展示／交付契約測試，以及模擬沒有 Altair／Streamlit 的每日環境測試。另以保存模型包和固定來源重播歷史研究，核對預測與衍生表。

```bash
python -m pytest -q tests/test_v2_common_training.py tests/test_v2_common_books.py tests/test_v2_common_publication.py
python -m pytest -q tests dashboard_v2/tests docs/v2_integration/tests docs/v2_integration/v2_improvement_notes/tests
python -m scripts.research_v2_common --mode replay
```

推送至 `main` 後由 `tests/**` 路徑規則啟動新每日執行。應檢查新 commit 的流程；重跑舊 commit 仍會使用舊測試。

## 本機驗證結果

- 相關歷史訓練、帳本、發布契約測試：41 passed。
- 完整主專案、展示及交付契約：627 passed。
- 模擬缺Altair／Streamlit的每日環境：492 passed、13個可選展示項目略過。
- 保存模型包重播4,950筆預測，最大差8.881784197001252e-16；12張衍生財務／覆蓋CSV逐位元一致。
- 原凍結權重、模型卡與bundle雜湊通過檢查，維持原值。
