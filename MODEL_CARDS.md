# 模型卡

## 444

- 識別：`444`
- 來源：`notebooks/444_baseline.ipynb`
- 形態：單變量 LSTM，固定部位參考
- 訓練：每次執行重新訓練。`EPOCHS = 50`，`PATIENCE = 5`，`BATCH_SIZE = 32`，`SEED = 42`
- 測試窗：150 個交易日。驗證窗：60 個交易日
- 協議：`lstm-evaluation-v1`
- 已保存正式結果：截至 2026-11-04，淨報酬 21.376684850112927%
- 沒有凍結權重。`model_hash` 在統一 CSV 寫「不適用」

## 555（共同核心主研究）

- 識別：`555`；協議：`paired-444-core-allocation-v1`
- 同日成功的 444 預測作共同核心，加入預先固定的動態權重、新聞與市場風控；不另訓練不同核心作主對照。
- 所有主研究組共用次日收盤、交易日、價格與費率，包含新聞／風控 2×2 消融與同曝險等權。
- 一日模型標籤在次日收盤前已完成；配置是延遲訊號規則，不把該段預測報酬當成新進部位的可賺報酬。
- 歷史新聞來源與已觀察區間仍屬探索性。推薦／研究輸出沒有券商訂單。

## 原六特徵 555（保留的延伸／歷史診斷）

- `notebooks/555_multivariate.ipynb` 保留原始模型；以 `--legacy-555` 或原 reproducible fixture 重播。
- Close、MA20、RSI_14、MACD_Hist、Market_Vol_20d、Daily_Sentiment；原同收盤 ROI 僅為理想化歷史紀錄，不與新主研究混排。

## 555 V2

- 識別：`555_v2`
- 版本：`rolling_selected_fold16_window756_carried`
- 結構：兩層 LSTM(50)、Dropout 0.2、Dense(1)、MSE、Adam 學習率 0.001
- 視窗：756個可用訓練訊號／標籤；連續訊號的10日輸入聯集為765列。早期停止樣本60個訊號；時間步10
- 權重：`models/model_555_v2/weights/weights.npz`
- SHA-256：`53a1467fdba66e7f22a45e5f2fbc1cb78fc99825b15d516b3e7153d7bfc28144`
- Scaler：`models/model_555_v2/scaler/model_card.json`，只含當初訓練列擬合的 MinMax 狀態
- 成交假設：下一交易日收盤，每 5 個交易日再平衡，Risk-Off 總曝險上限 0.50
- 研究結論：尚未證明優於固定等權或加權指數
- 預測輸出不是 V2.11 研究樣本

## V2.11

- 識別：`v211`
- 模型與 555 V2 相同，不另外訓練
- 職責：保存訊號、用下一交易日收盤模擬成交、在標籤成熟後評分
- 第一筆匯入訊號記錄時間：2026-11-05 23:23:09 +08:00；缺獨立進場前發布證據，不能僅憑Created_At認證可信任前瞻
- 60個成熟訊號日僅為運作檢查；不證明新聞增益或超額。程式內的升級旗標維持否

## 新固定設定的多變量延伸

- `scripts.research_extension`，兩候選：價格 LSTM 與六特徵情緒／風控 LSTM。固定 LSTM32、Dropout0.2、最多20epochs、patience3、seed42。
- 訓練756、驗證60、測試150個成熟訊號；六交易日標籤隔離，train-only scaler。
- 驗證橫斷面 Spearman 需比價格控制提高超過0.02才採情緒候選；先保存 selection/weights 再計 test。
- 可讀保存權重作 forecast，不擬合、不重新選模、不發布 live 訊號。結果仍屬已觀察歷史期間的探索。

## V3／0050新聞研究

- `scripts.research_v3`；價格／新聞Ridge與價格／新聞LSTM32，共四個固定候選，共用時序切分。
- 預測個股相對0050的下一收盤至其後五日含息超額百分點，三折成本後驗證選擇，最後60訊號只作探索性留出。
- 實測未通過新聞採用門檻，回到B0／0050；沒有合格新模型、沒有取代原權重或每日系統。
- 模型、Scaler、特徵順序、切分與失敗理由見[保存結果](reports/research_v3/2026-11-10/summary.md)。歷史新聞可用性仍受限。
