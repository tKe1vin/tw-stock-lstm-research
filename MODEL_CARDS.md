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

## 555

- 識別：`555`
- 來源：`notebooks/555_multivariate.ipynb`
- 特徵順序：Close、MA20、RSI_14、MACD_Hist、Market_Vol_20d、Daily_Sentiment
- 訓練：每次執行重新訓練。`EPOCHS = 50`，`PATIENCE = 5`，`BATCH_SIZE = 32`，`SEED = 42`，`TIME_STEP = 10`
- 風控沿用 Notebook 內的動態曝險，股票總曝險在 Risk-Off 時上限 0.50
- 協議：`lstm-evaluation-v1`
- 已保存正式結果：截至 2026-11-04，淨報酬 11.953686872710323%
- 不讀取 555 V2 的 `weights.npz`

## 555 V2

- 識別：`555_v2`
- 版本：`rolling_selected_fold16_window756_carried`
- 結構：兩層 LSTM(50)、Dropout 0.2、Dense(1)、MSE、Adam 學習率 0.001
- 視窗：756 個交易日。早期停止樣本 60 日。時間步 10
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
- 第一筆可信任訊號：2026-11-05 23:23:09 +08:00
- 升級門檻：至少 60 個成熟訊號日，而且仍須由人決定。程式內的升級旗標維持否
