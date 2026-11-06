# 資料字典

## 價格

`data/prices/historical_prices.csv` 與 V2.11 市場帳本的 `prices.csv` 都使用：

| 欄位 | 意義 |
| --- | --- |
| Date | 交易日 |
| Close | yfinance `auto_adjust=False` 的未調整收盤價 |
| ticker | `2330.TW`、`2317.TW`、`2603.TW`、`2002.TW`、`2891.TW` 或 `^TWII` |

可重現測試資料另封存於 data/examples/reproducible/，截止 2026-11-04。GitHub 正式匯入基準截至 2026-11-05。每日新增資料使用 V2.11 市場帳本，經驗證後複製為共用快照。

## 情緒

`data/sentiment_features_2026.csv` 欄位：

| 欄位 | 意義 |
| --- | --- |
| Date | 訊號日 |
| ticker | 五檔股票之一 |
| Daily_Sentiment | 當日情緒分數 |
| News_Count | 計入該訊號日的新聞數 |
| Latest_News_Timestamp | 納入新聞的實際發布時間 |
| Source_Status | `observed_article` 或 `verified_no_matching_article` |
| Scoring_Model | 評分模型識別 |

13:30 以後的新聞歸到下一個交易日。抓取未完成時，這一天保持未就緒，不用前一日分數填補。

## 555 V2 預測日期

`outputs/555_v2/執行日/prediction.csv` 與 `dates.json` 把四個日期分開：

| 欄位 | 中文 | 規則 |
| --- | --- | --- |
| Signal_Date | 訊號日 | 已完成交易日 |
| Run_Date | 執行日 | 程式寫出檔案的台北日曆日 |
| Data_AsOf | 資料截止日 | 必須等於訊號日 |
| Planned_Execution_Date | 預定成交日 | 下一個交易日收盤 |

統一 CSV 的對應欄位是 `signal_date`、`run_date`、`data_asof`、`intended_execution_date`。`predicted_return_pct` 是百分點。`0.677330` 代表 +0.677330%。權重是資金比例。沒有成交價時寫「不適用」。

## V2.11 訊號

`experiments/results/v2/v211/daily/2026-11-05.csv` 的 `Created_At` 是原始發布時間。`Execution_Price` 與 `Actual_5d_Pct` 在發布時留空。成交與標籤寫在帳本的其他檔案，不回寫這份訊號。

## 正式 444 與 555

Notebook 輸出的 `Signal_Date` 是資料截止日，`Target_Date` 是 Notebook 自己的目標日。協議名稱是 `lstm-evaluation-v1`。這些欄位會原樣進入統一 CSV，不改單位。
