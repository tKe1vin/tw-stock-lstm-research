# 資料字典

## 價格

`data/prices/historical_prices.csv` 與 V2.11 市場帳本的 `prices.csv` 都使用：

| 欄位 | 意義 |
| --- | --- |
| Date | 交易日 |
| Close | yfinance `auto_adjust=False` 回傳的 Close；此選項本身不證明歷史為名目原價，也不認證股利／分割處理 |
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

嚴格晚於13:30:00的新聞歸到下一個交易日，13:30:00本身可歸當日。抓取未完成時，這一天保持未就緒，不用前一日分數填補。新逐篇sidecar另記錄實際可用時間、來源與精確模型／Tokenizer checkpoint；舊列不因欄位新增而補成已驗證。

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

`experiments/results/v2/v211/daily/2026-11-05.csv` 的 `Created_At` 是開發者匯入檔記錄的時間，缺可獨立查證的進場前公開證據。`Execution_Price` 與 `Actual_5d_Pct` 在該檔留空。成交與標籤寫在帳本的其他檔案，不回寫這份訊號。

## V3獨立資料

V3不沿用舊Close作已認證的原價。新契約使用明確名目OHLCV、Tradable、Stock_Splits、Dividends、Return_of_Capital及付款日，見[資料契約](research/v3/data_contract.md)。含息研究指數與實際可用現金帳本分開；FinMind完整行情只在授權本機快照，公開結果保留雜湊與彙總，clone不包含全部輸入。

## 正式 444 與 555

Notebook 輸出的 `Signal_Date` 是資料截止日，`Target_Date` 是 Notebook 自己的目標日。協議名稱是 `lstm-evaluation-v1`。這些欄位會原樣進入統一 CSV，不改單位。

## 統一 CSV v2

新輸出 metadata 的 schema_version 為 unified-signal-v2；欄位見 common/schemas/unified_signal_v2.json。保留所有舊欄位名稱，新增 forecast_entry_date、scheduled_rebalance_date、trade_eligible。舊 v1 檔案與其 schema 保留不改；讀取時依欄名，不以固定欄位位置猜意義。execution_status=historical_replay_no_order 的補正輸出不列入前瞻訊號或成交。
