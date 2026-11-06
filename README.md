# 台股 444／555／555 V2／V2.11 每日研究自動化

本版本整合原正式版的新聞與 GitHub 自動化，以及 Complete 套件的凍結候選與研究帳本。四個系統各自保存輸出；新聞與行情只更新一次。

| 系統 | 工作 | 每日輸出 |
| --- | --- | --- |
| 444 | 原始單變量 LSTM，獨立重新訓練；固定持股參考 | `outputs/444/日期/` |
| 555 | 原始六特徵 LSTM、新聞情緒及動態風控，獨立重新訓練 | `outputs/555/日期/` |
| 555 V2 | 使用已凍結的 rolling_selected 權重預測五日報酬，不重訓 | `outputs/555_v2/日期/` |
| V2.11 | 同一凍結模型的不可覆寫訊號、次日收盤模擬成交、成熟標籤 | `experiments/results/v2/v211/` |

## GitHub 每日執行

工作流程：[Daily four-system LSTM research](../../actions/workflows/daily-lstm.yml)。使用 Ubuntu CPU、Python 3.12。

- 週一至週五台北 **17:37** 執行，**21:37** 再試一次。GitHub 排程可能延遲。
- 以 XTAI 日曆檢查交易日；當日尚未收盤或休市時跳過，不把上個交易日冒充今天。
- 先共用更新市場與新聞，再依序處理 **V2.11 → 555 V2 → 444 → 555**，讓研究訊號先保存。
- 同交易日已成功的模型預設不重訓；V2.11 仍處理成交和已成熟標籤。`force` 僅強制重跑模型輸出，不能改寫已發布訊號。
- 其中一步失敗仍保留其他合法結果。精簡市場狀態、新聞快取、訊號、預測歷史與報表提交 Git，完整執行檔與日誌上傳 Actions artifacts，保留 30 天。
- 資料等待或程式失敗會在 Actions 如實顯示失敗，摘要區分 `waiting` 與 `failed`。

第一次部署或修改程式也會觸發檢查；資料與報表的自動提交不會循環觸發。需要手動補跑時，到 Actions 選取工作流程，再按 **Run workflow**。必須允許 GitHub Actions 執行，且 workflow 的 `contents: write` 不被帳號或組織政策禁止。

## 結果在哪裡

- [`reports/latest/summary.md`](reports/latest/summary.md)：四系統狀態與當日結果摘要。
- [`reports/latest/four_systems.json`](reports/latest/four_systems.json)：各系統日期與完整性。
- `reports/daily/YYYY-MM-DD/`：每日摘要、444／555 的同協議比較。
- `reports/journals/444.md`、`555.md`：個別研究日誌，只核對當時真正保存的預測。
- `outputs/{444,555}/recommendation_history/`：跨日保存的一日預測。
- `experiments/results/v2/v211/daily/`：原始不可覆寫研究訊號。
- Actions 每次執行的 artifacts：完整 Notebook runs、圖及日誌。

摘要會明確標示當日尚未完成的系統；過去成功結果保留原日期，不會被標成今天。444／555 的 `lstm-evaluation-v1` 與 V2 的下一交易日收盤帳本分開解讀，不將四者 ROI 混排。

## 共用資料與新聞

唯一每日市場來源為 `experiments/results/v2/v211/market/`，當日驗證通過後複製到 `data/snapshots/交易日/`，供 444／555 使用；V2 模型與帳本讀同一來源。更新市場的子程序結束後才啟動模型，釋放 NLP 記憶體。

新聞沿用正式版的 **鉅亨網台股類別＋Erlangshen 中文情緒分類模型**：逐篇去重，分數為 P(positive)−P(negative)，同股票同訊號日取平均。13:30 後的新聞歸到下一個 XTAI 交易日。抓取涵蓋前一交易日至當日的完整日曆區間，包含週末與長假，成功後記錄 `news_scan_coverage.json`。

只有查詢完整且沒有符合股票的文章時才記為零分／`verified_no_matching_article`；來源失敗保持等待，不沿用昨日情緒。這不代表全市場沒有新聞。保留原有股票篩選及情緒模型，沒有混入歷史 SnowNLP 原型或重算舊分數。

保留 GitHub 正式資料截至 2026-11-05 的版本，以及相容的逐篇新聞快取。共用市場沿用 V2.11 已修補的歷史交易日（例如 2025-08-01），因此新版每日 555 的有效訓練日期可能比舊正式來源多，不能承諾數值完全不變；原 11/04 重播輸入另外封存。`data/prices/historical_prices.csv`、`data/sentiment_features_2026.csv` 作為匯入基準，不由新版每日流程改寫；每日新增資料在上述 canonical 市場帳本。

## 權重與原始訊號

凍結權重 SHA-256：`53a1467fdba66e7f22a45e5f2fbc1cb78fc99825b15d516b3e7153d7bfc28144`。

2026-11-05 原始研究訊號 SHA-256：`37591d21300977b9486212fc86fe257bb050db47177fa2f41bd7a8904919f659`。它來自 Complete 套件所保存的開發者紀錄，記載發布時間為 2026-11-05 23:23:09 +08:00；不把本次 GitHub 上傳時間當成原發布時間。

預設 `research/v211/config/ledger.yaml` 使用 `imported_research`，在此儲存庫接續該封存帳本。`new_independent` 使用獨立訊號帳本、共用凍結權重與經前綴驗證的市場鏡像，不匯入 11/05 訊號。請只指定一個儲存庫／runner 作為研究帳本寫入端，不合併多台電腦各自發布的帳本。

## 本機操作與驗證

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install 'torch==2.15.0' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-ci.txt
python -m pytest -q
python -m scripts.verify_models
python -m scripts.run_all_daily --mode check
python -m scripts.run_all_daily --mode full
```

`check` 為唯讀：不下載、不訓練、不寫預測或帳本。結束碼 0 表示就緒，2 表示資料尚未到齊，1 表示錯誤。`full --skip-download` 僅使用已存在的完整快照。舊指令 `python -m scripts.pipeline` 也轉到四系統入口。

可重現的 2026-11-04 歷史輸入另放在 `data/examples/reproducible/`，雜湊與 Complete 套件一致：

```bash
python -m scripts.run_model --model 444 --mode full --fixture reproducible
python -m scripts.run_model --model 555 --mode full --fixture reproducible
```

歷史重播與正式跨日預測歷史分開，不回填研究訊號。`tests/legacy_delivery/` 保留交付當下的舊快照測試，預設不執行；目前 CI 使用固定時間與隔離資料的測試。

## 研究結果的限制

444 的固定持股 ROI 不代表 LSTM 擇時獲利。555 採已揭露的理想化同日收盤成交假設。V2 的探索性歷史結果尚未優於可比固定等權；最新凍結權重的成效須看後續訊號，不能挪用滾動研究的報酬。

歷史情緒大部分缺乏完整新聞來源與當時可取得時間；新資料的稽核欄位不能追溯證明舊分數。股價使用未調整 Close，未另計股利現金流；價格基礎績效不等於含息總報酬。V2.11 至少 60 個成熟訊號日只是人工審視門檻，程式不會自動更換正式 555。

完整歷史探索的中間預測大檔保留在原始 ZIP；GitHub 保留研究報告、程式、必要市場狀態與凍結權重。

沒有券商下單介面。`notebooks/original_*` 保留來源追溯，不在每日流程重複執行。
