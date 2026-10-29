# 台灣五檔股票 444 / 555 每日自動化（研究用途）

本專案保留 **兩份各自獨立**的 Notebook：`444` 是單變量收盤價 LSTM 與固定股數等權重回測；`555` 是六特徵多變量 LSTM、新聞情緒與風險配置。**不將 444 / 555 合成一支模型程式**。每日由外層腳本執行和驗收，最後第三方比較器才讀取結果。

> **目前狀態**：已完成離線程式串接與測試，**尚未在 GitHub Actions 真實完成 TensorFlow 訓練或取得線上行情／新聞**。必須實際執行 GitHub 工作流程後才可確認模型結果與時效性。原始情緒資料僅到 2026-10-27。

## 一、每日自動流程

```text
GitHub Actions（台北時間 17:37，週一至週五）
  ├─ 查 XTAI 交易日曆／資料截止日（台北 15:00 後）
  ├─ prices.py：一次下載 5 檔 + ^TWII 原始收盤價，儲存共用 CSV
  ├─ news_update.py：鉅亨網新聞增量爬取／13:30 切分／假日順延
  │    └─ 中文情緒分類模型與逐篇新聞去重，更新舊有情緒 CSV
  ├─ 444_baseline.ipynb：單獨訓練／歷史回測／下一交易日預測
  │    └─ 444 專屬結果、保存預測與工作日誌
  ├─ 555_multivariate.ipynb：單獨訓練／歷史回測／動態配置
  │    └─ 555 專屬結果、保存預測與工作日誌
  ├─ compare.py：只比較相同測試日期 + 同一份價格快照的結果
  ├─ journal.py：個別檢討之前「真正存下來」的建議（不補做）
  └─ GitHub：提交輕量報表/每日情緒/預測歷史；大檔上傳 Actions artifact
```

| 檔案／目錄 | 功能 |
|---|---|
| `.github/workflows/daily-lstm.yml` | 每日排程、測試、執行、保存與失敗紀錄 |
| `notebooks/444_baseline.ipynb` | 你提供的 444，僅將路徑和日期參數化 |
| `notebooks/555_multivariate.ipynb` | 你提供的 555，僅將模型保存開關參數化 |
| `notebooks/original_...` | 原始新聞爬蟲、新聞分析、555 附加日誌，保留做追溯，不在主流程重複執行 |
| `scripts/prices.py` | 共用股價資料，避免 444、555 各抓到不同版本 |
| `scripts/news_update.py` | 新聞增量與去重；補齊新的每個 XTAI 訊號日 |
| `scripts/execute_nb.py` | **分開**執行兩個 Notebook；只略過 444 的重複繪圖/30 天情境 |
| `scripts/baseline_adapter.py` | 444 標準化輸出、下一日預測、真正的歷史建議存檔 |
| `scripts/compare.py` | 統一 ROI、Sharpe、MDD、方向準確率等指標並稽核 |
| `scripts/journal.py` | 每日兩份日誌；先前預測對答案 |
| `reports/latest/` | 供未來前端讀取的最新標準化 CSV 與 JSON |
| `reports/daily/YYYY-MM-DD/` | 每日摘要、比較結果與股票誤差表 |
| `reports/journals/{444,555}.md` | 兩份每日累積日誌 |
| `outputs/*/recommendation_history/` | 每次真正產生的預測，跨日檢討不能回頭改寫 |

## 二、放進 GitHub（一步一步）

1. 到 [GitHub 新增 repository](https://github.com/new)，建立例如 `tw-stock-lstm-research`。建議先選 **Private**，避免意外公開研究工作與原始資料。**不要勾選自動建立 README**，減少初次推送衝突。
2. 下載並解壓本專案 ZIP，將整個 `stock_lstm_automation` 資料夾用 Cursor 或 VS Code 開啟。請勿只上傳 `.ipynb`，因為 `.github/workflows/`、`scripts/` 和 `data/` 都是必需的。
3. 在 Cursor 終端機輸入以下指令，將 `你的帳號` 和 repo 名稱替換為自己的。若已安裝 GitHub Desktop，也可以選擇 `Add existing repository` 然後 Publish repository。

   ```bash
   git init
   git branch -M main
   git add .
   git commit -m "Initial separate LSTM 444 and 555 automation"
   git remote add origin https://github.com/你的帳號/tw-stock-lstm-research.git
   git push -u origin main
   ```

4. 確認 GitHub 上可以看到 `.github/workflows/daily-lstm.yml`。在 repo 的 **Actions** 頁面選 `Daily 444 vs 555 research pipeline`，按 **Run workflow** 手動測試。此操作會真的安裝 CPU 版 PyTorch、TensorFlow 並訓練兩組模型；若 GitHub 未授權 Actions，先在 Settings → Actions → General 開啟。
5. 在 **Settings → Actions → General → Workflow permissions**，允許工作流程能寫入 repository。YAML 內已設定 `permissions: contents: write`；若組織政策禁止寫入，會在最後 `git push` 階段失敗。
6. 工作成功後，在 `reports/latest/comparison.md` 看 **444／555／大盤** 的實測指標；在 `reports/journals/444.md` 和 `reports/journals/555.md` 看各自日誌；在 `reports/daily/日期/` 看當天報表。在該次 Actions → Artifacts 可以取得完整執行輸出（保存 14 天）。
7. 之後每週一到週五台北 **17:37** 自動觸發；若碰到休市，系統看 XTAI 日曆，並且同一個成功交易日不重訓兩次。GitHub 排程**可能延遲或偶爾不觸發**，不是 SLA。若需要補跑，回到 Actions 手動 Run workflow，選 `force=true`（強制重跑）或在原始預測沒完成時直接正常重跑。

> 手動補跑時若上游新聞 API、Hugging Face 或 yfinance 暫時離線，資料取得步驟會停止並標示失敗；**不會為了讓流程顯示成功而套用過期價格**。可稍後再次按 Run workflow，排查在 Actions 的 `logs/pipeline.log`。

## 三、本機測試（Windows / Cursor）

推薦 Python **3.11**（與 Actions 一致）。在 Cursor 終端機、專案根目錄：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dev.txt
python -m pytest -q tests
python -m scripts.pipeline --check
python -m scripts.pipeline
```

你現有的 Python 3.12.10 可先使用，但要先確認 TensorFlow、PyTorch 和所有套件在該環境成功安裝；若安裝相依套件衝突，改用專案建議的 Python 3.11 虛擬環境，不必刪掉本機原有 Python。

`python -m scripts.pipeline --check` 只檢查情緒資料和股價檔，不訓練模型。`--no-fetch` 可用於已有完整當天價格／情緒的離線重現；**不能對沒有股價或情緒的日期假裝跑完**。

## 四、指標、實驗公平性

- **ROI (%)**：結算後 NAV / 初始本金 − 1；兩版各自反映其回測策略（444 固定持有，555 每日動態配置），已計入其 Notebook 設定的交易成本。
- **Sharpe**：交易日 252 天年化，無風險利率預設 0，從同一組日淨值計算。
- **MDD (%)**：最大回撤，負值表示跌幅（例如 −20%）。
- **方向準確率 (%)**：LSTM 對隔日漲跌／平盤方向的預測正確率；**不是逐筆交易勝率**。
- **正報酬日比例 (%)**：稅費後投組每日淨值上漲日佔比；**也不是完成一買一賣的交易勝率**。
- **逐筆交易勝率**：444 沒有建立進出場交易策略，555 現在也沒有完整往返成交明細，因此不憑空填一個勝率，輸出保持 `NaN`。若組長要真正的「交易勝率」，兩份程式須先制定交易規則和完成交易紀錄。
- **五檔股票 MAE / RMSE / 方向準確率**：在 `reports/latest/per_stock_comparison.csv` 直接比較。
- **一致性稽核**：使用 `evaluation_protocol.json` 比對資料截止日、實測日期 SHA-256、價格矩陣 SHA-256、本金、交易成本、視窗長度等；有差異就**拒絕輸出比較**。

### 必須向教授說明的限制

1. **訓練資料不完全相同**：444 保留 2014 年起的原始單變量訓練來源；555 的新聞情緒特徵起於 2021 年，且有指標暖機期。因此這是相同測試期間與股價的**描述性比較**，不是能單憑結果證明「加新聞就一定較佳」的嚴格特徵消融實驗。若要做因果／消融聲稱，還要統一訓練樣本日期並在同一模型架構下做有無新聞的控制實驗。
2. **舊情緒模型來源混雜**：歷史情緒 CSV 來自你們先前的流程，舊筆資料未確認是否全用同一 NLP 模型量化。新資料採 `IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment`（中文正負情緒分類，並非專門訓練於台股金融新聞的 FinBERT）；**新舊分數可能有系統性差異**。建議最終論文統一重算 2021–2026 全部新聞的情緒，建立完整發布時間與來源稽核，再重跑 555。
3. **新聞源覆蓋**：本流程只向鉅亨網台股新聞類別抓取，`verified_no_matching_article` 僅表示該來源查詢成功且沒有符合條件的新聞，**不是證明全市場沒有新聞**。來源 API 可能改版或有授權、速率限制；遇失敗應修爬蟲而非補造資料。
4. **13:30 截止不等於可成交**：用 13:30 當天收盤價形成訊號並假設可以在同一收盤價交易，屬理想化 close-to-close 模擬；真實策略會有執行延遲、滑價、無法在收盤後成交的問題。要做可交易策略須兩版一致改成下一可成交時點。
5. **資源與 GitHub 限制**：十餘個 TensorFlow 模型 + NLP 分類器在 GitHub 免費 CPU runner 執行可能耗時或超出計算資源／儲存額度。已設 330 分鐘工作時間限制，若資源不夠建議自架 Runner／GPU 工作站。**Actions 的每日排程並不保證準時**。
6. **長期資料留存**：輕量的每日快照、預測、歷史情緒和比較報表會提交 Git；大容量模型檔預設不儲存，完整 runs 放 Actions Artifacts 只保存 14 天，重要版本請另備份並考慮 GitHub Releases／雲端物件儲存。若日後要從模型權重直接延續訓練，應增加模型檔儲存方案。

## 五、每日開發與資料重現約定

- `notebooks/` 是模型原始碼；`scripts/` 是外部串接和報告，彼此獨立。
- `data/sentiment_features_2026.csv` 是唯一正式新增情緒來源；`lstm_multivariate_features_2026.csv` 為舊稱，**不再維持兩份可能互相覆蓋的真實來源**。555 透過 `LSTM_FEATURE_CSV` 直接讀取唯一檔案。
- `data/prices/historical_prices.csv` 一次下載，兩個模型都讀同一 CSV，避免獨立抓股價時產生資料版本差異。
- 如果兩版模型檔要存成 `.keras`，可在本機設定 `LSTM_SAVE_MODELS=1`；預設 GitHub 不保存以免爆 repo。請勿直接用 git 提交巨大 ZIP、`.keras`、`.joblib`。
- 每次每日預測都依實際執行時間保存。**不會用今天重訓的模型補寫昨天原本不存在的預測**。
- 沒有提供外部交易券商憑證，也沒有下單 API；全程都只做研究與模擬比較。
