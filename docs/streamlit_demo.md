# Streamlit 初步結果展示

此頁展示已保存的探索性研究，不訓練、抓行情、改選模型或覆寫每日系統。入口為 [`dashboard/app.py`](../dashboard/app.py)，根目錄的 `streamlit_app.py` 也可啟動相同頁面。

## 啟動

建議獨立使用 Python 3.12 虛擬環境，無須安裝每日訓練的 TensorFlow／PyTorch 套件或提供 FinMind 金鑰：

```bash
python3.12 -m venv .venv-dashboard
. .venv-dashboard/bin/activate
python -m pip install -r dashboard/requirements.txt
python -m streamlit run dashboard/app.py
```

在 repository 根目錄執行，開啟終端顯示的網址，預設為 `http://localhost:8501`。Windows 可用 `.venv-dashboard\Scripts\Activate.ps1` 啟用環境。

若自行使用 Streamlit Community Cloud，選擇本 repository、`main` 分支與 `dashboard/app.py` 入口；套件清單位於同目錄 `dashboard/requirements.txt`。本次交付提供可執行程式，不代表已建立公開託管網址。部署應保留完整 checkout 的報告與展示快照；不需上傳私人行情快取或任何金鑰。

## 展示內容與目前結論

五個分頁提供結果總覽、配對驗證、成本與風險、每日運作及下一輪研究。側欄切換研究版本；總覽可切換成本後淨值、相對0050財富與回撤，也可加入等權控制。底部可下載保存的績效及逐折驗證CSV。

| 研究 | 實際留出帳本期間 | 新聞 Ridge 淨報酬 | 同期0050淨報酬 | 採用結果 |
| --- | --- | ---: | ---: | --- |
| V3.3／20交易日 | 2026-04-15～11-09 | 13.0049% | 15.9048% | B0／0050；新聞拒絕採用 |
| V3.2／5交易日 | 2026-08-04～11-03 | −0.5767% | −0.5767% | B0／0050；新聞拒絕採用 |

V3.3新聞比同架構價格控制高0.1181個百分點，仍比0050低2.8999個百分點。三折驗證未通過，留出結果只作診斷，不因此改選。V3.2與V3.3使用不同期間及協議，不能用兩個ROI排序當成進步證據。等權控制超過0050也不能歸因於新聞。

歷史新聞缺完整首次取得證據、五檔股票缺研究前選定證據、次日收盤成交是理想化假設；已看過的歷史全部屬探索。下一輪設計詳見 [V3研究討論](v3_next_research.md)，尚未實作或產生新績效。

## 資料來源與保護

- 績效、逐折驗證、選擇紀錄及設定直接來自 [`reports/research_v3_3/2026-11-17/`](../reports/research_v3_3/2026-11-17/) 與 [`reports/research_v3/2026-11-10/`](../reports/research_v3/2026-11-10/)。
- 曲線快照只含日期、組別、衍生投資組合淨值指數與回撤，位於 [`dashboard/assets/`](../dashboard/assets/)。不含FinMind原始／完整標準化行情、逐股報價、新聞全文或私人來源路徑。
- 淨值是已保存帳本NAV除以共同初始資金再乘100，保留進場成本；回撤以初始100計入高水位。相對財富是策略淨值除以0050淨值減一，與報酬百分點差不同。
- `manifest.json` 記錄快照、公開報告與私人研究來源的雜湊鏈。載入時檢查來源雜湊、四組完整共同日期、有限正淨值、ROI及最大回撤一致性；缺檔或不一致會停止顯示績效，不用範例值代替。
- 每日運作讀取checkout的 `reports/latest/four_systems.json`，分開顯示資料日與真正執行日。頁面按鈕僅重新讀取本機報告；最新每日狀態需更新checkout或重新部署。

原始報告、凍結模型、訊號與每日排程未被展示頁修改。離線查看不需要外部資料服務；GitHub文件連結在點擊後才需要網路。

## 驗證與展示快照重建

```bash
python -m pip install 'pytest>=8,<9'
python -m pytest -q tests/test_dashboard_data.py tests/test_dashboard_app.py
python -m scripts.export_dashboard_data
```

最後一個命令預設唯讀，只檢查已保留展示資料。測試涵蓋曲線與原績效一致、雜湊／缺檔阻擋、版本與圖表互動，以及操作前後研究檔案不變。獨立的 `dashboard.yml` CI安裝輕量展示依賴執行這些測試；沒有Streamlit的每日環境可略過UI測試。

2026-11-17本機驗證：全套416項通過，展示專用19項通過；凍結權重、模型卡與完整bundle雜湊均維持原值。測試不代表新聞模型已證明勝出。

只有持有合法保留的兩版私人研究目錄，且雜湊與原公開來源一致時，才能顯式匯出到**新的空目錄**：

```bash
python -m scripts.export_dashboard_data --write \
  --v33-run /private/retained-v33-run \
  --v32-run /private/retained-v32-run \
  --output-dir /tmp/new-dashboard-snapshot
```

這些是自行替換的私人目錄範例。匯出器拒絕覆寫既有快照，不更動舊來源manifest，也不複製受限制的行情快取。新增研究版本需要另行審核與登記，不能為了畫出勝出曲線而刷新來源。
