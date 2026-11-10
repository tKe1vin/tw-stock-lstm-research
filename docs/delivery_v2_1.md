# 2.1.0 完整程式交付與驗證

新交付名稱為 `TW_Stock_LSTM_Research_2.1.0.zip`，與原始 `TW_Stock_LSTM_Complete.zip` 分開。原 ZIP 保留歷史 bytes，不能代表目前 GitHub 版本。新版涵蓋目前共同核心 444／555、凍結 V2／V2.11、V3 程式與可公開的模型、結果、設定、測試及文件。

新 ZIP **不包含完整 FinMind 原始回覆或標準化行情**。供應商目前禁止對外再散布，V3 僅公開衍生成果與輸入雜湊；完整重播需要自行取得合法且雜湊一致的外部快照。程式／模型載入的驗證，與取得所有 V3 輸入後的數值重播，是不同的檢查。[V3 資料契約](../research/v3/data_contract.md)說明來源與限制。

## 建立新的封存檔

在已完成審查的 Git checkout，`VERSION.json` 版本必須為 `2.1.0`：

```bash
python -m scripts.build_delivery --output /your/delivery/TW_Stock_LSTM_Research_2.1.0.zip
```

命令只收集 Git 已追蹤檔與未被忽略的新檔，逐檔檢查來源、秘密資料與授權限制。Git 本體、環境／快取／日誌、所有舊 ZIP 及其 `.zip.sha256` 不進入新 ZIP。疑似憑證檔、高可信度憑證內容與 V3 的 provider raw／完整行情輸入會使建立失敗，不會印出憑證內容；`data/v3` 及 `finmind_*.csv` 等常見誤放位置也會拒絕。

檔案排序、ZIP 時間戳與權限固定，相同來源 bytes 可建立相同 ZIP bytes。內部 `PACKAGE_MANIFEST.json` 保存每個檔案的 SHA-256、大小及權限；外部 `.zip.sha256` 保存 ZIP 雜湊。既有輸出路徑不覆寫，來源在建立期間變動也會拒絕。

## 驗證下載與解壓內容

```bash
# 此步不需要 Git；--sha256 使用可信管道公布的雜湊
python -m scripts.build_delivery --verify /your/delivery/TW_Stock_LSTM_Research_2.1.0.zip --sha256 PUBLISHED_SHA256

# 解壓後在 TW_Stock_LSTM_Research_2.1.0/ 下執行
python -m scripts.build_delivery --verify-directory .
python -m pytest -q
python -m scripts.verify_package
python -m scripts.verify_models
python -m scripts.research_v3
```

支援 Python 3.12／CPU，依主 README 與固定的 `requirements-ci.txt` 安裝。ZIP 驗證會核對檔案內容、記錄權限及安全路徑。`--verify-directory` 不依賴 Git，可偵測檔案缺漏、新增、內容異動與路徑逃逸；解壓後不要求主機權限與 ZIP 的 POSIX 權限相同，一般 Python／pytest 快取不影響檢查。SHA 僅證明完整性，不單獨證明來源真實性或研究有效。

完整解壓測試應記錄實際 Python／套件版本、測試數量與失敗事項。使用既有虛擬環境進行乾淨解壓測試，不代表另一次全新依賴安裝成功。當天市場／新聞未就緒時，`run_all_daily --mode check` 可能如實等待，不能以離線測試通過冒稱已完成收盤實跑。

## 本輪實測

2026-11-10：checkout與獨立全新解壓目錄均 **326項測試通過**；解壓manifest、原權重／訊號hash、五股TensorFlow模型載入及V3預設唯讀入口通過。使用現成Python3.12 CPU環境，沒有冒稱在組長電腦全新安裝依賴。實際版本與範圍見[交付證據](audit_evidence/delivery_2026-11-10.json)。

此ZIP是2.1.0來源與資料狀態快照；GitHub Actions之後合法新增的每日狀態不回填封存檔。歷史重播的Git版本與V3合法外部固定來源仍依前述規格取得。

## 需要額外來源的功能

- 舊延伸模型的 `scripts.replay_extension_archive` 使用固定歷史 commit `b62034e4fa850933a632c8db6321b309ebad85f9` 隔離重播，需要包含該 Git object 的 checkout。ZIP 不含 `.git`，不能冒稱單靠 ZIP 即可執行這項歷史原碼重播；可另外取得 Git checkout。
- V3 真實行情／新聞重播需要合法且版本吻合的外部資料，並檢查公開來源雜湊。下載來源日後修訂時，精確重播應拒絕。
- 每日更新需要網路及當日完整來源，不包含券商下單整合。凍結權重、原模型卡與既有訊號仍以原始雜湊保護。

本文件描述交付方法；實際 ZIP 雜湊與乾淨解壓測試結果須以對應發行的驗證紀錄為準，不先宣稱未執行的檢查通過。
