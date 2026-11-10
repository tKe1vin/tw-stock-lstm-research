# 台股 LSTM、新聞情緒與動態配置研究

本專題研究新聞與市場風險資訊能否改善 LSTM 投資組合的風險調整後績效。2026-11-09 修訂以 **相同 444 預測核心**建立 555 主研究，控制模型、價格、日期、成交與成本，再分離動態配置、新聞與風控效果。既有凍結權重與已發布訊號保留原始 bytes。

- [V3：0050核心＋新聞超額模型](research/v3/design.md) · [實測與拒絕採用紀錄](reports/research_v3/2026-11-10/summary.md)
- [130題逐題狀態](docs/audit_130_status.md) · [訓練關鍵修正與歷史限制](docs/audit_130_critical.md)
- [修訂企畫書](docs/proposal_revised.md) · [Word 版本](docs/proposal_revised.docx)
- [研究協議](docs/research_protocol_v2.md) · [十項 P0 改善與歷史限制](docs/p0_remediation.md)
- [修訂後配對研究實測](reports/research_revision/2026-11-06/paired/comparison.md)
- [固定設定的多變量 LSTM 延伸](reports/research_extension/2026-11-06/summary.md)
- [每日狀態](reports/latest/summary.md) · [驗證紀錄](VALIDATION.md)

## 每日功能

| 系統 | 工作 | 輸出 |
| --- | --- | --- |
| 444 | 保留原單變量 LSTM 訓練與預測；主研究固定持股基準由共同帳本重算 | `outputs/444/日期/` |
| 555 | **444 相同預測＋動態權重＋新聞＋風控**；不另外訓練不同模型作主比較 | `outputs/555/日期/`、`reports/daily/日期/paired/` |
| 555 V2 | 沿用原凍結 rolling_selected 五日權重，只輸出預測 | `outputs/555_v2/日期/` |
| V2.11 | 不可覆寫訊號、次日收盤模擬成交、五日排程及成熟標籤 | `experiments/results/v2/v211/` |

原六特徵 555 Notebook 完整保留，使用 `--legacy-555` 執行，屬另一項模型延伸／歷史診斷。`--fixture reproducible` 仍重播原 444／六特徵 555，不把歷史重播回填為前瞻訊號。

GitHub Actions 維持台北週一至週五 **17:37** 執行、**21:37** 重試。以 XTAI 及經來源確認的特殊休市日判斷當日收盤；行情與新聞只更新一次，順序仍為 **V2.11 → 555 V2 → 444 → 555**。同日合法成功結果可重用，舊六特徵 555 不冒充新主研究。部分失敗仍保存其他合法結果；`check` 保持唯讀，等待資料回傳 2、錯誤回傳 1、完成回傳 0。沒有券商下單介面。

## 公平比較與研究結果

主研究提供固定持股、動態無新聞、動態有新聞、各自有／無風控，以及再平衡等權與相同政策曝險等權。所有股票組使用同一份 444 預測、次日收盤、手續費 0.1425%、賣出稅 0.3%、同一價格及期末處理。當日收盤後算出的權重不在同一收盤成交。

444 原 Notebook 的舊固定持股 ROI 與六特徵 555 的舊同收盤 ROI 保留作歷史紀錄。**主比較使用 paired 帳本重新計算兩者，不將新舊協議混排。**^TWII 為理論價格指數參考；不能將其視為可交易 ETF 或含息報酬。

截至 2026-11-06 的共同核心重播：444 固定持股淨 ROI **18.32%**，完整新聞／風控配置 **−2.93%**，相同政策曝險等權 **7.12%**。新聞未改善此固定配置的淨報酬；研究另提供換手、成本、風險狀態與固定區塊重抽樣分析。

延伸研究只訓練預先固定的價格 LSTM／情緒風控 LSTM 兩候選，先用驗證集選擇並保存權重，再計算測試預測。2026-03-23～11-02 的新聞候選 **16.18%**，可比風控等權 **16.83%**，理論大盤成本代理 **36.47%**。**尚未超越大盤；未因測試落後而重新調參。**該延伸沒有取代既有 V2 凍結模型。

2026-11-10新增V3，依使用者指定改與可交易的**0050**比較。已取得獨立原價／公司行動資料，實作新聞與價格Ridge／LSTM配對、整數股及股利付款日帳本；三折固定驗證均未取得合格新聞候選，採用結果為B0／0050。這不是已完成勝過0050的模型，最後60訊號仍屬探索性留出。實測、模型與拒絕理由皆保留，未替換原每日系統。

以上期間已被查看，歷史新聞也缺完整取得證據，全部屬探索性。較低 MDD、較高 Sharpe、新聞效果與交易成本的取捨都可作研究結果；不預設顯著或必然勝過大盤。

## 時點、來源與凍結保護

新聞使用鉅亨網台股類別及 `IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment`，分數為 P(positive)−P(negative)。13:30 後發布歸下一個 XTAI 訊號日；新增逐篇來源側錄，保存首次取得、完成時間、評分可用時間、內容／payload 雜湊及版本。合法收盤後取得可用於當晚決策；決策後才可用的資訊會被拒絕。來源失敗與經完整查詢的無匹配文章分開。

歷史無證據分數明標 `retrospective_unverified`，不補造時間。11/05 匯入訊號的發布時刻為開發者紀錄，缺可獨立核實的進場前證據；11/06 舊訊號的缺欄以 `signal_audit.json` 說明，不改寫原檔。

凍結 NPZ SHA-256 維持 `53a1467fdba66e7f22a45e5f2fbc1cb78fc99825b15d516b3e7153d7bfc28144`。完整 bundle 驗證模型卡、Scaler、特徵順序、架構、tau 與推論程式；每日流程只驗證，不自行重建 manifest。

`forecast_entry_date` 是每日預測的標籤進場日；`scheduled_rebalance_date` 與 `trade_eligible` 才說明帳本交易資格。11/06 預測可標假設 11/09 進場，但五日帳本當天不交易；11/13 必須使用 11/12 新訊號，不能沿用舊訊號。五日標籤為 `close[i+6]/close[i+1]-1`，切分／校準使用成熟日隔離六個交易日。

## 執行與重現

Python 3.12，CPU。正式 Actions 使用固定 `requirements-ci.txt`。

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install 'torch==2.15.0' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-ci.txt
python -m pytest -q
python -m scripts.verify_package
python -m scripts.verify_models
python -m scripts.run_all_daily --mode check
python -m scripts.run_all_daily --mode full
```

V3獨立入口預設唯讀：`python -m scripts.research_v3`。訓練與重播需依[資料契約](research/v3/data_contract.md)合法取得外部固定行情；公開Git與ZIP不含受限制的FinMind完整行情快照。

單獨執行 555 前須先有同日成功的 444 與完整快照；缺資料會等待，不用昨日核心替代。`--force` 不得改寫 V2.11 已發訊號。

```bash
python -m scripts.run_model --model 555 --mode full --legacy-555
python -m scripts.run_model --model 444 --mode full --fixture reproducible
python -m scripts.run_model --model 555 --mode full --fixture reproducible
python -m scripts.replay_paired_archive --manifest reports/research_revision/2026-11-06/paired/reproduction_manifest.json
python -m scripts.replay_extension_archive --run reports/research_extension/2026-11-06
python -m scripts.replay_extension_archive --run reports/research_extension/2026-11-06 --mode forecast --as-of 2026-11-06
```

配對研究的原始預測、價格、情緒、協議及建議以內容雜湊永久壓縮保存於 `data/research_snapshots/`，可逐位元重算投資組合 CSV；這不等於宣稱神經網路重新訓練必然逐位元一致。原始模型訓練檔、權重與日誌在 Actions artifacts 保留 **90 天**。延伸模型另保存本次訓練權重、Scaler、輸入、原始／修正來源及重現驗證。

修正現行程式後，舊報告仍嚴格檢查原來源雜湊。上述歷史重播命令在暫存目錄使用固定Git版本，不修改舊manifest、不重訓；需要Git與該歷史commit。新V3使用自身保存模型與外部固定輸入重播。

原 `TW_Stock_LSTM_Complete.zip` 是舊交付版本；新版交付與逐檔驗證見[2.1.0說明](docs/delivery_v2_1.md)。部分v28中間預測仍缺，不能宣稱舊歷史比較已完整重現；原研究價格已能以雜湊相同的 `data/examples/reproducible/prices.csv` 解析。`tests/legacy_delivery/` 保留舊快照測試，預設CI不執行。

若先前555匯出失敗，`python -m scripts.repair_saved_555 --as-of 資料日 --mode check` 僅查驗保存核心與當時快照；`--mode repair` 僅接受失敗且沒有已發布建議的匯出，明標歷史重播並保留原失敗，不回填前瞻訊號或成交。
