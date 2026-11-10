# 130 題稽核：訓練錯誤修正與證據邊界

日期：2026-11-10（臺北）。本紀錄涵蓋本次核查的訓練、切分與歷史證據問題，**不表示附件 130 題已全數通過**。V3 的新結果須另有其資料、選模、成本與新聞可用時間證據；舊結果不會因新測試通過而變成獨立驗證。

## 已修正且實際驗證

- **Q39、Q41、Q42、Q44：重複訓練的 Adam 狀態。** `experiments/v2/v28.py:Fitter.fit` 原先會將所有 optimizer variables 歸零，包含目前 Keras 的 `learning_rate`，讓第二次以後的 fit 不再更新權重。現在每次 fit 建立明確的 `Adam(learning_rate=0.001)`，重設迭代數與動量；只有明確提供 `previous` 才延續模型權重，跨股票／獨立折恢復原先種子權重。新輸出的 `Origin` 改為 `retrained_walk_forward_v28_optimizer_corrected`，不冒充原歷史生成版本。
- **Q43：最佳權重恢復。** 保留 `EarlyStopping(restore_best_weights=True)`；小型 TensorFlow 實測以恢復後的直接前向運算核對最小 validation loss。
- **Q10、Q41：舊訓練快取隔離。** 修正後的固定／排程訓練入口，在載入 TensorFlow 前拒絕舊 `Origin`、新舊混用、無 `Origin` 或預測／成本記錄不完整的快取；保留原 bytes，避免把已存的零學習率訓練當成修正後結果接續計算。
- **Q22、Q23：原始 444／六特徵 555 的 Scaler。** 從原始 notebook 只載入視窗建立函式，對驗證以後的特徵注入極端值，訓練矩陣與 Scaler 不變；原 notebook 未改寫。
- **Q14、Q16、Q17、Q18、Q26–Q28：切分重建。** 現行五日研究程式可重建 16 折，測試訊號自 2022-05-24 至 2026-03-17，測試訊號跨折不重複。標籤是 `100*(close[i+6]/close[i+1]-1)`，train／validation 的最後標籤必須早於下一區間。756 代表可用的訓練訊號／標籤數；連續訊號的 10 日輸入視窗聯集為 765 列，不能把三者混為一談。

實際執行：`python -m pytest tests/test_training_integrity.py -q`，**7 passed**（TensorFlow 2.20.0 CPU）。同一小網路連續兩次獨立 fit 與一次延續權重 fit，學習率均約 0.001、權重 L2 變動均大於 0.026，每次 optimizer iterations 均為 4；沒有訓練或替換任何正式模型。

完整可重算的檢查與逐折索引輸出：

```bash
python -m scripts.training_audit --include-indices --probe > training_audit.json
```

此命令只讀既有檔案，輸出至標準輸出；`--probe` 只用合成資料執行小型測試。16 折是對現行程式與保留來源的重建，不能代替缺失的原始執行紀錄。

另執行 `python -m scripts.training_audit --frozen-replay`，從 Git 保留的 `b62034e4fa850933a632c8db6321b309ebad85f9` 舊 Fitter 與新 Fitter 建立相同網路，直接載入同一凍結 NPZ／Scaler；**沒有 fit，也沒有停用正式 bundle 防護來發布訊號**。11/05、11/06、11/09 共 15 筆預測，新舊來源推論差為 0，與原發布值最大差約 `1.19e-7` 百分點，五股 Scaler 重建誤差均為 0。這支持本次 Adam 修正不改變現有凍結模型的推論；來源 manifest 仍須由明確維護步驟更新。此額外選項需要 Git 中保留該來源 commit；一般離線 ZIP 的檢查不依賴它。

## 不能追溯補證的問題

| 問題 | 目前可支持的結論與限制 |
| --- | --- |
| Q19、Q20、Q30、Q69、Q75、Q76 | 現有歷史被多次研究與檢視，屬探索；不能僅因程式不讀 test 選模就聲稱人為研究決策從未看過 test。新 V3 需先封存協議，再以未參與選模的新觀測檢視效果。 |
| Q32–Q36 | 歷史新聞缺首次觀測／取得時點的列，仍為 `retrospective_unverified`。Scalers／標籤隔離通過不證明新聞當時可得；不得補造時間戳。 |
| Q40 | 正式 444／原始六特徵 555 的 notebook 各自建立／編譯模型，沒有 V2.8 的 optimizer 全變數歸零迴圈；V2.9 已另用 fresh Adam。這個修正不認證其餘所有實驗的歷史訓練。 |
| Q46、Q47 | 固定種子不保證跨硬體／TensorFlow 完全相同；本次合成測試證明有更新，沒有給出所有研究模型跨平台重訓誤差界限。凍結權重推論需另外驗證。 |
| Q48、Q107、Q125 | 原 V2 凍結權重 SHA-256 維持 `53a1467fdba66e7f22a45e5f2fbc1cb78fc99825b15d516b3e7153d7bfc28144`；模型卡與已發布訊號未改寫。來源修正會使舊 bundle 來源雜湊失配，必須在明確維護更新、通過凍結推論回播後才能解除，不能每日自動洗入新雜湊。 |
| Q66、Q93–Q95 | `experiments/results/v2/v28/{predictions.csv,schedule_predictions.csv,schedule_cost.csv,fixed_predictions.csv}` 尚缺。+25.41%、+21.00%、58.1%／55.3% 及成本差的完整原始配對重算／不確定性仍不具足夠證據；不得虛構重建結果。其他原始預測曾在舊 ZIP 中保留，不能誤稱所有歷史預測都不存在。 |
| Q116、Q128 | 60 個成熟訊號日可檢查流程，不能單憑樣本數證明新聞模型勝出。局部 7 項測試也不代表全新解壓安裝或完整 130 題已驗收。 |

## 交易日曆查證

`scripts/common.py` 原先列出2026-07-10巴威及2026-10-23薔琵颱風休市但缺對應連結。本次已取得官方公告及交叉佐證，保留這兩個日期；另外修正日曆套件誤列2022-02-04為交易日的問題。沒有更改凍結資料、權重或已發布訊號。

- [期交所 2026-07-09 公告：7/10 巴威休市](https://www.taifex.com.tw/cht/11/newsDetail?newsType=1&idx=17137)。
- [期交所 2026-10-22 公告：10/23 薔琵休市](https://www.taifex.com.tw/cht/11/newsDetail?newsType=1&idx=17441)。
- [中央銀行 2026-07-10：巴威使臺北外匯市場休市](https://www.cbc.gov.tw/tw/cp-302-192572-f6e88-1.html)。
- [玉山證券重載證交所聲明：115年7月10日集中交易市場全日休市](https://m.esunsec.com.tw/news/instant-detail.aspx?id=%7B4C36DC2C-DFFB-4D60-8501-722CBF356D06%7D)。
- [MoneyDJ 重載證交所聲明：115年10月23日集中交易市場全日休市](https://www.moneydj.com/kmdj/news/newsviewer.aspx?a=9f12ad83-0500-422b-80ee-1328ad3f0ec5)。

上述期交所／央行公告各有明確市場範圍，不把它們當成證交所完整歷年交易日資料庫。每個原價資料缺口仍需逐筆查明；缺列不能自動當休市，停牌也不能自動用前價成交。

2022年春節由1/26封關至2/7開市，依[中央社引證交所公告](https://www.cna.com.tw/news/afe/202201260247.aspx)排除2/4。修正後的交易日表與本次獨立六資產資料的1,418個市場日期一致；保留[核查摘要](audit_evidence/v3_calendar_2026-11-10.json)，不是聲稱所有歷年市場均已認證。

## 其餘本輪程式修正

- 新聞：實際每日共用途徑記錄模型與Tokenizer確切checkpoint、評分設定及執行版本；舊列保持未驗證。修正13:30後微秒歸日錯誤，拒絕不合法分數、時間順序及來源雜湊。
- 指標：零預測不當作看跌；方向採負／零／正一致定義。拒絕非有限值、不完整五股橫斷面及不配對日期／標籤，不再默默刪樣本。原延伸600筆驗證重算仍為0.31／0.36，採用不變。[證據](audit_evidence/metrics_integrity_2026-11-10.json)。
- 帳本：Git已提交的seed之後行情、成熟評分、成交與NAV僅允許合法追加。所有候選列先驗證才寫入，重跑不重寫舊bytes；新輸出另有來源hash。無Git的首次checkpoint不追溯認證舊資料。[證據](audit_evidence/ledger_integrity_2026-11-10.json)。
- 舊報告：現行程式維持來源變動拒絕；以固定Git版本在隔離暫存目錄重播，不改歷史manifest、不重訓。來源已修正不代表舊報告可直接以新定義重新宣稱有效。

130題逐題狀態及未解證據邊界見[登錄](audit_130_status.md)；V3實測仍未取得合格新聞候選。
