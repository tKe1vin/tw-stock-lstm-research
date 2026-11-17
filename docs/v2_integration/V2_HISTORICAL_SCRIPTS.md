# V2.6～V2.10 歷史腳本分類（不整包複製）

比對來源：research／development `experiments/v2/*` vs 本整合分支（= main 既有 + dashboard_v2）。

| 腳本 | 判定 | 說明 |
| --- | --- | --- |
| `v26.py` | 歷史探索／結果保留 | 市場狀態＋Ridge／ElasticNet 候選篩選；非正式入口 |
| `v27.py` | 歷史探索／結果保留 | F1／F2 尺度無關特徵；不進正式執行鏈 |
| `v28.py` / `v28_rules.py` | **已在 main** | 滾動視窗規則與準備；V2.11 依賴 |
| `v29.py` | **已在 main** | `CorrectedFitter`；凍結推論依賴 |
| `v29b.py` / `v29b_gpu.py` | 歷史訓練探索 | Adam LR／GPU 重跑；結果目錄唯讀 |
| `v29c.py` / `v29c_profile.py` | 歷史訓練探索 | 加速訓練；**視窗選擇來源**已寫入 `model_card.window_source` |
| `v210.py` | 歷史稽核／影子追蹤 | 不訓練、不取代 555；現行入口為 V2.11 |
| `train.py` / `run.py` | 研究編排輔助 | 非正式凍結入口 |
| `v211*.py` | **已在 main** | 正式前瞻／市場／排程 |

## 移植決策

- **本次未移植** v26／v27／v29b／v29c／v210／train／run（非執行必要）。  
- 必要方法學已沉澱於：凍結 `model_card.json`、`v28_rules`、`v29.CorrectedFitter`、`v211.predict_rows`。  
- 歷史績效數字不得改寫，亦不得宣稱為正式前瞻績效。
