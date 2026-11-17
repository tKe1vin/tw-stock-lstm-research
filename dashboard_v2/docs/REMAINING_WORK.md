# 後續工作（Phase 8 後）

## 收盤後建議（人工）

1. 等台北 15:00 後，讓既有 `\TwStockLstmV211Daily`（約 18:30）跑完  
2. 確認本機 `prices_through`／`matured_scores` 是否納入 11/09  
3. 再執行 `safe_display_sync.py`  
4. 重整 8502  

## 需組長批准

1. GitHub 午夜政策合入正式 Actions  
2. 是否保存每份訊號當下的凍結 `sentiment.csv` 快照（以便衝突重播）  
3. Commit／Push group-clone 的 dashboard／audit  
4. 本機與 GitHub runner 程式對齊策略（目前四檔 v211* 皆不同）

## 不做

- 不重訓、不改凍結權重、不自動裝新排程、不回填錯過訊號
