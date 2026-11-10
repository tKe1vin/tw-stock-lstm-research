# 444／555 共同核心配對研究

資料截止：2026-11-10。本頁為主研究；原六特徵 555 的 Notebook 輸出另保留為延伸／歷史診斷。

全部配置組使用同一份 444 預測、次一交易日收盤、手續費 0.1425%、賣出稅 0.3%及共同期末處理。
歷史新聞未有完整取得證據，且期間已被查看；結果皆為探索性，不宣稱因果已驗證或前瞻績效。

| 組別 | 淨 ROI | Sharpe | MDD | 成本 |
| --- | ---: | ---: | ---: | ---: |
| static_equal444 | 20.01% | 1.552 | -13.54% | 675.72 |
| dynamic_no_news | 4.21% | 0.851 | -3.37% | 6660.83 |
| dynamic_news_no_risk | 3.92% | 0.694 | -4.57% | 10191.03 |
| dynamic_no_news_risk | 2.05% | 0.477 | -3.56% | 6326.93 |
| dynamic_news_risk555 | 0.77% | 0.190 | -4.35% | 8907.98 |
| equal_rebalanced_reference | 20.91% | 1.648 | -12.60% | 1282.03 |
| risk_matched_equal_reference | 6.39% | 1.395 | -2.99% | 5766.95 |
| market_price_reference | 28.75% | 1.524 | -20.31% | 0.00 |

同風控下新聞組與無新聞組的 ROI 差：-1.29 個百分點。這是配對策略差異，尚無統計顯著或歷史新聞因果認證。

444 固定持股的 ROI 不代表 LSTM 擇時獲利。模型的一日預測在次日收盤進場前已完成，其配置屬延遲訊號策略；首段漲跌未計入新部位收益。
^TWII 為無股利的理論價格參考；該列不扣交易成本，不視為可交易淨報酬。另列同交易機會及同政策總曝險等權組。

[逐日 NAV](nav.csv) · [成本與成交](fills.csv) · [權重](weights.csv) · [555 主研究當日配置](555_next_day_recommendation.csv)

[研究協議](protocol.json) · [永久輸入快照與程式雜湊](reproduction_manifest.json)

[固定配對效果與區塊重抽樣區間](analysis/contrasts.csv) · [按持有部位風險狀態分析](analysis/regime_effects.csv)

重播：`python -m scripts.paired_delivery --replay 本目錄/reproduction_manifest.json --output 新目錄`。重播讀取保存預測，不重新訓練。
