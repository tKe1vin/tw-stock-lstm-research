# 新聞與風控 LSTM 延伸候選

採用候選：`sentiment_risk_lstm`；採用只看驗證集，兩個候選與超參數事先固定。
本結果為已觀察歷史區間的探索性研究，不是獨立樣本外 alpha 證據。

| 策略 | ROI (%) | Sharpe | MDD (%) | 相對可比等權 (pp) |
| --- | ---: | ---: | ---: | ---: |
| price_lstm | 17.456 | 2.384 | -5.520 | 0.623 |
| sentiment_risk_lstm | 16.182 | 2.246 | -6.126 | -0.651 |
| matched_risk_equal | 16.833 | 2.283 | -6.446 | 0.000 |
| full_exposure_buy_hold | 27.017 | 2.006 | -13.471 | 10.184 |
| theoretical_twii_with_stock_cost_proxy_not_an_etf | 36.474 | 1.890 | -16.355 | 19.641 |

所有策略帳本起迄：2026-03-23 至 2026-11-02。最後日為最後一筆排程可成交訊號的五日持有到期日，僅平倉，不先建新倉。
終點修正依五日持有規格執行，不重訓、不重選模型、不調參；最初多持有四日的錯誤報告留在 [initial_terminal_extension_bug/](initial_terminal_extension_bug/)，不得當成正式結果使用，詳細差異見 [endpoint_correction.json](endpoint_correction.json)。
大盤列為理論指數加股票成本代理，並非可交易 ETF。所有股票帳本使用相同次日收盤、五日排程、成本及訊號日風控。
保留原始輸入、候選權重、Scaler、驗證採用決策、逐筆預測及帳本；沒有替換正式模型。
歷史新聞可取得時間未獲證明，零滑價與不含股利也限制解讀；不可為了超過大盤而重選測試結果。

唯讀帳本重播：`python -m scripts.research_extension --mode reproduce --output-dir reports/research_extension/2026-11-06`。
凍結候選預測：`python -m scripts.research_extension --mode forecast --output-dir reports/research_extension/2026-11-06 --as-of YYYY-MM-DD`；只輸出預測，不重訓、不發布研究樣本、不記成交。
首次保留訓練輸入至資料截止日的每個 XTAI 交易日都必須有完整股價；無法用缺價縮短五日標籤。
