# 資料、新聞與模型來源條件核查

核查日期：2026-11-10（臺北）。範圍為本專題目前使用的四類來源及公開交付方式；這是可核對的來源紀錄，**不是全部用途已獲授權或已取得法律認證的聲明**。程式庫、模型權重、資料內容及API存取各有不同條件；HTTP成功、無需登入與學術目的，均不能單獨代替授權證據。

| 來源 | 本次查得的條件 | 本專題處理與限制 |
| --- | --- | --- |
| FinMind | 方案表允許Free等級的非商業／學術用途；完整資料再散布及鏡像API不在授權內。原提供機關另有要求。 | V3完整原始回覆與標準化行情留在合法取得的本機快照；Git／新ZIP只交付程式、衍生研究成果及雜湊，不聲稱clone已含全部輸入。仍須依使用者實際方案、TWSE規定及使用情境核對。 |
| Erlangshen情緒模型 | 本次查得的確切模型卡revision明列Apache-2.0；上游Fengshenbang專案亦提供Apache-2.0條款。 | 可追溯本次模型卡宣告；若另行散布權重或修改版本，須保留適用授權、著作權及NOTICE等要求。模型卡宣告不替代訓練資料、新聞內容或其他第三方權利。 |
| yfinance／Yahoo | yfinance程式碼是Apache-2.0；其官方文件明示非Yahoo官方工具、研究教育定位，並提醒Yahoo Finance API為個人用途，下載資料權利應依Yahoo條款。 | 不把程式碼開源授權當作行情再散布權。適用地區Yahoo全文本次無法完整取得，不宣稱團隊、公開儲存庫或自動化用途已全面符合資料條件；既有行情來源需另有相應依據。 |
| 鉅亨新聞／API | 官方服務條款及著作權聲明保留新聞內容使用、重製、改作與散布等權利；另有提供API、RSS/XML的授權服務說明。 | 未查得供此非官方公開JSON端點使用的通用研究／自動化授權。公開結果保留識別、時間、雜湊與分數，不因端點可讀就新增發布新聞全文或API原回應。既有使用依據仍未完成認證；數值衍生物也不因此自動取得所有權利。 |

## 模型版本證據

本次透過Hugging Face官方API查得 `IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment` 當時 `main` 對應：

```text
revision: 7c257c5cde3225d0789acfa8d67eb043289b0295
README.md SHA-256: 4b1e2483bec20b57cceae2d6e88db997f5ff4638fce5cac340f497ecedf20ea1
model-card license: apache-2.0
```

這是**此次核查**的版本，不是舊歷史評分曾使用該版本的證明。新版 `SentimentScorer` 會解析精確commit，以同commit載入模型與Tokenizer，保存評分設定與runtime版本，且不啟用remote code；舊評分仍維持未驗證標示。沒有為本文件重新評分、改寫快取或下載整份模型權重。

## 主要來源與可用性

- [FinMind免責聲明與資料授權](https://finmind.github.io/Disclaimer/)：頁面版本日期2026-10-31；來源標示及第三節授權範圍。[方案及非商業／再散布矩陣](https://finmind.github.io/Pricing/)、[服務條款](https://finmind.github.io/PrivacyPolicy/)。
- [確切revision模型卡](https://huggingface.co/IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment/blob/7c257c5cde3225d0789acfa8d67eb043289b0295/README.md)、[官方模型API](https://huggingface.co/api/models/IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment)、[Fengshenbang官方LICENSE](https://github.com/IDEA-CCNL/Fengshenbang-LM/blob/main/LICENSE)。本次Hub檔案清單沒有單獨LICENSE檔，故明確稱「模型卡宣告」，不虛構額外授權文件。
- [yfinance官方文件的使用聲明](https://ranaroussi.github.io/yfinance/)、[官方程式碼LICENSE.txt](https://github.com/ranaroussi/yfinance/blob/main/LICENSE.txt)。文件連往[Yahoo條款](https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html)與[API條款](https://legal.yahoo.com/us/en/yahoo/terms/product-atos/apiforydn/index.html)；本次全文請求回傳999，Yahoo Finance說明頁回傳429，因此未把未讀全文寫成已驗證結論。
- [鉅亨服務條款第七節](https://www.cnyes.com/service.htm)、[著作權聲明](https://www.cnyes.com/announce.htm)、[官方新聞API／RSS授權產品說明](https://campaign.cnyes.com/topics/anuedatabase/news.html)。官方產品說明存在不代表本研究已訂閱該方案或獲得其授權。

本次沒有聯絡供應商、接受新契約、購買方案或變更服務帳號。若轉成商業服務、公開新聞資料庫、模型權重鏡像或更大規模的自動存取，必須另行核對相應使用情境，不能直接沿用本次有限核查作完整授權證明。
