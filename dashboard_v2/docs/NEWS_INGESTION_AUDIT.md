# 新聞稽核

V2.11 新聞快取欄位是 news_id、ticker、publish_time、Date、Daily_Sentiment、Model_ID。沒有標題與文章網址。

provenance JSONL 有 publish_time、first_seen_at、fetched_at、來源 API 與內容雜湊，內容範圍是 api_title_summary，但標題文字本身沒有留在檔案。畫面標 TITLE_NOT_STORED，不補標題。

情緒 0 標成 ZERO_UNVERIFIED，不當成中性。單篇是否進入模型決策標 UNVERIFIED。看板重新整理只重讀檔案，不抓新聞，也不寫正式快取。
