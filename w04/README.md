# w04 — HW1:Agentic Harness

[Stanford CS329Z Assignment 1](https://github.com/cs329z/assignment1-harness) 的實作。從零寫一個 agent harness:文字格式的工具呼叫、主迴圈、上下文壓縮、跨對話記憶、使用者核准與防注入。`hw1/` 只放我寫的部分;課程程式(MIT,見 `hw1/LICENSE`)請 clone 原 repo。

```bash
git clone https://github.com/cs329z/assignment1-harness && cd assignment1-harness
cp -R <this repo>/w04/hw1/cs329z_hw1 <this repo>/w04/hw1/dev_tools .
uv run python data/download.py
uv run pytest                                                            # 確定性測試
uv run python -m pytest -p dev_tools.claude_cli_lm -m live -s            # live,走 claude -p
```

## 驗收
確定性測試 287 個全過。下表是 live 結果,agent 與模擬使用者用 Claude Haiku、評審用 Sonnet;**沒有用課程的評分模型 `gpt-6-luna` 驗證,不代表正式分數**。

| 題目 | 結果 | 門檻 |
|---|---|---|
| priority | 58/60 | 85% |
| daily_digest | 5 天的 3 項檢查全過 | 全過 |
| email_qa | 24/25 | 70% |
| tool_protocol | 19/20 | 18 |
| search_docs | 19/20 | 15 |
| terminal | 10/10 | 8 |
| compaction | 通過,最大 3,998 tokens | 上限 4,000 |
| memory | 3/3 | 2 |
| guardrails | 注入 0 次,回答 6/7 | 注入 0 次,回答 6 |
| sim_eval | 20/20(修正前 15/20,只跑過一次) | 16 |

`field_notes` 與 `design_memo` 尚未撰寫。

## 心得
- 確定性測試全過不代表能用:第一輪 live 才發現模型會寫出 `<invoke:search_docs>` 這種亂語法,被當成最終答案。把「看起來想呼叫工具但格式錯」視為 malformed 後,文件題 14/20 → 19/20
- 記憶要保留順序:`recall` 按相關度排序時,更正過的新舊事實分不出先後,模型只好兩個都講
- 壓縮大多不用 LLM:先截短舊工具結果,不夠才寫摘要;使用者原話直接保留,限制條件不靠摘要品質
- 大輸出存檔再給路徑:21 萬字元的輸出之後,模型仍能用 `sed` 查指定行
- 判斷規則算不算 harness:模型不照做時,有沒有程式會擋下來?寫在 prompt 裡的只是請求
