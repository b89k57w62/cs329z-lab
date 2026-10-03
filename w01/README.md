# w01 — 最小 ReAct agent

只用標準庫實作 ReAct 迴圈。

```bash
python3 agent.py "我 9 月總共花多少?"
```

## 驗收
- 9 月花費:3 步算出 6080 ✓
- 檔案不存在:換年份重試後如實回報 ✓
- 同題 5 次:全對,但步數 2–5,2 次跳過 calculate

## 心得
- transcript 是唯一記憶,寫入前要清理
- prompt 規則是軟限制;只看答案會漏掉過程的不一致
