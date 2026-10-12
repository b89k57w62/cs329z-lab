# w02 — 結構化輸出(JSON schema 驗證 + 重試)

只用標準庫:自寫 schema 驗證器、容錯 JSON 抽取、帶錯誤回饋的重試迴圈。

```bash
python3 experiment.py   # 20 則樣本 × 有/無回饋,約 3 分鐘
```

## 驗收
- validate:缺欄位、型別錯、enum 錯、陣列元素錯皆回報正確路徑;bool 不算 integer ✓
- parse_json:code fence、前後文字、字串內大括號皆可解析;壞 JSON 回報行列位置 ✓
- extract:假 call_llm 驗證重試、回饋內容、超過上限拋 ExtractionError ✓
- 20 則:有回饋 / 無回饋皆 100% 一次成功(平均 1.00 次),無需重試案例

## 心得
- haiku + schema 放 system prompt 已足夠,重試迴圈未被觸發;要量出回饋效果需更嚴格的 schema
- validate 只驗格式不驗真實:#7「下週三」被填成 2026-10-21
