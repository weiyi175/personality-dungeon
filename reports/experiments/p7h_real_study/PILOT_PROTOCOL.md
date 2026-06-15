# Pilot 招募作業規程（人格迭代研究）

對應 pre-reg `人格迭代實驗_規劃_v1.md` §6b（程序 locks）、§8e（debrief 盲編碼）、§8g（participant_id 判別）。

## 1. participant_id 指派
- **一碼一人，序號永不重用**：P01、P02、…，即使受試者棄坑，該碼也不回收、不重指派給別人。
- naive 受試者：實驗者在交出前於 start screen「受試者代碼」欄填入其 `P0X`。
- **實驗者試玩永遠用 `dev`**（預設值，不用改）→ 自動排除於分析。
- 即時交錯安全：判別靠代碼不靠時間，dev 與 P0X 可穿插進行。

## 2. 臂別（arm）— 封存、事後還原
- 不在招募 log 寫 arm。collection 期間**保持 arm-blind**。
- count-balance 在後端按 **started run** 自動配臂（sticky per run_id）。
- 事後還原：以 `participant_id=P0X` 篩 sessions → 取 `run_id` → 讀 `iteration_arm`。
- **解盲時機**：debrief 三類盲編碼**完成後**才把 arm 與 participant 交叉。

## 3. debrief 盲編碼（§8e）
- manipulation_awareness 開放文字與 arm **分開保存**。
- rubric 三類：(A) 點到敘事 priming／(B) 點到連續/累積機制／(C) 都沒點到。
- 收完、遮住 arm 再依 rubric 編碼，**不可事後看 arm 用眼判**。
- B 類比例高（尤其 iterated）→ E1/q4 須打折解讀。

## 4. 指示語（hands-off，維持 naive）
- 實驗者 hands-off：不提「延續 / 迭代 / 同一個角色 / 操弄 / 兩種版本」。
- 中性指示：寫遺言 → 看分身闖地下城 → 崩壞後再寫 → 共三輪 → 結尾問卷。
- 不暗示任何跨週期關聯或研究假設。

## 5. 停止規則（綁 completed，不是 started）
- 持續招募到**兩臂都 ≥10 個 completed（完整 3 週期 + 問卷）**。
- 期間定期跑：`./venv/bin/python scripts/experiments/analyze_iteration_study.py`
  看 `selection.n_complete_3cycle_runs` 的 per-arm 數。
- count-balance 平衡的是 **started**，真人有 dropout → 預期每臂 started ~12–13 才湊到 10 completed。
- log 用 `pilot_recruitment_log.csv`（participant_id｜date｜completed y/n｜dropout/notes）。

## 6. pilot_start（次要 defense-in-depth）
- participant_id allowlist 已扛主力；pilot_start 只是次要保險。
- 招到 P01 那一刻記下 unix timestamp 當 `pilot_start`。
- 分析時帶 `--pilot-start <ts>`；在那之前留空（不阻塞）。

## 7. 分析指令
```
# 預設樣式 ^P\d{2,}$ 自動只納 naive；dev/EXP_PREPILOT 自動排除
./venv/bin/python scripts/experiments/analyze_iteration_study.py
# 收完後帶 pilot_start 次要保險：
./venv/bin/python scripts/experiments/analyze_iteration_study.py --pilot-start <P01_unix_ts>
# 或明確清單：
./venv/bin/python scripts/experiments/analyze_iteration_study.py --pilot-participants P01,P02,...
```
