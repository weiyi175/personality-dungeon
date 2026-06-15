# 人格迭代研究 Pilot 報告 v1 — 合法 Informative-Null

- **里程碑 tag**：`iter-pilot-v1-null`（main，2026-06-15）
- **pre-reg**：`人格迭代實驗_規劃_v1.md`
- **資料**：`reports/experiments/p7h_real_study/p7h_player_test_sessions.json`（+ survey、ab_test）
- **分析**：`scripts/experiments/analyze_iteration_study.py`（免 flag；gate＝participant_id `^P\d{2,}$`）

## TL;DR

Pilot 收滿 **naive 10/10**（iterated/reset，皆完整 3-cycle）。預登錄 PRIMARY **E1 兩臂無差異（雙尾 p=0.97）**，q4 兩臂皆 floor。這是一個**預登錄的合法 informative-null**（pre-reg §8f）：當迭代機制在玩家端**幾乎不可感**時，純敘事連續性 priming **推不動**自由書寫行為（E1）與主觀延續感（q4）。→ **mandate 解耦式 continuity v2。**

## 設計（pilot 前鎖定）

- 受試者間 A/B：iterated vs reset，3-cycle run。**narrative-led**（機制近乎無 experiential footprint；唯一隨臂變的視覺＝週期 2/3 的 priming 文字行）。
- DV 階層（LOCKED，§8d）：**M**＝QA 構造檢核（套套邏輯，排除於宣稱）；**E1**＝PRIMARY（自由遺言 SBERT `cos(cycle0,cycle2)`，**雙尾、方向開放**）；**q4**＝exploratory floor-check；**E2**＝exploratory。
- 停止規則：每臂 ≥10 個完整 3-cycle run。

## 樣本與來源稽核

- **iterated 10 / reset 10** 完整 3-cycle naive run（participant_id ∈ `^P\d{2,}$`，namespace P04–P31）。
- **污染處理**：`dev04` 同一人 4 分鐘內跨臂玩兩次（iterated→reset）→ **首輪（naive，iterated）保留為 P23、次輪隔離** `EXCL_dev04_run2_reset`；`P19` 同臂 reset 重複 → 首輪留、次輪隔離 `EXCL_P19_dup_reset`。
- **provenance 註**：第一批真人被誤打 `devNN` 前綴；relabel→P 時曾被 live 後端存檔覆寫（in-memory tracker 沖掉檔案編輯），**後端停止後重做才生效**（`relabel_pilot_pids.py`）。

## 結果（n=10/10）

| DV | iterated (median) | reset (median) | 檢定 | 讀法 |
|---|---|---|---|---|
| M（QA，套套邏輯） | 0.960 | 0.836 | p≈0.006 | 累積種子必然錨定，**非發現** |
| **E1（PRIMARY）** | **0.815** | **0.836** | **雙尾 p=0.97** | **零臂差**；實測 Δ≈0.02（d≈0.07） |
| q4（floor） | 3.0 | 4.0 | p=0.13 | 兩臂皆低 = both-floor |
| E2 climb-rate/step | 0.0076→0.0066 | 0.0077→0.0072 | — | 相近；iterated 略低（exploratory） |
| saturation | 0.97（釘極端） | 0.27 | within sat×E1≈−0.19 n.s. | exploratory |

**變異（供正式 N 回推，sd_upper95）**：E1 reset **0.320**（binding）/ iter 0.130；q4 reset 2.58 / iter 1.89。N 參考（雙尾 α=.05 power .80）：d=0.5→64/arm、d=0.8→26/arm（標準化，不吃 sd）。若以實測 Δ≈0.02 回推則 N>3000/arm，不可行。

## 詮釋

預登錄 **informative-null**（§8f）。三項交付達成：(1) **唯一可取得的乾淨 E1**（自由書寫行為）——只有機制隱形時量得到；(2) felt-continuity 的 **floor 證據**；(3) 建 **decoupled continuity v2** 的實證 mandate。

## Rigor caveats（任何宣稱必帶）

1. **n=10 只排除「大效應」，非嚴格等價。** d≈0.07、CI≈±0.9，容得下 d~0.3 的小效應（需 200+/arm，**不追**）。最強誠實宣稱＝**「無大效應」**。要升級成「在 SESOI 內統計等價」須**預登錄 SESOI + TOST 等價檢定**（本報告未跑）。
2. **q4 點估計反向**（reset 4.0 > iterated 3.0，n.s.）→ 進一步**反向於** iterated>continuity 的方向，非僅 null。
3. **blind debrief 編碼＝「太弱沒察覺」**（非空 debrief 僅 `不知道`、`沒感覺到差別`）——操弄**次閾於受試者覺察**，與「機制不可見 → 產不出 felt continuity」一致；**非**「察覺到但無效」。

## 這 mandate 什麼（v2）

機制**可見**的 v2 會**必然污染 E1**（顯示前世人格 → 玩家照抄 → E1 退化成套套邏輯）。故 v2 **反轉**：**E1 退役**為 DV；PRIMARY **換成 q4 / felt-continuity**（v2 把連續感做成可感 → 才量得到）；continuity 線索**必須與遺言/人格向量通道解耦**（世界狀態持續 / 重複 NPC /「第三條命」框定 / 前世遺言**主題** callback），**絕不顯示人格向量**。新 pre-reg 於分支 `feature/continuity-v2`，刻意 scope。**此 narrative-led pilot 是拿乾淨 E1 的唯一窗口（已用掉、已定案）。**

## 可重現

```
分析：./venv/bin/python scripts/experiments/analyze_iteration_study.py
gate ：participant_id ^P\d{2,}$（naive=20，iterated 10 / reset 10）
relabel：scripts/experiments/relabel_pilot_pids.py
里程碑：git checkout iter-pilot-v1-null
```
