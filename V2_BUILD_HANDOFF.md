# V2 Build Handoff — personality-dungeon

> ## ⛔ OBSOLETE — v2 build TERMINATED 2026-06-16（封版，不 build）
> **此 handoff 描述的 build 已終止，請勿執行。** Build-time 驗證發現 9D「機制自我」是
> P7-H 刻意的 static P₀ 控制（不個體化）→ claim A 不可被儀器化 → 終止、封版（非 failed
> hypothesis）。canonical 記錄與完整理由見 **`人格延續實驗_v2_規劃_v1.md` 頂部 STATUS=TERMINATED**
> + `研發日誌.md`（2026-06-16）+ 記憶 `apparatus-limits-static-personality`。
> 以下原始 handoff 內容保留供 trail，不再有效。
>
> ---

> 給新 session 用的自含 prompt + 檔案引用清單。直接讀這份檔，不靠貼上。
> 最後更新：2026-06-15（v1 pilot 封板後）

---

## 貼進新 session 的 handoff prompt

```
你是 personality-dungeon 的調查/build 者。

【先做,別跳】讀記憶 investigation-discipline:不驗不斷言——每個專案事實 trace 到
檔案:行號/實際資料/git，或實際跑出來；不從這段 prompt 或訓練記憶斷言。設計事實
用 print/讀檔確認，經驗問題跑最小決定性實驗。

【本 session 任務】v2「解耦式 felt-continuity」進 build 階段。

【先讀這些權威來源（此 prompt 只是索引，內容以檔案為準）】
1. 人格延續實驗_v2_規劃_v1.md 的 §0 LOCKED/TBD 表 —— v2 設計定案。分支 feature/continuity-v2。
2. 記憶：personality-iteration-study（v2 段）、game-vision-original（小火龍概念 1/2）、
   backend-restart-required（後端 port=8001、無 --reload、此 sandbox 對 listening socket
   送 SIGSTKFLT 跑不起後端、改任何 tracker 資料檔前先停後端）、
   frontend-architecture-doc + v2-immersive-ui（Godot 場景↔腳本↔端點地圖）。
3. v1 已封：reports/experiments/p7h_real_study/ITERATION_PILOT_REPORT_v1.md（informative-null）。

【狀態】v1=informative-null 封板（main, tag iter-pilot-v1-null）；v2=pre-reg 設計穩定、
待 build；ecology=收線不動（別重開）。

【v2 build TODO（全部以 §0 表為準，動工前先讀）】
A. 內容綁人格 callback（treatment）：攜帶 9D → 對母體 z-score → |z| top-2 traits →
   diegetic NPC 台詞（零逐字、不顯示向量）。trait 名權威 = api/ecology_tracker.py:34-38。
B. anti-match control：凍結 bank（v1-pilot 60 向量/20 人，hash 5e11082a8d4cf101，排 dev/EXP）
   → 從 top-2 不重疊子集隨機抽、排 self、逐 control 記客觀相異度（manipulation-check）。
   ⚠️ bank 來源/數字動工前在 p7h_player_test_sessions.json 上重驗（資料會變）。
C. 兩臂機制相同：都跑 3-cycle 累積，只差 callback 準確度（treatment 準確鏡 / control anti-match）。
D. DV（寫進問卷）：felt-continuity 4 題 battery（§5 草案，逐字題幹【待驗】需量表全文）
   + perceived-accuracy 1 題 + 再認 R1（干擾項認自己前世遺言原文，排在 battery 之後）。
E. firewall 不可破：絕不顯示人格向量/不照搬遺言原文。
F. analyzer：擴充 scripts/experiments/analyze_iteration_study.py（PRIMARY 換 battery 合成分、
   加 accuracy 共變 + 再認 check、移除 E1）。= build-time。

【動工前要拍的 TBD（§0 表）】k 值（2/3）、battery 逐字題幹（查 FSCQ/Ersner-Hershfield 等，
標待驗）、相異度帶（v2a）、raw↔d（v2a）、v2b N（TOST，本機無 statsmodels 待精算）。

【污染防線（承接 v1，別省）】participant_id gate ^P\d{2,}$、一人一臂一次、blind-to-arm
debrief 編碼、改資料檔前停後端。

【git】見記憶 git-commit-push-workflow：v2 工作留 feature/continuity-v2（git push origin HEAD），
別 push origin main；改前先 git status，別盲 add .。
```

---

## 檔案引用速查

### 規劃 / pre-reg（權威設計來源）

| 檔案 | 用途 |
|---|---|
| `人格延續實驗_v2_規劃_v1.md`（`feature/continuity-v2`） | v2 設計定案，**§0 LOCKED/TBD 表先看** |
| `人格迭代實驗_規劃_v1.md` | v1 pre-reg（§6 power、§8d DV 階層、§8f v2 mandate） |
| `reports/experiments/p7h_real_study/ITERATION_PILOT_REPORT_v1.md` | v1 pilot 結果（informative-null） |
| `reports/experiments/p7h_real_study/PILOT_PROTOCOL.md` | 招募作業規程 |

### 程式（build 要動 / 參照）

| 檔案 | 用途 |
|---|---|
| `api/ecology_tracker.py:34-38` | 9D FEATURE_NAMES — callback top-k 的 trait 名**權威** |
| `evolution/replicator_dynamics.py:808-919` | substrate：人格→3 訊號 + 循環 payoff + 向心突變（理解機制） |
| `scripts/experiments/analyze_iteration_study.py` | 分析器（gate `^P\d{2,}$`），v2 analyzer 基底 |
| `scripts/experiments/relabel_pilot_pids.py` | relabel 工具（資料維護） |

### 資料

| 檔案 | 用途 |
|---|---|
| `reports/experiments/p7h_real_study/p7h_player_test_sessions.json` | session/剖面（anti-match bank 來源：is_human+9D；**用前重驗**） |
| `reports/experiments/p7h_real_study/p7h_survey_responses.json` | 問卷（battery/accuracy/recognition 寫這裡） |
| `reports/experiments/p7h_real_study/ab_test_sessions.json` | A/B arm 分配 |

### Godot 前端（build 主戰場）

Windows repo：`/mnt/c/Users/n1166/personality-dungeon`

場景↔腳本↔端點地圖**查記憶 `frontend-architecture-doc` 與 `v2-immersive-ui`**（但以實際 repo 為準，記憶可能過時）。

### 記憶（自動載入，列出供明確引用）

| slug | 用途 |
|---|---|
| `investigation-discipline` | 怎麼工作（不驗不斷言） |
| `personality-iteration-study` | v1 封板 + v2 checkpoint |
| `game-vision-original` | 願景（小火龍/12 個性格） |
| `personality-ecology-layer` | ecology 收線（別重開） |
| `backend-restart-required` | port 8001 / sandbox SIGSTKFLT / 停後端再改資料 |
| `git-commit-push-workflow` | push 修正（`push origin HEAD` 不是 `push origin main`） |
| `frontend-architecture-doc` | Godot 場景地圖 |
| `v2-immersive-ui` | DungeonLifecycleSceneV2 詳細 |

### ecology（已收，參照不動）

`reports/ecology/` + `scripts/experiments/ecology_*.py`
（replicator_probe / heteroclinic_check / 9d_coverage / bstep0_driven_response）

---

## 待辦（優先序）

### 立即 / Ops（可自行做）

1. Push all unpushed refs：
   ```bash
   git push origin main
   git push origin iter-pilot-v1-null
   git push origin feature/continuity-v2
   ```
2. 後端重啟（已停，relabel 後需重啟載入 P 碼）：
   ```bash
   cd /home/user/personality-dungeon
   ./venv/bin/python -m api.server > logs/api_server.log 2>&1 &
   ```
3. 驗後端 + naive 仍 10/10：
   ```bash
   curl -s http://127.0.0.1:8001/survey/questions
   ./venv/bin/python scripts/experiments/analyze_iteration_study.py
   ```

### v2 Build（另開 fresh session，`feature/continuity-v2`）

4. Godot 端：內容綁 personality callback（treatment arm）
5. anti-match yoked control arm
6. Survey 擴充（§5 battery + accuracy + recognition）
7. Analyzer 擴充（PRIMARY 換 battery 合成分）

### 待驗 / 待精算（不擋 build，pre-reg 前補）

8. Battery 逐字題幹（paywall）：FSCQ (Sokol & Serper 2019, PubMed 31107602) + Ersner-Hershfield 2009 + Past Self-Continuity Scale JPA 2025
9. v2b N（TOST）：安裝 statsmodels → 精算 ±0.4 等價檢定 n/arm（手算約 100-110，superiority d=0.4=98/arm 不夠）
10. 相異度帶（v2a 估 accuracy 變異後定）

### Merge 衛生（最終 feature→main 時）

11. `feature/continuity-v2` 從 `31f5cc3` 分出（README 修之前）；merge 時保留 main 的 README（有 broken-ref 修正）
