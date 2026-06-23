# 接手 Prompt — 下個 Session（2026-06-24 交接）

> 你接手 personality-dungeon。**先讀自動載入的 memories（MEMORY.md + memory/*）**——它們是跨 session 的權威知識庫，本檔只給「當前戰線 + 下一步 + 怎麼判方向」。本檔 file:line 可能過時，斷言前用 verify-don't-assert 對現碼/資料查證。

## 0. 工作紀律（沿用）
- **verify-don't-assert**：每個專案事實主張都要 trace 到 file:line / 資料 / 跑一次，不可憑記憶。本 session 靠這個連抓 4 個錯（見 §3）。
- **rtk** 前綴所有指令（連 `&&` 鏈也是）。
- **雙 repo**：後端 `/home/user/personality-dungeon`（branch `docs/l3-bottleneck-prereg`，有 origin）；前端 `/mnt/c/Users/n1166/personality-dungeon`（branch `master`，**無 remote**——commit 只能本地、推不了）。
- **commit/push**：你 commit、**使用者 push**（`git push origin HEAD`，別用 `push origin main`）。只 stage 指定檔，別 `git add .`（會掃 logs/backup/別人改動）。commit footer：`Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`。
- **後端**：改 `api/` 後**使用者須手動重啟** `python -m api.server`（:8001，無 --reload）。sandbox **跑不了** listening socket（SIGSTKFLT），但**能當 client 讀** `http://localhost:8001`（read-only 探活 OK）。**POST 會寫 production 資料——別碰會寫的路徑**（教訓見 §3.4）。
- **Godot**：sandbox 跑不了；你能寫 `.gd`、靜態審錯，**跑場景/驗畫面只能使用者**。

## 1. 主線錨點（判方向前先對這個）
**研究皇冠 = 多樣性動力**：生態能否維持共存（diversity）vs 塌成 monoculture；多少 directional 壓力 g 會翻。`game-vision`：**diversity 才是目標，a>b 共存=成功、b>a 旋轉=feared collapse**。
- 研究已 **sim-complete**：g\*(β) 曲線（conditional on β）+ controls + ablation 全做完。
- **唯一開放經驗量 = 真實 β**（真人 authoring 對稀缺的響應銳度）→ 需**真人走 live loop + 稀缺變動**才能 bound（= 乙，非 solo）。
- **遊戲（經濟/PvP/地牢）是收集真人資料的載具，不是目的**。它存在是為了吸引/留住玩家來產生 β/同質化 的行為資料。

### ⚠ 反偏離檢查（使用者特別要求：給實驗方向前必跑）
**跳出當下問題、看整體趨勢**：提任何實驗/build 前先問——
1. 這件事**能不能 trace 到「讓我們更接近 bound β / 回答多樣性問題 / 收到真人資料」**？
2. 還是它是**側支**（為做遊戲而做遊戲、為機制而鑽機制）？**L3 dynamics 線就是前車之鑑**——鑽進去才發現是重新發現既有 finite-N quasi-cycle，已 SHELVED。
3. solo 能做的研究**已近天花板**；若你發現自己在「發明新 build 來填時間」，那就是偏離訊號——**寧可誠實說「solo 沒有高價值待辦了，瓶頸在使用者跑 pilot」**，也不要製造 busywork。
> 一句話：**main line = 真人收集（乙）**。不服務這條的，標記為 off-main-line 並講出來。

## 2. 當前狀態
**研究軌**：sim/instrument 全完成。`乙 pre-reg`（`docs/experiments/ecology_directional_pressure/BEHAVIORAL_BETA_PREREGISTRATION.md`）含 H0(β=0)、三鐵律、樣本量、**§8 pilot→confirm 協定 + 雙實驗隔離 + R0–R7 runbook**。β-instrument + power 建好驗好。現有資料 n≈199 真人 live 但 **β 無資訊量**（CI 橫跨 0，因稀缺沒變動 + 疑 pseudo-replication）→ 正是乙要解的。
**遊戲軌**：Inc2（錢包+門票）、Inc3（玩家地牢+零和 Rank+防禦 sink+raid，§10 firewall 裁定 SAFE）、**雙經濟循環全通**（冒險賺幣=生態幣+存活幣後端權威；PvP 花幣）。手感經使用者驗 OK。測試全綠。
**就緒度**：乙 runbook **R0 ✅**；**R1 pilot 可直接開跑**（真人玩冒險兩源賺幣、玩 PvP 花幣，monitor 看稀缺變異）。R1–R7 由使用者主導跑。

## 3. 本 session 的關鍵更正（別重蹈）
1. **run_id 慣例 store-specific**：ecology 真人 = `session_id` ∧ `outcome` 非空（**非** run_id 號）；P7-H 另一套。跨 store 套 run_id 會誤判（曾把 n=199 真人當 sim 報成 n=0）。
2. **wallet-credit 曾反掉**（已修）：真人 run_id 非空，舊 `if not req.run_id` 從不對真人觸發。
3. **intrinsic archetype dist 撤回**：唯穩健 = **balanced-thin**；agg/def 排序隨選法翻（原 [.48/.43/.09] 是 54-pilot artifact）。
4. **測試曾覆蓋 production 資料**：端點測試 `_ecology_save` 把 `reports/ecology/ecology_state.json` 覆蓋（210→1，11 筆未提交遺失）。已修（`fresh()` 把 OUT_DIR 導 tmp）。**跑會 WRITE 的東西前先確認寫去哪**（memory `isolate-write-targets-before-running`）。

## 4. TODO-list（優先級已排；先跑 §1 反偏離檢查）

**P0 — agent solo 可做、直接服務 pilot（建議起手）**
- [ ] **pilot 看板腳本**（read-only）：一條指令串 `GET /ecology/scarcity_variation` + `GET /wallet` + 對現 state 跑 `ecology_beta_fit` → 印 R1→R2 gate 就緒度（scarcity_std vs 0.2 target、n_real、β verdict）。降低使用者跑 pilot 的摩擦。**不 POST、不寫 prod。**

**P1 — agent 寫 .gd、使用者 Godot 驗（解鎖 β 絕對刻度 = R6）**
- [ ] **seen_scarcity 前端接線**：V2 author 前把顯示的稀缺快照隨 `/ecology/submit` 帶上（後端已收）。→ 確認 softplus link → β 從單調指標升級為**絕對 g\*(β) 刻度**。

**P2 — gated on pilot 結果（有資料才做）**
- [ ] pilot 資料進來後：查 scarcity_std 是否達 0.2 target + β 形狀合理 → 建議凍結（R2）+ 指派 config_version。
- [ ] 若自然稀缺不變動 → 設計**強 induction**（顯示輪換/隨機化）——但這**先要 firewall/coin-誠實性設計裁決**（顯示≠真 q），不可單方面寫碼。

**P3 — 研究 consolidation（solo，若想不靠收集推進貢獻）**
- [ ] **capstone 綜述**：把多樣性引擎 + g\*(β) conditional + 方法論（pre-reg/正控制/L3 metric-trap 案例）+ apparatus 限制 收成一份 paper 導向報告（目前只有零散 per-experiment 報告）。
- [ ] **grain 分析**：9D→3 archetype 投影會不會藏掉真實多樣性？用現有 199 筆 SBERT-9D 跑（現有資料即可，solo）。

**P4 — 乙 之後**
- [ ] **B'（PvP-同質化）pre-reg**：firewall 已保證與乙 隔離；null = reduced-form g\*（需先有 β）。待 Inc3 競技迴路成熟、另開。

**使用者主導（非 agent）**：R1 pilot 跑、R2 凍結決定、R4 confirmatory 收集（真人）、push、後端重啟、Godot smoke。

## 5. 指向
- 研究：`docs/experiments/ecology_directional_pressure/{BEHAVIORAL_BETA_PREREGISTRATION,BETA_INSTRUMENT,ECO_DP_RESULTS,DIRECTIONAL_PRESSURE_PREREGISTRATION}.md`、`scripts/experiments/ecology_beta_{fit,power}.py`、`p7h_intrinsic_archetype.py`。
- 遊戲：`地牢經濟_R3C隔離_規劃_v1.md` v1.4（§3 雙經濟循環圖 / §10 firewall 重審）、`api/{wallet_manager,pvp_manager,ecology_tracker,server}.py`、前端 `src/core/{PvpClient,WalletClient}.gd` + `src/ui/PvpScene.gd`。
- 進度：`研發日誌.md` 末節「🧪 生態行為版 β（乙）收集就緒…」。
- 後端 commit 軌（已 push 至 70b35fc；日誌更新 72ad2a7 待 push）：`32058f0`(firewall 重審)→`1788ab0`(Inc3)→`ae52828`(乙§8+日誌)→`70b35fc`(雙經濟 R0)→`72ad2a7`(日誌)。

**交接時間戳**: 2026-06-24 ／ **負責人**: Claude Opus 4.8 + User
