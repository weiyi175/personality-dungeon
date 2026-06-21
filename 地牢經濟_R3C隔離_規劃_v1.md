# 地牢經濟 — R3/C 隔離軌 規劃 v1（PvP 當隔離 game feature）

> **狀態**: DRAFT v1.2（規劃，未實作、未 commit）
> **日期**: 2026-06-22（v1.2：κ-sweep 模擬定案——採 **(ii)** 多樣性 coin 可買戰力；F6 在 per-submission 序列化下 sim 證**不致 whiplash**；新增穩定不變式 **S1** + α\*(κ) 安全表 + 權重 EMA=內建阻尼。v1.1：firewall review F1/F6/F5）
> **作者**: Claude Opus 4.8 + User
> **關係**: 與 reduced-form bifurcation pre-reg（研究軌）**並行**；本份是**遊戲軌**。
> 兩軌共識＝把 directional 壓力搬出 PvP（研究用抽象 `g`），PvP 在此**降為隔離 game feature**。
> 參照：`地牢設計_待回歸問題_parking_lot.md`（C 經濟）、記憶 `eco-dp-directional-pressure-reframe`、
> `personality-ecology-layer`；前置封存＝研發日誌「主線封存 + 遊戲層藍圖落地」§七（Increment 1 已建）。

---

## §0 一句話 + 範圍

把 Increment 2 的「解鎖經濟」**減掉污染向量**後落地：給 Increment 1 剛 ship 的金幣一個 **sink**（門票/防禦升級）
＋ **持久錢包**，**全部 archetype-agnostic**，並用一道 **firewall**（§2）確保 PvP/Rank/coin **不新增生態 archetype 分佈上的第二*方向性* selection operator**（量級耦合經 coin 價值存在但受控，見 F6）——以保護剛封存的皇冠結果（P7-H）與研究軌的多樣性 observable 不被污染。

**範圍（刻意收斂）**：① 隔離 firewall（6 invariants）；② 金幣 source/sink 表 + 持久錢包；③ 門票 sink + 防禦升級 sink；
④ 保留 Increment 1 的 RPS combat + Rank vs house 原樣。**不含**真玩家地牢的零和 Rank 轉移、行為實驗、coin→Rank 抵銷（§7 延後/丟棄）。

---

## §1 R3/C 決策與理由（承上輪裁定）

研究軌裁定鏈（verify-don't-assert，本 session）：PvP 的 3-RPS ＝生態 2026-06-19 退役的 payoff 形式；
PvP 與生態**共用 archetype latent state**；Increment 2 的 coin→Rank 接線會把 PvP **武裝成共用 state 上的第二算子** → 污染皇冠。
→ **directional 壓力搬出 PvP**，研究用抽象 `g·d_i`（pre-reg 已起草）；**PvP 在此走 R3/C：純隔離 game feature**。

**R3/C 對 PvP 的定位**：RPS combat 留作**玩家面 counter-play 手感**（「帶 X 來剋它」），
Rank 留作**PvP-內部位階**（vs house，非零和），coin 留作**消耗型經濟燃料**——
三者**互不跨帳、且都不回饋 will-authoring 的 archetype 選擇**。

> **parking C「零和守恆 vs 通膨」在 R3/C 下 MOOT**：Rank 維持 vs-house（無玩家間轉移帳本 → 無守恆可問），
> coin 與 Rank 解耦（無 coin→Rank）→ 那道題**因解耦而消失**，不需裁。（記憶 `eco-dp-directional-pressure-reframe`：A/B/C moot-by-deferral。）

---

## §2 ★ 隔離 Firewall（R3/C 的核心；6 條 invariant）

研究皇冠＝生態 archetype 分佈動力（由 will-authoring 響應 neg-freq 稀缺驅動）。PvP 要「牆在外」＝**不得影響該分佈**。
逐一封住耦合通道：

| # | Invariant | 為何 | 現況 |
|---|---|---|---|
| **F1** | **派系-authoring 解耦（零讀取）**：PvP 出戰派系＝challenge-time loadout 自由選；**PvP 路徑讀取 will-personality ＝ 零**——不只「不自動投影」，更**無任何 will-衍生 default / hint / 推薦**（一個「你的型是 X、推薦帶 X」也是耦合） | 否則「想贏 PvP」會誘導玩家 author 特定 archetype＝第二算子；default/hint 一樣誘導 | 後端**已解耦**（[challenge() 吃自由 faction 參數](api/server.py#L1699)）；invariant＝UI 維持自由選、picker 預設與 will 無關 |
| **F2** | **無 coin↔Rank 跨帳**：丟 parking C 的 coin→Rank 抵銷；Rank↛coin、coin↛Rank | coin→Rank 正是污染向量（讓 Rank churn 經 coin 回灌 archetype 誘因） | 現無此接線（未建）→ invariant＝**不要建** |
| **F3** | **sink 皆 archetype-agnostic**：門票成本/防禦升級**與派系無關**（同價同效，不論 archetype） | 否則花 coin 會偏置某 archetype＝經 sink 的耦合漏洞 | 新設計，§3 鎖 |
| **F4** | **PvP 不餵生態 `submit()`**：PvP session 不進 ecology `_recent` 窗 / 不呼叫 `submit()` | 生態分佈只能由 will-authoring 提交構成 | 現況 PvP 不呼 `submit()`（[ecology submit 只在冒險上傳](api/ecology_tracker.py#L170)）→ invariant＝**維持**；若 PvP 產 session，沿用 P7-H 清洗律（`run_id` 非空排除） |
| **F5** | **observable 分離 + 無第二資訊通道**：多樣性 observable（archetype entropy）只由 authoring 提交算，永不混入 Rank/coin flow；**且 PvP UI 不得把 live archetype 分佈當第二資訊通道餵 authoring**（地牢清單顯示 deployed 派系時別洩露族群分佈，超過既有稀缺提示的部分） | 研究讀數獨立性 + authoring 只該收一條稀缺訊號 | 設計約束，§6 驗 |
| **F6** | **量級耦合受控（非方向）**：coin **source** 是 archetype-coupled（neg-freq 給稀缺型更多 coin，intended、F-exempt）；把 coin 變得能買 PvP（門票/防禦）會**放大「當稀缺型」的 authoring 誘因 → 抬高 effective lam**。這**不是**第二*方向*算子（方向仍是 neg-freq）但是**量級耦合** | 改變既有合法 operator 的**強度** → 平移生態穩態 + 研究軌 production 操作點（pre-reg C3：lam↑ ⇒ g\*↓） | **sim 已解（2026-06-22 κ-sweep）**：per-submission 序列化下有效 α 極低 → F6 量級耦合**不致 whiplash**（即便 κ=8）；binding 旋鈕是 α（慣性）非 κ → **採 (ii)**（coin 可買戰力）。守穩定不變式 **S1**（§3）；α\*(κ) 安全表見 §3 |

**一句話 firewall（v1.1 誠實化）**：*coin 與 Rank 是玩家面消耗/位階；archetype 分佈是研究面 state；兩者間只有一條合法耦合——
neg-freq 稀缺顯示 → authoring（生態本來、intended 的那條）。firewall 封死**任何第二*方向*算子**（F1–F5）；
唯一殘留的是經 coin 價值的**量級耦合**（F6：改 effective lam、不改方向），把它**受控**而非假裝不存在。*

> **接受出範圍（腳註）**：注意力/心力耦合（PvP 太好玩 → authoring 變少/變草率 → 分佈漂移）不可測，明標接受、不 firewall。

---

## §3 經濟設計：source / sink 表 + 持久錢包（補 parking C 🔴 完整表）

**金幣 SOURCE（既有，不改）**
| source | 機制 | 出處 | archetype 耦合？ |
|---|---|---|---|
| 生態多樣性分 | neg-freq score → `score_to_coins`（稀缺 archetype 拿高 coin） | [ecology_tracker.py:85,193](api/ecology_tracker.py#L85) | **有**（intended：這就是逐利→多樣性那條合法耦合，F-exempt） |
| 存活金幣 | 獨立 `SURVIVAL_COIN_RATE=0.3`、不折進生態 score | 研發日誌 §五.3、前端 CollapseScreen | 無 |

**金幣 SINK（新建，全 archetype-agnostic — F3）**
| sink | 機制 | 校準鉤 | archetype 耦合？ |
|---|---|---|---|
| **挑戰門票** | 每次 challenge 扣固定 coin（gate farming、給 coin 用途） | `TICKET_COST`（待定；參考均衡支付 ~100/人 parking C 🟡，門票應顯著低於單場所得才不勸退、又高到有意義） | 無（同價不論派系） |
| **防禦升級** | 花 coin 提升**自己地牢**的守備層級（升 hold；archetype 無關的平層加成） | `DEFENSE_TIERS`（待定數值；效果＝降挑戰者勝率或抬 stake 不對稱，**不分派系**） | 無 |

**持久錢包（新建）**
- 後端 singleton（仿 [pvp_manager save/load](api/pvp_manager.py#L150) + ecology singleton）：per-player `balance` + **append-only ledger**（每筆 source/sink 事件落帳）。
- **記帳語意＝累加**（使用者反覆強調）：每筆 coin 來自不同體系各自計算後**累加**進 balance；sink 各自扣。ledger 誠實拆解來源，不混為單一數。
- 現況：`_total_coins` 只活在前端 CollapseScreen（非持久）→ 本步把它升級成後端持久錢包。

**Rank（既有，不改、不耦合）**：Increment 1 vs-house ±`stake`、floor 0、地牢 Rank 靜態（[challenge():114-147](api/pvp_manager.py#L114)）。**非零和、不碰 coin**。

**★ (ii)/(iii) 裁定（2026-06-22 κ-sweep 模擬，採 (ii)）**：多樣性 coin **可**買 PvP 戰力（**單一幣**、不拆聲望幣）。理由＝模擬證 F6 的量級耦合在現行裝置下**不致 whiplash**：

- **whiplash 由 α（族群每代重選率/慣性）主導、非 κ**。α≤0.5 時所有 κ（0→8）穩態共存；whiplash 只在 α→1（整代翻）出現。
- **α\*(κ) 安全表**（保多樣性 ≥0.9，bisection，σ=0.5/N=30）：`κ=1→α*0.82`、`κ=2→0.72`、`κ=4→0.66`、`κ=8→0.60`。**α < α\*(κ) 即安全。**
- **production 天然落在低 α 安全區**：`submit()` **每筆**就更新 window（[:211](api/ecology_tracker.py#L211)）＋權重 EMA（`eta=0.2`，[:189](api/ecology_tracker.py#L189)）→ per-submission 序列化、無大批次對凍結訊號同響應 → 有效 α≈O(1/W)，遠在 α\* 下（κ=8 仍 ~3–30× 邊際）。
- **權重 EMA `eta=0.2` ＝內建阻尼**（≈ parking D「稀缺加成遲滯/緩斜坡」，原以為未實作——其實這條 EMA 已半實作）。
- 危險角＝三重疊加 `α≥0.8 + 高 κ + 低 σ`；production 序列化使 α→1 幾不可達。
- artifacts：harness `scripts/experiments/ecology_reduced_form_bifurcation.py`、`reports/ecology/reduced_form_kappa_{sweep,refine}.{json,png}`。

**穩定不變式 S1（非 firewall、屬遊戲動力）**：**不可讓大批玩家對「凍結的稀缺快照」同時下注**（例：長週期顯示同一稀缺型 + 累積大量同響應提交 → 抬有效 α → 逼近 whiplash）。守法＝稀缺顯示細粒度刷新 / submit() 維持 per-submission 序列化 → α 低 → 任何合理 κ 安全。

> 誠實邊界：sim 是假設模型；`α↔真人重寫頻率`、`σ↔真人理性` 是建模假設，真值待行為版 B。結構結論（α 主導、序列化壓低 α、κ 在低 α 免費）穩健。

---

## §4 保留 / 改 / 丟（對 Increment 1 已建物）

| 項 | 動作 | 說明 |
|---|---|---|
| RPS authored M、`counter(owner)` 部署、`/pvp/dungeons`·`/pvp/challenge`、PvpScene | **保留原樣** | 玩家面 counter-play 手感，F1 下與 authoring 解耦 |
| Rank vs house、±stake、floor 0 | **保留原樣** | F2 下不碰 coin |
| 持久錢包 / 門票 / 防禦升級 / coin spend | **新建** | §3 |
| coin→Rank 抵銷、offset_ratio、coin_per_rank | **丟棄** | F2；parking C 對應參數刪除 |
| 真玩家地牢、零和 Rank 轉移、防禦**收入** | **延後**（§7） | 重新引入「第二算子」疑慮，gate 在 firewall 重審 + g\* |

---

## §5 parking C 逐項歸結

- 🔴 **零和守恆 vs 通膨** → **MOOT**（§1：Rank vs-house 無轉移帳本、coin 解耦 → 無守恆可問）。
- 🔴 **參數**：`offset_ratio`/`coin_per_rank` → **刪**（無 coin→Rank）；`rank_stake` → 留（Increment 1）；新增 `TICKET_COST`/`DEFENSE_TIERS`（待校準，不寫死）。
- 🔴 **欄位券＝金幣 source** → 併入 §3 source 表（待設計，本版未納，記著）。
- 🔴 **配對規則** → 簡化：從地牢清單挑 + 門票成本；分段/重複遞減**延後**。
- 🔴 **完整 source/sink 平衡表** → **本 §3 即第一版**（sources: 生態+存活；sinks: 門票+防禦）。實際數值校準 gate 在研究軌 g\*（定向壓力預算）+ 均衡支付 ~100/人。
- 🟡 **均衡支付 ~100/人** → 門票成本校準的輸入（sink 要壓得住 source 水位）。
- 🟡 **雙生券** → 延後（§7）。
- 🟢 **存活金幣修正** → 已做，升級成持久錢包的一個 source。
- 🟢 **多樣性 coin 是否買戰力（(ii) vs (iii)）** → **採 (ii)**（單一幣、coin 可買 PvP 戰力）；κ-sweep 證 F6 量級耦合在 per-submission 序列化下不致 whiplash（§3）。穩定靠不變式 S1（α 低），非靠拆幣。

---

## §6 驗收 / invariant 測試（spec 階段先列，build 時落測）

- **F1（收緊）**：UI/endpoint 路徑審計——PvP faction 來源 = 玩家顯式選擇，**PvP 路徑讀取 will-personality ＝ 零**（無自動投影、且 picker 無 will-衍生 default/hint/推薦）。測：同一玩家不同 authored will 下可帶同一派系挑戰、且 picker 預設與 will 無關；authoring 流程不讀 PvP 結果。
- **F2**：source/sink 表審計——無任何函式同時讀 coin 又寫 Rank（或反之）。測：challenge 結算不動 wallet；wallet spend 不動 Rank。
- **F3**：`TICKET_COST`/`DEFENSE_*` 對所有派系相同。測：三派門票/升級成本與效果張量逐元相等。
- **F4**：grep 證 PvP 路徑不呼 `ecology.submit()`；若 PvP 產 session 則 `run_id` 非空（被生態/分析清洗律排除）。
- **F5**：多樣性 observable 計算只吃 authoring 提交流；單元測喂入 PvP 事件應對 entropy **零影響**；PvP UI 不顯露超過稀缺提示的族群分佈。
- **F6 / S1（穩定）**：κ-sweep 已證 per-submission 序列化下 F6 不致 whiplash（採 (ii)）。守：稀缺顯示細粒度刷新、`submit()` 維持每筆更新 window+權重（**無 batch-against-frozen-snapshot**）。可選 regression＝跑 `ecology_reduced_form_bifurcation.py` 確認 production 有效 α < α\*(κ)（§3 表）。
- 經濟健全：wallet ledger 累加 = balance（對帳）；sink 不可使 balance < 0。

---

## §7 不在範圍 / 明確延後

- **真玩家地牢 + 零和 Rank 轉移 + 防禦收入**：重新引入「PvP 是否第二算子」疑慮 → **gate 在 firewall 重審 + 研究軌 g\***（先知道定向壓力預算，再決定零和 Rank 的 effective g 是否落在 g\* 下）。
- **行為宣稱（人在競技階梯下會不會同質化）**＝研究軌 B，需以 reduced-form g\* 為 null，另份 pre-reg。
- **雙生券（局內雙人格平行跑）**：parking C 🟡，rename + 玩家價值正當化，待定。
- **欄位券 source、配對分段/重複遞減**：記著，本版未納。

---

## §8 落點（接口/檔案，build 時參照）

- **後端**：新 `api/wallet_manager.py`（singleton + JSON save/load，仿 [pvp_manager](api/pvp_manager.py#L150)）：`balance(player)`、`credit(source, amount)`、`debit(sink, amount)`、append-only ledger。
- **endpoints**（`api/server.py`，仿 [/pvp/*](api/server.py#L1693)）：`GET /wallet`、`POST /wallet/spend`（門票/防禦，ValueError→422 餘額不足）；challenge 流程串門票扣款（**先扣門票、後判勝負**，扣款與 Rank 結算分離以守 F2）。
- **防禦升級**：掛在 `pvp_manager.Dungeon`（archetype-agnostic 的 hold 加成欄），challenge 勝率/stake 讀它；**不**新增派系維度。
- **前端**（/mnt/c repo）：PvpScene 加錢包顯示 + 門票確認 + 防禦升級按鈕；維持派系自由選（F1）。

---

## §9 Provenance / 與專案關係

- 上游：Increment 1（研發日誌 §七，commit `c373efc`/`f98e050`）；生態閉環（[ecology_tracker.py](api/ecology_tracker.py)）。
- 並行：reduced-form bifurcation pre-reg v2（研究軌，`docs/experiments/ecology_reduced_form_bifurcation/`）——本份的 sink 數值校準與「零和延後」皆 gate 在其 g\*。
- 對齊記憶：`eco-dp-directional-pressure-reframe`（PvP 降隔離 feature、A/B/C moot）、`personality-ecology-layer`、`game-vision-original`（diversity 才是真目標）。
- 主張：R3/C ＝**用隔離換研究乾淨**——PvP 好玩照舊，但**結構上不可能**污染多樣性結論。
