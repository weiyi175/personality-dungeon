# 地牢 Counter-Policy（L0↔L1 介面）— 規劃 v1

> **STATUS：DRAFT — combat 層 PARKED（2026-06-19，待 build 決策）。未實作。**
> ⚠ **本稿 §2–§7 多處已被本 session 的實驗推翻；以下 RECONCILIATION 為準，內文未逐段重寫（debt paid by banner）**：
> - **§1 命門假設 H_counter 已證偽**（9 桶最快崩壞方向全 = +v1；proximity 只量 v1-v2 平面，7/9 維對崩壞隱形）→ **counter-matrix 無法從幾何「測量」，combat M 必須 authored（修法 C）**。**§2.2「結構靠測量不靠美學」作廢**——本案有意識選 authored 剋制表。
> - **factions 改採 Option 4+**：**不**從 trait 分簇（A0 修正：真人遺言是 bold↔cautious 主軸 PC1+殘差，**非 3 簇**，PC1=38%）；3 faction = **PC1-PC2 平面 3 個 120° 方向 argmax**（真 partition，三派都有質量）；9 trait-argmax 桶降為**連續 individuation / substrate**。「野＝PC2＝多疑↔樂觀」是假設非測量，**收回**。
> - **parking-lot D 已決**：生態與 combat **共用 taxonomy/topology，不共用增益**；唯一約束＝**生態 payoff 須 a≥b**（共存，2026-06-19 Exp B 重驗 maxRe⟺sign(b−a)）；cr＝自由平衡旋鈕（≲0.8）。
> - **模擬階段結束**：剩兩個真正開放項——**combat M 箭頭+數值**、**第三派主題**——皆 **authored，只能 build+playtest，不可模擬**。
> 脈絡見 §1/§1b 與 `地牢設計_待回歸問題_parking_lot.md`。原「地基/命門/v1建法」框架見下，**讀時以本 banner 覆寫**。

---

## 0. 為何這是「一切的起點」

「地牢剋制主人」要是**真學習問題**，必須先成立：**不同 playstyle 的弱點方向不一樣**。
- 若成立 → counter-matrix 結構存在，地牢「估主人分佈 + 最佳回應」有意義，挑戰者「偵察偏食→帶剋制 build」有空間。
- 若不成立（一招 v1 打天下）→ 剋制無意義、PVP 無博弈、§成長/§等級/§PVP **全部空轉**。

所以這不是工程問題，是**先要回答的經驗問題**（§1）。

---

## 1. 命門假設與決定性測試

**假設 H_counter**：對 9 個 playstyle 桶，「最快崩壞的攻擊方向」彼此**不同**（存在 counter-matrix 結構）。

**決定性測試（建任何東西前先跑）**：用現有引擎，對每個桶的代表 playstyle，掃多個攻擊方向，量 time-to-collapse。
- **發散**（每桶有自己的弱點方向）→ H_counter 成立 → 填 `M`、往下建。
- **全收斂到 v1**（一招打天下）→ H_counter 證偽 → 「剋制主人」要重想：需在裝置加**玩家抵抗 / 多維攻擊面**，造出剋制結構後再回來。

### ✅ 測試結果（2026-06-19）：H_counter **證偽**（design-fact，非偶然）

**證據鏈**：
1. **讀碼（design-fact）**：experiment 用的 proximity ＝ `compute_bifurcation_distance(mode="projection")` ＝ **只量 v1-v2 平面投影** `hypot(v1_proj, v2_proj)/ε_c_app`（[bifurcation_detector.py:107-118](simulation/bifurcation_detector.py#L107)）。**9 維中有 7 維對崩壞完全隱形。** `compute_sensitive_direction` **永遠回 V1**（L151）；apply-event 只有 aligned(v1)/random。
2. **實測（一步幾何掃描，9 桶 × 211 方向，eps=0.02）**：9 個桶的「最快崩壞方向」**全部 = +v1**，cos(dir,v1)=**1.000**、100% in-plane；9 條最快方向兩兩 |cos| **min=mean=max=1.000**（同一條）。
3. **解析理由**：proximity ∝ 累積推力在 v1-v2 平面的投影，故 time-to-collapse ∝ 1/`hypot(d·V1, d·V2)`，對**任何** playstyle 都在 d=±v1 最快。V1 佔 99.93% 敏感變異 → v1 一招打天下。

**caveat（誠實）**：此為一步幾何版（未含 replicator 動力學）。但動力學至多造成各桶**崩壞速率**差異，**不會改變「哪個方向」最快**——因 (a) proximity 只看 v1-v2 投影、(b) 裝置攻擊硬編 v1。方向結論穩固。

**結論**：**現有 L0 不存在 counter-matrix 結構**——脆弱性被 proximity 度量壓成 1 維（v1）。「剋制主人」**在現裝置上不可建**。⇒ 進 §1b。

### §1b 意義與修法方向（路標，非死路）

**根因**：P7-H 裝置與 PVP counter-game 的需求**正好相反**：
| | P7-H 研究 | PVP counter-game |
|---|---|---|
| 想要 | **一招可靠打爆所有人**（乾淨 DV、大效應 d=3.25）✅ | **不同攻擊剋不同 playstyle**（counter-matrix）❌ |
| 結果 | v1 一維脆弱性正是優點 | v1 一維脆弱性正是死穴 |

⇒ **counter-matrix 結構必須「設計進去」，不能從現有幾何「發現」**（現有幾何剛被證明沒有結構）。PVP 層需要**自己的、playstyle-相關的失敗模型**，與研究裝置的 1 維 bifurcation DV **解耦**。三個修法家族（待討論）：
- **A：playstyle-相對崩壞**——每個 playstyle 有自己的失敗軸（攻擊型過度延伸/防禦型被淹沒…），弱點方向**按設計**因 playstyle 而異。
- **B：多維攻擊面 + 抵抗**——崩壞需攻擊命中 playstyle 的**真弱軸**，playstyle 沿強軸可抵抗 → counter 結構寫進結算規則。
- **C：換 DV**——PVP 結算不用 proximity，改用設計好的剋制表（如資源/HP 對戰，事件×build 經 counter-matrix 互動）。

> 連帶問題已一併回答：現況 aligned 100% 崩壞＝「怎麼打都贏」＝無博弈，正是 1 維脆弱性的同一個病。要的不是更強的攻擊，是**多維、挑對象**的脆弱性。

---

## 2. counter-matrix `M` — v1 設計

### 2.1 定義
```
M[a, b] ∈ [0,1] = 攻擊型 a 對 playstyle 桶 b 的「崩壞效力」
              （例：1 − 正規化存活時間，或崩壞機率）
```
- **列（攻擊型）**：K 個 canonical 攻擊方向（§2.2）。
- **欄（playstyle 桶）**：9 桶（trait argmax，接既有粒度決策）。
- **值**：**用引擎校準**（量「桶 b 族群在攻擊 a 下的 time-to-collapse」），然後**凍結成遊戲常數**。

### 2.2 攻擊型怎麼定（M 的列）
- 每個攻擊型 a = 一個 **9D 單位方向 `d_a`** + intensity/cadence profile（一個具體的事件序列參數化）。
- **起始假設 K=9**：攻擊-i = 實測上最快崩壞「桶-i playstyle」的方向。
- 命門測試後若弱點方向**聚成更少群** → 縮 K（例如只有 3 個有效方向）。
- ⚠ **不預設 M 是對角或循環**（RPS cyclic 結構我們的 L3 研究證明很難自然得到，不假設）；**M 的結構靠測量，不靠美學**。

### 2.3 地牢用法（估分佈 → 最佳回應）
```
q_B   = 主人 B 的 Trace → 9 桶經驗分佈（recency × clarity 加權）
score_a = Σ_b q_B[b] · M[a, b]              # 攻擊 a 對 B 分佈的期望效力
π_B   = softmax_λ(score_a)                  # 攻擊組合；λ = 集中度旋鈕
```
**挑戰結算**：A 攻 B 地牢 → 抽 `a ~ π_B` → A 崩壞機率 = `M[a, bucket(A)]`。
- A 玩 B 常玩桶 → π_B 剛好剋它 → 被剋。
- A 玩 B 罕玩桶 → π_B 沒覆蓋 → survive。＝「偵察偏食→帶剋制 build」精確機制。

### 2.4 ★ 集中度 λ ＝ 過擬合/泛化旋鈕（接成長稿）
- **高 λ**：π_B 尖銳集中在主人最常玩的桶 → 對那些桶超強，但**off-distribution 挑戰者一鑽就破** ＝ 過擬合。
- **低 λ**：π_B 鋪開 → 泛化、難破，但對主人主力桶沒那麼致命。
- ⇒ 地牢的**可破解性 = q_B 的分散度（主人多樣性）× π_B 的溫度 λ**。這把成長稿 §3「多樣性=泛化=可破解性」**機制化**了，λ 是它的連續旋鈕。

### 2.5 退化守衛（建前必查）
若校準後 `M` 出現**某攻擊型對所有桶都高效**（column/row 霸權）→ 遊戲退化成「永遠丟那招」。守衛：① 約束攻擊型（成本/CD/互斥）；② 在裝置加玩家抵抗讓錯配攻擊真的打不動。**命門測試順便檢查這個。**

---

## 3. counter-matrix `M` — v2 設計（前向相容，先設計避免過渡受阻）

### 3.1 學習式 M（per-dungeon）
- v1 的全域凍結 `M` → 變成 **prior**；每座地牢從**真實 PVP 結果**貝氏更新自己的 posterior `M_B`。
- 前向相容：**v1 的 M = v2 的先驗**，v2 只在上面加線上更新。v1 資料結構**原封不動被重用**。

### 3.2 連續 9D 攻擊
- v1 的 K 個離散攻擊型 → 變成連續效力函數 `E(d, p)`（d=9D 攻擊方向，p=9D playstyle 向量）。
- 前向相容：**v1 的 K 個 `d_a` = v2 連續球面上的 K 個取樣點**；v2 在其間內插/最佳化。**v1 的 9 桶 = v2 連續空間的粗網格**。

### 3.3 局內玩家操作
- v1：playstyle 由遺言在開局鎖定（單桶/單點）。
- v2：加局內選擇 → playstyle 變成**會演化的軌跡**，M 的「欄」從點變成軌跡，地牢需**局內重估桶**。
- 前向相容：**v1 的單桶 = v2 的初始態**；v2 加中途 re-bucketing。

### 3.4 ★ 前向相容總原則（核心，避免 v1→v2 阻礙）
> **v1 是 v2 的「粗離散 + 凍結 + 單點」版本；v2 是「連續 + 學習 + 軌跡」版本。三條升級各自只是把 v1 的某一維解凍：**
> | 維度 | v1 | v2 | v1 如何被重用 |
> |---|---|---|---|
> | M 的值 | 凍結常數 | 線上學習 | v1 值 = v2 prior |
> | M 的列 | K 離散方向 | 連續 9D | v1 方向 = v2 取樣點 |
> | M 的欄 | 9 桶/單點 | 連續軌跡 | v1 桶 = v2 初始網格 |
>
> **零丟棄原則**：只要 v1 的資料 schema（§4）把「攻擊型」存成 9D 方向、把「桶」存成 9D 質心，v1 的一切就都是 v2 的特例，過渡無需重建。

---

## 4. 資料 schema（為前向相容設計）

```
AttackType   { id, direction_9d, intensity_profile, cadence_profile }   # 9D 方向 → v2 連續可用
PlaystyleBucket { id, centroid_9d }                                      # 9D 質心 → v2 連續可用
CounterMatrix   { values: M[attack_id][bucket_id], calibrated_at, frozen:bool }  # frozen=false 即進 v2
DungeonPolicy_B { q_B(recency×clarity 加權), lambda, (v2: M_B posterior) }
Trace           { will_9d, bucket_id, clarity, outcome, survival }       # 餵 q_B
```
關鍵：**所有東西都掛著 9D 座標**，所以離散↔連續可無痛互轉。

---

## 5. 接到現有裝置（有界擴充，非重寫）

現裝置已會「沿一個方向（v1）施加持久事件」。擴充 = **把單一 aligned 方向，換成一組 `d_a`，地牢按 π_B 在其中選**。每個攻擊型 = 既有 apply-event 的方向 + intensity/cadence 參數化。是有界擴充。

---

## 6. 待測 / 待拍板

1. **命門測試（最高優先）**：9 桶弱點方向是否發散？（決定地基在不在）
2. **K（攻擊型數）**：測完才定（9？或聚成更少）。
3. **M 的值**：引擎校準後填。
4. **退化守衛**：是否需要約束攻擊型 / 加玩家抵抗。
5. **effectiveness 度量**：time-to-collapse vs 崩壞機率 vs 存活比，擇一。

---

## 7. 一句話

地牢 counter-policy = **估主人 playstyle 分佈 q_B → 對凍結 counter-matrix M 做最佳回應 π_B（溫度 λ＝過擬合/泛化旋鈕）→ 部署成事件序列**。整套站在命門假設「不同 playstyle 弱點方向不同」上，**建前必先實測**。v1=離散+凍結+單點，v2=連續+學習+軌跡，**v1 是 v2 的粗特例、零丟棄**（schema 把攻擊型/桶都掛 9D 座標即可無痛過渡）。
