# β-instrument：真人稀缺-響應銳度回收器

> **狀態**：BUILT + 自我回收驗證（2026-06-23）。純離線估計器，零 backend/前端改動 → 零 firewall 風險。
> **動機**：研究軌 ECO-DP 的唯一開放經驗量＝真實 β。甲（2026-06-22）證現有資料**不能** bound β。
> 本器把 apparatus **收集就緒**：一插真人就能算；沒真人時誠實判 INSUFFICIENT，不偽造數字。

## 1. 它估什麼

人在 author will **前**看到生態稀缺（前端 `fetch_snapshot()`「開局顯示稀缺型用」），
每原型的 advantage `adv_j = softplus(lam·(1/N − q_before_j))`（複刻 `ecology_tracker`）。
authoring 選擇建模為 conditional logit：

```
P(author archetype j | adv) = softmax(α_j + β · adv_j)
```

- **α_j** = intrinsic archetype 偏好（甲：真人 [.48/.43/.09]，balanced 薄）；α_balanced≡0 為基準。
- **β** = 稀缺響應銳度＝ECO-DP `g*(β)` 曲線的橫軸。**β=0 ⟺ 純 intrinsic、不理稀缺；β↑ ⟺ 越追稀缺**。

把 α 與 β 分離是關鍵：甲的 [.48/.43/.09] 是 α（intrinsic skew），不是 β。β 是全新的、要新收集才能量的東西。

## 2. 檔案

- 估計器：`scripts/experiments/ecology_beta_fit.py`
  - `load_real_submissions()` — 讀 `ecology_state.json`，篩真人（`run_id==""`，沿用 P7-H 清洗律）。
  - `reconstruct_advantage()` — 從每筆已存的 `score_components.q_before` 重建 adv（lam 取 `params.lam`）。
  - `fit_beta()` — MLE（scipy BFGS + analytic gradient），CI 由 observed-Fisher（inv Hessian）。
  - 三條 verdict guard：`INSUFFICIENT_N`（n<30）/ `UNIDENTIFIED`（稀缺零變異 → β 與截距共線）/ `OK`。
  - `--self-test` 合成自我回收。
- 測試：`tests/test_ecology_beta_fit.py`（9 passed）——自我回收覆蓋 β∈{0,1,2,4}、insufficient/unidentified verdict、advantage 重建複刻 tracker、β=0 CI 含 0。

## 3. 驗證（合成自我回收）

CI 在所有 true β 都覆蓋真值；點估計隨 n 收緊：

| true β | n=200 β̂ (95%CI) | n=1000 β̂ (95%CI) |
|---|---|---|
| 0.0 | 0.36 [−0.28, 1.00] | 0.07 [−0.22, 0.35] |
| 1.0 | 1.18 [0.50, 1.86] | 1.10 [0.79, 1.40] |
| 2.0 | 2.70 [1.89, 3.50] | 1.89 [1.56, 2.21] |
| 4.0 | 4.63 [3.59, 5.67] | 4.11 [3.68, 4.55] |

**樣本量靶（給乙）**：β≈2（ECO-DP 錨）時，**n≈1000 真人提交** → CI 半寬 ~±0.3，足以把
β=2 和 β=1 / β=4 分開（也就足以把 g\*(β) 的 robustness 預算釘在一個有用的區間）。
n=200 仍偏寬（半寬 ~±0.8）。

## 4. 真實資料的誠實裁定（2026-06-23）

```
verdict      : INSUFFICIENT_N
n_real       : 0  (total subs 210, artifacts dropped 2)
scarcity_std : 0.0000
```

210 筆中 208 為 sim/replay（run_id 非空）。**剩下 2 筆 run_id 空的並非真人**：
驗證發現兩筆 `session_id` 皆空、`ts` 僅差 ~21ms（程式/smoke 成對提交，非真人 author-under-scarcity）
→ 篩選加嚴為「run_id 空 **且** session_id 非空」後，**真正可用真人 β-觀測 = 0**。
（初版只看 run_id 把這 2 筆 artifact 當真人計入 n_real=2，2026-06-23 驗證踩到並修正。）
**β 現在不可估**——比「n=2」更乾淨地對齊甲：真人從沒走過 live ecology，收集根本還沒發生。

## 5. 與專案的接合

- **解除阻塞的對象**：ECO-DP 的 `g*(β)`（`ECO_DP_RESULTS.md` §4b）以 β 為唯一開放輸入。本器把
  「未來怎麼把 β 量出來」變成一個**已驗證、可執行**的離線步驟——乙一收集到真人資料就直接餵進來。
- **乙（行為版收集）的明確需求**：(a) ≥~1000 筆真人 live-ecology 提交；(b) 真人面對的稀缺要
  **隨時間變動**（否則 UNIDENTIFIED）——收集設計要讓生態狀態在收集期間有真實漂移。
- **F-safe**：純讀 `ecology_state.json` 的離線分析，不新增算子、不碰 Rank/coin、不改提交流 → 不觸任何 firewall invariant。
- **可選增強（backlog，需前端 + 你跑 Godot 驗）**：在 `/ecology/submit` 加 optional `seen_scarcity`
  passthrough，記人**實際看到**的快照（而非 submit 當下伺服器重算的 q），消除 display-vs-submit 漂移。
  現版用已存的 `q_before` 為代理，低流量下足夠。

## 6. Validity + Power（2026-06-23，`scripts/experiments/ecology_beta_power.py`）

在花成本收集前，先驗「量得準嗎 + 要收多少」。純模擬，用真實 intrinsic α=[.48/.43/.09]。

### 6A. Validity — 估計器是有效偵測器（mis-spec robust）

人若**不**照 softmax(α+β·adv) 響應，β̂ 還可信嗎？對幾種真實響應 DGP 套同一估計器（n=1500, high-var）：

| 真實響應 DGP | β̂ median | P(β̂>0 顯著) | P(誤報 null) | 判定 |
|---|---|---|---|---|
| softmax（自家模型） | 1.98 | 1.00 | — | ✓偵測（也回收真 β=2） |
| probit（換 link） | 2.50 | 1.00 | — | ✓偵測 |
| qlinear（對 −q 線性、不經 softplus） | 2.09 | 1.00 | — | ✓偵測 |
| rank（只認稀缺**排序**） | 7.45 | 1.00 | — | ✓偵測（絕對值膨脹） |
| **noresponse（null）** | 0.007 | — | **0.06** | ✓不誤報（≈名目 5%） |

**結論**：任何真實正向稀缺響應 → β̂>0 顯著（100%）；無響應 → β̂≈0、假陽性 ≈名目。
**但絕對 β 是 model-specific**（rank DGP 膨脹到 7.45）→ DGP 未知時，把 β 當**稀缺響應的單調指標 + 偵測器**用，
不當結構常數。要把 β 對接 g\*(β) 的**絕對**刻度，需另證真人響應確走 softplus(adv) link（或在 6 §可選增強裡記 seen_scarcity 後做 link 檢定）。

### 6B. Power/design — 給乙的收集協定

P(估得出 OK)=1.00（n≥100 皆可）；能否 **reject β=0** 與 CI 半寬取決於 n × **稀缺變動**。
各 regime 達成的 median scarcity_std：low 0.060 / mid 0.116 / high 0.221。

| 目標 | 低變動 (std .06) | 中變動 (std .12) | 高變動 (std .22) |
|---|---|---|---|
| **reject β=0**（人有理稀缺嗎），β≈2 | n≈3000 仍只 ~0.5 | n≈300（95%） | **n≈100–300（97–100%）** |
| **resolve 到 ±0.35**（分辨 β=1/2/4 → 釘 g\*(β) 區間），β≈2 | 達不到（n=3000 半寬仍 0.65） | n≈3000 | **n≈1000** |

**三條收集鐵律（乙 pre-reg 用）**：
1. **稀缺必須隨時間變動**——這是現狀真正死因：真實 2 筆的 scarcity_std=0.0092，**比最低 regime 還低**，
   再多 n 也是 `UNIDENTIFIED`。收集設計要主動讓生態狀態漂移（輪換顯示的稀缺型 / 跨時段收，別讓 q 釘死在 intrinsic）。
2. **只想證「人會追稀缺」**（reject β=0）：高變動下 **n≈300** 足夠。這是最低可發表門檻。
3. **想把 β 釘進 g\*(β) 的有用區間**（分辨 1/2/4）：高變動 **n≈1000**、中變動 n≈3000。對齊 §3 的 ~1000 靶。
