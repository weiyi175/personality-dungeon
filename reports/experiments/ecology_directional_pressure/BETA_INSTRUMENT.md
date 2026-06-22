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
n_real       : 2  (total subs 210)
scarcity_std : 0.0092
```

210 筆中只有 **2 筆真人**（其餘 208 為 sim/replay，run_id 非空）。雙重旗標：n=2 遠低於 30，
且那 2 筆面對的稀缺近乎恆定（std 0.0092 < 0.02 門檻）。**β 現在不可估**——與甲完全一致。

## 5. 與專案的接合

- **解除阻塞的對象**：ECO-DP 的 `g*(β)`（`ECO_DP_RESULTS.md` §4b）以 β 為唯一開放輸入。本器把
  「未來怎麼把 β 量出來」變成一個**已驗證、可執行**的離線步驟——乙一收集到真人資料就直接餵進來。
- **乙（行為版收集）的明確需求**：(a) ≥~1000 筆真人 live-ecology 提交；(b) 真人面對的稀缺要
  **隨時間變動**（否則 UNIDENTIFIED）——收集設計要讓生態狀態在收集期間有真實漂移。
- **F-safe**：純讀 `ecology_state.json` 的離線分析，不新增算子、不碰 Rank/coin、不改提交流 → 不觸任何 firewall invariant。
- **可選增強（backlog，需前端 + 你跑 Godot 驗）**：在 `/ecology/submit` 加 optional `seen_scarcity`
  passthrough，記人**實際看到**的快照（而非 submit 當下伺服器重算的 q），消除 display-vs-submit 漂移。
  現版用已存的 `q_before` 為代理，低流量下足夠。
