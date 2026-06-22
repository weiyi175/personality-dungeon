# ECO-DP — confirmatory g-sweep 結果（live 算子）

> **狀態**: confirmatory run 完成（2026-06-22）。pre-reg：`docs/experiments/ecology_directional_pressure/DIRECTIONAL_PRESSURE_PREREGISTRATION.md`。
> runner：`scripts/experiments/ecology_dp_gsweep.py`（判定 **live** `EcologyTracker`，唯一改動＝`_fitness += g·d_i`）。
> driver：`softmax(β=2 · advantage)`、d=one-hot Defensive、10 seeds(50–59)、3000 rounds(burn 1000/tail 1000)。

## 1. 控制（§7）— 全 PASS
| 控制 | 判準 | 結果 |
|---|---|---|
| **H-NC**（g=0 守共存） | fixation ≤1/10 | **0/10**、stationary entropy 1.086 ≈ log3 ✅ |
| **H-PC**（g=5 強制 monoculture） | fixation =10/10 → argmax(d)=Defensive | **10/10**、top_q 1.000 ✅ |

→ driver/接線健康，sweep 可解讀。

## 2. H1 — 單調 crossover（**非** g=1 分岔）
| g | 0.0 | 0.5 | 1.0 | 1.5 | 2.0 | 2.5 | 5.0 |
|---|---|---|---|---|---|---|---|
| entropy | 1.086 | 1.037 | 0.858 | 0.560 | 0.236 | 0.043 | ~0 |
| top_q | 0.387 | 0.487 | 0.664 | 0.830 | 0.944 | 0.991 | 1.000 |
| fixation | 0/10 | 0 | 0 | 0 | 0 | 10/10 | 10/10 |

- entropy 隨 g **平滑單調**下降；g=1 時 top_q 僅 **0.66**（離 monoculture 遠）。**確認**：softmax 軟地板使動力學嚴格 interior、**無 g=1 transcritical 分岔**（pre-reg Finding 1）。
- treatment grid 頂(2.0) 未 bracket fixation（max_q≥0.95）→ 補 exploratory g∈{2.5–4.0}（pre-reg §6 Finding 3）才 bracket。

## 3. g\*_apparatus（H1/H2 主數字）
- **g\*_entropy-mid = 1.52**、**g\*_fixation(max_q≥0.95) = 2.06**。兩者皆 **> 解析 g\*=1.0**。
- ⇒ **live 算子比教科書 g\*=1 robust 約 1.5–2×**（軟地板再播種撐住 minority）。

## 4. H2-faithful — apparatus 對 g\* 的扭曲歸因（ablation）
baseline g\*_fix=2.06 vs 解析 1.0：

| 成分變動 | g\*_ent | g\*_fix | 解讀 |
|---|---|---|---|
| baseline (softplus, β2, W50, instant) | 1.52 | **2.06** | — |
| linear advantage（關 softplus） | 1.33 | 1.89 | softplus 小幅**上推** ~0.17（floor 稀有 advantage） |
| **β=1**（更軟選擇） | 2.05 | **2.98** | **主導**：軟→robust↑ |
| **β=4**（更銳選擇） | 1.17 | **1.58** | 銳→趨近解析 1.0（β→∞ 回 1） |
| window=25 / 100 | 1.49 / 1.53 | 2.07 / 2.06 | **可忽略** |
| lagged_weights eta=0.2 / 1.0 | 1.52 / 1.52 | 2.07 / 2.06 | **可忽略**（Finding-2 實測無感） |

**結論**：g\* 的 ~2× 上推**由 soft selection（β=2 的軟 softmax 響應）主導**、softplus 次要、**finite-window 與 EMA-lag 可忽略**。意義＝引擎的「額外 robustness」主要來自玩家**軟響應**（weak-selection）而非貪婪 argmax；選擇一旦變銳（β→∞），robustness 消失、g\*→解析 1.0。

**Finding-2 解決**：driver 用瞬時 advantage（繞過權重 EMA）的擔憂，經 lagged_weights 對照**實測對 g\* 無影響**——概念上的 faithfulness gap 存在但**結果不敏感**。

## 4b. g\*(β) 主曲線 + 機制診斷（完成 pre-reg §4 探索清單）

**★ g\*(β)——robustness vs 響應銳度（headline）**
| β | 0.5 | 1.0 | 2.0（錨） | 4.0 | 8.0 |
|---|---|---|---|---|---|
| g\*_fix | >4（極 robust） | 2.98 | **2.06** | 1.58 | 1.35 |

→ **單調遞減**：β→0（軟響應）g\*→∞（不可破）、β→∞（argmax/銳）g\*→解析 1.0。**整個 ECO-DP 收斂成一句：反同質化預算 = 遞減函數 of 玩家響應銳度 β；production-anchored β=2 → ~2×。真實 β 是唯一開放經驗量（② 行為版）。**

**hysteresis**（center-init vs mono-init 穩態）：所有 g 的 gap = **0.000** → **無 bistability、無遲滯** → 乾淨單吸引子 crossover（**確認 Finding 1**：crossover 非 saddle-node 分岔）。

**time-to-fixation**（center-init）：g=2.0→14 rounds、g≥2.5→即時；單調（越過 g\* 越快）。

**per-vertex d 對稱性**：Aggressive 2.075 / Defensive 2.063 / Balanced 2.026 → 解析對稱性**成立**（差 <0.05；Balanced 略低＝§2 投影不對稱 caveat 的有限痕跡）。

artifacts：`eco_dp_diagnostics.json`。

## 5. 主張映射（§9）
NC 守共存 + PC 全塌 + 存在有限可觀測 g\*（1.5–2×）+ ablation 歸因 → **因果主張成立**：neg-freq 引擎被 directional 壓力在可觀測 g\* 處 crossover；apparatus 把它從解析 ~1 上推 1.5–2×、主因 soft selection；儀器忠實性受檢。直命「metric-defined vs real mechanism」——**「real」反同質化預算 ≈ 解析值 2 倍**。

## 6. 範圍 / 延後
- ① 在「玩家以 softmax(β) 響應 coin 誘因」**假設模型**下回答「算子會不會被推塌、在哪」；**不**回答「真人是否真的這樣響應」＝② 行為版（pre-reg §11），gate 在本 g\* 已出。
- artifacts：`sweep_combined.tsv`、`eco_dp_analysis.json`、`eco_dp_ablation.json`。
