# L3-BN Phase-2 Confirmatory Report

> **執行日期**：2026-06-18
> **預先註冊**：[docs/experiments/l3_bottleneck/L3_BOTTLENECK_PREREGISTRATION.md](../../../docs/experiments/l3_bottleneck/L3_BOTTLENECK_PREREGISTRATION.md)（commit ba8fd64，收案前鎖定）
> **選 cell 鎖定**：[docs/experiments/l3_bottleneck/phase2_cell_selection.json](../../../docs/experiments/l3_bottleneck/phase2_cell_selection.json)
> **分析腳本**：[scripts/experiments/analyze_l3bn_phase2.py](../../../scripts/experiments/analyze_l3bn_phase2.py)
> **原始輸出**：`reports/experiments/l3_bottleneck/phase2/`（per-cell summary TSV + per-seed CSV/provenance）

---

## 1. 結論（一句話）

確定性 mean-field replicator 達成**完美單向旋轉**（consistency=1.0、cycle_level=3、全 10 seeds），其**唯一差異為加入抽樣的孿生系統塌至 consistency≈0.51、cycle_level=2**；此塌陷在 B4/B5/T/C1 四個機制相異的 sampled cell（各 N=10）一致出現，L3 在 eta∈{.55,.60,.65,.70} 全閾值下 0 達成。**Level-3 旋轉的喪失被隔離為抽樣+均化離散化層的效應，而非 payoff 缺乏旋轉結構。**

## 2. Arm A — mean-field 正控制（H-PC）

| cell | n | cycle_level | consistency | 判定 |
|---|---|---|---|---|
| `g1_mean_field_delta0p000` | 10 | 全 = 3 | 全 = 1.0000 | ✅ **positive_control_holds** |

確定性系統（10 seeds 位元級相同）。旋轉結構在連續極限確證存在。

## 3. Arm B — sampled 處理組（H0-neg / H1-chance / H2-threshold-robust）

| cell | 家族 | n | **L3** | mean consistency | 95% CI | eta-sweep L3 (.55/.60/.65/.70) | 判定 |
|---|---|---|---|---|---|---|---|
| `g2_sampled_delta0p000` | B5（keystone，Arm A 抽樣孿生） | 10 | **0** | 0.5145 | [0.506, 0.523] | 0/0/0/0 | ✅ negative_holds |
| `beta0p60_k0p08` | B4 state-dep k | 10 | **0** | 0.5109 | [0.500, 0.521] | 0/0/0/0 | ✅ negative_holds |
| `t_lattice4_minibatch` | T local-minibatch（唯一有效 local 測試） | 10 | **0** | 0.5148 | [0.507, 0.523] | 0/0/0/0 | ✅ negative_holds |
| `g2_c1_small_world_b10p0_m0p5` | C1（原生窗 5000/1500/1500） | 10 | **0** | 0.3068 | [0.118, 0.496] | 0/0/0/0 | ✅ negative_holds |
| `g2_c1_small_world_b10p0_m0p5` | C1（鎖定窗 3000/1000/1000） | 10 | **1**(blip) | 0.4722 | [0.353, 0.591] | **1**/0/0/0 | ✅ negative_holds（容忍 ≤1） |

**keystone 因果對照**：同 B5 generator，mean-field consistency=**1.000**(L3) → 抽樣孿生 **0.5145**(L2)。離散化把一致性從 1.0 砸到 0.51。

## 4. 誠實的細節（據實報告，不美化）

1. **consistency 不是精確 0.50，而是 ~0.51。** B5/B4/T 三 cell 的 95% CI 不完全涵蓋 0.50（mean 0.511–0.515，d_vs_0.50≈0.74–1.27）。即存在**統計上可偵測、但旋轉上無關**的微弱殘留方向偏好——它比 L3 門檻 0.55 低一個量級，eta-sweep 在 0.55–0.70 全閾值 0/10 確認其不構成持續旋轉。**H1-chance 的精確版本應修正為「consistency 卡在 0.51 的弱偏置帶、遠離旋轉門檻」，而非「等於 chance」。**

2. **C1 那個 1/10 L3 是窗長縮短的噪聲假象。** 鎖定窗（3000/1000/1000）下 C1 出現 1/10 L3（max 0.5510 擦過 bar）；補跑的**原生窗（5000/1500/1500）給出 0/10**、mean 0.3068（遠低於 chance，over-homogenization 主動壓低一致性）。短窗 tail 較少 → consistency 估計變異大 → 偶發擦線。eta-sweep 也溶解此 blip（eta≥0.60 → 0）。兩者皆在容忍內，且互相佐證負結果。

3. **B4「最接近 bar」的 cell 回歸 chance。** Phase-1 的 3-seed mean=0.5306（曾是全場最接近 0.55），補到 10 seeds 後降到 0.5109——印證該「接近」是小樣本上偏波動，非真實趨近。

## 5. Deviations（相對 pre-reg）

| 項 | 內容 | 影響 |
|---|---|---|
| D1 | **C1 窗標準化**：c1_pairwise_scout 原生窗 5000/1500/1500 ≠ 鎖定窗 3000/1000/1000。主分析用鎖定窗（守單一協定承諾），另補原生窗作 robustness。 | 無損結論；原生窗更乾淨（0/10） |
| D2 | **B4 需 0.0 control**：b4_state_k 強制 `--beta-state-ks` 含 0.0 配對 control，故跑 {0.0, 0.6}。 | 無；0.0 control 亦 0/10 |
| D3 | **C2-uniform 排除**（pre-reg §3.2 已鎖）：數學退化（x_local≡x_global），未納入 confirmatory。 | 僅方法學警示 |
| D4 | **H1-chance 措辭修正**：consistency 為 ~0.51 弱偏置而非精確 0.50（見 §4.1）。 | 強化誠信；不改 L3 不可達主結論 |

## 6. 對論文映射

- 主張（pre-reg §9 第一列）達成：**Arm A mean-field L3/1.0 ＋ Arm B 四 cell 全 0–1/10 L3 ＋ eta-sweep 全 0**。
- 因果隔離成立：旋轉於連續極限存在、被抽樣+均化離散層摧毀；橫跨 B4/B5/T/C1 四機制。
- 正向邊界：paper_draft_v1 L0→L2（30/30，未重跑）為「機制可達上限」。
