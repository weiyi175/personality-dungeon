# 研究價值與文獻定位報告
## L3-Bottleneck Dynamics 線 — Sampled Synchronous Replicator 與 RPS 旋轉

> **日期**：2026-06-22
> **目的**：對照已發表文獻，誠實評估本研究 dynamics 發現的學術定位與價值。
> **方法**：精讀 6 篇相關論文全文/全文摘要，逐軸比對本研究的 b4 winding 結果與負結果。
> **一句話結論**：本研究的 dynamics 現象**獨立重新發現了既有的有限族群 quasi-cycle / coherence-resonance 結果**；其價值**不在** dynamics 主張本身，**而在**(a) 方法論嚴謹度（pre-registration + mean-field 正控制 + metric 自我證偽）、(b) 跨方法/跨領域的獨立複現、(c) 一個可複現的「指標偵測陷阱」案例。

---

## 1. 本研究 dynamics 線在做什麼

在一個跨玩家人格生態遊戲的 agent-based、sampled synchronous replicator substrate 上，問：**離散抽樣的 replicator 能否忠實保留 RPS 旋轉（cycle_metrics 的 Level 3）？**

兩個核心經驗結果（皆 pre-registered + 10-seed confirmatory，commit f07ab38）：
- **負結果（B5 keystone 等）**：同一 generator 下，確定性 mean-field 完美旋轉（consistency=1.0），其抽樣孿生塌至 consistency≈0.51（step-local turning 判據下 = 無方向旋轉）。
- **b4 反例（metric-sensitivity 補充，2026-06-21）**：state-dependent selection intensity（k_eff 隨 dominance 變，β-law）下，抽樣系統的**可觀測 proportion 仍帶穩定、定速、線性 window-scaling 的 winding 旋轉**（~2.85 turns/1000，品質等同 mean-field），但 step-local turning-consistency 因抽樣 jitter 讀成 0.51 而**漏看**。⇒「sampling destroys rotation」非普適，是 metric-defined。

---

## 2. 文獻對比表（本研究 vs 其他作者）

| 研究 | 確定性 mean-field | 噪聲對旋轉的作用 | selection intensity | 偵測法 | 方法類型 | pre-reg / 正控制 |
|---|---|---|---|---|---|---|
| **Traulsen, Claussen & Hauert (2006)**, *Phys. Rev. E* 74:011901 | — | sampling 噪聲 ~1/√N，N→∞ 回確定性 replicator（**「失真源＝抽樣」的解析證明**） | **常數 w** | 不偵測旋轉（只 2×2 PD/Snowdrift，無 RPS） | 解析 Fokker-Planck/Langevin | 不適用 |
| **McKane & Newman (2005)**, *Phys. Rev. Lett.* 94:218102 | 穩定 spiral（無持續環） | 噪聲**創造** coherent quasi-cycle（resonant amplification） | 常數 | power spectrum | 解析（linear noise approx） | 不適用 |
| **"Intrinsic noise & cycles"** (arXiv:1006.0825), RPS quasi-cycle | 穩定固定點 | 噪聲**創造** 大振幅 coherent quasi-cycle，可觀測頻率可見；解析預測出現的參數區域 | 常數 w=1 | power spectrum | 解析 van Kampen 展開 + Gillespie | 不適用 |
| **Yang, Rogers & Dawes (2017)**, *J. Theor. Biol.* 432:157 | 帶 mutation 的吸引極限環 | 噪聲**拖慢**環（週期↑∝1/mutation），不摧毀 | 常數 | hyperplane crossings | IBM + ODE | 無（標準） |
| **RSP via QBD process (2016)**, *Sci. Rep.* 6:28585 | 中性中心 | 噪聲隨強度誘發 quasi-heteroclinic cycle | 常數（state-dependent 指 birth-death **轉移率**，非 selection 強度） | 極限分布 heatmap | 解析 QBD 過程 | 無 |
| **Reichenbach, Mobilia & Frey (2007)**, *Nature* 448:1046（空間 RPS） | 中性環 | 有限-N 噪聲使軌跡 spiral-out → 物種滅絕/固著 | 常數 | 物種密度時間序列 | 空間 IBM | 無 |
| **Szolnoki, Mobilia, Jiang, Szczesny, Rucklidge & Perc (2014)**, *J. R. Soc. Interface* 11:20140735（綜述） | — | 整理上述全部：finite-N 旋轉、quasi-cycle、selection intensity 與共存時間**非單調** | 常數（全域） | spectral / 共存時間 | 綜述 | — |
| **本研究 (2026)** | **持續旋轉**（b4 mean-field consistency=1.0） | β=0 抽樣**摧毀**(random walk)；β>0 state-dependent intensity **保留**(steady winding) | **state-dependent**（k_eff 隨 dominance；本研究獨有的旋鈕） | **winding vs step-local turning 並陳，揭兩者不一致** | **agent-based applied multi-agent system** | **是（pre-reg + mean-field 正控制 + metric 證偽）** |

---

## 3. 逐篇定位（誠實對比）

**Traulsen-Claussen-Hauert (2006)** — 本研究的「失真主因＝抽樣離散層」這個因果主張，他們 20 年前已用 Langevin 方程解析證明：有限族群雜訊唯一來自隨機更新（抽樣），量級 ~1/√N，N→∞ 消失。**本研究的 mean-field-vs-sampled 對照 = 此結果的數值化重演。** 差異僅：他們只處理 2×2 賽局、不碰 RPS。

**McKane-Newman (2005) + arXiv:1006.0825** — 「有限族群 RPS 在可觀測頻率上出現 coherent、可被穩健指標（power spectrum）偵測的旋轉，且能解析預測其出現的參數區域」這整個現象，是 coherence-resonance / quasi-cycle 正典。**本研究的 b4 winding ＝這塊地的一個經驗點；他們有解析理論、相圖與功率譜。** 關鍵差異（見 §4）：他們的確定性系統是穩定固定點、噪聲**創造**環；本研究 b4 確定性系統本身就轉、抽樣**摧毀/保留**之分由 selection intensity 的 state-dependence 調控。

**Yang-Rogers-Dawes (2017)** — 與本研究負結果最像：mean-field 有環、有限族群重現不出來。差異：他們的結論是「拖慢」非「step-local 摧毀」，且用 hyperplane-crossing（穩健、非 step-local）偵測——這恰恰佐證**全領域早已避開 step-local turning 這種脆弱指標**，使本研究的 metric-disagreement 對領域不算新聞。

**QBD RSP (2016)** — 我原本最擔心它涵蓋 b4，精讀後否定：它的「state-dependent」指 birth-death **轉移率**（有限族群通用），**selection intensity 是常數**，且明確**沒有** random-walk→rotation 的轉變。**與 b4 的旋鈕不同物。**

**Reichenbach-Mobilia-Frey (2007)** — 有限-N RPS 的另一條正典（spiral-out→滅絕）。佐證「有限族群破壞 RPS 旋轉」是被反覆研究的成熟問題。

**Szolnoki-Perc 等 (2014) 綜述** — 把上述全部收進一篇權威綜述，並記載「selection intensity 與共存時間呈**非單調**關係」——本研究 b4 的 β=0.3 最佳／β≥0.6 over-damp 的非單調形狀，與此高度同形。

---

## 4. 誠實的新穎性裁決

**不新穎（已被反覆發表）**：
1. 「sampled replicator 無法忠實保留 RPS 旋轉、失真源＝抽樣離散層」＝ Traulsen et al. + Yang-Rogers-Dawes。
2. 「有限族群 RPS 的可觀測 coherent 旋轉、需穩健（spectral/winding）指標偵測」＝ McKane-Newman + 1006.0825。
3. 「step-local turning 會漏看旋轉」＝ 領域早用 crossing/spectrum，非新聞。

**字面上未被涵蓋、但不足以成獨立貢獻**：
- b4 的精確旋鈕——**state-dependent selection *intensity*（非全域常數 w，而是隨 dominance 變的回饋）作為旋轉的控制參數**——不在上述 constant-w 文獻裡。
- **但**要把它變成貢獻，需補：(a) 證 b4 mean-field 是真極限環非 spiral；(b) 隔離 state-dependence 是否本質（vs 只是換個方式調有效強度）；(c) Hopf/bifurcation 解析。**這三項本研究皆未做（「why」已撤回為開放）**，且補完＝進入 quasi-cycle/Hopf-RPS 這個擁擠且解析門檻高的場、高機率塌回「coherence resonance 的一個實例」。

**裁決**：dynamics 線作為**對 EGT 理論的獨立學術貢獻 → 不可行，建議封存**。

---

## 5. 那麼，本研究的價值在哪（盤點後的誠實結論）

dynamics 主張雖是重新發現，但這份工作仍有**三項可對外陳述的真實價值**：

1. **方法論示範（最強）**：在 agent-based EGT 模擬中，多數論文**不做** pre-registration，也少有 mean-field 正控制。本研究全程 pre-registered，並以 mean-field 正控制把「負結果」從「沒觀察到」升級為有因果對照；更進一步用 **metric-sensitivity 分析主動證偽自己**（親手查出 b4 反例 + 文獻 rediscovery）。**這套「預登記→正控制→自我證偽→文獻定位」的紀律，本身是可展示的研究素養**——對指導老師而言，能精準把自己的工作放進文獻、並誠實判定新穎性，正是研究者與調參者的分野。

2. **跨方法/跨領域的獨立複現**：既有 quasi-cycle 結果多來自解析（van Kampen）或生態/流行病 IBM。本研究在一個**全新的應用 apparatus**（跨玩家人格生態 + RL 地牢遊戲）上獨立重現同一現象——複現本身在當前可重現性危機下有價值。

3. **一個可複現的指標偵測陷阱案例**：本研究記錄了一個具體、可重跑的失敗模式——**step-local turning-consistency 在有限族群 jitter 下製造假陰性**（把 b4 的真旋轉判成 chance）。對任何要建類似 cycle-detection pipeline 的人，這是一個有用的 cautionary case（即使領域已偏好穩健指標，一個被完整 instrument 的反例仍有教學/工程價值）。

**建議定位**：把 dynamics 線封存為「rigorous replication + methodological exemplar」，並把研究重心移回**應用系統**（人格生態遊戲 apparatus、HCI/games-AI 線）與 **ecology directional-pressure** 設計線——那些不與 EGT theory 正面競爭，是本 project 真正稀缺之處。

---

## 6. 引用清單

- A. Traulsen, J. C. Claussen, C. Hauert (2006), "Coevolutionary dynamics in large, but finite populations," *Phys. Rev. E* 74, 011901. arXiv:cond-mat/0607270.
- A. J. McKane, T. J. Newman (2005), "Predator-prey cycles from resonant amplification of demographic stochasticity," *Phys. Rev. Lett.* 94, 218102.
- "Evolutionary dynamics, intrinsic noise and cycles of co-operation," arXiv:1006.0825（RPS coherent quasi-cycles，van Kampen 展開）.
- L. Yang, T. Rogers, A. Dawes (2017), "Demographic noise slows down cycles of dominance," *J. Theor. Biol.* 432, 157-168. arXiv:1708.00632.
- "Stochastic Evolution Dynamic of the Rock–Scissors–Paper Game Based on a Quasi Birth and Death Process" (2016), *Sci. Rep.* 6, 28585.
- T. Reichenbach, M. Mobilia, E. Frey (2007), "Mobility promotes and jeopardizes biodiversity in rock-paper-scissors games," *Nature* 448, 1046-1049.
- A. Szolnoki, M. Mobilia, L.-L. Jiang, B. Szczesny, A. M. Rucklidge, M. Perc (2014), "Cyclic dominance in evolutionary games: a review," *J. R. Soc. Interface* 11, 20140735.

> 本報告所有對比基於 2026-06-22 對上述論文之精讀；本研究側的數據與判定鏈見 `reports/experiments/l3_bottleneck/`（pre-reg、phase2 confirmatory、metric_sensitivity/FINDINGS.md）。
