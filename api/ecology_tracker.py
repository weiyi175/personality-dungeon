"""
Personality Ecology tracker (meta-layer, Path B: rotation).

跨玩家人格生態評分系統。把複製動力學從「單 session 內 300 代理人的策略旋轉」
上移到 meta 層（跨玩家伺服器生態）：

  每筆冒險上傳 → 9D 人格投影成 3 原型（aggressive/defensive/balanced）
  → 用循環克制 payoff 算「該原型在當前生態的適應度」→ 評分（稀有且剋星少 = 高分）
  → 更新生態佔比與動態權重 → 週期性快照 → classify_cycle_level 判定 L0–L3 旋轉。

路 B（旋轉）：評分用 evolution.independent_rl 的循環 payoff（非遞移），目的是讓
熱門原型被其「剋星」的普及度壓分、稀有原型加分，驅動 RPS 式旋轉而非靜止平衡點。
設計依據與決策見 repo 根 `人格生態評分_規劃_v1.md`。

不污染 P7-H：獨立 JSON store（ecology_*.json），與 player_test_tracker 完全隔離。
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path

from evolution.independent_rl import STRATEGY_SPACE, strategy_payoff_matrix

# 原型 = 基質層三策略，語意一致（攻擊/防守/平衡）。
ARCHETYPES: list[str] = list(STRATEGY_SPACE)  # ["aggressive", "defensive", "balanced"]
_NARCH = len(ARCHETYPES)

# 9D 特徵順序（與 DungeonLifecycleController.FEATURE_NAMES / bifurcation_detector 一致）。
FEATURE_NAMES: list[str] = [
    "impulsiveness", "assertiveness", "optimism",
    "risk_aversion", "suspicion", "endurance",
    "randomness", "stability_seeking", "curiosity",
]
_FIDX = {n: i for i, n in enumerate(FEATURE_NAMES)}


def _f(vec: list[float], name: str) -> float:
    i = _FIDX[name]
    return float(vec[i]) if i < len(vec) else 0.0


def personality_to_archetype_soft(vec9: list[float], *, tau: float = 0.6) -> list[float]:
    """9D 人格 → 3 原型的軟比例（softmax，Σ=1）。

    投影軸（§2，係數待 §7-B 校準）：
      aggressive : impulsiveness + assertiveness − risk_aversion
      defensive  : risk_aversion + suspicion + endurance + stability_seeking
      balanced   : optimism + curiosity − 極端度
    """
    s_agg = _f(vec9, "impulsiveness") + _f(vec9, "assertiveness") - _f(vec9, "risk_aversion")
    s_def = (_f(vec9, "risk_aversion") + _f(vec9, "suspicion")
             + _f(vec9, "endurance") + _f(vec9, "stability_seeking"))
    extremity = sum(abs(x) for x in vec9[:_NARCH * 3]) / max(1, len(vec9))
    s_bal = _f(vec9, "optimism") + _f(vec9, "curiosity") - extremity
    scores = [s_agg, s_def, s_bal]
    m = max(scores)
    exps = [math.exp((s - m) / tau) for s in scores]
    total = sum(exps) or 1.0
    return [e / total for e in exps]


def _softplus(x: float) -> float:
    # 數值安全：大 x 退化成 x，避免 exp 溢位。
    return x + math.log1p(math.exp(-x)) if x > 0 else math.log1p(math.exp(x))


def _std(xs: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mean = sum(xs) / n
    return math.sqrt(sum((x - mean) ** 2 for x in xs) / n)   # population std（與 numpy 預設一致）


# 評分區間 → 金幣獎勵（neg-freq 重校 2026-06-19，跑 scripts/.../ecology_will_replay.py +
# piece2_calibrate）。λ=2.0（EcologyParams.lam）下分數域 ~[46,103]。200 檔＝你選的派系當下
# 跌到 ~6% 窗佔比（瀕危）→ 分數破百＝「救活瀕危派系」jackpot（replay ~3%，稀有）；下 4 檔在
# [55,100) 對均衡。(score 下界 inclusive, coins)；由高到低比對。換 fitness/λ 須重跑校準。
COIN_BRACKETS: list[tuple[float, int]] = [
    (100.0, 200),   # 救活瀕危派系：選的派系跌到 ~6% 窗佔比，分數破百
    (67.7, 100),    # 稀缺
    (60.5, 50),     # 偏稀缺
    (55.0, 25),     # 偏普及
    (0.0, 10),      # 墊底：派系過度普及
]


def score_to_coins(score: float) -> int:
    for lo, coins in COIN_BRACKETS:
        if score >= lo:
            return coins
    return COIN_BRACKETS[-1][1]


@dataclass
class EcologyParams:
    """路 B 旋轉參數。payoff (a, b, cross) 沿用 SDD §11.2 BL2 鎖定值。"""
    a: float = 1.0
    b: float = 0.9
    cross: float = 0.20      # 循環 payoff 的額外非對稱耦合（旋轉本來就來自 a,b 的 RPS 結構，非 cross）
    lam: float = 2.0         # fitness → advantage 銳度（neg-freq 校準 2026-06-19：λ=2，見 COIN_BRACKETS）
    base: float = 100.0      # 名目分數量級
    eta: float = 0.2         # 動態權重 EMA 學習率
    window: int = 50         # 計算生態佔比的滑動窗（最近幾筆上傳）
    bin_every: int = 10      # 每 N 筆上傳產一個生態快照（餵 classify_cycle_level）
    tau: float = 0.6         # 原型投影 softmax 溫度


@dataclass
class EcologySubmission:
    run_id: str
    session_id: str
    ts: float
    archetype: str
    archetype_soft: list[float]
    personality_9d: list[float]
    score: float
    coins: int                 # 實質獎勵：依評分區間給的金幣（落檔保存）
    score_components: dict
    outcome: dict = field(default_factory=dict)
    # 乙 β-instrument：author 前**前端實際顯示**給人的稀缺佔比 [q_agg,q_def,q_bal]（人看了才響應）。
    # 與 score_components.q_before（submit 當下伺服器重算的 q）分開存，消除 display-vs-submit 漂移；
    # 空＝前端未傳，分析退回用 q_before 為代理。純記錄、不入評分、不碰算子（F-safe）。
    seen_scarcity: list[float] = field(default_factory=list)


@dataclass
class EcologySnapshot:
    bin_index: int
    ts: float
    counts: dict
    proportions: list[float]   # [q_agg, q_def, q_bal]，Σ=1
    weights: list[float]       # 當前動態權重（per archetype）


class EcologyTracker:
    """In-memory singleton，append-only 上傳 + 滾動生態狀態。save/load 仿 P7-H。"""

    def __init__(self, params: EcologyParams | None = None) -> None:
        self.params = params or EcologyParams()
        self._submissions: list[EcologySubmission] = []
        self._snapshots: list[EcologySnapshot] = []
        self._recent: deque[int] = deque(maxlen=self.params.window)  # 最近原型 index
        # 動態權重初始＝中性 advantage softplus(λ·0)=ln2≈0.693（λ-independent）：
        # 擋空生態 cold-start 200-flood（init=1.0 會讓早期每筆 ~100 分→秒觸 200 檔）。
        self._weights: list[float] = [math.log(2.0)] * _NARCH

    # ── 生態狀態 ────────────────────────────────────────────────────────────────

    def _proportions(self) -> list[float]:
        """當前滑動窗的原型佔比（窗空時回均勻 1/3）。"""
        if not self._recent:
            return [1.0 / _NARCH] * _NARCH
        counts = [0] * _NARCH
        for idx in self._recent:
            counts[idx] += 1
        n = len(self._recent)
        return [c / n for c in counts]

    def _fitness(self, q: list[float]) -> list[float]:
        """每原型的負頻率依賴適應度 fitness_i = 1/N − q_i（稀有→正、普及→負）。

        取代 Path-B 的 RPS 循環 payoff（2026-06-19）：實作 2026-06-18 鎖定的
        「逐利→多樣性」意圖＝直接獎勵稀缺。線性、有界（q_i∈[0,1] → fitness∈[−⅔,⅓]），
        是抗 whiplash 的溫和形式（避開 1/q、−log q 那種 q→0 爆衝、可被「搶當第一個選死派系」
        exploit 的形式）。強度旋鈕＝params.lam（advantage=softplus(lam·fitness)）；
        a/b/cross/RPS payoff matrix 自此為死參數（Path B 有意識退役）。
        """
        uniform = 1.0 / _NARCH
        return [uniform - q[i] for i in range(_NARCH)]

    def _advantage(self, fitness: list[float]) -> list[float]:
        return [_softplus(self.params.lam * fit) for fit in fitness]

    # ── 上傳 ────────────────────────────────────────────────────────────────────

    def submit(
        self,
        *,
        personality_9d: list[float],
        run_id: str = "",
        session_id: str = "",
        outcome: dict | None = None,
        seen_scarcity: list[float] | None = None,
    ) -> dict:
        """收一筆冒險經驗：評分（基於上傳『前』的生態，避免自評）→ 更新生態。

        seen_scarcity：前端 author 前顯示給人的稀缺佔比（乙 β-instrument 用；純記錄、不入評分）。
        """
        soft = personality_to_archetype_soft(personality_9d, tau=self.params.tau)
        i = max(range(_NARCH), key=lambda k: soft[k])

        # 1) 用「加入本筆前」的生態算分，避免玩家自己墊高自己。
        q_before = self._proportions()
        fitness = self._fitness(q_before)
        advantage = self._advantage(fitness)

        # 2) 動態權重 EMA（路 B：不設 target，由循環 fitness 驅動旋轉）。
        for k in range(_NARCH):
            self._weights[k] = ((1.0 - self.params.eta) * self._weights[k]
                                + self.params.eta * advantage[k])

        score = self.params.base * self._weights[i]
        coins = score_to_coins(score)
        components = {
            "fitness": fitness[i],
            "advantage": advantage[i],
            "weight": self._weights[i],
            "q_before": q_before,
        }

        rec = EcologySubmission(
            run_id=run_id, session_id=session_id, ts=time.time(),
            archetype=ARCHETYPES[i], archetype_soft=soft,
            personality_9d=list(personality_9d),
            score=score, coins=coins, score_components=components,
            outcome=outcome or {},
            seen_scarcity=list(seen_scarcity) if seen_scarcity else [],
        )
        self._submissions.append(rec)

        # 3) 把本筆計入生態，必要時產生快照 bin。
        self._recent.append(i)
        if len(self._submissions) % self.params.bin_every == 0:
            self._emit_snapshot()

        return {
            "score": score,
            "coins": coins,   # 實質獎勵：依評分區間給金幣
            "archetype": ARCHETYPES[i],
            "archetype_soft": soft,
            "db_proportions": self._proportions(),
            "advantage": dict(zip(ARCHETYPES, advantage)),
            "weights": dict(zip(ARCHETYPES, self._weights)),
            "n_submissions": len(self._submissions),
        }

    def _emit_snapshot(self) -> None:
        q = self._proportions()
        counts = {ARCHETYPES[k]: 0 for k in range(_NARCH)}
        for idx in self._recent:
            counts[ARCHETYPES[idx]] += 1
        self._snapshots.append(EcologySnapshot(
            bin_index=len(self._snapshots), ts=time.time(),
            counts=counts, proportions=q, weights=list(self._weights),
        ))

    # ── 儀表 ────────────────────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        return {
            "proportions": dict(zip(ARCHETYPES, self._proportions())),
            "weights": dict(zip(ARCHETYPES, self._weights)),
            "n_submissions": len(self._submissions),
            "n_bins": len(self._snapshots),
        }

    def collection_diagnostics(self) -> dict:
        """乙 收集監測：真人 live 提交面對的稀缺**變異**夠不夠（鐵律 1）。

        真人判準＝session_id ∧ outcome 雙非空（與 β-instrument 一致）。對每筆取人**看到**的稀缺
        （seen_scarcity 優先，否則退 q_before），用本算子的 advantage 變換，回報跨筆 advantage std。
        對齊 β-instrument 閾值：< 0.02 → 不可識別；gate 0.06；設計目標 ~0.2。純讀、不改狀態（F-safe）。
        """
        advs = []
        for s in self._submissions:
            if not s.session_id or not s.outcome:
                continue
            q = s.seen_scarcity or s.score_components.get("q_before")
            if not q or len(q) != _NARCH:
                continue
            advs.append(self._advantage(self._fitness(list(q))))
        n = len(advs)
        if n < 2:
            return {"n_real": n, "scarcity_std": 0.0, "meets_gate": False,
                    "meets_target": False, "note": "真人 live 筆數 < 2，無法評變異"}
        arr = [[a[k] for a in advs] for k in range(_NARCH)]
        stds = [float(_std(col)) for col in arr]
        scar = float(sorted(stds)[len(stds) // 2])   # median over archetypes
        return {
            "n_real": n,
            "scarcity_std": scar,
            "meets_gate": scar >= 0.06,          # 識別性 + low regime 下限
            "meets_target": scar >= 0.20,        # power 設計目標（高變動）
            "note": ("達設計目標" if scar >= 0.20 else
                     "過識別 gate 但變異偏低、power 不足" if scar >= 0.06 else
                     "變異過低 → 接近不可識別，需驅動稀缺漂移"),
        }

    def assess(self, *, burn_in: int = 0, tail: int | None = None) -> dict:
        """把生態快照的原型佔比時序餵 classify_cycle_level → L0–L3 旋轉判定。"""
        if len(self._snapshots) < 4:
            return {"level": 0, "n_bins": len(self._snapshots),
                    "note": "bins<4，樣本不足，旋轉判定無意義（見 §5）"}
        from analysis.cycle_metrics import classify_cycle_level
        series = {ARCHETYPES[k]: [s.proportions[k] for s in self._snapshots]
                  for k in range(_NARCH)}
        result = classify_cycle_level(series, burn_in=burn_in, tail=tail)
        return {"level": result.level, "n_bins": len(self._snapshots)}

    # ── 持久化（仿 player_test_tracker：記憶體 singleton + JSON，停-改-重啟）──────

    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "ecology_state.json"
        with open(path, "w") as f:
            json.dump({
                "params": asdict(self.params),
                "submissions": [asdict(s) for s in self._submissions],
                "snapshots": [asdict(s) for s in self._snapshots],
                "weights": self._weights,
                "recent": list(self._recent),
            }, f, ensure_ascii=False, indent=2)
        return path

    def load(self, out_dir: str | Path) -> bool:
        path = Path(out_dir) / "ecology_state.json"
        if not path.exists():
            return False
        with open(path) as f:
            data = json.load(f)
        if "params" in data:
            pf = {k for k in EcologyParams.__dataclass_fields__}
            self.params = EcologyParams(**{k: v for k, v in data["params"].items() if k in pf})
        sf = {k for k in EcologySubmission.__dataclass_fields__}
        self._submissions = [
            EcologySubmission(**{k: v for k, v in s.items() if k in sf})
            for s in data.get("submissions", [])
        ]
        nf = {k for k in EcologySnapshot.__dataclass_fields__}
        self._snapshots = [
            EcologySnapshot(**{k: v for k, v in s.items() if k in nf})
            for s in data.get("snapshots", [])
        ]
        self._weights = data.get("weights", [math.log(2.0)] * _NARCH)  # 中性 fallback（與 init 一致）
        self._recent = deque(data.get("recent", []), maxlen=self.params.window)
        return True


_tracker: EcologyTracker | None = None


def get_tracker() -> EcologyTracker:
    """Process-wide singleton（與 _get_ab_manager / _get_tracker 同模式）。"""
    global _tracker
    if _tracker is None:
        _tracker = EcologyTracker()
    return _tracker
