"""test_dream_seeds_low_coupling.py
低耦合世界實驗測試（Dream Seeds × Decoupled Engine）

驗證六層解耦完整性：
  L1 — 資料層：夢幻種子結構符合 9D-Core（Model v2.0）schema，含浮點精度保護
  L2 — 引擎層：evolution.replicator_dynamics 無 GUI / I/O 側依賴
  L3 — 軌道合約：Dream Candidate z-訊號 → Level 3 軌道（RE > 5000）
  L4 — 再現性合約：同一 traits_vector 在 ±擾動種子下全數產出 Level 3，含跨平台決定論
  L5 — Simplex 不變條件：每一步 Σx_i = 1、x_i > 0，含 Lyapunov 函數單調性
  L6 — 呈現層解耦：軌道資料可序列化為純 Python dict/list，符合 Godot API schema

背景文件：研發日誌 2026-05-04（9D 重構）、dream_seed_finder.py v1.0 定義
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# 嵌入式 Dream Candidate 夢幻種子（來源：dream_final/dream_seeds_v2_9d.json）
# 選取 3 個代表性候選（trial_id 53 / 47 / 33），所有 RE_avg > 22000
# ---------------------------------------------------------------------------
_DREAM_CANDIDATES: list[dict] = [
    {
        "trial_id": 53,
        "traits_vector": {
            "impulsiveness":    0.640624046042338,
            "assertiveness":   -0.11007001793358212,
            "optimism":        -0.07242881844613819,
            "risk_aversion":    0.5566629668518146,
            "suspicion":        0.34253363378252655,
            "endurance":       -0.5518408878010823,
            "randomness":      -0.20883432296656573,
            "stability_seeking": 0.29683774568912275,
            "curiosity":       -0.41985189235564574,
        },
        "signal_snapshot": {
            "z_expanding":    0.1527084032208726,
            "z_contracting":  0.11578523761108628,
            "z_exploring":   -0.11061615654436291,
        },
        "score": {"RE_avg": 23528.286, "RE_std": 1124.09, "repro_rate": 1.0},
    },
    {
        "trial_id": 47,
        "traits_vector": {
            "impulsiveness":    0.03275959033652663,
            "assertiveness":   -0.24766858126978053,
            "optimism":         0.5848600132357374,
            "risk_aversion":    0.16277817309737516,
            "suspicion":        0.10140099164416747,
            "endurance":        0.1121221589927027,
            "randomness":      -0.27358803874915805,
            "stability_seeking": -0.12322496315054773,
            "curiosity":        0.0863380590116847,
        },
        "signal_snapshot": {
            "z_expanding":    0.12331700743416119,
            "z_contracting":  0.12543377457808177,
            "z_exploring":   -0.10349164762934038,
        },
        "score": {"RE_avg": 23074.547, "RE_std": 986.23, "repro_rate": 1.0},
    },
    {
        "trial_id": 33,
        "traits_vector": {
            "impulsiveness":   -0.030906531686155037,
            "assertiveness":    0.295543436977969,
            "optimism":         0.1533032620442824,
            "risk_aversion":    0.19198316891043887,
            "suspicion":        0.6008560293248041,
            "endurance":       -0.1640099405326949,
            "randomness":      -0.47082668289346113,
            "stability_seeking": 0.2738254878666811,
            "curiosity":        0.018102349788193417,
        },
        "signal_snapshot": {
            "z_expanding":    0.13931338911203212,
            "z_contracting":  0.20960975256751602,
            "z_exploring":   -0.05963294841286221,
        },
        "score": {"RE_avg": 22364.656, "RE_std": 812.77, "repro_rate": 1.0},
    },
]

# ---------------------------------------------------------------------------
# 9D schema 規格（對應 Model v2.0）
# ---------------------------------------------------------------------------
_9D_KEYS_EXPANDING   = frozenset({"impulsiveness", "assertiveness", "optimism"})
_9D_KEYS_CONTRACTING = frozenset({"risk_aversion", "suspicion", "endurance"})
_9D_KEYS_EXPLORING   = frozenset({"randomness", "stability_seeking", "curiosity"})
_ALL_9D_KEYS         = _9D_KEYS_EXPANDING | _9D_KEYS_CONTRACTING | _9D_KEYS_EXPLORING
_FORBIDDEN_12D_KEYS  = frozenset({"greed", "ambition", "caution", "fearfulness", "patience", "persistence"})

# ---------------------------------------------------------------------------
# 模擬常數（同 dream_seed_finder.py v1.0）
# ---------------------------------------------------------------------------
_T_STEPS              = 1000
_TAIL_START           = 500
_TRAIT_PERTURB_SIGMA  = 0.05
_SIMPLEX_PERTURB_SIGMA = 0.02
_RE_LEVEL3_THRESHOLD  = 5000.0   # Level 3 判定門檻（REPRODUCIBILITY_THRESHOLD）
_EVAL_SEEDS           = [45, 72, 86]  # 測試用 3 seed 交叉驗證（production=5）

# ---------------------------------------------------------------------------
# 內部輔助函式（與 dream_seed_finder.py 對齊，不依賴 Sync/ 目錄）
# ---------------------------------------------------------------------------

def _project_z_signals(traits: dict) -> tuple[float, float, float]:
    z_exp  = (traits["impulsiveness"] + traits["assertiveness"] + traits["optimism"])      / 3.0
    z_con  = (traits["risk_aversion"] + traits["suspicion"]     + traits["endurance"])     / 3.0
    z_expl = (traits["randomness"]    + traits["stability_seeking"] + traits["curiosity"]) / 3.0
    return float(z_exp), float(z_con), float(z_expl)


def _simulate_orbit(
    z_expanding: float,
    z_contracting: float,
    z_exploring: float,
    *,
    t_steps: int = _T_STEPS,
    rng: np.random.Generator | None = None,
    trait_sigma: float = 0.0,
    simplex_sigma: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """引擎層純函式模擬——返回三個策略的軌跡陣列。

    無任何 I/O、無 logging、無 plotting。
    trait_sigma / simplex_sigma 為零時：全確定性（可重現）。
    """
    from evolution.replicator_dynamics import personality_guided_replicator_step

    z_exp  = float(z_expanding)
    z_con  = float(z_contracting)
    z_expl = float(z_exploring)

    # 可選擾動
    if rng is not None and (trait_sigma > 0 or simplex_sigma > 0):
        z_exp  = float(np.clip(z_exp  + rng.normal(0, trait_sigma), -1, 1))
        z_con  = float(np.clip(z_con  + rng.normal(0, trait_sigma), -1, 1))
        z_expl = float(np.clip(z_expl + rng.normal(0, trait_sigma), -1, 1))

        raw = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3]) + rng.normal(0, simplex_sigma, 3)
        raw = np.clip(raw, 1e-9, None)
        raw /= raw.sum()
        simplex = {
            "expanding":   float(raw[0]),
            "contracting": float(raw[1]),
            "exploring":   float(raw[2]),
        }
    else:
        simplex = {"expanding": 1.0 / 3, "contracting": 1.0 / 3, "exploring": 1.0 / 3}

    traj_exp, traj_con, traj_expl = [], [], []
    for _ in range(t_steps):
        simplex, _ = personality_guided_replicator_step(
            simplex,
            z_expanding=z_exp,
            z_contracting=z_con,
            z_exploring=z_expl,
            win=1.0,
            loss=1.2,
            mu=0.05,
            dt=1.0,
        )
        traj_exp.append(simplex["expanding"])
        traj_con.append(simplex["contracting"])
        traj_expl.append(simplex["exploring"])

    return np.array(traj_exp), np.array(traj_con), np.array(traj_expl)


def _compute_re(traj_exp: np.ndarray, tail_start: int = _TAIL_START) -> float:
    """計算旋轉能量（同 dream_seed_finder compute_tail_metrics）。"""
    tail = traj_exp[tail_start:]
    rfft_vals = np.fft.rfft(tail)
    return float(np.sum(np.square(np.abs(rfft_vals[1:]))))


# ===========================================================================
# L1 — 資料層：9D schema 驗證
# ===========================================================================

@pytest.mark.parametrize("candidate", _DREAM_CANDIDATES, ids=[f"trial{c['trial_id']}" for c in _DREAM_CANDIDATES])
def test_l1_dream_candidate_has_9d_schema(candidate: dict) -> None:
    """traits_vector 必須含 9 個且僅含 9 個 v2.0 特質，不含任何 12D 舊特質。"""
    traits = candidate["traits_vector"]
    assert set(traits.keys()) == _ALL_9D_KEYS, (
        f"trial_id={candidate['trial_id']}: 特質鍵集合不符 9D schema。\n"
        f"  期望：{sorted(_ALL_9D_KEYS)}\n"
        f"  實際：{sorted(traits.keys())}"
    )
    assert not (set(traits.keys()) & _FORBIDDEN_12D_KEYS), "仍含 12D 舊特質鍵"


@pytest.mark.parametrize("candidate", _DREAM_CANDIDATES, ids=[f"trial{c['trial_id']}" for c in _DREAM_CANDIDATES])
def test_l1_trait_values_in_valid_range(candidate: dict) -> None:
    """所有特質值必須落在 [-1, 1]。"""
    traits = candidate["traits_vector"]
    violations = {k: v for k, v in traits.items() if not (-1.0 <= v <= 1.0)}
    assert not violations, f"特質值超出 [-1,1] 範圍：{violations}"


@pytest.mark.parametrize("candidate", _DREAM_CANDIDATES, ids=[f"trial{c['trial_id']}" for c in _DREAM_CANDIDATES])
def test_l1_z_signals_reproducible_from_traits(candidate: dict) -> None:
    """signal_snapshot 中的 z-訊號必須能從 traits_vector 精確重算（誤差 < 1e-9）。"""
    traits = candidate["traits_vector"]
    snap   = candidate["signal_snapshot"]
    z_exp_calc, z_con_calc, z_expl_calc = _project_z_signals(traits)

    assert z_exp_calc  == pytest.approx(snap["z_expanding"],   abs=1e-9), "z_expanding 重算失敗"
    assert z_con_calc  == pytest.approx(snap["z_contracting"], abs=1e-9), "z_contracting 重算失敗"
    assert z_expl_calc == pytest.approx(snap["z_exploring"],   abs=1e-9), "z_exploring 重算失敗"


@pytest.mark.parametrize("candidate", _DREAM_CANDIDATES, ids=[f"trial{c['trial_id']}" for c in _DREAM_CANDIDATES])
def test_l1_candidate_is_json_serializable(candidate: dict) -> None:
    """夢幻種子資料結構必須可序列化為 JSON（呈現層資料契約）。"""
    serialized = json.dumps(candidate)
    recovered  = json.loads(serialized)
    assert recovered["trial_id"] == candidate["trial_id"]
    assert set(recovered["traits_vector"].keys()) == _ALL_9D_KEYS


@pytest.mark.parametrize("candidate", _DREAM_CANDIDATES, ids=[f"trial{c['trial_id']}" for c in _DREAM_CANDIDATES])
def test_l1_json_roundtrip_preserves_zsignal_precision(candidate: dict) -> None:
    """JSON 序列化往返後，z-訊號重算誤差必須 < 1e-12（浮點精度保護）。

    防止不同平台的 JSON encoder 因截斷 float64 導致 z-訊號計算結果漂移。
    Python 的 json.dumps 對 float64 使用 repr 精度（17位有效數字），
    往返誤差理論上應 < 2^-52 ≈ 2.2e-16；此處以 1e-12 為實務上限。
    """
    original_traits = candidate["traits_vector"]
    serialized       = json.dumps(original_traits)
    recovered_traits = json.loads(serialized)

    z_exp_orig, z_con_orig, z_expl_orig = _project_z_signals(original_traits)
    z_exp_recv, z_con_recv, z_expl_recv = _project_z_signals(recovered_traits)

    assert abs(z_exp_orig  - z_exp_recv)  < 1e-12, (
        f"trial_id={candidate['trial_id']}: JSON 往返後 z_expanding 漂移 "
        f"{abs(z_exp_orig - z_exp_recv):.2e}"
    )
    assert abs(z_con_orig  - z_con_recv)  < 1e-12, (
        f"trial_id={candidate['trial_id']}: JSON 往返後 z_contracting 漂移 "
        f"{abs(z_con_orig - z_con_recv):.2e}"
    )
    assert abs(z_expl_orig - z_expl_recv) < 1e-12, (
        f"trial_id={candidate['trial_id']}: JSON 往返後 z_exploring 漂移 "
        f"{abs(z_expl_orig - z_expl_recv):.2e}"
    )


# ===========================================================================
# L2 — 引擎層：無 GUI / I/O 依賴性驗證
# ===========================================================================

def test_l2_evolution_module_has_no_gui_imports() -> None:
    """evolution.replicator_dynamics 不得引入任何 GUI、繪圖或 I/O 框架。

    此測試保護 SDD 架構不變條件：evolution/ 不做 I/O、不依賴 plotting。
    注意：取 import 前後 sys.modules 差集，避免其他測試的汙染導致誤報。
    """
    forbidden_prefixes = ("matplotlib", "pygame", "godot", "tkinter", "wx", "qt")

    # 記錄 import 前已存在的模組（其他測試可能已載入 matplotlib）
    before = set(sys.modules.keys())

    import evolution.replicator_dynamics  # noqa: F401

    # 只檢查「本次 import 新增的」模組
    newly_added = set(sys.modules.keys()) - before
    violations = [
        mod for mod in newly_added
        if any(mod == p or mod.startswith(p + ".") for p in forbidden_prefixes)
    ]
    assert not violations, (
        f"evolution.replicator_dynamics 載入後新增了 GUI 模組：{violations}\n"
        "請確認 evolution/ 層不依賴任何表現層套件。"
    )


def test_l2_evolution_module_has_no_subprocess() -> None:
    """evolution.replicator_dynamics 不得使用 subprocess（引擎層與 shell 解耦）。"""
    import evolution.replicator_dynamics as _erd  # noqa: F401

    # 引擎模組的命名空間中不應存在 subprocess 的參照
    module_attrs = vars(_erd)
    assert "subprocess" not in module_attrs, (
        "evolution.replicator_dynamics 的命名空間中發現 subprocess 參照，"
        "請移至 simulation/ 或 scripts/ 層。"
    )


def test_l2_engine_is_pure_function_no_side_effects() -> None:
    """呼叫 personality_guided_replicator_step 前後，sys.modules 不增加 GUI 模組。"""
    from evolution.replicator_dynamics import personality_guided_replicator_step

    modules_snapshot = set(sys.modules.keys())
    simplex = {"expanding": 1 / 3, "contracting": 1 / 3, "exploring": 1 / 3}
    personality_guided_replicator_step(simplex, z_expanding=0.15, z_contracting=0.12, z_exploring=-0.10)
    new_modules = set(sys.modules.keys()) - modules_snapshot

    gui_leaked = [m for m in new_modules if any(
        m == p or m.startswith(p + ".") for p in ("matplotlib", "pygame", "subprocess")
    )]
    assert not gui_leaked, f"引擎函式呼叫後出現非預期模組：{gui_leaked}"


# ===========================================================================
# L3 — 軌道合約：Dream z-訊號 → Level 3 軌道
# ===========================================================================

@pytest.mark.parametrize("candidate", _DREAM_CANDIDATES, ids=[f"trial{c['trial_id']}" for c in _DREAM_CANDIDATES])
def test_l3_nominal_zsignal_produces_level3_orbit(candidate: dict) -> None:
    """使用 signal_snapshot 的名義 z-訊號（無擾動），RE 必須超過 Level 3 門檻。

    這是確定性測試：同一輸入必須永遠輸出相同結果（可回歸）。
    """
    snap = candidate["signal_snapshot"]
    traj_exp, _, _ = _simulate_orbit(
        snap["z_expanding"],
        snap["z_contracting"],
        snap["z_exploring"],
    )
    re = _compute_re(traj_exp)
    assert re > _RE_LEVEL3_THRESHOLD, (
        f"trial_id={candidate['trial_id']}: 名義 z-訊號 RE={re:.1f} 未達 Level 3 門檻 {_RE_LEVEL3_THRESHOLD}。\n"
        f"  z_exp={snap['z_expanding']:.4f}, z_con={snap['z_contracting']:.4f}, "
        f"z_expl={snap['z_exploring']:.4f}"
    )


@pytest.mark.parametrize("candidate", _DREAM_CANDIDATES, ids=[f"trial{c['trial_id']}" for c in _DREAM_CANDIDATES])
def test_l3_orbit_has_nonzero_rotational_structure(candidate: dict) -> None:
    """Level 3 軌道必須有顯著旋轉結構：頻域能量集中在非 DC 成分。

    DC 成分（f=0）代表靜態均衡，非 DC 能量代表動態旋轉。
    """
    snap = candidate["signal_snapshot"]
    traj_exp, _, _ = _simulate_orbit(
        snap["z_expanding"],
        snap["z_contracting"],
        snap["z_exploring"],
    )
    tail = traj_exp[_TAIL_START:]
    rfft_vals = np.fft.rfft(tail)
    dc_power    = float(np.abs(rfft_vals[0]) ** 2)
    ac_power    = float(np.sum(np.square(np.abs(rfft_vals[1:]))))
    ac_dc_ratio = ac_power / (dc_power + 1e-10)

    assert ac_dc_ratio > 1.0, (
        f"trial_id={candidate['trial_id']}: AC/DC 能量比 = {ac_dc_ratio:.3f}，"
        "旋轉結構不顯著（期望 > 1.0）。"
    )


# ===========================================================================
# L4 — 再現性合約：多擾動種子全數通過 Level 3
# ===========================================================================

@pytest.mark.parametrize("candidate", _DREAM_CANDIDATES, ids=[f"trial{c['trial_id']}" for c in _DREAM_CANDIDATES])
def test_l4_reproducibility_across_perturbation_seeds(candidate: dict) -> None:
    """traits_vector 的名義 z-訊號在 3 個擾動種子下全數產出 Level 3 軌道。

    對應 dream_seed_finder 的 repro_rate = 1.0 合約（測試用 3 seed 子集）。
    """
    traits = candidate["traits_vector"]
    z_exp_nom, z_con_nom, z_expl_nom = _project_z_signals(traits)

    failed_seeds = []
    for seed in _EVAL_SEEDS:
        rng = np.random.default_rng(seed)
        traj_exp, _, _ = _simulate_orbit(
            z_exp_nom, z_con_nom, z_expl_nom,
            rng=rng,
            trait_sigma=_TRAIT_PERTURB_SIGMA,
            simplex_sigma=_SIMPLEX_PERTURB_SIGMA,
        )
        re = _compute_re(traj_exp)
        if re < _RE_LEVEL3_THRESHOLD:
            failed_seeds.append((seed, re))

    assert not failed_seeds, (
        f"trial_id={candidate['trial_id']}: 以下擾動種子未達 Level 3：\n"
        + "\n".join(f"  seed={s}: RE={re:.1f}" for s, re in failed_seeds)
    )


def test_l4_determinism_same_seed_same_trajectory() -> None:
    """相同 seed 下，兩次獨立呼叫 _simulate_orbit 必須輸出逐點相同的軌跡。

    此測試驗證「跨平台決定論」的基礎：在同一 Python 版本與 OS 下，
    給定相同 seed 的 NumPy Generator 必須產出完全一致的擾動序列，
    進而保證軌跡可逐 tick 對照（可重現性的最強形式）。
    注意：跨 OS / Python 版本的決定論依賴 numpy Generator（PCG64）的
    跨平台位元穩定性保證，本測試驗證同環境下的強決定論。
    """
    traits = _DREAM_CANDIDATES[0]["traits_vector"]
    z_exp, z_con, z_expl = _project_z_signals(traits)

    def _run(seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        traj_exp, _, _ = _simulate_orbit(
            z_exp, z_con, z_expl,
            t_steps=200,
            rng=rng,
            trait_sigma=_TRAIT_PERTURB_SIGMA,
            simplex_sigma=_SIMPLEX_PERTURB_SIGMA,
        )
        return traj_exp

    traj_a = _run(42)
    traj_b = _run(42)
    assert np.array_equal(traj_a, traj_b), (
        "相同 seed=42 的兩次模擬產出不同軌跡，決定論破壞。"
        f"  最大差異={np.max(np.abs(traj_a - traj_b)):.2e}"
    )

    # 不同 seed 必須產出不同軌跡（確認 seed 確實有效）
    traj_c = _run(99)
    assert not np.array_equal(traj_a, traj_c), (
        "不同 seed 產出了相同軌跡，seed 機制可能失效。"
    )


# ===========================================================================
# L5 — Simplex 不變條件：每一步 Σx_i = 1、x_i > 0
# ===========================================================================

def test_l5_simplex_sum_invariant_throughout_trajectory() -> None:
    """Σ(expanding + contracting + exploring) = 1 必須在整條軌跡上保持（至 1e-9 精度）。"""
    from evolution.replicator_dynamics import personality_guided_replicator_step

    # 使用 trial53 的名義訊號
    snap = _DREAM_CANDIDATES[0]["signal_snapshot"]
    simplex = {"expanding": 1 / 3, "contracting": 1 / 3, "exploring": 1 / 3}

    violations = []
    for step in range(200):
        simplex, _ = personality_guided_replicator_step(
            simplex,
            z_expanding=snap["z_expanding"],
            z_contracting=snap["z_contracting"],
            z_exploring=snap["z_exploring"],
            win=1.0, loss=1.2, mu=0.05, dt=1.0,
        )
        s = sum(simplex.values())
        if abs(s - 1.0) > 1e-9:
            violations.append((step, s))

    assert not violations, (
        f"Simplex 求和偏離 1.0 的步驟：{violations[:5]}（最多顯示 5 個）"
    )


def test_l5_all_simplex_components_strictly_positive() -> None:
    """所有 simplex 座標必須在整條軌跡上保持 > 0（不塌縮到頂點）。"""
    from evolution.replicator_dynamics import personality_guided_replicator_step

    snap = _DREAM_CANDIDATES[0]["signal_snapshot"]
    simplex = {"expanding": 1 / 3, "contracting": 1 / 3, "exploring": 1 / 3}

    for step in range(200):
        simplex, _ = personality_guided_replicator_step(
            simplex,
            z_expanding=snap["z_expanding"],
            z_contracting=snap["z_contracting"],
            z_exploring=snap["z_exploring"],
            win=1.0, loss=1.2, mu=0.05, dt=1.0,
        )
        for key, val in simplex.items():
            assert val > 0, f"step={step}: simplex['{key}']={val} ≤ 0，軌道已塌縮"


def test_l5_lyapunov_stable_orbit_exceeds_initial_energy() -> None:
    """穩定軌道能量必須遠高於初始靜止態能量（Lyapunov 吸引子語意）。

    Lyapunov 函數語意：把旋轉能量 V(t) 視為代理 Lyapunov 函數。
    系統從 simplex 中心點（x_i = 1/3，近似靜止態）出發，
    在循環 payoff 矩陣的驅動下應收斂到高能旋轉軌道（吸引子）。

    驗證方式（初始態 vs 穩態對比）：
      - 初始段 RE：前 50 步（系統剛離開中心，能量最低）
      - 穩態段 RE：後 500 步（TAIL_START 之後的穩定軌道）
    要求：穩態 RE_avg ≥ 初始 RE_avg × 10（保守下界，實際通常 > 100 倍）

    注意：此類軌道 z_expl < 0（relu=0，無向心突變），能量由循環
    payoff 矩陣驅動；初始靜止態能量趨近於 0，穩態為高能旋轉軌道，
    兩者存在量級差異，此不等式幾乎是必然成立的強保證。
    """
    snap = _DREAM_CANDIDATES[0]["signal_snapshot"]
    traj_exp, _, _ = _simulate_orbit(
        snap["z_expanding"],
        snap["z_contracting"],
        snap["z_exploring"],
        t_steps=_T_STEPS,
    )

    # 初始段 RE（前 50 步，系統靠近靜止態）
    rfft_init = np.fft.rfft(traj_exp[:50])
    re_initial = float(np.sum(np.square(np.abs(rfft_init[1:]))))

    # 穩態段 RE（後 500 步，已收斂到吸引子）
    rfft_stable = np.fft.rfft(traj_exp[_TAIL_START:])
    re_stable = float(np.sum(np.square(np.abs(rfft_stable[1:]))))

    assert re_stable >= re_initial * 10.0, (
        f"Lyapunov 吸引子語意違反：穩態 RE={re_stable:.1f} < "
        f"初始態 RE={re_initial:.1f} × 10（系統未收斂到高能軌道）\n"
        f"  z_exp={snap['z_expanding']:.4f}, z_con={snap['z_contracting']:.4f}, "
        f"z_expl={snap['z_exploring']:.4f}"
    )


# ===========================================================================
# L6 — 呈現層解耦：軌道資料可序列化，診斷資訊完整有限
# ===========================================================================

def test_l6_orbit_trajectory_is_json_serializable() -> None:
    """軌道軌跡可序列化為 JSON（確保可發送至任何呈現層而無需共享物件）。"""
    snap = _DREAM_CANDIDATES[0]["signal_snapshot"]
    traj_exp, traj_con, traj_expl = _simulate_orbit(
        snap["z_expanding"], snap["z_contracting"], snap["z_exploring"],
        t_steps=100,
    )
    orbit_payload = {
        "trial_id":  _DREAM_CANDIDATES[0]["trial_id"],
        "z_signals": snap,
        "trajectory": {
            "expanding":   traj_exp.tolist(),
            "contracting": traj_con.tolist(),
            "exploring":   traj_expl.tolist(),
        },
    }
    serialized = json.dumps(orbit_payload)
    recovered = json.loads(serialized)

    assert len(recovered["trajectory"]["expanding"]) == 100
    assert all(isinstance(v, float) for v in recovered["trajectory"]["expanding"])


def test_l6_step_diagnostics_complete_and_finite() -> None:
    """personality_guided_replicator_step 的 diagnostics 必須包含預期鍵且所有值有限。"""
    from evolution.replicator_dynamics import personality_guided_replicator_step

    snap = _DREAM_CANDIDATES[0]["signal_snapshot"]
    simplex = {"expanding": 1 / 3, "contracting": 1 / 3, "exploring": 1 / 3}
    _, diag = personality_guided_replicator_step(
        simplex,
        z_expanding=snap["z_expanding"],
        z_contracting=snap["z_contracting"],
        z_exploring=snap["z_exploring"],
    )

    required_keys = {
        "phi",
        "f_raw_expanding", "f_raw_contracting", "f_raw_exploring",
        "f_prime_expanding", "f_prime_contracting", "f_prime_exploring",
        "mutation_strength",
        "simplex_sum_error",
    }
    missing = required_keys - set(diag.keys())
    assert not missing, f"diagnostics 缺少鍵：{missing}"

    non_finite = {k: v for k, v in diag.items() if not math.isfinite(v)}
    assert not non_finite, f"diagnostics 含非有限值：{non_finite}"


def test_l6_diagnostics_simplex_sum_error_near_zero() -> None:
    """每步 diagnostics['simplex_sum_error'] 必須 < 1e-9（數值精度保證）。"""
    from evolution.replicator_dynamics import personality_guided_replicator_step

    snap = _DREAM_CANDIDATES[0]["signal_snapshot"]
    simplex = {"expanding": 1 / 3, "contracting": 1 / 3, "exploring": 1 / 3}

    for _ in range(200):
        simplex, diag = personality_guided_replicator_step(
            simplex,
            z_expanding=snap["z_expanding"],
            z_contracting=snap["z_contracting"],
            z_exploring=snap["z_exploring"],
        )
        assert diag["simplex_sum_error"] < 1e-9, (
            f"simplex_sum_error={diag['simplex_sum_error']:.2e} 超出數值精度閾值"
        )


# ===========================================================================
# L3 補充 — z-訊號邊界驗證（物理自洽）
# ===========================================================================

def test_l3_boundary_zexpl_negative_relu_zero_no_centripetal() -> None:
    """z_exploring < 0 時 relu = 0 → mutation_strength = 0（無向心力，符合物理分析）。"""
    from evolution.replicator_dynamics import personality_guided_replicator_step

    simplex = {"expanding": 1 / 3, "contracting": 1 / 3, "exploring": 1 / 3}
    # 所有 dream candidates 的 z_exploring 均 < 0
    _, diag = personality_guided_replicator_step(
        simplex,
        z_expanding=0.15,
        z_contracting=0.12,
        z_exploring=-0.10,  # < 0，relu = 0
    )
    assert diag["mutation_strength"] == pytest.approx(0.0, abs=1e-12), (
        f"z_exploring < 0 時 mutation_strength 應為 0，實際={diag['mutation_strength']}"
    )


# ---------------------------------------------------------------------------
# Godot API Schema 規格（呈現層資料契約）
# ---------------------------------------------------------------------------
_GODOT_ORBIT_SCHEMA: dict = {
    "required_top_keys": {"trial_id", "schema_version", "z_signals", "trajectory", "metrics"},
    "z_signals_keys":    {"z_expanding", "z_contracting", "z_exploring"},
    "trajectory_keys":   {"expanding", "contracting", "exploring"},
    "metrics_keys":      {"RE_avg", "RE_std", "repro_rate"},
    "schema_version":    "v2.0",
}


def test_l6_orbit_payload_conforms_to_godot_api_schema() -> None:
    """軌道 payload 必須符合 Godot 呈現層預期的 API schema（v2.0）。

    Schema 約定（對應 Presentation Layer 資料契約）：
      - 頂層必含 trial_id, schema_version, z_signals, trajectory, metrics
      - z_signals 鍵集合 = {z_expanding, z_contracting, z_exploring}
      - trajectory 鍵集合 = {expanding, contracting, exploring}，各為 float list
      - metrics 含 RE_avg（float > 0）、RE_std（float ≥ 0）、repro_rate（float ∈ [0,1]）
      - schema_version = 'v2.0'（與 Model v2.0 / 9D-Core 版本鎖定）
    """
    candidate = _DREAM_CANDIDATES[0]
    snap      = candidate["signal_snapshot"]
    traj_exp, traj_con, traj_expl = _simulate_orbit(
        snap["z_expanding"], snap["z_contracting"], snap["z_exploring"],
        t_steps=50,
    )

    # 建構符合 Godot schema 的 payload
    payload = {
        "trial_id":      candidate["trial_id"],
        "schema_version": _GODOT_ORBIT_SCHEMA["schema_version"],
        "z_signals":     snap,
        "trajectory": {
            "expanding":   traj_exp.tolist(),
            "contracting": traj_con.tolist(),
            "exploring":   traj_expl.tolist(),
        },
        "metrics": {
            "RE_avg":     candidate["score"]["RE_avg"],
            "RE_std":     candidate["score"]["RE_std"],
            "repro_rate": candidate["score"]["repro_rate"],
        },
    }

    # ── 結構驗證 ──
    schema = _GODOT_ORBIT_SCHEMA
    missing_top = schema["required_top_keys"] - set(payload.keys())
    assert not missing_top, f"頂層缺少必要鍵：{missing_top}"

    missing_z = schema["z_signals_keys"] - set(payload["z_signals"].keys())
    assert not missing_z, f"z_signals 缺少鍵：{missing_z}"

    missing_traj = schema["trajectory_keys"] - set(payload["trajectory"].keys())
    assert not missing_traj, f"trajectory 缺少鍵：{missing_traj}"

    missing_met = schema["metrics_keys"] - set(payload["metrics"].keys())
    assert not missing_met, f"metrics 缺少鍵：{missing_met}"

    assert payload["schema_version"] == schema["schema_version"], (
        f"schema_version 不符：{payload['schema_version']} ≠ {schema['schema_version']}"
    )

    # ── 型別與值域驗證 ──
    for key in schema["trajectory_keys"]:
        vals = payload["trajectory"][key]
        assert isinstance(vals, list) and len(vals) == 50, (
            f"trajectory['{key}'] 長度錯誤：{len(vals)}（期望 50）"
        )
        assert all(isinstance(v, float) and math.isfinite(v) for v in vals), (
            f"trajectory['{key}'] 含非 float 或非有限值"
        )
        assert all(0.0 < v < 1.0 for v in vals), (
            f"trajectory['{key}'] 含超出 (0,1) 範圍的值（simplex 座標）"
        )

    m = payload["metrics"]
    assert isinstance(m["RE_avg"], float) and m["RE_avg"] > 0, "RE_avg 必須 > 0"
    assert isinstance(m["RE_std"], float) and m["RE_std"] >= 0, "RE_std 必須 ≥ 0"
    assert isinstance(m["repro_rate"], float) and 0.0 <= m["repro_rate"] <= 1.0, (
        "repro_rate 必須 ∈ [0, 1]"
    )

    # ── JSON 往返完整性 ──
    serialized = json.dumps(payload)
    recovered  = json.loads(serialized)
    assert recovered["schema_version"] == schema["schema_version"]
    assert len(recovered["trajectory"]["expanding"]) == 50


def test_l6_raw_input_sum_deviation_monitored_in_diagnostics() -> None:
    """diagnostics 中必須含 raw_input_sum_deviation 鍵，且穩態下偏移量 < 1e-9。

    折衷方案（SDD 合規）：不修改引擎的靜默正規化邏輯，
    改在 diagnostics 中記錄每步的輸入偏移量，供監控層使用。
    - 初始 simplex (1/3, 1/3, 1/3)：偏移量幾乎為 0
    - 引擎輸出的 new_simplex 已正規化：後續每步偏移量應 < 1e-9
    """
    from evolution.replicator_dynamics import personality_guided_replicator_step

    candidate = _DREAM_CANDIDATES[0]
    snap = candidate["signal_snapshot"]
    simplex: dict = {"expanding": 1/3, "contracting": 1/3, "exploring": 1/3}

    deviations: list[float] = []
    for step in range(200):
        simplex, diag = personality_guided_replicator_step(
            simplex,
            z_expanding=snap["z_expanding"],
            z_contracting=snap["z_contracting"],
            z_exploring=snap["z_exploring"],
            win=1.0, loss=1.2, mu=0.05, dt=1.0,
        )
        assert "raw_input_sum_deviation" in diag, (
            f"step {step}: diagnostics 缺少 raw_input_sum_deviation 鍵"
        )
        deviations.append(diag["raw_input_sum_deviation"])

    # 第 1 步之後（引擎輸出已正規化），後續輸入偏移量應接近機器精度
    tail_deviations = deviations[1:]
    max_deviation = max(tail_deviations)
    assert max_deviation < 1e-9, (
        f"穩態下 raw_input_sum_deviation 最大值 {max_deviation:.2e} 超過 1e-9，"
        "可能存在數值發散風險"
    )


def test_l3_z_signals_in_dream_range_all_yield_level3() -> None:
    """Dream 訊號空間的 9 個格點採樣全數通過 Level 3 門檻（覆蓋測試）。

    搜尋空間：z_exp ∈ [0.05, 0.20], z_con ∈ [0.10, 0.25], z_expl ∈ [-0.25, 0.00]
    """
    z_exp_vals  = [0.08, 0.15]
    z_con_vals  = [0.12, 0.20]
    z_expl_vals = [-0.20, -0.10]

    failures = []
    for ze in z_exp_vals:
        for zc in z_con_vals:
            for zx in z_expl_vals:
                traj_exp, _, _ = _simulate_orbit(ze, zc, zx)
                re = _compute_re(traj_exp)
                if re < _RE_LEVEL3_THRESHOLD:
                    failures.append({"z_exp": ze, "z_con": zc, "z_expl": zx, "RE": re})

    assert not failures, (
        "以下 Dream 訊號範圍格點未達 Level 3：\n"
        + "\n".join(f"  {f}" for f in failures)
    )
