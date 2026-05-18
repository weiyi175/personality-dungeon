#!/usr/bin/env python3
"""H7.7 Corner Escape Work：角點逃逸功與駐留時間發散量測

目的：從「觀察者」轉為「測量者」——精確量化各角點的駐留時間 τ_i(A)，
定位 τ_DEF(A) 的發散點，揭示異宿環（Heteroclinic Cycle）如何在噪聲
積累下被扯斷並焊接為單一吸收態（Corner Lock）。

────────────────────────────────────────────────────────────────────────────
理論背景
────────────────────────────────────────────────────────────────────────────
  H7.6 確認 L3 週期的微觀本質是「角點間慢速巡迴（Corner Cycle）」：
    DEF(p_def≈1) → AGG(p_agg≈1) → BAL(p_bal≈1) → DEF → ...
  系統在每個角點鄰域（p_i > 0.95）中駐留數百到千餘回合，再由噪聲
  誘導的逃逸不穩定流形（Unstable Manifold）驅動轉換。

  在標準複製子動力學中，角點 p_i=1 是邊界不動點：
    ṗ_i = p_i(1 - p_i)[fitness_i - f̄] → 0 as p_i → 1
  當噪聲將 R_{i→j}（角點 i 的逃逸率）打至接近零時，τ_i(A) 發散，
  角點 i 成為吸收態（Absorbing State）。

  核心假設：τ_DEF(A) 隨 A → A_c(seed99≈0.25) 發散（power-law 或 指數型）。

────────────────────────────────────────────────────────────────────────────
實驗設計
────────────────────────────────────────────────────────────────────────────
  Seeds        : {47, 97, 99}
                   47  ← L0 窗口位於 amp≈0.10（H7.6 發現）；細粒度搜索窗口邊界
                   97  ← deep-stable 最穩健種；A_c=0.20（H7.6）
                   99  ← 角點鎖死種；τ_DEF 發散點估計在 amp≈0.23–0.25

  Amplitudes   : ctrl + 細粒度區段
                   [0.07–0.15 step 0.01]  → 9 levels，覆蓋 seed47 L0 窗口
                   [0.20–0.26 step 0.01]  → 7 levels，覆蓋 seed99 鎖死邊界
                   [0.15, 0.30, 0.40]     → H7.6 錨點，確保跨實驗連貫性
                   共 21 個 amplitude（含 ctrl=0.0）

  規模         : 3 seeds × 21 amplitudes = 63 runs
  n_rounds     : 6000（固定）

────────────────────────────────────────────────────────────────────────────
核心量測：駐留時間（Dwell Time）
────────────────────────────────────────────────────────────────────────────
  角點鄰域閾值：p_i > CORNER_THRESHOLD（預設 0.95）
  
  per corner i ∈ {AGG, DEF, BAL}：
    n_visits_i    : 進入該角點鄰域的次數
    mean_dwell_i  : 平均每次駐留時間（rounds）= PRIMARY τ_i(A) 指標
    max_dwell_i   : 最長單次駐留（rounds）
    total_i       : 該角點在全程中的總佔用回合數
    last_entry_i  : 最後一次進入角點的時間
    last_exit_i   : 最後一次離開角點的時間（None = 仍在角點中，即鎖死）

  corner_seq : 壓縮後的角點切換序列，例如 "DEF(1800)→AGG(550)→BAL(950)→DEF(∞)"
  locked_corner : "AGG" | "DEF" | "BAL" | "NONE" — 晚期窗口角點佔用 > LOCK_RATIO
  lock_onset_t  : 最後一次角點轉換的時間（= 鎖死開始時間）

────────────────────────────────────────────────────────────────────────────
驗收條件（H7.7 3-gate）
────────────────────────────────────────────────────────────────────────────
  H7.7-G1  TAU_DIVERGENCE    seed99 τ_DEF 在 A≥0.25 時 > 3× baseline（ctrl 下的 τ_DEF）
  H7.7-G2  WINDOW_BOUNDARY   seed47 定位到 L0 窗口的上下邊界（L3→L0→L3 的精確邊界）
  H7.7-G3  ASYMMETRY         至少一個 seed 各角點 mean_dwell 具有統計顯著性不對稱
                              （max / min > 3）

────────────────────────────────────────────────────────────────────────────
輸出目錄結構
────────────────────────────────────────────────────────────────────────────
  outputs/h77_escape_work/
    ctrl/            seed47.csv  seed97.csv  seed99.csv
    amp_0.07/        seed47.csv  ...
    ...
    summary.json     2D matrix: seed → amp_label → EscapeMetric
                     + dwell_table per seed (τ_i vs amplitude)
                     + corner_sequences per seed per amplitude

Usage:
  PYTHONPATH=/home/user/personality-dungeon ./venv/bin/python -u \\
    scripts/run_w77_escape_work.py
  PYTHONPATH=/home/user/personality-dungeon ./venv/bin/python -u \\
    scripts/run_w77_escape_work.py --resume
  PYTHONPATH=/home/user/personality-dungeon ./venv/bin/python -u \\
    scripts/run_w77_escape_work.py --seeds 99 --amplitudes 0.20 0.21 0.22 0.23 0.24 0.25
  PYTHONPATH=/home/user/personality-dungeon ./venv/bin/python -u \\
    scripts/run_w77_escape_work.py --threshold 0.95 --lock-ratio 0.95
"""

from __future__ import annotations

import argparse
import csv as _csv
import json
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.cycle_metrics import classify_cycle_level
from players.base_player import DEFAULT_PERSONALITY_KEYS
from simulation.run_simulation import SimConfig, _write_timeseries_csv, simulate

# ──────────────────────────────────────────────────────────────────────────────
# Fixed golden-point values (H7.3 locked — DO NOT MODIFY)
# ──────────────────────────────────────────────────────────────────────────────
MU_BASE            = 0.30
LAMBDA_MU          = 0.05
LAMBDA_K           = 0.20
SELECTION_STRENGTH = 0.15
K_LOWER            = 0.05
K_UPPER            = 0.25
GAMMA_BASE         = 0.16
SYNERGY_POWER      = 3.2
FULL_ROUNDS        = 6000

# Corner analysis parameters
CORNER_THRESHOLD = 0.95   # p_i > this → system is "in" corner i
LOCK_RATIO       = 0.95   # fraction of late window in one corner → "locked"

# Analysis windows (H7.5/H7.6 compatible)
LATE_WIN_START  = 4000
LATE_WIN_LEN    = 2000   # [4000, 6000]
POST_WIN_START  = 2500
POST_WIN_LEN    = 1000   # [2500, 3500]
STD_BURN_IN     = 2000
STD_TAIL        = 2000   # [2000, 4000]

# Default experiment parameters
# Fine-grained: seed47 L0 window [0.07–0.15] + seed99 lock boundary [0.20–0.26]
# + H7.6 anchors [0.30, 0.40]
DEFAULT_SEEDS = [47, 97, 99]
DEFAULT_AMPLITUDES = [
    0.0,                                                    # ctrl
    0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, # seed47 L0 window
    0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26,              # seed99 lock boundary
    0.30, 0.40,                                             # H7.6 anchors
]


# ──────────────────────────────────────────────────────────────────────────────
# Persona setup (identical RNG to H7.5/H7.6 for reproducibility)
# ──────────────────────────────────────────────────────────────────────────────
def _noise_persona_setup(seed: int, amplitude: float) -> Callable:
    def cb(players: list, _strategy_space: list, _cfg: SimConfig) -> None:
        for idx, player in enumerate(players):
            rng = random.Random(int(seed) * 10000 + idx)
            for key in DEFAULT_PERSONALITY_KEYS:
                if hasattr(player, "personality") and key in player.personality:
                    player.personality[key] = rng.uniform(-amplitude, amplitude)
    return cb


# ──────────────────────────────────────────────────────────────────────────────
# Data types
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class CornerStats:
    """Dwell time statistics for one corner (AGG / DEF / BAL)."""
    n_visits: int          # number of distinct visits to this corner
    mean_dwell: float      # mean rounds per visit  ← PRIMARY τ_i(A) indicator
    max_dwell: int         # longest single dwell (rounds)
    total_rounds: int      # total rounds spent in this corner across full run
    last_entry_t: int      # round of last entry
    last_exit_t: int | None  # round of last exit (None = still in corner at end = lock)


@dataclass
class EscapeMetric:
    """Complete per-(seed, amplitude) result for H7.7."""
    seed: int
    amplitude: float
    amp_label: str
    csv_path: str

    # Standard level classification (H7.6-compatible)
    std_level: int          # [2000, 4000]
    post_level: int         # [2500, 3500]
    late_level: int         # [4000, 6000]  ← primary
    l3_late: bool
    l3_post: bool

    # CSD velocity std dev
    vs_std_post: float      # [2500, 3500] CSD indicator

    # Corner dwell time stats (full run, all rounds)
    stats_agg: CornerStats
    stats_def: CornerStats
    stats_bal: CornerStats

    # Corner analysis (late window only [4000, 6000])
    late_tau_agg: float     # mean dwell in AGG corner, late window
    late_tau_def: float     # mean dwell in DEF corner, late window  ← divergence indicator
    late_tau_bal: float     # mean dwell in BAL corner, late window

    # Corner lock detection
    locked_corner: str      # "AGG" | "DEF" | "BAL" | "NONE"
    lock_onset_t: int | None  # round of last successful corner transition (= lock onset)
    n_full_cycles: int      # number of complete DEF→AGG→BAL→DEF cycles

    # Compressed corner sequence string
    corner_seq: str         # e.g. "DEF(1800)→AGG(550)→BAL(950)→DEF(2550∞)"


# ──────────────────────────────────────────────────────────────────────────────
# Corner analysis functions
# ──────────────────────────────────────────────────────────────────────────────
def _label_corners(
    rows: list[dict[str, Any]],
    threshold: float = CORNER_THRESHOLD,
) -> list[str | None]:
    """Label each round with the dominant corner label, or None if no corner dominates."""
    labels: list[str | None] = []
    for r in rows:
        pa = float(r.get("p_aggressive", 0.0))
        pd = float(r.get("p_defensive",  0.0))
        pb = float(r.get("p_balanced",   0.0))
        if   pa >= threshold: labels.append("AGG")
        elif pd >= threshold: labels.append("DEF")
        elif pb >= threshold: labels.append("BAL")
        else:                 labels.append(None)
    return labels


def _compute_corner_stats(
    labels: list[str | None],
    corner: str,
    t_offset: int = 0,
) -> CornerStats:
    """Compute dwell time statistics for one corner from a label sequence."""
    visits: list[int] = []
    last_entry: int = 0
    last_exit: int | None = None
    in_corner = False
    entry_t = 0

    for t, lbl in enumerate(labels):
        abs_t = t + t_offset
        if not in_corner and lbl == corner:
            in_corner = True
            entry_t = abs_t
            last_entry = abs_t
        elif in_corner and lbl != corner:
            in_corner = False
            dwell = abs_t - entry_t
            visits.append(dwell)
            last_exit = abs_t

    # Still in corner at end of window → potentially locked
    if in_corner:
        dwell = len(labels) + t_offset - entry_t
        visits.append(dwell)
        last_exit = None   # still in corner

    if not visits:
        return CornerStats(
            n_visits=0, mean_dwell=0.0, max_dwell=0,
            total_rounds=0, last_entry_t=0, last_exit_t=0,
        )
    return CornerStats(
        n_visits=len(visits),
        mean_dwell=sum(visits) / len(visits),
        max_dwell=max(visits),
        total_rounds=sum(visits),
        last_entry_t=last_entry,
        last_exit_t=last_exit,
    )


def _corner_sequence_str(labels: list[str | None]) -> str:
    """Build a compact corner sequence string with dwell times.

    Example: "DEF(1823)→AGG(547)→BAL(953)→DEF(2677∞)"
    '∞' suffix means the system was still in that corner at end of run.
    '~' prefix on a segment means the run entered mid-sequence (not from start).
    """
    if not labels:
        return "(empty)"

    segments: list[tuple[str, int, bool]] = []   # (corner, dwell, is_open)
    current: str | None = None
    start = 0

    for t, lbl in enumerate(labels):
        if lbl != current:
            if current is not None:
                segments.append((current, t - start, False))
            current = lbl
            start = t

    # Final segment
    if current is not None:
        is_open = True  # still in this corner at end
        segments.append((current, len(labels) - start, is_open))
    else:
        # Ended in "no corner" zone
        pass

    parts: list[str] = []
    for corner, dwell, is_open in segments:
        if corner is None:
            parts.append(f"~({dwell})")
        else:
            suffix = "∞" if is_open else ""
            parts.append(f"{corner}({dwell}{suffix})")

    return "→".join(parts) if parts else "(no corners)"


def _count_full_cycles(labels: list[str | None]) -> int:
    """Count complete DEF→AGG→BAL→DEF cycles in the label sequence."""
    expected = ["DEF", "AGG", "BAL"]
    cycle_count = 0
    phase = 0   # 0=looking for DEF, 1=looking for AGG, 2=looking for BAL
    prev = None
    for lbl in labels:
        if lbl is None or lbl == prev:
            continue
        if phase == 0 and lbl == "DEF":
            phase = 1
        elif phase == 1 and lbl == "AGG":
            phase = 2
        elif phase == 2 and lbl == "BAL":
            phase = 0
            # Wait for next DEF to complete
        elif phase == 0 and lbl != "DEF":
            pass
        if phase == 0 and prev == "BAL" and lbl == "DEF":
            cycle_count += 1
        prev = lbl
    return cycle_count


def _detect_lock(
    labels: list[str | None],
    lock_ratio: float = LOCK_RATIO,
) -> tuple[str, int | None]:
    """Detect whether the system is locked in a corner during the late window.

    Returns (locked_corner, lock_onset_t) where:
    - locked_corner: "AGG"|"DEF"|"BAL"|"NONE"
    - lock_onset_t: absolute round of the last corner transition before locking
                    (i.e., the round where the system entered the final corner)
                    None if no lock detected.
    """
    n = len(labels)
    if n == 0:
        return ("NONE", None)

    for corner in ("AGG", "DEF", "BAL"):
        count = sum(1 for lbl in labels if lbl == corner)
        if count / n >= lock_ratio:
            # Find when the system last ENTERED this corner
            onset_t: int | None = None
            prev = None
            for t, lbl in enumerate(labels):
                if lbl == corner and prev != corner:
                    onset_t = t  # update each time we re-enter
                prev = lbl
            return (corner, onset_t)

    return ("NONE", None)


def _mean_dwell_in_window(
    labels: list[str | None],
    corner: str,
    t_offset: int = 0,
) -> float:
    """Mean dwell time for a corner within a given window label sequence."""
    stats = _compute_corner_stats(labels, corner, t_offset)
    return stats.mean_dwell


def _vs_std(rows: list[dict[str, Any]], start: int, length: int) -> float:
    end = min(start + length, len(rows))
    sub = rows[start:end]
    if len(sub) < 10:
        return float("nan")
    pa = [float(r.get("p_aggressive", 0.0)) for r in sub]
    pd = [float(r.get("p_defensive",  0.0)) for r in sub]
    pb = [float(r.get("p_balanced",   0.0)) for r in sub]
    vs: list[float] = []
    for i in range(len(sub) - 1):
        da, dd, db = pa[i+1]-pa[i], pd[i+1]-pd[i], pb[i+1]-pb[i]
        vs.append(math.sqrt(da*da + dd*dd + db*db))
    if not vs:
        return float("nan")
    mean = sum(vs) / len(vs)
    return math.sqrt(sum((v - mean)**2 for v in vs) / len(vs))


def _classify_window(rows: list[dict[str, Any]], start: int, length: int) -> int:
    end = min(start + length, len(rows))
    sub = rows[start:end]
    if len(sub) < 50:
        return 0
    props = {
        "aggressive": [float(r.get("p_aggressive", 0.0)) for r in sub],
        "defensive":  [float(r.get("p_defensive",  0.0)) for r in sub],
        "balanced":   [float(r.get("p_balanced",   0.0)) for r in sub],
    }
    result = classify_cycle_level(
        props, burn_in=0, tail=len(sub),
        amplitude_threshold=0.02, eta=0.6, min_turn_strength=0.0,
    )
    return int(result.level)


# ──────────────────────────────────────────────────────────────────────────────
# Core run
# ──────────────────────────────────────────────────────────────────────────────
def run_one(
    *,
    seed: int,
    amplitude: float,
    out_dir: Path,
    resume: bool = False,
    corner_threshold: float = CORNER_THRESHOLD,
    lock_ratio: float = LOCK_RATIO,
) -> EscapeMetric:
    is_ctrl   = amplitude == 0.0
    amp_label = "ctrl" if is_ctrl else f"{amplitude:.2f}"
    subdir    = out_dir / ("ctrl" if is_ctrl else f"amp_{amplitude:.2f}")
    subdir.mkdir(parents=True, exist_ok=True)
    out_csv   = subdir / f"seed{seed}.csv"

    cfg = SimConfig(
        n_players=300,
        n_rounds=FULL_ROUNDS,
        seed=int(seed),
        payoff_mode="matrix_ab",
        popularity_mode="sampled",
        gamma=GAMMA_BASE,
        epsilon=0.0,
        a=1.0,
        b=0.9,
        matrix_cross_coupling=0.20,
        init_bias=0.5,
        evolution_mode="personality_coupled",
        payoff_lag=1,
        selection_strength=float(SELECTION_STRENGTH),
        enable_events=False,
        events_json=None,
        out_csv=out_csv,
        memory_kernel=1,
        synergy_type="nonlinear",
        synergy_gamma=float(GAMMA_BASE),
        synergy_nonlinear_type="power",
        synergy_nonlinear_power=float(SYNERGY_POWER),
        personality_coupling_mu_base=float(MU_BASE),
        personality_coupling_lambda_mu=float(LAMBDA_MU),
        personality_coupling_lambda_k=float(LAMBDA_K),
        personality_coupling_mu_lower=0.0,
        personality_coupling_mu_upper=0.60,
        personality_coupling_k_lower=float(K_LOWER),
        personality_coupling_k_upper=float(K_UPPER),
        synergy_pulse_t_start=None,
        synergy_pulse_duration=None,
        synergy_pulse_delta_gamma=0.0,
    )

    if resume and out_csv.exists():
        with out_csv.open(newline="") as f:
            rows = list(_csv.DictReader(f))
    else:
        persona_cb = _noise_persona_setup(seed, amplitude) if not is_ctrl else None
        _strategy_space, rows = simulate(cfg, player_setup_callback=persona_cb)
        _write_timeseries_csv(out_csv, strategy_space=_strategy_space, rows=rows)

    # Level classifications
    std_level  = _classify_window(rows, STD_BURN_IN,   STD_TAIL)
    post_level = _classify_window(rows, POST_WIN_START, POST_WIN_LEN)
    late_level = _classify_window(rows, LATE_WIN_START, LATE_WIN_LEN)
    vs_std_post = _vs_std(rows, POST_WIN_START, POST_WIN_LEN)

    # Corner analysis — full run
    all_labels  = _label_corners(rows, corner_threshold)
    stats_agg   = _compute_corner_stats(all_labels, "AGG")
    stats_def   = _compute_corner_stats(all_labels, "DEF")
    stats_bal   = _compute_corner_stats(all_labels, "BAL")

    # Corner analysis — late window only
    late_labels = all_labels[LATE_WIN_START: LATE_WIN_START + LATE_WIN_LEN]
    late_tau_agg = _mean_dwell_in_window(late_labels, "AGG", LATE_WIN_START)
    late_tau_def = _mean_dwell_in_window(late_labels, "DEF", LATE_WIN_START)
    late_tau_bal = _mean_dwell_in_window(late_labels, "BAL", LATE_WIN_START)

    # Lock detection (late window)
    locked_corner, lock_rel_t = _detect_lock(late_labels, lock_ratio)
    lock_onset_t = (LATE_WIN_START + lock_rel_t) if lock_rel_t is not None else None

    # Full cycle count and sequence
    n_full_cycles = _count_full_cycles(all_labels)
    corner_seq = _corner_sequence_str(all_labels)

    rel_csv = str(out_csv.relative_to(ROOT)) if out_csv.is_relative_to(ROOT) else str(out_csv)
    return EscapeMetric(
        seed=int(seed),
        amplitude=float(amplitude),
        amp_label=amp_label,
        csv_path=rel_csv,
        std_level=std_level,
        post_level=post_level,
        late_level=late_level,
        l3_late=late_level >= 3,
        l3_post=post_level >= 3,
        vs_std_post=vs_std_post if not math.isnan(vs_std_post) else 0.0,
        stats_agg=stats_agg,
        stats_def=stats_def,
        stats_bal=stats_bal,
        late_tau_agg=late_tau_agg,
        late_tau_def=late_tau_def,
        late_tau_bal=late_tau_bal,
        locked_corner=locked_corner,
        lock_onset_t=lock_onset_t,
        n_full_cycles=n_full_cycles,
        corner_seq=corner_seq,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Summary building
# ──────────────────────────────────────────────────────────────────────────────
def _stats_to_dict(s: CornerStats) -> dict:
    return {
        "n_visits":    s.n_visits,
        "mean_dwell":  round(s.mean_dwell, 2),
        "max_dwell":   s.max_dwell,
        "total_rounds":s.total_rounds,
        "last_entry_t":s.last_entry_t,
        "last_exit_t": s.last_exit_t,
    }


def _build_summary(
    all_results: list[EscapeMetric],
    seeds: list[int],
    amplitudes: list[float],
) -> dict:
    by_seed: dict[str, list[EscapeMetric]] = {}
    for m in all_results:
        by_seed.setdefault(str(m.seed), []).append(m)

    matrix: dict[str, dict] = {}
    for seed in seeds:
        matrix[str(seed)] = {}
        for m in sorted(by_seed.get(str(seed), []), key=lambda x: x.amplitude):
            matrix[str(seed)][m.amp_label] = {
                "std_level":    m.std_level,
                "post_level":   m.post_level,
                "late_level":   m.late_level,
                "l3_late":      m.l3_late,
                "vs_std_post":  round(m.vs_std_post, 6),
                "dwell_agg":    _stats_to_dict(m.stats_agg),
                "dwell_def":    _stats_to_dict(m.stats_def),
                "dwell_bal":    _stats_to_dict(m.stats_bal),
                "late_tau_agg": round(m.late_tau_agg, 1),
                "late_tau_def": round(m.late_tau_def, 1),
                "late_tau_bal": round(m.late_tau_bal, 1),
                "locked_corner":m.locked_corner,
                "lock_onset_t": m.lock_onset_t,
                "n_full_cycles":m.n_full_cycles,
                "corner_seq":   m.corner_seq,
                "csv_path":     m.csv_path,
            }

    # Per-seed τ dwell tables (sorted by amplitude, for τ_i(A) curve)
    dwell_tables: dict[str, list[dict]] = {}
    for seed in seeds:
        rows_out = []
        for m in sorted(by_seed.get(str(seed), []), key=lambda x: x.amplitude):
            rows_out.append({
                "amplitude":    m.amplitude,
                "amp_label":    m.amp_label,
                "late_level":   m.late_level,
                "tau_agg":      round(m.stats_agg.mean_dwell, 1),
                "tau_def":      round(m.stats_def.mean_dwell, 1),
                "tau_bal":      round(m.stats_bal.mean_dwell, 1),
                "late_tau_def": round(m.late_tau_def, 1),
                "locked_corner":m.locked_corner,
                "n_full_cycles":m.n_full_cycles,
            })
        dwell_tables[str(seed)] = rows_out

    # Gate evaluation
    def _tau_def_baseline(seed: int) -> float:
        mets = [m for m in by_seed.get(str(seed), []) if m.amplitude == 0.0]
        return mets[0].stats_def.mean_dwell if mets else 0.0

    # G1: seed99 τ_DEF at A≥0.25 > 3× baseline
    tau99_baseline = _tau_def_baseline(99)
    tau99_high = max(
        (m.stats_def.mean_dwell for m in by_seed.get("99", []) if m.amplitude >= 0.25),
        default=0.0,
    )
    g1 = tau99_baseline > 0 and tau99_high > 3 * tau99_baseline

    # G2: seed47 has both L0 and L3 in fine-grained range
    seed47_late_levels = {m.amplitude: m.late_level for m in by_seed.get("47", [])}
    fine47 = [a for a in seed47_late_levels if 0.07 <= a <= 0.15]
    g2 = (any(seed47_late_levels.get(a, -1) == 0 for a in fine47) and
          any(seed47_late_levels.get(a, -1) >= 3 for a in fine47))

    # G3: any seed × any single amplitude shows max_corner/min_corner
    #     dwell ratio > 3.  Use per-amplitude max (not amplitude-average)
    #     to avoid algebraic wash-out from mixing cycling and locked amps.
    def _dwell_asymmetry(seed: int) -> tuple[float, float, str]:
        """Return (best_ratio, best_amp, best_label)."""
        mets = by_seed.get(str(seed), [])
        best_ratio, best_amp = 0.0, 0.0
        for m in mets:
            vals = [v for v in [m.stats_agg.mean_dwell,
                                m.stats_def.mean_dwell,
                                m.stats_bal.mean_dwell] if v > 0]
            if len(vals) >= 2:
                r = max(vals) / min(vals)
                if r > best_ratio:
                    best_ratio, best_amp = r, m.amplitude
        lbl = "ctrl" if best_amp == 0.0 else f"{best_amp:.2f}"
        return best_ratio, best_amp, lbl

    g3_details = {str(s): _dwell_asymmetry(s) for s in seeds}
    g3 = any(v[0] > 3.0 for v in g3_details.values())

    verdict = ("PASS"    if g1 and g2 and g3 else
               "PARTIAL" if g1 or  g2 or  g3 else
               "FAIL")

    return {
        "meta": {
            "experiment": "H7.7",
            "seeds": seeds,
            "amplitudes": amplitudes,
            "n_runs": len(all_results),
            "corner_threshold": CORNER_THRESHOLD,
            "lock_ratio": LOCK_RATIO,
            "golden_point": {
                "mu_base": MU_BASE, "lambda_mu": LAMBDA_MU,
                "SS": SELECTION_STRENGTH, "gamma_base": GAMMA_BASE,
                "synergy_power": SYNERGY_POWER,
            },
        },
        "matrix": matrix,
        "dwell_tables": dwell_tables,
        "gates": {
            "g1_tau_divergence": g1,
            "g2_window_boundary": g2,
            "g3_asymmetry": g3,
            "g1_detail": {
                "tau99_ctrl_baseline": round(tau99_baseline, 1),
                "tau99_max_at_ge025": round(tau99_high, 1),
                "ratio": round(tau99_high / tau99_baseline, 2) if tau99_baseline > 0 else None,
            },
            "g3_detail": {
                str(s): {
                    "best_ratio": round(g3_details[str(s)][0], 2),
                    "best_amp":   g3_details[str(s)][2],
                } for s in seeds
            },
        },
        "verdict": verdict,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Output printing
# ──────────────────────────────────────────────────────────────────────────────
def _ascii_tau_curve(
    dwell_rows: list[dict],
    corner: str = "DEF",
    width: int = 60,
) -> list[str]:
    """ASCII sparkline of τ_corner(A) vs amplitude."""
    key = f"tau_{corner.lower()}"
    vals = [r[key] for r in dwell_rows if r["amplitude"] > 0.0]
    if not vals or max(vals) == 0:
        return ["(no data)"]
    vmax = max(vals)
    lines = []
    for r in dwell_rows:
        if r["amplitude"] == 0.0:
            continue
        v = r[key]
        bar_len = int(v / vmax * width)
        bar = "█" * bar_len
        lock_mark = "🔒" if r["locked_corner"] == corner else "  "
        lvl = r["late_level"]
        lines.append(
            f"  A={r['amp_label']}  L{lvl}  {lock_mark}  [{bar:<{width}}]  τ={v:.0f}"
        )
    return lines


def _print_results(all_results: list[EscapeMetric], summary: dict) -> None:
    W = 120
    print()
    print("=" * W)
    print("H7.7 Corner Escape Work — Results")
    print(f"  Corner threshold: p_i > {CORNER_THRESHOLD}   "
          f"Lock ratio: {LOCK_RATIO}   n_runs: {len(all_results)}")
    print("=" * W)

    seeds_present = sorted({m.seed for m in all_results})
    amps_present  = sorted({m.amplitude for m in all_results})

    # ── Table 1: Late Level + Corner Lock ──────────────────────────────────────
    print()
    hdr_amps = "  ".join(f"{('ctrl' if a == 0.0 else f'{a:.2f}'):^6}" for a in amps_present)
    print(f"  Late Level + Lock  {'seed':>5}  {hdr_amps}")
    print("  " + "-" * (W - 2))
    by_sa = {(m.seed, m.amplitude): m for m in all_results}
    for seed in seeds_present:
        cells = []
        for amp in amps_present:
            m = by_sa.get((seed, amp))
            if m is None:
                cells.append("  ?  ")
            elif m.locked_corner != "NONE":
                cells.append(f"L{m.late_level}🔒")
            elif m.late_level >= 3:
                cells.append("L3✓ ")
            elif m.late_level == 2:
                cells.append("L2  ")
            elif m.late_level == 1:
                cells.append("L1  ")
            else:
                cells.append("L0  ")
        row = "  ".join(f"{c:^6}" for c in cells)
        print(f"  {' ':>5}  {seed:>5}  {row}")

    # ── Table 2: τ_DEF (mean dwell in DEF corner, full run) ──────────────────
    print()
    print(f"  τ_DEF (mean dwell in DEF corner, all rounds)  {'seed':>5}  {hdr_amps}")
    print("  " + "-" * (W - 2))
    for seed in seeds_present:
        cells = []
        for amp in amps_present:
            m = by_sa.get((seed, amp))
            cells.append(f"{m.stats_def.mean_dwell:>5.0f}" if m else "  ?  ")
        row = "  ".join(f"{c:^6}" for c in cells)
        print(f"  {' ':>5}  {seed:>5}  {row}")

    # ── Table 3: n_full_cycles ────────────────────────────────────────────────
    print()
    print(f"  Full cycles (DEF→AGG→BAL→DEF count)  {'seed':>5}  {hdr_amps}")
    print("  " + "-" * (W - 2))
    for seed in seeds_present:
        cells = []
        for amp in amps_present:
            m = by_sa.get((seed, amp))
            cells.append(f"{m.n_full_cycles:>3}" if m else " ?")
        row = "  ".join(f"{c:^6}" for c in cells)
        print(f"  {' ':>5}  {seed:>5}  {row}")

    print()
    print("=" * W)

    # ── τ_DEF ASCII curves ────────────────────────────────────────────────────
    dwell_tables = summary.get("dwell_tables", {})
    for seed in seeds_present:
        dt = dwell_tables.get(str(seed), [])
        if not dt:
            continue
        print()
        print(f"  τ_DEF(A) curve — seed {seed}")
        for line in _ascii_tau_curve(dt, "DEF"):
            print(line)

    print()
    print("=" * W)

    # ── Corner sequences ──────────────────────────────────────────────────────
    print()
    print("  Corner sequences (compressed):")
    for seed in seeds_present:
        print(f"  Seed {seed}:")
        for amp in amps_present:
            m = by_sa.get((seed, amp))
            if m is None:
                continue
            lock_s = f"  LOCKED→{m.locked_corner}@t={m.lock_onset_t}" if m.locked_corner != "NONE" else ""
            # Truncate long sequences
            seq = m.corner_seq if len(m.corner_seq) <= 80 else m.corner_seq[:77] + "..."
            print(f"    amp={m.amp_label:<5}  L{m.late_level}  cyc={m.n_full_cycles}{lock_s}")
            print(f"          {seq}")

    print()
    print("=" * W)

    # ── Gate summary ──────────────────────────────────────────────────────────
    gates = summary["gates"]
    g1d = gates.get("g1_detail", {})
    print()
    g1s = "✓" if gates["g1_tau_divergence"]  else "✗"
    g2s = "✓" if gates["g2_window_boundary"] else "✗"
    g3s = "✓" if gates["g3_asymmetry"]        else "✗"
    print(f"  H7.7-G1 TAU_DIVERGENCE    {g1s}  "
          f"τ_DEF(ctrl)={g1d.get('tau99_ctrl_baseline','?'):.0f}  "
          f"τ_DEF(A≥0.25)={g1d.get('tau99_max_at_ge025','?'):.0f}  "
          f"ratio={g1d.get('ratio','?')}")
    print(f"  H7.7-G2 WINDOW_BOUNDARY   {g2s}  "
          f"seed47 L0 window found in [0.07,0.15]")
    print(f"  H7.7-G3 ASYMMETRY         {g3s}  "
          f"corner dwell asymmetry > 3")
    print()
    print(f"  VERDICT: {summary['verdict']}")
    print("=" * W)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="H7.7 Corner Escape Work — Dwell Time Divergence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--resume", action="store_true",
                        help="Reuse existing CSVs instead of re-simulating.")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--amplitudes", type=float, nargs="+",
                        default=DEFAULT_AMPLITUDES,
                        help="Noise amplitudes (0.0 = ctrl, always included).")
    parser.add_argument("--threshold", type=float, default=CORNER_THRESHOLD,
                        metavar="THRESH",
                        help=f"Corner neighbourhood threshold (default: {CORNER_THRESHOLD})")
    parser.add_argument("--lock-ratio", type=float, default=LOCK_RATIO,
                        metavar="RATIO",
                        help=f"Fraction of late window for lock detection (default: {LOCK_RATIO})")
    args = parser.parse_args()

    seeds = sorted(set(args.seeds))
    amplitudes = sorted(set(args.amplitudes))
    if 0.0 not in amplitudes:
        amplitudes = [0.0] + amplitudes

    out_dir = ROOT / "outputs" / "h77_escape_work"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_json = out_dir / "summary.json"

    n_runs = len(seeds) * len(amplitudes)
    print("H7.7 Corner Escape Work")
    print(f"  Seeds       : {seeds}")
    print(f"  Amplitudes  : {amplitudes}")
    print(f"  Total runs  : {n_runs}")
    print(f"  Threshold   : p_i > {args.threshold}  (corner neighbourhood)")
    print(f"  Lock ratio  : {args.lock_ratio}  (late-window lock detection)")
    print(f"  Output dir  : {out_dir}")
    if args.resume:
        print(f"  Resume      : ON")
    print()

    all_results: list[EscapeMetric] = []
    for run_idx, seed in enumerate(seeds):
        for amp_idx, amp in enumerate(amplitudes):
            global_idx = run_idx * len(amplitudes) + amp_idx + 1
            amp_label = "ctrl" if amp == 0.0 else f"{amp:.2f}"

            m = run_one(
                seed=seed, amplitude=amp, out_dir=out_dir,
                resume=args.resume,
                corner_threshold=args.threshold,
                lock_ratio=args.lock_ratio,
            )
            all_results.append(m)

            lock_s = f"  LOCKED={m.locked_corner}@t={m.lock_onset_t}" if m.locked_corner != "NONE" else ""
            print(
                f"[{global_idx:>3}/{n_runs}]  seed={seed:>4}  amp={amp_label:<5}  "
                f"late=L{m.late_level}{'✓' if m.l3_late else ' '}  "
                f"τ_DEF={m.stats_def.mean_dwell:>6.0f}  "
                f"τ_AGG={m.stats_agg.mean_dwell:>6.0f}  "
                f"τ_BAL={m.stats_bal.mean_dwell:>6.0f}  "
                f"cyc={m.n_full_cycles}{lock_s}"
            )

    summary = _build_summary(all_results, seeds, amplitudes)
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    _print_results(all_results, summary)
    print(f"\nSummary saved → {summary_json}")


if __name__ == "__main__":
    main()
