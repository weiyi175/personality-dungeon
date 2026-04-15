"""T-series targeted follow-up: stronger k_local + longer + noise variants.

Follows blueprint §7.4 weak_positive protocol: one targeted follow-up (≤ 9 runs)
after original T-series returned weak_positive on minibatch conditions.

Predefined conditions (priority order)
---------------------------------------
1. t_lattice4_minibatch_strong  [PRIORITY]  lattice4,  k_local=0.12
2. t_sw10_minibatch_strong      [PRIORITY]  sw10,      k_local=0.12
3. t_lattice4_minibatch_longer  [optional]  lattice4,  k_local=0.08,  rounds=5000
4. t_sw10_minibatch_noise       [optional]  sw10,      k_local=0.08,  mutation=0.008

Default run: conditions 1+2 (6 runs). Use --conditions all for all 4 (12 runs).

Hard-stop decision rule
-----------------------
  PASS    → level3_seed_count ≥ 1 in either strong condition AND max_uplift > 0.02
  CLOSE   → all active conditions return 0/3 Level 3 AND max_uplift < 0.015
  WEAK    → otherwise (some uplift but no Level 3)

Usage
-----
./venv/bin/python -m simulation.t_series_followup \\
  --conditions strong \\
  --seeds 45,47,49 \\
  --out-root outputs/t_series_followup \\
  --summary-tsv outputs/t_series_followup_summary.tsv \\
  --combined-tsv outputs/t_series_followup_combined.tsv \\
  --decision-md outputs/t_series_followup_decision.md
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evolution.local_graph import GraphSpec
from simulation.t_series import (
    FAMILY,
    GATE_NAME,
    TCellConfig,
    _fmt,
    _graph_spec_for,
    _phase_amplitude_stability,
    _run_control_wellmixed,
    _safe_mean,
    _seed_metrics,
    _tail_diagnostics,
    _write_simplex_plot,
    _write_phase_amplitude_plot,
    _write_timeseries_csv,
    _write_tsv,
    _yn,
    run_t_cell,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Follow-up condition specs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FollowupSpec:
    tag: str          # "strong" | "longer" | "noise"
    condition: str
    topology: str
    k_local: float
    rounds: int
    burn_in: int
    tail: int
    mutation_rate: float = 0.0
    p_rewire: float = 0.10


FOLLOWUP_SPECS: list[FollowupSpec] = [
    FollowupSpec(tag="strong", condition="t_lattice4_minibatch_strong",
                 topology="lattice4", k_local=0.12, rounds=3000,
                 burn_in=1000, tail=1000),
    FollowupSpec(tag="strong", condition="t_sw10_minibatch_strong",
                 topology="small_world", k_local=0.12, rounds=3000,
                 burn_in=1000, tail=1000),
    FollowupSpec(tag="longer", condition="t_lattice4_minibatch_longer",
                 topology="lattice4", k_local=0.08, rounds=5000,
                 burn_in=2000, tail=2000),
    FollowupSpec(tag="noise", condition="t_sw10_minibatch_noise",
                 topology="small_world", k_local=0.08, rounds=3000,
                 burn_in=1000, tail=1000, mutation_rate=0.008),
]

FOLLOWUP_SUMMARY_FIELDNAMES = [
    "gate", "family", "condition", "followup_tag",
    "graph_topology", "p_rewire", "k_local", "mutation_rate",
    "rounds", "burn_in", "tail",
    "init_mode", "seed",
    "cycle_level", "stage3_score", "turn_strength",
    "env_gamma", "env_gamma_r2",
    "control_env_gamma", "control_stage3_score",
    "stage3_uplift_vs_control_seed", "has_level3_seed",
    "init_weight_dispersion",
    "graph_clustering_coefficient", "graph_mean_path_length",
    "spatial_autocorrelation_d1", "spatial_autocorrelation_d2",
    "mean_batch_phase_spread",
    "mean_edge_strategy_distance",
    "edge_disagreement_rate",
    "mean_local_growth_cosine_vs_global",
    "mean_player_growth_dispersion",
    "phase_amplitude_stability",
    "out_csv",
]

FOLLOWUP_COMBINED_FIELDNAMES = [
    "gate", "family", "condition", "followup_tag",
    "graph_topology", "p_rewire", "k_local", "mutation_rate",
    "rounds", "burn_in", "tail",
    "init_mode", "is_control", "n_seeds",
    "mean_cycle_level", "mean_stage3_score", "mean_turn_strength",
    "mean_env_gamma",
    "level_counts_json", "p_level_3", "level3_seed_count",
    "control_mean_env_gamma", "control_mean_stage3_score",
    "stage3_uplift_vs_control",
    "max_uplift_any_condition",
    "init_weight_dispersion",
    "graph_clustering_coefficient", "graph_mean_path_length",
    "spatial_autocorrelation_d1", "spatial_autocorrelation_d2",
    "spatial_autocorrelation_decay",
    "mean_batch_phase_spread",
    "mean_edge_strategy_distance",
    "edge_disagreement_rate",
    "mean_local_growth_cosine_vs_global",
    "mean_player_growth_dispersion",
    "phase_amplitude_stability",
    "verdict",
    "players", "rounds_actual", "out_dir",
]


def _select_specs(conditions_arg: str) -> list[FollowupSpec]:
    if conditions_arg == "strong":
        return [s for s in FOLLOWUP_SPECS if s.tag == "strong"]
    if conditions_arg == "all":
        return list(FOLLOWUP_SPECS)
    # comma-separated tags or condition names
    requested = {x.strip() for x in conditions_arg.split(",")}
    result = [s for s in FOLLOWUP_SPECS
              if s.tag in requested or s.condition in requested]
    if not result:
        raise ValueError(f"No matching specs for: {conditions_arg!r}")
    return result


# ---------------------------------------------------------------------------
# Follow-up verdict
# ---------------------------------------------------------------------------

def _compute_followup_verdict(
    *,
    level3_seed_count: int,
    stage3_uplift: float,
    mean_env_gamma: float,
    phase_amplitude_stability: float,
) -> str:
    if (level3_seed_count >= 1
            and stage3_uplift > 0.02
            and abs(mean_env_gamma) <= 5e-4):
        return "pass"
    if level3_seed_count >= 1 or stage3_uplift > 0.02:
        return "weak_positive"
    return "fail"


def _compute_overall_verdict(
    active_rows: list[dict[str, Any]],
) -> str:
    any_pass = any(str(r.get("verdict", "")) == "pass" for r in active_rows)
    any_weak = any(str(r.get("verdict", "")) == "weak_positive" for r in active_rows)
    all_l3_zero = all(int(r.get("level3_seed_count", 0)) == 0 for r in active_rows)
    max_uplift = max(
        (float(r.get("stage3_uplift_vs_control", 0)) for r in active_rows),
        default=0.0,
    )
    if any_pass:
        return "pass_t"
    if any_weak:
        return "weak_positive_t"
    if all_l3_zero and max_uplift < 0.015:
        return "close_t"
    return "fail_t"


# ---------------------------------------------------------------------------
# Decision markdown
# ---------------------------------------------------------------------------

def _write_followup_decision(
    path: Path,
    *,
    active_rows: list[dict[str, Any]],
    ctrl_ri_metrics: dict,
    ctrl_ui_metrics: dict,
    seeds: list[int],
    original_scout: str = "",
) -> None:
    overall = _compute_overall_verdict(active_rows)
    max_uplift = max(
        (float(r.get("stage3_uplift_vs_control", 0)) for r in active_rows),
        default=0.0,
    )
    all_l3_zero = all(int(r.get("level3_seed_count", 0)) == 0 for r in active_rows)

    lines = [
        "# T-series Targeted Follow-up — Decision",
        "",
        f"baseline_from: {original_scout or 'outputs/t_series_short_scout_decision.md'}",
        "",
        "## Follow-up Conditions",
        "",
    ]

    for row in active_rows:
        lines.append(
            f"condition: {row['condition']}"
            f"  tag={row['followup_tag']}"
            f"  k_local={row['k_local']}"
            f"  rounds={row.get('rounds_actual', row.get('rounds', ''))}"
            f"  mut={row.get('mutation_rate', '0.0')}"
            f"  seeds={row['n_seeds']}"
            f"  lv3={row['level3_seed_count']}"
            f"  uplift={row['stage3_uplift_vs_control']}"
            f"  gamma={row['mean_env_gamma']}"
            f"  pa_stab={row['phase_amplitude_stability']}"
            f"  cosine={row.get('mean_local_growth_cosine_vs_global', '')}"
            f"  pgd={row.get('mean_player_growth_dispersion', '')}"
            f"  bps={row.get('mean_batch_phase_spread', '')}"
            f"  sac_d1={row.get('spatial_autocorrelation_d1', '')}"
            f"  verdict={row['verdict']}"
        )

    lines.extend([
        "",
        "## Hard-Stop Evaluation (§7.4)",
        "",
        f"max_uplift_across_active={_fmt(max_uplift)}",
        f"all_level3_zero={_yn(all_l3_zero)}",
        "",
    ])

    if overall == "pass_t":
        lines.extend([
            "## Overall Verdict: pass_t",
            "",
            "- ≥1/3 seeds 達到 Level 3 且 uplift > 0.02 → 可升格 longer_confirm（seeds=0:29）",
        ])
    elif overall == "weak_positive_t":
        lines.extend([
            "## Overall Verdict: weak_positive_t",
            "",
            "- 部分 uplift 但無 Level 3 → 訊號仍弱，酌情決定是否再做一輪更大 follow-up",
            "- 建議：觀察 k_local 梯度；若 uplift 隨 k_local 單調升，可試 k_local=0.20",
        ])
    elif overall == "close_t":
        lines.extend([
            "## Overall Verdict: close_t",
            "",
            "- hard_stop 觸發：所有 follow-up active cells 0/3 Level 3 且 max_uplift < 0.015",
            "- T-series 正式 closure：",
            "  - B/C-series 負結果 + T-series 2×2 scout 弱正 + follow-up 再次負",
            "  - 結論：local clustering topology + local update + random Dirichlet init",
            "    均無法讓 phase domain 存活至全域可見 Level 3；",
            "    sampled discrete update 本身是 Level 2 plateau 的結構性瓶頸，",
            "    與玩家初始化策略和圖結構無關。",
            "  - T-series closure 寫入研發日誌。",
        ])
    else:
        lines.extend([
            f"## Overall Verdict: {overall}",
            "",
            "- 未觸發 hard-stop，但無法升格。",
        ])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_t_followup_scout(
    *,
    specs: list[FollowupSpec],
    seeds: list[int],
    players: int,
    a: float,
    b: float,
    cross: float,
    pairwise_imitation_strength: float,
    pairwise_beta: float,
    init_bias: float,
    memory_kernel: int,
    selection_strength: float,
    out_root: Path,
    summary_tsv: Path,
    combined_tsv: Path,
    decision_md: Path,
    original_scout_md: str = "",
) -> dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)

    # --- Controls (random_init + uniform_init) ---
    ctrl_ri_dir = out_root / "control_random_init"
    ctrl_ri_dir.mkdir(parents=True, exist_ok=True)
    ctrl_ri_per_seed: dict[int, dict] = {}
    for seed in seeds:
        ctrl_ri_per_seed[seed] = _run_control_wellmixed(
            seed=seed, players=players, rounds=3000, burn_in=1000, tail=1000,
            a=a, b=b, cross=cross, init_bias=init_bias, memory_kernel=memory_kernel,
            selection_strength=selection_strength, random_init=True,
            out_dir=ctrl_ri_dir,
        )

    ctrl_ui_dir = out_root / "control_uniform_init"
    ctrl_ui_dir.mkdir(parents=True, exist_ok=True)
    ctrl_ui_per_seed: dict[int, dict] = {}
    for seed in seeds:
        ctrl_ui_per_seed[seed] = _run_control_wellmixed(
            seed=seed, players=players, rounds=3000, burn_in=1000, tail=1000,
            a=a, b=b, cross=cross, init_bias=init_bias, memory_kernel=memory_kernel,
            selection_strength=selection_strength, random_init=False,
            out_dir=ctrl_ui_dir,
        )

    # Use random-init control as comparison baseline
    control_per_seed = ctrl_ri_per_seed

    all_summary_rows: list[dict[str, Any]] = []
    all_combined_rows: list[dict[str, Any]] = []
    active_combined: list[dict[str, Any]] = []

    for spec in specs:
        cond_out_dir = out_root / spec.condition
        cond_out_dir.mkdir(parents=True, exist_ok=True)

        gs = _graph_spec_for(spec.topology, p_rewire=spec.p_rewire)
        # lattice4 needs 300 players = 15×20
        n_players = players if spec.topology != "lattice4" else players

        per_seed_metrics: list[dict[str, Any]] = []
        per_seed_extra: list[dict[str, float]] = []

        for seed in seeds:
            config = TCellConfig(
                condition=spec.condition,
                graph_spec=gs,
                update_mode="minibatch",
                pairwise_imitation_strength=pairwise_imitation_strength,
                pairwise_beta=pairwise_beta,
                local_selection_strength=spec.k_local,
                players=n_players, rounds=spec.rounds, seed=seed,
                a=a, b=b, cross=cross,
                burn_in=spec.burn_in, tail=spec.tail,
                random_init=True, init_bias=init_bias,
                memory_kernel=memory_kernel,
                mutation_rate=spec.mutation_rate,
                out_dir=cond_out_dir,
            )
            result = run_t_cell(config)
            global_rows = result["global_rows"]
            round_diag = result["round_diag"]

            csv_path = cond_out_dir / f"seed{seed}.csv"
            _write_timeseries_csv(csv_path, global_rows)

            sm = _seed_metrics(global_rows, burn_in=spec.burn_in, tail=spec.tail)
            diag = _tail_diagnostics(round_diag, n_rows=len(global_rows),
                                     burn_in=spec.burn_in, tail=spec.tail)
            pa = _phase_amplitude_stability(global_rows, burn_in=spec.burn_in, tail=spec.tail)

            ctrl_m = control_per_seed[seed]["metrics"]
            uplift = float(sm["stage3_score"]) - float(ctrl_m["stage3_score"])

            per_seed_metrics.append(sm)
            per_seed_extra.append({
                **diag,
                "phase_amplitude_stability": pa,
                "init_weight_dispersion": result["init_weight_dispersion"],
                "graph_clustering_coefficient": result["graph_clustering_coefficient"],
                "graph_mean_path_length": result["graph_mean_path_length"],
                "spatial_autocorrelation_d1": result["spatial_autocorrelation_d1"],
                "spatial_autocorrelation_d2": result["spatial_autocorrelation_d2"],
                "uplift": uplift,
            })

            all_summary_rows.append({
                "gate": GATE_NAME, "family": FAMILY,
                "condition": spec.condition, "followup_tag": spec.tag,
                "graph_topology": spec.topology, "p_rewire": _fmt(spec.p_rewire),
                "k_local": _fmt(spec.k_local), "mutation_rate": _fmt(spec.mutation_rate),
                "rounds": spec.rounds, "burn_in": spec.burn_in, "tail": spec.tail,
                "init_mode": "random_init",
                "seed": seed,
                "cycle_level": int(sm["cycle_level"]),
                "stage3_score": _fmt(sm["stage3_score"]),
                "turn_strength": _fmt(sm["turn_strength"]),
                "env_gamma": _fmt(sm["env_gamma"]),
                "env_gamma_r2": _fmt(sm["env_gamma_r2"]),
                "control_env_gamma": _fmt(ctrl_m["env_gamma"]),
                "control_stage3_score": _fmt(ctrl_m["stage3_score"]),
                "stage3_uplift_vs_control_seed": _fmt(uplift),
                "has_level3_seed": _yn(int(sm["cycle_level"]) >= 3),
                "init_weight_dispersion": _fmt(result["init_weight_dispersion"]),
                "graph_clustering_coefficient": _fmt(result["graph_clustering_coefficient"]),
                "graph_mean_path_length": _fmt(result["graph_mean_path_length"]),
                "spatial_autocorrelation_d1": _fmt(result["spatial_autocorrelation_d1"]),
                "spatial_autocorrelation_d2": _fmt(result["spatial_autocorrelation_d2"]),
                "mean_batch_phase_spread": _fmt(diag["mean_batch_phase_spread"]),
                "mean_edge_strategy_distance": _fmt(diag["mean_edge_strategy_distance"]),
                "edge_disagreement_rate": _fmt(diag["edge_disagreement_rate"]),
                "mean_local_growth_cosine_vs_global": _fmt(diag["mean_local_growth_cosine_vs_global"]) if diag["mean_local_growth_cosine_vs_global"] else "",
                "mean_player_growth_dispersion": _fmt(diag["mean_player_growth_dispersion"]) if diag["mean_player_growth_dispersion"] else "",
                "phase_amplitude_stability": _fmt(pa),
                "out_csv": str(csv_path),
            })

        # Aggregate
        levels = [int(m["cycle_level"]) for m in per_seed_metrics]
        l3_count = sum(1 for l in levels if l >= 3)
        n_cond = len(per_seed_metrics)
        mean_s3 = _safe_mean([float(m["stage3_score"]) for m in per_seed_metrics])
        mean_ctrl_s3 = _safe_mean([float(control_per_seed[s]["metrics"]["stage3_score"]) for s in seeds])
        uplift_agg = mean_s3 - mean_ctrl_s3
        mean_gamma = _safe_mean([float(m["env_gamma"]) for m in per_seed_metrics])
        mean_ctrl_gamma = _safe_mean([float(control_per_seed[s]["metrics"]["env_gamma"]) for s in seeds])

        def _agg(key: str) -> float:
            vals = [float(e[key]) for e in per_seed_extra if e.get(key, "") != ""]
            return _safe_mean(vals) if vals else 0.0

        pa_agg = _agg("phase_amplitude_stability")
        sac1_agg = _agg("spatial_autocorrelation_d1")
        sac2_agg = _agg("spatial_autocorrelation_d2")
        sac_decay = sac1_agg / sac2_agg if abs(sac2_agg) > 1e-12 else 0.0

        verdict = _compute_followup_verdict(
            level3_seed_count=l3_count,
            stage3_uplift=uplift_agg,
            mean_env_gamma=mean_gamma,
            phase_amplitude_stability=pa_agg,
        )

        level_counts = {lv: levels.count(lv) for lv in range(4)}
        combined_row = {
            "gate": GATE_NAME, "family": FAMILY,
            "condition": spec.condition, "followup_tag": spec.tag,
            "graph_topology": spec.topology, "p_rewire": _fmt(spec.p_rewire),
            "k_local": _fmt(spec.k_local), "mutation_rate": _fmt(spec.mutation_rate),
            "rounds": spec.rounds, "burn_in": spec.burn_in, "tail": spec.tail,
            "init_mode": "random_init",
            "is_control": _yn(False), "n_seeds": n_cond,
            "mean_cycle_level": _fmt(_safe_mean([float(m["cycle_level"]) for m in per_seed_metrics])),
            "mean_stage3_score": _fmt(mean_s3),
            "mean_turn_strength": _fmt(_safe_mean([float(m["turn_strength"]) for m in per_seed_metrics])),
            "mean_env_gamma": _fmt(mean_gamma),
            "level_counts_json": json.dumps(level_counts, sort_keys=True),
            "p_level_3": _fmt(float(l3_count) / float(n_cond) if n_cond > 0 else 0.0),
            "level3_seed_count": l3_count,
            "control_mean_env_gamma": _fmt(mean_ctrl_gamma),
            "control_mean_stage3_score": _fmt(mean_ctrl_s3),
            "stage3_uplift_vs_control": _fmt(uplift_agg),
            "max_uplift_any_condition": "",   # filled in after aggregating all
            "init_weight_dispersion": _fmt(_agg("init_weight_dispersion")),
            "graph_clustering_coefficient": _fmt(_agg("graph_clustering_coefficient")),
            "graph_mean_path_length": _fmt(_agg("graph_mean_path_length")),
            "spatial_autocorrelation_d1": _fmt(sac1_agg),
            "spatial_autocorrelation_d2": _fmt(sac2_agg),
            "spatial_autocorrelation_decay": _fmt(sac_decay),
            "mean_batch_phase_spread": _fmt(_agg("mean_batch_phase_spread")),
            "mean_edge_strategy_distance": _fmt(_agg("mean_edge_strategy_distance")),
            "edge_disagreement_rate": _fmt(_agg("edge_disagreement_rate")),
            "mean_local_growth_cosine_vs_global": _fmt(_agg("mean_local_growth_cosine_vs_global")) if _agg("mean_local_growth_cosine_vs_global") else "",
            "mean_player_growth_dispersion": _fmt(_agg("mean_player_growth_dispersion")) if _agg("mean_player_growth_dispersion") else "",
            "phase_amplitude_stability": _fmt(pa_agg),
            "verdict": verdict,
            "players": n_players, "rounds_actual": spec.rounds,
            "out_dir": str(cond_out_dir),
        }
        active_combined.append(combined_row)

        # Plots for representative seed
        rep_seed = max(seeds, key=lambda s: (
            int(per_seed_metrics[seeds.index(s)]["cycle_level"]),
            float(per_seed_metrics[seeds.index(s)]["stage3_score"]),
        ))
        rep_rows: list[dict[str, Any]] = []
        with (cond_out_dir / f"seed{rep_seed}.csv").open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                rep_rows.append(dict(row))
        _write_simplex_plot(rep_rows, out_png=cond_out_dir / f"simplex_seed{rep_seed}.png",
                            title=f"{spec.condition} seed={rep_seed}",
                            burn_in=spec.burn_in, tail=spec.tail)
        _write_phase_amplitude_plot(rep_rows, out_png=cond_out_dir / f"phase_amplitude_seed{rep_seed}.png",
                                    title=f"{spec.condition} seed={rep_seed}",
                                    burn_in=spec.burn_in, tail=spec.tail)

    # Backfill max_uplift_any_condition into every row
    all_max_uplift = max(
        (float(r["stage3_uplift_vs_control"]) for r in active_combined),
        default=0.0,
    )
    for r in active_combined:
        r["max_uplift_any_condition"] = _fmt(all_max_uplift)

    all_combined_rows = active_combined

    _write_tsv(summary_tsv, fieldnames=FOLLOWUP_SUMMARY_FIELDNAMES, rows=all_summary_rows)
    _write_tsv(combined_tsv, fieldnames=FOLLOWUP_COMBINED_FIELDNAMES, rows=all_combined_rows)
    _write_followup_decision(decision_md, active_rows=active_combined,
                             ctrl_ri_metrics={s: ctrl_ri_per_seed[s]["metrics"] for s in seeds},
                             ctrl_ui_metrics={s: ctrl_ui_per_seed[s]["metrics"] for s in seeds},
                             seeds=seeds, original_scout=original_scout_md)

    return {
        "summary_tsv": str(summary_tsv),
        "combined_tsv": str(combined_tsv),
        "decision_md": str(decision_md),
        "overall_verdict": _compute_overall_verdict(active_combined),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="T-series targeted follow-up scout")
    parser.add_argument("--conditions", type=str, default="strong",
                        help="Which conditions to run: 'strong' (default, 6 runs), "
                             "'all' (12 runs), or comma-separated tags/condition names")
    parser.add_argument("--seeds", type=str, default="45,47,49")
    parser.add_argument("--players", type=int, default=300)
    parser.add_argument("--a", type=float, default=1.0)
    parser.add_argument("--b", type=float, default=0.9)
    parser.add_argument("--matrix-cross-coupling", type=float, default=0.20)
    parser.add_argument("--pairwise-imitation-strength", type=float, default=0.35)
    parser.add_argument("--pairwise-beta", type=float, default=8.0)
    parser.add_argument("--init-bias", type=float, default=0.12)
    parser.add_argument("--memory-kernel", type=int, default=3)
    parser.add_argument("--selection-strength", type=float, default=0.06)
    parser.add_argument("--original-scout-md", type=str,
                        default="outputs/t_series_short_scout_decision.md")
    parser.add_argument("--out-root", type=Path,
                        default=REPO_ROOT / "outputs" / "t_series_followup")
    parser.add_argument("--summary-tsv", type=Path,
                        default=REPO_ROOT / "outputs" / "t_series_followup_summary.tsv")
    parser.add_argument("--combined-tsv", type=Path,
                        default=REPO_ROOT / "outputs" / "t_series_followup_combined.tsv")
    parser.add_argument("--decision-md", type=Path,
                        default=REPO_ROOT / "outputs" / "t_series_followup_decision.md")

    args = parser.parse_args()
    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    specs = _select_specs(args.conditions)

    print(f"Running follow-up: {[s.condition for s in specs]}")
    print(f"Seeds: {seeds}")

    result = run_t_followup_scout(
        specs=specs,
        seeds=seeds,
        players=int(args.players),
        a=float(args.a),
        b=float(args.b),
        cross=float(args.matrix_cross_coupling),
        pairwise_imitation_strength=float(args.pairwise_imitation_strength),
        pairwise_beta=float(args.pairwise_beta),
        init_bias=float(args.init_bias),
        memory_kernel=int(args.memory_kernel),
        selection_strength=float(args.selection_strength),
        out_root=args.out_root,
        summary_tsv=args.summary_tsv,
        combined_tsv=args.combined_tsv,
        decision_md=args.decision_md,
        original_scout_md=str(args.original_scout_md),
    )
    print(f"summary_tsv={result['summary_tsv']}")
    print(f"combined_tsv={result['combined_tsv']}")
    print(f"decision_md={result['decision_md']}")
    print(f"overall_verdict={result['overall_verdict']}")


if __name__ == "__main__":
    main()
