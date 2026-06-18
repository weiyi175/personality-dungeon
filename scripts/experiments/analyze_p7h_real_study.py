#!/usr/bin/env python
"""
P7-H Part C: Confirmatory study analysis.

Reads landed JSON from the player-test tracker and survey manager (produced by
either the simulated run `run_p7h_player_test.py` or a real-player study via the
API `tracker.save()` / `survey.save()`) and runs the pre-registered analysis:

  H1 (primary, objective) : Welch two-sample t-test on total_displacement
                            (experiment > control) + Cohen's d with 95% CI.
  H2 (exploratory, subjective): Mann-Whitney U on survey UX composite + per-item
                                (effect size + CI reported; no go/no-go — see pre-reg §4).
  H3 (exploratory)        : Spearman correlation between n_critical_crossings
                            and survey UX composite.

Also reports achieved power at the observed N and the N required for 80% power,
and applies Holm correction across the secondary survey items.

Usage
-----
  ./venv/bin/python scripts/experiments/analyze_p7h_real_study.py \
      --sessions reports/experiments/p7h_player_test/p7h_player_test_sessions.json \
      --survey   reports/experiments/p7h_player_test/p7h_survey_responses.json \
      --out      reports/experiments/p7h_real_study
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.stats import norm

_ZA = norm.ppf(1 - 0.025)   # 1.96 (two-sided α=0.05)
_ZB = norm.ppf(0.80)        # 0.8416 (power 0.80)

# Pre-registered inclusion / exclusion constants (P7H_PREREGISTRATION.md §6)
_N_MIN = 10               # minimum action steps per session
_RT_MEDIAN_MIN_MS = 500   # sessions with median RT below this → bot exclusion
_RT_MEDIAN_MAX_MS = 180_000  # sessions with median RT above this → AFK exclusion


# ── Loading ───────────────────────────────────────────────────────────────────

def _rounds_to_collapse(traj: list) -> int:
    """Extract max rl_step_N round number from trajectory, falling back to step count."""
    rounds = []
    for step in traj:
        txt = step.get("action_text", "") if isinstance(step, dict) else getattr(step, "action_text", "")
        if txt.startswith("rl_step_"):
            suffix = txt[len("rl_step_"):]
            if suffix.isdigit():
                rounds.append(int(suffix))
    return max(rounds) if rounds else len(traj)


def _load_sessions(path: Path, human_only: bool = True) -> dict[str, list]:
    with open(path) as f:
        data = json.load(f)
    groups = {"control": [], "experiment": []}
    for _sid, s in data.get("sessions", {}).items():
        if human_only and not s.get("is_human", False):
            continue
        # 排除人格迭代研究 session：run_id 非空者由 run_id→experiment 邏輯強制配臂，
        # 會把迭代研究(naive P-碼 + 實驗者 dev/EXP_PREPILOT)灌進 P7-H 的 experiment 臂、
        # 污染 confirmatory A/B 平衡。P7-H confirmatory session 不帶 run_id。
        if s.get("run_id", ""):
            continue
        if s.get("ended_at") is None:
            continue
        traj = s.get("trajectory", [])
        if len(traj) < _N_MIN:
            continue
        rt_vals = [step["response_time_ms"] for step in traj if step.get("response_time_ms", 0) > 0]
        if rt_vals:
            median_rt = float(np.median(rt_vals))
            if median_rt < _RT_MEDIAN_MIN_MS or median_rt > _RT_MEDIAN_MAX_MS:
                continue
        s["rounds_to_collapse"] = _rounds_to_collapse(traj)
        groups.setdefault(s["group"], []).append(s)
    return groups


def _load_survey(path: Path) -> dict[str, list]:
    if not path.exists():
        return {"control": [], "experiment": []}
    with open(path) as f:
        data = json.load(f)
    groups = {"control": [], "experiment": []}
    for _sid, r in data.get("responses", {}).items():
        groups.setdefault(r["group"], []).append(r)
    return groups


# ── Statistics ────────────────────────────────────────────────────────────────

def cohens_d(exp: np.ndarray, ctrl: np.ndarray) -> tuple[float, tuple[float, float]]:
    n1, n2 = len(exp), len(ctrl)
    ps = np.sqrt(((n1 - 1) * exp.var(ddof=1) + (n2 - 1) * ctrl.var(ddof=1))
                 / (n1 + n2 - 2))
    d = (exp.mean() - ctrl.mean()) / ps if ps > 0 else 0.0
    se = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))
    return d, (d - 1.96 * se, d + 1.96 * se)


def n_per_group_for_power(d: float) -> float:
    if d == 0:
        return float("inf")
    return 2 * (_ZA + _ZB) ** 2 / d**2


def achieved_power(d: float, n_per_group: int) -> float:
    ncp = d * np.sqrt(n_per_group / 2)
    return float(1 - norm.cdf(_ZA - ncp) + norm.cdf(-_ZA - ncp))


def holm_correction(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted = {}
    prev = 0.0
    for rank, (key, p) in enumerate(items):
        adj = min(1.0, (m - rank) * p)
        adj = max(adj, prev)  # enforce monotonicity
        adjusted[key] = adj
        prev = adj
    return adjusted


# ── Analyses ──────────────────────────────────────────────────────────────────

def analyze_h1(groups: dict, dv_key: str = "total_displacement") -> dict:
    """Welch one-sided test (experiment > control) on a displacement DV.

    dv_key selects the objective DV:
      - "total_displacement" : net ‖p_final − p_initial‖ (pre-registered primary).
        Biased once an arm saturates proximity (the aligned arm throttles its own
        motion near the critical point), which can null/reverse the effect.
      - "path_displacement"  : Σ‖p_k − p_{k-1}‖ along the ordered path. Robust to
        saturation because it accumulates all motion regardless of endpoint
        position; offered as a saturation-robust alternative (REGIME_FINDING.md).
    """
    exp = np.array([s[dv_key] for s in groups["experiment"]])
    ctrl = np.array([s[dv_key] for s in groups["control"]])
    if len(exp) < 2 or len(ctrl) < 2:
        return {"error": "insufficient data"}
    t, p_two = stats.ttest_ind(exp, ctrl, equal_var=False)
    p_one = p_two / 2 if t > 0 else 1 - p_two / 2  # one-sided exp > ctrl
    d, ci = cohens_d(exp, ctrl)
    n_pg = min(len(exp), len(ctrl))
    return {
        "endpoint": dv_key,
        "control": {"n": len(ctrl), "mean": float(ctrl.mean()), "sd": float(ctrl.std(ddof=1))},
        "experiment": {"n": len(exp), "mean": float(exp.mean()), "sd": float(exp.std(ddof=1))},
        "welch_t": float(t),
        "p_one_sided": float(p_one),
        "p_two_sided": float(p_two),
        "cohens_d": float(d),
        "cohens_d_95ci": [float(ci[0]), float(ci[1])],
        "achieved_power": achieved_power(d, n_pg),
        "n_per_group_for_80pct": n_per_group_for_power(d),
        "significant": bool(p_one < 0.05),
    }


def analyze_h1_proximity(groups: dict, dv_key: str = "max_proximity") -> dict:
    """Welch one-sided test (experiment > control) on a PROXIMITY DV.

    The manipulation aligns events toward the bifurcation, so its unconfounded
    DV is how close to / how often at the critical point the trajectory gets —
    NOT raw displacement magnitude. Because event intensity is proximity-
    modulated (the aligned arm throttles its own force once near-critical), every
    displacement-magnitude DV (net or path) systematically favours control over a
    long sequence and can null/reverse the effect. A proximity DV does not have
    this confound: it rewards reaching the target the manipulation aims at.

    dv_key: "max_proximity" (continuous; recommended) or "n_critical_crossings"
    (count; the originally pre-registered DV, viable again now that the Space A/B
    fix means proximity is no longer structurally 1.0 for everyone).
    """
    exp = np.array([float(s[dv_key]) for s in groups["experiment"]])
    ctrl = np.array([float(s[dv_key]) for s in groups["control"]])
    if len(exp) < 2 or len(ctrl) < 2:
        return {"error": "insufficient data"}
    # Degenerate guard: a count DV (e.g. n_critical_crossings) can have zero
    # within-group variance (every experiment session crosses exactly once),
    # which makes Welch's t blow up to ±inf and triggers a scipy precision
    # warning. Report the direction without a spurious t in that case.
    if exp.std(ddof=1) == 0 and ctrl.std(ddof=1) == 0:
        same = exp.mean() == ctrl.mean()
        return {
            "endpoint": dv_key,
            "control": {"n": len(ctrl), "mean": float(ctrl.mean()), "sd": 0.0},
            "experiment": {"n": len(exp), "mean": float(exp.mean()), "sd": 0.0},
            "welch_t": None,
            "p_one_sided": 1.0 if same else (0.0 if exp.mean() > ctrl.mean() else 1.0),
            "p_two_sided": 1.0 if same else 0.0,
            "cohens_d": None,
            "cohens_d_95ci": [None, None],
            "achieved_power": None,
            "n_per_group_for_80pct": None,
            "significant": bool((not same) and exp.mean() > ctrl.mean()),
            "note": "zero within-group variance; separation is exact, t/d undefined",
        }
    t, p_two = stats.ttest_ind(exp, ctrl, equal_var=False)
    p_one = p_two / 2 if t > 0 else 1 - p_two / 2
    d, ci = cohens_d(exp, ctrl)
    n_pg = min(len(exp), len(ctrl))
    return {
        "endpoint": dv_key,
        "control": {"n": len(ctrl), "mean": float(ctrl.mean()), "sd": float(ctrl.std(ddof=1))},
        "experiment": {"n": len(exp), "mean": float(exp.mean()), "sd": float(exp.std(ddof=1))},
        "welch_t": float(t),
        "p_one_sided": float(p_one),
        "p_two_sided": float(p_two),
        "cohens_d": float(d),
        "cohens_d_95ci": [float(ci[0]), float(ci[1])],
        "achieved_power": achieved_power(d, n_pg),
        "n_per_group_for_80pct": n_per_group_for_power(d),
        "significant": bool(p_one < 0.05),
    }


def analyze_h1_survival(groups: dict) -> dict:
    """Welch one-sided test: experiment collapses sooner (fewer rounds) than control.

    Hypothesis direction: ctrl_rounds > exp_rounds.
    Cohen's d computed as (ctrl − exp) / pooled_sd so positive d = control survives longer.
    Right-censored sessions (no collapse before N_ROUNDS) contribute their final round;
    for proper censoring handling, a log-rank test is preferred at larger N.
    """
    exp = np.array([s["rounds_to_collapse"] for s in groups["experiment"]], dtype=float)
    ctrl = np.array([s["rounds_to_collapse"] for s in groups["control"]], dtype=float)
    if len(exp) < 2 or len(ctrl) < 2:
        return {"error": "insufficient data"}
    t, p_two = stats.ttest_ind(exp, ctrl, equal_var=False)
    # ctrl > exp → t = (exp - ctrl) should be negative → p_one = p_two/2 when t < 0
    p_one = p_two / 2 if t < 0 else 1 - p_two / 2
    # d = (ctrl − exp) / pooled_sd — positive means control survives longer
    n1, n2 = len(ctrl), len(exp)
    ps = np.sqrt(((n1 - 1) * ctrl.var(ddof=1) + (n2 - 1) * exp.var(ddof=1)) / (n1 + n2 - 2))
    d = float((ctrl.mean() - exp.mean()) / ps) if ps > 0 else 0.0
    se_d = np.sqrt((n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2)))
    ci = (d - 1.96 * se_d, d + 1.96 * se_d)
    n_pg = min(len(exp), len(ctrl))
    return {
        "endpoint": "rounds_to_collapse",
        "direction": "control > experiment (experiment collapses sooner)",
        "control": {"n": int(n1), "mean": float(ctrl.mean()), "sd": float(ctrl.std(ddof=1))},
        "experiment": {"n": int(n2), "mean": float(exp.mean()), "sd": float(exp.std(ddof=1))},
        "welch_t": float(t),
        "p_one_sided": float(p_one),
        "p_two_sided": float(p_two),
        "cohens_d": float(d),
        "cohens_d_95ci": [float(ci[0]), float(ci[1])],
        "achieved_power": achieved_power(d, n_pg),
        "n_per_group_for_80pct": n_per_group_for_power(d),
        "significant": bool(p_one < 0.05),
        "note": "right-censored (no-collapse) sessions contribute final round; "
                "log-rank test recommended at N≥30/group",
    }


def analyze_h2(survey: dict) -> dict:
    if not survey["experiment"] or not survey["control"]:
        return {"error": "no survey data"}

    def composite(r):
        return r["q1_naturalness"] + r["q2_fun"] + r["q3_replay"]

    exp_c = np.array([composite(r) for r in survey["experiment"]])
    ctrl_c = np.array([composite(r) for r in survey["control"]])
    u, p_comp = stats.mannwhitneyu(exp_c, ctrl_c, alternative="greater")

    items = {}
    raw_p = {}
    for key in ("q1_naturalness", "q2_fun", "q3_replay"):
        e = np.array([r[key] for r in survey["experiment"]])
        c = np.array([r[key] for r in survey["control"]])
        u_i, p_i = stats.mannwhitneyu(e, c, alternative="greater")
        raw_p[key] = float(p_i)
        items[key] = {
            "control_mean": float(c.mean()),
            "experiment_mean": float(e.mean()),
            "u": float(u_i),
            "p_raw": float(p_i),
        }
    holm = holm_correction(raw_p)
    for key in items:
        items[key]["p_holm"] = holm[key]
        items[key]["significant_holm"] = bool(holm[key] < 0.05)

    return {
        "composite": {
            "control_mean": float(ctrl_c.mean()),
            "experiment_mean": float(exp_c.mean()),
            "ux_lift": float(exp_c.mean() - ctrl_c.mean()),
            "mann_whitney_u": float(u),
            "p_one_sided": float(p_comp),
            "significant": bool(p_comp < 0.05),
        },
        "items": items,
    }


def analyze_h3(groups: dict, survey: dict) -> dict:
    # H3 x-variable = max_proximity (2026-06-06). The original pre-reg used
    # n_critical_crossings; it was first swapped to total_displacement when the OLD
    # broken apparatus pinned proximity at 1.0 for everyone. The Space A/B fix
    # restores a varying proximity, so we use max_proximity — consistent with the
    # new primary objective DV (H1) and closer to the original crossing intent.
    prox_by_sid, ux_by_sid = {}, {}
    for grp in groups.values():
        for s in grp:
            prox_by_sid[s["session_id"]] = float(s["max_proximity"])
    for grp in survey.values():
        for r in grp:
            ux_by_sid[r["session_id"]] = (
                r["q1_naturalness"] + r["q2_fun"] + r["q3_replay"]
            )
    sids = sorted(set(prox_by_sid) & set(ux_by_sid))
    if len(sids) < 3:
        return {"error": "insufficient paired data"}
    x = np.array([prox_by_sid[s] for s in sids])
    y = np.array([ux_by_sid[s] for s in sids])
    rho, p = stats.spearmanr(x, y)
    return {
        "n_pairs": len(sids),
        "x_variable": "max_proximity",
        "spearman_rho": float(rho),
        "p_two_sided": float(p),
        "significant": bool(p < 0.05),
        "deviation_note": "x=max_proximity (pre-reg: n_critical_crossings; "
                          "interim: total_displacement under the broken apparatus)",
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", required=True)
    ap.add_argument("--survey", required=True)
    ap.add_argument("--out", default="reports/experiments/p7h_real_study")
    ap.add_argument("--include-sim", action="store_true",
                    help="包含非真人 (is_human=False) 的模擬資料（wsim 等）")
    args = ap.parse_args()

    groups = _load_sessions(Path(args.sessions), human_only=not args.include_sim)
    survey = _load_survey(Path(args.survey))

    h1 = analyze_h1(groups, dv_key="total_displacement")
    h1_path = analyze_h1(groups, dv_key="path_displacement")
    h1_prox = analyze_h1_proximity(groups, dv_key="max_proximity")
    h1_cross = analyze_h1_proximity(groups, dv_key="n_critical_crossings")
    h1_surv = analyze_h1_survival(groups)
    h2 = analyze_h2(survey)
    h3 = analyze_h3(groups, survey)

    report = {
        # Primary objective DV (2026-06-06): max_proximity — unconfounded by the
        # proximity-modulated event intensity. See REGIME_FINDING.md.
        "H1_primary_max_proximity": h1_prox,
        # Survival DV: rounds until collapse (ctrl > exp = experiment collapses sooner).
        # Cleanest DV when collapse is proximity-triggered and censoring is modest.
        "H1_survival_rounds": h1_surv,
        # Secondary objective DV: net displacement (valid only sub-saturation).
        "H1b_total_displacement": h1,
        # Diagnostics: alternative DVs kept for transparency / regime auditing.
        "H1_diagnostics": {
            "path_displacement": h1_path,
            "n_critical_crossings": h1_cross,
        },
        "H2_survey": h2,
        "H3_correlation": h3,
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "p7h_real_study_analysis.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Console summary
    print("=" * 68)
    print("P7-H CONFIRMATORY STUDY ANALYSIS")
    print("=" * 68)
    def _print_dv(title: str, hp: dict, show_power: bool = True) -> None:
        if "error" in hp:
            print(f"\n{title}\n  (insufficient data)")
            return
        print(f"\n{title}")
        print(f"  control    n={hp['control']['n']}  mean={hp['control']['mean']:.5f}")
        print(f"  experiment n={hp['experiment']['n']}  mean={hp['experiment']['mean']:.5f}")
        if hp.get("welch_t") is None:
            print(f"  {hp.get('note', 'undefined t/d')}  "
                  f"p(1-sided)={hp['p_one_sided']:.2e}  "
                  f"{'✅ sig' if hp['significant'] else '✗ ns'}")
            return
        print(f"  Welch t={hp['welch_t']:.3f}  p(1-sided)={hp['p_one_sided']:.2e}  "
              f"{'✅ sig' if hp['significant'] else '✗ ns'}")
        print(f"  Cohen's d={hp['cohens_d']:.3f}  "
              f"95%CI [{hp['cohens_d_95ci'][0]:.3f}, {hp['cohens_d_95ci'][1]:.3f}]")
        if show_power:
            print(f"  achieved power={hp['achieved_power']:.3f}  "
                  f"(N/group for 80%: {hp['n_per_group_for_80pct']:.0f})")

    _print_dv("H1 (PRIMARY, objective): max bifurcation proximity", h1_prox)
    _print_dv("H1 (SURVIVAL, objective): rounds until collapse  [ctrl > exp]", h1_surv)
    if "error" not in h1_surv:
        print(f"  note: {h1_surv['note']}")
    _print_dv("H1b (secondary, objective): net displacement ‖Pf−P0‖ "
              "[valid only sub-saturation]", h1)
    print("\n── H1 diagnostics (confounded by proximity-modulated intensity) ──")
    _print_dv("  path length Σ‖Δ‖", h1_path, show_power=False)
    _print_dv("  n_critical_crossings", h1_cross, show_power=False)
    if "error" not in h2:
        c = h2["composite"]
        print("\nH2 (EXPLORATORY, subjective): survey UX composite  [effect size + CI; no go/no-go]")
        print(f"  control={c['control_mean']:.1f}  experiment={c['experiment_mean']:.1f}  "
              f"lift={c['ux_lift']:+.2f}")
        print(f"  Mann-Whitney U={c['mann_whitney_u']:.0f}  p={c['p_one_sided']:.2e}  "
              f"{'✅ sig' if c['significant'] else '✗ ns'}")
        for key, it in h2["items"].items():
            print(f"    {key:16s} ctrl={it['control_mean']:.1f} exp={it['experiment_mean']:.1f} "
                  f"p_holm={it['p_holm']:.2e} {'✅' if it['significant_holm'] else '✗'}")
    if "error" not in h3:
        print("\nH3 (exploratory): max_proximity ↔ UX composite  [pre-reg deviation: crossings→max_proximity]")
        print(f"  Spearman ρ={h3['spearman_rho']:.3f}  p={h3['p_two_sided']:.2e}  "
              f"(n={h3['n_pairs']})  {'✅ sig' if h3['significant'] else '✗ ns'}")

    print(f"\nAnalysis saved: {json_path}")


if __name__ == "__main__":
    main()
