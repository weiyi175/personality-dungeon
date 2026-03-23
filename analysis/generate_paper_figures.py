"""
Generate paper figures for Personality Dungeon v1 baseline.

Produces:
  outputs/figures/fig1_timeseries.png   — baseline vs event strategy proportions
  outputs/figures/fig2_phase_portrait.png — phase portrait (tail 25%)
  outputs/figures/fig3_multiseed.png    — 10-seed overlay (p_aggressive)
  outputs/figures/fig4_bifurcation.png  — bifurcation diagram (ss vs gamma_delta)
  outputs/figures/fig5_mechanism.png    — gate-stabilised limit cycle schematic

Usage:
  ./venv/bin/python -m analysis.generate_paper_figures
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from analysis.cycle_metrics import classify_cycle_level

OUT_DIR = Path("outputs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "aggressive": "#d62728",
    "defensive":  "#1f77b4",
    "balanced":   "#2ca02c",
}
STRATEGY_LABELS = {"aggressive": "Aggressive", "defensive": "Defensive", "balanced": "Balanced"}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_strategy_df(csv_path: str | Path, downsample: int = 4) -> dict[str, np.ndarray]:
    """Return dict with keys: round, p_aggressive, p_defensive, p_balanced."""
    path = Path(csv_path)
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    # apply downsample
    rows = rows[::downsample]
    result: dict[str, np.ndarray] = {}
    for key in ("round", "p_aggressive", "p_defensive", "p_balanced"):
        result[key] = np.array([float(r[key]) for r in rows])
    return result


def _tail(d: dict[str, np.ndarray], fraction: float = 0.25) -> dict[str, np.ndarray]:
    n = len(d["round"])
    start = int(n * (1 - fraction))
    return {k: v[start:] for k, v in d.items()}


def _tail_last_n(d: dict[str, np.ndarray], n_last: int) -> dict[str, np.ndarray]:
    start = max(0, len(d["round"]) - int(n_last))
    return {k: v[start:] for k, v in d.items()}


def _diagnose_cycle(csv_path: str | Path, *, tail: int = 1000) -> dict[str, float | int | None]:
    path = Path(csv_path)
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    series = {
        "aggressive": [float(r["p_aggressive"]) for r in rows],
        "defensive": [float(r["p_defensive"]) for r in rows],
        "balanced": [float(r["p_balanced"]) for r in rows],
    }
    burn_in = max(2400, len(rows) // 3) if len(rows) > 8000 else 2400
    result = classify_cycle_level(
        series,
        burn_in=burn_in,
        tail=int(tail),
        amplitude_threshold=0.02,
        corr_threshold=0.09,
        eta=0.55,
        stage3_method="turning",
        phase_smoothing=1,
        min_lag=2,
        max_lag=500,
    )
    score = result.stage3.score if result.stage3 is not None else None
    turn_strength = result.stage3.turn_strength if result.stage3 is not None else None
    return {
        "level": int(result.level),
        "score": float(score) if score is not None else None,
        "turn_strength": float(turn_strength) if turn_strength is not None else None,
        "rounds": int(len(rows)),
        "burn_in": int(burn_in),
        "tail": int(tail),
    }


def _format_metric(value: float | None, *, digits: int) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _gamma_from_json(json_path: str | Path) -> float | None:
    try:
        d = json.loads(Path(json_path).read_text())
        return d.get("envelope_gamma_comparison", {}).get("gamma_delta")
    except Exception:
        return None


def _level_from_json(json_path: str | Path) -> int | None:
    try:
        d = json.loads(Path(json_path).read_text())
        return d.get("derived_cycle_level")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Fig 1 — Time series: baseline vs event (side-by-side panels)
# ---------------------------------------------------------------------------

def fig1_timeseries(
    baseline_csv: str = "outputs/sweep_expected_baseline_seed42.csv",
    event_csv: str = "outputs/longrun_ss0p06_r8000_seed42.csv",
    out: str = "outputs/figures/fig1_timeseries.png",
) -> None:
    base_d = _load_strategy_df(baseline_csv, downsample=4)
    event_d = _load_strategy_df(event_csv, downsample=4)

    # baseline only has 4000 rounds; event has 8000
    # show rounds 0–4000 for both to make side-by-side fair
    mask_b = base_d["round"] <= 4000
    mask_e = event_d["round"] <= 4000
    base_sub = {k: v[mask_b] for k, v in base_d.items()}
    event_sub = {k: v[mask_e] for k, v in event_d.items()}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    for col, label in [("p_aggressive", "Aggressive"), ("p_defensive", "Defensive"), ("p_balanced", "Balanced")]:
        c = PALETTE[col.split("_")[1]]
        axes[0].plot(base_sub["round"], base_sub[col], color=c, lw=1.2, label=label, alpha=0.85)
        axes[1].plot(event_sub["round"], event_sub[col], color=c, lw=1.2, label=label, alpha=0.85)

    for ax, title, tag in zip(axes,
                                ["(a) Payoff-only baseline (Level 0)", "(b) Event loop enabled (Level 2)"],
                                ["Stationary — γ = +6.26×10⁻⁶", "Structural cycle — Δγ = −2.97×10⁻⁵"]):
        ax.set_xlabel("Round")
        ax.set_title(title, pad=6)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(0, 4000)
        ax.text(0.97, 0.97, tag, transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color="#555555",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.8))

    axes[0].set_ylabel("Strategy proportion")
    axes[1].legend(loc="center right", framealpha=0.85)

    fig.suptitle("Strategy Proportions: Payoff-Only Baseline vs. Personality Event Loop",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig1] saved → {out}")


# ---------------------------------------------------------------------------
# Fig 2 — Phase portrait (p_aggressive vs p_defensive, tail 25%)
# ---------------------------------------------------------------------------

def fig2_phase_portrait(
    baseline_csv: str = "outputs/sweep_expected_baseline_seed42.csv",
    event_csv: str = "outputs/longrun_ss0p06_r8000_seed42.csv",
    out: str = "outputs/figures/fig2_phase_portrait.png",
) -> None:
    base_d = _load_strategy_df(baseline_csv, downsample=2)
    event_d = _load_strategy_df(event_csv, downsample=2)

    base_tail = _tail(base_d, 0.40)   # last 40% for baseline (4000 rounds)
    event_tail = _tail(event_d, 0.25)  # last 25% of 8000 rounds = 2000

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, d, title, color, period_note in [
        (axes[0], base_tail, "(a) Baseline — tail 40%", "#1f77b4", "Level 0: diffuse cloud"),
        (axes[1], event_tail, "(b) Event loop — tail 25%", "#d62728", "Level 2: closed orbit"),
    ]:
        n = len(d["round"])
        alpha_vals = np.linspace(0.15, 0.9, n)
        for i in range(n - 1):
            ax.plot(d["p_aggressive"][i:i+2],
                    d["p_defensive"][i:i+2],
                    color=color, alpha=float(alpha_vals[i]), lw=0.8)
        # mark start and end
        ax.scatter(d["p_aggressive"][0], d["p_defensive"][0],
                   c="green", s=40, zorder=5, label="Start", marker="o")
        ax.scatter(d["p_aggressive"][-1], d["p_defensive"][-1],
                   c="black", s=40, zorder=5, label="End", marker="X")

        ax.set_xlabel("p(Aggressive)")
        ax.set_ylabel("p(Defensive)")
        ax.set_title(title, pad=6)
        ax.text(0.97, 0.97, period_note, transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color="#555555",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.8))
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=9, loc="lower right")

    fig.suptitle("Phase Portrait: p(Aggressive) vs p(Defensive)", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig2] saved → {out}")


# ---------------------------------------------------------------------------
# Fig 3 — Multi-seed overlay (p_aggressive time series, seeds 0-9)
# ---------------------------------------------------------------------------

def fig3_multiseed(
    seed_pattern: str = "outputs/robust_ss0p06_seed{seed}.csv",
    seeds: range = range(10),
    out: str = "outputs/figures/fig3_multiseed.png",
) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(16, 6), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    cmap = plt.cm.tab10
    for i, seed in enumerate(seeds):
        csv_path = seed_pattern.format(seed=seed)
        d = _load_strategy_df(csv_path, downsample=3)
        ax = axes_flat[i]
        for col, lbl in [("p_aggressive", "Agg"), ("p_defensive", "Def"), ("p_balanced", "Bal")]:
            c = PALETTE[col.split("_")[1]]
            ax.plot(d["round"], d[col], color=c, lw=1.0, alpha=0.8,
                    label=lbl if i == 0 else None)
        ax.set_title(f"seed {seed}", pad=4, fontsize=10)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(0, d["round"].max())

    # shared labels
    for ax in axes[1]:
        ax.set_xlabel("Round", fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel("Strategy proportion", fontsize=10)

    # legend from first panel
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.0, 1.0),
               fontsize=10, framealpha=0.9)

    fig.suptitle(
        "Multi-Seed Robustness: Strategy Proportions (σ = 0.06, rounds = 4 000, 10 seeds)\n"
        "Pr(Level ≥ 2) = 10/10 = 1.0",
        fontsize=12, y=1.02
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig3] saved → {out}")


# ---------------------------------------------------------------------------
# Fig 4 — Bifurcation diagram: ss vs gamma_delta and level
# ---------------------------------------------------------------------------

def fig4_bifurcation(
    json_pattern: str = "outputs/diag_grid_ft72_ss{ss_tag}_bias0p12.json",
    ss_values: list[tuple[float, str]] | None = None,
    extra_points: list[tuple[float, str]] | None = None,
    out: str = "outputs/figures/fig4_bifurcation.png",
) -> None:
    if ss_values is None:
        ss_values = [
            (0.02, "0p02"), (0.03, "0p03"), (0.04, "0p04"), (0.05, "0p05"),
            (0.055, None),  # extra boundary point
            (0.06, "0p06"), (0.065, None),  # extra boundary point
            (0.07, "0p07"), (0.08, "0p08"), (0.09, "0p09"), (0.10, "0p10"),
        ]

    ss_list: list[float] = []
    gd_list: list[float] = []
    lv_list: list[int] = []

    for ss_num, ss_tag in ss_values:
        if ss_tag is not None:
            jpath = json_pattern.format(ss_tag=ss_tag)
        else:
            # boundary points
            btag_map = {0.055: "boundary_ss0p055_bias0p12", 0.065: "boundary_ss0p065_bias0p12"}
            jpath = f"outputs/diag_{btag_map[ss_num]}.json"
        gd = _gamma_from_json(jpath)
        lv = _level_from_json(jpath)
        if gd is None:
            continue
        ss_list.append(ss_num)
        gd_list.append(gd * 1e5)   # scale to ×10⁻⁵ for readability
        lv_list.append(lv or 0)

    ss_arr = np.array(ss_list)
    gd_arr = np.array(gd_list)
    lv_arr = np.array(lv_list)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})

    # upper panel: gamma_delta
    ax_g = axes[0]
    ax_g.axhline(0, color="#999999", lw=0.8, ls="--", label="γ_delta = 0")
    ax_g.plot(ss_arr, gd_arr, "o-", color="#1f77b4", lw=1.8, ms=6, zorder=3, label="γ_delta (×10⁻⁵)")
    # shade Level 2 region
    level2_mask = lv_arr == 2
    if level2_mask.any():
        x_l2 = ss_arr[level2_mask]
        ax_g.axvspan(x_l2.min() - 0.003, x_l2.max() + 0.003,
                     alpha=0.08, color="#2ca02c", label="Level 2 region")
    ax_g.set_ylabel("γ_delta (×10⁻⁵)", fontsize=11)
    ax_g.set_title("Bifurcation Diagram: Selection Strength vs Δγ and Cycle Level\n"
                   "(ft = 0.72, hp = 0.10, bias = 0.12, rounds = 4 000, seed = 42)",
                   fontsize=11, pad=8)
    ax_g.legend(fontsize=9, loc="lower left")
    ax_g.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax_g.grid(axis="y", ls=":", alpha=0.4)

    # lower panel: cycle level
    ax_l = axes[1]
    colors_lv = [("#2ca02c" if lv >= 2 else "#d62728") for lv in lv_list]
    ax_l.scatter(ss_arr, lv_arr, c=colors_lv, s=60, zorder=3)
    ax_l.plot(ss_arr, lv_arr, "-", color="#555555", lw=1.0, zorder=2)
    ax_l.set_yticks([0, 1, 2, 3])
    ax_l.set_ylim(-0.3, 3.3)
    ax_l.set_ylabel("Cycle\nLevel", fontsize=11)
    ax_l.set_xlabel("Selection strength (σ)", fontsize=11)
    ax_l.grid(axis="y", ls=":", alpha=0.4)

    # annotate sweet-spot bracket
    ax_g.annotate("", xy=(0.065, gd_arr[ss_arr == 0.065][0] - 0.15),
                  xytext=(0.055, gd_arr[ss_arr == 0.065][0] - 0.15),
                  arrowprops=dict(arrowstyle="<->", color="#e37b40", lw=1.5))
    ax_g.text(0.060, gd_arr[ss_arr == 0.06][0] - 0.45,
              "tested\nsweet-spot", ha="center", va="top", fontsize=8, color="#e37b40")

    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig4] saved → {out}")


# ---------------------------------------------------------------------------
# Fig 5 — Mechanism schematic: gate-stabilised limit cycle
# ---------------------------------------------------------------------------

def fig5_mechanism(
    baseline_csv: str = "outputs/sweep_expected_baseline_seed42.csv",
    event_csv: str = "outputs/longrun_ss0p06_r8000_seed42.csv",
    out: str = "outputs/figures/fig5_mechanism.png",
) -> None:
    """Three-panel schematic of the gate-stabilised limit cycle mechanism.

    Panel (a): Real baseline phase portrait — diverging / cloud (γ > 0)
    Panel (b): Real event phase portrait — closed orbit (γ < 0) with gate zone shading
    Panel (c): Δγ vs σ log–log scaling (Δγ ∝ σ^0.08 ≈ const)
    """
    fig = plt.figure(figsize=(15, 4.8))
    gs = fig.add_gridspec(1, 3, wspace=0.38)

    # --- shared data ---
    base_d  = _load_strategy_df(baseline_csv, downsample=2)
    event_d = _load_strategy_df(event_csv, downsample=2)
    base_tail  = _tail(base_d,  0.60)   # last 60% of 4000-round baseline
    event_tail = _tail(event_d, 0.30)   # last 30% of 8000-round event run

    # -------------------------------------------------------------------
    # Panel (a): baseline phase portrait (marginally unstable — spirals out)
    # -------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0])
    n = len(base_tail["round"])
    alpha_vals = np.linspace(0.10, 0.70, n)
    for i in range(n - 1):
        ax_a.plot(base_tail["p_aggressive"][i:i+2],
                  base_tail["p_defensive"][i:i+2],
                  color="#1f77b4", alpha=float(alpha_vals[i]), lw=0.7)
    ax_a.scatter(base_tail["p_aggressive"][-1], base_tail["p_defensive"][-1],
                 c="black", s=25, zorder=5, marker="X")
    ax_a.set_xlabel("p(Aggressive)", fontsize=10)
    ax_a.set_ylabel("p(Defensive)", fontsize=10)
    ax_a.set_title("(a) No events\nγ > 0 — diffuse cloud", fontsize=10, pad=6)
    ax_a.set_xlim(0.10, 0.70); ax_a.set_ylim(0.10, 0.70)
    ax_a.text(0.50, 0.18, "marginally unstable\ncenter (γ = +6.26×10⁻⁶)",
              ha="center", va="bottom", fontsize=8, color="#1f77b4",
              bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#aaaaaa", alpha=0.85))
    # draw arrow hinting outward drift
    ax_a.annotate("", xy=(0.52, 0.56), xytext=(0.38, 0.42),
                  arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.2))

    # -------------------------------------------------------------------
    # Panel (b): event phase portrait with gate zone + closed orbit
    # -------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[1])
    n2 = len(event_tail["round"])
    alpha_vals2 = np.linspace(0.20, 0.90, n2)
    for i in range(n2 - 1):
        ax_b.plot(event_tail["p_aggressive"][i:i+2],
                  event_tail["p_defensive"][i:i+2],
                  color="#d62728", alpha=float(alpha_vals2[i]), lw=0.8)

    # shade gate-active zone (high p_aggressive → high final_risk → gate fires)
    ax_b.axvspan(0.55, 0.75, alpha=0.10, color="#d62728",
                 label="Gate-firing zone\n(Aggressive dominant)")
    ax_b.axhspan(0.55, 0.75, alpha=0.07, color="#1f77b4",
                 label="Gate-firing zone\n(Defensive dominant)")

    ax_b.set_xlabel("p(Aggressive)", fontsize=10)
    ax_b.set_ylabel("p(Defensive)", fontsize=10)
    ax_b.set_title("(b) With event gate\nγ < 0 — gate-stabilised limit cycle", fontsize=10, pad=6)
    ax_b.set_xlim(0.10, 0.75); ax_b.set_ylim(0.10, 0.75)
    ax_b.text(0.42, 0.15, "closed orbit\n(Δγ = −2.97×10⁻⁵)",
              ha="center", va="bottom", fontsize=8, color="#d62728",
              bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#aaaaaa", alpha=0.85))

    # annotate gate clipping arrows
    ax_b.annotate("gate clipping\n⬅ cuts amplitude",
                  xy=(0.61, 0.38), xytext=(0.63, 0.24),
                  arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.0),
                  fontsize=7.5, color="#d62728", ha="center")

    # -------------------------------------------------------------------
    # Panel (c): Δγ vs σ power-law scaling
    # -------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[2])

    # Extended boundary + sweep data (from §3.6)
    sigma_pts = np.array([0.005, 0.010, 0.015, 0.02, 0.03, 0.04, 0.05,
                          0.06,  0.07,  0.08,  0.09, 0.10, 0.12, 0.15, 0.18, 0.20])
    dg_pts = np.array([-2.74, -2.82, -2.83, -2.87, -2.94, -3.07, -3.15,
                       -2.97, -3.00, -3.10, -3.15, -3.15, -3.30, -3.43, -3.63, -3.76]) * 1e-5

    # power-law fit line
    sigma_fine = np.logspace(np.log10(0.003), np.log10(0.25), 80)
    dg_fit = -4.09e-5 * sigma_fine ** 0.08

    ax_c.scatter(sigma_pts, -dg_pts * 1e5, s=28, color="#2ca02c", zorder=4, label="Data (|Δγ|×10⁵)")
    ax_c.plot(sigma_fine, -dg_fit * 1e5, "--", color="#888888", lw=1.5, label="Fit: 4.09×σ^{0.08}")

    ax_c.set_xscale("log")
    ax_c.set_xlabel("Selection strength σ  (log scale)", fontsize=10)
    ax_c.set_ylabel("|Δγ| × 10⁵", fontsize=10)
    ax_c.set_title("(c) |Δγ| vs σ — sub-linear scaling\nconfirms σ-independence", fontsize=10, pad=6)
    ax_c.legend(fontsize=8.5, loc="upper left")
    ax_c.text(0.97, 0.18,
              "σ → 0: Δγ finite\n⟹ gate mechanism\nalways active",
              transform=ax_c.transAxes, ha="right", va="bottom",
              fontsize=8, color="#555555",
              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.85))
    ax_c.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax_c.grid(axis="y", ls=":", alpha=0.35)

    # σ-axis annotation: "faster" arrow
    ax_c.annotate("faster\ncycles →", xy=(0.18, 3.55), fontsize=7.5,
                  color="#555555", ha="center")

    fig.suptitle(
        "Gate-Stabilised Limit Cycle: Mechanism Summary\n"
        "(a) no gate → divergent   "
        "(b) gate active → closed orbit   "
        "(c) orbit persists at all σ",
        fontsize=11, y=1.02
    )
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig5] saved → {out}")


# ---------------------------------------------------------------------------
# Fig 6 — Cross-term transient strengthening vs long-run fallback
# ---------------------------------------------------------------------------

def fig6_cross_term_transient(
    control_csv: str = "outputs/control_cross0_seed45_sampled.csv",
    best_csv: str = "outputs/sampled_seed45_apv2_s1p5_i400_crossc0p16.csv",
    fallback_csv: str = "outputs/triplecheck_cross_seed45_r10000_c0p16.csv",
    out: str = "outputs/figures/fig6_cross_term_transient.png",
    summary_csv: str = "outputs/figures/table_cross_term_transient_summary.csv",
) -> None:
    specs = [
        {
            "key": "control",
            "label": "control cross=0.0, seed=45, r=8000",
            "csv": control_csv,
            "color": "#4c566a",
            "linestyle": "--",
        },
        {
            "key": "best",
            "label": "cross=0.16, seed=45, r=8000",
            "csv": best_csv,
            "color": "#c44e52",
            "linestyle": "-",
        },
        {
            "key": "fallback",
            "label": "cross=0.16, seed=45, r=10000 (tail=1000)",
            "csv": fallback_csv,
            "color": "#8172b2",
            "linestyle": ":",
        },
    ]

    for spec in specs:
        spec["diag"] = _diagnose_cycle(spec["csv"], tail=1000)
        spec["data"] = _tail_last_n(_load_strategy_df(spec["csv"], downsample=1), 1000)

    fig = plt.figure(figsize=(14.2, 8.2))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.7, 1.0], wspace=0.26, hspace=0.18)
    strategy_rows = [
        ("p_aggressive", "Aggressive", PALETTE["aggressive"]),
        ("p_defensive", "Defensive", PALETTE["defensive"]),
        ("p_balanced", "Balanced", PALETTE["balanced"]),
    ]

    time_axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    phase_ax = fig.add_subplot(gs[:, 1])

    for ax, (col, label, strategy_color) in zip(time_axes, strategy_rows):
        y_all = np.concatenate([spec["data"][col] for spec in specs])
        y_pad = max(0.015, float((y_all.max() - y_all.min()) * 0.18))
        y_lo = max(0.0, float(y_all.min() - y_pad))
        y_hi = min(1.0, float(y_all.max() + y_pad))
        for spec in specs:
            rounds = spec["data"]["round"]
            rounds = rounds - rounds[0] + 1
            diag = spec["diag"]
            metric_label = (
                f"{spec['label']} | L{diag['level']} "
                f"score={_format_metric(diag['score'], digits=4)}"
            )
            ax.plot(
                rounds,
                spec["data"][col],
                color=spec["color"],
                lw=2.0,
                ls=spec["linestyle"],
                alpha=0.95,
                label=metric_label,
            )
        ax.set_ylabel(f"{label}\nproportion", color=strategy_color)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlim(1, 1000)
        ax.grid(axis="y", ls=":", alpha=0.28)
        ax.text(
            0.99,
            0.92,
            label,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            color=strategy_color,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#d8dee9", alpha=0.85),
        )
    time_axes[-1].set_xlabel("Tail window round index (last 1000 rounds)")
    time_axes[0].legend(loc="upper left", fontsize=8.3, framealpha=0.92)
    time_axes[0].set_title(
        "Tail-1000 strategy proportions: short-run uplift vs long-run fallback",
        pad=8,
    )

    for spec in specs:
        d = spec["data"]
        alpha_vals = np.linspace(0.18, 0.92, len(d["round"]))
        for i in range(len(d["round"]) - 1):
            phase_ax.plot(
                d["p_aggressive"][i:i+2],
                d["p_defensive"][i:i+2],
                color=spec["color"],
                alpha=float(alpha_vals[i]),
                lw=1.0,
                ls=spec["linestyle"],
            )
        phase_ax.scatter(
            d["p_aggressive"][0],
            d["p_defensive"][0],
            color=spec["color"],
            s=36,
            marker="o",
            zorder=5,
        )
        phase_ax.scatter(
            d["p_aggressive"][-1],
            d["p_defensive"][-1],
            color=spec["color"],
            s=44,
            marker="X",
            zorder=6,
        )

    phase_x = np.concatenate([spec["data"]["p_aggressive"] for spec in specs])
    phase_y = np.concatenate([spec["data"]["p_defensive"] for spec in specs])
    x_pad = max(0.01, float((phase_x.max() - phase_x.min()) * 0.22))
    y_pad = max(0.01, float((phase_y.max() - phase_y.min()) * 0.22))
    phase_ax.set_xlim(max(0.0, float(phase_x.min() - x_pad)), min(1.0, float(phase_x.max() + x_pad)))
    phase_ax.set_ylim(max(0.0, float(phase_y.min() - y_pad)), min(1.0, float(phase_y.max() + y_pad)))
    phase_ax.set_xlabel("p(Aggressive)")
    phase_ax.set_ylabel("p(Defensive)")
    phase_ax.set_title("Phase portrait over the same tail-1000 windows", pad=8)
    phase_ax.grid(ls=":", alpha=0.28)
    phase_ax.set_aspect("equal", adjustable="box")
    phase_ax.text(
        0.03,
        0.03,
        "Cross-term tightens the short-run orbit,\n"
        "but the 10000-round tail falls back toward a looser Level 2 loop.",
        transform=phase_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#444444",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9),
    )

    fig.suptitle(
        "Cross-term transient strengthening is visible, but not persistent",
        fontsize=14,
        y=0.975,
    )
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.08, top=0.90, wspace=0.28, hspace=0.25)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    rows = []
    for spec in specs:
        diag = spec["diag"]
        rows.append(
            {
                "condition": spec["label"],
                "rounds": int(diag["rounds"]),
                "tail": int(diag["tail"]),
                "level": int(diag["level"]),
                "score": _format_metric(diag["score"], digits=4),
                "turn_strength": _format_metric(diag["turn_strength"], digits=6),
            }
        )
    with Path(summary_csv).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[fig6] saved → {out}")
    print(f"[fig6-table] saved → {summary_csv}")


def export_cross_term_robustness_table(
    out_csv: str = "outputs/figures/table_cross_term_robustness.csv",
    out_md: str = "outputs/figures/table_cross_term_robustness.md",
) -> None:
    specs = [
        ("control_cross0", "outputs/control_cross0_seed45_sampled.csv", 8000, 45, 0.0, 1000, "baseline"),
        ("cross0p16 (original best)", "outputs/sampled_seed45_apv2_s1p5_i400_crossc0p16.csv", 8000, 45, 0.16, 1000, "transient Level 3"),
        ("cross0p16 seed48", "outputs/multiseed_light_crossc0p16_seed48.csv", 8000, 48, 0.16, 1000, "seed sensitive"),
        ("cross0p16 seed51", "outputs/multiseed_light_crossc0p16_seed51.csv", 8000, 51, 0.16, 1000, "seed sensitive"),
        ("cross0p155 seed45", "outputs/triplecheck_cross_seed45_r10000_c0p155.csv", 10000, 45, 0.155, 1000, "long-run fallback"),
        ("cross0p16 seed45", "outputs/triplecheck_cross_seed45_r10000_c0p16.csv", 10000, 45, 0.16, 1000, "long-run fallback"),
        ("cross0p165 seed45", "outputs/triplecheck_cross_seed45_r10000_c0p165.csv", 10000, 45, 0.165, 1000, "long-run fallback"),
    ]
    rows = []
    for condition, csv_path, rounds, seed, cross, tail, note in specs:
        diag = _diagnose_cycle(csv_path, tail=tail)
        rows.append(
            {
                "Condition": condition,
                "Rounds": int(rounds),
                "Seed": int(seed),
                "cross": float(cross),
                "tail": int(tail),
                "Level": int(diag["level"]),
                "score": _format_metric(diag["score"], digits=4),
                "turn_strength": _format_metric(diag["turn_strength"], digits=6),
                "note": note,
            }
        )

    out_csv_path = Path(out_csv)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with out_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "| Condition | Rounds | Seed | cross | tail | Level | score | turn_strength | note |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['Condition']} | {row['Rounds']} | {row['Seed']} | {row['cross']:.3f} | {row['tail']} | {row['Level']} | {row['score']} | {row['turn_strength']} | {row['note']} |"
        )
    Path(out_md).write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[cross-table] saved → {out_csv}")
    print(f"[cross-table] saved → {out_md}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating paper figures …")
    fig1_timeseries()
    fig2_phase_portrait()
    fig3_multiseed()
    fig4_bifurcation()
    fig5_mechanism()
    fig6_cross_term_transient()
    export_cross_term_robustness_table()
    print(f"\nAll figures saved to {OUT_DIR}/")
