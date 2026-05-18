from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.cycle_metrics import (
    classify_cycle_level,
    peak_to_peak_amplitude,
    phase_direction_consistency_turning,
)

SUMMARY_JSON = ROOT / "outputs/exp_B12_window_sensitivity_summary.json"
OUT_DIR = ROOT / "outputs/actionA_survivor_stability"


@dataclass
class RollingPoint:
    round_end: int
    cycle_level: int
    stage2_passed: bool
    stage3_score: float
    turn_strength: float
    amp_mean: float
    passed: bool


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_series(csv_path: Path) -> dict[str, list[float]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return {
        "round": [int(float(r["round"])) for r in rows],
        "aggressive": [float(r["p_aggressive"]) for r in rows],
        "defensive": [float(r["p_defensive"]) for r in rows],
        "balanced": [float(r["p_balanced"]) for r in rows],
    }


def _find_survivor_seed(module_row: dict) -> int | None:
    short_rows = module_row["windows"]["burn1000_tail1000"].get("per_seed", [])
    long_rows = module_row["windows"]["burn2000_tail2000"].get("per_seed", [])
    long_level_map = {int(r["seed"]): int(r["level"]) for r in long_rows}
    candidates: list[int] = []
    for r in short_rows:
        seed = int(r["seed"])
        if int(r["level"]) >= 3 and long_level_map.get(seed, 0) < 3:
            candidates.append(seed)
    if not candidates:
        return None
    return min(candidates)


def _rolling_metrics(
    series3: dict[str, list[float]],
    *,
    window: int,
    step: int,
    eta: float,
    min_turn_strength: float,
) -> list[RollingPoint]:
    n = min(len(series3["aggressive"]), len(series3["defensive"]), len(series3["balanced"]))
    out: list[RollingPoint] = []
    if n < window:
        return out
    for end in range(window, n + 1, step):
        seg = {
            "aggressive": series3["aggressive"][end - window : end],
            "defensive": series3["defensive"][end - window : end],
            "balanced": series3["balanced"][end - window : end],
        }
        stage3 = phase_direction_consistency_turning(
            seg,
            strategies=("aggressive", "defensive", "balanced"),
            burn_in=0,
            tail=None,
            eta=eta,
            min_turn_strength=min_turn_strength,
            phase_smoothing=1,
        )
        cyc = classify_cycle_level(
            seg,
            burn_in=0,
            tail=None,
            amplitude_threshold=0.02,
            corr_threshold=0.09,
            eta=eta,
            stage3_method="turning",
            phase_smoothing=1,
            min_lag=2,
            max_lag=500,
        )
        amp_mean = (
            peak_to_peak_amplitude(seg["aggressive"])
            + peak_to_peak_amplitude(seg["defensive"])
            + peak_to_peak_amplitude(seg["balanced"])
        ) / 3.0
        out.append(
            RollingPoint(
                round_end=end,
                cycle_level=int(cyc.level),
                stage2_passed=bool(cyc.stage2.passed) if cyc.stage2 is not None else False,
                stage3_score=float(cyc.stage3.score if cyc.stage3 else stage3.score),
                turn_strength=float(cyc.stage3.turn_strength if cyc.stage3 else stage3.turn_strength),
                amp_mean=float(amp_mean),
                passed=bool(cyc.level >= 3),
            )
        )
    return out


def _first_sustained_dropout(roll: list[RollingPoint], sustain_windows: int) -> int | None:
    if not roll:
        return None
    flags = [1 if p.passed else 0 for p in roll]
    try:
        first_pass_idx = flags.index(1)
    except ValueError:
        return None
    n = len(flags)
    k = max(1, int(sustain_windows))
    for i in range(first_pass_idx + 1, max(first_pass_idx + 1, n - k + 1)):
        if flags[i] == 0 and all(flags[j] == 0 for j in range(i, i + k)):
            return int(roll[i].round_end)
    return None


def _summarize_phase(roll: list[RollingPoint]) -> tuple[float, float]:
    if not roll:
        return 0.0, 0.0
    n = len(roll)
    third = max(1, n // 3)
    head = roll[:third]
    tail = roll[-third:]
    head_mean = sum(p.stage3_score for p in head) / float(len(head))
    tail_mean = sum(p.stage3_score for p in tail) / float(len(tail))
    return float(head_mean), float(tail_mean)


def _summarize_stage2_pass_rate(roll: list[RollingPoint]) -> tuple[float, float]:
    if not roll:
        return 0.0, 0.0
    n = len(roll)
    third = max(1, n // 3)
    head = roll[:third]
    tail = roll[-third:]
    head_rate = sum(1 for p in head if p.stage2_passed) / float(len(head))
    tail_rate = sum(1 for p in tail if p.stage2_passed) / float(len(tail))
    return float(head_rate), float(tail_rate)


def _classify_fixed_windows(series3: dict[str, list[float]]) -> dict[str, dict[str, float | int]]:
    base_kwargs = dict(
        amplitude_threshold=0.02,
        corr_threshold=0.09,
        eta=0.55,
        stage3_method="turning",
        phase_smoothing=1,
        min_lag=2,
        max_lag=500,
    )
    short = classify_cycle_level(
        series3,
        burn_in=1000,
        tail=1000,
        **base_kwargs,
    )
    long = classify_cycle_level(
        series3,
        burn_in=2000,
        tail=2000,
        **base_kwargs,
    )
    return {
        "burn1000_tail1000": {
            "level": int(short.level),
            "stage3_score": float(short.stage3.score if short.stage3 else 0.0),
        },
        "burn2000_tail2000": {
            "level": int(long.level),
            "stage3_score": float(long.stage3.score if long.stage3 else 0.0),
        },
    }


def _render_figure(
    *,
    rounds: list[int],
    series3: dict[str, list[float]],
    roll: list[RollingPoint],
    eta: float,
    dropout_round: int | None,
    title: str,
    out_png: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    ax0 = axes[0]
    ax0.plot(rounds, series3["aggressive"], label="p_aggressive", linewidth=1.2)
    ax0.plot(rounds, series3["defensive"], label="p_defensive", linewidth=1.2)
    ax0.plot(rounds, series3["balanced"], label="p_balanced", linewidth=1.2)
    ax0.set_ylabel("proportion")
    ax0.set_title(title)
    ax0.legend(loc="upper right", ncol=3, fontsize=9)
    ax0.grid(alpha=0.25)

    roll_x = [p.round_end for p in roll]
    score_y = [p.stage3_score for p in roll]
    pass_y = [1.0 if p.passed else 0.0 for p in roll]
    level_y = [float(p.cycle_level) for p in roll]

    ax1 = axes[1]
    ax1.plot(roll_x, score_y, color="#1565C0", linewidth=1.6, label="rolling_stage3_score")
    ax1.axhline(eta, color="#C62828", linestyle="--", linewidth=1.2, label=f"eta={eta:.2f}")
    ax1.plot(roll_x, pass_y, color="#2E7D32", linewidth=1.0, alpha=0.75, label="pass_flag")
    ax1.plot(roll_x, level_y, color="#455A64", linewidth=1.0, alpha=0.65, label="cycle_level")
    ax1.set_ylabel("stage3")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="upper right", ncol=3, fontsize=9)

    ax2 = axes[2]
    ax2.plot(roll_x, [p.amp_mean for p in roll], color="#6A1B9A", linewidth=1.4, label="rolling_amp_mean")
    ax2.set_ylabel("amplitude")
    ax2.set_xlabel("round")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="upper right", fontsize=9)

    if dropout_round is not None:
        for ax in axes:
            ax.axvline(dropout_round, color="#EF6C00", linestyle=":", linewidth=1.6)
        ax0.text(
            dropout_round + 20,
            0.96,
            f"dropout~{dropout_round}",
            transform=ax0.get_xaxis_transform(),
            color="#EF6C00",
            fontsize=9,
            va="top",
        )

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    payload = _read_json(SUMMARY_JSON)
    results = payload.get("results", [])
    if not results:
        raise RuntimeError("No module rows found in exp_B12_window_sensitivity_summary.json")

    window = 1000
    step = 20
    eta = 0.55
    min_turn_strength = 0.0
    sustain_windows = 5

    report_rows: list[dict] = []

    for row in results:
        module = str(row["module"])
        condition = str(row["condition"])
        survivor_seed = _find_survivor_seed(row)
        if survivor_seed is None:
            continue

        csv_path = ROOT / "outputs" / f"recovery_{'B1' if module == 'w3_stackelberg' else 'B2'}_ss0.10_mk3_r6000" / condition / f"seed{survivor_seed}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing survivor seed CSV: {csv_path}")

        series = _read_series(csv_path)
        rounds = series["round"]
        series3 = {
            "aggressive": series["aggressive"],
            "defensive": series["defensive"],
            "balanced": series["balanced"],
        }

        roll = _rolling_metrics(
            series3,
            window=window,
            step=step,
            eta=eta,
            min_turn_strength=min_turn_strength,
        )
        dropout_round = _first_sustained_dropout(roll, sustain_windows=sustain_windows)
        score_head, score_tail = _summarize_phase(roll)
        s2_head_rate, s2_tail_rate = _summarize_stage2_pass_rate(roll)
        fixed = _classify_fixed_windows(series3)

        fig_name = f"actionA_{module}_seed{survivor_seed}.png"
        fig_path = OUT_DIR / fig_name
        _render_figure(
            rounds=rounds,
            series3=series3,
            roll=roll,
            eta=eta,
            dropout_round=dropout_round,
            title=f"Action A stability envelope: {module} / seed {survivor_seed}",
            out_png=fig_path,
        )

        report_rows.append(
            {
                "module": module,
                "condition": condition,
                "survivor_seed": survivor_seed,
                "csv": str(csv_path.relative_to(ROOT)),
                "window": window,
                "step": step,
                "eta": eta,
                "sustain_windows": sustain_windows,
                "dropout_round": dropout_round,
                "score_head_mean": score_head,
                "score_tail_mean": score_tail,
                "stage2_pass_rate_head": s2_head_rate,
                "stage2_pass_rate_tail": s2_tail_rate,
                "fixed_window_eval": fixed,
                "figure": str(fig_path.relative_to(ROOT)),
            }
        )

    summary = {
        "generated_at": datetime.now().isoformat(),
        "scope": {
            "selection_strength": 0.10,
            "memory_kernel": 3,
            "rounds": 6000,
            "source": str(SUMMARY_JSON.relative_to(ROOT)),
        },
        "method": {
            "rolling_window": window,
            "rolling_step": step,
            "stage3_eta": eta,
            "sustain_windows_for_dropout": sustain_windows,
        },
        "rows": report_rows,
    }

    out_json = OUT_DIR / "actionA_survivor_stability_summary.json"
    out_md = OUT_DIR / "actionA_survivor_stability_onepager.md"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Action A 診斷一頁摘要")
    lines.append("")
    lines.append("## 實驗設定")
    lines.append("- 來源資料: 既有 6000 輪 seed CSV（未重跑模擬）")
    lines.append("- 固定參數: SS=0.10, MK=3, rounds=6000")
    lines.append(f"- rolling 視窗: {window}, step: {step}, eta: {eta:.2f}")
    lines.append("")
    lines.append("## 核心發現")
    lines.append("| module | survivor seed | burn/tail=1000 | burn/tail=2000 | dropout round | score(head) | score(tail) | S2 pass(head) | S2 pass(tail) |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in report_rows:
        short = r["fixed_window_eval"]["burn1000_tail1000"]
        long = r["fixed_window_eval"]["burn2000_tail2000"]
        drop = r["dropout_round"] if r["dropout_round"] is not None else "NA"
        lines.append(
            "| {m} | {s} | L{l1} / {s31:.3f} | L{l2} / {s32:.3f} | {d} | {h:.3f} | {t:.3f} | {s2h:.3f} | {s2t:.3f} |".format(
                m=r["module"],
                s=r["survivor_seed"],
                l1=short["level"],
                s31=short["stage3_score"],
                l2=long["level"],
                s32=long["stage3_score"],
                d=drop,
                h=r["score_head_mean"],
                t=r["score_tail_mean"],
                s2h=r["stage2_pass_rate_head"],
                s2t=r["stage2_pass_rate_tail"],
            )
        )
    lines.append("")
    lines.append("## 圖表")
    for r in report_rows:
        lines.append(f"- {r['module']} seed {r['survivor_seed']}: {r['figure']}")
    lines.append("")
    lines.append("## 診斷判讀")
    lines.append("- 兩個模組的倖存 seed 都在短視窗可達 L3，但在長視窗退化為非 L3。")
    lines.append("- 末段 Stage3 score 仍可維持在高位，但 Stage2 pass rate 下滑，顯示失效更像是頻率一致性崩解而非單純相位方向消失。")
    lines.append("- rolling 指標可定位『掉隊時間』，支持亞穩態而非全域吸引子的解讀。")
    lines.append("- 下一步可在同 seed 上做 trait ablation（例如 z_exploring=0）驗證因果。")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"written: {out_json.relative_to(ROOT)}")
    print(f"written: {out_md.relative_to(ROOT)}")
    for r in report_rows:
        print(
            "module={m} seed={s} dropout={d} short=L{sl}/s3={ss:.3f} long=L{ll}/s3={ls:.3f}".format(
                m=r["module"],
                s=r["survivor_seed"],
                d=r["dropout_round"],
                sl=r["fixed_window_eval"]["burn1000_tail1000"]["level"],
                ss=r["fixed_window_eval"]["burn1000_tail1000"]["stage3_score"],
                ll=r["fixed_window_eval"]["burn2000_tail2000"]["level"],
                ls=r["fixed_window_eval"]["burn2000_tail2000"]["stage3_score"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
