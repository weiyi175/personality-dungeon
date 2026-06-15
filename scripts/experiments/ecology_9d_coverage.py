#!/usr/bin/env python3
"""(a) 9D 人格分佈 viz + balanced 餓死診斷。

問：q* 說 balanced 該佔 0.32，實測聚合只有 ~0.05。是動力學排斥，還是投影/抽樣？
（step 0 已證生態層不跑 replicator，所以一定是後者——這裡量化『後者』到底是什麼。）

資料：
  - 104 筆 ecology submission（personality_9d，已落檔的族群）
  - 150 筆 p7h 真人 will_sbert_vector（9D，真人分佈；legacy_wills.json 只有文字無向量，不用）

輸出：reports/ecology/coverage_9d.json + coverage_9d.png（PCA 2D 散點，archetype 著色）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from api.ecology_tracker import (  # noqa: E402
    ARCHETYPES, FEATURE_NAMES, personality_to_archetype_soft, _f,
)


def load_ecology() -> list[list[float]]:
    d = json.load(open("reports/ecology/ecology_state.json"))
    return [s["personality_9d"] for s in d["submissions"] if len(s["personality_9d"]) == 9]


def load_p7h_human() -> list[list[float]]:
    d = json.load(open("reports/experiments/p7h_real_study/p7h_player_test_sessions.json"))
    return [v["will_sbert_vector"] for v in d.get("sessions", {}).values()
            if v.get("is_human") and len(v.get("will_sbert_vector", [])) == 9]


def proj_scores(vec9: list[float]) -> tuple[float, float, float, float]:
    """重算 ecology 投影的三個原始分數 + extremity 懲罰（診斷用）。"""
    s_agg = _f(vec9, "impulsiveness") + _f(vec9, "assertiveness") - _f(vec9, "risk_aversion")
    s_def = (_f(vec9, "risk_aversion") + _f(vec9, "suspicion")
             + _f(vec9, "endurance") + _f(vec9, "stability_seeking"))
    extremity = sum(abs(x) for x in vec9[:9]) / max(1, len(vec9))
    s_bal = _f(vec9, "optimism") + _f(vec9, "curiosity") - extremity
    return s_agg, s_def, s_bal, extremity


def diagnose(name: str, vecs: list[list[float]]) -> dict:
    n = len(vecs)
    argmax = [int(np.argmax(personality_to_archetype_soft(v))) for v in vecs]
    frac = [argmax.count(k) / n for k in range(3)]
    sc = np.array([proj_scores(v) for v in vecs])   # (n,4): agg,def,bal,extremity
    out = {
        "name": name, "n": n,
        "argmax_fraction": dict(zip(ARCHETYPES, [round(x, 3) for x in frac])),
        "mean_raw_score": dict(zip(ARCHETYPES, [round(float(x), 3) for x in sc[:, :3].mean(0)])),
        "mean_extremity_penalty_on_balanced": round(float(sc[:, 3].mean()), 3),
        "balanced_argmax_count": argmax.count(2),
    }
    return out


def main() -> None:
    eco, hum = load_ecology(), load_p7h_human()
    print(f"ecology submissions: {len(eco)}   p7h human wills: {len(hum)}\n")

    diags = [diagnose("ecology_104", eco), diagnose("p7h_human_150", hum)]
    for d in diags:
        print(f"[{d['name']}] n={d['n']}")
        print(f"  argmax archetype fraction : {d['argmax_fraction']}")
        print(f"  mean RAW projection score : {d['mean_raw_score']}")
        print(f"  mean extremity penalty    : {d['mean_extremity_penalty_on_balanced']} "
              f"(subtracted from balanced score only)")
        print(f"  balanced won argmax       : {d['balanced_argmax_count']}/{d['n']}\n")

    # 結構性說明：三軸加法項數不對等。
    print("STRUCTURAL NOTE — projection axes are not symmetric:")
    print("  s_agg = impulsiveness + assertiveness − risk_aversion        (2 pos, 1 neg)")
    print("  s_def = risk_aversion + suspicion + endurance + stability    (4 pos)  ← inflated")
    print("  s_bal = optimism + curiosity − mean|all 9|                   (2 pos − penalty) ← deflated")
    print("  → balanced is handicapped BY CONSTRUCTION before any data; the extremity penalty")
    print("    grows with how 'opinionated' a will is, and SBERT will-vectors are rarely flat.\n")

    json.dump({"diagnostics": diags}, open("reports/ecology/coverage_9d.json", "w"), indent=2)

    # PCA 2D 散點。
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        X = np.array(eco + hum)
        labels = (["eco"] * len(eco)) + (["hum"] * len(hum))
        arch = [int(np.argmax(personality_to_archetype_soft(list(v)))) for v in X]
        Xc = X - X.mean(0)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        P = Xc @ Vt[:2].T
        colors = {0: "C3", 1: "C0", 2: "C2"}   # agg red, def blue, bal green
        fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
        for k in range(3):
            m = [i for i in range(len(X)) if arch[i] == k]
            ax[0].scatter(P[m, 0], P[m, 1], s=22, c=colors[k], label=ARCHETYPES[k], alpha=0.7,
                          edgecolors="none")
        ax[0].set_title(f"9D personalities → PCA 2D, colored by ecology archetype argmax\n"
                        f"(n={len(X)}: {len(eco)} ecology + {len(hum)} p7h human)", fontsize=9)
        ax[0].set_xlabel(f"PC1 ({S[0]**2/(S**2).sum()*100:.0f}%)")
        ax[0].set_ylabel(f"PC2 ({S[1]**2/(S**2).sum()*100:.0f}%)"); ax[0].legend(fontsize=8)

        # 右圖：三原始分數分佈，凸顯 balanced 系統性偏低。
        sc = np.array([proj_scores(list(v)) for v in X])
        ax[1].boxplot([sc[:, 0], sc[:, 1], sc[:, 2]], labels=ARCHETYPES)
        ax[1].axhline(0, color="k", lw=0.5, ls="--")
        ax[1].set_title("raw projection scores (pre-softmax)\nballanced sits lowest → rarely wins argmax",
                        fontsize=9)
        ax[1].set_ylabel("score")
        fig.tight_layout(); fig.savefig("reports/ecology/coverage_9d.png", dpi=130)
        print("saved reports/ecology/coverage_9d.png")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
