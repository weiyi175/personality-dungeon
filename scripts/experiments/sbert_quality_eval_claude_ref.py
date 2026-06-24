"""Step 0 hard metrics — SBERT-9D vs Claude(Opus) reference labels on held-out 35.

Reference = Opus judgments of will_text -> 9 traits in [-1,1], labelled from the
text with a fixed rubric (impulsive/deliberate, aggressive/passive, hopeful/dark,
cautious/reckless, distrustful/trusting, persistent/quitting, chaotic/orderly,
security/change, curious/incurious). NOT the project's original LLM teacher, so
this measures "does SBERT match Opus's reading" (= proxy for the user's own
face-validity judgement), not distillation fidelity.

Reports per trait: Pearson r (direction tracking, scale-free — the user's actual
complaint), R^2 (1 - SSres/SStot vs ref variance), MAE. Plus worst-disagreement
wills. Writes claude_reference_labels.json so future fixes reuse the same ref.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from api.personality_sbert_inference import infer_personality_vector_sbert  # noqa: E402
from api.schemas import PERSONALITY_BASIS  # noqa: E402

OUT_DIR = ROOT / "reports/experiments/sbert_quality"
SHORT = ("imp", "ast", "opt", "rsk", "sus", "end", "rnd", "stb", "cur")
# trait order: impulsiveness, assertiveness, optimism, risk_aversion, suspicion,
#              endurance, randomness, stability_seeking, curiosity
REF: dict[str, list[float]] = {
    "去攻打Boss帶戰士和精靈弓箭手": [0.1, 0.6, 0.2, -0.2, -0.2, 0.4, -0.3, 0.2, 0.0],
    "向盜賊學習偷竊和腳步輕盈技能": [0.0, -0.1, 0.0, 0.1, 0.6, 0.2, -0.1, -0.1, 0.5],
    "和精靈族一起攻打醜陋的地精": [0.1, 0.6, 0.1, -0.2, -0.3, 0.3, -0.2, 0.2, 0.0],
    "和精靈族一起練習射箭並且熟練用弓": [-0.3, 0.0, 0.2, 0.2, -0.3, 0.7, -0.4, 0.2, 0.3],
    "嘗試和仙子成為友好關係並學習種植": [-0.1, -0.4, 0.4, 0.1, -0.6, 0.2, -0.2, 0.5, 0.4],
    "嘗試帶著狗進地下城警戒周圍生物是否靠近": [-0.2, 0.0, 0.0, 0.5, 0.4, 0.3, -0.3, 0.5, 0.0],
    "嘗試繪製地下城地圖並且賣給哥布林商人": [0.0, 0.0, 0.2, 0.0, 0.2, 0.3, -0.2, 0.0, 0.6],
    "好奇為什麼地上有特殊符號": [-0.1, -0.2, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.9],
    "學基本的劍術與購買適合的裝備": [-0.2, 0.1, 0.1, 0.3, 0.0, 0.4, -0.3, 0.3, 0.2],
    "學習判斷寶箱怪": [-0.2, 0.0, 0.0, 0.4, 0.5, 0.2, -0.2, 0.2, 0.4],
    "專挑弱的小怪打": [0.1, 0.2, 0.0, 0.6, 0.3, -0.1, -0.1, 0.3, -0.2],
    "屠殺地下城怪物": [0.4, 0.8, 0.0, -0.5, 0.0, 0.2, 0.1, -0.3, -0.2],
    "左手拿盾右手拿劍再找牧師和弓箭手輔助": [-0.3, 0.3, 0.1, 0.2, -0.2, 0.4, -0.4, 0.4, -0.1],
    "帶上火把照明並且帶上許多燃油瓶去攻擊樹妖": [0.2, 0.5, 0.2, -0.1, 0.2, 0.3, -0.2, 0.0, 0.1],
    "帶著大鐵鎚嘗試破壞牆壁抄捷徑": [0.6, 0.4, 0.1, -0.4, -0.1, 0.0, 0.3, -0.3, 0.2],
    "帶著手榴彈遠程攻擊哥布林群體": [0.3, 0.5, 0.1, 0.1, 0.1, 0.2, 0.0, 0.0, 0.0],
    "帶著簡易醫療包尤其是繃帶治療小傷口和止血": [-0.4, -0.2, 0.0, 0.6, -0.1, 0.3, -0.4, 0.6, -0.1],
    "強迫工匠幫我偷竊強力的劍": [0.4, 0.5, -0.1, -0.2, 0.8, 0.0, 0.0, -0.4, -0.2],
    "我要健康的活下去": [-0.4, -0.1, 0.7, 0.4, -0.2, 0.3, -0.3, 0.6, -0.2],
    "打強力怪物": [0.3, 0.7, 0.2, -0.6, 0.0, 0.3, 0.0, -0.2, 0.0],
    "找聖光牧師克制黑暗生物": [-0.2, 0.2, 0.2, 0.2, -0.1, 0.3, -0.3, 0.3, 0.1],
    "把壞掉的裝備花重金修復並組隊去探索地下城": [-0.2, 0.0, 0.2, 0.3, -0.3, 0.5, -0.3, 0.4, 0.3],
    "毆打精靈": [0.6, 0.7, -0.2, -0.3, 0.2, 0.0, 0.4, -0.4, -0.2],
    "活在盡全力做好準備事項的細心生活": [-0.7, -0.1, 0.2, 0.6, 0.0, 0.6, -0.6, 0.7, -0.1],
    "深思熟慮再做每個決定": [-0.9, -0.1, 0.0, 0.5, 0.1, 0.3, -0.6, 0.5, 0.1],
    "漫無目的探索地下城": [0.3, -0.1, 0.2, -0.1, -0.2, 0.0, 0.6, -0.4, 0.5],
    "為了守護美好的世界而進入地下城變強": [0.0, 0.3, 0.6, 0.0, -0.4, 0.6, -0.3, 0.4, 0.1],
    "破壞矮人的信賴關係搶奪高品質裝備": [0.2, 0.4, -0.3, -0.2, 0.9, -0.1, 0.0, -0.5, -0.2],
    "綁架仙子族過上種田人生": [0.3, 0.3, 0.1, -0.1, 0.6, 0.2, 0.2, 0.2, -0.2],
    "自己當坦克拿盾牌並找三個精靈射手當輸出": [-0.3, 0.3, 0.1, 0.2, -0.3, 0.5, -0.5, 0.5, -0.1],
    "與商人交流用談判技巧壓低售價": [0.0, 0.4, 0.1, 0.1, 0.2, 0.1, -0.3, 0.0, 0.1],
    "苟且偷生": [-0.2, -0.5, -0.4, 0.6, 0.1, -0.2, -0.1, 0.4, -0.4],
    "複習以前的錯誤並嘗試改善": [-0.5, -0.1, 0.2, 0.2, 0.0, 0.5, -0.4, 0.3, 0.3],
    "買最好的裝備屠殺低等小怪賺金幣": [0.1, 0.4, 0.1, 0.4, 0.3, 0.2, -0.2, 0.2, -0.3],
    "跟精靈族學習射箭": [-0.3, 0.0, 0.2, 0.1, -0.3, 0.5, -0.3, 0.2, 0.4],
}


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denom) if denom > 1e-12 else float("nan")


def main() -> None:
    wills = list(REF.keys())
    ref = np.array([REF[w] for w in wills])
    pred = np.zeros_like(ref)
    for i, w in enumerate(wills):
        vec, _ = infer_personality_vector_sbert(w)
        pred[i] = [vec[t] for t in PERSONALITY_BASIS]

    print(f"=== SBERT-9D vs Opus reference, held-out N={len(wills)} ===")
    print(f"{'trait':<18} {'pearson_r':>10} {'R2':>8} {'MAE':>7}")
    rows = {}
    for j, t in enumerate(PERSONALITY_BASIS):
        y, yh = ref[:, j], pred[:, j]
        r = pearson(y, yh)
        ss_res = float(((y - yh) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum()) or 1e-12
        r2 = 1 - ss_res / ss_tot
        mae = float(np.abs(y - yh).mean())
        rows[t] = {"pearson_r": r, "r2": r2, "mae": mae}
        print(f"{t:<18} {r:>10.3f} {r2:>8.3f} {mae:>7.3f}")
    mean_r = float(np.nanmean([rows[t]["pearson_r"] for t in PERSONALITY_BASIS]))
    print(f"{'MEAN':<18} {mean_r:>10.3f}")

    err = np.abs(ref - pred).mean(axis=1)
    worst = np.argsort(-err)[:8]
    print("\n--- worst SBERT-vs-Opus disagreements ---")
    for i in worst:
        diffs = [(PERSONALITY_BASIS[j], float(ref[i, j]), float(pred[i, j]))
                 for j in np.argsort(-np.abs(ref[i] - pred[i]))[:3]]
        ds = ", ".join(f"{n}: ref{r:+.1f} vs pred{p:+.1f}" for n, r, p in diffs)
        print(f"  [{err[i]:.2f}] {wills[i]}  ({ds})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "claude_reference_labels.json").write_text(
        json.dumps(REF, ensure_ascii=False, indent=1))
    (OUT_DIR / "eval_vs_claude_ref.json").write_text(
        json.dumps({"n": len(wills), "per_trait": rows, "mean_pearson_r": mean_r},
                   ensure_ascii=False, indent=1))
    print(f"\nwrote {OUT_DIR}/claude_reference_labels.json, eval_vs_claude_ref.json")


if __name__ == "__main__":
    main()
