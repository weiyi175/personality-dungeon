"""驗證 P7-H 真人 intrinsic archetype 分佈——並揭穿「aggressive-leaning [.48/.43/.09]」artifact。

背景（2026-06-23）：記憶曾載「54 個 clean 真人（is_human ∧ run_id 空）→ [agg .48/def .43/bal .09]、
aggressive 主導」。驗證 ecology 的 run_id 慣例坑後回頭查 P7-H，發現同類陷阱：那 54 筆「run_id 空」
全是 cycle_index=-1、無歸屬的 pilot 批；「run_id 空」反而丟掉 200 筆 = 29 個有歸屬真受試 × cycles。

本腳本重算 archetype 分佈於數種真人選法，顯示：
  • agg vs def 排序**隨選法翻轉**（只有 54-子集是 agg 主導，離群）；
  • 唯一穩健特徵 = balanced-thin（~.08–.10）；
  • 一人一筆（29 受試）≈ [.38/.52/.10] defensive 主導，與 ecology 208 筆 live 的 α 同向。

用 will_sbert_vector（個體化的 9D，非靜態 will_personality_vector）+ ecology 的原型投影。
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np

from api.ecology_tracker import ARCHETYPES, personality_to_archetype_soft

SESSIONS = "reports/experiments/p7h_real_study/p7h_player_test_sessions.json"


def _dist(group, tau: float = 0.6):
    cnt = collections.Counter()
    for s in group:
        v = s.get("will_sbert_vector")
        if not v or len(v) != 9:
            continue
        soft = personality_to_archetype_soft(v, tau=tau)
        cnt[ARCHETYPES[int(np.argmax(soft))]] += 1
    tot = sum(cnt.values()) or 1
    return {a: round(cnt[a] / tot, 3) for a in ARCHETYPES}, tot


def main(path: str = SESSIONS) -> None:
    sess = list(json.loads(Path(path).read_text())["sessions"].values())
    humans = [s for s in sess if s.get("is_human")]
    h_runempty = [s for s in humans if not s.get("run_id")]
    h_runset = [s for s in humans if s.get("run_id")]

    print(f"sessions={len(sess)}  真人={len(humans)}  "
          f"run_id空={len(h_runempty)}  run_id非空={len(h_runset)}")
    print("獨立 participant_id：",
          f"全真人={len({s.get('participant_id') for s in humans})}",
          f"｜run_id空={len({s.get('participant_id') for s in h_runempty})}",
          f"｜run_id非空={len({s.get('participant_id') for s in h_runset})}")
    print("run_id空 的 cycle_index：",
          dict(collections.Counter(s.get("cycle_index") for s in h_runempty)))

    # 一人一筆：每 participant 取 cycle_index 最小者
    byp: dict = {}
    for s in humans:
        p = s.get("participant_id")
        if p is None:
            continue
        ci = s.get("cycle_index") or 0
        if p not in byp or ci < (byp[p].get("cycle_index") or 0):
            byp[p] = s

    print("\narchetype 分佈（will_sbert argmax, tau=.6）：")
    for label, grp in [
        ("54 (記憶用的, run_id空)", h_runempty),
        ("全真人", humans),
        ("被丟的 (run_id非空)", h_runset),
        ("一人一筆 (首 cycle)", list(byp.values())),
    ]:
        dd, n = _dist(grp)
        lead = max(dd, key=dd.get)
        print(f"  {label:24} n={n:4}  {dd}  ← {lead} 主導")
    print("\n結論：balanced-thin 穩健；agg vs def 排序隨選法翻；"
          "原『aggressive-leaning』是 54-pilot artifact。")


if __name__ == "__main__":
    main()
