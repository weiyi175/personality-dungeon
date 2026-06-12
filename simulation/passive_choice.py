"""Passive personality-driven event resolution.

"被動版"：給定一個人格向量與一個事件，由人格的 action-utility（weights·personality）
決定 AI 分身會選哪個選項，並結算成功/失敗與效果 —— 供「你的分身做了選擇 X，結果…」
的展示與「人格→選擇對應關係」的驗證使用。

與 EventLoader.process_turn 的差異：
  - 本模組是 *唯讀* 的：不修改任何 player 物件、不套用 trait/state delta、不做 decay。
  - 額外回傳每個選項的 utility 與 softmax「傾向機率」，方便 UI 顯示與分析。
  - 選中的選項與 EventLoader.choose_action 一致（argmax utility），確保行為對齊既有 runtime。

核心鏈（全部複用 EventLoader）：
    weights · personality  →  choose_action  →  final_risk  →  success_prob  →  outcome
"""

from __future__ import annotations

import math
import random
from typing import Any, Mapping, Sequence

from dungeon.event_loader import EventLoader


def _softmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    top = max(values)
    exps = [math.exp(v - top) for v in values]
    total = sum(exps) or 1.0
    return [e / total for e in exps]


def vector_to_personality(
    loader: EventLoader, vector: Sequence[float]
) -> dict[str, float]:
    """把 9D 向量（依 loader.dimensions_order）轉成 personality dict。"""
    dims = loader.dimensions_order
    if len(vector) != len(dims):
        raise ValueError(f"Expected {len(dims)} dims, got {len(vector)}")
    return {key: float(val) for key, val in zip(dims, vector)}


def resolve_passive_choice(
    loader: EventLoader,
    personality: Mapping[str, float],
    *,
    event_id: str | None = None,
    state: Mapping[str, float] | None = None,
    rng: random.Random | None = None,
    deterministic_outcome: bool = False,
) -> dict[str, Any]:
    """讓人格分身被動地面對一個事件並結算。

    Parameters
    ----------
    personality : 9D 人格 dict（依 loader.dimensions_order 的鍵）
    event_id    : 指定事件；None 則隨機抽一個
    state       : 可選的玩家狀態（stress/noise/...），影響 risk；預設全 0
    rng         : 隨機源（控制隨機抽事件與成敗擲骰）；None 則自建
    deterministic_outcome : True 時不擲骰，success = (success_prob >= 0.5)，
                            供可重現的驗證/快照使用

    Returns
    -------
    dict —— 含 event / options / chosen_action / outcome，見模組說明。
    """
    rng = rng if rng is not None else random.Random()
    event = (
        loader.get_event_template(event_id)
        if event_id is not None
        else loader.sample_event_template(rng=rng)
    )
    actions = list(event["actions"])

    # 1) 每個選項的 utility（人格對該選項的傾向）。不加隨機噪音，保持可解釋。
    utilities = [
        loader.compute_action_utility(action, personality, state=state)
        for action in actions
    ]
    lean_probs = _softmax(utilities)

    # 2) 選中的選項 = argmax utility（與 EventLoader.choose_action 一致）。
    chosen_idx = max(range(len(actions)), key=lambda i: utilities[i])
    chosen = actions[chosen_idx]

    options = [
        {
            "name": str(action["name"]),
            "name_zh": str(action.get("name_zh", action["name"])),
            "utility": round(float(utilities[i]), 6),
            "lean_prob": round(float(lean_probs[i]), 6),
            "chosen": i == chosen_idx,
        }
        for i, action in enumerate(actions)
    ]

    # 3) 結算成敗（複用 risk / success 模型）。
    event_type = str(event.get("type", ""))
    final_risk = loader.compute_final_risk(
        chosen, personality, state=state, event_type=event_type
    )
    success_prob = loader.compute_success_prob(chosen, final_risk, state=state)

    if deterministic_outcome:
        roll = None
        success = success_prob >= 0.5
    else:
        roll = rng.random()
        success = roll < success_prob

    # 4) 結果 payload（唯讀；不套用到任何 player）。
    if success:
        result_kind = "success"
        payload = dict(chosen.get("reward_effects", {}))
    else:
        failure = loader.choose_failure_outcome(
            list(chosen.get("failure_outcomes", [])), rng=rng
        )
        result_kind = str(failure.get("kind", "failure"))
        payload = dict(failure)

    type_labels_zh = loader.data.get("event_type_labels_zh", {})
    return {
        "event_id": str(event["event_id"]),
        "event_type": event_type,
        "event_type_zh": str(type_labels_zh.get(event_type, event_type)),
        "description": str(event.get("description", "")),
        "description_zh": str(event.get("description_zh", event.get("description", ""))),
        "options": options,
        "chosen_action": str(chosen["name"]),
        "chosen_action_zh": str(chosen.get("name_zh", chosen["name"])),
        "outcome": {
            "success": bool(success),
            "result_kind": result_kind,
            "final_risk": round(float(final_risk), 6),
            "success_prob": round(float(success_prob), 6),
            "roll": (round(float(roll), 6) if roll is not None else None),
            "payload": payload,
        },
    }
