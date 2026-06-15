"""
Survey manager for P7-H Phase IV player tests.

Three fixed questions per the design doc:
  Q1: 主觀感知 — 人格轉變是否感到自然？ (1-10)
  Q2: 遊戲體驗 — 邊界控制是否增加樂趣？ (1-10)
  Q3: 可玩性  — 是否願意繼續遊玩？      (1-10)

Optional open text for each question.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal

import numpy as np


QUESTIONS: list[dict] = [
    # q1_naturalness / q2_fun 已移除（2026-06-13）：兩題為 P7-H 舊建構（特質漂移自然度 /
    # 接近度邊界張力），與「人格迭代」研究無關，受試者普遍看不懂。資料模型欄位保留以相容
    # 既有 20 筆回應 + legacy，僅從題庫移除不再呈現。
    {
        "id": "q3_replay",
        "text": "是否願意繼續遊玩？",
        "scale": "1 (絕對不會) — 10 (非常想繼續)",
        "type": "scale",
    },
    {
        # q4 = exploratory FLOOR-CHECK（非 confirmatory，2026-06-13 重新定位）。
        # 角色：manipulation-sufficiency。預期兩臂都落地板（floor），且此 null 為 INFORMATIVE
        # ——它把「純敘事 priming 產生不了延續感」從 n=1 自述升級成 n≈20 可引用證據，作為建構
        # 解耦式 continuity v2 的實證 mandate。pre-reg 須標此預期 null 為 informative，非 failed
        # hypothesis。reset 組也問。題目文字保留。
        "id": "q4_continuity",
        "text": "這三段人生是否感覺像同一個逐步演化的角色？",
        "scale": "1 (完全是不同角色) — 10 (明顯是同一個逐步演化的角色)",
        "type": "scale",
    },
    {
        # Manipulation-awareness debrief（開放題，放最後，避免污染前面評分）。
        # 迭代研究有 narrative priming，特別需要這道保險：若多數人答中操弄，
        # E1/q4 須打折解讀。funnel-style，不提示具體操弄內容。
        "id": "manipulation_awareness",
        "text": "你覺得這個研究在測什麼？過程中有沒有感覺到什麼是被刻意設計來影響你的？（沒有也可留空）",
        "type": "text",
    },
]


@dataclass
class SurveyResponse:
    session_id: str
    group: Literal["control", "experiment"]
    submitted_at: float = field(default_factory=time.time)
    q1_naturalness: int = 0          # 1-10
    q2_fun: int = 0
    q3_replay: int = 0
    q4_continuity: int = 0           # 1-10；0=未作答（向後相容舊 3 題客戶端）
    manipulation_awareness: str = ""  # 開放 debrief（demand-characteristics 稽核）
    q1_comment: str = ""
    q2_comment: str = ""
    q3_comment: str = ""
    overall_comments: str = ""


class SurveyManager:
    def __init__(self) -> None:
        self._responses: dict[str, SurveyResponse] = {}

    def submit(
        self,
        session_id: str,
        group: str,
        q1: int, q2: int, q3: int,
        q4: int = 0,
        q1_comment: str = "",
        q2_comment: str = "",
        q3_comment: str = "",
        overall_comments: str = "",
        manipulation_awareness: str = "",
    ) -> dict:
        # q3_replay 是唯一仍呈現的 scale 題 → 必填 1–10。
        if not (1 <= q3 <= 10):
            return {"ok": False, "error": f"q3 must be 1–10, got {q3}"}
        # q1_naturalness/q2_fun 已自題庫移除（2026-06-13）；前端不再呈現 → 送 0。
        # q4（角色延續感）：0=未作答。三者皆允許 0（未作答）或 1–10。
        for label, v in [("q1", q1), ("q2", q2), ("q4", q4)]:
            if not (0 <= v <= 10):
                return {"ok": False, "error": f"{label} must be 0–10, got {v}"}

        self._responses[session_id] = SurveyResponse(
            session_id=session_id,
            group=group,  # type: ignore[arg-type]
            q1_naturalness=q1,
            q2_fun=q2,
            q3_replay=q3,
            q4_continuity=q4,
            manipulation_awareness=manipulation_awareness[:2000],
            q1_comment=q1_comment[:500],
            q2_comment=q2_comment[:500],
            q3_comment=q3_comment[:500],
            overall_comments=overall_comments[:1000],
        )
        return {"ok": True, "session_id": session_id}

    def get_response(self, session_id: str) -> dict | None:
        r = self._responses.get(session_id)
        return asdict(r) if r else None

    def summary(self) -> dict:
        groups: dict[str, list[SurveyResponse]] = {"control": [], "experiment": []}
        for r in self._responses.values():
            groups.get(r.group, groups["control"]).append(r)

        def _stats(responses: list[SurveyResponse]) -> dict:
            if not responses:
                return {"n": 0}
            q1 = [r.q1_naturalness for r in responses]
            q2 = [r.q2_fun for r in responses]
            q3 = [r.q3_replay for r in responses]
            return {
                "n": len(responses),
                "q1_naturalness": {"mean": round(float(np.mean(q1)), 2),
                                   "std": round(float(np.std(q1)), 2)},
                "q2_fun":         {"mean": round(float(np.mean(q2)), 2),
                                   "std": round(float(np.std(q2)), 2)},
                "q3_replay":      {"mean": round(float(np.mean(q3)), 2),
                                   "std": round(float(np.std(q3)), 2)},
                "composite_ux":   round(float(np.mean(q1 + q2 + q3)), 2),
            }

        ctrl = _stats(groups["control"])
        exp = _stats(groups["experiment"])

        # UX lift: experiment composite minus control composite
        ux_lift = 0.0
        if ctrl.get("n", 0) > 0 and exp.get("n", 0) > 0:
            ux_lift = exp["composite_ux"] - ctrl["composite_ux"]

        return {
            "total_responses": len(self._responses),
            "control": ctrl,
            "experiment": exp,
            "ux_lift": round(ux_lift, 3),
            "questions": QUESTIONS,
        }

    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "p7h_survey_responses.json"
        with open(path, "w") as f:
            json.dump(
                {
                    "responses": {
                        sid: asdict(r) for sid, r in self._responses.items()
                    },
                    "summary": self.summary(),
                },
                f,
                indent=2,
            )
        return path

    def load(self, out_dir: str | Path) -> bool:
        """Restore responses from a prior save(). Returns True if a file loaded.

        Used on server startup so a restart mid-study keeps survey responses.
        """
        path = Path(out_dir) / "p7h_survey_responses.json"
        if not path.exists():
            return False
        with open(path) as f:
            data = json.load(f)
        self._responses = {
            sid: SurveyResponse(**r) for sid, r in data.get("responses", {}).items()
        }
        return True


_survey: SurveyManager | None = None


def get_survey() -> SurveyManager:
    global _survey
    if _survey is None:
        _survey = SurveyManager()
    return _survey
