"""PVP / 地牢挑戰 — 最小 combat loop（Increment 1）。

3 派系 type-chart（authored 剋制表 M）。玩家的地牢「剋制主人」（Model 2：部署
counter(owner)），挑戰者帶剋制派系鑽破它。目的＝驗證 authored type-chart 的
counter-play 是否有趣。**H_counter 已證偽 → M 不可測、只能 authored + playtest**
（見 地牢counter-policy_L0L1介面_規劃_v1.md §1/§1b）。

v1 範圍（刻意最小）：種子合成地牢（solo 可測）、Rank vs house（非零和）、顯示
deployed 派系。真玩家地牢 / 零和 Rank / 防禦收入 / 金幣抵銷 = Increment 2。

復用：派系＝生態的 3 原型（will 9D → archetype，`ecology_tracker.personality_to_archetype_soft`
已端到端）。持久化仿 `ecology_tracker`（singleton + JSON，停-改-重啟）。
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

# 復用生態的 3 派系（aggressive=莽 / defensive=穩 / balanced=野）。
FACTIONS: list[str] = ["aggressive", "defensive", "balanced"]
FACTION_ZH: dict[str, str] = {"aggressive": "攻擊", "defensive": "防守", "balanced": "平衡"}

# ── Authored 剋制表 M（3-RPS，provisional，playtest 再 author）─────────────────
# 穩剋莽 / 莽剋野 / 野剋穩 = defensive ▸ aggressive ▸ balanced ▸ defensive。
# 穩剋莽 是唯一資料背書邊（recklessness→survival ρ=−0.98）；野剋穩 是 authored
# 反同質化 fiction。每派恰剋一、被剋一（純 3-RPS，無平手 except 同派鏡像）。
_BEATS: dict[str, str] = {
    "defensive": "aggressive",   # 穩剋莽
    "aggressive": "balanced",    # 莽剋野
    "balanced": "defensive",     # 野剋穩
}


def beats(a: str, b: str) -> bool:
    """派系 a 是否剋 b（同派或未知 → False）。"""
    return _BEATS.get(a) == b


def counter(f: str) -> str:
    """剋制 f 的派系（_BEATS 反查；3-RPS 下唯一）。"""
    for k, v in _BEATS.items():
        if v == f:
            return k
    raise ValueError(f"unknown faction: {f}")


@dataclass
class Dungeon:
    id: str
    owner: str
    owner_faction: str       # 主人 playstyle（v1 不洩漏給挑戰者；scout = v2）
    deployed_faction: str    # 地牢部署 = counter(owner_faction)（剋制主人）
    rank: int = 1000


@dataclass
class PvpParams:
    stake: int = 25          # 每場 Rank 賭注（v1 vs house）
    seed_rank: int = 1000


# 種子合成地牢（solo 可測；三派齊全 + 兩個重複「熱門」派，讓挑戰有選擇）。
_SEED_OWNERS: list[tuple[str, str, int]] = [
    ("夢魘騎士", "aggressive", 1080),
    ("石牆守衛", "defensive", 1020),
    ("風行學者", "balanced", 940),
    ("烈焰狂徒", "aggressive", 1140),
    ("古井隱者", "defensive", 880),
]


class PvpManager:
    """記憶體 singleton：種子地牢 + 單一本地玩家 Rank。save/load 仿 ecology_tracker。"""

    def __init__(self, params: PvpParams | None = None) -> None:
        self.params = params or PvpParams()
        self._dungeons: dict[str, Dungeon] = {}
        self._player_rank: int = self.params.seed_rank   # v1 單一本地玩家
        self._history: list[dict] = []

    def _seed(self) -> None:
        if self._dungeons:
            return
        for owner, fac, rank in _SEED_OWNERS:
            did = uuid.uuid4().hex[:8]
            self._dungeons[did] = Dungeon(
                id=did, owner=owner, owner_faction=fac,
                deployed_faction=counter(fac), rank=rank,
            )

    # ── 讀 ────────────────────────────────────────────────────────────────────
    def list_dungeons(self) -> dict:
        """可挑戰地牢清單。顯示 deployed 派系（1-hop 可讀：帶剋制它的派系來）。
        不洩漏 owner_faction（scout = v2）。"""
        self._seed()
        dungeons = [
            {
                "id": d.id,
                "owner": d.owner,
                "deployed_faction": d.deployed_faction,
                "deployed_zh": FACTION_ZH[d.deployed_faction],
                "counter_faction": counter(d.deployed_faction),     # 帶這個來剋它
                "counter_zh": FACTION_ZH[counter(d.deployed_faction)],
                "rank": d.rank,
            }
            for d in self._dungeons.values()
        ]
        return {"dungeons": dungeons, "your_rank": self._player_rank}

    # ── 寫 ────────────────────────────────────────────────────────────────────
    def challenge(self, challenger_faction: str, dungeon_id: str) -> dict:
        """挑戰：challenger 派系 vs 地牢 deployed 派系 → beats(a, d) 判勝負。
        v1 vs house：勝 +stake / 敗 −stake（只動玩家 Rank；地牢 Rank 靜態 = 顯示用）。"""
        self._seed()
        if challenger_faction not in FACTIONS:
            raise ValueError(f"unknown challenger_faction: {challenger_faction!r}")
        d = self._dungeons.get(dungeon_id)
        if d is None:
            raise KeyError(dungeon_id)

        win = beats(challenger_faction, d.deployed_faction)
        delta = self.params.stake if win else -self.params.stake
        self._player_rank = max(0, self._player_rank + delta)

        cf, df = FACTION_ZH[challenger_faction], FACTION_ZH[d.deployed_faction]
        if win:
            explain = "你的「%s」剋制了地牢部署的「%s」→ 攻破！" % (cf, df)
        elif beats(d.deployed_faction, challenger_faction):
            explain = "地牢部署的「%s」剋制了你的「%s」→ 被守住。" % (df, cf)
        else:  # 同派鏡像（無剋制）→ 判守方
            explain = "「%s」對上同型「%s」→ 鏡像僵持，守方勝。" % (cf, df)

        result = {
            "win": win,
            "your_faction": challenger_faction,
            "dungeon_owner": d.owner,
            "dungeon_deployed": d.deployed_faction,
            "rank_delta": delta,
            "your_rank": self._player_rank,
            "dungeon_rank": d.rank,
            "explain": explain,
        }
        self._history.append({**result, "dungeon_id": dungeon_id, "ts": time.time()})
        return result

    # ── 持久化（仿 ecology_tracker：記憶體 singleton + JSON，停-改-重啟）─────────
    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "pvp_state.json"
        with open(path, "w") as f:
            json.dump({
                "params": asdict(self.params),
                "dungeons": [asdict(d) for d in self._dungeons.values()],
                "player_rank": self._player_rank,
                "history": self._history[-200:],
            }, f, ensure_ascii=False, indent=2)
        return path

    def load(self, out_dir: str | Path) -> bool:
        path = Path(out_dir) / "pvp_state.json"
        if not path.exists():
            return False
        with open(path) as f:
            data = json.load(f)
        if "params" in data:
            pf = {k for k in PvpParams.__dataclass_fields__}
            self.params = PvpParams(**{k: v for k, v in data["params"].items() if k in pf})
        df = {k for k in Dungeon.__dataclass_fields__}
        self._dungeons = {
            d["id"]: Dungeon(**{k: v for k, v in d.items() if k in df})
            for d in data.get("dungeons", [])
        }
        self._player_rank = int(data.get("player_rank", self.params.seed_rank))
        self._history = data.get("history", [])
        return True


_manager: PvpManager | None = None


def get_manager() -> PvpManager:
    """Process-wide singleton（與 ecology get_tracker 同模式）。"""
    global _manager
    if _manager is None:
        _manager = PvpManager()
    return _manager
