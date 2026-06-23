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
    defense_level: int = 0   # Increment 3：archetype-agnostic 防禦層（F3，花 coin 升；防守落敗時 damp 移轉）
    is_player: bool = False  # 玩家自有地牢（deploy 產；不可被自己挑戰）


@dataclass
class PvpParams:
    stake: int = 25            # 每場 Rank 賭注（零和移轉基底）
    seed_rank: int = 1000
    defense_cost: int = 50     # Increment 3：每升一級防禦的 coin 成本（archetype-agnostic，F3）
    defense_reduction: int = 5 # 每級防禦在「防守落敗」時減少的移轉量（F3，與派系無關）


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
        self._player_rank: int = self.params.seed_rank   # 單一本地玩家
        self._player_dungeon_id: str | None = None       # 玩家自有地牢（deploy 後設）
        self._history: list[dict] = []

    # ── 戰鬥核心（零和 + 防禦）─────────────────────────────────────────────────
    def _resolve(self, attacker_faction: str, defender_faction: str,
                 defender_defense: int) -> tuple[bool, int]:
        """3-RPS 判勝負 → (attacker_wins, transfer)。transfer＝零和移轉量（loser→winner）。
        防禦只在**防守方落敗**時 damp 移轉（攻擊方的防禦不生效；F3：只依 level、與派系無關）。"""
        if beats(attacker_faction, defender_faction):
            attacker_wins = True
        else:                       # 被剋 或 同型鏡像 → 守方勝
            attacker_wins = False
        base = self.params.stake
        if attacker_wins:           # 守方落敗 → 防禦減免移轉
            transfer = max(0, base - defender_defense * self.params.defense_reduction)
        else:                       # 攻方落敗 → 全額（防禦不護攻擊）
            transfer = base
        return attacker_wins, transfer

    def _sync_player_dungeon(self) -> None:
        """玩家地牢 Rank 鏡像玩家 Rank（顯示一致）。"""
        if self._player_dungeon_id and self._player_dungeon_id in self._dungeons:
            self._dungeons[self._player_dungeon_id].rank = self._player_rank

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
                "defense_level": d.defense_level,
                "is_player": d.is_player,
            }
            for d in self._dungeons.values()
        ]
        return {"dungeons": dungeons, "your_rank": self._player_rank,
                "player_dungeon_id": self._player_dungeon_id}

    # ── 寫 ────────────────────────────────────────────────────────────────────
    def deploy(self, faction: str) -> dict:
        """玩家部署自有地牢（F1：自由選派系、零讀 will）。再次呼叫＝改部署。"""
        self._seed()
        if faction not in FACTIONS:
            raise ValueError(f"unknown faction: {faction!r}")
        if self._player_dungeon_id and self._player_dungeon_id in self._dungeons:
            self._dungeons[self._player_dungeon_id].deployed_faction = faction
        else:
            did = uuid.uuid4().hex[:8]
            self._dungeons[did] = Dungeon(
                id=did, owner="你的地牢", owner_faction=faction,
                deployed_faction=faction, rank=self._player_rank, is_player=True,
            )
            self._player_dungeon_id = did
        d = self._dungeons[self._player_dungeon_id]
        return {"dungeon_id": d.id, "deployed_faction": faction,
                "deployed_zh": FACTION_ZH[faction], "defense_level": d.defense_level}

    def upgrade_defense(self) -> dict:
        """升一級防禦（archetype-agnostic，F3）。coin 由 server 層先 debit；此處只升 level。"""
        if not self._player_dungeon_id or self._player_dungeon_id not in self._dungeons:
            raise ValueError("deploy a dungeon first")
        d = self._dungeons[self._player_dungeon_id]
        d.defense_level += 1
        return {"dungeon_id": d.id, "defense_level": d.defense_level,
                "cost": self.params.defense_cost}

    def challenge(self, challenger_faction: str, dungeon_id: str) -> dict:
        """玩家挑戰 NPC 地牢：零和——贏家 +transfer / 輸家 −transfer（雙方 Rank 雙向移轉）。
        守方（地牢）落敗時其 defense_level 減免 transfer（F3）。不可挑戰自己的地牢。"""
        self._seed()
        if challenger_faction not in FACTIONS:
            raise ValueError(f"unknown challenger_faction: {challenger_faction!r}")
        d = self._dungeons.get(dungeon_id)
        if d is None:
            raise KeyError(dungeon_id)
        if d.is_player:
            raise ValueError("cannot challenge your own dungeon")

        win, transfer = self._resolve(challenger_faction, d.deployed_faction, d.defense_level)
        delta = transfer if win else -transfer
        self._player_rank = max(0, self._player_rank + delta)
        d.rank = max(0, d.rank - delta)          # 零和：對手反向移轉
        self._sync_player_dungeon()

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
            "dungeon_rank_delta": -delta,        # 零和對手側
            "your_rank": self._player_rank,
            "dungeon_rank": d.rank,
            "explain": explain,
        }
        self._history.append({**result, "kind": "challenge", "dungeon_id": dungeon_id,
                              "ts": time.time()})
        return result

    def raid(self, raider_dungeon_id: str) -> dict:
        """NPC 地牢來犯你的地牢（你當防守方）：零和 + 你的 defense_level 在落敗時減免損失。
        防禦 sink 的價值在此兌現。需先 deploy。"""
        self._seed()
        if not self._player_dungeon_id or self._player_dungeon_id not in self._dungeons:
            raise ValueError("deploy a dungeon first")
        pd = self._dungeons[self._player_dungeon_id]
        r = self._dungeons.get(raider_dungeon_id)
        if r is None:
            raise KeyError(raider_dungeon_id)
        if r.is_player:
            raise ValueError("raider must be an NPC dungeon")

        # 來犯方用其 owner_faction 攻；你以 deployed + defense 守。
        attacker_wins, transfer = self._resolve(r.owner_faction, pd.deployed_faction,
                                                pd.defense_level)
        held = not attacker_wins
        delta = -transfer if attacker_wins else transfer    # 你的 Rank 變化
        self._player_rank = max(0, self._player_rank + delta)
        r.rank = max(0, r.rank - delta)                      # 零和
        self._sync_player_dungeon()

        rf, df = FACTION_ZH[r.owner_faction], FACTION_ZH[pd.deployed_faction]
        if held:
            explain = "你的「%s」守下了「%s」的來犯 → 守住！" % (df, rf)
        else:
            explain = "「%s」攻破了你的「%s」（防禦減免 %d）→ 失守。" % (
                rf, df, pd.defense_level * self.params.defense_reduction)

        result = {
            "held": held,
            "raider": r.owner,
            "raider_faction": r.owner_faction,
            "your_deployed": pd.deployed_faction,
            "defense_level": pd.defense_level,
            "rank_delta": delta,
            "your_rank": self._player_rank,
            "explain": explain,
        }
        self._history.append({**result, "kind": "raid", "raider_id": raider_dungeon_id,
                              "ts": time.time()})
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
                "player_dungeon_id": self._player_dungeon_id,
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
        self._player_dungeon_id = data.get("player_dungeon_id")
        self._history = data.get("history", [])
        return True


_manager: PvpManager | None = None


def get_manager() -> PvpManager:
    """Process-wide singleton（與 ecology get_tracker 同模式）。"""
    global _manager
    if _manager is None:
        _manager = PvpManager()
    return _manager
