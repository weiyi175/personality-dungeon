"""PVP manager（最小 combat loop）測試：authored 3-RPS M、剋制主人部署、
挑戰結算、Rank delta、save/load。見 api/pvp_manager.py。"""
from __future__ import annotations

from api.pvp_manager import (
    FACTIONS,
    Dungeon,
    PvpManager,
    PvpParams,
    beats,
    counter,
)


# ── M：3-RPS 結構（authored）──────────────────────────────────────────────────
def test_rps_cycle_each_beats_one_loses_one():
    """純 3-RPS：每派恰剋一個、被一個剋；無派系剋自己。"""
    for f in FACTIONS:
        won = [g for g in FACTIONS if beats(f, g)]
        lost = [g for g in FACTIONS if beats(g, f)]
        assert won == [g for g in FACTIONS if g != f and beats(f, g)]
        assert len(won) == 1, f"{f} 應恰剋 1 個，得 {won}"
        assert len(lost) == 1, f"{f} 應恰被 1 個剋，得 {lost}"
        assert not beats(f, f), f"{f} 不應剋自己（鏡像非剋制）"


def test_authored_edges():
    """鎖定的 authored 邊：穩剋莽 / 莽剋野 / 野剋穩。"""
    assert beats("defensive", "aggressive")   # 穩剋莽
    assert beats("aggressive", "balanced")    # 莽剋野
    assert beats("balanced", "defensive")     # 野剋穩
    assert not beats("aggressive", "defensive")  # 反向不成立


def test_counter_is_inverse_of_beats():
    """counter(f) 必剋 f；且 3-RPS 下唯一。"""
    for f in FACTIONS:
        assert beats(counter(f), f)


# ── 地牢「剋制主人」（Model 2）──────────────────────────────────────────────
def test_dungeon_deploys_counter_of_owner():
    """種子地牢的 deployed_faction = counter(owner_faction)。"""
    mgr = PvpManager()
    listing = mgr.list_dungeons()
    assert listing["dungeons"], "應有種子地牢"
    # 用內部 dict 比對 owner_faction → deployed
    for d in mgr._dungeons.values():
        assert d.deployed_faction == counter(d.owner_faction)


def test_counter_hint_beats_deployed():
    """清單給的 counter_faction 帶去打 deployed，必勝（1-hop 可讀）。"""
    mgr = PvpManager()
    for d in mgr.list_dungeons()["dungeons"]:
        assert beats(d["counter_faction"], d["deployed_faction"])


# ── 挑戰結算 ─────────────────────────────────────────────────────────────────
def _first_dungeon(mgr: PvpManager) -> dict:
    return mgr.list_dungeons()["dungeons"][0]


def test_challenge_win_with_counter():
    mgr = PvpManager()
    d = _first_dungeon(mgr)
    r0 = mgr._player_rank
    res = mgr.challenge(d["counter_faction"], d["id"])
    assert res["win"] is True
    assert res["rank_delta"] == mgr.params.stake
    assert res["your_rank"] == r0 + mgr.params.stake


def test_challenge_lose_with_wrong_faction():
    mgr = PvpManager()
    d = _first_dungeon(mgr)
    # 帶「被 deployed 剋」的派系 → 必敗
    losing = counter(d["counter_faction"])  # = 被 deployed 剋的那個
    assert beats(d["deployed_faction"], losing)
    r0 = mgr._player_rank
    res = mgr.challenge(losing, d["id"])
    assert res["win"] is False
    assert res["rank_delta"] == -mgr.params.stake
    assert res["your_rank"] == r0 - mgr.params.stake


def test_challenge_mirror_is_loss():
    """帶與 deployed 同派 → 鏡像僵持判守方（玩家敗）。"""
    mgr = PvpManager()
    d = _first_dungeon(mgr)
    res = mgr.challenge(d["deployed_faction"], d["id"])
    assert res["win"] is False
    assert res["rank_delta"] == -mgr.params.stake


def test_rank_floor_at_zero():
    mgr = PvpManager(PvpParams(stake=10000, seed_rank=50))
    d = _first_dungeon(mgr)
    losing = counter(d["counter_faction"])
    res = mgr.challenge(losing, d["id"])
    assert res["your_rank"] == 0  # 不破 0


def test_challenge_unknown_faction_raises():
    mgr = PvpManager()
    d = _first_dungeon(mgr)
    try:
        mgr.challenge("wizard", d["id"])
        assert False, "未知派系應 raise"
    except ValueError:
        pass


def test_challenge_unknown_dungeon_raises():
    mgr = PvpManager()
    try:
        mgr.challenge("aggressive", "nope")
        assert False, "未知地牢應 raise"
    except KeyError:
        pass


# ── 持久化 ───────────────────────────────────────────────────────────────────
def test_save_load_roundtrip(tmp_path):
    mgr = PvpManager()
    d = _first_dungeon(mgr)
    mgr.challenge(d["counter_faction"], d["id"])  # 改 player_rank + history
    mgr.save(tmp_path)

    mgr2 = PvpManager()
    assert mgr2.load(tmp_path) is True
    assert mgr2._player_rank == mgr._player_rank
    assert set(mgr2._dungeons) == set(mgr._dungeons)
    # 地牢身分一致
    for did, dg in mgr._dungeons.items():
        assert mgr2._dungeons[did].deployed_faction == dg.deployed_faction
        assert mgr2._dungeons[did].owner_faction == dg.owner_faction


def test_load_missing_returns_false(tmp_path):
    assert PvpManager().load(tmp_path) is False
