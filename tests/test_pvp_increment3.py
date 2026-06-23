"""Increment 3（真玩家地牢 + 零和 Rank + 防禦 sink）manager-level 測試。

戰鬥用 deterministic 3-RPS，故全部不靠隨機。對齊 §10 firewall 重審的不變式：
  零和（雙向移轉）、defense sink（F3 archetype-agnostic、防守落敗才減免）、
  F2（challenge/raid 不碰 wallet——manager 層本就無 wallet）、F4（不呼 ecology.submit）。
"""
from __future__ import annotations

import pytest

import api.pvp_manager as pm


def _mgr():
    m = pm.PvpManager()
    m._seed()
    return m


def _npc_by_faction(m, faction):
    return next(d for d in m._dungeons.values()
               if d.owner_faction == faction and not d.is_player)


def test_zero_sum_challenge_moves_both_ranks():
    m = _mgr()
    info = m.list_dungeons()["dungeons"][0]
    did, counter = info["id"], info["counter_faction"]
    before_dungeon = m._dungeons[did].rank
    before_player = m._player_rank
    r = m.challenge(counter, did)              # 帶剋制派系 → 必勝
    assert r["win"] is True
    assert r["rank_delta"] == 25 and r["dungeon_rank_delta"] == -25
    # 零和：玩家 +T、對手 −T
    assert m._player_rank == before_player + 25
    assert m._dungeons[did].rank == before_dungeon - 25
    assert r["rank_delta"] == -r["dungeon_rank_delta"]


def test_deploy_creates_player_dungeon_and_cannot_self_challenge():
    m = _mgr()
    d = m.deploy("aggressive")
    assert m._player_dungeon_id == d["dungeon_id"]
    assert m._dungeons[d["dungeon_id"]].is_player is True
    with pytest.raises(ValueError):
        m.challenge("defensive", d["dungeon_id"])     # 不可挑戰自己的地牢


def test_defense_sink_reduces_raid_loss():
    m = _mgr()
    m.deploy("defensive")                        # 你部署防守
    raider = _npc_by_faction(m, "balanced")      # balanced 剋 defensive → 來犯必破你
    # 無防禦：失守、損失 = stake 25
    r0 = m.raid(raider.id)
    assert r0["held"] is False and r0["rank_delta"] == -25
    # 升 3 級防禦（3×5=15 減免）→ 同樣失守但只損 25−15=10
    for _ in range(3):
        m.upgrade_defense()
    r1 = m.raid(raider.id)
    assert r1["held"] is False and r1["rank_delta"] == -10


def test_defense_floors_loss_at_zero():
    m = _mgr()
    m.deploy("defensive")
    raider = _npc_by_faction(m, "balanced")
    for _ in range(10):                          # 10×5=50 ≫ stake 25 → 地板 0
        m.upgrade_defense()
    r = m.raid(raider.id)
    assert r["held"] is False and r["rank_delta"] == 0


def test_raid_hold_pays_full_stake_no_defense_damp():
    m = _mgr()
    m.deploy("balanced")                         # balanced 剋 defensive
    raider = _npc_by_faction(m, "defensive")     # defensive 攻 balanced → 你守得住
    before = m._player_rank
    r = m.raid(raider.id)
    assert r["held"] is True and r["rank_delta"] == 25   # 守勝拿全額（防禦不 damp 勝方）
    assert m._player_rank == before + 25


def test_f3_defense_cost_and_effect_faction_agnostic():
    """F3：防禦成本/效果與部署派系無關（逐派相等）。"""
    losses = {}
    for fac, raider_fac in [("aggressive", "defensive"), ("defensive", "balanced"),
                            ("balanced", "aggressive")]:
        m = _mgr()
        m.deploy(fac)
        up = m.upgrade_defense()
        assert up["cost"] == pm.PvpManager().params.defense_cost   # 同價
        raider = _npc_by_faction(m, raider_fac)                    # 該派被剋 → 必失守
        losses[fac] = m.raid(raider.id)["rank_delta"]
    assert len(set(losses.values())) == 1        # 三派的失守損失相同（1 級防禦同效）


def test_f4_challenge_and_raid_do_not_touch_ecology():
    """F4：PvP 不餵生態——挑戰/raid 後生態提交數不變。"""
    import api.ecology_tracker as et
    et._tracker = et.EcologyTracker()
    n0 = len(et.get_tracker()._submissions)
    m = _mgr()
    info = m.list_dungeons()["dungeons"][0]
    m.challenge(info["counter_faction"], info["id"])
    m.deploy("defensive")
    m.raid(_npc_by_faction(m, "balanced").id)
    assert len(et.get_tracker()._submissions) == n0    # 生態完全沒被碰


def test_save_load_roundtrip_preserves_player_dungeon_and_defense(tmp_path):
    m = _mgr()
    m.deploy("aggressive")
    m.upgrade_defense()
    m.save(tmp_path)
    m2 = pm.PvpManager()
    assert m2.load(tmp_path) is True
    assert m2._player_dungeon_id == m._player_dungeon_id
    assert m2._dungeons[m2._player_dungeon_id].defense_level == 1
    assert m2._dungeons[m2._player_dungeon_id].is_player is True
