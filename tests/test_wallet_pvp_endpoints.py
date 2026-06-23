"""端點整合測試（TestClient = in-process ASGI，無 listening socket）。
涵蓋 (ii) 經濟接線：/wallet、/wallet/credit、/pvp/challenge 門票扣款 + F2 分離。"""
from __future__ import annotations

import tempfile

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

import api.ecology_tracker as et
import api.pvp_manager as pm
import api.server as srv
import api.wallet_manager as wm


def fresh(start=100, ticket=10) -> TestClient:
    """重設 singletons 為已知初值（不跑 lifespan → 不從 disk 覆蓋）。

    並把所有 save OUT_DIR 改指向 tmp——否則端點測試的 _ecology_save 等會把**生產資料**
    （reports/ecology/ecology_state.json 真人提交）覆蓋成測試的空 tracker（2026-06-23 踩過）。
    """
    wm._manager = wm.WalletManager(wm.WalletParams(starting_balance=start, ticket_cost=ticket))
    pm._manager = pm.PvpManager()
    et._tracker = et.EcologyTracker()
    td = tempfile.mkdtemp(prefix="pd_test_stores_")
    srv.ECOLOGY_OUT_DIR = srv.WALLET_OUT_DIR = srv.PVP_OUT_DIR = td
    return TestClient(srv.app)


def _a_dungeon(c):
    d = c.get("/pvp/dungeons").json()["dungeons"][0]
    return d["id"], d["counter_faction"]   # counter_faction 帶來 = 必勝


def test_wallet_get():
    j = fresh().get("/wallet").json()
    assert j["balance"] == 100 and j["ticket_cost"] == 10


def test_challenge_charges_ticket_and_separates_rank():
    c = fresh(start=100, ticket=10)
    did, cf = _a_dungeon(c)
    j = c.post("/pvp/challenge", json={"challenger_faction": cf, "dungeon_id": did}).json()
    assert j["win"] is True
    assert j["ticket_charged"] == 10
    assert j["balance"] == 90          # 門票扣 10
    assert j["rank_delta"] == 25       # Rank 結算獨立、未被 coin 觸碰（F2）


def test_challenge_insufficient_funds_402():
    c = fresh(start=5, ticket=10)
    did, cf = _a_dungeon(c)
    r = c.post("/pvp/challenge", json={"challenger_faction": cf, "dungeon_id": did})
    assert r.status_code == 402
    assert c.get("/wallet").json()["balance"] == 5    # 沒被扣


def test_challenge_bad_dungeon_404_refunds_ticket():
    c = fresh(start=100, ticket=10)
    r = c.post("/pvp/challenge", json={"challenger_faction": "aggressive", "dungeon_id": "nope"})
    assert r.status_code == 404
    assert c.get("/wallet").json()["balance"] == 100   # 門票已退


def test_wallet_credit_survival_ok():
    c = fresh(start=0)
    j = c.post("/wallet/credit", json={"source": "survival", "amount": 60}).json()
    assert j["balance"] == 60


def test_wallet_credit_rejects_non_whitelisted_source():
    c = fresh(start=0)
    r = c.post("/wallet/credit", json={"source": "ecology", "amount": 9999})
    assert r.status_code == 422        # 生態是後端權威 credit，不收前端此路
    assert c.get("/wallet").json()["balance"] == 0


def test_ecology_submit_credits_both_sources():
    """雙經濟 Loop α：真人 live 提交 → 生態幣 + 存活幣**兩源**一次入錢包（2026-06-23）。"""
    c = fresh(start=0)
    nine = [0.5, 0.5, 0.0, -0.3, 0.0, 0.0, 0.0, 0.0, 0.0]
    r = c.post("/ecology/submit", json={
        "personality_9d": nine, "run_id": "sess-x", "session_id": "sess-x",
        "outcome": {"rounds_survived": 120},
    }).json()
    assert r["coins"] > 0
    assert r["survival_coins"] == round(0.3 * 120)     # 存活幣 = rate × rounds = 36
    assert r["balance"] == r["coins"] + r["survival_coins"]   # 兩源累加（單一幣 ii）
    assert c.get("/wallet").json()["balance"] == r["coins"] + r["survival_coins"]


def test_ecology_submit_survival_zero_when_no_rounds():
    """outcome 無 rounds_survived → 存活幣 0（仍可有生態幣）。"""
    c = fresh(start=0)
    nine = [0.5, 0.5, 0.0, -0.3, 0.0, 0.0, 0.0, 0.0, 0.0]
    r = c.post("/ecology/submit", json={
        "personality_9d": nine, "session_id": "s1", "outcome": {"cycle": 0},
    }).json()
    assert r["survival_coins"] == 0
    assert r["balance"] == r["coins"]


def test_ecology_submit_artifact_not_credited():
    """無 session/outcome 的 smoke 提交 → 不 credit（不是真玩家）。"""
    c = fresh(start=0)
    nine = [0.5, 0.5, 0.0, -0.3, 0.0, 0.0, 0.0, 0.0, 0.0]
    r = c.post("/ecology/submit", json={
        "personality_9d": nine, "run_id": "", "session_id": "", "outcome": {},
    }).json()
    assert "balance" not in r                           # credit 分支未觸發
    assert c.get("/wallet").json()["balance"] == 0


def test_ecology_submit_stores_seen_scarcity_and_monitor():
    """乙：seen_scarcity 落檔 + /ecology/scarcity_variation monitor 可讀。"""
    c = fresh(start=0)
    nine = [0.5, 0.5, 0.0, -0.3, 0.0, 0.0, 0.0, 0.0, 0.0]
    c.post("/ecology/submit", json={
        "personality_9d": nine, "session_id": "s1", "outcome": {"rounds_survived": 50},
        "seen_scarcity": [0.6, 0.3, 0.1],
    })
    rec = et._tracker._submissions[-1]
    assert rec.seen_scarcity == [0.6, 0.3, 0.1]
    mon = c.get("/ecology/scarcity_variation").json()
    assert "scarcity_std" in mon and "meets_gate" in mon


def test_ecology_submit_rejects_bad_seen_scarcity():
    c = fresh(start=0)
    nine = [0.5, 0.5, 0.0, -0.3, 0.0, 0.0, 0.0, 0.0, 0.0]
    r = c.post("/ecology/submit", json={
        "personality_9d": nine, "session_id": "s1", "outcome": {"x": 1},
        "seen_scarcity": [0.5, 0.5],          # 長度錯
    })
    assert r.status_code == 422


# ── Increment 3：deploy / 防禦升級 / raid 端點 ──────────────────────────────────

def test_deploy_endpoint():
    c = fresh()
    j = c.post("/pvp/deploy", json={"faction": "aggressive"}).json()
    assert j["deployed_faction"] == "aggressive" and j["defense_level"] == 0
    assert c.get("/pvp/dungeons").json()["player_dungeon_id"] == j["dungeon_id"]


def test_defense_upgrade_debits_wallet_and_no_rank_write():
    """防禦升級＝coin sink：先扣 coin、升 level；F2：不直接寫 Rank。"""
    c = fresh(start=100)
    c.post("/pvp/deploy", json={"faction": "defensive"})
    rank_before = c.get("/pvp/dungeons").json()["your_rank"]
    j = c.post("/pvp/defense/upgrade").json()
    assert j["defense_level"] == 1
    assert j["coin_charged"] == 50 and j["balance"] == 50      # 扣 50
    assert c.get("/pvp/dungeons").json()["your_rank"] == rank_before   # F2：Rank 未動


def test_defense_upgrade_insufficient_funds_402():
    c = fresh(start=10)
    c.post("/pvp/deploy", json={"faction": "defensive"})
    r = c.post("/pvp/defense/upgrade")
    assert r.status_code == 402
    assert c.get("/wallet").json()["balance"] == 10           # 沒被扣


def test_defense_upgrade_without_deploy_422_refunds():
    c = fresh(start=100)
    r = c.post("/pvp/defense/upgrade")                        # 還沒 deploy
    assert r.status_code == 422
    assert c.get("/wallet").json()["balance"] == 100          # coin 已退


def test_raid_endpoint_wired():
    """/pvp/raid 接線：回傳 held/rank_delta/your_rank，移轉量在 stake 界內。
    （防禦減免的 deterministic 驗證在 test_pvp_increment3，owner_faction 隱藏不從端點露。）"""
    c = fresh(start=100)
    c.post("/pvp/deploy", json={"faction": "defensive"})
    raider = next(d for d in c.get("/pvp/dungeons").json()["dungeons"] if not d["is_player"])
    r = c.post("/pvp/raid", json={"raider_dungeon_id": raider["id"]}).json()
    assert isinstance(r["held"], bool)
    assert abs(r["rank_delta"]) <= 25 and "your_rank" in r


def test_raid_requires_deploy_first():
    c = fresh()
    raider = next(d for d in c.get("/pvp/dungeons").json()["dungeons"] if not d["is_player"])
    r = c.post("/pvp/raid", json={"raider_dungeon_id": raider["id"]})
    assert r.status_code == 422        # 還沒 deploy 自己的地牢
