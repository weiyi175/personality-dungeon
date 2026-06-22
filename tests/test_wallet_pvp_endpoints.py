"""端點整合測試（TestClient = in-process ASGI，無 listening socket）。
涵蓋 (ii) 經濟接線：/wallet、/wallet/credit、/pvp/challenge 門票扣款 + F2 分離。"""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

import api.pvp_manager as pm
import api.server as srv
import api.wallet_manager as wm


def fresh(start=100, ticket=10) -> TestClient:
    """重設 singletons 為已知初值（不跑 lifespan → 不從 disk 覆蓋）。"""
    wm._manager = wm.WalletManager(wm.WalletParams(starting_balance=start, ticket_cost=ticket))
    pm._manager = pm.PvpManager()
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


def _reset_ecology():
    import api.ecology_tracker as et
    et._tracker = et.EcologyTracker()


def test_ecology_submit_credits_real_player():
    """真人 live 提交（有 session_id + 冒險 outcome）→ 生態 coins 進錢包（2026-06-23 修正後）。"""
    _reset_ecology()
    c = fresh(start=0)
    nine = [0.5, 0.5, 0.0, -0.3, 0.0, 0.0, 0.0, 0.0, 0.0]
    r = c.post("/ecology/submit", json={
        "personality_9d": nine, "run_id": "sess-x", "session_id": "sess-x",
        "outcome": {"rounds_survived": 120},
    }).json()
    assert r["coins"] > 0
    assert r["balance"] == r["coins"]                  # 從 0 起、credit 了 coins
    assert c.get("/wallet").json()["balance"] == r["coins"]


def test_ecology_submit_artifact_not_credited():
    """無 session/outcome 的 smoke 提交 → 不 credit（不是真玩家）。"""
    _reset_ecology()
    c = fresh(start=0)
    nine = [0.5, 0.5, 0.0, -0.3, 0.0, 0.0, 0.0, 0.0, 0.0]
    r = c.post("/ecology/submit", json={
        "personality_9d": nine, "run_id": "", "session_id": "", "outcome": {},
    }).json()
    assert "balance" not in r                           # credit 分支未觸發
    assert c.get("/wallet").json()["balance"] == 0
