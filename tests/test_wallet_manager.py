"""WalletManager 單元測試（Increment 2 經濟 sink 地基）。"""
from __future__ import annotations

import pytest

from api.wallet_manager import InsufficientFunds, WalletManager, WalletParams


def _w(start=100, ticket=10):
    return WalletManager(WalletParams(starting_balance=start, ticket_cost=ticket))


def test_starting_balance():
    assert _w(start=100).balance() == 100


def test_credit_accumulates():
    w = _w(start=0)
    w.credit("ecology", 94)
    w.credit("survival", 60)
    assert w.balance() == 154


def test_debit_deducts():
    w = _w(start=100)
    w.debit("pvp_ticket", 10)
    assert w.balance() == 90


def test_debit_insufficient_raises_and_keeps_balance():
    w = _w(start=5)
    with pytest.raises(InsufficientFunds):
        w.debit("pvp_ticket", 10)
    assert w.balance() == 5            # balance 不變、不會 <0


def test_negative_amounts_rejected():
    w = _w()
    with pytest.raises(ValueError):
        w.credit("ecology", -1)
    with pytest.raises(ValueError):
        w.debit("pvp_ticket", -1)


def test_ledger_accumulation_equals_balance():
    w = _w(start=0)
    w.credit("ecology", 100, note="場1")
    w.credit("survival", 60)
    w.debit("pvp_ticket", 10)
    net = sum(e["amount"] if e["kind"] == "credit" else -e["amount"]
              for e in w.state()["ledger"])
    assert net == w.balance() == 150
    assert w.state()["ledger"][-1]["balance_after"] == 150


def test_ledger_records_source_and_sink_separately():
    w = _w(start=0)
    w.credit("ecology", 94)
    w.debit("pvp_ticket", 10)
    led = w.state()["ledger"]
    assert led[0]["kind"] == "credit" and led[0]["channel"] == "ecology"
    assert led[1]["kind"] == "debit" and led[1]["channel"] == "pvp_ticket"


def test_state_exposes_ticket_cost():
    assert _w(ticket=10).state()["ticket_cost"] == 10


def test_save_load_round_trip(tmp_path):
    w = _w(start=0)
    w.credit("ecology", 100)
    w.debit("pvp_ticket", 10)
    w.save(tmp_path)

    w2 = WalletManager()
    assert w2.load(tmp_path) is True
    assert w2.balance() == 90
    assert w2.params.ticket_cost == 10
    assert len(w2.state()["ledger"]) == 2


def test_load_missing_file_returns_false(tmp_path):
    assert WalletManager().load(tmp_path) is False
