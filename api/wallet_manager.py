"""玩家錢包 — Increment 2 經濟 sink 地基（R3/C，採 (ii) 單一幣）。

持久錢包：單一本地玩家 `balance` + **append-only ledger**（使用者反覆強調的「累加記帳」語意——
每筆 source 各自計算後累加、每筆 sink 各自扣，ledger 誠實拆解來源）。

- source（credit，archetype-agnostic 累加）：生態多樣性分、存活。
- sink（debit）：PvP 挑戰門票。**防禦升級待真玩家地牢**（game-spec §7 延後）。
- **F2**：coin 與 Rank **不跨帳**——門票是固定 coin 扣款、與 Rank 結算分離。
- **S1**：每筆 credit/debit 獨立落帳（per-event 序列化），不批次。
- save/load 仿 `pvp_manager`（記憶體 singleton + JSON，停-改-重啟）。

設計見 `地牢經濟_R3C隔離_規劃_v1.md`。⚠ 數值（starting_balance/ticket_cost）為 provisional，
待研究軌 g* + 均衡支付校準（game-spec §3/§5）。
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class WalletParams:
    starting_balance: int = 100   # dogfood 種子（provisional；讓 sink 立即可演示）
    ticket_cost: int = 10         # PvP 挑戰門票（provisional；modest ≪ 單場所得，F6/§3 待校準）


class InsufficientFunds(ValueError):
    """餘額不足以支付 sink（不可使 balance < 0）。"""


class WalletManager:
    """記憶體 singleton：單一本地玩家錢包（v1）。balance + append-only ledger。"""

    def __init__(self, params: WalletParams | None = None) -> None:
        self.params = params or WalletParams()
        self._balance: int = self.params.starting_balance
        self._ledger: list[dict] = []

    # ── 讀 ────────────────────────────────────────────────────────────────────
    def balance(self) -> int:
        return self._balance

    def state(self) -> dict:
        """錢包現況 + 近期帳目（給前端顯示）。"""
        return {
            "balance": self._balance,
            "ticket_cost": self.params.ticket_cost,
            "ledger": self._ledger[-50:],
        }

    # ── 寫（累加記帳；每筆獨立落帳）─────────────────────────────────────────────
    def credit(self, source: str, amount: int, note: str = "") -> dict:
        """加幣：各 source 各自計算後**累加**進 balance（生態 / 存活 …）。"""
        amount = int(amount)
        if amount < 0:
            raise ValueError(f"credit amount must be ≥0: {amount}")
        self._balance += amount
        return self._record("credit", source, amount, note)

    def debit(self, sink: str, amount: int, note: str = "") -> dict:
        """扣幣：sink（門票 …）。餘額不足 → InsufficientFunds，**balance 不會 <0**。"""
        amount = int(amount)
        if amount < 0:
            raise ValueError(f"debit amount must be ≥0: {amount}")
        if amount > self._balance:
            raise InsufficientFunds(f"需 {amount} coins，餘額 {self._balance}（{sink}）")
        self._balance -= amount
        return self._record("debit", sink, amount, note)

    def _record(self, kind: str, channel: str, amount: int, note: str) -> dict:
        entry = {"ts": time.time(), "kind": kind, "channel": channel,
                 "amount": amount, "balance_after": self._balance, "note": note}
        self._ledger.append(entry)
        return entry

    # ── 持久化（仿 pvp_manager：記憶體 singleton + JSON，停-改-重啟）───────────────
    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "wallet_state.json"
        with open(path, "w") as f:
            json.dump({
                "params": asdict(self.params),
                "balance": self._balance,
                "ledger": self._ledger[-500:],
            }, f, ensure_ascii=False, indent=2)
        return path

    def load(self, out_dir: str | Path) -> bool:
        path = Path(out_dir) / "wallet_state.json"
        if not path.exists():
            return False
        with open(path) as f:
            data = json.load(f)
        if "params" in data:
            pf = {k for k in WalletParams.__dataclass_fields__}
            self.params = WalletParams(**{k: v for k, v in data["params"].items() if k in pf})
        self._balance = int(data.get("balance", self.params.starting_balance))
        self._ledger = data.get("ledger", [])
        return True


_manager: WalletManager | None = None


def get_manager() -> WalletManager:
    """Process-wide singleton（與 pvp get_manager / ecology get_tracker 同模式）。"""
    global _manager
    if _manager is None:
        _manager = WalletManager()
    return _manager
