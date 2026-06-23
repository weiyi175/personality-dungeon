#!/usr/bin/env python3
"""乙 pilot 看板（read-only）：一條指令看 R1→R2 gate 就緒度。

把三個**唯讀**來源串成單一儀表，降低使用者跑 R1 pilot 的摩擦（handoff §4 P0）：
  1. GET /ecology/scarcity_variation — live in-memory 稀缺變異（鐵律 1，R1 收集中是否在動）
  2. GET /wallet                     — 雙經濟循環健康（兩源賺幣 + PvP 花幣是否在流動）
  3. ecology_beta_fit on disk state  — β verdict（R2 凍結就緒度：n_real / scarcity_std / β）

純讀契約（F-safe）：
  • 只發 GET。**絕不** POST、絕不寫 production 資料（教訓見 isolate-write-targets-before-running）。
  • 後端關閉也能跑——live 區段標 [unreachable]，仍從磁碟算 β verdict。

R1→R2 gate 判準（與 ecology_beta_fit / collection_diagnostics 對齊）：
  • 識別 gate：scarcity_std ≥ 0.06        （低於 → β 與 intrinsic 共線、不可識別）
  • 設計目標：scarcity_std ≥ 0.20        （power 足）
  • 樣本下限：n_real ≥ MIN_N (=30)
  • β verdict：fit_beta == "OK"

用法：
  python scripts/experiments/ecology_pilot_dashboard.py
  python scripts/experiments/ecology_pilot_dashboard.py --host http://localhost:8001 \
      --state reports/ecology/ecology_state.json
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

# 同目錄的 β-instrument（sys.path[0] = 本檔目錄，import 即可）。
try:
    from ecology_beta_fit import (  # type: ignore
        MIN_N,
        MIN_SCARCITY_STD,
        fit_beta,
        load_real_submissions,
    )
    _BETA_OK = True
    _BETA_ERR = ""
except Exception as exc:  # noqa: BLE001  — 看板要韌性：β 區段壞掉不該炸掉整個儀表
    _BETA_OK = False
    _BETA_ERR = f"{type(exc).__name__}: {exc}"
    MIN_N, MIN_SCARCITY_STD = 30, 0.02  # fallback 常數（僅供顯示閾值）

GATE_STD = 0.06     # 識別 gate（collection_diagnostics meets_gate）
TARGET_STD = 0.20   # power 設計目標（collection_diagnostics meets_target）

_OK, _WARN, _BAD = "✓", "△", "✗"  # ✓ △ ✗


def _get_json(url: str, timeout: float):
    """GET 一個 JSON 端點。回 (data, None) 或 (None, err_str)。只讀、不寫。"""
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8")), None
    except urllib.error.URLError as exc:
        return None, f"unreachable ({exc.reason})"
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def _mark(value: float, gate: float, target: float) -> str:
    if value >= target:
        return _OK
    if value >= gate:
        return _WARN
    return _BAD


def _fmt_std(diag: dict) -> str:
    scar = diag.get("scarcity_std")
    if scar is None:
        return "—"
    return f"{scar:.4f} {_mark(scar, GATE_STD, TARGET_STD)}"


def render(host: str, state_path: str, timeout: float, lam: float | None) -> int:
    """印看板。回 exit code（0 = R2 gate 就緒，1 = 還沒）。"""
    lines: list[str] = []
    lines.append("=" * 64)
    lines.append("  乙 PILOT 看板 — R1→R2 gate 就緒度 (read-only)")
    lines.append("=" * 64)

    # ── 1. LIVE：稀缺變異（R1 收集中是否在動）─────────────────────────────────
    diag, diag_err = _get_json(f"{host}/ecology/scarcity_variation", timeout)
    lines.append("")
    lines.append("[R1] LIVE 稀缺變異  (GET /ecology/scarcity_variation)")
    live_meets_target = False
    if diag_err:
        lines.append(f"   [unreachable] {diag_err} — 後端未啟動？改看下方 on-disk β verdict")
    else:
        live_meets_target = bool(diag.get("meets_target"))
        lines.append(f"   n_real       : {diag.get('n_real', '—')}")
        lines.append(f"   scarcity_std : {_fmt_std(diag)}   "
                     f"(gate {GATE_STD} / target {TARGET_STD})")
        lines.append(f"   meets_gate   : {_OK if diag.get('meets_gate') else _BAD}   "
                     f"meets_target : {_OK if live_meets_target else _BAD}")
        if diag.get("note"):
            lines.append(f"   note         : {diag['note']}")

    # ── 2. WALLET：雙經濟循環健康（幣有沒有在流動）──────────────────────────────
    wallet, w_err = _get_json(f"{host}/wallet", timeout)
    lines.append("")
    lines.append("[R1] 雙經濟循環  (GET /wallet)")
    if w_err:
        lines.append(f"   [unreachable] {w_err}")
    else:
        ledger = wallet.get("ledger", [])
        credits = Counter()
        debits = Counter()
        for e in ledger:
            (credits if e.get("kind") == "credit" else debits)[e.get("channel", "?")] += 1
        lines.append(f"   balance      : {wallet.get('balance', '—')}   "
                     f"ticket_cost : {wallet.get('ticket_cost', '—')}")
        lines.append(f"   ledger (近{len(ledger)}筆) : "
                     f"credit={dict(credits) or '∅'}  debit={dict(debits) or '∅'}")
        # 雙源賺幣 + 花幣 同時出現 = 循環活著
        sources = set(credits)
        circulating = len(sources) >= 2 and bool(debits)
        if not ledger:
            lines.append(f"   {_BAD} ledger 空 — pilot 尚未產生任何交易")
        elif circulating:
            lines.append(f"   {_OK} 雙源賺幣({len(sources)} channels) + 花幣 同時在動 — 循環活著")
        else:
            need = []
            if len(sources) < 2:
                need.append("≥2 賺幣源")
            if not debits:
                need.append("≥1 花幣(PvP門票)")
            lines.append(f"   {_WARN} 循環未完整：缺 {', '.join(need)}")

    # ── 3. ON-DISK：β verdict（R2 凍結就緒度）─────────────────────────────────
    lines.append("")
    lines.append(f"[R2] β verdict  (ecology_beta_fit on {state_path})")
    fit = None
    if not _BETA_OK:
        lines.append(f"   [skip] β-instrument import 失敗：{_BETA_ERR}")
    elif not Path(state_path).exists():
        lines.append(f"   [skip] 找不到 state 檔：{state_path}")
    else:
        try:
            kw = {"lam": lam} if lam is not None else {}
            data = load_real_submissions(state_path, **kw)
            fit = fit_beta(data)
            lines.append(f"   verdict      : {fit.verdict}")
            lines.append(f"   n_real       : {fit.n_real}  "
                         f"(total {fit.n_total}, artifacts {fit.n_artifact}, "
                         f"min_n {MIN_N})")
            lines.append(f"   adv source   : seen_scarcity {fit.n_seen} / "
                         f"q_before proxy {fit.n_real - fit.n_seen}")
            if fit.scarcity_std is not None:
                lines.append(f"   scarcity_std : {fit.scarcity_std:.4f} "
                             f"{_mark(fit.scarcity_std, GATE_STD, TARGET_STD)}")
            if fit.verdict == "OK":
                lines.append(f"   β (response) : {fit.beta:.3f}  ±{fit.beta_se:.3f}  "
                             f"95%CI [{fit.beta_ci[0]:.3f}, {fit.beta_ci[1]:.3f}]")
            if fit.note:
                lines.append(f"   note         : {fit.note}")
        except Exception as exc:  # noqa: BLE001
            lines.append(f"   [error] β-fit 失敗：{type(exc).__name__}: {exc}")

    # ── 總判：R2 凍結就緒？──────────────────────────────────────────────────────
    # 偏好 live 稀缺（反映進行中的 pilot）；無 live 時退回 disk fit 的 scarcity_std。
    scar_for_gate = None
    scar_src = ""
    if diag and diag.get("scarcity_std") is not None:
        scar_for_gate, scar_src = diag["scarcity_std"], "live"
    elif fit is not None and fit.scarcity_std is not None:
        scar_for_gate, scar_src = fit.scarcity_std, "disk"

    n_real = fit.n_real if fit is not None else (diag.get("n_real") if diag else None)
    beta_ok = bool(fit is not None and fit.verdict == "OK")

    checks = []
    checks.append(("n_real ≥ %d" % MIN_N,
                   n_real is not None and n_real >= MIN_N, f"={n_real}"))
    checks.append(("scarcity_std ≥ %.2f (target)" % TARGET_STD,
                   scar_for_gate is not None and scar_for_gate >= TARGET_STD,
                   f"={scar_for_gate:.4f} [{scar_src}]" if scar_for_gate is not None else "=—"))
    checks.append(("β verdict == OK", beta_ok,
                   f"={fit.verdict}" if fit is not None else "=—"))

    lines.append("")
    lines.append("-" * 64)
    lines.append("R2 凍結 gate（全綠才建議 freeze + 指派 config_version）：")
    for label, ok, detail in checks:
        lines.append(f"   {_OK if ok else _BAD} {label}  {detail}")

    all_ready = all(ok for _, ok, _ in checks)
    if all_ready:
        verdict_line = f"{_OK} R2 就緒：可建議凍結快照、開 R4 confirmatory 收集。"
    elif scar_for_gate is not None and scar_for_gate < GATE_STD:
        verdict_line = (f"{_BAD} R1 未過識別 gate：scarcity_std<{GATE_STD} → "
                        f"β 不可識別。先驅動稀缺漂移（P2 強 induction，需 firewall 裁決）。")
    else:
        verdict_line = f"{_WARN} R1 進行中：gate 尚未全綠，繼續收集。"
    lines.append("")
    lines.append(verdict_line)
    lines.append("=" * 64)

    print("\n".join(lines))
    return 0 if all_ready else 1


def main() -> None:
    ap = argparse.ArgumentParser(
        description="乙 pilot 看板（read-only）：R1→R2 gate 就緒度一覽")
    ap.add_argument("--host", default="http://localhost:8001",
                    help="後端 base URL（預設 http://localhost:8001）")
    ap.add_argument("--state", default="reports/ecology/ecology_state.json",
                    help="ecology_state.json 路徑（on-disk β-fit 來源）")
    ap.add_argument("--timeout", type=float, default=3.0,
                    help="每個 GET 的逾時秒數（預設 3.0）")
    ap.add_argument("--lam", type=float, default=None,
                    help="覆寫 advantage 重建的 lam（預設讀 state.params.lam）")
    args = ap.parse_args()
    sys.exit(render(args.host, args.state, args.timeout, args.lam))


if __name__ == "__main__":
    main()
