#!/usr/bin/env python3
"""
SDD 12D 封存遷移腳本
1. 從 SDD.md 提取 12D 段落 → SDD_12D_備份/SDD_12D.md
2. 從 SDD.md 刪除這些段落
3. 驗證結果

邊界（inclusive, 1-based）:
  A: §4.7 item 0              L1879-L1951
  B: §7.2 12D Vector          L2230-L2236
  C: §7.3 smoke 契約           L2237-L2279
  D: H6                       L2708-L2737
  E: H7 + H7.1-H7.5           L2738-L3023
  F: H7.6                     L3024-L3110
  G: H7.7                     L3111-L3207
  H: H8 系列（H8.0-H8.4+總結） L3208-L3658
  I: W2.1 12D content         L3739-L3789
"""
import hashlib, shutil, sys, pathlib

BASE = pathlib.Path("/home/user/personality-dungeon")
SDD  = BASE / "SDD.md"
OUT  = BASE / "SDD_12D_備份" / "SDD_12D.md"
BACKUP = BASE / "SDD_12D_備份" / "SDD.md.bak"

# ── 1) Read original ────────────────────────────────────────────────
lines = SDD.read_text(encoding="utf-8").splitlines(keepends=True)
print(f"[INFO] SDD.md loaded: {len(lines)} lines")

# Snapshot for safety
BACKUP.write_text("".join(lines), encoding="utf-8")
print(f"[INFO] Backup written: {BACKUP}")

# ── 2) Archive ranges (inclusive, 1-based) ──────────────────────────
ARCHIVE_RANGES = [
    (1879, 1951, "A", "§4.7 item 0：Personality Event Schema 加嚴（12 維人格事件系統接軌主線前置條件）"),
    (2230, 2236, "B", "§7.2：12 維 Personality Vector（下一階段，不阻塞主線）"),
    (2237, 2279, "C", "§7.3：Personality/Event 新主線最小 smoke 契約（2026-04-01）"),
    (2708, 2737, "D", "H6：完整 Personality + Event 世界模型主線（post-H5.5R reset）"),
    (2738, 3023, "E", "H7：Personality + Dynamic Coupling 主線（H7.1-H7.5）"),
    (3024, 3110, "F", "H7.6：Noise Amplitude Sweep：相轉移階數偵測"),
    (3111, 3207, "G", "H7.7：Corner Escape Work：角點逃逸功與駐留時間發散量測"),
    (3208, 3658, "H", "H8 系列：動態噪聲控制（NIAS Phase Trap，H8.0-H8.4 + 最終總結）"),
    (3739, 3789, "I", "W2.1：12D testament 契約（P_i∈[-1,1]^12，已由 9D EXP-1.2 重驗）"),
]

# ── 3) Build SDD_12D.md ─────────────────────────────────────────────
header = """\
# SDD_12D 封存記錄 — 12 維人格實驗（已由 9D 重驗，正式封存）

> **封存日期**：2026-05-25  
> **封存原因**：`players/rl_player.py` 於 commit `355e481`（2026-05-04）
>   正式從 12D 切換為 9D（Enneagram）。所有 12D 人格實驗已以 9D 重新驗證。
>   本文件僅供歷史查閱，不再作為主線參照。  
> **主線 Spec**：`SDD.md`（`/home/user/personality-dungeon/SDD.md`）
>
> **9D trait 集合**（現行主線）：
>   - Drivers：`impulsiveness`, `assertiveness`, `optimism`
>   - Stabilizers：`risk_aversion`, `suspicion`, `endurance`
>   - Explorers：`randomness`, `stability_seeking`, `curiosity`
>
> **禁用 12D keys**：`greed`, `ambition`, `caution`, `fearfulness`, `patience`, `persistence`
>
> **⛔ 本文件禁止再作為任何主線實驗的 Spec 依據**

---

"""

archive_parts = [header]
for start, end, tag, title in ARCHIVE_RANGES:
    section_header = f"## [{tag}] {title}\n\n"
    section_header += f"> 原 SDD.md 行號：L{start}–L{end}\n\n"
    # 0-based slice
    content = "".join(lines[start-1:end])
    archive_parts.append(section_header)
    archive_parts.append(content)
    archive_parts.append("\n\n---\n\n")
    print(f"[ARCHIVE] {tag}: L{start}-L{end} ({end-start+1} lines) → {title[:50]}")

OUT.write_text("".join(archive_parts), encoding="utf-8")
print(f"\n[INFO] SDD_12D.md written: {OUT}")

# ── 4) Remove 12D sections from SDD.md ─────────────────────────────
# Convert to 0-based, sort by start descending (remove from bottom up)
removal_ranges = sorted(
    [(s-1, e-1, tag) for (s, e, tag, _) in ARCHIVE_RANGES],
    key=lambda x: x[0],
    reverse=True,
)

new_lines = list(lines)
for s0, e0, tag in removal_ranges:
    removed_count = e0 - s0 + 1
    del new_lines[s0:e0+1]
    print(f"[DELETE] {tag}: 0-based [{s0},{e0}] → removed {removed_count} lines; new total {len(new_lines)}")

SDD.write_text("".join(new_lines), encoding="utf-8")
print(f"\n[INFO] SDD.md rewritten: {len(new_lines)} lines (was {len(lines)})")
print(f"[INFO] Removed {len(lines)-len(new_lines)} lines total")

# ── 5) Verification ─────────────────────────────────────────────────
print("\n=== Verification ===")
sdd_new = SDD.read_text(encoding="utf-8")
problems = []

# Should NOT contain these strings
forbidden = [
    "## H7.6 Noise Amplitude Sweep",
    "## H7.7 Corner Escape Work",
    "## H8 系列：動態噪聲控制",
    "## H8.0 Engine Extension",
    "## H8.1 可逆相位阱",
    "### 7.2 12 維 Personality Vector",
    "### 7.3 Personality/Event 新主線最小 smoke 契約（2026-04-01）",
    "H6 Gate 2 的整體 pass 條件",
    "P_i(ℓ) ∈ [-1,1]^{12}",
    "12 維 trait template",
]
for f in forbidden:
    if f in sdd_new:
        problems.append(f"STILL FOUND: {f!r}")
        print(f"[ERROR] Still found: {f!r}")
    else:
        print(f"[OK] Removed: {f!r}")

# Should STILL contain these
required = [
    "### EXP-1.2：逆反期機制（設計草案）",
    "W1：`In-loop Adaptive World` 主線",
    "### 7.4 文本 → 9 維人格推斷",
    "9D `DOMINANT_TEMPLATES`",
    "H5.5R：`personality + event-driven nonlinear payoff`",
]
for r in required:
    if r in sdd_new:
        print(f"[OK] Still present: {r!r}")
    else:
        problems.append(f"MISSING: {r!r}")
        print(f"[ERROR] Missing (should be kept): {r!r}")

if problems:
    print(f"\n[FAIL] {len(problems)} verification problem(s):")
    for p in problems:
        print(f"  - {p}")
    sys.exit(1)
else:
    print(f"\n[PASS] All verification checks passed!")
    print(f"[INFO] SDD.md: {len(new_lines)} lines")
    print(f"[INFO] SDD_12D.md: {OUT.stat().st_size} bytes")
