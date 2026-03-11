"""Plot/inspect simulation timeseries CSV (no third-party deps).

用法（專案根目錄）：
- python -m analysis.plot_timeseries outputs/timeseries.csv

這個工具只做「快速檢視」：
- 印出欄位資訊
- 印出最後幾回合策略比例
- 用簡單 ASCII bar 看趨勢（可在沒裝 matplotlib 的情況下使用）

若未來要高品質圖表：建議在 notebook 用 pandas/matplotlib。
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path


def _bar(x: float, width: int = 30) -> str:
    x = max(0.0, min(1.0, x))
    n = int(round(x * width))
    return "█" * n + " " * (width - n)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("Usage: python -m analysis.plot_timeseries <csv_path>")
        return 2

    path = Path(argv[0])
    if not path.exists():
        print(f"File not found: {path}")
        return 2

    with path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames or []

    strategy_cols = [c for c in fields if c.startswith("p_")]
    if not rows:
        print("CSV has no rows.")
        return 0

    print(f"Loaded: {path}")
    print(f"rows={len(rows)} cols={len(fields)}")
    print("strategy columns:", ", ".join(strategy_cols))

    tail = rows[-5:]
    print("\nLast 5 rounds (proportions):")
    for r in tail:
        t = int(float(r["round"]))
        parts = []
        for c in strategy_cols:
            parts.append(f"{c}={float(r[c]):.2f}")
        print(f"round={t:03d} " + " ".join(parts))

    print("\nASCII trend (last row):")
    last = rows[-1]
    for c in strategy_cols:
        v = float(last[c])
        print(f"{c:>12}: {v:.2f} |{_bar(v)}|")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
