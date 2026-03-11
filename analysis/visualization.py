"""Visualization helpers for research workflow.

設計原則
- simulation 不 import matplotlib
- notebook / scripts 只 import 本檔
- 本檔只讀 CSV 並畫圖

依賴
- 預設僅用標準庫可讀取資料
- 若要畫圖需安裝 matplotlib（可選）
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence


@dataclass(frozen=True)
class TimeSeries:
	rounds: List[int]
	avg_reward: List[float]
	avg_utility: List[float]
	proportions: Dict[str, List[float]]  # key: strategy, value: p_t
	weights: Dict[str, List[float]]  # key: strategy, value: w_t


def load_timeseries_csv(path: str | Path) -> TimeSeries:
	path = Path(path)
	with path.open() as f:
		reader = csv.DictReader(f)
		rows = list(reader)
		fields = reader.fieldnames or []

	p_cols = [c for c in fields if c.startswith("p_")]
	strategies = [c[2:] for c in p_cols]

	rounds: List[int] = []
	avg_reward: List[float] = []
	avg_utility: List[float] = []
	proportions: Dict[str, List[float]] = {s: [] for s in strategies}
	weights: Dict[str, List[float]] = {s: [] for s in strategies}

	for r in rows:
		rounds.append(int(float(r.get("round", 0))))

		ar = r.get("avg_reward", "")
		avg_reward.append(float(ar) if ar not in (None, "") else float("nan"))
		avg_utility.append(float(r.get("avg_utility", 0.0)))

		for s in strategies:
			proportions[s].append(float(r.get(f"p_{s}", 0.0)))
			weights[s].append(float(r.get(f"w_{s}", "nan")))

	return TimeSeries(
		rounds=rounds,
		avg_reward=avg_reward,
		avg_utility=avg_utility,
		proportions=proportions,
		weights=weights,
	)


def plot_proportions(ts: TimeSeries, *, title: str = "Strategy proportions") -> None:
	"""Plot strategy proportions over time (requires matplotlib)."""

	import matplotlib.pyplot as plt  # optional dependency

	for s, series in ts.proportions.items():
		plt.plot(ts.rounds, series, label=s)

	plt.ylim(0.0, 1.0)
	plt.xlabel("round")
	plt.ylabel("proportion")
	plt.title(title)
	plt.legend()
	plt.grid(True, alpha=0.3)


def plot_avg_utility(ts: TimeSeries, *, title: str = "Average utility") -> None:
	"""Plot average utility over time (requires matplotlib)."""

	import matplotlib.pyplot as plt  # optional dependency

	plt.plot(ts.rounds, ts.avg_utility, color="black")
	plt.xlabel("round")
	plt.ylabel("avg_utility")
	plt.title(title)
	plt.grid(True, alpha=0.3)


def plot_avg_reward(ts: TimeSeries, *, title: str = "Average reward") -> None:
	"""Plot average reward (last step) over time (requires matplotlib)."""

	import matplotlib.pyplot as plt  # optional dependency

	plt.plot(ts.rounds, ts.avg_reward, color="tab:gray")
	plt.xlabel("round")
	plt.ylabel("avg_reward")
	plt.title(title)
	plt.grid(True, alpha=0.3)


def plot_compare_aggressive(
	csv_paths: Sequence[str | Path],
	*,
	labels: Sequence[str] | None = None,
	title: str = "Compare p_aggressive",
) -> None:
	"""Overlay p_aggressive from multiple CSVs (requires matplotlib)."""

	import matplotlib.pyplot as plt  # optional dependency

	if labels is None:
		labels = [Path(p).stem for p in csv_paths]

	for path, lab in zip(csv_paths, labels):
		ts = load_timeseries_csv(path)
		if "aggressive" not in ts.proportions:
			continue
		plt.plot(ts.rounds, ts.proportions["aggressive"], label=lab)

	plt.ylim(0.0, 1.0)
	plt.xlabel("round")
	plt.ylabel("p_aggressive")
	plt.title(title)
	plt.legend()
	plt.grid(True, alpha=0.3)

