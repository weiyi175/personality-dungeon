from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_TARGET_SEEDS = [47, 79]
DEFAULT_SENTINEL_SEEDS: List[int] = []
DEFAULT_S3_HEALTHY_THRESHOLD = 0.8
DEFAULT_SENTINEL_S3_FLOOR = 0.85
DEFAULT_SMOKE_SEED_SCOPE = "core"
DEFAULT_EVAL_N_ROUNDS = 12000
DEFAULT_EVAL_BURN_IN = 4000
DEFAULT_EVAL_TAIL = 4000
PYTHON_BIN = "/home/user/personality-dungeon/venv/bin/python"
BASELINE_JSON = "outputs/phase_coupling_v2/p0_baseline_gate60_summary.json"
ANCHOR_PARAMS = {
    "horizon": 6,
    "decay": 0.8278,
    "local_mass": 0.653,
    "hop": 3,
    "memory_kernel": 4,
}

# Search bounds
HORIZON_MIN, HORIZON_MAX = 6, 8
DECAY_MIN, DECAY_MAX = 0.78, 0.84
LOCAL_MASS_MIN, LOCAL_MASS_MAX = 0.56, 0.70
HOP_MIN, HOP_MAX = 1, 3
MEMORY_MIN, MEMORY_MAX = 2, 4


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def score_tuple(result: dict) -> Tuple[float, float, float, float]:
    # Lower is better; lexicographic order.
    return (
        float(result["hard_new_l1_violation"]),
        float(result["new_l1"]),
        float(result["sentinel_floor_violation_count"]),
        float(result["broken_count_selected"]),
        -float(result["min_s3_selected"]),
        -float(result["s3_sum_selected"]),
    )


def dominates(a: Tuple[float, ...], b: Tuple[float, ...]) -> bool:
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


@dataclass(frozen=True)
class Params:
    horizon: int
    decay: float
    local_mass: float
    hop: int
    memory_kernel: int

    @property
    def neighbor_mass(self) -> float:
        return round(1.0 - self.local_mass, 3)

    def normalized(self) -> "Params":
        h = int(clamp(round(self.horizon), HORIZON_MIN, HORIZON_MAX))
        d = round(clamp(self.decay, DECAY_MIN, DECAY_MAX), 4)
        lm = round(clamp(self.local_mass, LOCAL_MASS_MIN, LOCAL_MASS_MAX), 3)
        hop = int(clamp(round(self.hop), HOP_MIN, HOP_MAX))
        mk = int(clamp(round(self.memory_kernel), MEMORY_MIN, MEMORY_MAX))
        return Params(horizon=h, decay=d, local_mass=lm, hop=hop, memory_kernel=mk)

    def key(self) -> Tuple[int, float, float, int, int]:
        p = self.normalized()
        return (p.horizon, p.decay, p.local_mass, p.hop, p.memory_kernel)


def random_params(rng: random.Random) -> Params:
    return Params(
        horizon=rng.randint(HORIZON_MIN, HORIZON_MAX),
        decay=round(rng.uniform(DECAY_MIN, DECAY_MAX), 4),
        local_mass=round(rng.uniform(LOCAL_MASS_MIN, LOCAL_MASS_MAX), 3),
        hop=rng.randint(HOP_MIN, HOP_MAX),
        memory_kernel=rng.randint(MEMORY_MIN, MEMORY_MAX),
    ).normalized()


def crossover_mutate(a: Params, b: Params, rng: random.Random, mutation_rate: float) -> Params:
    child = Params(
        horizon=rng.choice([a.horizon, b.horizon]),
        decay=(a.decay + b.decay) / 2.0,
        local_mass=(a.local_mass + b.local_mass) / 2.0,
        hop=rng.choice([a.hop, b.hop]),
        memory_kernel=rng.choice([a.memory_kernel, b.memory_kernel]),
    )

    # Polynomial-like light mutation
    if rng.random() < mutation_rate:
        child = Params(
            horizon=child.horizon + rng.choice([-1, 1]),
            decay=child.decay + rng.uniform(-0.01, 0.01),
            local_mass=child.local_mass + rng.uniform(-0.03, 0.03),
            hop=child.hop + rng.choice([-1, 1]),
            memory_kernel=child.memory_kernel + rng.choice([-1, 1]),
        )

    return child.normalized()


def evaluate_one(
    params: Params,
    eval_id: str,
    run_dir: Path,
    core_seeds: List[int],
    sentinel_seeds: List[int],
    smoke_seeds: List[int],
    s3_healthy_threshold: float,
    sentinel_s3_floor: float,
    eval_n_rounds: int,
    eval_burn_in: int,
    eval_tail: int,
) -> dict:
    p = params.normalized()
    eval_dir = run_dir / "evals"
    eval_dir.mkdir(parents=True, exist_ok=True)

    smoke_json = eval_dir / f"{eval_id}_smoke_summary.json"
    gate_json = eval_dir / f"{eval_id}_gate_summary.json"
    log_path = eval_dir / f"{eval_id}.log"

    selected_seeds = sorted(set(core_seeds + sentinel_seeds))

    cmd = [
        PYTHON_BIN,
        "-m",
        "simulation.b1_async_dispatch_gate",
        "--phase-id",
        f"tb2_b3_nsga2_{eval_id}",
        "--bridge-id",
        "b3_impact_spreading_v2",
        "--bridge-count",
        "1",
        "--anchor-profile-id",
        "phase2_pass_locked_v1",
        "--smoke-seeds",
        ",".join(str(s) for s in smoke_seeds),
        "--gate-seeds",
        ",".join(str(s) for s in selected_seeds),
        "--baseline-summary-json",
        BASELINE_JSON,
        "--n-rounds",
        str(eval_n_rounds),
        "--burn-in",
        str(eval_burn_in),
        "--tail",
        str(eval_tail),
        "--world-feedback-mode",
        "difficulty_only",
        "--event-dispatch-mode",
        "sync",
        "--event-dispatch-target-rate",
        "0.08",
        "--event-dispatch-fairness-window",
        "2000",
        "--event-dispatch-fairness-tolerance",
        "0.50",
        "--event-trigger-mode",
        "entropy_guard",
        "--event-trigger-entropy-threshold",
        "0.85",
        "--event-neutralize-payoff",
        "--event-reward-mode",
        "additive",
        "--event-impact-mode",
        "spread",
        "--event-impact-horizon",
        str(p.horizon),
        "--event-impact-decay",
        str(p.decay),
        "--impact-spread-kernel-id",
        "hierarchical_v2",
        "--impact-spread-local-mass",
        str(p.local_mass),
        "--impact-spread-neighbor-mass",
        str(p.neighbor_mass),
        "--impact-spread-neighbor-hop",
        str(p.hop),
        "--impact-spread-memory-kernel",
        str(p.memory_kernel),
        "--smoke-out-json",
        str(smoke_json),
        "--gate-out-json",
        str(gate_json),
    ]

    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, check=False)

    result = {
        "eval_id": eval_id,
        "params": asdict(p),
        "neighbor_mass": p.neighbor_mass,
        "return_code": proc.returncode,
        "smoke_json": str(smoke_json),
        "gate_json": str(gate_json),
        "log": str(log_path),
    }

    if not gate_json.exists():
        seed_s3 = {f"seed{s}_s3": -1.0 for s in selected_seeds}
        sentinel_s3 = [seed_s3[f"seed{s}_s3"] for s in sentinel_seeds]
        core_s3 = [seed_s3[f"seed{s}_s3"] for s in core_seeds]
        result.update(
            {
                "hard_new_l1_violation": 1,
                "new_l1": 99,
                "l1_total": 99,
                **seed_s3,
                "core_min_s3": min(core_s3) if core_s3 else -1.0,
                "core_s3_sum": float(sum(core_s3)),
                "sentinel_min_s3": min(sentinel_s3) if sentinel_s3 else 1.0,
                "sentinel_s3_sum": float(sum(sentinel_s3)),
                "min_s3_selected": -1.0,
                "s3_sum_selected": -float(len(selected_seeds)),
                "broken_count_selected": len(selected_seeds),
                "broken_count": len(selected_seeds),
                "sentinel_floor_violation_count": len(sentinel_seeds),
                "objectives": [1.0, 99.0, float(len(sentinel_seeds)), float(len(selected_seeds)), 1.0, float(len(selected_seeds))],
                "valid": False,
            }
        )
        return result

    data = json.loads(gate_json.read_text(encoding="utf-8"))
    outcomes = {int(r["seed"]): r for r in data.get("outcomes", [])}
    seed_s3_values = {s: float(outcomes.get(s, {}).get("s3", 0.0)) for s in selected_seeds}
    core_s3_values = {s: seed_s3_values[s] for s in core_seeds}
    sentinel_s3_values = {s: seed_s3_values[s] for s in sentinel_seeds}

    min_s3_selected = min(seed_s3_values.values())
    s3_sum_selected = sum(seed_s3_values.values())
    broken_count_selected = sum(int(v < s3_healthy_threshold) for v in seed_s3_values.values())

    core_min_s3 = min(core_s3_values.values()) if core_s3_values else min_s3_selected
    core_s3_sum = sum(core_s3_values.values()) if core_s3_values else 0.0
    sentinel_min_s3 = min(sentinel_s3_values.values()) if sentinel_s3_values else 1.0
    sentinel_s3_sum = sum(sentinel_s3_values.values()) if sentinel_s3_values else 0.0
    sentinel_floor_violation_count = sum(int(v < sentinel_s3_floor) for v in sentinel_s3_values.values())

    # Guarded multi-objective minimization targets
    # f1: hard gate violation flag for new_l1 (0 if fully fixed)
    # f2: new_l1 count on selected seeds
    # f3: sentinel floor violation count
    # f4: broken count on selected seeds
    # f5: -min_s3_selected
    # f6: -s3_sum_selected
    new_l1 = int(data.get("new_l1", 99))
    hard_new_l1_violation = int(new_l1 > 0)

    f1 = float(hard_new_l1_violation)
    f2 = float(new_l1)
    f3 = float(sentinel_floor_violation_count)
    f4 = float(broken_count_selected)
    f5 = -float(min_s3_selected)
    f6 = -float(s3_sum_selected)

    result.update(
        {
            "hard_new_l1_violation": hard_new_l1_violation,
            "new_l1": new_l1,
            "l1_total": int(data.get("l1", 99)),
            **{f"seed{s}_s3": v for s, v in seed_s3_values.items()},
            "core_min_s3": core_min_s3,
            "core_s3_sum": core_s3_sum,
            "sentinel_min_s3": sentinel_min_s3,
            "sentinel_s3_sum": sentinel_s3_sum,
            "min_s3_selected": min_s3_selected,
            "s3_sum_selected": s3_sum_selected,
            "broken_count_selected": broken_count_selected,
            "broken_count": broken_count_selected,
            "sentinel_floor_violation_count": sentinel_floor_violation_count,
            "mean_alignment": float(data.get("mean_impact_spread_alignment", -1.0)),
            "mass_error_max": float(data.get("impact_kernel_mass_error_max", 1e9)),
            "objectives": [f1, f2, f3, f4, f5, f6],
            "valid": True,
        }
    )
    return result


def non_dominated_sort(items: List[dict]) -> List[List[int]]:
    objs = [tuple(i["result"]["objectives"]) for i in items]
    s: List[List[int]] = [[] for _ in items]
    n = [0 for _ in items]
    fronts: List[List[int]] = [[]]

    for p_idx in range(len(items)):
        for q_idx in range(len(items)):
            if p_idx == q_idx:
                continue
            if dominates(objs[p_idx], objs[q_idx]):
                s[p_idx].append(q_idx)
            elif dominates(objs[q_idx], objs[p_idx]):
                n[p_idx] += 1
        if n[p_idx] == 0:
            fronts[0].append(p_idx)

    i = 0
    while i < len(fronts) and fronts[i]:
        next_front: List[int] = []
        for p_idx in fronts[i]:
            for q_idx in s[p_idx]:
                n[q_idx] -= 1
                if n[q_idx] == 0:
                    next_front.append(q_idx)
        if next_front:
            fronts.append(next_front)
        i += 1

    return fronts


def crowding_distance(front_indices: List[int], items: List[dict]) -> Dict[int, float]:
    if not front_indices:
        return {}
    distances = {idx: 0.0 for idx in front_indices}
    m = len(items[front_indices[0]]["result"]["objectives"])

    for obj_i in range(m):
        sorted_idx = sorted(front_indices, key=lambda idx: items[idx]["result"]["objectives"][obj_i])
        lo = items[sorted_idx[0]]["result"]["objectives"][obj_i]
        hi = items[sorted_idx[-1]]["result"]["objectives"][obj_i]
        distances[sorted_idx[0]] = math.inf
        distances[sorted_idx[-1]] = math.inf
        if hi == lo:
            continue
        for k in range(1, len(sorted_idx) - 1):
            prev_v = items[sorted_idx[k - 1]]["result"]["objectives"][obj_i]
            next_v = items[sorted_idx[k + 1]]["result"]["objectives"][obj_i]
            distances[sorted_idx[k]] += (next_v - prev_v) / (hi - lo)

    return distances


def nsga2_select_next(items: List[dict], pop_size: int) -> List[dict]:
    fronts = non_dominated_sort(items)
    selected: List[dict] = []
    for front in fronts:
        if not front:
            continue
        if len(selected) + len(front) <= pop_size:
            selected.extend(items[idx] for idx in front)
            continue

        dmap = crowding_distance(front, items)
        ranked = sorted(front, key=lambda idx: dmap[idx], reverse=True)
        remain = pop_size - len(selected)
        selected.extend(items[idx] for idx in ranked[:remain])
        break

    return selected


def assign_rank_crowding(items: List[dict]) -> List[dict]:
    # Return shallow copy with rank/crowding annotations for tournament.
    annotated = [{**i} for i in items]
    fronts = non_dominated_sort(annotated)
    for rank, front in enumerate(fronts):
        if not front:
            continue
        dmap = crowding_distance(front, annotated)
        for idx in front:
            annotated[idx]["rank"] = rank
            annotated[idx]["crowding"] = dmap.get(idx, 0.0)
    return annotated


def tournament_pick(pool: List[dict], rng: random.Random) -> dict:
    a, b = rng.sample(pool, 2)
    a_rank = a.get("rank", math.inf)
    b_rank = b.get("rank", math.inf)
    if a_rank < b_rank:
        return a
    if b_rank < a_rank:
        return b
    # rank tie -> higher crowding
    if a.get("crowding", 0.0) > b.get("crowding", 0.0):
        return a
    if b.get("crowding", 0.0) > a.get("crowding", 0.0):
        return b
    return a if rng.random() < 0.5 else b


def evaluate_batch(
    candidates: List[Params],
    eval_prefix: str,
    run_dir: Path,
    workers: int,
    cache: Dict[Tuple[int, float, float, int, int], dict],
    core_seeds: List[int],
    sentinel_seeds: List[int],
    smoke_seeds: List[int],
    s3_healthy_threshold: float,
    sentinel_s3_floor: float,
    eval_n_rounds: int,
    eval_burn_in: int,
    eval_tail: int,
) -> List[dict]:
    to_eval: List[Tuple[Params, str]] = []
    out: List[dict] = []

    for i, p in enumerate(candidates):
        pn = p.normalized()
        key = pn.key()
        if key in cache:
            out.append({"params": asdict(pn), "result": cache[key], "key": key})
        else:
            eval_id = f"{eval_prefix}_{i:03d}"
            to_eval.append((pn, eval_id))

    if to_eval:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(
                    evaluate_one,
                    p,
                    eval_id,
                    run_dir,
                    core_seeds,
                    sentinel_seeds,
                    smoke_seeds,
                    s3_healthy_threshold,
                    sentinel_s3_floor,
                    eval_n_rounds,
                    eval_burn_in,
                    eval_tail,
                ): p.key()
                for p, eval_id in to_eval
            }
            for fut in as_completed(futs):
                key = futs[fut]
                res = fut.result()
                cache[key] = res

    # rebuild in original order
    for p in candidates:
        pn = p.normalized()
        key = pn.key()
        out.append({"params": asdict(pn), "result": cache[key], "key": key})

    # Deduplicate by key while preserving first occurrence
    seen = set()
    uniq = []
    for row in out:
        if row["key"] in seen:
            continue
        seen.add(row["key"])
        uniq.append(row)
    return uniq


def build_neighbors(base: Params, rng: random.Random, limit: int) -> List[Params]:
    candidates = []

    deltas = [
        Params(base.horizon + 1, base.decay, base.local_mass, base.hop, base.memory_kernel),
        Params(base.horizon - 1, base.decay, base.local_mass, base.hop, base.memory_kernel),
        Params(base.horizon, base.decay + 0.01, base.local_mass, base.hop, base.memory_kernel),
        Params(base.horizon, base.decay - 0.01, base.local_mass, base.hop, base.memory_kernel),
        Params(base.horizon, base.decay, base.local_mass + 0.03, base.hop, base.memory_kernel),
        Params(base.horizon, base.decay, base.local_mass - 0.03, base.hop, base.memory_kernel),
        Params(base.horizon, base.decay, base.local_mass, base.hop + 1, base.memory_kernel),
        Params(base.horizon, base.decay, base.local_mass, base.hop - 1, base.memory_kernel),
        Params(base.horizon, base.decay, base.local_mass, base.hop, base.memory_kernel + 1),
        Params(base.horizon, base.decay, base.local_mass, base.hop, base.memory_kernel - 1),
        # mixed moves around c17/c08 ridge
        Params(base.horizon, base.decay + 0.005, base.local_mass + 0.02, base.hop, base.memory_kernel),
        Params(base.horizon, base.decay - 0.005, base.local_mass - 0.02, base.hop, base.memory_kernel),
    ]

    keys = set()
    for d in deltas:
        dn = d.normalized()
        if dn.key() == base.normalized().key() or dn.key() in keys:
            continue
        keys.add(dn.key())
        candidates.append(dn)

    rng.shuffle(candidates)
    return candidates[:limit]


def main() -> None:
    ap = argparse.ArgumentParser(description="TB2 B3 NSGA-II global search + top3 hill-climbing")
    ap.add_argument("--run-dir", type=str, default="")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260419)
    ap.add_argument("--pop-size", type=int, default=10)
    ap.add_argument("--generations", type=int, default=2)
    ap.add_argument("--mutation-rate", type=float, default=0.30)
    ap.add_argument("--hill-topk", type=int, default=3)
    ap.add_argument("--hill-steps", type=int, default=2)
    ap.add_argument("--hill-neighbors", type=int, default=6)
    ap.add_argument(
        "--target-seeds",
        type=str,
        default=",".join(str(s) for s in DEFAULT_TARGET_SEEDS),
        help="Comma-separated target seeds for local objective optimization, e.g. 47,79,86",
    )
    ap.add_argument(
        "--sentinel-seeds",
        type=str,
        default=",".join(str(s) for s in DEFAULT_SENTINEL_SEEDS),
        help="Comma-separated sentinel seeds for global guardrails, e.g. 46,58,80,101,72",
    )
    ap.add_argument("--s3-healthy-threshold", type=float, default=DEFAULT_S3_HEALTHY_THRESHOLD)
    ap.add_argument("--sentinel-s3-floor", type=float, default=DEFAULT_SENTINEL_S3_FLOOR)
    ap.add_argument(
        "--smoke-seed-scope",
        type=str,
        choices=["core", "selected"],
        default=DEFAULT_SMOKE_SEED_SCOPE,
        help="Scope for smoke seeds per evaluation: core or selected(core+sentinel)",
    )
    ap.add_argument("--eval-n-rounds", type=int, default=DEFAULT_EVAL_N_ROUNDS)
    ap.add_argument("--eval-burn-in", type=int, default=DEFAULT_EVAL_BURN_IN)
    ap.add_argument("--eval-tail", type=int, default=DEFAULT_EVAL_TAIL)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    core_seeds = [int(x.strip()) for x in args.target_seeds.split(",") if x.strip()]
    core_seeds = sorted(set(core_seeds))
    if not core_seeds:
        raise ValueError("--target-seeds must contain at least one seed")

    sentinel_seeds = [int(x.strip()) for x in args.sentinel_seeds.split(",") if x.strip()]
    sentinel_seeds = sorted(set(sentinel_seeds) - set(core_seeds))
    selected_seeds = sorted(set(core_seeds + sentinel_seeds))
    smoke_seeds = core_seeds if args.smoke_seed_scope == "core" else selected_seeds

    s3_healthy_threshold = float(args.s3_healthy_threshold)
    sentinel_s3_floor = float(args.sentinel_s3_floor)

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        seed_tag = "_".join(str(s) for s in selected_seeds)
        run_dir = Path(
            f"outputs/track_b_v2/b3_impact_spreading_v2/tuning_seed{seed_tag}_nsga2_hill_guarded_20260419"
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    cache: Dict[Tuple[int, float, float, int, int], dict] = {}

    # Initialize population
    population: List[Params] = []
    keys = set()

    # Always include the current best-known anchor so local refinement can continue from it.
    anchor = Params(**ANCHOR_PARAMS).normalized()
    keys.add(anchor.key())
    population.append(anchor)

    while len(population) < args.pop_size:
        p = random_params(rng)
        if p.key() in keys:
            continue
        keys.add(p.key())
        population.append(p)

    history = []

    for gen in range(args.generations + 1):
        # Evaluate current population
        eval_prefix = f"g{gen}_pop"
        pop_items = evaluate_batch(
            population,
            eval_prefix,
            run_dir,
            args.workers,
            cache,
            core_seeds,
            sentinel_seeds,
            smoke_seeds,
            s3_healthy_threshold,
            sentinel_s3_floor,
            args.eval_n_rounds,
            args.eval_burn_in,
            args.eval_tail,
        )

        # Log generation summary
        best_gen = min(pop_items, key=lambda x: score_tuple(x["result"]))
        history.append(
            {
                "generation": gen,
                "population_size": len(pop_items),
                "best": {
                    "params": best_gen["params"],
                    "result": best_gen["result"],
                },
            }
        )

        if gen == args.generations:
            population_items = pop_items
            break

        # Build mating pool from rank/crowding
        annotated = assign_rank_crowding(pop_items)
        offspring: List[Params] = []
        seen_off = set()

        while len(offspring) < args.pop_size:
            p1 = tournament_pick(annotated, rng)
            p2 = tournament_pick(annotated, rng)
            child = crossover_mutate(
                Params(**p1["params"]),
                Params(**p2["params"]),
                rng,
                mutation_rate=args.mutation_rate,
            )
            if child.key() in seen_off:
                child = random_params(rng)
            seen_off.add(child.key())
            offspring.append(child)

        off_items = evaluate_batch(
            offspring,
            f"g{gen}_off",
            run_dir,
            args.workers,
            cache,
            core_seeds,
            sentinel_seeds,
            smoke_seeds,
            s3_healthy_threshold,
            sentinel_s3_floor,
            args.eval_n_rounds,
            args.eval_burn_in,
            args.eval_tail,
        )

        # Environmental selection
        combined = pop_items + off_items
        # Deduplicate in combined by key, keep best (by score tuple)
        by_key = {}
        for item in combined:
            k = item["key"]
            if k not in by_key or score_tuple(item["result"]) < score_tuple(by_key[k]["result"]):
                by_key[k] = item
        combined_unique = list(by_key.values())

        next_items = nsga2_select_next(combined_unique, args.pop_size)
        population = [Params(**i["params"]) for i in next_items]

    # Global Pareto from all evaluated points
    all_items = [
        {
            "params": {
                "horizon": k[0],
                "decay": k[1],
                "local_mass": k[2],
                "hop": k[3],
                "memory_kernel": k[4],
            },
            "key": k,
            "result": v,
        }
        for k, v in cache.items()
    ]

    fronts = non_dominated_sort(all_items)
    pareto_front = [all_items[idx] for idx in fronts[0]] if fronts else []
    pareto_sorted = sorted(pareto_front, key=lambda x: score_tuple(x["result"]))

    global_top = pareto_sorted[: args.hill_topk]

    # Hill-climbing from top-k
    hill_results = []
    for i, seed_item in enumerate(global_top):
        current = seed_item
        trail = [current]
        for step in range(args.hill_steps):
            base_p = Params(**current["params"])
            neighbors = build_neighbors(base_p, rng, args.hill_neighbors)
            if not neighbors:
                break
            n_items = evaluate_batch(
                neighbors,
                eval_prefix=f"hill_{i}_s{step}",
                run_dir=run_dir,
                workers=args.workers,
                cache=cache,
                core_seeds=core_seeds,
                sentinel_seeds=sentinel_seeds,
                smoke_seeds=smoke_seeds,
                s3_healthy_threshold=s3_healthy_threshold,
                sentinel_s3_floor=sentinel_s3_floor,
                eval_n_rounds=args.eval_n_rounds,
                eval_burn_in=args.eval_burn_in,
                eval_tail=args.eval_tail,
            )
            best_n = min(n_items, key=lambda x: score_tuple(x["result"]))
            if score_tuple(best_n["result"]) < score_tuple(current["result"]):
                current = best_n
                trail.append(current)
            else:
                break

        hill_results.append(
            {
                "seed_rank": i + 1,
                "start": {
                    "params": seed_item["params"],
                    "result": seed_item["result"],
                },
                "final": {
                    "params": current["params"],
                    "result": current["result"],
                },
                "improved": score_tuple(current["result"]) < score_tuple(seed_item["result"]),
                "trail_length": len(trail),
            }
        )

    candidates_all = [i for i in global_top]
    for hr in hill_results:
        candidates_all.append({
            "params": hr["final"]["params"],
            "key": Params(**hr["final"]["params"]).key(),
            "result": hr["final"]["result"],
        })

    best_overall = min(candidates_all, key=lambda x: score_tuple(x["result"]))

    summary = {
        "search": {
            "method": "NSGA-II + hill-climbing",
            "seed": args.seed,
            "workers": args.workers,
            "pop_size": args.pop_size,
            "generations": args.generations,
            "mutation_rate": args.mutation_rate,
            "hill_topk": args.hill_topk,
            "hill_steps": args.hill_steps,
            "hill_neighbors": args.hill_neighbors,
            "target_seeds": core_seeds,
            "sentinel_seeds": sentinel_seeds,
            "selected_seeds": selected_seeds,
            "smoke_seeds": smoke_seeds,
            "smoke_seed_scope": args.smoke_seed_scope,
            "eval_n_rounds": args.eval_n_rounds,
            "eval_burn_in": args.eval_burn_in,
            "eval_tail": args.eval_tail,
            "s3_healthy_threshold": s3_healthy_threshold,
            "sentinel_s3_floor": sentinel_s3_floor,
            "objective": "guarded multi-objective: min hard_new_l1_violation, min new_l1, min sentinel_floor_violations, min broken_count_selected, max min_s3_selected, max s3_sum_selected",
        },
        "history": history,
        "global": {
            "evaluated_points": len(cache),
            "pareto_size": len(pareto_front),
            "top_pareto": [
                {"params": i["params"], "result": i["result"]}
                for i in global_top
            ],
        },
        "hill_climb": hill_results,
        "best_overall": {
            "params": best_overall["params"],
            "result": best_overall["result"],
            "score_tuple": score_tuple(best_overall["result"]),
        },
    }

    summary_path = run_dir / "nsga2_hill_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # Flat table for quick review
    rows = []
    for k, v in cache.items():
        row = {
            "horizon": k[0],
            "decay": k[1],
            "local_mass": k[2],
            "neighbor_mass": round(1.0 - k[2], 3),
            "hop": k[3],
            "memory_kernel": k[4],
            "new_l1": v["new_l1"],
            "hard_new_l1_violation": v["hard_new_l1_violation"],
            "sentinel_floor_violation_count": v["sentinel_floor_violation_count"],
            "broken_count_selected": v["broken_count_selected"],
            "min_s3_selected": v["min_s3_selected"],
            "s3_sum_selected": v["s3_sum_selected"],
            "core_min_s3": v["core_min_s3"],
            "core_s3_sum": v["core_s3_sum"],
            "sentinel_min_s3": v["sentinel_min_s3"],
            "sentinel_s3_sum": v["sentinel_s3_sum"],
        }
        for seed in selected_seeds:
            row[f"seed{seed}_s3"] = v.get(f"seed{seed}_s3", -1.0)
        rows.append(row)
    rows.sort(
        key=lambda r: (
            r["hard_new_l1_violation"],
            r["new_l1"],
            r["sentinel_floor_violation_count"],
            r["broken_count_selected"],
            -r["min_s3_selected"],
            -r["s3_sum_selected"],
        )
    )

    table_path = run_dir / "all_evaluations_sorted.json"
    table_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print("NSGA2_HILL_DONE")
    print("RUN_DIR", run_dir)
    print("EVALUATED_POINTS", len(cache))
    print("PARETO_SIZE", len(pareto_front))
    print("TARGET_SEEDS", core_seeds)
    print("SENTINEL_SEEDS", sentinel_seeds)
    print("SELECTED_SEEDS", selected_seeds)
    print("BEST_OVERALL", json.dumps(summary["best_overall"], ensure_ascii=False))
    print("SUMMARY_JSON", summary_path)
    print("ALL_EVALS_JSON", table_path)


if __name__ == "__main__":
    main()
