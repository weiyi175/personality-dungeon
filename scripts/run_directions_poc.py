#!/usr/bin/env python3
"""
Directions B/A/C Proof-of-Concept runner.
Quick validation of each direction with minimal runs (seed=45 only).
"""

import json
import subprocess
import sys
import csv
from pathlib import Path
from datetime import datetime

# ============================================================================
# PoC 配置
# ============================================================================

POC_CONFIG = {
    "B": {
        "name": "Pulsed Synergy",
        "runs": [
            {
                "name": "B2_sine_g004_T12",
                "params": {
                    "synergy_gamma": 0.04,
                    "synergy_type": "pulsed",
                    "synergy_pulse_type": "sine",
                    "synergy_pulse_period": 12,
                },
            },
            {
                "name": "B2_sine_g006_T20",
                "params": {
                    "synergy_gamma": 0.06,
                    "synergy_type": "pulsed",
                    "synergy_pulse_type": "sine",
                    "synergy_pulse_period": 20,
                },
            },
            {
                "name": "B2_sine_g008_T30",
                "params": {
                    "synergy_gamma": 0.08,
                    "synergy_type": "pulsed",
                    "synergy_pulse_type": "sine",
                    "synergy_pulse_period": 30,
                },
            },
            {
                "name": "B2_square_g004_T12",
                "params": {
                    "synergy_gamma": 0.04,
                    "synergy_type": "pulsed",
                    "synergy_pulse_type": "square",
                    "synergy_pulse_period": 12,
                },
            },
            {
                "name": "B2_square_g006_T20",
                "params": {
                    "synergy_gamma": 0.06,
                    "synergy_type": "pulsed",
                    "synergy_pulse_type": "square",
                    "synergy_pulse_period": 20,
                },
            },
            {
                "name": "B2_square_g008_T30",
                "params": {
                    "synergy_gamma": 0.08,
                    "synergy_type": "pulsed",
                    "synergy_pulse_type": "square",
                    "synergy_pulse_period": 30,
                },
            },
        ],
        "seed": 45,
        "rounds": 6000,
    },
    "A": {
        "name": "Local Synergy",
        "runs": [
            {
                "name": "A_local_tau70",
                "params": {
                    "synergy_gamma": 0.15,
                    "synergy_type": "local",
                    "synergy_local_similarity_threshold": 0.7,
                },
            },
            {
                "name": "A_local_tau50",
                "params": {
                    "synergy_gamma": 0.15,
                    "synergy_type": "local",
                    "synergy_local_similarity_threshold": 0.5,
                },
            },
        ],
        "seed": 45,
        "rounds": 6000,
    },
    "C": {
        "name": "Nonlinear Synergy",
        "runs": [
            {
                "name": "C2_piecewise_g010_eps0005",
                "params": {
                    "synergy_gamma": 0.10,
                    "synergy_type": "nonlinear",
                    "synergy_nonlinear_type": "piecewise",
                    "synergy_nonlinear_epsilon": 0.005,
                },
            },
            {
                "name": "C2_piecewise_g012_eps0005",
                "params": {
                    "synergy_gamma": 0.12,
                    "synergy_type": "nonlinear",
                    "synergy_nonlinear_type": "piecewise",
                    "synergy_nonlinear_epsilon": 0.005,
                },
            },
            {
                "name": "C2_power_g010_p25",
                "params": {
                    "synergy_gamma": 0.10,
                    "synergy_type": "nonlinear",
                    "synergy_nonlinear_type": "power",
                    "synergy_nonlinear_power": 2.5,
                },
            },
            {
                "name": "C2_power_g012_p30",
                "params": {
                    "synergy_gamma": 0.12,
                    "synergy_type": "nonlinear",
                    "synergy_nonlinear_type": "power",
                    "synergy_nonlinear_power": 3.0,
                },
            },
            {
                "name": "C2_power_g015_p25",
                "params": {
                    "synergy_gamma": 0.15,
                    "synergy_type": "nonlinear",
                    "synergy_nonlinear_type": "power",
                    "synergy_nonlinear_power": 2.5,
                },
            },
            {
                "name": "C2_power_g015_p30",
                "params": {
                    "synergy_gamma": 0.15,
                    "synergy_type": "nonlinear",
                    "synergy_nonlinear_type": "power",
                    "synergy_nonlinear_power": 3.0,
                },
            },
        ],
        "seed": 45,
        "rounds": 6000,
    },
}


def build_cli_args(params: dict, seed: int, rounds: int, output_csv: Path) -> list[str]:
    """從參數字典生成 CLI 參數列表。"""
    args = [
        "./venv/bin/python",
        "-m",
        "simulation.run_simulation",
        "--seed", str(seed),
        "--n-rounds", str(rounds),
        "--payoff-mode", "matrix_ab",
        "--evolution-mode", "deterministic",
        "--out-csv", str(output_csv),
        "--a", "1.0",
        "--b", "0.9",
        "--matrix-cross-coupling", "0.20",
        "--n-players", "300",
        "--selection-strength", "0.02",
        "--init-bias", "0.5",
        "--memory-kernel", "1",
    ]
    
    # Add synergy parameters
    for key, value in params.items():
        if key == "synergy_gamma":
            args.extend(["--synergy-gamma", str(value)])
        elif key == "synergy_type":
            args.extend(["--synergy-type", str(value)])
        elif key == "synergy_pulse_type":
            args.extend(["--synergy-pulse-type", str(value)])
        elif key == "synergy_pulse_period":
            args.extend(["--synergy-pulse-period", str(value)])
        elif key == "synergy_local_similarity_threshold":
            args.extend(["--synergy-local-similarity-threshold", str(value)])
        elif key == "synergy_nonlinear_type":
            args.extend(["--synergy-nonlinear-type", str(value)])
        elif key == "synergy_nonlinear_epsilon":
            args.extend(["--synergy-nonlinear-epsilon", str(value)])
        elif key == "synergy_nonlinear_power":
            args.extend(["--synergy-nonlinear-power", str(value)])
    
    return args


def run_poc_direction(direction: str) -> dict:
    """執行單一方向的 PoC。"""
    config = POC_CONFIG.get(direction)
    if not config:
        return {"status": "FAIL", "error": f"Unknown direction: {direction}"}
    
    result = {
        "direction": direction,
        "name": config["name"],
        "timestamp": datetime.now().isoformat(),
        "runs": [],
        "summary": {},
    }
    
    output_base = Path("outputs") / f"poc_directions_{direction}"
    output_base.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"PoC: 方向 {direction} - {config['name']}")
    print(f"{'='*70}")
    
    for i, run_spec in enumerate(config["runs"], 1):
        run_name = run_spec["name"]
        params = run_spec["params"]
        output_csv = output_base / f"{run_name}.csv"
        
        print(f"\n  [{i}/{len(config['runs'])}] {run_name}...")
        
        # 首選：程式化呼叫 simulation.run_simulation.simulate()
        try:
            from simulation.run_simulation import SimConfig, simulate

            cfg_kwargs = dict(
                n_players=300,
                n_rounds=int(config["rounds"]),
                seed=int(config["seed"]),
                payoff_mode="matrix_ab",
                popularity_mode="sampled",
                gamma=float(params.get("synergy_gamma", 0.0)),
                epsilon=0.0,
                a=1.0,
                b=0.9,
                matrix_cross_coupling=0.20,
                init_bias=0.5,
                evolution_mode="mean_field",
                payoff_lag=0,
                selection_strength=0.02,
            )
            # Inject synergy params into SimConfig when present
            if params.get("synergy_type"):
                cfg_kwargs["synergy_type"] = params.get("synergy_type")
            if params.get("synergy_gamma") is not None:
                cfg_kwargs["synergy_gamma"] = float(params.get("synergy_gamma"))
            if params.get("synergy_pulse_type") is not None:
                cfg_kwargs["synergy_pulse_type"] = params.get("synergy_pulse_type")
            if params.get("synergy_pulse_period") is not None:
                cfg_kwargs["synergy_pulse_period"] = int(params.get("synergy_pulse_period"))
            if params.get("synergy_local_similarity_threshold") is not None:
                cfg_kwargs["synergy_local_similarity_threshold"] = float(params.get("synergy_local_similarity_threshold"))
            if params.get("synergy_nonlinear_type") is not None:
                cfg_kwargs["synergy_nonlinear_type"] = params.get("synergy_nonlinear_type")
            if params.get("synergy_nonlinear_epsilon") is not None:
                cfg_kwargs["synergy_nonlinear_epsilon"] = float(params.get("synergy_nonlinear_epsilon"))
            if params.get("synergy_nonlinear_power") is not None:
                cfg_kwargs["synergy_nonlinear_power"] = float(params.get("synergy_nonlinear_power"))

            cfg = SimConfig(**cfg_kwargs)

            strategy_space, rows = simulate(cfg)

            # 寫出 CSV
            if rows:
                output_csv.parent.mkdir(parents=True, exist_ok=True)
                with open(output_csv, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(rows)

                print(f"      ✓ 成功 ({len(rows)} rows)")
                run_result = {
                    "name": run_name,
                    "status": "SUCCESS",
                    "params": params,
                    "csv_path": str(output_csv),
                    "n_rows": len(rows),
                }
            else:
                print(f"      ✗ 失敗 (no rows returned)")
                run_result = {
                    "name": run_name,
                    "status": "FAIL",
                    "error": "simulate() returned no rows",
                }

        except Exception as e:
            import traceback
            print(f"      程式化呼叫失敗: {e}")
            traceback.print_exc()
            # 次選：回退到 subprocess 呼叫 CLI（相容性保留）
            try:
                args = build_cli_args(
                    params=params,
                    seed=config["seed"],
                    rounds=config["rounds"],
                    output_csv=output_csv,
                )

                completed = subprocess.run(
                    args,
                    cwd=Path(__file__).parent.parent,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )

                if completed.returncode == 0:
                    print(f"      ✓ 成功 ({output_csv.stat().st_lines if output_csv.exists() else 0} lines)")
                    run_result = {
                        "name": run_name,
                        "status": "SUCCESS",
                        "params": params,
                        "csv_path": str(output_csv),
                    }
                else:
                    print(f"      ✗ 失敗 (exit code {completed.returncode})")
                    print(f"      stderr: {completed.stderr[:200]}")
                    run_result = {
                        "name": run_name,
                        "status": "FAIL",
                        "error": completed.stderr[:500],
                    }

            except subprocess.TimeoutExpired:
                print(f"      ✗ 超時 (> 300s)")
                run_result = {
                    "name": run_name,
                    "status": "TIMEOUT",
                }
            except Exception as e2:
                print(f"      ✗ 異常: {str(e2)}")
                run_result = {
                    "name": run_name,
                    "status": "ERROR",
                    "error": str(e2),
                }
        
        result["runs"].append(run_result)
    
    # Summarize
    success_count = sum(1 for r in result["runs"] if r.get("status") == "SUCCESS")
    result["summary"] = {
        "total_runs": len(result["runs"]),
        "successful_runs": success_count,
        "status": "PASS" if success_count == len(result["runs"]) else "PARTIAL",
    }
    
    print(f"\n  摘要: {success_count}/{len(result['runs'])} 成功")
    
    return result


def main():
    """執行所有三個方向的 PoC。"""
    print("\n" + "="*70)
    print("DIRECTIONS B/A/C PoC TEST SUITE")
    print("="*70)
    
    overall_result = {
        "timestamp": datetime.now().isoformat(),
        "directions": {},
        "overall_status": "PASS",
    }
    
    for direction in ["B", "A", "C"]:
        dir_result = run_poc_direction(direction)
        overall_result["directions"][direction] = dir_result
        
        if dir_result.get("summary", {}).get("status") != "PASS":
            overall_result["overall_status"] = "PARTIAL"
    
    # Save results
    output_file = Path("outputs") / "poc_directions_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(overall_result, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"PoC 測試完成")
    print(f"結果已保存: {output_file}")
    print(f"整體狀態: {overall_result['overall_status']}")
    print(f"{'='*70}\n")
    
    return 0 if overall_result["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
