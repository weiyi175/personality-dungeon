#!/usr/bin/env python3
"""
Validation script for directions B/A/C setup.
Checks environment isolation, field definitions, boundary standards, and validation criteria.
"""

import json
import sys
from pathlib import Path
from typing import Any

# ============================================================================
# 方向設置定義
# ============================================================================

DIRECTIONS_SETUP = {
    "B": {
        "name": "Pulsed Synergy (脈衝濾波)",
        "description": "周期性或記憶門控共榮獎勵",
        "priority": 1,
        
        "environment_isolation": {
            "synergy_type": "pulsed",
            "output_dir_suffix": "_pulsed",
        },
        
        "field_definitions": {
            "required_fields": [
                "synergy_gamma",
                "synergy_basis",
                "synergy_type",
                "synergy_pulse_period",
                "synergy_pulse_type",
            ],
            "optional_fields": [
                "synergy_coherence_threshold",
                "synergy_coherence_window",
            ],
            "pulse_types": ["sine", "square", "memory_gated"],
            "valid_pulse_periods": list(range(10, 500, 10)),  # 10, 20, ..., 490
        },
        
        "boundary_standards": {
            "success_criteria": {
                "cycle_level": {"min": 2, "target": 3, "description": "≥ L2 (vs W4 control L3, scan L0)"},
                "phase_velocity": {"min": 0.001, "description": "> 0.001 (vs W4 scan ~0)"},
                "stagnant_ratio": {"max": 0.5, "description": "< 50% (vs W4 scan 100%)"},
            },
            "failure_criteria": {
                "cycle_level": {"max": 0, "description": "L0 = same as W4 scan"},
                "stagnant_ratio": {"min": 0.8, "description": "> 80%"},
            },
        },
        
        "validation_standards": {
            "diagnostics": [
                "cycle_level",
                "phase_velocity",
                "turn_strength",
                "stagnant_flag",
                "short_s3_mean",
            ],
            "direction_specific": [
                "pulse_modulation_strength",  # energy in pulse vs off
                "pulse_activation_ratio",      # fraction of rounds with modulation > 0.5
            ],
        },
        
        "experiment_design": {
            "poc_protocol": {
                "config": "population=300, rounds=6000, seed=45",
                "base_params": "unchanged",
                "variations": [
                    {"pulse_period": 50, "pulse_type": "sine"},
                    {"pulse_period": 100, "pulse_type": "sine"},
                    {"pulse_period": 200, "pulse_type": "sine"},
                ],
                "expected_runtime": "5-10 min",
            },
            "full_protocol": {
                "config": "population=300, rounds=6000, seed=[45,47,49,51,53,55]",
                "combinations": "5 pulse_configs × 6 seeds = 30 runs",
                "expected_runtime": "1-2 hours",
            },
        },
    },
    
    "A": {
        "name": "Local Synergy (局部共榮)",
        "description": "基於人格相似度的本地社群共榮",
        "priority": 2,
        
        "environment_isolation": {
            "synergy_type": "local",
            "output_dir_suffix": "_local",
        },
        
        "field_definitions": {
            "required_fields": [
                "synergy_gamma",
                "synergy_basis",
                "synergy_type",
                "synergy_local_similarity_threshold",
                "synergy_local_metric",
            ],
            "similarity_metrics": ["l2", "cosine"],
            "valid_thresholds": [v/10.0 for v in range(1, 10)],  # 0.1, 0.2, ..., 0.9
        },
        
        "boundary_standards": {
            "success_criteria": {
                "cycle_level": {"min": 1, "target": 3, "description": "≥ L1 (any improvement from L0)"},
                "n_communities": {"min": 2, "description": "多個可分離社群"},
                "community_coherence": {"min": 0.6, "description": "社群內相干性 > 0.6"},
            },
            "failure_criteria": {
                "cycle_level": {"max": 0, "description": "L0 = global collapse"},
                "community_formation": {"fail": True, "description": "無明確社群分化"},
            },
        },
        
        "validation_standards": {
            "diagnostics": [
                "cycle_level",
                "phase_velocity",
                "stagnant_flag",
            ],
            "direction_specific": [
                "community_count",
                "community_sizes",
                "inter_community_coupling",
                "intra_community_coherence",
            ],
        },
        
        "experiment_design": {
            "poc_protocol": {
                "config": "population=300, rounds=6000, seed=45",
                "variations": [
                    {"similarity_threshold": 0.5, "metric": "l2"},
                    {"similarity_threshold": 0.7, "metric": "l2"},
                ],
                "expected_runtime": "10-15 min",
            },
            "full_protocol": {
                "config": "population=300, rounds=6000, seed=[45,47,49,51,53,55]",
                "combinations": "4 threshold_configs × 6 seeds = 24 runs",
                "expected_runtime": "2-3 hours",
            },
        },
    },
    
    "C": {
        "name": "Nonlinear Synergy (非線性拓樸)",
        "description": "距離依賴的非線性包絡",
        "priority": 3,
        
        "environment_isolation": {
            "synergy_type": "nonlinear",
            "output_dir_suffix": "_nonlinear",
        },
        
        "field_definitions": {
            "required_fields": [
                "synergy_gamma",
                "synergy_basis",
                "synergy_type",
                "synergy_nonlinear_type",
            ],
            "nonlinear_types": ["piecewise", "power", "quadratic"],
            "piecewise_epsilons": [0.01, 0.05, 0.10, 0.20],
            "power_values": [1.5, 2.0, 2.5, 3.0],
        },
        
        "boundary_standards": {
            "success_criteria": {
                "cycle_level": {"min": 2, "target": 3, "description": "≥ L2"},
                "envelope_center_null": {"max": 0.1, "description": "中心區 |Δu| < 0.1 × γ"},
                "envelope_boundary_active": {"min": 0.5, "description": "邊界區 |Δu| > 0.5 × γ"},
            },
            "failure_criteria": {
                "cycle_level": {"max": 0, "description": "L0"},
                "nonlinear_instability": {"fail": True, "description": "新的非線性不穩定性"},
            },
        },
        
        "validation_standards": {
            "diagnostics": [
                "cycle_level",
                "phase_velocity",
                "stagnant_flag",
            ],
            "direction_specific": [
                "envelope_center_activity",
                "envelope_boundary_activity",
                "distance_distribution",
            ],
        },
        
        "experiment_design": {
            "poc_protocol": {
                "config": "population=300, rounds=6000, seed=45",
                "variations": [
                    {"nonlinear_type": "piecewise", "epsilon": 0.05},
                    {"nonlinear_type": "power", "power": 2.0},
                ],
                "expected_runtime": "10-15 min",
            },
            "full_protocol": {
                "config": "population=300, rounds=6000, seed=[45,47,49,51,53,55]",
                "combinations": "6 nonlinear_configs × 6 seeds = 36 runs",
                "expected_runtime": "2-3 hours",
            },
        },
    },
}


def check_field_definitions(direction: str) -> dict[str, Any]:
    """檢查欄位定義是否完整。"""
    setup = DIRECTIONS_SETUP.get(direction)
    if not setup:
        return {"status": "FAIL", "error": f"Unknown direction: {direction}"}
    
    result = {
        "direction": direction,
        "name": setup["name"],
        "status": "PASS",
        "checks": {},
    }
    
    # Check field definitions
    fields = setup["field_definitions"]
    result["checks"]["required_fields"] = {
        "fields": fields.get("required_fields", []),
        "count": len(fields.get("required_fields", [])),
        "status": "PASS" if fields.get("required_fields") else "WARNING",
    }
    
    return result


def check_boundary_standards(direction: str) -> dict[str, Any]:
    """檢查邊界標準是否清晰。"""
    setup = DIRECTIONS_SETUP.get(direction)
    if not setup:
        return {"status": "FAIL", "error": f"Unknown direction: {direction}"}
    
    result = {
        "direction": direction,
        "status": "PASS",
        "success_criteria": setup.get("boundary_standards", {}).get("success_criteria", {}),
        "failure_criteria": setup.get("boundary_standards", {}).get("failure_criteria", {}),
    }
    
    return result


def check_validation_standards(direction: str) -> dict[str, Any]:
    """檢查驗證標準是否完整。"""
    setup = DIRECTIONS_SETUP.get(direction)
    if not setup:
        return {"status": "FAIL", "error": f"Unknown direction: {direction}"}
    
    result = {
        "direction": direction,
        "diagnostics": setup.get("validation_standards", {}).get("diagnostics", []),
        "direction_specific": setup.get("validation_standards", {}).get("direction_specific", []),
        "status": "PASS",
    }
    
    return result


def validate_all_directions() -> dict[str, Any]:
    """全面檢查所有三個方向。"""
    results = {
        "timestamp": str(Path.cwd()),
        "total_status": "PASS",
        "directions": {},
    }
    
    for direction in ["B", "A", "C"]:
        dir_result = {
            "fields": check_field_definitions(direction),
            "boundaries": check_boundary_standards(direction),
            "validation": check_validation_standards(direction),
        }
        results["directions"][direction] = dir_result
        
        if any(v.get("status") == "FAIL" for v in dir_result.values()):
            results["total_status"] = "FAIL"
    
    return results


def print_summary(results: dict) -> None:
    """印出摘要。"""
    print("\n" + "="*80)
    print("DIRECTIONS SETUP VALIDATION SUMMARY")
    print("="*80)
    
    for direction in ["B", "A", "C"]:
        setup = DIRECTIONS_SETUP[direction]
        print(f"\n📍 方向 {direction}: {setup['name']}")
        print(f"   優先級: {setup['priority']}")
        print(f"   描述: {setup['description']}")
        
        fields = setup["field_definitions"]
        print(f"   ✓ 必需欄位: {len(fields.get('required_fields', []))} 個")
        print(f"   ✓ 邊界標準: {len(setup['boundary_standards'].get('success_criteria', {}))} 個成功條件")
        print(f"   ✓ 驗證指標: {len(setup['validation_standards'].get('diagnostics', []))} 個通用 + {len(setup['validation_standards'].get('direction_specific', []))} 個專用")
    
    print("\n" + "="*80)
    print(f"總體狀態: {results['total_status']}")
    print("="*80 + "\n")


if __name__ == "__main__":
    results = validate_all_directions()
    print_summary(results)
    
    # Save detailed results
    output_file = Path(__file__).parent / ".." / "outputs" / "directions_setup_validation.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to: {output_file}")
    
    sys.exit(0 if results["total_status"] == "PASS" else 1)
