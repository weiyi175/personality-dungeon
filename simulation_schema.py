from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RunFamily(str, Enum):
    B1_SMOKE = "B1_SMOKE"
    B1H1_EXTENSION = "B1H1_EXTENSION"


class ControlCellType(str, Enum):
    BASELINE = "BASELINE"
    PURE_B1 = "PURE_B1"
    PURE_H = "PURE_H"
    HYBRID_BH = "HYBRID_BH"


class RunSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    run_id: str
    timestamp_utc: datetime
    seed: int

    run_family: RunFamily
    control_cell_type: ControlCellType
    is_extension: bool
    h_params_active: bool
    b_params_active: bool
    smoke_eligible: bool
    classification_reason: str

    b_event_dispatch_mode: str
    b_world_feedback_mode: str
    b_event_impact_mode: str
    b_event_reward_mode: str
    b_replicator_update_mode: str
    b_events_json: str
    b_event_dispatch_target_rate: float
    b_event_dispatch_fairness_window: int
    b_event_dispatch_fairness_tolerance: float

    h_memory_kernel: str
    h_selection_strength: float
    h_init_bias: float
    h_defaults_applied: bool

    fairness_fail_count: int
    boundary_hit_rate: float
    instability_warning_count: int
    mean_stage3_score: float
    mean_env_gamma: float
    entropy: float
    action_diversity: float

    gate_pass_smoke: Optional[bool] = None
    excluded_from_smoke_gate: bool
    exclusion_reason: Optional[str] = None

    @model_validator(mode="after")
    def validate_classification_logic(self) -> "RunSummary":
        if self.h_params_active:
            if self.run_family != RunFamily.B1H1_EXTENSION:
                raise ValueError(
                    "h_params_active=true requires run_family=B1H1_EXTENSION"
                )
            if not self.is_extension:
                raise ValueError("h_params_active=true requires is_extension=true")

        if self.run_family == RunFamily.B1_SMOKE and self.is_extension:
            raise ValueError("run_family=B1_SMOKE requires is_extension=false")

        if self.smoke_eligible:
            if self.run_family != RunFamily.B1_SMOKE:
                raise ValueError(
                    "smoke_eligible=true is only valid for run_family=B1_SMOKE"
                )
            if self.h_params_active:
                raise ValueError("smoke_eligible=true forbids h_params_active=true")
            if self.gate_pass_smoke is None:
                raise ValueError(
                    "smoke_eligible=true requires gate_pass_smoke to be set"
                )
            if self.excluded_from_smoke_gate:
                raise ValueError(
                    "smoke_eligible=true requires excluded_from_smoke_gate=false"
                )
        else:
            if self.gate_pass_smoke is not None:
                raise ValueError(
                    "smoke_eligible=false requires gate_pass_smoke to be null"
                )

        if self.excluded_from_smoke_gate and not self.exclusion_reason:
            raise ValueError(
                "excluded_from_smoke_gate=true requires exclusion_reason"
            )
        if not self.excluded_from_smoke_gate and self.exclusion_reason is not None:
            raise ValueError(
                "excluded_from_smoke_gate=false requires exclusion_reason=null"
            )

        return self


class AggregateCounts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    all_runs: int
    b1_smoke_runs: int
    extension_runs: int
    smoke_eligible_runs: int
    excluded_from_smoke_gate_runs: int


class SmokeGateAggregate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluated_runs: int
    pass_runs: int
    fail_runs: int
    pass_rate: float
    fairness_fail_total: int

    @model_validator(mode="after")
    def validate_smoke_math(self) -> "SmokeGateAggregate":
        if self.evaluated_runs != self.pass_runs + self.fail_runs:
            raise ValueError("evaluated_runs must equal pass_runs + fail_runs")

        if self.evaluated_runs == 0:
            if self.pass_rate != 0:
                raise ValueError("evaluated_runs=0 requires pass_rate=0")
        else:
            expected = self.pass_runs / self.evaluated_runs
            if abs(self.pass_rate - expected) > 1e-9:
                raise ValueError(
                    "pass_rate must equal pass_runs/evaluated_runs"
                )

        return self


class FamilyAggregate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n: int
    mean_stage3_score_avg: float
    mean_env_gamma_avg: float
    entropy_avg: float
    action_diversity_avg: float
    boundary_hit_rate_max: float


class CellCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    required_cells: List[ControlCellType]
    observed_cells: List[ControlCellType]
    missing_cells: List[ControlCellType]
    bridge_claim_complete: bool

    @model_validator(mode="after")
    def validate_coverage(self) -> "CellCoverage":
        req = set(self.required_cells)
        obs = set(self.observed_cells)
        missing = set(self.missing_cells)

        expected_missing = req - obs
        if missing != expected_missing:
            raise ValueError("missing_cells must equal required_cells - observed_cells")

        if self.bridge_claim_complete != (len(expected_missing) == 0):
            raise ValueError(
                "bridge_claim_complete must match whether missing_cells is empty"
            )

        return self


class QualityFlags(BaseModel):
    model_config = ConfigDict(extra="forbid")

    invalid_h_key_detected: bool
    mixed_summary_without_family_split: bool
    schema_validation_errors: int


class SummaryAggregate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    counts: AggregateCounts
    smoke_gate: SmokeGateAggregate
    by_run_family: Dict[RunFamily, FamilyAggregate]
    cell_coverage: CellCoverage
    quality_flags: QualityFlags


class MultiRunSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    summary_type: str
    protocol_name: str
    protocol_version: str
    generated_at_utc: datetime
    source_glob: str
    total_runs: int

    runs: List[RunSummary]
    aggregate: SummaryAggregate

    @model_validator(mode="after")
    def validate_top_level_consistency(self) -> "MultiRunSummary":
        run_count = len(self.runs)
        if self.total_runs != run_count:
            raise ValueError("total_runs must equal the number of runs")

        b1_smoke_runs = sum(r.run_family == RunFamily.B1_SMOKE for r in self.runs)
        extension_runs = sum(
            r.run_family == RunFamily.B1H1_EXTENSION for r in self.runs
        )
        smoke_eligible_runs = sum(r.smoke_eligible for r in self.runs)
        excluded_runs = sum(r.excluded_from_smoke_gate for r in self.runs)

        counts = self.aggregate.counts
        if counts.all_runs != run_count:
            raise ValueError("aggregate.counts.all_runs does not match runs length")
        if counts.b1_smoke_runs != b1_smoke_runs:
            raise ValueError(
                "aggregate.counts.b1_smoke_runs does not match recalculated value"
            )
        if counts.extension_runs != extension_runs:
            raise ValueError(
                "aggregate.counts.extension_runs does not match recalculated value"
            )
        if counts.smoke_eligible_runs != smoke_eligible_runs:
            raise ValueError(
                "aggregate.counts.smoke_eligible_runs does not match recalculated value"
            )
        if counts.excluded_from_smoke_gate_runs != excluded_runs:
            raise ValueError(
                "aggregate.counts.excluded_from_smoke_gate_runs does not match recalculated value"
            )

        if self.aggregate.smoke_gate.evaluated_runs != smoke_eligible_runs:
            raise ValueError(
                "aggregate.smoke_gate.evaluated_runs must equal smoke_eligible_runs"
            )

        for family in (RunFamily.B1_SMOKE, RunFamily.B1H1_EXTENSION):
            if family not in self.aggregate.by_run_family:
                raise ValueError(f"aggregate.by_run_family is missing key {family.value}")

        return self


def validate_summary_payload(payload: dict) -> MultiRunSummary:
    return MultiRunSummary.model_validate(payload)


class ProvenanceSummary(BaseModel):
    """Flexible provenance model for runtime/debug artifacts.

    This model intentionally allows extra keys to preserve forward compatibility
    with ad-hoc diagnostics while still enforcing core physical sanity checks.
    """

    model_config = ConfigDict(extra="ignore")

    seed: Optional[int] = None
    condition: Optional[str] = None
    dispatch_mode: Optional[str] = None
    world_mode: Optional[str] = None

    # Difficulty-related fields used across different provenance variants.
    difficulty_index_mean: Optional[float] = None
    mean_difficulty_index: Optional[float] = None

    # Common nested diagnostics containers.
    config: Optional[Dict[str, Any]] = None
    tail_diagnostics: Optional[Dict[str, Any]] = None
    round_diagnostics: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def validate_physical_ranges(self) -> "ProvenanceSummary":
        for field_name in ("difficulty_index_mean", "mean_difficulty_index"):
            value = getattr(self, field_name)
            if value is None:
                continue
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(
                    f"{field_name} must be in [0, 1], got {value}"
                )
        return self


def validate_provenance_payload(payload: dict) -> ProvenanceSummary:
    return ProvenanceSummary.model_validate(payload)


class LegacyGateResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    max_l1: Optional[int] = None
    min_healthy: Optional[int] = None
    healthy_threshold: Optional[float] = None
    l1_pass: Optional[bool] = None
    healthy_pass: Optional[bool] = None
    fairness_pass: Optional[bool] = None
    new_l1_pass: Optional[bool] = None
    invariant_neutrality_pass: Optional[bool] = None
    invariant_trigger_guard_pass: Optional[bool] = None
    invariant_overall_pass: Optional[bool] = None
    overall_pass: Optional[bool] = None


class LegacyOutcome(BaseModel):
    model_config = ConfigDict(extra="ignore")

    seed: int
    level: int
    s3: float
    turn: float
    gamma: float
    elapsed_sec: float


class LegacySummary(BaseModel):
    """Legacy smoke summary contract.

    Intended for JSON payloads shaped like:
    total_seeds/l1/l2/l3/outcomes/gate/stage/...
    """

    model_config = ConfigDict(extra="ignore")

    total_seeds: int
    l1: int
    l2: int
    l3: int
    healthy: Optional[int] = None
    marginal: Optional[int] = None
    fairness_fail_count: Optional[int] = None
    neutrality_fail_count: Optional[int] = None
    trigger_guard_fail_count: Optional[int] = None

    mean_s3: Optional[float] = None
    median_s3: Optional[float] = None
    p10_s3: Optional[float] = None
    mean_gamma: Optional[float] = None
    mean_event_neutrality_max_abs_mean: Optional[float] = None
    mean_event_trigger_guard_block_rate: Optional[float] = None

    new_l1: Optional[int] = None
    rescued: Optional[int] = None
    broke: Optional[int] = None

    gate: Optional[LegacyGateResult] = None
    stage: Optional[str] = None
    flow_lock: Optional[str] = None

    seeds: Optional[List[int]] = None
    outcomes: Optional[List[LegacyOutcome]] = None

    @model_validator(mode="after")
    def validate_legacy_consistency(self) -> "LegacySummary":
        if self.total_seeds < 0:
            raise ValueError("total_seeds must be >= 0")

        if self.l1 + self.l2 + self.l3 != self.total_seeds:
            raise ValueError("l1 + l2 + l3 must equal total_seeds")

        if self.healthy is not None and self.marginal is not None:
            if self.healthy + self.marginal > self.total_seeds:
                raise ValueError("healthy + marginal must be <= total_seeds")

        if self.seeds is not None:
            if self.total_seeds != len(self.seeds):
                raise ValueError("total_seeds must equal len(seeds)")
            if len(set(self.seeds)) != len(self.seeds):
                raise ValueError("seeds must be unique")

        if self.outcomes is not None:
            if self.total_seeds != len(self.outcomes):
                raise ValueError("total_seeds must equal len(outcomes)")

        if self.seeds is not None and self.outcomes is not None:
            outcome_seeds = [o.seed for o in self.outcomes]
            if set(outcome_seeds) != set(self.seeds):
                raise ValueError("outcomes.seed set must match seeds set")

        return self


def validate_legacy_summary_payload(payload: dict) -> LegacySummary:
    return LegacySummary.model_validate(payload)
