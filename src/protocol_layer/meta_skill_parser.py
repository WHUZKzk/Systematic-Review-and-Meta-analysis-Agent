"""
MetaSkillParser — loads and validates a Meta-Skill YAML into dataclasses.

The Meta-Skill YAML is the single authoritative declaration of the full
SR pipeline: stages, step sequences, data contracts, quality gates,
human checkpoints, and error policy.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class StepDef:
    step_id: str
    skill_id: Optional[str]
    tools_required: List[str]
    requires_llm: bool
    input_from: str          # "human_input" | "previous_step" | "stage_input" | "context_store"
    output_to: str           # "next_step" | "stage_output" | "context_store"
    description: str


@dataclass
class LLMAssignment:
    primary: str
    secondary: Optional[str] = None
    adjudicator: Optional[str] = None


@dataclass
class HumanCheckpoint:
    timing: str
    blocking: bool
    condition: str
    deliverable: str


@dataclass
class StageDef:
    stage_id: str
    stage_name: str
    agent_type: str          # "executor" | "dual_review" | "executor_verified"
    step_sequence: List[StepDef]
    llm_assignment: LLMAssignment
    human_checkpoints: List[HumanCheckpoint]


@dataclass
class DataContractField:
    field_name: str
    field_type: str          # "value" | "reference" | "summary"
    max_tokens: Optional[int]
    description: str


@dataclass
class DataContract:
    from_stage: str
    to_stage: str
    required_fields: List[DataContractField]
    optional_fields: List[Dict[str, str]]


@dataclass
class QualityGate:
    stage_id: str
    timing: str
    metric: str
    threshold: float
    operator: str            # ">=" | "<=" | "==" | ">"
    on_pass: str
    on_fail: str
    description: str


@dataclass
class FeedbackLoop:
    loop_id: str
    trigger_stage: str
    trigger_condition: str
    target_stage: str
    target_action: str
    max_iterations: int
    description: str


@dataclass
class ErrorPolicy:
    llm_call_timeout_seconds: int = 120
    llm_call_max_retries: int = 2
    tool_call_max_retries: int = 3
    on_unrecoverable_error: str = "halt_and_notify"


@dataclass
class MetaSkill:
    meta_skill_id: str
    version: str
    applicable_to: str
    evidence_base: str
    stages: List[StageDef]
    data_contracts: List[DataContract]
    quality_gates: List[QualityGate]
    feedback_loops: List[FeedbackLoop]
    error_policy: ErrorPolicy

    # ── Lookup helpers ─────────────────────────────────────────────────────

    def get_stage(self, stage_id: str) -> Optional[StageDef]:
        for s in self.stages:
            if s.stage_id == stage_id:
                return s
        return None

    def get_quality_gates(self, stage_id: str, timing: Optional[str] = None) -> List[QualityGate]:
        gates = [g for g in self.quality_gates if g.stage_id == stage_id]
        if timing:
            gates = [g for g in gates if g.timing == timing]
        return gates

    def get_human_checkpoints(self, stage_id: str) -> List[HumanCheckpoint]:
        stage = self.get_stage(stage_id)
        return stage.human_checkpoints if stage else []

    def get_data_contract(self, from_stage: str, to_stage: str) -> Optional[DataContract]:
        for dc in self.data_contracts:
            if dc.from_stage == from_stage and dc.to_stage == to_stage:
                return dc
        return None


# ── Parser ────────────────────────────────────────────────────────────────────

class MetaSkillParser:
    """Loads and parses a Meta-Skill YAML file."""

    @classmethod
    def load(cls, path: pathlib.Path) -> MetaSkill:
        with open(path, "r", encoding="utf-8") as fh:
            raw: Dict[str, Any] = yaml.safe_load(fh)
        return cls._parse(raw)

    @classmethod
    def load_from_str(cls, content: str) -> MetaSkill:
        raw = yaml.safe_load(content)
        return cls._parse(raw)

    @classmethod
    def _parse(cls, raw: Dict[str, Any]) -> MetaSkill:
        stages = [cls._parse_stage(s) for s in raw.get("stages", [])]
        data_contracts = [cls._parse_contract(dc) for dc in raw.get("data_contracts", [])]
        quality_gates = [cls._parse_gate(g) for g in raw.get("quality_gates", [])]
        feedback_loops = [cls._parse_loop(lp) for lp in raw.get("feedback_loops", [])]
        error_policy = cls._parse_error_policy(raw.get("error_policy", {}))

        return MetaSkill(
            meta_skill_id=raw["meta_skill_id"],
            version=raw.get("version", "1.0.0"),
            applicable_to=raw.get("applicable_to", ""),
            evidence_base=raw.get("evidence_base", ""),
            stages=stages,
            data_contracts=data_contracts,
            quality_gates=quality_gates,
            feedback_loops=feedback_loops,
            error_policy=error_policy,
        )

    @classmethod
    def _parse_stage(cls, raw: Dict) -> StageDef:
        steps = [cls._parse_step(s) for s in raw.get("step_sequence", [])]
        llm_raw = raw.get("llm_assignment", {})
        llm = LLMAssignment(
            primary=llm_raw.get("primary", "default"),
            secondary=llm_raw.get("secondary"),
            adjudicator=llm_raw.get("adjudicator"),
        )
        checkpoints = [
            HumanCheckpoint(
                timing=cp["timing"],
                blocking=cp.get("blocking", True),
                condition=cp.get("condition", "always"),
                deliverable=cp.get("deliverable", ""),
            )
            for cp in raw.get("human_checkpoints", [])
        ]
        return StageDef(
            stage_id=raw["stage_id"],
            stage_name=raw.get("stage_name", ""),
            agent_type=raw.get("agent_type", "executor"),
            step_sequence=steps,
            llm_assignment=llm,
            human_checkpoints=checkpoints,
        )

    @classmethod
    def _parse_step(cls, raw: Dict) -> StepDef:
        return StepDef(
            step_id=raw["step_id"],
            skill_id=raw.get("skill_id"),
            tools_required=raw.get("tools_required", []),
            requires_llm=raw.get("requires_llm", False),
            input_from=raw.get("input_from", "previous_step"),
            output_to=raw.get("output_to", "next_step"),
            description=raw.get("description", ""),
        )

    @classmethod
    def _parse_contract(cls, raw: Dict) -> DataContract:
        required = [
            DataContractField(
                field_name=f["field_name"],
                field_type=f.get("field_type", "value"),
                max_tokens=f.get("max_tokens"),
                description=f.get("description", ""),
            )
            for f in raw.get("required_fields", [])
        ]
        optional = raw.get("optional_fields", [])
        return DataContract(
            from_stage=raw["from_stage"],
            to_stage=raw["to_stage"],
            required_fields=required,
            optional_fields=optional,
        )

    @classmethod
    def _parse_gate(cls, raw: Dict) -> QualityGate:
        return QualityGate(
            stage_id=raw["stage_id"],
            timing=raw["timing"],
            metric=raw["metric"],
            threshold=float(raw["threshold"]),
            operator=raw.get("operator", ">="),
            on_pass=raw.get("on_pass", "continue"),
            on_fail=raw.get("on_fail", "warn_and_continue"),
            description=raw.get("description", ""),
        )

    @classmethod
    def _parse_loop(cls, raw: Dict) -> FeedbackLoop:
        return FeedbackLoop(
            loop_id=raw["loop_id"],
            trigger_stage=raw["trigger_stage"],
            trigger_condition=raw.get("trigger_condition", ""),
            target_stage=raw["target_stage"],
            target_action=raw.get("target_action", ""),
            max_iterations=int(raw.get("max_iterations", 1)),
            description=raw.get("description", ""),
        )

    @classmethod
    def _parse_error_policy(cls, raw: Dict) -> ErrorPolicy:
        return ErrorPolicy(
            llm_call_timeout_seconds=int(raw.get("llm_call_timeout_seconds", 120)),
            llm_call_max_retries=int(raw.get("llm_call_max_retries", 2)),
            tool_call_max_retries=int(raw.get("tool_call_max_retries", 3)),
            on_unrecoverable_error=raw.get("on_unrecoverable_error", "halt_and_notify"),
        )
