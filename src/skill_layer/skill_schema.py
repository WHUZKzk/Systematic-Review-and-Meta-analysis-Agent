"""
Skill Schema V1.0 — Python dataclass representation.

Each Skill is a YAML file; this module defines the in-memory structure
that all other components work with.

Disclosure levels (渐进式披露):
  L1_manifest   ~200 tok — always present in system prompt
  L2_active     ~800 tok — loaded when the step is active
  L3_reference  ~300 tok — loaded on demand via domain_references
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── Sub-structures ────────────────────────────────────────────────────────────

@dataclass
class Trigger:
    stage: str              # search | screening | extraction | synthesis
    step: str               # step ID within the stage, e.g. "step_1_pico_generation"
    disclosure_level: str   # L1_manifest | L2_active | L3_reference


@dataclass
class RequiredContext:
    context_key: str        # e.g. "review_question.pico"
    source: str             # human_input | stage_output | skill_output
    description: str


@dataclass
class DecisionProtocol:
    type: str               # decision_tree | template | checklist | schema
    token_budget: int
    content: Any            # Raw dict — structure varies by type


@dataclass
class OutputSchema:
    format: str             # json | yaml | markdown_structured
    schema: Dict            # JSON Schema dict
    example: Any = None     # Optional example output


@dataclass
class ValidationRule:
    rule_id: str
    description: str
    expression: str         # Python-style pseudo-code evaluated at runtime
    severity: str           # hard | soft
    on_fail: str            # retry | flag_human | reject


@dataclass
class BoundTool:
    tool_id: str
    usage: str
    required: bool = False


@dataclass
class DomainReference:
    ref_skill_id: str       # ID of the referenced Domain Knowledge Skill
    entry_key: str          # Specific entry within that skill
    injection_point: str    # decision_protocol | system_context


@dataclass
class Manifest:
    purpose: str
    key_constraints: List[str]
    output_type: str


# ── Main Skill dataclass ───────────────────────────────────────────────────────

@dataclass
class Skill:
    # Meta
    skill_id: str
    version: str
    domain: str
    skill_type: str         # methodological | domain_knowledge | template
    evidence_base: str
    description: str

    # Trigger & binding
    trigger: Trigger
    required_context: List[RequiredContext]

    # Core content (injected into LLM)
    decision_protocol: DecisionProtocol
    output_schema: OutputSchema

    # Validation (executed by code, not LLM)
    validation: List[ValidationRule]

    # L1 manifest (always in system prompt)
    manifest: Manifest

    # Optional
    bound_tools: List[BoundTool] = field(default_factory=list)
    domain_references: List[DomainReference] = field(default_factory=list)

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, data: dict) -> "Skill":
        """Parse a Skill from a YAML-loaded dict."""
        trigger_raw = data.get("trigger", {})
        trigger = Trigger(
            stage=trigger_raw.get("stage", ""),
            step=trigger_raw.get("step", ""),
            disclosure_level=trigger_raw.get("disclosure_level", "L2_active"),
        )

        required_context = [
            RequiredContext(
                context_key=rc.get("context_key", ""),
                source=rc.get("source", ""),
                description=rc.get("description", ""),
            )
            for rc in data.get("required_context", [])
        ]

        dp_raw = data.get("decision_protocol", {})
        decision_protocol = DecisionProtocol(
            type=dp_raw.get("type", "template"),
            token_budget=dp_raw.get("token_budget", 800),
            content=dp_raw.get("content", {}),
        )

        os_raw = data.get("output_schema", {})
        output_schema = OutputSchema(
            format=os_raw.get("format", "json"),
            schema=os_raw.get("schema", {}),
            example=os_raw.get("example"),
        )

        validation = [
            ValidationRule(
                rule_id=vr.get("rule_id", ""),
                description=vr.get("description", ""),
                expression=vr.get("expression", "True"),
                severity=vr.get("severity", "soft"),
                on_fail=vr.get("on_fail", "flag_human"),
            )
            for vr in data.get("validation", [])
        ]

        mf_raw = data.get("manifest", {})
        manifest = Manifest(
            purpose=mf_raw.get("purpose", ""),
            key_constraints=mf_raw.get("key_constraints", []),
            output_type=mf_raw.get("output_type", ""),
        )

        bound_tools = [
            BoundTool(
                tool_id=bt.get("tool_id", ""),
                usage=bt.get("usage", ""),
                required=bt.get("required", False),
            )
            for bt in data.get("bound_tools", [])
        ]

        domain_references = [
            DomainReference(
                ref_skill_id=dr.get("ref_skill_id", ""),
                entry_key=dr.get("entry_key", ""),
                injection_point=dr.get("injection_point", "system_context"),
            )
            for dr in data.get("domain_references", [])
        ]

        return cls(
            skill_id=data["skill_id"],
            version=data.get("version", "1.0.0"),
            domain=data.get("domain", ""),
            skill_type=data.get("skill_type", "methodological"),
            evidence_base=data.get("evidence_base", ""),
            description=data.get("description", ""),
            trigger=trigger,
            required_context=required_context,
            decision_protocol=decision_protocol,
            output_schema=output_schema,
            validation=validation,
            manifest=manifest,
            bound_tools=bound_tools,
            domain_references=domain_references,
        )

    # ── Rendering helpers ─────────────────────────────────────────────────────

    def render_l1_manifest(self) -> str:
        """Render the L1 manifest (~200 tok) for always-on system prompt inclusion."""
        constraints = "\n".join(f"  - {c}" for c in self.manifest.key_constraints)
        return (
            f"[SKILL: {self.skill_id}]\n"
            f"Purpose: {self.manifest.purpose}\n"
            f"Output: {self.manifest.output_type}\n"
            f"Key constraints:\n{constraints}"
        )

    def render_l2_protocol(self) -> str:
        """Render the full L2 decision_protocol for injection when the step is active."""
        import yaml as _yaml
        protocol_text = _yaml.dump(
            self.decision_protocol.content,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )
        schema_text = _yaml.dump(
            self.output_schema.schema,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )
        example_text = ""
        if self.output_schema.example:
            example_text = (
                "\n## Output Example\n"
                + _yaml.dump(
                    self.output_schema.example,
                    allow_unicode=True,
                    default_flow_style=False,
                    sort_keys=False,
                )
            )
        return (
            f"# SKILL: {self.skill_id} (v{self.version})\n"
            f"## Description\n{self.description}\n\n"
            f"## Decision Protocol ({self.decision_protocol.type})\n"
            f"{protocol_text}\n"
            f"## Output Schema ({self.output_schema.format})\n"
            f"{schema_text}"
            f"{example_text}\n"
            f"## Evidence Base\n{self.evidence_base}"
        )
