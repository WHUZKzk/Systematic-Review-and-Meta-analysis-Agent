"""
BaseAgent — abstract base class for all agents in the SR-MA pipeline.

Three concrete subclasses:
  ExecutorAgent    — executes multi-step tasks (search, extraction, synthesis)
  ReviewerAgent    — independently reviews/screens items
  AdjudicatorAgent — resolves disagreements between Reviewers

Architecture principles (from design doc):
  - Agent is a Skill executor, NOT a knowledge container.
  - Every LLM call is Skill-Scoped (L1 manifests always present, L2 on activation).
  - Structured input → Skill-constrained LLM reasoning → Structured output.
  - Validation failures trigger up to LLM_MAX_RETRIES correction loops.
"""

from __future__ import annotations
import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from skill_layer import Skill, SkillLoader, SkillValidator, ValidationResult, get_loader
from llm_backend import LLMInterface, LLMResponse, PromptContext, Message, get_registry

logger = logging.getLogger(__name__)


# ── Task / Output types ───────────────────────────────────────────────────────

@dataclass
class TaskInstruction:
    """
    Instruction passed from the Protocol Layer to an Agent.
    Contains all data the agent needs to execute one stage/step.
    """
    stage: str                          # e.g. "search" | "screening"
    step_id: str                        # e.g. "step_1_pico_generation"
    skill_ids: List[str]                # Skills to load for this step
    input_data: Dict[str, Any]          # Structured input data
    tool_configs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StageOutput:
    """Structured output returned by an Agent to the Protocol Layer."""
    stage: str
    step_id: str
    agent_id: str
    success: bool
    data: Any                           # The primary output (parsed dict/list)
    reasoning_chain: str = ""           # LLM's reasoning text (for audit)
    validation_result: Optional[ValidationResult] = None
    flagged_for_human: bool = False
    warnings: List[str] = field(default_factory=list)
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── Log entry ─────────────────────────────────────────────────────────────────

@dataclass
class LogEntry:
    timestamp: str
    event_type: str                     # skill_loaded | llm_call | tool_call | validation | decision
    details: Dict[str, Any]


# ── Tool protocol ─────────────────────────────────────────────────────────────

class Tool(ABC):
    """Minimal interface that all Tool Layer implementations must satisfy."""

    @property
    @abstractmethod
    def tool_id(self) -> str: ...

    @abstractmethod
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]: ...


@dataclass
class ToolResult:
    tool_id: str
    success: bool
    data: Any
    error: str = ""


# ── BaseAgent ─────────────────────────────────────────────────────────────────

class BaseAgent(ABC):
    """
    Abstract base for all SR-MA agents.

    Subclasses override `execute()` with their specific multi-step logic.
    All LLM interaction goes through `call_llm()`, which handles:
      - Skill-scoped prompt assembly (L1 + L2 + L3)
      - Structured output parsing
      - Validation-driven correction loops
      - Full execution logging
    """

    def __init__(
        self,
        llm: LLMInterface,
        skill_loader: Optional[SkillLoader] = None,
        agent_id: Optional[str] = None,
    ):
        self.agent_id: str = agent_id or f"{self.agent_type}-{uuid.uuid4().hex[:8]}"
        self._llm = llm
        self._skill_loader = skill_loader or get_loader()
        self._validator = SkillValidator()

        self.loaded_skills: Dict[str, Skill] = {}          # skill_id → Skill
        self.available_tools: Dict[str, Tool] = {}         # tool_id → Tool
        self.execution_log: List[LogEntry] = []

    # ── Abstract interface ────────────────────────────────────────────────────

    @property
    @abstractmethod
    def agent_type(self) -> str:
        """One of: executor | reviewer | adjudicator"""

    @abstractmethod
    def execute(self, task: TaskInstruction) -> StageOutput:
        """Main entry point called by the Protocol Layer."""

    # ── Skill management ──────────────────────────────────────────────────────

    def load_skills(self, skill_ids: List[str]) -> None:
        """Load and cache the requested Skills from the SkillLoader."""
        for sid in skill_ids:
            if sid in self.loaded_skills:
                continue
            skill = self._skill_loader.get(sid)
            self.loaded_skills[sid] = skill
            self._log("skill_loaded", {"skill_id": sid, "version": skill.version})
            logger.debug("%s loaded skill: %s", self.agent_id, sid)

    def get_skill(self, skill_id: str) -> Skill:
        if skill_id not in self.loaded_skills:
            self.load_skills([skill_id])
        return self.loaded_skills[skill_id]

    # ── Tool management ───────────────────────────────────────────────────────

    def register_tools(self, tools: List[Tool]) -> None:
        for tool in tools:
            self.available_tools[tool.tool_id] = tool
            logger.debug("%s registered tool: %s", self.agent_id, tool.tool_id)

    def call_tool(self, tool_id: str, input_data: Dict[str, Any]) -> ToolResult:
        if tool_id not in self.available_tools:
            err = f"Tool '{tool_id}' not registered in agent {self.agent_id}"
            logger.error(err)
            return ToolResult(tool_id=tool_id, success=False, data=None, error=err)

        tool = self.available_tools[tool_id]
        try:
            result_data = tool.run(input_data)
            self._log("tool_call", {
                "tool_id": tool_id,
                "input_keys": list(input_data.keys()),
                "success": True,
            })
            return ToolResult(tool_id=tool_id, success=True, data=result_data)
        except Exception as exc:
            self._log("tool_call", {"tool_id": tool_id, "success": False, "error": str(exc)})
            logger.error("Tool %s failed: %s", tool_id, exc)
            return ToolResult(tool_id=tool_id, success=False, data=None, error=str(exc))

    # ── LLM call with Skill-scoped prompts ───────────────────────────────────

    def call_llm(
        self,
        context: PromptContext,
        active_skill_id: Optional[str] = None,
        output_format: str = "json",
    ) -> tuple[LLMResponse, Any]:
        """
        Make a Skill-scoped LLM call with validation + correction loops.

        Args:
            context:          PromptContext with L1/L2/L3 sections + task data
            active_skill_id:  The Skill to validate the output against (if any)
            output_format:    "json" or "text"

        Returns:
            (LLMResponse, validated_output)
            validated_output is the parsed dict/list for JSON, or raw string for text.
        """
        context.output_format = output_format
        active_skill = self.loaded_skills.get(active_skill_id) if active_skill_id else None

        from config import LLM_MAX_RETRIES

        response = self._llm.call(context)
        output = response.parsed if output_format == "json" else response.content

        self._log("llm_call", {
            "model": response.model,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "active_skill": active_skill_id,
            "attempt": 1,
        })

        if active_skill is None:
            return response, output

        # Validation + correction loop
        for attempt in range(1, LLM_MAX_RETRIES + 1):
            vr = self.validate_output(output, active_skill)
            self._log("validation", {
                "skill_id": active_skill_id,
                "passed": vr.passed,
                "hard_failures": vr.hard_failures,
                "soft_warnings": vr.soft_warnings,
                "attempt": attempt,
            })

            if vr.passed:
                break

            if not vr.should_retry:
                # Non-retry failure (flag_human / reject)
                logger.warning(
                    "%s validation failed (no retry): %s", active_skill_id, vr.hard_failures
                )
                break

            if attempt == LLM_MAX_RETRIES:
                logger.warning(
                    "%s still failing after %d retries: %s",
                    active_skill_id, LLM_MAX_RETRIES, vr.hard_failures,
                )
                break

            # Build correction prompt and retry
            correction = self._validator.build_correction_prompt(output, vr, active_skill)
            response = self._llm.call_with_correction(context, correction)
            output = response.parsed if output_format == "json" else response.content
            self._log("llm_call", {
                "model": response.model,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "active_skill": active_skill_id,
                "attempt": attempt + 1,
                "correction": True,
            })

        return response, output

    def validate_output(self, output: Any, skill: Skill) -> ValidationResult:
        """Validate `output` against `skill`'s validation rules."""
        return self._validator.validate(output, skill)

    # ── PromptContext helpers ─────────────────────────────────────────────────

    def build_prompt_context(
        self,
        active_skill_id: str,
        task_instruction: str,
        task_data_dict: Dict[str, Any],
        history: Optional[List[Message]] = None,
        extra_skill_ids: Optional[List[str]] = None,
    ) -> PromptContext:
        """
        Build a PromptContext for the given active skill and task data.

        - L1 manifests: all loaded skills
        - L2 protocol:  the active skill's full decision protocol
        - L3 references: resolved domain_references of the active skill
        """
        # L1: manifests of all loaded skills (always-on)
        l1_manifests = [
            s.render_l1_manifest()
            for s in self.loaded_skills.values()
        ]

        # L2: active skill's full protocol
        active_skill = self.loaded_skills.get(active_skill_id)
        l2_protocol = active_skill.render_l2_protocol() if active_skill else ""

        # L3: domain references from the active skill
        l3_references: List[str] = []
        if active_skill:
            for dr in active_skill.domain_references:
                ref_skill = self.loaded_skills.get(dr.ref_skill_id)
                if ref_skill is None:
                    try:
                        ref_skill = self._skill_loader.get(dr.ref_skill_id)
                    except KeyError:
                        logger.warning("L3 reference skill not found: %s", dr.ref_skill_id)
                        continue
                # Extract the specific entry from the domain skill's content
                content = ref_skill.decision_protocol.content
                if isinstance(content, dict) and dr.entry_key in content:
                    import yaml as _yaml
                    l3_references.append(
                        f"[{dr.ref_skill_id} / {dr.entry_key}]\n"
                        + _yaml.dump(
                            content[dr.entry_key],
                            allow_unicode=True,
                            default_flow_style=False,
                        )
                    )

        task_data_str = json.dumps(task_data_dict, ensure_ascii=False, indent=2)

        return PromptContext(
            l1_manifests=l1_manifests,
            l2_protocol=l2_protocol,
            l3_references=l3_references,
            task_instruction=task_instruction,
            task_data=task_data_str,
            history=history or [],
        )

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log(self, event_type: str, details: Dict[str, Any]) -> None:
        entry = LogEntry(
            timestamp=datetime.utcnow().isoformat(),
            event_type=event_type,
            details=details,
        )
        self.execution_log.append(entry)

    def dump_log(self, path: Optional[Path] = None) -> List[Dict]:
        """Dump execution log as a list of dicts (optionally write to file)."""
        log_data = [
            {"timestamp": e.timestamp, "event": e.event_type, **e.details}
            for e in self.execution_log
        ]
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(log_data, fh, ensure_ascii=False, indent=2)
        return log_data

    def reset_log(self) -> None:
        self.execution_log.clear()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"id={self.agent_id}, "
            f"model={self._llm.model}, "
            f"skills={list(self.loaded_skills.keys())})"
        )
