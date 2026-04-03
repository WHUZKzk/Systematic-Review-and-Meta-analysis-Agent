"""
ExecutorAgent — execution-type agent for multi-step tasks.

Handles: search stage, extraction stage, synthesis stage.
Principle: Agent is a Skill executor, NOT a knowledge container.
           All domain logic lives in Skill YAMLs.

Step execution logic:
  - If requires_llm=True  → build PromptContext + call_llm with Skill
  - If requires_llm=False → delegate to registered step_handler (tool / pure code)
  - Skill is always loaded (for L1 manifest) even if LLM is not called
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from agent_layer.base_agent import BaseAgent, TaskInstruction, StageOutput, Tool
from llm_backend import LLMInterface
from skill_layer import SkillLoader

logger = logging.getLogger(__name__)


# ── Step definition ────────────────────────────────────────────────────────────

@dataclass
class StepSpec:
    """Specification for one step within an ExecutorAgent task."""
    step_id: str
    skill_id: Optional[str]            # None → no LLM, pure code/tool
    requires_llm: bool
    tools_required: List[str] = field(default_factory=list)
    description: str = ""


# ── ExecutorAgent ─────────────────────────────────────────────────────────────

class ExecutorAgent(BaseAgent):
    """
    Multi-step executor agent.

    Usage:
        agent = ExecutorAgent(llm=llm_interface)
        agent.register_tools([pubmed_tool, mesh_tool, ...])
        agent.register_step_handlers({
            "search_step_2": handler_fn,   # for non-LLM steps
            ...
        })
        output = agent.execute(task)

    Step handlers:
        A step handler is called for non-LLM steps (requires_llm=False).
        Signature: handler(step_context: dict, agent: ExecutorAgent) -> dict
        The return value becomes the step's output data.
    """

    @property
    def agent_type(self) -> str:
        return "executor"

    def __init__(
        self,
        llm: LLMInterface,
        skill_loader: Optional[SkillLoader] = None,
        agent_id: Optional[str] = None,
    ):
        super().__init__(llm=llm, skill_loader=skill_loader, agent_id=agent_id)
        # step_id → handler(step_context, agent) → dict
        self._step_handlers: Dict[str, Callable] = {}

    def register_step_handlers(self, handlers: Dict[str, Callable]) -> None:
        """Register code/tool handlers for non-LLM steps."""
        self._step_handlers.update(handlers)
        logger.debug("%s registered %d step handlers", self.agent_id, len(handlers))

    # ── Core execute ──────────────────────────────────────────────────────────

    def execute(self, task: TaskInstruction) -> StageOutput:
        """
        Execute a multi-step task.

        `task.input_data` must contain:
            "step_sequence": List[StepSpec]  — ordered list of steps
            "shared_context": dict           — initial shared data (grows as steps complete)

        Returns StageOutput with final aggregated data.
        """
        step_sequence: List[StepSpec] = task.input_data.get("step_sequence", [])
        shared_context: Dict[str, Any] = dict(task.input_data.get("shared_context", {}))

        if not step_sequence:
            return StageOutput(
                stage=task.stage,
                step_id="",
                agent_id=self.agent_id,
                success=False,
                data={},
                error="No step_sequence provided in task.input_data",
            )

        # Pre-load all skills that will be needed
        skill_ids = [s.skill_id for s in step_sequence if s.skill_id]
        if skill_ids:
            try:
                self.load_skills(skill_ids)
            except Exception as exc:
                logger.warning("Some skills failed to load: %s", exc)

        all_warnings: List[str] = []
        last_step_output: Dict[str, Any] = {}

        for step in step_sequence:
            logger.info(
                "%s executing step [%s] — %s",
                self.agent_id, step.step_id, step.description,
            )
            try:
                if step.requires_llm:
                    step_output = self._execute_llm_step(step, shared_context)
                else:
                    step_output = self._execute_code_step(step, shared_context)
            except Exception as exc:
                logger.error("Step %s failed with exception: %s", step.step_id, exc)
                return StageOutput(
                    stage=task.stage,
                    step_id=step.step_id,
                    agent_id=self.agent_id,
                    success=False,
                    data=shared_context,
                    error=f"Step {step.step_id} failed: {exc}",
                    warnings=all_warnings,
                )

            # Merge step output into shared context
            if isinstance(step_output, dict):
                shared_context[step.step_id] = step_output
                last_step_output = step_output
                # Propagate any warnings from this step
                step_warnings = step_output.pop("_warnings", [])
                all_warnings.extend(step_warnings)

            self._log("step_complete", {
                "step_id": step.step_id,
                "requires_llm": step.requires_llm,
                "output_keys": list(step_output.keys()) if isinstance(step_output, dict) else [],
            })

        return StageOutput(
            stage=task.stage,
            step_id=task.step_id,
            agent_id=self.agent_id,
            success=True,
            data=shared_context,
            warnings=all_warnings,
        )

    # ── LLM step execution ────────────────────────────────────────────────────

    def _execute_llm_step(
        self, step: StepSpec, shared_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a step that requires an LLM call."""
        skill = self.loaded_skills.get(step.skill_id) if step.skill_id else None

        # Build task instruction and data for this step
        task_instruction = self._build_step_instruction(step, skill)
        task_data = self._build_step_data(step, shared_context)

        if skill:
            context = self.build_prompt_context(
                active_skill_id=step.skill_id,
                task_instruction=task_instruction,
                task_data_dict=task_data,
            )
        else:
            # No skill: bare LLM call
            from llm_backend import PromptContext
            context = PromptContext(
                task_instruction=task_instruction,
                task_data=str(task_data),
                output_format="json",
            )

        _response, output = self.call_llm(
            context=context,
            active_skill_id=step.skill_id,
            output_format="json",
        )

        if not isinstance(output, dict):
            logger.warning("LLM output for step %s is not a dict: %s", step.step_id, type(output))
            output = {"raw_output": output}

        return output

    # ── Code/tool step execution ──────────────────────────────────────────────

    def _execute_code_step(
        self, step: StepSpec, shared_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a step that is handled by a registered step handler."""
        handler = self._step_handlers.get(step.step_id)
        if handler is None:
            raise ValueError(
                f"No step handler registered for step '{step.step_id}'. "
                f"Available handlers: {list(self._step_handlers.keys())}"
            )
        step_context = {
            "step": step,
            "shared_context": shared_context,
        }
        result = handler(step_context, self)
        return result if isinstance(result, dict) else {"result": result}

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_step_instruction(
        self, step: StepSpec, skill=None
    ) -> str:
        if skill:
            return (
                f"Execute step '{step.step_id}': {step.description}\n"
                f"Follow the decision protocol defined in skill '{step.skill_id}' exactly."
            )
        return f"Execute step '{step.step_id}': {step.description}"

    def _build_step_data(
        self, step: StepSpec, shared_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build the data dict for an LLM step.
        Includes: review_question from context + outputs of previous steps.
        """
        data: Dict[str, Any] = {}
        # Always include the review question if available
        if "review_question" in shared_context:
            data["review_question"] = shared_context["review_question"]

        # Include outputs of previous steps that this step might need
        # (include all steps completed so far — Protocol Layer would filter via data_contracts)
        for key, val in shared_context.items():
            if key != "step_sequence" and key not in data:
                data[key] = val

        return data
