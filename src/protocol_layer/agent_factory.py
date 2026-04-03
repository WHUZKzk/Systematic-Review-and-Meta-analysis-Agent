"""
AgentFactory — creates the correct agent instances for a stage based on
the Meta-Skill's llm_assignment and agent_type declarations.

Agent type mapping:
  "executor"          → ExecutorAgent with primary LLM
  "dual_review"       → ReviewerAgent pair (A + B) + AdjudicatorAgent
  "executor_verified" → ExecutorAgent with primary LLM (same as executor,
                        verification handled inside the pipeline step)

The factory does NOT register step handlers — those are the responsibility
of the specific pipeline module (search_pipeline, screening_pipeline, etc.).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from agent_layer import ExecutorAgent, ReviewerAgent, AdjudicatorAgent
from llm_backend import BackendRegistry, get_registry
from skill_layer import SkillLoader, get_loader
from protocol_layer.meta_skill_parser import StageDef

logger = logging.getLogger(__name__)


class AgentFactory:
    """
    Creates agent instances from a StageDef.

    Args:
        registry:     LLM backend registry (singleton by default)
        skill_loader: Skill loader (singleton by default)
    """

    def __init__(
        self,
        registry: Optional[BackendRegistry] = None,
        skill_loader: Optional[SkillLoader] = None,
    ):
        self._registry = registry or get_registry()
        self._skill_loader = skill_loader or get_loader()

    # ── Public API ────────────────────────────────────────────────────────

    def create_executor(self, stage: StageDef) -> ExecutorAgent:
        """Create an ExecutorAgent for the given stage."""
        llm = self._registry.get(stage.llm_assignment.primary)
        agent = ExecutorAgent(
            llm=llm,
            skill_loader=self._skill_loader,
        )
        logger.debug(
            "AgentFactory: created ExecutorAgent for stage '%s' (model=%s)",
            stage.stage_id, stage.llm_assignment.primary,
        )
        return agent

    def create_dual_review_pair(
        self, stage: StageDef
    ) -> Tuple[ReviewerAgent, ReviewerAgent, AdjudicatorAgent]:
        """
        Create heterogeneous reviewer pair + adjudicator for dual-review stages.

        Returns:
            (reviewer_a, reviewer_b, adjudicator)
        """
        llm_a, llm_b = self._registry.get_dual_review_pair()
        adjudicator_llm = self._registry.get_adjudicator()

        reviewer_a = ReviewerAgent(
            llm=llm_a,
            reviewer_id="reviewer_alpha",
            skill_loader=self._skill_loader,
        )
        reviewer_b = ReviewerAgent(
            llm=llm_b,
            reviewer_id="reviewer_beta",
            skill_loader=self._skill_loader,
        )
        adjudicator = AdjudicatorAgent(
            llm=adjudicator_llm,
            skill_loader=self._skill_loader,
        )

        logger.debug(
            "AgentFactory: created dual-review pair for stage '%s' "
            "(reviewer_a=%s, reviewer_b=%s, adjudicator=%s)",
            stage.stage_id,
            stage.llm_assignment.primary,
            stage.llm_assignment.secondary,
            stage.llm_assignment.adjudicator,
        )
        return reviewer_a, reviewer_b, adjudicator

    def create_for_stage(self, stage: StageDef) -> Dict[str, Any]:
        """
        Dispatch to the appropriate creation method based on agent_type.

        Returns a dict with keys matching the agent roles:
          - "executor"  for executor / executor_verified stages
          - "reviewer_a", "reviewer_b", "adjudicator"  for dual_review stages
        """
        if stage.agent_type == "dual_review":
            ra, rb, adj = self.create_dual_review_pair(stage)
            return {
                "reviewer_a": ra,
                "reviewer_b": rb,
                "adjudicator": adj,
            }
        else:
            return {"executor": self.create_executor(stage)}
