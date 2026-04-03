"""
ReviewerAgent — independent dual-review agent for systematic screening.

Key invariants (enforced here):
  1. A ReviewerAgent NEVER sees another reviewer's decisions or reasoning.
  2. No external tool calls — decisions are based purely on LLM reading of
     title/abstract against the eligibility criteria.
  3. Papers are processed in configurable batches to reduce API costs.
  4. The same Skill (screening_decision_logic) is loaded by both reviewers,
     but each uses a different LLM backend (heterogeneity policy).

Two modes:
  - title   : only title text, liberal pass (exclude only obviously irrelevant)
  - abstract: title + abstract, full criteria assessment
"""

from __future__ import annotations
import json
import logging
from typing import Any, Dict, List, Optional

from agent_layer.base_agent import BaseAgent, TaskInstruction, StageOutput
from llm_backend import LLMInterface, PromptContext
from skill_layer import SkillLoader

logger = logging.getLogger(__name__)

SCREENING_SKILL_ID = "screening.screening_decision_logic"


class ReviewerAgent(BaseAgent):
    """
    Screening reviewer that independently assesses papers.

    Args:
        llm:          LLM backend (must differ from the other reviewer's backend)
        reviewer_id:  Human-readable label, e.g. "reviewer_alpha"
        batch_size:   Papers per single LLM call (title: 20, abstract: 5 recommended)
    """

    def __init__(
        self,
        llm: LLMInterface,
        reviewer_id: str = "reviewer",
        batch_size_title: int = 20,
        batch_size_abstract: int = 5,
        skill_loader: Optional[SkillLoader] = None,
    ):
        super().__init__(llm=llm, skill_loader=skill_loader, agent_id=reviewer_id)
        self.batch_size_title = batch_size_title
        self.batch_size_abstract = batch_size_abstract

    @property
    def agent_type(self) -> str:
        return "reviewer"

    # ── Main execute ──────────────────────────────────────────────────────────

    def execute(self, task: TaskInstruction) -> StageOutput:
        """
        Screen a list of papers.

        task.input_data must contain:
            papers            : List[Dict]  [{pmid, title, abstract?}, ...]
            eligibility_criteria: Dict      confirmed criteria from step 1
            screening_mode    : str         "title" or "abstract"
        """
        papers: List[Dict] = task.input_data.get("papers", [])
        criteria: Dict = task.input_data.get("eligibility_criteria", {})
        mode: str = task.input_data.get("screening_mode", "abstract")

        if not papers:
            return StageOutput(
                stage=task.stage, step_id=task.step_id,
                agent_id=self.agent_id, success=True,
                data={"decisions": []},
            )

        # Load the screening skill
        self.load_skills([SCREENING_SKILL_ID])

        batch_size = self.batch_size_title if mode == "title" else self.batch_size_abstract
        all_decisions: List[Dict] = []
        warnings: List[str] = []

        for i in range(0, len(papers), batch_size):
            batch = papers[i : i + batch_size]
            logger.info(
                "%s screening batch %d/%d (%s mode, %d papers)",
                self.agent_id,
                i // batch_size + 1,
                (len(papers) + batch_size - 1) // batch_size,
                mode,
                len(batch),
            )
            try:
                batch_decisions = self._screen_batch(batch, criteria, mode)
                all_decisions.extend(batch_decisions)
            except Exception as exc:
                logger.error("%s batch %d failed: %s", self.agent_id, i // batch_size, exc)
                # On failure, mark all papers in batch as UNCERTAIN
                for paper in batch:
                    all_decisions.append({
                        "pmid": paper.get("pmid", "unknown"),
                        "decision": "UNCERTAIN",
                        "reasoning": f"Screening failed due to error: {exc}",
                        "criteria_assessment": [],
                    })
                warnings.append(f"Batch {i//batch_size + 1} failed: {exc}")

        return StageOutput(
            stage=task.stage,
            step_id=task.step_id,
            agent_id=self.agent_id,
            success=True,
            data={"decisions": all_decisions, "mode": mode},
            warnings=warnings,
        )

    # ── Batch screening ───────────────────────────────────────────────────────

    def _screen_batch(
        self,
        papers: List[Dict],
        criteria: Dict,
        mode: str,
    ) -> List[Dict]:
        """Screen one batch of papers; returns list of decision dicts."""
        task_instruction = self._build_screening_instruction(mode)
        task_data = self._build_batch_data(papers, criteria, mode)

        context = self.build_prompt_context(
            active_skill_id=SCREENING_SKILL_ID,
            task_instruction=task_instruction,
            task_data_dict=task_data,
        )

        _response, output = self.call_llm(
            context=context,
            active_skill_id=SCREENING_SKILL_ID,
            output_format="json",
        )

        if not isinstance(output, dict) or "decisions" not in output:
            # Try to recover if LLM returned a list directly
            if isinstance(output, list):
                output = {"decisions": output}
            else:
                logger.warning(
                    "%s unexpected output structure, using fallback UNCERTAIN",
                    self.agent_id,
                )
                return [
                    {"pmid": p["pmid"], "decision": "UNCERTAIN",
                     "reasoning": "LLM output format error", "criteria_assessment": []}
                    for p in papers
                ]

        decisions = output.get("decisions", [])

        # Ensure every paper in the batch has a decision (fill missing with UNCERTAIN)
        returned_pmids = {d.get("pmid") for d in decisions}
        for paper in papers:
            if paper["pmid"] not in returned_pmids:
                decisions.append({
                    "pmid": paper["pmid"],
                    "decision": "UNCERTAIN",
                    "reasoning": "Not assessed by LLM (missing from output)",
                    "criteria_assessment": [],
                })

        return decisions

    # ── Prompt builders ───────────────────────────────────────────────────────

    def _build_screening_instruction(self, mode: str) -> str:
        if mode == "title":
            return (
                "You are an independent systematic review screener performing TITLE screening.\n"
                "For each paper below, decide: INCLUDE (possibly relevant), EXCLUDE (definitely irrelevant), "
                "or UNCERTAIN (cannot tell from title alone).\n"
                "At this stage be LIBERAL — only exclude papers that are obviously irrelevant.\n"
                "Default to UNCERTAIN if unsure. Provide a brief reasoning for each decision."
            )
        return (
            "You are an independent systematic review screener performing ABSTRACT screening.\n"
            "For each paper, carefully evaluate the title AND abstract against every eligibility criterion.\n"
            "Apply ALL binary questions. Any criterion answered NO → EXCLUDE.\n"
            "All YES → INCLUDE. Any UNCERTAIN with no NO → overall UNCERTAIN.\n"
            "Provide reasoning and per-criterion assessment for every paper."
        )

    def _build_batch_data(
        self,
        papers: List[Dict],
        criteria: Dict,
        mode: str,
    ) -> Dict[str, Any]:
        papers_data = []
        for p in papers:
            entry: Dict[str, Any] = {"pmid": p.get("pmid", ""), "title": p.get("title", "")}
            if mode == "abstract":
                entry["abstract"] = p.get("abstract", "")
            papers_data.append(entry)

        # Build a compact criteria summary for the prompt
        criteria_summary = self._format_criteria(criteria)

        return {
            "screening_mode": mode,
            "eligibility_criteria_summary": criteria_summary,
            "papers_to_screen": papers_data,
        }

    def _format_criteria(self, criteria: Dict) -> Dict[str, Any]:
        """Format criteria as a compact dict for inclusion in the prompt."""
        binary_questions: List[Dict] = []
        ic = criteria.get("inclusion_criteria", {})
        for dim in ["population", "intervention", "comparison", "outcome", "study_design"]:
            dim_data = ic.get(dim, {})
            q = dim_data.get("binary_question", "")
            if q:
                binary_questions.append({"criterion_id": dim, "question": q})
        for add in ic.get("additional", []):
            q = add.get("binary_question", "")
            if q:
                binary_questions.append({"criterion_id": "additional", "question": q})

        exclusion_questions = [
            {"criterion_id": f"exclusion_{i}", "question": ex.get("binary_question", "")}
            for i, ex in enumerate(criteria.get("exclusion_criteria", []))
        ]

        return {
            "inclusion_binary_questions": binary_questions,
            "exclusion_binary_questions": exclusion_questions,
            "guidance": criteria.get("screening_guidance", {}),
        }
