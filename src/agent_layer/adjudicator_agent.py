"""
AdjudicatorAgent — resolves dual-review disagreements.

Key invariants (enforced here):
  1. Sees BOTH reviewers' decisions and reasoning (fully transparent to adjudicator).
  2. Does NOT know which reviewer is "A" or "B" — labels are anonymized to "Reviewer 1/2".
  3. Uses a THIRD LLM backend (adjudicator role), isolated from both reviewers.
  4. Processes disagreements one at a time (no batching — full attention per item).
  5. Agreed items are never sent here — only genuine disagreements.

Output per disagreement item: final_decision (INCLUDE_Tier2 | EXCLUDE) + reasoning.
"""

from __future__ import annotations
import logging
import random
from typing import Any, Dict, List, Optional

from agent_layer.base_agent import BaseAgent, TaskInstruction, StageOutput
from llm_backend import LLMInterface
from skill_layer import SkillLoader

logger = logging.getLogger(__name__)

ADJUDICATION_SKILL_ID = "screening.adjudication_protocol"


class AdjudicatorAgent(BaseAgent):
    """
    Third-party adjudicator for disagreement resolution.

    Args:
        llm:           Adjudicator LLM backend (must differ from both reviewers)
        skill_loader:  Shared skill loader
    """

    def __init__(
        self,
        llm: LLMInterface,
        skill_loader: Optional[SkillLoader] = None,
    ):
        super().__init__(llm=llm, skill_loader=skill_loader, agent_id="adjudicator")

    @property
    def agent_type(self) -> str:
        return "adjudicator"

    # ── Main execute ──────────────────────────────────────────────────────────

    def execute(self, task: TaskInstruction) -> StageOutput:
        """
        Adjudicate a list of disagreement items.

        task.input_data must contain:
            disagreement_queue   : List[Dict]  from dual_review_aggregation
            eligibility_criteria : Dict        confirmed criteria
        """
        queue: List[Dict] = task.input_data.get("disagreement_queue", [])
        criteria: Dict = task.input_data.get("eligibility_criteria", {})

        if not queue:
            return StageOutput(
                stage=task.stage, step_id=task.step_id,
                agent_id=self.agent_id, success=True,
                data={"adjudicated": []},
            )

        self.load_skills([ADJUDICATION_SKILL_ID])
        adjudicated: List[Dict] = []
        warnings: List[str] = []

        for item in queue:
            logger.info(
                "Adjudicator processing PMID %s (%s)",
                item.get("pmid"), item.get("disagreement_type"),
            )
            try:
                result = self._adjudicate_one(item, criteria)
                adjudicated.append(result)
            except Exception as exc:
                logger.error("Adjudication failed for PMID %s: %s", item.get("pmid"), exc)
                # Conservative fallback: INCLUDE_Tier2 on failure
                adjudicated.append({
                    "pmid": item.get("pmid"),
                    "final_decision": "INCLUDE_Tier2",
                    "chosen_reasoning": f"Adjudication error — conservative inclusion: {exc}",
                    "dissent_note": "Not available due to error",
                    "evidence_used": "",
                    "disagreement_source": "error",
                })
                warnings.append(f"Adjudication error for PMID {item.get('pmid')}: {exc}")

        return StageOutput(
            stage=task.stage,
            step_id=task.step_id,
            agent_id=self.agent_id,
            success=True,
            data={"adjudicated": adjudicated},
            warnings=warnings,
        )

    # ── Single-item adjudication ──────────────────────────────────────────────

    def _adjudicate_one(self, item: Dict[str, Any], criteria: Dict) -> Dict[str, Any]:
        """Adjudicate a single disagreement item."""
        pmid = item.get("pmid", "")
        title = item.get("title", "")
        abstract = item.get("abstract", "")

        # Anonymize reviewer labels (randomly assign 1/2 to prevent anchoring bias)
        r1_decision = item.get("reviewer_a_decision", "UNCERTAIN")
        r1_reasoning = item.get("reviewer_a_reasoning", "")
        r1_criteria  = item.get("reviewer_a_criteria", [])
        r2_decision  = item.get("reviewer_b_decision", "UNCERTAIN")
        r2_reasoning = item.get("reviewer_b_reasoning", "")
        r2_criteria  = item.get("reviewer_b_criteria", [])

        # Randomly swap labels to prevent model bias toward "Reviewer 1"
        if random.random() < 0.5:
            r1_decision, r2_decision = r2_decision, r1_decision
            r1_reasoning, r2_reasoning = r2_reasoning, r1_reasoning
            r1_criteria, r2_criteria = r2_criteria, r1_criteria

        task_instruction = (
            "You are a third-party adjudicator for a systematic review screening disagreement.\n"
            "Two independent reviewers reached different conclusions about this paper.\n"
            "Your task:\n"
            "  1. Analyze the source of disagreement\n"
            "  2. Re-apply the eligibility criteria to the paper\n"
            "  3. Reach a final decision: INCLUDE_Tier2 or EXCLUDE\n"
            "When uncertain after analysis, prefer INCLUDE_Tier2 (err on side of inclusion)."
        )

        task_data = {
            "disagreement_item": {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "disagreement_type": item.get("disagreement_type", ""),
                "reviewer_1": {
                    "decision": r1_decision,
                    "reasoning": r1_reasoning,
                    "criteria_assessment": r1_criteria,
                },
                "reviewer_2": {
                    "decision": r2_decision,
                    "reasoning": r2_reasoning,
                    "criteria_assessment": r2_criteria,
                },
            },
            "eligibility_criteria": {
                "binary_questions": self._extract_binary_questions(criteria),
                "guidance": criteria.get("screening_guidance", {}),
            },
        }

        context = self.build_prompt_context(
            active_skill_id=ADJUDICATION_SKILL_ID,
            task_instruction=task_instruction,
            task_data_dict=task_data,
        )

        _response, output = self.call_llm(
            context=context,
            active_skill_id=ADJUDICATION_SKILL_ID,
            output_format="json",
        )

        if not isinstance(output, dict):
            raise ValueError(f"Unexpected adjudicator output type: {type(output)}")

        # Ensure pmid is present
        output["pmid"] = pmid
        return output

    def _extract_binary_questions(self, criteria: Dict) -> List[Dict]:
        """Extract all binary questions as a flat list for the adjudicator."""
        questions = []
        ic = criteria.get("inclusion_criteria", {})
        for dim in ["population", "intervention", "comparison", "outcome", "study_design"]:
            q = ic.get(dim, {}).get("binary_question", "")
            if q:
                questions.append({"criterion_id": dim, "question": q})
        for add in ic.get("additional", []):
            q = add.get("binary_question", "")
            if q:
                questions.append({"criterion_id": "additional", "question": q})
        return questions
