"""
Screening Pipeline — Stage 2: Literature Screening

Implements the 7-phase dual-review workflow:

  Phase 1    eligibility_criteria_design   LLM (ExecutorAgent)  → unified_criteria
  Phase 1.5  pre-filter                    Code                  → filtered_pool
  Phase 2a   title screening               ReviewerAgent × 2     → title decisions
  Phase 2b   abstract screening            ReviewerAgent × 2     → abstract decisions
  Phase 3    agreement + kappa             Code                  → agreed/disagreements + κ
  Phase 4    adjudication                  AdjudicatorAgent      → resolved disagreements
  Phase 5    report                        Code                  → included_studies + report

Human checkpoint: after Phase 1 (criteria generation), the pipeline pauses and calls
  `human_checkpoint_callback(criteria) → bool`.  If the callback returns False or is
  not provided, criteria are accepted as-is (non-blocking mode).

Output (ScreeningResult):
    included_studies   — [{pmid, title, abstract, ...}]
    excluded_studies   — [{pmid, title, exclude_reason, ...}]
    uncertain_studies  — [{pmid, title, ...}]  (require full-text review)
    screening_report   — PRISMA flow counts + kappa + quality signals
    criteria           — confirmed eligibility criteria used
    execution_log      — full audit trail
"""

from __future__ import annotations
import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent_layer import ExecutorAgent, StepSpec, TaskInstruction, ReviewerAgent, AdjudicatorAgent
from llm_backend import get_registry
from skill_layer import get_loader

logger = logging.getLogger(__name__)

# Publication types to exclude during pre-filtering
EXCLUDED_PUB_TYPES = {
    "Review", "Systematic Review", "Meta-Analysis", "Editorial",
    "Letter", "Comment", "Guideline", "Practice Guideline",
    "Case Reports", "News", "Published Erratum", "Retraction of Publication",
}


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class ScreeningResult:
    included_studies: List[Dict[str, Any]]
    excluded_studies: List[Dict[str, Any]]
    uncertain_studies: List[Dict[str, Any]]   # Tier-2 includes needing full text
    screening_report: Dict[str, Any]
    criteria: Dict[str, Any]
    execution_log: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ── Kappa calculation ─────────────────────────────────────────────────────────

def compute_kappa(
    decisions_a: List[str], decisions_b: List[str]
) -> Tuple[float, float, str]:
    """
    Compute Cohen's kappa between two decision lists.

    Returns: (kappa, iaar, category_label)
    """
    assert len(decisions_a) == len(decisions_b), "Decision lists must be same length"
    n = len(decisions_a)
    if n == 0:
        return 0.0, 0.0, "poor"

    categories = ["INCLUDE", "EXCLUDE", "UNCERTAIN"]

    # Observed agreement
    agreements = sum(1 for a, b in zip(decisions_a, decisions_b) if a == b)
    po = agreements / n

    # Expected agreement
    pe = 0.0
    for cat in categories:
        p_a = sum(1 for d in decisions_a if d == cat) / n
        p_b = sum(1 for d in decisions_b if d == cat) / n
        pe += p_a * p_b

    kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0.0
    iaar = po

    if kappa >= 0.81:
        category = "almost_perfect"
    elif kappa >= 0.61:
        category = "substantial"
    elif kappa >= 0.41:
        category = "moderate"
    elif kappa >= 0.21:
        category = "fair"
    else:
        category = "slight_or_poor"

    return round(kappa, 4), round(iaar, 4), category


def kappa_system_action(kappa: float) -> str:
    if kappa >= 0.41:
        return "continue" if kappa >= 0.61 else "warn_and_continue"
    elif kappa >= 0.21:
        return "flag_human"
    return "block_until_human"


# ── Pipeline ──────────────────────────────────────────────────────────────────

class ScreeningPipeline:
    """
    Orchestrates the full dual-review screening stage.

    Args:
        candidate_pool:             List of paper dicts from SearchPipeline
        review_question:            PICO dict
        pico_terms:                 Term table from search stage
        eligibility_criteria:       Pre-confirmed criteria (skip Phase 1 if provided)
        human_checkpoint_callback:  fn(criteria: dict) -> bool; if None, non-blocking
        save_dir:                   Optional directory for intermediate results
        llm_backend_name:           Primary LLM backend (for ExecutorAgent + AdjudicatorAgent)
    """

    def __init__(
        self,
        candidate_pool: List[Dict[str, Any]],
        review_question: Dict[str, str],
        pico_terms: Optional[Dict] = None,
        eligibility_criteria: Optional[Dict] = None,
        human_checkpoint_callback: Optional[Callable] = None,
        save_dir: Optional[Path] = None,
        llm_backend_name: str = "default",
    ):
        self.candidate_pool = candidate_pool
        self.review_question = review_question
        self.pico_terms = pico_terms or {}
        self._preloaded_criteria = eligibility_criteria
        self.human_checkpoint_callback = human_checkpoint_callback
        self.save_dir = save_dir

        registry = get_registry()
        loader = get_loader()

        # ExecutorAgent (criteria generation)
        self._executor = ExecutorAgent(llm=registry.get(llm_backend_name), skill_loader=loader)

        # Heterogeneous dual-review pair
        llm_a, llm_b = registry.get_dual_review_pair()
        self._reviewer_a = ReviewerAgent(llm=llm_a, reviewer_id="reviewer_alpha",  skill_loader=loader)
        self._reviewer_b = ReviewerAgent(llm=llm_b, reviewer_id="reviewer_beta",   skill_loader=loader)

        # Adjudicator (third independent LLM)
        self._adjudicator = AdjudicatorAgent(llm=registry.get_adjudicator(), skill_loader=loader)

        self._all_warnings: List[str] = []

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self) -> ScreeningResult:
        """Execute the full 7-phase screening pipeline."""
        logger.info("ScreeningPipeline: starting with %d candidates", len(self.candidate_pool))

        # --- Phase 1: Criteria generation ------------------------------------
        criteria = self._phase1_generate_criteria()

        # --- Human checkpoint ------------------------------------------------
        criteria = self._human_checkpoint(criteria)

        # --- Phase 1.5: Pre-filter -------------------------------------------
        filtered_pool = self._phase1_5_prefilter(self.candidate_pool)

        # --- Phase 2a: Title screening (dual, sequential) --------------------
        title_decisions_a, title_decisions_b = self._phase2a_title_screening(filtered_pool)

        # --- Title agreement (liberal pass) ----------------------------------
        title_passed_pmids, title_excluded_pmids = self._title_liberal_pass(
            title_decisions_a, title_decisions_b
        )
        pmid_index = {p["pmid"]: p for p in filtered_pool if "pmid" in p}
        title_passed_pool = [pmid_index[pmid] for pmid in title_passed_pmids if pmid in pmid_index]

        logger.info(
            "Title screening: %d passed, %d excluded", len(title_passed_pool), len(title_excluded_pmids)
        )

        # --- Phase 2b: Abstract screening (dual, sequential) -----------------
        abs_decisions_a, abs_decisions_b = self._phase2b_abstract_screening(
            title_passed_pool, criteria
        )

        # --- Phase 3: Abstract agreement + kappa ----------------------------
        agreed_include, agreed_exclude, disagreement_queue, kappa_result = (
            self._phase3_agreement_and_kappa(abs_decisions_a, abs_decisions_b, title_passed_pool)
        )

        # --- Phase 4: Adjudication ------------------------------------------
        adjudicated = self._phase4_adjudication(disagreement_queue, criteria)

        # --- Phase 5: Merge results + report --------------------------------
        result = self._phase5_report(
            candidate_pool=self.candidate_pool,
            filtered_pool=filtered_pool,
            title_excluded_pmids=title_excluded_pmids,
            title_passed_pool=title_passed_pool,
            agreed_include=agreed_include,
            agreed_exclude=agreed_exclude,
            adjudicated=adjudicated,
            kappa_result=kappa_result,
            criteria=criteria,
        )

        if self.save_dir:
            self._save_result(result)

        return result

    # ── Phase 1: Criteria generation ─────────────────────────────────────────

    def _phase1_generate_criteria(self) -> Dict[str, Any]:
        if self._preloaded_criteria:
            logger.info("ScreeningPipeline: using pre-loaded eligibility criteria")
            return self._preloaded_criteria

        logger.info("ScreeningPipeline Phase 1: generating eligibility criteria")
        step_sequence = [
            StepSpec(
                "screening_step_1_criteria",
                "screening.eligibility_criteria_design",
                True, [],
                "Generate PICOS eligibility criteria as binary questions",
            )
        ]
        task = TaskInstruction(
            stage="screening",
            step_id="screening_step_1_criteria",
            skill_ids=["screening.eligibility_criteria_design"],
            input_data={
                "step_sequence": step_sequence,
                "shared_context": {
                    "review_question": self.review_question,
                    "pico_terms": self.pico_terms,
                },
            },
        )
        output = self._executor.execute(task)
        criteria = output.data.get("screening_step_1_criteria", {})
        if not criteria:
            raise RuntimeError("Criteria generation returned empty output")
        return criteria

    # ── Human checkpoint ──────────────────────────────────────────────────────

    def _human_checkpoint(self, criteria: Dict[str, Any]) -> Dict[str, Any]:
        if self.human_checkpoint_callback is None:
            logger.info("No human checkpoint callback — proceeding with generated criteria")
            return criteria
        try:
            result = self.human_checkpoint_callback(criteria)
            if isinstance(result, dict):
                # Callback returned modified criteria
                return result
            # Callback returned True/False (approve/reject)
            if not result:
                logger.warning("Human checkpoint rejected criteria — using original")
        except Exception as exc:
            logger.warning("Human checkpoint callback error: %s", exc)
        return criteria

    # ── Phase 1.5: Pre-filter ─────────────────────────────────────────────────

    def _phase1_5_prefilter(self, pool: List[Dict]) -> List[Dict]:
        """Remove clearly ineligible records by publication type."""
        filtered = []
        removed = 0
        for paper in pool:
            pub_types = set(paper.get("pub_types", []))
            if pub_types & EXCLUDED_PUB_TYPES:
                removed += 1
            else:
                filtered.append(paper)
        logger.info(
            "Pre-filter: removed %d records by pub type; %d remain", removed, len(filtered)
        )
        return filtered

    # ── Phase 2a: Title screening ─────────────────────────────────────────────

    def _phase2a_title_screening(
        self, pool: List[Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        """Run title screening with both reviewers independently."""
        logger.info("Phase 2a: title screening %d papers", len(pool))
        criteria = self._preloaded_criteria or {}  # criteria may not be generated yet

        def _review(reviewer: ReviewerAgent, name: str) -> List[Dict]:
            task = TaskInstruction(
                stage="screening",
                step_id=f"screening_step_2a_{name}",
                skill_ids=["screening.screening_decision_logic"],
                input_data={
                    "papers": pool,
                    "eligibility_criteria": criteria,
                    "screening_mode": "title",
                },
            )
            out = reviewer.execute(task)
            return out.data.get("decisions", [])

        decisions_a = _review(self._reviewer_a, "alpha")
        decisions_b = _review(self._reviewer_b, "beta")
        return decisions_a, decisions_b

    # ── Title liberal pass logic ──────────────────────────────────────────────

    def _title_liberal_pass(
        self,
        decisions_a: List[Dict],
        decisions_b: List[Dict],
    ) -> Tuple[List[str], List[str]]:
        """
        Liberal pass: BOTH reviewers must say EXCLUDE to exclude at title stage.
        Returns (passed_pmids, excluded_pmids).
        """
        a_map = {d["pmid"]: d["decision"] for d in decisions_a}
        b_map = {d["pmid"]: d["decision"] for d in decisions_b}
        all_pmids = set(a_map.keys()) | set(b_map.keys())

        passed: List[str] = []
        excluded: List[str] = []
        for pmid in all_pmids:
            da = a_map.get(pmid, "UNCERTAIN")
            db = b_map.get(pmid, "UNCERTAIN")
            if da == "EXCLUDE" and db == "EXCLUDE":
                excluded.append(pmid)
            else:
                passed.append(pmid)
        return passed, excluded

    # ── Phase 2b: Abstract screening ──────────────────────────────────────────

    def _phase2b_abstract_screening(
        self, pool: List[Dict], criteria: Dict
    ) -> Tuple[List[Dict], List[Dict]]:
        """Run abstract screening with both reviewers independently."""
        logger.info("Phase 2b: abstract screening %d papers", len(pool))

        def _review(reviewer: ReviewerAgent, name: str) -> List[Dict]:
            task = TaskInstruction(
                stage="screening",
                step_id=f"screening_step_2b_{name}",
                skill_ids=["screening.screening_decision_logic"],
                input_data={
                    "papers": pool,
                    "eligibility_criteria": criteria,
                    "screening_mode": "abstract",
                },
            )
            out = reviewer.execute(task)
            return out.data.get("decisions", [])

        decisions_a = _review(self._reviewer_a, "alpha")
        decisions_b = _review(self._reviewer_b, "beta")
        return decisions_a, decisions_b

    # ── Phase 3: Agreement + kappa ────────────────────────────────────────────

    def _phase3_agreement_and_kappa(
        self,
        decisions_a: List[Dict],
        decisions_b: List[Dict],
        pool: List[Dict],
    ) -> Tuple[List[str], List[str], List[Dict], Dict]:
        """
        Compute agreement, kappa, and build disagreement queue.
        Returns: (agreed_include_pmids, agreed_exclude_pmids, disagreement_queue, kappa_result)
        """
        a_map: Dict[str, Dict] = {d["pmid"]: d for d in decisions_a}
        b_map: Dict[str, Dict] = {d["pmid"]: d for d in decisions_b}
        pool_index: Dict[str, Dict] = {p["pmid"]: p for p in pool if "pmid" in p}

        all_pmids = list(set(a_map.keys()) | set(b_map.keys()))

        # Align decisions for kappa (use UNCERTAIN for missing)
        a_list = [a_map.get(pmid, {}).get("decision", "UNCERTAIN") for pmid in all_pmids]
        b_list = [b_map.get(pmid, {}).get("decision", "UNCERTAIN") for pmid in all_pmids]

        kappa, iaar, category = compute_kappa(a_list, b_list)
        action = kappa_system_action(kappa)
        kappa_result = {
            "kappa": kappa,
            "kappa_category": category,
            "iaar": iaar,
            "system_action": action,
            "total_papers": len(all_pmids),
        }

        logger.info(
            "Abstract screening kappa=%.3f (%s), IAAR=%.2f%%, action=%s",
            kappa, category, iaar * 100, action,
        )

        if action == "block_until_human":
            self._all_warnings.append(
                f"Kappa={kappa:.3f} is below 0.21 — pipeline should be blocked for human calibration."
            )
        elif action == "flag_human":
            self._all_warnings.append(
                f"Kappa={kappa:.3f} is below 0.41 — flagged for human review."
            )

        # Classify papers
        agreed_include: List[str] = []
        agreed_exclude: List[str] = []
        disagreement_queue: List[Dict] = []

        for pmid in all_pmids:
            da_dict = a_map.get(pmid, {"decision": "UNCERTAIN", "reasoning": "", "criteria_assessment": []})
            db_dict = b_map.get(pmid, {"decision": "UNCERTAIN", "reasoning": "", "criteria_assessment": []})
            da = da_dict.get("decision", "UNCERTAIN")
            db = db_dict.get("decision", "UNCERTAIN")

            if da == "INCLUDE" and db == "INCLUDE":
                agreed_include.append(pmid)
            elif da == "EXCLUDE" and db == "EXCLUDE":
                agreed_exclude.append(pmid)
            else:
                # Determine disagreement type
                if {da, db} == {"INCLUDE", "EXCLUDE"}:
                    dtype = "INCLUDE_vs_EXCLUDE"
                elif "UNCERTAIN" in {da, db} and "INCLUDE" in {da, db}:
                    dtype = "INCLUDE_vs_UNCERTAIN"
                elif "UNCERTAIN" in {da, db} and "EXCLUDE" in {da, db}:
                    dtype = "EXCLUDE_vs_UNCERTAIN"
                else:
                    dtype = "BOTH_UNCERTAIN"

                paper = pool_index.get(pmid, {"pmid": pmid, "title": "", "abstract": ""})
                disagreement_queue.append({
                    "pmid": pmid,
                    "title":    paper.get("title", ""),
                    "abstract": paper.get("abstract", ""),
                    "reviewer_a_decision":   da,
                    "reviewer_b_decision":   db,
                    "reviewer_a_reasoning":  da_dict.get("reasoning", ""),
                    "reviewer_b_reasoning":  db_dict.get("reasoning", ""),
                    "reviewer_a_criteria":   da_dict.get("criteria_assessment", []),
                    "reviewer_b_criteria":   db_dict.get("criteria_assessment", []),
                    "disagreement_type":     dtype,
                })

        logger.info(
            "Agreement: agreed_include=%d, agreed_exclude=%d, disagreements=%d",
            len(agreed_include), len(agreed_exclude), len(disagreement_queue),
        )
        return agreed_include, agreed_exclude, disagreement_queue, kappa_result

    # ── Phase 4: Adjudication ─────────────────────────────────────────────────

    def _phase4_adjudication(
        self, disagreement_queue: List[Dict], criteria: Dict
    ) -> List[Dict]:
        if not disagreement_queue:
            return []

        logger.info("Phase 4: adjudicating %d disagreements", len(disagreement_queue))
        task = TaskInstruction(
            stage="screening",
            step_id="screening_step_4_adjudication",
            skill_ids=["screening.adjudication_protocol"],
            input_data={
                "disagreement_queue": disagreement_queue,
                "eligibility_criteria": criteria,
            },
        )
        out = self._adjudicator.execute(task)
        adjudicated = out.data.get("adjudicated", [])
        self._all_warnings.extend(out.warnings)
        return adjudicated

    # ── Phase 5: Report ───────────────────────────────────────────────────────

    def _phase5_report(
        self,
        candidate_pool: List[Dict],
        filtered_pool: List[Dict],
        title_excluded_pmids: List[str],
        title_passed_pool: List[Dict],
        agreed_include: List[str],
        agreed_exclude: List[str],
        adjudicated: List[Dict],
        kappa_result: Dict,
        criteria: Dict,
    ) -> ScreeningResult:
        """Merge all results and build the final ScreeningResult."""
        pool_index: Dict[str, Dict] = {p["pmid"]: p for p in candidate_pool if "pmid" in p}

        # Classify adjudicated items
        adj_include = {r["pmid"] for r in adjudicated if r.get("final_decision") == "INCLUDE_Tier2"}
        adj_exclude = {r["pmid"] for r in adjudicated if r.get("final_decision") == "EXCLUDE"}
        adj_reasoning = {r["pmid"]: r for r in adjudicated}

        # Build result sets
        included_pmids = set(agreed_include) | adj_include
        excluded_pmids = (
            set(title_excluded_pmids)
            | set(agreed_exclude)
            | adj_exclude
        )

        # Papers in included_pmids are Tier-2 (need full-text confirmation)
        # Papers that remain UNCERTAIN after adjudication (neither include nor exclude)
        all_screened = set(p["pmid"] for p in filtered_pool if "pmid" in p)
        uncertain_pmids = all_screened - included_pmids - excluded_pmids

        def _build_record(pmid: str, extra: Optional[Dict] = None) -> Dict:
            r = dict(pool_index.get(pmid, {"pmid": pmid}))
            if extra:
                r.update(extra)
            return r

        included_studies = [_build_record(pmid, {"screening_tier": "Tier2"}) for pmid in included_pmids]
        excluded_studies = []
        for pmid in set(title_excluded_pmids):
            excluded_studies.append(_build_record(pmid, {"exclude_stage": "title", "exclude_reason": "Both reviewers excluded at title stage"}))
        for pmid in set(agreed_exclude):
            excluded_studies.append(_build_record(pmid, {"exclude_stage": "abstract", "exclude_reason": "Both reviewers excluded at abstract stage"}))
        for pmid in adj_exclude:
            r = adj_reasoning.get(pmid, {})
            excluded_studies.append(_build_record(pmid, {
                "exclude_stage": "adjudication",
                "exclude_reason": r.get("chosen_reasoning", "Adjudicator excluded"),
            }))
        uncertain_studies = [_build_record(pmid) for pmid in uncertain_pmids]

        # PRISMA flow counts
        pre_dedup = len(candidate_pool)
        after_prefilter = len(filtered_pool)
        after_title = len(title_passed_pool)
        after_abstract = len(agreed_include) + len(agreed_exclude) + len(adjudicated)

        screening_report = {
            "date": date.today().isoformat(),
            "prisma_flow": {
                "records_identified":         pre_dedup,
                "records_after_prefilter":    after_prefilter,
                "records_excluded_prefilter": pre_dedup - after_prefilter,
                "records_title_screened":     after_prefilter,
                "records_excluded_title":     len(title_excluded_pmids),
                "records_abstract_screened":  len(title_passed_pool),
                "records_excluded_abstract":  len(agreed_exclude),
                "records_adjudicated":        len(adjudicated),
                "records_included_tier2":     len(included_studies),
            },
            "agreement_analysis": {
                "kappa":              kappa_result.get("kappa"),
                "kappa_category":     kappa_result.get("kappa_category"),
                "iaar":               kappa_result.get("iaar"),
                "system_action":      kappa_result.get("system_action"),
                "total_papers":       kappa_result.get("total_papers"),
                "adjudicated_count":  len(adjudicated),
                "adjudicated_include": len(adj_include),
                "adjudicated_exclude": len(adj_exclude),
            },
            "exclusion_reasons": self._summarize_exclusion_reasons(excluded_studies),
            "warnings": self._all_warnings,
        }

        return ScreeningResult(
            included_studies=included_studies,
            excluded_studies=excluded_studies,
            uncertain_studies=uncertain_studies,
            screening_report=screening_report,
            criteria=criteria,
            execution_log=(
                self._executor.dump_log()
                + self._reviewer_a.dump_log()
                + self._reviewer_b.dump_log()
                + self._adjudicator.dump_log()
            ),
            warnings=self._all_warnings,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _summarize_exclusion_reasons(excluded: List[Dict]) -> Dict[str, int]:
        counts: Dict[str, int] = defaultdict(int)
        for rec in excluded:
            reason = rec.get("exclude_reason", "unknown")
            # Trim to first sentence for grouping
            short = reason.split(".")[0][:80] if reason else "unknown"
            counts[short] += 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    def _save_result(self, result: ScreeningResult) -> None:
        if not self.save_dir:
            return
        self.save_dir.mkdir(parents=True, exist_ok=True)
        with open(self.save_dir / "included_studies.json", "w", encoding="utf-8") as f:
            json.dump(result.included_studies, f, ensure_ascii=False, indent=2)
        with open(self.save_dir / "excluded_studies.json", "w", encoding="utf-8") as f:
            json.dump(result.excluded_studies, f, ensure_ascii=False, indent=2)
        with open(self.save_dir / "screening_report.json", "w", encoding="utf-8") as f:
            json.dump(result.screening_report, f, ensure_ascii=False, indent=2)
        with open(self.save_dir / "criteria.json", "w", encoding="utf-8") as f:
            json.dump(result.criteria, f, ensure_ascii=False, indent=2)
        logger.info("ScreeningPipeline: results saved to %s", self.save_dir)


# ── Convenience function ──────────────────────────────────────────────────────

def run_screening(
    candidate_pool: List[Dict],
    review_question: Dict[str, str],
    pico_terms: Optional[Dict] = None,
    eligibility_criteria: Optional[Dict] = None,
    human_checkpoint_callback: Optional[Callable] = None,
    save_dir: Optional[Path] = None,
    llm_backend: str = "default",
) -> ScreeningResult:
    """
    Convenience wrapper to run the screening pipeline.

    Args:
        candidate_pool:           Papers from search stage (list of dicts with pmid/title/abstract)
        review_question:          {"P": "...", "I": "...", "O": "..."}
        pico_terms:               PICO term table from search stage (optional)
        eligibility_criteria:     Pre-confirmed criteria (skip generation if provided)
        human_checkpoint_callback: fn(criteria) -> bool|dict; None = non-blocking
        save_dir:                 Directory to save results
        llm_backend:              Primary LLM backend name

    Returns:
        ScreeningResult with included/excluded/uncertain studies + report
    """
    pipeline = ScreeningPipeline(
        candidate_pool=candidate_pool,
        review_question=review_question,
        pico_terms=pico_terms,
        eligibility_criteria=eligibility_criteria,
        human_checkpoint_callback=human_checkpoint_callback,
        save_dir=save_dir,
        llm_backend_name=llm_backend,
    )
    return pipeline.run()
