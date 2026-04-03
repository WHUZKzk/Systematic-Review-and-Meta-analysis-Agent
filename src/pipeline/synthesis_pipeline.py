"""
synthesis_pipeline.py — Evidence synthesis and meta-analysis pipeline.

Phases:
  Phase 1: Synthesis plan + feasibility assessment (LLM)
  Phase 2: Data preparation (LLM + Python sandbox)
  Phase 3: Meta-analysis execution (R engine)
  Phase 4: Heterogeneity analysis + subgroup (R engine + LLM interpretation)
  Phase 5: Publication bias (R engine, Egger test, trim-and-fill)
  Phase 6: GRADE certainty of evidence (LLM)
  Phase 7: Evidence summary report (LLM, 7-section template)

Usage:
    from pipeline.synthesis_pipeline import run_synthesis, SynthesisResult

    result = run_synthesis(
        extraction_result=extraction_result,   # from extraction_pipeline
        review_question="...",
        save_dir=Path("outputs/synthesis"),
    )
"""

from __future__ import annotations

import json
import logging
import pathlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import config
from agent_layer import ExecutorAgent, TaskInstruction
from llm_backend import get_registry
from skill_layer import get_loader
from tool_layer.r_engine import RStatisticalEngineTool, MetaAnalysisRunner

logger = logging.getLogger(__name__)

# ── Skill IDs ─────────────────────────────────────────────────────────────────

_SKILL_FEASIBILITY   = "synthesis.feasibility_assessment"
_SKILL_EFFECT_MEAS   = "synthesis.effect_measure_selection"
_SKILL_TAU2          = "synthesis.heterogeneity_estimator_selection"
_SKILL_MODEL         = "synthesis.model_selection"
_SKILL_HETERO_INTERP = "synthesis.heterogeneity_interpretation"
_SKILL_EVIDENCE_SUM  = "synthesis.evidence_summary_template"
_SKILL_PLAN          = "synthesis.synthesis_plan"


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class OutcomeMetaResult:
    """Meta-analysis result for a single outcome group."""
    outcome_id: str
    outcome_name: str
    effect_measure: str
    model_type: str
    studies_included: int
    pooled_estimate: Optional[float]
    ci_lower: Optional[float]
    ci_upper: Optional[float]
    prediction_lower: Optional[float]
    prediction_upper: Optional[float]
    i2: Optional[float]
    tau2: Optional[float]
    q_p_value: Optional[float]
    heterogeneity_interpretation: Dict[str, Any] = field(default_factory=dict)
    subgroup_results: List[Dict] = field(default_factory=list)
    publication_bias: Dict[str, Any] = field(default_factory=dict)
    grade_certainty: str = "not assessed"
    grade_reasoning: str = ""
    r_available: bool = False
    forest_plot_path: Optional[str] = None


@dataclass
class SynthesisResult:
    """Full synthesis stage output."""
    synthesis_plan: Dict[str, Any]
    outcome_results: List[OutcomeMetaResult]
    evidence_summary: Dict[str, Any]
    synthesis_report: Dict[str, Any]
    studies_in_meta: int
    warnings: List[str] = field(default_factory=list)


# ── Pipeline ──────────────────────────────────────────────────────────────────

class SynthesisPipeline:
    """
    Multi-phase evidence synthesis pipeline.

    Args:
        llm_backend:   LLM backend (default model)
        save_dir:      Output directory
        human_checkpoint_callback:
                       Optional (stage, deliverable, payload) -> bool
    """

    def __init__(
        self,
        llm_backend=None,
        save_dir: Optional[pathlib.Path] = None,
        human_checkpoint_callback: Optional[Callable] = None,
    ):
        self._llm = llm_backend or get_registry().get("default")
        self._save_dir = save_dir or pathlib.Path("outputs/synthesis")
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_cb = human_checkpoint_callback

        skill_loader = get_loader()
        self._agent = ExecutorAgent(llm=self._llm, skill_loader=skill_loader)
        self._r_engine = RStatisticalEngineTool()
        self._runner = MetaAnalysisRunner(r_engine=self._r_engine)
        self._agent.register_tools([self._r_engine])

    # ── Main entry ─────────────────────────────────────────────────────────

    def run(
        self,
        extraction_result: Any,
        review_question: str,
    ) -> SynthesisResult:

        # Extract data from ExtractionResult or raw dict
        extraction_table = self._get_extraction_table(extraction_result)
        rob_summary = self._get_rob_summary(extraction_result)

        logger.info("SynthesisPipeline: %d rows in extraction table", len(extraction_table))
        warnings: List[str] = []

        # Phase 1: Synthesis plan + feasibility
        plan = self._phase1_plan(review_question, extraction_table, rob_summary)
        logger.info("Synthesis plan: %d outcome groups", len(plan.get("outcome_hierarchy", {}).get("primary", [])))

        # Human checkpoint: plan approval
        if self._checkpoint_cb:
            approved = self._checkpoint_cb(
                "synthesis_plan",
                "Synthesis plan: comparison groups, effect measures, analysis strategy",
                {"plan": plan},
            )
            if not approved:
                raise RuntimeError("Synthesis plan rejected at human checkpoint")

        feasibility = self._phase1_feasibility(extraction_table, plan)

        # Phase 2: Data preparation
        prepared_data = self._phase2_prepare(extraction_table, plan, feasibility)

        # Phase 3-5: Meta-analysis per outcome group
        outcome_results: List[OutcomeMetaResult] = []
        for og in feasibility.get("outcome_groups", []):
            outcome_id = og.get("outcome_id", "unknown")
            if og.get("feasibility") == "narrative_only":
                logger.info("Outcome %s: narrative only", outcome_id)
                outcome_results.append(OutcomeMetaResult(
                    outcome_id=outcome_id,
                    outcome_name=og.get("outcome_name", outcome_id),
                    effect_measure="N/A",
                    model_type="narrative",
                    studies_included=og.get("study_count", 0),
                    pooled_estimate=None, ci_lower=None, ci_upper=None,
                    prediction_lower=None, prediction_upper=None,
                    i2=None, tau2=None, q_p_value=None,
                ))
                continue

            try:
                outcome_result = self._run_outcome_meta(
                    outcome_id=outcome_id,
                    outcome_name=og.get("outcome_name", outcome_id),
                    study_data=prepared_data.get(outcome_id, []),
                    feasibility_entry=og,
                    warnings=warnings,
                )
                outcome_results.append(outcome_result)
            except Exception as exc:
                logger.error("Meta-analysis failed for outcome %s: %s", outcome_id, exc)
                warnings.append(f"Meta-analysis failed for {outcome_id}: {exc}")

        studies_in_meta = max(
            (r.studies_included for r in outcome_results
             if r.model_type != "narrative"),
            default=0,
        )

        # Phase 6: GRADE
        outcome_results = self._phase6_grade(outcome_results, rob_summary)

        # Phase 7: Evidence summary
        evidence_summary = self._phase7_evidence_summary(
            outcome_results, review_question, plan
        )

        report = self._build_report(outcome_results, plan, feasibility, warnings)

        self._save_outputs(plan, outcome_results, evidence_summary, report)

        return SynthesisResult(
            synthesis_plan=plan,
            outcome_results=outcome_results,
            evidence_summary=evidence_summary,
            synthesis_report=report,
            studies_in_meta=studies_in_meta,
            warnings=warnings,
        )

    # ── Phase 1: Plan + feasibility ─────────────────────────────────────────

    def _phase1_plan(
        self, review_question: str, extraction_table: List[Dict], rob_summary: Dict
    ) -> Dict:
        self._agent.load_skills([_SKILL_PLAN])
        context = self._agent.build_prompt_context(
            active_skill_id=_SKILL_PLAN,
            task_instruction="Develop a synthesis plan for this systematic review",
            task_data_dict={
                "review_question": review_question,
                "extraction_table_summary": self._summarize_extraction(extraction_table),
                "rob_summary": rob_summary,
            },
        )
        _raw, plan = self._agent.call_llm(
            context=context, active_skill_id=_SKILL_PLAN, output_format="json"
        )
        if not isinstance(plan, dict):
            plan = {
                "comparison_groups": [],
                "outcome_hierarchy": {"primary": [], "secondary": []},
                "analysis_strategy": {"subgroup_variables": [], "sensitivity_analyses": []},
            }
        return plan

    def _phase1_feasibility(self, extraction_table: List[Dict], plan: Dict) -> Dict:
        self._agent.load_skills([_SKILL_FEASIBILITY, _SKILL_EFFECT_MEAS, _SKILL_TAU2])
        context = self._agent.build_prompt_context(
            active_skill_id=_SKILL_FEASIBILITY,
            task_instruction="Assess meta-analysis feasibility for each outcome group",
            task_data_dict={
                "extraction_table_summary": self._summarize_extraction(extraction_table),
                "synthesis_plan": plan,
            },
        )
        _raw, feasibility = self._agent.call_llm(
            context=context, active_skill_id=_SKILL_FEASIBILITY, output_format="json"
        )
        if not isinstance(feasibility, dict):
            feasibility = {"outcome_groups": []}
        return feasibility

    # ── Phase 2: Data preparation ───────────────────────────────────────────

    def _phase2_prepare(
        self,
        extraction_table: List[Dict],
        plan: Dict,
        feasibility: Dict,
    ) -> Dict[str, List[Dict]]:
        """Organize extraction data by outcome group for meta-analysis."""
        prepared: Dict[str, List[Dict]] = {}

        for og in feasibility.get("outcome_groups", []):
            outcome_id = og.get("outcome_id", "")
            outcome_name = og.get("outcome_name", outcome_id)

            # Filter rows belonging to this outcome
            rows = [
                r for r in extraction_table
                if r.get("endpoint_id") == outcome_id
                or outcome_name.lower() in (r.get("endpoint_id") or "").lower()
            ]

            if not rows:
                # Try broader match
                rows = [
                    r for r in extraction_table
                    if r.get("standardized_form") in ("mean_sd", "events_n", "hr_ci95")
                ]

            # Build per-study entries (pair intervention vs. control)
            study_entries = self._pair_arms(rows, og)
            prepared[outcome_id] = study_entries

        return prepared

    def _pair_arms(self, rows: List[Dict], og: Dict) -> List[Dict]:
        """Pair intervention and control arms for each study and timepoint."""
        from collections import defaultdict
        # Group by (pmid, timepoint_id)
        grouped: Dict = defaultdict(lambda: {"intervention": None, "control": None})
        for row in rows:
            pmid = row.get("pmid", "")
            tp = row.get("timepoint_id", "T_final")
            arm = row.get("arm_id", "")
            key = (pmid, tp)

            if "control" in arm.lower() or "usual" in arm.lower() or "placebo" in arm.lower():
                grouped[key]["control"] = row
                grouped[key]["pmid"] = pmid
                grouped[key]["study"] = pmid
            else:
                grouped[key]["intervention"] = row
                grouped[key]["pmid"] = pmid
                grouped[key]["study"] = pmid

        entries = []
        for (pmid, tp), arms in grouped.items():
            int_row = arms.get("intervention")
            ctrl_row = arms.get("control")
            if not int_row or not ctrl_row:
                continue

            outcome_type = int_row.get("outcome_type", "continuous")
            entry = {"study": pmid, "pmid": pmid, "timepoint": tp}

            if outcome_type == "continuous":
                entry.update({
                    "mean_e": int_row.get("values", {}).get("mean"),
                    "sd_e":   int_row.get("values", {}).get("sd"),
                    "n_e":    int_row.get("values", {}).get("n"),
                    "mean_c": ctrl_row.get("values", {}).get("mean"),
                    "sd_c":   ctrl_row.get("values", {}).get("sd"),
                    "n_c":    ctrl_row.get("values", {}).get("n"),
                })
            elif outcome_type == "binary":
                entry.update({
                    "event_e": int_row.get("values", {}).get("events"),
                    "n_e":     int_row.get("values", {}).get("n"),
                    "event_c": ctrl_row.get("values", {}).get("events"),
                    "n_c":     ctrl_row.get("values", {}).get("n"),
                })
            elif outcome_type in ("time_to_event", "rate"):
                entry.update({
                    "TE":   int_row.get("values", {}).get("log_hr",
                             int_row.get("values", {}).get("log_effect")),
                    "seTE": int_row.get("values", {}).get("se_log_hr",
                             int_row.get("values", {}).get("se")),
                })

            entries.append(entry)

        return entries

    # ── Phase 3-5: Meta-analysis for one outcome ────────────────────────────

    def _run_outcome_meta(
        self,
        outcome_id: str,
        outcome_name: str,
        study_data: List[Dict],
        feasibility_entry: Dict,
        warnings: List[str],
    ) -> OutcomeMetaResult:

        effect_measure = feasibility_entry.get("effect_measure", "MD")
        tau2_method = feasibility_entry.get("tau2_estimator", "REML")
        outcome_type = feasibility_entry.get("outcome_type", "continuous")

        # Determine model type
        study_count = len(study_data)
        self._agent.load_skills([_SKILL_MODEL])

        # Use preliminary stats for model selection if k>0
        prelim_meta = {}
        if study_count >= 2:
            output_dir = self._save_dir / "forest_plots"
            output_dir.mkdir(exist_ok=True)
            if outcome_type == "continuous":
                prelim_meta = self._runner.run_continuous(
                    study_data, effect_measure, tau2_method, outcome_name, output_dir
                )
            elif outcome_type == "binary":
                prelim_meta = self._runner.run_binary(
                    study_data, effect_measure, tau2_method, outcome_name, output_dir
                )
            else:
                prelim_meta = self._runner.run_generic(
                    study_data, effect_measure, tau2_method, outcome_name, output_dir
                )

        results_dict = prelim_meta.get("results", {})
        r_avail = prelim_meta.get("r_available", False)

        # Phase 4: Heterogeneity interpretation
        hetero_interp = self._phase4_heterogeneity(outcome_id, results_dict)

        # Phase 5: Publication bias (requires k≥10)
        pub_bias = {}
        if study_count >= 10:
            pub_bias = self._phase5_pub_bias(outcome_id, outcome_name, study_data,
                                              outcome_type, effect_measure, tau2_method)

        # Find generated forest plot
        plot_path = None
        if r_avail:
            plots = prelim_meta.get("plots", [])
            if plots:
                plot_path = plots[0]

        return OutcomeMetaResult(
            outcome_id=outcome_id,
            outcome_name=outcome_name,
            effect_measure=effect_measure,
            model_type="random_effects",
            studies_included=study_count,
            pooled_estimate=results_dict.get("pooled_estimate"),
            ci_lower=results_dict.get("ci_lower"),
            ci_upper=results_dict.get("ci_upper"),
            prediction_lower=results_dict.get("prediction_lower"),
            prediction_upper=results_dict.get("prediction_upper"),
            i2=results_dict.get("i2"),
            tau2=results_dict.get("tau2"),
            q_p_value=results_dict.get("q_p_value"),
            heterogeneity_interpretation=hetero_interp,
            publication_bias=pub_bias,
            r_available=r_avail,
            forest_plot_path=plot_path,
        )

    # ── Phase 4: Heterogeneity ──────────────────────────────────────────────

    def _phase4_heterogeneity(self, outcome_id: str, results: Dict) -> Dict:
        self._agent.load_skills([_SKILL_HETERO_INTERP])
        context = self._agent.build_prompt_context(
            active_skill_id=_SKILL_HETERO_INTERP,
            task_instruction=f"Interpret heterogeneity for outcome: {outcome_id}",
            task_data_dict={
                "outcome_id": outcome_id,
                "meta_analysis_results": results,
            },
        )
        _raw, interp = self._agent.call_llm(
            context=context, active_skill_id=_SKILL_HETERO_INTERP, output_format="json"
        )
        if isinstance(interp, dict):
            hi_list = interp.get("heterogeneity_interpretations", [])
            return hi_list[0] if hi_list else interp
        return {}

    # ── Phase 5: Publication bias ───────────────────────────────────────────

    def _phase5_pub_bias(
        self,
        outcome_id: str,
        outcome_name: str,
        study_data: List[Dict],
        outcome_type: str,
        effect_measure: str,
        tau2_method: str,
    ) -> Dict:
        """Run trim-and-fill and Egger test via R engine."""
        trim_fill_code = f"""
            # Trim-and-Fill and Egger Test for publication bias
            # [re-runs meta-analysis then applies trim-and-fill]
        """
        # Build meta-analysis code + trim-and-fill
        if outcome_type == "continuous":
            meta_code = self._runner._build_continuous_code(
                study_data, effect_measure, tau2_method, outcome_name, None
            )
        elif outcome_type == "binary":
            meta_code = self._runner._build_binary_code(
                study_data, effect_measure, tau2_method, outcome_name, None
            )
        else:
            meta_code = self._runner._build_generic_code(
                study_data, effect_measure, tau2_method, outcome_name, None
            )

        bias_code = meta_code + textwrap.dedent("""
            import textwrap
            m_tf <- trimfill(m)
            egger_result <- tryCatch(metabias(m, method.bias="Egger"), error=function(e) NULL)
            bias_results <- list(
              k0_added = m_tf$k0,
              adjusted_estimate = exp(m_tf$TE.random),
              egger_p = if (!is.null(egger_result)) egger_result$p.value else NA
            )
            write_json(bias_results, .RESULTS_PATH, pretty=TRUE, auto_unbox=TRUE)
        """)
        import textwrap as _tw
        bias_code = meta_code + _tw.dedent("""
            m_tf <- tryCatch(trimfill(m), error=function(e) NULL)
            egger_result <- tryCatch(metabias(m, method.bias="Egger"), error=function(e) NULL)
            bias_results <- list(
              k0_added = if (!is.null(m_tf)) m_tf$k0 else NA,
              adjusted_estimate = if (!is.null(m_tf)) exp(m_tf$TE.random) else NA,
              egger_p = if (!is.null(egger_result)) egger_result$p.value else NA
            )
            results <- bias_results
        """)

        result = self._r_engine.execute({"r_code": bias_code})
        if result.success:
            return result.output.get("results", {})
        return {"error": result.error}

    # ── Phase 6: GRADE ─────────────────────────────────────────────────────

    def _phase6_grade(
        self, outcome_results: List[OutcomeMetaResult], rob_summary: Dict
    ) -> List[OutcomeMetaResult]:
        for outcome in outcome_results:
            if outcome.model_type == "narrative":
                outcome.grade_certainty = "not assessed (narrative)"
                continue
            certainty, reasoning = self._assess_grade_for_outcome(outcome, rob_summary)
            outcome.grade_certainty = certainty
            outcome.grade_reasoning = reasoning
        return outcome_results

    def _assess_grade_for_outcome(
        self, outcome: OutcomeMetaResult, rob_summary: Dict
    ) -> tuple[str, str]:
        """Simple GRADE assessment based on RoB distribution, I², and k."""
        domains_failed = []

        # Risk of bias domain
        overall = rob_summary.get("overall_counts", {})
        total = rob_summary.get("total_studies", 1)
        high_risk_pct = overall.get("high_risk", 0) / max(total, 1)
        if high_risk_pct >= 0.5:
            domains_failed.append("risk of bias (>50% high risk studies)")
        elif high_risk_pct >= 0.25:
            domains_failed.append("risk of bias (25-50% high risk studies)")

        # Inconsistency domain
        if outcome.i2 is not None:
            if outcome.i2 >= 75:
                domains_failed.append("inconsistency (I²≥75%)")
            elif outcome.i2 >= 50:
                domains_failed.append("inconsistency (I²≥50%)")

        # Imprecision: wide CI relative to MCID (simplified check)
        if outcome.ci_lower is not None and outcome.ci_upper is not None:
            ci_width = abs(outcome.ci_upper - outcome.ci_lower)
            if outcome.studies_included < 4 or ci_width > 2.0:
                domains_failed.append("imprecision (wide CI or small k)")

        # Map to certainty level
        downgrades = min(len(domains_failed), 3)
        levels = ["High", "Moderate", "Low", "Very Low"]
        certainty = levels[downgrades]
        reasoning = (
            f"Started as High (RCT evidence). Downgraded {downgrades} level(s): "
            + ("; ".join(domains_failed) if domains_failed else "no downgrading")
        )
        return certainty, reasoning

    # ── Phase 7: Evidence summary ───────────────────────────────────────────

    def _phase7_evidence_summary(
        self,
        outcome_results: List[OutcomeMetaResult],
        review_question: str,
        plan: Dict,
    ) -> Dict:
        self._agent.load_skills([_SKILL_EVIDENCE_SUM])

        outcomes_summary = [
            {
                "outcome_id": r.outcome_id,
                "outcome_name": r.outcome_name,
                "pooled_estimate": r.pooled_estimate,
                "ci_lower": r.ci_lower, "ci_upper": r.ci_upper,
                "prediction_lower": r.prediction_lower,
                "prediction_upper": r.prediction_upper,
                "i2": r.i2, "tau2": r.tau2,
                "k": r.studies_included,
                "grade_certainty": r.grade_certainty,
                "grade_reasoning": r.grade_reasoning,
                "effect_measure": r.effect_measure,
                "pub_bias": r.publication_bias,
            }
            for r in outcome_results
        ]

        context = self._agent.build_prompt_context(
            active_skill_id=_SKILL_EVIDENCE_SUM,
            task_instruction="Generate a structured evidence summary for this systematic review",
            task_data_dict={
                "review_question": review_question,
                "outcome_results": outcomes_summary,
                "synthesis_plan": plan,
            },
        )
        _raw, summary = self._agent.call_llm(
            context=context, active_skill_id=_SKILL_EVIDENCE_SUM, output_format="json"
        )
        return summary if isinstance(summary, dict) else {}

    # ── Report compilation ──────────────────────────────────────────────────

    def _build_report(
        self,
        outcome_results: List[OutcomeMetaResult],
        plan: Dict,
        feasibility: Dict,
        warnings: List[str],
    ) -> Dict:
        return {
            "total_outcomes_analyzed": len(outcome_results),
            "meta_analysis_outcomes": sum(
                1 for r in outcome_results if r.model_type != "narrative"
            ),
            "narrative_outcomes": sum(
                1 for r in outcome_results if r.model_type == "narrative"
            ),
            "r_was_available": any(r.r_available for r in outcome_results),
            "forest_plots": [r.forest_plot_path for r in outcome_results if r.forest_plot_path],
            "grade_summary": {
                r.outcome_id: r.grade_certainty for r in outcome_results
            },
            "warnings": warnings,
        }

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _get_extraction_table(extraction_result: Any) -> List[Dict]:
        if extraction_result is None:
            return []
        if isinstance(extraction_result, dict):
            return extraction_result.get("extraction_table", [])
        return getattr(extraction_result, "extraction_table", [])

    @staticmethod
    def _get_rob_summary(extraction_result: Any) -> Dict:
        if extraction_result is None:
            return {}
        if isinstance(extraction_result, dict):
            return extraction_result.get("rob_summary", {})
        return getattr(extraction_result, "rob_summary", {})

    @staticmethod
    def _summarize_extraction(table: List[Dict]) -> Dict:
        """Compact summary of extraction table for LLM context."""
        pmids = list({r.get("pmid") for r in table if r.get("pmid")})
        endpoints = list({r.get("endpoint_id") for r in table if r.get("endpoint_id")})
        outcome_types = list({r.get("outcome_type") for r in table if r.get("outcome_type")})
        return {
            "total_rows": len(table),
            "studies": pmids[:20],
            "endpoints": endpoints[:15],
            "outcome_types": outcome_types,
        }

    # ── Save outputs ────────────────────────────────────────────────────────

    def _save_outputs(
        self,
        plan: Dict,
        outcome_results: List[OutcomeMetaResult],
        evidence_summary: Dict,
        report: Dict,
    ) -> None:
        def _save(data: Any, name: str) -> None:
            path = self._save_dir / name
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2, default=str)

        _save(plan, "synthesis_plan.json")
        _save(
            [
                {k: v for k, v in r.__dict__.items()}
                for r in outcome_results
            ],
            "outcome_results.json",
        )
        _save(evidence_summary, "evidence_summary.json")
        _save(report, "synthesis_report.json")


# ── Convenience function ──────────────────────────────────────────────────────

def run_synthesis(
    extraction_result: Any,
    review_question: str,
    save_dir: Optional[pathlib.Path] = None,
    llm_backend=None,
    human_checkpoint_callback: Optional[Callable] = None,
) -> SynthesisResult:
    """
    Run the full synthesis pipeline.

    Args:
        extraction_result: ExtractionResult from extraction_pipeline, or dict with
                           'extraction_table' and 'rob_summary' keys.
                           Set to None to run with empty data (useful for testing).
        review_question:   PICO review question.
        save_dir:          Output directory.
        llm_backend:       LLM backend; defaults to config default model.
        human_checkpoint_callback:
                           Optional: (stage, deliverable, payload) -> bool

    Returns:
        SynthesisResult with outcome_results, evidence_summary, synthesis_report
    """
    pipeline = SynthesisPipeline(
        llm_backend=llm_backend,
        save_dir=save_dir,
        human_checkpoint_callback=human_checkpoint_callback,
    )
    return pipeline.run(
        extraction_result=extraction_result,
        review_question=review_question,
    )
