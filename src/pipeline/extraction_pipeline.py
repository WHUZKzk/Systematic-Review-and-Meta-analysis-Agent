"""
extraction_pipeline.py — Data extraction and standardization pipeline.

Stages (7 phases):
  Phase 1: Design extraction schema (LLM, skill: extraction_schema_design)
  Phase 2: Fetch full texts (tool: fulltext_fetcher + document_structure_parser)
  Phase 3: Build document maps (LLM per study, skill: document_map_schema)
  Phase 4: Classify outcome types (LLM, skill: outcome_type_classification)
  Phase 5: Extract + localize data (LLM per study, skill: extraction_localization)
  Phase 6: Verify + standardize (tools + LLM, skills: traceability_verification + data_standardization)
  Phase 7: Assess risk of bias (LLM per study, skills: rob2_rct + rob2_information_location)
  Phase 8: Compile extraction report

Entry point for loading papers:
  `included_studies` is a list of dicts from the screening pipeline:
    [{"pmid": "...", "title": "...", "abstract": "...", ...}, ...]
  Full texts are fetched automatically via FullTextFetcherTool (PMC or local PDF).
  To run with pre-loaded texts, set study["full_text"] in each dict before calling run_extraction().

Usage:
    from pipeline.extraction_pipeline import run_extraction, ExtractionResult

    result = run_extraction(
        included_studies=studies,          # list of dicts with pmid, title, abstract
        eligibility_criteria=criteria,     # from screening stage
        review_question="...",
        save_dir=Path("outputs/extraction"),
    )
"""

from __future__ import annotations

import json
import logging
import pathlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import config
from agent_layer import ExecutorAgent, TaskInstruction, StageOutput
from llm_backend import get_registry
from skill_layer import get_loader
from tool_layer.extraction_tools import (
    FullTextFetcherTool, DocumentStructureParser,
    SemanticLocatorTool, PythonSandboxTool, SourceTraceabilityVerifier,
    get_all_extraction_tools,
)

logger = logging.getLogger(__name__)

# ── Skill IDs ─────────────────────────────────────────────────────────────────

_SKILL_DOC_MAP      = "extraction.document_map_schema"
_SKILL_OUTCOME_TYPE = "extraction.outcome_type_classification"
_SKILL_LOCALIZE     = "extraction.extraction_localization"
_SKILL_VERIFY       = "extraction.traceability_verification"
_SKILL_STANDARDIZE  = "extraction.data_standardization"
_SKILL_VALIDATE3    = "extraction.three_layer_validation"
_SKILL_ROB2         = "extraction.rob2_rct"
_SKILL_ROB2_LOC     = "extraction.rob2_information_location"


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class StudyExtraction:
    """Extraction result for a single study."""
    pmid: str
    title: str
    document_map: Dict[str, Any] = field(default_factory=dict)
    outcome_classifications: List[Dict] = field(default_factory=list)
    extractions: List[Dict] = field(default_factory=list)
    standardized_data: List[Dict] = field(default_factory=list)
    validation_summary: Dict[str, Any] = field(default_factory=dict)
    rob_assessment: Dict[str, Any] = field(default_factory=dict)
    extraction_warnings: List[str] = field(default_factory=list)
    full_text_source: str = "unavailable"


@dataclass
class ExtractionResult:
    """Full extraction stage output."""
    extraction_schema: Dict[str, Any]
    study_extractions: List[StudyExtraction]
    extraction_table: List[Dict]          # Flattened table for synthesis
    rob_summary: Dict[str, Any]
    extraction_report: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)


# ── Pipeline ──────────────────────────────────────────────────────────────────

class ExtractionPipeline:
    """
    Multi-phase data extraction pipeline.

    Args:
        llm_backend:   LLM backend (default model)
        save_dir:      Directory for output files
        human_checkpoint_callback:
                       Optional callback for human review points
                       signature: (stage: str, deliverable: str, payload: dict) -> bool
    """

    def __init__(
        self,
        llm_backend=None,
        save_dir: Optional[pathlib.Path] = None,
        human_checkpoint_callback: Optional[Callable] = None,
    ):
        self._llm = llm_backend or get_registry().get("default")
        self._save_dir = save_dir or pathlib.Path("outputs/extraction")
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_cb = human_checkpoint_callback

        skill_loader = get_loader()
        self._agent = ExecutorAgent(llm=self._llm, skill_loader=skill_loader)

        # Register extraction tools
        tools = get_all_extraction_tools()
        self._agent.register_tools(tools)

        # Tool instances (direct use in non-LLM phases)
        self._fulltext_fetcher   = FullTextFetcherTool()
        self._doc_parser         = DocumentStructureParser()
        self._locator            = SemanticLocatorTool()
        self._sandbox            = PythonSandboxTool()
        self._verifier           = SourceTraceabilityVerifier()

    # ── Main entry ─────────────────────────────────────────────────────────

    def run(
        self,
        included_studies: List[Dict],
        eligibility_criteria: Dict,
        review_question: str,
    ) -> ExtractionResult:

        logger.info("ExtractionPipeline: starting with %d included studies",
                    len(included_studies))
        warnings: List[str] = []

        # Phase 1: Design extraction schema
        schema = self._phase1_design_schema(
            review_question, eligibility_criteria, included_studies
        )
        logger.info("Extraction schema designed: %d fields", len(schema.get("fields", [])))

        # Human checkpoint: schema approval
        if self._checkpoint_cb:
            approved = self._checkpoint_cb(
                "extraction_schema",
                "Extraction schema: field definitions, outcome mapping, units",
                {"schema": schema},
            )
            if not approved:
                raise RuntimeError("Extraction schema rejected at human checkpoint")

        # Phase 2-7: Per-study processing
        study_extractions: List[StudyExtraction] = []
        for i, study in enumerate(included_studies):
            pmid = str(study.get("pmid", f"study_{i}"))
            logger.info("Processing study %d/%d: PMID %s", i+1, len(included_studies), pmid)
            try:
                extraction = self._process_study(study, schema, eligibility_criteria)
                study_extractions.append(extraction)
            except Exception as exc:
                logger.error("Extraction failed for PMID %s: %s", pmid, exc)
                warnings.append(f"Extraction failed for PMID {pmid}: {exc}")
                study_extractions.append(StudyExtraction(
                    pmid=pmid,
                    title=study.get("title", ""),
                    extraction_warnings=[str(exc)],
                ))

        # Phase 8: Compile report
        extraction_table = self._build_extraction_table(study_extractions)
        rob_summary = self._build_rob_summary(study_extractions)
        report = self._build_report(study_extractions, schema, warnings)

        # Save outputs
        self._save_outputs(schema, study_extractions, extraction_table, rob_summary, report)

        return ExtractionResult(
            extraction_schema=schema,
            study_extractions=study_extractions,
            extraction_table=extraction_table,
            rob_summary=rob_summary,
            extraction_report=report,
            warnings=warnings,
        )

    # ── Phase 1: Schema design ──────────────────────────────────────────────

    def _phase1_design_schema(
        self,
        review_question: str,
        criteria: Dict,
        sample_studies: List[Dict],
    ) -> Dict:
        """Use LLM to design the extraction schema based on PICO and criteria."""
        self._agent.load_skills([])  # No specific skill for schema design at this layer

        task = TaskInstruction(
            stage="extraction",
            step_id="extraction_step_1_schema",
            input_data={
                "review_question": review_question,
                "eligibility_criteria": criteria,
                "sample_titles": [s.get("title", "") for s in sample_studies[:5]],
            },
            task_description=(
                "Design a structured extraction schema for this systematic review.\n"
                "Define all fields to extract from each included study:\n"
                "  1. Study characteristics (author, year, country, design, N, arms)\n"
                "  2. Population characteristics (age, condition, severity)\n"
                "  3. Intervention details (type, dose, duration, delivery)\n"
                "  4. Comparison/control details\n"
                "  5. Outcome fields (one entry per primary and secondary outcome)\n"
                "  6. Statistical fields (effect size, CI, p-value per outcome)\n"
                "Output as JSON with: fields (list of field defs), outcome_mapping, units_guide."
            ),
        )

        from llm_backend import PromptContext
        context = self._agent.build_prompt_context(
            active_skill_id=None,
            task_instruction=task.task_description,
            task_data_dict=task.input_data,
        )
        _raw, schema = self._agent.call_llm(
            context=context,
            active_skill_id=None,
            output_format="json",
        )

        if not isinstance(schema, dict):
            schema = {"fields": [], "outcome_mapping": {}, "units_guide": {}}

        schema.setdefault("fields", [])
        schema.setdefault("outcome_mapping", {})
        schema.setdefault("review_question", review_question)
        return schema

    # ── Per-study processing ────────────────────────────────────────────────

    def _process_study(
        self,
        study: Dict,
        schema: Dict,
        criteria: Dict,
    ) -> StudyExtraction:
        pmid = str(study.get("pmid", ""))
        title = study.get("title", "")
        extraction = StudyExtraction(pmid=pmid, title=title)

        # Phase 2: Fetch full text
        full_text, fmt, source = self._fetch_full_text(study)
        extraction.full_text_source = source

        if not full_text:
            extraction.extraction_warnings.append("Full text unavailable; using abstract only")
            full_text = study.get("abstract", "")
            fmt = "text"
            source = "abstract"

        # Parse document structure
        parse_result = self._doc_parser.execute({
            "full_text": full_text,
            "format": fmt,
            "pmid": pmid,
        })
        sections: List[Dict] = parse_result.data.get("sections", []) if parse_result.success else []

        if not sections:
            extraction.extraction_warnings.append("Document parsing produced no sections")
            sections = [{"location_id": "FullText", "title": "Full Text",
                          "text": full_text[:8000], "type": "section", "depth": 0}]

        # Phase 3: Document map
        extraction.document_map = self._phase3_document_map(pmid, title, full_text, sections)

        # Phase 4: Classify outcome types
        extraction.outcome_classifications = self._phase4_classify_outcomes(
            pmid, extraction.document_map
        )

        # Phase 5: Extract and localize
        extraction.extractions = self._phase5_extract(
            pmid, extraction.document_map, sections, criteria
        )

        # Phase 6: Verify and standardize
        verified = self._phase6_verify(pmid, extraction.extractions, sections)
        extraction.standardized_data, extraction.validation_summary = self._phase6_standardize(
            pmid, verified, extraction.document_map
        )

        # Phase 7: RoB assessment
        extraction.rob_assessment = self._phase7_rob(pmid, full_text, sections, extraction.document_map)

        return extraction

    def _fetch_full_text(self, study: Dict):
        """Return (text, format, source) for a study."""
        # Pre-loaded full text takes priority
        if study.get("full_text"):
            return study["full_text"], study.get("full_text_format", "text"), "preloaded"

        pmid = str(study.get("pmid", ""))
        if pmid:
            result = self._fulltext_fetcher.execute({"pmid": pmid})
            if result.success:
                return (
                    result.data["full_text"],
                    result.data.get("format", "text"),
                    result.data.get("source", "pmc"),
                )

        return None, "text", "unavailable"

    # ── Phase 3: Document map ───────────────────────────────────────────────

    def _phase3_document_map(
        self, pmid: str, title: str, full_text: str, sections: List[Dict]
    ) -> Dict:
        self._agent.load_skills([_SKILL_DOC_MAP])

        task_data = {
            "pmid": pmid,
            "title": title,
            "full_text_preview": full_text[:3000],
            "sections_available": [s["location_id"] for s in sections],
        }
        context = self._agent.build_prompt_context(
            active_skill_id=_SKILL_DOC_MAP,
            task_instruction=f"Build a document map for PMID {pmid}: '{title}'",
            task_data_dict=task_data,
        )
        _raw, doc_map = self._agent.call_llm(
            context=context,
            active_skill_id=_SKILL_DOC_MAP,
            output_format="json",
        )
        if not isinstance(doc_map, dict):
            doc_map = {"pmid": pmid, "arms": [], "endpoints": [], "timepoints": []}
        doc_map["pmid"] = pmid
        return doc_map

    # ── Phase 4: Outcome type classification ───────────────────────────────

    def _phase4_classify_outcomes(self, pmid: str, doc_map: Dict) -> List[Dict]:
        endpoints = doc_map.get("endpoints", [])
        if not endpoints:
            return []

        self._agent.load_skills([_SKILL_OUTCOME_TYPE])

        context = self._agent.build_prompt_context(
            active_skill_id=_SKILL_OUTCOME_TYPE,
            task_instruction=f"Classify outcome types for PMID {pmid}",
            task_data_dict={"pmid": pmid, "endpoints": endpoints},
        )
        _raw, result = self._agent.call_llm(
            context=context,
            active_skill_id=_SKILL_OUTCOME_TYPE,
            output_format="json",
        )
        if isinstance(result, dict):
            return result.get("classifications", [])
        return []

    # ── Phase 5: Extract and localize ──────────────────────────────────────

    def _phase5_extract(
        self,
        pmid: str,
        doc_map: Dict,
        sections: List[Dict],
        criteria: Dict,
    ) -> List[Dict]:
        arms = doc_map.get("arms", [])
        endpoints = doc_map.get("endpoints", [])
        timepoints = doc_map.get("timepoints", [])

        if not arms or not endpoints:
            return []

        # Use semantic locator to find relevant sections first
        section_texts = "\n\n".join(
            f"[{s['location_id']}] {s['title']}\n{s['text'][:500]}"
            for s in sections
        )

        self._agent.load_skills([_SKILL_LOCALIZE])

        task_data = {
            "pmid": pmid,
            "document_map": doc_map,
            "sections_index": [{"location_id": s["location_id"],
                                  "title": s["title"], "type": s["type"]}
                                 for s in sections],
            "criteria_summary": {
                k: v for k, v in criteria.items()
                if k in ("inclusion_criteria", "primary_outcomes")
            },
        }
        context = self._agent.build_prompt_context(
            active_skill_id=_SKILL_LOCALIZE,
            task_instruction=f"Extract all numeric data for PMID {pmid}",
            task_data_dict=task_data,
        )
        _raw, result = self._agent.call_llm(
            context=context,
            active_skill_id=_SKILL_LOCALIZE,
            output_format="json",
        )
        if isinstance(result, dict):
            return result.get("extractions", [])
        return []

    # ── Phase 6: Verify and standardize ────────────────────────────────────

    def _phase6_verify(
        self,
        pmid: str,
        extractions: List[Dict],
        sections: List[Dict],
    ) -> List[Dict]:
        if not extractions:
            return []

        # Tool-based verification
        verified = []
        for ext in extractions:
            location_id = ext.get("location_id", "")
            values = ext.get("values", {})
            main_value = next(iter(values.values()), None) if values else None

            if main_value is not None and location_id:
                vr = self._verifier.execute({
                    "value": main_value,
                    "location_id": location_id,
                    "sections": sections,
                    "tolerance": 0.5,
                })
                if vr.success:
                    ext = {**ext, "verification_status": vr.data.get("status", "mismatch"),
                           "confirmed_location_id": vr.data.get("found_at", location_id)}
                else:
                    ext = {**ext, "verification_status": "mismatch",
                           "confirmed_location_id": location_id}
            else:
                ext = {**ext, "verification_status": "unverified",
                       "confirmed_location_id": location_id}
            verified.append(ext)

        return verified

    def _phase6_standardize(
        self,
        pmid: str,
        verified_extractions: List[Dict],
        doc_map: Dict,
    ) -> tuple[List[Dict], Dict]:
        if not verified_extractions:
            return [], {}

        self._agent.load_skills([_SKILL_STANDARDIZE, _SKILL_VALIDATE3])

        task_data = {
            "pmid": pmid,
            "verified_extractions": verified_extractions,
            "document_map": doc_map,
        }
        context = self._agent.build_prompt_context(
            active_skill_id=_SKILL_STANDARDIZE,
            task_instruction=f"Standardize and validate extracted data for PMID {pmid}",
            task_data_dict=task_data,
        )
        _raw, result = self._agent.call_llm(
            context=context,
            active_skill_id=_SKILL_STANDARDIZE,
            output_format="json",
        )

        if isinstance(result, dict):
            standardized = result.get("standardized_data", verified_extractions)
            validation = {
                "overall_quality_score": result.get("overall_quality_score", None),
                "study_level_flag": result.get("study_level_flag", "unknown"),
            }
        else:
            standardized = verified_extractions
            validation = {}

        # Run Python sandbox for numeric conversions flagged in standardization
        standardized = self._apply_numeric_conversions(standardized)

        return standardized, validation

    def _apply_numeric_conversions(self, standardized: List[Dict]) -> List[Dict]:
        """Execute any pending numeric conversions via PythonSandbox."""
        result = []
        for item in standardized:
            template = item.get("template_applied", "")
            values = item.get("values", {})

            if "template_2" in template and "se" in values and "n" in values:
                code = "sd = se * sqrt(n)"
                sr = self._sandbox.execute({"code": code,
                                             "input_vars": {"se": values["se"], "n": values["n"]}})
                if sr.success and "sd" in sr.data:
                    values = {**values, "sd": round(sr.data["sd"], 4)}
                    item = {**item, "values": values}

            elif "template_3" in template and all(k in values for k in ("ci_lower", "ci_upper", "n")):
                code = "sd = (ci_upper - ci_lower) / (2 * 1.96) * sqrt(n)"
                sr = self._sandbox.execute({
                    "code": code,
                    "input_vars": {
                        "ci_lower": values["ci_lower"],
                        "ci_upper": values["ci_upper"],
                        "n": values["n"],
                    },
                })
                if sr.success and "sd" in sr.data:
                    values = {**values, "sd": round(sr.data["sd"], 4)}
                    item = {**item, "values": values}

            result.append(item)
        return result

    # ── Phase 7: Risk of bias ───────────────────────────────────────────────

    def _phase7_rob(
        self,
        pmid: str,
        full_text: str,
        sections: List[Dict],
        doc_map: Dict,
    ) -> Dict:
        self._agent.load_skills([_SKILL_ROB2, _SKILL_ROB2_LOC])

        # Use semantic locator to pre-locate RoB-relevant sections
        rob_relevant_sections = []
        for query in ["randomization allocation concealment", "blinding masking",
                       "missing data attrition ITT", "outcome assessment blinded",
                       "trial registration pre-specified protocol"]:
            lr = self._locator.execute({"query": query, "sections": sections, "top_k": 2})
            if lr.success:
                rob_relevant_sections.extend(lr.data.get("results", []))

        # Deduplicate
        seen_ids = set()
        unique_sections = []
        for s in rob_relevant_sections:
            if s["location_id"] not in seen_ids:
                seen_ids.add(s["location_id"])
                unique_sections.append(s)

        task_data = {
            "pmid": pmid,
            "document_map": {"arms": doc_map.get("arms", [])},
            "relevant_sections": [
                {"location_id": s["location_id"],
                 "title": s["title"],
                 "text": s["text"][:600]}
                for s in unique_sections[:8]
            ],
        }
        context = self._agent.build_prompt_context(
            active_skill_id=_SKILL_ROB2,
            task_instruction=f"Assess risk of bias using RoB 2 tool for PMID {pmid}",
            task_data_dict=task_data,
        )
        _raw, result = self._agent.call_llm(
            context=context,
            active_skill_id=_SKILL_ROB2,
            output_format="json",
        )
        if isinstance(result, dict):
            result["pmid"] = pmid
            return result
        return {"pmid": pmid, "overall_judgment": "unclear", "domain_assessments": []}

    # ── Phase 8: Report compilation ─────────────────────────────────────────

    def _build_extraction_table(
        self, study_extractions: List[StudyExtraction]
    ) -> List[Dict]:
        """Build a flat extraction table for synthesis."""
        rows = []
        for se in study_extractions:
            for item in se.standardized_data:
                row = {
                    "pmid": se.pmid,
                    "title": se.title,
                    "endpoint_id": item.get("endpoint_id"),
                    "arm_id": item.get("arm_id"),
                    "timepoint_id": item.get("timepoint_id"),
                    "outcome_type": item.get("outcome_type"),
                    "standardized_form": item.get("standardized_form"),
                    **item.get("values", {}),
                    "template_applied": item.get("template_applied", ""),
                    "location_id": item.get("location_id", ""),
                    "verification_status": item.get("verification_status", ""),
                }
                rows.append(row)
        return rows

    def _build_rob_summary(self, study_extractions: List[StudyExtraction]) -> Dict:
        """Summarize RoB assessments across all studies."""
        domain_counts: Dict[str, Dict[str, int]] = {}
        overall_counts: Dict[str, int] = {"low_risk": 0, "some_concerns": 0, "high_risk": 0, "unclear": 0}

        for se in study_extractions:
            rob = se.rob_assessment
            overall = rob.get("overall_judgment", "unclear")
            overall_counts[overall] = overall_counts.get(overall, 0) + 1

            for da in rob.get("domain_assessments", []):
                did = da.get("domain_id", "unknown")
                judgment = da.get("domain_judgment", "unclear")
                if did not in domain_counts:
                    domain_counts[did] = {"low_risk": 0, "some_concerns": 0, "high_risk": 0}
                domain_counts[did][judgment] = domain_counts[did].get(judgment, 0) + 1

        return {
            "overall_counts": overall_counts,
            "domain_counts": domain_counts,
            "total_studies": len(study_extractions),
            "high_risk_studies": [
                se.pmid for se in study_extractions
                if se.rob_assessment.get("overall_judgment") == "high_risk"
            ],
        }

    def _build_report(
        self,
        study_extractions: List[StudyExtraction],
        schema: Dict,
        pipeline_warnings: List[str],
    ) -> Dict:
        total = len(study_extractions)
        successful = sum(
            1 for se in study_extractions
            if se.standardized_data or se.document_map.get("arms")
        )
        full_text_sources = {}
        for se in study_extractions:
            src = se.full_text_source
            full_text_sources[src] = full_text_sources.get(src, 0) + 1

        quality_scores = [
            se.validation_summary.get("overall_quality_score", None)
            for se in study_extractions
            if se.validation_summary.get("overall_quality_score") is not None
        ]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else None

        return {
            "total_studies": total,
            "successful_extractions": successful,
            "full_text_sources": full_text_sources,
            "average_quality_score": round(avg_quality, 3) if avg_quality else None,
            "schema_fields_count": len(schema.get("fields", [])),
            "pipeline_warnings": pipeline_warnings,
            "study_level_flags": {
                se.pmid: se.validation_summary.get("study_level_flag", "unknown")
                for se in study_extractions
            },
        }

    # ── Save outputs ────────────────────────────────────────────────────────

    def _save_outputs(
        self,
        schema: Dict,
        study_extractions: List[StudyExtraction],
        extraction_table: List[Dict],
        rob_summary: Dict,
        report: Dict,
    ) -> None:
        def _save(data: Any, name: str) -> None:
            path = self._save_dir / name
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2, default=str)
            logger.info("Saved: %s", path)

        _save(schema, "extraction_schema.json")
        _save(extraction_table, "extraction_table.json")
        _save(rob_summary, "rob_summary.json")
        _save(report, "extraction_report.json")

        # Per-study files
        study_dir = self._save_dir / "studies"
        study_dir.mkdir(exist_ok=True)
        for se in study_extractions:
            data = {
                "pmid": se.pmid,
                "title": se.title,
                "full_text_source": se.full_text_source,
                "document_map": se.document_map,
                "outcome_classifications": se.outcome_classifications,
                "standardized_data": se.standardized_data,
                "validation_summary": se.validation_summary,
                "rob_assessment": se.rob_assessment,
                "warnings": se.extraction_warnings,
            }
            _save(data, f"studies/{se.pmid}.json")


# ── Convenience function ──────────────────────────────────────────────────────

def run_extraction(
    included_studies: List[Dict],
    eligibility_criteria: Dict,
    review_question: str,
    save_dir: Optional[pathlib.Path] = None,
    llm_backend=None,
    human_checkpoint_callback: Optional[Callable] = None,
) -> ExtractionResult:
    """
    Run the full extraction pipeline.

    Args:
        included_studies:     List of dicts from screening, each with pmid/title/abstract.
                              Optional: set study["full_text"] to skip API fetching.
        eligibility_criteria: Criteria dict from screening stage.
        review_question:      PICO review question string.
        save_dir:             Output directory.
        llm_backend:          LLM backend; defaults to config default model.
        human_checkpoint_callback:
                              Optional: (stage, deliverable, payload) -> bool

    Returns:
        ExtractionResult with extraction_table, rob_summary, extraction_report
    """
    pipeline = ExtractionPipeline(
        llm_backend=llm_backend,
        save_dir=save_dir,
        human_checkpoint_callback=human_checkpoint_callback,
    )
    return pipeline.run(
        included_studies=included_studies,
        eligibility_criteria=eligibility_criteria,
        review_question=review_question,
    )
