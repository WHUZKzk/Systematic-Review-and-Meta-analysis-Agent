"""
ProtocolEngine — the top-level orchestrator for the SR pipeline.

Responsibilities:
  1. Load and parse the Meta-Skill YAML.
  2. Execute stages in sequence, delegating to the correct pipeline module.
  3. Enforce data contracts between stages (validate required fields).
  4. Invoke quality gates and route based on results.
  5. Surface human checkpoints via a callback interface.
  6. Write every significant event to the AuditLogger.

Usage:
    engine = ProtocolEngine(
        meta_skill_path=Path("meta_skills/prisma_2020_therapeutic.yaml"),
        run_id="run_001",
        human_checkpoint_callback=my_callback,   # optional
    )
    result = engine.run(
        review_question="...",
        stages=["search", "screening"],   # subset, or None for all
        search_kwargs={},
        screening_kwargs={},
    )

The engine is intentionally thin: stage execution logic lives inside the
individual pipeline modules (search_pipeline.py, screening_pipeline.py, …).
The engine handles the orchestration envelope only.
"""

from __future__ import annotations

import json
import logging
import pathlib
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import config
from protocol_layer.meta_skill_parser import MetaSkill, MetaSkillParser, StageDef
from protocol_layer.context_router import SharedContextStore, AuditLogger
from protocol_layer.quality_gate_evaluator import QualityGateEvaluator, GateResult

logger = logging.getLogger(__name__)

# Human-checkpoint callback signature:
#   callback(stage_id: str, timing: str, deliverable: str, payload: dict) -> bool (approved)
HumanCheckpointCallback = Callable[[str, str, str, Dict], bool]


class PipelineError(Exception):
    """Raised when an unrecoverable pipeline error occurs."""


class QualityGateBlock(Exception):
    """Raised when a blocking quality gate fails."""
    def __init__(self, message: str, gate_results: List[GateResult]):
        super().__init__(message)
        self.gate_results = gate_results


class ProtocolEngine:
    """
    Top-level pipeline orchestrator.

    Args:
        meta_skill_path:              Path to Meta-Skill YAML.
        run_id:                       Unique run identifier (auto-generated if None).
        human_checkpoint_callback:    Optional callback; if None, all checkpoints auto-approve.
        context_store_path:           Path for persistent context store JSON.
        audit_log_path:               Path for JSONL audit log.
    """

    def __init__(
        self,
        meta_skill_path: pathlib.Path,
        run_id: Optional[str] = None,
        human_checkpoint_callback: Optional[HumanCheckpointCallback] = None,
        context_store_path: Optional[pathlib.Path] = None,
        audit_log_path: Optional[pathlib.Path] = None,
    ):
        self.run_id = run_id or f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.meta_skill: MetaSkill = MetaSkillParser.load(meta_skill_path)
        self._checkpoint_callback = human_checkpoint_callback
        self._gate_evaluator = QualityGateEvaluator()

        # Context store (persistent across stages)
        store_path = context_store_path or (
            config.CONTEXT_STORE_DIR / f"{self.run_id}_context.json"
        )
        self.context_store = SharedContextStore(store_path)

        # Audit log
        log_path = audit_log_path or (
            config.CONTEXT_STORE_DIR / f"{self.run_id}_audit.jsonl"
        )
        self.audit = AuditLogger(log_path, run_id=self.run_id)

        logger.info(
            "ProtocolEngine initialised: run_id=%s, meta_skill=%s",
            self.run_id, self.meta_skill.meta_skill_id,
        )

    # ── Main entry point ──────────────────────────────────────────────────

    def run(
        self,
        review_question: str,
        stages: Optional[List[str]] = None,
        save_dir: Optional[pathlib.Path] = None,
        **stage_kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Execute the pipeline.

        Args:
            review_question: Natural-language PICO research question.
            stages:          List of stage_ids to run (None = all).
            save_dir:        Directory for output files.
            **stage_kwargs:  Per-stage keyword arguments passed to pipeline functions.
                             Key pattern: "<stage_id>_kwargs" → dict
                             e.g. search_kwargs={"max_results": 1000}

        Returns:
            Dict with per-stage results and overall status.
        """
        save_dir = save_dir or pathlib.Path("outputs") / self.run_id
        save_dir.mkdir(parents=True, exist_ok=True)

        self.audit.pipeline_start(self.meta_skill.meta_skill_id, review_question)
        self.context_store.put("review_question", review_question)

        target_stages = stages or [s.stage_id for s in self.meta_skill.stages]
        pipeline_results: Dict[str, Any] = {}
        previous_stage_id: Optional[str] = None

        try:
            for stage_def in self.meta_skill.stages:
                if stage_def.stage_id not in target_stages:
                    continue

                logger.info("=== Starting stage: %s ===", stage_def.stage_id)
                self.audit.stage_start(stage_def.stage_id)
                t0 = time.time()

                # Validate incoming data contract
                if previous_stage_id:
                    self._validate_contract(previous_stage_id, stage_def.stage_id)

                # Get stage-specific kwargs
                kwargs = stage_kwargs.get(f"{stage_def.stage_id}_kwargs", {})

                # Execute stage
                stage_result = self._run_stage(
                    stage_def=stage_def,
                    review_question=review_question,
                    save_dir=save_dir,
                    **kwargs,
                )

                pipeline_results[stage_def.stage_id] = stage_result

                # Store stage output in context store
                self.context_store.put(
                    f"{stage_def.stage_id}.result",
                    self._serialize_result(stage_result),
                )

                # Evaluate post-stage quality gates
                self._evaluate_gates(
                    stage_def.stage_id,
                    f"after_stage",
                    self._extract_metrics(stage_def.stage_id, stage_result),
                )

                elapsed = time.time() - t0
                self.audit.stage_end(
                    stage_def.stage_id,
                    success=True,
                    output_summary=f"Completed in {elapsed:.1f}s",
                )
                logger.info("Stage '%s' completed in %.1fs", stage_def.stage_id, elapsed)
                previous_stage_id = stage_def.stage_id

        except QualityGateBlock as exc:
            self.audit.error(str(exc))
            pipeline_results["_error"] = {"type": "quality_gate_block", "message": str(exc)}
            raise

        except Exception as exc:
            self.audit.error(str(exc), exc=exc)
            logger.exception("Unrecoverable pipeline error: %s", exc)
            if self.meta_skill.error_policy.on_unrecoverable_error == "halt_and_notify":
                pipeline_results["_error"] = {"type": "unrecoverable", "message": str(exc)}
                raise PipelineError(str(exc)) from exc
            else:
                pipeline_results["_error"] = {"type": "skipped", "message": str(exc)}

        self.audit.pipeline_end(
            success="_error" not in pipeline_results,
            summary={k: "ok" for k in pipeline_results if not k.startswith("_")},
        )
        return pipeline_results

    # ── Stage dispatch ────────────────────────────────────────────────────

    def _run_stage(
        self,
        stage_def: StageDef,
        review_question: str,
        save_dir: pathlib.Path,
        **kwargs: Any,
    ) -> Any:
        """Dispatch to the appropriate pipeline module for a stage."""
        sid = stage_def.stage_id

        if sid == "search":
            return self._run_search_stage(review_question, save_dir, **kwargs)
        elif sid == "screening":
            return self._run_screening_stage(save_dir, **kwargs)
        elif sid == "extraction":
            return self._run_extraction_stage(save_dir, **kwargs)
        elif sid == "synthesis":
            return self._run_synthesis_stage(save_dir, **kwargs)
        else:
            raise PipelineError(f"Unknown stage_id: {sid!r}")

    def _run_search_stage(
        self, review_question: str, save_dir: pathlib.Path, **kwargs
    ) -> Any:
        from pipeline.search_pipeline import run_search
        from llm_backend import get_registry

        llm = get_registry().get("default")
        result = run_search(
            pico=review_question,
            save_dir=save_dir / "search",
            llm_backend=llm,
            **kwargs,
        )
        # Persist outputs to context store
        self.context_store.update({
            "search.candidate_pool": [p.__dict__ if hasattr(p, "__dict__") else p
                                       for p in result.candidate_pool],
            "search.pico_terms":     result.pico_terms,
            "search.main_query":     result.main_query,
            "search.search_report":  result.search_report,
        })
        return result

    def _run_screening_stage(self, save_dir: pathlib.Path, **kwargs) -> Any:
        from pipeline.screening_pipeline import run_screening

        candidate_pool = self.context_store.get("search.candidate_pool", [])
        pico_terms = self.context_store.get("search.pico_terms", {})
        review_question = self.context_store.get("review_question", "")

        result = run_screening(
            candidate_pool=candidate_pool,
            review_question=review_question,
            pico_terms=pico_terms,
            save_dir=save_dir / "screening",
            human_checkpoint_callback=(
                self._screening_checkpoint_adapter()
                if self._checkpoint_callback else None
            ),
            **kwargs,
        )
        # Persist outputs to context store
        self.context_store.update({
            "screening.included_studies":  [
                s.__dict__ if hasattr(s, "__dict__") else s
                for s in result.included_studies
            ],
            "screening.excluded_studies":  result.excluded_studies,
            "screening.criteria":          result.criteria,
            "screening.screening_report":  result.screening_report,
        })
        return result

    def _run_extraction_stage(self, save_dir: pathlib.Path, **kwargs) -> Any:
        from pipeline.extraction_pipeline import run_extraction

        included = self.context_store.get("screening.included_studies", [])
        criteria = self.context_store.get("screening.criteria", {})
        review_question = self.context_store.get("review_question", "")

        return run_extraction(
            included_studies=included,
            eligibility_criteria=criteria,
            review_question=review_question,
            save_dir=save_dir / "extraction",
            **kwargs,
        )

    def _run_synthesis_stage(self, save_dir: pathlib.Path, **kwargs) -> Any:
        from pipeline.synthesis_pipeline import run_synthesis

        extraction_result = self.context_store.get("extraction.result")
        review_question = self.context_store.get("review_question", "")

        return run_synthesis(
            extraction_result=extraction_result,
            review_question=review_question,
            save_dir=save_dir / "synthesis",
            **kwargs,
        )

    # ── Human checkpoints ─────────────────────────────────────────────────

    def trigger_human_checkpoint(
        self,
        stage_id: str,
        timing: str,
        deliverable: str,
        payload: Dict[str, Any],
        blocking: bool = True,
    ) -> bool:
        """
        Invoke the human checkpoint callback.

        Returns True (approved) when:
          - callback approves, OR
          - no callback registered (auto-approve), OR
          - checkpoint is non-blocking
        """
        if not blocking:
            logger.info("Non-blocking checkpoint at %s/%s — auto-continuing", stage_id, timing)
            if self._checkpoint_callback:
                self._checkpoint_callback(stage_id, timing, deliverable, payload)
            self.audit.human_checkpoint(stage_id, timing, blocking=False,
                                         deliverable=deliverable, approved=True)
            return True

        if not self._checkpoint_callback:
            logger.info("No checkpoint callback — auto-approving %s/%s", stage_id, timing)
            self.audit.human_checkpoint(stage_id, timing, blocking=True,
                                         deliverable=deliverable, approved=True)
            return True

        approved = self._checkpoint_callback(stage_id, timing, deliverable, payload)
        self.audit.human_checkpoint(stage_id, timing, blocking=True,
                                     deliverable=deliverable, approved=approved)
        return approved

    def _screening_checkpoint_adapter(self) -> Callable:
        """Wrap the engine's checkpoint callback to match ScreeningPipeline's expected signature."""
        def adapter(stage: str, deliverable: str, payload: dict) -> bool:
            return self.trigger_human_checkpoint(
                stage_id="screening",
                timing=f"after_{stage}",
                deliverable=deliverable,
                payload=payload,
                blocking=True,
            )
        return adapter

    # ── Quality gate helpers ──────────────────────────────────────────────

    def _evaluate_gates(
        self,
        stage_id: str,
        timing: str,
        metrics: Dict[str, Any],
    ) -> List[GateResult]:
        gates = self.meta_skill.get_quality_gates(stage_id, timing)
        if not gates:
            return []

        results = self._gate_evaluator.evaluate(gates, metrics)

        for r in results:
            self.audit.quality_gate(
                stage_id=stage_id,
                metric=r.gate.metric,
                value=r.metric_value,
                threshold=r.gate.threshold,
                passed=r.passed,
                action=r.action,
            )

        if QualityGateEvaluator.any_blocking(results):
            failing = [r for r in results if not r.passed and r.action == "block_until_human"]
            msg = "Blocking quality gate(s) failed:\n" + "\n".join(r.message for r in failing)
            raise QualityGateBlock(msg, failing)

        return results

    def _extract_metrics(self, stage_id: str, result: Any) -> Dict[str, Any]:
        """Extract quality-gate-relevant metrics from a stage result object."""
        metrics: Dict[str, Any] = {}
        if stage_id == "search":
            metrics["candidate_pool_size"] = len(getattr(result, "candidate_pool", []))
            pico = getattr(result, "pico_terms", {})
            covered = sum(
                1 for dim in ["population", "intervention", "comparison", "outcome"]
                if pico.get(dim)
            )
            metrics["pico_dimensions_covered"] = covered
        elif stage_id == "screening":
            metrics["included_count"] = len(getattr(result, "included_studies", []))
            report = getattr(result, "screening_report", {})
            metrics["kappa"] = report.get("kappa", 1.0)
        elif stage_id == "extraction":
            table = getattr(result, "extraction_table", [])
            if table:
                total_fields = sum(len(r) for r in table)
                filled_fields = sum(
                    1 for row in table for v in row.values() if v not in (None, "", "NR")
                )
                metrics["extraction_completeness_rate"] = (
                    filled_fields / total_fields if total_fields else 0.0
                )
        elif stage_id == "synthesis":
            metrics["studies_in_meta"] = getattr(result, "studies_in_meta", 0)
        return metrics

    # ── Data contract validation ──────────────────────────────────────────

    def _validate_contract(self, from_stage: str, to_stage: str) -> None:
        contract = self.meta_skill.get_data_contract(from_stage, to_stage)
        if not contract:
            return
        missing = []
        for field in contract.required_fields:
            key = f"{from_stage}.{field.field_name}"
            if self.context_store.get(key) is None:
                missing.append(key)
        if missing:
            logger.warning(
                "Data contract %s→%s: missing required fields: %s",
                from_stage, to_stage, missing,
            )

    # ── Serialisation helpers ─────────────────────────────────────────────

    @staticmethod
    def _serialize_result(result: Any) -> Any:
        if hasattr(result, "__dict__"):
            return {
                k: ProtocolEngine._serialize_result(v)
                for k, v in result.__dict__.items()
                if not k.startswith("_")
            }
        if isinstance(result, (list, tuple)):
            return [ProtocolEngine._serialize_result(i) for i in result]
        if isinstance(result, dict):
            return {k: ProtocolEngine._serialize_result(v) for k, v in result.items()}
        return result
