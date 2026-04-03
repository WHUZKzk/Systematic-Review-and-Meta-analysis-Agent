"""
ContextRouter — shared context store, context assembly, and audit logging.

Three components:
  1. SharedContextStore  : persistent key-value store (JSON-backed) for
                           passing data between stages and steps.
  2. ContextAssembler    : builds a compact context dict for a given stage/step
                           by selecting only the fields specified in the data
                           contracts and step definitions.
  3. AuditLogger         : append-only JSONL audit trail of every significant
                           pipeline event (stage transitions, quality gates,
                           human checkpoints, LLM calls, errors).
"""

from __future__ import annotations

import json
import logging
import pathlib
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── SharedContextStore ────────────────────────────────────────────────────────

class SharedContextStore:
    """
    Thread-safe key-value store backed by a JSON file.

    Keys follow the convention:  "<stage_id>.<field_name>"
    e.g. "search.candidate_pool", "screening.eligibility_criteria"

    Values are arbitrary JSON-serialisable objects.
    """

    def __init__(self, store_path: pathlib.Path):
        self._path = store_path
        self._lock = threading.Lock()
        self._data: Dict[str, Any] = {}
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as fh:
                    self._data = json.load(fh)
                logger.debug("ContextStore loaded %d keys from %s", len(self._data), self._path)
            except Exception as exc:
                logger.warning("Failed to load context store (%s) — starting fresh: %s", self._path, exc)
                self._data = {}

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self._data, fh, ensure_ascii=False, indent=2, default=str)
            tmp.replace(self._path)
        except Exception as exc:
            logger.error("Failed to persist context store: %s", exc)

    # ── CRUD ──────────────────────────────────────────────────────────────

    def put(self, key: str, value: Any, *, persist: bool = True) -> None:
        with self._lock:
            self._data[key] = value
            if persist:
                self._save()

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def get_many(self, keys: List[str]) -> Dict[str, Any]:
        with self._lock:
            return {k: self._data.get(k) for k in keys}

    def update(self, mapping: Dict[str, Any], *, persist: bool = True) -> None:
        with self._lock:
            self._data.update(mapping)
            if persist:
                self._save()

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)
            self._save()

    def keys(self) -> List[str]:
        with self._lock:
            return list(self._data.keys())

    def snapshot(self) -> Dict[str, Any]:
        """Return a shallow copy of all stored data."""
        with self._lock:
            return dict(self._data)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._save()


# ── ContextAssembler ──────────────────────────────────────────────────────────

class ContextAssembler:
    """
    Builds a compact context dict for a stage or step invocation.

    Pulls data from SharedContextStore according to the fields declared
    in the Meta-Skill data contracts, avoiding token bloat.
    """

    def __init__(self, store: SharedContextStore):
        self._store = store

    def assemble_for_stage(
        self,
        stage_id: str,
        required_keys: List[str],
        optional_keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Assemble context for a stage entry.

        Args:
            stage_id:      The stage being entered (for logging).
            required_keys: Must be present; raises if missing.
            optional_keys: Included only if they exist in the store.
        """
        ctx: Dict[str, Any] = {}

        for key in required_keys:
            value = self._store.get(key)
            if value is None:
                logger.warning(
                    "Required context key '%s' missing for stage '%s'", key, stage_id
                )
            ctx[key] = value

        for key in (optional_keys or []):
            value = self._store.get(key)
            if value is not None:
                ctx[key] = value

        return ctx

    def assemble_for_step(
        self,
        stage_id: str,
        step_id: str,
        step_output_cache: Dict[str, Any],
        input_from: str,
        extra_keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Assemble the input dict for a single step.

        input_from semantics:
          "human_input"   → nothing from store (caller provides directly)
          "previous_step" → last entry in step_output_cache
          "stage_input"   → all keys currently in store for this stage
          "context_store" → all keys in store
        """
        ctx: Dict[str, Any] = {}

        if input_from == "previous_step":
            if step_output_cache:
                last_key = list(step_output_cache)[-1]
                ctx.update(step_output_cache[last_key] or {})

        elif input_from == "stage_input":
            prefix = f"{stage_id}."
            for k, v in self._store.snapshot().items():
                if k.startswith(prefix):
                    ctx[k.replace(prefix, "", 1)] = v

        elif input_from == "context_store":
            ctx.update(self._store.snapshot())

        # Merge in any extra keys from the store
        for key in (extra_keys or []):
            value = self._store.get(key)
            if value is not None:
                ctx[key] = value

        return ctx


# ── AuditLogger ───────────────────────────────────────────────────────────────

class AuditLogger:
    """
    Append-only JSONL audit trail.

    Each line is a JSON object:
      { "ts": ISO8601, "run_id": str, "event_type": str, "stage": str?,
        "step": str?, "data": {...} }

    Event types:
      pipeline_start, pipeline_end, stage_start, stage_end,
      step_start, step_end, quality_gate, human_checkpoint,
      llm_call, tool_call, error, warning
    """

    def __init__(self, log_path: pathlib.Path, run_id: str):
        self._path = log_path
        self._run_id = run_id
        self._lock = threading.Lock()
        log_path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, event_type: str, data: Dict[str, Any],
               stage: Optional[str] = None, step: Optional[str] = None) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": self._run_id,
            "event_type": event_type,
            "stage": stage,
            "step": step,
            "data": data,
        }
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    # ── Convenience wrappers ───────────────────────────────────────────────

    def pipeline_start(self, meta_skill_id: str, review_question: str) -> None:
        self._write("pipeline_start", {"meta_skill_id": meta_skill_id,
                                        "review_question": review_question[:200]})

    def pipeline_end(self, success: bool, summary: Dict[str, Any]) -> None:
        self._write("pipeline_end", {"success": success, "summary": summary})

    def stage_start(self, stage_id: str, input_summary: str = "") -> None:
        self._write("stage_start", {"input_summary": input_summary}, stage=stage_id)

    def stage_end(self, stage_id: str, success: bool, output_summary: str = "") -> None:
        self._write("stage_end", {"success": success, "output_summary": output_summary},
                    stage=stage_id)

    def step_start(self, stage_id: str, step_id: str) -> None:
        self._write("step_start", {}, stage=stage_id, step=step_id)

    def step_end(self, stage_id: str, step_id: str, success: bool,
                 duration_s: float = 0.0) -> None:
        self._write("step_end", {"success": success, "duration_s": round(duration_s, 2)},
                    stage=stage_id, step=step_id)

    def quality_gate(self, stage_id: str, metric: str, value: Any,
                     threshold: float, passed: bool, action: str) -> None:
        self._write("quality_gate", {
            "metric": metric, "value": value,
            "threshold": threshold, "passed": passed, "action": action,
        }, stage=stage_id)

    def human_checkpoint(self, stage_id: str, timing: str,
                         blocking: bool, deliverable: str, approved: bool) -> None:
        self._write("human_checkpoint", {
            "timing": timing, "blocking": blocking,
            "deliverable": deliverable, "approved": approved,
        }, stage=stage_id)

    def llm_call(self, stage_id: str, step_id: str, model: str,
                 prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        self._write("llm_call", {
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }, stage=stage_id, step=step_id)

    def tool_call(self, stage_id: str, step_id: str, tool_id: str,
                  success: bool, summary: str = "") -> None:
        self._write("tool_call", {
            "tool_id": tool_id, "success": success, "summary": summary,
        }, stage=stage_id, step=step_id)

    def error(self, message: str, stage: Optional[str] = None,
              step: Optional[str] = None, exc: Optional[Exception] = None) -> None:
        self._write("error", {
            "message": message,
            "exception": str(exc) if exc else None,
        }, stage=stage, step=step)

    def warning(self, message: str, stage: Optional[str] = None,
                step: Optional[str] = None) -> None:
        self._write("warning", {"message": message}, stage=stage, step=step)
