"""
QualityGateEvaluator — evaluates quality gates defined in the Meta-Skill.

Each gate specifies:
  metric    : string name of the metric to evaluate
  threshold : numeric threshold
  operator  : comparison operator (>=, <=, ==, >, <)
  on_fail   : "warn_and_continue" | "block_until_human" | "retry_stage"

Gate evaluation result carries:
  passed    : bool
  action    : the on_pass or on_fail action string
  message   : human-readable description

The evaluator does NOT block execution — it returns the result and lets
the ProtocolEngine decide what to do based on the action string.
"""

from __future__ import annotations

import logging
import operator as op
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from protocol_layer.meta_skill_parser import QualityGate

logger = logging.getLogger(__name__)

# Supported comparison operators
_OPS = {
    ">=": op.ge,
    "<=": op.le,
    "==": op.eq,
    ">":  op.gt,
    "<":  op.lt,
    "!=": op.ne,
}


@dataclass
class GateResult:
    gate: QualityGate
    metric_value: Any
    passed: bool
    action: str       # "continue", "warn_and_continue", "block_until_human", "retry_stage"
    message: str


class QualityGateEvaluator:
    """
    Evaluates a list of quality gates against a metrics dict.

    Usage:
        evaluator = QualityGateEvaluator()
        results = evaluator.evaluate(gates, metrics)
        for r in results:
            if r.action == "block_until_human":
                ...
    """

    def evaluate(
        self,
        gates: List[QualityGate],
        metrics: Dict[str, Any],
    ) -> List[GateResult]:
        results: List[GateResult] = []
        for gate in gates:
            result = self._eval_one(gate, metrics)
            results.append(result)
            if result.passed:
                logger.info(
                    "Quality gate PASSED: %s = %s %s %s  [%s]",
                    gate.metric, result.metric_value, gate.operator,
                    gate.threshold, gate.stage_id,
                )
            else:
                logger.warning(
                    "Quality gate FAILED: %s = %s %s %s → action=%s  [%s]",
                    gate.metric, result.metric_value, gate.operator,
                    gate.threshold, result.action, gate.stage_id,
                )
        return results

    def evaluate_one(
        self,
        gate: QualityGate,
        metrics: Dict[str, Any],
    ) -> GateResult:
        return self._eval_one(gate, metrics)

    def _eval_one(self, gate: QualityGate, metrics: Dict[str, Any]) -> GateResult:
        value = metrics.get(gate.metric)

        if value is None:
            return GateResult(
                gate=gate,
                metric_value=None,
                passed=False,
                action=gate.on_fail,
                message=f"Metric '{gate.metric}' not found in metrics dict",
            )

        cmp_fn = _OPS.get(gate.operator, op.ge)
        try:
            numeric_value = float(value)
            passed = cmp_fn(numeric_value, float(gate.threshold))
        except (TypeError, ValueError):
            # Non-numeric: fall back to string equality
            passed = str(value) == str(gate.threshold)

        return GateResult(
            gate=gate,
            metric_value=value,
            passed=passed,
            action=gate.on_pass if passed else gate.on_fail,
            message=(
                f"{gate.metric} = {value} {gate.operator} {gate.threshold}: "
                f"{'PASS' if passed else 'FAIL'} — {gate.description}"
            ),
        )

    @staticmethod
    def any_blocking(results: List[GateResult]) -> bool:
        """Return True if any failed gate requires blocking."""
        return any(
            not r.passed and r.action == "block_until_human"
            for r in results
        )

    @staticmethod
    def any_retry(results: List[GateResult]) -> bool:
        """Return True if any failed gate requires stage retry."""
        return any(
            not r.passed and r.action == "retry_stage"
            for r in results
        )

    @staticmethod
    def summary(results: List[GateResult]) -> str:
        lines = []
        for r in results:
            status = "PASS" if r.passed else "FAIL"
            lines.append(f"  [{status}] {r.message}")
        return "\n".join(lines)
