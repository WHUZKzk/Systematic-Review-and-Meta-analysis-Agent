"""
Skill Validator — validates LLM output against a Skill's validation rules.

Each ValidationRule carries an `expression` (Python-style pseudo-code).
The validator evaluates the expression in a restricted sandbox with the
LLM output dict bound to `output`.

Severity semantics:
  hard  → validation failure must be corrected before pipeline continues
  soft  → validation failure is recorded as a warning; pipeline may continue

on_fail actions:
  retry       → caller should request an LLM correction (up to LLM_MAX_RETRIES)
  flag_human  → mark item for human review; pipeline continues
  reject      → outright reject the output (treat as hard-fail)
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

from .skill_schema import Skill, ValidationRule

logger = logging.getLogger(__name__)


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class RuleResult:
    rule_id: str
    passed: bool
    severity: str       # hard | soft
    on_fail: str        # retry | flag_human | reject
    message: str = ""


@dataclass
class ValidationResult:
    passed: bool                        # True only if all hard rules passed
    rule_results: List[RuleResult] = field(default_factory=list)
    flagged_for_human: bool = False
    should_retry: bool = False          # True if any hard rule requests retry
    hard_failures: List[str] = field(default_factory=list)
    soft_warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"ValidationResult: {'PASS' if self.passed else 'FAIL'}"]
        for rr in self.rule_results:
            status = "OK" if rr.passed else f"FAIL({rr.severity})"
            lines.append(f"  [{status}] {rr.rule_id}: {rr.message}")
        return "\n".join(lines)


# ── Validator ─────────────────────────────────────────────────────────────────

class SkillValidator:
    """
    Validates a dict output against the ValidationRule list of a Skill.

    The expression in each rule is evaluated with:
        output  — the LLM output dict
        len, list, dict, str, int, float, bool, type — builtins
    """

    # Safe builtins accessible inside rule expressions
    _SAFE_BUILTINS = {
        "len": len, "list": list, "dict": dict, "str": str,
        "int": int, "float": float, "bool": bool, "type": type,
        "isinstance": isinstance, "any": any, "all": all,
        "min": min, "max": max, "sum": sum,
    }

    def validate(self, output: Any, skill: Skill) -> ValidationResult:
        """
        Run all validation rules in `skill` against `output`.

        Args:
            output: The parsed LLM output (typically a dict).
            skill:  The Skill whose validation rules to apply.

        Returns:
            ValidationResult with pass/fail status and per-rule details.
        """
        rule_results: List[RuleResult] = []
        hard_failures: List[str] = []
        soft_warnings: List[str] = []
        flagged = False
        should_retry = False

        for rule in skill.validation:
            result = self._evaluate_rule(rule, output)
            rule_results.append(result)

            if not result.passed:
                if rule.severity == "hard":
                    hard_failures.append(f"{rule.rule_id}: {result.message}")
                    if rule.on_fail == "retry":
                        should_retry = True
                    elif rule.on_fail == "flag_human":
                        flagged = True
                else:  # soft
                    soft_warnings.append(f"{rule.rule_id}: {result.message}")
                    if rule.on_fail == "flag_human":
                        flagged = True

                logger.debug(
                    "Skill %s — rule %s %s: %s",
                    skill.skill_id, rule.rule_id,
                    "FAIL" if not result.passed else "WARN",
                    result.message,
                )

        passed = len(hard_failures) == 0

        vr = ValidationResult(
            passed=passed,
            rule_results=rule_results,
            flagged_for_human=flagged,
            should_retry=should_retry and not passed,
            hard_failures=hard_failures,
            soft_warnings=soft_warnings,
        )

        if not passed:
            logger.warning(
                "Skill validation FAILED for %s: %s",
                skill.skill_id, hard_failures,
            )
        elif soft_warnings:
            logger.info(
                "Skill validation passed with warnings for %s: %s",
                skill.skill_id, soft_warnings,
            )

        return vr

    def _evaluate_rule(self, rule: ValidationRule, output: Any) -> RuleResult:
        """Evaluate a single rule expression against `output`."""
        try:
            passed = bool(
                eval(  # noqa: S307
                    rule.expression,
                    {"__builtins__": self._SAFE_BUILTINS},
                    {"output": output},
                )
            )
            msg = "" if passed else f"Expression evaluated to False: {rule.expression}"
        except Exception as exc:
            passed = False
            msg = f"Expression error: {exc}"

        return RuleResult(
            rule_id=rule.rule_id,
            passed=passed,
            severity=rule.severity,
            on_fail=rule.on_fail,
            message=msg,
        )

    # ── Convenience ───────────────────────────────────────────────────────────

    def build_correction_prompt(
        self, output: Any, validation_result: ValidationResult, skill: Skill
    ) -> str:
        """
        Build a correction prompt to send back to the LLM when hard rules fail.
        The LLM should revise its output to satisfy the failed rules.
        """
        failures = "\n".join(f"  - {f}" for f in validation_result.hard_failures)
        import yaml as _yaml
        output_text = _yaml.dump(output, allow_unicode=True, default_flow_style=False)
        return (
            "Your previous output failed the following validation rules:\n"
            f"{failures}\n\n"
            "Your previous output was:\n"
            f"```\n{output_text}```\n\n"
            "Please revise your output to satisfy all the rules above. "
            f"Follow the output schema for skill '{skill.skill_id}'."
        )
