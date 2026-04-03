"""
LLM Backend Registry — manages named LLMInterface instances.

Provides:
  - get(name)  → returns the LLMInterface for a named role
  - register() → register a custom backend under a name
  - Heterogeneity policy: dual-review pair must use different model families
"""

from __future__ import annotations
import logging
from typing import Dict, Optional

from .llm_interface import LLMInterface

logger = logging.getLogger(__name__)


class BackendRegistry:
    """
    Central registry for LLM backends.

    Pre-configured roles (from config.MODELS):
        default     — general executor / adjudicator  (qwen)
        reviewer_a  — dual-review agent A            (deepseek)
        reviewer_b  — dual-review agent B            (gemini)
        adjudicator — adjudicator agent              (qwen)

    Backends are created lazily on first access.
    """

    def __init__(self):
        self._instances: Dict[str, LLMInterface] = {}

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, name: str, interface: LLMInterface) -> None:
        """Register a pre-built LLMInterface under `name`."""
        self._instances[name] = interface
        logger.debug("Registered LLM backend: %s → %s", name, interface.model)

    # ── Access ────────────────────────────────────────────────────────────────

    def get(self, name: str = "default") -> LLMInterface:
        """
        Return the LLMInterface registered under `name`.
        Creates it lazily from config.MODELS on first access.
        """
        if name not in self._instances:
            self._instances[name] = self._build_from_config(name)
        return self._instances[name]

    def get_dual_review_pair(self) -> tuple[LLMInterface, LLMInterface]:
        """
        Return (reviewer_a, reviewer_b) — the heterogeneous dual-review pair.
        Verifies that the two models are from different families.
        """
        a = self.get("reviewer_a")
        b = self.get("reviewer_b")
        self._assert_heterogeneous(a.model, b.model)
        return a, b

    def get_adjudicator(self) -> LLMInterface:
        return self.get("adjudicator")

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _build_from_config(name: str) -> LLMInterface:
        from config import MODELS, OPENROUTER_API_KEY, OPENROUTER_BASE_URL, LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_MAX_RETRIES
        if name not in MODELS:
            raise KeyError(
                f"No model configured for backend '{name}'. "
                f"Available: {list(MODELS.keys())}"
            )
        model = MODELS[name]
        logger.debug("Creating LLMInterface for '%s' → model '%s'", name, model)
        return LLMInterface(
            model=model,
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            max_retries=LLM_MAX_RETRIES,
        )

    # Model family prefixes used for heterogeneity check
    _FAMILY_MAP = {
        "deepseek": "deepseek",
        "google":   "google",
        "qwen":     "qwen",
        "openai":   "openai",
        "anthropic":"anthropic",
        "meta":     "meta",
        "mistral":  "mistral",
    }

    @classmethod
    def _get_family(cls, model_id: str) -> str:
        model_lower = model_id.lower()
        for prefix, family in cls._FAMILY_MAP.items():
            if model_lower.startswith(prefix):
                return family
        # Fall back to the first path component (e.g. "provider/model-name")
        return model_id.split("/")[0].lower()

    @classmethod
    def _assert_heterogeneous(cls, model_a: str, model_b: str) -> None:
        family_a = cls._get_family(model_a)
        family_b = cls._get_family(model_b)
        if family_a == family_b:
            raise ValueError(
                f"Dual-review heterogeneity violated: both reviewers are from "
                f"the '{family_a}' family ({model_a}, {model_b}). "
                "Configure reviewer_a and reviewer_b to use different model families."
            )
        logger.debug(
            "Heterogeneity check passed: reviewer_a=%s (%s) vs reviewer_b=%s (%s)",
            model_a, family_a, model_b, family_b,
        )


# ── Module-level singleton ────────────────────────────────────────────────────

_registry: Optional[BackendRegistry] = None


def get_registry() -> BackendRegistry:
    """Return the module-level singleton BackendRegistry."""
    global _registry
    if _registry is None:
        _registry = BackendRegistry()
    return _registry
