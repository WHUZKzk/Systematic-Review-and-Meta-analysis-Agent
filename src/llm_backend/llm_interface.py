"""
Unified LLM call interface via OpenRouter (OpenAI-compatible API).

Features:
  - Structured JSON output with automatic parsing
  - Skill-scoped prompts: system = L1 manifests + L2 active protocol
  - Auto-retry on validation failure (up to LLM_MAX_RETRIES)
  - Token counting and call logging
  - Graceful error handling with typed exceptions
"""

from __future__ import annotations
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import openai

logger = logging.getLogger(__name__)


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class Message:
    role: str   # system | user | assistant
    content: str


@dataclass
class LLMResponse:
    content: str                        # Raw text content
    parsed: Any = None                  # Parsed JSON/dict if format == "json"
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    finish_reason: str = ""


@dataclass
class PromptContext:
    """All inputs needed to build a single LLM call."""
    # Skill-derived system prompt sections
    l1_manifests: List[str] = field(default_factory=list)   # Always-on skill summaries
    l2_protocol: str = ""                                    # Active step protocol
    l3_references: List[str] = field(default_factory=list)  # On-demand domain refs

    # Task content
    task_instruction: str = ""
    task_data: str = ""                  # Serialised input data for this call

    # History (for multi-turn correction loops)
    history: List[Message] = field(default_factory=list)

    # Output format hint
    output_format: str = "json"          # json | text


# ── Exceptions ────────────────────────────────────────────────────────────────

class LLMError(Exception):
    """Raised when an LLM call fails after all retries."""


class LLMParseError(LLMError):
    """Raised when the LLM response cannot be parsed as expected JSON."""


# ── Interface ─────────────────────────────────────────────────────────────────

class LLMInterface:
    """
    Wraps an OpenRouter-backed OpenAI client.

    Args:
        model:    OpenRouter model ID, e.g. "qwen/qwen3.5-plus-02-15"
        api_key:  OpenRouter API key
        base_url: OpenRouter base URL (default: https://openrouter.ai/api/v1)
        temperature, max_tokens, max_retries: call parameters
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        max_retries: int = 2,
    ):
        from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

        self._client = openai.OpenAI(
            api_key=api_key or OPENROUTER_API_KEY,
            base_url=base_url or OPENROUTER_BASE_URL,
        )

    # ── Core call ─────────────────────────────────────────────────────────────

    def call(self, context: PromptContext) -> LLMResponse:
        """
        Execute an LLM call from a PromptContext.

        Builds:
            system prompt = L1 manifests + L2 protocol + L3 references
            user   prompt = task instruction + task data
        """
        messages = self._build_messages(context)
        return self._execute(messages, output_format=context.output_format)

    def call_raw(
        self,
        system: str,
        user: str,
        history: Optional[List[Message]] = None,
        output_format: str = "text",
    ) -> LLMResponse:
        """
        Convenience method for direct system+user prompt calls
        (bypasses PromptContext building).
        """
        messages: List[Dict] = [{"role": "system", "content": system}]
        for msg in (history or []):
            messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": user})
        return self._execute(messages, output_format=output_format)

    def call_with_correction(
        self,
        context: PromptContext,
        correction_prompt: str,
    ) -> LLMResponse:
        """
        Follow-up call that appends the assistant's previous response plus
        a correction prompt, then asks the LLM to revise.
        """
        # Build the correction as a user turn appended after history
        correction_context = PromptContext(
            l1_manifests=context.l1_manifests,
            l2_protocol=context.l2_protocol,
            l3_references=context.l3_references,
            task_instruction=context.task_instruction,
            task_data=context.task_data,
            history=context.history,
            output_format=context.output_format,
        )
        messages = self._build_messages(correction_context)
        messages.append({"role": "user", "content": correction_prompt})
        return self._execute(messages, output_format=context.output_format)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_messages(self, context: PromptContext) -> List[Dict]:
        system_parts: List[str] = []

        if context.l1_manifests:
            system_parts.append(
                "# Active Skills (Summary)\n" + "\n\n".join(context.l1_manifests)
            )
        if context.l2_protocol:
            system_parts.append(
                "# Active Protocol\n" + context.l2_protocol
            )
        if context.l3_references:
            system_parts.append(
                "# Domain References\n" + "\n\n".join(context.l3_references)
            )
        if context.output_format == "json":
            system_parts.append(
                "# Output Instruction\n"
                "Respond with valid JSON only. "
                "Do not include markdown code fences or any text outside the JSON object."
            )

        system_prompt = "\n\n".join(system_parts) if system_parts else "You are a helpful assistant."

        messages: List[Dict] = [{"role": "system", "content": system_prompt}]

        for msg in context.history:
            messages.append({"role": msg.role, "content": msg.content})

        user_content = ""
        if context.task_instruction:
            user_content += context.task_instruction + "\n\n"
        if context.task_data:
            user_content += context.task_data

        messages.append({"role": "user", "content": user_content.strip()})
        return messages

    def _execute(self, messages: List[Dict], output_format: str = "text") -> LLMResponse:
        """Send the message list to the API, parse response, handle errors."""
        kwargs: Dict[str, Any] = {
            "model":       self.model,
            "messages":    messages,
            "temperature": self.temperature,
            "max_tokens":  self.max_tokens,
        }
        # Request JSON output if needed
        if output_format == "json":
            kwargs["response_format"] = {"type": "json_object"}

        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 2):  # +2: initial + retries
            try:
                t0 = time.monotonic()
                response = self._client.chat.completions.create(**kwargs)
                latency = (time.monotonic() - t0) * 1000

                choice = response.choices[0]
                raw_content = choice.message.content or ""
                finish_reason = choice.finish_reason or ""

                usage = response.usage
                prompt_tokens     = usage.prompt_tokens     if usage else 0
                completion_tokens = usage.completion_tokens if usage else 0
                total_tokens      = usage.total_tokens      if usage else 0

                parsed = None
                if output_format == "json":
                    parsed = self._parse_json(raw_content)

                logger.debug(
                    "LLM call [%s] attempt=%d tokens=%d latency=%.0fms finish=%s",
                    self.model, attempt, total_tokens, latency, finish_reason,
                )

                return LLMResponse(
                    content=raw_content,
                    parsed=parsed,
                    model=self.model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    latency_ms=latency,
                    finish_reason=finish_reason,
                )

            except openai.RateLimitError as exc:
                wait = 2 ** attempt
                logger.warning("Rate limit hit (attempt %d), waiting %ds: %s", attempt, wait, exc)
                time.sleep(wait)
                last_exc = exc
            except openai.APIError as exc:
                logger.error("API error on attempt %d: %s", attempt, exc)
                last_exc = exc
                if attempt <= self.max_retries:
                    time.sleep(1)
                else:
                    break
            except LLMParseError:
                raise
            except Exception as exc:
                logger.error("Unexpected error on attempt %d: %s", attempt, exc)
                last_exc = exc
                break

        raise LLMError(f"LLM call failed after {self.max_retries + 1} attempts: {last_exc}")

    @staticmethod
    def _parse_json(text: str) -> Any:
        """Parse JSON from LLM response, stripping accidental markdown fences."""
        stripped = text.strip()
        # Remove ```json ... ``` or ``` ... ``` wrappers if present
        if stripped.startswith("```"):
            lines = stripped.split("\n")
            # Drop first line (``` or ```json) and last line (```)
            inner_lines = lines[1:]
            if inner_lines and inner_lines[-1].strip() == "```":
                inner_lines = inner_lines[:-1]
            stripped = "\n".join(inner_lines).strip()
        try:
            return json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise LLMParseError(f"Could not parse LLM output as JSON: {exc}\nRaw: {text[:300]}")
