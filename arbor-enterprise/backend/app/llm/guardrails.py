"""Content moderation and guardrails for LLM inputs/outputs.

TIER 3 - Point 14: Blocking Guardrails

Implements strict checking that BLOCKS potentially harmful outputs
instead of just logging them.

Features:
- Input validation: Block injection attacks, off-topic requests
- Output validation: Block hallucinations, unsafe content
- Configurable thresholds for hallucination detection
"""

import logging
import re
from dataclasses import dataclass
from typing import Any

from app.llm.gateway import get_llm_gateway

logger = logging.getLogger(__name__)


@dataclass
class GuardrailResult:
    """Result of a guardrail check."""

    allowed: bool
    reason: str
    risk_score: float = 0.0  # 0.0 = safe, 1.0 = maximum risk
    blocked_content: str | None = None
    replacement: str | None = None  # Safe replacement if blocked


class BlockedContentError(Exception):
    """Raised when content is blocked by guardrails."""

    def __init__(self, reason: str, risk_score: float = 1.0):
        self.reason = reason
        self.risk_score = risk_score
        super().__init__(f"Content blocked: {reason}")


# Safe fallback responses
SAFE_FALLBACK_RESPONSES = {
    "hallucination": (
        "I apologize, but I cannot verify all the details in my response. "
        "Please double-check any specific recommendations with the venue directly."
    ),
    "off_topic": (
        "I'm designed to help you discover great places and experiences. "
        "Could you tell me what kind of venue or experience you're looking for?"
    ),
    "unsafe": (
        "I'm not able to help with that request. "
        "I can help you discover restaurants, shops, and experiences instead."
    ),
    "injection": (
        "I noticed something unusual in your request. "
        "Could you please rephrase what you're looking for?"
    ),
}


class Guardrails:
    """Input/output guardrails for LLM safety.

    TIER 3 - Point 14: Blocking Guardrails

    Unlike permissive guardrails that only log, this implementation
    BLOCKS potentially harmful content and returns safe alternatives.
    """

    # Patterns to block in input (injection attacks)
    BLOCKED_PATTERNS = [
        r"ignore\s+(previous|above|all)\s+instructions",
        r"you\s+are\s+now\s+",
        r"forget\s+everything",
        r"system\s*:\s*",
        r"<\s*script",
        r"pretend\s+to\s+be",
        r"act\s+as\s+(if|though)",
        r"bypass\s+(security|filter|guardrail)",
        r"jailbreak",
        r"\[INST\]",  # Llama-style injection
        r"###\s*Human:",  # Anthropic-style injection
    ]

    # Topics that should be rejected
    OFF_TOPIC_KEYWORDS = [
        "hack", "exploit", "password", "credit card",
        "social security", "illegal", "weapon", "drug",
        "bomb", "terrorist", "murder", "suicide",
    ]

    # Hallucination detection thresholds
    HALLUCINATION_THRESHOLD = 0.8  # Block if score > 0.8

    def __init__(self, strict_mode: bool = True):
        """Initialize guardrails.

        Args:
            strict_mode: If True, blocks content instead of just logging
        """
        self._llm = None  # Lazy init to avoid circular import
        self.strict_mode = strict_mode
        self._compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.BLOCKED_PATTERNS
        ]

        # Stats for monitoring
        self._stats = {
            "input_checks": 0,
            "input_blocks": 0,
            "output_checks": 0,
            "output_blocks": 0,
            "hallucinations_detected": 0,
        }

    @property
    def llm(self):
        """Lazy-load LLM gateway."""
        if self._llm is None:
            self._llm = get_llm_gateway()
        return self._llm

    async def check_input(self, user_input: str) -> GuardrailResult:
        """Check if user input is safe and on-topic.

        TIER 3 - Point 14: Returns structured result with risk score.
        """
        self._stats["input_checks"] += 1

        # Pattern-based check (fast, catches injection attacks)
        for pattern in self._compiled_patterns:
            match = pattern.search(user_input)
            if match:
                self._stats["input_blocks"] += 1
                logger.warning(f"Injection attack blocked: {match.group()[:50]}")
                return GuardrailResult(
                    allowed=False,
                    reason="Input contains blocked pattern (possible injection)",
                    risk_score=1.0,
                    blocked_content=match.group(),
                    replacement=SAFE_FALLBACK_RESPONSES["injection"],
                )

        # Keyword check
        lower_input = user_input.lower()
        for keyword in self.OFF_TOPIC_KEYWORDS:
            if keyword in lower_input:
                self._stats["input_blocks"] += 1
                logger.warning(f"Off-topic keyword blocked: {keyword}")
                return GuardrailResult(
                    allowed=False,
                    reason=f"Input contains off-topic keyword: {keyword}",
                    risk_score=0.9,
                    blocked_content=keyword,
                    replacement=SAFE_FALLBACK_RESPONSES["off_topic"],
                )

        # LLM-based check for nuanced cases (for longer inputs)
        if len(user_input) > 200:
            try:
                result = await self.llm.complete(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Check if this input is appropriate for a lifestyle/shopping "
                                "discovery assistant. Answer with exactly: allowed or not_allowed"
                            ),
                        },
                        {"role": "user", "content": user_input},
                    ],
                    task_type="classification",
                    temperature=0,
                )
                if "not_allowed" in result.lower():
                    self._stats["input_blocks"] += 1
                    return GuardrailResult(
                        allowed=False,
                        reason="Input flagged by content moderation",
                        risk_score=0.7,
                        replacement=SAFE_FALLBACK_RESPONSES["unsafe"],
                    )
            except Exception as e:
                logger.warning(f"Guardrails LLM check failed: {e}")
                # Fail open for input checks

        return GuardrailResult(allowed=True, reason="ok", risk_score=0.0)

    async def check_output(
        self,
        response: str,
        context_entities: list[str],
        original_query: str = "",
    ) -> GuardrailResult:
        """Check if LLM output is factual and safe.

        TIER 3 - Point 14: BLOCKS hallucinations instead of just logging.

        Args:
            response: The LLM-generated response
            context_entities: List of known entity names from context
            original_query: The original user query (for relevance check)

        Returns:
            GuardrailResult with allowed status and any replacement
        """
        self._stats["output_checks"] += 1

        # Calculate hallucination risk score
        hallucination_score = await self._calculate_hallucination_score(
            response, context_entities
        )

        if hallucination_score > self.HALLUCINATION_THRESHOLD:
            self._stats["output_blocks"] += 1
            self._stats["hallucinations_detected"] += 1
            logger.warning(
                f"Hallucination blocked (score={hallucination_score:.2f}): {response[:100]}"
            )

            if self.strict_mode:
                return GuardrailResult(
                    allowed=False,
                    reason="Response may contain unverified information",
                    risk_score=hallucination_score,
                    replacement=SAFE_FALLBACK_RESPONSES["hallucination"],
                )

        # Check for unsafe content in output
        for keyword in self.OFF_TOPIC_KEYWORDS:
            if keyword in response.lower():
                self._stats["output_blocks"] += 1
                logger.warning(f"Unsafe content in output: {keyword}")
                return GuardrailResult(
                    allowed=False,
                    reason="Response contains inappropriate content",
                    risk_score=0.9,
                    replacement=SAFE_FALLBACK_RESPONSES["unsafe"],
                )

        return GuardrailResult(
            allowed=True,
            reason="ok",
            risk_score=hallucination_score,
        )

    async def _calculate_hallucination_score(
        self,
        response: str,
        context_entities: list[str],
    ) -> float:
        """Calculate hallucination risk score.

        Simple heuristic: Count potential entity names in response that
        are not in our known context.

        Returns:
            Score from 0.0 (safe) to 1.0 (likely hallucination)
        """
        if not context_entities:
            return 0.0

        # Extract potential entity names (capitalized words)
        potential_entities = set()
        for word in response.split():
            # Remove punctuation
            cleaned = re.sub(r"[^\w]", "", word)
            # Check if it looks like a proper noun
            if len(cleaned) > 3 and cleaned[0].isupper() and cleaned[1:].islower():
                potential_entities.add(cleaned)

        if not potential_entities:
            return 0.0

        # Count unknown entities
        known_entities = set(e.split()[0] for e in context_entities if e)
        unknown = potential_entities - known_entities

        # Score based on ratio of unknown entities
        unknown_ratio = len(unknown) / len(potential_entities)

        # Adjust score based on number of unknowns
        if len(unknown) > 5:
            # Many unknown entities = higher risk
            return min(1.0, unknown_ratio + 0.2)

        return unknown_ratio * 0.7  # Scale down for few unknowns

    def sanitize_input(self, text: str) -> str:
        """Remove potentially dangerous content from input."""
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", "", text)
        # Remove potential injections
        for pattern in self._compiled_patterns:
            text = pattern.sub("[REMOVED]", text)
        # Limit length
        return text[:2000]

    async def validate_and_block(
        self,
        response: str,
        context_entities: list[str],
    ) -> str:
        """Validate output and return safe version.

        TIER 3 - Point 14: The blocking implementation.

        Args:
            response: LLM-generated response
            context_entities: Known entity names

        Returns:
            Original response if safe, or safe replacement if blocked
        """
        result = await self.check_output(response, context_entities)

        if not result.allowed and result.replacement:
            logger.warning(f"Output blocked and replaced: {result.reason}")
            return result.replacement

        return response

    def get_stats(self) -> dict[str, Any]:
        """Get guardrails statistics for monitoring."""
        total_inputs = self._stats["input_checks"] or 1
        total_outputs = self._stats["output_checks"] or 1

        return {
            **self._stats,
            "input_block_rate": self._stats["input_blocks"] / total_inputs,
            "output_block_rate": self._stats["output_blocks"] / total_outputs,
            "strict_mode": self.strict_mode,
        }


# Singleton
_guardrails: Guardrails | None = None


def get_guardrails(strict_mode: bool = True) -> Guardrails:
    """Get singleton Guardrails instance."""
    global _guardrails
    if _guardrails is None:
        _guardrails = Guardrails(strict_mode=strict_mode)
    return _guardrails
