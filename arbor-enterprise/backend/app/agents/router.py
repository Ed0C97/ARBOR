"""Intent Router agent - classifies user queries."""

import json
import logging

from app.llm.gateway import get_llm_gateway
from app.llm.prompts import INTENT_CLASSIFIER, load_prompt

logger = logging.getLogger(__name__)


class IntentRouter:
    """Classify user intent and extract filters."""

    def __init__(self):
        self.gateway = get_llm_gateway()
        try:
            self.system_prompt = load_prompt(INTENT_CLASSIFIER)
        except FileNotFoundError:
            self.system_prompt = self._default_prompt()

    async def classify(self, query: str) -> dict:
        """Classify intent from user query. Returns structured classification."""
        try:
            response = await self.gateway.complete_json(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": query},
                ],
                task_type="classification",
            )
            result = json.loads(response)
            return self._validate(result)
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Intent classification failed: {e}")
            return {
                "intent": "DISCOVERY",
                "confidence": 0.5,
                "entities_mentioned": [],
                "filters": {},
                "style_keywords": [],
            }

    def _validate(self, result: dict) -> dict:
        valid_intents = {"DISCOVERY", "COMPARISON", "DETAIL", "HISTORY", "NAVIGATION", "GENERAL"}
        intent = result.get("intent", "DISCOVERY").upper()
        if intent not in valid_intents:
            intent = "DISCOVERY"

        return {
            "intent": intent,
            "confidence": float(result.get("confidence", 0.8)),
            "entities_mentioned": result.get("entities_mentioned", []),
            "filters": result.get("filters", {}),
            "style_keywords": result.get("style_keywords", []),
        }

    def _default_prompt(self) -> str:
        return (
            "Classify intent: DISCOVERY, COMPARISON, DETAIL, HISTORY, NAVIGATION, GENERAL. "
            "Return JSON with: intent, confidence, entities_mentioned, filters, style_keywords"
        )
