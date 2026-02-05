"""RAG Evaluation Suite using RAGAS methodology.

TIER 10 - Point 56: RAG Evaluation Suite (Ragas)

Automated evaluation of RAG pipeline quality using:
- Faithfulness: Is the answer grounded in context?
- Answer Relevancy: Does the answer address the question?
- Context Precision: Is retrieved context relevant?
- Context Recall: Did we retrieve all relevant context?

Runs on golden test set to measure quality over time.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RAGExample:
    """A single RAG evaluation example."""
    query: str
    answer: str
    contexts: list[str]
    ground_truth: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Result of evaluating a single example."""
    example_id: str
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    overall_score: float = 0.0
    details: dict = field(default_factory=dict)


@dataclass
class EvaluationReport:
    """Complete evaluation report for a RAG pipeline."""
    run_id: str
    timestamp: datetime
    n_examples: int
    
    # Aggregate metrics
    avg_faithfulness: float
    avg_answer_relevancy: float
    avg_context_precision: float
    avg_context_recall: float
    overall_score: float
    
    # Per-example results
    results: list[EvaluationResult] = field(default_factory=list)
    
    # Metadata
    model_info: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


class RAGEvaluator:
    """RAG Evaluation Suite.
    
    TIER 10 - Point 56: Ragas-style evaluation.
    
    Usage:
        evaluator = RAGEvaluator(llm_client)
        
        examples = [
            RAGExample(
                query="Best romantic restaurant in Milan?",
                answer="Giacomo is perfect for romantic dinners...",
                contexts=["Giacomo offers intimate setting...", ...],
                ground_truth="Giacomo is highly rated for romance",
            ),
        ]
        
        report = await evaluator.evaluate(examples)
        print(f"Overall Score: {report.overall_score:.2%}")
    """
    
    def __init__(self, llm_client=None):
        self.llm = llm_client
        self._evaluation_prompts = self._get_evaluation_prompts()
    
    def _get_evaluation_prompts(self) -> dict[str, str]:
        """Get prompts for LLM-based evaluation."""
        return {
            "faithfulness": """Given the following context and answer, evaluate if the answer is faithful to the context.
            
Context:
{context}

Answer:
{answer}

Is every claim in the answer supported by the context? Rate from 0 to 1.
Only respond with a number between 0 and 1.""",
            
            "answer_relevancy": """Given the question and answer, rate how relevant the answer is to the question.

Question: {query}
Answer: {answer}

Rate from 0 to 1 where 1 means the answer directly addresses the question.
Only respond with a number between 0 and 1.""",
            
            "context_precision": """Given the question and retrieved contexts, rate how relevant each context is.

Question: {query}

Contexts:
{contexts}

What percentage of the contexts are relevant to answering the question?
Only respond with a number between 0 and 1.""",
        }
    
    async def evaluate(
        self,
        examples: list[RAGExample],
        run_id: str | None = None,
    ) -> EvaluationReport:
        """Evaluate a list of RAG examples.
        
        Args:
            examples: List of RAG examples to evaluate
            run_id: Optional run identifier
            
        Returns:
            Complete evaluation report
        """
        from uuid import uuid4
        
        run_id = run_id or str(uuid4())[:8]
        results: list[EvaluationResult] = []
        errors: list[str] = []
        
        for i, example in enumerate(examples):
            try:
                result = await self._evaluate_single(example, f"{run_id}_{i}")
                results.append(result)
            except Exception as e:
                errors.append(f"Example {i}: {str(e)}")
                logger.warning(f"Evaluation failed for example {i}: {e}")
        
        # Calculate aggregates
        n = len(results) or 1
        
        avg_faithfulness = sum(r.faithfulness for r in results) / n
        avg_relevancy = sum(r.answer_relevancy for r in results) / n
        avg_precision = sum(r.context_precision for r in results) / n
        avg_recall = sum(r.context_recall for r in results) / n
        
        # Overall score (weighted average)
        overall = (
            0.35 * avg_faithfulness +
            0.25 * avg_relevancy +
            0.20 * avg_precision +
            0.20 * avg_recall
        )
        
        return EvaluationReport(
            run_id=run_id,
            timestamp=datetime.utcnow(),
            n_examples=len(examples),
            avg_faithfulness=round(avg_faithfulness, 3),
            avg_answer_relevancy=round(avg_relevancy, 3),
            avg_context_precision=round(avg_precision, 3),
            avg_context_recall=round(avg_recall, 3),
            overall_score=round(overall, 3),
            results=results,
            errors=errors,
        )
    
    async def _evaluate_single(
        self,
        example: RAGExample,
        example_id: str,
    ) -> EvaluationResult:
        """Evaluate a single RAG example."""
        result = EvaluationResult(example_id=example_id)
        
        # Faithfulness
        result.faithfulness = await self._score_faithfulness(
            example.answer, example.contexts
        )
        
        # Answer Relevancy
        result.answer_relevancy = await self._score_relevancy(
            example.query, example.answer
        )
        
        # Context Precision
        result.context_precision = await self._score_context_precision(
            example.query, example.contexts
        )
        
        # Context Recall (requires ground truth)
        if example.ground_truth:
            result.context_recall = await self._score_context_recall(
                example.ground_truth, example.contexts
            )
        else:
            result.context_recall = 0.5  # Default when no ground truth
        
        # Overall score
        result.overall_score = (
            0.35 * result.faithfulness +
            0.25 * result.answer_relevancy +
            0.20 * result.context_precision +
            0.20 * result.context_recall
        )
        
        return result
    
    async def _score_faithfulness(
        self,
        answer: str,
        contexts: list[str],
    ) -> float:
        """Score how faithful the answer is to the context."""
        if not contexts:
            return 0.0
        
        if self.llm:
            # Use LLM for evaluation
            try:
                context_text = "\n---\n".join(contexts)
                prompt = self._evaluation_prompts["faithfulness"].format(
                    context=context_text,
                    answer=answer,
                )
                response = await self.llm.complete(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                return self._parse_score(response)
            except Exception as e:
                logger.warning(f"LLM faithfulness scoring failed: {e}")
        
        # Fallback: lexical overlap
        return self._lexical_overlap(answer, " ".join(contexts))
    
    async def _score_relevancy(self, query: str, answer: str) -> float:
        """Score how relevant the answer is to the query."""
        if self.llm:
            try:
                prompt = self._evaluation_prompts["answer_relevancy"].format(
                    query=query,
                    answer=answer,
                )
                response = await self.llm.complete(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                return self._parse_score(response)
            except Exception as e:
                logger.warning(f"LLM relevancy scoring failed: {e}")
        
        # Fallback: keyword overlap with query
        return self._lexical_overlap(answer, query)
    
    async def _score_context_precision(
        self,
        query: str,
        contexts: list[str],
    ) -> float:
        """Score context precision (relevance of retrieved contexts)."""
        if not contexts:
            return 0.0
        
        if self.llm:
            try:
                contexts_text = "\n---\n".join(
                    f"{i+1}. {c}" for i, c in enumerate(contexts)
                )
                prompt = self._evaluation_prompts["context_precision"].format(
                    query=query,
                    contexts=contexts_text,
                )
                response = await self.llm.complete(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                return self._parse_score(response)
            except Exception as e:
                logger.warning(f"LLM context precision scoring failed: {e}")
        
        # Fallback: average overlap with query
        overlaps = [self._lexical_overlap(c, query) for c in contexts]
        return sum(overlaps) / len(overlaps)
    
    async def _score_context_recall(
        self,
        ground_truth: str,
        contexts: list[str],
    ) -> float:
        """Score context recall (coverage of ground truth)."""
        if not contexts:
            return 0.0
        
        # Check how much of ground truth is covered by contexts
        combined_context = " ".join(contexts).lower()
        gt_words = set(ground_truth.lower().split())
        
        covered = sum(1 for w in gt_words if w in combined_context)
        
        return covered / len(gt_words) if gt_words else 0.0
    
    def _lexical_overlap(self, text1: str, text2: str) -> float:
        """Calculate lexical overlap between two texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)  # Jaccard similarity
    
    def _parse_score(self, response: str) -> float:
        """Parse numeric score from LLM response."""
        import re
        
        # Extract first number from response
        match = re.search(r"(\d+\.?\d*)", response.strip())
        if match:
            score = float(match.group(1))
            # Normalize to 0-1 if needed
            if score > 1:
                score = score / 100 if score <= 100 else 1.0
            return min(1.0, max(0.0, score))
        
        return 0.5  # Default if parsing fails


# Golden test set for regular evaluation
GOLDEN_TEST_SET = [
    RAGExample(
        query="Best romantic restaurant in Milan?",
        answer="For a romantic dinner in Milan, I recommend Giacomo. It offers an intimate atmosphere with candlelit tables and exceptional Italian cuisine.",
        contexts=[
            "Giacomo Milano is known for its romantic ambiance with intimate seating and soft lighting.",
            "The restaurant serves traditional Milanese dishes with a modern twist.",
            "Reservations are recommended, especially for weekend dinners.",
        ],
        ground_truth="Giacomo is a top choice for romantic dining in Milan with intimate atmosphere.",
    ),
    RAGExample(
        query="Trendy cocktail bar for young professionals?",
        answer="Check out Nottingham Forest in Navigli. It's a speakeasy-style bar known for creative cocktails and a sophisticated crowd.",
        contexts=[
            "Nottingham Forest is a cocktail bar in the Navigli district.",
            "Known for innovative mixology and a chic atmosphere.",
            "Popular with young professionals and creatives.",
        ],
        ground_truth="Nottingham Forest is a trendy cocktail spot popular with professionals.",
    ),
]


async def run_golden_evaluation(llm_client=None) -> EvaluationReport:
    """Run evaluation on golden test set.
    
    TIER 10 - Point 56: Scheduled golden set evaluation.
    
    Call this:
    - After model/prompt changes
    - Weekly as part of quality monitoring
    """
    evaluator = RAGEvaluator(llm_client)
    report = await evaluator.evaluate(GOLDEN_TEST_SET, run_id="golden")
    
    logger.info(
        f"Golden evaluation complete: {report.overall_score:.1%} overall "
        f"(F={report.avg_faithfulness:.1%}, R={report.avg_answer_relevancy:.1%})"
    )
    
    return report
