"""
Real-time RAGAS evaluation.

Computes genuine RAGAS metrics (not mocked/heuristic scores) for every
query turn so the UI can show actual quality signals alongside the answer:

- faithfulness        : is the answer grounded in the retrieved context?
- answer_relevancy     : does the answer actually address the question?
- context_precision    : are the retrieved chunks relevant / well-ranked?

RAGAS metrics need an LLM (as judge) and an embedding model; we reuse the
app's existing judge LLM and the bge embeddings so no extra model downloads
are required. Because these are LLM-judged metrics, evaluation is wrapped in
try/except and degrades to `None` scores (surfaced clearly in the UI) rather
than breaking the main answer flow if the judge call fails.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, TypedDict

from langchain_core.documents import Document

from src import config

logger = logging.getLogger(__name__)


class EvalScores(TypedDict):
    faithfulness: float | None
    answer_relevancy: float | None
    context_precision: float | None
    error: str | None


def _empty_scores(error: str | None = None) -> EvalScores:
    return EvalScores(faithfulness=None, answer_relevancy=None, context_precision=None, error=error)


def evaluate_turn(
    question: str,
    answer: str,
    docs: list[Document],
    judge_llm: Any,
    embeddings: Any,
    ground_truth: str | None = None,
) -> EvalScores:
    """Score a single Q/A/context turn with real RAGAS metrics.

    Uses the RAGAS 0.2+ single-turn sample API so a single query can be
    scored synchronously without building a HuggingFace `Dataset` for a
    batch of one row. Falls back to reporting the error in `EvalScores`
    rather than raising, since evaluation must never block the answer
    already shown to the user.
    """
    if not config.RAGAS_ENABLED:
        return _empty_scores(error="RAGAS evaluation disabled via config.")
    if not docs:
        return _empty_scores(error="No retrieved context to evaluate against.")

    try:
        from ragas.dataset_schema import SingleTurnSample
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import AnswerRelevancy, ContextPrecision, Faithfulness
    except ImportError as exc:
        logger.warning("ragas not installed or incompatible version: %s", exc)
        return _empty_scores(error="RAGAS is not installed in this environment.")

    try:
        ragas_llm = LangchainLLMWrapper(judge_llm)
        ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

        contexts = [d.metadata.get("answer", d.page_content) for d in docs]

        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=ground_truth or answer,
        )

        faithfulness = Faithfulness(llm=ragas_llm)
        answer_relevancy = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
        context_precision = ContextPrecision(llm=ragas_llm)

        async def _score_all() -> tuple[float, float, float]:
            f_score, r_score, p_score = await asyncio.gather(
                faithfulness.single_turn_ascore(sample),
                answer_relevancy.single_turn_ascore(sample),
                context_precision.single_turn_ascore(sample),
                return_exceptions=False,
            )
            return f_score, r_score, p_score

        f_score, r_score, p_score = asyncio.run(_score_all())

        return EvalScores(
            faithfulness=round(float(f_score), 3),
            answer_relevancy=round(float(r_score), 3),
            context_precision=round(float(p_score), 3),
            error=None,
        )
    except Exception as exc:  # noqa: BLE001 - evaluation must fail soft
        logger.exception("RAGAS evaluation failed")
        return _empty_scores(error=f"Evaluation failed: {exc}")
