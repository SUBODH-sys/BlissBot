"""
Guardrails layer.

Two independent safety mechanisms, both implemented as plain functions /
LCEL-compatible Runnables so they compose naturally into the main chain:

1. Crisis / safety-sensitive guardrail (`detect_crisis`)
   Deterministic, rule-based, and runs BEFORE generation. Triggers on either
   (a) the `safety_sensitive` metadata flag on retrieved documents, or
   (b) lexical crisis-intent matches in the raw user query. Because it's
   rule-based it is fast, free, and does not depend on an LLM call
   succeeding -- crisis handling must never silently fail open.

2. Hallucination / faithfulness guardrail (`build_verification_chain`)
   A secondary LCEL step where a judge LLM scores whether the generated
   answer is actually supported by the retrieved context. If the score is
   below threshold, the raw answer is swapped for a safe fallback notice
   instead of being shown to the user.
"""
from __future__ import annotations

import logging
import re
from typing import Any, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda

from src import config

logger = logging.getLogger(__name__)

_CRISIS_PATTERN = re.compile(
    "|".join(re.escape(kw) for kw in config.CRISIS_KEYWORDS), flags=re.IGNORECASE
)


class CrisisCheckResult(TypedDict):
    is_crisis: bool
    triggered_by: str  # "query" | "metadata" | "none"
    matched_terms: list[str]


def _matched_crisis_terms(text: str) -> list[str]:
    return sorted({m.group(0).lower() for m in _CRISIS_PATTERN.finditer(text)})


def detect_crisis(query: str, docs: list[Document]) -> CrisisCheckResult:
    """Determine whether crisis/safety handling should be applied.

    Checked in this order so we can report *why* it triggered:
      1. Retrieved-document metadata flags (`safety_sensitive`) -- catches
         cases where the query itself is oblique but the best-matching
         reference material is about self-harm/crisis topics.
      2. Lexical match against the user's raw query -- catches explicit
         crisis language even if retrieval happens to return weak matches.
    """
    flagged_docs = [
        d for d in docs if config.SAFETY_SENSITIVE_FLAG in (d.metadata.get("flags") or [])
    ]
    if flagged_docs:
        return CrisisCheckResult(is_crisis=True, triggered_by="metadata", matched_terms=[])

    terms = _matched_crisis_terms(query)
    if terms:
        return CrisisCheckResult(is_crisis=True, triggered_by="query", matched_terms=terms)

    return CrisisCheckResult(is_crisis=False, triggered_by="none", matched_terms=[])


def apply_crisis_disclaimer(answer: str, crisis_result: CrisisCheckResult) -> str:
    """Prepend the standard crisis helpline disclaimer when triggered."""
    if not crisis_result["is_crisis"]:
        return answer
    return f"{config.CRISIS_DISCLAIMER}\n\n---\n\n{answer}"


# --------------------------------------------------------------------------- #
# Hallucination / faithfulness verification
# --------------------------------------------------------------------------- #

_VERIFICATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict factual auditor. You are given a CONTEXT (retrieved "
            "reference material) and an ANSWER that was generated from it. Your job "
            "is ONLY to judge whether every factual claim in the ANSWER is directly "
            "supported by the CONTEXT. Do not judge helpfulness, style, or whether "
            "the answer is a good answer to the question -- only factual groundedness.\n\n"
            "Respond with ONLY a compact JSON object, no markdown fences, no prose, "
            'matching exactly this schema: {{"faithfulness_score": <float 0.0-1.0>, '
            '"unsupported_claims": [<string>, ...], "reasoning": <short string>}}\n\n'
            "faithfulness_score of 1.0 means every claim is fully supported. "
            "0.0 means the answer is entirely fabricated or contradicts the context.",
        ),
        (
            "human",
            "CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nANSWER TO AUDIT:\n{raw_answer}",
        ),
    ]
)


class VerificationResult(TypedDict):
    faithfulness_score: float
    unsupported_claims: list[str]
    reasoning: str
    passed: bool


def build_verification_chain(judge_llm) -> Runnable:
    """LCEL chain: {context, question, raw_answer} -> VerificationResult.

    Kept as its own composable Runnable (prompt | llm | parser) so it can be
    unit-tested or reused outside the main generation chain, and so a
    parsing/LLM failure degrades safely rather than crashing the app.
    """

    def _parse_and_score(payload: dict[str, Any]) -> VerificationResult:
        score = float(payload.get("faithfulness_score", 0.0))
        score = max(0.0, min(1.0, score))
        return VerificationResult(
            faithfulness_score=score,
            unsupported_claims=list(payload.get("unsupported_claims", [])),
            reasoning=str(payload.get("reasoning", "")),
            passed=score >= config.FAITHFULNESS_FALLBACK_THRESHOLD,
        )

    chain = _VERIFICATION_PROMPT | judge_llm | JsonOutputParser() | RunnableLambda(_parse_and_score)

    def _safe_invoke(payload: dict[str, Any]) -> VerificationResult:
        try:
            return chain.invoke(payload)
        except Exception:  # noqa: BLE001 - guardrail must never crash the app
            logger.exception("Faithfulness verification failed; failing safe (blocking answer).")
            return VerificationResult(
                faithfulness_score=0.0,
                unsupported_claims=[],
                reasoning="Verifier error; blocked as a precaution.",
                passed=False,
            )

    return RunnableLambda(_safe_invoke)


def apply_faithfulness_guardrail(raw_answer: str, verification: VerificationResult) -> str:
    """Swap in the safe fallback notice when the answer fails verification."""
    if verification["passed"]:
        return raw_answer
    logger.warning(
        "Answer failed faithfulness guardrail (score=%.2f): %s",
        verification["faithfulness_score"],
        verification["reasoning"],
    )
    return config.FALLBACK_NOTICE
