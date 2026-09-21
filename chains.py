"""
LCEL chain assembly.

This is the only module that wires everything together, using pure
Runnable composition (`|`, `RunnablePassthrough.assign`, `RunnableLambda`) --
no legacy `Chain` classes (`RetrievalQA`, `ConversationalRetrievalChain`,
etc.) anywhere.

Pipeline shape
--------------
question
  -> retrieve docs (hybrid + reranked)
  -> format context
  -> detect crisis (rule-based guardrail, runs pre-generation)
  -> generate raw answer (prompt | llm | parser)
  -> verify faithfulness (judge LLM guardrail)
  -> apply faithfulness fallback + crisis disclaimer
  -> final answer + full trace (docs, crisis info, verification info)
"""
from __future__ import annotations

import logging
from typing import Any, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from src import config, guardrails
from src.retrieval import format_docs

logger = logging.getLogger(__name__)

GENERATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a careful health-information assistant. Answer the user's "
            "question using ONLY the information in the provided CONTEXT. "
            "If the context does not contain enough information to answer, say so "
            "plainly instead of guessing. Do not provide a diagnosis, prescription, "
            "or dosage instructions -- point the user to a qualified professional for "
            "those. Keep answers concise and cite which source(s) you drew from by "
            "their [Source N] label.\n\nCONTEXT:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


class RAGResult(TypedDict):
    question: str
    answer: str
    raw_answer: str
    docs: list[Document]
    crisis: guardrails.CrisisCheckResult
    verification: guardrails.VerificationResult


def get_chat_model(model_name: str, temperature: float = config.LLM_TEMPERATURE):
    """Instantiate the chat model for the configured provider.

    Kept as a small factory so swapping providers (OpenAI/Anthropic/local)
    is a one-line change and doesn't ripple through the chain definitions.
    """
    if config.LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model_name, temperature=temperature)
    if config.LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model_name, temperature=temperature)
    raise ValueError(f"Unsupported LLM_PROVIDER '{config.LLM_PROVIDER}'")


def build_rag_chain(
    retriever: BaseRetriever,
    llm: Any | None = None,
    judge_llm: Any | None = None,
) -> Runnable:
    """Assemble the full production LCEL pipeline.

    Returns a Runnable whose `.invoke({"question": "..."})` yields a
    `RAGResult` dict containing the final (guardrail-applied) answer plus
    the full intermediate trace, so the UI layer can display retrieved
    sources, crisis status, and verification scores without recomputing
    anything.
    """
    llm = llm or get_chat_model(config.LLM_MODEL_NAME)
    judge_llm = judge_llm or get_chat_model(config.JUDGE_LLM_MODEL_NAME, temperature=0.0)

    generation_chain = GENERATION_PROMPT | llm | StrOutputParser()
    verification_chain = guardrails.build_verification_chain(judge_llm)

    def _finalize(payload: dict[str, Any]) -> RAGResult:
        answer = guardrails.apply_faithfulness_guardrail(
            payload["raw_answer"], payload["verification"]
        )
        answer = guardrails.apply_crisis_disclaimer(answer, payload["crisis"])
        return RAGResult(
            question=payload["question"],
            answer=answer,
            raw_answer=payload["raw_answer"],
            docs=payload["docs"],
            crisis=payload["crisis"],
            verification=payload["verification"],
        )

    chain = (
        RunnablePassthrough.assign(docs=RunnableLambda(lambda x: retriever.invoke(x["question"])))
        .assign(context=RunnableLambda(lambda x: format_docs(x["docs"])))
        .assign(crisis=RunnableLambda(lambda x: guardrails.detect_crisis(x["question"], x["docs"])))
        .assign(raw_answer=generation_chain)
        .assign(
            verification=RunnableLambda(
                lambda x: verification_chain.invoke(
                    {"context": x["context"], "question": x["question"], "raw_answer": x["raw_answer"]}
                )
            )
        )
        | RunnableLambda(_finalize)
    )

    return chain


def run_query(chain: Runnable, question: str) -> RAGResult:
    """Thin, error-handled entrypoint for the UI layer.

    Any unexpected failure anywhere in the pipeline is caught here and
    turned into a safe, user-facing fallback result rather than an
    unhandled exception reaching Streamlit.
    """
    try:
        return chain.invoke({"question": question})
    except Exception:  # noqa: BLE001
        logger.exception("RAG chain failed for question: %r", question)
        return RAGResult(
            question=question,
            answer=(
                "Something went wrong while processing your question. "
                "Please try again in a moment."
            ),
            raw_answer="",
            docs=[],
            crisis=guardrails.detect_crisis(question, []),
            verification=guardrails.VerificationResult(
                faithfulness_score=0.0,
                unsupported_claims=[],
                reasoning="Pipeline error.",
                passed=False,
            ),
        )
