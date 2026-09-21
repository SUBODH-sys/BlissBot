"""
Central configuration for the RAG application.

All tunable constants live here so the rest of the codebase never hardcodes
model names, weights, or paths. Values can be overridden via environment
variables for different deployment environments (dev / staging / prod).
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = Path(os.getenv("RAG_DATA_PATH", str(BASE_DIR / "data" / "qa_dataset.json")))
FAISS_INDEX_DIR = Path(os.getenv("RAG_FAISS_DIR", str(BASE_DIR / "storage" / "faiss_index")))

# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
EMBEDDING_MODEL_NAME = os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
# bge models are trained to use a query instruction prefix for asymmetric
# (short query -> longer passage) retrieval quality. We embed questions on
# both sides here (question <-> question), so we keep this empty by default;
# flip to the official bge instruction if you later embed question->answer pairs.
EMBEDDING_QUERY_INSTRUCTION = os.getenv("RAG_EMBEDDING_QUERY_INSTRUCTION", "")

# --------------------------------------------------------------------------- #
# Reranking
# --------------------------------------------------------------------------- #
# "flashrank" (fast, CPU-friendly, no torch dependency) or "cross_encoder"
# (uses sentence-transformers CrossEncoder, more accurate, heavier).
RERANK_METHOD = os.getenv("RAG_RERANK_METHOD", "flashrank")
FLASHRANK_MODEL_NAME = os.getenv("RAG_FLASHRANK_MODEL", "ms-marco-MiniLM-L-12-v2")
CROSS_ENCODER_MODEL_NAME = os.getenv(
    "RAG_CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# --------------------------------------------------------------------------- #
# LLM
# --------------------------------------------------------------------------- #
LLM_PROVIDER = os.getenv("RAG_LLM_PROVIDER", "openai")  # "openai" | "anthropic"
LLM_MODEL_NAME = os.getenv("RAG_LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("RAG_LLM_TEMPERATURE", "0.0"))

# Separate (ideally cheaper/faster) model used purely for the faithfulness /
# hallucination judge and for RAGAS evaluation, so evaluator drift doesn't
# track 1:1 with the generator model.
JUDGE_LLM_MODEL_NAME = os.getenv("RAG_JUDGE_LLM_MODEL", LLM_MODEL_NAME)

# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
DENSE_WEIGHT = float(os.getenv("RAG_DENSE_WEIGHT", "0.6"))
SPARSE_WEIGHT = float(os.getenv("RAG_SPARSE_WEIGHT", "0.4"))
TOP_K_DENSE = int(os.getenv("RAG_TOP_K_DENSE", "10"))
TOP_K_SPARSE = int(os.getenv("RAG_TOP_K_SPARSE", "10"))
TOP_K_RERANK = int(os.getenv("RAG_TOP_K_RERANK", "4"))

# --------------------------------------------------------------------------- #
# Guardrails
# --------------------------------------------------------------------------- #
SAFETY_SENSITIVE_FLAG = "safety_sensitive"

# Lightweight lexical trigger list for detecting crisis intent directly in the
# *user's query*, independent of what gets retrieved (e.g. the retriever
# might miss and return low-relevance docs, but the query itself still
# signals risk and must be handled safely regardless of retrieval quality).
CRISIS_KEYWORDS: tuple[str, ...] = (
    "suicide", "suicidal", "kill myself", "end my life", "want to die",
    "self-harm", "self harm", "hurting myself", "cutting myself",
    "overdose", "no reason to live", "better off dead", "can't go on",
    "ending it all",
)

CRISIS_DISCLAIMER = (
    "**If you or someone you know is in crisis, help is available right now:**\n\n"
    "- **Call or text 988** — the 988 Suicide & Crisis Lifeline (US), available 24/7\n"
    "- **Text HOME to 741741** — Crisis Text Line\n"
    "- **Call 911** if there is immediate danger to life\n"
    "- Outside the US, please contact your local emergency number or a local crisis line.\n\n"
    "The information below is educational and does not replace care from a qualified "
    "professional or emergency responder."
)

# --------------------------------------------------------------------------- #
# Hallucination / faithfulness guardrail
# --------------------------------------------------------------------------- #
# If the verifier's faithfulness score (0-1, self-reported by the judge LLM)
# falls below this threshold, we do not show the generated answer and instead
# show a safe fallback message.
FAITHFULNESS_FALLBACK_THRESHOLD = float(os.getenv("RAG_FAITHFULNESS_THRESHOLD", "0.5"))

FALLBACK_NOTICE = (
    "I couldn't verify this answer against the available reference material with "
    "enough confidence to show it to you. Rather than risk giving you inaccurate "
    "information on a health-related topic, please consult the linked sources "
    "directly or speak with a qualified professional."
)

# --------------------------------------------------------------------------- #
# RAGAS evaluation
# --------------------------------------------------------------------------- #
RAGAS_ENABLED = os.getenv("RAG_RAGAS_ENABLED", "true").lower() == "true"
