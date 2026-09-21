"""
Retrieval layer.

Builds a hybrid (dense + sparse) retriever and wraps it with a reranking
compressor. The final object is a standard LangChain `BaseRetriever`, which
means it is *already* a Runnable and can be dropped straight into an LCEL
`|` chain with no adapter code.

Pipeline:  FAISS (dense, question-embeddings)  ┐
                                                 ├─> EnsembleRetriever ─> Reranker ─> top-k Documents
           BM25 (sparse, question text)        ┘
"""
from __future__ import annotations

import logging

from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src import config

logger = logging.getLogger(__name__)


def build_bm25_retriever(documents: list[Document], k: int = config.TOP_K_SPARSE) -> BM25Retriever:
    """Sparse keyword retriever over the same question-only corpus used for FAISS.

    Keeping BM25 and FAISS over the *identical* text (questions only) means
    the ensemble is genuinely combining two matching strategies on the same
    field, rather than accidentally biasing toward whichever side has richer
    text.
    """
    retriever = BM25Retriever.from_documents(documents)
    retriever.k = k
    return retriever


def build_ensemble_retriever(
    vectorstore,
    bm25_retriever: BM25Retriever,
    dense_weight: float = config.DENSE_WEIGHT,
    sparse_weight: float = config.SPARSE_WEIGHT,
) -> EnsembleRetriever:
    """Combine dense (FAISS) and sparse (BM25) retrieval via Reciprocal Rank Fusion.

    weights=[0.6, 0.4] favors semantic/dense matches, which handles
    paraphrased user queries well, while BM25 still rescues exact
    terminology matches (drug names, symptom keywords) dense embeddings can
    sometimes under-weight.
    """
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": config.TOP_K_DENSE})
    return EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[dense_weight, sparse_weight],
    )


def _build_flashrank_compressor(top_n: int = config.TOP_K_RERANK):
    from langchain.retrievers.document_compressors import FlashrankRerank

    return FlashrankRerank(model=config.FLASHRANK_MODEL_NAME, top_n=top_n)


def _build_cross_encoder_compressor(top_n: int = config.TOP_K_RERANK):
    from langchain.retrievers.document_compressors import CrossEncoderReranker
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder

    cross_encoder = HuggingFaceCrossEncoder(model_name=config.CROSS_ENCODER_MODEL_NAME)
    return CrossEncoderReranker(model=cross_encoder, top_n=top_n)


def build_reranking_retriever(
    base_retriever: BaseRetriever,
    method: str = config.RERANK_METHOD,
    top_n: int = config.TOP_K_RERANK,
) -> ContextualCompressionRetriever:
    """Wrap the ensemble retriever with a cross-encoder style reranker.

    `method="flashrank"` (default) uses FlashRank -- a small ONNX
    cross-encoder with no torch dependency, so it's fast and light enough
    for interactive demos. `method="cross_encoder"` uses a
    sentence-transformers CrossEncoder for higher accuracy at the cost of
    latency/footprint. Both implement LangChain's `BaseDocumentCompressor`
    interface, so they're interchangeable behind `ContextualCompressionRetriever`.
    """
    if method == "flashrank":
        try:
            compressor = _build_flashrank_compressor(top_n=top_n)
        except ImportError:
            logger.warning(
                "flashrank not installed; falling back to cross_encoder reranker. "
                "Run `pip install flashrank` to use the configured method."
            )
            compressor = _build_cross_encoder_compressor(top_n=top_n)
    elif method == "cross_encoder":
        compressor = _build_cross_encoder_compressor(top_n=top_n)
    else:
        raise ValueError(f"Unknown RERANK_METHOD '{method}'. Use 'flashrank' or 'cross_encoder'.")

    return ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )


def build_retrieval_pipeline(
    documents: list[Document],
    vectorstore,
) -> BaseRetriever:
    """Full retrieval stack: dense + sparse ensemble, then reranked to top-k.

    Returns a single `BaseRetriever` (a Runnable) ready to be used directly
    inside an LCEL chain, e.g. `retriever.invoke(question)` or
    `retriever | RunnableLambda(format_docs)`.
    """
    bm25_retriever = build_bm25_retriever(documents)
    ensemble_retriever = build_ensemble_retriever(vectorstore, bm25_retriever)
    reranking_retriever = build_reranking_retriever(ensemble_retriever)
    logger.info(
        "Retrieval pipeline ready: dense(%d) + sparse(%d) -> ensemble -> rerank(top_%d, %s)",
        config.TOP_K_DENSE,
        config.TOP_K_SPARSE,
        config.TOP_K_RERANK,
        config.RERANK_METHOD,
    )
    return reranking_retriever


def format_docs(docs: list[Document]) -> str:
    """Render retrieved Documents into an LLM-ready context block.

    Pulls the *answer* text back out of metadata (the vector index only
    stores questions) so the generator sees full Q&A context, each entry
    tagged with its doc_id/source for traceable, citable generation.
    """
    blocks = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        blocks.append(
            f"[Source {i}] doc_id={meta.get('doc_id', 'unknown')} "
            f"source={meta.get('source', 'unknown')}\n"
            f"Q: {meta.get('question', doc.page_content)}\n"
            f"A: {meta.get('answer', '')}"
        )
    return "\n\n".join(blocks) if blocks else "No relevant context found."
