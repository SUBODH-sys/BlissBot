"""
Streamlit UI for the enterprise RAG demo.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import chains, config, evaluation, ingestion, retrieval  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Safety-Aware Medical RAG", page_icon="🩺", layout="wide")


# --------------------------------------------------------------------------- #
# Cached resource construction (runs once per process / cache invalidation)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading corpus and building indices...")
def get_pipeline_resources(data_path_str: str, force_rebuild: bool):
    documents, docstore, vectorstore, embeddings = ingestion.load_corpus(
        Path(data_path_str), force_rebuild_index=force_rebuild
    )
    retriever = retrieval.build_retrieval_pipeline(documents, vectorstore)
    llm = chains.get_chat_model(config.LLM_MODEL_NAME)
    judge_llm = chains.get_chat_model(config.JUDGE_LLM_MODEL_NAME, temperature=0.0)
    rag_chain = chains.build_rag_chain(retriever, llm=llm, judge_llm=judge_llm)
    return {
        "docstore": docstore,
        "retriever": retriever,
        "rag_chain": rag_chain,
        "llm": llm,
        "judge_llm": judge_llm,
        "embeddings": embeddings,
        "num_docs": len(documents),
    }


def render_sidebar(resources: dict) -> None:
    with st.sidebar:
        st.header("⚙️ System status")
        st.caption(f"Corpus size: **{resources['num_docs']}** QA pairs")
        st.caption(f"Embedding model: `{config.EMBEDDING_MODEL_NAME}`")
        st.caption(f"Rerank method: `{config.RERANK_METHOD}`")
        st.caption(f"Generator LLM: `{config.LLM_MODEL_NAME}`")
        st.caption(f"Judge LLM: `{config.JUDGE_LLM_MODEL_NAME}`")

        if not os.getenv("OPENAI_API_KEY") and config.LLM_PROVIDER == "openai":
            st.warning(
                "OPENAI_API_KEY is not set. Set it in your environment or a `.env` "
                "file before asking a question.",
                icon="⚠️",
            )

        st.divider()
        st.header("🔧 Retrieval settings")
        st.caption(f"Hybrid weights — dense: {config.DENSE_WEIGHT}, sparse: {config.SPARSE_WEIGHT}")
        st.caption(f"Top-k retrieved → reranked to: {config.TOP_K_RERANK}")
        if st.button("Force rebuild FAISS index"):
            st.cache_resource.clear()
            st.rerun()

        st.divider()
        st.caption(
            "This is an educational demo. It is not a substitute for professional "
            "medical advice, diagnosis, or treatment."
        )


def render_result(result: chains.RAGResult, resources: dict) -> None:
    if result["crisis"]["is_crisis"]:
        st.error("Crisis-related content detected — safety resources shown above the answer.", icon="🚨")

    st.markdown(result["answer"])

    verification = result["verification"]
    with st.expander("🔍 Faithfulness guardrail", expanded=not verification["passed"]):
        st.metric("Faithfulness score (judge LLM)", f"{verification['faithfulness_score']:.2f}")
        st.caption(f"Threshold for pass: {config.FAITHFULNESS_FALLBACK_THRESHOLD}")
        st.write(verification["reasoning"] or "—")
        if verification["unsupported_claims"]:
            st.write("**Unsupported claims flagged:**")
            for claim in verification["unsupported_claims"]:
                st.write(f"- {claim}")

    with st.expander(f"📚 Retrieved sources ({len(result['docs'])})", expanded=False):
        if not result["docs"]:
            st.write("No sources retrieved.")
        for i, doc in enumerate(result["docs"], start=1):
            meta = doc.metadata
            flags = meta.get("flags") or []
            flag_badges = " ".join(f"`{f}`" for f in flags) if flags else "—"
            st.markdown(
                f"**[Source {i}]** `{meta.get('doc_id')}`  \n"
                f"**Source org:** {meta.get('source', 'unknown')} &nbsp;|&nbsp; "
                f"**Topic:** {meta.get('topic_group', 'unknown')} &nbsp;|&nbsp; "
                f"**Region:** {meta.get('region', 'unknown')} &nbsp;|&nbsp; "
                f"**Flags:** {flag_badges}"
            )
            st.caption(meta.get("url", ""))
            st.write(f"Q: {meta.get('question')}")
            st.write(f"A: {meta.get('answer')}")
            st.markdown("---")

    with st.expander("📊 RAGAS evaluation", expanded=False):
        with st.spinner("Scoring answer with RAGAS metrics..."):
            scores = evaluation.evaluate_turn(
                question=result["question"],
                answer=result["raw_answer"] or result["answer"],
                docs=result["docs"],
                judge_llm=resources["judge_llm"],
                embeddings=resources["embeddings"],
            )
        if scores["error"]:
            st.info(scores["error"])
        else:
            cols = st.columns(3)
            cols[0].metric("Faithfulness", scores["faithfulness"])
            cols[1].metric("Answer relevancy", scores["answer_relevancy"])
            cols[2].metric("Context precision", scores["context_precision"])


def main() -> None:
    st.title("🩺 Safety-Aware Medical RAG")
    st.caption(
        "Hybrid FAISS + BM25 retrieval, cross-encoder reranking, crisis guardrails, "
        "and live faithfulness/RAGAS scoring — built with LCEL."
    )

    resources = get_pipeline_resources(str(config.DATA_PATH), force_rebuild=False)
    render_sidebar(resources)

    if "history" not in st.session_state:
        st.session_state.history = []

    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            render_result(turn["result"], resources)

    question = st.chat_input("Ask a health or safety question...")
    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            start = time.perf_counter()
            with st.spinner("Retrieving, reranking, and generating..."):
                try:
                    result = chains.run_query(resources["rag_chain"], question)
                except Exception:  # noqa: BLE001
                    logger.exception("Unhandled error answering question")
                    st.error("An unexpected error occurred. Please try again.")
                    return
            elapsed = time.perf_counter() - start
            render_result(result, resources)
            st.caption(f"Answered in {elapsed:.2f}s")

        st.session_state.history.append({"question": question, "result": result})


if __name__ == "__main__":
    main()
