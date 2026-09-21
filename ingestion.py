"""
Ingestion layer.

Responsibilities
-----------------
1. Load the raw JSON QA dataset and validate its shape.
2. Build one `Document` per QA pair whose `page_content` is the *question*
   only (this is what gets embedded) -- retrieval precision is much higher
   when we match query-to-question rather than query-to-answer, since users
   phrase their input as questions themselves.
3. Retain the full record (question, answer, and all metadata) in an
   in-memory docstore keyed by `doc_id`, so the generation step always has
   access to the complete Q&A pair and its provenance, not just the
   embedded fragment.
4. Build and persist a FAISS index over those question-only Documents.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TypedDict

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src import config

logger = logging.getLogger(__name__)


class QARecord(TypedDict):
    doc_id: str
    question: str
    answer: str
    metadata: dict


class DataValidationError(ValueError):
    """Raised when the input JSON dataset does not match the expected schema."""


REQUIRED_TOP_LEVEL_KEYS = {"doc_id", "question", "answer", "metadata"}


def load_qa_dataset(path: Path = config.DATA_PATH) -> list[QARecord]:
    """Load and validate the raw QA JSON file.

    Raises
    ------
    FileNotFoundError
        If the dataset file does not exist.
    DataValidationError
        If any record is missing required fields.
    json.JSONDecodeError
        If the file is not valid JSON.
    """
    if not path.exists():
        raise FileNotFoundError(f"QA dataset not found at {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise DataValidationError("Dataset root must be a JSON array of QA records.")

    records: list[QARecord] = []
    for i, item in enumerate(raw):
        missing = REQUIRED_TOP_LEVEL_KEYS - item.keys()
        if missing:
            raise DataValidationError(
                f"Record at index {i} (doc_id={item.get('doc_id', '?')}) "
                f"is missing required fields: {sorted(missing)}"
            )
        if not item["question"].strip() or not item["answer"].strip():
            raise DataValidationError(
                f"Record at index {i} (doc_id={item.get('doc_id')}) has an empty "
                f"question or answer."
            )
        records.append(item)  # type: ignore[arg-type]

    logger.info("Loaded %d QA records from %s", len(records), path)
    return records


def build_docstore(records: list[QARecord]) -> dict[str, QARecord]:
    """Full-fidelity lookup table: doc_id -> complete QA record.

    This is the source of truth for generation context. The vector index
    only ever holds *questions*; the answer text and metadata are always
    pulled back out of this docstore.
    """
    docstore: dict[str, QARecord] = {}
    for record in records:
        doc_id = record["doc_id"]
        if doc_id in docstore:
            logger.warning("Duplicate doc_id '%s' encountered; overwriting.", doc_id)
        docstore[doc_id] = record
    return docstore


def build_question_documents(records: list[QARecord]) -> list[Document]:
    """Build one Document per record with page_content = question only.

    All original metadata is preserved and enriched with `doc_id` and
    `answer` so downstream retrievers/rerankers/chains can access the full
    context without a second lookup -- while the *embedded* text stays
    tightly focused on the question for high-precision matching.
    """
    documents: list[Document] = []
    for record in records:
        metadata = dict(record.get("metadata", {}))
        metadata["doc_id"] = record["doc_id"]
        metadata["answer"] = record["answer"]
        metadata["question"] = record["question"]
        # Normalize flags to a list so downstream `in` checks are reliable
        # even if the source JSON omitted the field.
        metadata.setdefault("flags", [])
        documents.append(Document(page_content=record["question"], metadata=metadata))
    return documents


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Construct the shared embedding model (BAAI/bge-small-en-v1.5).

    CPU-friendly and small enough (~130MB) for local/demo deployment while
    still ranking well on MTEB retrieval benchmarks for its size class.
    """
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
        query_instruction=config.EMBEDDING_QUERY_INSTRUCTION or None,
    )


def build_or_load_faiss_index(
    documents: list[Document],
    embeddings: HuggingFaceEmbeddings,
    index_dir: Path = config.FAISS_INDEX_DIR,
    force_rebuild: bool = False,
):
    """Build a FAISS index from question Documents, or load a cached one.

    Rebuilding an embedding index on every process start is wasteful for a
    static/slow-changing corpus, so we persist to disk and only rebuild when
    asked or when no cache exists.
    """
    from langchain_community.vectorstores import FAISS

    index_file = index_dir / "index.faiss"
    if index_file.exists() and not force_rebuild:
        logger.info("Loading cached FAISS index from %s", index_dir)
        return FAISS.load_local(
            str(index_dir), embeddings, allow_dangerous_deserialization=True
        )

    logger.info("Building FAISS index over %d question documents", len(documents))
    vectorstore = FAISS.from_documents(documents, embeddings)
    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))
    return vectorstore


def load_corpus(
    data_path: Path = config.DATA_PATH,
    force_rebuild_index: bool = False,
) -> tuple[list[Document], dict[str, QARecord], "FAISS", HuggingFaceEmbeddings]:  # noqa: F821
    """Convenience entrypoint used by the retrieval layer / app startup.

    Returns (question_documents, docstore, faiss_vectorstore, embeddings).
    """
    records = load_qa_dataset(data_path)
    docstore = build_docstore(records)
    documents = build_question_documents(records)
    embeddings = get_embedding_model()
    vectorstore = build_or_load_faiss_index(
        documents, embeddings, force_rebuild=force_rebuild_index
    )
    return documents, docstore, vectorstore, embeddings
