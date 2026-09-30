"""
rag_pipeline.py
Reusable RAG pipeline: JSON loading -> embeddings -> ChromaDB -> hybrid (dense + BM25 / RRF) retrieval -> Groq LLM.
Import this from both the pipeline notebook and the evaluation notebook.
"""
import os
import re
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import JSONLoader
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths (resolved from the project root, so they work from any notebook/script)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JSON_PATH = PROJECT_ROOT / "data" / "json_files" / "bliss_corpus.json"
DEFAULT_PERSIST_DIR = PROJECT_ROOT / "data" / "vector_store"
DEFAULT_COLLECTION = "json_qa_documents"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_LLM_MODEL = "qwen/qwen3.8-27b"


# ---------------------------------------------------------------------------
# Data ingestion
# ---------------------------------------------------------------------------
def _extract_metadata(record: dict, metadata: dict) -> dict:
    meta = record.get("metadata", {})
    metadata["doc_id"] = record.get("doc_id")
    metadata["source"] = meta.get("source")
    metadata["url"] = meta.get("url")
    metadata["topic_group"] = meta.get("topic_group")
    metadata["flags"] = meta.get("flags")
    return metadata


def load_documents(file_path: Path = DEFAULT_JSON_PATH) -> List[Document]:
    """Load the Q&A JSON corpus as LangChain Documents (one Document per Q&A pair, no chunking needed)."""
    loader = JSONLoader(
        file_path=str(file_path),
        jq_schema=".[]",
        content_key='. | "Question: " + .question + "\\nAnswer: " + .answer',
        is_content_key_jq_parsable=True,
        metadata_func=_extract_metadata,
    )
    return loader.load()


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
@dataclass
class EmbeddedDocument:
    """Pairs an embedding vector with its source document."""
    embedding: np.ndarray
    text: str
    metadata: Dict[str, Any]


class EmbeddingManager:
    """Generates embeddings with SentenceTransformer."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, verbose: bool = False):
        self.model_name = model_name
        self.verbose = verbose
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            print(f"Loading embedding model: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model loaded. Embedding dimension: {self.model.get_embedding_dimension()}")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise

    def generate_embeddings(self, texts: List[str], show_progress_bar: bool = False) -> np.ndarray:
        if not self.model:
            raise ValueError("Model not loaded.")
        if self.verbose:
            print(f"Generating embeddings for {len(texts)} texts...")
        return self.model.encode(texts, show_progress_bar=show_progress_bar)

    def embed_documents(self, documents: List[Document]) -> List[EmbeddedDocument]:
        if not documents:
            raise ValueError("No documents provided for embedding.")
        texts = [d.page_content for d in documents]
        embeddings = self.generate_embeddings(texts, show_progress_bar=True)
        return [
            EmbeddedDocument(embeddings[i], documents[i].page_content, documents[i].metadata)
            for i in range(len(documents))
        ]


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------
class VectorStore:
    """Manages document embeddings in a persistent ChromaDB collection."""

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION,
        persist_directory: Path = DEFAULT_PERSIST_DIR,
    ):
        self.collection_name = collection_name
        self.persist_directory = str(persist_directory)
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "JSON Q&A document embeddings for RAG", "hnsw:space": "cosine"},
            )
            print(f"Vector store ready. Collection: {self.collection_name} "
                  f"({self.collection.count()} documents)")
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    @staticmethod
    def _flatten_list(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """ChromaDB only accepts str/int/float/bool metadata values, so flatten lists and None."""
        clean = {}
        for k, v in metadata.items():
            if isinstance(v, list):
                clean[k] = ", ".join(str(x) for x in v)
            elif v is None:
                clean[k] = ""
            else:
                clean[k] = v
        return clean

    def add_documents(self, documents: List[Document], embeddings: np.ndarray):
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        ids, metadatas, texts, emb_list = [], [], [], []
        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            ids.append(doc.metadata.get("doc_id") or f"doc_{uuid.uuid4().hex[:8]}_{i}")
            meta = self._flatten_list(doc.metadata)
            meta["doc_index"] = i
            meta["content_length"] = len(doc.page_content)
            metadatas.append(meta)
            texts.append(doc.page_content)
            emb_list.append(emb.tolist())

        self.collection.add(ids=ids, embeddings=emb_list, metadatas=metadatas, documents=texts)
        print(f"Added {len(documents)} documents. Total in collection: {self.collection.count()}")


def index_documents(
    documents: List[Document],
    vector_store: VectorStore,
    embedding_manager: EmbeddingManager,
    force: bool = False,
) -> None:
    """Embed and store documents ONLY if the collection is empty (avoids duplicate-id errors on re-runs)."""
    if vector_store.collection.count() > 0 and not force:
        print("Collection already populated, skipping indexing.")
        return
    embeddings = embedding_manager.generate_embeddings(
        [d.page_content for d in documents], show_progress_bar=True
    )
    vector_store.add_documents(documents, embeddings)


# ---------------------------------------------------------------------------
# BM25 (lexical) index
# ---------------------------------------------------------------------------
def tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Index:
    """BM25 index built from the documents already stored in the VectorStore's collection."""

    def __init__(self, vector_store: VectorStore):
        all_docs = vector_store.collection.get(include=["documents", "metadatas"])
        self.doc_ids = all_docs["ids"]
        self.doc_texts = all_docs["documents"]
        self.doc_metadatas = all_docs["metadatas"]
        self.bm25 = BM25Okapi([tokenize(t) for t in self.doc_texts])
        print(f"BM25 index built over {len(self.doc_ids)} documents")

    def query(self, query_text: str, top_k: int) -> List[Dict[str, Any]]:
        scores = self.bm25.get_scores(tokenize(query_text))
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            {
                "id": self.doc_ids[i],
                "content": self.doc_texts[i],
                "metadata": self.doc_metadatas[i],
                "score": float(scores[i]),
            }
            for i in top_indices
        ]


# ---------------------------------------------------------------------------
# Hybrid retriever (dense + BM25 via Reciprocal Rank Fusion)
# ---------------------------------------------------------------------------
class HybridRetriever:
    """Fuses dense (cosine) and BM25 rankings with Reciprocal Rank Fusion.

    retrieve() returns a list of dicts with:
    'id', 'content', 'metadata', 'similarity_score' (fused RRF score), 'distance', 'rank'.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_manager: EmbeddingManager,
        bm25_index: BM25Index,
        fusion_pool_size: int = 20,
        rrf_k: int = 10,
        verbose: bool = False,
    ):
        """
        Args:
            fusion_pool_size: candidates pulled from EACH retriever before fusion
            rrf_k: RRF damping constant (10 here; 60 is the common default in the literature)
            verbose: print retrieval details on every call
        """
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.bm25_index = bm25_index
        self.fusion_pool_size = fusion_pool_size
        self.rrf_k = rrf_k
        self.verbose = verbose

    def _dense_candidates(self, query: str, pool_size: int, where: Optional[Dict[str, Any]] = None):
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        kwargs = {"query_embeddings": [query_embedding.tolist()], "n_results": pool_size}
        if where:
            kwargs["where"] = where
        results = self.vector_store.collection.query(**kwargs)

        candidates = {}
        if results["documents"] and results["documents"][0]:
            for doc_id, doc, meta, dist in zip(
                results["ids"][0], results["documents"][0],
                results["metadatas"][0], results["distances"][0],
            ):
                candidates[doc_id] = {
                    "content": doc, "metadata": meta,
                    "similarity_score": 1 - dist, "distance": dist,
                }
        return candidates

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        score_threshold: float = 0.0,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Args:
            score_threshold: minimum fused RRF score (NOT comparable to cosine similarity; leave at 0.0)
            where: metadata filter, applied on the dense side only
        """
        if self.verbose:
            print(f"Retrieving for: '{query}' (top_k={top_k}, pool={self.fusion_pool_size}, rrf_k={self.rrf_k})")

        try:
            dense = self._dense_candidates(query, self.fusion_pool_size, where=where)
            bm25 = self.bm25_index.query(query, self.fusion_pool_size)

            rrf_scores: Dict[str, float] = {}
            for rank, doc_id in enumerate(dense.keys(), start=1):
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank)
            for rank, cand in enumerate(bm25, start=1):
                rrf_scores[cand["id"]] = rrf_scores.get(cand["id"], 0.0) + 1.0 / (self.rrf_k + rank)

            fused_order = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
            bm25_lookup = {c["id"]: c for c in bm25}

            retrieved = []
            for doc_id, fused_score in fused_order:
                if fused_score < score_threshold:
                    continue

                if doc_id in dense:
                    content = dense[doc_id]["content"]
                    metadata = dense[doc_id]["metadata"]
                    distance = dense[doc_id]["distance"]
                else:  # found only by BM25
                    content = bm25_lookup[doc_id]["content"]
                    metadata = bm25_lookup[doc_id]["metadata"]
                    distance = None

                retrieved.append({
                    "id": doc_id,
                    "content": content,
                    "metadata": metadata,
                    "similarity_score": fused_score,
                    "distance": distance,
                    "rank": len(retrieved) + 1,
                })
                if len(retrieved) >= top_k:
                    break

            if self.verbose:
                print(f"Retrieved {len(retrieved)} documents")
            return retrieved

        except Exception as e:
            print(f"Error during hybrid retrieval: {e}")
            return []


# ---------------------------------------------------------------------------
# LLM + RAG function
# ---------------------------------------------------------------------------
def build_llm(
    model_name: str = DEFAULT_LLM_MODEL,
    temperature: float = 0.1,
    max_tokens: int = 1024,
) -> ChatGroq:
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY missing from .env")
    return ChatGroq(groq_api_key=api_key, model_name=model_name,
                    temperature=temperature, max_tokens=max_tokens)


def build_hybrid_retriever(
    collection_name: str = DEFAULT_COLLECTION,
    persist_directory: Path = DEFAULT_PERSIST_DIR,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    verbose: bool = False,
) -> HybridRetriever:
    """Load the persisted Chroma store and assemble the retriever (no re-embedding of the corpus)."""
    embedding_manager = EmbeddingManager(embedding_model, verbose=verbose)
    vector_store = VectorStore(collection_name, persist_directory)
    if vector_store.collection.count() == 0:
        raise RuntimeError("Vector store is empty. Run index_documents(load_documents(), ...) first.")
    bm25_index = BM25Index(vector_store)
    return HybridRetriever(vector_store, embedding_manager, bm25_index, verbose=verbose)


PROMPT_TEMPLATE = (
    "Use the following context to answer the question concisely. "
    "Answer only from the context.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


def rag_simple(query: str, retriever, llm, top_k: int = 3) -> str:
    """Retrieve context, then generate an answer with the LLM."""
    results = retriever.retrieve(query, top_k=top_k)
    context = "\n\n".join(d["content"] for d in results) if results else ""
    if not context:
        return "No relevant context found to answer the question."

    # Build the prompt with .format on the TEMPLATE only, so braces inside retrieved text are safe.
    prompt = PROMPT_TEMPLATE.format(context=context, question=query)
    return llm.invoke(prompt).content