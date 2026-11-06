"""
vector_store.py - ChromaDB dual collection manager with local SentenceTransformer embeddings.
Indexes fixed-size-with-overlap chunks and sentence-based chunks into separate collections.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
import chromadb
from chromadb.api.types import Documents, Embeddings
from sentence_transformers import SentenceTransformer

from app.config import (
    BASE_DIR,
    KB_DIR,
    COLLECTION_FIXED,
    COLLECTION_SENTENCE,
    SIMILARITY_THRESHOLD
)
from app.rag.chunking import load_and_chunk_all

# ChromaDB persistence directory
CHROMA_PERSIST_DIR = BASE_DIR / "data" / "chroma_db"

class LocalSentenceTransformerEF(chromadb.EmbeddingFunction[Documents]):
    """
    Offline local embedding function using pre-cached all-MiniLM-L6-v2.
    Requires zero network access and zero API keys.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name, local_files_only=True)
        
    @staticmethod
    def name() -> str:
        return "local_sentence_transformer"

    def __call__(self, input: Documents) -> Embeddings:
        if isinstance(input, str):
            input = [input]
        embeddings = self.model.encode(input, normalize_embeddings=True)
        return embeddings.tolist()

# Singleton Chroma Client & Embedding Function
_client = None
_embedding_function = None

def get_chroma_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    return _client

def get_embedding_function() -> LocalSentenceTransformerEF:
    global _embedding_function
    if _embedding_function is None:
        _embedding_function = LocalSentenceTransformerEF()
    return _embedding_function

def index_knowledge_base():
    """
    Chunks all documents in knowledge_base/ under both strategies
    and upserts into two separate ChromaDB collections.
    """
    client = get_chroma_client()
    ef = get_embedding_function()
    
    fixed_chunks, sentence_chunks = load_and_chunk_all(KB_DIR)
    
    # 1. Index Fixed-size chunks collection
    col_fixed = client.get_or_create_collection(
        name=COLLECTION_FIXED,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )
    if fixed_chunks:
        col_fixed.upsert(
            ids=[c["chunk_id"] for c in fixed_chunks],
            documents=[c["text"] for c in fixed_chunks],
            metadatas=[
                {"doc_id": c["doc_id"], "strategy": c["strategy"], "chunk_index": c["chunk_index"]}
                for c in fixed_chunks
            ]
        )
        
    # 2. Index Sentence chunks collection
    col_sent = client.get_or_create_collection(
        name=COLLECTION_SENTENCE,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )
    if sentence_chunks:
        col_sent.upsert(
            ids=[c["chunk_id"] for c in sentence_chunks],
            documents=[c["text"] for c in sentence_chunks],
            metadatas=[
                {"doc_id": c["doc_id"], "strategy": c["strategy"], "sentence_index": c["sentence_index"]}
                for c in sentence_chunks
            ]
        )
        
    print(f"Indexed {len(fixed_chunks)} fixed chunks into collection '{COLLECTION_FIXED}'.")
    print(f"Indexed {len(sentence_chunks)} sentence chunks into collection '{COLLECTION_SENTENCE}'.")
    return len(fixed_chunks), len(sentence_chunks)

def upsert_chunks(collection_name: str, chunks: List[Dict[str, Any]]) -> int:
    """Upserts a list of chunks into the designated ChromaDB collection."""
    if not chunks:
        return 0
    client = get_chroma_client()
    ef = get_embedding_function()
    col = client.get_or_create_collection(
        name=collection_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )
    metas = []
    for c in chunks:
        m = {"doc_id": c["doc_id"], "strategy": c["strategy"]}
        if "chunk_index" in c:
            m["chunk_index"] = c["chunk_index"]
        if "sentence_index" in c:
            m["sentence_index"] = c["sentence_index"]
        metas.append(m)
    col.upsert(
        ids=[c["chunk_id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=metas
    )
    return len(chunks)

def query_collection(collection_name: str, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Retrieves top_k relevant chunks from the specified collection.
    Converts cosine distance to cosine similarity (similarity = 1.0 - distance).
    """
    client = get_chroma_client()
    ef = get_embedding_function()
    
    col = client.get_collection(name=collection_name, embedding_function=ef)
    results = col.query(
        query_texts=[query_text],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    retrieved = []
    if results and results["documents"] and results["documents"][0]:
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]
        
        for doc, meta, dist in zip(docs, metas, dists):
            # Chroma cosine distance is in [0, 2], where 0 is identical
            # Cosine similarity = 1.0 - (distance / 2.0) or 1.0 - distance for normalized vectors
            similarity = max(0.0, 1.0 - float(dist))
            retrieved.append({
                "text": doc,
                "metadata": meta,
                "doc_id": meta.get("doc_id", "unknown"),
                "distance": float(dist),
                "similarity": round(similarity, 4)
            })
            
    return retrieved
