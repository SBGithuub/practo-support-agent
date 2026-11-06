"""
chunking.py - Two document chunking strategies: Fixed-size with overlap and Sentence-based.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any

def get_sentence_chunks(doc_text: str, doc_name: str) -> List[Dict[str, Any]]:
    """
    Splits document into granular sentence-based chunks.
    Preserves document parent metadata for precision/recall evaluation.
    """
    # Regex split on sentence boundaries
    raw_sentences = re.split(r'(?<=[.!?])\s+', doc_text.strip())
    chunks = []
    
    for idx, sent in enumerate(raw_sentences):
        sent_clean = sent.strip()
        if not sent_clean:
            continue
            
        chunk_id = f"{doc_name}_sent_{idx+1}"
        chunks.append({
            "chunk_id": chunk_id,
            "doc_id": doc_name,
            "text": sent_clean,
            "strategy": "sentence",
            "sentence_index": idx + 1
        })
        
    return chunks

def get_fixed_chunks(doc_text: str, doc_name: str, chunk_size: int = 180, overlap: int = 40) -> List[Dict[str, Any]]:
    """
    Splits document into fixed-size character windows with designated overlap.
    Preserves document parent metadata for precision/recall evaluation.
    """
    text = doc_text.strip()
    chunks = []
    start = 0
    chunk_idx = 1
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_str = text[start:end].strip()
        
        if chunk_str:
            chunk_id = f"{doc_name}_fixed_{chunk_idx}"
            chunks.append({
                "chunk_id": chunk_id,
                "doc_id": doc_name,
                "text": chunk_str,
                "strategy": "fixed",
                "chunk_index": chunk_idx,
                "start_char": start,
                "end_char": end
            })
            chunk_idx += 1
            
        if end >= len(text):
            break
        start += (chunk_size - overlap)
        
    return chunks

def load_and_chunk_all(kb_dir: Path) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Loads all .txt documents in knowledge_base/ and creates both chunking sets.
    """
    fixed_chunks = []
    sentence_chunks = []
    
    for file_path in sorted(kb_dir.glob("*.txt")):
        doc_name = file_path.name
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            
        fixed_chunks.extend(get_fixed_chunks(text, doc_name))
        sentence_chunks.extend(get_sentence_chunks(text, doc_name))
        
    return fixed_chunks, sentence_chunks

