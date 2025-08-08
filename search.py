# search.py

from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np
from typing import List, Dict, Tuple
import re
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Processes a document by breaking it into high-quality, overlapping chunks.
    This ensures that context is not lost between chunks.
    """
    def __init__(self, chunk_size: int = 768, chunk_overlap_ratio: float = 0.15):
        self.chunk_size = chunk_size
        self.chunk_overlap = int(chunk_size * chunk_overlap_ratio)
        logger.info(f"Initialized DocumentProcessor with chunk_size={self.chunk_size} and overlap={self.chunk_overlap}")

    def create_chunks(self, text: str) -> List[Dict[str, any]]:
        """Creates high-quality, overlapping chunks from the document text."""
        if not text:
            return []

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split by sentences to avoid breaking words or ideas
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk_sentences = []
        current_length = 0
        chunk_id = 0

        for sentence in sentences:
            # If the current chunk is full, store it and start a new one
            if current_length + len(sentence) > self.chunk_size and current_chunk_sentences:
                chunk_text = " ".join(current_chunk_sentences)
                chunks.append({"id": chunk_id, "text": chunk_text})
                chunk_id += 1
                
                # Create the next chunk with an overlap of the last few sentences
                overlap_text = " ".join(current_chunk_sentences)
                overlap_start_index = max(0, len(overlap_text) - self.chunk_overlap)
                overlapped_part = overlap_text[overlap_start_index:].split(' ', 1)[-1] # Ensure we don't start mid-word
                
                current_chunk_sentences = [overlapped_part, sentence]
                current_length = len(overlapped_part) + len(sentence)
            else:
                current_chunk_sentences.append(sentence)
                current_length += len(sentence)

        # Add the final chunk
        if current_chunk_sentences:
            chunks.append({"id": chunk_id, "text": " ".join(current_chunk_sentences)})
            
        logger.info(f"Created {len(chunks)} high-quality chunks from the document.")
        return chunks

class SemanticSearch:
    """
    A powerful two-stage search system:
    1. Fast Retrieval: Quickly finds a broad set of potentially relevant chunks.
    2. High-Accuracy Re-ranking: Carefully re-ranks the candidates to find the best matches for the question.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.retrieval_model = SentenceTransformer(model_name)
        self.rerank_model = CrossEncoder(cross_encoder_model)
        logger.info(f"Loaded retrieval model '{model_name}' and re-ranker '{cross_encoder_model}'")

    def build_index(self, chunks: List[Dict[str, any]]) -> faiss.Index:
        """Builds a FAISS index for the fast retrieval stage."""
        if not chunks:
            raise ValueError("Cannot build index from empty or invalid chunks.")
            
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.retrieval_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings.astype('float32'))
        
        logger.info(f"Built FAISS index with {index.ntotal} vectors.")
        return index

    def search(self, index: faiss.Index, query: str, chunks: List[Dict[str, any]], top_k: int = 5) -> List[Dict[str, any]]:
        """Performs the two-stage search to find the most relevant chunks."""
        if not chunks:
            return []

        # Stage 1: Fast Retrieval using FAISS
        # We retrieve more candidates than we need (e.g., 25) to ensure we don't miss the best ones.
        initial_k = min(25, len(chunks))
        query_embedding = self.retrieval_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        _, indices = index.search(query_embedding.astype('float32'), initial_k)
        
        retrieved_chunks = [chunks[i] for i in indices[0]]

        # Stage 2: High-Accuracy Re-ranking using the Cross-Encoder
        # The cross-encoder is more powerful because it compares the query directly with each chunk text.
        rerank_inputs = [[query, chunk['text']] for chunk in retrieved_chunks]
        scores = self.rerank_model.predict(rerank_inputs)

        # Combine the chunks with their new, more accurate scores
        reranked_results = sorted(zip(scores, retrieved_chunks), key=lambda x: x[0], reverse=True)
        
        # Add the final relevance score to each chunk dictionary for traceability
        final_chunks = []
        for score, chunk in reranked_results[:top_k]:
            chunk['relevance_score'] = float(score)
            final_chunks.append(chunk)

        logger.info(f"Re-ranked {len(retrieved_chunks)} chunks, returning top {len(final_chunks)} for the LLM.")
        return final_chunks