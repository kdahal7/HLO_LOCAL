# decision_engine.py

import hashlib
import time
import logging
from typing import Dict, Any, List, TYPE_CHECKING

# Avoid circular import issues
if TYPE_CHECKING:
    from local_llm_processor import LLMProcessor

# Add the missing logger configuration
logger = logging.getLogger(__name__)

class DecisionEngine:
    """
    Coordinates the process of generating an answer, with smart, contextual caching.
    """
    def __init__(self, llm_processor: 'LLMProcessor'):
        self.llm_processor = llm_processor
        self.cache = {}
        self.max_cache_size = 200
        logger.info("Initialized DecisionEngine with contextual caching.")

    def _get_cache_key(self, question: str, doc_hash: str) -> str:
        """Creates a unique cache key based on the question and the document's hash."""
        key_text = f"{question.lower().strip()}|{doc_hash}"
        return hashlib.md5(key_text.encode()).hexdigest()

    def generate_answer(self, question: str, doc_hash: str, retrieved_chunks: List[Dict]) -> Dict[str, Any]:
        """Generates an answer for a question, using the cache if a previous answer for the same document exists."""
        
        cache_key = self._get_cache_key(question, doc_hash)
        if cache_key in self.cache:
            logger.info(f"✓ Cache hit for question: '{question[:30]}...' on doc: {doc_hash}")
            return self.cache[cache_key]

        logger.info(f"▷ Cache miss. Generating new answer for: '{question[:30]}...'")
        
        # Generate a new answer using our high-quality LLM processor
        result = self.llm_processor.generate_answer(question, retrieved_chunks)
        
        # Cache the new result if it's confident
        if result.get("confidence") in ["high", "medium"]:
            if len(self.cache) >= self.max_cache_size:
                # Simple First-In-First-Out (FIFO) cache eviction
                oldest_key = next(iter(self.cache))
                self.cache.pop(oldest_key)
                logger.debug(f"Evicted cache entry: {oldest_key}")
            self.cache[cache_key] = result
        
        return result

    def get_cache_stats(self) -> Dict[str, Any]:
        return {
            "cache_size": len(self.cache),
            "max_cache_size": self.max_cache_size,
            "cache_hit_ratio": getattr(self, '_cache_hits', 0) / max(getattr(self, '_total_requests', 1), 1)
        }
        
    def clear_cache(self):
        self.cache.clear()
        logger.info("Decision engine cache has been cleared.")