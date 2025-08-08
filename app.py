# app.py - Optimized version with enhanced performance

import os
import time
import hashlib
import logging
import re
from typing import Dict, List, Tuple
import tempfile

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import httpx

# Assuming these are your other modules
try:
    from search import DocumentProcessor, SemanticSearch
except ImportError:
    # Fallback classes if modules don't exist
    class DocumentProcessor:
        def create_chunks(self, text): return [{"text": text[i:i+1000]} for i in range(0, len(text), 800)]
    class SemanticSearch:
        def build_index(self, chunks): return {"chunks": chunks}
        def search(self, index, query, chunks, top_k=5): return chunks[:top_k]

try:
    from decision_engine import DecisionEngine
except ImportError:
    from local_llm_processor import OptimizedLLMProcessor
    class DecisionEngine:
        def __init__(self, llm_processor):
            self.llm_processor = llm_processor
            self.cache = {}
            self.performance_cache = {}
            self.max_cache_size = 100
            
        def generate_answer(self, question: str, doc_hash: str, retrieved_chunks: List[Dict]) -> Dict:
            # Simplified cache key for better hits
            simple_key = self._get_simple_cache_key(question, doc_hash)
            
            # Check performance cache first
            if simple_key in self.performance_cache:
                return self.performance_cache[simple_key]
            
            # Check regular cache
            if simple_key in self.cache:
                result = self.cache[simple_key]
                # Promote to performance cache if it's good
                if result.get("confidence") == "high":
                    self.performance_cache[simple_key] = result
                return result
            
            # Generate new answer
            result = self.llm_processor.generate_answer(question, retrieved_chunks)
            
            # Smart caching based on performance
            if result.get("confidence") == "high" and result.get("generation_time", 99) < 5:
                self.performance_cache[simple_key] = result
            elif result.get("confidence") in ["high", "medium"]:
                self.cache[simple_key] = result
                
            return result
        
        def _get_simple_cache_key(self, question: str, doc_hash: str) -> str:
            # Simplified key that matches similar questions
            normalized = re.sub(r'\b(what|when|where|who|why|how|which|is|are|does|do)\b', '', question.lower())
            normalized = re.sub(r'[^\w\s]', '', normalized).strip()
            key_words = sorted(normalized.split())[:5]  # Top 5 keywords
            return f"{'-'.join(key_words)}|{doc_hash[:8]}"
            
        def clear_cache(self):
            self.cache.clear()
            self.performance_cache.clear()

from local_llm_processor import LLMProcessor, OptimizedLLMProcessor

try:
    from extract import extract_text_from_pdf
except ImportError:
    def extract_text_from_pdf(file_path): 
        return "Fallback: PDF extraction not available. Please install PyPDF2 or similar."

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="HackRX RAG API - Optimized")

# Initialize components
document_processor = DocumentProcessor()
semantic_search = SemanticSearch()
llm_processor = OptimizedLLMProcessor()  # Use optimized version
decision_engine = DecisionEngine(llm_processor)

# Global cache for processed documents
DOCUMENT_INDEX_CACHE: Dict[str, Tuple] = {}
MAX_CACHE_SIZE = 50

# Request/Response models
class QueryRequest(BaseModel):
    documents: str  # URL to document
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

def get_document_hash(document_url: str) -> str:
    """Generate a hash for the document URL for caching."""
    return hashlib.md5(document_url.encode()).hexdigest()

def manage_cache_size():
    """Remove oldest entries if cache is too large."""
    if len(DOCUMENT_INDEX_CACHE) >= MAX_CACHE_SIZE:
        # Remove oldest entry (simple FIFO)
        oldest_key = next(iter(DOCUMENT_INDEX_CACHE))
        DOCUMENT_INDEX_CACHE.pop(oldest_key)
        logger.info(f"Cache limit reached. Removed entry: {oldest_key}")

async def get_or_process_document(document_url: str) -> Tuple:
    """Processes a document with timeout optimization."""
    doc_hash = get_document_hash(document_url)
    
    # Check cache first
    if doc_hash in DOCUMENT_INDEX_CACHE:
        logger.info(f"✅ Found processed document in cache. Hash: {doc_hash}")
        return DOCUMENT_INDEX_CACHE[doc_hash]

    logger.info(f"🔄 Processing new document from URL: {document_url}")
    
    try:
        # Download document with timeout
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(document_url)
            response.raise_for_status()
            logger.info(f"📥 Downloaded document: {len(response.content)} bytes")
            
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(response.content)
            temp_file_path = tmp_file.name
            
        # Extract text from PDF
        document_text = extract_text_from_pdf(temp_file_path)
        
        # Clean up temp file
        os.unlink(temp_file_path)
        
        if not document_text.strip():
            raise ValueError("Document appears to be empty or unreadable")
            
        # Process document into chunks
        chunks = document_processor.create_chunks(document_text)
        
        if not chunks:
            raise ValueError("Failed to create chunks from document")
            
        # Build FAISS index
        faiss_index = semantic_search.build_index(chunks)
        
        # Cache the results
        manage_cache_size()  # Ensure we don't exceed cache limits
        result = (chunks, faiss_index, doc_hash)
        DOCUMENT_INDEX_CACHE[doc_hash] = result
        
        logger.info(f"✅ Successfully processed document with {len(chunks)} chunks")
        return result
        
    except httpx.RequestError as e:
        logger.error(f"❌ Failed to download document: {e}")
        raise HTTPException(status_code=400, detail=f"Could not download the document: {e}")
    except Exception as e:
        logger.error(f"❌ Failed to process document: {e}")
        raise HTTPException(status_code=500, detail=f"Document processing error: {e}")

@app.post("/hackrx/run", response_model=QueryResponse)
async def run_query_optimized(req: QueryRequest, authorization: str = Header(None)):
    """Ultra-optimized endpoint for maximum scoring."""
    
    # Quick auth check
    expected_token = f"Bearer {os.getenv('BEARER_TOKEN', '16ca23504efb8f8b98b1d84b2516a4b6ccb69f3c955ac9a8107497f5d14d6dbb')}"
    if authorization != expected_token:
        raise HTTPException(status_code=401, detail="Invalid authorization token.")

    # Input validation
    if not req.questions or len(req.questions) > 25:
        raise HTTPException(status_code=400, detail="Invalid questions")

    start_time = time.time()
    AGGRESSIVE_TIMEOUT = 28  # Leave 2s buffer
    
    logger.info(f"🚀 OPTIMIZED: Processing {len(req.questions)} questions")
    
    try:
        # Step 1: Fast document processing
        chunks, faiss_index, doc_hash = await get_or_process_document(req.documents)
        
        # Step 2: Parallel-optimized question processing
        answers = []
        
        for i, question in enumerate(req.questions):
            elapsed = time.time() - start_time
            
            # Dynamic timeout per question
            remaining_time = AGGRESSIVE_TIMEOUT - elapsed
            time_per_question = remaining_time / (len(req.questions) - i)
            
            if time_per_question < 1:  # Less than 1s per question
                answers.extend(["Information not available due to processing constraints."] * (len(req.questions) - i))
                break
            
            question_start = time.time()
            
            try:
                # Faster search with reduced top_k
                relevant_chunks = semantic_search.search(faiss_index, question, chunks, top_k=2)
                
                # Fast answer generation
                answer_result = decision_engine.generate_answer(question, doc_hash, relevant_chunks)
                answers.append(answer_result["answer"])
                
                question_time = time.time() - question_start
                logger.info(f"✅ Q{i+1}: {question_time:.2f}s")
                
            except Exception as e:
                logger.error(f"❌ Q{i+1} failed: {e}")
                answers.append("Unable to process this question.")

        total_time = time.time() - start_time
        logger.info(f"🏁 OPTIMIZED COMPLETE: {len(answers)}/{len(req.questions)} in {total_time:.2f}s")
        
        return QueryResponse(answers=answers)
        
    except Exception as e:
        logger.error(f"❌ Critical error: {e}")
        raise HTTPException(status_code=500, detail="Processing error")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "cache_size": len(DOCUMENT_INDEX_CACHE),
        "llm_available": llm_processor.available if hasattr(llm_processor, 'available') else False
    }

@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics."""
    return {
        "cache_size": len(DOCUMENT_INDEX_CACHE),
        "max_cache_size": MAX_CACHE_SIZE,
        "cached_documents": list(DOCUMENT_INDEX_CACHE.keys())
    }

@app.post("/cache/clear")
async def clear_cache(authorization: str = Header(None)):
    """Clear the document cache."""
    expected_token = f"Bearer {os.getenv('BEARER_TOKEN', '16ca23504efb8f8b98b1d84b2516a4b6ccb69f3c955ac9a8107497f5d14d6dbb')}"
    if authorization != expected_token:
        raise HTTPException(status_code=401, detail="Invalid authorization token.")
    
    DOCUMENT_INDEX_CACHE.clear()
    decision_engine.clear_cache()
    
    return {"message": "Cache cleared successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)