"""Retrieval system for document search."""

import logging
from typing import List, Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)


class RetrievalSystem:
    """Document retrieval system using vector similarity."""

    def __init__(self, vector_store, embedding_generator):
        """Initialize retrieval system."""
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator

    def retrieve(self, query: str, top_k: int = 5, threshold: float = 0.4) -> List[Dict]:
        """Retrieve relevant documents for a query."""
        try:
            query_embedding = self.embedding_generator.encode([query])
            
            results = self.vector_store.search(query_embedding, k=top_k)
            
            # DEBUG: Log all scores before filtering
            if results:
                logger.debug(f"Raw retrieval scores: {[r.get('score', 'N/A') for r in results]}")
            
            filtered = [r for r in results if r.get('score', 1.0) < threshold]
            
            return filtered if filtered else results[:top_k]

        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []

    def retrieve_context(self, query: str, top_k: int = 5, threshold: float = 0.4, 
                         work_filter: Optional[str] = None, author_filter: Optional[str] = None) -> str:
        """Retrieve relevant context as formatted string."""
        
        # Apply filters BEFORE retrieval by using higher top_k
        # FAISS doesn't support metadata filtering, so we retrieve more and filter
        effective_top_k = top_k * 3  # Retrieve 3x more to account for filtering
        
        results = self.retrieve(query, top_k=effective_top_k, threshold=threshold)
        
        # Filter by work_id (not source/folder name)
        if work_filter:
            if isinstance(work_filter, list):
                # Filter: keep results where work_id matches ANY of the filters
                results = [r for r in results if r.get('work_id') in work_filter]
            else:
                results = [r for r in results if r.get('work_id') == work_filter]
        
        # Filter by author
        if author_filter:
            if isinstance(author_filter, list):
                results = [r for r in results if r.get('author') in author_filter]
            else:
                results = [r for r in results if r.get('author') == author_filter]
        
        # Re-sort by score and limit to top_k
        results.sort(key=lambda x: x.get('score', float('inf')))
        results = results[:top_k]
        
        if not results:
            return ""
        
        context_parts = []
        for r in results:
            text = r.get('text', '')
            source = r.get('source', '')
            work_id = r.get('work_id', '')
            if text:
                # Include work_id in source for clarity
                display_source = f"{work_id}-{source.split('-', 1)[-1]}" if work_id else source
                context_parts.append(f"[Source: {display_source}]\n{text}")
        
        return "\n\n".join(context_parts)
