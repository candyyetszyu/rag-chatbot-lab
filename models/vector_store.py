"""Vector Store for document embeddings using FAISS."""

import os
import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
import faiss

logger = logging.getLogger(__name__)


class VectorStore:
    """FAISS-based vector store for document retrieval."""

    def __init__(self, embedding_dim: int = 384):
        """Initialize vector store."""
        self.embedding_dim = embedding_dim
        self.index = None
        self.metadata: List[Dict] = []
        self._initialize_index()

    def _initialize_index(self):
        """Initialize FAISS index."""
        try:
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            logger.info(f"FAISS index initialized with dimension {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to initialize FAISS index: {e}")
            self.index = None

    def add_texts(self, texts: List[str], embeddings: np.ndarray, metadata: List[Dict] = None):
        """Add texts with their embeddings to the store."""
        if self.index is None:
            raise RuntimeError("Vector store not initialized")

        if len(texts) != len(embeddings):
            raise ValueError(f"Number of texts ({len(texts)}) must match embeddings ({len(embeddings)})")

        self.index.add(embeddings.astype('float32'))
        
        for i, text in enumerate(texts):
            meta = metadata[i] if metadata else {}
            meta['text'] = text
            self.metadata.append(meta)

        logger.info(f"Added {len(texts)} texts to vector store (total: {self.index.ntotal})")

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Search for similar texts."""
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Vector store is empty")
            return []

        try:
            distances, indices = self.index.search(
                query_embedding.reshape(1, -1).astype('float32'), 
                min(k, self.index.ntotal)
            )

            results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.metadata):
                    result = self.metadata[idx].copy()
                    result['score'] = float(dist)
                    results.append(result)

            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def save(self, index_path: str, metadata_path: str):
        """Save index and metadata to disk."""
        if self.index is None:
            raise RuntimeError("Nothing to save")

        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        faiss.write_index(self.index, index_path)
        
        import json
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved index to {index_path}")

    def build_index(self, embeddings: np.ndarray, metadata: List[Dict]):
        """Build index from embeddings and metadata."""
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(embeddings.astype('float32'))
        self.metadata = metadata
        logger.info(f"Built index with {self.count} vectors")

    def load(self, index_path: str, metadata_path: str):
        """Load index and metadata from disk."""
        if not os.path.exists(index_path):
            logger.warning(f"Index file not found: {index_path}")
            return False

        try:
            self.index = faiss.read_index(index_path)
            
            import json
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)

            logger.info(f"Loaded index with {self.index.ntotal} vectors")
            return True

        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False

    @property
    def count(self) -> int:
        """Return number of vectors in store."""
        return self.index.ntotal if self.index else 0
