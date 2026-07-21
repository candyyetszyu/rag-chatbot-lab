"""
Embedding Generation Utility
Generates and manages text embeddings for the RAG system.
"""

import logging
import os
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from sentence_transformers import SentenceTransformer
import torch

from config import config
from services.document_processor import TextChunk, DocumentProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Handles text embedding generation and management"""

    def __init__(self):
        self.model_name = config.model.embedding_model
        self.device = config.model.device
        self.embeddings_path = config.paths.get_absolute_path(config.paths.vector_store_path)
        # batch_size kept constant — the lab doesn't expose runtime tuning knobs
        self.batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

        # Ensure embeddings directory exists
        self.embeddings_path.mkdir(parents=True, exist_ok=True)

        # Initialize model
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            logger.info("📥 This may take a few minutes on first run (downloading model files)...")

            # Load model with explicit progress
            import os
            from pathlib import Path

            # Check if model is already cached
            cache_dir = Path.home() / ".cache" / "torch" / "sentence_transformers" / self.model_name.replace("/", "_")
            if cache_dir.exists():
                logger.info("✅ Using cached model files")
            else:
                logger.info("⬇️  Downloading model files (this may take several minutes)...")

            # Set environment variables to prevent segmentation faults
            os.environ['OMP_NUM_THREADS'] = '1'
            os.environ['MKL_NUM_THREADS'] = '1'
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU only
            os.environ['OMP_MAX_ACTIVE_LEVELS'] = '1'

            # Force CPU usage to avoid segfaults
            self.device = "cpu"

            # Load with explicit device specification
            self.model = SentenceTransformer(self.model_name, device="cpu")

            # Ensure model is on CPU
            self.model = self.model.to("cpu")
            logger.info("Using CPU for embeddings (safe mode)")

            logger.info("✅ Embedding model loaded successfully")

        except Exception as e:
            logger.error(f"❌ Error loading embedding model: {str(e)}")
            raise

    def get_embedding_dim(self) -> int:
        """Get the embedding dimension"""
        if self.model:
            return self.model.get_sentence_embedding_dimension()
        return 384  # Default for all-MiniLM-L6-v2

    def encode(self, texts: list) -> np.ndarray:
        """Encode texts to embeddings (shorthand for generate_query_embedding with multiple texts)"""
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        return self.model.encode(texts, convert_to_numpy=True)

    def generate_embeddings_for_chunks(self, chunks: List[TextChunk]) -> np.ndarray:
        """Generate embeddings for a list of text chunks"""
        if not chunks:
            return np.array([])

        texts = [chunk.text for chunk in chunks]
        logger.info(f"Generating embeddings for {len(texts)} text chunks...")

        try:
            # Generate embeddings in batches
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )

            logger.info(f"Generated embeddings shape: {embeddings.shape}")
            return embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    def generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for a search query"""
        try:
            embedding = self.model.encode([query], convert_to_numpy=True)
            return embedding[0]  # Return single embedding, not batch

        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            raise

    def save_embeddings(self,
                       embeddings: np.ndarray,
                       chunks: List[TextChunk],
                       filename: str = "document_embeddings"):
        """Save embeddings and associated metadata to disk"""
        embeddings_file = self.embeddings_path / f"{filename}.pkl"
        metadata_file = self.embeddings_path / f"{filename}_metadata.pkl"

        try:
            # Prepare metadata
            metadata = []
            for chunk in chunks:
                metadata.append({
                    "text": chunk.text,
                    "source": chunk.source,
                    "page": chunk.page,
                    "work_id": chunk.work_id,
                    "author": chunk.author,
                    "title": chunk.title,
                    "chunk_index": chunk.chunk_index
                })

            # Save embeddings
            with open(embeddings_file, 'wb') as f:
                pickle.dump(embeddings, f)

            # Save metadata
            with open(metadata_file, 'wb') as f:
                pickle.dump(metadata, f)

            logger.info(f"Saved embeddings and metadata to {embeddings_file} and {metadata_file}")

        except Exception as e:
            logger.error(f"Error saving embeddings: {str(e)}")
            raise

    def load_embeddings(self, filename: str = "document_embeddings") -> Tuple[np.ndarray, List[Dict]]:
        """Load embeddings and metadata from disk"""
        embeddings_file = self.embeddings_path / f"{filename}.pkl"
        metadata_file = self.embeddings_path / f"{filename}_metadata.pkl"

        if not embeddings_file.exists() or not metadata_file.exists():
            raise FileNotFoundError(f"Embedding files not found: {embeddings_file}, {metadata_file}")

        try:
            # Load embeddings
            with open(embeddings_file, 'rb') as f:
                embeddings = pickle.load(f)

            # Load metadata
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)

            logger.info(f"Loaded embeddings shape: {embeddings.shape}, metadata count: {len(metadata)}")
            return embeddings, metadata

        except Exception as e:
            logger.error(f"Error loading embeddings: {str(e)}")
            raise

    def generate_embeddings_batch(self, queries: List[str], show_progress: bool = False) -> np.ndarray:
        """Generate embeddings for multiple queries in optimized batches
        
        This method is optimized for parallel processing of multiple queries,
        useful for batch retrieval or recommendation scenarios.
        
        Args:
            queries: List of query strings to embed
            show_progress: Whether to show progress bar
            
        Returns:
            numpy array of embeddings with shape (n_queries, embedding_dim)
        """
        if not queries:
            return np.array([])
        
        try:
            import time
            start_time = time.time()
            
            # Generate embeddings in a single batch (optimized by sentence-transformers)
            embeddings = self.model.encode(
                queries,
                batch_size=min(len(queries), self.batch_size * 2),  # Slightly larger batch for queries
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True  # Normalize for cosine similarity
            )
            
            elapsed = time.time() - start_time
            logger.info(f"Generated {len(queries)} embeddings in {elapsed:.2f}s ({(len(queries)/elapsed):.1f} queries/sec)")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error in batch embedding generation: {str(e)}")
            raise

    def embeddings_exist(self, filename: str = "document_embeddings") -> bool:
        """Check if embeddings files exist"""
        embeddings_file = self.embeddings_path / f"{filename}.pkl"
        metadata_file = self.embeddings_path / f"{filename}_metadata.pkl"
        return embeddings_file.exists() and metadata_file.exists()

    def generate_and_save_all_embeddings(self) -> None:
        """Generate and save embeddings for all processed documents"""
        logger.info("Starting embedding generation for all documents...")

        # Load all processed text chunks
        processor = DocumentProcessor()
        all_chunks = processor.get_all_chunks()

        if not all_chunks:
            logger.warning("No text chunks found. Make sure documents are processed first.")
            return

        logger.info(f"Found {len(all_chunks)} text chunks across all documents")

        # Generate embeddings
        embeddings = self.generate_embeddings_for_chunks(all_chunks)

        # Save embeddings
        self.save_embeddings(embeddings, all_chunks)

        logger.info("Embedding generation completed successfully")

    def update_embeddings_with_new_chunks(self, new_chunks: List[TextChunk]) -> None:
        """Add embeddings for new chunks to existing embeddings"""
        if not new_chunks:
            return

        # Generate new embeddings
        new_embeddings = self.generate_embeddings_for_chunks(new_chunks)

        # Load existing embeddings if they exist
        if self.embeddings_exist():
            existing_embeddings, existing_metadata = self.load_embeddings()

            # Combine embeddings and metadata
            combined_embeddings = np.vstack([existing_embeddings, new_embeddings])

            combined_metadata = existing_metadata.copy()
            for chunk in new_chunks:
                combined_metadata.append({
                    "text": chunk.text,
                    "source": chunk.source,
                    "page": chunk.page,
                    "work_id": chunk.work_id,
                    "author": chunk.author,
                    "title": chunk.title,
                    "chunk_index": chunk.chunk_index
                })

            # Save combined embeddings
            self.save_embeddings(combined_embeddings, [TextChunk(**meta) for meta in combined_metadata])

        else:
            # No existing embeddings, just save the new ones
            self.save_embeddings(new_embeddings, new_chunks)

        logger.info(f"Updated embeddings with {len(new_chunks)} new chunks")

    def get_embedding_stats(self) -> Dict:
        """Get statistics about the current embeddings"""
        if not self.embeddings_exist():
            return {"error": "No embeddings found"}

        try:
            embeddings, metadata = self.load_embeddings()

            # Count chunks by work
            work_counts = {}
            for meta in metadata:
                work_id = meta.get("work_id")
                if work_id:
                    work_counts[work_id] = work_counts.get(work_id, 0) + 1

            # Count chunks by author
            author_counts = {}
            for meta in metadata:
                author = meta.get("author")
                if author:
                    author_counts[author] = author_counts.get(author, 0) + 1

            return {
                "total_chunks": len(metadata),
                "embedding_dimension": embeddings.shape[1],
                "works_count": len(work_counts),
                "authors_count": len(author_counts),
                "work_distribution": work_counts,
                "author_distribution": author_counts,
                "embedding_model": self.model_name
            }

        except Exception as e:
            return {"error": f"Error getting embedding stats: {str(e)}"}


def main():
    """Main function for testing embedding generation"""
    generator = EmbeddingGenerator()

    # Check if embeddings already exist
    if generator.embeddings_exist():
        print("Embeddings already exist. Getting stats...")
        stats = generator.get_embedding_stats()
        print(f"Embedding Statistics:")
        print(f"- Total chunks: {stats.get('total_chunks', 'N/A')}")
        print(f"- Embedding dimension: {stats.get('embedding_dimension', 'N/A')}")
        print(f"- Works count: {stats.get('works_count', 'N/A')}")
        print(f"- Authors count: {stats.get('authors_count', 'N/A')}")
        print(f"- Model: {stats.get('embedding_model', 'N/A')}")
    else:
        print("No embeddings found. Generating new embeddings...")
        generator.generate_and_save_all_embeddings()

    # Test query embedding
    test_query = "What is the significance of the kitchen in this story?"
    query_embedding = generator.generate_query_embedding(test_query)
    print(f"\nTest query embedding shape: {query_embedding.shape}")

if __name__ == "__main__":
    main()