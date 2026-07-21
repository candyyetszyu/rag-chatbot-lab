"""
System Setup Script
Initializes the RAG Chatbot Lab system by processing documents and building embeddings.
"""

# Set environment variables to prevent segmentation faults
import os
import platform

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_DISABLE_CUDNN'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

if platform.system() == "Darwin" and platform.machine() == "arm64":
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    thread_limit = "4" if cpu_count >= 8 else "2"
    os.environ["OMP_NUM_THREADS"] = thread_limit
    os.environ["MKL_NUM_THREADS"] = thread_limit
    os.environ["OPENBLAS_NUM_THREADS"] = thread_limit
else:
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"

import logging
import sys
import time
import gc
from pathlib import Path

# Add the current directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from config import config, ensure_directories

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_faiss():
    """Test if FAISS is working properly"""
    try:
        import numpy as np
        import faiss

        # Create small test to verify FAISS works
        test_data = np.random.random((10, 5)).astype('float32')
        index = faiss.IndexFlatL2(5)
        index.add(test_data)

        # Test search
        query = np.random.random((1, 5)).astype('float32')
        D, I = index.search(query, 2)

        logger.info("✅ FAISS test passed")
        return True

    except Exception as e:
        logger.error(f"❌ FAISS test failed: {e}")
        return False

def test_model_loading():
    """Test if sentence transformers can load without segfault"""
    try:
        import os

        # Set PyTorch environment variables to prevent segfaults on macOS
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        from sentence_transformers import SentenceTransformer
        import torch

        # Test with a minimal model load
        logger.info("Testing model loading...")
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # Test a simple encoding
        test_text = ["This is a test sentence."]
        embeddings = model.encode(test_text, convert_to_numpy=True)

        logger.info("✅ Model loading test passed")
        return True

    except Exception as e:
        logger.error(f"❌ Model loading test failed: {e}")
        return False

def fix_faiss_if_needed():
    """Try to fix common FAISS issues"""
    if not test_faiss():
        logger.warning("FAISS issues detected. Attempting fix...")

        # On macOS, try to reinstall a compatible version
        if platform.system() == "Darwin":
            try:
                import subprocess
                logger.info("Trying FAISS version 1.7.3...")

                subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "faiss-cpu"],
                             capture_output=True)
                result = subprocess.run([sys.executable, "-m", "pip", "install", "faiss-cpu==1.7.3"],
                                      capture_output=True, text=True)

                if result.returncode == 0 and test_faiss():
                    logger.info("✅ FAISS fixed with version 1.7.3")
                    return True

            except Exception as e:
                logger.error(f"Auto-fix failed: {e}")

        return False
    return True

def setup_system():
    """Main setup function with robust error handling"""
    start_time = time.time()

    try:
        logger.info("Starting RAG Chatbot Lab system setup...")

        # 1. Set safe environment variables for macOS
        import os
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        logger.info("Set PyTorch environment variables for macOS compatibility")

        # 2. Test model loading first (this was causing the segfault)
        logger.info("Testing model loading compatibility...")
        if not test_model_loading():
            logger.error("Model loading test failed. Cannot proceed with setup.")
            return setup_with_basic_functionality(start_time)

        # 3. Test and fix FAISS
        logger.info("Testing FAISS compatibility...")
        if not fix_faiss_if_needed():
            logger.error("FAISS issues could not be resolved automatically.")
            logger.info("Manual fix: pip uninstall faiss-cpu && pip install faiss-cpu==1.7.3")
            return setup_with_basic_functionality(start_time)

        # 4. Ensure directories exist
        logger.info("Creating necessary directories...")
        ensure_directories()

        # 5. Process documents
        logger.info("Processing OCR documents...")
        from services.document_processor import DocumentProcessor

        processor = DocumentProcessor()
        processed_docs = processor.process_all_documents()

        if not processed_docs:
            logger.warning("No documents were processed. Check OCR data path.")
            logger.info("Continuing with basic setup...")

        logger.info(f"Successfully processed {len(processed_docs)} documents")

        # 6. Generate embeddings with memory management
        logger.info("Generating embeddings for all text chunks...")
        from utils.embedding_generator import EmbeddingGenerator

        embedding_generator = EmbeddingGenerator()

        if embedding_generator.embeddings_exist():
            logger.info("Embeddings already exist. Checking if update is needed...")
            stats = embedding_generator.get_embedding_stats()
            logger.info(f"Current embeddings: {stats.get('total_chunks', 0)} chunks")

            # Skip regeneration if embeddings exist and documents weren't processed
            if not processed_docs:
                logger.info("Using existing embeddings")
            else:
                logger.info("Regenerating embeddings...")
                embedding_generator.generate_and_save_all_embeddings()
        else:
            embedding_generator.generate_and_save_all_embeddings()

        # Clear memory before FAISS operations
        gc.collect()

        # 7. Build vector store with enhanced error handling
        logger.info("Building FAISS vector index...")
        return build_vector_store_safe(start_time)

    except Exception as e:
        logger.error(f"System setup failed: {str(e)}")
        return False

def build_vector_store_safe(start_time):
    """Build vector store with safety measures, and actually persist it."""
    try:
        from models.vector_store import VectorStore
        from utils.embedding_generator import EmbeddingGenerator

        embedding_generator = EmbeddingGenerator()
        if not embedding_generator.embeddings_exist():
            logger.warning("No embeddings on disk to build a vector store from.")
            return setup_with_basic_functionality(start_time)

        embeddings, metadata = embedding_generator.load_embeddings()

        vector_store = VectorStore(embedding_dim=embedding_generator.get_embedding_dim())
        logger.info("Building FAISS index from embeddings...")
        try:
            vector_store.build_index(embeddings, metadata)
        except Exception as faiss_error:
            logger.error(f"FAISS index building failed: {faiss_error}")
            return setup_with_basic_functionality(start_time)

        if vector_store.count == 0:
            logger.error("Vector store is empty after build. Setup may have failed.")
            return setup_with_basic_functionality(start_time)

        index_path = config.paths.get_absolute_path(config.paths.vector_store_path) / "faiss_index.idx"
        metadata_path = config.paths.get_absolute_path(config.paths.vector_store_path) / "faiss_metadata.json"
        vector_store.save(str(index_path), str(metadata_path))

        logger.info(f"Vector store contains {vector_store.count} vectors")
        logger.info(f"Saved FAISS index to {index_path}")

        setup_time = time.time() - start_time
        logger.info(f"System setup completed successfully in {setup_time:.2f} seconds")

        print_success_message(True)
        return True

    except Exception as e:
        logger.error(f"Vector store setup failed: {str(e)}")
        return setup_with_basic_functionality(start_time)

def setup_with_basic_functionality(start_time):
    """Setup basic functionality when vector store fails."""
    try:
        logger.info("Setting up basic functionality without vector search...")

        from services.document_processor import DocumentProcessor
        processor = DocumentProcessor()
        logger.info("✅ Document processor available")

        setup_time = time.time() - start_time
        logger.info(f"Basic setup completed in {setup_time:.2f} seconds")

        print_success_message(False)
        return True

    except Exception as e:
        logger.error(f"Basic setup failed: {str(e)}")
        return False

def print_success_message(full_setup=True):
    """Print setup completion message."""
    print("\n" + "=" * 60)
    if full_setup:
        print("🎉 RAG Chatbot Lab setup complete!")
        print("✅ All features available:")
        print("   - Document processing")
        print("   - Vector search and embeddings")
        print("   - Context-aware responses")
    else:
        print("⚠️  RAG Chatbot Lab basic setup complete!")
        print("✅ Available features:")
        print("   - Document processing")
        print("   - Basic chatbot functionality")
        print("❌ Limited features:")
        print("   - No vector search")
        print("   - No context-aware responses")

    print("\n🧪 Verify the system:")
    print("   python tests/quick_test.py  # Quick end-to-end check")

    print("\n💬 Start chatting:")
    print("   python -m commands.cli      # Interactive menu-driven CLI")
    print("   python -m commands.doctor   # Troubleshoot a specific setup step")

    if not full_setup:
        print("\n🔧 To enable full features:")
        print("   pip uninstall faiss-cpu")
        print("   pip install faiss-cpu==1.7.3")
        print("   python -m scripts.setup_system")

    print("="*60)

def main():
    """Main function."""
    print("RAG Chatbot Lab System Setup")
    print("=" * 50)

    success = setup_system()

    if success:
        print("\n✅ System setup completed successfully!")
        print("\nYou can now chat with:")
        print("python -m commands.cli")
    else:
        print("\n❌ System setup failed!")
        print("Check the logs above for error details.")
        sys.exit(1)

if __name__ == "__main__":
    main()