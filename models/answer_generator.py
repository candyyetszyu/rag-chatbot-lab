"""Answer generator with RAG support."""

import os
import logging
from typing import List, Dict, Optional, Tuple
from enum import Enum
from config import config
from .llm_providers import (
    LLMConfig, 
    LLMProvider, 
    HuggingFaceProvider, 
    OpenAIProvider, 
    AnthropicProvider, 
    DeepSeekProvider,
    GrokProvider,
    KimiProvider,
    MinimaxProvider,
    QwenProvider,
    GLMProvider,
    create_llm_provider
)
from .retrieval_system import RetrievalSystem
from .vector_store import VectorStore
from utils.embedding_generator import EmbeddingGenerator

logger = logging.getLogger(__name__)


class InteractionMode(Enum):
    """Interaction modes for the chatbot."""
    GENERAL = "general"
    TEXT_SPECIFIC = "text_specific"


class AnswerGenerator:
    """Answer generator with retrieval and multiple modes."""

    def __init__(self, config, vector_store: VectorStore = None, embedding_generator: EmbeddingGenerator = None):
        """Initialize the enhanced answer generator with shared components."""
        self.config = config
        
        # Use shared instances if provided, otherwise create new ones
        if embedding_generator is not None:
            self.embedding_generator = embedding_generator
        else:
            self.embedding_generator = EmbeddingGenerator()
        
        if vector_store is not None:
            self.vector_store = vector_store
        else:
            self.vector_store = VectorStore(
                embedding_dim=self.embedding_generator.get_embedding_dim()
            )
        
        # Try to load existing embeddings from disk
        self._load_existing_embeddings()
        
        self.retrieval_system = RetrievalSystem(
            vector_store=self.vector_store,
            embedding_generator=self.embedding_generator
        )
        
        hf_config = LLMConfig(
            provider=LLMProvider.HUGGINGFACE,
            model_name=config.llm_providers.huggingface["model_name"],
            temperature=config.llm_providers.huggingface["temperature"],
            max_tokens=config.llm_providers.huggingface["max_tokens"],
            top_p=config.llm_providers.huggingface["top_p"]
        )
        
        self.llm_provider = HuggingFaceProvider(hf_config)
        self._current_provider_name = "huggingface"
        self.rag_enabled = True

        logger.info(f"AnswerGenerator initialized with {config.llm_providers.huggingface['model_name']}")
        logger.info(f"Vector store has {self.vector_store.count} embeddings loaded")

    def get_provider_name(self) -> str:
        """Get the LLM provider name"""
        return self._current_provider_name

    def _load_existing_embeddings(self):
        """Load existing embeddings from disk if available"""
        try:
            embeddings_path = config.paths.get_absolute_path(config.paths.vector_store_path)
            index_path = embeddings_path / "faiss_index.idx"
            metadata_path = embeddings_path / "faiss_metadata.json"
            
            # Try to load FAISS index first
            if index_path.exists() and metadata_path.exists():
                success = self.vector_store.load(str(index_path), str(metadata_path))
                if success:
                    logger.info(f"Loaded {self.vector_store.count} embeddings from FAISS index")
                    return
            
            # Fall back to loading from pickle files
            if self.embedding_generator.embeddings_exist():
                logger.info("Loading embeddings from pickle files...")
                embeddings, metadata = self.embedding_generator.load_embeddings()
                
                # Build FAISS index from loaded embeddings
                self.vector_store.build_index(embeddings, metadata)
                logger.info(f"Built FAISS index with {self.vector_store.count} embeddings")
            else:
                logger.warning("No existing embeddings found. Vector store is empty.")
                
        except Exception as e:
            logger.error(f"Error loading existing embeddings: {str(e)}")
            logger.warning("Vector store will be empty until embeddings are generated.")

    def set_llm_provider(self, provider_name: str) -> bool:
        """Switch LLM provider using the factory pattern"""
        try:
            provider = provider_name.lower()
            
            # Map provider names to enum values and config keys
            provider_mapping = {
                "huggingface": (LLMProvider.HUGGINGFACE, "huggingface", None),
                "openai": (LLMProvider.OPENAI, "openai", "OPENAI_API_KEY"),
                "anthropic": (LLMProvider.ANTHROPIC, "anthropic", "ANTHROPIC_API_KEY"),
                "deepseek": (LLMProvider.DEEPSEEK, "deepseek", "DEEPSEEK_API_KEY"),
                "grok": (LLMProvider.GROK, "grok", "GROK_API_KEY"),
                "kimi": (LLMProvider.KIMI, "kimi", "KIMI_API_KEY"),
                "minimax": (LLMProvider.MINIMAX, "minimax", "MINIMAX_API_KEY"),
                "qwen": (LLMProvider.QWEN, "qwen", "QWEN_API_KEY"),
                "glm": (LLMProvider.GLM, "glm", "GLM_API_KEY"),
            }
            
            if provider not in provider_mapping:
                logger.error(f"Unknown provider: {provider_name}")
                return False
            
            provider_enum, config_key, api_key_env = provider_mapping[provider]
            
            # Check for API key if required (HuggingFace doesn't need explicit check)
            if api_key_env:
                api_key = os.getenv(api_key_env)
                if not api_key:
                    logger.error(f"{api_key_env} not configured")
                    return False
            
            # Get provider configuration
            provider_config = getattr(self.config.llm_providers, config_key)
            
            # Prepare additional parameters
            additional_params = {}
            if api_key_env:
                additional_params["api_key"] = os.getenv(api_key_env)
            
            # Add base_url if it exists in config
            if "base_url" in provider_config:
                additional_params["base_url"] = provider_config["base_url"]
            
            # Create provider configuration
            llm_config = LLMConfig(
                provider=provider_enum,
                model_name=provider_config["model_name"],
                temperature=provider_config["temperature"],
                max_tokens=provider_config["max_tokens"],
                top_p=provider_config["top_p"],
                additional_params=additional_params
            )
            
            # Use factory to create provider
            self.llm_provider = create_llm_provider(llm_config)
            self._current_provider_name = provider
            
            logger.info(f"Switched to {provider_name} provider")
            return True
                
        except Exception as e:
            logger.error(f"Error switching provider: {str(e)}")
            return False

    def check_model_health(self) -> Dict:
        """Check if all components are healthy"""
        return {
            "status": "healthy",
            "embedding_model": "loaded",
            "llm_provider": "initialized" if self.llm_provider.client else "error",
            "vector_store": "ready" if self.vector_store.count > 0 else "empty"
        }

    def set_rag_enabled(self, enabled: bool):
        """Enable or disable RAG"""
        self.rag_enabled = enabled

    def generate_educational_response(
        self,
        user_message: str,
        retrieval_context: str,
        text_works: list = None,
        mode: InteractionMode = None,
        conversation_history: Optional[List[Dict]] = None,
    ) -> Dict:
        """Generate an educational response with optional multi-turn conversation history."""
        try:
            system_prompt = config.modes.general_mode_prompt if mode == InteractionMode.GENERAL else config.modes.text_specific_mode_prompt
            
            response = self.llm_provider.generate_response(
                system_prompt=system_prompt,
                user_prompt=user_message,
                context=retrieval_context,
                conversation_history=conversation_history,
            )
            
            return {
                "response": response.content,
                "context_used": bool(retrieval_context),
                "sources": self._format_sources(retrieval_context),
                "provider_used": self._current_provider_name,
                "model_used": response.model,
                "metadata": response.metadata
            }

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "response": "I apologize, but I encountered an error. Please try again.",
                "context_used": False,
                "sources": [],
                "provider_used": self._current_provider_name,
                "metadata": {"error": str(e)}
            }

    def _format_sources(self, context: str) -> List[str]:
        """Extract source names from context."""
        sources = []
        for line in context.split('\n'):
            if line.startswith('[Source:'):
                source = line.replace('[Source:', '').replace(']', '').strip()
                if source and source not in sources:
                    sources.append(source)
        return sources
