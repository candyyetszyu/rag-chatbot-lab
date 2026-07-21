"""
Chatbot Service
Orchestrates the RAG pipeline with dual-mode operation and multiple LLM
providers.
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from config import config
from services.conversation_logger import ConversationLogger
from models.vector_store import VectorStore
from models.retrieval_system import RetrievalSystem
from models.answer_generator import AnswerGenerator, InteractionMode
from services.session_service import SessionService, ChatSession

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChatResponse:
    """Response payload returned by the enhanced chatbot service."""
    message: str
    session_id: str
    context_used: bool = False
    sources: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    interaction_mode: str = "general"
    provider_used: Optional[str] = None
    model_used: Optional[str] = None

class ChatbotService:
    """Chatbot service with dual-mode and multi-LLM support"""

    def __init__(self):
        self.vector_store = VectorStore()

        self.answer_generator = AnswerGenerator(
            config,
            vector_store=self.vector_store,
        )

        self.retrieval_system = RetrievalSystem(
            vector_store=self.vector_store,
            embedding_generator=self.answer_generator.embedding_generator,
        )

        self.conversation_logger = ConversationLogger()
        self.session_service = SessionService()
        self.active_sessions = self.session_service.active_sessions
        self.session_timeout = self.session_service.session_timeout

        self.response_cache: Dict[str, Tuple[Any, datetime]] = {}
        self.cache_max_size = 1000
        self.cache_ttl = timedelta(hours=1)

        self._initialize_system()

    def _get_cached_response(self, cache_key: str) -> Optional[Any]:
        """Get cached response if valid"""
        if cache_key in self.response_cache:
            response, timestamp = self.response_cache[cache_key]
            if datetime.now() - timestamp < self.cache_ttl:
                logger.debug(f"Cache hit for key: {cache_key}")
                return response
            else:
                del self.response_cache[cache_key]
        return None

    def _cache_response(self, cache_key: str, response: Any):
        """Cache a response"""
        # Simple cache eviction when full
        if len(self.response_cache) >= self.cache_max_size:
            # Remove oldest entry
            oldest_key = min(self.response_cache.keys(),
                            key=lambda k: self.response_cache[k][1])
            del self.response_cache[oldest_key]

        self.response_cache[cache_key] = (response, datetime.now())
        logger.debug(f"Cached response for key: {cache_key}")

    def _get_cache_key(self, message: str, session) -> str:
        """Build a cache key from message, mode, and focused texts."""
        focused = "|".join(session.focused_texts or [])
        return f"{session.interaction_mode.value}:{focused}:{message}"

    def _initialize_system(self):
        """Initialize the enhanced chatbot system"""
        try:
            logger.info("Initializing enhanced chatbot system...")

            # Load existing vector index if it exists
            index_path = config.paths.get_absolute_path(config.paths.vector_store_path) / "faiss_index.idx"
            metadata_path = config.paths.get_absolute_path(config.paths.vector_store_path) / "faiss_metadata.json"

            if index_path.exists():
                logger.info(f"Loading vector index from {index_path}")
                loaded = self.vector_store.load(str(index_path), str(metadata_path))
                if loaded:
                    logger.info(f"Loaded vector store with {self.vector_store.count} vectors")
                else:
                    logger.warning("Failed to load vector store, will use empty vector store")
            else:
                logger.info("No existing vector index found, will use empty vector store")
                logger.info("Embeddings will be generated on first request if needed")

            # Check answer generator health
            try:
                health = self.answer_generator.check_model_health()
                if health["status"] != "healthy":
                    logger.warning(f"Answer generator not healthy: {health}")
            except Exception as e:
                logger.warning(f"Could not check answer generator health: {str(e)}")

            logger.info("Chatbot system initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing enhanced chatbot system: {str(e)}")
            logger.warning("System will continue with limited functionality")
            # Don't raise - allow the service to initialize even if vector store fails

    def start_chat_session(self, user_id: str = None, mode: str = "general") -> Tuple[str, Dict[str, Any]]:
        """Start a new chat session with the specified interaction mode.

        Returns ``(session_id, metadata)``. ``metadata`` is the same dict the
        underlying ``SessionService`` produced plus a ``rag_enabled`` flag so
        callers can render an initial state for the client.
        """
        session_id, metadata = self.session_service.create_session(user_id, mode)
        session = self.session_service.get_session(session_id)
        self.conversation_logger.start_session(session_id, user_id, {
            "interaction_mode": session.interaction_mode.value,
            "rag_enabled": session.rag_enabled,
        })
        logger.info(
            f"Started new enhanced chat session: {session_id} (mode: {session.interaction_mode.value})"
        )
        return session_id, metadata

    def set_session_mode(self, session_id: str, mode: str) -> bool:
        """Switch session interaction mode"""
        # Get current mode before changing
        session = self.session_service.get_session(session_id)
        if not session:
            logger.error(f"Session {session_id} not found")
            return False
        
        old_mode = session.interaction_mode
        
        # Delegate to SessionService
        result = self.session_service.set_session_mode(session_id, mode)
        
        if result:
            # Get updated session
            session = self.session_service.get_session(session_id)
            
            # Clear cache when mode changes to ensure fresh responses
            if old_mode != session.interaction_mode:
                self.response_cache.clear()
                logger.debug(f"Cache cleared for session {session_id} due to mode change")

            logger.info(f"Session {session_id} mode changed from {old_mode.value} to {session.interaction_mode.value}")

            # Log mode change
            session.conversation_history.append({
                "type": "system",
                "message": f"Mode changed to {session.interaction_mode.value}",
                "timestamp": datetime.now().isoformat(),
                "old_mode": old_mode.value
            })

        return result

    def set_session_provider(self, session_id: str, provider: str) -> bool:
        """Set preferred LLM provider for a session"""
        # Check if provider is available
        available_providers = self.get_llm_providers()
        provider_info = next((p for p in available_providers if p["id"] == provider), None)

        if not provider_info or not provider_info.get("available", False):
            logger.error(f"Provider {provider} not available or not configured")
            return False

        # Delegate to SessionService
        result = self.session_service.set_session_provider(session_id, provider)
        
        if result:
            # Actually switch the provider in answer_generator
            try:
                success = self.answer_generator.set_llm_provider(provider)
                if success:
                    logger.info(f"Session {session_id} provider set to {provider}")
                    return True
                else:
                    logger.error(f"Failed to switch to provider {provider}")
                    return False
            except Exception as e:
                logger.error(f"Error switching provider: {str(e)}")
                return False
        
        return False

    def set_session_rag(self, session_id: str, rag_enabled: bool) -> bool:
        """Toggle RAG for a session"""
        # Delegate to SessionService
        result = self.session_service.set_session_rag(session_id, rag_enabled)
        
        if result:
            logger.info(f"Session {session_id} RAG {'enabled' if rag_enabled else 'disabled'}")
        
        return result

    def process_message(self,
                       message: str,
                       session_id: str,
                       work_filter: Optional[List[str]] = None,
                       author_filter: Optional[List[str]] = None,
                       override_mode: Optional[str] = None) -> ChatResponse:
        """Process a user message with enhanced features"""

        # Get or create session
        session = self._get_or_create_session(session_id)
        session.last_activity = datetime.now()

        # Check cache for identical queries
        cache_key = self._get_cache_key(message, session)
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            logger.info(f"Returning cached response for: {message[:50]}...")
            return cached_response

        try:
            # Determine interaction mode
            if override_mode:
                try:
                    interaction_mode = InteractionMode(override_mode)
                except ValueError:
                    interaction_mode = session.interaction_mode
            else:
                interaction_mode = session.interaction_mode

            # Apply session preferences
            if session.preferred_provider:
                self.answer_generator.set_llm_provider(session.preferred_provider)

            # Set RAG enabled/disabled based on session setting
            self.answer_generator.set_rag_enabled(session.rag_enabled)

            # Add user message to history
            session.conversation_history.append({
                "type": "user",
                "message": message,
                "timestamp": datetime.now().isoformat(),
                "mode": interaction_mode.value
            })

            # Retrieve relevant context only if RAG is enabled
            if session.rag_enabled:
                retrieval_context = self.retrieval_system.retrieve_context(
                    message,
                    work_filter=work_filter,
                    author_filter=author_filter
                )
            else:
                retrieval_context = ""

            # Determine text works for context
            text_works = work_filter or session.focused_texts

            # When no RAG context found, prepare minimal context with work metadata
            # This lets the LLM handle all cases through its system prompt
            context_to_use = retrieval_context
            if not context_to_use and text_works:
                from config import get_document_by_id
                work_context_parts = []
                for work_id in text_works:
                    work = get_document_by_id(work_id)
                    if work:
                        work_context_parts.append(
                            f"[Document: {work.get('title', work_id)} by {work.get('author', 'Unknown')} ({work.get('year', '')})]"
                        )
                if work_context_parts:
                    context_to_use = " ".join(work_context_parts)

            # Build prior-turn history for multi-turn context (exclude the just-appended user message)
            prior_history = session.conversation_history[:-1]

            # Always use LLM - system prompt handles off-topic redirection
            response_data = self.answer_generator.generate_educational_response(
                user_message=message,
                retrieval_context=context_to_use or "",
                text_works=text_works,
                mode=interaction_mode,
                conversation_history=prior_history if prior_history else None,
            )

            response_text = response_data["response"]
            context_used = response_data["context_used"]

            # Add response to history
            session.conversation_history.append({
                "type": "assistant",
                "message": response_text,
                "timestamp": datetime.now().isoformat(),
                "context_used": context_used,
                "mode": interaction_mode.value,
                "provider": response_data.get("provider_used"),
                "model": response_data.get("model_used")
            })

            # Log the conversation turn
            self.conversation_logger.log_turn(
                session_id=session_id,
                user_message=message,
                ai_response=response_text,
                context_used=context_used,
                sources=response_data.get("sources", []),
                metadata={
                    **response_data.get("metadata", {}),
                    "work_filter": work_filter,
                    "author_filter": author_filter,
                    "interaction_mode": interaction_mode.value,
                    "provider_used": response_data.get("provider_used"),
                    "model_used": response_data.get("model_used")
                },
                response_time=0.0
            )

            response = ChatResponse(
                message=response_text,
                session_id=session_id,
                context_used=context_used,
                sources=response_data.get("sources", []),
                metadata={
                    **response_data.get("metadata", {}),
                },
                interaction_mode=interaction_mode.value,
                provider_used=response_data.get("provider_used"),
                model_used=response_data.get("model_used")
            )

            # Cache the response
            self._cache_response(cache_key, response)

            return response

        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")

            return ChatResponse(
                message=f"I apologize, but I encountered an error processing your message. Please try again or contact support if the issue persists.",
                session_id=session_id,
                context_used=False,
                metadata={"error": str(e), "interaction_mode": session.interaction_mode.value}
            )

    def process_message_for_texts(self,
                                  message: str,
                                  session_id: str,
                                  text_ids: List[str]) -> ChatResponse:
        """Process a user message with focus on specific texts (text-specific mode)"""

        # Get or create session
        session = self._get_or_create_session(session_id)

        # Set session to text-specific mode and update focused texts
        session.interaction_mode = InteractionMode.TEXT_SPECIFIC
        session.focused_texts = text_ids
        session.last_activity = datetime.now()

        # Map text IDs to work IDs for the retrieval system.
        from config import get_document_by_id
        work_filters = []
        for text_id in text_ids:
            work = get_document_by_id(text_id)
            if work:
                work_filters.append(work.get("id", text_id))
            else:
                logger.warning(f"Unknown document ID '{text_id}' — skipping from filter")
        if not work_filters:
            work_filters = list(text_ids)

        # Check if this is a summary-type query
        is_summary_query = self._is_summary_query(message)

        # For summary queries, try to get full text
        if is_summary_query:
            full_text_context = self._get_full_text_for_texts(text_ids)
            if full_text_context:
                logger.info(f"Using full text context for summary query ({len(full_text_context)} chars)")
                # Use full text as context instead of RAG
                return self._generate_with_full_text(
                    message=message,
                    session_id=session_id,
                    full_text_context=full_text_context,
                    text_ids=text_ids
                )

        return self.process_message(
            message=message,
            session_id=session_id,
            work_filter=work_filters,
            override_mode="text_specific"
        )

    def _is_summary_query(self, message: str) -> bool:
        """Detect if query is requesting a summary or overview"""
        summary_keywords = [
            "summarize", "summary", "summarise", "overview", "overall",
            "what happens", "what is the story", "plot summary",
            "main idea", "main points", "key points", "recap",
            "brief summary", "give me the gist", "what's this about",
            "tell me about the text", "describe the text", "explain the story"
        ]
        message_lower = message.lower().strip()
        return any(keyword in message_lower for keyword in summary_keywords)

    def _get_full_text_for_texts(self, text_ids: List[str]) -> str:
        """Get full text content from OCR files for the given texts"""
        import unicodedata
        from config import get_document_by_id, get_document_by_folder, config
        import json
        from services.document_processor import DocumentProcessor
        from pathlib import Path

        processor = DocumentProcessor()
        full_texts = []

        def normalize_for_comparison(name: str) -> str:
            """Normalize folder name for comparison - handles curly quotes, spaces, etc."""
            normalized = unicodedata.normalize('NFD', name)
            normalized = normalized.replace(chr(0x201C), '"').replace(chr(0x201D), '"')
            normalized = normalized.replace(chr(0x2018), "'").replace(chr(0x2019), "'")
            normalized = ' '.join(normalized.split())
            return normalized.lower().strip()

        logger.debug(f"_get_full_text_for_texts called with text_ids={text_ids}")

        ocr_base_path = config.paths.processed_output_path
        if not isinstance(ocr_base_path, Path):
            ocr_base_path = Path(str(ocr_base_path))

        available_folders = {}
        if ocr_base_path.exists():
            for folder in ocr_base_path.iterdir():
                if folder.is_dir():
                    norm_name = normalize_for_comparison(folder.name)
                    available_folders[norm_name] = folder
                    logger.debug(f"Found folder: '{folder.name}' -> normalized: '{norm_name}'")

        for text_id in text_ids:
            logger.debug(f"Processing text_id: {text_id}")

            norm_text_id = normalize_for_comparison(text_id)

            folder_path = available_folders.get(norm_text_id)

            if not folder_path:
                for norm_name, folder in available_folders.items():
                    if norm_text_id in norm_name or norm_name in norm_text_id:
                        folder_path = folder
                        logger.debug(f"Partial match: '{text_id}' -> '{folder.name}'")
                        break

            if not folder_path:
                logger.warning(f"No folder found for text_id: {text_id}")
                continue

            work = get_document_by_id(text_id)
            if work is None:
                work = get_document_by_folder(text_id)
            logger.debug(f"work metadata lookup for '{text_id}': {work}")

            if work is None:
                logger.warning(f"No work metadata for text_id: {text_id}")
                work = {"title": folder_path.name, "author": "Unknown", "year": ""}

            ocr_path = folder_path / "ocr_output.json"
            logger.debug(f"ocr_path: {ocr_path}")

            if not ocr_path.exists():
                logger.warning(f"OCR file not found: {ocr_path}")
                continue

            with open(ocr_path, 'r', encoding='utf-8') as f:
                ocr_data = json.load(f)

            logger.debug(f"ocr_data type: {type(ocr_data)}")

            cleaned_pages = processor._extract_and_clean_text(ocr_data)
            logger.debug(f"cleaned_pages count: {len(cleaned_pages) if cleaned_pages else 0}")

            full_text = "\n\n".join(page_text for _, page_text in cleaned_pages)

            if full_text:
                title = work.get('title', 'Unknown') if isinstance(work, dict) else 'Unknown'
                author = work.get('author', 'Unknown') if isinstance(work, dict) else 'Unknown'
                year = work.get('year', '') if isinstance(work, dict) else ''
                header = f"=== {title} by {author} ({year}) ===\n"
                full_texts.append(header + full_text)
                logger.info(f"Loaded full text for {title}: {len(full_text)} chars")

        result = "\n\n".join(full_texts) if full_texts else ""
        logger.debug(f"_get_full_text_for_texts returning {len(result)} chars")
        return result

    def _generate_with_full_text(self, message: str, session_id: str,
                                  full_text_context: str, text_ids: List[str]) -> ChatResponse:
        """Generate response using full text context for summary queries"""
        try:
            from config import config
            from models.answer_generator import InteractionMode

            system_prompt = config.modes.summary_mode_prompt

            response = self.answer_generator.llm_provider.generate_response(
                system_prompt=system_prompt,
                user_prompt=f"Please provide a summary of the following text:\n\nUser's question: {message}\n\nFull text:\n{full_text_context[:15000]}",
                context=""
            )

            from config import get_document_by_folder
            sources = []
            for text_id in text_ids:
                work = get_document_by_folder(text_id)
                if work:
                    sources.append(f"{work.get('title', text_id)} by {work.get('author', 'Unknown')}")

            return ChatResponse(
                message=response.content,
                session_id=session_id,
                context_used=True,
                sources=sources,
                interaction_mode="text_specific",
                provider_used=self.answer_generator.get_provider_name(),
                model_used=response.model,
                metadata={
                    "mode": "full_text_summary",
                    "documents_retrieved": len(text_ids)
                }
            )

        except Exception as e:
            logger.error(f"Error generating full text response: {e}")
            return ChatResponse(
                message=f"I apologize, but I encountered an error generating the summary. Please try again.",
                session_id=session_id,
                context_used=False,
                metadata={"error": str(e), "interaction_mode": "text_specific"}
            )

    def _get_or_create_session(self, session_id: str) -> ChatSession:
        """Get existing session or create a new one under the given session_id."""
        session = self.session_service.get_session(session_id)
        if session:
            return session

        # Session not found (e.g. after server restart or worker switch).
        # Create a fresh session and register it under the *provided* session_id
        # so that all subsequent messages in this browser session reuse the same
        # in-memory state and history accumulates correctly.
        logger.info(f"Session {session_id} not found — creating placeholder session")
        now = datetime.now()
        new_session = ChatSession(
            session_id=session_id,
            created_at=now,
            last_activity=now,
        )
        self.session_service.active_sessions[session_id] = new_session
        return new_session

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get enhanced session information"""
        session = self.session_service.get_session(session_id)
        if not session:
            return None

        return {
            "session_id": session.session_id,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "message_count": len([h for h in session.conversation_history if h.get("type") == "user"]),
            "interaction_mode": session.interaction_mode.value,
            "focused_texts": session.focused_texts or [],
            "preferred_provider": session.preferred_provider,
            "rag_enabled": session.rag_enabled,
        }

    def get_llm_providers(self) -> List[Dict[str, Any]]:
        """Get available LLM providers based on configured API keys"""
        import os
        providers = []

        # HuggingFace is always available
        providers.append({
            "id": "huggingface",
            "name": "HuggingFace Inference API",
            "description": "Call Qwen/Qwen2.5-7B-Instruct via HF Router API",
            "available": True,
            "is_active": self.answer_generator.get_provider_name() == "huggingface"
        })

        # OpenAI - check for API key
        openai_key = os.getenv("OPENAI_API_KEY")
        providers.append({
            "id": "openai",
            "name": "OpenAI GPT-4o",
            "description": "Fast, high-quality responses via OpenAI API" if openai_key else "Configure OPENAI_API_KEY to enable",
            "available": bool(openai_key),
            "is_active": self.answer_generator.get_provider_name() == "openai"
        })

        # Anthropic - check for API key
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        providers.append({
            "id": "anthropic",
            "name": "Anthropic Claude",
            "description": "Thoughtful analysis via Claude API" if anthropic_key else "Configure ANTHROPIC_API_KEY to enable",
            "available": bool(anthropic_key),
            "is_active": self.answer_generator.get_provider_name() == "anthropic"
        })

        # DeepSeek - check for API key
        deepseek_key = os.getenv("DEEPSEEK_API_KEY")
        providers.append({
            "id": "deepseek",
            "name": "DeepSeek Chat",
            "description": "Cost-effective reasoning via DeepSeek API" if deepseek_key else "Configure DEEPSEEK_API_KEY to enable",
            "available": bool(deepseek_key),
            "is_active": self.answer_generator.get_provider_name() == "deepseek"
        })

        # Grok - check for API key
        grok_key = os.getenv("GROK_API_KEY")
        providers.append({
            "id": "grok",
            "name": "Grok (xAI)",
            "description": "Witty responses via Grok API" if grok_key else "Configure GROK_API_KEY to enable",
            "available": bool(grok_key),
            "is_active": self.answer_generator.get_provider_name() == "grok"
        })

        # Kimi - check for API key
        kimi_key = os.getenv("KIMI_API_KEY")
        providers.append({
            "id": "kimi",
            "name": "Kimi (Moonshot)",
            "description": "Long-context reasoning via Kimi API" if kimi_key else "Configure KIMI_API_KEY to enable",
            "available": bool(kimi_key),
            "is_active": self.answer_generator.get_provider_name() == "kimi"
        })

        # Minimax - check for API key
        minimax_key = os.getenv("MINIMAX_API_KEY")
        providers.append({
            "id": "minimax",
            "name": "Minimax",
            "description": "Multimodal AI via Minimax API" if minimax_key else "Configure MINIMAX_API_KEY to enable",
            "available": bool(minimax_key),
            "is_active": self.answer_generator.get_provider_name() == "minimax"
        })

        # Qwen - check for API key
        qwen_key = os.getenv("QWEN_API_KEY")
        providers.append({
            "id": "qwen",
            "name": "Qwen (Alibaba Cloud)",
            "description": "Powerful reasoning via Qwen API" if qwen_key else "Configure QWEN_API_KEY to enable",
            "available": bool(qwen_key),
            "is_active": self.answer_generator.get_provider_name() == "qwen"
        })

        # GLM - check for API key
        glm_key = os.getenv("GLM_API_KEY")
        providers.append({
            "id": "glm",
            "name": "GLM (Zhipu AI)",
            "description": "Bilingual AI via ChatGLM API" if glm_key else "Configure GLM_API_KEY to enable",
            "available": bool(glm_key),
            "is_active": self.answer_generator.get_provider_name() == "glm"
        })

        return providers

    def get_system_stats(self) -> Dict[str, Any]:
        """Get enhanced system statistics"""
        try:
            # Clean up expired sessions
            self._cleanup_expired_sessions()

            # Get basic stats
            active_sessions = len(self.active_sessions)
            corpus_info = {
                "supports_ai_generation": False
            }

            # Get vector store stats
            try:
                index_size = self.vector_store.count if hasattr(self.vector_store, 'count') else 0
                dimension = self.vector_store.embedding_dim if hasattr(self.vector_store, 'embedding_dim') else 0
                works_count = 0
                if hasattr(self.vector_store, 'metadata') and self.vector_store.metadata:
                    work_ids = set()
                    for meta in self.vector_store.metadata:
                        if isinstance(meta, dict) and 'work_id' in meta:
                            work_ids.add(meta['work_id'])
                    works_count = len(work_ids)

                vector_store_info = {
                    "index_size": index_size,
                    "dimension": dimension,
                    "works_count": works_count
                }
            except Exception as e:
                logger.warning(f"Error getting vector store stats: {e}")
                vector_store_info = {"index_size": 0, "dimension": 0, "works_count": 0}

            # Get retrieval system stats
            retrieval_info = {
                "status": "healthy" if self.retrieval_system else "error"
            }

            # Get enhanced answer generator stats
            try:
                health = self.answer_generator.check_model_health()
                answer_generator_info = {
                    "current_mode": "general",
                    "active_provider": self.answer_generator.get_provider_name(),
                    "health_status": health,
                    "rag_enabled": self.answer_generator.rag_enabled
                }
            except Exception as e:
                logger.warning(f"Error getting answer generator stats: {e}")
                answer_generator_info = {"error": str(e)}

            return {
                "active_sessions": active_sessions,
                "corpus": corpus_info,
                "vector_store": vector_store_info,
                "retrieval_system": retrieval_info,
                "answer_generator": answer_generator_info
            }

        except Exception as e:
            logger.error(f"Error getting system stats: {str(e)}")
            return {
                "error": str(e),
                "active_sessions": len(self.active_sessions),
                "corpus": {},
                "vector_store": {"index_size": 0},
                "retrieval_system": {"status": "error"},
                "answer_generator": {"status": "error"}
            }

    def _cleanup_expired_sessions(self):
        """Clean up expired sessions by delegating to SessionService."""
        try:
            self.session_service.cleanup_expired_sessions()
        except Exception as e:
            logger.error(f"Error cleaning up sessions: {str(e)}")
