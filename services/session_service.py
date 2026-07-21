"""
Session Service
Manages chat sessions with interaction-mode tracking.
"""

import logging
import uuid
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from models.answer_generator import InteractionMode

logger = logging.getLogger(__name__)


@dataclass
class ChatSession:
    """Represents an active chat session in the RAG Chatbot Lab."""
    session_id: str
    created_at: datetime
    last_activity: datetime
    conversation_history: List[Dict] = field(default_factory=list)

    interaction_mode: InteractionMode = InteractionMode.GENERAL
    focused_texts: Optional[List[str]] = None
    preferred_provider: Optional[str] = None
    rag_enabled: bool = True


class SessionService:
    """Service for managing chat sessions."""

    def __init__(self):
        self.active_sessions: Dict[str, ChatSession] = {}
        self.session_timeout = timedelta(hours=2)

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        return self.active_sessions.get(session_id)

    def create_session(self, user_id: Optional[str] = None, mode: str = "general") -> Tuple[str, Dict]:
        """Create a new chat session and return ``(session_id, metadata)``.

        ``metadata`` is a small dict callers can echo to the client so the
        response shape is consistent regardless of whether any extra seed
        content is supplied.
        """
        session_id = str(uuid.uuid4())
        now = datetime.now()

        try:
            interaction_mode = InteractionMode(mode)
        except ValueError:
            interaction_mode = InteractionMode.GENERAL
            logger.warning(f"Invalid mode '{mode}', defaulting to general")

        session = ChatSession(
            session_id=session_id,
            created_at=now,
            last_activity=now,
            interaction_mode=interaction_mode,
        )

        self.active_sessions[session_id] = session

        logger.info(f"Created new session: {session_id} (mode: {interaction_mode.value})")
        return session_id, {
            "session_id": session_id,
            "created_at": now.isoformat(),
            "interaction_mode": interaction_mode.value,
            "rag_enabled": session.rag_enabled,
        }

    def set_session_mode(self, session_id: str, mode: str) -> bool:
        if session_id not in self.active_sessions:
            logger.error(f"Session {session_id} not found")
            return False
        try:
            new_mode = InteractionMode(mode)
        except ValueError:
            logger.error(f"Invalid mode: {mode}")
            return False

        session = self.active_sessions[session_id]
        old_mode = session.interaction_mode
        session.interaction_mode = new_mode
        session.last_activity = datetime.now()
        if new_mode == InteractionMode.GENERAL:
            session.focused_texts = None
        logger.info(f"Session {session_id} mode changed from {old_mode.value} to {new_mode.value}")
        return True

    def set_session_provider(self, session_id: str, provider: str) -> bool:
        if session_id not in self.active_sessions:
            logger.error(f"Session {session_id} not found")
            return False
        session = self.active_sessions[session_id]
        session.preferred_provider = provider
        session.last_activity = datetime.now()
        logger.info(f"Session {session_id} provider set to {provider}")
        return True

    def set_session_rag(self, session_id: str, rag_enabled: bool) -> bool:
        if session_id not in self.active_sessions:
            logger.error(f"Session {session_id} not found")
            return False
        session = self.active_sessions[session_id]
        session.rag_enabled = rag_enabled
        session.last_activity = datetime.now()
        logger.info(f"Session {session_id} RAG {'enabled' if rag_enabled else 'disabled'}")
        return True

    def cleanup_expired_sessions(self) -> int:
        now = datetime.now()
        expired = [
            sid for sid, session in self.active_sessions.items()
            if now - session.last_activity > self.session_timeout
        ]
        for sid in expired:
            del self.active_sessions[sid]
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")
        return len(expired)
