"""
Conversation Logger Service
Writes conversation turns and session metadata to local JSON files in
``data/conversations/``. No database; this is the CLI-friendly storage
backend for the RAG Chatbot Lab.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    timestamp: str
    session_id: str
    user_id: Optional[str]
    user_message: str
    ai_response: str
    context_used: bool
    sources: List[str]
    metadata: Dict[str, Any]
    response_time: Optional[float] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ConversationSession:
    session_id: str
    user_id: Optional[str]
    start_time: str
    end_time: Optional[str]
    total_turns: int
    conversation_turns: List[ConversationTurn]
    session_metadata: Dict[str, Any]
    interaction_mode: str = "general"
    preferred_provider: Optional[str] = None
    rag_enabled: bool = True
    focused_texts: List[str] = None

    def __post_init__(self):
        if self.focused_texts is None:
            self.focused_texts = []

    def to_dict(self) -> Dict:
        return asdict(self)


class ConversationLogger:
    """Writes conversation turns to JSON files under ``data/conversations/``."""

    def __init__(self):
        self.conversations_dir = config.paths.get_absolute_path("./data/conversations")
        self.conversations_dir.mkdir(parents=True, exist_ok=True)
        (self.conversations_dir / "sessions").mkdir(exist_ok=True)
        self.active_sessions: Dict[str, ConversationSession] = {}

    def start_session(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        interaction_mode: str = "general",
        preferred_provider: Optional[str] = None,
        rag_enabled: bool = True,
        focused_texts: Optional[List[str]] = None,
    ) -> None:
        session = ConversationSession(
            session_id=session_id,
            user_id=user_id,
            start_time=datetime.utcnow().isoformat(),
            end_time=None,
            total_turns=0,
            conversation_turns=[],
            session_metadata=metadata or {},
            interaction_mode=interaction_mode,
            preferred_provider=preferred_provider,
            rag_enabled=rag_enabled,
            focused_texts=focused_texts or [],
        )
        self.active_sessions[session_id] = session
        logger.info(f"Started logging session: {session_id}")

    def log_turn(
        self,
        session_id: str,
        user_message: str,
        ai_response: str,
        context_used: bool = False,
        sources: List[str] = None,
        metadata: Dict[str, Any] = None,
        response_time: Optional[float] = None,
        user_id: Optional[str] = None,
    ) -> None:
        if session_id not in self.active_sessions:
            self.start_session(session_id, user_id, interaction_mode="general")

        session = self.active_sessions[session_id]
        turn = ConversationTurn(
            timestamp=datetime.utcnow().isoformat(),
            session_id=session_id,
            user_id=user_id or session.user_id,
            user_message=user_message,
            ai_response=ai_response,
            context_used=context_used,
            sources=sources or [],
            metadata=metadata or {},
            response_time=response_time,
        )
        session.conversation_turns.append(turn)
        session.total_turns += 1

        if session.total_turns % 5 == 0:
            self._auto_save_session(session_id)

        logger.debug(f"Logged turn for session {session_id}: {len(user_message)} chars")

    def end_session(self, session_id: str) -> Optional[ConversationSession]:
        if session_id not in self.active_sessions:
            logger.warning(f"Attempt to end non-existent session: {session_id}")
            return None

        session = self.active_sessions[session_id]
        session.end_time = datetime.utcnow().isoformat()
        self.save_session(session_id)
        completed_session = self.active_sessions.pop(session_id)
        logger.info(f"Ended session {session_id}: {session.total_turns} turns")
        return completed_session

    def save_session(self, session_id: str) -> bool:
        if session_id not in self.active_sessions:
            logger.error(f"Cannot save non-existent session: {session_id}")
            return False

        session = self.active_sessions[session_id]
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"session_{session_id[:8]}_{timestamp}.json"
            filepath = self.conversations_dir / "sessions" / filename
            session_data = session.to_dict()
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved session to: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving session {session_id}: {str(e)}")
            return False

    def _auto_save_session(self, session_id: str) -> None:
        try:
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                filename = f"session_{session_id[:8]}_backup.json"
                filepath = self.conversations_dir / "sessions" / filename
                session_data = session.to_dict()
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(session_data, f, indent=2, ensure_ascii=False)
                logger.debug(f"Auto-saved session {session_id}")
        except Exception as e:
            logger.error(f"Error auto-saving session {session_id}: {str(e)}")

def main():
    logger_service = ConversationLogger()
    test_session_id = "test_123"
    logger_service.start_session(test_session_id, "test_user", interaction_mode="general")
    logger_service.log_turn(
        test_session_id,
        "What does the corpus say?",
        "Based on the retrieved passages, the corpus suggests...",
        context_used=True,
        sources=["doc_id_1"],
        response_time=1.5,
    )
    session = logger_service.end_session(test_session_id)
    print(f"Test session completed: {session.total_turns} turns")


if __name__ == "__main__":
    main()
