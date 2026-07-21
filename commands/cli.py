#!/usr/bin/env python3
"""
RAG Chatbot Lab — interactive CLI for chatting with the loaded corpus.

Run:
    python -m commands.cli

A main menu lets you start chatting, switch provider/mode/document filter,
toggle RAG, view stats, save the conversation, or run the doctor diagnostic
tool. Inside the chat loop, type /menu to come back to the menu, /works for
the corpus + "add your own document" guide, or /quit to save and exit.
"""

import logging
# Must run before any service/model module imports, since each of them calls
# logging.basicConfig(level=INFO) itself; whichever call happens first wins
# for the whole process. Keeping it at WARNING here is what makes the CLI
# feel like a clean menu instead of a wall of INFO log lines.
logging.basicConfig(level=logging.WARNING)

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

try:
    from services.chatbot_service import ChatbotService
    from config import config
    from models.answer_generator import InteractionMode
    from commands.doctor import run_doctor
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

console = Console()

MENU_CHAT = "Start chatting"
MENU_PROVIDER = "Switch provider"
MENU_MODE = "Switch mode (general / text_specific)"
MENU_FILTER = "Pick a document to focus on"
MENU_RAG = "Toggle RAG on/off"
MENU_INFO = "Session info / stats"
MENU_WORKS = "View corpus (and how to add your own document)"
MENU_SAVE = "Save conversation"
MENU_DOCTOR = "Run doctor (diagnose a setup step)"
MENU_QUIT = "Quit"

MAIN_MENU_CHOICES = [
    MENU_CHAT, MENU_PROVIDER, MENU_MODE, MENU_FILTER, MENU_RAG,
    MENU_INFO, MENU_WORKS, MENU_SAVE, MENU_DOCTOR, MENU_QUIT,
]

ADD_DOCUMENT_GUIDE = (
    "1. mkdir -p knowledge_base/<doc_id>\n"
    "2. Create knowledge_base/<doc_id>/meta.yaml:\n"
    "     id: <doc_id>          # unique slug -- what you pass to the filter menu\n"
    "     title: \"...\"\n"
    "     author: \"...\"\n"
    "     year: \"...\"\n"
    "     genre: \"...\"\n"
    "     source: \"...\"\n"
    "     folder: \"<doc_id>\"\n"
    "     ocr_file: \"ocr_output.json\"   # default; can be renamed\n"
    "3. Add your text as knowledge_base/<doc_id>/ocr_output.json\n"
    "   (a flat JSON object of page_N -> page text, e.g. {\"page_1\": \"...\"})\n"
    "4. Rebuild the index: python tests/quick_test.py --rebuild\n"
    "5. Restart the CLI -- your new <doc_id> shows up in this list"
)


class CLITester:
    """Interactive CLI for the RAG Chatbot Lab."""

    def __init__(self):
        self.service: Optional[ChatbotService] = None
        self.session_id: Optional[str] = None
        self.conversation_log = []
        self.logs_dir = Path("./data/conversations")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.current_text_filter: Optional[list] = None
        self.available_documents: list = []
        self.current_mode = "general"
        self.current_provider: Optional[str] = None
        self.rag_enabled = True

    def initialize_system(self) -> bool:
        try:
            with console.status("[bold cyan]Initializing RAG Chatbot Lab...[/bold cyan]"):
                self.service = ChatbotService()
                self._load_available_documents()
            return True
        except Exception as e:
            console.print(f"[red]Failed to initialize system: {e}[/red]")
            return False

    def _load_available_documents(self):
        try:
            from config import get_active_documents
            self.available_documents = get_active_documents()
        except Exception as e:
            console.print(f"[yellow]Could not load documents list: {e}[/yellow]")
            self.available_documents = []

    def start_session(self) -> bool:
        try:
            self.current_text_filter = None
            self.session_id, _ = self.service.start_chat_session(mode=self.current_mode)
            return True
        except Exception as e:
            console.print(f"[red]Failed to start session: {e}[/red]")
            return False

    def print_startup_summary(self):
        table = Table(title="RAG Chatbot Lab", show_header=False)
        table.add_row("Corpus documents", str(len(self.available_documents)))
        try:
            stats = self.service.get_system_stats()
            index_size = stats.get('vector_store', {}).get('index_size', 'N/A')
        except Exception:
            index_size = "N/A"
        table.add_row("Vector index", f"{index_size} vectors")
        table.add_row("Active provider", self.current_provider or "default")
        table.add_row("RAG", "enabled" if self.rag_enabled else "disabled")
        console.print(table)

    def main_menu_loop(self):
        while True:
            choice = questionary.select(
                "What would you like to do?",
                choices=MAIN_MENU_CHOICES,
            ).ask()

            if choice is None or choice == MENU_QUIT:
                self.save_conversation()
                console.print("[bold]Saved conversation to data/conversations/. Bye![/bold]")
                return
            elif choice == MENU_CHAT:
                self.chat_loop()
            elif choice == MENU_PROVIDER:
                self.menu_switch_provider()
            elif choice == MENU_MODE:
                self.menu_switch_mode()
            elif choice == MENU_FILTER:
                self.menu_pick_document()
            elif choice == MENU_RAG:
                self.menu_toggle_rag()
            elif choice == MENU_INFO:
                self.show_session_info()
                self.show_system_stats()
            elif choice == MENU_WORKS:
                self.show_available_documents()
            elif choice == MENU_SAVE:
                self.save_conversation()
            elif choice == MENU_DOCTOR:
                run_doctor()

    def chat_loop(self):
        console.print(Panel(
            "Type your message. [bold]/menu[/bold] returns to the main menu, "
            "[bold]/works[/bold] shows the corpus + how to add a document, "
            "[bold]/quit[/bold] saves and exits, [bold]/help[/bold] for this reminder.",
            title="Chat",
        ))
        interrupted = False
        while True:
            try:
                user_input = input("You: ").strip()
                interrupted = False
                if not user_input:
                    continue
                if user_input == "/menu":
                    return
                if user_input in ("/quit", "/exit"):
                    self.save_conversation()
                    console.print("[bold]Saved conversation to data/conversations/. Bye![/bold]")
                    sys.exit(0)
                if user_input == "/help":
                    console.print("[dim]/menu -- back to main menu   /works -- corpus + add-a-doc guide   /quit -- save and exit[/dim]")
                    continue
                if user_input == "/works":
                    self.show_available_documents()
                    continue
                self.process_message(user_input)
            except KeyboardInterrupt:
                if interrupted:
                    console.print("\n[bold]Force quit — saving conversation. Bye![/bold]")
                    self.save_conversation()
                    sys.exit(0)
                interrupted = True
                console.print("\n[yellow]Chat interrupted. /menu to continue, Ctrl-C again to force quit.[/yellow]")
            except EOFError:
                return

    def process_message(self, message: str):
        try:
            with console.status("[cyan]Thinking...[/cyan]"):
                start_time = datetime.now()
                if self.current_text_filter:
                    response = self.service.process_message_for_texts(
                        message=message,
                        session_id=self.session_id,
                        text_ids=self.current_text_filter,
                    )
                else:
                    response = self.service.process_message(message, self.session_id)
                response_time = (datetime.now() - start_time).total_seconds()

            console.print(Panel(response.message, title="Assistant"))

            footer_bits = []
            if response.sources:
                footer_bits.append(f"sources: {', '.join(response.sources)}")
            if getattr(response, 'provider_used', None):
                footer_bits.append(f"provider: {response.provider_used}")
            footer_bits.append(f"{response_time:.2f}s")
            console.print(f"[dim]{' | '.join(footer_bits)}[/dim]")

            self.log_conversation(message, response, response_time)
        except Exception as e:
            console.print(f"[red]Error processing message: {e}[/red]")

    def menu_switch_provider(self):
        try:
            providers = self.service.get_llm_providers() if self.service else []
        except Exception as e:
            console.print(f"[red]Error listing providers: {e}[/red]")
            return

        usable = [p for p in providers if p.get("available")]
        if not usable:
            console.print("[yellow]No providers have a key configured in .env. Add one and restart the CLI.[/yellow]")
            return

        choice = questionary.select(
            "Choose a provider:",
            choices=[f"{p['id']} ({p['name']})" for p in usable] + ["Cancel"],
        ).ask()
        if not choice or choice == "Cancel":
            return
        provider_id = choice.split(" ")[0]

        if self.session_id and self.service:
            success = self.service.set_session_provider(self.session_id, provider_id)
            if success:
                self.current_provider = provider_id
                console.print(f"[green]Provider set to {provider_id}[/green]")
            else:
                console.print(f"[red]Could not switch to {provider_id}[/red]")

    def menu_switch_mode(self):
        new_mode = questionary.select(
            "Choose a mode:",
            choices=["general", "text_specific"],
        ).ask()
        if not new_mode:
            return
        old_mode = self.current_mode
        self.current_mode = new_mode
        if self.session_id and self.service:
            success = self.service.set_session_mode(self.session_id, new_mode)
            if success:
                console.print(f"[green]Mode: {old_mode} -> {new_mode}[/green]")
                if new_mode == "general":
                    self.current_text_filter = None
                elif not self.current_text_filter:
                    console.print("[dim]Pick a document (\"Pick a document to focus on\") to focus text_specific mode.[/dim]")
            else:
                self.current_mode = old_mode
                console.print("[red]Failed to update session mode[/red]")

    def menu_toggle_rag(self):
        new_setting = not self.rag_enabled
        if self.session_id and self.service:
            success = self.service.set_session_rag(self.session_id, new_setting)
            if success:
                self.rag_enabled = new_setting
                console.print(f"[green]RAG: {'on' if new_setting else 'off'}[/green]")
            else:
                console.print("[red]Failed to update RAG setting[/red]")

    def menu_pick_document(self):
        if not self.available_documents:
            console.print("[yellow]No documents loaded yet.[/yellow]")
            self._print_add_document_guide()
            return

        choice = questionary.select(
            "Pick a document to focus on:",
            choices=[f"{d.get('id', 'unknown')} -- {d.get('title', '(untitled)')}" for d in self.available_documents] + ["Clear filter", "Cancel"],
        ).ask()
        if not choice or choice == "Cancel":
            return
        if choice == "Clear filter":
            self.current_text_filter = None
            if self.session_id and self.service:
                self.service.set_session_mode(self.session_id, "general")
                self.current_mode = "general"
            console.print("[green]Filter cleared -- back to whole-corpus mode.[/green]")
            return

        doc_id = choice.split(" -- ")[0]
        self._apply_filter(doc_id)

    def _apply_filter(self, doc_id: str):
        doc_info = next((d for d in self.available_documents if d.get("id") == doc_id), None)
        folder_name = doc_info.get("folder") if doc_info else doc_id
        self.current_text_filter = [folder_name]
        if self.session_id and self.service:
            self.service.set_session_mode(self.session_id, "text_specific")
            self.current_mode = "text_specific"
        console.print(f"[green]Filter set to {doc_id}[/green]")

    def show_system_stats(self):
        try:
            stats = self.service.get_system_stats()
            answer_gen_info = stats.get('answer_generator', {})
            table = Table(title="Lab stats", show_header=False)
            table.add_row("Corpus documents", str(len(self.available_documents)))
            table.add_row("Vector store", f"{stats.get('vector_store', {}).get('index_size', 'N/A')} vectors")
            table.add_row("Active provider", answer_gen_info.get('active_provider', 'default'))
            table.add_row("Model health", answer_gen_info.get('health_status', {}).get('status', 'unknown'))
            console.print(table)
        except Exception as e:
            console.print(f"[red]Error getting stats: {e}[/red]")

    def show_session_info(self):
        try:
            info = self.service.get_session_info(self.session_id)
            if not info:
                console.print("[yellow]No session. Start a chat message first.[/yellow]")
                return
            sid = (info.get('session_id') or 'N/A')
            table = Table(title="Session", show_header=False)
            table.add_row("id", f"{sid[:8]}…")
            table.add_row("messages", str(info.get('message_count', 0)))
            table.add_row("mode", info.get('interaction_mode', 'N/A'))
            table.add_row("RAG", 'on' if info.get('rag_enabled') else 'off')
            table.add_row("filter", ', '.join(info.get('focused_texts', [])) or 'none')
            table.add_row("provider", info.get('preferred_provider') or 'default')
            console.print(table)
        except Exception as e:
            console.print(f"[red]Error getting session info: {e}[/red]")

    def show_available_documents(self):
        """List the active corpus AND teach how to add a new <doc_id>.

        This always shows the "add your own document" walkthrough -- whether
        the corpus is populated or empty -- since that's how a student turns
        this from "someone else's demo" into their own agent.
        """
        if self.available_documents:
            table = Table(title=f"Active corpus — {len(self.available_documents)} document(s)")
            table.add_column("id")
            table.add_column("Title")
            table.add_column("Author")
            for doc in self.available_documents:
                doc_id = doc.get("id", "unknown")
                title = doc.get("title") or "(untitled)"
                author = (doc.get("author") or "").strip() or "—"
                table.add_row(doc_id, title, author)
            console.print(table)
        else:
            console.print("[yellow]No documents loaded yet -- the corpus is empty.[/yellow]")

        self._print_add_document_guide()

    def _print_add_document_guide(self):
        console.print(Panel(
            ADD_DOCUMENT_GUIDE,
            title="Want to add your own document?",
            border_style="cyan",
        ))

    def log_conversation(self, user_message: str, response, response_time: float):
        self.conversation_log.append({
            "timestamp": datetime.now().isoformat(),
            "user_message": user_message,
            "ai_response": response.message,
            "context_used": response.context_used,
            "sources": response.sources,
            "metadata": response.metadata,
            "response_time": response_time,
        })

    def save_conversation(self):
        if not self.conversation_log:
            console.print("[dim]No conversation to save.[/dim]")
            return
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sid_short = (self.session_id or "unknown")[:8]
            filename = f"conversation_{sid_short}_{timestamp}.json"
            filepath = self.logs_dir / filename
            conversation_data = {
                "session_id": self.session_id,
                "created_at": datetime.now().isoformat(),
                "total_turns": len(self.conversation_log),
                "conversation": self.conversation_log,
            }
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)
            console.print(f"[green]Conversation saved to: {filepath}[/green]")
        except Exception as e:
            console.print(f"[red]Error saving conversation: {e}[/red]")


def main():
    tester = CLITester()
    if not tester.initialize_system():
        return
    if not tester.start_session():
        return
    tester.print_startup_summary()
    try:
        tester.main_menu_loop()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        tester.save_conversation()


if __name__ == "__main__":
    main()
