#!/usr/bin/env python3
"""
Doctor — diagnose one setup step at a time.

Run standalone:
    python -m commands.doctor

Or from the main chat CLI's menu ("Run doctor").
"""

from __future__ import annotations

import logging
# Keep this before any service/model imports -- see the matching comment in
# commands/cli.py for why this must run first.
logging.basicConfig(level=logging.WARNING)

import os
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import questionary
from rich.console import Console
from rich.panel import Panel

from commands.doctor_checks import STEPS, CheckResult

console = Console()

# Only providers that models.llm_providers.LLMProvider actually supports.
# (Ollama is detectable in doctor_checks but has no LLMProvider enum member,
# so it's intentionally left out of the "explain with LLM" path here.)
PROVIDER_ENV_MAP = {
    "huggingface": ["HF_TOKEN", "HUGGINGFACE_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "deepseek": ["DEEPSEEK_API_KEY"],
    "grok": ["GROK_API_KEY"],
    "kimi": ["KIMI_API_KEY"],
    "minimax": ["MINIMAX_API_KEY"],
    "qwen": ["QWEN_API_KEY"],
    "glm": ["GLM_API_KEY"],
}


def _configured_providers() -> List[str]:
    found = []
    for provider, var_names in PROVIDER_ENV_MAP.items():
        if any(os.getenv(v) for v in var_names):
            found.append(provider)
    return found


def _explain_with_llm(provider_name: str, failing: List[CheckResult]) -> Optional[str]:
    """Ask the student's configured LLM to explain the failing checks."""
    from config import config
    from models.llm_providers import LLMConfig, LLMProvider, create_llm_provider

    provider_config = getattr(config.llm_providers, provider_name)
    provider_enum = LLMProvider(provider_name)

    additional_params = {}
    for var_name in PROVIDER_ENV_MAP.get(provider_name, []):
        key = os.getenv(var_name)
        if key:
            additional_params["api_key"] = key
            break
    if "base_url" in provider_config:
        additional_params["base_url"] = provider_config["base_url"]

    llm_config = LLMConfig(
        provider=provider_enum,
        model_name=provider_config["model_name"],
        temperature=0.3,
        max_tokens=400,
        top_p=0.9,
        additional_params=additional_params,
    )

    try:
        llm = create_llm_provider(llm_config)
        failures_text = "\n".join(f"- {c.name}: {c.detail}" for c in failing)
        response = llm.generate_response(
            system_prompt=(
                "You are helping a student debug their local RAG chatbot lab setup. "
                "Be concise (under 150 words) and give the exact command or file edit "
                "needed to fix the failing check(s) below."
            ),
            user_prompt=f"These checks failed:\n{failures_text}",
        )
        return response.content
    except Exception as exc:  # noqa: BLE001
        console.print(f"[yellow]Could not reach {provider_name} to explain the failure: {exc}[/yellow]")
        return None


def _print_results(results: List[CheckResult]) -> None:
    for result in results:
        icon = "[green]PASS[/green]" if result.ok else "[red]FAIL[/red]"
        console.print(f"  {icon}  {result.name} — {result.detail}")


def _static_hints_panel(failing: List[CheckResult], title: str) -> Panel:
    body = "\n".join(f"- {r.name}: {r.fix_hint}" for r in failing if r.fix_hint)
    return Panel(body or "See STUDENT_GUIDE.md Troubleshooting section.", title=title, border_style="yellow")


def _run_step(label: str, check_fn) -> None:
    console.rule(label)

    if check_fn.__name__ == "check_chat":
        run_live = questionary.confirm(
            "Also send a live test message to the active provider? (uses quota)",
            default=False,
        ).ask()
        results = check_fn(run_live_test=bool(run_live))
    else:
        results = check_fn()

    _print_results(results)

    failing = [r for r in results if not r.ok]
    if not failing:
        console.print(Panel("[green]All checks passed for this step.[/green]"))
        return

    providers = _configured_providers()
    if not providers:
        console.print(_static_hints_panel(failing, "No LLM key configured — static fix hints"))
        return

    provider_name = providers[0]
    if len(providers) > 1:
        provider_name = questionary.select(
            "Multiple providers configured — which should explain this?",
            choices=providers,
        ).ask()
        if not provider_name:
            console.print(_static_hints_panel(failing, "Static fix hints"))
            return

    console.print(f"[dim]Asking {provider_name} to explain the failure...[/dim]")
    explanation = _explain_with_llm(provider_name, failing)
    if explanation:
        console.print(Panel(explanation, title=f"{provider_name} says", border_style="cyan"))
    else:
        console.print(_static_hints_panel(failing, "Static fix hints"))


def run_doctor() -> None:
    console.print(Panel("[bold]RAG Chatbot Lab — doctor[/bold]\nPick a step to diagnose.", border_style="blue"))
    while True:
        choice = questionary.select(
            "Which step are you stuck on?",
            choices=[label for label, _ in STEPS] + ["Exit"],
        ).ask()
        if choice is None or choice == "Exit":
            break
        label, check_fn = next(s for s in STEPS if s[0] == choice)
        _run_step(label, check_fn)


if __name__ == "__main__":
    run_doctor()
