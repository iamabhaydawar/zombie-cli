from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from zli.config import settings
from zli.display.renderer import (
    render_error,
    render_final,
    render_models_table,
)
from zli.schemas import (
    ExecutionMode,
    FinalResponse,
    MSRState,
    RoutedPlan,
    SpecialistOutput,
    TaskRequest,
    VerifiedOutput,
)

app = typer.Typer(
    name="msr",
    help="Model Specialist Router — routes your query to the best narrow AI specialist.",
    add_completion=False,
)
console = Console()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _render_orchestrate(plan: RoutedPlan, verbose: bool) -> None:
    mode_color = {
        ExecutionMode.SINGLE:     "green",
        ExecutionMode.PARALLEL:   "cyan",
        ExecutionMode.SEQUENTIAL: "yellow",
    }.get(plan.execution_mode, "white")

    multi = "[bold]multi-intent[/bold]" if plan.is_multi_intent else "single-intent"
    console.print(
        f" Orchestrating... {multi}  "
        f"[bold {mode_color}]{plan.execution_mode.value}[/bold {mode_color}]  "
        f"[dim]{len(plan.subtasks)} subtask(s)[/dim]"
    )
    if verbose:
        console.print(f"   [dim]{plan.routing_rationale}[/dim]")
        for i, st in enumerate(plan.subtasks, 1):
            dep_str = f" (after {st.depends_on})" if st.depends_on else ""
            console.print(
                f"   [dim]{i}. [{st.task_type.value}]{dep_str} conf={st.confidence:.2f} — "
                f"{st.prompt[:80]}[/dim]"
            )


# ── Main command ───────────────────────────────────────────────────────────────

@app.command()
def ask(
    query: str = typer.Argument(..., help="Your question or task."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show orchestration plan, model trace, token counts."),
    latency: int = typer.Option(30_000, "--latency", help="Max latency budget in milliseconds."),
    context_file: Optional[Path] = typer.Option(None, "--context-file", help="Path to a file whose contents are injected as context."),
    no_verify: bool = typer.Option(False, "--no-verify", help="Skip the verifier layer (faster, less safe)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show orchestration plan only — no specialist calls, no cost."),
    output_format: str = typer.Option("text", "--output-format", help="Output format: text | json | markdown."),
) -> None:
    """Send a query through the specialist router pipeline."""
    from zli.graph.graph import graph
    from zli.orchestrator.orchestrator import orchestrate

    context: str | None = None
    if context_file:
        if not context_file.exists():
            render_error(f"Context file not found: {context_file}")
            raise typer.Exit(1)
        context = context_file.read_text(encoding="utf-8")

    request = TaskRequest(query=query, context=context, max_latency_ms=latency)

    # ── Dry run: show orchestration plan, then exit ────────────────────────────
    if dry_run:
        console.print("\n[dim]Dry run — orchestration plan only, no specialist calls.[/dim]\n")
        try:
            plan = orchestrate(request)
            _render_orchestrate(plan, verbose=True)
            console.print()
            from rich.panel import Panel
            from rich.table import Table

            table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 2))
            table.add_column("#")
            table.add_column("Type")
            table.add_column("Conf")
            table.add_column("Deps")
            table.add_column("Prompt")
            for i, st in enumerate(plan.subtasks, 1):
                table.add_row(
                    str(i),
                    st.task_type.value,
                    f"{st.confidence:.2f}",
                    ", ".join(st.depends_on) or "—",
                    st.prompt[:90],
                )
            console.print(Panel(table, title=f"[bold cyan]Execution Plan — {plan.execution_mode.value.upper()}[/bold cyan]"))
        except Exception as exc:
            render_error(str(exc))
            raise typer.Exit(1)
        return

    # ── Full pipeline via LangGraph ────────────────────────────────────────────
    console.print()
    t_start = time.monotonic()

    initial_state: MSRState = {
        "request":           request.model_dump(mode="json"),
        "routed_plan":       None,
        "specialist_outputs": [],
        "verified_outputs":  [],
        "final_response":    None,
        "error":             None,
        "retry_subtask_ids": [],
    }

    try:
        final_state: dict = {}
        for event in graph.stream(initial_state, stream_mode="updates"):
            for node_name, node_output in event.items():
                final_state.update(node_output)

                if node_name == "orchestrate" and node_output.get("routed_plan"):
                    plan = RoutedPlan.model_validate(node_output["routed_plan"])
                    _render_orchestrate(plan, verbose=verbose)

                elif node_name.startswith("specialist_"):
                    for so_dict in node_output.get("specialist_outputs", []):
                        so = SpecialistOutput.model_validate(so_dict)
                        console.print(
                            f" Specialist...  [bold cyan]{so.task_type.value}[/bold cyan] "
                            f"via [yellow]{so.model_used}[/yellow] "
                            f"[dim]({so.latency_ms}ms)[/dim]"
                        )

                elif node_name == "verify" and node_output.get("verified_outputs"):
                    if not no_verify:
                        vos = [VerifiedOutput.model_validate(v) for v in node_output["verified_outputs"]]
                        verdicts = "  ".join(
                            f"[green]{v.verdict.value}[/green]" if v.verdict.value == "pass"
                            else f"[yellow]{v.verdict.value}[/yellow]" if v.verdict.value == "retry"
                            else f"[red]{v.verdict.value}[/red]"
                            for v in vos
                        )
                        scores = "  ".join(f"{v.score:.2f}" for v in vos)
                        console.print(f" Verifying...   {verdicts}  [dim]scores: {scores}[/dim]")
                        if verbose:
                            for v in vos:
                                for issue in v.issues:
                                    console.print(f"   [dim red]⚠ {issue}[/dim red]")

    except Exception as exc:
        render_error(f"Pipeline error: {exc}")
        raise typer.Exit(1)

    # ── Render final answer ────────────────────────────────────────────────────
    if final_state.get("error") and not final_state.get("final_response"):
        render_error(final_state["error"])
        raise typer.Exit(1)

    if final_state.get("final_response"):
        fr_dict = final_state["final_response"]
        fr_dict["total_latency_ms"] = int((time.monotonic() - t_start) * 1000)
        final_response = FinalResponse.model_validate(fr_dict)
        console.print(f" Synthesizing...  [yellow]{settings.synthesizer_model}[/yellow]")
        render_final(final_response, output_format=output_format, verbose=verbose)
    else:
        render_error("No response was generated.")
        raise typer.Exit(1)


# ── Config subcommand ──────────────────────────────────────────────────────────

config_app = typer.Typer(help="Configuration and API key utilities.")
app.add_typer(config_app, name="config")


@config_app.command("check")
def config_check() -> None:
    """Ping each configured API to verify keys are valid."""
    import litellm

    checks = [
        ("Anthropic (Claude)", settings.anthropic_api_key, "claude-haiku-4-5"),
        ("OpenAI (GPT)",       settings.openai_api_key,    "gpt-4o-mini"),
        ("Groq (DeepSeek)",    settings.groq_api_key,      "groq/llama3-8b-8192"),
        ("Google (Gemini)",    settings.gemini_api_key,    "gemini/gemini-2.0-flash"),
        ("xAI (Grok)",         settings.xai_api_key,       "xai/grok-3"),
    ]

    console.print("\n[bold]Checking API keys...[/bold]\n")
    all_ok = True
    for name, key, model in checks:
        if not key or key.endswith("..."):
            console.print(f"  [yellow]SKIP[/yellow]  {name} — key not set")
            continue
        try:
            litellm.completion(
                model=model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
                timeout=10,
            )
            console.print(f"  [green]OK[/green]    {name}")
        except Exception as exc:
            console.print(f"  [red]FAIL[/red]  {name} — {exc}")
            all_ok = False

    console.print()
    if all_ok:
        console.print("[bold green]All configured keys are reachable.[/bold green]")
    else:
        console.print("[bold yellow]Some keys failed. Check your .env file.[/bold yellow]")


@config_app.command("show")
def config_show() -> None:
    """Print current settings (API keys masked)."""
    def mask(val: str) -> str:
        if not val or val.endswith("..."):
            return "[dim]not set[/dim]"
        return val[:8] + "..." + val[-4:] if len(val) > 12 else "***"

    console.print("\n[bold]Current ZLI settings:[/bold]\n")
    console.print(f"  ANTHROPIC_API_KEY  : {mask(settings.anthropic_api_key)}")
    console.print(f"  OPENAI_API_KEY     : {mask(settings.openai_api_key)}")
    console.print(f"  GROQ_API_KEY       : {mask(settings.groq_api_key)}")
    console.print(f"  GEMINI_API_KEY     : {mask(settings.gemini_api_key)}")
    console.print(f"  PERPLEXITY_API_KEY : {mask(settings.perplexity_api_key)}")
    console.print(f"  XAI_API_KEY        : {mask(settings.xai_api_key)}")
    console.print(f"\n  Timeout            : {settings.zli_default_timeout_s}s")
    console.print(f"  Max retries        : {settings.zli_max_retries}")
    console.print(f"  Orchestrator model : {settings.planner_model}")
    console.print(f"  Verifier model     : {settings.verifier_model}")
    console.print(f"  Synthesizer model  : {settings.synthesizer_model}\n")


# ── Models subcommand ──────────────────────────────────────────────────────────

@app.command("models")
def models_list() -> None:
    """Show which model handles each task type."""
    render_models_table(settings.model_map)


if __name__ == "__main__":
    app()
