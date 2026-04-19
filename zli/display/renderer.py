from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from zli.schemas import FinalResponse, RoutedTask, ExecutionPlan, VerifiedOutput

console = Console()


def render_routing(routed: RoutedTask, verbose: bool = False) -> None:
    parts = [
        f"[bold cyan]task[/bold cyan]=[yellow]{routed.task_type.value}[/yellow]",
        f"[bold cyan]complexity[/bold cyan]=[yellow]{routed.complexity.value}[/yellow]",
        f"[bold cyan]confidence[/bold cyan]=[yellow]{routed.confidence:.2f}[/yellow]",
    ]
    console.print(" Routing...   " + "  ".join(parts))
    if verbose:
        console.print(f"   [dim]{routed.routing_rationale}[/dim]")


def render_plan(plan: ExecutionPlan, verbose: bool = False) -> None:
    n = len(plan.subtasks)
    console.print(
        f" Planning...   [bold cyan]subtasks[/bold cyan]=[yellow]{n}[/yellow]  "
        f"[bold cyan]parallel[/bold cyan]=[yellow]{plan.is_parallel}[/yellow]"
    )
    if verbose:
        console.print(f"   [dim]{plan.plan_rationale}[/dim]")
        for st in plan.subtasks:
            console.print(f"   [dim]• [{st.task_type.value}] {st.prompt[:80]}[/dim]")


def render_specialist_progress(model: str, task_type: str, latency_ms: int) -> None:
    console.print(
        f" Specialist... [bold cyan]{task_type}[/bold cyan] via [yellow]{model}[/yellow] "
        f"[dim]({latency_ms}ms)[/dim]"
    )


def render_verification(verified: list[VerifiedOutput], verbose: bool = False) -> None:
    verdicts = [v.verdict.value for v in verified]
    scores = [f"{v.score:.2f}" for v in verified]
    console.print(
        f" Verifying...  [bold cyan]verdicts[/bold cyan]=[yellow]{', '.join(verdicts)}[/yellow]  "
        f"[bold cyan]scores[/bold cyan]=[yellow]{', '.join(scores)}[/yellow]"
    )
    if verbose:
        for v in verified:
            if v.issues:
                for issue in v.issues:
                    console.print(f"   [dim red]⚠ {issue}[/dim red]")


def render_final(response: FinalResponse, output_format: str = "text", verbose: bool = False) -> None:
    console.rule()

    model_trace = " + ".join(response.model_trace) if response.model_trace else "unknown"
    subtitle = (
        f"via {model_trace} | "
        f"{response.total_latency_ms}ms | "
        f"{response.total_tokens:,} tokens"
    )

    if output_format == "markdown":
        console.print(Panel(Markdown(response.answer), title="[bold green] Answer[/bold green]", subtitle=subtitle))
    elif output_format == "json":
        import json
        console.print_json(json.dumps(response.model_dump(), default=str))
    else:
        console.print(Panel(response.answer, title="[bold green] Answer[/bold green]", subtitle=subtitle))

    if response.sources:
        console.print("\n[bold cyan]Sources:[/bold cyan]")
        for src in response.sources:
            console.print(f"  • {src}")

    if verbose and response.warnings:
        console.print("\n[bold yellow]Warnings:[/bold yellow]")
        for w in response.warnings:
            console.print(f"  [yellow]⚠ {w}[/yellow]")


def render_dry_run(routed: RoutedTask, plan: ExecutionPlan) -> None:
    console.print(Panel(
        f"[bold]Task type:[/bold] {routed.task_type.value}\n"
        f"[bold]Complexity:[/bold] {routed.complexity.value}\n"
        f"[bold]Confidence:[/bold] {routed.confidence:.2f}\n"
        f"[bold]Rationale:[/bold] {routed.routing_rationale}\n\n"
        f"[bold]Subtasks ({len(plan.subtasks)}):[/bold]\n"
        + "\n".join(f"  {i+1}. [{st.task_type.value}] {st.prompt[:100]}" for i, st in enumerate(plan.subtasks)),
        title="[bold cyan] Dry Run — Routing + Plan[/bold cyan]",
    ))


def render_models_table(model_map: dict) -> None:
    table = Table(title="Specialist Model Map", show_header=True, header_style="bold cyan")
    table.add_column("Task Type", style="yellow")
    table.add_column("Primary Model")
    table.add_column("Fallback Model")
    for task_type, models in model_map.items():
        table.add_row(task_type, models["primary"], models["fallback"])
    console.print(table)


def render_error(message: str) -> None:
    console.print(Panel(f"[red]{message}[/red]", title="[bold red] Error[/bold red]"))
