from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console

from . import __version__
from .reporter import Reporter
from .runner import Runner
from .store import RegressionStore

console = Console()


@click.group()
@click.version_option(__version__, prog_name="tw")
def main() -> None:
    """Tracewright — test AI agent trajectories, not just outputs."""


# ---------------------------------------------------------------------------
@main.command()
@click.argument("spec_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--filter", "-f", "filter_name",
    default=None,
    metavar="NAME",
    help="Run only scenarios whose name contains NAME (case-insensitive).",
)
@click.option(
    "--junit", "-j", "junit_path",
    default=None,
    type=click.Path(path_type=Path),
    metavar="FILE",
    help="Also write JUnit XML results to FILE.",
)
@click.option(
    "--save-failures/--no-save-failures",
    default=True,
    show_default=True,
    help="Persist failed scenarios to the regression store (.tracewright/).",
)
def run(
    spec_path: Path,
    filter_name: str | None,
    junit_path: Path | None,
    save_failures: bool,
) -> None:
    """Run test scenarios from a YAML spec file."""
    reporter = Reporter(console)
    store = RegressionStore() if save_failures else None
    runner = Runner(reporter=reporter, store=store)

    passed, failed = runner.run_spec(spec_path, filter_name=filter_name)

    if junit_path:
        reporter.write_junit(junit_path)

    reporter.print_summary(passed, failed)

    sys.exit(1 if failed > 0 else 0)


# ---------------------------------------------------------------------------
@main.group()
def store() -> None:
    """Manage the regression store (.tracewright/)."""


@store.command("list")
def store_list() -> None:
    """List all stored failure records."""
    s = RegressionStore()
    failures = s.list_failures()
    if not failures:
        console.print("[dim]No stored failures.[/dim]")
        return
    for f in failures:
        tags = ", ".join(f.get("failed_assertions", []))[:80]
        console.print(
            f"[red]✗[/red]  [bold]{f['scenario']}[/bold]  "
            f"[dim]{f['saved_at']}[/dim]\n"
            f"    spec: {f['spec_path']}\n"
            f"    failures: {tags}"
        )


@store.command("clear")
@click.confirmation_option(prompt="Delete all stored failures?")
def store_clear() -> None:
    """Delete all stored failure records."""
    removed = RegressionStore().clear()
    console.print(f"[green]Cleared {removed} record(s).[/green]")
