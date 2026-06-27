from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.rule import Rule
from rich.table import Table
from rich import box

from .assertions import AssertionResult
from .trajectory import Trajectory


@dataclass
class ScenarioReport:
    name: str
    trajectory: Trajectory
    results: list[AssertionResult]

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def failure_messages(self) -> list[str]:
        return [r.message for r in self.results if not r.passed]


class Reporter:
    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self._reports: list[ScenarioReport] = []

    # ------------------------------------------------------------------
    def start_suite(self, spec_path: Path) -> None:
        self.console.print(Rule(f"[bold blue]Tracewright[/bold blue]  {spec_path}", style="blue"))

    # ------------------------------------------------------------------
    def report_scenario(
        self,
        scenario_name: str,
        trajectory: Trajectory,
        results: list[AssertionResult],
    ) -> None:
        report = ScenarioReport(name=scenario_name, trajectory=trajectory, results=results)
        self._reports.append(report)

        icon = "[green]✓[/green]" if report.passed else "[red]✗[/red]"
        duration = (
            f"[dim]{trajectory.duration_seconds:.2f}s[/dim]"
            if trajectory.duration_seconds is not None
            else ""
        )
        self.console.print(f"  {icon} [bold]{scenario_name}[/bold] {duration}")

        for result in results:
            if not result.passed:
                self.console.print(f"      [red]↳ [{result.assertion_type}] {result.message}[/red]")
            else:
                self.console.print(
                    f"      [dim green]↳ [{result.assertion_type}] {result.message}[/dim green]"
                )

    # ------------------------------------------------------------------
    def print_summary(self, passed: int, failed: int) -> None:
        total = passed + failed
        self.console.print(Rule(style="blue"))
        if failed == 0:
            self.console.print(
                f"[bold green]All {total} scenario(s) passed.[/bold green]"
            )
        else:
            self.console.print(
                f"[bold red]{failed}/{total} scenario(s) failed.[/bold red]"
            )

    # ------------------------------------------------------------------
    def write_junit(self, output_path: Path) -> None:
        suite = ET.Element(
            "testsuite",
            name="tracewright",
            tests=str(len(self._reports)),
            failures=str(sum(1 for r in self._reports if not r.passed)),
        )

        for report in self._reports:
            case = ET.SubElement(
                suite,
                "testcase",
                name=report.name,
                time=str(
                    round(report.trajectory.duration_seconds or 0.0, 3)
                ),
            )
            for result in report.results:
                if not result.passed:
                    ET.SubElement(
                        case,
                        "failure",
                        message=result.message,
                        type=result.assertion_type,
                    )

        tree = ET.ElementTree(suite)
        ET.indent(tree, space="  ")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tree.write(output_path, encoding="unicode", xml_declaration=True)
        self.console.print(f"[dim]JUnit XML written to {output_path}[/dim]")
