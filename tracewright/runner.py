from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Optional

import yaml

from .assertions import AssertionResult, build_assertion
from .adapters.base import BaseAdapter
from .adapters.generic import GenericAdapter
from .reporter import Reporter
from .store import RegressionStore
from .trajectory import Trajectory


def _load_agent(spec_dir: Path, module_name: str, factory_name: str) -> Any:
    """Import *module_name*.py from *spec_dir* and call *factory_name*() to get the agent."""
    module_path = spec_dir / f"{module_name}.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Agent module not found: {module_path}")

    # Use a unique key to avoid collisions when running multiple specs
    import_key = f"_tw_{spec_dir}_{module_name}"
    spec = importlib.util.spec_from_file_location(import_key, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[import_key] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    if not hasattr(module, factory_name):
        raise AttributeError(
            f"Module '{module_name}' has no attribute '{factory_name}'"
        )
    return getattr(module, factory_name)


def _build_adapter(adapter_type: str, agent: Any) -> BaseAdapter:
    if adapter_type == "generic":
        return GenericAdapter(agent)
    if adapter_type == "smolagents":
        from .adapters.smolagents import SmolAgentsAdapter
        return SmolAgentsAdapter(agent)
    raise ValueError(
        f"Unknown adapter '{adapter_type}'. Choose 'generic' or 'smolagents'."
    )


class ScenarioSpec:
    def __init__(self, name: str, agent_input: str, assertion_specs: list[dict[str, Any]]):
        self.name = name
        self.agent_input = agent_input
        self.assertion_specs = assertion_specs


class Runner:
    def __init__(
        self,
        reporter: Optional[Reporter] = None,
        store: Optional[RegressionStore] = None,
    ):
        self.reporter = reporter or Reporter()
        self.store = store

    # ------------------------------------------------------------------
    def run_spec(
        self,
        spec_path: Path,
        filter_name: Optional[str] = None,
    ) -> tuple[int, int]:
        """
        Load and run a YAML spec file.

        Returns (passed_count, failed_count).
        """
        spec_path = spec_path.resolve()
        spec_dir = spec_path.parent
        raw = yaml.safe_load(spec_path.read_text())

        agent_cfg: dict[str, Any] = raw.get("agent", {})
        module_name: str = agent_cfg.get("module", "agent")
        factory_name: str = agent_cfg.get("factory", "create_agent")
        adapter_type: str = agent_cfg.get("adapter", "generic")

        scenarios = self._parse_scenarios(raw.get("scenarios", []))

        if filter_name:
            scenarios = [s for s in scenarios if filter_name.lower() in s.name.lower()]
            if not scenarios:
                self.reporter.console.print(
                    f"[yellow]No scenarios match filter '{filter_name}'[/yellow]"
                )
                return 0, 0

        self.reporter.start_suite(spec_path)

        passed = failed = 0

        for scenario in scenarios:
            # Fresh agent per scenario to avoid state bleed
            factory = _load_agent(spec_dir, module_name, factory_name)
            agent = factory()
            adapter = _build_adapter(adapter_type, agent)

            trajectory, results = self._run_scenario(scenario, adapter)

            self.reporter.report_scenario(scenario.name, trajectory, results)

            if all(r.passed for r in results):
                passed += 1
            else:
                failed += 1
                if self.store:
                    self.store.save_failure(
                        trajectory,
                        failed_assertions=[r.message for r in results if not r.passed],
                        spec_path=str(spec_path),
                    )

        return passed, failed

    # ------------------------------------------------------------------
    def _run_scenario(
        self,
        scenario: ScenarioSpec,
        adapter: BaseAdapter,
    ) -> tuple[Trajectory, list[AssertionResult]]:
        trajectory = adapter.run(scenario.agent_input, scenario.name)
        assertions = [build_assertion(spec) for spec in scenario.assertion_specs]
        results = [a.check(trajectory) for a in assertions]
        return trajectory, results

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_scenarios(raw_scenarios: list[dict[str, Any]]) -> list[ScenarioSpec]:
        scenarios = []
        for item in raw_scenarios:
            name = item.get("name", "unnamed")
            agent_input = item.get("input", "")
            assertion_specs = item.get("assertions", [])
            scenarios.append(ScenarioSpec(name, agent_input, assertion_specs))
        return scenarios
