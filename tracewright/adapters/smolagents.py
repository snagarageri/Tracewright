from __future__ import annotations

from datetime import datetime
from typing import Any

from ..trajectory import Step, ToolCall, Trajectory
from .base import BaseAdapter


class SmolAgentsAdapter(BaseAdapter):
    """
    Adapter for HuggingFace smolagents (CodeAgent / ToolCallingAgent).

    The wrapped agent must have a `.run(task: str)` method. After the run,
    the adapter parses `agent.logs` (list of ActionStep / PlanningStep objects
    or dicts) to reconstruct the trajectory.
    """

    def __init__(self, agent: Any):
        self.agent = agent

    def run(self, agent_input: str, scenario_name: str) -> Trajectory:
        traj = Trajectory(scenario_name=scenario_name, agent_input=agent_input)
        try:
            result = self.agent.run(agent_input)
            traj.final_output = str(result) if result is not None else ""
            traj.steps = self._parse_logs(getattr(self.agent, "logs", []))
        except Exception as exc:
            traj.error = str(exc)
        finally:
            traj.finished_at = datetime.utcnow()
        return traj

    def _parse_logs(self, logs: list[Any]) -> list[Step]:
        steps: list[Step] = []
        for i, entry in enumerate(logs):
            parsed = self._parse_entry(i, entry)
            if parsed:
                steps.extend(parsed)
        return steps

    def _parse_entry(self, index: int, entry: Any) -> list[Step]:
        # smolagents >= 1.0 uses dataclass/object steps; older versions used dicts
        if isinstance(entry, dict):
            return self._parse_dict_entry(index, entry)
        return self._parse_object_entry(index, entry)

    def _parse_dict_entry(self, index: int, entry: dict[str, Any]) -> list[Step]:
        results: list[Step] = []
        if "thought" in entry or "rationale" in entry:
            thought = entry.get("thought") or entry.get("rationale", "")
            results.append(Step(index=index, type="thought", content=str(thought)))

        if "tool_name" in entry or "action" in entry:
            tool_name = entry.get("tool_name") or entry.get("action", "unknown")
            tool_args = entry.get("tool_arguments") or entry.get("action_input") or {}
            if isinstance(tool_args, str):
                tool_args = {"input": tool_args}
            tool_result = entry.get("observation") or entry.get("tool_output")
            tc = ToolCall(name=str(tool_name), args=tool_args, result=tool_result)
            results.append(Step(index=index, type="tool_call", content=str(tool_name), tool_call=tc))

        if "observation" in entry and "tool_name" not in entry:
            results.append(Step(index=index, type="observation", content=str(entry["observation"])))

        return results

    def _parse_object_entry(self, index: int, entry: Any) -> list[Step]:
        results: list[Step] = []
        # ActionStep (smolagents >= 1.0)
        if hasattr(entry, "model_output") and entry.model_output:
            results.append(Step(index=index, type="thought", content=str(entry.model_output)))

        if hasattr(entry, "tool_calls") and entry.tool_calls:
            for tc_obj in entry.tool_calls:
                name = getattr(tc_obj, "name", str(tc_obj))
                args = getattr(tc_obj, "arguments", {}) or {}
                tc = ToolCall(name=name, args=args)
                results.append(Step(index=index, type="tool_call", content=name, tool_call=tc))

        if hasattr(entry, "observations") and entry.observations:
            results.append(Step(index=index, type="observation", content=str(entry.observations)))

        # PlanningStep
        if hasattr(entry, "plan") and entry.plan:
            results.append(Step(index=index, type="thought", content=f"[plan] {entry.plan}"))

        return results
