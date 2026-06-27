from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

from ..trajectory import Step, ToolCall, Trajectory
from .base import BaseAdapter


class TrajectoryRecorder:
    """Passed to generic agents so they can record steps as they execute."""

    def __init__(self, scenario_name: str, agent_input: str):
        self._trajectory = Trajectory(
            scenario_name=scenario_name,
            agent_input=agent_input,
        )
        self._counter = 0

    def _add(self, step_type: str, content: str, tool_call: ToolCall | None = None) -> None:
        self._trajectory.steps.append(
            Step(index=self._counter, type=step_type, content=content, tool_call=tool_call)
        )
        self._counter += 1

    def thought(self, content: str) -> None:
        self._add("thought", content)

    def tool_call(
        self,
        name: str,
        args: dict[str, Any] | None = None,
        result: Any = None,
        error: str | None = None,
    ) -> None:
        tc = ToolCall(name=name, args=args or {}, result=result, error=error)
        self._add("tool_call", name, tool_call=tc)

    def observation(self, content: Any) -> None:
        self._add("observation", str(content))

    def final_answer(self, content: str) -> None:
        self._add("final_answer", content)

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory


class GenericAdapter(BaseAdapter):
    """
    Wraps any callable agent that accepts (task: str, recorder: TrajectoryRecorder) -> str.

    The agent is responsible for calling recorder methods to document its steps.
    """

    def __init__(self, agent_fn: Callable[[str, TrajectoryRecorder], str]):
        self.agent_fn = agent_fn

    def run(self, agent_input: str, scenario_name: str) -> Trajectory:
        recorder = TrajectoryRecorder(scenario_name=scenario_name, agent_input=agent_input)
        try:
            output = self.agent_fn(agent_input, recorder)
            recorder.trajectory.final_output = str(output) if output is not None else ""
        except Exception as exc:
            recorder.trajectory.error = str(exc)
        finally:
            recorder.trajectory.finished_at = datetime.utcnow()
        return recorder.trajectory
