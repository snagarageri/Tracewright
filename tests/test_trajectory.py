from datetime import datetime

import pytest

from tracewright.trajectory import Step, ToolCall, Trajectory


def make_trajectory(**kwargs) -> Trajectory:
    defaults = dict(scenario_name="test", agent_input="hello")
    return Trajectory(**{**defaults, **kwargs})


def make_tool_step(index: int, tool_name: str, args: dict = None) -> Step:
    tc = ToolCall(name=tool_name, args=args or {})
    return Step(index=index, type="tool_call", content=tool_name, tool_call=tc)


class TestTrajectoryProperties:
    def test_tool_names_empty(self):
        traj = make_trajectory()
        assert traj.tool_names == []

    def test_tool_names_populated(self):
        traj = make_trajectory()
        traj.steps.append(make_tool_step(0, "search"))
        traj.steps.append(make_tool_step(1, "calculator"))
        assert traj.tool_names == ["search", "calculator"]

    def test_tool_calls_filters_non_tool_steps(self):
        traj = make_trajectory()
        traj.steps.append(Step(index=0, type="thought", content="thinking"))
        traj.steps.append(make_tool_step(1, "search"))
        assert len(traj.tool_calls) == 1
        assert traj.tool_calls[0].name == "search"

    def test_thoughts_property(self):
        traj = make_trajectory()
        traj.steps.append(Step(index=0, type="thought", content="I should search"))
        traj.steps.append(make_tool_step(1, "search"))
        assert traj.thoughts == ["I should search"]

    def test_duration_seconds_none_when_not_finished(self):
        traj = make_trajectory()
        assert traj.duration_seconds is None

    def test_duration_seconds_computed(self):
        traj = make_trajectory()
        traj.finished_at = datetime(2024, 1, 1, 0, 0, 5)
        traj.started_at = datetime(2024, 1, 1, 0, 0, 0)
        assert traj.duration_seconds == pytest.approx(5.0)

    def test_trajectory_id_is_unique(self):
        t1 = make_trajectory()
        t2 = make_trajectory()
        assert t1.id != t2.id

    def test_error_defaults_to_none(self):
        traj = make_trajectory()
        assert traj.error is None


class TestToolCall:
    def test_tool_call_with_error(self):
        tc = ToolCall(name="fetch", args={"url": "http://x"}, error="timeout")
        assert tc.error == "timeout"
        assert tc.result is None

    def test_tool_call_serialization(self):
        tc = ToolCall(name="search", args={"q": "AI"}, result="found it")
        data = tc.model_dump()
        assert data["name"] == "search"
        assert data["args"]["q"] == "AI"
        assert data["result"] == "found it"
