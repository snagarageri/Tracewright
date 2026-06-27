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

    def test_new_fields_default_to_none(self):
        tc = ToolCall(name="search", args={})
        assert tc.latency_ms is None
        assert tc.cost_usd is None

    def test_new_fields_set_and_serialized(self):
        tc = ToolCall(name="search", args={}, latency_ms=123.4, cost_usd=0.001)
        data = tc.model_dump()
        assert data["latency_ms"] == pytest.approx(123.4)
        assert data["cost_usd"] == pytest.approx(0.001)


def make_tool_step_with_costs(
    index: int,
    tool_name: str,
    args: dict = None,
    latency_ms: float = None,
    cost_usd: float = None,
) -> Step:
    tc = ToolCall(name=tool_name, args=args or {}, latency_ms=latency_ms, cost_usd=cost_usd)
    return Step(index=index, type="tool_call", content=tool_name, tool_call=tc)


class TestNewMethods:
    # --- calls_to -----------------------------------------------------------

    def test_calls_to_returns_matching(self):
        traj = make_trajectory()
        traj.steps.append(make_tool_step(0, "search"))
        traj.steps.append(make_tool_step(1, "fetch"))
        traj.steps.append(make_tool_step(2, "search"))
        result = traj.calls_to("search")
        assert len(result) == 2
        assert all(tc.name == "search" for tc in result)

    def test_calls_to_no_match_returns_empty(self):
        traj = make_trajectory()
        traj.steps.append(make_tool_step(0, "search"))
        assert traj.calls_to("calculator") == []

    def test_calls_to_no_tools_at_all(self):
        traj = make_trajectory()
        assert traj.calls_to("anything") == []

    # --- has_tool_loop ------------------------------------------------------

    def test_has_tool_loop_false_when_no_tools(self):
        assert not make_trajectory().has_tool_loop()

    def test_has_tool_loop_false_when_different_tools(self):
        traj = make_trajectory()
        traj.steps.append(make_tool_step(0, "search", {"q": "AI"}))
        traj.steps.append(make_tool_step(1, "fetch", {"q": "AI"}))
        assert not traj.has_tool_loop()

    def test_has_tool_loop_false_when_same_tool_different_args(self):
        traj = make_trajectory()
        traj.steps.append(make_tool_step(0, "search", {"q": "AI"}))
        traj.steps.append(make_tool_step(1, "search", {"q": "Python"}))
        assert not traj.has_tool_loop()

    def test_has_tool_loop_true_on_consecutive_identical_pair(self):
        traj = make_trajectory()
        traj.steps.append(make_tool_step(0, "search", {"q": "AI"}))
        traj.steps.append(make_tool_step(1, "search", {"q": "AI"}))
        assert traj.has_tool_loop()

    def test_has_tool_loop_true_on_three_identical_in_a_row(self):
        traj = make_trajectory()
        traj.steps.append(make_tool_step(0, "search", {"q": "AI"}))
        traj.steps.append(make_tool_step(1, "search", {"q": "AI"}))
        traj.steps.append(make_tool_step(2, "search", {"q": "AI"}))
        assert traj.has_tool_loop()

    def test_has_tool_loop_false_when_separated_by_different_call(self):
        traj = make_trajectory()
        traj.steps.append(make_tool_step(0, "search", {"q": "AI"}))
        traj.steps.append(make_tool_step(1, "fetch", {"url": "x"}))
        traj.steps.append(make_tool_step(2, "search", {"q": "AI"}))
        assert not traj.has_tool_loop()

    # --- total_cost_usd -----------------------------------------------------

    def test_total_cost_usd_no_tools(self):
        assert make_trajectory().total_cost_usd() == 0.0

    def test_total_cost_usd_all_none(self):
        traj = make_trajectory()
        traj.steps.append(make_tool_step_with_costs(0, "search", cost_usd=None))
        traj.steps.append(make_tool_step_with_costs(1, "fetch", cost_usd=None))
        assert traj.total_cost_usd() == 0.0

    def test_total_cost_usd_partial_none(self):
        traj = make_trajectory()
        traj.steps.append(make_tool_step_with_costs(0, "search", cost_usd=0.01))
        traj.steps.append(make_tool_step_with_costs(1, "fetch", cost_usd=None))
        traj.steps.append(make_tool_step_with_costs(2, "calc", cost_usd=0.005))
        assert traj.total_cost_usd() == pytest.approx(0.015)

    def test_total_cost_usd_all_set(self):
        traj = make_trajectory()
        traj.steps.append(make_tool_step_with_costs(0, "a", cost_usd=1.0))
        traj.steps.append(make_tool_step_with_costs(1, "b", cost_usd=2.5))
        assert traj.total_cost_usd() == pytest.approx(3.5)

    # --- total_latency_ms ---------------------------------------------------

    def test_total_latency_ms_no_tools(self):
        assert make_trajectory().total_latency_ms() == 0.0

    def test_total_latency_ms_partial_none(self):
        traj = make_trajectory()
        traj.steps.append(make_tool_step_with_costs(0, "a", latency_ms=100.0))
        traj.steps.append(make_tool_step_with_costs(1, "b", latency_ms=None))
        traj.steps.append(make_tool_step_with_costs(2, "c", latency_ms=50.5))
        assert traj.total_latency_ms() == pytest.approx(150.5)

    def test_total_latency_ms_all_none(self):
        traj = make_trajectory()
        traj.steps.append(make_tool_step_with_costs(0, "a", latency_ms=None))
        assert traj.total_latency_ms() == 0.0

    # --- step_count ---------------------------------------------------------

    def test_step_count_empty(self):
        assert make_trajectory().step_count() == 0

    def test_step_count_with_mixed_steps(self):
        traj = make_trajectory()
        traj.steps.append(Step(index=0, type="thought", content="thinking"))
        traj.steps.append(make_tool_step(1, "search"))
        traj.steps.append(Step(index=2, type="observation", content="result"))
        assert traj.step_count() == 3

    # --- first_tool_call ----------------------------------------------------

    def test_first_tool_call_no_tools(self):
        assert make_trajectory().first_tool_call() is None

    def test_first_tool_call_returns_first(self):
        traj = make_trajectory()
        traj.steps.append(Step(index=0, type="thought", content="thinking"))
        traj.steps.append(make_tool_step(1, "search", {"q": "first"}))
        traj.steps.append(make_tool_step(2, "fetch", {"url": "second"}))
        result = traj.first_tool_call()
        assert result is not None
        assert result.name == "search"
        assert result.args == {"q": "first"}
