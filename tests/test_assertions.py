import pytest

from tracewright.assertions import (
    CallCount,
    CalledTool,
    DidNotCallTool,
    LLMJudge,
    MaxCost,
    NoError,
    NoToolLoops,
    OutputContains,
    OutputMatches,
    StepCount,
    ToolCalledWith,
    ToolOrder,
    build_assertion,
)
from tracewright.trajectory import Step, ToolCall, Trajectory


def make_traj(
    output: str = "done",
    error: str = None,
    tools: list[tuple[str, dict]] = None,
    steps: int = 0,
) -> Trajectory:
    traj = Trajectory(scenario_name="test", agent_input="test input")
    traj.final_output = output
    traj.error = error
    for i, (name, args) in enumerate(tools or []):
        tc = ToolCall(name=name, args=args)
        traj.steps.append(Step(index=i, type="tool_call", content=name, tool_call=tc))
    for i in range(steps):
        idx = len(traj.steps)
        traj.steps.append(Step(index=idx, type="thought", content=f"step {i}"))
    return traj


def make_traj_with_costs(tool_costs: list[tuple[str, float | None]]) -> Trajectory:
    """Build a trajectory where each tool call carries an optional cost_usd."""
    traj = Trajectory(scenario_name="test", agent_input="test input")
    traj.final_output = "done"
    for i, (name, cost) in enumerate(tool_costs):
        tc = ToolCall(name=name, args={}, cost_usd=cost)
        traj.steps.append(Step(index=i, type="tool_call", content=name, tool_call=tc))
    return traj


class TestNoError:
    def test_passes_when_no_error(self):
        assert NoError().check(make_traj()).passed

    def test_fails_when_error_set(self):
        result = NoError().check(make_traj(error="timeout"))
        assert not result.passed
        assert "timeout" in result.message


class TestOutputContains:
    def test_passes_case_insensitive(self):
        assert OutputContains("hello").check(make_traj(output="Hello World")).passed

    def test_fails_when_missing(self):
        result = OutputContains("missing").check(make_traj(output="something else"))
        assert not result.passed

    def test_case_sensitive_mode(self):
        assert not OutputContains("hello", case_sensitive=True).check(make_traj(output="Hello")).passed
        assert OutputContains("Hello", case_sensitive=True).check(make_traj(output="Hello")).passed

    def test_empty_output(self):
        assert not OutputContains("text").check(make_traj(output="")).passed


class TestOutputMatches:
    def test_matches_regex(self):
        assert OutputMatches(r"\d{4}").check(make_traj(output="Year 2024")).passed

    def test_no_match(self):
        result = OutputMatches(r"\d{10}").check(make_traj(output="short"))
        assert not result.passed

    def test_default_case_insensitive(self):
        assert OutputMatches("HELLO").check(make_traj(output="hello world")).passed


class TestCalledTool:
    def test_passes_when_tool_called(self):
        traj = make_traj(tools=[("search", {"q": "AI"})])
        assert CalledTool("search").check(traj).passed

    def test_fails_when_tool_not_called(self):
        result = CalledTool("calculator").check(make_traj())
        assert not result.passed
        assert "calculator" in result.message

    def test_multiple_tools(self):
        traj = make_traj(tools=[("search", {}), ("fetch", {})])
        assert CalledTool("search").check(traj).passed
        assert CalledTool("fetch").check(traj).passed
        assert not CalledTool("missing").check(traj).passed


class TestToolCalledWith:
    def test_passes_with_matching_args(self):
        traj = make_traj(tools=[("search", {"q": "AI", "lang": "en"})])
        assert ToolCalledWith("search", {"q": "AI"}).check(traj).passed

    def test_fails_with_wrong_arg_value(self):
        traj = make_traj(tools=[("search", {"q": "Python"})])
        result = ToolCalledWith("search", {"q": "AI"}).check(traj)
        assert not result.passed

    def test_fails_when_tool_never_called(self):
        result = ToolCalledWith("search", {"q": "AI"}).check(make_traj())
        assert not result.passed
        assert "never called" in result.message

    def test_empty_expected_args_always_passes_if_tool_called(self):
        traj = make_traj(tools=[("search", {"q": "anything"})])
        assert ToolCalledWith("search", {}).check(traj).passed


class TestStepCount:
    def test_passes_within_range(self):
        traj = make_traj(steps=3)
        assert StepCount(min_steps=1, max_steps=5).check(traj).passed

    def test_fails_below_min(self):
        traj = make_traj(steps=1)
        result = StepCount(min_steps=3).check(traj)
        assert not result.passed

    def test_fails_above_max(self):
        traj = make_traj(steps=10)
        result = StepCount(max_steps=5).check(traj)
        assert not result.passed

    def test_no_bounds_always_passes(self):
        traj = make_traj(steps=100)
        assert StepCount().check(traj).passed

    def test_exact_boundary(self):
        traj = make_traj(steps=5)
        assert StepCount(min_steps=5, max_steps=5).check(traj).passed


class TestDidNotCallTool:
    def test_passes_when_tool_never_called(self):
        assert DidNotCallTool("delete_record").check(make_traj()).passed

    def test_fails_when_tool_was_called(self):
        traj = make_traj(tools=[("delete_record", {"id": 42})])
        result = DidNotCallTool("delete_record").check(traj)
        assert not result.passed

    def test_fail_message_lists_invocations(self):
        traj = make_traj(tools=[
            ("delete_record", {"id": 1}),
            ("delete_record", {"id": 2}),
        ])
        result = DidNotCallTool("delete_record").check(traj)
        assert not result.passed
        assert "2 time(s)" in result.message
        assert "delete_record" in result.message

    def test_other_tools_do_not_cause_failure(self):
        traj = make_traj(tools=[("search", {}), ("fetch", {})])
        assert DidNotCallTool("delete_record").check(traj).passed


class TestToolOrder:
    def test_passes_with_exact_contiguous_sequence(self):
        traj = make_traj(tools=[("search", {}), ("summarize", {})])
        assert ToolOrder(["search", "summarize"]).check(traj).passed

    def test_passes_when_other_tools_between_required(self):
        traj = make_traj(tools=[("search", {}), ("log", {}), ("summarize", {})])
        assert ToolOrder(["search", "summarize"]).check(traj).passed

    def test_passes_with_single_element_sequence(self):
        traj = make_traj(tools=[("search", {}), ("fetch", {})])
        assert ToolOrder(["fetch"]).check(traj).passed

    def test_passes_with_empty_sequence(self):
        assert ToolOrder([]).check(make_traj()).passed

    def test_fails_when_required_tool_missing(self):
        traj = make_traj(tools=[("search", {})])
        result = ToolOrder(["search", "summarize"]).check(traj)
        assert not result.passed
        assert "summarize" in result.message or "search" in result.message

    def test_fails_when_order_reversed(self):
        traj = make_traj(tools=[("summarize", {}), ("search", {})])
        result = ToolOrder(["search", "summarize"]).check(traj)
        assert not result.passed

    def test_fails_when_no_tools_called(self):
        result = ToolOrder(["search"]).check(make_traj())
        assert not result.passed

    def test_fail_message_shows_actual_order(self):
        traj = make_traj(tools=[("summarize", {}), ("search", {})])
        result = ToolOrder(["search", "summarize"]).check(traj)
        assert "summarize" in result.message or "search" in result.message


class TestNoToolLoops:
    def test_passes_with_no_tools(self):
        assert NoToolLoops().check(make_traj()).passed

    def test_passes_when_same_tool_different_args(self):
        traj = make_traj(tools=[("search", {"q": "AI"}), ("search", {"q": "Python"})])
        assert NoToolLoops().check(traj).passed

    def test_passes_when_same_tool_not_consecutive(self):
        traj = make_traj(tools=[
            ("search", {"q": "AI"}), ("fetch", {}), ("search", {"q": "AI"})
        ])
        assert NoToolLoops().check(traj).passed

    def test_fails_on_consecutive_identical_pair(self):
        traj = make_traj(tools=[("search", {"q": "AI"}), ("search", {"q": "AI"})])
        result = NoToolLoops().check(traj)
        assert not result.passed

    def test_fail_message_names_the_looping_tool(self):
        traj = make_traj(tools=[("search", {"q": "AI"}), ("search", {"q": "AI"})])
        result = NoToolLoops().check(traj)
        assert not result.passed
        assert "search" in result.message

    def test_fail_message_includes_args(self):
        traj = make_traj(tools=[("fetch", {"url": "http://x"}), ("fetch", {"url": "http://x"})])
        result = NoToolLoops().check(traj)
        assert "http://x" in result.message or "url" in result.message


class TestMaxCost:
    def test_passes_when_no_costs(self):
        assert MaxCost(1.0).check(make_traj()).passed

    def test_passes_when_total_below_max(self):
        traj = make_traj_with_costs([("search", 0.01), ("fetch", 0.02)])
        assert MaxCost(0.05).check(traj).passed

    def test_passes_at_exact_limit(self):
        traj = make_traj_with_costs([("search", 0.05)])
        assert MaxCost(0.05).check(traj).passed

    def test_passes_when_all_costs_none(self):
        traj = make_traj_with_costs([("search", None), ("fetch", None)])
        assert MaxCost(0.0).check(traj).passed

    def test_passes_with_partial_none_within_limit(self):
        traj = make_traj_with_costs([("search", 0.01), ("fetch", None)])
        assert MaxCost(0.05).check(traj).passed

    def test_fails_when_total_exceeds_max(self):
        traj = make_traj_with_costs([("search", 0.03), ("fetch", 0.03)])
        result = MaxCost(0.05).check(traj)
        assert not result.passed

    def test_fail_message_shows_actual_and_limit(self):
        traj = make_traj_with_costs([("search", 0.10)])
        result = MaxCost(0.05).check(traj)
        assert not result.passed
        assert "0.1000" in result.message
        assert "0.0500" in result.message


class TestCallCount:
    def test_passes_when_count_matches(self):
        traj = make_traj(tools=[("search", {}), ("search", {})])
        assert CallCount("search", 2).check(traj).passed

    def test_passes_when_zero_calls_expected_and_none_made(self):
        assert CallCount("search", 0).check(make_traj()).passed

    def test_fails_when_too_many_calls(self):
        traj = make_traj(tools=[("search", {}), ("search", {}), ("search", {})])
        result = CallCount("search", 2).check(traj)
        assert not result.passed

    def test_fails_when_too_few_calls(self):
        traj = make_traj(tools=[("search", {})])
        result = CallCount("search", 3).check(traj)
        assert not result.passed

    def test_fail_message_shows_actual_count(self):
        traj = make_traj(tools=[("search", {})])
        result = CallCount("search", 3).check(traj)
        assert not result.passed
        assert "1" in result.message  # actual
        assert "3" in result.message  # expected

    def test_counts_only_named_tool(self):
        traj = make_traj(tools=[("search", {}), ("fetch", {}), ("search", {})])
        assert CallCount("search", 2).check(traj).passed
        assert CallCount("fetch", 1).check(traj).passed


class TestBuildAssertion:
    def test_build_no_error(self):
        a = build_assertion({"type": "no_error"})
        assert isinstance(a, NoError)

    def test_build_output_contains(self):
        a = build_assertion({"type": "output_contains", "value": "hello"})
        assert isinstance(a, OutputContains)

    def test_build_called_tool(self):
        a = build_assertion({"type": "called_tool", "name": "search"})
        assert isinstance(a, CalledTool)

    def test_build_step_count(self):
        a = build_assertion({"type": "step_count", "min": 1, "max": 5})
        assert isinstance(a, StepCount)
        assert a.min_steps == 1
        assert a.max_steps == 5

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown assertion type"):
            build_assertion({"type": "nonexistent"})

    def test_build_must_not_call(self):
        a = build_assertion({"type": "must_not_call", "name": "delete_record"})
        assert isinstance(a, DidNotCallTool)
        assert a.tool_name == "delete_record"

    def test_build_tool_order(self):
        a = build_assertion({"type": "tool_order", "sequence": ["search", "summarize"]})
        assert isinstance(a, ToolOrder)
        assert a.sequence == ["search", "summarize"]

    def test_build_no_tool_loops(self):
        a = build_assertion({"type": "no_tool_loops"})
        assert isinstance(a, NoToolLoops)

    def test_build_max_cost_usd(self):
        a = build_assertion({"type": "max_cost_usd", "max": 0.05})
        assert isinstance(a, MaxCost)
        assert a.max_usd == pytest.approx(0.05)

    def test_build_call_count(self):
        a = build_assertion({"type": "call_count", "name": "search", "count": 1})
        assert isinstance(a, CallCount)
        assert a.tool_name == "search"
        assert a.count == 1
