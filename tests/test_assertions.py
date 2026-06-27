import pytest

from tracewright.assertions import (
    CalledTool,
    LLMJudge,
    NoError,
    OutputContains,
    OutputMatches,
    StepCount,
    ToolCalledWith,
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
