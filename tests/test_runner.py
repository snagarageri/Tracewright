from pathlib import Path

import pytest
import yaml

from tracewright.adapters.generic import GenericAdapter, TrajectoryRecorder
from tracewright.assertions import CalledTool, NoError, OutputContains
from tracewright.runner import Runner, ScenarioSpec, _build_adapter, _load_agent
from tracewright.trajectory import Trajectory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def simple_agent(task: str, recorder: TrajectoryRecorder) -> str:
    recorder.thought(f"Processing: {task}")
    recorder.tool_call("echo", {"input": task}, result=task)
    recorder.observation(task)
    return f"result: {task}"


def erroring_agent(task: str, recorder: TrajectoryRecorder) -> str:
    raise RuntimeError("agent failed intentionally")


# ---------------------------------------------------------------------------
# GenericAdapter
# ---------------------------------------------------------------------------

class TestGenericAdapter:
    def test_records_steps_and_output(self):
        adapter = GenericAdapter(simple_agent)
        traj = adapter.run("hello", "test")
        assert traj.final_output == "result: hello"
        assert traj.error is None
        assert len(traj.steps) >= 3
        assert "echo" in traj.tool_names

    def test_captures_agent_exception(self):
        adapter = GenericAdapter(erroring_agent)
        traj = adapter.run("boom", "error test")
        assert traj.error is not None
        assert "intentionally" in traj.error
        assert traj.final_output is None

    def test_finished_at_set(self):
        adapter = GenericAdapter(simple_agent)
        traj = adapter.run("hi", "t")
        assert traj.finished_at is not None
        assert traj.duration_seconds is not None


# ---------------------------------------------------------------------------
# TrajectoryRecorder
# ---------------------------------------------------------------------------

class TestTrajectoryRecorder:
    def test_thought_step(self):
        rec = TrajectoryRecorder("s", "i")
        rec.thought("thinking")
        assert rec.trajectory.steps[0].type == "thought"
        assert rec.trajectory.steps[0].content == "thinking"

    def test_tool_call_step(self):
        rec = TrajectoryRecorder("s", "i")
        rec.tool_call("search", {"q": "AI"}, result="found")
        step = rec.trajectory.steps[0]
        assert step.type == "tool_call"
        assert step.tool_call.name == "search"
        assert step.tool_call.result == "found"

    def test_observation_step(self):
        rec = TrajectoryRecorder("s", "i")
        rec.observation(42)
        assert rec.trajectory.steps[0].type == "observation"
        assert rec.trajectory.steps[0].content == "42"

    def test_step_indexes_increment(self):
        rec = TrajectoryRecorder("s", "i")
        rec.thought("a")
        rec.thought("b")
        rec.thought("c")
        assert [s.index for s in rec.trajectory.steps] == [0, 1, 2]


# ---------------------------------------------------------------------------
# Runner + assertions pipeline
# ---------------------------------------------------------------------------

class TestRunner:
    def _run_with_agent(self, agent_fn, assertion_specs):
        runner = Runner(store=None)
        adapter = GenericAdapter(agent_fn)
        scenario = ScenarioSpec("test scenario", "hello", assertion_specs)
        traj, results = runner._run_scenario(scenario, adapter)
        return traj, results

    def test_no_error_passes(self):
        _, results = self._run_with_agent(
            simple_agent, [{"type": "no_error"}]
        )
        assert all(r.passed for r in results)

    def test_output_contains_passes(self):
        _, results = self._run_with_agent(
            simple_agent, [{"type": "output_contains", "value": "result"}]
        )
        assert results[0].passed

    def test_called_tool_passes(self):
        _, results = self._run_with_agent(
            simple_agent, [{"type": "called_tool", "name": "echo"}]
        )
        assert results[0].passed

    def test_error_agent_fails_no_error_assertion(self):
        _, results = self._run_with_agent(
            erroring_agent, [{"type": "no_error"}]
        )
        assert not results[0].passed

    def test_multiple_assertions_all_run(self):
        _, results = self._run_with_agent(
            simple_agent,
            [
                {"type": "no_error"},
                {"type": "output_contains", "value": "result"},
                {"type": "called_tool", "name": "echo"},
                {"type": "called_tool", "name": "does_not_exist"},
            ],
        )
        assert results[0].passed   # no_error
        assert results[1].passed   # output_contains
        assert results[2].passed   # called echo
        assert not results[3].passed  # missing tool


# ---------------------------------------------------------------------------
# run_spec integration (uses temp YAML + agent file)
# ---------------------------------------------------------------------------

class TestRunSpec:
    def test_run_spec_all_pass(self, tmp_path):
        agent_code = """
from tracewright.adapters.generic import TrajectoryRecorder

def create_agent():
    def agent(task: str, recorder: TrajectoryRecorder) -> str:
        recorder.thought("processing")
        recorder.tool_call("reverse", {"text": task}, result=task[::-1])
        return task[::-1]
    return agent
"""
        spec = {
            "agent": {"module": "mock_agent", "factory": "create_agent", "adapter": "generic"},
            "scenarios": [
                {
                    "name": "reverse test",
                    "input": "hello",
                    "assertions": [
                        {"type": "no_error"},
                        {"type": "output_contains", "value": "olleh"},
                        {"type": "called_tool", "name": "reverse"},
                    ],
                }
            ],
        }
        (tmp_path / "mock_agent.py").write_text(agent_code)
        (tmp_path / "tests.yaml").write_text(yaml.dump(spec))

        runner = Runner(store=None)
        passed, failed = runner.run_spec(tmp_path / "tests.yaml")

        assert passed == 1
        assert failed == 0

    def test_run_spec_partial_fail(self, tmp_path):
        agent_code = """
from tracewright.adapters.generic import TrajectoryRecorder

def create_agent():
    def agent(task: str, recorder: TrajectoryRecorder) -> str:
        return "ok"
    return agent
"""
        spec = {
            "agent": {"module": "mock_agent2", "factory": "create_agent", "adapter": "generic"},
            "scenarios": [
                {
                    "name": "expects missing tool",
                    "input": "hi",
                    "assertions": [
                        {"type": "no_error"},
                        {"type": "called_tool", "name": "nonexistent"},
                    ],
                }
            ],
        }
        (tmp_path / "mock_agent2.py").write_text(agent_code)
        (tmp_path / "tests.yaml").write_text(yaml.dump(spec))

        runner = Runner(store=None)
        passed, failed = runner.run_spec(tmp_path / "tests.yaml")

        assert passed == 0
        assert failed == 1
