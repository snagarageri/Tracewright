from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from .trajectory import Trajectory


@dataclass
class AssertionResult:
    passed: bool
    message: str
    assertion_type: str


class Assertion(ABC):
    @abstractmethod
    def check(self, trajectory: Trajectory) -> AssertionResult:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class NoError(Assertion):
    @property
    def name(self) -> str:
        return "no_error"

    def check(self, trajectory: Trajectory) -> AssertionResult:
        if trajectory.error:
            return AssertionResult(False, f"Agent raised an error: {trajectory.error}", self.name)
        return AssertionResult(True, "Agent completed without errors", self.name)


class OutputContains(Assertion):
    def __init__(self, value: str, case_sensitive: bool = False):
        self.value = value
        self.case_sensitive = case_sensitive

    @property
    def name(self) -> str:
        return "output_contains"

    def check(self, trajectory: Trajectory) -> AssertionResult:
        output = trajectory.final_output or ""
        haystack = output if self.case_sensitive else output.lower()
        needle = self.value if self.case_sensitive else self.value.lower()
        passed = needle in haystack
        msg = (
            f"Output contains '{self.value}'"
            if passed
            else f"Output does not contain '{self.value}'. Got: {output!r}"
        )
        return AssertionResult(passed, msg, self.name)


class OutputMatches(Assertion):
    def __init__(self, pattern: str, flags: int = re.IGNORECASE):
        self.pattern = pattern
        self.flags = flags

    @property
    def name(self) -> str:
        return "output_matches"

    def check(self, trajectory: Trajectory) -> AssertionResult:
        output = trajectory.final_output or ""
        match = re.search(self.pattern, output, self.flags)
        passed = match is not None
        msg = (
            f"Output matches pattern '{self.pattern}'"
            if passed
            else f"Output does not match pattern '{self.pattern}'. Got: {output!r}"
        )
        return AssertionResult(passed, msg, self.name)


class CalledTool(Assertion):
    def __init__(self, tool_name: str):
        self.tool_name = tool_name

    @property
    def name(self) -> str:
        return "called_tool"

    def check(self, trajectory: Trajectory) -> AssertionResult:
        called = self.tool_name in trajectory.tool_names
        msg = (
            f"Tool '{self.tool_name}' was called"
            if called
            else f"Tool '{self.tool_name}' was not called. Called tools: {trajectory.tool_names}"
        )
        return AssertionResult(called, msg, self.name)


class ToolCalledWith(Assertion):
    def __init__(self, tool_name: str, expected_args: dict[str, Any]):
        self.tool_name = tool_name
        self.expected_args = expected_args

    @property
    def name(self) -> str:
        return "tool_called_with"

    def check(self, trajectory: Trajectory) -> AssertionResult:
        matching_calls = [tc for tc in trajectory.tool_calls if tc.name == self.tool_name]
        if not matching_calls:
            return AssertionResult(
                False,
                f"Tool '{self.tool_name}' was never called",
                self.name,
            )
        for tc in matching_calls:
            if all(tc.args.get(k) == v for k, v in self.expected_args.items()):
                return AssertionResult(
                    True,
                    f"Tool '{self.tool_name}' called with expected args",
                    self.name,
                )
        actual = [tc.args for tc in matching_calls]
        return AssertionResult(
            False,
            f"Tool '{self.tool_name}' was called but not with {self.expected_args}. Actual calls: {actual}",
            self.name,
        )


class StepCount(Assertion):
    def __init__(self, min_steps: Optional[int] = None, max_steps: Optional[int] = None):
        self.min_steps = min_steps
        self.max_steps = max_steps

    @property
    def name(self) -> str:
        return "step_count"

    def check(self, trajectory: Trajectory) -> AssertionResult:
        count = len(trajectory.steps)
        if self.min_steps is not None and count < self.min_steps:
            return AssertionResult(
                False,
                f"Expected at least {self.min_steps} steps, got {count}",
                self.name,
            )
        if self.max_steps is not None and count > self.max_steps:
            return AssertionResult(
                False,
                f"Expected at most {self.max_steps} steps, got {count}",
                self.name,
            )
        return AssertionResult(True, f"Step count {count} is within bounds", self.name)


class LLMJudge(Assertion):
    def __init__(self, criteria: str, model: str = "gpt-4o-mini"):
        self.criteria = criteria
        self.model = model

    @property
    def name(self) -> str:
        return "llm_judge"

    def check(self, trajectory: Trajectory) -> AssertionResult:
        try:
            from openai import OpenAI
        except ImportError:
            return AssertionResult(
                False,
                "openai package is required for llm_judge assertions",
                self.name,
            )

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return AssertionResult(
                False,
                "OPENAI_API_KEY environment variable not set — cannot run llm_judge",
                self.name,
            )

        steps_summary = "\n".join(
            f"  [{s.type}] {s.content[:200]}" for s in trajectory.steps
        )
        prompt = f"""You are evaluating an AI agent's trajectory.

Task given to agent: {trajectory.agent_input}

Steps taken:
{steps_summary if steps_summary else "  (no steps recorded)"}

Final output: {trajectory.final_output or "(none)"}

Evaluation criteria: {self.criteria}

Did the agent meet this criteria? Respond with JSON only:
{{"passed": true/false, "reason": "brief explanation"}}"""

        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or "{}"
            data = json.loads(raw)
            passed = bool(data.get("passed", False))
            reason = data.get("reason", "No reason provided")
            return AssertionResult(passed, f"LLM judge: {reason}", self.name)
        except Exception as exc:
            return AssertionResult(False, f"LLM judge failed: {exc}", self.name)


_ASSERTION_REGISTRY: dict[str, type[Assertion]] = {
    "no_error": NoError,
    "output_contains": OutputContains,
    "output_matches": OutputMatches,
    "called_tool": CalledTool,
    "tool_called_with": ToolCalledWith,
    "step_count": StepCount,
    "llm_judge": LLMJudge,
}


def build_assertion(spec: dict[str, Any]) -> Assertion:
    atype = spec.get("type")
    if atype not in _ASSERTION_REGISTRY:
        known = ", ".join(_ASSERTION_REGISTRY)
        raise ValueError(f"Unknown assertion type '{atype}'. Known types: {known}")

    if atype == "no_error":
        return NoError()
    if atype == "output_contains":
        return OutputContains(spec["value"], spec.get("case_sensitive", False))
    if atype == "output_matches":
        return OutputMatches(spec["pattern"])
    if atype == "called_tool":
        return CalledTool(spec["name"])
    if atype == "tool_called_with":
        return ToolCalledWith(spec["name"], spec.get("args", {}))
    if atype == "step_count":
        return StepCount(spec.get("min"), spec.get("max"))
    if atype == "llm_judge":
        return LLMJudge(spec["criteria"], spec.get("model", "gpt-4o-mini"))

    raise ValueError(f"Unhandled assertion type: {atype}")
