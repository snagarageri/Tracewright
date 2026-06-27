from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    error: Optional[str] = None
    latency_ms: Optional[float] = None
    cost_usd: Optional[float] = None


class Step(BaseModel):
    index: int
    type: str  # "thought" | "tool_call" | "observation" | "final_answer"
    content: str
    tool_call: Optional[ToolCall] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Trajectory(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    scenario_name: str
    agent_input: str
    steps: list[Step] = Field(default_factory=list)
    final_output: Optional[str] = None
    started_at: datetime = Field(default_factory=datetime.utcnow)
    finished_at: Optional[datetime] = None
    error: Optional[str] = None

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        return None

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [s.tool_call for s in self.steps if s.tool_call is not None]

    @property
    def tool_names(self) -> list[str]:
        return [tc.name for tc in self.tool_calls]

    @property
    def thoughts(self) -> list[str]:
        return [s.content for s in self.steps if s.type == "thought"]

    def calls_to(self, tool_name: str) -> list[ToolCall]:
        return [tc for tc in self.tool_calls if tc.name == tool_name]

    def has_tool_loop(self) -> bool:
        calls = self.tool_calls
        for i in range(len(calls) - 1):
            if calls[i].name == calls[i + 1].name and calls[i].args == calls[i + 1].args:
                return True
        return False

    def total_cost_usd(self) -> float:
        return sum(tc.cost_usd for tc in self.tool_calls if tc.cost_usd is not None)

    def total_latency_ms(self) -> float:
        return sum(tc.latency_ms for tc in self.tool_calls if tc.latency_ms is not None)

    def step_count(self) -> int:
        return len(self.steps)

    def first_tool_call(self) -> Optional[ToolCall]:
        return next(iter(self.tool_calls), None)
