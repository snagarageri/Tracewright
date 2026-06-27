"""
Generic example agent — no external dependencies required.

The agent handles three simple commands:
  - "uppercase <text>"   → uppercases the text
  - "count <text>"       → counts the words
  - anything else        → echoes the input back

Each operation is recorded as a tool call so Tracewright can assert on the
trajectory (not just the final output).
"""

from tracewright.adapters.generic import TrajectoryRecorder


def create_agent():
    """Factory called by the runner to produce a fresh agent per scenario."""

    def agent(task: str, recorder: TrajectoryRecorder) -> str:
        recorder.thought(f"Received task: {task!r}")

        lower = task.lower()

        if lower.startswith("uppercase"):
            text = task[len("uppercase"):].strip() or task
            recorder.tool_call("text_transform", {"operation": "upper", "text": text})
            result = text.upper()
            recorder.observation(result)

        elif lower.startswith("count"):
            text = task[len("count"):].strip() or task
            recorder.tool_call("word_count", {"text": text})
            count = len(text.split())
            result = str(count)
            recorder.observation(result)

        else:
            recorder.tool_call("echo", {"input": task})
            result = task
            recorder.observation(result)

        recorder.final_answer(result)
        return result

    return agent
