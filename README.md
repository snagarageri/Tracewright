# Tracewright

> **Playwright for AI agents** — test trajectories, not just outputs.

Tracewright lets you write declarative YAML test specs that verify *how* your agent
behaved (which tools it called, in what order, with what arguments) rather than just
checking the final answer.

## Quick start

```bash
pip install -e ".[dev]"
tw run examples/generic_example/tests.yaml
```

## Writing a test spec

```yaml
# tests.yaml
agent:
  module: agent          # agent.py relative to this file
  factory: create_agent  # callable that returns the agent
  adapter: generic       # "generic" or "smolagents"

scenarios:
  - name: uppercase conversion
    input: "uppercase hello world"
    assertions:
      - type: no_error
      - type: called_tool
        name: text_transform
      - type: output_contains
        value: "HELLO WORLD"
      - type: step_count
        min: 1
        max: 10
```

## Assertion types

| Type | Required keys | Description |
|---|---|---|
| `no_error` | — | Agent completed without raising |
| `output_contains` | `value` | Final output contains `value` (case-insensitive by default) |
| `output_matches` | `pattern` | Final output matches regex `pattern` |
| `called_tool` | `name` | Tool `name` was invoked at least once |
| `tool_called_with` | `name`, `args` | Tool `name` was called with at least these `args` |
| `step_count` | `min?`, `max?` | Number of recorded steps is within bounds |
| `llm_judge` | `criteria` | GPT-4o-mini evaluates `criteria` against the full trajectory |

## CLI

```
tw run <spec.yaml>                  # run all scenarios
tw run <spec.yaml> --filter <name>  # run matching scenarios only
tw run <spec.yaml> --junit out.xml  # also emit JUnit XML
tw store list                       # show stored failures
tw store clear                      # wipe the regression store
```

## Writing a generic agent

```python
# agent.py
from tracewright.adapters.generic import TrajectoryRecorder

def create_agent():
    def agent(task: str, recorder: TrajectoryRecorder) -> str:
        recorder.thought("planning my approach")
        recorder.tool_call("search", {"q": task}, result="some result")
        recorder.observation("some result")
        return "final answer"
    return agent
```

## smolagents support

```bash
pip install -e ".[smolagents]"
tw run examples/smolagents_example/tests.yaml
```

Set `adapter: smolagents` in your spec and point `factory` at a function that
returns a `CodeAgent` or `ToolCallingAgent`.
