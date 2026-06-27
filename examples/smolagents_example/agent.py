"""
smolagents example — requires: pip install tracewright[smolagents]

Uses a ToolCallingAgent with a mock weather tool so the example runs without
any real API key. Swap HfApiModel for the model of your choice and replace
get_weather with real tools to go further.
"""

try:
    from smolagents import HfApiModel, ToolCallingAgent, tool
except ImportError as exc:
    raise ImportError(
        "smolagents is not installed. Run: pip install tracewright[smolagents]"
    ) from exc


@tool
def get_weather(city: str) -> str:
    """Return the current weather for *city* (mock implementation).

    Args:
        city: Name of the city to look up.
    """
    conditions = {"paris": "Sunny, 22°C", "london": "Cloudy, 15°C", "tokyo": "Rainy, 18°C"}
    return conditions.get(city.lower(), f"Partly cloudy, 20°C in {city}")


def create_agent():
    """Factory called by the runner to produce a fresh agent per scenario."""
    model = HfApiModel()
    return ToolCallingAgent(tools=[get_weather], model=model, max_steps=5)
