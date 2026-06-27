from abc import ABC, abstractmethod

from ..trajectory import Trajectory


class BaseAdapter(ABC):
    """Contract every agent adapter must satisfy."""

    @abstractmethod
    def run(self, agent_input: str, scenario_name: str) -> Trajectory:
        """Execute the agent for *agent_input* and return a captured Trajectory."""
        ...
