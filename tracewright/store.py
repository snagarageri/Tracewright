from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from .trajectory import Trajectory

_DEFAULT_DIR = Path(".tracewright")


class RegressionStore:
    """Persists failed scenarios to disk for later re-inspection or regression runs."""

    def __init__(self, store_dir: Path = _DEFAULT_DIR):
        self.store_dir = store_dir
        self.store_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def save_failure(
        self,
        trajectory: Trajectory,
        failed_assertions: list[str],
        spec_path: str,
    ) -> Path:
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        safe_name = trajectory.scenario_name.replace(" ", "_").replace("/", "-")
        filename = f"{safe_name}__{ts}.json"
        record: dict[str, Any] = {
            "scenario": trajectory.scenario_name,
            "spec_path": spec_path,
            "saved_at": datetime.utcnow().isoformat(),
            "agent_input": trajectory.agent_input,
            "final_output": trajectory.final_output,
            "error": trajectory.error,
            "failed_assertions": failed_assertions,
            "trajectory": trajectory.model_dump(mode="json"),
        }
        path = self.store_dir / filename
        path.write_text(json.dumps(record, indent=2, default=str))
        return path

    # ------------------------------------------------------------------
    def list_failures(self) -> list[dict[str, Any]]:
        records = []
        for p in sorted(self.store_dir.glob("*.json")):
            try:
                data = json.loads(p.read_text())
                records.append(
                    {
                        "file": str(p),
                        "scenario": data.get("scenario", p.stem),
                        "spec_path": data.get("spec_path", ""),
                        "saved_at": data.get("saved_at", ""),
                        "failed_assertions": data.get("failed_assertions", []),
                    }
                )
            except (json.JSONDecodeError, OSError):
                pass
        return records

    # ------------------------------------------------------------------
    def clear(self) -> int:
        removed = 0
        for p in self.store_dir.glob("*.json"):
            p.unlink()
            removed += 1
        return removed

    # ------------------------------------------------------------------
    def load_trajectory(self, file_path: Path) -> tuple[Trajectory, dict[str, Any]]:
        data = json.loads(file_path.read_text())
        traj = Trajectory.model_validate(data["trajectory"])
        return traj, data
