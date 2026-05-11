"""
Readable registry for the model structure.

This file maps the task folders without importing the numerical modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ModelModule:
    task: str
    role: str
    primary_entry: Path
    connects_to_task1_as: str
    output_interface: Optional[str] = None


MODEL_MODULES = [
    ModelModule(
        task="Task 1",
        role="Backbone dynamic franchise optimization model",
        primary_entry=Path("project/model/task1/pipeline.py"),
        connects_to_task1_as="backbone",
        output_interface="period_log and summary tables",
    ),
    ModelModule(
        task="Task 2",
        role="Roster and transfer-market optimization",
        primary_entry=Path("project/model/task2/pipeline.py"),
        connects_to_task1_as="initial state provider",
        output_interface="project/data/processed/task2/task1_initial_state.json",
    ),
    ModelModule(
        task="Task 3",
        role="League expansion impact analysis",
        primary_entry=Path("project/model/task3/task1_impact.py"),
        connects_to_task1_as="exogenous shock provider",
        output_interface="project/model/task3/outputs/task3_to_task1_<team>_<city>.json",
    ),
    ModelModule(
        task="Task 4",
        role="Dynamic ticket pricing and attendance feedback",
        primary_entry=Path("project/model/task4/mpc_integration.py"),
        connects_to_task1_as="pricing and revenue extension",
        output_interface="period pricing, ticket revenue, attendance-adjusted home advantage",
    ),
    ModelModule(
        task="Task 5",
        role="Superstar injury shock and resilience analysis",
        primary_entry=Path("project/model/task5/impact_analysis.py"),
        connects_to_task1_as="risk and shock extension",
        output_interface="delta J, win loss, revenue loss, valuation loss, risk change",
    ),
]


def format_model_map() -> str:
    lines = ["Task1-backed model structure", ""]
    for module in MODEL_MODULES:
        lines.extend(
            [
                f"{module.task}: {module.role}",
                f"  Entry: {module.primary_entry}",
                f"  Task1 link: {module.connects_to_task1_as}",
            ]
        )
        if module.output_interface:
            lines.append(f"  Interface: {module.output_interface}")
        lines.append("")
    return "\n".join(lines).rstrip()


if __name__ == "__main__":
    print(format_model_map())
