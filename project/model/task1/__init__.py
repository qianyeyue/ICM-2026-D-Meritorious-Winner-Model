"""Task 1 backbone interface."""


def __getattr__(name):
    if name == "Task1RunConfig":
        from .pipeline import Task1RunConfig

        return Task1RunConfig
    if name == "run_task1":
        from .pipeline import run_task1

        return run_task1
    if name == "run_task1_with_task2_state":
        from .pipeline import run_task1_with_task2_state

        return run_task1_with_task2_state
    if name == "run_task1_with_task3_shock":
        from .pipeline import run_task1_with_task3_shock

        return run_task1_with_task3_shock
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "Task1RunConfig",
    "run_task1",
    "run_task1_with_task2_state",
    "run_task1_with_task3_shock",
]
