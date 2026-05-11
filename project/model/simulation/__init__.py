"""Simulation utilities for the public model release."""

def __getattr__(name):
    if name == "MPCConfig":
        from .mpc_simulation import MPCConfig
        return MPCConfig
    elif name == "ControlAction":
        from .mpc_simulation import ControlAction
        return ControlAction
    elif name == "TeamState":
        from .mpc_simulation import TeamState
        return TeamState
    elif name == "run_mpc_simulation":
        from .mpc_simulation import run_mpc_simulation
        return run_mpc_simulation
    elif name == "run_multi_year_mpc_simulation":
        from .mpc_simulation import run_multi_year_mpc_simulation
        return run_multi_year_mpc_simulation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "MPCConfig",
    "ControlAction",
    "TeamState",
    "run_mpc_simulation",
    "run_multi_year_mpc_simulation",
]
