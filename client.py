"""Client utilities for interacting with the HelpdeskEnv environment.

For most use cases you do not need a separate HTTP client --
start the FastAPI server and use /reset, /step, /state directly,
or use the underlying primitives from the helpdeskenv package.
"""

__all__ = ["HelpdeskEnv"]
