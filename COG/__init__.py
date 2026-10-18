"""lol_teamfight package

Refactored pipeline modules for the LoL teamfight outcome prediction project.

Core convention
---------------
- `FightRef.t_start` is a **minute index** into `cache["minute_ts"]`.

This package exists to keep the root scripts small and to make it easier to
unit-test and reuse the pipeline pieces.
"""

from __future__ import annotations

__all__ = [
    "__version__",
]

__version__ = "0.1.0"
