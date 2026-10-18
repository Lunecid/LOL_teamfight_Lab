"""The boundary spec: one versioned record that *is* the fight definition.

A ``BoundarySpec`` holds every number the detector and the scale classes need,
with where each came from (estimated from this corpus slice, inherited from
the pooled estimate, or a mechanics default).  It serialises to JSON, and
``detector_overrides()`` turns it into the config keys of the kill-cluster
detector, so re-running the pipeline on a new patch re-parameterises the
detector without touching code.

Scale classes are just thresholds on the smaller side's participant count:
pick (min <= pick_max_min), skirmish (up to skirmish_max_min), teamfight
(above).  They are reporting axes, never inputs to a model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

DEFAULTS = {"gap_s": 18.0, "diameter_u": 4000.0, "validity_radius_u": 1800.0, "lead_s": 10.0, "min_per_team": 2}


@dataclass
class BoundarySpec:
    scope: str                                  # "patch:15.14" | "pooled"
    patches: List[str]
    n_matches: int
    n_intervals: int
    n_pairs_in_window: int

    # temporal
    gap_s: float
    gap_source: str                             # "valley" | "pooled" | "default"
    gap_ci_s: Optional[List[float]] = None
    gap_depth: Optional[float] = None
    gap_ok: bool = False
    gap_plateau_s: Optional[List[float]] = None  # contiguous gap band with ARI >= plateau_ari vs gap_s
    plateau_ari: float = 0.90

    # spatial
    diameter_u: float = DEFAULTS["diameter_u"]
    diameter_source: str = "default"            # "crossover" | "pooled" | "default"
    diameter_ci_u: Optional[List[float]] = None
    diameter_mass_near: Optional[float] = None  # share of in-window pairs within +-10% of D

    # presence gate
    validity_radius_u: float = DEFAULTS["validity_radius_u"]
    radius_source: str = "default"              # "datadragon:<version>" | "default"
    radius_coverage: Optional[float] = None
    min_per_team: int = DEFAULTS["min_per_team"]

    # cutoff
    lead_s: float = DEFAULTS["lead_s"]
    lead_source: str = "fixed"                  # no data-driven estimate yet

    # scale classes
    pick_max_min: int = 1
    skirmish_max_min: int = 3

    provenance: Dict = field(default_factory=dict)

    # ---- behaviour ----
    def detector_overrides(self) -> Dict[str, float]:
        return {"TF2_KILL_CLUSTER_GAP_MS": int(round(self.gap_s * 1000)),
                "CLUSTER_MAX_DIAMETER": float(self.diameter_u),
                "TF2_VALIDITY_RADIUS": float(self.validity_radius_u),
                "TF2_ENGAGE_PRE_KILL_MS": int(round(self.lead_s * 1000)),
                "TF2_MIN_PER_TEAM": int(self.min_per_team)}

    def scale_class(self, blue, red) -> np.ndarray:
        b, r = np.asarray(blue, dtype=int), np.asarray(red, dtype=int)
        mn = np.minimum(b, r)
        out = np.full(mn.shape, "teamfight", dtype=object)
        out[mn <= self.skirmish_max_min] = "skirmish"
        out[mn <= self.pick_max_min] = "pick"
        out[(b < 0) | (r < 0)] = "unknown"
        return out.astype(str)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")

    @classmethod
    def from_dict(cls, d: dict) -> "BoundarySpec":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, path) -> "BoundarySpec":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def summary(self) -> str:
        ci = f" [{self.gap_ci_s[0]:.1f}, {self.gap_ci_s[1]:.1f}]" if self.gap_ci_s else ""
        pl = f" plateau {self.gap_plateau_s[0]:g}-{self.gap_plateau_s[1]:g}" if self.gap_plateau_s else ""
        dci = f" [{self.diameter_ci_u[0]:.0f}, {self.diameter_ci_u[1]:.0f}]" if self.diameter_ci_u else ""
        return (f"{self.scope}: G={self.gap_s:.1f}s{ci}{pl} ({self.gap_source}); "
                f"D={self.diameter_u:.0f}u{dci} ({self.diameter_source}); "
                f"R={self.validity_radius_u:.0f}u ({self.radius_source}); "
                f"scale pick<={self.pick_max_min} skirmish<={self.skirmish_max_min} teamfight>{self.skirmish_max_min}")
