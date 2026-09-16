"""Validity radius from the patch's own ability ranges (Data Dragon).

The presence gate asks whether champions stood within R of the first kill at
the cutoff.  R is not observable from Match-V5 (non-participants' positions
at kill time are unknown), so it is anchored in game mechanics: the radius
that covers a target share of all ability casts and basic attacks.  This is
the same construction as Schubert et al. (2016), who set the combat range at
the distance covering ~85% of observed damage; here the source is the
patch's Data Dragon spell data, so R can be recomputed per patch.

Network access is needed once per patch; results are cached as JSON.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

DDRAGON_FULL = "https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/championFull.json"
DDRAGON_VERSIONS = "https://ddragon.leagueoflegends.com/api/versions.json"
GLOBAL_RANGE = 25_000  # Data Dragon's sentinel for global / self-cast spells


def version_for_patch(patch: str, versions: Optional[list] = None) -> str:
    """'15.14' -> the Data Dragon version with that prefix (e.g. '15.14.1')."""
    if versions:
        for v in versions:
            if v.startswith(patch + "."):
                return v
    return f"{patch}.1"


def fetch_json(url: str, timeout: float = 30.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def load_champion_full(patch: str, cache_dir: Optional[Path] = None) -> tuple:
    """Return (version, championFull dict); cached on disk when cache_dir is given."""
    versions = None
    try:
        versions = fetch_json(DDRAGON_VERSIONS)
    except Exception:
        pass
    version = version_for_patch(patch, versions)
    if cache_dir is not None:
        cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
        cached = cache_dir / f"championFull_{version}.json"
        if cached.exists():
            return version, json.loads(cached.read_text(encoding="utf-8"))
    data = fetch_json(DDRAGON_FULL.format(version=version))
    if cache_dir is not None:
        (Path(cache_dir) / f"championFull_{version}.json").write_text(json.dumps(data), encoding="utf-8")
    return version, data


def range_table(champion_full: dict) -> dict:
    """Max-rank range of every spell and every basic attack range."""
    spells, attacks, n_champ = [], [], 0
    for champ in champion_full.get("data", {}).values():
        n_champ += 1
        try:
            attacks.append(float(champ["stats"]["attackrange"]))
        except Exception:
            pass
        for sp in champ.get("spells", []) or []:
            rng = sp.get("range")
            if isinstance(rng, list) and rng:
                try:
                    r = float(max(rng))
                except Exception:
                    continue
                if r > 0:
                    spells.append(r)
    return {"n_champions": n_champ, "spell_ranges": np.asarray(spells), "attack_ranges": np.asarray(attacks)}


def coverage_at(ranges: np.ndarray, r: float) -> float:
    return float(np.mean(ranges <= r)) if ranges.size else float("nan")


def radius_for_coverage(ranges: np.ndarray, q: float, exclude_global: bool = True) -> float:
    x = ranges[ranges < GLOBAL_RANGE] if exclude_global else ranges
    return float(np.quantile(x, q)) if x.size else float("nan")


def estimate_validity_radius(patch: str, reference_radius: float = 1800.0, coverage_target: float = 0.90,
                             cache_dir: Optional[Path] = None) -> dict:
    """Per-patch check of the validity radius.

    R itself stays the mechanics anchor (``reference_radius``); what a patch
    can change is *how much* of the ability-range distribution that radius
    covers.  The report gives the coverage at the reference radius (globals in
    the denominator, as in CONSTANTS_JUSTIFICATION.md) and, for information,
    the spell-range quantile at ``coverage_target`` with and without global
    spells.  Drift is judged on the coverage rather than by moving R, because
    the quantile is very sensitive to whether globals are counted (at 0.90:
    about 1,250 u without them, about 2,000 u with them).
    """
    try:
        version, data = load_champion_full(patch, cache_dir)
    except Exception as e:
        return {"ok": False, "patch": patch, "reason": f"datadragon unavailable: {e}"}
    t = range_table(data)
    spells = t["spell_ranges"]
    return {"ok": bool(spells.size), "patch": patch, "version": version, "n_champions": t["n_champions"],
            "n_spells": int(spells.size), "radius_u": float(reference_radius),
            "coverage_at_reference": coverage_at(spells, reference_radius), "reference_radius_u": reference_radius,
            "quantile_all_spells_at_target": radius_for_coverage(spells, coverage_target, exclude_global=False),
            "quantile_nonglobal_at_target": radius_for_coverage(spells, coverage_target, exclude_global=True),
            "coverage_target": coverage_target,
            "attack_range_max": float(t["attack_ranges"].max()) if t["attack_ranges"].size else None,
            "spell_range_median": float(np.median(spells)) if spells.size else None,
            "global_share": float(np.mean(spells >= GLOBAL_RANGE)) if spells.size else None}
