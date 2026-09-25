"""Champion attribute vectors from the Data Dragon v2 tables (v4-exact, StateV3 player block).

Source: ``config/game_rules/datadragon_v2/champions_<patch>.json`` (built by
``scripts/exact_v4/ev4_00_fetch_ddragon.py``).  Only static, pre-game information is used: a
champion's attributes do not depend on the query time, so there is no time argument and no
frame / event access here.

Champion IDs: ``meta['static_meta']['champion_by_pid']`` is the source of truth (it is what
``pipeline_cache`` writes into the ``node_minute`` ``champion_id`` column, see
``pipeline_cache.py:340``).  ``champion_ids_from_meta`` optionally cross-checks it against the
first ``node_minute`` frame (t = 0, draft information) and raises on disagreement.

Vector layout (``CHAMPION_VECTOR_FIELDS``, all floats):
  champ_unknown                     1.0 when the ID is not in the table (every other field NaN)
  tag_<Assassin|Fighter|Mage|Marksman|Support|Tank>   class-tag indicators
  is_ranged, attackrange            is_ranged = attackrange > 300 (Rakan 300 -> melee)
  partype_<Mana|Energy|none|other>  one-hot; 'None' and '' -> none, any other resource -> other
  base_<healthMax|armor|magicResist|attackDamage|attackSpeed|movementSpeed>
  growth_<healthMax|armor|magicResist|attackDamage|attackSpeed_pct>
        (per level; attack-speed growth is a percent per level as in Data Dragon.  Data Dragon has
         no per-level movement-speed growth, so no growth_movementSpeed column is emitted.)
  info_<attack|defense|magic|difficulty>
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping, Optional

import numpy as np

DD2_DIR = Path(__file__).resolve().parents[1] / "config" / "game_rules" / "datadragon_v2"
MODE_KEY_MIN = 10000  # 16.15 lists alternate-mode champions (e.g. 60001 'Jade_Annie'); keep key < 10000

CLASS_TAGS = ("Assassin", "Fighter", "Mage", "Marksman", "Support", "Tank")
PARTYPE_CATEGORIES = ("Mana", "Energy", "none", "other")
RANGED_THRESHOLD = 300.0
BASE_STATS = ("healthMax", "armor", "magicResist", "attackDamage", "attackSpeed", "movementSpeed")
GROWTH_STATS = ("healthMax", "armor", "magicResist", "attackDamage", "attackSpeed_pct")
INFO_FIELDS = ("attack", "defense", "magic", "difficulty")

CHAMPION_VECTOR_FIELDS = (
    ("champ_unknown",)
    + tuple(f"tag_{t}" for t in CLASS_TAGS)
    + ("is_ranged", "attackrange")
    + tuple(f"partype_{p}" for p in PARTYPE_CATEGORIES)
    + tuple(f"base_{s}" for s in BASE_STATS)
    + tuple(f"growth_{s}" for s in GROWTH_STATS)
    + tuple(f"info_{s}" for s in INFO_FIELDS)
)


@dataclass(frozen=True)
class ChampionTable:
    patch: str
    version: str
    champions: Mapping[int, Mapping]  # key -> read-only record
    by_name: Mapping[str, int] = field(default_factory=dict)  # Data Dragon id name (e.g. 'MonkeyKing') -> key
    dropped_mode_keys: tuple = ()

    def __contains__(self, champion_id) -> bool:
        return _as_key(champion_id) in self.champions

    def __len__(self) -> int:
        return len(self.champions)


def _freeze(obj):
    if isinstance(obj, dict):
        return MappingProxyType({k: _freeze(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return tuple(_freeze(v) for v in obj)
    return obj


def _as_key(champion_id) -> Optional[int]:
    """int key for a valid champion id, None for missing / non-integral / non-positive / mode ids."""
    if champion_id is None or isinstance(champion_id, bool):
        return None
    try:
        f = float(champion_id)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f) or f != int(f) or f <= 0 or f >= MODE_KEY_MIN:
        return None
    return int(f)


@lru_cache(maxsize=None)
def _load_cached(patch: str, dd_dir: str) -> ChampionTable:
    path = Path(dd_dir) / f"champions_{patch}.json"
    if not path.exists():
        raise FileNotFoundError(f"no Data Dragon v2 champion table for patch {patch!r}: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if str(raw.get("patch")) != patch:
        raise ValueError(f"{path.name}: patch field {raw.get('patch')!r} != {patch!r}")
    champs, dropped = {}, []
    for k, rec in raw["champions"].items():
        key = int(k)
        if int(rec.get("key", key)) != key:
            raise ValueError(f"{path.name}: key mismatch for {k}")
        if key >= MODE_KEY_MIN:
            dropped.append(key)
            continue
        bad = set(rec.get("tags") or ()) - set(CLASS_TAGS)
        if bad:
            raise ValueError(f"{path.name}: {rec.get('name')} has non-class tags {sorted(bad)}")
        champs[key] = _freeze(rec)
    by_name = {}
    for key, rec in champs.items():
        for nm in {rec.get("name"), rec.get("display_name")} - {None, ""}:
            by_name.setdefault(_norm_name(nm), key)
    return ChampionTable(patch=patch, version=str(raw.get("version")),
                         champions=MappingProxyType(champs), by_name=MappingProxyType(by_name),
                         dropped_mode_keys=tuple(sorted(dropped)))


def _norm_name(name) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def load_champion_table_v2(patch: str, dd_dir: Optional[Path] = None) -> ChampionTable:
    """Read-only champion table for ``patch`` ('15.14', '16.13', ...), keys < 10000 only (cached)."""
    return _load_cached(str(patch), str(Path(dd_dir) if dd_dir is not None else DD2_DIR))


def _partype_category(partype) -> str:
    p = "" if partype is None else str(partype).strip()
    if p in ("Mana", "Energy"):
        return p
    if p in ("", "None"):
        return "none"
    return "other"


def _num(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def unknown_champion_vector() -> dict:
    out = {f: float("nan") for f in CHAMPION_VECTOR_FIELDS}
    out["champ_unknown"] = 1.0
    return out


def _records(table) -> Mapping:
    return table.champions if isinstance(table, ChampionTable) else table


def champion_vector(champion_id, table) -> dict:
    """Ordered dict (``CHAMPION_VECTOR_FIELDS``) of floats for one champion.

    ``table`` is a ``ChampionTable`` or a plain ``{int key: record}`` mapping.  An ID that is
    missing, non-positive, >= 10000 or absent from the table gives ``unknown_champion_vector()``
    (all NaN, ``champ_unknown`` = 1).
    """
    key = _as_key(champion_id)
    rec = _records(table).get(key) if key is not None else None
    if rec is None:
        return unknown_champion_vector()
    tags = set(rec.get("tags") or ())
    out = {"champ_unknown": 0.0}
    for t in CLASS_TAGS:
        out[f"tag_{t}"] = 1.0 if t in tags else 0.0
    ar = _num(rec.get("attackrange", (rec.get("stats") or {}).get("attackrange")))
    out["is_ranged"] = float("nan") if math.isnan(ar) else float(ar > RANGED_THRESHOLD)
    out["attackrange"] = ar
    cat = _partype_category(rec.get("partype"))
    for p in PARTYPE_CATEGORIES:
        out[f"partype_{p}"] = 1.0 if p == cat else 0.0
    base, growth, info = rec.get("base") or {}, rec.get("growth") or {}, rec.get("info") or {}
    for s in BASE_STATS:
        out[f"base_{s}"] = _num(base.get(s))
    for s in GROWTH_STATS:
        out[f"growth_{s}"] = _num(growth.get(s))
    for s in INFO_FIELDS:
        out[f"info_{s}"] = _num(info.get(s))
    return {f: out[f] for f in CHAMPION_VECTOR_FIELDS}


def champion_matrix(champion_ids: Iterable, table) -> np.ndarray:
    """(n, len(CHAMPION_VECTOR_FIELDS)) float64 array, one row per id, in ``CHAMPION_VECTOR_FIELDS`` order."""
    rows = [list(champion_vector(c, table).values()) for c in champion_ids]
    return np.asarray(rows, dtype=np.float64).reshape(len(rows), len(CHAMPION_VECTOR_FIELDS))


def champion_ids_from_meta(meta: Mapping, node_minute=None, champion_col: Optional[int] = None) -> dict:
    """{participant id 1..10: champion id} from ``meta['static_meta']['champion_by_pid']``.

    When ``node_minute`` (T, 10, F) and ``champion_col`` are given, the first frame (t = 0) is
    cross-checked (row i is participant i + 1) and a disagreement raises ``ValueError``.
    Missing participants map to 0 (-> unknown vector).
    """
    sm = (meta or {}).get("static_meta") or {}
    cbp = sm.get("champion_by_pid") or {}
    out = {}
    for pid in range(1, 11):
        v = cbp.get(str(pid), cbp.get(pid))
        k = _as_key(v)
        out[pid] = 0 if k is None else k
    if node_minute is not None and champion_col is not None:
        nm = np.asarray(node_minute)
        if nm.ndim != 3 or nm.shape[1] != 10 or nm.shape[0] < 1:
            raise ValueError(f"unexpected node_minute shape {nm.shape}")
        for pid in range(1, 11):
            k = _as_key(nm[0, pid - 1, champion_col])
            if (0 if k is None else k) != out[pid]:
                raise ValueError(f"champion id mismatch for pid {pid}: meta {out[pid]} vs node_minute "
                                 f"{nm[0, pid - 1, champion_col]}")
    return out


def resolve_name(name, table: ChampionTable) -> Optional[int]:
    """Champion key for a Data Dragon id name or display name ('MonkeyKing' / 'Wukong'), else None."""
    return table.by_name.get(_norm_name(name))
