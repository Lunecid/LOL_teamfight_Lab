"""Patch-pinned ID lookup diagnostics; unknown IDs never become absence."""
from __future__ import annotations

import json
from collections import Counter
from functools import lru_cache

import numpy as np

from studies.cache_source_audit_20261002.champion_map import load_champion_map
from studies.cache_source_audit_20261002 import config as AC
from .schema import STATIC_NAMES
from .source import REPORT, ROOT, file_sha256

TABLE_DIRECTORY = REPORT / "static_maps_verified_20261002"


@lru_cache(maxsize=3)
def lookups(patch: str) -> dict:
    if patch not in {"15.14", "15.15", "15.16"}:
        raise ValueError("lookup patch outside frozen main split")
    manifest = json.loads((TABLE_DIRECTORY / "MANIFEST.json").read_text(encoding="utf-8"))
    if manifest.get("complete") is not True:
        raise ValueError("public static source fetch incomplete")
    tables = {}
    hashes = {}
    for kind in ("summoner", "runesReforged"):
        name = f"{kind}_{patch}.1.json"
        path = TABLE_DIRECTORY / name
        if file_sha256(path) != manifest["tables"][name]["sha256"]:
            raise ValueError("public static lookup bytes changed")
        tables[kind] = json.loads(path.read_text(encoding="utf-8"))
        hashes[name] = file_sha256(path)
    item_path = ROOT / "outputs" / "reest_exact_v4_20260925" / "datadragon_raw" / f"item_{patch}.1.json"
    item_table = json.loads(item_path.read_text(encoding="utf-8"))
    if item_table.get("version") != patch + ".1":
        raise ValueError("item lookup version differs from frozen patch")
    hashes[item_path.name] = file_sha256(item_path)
    champions = load_champion_map(patch)
    if champions["load_status"] != "pass" or champions["sha256"] != AC.DATADRAGON_EXPECTED_SHA256[patch]:
        raise ValueError("frozen champion source map identity failed")
    hashes[AC.DATADRAGON_PATHS[patch].name] = champions["sha256"]
    return {"champions": frozenset(champions["id_to_names"]),
            "spells": frozenset(int(row["key"]) for row in tables["summoner"]["data"].values()),
            "styles": frozenset(style["id"] for style in tables["runesReforged"]),
            "runes": frozenset(r["id"] for s in tables["runesReforged"] for slot in s["slots"] for r in slot["runes"]),
            "items": frozenset(int(key) for key in item_table["data"]), "source_hashes": hashes}


def static_lookup_counts(static_ids: np.ndarray, bans: np.ndarray, patch: str) -> dict:
    if static_ids.shape != (10, 14) or static_ids.dtype.kind not in "iu" or bans.shape != (2, 5):
        raise ValueError("static selection shape or dtype invalid")
    source = lookups(patch)
    result = {}
    for index, name in enumerate(STATIC_NAMES):
        allowed = source["champions"] if name == "champion_id" else (
            source["spells"] if name.startswith("summoner_") else source["styles"] if "style" in name else
            None if name.startswith("stat_perk_") else source["runes"])
        counts = Counter()
        for value in static_ids[:, index].tolist():
            status = "nonpositive_cache_ID_presence_unknown" if value <= 0 else (
                "recorded_ID_map_unavailable" if allowed is None else
                "canonical_ID_known" if value in allowed else "positive_ID_lookup_missing")
            counts[status] += 1
        result[name] = dict(counts)
    counts = Counter("canonical_ID_known" if int(value) in source["champions"] else
                     "nonpositive_cache_ID_presence_unknown" if int(value) <= 0 else
                     "positive_ID_lookup_missing" for value in bans.reshape(-1))
    result["bans"] = dict(counts)
    return result
