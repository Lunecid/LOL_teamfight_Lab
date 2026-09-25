"""gameplay/champion_attributes.py: Data Dragon v2 champion vectors (v4-exact StateV3 player block)."""
import json
import math
import random
from pathlib import Path

import numpy as np
import pytest

from gameplay.champion_attributes import (
    CHAMPION_VECTOR_FIELDS, CLASS_TAGS, ChampionTable, champion_ids_from_meta, champion_matrix,
    champion_vector, load_champion_table_v2, resolve_name, unknown_champion_vector,
)

CACHE = Path("D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13")
PATCH_INDEX = CACHE.parent / (CACHE.name + "_patch_index.json")
N_SAMPLE = 150


@pytest.fixture(scope="module")
def t1514():
    return load_champion_table_v2("15.14")


@pytest.fixture(scope="module")
def sample_1514():
    """Deterministic sample of cached 15.14 matches (only 15.14 ids are ever selected)."""
    if not PATCH_INDEX.exists():
        pytest.skip("match cache not available")
    idx = json.loads(PATCH_INDEX.read_text(encoding="utf-8"))
    ids = sorted(k for k, v in idx.items() if v == "15.14")
    del idx
    return sorted(random.Random(1514).sample(ids, N_SAMPLE))


def test_vector_layout_and_known_champion(t1514):
    v = champion_vector(1, t1514)  # Annie
    assert tuple(v) == CHAMPION_VECTOR_FIELDS
    assert v["champ_unknown"] == 0.0
    assert v["tag_Mage"] == 1.0 and v["tag_Support"] == 1.0 and v["tag_Tank"] == 0.0
    assert v["is_ranged"] == 1.0 and v["attackrange"] == 625.0
    assert v["partype_Mana"] == 1.0 and v["partype_Energy"] + v["partype_none"] + v["partype_other"] == 0.0
    assert v["base_healthMax"] == 560 and v["growth_healthMax"] == 96 and v["growth_attackSpeed_pct"] == 1.36
    assert v["base_movementSpeed"] == 335 and v["info_magic"] == 10
    assert not any(math.isnan(x) for x in v.values())


def test_all_table_champions_well_formed(t1514):
    M = champion_matrix(sorted(t1514.champions), t1514)
    assert M.shape == (len(t1514), len(CHAMPION_VECTOR_FIELDS)) and not np.isnan(M).any()
    col = {f: i for i, f in enumerate(CHAMPION_VECTOR_FIELDS)}
    tags = M[:, [col[f"tag_{t}"] for t in CLASS_TAGS]]
    assert set(np.unique(tags)) <= {0.0, 1.0} and (tags.sum(1) >= 1).all()
    part = M[:, [col[c] for c in ("partype_Mana", "partype_Energy", "partype_none", "partype_other")]]
    assert (part.sum(1) == 1).all()
    assert ((M[:, col["attackrange"]] > 300) == (M[:, col["is_ranged"]] == 1)).all()
    for k, rec in t1514.champions.items():
        assert set(rec["tags"]) <= set(CLASS_TAGS), rec["name"]


def test_partype_and_range_edge_cases(t1514):
    by = {rec["name"]: k for k, rec in t1514.champions.items()}
    assert champion_vector(by["Garen"], t1514)["partype_none"] == 1.0          # 'None'
    assert champion_vector(by["Belveth"], t1514)["partype_none"] == 1.0        # ''
    assert champion_vector(by["Zed"], t1514)["partype_Energy"] == 1.0
    assert champion_vector(by["Tryndamere"], t1514)["partype_other"] == 1.0    # Fury
    assert champion_vector(by["Rakan"], t1514)["is_ranged"] == 0.0             # 300 is not > 300
    assert champion_vector(by["Lillia"], t1514)["is_ranged"] == 1.0            # 325


@pytest.mark.parametrize("cid", [None, 0, -5, 99999, 60001, 1.5, "x", float("nan"), 12345])
def test_unknown_champion_vector(t1514, cid):
    v = champion_vector(cid, t1514)
    assert tuple(v) == CHAMPION_VECTOR_FIELDS and v["champ_unknown"] == 1.0
    assert all(math.isnan(x) for f, x in v.items() if f != "champ_unknown")
    assert v.keys() == unknown_champion_vector().keys()


def test_float_and_string_ids_resolve(t1514):
    assert champion_vector(1.0, t1514) == champion_vector("1", t1514) == champion_vector(np.float32(1), t1514)


def test_plain_mapping_table(t1514):
    assert champion_vector(1, dict(t1514.champions)) == champion_vector(1, t1514)


def test_table_is_read_only(t1514):
    with pytest.raises(TypeError):
        t1514.champions[1] = {}
    with pytest.raises(TypeError):
        t1514.champions[1]["tags"] = ()
    assert load_champion_table_v2("15.14") is t1514


def test_new_champions_in_16_13():
    t = load_champion_table_v2("16.13")
    assert isinstance(t, ChampionTable)
    for name in ("Locke", "Zaahen"):
        k = resolve_name(name, t)
        assert k is not None, name
        v = champion_vector(k, t)
        assert v["champ_unknown"] == 0.0 and sum(v[f"tag_{c}"] for c in CLASS_TAGS) >= 1
    assert resolve_name("Locke", load_champion_table_v2("15.14")) is None
    assert champion_vector(805, load_champion_table_v2("15.14"))["champ_unknown"] == 1.0


def test_16_15_mode_entries_dropped():
    t = load_champion_table_v2("16.15")
    assert all(k < 10000 for k in t.champions) and len(t.dropped_mode_keys) > 0
    assert champion_vector(60001, t)["champ_unknown"] == 1.0
    assert set(load_champion_table_v2("16.13").champions) <= set(t.champions)


def test_missing_patch_raises():
    with pytest.raises(FileNotFoundError):
        load_champion_table_v2("9.99")


def test_ids_from_meta_crosscheck():
    meta = {"static_meta": {"champion_by_pid": {str(p): 100 + p for p in range(1, 10)}}}  # pid 10 missing
    ids = champion_ids_from_meta(meta)
    assert ids[1] == 101 and ids[10] == 0
    nm = np.zeros((3, 10, 4), np.float32)
    nm[:, :9, 2] = np.arange(101, 110)
    assert champion_ids_from_meta(meta, nm, 2) == ids
    nm[0, 3, 2] = 7
    with pytest.raises(ValueError):
        champion_ids_from_meta(meta, nm, 2)


def test_all_1514_cached_champions_resolve(t1514, sample_1514):
    from core.config import NODE_IDX
    col = NODE_IDX["champion_id"]
    seen = set()
    for i, mid in enumerate(sample_1514):
        meta = json.loads((CACHE / f"{mid}.meta.json").read_text(encoding="utf-8"))
        assert meta["patch"] == "15.14"
        nm = np.load(CACHE / f"{mid}.npz")["node_minute"] if i < 30 else None
        ids = champion_ids_from_meta(meta, nm, col if nm is not None else None)
        names = meta["static_meta"]["champion_name_by_pid"]
        for pid, k in ids.items():
            assert champion_vector(k, t1514)["champ_unknown"] == 0.0, (mid, pid, k)
            assert resolve_name(names[str(pid)], t1514) == k, (mid, pid, names[str(pid)])
            seen.add(k)
    assert len(seen) > 100
