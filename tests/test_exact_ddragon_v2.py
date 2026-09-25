"""Data Dragon v2 extracts (scripts/exact_v4/ev4_00_fetch_ddragon.py) cover all v4-exact patches."""
import json
import numbers
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DD2 = REPO / "config/game_rules/datadragon_v2"
PATCHES = ["15.14", "15.15", "15.16", "15.18", "15.19", "15.20", "15.21", "15.22", "16.13", "16.14", "16.15"]
CLASS_TAGS = {"Assassin", "Fighter", "Mage", "Marksman", "Support", "Tank"}


def _load(name):
    return json.loads((DD2 / name).read_text(encoding="utf-8"))


def test_index_has_all_patches():
    idx = _load("index.json")
    assert set(PATCHES) <= set(idx["patches"])
    assert not idx.get("unresolved_patches")
    for p in PATCHES:
        e = idx["patches"][p]
        assert e["version"].startswith(p + "."), (p, e["version"])
        assert e["raw"]["item"]["sha256"] and e["raw"]["championFull"]["sha256"]


@pytest.mark.parametrize("patch", PATCHES)
def test_patch_files_exist_and_match_index(patch):
    idx = _load("index.json")["patches"][patch]
    items = _load(f"items_{patch}.json")
    champs = _load(f"champions_{patch}.json")
    assert items["patch"] == champs["patch"] == patch
    assert items["version"] == champs["version"] == idx["version"]
    assert len(items["items"]) == idx["items"] > 0
    assert len(champs["champions"]) == idx["champions"] > 0


@pytest.mark.parametrize("patch", PATCHES)
def test_every_champion_has_class_tag(patch):
    champs = _load(f"champions_{patch}.json")["champions"]
    for key, ch in champs.items():
        assert int(key) == ch["key"]
        assert len(ch["tags"]) >= 1, ch["name"]
        assert set(ch["tags"]) <= CLASS_TAGS, (ch["name"], ch["tags"])
        assert isinstance(ch["attackrange"], numbers.Real) and ch["attackrange"] > 0
        assert set(ch["base"]) >= {"healthMax", "armor", "magicResist", "attackDamage", "attackSpeed", "movementSpeed"}
        assert set(ch["info"]) == {"attack", "defense", "magic", "difficulty"}


@pytest.mark.parametrize("patch", PATCHES)
def test_item_gold_fields_numeric(patch):
    items = _load(f"items_{patch}.json")["items"]
    for iid, it in items.items():
        for k in ("gold_total", "gold_base", "gold_sell"):
            assert isinstance(it[k], numbers.Real) and not isinstance(it[k], bool), (iid, k)
            assert it[k] >= 0, (iid, k)
        assert all(isinstance(v, numbers.Real) for v in it["stats"].values()), iid
        assert isinstance(it["map_sr"], bool) and isinstance(it["inStore"], bool)
        assert it["depth"] is None or isinstance(it["depth"], int)


def test_old_datadragon_folder_untouched_layout_compatible():
    """v2 base/growth blocks equal the v1 extract for the patches both folders hold."""
    old = REPO / "config/game_rules/datadragon"
    for p in ("15.14", "15.15", "15.16"):
        f = old / f"champions_{p}.json"
        if not f.exists():
            pytest.skip("v1 extract absent")
        v1 = json.loads(f.read_text(encoding="utf-8"))["champions"]
        v2 = _load(f"champions_{p}.json")["champions"]
        assert set(v1) == set(v2)
        for k in v1:
            assert v1[k]["base"] == v2[k]["base"] and v1[k]["growth"] == v2[k]["growth"], (p, k)
