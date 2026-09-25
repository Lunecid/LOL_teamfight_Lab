"""Item state at an exact millisecond from ITEM_* events <= t (v4-exact, stage 1).

The timeline carries only shop events (ITEM_PURCHASED / SOLD / UNDO / DESTROYED with an item id);
item gold, stats and recipes come from the Data Dragon v2 extracts in
config/game_rules/datadragon_v2/.  Nothing after t is read: a replay stops at the first event with
timestamp > t, and frames are never touched.

Two replay modes share one function, `replay_inventory`:

  legacy (exact=False)  the body of EvidenceStateBuilder._replay_inventory, moved here unchanged;
                        the method calls it, so the evidence-state pilot reproduces.
  exact  (exact=True)   the same event algebra plus the rules the timeline needs, each checked on
                        15.14 events (not on the stat validation):
                          0. the destroys of one ms are a multiset, allocated to that ms's purchases through
                             their recipe trees (_allocate_components; 2422 counts as 1001 Boots, 2421 builds
                             3157); a purchase costs gold_total minus the consumed items (purchase_price, =
                             gold_base when exactly the recipe was consumed); only unallocated destroys are
                             'loose' (Viego rule, transforms, consumption);
                          1. an ITEM_UNDO of a combine purchase restores exactly the component copies that
                             purchase consumed (the timeline emits no event for them); the undo of a purchase
                             that cost nothing is logged as beforeId 0 / afterId X / goldGain 0 and undoes it
                             (a trinket swap gives the old trinket back);
                          2. a destroy that is not a same-ms combine and whose item is the
                             `specialRecipe` source of another item transforms it (3003 -> 3040,
                             3004 -> 3042, 3119 -> 3121, 3865 -> 3866 -> 3867, ...) and 2420 -> 2421
                             (Seeker's Armguard after its stasis);
                          3. a destroy of a quest item whose every upgrade has the same stats and is
                             built only from it (3867 Bounty of Worlds) keeps the item as a proxy for
                             the unobserved upgrade; a later purchase of an upgrade replaces it (free), and a
                             sale of an upgrade that is not held sells the proxy;
                          4. removal of an item not held is counted in `diagnostics`, not guessed;
                          5. a consumable destroyed and bought at the same ms (listed destroy-first) is
                             bought, then consumed;
                          6. ITEM_DESTROYED of 4638 Watchful Wardstone outside a combine is not a removal;
                        and three champion/rune/slot rules when the pack meta is given (ItemStateIndex):
                        Viego (234) possession swaps (loose destroys within 30 s after one of his
                        takedowns) are not consumption; rune 8304 grants 2422 at 12:00 minus 45 s per
                        takedown, or at the player's first 2422 event if that is earlier
                        (magical_footwear_grant_ms); the two World Atlas purchases logged with
                        participantId 0 at 0 ms go to slots 5 and 10 (shop_events_by_participant).
                        16.x (patch major >= 16 only, ItemStateIndex with `patch`): role-quest items and the
                        role-bound slot, quest-granted tier-3 boots, the support ward slot, Recall items, the trinket
                        slot / Eye of the Herald, blind undos and two rune grants (_R16Context, rules and evidence in
                        config/game_rules/item_rules_16x.json).  A 15.x replay never enters that code.
                        15.x (patch major == 15 only, ItemStateIndex with `patch`): the Symbiotic Soles upgrade
                        3010 -> 3013, the trinket slot (3340 / Fiddlesticks 3330 at 0 ms, a trinket purchase
                        replaces the held trinket, Eye of the Herald 3513) and the Biscuit Delivery rune grant
                        (_R15Context, rules and evidence in config/game_rules/item_rules_15x.json; the Triple Tonic
                        grant R15-5 was removed by author decision 2026-09-25, see its 'removed_rules').  A
                        replay without a patch, or of another major, never enters that code.

Features (player_item_vector) are the fixed ITEM_VECTOR_NAMES: 13 Data Dragon stat sums (ITEM_STAT_KEYS),
item gold owned, completed item count, boots tier, five effect flags (config/game_rules/item_effect_flags.json:
stasis, revive, cleanse, shield, lifeline) and the count of item ids absent from the patch table (22 fields).
'lifeline' (Sterak's Gage, Immortal Shieldbow, Maw of Malmortius, Seraph's Embrace) is stored as per-patch id sets
('ids_by_patch'), resolved by name in every datadragon_v2 patch; pass `patch` to load_effect_flags / ItemStateIndex.

check_unknown_rate guards an extract: it raises UnknownItemRateError when item ids absent from the patch table exceed
max_rate (0.5 %) of the item observations.
"""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
DD2_DIR = ROOT / "config/game_rules/datadragon_v2"
EFFECT_FLAGS_PATH = ROOT / "config/game_rules/item_effect_flags.json"

# Values of scripts.fetch_datadragon_tables.ITEM_STAT_MAP, in its order (checked against datadragon_v2/index.json).
ITEM_STAT_KEYS: Tuple[str, ...] = (
    "attackDamage", "abilityPower", "armor", "magicResist", "healthMax", "powerMax", "attackSpeed_pct",
    "movementSpeed", "movementSpeed_pct", "critChance", "lifesteal", "healthRegen", "powerRegen",
)
EFFECT_FLAGS: Tuple[str, ...] = ("stasis", "revive", "cleanse", "shield", "lifeline")
MAX_UNKNOWN_ITEM_RATE = 0.005
ITEM_VECTOR_NAMES: Tuple[str, ...] = (
    tuple(f"item_{k}" for k in ITEM_STAT_KEYS)
    + ("item_gold_owned", "completed_item_count", "boots_tier")
    + tuple(f"flag_{f}" for f in EFFECT_FLAGS)
    + ("unknown_item_count",)
)
COMPLETED_MIN_GOLD = 2000
# Transformations Data Dragon does not record as specialRecipe: source -> result.
EXTRA_TRANSFORMS: Dict[int, int] = {2420: 2421}          # Seeker's Armguard -> Shattered Armguard (stasis used)
MODE_VARIANT_MIN_ID = 100_000                            # 22xxxx Arena, 32xxxx, 66xxxx, 773xxx mode copies
# Items whose ITEM_DESTROYED does not remove them: 4638 Watchful Wardstone is destroyed repeatedly while held
# (15.14: destroy events after its purchase, and again after an earlier destroy), i.e. stored-ward bookkeeping.
DESTROY_IS_NOT_REMOVAL = frozenset({4638})
VIEGO_ID = 234                                           # possession swaps inventories through ITEM_DESTROYED
# A possession needs a takedown: with takedown times given, a Viego destroy is a possession swap only within this
# window after one of his takedowns (15.14, 600 matches: 1,813/1,814 multi-item destroy groups follow a takedown
# within 30 s; 74/136 lone jungle-pet destroys do not, and those match the pet consumption seen for other junglers).
VIEGO_POSSESSION_WINDOW_MS = 30_000
MAGICAL_FOOTWEAR_RUNE, MAGICAL_FOOTWEAR_ITEM = 8304, 2422
MAGICAL_FOOTWEAR_MS, MAGICAL_FOOTWEAR_STEP_MS = 720_000, 45_000
WORLD_ATLAS_ID, SUPPORT_ATLAS_SLOTS = 3865, (5, 10)     # participantId-0 purchases at 0 ms -> utility slots
# A held item that the shop accepts in place of a recipe component.  2422 Slightly Magical Footwear counts as 1001
# Boots in every boots recipe; Data Dragon's 2422 'into' omits 3010 / 3005 (15.14: 4 held-out upgrades 2422 -> 3010
# were charged 300 less than gold_total, i.e. as from Boots).
COMPONENT_SUBSTITUTES: Dict[int, Tuple[int, ...]] = {1001: (2422,)}
# 16.x rules (patch major >= 16 only; 15.x replays never read them): config/game_rules/item_rules_16x.json.
RULES_16X_PATH = ROOT / "config/game_rules/item_rules_16x.json"
RULES_16X_MIN_MAJOR = 16
# 15.x rules (patch major == 15 only; no patch / other majors never read them): config/game_rules/item_rules_15x.json.
RULES_15X_PATH = ROOT / "config/game_rules/item_rules_15x.json"
RULES_15X_MAJOR = 15

ShopEvent = Tuple[int, str, int, float, int, int]        # ts, kind, item, gold, before, after
SHOP_TYPES = {"ITEM_PURCHASED": "buy", "ITEM_SOLD": "sell", "ITEM_UNDO": "undo", "ITEM_DESTROYED": "destroy"}

_TABLE_CACHE: Dict[Tuple[str, bool], Dict[int, dict]] = {}


# ---------------------------------------------------------------------------------------------- tables
def load_item_table_v2(patch: str, sr_only: bool = True, dd_dir: Optional[Path] = None) -> Dict[int, dict]:
    """Data Dragon v2 item table for `patch` as {item_id: record}.

    sr_only keeps every Summoner's Rift entry (map_sr) and every 4-digit base id whatever its map flags
    (3513 Eye of the Herald and, in 16.15, 4643 are SR events but not flagged map_sr); it drops the
    6-digit mode copies that are not on SR (22xxxx Arena, 16.15 773xxx).  inStore is NOT required:
    transformed items (3040, 3866, 3867, 2421 ...) are held without being sold.
    """
    key = (str(patch), bool(sr_only))
    if dd_dir is None and key in _TABLE_CACHE:
        return _TABLE_CACHE[key]
    path = Path(dd_dir or DD2_DIR) / f"items_{patch}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))["items"]
    table = {int(k): v for k, v in raw.items()}
    if sr_only:
        table = {k: v for k, v in table.items() if v.get("map_sr") or k < 10_000}
    if dd_dir is None:
        _TABLE_CACHE[key] = table
    return table


def _flag_ids(spec_flag: Mapping, patch: Optional[str], items_table: Optional[Mapping[int, dict]]) -> frozenset:
    """Id set of one flag: 'ids' (every patch) or 'ids_by_patch' (that patch; without a patch, the union over
    patches, restricted to the table's ids when a table is given)."""
    if "ids" in spec_flag:
        return frozenset(int(i) for i in spec_flag["ids"])
    by_patch = spec_flag["ids_by_patch"]
    if patch is not None:
        if str(patch) not in by_patch:
            raise KeyError(f"effect flag has no id set for patch {patch}")
        return frozenset(int(i) for i in by_patch[str(patch)])
    union = {int(i) for ids in by_patch.values() for i in ids}
    if items_table is not None:
        union = {i for i in union if i in items_table}
    return frozenset(union)


def load_effect_flags(items_table: Optional[Mapping[int, dict]] = None,
                      path: Optional[Path] = None, patch: Optional[str] = None) -> Dict[str, frozenset]:
    """{flag: frozenset(ids)} from item_effect_flags.json; with a table, every id must be in it.

    Per-patch flags ('lifeline') use `patch`'s set; without `patch` they use the union of all patches' sets
    (restricted to the table when one is given; the sets are identical in the 11 patches).
    """
    spec = json.loads(Path(path or EFFECT_FLAGS_PATH).read_text(encoding="utf-8"))
    flags = {f: _flag_ids(spec["flags"][f], patch, items_table) for f in EFFECT_FLAGS}
    if items_table is not None:
        missing = {f: sorted(i for i in ids if i not in items_table) for f, ids in flags.items()}
        missing = {f: v for f, v in missing.items() if v}
        if missing:
            raise KeyError(f"effect flag ids absent from the item table: {missing}")
    return flags


def resolve_ids_by_name(items_table: Mapping[int, dict], names: Iterable[str]) -> Dict[str, List[int]]:
    """{name: sorted ids} of Summoner's Rift entries (map_sr) carrying each name; Arena / mode copies
    (map_sr false) are excluded.  An empty list means the name is missing (or renamed) in the table."""
    out: Dict[str, List[int]] = {n: [] for n in names}
    for iid, it in items_table.items():
        if it.get("name") in out and it.get("map_sr"):
            out[it["name"]].append(int(iid))
    return {n: sorted(v) for n, v in out.items()}


def verify_effect_flags(patches: Optional[Sequence[str]] = None, path: Optional[Path] = None) -> Dict[str, dict]:
    """{patch: {flag: [problems]}} for the patches (default: the JSON's list); empty dicts = all resolve.

    A problem is an id that is absent, not map_sr or carries another name than recorded, or (for flags with
    'resolve_names') 'name missing: <name>' / 'unlisted id <id>' when name resolution and the stored set differ.
    """
    spec = json.loads(Path(path or EFFECT_FLAGS_PATH).read_text(encoding="utf-8"))
    out: Dict[str, dict] = {}
    for patch in (patches or spec["patches"]):
        table = load_item_table_v2(patch)
        bad = {}
        for f in EFFECT_FLAGS:
            sf = spec["flags"][f]
            names = sf["names"]
            ids = sorted(_flag_ids(sf, str(patch), None))
            wrong: list = [i for i in ids
                           if i not in table or not table[i].get("map_sr") or table[i].get("name") != names.get(str(i))]
            if "resolve_names" in sf:
                res = resolve_ids_by_name(table, sf["resolve_names"])
                wrong += [f"name missing: {n}" for n, v in res.items() if not v]
                wrong += [f"unlisted id {i}" for v in res.values() for i in v if i not in ids]
            if wrong:
                bad[f] = wrong
        out[str(patch)] = bad
    return out


class UnknownItemRateError(ValueError):
    """Item ids absent from the patch table exceed the allowed share of item observations."""


def check_unknown_rate(counts, total: float, max_rate: float = MAX_UNKNOWN_ITEM_RATE) -> float:
    """Rate of unknown item ids = sum(counts) / total; raises UnknownItemRateError when it exceeds max_rate.

    counts: {item_id: n} of ids absent from the patch table (or the plain number of such observations);
    total:  number of item observations the counts come from (e.g. item ids in ITEM_* events or inventory
            slots).  For the Stage 2 extract step: stop when unknown ids exceed 0.5 %; a rate equal to max_rate passes.
    """
    n_unknown = float(sum(counts.values()) if isinstance(counts, Mapping) else counts)
    total = float(total)
    if n_unknown < 0 or total < 0 or not (0.0 <= max_rate <= 1.0):
        raise ValueError("counts and total must be non-negative and max_rate in [0, 1]")
    if total == 0:
        if n_unknown > 0:
            raise UnknownItemRateError(f"{n_unknown:.0f} unknown item ids but no item observations")
        return 0.0
    if n_unknown > total:
        raise ValueError(f"unknown count {n_unknown:.0f} exceeds total {total:.0f}")
    rate = n_unknown / total
    if rate > max_rate:
        top = ""
        if isinstance(counts, Mapping):
            top = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])[:10])
        raise UnknownItemRateError(f"unknown item id rate {rate:.4%} > {max_rate:.2%} "
                                   f"({n_unknown:.0f} of {total:.0f})" + (f"; top ids {top}" if top else ""))
    return rate


def transform_map(items_table: Mapping[int, dict]) -> Dict[int, int]:
    """source -> result for items that turn into another without a purchase (specialRecipe + EXTRA_TRANSFORMS)."""
    out: Dict[int, int] = {}
    for iid, it in sorted(items_table.items()):
        src = it.get("specialRecipe")
        if src and iid < MODE_VARIANT_MIN_ID and int(src) < MODE_VARIANT_MIN_ID and int(src) not in out:
            out[int(src)] = int(iid)
    for s, t in EXTRA_TRANSFORMS.items():
        if s in items_table and t in items_table:
            out[s] = t
    return out


def proxy_sources(items_table: Mapping[int, dict]) -> frozenset:
    """Quest items whose upgrades are built only from them and carry identical stats (3867 Bounty of Worlds)."""
    out = set()
    for iid, it in items_table.items():
        into = [t for t in it.get("into", []) if t in items_table]
        if into and all(items_table[t].get("from") == [iid] and items_table[t].get("stats", {}) == it.get("stats", {})
                        for t in into):
            out.add(int(iid))
    return frozenset(out)


_RULE_CACHE: Dict[int, tuple] = {}
_RULES16_CACHE: Dict[str, dict] = {}


def patch_major(patch) -> Optional[int]:
    """16 for '16.15', 15 for '15.14'; None when the patch is missing or not 'major.minor...'."""
    try:
        return int(str(patch).split(".")[0])
    except (TypeError, ValueError):
        return None


def uses_rules_16x(patch) -> bool:
    """True iff the 16.x item rules apply: patch major >= 16 (a missing patch never switches them on)."""
    m = patch_major(patch) if patch is not None else None
    return m is not None and m >= RULES_16X_MIN_MAJOR


def load_rules_16x(path: Optional[Path] = None) -> dict:
    """The 16.x rule file as a replay-ready dict (ids as ints); the raw 'rules' list keeps its evidence."""
    key = str(path or RULES_16X_PATH)
    hit = _RULES16_CACHE.get(key)
    if hit is not None:
        return hit
    spec = json.loads(Path(key).read_text(encoding="utf-8"))
    q = spec["role_quests"]
    sup = spec["support_ward_slot"]
    r = {
        "role_by_slot": tuple(spec["role_by_slot"]),
        "start": {role: v for role, v in q["start"].items()},
        "top_teleport_spell": int(q["top_teleport_spell"]),
        "transforms": {int(k): int(v) for k, v in q["transforms"].items()},
        "chains": {role: frozenset(int(i) for i in ids) for role, ids in q["chains"].items()},
        "support_quest": int(sup["quest"]), "support_reward": int(sup["reward"]),
        "support_completion_destroy": int(sup["completion_destroy"]), "ward": int(sup["ward"]),
        "mid_quest": int(spec["mid_tier3_boots"]["quest"]),
        "bot_boots_reward": int(spec["bot_boots_slot"]["reward"]),
        "transient_ids": frozenset(int(i) for i in spec["transient_ids"]["ids"]),
        "start_trinket": int(spec["trinket_slot"]["start_trinket"]),
        "herald_eye": int(spec["trinket_slot"]["herald_eye"]),
        "start_trinket_by_champion": {int(k): int(v) for k, v in
                                      spec["trinket_slot"].get("start_trinket_by_champion_id", {}).items()},
        "rune_grants_at_ms": {int(k): (int(v["item"]), tuple(int(t) for t in v["at_ms"]))
                              for k, v in spec.get("rune_grants", {}).items() if "at_ms" in v},
        "rune_grants_at_level": {int(k): {int(lv): int(i) for lv, i in v["at_level"].items()}
                                 for k, v in spec.get("rune_grants", {}).items() if "at_level" in v},
        "rules": spec.get("rules", []),
    }
    _RULES16_CACHE[key] = r
    return r


def uses_rules_15x(patch) -> bool:
    """True iff the 15.x item rules apply: patch major == 15 (a missing patch never switches them on)."""
    m = patch_major(patch) if patch is not None else None
    return m == RULES_15X_MAJOR


_RULES15_CACHE: Dict[str, dict] = {}


def load_rules_15x(path: Optional[Path] = None) -> dict:
    """The 15.x rule file as a replay-ready dict (ids as ints); the raw 'rules' list keeps its evidence.
    The rune-grant keys have the layout of load_rules_16x, so rune_grants_16x serves both."""
    key = str(path or RULES_15X_PATH)
    hit = _RULES15_CACHE.get(key)
    if hit is not None:
        return hit
    spec = json.loads(Path(key).read_text(encoding="utf-8"))
    ts = spec["trinket_slot"]
    r = {
        "loose_transforms": {int(k): int(v) for k, v in spec["loose_destroy_transforms"]["map"].items()},
        "start_trinket": int(ts["start_trinket"]),
        "start_trinket_by_champion": {int(k): int(v) for k, v in ts.get("start_trinket_by_champion_id", {}).items()},
        "herald_eye": int(ts["herald_eye"]),
        "rune_grants_at_ms": {int(k): (int(v["item"]), tuple(int(t) for t in v["at_ms"]))
                              for k, v in spec.get("rune_grants", {}).items() if "at_ms" in v},
        "rune_grants_at_level": {int(k): {int(lv): int(i) for lv, i in v["at_level"].items()}
                                 for k, v in spec.get("rune_grants", {}).items() if "at_level" in v},
        "rules": spec.get("rules", []),
    }
    _RULES15_CACHE[key] = r
    return r


def rune_grants_16x(r: Mapping, events: Sequence[dict], pid: int, runes: Optional[Mapping]) -> List[Tuple[int, int]]:
    """(ts, item) rune grants without a timeline event (16.x R16-11 / R16-12, 15.x R15-4; `r` from load_rules_16x or
    load_rules_15x): Biscuit Delivery at fixed times, Triple Tonic (16.x only; 15.x R15-5 was removed 2026-09-25) at
    the player's LEVEL_UP events (first event reaching the level).  Causal: a grant at ts needs only events <= ts."""
    if not runes:
        return []
    ids = {int(v) for k, v in runes.items() if "rune" in str(k) and v is not None}
    out: List[Tuple[int, int]] = []
    for rune, (item, times) in r["rune_grants_at_ms"].items():
        if rune in ids:
            out += [(int(t), item) for t in times]
    for rune, by_level in r["rune_grants_at_level"].items():
        if rune in ids:
            first: Dict[int, int] = {}
            for e in events:
                if e.get("type") == "LEVEL_UP" and int(e.get("participantId", 0) or 0) == pid:
                    lv = int(e.get("level", 0) or 0)
                    first[lv] = min(first.get(lv, int(e.get("timestamp", 0))), int(e.get("timestamp", 0)))
            out += [(first[lv], item) for lv, item in sorted(by_level.items()) if lv in first]
    return sorted(out)


def role_quest_start(r: Mapping, role: str, summoner_spells=None) -> int:
    """The role-quest item a player of `role` holds from 0 ms (16.x, R16-1).  TOP depends on the pre-game summoner
    spells ({'summoner_spell_1_id': ..} or a list): Teleport (12) -> 1222, else 1200; unknown spells -> 1200."""
    start = r["start"][role]
    if isinstance(start, Mapping):
        vals = list(summoner_spells.values()) if isinstance(summoner_spells, Mapping) else list(summoner_spells or [])
        tp = r["top_teleport_spell"] in {int(v) for v in vals if v is not None}
        start = start["with_teleport"] if tp else start["without_teleport"]
    return int(start)


def tier3_boots_map(items_table: Mapping[int, dict]) -> Dict[int, int]:
    """tier-2 boots -> its free tier-3 upgrade: the 'into' item built only from it with gold_base 0 (16.15 Data
    Dragon: 3006 -> 3172, 3009 -> 3170, 3020 -> 3175, 3010 -> 3013, ...; 3172 Gunmetal Greaves carries no 'Boots'
    tag, so the upgrade's tags are not required)."""
    out: Dict[int, int] = {}
    for iid, it in sorted(items_table.items()):
        if not is_boots(it) or iid >= MODE_VARIANT_MIN_ID:
            continue
        for t in it.get("into") or []:
            u = items_table.get(int(t)) or {}
            if list(u.get("from") or []) == [iid] and float(u.get("gold_base", 0)) == 0.0:
                out[int(iid)] = int(t)
                break
    return out


def _recipe_rules(items_table: Mapping[int, dict]) -> Tuple[Dict[int, int], frozenset]:
    hit = _RULE_CACHE.get(id(items_table))
    if hit is None or hit[0] is not items_table:
        hit = (items_table, transform_map(items_table), proxy_sources(items_table))
        _RULE_CACHE[id(items_table)] = hit
    return hit[1], hit[2]


# ---------------------------------------------------------------------------------------------- events
def shop_events_by_participant(events: Iterable[dict], items_table: Mapping[int, dict],
                               participants: Iterable[int] = range(1, 11), *,
                               assign_support_atlas: bool = False) -> Dict[int, List[ShopEvent]]:
    """ITEM_* timeline events as time-sorted ShopEvent lists per participant (EvidenceStateBuilder encoding).

    Events with participantId 0 belong to no player and are dropped, except (assign_support_atlas=True, the exact
    mode) the World Atlas purchases the timeline logs with participantId 0 at 0 ms: the first goes to slot 5 and
    the second to slot 10 (participant order = assigned role; SUPPORT_ATLAS_SLOTS).  Evidence (15.14, the 300
    rule-check matches, seed 20260925): every match has exactly two such events, all at 0 ms, and in 296/300 the
    later World Atlas family events (3865-3877) come from slots 5 and 10 (the other 4 have none); the frame at 60 s
    shows the 400 gold spent by slots 5 and 10.  The sort is stable, so same-ms events keep their timeline order.
    """
    out: Dict[int, List[ShopEvent]] = {int(p): [] for p in participants}
    atlas_slots = list(SUPPORT_ATLAS_SLOTS) if assign_support_atlas else []
    for e in sorted((e for e in events if e.get("type") in SHOP_TYPES), key=lambda e: int(e.get("timestamp", 0))):
        pid = int(e.get("participantId", 0) or 0)
        if (pid == 0 and atlas_slots and e.get("type") == "ITEM_PURCHASED"
                and int(e.get("itemId", 0) or 0) == WORLD_ATLAS_ID and int(e.get("timestamp", 0)) == 0):
            pid = atlas_slots.pop(0)
        if pid not in out:
            continue
        ts, kind = int(e.get("timestamp", 0)), SHOP_TYPES[e["type"]]
        iid = int(e.get("itemId", 0) or 0)
        if kind == "buy":
            out[pid].append((ts, "buy", iid, 0.0, 0, 0))
        elif kind == "sell":
            out[pid].append((ts, "sell", iid, float(items_table.get(iid, {}).get("gold_sell", 0)), 0, 0))
        elif kind == "undo":
            out[pid].append((ts, "undo", 0, float(e.get("goldGain", 0) or 0),
                             int(e.get("beforeId", 0) or 0), int(e.get("afterId", 0) or 0)))
        else:
            out[pid].append((ts, "destroy", iid, 0.0, 0, 0))
    return out


# ---------------------------------------------------------------------------------------------- replay
def replay_inventory(shop_events: Sequence[ShopEvent], items_table: Mapping[int, dict], upto_ms: int,
                     since_ms: Optional[int] = None, *, exact: bool = False,
                     grants: Sequence[Tuple[int, int]] = (), champion_id: Optional[int] = None,
                     diagnostics: Optional[dict] = None, takedowns_ms: Optional[Sequence[int]] = None,
                     rules16: Optional[Mapping] = None, role: Optional[str] = None,
                     rules15: Optional[Mapping] = None):
    """Inventory at `upto_ms`; also the gold delta and the stat delta of transactions in (since_ms, upto_ms].

    shop_events: time-sorted ShopEvent tuples of ONE participant (shop_events_by_participant).  Only events with
    ts <= upto_ms are read.
    exact=False: the legacy EvidenceStateBuilder._replay_inventory body, unchanged.
    exact=True: the rules in the module docstring, plus
        grants       (ts, item) items that arrive without an event (magical_footwear_grant_ms); ts <= upto_ms only;
        champion_id  234 (Viego) ignores destroys that are possession swaps, not consumption;
        takedowns_ms Viego's takedown times; when given, a destroy counts as a possession swap only within
                     VIEGO_POSSESSION_WINDOW_MS after a takedown <= its timestamp (None: every candidate is a swap);
        diagnostics  dict filled with counters: unowned_removal, transforms, proxies, undo_restored, granted,
                     viego_ignored, and unowned_ids {item: count};
        rules16      load_rules_16x() for a patch with major >= 16 (ItemStateIndex passes it; None = 15.x replay,
                     whose code path is unchanged), with role = the player's slot role ('TOP' ... 'UTILITY').
        rules15      load_rules_15x() for a patch with major 15 (ItemStateIndex passes it; None = no 15.x rule);
                     ignored when rules16 is given.
    """
    if not exact:
        return _replay_legacy(shop_events, items_table, upto_ms, since_ms)
    return _replay_exact(shop_events, items_table, upto_ms, since_ms, grants, champion_id, diagnostics,
                         takedowns_ms, rules16, role, rules15)


def _replay_legacy(shop_events, items_table, upto, since):
    # Moved verbatim from gameplay/evidence_state.py EvidenceStateBuilder._replay_inventory (self.ev[pid].shop ->
    # shop_events, self.items -> items_table).  Keep unchanged: the evidence-state pilot reproduces through it.
    inv: List[int] = []
    gold_delta, stat_delta = 0.0, defaultdict(float)
    destroyed_at: Dict[int, set] = defaultdict(set)
    for ts, kind, iid, gold, before, after in shop_events:
        if ts > upto:
            break
        if kind == "destroy":
            destroyed_at[ts].add(iid)
    for ts, kind, iid, gold, before, after in shop_events:
        if ts > upto:
            break
        added, removed, delta_gold = [], [], 0.0
        if kind == "buy":
            it = items_table.get(iid, {})
            components = set(it.get("from", []))
            paid = it.get("gold_base", 0) if (components & destroyed_at.get(ts, set())) else it.get("gold_total", 0)
            added, delta_gold = [iid], -float(paid)
        elif kind == "sell":
            removed, delta_gold = [iid], gold
        elif kind == "undo":
            removed, added, delta_gold = ([before] if before else []), ([after] if after else []), gold
        else:                                                       # destroy: consumed / combined
            removed = [iid]
        for x in added:
            inv.append(x)
        for x in removed:
            if x in inv:
                inv.remove(x)
        if since is not None and ts > since:
            gold_delta += delta_gold
            for x in added:
                for k, v in items_table.get(x, {}).get("stats", {}).items():
                    stat_delta[k] += v
            for x in removed:
                for k, v in items_table.get(x, {}).get("stats", {}).items():
                    stat_delta[k] -= v
    return inv, gold_delta, stat_delta


def _consume_after_purchase(stream, items_table, diag, also=None):
    """Rule 5: at one ms the timeline lists 'destroy X' before 'buy X' when a consumable is bought and used at
    once (elixirs); move such destroys after the purchase of X in that ms.  Stable otherwise.
    also(ts, X) -> bool (16.x only): further ids to move the same way (tier-2 boots bought after the mid quest,
    which the game upgrades at once, logging a destroy of X before the purchase of X)."""
    out, i = [], 0
    while i < len(stream):
        j = i
        while j < len(stream) and stream[j][0] == stream[i][0]:
            j += 1
        grp = stream[i:j]
        bought = {e[2] for e in grp if e[1] == "buy"}
        late = [k for k, e in enumerate(grp) if e[1] == "destroy" and e[2] in bought
                and (is_consumable(items_table.get(e[2], {})) or (also is not None and also(e[0], e[2])))
                and any(g[1] == "buy" and g[2] == e[2] for g in grp[k + 1:])]
        if late:
            diag["reordered"] += len(late)
            grp = [e for k, e in enumerate(grp) if k not in late] + [grp[k] for k in late]
        out.extend(grp)
        i = j
    return out


def _allocate_components(bought: int, avail: Counter, items_table: Mapping[int, dict]) -> Tuple[List[int], int]:
    """Components a purchase of `bought` consumes out of `avail` (same-ms destroys not yet used; mutated).

    Walks the recipe tree: a direct component that was destroyed at this ms is consumed (each destroyed copy at
    most once, so recipe 3133 = [1036, 2022, 1036] with one 1036 destroyed consumes one 1036); a component that
    was not destroyed is looked up through its own components (a Long Sword inside the Serrated Dirk of a
    Youmuu's); a missing component with a destroyed COMPONENT_SUBSTITUTES item takes that item instead (B2: 2422
    Slightly Magical Footwear in place of 1001 Boots, in tier-2 and, through them, tier-3 boots).  Then any other
    destroyed item d that is not in the recipe tree but whose 'into' lists `bought` or an item of its recipe tree
    (2421 Shattered Armguard -> 3157 Zhonya's) is consumed once.
    Returns (consumed ids, number of substitutes among them)."""
    consumed: List[int] = []
    tree = {int(bought)}
    n_sub = Counter()

    def walk(node: int, depth: int) -> None:
        for c in items_table.get(node, {}).get("from", []) or []:
            c = int(c)
            tree.add(c)
            if avail.get(c, 0) > 0:
                avail[c] -= 1
                consumed.append(c)
                continue
            sub = next((s for s in COMPONENT_SUBSTITUTES.get(c, ()) if avail.get(s, 0) > 0), None)
            if sub is not None:
                avail[sub] -= 1
                consumed.append(sub)
                n_sub["sub"] += 1
            elif depth < 4:
                walk(c, depth + 1)

    walk(int(bought), 0)
    for d in sorted(avail):
        if avail[d] > 0 and d not in tree and tree & set(items_table.get(d, {}).get("into") or []):
            avail[d] -= 1
            consumed.append(int(d))
            n_sub["sub"] += 1
    return consumed, n_sub["sub"]


def purchase_price(bought: int, consumed: Sequence[int], items_table: Mapping[int, dict]) -> float:
    """Gold a purchase costs: gold_base when exactly the direct recipe was consumed, otherwise gold_total minus the
    gold_total of the consumed items (the shop charges the missing components; nothing consumed = gold_total)."""
    it = items_table.get(bought, {})
    if consumed and Counter(consumed) == Counter(int(c) for c in it.get("from", []) or []):
        return float(it.get("gold_base", 0))
    val = float(it.get("gold_total", 0)) - sum(float(items_table.get(c, {}).get("gold_total", 0)) for c in consumed)
    return max(0.0, val)


_T3_CACHE: Dict[int, tuple] = {}


def _tier3_cached(items_table: Mapping[int, dict]) -> Dict[int, int]:
    hit = _T3_CACHE.get(id(items_table))
    if hit is None or hit[0] is not items_table:
        hit = (items_table, tier3_boots_map(items_table))
        _T3_CACHE[id(items_table)] = hit
    return hit[1]


class _R16Context:
    """The 16.x rules (config/game_rules/item_rules_16x.json) for ONE player's exact replay; built only when the
    patch major is >= 16, so a 15.x replay never enters this code.  All look-ups use the replayed stream, i.e.
    events <= t only.  Rule ids (R16-*) refer to the rule file, which lists the evidence of each:

      R16-1  role-quest item at 0 ms (ItemStateIndex grant, slot role) - handled by the caller's grants;
      R16-2  a destroy of a held quest token turns it into its reward (1200 -> 1220, 1222 -> 1221, 1201 -> 1206,
             1202 -> 1207, 1204 -> 1209); a destroy of a held reward placeholder (1207, 1208) or of the support
             token 1203 outside its completion ms removes it;
      R16-3  support ward slot: 1203 -> 1208 when 3866 is destroyed at that ms (quest completion; a 3866 destroy
             while the slot holds a ward switches the placeholder silently); a 2055 destroy at the ms of a
             1203 / 1208 destroy is the ward moving into the slot (no removal); when the player's last 2055
             leaves (placed, sold, undone) the placeholder of the current quest stage comes back;
      R16-4  mid quest: from the ms of the 1201 destroy on, a loose destroy of a held tier-2 boots turns it into its
             free tier-3 upgrade (tier3_boots_map), and a 'destroy X' listed before 'buy X' at one ms is moved
             after the purchase (the game upgrades the new boots at once);
      R16-5  bot quest: a loose boots destroy at the ms of a 1207 destroy is the boots moving into the role slot;
      R16-6  2001 Recall / 2002 Enhanced Recall events are dropped (created without events at Baron / Herald /
             Dragon kills, never held at game end, no stats);
      R16-7  trinket slot: 3340 at 0 ms (caller's grant); a trinket purchase without a same-ms trinket destroy
             replaces the held trinket (the undo gives it back);
      R16-8  Eye of the Herald: a loose trinket destroy (no trinket bought at that ms) at the ms of a 2001 / 2002
             destroy puts 3513 into the trinket slot; the 3513 destroy (used) gives the stored trinket back;
      R16-9  an ITEM_UNDO with beforeId 0 and afterId 0 undoes the player's last purchase (its item may have been
             upgraded at once, R16-4, or moved into the ward slot, R16-3) and restores what that purchase consumed;
      R16-10 Viego: a destroy group that would be a possession swap is his own quest completing when it holds a
             destroy of his own quest-chain item and at most one other loose destroy, and the R16-8 herald
             exchange is not a swap either (viego_exempt);
      R16-11 / R16-12 rune grants (ItemStateIndex, rune_grants_16x): Biscuit Delivery (8345) 2010 at 2:00, 4:00,
             6:00; Triple Tonic (8313) 2151 / 2152 / 2150 at the LEVEL_UP to 3 / 6 / 9;
      R16-13 Fiddlesticks (9) starts with 3330 Scarecrow Effigy instead of 3340 (caller's grant).
    """

    def __init__(self, stream, items_table, r, role, diag):
        self.items, self.r, self.role, self.diag = items_table, r, role, diag
        for k in ("r16_transient_dropped", "r16_quest_transform", "r16_placeholder_removed", "r16_ward_moved",
                  "r16_ward_slot_refill", "r16_boots_moved", "r16_tier3_boots", "r16_trinket_replaced",
                  "r16_herald_swap", "r16_herald_used", "r16_blind_undo"):
            diag.setdefault(k, 0)
        n0 = len(stream)
        self.transient_ms = frozenset(e[0] for e in stream if e[1] == "destroy" and e[2] in r["transient_ids"])
        self.stream = [e for e in stream if not (e[1] != "grant" and e[2] in r["transient_ids"])]
        diag["r16_transient_dropped"] += n0 - len(self.stream)
        self.chain = r["chains"].get(role, frozenset())
        self.t3 = _tier3_cached(items_table)
        self.d_ms: Dict[int, Counter] = defaultdict(Counter)
        self.b_ms: Dict[int, Counter] = defaultdict(Counter)
        for e in self.stream:
            if e[1] == "destroy":
                self.d_ms[e[0]][e[2]] += 1
            elif e[1] == "buy":
                self.b_ms[e[0]][e[2]] += 1
        self.own_quest_n = {ts: sum(n for i, n in c.items() if i in self.chain) for ts, c in self.d_ms.items()}
        mq = r["mid_quest"]
        self.mid_done_ms = (min((ts for ts, c in self.d_ms.items() if c.get(mq)), default=None)
                            if mq in self.chain else None)
        self.is_support = r["support_quest"] in self.chain
        self.support_stage = r["support_quest"]
        self.stored_trinket: Optional[int] = None
        self.buys: List[int] = []

    def _trinket(self, i) -> bool:
        return is_trinket(self.items.get(int(i), {}))

    def viego_exempt(self, ts: int, iid: int, inv, n_loose: int) -> bool:
        """Viego (R16-10): not a possession swap when the ms holds a destroy of his own quest-chain item and at most
        one other loose destroy (his quest completing: the jungle pet, the boots), or when the destroy is the
        Eye-of-the-Herald exchange of R16-8 (a lone trinket destroy at a Recall-item ms, or the held 3513 used)."""
        n = self.own_quest_n.get(ts, 0)
        if n > 0 and n_loose - n <= 1:
            return True
        herald = self.r["herald_eye"]
        if iid == herald and herald in inv:
            return True
        return n_loose == 1 and ts in self.transient_ms and iid != herald and self._trinket(iid)

    def reorder_also(self, ts: int, x: int) -> bool:                       # R16-4
        return self.mid_done_ms is not None and ts >= self.mid_done_ms and x in self.t3

    def on_buy(self, ts, iid, inv, removed, restore):
        self.buys.append(iid)
        herald = self.r["herald_eye"]
        if self._trinket(iid) and iid != herald and not any(self._trinket(d) for d in self.d_ms.get(ts, ())):
            held = [x for x in inv if self._trinket(x) and x != herald and x not in removed]
            if held:                                                        # R16-7
                removed, restore = list(removed) + held, list(restore) + held
                self.diag["r16_trinket_replaced"] += 1
        return removed, restore

    def on_blind_undo(self, inv, purchases):                                # R16-9
        for k in range(len(self.buys) - 1, -1, -1):
            b = self.buys[k]
            if purchases.get(b):
                del self.buys[k]
                restored = purchases[b].pop()
                cur = b if b in inv else (self.t3.get(b) if self.t3.get(b) in inv else None)
                self.diag["r16_blind_undo"] += 1
                self.diag["undo_restored"] += len(restored)
                return ([cur] if cur is not None else []), list(restored)
        return [], []

    def on_loose_destroy(self, ts, iid, inv):
        r, d = self.r, self.d_ms.get(ts, Counter())
        if iid in self.chain and iid in inv:                                # R16-2 / R16-3
            if iid == r["support_quest"]:
                if d.get(r["support_completion_destroy"]):
                    self.support_stage = r["support_reward"]
                    self.diag["r16_quest_transform"] += 1
                    return True, [iid], [r["support_reward"]]
                self.diag["r16_placeholder_removed"] += 1
                return True, [iid], []
            if iid in r["transforms"]:
                self.diag["r16_quest_transform"] += 1
                return True, [iid], [r["transforms"][iid]]
            self.diag["r16_placeholder_removed"] += 1
            return True, [iid], []
        if iid == r["ward"] and self.is_support and (d.get(r["support_quest"]) or d.get(r["support_reward"])):
            self.diag["r16_ward_moved"] += 1                                # R16-3
            return True, [], []
        if (is_boots(self.items.get(iid, {})) and r["bot_boots_reward"] in self.chain
                and d.get(r["bot_boots_reward"])):
            self.diag["r16_boots_moved"] += 1                               # R16-5
            return True, [], []
        if self.mid_done_ms is not None and ts >= self.mid_done_ms and iid in self.t3 and iid in inv:
            self.diag["r16_tier3_boots"] += 1                               # R16-4
            return True, [iid], [self.t3[iid]]
        herald = r["herald_eye"]
        if (iid != herald and self._trinket(iid) and iid in inv and ts in self.transient_ms
                and not any(self._trinket(b) for b in self.b_ms.get(ts, ()))):
            self.stored_trinket = iid                                       # R16-8
            self.diag["r16_herald_swap"] += 1
            return True, [iid], [herald]
        if iid == herald and herald in inv:
            back, self.stored_trinket = self.stored_trinket, None
            self.diag["r16_herald_used"] += 1
            return True, [herald], ([back] if back is not None else [])
        return False, [iid], []

    def after_event(self, ts, kind, iid, before, inv, removed):
        r = self.r
        if not self.is_support:
            return []
        if kind == "destroy" and iid == r["support_completion_destroy"]:
            self.support_stage = r["support_reward"]                        # R16-3 (slot held a ward)
        w = r["ward"]
        if w in removed and w not in inv and r["support_quest"] not in inv and r["support_reward"] not in inv:
            self.diag["r16_ward_slot_refill"] += 1
            return [self.support_stage]
        return []


class _R15Context:
    """The 15.x rules (config/game_rules/item_rules_15x.json) for ONE player's exact replay; built only when the
    patch major is 15 (and no 16.x context exists), so a replay without a patch or of another major never enters
    this code.  Look-ups use the replayed stream (events <= t only).  Rule ids (R15-*) refer to the rule file,
    which lists the evidence of each:

      R15-1  a loose destroy (not consumed by a purchase at that ms) of a held 3010 Symbiotic Soles turns it into
             3013 Synchronized Souls (loose_destroy_transforms);
      R15-2  trinket slot at 0 ms: 3340 Stealth Ward, 3330 Scarecrow Effigy for Fiddlesticks (9) (caller's grant);
      R15-3  a trinket purchase without a same-ms trinket destroy replaces the held trinket (its undo, logged
             beforeId 0 / afterId X / goldGain 0, gives it back: rule 1);
      R15-4  rune grant (ItemStateIndex, rune_grants_16x): Biscuit Delivery (8345) 2010 at 2:00, 4:00, 6:00;
      (R15-5, the Triple Tonic grant, was removed by author decision 2026-09-25: the rule file's 'removed_rules';
             15.x replays grant no 2150 / 2151 / 2152)
      R15-6  Eye of the Herald: a loose destroy of a held trinket with no trinket bought at that ms puts 3513 into
             the trinket slot; the destroy of the held 3513 (used) gives the stored trinket back;
      R15-7  Viego: a lone loose trinket destroy and the destroy of a held 3513 are not possession swaps.
    """

    def __init__(self, stream, items_table, r, diag):
        self.items, self.r, self.diag = items_table, r, diag
        for k in ("r15_loose_transform", "r15_trinket_replaced", "r15_herald_swap", "r15_herald_used"):
            diag.setdefault(k, 0)
        self.stream = stream
        self.trinket_buy_ms = frozenset(e[0] for e in stream if e[1] == "buy" and self._trinket(e[2]))
        self.trinket_destroy_ms = frozenset(e[0] for e in stream if e[1] == "destroy" and self._trinket(e[2]))
        self.stored_trinket: Optional[int] = None

    def _trinket(self, i) -> bool:
        return is_trinket(self.items.get(int(i), {}))

    def viego_exempt(self, ts: int, iid: int, inv, n_loose: int) -> bool:          # R15-7
        herald = self.r["herald_eye"]
        if iid == herald and herald in inv:
            return True
        return n_loose == 1 and self._trinket(iid)

    def on_buy(self, ts, iid, inv, removed, restore):                              # R15-3
        herald = self.r["herald_eye"]
        if self._trinket(iid) and iid != herald and ts not in self.trinket_destroy_ms:
            held = [x for x in inv if self._trinket(x) and x != herald and x not in removed]
            if held:
                removed, restore = list(removed) + held, list(restore) + held
                self.diag["r15_trinket_replaced"] += 1
        return removed, restore

    def on_loose_destroy(self, ts, iid, inv):
        r = self.r
        if iid in r["loose_transforms"] and iid in inv:                            # R15-1
            self.diag["r15_loose_transform"] += 1
            return True, [iid], [r["loose_transforms"][iid]]
        herald = r["herald_eye"]
        if iid != herald and self._trinket(iid) and iid in inv and ts not in self.trinket_buy_ms:
            self.stored_trinket = iid                                              # R15-6
            self.diag["r15_herald_swap"] += 1
            return True, [iid], [herald]
        if iid == herald and herald in inv:
            back, self.stored_trinket = self.stored_trinket, None
            self.diag["r15_herald_used"] += 1
            return True, [herald], ([back] if back is not None else [])
        return False, [iid], []


def _replay_exact(shop_events, items_table, upto, since, grants, champion_id, diagnostics, takedowns=None,
                  rules16=None, role=None, rules15=None):
    tmap, proxies = _recipe_rules(items_table)
    diag = diagnostics if diagnostics is not None else {}
    for k in ("unowned_removal", "transforms", "proxies", "undo_restored", "granted", "viego_ignored", "reordered",
              "storage_ignored", "substituted"):
        diag.setdefault(k, 0)
    diag.setdefault("unowned_ids", {})
    # grants go first among same-ms events so a combine at that ms can consume them
    stream = sorted([(int(ts), "grant", int(i), 0.0, 0, 0) for ts, i in grants if int(ts) <= upto]
                    + [e for e in shop_events if e[0] <= upto], key=lambda e: e[0])
    x16 = _R16Context(stream, items_table, rules16, role, diag) if rules16 is not None else None
    if x16 is not None:
        stream = x16.stream
    x15 = _R15Context(stream, items_table, rules15, diag) if (rules15 is not None and x16 is None) else None
    stream = _consume_after_purchase(stream, items_table, diag, also=None if x16 is None else x16.reorder_also)
    # per ms: allocate the destroyed copies (a multiset) to the purchases of that ms, in timeline order
    consumed_by: Dict[int, List[int]] = {}                      # stream index of a purchase -> consumed ids
    swapped_by: Dict[int, List[int]] = {}                       # trinket purchase -> trinket destroyed at that ms
    combined_left: Dict[int, Counter] = {}                      # ms -> destroyed copies that went into a purchase
    loose_destroys: Dict[int, int] = defaultdict(int)          # destroys at a ms not explained by a purchase
    i = 0
    while i < len(stream):
        j = i
        while j < len(stream) and stream[j][0] == stream[i][0]:
            j += 1
        destroyed = Counter(stream[k][2] for k in range(i, j) if stream[k][1] == "destroy")
        avail = Counter(destroyed)
        for k in range(i, j):
            if stream[k][1] == "buy":
                consumed_by[k], n_sub = _allocate_components(stream[k][2], avail, items_table)
                diag["substituted"] += n_sub
                if is_trinket(items_table.get(stream[k][2], {})):
                    swapped_by[k] = [d for d in sorted(avail) if avail[d] > 0
                                     and is_trinket(items_table.get(d, {}))][:1]
        combined_left[stream[i][0]] = destroyed - avail
        loose_destroys[stream[i][0]] = sum(avail.values())
        i = j
    inv: List[int] = []
    gold_delta, stat_delta = 0.0, defaultdict(float)
    purchases: Dict[int, List[List[int]]] = defaultdict(list)    # components each purchase consumed (for undo)
    for k_ev, (ts, kind, iid, gold, before, after) in enumerate(stream):
        added, removed, delta_gold = [], [], 0.0
        if kind == "grant":
            added = [iid]
            diag["granted"] += 1
        elif kind == "buy":
            frm = list(items_table.get(iid, {}).get("from", []))
            consumed = list(consumed_by.get(k_ev, []))
            restore = list(consumed) + swapped_by.get(k_ev, [])
            # rule 3: buying the upgrade of a held proxy replaces it (and is priced as consuming it)
            if len(frm) == 1 and frm[0] in proxies and frm[0] in inv and not consumed:
                removed, consumed, restore = [frm[0]], [frm[0]], [frm[0]]
            added, delta_gold = [iid], -purchase_price(iid, consumed, items_table)
            if x16 is not None:
                removed, restore = x16.on_buy(ts, iid, inv, removed, restore)
            elif x15 is not None:
                removed, restore = x15.on_buy(ts, iid, inv, removed, restore)
            purchases[iid].append(restore)
        elif kind == "sell":
            removed, delta_gold = [iid], gold
            frm = list(items_table.get(iid, {}).get("from", []))
            # rule 3: selling an unobserved upgrade of a held proxy sells the proxy (3867 held for 3870)
            if iid not in inv and len(frm) == 1 and frm[0] in proxies and frm[0] in inv:
                removed = [frm[0]]
                diag["proxies"] += 1
        elif kind == "undo":
            removed, added, delta_gold = ([before] if before else []), ([after] if after else []), gold
            if x16 is not None and not before and not after:
                removed, added = x16.on_blind_undo(inv, purchases)
            elif before and not after and purchases.get(before):
                restored = purchases[before].pop()                # rule 1: exactly the copies destroyed for it
                added = added + restored
                diag["undo_restored"] += len(restored)
            elif not before and after and not gold and after in inv and purchases.get(after):
                # rule 1: the undo of a purchase that cost nothing (trinket swap, support-quest upgrade) is
                # logged as beforeId 0 / afterId X / goldGain 0 right after the purchase of X, not as a re-add
                restored = purchases[after].pop()
                removed, added = [after], restored
                diag["undo_restored"] += len(restored)
        else:                                                       # destroy: consumed / combined / transformed
            removed = [iid]
            left = combined_left.get(ts)
            is_combined = bool(left) and left.get(iid, 0) > 0
            if is_combined:
                left[iid] -= 1
            if not is_combined:
                it = items_table.get(iid, {})
                possession = takedowns is None or any(ts - VIEGO_POSSESSION_WINDOW_MS <= k <= ts for k in takedowns)
                viego_swap = champion_id == VIEGO_ID and possession and (loose_destroys[ts] >= 2
                                                                         or not is_consumable(it))
                handled16 = False
                if x16 is not None:
                    if viego_swap and x16.viego_exempt(ts, iid, inv, loose_destroys[ts]):
                        viego_swap = False                          # 16.x: quest completion / herald, not a swap
                    if not viego_swap:
                        handled16, removed, added = x16.on_loose_destroy(ts, iid, inv)
                elif x15 is not None:
                    if viego_swap and x15.viego_exempt(ts, iid, inv, loose_destroys[ts]):
                        viego_swap = False                          # 15.x: herald exchange / trinket, not a swap
                    if not viego_swap:
                        handled16, removed, added = x15.on_loose_destroy(ts, iid, inv)
                if handled16:
                    pass
                elif viego_swap:
                    removed = []                                    # possession swap, not consumption
                    diag["viego_ignored"] += 1
                elif iid in DESTROY_IS_NOT_REMOVAL:                 # rule 6
                    removed = []
                    diag["storage_ignored"] += 1
                elif iid in tmap:                                   # rule 2
                    added = [tmap[iid]]
                    diag["transforms"] += 1
                elif iid in proxies:                                # rule 3
                    removed, added = [], ([] if iid in inv else [iid])
                    diag["proxies"] += 1
        for x in added:
            inv.append(x)
        for x in removed:
            if x in inv:
                inv.remove(x)
            else:
                diag["unowned_removal"] += 1                        # rule 4
                diag["unowned_ids"][x] = diag["unowned_ids"].get(x, 0) + 1
        if x16 is not None:
            more = x16.after_event(ts, kind, iid, before, inv, removed)
            if more:
                inv.extend(more)
                added = added + more
        if since is not None and ts > since:
            gold_delta += delta_gold
            for x in added:
                for k, v in items_table.get(x, {}).get("stats", {}).items():
                    stat_delta[k] += v
            for x in removed:
                for k, v in items_table.get(x, {}).get("stats", {}).items():
                    stat_delta[k] -= v
    return inv, gold_delta, stat_delta


def magical_footwear_grant_ms(events: Iterable[dict], pid: int, runes: Optional[Mapping] = None) -> Optional[int]:
    """When rune 8304 (Magical Footwear) hands `pid` item 2422: 12:00 minus 45 s per takedown before it.

    None when runes are given and 8304 is not among them.  Uses CHAMPION_KILL events only; the grant time t_g is
    the earliest t with t >= 720 000 - 45 000 * takedowns(<= t), so it depends only on kills <= t_g.
    B3: the result is the earlier of that model and the player's first ITEM_* event on 2422 (itemId, beforeId or
    afterId), so a destroy / sell / upgrade of the boots never hits an unowned item.  Still causal: the grant is
    in the state at t iff min(model, first 2422 event) <= t, which needs only events <= t.
    """
    if runes is not None and MAGICAL_FOOTWEAR_RUNE not in {int(v) for k, v in runes.items()
                                                           if "rune" in str(k) and v is not None}:
        return None
    events = list(events)
    tk = sorted(int(e.get("timestamp", 0)) for e in events if e.get("type") == "CHAMPION_KILL" and (
        int(e.get("killerId", 0) or 0) == pid or pid in [int(a) for a in (e.get("assistingParticipantIds") or [])]))
    best = MAGICAL_FOOTWEAR_MS
    for k, t in enumerate(tk, start=1):
        best = min(best, max(t, MAGICAL_FOOTWEAR_MS - MAGICAL_FOOTWEAR_STEP_MS * k))
    for e in events:
        if (e.get("type") in SHOP_TYPES and int(e.get("participantId", 0) or 0) == pid
                and MAGICAL_FOOTWEAR_ITEM in (int(e.get("itemId", 0) or 0), int(e.get("beforeId", 0) or 0),
                                              int(e.get("afterId", 0) or 0))):
            best = min(best, int(e.get("timestamp", 0)))
    return int(best)


# ---------------------------------------------------------------------------------------------- features
def is_consumable(it: Mapping) -> bool:
    return bool(it.get("consumed")) or "Consumable" in (it.get("tags") or [])


def is_trinket(it: Mapping) -> bool:
    return "Trinket" in (it.get("tags") or [])


def is_boots(it: Mapping) -> bool:
    return "Boots" in (it.get("tags") or [])


def is_completed_item(iid: int, items_table: Mapping[int, dict]) -> bool:
    """Non-consumable, non-boots, gold_total >= 2000, and (depth >= 3, or no 'into', or only Ornn upgrades)."""
    it = items_table.get(iid)
    if it is None or is_consumable(it) or is_boots(it):
        return False
    if float(it.get("gold_total", 0)) < COMPLETED_MIN_GOLD:
        return False
    into = list(it.get("into") or [])
    only_ornn = bool(into) and all((items_table.get(t) or {}).get("requiredAlly") == "Ornn" for t in into)
    return (it.get("depth") or 0) >= 3 or not into or only_ornn


def boots_tier(iid: int, items_table: Mapping[int, dict]) -> int:
    """0 not boots, 1 basic boots (no recipe depth: 1001, 2422), 2 any upgraded boots (depth >= 2)."""
    it = items_table.get(iid)
    if it is None or not is_boots(it):
        return 0
    return 2 if (it.get("depth") or 0) >= 2 else 1


def player_item_vector(inventory: Sequence[int], items_table: Mapping[int, dict],
                       effect_flags: Optional[Mapping[str, frozenset]] = None) -> Dict[str, float]:
    """The ITEM_VECTOR_NAMES features of one inventory.  Ids absent from the table count only as unknown."""
    flags = effect_flags if effect_flags is not None else load_effect_flags()
    v = {n: 0.0 for n in ITEM_VECTOR_NAMES}
    held = set()
    for iid in inventory:
        it = items_table.get(int(iid))
        if it is None:
            v["unknown_item_count"] += 1.0
            continue
        held.add(int(iid))
        for k, val in (it.get("stats") or {}).items():
            if k in ITEM_STAT_KEYS:
                v[f"item_{k}"] += float(val)
        v["item_gold_owned"] += float(it.get("gold_total", 0))
        v["completed_item_count"] += 1.0 if is_completed_item(int(iid), items_table) else 0.0
        v["boots_tier"] = max(v["boots_tier"], float(boots_tier(int(iid), items_table)))
    for f in EFFECT_FLAGS:
        v[f"flag_{f}"] = 1.0 if held & flags[f] else 0.0
    return v


def _static_meta(meta: Mapping) -> dict:
    sm = (meta or {}).get("static_meta") or {}
    return ast.literal_eval(sm) if isinstance(sm, str) else dict(sm)


class ItemStateIndex:
    """Index a match's ITEM_* events once; answer inventories and item vectors at any t (events <= t only).

    champion_by_pid / runes_by_pid (pack meta static_meta) switch on the Viego rule and the Magical Footwear
    grant; without them neither applies.  `patch` selects the per-patch effect-flag sets (lifeline); without it
    the union over patches is used (identical in the 11 patches).
    16.x (exact mode, patch major >= 16; `patch` is required, a missing patch never switches them on): the rules of
    config/game_rules/item_rules_16x.json (_R16Context) with the role of each participant from its slot
    (pid 1-5 / 6-10 = TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY: 5,530/5,530 16.15 players, teamPosition) and, at 0 ms,
    the role-quest item of that role (TOP: 1222 when Teleport (12) is among the pre-game summoner spells
    `summoner_spells_by_pid`, else 1200) and the 3340 trinket.  rules_16x=False switches them off (the 16.15
    validation's 'before' replay), True forces them on; None (default) decides by the patch.
    15.x (exact mode, patch major == 15; `patch` is required): the rules of config/game_rules/item_rules_15x.json
    (_R15Context) and, per participant 1-10, the grants 3340 at 0 ms (3330 for Fiddlesticks, champion 9; needs
    champion_by_pid) and Biscuit Delivery 2010 at 2:00 / 4:00 / 6:00 (needs runes_by_pid; no Triple Tonic grant,
    R15-5 removed 2026-09-25).  rules_15x=False switches them off (the pre-15X replay), True forces them on (when
    no 16.x rule is active); None (default) decides by the patch.
    """

    def __init__(self, events: Iterable[dict], items_table: Mapping[int, dict], *, exact: bool = True,
                 effect_flags: Optional[Mapping[str, frozenset]] = None, participants: Iterable[int] = range(1, 11),
                 champion_by_pid: Optional[Mapping] = None, runes_by_pid: Optional[Mapping] = None,
                 patch: Optional[str] = None, summoner_spells_by_pid: Optional[Mapping] = None,
                 rules_16x: Optional[bool] = None, rules_15x: Optional[bool] = None):
        events = list(events)
        self.items = items_table
        self.exact = exact
        self.patch = None if patch is None else str(patch)
        self.flags = effect_flags if effect_flags is not None else load_effect_flags(items_table, patch=self.patch)
        self.shop = shop_events_by_participant(events, items_table, participants, assign_support_atlas=exact)
        self.champion = {int(k): int(v) for k, v in (champion_by_pid or {}).items()}
        self.grants: Dict[int, List[Tuple[int, int]]] = {pid: [] for pid in self.shop}
        for pid in self.shop:
            runes = (runes_by_pid or {}).get(str(pid), (runes_by_pid or {}).get(pid))
            if runes is not None:
                tg = magical_footwear_grant_ms(events, pid, runes)
                if tg is not None:
                    self.grants[pid].append((tg, MAGICAL_FOOTWEAR_ITEM))
        on16 = uses_rules_16x(self.patch) if rules_16x is None else bool(rules_16x)
        self.rules16 = load_rules_16x() if (exact and on16) else None
        self.role: Dict[int, Optional[str]] = {pid: None for pid in self.shop}
        if self.rules16 is not None:
            for pid in self.shop:
                if 1 <= pid <= 10:
                    self.role[pid] = self.rules16["role_by_slot"][(pid - 1) % 5]
                    start = role_quest_start(self.rules16, self.role[pid],
                                             (summoner_spells_by_pid or {}).get(str(pid),
                                                                                (summoner_spells_by_pid or {}).get(pid)))
                    trinket = self.rules16["start_trinket_by_champion"].get(self.champion.get(pid),
                                                                            self.rules16["start_trinket"])
                    runes = (runes_by_pid or {}).get(str(pid), (runes_by_pid or {}).get(pid))
                    self.grants[pid] = sorted([(0, start), (0, trinket)] + self.grants[pid]
                                              + rune_grants_16x(self.rules16, events, pid, runes),
                                              key=lambda g: g[0])
        on15 = uses_rules_15x(self.patch) if rules_15x is None else bool(rules_15x)
        self.rules15 = load_rules_15x() if (exact and on15 and self.rules16 is None) else None
        if self.rules15 is not None:
            for pid in self.shop:
                if 1 <= pid <= 10:
                    trinket = self.rules15["start_trinket_by_champion"].get(self.champion.get(pid),
                                                                            self.rules15["start_trinket"])
                    runes = (runes_by_pid or {}).get(str(pid), (runes_by_pid or {}).get(pid))
                    self.grants[pid] = sorted([(0, trinket)] + self.grants[pid]
                                              + rune_grants_16x(self.rules15, events, pid, runes),
                                              key=lambda g: g[0])
        self.takedowns: Dict[int, Optional[List[int]]] = {pid: None for pid in self.shop}
        for pid in self.shop:
            if self.champion.get(pid) == VIEGO_ID:
                self.takedowns[pid] = sorted(
                    int(e.get("timestamp", 0)) for e in events if e.get("type") == "CHAMPION_KILL" and (
                        int(e.get("killerId", 0) or 0) == pid
                        or pid in [int(a) for a in (e.get("assistingParticipantIds") or [])]))

    @classmethod
    def from_pack(cls, pack: Mapping, items_table: Optional[Mapping[int, dict]] = None, **kw) -> "ItemStateIndex":
        """From a cache pack ({'events', 'meta'}); the table and the effect-flag patch default to the pack's patch."""
        meta = pack.get("meta") or {}
        sm = _static_meta(meta)
        table = items_table if items_table is not None else load_item_table_v2(str(meta["patch"]))
        if "patch" not in kw and meta.get("patch") is not None:
            kw["patch"] = str(meta["patch"])
        return cls(pack["events"], table, champion_by_pid=sm.get("champion_by_pid"),
                   runes_by_pid=sm.get("runes_by_pid"), summoner_spells_by_pid=sm.get("summoner_spells_by_pid"), **kw)

    def replay(self, pid: int, t_ms: int, since_ms: Optional[int] = None, diagnostics: Optional[dict] = None):
        """replay_inventory for `pid` at t_ms with this index's grants, champion and takedowns:
        (inventory, gold delta, stat delta of the events in (since_ms, t_ms])."""
        pid = int(pid)
        return replay_inventory(self.shop[pid], self.items, int(t_ms), since_ms, exact=self.exact,
                                grants=self.grants.get(pid, ()), champion_id=self.champion.get(pid),
                                diagnostics=diagnostics, takedowns_ms=self.takedowns.get(pid),
                                rules16=self.rules16, role=self.role.get(pid), rules15=self.rules15)

    def inventory(self, pid: int, t_ms: int, diagnostics: Optional[dict] = None) -> List[int]:
        return self.replay(pid, t_ms, diagnostics=diagnostics)[0]

    def vector(self, pid: int, t_ms: int) -> Dict[str, float]:
        return player_item_vector(self.inventory(pid, t_ms), self.items, self.flags)

    def vectors_at(self, t_ms: int) -> Dict[int, Dict[str, float]]:
        return {pid: self.vector(pid, t_ms) for pid in self.shop}


def item_vectors_at(events: Iterable[dict], items_table: Mapping[int, dict], t_ms: int, *, exact: bool = True,
                    effect_flags: Optional[Mapping[str, frozenset]] = None, champion_by_pid: Optional[Mapping] = None,
                    runes_by_pid: Optional[Mapping] = None, patch: Optional[str] = None) -> Dict[int, Dict[str, float]]:
    """All 10 players' item vectors at t_ms from events with timestamp <= t_ms."""
    return ItemStateIndex(events, items_table, exact=exact, effect_flags=effect_flags,
                          champion_by_pid=champion_by_pid, runes_by_pid=runes_by_pid, patch=patch).vectors_at(t_ms)
