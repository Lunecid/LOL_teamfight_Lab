"""Fetch the Data Dragon item and champion tables for the corpus patches and keep compact extracts.

Evidence-based state reconstruction needs two reference tables the timeline does not carry:
  * item gold and stat contributions  (ITEM_PURCHASED/SOLD/UNDO events give only itemId)
  * champion base stats and per-level growth (LEVEL_UP events give only the new level)

Raw files are cached under outputs/datadragon_raw (gitignored); the extracts written to
config/game_rules/datadragon/ are small and versioned.  Data Dragon's `stats` block omits
several modern stats (ability haste, lethality, omnivamp, magic penetration); those are left
out of the extract and the reconstruction reports them as 'held' rather than guessing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.ability_range import DDRAGON_VERSIONS, fetch_json, version_for_patch

ITEM_URL = "https://ddragon.leagueoflegends.com/cdn/{v}/data/en_US/item.json"
CHAMP_URL = "https://ddragon.leagueoflegends.com/cdn/{v}/data/en_US/championFull.json"

# Data Dragon stat key -> timeline championStats field it contributes to.
ITEM_STAT_MAP = {
    "FlatPhysicalDamageMod": "attackDamage", "FlatMagicDamageMod": "abilityPower",
    "FlatArmorMod": "armor", "FlatSpellBlockMod": "magicResist",
    "FlatHPPoolMod": "healthMax", "FlatMPPoolMod": "powerMax",
    "PercentAttackSpeedMod": "attackSpeed_pct", "FlatMovementSpeedMod": "movementSpeed",
    "PercentMovementSpeedMod": "movementSpeed_pct", "FlatCritChanceMod": "critChance",
    "PercentLifeStealMod": "lifesteal", "FlatHPRegenMod": "healthRegen",
    "FlatMPRegenMod": "powerRegen",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patches", default="15.14,15.15,15.16")
    ap.add_argument("--raw-dir", type=Path, default=ROOT / "outputs/datadragon_raw")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "config/game_rules/datadragon")
    a = ap.parse_args()
    a.raw_dir.mkdir(parents=True, exist_ok=True)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    versions = fetch_json(DDRAGON_VERSIONS)
    index = {}
    for patch in a.patches.split(","):
        v = version_for_patch(patch, versions)
        raw_item, raw_champ = a.raw_dir / f"item_{v}.json", a.raw_dir / f"championFull_{v}.json"
        if not raw_item.exists():
            raw_item.write_text(json.dumps(fetch_json(ITEM_URL.format(v=v))), encoding="utf-8")
        if not raw_champ.exists():
            raw_champ.write_text(json.dumps(fetch_json(CHAMP_URL.format(v=v))), encoding="utf-8")
        items = json.loads(raw_item.read_text(encoding="utf-8"))["data"]
        champs = json.loads(raw_champ.read_text(encoding="utf-8"))["data"]
        item_extract = {}
        for iid, it in items.items():
            stats = {ITEM_STAT_MAP[k]: float(val) for k, val in (it.get("stats") or {}).items() if k in ITEM_STAT_MAP}
            item_extract[int(iid)] = {"name": it.get("name"), "gold_total": int(it["gold"]["total"]),
                                      "gold_base": int(it["gold"]["base"]), "gold_sell": int(it["gold"]["sell"]),
                                      "from": [int(x) for x in it.get("from", [])], "into": [int(x) for x in it.get("into", [])],
                                      "consumed": bool(it.get("consumed", False)), "stats": stats,
                                      "unmapped_stat_keys": sorted(k for k in (it.get("stats") or {}) if k not in ITEM_STAT_MAP)}
        champ_extract = {}
        for name, ch in champs.items():
            s = ch["stats"]
            champ_extract[int(ch["key"])] = {"name": name, "base": {
                "healthMax": s["hp"], "powerMax": s["mp"], "armor": s["armor"], "magicResist": s["spellblock"],
                "attackDamage": s["attackdamage"], "attackSpeed": s["attackspeed"], "movementSpeed": s["movespeed"],
                "healthRegen": s["hpregen"], "powerRegen": s["mpregen"]}, "growth": {
                "healthMax": s["hpperlevel"], "powerMax": s["mpperlevel"], "armor": s["armorperlevel"],
                "magicResist": s["spellblockperlevel"], "attackDamage": s["attackdamageperlevel"],
                "attackSpeed_pct": s["attackspeedperlevel"], "healthRegen": s["hpregenperlevel"],
                "powerRegen": s["mpregenperlevel"]}}
        (a.out_dir / f"items_{patch}.json").write_text(json.dumps({"patch": patch, "version": v, "items": item_extract}, ensure_ascii=False), encoding="utf-8")
        (a.out_dir / f"champions_{patch}.json").write_text(json.dumps({"patch": patch, "version": v, "champions": champ_extract}, ensure_ascii=False), encoding="utf-8")
        unmapped = sorted({k for it in item_extract.values() for k in it["unmapped_stat_keys"]})
        index[patch] = {"version": v, "items": len(item_extract), "champions": len(champ_extract),
                        "raw_bytes": raw_item.stat().st_size + raw_champ.stat().st_size, "unmapped_item_stat_keys": unmapped}
        print(f"[ddragon] {patch} -> {v}: {len(item_extract)} items, {len(champ_extract)} champions, "
              f"raw {index[patch]['raw_bytes']/1e6:.1f} MB, unmapped stat keys {unmapped}", flush=True)
    (a.out_dir / "index.json").write_text(json.dumps({"stat_map": ITEM_STAT_MAP, "patches": index,
        "growth_formula": "stat(L) = base + growth * (L-1) * (0.7025 + 0.0175*(L-1))",
        "note": "Data Dragon stats omit ability haste, lethality/armor pen, magic pen, omnivamp; those fields are held, not reconstructed"}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
