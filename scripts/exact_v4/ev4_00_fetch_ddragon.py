"""Stage 0 (v4-exact): fetch Data Dragon item / champion tables into config/game_rules/datadragon_v2/.

Differences from scripts/fetch_datadragon_tables.py (left untouched, still feeds datadragon/):
  * the Data Dragon version is resolved strictly by prefix ('15.14' -> '15.14.<n>'); a patch with no
    matching version is reported and skipped, never guessed;
  * raw responses are stored byte-for-byte (sha256 and size recorded) outside the repo;
  * items carry tags, depth, maps['11'], inStore, requiredAlly/requiredChampion, specialRecipe;
  * champions carry key, tags, partype, info, the full Data Dragon stats block (incl. attackrange)
    and the old base/growth layout;
  * index.json is merged: patches already in the index are never dropped.

Network: https://ddragon.leagueoflegends.com only (versions.json, item.json, championFull.json).
Single process.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.fetch_datadragon_tables import CHAMP_URL, ITEM_STAT_MAP, ITEM_URL  # noqa: E402
from analysis.ability_range import DDRAGON_VERSIONS  # noqa: E402

PATCHES = ["15.14", "15.15", "15.16", "15.18", "15.19", "15.20", "15.21", "15.22", "16.13", "16.14", "16.15"]
RAW_DIR = Path(r"C:/Users/todtj/문서/LOL_Teamfight/outputs/reest_exact_v4_20260925/datadragon_raw")
OUT_DIR = ROOT / "config/game_rules/datadragon_v2"
ALLOWED_HOST = "https://ddragon.leagueoflegends.com/"
SR_MAP_ID = "11"  # Summoner's Rift


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fetch_bytes(url: str, timeout: float = 60.0) -> bytes:
    if not url.startswith(ALLOWED_HOST):
        raise ValueError(f"refusing non-Data-Dragon URL {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "lol-teamfight-ev4/0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def get_raw(url: str, path: Path, log: list) -> bytes:
    """Return the raw bytes of `path`, downloading `url` once if the file is missing."""
    if path.exists():
        b = path.read_bytes()
        log.append({"file": path.name, "action": "cached", "bytes": len(b)})
        return b
    b = fetch_bytes(url)
    json.loads(b.decode("utf-8"))  # must parse before we keep it
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(b)
    tmp.replace(path)
    log.append({"file": path.name, "action": "downloaded", "bytes": len(b), "url": url})
    return b


def _vkey(v: str):
    return tuple(int(x) for x in v.split("."))


def resolve_version(patch: str, versions: list) -> tuple:
    """('15.14', versions) -> (chosen, candidates).  chosen is None when no 'patch.<int>' version exists."""
    pat = re.compile(r"^" + re.escape(patch) + r"\.\d+$")
    cands = sorted((v for v in versions if isinstance(v, str) and pat.match(v)), key=_vkey)
    return (cands[-1] if cands else None), cands


def extract_items(items: dict) -> dict:
    out = {}
    for iid, it in items.items():
        raw_stats = it.get("stats") or {}
        g = it.get("gold") or {}
        rec = {
            "name": it.get("name"),
            "gold_total": int(g.get("total", 0)), "gold_base": int(g.get("base", 0)),
            "gold_sell": int(g.get("sell", 0)), "gold_purchasable": bool(g.get("purchasable", False)),
            "from": [int(x) for x in it.get("from", [])], "into": [int(x) for x in it.get("into", [])],
            "consumed": bool(it.get("consumed", False)),
            "stats": {ITEM_STAT_MAP[k]: float(v) for k, v in raw_stats.items() if k in ITEM_STAT_MAP},
            "unmapped_stat_keys": sorted(k for k in raw_stats if k not in ITEM_STAT_MAP),
            "tags": list(it.get("tags", [])),
            "depth": (int(it["depth"]) if "depth" in it else None),  # absent in Data Dragon for basic items
            "map_sr": bool((it.get("maps") or {}).get(SR_MAP_ID, False)),
            "inStore": bool(it.get("inStore", True)),  # Data Dragon omits the key when true
            "requiredAlly": it.get("requiredAlly"),
            "requiredChampion": it.get("requiredChampion"),
        }
        if "specialRecipe" in it:
            rec["specialRecipe"] = int(it["specialRecipe"])
        out[int(iid)] = rec
    return out


def extract_champions(champs: dict) -> dict:
    out = {}
    for cid, ch in champs.items():
        s = ch["stats"]
        out[int(ch["key"])] = {
            "key": int(ch["key"]), "name": cid, "display_name": ch.get("name"),
            "tags": list(ch.get("tags", [])), "partype": ch.get("partype"),
            "info": {k: ch.get("info", {}).get(k) for k in ("attack", "defense", "magic", "difficulty")},
            "attackrange": float(s["attackrange"]),
            "stats": {k: float(v) for k, v in s.items()},
            # same layout as config/game_rules/datadragon/champions_<patch>.json
            "base": {
                "healthMax": s["hp"], "powerMax": s["mp"], "armor": s["armor"], "magicResist": s["spellblock"],
                "attackDamage": s["attackdamage"], "attackSpeed": s["attackspeed"], "movementSpeed": s["movespeed"],
                "healthRegen": s["hpregen"], "powerRegen": s["mpregen"]},
            "growth": {
                "healthMax": s["hpperlevel"], "powerMax": s["mpperlevel"], "armor": s["armorperlevel"],
                "magicResist": s["spellblockperlevel"], "attackDamage": s["attackdamageperlevel"],
                "attackSpeed_pct": s["attackspeedperlevel"], "healthRegen": s["hpregenperlevel"],
                "powerRegen": s["mpregenperlevel"]},
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patches", default=",".join(PATCHES))
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    a = ap.parse_args()
    a.raw_dir.mkdir(parents=True, exist_ok=True)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    log: list = []
    vb = get_raw(DDRAGON_VERSIONS, a.raw_dir / "versions.json", log)
    versions = json.loads(vb.decode("utf-8"))

    idx_path = a.out_dir / "index.json"
    index = json.loads(idx_path.read_text(encoding="utf-8")) if idx_path.exists() else {}
    patches_idx = dict(index.get("patches", {}))
    unresolved = []
    for patch in [p.strip() for p in a.patches.split(",") if p.strip()]:
        v, cands = resolve_version(patch, versions)
        if v is None:
            unresolved.append(patch)
            print(f"[ddragon-v2] {patch}: NO Data Dragon version with prefix '{patch}.' - skipped (not guessed)", flush=True)
            continue
        raw_item = a.raw_dir / f"item_{v}.json"
        raw_champ = a.raw_dir / f"championFull_{v}.json"
        ib = get_raw(ITEM_URL.format(v=v), raw_item, log)
        cb = get_raw(CHAMP_URL.format(v=v), raw_champ, log)
        items = extract_items(json.loads(ib.decode("utf-8"))["data"])
        champs = extract_champions(json.loads(cb.decode("utf-8"))["data"])
        f_items, f_champs = a.out_dir / f"items_{patch}.json", a.out_dir / f"champions_{patch}.json"
        f_items.write_text(json.dumps({"patch": patch, "version": v, "items": items}, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        f_champs.write_text(json.dumps({"patch": patch, "version": v, "champions": champs}, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        unmapped = sorted({k for it in items.values() for k in it["unmapped_stat_keys"]})
        untagged = sorted(c["name"] for c in champs.values() if not c["tags"])
        entry = {
            "version": v, "version_candidates": cands,
            "items": len(items), "items_sr_instore": sum(1 for it in items.values() if it["map_sr"] and it["inStore"]),
            "champions": len(champs), "champions_untagged": untagged,
            # Mode-variant entries (e.g. 16.15 ships 60 'Jade_*' champions, keys 60001+); kept, flagged here.
            "champions_key_ge_10000": sorted({c["name"].split("_")[0] + "_*" for c in champs.values() if c["key"] >= 10000}),
            "champions_key_ge_10000_n": sum(1 for c in champs.values() if c["key"] >= 10000),
            "unmapped_item_stat_keys": unmapped,
            "raw": {
                "item": {"file": raw_item.name, "bytes": len(ib), "sha256": sha256_bytes(ib)},
                "championFull": {"file": raw_champ.name, "bytes": len(cb), "sha256": sha256_bytes(cb)},
            },
            "extract_sha256": {f_items.name: sha256_bytes(f_items.read_bytes()), f_champs.name: sha256_bytes(f_champs.read_bytes())},
            "built_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        prev = patches_idx.get(patch)
        if prev and prev.get("version") != v:
            entry["previous"] = prev  # keep, never drop
        patches_idx[patch] = entry
        print(f"[ddragon-v2] {patch} -> {v} (candidates {cands}): {len(items)} items, {len(champs)} champions, "
              f"raw {len(ib)+len(cb):,} B, untagged {untagged}, unmapped stat keys {unmapped}", flush=True)

    index.update({
        "stat_map": ITEM_STAT_MAP,
        "patches": dict(sorted(patches_idx.items(), key=lambda kv: _vkey(kv[0]))),
        "versions_json": {"bytes": len(vb), "sha256": sha256_bytes(vb)},
        "raw_dir": str(a.raw_dir),
        "growth_formula": "stat(L) = base + growth * (L-1) * (0.7025 + 0.0175*(L-1))",
        "note": ("Data Dragon stats omit ability haste, lethality/armor pen, magic pen, omnivamp; those fields are held, "
                 "not reconstructed. depth=null means the key is absent (basic item). map_sr = maps['11']."),
        "unresolved_patches": sorted((set(index.get("unresolved_patches", [])) | set(unresolved)) - set(patches_idx)),
    })
    idx_path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"downloads": log, "unresolved": unresolved}, indent=1), flush=True)
    return 1 if unresolved else 0


if __name__ == "__main__":
    sys.exit(main())
