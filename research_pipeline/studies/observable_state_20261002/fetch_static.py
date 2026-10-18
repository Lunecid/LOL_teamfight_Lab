"""Fetch only public patch-pinned Riot selection tables, without credentials."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


def fetch_tables(directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {"source": "public Riot Data Dragon", "match_API_called": False,
                "credentials_used": False, "tables": {}}
    try:
        for patch in ("15.14", "15.15", "15.16"):
            for kind in ("summoner", "runesReforged"):
                url = f"https://ddragon.leagueoflegends.com/cdn/{patch}.1/data/en_US/{kind}.json"
                with urlopen(url, timeout=30) as response:
                    raw = response.read((1 << 20) + 1)
                if len(raw) > (1 << 20):
                    raise ValueError("unexpected public selection table size")
                payload = json.loads(raw)
                if kind == "summoner":
                    ids = [int(row["key"]) for row in payload["data"].values()]
                else:
                    ids = [row["id"] for row in payload]
                    ids += [r["id"] for style in payload for slot in style["slots"] for r in slot["runes"]]
                if not ids or len(ids) != len(set(ids)) or any(type(i) is not int or i <= 0 for i in ids):
                    raise ValueError("canonical selection table IDs invalid")
                name = f"{kind}_{patch}.1.json"
                (directory / name).write_bytes(raw)
                manifest["tables"][name] = {"url": url, "sha256": hashlib.sha256(raw).hexdigest(),
                                             "bytes": len(raw), "n_IDs": len(ids)}
        manifest["complete"] = True
    except Exception as exc:
        manifest["complete"] = False
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (directory / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    print(json.dumps(fetch_tables(parser.parse_args().out)))
