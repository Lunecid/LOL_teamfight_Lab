"""Fetch immutable public 15.14.1 static data, four workers, without match data."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parent / "knowledge_15_14_v1"
RAW = ROOT / "sources"
VERSION = "15.14.1"
BASE = f"https://ddragon.leagueoflegends.com/cdn/{VERSION}/data/en_US"


def fetch(name, url):
    target = RAW / name
    target.parent.mkdir(parents=True, exist_ok=True)
    start = datetime.now(timezone.utc).isoformat()
    errors = []
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "PatchKnowledgeResearch/1.0"})
            with urllib.request.urlopen(request, timeout=25) as response:
                body = response.read()
                status = response.status
            if name.endswith(".json"):
                json.loads(body)
            target.write_bytes(body)
            return {"path": str(target.relative_to(ROOT)), "url": url, "status": status,
                    "retrieved_at": start, "sha256": hashlib.sha256(body).hexdigest(),
                    "bytes": len(body), "attempt": attempt + 1}
        except Exception as exc:
            errors.append(type(exc).__name__ + ": " + str(exc))
            if attempt < 2:
                time.sleep(0.4 * (attempt + 1))
    return {"path": str(target.relative_to(ROOT)), "url": url, "retrieved_at": start, "errors": errors}


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    index_receipt = fetch("ddragon/champion.json", BASE + "/champion.json")
    if "errors" in index_receipt:
        raise RuntimeError(index_receipt)
    index = json.loads((RAW / "ddragon/champion.json").read_text(encoding="utf-8"))
    assert index["version"] == VERSION
    jobs = [(f"ddragon/champion/{name}.json", f"{BASE}/champion/{name}.json") for name in sorted(index["data"])]
    jobs += [(f"ddragon/{name}.json", f"{BASE}/{name}.json") for name in ["item", "summoner", "runesReforged", "map"]]
    receipts = [index_receipt]
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = {pool.submit(fetch, name, url): name for name, url in jobs}
        for n, future in enumerate(as_completed(pending), 1):
            receipts.append(future.result())
            if n % 25 == 0:
                print(json.dumps({"completed": n, "total": len(jobs)}, ensure_ascii=False), flush=True)
    receipts.sort(key=lambda row: row["path"])
    result = {"build": VERSION, "locale": "en_US", "expected_champions": len(index["data"]),
              "receipts": receipts, "errors": [r for r in receipts if "errors" in r]}
    (ROOT / "collection_manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"expected_champions": len(index["data"]), "files": len(receipts), "errors": len(result["errors"])}), flush=True)


if __name__ == "__main__":
    main()
