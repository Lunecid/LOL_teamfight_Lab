"""Stage 0 (v4-exact): input manifest.

Writes <OUT>/records/manifest_<UTC timestamp>.json with
  * worktree git HEAD, sha256 of `git diff HEAD` and of `git status --porcelain` (when git is available);
  * sha256 of the canonical JSON of the 'v4-exact' preset dict;
  * sha256 + size of every file in config/game_rules/datadragon_v2/;
  * sha256 + size of scripts/exact_v4/*.py;
  * sha256 + size of the R2 baseline files and the diagnostic kill table.
Single process; files are hashed in 8 MB chunks.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core.presets import PRESETS  # noqa: E402

OUT_ROOT = Path(r"C:/Users/todtj/문서/LOL_Teamfight/outputs/reest_exact_v4_20260925")
R2_DIR = Path(r"C:/Users/todtj/AppData/Local/Temp/claude/C--Users-todtj-Downloads-master-ver2/"
              r"535bfc56-69fd-4aa3-9b98-752823a3b74c/scratchpad/redesign_r2")
BASELINES = {
    "r2_finals_kill_full": R2_DIR / "finals_r2_kill_full.tsv",
    "r2_results": R2_DIR / "r2_results.json",
    "diag_kills_full": Path(r"C:/Users/todtj/문서/LOL_Teamfight/outputs/diag_survival_dbscan_20260925/p1/kills_full.npz"),
}
PRESET = "v4-exact"


def sha256_file(p: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def file_rec(p: Path) -> dict:
    if not p.exists():
        return {"path": str(p), "exists": False}
    return {"path": str(p), "exists": True, "bytes": p.stat().st_size, "sha256": sha256_file(p)}


def git_info() -> dict:
    def run(*args):
        r = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, timeout=120)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.decode("utf-8", "replace")[-500:])
        return r.stdout
    try:
        head = run("rev-parse", "HEAD").decode().strip()
        diff = run("diff", "HEAD", "--binary")
        status = run("status", "--porcelain", "--untracked-files=all")
        return {"available": True, "head": head,
                "diff_head_sha256": hashlib.sha256(diff).hexdigest(), "diff_head_bytes": len(diff),
                "status_porcelain_sha256": hashlib.sha256(status).hexdigest(),
                "status_porcelain": status.decode("utf-8", "replace").splitlines()}
    except Exception as e:  # git missing or not a repo
        return {"available": False, "error": str(e)}


def preset_hash(name: str) -> dict:
    d = PRESETS[name]
    canon = json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return {"name": name, "values": d, "sha256": hashlib.sha256(canon.encode("utf-8")).hexdigest(),
            "canonical": "json.dumps(sort_keys=True, separators=(',',':'), ensure_ascii=False), utf-8"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    a = ap.parse_args()
    now = _dt.datetime.now(_dt.timezone.utc)
    rec_dir = a.out_root / "records"
    rec_dir.mkdir(parents=True, exist_ok=True)
    dd_dir = ROOT / "config/game_rules/datadragon_v2"
    manifest = {
        "created_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stage": "v4-exact stage 0",
        "worktree": str(ROOT),
        "python": sys.version.split()[0],
        "git": git_info(),
        "preset": preset_hash(PRESET),
        "datadragon_v2": {p.name: file_rec(p) for p in sorted(dd_dir.glob("*")) if p.is_file()},
        "scripts_exact_v4": {p.name: file_rec(p) for p in sorted((ROOT / "scripts/exact_v4").glob("*.py"))},
        "baselines": {k: file_rec(p) for k, p in BASELINES.items()},
        "not_yet_recorded": ["match lists (defined by the stage-1/2 detection scripts)"],
    }
    missing = [k for k, v in manifest["baselines"].items() if not v["exists"]]
    out = rec_dir / f"manifest_{now.strftime('%Y%m%dT%H%M%SZ')}.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[manifest] wrote {out}")
    print(f"[manifest] git={manifest['git'].get('head', manifest['git'].get('error'))} preset_sha256={manifest['preset']['sha256']}")
    print(f"[manifest] datadragon_v2 files={len(manifest['datadragon_v2'])} scripts={list(manifest['scripts_exact_v4'])} missing_baselines={missing}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
