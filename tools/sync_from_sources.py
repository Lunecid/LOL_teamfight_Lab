"""Copy code and small records from the two source workspaces into this repository.

Sources (never modified):
  B = C:\\Users\\todtj\\문서\\LOL_Teamfight                     -> master_ver/
  C = C:\\Users\\todtj\\.codex\\visualizations\\2026\\10\\15\\01a13f40-d38f-76e3-9108-1b668280290d
        github_export_lol_teamfight/{research_pipeline,common_training_r1} -> research_pipeline/, common_training_r1/
        everything else small and non-data                              -> research_pipeline/_local/

Repeatable: run again to refresh; files whose size and SHA-256 already match are skipped.
Default is a dry run. Use --apply to copy. Each run writes manifests under tools/manifests/:
  MANIFEST_<target>.tsv   relpath, bytes, sha256, source path
  EXCLUDED_<target>.tsv   relpath, bytes, reason

An apply into a Master tree with module-layout.json is refused: that tree is a
derived code revision, not a mirror. Refresh original sources in a separate
unmodified checkout and review the changes before importing them here.

Usage: python tools/sync_from_sources.py [master_ver|research_pipeline|all] [--apply] [--max-mb 20]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

HUB = Path(__file__).resolve().parent.parent
B = Path(r"C:\Users\todtj\문서\LOL_Teamfight")
C = Path(r"C:\Users\todtj\.codex\visualizations\2026\10\15\01a13f40-d38f-76e3-9108-1b668280290d")
MANIFESTS = HUB / "tools" / "manifests"

SKIP_DIR_NAMES = {"__pycache__", ".pytest_cache", ".git", ".mypy_cache", ".ruff_cache"}
SKIP_DIR_PREFIX = (".pytest", "tmp", ".tmp", "final_tests_", "io_tests_", "final_pytest_cache", "io_pytest_cache")
B_ROOT_SKIP = {"outputs", "worktrees", ".tmp", "tmp", ".cursor-handoff", ".superpowers", ".git", ".codex", ".claude", ".agents"}
B_ROOT_FILE_SKIP = {"grep.exe.stackdump", ".gitignore"}
DATA_EXT = {".npz", ".npy", ".csv", ".jsonl", ".joblib", ".zip", ".pt", ".pkl", ".parquet", ".feather", ".pyc"}
C_PRODUCT_PARTS = {"production_01", "audit_01", "full_01", "full_02", "smoke_01", "shards", "review_02", "RAW_CHECKPOINTS", "RAW_CHECKPOINTS_v2"}


def lp(p: Path | str) -> str:
    s = str(p)
    return s if s.startswith("\\\\?\\") else "\\\\?\\" + s


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(lp(p), "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def skip_dir(name: str) -> bool:
    return name in SKIP_DIR_NAMES or name.startswith(SKIP_DIR_PREFIX)


def walk(base: Path):
    errors: list[str] = []
    for dp, dn, fn in os.walk(lp(base), onerror=lambda e: errors.append(str(e.filename))):
        dn[:] = [d for d in dn if not skip_dir(d)]
        for f in fn:
            yield Path(dp.removeprefix("\\\\?\\")) / f
    for e in errors:
        print("  unreadable dir (skipped):", e.removeprefix("\\\\?\\"), file=sys.stderr)


def plan_master_ver(max_mb: float):
    items, excluded = [], []
    for entry in sorted(os.listdir(lp(B))):
        src = B / entry
        if entry in B_ROOT_SKIP or entry.startswith((".pytest", "tmp")) or entry in B_ROOT_FILE_SKIP:
            continue
        if src.is_dir():
            for f in walk(src):
                items.append((f, Path("master_ver") / f.relative_to(B)))
        else:
            items.append((src, Path("master_ver") / entry))
    return filter_size(items, excluded, max_mb), excluded


def plan_research_pipeline(max_mb: float):
    items, excluded = [], []
    exp = C / "github_export_lol_teamfight"
    for sub in ("research_pipeline", "common_training_r1"):
        for f in walk(exp / sub):
            items.append((f, Path(sub) / f.relative_to(exp / sub)))
    for name in ("stage_research_code.py", "files_to_publish.json", "README.md"):
        if (exp / name).exists():
            items.append((exp / name, Path("research_pipeline") / "export_tools" / name))
    # local-only: originals, docs and small records of C (not data products)
    for entry in sorted(os.listdir(lp(C))):
        src = C / entry
        if entry == "github_export_lol_teamfight" or skip_dir(entry):
            continue
        files = list(walk(src)) if src.is_dir() else [src]
        for f in files:
            rel = f.relative_to(C)
            if any(part in C_PRODUCT_PARTS or part.startswith("attempt_") for part in rel.parts):
                excluded.append((rel, f.stat().st_size, "run product dir"))
                continue
            if f.suffix.lower() in DATA_EXT:
                excluded.append((rel, f.stat().st_size, f"data ext {f.suffix.lower()}"))
                continue
            if f.suffix.lower() == ".json" and f.stat().st_size > 1_000_000:
                excluded.append((rel, f.stat().st_size, "json > 1 MB"))
                continue
            items.append((f, Path("research_pipeline") / "_local" / rel))
    return filter_size(items, excluded, max_mb), excluded


def filter_size(items, excluded, max_mb):
    keep = []
    for src, dst in items:
        sz = src.stat().st_size
        if sz > max_mb * 1e6:
            excluded.append((dst, sz, f"> {max_mb:g} MB"))
        else:
            keep.append((src, dst))
    return keep


def run(target: str, items, excluded, apply: bool):
    total = sum(s.stat().st_size for s, _ in items)
    print(f"[{target}] copy {len(items)} files {total/1e6:.0f} MB; excluded {len(excluded)} files {sum(e[1] for e in excluded)/1e6:.0f} MB")
    if not apply:
        return
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    done = skipped = 0
    rows = []
    for src, dst in items:
        out = HUB / dst
        sz = src.stat().st_size
        h = sha256(src)
        if out.exists() and out.stat().st_size == sz and sha256(out) == h:
            skipped += 1
        else:
            os.makedirs(lp(out.parent), exist_ok=True)
            shutil.copy2(lp(src), lp(out))
            if sha256(out) != h:
                raise SystemExit(f"hash mismatch after copy: {dst}")
            done += 1
        rows.append((dst.as_posix(), sz, h, str(src)))
    with open(MANIFESTS / f"MANIFEST_{target}.tsv", "w", encoding="utf-8", newline="") as fh:
        fh.write("relpath\tbytes\tsha256\tsource\n")
        for r in sorted(rows):
            fh.write("\t".join(map(str, r)) + "\n")
    with open(MANIFESTS / f"EXCLUDED_{target}.tsv", "w", encoding="utf-8", newline="") as fh:
        fh.write("relpath\tbytes\treason\n")
        for rel, sz, why in sorted(excluded, key=lambda e: str(e[0])):
            fh.write(f"{Path(rel).as_posix()}\t{sz}\t{why}\n")
    print(f"[{target}] copied {done}, already identical {skipped}; manifests in {MANIFESTS}")


def refuse_overlay_overwrite(target: str, apply: bool):
    """Stop a mirror refresh before it replaces a reorganized Master tree."""
    if apply and target in ("master_ver", "all") and (HUB / "master_ver" / "module-layout.json").is_file():
        raise SystemExit(
            "Master contains a derived module layout; --apply would overwrite its "
            "compatibility entry points. Refresh original sources in a separate "
            "unmodified checkout, then review and integrate the changes."
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", choices=["master_ver", "research_pipeline", "all"], nargs="?", default="all")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--max-mb", type=float, default=20.0)
    a = ap.parse_args()
    refuse_overlay_overwrite(a.target, a.apply)
    if a.target in ("master_ver", "all"):
        items, ex = plan_master_ver(a.max_mb)
        run("master_ver", items, ex, a.apply)
    if a.target in ("research_pipeline", "all"):
        items, ex = plan_research_pipeline(a.max_mb)
        run("research_pipeline", items, ex, a.apply)


if __name__ == "__main__":
    main()
