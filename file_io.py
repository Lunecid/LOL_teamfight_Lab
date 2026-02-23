from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# [FIX] write_log moved — using local stub
import logging
_logger = logging.getLogger(__name__)
def write_log(msg, fp=None):
    _logger.info(msg)
    if fp is not None:
        from pathlib import Path
        Path(fp).parent.mkdir(parents=True, exist_ok=True)
        with open(fp, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')



RefKeyFn = Callable[[Any], str]


def ensure_dir(p: Path) -> Path:
    """Create directory if missing and return it."""
    p.mkdir(parents=True, exist_ok=True)
    return p


def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _ensure_parent(fp: Path) -> None:
    fp.parent.mkdir(parents=True, exist_ok=True)


def save_text_lines(fp: Path, lines: List[str]) -> None:
    _ensure_parent(fp)
    with open(fp, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(str(ln).rstrip("\n") + "\n")


def save_kv_csv(
    fp: Path,
    rows: Iterable[Tuple[str, float]],
    k: str = "feature",
    v: str = "importance",
) -> None:
    _ensure_parent(fp)
    with open(fp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([k, v])
        for key, val in rows:
            w.writerow([str(key), float(val)])


def _default_ref_key() -> RefKeyFn:
    from fight_types import ref_key  # local import to avoid import cycles

    return ref_key


def dump_fight_refs_csv(out_fp: Path, refs: List[Any], split: str, ref_key_fn: Optional[RefKeyFn] = None) -> None:
    """Dump FightRef list into a CSV for quick debugging/inspection."""
    ref_key_fn = ref_key_fn or _default_ref_key()

    _ensure_parent(out_fp)

    fieldnames = [
        "split",
        "ref_key",
        "match_id",
        "patch",
        "t_start",
        "t_start_ts",
        "t_start_min",
        "t_start_sec",
        "t_start_ms_approx",
    ]

    with open(out_fp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in refs:
            t_start = int(getattr(r, "t_start", 0))
            t_start_ts = int(getattr(r, "t_start_ts", -1))
            w.writerow(
                {
                    "split": split,
                    "ref_key": ref_key_fn(r),
                    "match_id": str(getattr(r, "match_id", "")),
                    "patch": str(getattr(r, "patch", "")),
                    "t_start": t_start,
                    "t_start_ts": t_start_ts,
                    "t_start_min": float(t_start),
                    "t_start_sec": float(t_start) * 60.0,
                    "t_start_ms_approx": float(t_start) * 60000.0,
                }
            )


def dump_predictions_csv(
    out_fp: Path,
    refs: List[Any],
    y_true: Sequence[int],
    probs: Sequence[float],
    split: str,
    ref_key_fn: Optional[RefKeyFn] = None,
) -> None:
    """Dump per-ref predictions to CSV."""
    ref_key_fn = ref_key_fn or _default_ref_key()
    _ensure_parent(out_fp)

    fieldnames = [
        "split",
        "ref_key",
        "match_id",
        "patch",
        "t_start",
        "t_start_ts",
        "y",
        "p",
    ]

    with open(out_fp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r, y, p in zip(refs, y_true, probs):
            ts = getattr(r, "t_start_ts", -1)
            w.writerow(
                {
                    "split": split,
                    "ref_key": ref_key_fn(r),
                    "match_id": str(getattr(r, "match_id", "")),
                    "patch": str(getattr(r, "patch", "")),
                    "t_start": int(getattr(r, "t_start", 0)),
                    "t_start_ts": int(getattr(r, "t_start_ts", -1)),
                    "y": int(y),
                    "p": float(p),
                }
            )


def log_patch_block(title: str, patch_counts: Dict[str, int], log_fp: Path, max_items: int = 30) -> None:
    """Small helper to keep patch-count logging consistent."""

    items = list(patch_counts.items())
    total = sum(v for _, v in items)
    show = items[:max_items]
    s = ", ".join([f"{k}:{v}" for k, v in show])
    if len(items) > max_items:
        s += f" (+{len(items)-max_items} more)"
    s += f" | total={total}"
    write_log(f"[PATCH] {title}: {s}", log_fp)
