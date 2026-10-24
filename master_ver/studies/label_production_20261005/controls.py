"""Non-event control windows for Q_DEV h90 keys: phase 1, matching (PROTOCOL_KO.md §2, 32-34행).

For each Q_DEV engagement j (pre tau_j, post e_j, length D_j = e_j - tau_j), the
candidate starts are the same match's supported nominal V queries s.  A candidate
is kept when:
  - control_end = s + D_j lies inside the observed frames (<= last frame) and before GAME_END;
  - the closed interval [s, control_end] holds no CHAMPION_KILL and does not overlap
    any component's closed [pre, last kill] (every component, eligible or not);
  - its pre-frame-age bin ([0,5] / (5,10] s) and game-time bin (<15 / 15-25 / >=25 min)
    equal the engagement's.
The kept candidate with the smallest |s - tau_j| is chosen (ties: earlier s, then
nominal order).  Starts may be reused; reuse and unmatched counts are reported.
Matching never looks at W, SVI or q.  Phase 2 (states at control ends that the
bank does not hold) is a separate sidecar version; this phase counts them.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studies.label_production_20261005 import assemble, engagements as eng_mod  # noqa: E402

VERSION = "label_controls_matching_v1_20261005"
PROJECTION = ROOT / "outputs/v_input_projection_20261002/full_raw_20261002_1"
CHAMPION_KILL = 1  # FIELD_MANIFEST event_type_names index
Q_DEV = 3


def _game_bin(ms: np.ndarray) -> np.ndarray:
    return np.digitize(np.asarray(ms) / 60000.0, (15, 25))


def _age_bin(age_s: np.ndarray) -> np.ndarray:
    age = np.asarray(age_s, dtype=np.float64)
    return np.where((age >= 0) & (age <= 5), 0, np.where((age > 5) & (age <= 10), 1, -1))


def match_controls(labels: Path) -> dict[str, Any]:
    labels = Path(labels)
    eng = dict(np.load(labels / "ENGAGEMENTS.npz", allow_pickle=False))
    q_rows = np.flatnonzero((eng["role"] == Q_DEV) & eng["valid_h90"])
    by_match: dict[int, list[int]] = {}
    for row in q_rows:
        by_match.setdefault(int(eng["match_index"][row]), []).append(int(row))
    matches = np.load(PROJECTION / "MATCH_TABLE.npy", mmap_mode="r", allow_pickle=False)
    nominal = np.load(PROJECTION / "V_ROWS.npy", mmap_mode="r", allow_pickle=False)
    match_queries = np.load(PROJECTION / "MATCH_QUERIES.npy", mmap_mode="r", allow_pickle=False)
    first_query, n_queries = match_queries["first_query"], match_queries["n_queries"]
    queries = np.load(PROJECTION / "QUERY_TABLE.npy", mmap_mode="r", allow_pickle=False)
    frame_age = np.load(PROJECTION / "FRAME_AGE_SECONDS.npy", mmap_mode="r", allow_pickle=False)
    key_of = {bytes(k).decode("ascii"): i for i, k in enumerate(matches["key"])}
    n = len(q_rows)
    out = {name: np.full(len(eng["role"]), -1, dtype=np.int64) for name in ("control_start_query", "control_start_ms", "control_end_ms")}
    reasons = {"no_supported_start": 0, "no_candidate_after_rules": 0}
    end_in_bank = 0
    for directory in eng_mod.shard_dirs():
        with (directory / "MATCH_LEDGER.jsonl").open(encoding="utf-8") as stream:
            ledger = [json.loads(line) for line in stream if line.strip()]
        local_of = {key_of[str(entry["canonical_match_key"])]: i for i, entry in enumerate(ledger)}
        wanted = [m for m in local_of if m in by_match]
        if not wanted:
            continue
        with np.load(directory / "BANK.npz", allow_pickle=False) as bank:
            arrays = {name: bank[name] for name in ("event_timestamp", "event_type_code", "component_values", "minute_ts", "end_ms")}
            offsets = {name: bank[f"offset_{name}"] for name in arrays}
        col = {name: i for i, name in enumerate(eng_mod.COMPONENT_COLUMNS)}
        for match_index in wanted:
            i = local_of[match_index]
            sl = {name: slice(int(offsets[name][i]), int(offsets[name][i + 1])) for name in arrays}
            times = arrays["event_timestamp"][sl["event_timestamp"]].astype(np.int64)
            kills = np.sort(times[arrays["event_type_code"][sl["event_type_code"]] == CHAMPION_KILL])
            comps = arrays["component_values"][sl["component_values"]]
            spans = np.stack([comps[:, col["f"]].astype(np.int64) - 10_000, comps[:, col["L"]].astype(np.int64)], axis=1)
            frames = arrays["minute_ts"][sl["minute_ts"]].astype(np.int64)
            end_ms = arrays["end_ms"][sl["end_ms"]]
            game_end = int(end_ms[0]) if len(end_ms) else -1
            match = matches[match_index]
            rows = nominal[int(match["first_nominal_row"]):int(match["first_nominal_row"]) + int(match["n_nominal_rows"])]
            rows = rows[np.asarray(rows["supported"], dtype=bool)]
            if not len(rows):
                reasons["no_supported_start"] += len(by_match[match_index])
                continue
            start_q = int(first_query[match_index]) + rows["query_ref"].astype(np.int64)
            start_ms = rows["actual_ms"].astype(np.int64)
            start_age_bin = _age_bin(frame_age[start_q])
            start_game_bin = _game_bin(start_ms)
            fq = int(first_query[match_index])
            match_query_ms = set(int(v) for v in queries["query_ms"][fq:fq + int(n_queries[match_index])])
            for row in by_match[match_index]:
                tau, post = int(eng["pre_ms"][row]), int(eng["post_ms_h90"][row])
                length = post - tau
                ends = start_ms + length
                ok = (ends <= (frames[-1] if len(frames) else -1)) & ((game_end < 0) | (ends < game_end))
                ok &= start_age_bin == int(_age_bin(eng["pre_frame_age_ms"][row] / 1000.0))
                ok &= start_game_bin == int(_game_bin(tau))
                if len(kills):
                    first_kill_at_or_after = np.searchsorted(kills, start_ms, side="left")
                    ok &= ~((first_kill_at_or_after < len(kills)) & (kills[np.minimum(first_kill_at_or_after, len(kills) - 1)] <= ends))
                if len(spans):
                    overlap = (start_ms[:, None] <= spans[None, :, 1]) & (spans[None, :, 0] <= ends[:, None])
                    ok &= ~overlap.any(axis=1)
                candidates = np.flatnonzero(ok)
                if not len(candidates):
                    reasons["no_candidate_after_rules"] += 1
                    continue
                distance = np.abs(start_ms[candidates] - tau)
                best = candidates[np.lexsort((candidates, start_ms[candidates], distance))[0]]
                out["control_start_query"][row] = int(start_q[best])
                out["control_start_ms"][row] = int(start_ms[best])
                out["control_end_ms"][row] = int(ends[best])
                end_in_bank += int(int(ends[best]) in match_query_ms)
    matched = out["control_start_query"][q_rows] >= 0
    starts, reuse = np.unique(out["control_start_query"][q_rows][matched], return_counts=True)
    ends = {(int(eng["match_index"][r]), int(out["control_end_ms"][r])) for r in q_rows[matched]}
    record = {"version": VERSION, "utc": datetime.now(timezone.utc).isoformat(), "q_dev_keys": int(n), "matched": int(matched.sum()),
              "unmatched": int((~matched).sum()), "unmatched_reasons": reasons, "matched_rate": float(matched.mean()) if n else None,
              "distinct_control_starts": int(len(starts)), "max_start_reuse": int(reuse.max()) if len(reuse) else 0,
              "distinct_control_ends": int(len(ends)), "control_ends_already_in_bank": int(end_in_bank),
              "control_ends_needing_new_state": int(len(ends)) - int(end_in_bank),
              "rules": "PROTOCOL_KO.md §2 L32: no kill and no component [pre,L] overlap in [s,end]; same age/game-time bins; min |s-tau|, earlier, stable"}
    table = {"canonical_match_key": eng["canonical_match_key"][q_rows], "component_key": eng["component_key"][q_rows],
             **{name: values[q_rows] for name, values in out.items()}}
    record["table_sha256"] = assemble.write_npz(labels / "CONTROLS_MATCHING_h90.npz", table)
    (labels / "CONTROLS_MATCHING_h90.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(match_controls(Path(args.labels)), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
