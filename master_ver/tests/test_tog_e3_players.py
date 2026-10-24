"""Tests for the E3 player matrix (PRESPEC_E3_KO.md sections 2, 3, 6 and 9; PLAN_E1_E5_KO.md Task 7).

Every test builds a fake external set under tmp_path: a row table with the three columns the matrix reads
(match_index, canonical_match_key, patch; two rows per match), a patch index (match_id -> patch) and detail JSON files
whose participants carry made-up identifiers.  The real row table, patch index and detail files are never touched.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from studies.external_validation_20261009.e2a_common import canonical_key
from studies.tog_experiments_20261012 import e3_players as P
from studies.tog_experiments_20261012.common import sha256_file

DATASET = "kr_15_18_22"


def roster(tag):
    """Ten made-up player identifiers, participantId 1-10."""
    return [f"fake-{tag}-{i:02d}" for i in range(10)]


def key(match_id):
    return canonical_key(match_id).encode("ascii")


def spec(match_index, patch, puuids, **extra):
    return {"match_index": match_index, "patch": patch, "puuids": puuids, **extra}


def detail_doc(match_id, puuids, *, order=None, meta_id=None):
    """A detail JSON shaped like the Riot match payload: participants with participantId and, unless None, puuid."""
    parts = [{"participantId": i + 1, "teamId": 100 if i < 5 else 200, **({} if u is None else {"puuid": u})}
             for i, u in enumerate(puuids)]
    if order is not None:
        parts = [parts[i] for i in order]
    return {"metadata": {"matchId": meta_id or match_id, "participants": [u or "" for u in puuids]},
            "info": {"queueId": 420, "participants": parts}}


def make_set(root, matches, rows_per_match=2):
    """Write a fake external set.  ``matches`` maps match_id -> spec(); spec keys beyond the three columns:
    file (None = no detail file, bytes = written as is), in_index (False = left out of the patch index),
    index_patch (the patch index's patch, default the row table's), order (participant list order), meta_id."""
    root = Path(root)
    detail = root / "detail"
    detail.mkdir(parents=True)
    index, match_index, keys, patches = {}, [], [], []
    for match_id, s in matches.items():
        if s.get("in_index", True):
            index[match_id] = s.get("index_patch", s["patch"])
        for _ in range(rows_per_match):
            match_index.append(s["match_index"])
            keys.append(key(match_id))
            patches.append(s["patch"].encode("ascii"))
        content = s.get("file", "doc")
        if content == "doc":
            doc = detail_doc(match_id, s["puuids"], order=s.get("order"), meta_id=s.get("meta_id"))
            (detail / f"{match_id}.json").write_text(json.dumps(doc), encoding="utf-8")
        elif content is not None:
            (detail / f"{match_id}.json").write_bytes(content)
    rows = root / "Q_ROWS_h90.npz"
    np.savez(rows, match_index=np.asarray(match_index, dtype=np.int64), canonical_match_key=np.asarray(keys, dtype="S64"),
             patch=np.asarray(patches, dtype="S8"))
    patch_index = root / "patch_index.json"
    patch_index.write_text(json.dumps(index), encoding="utf-8")
    return {"rows_path": rows, "patch_index_path": patch_index, "detail_dir": detail}


# ---------------------------------------------------------------- the plan's two tests

def test_missing_puuid_is_minus_one(tmp_path):
    a = roster("a")
    a[3] = None                       # participantId 4 has no puuid field
    b = roster("b")
    b[0] = ""                         # participantId 1 has an empty puuid
    paths = make_set(tmp_path, {"KR_1000000001": spec(0, "15.18", a), "KR_1000000002": spec(1, "15.18", b)})
    keys, idx = P.player_matrix(DATASET, **paths)
    assert idx.dtype == np.int64 and idx.shape == (2, 10)
    assert idx[0, 3] == -1 and idx[1, 0] == -1
    assert (np.delete(idx[0], 3) >= 0).all() and (idx[1, 1:] >= 0).all()
    assert len(np.unique(idx[idx >= 0])) == 18


def test_ten_distinct_per_match(tmp_path):
    paths = make_set(tmp_path, {"KR_1000000001": spec(0, "15.18", roster("a")),
                                "KR_1000000002": spec(1, "15.19", roster("b"))})
    keys, idx = P.player_matrix(DATASET, **paths)
    assert keys.tolist() == [key("KR_1000000001"), key("KR_1000000002")]
    for row in idx:
        assert (row >= 0).all() and len(set(row.tolist())) == 10
    assert set(idx[0].tolist()).isdisjoint(idx[1].tolist())


# ---------------------------------------------------------------- numbering, order and failures

def test_same_player_same_index_across_matches(tmp_path):
    a, b = roster("a"), roster("b")
    b[6] = a[2]                       # the player of participantId 3 in match A is participantId 7 in match B
    paths = make_set(tmp_path, {"KR_1000000001": spec(0, "15.18", a), "KR_1000000002": spec(1, "15.18", b)})
    _, idx = P.player_matrix(DATASET, **paths)
    assert idx[0, 2] == idx[1, 6]
    assert len(set(idx[0].tolist()) & set(idx[1].tolist())) == 1
    assert len(np.unique(idx)) == 19
    assert idx[0].tolist() == list(range(10))           # dense numbers in first-appearance order


def test_order_is_ascending_match_index_and_participant_id(tmp_path):
    early, late = roster("early"), roster("late")
    late[0] = early[4]                # participantId 1 of the later match is participantId 5 of the earlier one
    paths = make_set(tmp_path, {
        "KR_1000000009": spec(7, "15.20", late, order=list(range(9, -1, -1))),   # first in the row table, stored reversed
        "KR_1000000001": spec(2, "15.18", early)})
    built = P.build(DATASET, **paths)
    assert built.match_index.tolist() == [2, 7]
    assert built.match_keys.tolist() == [key("KR_1000000001"), key("KR_1000000009")]
    assert built.player_idx[1, 0] == built.player_idx[0, 4]
    # numbers follow matrix order, then participantId 1-10, never the participant list order of the file
    assert built.player_idx.tolist() == [list(range(10)), [4] + list(range(10, 19))]


def test_unmapped_matches_are_counted_not_dropped(tmp_path):
    dup = roster("h")
    dup[8] = dup[1]
    paths = make_set(tmp_path, {
        "KR_1000000001": spec(0, "15.18", roster("a")),
        "KR_1000000002": spec(1, "15.18", roster("b"), file=None),                 # no detail file
        "KR_1000000003": spec(2, "15.18", roster("c"), file=b"{not json"),         # unreadable
        "KR_1000000004": spec(3, "15.18", roster("d"), in_index=False),            # key not in the patch index
        "KR_1000000005": spec(4, "15.19", roster("e"), index_patch="15.18"),       # patch index disagrees
        "KR_1000000006": spec(5, "15.18", roster("f"), meta_id="KR_1999999999"),   # file names another match
        "KR_1000000007": spec(6, "15.18", roster("g")[:9]),                        # nine participants
        "KR_1000000008": spec(7, "15.18", dup)})                                   # one puuid twice
    built = P.build(DATASET, **paths)
    assert built.player_idx.shape == (8, 10) and built.match_index.tolist() == list(range(8))
    assert (built.player_idx[0] >= 0).all() and (built.player_idx[1:] == -1).all()
    assert P.failure_counts(built.map_status) == {
        "key_not_in_patch_index": 1, "patch_mismatch": 1, "detail_file_missing": 1, "detail_unreadable": 1,
        "match_id_mismatch": 1, "participants_malformed": 1, "duplicate_puuid": 1}
    assert built.map_status[0] == 0 and (built.map_status[1:] != 0).all()
    assert np.unique(built.player_idx).tolist() == [-1] + list(range(10))   # a failed match numbers nobody


def test_unknown_dataset_is_refused_and_sources_follow_prespec():
    with pytest.raises(ValueError):
        P.player_matrix("kr_16_13")
    src = P.SOURCES[DATASET]
    assert src["rows_path"] == Path(r"D:\LOL_External_20261009\q_inputs\kr_15_18_22\Q_ROWS_h90.npz")
    assert src["patch_index_path"] == Path(r"D:\LOL_External_20261009\match_cache_kr_15_18_22_patch_index.json")
    assert src["detail_dir"] == Path(r"C:\Users\todtj\PycharmProjects\Lol_project\data\raw\matches\kr\detail")
    assert P.TOG_E3 == Path(r"D:\LOL_Training_20261004\tog_e3_1")


# ---------------------------------------------------------------- the run: written once, no identifiers, BLOCKED rule

def test_run_writes_once_without_identifiers(tmp_path):
    a, b = roster("a"), roster("b")
    a[5] = None
    b[6] = a[2]
    paths = make_set(tmp_path / "in", {"KR_1000000002": spec(5, "15.19", b), "KR_1000000001": spec(3, "15.18", a)})
    out = tmp_path / "out"
    assert P.run(out, DATASET, **paths) == 0
    npz, meta = out / "PLAYERS_kr_15_18_22.npz", out / "PLAYERS_kr_15_18_22.json"
    with np.load(npz, allow_pickle=False) as data:
        assert sorted(data.files) == ["canonical_match_key", "map_status", "match_index", "player_idx",
                                      "players_by_match_count"]
        assert data["match_index"].tolist() == [3, 5]
        assert data["canonical_match_key"].tolist() == [key("KR_1000000001"), key("KR_1000000002")]
        assert data["player_idx"].dtype == np.int64 and data["player_idx"][0, 5] == -1
        assert data["players_by_match_count"].tolist() == [0, 17, 1]    # 18 players: 17 in one match, one in two
    record = json.loads(meta.read_text(encoding="utf-8"))
    assert record["rows"] == 4 and record["matches"] == 2
    assert record["mapping"]["failed"] == 0 and record["mapping"]["failed_share"] == 0.0
    assert record["mapping"]["blocked"] is False
    assert record["players"]["distinct"] == 18 and record["players"]["slots_without_puuid"] == 1
    assert record["players"]["matches_by_identified_players"] == {"9": 1, "10": 1}
    assert record["matches_per_player"]["histogram"] == {"1": 17, "2": 1}
    assert record["npz_sha256"] == sha256_file(npz)
    assert set(record["run"]["input_sha256"]) == {Path(paths["rows_path"]).as_posix(),
                                                   Path(paths["patch_index_path"]).as_posix()}
    secrets = [u.encode("ascii") for u in a + b if u]
    for path in out.iterdir():
        blob = path.read_bytes()
        assert not any(s in blob for s in secrets), path.name
    with pytest.raises(FileExistsError):
        P.run(out, DATASET, **paths)


def test_more_than_one_percent_failed_is_blocked(tmp_path):
    paths = make_set(tmp_path / "in", {"KR_1000000001": spec(0, "15.18", roster("a")),
                                       "KR_1000000002": spec(1, "15.18", roster("b"), file=None)})
    out = tmp_path / "out"
    assert P.run(out, DATASET, **paths) == 2
    assert not (out / "PLAYERS_kr_15_18_22.npz").exists() and not (out / "PLAYERS_kr_15_18_22.json").exists()
    blocked = json.loads((out / "E3_BLOCKED_players_1.json").read_text(encoding="utf-8"))
    assert blocked["blocked"] is True
    assert blocked["mapping"]["failed"] == 1 and blocked["mapping"]["failed_share"] == 0.5
    assert blocked["mapping"]["by_reason"]["detail_file_missing"] == 1
    assert b"fake-" not in (out / "E3_BLOCKED_players_1.json").read_bytes()


def test_one_percent_failed_is_not_blocked(tmp_path):
    matches = {f"KR_2{n:09d}": spec(n, "15.18", roster(f"m{n}")) for n in range(100)}
    matches["KR_2000000050"]["file"] = None            # 1 of 100 matches = 1 %, not more than 1 %
    paths = make_set(tmp_path / "in", matches, rows_per_match=1)
    out = tmp_path / "out"
    assert P.run(out, DATASET, **paths) == 0
    with np.load(out / "PLAYERS_kr_15_18_22.npz", allow_pickle=False) as data:
        assert (data["player_idx"][50] == -1).all() and data["map_status"][50] != 0
        assert (np.delete(data["player_idx"], 50, axis=0) >= 0).all()
    record = json.loads((out / "PLAYERS_kr_15_18_22.json").read_text(encoding="utf-8"))
    assert record["mapping"]["failed"] == 1 and record["mapping"]["failed_share"] == 0.01
    assert record["mapping"]["blocked"] is False
    assert record["players"]["matches_by_identified_players"] == {"0": 1, "10": 99}
