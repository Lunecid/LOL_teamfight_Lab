"""A3 temporal windows (CoG review R2): the inputs-changed assertion, row identity across settings, the training-rows-only
column filter and bootstrap pairing of scripts/run_window_sweep_v33.py, and the leak-relevant pieces of
scripts/run_sequence_baseline_v33.py.  Unit tests use synthetic arrays; one integration test builds two settings from a
dozen cached matches and is skipped when the match cache is absent."""
import importlib.util
import itertools
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


SW = _load("run_window_sweep_v33")


def _matrix(seed, n=40, d=15):
    return np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)


def _ref(i, match="m0", ts=None):
    from core.fight_types import FightRef
    return FightRef(match_id=match, patch="15.14", t_start=0, t_start_ts=int(ts if ts is not None else 200000 + 1000 * i),
                    label_end_ts=-1, first_kill_ts=215000 + 1000 * i, last_kill_ts=220000 + 1000 * i, det_cluster_blue=3,
                    det_cluster_red=2, det_present_blue=4, det_present_red=4, anchor_x=0.25, anchor_y=0.5)


# ------------------------------------------------------------------ settings

def test_default_settings_are_the_briefed_grid():
    s = SW.parse_settings(SW.DEFAULT_SETTINGS)
    assert s == [(15, 5000), (30, 5000), (60, 5000), (120, 5000), (30, 2500), (30, 10000)]
    assert [SW.n_bins(*x) for x in s] == [3, 6, 12, 24, 12, 3]
    assert SW.parse_settings("30, 30:5000, 60") == [(30, 5000), (60, 5000)]
    assert SW.parse_settings(SW.REFERENCE) == [(30, 5000)]
    with pytest.raises(ValueError):
        SW.parse_settings("30:7000")


# ------------------------------------------------------------------ inputs changed

def test_inputs_changed_assertion_fires_on_identical_matrices():
    names = [f"f{i}" for i in range(14)] + [SW.FRAME_AGE]
    tele = SW.telemetry_columns(names)
    assert len(tele) == 14 and names.index(SW.FRAME_AGE) not in tele.tolist()
    a = _matrix(0)
    with pytest.raises(SW.InputsUnchangedError, match="identical telemetry"):
        SW.assert_inputs_changed({"ctx30_bin5000": SW.matrix_sha1(a, tele), "ctx60_bin5000": SW.matrix_sha1(a.copy(), tele)})

    b = a.copy()
    b[:, -1] += 1.0                                   # only frame_age differs: not a telemetry change
    with pytest.raises(SW.InputsUnchangedError):
        SW.assert_inputs_changed({"x": SW.matrix_sha1(a, tele), "y": SW.matrix_sha1(b, tele)})

    c = a.copy()
    c[3, 2] = np.nextafter(c[3, 2], np.float32(np.inf))   # one telemetry cell, one ulp
    hashes = {"a": SW.matrix_sha1(a, tele), "b": SW.matrix_sha1(a.copy(), tele), "c": SW.matrix_sha1(c, tele)}
    with pytest.raises(SW.InputsUnchangedError, match="'a', 'b'"):
        SW.assert_inputs_changed(hashes)          # one identical pair among three settings is enough to fail
    assert SW.assert_inputs_changed({"a": hashes["a"], "c": hashes["c"]}) == []
    with pytest.raises(SW.InputsUnchangedError):
        SW.assert_inputs_changed({"a": hashes["a"]})   # a single setting proves nothing


def test_matrix_sha1_is_chunk_invariant_and_shape_sensitive():
    a = _matrix(1, n=1000, d=9)
    cols = np.array([0, 3, 8])
    assert SW.matrix_sha1(a, cols, chunk=7) == SW.matrix_sha1(a, cols, chunk=4096) == SW.matrix_sha1(np.ascontiguousarray(a[:, cols]))
    assert SW.matrix_sha1(a[:500]) != SW.matrix_sha1(a)
    flat = np.ascontiguousarray(a[:, :8])
    assert SW.matrix_sha1(flat) != SW.matrix_sha1(flat.reshape(500, 16))   # same bytes, different shape


# ------------------------------------------------------------------ row identity

def test_row_identity_uses_common_rows_and_rejects_label_disagreement():
    ref = [("m1", 1000, 0), ("m1", 5000, 1), ("m2", 2000, 2), ("m3", 700, 3)]
    other = [ref[2], ref[0], ref[3]]                               # other order, one engagement missing
    labels = {"ref": np.array([1, 0, 1, 0]), "other": np.array([1, 1, 0])}
    common, take, y, mismatch = SW.align_settings({"ref": ref, "other": other}, labels, "ref")
    assert common == sorted([ref[0], ref[2], ref[3]])
    for name, keys in (("ref", ref), ("other", other)):
        assert [keys[i] for i in take[name]] == common
    assert y.tolist() == [1, 1, 0] and mismatch == {"ref": 0, "other": 0}

    with pytest.raises(SW.RowIdentityError, match="labels differ"):
        SW.align_settings({"ref": ref, "other": other}, {"ref": labels["ref"], "other": np.array([0, 1, 0])}, "ref")
    with pytest.raises(SW.RowIdentityError, match="no engagement"):
        SW.align_settings({"ref": ref[:1], "other": ref[1:]}, {"ref": np.array([1]), "other": np.array([0, 1, 0])}, "ref")
    with pytest.raises(SW.RowIdentityError, match="duplicate"):
        SW.align_settings({"ref": ref + [ref[0]], "other": other}, {"ref": np.array([1, 0, 1, 0, 1]), "other": labels["other"]}, "ref")


def test_row_key_carries_index_position_so_equal_timestamps_stay_distinct(monkeypatch):
    import train.baseline as tb
    refs = [_ref(0), _ref(1), _ref(2), _ref(3, match="m1", ts=200000), _ref(4, match="m1", ts=200000)]
    rng = np.random.default_rng(3)
    seqs = {id(r): rng.standard_normal((6, 4)).astype(np.float32) for r in refs}
    names = [f"c{i}" for i in range(28)] + [SW.FRAME_AGE]
    original = tb.seq_to_tabular

    def fake_builder(refs_in, feature_set):
        used, rows = [], []
        for j, r in enumerate(refs_in):
            if j == 2:
                continue                                           # the builder may drop an engagement
            rows.append(np.concatenate([tb.seq_to_tabular(seqs[id(r)]), [60.0]]).astype(np.float32))
            used.append(r)
        return np.stack(rows), np.array([1, 0, 1, 0]), names, used

    monkeypatch.setattr(tb, "build_tabular_Xy", fake_builder)
    out = SW.build_setting(refs, keep_seq=True)
    assert tb.seq_to_tabular is original                           # the capture hook is removed again
    assert out["row_ref_ordinal"].tolist() == [0, 1, 3, 4]
    assert (out["L"], out["D_seq"], out["seq"].shape) == (6, 4, (4, 6, 4))
    for i, j in enumerate([0, 1, 3, 4]):
        np.testing.assert_array_equal(out["seq"][i], seqs[id(refs[j])])
        np.testing.assert_array_equal(tb.seq_to_tabular(out["seq"][i]), out["X"][i, :-1])
    keys = SW.row_keys(out)
    assert keys[2] == ("m1", 200000, 3) and keys[3] == ("m1", 200000, 4) and len(set(keys)) == 4


def _synthetic_pack():
    ts = np.array([0, 60000, 120000, 180000, 240000], dtype=np.int64)
    rng = np.random.default_rng(11)
    events = [{"type": "CHAMPION_KILL", "timestamp": t, "killerId": 2, "victimId": 7, "position": {"x": 100.0, "y": 200.0}}
              for t in (30000, 119999, 120000, 150000)]
    return {"minute_ts": ts, "node_minute": rng.standard_normal((5, 10, 4)).astype(np.float32),
            "global_minute": rng.standard_normal((5, 3)).astype(np.float32),
            "gold_team_minute": rng.standard_normal((5, 2)).astype(np.float32),
            "xy_raw_minute": rng.standard_normal((5, 10, 2)).astype(np.float32), "events": events,
            "meta": {"team_map": {}, "anchors": {"dragon": [[1.0, 2.0]]}}, "_event_index_built": True}


def test_perturb_pack_rewrites_only_one_side_of_the_cutoff():
    pack = _synthetic_pack()
    cutoff = 120000                                                    # a frame and an event sit exactly at the cutoff
    before = {k: np.array(pack[k], copy=True) for k in SW.FRAME_KEYS}
    fut = SW.perturb_pack(pack, cutoff, seed=3, side="future")
    for k in SW.FRAME_KEYS:
        np.testing.assert_array_equal(pack[k], before[k])             # the original pack is never mutated
        np.testing.assert_array_equal(fut[k][:2], before[k][:2])      # frames before the cutoff untouched
        assert not np.any(fut[k][2:] == before[k][2:])                # frames at/after the cutoff all rewritten
    early = [e for e in fut["events"] if int(e["timestamp"]) < cutoff]
    assert early == [e for e in pack["events"] if int(e["timestamp"]) < cutoff]
    at_cut = {e["type"] for e in fut["events"] if int(e["timestamp"]) == cutoff}
    assert {"CHAMPION_KILL", "ELITE_MONSTER_KILL", "BUILDING_KILL", "ITEM_PURCHASED", "WARD_PLACED"} <= at_cut
    assert fut["meta"]["anchors"] != pack["meta"]["anchors"] and pack["meta"]["anchors"] == {"dragon": [[1.0, 2.0]]}
    assert "_event_index_built" not in fut                            # no derived index of the source pack survives
    assert fut["events_ts"].tolist() == [int(e["timestamp"]) for e in fut["events"]]   # rebuilt for the rewritten events
    past = SW.perturb_pack(pack, cutoff, seed=3, side="past")
    for k in SW.FRAME_KEYS:
        assert not np.any(past[k][:2] == before[k][:2])
        np.testing.assert_array_equal(past[k][2:], before[k][2:])
    assert [e["timestamp"] for e in past["events"]] == [e["timestamp"] for e in pack["events"]]
    with pytest.raises(ValueError):
        SW.perturb_pack(pack, cutoff, seed=3, side="both")


def test_compare_indexes_is_order_free_and_field_exact():
    a = [_ref(i) for i in range(4)]
    assert SW.compare_indexes(a, list(reversed([_ref(i) for i in range(4)])))["identical"]
    b = [_ref(i) for i in range(4)]
    b[1].det_cluster_red = 5
    r = SW.compare_indexes(a, b)
    assert not r["identical"] and (r["only_reference"], r["only_setting"]) == (1, 1)
    assert SW.index_sha1(a) == SW.index_sha1(list(reversed(a))) != SW.index_sha1(b)


# ------------------------------------------------------------------ learners and bootstrap

def test_constant_column_filter_reads_training_rows_only():
    X = np.zeros((6, 3), dtype=np.float32)
    X[:, 0] = [1, 2, 3, 4, 5, 6]            # varies on training rows
    X[3:, 1] = [7, 8, 9]                    # varies only on rows outside training
    train = np.array([True, True, True, False, False, False])
    assert SW.nonconstant_columns(X).tolist() == [0, 1]
    assert SW.nonconstant_columns(X, train).tolist() == [0]
    assert SW.nonconstant_columns(X, np.flatnonzero(train), chunk=2).tolist() == [0]


def test_patch_fit_never_reads_test_rows():
    rng = np.random.default_rng(0)
    n = 600
    X = rng.standard_normal((n, 5)).astype(np.float32)
    y = (X[:, 0] + 0.5 * rng.standard_normal(n) > 0).astype(np.int8)
    tr, va, te = (np.arange(n) % 3 == k for k in range(3))
    first = SW.fit_patch_lgbm(X, y, tr, va, te, n_jobs=1)
    X2 = X.copy()
    X2[te] = rng.standard_normal((int(te.sum()), 5)).astype(np.float32) * 100.0   # rewrite every test row
    X2[te, 3] = 0.0
    second = SW.fit_patch_lgbm(X2, y, tr, va, te, n_jobs=1)
    assert first["val_auc"] == second["val_auc"] and first["best_iteration"] == second["best_iteration"]
    assert first["n_features_nonconstant"] == second["n_features_nonconstant"] == 5


def test_paired_bootstrap_resamples_matches_jointly():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(60).astype(str), 3)
    y = rng.integers(0, 2, len(groups))
    a = rng.random(len(groups))
    out = SW.match_bootstrap(y, {"a": a, "b": a.copy()}, groups, n_boot=50, seed=1, pairs=[("b", "a")])
    d = out["paired"]["b - a"]
    assert d["delta"] == 0.0 and d["ci_lo"] == 0.0 and d["ci_hi"] == 0.0 and out["n_matches"] == 60
    _, inverse = np.unique(groups, return_inverse=True)
    w = SW.match_weights(np.random.default_rng(5), inverse, 60)
    rows = np.repeat(np.arange(len(y)), w.astype(int))
    assert SW.weighted_auc(y, a, w) == pytest.approx(roc_auc_score(y[rows], a[rows]), abs=1e-12)


# ------------------------------------------------------------------ sequence baseline

def test_sequence_standardiser_uses_training_rows_only():
    SB = _load("run_sequence_baseline_v33")
    rng = np.random.default_rng(0)
    seq = rng.standard_normal((10, 4, 3)).astype(np.float32)
    fa = (rng.random(10) * 60).astype(np.float32)
    tr = np.array([0, 1, 2, 3])
    stats = SB.fit_standardiser(seq, fa, tr)
    seq2, fa2 = seq.copy(), fa.copy()
    seq2[4:] += 1000.0
    fa2[4:] = 1e6
    stats2 = SB.fit_standardiser(seq2, fa2, tr)
    np.testing.assert_array_equal(stats["mu"], stats2["mu"])
    np.testing.assert_array_equal(stats["sd"], stats2["sd"])
    Z, nonfinite = SB.standardise(seq, fa, stats, clip=10.0)
    assert Z.shape == (10, 4, 4) and nonfinite == 0
    np.testing.assert_allclose(Z[tr, :, :3].reshape(-1, 3).mean(axis=0), 0.0, atol=1e-5)
    assert np.all(Z[:, :, 3] == Z[:, :1, 3])                        # frame age is constant across steps
    Zc, _ = SB.standardise(seq2, fa2, stats, clip=10.0)
    assert np.abs(Zc).max() <= 10.0


def _fake_sweep(tmp_path, n=12, L=6, D=5):
    """A minimal sweep directory: common rows, a sequence cache written in a shuffled order, and its summary."""
    SB = _load("run_sequence_baseline_v33")
    rng = np.random.default_rng(4)
    mids = np.array([f"m{i // 3}" for i in range(n)])
    ts = np.arange(n, dtype=np.int64) * 1000 + 200000
    ordinal = np.arange(n, dtype=np.int64) * 2
    y = rng.integers(0, 2, n).astype(np.int8)
    patch = np.array(["15.14", "15.15", "15.16"])[np.arange(n) % 3]
    rows = {"match_id": mids, "engage_ts": ts, "ref_ordinal": ordinal, "y": y, "patch": patch}
    seq = rng.standard_normal((n, L, D)).astype(np.float32)
    fa = rng.random(n).astype(np.float32)
    perm = rng.permutation(n)
    preset_values = {"TIME_NORM_ABSOLUTE": True, "ANCHORS_CAUSAL": True, "TAB_FRAME_AGE_FEATURE": True}
    summary = {"fight_index": {"sha1": "idx"}, "match_list_sha1": "ml", "source_sha1": "src", "script_sha1": "sw",
               "git": {"commit": "c0"}, "feature_set": "full", "preset_values": preset_values}
    name = SW.setting_name(30, 5000)
    sig = {"kind": "sequence", "setting": name, "preset": SW.PRESET, "label": SW.LABEL_KEY, "tie_policy": "drop",
           "seq_key": "x_seq", "ctx_sec": 30, "bin_ms": 5000, "index_sha1": "idx", "match_list_sha1": "ml",
           "source_sha1": "src", "script_sha1": "sw", "feature_set": "full", "preset_values": preset_values}
    path = tmp_path / f"seq_{name}.npz"

    def write(labels=y):
        SW.atomic_savez(path, seq=seq[perm], frame_age=fa[perm], y=labels[perm], match_id=mids[perm], engage_ts=ts[perm],
                        ref_ordinal=ordinal[perm], patch=patch[perm], seq_names=np.array([f"c{j}" for j in range(D)]))
        SW.write_json(SW.sidecar(path), {"signature": SW.canonical(sig), "L": L, "D_seq": D, "seq_key": "x_seq"})

    write()
    return SB, {"summary": summary, "rows": rows}, name, seq, fa, write, sig, path


def test_sequence_loader_realigns_cached_rows_to_the_common_rows(tmp_path):
    SB, sweep, name, seq, fa, write, sig, path = _fake_sweep(tmp_path)
    got = SB.load_sequences(tmp_path, name, sweep, 30, 5000)
    np.testing.assert_array_equal(got["seq"], seq)                     # common-row order restored from a shuffled cache
    np.testing.assert_array_equal(got["frame_age"], fa)
    assert got["subset_or_reordered"] and (got["L"], got["D"]) == (6, 5)
    with pytest.raises(SB.SweepMismatch, match="ctx_sec"):
        SB.load_sequences(tmp_path, name, sweep, 60, 5000)            # a cache for another window is refused
    write(labels=1 - sweep["rows"]["y"])
    with pytest.raises(SB.SweepMismatch, match="labels differ"):
        SB.load_sequences(tmp_path, name, sweep, 30, 5000)
    write()
    SW.write_json(SW.sidecar(path), {"signature": SW.canonical({**sig, "preset_values": {**sig["preset_values"],
                                                                                         "ANCHORS_CAUSAL": False}})})
    with pytest.raises(SB.SweepMismatch):
        SB.load_sequences(tmp_path, name, sweep, 30, 5000)            # a leak-era feature path is refused
    write()
    rows = dict(sweep["rows"])
    rows["engage_ts"] = rows["engage_ts"].copy()
    rows["engage_ts"][0] += 1
    with pytest.raises(SB.SweepMismatch, match="absent"):
        SB.load_sequences(tmp_path, name, {**sweep, "rows": rows}, 30, 5000)


def test_sequence_models_forward_and_bigru_final_state_readout():
    torch = pytest.importorskip("torch")
    SB = _load("run_sequence_baseline_v33")
    torch.manual_seed(0)
    x = torch.randn(5, 6, 8)
    hp = {"hidden": 16, "layers": 2, "dropout": 0.0, "readout": "final_states", "head_hidden": 8, "head_dropout": 0.0}
    gru = SB.SequenceClassifier("bigru", 8, 6, hp).eval()
    assert gru(x).shape == (5,)
    out, h_n = gru.encoder.rnn(x)
    torch.testing.assert_close(gru.encode(x), torch.cat([out[:, -1, :16], out[:, 0, 16:]], dim=-1))
    x2 = x.clone()
    x2[:, 0] += 1.0                                                  # the first bin reaches the backward half of the readout
    assert not torch.allclose(gru.encode(x2)[:, 16:], gru.encode(x)[:, 16:])
    trf = SB.SequenceClassifier("transformer", 8, 6, {"d_model": 16, "nhead": 4, "layers": 1, "dropout": 0.0,
                                                      "head_hidden": 8, "head_dropout": 0.0}).eval()
    assert trf(x).shape == (5,)
    ck = SB.SequenceClassifier("transformer", 8, 6, {"d_model": 16, "nhead": 4, "layers": 1, "dropout": 0.0, "head_hidden": 8,
                                                     "head_dropout": 0.0, "grad_checkpoint": True})
    ck.load_state_dict(trf.state_dict())
    ck.train()
    loss = ck(x).sum()
    loss.backward()
    assert all(p.grad is not None for p in ck.head.parameters())


def test_sequence_fit_resumes_from_checkpoint_to_the_same_predictions(tmp_path):
    torch = pytest.importorskip("torch")
    SB = _load("run_sequence_baseline_v33")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((120, 5, 6)).astype(np.float32)
    y = (X[:, -1, 0] + 0.3 * rng.standard_normal(120) > 0).astype(np.int8)
    tr, va, te = np.arange(0, 60), np.arange(60, 90), np.arange(90, 120)
    hp = {"hidden": 8, "layers": 1, "dropout": 0.2, "readout": "final_states", "head_hidden": 8, "head_dropout": 0.2,
          "batch_size": 16, "eval_batch_size": 64, "lr": 1e-2, "weight_decay": 1e-2, "max_epochs": 3, "patience": 10,
          "min_delta": 0.0, "grad_clip": 1.0}
    dev = torch.device("cpu")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        kw = dict(seed=7, device=dev, amp_dtype=None)
        full = SB.fit_sequence_model("bigru", X, y, tr, va, te, hp, ckpt_path=None, ckpt_signature={"run": 1}, **kw)
        ckpt = tmp_path / "ckpt.pt"
        SB.fit_sequence_model("bigru", X, y, tr, va, te, {**hp, "max_epochs": 1}, ckpt_path=ckpt, ckpt_signature={"run": 1}, **kw)
        assert ckpt.exists()                                           # an interruption after epoch 1
        resumed = SB.fit_sequence_model("bigru", X, y, tr, va, te, hp, ckpt_path=ckpt, ckpt_signature={"run": 1}, **kw)
        assert resumed["fit"]["resumed_from_epoch"] == 1 and resumed["fit"]["epochs_run"] == 3
        assert [h["val_auc"] for h in resumed["fit"]["history"]] == [h["val_auc"] for h in full["fit"]["history"]]
        np.testing.assert_array_equal(resumed["pred_test"], full["pred_test"])
        foreign = SB.fit_sequence_model("bigru", X, y, tr, va, te, hp, ckpt_path=ckpt, ckpt_signature={"run": 2}, **kw)
        assert foreign["fit"]["resumed_from_epoch"] is None            # another run's checkpoint is ignored
    finally:
        torch.set_num_threads(threads)


# ------------------------------------------------------------------ integration on cached matches

def _cache_dir():
    try:
        from core.config import CACHE_DIR
    except Exception:
        return None
    try:
        return CACHE_DIR if CACHE_DIR.exists() and next(CACHE_DIR.glob("*.meta.json"), None) is not None else None
    except OSError:
        return None


@pytest.mark.skipif(_cache_dir() is None, reason="match cache not available")
def test_cached_matches_keep_engagements_and_change_inputs_across_windows(monkeypatch):
    from core.config import CACHE_DIR, cfg
    saved = dict(vars(cfg))
    monkeypatch.setenv("LOL_CFG_PRESET", SW.PRESET)
    monkeypatch.setenv("LOL_CFG_OVERRIDES", "")
    try:
        mids = sorted(p.name[: -len(".meta.json")] for p in itertools.islice(CACHE_DIR.glob("*.meta.json"), 12))
        refs = SW.build_index(mids, 30, 5000, 1)
        assert len(refs) >= 3, "too few engagements in the first cached matches"
        for ctx, bin_ms in ((60, 5000), (30, 2500)):
            assert SW.compare_indexes(refs, SW.build_index(mids, ctx, bin_ms, 1))["identical"]
        built = {}
        for ctx, bin_ms in ((30, 5000), (60, 5000), (30, 2500)):
            SW.configure(ctx, bin_ms)
            built[(ctx, bin_ms)] = SW.build_setting(refs, keep_seq=True)
        a, b, c = built[(30, 5000)], built[(60, 5000)], built[(30, 2500)]
        assert (a["L"], b["L"], c["L"]) == (6, 12, 12)
        assert a["names"] == b["names"] == c["names"] and a["names"][-1] == SW.FRAME_AGE
        keys = {n: SW.row_keys(d) for n, d in (("a", a), ("b", b), ("c", c))}
        assert keys["a"] == keys["b"] == keys["c"]
        common, take, y, mismatch = SW.align_settings(keys, {"a": a["y"], "b": b["y"], "c": c["y"]}, "a")
        assert len(common) == len(keys["a"]) and not any(mismatch.values())
        np.testing.assert_array_equal(a["X"][:, -1], b["X"][:, -1])  # frame age does not depend on the window
        tele = SW.telemetry_columns(a["names"])
        hashes = {n: SW.matrix_sha1(d["X"], tele) for n, d in (("a", a), ("b", b), ("c", c))}
        assert SW.assert_inputs_changed(hashes) == []
        with pytest.raises(SW.InputsUnchangedError):
            SW.assert_inputs_changed({"a": hashes["a"], "a_rebuilt": SW.matrix_sha1(a["X"].copy(), tele)})
        from gameplay.features import seq_to_tabular
        for i in range(len(y)):
            np.testing.assert_array_equal(seq_to_tabular(a["seq"][i]), a["X"][i, :-1])
        SW.configure(60, 5000)
        probe = SW.leak_probe(refs, 3, 7)
        assert probe["passed"], probe
    finally:
        for k in set(vars(cfg)) - set(saved):
            delattr(cfg, k)
        for k, v in saved.items():
            setattr(cfg, k, v)


@pytest.mark.skipif(_cache_dir() is None, reason="match cache not available")
@pytest.mark.parametrize("leak", ["events_at_or_after_cutoff", "frames_at_or_after_cutoff"])
def test_leak_probe_fails_when_a_feature_reads_past_the_cutoff(monkeypatch, leak):
    """Negative control: the probe must catch a feature that reads data stamped at or after the cutoff."""
    import train.baseline as tb
    from core.config import CACHE_DIR, cfg
    saved = dict(vars(cfg))
    monkeypatch.setenv("LOL_CFG_PRESET", SW.PRESET)
    monkeypatch.setenv("LOL_CFG_OVERRIDES", "")
    try:
        mids = sorted(p.name[: -len(".meta.json")] for p in itertools.islice(CACHE_DIR.glob("*.meta.json"), 12))
        refs = SW.build_index(mids, 30, 5000, 1)
        assert len(refs) >= 3
        SW.configure(30, 5000)
        assert SW.leak_probe(refs, 2, 7)["passed"]                     # the canonical path passes first
        seen = {}
        original_ms, original_feats = tb.build_ms_sequence, tb.build_sequence_features

        def remember_pack(pack, *a, **kw):
            seen["pack"], seen["cut"] = pack, int(kw["engage_ts"])
            return original_ms(pack, *a, **kw)

        def leaky_features(raw, *a, **kw):
            feats = dict(original_feats(raw, *a, **kw))
            pack, cut = seen["pack"], seen["cut"]
            if leak == "events_at_or_after_cutoff":
                value = float(sum(1 for e in pack["events"] if int(e.get("timestamp", -1) or -1) >= cut))
            else:
                ts = np.asarray(pack["minute_ts"], dtype=np.int64)
                value = float(np.nansum(np.asarray(pack["node_minute"], dtype=np.float64)[ts >= cut]))
            x = np.array(feats["x_seq"], dtype=np.float32, copy=True)
            x[-1, 0] = np.float32(value)
            feats["x_seq"] = x
            return feats

        monkeypatch.setattr(tb, "build_ms_sequence", remember_pack)
        monkeypatch.setattr(tb, "build_sequence_features", leaky_features)
        probe = SW.leak_probe(refs, 3, 7)
        assert probe["probed"] == 3 and not probe["passed"], probe
        assert len(probe["failures"]) == 3 and probe["sequence_identical_after_future_perturbation"] == 0
    finally:
        for k in set(vars(cfg)) - set(saved):
            delattr(cfg, k)
        for k, v in saved.items():
            setattr(cfg, k, v)
