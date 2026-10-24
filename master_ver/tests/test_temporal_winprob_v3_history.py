"""P2 contract tests: causal histories, masks, replay equality, guards, save/load and frozen legacy hashes."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
WT = ROOT / 'worktrees' / 'engagement-state-value'
sys.path.insert(0, str(WT))
sys.dont_write_bytecode = True

from gameplay.state_value import SNAPSHOT_FIELDS, State  # noqa: E402
from gameplay.state_value_v2 import STATE_VERSION, StateBuilder, StateV2  # noqa: E402
import train.temporal_history_winprob_v3 as T  # noqa: E402

P1 = ROOT / 'outputs' / 'independent_v2_participant_order'
NODE = list(SNAPSHOT_FIELDS) + ['champion_id']
MATCH = 'TEST_0001'


def make_pack(match=MATCH, n_frames=16, seed=0):
    rng = np.random.default_rng(seed)
    ts = np.asarray([0] + [60000 * k + int(rng.integers(1, 90)) for k in range(1, n_frames)], dtype=np.int64)
    node = np.zeros((n_frames, 10, len(NODE)), dtype=np.float32)
    for k in range(n_frames):
        for pid in range(1, 11):
            node[k, pid - 1, :len(SNAPSHOT_FIELDS)] = rng.random(len(SNAPSHOT_FIELDS)).astype(np.float32) + k * .1
            node[k, pid - 1, -1] = 100 + 7 * pid
    ev = [dict(type='CHAMPION_KILL', timestamp=int(t), killerId=int(k), victimId=int(v))
          for t, k, v in [(95000, 1, 6), (95000, 2, 7), (190123, 7, 3), (310000, 4, 9), (430500, 8, 2), (610000, 5, 10)]]
    ev += [dict(type='ELITE_MONSTER_KILL', timestamp=300000, killerTeamId=100, killerId=2, monsterType='DRAGON',
                monsterSubType='FIRE_DRAGON'),
           dict(type='ELITE_MONSTER_KILL', timestamp=480000, killerTeamId=200, killerId=7, monsterType='HORDE'),
           dict(type='BUILDING_KILL', timestamp=540000, teamId=200, buildingType='TOWER_BUILDING', towerType='OUTER_TURRET'),
           dict(type='TURRET_PLATE_DESTROYED', timestamp=400000, teamId=100),
           dict(type='DRAGON_SOUL_GIVEN', timestamp=720000, teamId=100, name='Infernal'),
           dict(type='ELITE_MONSTER_KILL', timestamp=780000, killerTeamId=100, killerId=1, monsterType='BARON_NASHOR'),
           dict(type='ITEM_PURCHASED', timestamp=15000, participantId=3),
           dict(type='GAME_END', timestamp=int(ts[-1]), winningTeam=100)]
    meta = dict(match_id=match, team_map={str(p): 100 if p <= 5 else 200 for p in range(1, 11)},
                role_slots={str(p): p - 1 for p in range(1, 11)})
    return dict(minute_ts=ts, node_minute=node, meta=meta, events=ev)


def names_of(pack):
    return list(StateBuilder(pack, NODE).at(60000).values)


@pytest.fixture(scope='module')
def candidate_a():
    info = json.loads((P1 / 'models' / 'inference_adapter.json').read_text(encoding='utf-8'))
    return T.CandidateA.load(P1 / 'models' / 'expanded_model_v2.joblib', info['model_sha256']['expanded'])


def tiny_b(names, seeds=(1, 2, 3), tmp=None):
    X = np.concatenate([T.assemble_history(T.BatchSource(MATCH, make_pack(seed=s), NODE), 600000, names).matrix()
                        for s in range(3)])
    pre = T.HistoryPreprocessor.fit(names, X)
    cals = [{'method': 'raw'}, {'method': 'sigmoid', 'sigmoid_coef': 1.1, 'sigmoid_intercept': -.05},
            {'method': 'isotonic', 'isotonic_x': np.asarray([0., .5, 1.]), 'isotonic_y': np.asarray([.1, .5, .9])}]
    members = []
    for seed, cal in zip(seeds, cals):
        rng = np.random.default_rng(seed)
        members.append({'seed': seed, 'calibration': cal, 'params': {
            'kernel': rng.normal(0, .1, (8, pre.input_width)), 'recurrent_kernel': rng.normal(0, .3, (8, 8)),
            'bias': rng.normal(0, .1, 8), 'output_kernel': rng.normal(0, .5, 8), 'output_bias': np.asarray(.1)}})
    b = T.CandidateB(pre, members)
    if tmp is not None:
        b.save(tmp)
        b = T.CandidateB.load(tmp)
    return b


@pytest.fixture(scope='module')
def candidate_b(tmp_path_factory):
    return tiny_b(names_of(make_pack()), tmp=tmp_path_factory.mktemp('bundle'))


def test_history_times_masks_and_padding():
    pack = make_pack()
    names = names_of(pack)
    h = T.assemble_history(T.BatchSource(MATCH, pack, NODE), 180000, names)
    assert h.times == tuple(180000 - (8 - k) * 60000 for k in range(9))
    assert h.mask == (False,) * 5 + (True,) * 4
    X = h.matrix()
    assert (X[:5] == 0).all() and np.isfinite(X).all()
    assert h.snapshot_ms[:5] == (-1,) * 5 and h.snapshot_ms[-1] <= 180000
    full = T.assemble_history(T.BatchSource(MATCH, pack, NODE), 540000, names)
    assert all(full.mask)
    with pytest.raises(T.SchemaError):
        T.CausalHistory(MATCH, 180000, 0, (None,) * 9, tuple(names), 'x')  # last position cannot be masked
    with pytest.raises(T.SchemaError):
        T.CausalHistory(MATCH, 180000, 0, h.states[:8], tuple(names), 'x')
    with pytest.raises(T.SchemaError):
        T.history_times(180000.5)


def test_masked_positions_do_not_affect_prediction(candidate_b):
    pack = make_pack()
    names = list(candidate_b.state_names)
    h = T.assemble_history(T.BatchSource(MATCH, pack, NODE), 200000, names)
    Xnum, C, mask = candidate_b.preprocessor.encode_histories(h.matrix()[None], np.asarray([h.mask]))
    p0, _, _ = candidate_b.predict_encoded(Xnum, C, mask)
    garbage = Xnum.copy()
    garbage[~mask] = 123.0
    Cg = C.copy()
    Cg[~mask] = 0
    p1, _, _ = candidate_b.predict_encoded(garbage, Cg, mask)
    assert np.array_equal(p0, p1)
    with pytest.raises(T.SchemaError):
        H = h.matrix()[None].copy()
        H[0, 0] = 1.
        candidate_b.preprocessor.encode_histories(H, np.asarray([h.mask]))


def perturb_after(pack, t):
    q = copy.deepcopy(pack)
    ts = q['minute_ts']
    q['node_minute'][ts > t] = q['node_minute'][ts > t] * -3.5 + 777.
    mutated = []
    for e in pack['events']:
        m = dict(e)
        if int(e['timestamp']) > t:
            for k in ('teamId', 'killerTeamId', 'winningTeam'):
                if int(m.get(k, 0) or 0) in (100, 200):
                    m[k] = 300 - int(m[k])
            for k in ('killerId', 'victimId'):
                if int(m.get(k, 0) or 0) > 0:
                    m[k] = int(m[k]) % 10 + 1
        mutated.append(m)
    mutated += [dict(type='DRAGON_SOUL_GIVEN', timestamp=t + 1, teamId=200, name='Cloud', dragonSoul='FIRE'),
                dict(type='ELITE_MONSTER_KILL', timestamp=t + 1, killerTeamId=200, killerId=6, monsterType='BARON_NASHOR'),
                dict(type='CHAMPION_KILL', timestamp=t + 1, killerId=1, victimId=6),
                dict(type='GAME_END', timestamp=t + 2, winningTeam=200)]
    q['events'] = mutated
    return q


def test_future_perturbation_invariance(candidate_a, candidate_b):
    pack = make_pack()
    names = list(candidate_a.state_names)
    assert names == names_of(pack) and tuple(names) == candidate_b.state_names
    sa, sb = T.TemporalWinProbabilityService(candidate_a), T.TemporalWinProbabilityService(candidate_b)
    for t in (120000, 300000, 300001, 479999, 700000):
        base = T.assemble_history(T.BatchSource(MATCH, pack, NODE), t, names)
        pert = T.assemble_history(T.BatchSource(MATCH, perturb_after(pack, t), NODE), t, names)
        assert np.array_equal(base.matrix(), pert.matrix()) and base.mask == pert.mask
        assert base.snapshot_ms == pert.snapshot_ms
        assert sa.predict(base).p == sa.predict(pert).p
        assert sb.predict(base).p == sb.predict(pert).p


def replay_log(pack, queries):
    """Chronological merge of frames and events (frames first at equal timestamps)."""
    ts = pack['minute_ts']
    stream = sorted([(int(t), 0, i) for i, t in enumerate(ts)] +
                    [(int(e['timestamp']), 1, i) for i, e in enumerate(pack['events'])], key=lambda r: (r[0], r[1]))
    log = T.ObservationLog(MATCH, pack['meta']['team_map'], NODE)
    pos = 0
    for q in sorted(queries):
        while pos < len(stream) and stream[pos][0] <= q:
            t, kind, i = stream[pos]
            if kind == 0:
                log.add_frame(MATCH, t, pack['node_minute'][i], NODE)
            else:
                log.add_event(MATCH, pack['events'][i])
            pos += 1
        yield q, log


def test_batch_and_incremental_histories_equal_at_current_query(candidate_a, candidate_b):
    pack = make_pack()
    names = list(candidate_a.state_names)
    sa, sb = T.TemporalWinProbabilityService(candidate_a), T.TemporalWinProbabilityService(candidate_b)
    batch = T.BatchSource(MATCH, pack, NODE)
    queries = [120000, 179999, 180000, int(pack['minute_ts'][3]) - 1, int(pack['minute_ts'][3]), 300000, 300001,
               430500, 555555, 720000, 780000, int(pack['minute_ts'][-1]) - 1]
    for q, log in replay_log(pack, queries):
        hb = T.assemble_history(batch, q, names)
        hi = T.assemble_history(log.source(q), q, names)
        assert np.array_equal(hb.matrix(), hi.matrix()) and hb.mask == hi.mask and hb.snapshot_ms == hi.snapshot_ms
        pa_b, pa_i = sa.predict(hb), sa.predict(hi)
        pb_b, pb_i = sb.predict(hb), sb.predict(hi)
        assert pa_b.p == pa_i.p and pb_b.p == pb_i.p and pb_b.components == pb_i.components
        assert hi.source_kind == 'incremental_log' and pa_i.snapshot_age_ms == q - hi.snapshot_ms[-1]


def test_observation_log_guards():
    pack = make_pack()
    names = names_of(pack)
    ts = pack['minute_ts']
    log = T.ObservationLog(MATCH, pack['meta']['team_map'], NODE)
    log.add_frame(MATCH, 0, pack['node_minute'][0], NODE)
    log.add_frame(MATCH, int(ts[1]), pack['node_minute'][1], NODE)
    with pytest.raises(T.ChronologyError):
        log.add_frame(MATCH, int(ts[1]), pack['node_minute'][1], NODE)
    with pytest.raises(T.ChronologyError):
        log.add_event(MATCH, dict(type='CHAMPION_KILL', timestamp=int(ts[1]) - 5, killerId=1, victimId=6))
    with pytest.raises(T.CrossMatchError):
        log.add_event('OTHER_MATCH', dict(type='CHAMPION_KILL', timestamp=int(ts[1]) + 5, killerId=1, victimId=6))
    with pytest.raises(T.CrossMatchError):
        log.add_frame('OTHER_MATCH', int(ts[2]), pack['node_minute'][2], NODE)
    with pytest.raises(T.SchemaError):
        log.add_frame(MATCH, int(ts[2]), pack['node_minute'][2], list(reversed(NODE)))
    changed = pack['node_minute'][2].copy()
    changed[0, -1] = 999
    with pytest.raises(T.SchemaError):
        log.add_frame(MATCH, int(ts[2]), changed, NODE)
    with pytest.raises(T.FutureObservationError):
        log.source(int(ts[1]) - 1)
    h = T.assemble_history(log.source(int(ts[1]) + 10), int(ts[1]) + 10, names)
    with pytest.raises(ValueError):
        T.assemble_history(log.source(int(ts[1]) + 10), int(ts[1]) + 11, names)
    log.add_event(MATCH, dict(type='GAME_END', timestamp=int(ts[1]) + 20, winningTeam=100))
    with pytest.raises(ValueError):
        log.source(int(ts[1]) + 20)
    with pytest.raises(T.SchemaError):
        T.ObservationLog(MATCH, {1: 100}, NODE)
    assert h.mask[-1]


def test_observed_state_history_guards():
    pack = make_pack()
    names = names_of(pack)
    b = StateBuilder(pack, NODE)
    q = 300000
    obs = [(MATCH, b.at(t)) for t in range(0, q + 1, 60000)]
    h = T.CausalHistory.from_observed_states(MATCH, q, 0, obs, names)
    assert np.array_equal(h.matrix(), T.assemble_history(T.BatchSource(MATCH, pack, NODE), q, names).matrix())
    with pytest.raises(T.ChronologyError):
        T.CausalHistory.from_observed_states(MATCH, q, 0, obs[::-1], names)
    with pytest.raises(T.CrossMatchError):
        T.CausalHistory.from_observed_states(MATCH, q, 0, obs[:-1] + [('OTHER', obs[-1][1])], names)
    with pytest.raises(T.FutureObservationError):
        T.CausalHistory.from_observed_states(MATCH, q, 0, obs + [(MATCH, b.at(q + 1))], names)
    legacy = State(obs[-1][1].values, q, obs[-1][1].snapshot_ms)
    with pytest.raises(T.SchemaError):
        T.CausalHistory.from_observed_states(MATCH, q, 0, obs[:-1] + [(MATCH, legacy)], names)
    s = obs[-1][1]
    reordered = StateV2(dict(reversed(list(s.values.items()))), s.query_ms, s.snapshot_ms, STATE_VERSION, 0)
    with pytest.raises(T.SchemaError):
        T.CausalHistory.from_observed_states(MATCH, q, 0, obs[:-1] + [(MATCH, reordered)], names)
    with pytest.raises(ValueError):
        T.CausalHistory.from_observed_states(MATCH, q, 0, obs[:2] + obs[3:], names)  # missing position, no interpolation


def test_service_and_delta_guards(candidate_a, candidate_b, tmp_path):
    pack = make_pack()
    names = list(candidate_a.state_names)
    src = T.BatchSource(MATCH, pack, NODE)
    sa, sb = T.TemporalWinProbabilityService(candidate_a), T.TemporalWinProbabilityService(candidate_b)
    pre, post = sb.predict_source(src, 299999), sb.predict_source(src, 390000)
    d = T.endpoint_delta(pre, post)
    assert d['delta'] == post.p - pre.p and d['model']['model_sha256'] == candidate_b.model_sha256
    other_b = tiny_b(names, seeds=(4, 5, 6), tmp=tmp_path / 'other')
    with pytest.raises(T.ModelVersionError):
        T.endpoint_delta(pre, T.TemporalWinProbabilityService(other_b).predict_source(src, 390000))
    with pytest.raises(T.ModelVersionError):
        T.endpoint_delta(sa.predict_source(src, 299999), post)
    with pytest.raises(T.ChronologyError):
        T.endpoint_delta(post, pre)
    other = T.TemporalWinProbabilityService(candidate_b).predict_source(T.BatchSource('TEST_0002', make_pack('TEST_0002'), NODE), 390000)
    with pytest.raises(T.CrossMatchError):
        T.endpoint_delta(pre, other)
    with pytest.raises(T.CrossMatchError):
        T.BatchSource('TEST_0002', pack, NODE)
    with pytest.raises(T.ModelVersionError):
        T.TemporalWinProbabilityService(tiny_b(names))  # unsaved, unhashed model
    h = T.assemble_history(src, 300000, names)
    wrong = T.CausalHistory.__new__(T.CausalHistory)
    object.__setattr__(wrong, '__dict__', dict(h.__dict__, history_version='old_history'))
    with pytest.raises(T.ModelVersionError):
        sb.predict(wrong)
    with pytest.raises(T.ModelVersionError):
        candidate_b.predict_history_arrays(h.matrix()[None], np.asarray([h.mask]), names, 'objective_history_v1', T.HISTORY_VERSION)
    with pytest.raises(T.SchemaError):
        candidate_b.predict_history_arrays(h.matrix()[None], np.asarray([h.mask]), names[::-1], STATE_VERSION, T.HISTORY_VERSION)


def test_candidate_b_save_load_identity_and_tamper(candidate_b, tmp_path):
    pack = make_pack(seed=3)
    names = list(candidate_b.state_names)
    hs = [T.assemble_history(T.BatchSource(MATCH, pack, NODE), q, names) for q in (130000, 420000, 800000)]
    p0, m0 = candidate_b.predict_histories(hs)
    manifest = candidate_b.save(tmp_path / 'b')
    loaded = T.CandidateB.load(tmp_path / 'b', expected_bundle_sha256=manifest['bundle_sha256'])
    p1, m1 = loaded.predict_histories(hs)
    assert np.array_equal(p0, p1) and all(np.array_equal(m0[k], m1[k]) for k in m0)
    with pytest.raises(T.ModelVersionError):
        T.CandidateB.load(tmp_path / 'b', expected_bundle_sha256='0' * 64)
    raw = (tmp_path / 'b' / 'member_seed1.npz').read_bytes()
    (tmp_path / 'b' / 'member_seed1.npz').write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    with pytest.raises(T.ModelVersionError):
        T.CandidateB.load(tmp_path / 'b')
    (tmp_path / 'b' / 'member_seed1.npz').write_bytes(raw)
    m = json.loads((tmp_path / 'b' / 'manifest.json').read_text(encoding='utf-8'))
    m['candidate_version'] = 'simplernn_history_v0'
    m.pop('bundle_sha256')
    m['bundle_sha256'] = T.sha256_json(m)
    (tmp_path / 'b' / 'manifest.json').write_text(json.dumps(m), encoding='utf-8')
    with pytest.raises(T.ModelVersionError):
        T.CandidateB.load(tmp_path / 'b')


def test_candidate_a_hash_and_version_guards(candidate_a):
    with pytest.raises(T.ModelVersionError):
        T.CandidateA.load(P1 / 'models' / 'expanded_model_v2.joblib', '0' * 64)
    bad = copy.copy(candidate_a.adapter)
    bad.model_version = 'independent_wp_v1'
    with pytest.raises(T.ModelVersionError):
        T.CandidateA(bad, candidate_a.model_sha256)


def test_unseen_champion_contributes_zero(candidate_b):
    pack = make_pack()
    names = list(candidate_b.state_names)
    h = T.assemble_history(T.BatchSource(MATCH, pack, NODE), 500000, names)
    H = h.matrix()[None]
    mask = np.asarray([h.mask])
    Xnum, C, _ = candidate_b.preprocessor.encode_histories(H, mask)
    H2 = H.copy()
    H2[0, :, names.index('participant_slot3_champion_id')] = 99999.
    X2, C2, _ = candidate_b.preprocessor.encode_histories(H2, mask)
    assert (C2[0, :, 3] == -1).all() and np.array_equal(X2, Xnum)
    C_removed = C.copy()
    C_removed[0, :, 3] = -1
    assert np.array_equal(candidate_b.predict_encoded(X2, C2, mask)[0], candidate_b.predict_encoded(Xnum, C_removed, mask)[0])


def test_sequence_membership_detects_leakage():
    table_match = np.asarray(['m1'] * 10 + ['m2'] * 10)
    table_time = np.asarray(list(range(0, 600000, 60000)) * 2, dtype=np.int64)
    q = np.asarray([540000, 540000], dtype=np.int64)
    times = q[:, None] - (8 - np.arange(9)) * 60000
    idx = np.stack([np.arange(1, 10), np.arange(11, 20)])
    mask = np.ones((2, 9), dtype=bool)
    ok = T.check_sequence_membership(table_match, table_time, idx, mask, times, np.asarray(['m1', 'm2']), q)
    assert ok['violations'] == 0
    bad = idx.copy()
    bad[1, 4] = 5
    assert T.check_sequence_membership(table_match, table_time, bad, mask, times, np.asarray(['m1', 'm2']), q)['cross_match_positions'] == 1
    m2 = mask.copy()
    m2[0, -1] = False
    assert T.check_sequence_membership(table_match, table_time, idx, m2, times, np.asarray(['m1', 'm2']), q)['violations'] >= 1


def test_numpy_inference_matches_torch_training_forward():
    pytest.importorskip('torch')
    pack = make_pack()
    names = names_of(pack)
    src = T.BatchSource(MATCH, pack, NODE)
    hs = [T.assemble_history(src, q, names) for q in range(120000, 900000, 60000)]
    H, mask = np.stack([h.matrix() for h in hs]), np.asarray([h.mask for h in hs])
    pre = T.HistoryPreprocessor.fit(names, H[mask])
    Z, C = pre.encode_states(H[mask])
    index = np.full(mask.shape, -1)
    index[mask] = np.arange(mask.sum())
    y = (np.arange(len(hs)) % 2).astype(float)
    params, log, eval_torch = T.train_simplernn(Z, C, index, mask, y, pre.input_width, seed=17, epochs=2, batch_size=4, threads=1)
    assert len(log) == 2 and all(r['finite_parameters'] for r in log)
    Xnum, Ce, _ = pre.encode_histories(H, mask)
    p_np, _ = T.simplernn_forward(params, Xnum, Ce, mask)
    assert np.max(np.abs(p_np - eval_torch(Xnum, Ce, mask))) < 1e-5


def _sha(path):
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def test_frozen_legacy_hashes():
    after = json.loads((P1 / 'hashes_after.json').read_text(encoding='utf-8'))
    for rel, digest in after['legacy_sources'].items():
        base = ROOT if rel.startswith('tests/') else WT
        assert _sha(base / rel) == digest, rel
    for rel, digest in after['state_value_v2_fix'].items():
        assert _sha(ROOT / 'outputs' / 'state_value_v2_fix' / rel) == digest, rel
    info = json.loads((P1 / 'models' / 'inference_adapter.json').read_text(encoding='utf-8'))
    assert _sha(WT / 'train' / 'independent_winprob_v2.py') == info['module_sha256']
    for fam, digest in info['model_sha256'].items():
        assert _sha(P1 / 'models' / f'{fam}_model_v2.joblib') == digest
