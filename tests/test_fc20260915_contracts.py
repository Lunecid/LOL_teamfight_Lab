"""Contract tests for the 2026-09-15 full-corpus pipeline (synthetic and TRAIN-only fixtures; no TEST data).

Covers: split membership formulas, V query sampling, positive-slope calibration, OOF self-outcome exclusion,
future perturbation invariance, causal endpoint boundaries, no postgame predictors, transform fit origins,
model serialization guards, q candidate reproducibility and independent metric reconstruction.
"""
import copy
import hashlib
import inspect
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
os.environ.setdefault('LOL_OUTPUT_ROOT', str(ROOT / 'outputs/full_corpus_training_20260915/runtime'))
import fc20260915_common as C  # noqa: E402
sys.path.insert(0, str(C.WT))
import engagement_labels_v3_rules as R  # noqa: E402


class _St:
    def log(self, msg):
        pass


# ----------------------------------------------------------------------------------------- partitions
def test_partition_formulas_match_spec_exactly():
    for mid in ('KR_7715477686', 'KR_7715479065', 'NA1_5300000000', 'KR_1'):
        fold = int(hashlib.sha256(('full-v-oof-20260915:' + mid).encode()).hexdigest()[:8], 16) % 5
        role = ['V_CAL', 'V_SELECT', 'Q_CAL', 'Q_SELECT'][int(hashlib.sha256(('full-val-20260915:' + mid).encode()).hexdigest()[:8], 16) % 4]
        assert C.train_fold(mid) == fold
        assert C.validation_role(mid) == role


def test_manifest_roles_and_partition_coverage():
    import csv
    rows = list(csv.DictReader(open(C.MAIN_MANIFEST, encoding='utf-8')))
    by = {}
    for r in rows:
        by.setdefault(r['role'], []).append(r['match_id'])
    assert {k: len(v) for k, v in by.items()} == {k: v[1] for k, v in C.MAIN_EXPECTED.items()}
    folds = np.bincount([C.train_fold(m) for m in by['TRAIN']], minlength=5)
    roles = {r: 0 for r in C.VAL_ROLES}
    for m in by['VALIDATION']:
        roles[C.validation_role(m)] += 1
    assert folds.sum() == 74673 and (folds > 14000).all()
    assert sum(roles.values()) == 74748 and min(roles.values()) > 17000
    assert not (set(by['TRAIN']) & set(by['VALIDATION'])) and not (set(by['TEST']) & (set(by['TRAIN']) | set(by['VALIDATION'])))


# ----------------------------------------------------------------------------------------- V sampling
def test_value_grid_and_bucket_sampling():
    grid = C.value_grid(terminal_ms=1_830_000, first_frame_ms=0, last_frame_ms=1_830_500)
    assert grid[0] == 120000 and grid[-1] == 1_800_000 and all(t < 1_830_000 for t in grid)
    assert C.value_grid(1_800_000, 0, 1_900_000)[-1] == 1_740_000          # strictly before terminal
    assert C.value_grid(1_800_000, 0, 1_500_000)[-1] == 1_500_000          # never after the last frame
    s = C.bucket_sample('KR_X', grid)
    buckets = sorted({t // 300000 for t in grid})
    assert [t // 300000 for t in s] == buckets and set(s) <= set(grid)
    for t in s:
        same = [u for u in grid if u // 300000 == t // 300000]
        assert t == min(same, key=lambda u: hashlib.sha256(f'KR_X:{u}:full-v-query-20260915'.encode()).hexdigest())
    assert s == C.bucket_sample('KR_X', list(reversed(grid)))
    assert list(inspect.signature(C.bucket_sample).parameters) == ['match_id', 'grid']   # no outcome input


# ----------------------------------------------------------------------------------------- calibration
def test_positive_slope_sigmoid_constraint_weights_and_fit():
    rng = np.random.default_rng(0)
    p = rng.uniform(.02, .98, 4000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    cal = C.PositiveSlopeSigmoid().fit(p, y, np.ones(p.size))
    assert abs(cal.b - 1) < .15 and abs(cal.a) < .15 and cal.fit_info['success']
    anti = C.PositiveSlopeSigmoid().fit(p, 1 - y, np.ones(p.size))
    assert anti.b >= C.PositiveSlopeSigmoid.SLOPE_MIN and anti.fit_info['slope_bound_active']
    q = np.sort(p)
    assert np.all(np.diff(cal.predict(q)) >= 0)
    dup = C.PositiveSlopeSigmoid().fit(np.concatenate([p, p[:100]]), np.concatenate([y, y[:100]]), np.ones(p.size + 100))
    w = np.ones(p.size)
    w[:100] = 2
    wt = C.PositiveSlopeSigmoid().fit(p, y, w)
    assert abs(dup.a - wt.a) < 1e-6 and abs(dup.b - wt.b) < 1e-6
    assert np.array_equal(C.IdentityCalibrator().predict(p), p)


# ----------------------------------------------------------------------------------------- synthetic V data
def _synthetic_states(n_matches=240, seed=1):
    rng = np.random.default_rng(seed)
    names = ['time_minutes', 'time_minutes_sq', 'snapshot_age_s', 'unknown_objective_team_count'] + \
            [f'participant_slot{i}_totalGold_norm' for i in range(10)] + [f'participant_slot{i}_champion_id' for i in range(10)] + \
            ['blue_kills', 'red_kills']
    rows, g, y = [], [], []
    for m in range(n_matches):
        w = int(rng.integers(0, 2))
        for q in range(int(rng.integers(3, 8))):
            t = 2 + 5 * q
            x = [t, t * t, rng.uniform(0, 60), 0] + list(rng.normal(t / 30 + (w - .5) * .2, .1, 10)) + \
                list(rng.integers(1, 30, 10)) + [rng.poisson(t / 2 + 2 * w), rng.poisson(t / 2 + 2 * (1 - w))]
            rows.append(x)
            g.append(f'KR_{m}')
            y.append(w)
    return names, np.asarray(rows, dtype=float), np.asarray(g), np.asarray(y)


def test_oof_self_outcome_exclusion_and_membership():
    import fc20260915_fit_v as FV
    names, X, g, y = _synthetic_states()
    fold = np.asarray([C.train_fold(m) for m in g])
    k = 2
    tr = fold != k
    m1, cols, rec = FV.fit_base(X[tr], y[tr], g[tr], names, _St(), 'fold')
    y_flip = y.copy()
    y_flip[fold == k] = 1 - y_flip[fold == k]                     # held-out outcomes changed
    m2, _, rec2 = FV.fit_base(X[fold != k], y_flip[fold != k], g[fold != k], names, _St(), 'fold')
    keep = [names.index(c) for c in cols]
    held = X[fold == k][:, keep]
    assert np.array_equal(m1.predict_proba(held), m2.predict_proba(held))
    assert rec['fit_match_ids_sha256'] == rec2['fit_match_ids_sha256']
    assert not (set(g[fold == k]) & set(g[tr]))
    y_in = y.copy()
    y_in[(fold != k)] = 1 - y_in[(fold != k)]                       # in-fit outcomes changed -> model changes
    m3, _, _ = FV.fit_base(X[tr], y_in[tr], g[tr], names, _St(), 'fold')
    assert not np.array_equal(m1.predict_proba(held), m3.predict_proba(held))


def test_transform_fit_origin_and_categorical_champions():
    import fc20260915_fit_v as FV
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    names, X, g, y = _synthetic_states()
    tr = np.asarray([C.train_fold(m) for m in g]) != 0
    model, cols, rec = FV.fit_base(X[tr], y[tr], g[tr], names, _St(), 'origin')
    keep = [names.index(c) for c in cols]
    num = [i for i, c in enumerate(cols) if not c.endswith('champion_id')]
    Xn = X[tr][:, keep][:, num]
    ref = StandardScaler().fit(SimpleImputer(strategy='median').fit_transform(Xn))
    sc = model.named_steps['preprocess'].named_transformers_['numeric'].named_steps['scale']
    assert np.allclose(sc.mean_, ref.mean_, rtol=0, atol=1e-12) and np.allclose(sc.scale_, ref.scale_, rtol=0, atol=1e-12)
    assert rec['champion_one_hot'] == [f'participant_slot{i}_champion_id' for i in range(10)]
    assert 'snapshot_age_s' not in cols


def test_adapter_serialization_and_guards():
    import joblib
    import fc20260915_fit_v as FV
    names, X, g, y = _synthetic_states()
    model, cols, rec = FV.fit_base(X, y, g, names, _St(), 'ser')
    cal = C.PositiveSlopeSigmoid().fit(model.predict_proba(X[:, [names.index(c) for c in cols]])[:, 1], y, C.weights(g))
    ad = C.FullCorpusWinProbV2(names, cols, model, cal, dict(adapter='test'))
    p = ad.predict_matrix(X, names, C.STATE_VERSION)
    with tempfile.TemporaryDirectory(dir=str(C.OUT / 'probes')) as d:
        path = Path(d) / 'a.joblib'
        joblib.dump(ad, path)
        sha = C.sha256_file(path)
        re = C.load_v_adapter(path, sha)
        assert np.array_equal(re.predict_matrix(X, names, C.STATE_VERSION), p)
        with pytest.raises(ValueError):
            C.load_v_adapter(path, '0' * 64)
    with pytest.raises(ValueError):
        ad.predict_matrix(X, names, 'objective_history_v1')
    with pytest.raises(ValueError):
        ad.predict_matrix(X, list(reversed(names)), C.STATE_VERSION)
    with pytest.raises(ValueError):
        C.FullCorpusWinProbV2(names, cols + ['snapshot_age_s'], model, cal, {})


# ----------------------------------------------------------------------------------------- boundaries / causality
def test_endpoint_rule_causal_boundaries():
    L = 100_000
    e, r, _ = R.endpoint_rule(L, 90, R.INF, R.INF, 10_000_000)
    assert e == L + 90_000 and r == ('horizon',)
    e, r, _ = R.endpoint_rule(L, 90, L + 30_000, R.INF, 10_000_000)
    assert e == L + 29_999 and r == ('next_kill',)
    e, r, _ = R.endpoint_rule(L, 90, R.INF, L + 20_000, 10_000_000)
    assert e == L + 19_999 and r == ('next_engagement_start',)
    e, r, _ = R.endpoint_rule(L, 90, L + 40_001, L + 40_001, L + 40_001)
    assert e == L + 40_000 and r == ('next_kill', 'next_engagement_start', 'game_end')
    assert R.next_kill_after([90_000, 100_000, 100_000, 130_000], L) == 130_000     # strictly after L
    assert R.effective_next_start(500, 500) == R.INF and R.effective_next_start(499, 500) == 499
    e, _, _ = R.endpoint_rule(L, 90, R.INF, L - 5_000 + 1, 10_000_000)             # overlap next_start <= L
    assert not all(R.endpoint_validity(e, L, L - 30_000, 0, 10_000_000).values())
    assert R.label_from_delta(0.0) == 0 and R.label_from_delta(1e-15) == 1


def _pack():
    sys.path.insert(0, str(C.WT))
    from gameplay.state_value import SNAPSHOT_FIELDS
    names = list(SNAPSHOT_FIELDS) + ['champion_id']
    node = np.zeros((4, 10, len(names)))
    for pid in range(1, 11):
        node[:, pid - 1, :] = pid
    return dict(minute_ts=np.array([0, 60000, 120000, 180000]), node_minute=node,
                meta=dict(team_map={p: 100 if p <= 5 else 200 for p in range(1, 11)}),
                events=[dict(type='CHAMPION_KILL', timestamp=50_000, killerId=1, victimId=6)]), names


def test_future_perturbation_invariance_synthetic():
    from gameplay.state_value_v2 import StateBuilder
    p, n = _pack()
    q = 90_000
    base = StateBuilder(p, n).at(q).values
    f = copy.deepcopy(p)
    f['node_minute'][f['minute_ts'] > q] = 777.0
    f['events'] += [dict(type='CHAMPION_KILL', timestamp=q + 1, killerId=6, victimId=1),
                    dict(type='ELITE_MONSTER_KILL', timestamp=q + 5, killerTeamId=100, monsterType='BARON_NASHOR'),
                    dict(type='GAME_END', timestamp=q + 10, winningTeam=200)]
    assert StateBuilder(f, n).at(q).values == base


def test_future_perturbation_invariance_real_train_fixture():
    """TRAIN-only real cache fixture: mutate every frame/event after q_pre; pre state bitwise unchanged."""
    import csv
    import fc20260915_extract as X
    ctx = X._context()
    ctx['cio'].CACHE_DIR = C.CACHE_MAIN
    ex_rows = {}
    with open(C.EXPOSURES, encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            if r['patch'] == '15.14':
                ex_rows.setdefault(r['match'], []).append(r)
            if len(ex_rows) >= 6:
                break
    names = X.expected_state_names()
    for mid, rows in list(ex_rows.items())[:5]:
        pack = ctx['cio'].load_match_cache(mid)
        q = int(rows[0]['s']) - 1
        base = ctx['state_matrix']([ctx['V2'](pack, ctx['node_names']).at(q)], names)
        mutated = dict(pack)
        ts = np.asarray(pack['minute_ts'])
        node = np.array(pack['node_minute'], copy=True)
        node[ts > q] = node[ts > q] * -2.0 + 55.0
        mutated['node_minute'] = node
        mutated['events'] = [e for e in pack['events'] if int(e['timestamp']) <= q] + \
            [dict(type='DRAGON_SOUL_GIVEN', timestamp=q + 1, teamId=200, name='Ocean'),
             dict(type='BUILDING_KILL', timestamp=q + 2, teamId=100, buildingType='INHIBITOR_BUILDING')]
        assert np.array_equal(ctx['state_matrix']([ctx['V2'](mutated, ctx['node_names']).at(q)], names), base)


# ----------------------------------------------------------------------------------------- predictors
def test_no_postgame_predictors_in_v_or_q():
    import fc20260915_extract as X
    names = X.expected_state_names()
    forbidden = ('winner', 'terminal', 'duration', 'game_end', 'endpoint', 'label', 'fold', 'patch', 'match')
    assert not [n for n in names if any(t in n for t in forbidden)]
    from train.temporal_winprob import feature_matrix
    _, cols = feature_matrix(np.zeros((1, len(names))), names, 'expanded')
    assert 'snapshot_age_s' not in cols
    q_names = [n for n in names if n != 'snapshot_age_s'] + [C.P_PRE]
    sets = C.q_feature_sets(q_names)
    assert len(sets['ridge']) == 352
    bad = [n for n in sets['ridge'] + sets['economic'] if any(t in n for t in C.Q_FORBIDDEN_TOKENS) and n != C.P_PRE]
    assert not bad and not [n for n in sets['ridge'] if n.endswith(('_x', '_y'))]


def test_q_candidates_reproducible_after_serialization():
    import joblib
    import fc20260915_fit_q as FQ
    rng = np.random.default_rng(3)
    n = 3000
    names = ['time_minutes', 'participant_slot0_totalGold_norm', 'blue_kills', 'red_kills', 'participant_slot0_champion_id', C.P_PRE]
    X = np.column_stack([rng.uniform(2, 40, n), rng.uniform(0, 1, n), rng.poisson(5, n), rng.poisson(5, n),
                         rng.integers(1, 100, n), rng.uniform(.05, .95, n)]).astype(float)
    y = (rng.uniform(size=n) < X[:, -1]).astype(int)
    g = np.asarray([f'KR_{i // 3}' for i in range(n)])
    part = np.asarray([C.h8('t:', m) % 3 for m in g])
    M = dict(train=part == 0, calibrate=part == 1, select=part == 2)
    sets = C.q_feature_sets(names)
    cix = {k: [names.index(v) for v in vs] for k, vs in sets.items()}
    B, fl = FQ.fit_candidates(X, y, g, M, cix, sets, dict(test=True))
    assert abs(B['constant']['prior'] - np.average(y[M['train']], weights=C.weights(g[M['train']]))) < 1e-15
    o = FQ.origins_check(B, X, y, g, M, cix)
    assert all(o.values())
    with tempfile.TemporaryDirectory(dir=str(C.OUT / 'probes')) as d:
        for c in C.Q_CANDIDATES:
            joblib.dump(B[c], Path(d) / f'{c}.joblib')
            assert np.array_equal(C.predict_q_bundle(joblib.load(Path(d) / f'{c}.joblib'), X), C.predict_q_bundle(B[c], X))


# ----------------------------------------------------------------------------------------- metrics
def test_independent_metric_reconstruction():
    rng = np.random.default_rng(5)
    n = 5000
    p = rng.uniform(0, 1, n)
    p[:50] = 0.0
    p[50:100] = 1.0
    p[100:400] = .5
    y = (rng.uniform(size=n) < p).astype(int)
    g = np.asarray([f'm{i // 4}' for i in range(n)])
    w = C.weights(g)
    s = C.score(y, p, w)
    assert abs(C.auc_direct(y, p, w) - s['auc']) < 1e-12
    assert abs(C.brier_direct(y, p, w) - s['brier']) < 1e-12
    assert abs(C.logloss_direct(y, p, w) - s['logloss']) < 1e-10
    tot = np.bincount(np.unique(g, return_inverse=True)[1], weights=w)
    assert np.allclose(tot, tot[0]) and abs(w.mean() - 1) < 1e-12
