"""Label validity: pure analysis functions (no I/O). Tested on synthetic data by tests/test_lv20260915_contracts.py.

Definitions follow protocol.json (analysis section). Weights: equal total weight per match within each cell.
"""
from __future__ import annotations

import hashlib

import numpy as np

import fc20260915_common as C

SPARSE_MATCHES = 30
DRAGON_ELEMENTS = ('AIR', 'EARTH', 'FIRE', 'WATER', 'HEXTECH', 'CHEMTECH', 'OTHER')
OBJECTIVES_OWNED = ('baron', 'dragon', 'elder', 'herald', 'horde', 'atakhan', 'soul_owned')
RESOURCE_CODES = {1: 'blue_gain_relative', 0: 'observed_exact_zero', -1: 'red_gain_relative', 2: 'stale_same_frame'}
KILL_CODES = {1: 'blue_more_kills', 0: 'tie', -1: 'red_more_kills'}


# ------------------------------------------------------------------ basic
def sign3(x):
    x = np.asarray(x, dtype=float)
    return np.where(x > 0, 1, np.where(x < 0, -1, 0)).astype(np.int8)


def label_from_delta(delta):
    """Y = 1[delta > 0]; exact zero -> 0; NaN -> -1 (invalid row)."""
    d = np.asarray(delta, dtype=float)
    return np.where(np.isfinite(d), (d > 0).astype(np.int8), -1).astype(np.int8)


def wrate(ind, g):
    """(row-weighted, match-weighted) mean of a boolean/0-1 indicator."""
    ind = np.asarray(ind, dtype=float)
    if not len(ind):
        return None, None
    return float(ind.mean()), float(np.average(ind, weights=C.weights(g)))


def cell_meta(g):
    n = int(len(g))
    m = int(len(np.unique(g))) if n else 0
    return dict(rows=n, matches=m, empty=n == 0, sparse_lt30_matches=m < SPARSE_MATCHES)


def spearman(a, b):
    from scipy.stats import rankdata
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def agreement_cell(yA, yB, dA, dB, g):
    """A-vs-B label agreement for one cell of valid rows."""
    yA, yB = np.asarray(yA), np.asarray(yB)
    dA, dB = np.asarray(dA, dtype=float), np.asarray(dB, dtype=float)
    out = cell_meta(g)
    if out['empty']:
        return out
    dis = yA != yB
    out['disagreement_row'], out['disagreement_match_weighted'] = wrate(dis, g)
    out['disagreement_rows'] = int(dis.sum())
    for nm, y, d in (('A', yA, dA), ('B', yB, dB)):
        out[f'positive_rate_{nm}_row'], out[f'positive_rate_{nm}_match_weighted'] = wrate(y == 1, g)
        out[f'exact_zero_delta_{nm}'] = int(np.sum(d == 0))
    diff = dB - dA
    ad = np.abs(diff)
    out.update(delta_diff_mean_B_minus_A=float(diff.mean()), delta_absdiff_mean=float(ad.mean()),
               delta_absdiff_median=float(np.median(ad)), delta_absdiff_p90=float(np.quantile(ad, .9)),
               delta_pearson=float(np.corrcoef(dA, dB)[0, 1]) if len(dA) > 2 and np.std(dA) > 0 and np.std(dB) > 0 else None,
               delta_spearman=spearman(dA, dB),
               A_pos_B_neg_rows=int(np.sum((yA == 1) & (yB == 0))), A_neg_B_pos_rows=int(np.sum((yA == 0) & (yB == 1))))
    return out


def paired_match_bootstrap(g, indicators, reps, seed):
    """Match-resampling bootstrap of row- and match-weighted means of several indicators (same draws for all)."""
    g = np.asarray(g)
    u, inv = np.unique(g, return_inverse=True)
    nm = len(u)
    if nm == 0:
        return dict(matches=0, rows=0, empty=True)
    w = C.weights(g)
    Wm = np.bincount(inv, weights=w, minlength=nm)
    Rm = np.bincount(inv, minlength=nm).astype(float)
    Sw = {k: np.bincount(inv, weights=w * np.asarray(v, dtype=float), minlength=nm) for k, v in indicators.items()}
    Sr = {k: np.bincount(inv, weights=np.asarray(v, dtype=float), minlength=nm) for k, v in indicators.items()}
    rng = np.random.default_rng(seed)
    draws = {k: dict(row=np.empty(reps), match_weighted=np.empty(reps)) for k in indicators}
    for r in range(reps):
        mult = np.bincount(rng.integers(0, nm, size=nm), minlength=nm).astype(float)
        mw, mr = mult @ Wm, mult @ Rm
        for k in indicators:
            draws[k]['match_weighted'][r] = (mult @ Sw[k]) / mw
            draws[k]['row'][r] = (mult @ Sr[k]) / mr
    out = dict(replicates=reps, seed=seed, matches=int(nm), rows=int(len(g)), sparse_lt30_matches=nm < SPARSE_MATCHES,
               unit='matches resampled with replacement; equal per-match weight x multiplicity; fixed models',
               estimates={})
    for k, v in indicators.items():
        v = np.asarray(v, dtype=float)
        out['estimates'][k] = dict(
            row=dict(estimate=float(v.mean()), ci95=np.quantile(draws[k]['row'], [.025, .975]).tolist()),
            match_weighted=dict(estimate=float(np.average(v, weights=w)),
                                ci95=np.quantile(draws[k]['match_weighted'], [.025, .975]).tolist()))
    out['_draws_first5'] = {k: {kk: vv[:5].tolist() for kk, vv in d.items()} for k, d in draws.items()}
    return out


def naive_bootstrap_replicates(g, ind, reps, seed):
    """Independent slow recomputation: expand resampled matches row by row (used in validation only)."""
    g = np.asarray(g)
    u, inv = np.unique(g, return_inverse=True)
    rows_of = [np.flatnonzero(inv == i) for i in range(len(u))]
    rng = np.random.default_rng(seed)
    res = []
    ind = np.asarray(ind, dtype=float)
    for _ in range(reps):
        pick = rng.integers(0, len(u), size=len(u))
        num = den = numr = denr = 0.0
        for i in pick:
            r = rows_of[i]
            num += ind[r].mean()   # equal total weight per match: mean within match
            den += 1.0
            numr += ind[r].sum()
            denr += len(r)
        res.append((numr / denr, num / den))
    return res


def generate_labels(adapter_ids, adapters, names, state_version, X_pre, pre_ok, X_post, valid):
    """One adapter per row for BOTH ends. X_post / valid: dicts keyed by horizon. Invalid rows -> NaN / -1."""
    n = len(adapter_ids)
    p_pre = np.full(n, np.nan)
    p_post = {h: np.full(n, np.nan) for h in X_post}
    used = np.full(n, '', dtype='U16')
    for aid in sorted(set(np.asarray(adapter_ids).tolist())):
        rows = np.asarray(adapter_ids) == aid
        ad = adapters[aid]
        used[rows] = aid
        r = rows & (np.asarray(pre_ok) == 1)
        if r.any():
            p_pre[r] = ad.predict_matrix(X_pre[r], names, state_version)
        for h in X_post:
            v = rows & (np.asarray(valid[h]) == 1)
            if v.any():
                if not np.all(np.asarray(pre_ok)[v] == 1):
                    raise ValueError('valid row without a pre state')
                p_post[h][v] = ad.predict_matrix(X_post[h][v], names, state_version)
    delta = {h: np.where(np.asarray(valid[h]) == 1, p_post[h] - p_pre, np.nan) for h in X_post}
    Y = {h: label_from_delta(delta[h]) for h in X_post}
    return dict(p_pre=p_pre, p_post=p_post, delta=delta, Y=Y, adapter_used=used)


# ------------------------------------------------------------------ strata
def bin_abs_delta(d):
    a = np.abs(np.asarray(d, dtype=float))
    return np.where(a <= .005, '[0,.005]', np.where(a <= .01, '(.005,.01]', np.where(a <= .02, '(.01,.02]', '>.02')))


def bin_p_pre(p):
    p = np.asarray(p, dtype=float)
    return np.where(p < .2, '[0,.2)', np.where(p < .4, '[.2,.4)', np.where(p < .6, '[.4,.6)', np.where(p < .8, '[.6,.8)', '[.8,1]'))))


def bin_start_minutes(s_ms):
    m = np.asarray(s_ms, dtype=float) / 60000.
    return np.where(m < 2, '<2', np.where(m < 10, '[2,10)', np.where(m < 20, '[10,20)', np.where(m < 30, '[20,30)', '>=30'))))


def bin_age_s(a):
    a = np.asarray(a, dtype=float)
    return np.where(a < 15, '[0,15)', np.where(a < 30, '[15,30)', np.where(a < 45, '[30,45)', np.where(a < 60, '[45,60)', '>=60'))))


FINE_NAMES = {0: 'pick', 1: 'skirmish', 2: 'teamfight', -1: 'unknown'}


# ------------------------------------------------------------------ observed quantities from StateV2
def observed_from_states(names, X_pre, X_post):
    ix = {n: i for i, n in enumerate(names)}

    def col(X, n):
        return X[:, ix[n]]

    def team_sum(X, suffix, team):
        slots = range(0, 5) if team == 'blue' else range(5, 10)
        return sum(col(X, f'participant_slot{i}_{suffix}') for i in slots)

    out = dict(
        kills_blue=col(X_post, 'blue_kills') - col(X_pre, 'blue_kills'),
        kills_red=col(X_post, 'red_kills') - col(X_pre, 'red_kills'),
        deaths_blue=sum(col(X_post, f'participant_slot{i}_deaths') - col(X_pre, f'participant_slot{i}_deaths') for i in range(5)),
        deaths_red=sum(col(X_post, f'participant_slot{i}_deaths') - col(X_pre, f'participant_slot{i}_deaths') for i in range(5, 10)),
        gold_diff_pre_norm=team_sum(X_pre, 'totalGold_norm', 'blue') - team_sum(X_pre, 'totalGold_norm', 'red'),
        gold_diff_post_norm=team_sum(X_post, 'totalGold_norm', 'blue') - team_sum(X_post, 'totalGold_norm', 'red'),
        xp_diff_pre_norm=team_sum(X_pre, 'xp_norm', 'blue') - team_sum(X_pre, 'xp_norm', 'red'),
        xp_diff_post_norm=team_sum(X_post, 'xp_norm', 'blue') - team_sum(X_post, 'xp_norm', 'red'),
    )
    out['kill_diff'] = out['kills_blue'] - out['kills_red']
    out['gold_diff_change_norm'] = out['gold_diff_post_norm'] - out['gold_diff_pre_norm']
    out['xp_diff_change_norm'] = out['xp_diff_post_norm'] - out['xp_diff_pre_norm']
    for team in ('blue', 'red'):
        for obj in ('baron', 'elder', 'herald', 'horde', 'atakhan', 'dragons'):
            out[f'state_{obj}_{team}'] = col(X_post, f'{team}_{obj}') - col(X_pre, f'{team}_{obj}')
        for d in DRAGON_ELEMENTS:
            out[f'state_dragon_{d}_{team}'] = col(X_post, f'{team}_dragon_{d}') - col(X_pre, f'{team}_dragon_{d}')
        out[f'state_soul_recorded_{team}'] = col(X_post, f'{team}_soul_event_recorded') - col(X_pre, f'{team}_soul_event_recorded')
    return out


def resource_category(change, pre_snapshot, post_snapshot):
    """2 = stale same frame (not observed), else sign3(change)."""
    s = sign3(change)
    return np.where(np.asarray(post_snapshot) == np.asarray(pre_snapshot), 2, s).astype(np.int8)


def direction_conflict(y_primary, kill_sign, gold_cat):
    """Observed kill/gold direction nonzero and opposite to the A direction (+1 if Y=1 else -1); stale gold ignored."""
    a_dir = np.where(np.asarray(y_primary) == 1, 1, -1)
    ks = np.asarray(kill_sign)
    gc = np.asarray(gold_cat)
    kill_conf = (ks != 0) & (ks != a_dir)
    gold_conf = np.isin(gc, (1, -1)) & (gc != a_dir)
    return kill_conf, gold_conf


# ------------------------------------------------------------------ objective tables
def objective_team_columns(counts, count_keys, obj):
    """(blue, red, total, unknown) counts for an owned category from counter matrices."""
    k = list(count_keys)
    total = counts[:, k.index(obj)]
    blue = counts[:, k.index(obj + '_blue')]
    red = counts[:, k.index(obj + '_red')]
    return blue, red, total, total - blue - red


# ------------------------------------------------------------------ raw-event recount (independent of stored counters)
def raw_event_rows(events):
    """Normalize raw cache events to (ts, type, dict) with integer timestamps."""
    out = []
    for e in events:
        try:
            ts = int(e.get('timestamp'))
        except (TypeError, ValueError):
            continue
        out.append((ts, str(e.get('type')), e))
    out.sort(key=lambda x: x[0])
    return out


def _int(v):
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        return 0


def raw_interval_counts(rows, team_map, lo_excl, hi_incl):
    """Category counts, owned blue/red counts and credited kills for lo < ts <= hi from raw events."""
    c = {}

    def add(k, n=1):
        c[k] = c.get(k, 0) + n
    for ts, typ, e in rows:
        if ts <= lo_excl or ts > hi_incl:
            continue
        if typ == 'CHAMPION_KILL':
            add('champion_kill')
            killer = _int(e.get('killerId'))
            if killer in team_map:
                add('credited_kill_' + ('blue' if team_map[killer] == 100 else 'red'))
        elif typ == 'ELITE_MONSTER_KILL':
            team = _int(e.get('killerTeamId'))
            mt = str(e.get('monsterType', ''))
            sub = str(e.get('monsterSubType', '')).upper()
            if mt == 'DRAGON':
                if sub == 'ELDER_DRAGON':
                    cats = ['elder']
                else:
                    el = sub[:-7] if sub.endswith('_DRAGON') else sub
                    cats = ['dragon', 'dragon_' + (el if el in DRAGON_ELEMENTS else 'OTHER')]
            else:
                cats = [{'BARON_NASHOR': 'baron', 'RIFTHERALD': 'herald', 'HORDE': 'horde', 'ATAKHAN': 'atakhan'}.get(mt, 'monster_other')]
            for cat in cats:
                add(cat)
                if cat in ('baron', 'dragon', 'elder', 'herald', 'horde', 'atakhan') and team in (100, 200):
                    add(f"{cat}_{'blue' if team == 100 else 'red'}")
        elif typ == 'DRAGON_SOUL_GIVEN':
            team = _int(e.get('teamId'))
            if team in (100, 200):
                add('soul_owned')
                add(f"soul_owned_{'blue' if team == 100 else 'red'}")
            else:
                add('soul_teamid0_unassigned')
        elif typ == 'BUILDING_KILL':
            lost = _int(e.get('teamId'))
            cat = 'inhibitor' if e.get('buildingType') == 'INHIBITOR_BUILDING' else 'tower'
            add(cat)
            if lost in (100, 200):
                add(f"{cat}_{'blue' if lost == 200 else 'red'}")
        elif typ == 'TURRET_PLATE_DESTROYED':
            add('plate')
    return c


# ------------------------------------------------------------------ review packet selection
def _h(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def select_packet(match, s, strata, seed):
    """strata: ordered [(name, k, mask)]. Unique match across the packet; priority = list order.

    Returns (cases, report). Inclusion probability conditional on higher-priority selections, treating the hash as a
    uniform random permutation: min(1, k / M_remaining) * 1 / r_match.
    """
    match = np.asarray(match).astype(str)
    s = np.asarray(s).astype(np.int64)
    used = set()
    cases, report = [], []
    for name, k, mask in strata:
        idx = np.flatnonzero(np.asarray(mask, dtype=bool))
        by_match = {}
        for i in idx.tolist():
            by_match.setdefault(match[i], []).append(i)
        eligible_matches = len(by_match)
        remaining = {m: rows for m, rows in by_match.items() if m not in used}
        M = len(remaining)
        ranked = sorted(remaining, key=lambda m: _h(f'lv20260915-packet:{seed}:{name}:match:{m}'))
        chosen = ranked[:k]
        for m in chosen:
            rows = remaining[m]
            i = min(rows, key=lambda j: _h(f'lv20260915-packet:{seed}:{name}:row:{m}:{int(s[j])}'))
            cases.append(dict(stratum=name, row_index=int(i), match=m, s=int(s[i]), eligible_rows_in_match=len(rows),
                              eligible_matches_remaining=M, k=k,
                              inclusion_probability_conditional=min(1.0, k / M) / len(rows) if M else None))
            used.add(m)
        report.append(dict(stratum=name, k=k, eligible_rows=int(len(idx)), eligible_matches=eligible_matches,
                           eligible_matches_remaining=M, selected=len(chosen), shortage=max(0, k - M),
                           match_inclusion_probability=min(1.0, k / M) if M else None))
    return cases, report


def case_order_key(seed, m, s):
    return _h(f'lv20260915-case-order:{seed}:{m}:{int(s)}')


REVIEWER_ALLOWED_FIELDS = ('case_id', 'times', 'scale', 'champions', 'pre_state', 'end_state', 'resources', 'timeline',
                           'terminating_event', 'unavailable')
REVIEWER_FORBIDDEN_TOKENS = ('p_pre', 'p_post', 'delta', 'Δ', 'Y_h', 'Y_A', 'Y_B', 'B_reg', 'B_econ', 'stratum', 'S1_', 'S2_',
                             'S3_', 'S4_', 'winningTeam', 'winner', 'q_pred', 'ridge', 'isotonic', 'probability_A',
                             'KR_', 'NA1_', 'EUW', 'gameId', 'realTimestamp', 'puuid', 'summonerName', 'riotId')
