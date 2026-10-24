"""P3 pure rules: B-rule endpoints, tied terminal reasons, labels and raw-event attribution.

No I/O, no model code and no worktree imports, so the rules can be unit-tested on synthetic data and
imported by the P3 runner without side effects. EXPLORATORY candidate-label definitions, not ground truth.

Endpoint rule (operational B rule of scripts/event_boundary_cif.py, OUR event-based design):
    next_kill  = first raw CHAMPION_KILL timestamp strictly greater than L (absent -> +inf)
    next_start = stored next eligible engagement start; the parent exposure table stores the game end
                 as a sentinel when no later engagement exists (next_start >= end -> absent -> +inf)
    endpoint_h = min(L + 1000*h, next_kill - 1, next_start - 1, game_end - 1)     [integer ms]
The state at the endpoint includes every observation with timestamp <= endpoint (closed inclusion).
An objective acquisition is never a terminal event.
"""
from __future__ import annotations

import math

HORIZONS_S = (60, 90, 120)
PRIMARY_HORIZON_S = 90
REASONS = ('horizon', 'next_kill', 'next_engagement_start', 'game_end')
NEAR_ZERO_BANDS = (0.005, 0.01, 0.02)
INF = math.inf

DRAGON_ELEMENTS = ('AIR', 'EARTH', 'FIRE', 'WATER', 'HEXTECH', 'CHEMTECH', 'OTHER')
# Raw objective / structure categories (both teams; ownership kept separately where recorded).
OBJECTIVE_CATEGORIES = (('baron', 'dragon', 'elder', 'herald', 'horde', 'atakhan', 'soul_owned', 'soul_teamid0_unassigned')
                        + tuple(f'dragon_{d}' for d in DRAGON_ELEMENTS))
STRUCTURE_CATEGORIES = ('tower', 'inhibitor', 'plate')
OWNED_CATEGORIES = ('baron', 'dragon', 'elder', 'herald', 'horde', 'atakhan', 'soul_owned', 'tower', 'inhibitor')
EVENT_CATEGORIES = ('champion_kill',) + OBJECTIVE_CATEGORIES + STRUCTURE_CATEGORIES + ('monster_other',)
COUNT_KEYS = EVENT_CATEGORIES + tuple(f'{c}_{t}' for c in OWNED_CATEGORIES for t in ('blue', 'red'))
NON_KILL_PREDICTOR_TYPES = ('ELITE_MONSTER_KILL', 'DRAGON_SOUL_GIVEN', 'BUILDING_KILL', 'TURRET_PLATE_DESTROYED')


def _int_ms(value, what):
    if isinstance(value, bool):
        raise TypeError(f'{what} must be integer milliseconds')
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    try:
        import numpy as np
        if isinstance(value, np.integer):
            return int(value)
    except ImportError:  # pragma: no cover
        pass
    raise TypeError(f'{what} must be integer milliseconds, got {value!r}')


def next_kill_after(kill_times_sorted, last_kill_ms):
    """First kill timestamp strictly greater than L; +inf if none. kill_times_sorted must be ascending."""
    from bisect import bisect_right
    i = bisect_right(kill_times_sorted, last_kill_ms)
    return kill_times_sorted[i] if i < len(kill_times_sorted) else INF


def effective_next_start(next_start_stored, game_end):
    """Original sentinel semantics (event_boundary_cif.py): next_start >= end means no later engagement."""
    return next_start_stored if next_start_stored < game_end else INF


def endpoint_candidates(L, h_s, next_kill, next_start_eff, game_end):
    """Candidate terminal times (ms); +inf where the event is absent."""
    L = _int_ms(L, 'L')
    return {'horizon': L + 1000 * int(h_s),
            'next_kill': next_kill - 1 if next_kill != INF else INF,
            'next_engagement_start': next_start_eff - 1 if next_start_eff != INF else INF,
            'game_end': game_end - 1 if game_end != INF else INF}


def endpoint_rule(L, h_s, next_kill, next_start_eff, game_end):
    """Returns (endpoint_ms, tied_reasons tuple in REASONS order, candidates dict)."""
    cand = endpoint_candidates(L, h_s, next_kill, next_start_eff, game_end)
    e = min(cand.values())
    if e == INF:
        raise ValueError('endpoint is unbounded')
    e = _int_ms(e, 'endpoint')
    reasons = tuple(r for r in REASONS if cand[r] == e)
    return e, reasons, cand


def endpoint_validity(endpoint, L, q_pre, support_start, last_frame):
    """Validity flags; every flag must be True for a usable row (failures block success, never dropped)."""
    return {'endpoint_ge_L': endpoint >= L, 'endpoint_gt_q_pre': endpoint > q_pre,
            'endpoint_le_last_frame': endpoint <= last_frame, 'q_pre_ge_support': q_pre >= support_start,
            'q_pre_le_last_frame': q_pre <= last_frame}


def label_from_delta(delta):
    """Y = 1(delta > 0). Exact zero is non-improvement (Y = 0), reported separately."""
    if not math.isfinite(delta):
        raise ValueError('non-finite delta')
    return int(delta > 0)


def near_zero_flags(delta):
    """Sensitivity descriptions |delta| <= band; NOT label thresholds."""
    return {f'abs_delta_le_{b}': abs(delta) <= b for b in NEAR_ZERO_BANDS}


def event_category(e):
    """(category, team) for raw events; team = acquiring team (100/200) or 0 when not recorded."""
    typ = e.get('type')
    if typ == 'CHAMPION_KILL':
        return 'champion_kill', 0
    if typ == 'ELITE_MONSTER_KILL':
        team = int(e.get('killerTeamId', 0) or 0)
        monster, sub = e.get('monsterType', ''), str(e.get('monsterSubType', '')).upper()
        if monster == 'DRAGON':
            if sub == 'ELDER_DRAGON':
                return 'elder', team
            return 'dragon', team
        return {'BARON_NASHOR': 'baron', 'RIFTHERALD': 'herald', 'HORDE': 'horde', 'ATAKHAN': 'atakhan'}.get(monster, 'monster_other'), team
    if typ == 'DRAGON_SOUL_GIVEN':
        team = int(e.get('teamId', 0) or 0)
        return ('soul_owned' if team in (100, 200) else 'soul_teamid0_unassigned'), (team if team in (100, 200) else 0)
    if typ == 'BUILDING_KILL':
        lost = int(e.get('teamId', 0) or 0)
        return ('inhibitor' if e.get('buildingType') == 'INHIBITOR_BUILDING' else 'tower'), (300 - lost if lost in (100, 200) else 0)
    if typ == 'TURRET_PLATE_DESTROYED':
        lost = int(e.get('teamId', 0) or 0)
        return 'plate', (300 - lost if lost in (100, 200) else 0)
    return None, 0


def dragon_element(e):
    sub = str(e.get('monsterSubType', '')).upper()
    kind = sub[:-7] if sub.endswith('_DRAGON') else sub
    return kind if kind in DRAGON_ELEMENTS else 'OTHER'


def categorize(e):
    """All category labels an event contributes to (dragon also contributes its element)."""
    cat, team = event_category(e)
    if cat is None:
        return [], team
    cats = [cat]
    if cat == 'dragon':
        cats.append('dragon_' + dragon_element(e))
    return cats, team


def count_events(events_sorted, lo_exclusive, hi_inclusive):
    """Counts per category (and owned blue/red) for lo < timestamp <= hi. events_sorted: [(ts, cats, team)]."""
    out = {}
    for ts, cats, team in events_sorted:
        if ts <= lo_exclusive:
            continue
        if ts > hi_inclusive:
            break
        for c in cats:
            out[c] = out.get(c, 0) + 1
            if c in OWNED_CATEGORIES and team in (100, 200):
                k = f"{c}_{'blue' if team == 100 else 'red'}"
                out[k] = out.get(k, 0) + 1
    return out
