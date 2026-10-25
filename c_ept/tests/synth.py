"""Synthetic matches for the unit tests (no raw data needed). Everything here is deterministic, module-level and picklable.

``make_record`` builds one ``MatchRecord`` from kills, objectives and extra events; ``records_from_seed`` builds a random one from a
match key (the ``records`` argument of the S1/S2 runners); ``raw_timeline`` / ``raw_detail`` / ``write_raw`` write a record back as
Match-V5 files (with identifier strings planted in them) so that ``reader.read_match`` can be tested end to end. ``kt`` builds a
``units.KillTable`` from a few hand-written rows and ``TEAM`` is the team array that goes with it.

Frame clock: frame 0 is at 0 ms, frame k >= 1 at 60,000 k + 300 ms (a real timeline is a little late: median 292 ms), and the
regular frames end before ``T_ms``. Events have the field names of the raw files; the ones the model does not use are filled
with neutral values.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import re
from pathlib import Path

import numpy as np

from c_ept import config as C
from c_ept.domain import units
from c_ept.data.reader import MatchRecord, champion_name_id

TEAM = np.array([100] * 5 + [200] * 5, dtype=np.int16)     # the (10,) team ids of the hand-built tables: blue 1-5, red 6-10

# Strings that raw_detail / raw_timeline plant in the identifier fields; a record must never contain any of them.
IDENTIFIER_SENTINELS = ('SENTINEL-PUUID', 'SENTINEL-RIOT-ID', 'SENTINEL-SUMMONER')

# (Data Dragon champion key, champion name): the pool the synthetic champions are drawn from.
CHAMPION_POOL = ((266, 'Aatrox'), (103, 'Ahri'), (84, 'Akali'), (12, 'Alistar'), (22, 'Ashe'), (53, 'Blitzcrank'), (51, 'Caitlyn'),
                 (122, 'Darius'), (81, 'Ezreal'), (105, 'Fizz'), (86, 'Garen'), (120, 'Hecarim'), (24, 'Jax'), (222, 'Jinx'),
                 (145, 'Kaisa'), (64, 'LeeSin'), (117, 'Lulu'), (99, 'Lux'), (21, 'MissFortune'), (61, 'Orianna'), (412, 'Thresh'),
                 (23, 'Tryndamere'), (106, 'Volibear'), (238, 'Zed'))
DEFAULT_CHAMPS = (86, 120, 61, 51, 412, 23, 64, 238, 222, 117)     # Garen ... Lulu: top, jungle, mid, bot, support per team
_NAME = dict(CHAMPION_POOL)
_TEAM = [100] * 5 + [200] * 5

# championStats: (value at level 0, increase per level), by raw key
_CS = {'abilityHaste': (0, 0), 'abilityPower': (0, 2), 'armor': (30, 3.5), 'armorPen': (0, 0), 'armorPenPercent': (0, 0),
       'attackDamage': (60, 3), 'attackSpeed': (100, 1.5), 'bonusArmorPenPercent': (0, 0), 'bonusMagicPenPercent': (0, 0),
       'ccReduction': (0, 0), 'cooldownReduction': (0, 0), 'health': (600, 90), 'healthMax': (600, 90), 'healthRegen': (8, 0.4),
       'lifesteal': (0, 0), 'magicPen': (0, 0), 'magicPenPercent': (0, 0), 'magicResist': (32, 1.3), 'movementSpeed': (340, 0),
       'omnivamp': (0, 0), 'physicalVamp': (0, 0), 'power': (300, 40), 'powerMax': (300, 40), 'powerRegen': (8, 0.8), 'spellVamp': (0, 0)}
assert set(_CS) == set(C.CHAMPION_STATS_KEYS)
_CS_BASE = np.array([_CS[k][0] for k in C.CHAMPION_STATS_KEYS], dtype=np.float64)
_CS_STEP = np.array([_CS[k][1] for k in C.CHAMPION_STATS_KEYS], dtype=np.float64)


# ---- kill tables ----------------------------------------------------------------------------------------------------------
def kt(rows):
    """KillTable from rows ``(t, x[, killer, victim, assists, bounty, shutdown])``; y = 0, killer 1, victim 6, no assists, bounty 300."""
    defaults = (1, 6, (), 300, 0)
    full = [tuple(r) + defaults[len(r) - 2:] for r in rows]

    def col(i, dtype):
        return np.array([r[i] for r in full], dtype=dtype)

    x = col(1, float)
    return units.KillTable(t=col(0, np.int64), x=x, y=np.zeros(len(full)), killer=col(2, np.int8), victim=col(3, np.int8),
                           amask=np.array([sum(1 << (a - 1) for a in r[4]) for r in full], dtype=np.int16),
                           bounty=col(5, np.int32), shutdown=col(6, np.int32), pos_missing=np.isnan(x))


# ---- frames ---------------------------------------------------------------------------------------------------------------
def _frame_grid(T_ms: int, n_frames: int | None) -> np.ndarray:
    if n_frames is None:
        n_frames = 1 + max(0, (T_ms - 301) // 60_000)         # frames 0 and 60,000 k + 300 < T_ms
    ts = np.array(([0] + [60_000 * k + 300 for k in range(1, n_frames)])[:n_frames], dtype=np.int64)
    if len(ts) and ts[-1] >= T_ms:
        raise ValueError(f'{n_frames} frames reach {int(ts[-1])} ms, not before the game end {T_ms} ms')
    return ts


def _frame_block(ks, phase: int = 0) -> np.ndarray:
    """(len(ks), 10, 33) frame values for frame indices ``ks``: a smooth path on the map (shifted by ``phase``), growing stats."""
    k = np.asarray(ks, dtype=np.float64)[:, None]                                  # (F, 1)
    i = np.arange(10, dtype=np.float64)[None, :]                                   # (1, 10)
    level = np.minimum(18.0, 1.0 + np.floor(2.0 * k / 3.0)) + 0.0 * i               # (F, 10)
    out = np.empty((k.shape[0], 10, len(C.FRAME_FIELDS)))
    out[..., 0] = 400.0 + np.mod(1_000.0 + 1_237.0 * i + 211.0 * k + phase, 14_000.0)
    out[..., 1] = 400.0 + np.mod(13_000.0 - 997.0 * i + 173.0 * k - phase, 14_000.0)
    out[..., 2] = level
    out[..., 3] = 280.0 * (level - 1.0) + 40.0 * k + i
    out[..., 4] = 400.0 + 35.0 * k + 7.0 * i
    out[..., 5] = 500.0 + 420.0 * k + 29.0 * i
    out[..., 6] = 6.0 * k + i
    out[..., 7] = np.where((i == 1) | (i == 6), 3.0 * np.floor(k / 2.0), 0.0)
    out[..., 8:] = _CS_BASE + _CS_STEP * level[..., None]
    return out


# ---- events ---------------------------------------------------------------------------------------------------------------
def _py(v):
    return v.item() if hasattr(v, 'item') else v


def _put_position(e: dict, x, y) -> None:
    x, y = (None if v is None else _py(v) for v in (x, y))
    if x is None or y is None or np.isnan(x) or np.isnan(y):
        return                                                                       # no position, as in a file that has none
    e['position'] = {'x': x, 'y': y}


def _put_assists(e: dict, assists) -> None:
    if len(assists):                                                                  # the raw files leave the key out when empty
        e['assistingParticipantIds'] = [int(a) for a in assists]


def _kill_event(d: dict) -> dict:
    e = {'type': 'CHAMPION_KILL', 'timestamp': int(d['t']), 'killerId': int(d['killer']), 'victimId': int(d['victim']),
         'bounty': int(d.get('bounty', 300)), 'shutdownBounty': int(d.get('shutdown', 0)), 'killStreakLength': 0,
         'victimDamageDealt': [], 'victimDamageReceived': []}
    _put_position(e, d['x'], d['y'])
    _put_assists(e, d.get('assists', ()))
    return e


def _objective_event(d: dict) -> dict:
    """ELITE_MONSTER_KILL (team = killerTeamId), BUILDING_KILL / TURRET_PLATE_DESTROYED (team = teamId, the owner that lost it)."""
    typ = d['type']
    e = {'type': typ, 'timestamp': int(d['t']), 'killerId': int(d.get('killer', 0))}
    if typ in ('ELITE_MONSTER_KILL', 'BUILDING_KILL'):
        e['bounty'] = 0
    _put_position(e, d['x'], d['y'])
    _put_assists(e, d.get('assists', ()))
    if d.get('team') is not None:
        e['killerTeamId' if typ == 'ELITE_MONSTER_KILL' else 'teamId'] = int(d['team'])
    for arg, field in (('monster', 'monsterType'), ('sub', 'monsterSubType'), ('building', 'buildingType'), ('tower', 'towerType'),
                       ('lane', 'laneType')):
        if d.get(arg) is not None:
            e[field] = d[arg]
    return e


def _runes(champ_id: int, i: int) -> dict:
    """A neutral cache-format runes entry (no rune that switches on a legacy item rule); the junglers (index 1, 6) carry Smite."""
    return {'champion_name_id': champion_name_id(_NAME.get(champ_id, f'Champion{champ_id}')), 'summoner_spell_1_id': 4,
            'summoner_spell_2_id': 11 if i in (1, 6) else 14, 'primary_style_id': 8000, 'sub_style_id': 8200, 'primary_rune_1': 8005,
            'primary_rune_2': 9111, 'primary_rune_3': 9104, 'primary_rune_4': 8014, 'sub_rune_1': 8226, 'sub_rune_2': 8210,
            'stat_perk_offense': 5008, 'stat_perk_flex': 5008, 'stat_perk_defense': 5001}


def make_record(*, T_ms=1_200_000, kills=(), objectives=(), events=(), patch='15.18', n_frames=None, key='SYN_0') -> MatchRecord:
    """One synthetic match.

    kills:      ``dict(t, x, y, killer, victim, assists=(), bounty=300, shutdown=0)`` -> CHAMPION_KILL (x or y None/NaN: no position).
    objectives: ``dict(t, type, x, y, killer=0, assists=(), team=None, monster=None, sub=None, building=None, tower=None, lane=None)``
                -> an event of that ``type`` with the raw field names (monsterType, monsterSubType, buildingType, towerType,
                laneType; ``team`` is killerTeamId for ELITE_MONSTER_KILL, teamId otherwise; None leaves the field out).
    events:     further raw event dicts, copied as they are.
    The events are ordered by time (same ms: kills, objectives, events, in the order given); GAME_END at ``T_ms`` comes last.
    n_frames:   number of regular frames (default: all frames before ``T_ms``); they must end before ``T_ms``.
    """
    T_ms = int(T_ms)
    if T_ms < 1:
        raise ValueError('T_ms must be positive')
    frame_ts = _frame_grid(T_ms, n_frames)
    extra = [copy.deepcopy(dict(e)) for e in events]
    if any(e.get('type') == 'GAME_END' for e in extra):
        raise ValueError('GAME_END is added by make_record')
    evs = [_kill_event(d) for d in kills] + [_objective_event(d) for d in objectives] + extra
    late = [e['timestamp'] for e in evs if e['timestamp'] > T_ms]
    if late:
        raise ValueError(f'event at {min(late)} ms after the game end {T_ms} ms')
    evs.sort(key=lambda e: e['timestamp'])                                            # stable
    evs.append({'type': 'GAME_END', 'timestamp': T_ms, 'winningTeam': 100})
    F = len(frame_ts)
    return MatchRecord(
        key=key, patch=patch, T_ms=T_ms, team=np.array(_TEAM, dtype=np.int16), champ=np.array(DEFAULT_CHAMPS, dtype=np.int32),
        runes_by_pid={str(i + 1): _runes(c, i) for i, c in enumerate(DEFAULT_CHAMPS)}, frame_ts=frame_ts,
        frame_num=_frame_block(range(F)), end_xy=_frame_block([F])[0][:, :2].copy(), events=tuple(evs))


# ---- random matches ---------------------------------------------------------------------------------------------------------
_OBJECTIVE_KINDS = (   # (weight, event type, monster, sub, building, tower)
    (0.30, 'TURRET_PLATE_DESTROYED', None, None, None, None), (0.12, 'BUILDING_KILL', None, None, 'TOWER_BUILDING', 'OUTER_TURRET'),
    (0.05, 'BUILDING_KILL', None, None, 'TOWER_BUILDING', 'INNER_TURRET'), (0.05, 'BUILDING_KILL', None, None, 'INHIBITOR_BUILDING', None),
    (0.17, 'ELITE_MONSTER_KILL', 'DRAGON', 'FIRE_DRAGON', None, None), (0.03, 'ELITE_MONSTER_KILL', 'DRAGON', 'ELDER_DRAGON', None, None),
    (0.10, 'ELITE_MONSTER_KILL', 'HORDE', None, None, None), (0.06, 'ELITE_MONSTER_KILL', 'RIFTHERALD', None, None, None),
    (0.07, 'ELITE_MONSTER_KILL', 'BARON_NASHOR', None, None, None), (0.05, 'ELITE_MONSTER_KILL', 'ATAKHAN', None, None, None))
_LANES = ('TOP_LANE', 'MID_LANE', 'BOT_LANE')


def _team_pids(side: int) -> list[int]:
    return list(range(1 + 5 * side, 6 + 5 * side))                                    # side 0 blue (1-5), side 1 red (6-10)


def _random_kills(rng, T_ms: int) -> list[dict]:
    """Bursts of kills (gaps below 14 s, so many chain into one engagement) at random places, always across the teams."""
    kills = []
    n_bursts = int(rng.integers(4, 28))
    for centre in np.sort(rng.integers(100_000, max(100_001, T_ms - 20_000), n_bursts)):
        cx, cy = rng.uniform(800.0, 14_000.0, 2)
        t = int(centre)
        for j in range(min(8, int(rng.geometric(0.4)))):
            t += int(rng.integers(0, 14_000)) if j else 0
            if t >= T_ms:
                break
            side = int(rng.integers(0, 2))                                            # the side that scores the kill
            execution = rng.random() < 0.03                                           # killer 0: minions, tower, monster
            killer = 0 if execution else int(rng.choice(_team_pids(side)))
            victim = int(rng.choice(_team_pids(1 - side)))
            helpers = [p for p in _team_pids(side) if p != killer]
            assists = tuple(sorted(int(a) for a in rng.choice(helpers, int(rng.integers(0, 4)), replace=False)))
            x, y = np.clip(rng.normal([cx, cy], 700.0), 0.0, 14_875.0)
            kills.append(dict(t=t, x=int(x), y=int(y), killer=killer, victim=victim, assists=assists,
                              bounty=int(rng.choice([150, 300, 450])), shutdown=int(rng.choice([0, 0, 0, 150, 300]))))
    return kills


def _random_objectives(rng, T_ms: int, kills: list[dict]) -> list[dict]:
    """Objective events; most of them soon after a kill and near it (so that some attach to an engagement), a few without position
    or without team."""
    weights = np.array([k[0] for k in _OBJECTIVE_KINDS])
    out = []
    for _ in range(int(rng.integers(8, 45))):
        _, typ, monster, sub, building, tower = _OBJECTIVE_KINDS[int(rng.choice(len(_OBJECTIVE_KINDS), p=weights / weights.sum()))]
        taker = int(rng.integers(0, 2))                                               # side that gets the objective
        if kills and rng.random() < 0.6:
            kd = kills[int(rng.integers(0, len(kills)))]
            t = kd['t'] + int(rng.integers(0, 20_000))
            x, y = np.clip(rng.normal([kd['x'], kd['y']], 600.0), 0.0, 14_875.0)
        else:
            t = int(rng.integers(60_000, T_ms))
            x, y = rng.uniform(0.0, 14_875.0, 2)
        killer = 0 if (typ != 'ELITE_MONSTER_KILL' and rng.random() < 0.2) else int(rng.choice(_team_pids(taker)))
        helpers = [p for p in _team_pids(taker) if p != killer]
        assists = tuple(sorted(int(a) for a in rng.choice(helpers, int(rng.integers(0, 3)), replace=False)))
        team = 100 + 100 * taker if typ == 'ELITE_MONSTER_KILL' else 200 - 100 * taker      # owner that lost the building / plate
        d = dict(t=min(t, T_ms - 1), type=typ, x=int(x), y=int(y), killer=killer, assists=assists, team=team, monster=monster,
                 sub=sub, building=building, tower=tower)
        if typ != 'ELITE_MONSTER_KILL':
            d['lane'] = _LANES[int(rng.integers(0, 3))]
        if rng.random() < 0.03:
            d['x'] = d['y'] = None                                                    # no position
        if rng.random() < 0.02:
            d['team'] = None                                                          # unknown team
        out.append(d)
    return out


def records_from_seed(key: str, patch: str) -> MatchRecord:
    """A random synthetic match; the digits of ``key`` are the seed (same key and patch: the same record). Kills cross the teams."""
    digits = re.sub(r'\D', '', key)
    rng = np.random.default_rng(int(digits) if digits else 0)
    T_ms = int(rng.integers(900_000, 2_100_000))
    kills = _random_kills(rng, T_ms)
    objectives = _random_objectives(rng, T_ms, kills)
    rec = make_record(T_ms=T_ms, kills=kills, objectives=objectives, patch=patch, key=key)
    picks = rng.choice(len(CHAMPION_POOL), 10, replace=False)
    champs = [CHAMPION_POOL[int(j)][0] for j in picks]
    phase = int(rng.integers(0, 14_000))
    F = len(rec.frame_ts)
    return dataclasses.replace(
        rec, champ=np.array(champs, dtype=np.int32), runes_by_pid={str(i + 1): _runes(c, i) for i, c in enumerate(champs)},
        frame_num=_frame_block(range(F), phase), end_xy=_frame_block([F], phase)[0][:, :2].copy())


# ---- record -> raw files ------------------------------------------------------------------------------------------------------
def assert_valid(rec: MatchRecord) -> None:
    """Check a MatchRecord, from ``read_match`` or from this module; raises AssertionError.

    Everything asserted here is guaranteed by ``read_match`` (it raises MatchRejected or MatchFormatError otherwise): teams 100 x 5
    then 200 x 5, a 14-entry runes dict for the pids '1'-'10', the first frame at 0 ms, strictly increasing frame times,
    ``frame_ts[-1] < T_ms``, events ordered by time with the GAME_END at ``T_ms`` as the last event, the array shapes and dtypes.
    ``read_match`` also checks in the file that GAME_END is in the last frame and that every champion id is an integer; a record
    does not show that. (Its arrays are read-only too; tested separately.)
    Not asserted, because it holds only for synth records: the frame grid 0, 60,300, 120,300, ... ms (real frames are about 60 s
    apart, with jitter), kills across the teams, and a record without any regular frame (``n_frames=0``).
    """
    F = len(rec.frame_ts)
    assert rec.team.shape == (10,) and rec.team.dtype == np.int16 and list(rec.team) == _TEAM
    assert rec.champ.shape == (10,) and rec.champ.dtype == np.int32
    assert rec.frame_ts.ndim == 1 and rec.frame_ts.dtype == np.int64 and np.all(np.diff(rec.frame_ts) > 0)
    assert F == 0 or (rec.frame_ts[0] == 0 and rec.frame_ts[-1] < rec.T_ms)
    assert rec.frame_num.shape == (F, 10, len(C.FRAME_FIELDS)) and rec.frame_num.dtype == np.float64
    assert rec.end_xy.shape == (10, 2) and rec.end_xy.dtype == np.float64
    ts = [e['timestamp'] for e in rec.events]
    assert ts == sorted(ts) and rec.events[-1]['type'] == 'GAME_END' and rec.events[-1]['timestamp'] == rec.T_ms
    assert sorted(rec.runes_by_pid) == sorted(str(i) for i in range(1, 11))
    assert all(len(v) == 14 for v in rec.runes_by_pid.values())


def _number(v: float):
    return int(v) if float(v).is_integer() else float(v)


def _participant_frame(pid: int, row: np.ndarray) -> dict | None:
    """Raw participantFrames entry from a row of frame values (NaN: left out); None when every value is NaN."""
    pf, pos, cs = {'participantId': pid}, {}, {}
    for name, v in zip(C.FRAME_FIELDS, row):
        if np.isnan(v):
            continue
        if name in ('x', 'y'):
            pos[name] = _number(v)
        elif name.startswith('cs_'):
            cs[name[3:]] = _number(v)
        else:
            pf[name] = _number(v)
    if pos:
        pf['position'] = pos
    if cs:
        pf['championStats'] = cs
    return pf if len(pf) > 1 else None


def _participant_frames(rows: np.ndarray) -> dict:
    return {str(i + 1): pf for i in range(10) if (pf := _participant_frame(i + 1, rows[i])) is not None}


def raw_timeline(rec: MatchRecord) -> dict:
    """The record as a Match-V5 timeline: a frame holds the events after the previous frame up to its own time; the last frame is
    the game-end frame (end positions, the events after the last regular frame, GAME_END). Identifier sentinels are planted."""
    bounds = [int(t) for t in rec.frame_ts] + [rec.T_ms]
    frames = [{'timestamp': b, 'participantFrames': {}, 'events': []} for b in bounds]
    for k in range(len(rec.frame_ts)):
        frames[k]['participantFrames'] = _participant_frames(rec.frame_num[k])
    end_rows = np.full((10, len(C.FRAME_FIELDS)), np.nan)
    end_rows[:, :2] = rec.end_xy
    frames[-1]['participantFrames'] = _participant_frames(end_rows)
    for e in rec.events:
        frames[int(np.searchsorted(bounds, e['timestamp'], side='left'))]['events'].append(copy.deepcopy(e))
    return {'metadata': {'dataVersion': '2', 'matchId': rec.key, 'participants': [f'{IDENTIFIER_SENTINELS[0]}-{i}' for i in range(10)]},
            'info': {'endOfGameResult': 'GameComplete', 'frameInterval': 60_000, 'gameId': 0, 'frames': frames,
                     'participants': [{'participantId': i + 1, 'puuid': f'{IDENTIFIER_SENTINELS[0]}-{i}'} for i in range(10)]}}


def raw_detail(rec: MatchRecord) -> dict:
    """The record as a Match-V5 detail file. The champion name is taken from CHAMPION_POOL by champion key, so a record whose
    runes_by_pid[...]['champion_name_id'] came from another name does not round-trip."""
    parts = []
    for i in range(10):
        r = rec.runes_by_pid[str(i + 1)]
        champ_id = int(rec.champ[i])
        parts.append({
            'participantId': i + 1, 'teamId': int(rec.team[i]), 'championId': champ_id,
            'championName': _NAME.get(champ_id, f'Champion{champ_id}'), 'summoner1Id': r['summoner_spell_1_id'],
            'summoner2Id': r['summoner_spell_2_id'], 'puuid': f'{IDENTIFIER_SENTINELS[0]}-{i}',
            'riotIdGameName': f'{IDENTIFIER_SENTINELS[1]}-{i}', 'riotIdTagline': 'XYZ', 'summonerName': f'{IDENTIFIER_SENTINELS[2]}-{i}',
            'summonerId': f'{IDENTIFIER_SENTINELS[2]}-ID-{i}',
            'perks': {'statPerks': {'defense': r['stat_perk_defense'], 'flex': r['stat_perk_flex'], 'offense': r['stat_perk_offense']},
                      'styles': [{'description': 'primaryStyle', 'style': r['primary_style_id'],
                                  'selections': [{'perk': r[f'primary_rune_{j}'], 'var1': 0, 'var2': 0, 'var3': 0} for j in range(1, 5)]},
                                 {'description': 'subStyle', 'style': r['sub_style_id'],
                                  'selections': [{'perk': r[f'sub_rune_{j}'], 'var1': 0, 'var2': 0, 'var3': 0} for j in range(1, 3)]}]}})
    return {'metadata': {'matchId': rec.key, 'participants': [p['puuid'] for p in parts]},
            'info': {'gameDuration': rec.T_ms // 1000, 'gameVersion': f'{rec.patch}.1.1', 'mapId': 11, 'participants': parts, 'teams': []}}


def write_raw(root, patch: str, key: str, timeline: dict | None, detail: dict | None) -> None:
    """Write the two files of a match under ``<root>/<patch>/raw/<region>/{timeline,detail}/``; None leaves a file out."""
    base = Path(root) / patch / 'raw' / C.RAW_REGION[patch]
    for kind, obj in (('timeline', timeline), ('detail', detail)):
        if obj is not None:
            (base / kind).mkdir(parents=True, exist_ok=True)
            (base / kind / f'{key}.json').write_text(json.dumps(obj), encoding='utf-8')


# ---- role function for synthetic match lists ----------------------------------------------------------------------------------------
S1_MACROS = ('FIT', 'STOP', 'DEV', 'CAL', 'AUDIT')       # the macro roles S1 extracts, in the order the test lists use


def roles_of(matches, flip=()):
    """A stand-in for ``cohort.role_of`` for synthetic keys, which the cohort split does not know (the real split gives a synthetic key a
    role of its own, not the one a test list names). It answers, for each (patch, key) of ``matches``, the macro role that the list itself
    carries, and the next of the five S1 macros for the keys in ``flip``. So ``run_s1.verify_roles`` really asks and compares in every
    synthetic run (the argument ``role_of`` of ``run_s1`` / ``build_store`` / ``golden``, as ``records`` stands in for the reader); the
    refusals are tested with ``flip`` and with repeated entries (test_run_s1), the real split with real keys."""
    table = {(patch, key): macro for patch, key, macro in matches}
    flip = set(flip)

    def role_of(key, patch):
        macro = table[(patch, key)]
        return 'SYNTHETIC_SPLIT_ROLE', (S1_MACROS[(S1_MACROS.index(macro) + 1) % len(S1_MACROS)] if key in flip else macro)
    return role_of


# ---- sample processes for the E6 tests: (pid, name, command line) ----------------------------------------------------------------
# The shapes are those of this machine: the Strata server is `python.exe D:\Strata\serve\server.py ...` (start_strata.ps1) and the idle
# MCP bridges of other sessions are `python.exe ...\strata-worker\strata_worker_mcp.py`. The paths are made up (user name "x").
STRATA_SERVER = (4101, 'python.exe', [r'D:\Strata\.venv\Scripts\python.exe', r'D:\Strata\serve\server.py', '--engine', 'strata', '--config',
                                      r'D:\Strata\strata-iq2_xs.json', '--host', '127.0.0.1', '--port', '8080'])
STRATA_BRIDGE = (4102, 'python.exe', [r'C:\Users\x\strata-worker\.venv\Scripts\python.exe',
                                      r'C:\Users\x\strata-worker\strata_worker_mcp.py'])
LLAMA_SERVER = (4103, 'llama-server.exe', ['llama-server.exe', '-m', 'model.gguf', '--port', '8081'])
OLLAMA_APP = (4104, 'ollama.exe', ['ollama.exe', 'serve'])
UNRELATED_PYTHON = (4105, 'python.exe', ['python.exe', '-m', 'pytest', 'c_ept/tests/test_runlog.py', '-q'])
