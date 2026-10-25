"""Toy S1 and S2 stores, toy numbers and the numbers of math-control note 03, shared by the tests of stage S3 (ruling R1).

Nothing here reads a run folder, the raw files or the catalog; every store is written with ``store_core.Store`` under a pytest folder and
holds only the arrays the S3 code reads, plus ``matches/key`` with sentinel text, which that code must never read. Test modules import
what they need from here (``toy_s1`` and ``toy_s2`` are pytest fixtures).

A toy match is a ``ToyMatch``: a role, the gold lead of the blue team at every regular frame, the game end, and the objectives and kills in
S1 order. ``build_toy_s1`` turns a list of them into S1 arrays with the encodings of ``run_s1`` (kills/killer 0 = an execution, kills/amask
one bit per assisting participant, objectives/sub -1 outside the towers, objectives/taker -1 = unknown); ``build_toy_s2`` adds an S2
store whose frame-clock views carry a known pattern in ``x/summary``; ``toy_record`` turns a standard-team toy match into a ``MatchRecord``
so that a test can send the same matches through the real S1 extraction and compare.

The default frame grid is ``synth``'s: frame 0 at 0 ms, frame k >= 1 at 60,000 k + 300 ms. The toy matches (TOY_MATCHES; the number is
the S1 match row) are:

    match 0  FIT   12 frames, game end 700,000 ms: a red dragon (minute 7), objectives at the first and last minutes, a kill without a
                   champion, kills on frame times; its last frame row has no next regular frame
    match 1  FIT   11 frames, game end 630,000 ms: a dragon, a plate (same team) and an inhibitor (other team) in minute 6, two plates and
                   an atakhan in minute 8, unknown objectives; its last frame row has a next regular frame
    match 2  STOP   9 frames, game end 560,000 ms: teams interleaved (blue = participants 1, 3, 5, 7, 9)
    match 3  DEV   10 frames, game end 640,000 ms
    match 4  FIT    2 frames, game end 100,000 ms: no frame row at all
    match 5  FIT    6 frames, game end 420,000 ms: no objective, no kill

Every ``TOY_*`` number of the toy matches below was worked out by hand from the lists above it (the comments show the working); the tests
also compare the code with a second, brute-force implementation written from the text of the rules.

The section after that holds what the estimators of ``flowprice`` (stage S3 Task 2) are tested on: synthetic ``EventTable`` and
``WindowTable`` objects with a known truth (no store), the hand-built event table of note 03 section A, and ``planted_baron_matches``,
toy matches whose baron has a planted effect on the lead, so that the same truth can be recovered through the S1 and S2 stores.

The last section serves the labels of ``c_ept.value`` (stage S3 Task 4). ``toy_s1`` above has no units, so these toys go through the real
S1 extraction: ``VALUE_TOY_MATCHES`` are synthetic ``MatchRecord`` objects (kills with positions and assisters, objectives with credit,
LEVEL_UP events) sent through ``run_s1.build_store``, which gives the units, the attribution and conversion of the objectives, the onsets
and the frame rows exactly as S1 makes them. ``build_value_s1`` / the fixture ``toy_value_s1`` is that store, ``TOY_LEVELS`` the victim
levels of its kills, ``TOY_PRICES`` / ``toy_prices`` and ``toy_rules`` the inputs of ``value.unit_values``, and the ``TOY_*`` numbers of
that section were worked out by hand from the lists (the working is in the comments). ``random_value_matches`` makes random matches
with LEVEL_UP events for the brute-force comparisons.

The section at the very end ("the feature arrays of the toy S2", stage S3 Task 5) serves ``c_ept.features``: ``build_toy_s2(...,
features=True)`` (the ``toy_s2`` fixture) adds ``x/players_num`` and ``x/geo_pp`` and, in the frame-clock views, the game time, the
"is up" flags and two constant summary columns; ``TOY_VIEWS`` are the frame-clock views of the toy, ``TOY_NAN_VIEWS`` the views whose
cross-team geodesic block is empty, and ``build_feature_store`` writes a store of explicit arrays for the edge cases.

The "heads world" at the very end (stage S3 Task 6, the B0 heads) is a small random S1, S2 and label store with the real links between the
tables (``build_heads_world``, the fixture ``heads_world``); it is the training and scoring data of the head tests. ``HeadsOracle`` gives
the plain-loop answers about it (which view, which onsets, which units) that the row tests of B0 and B1 compare the code with (Task 7).
After it (stage S3 Task 7, B2) come the "onset toy", a tiny S1 with eleven frame rows and the bins of their onsets written out by hand
(``build_onset_toy_s1``, the fixture ``onset_toy_s1``, the ``TOY_*`` numbers of that section), for the target of the next-minute classifier.

The last section (stage S3 Task 9, ``c_ept.compose``) holds the "composer toys": three head sets that implement the ``ComposerHeads``
protocol with laws that are written out by hand (``ToyHeads``: the heads of note 03 section E; ``KOToyHeads``: the same with an object and a
conversion head that read K and O, the B1-like case; ``RichToyHeads``: smooth laws that differ in every argument, for the brute-force
oracle), the row tables of ``TOY_ROW_SPECS`` and the builders ``toy_row_table``, ``toy_tier_model`` and ``toy_composer``. Four probes
serve the fix round of Task 9: ``ProcessProbeHeads`` (where was a block computed), ``CountingToyHeads`` and ``CountingTier`` (what did the
calling process compute), ``FileLogToyHeads`` and ``EventToyHeads`` (which rows were computed, by workers or here, and in what order).
"""
# Ref: docs/superpowers/notes/2026-10-21-math-control-03.md sections A-G (the TOY_* numbers of the second half of this file; the section is
#      named at each constant) and docs/superpowers/plans/2026-10-21-c-ept-s3.md Task 1 (the toy matches and their constants).
# Ref: docs/superpowers/specs/2026-10-20-c-ept-design.md section 8.1 (D7: event minute, relative minutes -4 .. +2, window rows).
from __future__ import annotations

import collections
import dataclasses
import gc
import math
import os
import re
import shutil
import types
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from c_ept import config as C
from c_ept.valuation import flowprice
from c_ept.data import frame_rows as frame_rows_mod
from c_ept.data.store_core import Store
from c_ept.tests import synth

BLUE, RED, NOBODY = 0, 1, -1                       # objectives/taker
TEAM_BLUE, TEAM_RED = 100, 200
BLUE5, RED5 = (TEAM_BLUE,) * 5, (TEAM_RED,) * 5
TOTAL_GOLD = C.FRAME_FIELDS.index('totalGold')


# ---- the toy matches ------------------------------------------------------------------------------------------------------------
# group name -> (index in config.O_KINDS, tower tier or -1): written out here, not read from c_ept.flowprice
KIND_SUB = {'dragon': (0, -1), 'elder': (1, -1), 'grubs': (2, -1), 'herald': (3, -1), 'baron': (4, -1), 'atakhan': (5, -1),
            'tower_outer': (6, 0), 'tower_inner': (6, 1), 'tower_base': (6, 2), 'tower_nexus': (6, 3), 'inhibitor': (7, -1),
            'plate': (8, -1),
            'unknown_kind': (-1, -1),            # another monster or building: S1 stores kind -1
            'tower_unknown_tier': (6, -1)}       # a tower whose towerType the file does not name: sub -1
GROUP_NAMES = tuple(name for name in KIND_SUB if name not in ('unknown_kind', 'tower_unknown_tier'))      # the 12 groups, in order
if not all(C.O_KINDS[kind] == ('tower' if name.startswith('tower_') else name) for name, (kind, _) in KIND_SUB.items() if kind >= 0):
    raise RuntimeError('the kind indices of KIND_SUB do not follow config.O_KINDS')


def obj(t: int, name: str, taker: int) -> tuple:
    """An objective of a toy match: (time ms, group name or 'unknown_kind' / 'tower_unknown_tier', BLUE | RED | NOBODY)."""
    if name not in KIND_SUB:
        raise ValueError(f'unknown objective name {name!r}')
    return int(t), name, int(taker)


def kill(t: int, killer: int, bounty: int = 300, shutdown: int = 0, assisters=()) -> tuple:
    """A CHAMPION_KILL of a toy match: (time ms, killer participant 1-10 or 0 for an execution, bounty, shutdown bounty, assisters)."""
    return int(t), int(killer), int(bounty), int(shutdown), tuple(int(a) for a in assisters)


def frame_times(n_frames: int) -> tuple:
    """The default frame grid of the toy matches (that of ``synth``): 0, 60,300, 120,300, ... ms."""
    return tuple(([0] + [60_000 * k + 300 for k in range(1, n_frames)])[:n_frames])


@dataclass(frozen=True)
class ToyMatch:
    """One toy match; see the module text. ``lead[k]`` (a multiple of 5) is the gold of the blue team minus that of the red team at
    regular frame k; the objectives and kills are in time order, as S1 stores them."""
    role: str
    lead: tuple
    T_ms: int
    objectives: tuple = ()
    kills: tuple = ()
    team: tuple = BLUE5 + RED5

    @property
    def ts(self) -> tuple:
        return frame_times(len(self.lead))


# ---- match 0 (FIT) ----------------------------------------------------------------------------------------------------------------
# minute j of an event covers (ts[j-1], ts[j]]; ts = 0, 60,300, 120,300, 180,300, 240,300, 300,300, 360,300, 420,300, 480,300, 540,300,
# 600,300, 660,300 for j = 0 .. 11; j = 12 is the time after the last regular frame.
# gold lead by frame; its changes by minute are 100, 150, 50, 200, 100, 100, 100, -300, -200, -100, 50 for j = 1 .. 11
LEAD_M0 = (0, 100, 250, 300, 500, 600, 700, 800, 500, 300, 200, 250)
OBJECTIVES_M0 = (
    obj(200_000, 'tower_outer', BLUE),         # j = 4: 180,300 < t <= 240,300; too early for a complete window (4 - 5 < 0)
    obj(250_000, 'baron', BLUE),               # j = 5: the first complete window (5 - 5 = 0)
    obj(300_300, 'plate', BLUE),               # j = 5: exactly on the frame time 300,300 (the frame time belongs to the minute it ends)
    obj(300_301, 'grubs', RED),                # j = 6: one ms later
    obj(350_000, 'unknown_kind', BLUE),        # no kind: left out and counted
    obj(400_000, 'dragon', RED),               # j = 7: the dragon taken by red
    obj(401_000, 'tower_inner', BLUE),         # j = 7: same minute, the other team
    obj(450_000, 'inhibitor', NOBODY),         # no taker: left out and counted
    obj(455_000, 'tower_unknown_tier', BLUE),  # a tower without a tier: no group, left out and counted
    obj(540_000, 'elder', BLUE),               # j = 9: k + 2 = 11 is the last regular frame, still complete
    obj(600_000, 'herald', RED),               # j = 10: k + 2 = 12 is past the last regular frame
    obj(690_000, 'tower_nexus', BLUE),         # j = 12: after the last regular frame
)
KILLS_M0 = (
    kill(100_000, 5, 300, 0, (4,)),            # j = 2:  blue,  300 gold, 1 assist
    kill(150_000, 1, 300, 0, (2, 3)),          # j = 3:  blue,  300 gold, 2 assists
    kill(270_000, 7, 300, 150, (6, 8, 9)),     # j = 5:  red,   450 gold, 3 assists
    kill(330_000, 2, 450, 0),                  # j = 6:  blue,  450 gold
    kill(390_000, 0, 300, 0, (3,)),            # j = 7:  an execution (no champion killer): pays no team, its assister neither
    kill(395_000, 6, 300, 0, (7,)),            # j = 7:  red,   300 gold, 1 assist
    kill(420_300, 3, 150, 0),                  # j = 7:  blue,  150 gold, exactly on the frame time 420,300
    kill(420_301, 8, 300, 300, (6, 7, 9, 10)),  # j = 8: red,   600 gold, 4 assists
    kill(500_000, 4, 300, 100, (1, 2)),        # j = 9:  blue,  400 gold, 2 assists
    kill(650_000, 9, 300, 0),                  # j = 11: red,   300 gold
    kill(680_000, 1, 300, 0),                  # j = 12: after the last regular frame
)
# ---- match 1 (FIT) ----------------------------------------------------------------------------------------------------------------
LEAD_M1 = (0, 50, 100, 200, 250, 300, 500, 450, 400, 600, 650)       # changes 50, 50, 100, 50, 50, 200, -50, -50, 200, 50 for j = 1 .. 10
OBJECTIVES_M1 = (
    obj(170_000, 'tower_base', RED),           # j = 3: too early
    obj(250_000, 'plate', NOBODY),             # no taker: left out and counted
    obj(310_000, 'dragon', BLUE),              # j = 6
    obj(330_000, 'unknown_kind', BLUE),        # no kind: left out and counted
    obj(350_000, 'plate', BLUE),               # j = 6: same team and minute as the dragon
    obj(355_000, 'inhibitor', RED),            # j = 6: the other team
    obj(420_300, 'baron', RED),                # j = 7: on the frame time
    obj(440_000, 'plate', BLUE),               # j = 8: two plates of one team in one minute
    obj(445_000, 'plate', BLUE),               # j = 8
    obj(480_300, 'atakhan', BLUE),             # j = 8: on the frame time; k + 2 = 10 is the last regular frame, still complete
    obj(481_000, 'grubs', BLUE),               # j = 9: k + 2 = 11 is past the last regular frame
    obj(590_000, 'tower_outer', RED),          # j = 10
    obj(610_000, 'elder', BLUE),               # j = 11: after the last regular frame
)
KILLS_M1 = (
    kill(200_000, 2, 300, 0, (1,)),            # j = 4:  blue,  300 gold, 1 assist
    kill(300_300, 0, 300, 0, (2,)),            # j = 5:  an execution
    kill(310_500, 8, 300, 150),                # j = 6:  red,   450 gold
    kill(430_000, 4, 450, 0, (1, 2, 3)),       # j = 8:  blue,  450 gold, 3 assists
    kill(520_000, 7, 300, 0, (6,)),            # j = 9:  red,   300 gold, 1 assist
)
# ---- matches 2-5 --------------------------------------------------------------------------------------------------------------------
LEAD_M2 = (0, 10, 30, 30, 60, 100, 100, 150, 200)
TEAM_M2 = (100, 200) * 5                                         # blue = participants 1, 3, 5, 7, 9
OBJECTIVES_M2 = (obj(250_000, 'dragon', BLUE), obj(330_000, 'baron', RED))
KILLS_M2 = (kill(200_000, 1, 300, 0, (3,)), kill(260_000, 2, 300, 100, (4, 6)), kill(270_000, 5, 450))
LEAD_M3 = (0, 5, 15, 40, 40, 90, 140, 130, 200, 260)
OBJECTIVES_M3 = (obj(310_000, 'grubs', BLUE), obj(480_300, 'tower_inner', RED), obj(600_000, 'elder', BLUE))
KILLS_M3 = (kill(220_000, 3, 300, 0, (1, 2)), kill(400_000, 8, 300, 150))
LEAD_M4 = (0, 50)
OBJECTIVES_M4 = (obj(50_000, 'tower_outer', BLUE), obj(90_000, 'inhibitor', RED))
KILLS_M4 = (kill(30_000, 1, 300, 0),)
LEAD_M5 = (0, 20, 40, 60, 80, 100)

TOY_MATCHES = (
    ToyMatch('FIT', LEAD_M0, 700_000, OBJECTIVES_M0, KILLS_M0),
    ToyMatch('FIT', LEAD_M1, 630_000, OBJECTIVES_M1, KILLS_M1),
    ToyMatch('STOP', LEAD_M2, 560_000, OBJECTIVES_M2, KILLS_M2, team=TEAM_M2),
    ToyMatch('DEV', LEAD_M3, 640_000, OBJECTIVES_M3, KILLS_M3),
    ToyMatch('FIT', LEAD_M4, 100_000, OBJECTIVES_M4, KILLS_M4),
    ToyMatch('FIT', LEAD_M5, 420_000),
)


# ---- the toy S1 store -----------------------------------------------------------------------------------------------------------
def toy_gold(spec: ToyMatch) -> np.ndarray:
    """(F, 10) totalGold of every participant at every regular frame of a toy match, such that the sum over the blue team minus the sum over
    the red team is ``spec.lead[k]``: each red player holds 1,000 + 100 k + 10 j (j = 0 .. 4 inside the team) and each blue player the same
    plus lead / 5. Exact in float32."""
    blue = [i for i in range(10) if spec.team[i] == TEAM_BLUE]
    red = [i for i in range(10) if spec.team[i] == TEAM_RED]
    if len(blue) != 5 or len(red) != 5 or len(blue) + len(red) != 10:
        raise ValueError('a toy team array holds five blue and five red ids')
    if any(v % 5 for v in spec.lead):
        raise ValueError('a toy gold lead is a multiple of 5')
    gold = np.zeros((len(spec.lead), 10))
    for k, lead in enumerate(spec.lead):
        base = 1_000 + 100 * k
        for j, i in enumerate(red):
            gold[k, i] = base + 10 * j
        for j, i in enumerate(blue):
            gold[k, i] = base + 10 * j + lead // 5
    return gold


def _put(s: Store, name: str, values, dtype) -> None:
    values = np.asarray(values, dtype=dtype)
    a = s.allocate(name, values.shape, dtype)
    a[...] = values
    a.flush()
    del a


KEY_PREFIX = 'TOYKEY'          # the toy match keys; no S3 code may read ``matches/key`` or let this text into a message


def toy_key(m: int) -> str:
    return f'{KEY_PREFIX}_{m:04d}'


def build_toy_s1(root, matches=None) -> Store:
    """The S1 store of the toy matches (default ``TOY_MATCHES``) under ``root``, complete: the arrays the S3 code reads, in the dtypes of
    ``run_s1.SPEC``, with S1's offsets and its frame-clock rows (made by ``frame_rows.frame_rows``, the real rule), and ``matches/key``
    filled with ``toy_key`` sentinels that a test can look for in messages."""
    matches = TOY_MATCHES if matches is None else tuple(matches)
    s = Store(root)
    cols = {name: [] for name in ('ts', 'num', 'kt', 'killer', 'bounty', 'shutdown', 'amask', 'ot', 'kind', 'sub', 'taker',
                                  'fr_match', 'fr_k', 'fr_t0')}
    counts = {'frame': [], 'kill': [], 'obj': [], 'row': []}
    for m, spec in enumerate(matches):
        if spec.role not in C.ROLES:
            raise ValueError(f'unknown role {spec.role!r}')
        ts = np.array(spec.ts, dtype=np.int64)
        for what, rows in (('objectives', spec.objectives), ('kills', spec.kills)):
            if any(b[0] < a[0] for a, b in zip(rows, rows[1:])):
                raise ValueError(f'the {what} of a toy match are in time order, as S1 stores them')
        num = np.zeros((len(ts), 10, len(C.FRAME_FIELDS)))
        num[:, :, TOTAL_GOLD] = toy_gold(spec)
        cols['ts'].append(ts)
        cols['num'].append(num)
        counts['frame'].append(len(ts))
        for t, killer, bounty, shutdown, assisters in spec.kills:
            cols['kt'].append(t)
            cols['killer'].append(killer)
            cols['bounty'].append(bounty)
            cols['shutdown'].append(shutdown)
            cols['amask'].append(sum(1 << (a - 1) for a in assisters if 1 <= a <= 10))
        counts['kill'].append(len(spec.kills))
        for t, name, taker in spec.objectives:
            kind, sub = KIND_SUB[name]
            cols['ot'].append(t)
            cols['kind'].append(kind)
            cols['sub'].append(sub)
            cols['taker'].append(taker)
        counts['obj'].append(len(spec.objectives))
        rows = frame_rows_mod.frame_rows(types.SimpleNamespace(frame_ts=ts, T_ms=spec.T_ms))
        cols['fr_match'].extend([m] * len(rows.k))
        cols['fr_k'].extend(rows.k.tolist())
        cols['fr_t0'].extend(rows.t0.tolist())
        counts['row'].append(len(rows.k))
    off = lambda c: np.concatenate([[0], np.cumsum(c)])            # noqa: E731
    _put(s, 'matches/key', [toy_key(m).encode('ascii') for m in range(len(matches))], 'S24')
    _put(s, 'matches/role', [C.ROLES.index(sp.role) for sp in matches], 'i1')
    _put(s, 'matches/team', [sp.team for sp in matches], 'i2')
    _put(s, 'matches/T_ms', [sp.T_ms for sp in matches], 'i8')
    for name, key in (('frame_off', 'frame'), ('kill_off', 'kill'), ('obj_off', 'obj'), ('row_off', 'row')):
        _put(s, f'matches/{name}', off(counts[key]), 'i8')
    _put(s, 'frames/ts', np.concatenate(cols['ts']) if cols['ts'] else [], 'i8')
    _put(s, 'frames/num', np.concatenate(cols['num']) if cols['num'] else np.zeros((0, 10, len(C.FRAME_FIELDS))), 'f4')
    for table, names in (('kills', (('t', 'kt', 'i8'), ('killer', 'killer', 'i1'), ('bounty', 'bounty', 'i4'),
                                    ('shutdown', 'shutdown', 'i4'), ('amask', 'amask', 'i2'))),
                         ('objectives', (('t', 'ot', 'i8'), ('kind', 'kind', 'i1'), ('sub', 'sub', 'i1'), ('taker', 'taker', 'i1'))),
                         ('frame_rows', (('match', 'fr_match', 'i4'), ('k', 'fr_k', 'i2'), ('t0', 'fr_t0', 'i8')))):
        for column, key, dtype in names:
            _put(s, f'{table}/{column}', cols[key], dtype)
    s.finalize()
    return s


# ---- the toy S2 store -----------------------------------------------------------------------------------------------------------
# the blue_/red_ pairs of the S2 summary that the controls of the window table use, in the order of c_ept.flowprice.CTRL_NAMES (the
# plates are three columns), and the two buff-seconds pairs
SUMMARY_PAIRS = ('towers_outer', 'towers_inner', 'towers_base', 'towers_nexus', 'inhibs_down', 'plates_top', 'plates_mid', 'plates_bot',
                 'dragons', 'elders', 'barons', 'heralds', 'atakhans', 'voidgrubs', 'has_soul')
JUNK = 7777.0                    # what every column of an event view holds: a frame-clock lookup that picked one would show it


def summary_values(row: int) -> dict:
    """{summary column name: value} of the frame-clock view of the global frame row ``row`` for the columns above and the buff seconds; the
    blue and red values are different small integers that change with the row, so that a swapped team or a wrong column shows. The buff
    seconds are 0 or positive in all four combinations."""
    out = {}
    for i, name in enumerate(SUMMARY_PAIRS):
        out['blue_' + name] = float((3 * i + row) % 7)
        out['red_' + name] = float((2 * i + 3 * row + 1) % 5)
    out['blue_baron_buff_s'] = 10.0 + row if row % 2 == 0 else 0.0
    out['red_baron_buff_s'] = 5.5 if (row // 2) % 2 == 0 else 0.0
    out['blue_elder_buff_s'] = 20.0 + row if (row // 3) % 2 == 0 else 0.0
    out['red_elder_buff_s'] = 8.0 if row % 3 == 0 else 0.0
    return out


def random_summary_values(row: int) -> dict:
    """Like ``summary_values`` but independent random integers 0 .. 5 for every blue_ and red_ column and the buff seconds on or off at
    random, fixed by the row number: no short period, so the 15 controls are linearly independent over a few thousand rows."""
    rng = np.random.default_rng(10_000 + row)
    out = {}
    for name in SUMMARY_PAIRS:
        out['blue_' + name], out['red_' + name] = float(rng.integers(0, 6)), float(rng.integers(0, 6))
    for buff in ('baron', 'elder'):
        for team in ('blue', 'red'):
            out[f'{team}_{buff}_buff_s'] = float(rng.integers(0, 2)) * float(rng.integers(1, 100))
    return out


def build_toy_s2(root, s1: Store, events_per_match: int = 2, values=None, features: bool = False) -> Store:
    """An S2 store for the S1 store ``s1``: per match its frame-clock views (src 0, one per S1 frame row, in S1 order), then
    ``events_per_match`` event views (src 1 and 2) whose ``src_index`` numbers run from 0 on, so that they collide with frame-row numbers.
    ``x/summary`` holds ``values(frame row)`` (default ``summary_values``) in the frame-clock views and ``JUNK`` in the event views.

    The patterns of ``summary_values`` repeat with short periods, so over many rows some of the blue-minus-red controls that the window
    regression reads add up to a constant: a regression on them is singular. A test that fits the window regression on a big toy store
    passes ``values=random_summary_values`` (independent random values).

    ``features`` True adds what the baseline features of stage S3 Task 5 read (see "the feature arrays of the toy S2" at the end of this
    file): ``x/players_num`` and ``x/geo_pp`` for every view, and in the frame-clock views the summary cells of ``toy_feature_summary``.
    The event views stay ``JUNK`` in every summary column. The toy missing positions (``TOY_MISSING``) are meant for the default toy."""
    from c_ept.preprocessing import tokens                                       # the column names of the summary; loads the hash-pinned LEGACY modules
    index = {name: i for i, name in enumerate(tokens.SUMMARY)}
    row_off = np.asarray(s1.array('matches/row_off'))
    frame_k, frame_t0 = np.asarray(s1.array('frame_rows/k')), np.asarray(s1.array('frame_rows/t0'))
    role = np.asarray(s1.array('matches/role'))
    views = {c: [] for c in ('src', 'src_index', 'match', 'ref', 't0', 'frame_k')}
    summary = []
    event_row = 0
    for m in range(len(row_off) - 1):
        for r in range(int(row_off[m]), int(row_off[m + 1])):
            for c, v in (('src', 0), ('src_index', r), ('match', m), ('ref', int(frame_t0[r])), ('t0', int(frame_t0[r])),
                         ('frame_k', int(frame_k[r]))):
                views[c].append(v)
            row = np.zeros(len(tokens.SUMMARY), dtype=np.float32)
            for name, value in (summary_values if values is None else values)(r).items():
                row[index[name]] = value
            if features:
                for name, value in toy_feature_summary(r, int(frame_t0[r]), fit=C.ROLES[int(role[m])] == 'FIT').items():
                    row[index[name]] = value
            summary.append(row)
        for j in range(events_per_match):
            for c, v in (('src', 1 + j % 2), ('src_index', event_row), ('match', m), ('ref', 100_000 + event_row),
                         ('t0', 60_300), ('frame_k', 1)):
                views[c].append(v)
            summary.append(np.full(len(tokens.SUMMARY), JUNK, dtype=np.float32))
            event_row += 1
    s = Store(root)
    for c, dtype in (('src', 'i1'), ('src_index', 'i4'), ('match', 'i4'), ('ref', 'i8'), ('t0', 'i8'), ('frame_k', 'i2')):
        _put(s, f'views/{c}', views[c], dtype)
    _put(s, 'x/summary', np.array(summary).reshape(-1, len(tokens.SUMMARY)), 'f4')
    if features:
        _put(s, 'x/players_num', toy_players(len(summary)), 'f4')
        _put(s, 'x/geo_pp', toy_geo(len(summary)), 'f2')
    s.finalize()
    return s


# ---- a toy match as a MatchRecord (to go through the real S1 extraction) --------------------------------------------------------
_RAW = {   # name -> (event type, monsterType, monsterSubType, buildingType, towerType)
    'dragon': ('ELITE_MONSTER_KILL', 'DRAGON', 'FIRE_DRAGON', None, None),
    'elder': ('ELITE_MONSTER_KILL', 'DRAGON', 'ELDER_DRAGON', None, None),
    'grubs': ('ELITE_MONSTER_KILL', 'HORDE', None, None, None),
    'herald': ('ELITE_MONSTER_KILL', 'RIFTHERALD', None, None, None),
    'baron': ('ELITE_MONSTER_KILL', 'BARON_NASHOR', None, None, None),
    'atakhan': ('ELITE_MONSTER_KILL', 'ATAKHAN', None, None, None),
    'tower_outer': ('BUILDING_KILL', None, None, 'TOWER_BUILDING', 'OUTER_TURRET'),
    'tower_inner': ('BUILDING_KILL', None, None, 'TOWER_BUILDING', 'INNER_TURRET'),
    'tower_base': ('BUILDING_KILL', None, None, 'TOWER_BUILDING', 'BASE_TURRET'),
    'tower_nexus': ('BUILDING_KILL', None, None, 'TOWER_BUILDING', 'NEXUS_TURRET'),
    'inhibitor': ('BUILDING_KILL', None, None, 'INHIBITOR_BUILDING', None),
    'plate': ('TURRET_PLATE_DESTROYED', None, None, None, None),
    'unknown_kind': ('ELITE_MONSTER_KILL', 'SOME_OTHER_MONSTER', None, None, None),
    'tower_unknown_tier': ('BUILDING_KILL', None, None, 'TOWER_BUILDING', 'SOME_OTHER_TURRET'),
}


def toy_record(spec: ToyMatch, key: str, patch: str = '15.18'):
    """The ``MatchRecord`` of a toy match with the standard teams (participants 1-5 blue), whose S1 extraction must give the arrays of
    ``build_toy_s1``: the same kills (a champion killer gets a victim on the other team), objectives and gold."""
    import dataclasses

    from c_ept.tests import synth
    if tuple(spec.team) != BLUE5 + RED5:
        raise ValueError('only the standard team layout can be sent through the extraction')
    kills = [dict(t=t, x=5_000 + 10 * i, y=5_000, killer=killer, victim=(1 if killer > 5 else 6) if killer else 1,
                  assists=assisters, bounty=bounty, shutdown=shutdown)
             for i, (t, killer, bounty, shutdown, assisters) in enumerate(spec.kills)]
    objectives = []
    for i, (t, name, taker) in enumerate(spec.objectives):
        typ, monster, sub, building, tower = _RAW[name]
        elite = typ == 'ELITE_MONSTER_KILL'
        team = None if taker == NOBODY else (TEAM_BLUE + 100 * taker if elite else TEAM_RED - 100 * taker)    # a building: the owner
        objectives.append(dict(t=t, type=typ, x=6_000 + 10 * i, y=6_000, killer=0, team=team, monster=monster, sub=sub, building=building,
                               tower=tower, lane=None if elite else 'MID_LANE'))
    rec = synth.make_record(T_ms=spec.T_ms, kills=kills, objectives=objectives, patch=patch, key=key, n_frames=len(spec.lead))
    num = np.array(rec.frame_num)
    num[:, :, TOTAL_GOLD] = toy_gold(spec)
    return dataclasses.replace(rec, frame_num=num)


# ---- numbers worked out by hand for the toy matches --------------------------------------------------------------------------------
# "match N" is the S1 match row N of TOY_MATCHES; "frame row" is a global S1 frame-row number; the event minute j is as in flowprice.
TOY_FIT_MATCHES = {0, 1, 4, 5}
TOY_STOP_MATCHES, TOY_DEV_MATCHES = {2}, {3}
TOY_TS_M0 = frame_times(12)
TOY_LEAD_M0 = LEAD_M0
TOY_N_FRAMES = (12, 11, 9, 10, 2, 6)                    # regular frames per toy match

# the dragon of red in match 0 (minute 7; complete: 7 - 5 >= 0 and 7 + 2 = 9 <= 11); r = -4 .. +2 is minute 3 .. 9
TOY_T_RED_DRAGON = 400_000
TOY_K_RED_DRAGON = 7
#   changes of the lead (blue - red) in minutes 3 .. 9: 50, 200, 100, 100, 100, -300, -200; red took the dragon, so every sign flips
TOY_Y_RED_DRAGON = (-50, -200, -100, -100, -100, 300, 200)
#   what else happened in those minutes, from red's side (what blue got counts negative):
#   kill gold in minutes 3 .. 9: -300, 0, +450, -450, +300 - 150 = +150, +600, -400 (the execution at 390,000 pays nobody);
#   objectives: minute 4 blue outer tower -1, minute 5 blue baron -1 and blue plate -1, minute 6 red grubs +1, minute 7 blue inner
#   tower -1 (the red dragon itself is left out), minute 9 blue elder -1
TOY_CO_RED_DRAGON = {'kill_gold': (-300, 0, 450, -450, 150, 600, -400),
                     'tower_outer': (0, -1, 0, 0, 0, 0, 0), 'baron': (0, 0, -1, 0, 0, 0, 0), 'plate': (0, 0, -1, 0, 0, 0, 0),
                     'grubs': (0, 0, 0, 1, 0, 0, 0), 'tower_inner': (0, 0, 0, 0, -1, 0, 0), 'elder': (0, 0, 0, 0, 0, 0, -1)}

# the baron of blue in match 0 (t = 250,000, minute 5): the first minute with a complete window; r = -4 .. +2 is minute 1 .. 7, so
# lead[0] is used
#   changes of the lead in minutes 1 .. 7: 100, 150, 50, 200, 100, 100, 100
TOY_Y_BARON_M0 = (100, 150, 50, 200, 100, 100, 100)
#   kill gold (blue - red) in minutes 1 .. 7: 0, +300, +300, 0, -450, +450, -150 (the execution of minute 7 pays nobody);
#   objectives: minute 4 blue outer tower +1, minute 5 blue plate +1 (the baron itself is left out), minute 6 red grubs -1,
#   minute 7 red dragon -1 and blue inner tower +1
TOY_CO_BARON_M0 = {'kill_gold': (0, 300, 300, 0, -450, 450, -150),
                   'tower_outer': (0, 0, 0, 1, 0, 0, 0), 'plate': (0, 0, 0, 0, 1, 0, 0), 'grubs': (0, 0, 0, 0, 0, -1, 0),
                   'dragon': (0, 0, 0, 0, 0, 0, -1), 'tower_inner': (0, 0, 0, 0, 0, 0, 1)}
# the elder of blue in match 0 (t = 540,000, minute 9): the last minute with a complete window; r = -4 .. +2 is minute 5 .. 11, and
# 11 is the last regular frame
#   changes of the lead in minutes 5 .. 11: 100, 100, 100, -300, -200, -100, 50
TOY_Y_ELDER_M0 = (100, 100, 100, -300, -200, -100, 50)
#   kill gold in minutes 5 .. 11: -450, +450, -150, -600, +400, 0, -300;
#   objectives: minute 5 blue baron +1 and blue plate +1, minute 6 red grubs -1, minute 7 red dragon -1 and blue inner tower +1,
#   minute 10 red herald -1 (the nexus tower of minute 12 is outside the window)
TOY_CO_ELDER_M0 = {'kill_gold': (-450, 450, -150, -600, 400, 0, -300),
                   'baron': (1, 0, 0, 0, 0, 0, 0), 'plate': (1, 0, 0, 0, 0, 0, 0), 'grubs': (0, -1, 0, 0, 0, 0, 0),
                   'dragon': (0, 0, -1, 0, 0, 0, 0), 'tower_inner': (0, 0, 1, 0, 0, 0, 0), 'herald': (0, 0, 0, 0, 0, -1, 0)}

# the dragon and the plate of blue in match 1, minute 6 (complete: 6 - 5 >= 0 and 6 + 2 = 8 <= 10); r = -4 .. +2 is minute 2 .. 8
TOY_T_DRAGON, TOY_T_PLATE = 310_000, 350_000
TOY_Y_DRAGON_M1 = (50, 100, 50, 50, 200, -50, -50)      # blue took it: the changes of minutes 2 .. 8 as they are
#   kill gold in minutes 2 .. 8: minute 4 +300, minute 6 -450 (red), minute 8 +450, the execution in minute 5 pays nobody;
#   objectives from blue's side: minute 3 red base tower -1, minute 6 the plate +1 and the red inhibitor -1 (the dragon itself is
#   left out), minute 7 red baron -1, minute 8 two plates +2 and the atakhan +1
TOY_CO_DRAGON_M1 = {'kill_gold': (0, 0, 300, 0, -450, 0, 450),
                    'tower_base': (0, -1, 0, 0, 0, 0, 0), 'plate': (0, 0, 0, 0, 1, 0, 2), 'inhibitor': (0, 0, 0, 0, -1, 0, 0),
                    'baron': (0, 0, 0, 0, 0, -1, 0), 'atakhan': (0, 0, 0, 0, 0, 0, 1)}

# events of the FIT toy matches (9 + 11 + 2 known objectives), by group in the order dragon, elder, grubs, herald, baron, atakhan,
# tower_outer, tower_inner, tower_base, tower_nexus, inhibitor, plate
TOY_N_EVENTS = 22
TOY_EVENTS_BY_GROUP = (2, 2, 2, 1, 2, 1, 3, 1, 1, 1, 2, 4)
# incomplete: match 0: outer tower (minute 4), herald (10), nexus tower (12); match 1: base tower (3), grubs (9), outer tower (10),
# elder (11); match 4: outer tower (1) and inhibitor (2 = after its last frame)
TOY_N_INCOMPLETE = 9
TOY_INCOMPLETE_BY_GROUP = (0, 1, 1, 1, 0, 0, 3, 0, 1, 1, 1, 0)
# (match, event minute) -> is the window complete (k - 5 >= 0 and k + 2 <= last regular frame): match 0 has 12 frames, so 5 <= k <= 9;
# match 1 has 11, so 5 <= k <= 8; the two frames of match 4 never
TOY_EVENT_COMPLETE = {(0, 4): False, (0, 5): True, (0, 6): True, (0, 7): True, (0, 9): True, (0, 10): False, (0, 12): False,
                      (1, 3): False, (1, 6): True, (1, 7): True, (1, 8): True, (1, 9): False, (1, 10): False, (1, 11): False,
                      (4, 1): False, (4, 2): False}
TOY_N_UNKNOWN = 5                                       # match 0: 3, match 1: 2
TOY_MINUTES = {   # (match, time) -> event minute of a few events on or next to a frame time
    (0, 300_300): 5, (0, 300_301): 6, (1, 420_300): 7, (1, 480_300): 8, (1, 481_000): 9, (0, 690_000): 12, (4, 50_000): 1,
    (4, 90_000): 2}

# window rows: the frame rows of the FIT matches are match 0: k = 2 .. 11 (10), match 1: k = 2 .. 9 (8), match 4: none,
# match 5: k = 2 .. 5 (4)
TOY_N_FIT_FRAME_ROWS = 22
TOY_N_DROPPED_LAST = 2                                  # match 0 at k = 11 and match 5 at k = 5: no regular frame k + 1
TOY_N_FIT_WINDOW_ROWS = 20
#   known objectives of the FIT matches that lie in no kept window (kept windows: match 0 minutes 3 .. 11, match 1 minutes 3 .. 10,
#   match 5 minutes 3 .. 5): the nexus tower of match 0 (minute 12), the elder of match 1 (minute 11) and the two objectives of
#   match 4 (no window at all)
TOY_OBJ_UNWINDOWED = (0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0)
# the window rows of match 0, k = 2 .. 10 (frame rows 0 .. 8): y = lead[k+1] - lead[k]; the objectives by group in minute k + 1
# (blue - red); the kill gold, kill count and assist count of minute k + 1 (blue - red)
TOY_WINDOW_M0 = {
    # k: (y, {group: blue - red count}, (kill gold, kills, assists))
    2: (50, {}, (300, 1, 2)),
    3: (200, {'tower_outer': 1}, (0, 0, 0)),
    4: (100, {'baron': 1, 'plate': 1}, (-450, -1, -3)),
    5: (100, {'grubs': -1}, (450, 1, 0)),
    6: (100, {'dragon': -1, 'tower_inner': 1}, (-150, 0, -1)),
    7: (-300, {}, (-600, -1, -4)),
    8: (-200, {'elder': 1}, (400, 1, 2)),
    9: (-100, {'herald': -1}, (0, 0, 0)),
    10: (50, {}, (-300, -1, 0)),
}
# the controls of three window rows of match 0 by frame row, in the order of c_ept.flowprice.CTRL_NAMES: lead x phase (8), phase (8),
# momentum, cumulative kill difference, the 13 summary differences (blue - red), baron buff, elder buff
#   frame row 0 (k = 2): lead 250 at 120.3 s (phase 0 = [120, 300) s), momentum 250 - 100, one blue kill so far (at 100,000); the summary
#   pattern of ``summary_values(0)`` gives towers_outer 0 - 1 = -1, ..., has_soul 0 - 4 = -4; both buffs are on for both teams: 0 and 0
#   frame row 1 (k = 3): lead 300 at 180.3 s (phase 0), momentum 50, two kills; summary of frame row 1; baron buff off for blue and on for
#   red: -1; elder buff on for blue and off for red: +1
#   frame row 5 (k = 7): lead 800 at 420.3 s (phase 1 = [300, 600) s), momentum 100; kills up to 420,300 in time order: blue, blue,
#   red, blue, (an execution: nobody), red, blue = +1 +1 -1 +1 -1 +1 = 2
TOY_CTRL_ROWS_M0 = {
    0: (250, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 150, 1, -1, 0, 6, 0, 1, 1, 1, 2, 1, 2, 1, 2, -4, 0, 0),
    1: (300, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 50, 2, -3, 3, -3, 3, 4, 0, 4, -2, -1, 5, -1, 5, -1, -1, 1),
    5: (0, 800, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 100, 2, 4, -2, 4, -2, -1, 9, -1, 0, -1, 0, 6, 0, 1, -1, 0),
}


# ---- the fixtures -----------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope='module')
def toy_s1(tmp_path_factory):
    """The S1 store of ``TOY_MATCHES``, written once per test module and read only."""
    folder = tmp_path_factory.mktemp('toy_s1')
    s = build_toy_s1(folder / 's1')
    yield s
    del s
    gc.collect()                                   # Windows does not delete a folder with open memory maps
    shutil.rmtree(folder, ignore_errors=True)


@pytest.fixture(scope='module')
def toy_s2(tmp_path_factory, toy_s1):
    """The S2 store that goes with ``toy_s1``, written once per test module and read only."""
    folder = tmp_path_factory.mktemp('toy_s2')
    s = build_toy_s2(folder / 's2', toy_s1, features=True)
    yield s
    del s
    gc.collect()
    shutil.rmtree(folder, ignore_errors=True)


# ---- the numbers of math-control note 03 (docs/superpowers/notes/2026-10-21-math-control-03.md), for the later S3 tests ------------------
# note 03 section A (flow price: stacked event study with event fixed effects, P2): two events, relative minutes r = -4 .. +2, the
# changes of the taker's gold lead (A1 hand calculation); the other numbers are the result of the fixed-effects regression of A1
TOY_EVENT_Y = ((40, 60, 50, 50, 900, 200, 100), (-20, 0, 30, 10, 700, 300, 60))
TOY_BETA7 = (-20, 0, 10, 0, 770, 220, 50)               # beta_{-4..+2} with beta_{-1} = 0
TOY_PRICE = 1047.5                                      # (770 + 220 + 50) + 3 x 7.5
TOY_PRETREND = 30.0                                     # (10 + 0) - (-20 + 0)
# note 03 section B (immediate value: the minute window regression, P1): six minutes, the change of the lead and the baron count
# (blue - red)
TOY_WINDOW_Y = (100, -50, 1600, 50, -1450, 0)
TOY_WINDOW_TREAT = (0, 0, 1, 0, -1, 0)
TOY_IMMEDIATE = 1525.0                                  # the slope; the intercept is 250 / 6
# note 03 section C (Poisson NLL with exposure, log y! included, P4 and P10)
TOY_Y = (1, 0, 0, 2)
TOY_RATE = (0.02, 0.02, 0.02, 0.05)                     # per second
TOY_EXP = (10.0, 10.0, 0.0, 10.0)                       # seconds; the third cell is not observed
TOY_POISSON_NLL = 4.5888795
# note 03 section D (gold bins and bin means, P4): bin = min(gold // 100, 30)
TOY_GOLD = (300, 300, 450, 350, 3200, 4100)
TOY_GOLD_BINS = (3, 3, 4, 3, 30, 30)
TOY_GOLD_BIN3_MEAN = 950 / 3
TOY_GOLD_OVERFLOW_MEAN = 3650.0
# note 03 section E (one composed row, P6 and P10): onsets 0.01 per second, marks 0.6 / 0.4, t0 = 1,000 s, the game ends at T
TOY_T0_S = 1000.0
TOY_T_END_FAR_S = 5000.0
TOY_T_END_CUT_S = 1075.0
TOY_EXPECTED_V60 = 32.16                                # T = 5,000 s
TOY_EXPECTED_V60_CUT = 22.12                            # T = 1,075 s
# note 03 section F (calibration, P7)
TOY_CAL_E = (0, 0, 100, 100, 200, 200)
TOY_CAL_V = (0, 50, 50, 150, 150, 250)
TOY_CAL_SLOPE = 0.875                                   # the intercept is 125 / 6
TOY_CAL_BIAS = 25 / 3                                   # mean(V - E)
TOY_CAL_W = (0.25, 0.25, 0.25, 0.25, 0.5, 0.5)          # 1 / n_m: match A has four rows, match B two
TOY_CAL_WSLOPE = 12187.5 / 13750
# note 03 section G (overdispersion, P8): Pearson phi with mu = 1
TOY_PHI_Y = (0, 0, 3, 0, 1, 2)
TOY_PHI = 4 / 3
# note 03 section A, "try it yourself" (A3): the r = -2 and r = -1 minutes of event 1 changed to 150 and 250
TOY_EVENT_Y_TRY = ((40, 60, 150, 250, 900, 200, 100), (-20, 0, 30, 10, 700, 300, 60))
TOY_BETA7_TRY = (-120, -100, -40, 0, 670, 120, -50)
TOY_PRETREND_TRY = 180.0
# note 03 section B, "try it yourself": the fifth minute -1,450 changed to -1,550
TOY_WINDOW_Y_TRY = (100, -50, 1600, 50, -1550, 0)
TOY_IMMEDIATE_TRY = 1575.0


# ---- synthetic tables with a known truth, for the estimators of ``flowprice`` (stage S3 Task 2) ---------------------------------------
# These are real ``flowprice.EventTable`` / ``WindowTable`` objects built from random numbers, not read from a store, so that an estimator
# can be tested against the truth that made the data. ``y`` of an event is already signed by the taker, as the event table stores it.
TRUE_BETA = np.array([10.0, 30.0, 40.0, 30.0, 800.0, 250.0, 80.0])     # beta_{-4..+2} with the level of r = -1 (30) still in: the means of
#                                                                       note 03 A1, so that TRUE_BETA - TRUE_BETA[3] is TOY_BETA7
TRUE_GAMMA = np.array([1.0, 120.0, -80.0, 40.0, -60.0, 200.0, -150.0, 90.0, 60.0, -30.0, 10.0, 75.0, -40.0])      # the 13 ``co`` columns
TRUE_WINDOW_CO = np.array([1.0, 20.0, 5.0])                            # kill gold, kill count, assist count of the window regression
if not np.allclose(TRUE_BETA - TRUE_BETA[3], TOY_BETA7):
    raise RuntimeError('TRUE_BETA must give the beta of note 03 section A once the level of r = -1 is taken out')


def _event_table(match, group, taker, k, complete, y, co, n_matches) -> 'flowprice.EventTable':
    """A ``flowprice.EventTable`` from plain arrays (the dtypes of ``event_table``); the time is a made-up point in the minute."""
    n = len(match)
    return flowprice.EventTable(
        match=np.asarray(match, dtype=np.int32), group=np.asarray(group, dtype=np.int8), taker=np.asarray(taker, dtype=np.int8),
        t=60_000 * np.asarray(k, dtype=np.int64) - 15_000, k=np.asarray(k, dtype=np.int32), complete=np.asarray(complete, dtype=np.bool_),
        y=np.asarray(y, dtype=np.float64).reshape(n, 7), co=np.asarray(co, dtype=np.float64).reshape(n, 7, 13), n_unknown=0,
        n_matches=int(n_matches))


def event_table_from_y(y, group: str = 'baron') -> 'flowprice.EventTable':
    """One event per row of ``y`` (n, 7), each in a match of its own, all complete, with no kill and no other objective around it."""
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    return _event_table(np.arange(n), np.full(n, GROUP_NAMES.index(group)), np.zeros(n), np.full(n, 8), np.ones(n, dtype=bool), y,
                        np.zeros((n, 7, 13)), n)


def synthetic_events(beta=TRUE_BETA, n: int = 1000, event_sd: float = 0.0, seed: int = 0, group: str = 'baron', noise_sd: float = 30.0,
                     per_match: int = 2, n_incomplete: int = 0, own_group_co: bool = True) -> 'flowprice.EventTable':
    """``n`` complete events of ``group`` (and ``n_incomplete`` incomplete ones, with NaN ``y`` and ``co``, spread among them),
    ``per_match`` to a match, ordered by match: y[e, r] = a_e + beta_r + TRUE_GAMMA . co[e, r] + noise with a_e ~ N(0, event_sd), noise ~
    N(0, noise_sd).

    ``co`` is random, and its kill gold is larger in the event's own minute (r = 0), so an estimator that leaves the controls out is
    biased in beta_0 by about 250. ``own_group_co`` False leaves the column of the group itself at zero, as it is in real data for a group
    that cannot happen twice in a minute (a column without variation)."""
    rng = np.random.default_rng(seed)
    beta = np.asarray(beta, dtype=np.float64)
    g, total = GROUP_NAMES.index(group), n + n_incomplete
    co = np.zeros((total, 7, 13))
    co[:, :, 0] = 50.0 * np.round((rng.normal(0.0, 300.0, (total, 7)) + 250.0 * (np.arange(7) == 4)) / 50.0)
    co[:, :, 1:] = rng.choice([-1.0, 0.0, 1.0], size=(total, 7, 12), p=[0.06, 0.88, 0.06])
    if not own_group_co:
        co[:, :, 1 + g] = 0.0
    y = rng.normal(0.0, event_sd, (total, 1)) + beta[None, :] + co @ TRUE_GAMMA + rng.normal(0.0, noise_sd, (total, 7))
    complete = np.ones(total, dtype=bool)
    complete[rng.permutation(total)[:n_incomplete]] = False
    y[~complete] = np.nan
    co[~complete] = np.nan
    match = np.arange(total) // per_match
    return _event_table(match, np.full(total, g), rng.integers(0, 2, total), rng.integers(5, 30, total), complete, y, co,
                        (total + per_match - 1) // per_match)


def with_event_level_shift(table, seed: int = 0, sd: float = 400.0) -> 'flowprice.EventTable':
    """The same table with a random constant added to every minute of each event's ``y`` (a different level of the lead in each event)."""
    shift = np.random.default_rng(seed).normal(0.0, sd, (len(table.y), 1))
    return replace(table, y=table.y + shift)


def concat_events(tables) -> 'flowprice.EventTable':
    """The events of several tables in one table, ordered by match (stable); ``n_matches`` is the largest of the tables."""
    tables = list(tables)
    fields = ('match', 'group', 'taker', 't', 'k', 'complete', 'y', 'co')
    cat = {f: np.concatenate([getattr(t, f) for t in tables]) for f in fields}
    order = np.argsort(cat['match'], kind='stable')
    return flowprice.EventTable(**{f: cat[f][order] for f in fields}, n_unknown=sum(t.n_unknown for t in tables),
                                n_matches=max(t.n_matches for t in tables))


# mixed_events: every group has events. herald has a ramp before the event (a pre-trend of 200), every other group is flat
BETA_FLAT = np.array([0.0, 0.0, 0.0, 0.0, 700.0, 200.0, 50.0])
BETA_RAMP = np.array([0.0, 50.0, 100.0, 150.0, 900.0, 300.0, 80.0])
RAMP_PRETREND = 200.0                                    # (100 + 150) - (0 + 50)
N_MIXED_MATCHES = 160


def mixed_events(seed: int = 0, n: int = 300, n_incomplete: int = 20, noise_sd: float = 30.0, event_sd: float = 300.0):
    """Events of all 12 groups in 160 matches (``n`` complete and ``n_incomplete`` incomplete ones per group, two to a match): the true
    beta of herald is ``BETA_RAMP`` (pre-trend 200), that of every other group ``BETA_FLAT`` (pre-trend 0). Dragon and baron have no
    ``co`` column of their own (it cannot happen twice in a minute), as in real data."""
    tables = [synthetic_events(BETA_RAMP if name == 'herald' else BETA_FLAT, n, event_sd, seed * 100 + i, name, noise_sd,
                               n_incomplete=n_incomplete, own_group_co=name not in ('dragon', 'baron'))
              for i, name in enumerate(GROUP_NAMES)]
    return concat_events(tables)


DEFAULT_WINDOW_EFFECT = {name: 100.0 * (i + 1) for i, name in enumerate(GROUP_NAMES)}      # dragon 100, elder 200, ..., plate 1,200


def synthetic_windows(n: int = 4000, effect=None, seed: int = 0, noise_sd: float = 100.0, rows_per_match: int = 24,
                      n_matches: int | None = None, n_obj_unwindowed=None) -> 'flowprice.WindowTable':
    """``n`` window rows in matches of about ``rows_per_match`` rows (or in ``n_matches`` matches), with
        y = sum_g effect[g] treat_g + TRUE_WINDOW_CO . co + ctrl . c + noise      (effect: group name -> gold; a group not named: 0)
    ``treat`` is +-1 in about 5% of the cells of each group; ``ctrl`` has the structure of ``CTRL_NAMES`` (one phase indicator per row, the
    lead in that phase's column) with random values, so that the 48 columns are independent."""
    rng = np.random.default_rng(seed)
    eff = np.array([(effect or {}).get(name, 0.0) for name in GROUP_NAMES], dtype=np.float64)
    treat = np.where(rng.random((n, 12)) < 0.05, rng.choice([-1.0, 1.0], (n, 12)), 0.0)
    co = np.column_stack([50.0 * np.round(rng.normal(0.0, 300.0, n) / 50.0), np.round(rng.normal(0.0, 1.2, n)),
                          np.round(rng.normal(0.0, 2.0, n))])
    ctrl = np.zeros((n, 33))
    phase, at = rng.integers(0, 8, n), np.arange(n)
    ctrl[at, phase] = 5.0 * np.round(rng.normal(0.0, 400.0, n) / 5.0)
    ctrl[at, 8 + phase] = 1.0
    ctrl[:, 16] = 5.0 * np.round(rng.normal(0.0, 60.0, n) / 5.0)
    ctrl[:, 17] = np.round(rng.normal(0.0, 3.0, n))
    ctrl[:, 18:31] = np.round(rng.normal(0.0, 1.5, (n, 13)))
    ctrl[:, 31:33] = rng.integers(-1, 2, (n, 2))
    y = treat @ eff + co @ TRUE_WINDOW_CO + ctrl @ rng.normal(0.0, 15.0, 33) + rng.normal(0.0, noise_sd, n)
    n_matches = (n + rows_per_match - 1) // rows_per_match if n_matches is None else n_matches
    match = np.sort(rng.integers(0, n_matches, n))
    return flowprice.WindowTable(
        match=match.astype(np.int32), row=np.arange(n, dtype=np.int64), k=rng.integers(2, 30, n).astype(np.int32), y=y, treat=treat, co=co,
        ctrl=ctrl, n_dropped_last=0,
        n_obj_unwindowed=np.zeros(12, dtype=np.int64) if n_obj_unwindowed is None else np.asarray(n_obj_unwindowed, dtype=np.int64),
        n_unknown=0, n_matches=int(n_matches))


def window_design(w) -> np.ndarray:
    """(n, 48) the design matrix of the window regression, written out: treat (12), co (3), ctrl (33); no intercept."""
    return np.column_stack([w.treat, w.co, w.ctrl])


def mixed_windows(seed: int = 0, n: int = 6000, n_obj_unwindowed=None, n_matches: int = N_MIXED_MATCHES) -> 'flowprice.WindowTable':
    """Window rows in the matches of ``mixed_events`` (160 for its defaults; pass ``events.n_matches`` for other sizes): group g has the
    immediate effect ``DEFAULT_WINDOW_EFFECT`` (100 (g + 1) gold)."""
    return synthetic_windows(n, DEFAULT_WINDOW_EFFECT, seed, n_matches=n_matches, n_obj_unwindowed=n_obj_unwindowed)


# ---- toy matches with a planted baron effect, for the end-to-end tests through the S1 and S2 stores --------------------------------------
PLANT_EFFECT = (1500, 400, 100)       # gold that a baron adds to the taker's lead in its own minute and the two minutes after it
PLANT_PRICE = 2000.0                  # their sum; there is no trend before the event, so this is the flow price p of the baron
PLANT_FRAMES = 16


def planted_baron_matches(n: int = 400, seed: int = 0, noise_sd: float = 60.0, drift_sd: float = 100.0) -> tuple:
    """``n`` FIT toy matches of 16 frames, each with one baron (taker blue or red, in a random minute 5 .. 13, so that its window is
    complete) and nothing else: the change of the lead in every minute is noise + a level that is the match's own (drift) + the taker's
    sign x ``PLANT_EFFECT`` in the baron's minute and the two after it. The game ends 40 s after the last frame, so that the last frame row
    exists and is dropped (it has no next regular frame)."""
    rng = np.random.default_rng(seed)
    ts = frame_times(PLANT_FRAMES)
    specs = []
    for _ in range(n):
        k, taker = int(rng.integers(5, PLANT_FRAMES - 2)), int(rng.integers(0, 2))
        sign = 1 if taker == BLUE else -1
        change = 5.0 * np.round((rng.normal(0.0, noise_sd, PLANT_FRAMES) + rng.normal(0.0, drift_sd)) / 5.0)
        for r, gold in enumerate(PLANT_EFFECT):
            change[k + r] += sign * gold
        change[0] = 0.0
        lead = tuple(int(v) for v in np.cumsum(change))
        specs.append(ToyMatch('FIT', lead, ts[-1] + 40_000, (obj(ts[k - 1] + 30_000, 'baron', taker),), ()))
    return tuple(specs)


# ---- toy matches with engagement units, for the labels of ``value`` (stage S3 Task 4) ------------------------------------------------
# Four matches, written as synthetic MatchRecords and sent through the real S1 extraction (``build_value_s1``). Positions put the kills of
# one engagement within 2,000 units of each other (they link into one unit) and different engagements far apart. Roles and patches:
#
#     match 0  FIT  15.18  T = 700,000, 12 frames, rows k = 2 .. 11     units 0-5   (the hand-worked match)
#     match 1  STOP 15.20  T = 520,000,  9 frames, rows k = 2 .. 8      units 6-8
#     match 2  DEV  15.21  T = 400,000,  7 frames, rows k = 2 .. 6      no kill at all, one objective
#     match 3  CAL  15.19  T = 300,000,  5 frames, rows k = 2 .. 4      unit 9 (a role stage S3 does not value: NaN labels)
#
# Match 0, by unit (the first kill f; the victim levels come from the LEVEL_UP events at the end of the list; "pool" is the assist pool
# min(0.5 bounty, 0.5 base(level)) m(t) with m(t) = 0.5 + 0.5 clip((t / 1000 - 90) / 120, 0, 1), base(3) = 300, base(7) = 310,
# base(18) = 420; the kill times are multiples of 15 s, so that m is a binary fraction and the sums are exact):
#   unit 0  f 135,000  blue 4 kills red 9 (level 3) with one assister, bounty 300: pool = min(150, 150) x 0.6875 = 103.125, gold 403.125
#   unit 1  f 150,000  the hand example (TOY_U), three kills in 15 s, 3 + 3 people:
#             150,000 blue 1 kills red 6 (level 7: the LEVEL_UP to 7 is on the kill's own ms), assisters 2, 3, bounty 400:
#                       pool = min(200, 155) x 0.75 = 116.25, gold 516.25 (blue)
#             155,000 red 7 kills blue 2 (level 9), no assister, bounty 300 + shutdown 150: gold 450 (red)
#             165,000 blue 3 kills red 8 (level 18), one assister, bounty 500: pool = min(250, 210) x 0.8125 = 170.625, gold 670.625 (blue)
#           kill gold blue 516.25 + 670.625 = 1,186.875, red 450.
#           attributed objectives (credit and position, inside [f - 15 s, delta) = [135,000, 170,000)): the dragon taken by blue at
#             166,000 (killer 1, assister 3: two shared people) -> 410 for blue; an inner tower taken by red 7 at 168,000 -> 350 for red.
#           converted objectives (delta = 170,000, window (170,000, 230,000]): the herald taken by blue at 190,000 (no credit) -> 630 for
#             blue; a plate taken by red 8 at 200,000 (no unit claims it) -> 160 for red. The dragon taken by red at 167,000 has credit
#             but no position, so no unit claims it, and it is converted to unit 0 (delta 155,000 < 167,000), not to unit 1 (170,000).
#           f0 of the unit = (1,186.875 + 410) - (450 + 350) = 796.875 (TOY_F0); f0 of the conversions = 630 - 160 = 470.
#   unit 2  f 300,000  an execution (no champion killer) with an assister: pays nobody. Conversion: an outer tower taken by blue at
#           330,000 (delta = 320,000) -> 520 for blue; the dragon at 340,000 has no team and the monster at 341,000 no known kind: rest.
#   unit 3  f 400,000  a trade, blue 4 kills red 9 (bounty 300) and 2 s later red 8 kills blue 4 (bounty 300): 300 against 300, f0 = 0, a
#           draw; a tower taken by blue whose tier is unknown, at 405,000, is attributed to it and is worth nothing (no price).
#   unit 4  f 500,000  blue 2 kills red 7 (bounty 300): 300; an inner tower taken by blue at 505,000 is attributed: 350 (TOY_TOWER_U).
#   unit 5  f 680,000  blue 5 kills red 10 (bounty 300): 300; after T - 30 s = 670,000, so out of scope. A plate taken by blue at 600,000
#           (the conversion window of unit 4 ends at 580,000) is rest.
# Match 1: unit 6 f 60,000 (out of scope) red 6 kills blue 1 (level 2) with four assisters, bounty 500, before 90 s: pool = 150 x 0.5
#   = 75, gold 575; unit 7 f 210,000 blue 3 kills red 6 (level 5) with four assisters, bounty 200 + shutdown 200, at 210 s (m = 1): the
#   pool is min(100, 150) = 100 (not 150, which a pool over bounty + shutdown would give), gold 500,
#   the baron taken by blue at 215,000 is attributed (1,830); unit 8 f 300,000 red 9 kills blue 3 (level 12, base 360) with one assister,
#   bounty 450 + shutdown 300: pool = min(225, 180) = 180, gold 930 (red), and 2 s later blue 1 kills red 9 (bounty 300): f0 = 300 - 930
#   = -630; the elder taken by red at 330,000 is converted (1,520 for red).
# Match 2: a dragon taken by blue at 150,000 belongs to no unit (rest).
# Match 3: one blue kill with an assister and an attributed inner tower taken by blue.
TOY_PRICES = {'dragon': 410.0, 'elder': 1520.0, 'grubs': 260.0, 'herald': 630.0, 'baron': 1830.0, 'atakhan': 710.0, 'tower_outer': 520.0,
              'tower_inner': 350.0, 'tower_base': 310.0, 'tower_nexus': 330.0, 'inhibitor': 450.0, 'plate': 160.0}
if tuple(TOY_PRICES) != GROUP_NAMES or len(set(TOY_PRICES.values())) != 12:
    raise RuntimeError('the toy prices are one number for each of the 12 groups, all different')


def value_kill(t, x, y, killer, victim, assisters=(), bounty=300, shutdown=0) -> dict:
    """A CHAMPION_KILL for ``synth.make_record`` (killer 0: an execution)."""
    return dict(t=int(t), x=x, y=y, killer=int(killer), victim=int(victim), assists=tuple(int(a) for a in assisters), bounty=int(bounty),
                shutdown=int(shutdown))


def value_obj(t, name, taker, killer=0, assisters=(), at=None) -> dict:
    """An objective for ``synth.make_record``: ``name`` as in ``KIND_SUB``, ``taker`` BLUE | RED | NOBODY, the credited champions (killer 0:
    none) and the position ``at`` = (x, y) (None: the event has no position)."""
    typ, monster, sub, building, tower = _RAW[name]
    elite = typ == 'ELITE_MONSTER_KILL'
    team = None if taker == NOBODY else (TEAM_BLUE + 100 * taker if elite else TEAM_RED - 100 * taker)     # a building: the owner
    x, y = (None, None) if at is None else at
    return dict(t=int(t), type=typ, x=x, y=y, killer=int(killer), assists=tuple(int(a) for a in assisters), team=team, monster=monster,
                sub=sub, building=building, tower=tower, lane=None if elite else 'MID_LANE')


def level_up(pid, level, t) -> dict:
    """A raw LEVEL_UP event."""
    return {'type': 'LEVEL_UP', 'timestamp': int(t), 'participantId': int(pid), 'level': int(level)}


def value_record_0(key, patch):
    kills = [value_kill(135_000, 12_000, 12_000, 4, 9, (5,), 300),
             value_kill(150_000, 5_000, 5_000, 1, 6, (2, 3), 400),
             value_kill(155_000, 5_100, 5_050, 7, 2, (), 300, 150),
             value_kill(165_000, 5_200, 5_100, 3, 8, (1,), 500),
             value_kill(300_000, 8_000, 8_000, 0, 9, (1,), 300),
             value_kill(400_000, 3_000, 12_000, 4, 9, (), 300),
             value_kill(402_000, 3_100, 12_050, 8, 4, (), 300),
             value_kill(500_000, 10_000, 4_000, 2, 7, (), 300),
             value_kill(680_000, 7_000, 7_000, 5, 10, (), 300)]
    objectives = [value_obj(166_000, 'dragon', BLUE, 1, (3,), (5_150, 5_050)),
                  value_obj(167_000, 'dragon', RED, 7, (), None),
                  value_obj(168_000, 'tower_inner', RED, 7, (), (5_000, 5_100)),
                  value_obj(190_000, 'herald', BLUE, 0, (), (5_000, 5_000)),
                  value_obj(200_000, 'plate', RED, 8, (), (5_000, 5_200)),
                  value_obj(330_000, 'tower_outer', BLUE, 0, (), (8_000, 8_000)),
                  value_obj(340_000, 'dragon', NOBODY, 0, (), (8_000, 8_000)),
                  value_obj(341_000, 'unknown_kind', BLUE, 0, (), (8_000, 8_000)),
                  value_obj(405_000, 'tower_unknown_tier', BLUE, 4, (), (3_050, 12_000)),
                  value_obj(505_000, 'tower_inner', BLUE, 2, (), (10_000, 4_050)),
                  value_obj(600_000, 'plate', BLUE, 0, (), (9_000, 9_000))]
    # the level of each victim at its kill is the highest LEVEL_UP level at or before the kill ms: 9 -> 3 at 135,000, 6 -> 7 at 150,000
    # (the LEVEL_UP to 7 is on the kill's own ms; the one to 8 is one ms later and does not count), 2 -> 9, 8 -> 18, 9 -> 5 at 300,000
    # and at 400,000, 4 -> 11, 7 -> 8, 10 -> 13; the LEVEL_UP of player 1 is on a player that is never a victim here
    events = [level_up(9, 3, 60_000), level_up(8, 17, 100_000), level_up(6, 6, 100_000), level_up(1, 18, 100_000), level_up(2, 9, 120_000),
              level_up(6, 7, 150_000), level_up(6, 8, 150_001), level_up(8, 18, 160_000), level_up(9, 5, 250_000), level_up(4, 11, 280_000),
              level_up(7, 8, 400_000), level_up(10, 13, 600_000)]
    return synth.make_record(T_ms=700_000, kills=kills, objectives=objectives, events=events, patch=patch, key=key)


def value_record_1(key, patch):
    kills = [value_kill(60_000, 2_000, 2_000, 6, 1, (7, 8, 9, 10), 500),
             value_kill(210_000, 6_000, 6_000, 3, 6, (1, 2, 4, 5), 200, 200),
             value_kill(300_000, 9_000, 9_000, 9, 3, (6,), 450, 300),
             value_kill(302_000, 9_100, 9_050, 1, 9, (), 300)]
    objectives = [value_obj(215_000, 'baron', BLUE, 3, (1,), (6_050, 6_000)),
                  value_obj(330_000, 'elder', RED, 0, (), (9_000, 9_000))]
    events = [level_up(1, 2, 50_000), level_up(6, 5, 100_000), level_up(3, 12, 200_000), level_up(9, 4, 250_000)]
    return synth.make_record(T_ms=520_000, kills=kills, objectives=objectives, events=events, patch=patch, key=key)


def value_record_2(key, patch):
    return synth.make_record(T_ms=400_000, objectives=[value_obj(150_000, 'dragon', BLUE, 1, (), (5_000, 5_000))], patch=patch, key=key)


def value_record_3(key, patch):
    kills = [value_kill(150_000, 5_000, 5_000, 2, 7, (1,), 300)]
    objectives = [value_obj(155_000, 'tower_inner', BLUE, 2, (), (5_050, 5_000))]
    return synth.make_record(T_ms=300_000, kills=kills, objectives=objectives, patch=patch, key=key)


VALUE_TOY_MATCHES = (('FIT', '15.18', value_record_0), ('STOP', '15.20', value_record_1), ('DEV', '15.21', value_record_2),
                     ('CAL', '15.19', value_record_3))
_VALUE_BY_KEY = {toy_key(i): builder for i, (_, _, builder) in enumerate(VALUE_TOY_MATCHES)}


def value_toy_matches() -> list:
    """The ordered match list [(patch, key, macro), ...] of ``VALUE_TOY_MATCHES`` that ``run_s1.build_store`` takes."""
    return [(patch, toy_key(i), role) for i, (role, patch, _) in enumerate(VALUE_TOY_MATCHES)]


def value_toy_records(key, patch):
    """The ``records`` argument of ``run_s1.build_store`` and ``value.kill_levels`` for the toy matches (module level: workers import
    it)."""
    return _VALUE_BY_KEY[key](key, patch)


def value_toy_records_that_leak_a_key(key, patch):
    """Like ``value_toy_records``, but the file of the match 1 raises a KeyError whose message is the match key (module level: workers
    import it). A caller must not let the key out."""
    if key == toy_key(1):
        raise KeyError(key)
    return value_toy_records(key, patch)


def build_value_s1(root, matches=None, records=None, work=None) -> Store:
    """The S1 store of the value toys (default) or of any ``matches`` list with its ``records`` function, made by the real extraction
    (``run_s1.build_store``, one worker, E6 answered with an empty process list) under ``root``, loaded and verified. The block files go
    to ``work`` (default: a folder next to ``root``)."""
    from c_ept.workflows import run_s1
    from c_ept.tests.conftest import empty_process_list
    matches = value_toy_matches() if matches is None else list(matches)
    records = value_toy_records if records is None else records
    root = Path(root)
    work = root.parent / (root.name + '_blocks') if work is None else Path(work)
    with empty_process_list():
        run_s1.build_store(matches, root, work, workers=1, records=records, role_of=synth.roles_of(matches))
    return run_s1.load_s1(root)


def toy_record_with_level_up_at_kill(key=None, patch='15.18'):
    """One kill of participant 6 at 200,000 ms. Player 6 reaches level 3 at 100,000, level 4 at 200,000 (the kill's own ms: it counts) and
    level 5 at 200,001 (one ms later: it does not); player 1, the killer, reaches level 9 at 150,000 (not the victim's)."""
    events = [level_up(6, 3, 100_000), level_up(6, 4, 200_000), level_up(6, 5, 200_001), level_up(1, 9, 150_000)]
    return synth.make_record(T_ms=600_000, kills=[value_kill(200_000, 5_000, 5_000, 1, 6)], events=events, patch=patch,
                             key=toy_key(9_000) if key is None else key)


TOY_LEVEL_AFTER_UP = 4
# the victim level of every kill of the value toys, in S1 kill order (match 0: 9 kills, match 1: 4, match 2: none, match 3: 1); -1 where
# ``value.kill_levels`` is not asked for the role (CAL)
TOY_LEVELS = np.array([3, 7, 9, 18, 5, 5, 11, 8, 13, 2, 5, 12, 4, -1], dtype=np.int16)
TOY_N_KILLS, TOY_N_UNITS, TOY_N_ROWS, TOY_N_OBJECTIVES = 14, 10, 25, 15
TOY_ROW_OFF = (0, 10, 17, 22, 25)                       # frame rows of each match (k = 2 .. 11, 2 .. 8, 2 .. 6, 2 .. 4)
TOY_U, TOY_F0 = 1, 796.875
TOY_EXEC_U, TOY_DRAW_U, TOY_TOWER_U = 2, 3, 4
TOY_VALUED_UNITS = tuple(range(9))                      # units of the matches 0-2; unit 9 is in the CAL match
# per unit: kill gold, objective value and conversion value (blue, red), champion kills (blue, red), gold bin (-1: no kills), winner
TOY_UNIT_VALUES = {
    0: dict(kill_gold=(403.125, 0.0), obj_value=(0.0, 0.0), conv_value=(0.0, 410.0), n_kills=(1, 0), gold_bin=(4, -1), winner=1),
    1: dict(kill_gold=(1186.875, 450.0), obj_value=(410.0, 350.0), conv_value=(630.0, 160.0), n_kills=(2, 1), gold_bin=(11, 4), winner=1),
    2: dict(kill_gold=(0.0, 0.0), obj_value=(0.0, 0.0), conv_value=(520.0, 0.0), n_kills=(0, 0), gold_bin=(-1, -1), winner=0),
    3: dict(kill_gold=(300.0, 300.0), obj_value=(0.0, 0.0), conv_value=(0.0, 0.0), n_kills=(1, 1), gold_bin=(3, 3), winner=0),
    4: dict(kill_gold=(300.0, 0.0), obj_value=(350.0, 0.0), conv_value=(0.0, 0.0), n_kills=(1, 0), gold_bin=(3, -1), winner=1),
    5: dict(kill_gold=(300.0, 0.0), obj_value=(0.0, 0.0), conv_value=(0.0, 0.0), n_kills=(1, 0), gold_bin=(3, -1), winner=1),
    6: dict(kill_gold=(0.0, 575.0), obj_value=(0.0, 0.0), conv_value=(0.0, 0.0), n_kills=(0, 1), gold_bin=(-1, 5), winner=-1),
    7: dict(kill_gold=(500.0, 0.0), obj_value=(1830.0, 0.0), conv_value=(0.0, 0.0), n_kills=(1, 0), gold_bin=(5, -1), winner=1),
    8: dict(kill_gold=(300.0, 930.0), obj_value=(0.0, 0.0), conv_value=(0.0, 1520.0), n_kills=(1, 1), gold_bin=(3, 9), winner=-1),
}
# f0 of the unit = (kill gold + objective value) of blue - that of red; f0 of the conversions = conversion value of blue - of red
TOY_F0_UNIT = (403.125, 796.875, 0.0, 0.0, 650.0, 300.0, -575.0, 2330.0, -630.0)
TOY_F0_CONV = (-410.0, 470.0, 520.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1520.0)
TOY_SURE = (True, True, False, False, True, True, True, True, True)           # |f0 of the unit| >= 150
# V_start of the 25 frame rows for H = 60, 120 and 180 s: the units with f in (t0, min(t0 + H, hi)] that are in scope, f0(unit) + f0
# (conversions) each (unit 0: -6.875, unit 1: 1,266.875, unit 2: 520, unit 3: 0, unit 4: 650, unit 7: 2,330, unit 8: -2,150; unit 5 and 6
# are out of scope); the three rows of the CAL match are NaN
TOY_V_START = np.array([
    # match 0: t0 = 120,300 + 60,000 (k - 2), hi = min(t0 + 180,000, 670,000)
    [1260.0, 1260.0, 1780.0], [0.0, 520.0, 520.0], [520.0, 520.0, 520.0], [0.0, 0.0, 0.0], [0.0, 0.0, 650.0],
    [0.0, 650.0, 650.0], [650.0, 650.0, 650.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
    # match 1: hi = min(t0 + 180,000, 490,000)
    [0.0, 2330.0, 180.0], [2330.0, 180.0, 180.0], [-2150.0, -2150.0, -2150.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    # match 2: no unit
    [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
    # match 3 (CAL)
    [np.nan] * 3, [np.nan] * 3, [np.nan] * 3])
# the objectives of the valued matches (0-2): 14, of them 5 attributed, 5 converted and 4 in neither place (3 in match 0: the dragon
# without a team, the monster of unknown kind and the plate after every window; 1 in match 2); 2 have no group (the monster of unknown
# kind and the tower without a tier), 1 has no taker
TOY_OBJECTIVE_COUNTS = dict(all=14, attributed=5, converted=5, rest=4, unknown_group=2, unknown_taker=1)
# the champion kills of the valued matches: 12 (and one execution); the gold of the kills and the value of the objectives by team
# (blue, red)
TOY_KILL_COUNTS = dict(champion=12, executions=1, all=13)
TOY_KILL_GOLD_TOTAL = (3290.0, 2255.0)
TOY_OBJ_VALUE_TOTAL = dict(all=(4310.0, 2440.0), attributed=(2590.0, 350.0), converted=(1150.0, 2090.0), rest=(570.0, 0.0))


@pytest.fixture(scope='module')
def toy_prices():
    """The prices of the toy groups (a copy of ``TOY_PRICES``)."""
    return dict(TOY_PRICES)


@pytest.fixture(scope='module')
def toy_rules():
    """The built-in rules of decision P3."""
    from c_ept.valuation import rulepay
    return rulepay.P3_RULES


@pytest.fixture(scope='module')
def toy_value_s1(tmp_path_factory):
    """The S1 store of ``VALUE_TOY_MATCHES`` made by the real extraction, written once per test module and read only."""
    folder = tmp_path_factory.mktemp('toy_value_s1')
    s = build_value_s1(folder / 's1')
    yield s
    del s
    gc.collect()                                   # Windows does not delete a folder with open memory maps
    shutil.rmtree(folder, ignore_errors=True)


@pytest.fixture(scope='module')
def toy_uv(toy_value_s1, toy_prices, toy_rules):
    """``value.unit_values`` of the value toys with ``TOY_LEVELS``."""
    from c_ept.valuation import value
    return value.unit_values(toy_value_s1, toy_prices, toy_rules, TOY_LEVELS)


# ---- random matches with LEVEL_UP events, for the brute-force comparisons of ``value`` ---------------------------------------------------
RANDOM_ROLES = (('FIT', '15.18'), ('FIT', '15.19'), ('STOP', '15.20'), ('DEV', '15.21'), ('CAL', '15.19'), ('AUDIT', '15.21'))


def random_value_matches(n: int = 12, first: int = 1000) -> list:
    """The match list [(patch, key, macro)] of ``n`` random matches: the roles run through ``RANDOM_ROLES`` (FIT, FIT, STOP, DEV, CAL,
    AUDIT); the keys are ``toy_key(first ...)``, whose digits are the seed of the match."""
    return [(RANDOM_ROLES[i % len(RANDOM_ROLES)][1], toy_key(first + i), RANDOM_ROLES[i % len(RANDOM_ROLES)][0]) for i in range(n)]


def random_value_record(key, patch):
    """A random synthetic match (``synth.records_from_seed``) with LEVEL_UP events: every player gets a random increasing level path, and
    about 40% of the deaths of a player are also the ms of one of its LEVEL_UPs (the same-ms rule)."""
    rec = synth.records_from_seed(key, patch)
    rng = np.random.default_rng(int(re.sub(r'\D', '', key)) + 777)
    deaths = [e for e in rec.events if e['type'] == 'CHAMPION_KILL']
    ups = []
    for pid in range(1, 11):
        times = {int(t) for t in rng.integers(20_000, rec.T_ms, size=int(rng.integers(0, 15)))}
        times |= {e['timestamp'] for e in deaths if e['victimId'] == pid and rng.random() < 0.4}
        ups += [level_up(pid, min(18, 2 + j), t) for j, t in enumerate(sorted(times))]
    body = sorted(list(rec.events[:-1]) + ups, key=lambda e: e['timestamp'])                       # stable: a kill stays before a LEVEL_UP
    return dataclasses.replace(rec, events=tuple(body) + (rec.events[-1],))


# ---- the feature arrays of the toy S2 (stage S3 Task 5, ``c_ept.features``) --------------------------------------------------------------
# The default toy S2 (``build_toy_s2(root, toy_s1)``) has 49 views: per match its frame-clock views (one per S1 frame row: 10, 8, 7, 8, 0
# and 4 for the six toy matches, the differences of ``matches/row_off``), then two event views. The view numbers are
#
#     match 0  FIT   frame views  0-9    event views 10, 11
#     match 1  FIT   frame views 12-19   event views 20, 21
#     match 2  STOP  frame views 22-28   event views 29, 30
#     match 3  DEV   frame views 31-38   event views 39, 40
#     match 4  FIT   (no frame row)      event views 41, 42
#     match 5  FIT   frame views 43-46   event views 47, 48
#
# 37 frame-clock views, 12 event views, and 22 + 8 = 30 views of FIT matches (0-21 and 41-48).
TOY_N_VIEWS = 49
TOY_ROWS_PER_MATCH = (10, 8, 7, 8, 0, 4)
TOY_FRAME_VIEWS = (*range(0, 10), *range(12, 20), *range(22, 29), *range(31, 39), *range(43, 47))
TOY_EVENT_VIEWS = (10, 11, 20, 21, 29, 30, 39, 40, 41, 42, 47, 48)
TOY_FIT_VIEWS = (*range(0, 22), *range(41, 49))
TOY_VIEWS = np.array(TOY_FRAME_VIEWS, dtype=np.int64)          # "the toy views" of the tests of ``features``: the t0 rows, no NaN in b0
TOY_ALL_VIEWS = np.arange(TOY_N_VIEWS)
# summary columns and a player column that the toy makes constant (the other columns of the toy vary; ``t_s`` varies too but is always
# dropped by name):
TOY_CONSTANT_COLUMN = 'rift_element_known'              # JUNK in every view, event views included
TOY_FIT_CONSTANT_COLUMN = 'rift_element_air'            # JUNK in the views of FIT matches, 5 in the frame views of the other matches
TOY_CONSTANT_PLAYER_COLUMN = 'flag_stasis'              # 0 for every player in every view
TOY_DROPPED_ALL = ('t_s', TOY_CONSTANT_COLUMN, TOY_CONSTANT_PLAYER_COLUMN)                            # over all 49 views
TOY_DROPPED_FIT = ('t_s', TOY_CONSTANT_COLUMN, TOY_FIT_CONSTANT_COLUMN, TOY_CONSTANT_PLAYER_COLUMN)   # over the 30 FIT views
# positions that the toy leaves out: view -> the players (0-4 blue, 5-9 red) without a position in it. Their x, y and wall_fraction are NaN
# and so are the row and the column of ``geo_pp``. In 3, 14 and 25 a few players are missing and the cross-team block keeps 20, 20 and 16
# pairs; in 10, 29, 41 and 42 a whole team is missing and no cross-team pair is left (b0 has NaN in its two geodesic columns there).
TOY_MISSING = {3: (2,), 14: (7,), 25: (0, 5), 10: (0, 1, 2, 3, 4), 29: (0, 1, 2, 3, 4), 41: (5, 6, 7, 8, 9), 42: tuple(range(10))}
TOY_NAN_VIEWS = (10, 29, 41, 42)
TOY_NAN_FIT_VIEWS = (10, 41, 42)                        # the NaN views of FIT matches (view 29 is in the STOP match)
# geo_pp[v, i, j] = |i - j| x 100 + (v mod 5) x 10 for i != j (0 on the diagonal), exact in float16. The cross-team block (i = 0-4 blue,
# j = 5-9 red) has the gaps d = j - i = 1 .. 9 with 1, 2, 3, 4, 5, 4, 3, 2, 1 pairs: the sum of d over the 25 pairs is
# 1 + 4 + 9 + 16 + 25 + 24 + 21 + 16 + 9 = 125, so the mean gap is 5 and the smallest is 1 (i = 4, j = 5). The value of a pair is
# 100 d + 10 (v mod 5), so with all 25 pairs: minimum 100 + 10 (v mod 5) and mean 500 + 10 (v mod 5).
#   view 0, nobody missing:        min 100, mean 500
#   view 3, player 2 missing:      its 5 pairs (j = 5 .. 9, d = j - 2, sum 3 + 4 + 5 + 6 + 7 = 25) go; the smallest gap 1 stays (4, 5):
#                                  min 100 + 30 = 130, mean gap (125 - 25) / 20 = 5: 500 + 30 = 530
#   view 14, player 7 missing:     its 5 pairs (i = 0 .. 4, d = 7 - i, sum 25) go: min 100 + 40 = 140, mean 500 + 40 = 540
#   view 25, players 0 and 5 gone: 16 pairs i = 1 .. 4, j = 6 .. 9; the sum of d is 4 (6 + 7 + 8 + 9) - 4 (1 + 2 + 3 + 4) = 120 - 40 = 80,
#                                  mean gap 80 / 16 = 5, smallest gap 6 - 4 = 2 (v mod 5 = 0): min 200, mean 500
TOY_GEO_HAND = {0: (100.0, 500.0), 3: (130.0, 530.0), 14: (140.0, 540.0), 25: (200.0, 500.0)}


def toy_players(n_views: int) -> np.ndarray:
    """(n_views, 10, 81) float32 ``x/players_num`` of the toy S2: the cell of view v, player p and column c is
    (v (c + 3) + p (c mod 7 + 1) + c) mod 17, a whole number 0 .. 16 (exact in float32), so that every column varies with the player
    (the multiplier of p is 1 .. 7, never a multiple of 17). ``TOY_CONSTANT_PLAYER_COLUMN`` is 0 instead, and the players of
    ``TOY_MISSING`` have NaN in x, y and wall_fraction."""
    from c_ept.preprocessing import tokens
    column = {name: i for i, name in enumerate(tokens.PLAYER_NUM)}
    v, p, c = np.ogrid[:n_views, :10, :len(tokens.PLAYER_NUM)]
    a = ((v * (c + 3) + p * (c % 7 + 1) + c) % 17).astype(np.float32)
    a[:, :, column[TOY_CONSTANT_PLAYER_COLUMN]] = 0.0
    for view, players in TOY_MISSING.items():
        for q in players:
            if view < n_views:
                a[view, q, [column['x'], column['y'], column['wall_fraction']]] = np.nan
    return a


def toy_geo(n_views: int) -> np.ndarray:
    """(n_views, 10, 10) float16 ``x/geo_pp`` of the toy S2: |i - j| x 100 + (v mod 5) x 10 off the diagonal, 0 on it, NaN in the row and
    the column of every player of ``TOY_MISSING``."""
    v, i, j = np.ogrid[:n_views, :10, :10]
    g = np.where(i == j, 0.0, np.abs(i - j) * 100.0 + (v % 5) * 10.0)
    for view, players in TOY_MISSING.items():
        for q in players:
            if view < n_views:
                g[view, q, :] = np.nan
                g[view, :, q] = np.nan
    return g.astype(np.float16)


def toy_feature_summary(row: int, ref_ms: int, fit: bool) -> dict:
    """{summary column: value} that ``build_toy_s2(..., features=True)`` adds to the frame-clock view of the S1 frame row ``row`` (reference
    time ``ref_ms``, a FIT match or not): the game time, the five "is up" flags with different short patterns, ``TOY_CONSTANT_COLUMN`` and
    ``TOY_FIT_CONSTANT_COLUMN``."""
    return {'game_time_s': ref_ms / 1000.0, 'dragon_up': float(row % 2), 'baron_up': float((row + 1) % 3 == 0),
            'herald_up': float(row // 2 % 2), 'atakhan_up': float(row // 3 % 2), 'voidgrub_up': float(row // 4 % 2),
            TOY_CONSTANT_COLUMN: JUNK, TOY_FIT_CONSTANT_COLUMN: JUNK if fit else 5.0}


def blank_feature_arrays(n: int) -> tuple:
    """(summary (n, 107) f4, players_num (n, 10, 81) f4, geo_pp (n, 10, 10) f2), all zero: the starting point of a hand-made store."""
    from c_ept.preprocessing import tokens
    return (np.zeros((n, len(tokens.SUMMARY)), np.float32), np.zeros((n, 10, len(tokens.PLAYER_NUM)), np.float32),
            np.zeros((n, 10, 10), np.float16))


def build_feature_store(root, summary, players_num, geo_pp) -> Store:
    """A complete store of the three arrays that the baseline features read (``x/summary``, ``x/players_num``, ``x/geo_pp``) and nothing
    else, for the edge cases that the shared toy S2 does not have."""
    s = Store(root)
    _put(s, 'x/summary', summary, 'f4')
    _put(s, 'x/players_num', players_num, 'f4')
    _put(s, 'x/geo_pp', geo_pp, 'f2')
    s.finalize()
    return s


# ---- the heads world: S1, S2 and label arrays for the baseline heads (stage S3 Task 6) --------------------------------------------------
# A small random world with the layout and the dtypes of the real stores, holding only the arrays that the head code reads. It is made
# from made-up units by the real functions of ``frame_rows`` (rows, exposure, onsets) and ``event_rows`` (the M3 and M4 views), so that the
# links between the tables are the real ones:
#
#   S1      matches/{key, role, T_ms}, frame_rows/{match, k, t0, hi, exposure_ms}, onsets/{row, bin, mark, unit},
#           units/{match, kill0, n_kills, f, delta, size, mark, in_scope, K, O, conv, conv_exposure_ms}, kills/{x, y, pos_missing},
#           event_rows/{unit, kind, ref, frame_k, t0, frame_row}
#   S2      views/{src, src_index, match, ref, t0, frame_k}, x/summary, x/players_num, x/geo_pp. The views are in a scrambled order and
#           the event views' ``src_index`` numbers collide with the frame-row numbers, so a mapping that ignores ``src`` or assumes an
#           order picks a wrong view; every view has its own random features, so a wrong view shows in the features
#   labels  labels/{kill_gold, gold_bin}
#
# Eight matches with the roles of WORLD_ROLES. Built in: units before the scope start (no onset, no event view), a unit that runs to the
# game end (an M3 view and no M4 view, conversion exposure 0), a unit with no champion kill (K = 0, no gold), first kills without a
# position, kill gold above 3,000 (the open gold bin), a cell of O and of the conversions that is always 0 (WORLD_ZERO_CELL), views whose
# cross-team distances are all missing (b0 has NaN there), and a CAL match (a role that is not valued: NaN kill gold, gold bin -1).
WORLD_ROLES = ('FIT', 'FIT', 'FIT', 'FIT', 'STOP', 'DEV', 'DEV', 'CAL')
WORLD_SEED = 20261021
WORLD_ZERO_CELL = 2                                    # the cell 2 kind + taker of the elder dragon of the blue team: never happens
N_K_FIELDS = 8                                         # units/K: kills, bounty, shutdown, assists; blue then red


@dataclass(frozen=True, eq=False)
class HeadsWorld:
    """The three stores of the heads world and the b0 FIT means of its S2 (computed over the views of the FIT matches)."""
    s1: Store
    s2: Store
    labels: Store
    fit_views: np.ndarray
    fit_means: np.ndarray


def _world_match(rng, role: str) -> dict:
    """The arrays of one made-up match with local indices (a unit, row, kill or event number counts inside the match)."""
    from c_ept.data import event_rows as event_rows_mod
    from c_ept.valuation import value
    T = int(rng.integers(5_600, 9_000)) * 100
    ts = np.array(frame_times((T - 301) // 60_000 + 1), dtype=np.int64)
    rec = types.SimpleNamespace(frame_ts=ts, T_ms=T)
    rows = frame_rows_mod.frame_rows(rec)
    n = int(rng.integers(8, 13))
    f = np.sort(rng.integers(60_000, T - 4_000, n)).astype(np.int64)
    in_scope = (f >= C.SCOPE_FIRST_MS) & (f <= T - C.END_CUT_MS)
    scope = np.flatnonzero(in_scope)
    last = f + rng.integers(0, 9_000, n)
    if len(scope):
        last[scope[-1]] = T + 1_000                    # the last unit in scope runs to the game end: no M4 view, no conversion window
    delta = np.maximum(last + 1, f + C.DELTA_AFTER_F_MS)
    kb, kr = rng.integers(0, 9, n), rng.integers(0, 9, n)             # up to 8 kills: the layers 1 .. 6 and the cap at 6 all occur
    if len(scope) > 2:
        kb[scope[1]] = kr[scope[1]] = 0                # a unit with no champion kill: an execution only
    n_kills = np.maximum(kb + kr + rng.integers(0, 2, n), 1)
    K = rng.integers(0, 900, (n, N_K_FIELDS))
    K[:, 0], K[:, 1] = kb, kr
    O = rng.poisson(0.12, (n, 2 * len(C.O_KINDS)))
    exposure = np.maximum(np.minimum(delta + C.CONV_MS, T) - delta, 0)
    conv = rng.poisson(0.1, O.shape) * (exposure > 0)[:, None]
    O[:, WORLD_ZERO_CELL] = conv[:, WORLD_ZERO_CELL] = 0
    kill0 = np.cumsum(n_kills) - n_kills
    x, y = rng.uniform(0.0, 14_875.0, n_kills.sum()), rng.uniform(0.0, 14_875.0, n_kills.sum())
    missing = np.zeros(len(x), dtype=bool)
    missing[kill0[rng.random(n) < 0.25]] = True        # a quarter of the first kills have no position
    x[missing] = y[missing] = np.nan
    mark = np.minimum(rng.geometric(0.1, n) - 1, 51)
    on = frame_rows_mod.onsets(rows, types.SimpleNamespace(f=f, mark=mark))
    ev = event_rows_mod.event_rows(rec, types.SimpleNamespace(in_scope=in_scope, f=f, delta=delta), rows)
    kills = np.stack([kb, kr], axis=1)
    kill_gold = np.where(kills > 0, 5.0 * np.round(kills * rng.uniform(120.0, 520.0, (n, 2)) / 5.0) + 2_500.0 * (rng.random((n, 2)) < 0.15),
                         0.0)
    valued = role != 'CAL'
    gold_bin = np.where((kills > 0) & valued, value.gold_bin(kill_gold), -1)
    return dict(role=role, T=T, rows=rows, on=on, ev=ev, n=n, f=f, delta=delta, in_scope=in_scope, K=K, O=O, conv=conv, exposure=exposure,
                kill0=kill0, n_kills=n_kills, x=x, y=y, missing=missing, mark=mark, size=rng.integers(0, 4, n),
                kill_gold=np.where(valued, kill_gold, np.nan), gold_bin=gold_bin)


def build_heads_world(root, seed: int = WORLD_SEED, roles=WORLD_ROLES) -> HeadsWorld:
    """The S1, S2 and label stores of the heads world under ``root`` (three folders), complete; see the section text above."""
    from c_ept.preprocessing import features, tokens
    rng = np.random.default_rng(seed)
    root = Path(root)
    cols = collections.defaultdict(list)
    base = dict(row=0, unit=0, kill=0, event=0)
    for m, role in enumerate(roles):
        p = _world_match(rng, role)
        rows, on, ev = p['rows'], p['on'], p['ev']
        nr, nu, nk, ne = len(rows.k), p['n'], int(p['n_kills'].sum()), len(ev.kind)
        for name, value_ in (
                ('matches/key', toy_key(m).encode('ascii')), ('matches/role', C.ROLES.index(role)), ('matches/T_ms', p['T'])):
            cols[name].append(np.array([value_]))
        for name, value_ in (
                ('frame_rows/match', np.full(nr, m)), ('frame_rows/k', rows.k), ('frame_rows/t0', rows.t0), ('frame_rows/hi', rows.hi),
                ('frame_rows/exposure_ms', rows.exposure_ms), ('onsets/row', on.row + base['row']), ('onsets/bin', on.bin),
                ('onsets/mark', on.mark), ('onsets/unit', on.unit + base['unit']), ('units/match', np.full(nu, m)),
                ('units/kill0', p['kill0'] + base['kill']), ('units/n_kills', p['n_kills']), ('units/f', p['f']),
                ('units/delta', p['delta']),
                ('units/size', p['size']), ('units/mark', p['mark']), ('units/in_scope', p['in_scope']), ('units/K', p['K']),
                ('units/O', p['O']), ('units/conv', p['conv']), ('units/conv_exposure_ms', p['exposure']), ('kills/x', p['x']),
                ('kills/y', p['y']), ('kills/pos_missing', p['missing']), ('event_rows/unit', ev.unit + base['unit']),
                ('event_rows/kind', ev.kind), ('event_rows/ref', ev.ref), ('event_rows/frame_k', ev.frame_k), ('event_rows/t0', ev.t0),
                ('event_rows/frame_row', ev.frame_row), ('labels/kill_gold', p['kill_gold']), ('labels/gold_bin', p['gold_bin'])):
            cols[name].append(np.asarray(value_))
        base['row'] += nr
        base['unit'] += nu
        base['kill'] += nk
        base['event'] += ne
    arrays = {name: np.concatenate(parts) for name, parts in cols.items()}
    dtypes = {'matches/key': 'S24', 'matches/role': 'i1', 'matches/T_ms': 'i8', 'frame_rows/match': 'i4', 'frame_rows/k': 'i2',
              'frame_rows/t0': 'i8', 'frame_rows/hi': 'i8', 'frame_rows/exposure_ms': 'i4', 'onsets/row': 'i4', 'onsets/bin': 'i1',
              'onsets/mark': 'i2', 'onsets/unit': 'i4', 'units/match': 'i4', 'units/kill0': 'i4', 'units/n_kills': 'i2', 'units/f': 'i8',
              'units/delta': 'i8', 'units/size': 'i1', 'units/mark': 'i2', 'units/in_scope': '?', 'units/K': 'i4', 'units/O': 'i2',
              'units/conv': 'i2', 'units/conv_exposure_ms': 'i4', 'kills/x': 'f8', 'kills/y': 'f8', 'kills/pos_missing': '?',
              'event_rows/unit': 'i4', 'event_rows/kind': 'i1', 'event_rows/ref': 'i8', 'event_rows/frame_k': 'i2', 'event_rows/t0': 'i8',
              'event_rows/frame_row': 'i4', 'labels/kill_gold': 'f8', 'labels/gold_bin': 'i1'}
    s1, labels = Store(root / 's1'), Store(root / 'labels')
    for name, dtype in dtypes.items():
        _put(labels if name.startswith('labels/') else s1, name, arrays[name], dtype)
    s1.finalize()
    labels.finalize()

    # the views: the frame-clock views of all rows, then the event views of all event rows, scrambled together
    n_frame, n_event = len(arrays['frame_rows/t0']), len(arrays['event_rows/kind'])
    views = {
        'src': np.concatenate([np.zeros(n_frame), arrays['event_rows/kind']]),
        'src_index': np.concatenate([np.arange(n_frame), np.arange(n_event)]),
        'match': np.concatenate([arrays['frame_rows/match'], arrays['units/match'][arrays['event_rows/unit']]]),
        'ref': np.concatenate([arrays['frame_rows/t0'], arrays['event_rows/ref']]),
        't0': np.concatenate([arrays['frame_rows/t0'], arrays['event_rows/t0']]),
        'frame_k': np.concatenate([arrays['frame_rows/k'], arrays['event_rows/frame_k']])}
    n_views = n_frame + n_event
    order = rng.permutation(n_views)
    s2 = Store(root / 's2')
    for name, dtype in (('src', 'i1'), ('src_index', 'i4'), ('match', 'i4'), ('ref', 'i8'), ('t0', 'i8'), ('frame_k', 'i2')):
        views[name] = views[name][order]
        _put(s2, f'views/{name}', views[name], dtype)
    summary = rng.integers(0, 6, (n_views, len(tokens.SUMMARY))).astype(np.float32)
    summary[:, tokens.SUMMARY.index('game_time_s')] = views['ref'] / 1000.0
    players = rng.normal(5.0, 3.0, (n_views, 10, len(tokens.PLAYER_NUM))).astype(np.float32)
    geo = (rng.integers(1, 90, (n_views, 10, 10)) * 100).astype(np.float16)
    geo[:, np.arange(10), np.arange(10)] = 0
    for view in rng.permutation(n_views)[:3]:           # three views whose cross-team distances are all missing: b0 has NaN there
        geo[view, :5, 5:] = np.nan
        geo[view, 5:, :5] = np.nan
    _put(s2, 'x/summary', summary, 'f4')
    _put(s2, 'x/players_num', players, 'f4')
    _put(s2, 'x/geo_pp', geo, 'f2')
    s2.finalize()
    fit = np.flatnonzero(features.split_views(s1, s2, 'FIT'))
    return HeadsWorld(s1=s1, s2=s2, labels=labels, fit_views=fit, fit_means=features.b0_fit_means(s2, fit))


@pytest.fixture(scope='module')
def heads_world(tmp_path_factory):
    """The heads world (``build_heads_world``), written once per test module and read only."""
    folder = tmp_path_factory.mktemp('heads_world')
    world = build_heads_world(folder)
    yield world
    del world
    gc.collect()                                   # Windows does not delete a folder with open memory maps
    shutil.rmtree(folder, ignore_errors=True)


class HeadsOracle:
    """The arrays of the heads world as numpy arrays, and the answers 'which view', 'which onsets', 'which units', 'which zone' found by
    plain loops written from the text of the rules (never from the code under test). The tests of the rows of B0 and B1 compare the code
    with it. ``view(src, index)`` is the one S2 view with that ``views/src`` and ``views/src_index`` (the event views have colliding
    ``src_index`` numbers); ``event_row(u, kind)`` the event row of a unit of that kind or None."""

    def __init__(self, w):
        g = lambda n: np.asarray(w.s1.array(n))                                                       # noqa: E731
        self.w = w
        self.role, self.fr_match, self.expo = g('matches/role'), g('frame_rows/match'), g('frame_rows/exposure_ms')
        self.on_row, self.on_bin, self.on_unit = g('onsets/row'), g('onsets/bin'), g('onsets/unit')
        self.u_match, self.mark, self.size, self.K, self.O = g('units/match'), g('units/mark'), g('units/size'), g('units/K'), g('units/O')
        self.conv, self.cexp, self.in_scope = g('units/conv'), g('units/conv_exposure_ms'), g('units/in_scope')
        self.kill0 = g('units/kill0')
        self.kx, self.ky, self.kmiss = g('kills/x'), g('kills/y'), g('kills/pos_missing')
        self.ev_unit, self.ev_kind = g('event_rows/unit'), g('event_rows/kind')
        self.src, self.src_index = np.asarray(w.s2.array('views/src')), np.asarray(w.s2.array('views/src_index'))
        self.gold_bin, self.kill_gold = np.asarray(w.labels.array('labels/gold_bin')), np.asarray(w.labels.array('labels/kill_gold'))

    def role_of_match(self, m):
        return C.ROLES[int(self.role[m])]

    def view(self, src, index):
        hits = [v for v in range(len(self.src)) if self.src[v] == src and self.src_index[v] == index]
        assert len(hits) == 1, (src, index, hits)
        return hits[0]

    def zone(self, u):
        from c_ept.preprocessing import terrain
        k = self.kill0[u]
        return -1 if self.kmiss[k] else int(terrain.zone_of(self.kx[k], self.ky[k]))

    def onsets(self, role):
        return [i for i in range(len(self.on_row)) if self.role_of_match(self.fr_match[self.on_row[i]]) == role]

    def frame_rows(self, role):
        return [r for r in range(len(self.fr_match)) if self.role_of_match(self.fr_match[r]) == role]

    def units(self, role):
        """The units of the matches of ``role`` that are in scope (the units of the open heads)."""
        return [u for u in range(len(self.u_match)) if self.role_of_match(self.u_match[u]) == role and self.in_scope[u]]

    def event_row(self, u, kind):
        hits = [e for e in range(len(self.ev_unit)) if self.ev_unit[e] == u and self.ev_kind[e] == kind]
        assert len(hits) <= 1
        return hits[0] if hits else None

    def kblock(self, u):
        """kills blue, kills red, kill gold blue and red in thousands: the K columns of the masked M4 input."""
        return np.array([self.K[u, 0], self.K[u, 1], self.kill_gold[u, 0] / 1000.0, self.kill_gold[u, 1] / 1000.0])


# ---- the onset toy of B2 (stage S3 Task 7) -------------------------------------------------------------------------------------------
# Three matches (FIT, STOP, DEV) with 6, 3 and 2 frame rows, eleven in all; the bins of the onsets of every row are written out (the
# real bin of an onset is (f - t0 - 1) // 10 s, 0 .. 17). A row is positive for B2 when it has an onset in the bins 0 .. 5 (the first
# kill within 60 s: decision P5). By hand, row by row (the first bin, or 18 = "no onset"; positive when it is below 6):
#   row 0  bins 3, 9    first 3   positive
#   row 1  bin  6       first 6   NOT positive: bin 6 is the first bin of the second minute
#   row 2  none         first 18  not positive
#   row 3  bin  5       first 5   positive: bin 5 is the last bin of the first minute
#   row 4  bins 0, 0    first 0   positive: two engagements in one bin are two onsets, one positive row
#   row 5  bin 17       first 17  not positive
#   row 6  bin  2       first 2   positive          (STOP)
#   row 7  none         first 18  not positive      (STOP)
#   row 8  bins 7, 12   first 7   not positive      (STOP)
#   row 9  bins 5, 5    first 5   positive          (DEV)
#   row 10 bin  8       first 8   not positive      (DEV)
TOY_ONSET_ROLES = ('FIT', 'STOP', 'DEV')
TOY_ONSET_ROWS = (6, 3, 2)                                        # frame rows of each match
TOY_ONSET_N_ROWS = 11
TOY_N_STOP_ROWS = 3
TOY_ONSET_BINS = ((3, 9), (6,), (), (5,), (0, 0), (17,), (2,), (), (7, 12), (5, 5), (8,))
TOY_FIRST_ONSET_BIN = np.array([3, 6, 18, 5, 0, 17, 2, 18, 7, 5, 8])
TOY_N_POSITIVE = 5                                                # rows 0, 3, 4, 6, 9


def build_onset_toy_s1(root) -> Store:
    """The S1 store of the onset toy under ``root``: ``matches/role``, the frame rows with full exposure (10 s in each of the 18 bins) and
    the onsets of ``TOY_ONSET_BINS`` (one unit each, in the order of the rows)."""
    if len(TOY_ONSET_BINS) != TOY_ONSET_N_ROWS or sum(TOY_ONSET_ROWS) != TOY_ONSET_N_ROWS:
        raise RuntimeError('the onset toy has eleven frame rows')
    s = Store(root)
    match = np.repeat(np.arange(len(TOY_ONSET_ROWS)), TOY_ONSET_ROWS)
    k = np.concatenate([np.arange(2, 2 + n) for n in TOY_ONSET_ROWS])
    counts = [len(bins) for bins in TOY_ONSET_BINS]
    _put(s, 'matches/role', [C.ROLES.index(role) for role in TOY_ONSET_ROLES], 'i1')
    _put(s, 'frame_rows/match', match, 'i4')
    _put(s, 'frame_rows/k', k, 'i2')
    _put(s, 'frame_rows/t0', 60_000 * k + 300, 'i8')
    _put(s, 'frame_rows/exposure_ms', np.full((TOY_ONSET_N_ROWS, C.N_BINS), C.BIN_MS), 'i4')
    _put(s, 'onsets/row', np.repeat(np.arange(TOY_ONSET_N_ROWS), counts), 'i4')
    _put(s, 'onsets/bin', np.concatenate([np.array(bins, dtype=np.int64) for bins in TOY_ONSET_BINS if bins]), 'i1')
    _put(s, 'onsets/unit', np.arange(sum(counts)), 'i4')
    _put(s, 'onsets/mark', np.zeros(sum(counts), dtype=np.int64), 'i2')
    s.finalize()
    return s


@pytest.fixture(scope='module')
def onset_toy_s1(tmp_path_factory):
    """The onset toy (``build_onset_toy_s1``), written once per test module and read only."""
    folder = tmp_path_factory.mktemp('onset_toy')
    s1 = build_onset_toy_s1(folder)
    yield s1
    del s1
    gc.collect()
    shutil.rmtree(folder, ignore_errors=True)


# ---- the composer toys (stage S3 Task 9, ``c_ept.compose``) -----------------------------------------------------------------------------
# Three head sets that implement ``heads.ComposerHeads`` with laws written out by hand, and the row tables to compose them on. They hold no
# store: ``bind`` keeps the view numbers it is given (the row feature of a row is [row number, view number]), so that a test can see which
# rows and views a head was asked about.
#
# ``ToyHeads`` is the head set of math-control note 03 section E: onsets 0.01 per second in every bin, the mark 0 (blue takes the first
# kill) with probability 0.6 and the mark 1 (red does) with 0.4, one uniform zone law and one size; the kills of the team that took the
# first kill are Poisson(ln 2) and those of the other team Poisson(ln 4/3); the gold bin is fixed (bin 6, representative 600, for the
# first-kill team and bin 4, representative 400, for the other); the dragon cell of the first-kill team has the mean 0.1; its tower
# conversion runs at 0.002 per second. The prices are 200 for the dragon and 400 for every tower tier, so that the expected tower price is
# 400 whatever law the tier model has. By hand (note 03 E, quoted at ``TOY_EXPECTED_V60``): given the mark 0 the blue kill gold is
# (1 - 1/2) x 600 = 300, the red one (1 - 3/4) x 400 = 100, the dragon 0.1 x 200 = 20, so f0 = 220, and the tower conversion adds
# 0.002 x 400 x (conversion seconds) = 0.8 x seconds; the mark 1 is the mirror image with the opposite sign. Hence the expectation of one
# onset is 0.6 (220 + 0.8 c) - 0.4 (220 + 0.8 c) = 0.2 (220 + 0.8 c), c the conversion exposure of its bin, and
#     E[V_H] = sum over the bins q < H / 10 s of  0.01 x (seconds of the bin) x 0.2 x (220 + 0.8 c_q).
# ``KOToyHeads`` changes only the object and the conversion head, which read K and O (the B1 case, ``analytic_ok`` False):
#     objects:      the blue dragon cell has the mean 0.1 x (kills of blue), the red dragon cell 0.05 x (kills of red)
#     conversions:  the blue tower cell 0.001 x (blue dragons) + 0.0005 x (kills of red) per second, the red tower cell
#                   0.002 x (kill gold of blue in thousands) per second
# ``RichToyHeads`` has laws that differ in the row, the bin, the mark, the zone, the size, the kill layer, the team and the cell, all of
# them everyday arithmetic on whole numbers (see the formulas), with the kill means up to about 6 so that the tail of six or more kills
# has weight; they serve the brute-force oracle of the closed form. Every method is elementwise in its arguments, so a row of the answer
# does not depend on the other rows of the call.
TOY_COMPOSE_PRICES = {'dragon': 200.0, 'elder': 1500.0, 'grubs': 260.0, 'herald': 630.0, 'baron': 1800.0, 'atakhan': 700.0,
                      'tower_outer': 400.0, 'tower_inner': 400.0, 'tower_base': 400.0, 'tower_nexus': 400.0, 'inhibitor': 450.0,
                      'plate': 160.0}
TOY_GOLD_REPS = 100.0 * np.arange(31)                    # the representative of the gold bin b is 100 b: bin 6 is 600, bin 4 is 400
# (t0 seconds, game end seconds) of the toy rows: row 0 is far from the end (note 03 E: 32.16), row 1 ends at 1,075 s (22.12), row 2 at
# 1,100 s (seven full bins, conversions cut for the later bins), row 3 is later in a long game, row 4 has no exposure from the bin 12 on
TOY_ROW_SPECS = ((1000.0, 5000.0), (1000.0, 1075.0), (1000.0, 1100.0), (1500.0, 2400.0), (1000.0, 1150.0))
TOY_ROWS = tuple(range(len(TOY_ROW_SPECS)))
LN2, LN43 = math.log(2.0), math.log(4.0 / 3.0)


class _ToyBase:
    """What the toy head sets share: the flags, ``bind`` (keeps the views), ``row_view`` ((n, 2) float32: the row number and its view)."""
    feature_kind = 'b0'
    analytic_ok = True
    obj_uses_K = False
    conv_uses_KO = False
    N_VIEW_COLUMNS = 2

    def __init__(self):
        self.n_bind = 0
        self.bound_views = None
        self.bound_with = None

    def bind(self, s2, views) -> None:
        self.n_bind += 1
        self.bound_views = np.array(views, dtype=np.int64)
        self.bound_with = id(s2)                                  # the store it was bound to (a test compares the id)

    def _rows(self, rows) -> np.ndarray:
        if self.bound_views is None:
            raise RuntimeError('call bind(s2, views) before asking the heads for rows')
        r = np.asarray(rows)
        if r.dtype.kind not in 'iu' or r.ndim != 1:
            raise TypeError('rows must be a 1-d integer array')
        if r.size and (r.min() < 0 or r.max() >= len(self.bound_views)):
            raise ValueError(f'rows must lie in 0 .. {len(self.bound_views) - 1} (the bound views)')
        return r.astype(np.int64)

    def row_view(self, rows) -> np.ndarray:
        r = self._rows(rows)
        return np.column_stack([r, self.bound_views[r]]).astype(np.float32)

    def _m4_codes(self, m4_inputs) -> tuple:
        """(q, m, z, the whole matrix) of an input of ``features.m4_masked_inputs`` made on a row view of this toy (2 columns)."""
        A = np.asarray(m4_inputs, dtype=np.float64)
        d = self.N_VIEW_COLUMNS
        return A[:, d].astype(np.int64), A[:, d + 1].astype(np.int64), A[:, d + 2].astype(np.int64), A


class ToyHeads(_ToyBase):
    """The heads of note 03 section E (see the section text above). The flags are arguments, so that ``ToyHeads(analytic_ok=False)`` is
    a head set that refuses a closed form."""

    def __init__(self, analytic_ok: bool = True, obj_uses_K: bool = False, conv_uses_KO: bool = False, rate: float = 0.01):
        super().__init__()
        self.analytic_ok, self.obj_uses_K, self.conv_uses_KO = bool(analytic_ok), bool(obj_uses_K), bool(conv_uses_KO)
        self.rate_per_s = float(rate)

    @staticmethod
    def first(m) -> np.ndarray:
        """0 when blue took the first kill (the mark 0, and every even mark), 1 when red did."""
        return np.asarray(m, dtype=np.int64) % 2

    def rate(self, rows) -> np.ndarray:
        return np.full((len(self._rows(rows)), C.N_BINS), self.rate_per_s)

    def mark_prob(self, rows, q) -> np.ndarray:
        p = np.zeros((len(self._rows(rows)), C.N_MARKS))
        p[:, 0], p[:, 1] = 0.6, 0.4
        return p

    def zone_prob(self, rows, q, m) -> np.ndarray:
        return np.full((len(self._rows(rows)), 10), 0.1)

    def size_prob(self, rows, q, m, z) -> np.ndarray:
        p = np.zeros((len(self._rows(rows)), 4))
        p[:, 0] = 1.0
        return p

    def kill_rate(self, rows, q, m, z, s) -> np.ndarray:
        self._rows(rows)
        blue_first = self.first(m) == 0
        return np.column_stack([np.where(blue_first, LN2, LN43), np.where(blue_first, LN43, LN2)])

    def gold_prob(self, rows, q, m, z, s, layer) -> np.ndarray:
        n = len(self._rows(rows))
        blue_first = self.first(m) == 0
        p = np.zeros((n, 2, 31))
        p[np.arange(n), 0, np.where(blue_first, 6, 4)] = 1.0
        p[np.arange(n), 1, np.where(blue_first, 4, 6)] = 1.0
        return p

    def obj_rate(self, rows, q, m, z, s, K) -> np.ndarray:
        n = len(self._rows(rows))
        out = np.zeros((n, 2 * len(C.O_KINDS)))
        out[np.arange(n), self.first(m)] = 0.1                      # the dragon cell (kind 0) of the first-kill team
        return out

    def conv_rate(self, m4_inputs) -> np.ndarray:
        q, m, z, A = self._m4_codes(m4_inputs)
        out = np.zeros((len(A), 2 * len(C.O_KINDS)))
        out[np.arange(len(A)), 2 * C.O_KINDS.index('tower') + self.first(m)] = 0.002          # the tower cell of the first-kill team
        return out


class KOToyHeads(ToyHeads):
    """``ToyHeads`` whose object and conversion heads read K and O (the case of B1): see the section text above for the laws."""

    def __init__(self, rate: float = 0.01):
        super().__init__(analytic_ok=False, obj_uses_K=True, conv_uses_KO=True, rate=rate)

    def obj_rate(self, rows, q, m, z, s, K) -> np.ndarray:
        n = len(self._rows(rows))
        K = np.asarray(K, dtype=np.float64)
        out = np.zeros((n, 2 * len(C.O_KINDS)))
        out[:, 0] = 0.1 * K[:, 0]
        out[:, 1] = 0.05 * K[:, 1]
        return out

    def conv_rate(self, m4_inputs) -> np.ndarray:
        q, m, z, A = self._m4_codes(m4_inputs)
        d = self.N_VIEW_COLUMNS
        kills_red, gold_blue_k, blue_dragons = A[:, d + 4], A[:, d + 5], A[:, d + 7]       # kills_red, gold_blue_k, O of the cell 0
        out = np.zeros((len(A), 2 * len(C.O_KINDS)))
        tower = 2 * C.O_KINDS.index('tower')
        out[:, tower] = 0.001 * blue_dragons + 0.0005 * kills_red
        out[:, tower + 1] = 0.002 * gold_blue_k
        return out


class RichToyHeads(_ToyBase):
    """Laws that differ in every argument, as formulas on whole numbers (r the row, q the bin, m the mark, z the zone, s the size, b the
    gold bin, t the team 0 / 1, c the cell 0 .. 17, L the kill layer 1 .. 6):

        rate        0.004 + 0.0007 q + 0.001 (r mod 3)                      per second
        mark        proportional to  1 + (7 m + 3 q + r) mod 11
        zone        proportional to  1 + (5 z + m + q + r) mod 7
        size        proportional to  1 + (3 s + z + m + q + r) mod 5
        kills       blue  0.2 + 0.7 ((m + q) mod 8) + 0.1 s + 0.02 z + 0.05 (r mod 3)
                    red   0.1 + 0.5 ((m + 3 z) mod 9) + 0.15 s + 0.01 q + 0.05 (r mod 3)
        gold        proportional to  1 + (b (2 + L) + 3 t + m + z + s + r) mod 9
        objects     0.01 ((c + m + z + s + q + r) mod 6)
        conversion  0.0005 ((3 c + q + m + z + r) mod 7)                    per second (r: the first column of the row view)
    """

    @staticmethod
    def _norm(w) -> np.ndarray:
        return w / w.sum(axis=-1, keepdims=True)

    def rate(self, rows) -> np.ndarray:
        r = self._rows(rows)
        return 0.004 + 0.0007 * np.arange(C.N_BINS)[None, :] + 0.001 * (r % 3)[:, None]

    def mark_prob(self, rows, q) -> np.ndarray:
        r, q = self._rows(rows), np.asarray(q, dtype=np.int64)
        return self._norm(1.0 + ((np.arange(C.N_MARKS)[None, :] * 7 + q[:, None] * 3 + r[:, None]) % 11))

    def zone_prob(self, rows, q, m) -> np.ndarray:
        r, q, m = self._rows(rows), np.asarray(q, dtype=np.int64), np.asarray(m, dtype=np.int64)
        return self._norm(1.0 + ((np.arange(10)[None, :] * 5 + (m + q + r)[:, None]) % 7))

    def size_prob(self, rows, q, m, z) -> np.ndarray:
        r, q, m, z = self._rows(rows), np.asarray(q, dtype=np.int64), np.asarray(m, dtype=np.int64), np.asarray(z, dtype=np.int64)
        return self._norm(1.0 + ((np.arange(4)[None, :] * 3 + (z + m + q + r)[:, None]) % 5))

    def kill_rate(self, rows, q, m, z, s) -> np.ndarray:
        q, m, z, s = (np.asarray(a, dtype=np.int64) for a in (q, m, z, s))
        r = self._rows(rows) % 3
        return np.column_stack([0.2 + 0.7 * ((m + q) % 8) + 0.1 * s + 0.02 * z + 0.05 * r,
                                0.1 + 0.5 * ((m + 3 * z) % 9) + 0.15 * s + 0.01 * q + 0.05 * r])

    def gold_prob(self, rows, q, m, z, s, layer) -> np.ndarray:
        r = self._rows(rows)
        m, z, s, layer = (np.asarray(a, dtype=np.int64) for a in (m, z, s, layer))
        b, t = np.arange(31)[None, None, :], np.arange(2)[None, :, None]
        return self._norm(1.0 + ((b * (2 + layer[:, :, None]) + 3 * t + (m + z + s + r)[:, None, None]) % 9))

    def obj_rate(self, rows, q, m, z, s, K) -> np.ndarray:
        r = self._rows(rows)
        q, m, z, s = (np.asarray(a, dtype=np.int64) for a in (q, m, z, s))
        return 0.01 * ((np.arange(2 * len(C.O_KINDS))[None, :] + (m + z + s + q + r)[:, None]) % 6)

    def conv_rate(self, m4_inputs) -> np.ndarray:
        q, m, z, A = self._m4_codes(m4_inputs)
        r = A[:, 0].astype(np.int64)                                      # the row view of this toy starts with the number of the row
        return 0.0005 * ((3 * np.arange(2 * len(C.O_KINDS))[None, :] + (q + m + z + r)[:, None]) % 7)


TOY_B1_LIKE_HEADS = KOToyHeads()                         # no closed form, K and O are read: ``Composer.expect`` must refuse it


class ProcessProbeHeads(ToyHeads):
    """``ToyHeads`` whose onset rate is three times higher in the process that made the heads than in any other process: a probe of where
    a block is computed (a worker gets the heads pickled together with the number of the process that made them)."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.home_pid = os.getpid()

    def rate(self, rows) -> np.ndarray:
        return super().rate(rows) * (3.0 if os.getpid() == self.home_pid else 1.0)


class CountingToyHeads(ToyHeads):
    """``ToyHeads`` that counts how often four of its heads were asked (``n_calls``), in the process that holds the object: a worker counts
    in its own copy, so the count of the calling process shows what the calling process computed."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.n_calls = collections.Counter()

    def rate(self, rows) -> np.ndarray:
        self.n_calls['rate'] += 1
        return super().rate(rows)

    def kill_rate(self, rows, q, m, z, s) -> np.ndarray:
        self.n_calls['kill_rate'] += 1
        return super().kill_rate(rows, q, m, z, s)

    def obj_rate(self, rows, q, m, z, s, K) -> np.ndarray:
        self.n_calls['obj_rate'] += 1
        return super().obj_rate(rows, q, m, z, s, K)

    def conv_rate(self, m4_inputs) -> np.ndarray:
        self.n_calls['conv_rate'] += 1
        return super().conv_rate(m4_inputs)


class FileLogToyHeads(ToyHeads):
    """``ToyHeads`` that appends the row of every call of ``rate`` to a text file: the file shows which rows were computed, also by worker
    processes (the composer asks the rate head once for each row it computes)."""

    def __init__(self, path, **kw):
        super().__init__(**kw)
        self.path = str(path)

    def rate(self, rows) -> np.ndarray:
        with open(self.path, 'a', encoding='ascii') as f:
            f.write(f'{int(np.asarray(rows)[0])}\n')
        return super().rate(rows)


class EventToyHeads(ToyHeads):
    """``ToyHeads`` that appends ``'row<i>'`` to ``events`` (a list that a fake E6 check can share) at every call of ``rate``: in this
    process only (a worker appends to its own copy of the list)."""

    def __init__(self, events, **kw):
        super().__init__(**kw)
        self.events = events

    def rate(self, rows) -> np.ndarray:
        self.events.append(f'row{int(np.asarray(rows)[0])}')
        return super().rate(rows)


class CountingTier:
    """A tier law around a ``towertier.TierModel``: it counts its calls (``n_calls``, in the process that holds the object) and can make the
    answer depend on the number of inputs of a call (``batch_effect``: the probabilities are scaled by 1 + batch_effect x inputs), as the
    last bit of a library product can. ``sizes`` lists the number of inputs of every call."""

    def __init__(self, base, batch_effect: float = 0.0):
        self.base, self.batch_effect, self.n_calls, self.sizes = base, float(batch_effect), 0, []

    def prob(self, destroyed_victim, taker, t_s) -> np.ndarray:
        self.n_calls += 1
        self.sizes.append(len(t_s))
        p = self.base.prob(destroyed_victim, taker, t_s)
        return p * (1.0 + self.batch_effect * len(p))


def toy_row_table(specs=TOY_ROW_SPECS, destroyed=None, frame_row=None):
    """The ``compose.RowTable`` of the rows ``specs`` = ((t0 seconds, game end seconds), ...): the exposure of the occurrence bins is the
    one of ``frame_rows.frame_rows`` (the window (t0, min(t0 + 180 s, T - 30 s)] cut into 10 s bins), the match of row i is i // 2, the
    frame row is i (or ``frame_row``, the global numbers of the rows) and the view is 100 + i (a number that is not the row number).
    ``destroyed`` (n, 2, 4): the tower state, default no tower."""
    from c_ept.valuation import compose
    t0 = np.array([round(1000 * a) for a, _ in specs], dtype=np.int64)
    T = np.array([round(1000 * b) for _, b in specs], dtype=np.int64)
    hi = np.minimum(t0 + C.H_OCC_MS, T - C.END_CUT_MS)
    exposure = np.clip(hi[:, None] - (t0[:, None] + C.BIN_MS * np.arange(C.N_BINS)[None, :]), 0, C.BIN_MS)
    n = len(specs)
    state = np.zeros((n, 2, 4), dtype=np.int64) if destroyed is None else destroyed
    return compose.RowTable(frame_row=np.arange(n) if frame_row is None else frame_row, match=np.arange(n) // 2, t0_ms=t0,
                            exposure_ms=exposure, T_ms=T, views=100 + np.arange(n), destroyed=state, roles=('DEV',))


def rows_with_zero_tail_from_bin(b: int, t0s=(1000.0, 1300.0, 1700.0)):
    """A row table whose rows have no exposure from the occurrence bin ``b`` on: the game ends 10 b + 30 s after t0, so that the window
    (t0, T - 30 s] holds exactly b bins."""
    return toy_row_table([(t0, t0 + 10.0 * b + 30.0) for t0 in t0s])


def toy_tier_model():
    """A tier law of one tier (the outer tower, always): with every tower tier at 400 the expected tower price is 400 whatever the state."""
    from c_ept.valuation import towertier
    k = towertier.N_FEATURES
    return towertier.TierModel([0], np.zeros((1, k)), np.zeros(1), np.zeros(k), np.ones(k))


def toy_composer(S: int = 1000, seed: int = 20261021, rows=None, heads=None, prices=None, tier_model=None, gold_reps=None, s2=None):
    """A ``compose.Composer`` on the toy rows (``toy_row_table()``), the heads of note 03 (``ToyHeads()``), the toy prices, the toy tier law
    and the gold representatives 100 b; every part can be replaced."""
    from c_ept.valuation import compose
    return compose.Composer(ToyHeads() if heads is None else heads, TOY_COMPOSE_PRICES if prices is None else prices,
                            toy_tier_model() if tier_model is None else tier_model, TOY_GOLD_REPS if gold_reps is None else gold_reps,
                            toy_row_table() if rows is None else rows, s2, S=S, seed=seed)
