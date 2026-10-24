r"""D7: the minute gold-lead panel, the event table and the window table (part 1), then the estimators of the flow price and of the
immediate value, the match bootstrap, the price table and the ``flowprice`` command line (part 2, from "Estimators" below).

Minute panel. ``lead_series(s1, m)`` gives, for the regular frames k = 0 .. F - 1 of match row m, the frame times and
lead[k] = (sum of totalGold of the blue team) - (sum of totalGold of the red team); the teams come from ``matches/team``. Only the
regular frames are there: S1 keeps the gold of the regular frames and the positions of the game-end frame (``reader.py``), so nothing is
known about the gold at the game end (ruling R5). The minute k of a time t is the smallest k with t <= ts[k], that is ts[k-1] < t <= ts[k]
(a time on a frame time belongs to the minute that frame ends), and F for a time after the last regular frame. The change of minute k is
lead[k] - lead[k-1]. A match that lacks totalGold in a regular frame is refused (ValueError naming the match row and the count): real
timelines have it everywhere, and a gap would otherwise turn into a made-up number.

Event table. ``event_table`` has one row per objective with a known group and a known taker (``config.O_KINDS``, a tower by its tier
``objectives/sub``; the rule of ``targets._counted`` plus a known tier). Objectives without a group or a taker are left out and counted in
``n_unknown``. For an event in minute k and relative minute r = -4 .. +2 (``REL_MINUTES``):

    y[r]      lead[k+r] - lead[k+r-1], the change of that minute, with the sign of the taker (+1 when blue took the objective, -1 when red
              did), so that it reads as the gain of the team that got the objective
    co[r, :]  ``CO_NAMES``: the kill gold of the minute, then the objectives of each group in it, all with the sign of the taker: (what the
              taker's team got) - (what the other team got) in (ts[k+r-1], ts[k+r]]. Kill gold = bounty + shutdown bounty of the kills by a
              champion; a kill without a champion killer (an execution) has no team and adds nothing. The event itself is left out of its
              own minute; a second objective of the same group or of another group in that minute is counted, and is an event of its own.

The window of an event is complete when k - 5 >= 0 and k + 2 <= F - 1 (all of lead[k-5] .. lead[k+2] exist). An event with an incomplete
window stays in the table with ``complete`` False and ``y`` and ``co`` NaN: it is left out of the estimate and counted by group
(``EventTable.counts_by_group``), never filled with zeros. (A zero ``co`` would look like "no kill, no objective in those minutes", which is
not known.)

Window table. One row per frame-clock row (``frame_rows``: regular frame k >= 2 with ts[k] < T - 30 s) of the matches of the asked roles,
whose window is the minute after the frame, (ts[k], ts[k+1]]. A row whose frame k + 1 is not a regular frame (the last regular frame) is
dropped and counted in ``n_dropped_last``, and the objectives that fall in no kept window are counted by group in ``n_obj_unwindowed``
(ruling R5). ``y`` = lead[k+1] - lead[k] (blue sign), ``treat`` the objectives of each group in the window (blue - red), ``co`` the kill
gold, kill count and assist count of the window (blue - red; an assist is a participant 1-10 in ``kills/amask``). ``ctrl``
(``CTRL_NAMES``) holds the state at the start of the window: lead[k] times the phase indicator of the frame time (``PHASE_EDGES_S``, the
last bin open), the 8 phase indicators, the momentum lead[k] - lead[k-1], the champion-kill difference up to ts[k] and, from the S2
frame-clock view of the row, the blue - red differences of the objective state and the buff indicators.

Roles. Only FIT, STOP and DEV are read (``S3_ROLES``); TEST and EXTERNAL are refused as holdout roles and any other role as not used, before
any array is read. The S2 store is read through ``views/src`` == 0 and ``views/src_index`` == the global frame-row number; the other views
(M3 and M4 event views) are never used, and S1 and S2 are checked against each other (match, reference time and frame number of every view).
Messages carry match row numbers and counts, never a match key; ``matches/key`` is not read.

Estimators (part 2). Both regressions are fitted from per-match sufficient statistics (``MatchStats``: X'X, X'y and the number of rows or
events of every match). A match bootstrap is then a weighted sum of small matrices per replicate, not a refit on 10^6 rows, and the weights
of a replicate are the only thing that changes.

    flow price     ``flow_stats(event_table, group)``. The complete events of one group, stacked:
                       y[e, r] = alpha_e + beta_r + gamma . co[e, r] + noise,    r = -4 .. +2,  beta_{-1} = 0,
                   with a free level alpha_e for every event (the event fixed effect of P2: it takes out everything that is constant inside
                   an event, seen or unseen). It is fitted in the demeaned form: each event's mean over its seven rows is subtracted from y
                   and from every column (Frisch-Waugh-Lovell, which gives exactly the coefficients of the regression on one dummy per
                   event). The columns left are the six relative-minute indicators (r != -1) and 13 ``co`` columns: d = 19. ``flow_beta``
                   puts the 0 of r = -1 back, giving the seven beta. The price p = sum_{r=0..2} beta_r - 3 mean(beta_{-4..-1}) and the
                   pre-trend zeta = (beta_{-2} + beta_{-1}) - (beta_{-4} + beta_{-3}) are contrasts (their weights add up to 0), so
                   the choice of r = -1 as the reference does not move them. Only complete events enter; an incomplete one (NaN) is
                   counted, not filled (``price_table``: ``n_excluded``).
    immediate      ``window_stats(window_table)``. One regression of y = lead[k+1] - lead[k] on [treat (12), co (3), ctrl (33)], d = 48
                   and no intercept: the eight phase indicators of ``ctrl`` add up to 1 in every row and play that part. The coefficient of
                   treat_psi is the immediate value of group psi (P1); that of the kill gold of the window is ``kill_gold_coef``.

``solve(stats, w)`` solves the normal equations for match weights ``w`` (all 1 if None). The system is equilibrated first (each column
scaled to a unit diagonal), because the columns differ by 10^6 in scale (a lead times a phase against an indicator). A column with no
variation in the weighted sample (an exactly zero diagonal: the column of the group itself in a group that cannot happen twice in a
minute, a phase that does not occur) has no estimable coefficient: NaN, and the others are estimated without it. A system whose other
columns are collinear (the equilibrated matrix is singular to double precision) is not estimable: all NaN. Nothing is regularised and
nothing is filled.

``bootstrap(stats, B, seed)`` resamples whole matches: replicate j has the weights ``default_rng(seed).multinomial(M, 1/M)`` (the j-th
draw of one stream, M the number of matches in ``stats``), solved from sum_u w_u X'X_u and sum_u w_u X'y_u. A singular replicate is a row
of NaN and is counted (``singular_count``). ``price_table`` computes the price and the pre-trend again on every replicate and takes the
percentile interval (2.5, 97.5) of the finite ones; it never takes the price of the interval ends of beta.

Price table. For every group: the flow fit (beta, price, pre-trend with their intervals), ``adopted`` = the interval of the pre-trend
contains 0, the immediate value with its interval, and the value used: the flow price if the group is not in ``FIXED_IMMEDIATE``, its
pre-trend is adopted and the flow fit is estimable, else the immediate value. A group whose flow fit is not estimable keeps its counts and
gets None for the flow values; the other groups go on. ``flow_rank`` and ``window_rank`` say how many columns of a fit vary and what rank
they have (``rank_of``): they differ when columns are collinear, which is why a fit is not estimable. ``n_events`` and ``n_excluded`` are
the complete and the incomplete events of the group, ``n_obj_unwindowed`` the objectives that fall in no kept window (ruling R5) with
their share of the group's objectives.

Command line. ``python -m c_ept.flowprice --s1 <S1 run> --s2 <S2 run> [--run <s3a run>]`` checks E6 first, reads the FIT matches of the
two stores, writes ``<s3a run>\prices\PRICES.json`` (atomically, never over an existing one) and records the run in RUN.json and in the
run index.

방법에서의 위치. 이 모듈은 오브젝트 하나가 가져간 팀의 골드 우위에 뒤 흐름으로 준 효과를 골드로 환산한 값을 FIT 경기에서 추정한다: 쌓인 사건 연구로 구하는 흐름 가격 p와, 그것을 쓸 수 없는 묶음에 쓰는
즉시값 θ_ψ이다(설계서 8.1절, 계획 P1·P2, 결정 D7). 먼저 분마다의 골드 우위 lead를 만들고(part 1), 오브젝트 사건마다 사건이 든 분을 0으로 둔 상대 분 r = −4, …, +2의 우위 변화를 쌓은 사건 표와 분 프레임
행마다 그 뒤 한 분의 우위 변화와 오브젝트 수를 담은 창 표를 만든다. 그다음(part 2) 두 회귀를 경기별 충분 통계량으로 맞추고 경기를 통째로 다시 뽑는 부트스트랩으로 구간을 내며, 묶음마다 쓸 값을 PRICES.json의
value로 정한다. 이 값이 f₀의 오브젝트 몫이 되고 합성기의 가격이 된다((VAL-3)).

용어. 정규 프레임: 경기 상태를 1분마다 찍은 기록의 프레임이고 시각이 ts[k]이며, 분 k는 ts[k−1] < t ≤ ts[k]인 시각 t가 드는 한 분이다. 사건: 묶음과 가져간 팀을 아는 오브젝트 하나이다.
묶음 ψ: 가격을 매기는 오브젝트의 12가지 묶음이다(dragon, elder, grubs, herald, baron, atakhan, 포탑 4단계, inhibitor, plate). 가져간 팀의 부호 σ: 파랑이 가져가면 +1,
빨강이면 −1이다. 창이 완전하다: 사건 분 k에서 정규 프레임 k − 5 … k + 2가 모두 있다. 분 프레임: 정규 프레임 가운데 번호가 2 이상이고 시각이 경기 끝 30초 전보다 이른(ts[k] < T − 30 s) 것이며 t₀는 그 시각이고, 창 표의 한 행이다(설계서 11절 8).

식 목록. 식 번호는 (모듈 약어-번호)이고 식과 기호는 docs/superpowers/notes/equation-index.md에 있다. 회귀의 오차는 기호 없이 "오차"라 쓴다. 앞 추세는 ζ, 시기 번호는 ph, 묶음 번호는 ψ이다.
    (FLOW-1)   분 골드 우위       lead[k] = Σ_{참가자} σ(팀)·totalGold[k],   시각 t가 든 분 k(t) = min{k : t ≤ ts[k]}
    (FLOW-2)   사건 행            y[e, r] = σ_e·(lead[k_e + r] − lead[k_e + r − 1]),   co[e, r]: 같은 분의 다른 사건
    (FLOW-3)   창 행              y_i = lead[k_i + 1] − lead[k_i],   treat_i, co_i, ctrl_i
    (FLOW-4)   쌓인 사건 연구     y[e, r] = α_e + β_r + γ·co[e, r] + 오차,   β_{−1} = 0       (사건 안 평균을 뺀 형태로 맞춘다)
    (FLOW-5)   즉시값 회귀        y_i = Σ_ψ θ_ψ·treat_iψ + (같은 분의 킬) + (창 시작 상태) + 오차,   절편 없음
    (FLOW-6)   정규방정식의 해    (Σ_u w_u·X_u'X_u)·θ = Σ_u w_u·X_u'y_u       (열 평형화; 변하지 않는 열은 NaN)
    (FLOW-7)   경기 부트스트랩    θ*_j = (FLOW-6)의 해(w = d_j),   d_j ~ Multinomial(M; 1/M, …, 1/M)   (j = 1, …, B)
    (FLOW-8)   흐름 가격          p = Σ_{r=0}^{2} β_r − 3·평균(β_{−4}, …, β_{−1})
    (FLOW-9)   앞 추세            ζ = (β_{−2} + β_{−1}) − (β_{−4} + β_{−3})
    (FLOW-10)  채택과 쓰는 값     채택 ⇔ ζ의 구간 ∋ 0;   value = p (흐름) 또는 θ_ψ (즉시값)

공통 가정.
    (가) 정규 프레임만(판정 R5). 경기 끝 프레임의 골드는 S1에 없어 알 수 없으므로 정규 프레임만 쓰고, 다음 정규 프레임이 없는 창 행은 빼서 센다.
    (나) 가져간 팀의 부호. 사건 표의 y와 co에는 σ_e를 곱해 파랑이 가져간 사건과 빨강이 가져간 사건을 가져간 팀의 이득으로 합치고, 창 표는 파랑 − 빨강의 부호를 그대로 쓴다.
    (다) 경기 단위. 같은 경기의 행은 같은 흐름을 공유해 서로 의존하므로 다른 경기끼리만 독립이라고 가정하고 구간은 경기를 통째로 다시 뽑아 만든다
         (설계서 8.1절, 계획 전역 제약의 D7 항). 경기별 합은 그 재표본을 재적합 없이 계산하는 수단이며 점추정은 모든 행을 쓴 최소제곱과 같다.
    (라) 읽는 경기. 표를 만드는 함수는 FIT, STOP, DEV를 받고 홀드아웃 역할과 S3가 쓰지 않는 역할은 읽기 전에 거부하며, 명령줄은 FIT 경기만 읽는다(계획 전역 제약의 분할과 D7 항).
"""
# Ref: docs/superpowers/specs/2026-10-20-c-ept-design.md section 8.1 (D7: relative minutes -4 .. +2, controls, the immediate value as the
#      coefficient of a minute window regression, tower tiers) and section 11, items 16 and 17 (the tier column `sub`; objectives without a
#      known kind or taker are not counted).
# Ref: docs/superpowers/plans/2026-10-21-c-ept-s3.md decisions P1 (immediate value: the coefficient of the window regression, controls from
#      the state at the window start and the kills of the window), P2 (event fixed effects, the other events of the minute as controls)
#      and Tasks 1 and 2; rulings R4 (the probe file) and R5 (regular frames only).
# Ref: D:\LOL_Data\runs\20261020_flow_price_probe_1\flow_price_probe.py (the prototype: window rows, treatment counts, controls and the
#      phase edges).
# Ref: Cengiz, Dube, Lindner, Zipperer (2019), "The effect of minimum wages on low-wage jobs", Quarterly Journal of Economics
#      134(3): 1405-1454; doi:10.1093/qje/qjz014 (the stacked event study: events stacked, an effect for each event, one set of
#      relative-time coefficients).
# Ref: Frisch and Waugh (1933); doi:10.2307/1907330, and Lovell (1963); doi:10.1080/01621459.1963.10480682 (the regression on a set of
#      dummies equals the regression on the variables with their group means taken out).
# Ref: Efron (1979), "Bootstrap methods: another look at the jackknife", Annals of Statistics 7(1): 1-26; doi:10.1214/aos/1176344552,
#      and Cameron, Gelbach, Miller (2008), "Bootstrap-based improvements for inference with clustered errors", Review of Economics
#      and Statistics 90(3): 414-427; doi:10.1162/rest.90.3.414 (resample whole clusters, here matches; percentile intervals).
from __future__ import annotations

import argparse
import datetime as dt
import functools
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from c_ept import config as C
from c_ept import reader
from c_ept.casts import narrow
from c_ept.units import BLUE, RED, _ALL_PIDS, _POPCOUNT, _is_pid

# ---- names and constants ---------------------------------------------------------------------------------------------------------
GROUPS = ('dragon', 'elder', 'grubs', 'herald', 'baron', 'atakhan', 'tower_outer', 'tower_inner', 'tower_base', 'tower_nexus', 'inhibitor',
          'plate')
N_GROUPS = len(GROUPS)
FIXED_IMMEDIATE = ('tower_base', 'tower_nexus', 'inhibitor')       # always priced by the immediate value (design 8.1)
REL_MINUTES = (-4, -3, -2, -1, 0, 1, 2)
N_REL = len(REL_MINUTES)
R0 = REL_MINUTES.index(0)                                           # position of the event's own minute
CO_NAMES = ('kill_gold',) + GROUPS
WINDOW_CO_NAMES = ('kill_gold', 'kill_count', 'assist_count')
PHASE_EDGES_S = (120, 300, 600, 900, 1200, 1500, 1800, 2100)        # seconds on the frame time; bin ph is [edge ph, edge ph+1), last open
N_PHASES = len(PHASE_EDGES_S)
S3_ROLES = ('FIT', 'STOP', 'DEV')
_HOLDOUT_ROLES = tuple(reader.HOLDOUT_ROLES.values())               # TEST, EXTERNAL

# the objective-state differences of the controls (blue_<name> - red_<name> of the S2 summary); the plates add up the three lanes
_SUMMARY_DIFFS = ('towers_outer', 'towers_inner', 'towers_base', 'towers_nexus', 'inhibs_down', 'plates', 'dragons', 'elders', 'barons',
                  'heralds', 'atakhans', 'voidgrubs', 'has_soul')
_SUMMARY_PARTS = {'plates': ('plates_top', 'plates_mid', 'plates_bot')}
_BUFFS = ('baron', 'elder')                                         # blue_<buff>_buff_s > 0 minus red_<buff>_buff_s > 0
CTRL_NAMES = (tuple(f'lead_x_phase{i}' for i in range(N_PHASES)) + tuple(f'phase{i}' for i in range(N_PHASES))
              + ('momentum', 'cum_kill_diff') + _SUMMARY_DIFFS + tuple(f'{b}_buff' for b in _BUFFS))
_FIRST_SUMMARY = CTRL_NAMES.index(_SUMMARY_DIFFS[0])                # the controls from here on come from S2

_GOLD = C.FRAME_FIELDS.index('totalGold')
_REL = np.array(REL_MINUTES, dtype=np.int64)
_EDGES_MS = np.array(PHASE_EDGES_S, dtype=np.int64) * 1000
_OFFSETS = ('frame_off', 'kill_off', 'obj_off', 'row_off')          # the matches/ arrays that cut the tables per match
_CHUNK = 1 << 17                                                    # views read from x/summary at a time

# part 2: the estimators
BOOT_B = 2000                                                       # D7: matches are resampled 2,000 times ...
BOOT_SEED = 20261021                                                # ... from this seed
FIT_ROLES = ('FIT',)                                                # the prices are estimated on FIT only (D7)
DEFAULT_PROBE = Path(r'D:\LOL_Data\runs\20261020_flow_price_probe_1\flow_price_probe.json')      # ruling R4: read only if the file exists
LOCK_NAME = 'S3A.lock'                                              # in the run folder: one call at a time
S2_DIR = 's2'                                                       # the store under an S2 run folder (``store_build.STORE_DIR``)
_REF = REL_MINUTES.index(-1)                                        # the relative minute that beta is measured from (beta = 0 there)
_KEEP = tuple(j for j in range(N_REL) if j != _REF)                 # the positions of the six minutes that have a coefficient
FLOW_COLUMNS = tuple(f'r{REL_MINUTES[j]:+d}' for j in _KEEP) + tuple(f'co_{name}' for name in CO_NAMES)               # d = 19
WINDOW_COLUMNS = tuple(f'treat_{g}' for g in GROUPS) + tuple(f'co_{name}' for name in WINDOW_CO_NAMES) + CTRL_NAMES   # d = 48
_DUMMIES = np.eye(N_REL)[:, list(_KEEP)]                            # (7, 6): row r is 1 in the column of its minute, r = -1 has none
_KILL_GOLD = len(GROUPS)                                            # the column of the kill gold in the window regression
_BOOT_BYTES = 64 * 2 ** 20                                          # memory for the match weights of one chunk of replicates
_PERCENTILES = (2.5, 97.5)

_TOWER = C.O_KINDS.index('tower')
_BY_KIND = np.full(len(C.O_KINDS), -1, dtype=np.int64)              # group of a kind that does not depend on the tier
for _i, _name in enumerate(C.O_KINDS):
    if _name != 'tower':
        if _name not in GROUPS:
            raise RuntimeError(f'config.O_KINDS has the kind {_name!r}, which is not a group of the flow prices')
        _BY_KIND[_i] = GROUPS.index(_name)
_FIRST_TOWER = GROUPS.index('tower_outer')
if GROUPS[_FIRST_TOWER:_FIRST_TOWER + 4] != ('tower_outer', 'tower_inner', 'tower_base', 'tower_nexus'):
    raise RuntimeError('the four tower groups must be consecutive, in the order of objectives/sub 0-3')


def group_of(kind, sub) -> np.ndarray:
    """Index in ``GROUPS`` (int8) of objectives by their ``config.O_KINDS`` index ``kind`` and tower tier ``sub`` (0-3 for a tower, which
    alone reads it), -1 where the group is not known: a kind that is not in the config (-1, another monster or building) or a tower
    without a tier. Works on arrays and on single numbers.

    오브젝트의 종류와 포탑 단계로 묶음 ψ를 정한다((FLOW-2), (FLOW-5)의 묶음). 단계는 포탑만 읽으며(설계서 8.1절의 포탑 단계), 종류나 단계를 모르면 묶음이 없어 −1이다. 반환값은 GROUPS의
    순서로 0부터 센 번호이다.
    """
    kind = np.asarray(kind, dtype=np.int64)
    sub = np.asarray(sub, dtype=np.int64)
    known = (kind >= 0) & (kind < len(C.O_KINDS))
    out = np.where(known, _BY_KIND[np.clip(kind, 0, len(C.O_KINDS) - 1)], -1)
    tiered = known & (kind == _TOWER)
    out = np.where(tiered, np.where((sub >= 0) & (sub < 4), _FIRST_TOWER + sub, -1), out)
    return narrow('group', out, np.int8)


def phase_of(t_ms) -> np.ndarray:
    """Phase bin (int8, 0 .. 7) of a game time in ms: bin ph covers [PHASE_EDGES_S[ph], PHASE_EDGES_S[ph+1]) seconds and the last bin is
    open. A time before the first edge is in bin 0.

    (FLOW-3)의 시기 번호 ph(t)를 정한다: 프레임 시각 t가 든 구간의 번호이고 마지막 구간은 열려 있다. 시기 구간은 계획 전역 제약이 정한다.
    """
    i = np.searchsorted(_EDGES_MS, np.asarray(t_ms, dtype=np.int64), side='right') - 1
    return narrow('phase', np.clip(i, 0, N_PHASES - 1), np.int8)


def _role_codes(roles) -> np.ndarray:
    """Sorted ``config.ROLES`` indices of ``roles`` (a name or names). ValueError for no role, a holdout role (TEST, EXTERNAL), a role
    stage S3 does not use (CAL, AUDIT, UNUSED) and a name that is no role. Reads nothing: the refusal comes before any array."""
    if isinstance(roles, str):
        roles = (roles,)
    roles = tuple(roles)
    if not roles:
        raise ValueError(f'no role given: name at least one of {", ".join(S3_ROLES)}')
    for role in roles:
        if role in _HOLDOUT_ROLES:
            raise ValueError(f'role {role} is a holdout role: stage S3 never reads it')
        if role not in C.ROLES:
            raise ValueError(f'unknown role {role!r}: not one of {C.ROLES}')
        if role not in S3_ROLES:
            raise ValueError(f'role {role} is not used by stage S3, which reads {", ".join(S3_ROLES)}')
    return np.array(sorted({C.ROLES.index(r) for r in roles}), dtype=np.int64)


def _team_sign(team) -> np.ndarray:
    """(..., 10) float: +1 for a blue participant (team id 100), -1 for a red one (200); ValueError for any other id.

    (FLOW-1)의 팀 부호 σ(팀)이다(파랑 +1, 빨강 −1).
    """
    team = np.asarray(team)
    if not np.isin(team, (BLUE, RED)).all():
        raise ValueError(f'a team id is neither {BLUE} (blue) nor {RED} (red) in matches/team')
    return np.where(team == BLUE, 1.0, -1.0)


# ---- the minute panel -----------------------------------------------------------------------------------------------------------------
def _lead(s1, f0: int, f1: int, sign: np.ndarray, m: int) -> np.ndarray:
    """lead of the frames f0 .. f1 - 1 of the match row m: sum over the participants of sign x totalGold (float64).

    (FLOW-1)의 lead[k]를 한 경기의 정규 프레임마다 계산한다(식은 lead_series에 있다).
    """
    gold = np.array(s1.array('frames/num')[f0:f1, :, _GOLD], dtype=np.float64)
    missing = int(np.isnan(gold).any(axis=1).sum())
    if missing:
        raise ValueError(f'match row {m}: totalGold is missing in {missing} of {f1 - f0} frames')
    return gold @ sign


def lead_series(s1, m: int) -> tuple[np.ndarray, np.ndarray]:
    """``(ts, lead)`` of the regular frames of match row ``m``: ts int64 (ms), lead float64 = sum of totalGold of the blue team - that of
    the red team. The game-end frame is not there (S1 keeps no gold for it).

    수식
    ----
    (FLOW-1)  lead[k] = Σ_{참가자} σ(팀)·totalGold[k],      k = 0, …, F − 1          (σ(파랑) = +1, σ(빨강) = −1)
              시각 t가 든 분: k(t) = min{k : t ≤ ts[k]}   (t가 마지막 정규 프레임 뒤이면 F),      분 k의 변화: lead[k] − lead[k − 1]

      기호                | 뜻                                       | 단위          | 범위         | 코드 변수
      F, k                | 한 경기의 정규 프레임 수, 프레임·분 번호 | 개, —         | ≥ 1; 0, …, F | len(ts), k
      ts[k]               | 정규 프레임 k의 시각                     | s (코드는 ms) | 오름차순     | ts
      lead[k]             | 프레임 k의 골드 우위(파랑 − 빨강)        | 골드          | 부호 있음    | lead
      σ(팀), totalGold[k] | 참가자의 팀 부호, 프레임 k의 총 골드     | —, 골드       | ±1; ≥ 0      | sign, frames/num의 totalGold

    해석
    ----
    식 (FLOW-1)의 lead는 파랑 팀의 총 골드에서 빨강 팀의 총 골드를 뺀 우위이고, 분 k의 변화는 그 분을 끝내는 프레임과 앞 프레임의 차이이다. 프레임 시각과 같은 시각은 그 프레임이 끝내는
    분에 든다(앞은 열리고 뒤는 닫힌 구간). 정규 프레임만 있다: S1은 경기 끝 프레임의 골드를 두지 않아 경기 끝의 골드는 알 수 없으므로 마지막 정규 프레임 뒤의 시각은 분 F에 둔다(공통 가정 (가)).
    totalGold가 빠진 프레임이 있는 경기는 만들어 낸 수가 우위에 들어가지 않도록 거부한다(Task 1 구현의 규약이다).

    근거
    ----
    계획 Task 1(분당 골드 우위), 판정 R5(정규 프레임만), 설계서 8.1절(분당 골드 우위 변화), 검산 노트 math-control-03 A절(손 계산; 시험이 같은 수를 확인한다).
    """
    off = s1.array('matches/frame_off')
    f0, f1 = int(off[m]), int(off[m + 1])
    sign = _team_sign(np.array(s1.array('matches/team')[m]))
    return np.array(s1.array('frames/ts')[f0:f1], dtype=np.int64), _lead(s1, f0, f1, sign, m)


@dataclass(frozen=True, eq=False)
class _MatchData:
    """What the two tables need of one match. ``o_*`` are its objectives with a group and a taker, in S1 order."""
    ts: np.ndarray           # (F,) int64
    lead: np.ndarray         # (F,) float64
    o_t: np.ndarray          # (n,) int64
    o_group: np.ndarray      # (n,) int64
    o_taker: np.ndarray      # (n,) int64, 0 blue / 1 red
    o_sign: np.ndarray       # (n,) float64, +1 blue / -1 red
    o_minute: np.ndarray     # (n,) int64, event minute in 0 .. F
    n_unknown: int           # objectives left out for want of a group or a taker
    obj: np.ndarray          # (F + 1, 12) float64, blue - red objectives per minute and group
    kill_gold: np.ndarray    # (F + 1,) float64, blue - red kill gold per minute
    kill_count: np.ndarray   # (F + 1,) float64, blue - red champion kills per minute
    assist_count: np.ndarray  # (F + 1,) float64, blue - red assists per minute


class _S1Reader:
    """The S1 arrays of the asked roles: the offsets, roles and teams are loaded once, the big arrays are sliced per match."""

    def __init__(self, s1, roles):
        codes = _role_codes(roles)                                  # first: a refused role reads nothing
        self.s1 = s1
        self.off = {name: np.array(s1.array(f'matches/{name}'), dtype=np.int64) for name in _OFFSETS}
        n = len(self.off['frame_off']) - 1
        role = np.array(s1.array('matches/role'))
        team = np.array(s1.array('matches/team'))
        if any(len(o) != n + 1 for o in self.off.values()) or len(role) != n or len(team) != n:
            raise ValueError('the arrays of matches/ differ in length')
        self.matches = np.flatnonzero(np.isin(role, codes))
        _team_sign(team[self.matches])                              # the ids of the matches that are read must be blue or red
        self.sign = np.where(team == BLUE, 1.0, -1.0)              # (n, 10)

    def load(self, m: int) -> _MatchData:
        """한 경기의 (FLOW-1) lead와 분 k(t), 그리고 (FLOW-2), (FLOW-3)이 쓰는 분별 합을 만든다: 묶음별 오브젝트 수(파랑 − 빨강)와 챔피언 킬의 골드·수·도움 수(파랑 − 빨강)이다.
        식은 lead_series, event_table, window_table에 있다.
        """
        s1, off = self.s1, self.off
        f0, f1 = int(off['frame_off'][m]), int(off['frame_off'][m + 1])
        F = f1 - f0
        ts = np.array(s1.array('frames/ts')[f0:f1], dtype=np.int64)
        sign = self.sign[m]
        lead = _lead(s1, f0, f1, sign, m)
        # objectives: group and taker must be known; the minute is the frame that ends it
        o0, o1 = int(off['obj_off'][m]), int(off['obj_off'][m + 1])
        o_t = np.array(s1.array('objectives/t')[o0:o1], dtype=np.int64)
        kind = np.array(s1.array('objectives/kind')[o0:o1], dtype=np.int64)
        sub = np.array(s1.array('objectives/sub')[o0:o1], dtype=np.int64)
        taker = np.array(s1.array('objectives/taker')[o0:o1], dtype=np.int64)
        group = group_of(kind, sub).astype(np.int64)
        known = (group >= 0) & (taker >= 0) & (taker <= 1)
        o_t, group, taker = o_t[known], group[known], taker[known]
        o_sign = 1.0 - 2.0 * taker
        # 식 (FLOW-1)의 분 k(t): 왼쪽으로 찾으므로 프레임 시각과 같은 시각은 그 프레임이 끝내는 분에 든다. 분별 오브젝트 수는 부호 σ를 곱해 파랑 − 빨강으로 더한다.
        o_minute = np.searchsorted(ts, o_t, side='left')
        obj = np.bincount(o_minute * N_GROUPS + group, weights=o_sign, minlength=(F + 1) * N_GROUPS).reshape(F + 1, N_GROUPS)
        # kills: only a champion killer has a team; an execution adds nothing
        k0, k1 = int(off['kill_off'][m]), int(off['kill_off'][m + 1])
        k_t = np.array(s1.array('kills/t')[k0:k1], dtype=np.int64)
        killer = np.array(s1.array('kills/killer')[k0:k1], dtype=np.int64)
        # 식 (FLOW-2)의 킬 골드는 bounty와 shutdown bounty만이고 도움 풀은 넣지 않는다.
        gold = (np.array(s1.array('kills/bounty')[k0:k1], dtype=np.int64)
                + np.array(s1.array('kills/shutdown')[k0:k1], dtype=np.int64)).astype(np.float64)
        assists = _POPCOUNT[np.array(s1.array('kills/amask')[k0:k1], dtype=np.int64) & _ALL_PIDS].astype(np.float64)
        side = np.where(_is_pid(killer), sign[np.clip(killer - 1, 0, 9)], 0.0)
        k_minute = np.searchsorted(ts, k_t, side='left')
        size = F + 1
        return _MatchData(
            ts=ts, lead=lead, o_t=o_t, o_group=group, o_taker=taker, o_sign=o_sign, o_minute=o_minute, n_unknown=int((~known).sum()),
            obj=obj, kill_gold=np.bincount(k_minute, weights=side * gold, minlength=size),
            kill_count=np.bincount(k_minute, weights=side, minlength=size),
            assist_count=np.bincount(k_minute, weights=side * assists, minlength=size))


def _cat(parts: list, tail: tuple, dtype) -> np.ndarray:
    """The parts joined, or an empty array of shape (0, *tail) when there are none."""
    return np.concatenate(parts) if parts else np.empty((0, *tail), dtype=dtype)


# ---- the event table ------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class EventTable:
    """One row per objective with a known group and taker of the matches of the asked roles, ordered by match row, then S1 order (time).

    ``match`` is the S1 match row. ``taker`` 0 blue, 1 red. ``k`` the event minute (F for a time after the last regular frame). ``y`` and
    ``co`` are as in the module text (columns ``REL_MINUTES`` and ``CO_NAMES``); an incomplete window has ``complete`` False and ``y`` and
    ``co`` NaN. ``n_unknown`` objectives were left out for want of a group or a taker, ``n_matches`` matches were read.

    ``y``와 ``co``의 식은 event_table에 있다((FLOW-2)).
    """
    match: np.ndarray        # (n,) int32
    group: np.ndarray        # (n,) int8, index in GROUPS
    taker: np.ndarray        # (n,) int8
    t: np.ndarray            # (n,) int64, ms
    k: np.ndarray            # (n,) int32
    complete: np.ndarray     # (n,) bool
    y: np.ndarray            # (n, 7) float64
    co: np.ndarray           # (n, 7, 13) float64
    n_unknown: int
    n_matches: int

    def index_of(self, match: int, t: int, group=None) -> int:
        """The row of the event of match row ``match`` at the time ``t`` (ms). KeyError if there is none; ValueError if several share the
        time (two objectives in one ms): name the ``group`` (its name or index) then.

        (FLOW-2)의 사건 e를 경기와 시각으로 찾는 조회이며, 시각이 같은 사건이 둘 이상이면 어느 것인지 알 수 없어 묶음을 말하게 한다.
        """
        hit = np.flatnonzero((self.match == match) & (self.t == t))
        if group is not None:
            hit = hit[self.group[hit] == (GROUPS.index(group) if isinstance(group, str) else int(group))]
        if hit.size == 0:
            raise KeyError(f'no event of match row {match} at {t} ms')
        if hit.size > 1:
            raise ValueError(f'{hit.size} events of match row {match} at {t} ms: name the group')
        return int(hit[0])

    def counts_by_group(self) -> dict:
        """{'events', 'complete', 'incomplete'}: for each, the number of events of each group (int64, shape (12,), order ``GROUPS``).

        (FLOW-2)의 사건을 묶음마다 센다. 추정에서 뺀 불완전한 사건의 수를 알리기 위해서이다(검토 초점 1).
        """
        def count(mask):
            return np.bincount(self.group[mask].astype(np.int64), minlength=N_GROUPS).astype(np.int64)
        every = np.ones(len(self.group), dtype=bool)
        return {'events': count(every), 'complete': count(self.complete), 'incomplete': count(~self.complete)}


def _event_rows(d: _MatchData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(complete, y, co)`` of the events of one match (see the module text); ``y`` and ``co`` are NaN for an incomplete window.

    (FLOW-2)의 완전 여부와 y[e, r], co[e, r]를 한 경기의 사건에 대해 계산한다(식은 event_table에 있다).
    """
    F, k = len(d.ts), d.o_minute
    # 식 (FLOW-2)의 완전한 창: lead[k − 5] … lead[k + 2]가 모두 있어야 상대 분 −4 … +2의 변화를 만들 수 있다.
    complete = (k - 5 >= 0) & (k + 2 <= F - 1)
    y = np.full((len(k), N_REL), np.nan)
    co = np.full((len(k), N_REL, 1 + N_GROUPS), np.nan)             # an incomplete window has no controls either: NaN, never a 0
    c = np.flatnonzero(complete)
    if c.size:
        j = k[c, None] + _REL[None, :]                              # (nc, 7): the minute of each relative minute
        s = d.o_sign[c, None]                                       # +1 blue took it, -1 red: the sign of the taker
        # 식 (FLOW-2)의 y[e, r]와 co[e, r]: 가져간 팀의 부호 σ_e(코드 변수 s)를 곱해 파랑이 가져간 사건과 빨강이 가져간 사건을 한 방향으로 모으고, 사건 자신을 뺀다.
        y[c] = s * (d.lead[j] - d.lead[j - 1])
        co[c, :, 0] = s * d.kill_gold[j]
        co[c, :, 1:] = s[:, :, None] * d.obj[j]
        co[c, R0, 1 + d.o_group[c]] -= 1.0                          # the event itself is in its own minute with s * s = +1: leave it out
    return complete, y, co


def event_table(s1, roles=('FIT',)) -> EventTable:
    """The event table of the matches of ``roles`` (FIT, STOP, DEV) of the S1 store ``s1``; see the module text.

    수식
    ----
    (FLOW-2)  사건 e: 묶음 ψ_e, 가져간 팀의 부호 σ_e, 사건이 든 분 k_e = k(t_e).      r = −4, …, +2에 대해
              y[e, r]     = σ_e·(lead[k_e + r] − lead[k_e + r − 1])
              co[e, r, 킬 골드] = σ_e·Σ_{챔피언 킬 ∈ 분 k_e + r} σ(킬러 팀)·(bounty + shutdown)
              co[e, r, ψ] = σ_e·Σ_{오브젝트 ∈ 분 k_e + r, 묶음 ψ} σ(가져간 팀)  −  [r = 0 그리고 ψ = ψ_e]        (ψ마다)
              완전 ⇔ k_e − 5 ≥ 0 그리고 k_e + 2 ≤ F − 1.   불완전한 사건은 y와 co가 NaN이다.

      기호        | 뜻                                                       | 단위     | 범위      | 코드 변수
      e, ψ_e, k_e | 사건 번호, 그 묶음, 그 사건이 든 분                      | —        | …         | 행, group, k
      σ_e         | 가져간 팀의 부호(파랑 +1, 빨강 −1)                       | —        | ±1        | o_sign (코드 변수 s)
      r           | 사건 분 기준 상대 분                                     | 분       | −4, …, +2 | REL_MINUTES
      y[e, r]     | 가져간 팀의 우위 변화                                    | 골드     | 부호 있음 | y
      co[e, r, ·] | 같은 분의 킬 골드와 묶음별 오브젝트 수(가져간 팀 − 상대) | 골드, 개 | 부호 있음 | co (열은 CO_NAMES)

    해석
    ----
    식 (FLOW-2)의 y는 사건 e를 가져간 팀의 분당 우위 변화이며 사건이 든 분과 그 앞뒤의 분을 상대 분 r로 나란히 쌓는다.
    co는 같은 상대 분의 다른 사건으로 가져간 팀 기준의 킬 골드(기록 bounty와 shutdown bounty만이고 도움 풀은 넣지 않는다)와 묶음별 오브젝트 수이며, 사건 자신은 자기 분에서 빼고 같은 분의 다른
    오브젝트는 따로 사건이 된다(계획 P2, 검토 초점 2). 챔피언 킬러가 없는 킬(처형)은 어느 팀에도 지급하지 않으므로 킬 골드에 더하지 않고(설계서 3.2절), 묶음이나 가져간 팀을 모르는 오브젝트는
    사건이 아니어서 빼고 센다(계획 Task 1). 창이 완전하지 않은 사건은 표에 남기되 y와 co를 NaN으로 두어 추정에서 빼고 묶음별로 센다: 0으로 채우면 그 분에 킬도 오브젝트도 없었다는 뜻으로 읽히기
    때문이다(계획 전역 제약의 D7 항, 검토 초점 1).

    근거
    ----
    계획 P2(같은 상대 분의 다른 사건을 통제), 계획 전역 제약의 D7 항(사건 분과 완전한 창), Task 1(사건 표), 검토 초점 1·2(불완전한 사건은 NaN, 같은 분의 오브젝트는 각자 사건), 판정 R5, 설계서 8.1절(상대 분과
    가져간 팀의 부호)과 3.2절(처형), 검산 노트 math-control-03 A절(손 계산; 시험이 같은 수를 확인한다).
    Ref: Cengiz, D., Dube, A., Lindner, A. and Zipperer, B. (2019), Quarterly Journal of Economics 134(3): 1405-1454;
         doi:10.1093/qje/qjz014 (사건을 쌓아 상대 시간 계수를 맞춘다).
    """
    rd = _S1Reader(s1, roles)
    parts = {name: [] for name in ('match', 'group', 'taker', 't', 'k', 'complete', 'y', 'co')}
    n_unknown = 0
    for m in rd.matches:
        d = rd.load(int(m))
        n_unknown += d.n_unknown
        complete, y, co = _event_rows(d)
        for name, value in (('match', np.full(len(d.o_t), m, dtype=np.int64)), ('group', d.o_group), ('taker', d.o_taker), ('t', d.o_t),
                            ('k', d.o_minute), ('complete', complete), ('y', y), ('co', co)):
            parts[name].append(value)
    return EventTable(
        match=narrow('match', _cat(parts['match'], (), np.int64), np.int32),
        group=narrow('group', _cat(parts['group'], (), np.int64), np.int8),
        taker=narrow('taker', _cat(parts['taker'], (), np.int64), np.int8), t=_cat(parts['t'], (), np.int64),
        k=narrow('k', _cat(parts['k'], (), np.int64), np.int32), complete=_cat(parts['complete'], (), np.bool_),
        y=_cat(parts['y'], (N_REL,), np.float64), co=_cat(parts['co'], (N_REL, 1 + N_GROUPS), np.float64),
        n_unknown=n_unknown, n_matches=len(rd.matches))


# ---- the window table -----------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class WindowTable:
    """One row per kept frame-clock row of the matches of the asked roles, ordered by S1 frame row; see the module text.

    ``match`` the S1 match row, ``row`` the global S1 frame-row number (= ``frame_rows/*`` index), ``k`` the regular frame. ``treat`` (12
    groups), ``co`` (kill gold, kill count, assist count) and ``ctrl`` (``CTRL_NAMES``) are float64. ``n_dropped_last`` rows had no regular
    frame k + 1 and are not in the table; ``n_obj_unwindowed`` (12,) counts, by group, the known objectives that fall in no kept window;
    ``n_unknown`` objectives were left out for want of a group or a taker; ``n_matches`` matches were read.

    각 열의 식은 window_table에 있다((FLOW-3)).
    """
    match: np.ndarray             # (n,) int32
    row: np.ndarray               # (n,) int64
    k: np.ndarray                 # (n,) int32
    y: np.ndarray                 # (n,) float64
    treat: np.ndarray             # (n, 12) float64
    co: np.ndarray                # (n, 3) float64
    ctrl: np.ndarray              # (n, 33) float64
    n_dropped_last: int
    n_obj_unwindowed: np.ndarray  # (12,) int64
    n_unknown: int
    n_matches: int


@functools.lru_cache(maxsize=1)
def _summary_plan() -> tuple:
    """``(differences, buffs, columns)`` of the S2 summary that the controls read: for each of ``_SUMMARY_DIFFS`` the column numbers of its
    blue and its red parts, for each buff the columns of its blue and red seconds, and the sorted set of all of them. Imports ``tokens``
    here, so that a caller of the event table does not load the S2 modules."""
    from c_ept import tokens
    index = {name: i for i, name in enumerate(tokens.SUMMARY)}

    def column(name):
        if name not in index:
            raise ValueError(f'the S2 summary has no column {name!r}')
        return index[name]
    differences = tuple((tuple(column(f'blue_{p}') for p in _SUMMARY_PARTS.get(name, (name,))),
                         tuple(column(f'red_{p}') for p in _SUMMARY_PARTS.get(name, (name,)))) for name in _SUMMARY_DIFFS)
    buffs = tuple((column(f'blue_{b}_buff_s'), column(f'red_{b}_buff_s')) for b in _BUFFS)
    used = sorted({c for blue, red in differences for c in blue + red} | {c for pair in buffs for c in pair})
    return differences, buffs, tuple(used)


def _summary_controls(s1, s2, rows: np.ndarray, match: np.ndarray, k: np.ndarray, t0: np.ndarray) -> np.ndarray:
    """(n, 15) float64: the S2 controls (the 13 differences, then the 2 buff indicators) of the S1 frame rows ``rows``, from their
    frame-clock views (``views/src`` == 0 and ``views/src_index`` == the row). ValueError unless every row has exactly one such view and the
    views belong to these rows (match, reference time and frame number agree with S1), and unless the cells read are finite.

    (FLOW-3)의 통제 가운데 S2 보기에서 읽는 열을 만든다: 오브젝트 상태의 파랑 − 빨강 차이 13개와 바론·장로 버프 표시의 파랑 − 빨강 2개이다(식은 window_table에 있다).
    """
    n = len(rows)
    differences, buffs, used = _summary_plan()
    out = np.empty((n, len(differences) + len(buffs)))
    if n == 0:
        return out
    src = np.array(s2.array('views/src'), dtype=np.int64)
    src_index = np.array(s2.array('views/src_index'), dtype=np.int64)
    n_rows = len(s1.array('frame_rows/k'))
    frame_views = np.flatnonzero(src == 0)
    claimed = src_index[frame_views]
    if claimed.size and ((claimed < 0) | (claimed >= n_rows)).any():
        raise ValueError('a frame-clock view of S2 names a frame row that the S1 store does not have')
    if len(np.unique(claimed)) != len(claimed):
        raise ValueError('S2 holds more than one frame-clock view for a frame row')
    view_of = np.full(n_rows, -1, dtype=np.int64)
    view_of[claimed] = frame_views
    view = view_of[rows]
    if (view < 0).any():
        raise ValueError(f'S2 has no frame-clock view for {int((view < 0).sum())} of the {n} window rows')
    for name, want in (('views/match', match), ('views/ref', t0), ('views/frame_k', k)):
        bad = int((np.asarray(s2.array(name))[view].astype(np.int64) != want).sum())
        if bad:
            raise ValueError(f'{name} differs from the S1 frame rows in {bad} of the {n} window rows: this S2 store was not built from '
                             f'this S1 store')
    summary = s2.array('x/summary')
    cells = np.empty((n, len(used)))
    for lo in range(0, n, _CHUNK):
        cells[lo:lo + _CHUNK] = np.asarray(summary[view[lo:lo + _CHUNK]])[:, list(used)]
    if not np.isfinite(cells).all():
        raise ValueError(f'x/summary holds {int((~np.isfinite(cells)).sum())} cells that are not finite in the columns of the controls')
    at = {c: i for i, c in enumerate(used)}
    for q, (blue, red) in enumerate(differences):
        out[:, q] = cells[:, [at[c] for c in blue]].sum(axis=1) - cells[:, [at[c] for c in red]].sum(axis=1)
    for q, (blue, red) in enumerate(buffs, start=len(differences)):
        out[:, q] = (cells[:, at[blue]] > 0).astype(np.float64) - (cells[:, at[red]] > 0).astype(np.float64)
    return out


def window_table(s1, s2, roles=('FIT',)) -> WindowTable:
    """The window table of the matches of ``roles`` (FIT, STOP, DEV): the S1 store ``s1`` and the S2 store ``s2`` built from it; see the
    module text.

    수식
    ----
    (FLOW-3)  창 표의 행 i: 경기 u의 정규 프레임 k_i (t₀ = ts[k_i]), 창은 다음 한 분 (ts[k_i], ts[k_i + 1]]
              y_i = lead[k_i + 1] − lead[k_i]
              treat_iψ = Σ_{오브젝트 ∈ 분 k_i + 1, 묶음 ψ} σ(가져간 팀)                                  (ψ마다; 파랑 − 빨강)
              co_i = (킬 골드, 킬 수, 도움 수)의 분 k_i + 1 합                                              (파랑 − 빨강; 킬 골드는 기록 bounty + shutdown)
              ctrl_i = ( lead[k_i]·ind(t₀) (8), ind(t₀) (8), lead[k_i] − lead[k_i − 1], Σ_{분 ≤ k_i}(파랑 − 빨강 챔피언 킬 수),
                         S2 보기의 파랑 − 빨강 오브젝트 상태 (13), 바론·장로 버프 표시의 파랑 − 빨강 (2) )      (33열; 창 시작 상태)
              ind(t₀) = 시기 ph(t₀)의 원핫 벡터(8개).   행은 k_i + 1 ≤ F − 1일 때만 남기고, 아니면 빼서 센다.

      기호            | 뜻                                                  | 단위         | 범위      | 코드 변수
      i, u, k_i       | 창 표의 행(분 프레임 행), 경기, 그 정규 프레임      | —            | k_i ≥ 1   | 행, match, k
      y_i             | 프레임 뒤 한 분의 우위 변화(파랑 − 빨강)            | 골드         | 부호 있음 | y
      treat_iψ, co_i  | 그 분에 묶음 ψ를 가져간 수; 킬 골드, 킬 수, 도움 수 | 개; 골드, 개 | 부호 있음 | treat, co
      ctrl_i, ind, ph | 창 시작 상태의 33열, 시기의 원핫 벡터, 시기 번호    | 혼합         | —         | ctrl (열은 CTRL_NAMES), phase_of

    해석
    ----
    식 (FLOW-3)의 y_i는 프레임 k_i 뒤 한 분의 파랑 우위 변화이고 treat는 그 분에 묶음마다 가져간 오브젝트 수(파랑 − 빨강)이며, 사건이 없는 분이 반사실이다(계획 P1). 통제는 창 시작 상태(우위와 시기,
    기세, 누적 킬 차, S2 프레임 시계 보기의 오브젝트 상태와 버프)와 같은 분의 킬이고, 창 시작 상태는 결과보다 먼저 정해진 값이어서 통제로 쓸 수 있다(검산 노트 03 A절의 A3). 다음 정규 프레임이 없는 마지막
    행은 우위 변화를 알 수 없어 표에서 빼고 세며, 남은 어느 창에도 들지 않는 오브젝트(대개 경기 끝 가까이의 것)는 묶음별로 센다(판정 R5). S2 보기가 S1의 행과 맞지 않으면 다른 자료에서 만든 것이므로 거부한다.
    시기 표시 여덟 개는 한 행에서 합이 1이어서 (FLOW-5)의 절편 구실을 한다.

    근거
    ----
    계획 P1(분 창 회귀의 계수; 창 시작 상태 통제), Task 1(창 표), 계획 전역 제약의 시기 구간, 판정 R5, 설계서 8.1절(즉시값), 검산 노트 math-control-03 A절(창 시작 상태 통제)과
    B절(손 계산; 시험이 같은 수를 확인한다).
    """
    rd = _S1Reader(s1, roles)
    row_k, row_t0, row_match = s1.array('frame_rows/k'), s1.array('frame_rows/t0'), s1.array('frame_rows/match')
    parts = {name: [] for name in ('match', 'row', 'k', 't0', 'y', 'treat', 'co', 'ctrl')}
    dropped, n_unknown = 0, 0
    unwindowed = np.zeros(N_GROUPS, dtype=np.int64)
    for m in rd.matches:
        m = int(m)
        d = rd.load(m)
        n_unknown += d.n_unknown
        F = len(d.ts)
        r0, r1 = int(rd.off['row_off'][m]), int(rd.off['row_off'][m + 1])
        rows = np.arange(r0, r1, dtype=np.int64)
        k = np.array(row_k[r0:r1], dtype=np.int64)
        t0 = np.array(row_t0[r0:r1], dtype=np.int64)
        if (np.array(row_match[r0:r1], dtype=np.int64) != m).any() or ((k < 1) | (k >= F)).any() or (d.ts[k] != t0).any():
            raise ValueError(f'match row {m}: frame_rows/* does not agree with the offsets or with frames/ts')
        # 식 (FLOW-3)의 행 조건: 분 k + 1을 끝내는 정규 프레임이 있는 행만 남기고 나머지는 빼서 센다(판정 R5).
        keep = k + 1 <= F - 1                                        # the minute after frame k ends at a regular frame
        dropped += int((~keep).sum())
        rows, k, t0 = rows[keep], k[keep], t0[keep]
        kept_minute = np.zeros(F + 1, dtype=bool)
        kept_minute[k + 1] = True
        unwindowed += np.bincount(d.o_group[~kept_minute[d.o_minute]], minlength=N_GROUPS).astype(np.int64)
        # 식 (FLOW-3)의 창 시작 상태 ctrl의 앞 열: 우위 × 시기 원핫, 시기 원핫, 기세 lead[k] − lead[k − 1], 분 k까지의 누적 킬 차. 나머지 열은 아래에서 S2 보기로 채운다.
        phase, at = phase_of(t0).astype(np.int64), np.arange(len(k))
        ctrl = np.zeros((len(k), len(CTRL_NAMES)))
        ctrl[at, phase] = d.lead[k]
        ctrl[at, N_PHASES + phase] = 1.0
        ctrl[:, 2 * N_PHASES] = d.lead[k] - d.lead[k - 1]
        ctrl[:, 2 * N_PHASES + 1] = np.cumsum(d.kill_count)[k]
        # 식 (FLOW-3)의 y_i, treat, co: 프레임 뒤 한 분의 값이다.
        for name, value in (('match', np.full(len(k), m, dtype=np.int64)), ('row', rows), ('k', k), ('t0', t0),
                            ('y', d.lead[k + 1] - d.lead[k]), ('treat', d.obj[k + 1]),
                            ('co', np.column_stack([d.kill_gold[k + 1], d.kill_count[k + 1], d.assist_count[k + 1]])), ('ctrl', ctrl)):
            parts[name].append(value)
    rows = _cat(parts['row'], (), np.int64)
    match = _cat(parts['match'], (), np.int64)
    k = _cat(parts['k'], (), np.int64)
    ctrl = _cat(parts['ctrl'], (len(CTRL_NAMES),), np.float64)
    # 식 (FLOW-3)의 통제 가운데 S2 보기에서 읽는 열을 채운다.
    ctrl[:, _FIRST_SUMMARY:] = _summary_controls(s1, s2, rows, match, k, _cat(parts['t0'], (), np.int64))
    return WindowTable(
        match=narrow('match', match, np.int32), row=rows, k=narrow('k', k, np.int32), y=_cat(parts['y'], (), np.float64),
        treat=_cat(parts['treat'], (N_GROUPS,), np.float64), co=_cat(parts['co'], (3,), np.float64), ctrl=ctrl, n_dropped_last=dropped,
        n_obj_unwindowed=unwindowed, n_unknown=n_unknown, n_matches=len(rd.matches))


# =====================================================================================================================================
# Estimators (part 2)
# =====================================================================================================================================
# ---- per-match sufficient statistics --------------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class MatchStats:
    """The normal equations of a regression, summed per match: ``xtx[u]`` = X'X and ``xty[u]`` = X'y over the rows of the match
    ``match[u]`` (the seven rows of each event for the flow price, the window rows for the immediate value); ``n_rows[u]`` counts its events
    or window rows; ``columns`` names the d columns of X. The matches are in increasing order of the S1 match row and appear once each. The
    sum over any set of matches, with or without weights, is the normal equations of the regression on those matches.

    (FLOW-6)의 입력 X_u'X_u와 X_u'y_u를 경기 u마다 담는다. 합의 가중치만 바꾸면 되므로 부트스트랩이 재적합 없이 복제를 얻는다((FLOW-7)).
    """
    xtx: np.ndarray       # (M, d, d) float64, symmetric
    xty: np.ndarray       # (M, d) float64
    match: np.ndarray     # (M,) int64
    n_rows: np.ndarray    # (M,) int64
    columns: tuple        # d names

    def __post_init__(self):
        m, p = self.xty.shape
        if self.xtx.shape != (m, p, p) or self.match.shape != (m,) or self.n_rows.shape != (m,) or len(self.columns) != p:
            raise ValueError(f'MatchStats: xtx {self.xtx.shape}, xty {self.xty.shape}, match {self.match.shape}, '
                             f'n_rows {self.n_rows.shape} and {len(self.columns)} column names do not fit')


def _sorted_by_match(match: np.ndarray, *arrays) -> tuple:
    """``match`` and the row arrays in increasing order of ``match`` (stable); unchanged when they are in order already."""
    if match.size > 1 and (np.diff(match) < 0).any():
        order = np.argsort(match, kind='stable')
        return (match[order], *(a[order] for a in arrays))
    return (match, *arrays)


def _by_match(match: np.ndarray, x: np.ndarray, y: np.ndarray, columns: tuple, unit: int = 1) -> MatchStats:
    """``MatchStats`` of the rows ``x`` (n, d) and ``y`` (n,) of the matches ``match`` (n,), in increasing order of the match so that the
    rows of a match are together. ``unit`` rows make one counted row (7 for an event).

    (FLOW-6)의 경기별 합 X_u'X_u와 X_u'y_u를 계산한다.
    """
    n, p = x.shape
    if n:
        starts = np.flatnonzero(np.r_[True, match[1:] != match[:-1]])
        ends = np.r_[starts[1:], n]
    else:
        starts = ends = np.empty(0, dtype=np.int64)
    xtx, xty = np.empty((len(starts), p, p)), np.empty((len(starts), p))
    for i, (a, b) in enumerate(zip(starts.tolist(), ends.tolist())):
        block = x[a:b]
        gram = block.T @ block
        xtx[i] = 0.5 * (gram + gram.T)                                # a product of this form is symmetric; this makes it exactly so
        xty[i] = block.T @ y[a:b]
    return MatchStats(xtx=xtx, xty=xty, match=match[starts].astype(np.int64), n_rows=((ends - starts) // unit).astype(np.int64),
                      columns=tuple(columns))


def stats_from_design(x, y, match, columns=None) -> MatchStats:
    """The ``MatchStats`` of a design matrix ``x`` (n, d), an outcome ``y`` (n,) and the integer match of each row ``match`` (n,), in any
    order. ValueError for shapes that do not fit and for a value that is not finite. It fits a hand-made design (note 03).

    손으로 만든 설계(x, y, 경기 번호)에서 (FLOW-6)의 경기별 합을 만드는 입구이며, 검산 노트 03의 손 계산을 같은 풀이에 넣어 맞대는 데 쓴다.
    """
    x, y, match = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), np.asarray(match)
    if x.ndim != 2 or y.shape != (len(x),) or match.shape != (len(x),) or not np.issubdtype(match.dtype, np.integer):
        raise ValueError(f'x, y and match must be an (n, p) array, an (n,) array and an (n,) array of integers, got {x.shape}, '
                         f'{y.shape} and {match.shape} {match.dtype}')
    bad = int((~np.isfinite(x)).sum() + (~np.isfinite(y)).sum())
    if bad:
        raise ValueError(f'{bad} values of the design or of the outcome are not finite')
    columns = tuple(f'x{i}' for i in range(x.shape[1])) if columns is None else tuple(columns)
    match, x, y = _sorted_by_match(match.astype(np.int64), x, y)
    return _by_match(match, x, y, columns)


def _group_index(group) -> int:
    """The index in ``GROUPS`` of a group given by name or by index; ValueError for anything else."""
    if isinstance(group, str):
        if group not in GROUPS:
            raise ValueError(f'unknown group {group!r}: not one of {", ".join(GROUPS)}')
        return GROUPS.index(group)
    if isinstance(group, (bool, np.bool_)) or not isinstance(group, (int, np.integer)) or not 0 <= group < N_GROUPS:
        raise ValueError(f'unknown group {group!r}: a name of GROUPS or an index 0 .. {N_GROUPS - 1}')
    return int(group)


def flow_stats(table: EventTable, group) -> MatchStats:
    """The statistics of the flow-price regression of one group (a name of ``GROUPS`` or its index): the complete events of the group, each
    with its seven rows demeaned inside the event (the event effect), the columns ``FLOW_COLUMNS`` (the six relative-minute indicators,
    r = -1 left out, and the 13 ``co`` columns), d = 19. ``n_rows`` counts events. The incomplete events (NaN) are not used. ValueError for
    a complete event whose ``y`` or ``co`` is not finite.

    수식
    ----
    (FLOW-4)  y[e, r] = α_e + Σ_{r' ≠ −1} β_{r'}·1[r = r'] + γ·co[e, r] + 오차,      r = −4, …, +2,   β_{−1} = 0
              x[e, r] = (1[r = −4], 1[r = −3], 1[r = −2], 1[r = 0], 1[r = 1], 1[r = 2], co[e, r]) ∈ ℝ^d,      d = 6 + 13 = 19
              x̃[e, r] = x[e, r] − (1/7)·Σ_{r'} x[e, r'],      ỹ[e, r] = y[e, r] − (1/7)·Σ_{r'} y[e, r']      (사건 안 일곱 줄의 평균을 뺀다)
              θ = (β_{−4}, β_{−3}, β_{−2}, β_0, β_1, β_2, γ)는 경기 u의 사건들의 x̃, ỹ로 만든 X_u'X_u, X_u'y_u에서 (FLOW-6)이 푼다.

      기호   | 뜻                                                  | 단위          | 범위          | 코드 변수
      e, r   | 완전한 사건 번호, 상대 분                           | —, 분         | …; −4, …, +2  | 행, REL_MINUTES
      α_e    | 사건 e의 수준(사건 고정효과)                        | 골드          | 유한          | — (평균을 빼서 없앤다)
      β_r, γ | 상대 분 r의 계수(β_{−1} = 0이 기준), co의 계수 13개 | 골드, 골드/co | 유한 또는 NaN | theta[:6], theta[6:]
      x̃, ỹ   | 사건 안 평균을 뺀 설계 행과 반응                    | —, 골드       | 유한          | x, y
      d      | 설계의 열 수(상대 분 지시 6 + co 13)                | 열            | 19            | FLOW_COLUMNS

    해석
    ----
    식 (FLOW-4)는 한 묶음의 완전한 사건들을 쌓아 상대 분마다 하나의 계수 β_r을 두는 쌓인 사건 연구이고, 사건마다 자기 수준 α_e를 두는 사건 고정효과가 사건 안에서 일정한 것을 관측된 것과 관측되지 않은
    것 모두 흡수한다(계획 P2). 고정효과는 사건 안 평균을 y와 모든 열에서 빼는 형태로 맞추며 이는 사건마다 더미를 둔 회귀와 같은 계수를 낸다(Frisch–Waugh–Lovell). 사건마다 일곱 줄이 모두 있는 균형
    패널에서는 사건 단위 통제가 가격과 앞 추세의 대비를 바꾸지 못하므로 통제는 같은 상대 분의 다른 사건 co만 둔다(계획 P2).
    사건 시점의 우위 lead[k − 1] 같은 사건 시점 상태는 사건 안에서 일정하여 넣어도 고정효과에 흡수되므로 따로 두지 않고, 결과보다 먼저 정해진 상태 통제는 즉시값의 창 회귀에 둔다(계획 P2, 검산 노트 03 A3).
    완전하지 않은 사건은 쓰지 않고, 완전한 사건의 y나 co가 유한하지 않으면 거부한다(계획 전역 제약의 D7 항).

    근거
    ----
    계획 P2(사건 고정효과와 같은 상대 분의 다른 사건을 통제), 설계서 8.1절(쌓은 사건 연구), 검산 노트 math-control-03 A절(고정효과와 균형 패널의 손 계산; 시험이 같은 수를 확인한다).
    Ref: Cengiz, D., Dube, A., Lindner, A. and Zipperer, B. (2019), Quarterly Journal of Economics 134(3): 1405-1454;
         doi:10.1093/qje/qjz014 (사건을 쌓아 상대 시간 계수를 맞춘다). Frisch, R. and Waugh, F. V. (1933), Econometrica 1(4): 387-401;
         doi:10.2307/1907330, and Lovell, M. C. (1963), JASA 58(304): 993-1010; doi:10.1080/01621459.1963.10480682 (더미 회귀는 집단 평균을 뺀 회귀와 같다).
    """
    g = _group_index(group)
    sel = np.flatnonzero((table.group == g) & table.complete)
    match, sel = _sorted_by_match(np.asarray(table.match[sel], dtype=np.int64), sel)
    y, co = table.y[sel], table.co[sel]
    bad = int((~np.isfinite(y)).sum() + (~np.isfinite(co)).sum())
    if bad:
        raise ValueError(f'group {GROUPS[g]}: {bad} values of its complete events (y or co) are not finite')
    n = len(sel)
    # 식 (FLOW-4)의 설계 x[e, r]: r = −1을 뺀 상대 분 지시 여섯 개와 co 열세 개이다.
    x = np.concatenate([np.broadcast_to(_DUMMIES, (n, N_REL, len(_KEEP))), co], axis=2)          # (n, 7, 19)
    # 식 (FLOW-4)의 사건 고정효과 α_e: 사건 안 일곱 줄의 평균을 x와 y에서 뺀다. 사건마다 더미를 둔 회귀와 같은 계수이다.
    x -= x.mean(axis=1, keepdims=True)                                                              # the event effect: the event mean out
    y = y - y.mean(axis=1, keepdims=True)
    return _by_match(np.repeat(match, N_REL), x.reshape(-1, x.shape[2]), y.reshape(-1), FLOW_COLUMNS, unit=N_REL)


def window_stats(wtable: WindowTable) -> MatchStats:
    """The statistics of the window regression of the immediate values: one row per window row, the columns ``WINDOW_COLUMNS``
    (treat 12, co 3, ctrl 33), d = 48, no intercept. ``n_rows`` counts window rows. ValueError for a value that is not finite.

    수식
    ----
    (FLOW-5)  y_i = Σ_ψ θ_ψ·treat_iψ + θ_co·co_i + θ_ctrl·ctrl_i + 오차                         (절편 없음)
              x_i = (treat_i, co_i, ctrl_i) ∈ ℝ^d,      d = 12 + 3 + 33 = 48,      θ = (θ_ψ (12개), θ_co (3개), θ_ctrl (33개))는 (FLOW-6)이 푼다.
              θ_ψ = 묶음 ψ의 즉시값,      θ_co의 첫 성분 = 창의 킬 골드의 계수(kill_gold_coef)

      기호         | 뜻                                              | 단위      | 범위          | 코드 변수
      x_i, d       | 창 표의 행 i의 설계 행(treat, co, ctrl)과 열 수 | —, 열     | 유한; 48      | x, WINDOW_COLUMNS
      θ_ψ          | 묶음 ψ의 즉시값(treat의 계수)                   | 골드/개   | 유한 또는 NaN | theta[:12]
      θ_co, θ_ctrl | co와 ctrl의 계수                                | 열의 단위 | 유한 또는 NaN | theta[12:]

    해석
    ----
    식 (FLOW-5)는 한 분의 우위 변화를 그 분에 가져간 묶음별 오브젝트 수에 회귀한 것이며, 사건이 없는 분이 반사실이므로 θ_ψ가 묶음 ψ의 즉시값, 곧 창 끝에서 본 효과이다(계획 P1). 앞 분을 기준으로
    쓰지 않으므로 앞 추세 시험에 실패한 묶음에도 쓸 수 있다(계획 P1). 통제는 창 시작 상태와 같은 분의 킬(골드, 수, 도움 수)이다(계획 P1). 절편은 두지 않는다: 시기 표시 여덟 개가 한 행에서 합이 1이어서 그
    구실을 한다(계획 Task 2). 파랑이 가져간 오브젝트는 +1, 빨강이 가져간 것은 −1로 세고 반응도 파랑 − 빨강이므로 θ_ψ는 두 팀에 같은 크기라고 둔 가져간 팀의 이득이다(검산 노트 03 B절).

    근거
    ----
    계획 P1(즉시값은 분 창 회귀의 계수), Task 2(설계 행렬, 절편 없음), 설계서 8.1절(즉시값), 검산 노트 math-control-03 B절(손 계산; 시험이 같은 수를 확인한다).
    """
    x = np.column_stack([wtable.treat, wtable.co, wtable.ctrl]).astype(np.float64)
    y = np.asarray(wtable.y, dtype=np.float64)
    bad = int((~np.isfinite(x)).sum() + (~np.isfinite(y)).sum())
    if bad:
        raise ValueError(f'the window table holds {bad} values that are not finite')
    match, x, y = _sorted_by_match(np.asarray(wtable.match, dtype=np.int64), x, y)
    return _by_match(match, x, y, WINDOW_COLUMNS)


# ---- solving, the bootstrap and the price formulas -----------------------------------------------------------------------------------
def _check_count(name: str, value, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f'{name} must be an int of at least {minimum}, got {value!r}')
    return int(value)


def _weighted_sums(stats: MatchStats, w) -> tuple:
    """``(sum_u w_u X'X_u, sum_u w_u X'y_u)``; every weight 1 when ``w`` is None. ValueError unless ``w`` holds one finite number >= 0 per
    match.

    (FLOW-6)의 Σ_u w_u·X_u'X_u와 Σ_u w_u·X_u'y_u를 계산한다(식은 solve에 있다).
    """
    if w is None:
        return stats.xtx.sum(axis=0), stats.xty.sum(axis=0)
    w = np.asarray(w, dtype=np.float64)
    if w.shape != (len(stats.match),) or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError(f'the weights must be one finite number >= 0 for each of the {len(stats.match)} matches, got shape {w.shape}')
    return np.tensordot(w, stats.xtx, axes=1), w @ stats.xty


def _equilibrate(a: np.ndarray) -> tuple:
    """Equilibrate a stack of normal matrices ``a`` (n, d, d; each X'WX, positive semi-definite): ``(scaled, scale, live, eigen, tol)``.

    ``scaled`` is ``a`` with each column scaled to a unit diagonal, so that columns of very different scale do not decide the rank. A
    column whose diagonal is exactly 0 has no variation in the weighted sample (and its row and column of ``a`` are 0): it is not ``live``
    and becomes a unit vector. ``eigen`` are the eigenvalues of ``scaled`` (ascending) and ``tol`` is d eps times the largest of them: an
    eigenvalue at or below it makes the live columns collinear to double precision.

    (FLOW-6)의 열 평형화와 고유값 검사를 계산한다(식은 solve에 있다).
    """
    p = a.shape[1]
    a = 0.5 * (a + a.transpose(0, 2, 1))
    diag = np.einsum('nii->ni', a)
    live = diag > 0.0
    scale = np.where(live, 1.0 / np.sqrt(np.where(live, diag, 1.0)), 0.0)
    scaled = a * scale[:, :, None] * scale[:, None, :]
    scaled[:, np.arange(p), np.arange(p)] = 1.0                       # a column without variation becomes a unit vector
    eigen = np.linalg.eigvalsh(scaled)
    return scaled, scale, live, eigen, p * np.finfo(np.float64).eps * eigen[:, -1]


def _solve_normal(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve a stack of normal equations ``a[i] theta = b[i]`` (a (n, d, d), b (n, d)) and return the (n, d) coefficients, NaN where they
    are not estimable: the coefficient of a column without variation (the others are estimated without it), and every coefficient of a
    system whose live columns are collinear (see ``_equilibrate``).

    (FLOW-6)의 해와 NaN 규칙을 계산한다(식은 solve에 있다).
    """
    n, p = b.shape
    scaled, scale, live, eigen, tol = _equilibrate(a)
    ok = eigen[:, 0] > tol
    out = np.full((n, p), np.nan)
    if ok.any():
        theta = np.linalg.solve(scaled[ok], (b[ok] * scale[ok])[:, :, None])[:, :, 0] * scale[ok]
        theta[~live[ok]] = np.nan
        out[ok] = theta
    return out


def solve(stats: MatchStats, w=None) -> np.ndarray:
    """The (d,) coefficients of the regression on the matches of ``stats`` with the match weights ``w`` (all 1 if None): NaN for a column
    without variation and for every column when the system is singular or empty (see ``_solve_normal``).

    수식
    ----
    (FLOW-6)  XX(w) = Σ_u w_u·X_u'X_u,     Xy(w) = Σ_u w_u·X_u'y_u,       w_u = 1 (w = None)
              변하는 열 L = {c : XX(w)_cc > 0},   D = diag(D_c),   D_c = 1/√XX(w)_cc (c ∈ L),   D_c = 0 (c ∉ L)
              Ã = D·XX(w)·D (c ∉ L인 행과 열은 단위 벡터로 바꾼다),   θ̂_c = [D·Ã⁻¹·D·Xy(w)]_c (c ∈ L),   θ̂_c = NaN (c ∉ L)
              Ã의 가장 작은 고유값 ≤ d·eps·(가장 큰 고유값)이면 θ̂ 전체가 NaN이다 (eps = 부동소수점의 기계 엡실론).

      기호         | 뜻                                              | 단위 | 범위          | 코드 변수
      w_u          | 경기 u의 가중치                                 | —    | 유한, ≥ 0     | w[u]
      XX(w), Xy(w) | 가중한 경기별 합 (d×d 행렬, d개 벡터)           | …    | 대칭 행렬     | xtx 합, xty 합 (_weighted_sums)
      L, D, Ã      | 변하는 열의 집합, 열의 크기 보정(대각 행렬), 평형화한 행렬 | — | D_c > 0 (c ∈ L) | live, scale, scaled
      θ̂_c          | 열 c의 계수                                     | …    | 유한 또는 NaN | 반환값

    해석
    ----
    식 (FLOW-6)은 경기별 합으로 쓴 정규방정식 (Σ_u w_u X_u'X_u)θ = Σ_u w_u X_u'y_u의 해이며, 가중치가 모두 1이면 모든 경기의 행을 한꺼번에 쓴 최소제곱과 같고 경기 가중치는 부트스트랩이 재표본을 가중 합으로
    나타내게 한다. 열마다 대각을 1로 맞추는 평형화를 먼저 하는 것은 열의 크기가 크게 달라(우위에 시기 표시를 곱한 열과 표시 열) 크기가 순위 판정을 정하지 않게 하기 위해서이다. 대각이 정확히 0인 열(가중 표본에서 값이 모두 0인 열; 흐름 회귀에서는 사건 안에서 변하지 않는 열)은
    정규방정식에 아무것도 더하지 않으므로 그 열의 계수만 NaN으로 두고 나머지는 그 열 없이 정확히 추정한다(판정 R9). 변하는 열들이 공선이면 해가 하나로 정해지지 않으므로 벌점이나 값 채우기로 해를 고르지 않고
    모든 계수를 NaN으로 두며, 이때 공선에 들지 않아 추정할 수 있는 열의 계수도 함께 비운다(Task 2 검토의 열린 항목). 공선 판정의 문턱 d·eps·(가장 큰 고유값)은 큰 비정수 설계에서 거의 정확한 공선성을 놓칠 수 있다(Task 2 검토의 열린 항목).

    근거
    ----
    판정 R9(변하지 않는 열만 NaN), 계획 Task 2(경기 가중을 받는 정규방정식의 해), 검산 노트 math-control-03 A절·B절(손 계산; 시험이 같은 수를 확인한다).
    """
    xtx, xty = _weighted_sums(stats, w)
    return _solve_normal(xtx[None], xty[None])[0]


def rank_of(stats: MatchStats, w=None) -> dict:
    """``{'live': ..., 'rank': ...}`` of the normal equations of ``stats`` with the match weights ``w``: the number of columns that vary
    and the rank of those columns (equilibrated, with the tolerance of ``solve``). They are equal exactly when ``solve`` can estimate the
    columns that vary; ``live - rank`` is the number of independent linear relations among them.

    (FLOW-6)이 추정할 수 있는지를 센다: 변하는 열의 수와 그 열들의 순위는 공선일 때 달라지며, price_table이 묶음마다 보고한다.
    """
    xtx, _ = _weighted_sums(stats, w)
    _, _, live, eigen, tol = _equilibrate(xtx[None])
    return {'live': int(live[0].sum()), 'rank': int((eigen[0] > tol[0]).sum() - (~live[0]).sum())}


def bootstrap(stats: MatchStats, B: int = BOOT_B, seed: int = BOOT_SEED) -> np.ndarray:
    """(B, d) coefficients of ``B`` match-bootstrap replicates. Replicate j has the match weights of the j-th draw of one stream,
    ``numpy.random.default_rng(seed).multinomial(M, 1/M)`` with M the number of matches in ``stats``, and is solved from the weighted sums
    of the per-match X'X and X'y. The weights are drawn one replicate at a time, so the result does not depend on how the replicates are
    chunked. A singular replicate is a row of NaN (``singular_count``); a column without variation in a replicate is NaN in it. Empty
    ``stats`` give an array of NaN.

    수식
    ----
    (FLOW-7)  θ*_j = (FLOW-6)의 해(w = d_j),       d_j = (d_j1, …, d_jM) ~ Multinomial(M; 1/M, …, 1/M)       (j = 1, …, B)
              d_j는 default_rng(seed)의 한 난수열에서 복제마다 한 번씩 차례로 뽑으며 (NLL-5)와 같은 뽑기이다. 특이한 복제는 θ*_j 전체가 NaN이다.

      기호      | 뜻                                      | 단위 | 범위          | 코드 변수
      M, u      | stats의 경기 수, 경기 번호              | —    | ≥ 1; 1, …, M  | m = len(stats.match)
      d_ju, d_j | 복제 j에서 경기 u가 뽑힌 횟수와 그 벡터 | 회   | 0, …, M       | weights (복제마다 한 행)
      θ*_j      | 복제 j의 계수 벡터                      | …    | 유한 또는 NaN | out[j − 1]
      B, seed   | 복제의 수, 난수 씨앗                    | —    | ≥ 1; ≥ 0      | B, seed

    해석
    ----
    식 (FLOW-7)은 경기를 균등하게 M번 복원 추출한 횟수 d_ju를 경기 가중치로 쓴 (FLOW-6)의 해이며, 경기별 합에 가중만 곱하므로 행 전체를 다시 맞추지 않고 복제를 얻고 같은 경기 안 행들의 상관이 보존된다.
    복제 j의 횟수 d_j는 경기 수와 씨앗과 복제 번호에만 의존하고, 복제를 하나씩 뽑으므로 B와 복제를 묶는 크기에 의존하지 않는다. 특이한 복제는 전체가 NaN이고 따로 세며, 한 복제에서 변하지 않는 열은
    그 열만 NaN이다. 경기가 하나도 없으면 NaN 배열을 돌려준다.

    근거
    ----
    계획 전역 제약의 D7 항(B와 시드), 설계서 8.1절(경기 부트스트랩), (NLL-5)(같은 뽑기).
    Ref: Efron, B. (1979), Annals of Statistics 7(1): 1-26; doi:10.1214/aos/1176344552 (표본을 다시 뽑아 통계량의 분포를 얻는다).
    Ref: Cameron, A. C., Gelbach, J. B. and Miller, D. L. (2008), Review of Economics and Statistics 90(3): 414-427;
         doi:10.1162/rest.90.3.414 (군집을 통째로 다시 뽑는다).
    """
    B = _check_count('B', B, 1)
    seed = _check_count('seed', seed, 0)
    m, p = len(stats.match), stats.xty.shape[1]
    out = np.full((B, p), np.nan)
    if m == 0:
        return out
    rng = np.random.default_rng(seed)
    pvals = np.full(m, 1.0 / m)
    flat = stats.xtx.reshape(m, p * p)
    chunk = max(1, min(B, _BOOT_BYTES // (8 * m)))
    for lo in range(0, B, chunk):
        hi = min(B, lo + chunk)
        weights = np.empty((hi - lo, m))
        for i in range(hi - lo):
            weights[i] = rng.multinomial(m, pvals)
        out[lo:hi] = _solve_normal((weights @ flat).reshape(-1, p, p), weights @ stats.xty)
    return out


def singular_count(boot: np.ndarray) -> int:
    """The number of replicates of ``bootstrap`` that were singular: the rows that are NaN throughout.

    (FLOW-7)의 복제 가운데 풀 수 없어 전체가 NaN으로 끝난 것의 수이다.
    """
    return int(np.isnan(boot).all(axis=1).sum())


def flow_beta(theta) -> np.ndarray:
    """beta_{-4..+2} (seven values, last axis) from the coefficients of the flow regression: its first six entries are the minutes
    r = -4, -3, -2, 0, +1, +2, and beta_{-1} = 0 is put back in the fourth place. Works on a stack of coefficient vectors.

    (FLOW-4)의 계수 θ의 앞 여섯 개에 기준 β_{−1} = 0을 넷째 자리에 넣어 β_{−4}, …, β_{+2}의 일곱 개를 만든다. (FLOW-8), (FLOW-9)의 입력이다.
    """
    theta = np.asarray(theta, dtype=np.float64)
    if theta.ndim == 0 or theta.shape[-1] < len(_KEEP):
        raise ValueError(f'theta needs at least {len(_KEEP)} entries (the relative minutes that are not r = -1), got shape {theta.shape}')
    return np.insert(theta[..., :len(_KEEP)], _REF, 0.0, axis=-1)


def _beta7(beta) -> np.ndarray:
    beta = np.asarray(beta, dtype=np.float64)
    if beta.ndim == 0 or beta.shape[-1] != N_REL:
        raise ValueError(f'beta needs {N_REL} entries (r = -4 .. +2) in its last axis, got shape {beta.shape}')
    return beta


def price_from_beta(beta7) -> np.ndarray:
    """The flow price p = sum_{r=0}^{2} beta_r - 3 mean(beta_{-4..-1}) of beta (seven values in the last axis; a stack is fine).

    수식
    ----
    (FLOW-8)  p = Σ_{r=0}^{2} β_r − 3·평균(β_{−4}, β_{−3}, β_{−2}, β_{−1}) = β_0 + β_1 + β_2 − (3/4)·(β_{−4} + β_{−3} + β_{−2} + β_{−1})

      기호 | 뜻                           | 단위 | 범위      | 코드 변수
      p    | 흐름 가격                    | 골드 | 부호 있음 | 반환값
      β_r  | 상대 분 r의 계수(β_{−1} = 0) | 골드 | 유한 또는 NaN | beta7 (마지막 축의 일곱 값)

    해석
    ----
    식 (FLOW-8)의 p는 사건 분과 그 뒤 두 분의 변화 합에서, 사건 전 네 분의 평균 변화를 같은 길이(세 분)로 늘린 평소 흐름을 뺀 값, 곧 사건 뒤 세 분 동안 평소보다 더 번 골드이다. 계수의 합이 0인 대비이므로
    기준을 r = −1로 잡은 것이 값을 바꾸지 않는다(검산 노트 03 A1). 쌓은 배열에는 같은 식을 마지막 축에 적용하며, β에 NaN이 있으면 p도 NaN이다.

    근거
    ----
    설계서 8.1절(가격의 정의), 계획 전역 제약의 D7 항, 검산 노트 math-control-03 A1(대비는 기준과 무관), 검산 노트 math-control-03 A절(손 계산; 시험이 같은 수를 확인한다).
    """
    b = _beta7(beta7)
    return b[..., R0:].sum(axis=-1) - 3.0 * b[..., :R0].mean(axis=-1)


def pretrend_from_beta(beta7) -> np.ndarray:
    """The pre-trend zeta = (beta_{-2} + beta_{-1}) - (beta_{-4} + beta_{-3}) of beta (seven values in the last axis; a stack is fine).

    수식
    ----
    (FLOW-9)  ζ = (β_{−2} + β_{−1}) − (β_{−4} + β_{−3})

      기호 | 뜻                           | 단위 | 범위      | 코드 변수
      ζ    | 앞 추세                      | 골드 | 부호 있음 | 반환값
      β_r  | 상대 분 r의 계수(β_{−1} = 0) | 골드 | 유한 또는 NaN | beta7 (마지막 축의 일곱 값)

    해석
    ----
    식 (FLOW-9)의 ζ는 사건 직전 두 분의 변화가 그 앞 두 분보다 가파른 정도이고, 0과 구별되면 앞 분 평균을 반사실로 믿을 수 없다는 뜻이어서 그 묶음은 즉시값으로 물러난다(설계서 8.1절; 검산 노트 03 A절).
    계수의 합이 0인 대비이므로 기준 r = −1의 선택에 따라 값이 달라지지 않는다(검산 노트 03 A1). 쌓은 배열에는 같은 식을 마지막 축에 적용한다.

    근거
    ----
    설계서 8.1절(앞 추세 시험), 계획 전역 제약의 D7 항(채택), 검산 노트 math-control-03 A1(대비는 기준과 무관), 검산 노트 math-control-03 A절(손 계산; 시험이 같은 수를 확인한다).
    """
    b = _beta7(beta7)
    return (b[..., 2] + b[..., 3]) - (b[..., 0] + b[..., 1])


# ---- the price table -------------------------------------------------------------------------------------------------------------------
def _number(x):
    """A float, or None for NaN and infinity (JSON has neither)."""
    x = float(x)
    return x if math.isfinite(x) else None


def _interval(values):
    """The percentile interval (2.5, 97.5) of the finite values as [lo, hi], or None when there is none.

    (FLOW-10)의 백분위 구간이다: 유한한 복제만 쓰며 하나도 없으면 구간이 없다(식은 price_table에 있다).
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    lo, hi = np.percentile(v, _PERCENTILES)
    return [float(lo), float(hi)]


def _without_variation(stats: MatchStats, start: str = '') -> list:
    """The names of the columns (starting with ``start``) that have no variation in the sum over all matches: an exactly zero diagonal.

    (FLOW-6)에서 변하지 않아 계수가 NaN이 되는 열의 이름을 모은다.
    """
    if not len(stats.match):
        return []
    return [c for c, d in zip(stats.columns, np.diagonal(stats.xtx.sum(axis=0))) if d == 0.0 and c.startswith(start)]


def _flow_fit(stats: MatchStats, B: int, seed: int) -> dict:
    """The flow fields of one group: beta, price and pre-trend with their intervals (price and pre-trend are recomputed on every replicate,
    not taken from the interval ends of beta), None where the fit is not estimable, and the number of usable replicates.

    (FLOW-8), (FLOW-9)의 값과 구간을 만든다: 점추정은 전 표본의 β에서, 구간은 복제마다 다시 계산한 p와 ζ에서 낸다(식은 price_table에 있다).
    """
    beta = flow_beta(solve(stats))
    beta_b = flow_beta(bootstrap(stats, B, seed)[:, :len(_KEEP)])
    price_b, trend_b = price_from_beta(beta_b), pretrend_from_beta(beta_b)
    intervals = [_interval(beta_b[:, j]) for j in range(N_REL)]
    return {'beta': [float(v) for v in beta] if np.isfinite(beta).all() else None,
            'beta_ci': intervals if all(i is not None for i in intervals) else None,
            'price_flow': _number(price_from_beta(beta)), 'price_flow_ci': _interval(price_b),
            'pretrend': _number(pretrend_from_beta(beta)), 'pretrend_ci': _interval(trend_b),
            'n_boot_flow': int(np.isfinite(price_b).sum())}


def price_table(etable: EventTable, wtable: WindowTable, B: int = BOOT_B, seed: int = BOOT_SEED, check=None) -> dict:
    """The flow price and the immediate value of every group, and which one is used (see the module text); a dict of plain JSON values.

    One entry per group name with the fields ``fixed_immediate``, ``n_events`` (complete) and ``n_excluded`` (incomplete) events,
    ``n_event_matches``, ``beta`` (seven values, beta_{-1} = 0) and ``beta_ci``, ``price_flow`` and ``price_flow_ci``, ``pretrend`` and
    ``pretrend_ci``, ``adopted`` (the interval of the pre-trend contains 0), ``n_boot_flow``, ``flow_controls_without_variation``,
    ``flow_rank``, ``immediate`` and ``immediate_ci``, ``n_boot_immediate``, ``used`` ('flow' or 'immediate'), ``value``,
    ``n_obj_unwindowed`` and its share ``obj_unwindowed_share`` of the objectives of the group. ``used`` is 'immediate' for a group in
    ``FIXED_IMMEDIATE``, for a group whose pre-trend is not adopted and for a group whose flow fit is not estimable (the flow fields are
    then None, the counts stay); ``value`` is the price flow or the immediate value that is used (None if that one is not estimable).
    Besides the groups: ``n_matches``, ``n_window_rows``, ``n_dropped_last``, ``n_unknown_objectives``, ``kill_gold_coef`` and its
    interval, ``window_controls_without_variation``, ``window_rank``, ``B`` and ``seed``. ``check`` (a function without arguments, for
    example the E6 guard) is called before the window fit and before each group. ValueError if the two tables were made from a different
    number of matches.

    수식
    ----
    (FLOW-10)  구간: 유한한 복제의 2.5, 97.5 백분위수(선형 보간). p*_j = (FLOW-8)(β*_j), ζ*_j = (FLOW-9)(β*_j)를 복제마다 다시 계산하며,
               β*_j는 (FLOW-7)의 복제 계수에 β_{−1} = 0을 넣은 일곱 값이다. 유한한 복제가 하나도 없으면 구간은 없다.
               채택_ψ ⇔ ζ의 구간 [ζ_lo, ζ_hi]가 ζ_lo ≤ 0 ≤ ζ_hi이다.
               쓰는 것_ψ = 즉시값 (ψ ∈ FIXED_IMMEDIATE, 또는 채택하지 않음, 또는 p를 낼 수 없음),  그 밖에는 흐름.
               value_ψ = p_ψ (흐름) 또는 θ_ψ (즉시값)

      기호         | 뜻                                            | 단위 | 범위           | 코드 변수
      p_ψ, ζ_ψ     | 묶음 ψ의 흐름 가격과 앞 추세(전 표본의 β에서) | 골드 | 부호 있음      | price_flow, pretrend
      p*_j, ζ*_j   | 복제 j의 흐름 가격과 앞 추세                  | 골드 | 유한 또는 NaN  | price_b, trend_b
      [ζ_lo, ζ_hi] | ζ의 2.5–97.5 백분위 구간                      | 골드 | —              | pretrend_ci
      θ_ψ          | 묶음 ψ의 즉시값 ((FLOW-5))                    | 골드 | 유한 또는 NaN  | immediate
      value_ψ      | 묶음 ψ의 쓰는 값(PRICES.json의 value)         | 골드 | 유한 또는 없음 | value

    해석
    ----
    식 (FLOW-10)은 앞 추세 ζ의 경기 부트스트랩 구간이 0을 포함할 때만 흐름 가격을 쓰는 채택 규칙이며, 포함하지 않으면 앞 분 평균을 반사실로 믿을 수 없으므로 앞 분을 기준으로 쓰지 않는 즉시값으로 물러난다(설계서 8.1절, 계획 P1).
    구간이 0을 포함하는 것은 앞 추세가 없다는 증거가 아니라 이 자료로 0과 구별되지 않는다는 뜻이며, 사건이 적은 묶음에서는 이 시험의 힘이 약하다.
    억제기 포탑, 넥서스 포탑, 억제기는 가져간 뒤 곧 경기가 끝나 뒤 흐름이 잘리는 일이 많으므로 늘 즉시값을 쓰며(설계서 8.1절), 흐름 쪽을 추정할 수 없는 묶음(공선 등)도 개수를 남기고 흐름 값을 비운 채 즉시값을 쓰고 다른 묶음은 계속한다(판정 R9).
    가격과 앞 추세의 구간은 복제마다 β에서 다시 계산한 값의 백분위 구간이며 β의 구간 끝에서 가격을 만들지 않는다: p와 ζ는 β들의 대비여서 β 사이의 상관이 구간에 들어가야 하기 때문이다 [도출].
    두 표가 서로 다른 수의 경기로 만들어졌으면 같은 경기의 것이 아니므로 거부한다.

    근거
    ----
    설계서 8.1절(채택 조건과 즉시값 고정), 계획 전역 제약의 D7 항(채택, 고정 묶음, B와 시드), 계획 P1(즉시값), 판정 R9(특이한 묶음은 개수를 남기고 값을 비운다), 관문 G-S3a(앞 추세 규칙 유지 승인).
    """
    B = _check_count('B', B, 1)
    seed = _check_count('seed', seed, 0)
    if etable.n_matches != wtable.n_matches:
        raise ValueError(f'the event table was made from {etable.n_matches} matches and the window table from {wtable.n_matches}: they '
                         'must be made from the same matches')
    tick = check if check is not None else (lambda: None)
    objectives = etable.counts_by_group()
    tick()
    # 식 (FLOW-5): 창 회귀는 한 번만 맞춘다. θ_ψ는 묶음마다, 킬 골드의 계수는 표 전체에 쓴다.
    ws = window_stats(wtable)
    theta, boot = solve(ws), bootstrap(ws, B, seed)
    table = {'n_matches': int(wtable.n_matches), 'n_window_rows': int(len(wtable.y)), 'n_dropped_last': int(wtable.n_dropped_last),
             'n_unknown_objectives': int(etable.n_unknown), 'kill_gold_coef': _number(theta[_KILL_GOLD]),
             'kill_gold_coef_ci': _interval(boot[:, _KILL_GOLD]), 'window_controls_without_variation': _without_variation(ws),
             'window_rank': rank_of(ws), 'B': B, 'seed': seed}
    for g, name in enumerate(GROUPS):
        tick()
        fs = flow_stats(etable, name)
        flow = _flow_fit(fs, B, seed)
        # 식 (FLOW-10)의 채택: ζ의 구간이 0을 포함할 때만이다(구간이 없으면 채택하지 않는다).
        pre = flow['pretrend_ci']
        adopted = pre is not None and pre[0] <= 0.0 <= pre[1]
        immediate = _number(theta[g])
        fixed = name in FIXED_IMMEDIATE
        # 식 (FLOW-10)의 쓰는 것: 고정 묶음, 채택하지 않은 묶음, 흐름 가격을 낼 수 없는 묶음은 즉시값이다.
        used = 'immediate' if fixed or not adopted or flow['price_flow'] is None else 'flow'
        known = int(objectives['events'][g])
        table[name] = {
            'fixed_immediate': fixed, 'n_events': int(objectives['complete'][g]), 'n_excluded': int(objectives['incomplete'][g]),
            'n_event_matches': len(fs.match), **flow, 'adopted': adopted, 'flow_controls_without_variation': _without_variation(fs, 'co_'),
            'flow_rank': rank_of(fs),
            'immediate': immediate, 'immediate_ci': _interval(boot[:, g]), 'n_boot_immediate': int(np.isfinite(boot[:, g]).sum()),
            'used': used, 'value': flow['price_flow'] if used == 'flow' else immediate,
            'n_obj_unwindowed': int(wtable.n_obj_unwindowed[g]),
            'obj_unwindowed_share': float(wtable.n_obj_unwindowed[g]) / known if known else None}
    return table


# ---- the comparison with the prototype ---------------------------------------------------------------------------------------------------
def _probe_pair(entry):
    """``(point, [lo, hi])`` of the horizon 0 of a prototype entry ({'point': [...], 'lo': [...], 'hi': [...]}); None when it is not there
    or not finite."""
    try:
        point, lo, hi = (_number(entry[key][0]) for key in ('point', 'lo', 'hi'))
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    return None if None in (point, lo, hi) else (point, [lo, hi])


def probe_block(table: dict, path) -> dict:
    """Our immediate values beside the prototype's (ruling R4): for each group ``immediate`` (ours), ``probe`` (``price[group]['point'][0]``
    of the prototype's JSON), its interval ``probe_ci`` and the ``difference``; the same for ``kill_gold_coef``. A report only: nothing is
    asserted. The file is read only if it exists; one that cannot be read gives ``error``, one without an entry gives None for it.

    (FLOW-5)의 즉시값 θ_ψ를 시제품의 값과 나란히 적는 보고 블록이다(판정 R4). 값을 단언하지 않으며, 파일이 있을 때만 읽는다.
    """
    path = Path(path)
    block = {'path': str(path), 'found': False}
    if not path.is_file():
        return block
    block['found'] = True
    try:
        probe = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(probe, dict):
            raise ValueError('the top level is not an object')
    except (OSError, ValueError) as e:
        block['error'] = f'{type(e).__name__}: {e}'
        return block
    block['probe_rows'], block['probe_matches'] = probe.get('rows'), probe.get('matches')
    groups = {}
    for name in GROUPS:
        pair = _probe_pair((probe.get('price') or {}).get(name)) if isinstance(probe.get('price'), dict) else None
        ours = table[name]['immediate']
        groups[name] = None if pair is None else {'immediate': ours, 'probe': pair[0], 'probe_ci': pair[1],
                                                  'difference': None if ours is None else ours - pair[0]}
    block['groups'] = groups
    pair = _probe_pair(probe.get('kill_gold_coef'))
    ours = table['kill_gold_coef']
    block['kill_gold_coef'] = None if pair is None else {'kill_gold_coef': ours, 'probe': pair[0], 'probe_ci': pair[1],
                                                         'difference': None if ours is None else ours - pair[0]}
    return block


# ---- command line ------------------------------------------------------------------------------------------------------------------------
class RunLocked(RuntimeError):
    """Another call holds the lock of the run folder."""


def _lock_message(path: Path) -> str:
    try:
        held = json.loads(path.read_text(encoding='utf-8'))
        pid, created = held.get('pid'), held.get('created')
    except (OSError, ValueError, AttributeError):
        pid = created = None
    who = f'made {created} by pid {pid}' if pid is not None else 'its content cannot be read'
    return (f'another call holds the lock {path} ({who}). Only one call may work on a run folder at a time. If none is running (a power '
            f'cut or a killed process leaves the lock behind), delete that file yourself; this command never deletes it.')


def _acquire_lock(run: Path) -> Path:
    """Create ``<run>/S3A.lock`` exclusively (the run folder is made when needed) and return its path; RunLocked if it is there. The
    file holds the pid and the time and nothing else."""
    run.mkdir(parents=True, exist_ok=True)
    path = run / LOCK_NAME
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, 'O_BINARY', 0))
    except FileExistsError:
        raise RunLocked(_lock_message(path)) from None
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as f:
            f.write(json.dumps({'pid': os.getpid(), 'created': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}) + '\n')
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def _release_lock(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError as e:
        print(f'WARNING: the lock {path} could not be removed ({type(e).__name__}); delete it before the next call on this run',
              file=sys.stderr, flush=True)


def _refuse(message: str) -> int:
    print(message, file=sys.stderr, flush=True)
    return 2


def _run_name(path) -> str:
    """The run folder name of a store: the folder itself, or the parent of a store folder (the one that holds STORE.json)."""
    path = Path(path)
    return path.parent.name if (path / 'STORE.json').is_file() else path.name


def _open_s2(path, verify: bool):
    """The S2 store of an S2 run folder, or of the store folder itself: ValueError / FileNotFoundError unless it is complete (and, with
    ``verify``, every file has its recorded SHA-256)."""
    from c_ept import store_core
    path = Path(path)
    root = path if (path / 'STORE.json').is_file() else path / S2_DIR
    if not (root / 'STORE.json').is_file():
        raise FileNotFoundError(f'no S2 store under {path}')
    store = store_core.Store(root)
    if not store.manifest.get('complete'):
        raise ValueError(f'the S2 store {root} is not complete')
    if verify:
        store.verify()
    return store


def _write_prices(a, run: Path, s1, s2) -> int:
    """The part of the command that runs with the lock on ``run`` held (see ``main``)."""
    from c_ept import runlog, store_core
    from c_ept.run_s1 import StrataRunning, _PeakMemory, require_no_strata          # here and not at module level (the S1 chain)
    prices = run / 'prices' / 'PRICES.json'
    if prices.exists():
        return _refuse(f'{prices} exists: a run folder is never edited, so PRICES.json is not overwritten. Give another --run, or none.')
    s1_sha, s2_sha = runlog.sha256_json(s1.manifest['files']), runlog.sha256_json(s2.manifest['files'])
    if (run / 'RUN.json').exists():                                  # a run that an E6 stop (or a crash) left without PRICES.json
        manifest = json.loads((run / 'RUN.json').read_text(encoding='utf-8'))
        if ((manifest.get('s1') or {}).get('files_sha256'), (manifest.get('s2') or {}).get('files_sha256')) != (s1_sha, s2_sha):
            return _refuse(f'{run / "RUN.json"} records other input stores than the S1 and S2 stores given: a run folder holds the prices '
                           'of one pair of stores. Give another --run, or none.')
        started = manifest.get('args') or {}
        if (started.get('B'), started.get('seed')) != (a.B, a.seed):
            return _refuse(f'{run / "RUN.json"} records a start with B {started.get("B")} and seed {started.get("seed")}, not {a.B} and '
                           f'{a.seed}: a run folder holds one set of prices. Give the same --B and --seed, or another --run.')
    else:
        manifest = runlog.write_manifest(run, {
            'args': {'B': a.B, 'seed': a.seed, 'roles': list(FIT_ROLES), 'probe': str(a.probe), 'verify': bool(a.verify)},
            's1': {'run': _run_name(a.s1), 'files_sha256': s1_sha}, 's2': {'run': _run_name(a.s2), 'files_sha256': s2_sha}})
    print(f'S3a run folder: {run}; B {a.B}, seed {a.seed}', flush=True)
    t0 = time.perf_counter()
    stage = {}
    try:
        with _PeakMemory() as mem:
            etable = event_table(s1, FIT_ROLES)
            stage['events_s'] = time.perf_counter() - t0
            print(f'  event table: {len(etable.t)} events, {int(etable.complete.sum())} complete, {stage["events_s"]:.1f} s', flush=True)
            require_no_strata()                                                    # E6 again between the stages
            wtable = window_table(s1, s2, FIT_ROLES)
            stage['windows_s'] = time.perf_counter() - t0 - stage['events_s']
            print(f'  window table: {len(wtable.y)} rows, {wtable.n_dropped_last} dropped for want of a next regular frame, '
                  f'{stage["windows_s"]:.1f} s', flush=True)
            require_no_strata()
            table = price_table(etable, wtable, a.B, a.seed, check=require_no_strata)
            stage['fits_s'] = time.perf_counter() - t0 - stage['events_s'] - stage['windows_s']
    except StrataRunning as e:
        return _refuse(f'{e} No PRICES.json was written; run the command again with --run {run} once the model server is stopped.')
    code = runlog.code_state()                                       # the code of this call (a continued run's RUN.json has the first)
    table['probe'] = probe_block(table, a.probe)
    table['provenance'] = {'s1_files_sha256': s1_sha, 's2_files_sha256': s2_sha, **code}
    prices.parent.mkdir(exist_ok=True)
    store_core.write_atomic(prices, json.dumps(table, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    digest = store_core.sha(prices)
    wall = time.perf_counter() - t0
    by_flow = [g for g in GROUPS if table[g]['used'] == 'flow']
    manifest = runlog.amend_manifest(run, {'s3a_result': {
        'complete': True, 'wall_s': round(wall, 1), 'stage_s': {k: round(v, 1) for k, v in stage.items()},
        'peak_private_gib': round(mem.peak / 2 ** 30, 3) if mem.peak else None, 'memory_samples': mem.samples, 'B': a.B, 'seed': a.seed,
        'n_matches': table['n_matches'], 'n_events': int(etable.complete.sum()), 'n_window_rows': table['n_window_rows'],
        'n_dropped_last': table['n_dropped_last'], 'used_flow': by_flow, 'used_immediate': [g for g in GROUPS if g not in by_flow],
        'prices_sha256': digest, 'files_sha256': {'prices/PRICES.json': digest}, 'code_state_at_end': code}})
    runlog.register(run, manifest, a.catalog_db)
    print(f'S3a done: {len(by_flow)} of {N_GROUPS} groups priced by the flow, {N_GROUPS - len(by_flow)} by the immediate value; '
          f'{wall:.1f} s, peak {manifest["s3a_result"]["peak_private_gib"]} GiB private; {prices}', flush=True)
    return 0


def main(argv=None) -> int:
    """명령줄: FIT 경기의 사건 표와 창 표를 만들고((FLOW-2), (FLOW-3)) price_table((FLOW-10))로 PRICES.json을 쓴다. 모델 서버가 도는 동안에는 시작하지 않으며(계획 전역 제약의 E6)
    이미 있는 PRICES.json은 덮어쓰지 않는다.
    """
    from c_ept import runlog
    from c_ept.run_s1 import StrataRunning, load_s1, require_no_strata              # here and not at module level (the S1 chain)
    ap = argparse.ArgumentParser(prog='python -m c_ept.flowprice',
                                 description='S3a: the flow prices and the immediate values of the FIT matches -> prices\\PRICES.json.')
    ap.add_argument('--s1', type=Path, required=True, help='the S1 run folder (or its s1 store folder); read only')
    ap.add_argument('--s2', type=Path, required=True, help='the S2 run folder (or its s2 store folder); read only')
    ap.add_argument('--run', type=Path, default=None,
                    help='the s3a run folder to write into (made if missing; it must hold no PRICES.json yet); default: a new '
                         '<runs root>\\<YYYYMMDD>_c_ept_s3a_<n>')
    ap.add_argument('--probe', type=Path, default=DEFAULT_PROBE,
                    help='the prototype JSON to set the immediate values beside (read only if it exists; ruling R4)')
    ap.add_argument('--B', type=int, default=BOOT_B, help='bootstrap replicates (matches resampled)')
    ap.add_argument('--seed', type=int, default=BOOT_SEED)
    ap.add_argument('--verify', action='store_true', help='check every file of both stores against its recorded SHA-256 first (slow)')
    ap.add_argument('--runs-root', type=Path, default=C.RUNS_ROOT)
    ap.add_argument('--catalog-db', type=Path, default=C.CATALOG_DB)
    a = ap.parse_args(argv)
    for flag, value, low in (('--B', a.B, 1), ('--seed', a.seed, 0)):
        if value < low:
            return _refuse(f'{flag} must be at least {low}, got {value}')
    try:
        require_no_strata()                                  # E6: before a store is opened and before the run folder is made
    except StrataRunning as e:
        return _refuse(f'{e} (The E6 guard of the S1 runner stops this command too; no run folder was made.)')
    try:
        s1 = load_s1(a.s1, verify=a.verify)
    except (FileNotFoundError, ValueError) as e:
        return _refuse(f'the S1 run cannot be used: {e}')
    try:
        s2 = _open_s2(a.s2, a.verify)
    except (FileNotFoundError, ValueError) as e:
        return _refuse(f'the S2 run cannot be used: {e}')
    run = a.run if a.run is not None else runlog.new_run('s3a', root=a.runs_root)
    try:
        lock = _acquire_lock(run)
    except RunLocked as e:
        return _refuse(str(e))
    try:
        return _write_prices(a, run, s1, s2)
    finally:
        _release_lock(lock)


if __name__ == '__main__':
    raise SystemExit(main())
