r"""Sweep of ``c_ept.state`` over every S1 match: how the BUILDING_KILL events match the landmarks, and the state at the regular frames.

What it measures, per match (``reader.read_match``):

* every BUILDING_KILL is matched to a landmark (``state.building_kills``): how many there are, how many fit no landmark, how many hit a
  building that was already down (a nexus turret or inhibitor that had come back is not one of these: it is counted apart as a re-kill
  after the respawn), the last split by the kind of building; how many have no position and how many lie more than ``FAR_U`` map units
  from the anchor of their landmark (the nexus turrets of a team are over 600 units apart, so the nearest-turret rule needs positions);
* the state at every regular frame and at the game end, from a ``StateIndex``: ``alive_respawn``, ``objective_state`` and
  ``landmark_state`` are checked against the output value rules (flags 0 or 1, seconds not negative, NaN only where LEGACY defines it:
  the three ``state.NAN_FEATURES`` and the seconds of the herald, grub and atakhan nests); a time at which a rule breaks is one
  ``policy_violations``. ``landmark_cells_not_finite`` splits into LEGACY's NaN (``..._legacy_nan``) and anything else (``..._other``);
  the NaN cells of the objective state are counted per feature. Float32 is an upstream output contract; this diagnostic does not inspect dtype;
* at three of those times (the third frame, the middle frame, the game end) the module functions run too and are compared with the
  index bit for bit (``index_vs_function_mismatch``);
* the standing turrets and inhibitors of the landmark state against LEGACY's counts per team and tier (``tower_count_mismatch``,
  ``inhibitor_count_mismatch``).

A match that raises is counted by the error's class and the code position (never its message, which could hold a key) and keeps the
counts it had reached; ``matches_with_exception`` is the number of those matches.

Input: the S1 run folder (or its ``s1`` store); the matches are those of its store with the role FIT, STOP or DEV (``common.QA_ROLES``,
decision B21: TEST, CAL, AUDIT, EXTERNAL and UNUSED matches are not read, and their keys are not decoded; no holdout patch). They
are read with ``reader.read_match`` and its default root. ``--workers N`` (default 6; 1 runs in this process), ``--limit N`` (the first N
matches, a smoke test) and ``--sample N --seed S`` (N matches drawn at random with that seed). Output: counts only, on stdout and with
``--out FILE`` as JSON; no key or identifier is printed or written, and nothing is written into the S1 run. Refuses to start while a
Strata server runs (E6), and checks it once, before the pool starts.

Reading the counts of a nexus turret: 'no kill on a destroyed landmark' means something there only together with
``building_kills_far_from_their_landmark`` and ``tower_count_mismatch`` (a second kill of the same nexus turret while its partner stands
is given to the partner as ``ok``; see ``state.building_kills``). The counts of the S1 run of 2026-10-20 were taken on 2026-10-21 at
5680022 and are in the S2 report; they are not repeated here, where they would go stale.

    python -B -m c_ept.checks.sweep_state D:\LOL_Data\runs\20261020_c_ept_s1_1 [--out counts.json] [--workers 6] [--limit N]


방법에서의 위치. 이 모듈은 건물 파괴 사건의 표지점 대응과 사건 유래 상태의 출력 계약을 진단한다.
표지점은 건물·둥지의 규칙상 위치이고, 상태는 기준 시각까지 보이는 생존·부활·오브젝트 값이다(Task 10).
입력은 S1 경기 목록과 판독 기록이고, 출력은 위반·미지정·대응 오차의 집계이다.
식 목록.
    (SWEEPS-1) 건물 사건과 대응 표지점의 거리 초과 집계.
    (SWEEPS-2) 표지점별 건물 상태와 팀·종류 집계의 불일치 수.
    (SWEEPS-3) 허용 NaN을 구분한 비유한 셀의 집계.
공통 가정. frame_ts는 ms이며 참조 시각 집합은 정기 프레임과 경기 끝의 합집합이다.
출력 시계의 남은 시간은 s, 위치는 지도 단위이며 배열 행·열은 0부터 시작한다.
상태 함수의 float32 계약은 S1–S2 Task 10 판정 R10-f에 있지만 이 진단의 정책 함수가 dtype 자체를 검사하지는 않는다.
LEGACY 허용 NaN은 오브젝트의 NAN_FEATURES와 지정 둥지의 남은 시간 칸에서만 예외 처리한다(S1–S2 Task 10 판정 R10-f).
거리와 건물 수 비교는 진단이며 기록의 모든 사건·물리 상태가 옳다는 보장은 아니다(S1–S2 Task 10 판정 R10-d).
함수·색인 대조 시각은 세 번째 또는 마지막 프레임·가운데 프레임·경기 끝의 집합이며, 프레임이 없으면 경기 끝만 남긴다(S1–S2 Task 10 판정 R10-d).
"""
from __future__ import annotations

import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from c_ept.data import reader
from c_ept.domain import state
from c_ept.checks import common
from c_ept.domain.units import event_xy

SCRIPT = 'c_ept.checks.sweep_state'
CHUNK = 100                  # matches per task of the pool
FAR_U = 300.0                # a kill farther than this (map units) from the anchor of its landmark is counted
COUNT_KEYS = ('matches', 'building_kill_events', 'building_kills_unmatched', 'building_kills_on_destroyed_landmark',
              'building_kills_without_position', 'building_kills_far_from_their_landmark',
              'building_kills_on_destroyed_lane_turret', 'building_kills_on_destroyed_inhibitor',
              'building_kills_on_destroyed_nexus_turret', 'nexus_rekills_after_respawn', 'inhibitor_rekills_after_respawn', 'state_refs',
              'function_refs', 'index_vs_function_mismatch', 'landmark_cells', 'landmark_cells_not_finite',
              'landmark_cells_not_finite_legacy_nan', 'landmark_cells_not_finite_other', 'objective_cells_not_finite_other',
              'tower_count_mismatch', 'inhibitor_count_mismatch', 'policy_violations')
NAN_PREFIX = 'objective_nan_cells:'                              # per feature, in the counter; moved to its own dict in the output
_NAN_COLUMNS = [state.OBJ_STATE_NAMES.index(n) for n in state.NAN_FEATURES]
_LM_NAN_OK = np.zeros((len(state.LANDMARKS), 2), dtype=bool)      # the cells where LEGACY defines NaN: the seconds of these nests
_LM_NAN_OK[[i for i, lm in enumerate(state.LANDMARKS) if lm['kind'] in ('herald', 'grubs', 'atakhan')], 1] = True
_TIER_NAMES = (('outer', 'OUTER_TURRET'), ('inner', 'INNER_TURRET'), ('base', 'BASE_TURRET'), ('nexus', 'NEXUS_TURRET'))
_TIERS = [([i for i, lm in enumerate(state.LANDMARKS[:28]) if (lm['team'], lm['tower']) == (team, tier)],
           state.OBJ_STATE_NAMES.index(f'{side}_towers_{short}'))
          for team, side in ((100, 'blue'), (200, 'red')) for short, tier in _TIER_NAMES]
_INHIBITORS = [([i for i, lm in enumerate(state.LANDMARKS[:28]) if lm['team'] == team and lm['kind'] == 'inhibitor'],
                state.OBJ_STATE_NAMES.index(f'{side}_inhibs_down')) for team, side in ((100, 'blue'), (200, 'red'))]


def breaks_a_rule(a, r, obj, lm) -> bool:
    """Whether the outputs of one reference time break a rule: flags not 0 or 1, seconds negative or not finite, a dead player without
    time left (or a living one with), NaN or infinity where LEGACY does not define NaN, infinity anywhere.

    S1–S2 Task 10 판정 R10-f의 플래그·남은 초·허용 NaN 조건을 검사하되 dtype 자체는 검사하지 않는다.
    일부 조건에서 True가 되면 Python의 or가 뒤 검사를 생략하므로 모든 칸의 개별 위반 목록을 만드는 함수는 아니다.
    """
    obj_nan, lm_nan = np.isnan(obj), np.isnan(lm) & ~_LM_NAN_OK
    obj_nan[_NAN_COLUMNS] = False
    return bool(not set(a.tolist()) <= {0.0, 1.0} or not np.isfinite(r).all() or (r < 0).any() or ((a == 0) != (r > 0)).any()
                or np.isinf(obj).any() or obj_nan.any() or np.isinf(lm).any() or lm_nan.any()
                or not set(lm[:, 0].tolist()) <= {0.0, 1.0} or (lm[~np.isnan(lm[:, 1]), 1] < 0).any())


def count_record(rec, c: Counter | None = None) -> Counter:
    """The counts of one match (``COUNT_KEYS``, and ``NAN_PREFIX + feature`` for the NaN cells of the objective state), added to ``c``
    if given: a counter that the caller keeps keeps what was counted before an exception.

    수식
    ----
    (SWEEPS-1) far_count += 1[(event_x − anchor_x)^2 + (event_y − anchor_y)^2 > FAR_U^2]
                           (event_x가 NaN이 아니고 대응 표지점이 있을 때).
    (SWEEPS-2) tower_mismatch += Σ_{tier_group} 1[Σ_{landmark_row ∈ tier_rows} landmark_alive ≠ objective_count];
               inhibitor_mismatch += Σ_{team_group} 1[Σ_{landmark_row ∈ inhibitor_rows} 1[landmark_alive = 0] ≠ objective_count].
    (SWEEPS-3) landmark_cells += size(landmark_values);
               landmark_bad += Σ 1[¬finite(landmark_values)]; landmark_legacy_nan += Σ 1[nan(landmark_values) ∧ allowed_nan];
               landmark_other += Σ 1[¬finite(landmark_values) ∧ ¬(nan(landmark_values) ∧ allowed_nan)];
               objective_other += Σ_{column ∉ allowed_columns} 1[¬finite(objective_values)]
                                  + Σ_{column ∈ allowed_columns} 1[inf(objective_values)].

    기호 | 뜻 | 단위 | 범위 | 코드 변수
    event_x, event_y | 사건 위치 | 지도 단위 | 원값; x만 결측 분기 | x, y
    anchor_x, anchor_y | 대응 표지점 좌표 | 지도 단위 | LANDMARKS의 값 | lm['x'], lm['y']
    FAR_U | 거리 초과 경계 | 지도 단위 | 모듈 상수 300 | FAR_U
    far_count | 경계 밖 건물 사건 수 | 사건 | 0 이상 | building_kills_far_from_their_landmark
    landmark_row, landmark_alive | 표지점 행·생존 열 값 | — | 0부터 시작; 계약상 0/1 | rows, lm[rows, 0]
    tier_group, tier_rows | 팀·포탑 단계 묶음과 행 목록 | — | _TIERS가 정한 묶음 | _TIERS의 rows
    team_group, inhibitor_rows | 팀별 억제기 묶음과 행 목록 | — | _INHIBITORS가 정한 묶음 | _INHIBITORS의 rows
    objective_count | 대응 오브젝트 상태의 집계 | 건물 | 상태 반환값 | obj[col]
    tower_mismatch, inhibitor_mismatch | 참조 시각별 불일치 묶음 수의 누적 | 묶음 | 0 이상 | c['tower_count_mismatch'], c['inhibitor_count_mismatch']
    landmark_values, objective_values | 표지점·오브젝트 상태 전체 배열 | 열별 0/1·s·개 등 | state 반환 계약 | lm, obj
    allowed_nan, allowed_columns, column | 표지점 허용 NaN 마스크·오브젝트 허용 열·열 번호 | — | bool 마스크; 0부터 시작 | _LM_NAN_OK, _NAN_COLUMNS, col
    landmark_cells, landmark_bad, landmark_legacy_nan, landmark_other, objective_other | 전체 표지점 셀·비유한·허용 NaN·나머지 비유한·오브젝트 비허용 수 | 셀 | 0 이상 | c['landmark_cells'], c['landmark_cells_not_finite'], c['landmark_cells_not_finite_legacy_nan'], c['landmark_cells_not_finite_other'], c['objective_cells_not_finite_other']
    finite, nan, inf, size | 유한·NaN·무한 검사와 원소 수 | — | 원소별 bool; 원소 수 | np.isfinite, np.isnan, np.isinf, lm.size
    1[·] | 조건을 만족하는 표시 | — | 0 또는 1 | 비교식

    해석
    ----
    식 (SWEEPS-1)은 경계와 같은 거리는 제외하며, x가 NaN이면 거리 대신 위치 없음으로 분류한다(Task 10 판정 R10-d).
    나머지 비유한 좌표를 이 거리 식이 별도로 거부하는 것은 아니므로 물리적 거리의 완전한 유효성 검사는 아니다(S1–S2 Task 10 판정 R10-d).
    식 (SWEEPS-2)는 서 있는 포탑과 파괴된 억제기를 각각 대응 집계와 비교한다(Task 10의 표지점 계약).
    지정 참조 시각의 함수·색인 대조는 허용 NaN을 같은 값으로 취급하고, 전체 관측 시각의 완전한 동등성 증명은 아니다(S1–S2 Task 10 판정 R10-d).
    식 (SWEEPS-3)은 허용 NaN과 나머지 비유한 값을 분리하며, 전달된 Counter는 예외 이전 가산을 남긴다(S1–S2 Task 10 판정 R10-d·R10-f의 진단 계약).

    근거
    ----
    S1–S2 Task 10 판정 R10-d·R10-f(상태 진단·범위), S1–S2 계획 Task 10(표지점 대응).
    """
    c = Counter({key: 0 for key in COUNT_KEYS}) if c is None else c
    c['matches'] += 1
    kills = state.building_kills(rec)
    c['building_kill_events'] += len(kills)
    for e, k in zip([e for e in rec.events if e.get('type') == 'BUILDING_KILL'], kills, strict=True):
        lm = state.LANDMARKS[k.landmark] if k.landmark >= 0 else None
        x, y = event_xy(e)
        # (SWEEPS-1)은 위치 없음 분기 뒤에 엄격한 거리 초과만 센다(S1–S2 Task 10 판정 R10-d).
        if np.isnan(x):
            c['building_kills_without_position'] += 1
        elif lm is not None and (x - lm['x']) ** 2 + (y - lm['y']) ** 2 > FAR_U ** 2:
            c['building_kills_far_from_their_landmark'] += 1
        what = 'lane_turret'
        if lm and lm['kind'] == 'inhibitor':
            what = 'inhibitor'
        elif lm and lm['tower'] == 'NEXUS_TURRET':
            what = 'nexus_turret'
        if k.status == 'unmatched':
            c['building_kills_unmatched'] += 1
        elif k.status == 'already_down':
            c['building_kills_on_destroyed_landmark'] += 1
            c[f'building_kills_on_destroyed_{what}'] += 1
        elif k.status == 'after_respawn':
            c['inhibitor_rekills_after_respawn' if what == 'inhibitor' else 'nexus_rekills_after_respawn'] += 1
    index = state.StateIndex(rec)
    n = len(rec.frame_ts)
    refs = sorted({int(t) for t in rec.frame_ts} | {rec.T_ms})
    compared = {int(rec.frame_ts[min(2, n - 1)]) if n else rec.T_ms, int(rec.frame_ts[n // 2]) if n else rec.T_ms, rec.T_ms}
    for ref in refs:
        a, r = index.alive_respawn(ref)
        obj = index.objective_state(ref)
        lm = index.landmark_state(ref, obj)
        c['state_refs'] += 1
        c['policy_violations'] += breaks_a_rule(a, r, obj, lm)
        if ref in compared:
            c['function_refs'] += 1
            ra, rr = state.alive_respawn(rec, ref)
            robj = state.objective_state(rec, ref)
            rlm = state.landmark_state(rec, ref, robj)
            equal = (np.array_equal(ra, a) and np.array_equal(rr, r) and np.array_equal(robj, obj, equal_nan=True)
                     and np.array_equal(rlm, lm, equal_nan=True))
            c['index_vs_function_mismatch'] += not equal
        # (SWEEPS-3)은 허용 NaN을 나머지 비유한 셀과 분리한다(S1–S2 Task 10 판정 R10-f).
        not_finite, legacy_nan = ~np.isfinite(lm), np.isnan(lm) & _LM_NAN_OK
        c['landmark_cells'] += lm.size
        c['landmark_cells_not_finite'] += int(not_finite.sum())
        c['landmark_cells_not_finite_legacy_nan'] += int(legacy_nan.sum())
        c['landmark_cells_not_finite_other'] += int((not_finite & ~legacy_nan).sum())
        for col in _NAN_COLUMNS:
            c[NAN_PREFIX + state.OBJ_STATE_NAMES[col]] += int(np.isnan(obj[col]))
        other = ~np.isfinite(obj)
        other[_NAN_COLUMNS] = np.isinf(obj[_NAN_COLUMNS])
        c['objective_cells_not_finite_other'] += int(other.sum())
        # (SWEEPS-2)는 서 있는 포탑과 파괴된 억제기를 대응 집계에 맞춘다(S1–S2 계획 Task 10).
        c['tower_count_mismatch'] += int(sum(lm[rows, 0].sum() != obj[col] for rows, col in _TIERS))
        c['inhibitor_count_mismatch'] += int(sum((lm[rows, 0] == 0).sum() != obj[col] for rows, col in _INHIBITORS))
    return c


def work(pairs):
    """``(counts, errors)`` of a list of ``(patch, key)``; an exception is counted by class and code position, never by message.

    S1–S2 Task 10 판정 R10-d의 경기별 진단을 누적하며, 예외 경기의 이미 센 값도 유지한다.
    """
    total, errors = Counter(), Counter()
    for patch, key in pairs:
        part = Counter()
        try:
            count_record(reader.read_match(key, patch), part)
        except Exception as e:                                # the sweep goes on and reports what failed
            where = traceback.extract_tb(e.__traceback__)[-1]
            errors[f'{type(e).__name__} at line {where.lineno} in {where.name} of {Path(where.filename).name} (patch {patch})'] += 1
        total.update(part)
    return total, errors


def main(argv=None) -> int:
    """S1–S2 Task 10 판정 R10-d의 전수 또는 표본 진단 경로를 연결하며, 선택된 목록의 처리 결과를 합친다.

    결정 B21: 목록은 QA 역할(FIT·STOP·DEV)의 경기만이다. --limit(앞부분)과 --sample(무작위 추출)도 이 목록 안에서만 고르므로
    TEST·CAL·AUDIT·EXTERNAL·UNUSED 경기의 원시 기록은 어떤 옵션으로도 열리지 않는다. 이전에는 저장소의 모든 역할이 대상이었다.
    """
    ap = common.arguments(__doc__, limit=True)
    ap.add_argument('--workers', type=int, default=6, help='worker processes (1: everything in this process)')
    ap.add_argument('--sample', type=int, help='read N matches drawn at random (see --seed) from the FIT, STOP and DEV matches instead of all of them')
    ap.add_argument('--seed', type=int, default=0, help='seed of the --sample draw (default 0)')
    args = common.parse(ap, argv)
    from c_ept.workflows import run_s1                                  # inside the function: a view builder never imports the S1 runner chain
    try:
        run_s1.require_no_strata()                            # E6: before any raw file is read and before the pool starts
    except run_s1.StrataRunning as e:
        return common.refuse(str(e))
    try:
        matches = common.s1_matches(args.run, roles=common.QA_ROLES)      # B21: FIT, STOP and DEV only; --limit and --sample draw from these
    except FileNotFoundError as e:
        return common.refuse(str(e))
    if args.sample is not None:
        pick = np.random.default_rng(args.seed).choice(len(matches), size=min(args.sample, len(matches)), replace=False)
        matches = [matches[int(i)] for i in sorted(pick.tolist())]
    matches = matches[:args.limit]
    chunks = [[(patch, key) for key, patch, _ in matches[i:i + CHUNK]] for i in range(0, len(matches), CHUNK)]
    total, errors = Counter({key: 0 for key in COUNT_KEYS}), Counter()
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(work, chunks))
    else:
        results = [work(chunk) for chunk in chunks]
    for part, bad in results:
        total.update(part)
        errors.update(bad)
    nan_cells = {k[len(NAN_PREFIX):]: total.pop(k) for k in [k for k in total if k.startswith(NAN_PREFIX)]}
    total['matches_with_exception'] = sum(errors.values())
    return common.finish(args, SCRIPT, f'sweep of the state over {len(matches)} S1 matches', total, errors=dict(errors),
                         objective_nan_cells_by_feature=dict(sorted(nan_cells.items())))


if __name__ == '__main__':
    raise SystemExit(main())
