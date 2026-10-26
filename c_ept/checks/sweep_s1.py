r"""Sweep of ``c_ept.traits`` over every S1 match: the level clock against the frames, the champion traits and the item vectors.

What it measures, per match: ``reader.read_match``; ``traits.LevelIndex`` (its checks of the LEVEL_UP events) and ``level_at`` against the
frame level at every regular frame, as ``measure_levels`` does; ``champion_traits`` at up to three frames (the third regular frame, the
middle one and the last; they coincide in a short match, so the set collapses; counting champions that are not in the table);
``ItemIndex`` built and its vectors taken at three times (at the third regular frame, at 600 s, at the game's end; counting players
with an unknown item). A match that raises is counted by the error's class and the code position (never its message, which
could hold a key).

Input: the S1 run folder (or its ``s1`` store); the matches are those of its store with the role FIT, STOP or DEV (``common.QA_ROLES``,
decision B21: TEST, CAL, AUDIT, EXTERNAL and UNUSED matches are not read; the counts of 2026-10-20 below were taken over all 28,701 matches
of the S1 run, before this restriction). They are read with ``reader.read_match`` and its default root. ``--workers N`` (default 6; 1 runs in this process) and ``--limit N`` (the first N matches, a
smoke test). Output: counts only, on stdout and with ``--out FILE`` as JSON; no key or identifier is printed or written, and nothing is
written into the S1 run. Refuses to start while a Strata server runs (E6), and checks it once, before the pool starts.

Counts of 2026-10-20, run at commit 39195c7 (as a scratch script, then moved here; the whole sweep was not run again from here): 28,701
matches, 0 errors; 757,315 regular frames, 7,573,150 player-frames, 3,647,957 LEVEL_UPs of which 1,888 on a frame's ms; 150 player-frames
differ, all 150 ``lag_by_one_at_frame_ms``, none ``other``; 855,620 trait rows with 0 unknown champions; 861,030 item vectors, all finite
(the replay raises otherwise), 0 players with an unknown item; 221 matches have fewer than 3 regular frames.

    python -B -m c_ept.checks.sweep_s1 D:\LOL_Data\runs\20261020_c_ept_s1_1 [--out counts.json] [--workers 6] [--limit N]


방법에서의 위치. 이 모듈은 S1 목록의 레벨 대조와 일부 시각의 챔피언·아이템 미지정 항목을 집계한다.
특성은 레벨별 챔피언 능력치이고 아이템 벡터는 기준 시각까지 재생한 소지품의 수치 표현이다(Task 9).
입력은 S1 경기 목록이고, 출력은 집계와 예외 종류·발생 위치별 수이다.
식 목록.
    (SWEEP1-1) 선택 프레임·시각의 미지정 챔피언·아이템 집계.
    전체 프레임의 레벨 차이는 공통 식 (CHK-1)을 재사용한다.
공통 가정. 한 경기에는 정기 프레임이 있다고 전제하며, 없는 경우 이 함수의 배열 접근이 실패할 수 있다.
프레임 행은 0부터 시작하고 기준 시각은 ms이다.
경기별 오류는 호출자가 집계하고 다음 경기로 진행하며, 성공하지 않은 경기의 완결 집계를 보장하지 않는다.
출처: S1–S2 판정 R37 및 Task 9 fix1 M5.
"""
from __future__ import annotations

import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from c_ept.data import reader
from c_ept.workflows import run_s1
from c_ept.preprocessing import traits
from c_ept.checks import common

SCRIPT = 'c_ept.checks.sweep_s1'
SWEEP_KEYS = ('matches_with_fewer_than_3_frames', 'trait_rows', 'unknown_champion_rows', 'item_vectors', 'unknown_item_count_rows')
CHUNK = 100                  # matches per task of the pool
_UNKNOWN = traits.TRAIT_NAMES.index('champ_unknown')


def one_match(patch: str, key: str) -> Counter:
    """The counts of one match (``LEVEL_KEYS`` and ``SWEEP_KEYS``).

    수식
    ----
    (SWEEP1-1) third_row = min(2, frame_count − 1);
               frame_set = sorted{third_row, frame_count // 2, frame_count − 1};
               time_list = [int(frame_ts[third_row]), 600000, T_ms];
               trait_rows = 10·|frame_set|, unknown_champion_rows = Σ_{frame_row ∈ frame_set} int(Σ_player trait_unknown);
               item_vectors = 10·len(time_list), unknown_item_rows = Σ_{ref_ms ∈ time_list} Σ_player 1[item_unknown > 0].

    기호 | 뜻 | 단위 | 범위 | 코드 변수
    frame_count, third_row | 프레임 수·세 번째 또는 마지막 행 | 프레임; — | 양의 수 전제; 0부터 시작 | n, third
    frame_ts, player | 프레임 시각 배열·선수 행 | ms; — | 기록 순서; 0–9 | rec.frame_ts, 선수 축
    frame_set, frame_row | 중복 제거한 표본 행·그 원소 | — | 정렬된 행 집합 | sorted 집합, k
    time_list, ref_ms | 아이템 기준 시각의 목록·그 원소 | ms | 목록의 중복 유지 | for t의 세 시각
    T_ms | 경기 끝 시각 | ms | 기록 값 | rec.T_ms
    trait_unknown | 특성의 미지정 챔피언 열 | — | traits 반환값 | champion_traits(...)[:, _UNKNOWN]
    item_unknown | 선수별 미지정 아이템 수 | 개 | ItemIndex 반환값 | items.vectors(t)[:, unknown_items]
    trait_rows, unknown_champion_rows | 검사한 특성 행·미지정 열 합 | 선수 행 | 0 이상 전제 | c['trait_rows'], c['unknown_champion_rows']
    item_vectors, unknown_item_rows | 검사한 아이템 벡터·미지정 양수 행 | 선수 행 | 0 이상 | c의 item_vectors, unknown_item_count_rows
    1[·], int, // | 표시·정수 변환·정수 나눗셈 | — | Python/NumPy 규칙 | 조건식, int, //

    해석
    ----
    식 (SWEEP1-1)은 전체 경기 레벨 대조와 별도로 대표 위치의 특성·아이템 미지정 항목을 센다(Task 9 fix1 M5).
    프레임 표본은 집합으로 중복을 제거하지만 아이템 시각 목록은 같은 시각이 겹쳐도 각각 센다(Task 9 진단 범위).
    챔피언 값은 열 합을 int로 바꾸고 아이템은 양수인 선수 행을 세므로 같은 종류의 합산이 아니다(Task 9의 특성·아이템 진단).
    정기 프레임이 없는 입력을 이 함수가 사전에 거부하거나 빈 집계로 바꾸지는 않는다(Task 9의 판독 기록 전제).

    근거
    ----
    S1–S2 계획 Task 9(특성·아이템 대조), S1–S2 판정 R37 및 Task 9 fix1 M5(진단 보존).
    """
    rec = reader.read_match(key, patch)
    c = common.compare_levels(rec)
    n = len(rec.frame_ts)
    third = min(2, n - 1)                                     # a few matches have fewer than 3 regular frames
    c['matches_with_fewer_than_3_frames'] = int(n < 3)
    # (SWEEP1-1)의 프레임 표본은 중복을 제거해 같은 행을 다시 세지 않는다(Task 9 fix1 M5).
    for k in sorted({third, n // 2, n - 1}):
        c['trait_rows'] += 10
        c['unknown_champion_rows'] += int(traits.champion_traits(rec, rec.frame_num[k, :, 2])[:, _UNKNOWN].sum())
    items, unknown_items = traits.ItemIndex(rec), traits.item_vector_names().index('unknown_item_count')
    # (SWEEP1-1)의 아이템 시각은 목록이므로 겹친 시각도 각각 센다(Task 9 fix1 M5).
    for t in (int(rec.frame_ts[third]), 600_000, rec.T_ms):
        c['item_vectors'] += 10
        c['unknown_item_count_rows'] += int((items.vectors(t)[:, unknown_items] > 0).sum())
    return c


def work(pairs):
    """``(counts, errors)`` of a list of ``(patch, key)``; an exception is counted by class and code position, never by message.

    S1–S2 판정 R37의 경기별 진단 오류를 메시지 대신 종류·코드 위치로 집계하고 다음 경기를 처리한다.
    """
    total, errors = Counter(), Counter()
    for patch, key in pairs:
        try:
            total.update(one_match(patch, key))
        except Exception as e:                                # the sweep goes on and reports what failed
            where = traceback.extract_tb(e.__traceback__)[-1]
            errors[f'{type(e).__name__} at line {where.lineno} in {where.name} of {Path(where.filename).name} (patch {patch})'] += 1
    return total, errors


def main(argv=None) -> int:
    """S1–S2 판정 R37의 목록을 묶음으로 나누어 집계하며, E6 시작 검사는 작업자 풀을 만들기 전에 수행한다.

    결정 B21: 목록은 QA 역할(FIT·STOP·DEV)의 경기만이고 --limit은 그 목록의 앞부분이다. 다른 역할의 원시 기록은 열지 않는다.
    """
    ap = common.arguments(__doc__, limit=True)
    ap.add_argument('--workers', type=int, default=6, help='worker processes (1: everything in this process)')
    args = common.parse(ap, argv)
    try:
        run_s1.require_no_strata()                            # E6: before any raw file is read and before the pool starts
    except run_s1.StrataRunning as e:
        return common.refuse(str(e))
    try:
        matches = common.s1_matches(args.run, roles=common.QA_ROLES)[:args.limit]       # B21: FIT, STOP and DEV only
    except FileNotFoundError as e:
        return common.refuse(str(e))
    chunks = [[(patch, key) for key, patch, _ in matches[i:i + CHUNK]] for i in range(0, len(matches), CHUNK)]
    total, errors = common.new_level_counts(), Counter()
    total.update(Counter({key: 0 for key in SWEEP_KEYS}))
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(work, chunks))
    else:
        results = [work(chunk) for chunk in chunks]
    for part, bad in results:
        total.update(part)
        errors.update(bad)
    return common.finish(args, SCRIPT, f'sweep of the traits over {len(matches)} S1 matches', total, errors=dict(errors))


if __name__ == '__main__':
    raise SystemExit(main())
