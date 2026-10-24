r"""Level clock against the frame level on a sample of 200 S1 matches, and how close LEVEL_UPs lie to a frame.

What it measures: the same comparison as ``measure_levels`` (``traits.level_at`` against the frame level at every regular frame, counted
per player-frame and split by cause) on a sample of the S1 matches, and, for every LEVEL_UP, how long before the next regular frame it
happens (on exactly the frame's ms, within 10 ms, 100 ms, 1 s). The sample is the first 40 FIT matches and the first 20 matches of every
other role per patch, by sorted key: 200 matches of the four patches 15.18-15.21.

Input: the S1 run folder (or its ``s1`` store), for the list of its matches; the raw matches are read with ``reader.read_match`` and its
default root. ``--limit N`` reads only the first N of the sample. Output: counts only, on stdout and with ``--out FILE`` as JSON; no match
key or identifier is printed or written, and nothing is written into the S1 run. Refuses to start while a Strata server runs (E6).

Counts of 2026-10-20, run at commit 39195c7 (as a scratch script, then moved here; not run again from here on the whole sample): 200
matches, 2 of 51,230 player-frames differ, both a LEVEL_UP at exactly the frame's ms that the frame does not show yet. It was the first
look; ``sweep_s1`` gives the counts over every S1 match.

    python -B -m c_ept.checks.measure_levels_wide D:\LOL_Data\runs\20261020_c_ept_s1_1 [--out counts.json] [--limit N]


방법에서의 위치. 이 모듈은 패치·역할별 앞부분 표본의 레벨 차이와 상승 사건의 프레임 근접성을 함께 기록한다.
근접성은 상승 사건에서 그 시각 이상인 첫 정기 프레임까지의 대기 시간이다(Task 9 fix1 M5).
입력은 S1 목록의 표본과 경기 기록이고, 출력은 레벨 대조·근접성의 누적 집계이다.
식 목록.
    (LEVELW-1) 다음 프레임의 대기 ms와 누적 근접성 수.
    공통 레벨 차이 식은 common.compare_levels의 (CHK-1)을 재사용한다.
공통 가정. frame_ts는 정렬되어 있다고 전제하며 이 함수에서 재정렬·검사하지 않는다.
표본은 정렬 목록 앞부분이므로 확률 표본이라는 보장은 없다(common.sample_matches).
모든 시각 차이는 ms이고, 프레임 행은 0부터 시작한다.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from c_ept import run_s1
from c_ept.checks import common

SCRIPT = 'c_ept.checks.measure_levels_wide'
PROXIMITY_KEYS = ('level_ups_before_a_frame', 'level_ups_0_ms_before_a_frame', 'level_ups_within_10_ms_before_a_frame',
                  'level_ups_within_100_ms_before_a_frame', 'level_ups_within_1000_ms_before_a_frame')


def proximity(rec) -> Counter:
    """How close the LEVEL_UPs of a match lie before the next regular frame, cumulative (0 ms, within 10 / 100 / 1000 ms). A LEVEL_UP after
    the last regular frame has no next frame and is not counted.

    수식
    ----
    (LEVELW-1) next_row = int(searchsorted(frame_ts, event_ms, side='left'));
               wait_ms = int(frame_ts[next_row]) − event_ms (next_row < len(frame_ts));
               eligible = Σ_{up_entry} 1[next_row < len(frame_ts)];
               zero_count = Σ_{유효 up_entry} 1[wait_ms ≤ 0];
               near_count_{bound_ms} = Σ_{유효 up_entry} 1[wait_ms ≤ bound_ms], bound_ms ∈ {10, 100, 1000}.

    기호 | 뜻 | 단위 | 범위 | 코드 변수
    frame_ts | 정기 프레임 시각 배열 | ms | 정렬을 전제 | rec.frame_ts
    event_ms, up_entry | 상승 시각·상승 사건 | ms; — | LEVEL_UP만 선택 | e['timestamp'], e
    next_row | 그 시각 이상인 첫 프레임의 0부터 시작하는 행 | — | 0–len(frame_ts) | j
    wait_ms | 선택 프레임까지의 시간 차 | ms | 정렬·정상 시각이면 0 이상 | wait
    bound_ms | 양의 누적 근접성 경계 | ms | 10, 100, 1000 | bound
    eligible, zero_count, near_count | 뒤 프레임이 있는 사건 수·0 이하 수·양의 경계별 수 | 사건 | 0 이상 | c['level_ups_before_a_frame'], c['level_ups_0_ms_before_a_frame']; c[key], key는 PROXIMITY_KEYS[2:]의 각 문자열
    1[·], searchsorted | 조건 표시·정렬 배열의 왼쪽 삽입 위치 | — | 0/1; 0부터 길이 | 조건식, np.searchsorted
    int | Python 정수 변환 | 입력 단위 | 프레임 시각은 정수 ms 전제 | int

    해석
    ----
    식 (LEVELW-1)은 프레임과 같은 ms의 사건도 그 프레임에 배정한다(Task 9의 동시각 레벨 진단).
    0 경계는 실제로 0 이하를 세며, 대기 시간이 비음수인 정상 입력에서만 동시각 수와 같다(Task 9의 정기 프레임 전제).
    경계별 집계는 누적이므로 한 사건이 여러 경계에 포함될 수 있다(Task 9 fix1 M5의 근접성 기록).
    마지막 프레임 뒤의 사건과 빈 프레임 배열은 대기 시간을 만들지 않고 모든 근접성 수에서 제외한다(Task 9 진단 범위).
    정렬되지 않은 입력에서도 비음수 대기 시간을 보장하는 검사는 없다(Task 9의 정기 프레임 전제).

    근거
    ----
    S1–S2 계획 Task 9(동시각 레벨 대조), S1–S2 판정 R37 및 Task 9 fix1 M5(표본 진단).
    """
    c = Counter({key: 0 for key in PROXIMITY_KEYS})
    for e in rec.events:
        if e['type'] != 'LEVEL_UP':
            continue
        # (LEVELW-1)의 왼쪽 삽입은 같은 ms의 프레임을 포함한다(S1–S2 계획 Task 9).
        j = int(np.searchsorted(rec.frame_ts, e['timestamp'], side='left'))
        if j < len(rec.frame_ts):
            wait = int(rec.frame_ts[j]) - e['timestamp']
            c['level_ups_before_a_frame'] += 1
            for key, bound in zip(PROXIMITY_KEYS[1:], (0, 10, 100, 1000)):
                c[key] += wait <= bound
    return c


def main(argv=None) -> int:
    """S1–S2 판정 R37의 표본에서 공통 레벨 비교와 다음 프레임 근접성 집계를 합친다."""
    args = common.parse(common.arguments(__doc__, limit=True), argv)
    try:
        run_s1.require_no_strata()                          # E6: before any raw file is read
    except run_s1.StrataRunning as e:
        return common.refuse(str(e))
    try:
        sample = common.sample_matches(args.run)
    except FileNotFoundError as e:
        return common.refuse(str(e))
    total = common.new_level_counts()
    total.update(Counter({key: 0 for key in PROXIMITY_KEYS}))
    for rec in common.records(sample, args.limit):
        total.update(common.compare_levels(rec))
        total.update(proximity(rec))
    return common.finish(args, SCRIPT, 'level clock against the frame level, sample of S1 matches', total)


if __name__ == '__main__':
    raise SystemExit(main())
