r"""Level clock against the frame level on the 20 golden matches (the measurement behind ``test_level_at_matches_frame_level``).

What it measures: at every regular frame of the 20 matches of the S1 golden lock, whether ``traits.level_at(rec, <the frame's ms>)`` (the
highest LEVEL_UP level of each player at or before that ms) equals the level in the frame. It counts the matches, the regular frames, the
player-frames (10 per frame), the LEVEL_UP events, those that fall on exactly a frame's ms, and the player-frames that differ, split by
cause (a frame that lags the clock by one level while a LEVEL_UP of that player falls on its ms, or another cause).

Input: the S1 run folder (or its ``s1`` store); the lock is read from ``<run>\golden\s1\matches`` (``--golden-dir`` names another golden
folder). The 20 raw matches are read with ``reader.read_match`` and its default root. Output: counts only, on stdout and with ``--out FILE``
as JSON; no match key or identifier is printed or written, and nothing is written into the S1 run. Refuses to start while a Strata server
runs (E6).

Counts of 2026-10-20, run at commit 39195c7 (as a scratch script, then moved here; run again from here with the same result): 20 matches,
582 regular frames, 5,820 player-frames, 4 LEVEL_UPs on a frame's ms, 0 differences. ``test_level_at_matches_frame_level`` pins them.

    python -B -m c_ept.checks.measure_levels D:\LOL_Data\runs\20261020_c_ept_s1_1 [--out counts.json]


방법에서의 위치. 이 모듈은 기준값 잠금 목록의 레벨 시계와 프레임 레벨을 공통 비교기로 대조한다.
기준값 잠금은 S1 추출 배열의 비교 기준이며, 이 진단은 그 목록의 경기 기록을 다시 읽는다(Task 9 fix1 M5).
입력은 S1 실행 경로와 선택적 잠금 경로이고, 출력은 공통 레벨 집계이다.
식 목록. 자체 계산식은 없으며 수치 대조는 common.compare_levels의 (CHK-1)이 소유한다.
공통 가정. 목록 선택과 판독은 공통 모듈에 위임하고, 실제 레벨 정의는 traits.LevelIndex가 소유한다.
실행 시 E6를 먼저 검사하며 이 모듈을 불러오는 것만으로 진단하지 않는다(지침 E6; Task 9 fix1 M5).
"""
from __future__ import annotations

from c_ept.workflows import run_s1
from c_ept.checks import common

SCRIPT = 'c_ept.checks.measure_levels'


def main(argv=None) -> int:
    """S1–S2 판정 R37의 기준값 목록을 공통 레벨 비교기로 집계하며, 판독 실패는 이 경로에서 숨기지 않는다."""
    args = common.parse(common.arguments(__doc__, golden=True), argv)
    try:
        run_s1.require_no_strata()                          # E6: before any raw file is read
    except run_s1.StrataRunning as e:
        return common.refuse(str(e))
    try:
        matches = common.golden_matches(args.run, args.golden_dir)
    except FileNotFoundError as e:
        return common.refuse(str(e))
    total = common.new_level_counts()
    for rec in common.records(matches):
        total.update(common.compare_levels(rec))
    return common.finish(args, SCRIPT, 'level clock against the frame level, golden matches', total)


if __name__ == '__main__':
    raise SystemExit(main())
