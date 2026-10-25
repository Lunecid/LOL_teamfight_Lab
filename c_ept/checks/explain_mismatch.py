r"""The cause of every difference between the level clock and the frame level, on the sample of ``measure_levels_wide``.

What it measures: for every player-frame of the sample (the first 40 FIT matches and the first 20 matches of every other role per patch of
the S1 store, 200 matches) where ``traits.level_at`` differs from the frame level, it gives the cause (``lag_by_one_at_frame_ms``: the frame
shows one level less than the clock and a LEVEL_UP of that player falls on exactly the frame's ms; or ``other``) together with the
LEVEL_UPs of that player within 5 s of the frame and the player's frame levels around it. The numbers are frame indices, times in ms,
participant ids (1-10), levels, and the position of the match in the sample, its patch and its role; never a key or an identifier.

Input: the S1 run folder (or its ``s1`` store), for the list of its matches; the raw matches are read with ``reader.read_match`` and its
default root. ``--limit N`` reads only the first N of the sample. Output: the counts and the first 40 differing player-frames on stdout, and
with ``--out FILE`` the counts and all of them as JSON; nothing is written into the S1 run. Refuses to start while a Strata server runs
(E6).

Counts of 2026-10-20, run at commit 39195c7 (as a scratch script, then moved here): the 2 differences of the 200-match sample were both
``lag_by_one_at_frame_ms`` (a LEVEL_UP at dt = 0 ms); in those two cells, and only there, the next frame shows the new level (read from
their ``frame_levels_around``). The cause label itself tests the LEVEL_UP on the frame's ms and the lag of one level, not the next
frame, so it says nothing about the next frame of any other cell.

    python -B -m c_ept.checks.explain_mismatch D:\LOL_Data\runs\20261020_c_ept_s1_1 [--out counts.json] [--limit N]


방법에서의 위치. 이 모듈은 레벨 불일치 셀 주변의 사건·프레임을 수치로 묶어 원인 판단의 근거를 제공한다.
불일치 셀은 common.compare_levels가 찾은 프레임·참가자 한 쌍이다(Task 9 fix1 M4·M5).
입력은 S1에서 고른 경기와 불일치 셀이며, 출력은 기준 프레임에 대한 상대 사건 시각과 주변 레벨이다.
식 목록.
    (MISM-1) 프레임 기준 사건의 상대 ms와 주변 프레임 절단.
공통 가정. frame은 0부터 시작하는 프레임 행이고 pid는 1부터 시작하는 참가자 ID이다.
시각은 ms이며, 양의 상대값은 프레임 뒤의 사건을 뜻한다.
이 함수는 원인을 단정하거나 허용 셀을 잠그지 않고, 판단에 필요한 수치 창을 반환한다.
식별자는 출력하지 않는 호출자 경로를 전제하며 전달된 cell 자체를 정화하는 함수는 아니다.
"""
from __future__ import annotations

from c_ept.workflows import run_s1
from c_ept.checks import common

SCRIPT = 'c_ept.checks.explain_mismatch'
SHOWN = 40            # player-frames printed (all of them are in the JSON)
WINDOW_MS = 5_000     # LEVEL_UPs of the player this close to the frame are listed


def explain(rec, cell: dict) -> dict:
    """``cell`` (from ``common.compare_levels``) with the LEVEL_UPs of its player within ``WINDOW_MS`` of the frame, as [dt ms, level]
    (dt = the LEVEL_UP's ms minus the frame's), and the player's frame levels of the frames around it.

    수식
    ----
    (MISM-1) relative_ms = event_ms − frame_ms;
             selected_ups = sorted([(relative_ms, event_level): 같은 참가자, |relative_ms| ≤ WINDOW_MS]) (중복 보존);
             frame_window = [max(0, frame_row − 2), frame_row + 3) ∩ [0, len(frame_ts)).

    기호 | 뜻 | 단위 | 범위 | 코드 변수
    event_ms, frame_ms | 상승 사건·대상 프레임 시각 | ms | 기록 값 | e['timestamp'], ms
    frame_ts | 정기 프레임의 시각 배열 | ms | 기록 순서 | rec.frame_ts
    relative_ms | 프레임 기준 사건 시각 | ms | 선택되면 −WINDOW_MS 이상 WINDOW_MS 이하 | ups의 첫 성분
    event_level | 상승 후 레벨 | 레벨 | 기록 값, 여기서 검사하지 않음 | e['level']
    selected_ups | 중복을 보존한 상대 시각·레벨의 정렬 목록 | ms·레벨 쌍 | 빈 목록 가능 | ups
    WINDOW_MS | 주변 사건의 반폭 | ms | 모듈 상수 5000 | WINDOW_MS
    frame_row, frame_window | 대상 행·주변 프레임 행 구간 | — | 0부터 시작, 끝 제외 | k, around의 슬라이스
    participant_id | 대상 선수의 ID | — | 1–10을 전제 | pid; 배열 행은 pid − 1

    해석
    ----
    식 (MISM-1)은 같은 시각 사건의 상대값을 0으로 남기고 양끝을 포함한 사건 창을 만든다(Task 9 fix1 M4·M5).
    프레임 창의 오른쪽 끝은 제외되며 기록 끝에서는 실제 배열 길이에서 잘린다(Task 9 fix1 M5의 주변 기록).
    프레임 레벨은 판독기의 float64 원본에서 int로 바꾸어 출력하고, 사건이 없으면 빈 목록을 남긴다(Task 9 진단 출력).
    입력 셀의 프레임·참가자 유효성은 호출자가 만든 셀이라는 전제이며 여기서 별도 검사하지 않는다(common.compare_levels).

    근거
    ----
    S1–S2 판정 R37 및 Task 9 fix1 M4·M5(차이 원인·주변 기록).
    """
    k, ms, pid = cell['frame'], cell['ms'], cell['pid']
    # (MISM-1)의 부호를 유지하여 프레임 앞·뒤 사건을 구분한다(Task 9 fix1 M5).
    ups = sorted((e['timestamp'] - ms, e['level']) for e in rec.events
                 if e['type'] == 'LEVEL_UP' and e['participantId'] == pid and abs(e['timestamp'] - ms) <= WINDOW_MS)
    # (MISM-1)의 참가자 ID를 배열 행으로 바꾸고 기록 경계에서 창을 자른다(Task 9 fix1 M5).
    around = rec.frame_num[max(0, k - 2):k + 3, pid - 1, 2]
    return {**cell, 'level_ups_within_5s': [list(u) for u in ups], 'frame_levels_around': [int(v) for v in around],
            'frame_ms_around': [int(v) for v in rec.frame_ts[max(0, k - 2):k + 3]]}


def main(argv=None) -> int:
    """S1–S2 판정 R37의 표본 레벨 대조에 주변 수치 창을 덧붙이며, 판독 오류를 무시하고 다음 경기로 넘어가지 않는다."""
    args = common.parse(common.arguments(__doc__, limit=True), argv)
    try:
        run_s1.require_no_strata()                          # E6: before any raw file is read
    except run_s1.StrataRunning as e:
        return common.refuse(str(e))
    try:
        sample = common.sample_matches(args.run)
    except FileNotFoundError as e:
        return common.refuse(str(e))
    total, explained = common.new_level_counts(), []
    for n, ((_, patch, macro), rec) in enumerate(zip(sample[:args.limit], common.records(sample, args.limit))):
        cells = []
        total.update(common.compare_levels(rec, cells))
        explained += [{'sample': n, 'patch': patch, 'macro': macro, **explain(rec, c)} for c in cells]
    for cell in explained[:SHOWN]:
        print(cell)
    return common.finish(args, SCRIPT, 'level clock against the frame level, sample of S1 matches', total, cells=explained)


if __name__ == '__main__':
    raise SystemExit(main())
