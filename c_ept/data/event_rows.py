"""Event-clock rows of the C model: the open views of M3 (at the first kill f) and of M4 (at delta), one row per unit and kind.

An event-clock row is one engagement unit seen at a reference time ``ref`` on the ms event clock: ``ref = f`` for M3 (the model has seen
the match up to the first kill and predicts the rest of the engagement) and ``ref = delta`` for M4 (it has seen the engagement and
predicts the conversion). The view holds the last regular frame before ``ref`` and the events up to ``ref``; this table fixes the frame:

    frame_k    the last regular frame with a time strictly before ref (A3: a view at reference time ref uses frames before ref and events
               up to ref, so a frame at the same ms as ref is not used). ``MatchRecord.frame_ts`` / ``frame_num`` are indexed by it.
    t0         frame_ts[frame_k]
    age_ms     ref - t0, the frame age of the view (> 0)
    frame_row  the index of that frame in the ``FrameRows`` of the match when it is a frame-clock row (frame 2 or later and before
               T - END_CUT_MS), else -1. Two kinds of regular frame are no row: frames 0 and 1 (the M3 view of a first kill that is at
               or before the time of frame 2, a few ms after 2:00, is frame 1) and frames at or after T - END_CUT_MS (the M4 view of a
               delta that lies after the first such frame). The spec asks for the number of rows with frame_row = -1 to be reported.

Rows: every in-scope unit has an M3 row; it has an M4 row too when delta < T (a unit that ends at the game end or later has an empty
conversion window (delta, min(delta + CONV_MS, T)]). A unit with no regular frame before ``ref`` has no row for that reference; this
happens only for a record without any regular frame (for a match from ``read_match`` frame 0 is at 0 ms and an in-scope f is at least
SCOPE_FIRST_MS). The rows are ordered by (unit, kind). These are the open views used by the module (conditional) evaluation; the masked
views are the onsets of ``frame_rows`` and are not stored here. ``unit`` is the local index into the UnitTable of the match (the S1
assembly adds the match offset).

이 모듈은 열린 보기(M3는 첫 킬 f, M4는 교전 끝 δ에서 다시 묻는 보기)가 읽는 정규 프레임을 정한다. 교전 단위는 한 교전이고 첫 킬 시각 f와 끝 δ를 가지며, 범위 안의 단위마다 기준 시각 ref = f(M3)가 있고, δ < T이면
ref = δ(M4)도 있다. 프레임 번호 k는 ``MatchRecord.frame_ts``의 0 기준 색인이고, 열린 보기의 질문 시각 t₀는 그 프레임의 시각, 프레임 나이 a는 ref − t₀이다. 표의 행은 단위 번호(UnitTable의 0
기준 색인)와 종류 순서로 놓이며, S2는 이 표의 (ref, k)로 보기를 만들고, frame_row는 프레임 나이 분석과 보고에 쓴다(S1–S2 계획 Task 6).

식 목록 (시각은 정수 ms이다)
  (EVROW-1) event_rows: 프레임 k = (ts_j < ref인 j의 개수) − 1, 행 조건 k ≥ 0 ∧ (M3 ∨ ref < T), t₀ = ts_k, 프레임 나이 a = ref − t₀
  (EVROW-2) event_rows: 프레임 시계 행 번호 frame_row = pos(rk(pos) = k이면), 아니면 −1

공통 가정. 코드가 검사하는 것은 rows.k가 엄격히 증가하는 것뿐이다. frame_ts가 오름차순인 것, rows와 u가 rec와 같은 경기에서 나온 것, k(int16), 단위 번호(int32),
frame_row(int32)와 종류(int8)로 바꾸는 값의 범위는 검사하지 않는다.
"""
# Ref: docs/superpowers/specs/2026-10-20-c-ept-design.md section 4 (event clock: M3 predicted again at f and M4 at delta from the last
#      frame, age a = reference time - t0), sections 3.2 and 3.3 (the open views serve the module evaluation, M4 needs delta < T for a
#      conversion window) and section 11 item 8 (the frame of an open view is the regular frame just before f or delta; frame_row = -1
#      when it is no frame-clock row, and the count is reported).
# Ref: D:\LOL_Data\runs\20261019_link_rule_fit_1\event_stream_probe.py (the prototype: j = searchsorted(frame times, f, 'left') - 1, the
#      last frame strictly before f, age a = f - t0).
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from c_ept.frame_rows import FrameRows
from c_ept.reader import MatchRecord
from c_ept.units import UnitTable

KIND_M3, KIND_M4 = 1, 2        # the view at the first kill f, the view at delta


@dataclass(frozen=True, eq=False)
class EventRows:
    """The event-clock rows of one match, ordered by (unit, kind); see the module text for the columns."""
    unit: np.ndarray           # int32, index into the UnitTable of the match
    kind: np.ndarray           # int8, KIND_M3 or KIND_M4
    ref: np.ndarray            # int64, ms: f (M3) or delta (M4), the reference time of the view
    frame_k: np.ndarray        # int16, index into MatchRecord.frame_ts: the last regular frame with a time < ref
    t0: np.ndarray             # int64, ms: frame_ts[frame_k]
    age_ms: np.ndarray         # int64, ms: ref - t0
    frame_row: np.ndarray      # int32, index into the FrameRows of the match when frame_k is a frame-clock row, else -1


def event_rows(rec: MatchRecord, u: UnitTable, rows: FrameRows) -> EventRows:
    """The M3 and M4 rows of the in-scope units ``u`` of the match ``rec``, with the frame-clock rows ``rows`` of the same match.

    ``rows.k`` must increase strictly (``frame_rows`` makes it so); rows out of order are refused, because the search for the row of a
    frame would be wrong without a message. ``rows`` may be empty (a match with fewer than three regular frames, or none of them before
    T - END_CUT_MS): every ``frame_row`` is then -1. The result holds new arrays, not views of the inputs; with nothing to view every
    array is empty with its dtype.

    수식
    ----
    (EVROW-1)  ref(i, M3) = f(i);  ref(i, M4) = δ(i);  k = #{j : ts_j < ref} − 1 (엄격);  행 조건  k ≥ 0  ∧  (종류 = M3  ∨  ref < T)
               t₀ = ts_k;  a = ref − t₀ (정수이고 k의 정의 때문에 ≥ 1)
    (EVROW-2)  pos = min{row : rk(row) ≥ k} (행이 없으면 행의 수);  frame_row = pos (rk(pos) = k일 때), 아니면 −1

      기호    | 뜻                                          | 단위 | 범위                        | 코드 변수
      i       | 범위 안 단위 번호(0부터)                    | —    | UnitTable의 in_scope 행     | unit
      f(i), δ(i) | 단위 i의 첫 킬 시각, 교전 끝             | ms   | 정수                        | u.f, u.delta
      ref     | 보기의 기준 시각                            | ms   | 정수                        | ref
      ts_j    | 정규 프레임 j의 시각                        | ms   | 정수(오름차순은 가정)       | ts (rec.frame_ts)
      k       | 보기가 쓰는 정규 프레임 번호(0부터)         | —    | 0, …, len(ts) − 1           | frame_k (int16으로 변환)
      T       | 경기 끝 시각                                | ms   | 정수                        | rec.T_ms
      t₀      | 프레임 k의 시각                             | ms   | ts_k < ref                  | t0
      a       | 프레임 나이                                 | ms   | ≥ 1                         | age_ms (전역 표의 a는 s)
      rk(row) | 프레임 시계 행 row의 프레임 번호            | —    | 엄격히 증가(검사함)         | rows.k
      pos     | rk에서 k 이상인 첫 행(0부터)                | —    | 0, …, 행의 수               | pos
      frame_row | 보기의 프레임에 해당하는 행 번호          | —    | −1 또는 0, …, 행의 수 − 1   | frame_row (int32로 변환)

    해석
    ----
    식 (EVROW-1)은 열린 보기의 프레임을 ref보다 엄격히 앞선 마지막 정규 프레임으로 정하므로, ref와 같은 ms의 프레임은 쓰지 않는다. 보기가 ref 이하의 사건과
    ref 앞의 프레임만 읽는 인과 규칙(S1–S2 전역 제약 인과(A3))이기 때문이다. M4의 전환 창은 δ < T일 때만 비지 않으므로 이 조건은 M4 행에만 두고, 두 종류 모두 ref 앞에
    정규 프레임이 없으면(k = −1) 행이 없다. 식 (EVROW-2)의 frame_row는 보기의 프레임이 주어진 프레임 시계 행에 있으면 그 행 번호이고, 없으면 −1이다. 행을
    (FRROW-1)로 만들었다면 −1의 원인은 행이 아닌 프레임 0, 1과 경기 끝 컷 이후의 프레임 둘뿐이다(S1–S2 판정 20).

    근거
    ----
    설계서 4절(프레임 나이 a), 3.2절·3.3절(M4는 전환 창이 있을 때만), 11절 항목 8(행이 아닌 프레임의 수 보고); S1–S2 전역 제약 인과(A3)(보기는 ref 앞의 프레임);
    S1–S2 계획 Task 6과 S1–S2 판정 20(행이 아닌 프레임의 두 원인).
    """
    rk = np.asarray(rows.k, dtype=np.int64)
    if rk.size > 1 and np.any(rk[1:] <= rk[:-1]):
        raise ValueError('frame-clock rows must be in frame order: k must increase strictly')
    ts = np.asarray(rec.frame_ts, dtype=np.int64)
    unit = np.flatnonzero(u.in_scope)                                   # ascending: the units are in unit order
    # (EVROW-1)의 후보: 단위마다 M3(ref = f)와 M4(ref = δ)를 한 쌍씩 만든 뒤 아래에서 조건에 맞는 것만 남긴다.
    # the candidates in (unit, kind) order: for every in-scope unit the M3 reference f, then the M4 reference delta
    c_unit = np.repeat(unit, 2)
    c_kind = np.tile(np.array([KIND_M3, KIND_M4], dtype=np.int8), unit.size)
    c_ref = np.stack([u.f[unit], u.delta[unit]], axis=1).astype(np.int64).ravel()
    c_k = np.searchsorted(ts, c_ref, side='left') - 1                   # the last frame strictly before ref; -1: there is none
    keep = (c_k >= 0) & ((c_kind == KIND_M3) | (c_ref < rec.T_ms))      # an M4 row needs delta < T: else its conversion window is empty
    ref, k = c_ref[keep], c_k[keep]
    t0 = ts[k]
    # (EVROW-2): 프레임이 행인지 한 번의 탐색으로 가리는 이유와 마지막 행 뒤의 처리는 다음 영어 주석과 같다.
    # The row of the frame: rk increases, so one search finds the first row with k' >= k, and the frame is a row iff that k' is k. The
    # -1 behind the last row makes a frame after the last row find no row. Not k - 2: a frame at or after T - END_CUT_MS has no row.
    pos = np.searchsorted(rk, k, side='left')
    frame_row = np.where(np.append(rk, -1)[pos] == k, pos, -1)
    return EventRows(unit=c_unit[keep].astype(np.int32), kind=c_kind[keep], ref=ref, frame_k=k.astype(np.int16), t0=t0, age_ms=ref - t0,
                     frame_row=frame_row.astype(np.int32))
