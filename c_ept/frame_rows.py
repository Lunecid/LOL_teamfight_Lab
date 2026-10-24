"""Frame-clock rows of the C model, their exposure per occurrence bin, and the onsets table (row, bin, mark, unit).

A row is a regular frame k >= 2 (the first frame at or after 2:00) whose time t0 lies before T - END_CUT_MS; t0 is the real frame time,
so a row at reference time t0 may use that frame itself (A3). It looks ahead over the occurrence window (t0, hi] with
hi = min(t0 + H_OCC_MS, T - END_CUT_MS), cut into N_BINS bins of BIN_MS. The evaluation horizons of ``config.EVAL_H_MS`` use the first
6 / 12 / 18 bins of the same row.

    exposure_ms[q]   the part of bin q = (t0 + BIN_MS q, t0 + BIN_MS (q + 1)] that lies inside (t0, hi]: BIN_MS for a full bin, less
                     for the bin that holds the cut at the game end, 0 for the bins after it. This is the time Delta_q during which
                     the counting process of bin q is observed; the end of the game is a competing end of observation, not a quiet
                     stretch.

An onset is one engagement unit seen from one row: the pair (row, unit) with t0 < f <= hi, where f is the unit's first kill. The first
kill is what is counted (one per engagement), so a unit appears in every row whose window holds f: with frames 60 s apart and a 180 s
window, in three rows. A unit with f <= t0 is in progress at that row and is no occurrence of it (the event clock covers it). The bin
is (f - t0 - 1) // BIN_MS, the mark is the unit's first-kill mark and ``unit`` is the local index into the UnitTable of the match (the
S1 assembly adds the match offset). The triples (row, bin, mark) are the training pairs of the masked views of M3 and M4. The onsets
table itself IS stored (the ``onsets/*`` arrays of the S1 store); what is not stored separately is the masked-view input: the view of
a pair is the view of its frame-clock row (``row``, at its reference time t0), which the frame-clock rows already give.

이 모듈은 한 경기의 프레임 시계 행, 행마다 발생 구간별 노출, 그리고 발생(onset) 표를 만든다. 정규 프레임은 ``MatchRecord.frame_ts``의 항목이고 프레임 번호 k는 0에서 시작하는 그 색인이다. 교전 단위는 한 교전이고 첫
킬 시각 f를 가지며, 표지 m은 그 단위의 첫 킬을 가리키는 번호(선제 킬 쌍 50개와 처형 2개; 전역 기호 m)이다. 행은 k ≥ 2이고 시각이 T_cut = T − END_CUT_MS보다 엄격히 앞서는 정규 프레임이고, 질문 시각 t₀는 그
프레임의 실제 시각(ms)이다. 발생 창은 (t₀, hi]이며 BIN_MS 폭의 구간 q = 0, …, N_BINS − 1로 나뉜다. 발생은 f가 창에 드는 (행, 단위) 쌍이고, 구간마다 관측 시간(노출)을 센다.

식 목록 (시각은 정수 ms이다)
  (FRROW-1) frame_rows: 행 선택 ts_k < T_cut(k ≥ 2), 질문 시각 t₀ = ts_k, 창의 끝 hi = min(t₀ + H_occ, T_cut)
  (FRROW-2) frame_rows: 구간 q의 노출 Δ_q = 구간 (t₀ + BIN_MS·q, t₀ + BIN_MS·(q + 1)]과 창 (t₀, hi]의 겹친 길이
  (FRROW-3) onsets: 발생 조건 t₀ < f ≤ hi와 구간 번호 q = ⌊(f − t₀ − 1) / BIN_MS⌋

공통 가정. 코드가 검사하는 것은 onsets 입력의 f가 줄어들지 않는 것뿐이다. frame_ts가 오름차순인 것, 프레임 2가 SCOPE_FIRST_MS 이후인 것(S1 말뭉치의 관측; S1–S2 장부 Task 5 미결 메모)과 k, 구간 번호,
노출을 좁은 정수형으로 바꾸는 값의 범위는 검사하지 않는다. N_BINS · BIN_MS = H_OCC_MS는 config.py가 import 때 assert로 확인한다.
"""
# Ref: docs/superpowers/specs/2026-10-20-c-ept-design.md section 3.1 (M2: one first kill per engagement, 10 s bins, exposure cut at
#      T - 30 s), section 4 (frame clock: rows from 2:00 while t0 < T - 30 s, window (t0, min(t0 + 180 s, T - 30 s)]) and section 11
#      item 7 (t0 is the real frame time, not the nominal minute).
# Ref: D:\LOL_Data\runs\20261019_link_rule_fit_1\g2_anchor_check.py (the prototype: rows t0 = 120, 180, ... s while t0 < T - 30 s, window
#      (t0, min(t0 + 60, T - 30)], counts of the first kill f in the window).
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from c_ept import config as C
from c_ept.reader import MatchRecord
from c_ept.units import UnitTable

FIRST_ROW_FRAME = 2      # regular frame 2 is the first at or after 2:00 (frame k is at about 60 s k); frames 0 and 1 are no rows


@dataclass(frozen=True, eq=False)
class FrameRows:
    """The frame-clock rows of one match, in frame order. ``k`` indexes ``MatchRecord.frame_ts`` / ``frame_num``."""
    k: np.ndarray              # int16, regular-frame index, >= 2
    t0: np.ndarray             # int64, ms: the time of frame k, the reference time of the row
    hi: np.ndarray             # int64, ms: end of the occurrence window, min(t0 + H_OCC_MS, T - END_CUT_MS)
    exposure_ms: np.ndarray    # (n, N_BINS) int32: the part of each bin inside (t0, hi], 0 .. BIN_MS


@dataclass(frozen=True, eq=False)
class Onsets:
    """The (row, unit) pairs with t0 < f <= hi, ordered by row and, within a row, by unit (that is, by f)."""
    row: np.ndarray            # int32, index into the FrameRows of the match
    bin: np.ndarray            # int8, 0 .. N_BINS - 1
    mark: np.ndarray           # int16, first-kill mark of the unit, 0 .. N_MARKS - 1
    unit: np.ndarray           # int32, index into the UnitTable of the match


def frame_rows(rec: MatchRecord) -> FrameRows:
    """The rows of a match: every regular frame k >= 2 with ``frame_ts[k] < T_ms - END_CUT_MS`` (strictly before).

    A match with fewer than three regular frames, or with none of them before the cut, has no rows: every array is empty with its
    dtype and ``exposure_ms`` has shape (0, N_BINS). The result holds new arrays, not views of the record.

    수식
    ----
    (FRROW-1)  T_cut = T − END_CUT_MS;  k ∈ {2, …, len(ts) − 1} 중 ts_k < T_cut(엄격);  t₀(k) = ts_k;  hi(k) = min(t₀(k) + H_occ, T_cut)
    (FRROW-2)  Δ_q(k) = min(max(hi(k) − (t₀(k) + BIN_MS · q), 0), BIN_MS),  q = 0, …, N_BINS − 1

      기호   | 뜻                                | 단위 | 범위                    | 코드 변수
      ts_k   | 정규 프레임 k의 시각              | ms   | 정수(오름차순은 가정)   | ts[k] (rec.frame_ts)
      T      | 경기 끝 시각                      | ms   | 정수                    | rec.T_ms
      T_cut  | 마지막으로 보는 시각              | ms   | 정수                    | end
      k      | 행의 정규 프레임 번호(0부터)      | —    | 2, …, len(ts) − 1       | k (int16으로 변환)
      t₀(k)  | 행의 질문 시각                    | ms   | < T_cut                 | t0
      H_occ  | 발생 창의 길이                    | ms   | 정수                    | C.H_OCC_MS
      hi(k)  | 발생 창의 끝                      | ms   | t₀(k) < hi(k) ≤ T_cut   | hi
      q      | 구간 번호(0부터)                  | —    | 0, …, N_BINS − 1        | exposure_ms의 열 번호
      Δ_q(k) | 구간 q의 노출                     | ms   | 0, …, BIN_MS            | exposure_ms[:, q] (int32)

    해석
    ----
    식 (FRROW-1)은 질문 행을 세 번째 정규 프레임(k = 2)부터 T_cut 앞까지의 프레임으로 정하고, 질문 시각을 명목 분이 아닌 프레임의 실제 시각으로 둔다.
    행이 2:00부터라는 설계를 프레임 번호로 구현하므로(프레임 k는 약 k분) 프레임 0, 1은 행이 아니며, T_cut과 같은 시각의 프레임도 엄격한 부등호 때문에 행이 아니다.
    식 (FRROW-2)의 Δ_q는 구간이 관측 창 안에 든 길이이므로 온전한 구간은 구간 폭 전체이고, 컷이 안에 든 구간은 그보다 짧으며, 컷 뒤의 구간은 0이다.
    경기 끝 컷 뒤를 노출에서 빼는 것은 경기 끝이 조용한 시간이 아니라 관측을 끝내는 경쟁 종결이기 때문이다. 노출이 0인 칸은 (NLL-1)에서 점수가 0이다.

    근거
    ----
    설계서 3.1절(구간과 노출 컷), 4절(프레임 시계 행의 시작과 창), 11절 항목 7(실제 프레임 시각); S1–S2 계획 Task 5(행과 노출).
    """
    ts = np.asarray(rec.frame_ts, dtype=np.int64)
    # (FRROW-1)의 T_cut: 행의 시각과 창의 끝을 모두 이 값에서 자르므로 컷 뒤의 시간은 어느 행의 노출에도 들지 않는다.
    end = rec.T_ms - C.END_CUT_MS
    k = np.arange(FIRST_ROW_FRAME, len(ts), dtype=np.int64)         # empty when there are fewer than three frames
    k = k[ts[FIRST_ROW_FRAME:] < end]
    t0 = ts[k]
    hi = np.minimum(t0 + C.H_OCC_MS, end)
    # (FRROW-2)의 Δ_q: 구간 시작을 열로 펼친 뒤 창 끝과의 차를 [0, BIN_MS]로 자르면 구간과 창의 겹친 길이가 된다.
    bin_start = t0[:, None] + C.BIN_MS * np.arange(C.N_BINS, dtype=np.int64)
    exposure = np.clip(hi[:, None] - bin_start, 0, C.BIN_MS)
    return FrameRows(k=k.astype(np.int16), t0=t0, hi=hi, exposure_ms=exposure.astype(np.int32))


def onsets(rows: FrameRows, u: UnitTable) -> Onsets:
    """The onsets of the units ``u`` in the rows ``rows`` of one match: every (row, unit) with ``t0 < f <= hi``.

    A unit with f <= t0 is not in that row. Units are not filtered by ``in_scope``: for rows made by ``frame_rows`` every unit in a window
    is in scope whenever frame 2 is at or after SCOPE_FIRST_MS, which holds on the S1 corpus but is not checked here, since f > t0 and
    f <= hi <= T - END_CUT_MS.
    ``u.f`` must not decrease (``build_units`` orders the units by their first kill); a unit table that does not is refused, because
    the window search would be wrong without a message.

    수식
    ----
    (FRROW-3)  onset(row, i) ⇔ t₀(row) < f(i) ≤ hi(row);   q(row, i) = ⌊(f(i) − t₀(row) − 1) / BIN_MS⌋;   m(row, i) = m(i)
               순서: 행 번호 row 오름차순, 한 행 안에서는 단위 번호 i 오름차순(f 오름차순)

      기호   | 뜻                                          | 단위 | 범위                      | 코드 변수
      row    | 행 번호(0부터, ``rows``의 색인)             | —    | 0, …, 행 수 − 1           | row (int32로 변환)
      i      | 단위 번호(0부터, 이 경기 UnitTable의 색인)  | —    | 0, …, 단위 수 − 1         | unit (int32로 변환)
      f(i)   | 단위 i의 첫 킬 시각                         | ms   | 정수, i에 대해 비감소     | u.f
      t₀(row)| 행 row의 질문 시각                          | ms   | 정수                      | rows.t0
      hi(row)| 행 row의 창의 끝                            | ms   | 정수                      | rows.hi
      q      | 발생의 구간 번호                            | —    | 0, …, N_BINS − 1          | bin (int8로 변환)
      m(i)   | 단위 i의 표지(선제 킬 쌍 또는 처형)         | —    | 0, …, N_MARKS − 1         | u.mark (int16으로 변환)

    해석
    ----
    식 (FRROW-3)은 첫 킬이 행의 창 (t₀, hi]에 드는 (행, 단위) 쌍을 발생으로 센다. 아래 끝을 엄격하게 둔 것은 f ≤ t₀인 교전이 그 행에서 이미 진행 중이어서 사건 시계가
    맡기 때문이고(S1–S2 전역 제약 D1–D4), 위 끝을 포함하는 것은 설계서 4절의 창 정의를 따른 것이다. 같은 교전은 f를 창에 담는 행마다 한 번씩 나타나 여러 학습 쌍을 내며,
    그 개수는 프레임 간격과 창 길이로 정해지고 이 함수는 제한하지 않는다. 구간 번호의 −1은 오른쪽이 닫힌 구간의 끝 경계를 그 구간에 넣으므로, f ≤ hi인 발생이 든 구간은
    (FRROW-2)에서 노출이 양수이다. 구간 번호가 0부터 마지막 구간까지에 드는 것은 행을 (FRROW-1)로 만들었을 때만 성립한다.

    근거
    ----
    설계서 3.1절(첫 킬 하나를 한 번 센다), 4절(발생 창의 정의); S1–S2 계획 Task 5(행마다 한 번, 구간 경계 시험), 전역 제약 예측 계약(D1–D4)(진행 중 교전은 발생 아님); 구간별 노출은 (FRROW-2).
    """
    f = u.f
    if f.size > 1 and np.any(f[1:] < f[:-1]):
        raise ValueError('first-kill times must not decrease: the units are in time order')
    # (FRROW-3)의 조건 t₀ < f ≤ hi: f가 정렬되어 있으므로 두 번의 오른쪽 탐색이 한 행의 발생을 이어진 구간 [first, stop)으로 만든다.
    first = np.searchsorted(f, rows.t0, side='right')               # units before this index have f <= t0
    stop = np.searchsorted(f, rows.hi, side='right')                # units before this index have f <= hi
    count = stop - first
    row = np.repeat(np.arange(len(rows.t0), dtype=np.int64), count)
    start = np.cumsum(count) - count                                # where the onsets of each row begin in the output
    unit = np.arange(int(count.sum()), dtype=np.int64) - np.repeat(start - first, count)
    # (FRROW-3)의 q: 정수 ms의 바닥 나눗셈이다. f − t₀ − 1 ≥ 0이므로 음수 쪽으로 내림하는 경우는 없다.
    bin_ = (f[unit] - rows.t0[row] - 1) // C.BIN_MS
    return Onsets(row=row.astype(np.int32), bin=bin_.astype(np.int8), mark=u.mark[unit].astype(np.int16), unit=unit.astype(np.int32))
