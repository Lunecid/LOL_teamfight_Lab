"""How many engagement onsets that are decided online are later absorbed: the post-hoc merge of review item A1-F4, as a number.

A kill-chain unit is the connected component of the kills linked by the thesis connection rule (C-9): two kills are linked when they lie
within ``G_ms`` of each other in time (15 s, the kill and assist window) and within ``D`` map units in space (2,000 u). The definition stays
as it is, because it is the one the thesis states. What a *causal* model can know at the moment of a kill is narrower: it sees only the
kills up to then. A kill that links to no earlier kill is the onset of a new unit, decided on the spot. A later kill can link two units that
were separate and merge them; the unit that began later then has no onset of its own any more, it was absorbed. ``absorbed_share`` counts
that, so that the author can decide from one number (the plan's threshold is 2 %) whether an onset definition that sees the future is
acceptable for the causal onset (the M2 hazard).

Input and output. Arrays the caller has already read: the kill times (ms) and the kill positions (map units) of ONE match, from the FIT
role only (see below). The function reads no file, no store and no match, imports nothing but numpy, and returns a dict of plain numbers.
Counting over many matches is the caller's sum of ``n_onsets`` and ``n_absorbed`` (never a mean of the shares).

FIT only. The plan (B15, M15) allows the measurement on the FIT kills only: it feeds a design decision about the hazard, and a number taken
on STOP, DEV, CAL, AUDIT or TEST kills would be a look at held-out matches. The signature of the plan has no ``role`` argument, so this
module cannot check it: the caller (the S4b runner, after approval) is responsible for passing the FIT kills, and nothing here opens a
role. The module ``common`` and the sweeps of ``c_ept.checks`` read only the QA roles for the same reason (B21).

Order of kills. The kills are processed one ms at a time. A causal model at the ms t sees all kills of that ms together, so there is no
online order among them: the kills of one ms are linked to the earlier units and to each other as one batch, a component of the batch with
no earlier unit is ONE onset, and a component that joins m earlier units absorbs m - 1 onsets. A kill of an ms therefore never absorbs
an onset decided at the same ms (B15 counts absorption by a LATER kill), and the result does not depend on how the caller lists the kills
(review I1; the internal order of the batch is only for processing). Linking is symmetric and ``<=`` at both thresholds.


방법에서의 위치. 이 모듈은 발생(onset) 정의를 인과적으로 바꿀지 판단하는 측정 도구이다(결정 B15). 이 모듈은 아무 자료도 읽지 않는다.
교전 단위는 연결 규칙(G, D)으로 이어진 킬의 연결 성분이고, 인과 모형은 킬이 일어난 시점에 그때까지의 킬만 안다. 나중 킬이 두 단위를 이으면
먼저 정한 발생 하나는 사라진다. 그 몫을 센다.
식 목록.
    (ONS-1) 연결: link(i, j) = 1[|t_i − t_j| ≤ G_ms 그리고 ‖p_i − p_j‖ ≤ D].
    (ONS-2) 실시간 발생: 같은 ms의 킬을 한 묶음 B로 보고, B 안의 킬끼리와 앞선 킬(t_j < t_B)에 link로 이은 성분 c마다
            onset(c) = 1[c가 앞선 킬과 하나도 이어지지 않음].
    (ONS-3) 병합: 묶음의 성분 c가 서로 다른 m_c개의 기존 단위에 이어지면 c와 그 m_c개가 한 단위가 되고, 흡수된 발생은 m_c − 1개이다.
    (ONS-4) 집계: n_onsets = Σ_c onset(c); n_absorbed = Σ_c max(m_c − 1, 0); n_units = n_onsets − n_absorbed;
            absorbed_share = n_absorbed / n_onsets (n_onsets = 0 이면 정의되지 않아 None).
공통 가정. 시간은 ms, 위치는 지도 단위, 한 경기의 킬만 한 번에 넣는다. 위치가 없는 킬은 호출자가 먼저 걸러야 하며 이 함수는 NaN을 받지 않는다.
"""
from __future__ import annotations

import numpy as np

DEFAULT_G_MS = 15_000       # the thesis kill and assist window (C-9), ms
DEFAULT_D = 2_000           # the thesis connection distance (C-9), map units


def _numeric_array(a, name: str) -> np.ndarray:
    """``a`` as a float64 or int64 array; a bool, text or object array is a TypeError (the message holds no value)."""
    a = np.asarray(a)
    if a.dtype.kind == 'b':
        raise TypeError(f'{name} must be a numeric array, not a bool array')
    if a.dtype.kind not in 'iuf':
        raise TypeError(f'{name} must be a numeric array')
    return a.astype(np.int64 if a.dtype.kind in 'iu' else np.float64, copy=False)


def _threshold(value, name: str):
    """``value`` if it is a real number that is finite and not negative (a bool is not one); a TypeError or ValueError otherwise."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f'{name} must be a number')
    if not np.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be a finite number that is not negative')
    return value.item() if isinstance(value, np.generic) else value


def absorbed_share(kill_times, kill_xy, G_ms=DEFAULT_G_MS, D=DEFAULT_D) -> dict:
    """The kills of one match, FIT only, in the online order, one ms at a time: how many onsets were decided and how many were absorbed later.

    ``kill_times``: shape (n,), ms. ``kill_xy``: shape (n, 2), map units, one row per kill, finite. ``G_ms`` and ``D`` are the thresholds
    of the connection rule (defaults 15,000 ms and 2,000 u). Returns ``n_kills``, ``n_onsets`` (the components of the kills of one ms that
    link to no earlier kill),
    ``n_units`` (the connected components at the end), ``n_absorbed`` (``n_onsets - n_units``), ``absorbed_share``
    (``n_absorbed / n_onsets`` as a float, None for no kill), ``G_ms`` and ``D``. The inputs are not changed.

    The caller passes FIT kills only (the signature has no role argument, see the module text). A wrong shape, a length that differs, a
    value that is not finite, a bool or text array, or a threshold that is negative or not finite is a ValueError or TypeError whose
    message names the argument and never a value.

    수식
    ----
    (ONS-1) link(i, j) = 1[|t_i − t_j| ≤ G_ms 그리고 (x_i − x_j)^2 + (y_i − y_j)^2 ≤ D^2].
    (ONS-2) 같은 ms의 킬 묶음 B(t_i = t_B)에서, B 안의 link로 이은 성분 c마다
            onset(c) = 1[모든 i ∈ c, j ∉ B, t_j < t_B 에서 link(i, j) = 0].
    (ONS-3) m_c = |{root(j) : i ∈ c, t_j < t_B, link(i, j) = 1}| (c가 이은 서로 다른 기존 단위의 수);
            m_c ≥ 2 이면 그 m_c개 단위와 c가 한 단위로 합쳐지고 흡수된 발생은 m_c − 1개이다.
    (ONS-4) n_onsets = Σ_c onset(c); n_absorbed = Σ_c max(m_c − 1, 0); n_units = n_onsets − n_absorbed;
            absorbed_share = n_absorbed / n_onsets.

    기호 | 뜻 | 단위 | 범위 | 코드 변수
    i, j | 시간순 킬 번호 | — | 0 … n−1 | i, j
    B, c | 같은 ms의 킬 묶음·그 묶음 안의 연결 성분 | — | 묶음은 시각이 같은 킬 | g0:g1, components
    t_i | 킬 i의 시각 | ms | 유한 | t
    (x_i, y_i) | 킬 i의 위치 | 지도 단위 | 유한 | xy
    G_ms | 같은 교전으로 보는 최대 시간 간격(킬·어시스트 인정 창) | ms | 0 이상, 유한 | G_ms
    D | 같은 교전으로 보는 최대 거리(연결 거리) | 지도 단위 | 0 이상, 유한 | D
    root(j) | 킬 j가 속한 단위(union-find 뿌리) | — | 킬 번호 | find(j)
    m_c | 묶음 성분 c가 한꺼번에 잇는 서로 다른 기존 단위의 수 | 단위 | 0 이상 | len(roots)
    n_onsets, n_absorbed, n_units | 실시간 발생·흡수된 발생·끝의 단위 수 | 개 | 0 ≤ n_absorbed < n_onsets 또는 모두 0 | 반환 dict
    absorbed_share | 실시간 발생 가운데 나중에 흡수된 몫 | — | [0, 1) 또는 None | 반환 dict

    해석
    ----
    같은 ms의 킬 사이에는 실시간 순서가 없다. 그 시각의 모형은 같은 ms의 킬을 한꺼번에 보므로 묶음으로 처리하고, 같은 ms의 킬이 같은 ms에
    정한 발생을 흡수하는 일은 없다(좌표 순서 같은 임의 규칙에 몫이 의존하지 않게 하는 검토 I1의 수정이다). 입력 순서도 결과에 영향이 없다.
    식 (ONS-2)는 인과 모형이 그 ms에 아는 것만으로 발생을 정한다: 앞의 킬과 이어지지 않으면 새 교전의 시작이다. 식 (ONS-3)은 나중 킬이
    두 단위를 이어 붙이는 경우로, 이때 나중에 시작한 단위는 처음부터 한 교전의 일부였던 것이 되어 먼저 정한 발생이 틀린 것으로 판명된다
    (검토 A1-F4의 다리 킬). 이 몫이 작으면 현재 정의(연결 성분)를 그대로 두고 인과 발생과의 차이를 무시할 수 있고, 크면 인과적 발생 정의를
    저자 결정으로 올려야 한다(계획 B15, 기준 2 %). 경계는 시간 간격 G_ms와 거리 D 모두 같으면 이어진다(≤). 킬이 없으면 몫은 정의되지
    않으므로 0이 아니라 None이다. 한 단위에 킬이 계속 붙는 사슬은 발생 하나이고 흡수는 0이다.

    근거
    ----
    S4b 준비 계획 B15·Task M15, 검토 A1-F4(사후 병합 발생); 학위논문 연결 규칙(G 15 s = 킬·어시스트 인정 창, L 2,000 u, C-9).
    """
    t = _numeric_array(kill_times, 'kill_times')
    xy = _numeric_array(kill_xy, 'kill_xy')
    if t.ndim != 1:
        raise ValueError('kill_times must be one-dimensional')
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError('kill_xy must have the shape (n, 2)')
    if len(xy) != len(t):
        raise ValueError('kill_xy needs one row per kill in kill_times')
    if not (np.isfinite(t).all() and np.isfinite(xy).all()):
        raise ValueError('kill_times and kill_xy must be finite (drop the kills without a position before the call)')
    G_ms, D = _threshold(G_ms, 'G_ms'), _threshold(D, 'D')
    n = len(t)
    order = np.lexsort((xy[:, 1], xy[:, 0], t))                  # time order; (x, y) only makes the processing deterministic, the result does not use it
    t, xy = t[order], xy[order].astype(np.float64, copy=False)
    d_sq = float(D) * float(D)

    # Ref: 연결 성분을 한 번에 하나씩 세는 union-find(서로소 집합). Tarjan, R. E. (1975), "Efficiency of a good but not linear set union
    #      algorithm", J. ACM 22(2); 경로 반감: Tarjan & van Leeuwen (1984), "Worst-case analysis of set union algorithms", J. ACM 31(2).
    parent = list(range(n))                                      # union-find over the time-ordered kills

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]                        # path halving
            a = parent[a]
        return a

    n_onsets = n_absorbed = 0
    lo = 0                                                       # the first earlier kill still within G_ms of this ms (t is sorted)
    g0 = 0
    while g0 < n:                                                # one batch = all kills of one ms, g0 .. g1 − 1
        g1 = g0 + 1
        while g1 < n and t[g1] == t[g0]:
            g1 += 1
        while t[g0] - t[lo] > G_ms:
            lo += 1
        m = g1 - g0
        # (ONS-2): the earlier units that each kill of the batch links to, taken before the batch merges anything (the roots are of the earlier kills)
        earlier = [{find(int(j)) for j in np.flatnonzero((xy[lo:g0, 0] - xy[i, 0]) ** 2 + (xy[lo:g0, 1] - xy[i, 1]) ** 2 <= d_sq) + lo}
                   if lo < g0 else set() for i in range(g0, g1)]
        # the kills of the batch link to each other (the same ms is within G_ms); their components are found by a small union-find
        batch = list(range(m))
        if m > 1:
            bx, by = xy[g0:g1, 0], xy[g0:g1, 1]
            close = (bx[:, None] - bx[None, :]) ** 2 + (by[:, None] - by[None, :]) ** 2 <= d_sq
            for a in range(m):
                for b in range(a + 1, m):
                    if close[a, b]:
                        ra, rb = a, b
                        while batch[ra] != ra:
                            ra = batch[ra]
                        while batch[rb] != rb:
                            rb = batch[rb]
                        if ra != rb:
                            batch[max(ra, rb)] = min(ra, rb)
        components: dict = {}
        for a in range(m):
            r = a
            while batch[r] != r:
                r = batch[r]
            comp = components.setdefault(r, [set(), []])
            comp[0] |= earlier[a]
            comp[1].append(g0 + a)
        for roots, members in components.values():
            if not roots:
                n_onsets += 1                                    # a component of this ms with no earlier unit is one new engagement
                top = members[0]
            else:
                roots = {find(r) for r in roots}             # an earlier component of this ms may already have joined some of these units
                n_absorbed += len(roots) - 1                     # (ONS-3): it joins m_c earlier units, m_c − 1 onsets are absorbed
                top = min(roots)
                for r in roots:
                    parent[r] = top
            for i in members:
                parent[i] = top
        g0 = g1
    n_units = len({find(i) for i in range(n)})
    assert n_units == n_onsets - n_absorbed                      # (ONS-4) the count identity of the union-find
    return {'n_kills': int(n), 'n_onsets': int(n_onsets), 'n_units': int(n_units), 'n_absorbed': int(n_absorbed),
            'absorbed_share': float(n_absorbed / n_onsets) if n_onsets else None, 'G_ms': G_ms, 'D': D}
