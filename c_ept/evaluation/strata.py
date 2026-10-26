r"""The strata of the evaluation and the price sensitivity (S4b preparation M13; decisions D18 and D19, the small items B17a and B22).

Secondary analyses of the S5/S7 evaluation read the same endpoints on parts of the rows. This module makes the masks of those parts from the arrays of the stores and the pinned rule tables; it
reads no match file and no label store, and it never changes an endpoint.

* Champion strata (decision D18). The trait vector of a champion identifies it inside a patch (171 of 171 distinct vectors), so the claim of C3 is narrowed to adaptation to champions whose stats
  changed. On 15.22 the rows with at least one champion whose trait signature (``traits.trait_signature``: the static, base and growth values of the Data Dragon table, no level) differs from 15.20,
  and the rows without, are compared by the difference Delta = metric(trait) - metric(ID); on 16.13, where nearly every champion changed, the rows with a champion that 15.20 does not have are
  reported on their own. ``champion_strata`` gives three masks that partition the rows: ``unseen`` (a champion without a signature in its own patch or in the reference patch: it cannot be said to be
  changed or unchanged), ``changed`` (no unseen champion and a changed one) and ``unchanged`` (no unseen champion and no changed one). An unseen champion is never folded into the other two.
* The rows with objectives from before t0 (B17a). The truth of a t0 row holds the objectives that the onsets of the row attribute to [tau, delta), so for an onset with tau <= t0 < f it holds the
  objectives of [tau, t0], which the t0 input already contains. ``pre_t0_objective_rows`` marks those rows (27.7 % of the rows in the evidence of the review); the definition is kept and the stratum is
  reported separately.
* The sensitivity "all flow prices" (decision D19). The primary f0 keeps the prices of G-S3a. The pre-registered sensitivity prices every group by its flow price (``value.load_flow_prices``; the base and
  nexus towers and the inhibitors keep the immediate value) with a second ``Composer`` of the same seed (``flow_composer``: the random draws pair with the primary composer, only the values move) and
  recomputes the truth in memory (``value.reprice_units``; no label store). It reports the agreement of the winner signs (``winner_sign_agreement``), the rank correlation of the realised V_start
  (``v_start_rank_correlation``) and whether a primary verdict changes (the consumer of the endpoints decides that). A1 is not retrained and is left out of it.

수식
    (STR-1)  변함(c, p) ⇔ σ(p, c) ≠ σ(ref, c)  (둘 다 None이 아닐 때) ;  처음 봄(c, p) ⇔ σ(p, c) = None 또는 σ(ref, c) = None
             행 r: 처음 봄 ⇔ ∃ c ∈ r 처음 봄 ;  변한 행 ⇔ ¬처음 봄 ∧ ∃ c ∈ r 변함 ;  안 변한 행 ⇔ ¬처음 봄 ∧ ¬∃ 변함
    (STR-2)  t₀ 앞 오브젝트 행 r ⇔ ∃ (r, u) ∈ 발생 : τ_u ≤ t₀_r < f_u

      기호      | 뜻                                                     | 단위   | 범위              | 코드 변수
      σ(p, c)   | 패치 p에서 챔피언 c의 특성 서명 (``traits.trait_signature``) | —      | 튜플 또는 None    | trait_signature
      ref       | 기준 패치 (결정 D18의 15.20)                              | —      | 표가 있는 패치    | reference
      r, c      | 경기 행, 그 행의 챔피언 열 10개                          | —      | —                 | match_champ[r]
      τ_u, f_u  | 교전 u의 시작 규칙 앵커 (f − 15 s)와 첫 킬 시각          | ms     | 정수              | unit_tau_ms, unit_f_ms
      t₀_r      | 프레임 행 r의 기준 시각                                   | ms     | 정수              | t0_ms

해석
    층은 보조 분석(Holm 밖)의 분모를 가르는 마스크일 뿐 예측이나 점수를 바꾸지 않는다. "변함"은 표의 값이 정확히 다름이고, 패치 이름이 같은 행(기준 패치 자신)은 변함이 없다. 처음 보는 챔피언(기준
    패치에 없는 새 챔피언, 자기 패치 표에 없는 키)이 있는 행은 비교할 서명이 없으므로 변함에도 안 변함에도 넣지 않고 따로 센다(결정 D18: 16.13의 새 챔피언 둘). t₀ 앞 오브젝트 행은 τ ≤ t₀ < f인
    발생을 가진 행이며 경계(τ = t₀)를 포함하고 f = t₀는 제외한다. 모든 시각은 정수 ms로 읽어 경계 행이 흔들리지 않는다.

근거
    결정 D18 (B7을 챔피언 신원 코드로 읽는 층 비교; 15.22에서 바뀐 챔피언 층, 16.13 보조), 검토 EPT-T010 영역 1의 5번 질문과 결정 B17a (t₀ 행 정답에 [τ, t₀]의 오브젝트가 들어간다: 정의를
    유지하고 τ ≤ t₀ < f인 행을 보조 층으로 보고한다), 계획 M13.
"""
# Ref: harness/decisions.md D18 (2026-10-25, B7: the champion trait vector as an identity code; the stratum of champions changed since 15.20) and D19 (the price adoption rule and the
#      pre-registered sensitivity "all flow prices"); docs/superpowers/plans/2026-10-26-c-ept-s4b-prep.md Task M13 and the small items B17a and B22.
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from c_ept import config as C
from c_ept.preprocessing import traits
from c_ept.valuation import compose, value

REFERENCE_PATCH = '15.20'            # decision D18: the champion stratum is "changed versus 15.20"


class ChampionStrata(NamedTuple):
    """Three boolean masks over the rows that partition them (``changed``, ``unchanged``, ``unseen``); see the module text."""
    changed: np.ndarray
    unchanged: np.ndarray
    unseen: np.ndarray


def _patch_names(match_patch, n: int) -> np.ndarray:
    """The patch of each of the ``n`` rows as an array of names: a text array as it is, an integer array as the numbers of ``config.PATCHES`` (the form of ``matches/patch``)."""
    a = np.asarray(match_patch)
    if a.ndim != 1 or len(a) != n:
        raise ValueError(f'match_patch must be 1-d with one entry for each of the {n} rows of match_champ, not shape {a.shape}')
    if a.size == 0:
        return a.astype(str)
    if a.dtype.kind in 'iu':
        if a.min() < 0 or a.max() >= len(C.PATCHES):
            raise ValueError(f'match_patch: a patch number outside 0 .. {len(C.PATCHES) - 1} (config.PATCHES)')
        return np.array(C.PATCHES)[a.astype(np.int64)]
    if a.dtype.kind in 'US' or a.dtype == object:
        return a.astype(str)
    raise TypeError(f'match_patch must hold patch names or patch numbers, not {a.dtype}')


def _signature(patch: str, champ: int, what: str):
    try:
        return traits.trait_signature(patch, champ)
    except FileNotFoundError:
        raise ValueError(f'{what}: there is no Data Dragon champion table for the patch {patch!r}') from None


def champion_strata(match_champ, match_patch, reference: str = REFERENCE_PATCH) -> ChampionStrata:
    """The ``ChampionStrata`` of the rows: ``match_champ`` (n, 10) integer Data Dragon keys, ``match_patch`` (n,) the patch of each row (names such as '15.22', or the numbers of
    ``config.PATCHES``), ``reference`` the patch the champions are compared with (default 15.20, decision D18).

    수식
    ----
    (STR-1) as in the module text: unseen first, then changed, then unchanged.

    해석
    ----
    서명은 (패치, 챔피언)마다 한 번만 읽고(표 캐시), 행은 서명의 비교 결과를 열 방향으로 모아 정한다. 표가 없는 패치는 ValueError이다(패치 이름 하나만 말한다).
    """
    if not isinstance(reference, str):
        raise TypeError(f'reference must be a patch name such as "15.20", not {type(reference).__name__}')
    champs = np.asarray(match_champ)
    if champs.ndim != 2 or champs.shape[1] != 10:
        raise ValueError(f'match_champ must have shape (n, 10), one row of ten champion keys for each match, not {champs.shape}')
    if champs.dtype.kind not in 'iu':
        raise TypeError(f'match_champ must hold whole champion keys, not {champs.dtype}')
    champs = champs.astype(np.int64)
    names = _patch_names(match_patch, len(champs))
    unseen, any_changed = np.zeros(len(champs), dtype=bool), np.zeros(len(champs), dtype=bool)
    for patch in np.unique(names):
        rows = np.flatnonzero(names == patch)
        keys = np.unique(champs[rows])
        flags = np.zeros((len(keys), 2), dtype=bool)                                   # per key: (unseen, changed)
        for j, key in enumerate(keys.tolist()):
            own = _signature(str(patch), key, 'match_patch')
            ref = _signature(reference, key, 'reference')
            flags[j] = (True, False) if own is None or ref is None else (False, own != ref)
        at = np.searchsorted(keys, champs[rows])
        unseen[rows], any_changed[rows] = flags[at, 0].any(axis=1), flags[at, 1].any(axis=1)
    return ChampionStrata(changed=~unseen & any_changed, unchanged=~unseen & ~any_changed, unseen=unseen)


def changed_champion_rows(match_champ, match_patch, reference: str = REFERENCE_PATCH) -> np.ndarray:
    """(n,) bool: the rows with at least one champion whose trait signature in its patch differs from ``reference`` (decision D18), no champion of the row being unseen (those rows
    are ``unseen_champion_rows`` and in neither this stratum nor its complement)."""
    return champion_strata(match_champ, match_patch, reference).changed


def unseen_champion_rows(match_champ, match_patch, reference: str = REFERENCE_PATCH) -> np.ndarray:
    """(n,) bool: the rows with a champion that has no trait signature in its own patch or in ``reference`` (the new champions of 16.13, an empty slot, a mode id): decision D18 reports them on their own."""
    return champion_strata(match_champ, match_patch, reference).unseen


def _whole_ms(name: str, values) -> np.ndarray:
    a = np.asarray(values)
    if a.ndim != 1:
        raise ValueError(f'{name} must be a 1-d array, not shape {a.shape}')
    if a.dtype.kind not in 'iu':
        raise TypeError(f'{name} must hold whole numbers (ms or row numbers), not {a.dtype}')
    return a.astype(np.int64)


def pre_t0_objective_rows(t0_ms, onset_row, onset_unit, unit_tau_ms, unit_f_ms) -> np.ndarray:
    """(len(t0_ms),) bool: the frame rows with an onset (``onset_row[k]``, ``onset_unit[k]``) whose unit has tau <= t0 < f, in whole ms (B17a, equation (STR-2)).

    ``t0_ms`` is the reference time of every frame row, ``onset_row`` and ``onset_unit`` the row and the unit of every onset (``onsets/row``, ``onsets/unit``), ``unit_tau_ms`` and ``unit_f_ms`` the
    start anchor and the first-kill time of every unit (``units/tau``, ``units/f``). ValueError for a row or a unit number the tables do not have, for arrays of other lengths and for a
    time that is not a whole number of ms (TypeError).

    해석: 그 행의 정답에는 발생의 귀속 창 [τ, δ)가 들어 있으므로 τ ≤ t₀ < f이면 [τ, t₀]의 오브젝트도 정답에 들어 있고, 그 오브젝트는 t₀ 입력에도 이미 있다.
    """
    t0 = _whole_ms('t0_ms', t0_ms)
    tau, f = _whole_ms('unit_tau_ms', unit_tau_ms), _whole_ms('unit_f_ms', unit_f_ms)
    row, unit = _whole_ms('onset_row', onset_row), _whole_ms('onset_unit', onset_unit)
    if len(row) != len(unit):
        raise ValueError(f'onset_row and onset_unit must have one length (one entry for each onset), not {len(row)} and {len(unit)}')
    if len(tau) != len(f):
        raise ValueError(f'unit_f_ms has {len(f)} entries but unit_tau_ms has {len(tau)}: one of each for every unit')
    if row.size and (row.min() < 0 or row.max() >= len(t0)):
        raise ValueError(f'onset_row must lie in 0 .. {len(t0) - 1}, the rows of t0_ms')
    if unit.size and (unit.min() < 0 or unit.max() >= len(tau)):
        raise ValueError(f'onset_unit must lie in 0 .. {len(tau) - 1}, the units of unit_tau_ms')
    hit = (tau[unit] <= t0[row]) & (t0[row] < f[unit])
    out = np.zeros(len(t0), dtype=bool)
    out[row[hit]] = True
    return out


# ---- the sensitivity "all flow prices" (decision D19) ---------------------------------------------------------------------------------------
def flow_composer(primary, flow_prices, s2, heads):
    """The second ``Composer`` of the sensitivity "all flow prices" (decision D19): the composer ``primary`` with the price table ``flow_prices`` (``value.load_flow_prices``) and nothing else changed.

    It has the seed, the number of simulations, the rows, the tier law, the gold representatives and the settings (kill law, zone mark, exposure) of ``primary``, so every random draw of a row is the
    draw of the primary composer and only the value of an onset moves (``Composer._draw_row`` draws before it prices). ``heads`` must be a chain of heads of its own, the same model as the primary
    chain: a chain is bound to the rows of one composer, and binding the primary chain again would invalidate the primary composer (ValueError for the same chain). ``s2`` is the S2 store the
    chain is bound to. The truth that the sensitivity compares with is made in memory by ``value.reprice_units``; no label store is made.

    ``heads`` must also have the ``chunk_views`` of the primary chain (S4b preparation M14 review I1; ``EPTComposerHeads.chunk_views``, None for one pass): the chunk size moves the head outputs at the last digits
    of float32 and a Composer compares uniforms with cumulative probabilities, so another size could flip a draw (ValueError, before the chain is bound). Heads without the attribute (B0, B1) count as None.

    A1 is not retrained for the sensitivity and does not take part in it (decision D19).

    해석: 가격은 표본을 뽑은 뒤에만 읽히므로 같은 시드의 두 합성기는 같은 표본을 서로 다른 가격으로 매긴다. 그래서 짝지은 차이가 표본 잡음 없이 가격 정의의 효과만 담는다.
    """
    if not isinstance(primary, compose.Composer):
        raise TypeError('primary must be a compose.Composer')
    if heads is primary.heads:
        raise ValueError('heads must be a chain of heads of its own, not the chain of the primary composer: binding it again would invalidate the primary composer')
    if getattr(heads, 'chunk_views', None) != getattr(primary.heads, 'chunk_views', None):
        raise ValueError('the chain of heads and the chain of the primary composer must have the same chunk_views: another chunk size moves the head outputs at the last digits of float32 and can flip a draw, '
                         'which would put sampling noise into the paired difference of the two price tables')
    return compose.Composer(heads, flow_prices, primary.tier_model, primary.gold_reps, primary.rows, s2, S=primary.S, seed=primary.seed,
                            kill_law=primary.kill_law, zone_mark=primary.zone_mark, exposure=primary.exposure)


def _paired(name_a: str, a, name_b: str, b) -> tuple:
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if x.ndim != 1 or x.shape != y.shape:
        raise ValueError(f'{name_a} and {name_b} must be 1-d arrays of one length, not shapes {x.shape} and {y.shape}')
    keep = ~np.isnan(x) & ~np.isnan(y)
    return x[keep], y[keep]


def winner_sign_agreement(f0_primary, f0_flow) -> float:
    """The share of the units whose winner sign is the same under the two price tables (decision D19): the sign of f0 rounded to a millionth of a gold (``value.winner_of``: +1, 0 a draw, -1), over
    the units that are not NaN in either table. NaN when there is no such unit.

    수식
    ----
    A = #{u : W(f₀_u) = W(f₀′_u)} / #{u}      W = sign(round(f₀, 6)) ∈ {−1, 0, +1}
    """
    x, y = _paired('f0_primary', f0_primary, 'f0_flow', f0_flow)
    if not len(x):
        return float('nan')
    return float(np.mean(value.winner_of(x)[0] == value.winner_of(y)[0]))


def _average_ranks(x: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    last = np.cumsum(counts)
    return ((last - counts + 1 + last) / 2.0)[inverse]


def v_start_rank_correlation(v_primary, v_flow) -> float:
    """The Spearman rank correlation of the realised V_start under the two price tables (decision D19), over the rows that are not NaN in either; ties take their average rank. NaN for fewer than two
    rows and when either side is constant.

    수식
    ----
    ρ = Σ (R_i − R̄)(R′_i − R̄′) / √[ Σ (R_i − R̄)² · Σ (R′_i − R̄′)² ]      R, R′ = V_start의 평균 순위 (같은 값은 순위의 평균)
    """
    x, y = _paired('v_primary', v_primary, 'v_flow', v_flow)
    if len(x) < 2:
        return float('nan')
    rx, ry = _average_ranks(x), _average_ranks(y)
    dx, dy = rx - rx.mean(), ry - ry.mean()
    denominator = np.sqrt((dx * dx).sum() * (dy * dy).sum())
    return float((dx * dy).sum() / denominator) if denominator > 0 else float('nan')
