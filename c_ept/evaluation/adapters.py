"""Inference-only scientific adapters. Call only after the operator's terminal guard.

The caller owns role-scoped stores, best-only verified model loading, frozen FIT
stats/scales, identity and resource guards. This module never opens an artifact.

Roles (S4b preparation K6d, review I3 of K6b). ``run_inference`` evaluates the rows of one role, ``role``: 'DEV' (the default: every earlier call) or 'STOP' (the whole STOP split, the data
of the S4b decisions, decision D13). The role 'TEST' is possible only with the ``HoldoutOpening`` that ``run.evaluate.HoldoutGate.open`` returns and that no other code can make; every other role
(FIT, CAL, AUDIT, EXTERNAL, ...) is refused before any store is read. The row views of the stores refuse a holdout role as well (``flowprice._role_codes``), so even with an opening the role TEST
stays closed until a later task gives the stores a TEST reader: the opening is one lock of two.

해석: 추론은 한 역할의 행만 읽는다. DEV는 탐색용(기본값), STOP은 S4b 판정의 자료(결정 D13)이고, TEST는 관문이 한 번 열 때 주는 증표가 있어야만 요청할 수 있다.
"""
import sys
from contextlib import contextmanager

import numpy as np
import torch

from c_ept import config as C
from c_ept.data import head_rows
from c_ept.ept import data, loss
from c_ept.ept.heads import EPT, FrameAveraged
from c_ept.models import baselines_b1 as B1, killlaw, nll as nll_math
from c_ept.preprocessing import features
from c_ept.valuation import compose, towertier, value
from .endpoints import HORIZONS, endpoint, m2_endpoints, m4_endpoints, primary_rows

# G9b structural estimand: M2 whole layer plus M3/M4 open views. The five
# masked M3/M4 training blocks remain diagnostics and never enter this mean.
STRUCTURAL_HEADS = ('M2Rate/masked', 'M2Mark/masked', 'M2Zone/masked',
                    'M3Size/open', 'M3Kills/open', 'M3Gold/open', 'M3Obj/open', 'M4Conv/open')


def _numpy(t):
    return t.detach().cpu().numpy()


@contextmanager
def _evaluation(model):
    modes = [(m, m.training) for m in model.modules()]
    try:
        model.eval()
        with torch.inference_mode():
            yield
    finally:
        for m, mode in modes:
            m.training = mode


def _law(name, a, shape, probability=False):
    return compose._checked(name, a, shape, unit_sum=probability)


class B1OpenHeads:
    """Open event-view heads; there is deliberately no composer fallback."""
    def __init__(self, b1):
        missing = set(B1.OPEN_HEAD_NAMES) - set(b1.open)
        if missing:
            raise ValueError(f'missing B1 open heads: {sorted(missing)}')
        self.heads = b1.open

    def laws(self, numeric, *, check, d=None):
        """The size law, the kill rates (n, 4, 2) and the gold laws of the open units ``numeric`` (n, columns). ``d`` (n,) is the first-kill team of each unit (0 blue, 1 red,
        −1 an execution): an open kill head of the shifted law (decision D5: its design has the slot ``d``) needs it and is given it as the category ``d``; a head without
        that slot is called as before and ``d`` is not read. ValueError when the head has the slot and ``d`` is None."""
        n = len(numeric)
        check()
        size = _law('B1 open size', self.heads['M3Size'].predict_blocks(numeric), (n, 4), True)
        kill = np.empty((n, 4, 2))
        gold = np.empty((n, 4, 2, 6, 31))
        layout = getattr(self.heads['M3Kills'], 'layout_', None)
        wants_first_team = layout is not None and B1.FIRST_TEAM in layout.slots
        if wants_first_team and d is None:
            raise ValueError('the open B1 kill head of the shifted kill law needs the first-kill team d of every unit')
        for s in range(4):
            check()
            ss = np.full(n, s, dtype=np.int64)
            kill[:, s] = self.heads['M3Kills'].predict_blocks(numeric, s=ss, **({B1.FIRST_TEAM: d} if wants_first_team else {}))
            for t in (0, 1):
                for layer in range(1, 7):
                    check()
                    gold[:, s, t, layer - 1] = self.heads['M3Gold'].predict_blocks(
                        numeric, np.full(n, t, dtype=np.int64), s=ss,
                        layer=np.full(n, layer, dtype=np.int64))
        return size, _law('B1 open kills', kill, (n, 4, 2)), _law('B1 open gold', gold, (n, 4, 2, 6, 31), True)

    def objects(self, numeric, size, generated_K):
        k = np.asarray(generated_K, dtype=np.float64).copy()
        k[:, 2:] /= 1000.0
        x = np.column_stack((numeric, k)).astype(np.float32)
        return _law('B1 open objects', self.heads['M3Obj'].predict_blocks(x, s=size), (len(size), 18))


def _open_draws(laws, objects, prices, gold_reps, *, row, match, draws, seed, check, batch_size, d=None, kill_law='poisson'):
    """The Monte Carlo draws of one unit for one model: ``(rates (draws, 18), O (draws, 18), f (draws,))``.

    ``batch_size`` is the number of draws per ``objects`` call, a positive integer, or None for every draw of the row in one call.
    ``draws`` is an integer of at least 2 (ValueError otherwise), the rule of ``run_inference``: the Monte Carlo standard errors of
    ``sample_open_value`` divide by ``draws - 1``, and with ``batch_size=None`` the chunk is ``draws``, so ``draws = 0`` would be a ``range`` step of 0.
    ``check()`` is called once, before the first ``objects`` call. Chunking decides only how the model is
    called, never which random numbers are drawn: all draws of the generator happen before the calls (s, K) or after them (O).

    ``kill_law`` and ``d`` (S4b preparation M1, decision D5): with the default ``'poisson'`` the kills of both teams are Poisson counts of the rates, as in every earlier
    run, and ``d`` is not read. With ``'shifted'`` ``d`` is the first-kill team of the unit (0 blue, 1 red, −1 an execution start: the plain law; ValueError if it is None
    or another number): its entry of the rates is the mean λ_d of the excess N_d and its count is 1 + N_d (``killlaw.sample``: the same single Poisson call, then 1 added,
    so the random stream is that of the plain law). The kill rates must come from heads of the same law.
    """
    if batch_size is not None and (isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1):
        raise ValueError('batch_size must be a positive integer, or None for every draw of the row in one call')
    if isinstance(draws, bool) or not isinstance(draws, (int, np.integer)) or draws < 2:
        raise ValueError('draws must be an integer >= 2')
    killlaw.check_law(kill_law)
    if kill_law == 'shifted' and (d is None or isinstance(d, (bool, np.bool_)) or not isinstance(d, (int, np.integer)) or not -1 <= d <= 1):
        raise ValueError(f'the shifted kill law needs the first-kill team d of the unit (0 blue, 1 red, -1 an execution), not {d!r}')
    size, kills, gold = laws
    size = _law('size', size, (4,), True)
    kills = _law('kills', kills, (4, 2))
    gold = _law('gold', gold, (4, 2, 6, 31), True)
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(match), int(row)]))
    s = compose._inverse_cdf(np.broadcast_to(size, (draws, 4)), rng.random(draws))
    k = killlaw.sample(rng, kills[s], np.full(draws, d, dtype=np.int64) if kill_law == 'shifted' else None, kill_law)
    layer = np.clip(k, 1, 6) - 1
    u = rng.random((draws, 2))
    bins = np.column_stack([compose._inverse_cdf(gold[s, t, layer[:, t]], u[:, t]) for t in (0, 1)])
    g = np.where(k > 0, gold_reps[bins], 0.0)
    K = np.column_stack((k, g))
    rates = np.empty((draws, 18))
    chunk = draws if batch_size is None else batch_size
    check()
    for lo in range(0, draws, chunk):
        hi = min(lo + chunk, draws)
        rates[lo:hi] = objects(s[lo:hi], K[lo:hi])
    rates = _law('objects', rates, (draws, 18))
    O = rng.poisson(rates)
    f = g[:, 0] - g[:, 1] + (O * prices).sum(axis=1)
    return rates, O, f


def sample_open_value(laws, objects, prices, gold_reps, *, row, match, draws, seed, check, batch_size, d=None, kill_law='poisson'):
    """Sample the same size -> kills -> gold -> objects law as Composer (no conversion).

    objects receives ONLY generated K. RNG is stable per (seed, match, unit),
    independent of endpoint ordering and batching. Paired models share initial
    streams; Poisson rejection sampling need not consume equal random numbers.
    batch_size: draws per objects call (None: every draw of the row in one call). A model whose rows do
    not depend on each other (B1) takes None; the neural model keeps the run's batch_size because the
    shape of a GPU call can change the last bits. The result does not depend on it for such a model.
    kill_law, d: the kill law of the heads and, for the shifted law (decision D5), the first-kill team of the unit; see ``_open_draws``.
    """
    *_, f = _open_draws(laws, objects, prices, gold_reps, row=row, match=match, draws=draws, seed=seed,
                        check=check, batch_size=batch_size, d=d, kill_law=kill_law)
    plus, draw = float(np.mean(f > 0)), float(np.mean(f == 0))
    # Unbiased sample-variance estimate of the MC mean's SE; zero at boundaries.
    se = np.sqrt(plus * (1 - plus) / (draws - 1))
    draw_se = np.sqrt(draw * (1 - draw) / (draws - 1))
    return plus, draw, float(se), float(draw_se)


def open_cell_values(bundle, match, unit):
    """Fixed flow prices and the existing tower-tier expectation at M3 time f."""
    prices = compose._prices(bundle.prices)
    when = np.asarray(bundle.s1.array('units/f'))[unit].astype(np.int64)
    destroyed = towertier.destroyed_before(bundle.s1, match, when)
    out = np.empty((len(unit), 18), dtype=np.float64)
    for kind, name in enumerate(C.O_KINDS):
        for team in (0, 1):
            if name == 'tower':
                taker = np.full(len(unit), team, dtype=np.int64)
                victim = towertier.victim_counts(destroyed, taker)
                price = towertier.expected_tower_price(bundle.tier.prob(victim, taker, when / 1000.), prices)
            else:
                price = prices[name]
            out[:, 2 * kind + team] = price if team == 0 else -price
    if not np.isfinite(out).all():
        raise ValueError('nonfinite open value prices')
    return out


def _output(fa, enc, head, b):
    pos = b['pos']
    q, m, z = (b.get(k) for k in ('q', 'm', 'z'))
    if head == 'M2Rate':
        return fa.m2_rate(enc, b['e'], pos=pos)
    if head in ('M2Mark', 'M2Zone'):
        m2 = fa.m2(enc, 1.0, pos=pos)
        return m2.mark_logp if head == 'M2Mark' else m2.zone_logp
    if head in ('M3Size', 'M3Kills', 'M3Gold'):
        m3 = fa.m3(enc, q, m, z, pos=pos)
        return {'M3Size': m3.size_logp, 'M3Kills': m3.kill_rate, 'M3Gold': m3.gold_logp}[head]
    if head == 'M3Obj':
        return fa.m3_obj(enc, q, m, z, b['s'], b['K'], pos=pos)
    return fa.m4(enc, q, m, z, b['K'], b['O'], b['e'], pos=pos)


def _b1_predict(head, kind, fitted, parts, columns, idx, kill_law='poisson'):
    """Use the canonical training design including thousand-gold and zone coding. ``kill_law``: the open M3Kills design of the shifted law has the slot ``d``
    (``B1.make_layout``); every other design, and every design of the plain law, is ``B1.SLOTS``."""
    p = parts
    if head == 'M4Conv' and kind == 'masked':
        X = B1._m4_design(p.m4[idx], len(columns))
    else:
        take = idx if p.take is None else p.take[idx]
        slots = B1.SLOTS[(head, kind)] if kill_law == 'poisson' else B1.make_layout(head, kind, columns, kill_law).slots
        X = B1._assemble(slots, p.feat, take,
                         {k: v[idx] for k, v in p.codes.items()},
                         None if p.K is None else p.K[idx], None if p.O is None else p.O[idx])
    if head == 'M3Gold':
        return fitted.predict(X, p.team[idx])
    return fitted.predict(X)


# ---- B1 on both team frames (S4b preparation M13, decision B22) --------------------------------------------------------------------------------------------
# The EPT evaluation is the average of the model over the recorded frame and its team swap (``heads.FrameAveraged``, equation (EH-9)); B1 gets the same at test time as an S7 option, so that the comparison is not
# one of an exactly team-symmetric model against one that is not. B1 predicts from the features of the swapped views and the swapped conditions, the output goes back to the recorded frame by the table of the head,
# and the two answers are averaged.
#
#   (B22-1)  p̄(x, c) = ½ [ p(x, c) + Π_h p(σx, σc) ]         Π_h = MARK_PERM (M2Mark), ZONE_PERM (M2Zone), TEAM_PERM (M3Kills), CELL_PERM (M3Obj, M4Conv), the identity for the other heads
#            σc: m ↦ MARK_PERM[m],  z ↦ ZONE_PERM[z] (−1 stays),  K ↦ K[K_PERM],  O ↦ O[CELL_PERM],  gold entry team ↦ 1 − team,  d ↦ 1 − d (−1 stays);  q, s and the layer stay
#
#      기호      | 뜻                                                              | 단위 | 범위          | 코드 변수
#      p(x, c)   | 머리의 예측 (확률 또는 율), 특징 x와 조건 c에서                   | —    | 머리마다      | ``_b1_predict`` 의 반환값
#      σx        | 팀을 맞바꾼 보기의 B1 특징 (``features.b1_swapped``)              | 열마다 | 유한 또는 NaN | 맞바꾼 행의 ``feat``
#      σc        | 맞바꾼 틀의 조건 (표지, 구역, K, O, 골드 항목의 팀, 첫 킬 팀)     | —    | 위와 같음     | ``swap_conditions``
#      Π_h       | 맞바꾼 틀의 출력 축을 기록한 틀로 되돌리는 순열 (자기 역)          | —    | —             | ``back_permutation``
#
# 해석: 팀 교환은 두 번 쓰면 제자리이므로, 맞바꾼 틀의 예측을 되돌려 원래 틀의 예측과 평균하면 머리가 팀 교환에 불변이 아니어도 평균은 어느 팀을 파랑이라 부르는지와 무관하다(같은 수 두 개의 합은 순서에
# 상관없어 비트까지 같다). 점수(NLL)는 기록한 틀의 목표와 노출로 평균한 예측을 채점한다. 평균은 확률과 율의 산술평균이며(EPT의 (EH-9)와 같다), 열린 머리의 몬테카를로는 평균한 법칙과 평균한 오브젝트
# 율로 표본을 뽑는다(난수는 그대로이다).
# Ref: Puny et al. (2022), "Frame Averaging for Invariant and Equivariant Network Design", ICLR (the average of a model over the elements of a group, here the two frames, is exactly invariant).
_BACK_AXIS = {'mark': data.MARK_PERM, 'zone': data.ZONE_PERM, 'teams': data.TEAM_PERM, 'cells': data.CELL_PERM}


def swap_first_team(d) -> np.ndarray:
    """The first-kill team d (0 blue, 1 red, -1 an execution: no first team) seen from the other side: ``1 - d``, and -1 stays (the rule of ``data.OPTIONAL_FIELD_SWAP``)."""
    d = np.asarray(d, dtype=np.int64)
    return np.where(d >= 0, 1 - d, d)


def back_permutation(head, kind):
    """The permutation Π_h of the last axis of a prediction of ``(head, kind)`` that moves the answer for the team-swapped frame back to the recorded frame (equation (B22-1)), or None for a head whose
    answer has no team-dependent axis (the rate and the size law; the gold law of a team entry is moved by the swap of the team of the entry). It is the swap that ``data.FIELD_SWAP`` applies to the target ``y`` of the head:
    the 52 marks, the 10 zones, the two teams of the kills, the 18 cells of the objects and the conversions. ValueError for a head and kind that do not exist."""
    if (head, kind, 'y') not in data.FIELD_SWAP:
        raise ValueError(f'there is no {kind} head {head!r}')
    change = data.FIELD_SWAP[(head, kind, 'y')]
    return None if change is None else _BACK_AXIS[change]


def swap_conditions(parts, head, kind):
    """The ``HeadParts`` of a B1 head with every team-dependent input changed to the other side (equation (B22-1), σc): the mark ``m`` and the zone ``z`` of the conditions, the first-kill team ``d``, the two teams of the K and
    O blocks (also the K and O columns of the masked M4 matrix, whose categories q, m, z are kept in step with its columns), the team of a gold entry and ``first_team``. The tables are those of ``data`` (the rule engine of the EPT
    batches). The features ``feat`` (``features.b1_swapped`` makes the swapped ones), the targets, the exposures, the match and row numbers and q, s and the layer are the same objects. The input is not changed; applied twice, the rows come back bit for bit.

    해석: 같은 경기를 반대편에서 본 행의 조건이다. 특징은 따로 맞바꾼 보기에서 만들고(``features.b1_swapped``), 정답과 노출은 기록한 틀의 것을 쓴다(점수는 기록한 틀에서 매긴다)."""
    if (head, kind, 'y') not in data.FIELD_SWAP:
        raise ValueError(f'there is no {kind} head {head!r}')
    codes = dict(parts.codes)
    if 'm' in codes:
        codes['m'] = data.MARK_PERM[np.asarray(codes['m'], dtype=np.int64)]
    if 'z' in codes:
        z = np.asarray(codes['z'], dtype=np.int64)
        codes['z'] = np.where(z >= 0, data.ZONE_PERM[np.maximum(z, 0)], z)
    if 'd' in codes:
        codes['d'] = swap_first_team(codes['d'])
    m4 = parts.m4
    if m4 is not None:
        d = parts.feat.shape[1]
        if m4.ndim != 2 or m4.shape[1] != d + len(features.M4_EXTRA_NAMES) + head_rows.N_CELLS or not {'m', 'z'} <= set(codes):
            raise ValueError('the masked M4 matrix must be [feat | q m z | K | O] with the categories m and z among the codes')
        m4 = np.array(m4)
        m4[:, d + 1], m4[:, d + 2] = codes['m'], codes['z']                      # the categories q, m, z are the columns d, d + 1, d + 2
        m4[:, d + 3:d + 7] = m4[:, d + 3:d + 7][:, data.K_PERM]
        m4[:, d + 7:] = m4[:, d + 7:][:, data.CELL_PERM]
    K = None if parts.K is None else np.asarray(parts.K)[:, data.K_PERM]
    O = None if parts.O is None else np.asarray(parts.O)[:, data.CELL_PERM]
    team = None if parts.team is None else 1 - np.asarray(parts.team)
    first_team = None if parts.first_team is None else swap_first_team(parts.first_team)
    return parts._replace(codes=codes, K=K, O=O, m4=m4, team=team, first_team=first_team)


def b1_predict_averaged(head, kind, fitted, parts, parts_swapped, columns, idx, kill_law='poisson'):
    """The B1 prediction of the rows ``idx`` of a head averaged over the recorded frame (``parts``) and the team-swapped frame (``parts_swapped``: the features of the swapped views and ``swap_conditions`` of the same
    rows), the swapped answer moved back by ``back_permutation`` (equation (B22-1)). The arguments after ``fitted`` are those of ``_b1_predict``; the rows of the two parts are the same rows in the same order.

    해석: 기록한 틀의 예측과 맞바꾼 틀의 예측을 되돌린 것의 산술평균이다. 두 수의 합은 순서에 매이지 않으므로 파랑과 빨강의 이름을 바꿔도 평균은 되돌린 축까지 같은 비트이다."""
    p = _b1_predict(head, kind, fitted, parts, columns, idx, kill_law)
    q = _b1_predict(head, kind, fitted, parts_swapped, columns, idx, kill_law)
    perm = back_permutation(head, kind)
    return 0.5 * (p + (q if perm is None else q[..., perm]))


class B1SwapAveragedOpenHeads:
    """``B1OpenHeads`` on both team frames (equation (B22-1)): the open M3 laws (size, the kills of the two teams by size, the gold law of each team by size and layer) and the object rates, each averaged over the recorded
    frame and the swapped one; the team axes of the swapped answer go back by ``TEAM_PERM`` and the cells by ``CELL_PERM``, and the K that the objects condition on is swapped by ``K_PERM`` for the swapped frame. The
    first-kill team d of the shifted kill law is swapped with the frame. ``numeric_swapped`` are the swapped feature rows of the same units (``features.b1_swapped``). Like ``B1OpenHeads`` there is no composer fallback."""

    def __init__(self, open_heads: B1OpenHeads):
        self.open_heads = open_heads

    def laws(self, numeric, numeric_swapped, *, check, d=None):
        """``(size (n, 4), kill (n, 4, 2), gold (n, 4, 2, 6, 31))`` averaged over the two frames; ``d`` as for ``B1OpenHeads.laws``."""
        a = self.open_heads.laws(numeric, check=check, **({} if d is None else {'d': d}))
        b = self.open_heads.laws(numeric_swapped, check=check, **({} if d is None else {'d': swap_first_team(d)}))
        return 0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1][..., data.TEAM_PERM]), 0.5 * (a[2] + b[2][:, :, data.TEAM_PERM])

    def objects(self, numeric, numeric_swapped, size, generated_K):
        """(n, 18) the object rates given the size and the generated K, averaged over the two frames."""
        K = np.asarray(generated_K, dtype=np.float64)
        a = self.open_heads.objects(numeric, size, K)
        b = self.open_heads.objects(numeric_swapped, size, K[:, data.K_PERM])
        return 0.5 * (a + b[:, data.CELL_PERM])


def _aggregate(match, row, scores):
    """HR-5: sum bins/teams within an evaluation row; never silently inner-join."""
    ids, first, inv = np.unique(row, return_index=True, return_inverse=True)
    if not np.array_equal(match, match[first][inv]):
        raise ValueError('one evaluation row has multiple matches')
    if not np.isfinite(scores).all():
        raise ValueError('nonfinite NLL')
    return match[first], ids, np.bincount(inv, weights=scores, minlength=len(ids))


def _nll_record(head, kind, match, row, ept, b1, counts, scales, b0=None):
    """Called only after paired target/condition/scored-row equality is proven. ``b0``: the HR-5 NLL of B0 on the same rows (S4b preparation M11, decision D16), the field ``nll_b0``;
    None (a run without B0) leaves the record as it was, with no such key."""
    key = loss.block_key(head, kind)
    record = dict(name=key, match=match, row=row, nll_ept=ept, nll_b1=b1, counts=counts,
                  common_head=scales is not None and key in STRUCTURAL_HEADS,
                  common_head_basis='G9b M2 whole layer plus M3/M4 open only; paired canonical targets/conditions/views/scored rows, HR-5 sums')
    if scales is not None:
        record.update(scale=float(scales[key]), nll_ept_normalized=ept / scales[key], nll_b1_normalized=b1 / scales[key])
    if b0 is not None:
        record['nll_b0'] = b0
    return record


def _empty_block(head, kind, reason, excluded, scales, nll_rows, endpoints, with_b0=False):
    ids, values = np.empty(0, dtype=np.int64), np.empty(0, dtype=float)
    nll_rows.append(_nll_record(head, kind, ids, ids.copy(), values, values.copy(),
                               {'eligible': 0, 'unavailable_reason': reason,
                                'excluded_zero_exposure_cells_or_units': excluded}, scales, values.copy() if with_b0 else None))
    if head == 'M2Rate':
        zero = np.empty((0, 18))
        out = m2_endpoints(ids, ids, zero, zero, zero, zero)
        for ep in out:
            ep['counts'].update(candidate=excluded, zero_exposure=excluded, unavailable_reason=reason)
        endpoints.extend(out)
    elif (head, kind) == ('M4Conv', 'open'):
        zero = np.empty((0, 18))
        out = m4_endpoints(ids, ids, zero, values, zero, zero, excluded_no_window=excluded)
        for ep in out:
            ep['counts']['unavailable_reason'] = reason
        endpoints.extend(out)


# Provenance wording of the T003 path (existing whole-FIT B1 versus the FIT/STOP 25% EPT pilot), recorded in the
# 'comparison' of the provenance of run_inference. The matched-25% comparison passes its own text
# (comparison.protocol.matched_text) instead. The summary label of the same path is metrics.T003_SUMMARY_COMPARISON.
T003_PROVENANCE_COMPARISON = 'whole-FIT B1 versus FIT/STOP-25%-EPT pilot; no generalization or controlled architecture claim'


GOLD_REP_SOURCES = ('labels', 'in_scope_fit')    # where the representatives of the gold bins come from: labels/gold_reps (every earlier run) or the in-scope FIT means (decision D21 B16)
EXPOSURES = compose.EXPOSURES                    # the exposure of the t0 forecasts (M2 occurrence probabilities): the observed one of every earlier run or the nominal one of a T-blind forecast (decision D14)
POPULATIONS = ('all', 'primary', 'secondary')    # the rows of the t0 forecasts: every row (every earlier run), the primary rows (t0 + H <= T - 30 s) or the secondary ones, the rest (decision D14)


def check_exposure_population(exposure, population) -> None:
    """ValueError naming ``exposure`` or ``population`` unless they are among ``EXPOSURES`` and ``POPULATIONS`` (a short string is shown, any other value by its type only), and ValueError when the nominal
    exposure is asked for another population than the primary one (S4b preparation M12, decision D14).

    해석: 공칭 노출(10 s)은 지평 H가 경기 끝 컷 안에 온전히 드는 행(주 모집단)에서만 관측 노출과 같다. 다른 행에서는 창이 T로 잘려 있으므로 공칭 노출을 쓰려면 T를 읽어 창을 새로 정해야 하고,
    그것은 t₀ 예측이 T를 읽는 길이어서 거부한다."""
    for name, val, allowed in (('exposure', exposure, EXPOSURES), ('population', population, POPULATIONS)):
        if not isinstance(val, str) or val not in allowed:
            shown = repr(val[:40]) if isinstance(val, str) else type(val).__name__
            raise ValueError(f'{name} must be one of {allowed}, not {shown}')
    if exposure == 'nominal' and population != 'primary':
        raise ValueError(f"exposure 'nominal' needs population 'primary': the nominal exposure is the observed one only on the rows whose horizon the game end does not cut, and on any other row it would have "
                         f'to be read from T; population {population!r} is refused')


def _primary_masks(s1, frame_row) -> dict:
    """``{H: bool mask}`` for the horizons 60, 120, 180 s: which of the frame rows ``frame_row`` (the global frame-row numbers of the rows of the role) are primary (``primary_rows``). This is the one place of the evaluation
    where the end of the game T enters, and it enters as the window end ``frame_rows/hi`` = min(t0 + 180 s, T - 30 s) in whole ms (decision D14: T only in the training censoring and in the population filter).
    The two whole arrays are indexed at the given rows of the role only.

    해석: 주 모집단의 선별에만 T(를 담은 관측 창의 끝 hi)를 쓴다. 정수 ms로 읽어 경계 행이 흔들리지 않는다."""
    row = np.asarray(frame_row, dtype=np.int64)
    t0, hi = np.asarray(s1.array('frame_rows/t0')), np.asarray(s1.array('frame_rows/hi'))
    if t0.ndim != 1 or t0.shape != hi.shape or (len(row) and (row.min() < 0 or row.max() >= len(t0))):
        raise ValueError('frame_rows/t0 and frame_rows/hi must be one-dimensional arrays of one length that hold every frame row of the rows')
    return {h: primary_rows(t0[row], hi[row], h) for h in HORIZONS}


def primary_masks(s1, frame_row) -> dict:
    """The public name of ``_primary_masks`` (S4b preparation K6e): ``{H: bool mask}`` of the primary rows among the given global frame rows, for H = 60, 120, 180 s.

    The composed V_start pass of the evaluation runner needs the very population filter of the t0 occurrence endpoints (decision D14: t0 + H <= T - 30 s, as ``hi - t0 >= H x 1000`` ms), and the runner
    is a strict module that may not import a private name. This module stays the one reader of the window end ``frame_rows/hi`` (``test_leakage``), so the runner asks for the masks here.

    해석: 합성 V_start의 주 모집단도 발생 끝점과 같은 필터(t₀ + H ≤ T − 30 s)여야 하므로 같은 함수를 부른다. 창의 끝을 읽는 곳은 이 모듈 하나로 남는다."""
    return _primary_masks(s1, frame_row)


def _nominal_exposure(observed, masks) -> np.ndarray:
    """The nominal exposure (10 s in every bin, the dtype of ``observed``) of the M2 occurrence endpoints (decision D14). Under the nominal exposure only the primary rows are scored, and on them the observed exposure of
    the first H/10 bins is already 10 s: that is checked here (ValueError otherwise, a store whose ``frame_rows/hi`` and ``frame_rows/exposure_ms`` disagree), so the nominal values are the observed ones on the rows used."""
    nominal = np.full_like(observed, C.BIN_MS / 1000.0)
    for h in HORIZONS:
        q = h * 1000 // C.BIN_MS
        if not np.array_equal(observed[masks[h], :q], nominal[masks[h], :q]):
            raise ValueError(f'the primary rows of the horizon {h} s do not all have the whole 10 s of exposure in the first {q} bins: frame_rows/hi and frame_rows/exposure_ms disagree')
    return nominal


RUN_ROLES = ('DEV', 'STOP')      # the roles that ``run_inference`` takes without further proof; 'TEST' needs the ``HoldoutOpening`` of the gate


class HoldoutOpening(dict):
    """The capability to evaluate the role TEST: the ``test_opened`` entry that ``run.evaluate.HoldoutGate.open`` wrote, as a read-only-by-convention dict (every use of the entry as a mapping works).

    Only ``HoldoutGate.open`` can make one: the constructor refuses every other caller (it looks at the function that calls it, ``c_ept.run.evaluate`` ``open``), so code that has no
    opening of the one TEST ledger cannot ask ``run_inference`` for the role TEST. This is a guard against use by mistake and a mark in the call chain, not a defence against code that
    forges frames: the lock that does not depend on it is the stores, which refuse a holdout role.

    해석: TEST를 평가하려면 관문이 TEST를 연 줄(원장의 test_opened)을 증표로 들고 와야 한다. 증표는 관문의 open 함수 안에서만 만들어진다."""

    def __init__(self, entry):
        caller = sys._getframe(1)
        if caller.f_globals.get('__name__') != 'c_ept.run.evaluate' or caller.f_code.co_name != 'open':
            raise TypeError('a HoldoutOpening is made by HoldoutGate.open only: nobody else can open the role TEST')
        if not isinstance(entry, dict) or entry.get('event') != 'test_opened':
            raise ValueError('a HoldoutOpening is the test_opened entry of the access ledger')
        super().__init__(entry)


def check_role(role, holdout) -> str:
    """``role`` itself when ``run_inference`` may evaluate it: 'DEV', 'STOP', or 'TEST' with the ``HoldoutOpening`` ``holdout``; ValueError otherwise (a short string is shown, any other value by
    its type only). A ``holdout`` for a role other than TEST is refused too, so an opening is never carried along unused.

    해석: DEV와 STOP은 증표 없이, TEST는 관문이 준 증표와 함께만 받는다. 그 밖의 역할(FIT, CAL, AUDIT 등)은 저장소를 읽기 전에 거부한다."""
    if not isinstance(role, str) or role not in (*RUN_ROLES, 'TEST'):
        shown = repr(role[:40]) if isinstance(role, str) else type(role).__name__
        raise ValueError(f"role must be 'DEV', 'STOP' or (with the opening of the gate) 'TEST', not {shown}")
    if role == 'TEST':
        if not isinstance(holdout, HoldoutOpening):
            raise ValueError("the role 'TEST' needs the HoldoutOpening that HoldoutGate.open returns: it is opened once, by the gate, and by nothing else")
    elif holdout is not None:
        raise ValueError(f'a holdout opening belongs to the role TEST only, not to {role!r}')
    return role


def check_gold_reps_source(source) -> str:
    """``source`` itself if it is one of ``GOLD_REP_SOURCES``; ValueError naming ``gold_reps`` otherwise (a short string is shown, any other value by its type only)."""
    if not isinstance(source, str) or source not in GOLD_REP_SOURCES:
        shown = repr(source[:40]) if isinstance(source, str) else type(source).__name__
        raise ValueError(f'gold_reps must be one of {GOLD_REP_SOURCES}, not {shown}')
    return source


def _check_b0(b0, kill_law, k_condition):
    """ValueError unless ``b0`` is a B0 set fitted for the kill law and the K condition of the run (S4b preparation M11, decisions D16 and D24). ``B0Heads.nll`` builds its rows with its own condition and cannot catch a
    set that is asked for another one, so it is checked here, before any store is read; the attribute absent means the plain law and the raw gold, as for B1.

    B0의 분모는 모형과 같은 행·같은 조건에서 채점돼야 하므로 킬 법칙과 K 조건이 다른 B0는 저장소를 읽기 전에 거부한다."""
    if not all(hasattr(b0, name) for name in ('fit_roles', 'masked', 'open')):
        raise ValueError('b0 must be a fitted B0 head set (B0Heads) with fit_roles, masked and open heads')
    if tuple(b0.fit_roles) != ('FIT',):
        raise ValueError('the B0 heads must be fitted on FIT only: the denominator of the relative NLL is the FIT refit evaluated on the rows of the role')
    if getattr(b0, 'kill_law', 'poisson') != kill_law:
        raise ValueError(f'the B0 heads were fitted for kill_law {getattr(b0, "kill_law", "poisson")!r} but the inference asks for {kill_law!r}')
    if getattr(b0, 'k_condition', 'raw') != k_condition:
        raise ValueError(f'the B0 heads were fitted for k_condition {getattr(b0, "k_condition", "raw")!r} but the inference asks for {k_condition!r}')
    for kind, names in (('masked', head_rows.HEAD_NAMES), ('open', head_rows.OPEN_HEAD_NAMES)):
        if set(getattr(b0, kind)) != set(names):
            raise ValueError(f'incomplete B0 {kind} heads')


class _BlockProgress:
    """The calls of the optional ``progress`` callback of ``run_inference``: ``progress('begin', block, {})`` when a head block starts and ``progress('end', block, counts)`` when it is done (also
    for an empty block), ``block`` the key of the loss block ('M2Rate/masked', ...). The counts are whole numbers: ``rows`` (the rows of the block that are scored), ``views`` (the views that the block
    encodes, summed over its batches: a view that two batches share counts twice) and, for the open M3 size block with a Monte Carlo, ``units`` (the units that are drawn for) and ``draws`` (per unit).
    With ``progress`` None every method does nothing. A block that is cut short by an exception has no end call (the caller knows from the exception).

    해석: 머리 블록 13개가 각각 언제 시작해 얼마나 걸렸는지 밖에서 볼 수 있게 하는 알림이다. 배열을 읽거나 바꾸지 않으므로 결과의 비트는 알림이 없을 때와 같다."""

    def __init__(self, progress):
        self.progress, self.open = progress, None

    def begin(self, block: str) -> None:
        self.end()
        if self.progress is not None:
            self.open = (block, {'rows': 0, 'views': 0})
            self.progress('begin', block, {})

    def count(self, **counts) -> None:
        if self.open is not None:
            self.open[1].update(counts)

    def add(self, **counts) -> None:
        if self.open is not None:
            for key, number in counts.items():
                self.open[1][key] = self.open[1].get(key, 0) + int(number)

    def end(self) -> None:
        if self.open is not None:
            block, counts = self.open
            self.open = None
            self.progress('end', block, dict(counts))


def run_inference(bundle, model, stats, b1, *, check, device='cuda', batch_size=32,
                  draws=1000, seed=20261022, scales=None, comparison=T003_PROVENANCE_COMPARISON, kill_law='poisson', winner_truth='gold',
                  k_condition='raw', gold_reps='labels', eval_average=True, zone_mark=True, b0=None, exposure='observed', population='all', b1_swap_average=False,
                  role='DEV', holdout=None, monte_carlo=True, progress=None):
    """Return paired endpoint arrays and HR-5 per-row NLLs of the rows of one role: DEV (the default), STOP (S4b preparation K6d), or TEST with the opening of the gate only.

    bundle: role-scoped s1/s2/labels stores, prices, tier (already loaded). It must carry
        an explicit ``role`` that equals the argument ``role``; a bundle without ``role`` is refused, not defaulted.
    role, holdout: the role whose rows are evaluated (``RUN_ROLES``: 'DEV', the default, which is every earlier call, or 'STOP', the whole STOP split on which decision D13 puts every S4b decision) and, for
        the role 'TEST' only, the ``HoldoutOpening`` that ``HoldoutGate.open`` returns (``check_role``). Every other role, and TEST without that opening, is a ValueError before any store is read. Every
        row selection of the function (the rows of the EPT, of B1 and its swapped frame and of B0, the masks of the frames, onsets and units) takes this role, and so does the 'role' of the provenance; the
        default 'DEV' reproduces every earlier result bit for bit. Each arm used the STOP rows that its own training run recorded for early stopping (the EPT its STOP rows, B1 the ``stop_rows`` of
        its ``fit_head``; a matched B1 stops on the same matched subset as the EPT, an unmatched ``B1Heads.fit`` on the whole STOP): this call cannot see either row set, ``B1Heads`` records the role
        only. So the numbers of the role STOP are the decision data of S4b (decision D13) and not a held-out estimate; the provenance says so (``role_note``, only for a role other than DEV).
    check: mandatory zero-argument terminal/authorization/resource guard.
    batch_size: rows per head call; the neural Monte Carlo of the open M3 value also takes this many draws per call,
        B1's takes every draw of a row in one call (the result does not depend on it for B1).
    scales: existing parent-run block -> positive B0 scale mapping, never fitted.
    comparison: provenance text of the design (label only; no estimate depends on it).
    kill_law: the kill law of the M3Kills heads of both models (S4b preparation M1, decision D5): 'poisson' (the default) is the law of every earlier run, 'shifted' makes
        the first-kill team's count 1 + N_d: the rows carry the first-kill team, the NLL of the M3Kills blocks and the open draws of the M3 value follow the law.
        Only the law of ``b1`` is checked here (``b1.kill_law`` must equal this argument: ValueError otherwise, before any store access). The law of the EPT is NOT
        checked here, because the model object carries no identity: the caller (the kernel, K6) must pass the law it reads from the training identity of the checkpoint
        (``identity.get('kill_law', 'poisson')``; absent means the plain law) and must refuse a mixture of checkpoints of different laws.
    winner_truth: the truth side of the M3 sign endpoint (``M3/open/positive_value``: y = f > 0 and draw_truth = f == 0), S4b preparation M2, decision D5 tie rule (i):
        'gold' (the default) is labels/f0_unit, the raw kill gold plus the objectives, as in every earlier run; 'binned' computes the gold term of f on the models' variable, the gold
        of a team read as the representative of its gold bin (``value.winner_truth``: labels/gold_bin and the same ``labels/gold_reps`` array that the open Monte Carlo reads), so that two
        teams of one bin tie on the gold term on both sides. The objective term of the 'binned' truth stays the realized attributed prices of labels/obj_value: for towers the realized
        tier price, whereas the models price a tower at the expected tier price (``open_cell_values``), so for units with towers the truth f and the model f differ even where the gold terms
        agree; a pair of one bin is a tie when the realized objective values of the two teams are equal (whether to price the realized counts at the models' cell prices is an open
        question for the author). Only the truth of that one endpoint changes; no prediction, NLL or other endpoint does. A name that is not one of ``value.WINNER_TRUTH_RULES`` is a
        ValueError before any store access. The provenance records the rule only when it is not 'gold' (absent means 'gold'); the caller (K6) writes the effective value in its ledger.
    k_condition: the gold of the K block of the rows on which the K-conditioned heads (the object and conversion heads, masked and open) of both models were trained and are scored
        here (S4b preparation M3, decision D21 B16): 'raw' (the default) is the true kill gold of the labels, as in every earlier run; 'binned' is the representative of the observed gold
        bin (0 for a team without a kill), the K that the composer feeds, so the NLL of the four K blocks is scored on the variable the heads were trained on. The rows of the role of both models are
        made with it (``data.ept_head_rows`` and ``B1.HeadRows``, the same condition). Only the condition of ``b1`` is checked here (``b1.k_condition`` must equal this argument: ValueError otherwise,
        before any store access); the condition of the EPT is NOT checked, because the model object carries no identity: the caller (the kernel, K6) must pass the condition it reads from the
        training identity of the checkpoint (``identity.get('k_condition', 'raw')``) and must refuse a mixture of checkpoints of different conditions.
    gold_reps: where the 31 representatives of the gold bins come from (``GOLD_REP_SOURCES``), the one array that the open Monte Carlo draws of both models and the 'binned' winner truth read:
        'labels' (the default) is ``labels/gold_reps`` as the label store holds it, the means of every FIT unit, as in every earlier run; 'in_scope_fit' is the means of the FIT units in
        scope (``value.fit_gold_bin_means(population='in_scope')``), the representatives of the 'binned' rows (``HeadRows.k_reps``). ``k_condition='binned'`` needs 'in_scope_fit' (ValueError
        otherwise: rows trained on the representatives and draws made with other ones would leave the support of the heads); the raw rows may use either. The array is checked to be the one
        of the rows when the condition is 'binned', and then the sha256 of the 31 numbers (``value.gold_reps_sha256``: float64 little-endian bytes, C order) must equal the
        ``b1.k_reps_sha256`` that the binned B1 set recorded when it was fitted (ValueError otherwise, also for a binned set without one, before the Monte Carlo): heads fitted on another
        label store would be scored on another K. The provenance records the source, the sha256 and ``gold_reps_fallback`` (the bins without an in-scope FIT entry, which hold the middle
        of their range; a list of bin numbers), only when the source is not 'labels'. The label arrays
        labels/kill_gold and labels/gold_bin and the S1 arrays matches/role, units/match and units/in_scope are then loaded whole and read at the FIT units only (the label store holds no
        holdout role).
    eval_average: whether the EPT is evaluated with the two-frame average (S4b preparation M7, decisions D10 P1 and D17): True (the default) is every earlier evaluation, exactly invariant under the
        team swap; False evaluates the recorded frame only (``FrameAveraged(model, average=False)``, one encoder call for each chunk), the evaluation of the 'no symmetry' check
        (``swap_mode='none', eval_average=False``). B1 is not affected. The model object carries no identity, so this is NOT read from the checkpoint here: the caller (the kernel, K6) must pass the
        value it reads from the training identity (``identity.get('eval_average', True)``; absent means True) and refuse a mixture of checkpoints of different settings. A ``FrameAveraged`` model must
        agree with it (ValueError otherwise, before any store access), and a value that is not a bool is a ValueError. The provenance says ``frame_averaged`` False, ``eval_average`` False and
        a note only when it is False (absent means True).
    zone_mark: whether the EPT has the zone mark (S4b preparation M9, decision D9): True (the default) is every earlier evaluation. False evaluates a model that was trained without it (A7): its M2Zone/masked
        block is not scored (the record of the block is the empty "unavailable" one with the reason 'zone_mark_off', so the thirteen records stay and a consumer finds why), the other twelve blocks
        are scored as before, and the open Monte Carlo of the M3 value is unchanged (the heads do not read z). The model object carries a setting of its own (``heads.zone_mark``) but no identity, so the
        caller (the kernel, K6) passes the value it reads from the training identity (``identity.get('zone_mark', True)``; absent means True) and must refuse a mixture of checkpoints of different
        settings; a model (or ``FrameAveraged``) that declares the other value is a ValueError before any store access, and a value that is not a bool is a ValueError. B1 is not affected. The
        provenance says ``zone_mark`` False and a note only when it is False (absent means True).
    b0: a fitted ``baselines_b0.B0Heads`` (S4b preparation M11, decision D16) or None (the default: no B0, the result of every earlier run). With a B0 each NLL record gets ``nll_b0``, the HR-5 NLL of B0 on
        exactly the rows the model is scored on: B0 builds its own rows of the same role (``b0.head_rows``, the same ``matches=`` selection as the model's, here every match of the role), the targets, exposures,
        matches, evaluation rows and conditions are proved equal to those of the EPT and B1 rows (a masked M3Obj row of B0 has no K, plan P6, so K is left out of that one comparison), and B0's
        entries are aggregated over the entries the model scores (``loss.scored_rows``: an M2Zone entry without a zone, a window or exposure of 0 and a gold entry without a bin are not rows), so the
        record holds one number for each scored row of the model. ``metrics.relative_structural`` turns such records into R_h and the mean over the eight heads. B0 must agree with the run: its ``kill_law``
        and ``k_condition`` must equal the arguments, it must have been fitted on FIT only and hold all thirteen heads (ValueError otherwise, before any store access; the attribute absent means plain and raw),
        and under 'binned' its ``k_reps_sha256`` must equal the sha256 of the ``in_scope_fit`` representatives of this evaluation, a binned B0 without a record being refused (before the Monte Carlo). With
        ``zone_mark=False`` the M2Zone block is empty for both models (reason zone_mark_off) and so is its ``nll_b0``. The provenance gets a ``b0`` entry (its law and condition, and the sha256 when binned).
    exposure, population: the exposure and the rows of the t0 forecasts of this evaluation, the M2 occurrence endpoints ``M2/t0/H*/occurrence`` (S4b preparation M12, decision D14). ``exposure`` is
        ``'observed'`` (the default, every earlier run: the probability uses the observed exposure, cut by the game end) or ``'nominal'`` (T-blind: 10 s per occurrence bin; ``EXPOSURES``); ``population`` is
        ``'all'`` (the default: every row with exposure, every earlier run), ``'primary'`` (the rows whose horizon the game end does not cut, ``endpoints.primary_rows``: hi - t0 >= H x 1000 ms with
        hi = ``frame_rows/hi``, i.e. t0 + H <= T - 30 s; selected for each horizon H) or ``'secondary'`` (the complement; both comparators are scored on the same observed exposure of the same rows);
        ``POPULATIONS``. The nominal exposure is allowed on the primary rows only (ValueError otherwise, before any store access): on them the observed exposure of the first H/10 bins is the nominal 10 s, so the
        nominal values are the observed ones and T enters only through the population filter (``_primary_masks``, the one place that reads ``frame_rows/t0`` and ``frame_rows/hi``). Only the M2 occurrence endpoints
        are selected, and their counts gain ``population_candidate`` and ``population_selected``; the M3 sign endpoint and the M4 endpoints are delta-conditional and keep the observed exposure and every row, and the
        NLL records (a likelihood under the observation design, not a forecast) are untouched. The provenance records the two settings, with a note, only when they are not the defaults (absent means observed and all).
    b1_swap_average: whether B1 is evaluated with the two-team-frame average that the EPT evaluation has (S4b preparation M13, decision B22; an S7 option): False (the default) is every earlier evaluation, B1 on the recorded
        frame only. True predicts every B1 head also from the features of the team-swapped views (``features.b1_swapped``) and the swapped conditions (``swap_conditions``), moves the answer back by the table of
        the head (``back_permutation``) and averages the two (``b1_predict_averaged``; equation (B22-1)); the open Monte Carlo draws from the averaged laws and object rates (``B1SwapAveragedOpenHeads``) with the
        same random numbers. It changes B1 only: the NLL of every B1 block, ``p_b1`` of the endpoints and the B0 columns are scored as before on the recorded targets and exposures, the EPT is untouched. It needs the
        B1 set to have been fitted on a team-closed column list when it dropped columns (``dropped_columns(pooled=True)``): a set whose ``dropped`` list drops one member of a pair of summary columns that the team
        swap exchanges and keeps the other (``features.swap_unclosed_columns``) is a ValueError that names the pair, before any store is read, because the swapped frame would feed the kept column the
        values of the dropped one that were constant in FIT. The provenance records the setting with a note only when it is True (absent means False).
    monte_carlo: whether the open M3 value is drawn (S4b preparation K6d, decision D30): True (the default) is every earlier evaluation. False is the closed-form path of the S4b decisions on STOP: the
        sampling of the size -> kills -> gold -> objects chain of the open units is not done (the one expensive part: ``draws`` per unit and model), so the M3 sign endpoint ``M3/open/positive_value`` is
        absent, ``labels/f0_unit`` and the binned winner truth are not read and ``open_cell_values`` is not made; every other endpoint (the M2 occurrence and mark, the M4 conversion) and every NLL record
        are bit for bit those of the full evaluation. The provenance says ``monte_carlo`` False with a note, only when it is False, and then shows ``draws`` 0 and has no ``seed``, ``rng`` or
        ``mc_uncertainty`` (nothing was drawn).
        해석: 열린 M3 값은 단위마다 `draws`번 사슬(규모, 킬, 골드, 오브젝트)을 뽑아 양의 값일 확률을 어림하는 값비싼 부분이고, 닫힌 꼴 점수(블록별 NLL, M2 발생 확률, M4 전환)는 이 추첨을 읽지 않는다.
        그래서 추첨과 그 끝점(`M3/open/positive_value`)만 빼도 나머지 기록의 비트는 전체 평가와 같다. 추첨이 없으므로 f0 참값 라벨도 읽지 않는다(S4b 판정은 STOP의 닫힌 꼴 값이다, D30).
    progress: None (the default: nothing is called) or a function ``progress(event, block, counts)`` called with ``'begin'`` and empty counts when a head block starts and with ``'end'`` and its whole-number
        counts (``rows``, ``views``, and for the open M3 size block ``units`` and ``draws``) when it is done, in the order of ``data.ROW_FIELDS`` (S4b preparation K6d, review M3: which block or which Monte
        Carlo used the time of a run that hit its cap). It reads and changes no array, so the result is the bits of a call without it. The guard ``check`` runs between the ``'end'`` of a block and the
        ``'begin'`` of the next, so a guard that fails at a block boundary leaves no block open (K6d re-check R1).
    NLLs remain raw; scale and normalized arrays are additional explicit fields.
    """
    if not callable(check):
        raise TypeError('check must be callable')
    check()  # before any store access
    check_role(role, holdout)                      # the role first: nothing but 'DEV', 'STOP' or the gate's 'TEST' reaches a store
    for name, val, minimum in (('batch_size', batch_size, 1), ('draws', draws, 2), ('seed', seed, 0)):
        if isinstance(val, bool) or not isinstance(val, (int, np.integer)) or val < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    if not isinstance(comparison, str) or not comparison.strip():
        raise ValueError('comparison must be a non-empty string')
    killlaw.check_law(kill_law)
    value.check_winner_truth(winner_truth)
    head_rows.check_k_condition(k_condition)
    check_gold_reps_source(gold_reps)
    if not isinstance(eval_average, bool):
        raise ValueError(f'eval_average must be True (the two-frame average) or False (the recorded frame only), not {eval_average!r}')
    if not isinstance(zone_mark, bool):
        raise ValueError(f'zone_mark must be True (the model has the zone mark) or False (A7: it was trained without it), not {zone_mark!r}')
    check_exposure_population(exposure, population)
    if progress is not None and not callable(progress):
        raise ValueError('progress must be None or a function progress(event, block, counts) that is called at the begin and the end of every head block')
    if not isinstance(monte_carlo, bool):
        raise ValueError(f'monte_carlo must be True (the open M3 value is drawn, every earlier run) or False (the closed-form records only), not {monte_carlo!r}')
    if not isinstance(b1_swap_average, bool):
        raise ValueError(f'b1_swap_average must be True (B1 is averaged over the two team frames, decision B22) or False (the recorded frame only, every earlier run), not {b1_swap_average!r}')
    if k_condition == 'binned' and gold_reps != 'in_scope_fit':
        raise ValueError("k_condition 'binned' needs gold_reps 'in_scope_fit': the rows are the representatives of the in-scope FIT units, so the draws must use the same 31 numbers")
    # Which roles (spec 1 anti-goal: no CAL/AUDIT access, TEST only through the gate): 'DEV' (the default, exploratory), 'STOP' (the whole STOP split, the data of the S4b decisions, decision D13) and 'TEST' only with the
    # ``HoldoutOpening`` of the gate (``check_role`` above, before this line); FIT is refused because its rows trained the models, and CAL, AUDIT and EXTERNAL because nothing here uses them. Two different things hold here
    # and neither is the whole guarantee.
    # The rows are selected below only through row views whose role argument is the variable ``role`` of this function (it is not read from the bundle, not from the stores and never a literal other than the default
    # of the argument); that is what restricts the evaluation. Three arrays are loaded whole and then used only at the positions of the role: units/in_scope is combined with the
    # unit mask of the role, labels/f0_unit is indexed by the open units of the role, and onsets/unit is indexed by the rows of the role (inside data._block_rows).
    # Under the shifted kill law (decision D5) a fourth, units/mark, is loaded whole (below) and indexed only by the open units of the role (fields['row']).
    # Under population 'primary' or 'secondary' (decision D14) two more, frame_rows/t0 and frame_rows/hi, are loaded whole (``_primary_masks``) and indexed only by the frame rows of the role of the M2Rate block.
    # Under b1_swap_average (decision B22) no array beyond the ones the B1 rows read is loaded: the swapped rows (``sw_rows``) are a second ``HeadRows`` of the same role with the swapped feature builder.
    # Under winner_truth='binned' (decision D5) two more, labels/gold_bin and labels/obj_value, are loaded whole (below) and indexed only by the open units of the role
    # (the early check) and by the rows of the M3 sign endpoint (the end), which are open units of the role.
    # Under gold_reps='in_scope_fit' or k_condition='binned' (decision D21 B16) the representatives are the in-scope FIT bin means of the label store: labels/kill_gold, labels/gold_bin,
    # matches/role, units/match and units/in_scope are loaded whole and read at the units of the FIT matches only (``value.fit_gold_bin_means``); the label store holds FIT, STOP and DEV
    # labels and no holdout role, and the rows of the role themselves are still selected only through the role above. (The representatives are FIT means for every role: a STOP evaluation
    # reads them from FIT, never from the STOP rows it scores.)
    # The bundle must also declare the same role explicitly, which refuses a stand-in or hand-built bundle before any store read; the
    # production builder sets it statically on a bundle whose stores also hold FIT and STOP rows, so it declares the intended use and
    # does not check the stores. A missing role used to default to 'DEV', which made this check unable to fire. The message names the role of the call.
    if getattr(bundle, 'role', None) != role:
        raise ValueError(f'only {role} is permitted: bundle.role must be explicitly "{role}", the role of this call')
    if tuple(b1.fit_roles) != ('FIT',) or tuple(b1.stop_roles) != ('STOP',):
        raise ValueError('B1 requires FIT training and STOP selection')
    if getattr(b1, 'kill_law', 'poisson') != kill_law:
        raise ValueError(f'the B1 heads were fitted for kill_law {getattr(b1, "kill_law", "poisson")!r} but the inference asks for {kill_law!r}')
    if getattr(b1, 'k_condition', 'raw') != k_condition:
        raise ValueError(f'the B1 heads were fitted for k_condition {getattr(b1, "k_condition", "raw")!r} but the inference asks for {k_condition!r}')
    if b0 is not None:
        _check_b0(b0, kill_law, k_condition)
    if b1_swap_average:
        unclosed = features.swap_unclosed_columns(b1.dropped)
        if unclosed:
            raise ValueError('b1_swap_average needs a B1 set whose dropped columns the team swap closes (fit it with dropped_columns(pooled=True)): the summary column pair(s) '
                             + ', '.join(f'{c} (dropped) / {p} (kept)' for c, p in unclosed) + ' would feed the kept column the constant values of the dropped one in the swapped frame')
    law = {} if kill_law == 'poisson' else {'kill_law': kill_law}              # the plain law is not passed: the calls are those of every earlier run
    condition = {} if k_condition == 'raw' else {'k_condition': k_condition}   # nor is the raw gold
    from c_ept.comparison.models import supported_model
    if not isinstance(model, FrameAveraged) and not supported_model(model):
        raise TypeError('model must be a registered EPT/N2 or FrameAveraged model')
    fa = model if isinstance(model, FrameAveraged) else FrameAveraged(model, **({} if eval_average else {'average': False}))      # the default wrapper is the one of every earlier evaluation
    if not supported_model(fa.model):
        raise TypeError('FrameAveraged must wrap a registered EPT/N2 model')
    declared_zone = getattr(getattr(fa.model, 'heads', None), 'zone_mark', None)           # the EPT declares its own setting; the N2 model declares none
    if declared_zone is not None and declared_zone != zone_mark:
        raise ValueError(f'the model has zone_mark {declared_zone!r} but the inference asks for zone_mark {zone_mark!r}: its z conditions and its M2Zone block would be read the other way')
    if fa.average is not eval_average:
        raise ValueError(f'the FrameAveraged model has average={fa.average} but eval_average={eval_average}: the evaluation frames of the model and of the setting must agree')
    wanted_device = torch.device(device)
    if any(p.device.type != wanted_device.type or (wanted_device.index is not None and p.device.index != wanted_device.index)
           or p.dtype != torch.float32 for p in fa.parameters()):
        raise ValueError('caller must load the float32 model on the requested device')
    if scales is not None and (set(scales) != set(loss.BLOCKS) or any(
            isinstance(v, bool) or not np.isfinite(v) or v <= 0 for v in scales.values())):
        raise ValueError('scales must be the frozen positive thirteen-block mapping')
    for kind, names in (('masked', B1.HEAD_NAMES), ('open', B1.OPEN_HEAD_NAMES)):
        if set(getattr(b1, kind)) != set(names):
            raise ValueError(f'incomplete B1 {kind} heads')
    open_heads = B1OpenHeads(b1)
    open_swapped = B1SwapAveragedOpenHeads(open_heads) if b1_swap_average else None
    e_rows = data.ept_head_rows(bundle.s1, bundle.s2, bundle.labels, role, **law, **condition)
    b_rows = B1.HeadRows(bundle.s1, bundle.s2, bundle.labels, b1.dropped, role, **law, **condition)
    # B22: the second frame of B1, the same rows with the features of the team-swapped views (their conditions are swapped block by block, ``swap_conditions``)
    sw_rows = head_rows.HeadRows(bundle.s1, bundle.s2, bundle.labels, role, lambda store, views: features.b1_swapped(store, views, b1.dropped), **law, **condition) if b1_swap_average else None
    # Prove legitimate empty populations from canonical masks, never by catching
    # a missing-field/view error. All nonempty blocks keep canonical validation.
    for field in ('_frame_in', '_onset_in', '_unit_in'):
        if not np.array_equal(getattr(e_rows, field), getattr(b_rows, field)):
            raise ValueError(f'paired {role} role masks disagree')
        if sw_rows is not None and not np.array_equal(getattr(sw_rows, field), getattr(b_rows, field)):
            raise ValueError(f'paired {role} role masks disagree (swapped B1 frame)')
    has_frame = bool(e_rows._frame_in.any())
    has_onset = bool(e_rows._onset_in.any())
    open_mask = e_rows._unit_in & np.asarray(bundle.s1.array('units/in_scope'))
    has_open = bool(open_mask.any())
    if has_open and monte_carlo:
        # Truth of the open M3 population, f0_u for every in-scope unit u of the role, must be finite and present.
        # It is checked here, before the first head: the Monte Carlo below takes hours and a single bad
        # label used to abort the run only after all of it (the same condition is re-checked on the exact
        # rows at the end, where it reads the p+ truth). Units outside the open population are not read.
        open_unit = np.flatnonzero(open_mask)
        f0_open = np.asarray(bundle.labels.array('labels/f0_unit'))
        if f0_open.ndim != 1 or open_unit[-1] >= len(f0_open) or not np.isfinite(f0_open[open_unit]).all():
            raise ValueError('missing/nonfinite M3 f0 truth')
    onset_unit = np.asarray(bundle.s1.array('onsets/unit')).astype(np.int64)
    # the mark of every unit, read only for the shifted kill law and then only at the open units of the role (the first-kill team of the units of the open M3 draws)
    unit_mark = np.asarray(bundle.s1.array('units/mark')).astype(np.int64) if kill_law == 'shifted' else None
    # The ONE array of representatives of this run: the open draws of both models and the binned winner truth below all read this local (M2 review m2). 'labels' is the array of the label
    # store, as in every earlier run; 'in_scope_fit' is the in-scope FIT means of the label store, the very function that makes the representatives of the binned rows.
    reps_fallback = None
    if gold_reps == 'labels':
        reps = compose._gold_reps(bundle.labels.array('labels/gold_reps'))
    else:
        found = value.fit_gold_bin_means(bundle.s1, bundle.labels, population='in_scope')      # refuses a store without any in-scope FIT entry (review m1)
        reps, reps_fallback = compose._gold_reps(found.means), [int(b) for b in found.fallback]
        for rows_ in (e_rows, b_rows):                                       # the binned rows hold the same 31 numbers (a belt over the one function): never two sources
            if rows_.k_reps() is not None and not np.array_equal(rows_.k_reps(), reps):
                raise ValueError('the representatives of the binned rows and of the draws differ')
    if k_condition == 'binned':
        # Review m4: the B1 heads were fitted on the 31 numbers of their own label store; these are the ones of THIS store. Another store would score the K heads on another K.
        recorded = getattr(b1, 'k_reps_sha256', None)
        if recorded is None:
            raise ValueError('the B1 heads of the binned K condition record no k_reps_sha256 (the representatives they were fitted on): they cannot be checked against this label store')
        if recorded != value.gold_reps_sha256(reps):
            raise ValueError('the B1 heads were fitted on other representatives of the gold bins than this label store gives (k_reps_sha256 differs)')
        if b0 is not None:                                                                  # the same two refusals for the B0 heads (decision D24; M3b review I1)
            recorded0 = getattr(b0, 'k_reps_sha256', None)
            if recorded0 is None:
                raise ValueError('the B0 heads of the binned K condition record no k_reps_sha256 (the representatives they were fitted on): they cannot be checked against this label store')
            if recorded0 != value.gold_reps_sha256(reps):
                raise ValueError('the B0 heads were fitted on other representatives of the gold bins than this label store gives (k_reps_sha256 differs)')
    # The rows of B0: the same role again (every match of the role, as the rows of the two models), with the B0 means, kill law and K condition (``B0Heads.head_rows`` also checks the condition and the sha of the rows).
    z_rows = None if b0 is None else b0.head_rows(bundle.s1, bundle.s2, bundle.labels, role)
    if z_rows is not None:
        for field in ('_frame_in', '_onset_in', '_unit_in'):
            if not np.array_equal(getattr(z_rows, field), getattr(e_rows, field)):
                raise ValueError(f'paired {role} role masks disagree (B0)')
    if winner_truth == 'binned' and monte_carlo:
        # The labels of the binned winner truth (decision D5, tie rule (i)): f is the gold of each team read as the representative of its bin, plus the objectives.
        # The default reads none of them. Like the f0 above, a bad label must abort here, before the Monte Carlo that takes hours, and not after it.
        truth_bins = np.asarray(bundle.labels.array('labels/gold_bin'))
        truth_obj = np.asarray(bundle.labels.array('labels/obj_value'))
        if has_open:
            if truth_bins.shape != (len(f0_open), 2) or truth_obj.shape != (len(f0_open), 2):
                raise ValueError('winner_truth: labels/gold_bin and labels/obj_value must hold two entries for each unit of labels/f0_unit')
            value.winner_truth(f0_open[open_unit], truth_bins[open_unit], reps, truth_obj[open_unit], 'binned')
    endpoints, nll_rows, m3_parts = [], [], []
    blocks = _BlockProgress(progress)
    with _evaluation(fa):
        for head, kind in data.ROW_FIELDS:
            # K6d re-check R1: the end of the block before is reported FIRST (it is also the end of an empty block, whose ``continue`` needs no call of its own), then the guard, then the begin of this block. A guard
            # that fails here (the wall cap, a stop request) therefore finds no block open: the block that was finished stays finished and the failure is not charged to a block that has not begun.
            # In words (Korean): 앞 블록의 끝 알림 -> 경계 점검 -> 이 블록의 시작 알림 순서이다. 점검이 이 자리에서 실패해도 열려 있는 블록이 없어 이미 끝난 블록이 실패로 닫히지 않는다. 계산에는 영향이 없다.
            blocks.end()
            check()
            blocks.begin(loss.block_key(head, kind))
            empty_reason = (f'no_{role}_frame_rows' if head == 'M2Rate' and not has_frame else
                            f'no_{role}_onsets' if kind == 'masked' and head != 'M2Rate' and not has_onset else
                            f'no_{role}_in_scope_units' if kind == 'open' and not has_open else None)         # 'no_DEV_...' for the default role, as before
            if empty_reason:
                _empty_block(head, kind, empty_reason, 0, scales, nll_rows, endpoints, b0 is not None)
                continue
            if head == 'M2Zone' and not zone_mark:          # a model without the zone mark has no zone law to score: the block is unavailable, not a number
                _empty_block(head, kind, 'zone_mark_off', 0, scales, nll_rows, endpoints, b0 is not None)
                continue
            ep = e_rows.parts(head, kind, masked_obj_k=True)
            bp = b_rows.parts(head, kind, masked_obj_k=True)
            bq = None if sw_rows is None else swap_conditions(sw_rows.parts(head, kind, masked_obj_k=True), head, kind)       # B22: the rows of the team-swapped frame
            if bq is not None:                                                  # the same rows in the same order, as ``b1_predict_averaged`` assumes
                for field in ('match', 'row', 'y', 'e', 'take'):
                    a, b = getattr(bp, field), getattr(bq, field)
                    if (a is None) != (b is None) or (a is not None and not np.array_equal(a, b)):
                        raise ValueError(f'{head}/{kind}: swapped B1 frame {field} mismatch')
                if bp.feat.shape != bq.feat.shape:
                    raise ValueError(f'{head}/{kind}: swapped B1 frame feat shape mismatch')
            for field in ('match', 'row', 'y', 'e', 'team', 'K', 'O', 'first_team'):
                a, b = getattr(ep, field), getattr(bp, field)
                if (a is None) != (b is None) or (a is not None and not np.array_equal(a, b)):
                    raise ValueError(f'{head}/{kind}: paired {field} mismatch')
            if set(ep.codes) != set(bp.codes) or any(not np.array_equal(ep.codes[k], bp.codes[k]) for k in ep.codes):
                raise ValueError(f'{head}/{kind}: paired condition mismatch')
            views = e_rows.view_numbers(head, kind)
            if not np.array_equal(views, b_rows.view_numbers(head, kind)):
                raise ValueError(f'{head}/{kind}: paired view mismatch')
            if b0 is not None and not np.array_equal(views, z_rows.view_numbers(head, kind)):          # the design of B0 is built from these views (M11 review m5)
                raise ValueError(f'{head}/{kind}: paired B0 view mismatch')
            fields, views, _ = data._block_rows(head, kind, ep, views, onset_unit)
            n = len(fields['row'])
            blocks.count(rows=n)
            if not n:
                excluded = len(e_rows.frame().rows) if head == 'M2Rate' else int(ep.n_zero_exposure)
                _empty_block(head, kind, 'no_eligible_rows_under_canonical_head_mask', excluded, scales, nll_rows, endpoints, b0 is not None)
                continue
            bpred, bscores = [], []
            fitted = getattr(b1, kind)[head]
            for lo in range(0, len(bp.row), batch_size):
                check()
                idx = np.arange(lo, min(lo + batch_size, len(bp.row)))
                p = (_b1_predict(head, kind, fitted, bp, b1.columns, idx, kill_law) if bq is None
                     else b1_predict_averaged(head, kind, fitted, bp, bq, b1.columns, idx, kill_law))                    # B22: the average over the two team frames
                if bp.e is not None:
                    e = bp.e[idx] if p.ndim == 1 else bp.e[idx, None]
                    if head == 'M3Kills' and kill_law != 'poisson':
                        score = -killlaw.logpmf(bp.y[idx], p, bp.first_team[idx], kill_law).sum(-1)       # (KL-3): the first team's count is 1 + N_d
                    else:
                        score = nll_math.poisson(bp.y[idx], p, e)
                        if score.ndim > 1:
                            score = score.sum(-1)
                else:
                    score = nll_math.categorical(bp.y[idx], p)
                bpred.append(p)
                bscores.append(score)
            bpred, bscores = np.concatenate(bpred), np.concatenate(bscores)
            if head == 'M2Rate':
                brate = np.zeros((n, C.N_BINS))
                take = np.searchsorted(fields['row'], bp.row)
                brate[take, bp.codes['q']] = bpred
            else:
                brate = bpred
            scores, seen, output = [], [], []
            cell_values = open_cell_values(bundle, fields['match'], fields['row']) if (head, kind) == ('M3Size', 'open') and monte_carlo else None
            for lo in range(0, n, batch_size):
                check()
                idx = np.arange(lo, min(lo + batch_size, n))
                unique, inv = np.unique(views[idx], return_inverse=True)
                blocks.add(views=len(unique))
                vt = {k: v.to(wanted_device) for k, v in data.ViewTensors(bundle.s2, unique, stats).items()}
                enc = fa.encode(vt)
                b = {k: torch.as_tensor(v[idx], device=wanted_device) for k, v in fields.items()}
                b['pos'] = torch.as_tensor(inv, device=wanted_device)
                pred = _output(fa, enc, head, b)
                scores.append(_numpy(loss.head_nll(head, pred, b, **law)))
                seen.append(_numpy(loss.scored_rows(head, b)))
                if head in ('M2Rate', 'M4Conv'):
                    output.append(_numpy(pred))  # these are expected counts, not rates
                if (head, kind) == ('M3Size', 'open') and monte_carlo:
                    m3 = fa.m3(enc, None, None, None, pos=b['pos'])
                    e_laws = tuple(_numpy(t) for t in (m3.size_logp.exp(), m3.kill_rate, m3.gold_logp.exp()))
                    numeric = bp.feat[idx] if bp.take is None else bp.feat[bp.take[idx]]
                    # the first-kill team of each unit (the shifted kill law only): 0 blue, 1 red, -1 an execution start
                    first = killlaw.first_team(unit_mark[fields['row'][idx]]) if kill_law == 'shifted' else None
                    if bq is None:
                        b_laws = open_heads.laws(numeric, check=check, **({} if first is None else {'d': first}))
                    else:                                                       # B22: the laws of the recorded frame and of the swapped frame, averaged
                        numeric_swapped = bq.feat[idx] if bq.take is None else bq.feat[bq.take[idx]]
                        b_laws = open_swapped.laws(numeric, numeric_swapped, check=check, **({} if first is None else {'d': first}))
                    for j, at in enumerate(idx):
                        def e_obj(s, K):
                            k = K.copy()
                            k[:, 2:] /= 1000.
                            return _numpy(fa.m3_obj(enc, None, None, None,
                                torch.as_tensor(s, device=wanted_device),
                                torch.as_tensor(k, dtype=torch.float32, device=wanted_device),
                                pos=b['pos'][j].expand(len(s))))
                        def b_obj(s, K):
                            if bq is None:
                                return open_heads.objects(np.repeat(numeric[j:j + 1], len(s), axis=0), s, K)
                            return open_swapped.objects(np.repeat(numeric[j:j + 1], len(s), axis=0), np.repeat(numeric_swapped[j:j + 1], len(s), axis=0), s, K)
                        common = dict(row=int(fields['row'][at]), match=int(fields['match'][at]), draws=draws,
                                      seed=seed, check=check)
                        if first is not None:
                            common.update(d=int(first[j]), kill_law=kill_law)
                        # The Monte Carlo chunk of each model (EPT-T011 A4-F6). The neural model keeps the run's batch_size: the shape of a GPU
                        # call can change the last bits. A LightGBM row does not depend on the other rows, so B1 takes every draw of the row
                        # in one call. The random draws are not affected by either.
                        pair = [sample_open_value(tuple(a[j] for a in laws), obj, cell_values[at], reps, batch_size=chunk, **common)
                                for laws, obj, chunk in ((e_laws, e_obj, batch_size), (b_laws, b_obj, None))]
                        m3_parts.append((common['match'], common['row'], *pair[0], *pair[1]))
                    blocks.add(units=len(idx))
            if (head, kind) == ('M3Size', 'open') and monte_carlo:
                blocks.count(draws=int(draws))
            escores, keep = np.concatenate(scores), np.concatenate(seen)
            bkeep = bp.y >= 0 if bp.e is None else bp.e > 0
            em, er, en = _aggregate(fields['match'][keep], fields['row'][keep], escores[keep])
            bm, br, bn = _aggregate(bp.match[bkeep], bp.row[bkeep], bscores[bkeep])
            if not np.array_equal(em, bm) or not np.array_equal(er, br):
                raise ValueError(f'{head}/{kind}: paired scored rows mismatch')
            zn = None
            if b0 is not None:
                # (D16) B0 on the very rows of the model: the targets, exposures, matches, rows and conditions are proved equal first (a masked M3Obj row of B0 has no K: plan P6), then B0's entries are
                # aggregated over the entries the model scores. ``bkeep`` is that mask for B1 (a zone or bin of -1, an exposure of 0), and the rows are equal, so it is the one of B0 too.
                zd, zp = z_rows.data(head, kind), z_rows.parts(head, kind)
                for field in ('match', 'row', 'y', 'e', 'team', 'first_team'):
                    a, b = getattr(ep, field), getattr(zd, field)
                    if (a is None) != (b is None) or (a is not None and not np.array_equal(a, b)):
                        raise ValueError(f'{head}/{kind}: paired B0 {field} mismatch')
                for field in ('K', 'O'):
                    a, b = getattr(ep, field), getattr(zp, field)
                    if field == 'K' and (head, kind) == ('M3Obj', 'masked'):
                        continue
                    if (a is None) != (b is None) or (a is not None and not np.array_equal(a, b)):
                        raise ValueError(f'{head}/{kind}: paired B0 {field} mismatch')
                if set(ep.codes) != set(zp.codes) or any(not np.array_equal(ep.codes[k], zp.codes[k]) for k in ep.codes):
                    raise ValueError(f'{head}/{kind}: paired B0 condition mismatch')
                zm, zr, zn = _aggregate(zd.match[bkeep], zd.row[bkeep], b0.entry_nll(head, kind, zd)[bkeep])
                if not np.array_equal(em, zm) or not np.array_equal(er, zr):
                    raise ValueError(f'{head}/{kind}: paired B0 scored rows mismatch')
                del zd, zp
            record = _nll_record(head, kind, em, er, en, bn,
                                 {'eligible': len(er), 'unscored_design_entries': int((~keep).sum()),
                                  'excluded_zero_exposure_cells_or_units': int(ep.n_zero_exposure)}, scales, zn)
            nll_rows.append(record)
            if head == 'M2Rate':
                mu = np.concatenate(output)
                erate = np.divide(mu, fields['e'], out=np.zeros_like(mu), where=fields['e'] > 0)
                if population == 'all':                                              # the call of every earlier run
                    occurrence = m2_endpoints(fields['match'], fields['row'], fields['y'], fields['e'], erate, brate)
                else:
                    # (D14) the primary rows of each horizon from the window ends in whole ms, or their complement; under the nominal exposure the endpoint uses the nominal 10 s (equal to the observed one on the rows used)
                    masks = _primary_masks(bundle.s1, fields['row'])
                    chosen = masks if population == 'primary' else {h: ~m for h, m in masks.items()}
                    occurrence = m2_endpoints(fields['match'], fields['row'], fields['y'], _nominal_exposure(fields['e'], masks) if exposure == 'nominal' else fields['e'], erate, brate,
                                              population=chosen)
                excluded = len(e_rows.frame().rows) - n
                for item in occurrence:
                    item['counts']['candidate'] += excluded
                    item['counts']['zero_exposure'] += excluded
                    if 'population_candidate' in item['counts']:         # one denominator with 'candidate': the frame rows without exposure are candidates that no population selects (M12 review m2)
                        item['counts']['population_candidate'] += excluded
                endpoints.extend(occurrence)
            elif (head, kind) == ('M4Conv', 'open'):
                mu = np.concatenate(output)
                erate = np.divide(mu, fields['e'][:, None], out=np.zeros_like(mu), where=fields['e'][:, None] > 0)
                endpoints.extend(m4_endpoints(fields['match'], fields['row'], fields['y'], fields['e'], erate, brate,
                                             excluded_no_window=ep.n_zero_exposure))
        blocks.end()                                                              # the last block of the loop
    check()
    if population != 'all':                      # an M2 block that is empty (``_empty_block``) selects no row but keeps the counts of the block: candidates as for 'candidate', none selected
        for ep in endpoints:
            if ep['module'] == 'M2' and 'population_selected' not in ep['counts']:
                ep['counts'].update(population_candidate=ep['counts']['candidate'], population_selected=0)
    match = np.asarray([v[0] for v in m3_parts], dtype=np.int64)
    row = np.asarray([v[1] for v in m3_parts], dtype=np.int64)
    a = np.asarray([v[2:] for v in m3_parts], dtype=np.float64).reshape(-1, 8)
    truth = np.asarray(bundle.labels.array('labels/f0_unit'))[row] if len(row) else np.empty(0)
    if monte_carlo and (truth.shape != row.shape or not np.isfinite(truth).all()):
        raise ValueError('missing/nonfinite M3 f0 truth')
    # The truth side f of the winner endpoint (y = f > 0, draw_truth = f == 0), decision D5 tie rule (i), value.winner_truth:
    #   'gold'    f = f0                                                    the labels as they are (a copy of the same floats): every earlier run
    #   'binned'  f = (g_blue + obj_blue) - (g_red + obj_red),  g_t = gold_reps[bin_t] if the team has a kill (bin_t >= 0) else 0
    # The gold term of the 'binned' f is the variable of the model side (_open_draws: g = gold_reps[bins] where k > 0, 0 elsewhere; the same reps array), so two teams of one
    # bin tie on the gold term on both sides. The objective term is NOT the model's: obj is the realized attributed price of labels/obj_value (a tower at its realized tier
    # price), while _open_draws prices the drawn counts at open_cell_values (a tower at the expected tier price). A pair of one bin is a tie when the realized objectives are equal.
    # In words (Korean): 모형은 팀의 킬 골드를 구간 대표값으로만 읽으므로 참값의 골드 항도 같은 대표값으로 계산해야 같은 구간의 두 팀이 양쪽에서 같이 동률이 된다. 오브젝트 항은 실현된 가격 그대로이다.
    # without the Monte Carlo (K6d, the closed-form STOP path, decision D30) there is no M3 sign endpoint: the other endpoints and every NLL record are as before.
    # In words (Korean): 추첨을 하지 않으면 양의 값 확률 끝점은 만들 수 없고(참값도 읽지 않는다), 추첨과 무관한 나머지 끝점과 13개 NLL 기록은 전체 평가와 같은 값이다.
    if monte_carlo:
        if winner_truth == 'gold':
            truth = value.winner_truth(truth, None, None, None, 'gold')
        else:
            truth = value.winner_truth(truth, truth_bins[row], reps, truth_obj[row], 'binned')
        endpoints.insert(3, endpoint('M3/open/positive_value', 'M3', 'open', None, 'f0_gt0', match, row, truth > 0,
            a[:, 0], a[:, 4], draw_truth=truth == 0, p_draw_ept=a[:, 1], p_draw_b1=a[:, 5],
            mc_se_ept=a[:, 2], mc_se_b1=a[:, 6], mc_se_draw_ept=a[:, 3], mc_se_draw_b1=a[:, 7],
            counts={'candidate': len(row), 'eligible': len(row), 'draw': int((truth == 0).sum()),
                    'unavailable_reason': None if len(row) else f'no_{role}_in_scope_units'}))
    result = {'endpoints': endpoints, 'nll': nll_rows, 'provenance': {
        'adapter': 'EPT-T003/open-v1', 'role': role, 'frame_averaged': eval_average, 'draws': int(draws), 'seed': int(seed),
        'rng': 'NumPy default_rng SeedSequence([seed,match,unit]); same initial streams, no strict Poisson CRN guarantee',
        'mc_uncertainty': 'Bernoulli sample SE; boundary zero SE does not prove exact probability',
        'm2_window': 'existing observed exposure clipped at occurrence boundary T-30s; no full-H filter',
        'm3_tower_value': 'canonical expected tower-tier price at first-kill time f, destroyed_before(f)',
        'm3_nll': 'conditional joint-law factors scored at observed size/K; never used for p+ prediction',
        'm4_condition': 'actual K/O at delta; existing exposure clipped at T',
        'nll_weighting': 'HR-5 bin/team sum per row, then endpoint match balancing; not training loss',
        'scales': None if scales is None else dict(scales), 'stats_sha256': stats.sha256,
        'structural_heads': list(STRUCTURAL_HEADS),
        'comparison': comparison,
        # absent means True, every earlier run: the open M3 value was drawn (the M3 sign endpoint is there)
        **({} if monte_carlo else {'monte_carlo': False, 'monte_carlo_note': 'no Monte Carlo draw of the open M3 value was made: the M3/open/positive_value endpoint is absent and no f0 truth was read; '
                                                                             'the other endpoints and every NLL record are those of the full evaluation, bit for bit (K6d, the closed-form STOP path, decision D30)'}),
        # the role: absent note means DEV, every earlier run. STOP is the data of the early stopping of both arms (decision D13), which the numbers must not be read as a held-out estimate of
        **({} if role == 'DEV' else {'role_note': f'the rows of the role {role}: each arm used the STOP rows that its own training run recorded for early stopping (the EPT its STOP rows, B1 the stop_rows of its '
                                                  'fit_head); this call cannot see either row set (B1Heads records the role only); the rows are the data of the S4b decisions (decision D13) and '
                                                  'the numbers are not a held-out estimate' if role == 'STOP' else
                                                  f'the rows of the role {role}, opened once by the gate (decision D20)'}),
        # recorded only when it is not the plain law, so that the provenance of a plain-law run is that of every earlier run
        **({} if kill_law == 'poisson' else {'kill_law': kill_law, 'kill_law_note': 'M3Kills scored and the open M3 draws made with K_d = 1 + N_d for the first-kill team '
                                                                                     'of the unit (decision D5); the NLL is the shifted score, not comparable with a plain-law NLL'}),
        # the same for the gold of the K block of the rows: absent means 'raw', the true kill gold of the labels
        **({} if k_condition == 'raw' else {'k_condition': k_condition, 'k_condition_note': 'the K-conditioned heads (object and conversion, masked and open) of both models were '
                                                                                            'trained on, and are scored here on, the representative of the observed gold bin '
                                                                                            '(0 for a team without a kill), the K that the composer feeds (decision D21 B16); '
                                                                                            'the NLL of these four blocks is not comparable with a raw-gold NLL'}),
        # the same for the source of the representatives: absent means 'labels', labels/gold_reps as the label store holds it; the sha256 is that of the 31 float64 numbers
        # (little-endian, C order: ``value.gold_reps_sha256``) and the fallback lists the bins without an in-scope FIT entry (they hold the middle of their range)
        **({} if gold_reps == 'labels' else {'gold_reps': gold_reps, 'gold_reps_sha256': value.gold_reps_sha256(reps), 'gold_reps_fallback': reps_fallback,
                                             'gold_reps_note': 'the representatives of the gold bins (the open Monte Carlo draws and the binned winner truth read this one array) are the '
                                                                'means of the FIT units in scope, not labels/gold_reps (decision D21 B16)'}),
        # the same for the frames of the EPT evaluation: absent means True, the two-frame average (S4b preparation M7, decisions D10 P1 and D17)
        **({} if eval_average else {'eval_average': False, 'eval_average_note': 'the EPT was evaluated in the recorded frame only, with no two-frame average (the D17 "no symmetry" '
                                                                                'check); its NLL and its endpoints are not invariant under the team swap and not comparable with an averaged evaluation; B1 is unchanged'}),
        # the same for the zone mark of the EPT: absent means True (M2Zone scored, z a condition); False (A7) leaves the M2Zone block unavailable (decision D9)
        **({} if zone_mark else {'zone_mark': False, 'zone_mark_note': 'the EPT was trained and evaluated without the zone mark (A7): the M2Zone/masked block is unavailable (reason zone_mark_off) and the other '
                                                                       'blocks are scored on a model whose M3, object and conversion heads do not read z; compare the shared M2Rate and M2Mark blocks, not the whole M2 NLL (decision D9)'}),
        # the exposure and the rows of the t0 forecasts (S4b preparation M12, decision D14): absent means observed and all, every earlier run
        **({} if exposure == 'observed' else {'exposure': exposure, 'exposure_note': 'the M2 occurrence endpoints use the nominal exposure (10 s per occurrence bin, no game end) on the primary rows only, where it is '
                                                                                    'the observed exposure of the first H/10 bins; the game end enters only through the population filter (decision D14); the NLL records and the M3 and '
                                                                                    'M4 endpoints are unchanged'}),
        **({} if population == 'all' else {'population': population, 'population_note': 'the M2 occurrence endpoints are restricted to the ' + (
            'primary rows of each horizon H (hi - t0 >= H x 1000 ms, i.e. t0 + H <= T - 30 s): the occurrence truth is not truncated by the game end' if population == 'primary' else
            'secondary rows of each horizon H (the rows that are not primary: the window is cut by the game end), with the same observed exposure for every comparator') +
            ' (decision D14); counts: population_candidate, population_selected; the NLL records and the M3 and M4 endpoints are not restricted'}),
        # the two-team-frame average of B1 (S4b preparation M13, decision B22): absent means False, B1 on the recorded frame only
        **({} if not b1_swap_average else {'b1_swap_average': True, 'b1_swap_average_note': 'B1 was evaluated as the average over the recorded team frame and the team-swapped frame (decision B22, S7 option): '
                                                                                             'its heads predict also from the features of the swapped views and the swapped conditions, the answer is moved back '
                                                                                             'by the table of the head and averaged, the NLL is scored on the recorded targets, and the open Monte Carlo draws from '
                                                                                             'the averaged laws; the EPT, the targets and the random numbers are unchanged'}),
        # the B0 heads of the relative NLL (S4b preparation M11, decision D16): absent means no B0 (no nll_b0 in the records)
        # plus the identity of the set as files (sha256 of the array file and of the JSON that ``save`` would write), M11 review m4
        **({} if b0 is None else {'b0': {'kill_law': getattr(b0, 'kill_law', 'poisson'), 'k_condition': getattr(b0, 'k_condition', 'raw'),
                                         **({} if k_condition == 'raw' else {'k_reps_sha256': b0.k_reps_sha256}), **b0.saved_sha256()}}),
        # the same for the rule of the winner truth: absent means 'gold', the raw f0 of the labels
        **({} if winner_truth == 'gold' else {'winner_truth': winner_truth, 'winner_truth_note': 'the truth of the M3 sign endpoint (f > 0, f == 0) has its gold term on the '
                                                                                                   'gold_bin representatives, as the models do (decision D5, tie rule (i)), and its objective '
                                                                                                   'term at the realized attributed prices of labels/obj_value (a tower at its realized tier '
                                                                                                   'price; the models price a tower at the expected tier price): a pair of one gold bin ties '
                                                                                                   'when the realized objective values are equal; no prediction or NLL depends on it'})}}
    if not monte_carlo:                                      # nothing was drawn: no draws, and no seed, stream or sampling error to speak of (review M2)
        provenance = result['provenance']
        provenance['draws'] = 0
        for key in ('seed', 'rng', 'mc_uncertainty'):
            del provenance[key]
    return result
