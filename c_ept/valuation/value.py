r"""[모듈 개요 및 하네스 위치]
이 모듈은 교전 모형의 지도학습용 정답 타깃 라벨(f₀ 교전 단위 가치, f_{0,conv} 사후 60초 전환 가치,
승자 W, 확실한 승패 여부, 31개 킬 골드 구간, V_start 미래 누적 가치)을 생성하고
경제적/오브젝트 보존 회계 항등식을 전수 검증하는 핵심 정답 생성 및 평가 모듈이다 (stage S3 Task 4).

A. 하네스 및 파이프라인 내 위치:
   - S1 저장소의 원시 사건(킬, 오브젝트, 프레임)과 가격표(PRICES.json), 공식 게임 보상 규칙(rulepay)을
     결합하여 각 교전의 최종 경제적 순 골드 가치를 라벨링한다.
   - S3 평가 및 EPT/B0/B1 모델 학습의 기준 정답(Ground Truth)을 생성하여 `labels/` 스토어에 기록한다.
   - 엄격한 보존 회계(accounting, 판정 R11): 귀속 오브젝트 + 전환 오브젝트 + 미귀속 잔여 = 전체 오브젝트
     항등식이 단 1개의 오차도 없이 성립함을 전수 검증한다 (`max_abs_error == 0`).

B. 모듈 아키텍처 및 8대 블록 구성:
   - 블록 1: 전역 상수 및 설정 (GOLD_EDGES: 100골드 간격 31구간, SURE_F0: |f₀| ≥ 150)
   - 블록 2: 가격표 로드 및 오브젝트 단가 매핑 (load_prices, object_price)
   - 블록 3: 킬 골드 구간 분할 및 승자 판정 (gold_bin, winner_of, gold_bin_means; 식 VAL-1, VAL-2)
   - 블록 4: 피해자 레벨 인덱싱 및 어시스트 풀 연동 (kill_levels, kill_levels_for_record)
   - 블록 5: 교전 단위 가치 산출 엔진 (UnitValues, unit_values; 식 VAL-4, VAL-5)
   - 블록 6: 향후 예측 시계별 누적 가치 계산 (v_start; H = 30, 60, 90, 120, 180s; 식 VAL-6)
   - 블록 7: 엄밀한 경제/오브젝트 보존 회계 감사 (accounting; 판정 R11, 식 VAL-7 ~ VAL-9)
   - 블록 8: 정답 라벨 저장소 원자적 파일 직렬화 (write_labels)

C. 경제적 가치 정의 및 회계 체계:
   1) 킬 골드 (Kill Gold): 기록된 기본 현상금 + 제압 골드(shutdown) + rulepay 기반 어시스트 풀 (처형은 0골드).
   2) 귀속 오브젝트 가치 (Attributed Objective Value): 교전 중 획득된 오브젝트의 경제적 가치 (흐름 가격 또는 즉시 가치).
   3) 교전 단위 가치 f₀(unit) = (Blue 킬골드 − Red 킬골드) + (Blue 귀속오브젝트가치 − Red 귀속오브젝트가치)
   4) 사후 60초 전환 가치 f₀(conv) = 교전 종료 후 60초 이내에 획득된 오브젝트들의 순 가치.
   5) 승자 판정: W = sign(round(f₀, 6)), 유의미한 승패: |f₀| ≥ 150.
   6) V_start(t₀, H) = t₀ 시점 이후 H초 내에 시작하는 모든 교전들의 (f₀ + f_{0,conv}) 총합.

D. 핵심 수식 (VAL-1 ~ VAL-9):
   (VAL-1)  골드 구간        b = min(⌊g / 100⌋, 30)   (g = 킬 골드; 0~3000골드 100단위 + 초과 30번 구간)
   (VAL-2)  승자 및 확실층   W = sign(⌊f₀ · 10⁶⌉ / 10⁶),   sure = 1[|f₀| ≥ 150]
   (VAL-3)  오브젝트 단가    price_c = price_table[group(c)]
   (VAL-4)  교전 단위 가치   f₀ = g_blue − g_red + Σ_c sign_c · price_c · O_c
   (VAL-5)  전환 가치        f_{0,conv} = Σ_c sign_c · price_c · conversions_c
   (VAL-6)  누적 교전 가치   V_start(t₀, H) = Σ_{unit ∈ (t₀, t₀+H]} (f₀ + f_{0,conv})
   (VAL-7)  오브젝트 보존    O_total = O_attributed + O_converted + O_rest  (오차 0 필수)
   (VAL-8)  전투 통계 보존   K_total = K_attributed + K_rest
   (VAL-9)  경제 가치 보존   Value_total = Value_attributed + Value_converted + Value_rest

E. 하네스 제약 및 무결성 보장:
   - 엄격한 인과성(Causality): t₀ 시점 평가에 미래 정보 누설 차단.
   - 단일 진실 공급원(Single Source of Truth): rulepay 및 flowprice 가격표의 SHA-256 검증.
"""
# Ref: docs/superpowers/plans/2026-10-21-c-ept-s3.md decisions P3 (kill gold = bounty + shutdown + assist pool, executions pay nothing, no
#      rule payout added to a flow price) and P4 (K = kills and kill gold, 31 gold bins, bin means from FIT as representatives), Task 4, and
#      the rulings R11 (the accounting identity is exact on counts) and R12 (winner, gold bin, level conventions) of the S3 ledger.
# Ref: docs/superpowers/specs/2026-10-20-c-ept-design.md sections 3.4 and 8 (V_start over the first kills in (t0, t0 + H], f0 = kill gold +
#      objective prices, winner W = sign f0(K, O) and the sure stratum |f0| >= 150), section 3.2 (executions pay no team) and section 11,
#      item 17 (an objective without a known kind or taker is counted as rest).
# Ref: LEGACY gameplay/event_survival.py::death_intervals (the level of a victim at its kill: the highest LEVEL_UP at or before the kill ms,
#      as in c_ept.state), through c_ept.traits.LevelIndex.
from __future__ import annotations

import functools
import json
import math
import numbers
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

from c_ept import config as C
from c_ept.valuation import flowprice
from c_ept.data import reader
from c_ept.valuation import rulepay
from c_ept.data import store_core
from c_ept.preprocessing import traits
from c_ept.domain import units
from c_ept.data.casts import narrow
from c_ept.domain.units import BLUE, RED, _ALL_PIDS, _POPCOUNT, _is_pid

LABEL_ROLES = flowprice.S3_ROLES                       # FIT, STOP, DEV: the matches stage S3 values
N_GOLD_BINS = 31
GOLD_EDGES = np.arange(N_GOLD_BINS, dtype=np.float64) * 100.0           # lower edge of each bin: 0, 100, ..., 3,000 (the last bin is open)
GOLD_EDGES.setflags(write=False)
GOLD_FALLBACK = np.append(GOLD_EDGES[:-1] + 50.0, 3000.0)               # the representative of a bin with no entry
GOLD_FALLBACK.setflags(write=False)
F0_DECIMALS = 6                                        # f0 is rounded to a millionth of a gold before its sign is taken
SURE_F0 = 150.0                                        # the sure stratum: |f0| >= 150
LABEL_ARRAYS = (('kill_gold', 'f8'), ('obj_value', 'f8'), ('conv_value', 'f8'), ('f0_unit', 'f8'), ('f0_conv', 'f8'), ('winner', 'i1'),
                ('sure', '?'), ('gold_bin', 'i1'), ('v_start', 'f8'), ('gold_reps', 'f8'))
_N_GROUPS = flowprice.N_GROUPS
_N_CELLS = 2 * len(C.O_KINDS)                          # the objective cell of S1: 2 kind + taker
_SPAN = 1 << 40                                        # ms: a match number times this plus a time orders the (match, time) pairs
_UNIT_FIELDS = ('kill_gold', 'obj_value', 'conv_value', 'f0_unit', 'f0_conv', 'winner', 'sure', 'gold_bin', 'n_kills', 'valued')


def _wbincount(index: np.ndarray, *, weights: np.ndarray, minlength: int) -> np.ndarray:
    """``np.bincount(index, weights, minlength)`` as float64: numpy returns an integer array of zeros for an empty ``index``."""
    return np.bincount(index, weights=weights, minlength=minlength).astype(np.float64)


# =============================================================================
# 블록 2: 가격표 로드 및 오브젝트 단가 매핑 (식 VAL-3)
# =============================================================================
def load_prices(path) -> dict[str, float]:
    """The price of each group, ``{group: value}`` in the order of ``flowprice.GROUPS``, from a ``PRICES.json`` of the s3a run.

    The value used (flow price or immediate value, whichever the run adopted) is read from ``<group>["value"]``. A missing group, an entry
    without a ``value``, a ``value`` that is None, not a number or not finite is a ValueError that names the group.

    (VAL-3)의 가격표 price_ψ를 읽는다. 값은 흐름 가격이거나 즉시값(늘 즉시값을 쓰는 묶음, 앞 추세 시험에 실패한 묶음, 흐름 쪽을 추정할 수 없는 묶음)이며, 어느 쪽인지는 s3a 실행이 (FLOW-10)으로 정하였다(설계서 8.1절의 채택 조건).
    """
    doc = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(doc, dict):
        raise ValueError('the price table must be a JSON object with one entry for each group')
    out = {}
    for name in flowprice.GROUPS:
        entry = doc.get(name)
        if not isinstance(entry, dict) or 'value' not in entry:
            raise ValueError(f'the price table has no entry with a value for the group {name}')
        out[name] = _price(name, entry['value'])
    return out


def _price(name: str, v) -> float:
    if v is None:
        raise ValueError(f'the price of {name} is None (its flow fit and its immediate value were both unavailable)')
    if isinstance(v, bool) or not isinstance(v, numbers.Real) or not math.isfinite(v):
        raise ValueError(f'the price of {name} is not a finite number')
    return float(v)


def _price_table(prices) -> np.ndarray:
    """(12,) float64 prices in the order of ``flowprice.GROUPS``; TypeError for something that is not a mapping, ValueError for a group
    that is missing or not a finite number."""
    if not isinstance(prices, Mapping):
        raise TypeError('prices must be the {group: price} dict that load_prices returns')
    return np.array([_price(name, prices.get(name)) for name in flowprice.GROUPS], dtype=np.float64)


def object_price(kind, sub, prices) -> np.ndarray:
    """The price of objectives by their ``config.O_KINDS`` index ``kind`` and tower tier ``sub`` (0-3), float64, the shape of the inputs.

    A tower is priced by its tier (outer, inner, base, nexus), every other kind by its own group. Where the group is not known (another
    monster or building: kind -1, a tower without a tier, a kind or tier that does not exist) the price is 0; ``accounting`` counts those.

    (VAL-3)의 오브젝트 하나의 가격 price_ψ이다: 종류와 포탑 단계로 묶음 ψ를 알면 그 묶음의 가격이고 모르면 0이다(공통 가정 (다)).
    """
    table = _price_table(prices)
    group = flowprice.group_of(kind, sub).astype(np.int64)
    return np.where(group >= 0, table[np.clip(group, 0, _N_GROUPS - 1)], 0.0)


# =============================================================================
# 블록 3: 킬 골드 구간 분할 및 승자 판정 (식 VAL-1, VAL-2)
# =============================================================================
def gold_bin(gold) -> np.ndarray:
    """The bin (int8, 0 .. 30) of kill gold: floor(gold / 100), the open bin 30 for 3,000 and more. ValueError for gold that is negative or
    not finite (nothing is clipped into a bin).

    수식
    ----
    (VAL-6)  b = min(⌊g / 100⌋, 30)        (g ≥ 0 유한; 0–2,999는 100골드씩 30구간이고 3,000 이상은 열린 구간 30이다)

      기호 | 뜻                | 단위 | 범위     | 코드 변수
      g    | 팀의 킬 골드      | 골드 | ≥ 0 유한 | gold
      b    | 킬 골드 구간 번호 | —    | 0, …, 30 | 반환값(int8)

    해석
    ----
    식 (VAL-6)은 킬 골드를 100골드 폭의 구간으로 나누고 3,000 이상은 한 구간에 모은 31구간이며, 모든 모형이 같은 구간을 쓸 때에만 골드 NLL을 맞댈 수 있다((NLL-3), 계획 P4). 음수나 유한하지
    않은 골드는 어느 구간에도 밀어 넣지 않고 거부한다. 킬이 없는 팀에 구간 −1을 주는 것은 이 함수가 아니라 교전 단위의 라벨((VAL-2)–(VAL-4))을 만드는 단계이다(판정 R12).

    근거
    ----
    계획 P4(킬 골드를 구간 범주로 맞춘다), 판정 R12(킬이 없는 팀은 구간이 없다), 검산 노트 math-control-03 D절(구간 나누기; 시험이 같은 수를 확인한다).
    """
    g = np.asarray(gold, dtype=np.float64)
    bad = ~(np.isfinite(g) & (g >= 0))
    if bad.any():
        raise ValueError(f'gold_bin: {int(bad.sum())} value(s) are negative or not finite')
    return narrow('gold_bin', np.searchsorted(GOLD_EDGES, g, side='right') - 1, np.int8)


def winner_of(f0_unit) -> tuple[np.ndarray, np.ndarray]:
    """``(winner, sure)`` of the f0 of units (blue minus red): winner int8 +1 (blue), 0 (a draw) or -1 (red) is the sign of f0 rounded to a
    millionth of a gold; sure is |rounded f0| >= 150. ValueError for a value that is not finite.

    수식
    ----
    (VAL-5)  W_i = sign(round(f₀_i, 6)) ∈ {−1, 0, +1},     확실한 승패 ⇔ |round(f₀_i, 6)| ≥ 150

      기호 | 뜻                                      | 단위 | 범위            | 코드 변수
      f₀_i | 교전 i의 f₀(K, O) (파랑 − 빨강)         | 골드 | 유한, 부호 있음 | f0_unit
      round(f₀_i, 6) | f₀_i를 소수 여섯째 자리까지 반올림한 값 | 골드 | 유한 | r
      W_i  | 승자: +1 파랑, 0 무승부, −1 빨강        | —    | −1, 0, +1       | winner(int8)
      확실 | 확실한 승패 층의 표시                   | —    | 참, 거짓        | sure

    해석
    ----
    식 (VAL-5)의 승자는 킬 골드(도움 풀을 포함한다)와 오브젝트 가격을 합한 f₀의 부호이고 0은 무승부로 따로 센다. 부호를 반올림한 값에서 읽으므로 0.1 + 0.2 − 0.3처럼 부동소수점 오차만 남은 합은 무승부이다.
    |f₀| ≥ 150인 교전은 승패가 뚜렷한 층으로 따로 보고하며 이 층도 같은 반올림 값으로 가른다. 이 함수는 받은 값의 부호만 읽으며, 라벨을 만드는 단계가 전환을 뺀 f₀_i를 넘기므로 라벨의 승자에는 전환이
    없다(전환을 넣은 승자는 민감도이다; 설계서 8절). 유한하지 않은 값은 거부한다.

    근거
    ----
    판정 R12(승자는 반올림한 f₀의 부호), 설계서 8절(승자 W와 확실한 승패 층), 계획 전역 제약의 f₀ 항(무승부는 따로 센다).
    """
    f0 = np.asarray(f0_unit, dtype=np.float64)
    if not np.isfinite(f0).all():
        raise ValueError('winner_of: f0 must be finite')
    r = np.round(f0, F0_DECIMALS)
    return (r > 0).astype(np.int8) - (r < 0).astype(np.int8), np.abs(r) >= SURE_F0


class GoldReps(NamedTuple):
    """The representatives of the gold bins: ``means`` (31,) float64 and ``fallback``, the bins that had no entry and took their middle.

    (VAL-7)의 구간 대표값 rep_b(``means``)와 항목이 없어 가운데를 쓴 구간의 번호(``fallback``)이다.
    """
    means: np.ndarray
    fallback: tuple


def _unit_selection(selected, n_units: int) -> np.ndarray:
    """A (n_units,) bool mask from a boolean mask or an array of unit numbers; ValueError for anything else."""
    a = np.asarray(selected)
    if a.dtype == np.bool_:
        if a.shape != (n_units,):
            raise ValueError(f'a unit mask must have one entry for each of the {n_units} units, not shape {a.shape}')
        return a
    if a.dtype.kind in 'iu' and a.ndim == 1:
        if a.size and (a.min() < 0 or a.max() >= n_units):
            raise ValueError(f'unit numbers must lie in 0 .. {n_units - 1}')
        mask = np.zeros(n_units, dtype=bool)
        mask[a] = True
        return mask
    raise ValueError('units must be a boolean mask or a 1-d array of unit numbers')


def gold_bin_means(uv: 'UnitValues', units) -> GoldReps:
    """The representative of each gold bin: the mean kill gold of the team entries in it, over the selected units (the FIT units: see
    ``role_units``). An entry is a team of a valued unit with at least one kill (bin >= 0); a team without a kill is no entry. The mean is
    exact (``math.fsum``), so ``count x mean`` summed over the bins reproduces the gold of the entries. A bin with no entry takes the middle
    of its range (100 b + 50; 3,000 for the open bin) and is named in ``fallback``.

    수식
    ----
    (VAL-7)  rep_b = (1 / |E_b|) · Σ_{e ∈ E_b} g_e                         (E_b ≠ ∅)
             rep_b = 100·b + 50 (b < 30),   rep_30 = 3,000                 (E_b = ∅; 그 b를 fallback에 적는다)
             E_b = 선택한 교전 단위 가운데 값이 매겨진 단위의 팀 항목 e = (i, t) 가운데 킬이 하나 이상이고 구간이 b인 것, g_e = 그 항목의 킬 골드

      기호  | 뜻                      | 단위 | 범위     | 코드 변수
      E_b   | 구간 b의 팀 항목의 집합 | —    | 0개 이상 | gold[lo:hi]
      g_e   | 항목 e의 킬 골드        | 골드 | ≥ 0 유한 | uv.kill_gold
      rep_b | 구간 b의 대표값         | 골드 | ≥ 0      | means[b]

    해석
    ----
    식 (VAL-7)의 대표값은 구간에 든 실제 킬 골드의 평균이므로, 항목 수에 대표값을 곱해 구간마다 더하면 선택한 항목의 킬 골드 합이 그대로 나온다. 구간의 가운데를 대표값으로 쓰면 킬 골드가
    구간의 시작에 몰려 있을 때 한쪽으로 치우치고 열린 구간에는 가운데가 없다(계획 P4). 항목이 없는 구간은 가운데(열린 구간은 아래끝 3,000)를 쓰고 그 번호를 따로 알린다(판정 R12). 킬이 없는 팀은 항목이 아니다.
    항목은 호출한 쪽이 고른 교전 단위에서 나오며(실행기는 FIT 경기의 단위를 고른다) 단위가 범위 안인지는 따지지 않는데, 이는 S3 as-built 기록의 열린 항목이다.

    근거
    ----
    계획 P4(구간 대표값은 FIT 구간 평균), 판정 R12(항목이 없는 구간은 가운데, 목록으로 알림), 검산 노트 math-control-03 D절(구간 평균은 합을 재현한다; 시험이 같은 수를 확인한다).
    """
    n = len(uv.f0_unit)
    sel = _unit_selection(units, n)
    # 식 (VAL-7)의 항목 집합 E_b: 선택한 단위의 두 팀 가운데 구간이 0 이상인 팀(킬이 있는 팀)이다. 값이 매겨지지 않은 단위의 구간은 −1이어서 들지 않는다.
    bins, gold = uv.gold_bin[sel].ravel().astype(np.int64), uv.kill_gold[sel].ravel()
    keep = bins >= 0
    bins, gold = bins[keep], gold[keep]
    if not np.isfinite(gold).all():
        raise ValueError('gold_bin_means: a team entry with a kill has no finite kill gold')
    order = np.argsort(bins, kind='stable')
    gold, start = gold[order], np.searchsorted(bins[order], np.arange(N_GOLD_BINS + 1))
    means, fallback = GOLD_FALLBACK.copy(), []
    # 식 (VAL-7)의 평균: 구간마다 math.fsum으로 정확히 더한 합을 항목 수로 나눈다. 항목이 없으면 대체값(가운데, 열린 구간은 3,000)을 두고 번호를 남긴다.
    for b in range(N_GOLD_BINS):
        lo, hi = int(start[b]), int(start[b + 1])
        if hi > lo:
            means[b] = math.fsum(gold[lo:hi].tolist()) / (hi - lo)
        else:
            fallback.append(b)
    return GoldReps(means, tuple(fallback))


# =============================================================================
# 블록 4: 피해자 레벨 인덱싱 및 어시스트 풀 연동
# =============================================================================
def _record_kill_levels(rec) -> tuple:
    """``(kill table, victim levels)`` of one record: the kills as S1 stores them and, per kill, the level of its victim at the kill ms.

    (VAL-1)의 킬 표와 킬마다의 피해자 레벨을 계산한다(식은 kill_levels_for_record에 있다).
    """
    kt = units.kill_table(rec)
    index = traits.LevelIndex(rec)
    level = np.empty(len(kt.t), dtype=np.int16)
    for i, (t, victim) in enumerate(zip(kt.t.tolist(), kt.victim.tolist())):
        level[i] = index.at(t)[victim - 1]
    return kt, level


def kill_levels_for_record(rec) -> np.ndarray:
    """(n_kills,) int16: the level of the victim of each CHAMPION_KILL of a ``MatchRecord`` at the kill, in record (S1) order.

    The level is the highest LEVEL_UP level of the victim at or before the kill ms (1 if none): a LEVEL_UP on the ms of the kill is already
    counted (``traits.LevelIndex``). An execution has a victim too, so every kill gets a level.

    수식
    ----
    (VAL-1)  lvl(킬) = max( {1} ∪ { L : 킬 피해자의 LEVEL_UP이 레벨 L을 올린 시각 ≤ 킬 시각 } )

      기호    | 뜻                            | 단위 | 범위 | 코드 변수
      lvl(킬) | 킬 순간 피해자의 레벨         | 레벨 | 1–18 | 반환값의 킬 번호째 원소
      L       | 피해자의 LEVEL_UP이 올린 레벨 | 레벨 | 1–18 | traits.LevelIndex의 사건
      시각    | 킬의 시각, LEVEL_UP의 시각    | ms   | ≥ 0  | 킬 표의 t, 사건의 timestamp

    해석
    ----
    식 (VAL-1)은 킬 순간에 피해자가 가진 레벨, 곧 그 시각까지 올린 가장 높은 레벨이며, 킬과 같은 ms의 레벨 올림은 이미 반영하고 레벨 올림이 하나도 없으면 1이다. 도움 풀의 기본 현상금이
    피해자의 레벨로 정해지므로 필요하며(계획 P3), 처형에도 피해자가 있어 모든 킬이 레벨을 가진다.

    근거
    ----
    계획 P3(도움 풀은 피해자 레벨의 기본 현상금), Task 4(피해자 레벨; 같은 ms의 레벨 올림은 반영). 같은 규약의 원본은 모듈 머리의 LEGACY death_intervals 줄이다.
    """
    return _record_kill_levels(rec)[1]


def _check_workers(workers) -> None:
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError(f'workers must be an int of at least 1, got {workers!r}')


def _level_task(item, records):
    """One match in a worker: ``(row, times, victims, levels)``, or ``(row, None, None, None)`` for a file that cannot be read.

    ``MatchFormatError``, ``MatchRejected`` and ``HoldoutRefused`` carry no key and go on as they are. Any other error is looked at the way
    ``store_build`` looks at the errors of its workers: when the text of the error or of any error in its chain of causes and contexts
    holds the match key, or the game id on its own (the number after the last underscore of the key), it is raised again as a ValueError
    with the key replaced by <key> and the id by <id>, outside the ``except`` clause so that no cause or context keeps them.
    """
    row, patch, key = item
    caught = None
    try:
        kt, level = _record_kill_levels(records(key, patch))
        return row, kt.t, kt.victim, level
    except reader.MatchReadError:
        return row, None, None, None
    except (reader.MatchFormatError, reader.MatchRejected, reader.HoldoutRefused):
        raise
    except Exception as e:                      # anything else is looked at for the key below
        caught = e
    try:
        text = str(caught)
    except Exception:
        text = ''
    # The helpers of the S2 build (the chain text, the key and its game id, the scrub). Imported here and not at module level: only a worker
    # that has met an unexpected error needs them, and ``store_build`` is the heavy end of the S2 chain (E6, see ``kill_levels``).
    from c_ept.data import store_build
    if any(part in store_build._chain_text(caught) for part in store_build._key_parts(key)):
        raise ValueError(f'{type(caught).__name__}: {store_build._scrub(text, key)} (match row {row}, patch {patch})') from None
    raise caught


def kill_levels(s1, roles=LABEL_ROLES, workers: int = 8, *, records=reader.read_match, report: dict | None = None) -> np.ndarray:
    """(n_kills,) int16, aligned with ``kills/*`` of the S1 store: the victim level of every kill of the matches of ``roles``, -1 elsewhere.

    ``roles`` (FIT, STOP, DEV; ``flowprice`` refuses a holdout role or one that stage S3 does not use, before anything is read) are read
    from the raw files with ``records(key, patch)`` (``reader.read_match``) in a pool of ``workers`` processes (1: in this process). E6
    first: ``run_s1.require_no_strata()`` raises ``StrataRunning`` while a model server runs, before any file is opened. The kills of a raw
    match must be those of the S1 store (same count, times and victims); if not, ValueError names the match row (never the key). A file
    that cannot be read (``MatchReadError``) is counted, its kills keep -1, and a RuntimeWarning says how many; ``unit_values`` then refuses
    them. ``report``, if given, is filled with ``matches``, ``unreadable``, ``kills_read`` and ``kills_unread``.

    역할 roles의 경기마다 (VAL-1)의 피해자 레벨을 원시 파일에서 읽어 S1 킬 순서로 모으는 단계이다. 값을 매기지 않는 경기의 킬은 −1로 두어 레벨이 없음을 표시하고 읽지 못한 파일의 킬도 −1로 남으며,
    값이 매겨진 경기에 −1인 킬이 있으면 unit_values가 거부한다(판정 R12). 원시 파일을 읽으므로 모델 서버가 도는 동안에는 시작하지 않는다(계획 전역 제약의 E6).
    """
    codes = flowprice._role_codes(roles)
    _check_workers(workers)
    from c_ept.workflows import run_s1                      # inside the function: no module-level path to the S1 runner (E6)
    run_s1.require_no_strata()
    role = np.asarray(s1.array('matches/role'))
    patch = np.asarray(s1.array('matches/patch'), dtype=np.int64)
    kill_off = np.asarray(s1.array('matches/kill_off'), dtype=np.int64)
    key = s1.array('matches/key')
    rows = np.flatnonzero(np.isin(role, codes))
    items = [(int(m), C.PATCHES[int(patch[m])], bytes(key[m]).decode('ascii')) for m in rows]
    out = np.full(int(kill_off[-1]), -1, dtype=np.int16)
    s1_t, s1_victim = s1.array('kills/t'), s1.array('kills/victim')
    unreadable = kills_read = kills_unread = 0
    with run_s1._mapper(workers) as mapper:
        for m, t, victim, level in mapper(functools.partial(_level_task, records=records), items):
            lo, hi = int(kill_off[m]), int(kill_off[m + 1])
            if t is None:
                unreadable += 1
                kills_unread += hi - lo
                continue
            if len(t) != hi - lo or not (np.array_equal(t, s1_t[lo:hi]) and np.array_equal(victim, s1_victim[lo:hi])):
                raise ValueError(f'match row {m}: the raw match no longer agrees with the S1 kills ({hi - lo} kills in S1, {len(t)} read, '
                                 'or other times or victims): S1 was built from other files')
            out[lo:hi] = level
            kills_read += hi - lo
    if report is not None:
        report.update(matches=len(items), unreadable=unreadable, kills_read=kills_read, kills_unread=kills_unread)
    if unreadable:
        warnings.warn(f'{unreadable} of {len(items)} matches could not be read: their {kills_unread} kills keep the level -1',
                      RuntimeWarning, stacklevel=2)
    return out


# =============================================================================
# 블록 5: 교전 단위 가치 산출 엔진 (UnitValues, unit_values; 식 VAL-4, VAL-5)
# =============================================================================
@dataclass(frozen=True, eq=False)
class UnitValues:
    """The labels of the units of an S1 store, in S1 unit order (blue first in every pair); see the module text.

    ``kill_gold``, ``obj_value`` (attributed objectives) and ``conv_value`` (converted objectives) are (n, 2) float64; ``f0_unit`` and
    ``f0_conv`` (n,) float64; ``winner`` int8, ``sure`` bool; ``gold_bin`` (n, 2) int8 (-1: the team has no kill in the unit); ``n_kills``
    (n, 2) int32 (champion kills by team; an execution is no kill of a team); ``valued`` (n,) bool (the unit is in a match of the valued
    roles; else the float labels are NaN, winner 0, sure False, gold_bin -1, n_kills 0); ``match_valued`` (n_matches,) bool.

    각 필드의 식은 unit_values에 있다: kill_gold와 n_kills (VAL-2), obj_value와 conv_value (VAL-3), f0_unit과 f0_conv (VAL-4), winner와 sure (VAL-5),
    gold_bin (VAL-6).
    """
    kill_gold: np.ndarray
    obj_value: np.ndarray
    conv_value: np.ndarray
    f0_unit: np.ndarray
    f0_conv: np.ndarray
    winner: np.ndarray
    sure: np.ndarray
    gold_bin: np.ndarray
    n_kills: np.ndarray
    valued: np.ndarray
    match_valued: np.ndarray


def _valued_matches(s1, roles) -> np.ndarray:
    """(n_matches,) bool: the matches whose role is in ``roles``. ValueError for a holdout role, a role stage S3 does not use or no role."""
    return np.isin(np.asarray(s1.array('matches/role')), flowprice._role_codes(roles))


def role_units(s1, roles) -> np.ndarray:
    """(n_units,) bool: the units of the matches of ``roles`` (for example ``('FIT',)`` for ``gold_bin_means``).

    (VAL-7)의 항목을 낼 교전 단위를 경기의 역할로 고르는 마스크이다.
    """
    return _valued_matches(s1, roles)[np.asarray(s1.array('units/match'), dtype=np.int64)]


def _check_rules(rules) -> None:
    if rules is not None and not isinstance(rules, rulepay.RuleTable):
        raise TypeError('rules must be a RuleTable (RuleTable.load(path, patches=...)) or None for the built-in P3 rules')


def _check_levels(levels, n_kills: int) -> np.ndarray:
    a = np.asarray(levels)
    if a.ndim != 1 or len(a) != n_kills or a.dtype.kind not in 'iu':
        raise ValueError(f'kill_levels must be a 1-d integer array with one entry for each of the {n_kills} S1 kills (value.kill_levels)')
    return a.astype(np.int64)


@dataclass(frozen=True, eq=False)
class _PaidKills:
    """The champion kills of the valued matches, in S1 order: the match row, the unit the kill claims (``kills/unit``, unchecked), the team
    that scored it (0 blue, 1 red) and its gold (bounty + shutdown + assist pool)."""
    match: np.ndarray
    unit: np.ndarray
    side: np.ndarray
    gold: np.ndarray


def _kill_match(s1) -> np.ndarray:
    off = np.asarray(s1.array('matches/kill_off'), dtype=np.int64)
    return np.repeat(np.arange(len(off) - 1, dtype=np.int64), np.diff(off))


def _paid_kills(s1, rules, levels, match_valued) -> _PaidKills:
    """The kill gold of the champion kills of the valued matches. ValueError for a kill of a valued match without a victim level (1 .. 18),
    for a killer on neither team, and for what ``rulepay.assist_pools`` refuses (a patch the rules do not serve, a negative bounty).

    (VAL-2)의 킬마다의 골드 bounty + shutdown + pool을 챔피언 킬러가 있는 킬에 대해 계산한다(식은 unit_values에 있다).
    """
    kill_match = _kill_match(s1)
    levels = _check_levels(levels, len(kill_match))
    in_valued = match_valued[kill_match]
    bad = in_valued & ((levels < 1) | (levels > traits.MAX_LEVEL))
    if bad.any():
        raise ValueError(f'{int(bad.sum())} kills of {len(np.unique(kill_match[bad]))} valued matches have no victim level in 1 .. '
                         f'{traits.MAX_LEVEL} (value.kill_levels gives -1 outside the roles it is asked for and for a match it could not '
                         'read): run it for the roles of these labels')
    killer = np.asarray(s1.array('kills/killer'), dtype=np.int64)
    use = np.flatnonzero(in_valued & _is_pid(killer))                      # an execution (no champion killer) pays nobody and has no pool
    match = kill_match[use]
    team = np.asarray(s1.array('matches/team'), dtype=np.int64)[match, killer[use] - 1]
    if not np.isin(team, (BLUE, RED)).all():
        raise ValueError(f'a killer is on a team that is neither blue ({BLUE}) nor red ({RED}) in matches/team')
    bounty = np.asarray(s1.array('kills/bounty'), dtype=np.float64)[use]
    shutdown = np.asarray(s1.array('kills/shutdown'), dtype=np.float64)[use]
    assisters = _POPCOUNT[np.asarray(s1.array('kills/amask'), dtype=np.int64)[use] & _ALL_PIDS].astype(np.int64)
    t_ms = np.asarray(s1.array('kills/t'), dtype=np.int64)[use]
    patch = np.array(C.PATCHES)[np.asarray(s1.array('matches/patch'), dtype=np.int64)][match]
    pool = rulepay.assist_pools(bounty, levels[use], t_ms, patch, assisters, rules) if len(use) else np.zeros(0)
    return _PaidKills(match=match, unit=np.asarray(s1.array('kills/unit'), dtype=np.int64)[use], side=(team == RED).astype(np.int64),
                      gold=bounty + shutdown + pool)


@dataclass(frozen=True, eq=False)
class _Objectives:
    """The objectives of an S1 store: the match row, the price group (-1: not known), the price, the taker (0 blue, 1 red, -1 unknown), the
    unit it is attributed to and the unit it is converted to (-1: none), and the S1 kind (for the cells of ``units/O``)."""
    match: np.ndarray
    group: np.ndarray
    price: np.ndarray
    taker: np.ndarray
    unit: np.ndarray
    conv_unit: np.ndarray
    kind: np.ndarray


def _objectives(s1, table: np.ndarray) -> _Objectives:
    """(VAL-3)의 오브젝트마다의 묶음 ψ와 가격 price_ψ를 S1 표에서 만든다(식은 unit_values에 있다)."""
    off = np.asarray(s1.array('matches/obj_off'), dtype=np.int64)
    kind = np.asarray(s1.array('objectives/kind'), dtype=np.int64)
    sub = np.asarray(s1.array('objectives/sub'), dtype=np.int64)
    group = flowprice.group_of(kind, sub).astype(np.int64)
    return _Objectives(match=np.repeat(np.arange(len(off) - 1, dtype=np.int64), np.diff(off)), group=group,
                       price=np.where(group >= 0, table[np.clip(group, 0, _N_GROUPS - 1)], 0.0),
                       taker=np.asarray(s1.array('objectives/taker'), dtype=np.int64),
                       unit=np.asarray(s1.array('objectives/unit'), dtype=np.int64),
                       conv_unit=np.asarray(s1.array('objectives/conv_unit'), dtype=np.int64), kind=kind)


def _cells(unit: np.ndarray, taker: np.ndarray, price: np.ndarray, mask: np.ndarray, n_units: int) -> np.ndarray:
    """(n_units, 2): the summed ``price`` of the objectives in ``mask`` by unit and taker (blue, red); objectives without a taker of
    blue or red are not on a team and are left out.

    (VAL-3)의 합 obj_it, conv_it를 계산한다(식은 unit_values에 있다).
    """
    use = mask & ((taker == 0) | (taker == 1))
    return _wbincount(unit[use] * 2 + taker[use], weights=price[use], minlength=2 * n_units).reshape(n_units, 2)


def _objective_unit_check(unit: np.ndarray, match: np.ndarray, unit_match: np.ndarray, what: str, valued_obj: np.ndarray) -> None:
    """ValueError unless every unit an objective points to exists and belongs to the objective's own match."""
    claimed = valued_obj & (unit >= 0)
    if (unit[claimed] >= len(unit_match)).any() or (unit_match[unit[claimed]] != match[claimed]).any():
        raise ValueError(f'{what}: an objective points to a unit that does not exist or belongs to another match (S1 store not consistent)')


def unit_values(s1, prices, rules, kill_levels, roles=LABEL_ROLES) -> UnitValues:
    """The labels of every unit of the S1 store ``s1`` (see the module text).

    ``prices``: ``load_prices`` output. ``rules``: a ``rulepay.RuleTable`` (``RuleTable.load(path, patches=rulepay.P3_PATCHES)``) or None
    for the built-in P3 rules. ``kill_levels``: ``value.kill_levels`` (aligned with ``kills/*``). The units of the matches whose role is in
    ``roles`` (default FIT, STOP and DEV) are valued; ValueError if one of their kills has no victim level, if a kill or objective points
    to a unit of another match or to none, or if the prices or rules are unusable. The result's arrays are read only.

    수식
    ----
    (VAL-2)  g_it = Σ_{킬 ∈ (i, t)} (bounty + shutdown + pool),      k_it = #{킬 ∈ (i, t)}
             (i, t) = 교전 단위 i에 속하고 킬러가 팀 t인 챔피언 킬이다. pool은 도움이 하나 이상일 때만 있는 도움 풀(rulepay.assist_pools)이다.
    (VAL-3)  obj_it = Σ_{귀속 오브젝트 ∈ (i, t)} price_ψ,      conv_it = Σ_{전환 오브젝트 ∈ (i, t)} price_ψ
             (i, t) = 교전 단위 i에 귀속(또는 전환)되고 가져간 팀이 t인 오브젝트이다. ψ는 그 오브젝트의 묶음이며 묶음을 모르면 price = 0이다.
    (VAL-4)  f₀_i = (g_{i,blue} + obj_{i,blue}) − (g_{i,red} + obj_{i,red}),      f₀conv_i = conv_{i,blue} − conv_{i,red}
             값이 매겨지지 않은 단위는 g, obj, conv, f₀가 NaN이고 W = 0, 확실 = 거짓, 킬 골드 구간 = −1이다.

      기호                   | 뜻                                              | 단위     | 범위              | 코드 변수
      i, t                   | 교전 단위 번호, 팀 (blue 먼저, 그다음 red)      | —        | 1, …, n_units     | 행, 열 0과 1
      bounty, shutdown, pool | 킬 하나의 기록 bounty, shutdown bounty, 도움 풀 | 골드     | ≥ 0               | kills/bounty, kills/shutdown, rulepay
      g_it, k_it             | 팀의 킬 골드, 챔피언 킬의 수                    | 골드, 개 | ≥ 0; 0, 1, …      | kill_gold, n_kills
      ψ, price_ψ             | 오브젝트의 묶음, 그 묶음의 가격                 | —, 골드  | 12가지; 부호 있음 | prices, flowprice.group_of
      obj_it, conv_it        | 귀속, 전환 오브젝트의 가격 합                   | 골드     | 유한, 부호 있음   | obj_value, conv_value
      f₀_i, f₀conv_i         | 교전 단위의 f₀(K, O), f₀(전환) (파랑 − 빨강)    | 골드     | 유한, 부호 있음   | f0_unit, f0_conv

    해석
    ----
    식 (VAL-4)의 f₀는 파랑이 얻은 킬 골드와 오브젝트 가격에서 빨강의 것을 뺀 골드 단위의 가치이고, 킬 골드는 도움 풀을 이미 포함하므로 도움을 따로 더하지 않으며 오브젝트는 흐름 가격이나
    즉시값이다(계획 P3, P4; 설계서 8절). 챔피언 킬러가 없는 킬(처형)은 어느 팀의 합에도 없고 도움 풀도 없다(계획 P3). 오브젝트는 귀속과 전환을 나누어 더하므로 합성의 f₀(K, O) + f₀(전환)과 같은
    분해이고, 전환은 f₀conv에만 들어가 승자에 쓰이지 않는다(설계서 3.3절 D4, 8절). 킬이나 오브젝트가 다른 경기의 교전 단위나 없는 교전 단위를 가리키면 저장소가 일관되지 않은 것이므로 거부한다.
    값이 매겨지지 않은 경기의 단위는 NaN으로 두어 값이 없음을 숫자로 읽지 않게 한다(공통 가정 (가)).

    근거
    ----
    계획 P3(도움 풀을 포함한 킬 골드, 처형은 지급 없음), P4(K는 킬과 킬 골드), Task 4(교전 단위의 f₀ 정답), 계획 전역 제약의 f₀ 항(흐름 가격에 규칙 지급액을 더하지 않는다), 설계서 3.3절(귀속과 전환)과
    8절(f₀), 판정 R12(레벨 없는 킬은 거부).
    """
    table = _price_table(prices)
    _check_rules(rules)
    match_valued = _valued_matches(s1, roles)
    unit_match = np.asarray(s1.array('units/match'), dtype=np.int64)
    n_units = len(unit_match)
    valued = match_valued[unit_match]
    paid = _paid_kills(s1, rules, kill_levels, match_valued)
    if ((paid.unit < 0) | (paid.unit >= n_units)).any() or (unit_match[paid.unit] != paid.match).any():
        raise ValueError('a champion kill belongs to no unit, or to a unit of another match (kills/unit): S1 store not consistent')
    # 식 (VAL-2): 킬마다의 골드를 칸 (교전 단위, 팀)으로 더하고 같은 칸의 챔피언 킬 수를 센다. 칸 번호는 단위 × 2 + 팀이고 blue가 먼저이다.
    cell = paid.unit * 2 + paid.side
    kill_gold = _wbincount(cell, weights=paid.gold, minlength=2 * n_units).reshape(n_units, 2)
    n_kills = np.bincount(cell, minlength=2 * n_units).reshape(n_units, 2)
    obj = _objectives(s1, table)
    valued_obj = match_valued[obj.match]
    _objective_unit_check(obj.unit, obj.match, unit_match, 'objectives/unit', valued_obj)
    _objective_unit_check(obj.conv_unit, obj.match, unit_match, 'objectives/conv_unit', valued_obj)
    # 식 (VAL-3): 귀속된 오브젝트와 전환된 오브젝트의 가격을 칸마다 더한다.
    obj_value = _cells(obj.unit, obj.taker, obj.price, valued_obj & (obj.unit >= 0), n_units)
    conv_value = _cells(obj.conv_unit, obj.taker, obj.price, valued_obj & (obj.conv_unit >= 0), n_units)
    # 값이 매겨지지 않은 단위는 NaN으로 덮어 숫자로 읽히지 않게 한다(공통 가정 (가)).
    kill_gold[~valued], obj_value[~valued], conv_value[~valued] = np.nan, np.nan, np.nan
    # 식 (VAL-4): 파랑의 킬 골드와 귀속 오브젝트 몫에서 빨강의 것을 빼고, 전환은 f₀conv에 따로 둔다.
    f0_unit = (kill_gold[:, 0] + obj_value[:, 0]) - (kill_gold[:, 1] + obj_value[:, 1])
    f0_conv = conv_value[:, 0] - conv_value[:, 1]
    winner, sure = np.zeros(n_units, dtype=np.int8), np.zeros(n_units, dtype=bool)
    winner[valued], sure[valued] = winner_of(f0_unit[valued])
    # 식 (VAL-6): 킬이 하나 이상인 팀만 킬 골드 구간 b를 갖고 나머지는 −1이다.
    bins = np.full((n_units, 2), -1, dtype=np.int8)
    has_kill = valued[:, None] & (n_kills >= 1)
    bins[has_kill] = gold_bin(kill_gold[has_kill])
    out = UnitValues(kill_gold=kill_gold, obj_value=obj_value, conv_value=conv_value, f0_unit=f0_unit, f0_conv=f0_conv, winner=winner,
                     sure=sure, gold_bin=bins, n_kills=narrow('n_kills', n_kills, np.int32), valued=valued, match_valued=match_valued)
    for name in _UNIT_FIELDS + ('match_valued',):
        getattr(out, name).setflags(write=False)
    return out


# =============================================================================
# 블록 6: 향후 예측 시계별 누적 가치 계산 (v_start; 식 VAL-6)
# =============================================================================
def _check_uv(s1, uv: UnitValues) -> int:
    """The number of units; ValueError if the labels do not have the S1 store's units and matches."""
    n_units, n_matches = len(s1.array('units/match')), len(s1.array('matches/role'))
    for name in _UNIT_FIELDS:
        if len(getattr(uv, name)) != n_units:
            raise ValueError(f'the labels hold {len(getattr(uv, name))} units in {name}, the S1 store {n_units}: made from another store?')
    if len(uv.match_valued) != n_matches:
        raise ValueError(f'the labels hold {len(uv.match_valued)} matches, the S1 store {n_matches}: made from another store?')
    return n_units


def _horizons(horizons_ms) -> tuple:
    try:
        hs = tuple(horizons_ms)
    except TypeError:
        raise ValueError('horizons_ms must be a sequence of whole milliseconds') from None
    if not hs or any(isinstance(h, bool) or not isinstance(h, (int, np.integer)) or h <= 0 for h in hs):
        raise ValueError('horizons_ms must hold positive whole milliseconds')
    return tuple(int(h) for h in hs)


def v_start(s1, uv: UnitValues, horizons_ms=C.EVAL_H_MS) -> np.ndarray:
    """(n_rows, len(horizons_ms)) float64, in the order of the S1 frame rows: V_start of each row for each horizon H.

    The sum of ``f0_unit + f0_conv`` over the in-scope units of the row's match with a first kill f in (t0, min(t0 + H, hi)], t0 and hi
    from ``frame_rows``. Found by searching the sorted first-kill times of each match; the ``onsets`` table is not read. A row of a match
    that is not valued has NaN. A valued match without a unit in the window gives 0.

    수식
    ----
    (VAL-8)  V_start(u, t₀, H) = Σ_{i: 경기 u의 범위 안 교전, t₀ < f_i ≤ min(t₀ + H, hi)} (f₀_i + f₀conv_i)

      기호              | 뜻                                           | 단위  | 범위                  | 코드 변수
      u, t₀, hi         | 분 프레임 행의 경기, 기준 시각, 관측 창의 끝 | —, ms | —                     | frame_rows의 match, t0, hi
      H                 | 평가 지평                                    | ms    | config.EVAL_H_MS의 값 | horizons_ms
      f_i               | 교전 단위 i의 첫 킬의 시각                   | ms    | ≥ 0                   | units/f
      f₀_i, f₀conv_i    | 교전 단위의 f₀(K, O), f₀(전환) (파랑 − 빨강) | 골드  | 유한, 부호 있음       | uv.f0_unit, uv.f0_conv
      V_start(u, t₀, H) | 행과 지평마다의 값                           | 골드  | 부호 있음 또는 NaN    | 반환값의 열

    해석
    ----
    식 (VAL-8)은 첫 킬이 (t₀, t₀ + H] 안에 있는 교전들의 가치를 더하며(설계서 3.4절), 관측 창의 끝 hi에서 잘라 경기 끝에 가까운 행은 더 짧은 창을 본다(계획 전역 제약의 V_start 항). 교전의 뒤 흐름(전환)은 그 교전의 첫 킬이 창
    안이면 t₀ + H를 넘어서도 통째로 센다(설계서 3.4절). 첫 킬이 t₀ 이전인 진행 중 교전은 이 행의 대상이 아니다(설계서 3.1절). 이 값은 같은 행의 발생 표에서 구간 번호가 H/10 s보다 작은 교전을
    더한 값과 같아야 하며 이 함수는 발생 표를 읽지 않고 구한다(계획 전역 제약의 V_start 항). 범위 밖 교전은 합에 넣지 않고(설계서 2절의 범위), 값이 매겨지지 않은
    경기의 행은 NaN이다(공통 가정 (가)).

    근거
    ----
    설계서 3.4절(V_start의 정의)과 3.1절(첫 킬이 기준점), 계획 전역 제약의 V_start 항(발생 표와 같아야 한다)과 상수(범위).
    """
    horizons = _horizons(horizons_ms)
    _check_uv(s1, uv)
    unit_match = np.asarray(s1.array('units/match'), dtype=np.int64)
    f = np.asarray(s1.array('units/f'), dtype=np.int64)
    in_scope = np.asarray(s1.array('units/in_scope'), dtype=bool)
    row_match = np.asarray(s1.array('frame_rows/match'), dtype=np.int64)
    t0 = np.asarray(s1.array('frame_rows/t0'), dtype=np.int64)
    hi = np.asarray(s1.array('frame_rows/hi'), dtype=np.int64)
    n_rows = len(t0)
    if (f.size and (f.min() < 0 or f.max() >= _SPAN)) or (n_rows and (t0.min() < 0 or hi.max() >= _SPAN)):
        raise ValueError('a time lies outside 0 .. 2^40 ms')
    # 식 (VAL-8)의 교전 집합: 경기 번호와 첫 킬 시각을 한 정렬 키로 묶어, 행마다 첫 킬이 (t₀, hi]에 드는 교전의 구간을 이진 탐색으로 찾고 범위 밖 교전은 뺀다.
    key = unit_match * _SPAN + f
    if key.size > 1 and (np.diff(key) < 0).any():
        raise ValueError('the units are not ordered by match and first-kill time')
    first = np.searchsorted(key, row_match * _SPAN + t0, side='right')               # units before this have f <= t0
    stop = np.searchsorted(key, row_match * _SPAN + hi, side='right')                # units before this have f <= hi
    count = np.maximum(stop - first, 0)
    row = np.repeat(np.arange(n_rows, dtype=np.int64), count)
    unit = np.arange(int(count.sum()), dtype=np.int64) - np.repeat(np.cumsum(count) - count, count) + np.repeat(first, count)
    keep = in_scope[unit]
    row, unit = row[keep], unit[keep]
    # 식 (VAL-8)의 합: 지평 H마다 첫 킬이 t₀ + H 이내인 교전만 남겨 f₀ + f₀conv를 행별로 더한다. 값이 매겨지지 않은 경기의 행은 마지막에 NaN으로 덮는다.
    total = uv.f0_unit + uv.f0_conv
    out = np.empty((n_rows, len(horizons)), dtype=np.float64)
    for j, h in enumerate(horizons):
        within = f[unit] <= t0[row] + h
        out[:, j] = _wbincount(row[within], weights=total[unit[within]], minlength=n_rows)
    out[~uv.match_valued[row_match]] = np.nan
    return out


# =============================================================================
# 블록 7: 엄밀한 경제/오브젝트 보존 회계 감사 (accounting; 판정 R11, 식 VAL-7 ~ VAL-9)
# =============================================================================
def _objective_identity(s1, obj: _Objectives, valued_obj, match_valued, unit_match, valued_unit) -> dict:
    """The objective identity per (match, group, taker) in counts, and the labels against the cells ``units/O`` and ``units/conv`` of S1.

    Every objective of a valued match is in one cell (its match, its group - or none - and its taker - or none), and in one of three
    places: attributed (``unit`` >= 0), converted (``conv_unit`` >= 0) or rest (neither). The identity is attributed + converted + rest =
    all in every cell; an objective that is both attributed and converted breaks it. The second check recounts the attributed and the
    converted objectives of every valued unit by kind and taker and compares them with the S1 columns (they are made by another code path).

    (VAL-9)의 오브젝트 항등식과 units/O, units/conv의 재집계를 계산한다(식은 accounting에 있다).
    """
    n_matches, n_units = len(match_valued), len(unit_match)
    attributed, converted = obj.unit >= 0, obj.conv_unit >= 0
    rest = ~attributed & ~converted
    n_group = _N_GROUPS + 1                                                              # the last column: no group
    taker3 = np.where((obj.taker == 0) | (obj.taker == 1), obj.taker, 2)               # the last value: no taker
    cell = (obj.match * n_group + np.where(obj.group >= 0, obj.group, _N_GROUPS)) * 3 + taker3

    def counted(mask):
        return np.bincount(cell[mask & valued_obj], minlength=n_matches * n_group * 3)
    n_all, n_att, n_conv, n_rest = (counted(m) for m in (np.ones(len(cell), dtype=bool), attributed, converted, rest))
    per_match = np.abs(n_att + n_conv + n_rest - n_all).reshape(n_matches, -1).max(axis=1) if n_matches else np.zeros(0, np.int64)
    per_match = np.array(per_match, dtype=np.int64)
    counted_kind = (obj.kind >= 0) & (obj.kind < len(C.O_KINDS)) & ((obj.taker == 0) | (obj.taker == 1))
    per_unit = np.zeros(n_units, dtype=np.int64)
    for name, claims in (('O', obj.unit), ('conv', obj.conv_unit)):
        use = valued_obj & (claims >= 0) & counted_kind
        recount = np.bincount(claims[use] * _N_CELLS + 2 * obj.kind[use] + obj.taker[use], minlength=n_units * _N_CELLS)
        stored = np.asarray(s1.array(f'units/{name}'), dtype=np.int64).reshape(n_units, _N_CELLS)
        if n_units:
            gap = np.abs(recount.reshape(n_units, _N_CELLS) - np.where(valued_unit[:, None], stored, 0))
            per_unit = np.maximum(per_unit, gap.max(axis=1))
    np.maximum.at(per_match, unit_match, per_unit)
    return {'per_match': per_match, 'rest': rest,
            'totals': {'all': int(n_all.sum()), 'attributed': int(n_att.sum()), 'converted': int(n_conv.sum()), 'rest': int(n_rest.sum()),
                       'both': int((attributed & converted & valued_obj).sum()), 'unknown_group': int((valued_obj & (obj.group < 0)).sum()),
                       'unknown_taker': int((valued_obj & (obj.taker != 0) & (obj.taker != 1)).sum())}}


def _kill_identity(s1, paid: _PaidKills, uv: UnitValues, match_valued, unit_match, valued_unit) -> dict:
    """The kill identity per (match, team) in counts, and the labels against the champion kills of ``units/K`` of S1.

    The champion kills of the valued matches are in units (``kills/unit`` points to a unit of the kill's own match) as far as the
    store says; the labels' ``n_kills`` must be the recount of those by unit and team, and ``units/K`` (kills_b, kills_r).

    (VAL-9)의 킬 항등식과 n_kills, units/K의 재집계를 계산한다(식은 accounting에 있다).
    """
    n_matches, n_units = len(match_valued), len(unit_match)
    in_range = (paid.unit >= 0) & (paid.unit < n_units)
    in_unit = in_range.copy()
    in_unit[in_range] = unit_match[paid.unit[in_range]] == paid.match[in_range]            # the unit exists and is in the kill's own match
    cell = paid.match * 2 + paid.side
    per_match = np.abs(np.bincount(cell, minlength=n_matches * 2) - np.bincount(cell[in_unit], minlength=n_matches * 2))
    per_match = np.array(per_match.reshape(n_matches, 2).max(axis=1), dtype=np.int64)
    recount = np.bincount(paid.unit[in_unit] * 2 + paid.side[in_unit], minlength=2 * n_units).reshape(n_units, 2)
    k_stored = np.where(valued_unit[:, None], np.asarray(s1.array('units/K'), dtype=np.int64)[:, :2], 0)
    per_unit = np.maximum(np.abs(recount - uv.n_kills).max(axis=1), np.abs(k_stored - uv.n_kills).max(axis=1))
    np.maximum.at(per_match, unit_match, per_unit)
    killer = np.asarray(s1.array('kills/killer'), dtype=np.int64)
    in_valued = match_valued[_kill_match(s1)]
    return {'per_match': per_match, 'in_unit': in_unit,
            'totals': {'champion': int(len(paid.gold)), 'in_units': int(in_unit.sum()),
                       'executions': int((in_valued & ~_is_pid(killer)).sum()), 'all': int(in_valued.sum())}}


def _value_identity(paid: _PaidKills, obj: _Objectives, rest, uv: UnitValues, valued_obj, unit_match, valued_unit) -> dict:
    """The same two identities in gold (``math.fsum`` totals, per-match sums from numpy): the gold of the kills against the kill gold of
    the units, and every objective's price against attributed + converted + rest. Gold sums depend on the order of the additions, so
    these are a diagnostic and not expected to be exactly 0.

    (VAL-9)의 골드 항등식을 계산한다(식은 accounting에 있다).
    """
    n_matches = len(uv.match_valued)
    by_unit = unit_match[valued_unit]

    def fsum(a):
        return math.fsum(a.tolist())
    gold_raw = _wbincount(paid.match * 2 + paid.side, weights=paid.gold, minlength=n_matches * 2).reshape(n_matches, 2)
    gold_lab = np.stack([_wbincount(by_unit, weights=uv.kill_gold[valued_unit, s], minlength=n_matches) for s in (0, 1)], axis=1)
    kill_gold_error = np.abs(gold_raw - gold_lab).max(axis=1)
    kills_raw = [fsum(paid.gold[paid.side == s]) for s in (0, 1)]
    kills_lab = [fsum(uv.kill_gold[valued_unit, s]) for s in (0, 1)]
    value_all = [fsum(obj.price[valued_obj & (obj.taker == s)]) for s in (0, 1)]
    value_rest = [fsum(obj.price[rest & valued_obj & (obj.taker == s)]) for s in (0, 1)]
    value_att = [fsum(uv.obj_value[valued_unit, s]) for s in (0, 1)]
    value_conv = [fsum(uv.conv_value[valued_unit, s]) for s in (0, 1)]
    object_error = np.zeros(n_matches)
    for s in (0, 1):
        mine = valued_obj & (obj.taker == s)
        raw = _wbincount(obj.match[mine], weights=obj.price[mine], minlength=n_matches)
        lab = (_wbincount(by_unit, weights=uv.obj_value[valued_unit, s] + uv.conv_value[valued_unit, s], minlength=n_matches)
               + _wbincount(obj.match[mine & rest], weights=obj.price[mine & rest], minlength=n_matches))
        object_error = np.maximum(object_error, np.abs(raw - lab))
    gaps = [abs(a - b) for a, b in zip(kills_raw, kills_lab)]                                # the kill gold of the units against the kills
    gaps += [abs(value_all[s] - (value_att[s] + value_conv[s] + value_rest[s])) for s in (0, 1)]    # every objective in one of three places
    worst = max(gaps + [float(kill_gold_error.max(initial=0.0)), float(object_error.max(initial=0.0))])
    return {'worst': float(worst), 'kill_gold_error': kill_gold_error, 'objective_value_error': object_error,
            'kill_gold': {'kills': kills_raw, 'units': kills_lab},
            'objective_value': {'all': value_all, 'attributed': value_att, 'converted': value_conv, 'rest': value_rest}}


def accounting(s1, uv: UnitValues, prices, rules, kill_levels, roles=LABEL_ROLES) -> dict:
    """The two identities of the design over the valued matches, with per-match and total figures (see the module text).

    Returns ``max_abs_error`` (float, the largest count-level difference; must be 0.0), ``max_abs_value_error`` (gold, a diagnostic),
    ``n_matches``, ``n_valued_matches``, ``total`` (plain numbers and lists: ``objectives`` all / attributed / converted / rest / both /
    unknown_group / unknown_taker, ``kills`` champion / in_units / executions / all, ``kill_gold`` of the kills and of the units by team,
    ``objective_value`` all / attributed / converted / rest by team) and ``per_match`` (arrays over the S1 matches:
    ``objective_count_error``, ``kill_count_error``, ``kill_gold_error``, ``objective_value_error``). A kill without a unit, an objective in
    two places or labels that do not match the S1 counts are not an exception but a non-zero error. ValueError if the labels are not those
    of this store and these roles, if an objective points to a unit of another match, or for levels, prices or rules that cannot be used.

    수식
    ----
    (VAL-9)  개수:  n_귀속 + n_전환 + n_나머지 = n_전체                    (경기 u, 묶음 ψ 또는 없음, 가져간 팀 또는 없음마다)
                    #{교전 단위에 든 챔피언 킬 (u, t)} = #{챔피언 킬 (u, t)}   (경기 u, 팀 t마다)
                    교전 단위마다 다시 센 귀속·전환 오브젝트와 챔피언 킬 = S1의 units/O, units/conv, units/K와 라벨의 n_kills
             max_abs_error = 위 차이들의 절댓값의 최댓값                        (0이어야 한다)
             골드:  max_abs_value_error = max(|킬 골드의 합 − 교전 단위의 kill_gold 합|, |오브젝트 가격 합 − (귀속 + 전환 + 나머지의 가격 합)|)
                    (팀마다의 전체 합과 경기·팀마다의 합 모두이다)

      기호                             | 뜻                                                          | 단위 | 범위    | 코드 변수
      n_귀속, n_전환, n_나머지, n_전체 | 칸의 오브젝트 수: 귀속된 것, 전환된 것, 둘 다 아닌 것, 전부 | 개   | 0, 1, … | n_att, n_conv, n_rest, n_all
      max_abs_error                    | 개수 수준의 가장 큰 차이                                    | 개   | ≥ 0     | max_abs_error
      max_abs_value_error              | 골드 수준의 가장 큰 차이                                    | 골드 | ≥ 0     | max_abs_value_error

    해석
    ----
    식 (VAL-9)는 오브젝트 하나가 교전 귀속, 전환, 나머지 가운데 정확히 한 곳에만 들고 챔피언 킬이 모두 어느 교전 단위에 든다는 분할의 항등식이며(설계서 2절: 모든 킬은 한 교전 단위에 든다; 3.3절 D4), 개수 수준에서는 정확히 맞으므로
    오차 0을 판정에 쓴다(계획 P7 (iii), 판정 R11). 골드 수준은 같은 분할의 가격 합이지만 부동소수점 덧셈은 묶는 순서에 따라 달라지므로 반올림 오차만 허용하며, 그 허용오차와 관문 판정 곁의 보고는 판정 R48이
    정한다. S1이 다른 길로 만든 열 units/O, units/conv, units/K와 맞대는 재집계는 정답 라벨이 S1과 같은 개수를 세는지의 교차 검사이고 그 차이도 max_abs_error에 들어간다. 킬이 어느 교전 단위에도 들지
    않거나 오브젝트가 두 곳에 있거나 라벨이 S1의 개수와 다른 것처럼 회계가 어긋나는 자료는 예외로 멈추지 않고 0이 아닌 오차로 알려 경기별로 보고하고 관문 (iii)이 판정하게 하며(판정 R11), 라벨이 이 저장소와 이 역할의 것이 아니거나 오브젝트가 다른 경기의 교전 단위를 가리키면 거부한다.
    묶음이나 가져간 팀을 모르는 오브젝트도 칸에 들어 있어 개수 항등식에 포함된다.

    근거
    ----
    판정 R11(회계 항등식은 개수에서 정확히 맞는다), R48(골드 수준은 반올림 오차만), 계획 P7 (iii)(회계 항등식의 오차), 설계서 3.3절(유일 귀속 D4)과 3.4절(중복 없음 D4).
    """
    table = _price_table(prices)
    _check_rules(rules)
    match_valued = _valued_matches(s1, roles)
    _check_uv(s1, uv)
    unit_match = np.asarray(s1.array('units/match'), dtype=np.int64)
    valued_unit = match_valued[unit_match]
    if not np.array_equal(valued_unit, uv.valued):
        raise ValueError('the labels value other units than the roles asked for in accounting')
    obj = _objectives(s1, table)
    valued_obj = match_valued[obj.match]
    _objective_unit_check(obj.unit, obj.match, unit_match, 'objectives/unit', valued_obj)
    _objective_unit_check(obj.conv_unit, obj.match, unit_match, 'objectives/conv_unit', valued_obj)
    paid = _paid_kills(s1, rules, kill_levels, match_valued)
    # 식 (VAL-9): 오브젝트 항등식, 킬 항등식, 골드 항등식을 각각 계산한다. max_abs_error는 개수 수준의 앞 둘에서만 만든다.
    objects = _objective_identity(s1, obj, valued_obj, match_valued, unit_match, valued_unit)
    kills = _kill_identity(s1, paid, uv, match_valued, unit_match, valued_unit)
    gold = _value_identity(paid, obj, objects['rest'], uv, valued_obj, unit_match, valued_unit)
    return {
        'max_abs_error': float(max(int(objects['per_match'].max(initial=0)), int(kills['per_match'].max(initial=0)))),
        'max_abs_value_error': gold['worst'], 'n_matches': len(match_valued), 'n_valued_matches': int(match_valued.sum()),
        'total': {'objectives': objects['totals'], 'kills': kills['totals'], 'kill_gold': gold['kill_gold'],
                  'objective_value': gold['objective_value']},
        'per_match': {'objective_count_error': objects['per_match'], 'kill_count_error': kills['per_match'],
                      'kill_gold_error': gold['kill_gold_error'], 'objective_value_error': gold['objective_value_error']}}


# =============================================================================
# 블록 8: 정답 라벨 저장소 원자적 파일 직렬화 (write_labels)
# =============================================================================
def write_labels(out_dir, s1, uv: UnitValues, v_start, gold_reps, provenance) -> Path:
    """Write the labels as a finalized ``store_core.Store`` at ``out_dir`` and return that folder.

    The arrays are ``labels/<name>`` for the names of ``LABEL_ARRAYS``, in the S1 row order of the units and of the frame rows
    (``v_start``: (n_rows, 3), the evaluation horizons; ``gold_reps``: (31,), the bin representatives). ``LABELS.json`` holds ``provenance``
    (a dict of JSON values with text keys: the hashes of the prices, the rules and the S1 files, ``runlog.code_state()``) and no key of
    ours. Everything is checked before a file is made. FileExistsError if ``out_dir`` already holds a store or a LABELS.json: a label
    store is never written over; ValueError if an array does not fit the S1 store; TypeError for provenance that is not such a dict.

    (VAL-2)–(VAL-8)이 만든 값을 정답 저장소에 쓰는 단계이며, 배열은 교전 단위와 분 프레임 행의 S1 행 순서이고 gold_reps만 구간 순서이다. 쓰기 전에 모든 배열의 모양과 자료형을 확인하고, 이미 있는 저장소를 덮어쓰지 않는 것은 불변 실행 폴더의 규칙에서 나온다(설계서 9절).
    """
    out = Path(out_dir)
    n_units = _check_uv(s1, uv)
    n_rows = len(s1.array('frame_rows/t0'))
    v, reps = np.asarray(v_start), np.asarray(gold_reps)
    if v.dtype != np.float64 or v.shape != (n_rows, len(C.EVAL_H_MS)):
        raise ValueError(f'v_start must be float64 of shape ({n_rows}, {len(C.EVAL_H_MS)}), got {v.dtype} {v.shape}')
    if reps.dtype != np.float64 or reps.shape != (N_GOLD_BINS,) or not np.isfinite(reps).all():
        raise ValueError(f'gold_reps must be {N_GOLD_BINS} finite float64 values, got {reps.dtype} {reps.shape}')
    if not isinstance(provenance, dict) or not all(isinstance(k, str) for k in provenance):
        raise TypeError('provenance must be a dict with text keys')
    text = json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + '\n'          # ValueError for NaN, TypeError for objects
    given = {'kill_gold': uv.kill_gold, 'obj_value': uv.obj_value, 'conv_value': uv.conv_value, 'f0_unit': uv.f0_unit,
             'f0_conv': uv.f0_conv, 'winner': uv.winner, 'sure': uv.sure, 'gold_bin': uv.gold_bin, 'v_start': v, 'gold_reps': reps}
    rows = {'v_start': n_rows, 'gold_reps': N_GOLD_BINS}                                       # every other array has one row for each unit
    arrays = {}
    for name, dtype in LABEL_ARRAYS:
        arrays[name] = np.ascontiguousarray(given[name])
        if arrays[name].dtype != np.dtype(dtype) or len(arrays[name]) != rows.get(name, n_units):
            raise ValueError(f'labels/{name} must have the dtype {dtype} and {rows.get(name, n_units)} rows')
    if (out / 'STORE.json').exists() or (out / 'LABELS.json').exists():
        raise FileExistsError(f'{out} already holds a label store: it is not written over')
    store_core.require_free_bytes(out, sum(a.nbytes for a in arrays.values()))
    out.mkdir(parents=True, exist_ok=True)
    s = store_core.Store(out)
    names = []
    for name, dtype in LABEL_ARRAYS:
        m = s.allocate(f'labels/{name}', arrays[name].shape, dtype)
        m[...] = arrays[name]
        m.flush()
        del m
        names.append(f'labels/{name}')
    s.mark_done('write', 0, names)
    store_core.write_atomic(out / 'LABELS.json', text)
    s.finalize()
    return out
