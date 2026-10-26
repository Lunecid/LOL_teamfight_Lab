r"""[모듈 개요 및 하네스 위치]
이 모듈은 교전 과정 모형(EPT)이 넘어서야 할 1차 고성능 기준선인 B1 그래디언트 부스팅 트리(LightGBM)
앙상블 및 다음 1분 발생 이진 분류기(B2)를 구현한다 (stage S3 Task 7).

A. 하네스 및 파이프라인 내 위치:
   - B0(baselines_b0, Task 6)와 동일한 머리 이름, 동일한 가린 및 열린 구조, 동일한 ComposerHeads 프로토콜,
     동일한 head_rows 데이터 행 규격을 공유한다 (판정 R18).
   - B0가 39개 원시 열의 로그 선형 및 다항 로짓 GLM인 반면, B1은 요약 통계량·팀 합계/최댓값·거리 등이 포함된
     b1 특징 테이블(features.b1, 약 440개 열)을 기반으로 LightGBM 부스팅 트리를 학습한다.
   - B1은 EPT가 넘어야 하는 정량적 하한선(floor)이자, 합성기(composer)의 2차 평가 기준선이다 (결정 P6).

B. 모듈 아키텍처 및 11대 블록 구성:
   - 블록 1: 전역 상수 및 고정 하이퍼파라미터 (FIXED_PARAMS, seed=1, 결정 P9)
   - 블록 2: 설계 행렬(Design Matrix) 슬롯 및 레이아웃 (SLOTS, Layout, make_layout; 식 B1-1)
   - 블록 3: 설계 행렬 및 범주형 열 유효성 검증 유틸리티 (_design_matrix, _check_categorical)
   - 블록 4: 부스팅 기저 클래스 (_Boosted) 및 조기 종료 (early stopping; 식 B1-5)
   - 블록 5: 포아송 부스팅 헤드 (_PoissonBoosted; 식 B1-2, B1-3 오프셋 및 제프리스 복구)
   - 블록 6: 다중 분류 부스팅 헤드 (_CategoricalBoosted; 식 B1-4 소프트맥스 및 균등 섞기)
   - 블록 7: 8대 개별 헤드 (M2Rate, M2Mark, M2Zone, M3Size, M3Kills, M3Gold, M3Obj, M4Conv)
   - 블록 8: 훈련 및 평가용 행 집합 추출기 (HeadRows)
   - 블록 9: B2 다음 1분 발생 이진 분류기 (B2Boosted, B2Rows; 식 B1-6 ~ B1-8)
   - 블록 10: 합성기 연동 프로토콜 어댑터 (_MaskedChain, ComposerHeads 구현)
   - 블록 11: 헤드 번들 및 파일 직렬화 엔진 (B1Heads, save_head, load_head; JSON + model text)

C. 설계 행렬(Design Matrix) 슬롯 및 텐서 규격 (식 B1-1):
   - M2Rate  : [b1 feat, q]                            -> (n, 18) 포아송 초당 첫 킬 발생률
   - M2Mark  : [b1 feat, q]                            -> (n, 52) 52개 표지 다중 분류 확률
   - M2Zone  : [b1 feat, q, m]                         -> (n, 10) 10개 구역 다중 분류 확률
   - M3Size  : [b1 feat, q, m, z]                      -> (n, 4) 4개 규모 범주 확률
   - M3Kills : [b1 feat, q, m, z, s]                   -> (n, 2) Blue 및 Red 기대 킬 수 (포아송)
   - M3Gold  : [b1 feat, q, m, z, s, layer]            -> (n, 2, 31) 31개 킬 골드 구간 확률
   - M3Obj   : [b1 feat, K, q, m, z, s]                -> (n, 18) 18개 오브젝트 칸별 기대치 (포아송)
   - M4Conv  : [b1 feat, q, m, z, K, O]                -> (n, 18) 18개 오브젝트 칸 초당 전환율

D. 핵심 수식 (B1-1 ~ B1-8):
   (B1-1)  설계 행        x_i = [b1 열 | K 블록 | O 블록 | 범주 열 q, m, z + 1, s, ξ − 1]
   (B1-2)  포아송 부스팅  log μ_{ic} = log e_i + γ_c + F_c(x_i),   γ_c = log(Σ_i y_{ic} / Σ_i e_i)
   (B1-3)  사건 없는 칸   Σ_i y_{ic} = 0이면 나무 없이 λ_c = 0.5 / Σ_i e_i (제프리스 사전분포 사후평균)
   (B1-4)  분류 부스팅    p^model_{ic} = softmax_{c∈A}(F_c(x_i)),   p_{ic} = (1 − ν) · p^model_{ic} + ν / C
   (B1-5)  조기 종료      R̂ = argmin_{R ≤ R_run} Loss_stop(R) (STOP 행 기준 인내 50 라운드)
   (B1-6)  B2 목표        y_i = 1[Σ_{q=0}^{5} n_q ≥ 1] (프레임 행 기준 향후 60초 내 발생 여부)
   (B1-7)  B2 확률        p_i = (1 − ν) · p^model_i + ν / 2,   p^model_i = 1 / (1 + exp(−F(x_i)))
   (B1-8)  B2 점수        ℓ_i = −log p_i (y_i = 1), −log(1 − p_i) (y_i = 0);   Brier = (p_i − y_i)²

E. 하네스 제약 및 결정성 보장:
   - 고정 하이퍼파라미터(결정 P9): seed=1, deterministic=True, force_row_wise=True, 8 스레드 고정.
   - 단일 칸 및 헤드 단위 메모리 최적화: _Design 객체를 통해 float32 행렬을 전달하고 LightGBM이 binning하는 즉시 메모리를 해제.
   - 엄격한 직렬화: pickle 및 joblib 대신 UTF-8 텍스트(model_to_string)와 SHA-256 체크섬 JSON 사용.
"""
# Ref: docs/superpowers/plans/2026-10-21-c-ept-s3.md decisions P4 (the factorisation and the probability families), P5 (the target of B2),
#      P6 (the masked M3 object and M4 heads of B1 are conditioned on K and on K and O), P8 (the overdispersion test uses the B1 M2 rate),
#      P9 (fixed LightGBM parameters, seed 1, equal weight for every training row) and Task 7; controller rulings R3 (the protocol), R13
#      (every categorical output is mixed with the uniform law), R18 (the rows are shared with B0), R19 (a model for each team in the kill
#      and the gold heads), R21 (the signature of fit) and R22 (left-out rows are counted, a cell without events gets the Jeffreys rate).
# Ref: docs/superpowers/specs/2026-10-20-c-ept-design.md section 9 (baselines B0 and B1) and sections 3.2 and 3.3 (the masked and the open
#      heads).
# Ref: Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q. and Liu, T.-Y. (2017), "LightGBM: A Highly Efficient Gradient
#      Boosting Decision Tree", Advances in Neural Information Processing Systems 30: 3146-3154 (the learner; a NaN is a missing value).
# Ref: Friedman, J. H. (2001), "Greedy function approximation: a gradient boosting machine", Annals of Statistics 29(5): 1189-1232;
#      doi:10.1214/aos/1013203451 (gradient boosting with shrinkage; the Poisson and the multinomial deviance as the loss).
# Ref: Fisher, W. D. (1958), "On grouping for maximum homogeneity", Journal of the American Statistical Association 53(284): 789-798;
#      doi:10.1080/01621459.1958.10501479 (the optimal split of a categorical feature that LightGBM uses).
# Ref: Cameron, A. C. and Trivedi, P. K. (2013), "Regression Analysis of Count Data", 2nd ed., Cambridge University Press, chapter 3;
#      doi:10.1017/CBO9781139013567 (Poisson regression with an exposure: the mean is exposure times rate, the offset is the log exposure;
#      here the init score).
# Ref: Prechelt, L. (2012), "Early stopping - but when?", in Neural Networks: Tricks of the Trade, 2nd ed., Springer LNCS 7700: 53-67;
#      doi:10.1007/978-3-642-35289-8_5 (stop when the validation loss has not improved for a fixed number of rounds, keep the best round).
# Ref: Chen, S. F. and Goodman, J. (1999), "An empirical study of smoothing techniques for language modeling", Computer Speech and Language
#      13(4): 359-394; doi:10.1006/csla.1999.0128 (interpolation with the uniform distribution, so that no outcome has probability zero).
# Ref: Brier, G. W. (1950), "Verification of forecasts expressed in terms of probability", Monthly Weather Review 78(1): 1-3;
#      doi:10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2 (the score of the binary B2 forecast).
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

import lightgbm as lgb
import numpy as np
from scipy.special import expit

from c_ept import config as C
from c_ept.preprocessing import features
from c_ept.data import head_rows, store_core
from c_ept.models import heads, killlaw, nll as nll_scores
from c_ept.data.casts import narrow
# the names of the heads, their shapes, and the rows and scores that B0 and B1 share (ruling R18)
from c_ept.data.head_rows import (HEAD_NAMES, KINDS, MIX, N_CELLS, N_GOLD, N_K, N_LAYER, N_M, N_Q, N_SIZE, N_ZONE, OPEN_HEAD_NAMES,  # noqa: F401
                             PREDICT_ALL_WIDTHS, TEAMS, HeadBatch, HeadData, HeadScores)

# =============================================================================
# 블록 1: 전역 상수 및 고정 하이퍼파라미터 (decision P9)
# =============================================================================
FIXED_PARAMS = {'num_leaves': 63, 'learning_rate': 0.1, 'min_data_in_leaf': 200, 'feature_fraction': 0.8, 'bagging_fraction': 1.0,
                'lambda_l2': 1.0, 'max_bin': 255, 'deterministic': True, 'force_row_wise': True, 'num_threads': 8, 'seed': 1,
                'verbosity': -1}
MAX_ROUNDS = 1000                 # at most this many boosting rounds
EARLY_STOP = 50                   # stop after this many rounds without improvement on the STOP rows
NUM_THREADS = FIXED_PARAMS['num_threads']
_RESERVED = frozenset({'objective', 'metric', 'num_class', 'categorical_feature', 'init_score', 'label', 'weight'})   # set by the head
SCHEMA = 'b1_heads_v2'            # v2: every head has a folder of its own in ``models/`` (``save_head``); v1 had the texts in ``models/``
HEAD_SCHEMA = 'b1_head_v1'        # the ``head.json`` of one saved head
JSON_NAME, B2_JSON_NAME, MODEL_DIR, HEAD_JSON = 'b1_heads.json', 'b2.json', 'models', 'head.json'
B2_BINS = C.EVAL_H_MS[0] // C.BIN_MS      # B2 looks 60 s ahead: the bins 0 .. 5 of the row (decision P5)
_CHUNK = 1 << 16                  # rows per step of the design fill and of the checks
_PREDICT_CHUNK = 1 << 18          # design rows per call of the composer's boosters


# =============================================================================
# 블록 2: 설계 행렬(Design Matrix) 슬롯 및 레이아웃 (식 B1-1)
# =============================================================================
# the pieces of a design, in the order of its columns. 'feat' is the b1 row, 'K' the 4 columns kills blue, kills red, kill gold blue and
# red in thousands, 'O' the 18 objective counts; q m z s layer one categorical column each.
# 식 (B1-1)의 x_i는 이 조각을 이 순서로 이은 행이다. 머리가 K와 O를 읽는지는 계획 P6이 정한다.
SLOTS = {
    ('M2Rate', 'masked'): ('feat', 'q'),
    ('M2Mark', 'masked'): ('feat', 'q'),
    ('M2Zone', 'masked'): ('feat', 'q', 'm'),
    ('M3Size', 'masked'): ('feat', 'q', 'm', 'z'),
    ('M3Kills', 'masked'): ('feat', 'q', 'm', 'z', 's'),
    ('M3Gold', 'masked'): ('feat', 'q', 'm', 'z', 's', 'layer'),
    ('M3Obj', 'masked'): ('feat', 'K', 'q', 'm', 'z', 's'),              # K: decision P6
    ('M4Conv', 'masked'): ('feat', 'q', 'm', 'z', 'K', 'O'),             # the columns of features.m4_masked_inputs, in its order
    ('M3Size', 'open'): ('feat',),
    ('M3Kills', 'open'): ('feat', 's'),
    ('M3Gold', 'open'): ('feat', 's', 'layer'),
    ('M3Obj', 'open'): ('feat', 'K', 's'),
    ('M4Conv', 'open'): ('feat', 'K', 'O'),
}
CATEGORIES = ('q', 'm', 'z', 's', 'layer')
# the raw values of a category: lo .. lo + n - 1; the column holds value - lo (0 .. n - 1)
_LEVELS = {'q': (0, N_Q), 'm': (0, N_M), 'z': (-1, N_ZONE + 1), 's': (0, N_SIZE), 'layer': (1, N_LAYER)}
# The first-kill team (S4b preparation M1, decision D5) is a category of the open M3Kills design under the shifted kill law only (``make_layout(..., kill_law)``), so
# it is kept apart from the tables of the categories of every design: -1 an execution (no first team), 0 blue, 1 red; the column holds d + 1 (0 .. 2).
FIRST_TEAM = 'd'
_ALL_LEVELS = {**_LEVELS, FIRST_TEAM: (-1, 3)}
K_NAMES = features.M4_EXTRA_NAMES[3:]                  # kills_blue, kills_red, gold_blue_k, gold_red_k
_WIDTH = {'K': N_K, 'O': N_CELLS, **{name: 1 for name in CATEGORIES + (FIRST_TEAM,)}}
NUMERIC_SLOTS = ('feat', 'K', 'O')
# the last dimension of what a head gives for a row (``heads.OUTPUT_WIDTHS``), for the composer's answer to no rows
_OUT_SHAPE = {'M2Mark': (N_M,), 'M2Zone': (N_ZONE,), 'M3Size': (N_SIZE,), 'M3Kills': (2,), 'M3Gold': (N_GOLD,), 'M3Obj': (N_CELLS,)}


class Layout(NamedTuple):
    """The columns of a design: the ``slots`` in column order, the column ``names``, the ``categorical`` column indices and ``d``, the
    number of b1 columns (the width of the 'feat' slot).

    (B1-1)의 x_i의 열 배치이다. 범주 열의 번호를 LightGBM에 알리고, 저장한 머리의 열 이름을 불러올 때 이 모듈의 설계와 대조하는 데 쓴다."""
    slots: tuple
    names: tuple
    categorical: tuple
    d: int

    @property
    def width(self) -> int:
        return len(self.names)


def make_layout(head: str, kind: str, columns, kill_law: str = 'poisson') -> Layout:
    """The ``Layout`` of the design of the head ``head`` of ``kind``, when the b1 columns are ``columns`` (``features.b1_columns``). ``kill_law`` is the kill law
    of the M3Kills head (decision D5): under ``'shifted'`` the design of the open M3Kills head has the category ``d`` (the first-kill team) after ``s``, the one input
    that tells a team's cell whether its rate is the rate λ_d of the excess N_d or the rate μ of the plain count; every other design is that of the plain law.

    수식
    ----
    (B1-1)  x_i = [b1 열 x_{i1}, …, x_{id} | K 블록 | O 블록 | 범주 열 q, m, z + 1, s, ξ − 1]
            머리마다 SLOTS가 적은 조각만 SLOTS의 순서로 이어 붙인다(예: 가린 M4Conv는 b1 열, q, m, z + 1, K 블록, O 블록의 순서이다).
            K 블록 = (k_blue, k_red, g_blue / 1000, g_red / 1000),   O 블록 = (O_0, …, O_17)

      기호                  | 뜻                                                              | 단위    | 범위                       | 코드 변수
      x_{ih}                | 설계 행 i의 b1 특징 h(가린 머리는 t₀ 행, 열린 머리는 사건 보기) | 열 단위 | 유한 또는 NaN(결측)        | feat[:, h]
      d                     | b1 열의 수                                                      | 열      | ≥ 1                        | Layout.d
      k_blue, k_red         | 팀별 킬 수(K 블록의 앞 두 열)                                   | 개      | 0, 1, …                    | K[:, :2]
      g_blue, g_red         | 팀별 킬 골드(K 블록에는 천 골드 단위로 넣는다)                  | 골드    | ≥ 0                        | K[:, 2:]
      O_c                   | 오브젝트 칸 c의 개수                                            | 개      | 0, 1, …                    | O[:, c]
      q, m, z + 1, s, ξ − 1 | 범주 열의 값(원래 값에서 가장 낮은 값을 뺀 것)                  | —       | 0–17, 0–51, 0–10, 0–3, 0–5 | codes

    해석
    ----
    식 (B1-1)은 한 머리가 읽는 입력을 b1 열, 조건이 되는 결과의 일부(K, O), 범주 열 가운데 그 머리의 조각을 이어 붙인 행으로 정의한다. 범주는 수준마다 열을 펼치지 않고 정수 한 열로 두어 LightGBM이 수준의 묶음을 한
    번에 가르게 한다(LightGBM의 기본값에서 수준이 셋 이하인 범주는 한 수준 대 나머지로 가르지만 이 설계의 범주 열은 모두 넷 이상이다) (결측 칸도 수준 하나로 센다) [Fisher 1958]. LightGBM이 범주 열의 음수를
    결측으로 읽으므로 위치 없음(z = −1)은 자기 수준 0이 되고(B0와 같다) 킬 수 층 1–6은 0–5가 된다. b1 열의 NaN은 그대로 두어 LightGBM이 결측으로 읽으며, 결측을 FIT 평균으로 채우는 B0와 다르다(판정 R2).
    가린 오브젝트 머리의 K와 가린 전환 머리의 K와 O는 학습에서는 실제 값이고 합성에서는 표본 값이다(계획 P6).

    근거
    ----
    계획 P6(머리가 조건으로 삼는 입력), 판정 R2(B0의 결측 채우기), R18(행은 B0와 공유), 설계서 3.3절 누설 시험 (a)(가린 M4 입력은 ``features.m4_masked_inputs``가 만든다)와 5.3절(챔피언,
    아이템, 패치 ID는 입력이 아니다), 계획 전역 제약의 구역 z(위치 없음은 "없음" 범주).
    Ref: Fisher, W. D. (1958), Journal of the American Statistical Association 53(284): 789-798;
         doi:10.1080/01621459.1958.10501479 (범주 특징의 최적 분할).
    """
    if (head, kind) not in SLOTS:
        raise ValueError(f'there is no {kind} head {head!r}; the heads are {[k for k in SLOTS]}')
    killlaw.check_law(kill_law)
    slots = SLOTS[(head, kind)] + ((FIRST_TEAM,) if kill_law == 'shifted' and (head, kind) == ('M3Kills', 'open') else ())
    names, categorical = [], []
    for slot in slots:
        if slot == 'feat':
            names += list(columns)
        elif slot == 'K':
            names += list(K_NAMES)
        elif slot == 'O':
            names += list(features.O_NAMES)
        else:
            categorical.append(len(names))
            names.append(slot)
    if len(set(names)) != len(names):
        raise ValueError('the design has a column name twice: ' + ', '.join(sorted({n for n in names if names.count(n) > 1})))
    return Layout(slots, tuple(names), tuple(categorical), len(columns))


def _int_codes(name: str, values) -> np.ndarray:
    """``values`` as an int64 1-d array. TypeError for a dtype that is not an integer (a bool, a float), ValueError for another shape."""
    a = np.asarray(values)
    if a.size == 0 and a.ndim <= 1:
        a = a.astype(np.int64)
    if a.dtype.kind not in 'iu':
        raise TypeError(f'{name} must be an integer array, not {a.dtype}')
    if a.ndim != 1:
        raise ValueError(f'{name} must be 1-d, not shape {a.shape}')
    return a.astype(np.int64)


def _category(slot: str, values, n: int) -> np.ndarray:
    """The column of the category ``slot`` for the raw values ``values`` (n,): value - lo, 0 .. levels - 1, as float32.

    (B1-1)의 범주 열 값이다: 원래 값에서 가장 낮은 값 lo를 뺀다(구역은 z + 1, 킬 수 층은 ξ − 1, 나머지는 그대로)."""
    a = _int_codes(slot, values)
    if len(a) != n:
        raise ValueError(f'{slot} has {len(a)} entries but there are {n} rows')
    lo, levels = _ALL_LEVELS[slot]
    if a.size and (a.min() < lo or a.max() > lo + levels - 1):
        raise ValueError(f'{slot} must lie in {lo} .. {lo + levels - 1}')
    return (a - lo).astype(np.float32)


def _block(slot: str, values, n: int) -> np.ndarray:
    """The K block (kills and kill gold in thousands) or the O block of a design, checked: (n, 4) or (n, 18), finite, not negative."""
    a = np.asarray(values, dtype=np.float64)
    if a.shape != (n, _WIDTH[slot]):
        raise ValueError(f'{slot} must have shape ({n}, {_WIDTH[slot]}), not {a.shape}')
    if not np.isfinite(a).all() or (a < 0).any():
        raise ValueError(f'{slot} must be finite and not negative')
    return a


def _assemble(slots, feat, take, codes, K=None, O=None) -> np.ndarray:
    """The design (n, width) float32 of the rows. Row i reads the b1 row ``feat[take[i]]`` (``feat[i]`` when ``take`` is None); ``codes``
    holds the raw values of the categories in ``slots``, ``K`` the K block (kills blue, kills red, kill gold blue and red in thousands)
    and ``O`` the 18 objective counts, when the slots have them. The b1 rows are copied chunk by chunk, so the 8.2 M rows of M2Rate are
    not gathered a second time.

    (B1-1)의 x_i를 SLOTS의 조각마다 열 위치에 놓아 만든다."""
    feat = np.asarray(feat)
    if feat.ndim != 2:
        raise ValueError(f'the b1 rows must be a 2-d array, not shape {feat.shape}')
    if take is not None:
        take = _int_codes('take', take)
        if take.size and (take.min() < 0 or take.max() >= len(feat)):
            raise ValueError(f'take must lie in 0 .. {len(feat) - 1}')
    n = len(feat) if take is None else len(take)
    d = feat.shape[1]
    pieces = {'K': K, 'O': O}
    width = sum(d if slot == 'feat' else _WIDTH[slot] for slot in slots)
    X = np.empty((n, width), dtype=np.float32)
    at = 0
    for slot in slots:
        if slot == 'feat':
            for lo in range(0, n, _CHUNK):
                hi = min(lo + _CHUNK, n)
                X[lo:hi, at:at + d] = feat[lo:hi] if take is None else feat[take[lo:hi]]
        elif slot in ('K', 'O'):
            if pieces[slot] is None:
                raise ValueError(f'the design needs the {slot} block')
            X[:, at:at + _WIDTH[slot]] = narrow(slot, _block(slot, pieces[slot], n), np.float32)
        else:
            if slot not in codes:
                raise ValueError(f'the design needs the category {slot}')
            X[:, at] = _category(slot, codes[slot], n)
        at += d if slot == 'feat' else _WIDTH[slot]
    return X


def _m4_design(m4, d: int) -> np.ndarray:
    """The design of the masked M4 head from the matrix of ``features.m4_masked_inputs`` (n, d + 25): the same matrix with the zone column
    shifted by one (the zone -1, "no position", is the level 0). The matrix is changed in place when it is a float32 array of its own; a
    copy is made otherwise.

    (B1-1)의 가린 M4 머리 설계를 만든다: ``features.m4_masked_inputs``의 행렬에서 구역 열만 z + 1로 옮긴다."""
    m4 = np.asarray(m4)
    if m4.ndim != 2 or m4.shape[1] != d + len(features.M4_EXTRA_NAMES) + N_CELLS:
        raise ValueError(f'the masked M4 input must have {d + len(features.M4_EXTRA_NAMES) + N_CELLS} columns (the b1 row, q, m, z, K '
                         f'and O), not shape {m4.shape}')
    if m4.dtype != np.float32 or not m4.flags.c_contiguous or not m4.flags.writeable:
        m4 = narrow('the masked M4 input', m4 if m4.dtype == np.float32 else m4.astype(np.float64), np.float32)      # a new array
    for j, slot in enumerate(('q', 'm', 'z')):                                       # the categories are columns d, d + 1, d + 2
        col = m4[:, d + j]
        lo, levels = _LEVELS[slot]
        if col.size and (col.min() < lo or col.max() > lo + levels - 1 or not np.array_equal(col, np.rint(col))):
            raise ValueError(f'the {slot} column of the masked M4 input must hold whole numbers in {lo} .. {lo + levels - 1}')
    m4[:, d + 2] += 1.0
    return m4


class _Design:
    """A training design that one receiver owns: ``take()`` hands the array over once and forgets it. A head fitted through ``fit_head``
    gets its design in one of these, so that the last reference to the float32 matrix (14 GiB for M2Rate) can be dropped as soon as
    LightGBM has binned it, instead of living until the caller returns (the binned matrix is a quarter of the size)."""

    def __init__(self, X):
        self._X = X

    def take(self):
        if self._X is None:
            raise RuntimeError('the design was taken already')
        X, self._X = self._X, None
        return X


# =============================================================================
# 블록 3: 설계 행렬 및 범주형 열 검증 유틸리티
# =============================================================================
def _design_matrix(X) -> np.ndarray:
    """The design as a float32 C-contiguous 2-d array (no copy when it is one already). A NaN stays (LightGBM reads it as missing); an
    infinite value is refused. TypeError for a dtype that is not numeric, ValueError for a shape without rows or columns."""
    A = np.asarray(X)
    if A.dtype.kind not in 'fiub':
        raise TypeError(f'the design must be numeric, not {A.dtype}')
    if A.ndim != 2 or A.shape[0] == 0 or A.shape[1] == 0:
        raise ValueError(f'the design must be a 2-d array with rows and columns, not shape {A.shape}')
    if A.dtype != np.float32:
        A = narrow('the design', A.astype(np.float64), np.float32)
    elif not A.flags.c_contiguous:
        A = np.ascontiguousarray(A)
    for lo in range(0, len(A), _CHUNK):
        if np.isinf(A[lo:lo + _CHUNK]).any():
            raise ValueError('the design holds an infinite value')
    return A


def _check_categorical(X: np.ndarray, categorical) -> tuple:
    """The categorical column indices as a tuple of ints, after checking that they are columns of ``X`` and that each holds whole numbers
    from 0 up (LightGBM reads a negative value as missing and wants integers)."""
    cat = tuple(sorted(int(c) for c in categorical))
    if len(set(cat)) != len(cat):
        raise ValueError('the categorical columns must be distinct')
    for c in cat:
        if not 0 <= c < X.shape[1]:
            raise ValueError(f'the categorical column {c} is not a column of the design ({X.shape[1]} columns)')
        col = X[:, c]
        if not np.isfinite(col).all() or (col < 0).any() or not np.array_equal(col, np.floor(col)):
            raise ValueError(f'the categorical column {c} must hold whole numbers from 0 up')
    return cat


def _split_params(params) -> tuple:
    """(LightGBM parameters, rounds, patience) from the fixed parameters and the overrides ``params`` (``num_iterations`` is the number of
    rounds, ``early_stopping_round`` the patience). ValueError for a key that the head sets itself."""
    p = dict(FIXED_PARAMS)
    if params is not None:
        if not isinstance(params, Mapping):
            raise TypeError(f'params must be a mapping, not {type(params).__name__}')
        bad = sorted(set(params) & _RESERVED)
        if bad:
            raise ValueError(f'params must not set {", ".join(bad)}: the head sets them')
        p.update(params)
    rounds, patience = p.pop('num_iterations', MAX_ROUNDS), p.pop('early_stopping_round', EARLY_STOP)
    for name, value in (('num_iterations', rounds), ('early_stopping_round', patience)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f'{name} must be a whole number of at least 1, not {value!r}')
    return p, int(rounds), int(patience)


# =============================================================================
# 블록 4: 부스팅 기저 클래스 (_Boosted) 및 조기 종료 (식 B1-5)
# =============================================================================
class _Boosted:
    """What the boosted heads share: the parameters, the layout, one LightGBM booster for each cell and the files. A head holds
    ``boosters_`` (one per cell; None for a cell that has no booster), ``info_`` (one dict per cell: best_iteration, num_trees,
    rounds_run, stopped_early, best_score), ``layout_`` (the ``Layout`` of the design, when it was given), ``categorical_`` (the
    categorical columns) and ``n_features_``. ``predict(X)`` reads a full design matrix, ``predict_blocks(numeric, **codes)`` the numeric
    part and the raw categories (the composer's way).

    수식
    ----
    (B1-5)  R̂ = argmin_{R ≤ R_run} Loss_stop(R)
            Loss_stop(R) = R 라운드 뒤 STOP 행의 평균 손실(섞기 전의 확률로 잰다): 포아송은 μ − y·log μ, 다중 분류는 −log p^model_{i,c_i}, 이진은 −log(맞힌 쪽의 확률)
            R_run은 학습이 돈 라운드 수이다: 최대 R_max까지 돌되 Loss_stop이 인내 라운드 수(계획 P9) 동안 낮아지지 않으면 멈춘다. 모형은 R̂ 라운드까지의 나무만 남긴다.

      기호         | 뜻                                    | 단위   | 범위        | 코드 변수
      R            | 라운드(나무를 더하는 단계) 번호        | 라운드 | 1, …, R_max | —
      R̂            | 남긴 라운드 수                         | 라운드 | 1, …, R_run | best_iteration
      R_run        | 학습이 돈 라운드 수                    | 라운드 | 1, …, R_max | rounds_run
      R_max        | 최대 라운드 수                         | 라운드 | ≥ 1         | rounds (MAX_ROUNDS)
      Loss_stop(R) | R 라운드 뒤 STOP 행의 평균 손실        | nat    | 유한        | record['stop'][metric]

    해석
    ----
    식 (B1-5)는 나무를 계속 쌓으면 훈련 행의 손실은 줄지만 STOP 행처럼 적합에 쓰지 않은 행의 손실은 어느 라운드부터 오르기 시작하므로, 그 손실이 가장 낮았던 라운드까지만 남긴다는 조기 종료 규칙이다 [Prechelt 2012]. B1
    머리 전체의 적합과 B2의 적합은 조기 종료 역할이 적합 역할과 겹치면 거부하므로, 그 경로에서 라운드 수는 훈련 행에 맞춰지지 않는다. 손실은 부스터가 내는 확률로 재며 균등 분포 섞기 (B1-4)는 들어가지 않는다. STOP 행을 주지
    않으면 최대 라운드를 모두 남긴다. 범주 머리에서 훈련에 없던 범주의 검증 행은 부스터가 확률을 줄 수 없어 이 손실에서 뺀다(모듈 글).

    근거
    ----
    계획 P9(고정 모수: 최대 라운드와 조기 종료의 인내 라운드), Task 7(STOP 행에서 조기 종료), 판정 R21(fit의 roles 서명).
    Ref: Prechelt, L. (2012), "Early stopping - but when?", in Neural Networks: Tricks of the Trade, 2nd ed., Springer LNCS 7700: 53-67;
         doi:10.1007/978-3-642-35289-8_5 (검증 손실이 낮아지지 않으면 멈추고 가장 좋은 라운드를 남긴다).
    """
    name = ''
    objective = ''
    metric = ''

    @property
    def parts(self) -> tuple:
        """The single models that this head is made of (itself; the gold head has one for each team)."""
        return (self,)

    def __init__(self, kind: str = 'masked', params=None):
        if kind not in KINDS:
            raise ValueError(f'kind must be one of {KINDS}, not {kind!r}')
        self.kind = kind
        self.params, self.rounds, self.patience = _split_params(params)
        self._label = self.name
        self.layout_ = None
        self.categorical_ = ()
        self.n_features_ = 0
        self.boosters_ = []
        self.info_ = []
        self.n_fit_ = 0
        self.n_valid_ = 0

    # -- the layout ------------------------------------------------------------------------------------------------------------------------
    def _set_layout(self, layout, categorical, X: np.ndarray) -> tuple:
        """Record the layout and the categorical columns of the design ``X``; return the categorical columns (checked)."""
        if layout is not None:
            if layout.width != X.shape[1]:
                raise ValueError(f'{self.name}: the layout has {layout.width} columns but the design has {X.shape[1]}')
            categorical = layout.categorical
        self.layout_ = layout
        self.categorical_ = _check_categorical(X, categorical)
        self.n_features_ = X.shape[1]
        return self.categorical_

    def _need_fit(self) -> None:
        if not self.info_:
            raise RuntimeError(f'{self.name} is not fitted')

    # -- the boosters ----------------------------------------------------------------------------------------------------------------------
    def _lgb_params(self, **extra) -> dict:
        return {**self.params, 'objective': self.objective, 'metric': self.metric, **extra}

    def _boost(self, design, cells, valid, check, **extra) -> list:
        """Train one booster for each entry of ``cells`` on the training design (a ``_Design`` that this method owns): ``cells[c]`` is
        ``(label, init_score or None)`` or None for a cell that has no booster. ``valid`` is None or ``(Xv, vcells)`` with ``vcells[c]``
        like ``cells[c]``. The binned Dataset is made once and only its label and init score change from cell to cell; the float32
        design is dropped as soon as LightGBM has binned it. Returns a list of ``(booster or None, info)``.

        (B1-2), (B1-4), (B1-7)의 목적함수로 부스터를 훈련하고 (B1-5)의 조기 종료를 건다."""
        params = self._lgb_params(**extra)
        categorical = list(self.categorical_) or 'auto'
        ds = dv = None
        out = []
        for c, cell in enumerate(cells):
            if cell is None:
                out.append((None, None))
                continue
            if check is not None:
                check()
            label, init = cell
            if ds is None:                                       # the first booster: the Dataset is made and binned
                X = design.take()
                ds = lgb.Dataset(X, label=label, init_score=init, categorical_feature=categorical, params=params, free_raw_data=True)
                ds.construct()                                   # LightGBM drops its reference to X here (free_raw_data) ...
                del X                                            # ... and so do we: the float32 matrix is freed, the binned one stays
                if valid is not None:
                    vlabel, vinit = valid[1][c]
                    dv = lgb.Dataset(valid[0], label=vlabel, init_score=vinit, reference=ds, categorical_feature=categorical,
                                     params=params, free_raw_data=True)
            else:                                                # the next ones reuse the binned matrix
                ds.set_label(label)
                if init is not None:
                    ds.set_init_score(init)
                if dv is not None:
                    vlabel, vinit = valid[1][c]
                    dv.set_label(vlabel)
                    if vinit is not None:
                        dv.set_init_score(vinit)
            record = {}
            # 식 (B1-5): 검증 Dataset(STOP 행)이 있을 때만 조기 종료를 건다. 라운드마다의 지표 Loss_stop(R)을 ``record``가 모은다.
            callbacks = [lgb.record_evaluation(record)]
            if dv is not None:
                callbacks.append(lgb.early_stopping(self.patience, first_metric_only=True, verbose=False))
            booster = lgb.train(params, ds, num_boost_round=self.rounds, valid_sets=None if dv is None else [dv], valid_names=['stop'],
                                callbacks=callbacks)
            text = booster.model_to_string()                     # the rounds up to the best one when early stopping chose it
            kept = lgb.Booster(model_str=text)                   # what a load gives back: the saved head is this one
            rounds_run = len(record['stop'][self.metric]) if dv is not None else self.rounds
            score = booster.best_score['stop'][self.metric] if dv is not None else None
            out.append((kept, {'best_iteration': int(kept.current_iteration()), 'num_trees': int(kept.num_trees()),
                               'rounds_run': int(rounds_run), 'stopped_early': bool(rounds_run < self.rounds),
                               'best_score': None if score is None or not np.isfinite(score) else float(score)}))
            del booster
        return out

    def predict_blocks(self, numeric, **codes) -> np.ndarray:
        """The prediction from the numeric part of the design (the b1 columns, then K and then O for the heads that have them, K in
        thousands) and the raw categories (keyword = slot name): the design is made by the function that makes the training designs.

        합성기가 부르는 길이다. 입력은 (B1-1)의 숫자 열과 원래 범주 값이다."""
        return self.predict(self._blocks_design(numeric, codes))

    def _blocks_design(self, numeric, codes) -> np.ndarray:
        self._need_fit()
        if self.layout_ is None:
            raise ValueError(f'{self.name} was fitted without a layout, so the block form cannot be used')
        slots = self.layout_.slots
        A = np.asarray(numeric)
        if A.ndim != 2:
            raise ValueError(f'{self.name}: the numeric part must be 2-d, not shape {A.shape}')
        widths = {s: (self.layout_.d if s == 'feat' else _WIDTH[s]) for s in slots if s in NUMERIC_SLOTS}
        if A.shape[1] != sum(widths.values()):
            raise ValueError(f'{self.name}: the numeric part must have {sum(widths.values())} columns, not shape {A.shape}')
        wanted = {s for s in slots if s in CATEGORIES or s == FIRST_TEAM}
        if set(codes) != wanted:
            raise ValueError(f'{self.name}: the codes must be given for the categories {sorted(wanted)}, not {sorted(codes)}')
        parts, at = {}, 0
        for s in NUMERIC_SLOTS:
            if s in widths:
                parts[s] = A[:, at:at + widths[s]]
                at += widths[s]
        return _assemble(slots, parts['feat'], None, codes, parts.get('K'), parts.get('O'))

    # -- storage ---------------------------------------------------------------------------------------------------------------------------
    def state(self) -> tuple:
        """(texts, meta): ``texts`` maps the part name of every booster (``c00``, ``c01`` ...) to its ``model_to_string`` text, ``meta``
        is a dict of plain JSON values."""
        self._need_fit()
        texts = {f'c{c:02d}': b.model_to_string() for c, b in enumerate(self.boosters_) if b is not None}
        cells = [dict(info, part=None if b is None else f'c{c:02d}') for c, (b, info) in enumerate(zip(self.boosters_, self.info_))]
        meta = {'class': self.name, 'kind': self.kind, 'n_fit': int(self.n_fit_), 'n_valid': int(self.n_valid_),
                'n_features': int(self.n_features_), 'categorical': list(self.categorical_),
                'slots': None if self.layout_ is None else list(self.layout_.slots),
                'params': {**self.params, 'num_iterations': self.rounds, 'early_stopping_round': self.patience}, 'cells': cells}
        return texts, meta

    @classmethod
    def from_state(cls, texts: Mapping, meta: Mapping, layout=None):
        head = cls(meta['kind'], meta['params'])
        head._restore(texts, meta, layout)
        return head

    def _restore(self, texts: Mapping, meta: Mapping, layout=None) -> None:
        self.kind = meta['kind']
        self.layout_ = layout
        self.categorical_ = tuple(meta['categorical'])
        self.n_features_ = int(meta['n_features'])
        self.n_fit_, self.n_valid_ = int(meta['n_fit']), int(meta['n_valid'])
        self.boosters_ = [None if cell['part'] is None else lgb.Booster(model_str=texts[cell['part']]) for cell in meta['cells']]
        self.info_ = [{k: v for k, v in cell.items() if k not in ('part',)} for cell in meta['cells']]

    def summary(self) -> dict:
        """The counts of the fit as plain JSON values: the rounds kept (``best_iteration``) and the trees of every model, the rounds run,
        whether early stopping chose the round, and the left-out rows."""
        self._need_fit()
        return {'kind': self.kind, 'n_fit': int(self.n_fit_), 'n_valid': int(self.n_valid_), 'width': int(self.n_features_),
                'categorical': list(self.categorical_),
                'best_iteration': [int(i['best_iteration']) for i in self.info_], 'num_trees': [int(i['num_trees']) for i in self.info_],
                'rounds_run': [int(i['rounds_run']) for i in self.info_], 'stopped_early': [bool(i['stopped_early']) for i in self.info_]}


def _no_booster_info() -> dict:
    return {'best_iteration': 0, 'num_trees': 0, 'rounds_run': 0, 'stopped_early': False, 'best_score': None}


# =============================================================================
# 블록 5: 포아송 부스팅 헤드 (_PoissonBoosted: 식 B1-2, B1-3)
# =============================================================================
class _PoissonBoosted(_Boosted):
    """A Poisson boosted model with the log link, per second of exposure: the rate of a cell. One booster per cell (a column of the
    target); ``fit(X, Y, e)`` takes the design, the counts (n,) or (n, J) and the exposure in seconds. A target given as a 1-d array makes
    a head of one cell that predicts a vector; a 2-d target (also with one column) makes ``predict`` return a matrix.

    수식
    ----
    (B1-2)  log μ_{ic} = log e_i + γ_c + F_c(x_i),    λ_{ic} = exp(γ_c + F_c(x_i)),    γ_c = log(Σ_i y_{ic} / Σ_i e_i)
            (합은 e_i > 0인 행에 대해서, 칸 c마다. F_c는 R̂ 라운드까지의 나무의 원점수 합이고 Loss_c = Σ_i (μ_{ic} − y_{ic}·log μ_{ic})를 낮추는 방향으로 한 그루씩 더한다)
    (B1-3)  칸 c에 사건이 없으면(Σ_i y_{ic} = 0) 나무를 두지 않고 γ_c = log(0.5 / Σ_i e_i), F_c = 0, 곧 λ_c = 0.5 / Σ_i e_i를 쓴다.

      기호           | 뜻                                            | 단위              | 범위             | 코드 변수
      c, J           | 칸 번호, 칸의 수                              | —, 개             | 0, …, J − 1; ≥ 1 | Y의 열, Y.shape[1]
      y_{ic}         | 행 i의 칸 c에서 센 개수                       | 개                | 0, 1, …          | Y[:, c]
      e_i            | 행 i의 노출(적합에는 e_i > 0인 행만)          | s (M3는 1)        | > 0              | e
      γ_c            | 칸 c의 기준 로그 발생률(나무가 시작하는 절편) | 로그 척도         | 유한             | base_[c]
      F_c(x_i)       | 칸 c의 나무 합(원점수; 오프셋과 γ_c는 제외)   | 로그 척도         | 유한             | _raw(X)[:, c]
      μ_{ic}, λ_{ic} | 기대 개수 e_i·λ_{ic}, 예측 발생률             | 개, 1/s (M3는 개) | > 0              | expected(X, e), predict(X)
      Loss_c         | 칸 c의 부스팅 손실                            | nat               | 유한             | LightGBM 목적함수 'poisson'

    해석
    ----
    식 (B1-2)는 칸 c의 개수를 평균 μ_{ic} = e_i·λ_{ic}의 포아송으로 보고, 노출의 로그를 오프셋으로 고정한 채 나무의 합이 로그 발생률을 맞추는 모형이며, 예측은 노출 없이 초당 발생률 λ_{ic}이다. Loss_c는
    (NLL-1)의 칸 점수 ℓ_{ic}의 합에서 모형과 무관한 Σ_i log(y_{ic}!)를 뺀 것이어서 같은 방향으로 낮아진다 [도출]. 기대 개수가 노출에 비례함은 오프셋이 보장한다 [Cameron & Trivedi 2013].
    LightGBM은 초기 점수가 주어지면 평균에서 출발하지 않으므로 γ_c를 오프셋에 더해 나무가 칸의 평균 발생률에서 출발하게 한다(판정 R23). 사건이 하나도 없는 칸은 발생률이 0으로 내려갈수록 우도가 올라 최대우도 해가 없으므로 나무를
    두지 않고 제프리스 사전분포의 사후 평균인 상수 발생률을 쓰며, 이는 B0와 같다(판정 R22, (B0-3)).

    근거
    ----
    계획 P4(포아송 칸), P9(고정 모수), P10(노출), Task 7(오프셋), 판정 R19(킬은 팀마다 칸), R22(사건 없는 칸), R23(init score의 형태), 설계서 3.1절(계수 과정), 검산 노트
    math-control-03 C절(칸 점수).
    Ref: Cameron, A. C. and Trivedi, P. K. (2013), "Regression Analysis of Count Data", 2nd ed., Cambridge University Press;
         doi:10.1017/CBO9781139013567 (노출이 있는 포아송, 로그 노출 오프셋). Jeffreys, H. (1946), Proc. R. Soc. Lond. A 186(1007): 453-461;
         doi:10.1098/rspa.1946.0056 (제프리스 사전분포).
    """
    objective = 'poisson'
    metric = 'poisson'

    def __init__(self, kind: str = 'masked', params=None):
        super().__init__(kind, params)
        self.squeeze_ = False
        self.n_zero_exposure_ = 0
        self.base_ = self.n_events_ = self.degenerate_ = None

    @staticmethod
    def _inputs(name: str, X, Y, e) -> tuple:
        """The checked design, counts (n, J), exposure (n,), whether the target was 1-d and the number of rows without exposure; the rows
        without exposure are left out (a count in one is an error)."""
        X = _design_matrix(X)
        n = len(X)
        Y = np.asarray(Y)
        if Y.ndim not in (1, 2) or Y.shape[0] != n:
            raise ValueError(f'{name}: the counts must have {n} rows (one per design row), not shape {Y.shape}')
        if Y.dtype.kind not in 'fiu':
            raise TypeError(f'{name}: the counts must be numbers, not {Y.dtype}')
        squeeze = Y.ndim == 1
        Y = Y.reshape(n, -1).astype(np.float64)
        e = np.asarray(e, dtype=np.float64)
        if e.shape != (n,):
            raise ValueError(f'{name}: the exposure must have shape ({n},), not {e.shape}')
        if not np.isfinite(e).all() or (e < 0).any():
            raise ValueError(f'{name}: an exposure is negative or not finite')
        if not np.isfinite(Y).all() or (Y < 0).any() or (Y != np.floor(Y)).any():
            raise ValueError(f'{name}: a count is not a non-negative whole number')
        keep = e > 0
        if (Y[~keep] > 0).any():
            raise ValueError(f'{name}: {int((Y[~keep] > 0).any(axis=1).sum())} row(s) have a count but no exposure')
        if not keep.any():
            raise ValueError(f'{name}: no row has any exposure')
        if not keep.all():
            X, Y, e = X[keep], Y[keep], e[keep]
        return X, Y, e, squeeze, int((~keep).sum())

    def fit(self, X, Y, e, *, categorical=(), layout=None, valid=None, check=None):
        """Fit on the rows with exposure. ``valid``: None, or ``(Xv, Yv, ev)`` the rows that early stopping reads (the STOP rows).
        ``layout``: the ``Layout`` of the design (its categorical columns are used). ``check``: a function without arguments (the E6
        guard) called before every booster. ``X`` is an array or a ``_Design`` (which this call consumes and drops once LightGBM has
        binned it).

        (B1-2)와 (B1-3)을 칸마다 맞추고 머리 자신을 돌려준다."""
        X, Y, e, squeeze, n_zero = self._inputs(self.name, X.take() if isinstance(X, _Design) else X, Y, e)
        self._set_layout(layout, categorical, X)
        cells = Y.shape[1]
        if valid is not None:
            if len(valid) != 3:
                raise ValueError(f'{self.name}: valid must be (X, counts, exposure)')
            Xv, Yv, ev, _, _ = self._inputs(self.name + ' (validation)', *valid)
            if Xv.shape[1] != X.shape[1] or Yv.shape[1] != cells:
                raise ValueError(f'{self.name}: the validation rows must have {X.shape[1]} columns and {cells} cell(s)')
        self.squeeze_ = squeeze
        self.n_zero_exposure_ = n_zero
        self.n_events_ = Y.sum(axis=0).astype(np.int64)
        self.degenerate_ = self.n_events_ == 0
        total = float(e.sum())
        # 식 (B1-2)의 γ_c와 (B1-3)의 상수: 사건 없는 칸은 log(0.5 / Σ e), 나머지는 log(Σ y / Σ e)이다. 두 합은 e > 0인 행에 대한 것이다(``_inputs``가
        # 노출 0 행을 이미 뺐다).
        # the average rate of the cell, or the Jeffreys posterior mean of a cell without event: the trees start there
        self.base_ = np.where(self.degenerate_, np.log(0.5 / total), np.log(np.maximum(self.n_events_, 1) / total))
        # 식 (B1-2)의 오프셋 log e_i + γ_c를 init score로 준다. 노출은 입력이 아니라 오프셋이다(판정 R23).
        log_e = np.log(e)
        cell_data = [None if self.degenerate_[c] else (Y[:, c], log_e + self.base_[c]) for c in range(cells)]
        valid_data = None
        if valid is not None:
            # 검증 행의 오프셋도 훈련 행의 γ_c를 쓴다(판정 R23): 조기 종료의 Loss_stop (B1-5)가 적합한 모형과 같은 모형의 손실을 본다.
            log_ev = np.log(ev)
            valid_data = (Xv, [(Yv[:, c], log_ev + self.base_[c]) for c in range(cells)])
            self.n_valid_ = int(len(ev))
        else:
            self.n_valid_ = 0
        self.n_fit_ = int(len(e))
        design = _Design(X)
        del X                                                    # the design has one owner from here on: _boost frees it once it is binned
        trained = self._boost(design, cell_data, valid_data, check)
        self.boosters_ = [b for b, _ in trained]
        self.info_ = [_no_booster_info() if i is None else i for _, i in trained]
        return self

    def _raw(self, X) -> np.ndarray:
        """(n, cells) the raw score of every booster (0 for a cell without one; the init score is not in it), float64.

        (B1-2)의 나무 합 F_c(x_i)를 칸마다 계산한다."""
        self._need_fit()
        X = _design_matrix(X)
        if X.shape[1] != self.n_features_:
            raise ValueError(f'{self.name}: the design must have {self.n_features_} columns, not shape {X.shape}')
        out = np.zeros((len(X), len(self.boosters_)))
        for c, booster in enumerate(self.boosters_):
            if booster is not None:
                for lo in range(0, len(X), _PREDICT_CHUNK):
                    out[lo:lo + _PREDICT_CHUNK, c] = booster.predict(X[lo:lo + _PREDICT_CHUNK], raw_score=True, num_threads=NUM_THREADS)
        return out

    def predict(self, X) -> np.ndarray:
        """The rate per second of every cell: (n,) for a head fitted on a 1-d target, else (n, J).

        (B1-2)의 λ_{ic} = exp(γ_c + F_c(x_i))를 낸다. exp가 넘쳐 유한하지 않은 값은 (NLL-1)의 정의역 밖이므로 거부한다."""
        Z = self._raw(X) + self.base_
        with np.errstate(over='ignore'):
            rate = np.exp(Z)
        if not np.isfinite(rate).all():
            raise ValueError(f'{self.name}: the rate of {int((~np.isfinite(rate)).any(axis=1).sum())} row(s) is not finite (the raw score '
                             'is out of the range of exp)')
        return rate[:, 0] if self.squeeze_ else rate

    def rate(self, X) -> np.ndarray:
        """The rate per unit of exposure of every cell. Same as ``predict``."""
        return self.predict(X)

    def expected(self, X, exposure_s) -> np.ndarray:
        """The expected count of every cell: ``rate(X) * exposure_s``. ``exposure_s``: a number, one value per row (n,) or one per cell
        and row (n, J), in seconds. ValueError for a negative or non-finite exposure.

        (B1-2)의 μ_{ic} = e_i·λ_{ic}를 계산한다."""
        rate = self.predict(X)
        ex = np.asarray(exposure_s, dtype=np.float64)
        if not np.isfinite(ex).all() or (ex < 0).any():
            raise ValueError(f'{self.name}: the exposure must be finite and not negative')
        if ex.ndim == 1 and rate.ndim == 2:
            ex = ex[:, None]
        return rate * ex

    def nll(self, X, Y, e, cells: bool = False) -> np.ndarray:
        """The Poisson NLL of the counts ``Y`` with the exposure ``e`` (``nll.poisson``, ``log y!`` included): one value per row, the
        cells summed, or (n, J) with ``cells``. A row with no exposure scores 0 (its rate is not read).

        (B1-2)의 발생률과 노출로 (NLL-1)의 칸 점수 ℓ_{ic}를 내고, 칸을 더하면 설계 행의 점수 ℓ_i이다((HR-5)의 입력)."""
        rate = self.predict(X)
        Y = np.asarray(Y, dtype=np.float64)
        e = np.asarray(e, dtype=np.float64)
        per_cell = nll_scores.poisson(Y, rate, e[:, None] if rate.ndim == 2 and e.ndim == 1 else e)
        return per_cell if cells or per_cell.ndim == 1 else per_cell.sum(axis=1)

    def state(self) -> tuple:
        texts, meta = super().state()
        for c, cell in enumerate(meta['cells']):
            cell.update(base=float(self.base_[c]), n_events=int(self.n_events_[c]), degenerate=bool(self.degenerate_[c]))
        meta.update(squeeze=bool(self.squeeze_), n_zero_exposure=int(self.n_zero_exposure_))
        return texts, meta

    def _restore(self, texts: Mapping, meta: Mapping, layout=None) -> None:
        super()._restore(texts, meta, layout)
        self.squeeze_, self.n_zero_exposure_ = bool(meta['squeeze']), int(meta['n_zero_exposure'])
        self.base_ = np.array([cell['base'] for cell in meta['cells']], dtype=np.float64)
        self.n_events_ = np.array([cell['n_events'] for cell in meta['cells']], dtype=np.int64)
        self.degenerate_ = np.array([cell['degenerate'] for cell in meta['cells']], dtype=np.bool_)
        self.info_ = [{k: v for k, v in info.items() if k not in ('base', 'n_events', 'degenerate')} for info in self.info_]

    def summary(self) -> dict:
        out = super().summary()
        out.update(n_zero_exposure=int(self.n_zero_exposure_), n_events=[int(v) for v in self.n_events_],
                   degenerate=[bool(v) for v in self.degenerate_])
        return out


# =============================================================================
# 블록 6: 다중 분류 부스팅 헤드 (_CategoricalBoosted: 식 B1-4)
# =============================================================================
class _CategoricalBoosted(_Boosted):
    """A multiclass boosted model over ``N_CLASSES`` classes with the uniform mixing of ruling R13. ``fit(X, y)`` takes the design and the
    class of every row (0 .. C - 1, or -1 for "no category": left out and counted in ``n_missing_``).

    수식
    ----
    (B1-4)  p^model_{ic} = exp(F_c(x_i)) / Σ_{c'∈A} exp(F_{c'}(x_i))   (c ∈ A),      p^model_{ic} = 0   (c ∉ A)
            p_{ic} = (1 − ν)·p^model_{ic} + ν / C                                    (c = 0, …, C − 1; 섞기는 (B0-6), (HEAD-5)와 같다)
            F = (F_c)_{c∈A}는 Loss = −Σ_i log p^model_{i,c_i} (합은 c_i ≥ 0인 행)를 낮추는 나무의 합이다. |A| = 1이면 나무 없이 p^model = 1이다.

      기호         | 뜻                               | 단위      | 범위            | 코드 변수
      F_c(x_i)     | 범주 c의 나무 합(원점수)         | 로그 척도 | 유한            | booster.predict(raw_score=True)의 열
      A            | 훈련 행에 나온 범주의 집합       | —         | 1개 이상        | classes_
      p^model_{ic} | 부스터가 낸 범주 확률(A 밖은 0)  | —         | [0, 1]          | _output의 full
      p_{ic}       | 섞은 뒤의 범주 확률              | —         | > 0             | predict의 반환값
      ν, C         | 균등 분포의 몫과 범주의 수       | —, 개     | ν = 1e-6, C ≥ 1 | MIX, N_CLASSES
      c_i          | 행 i의 실제 범주(−1은 범주 없음) | —         | −1, 0, …, C − 1 | y
      Loss         | 부스팅 손실(다중 분류 로그 손실) | nat       | ≥ 0             | LightGBM 목적함수 'multiclass'

    해석
    ----
    식 (B1-4)는 범주 확률을 나무 합의 소프트맥스로 두고 로그 손실을 낮추는 다항 로지스틱 모형의 부스팅 판이다 [Friedman 2001]. B0의 (B0-5)와 같은 확률 족이므로 두 모형의 점수 (NLL-2)가 그대로 맞대어진다(계획
    P4). 부스터는 훈련 행에 나온 범주 A에서만 확률을 내고 나머지 범주는 균등 분포의 몫 ν / C만 가지므로, 훈련에 없던 범주가 나와도 점수 (NLL-2)가 유한하다(판정 R13). 범주가 −1(위치 없음)인 항목은 (NLL-2)가 읽지
    않으므로 적합에서 빼고(계획 전역 제약의 구역 z) 그 수를 센다(판정 R22). 범주가 하나뿐이면 LightGBM이 둘 이상을 요구하므로 나무 없는 상수 모형이다.

    근거
    ----
    계획 P4(범주 출력은 모든 모형이 같은 확률 족), Task 7(multiclass 목적함수), 판정 R13(균등 분포 섞기), R22(점수에서 뺀 행의 수), 계획 전역 제약의 구역 z(위치 없음은 ρ 항에서 뺀다), 설계서 3.1절(위치
    표지 ρ)과 8절(M2 쌍 표지 log loss, M3 규모).
    Ref: Friedman, J. H. (2001), Annals of Statistics 29(5): 1189-1232; doi:10.1214/aos/1013203451 (다중 분류 로지스틱 우도의 부스팅).
         Chen, S. F. and Goodman, J. (1999), Computer Speech and Language 13(4); doi:10.1006/csla.1999.0128 (균등 분포와의 보간).
    """
    objective = 'multiclass'
    metric = 'multi_logloss'
    N_CLASSES = 0
    _SCORE = staticmethod(nll_scores.categorical)

    def __init__(self, kind: str = 'masked', params=None):
        super().__init__(kind, params)
        self.classes_ = None
        self.n_missing_ = 0
        self.n_valid_unseen_ = 0
        self.n_valid_missing_ = 0

    def fit(self, X, y, *, categorical=(), layout=None, valid=None, check=None):
        """Fit on the rows that have a class. ``valid``: None, or ``(Xv, yv)`` the rows that early stopping reads (the STOP rows): the
        ones without a class (-1) or with a class that the training rows lack are left out and counted. ``X`` is an array or a
        ``_Design`` (which this call consumes).

        (B1-4)의 부스터를 훈련에 나온 범주 A로 맞춘다."""
        X = _design_matrix(X.take() if isinstance(X, _Design) else X)
        y = _int_codes('y', y)
        if y.shape != (len(X),):
            raise ValueError(f'{self.name}: the classes must have shape ({len(X)},), not {y.shape}')
        if y.min(initial=0) < -1 or y.max(initial=0) >= self.N_CLASSES:
            raise ValueError(f'{self.name}: the classes must lie in -1 .. {self.N_CLASSES - 1}')
        self._set_layout(layout, categorical, X)
        # 식 (B1-4)의 합은 범주가 있는 행(c_i ≥ 0)에 대해서이다. c_i = −1은 범주가 관측되지 않았으므로 뺀다(계획 전역 제약의 구역 z).
        keep = y >= 0
        self.n_missing_ = int((~keep).sum())
        if not keep.any():
            raise ValueError(f'{self.name}: no row has a class')
        if not keep.all():
            X, y = X[keep], y[keep]
        # A = 훈련에 나온 범주. 부스터는 A에 대해서만 맞추고, 나머지 범주는 (B1-4)의 균등 분포 몫만 갖는다.
        classes = np.unique(y)
        self.classes_ = classes
        self.n_fit_ = int(len(y))
        self.n_valid_ = self.n_valid_unseen_ = self.n_valid_missing_ = 0
        vdata = None
        if valid is not None:
            if len(valid) != 2:
                raise ValueError(f'{self.name}: valid must be (X, classes)')
            Xv, yv = _design_matrix(valid[0]), _int_codes('valid y', valid[1])
            if yv.shape != (len(Xv),) or Xv.shape[1] != X.shape[1]:
                raise ValueError(f'{self.name}: the validation rows must have {X.shape[1]} columns and one class each')
            if yv.min(initial=0) < -1 or yv.max(initial=0) >= self.N_CLASSES:
                raise ValueError(f'{self.name}: the validation classes must lie in -1 .. {self.N_CLASSES - 1}')
            known = np.isin(yv, classes)
            self.n_valid_missing_ = int((yv < 0).sum())
            self.n_valid_unseen_ = int(((yv >= 0) & ~known).sum())
            if not known.any():
                raise ValueError(f'{self.name}: no validation row has a class that the training rows have')
            vdata = (Xv[known], np.searchsorted(classes, yv[known]))
            self.n_valid_ = int(known.sum())
        if len(classes) == 1:                                    # one class: the constant model (LightGBM needs two)
            self.boosters_, self.info_ = [None], [_no_booster_info()]
            return self
        label = np.searchsorted(classes, y)                      # the dense numbering of the classes that occur
        valid_data = None if vdata is None else (vdata[0], [(vdata[1], None)])
        design = _Design(X)
        del X                                                    # one owner from here on: _boost frees the design once it is binned
        (booster, info), = self._boost(design, [(label, None)], valid_data, check, num_class=len(classes))
        self.boosters_, self.info_ = [booster], [info]
        return self

    def _output(self, Z: np.ndarray) -> np.ndarray:
        """The probabilities of the C classes: softmax over the classes that occurred, put on the full class set and mixed (ruling R13).

        (B1-4)의 소프트맥스와 균등 분포 섞기를 계산한다."""
        P = np.exp(Z - Z.max(axis=1, keepdims=True))
        P /= P.sum(axis=1, keepdims=True)
        full = np.zeros((len(Z), self.N_CLASSES))
        full[:, self.classes_] = P
        return (1.0 - MIX) * full + MIX / self.N_CLASSES

    def predict(self, X) -> np.ndarray:
        """(n, C) the probabilities of the C classes: full width, strictly positive, adding up to 1 (ruling R13).

        (B1-4)의 p_{ic}를 낸다."""
        self._need_fit()
        if self.boosters_[0] is None:
            Z = np.zeros((len(_design_matrix(X)), 1))
        else:
            X = _design_matrix(X)
            if X.shape[1] != self.n_features_:
                raise ValueError(f'{self.name}: the design must have {self.n_features_} columns, not shape {X.shape}')
            Z = np.empty((len(X), len(self.classes_)))
            for lo in range(0, len(X), _PREDICT_CHUNK):
                Z[lo:lo + _PREDICT_CHUNK] = self.boosters_[0].predict(X[lo:lo + _PREDICT_CHUNK], raw_score=True, num_threads=NUM_THREADS)
        return self._output(Z)

    def nll(self, X, y) -> np.ndarray:
        """-log p of the class of every row (``nll.categorical`` / ``nll.gold``); a row with the class -1 scores 0.

        (B1-4)의 p_{ic}로 점수 (NLL-2)(골드 머리는 (NLL-3))를 낸다."""
        return type(self)._SCORE(y, self.predict(X))

    def state(self) -> tuple:
        texts, meta = super().state()
        meta.update(classes=[int(c) for c in self.classes_], n_missing=int(self.n_missing_), n_valid_unseen=int(self.n_valid_unseen_),
                    n_valid_missing=int(self.n_valid_missing_))
        return texts, meta

    def _restore(self, texts: Mapping, meta: Mapping, layout=None) -> None:
        super()._restore(texts, meta, layout)
        self.classes_ = np.array(meta['classes'], dtype=np.int64)
        self.n_missing_, self.n_valid_unseen_ = int(meta['n_missing']), int(meta['n_valid_unseen'])
        self.n_valid_missing_ = int(meta['n_valid_missing'])

    def summary(self) -> dict:
        out = super().summary()
        out.update(n_missing=int(self.n_missing_), n_valid_unseen=int(self.n_valid_unseen_), classes=[int(c) for c in self.classes_])
        return out


# =============================================================================
# 블록 7: 8대 개별 헤드 (M2Rate, M2Mark, M2Zone, M3Size, M3Kills, M3Gold, M3Obj, M4Conv)
# =============================================================================
class M2Rate(_PoissonBoosted):
    """The first-kill rate per second of a (row, bin) cell: the design is [b1 row, q].

    (HEAD-1)의 λ_q를 (B1-2)로 낸다. 행은 (HR-1)의 (t₀ 행, 구간 q) 쌍이고 노출은 그 구간을 관측한 시간 Δ_q이다."""
    name = 'M2Rate'


class M2Mark(_CategoricalBoosted):
    """첫 킬의 표지 52개를 t₀ 행의 b1 특징과 구간 q로 예측한다. (HEAD-2)의 π(m | q)를 (B1-4)로 낸다."""
    name, N_CLASSES = 'M2Mark', N_M


class M2Zone(_CategoricalBoosted):
    """첫 킬의 구역 10개를 b1 특징과 q, m으로 예측한다. (HEAD-2)의 ρ(z | q, m)를 (B1-4)로 낸다. 위치를 모르는 첫 킬(z = −1)은 이 분포의 결과가 아니므로
    적합에서 빼고 센다(계획 전역 제약의 구역 z)."""
    name, N_CLASSES = 'M2Zone', N_ZONE


class M3Size(_CategoricalBoosted):
    """규모 4개를 예측한다: 가린 머리는 t₀ 행의 b1 특징과 q, m, z로, 열린 머리는 교전 사건 보기의 b1 특징만으로 예측한다(SLOTS). (HEAD-2)의 P(s | q, m, z)를 (B1-4)로 낸다. 입력의 z = −1은
    "위치 없음" 수준으로 읽는다(계획 전역 제약의 구역 z)."""
    name, N_CLASSES = 'M3Size', N_SIZE


class M3Kills(_PoissonBoosted):
    """The kills of the two teams (ruling R19): a Poisson head with two cells, 0 the blue and 1 the red team, on one row per onset; each
    cell is a booster of its own. ``predict`` gives the mean kills (n, 2), blue then red.

    (HEAD-3)의 μ_blue, μ_red를 (B1-2)로 낸다(노출 1). 두 팀의 칸은 부스터를 따로 둔다(판정 R19).

    ``kill_law`` (S4b 준비 M1, 결정 D5): ``'poisson'``(기본값)은 위의 머리 그대로이다. ``'shifted'``는 첫 킬 팀 d의 킬 수를 K_d = 1 + N_d로 둔다.

    수식
    ----
    (B1-2k)  y*_{i,t} = y_{i,t} − 1[t = d_i] ;   log λ*_{i,t} = γ_t + F_t(x_i) ,   γ_t = log(Σ_i y*_{i,t} / n),   LightGBM 목적함수 'poisson'의 라벨 = y*
             exp(γ_t + F_t(x_i)) = λ_{i,t} (d_i = t, 초과분 N_d의 평균) 또는 μ_{i,t} (d_i ≠ t, 킬 수의 평균) ;   점수 ℓ_{i,t}는 ``killlaw.logpmf``의 음수 (KL-3)

      기호      | 뜻                                                 | 단위 | 범위     | 코드 변수
      d_i       | 행 i의 첫 킬 팀 (0 파랑, 1 빨강, −1 처형)           | —    | −1, 0, 1 | first_team
      y*_{i,t}  | 칸 t의 유효 개수: 첫 킬 팀이면 y − 1, 아니면 y      | 개   | 0, 1, … | killlaw.excess

    해석: 포아송 목적함수는 라벨이 정수이기만 하면 되므로 첫 킬 팀 칸의 라벨을 y − 1로 두면 같은 목적함수가 λ_d를 적합한다. 한 칸이 두 종류의 행을 함께 보므로 나무가 d = t인 행과 아닌 행을
    가르는 입력이 있어야 한다: 가린 머리는 범주 열 m(d = m // 25)이, 열린 머리는 ``make_layout``이 더하는 범주 열 d가 그 입력이다. 조기 종료의 STOP 행도 같은 방식으로 y*를 라벨로 쓰므로
    (``valid_first_team``) 손실 (B1-5)가 적합한 모형과 같은 모형의 손실이다."""
    name = 'M3Kills'

    def __init__(self, kind: str = 'masked', params=None, kill_law: str = 'poisson'):
        super().__init__(kind, params)
        self.kill_law = killlaw.check_law(kill_law)

    def fit(self, X, Y, e, *, categorical=(), layout=None, valid=None, check=None, first_team=None, valid_first_team=None):
        """(B1-2) with the labels of the kill law: the counts ``Y`` for the plain law, their excess for the shifted law, whose first teams ``first_team`` (n,) and, with
        ``valid``, ``valid_first_team`` (n_valid,) are required (ValueError otherwise, and for a first team with no kill)."""
        if self.kill_law == 'shifted':
            if first_team is None or (valid is not None and valid_first_team is None):
                raise ValueError(f'{self.name}: the shifted kill law needs the first-kill team of every row of the fit and of the validation rows')
            Y = killlaw.excess(Y, first_team, self.kill_law)
            if valid is not None:
                valid = (valid[0], killlaw.excess(valid[1], valid_first_team, self.kill_law), valid[2])
        return super().fit(X, Y, e, categorical=categorical, layout=layout, valid=valid, check=check)

    def nll(self, X, Y, e, cells: bool = False, first_team=None) -> np.ndarray:
        """The NLL of the kill counts ``Y`` (n, 2): ``_PoissonBoosted.nll`` for the plain law, the shifted score (KL-3) with the first teams ``first_team`` (n,) for the shifted
        law (required; the exposure is 1). One value per row, or (n, 2) with ``cells``."""
        if self.kill_law == 'poisson':
            return super().nll(X, Y, e, cells)
        if first_team is None:
            raise ValueError(f'{self.name}: the shifted kill law needs the first-kill team of every row (first_team)')
        if not np.array_equal(np.asarray(e, dtype=np.float64), np.ones(len(np.asarray(Y)))):
            raise ValueError(f'{self.name}: the kill counts have the exposure 1')
        per_cell = -killlaw.logpmf(Y, self.predict(X), first_team, self.kill_law)
        return per_cell if cells else per_cell.sum(axis=1)

    def state(self) -> tuple:
        texts, meta = super().state()
        if self.kill_law != 'poisson':
            meta['kill_law'] = self.kill_law                       # absent means the plain law: a plain head is saved as every earlier head was
        return texts, meta

    def _restore(self, texts: Mapping, meta: Mapping, layout=None) -> None:
        super()._restore(texts, meta, layout)
        self.kill_law = killlaw.check_law(meta.get('kill_law', 'poisson'))

    def summary(self) -> dict:
        out = super().summary()
        if self.kill_law != 'poisson':
            out['kill_law'] = self.kill_law
        return out


class _GoldModel(_CategoricalBoosted):
    """The gold model of one team: a member of ``M3Gold``.

    (HEAD-3)의 골드 구간 분포를 (B1-4)로 맞춘 한 팀의 모형이다(범주 31개; 점수는 (NLL-3))."""
    name, N_CLASSES = 'M3Gold', N_GOLD
    _SCORE = staticmethod(nll_scores.gold)


class M3Gold:
    """The kill-gold bin of a team with at least one kill, 31 classes, with a multiclass model for each team (ruling R19). ``members`` are
    the models of the blue and of the red team (categorical heads: R13 mixing) and ``member(t)`` that of team t (0 blue, 1 red).
    ``fit(X, y, team, ...)`` fits each member on the rows of its team; ``predict(X, team)``, ``nll(X, y, team)`` and
    ``predict_blocks(numeric, team, **codes)`` use the member of the team of each row. A team with no row is an error.

    (HEAD-3)의 골드 구간 분포를 팀마다 (B1-4)로 맞춘다. 행은 킬이 k ≥ 1인 팀의 항목이고 킬 수 층 ξ = min(k, 6)이 입력 열이다((HR-3)). 킬이 0인 팀의 골드는 0으로 정해지므로
    그 팀은 행이 없다(계획 P4)."""
    name = 'M3Gold'
    N_CLASSES = N_GOLD

    def __init__(self, kind: str = 'masked', params=None):
        if kind not in KINDS:
            raise ValueError(f'kind must be one of {KINDS}, not {kind!r}')
        self.kind = kind
        self.members = (_GoldModel(kind, params), _GoldModel(kind, params))
        for team, member in zip(TEAMS, self.members):
            member._label = f'{self.name} ({team} team)'

    @property
    def parts(self) -> tuple:
        return self.members

    def member(self, team) -> _GoldModel:
        """The model of team ``team``: 0 blue, 1 red."""
        if isinstance(team, bool) or not isinstance(team, (int, np.integer)) or team not in (0, 1):
            raise ValueError(f'team must be 0 (blue) or 1 (red), not {team!r}')
        return self.members[int(team)]

    @staticmethod
    def _teams(team, n: int) -> np.ndarray:
        t = _int_codes('team', team)
        if len(t) != n:
            raise ValueError(f'team has {len(t)} entries but there are {n} rows')
        if t.size and (t.min() < 0 or t.max() > 1):
            raise ValueError('team must lie in 0 (blue) .. 1 (red)')
        return t

    def fit(self, X, y, team, *, categorical=(), layout=None, valid=None, check=None):
        """Fit the member of each team on the rows of that team. ``y`` is the gold bin (0 .. 30, -1 = none: left out and counted),
        ``team`` the team of every row (0 blue, 1 red). ``valid``: None, or ``(Xv, yv, teamv)``, split by team the same way. ``X`` is an
        array or a ``_Design``.

        팀의 항목만으로 팀마다 (B1-4)를 맞춘다. 한 팀의 항목이 하나도 없으면 그 팀의 모델을 만들 수 없어 거부한다(판정 R19)."""
        X = _design_matrix(X.take() if isinstance(X, _Design) else X)
        team = self._teams(team, len(X))
        y = _int_codes('y', y)
        if y.shape != (len(X),):
            raise ValueError(f'{self.name}: the classes must have shape ({len(X)},), not {y.shape}')
        if valid is not None:
            if len(valid) != 3:
                raise ValueError(f'{self.name}: valid must be (X, classes, team)')
            Xv = _design_matrix(valid[0])
            yv = _int_codes('valid y', valid[1])
            tv = self._teams(valid[2], len(Xv))
        for t, member in enumerate(self.members):
            rows = np.flatnonzero(team == t)
            if rows.size == 0:
                raise ValueError(f'{self.name}: no row of the {TEAMS[t]} team')
            v = None
            if valid is not None:
                vrows = np.flatnonzero(tv == t)
                if vrows.size == 0:
                    raise ValueError(f'{self.name}: no validation row of the {TEAMS[t]} team')
                v = (Xv[vrows], yv[vrows])
            member.fit(X[rows], y[rows], categorical=categorical, layout=layout, valid=v, check=check)
        return self

    def predict(self, X, team) -> np.ndarray:
        """(n, 31) the gold law of every row, from the model of its team."""
        X = _design_matrix(X)
        team = self._teams(team, len(X))
        out = np.empty((len(X), N_GOLD))
        for t, member in enumerate(self.members):
            rows = np.flatnonzero(team == t)
            if rows.size:
                out[rows] = member.predict(X[rows])
        return out

    def predict_blocks(self, numeric, team, **codes) -> np.ndarray:
        """(n, 31) the same from the numeric part and the raw categories (the composer's way)."""
        numeric = np.asarray(numeric)
        n = len(numeric)
        team = self._teams(team, n)
        arrays = {name: _int_codes(name, values) for name, values in codes.items()}
        for name, a in arrays.items():
            if len(a) != n:
                raise ValueError(f'{name} has {len(a)} entries but the numeric part has {n} rows')
        out = np.empty((n, N_GOLD))
        for t, member in enumerate(self.members):
            rows = np.flatnonzero(team == t)
            if rows.size:
                out[rows] = member.predict_blocks(numeric[rows], **{k: v[rows] for k, v in arrays.items()})
        return out

    def nll(self, X, y, team) -> np.ndarray:
        """-log p of the gold bin of every row under the model of its team; a row with the bin -1 (no kill) scores 0."""
        X = _design_matrix(X)
        team = self._teams(team, len(X))
        y = _int_codes('y', y)
        if y.shape != (len(X),):
            raise ValueError(f'{self.name}: the classes must have shape ({len(X)},), not {y.shape}')
        out = np.zeros(len(X))
        for t, member in enumerate(self.members):
            rows = np.flatnonzero(team == t)
            if rows.size:
                out[rows] = member.nll(X[rows], y[rows])
        return out

    # -- what the members have in common ---------------------------------------------------------------------------------------------------
    def _need_fit(self) -> None:
        for member in self.members:
            member._need_fit()

    @property
    def n_features_(self) -> int:
        return self.members[0].n_features_

    @property
    def categorical_(self) -> tuple:
        return self.members[0].categorical_

    @property
    def layout_(self):
        return self.members[0].layout_

    @property
    def n_fit_(self) -> int:
        return sum(m.n_fit_ for m in self.members)

    @property
    def n_missing_(self) -> int:
        return sum(m.n_missing_ for m in self.members)

    # -- storage ---------------------------------------------------------------------------------------------------------------------------
    def state(self) -> tuple:
        """(texts, meta): the texts of both members (prefixed ``blue.`` and ``red.``) and the meta of both under ``members``."""
        self._need_fit()
        texts, members = {}, {}
        for team, member in zip(TEAMS, self.members):
            member_texts, members[team] = member.state()
            texts.update({f'{team}.{part}': text for part, text in member_texts.items()})
        return texts, {'class': type(self).__name__, 'kind': self.kind, 'n_fit': int(self.n_fit_), 'members': members}

    @classmethod
    def from_state(cls, texts: Mapping, meta: Mapping, layout=None) -> 'M3Gold':
        head = cls(meta['kind'], meta['members']['blue']['params'])
        for i, team in enumerate(TEAMS):
            prefix = f'{team}.'
            head.members[i]._restore({k[len(prefix):]: v for k, v in texts.items() if k.startswith(prefix)}, meta['members'][team], layout)
        return head

    def summary(self) -> dict:
        """The counts of the fit as plain JSON values: the lists have one entry for each team (blue, red), ``teams`` the summary of each
        member."""
        self._need_fit()
        teams = {team: member.summary() for team, member in zip(TEAMS, self.members)}
        first = {key: [t[key][0] for t in teams.values()] for key in ('best_iteration', 'num_trees', 'rounds_run', 'stopped_early')}
        return {'kind': self.kind, 'n_fit': int(self.n_fit_), 'n_valid': sum(t['n_valid'] for t in teams.values()),
                'width': teams['blue']['width'], 'categorical': teams['blue']['categorical'], **first,
                'n_missing': int(self.n_missing_), 'teams': teams}


class M3Obj(_PoissonBoosted):
    """The objectives of each cell (18 Poisson models, exposure 1): masked: [b1 row, K, q, m, z, s]; open: [b1 row, K, s].

    (HEAD-4)의 포아송 평균 μ_c를 칸마다 (B1-2)로 낸다. 가린 M3Obj가 K를 읽는 것은 계획 P6이다: 학습에서는 실제 K, 합성에서는 뽑은 K이다. 열린 M3Obj는 설계서 3.2절의 분해 P(O | K, s)에 따라 실제 K를
    읽는다."""
    name = 'M3Obj'


class M4Conv(_PoissonBoosted):
    """The objectives converted after the engagement (18 Poisson models, the exposure is the conversion window in seconds): masked: the
    matrix of ``features.m4_masked_inputs``; open: [b1 row, K, O].

    (HEAD-4)의 전환율 λ_c를 칸마다 (B1-2)로 낸다. 노출은 전환 창의 길이(초)이다. 가린 머리의 입력은 (B1-1)의 ``features.m4_masked_inputs`` 행렬이다(계획 P6)."""
    name = 'M4Conv'


HEAD_CLASSES = {cls.name: cls for cls in (M2Rate, M2Mark, M2Zone, M3Size, M3Kills, M3Gold, M3Obj, M4Conv)}
if tuple(HEAD_CLASSES) != HEAD_NAMES or not set(OPEN_HEAD_NAMES) <= set(HEAD_NAMES) or set(SLOTS) != head_rows.VALID_HEADS:
    raise RuntimeError('the head classes, HEAD_NAMES and the designs differ')


# =============================================================================
# 블록 8: 훈련 및 평가용 행 집합 추출기 (HeadRows)
# =============================================================================
class HeadRows(head_rows.HeadRows):
    """The rows that the B1 heads are trained or scored on, for the matches of one role of an (S1, S2, labels) triple: the rows of
    ``head_rows.HeadRows`` with the b1 features without the columns ``dropped`` (``features.dropped_columns``, from the FIT views).

    ``data(head, kind)`` gives the ``HeadData`` of one head, the design being a float32 matrix of the slots of ``SLOTS`` and the layout a
    ``Layout``. ValueError for an S2 store or label store that does not belong to the S1 store (counts only, never a key), for a role that
    is a holdout role or not used by S3, for a dropped name that is no column, and for a unit that lacks a view.

    ``kill_law`` and ``k_condition`` (S4b preparation M1 and M3) go to ``head_rows.HeadRows``: the kill law of the M3Kills rows, and the gold of the K block of the object and
    conversion rows ('raw' the true gold of the labels, 'binned' the representative of the observed gold bin, ``head_rows`` equation (HR-9)); the defaults are the rows of every
    earlier run.

    행의 정의는 (HR-1)–(HR-4)이고, 이 클래스는 거기에 b1 특징(결측 칸은 NaN으로 둔다)과 SLOTS의 설계를 얹는다(판정 R18)."""

    def __init__(self, s1, s2, labels, dropped, role, *, matches=None, kill_law: str = 'poisson', k_condition: str = 'raw'):
        self.dropped = tuple(dropped) if not isinstance(dropped, str) else dropped
        self.columns = features.b1_columns(self.dropped)                       # refuses a string and a name that is no column
        super().__init__(s1, s2, labels, role, lambda store, views: features.b1(store, views, self.dropped), matches=matches, kill_law=kill_law,
                         k_condition=k_condition)

    def data(self, head: str, kind: str) -> HeadData:
        """The rows of the head ``head`` ('M2Rate', ... 'M4Conv') of ``kind`` ('masked' or 'open'), see ``HeadData``. ValueError for a head
        and kind that do not exist (M2 has no open heads).

        (B1-1)의 설계 X와 (HR-1)–(HR-5)의 y, e, 경기 번호, 평가 행 번호를 묶어 ``HeadData``로 낸다. 가린 오브젝트 머리는 K를 읽는다(``masked_obj_k``, 계획 P6)."""
        if (head, kind) not in SLOTS:
            raise ValueError(f'there is no {kind} head {head!r}; the heads are {[k for k in SLOTS]}')
        layout = make_layout(head, kind, self.columns, self.kill_law)
        p = self.parts(head, kind, masked_obj_k=True)                          # B1 conditions the masked object head on K (decision P6)
        if (head, kind) == ('M4Conv', 'masked'):
            X = _m4_design(p.m4, layout.d)                                     # the matrix of features.m4_masked_inputs, zone shifted
        else:
            X = _assemble(layout.slots, p.feat, p.take, p.codes, p.K, p.O)
        return HeadData(X, p.y, p.e, p.match, p.row, layout, p.n_zero_exposure, p.team, p.first_team)


# =============================================================================
# 블록 9: B2 다음 1분 발생 이진 분류기 (식 B1-6 ~ B1-8)
# =============================================================================
def valid_rows(s1, role='STOP') -> np.ndarray:
    """(n frame rows,) bool: the frame rows of the matches of ``role`` (the STOP role, as the early stopping of B2 and M2Rate reads only
    these). The other heads read the STOP onsets or units through ``HeadRows(..., 'STOP')``.

    조기 종료 (B1-5)가 읽는 역할 role(기본 STOP)의 프레임 행을 표시한다(분할은 계획 전역 제약의 역할 표가 정한다)."""
    return features.split_rows(s1, role)


def b2_target(s1, rows) -> np.ndarray:
    """(len(rows),) bool: whether the frame row has at least one onset in the bins 0 .. 5 (the first kill of an engagement within 60 s of
    the row: decision P5). ``rows``: indices into ``frame_rows``. ValueError for a row that is not in the store.

    수식
    ----
    (B1-6)  y_i = 1[Σ_{q=0}^{5} n_q ≥ 1]                          (프레임 행 i의 앞 60초, 곧 구간 0–5 안에 첫 킬이 하나 이상)

      기호 | 뜻                                      | 단위 | 범위    | 코드 변수
      y_i  | 프레임 행 i의 목표(앞 60초 안 발생 여부) | —    | 0, 1    | 반환값의 원소
      n_q  | 프레임 행 i의 구간 q의 발생 수           | 개   | 0, 1, … | 발생 표의 구간 번호(``onsets_bin``)

    해석
    ----
    식 (B1-6)은 M2가 확률을 주는 바로 그 사건, 곧 앞 60초 안에 교전의 첫 킬이 하나 이상 있었는가를 0 또는 1로 센다. 목표가 M2와 같아야 M2의 60초 발생 확률과 B2의 확률을 같은 사건의 Brier 점수로 맞댈 수 있다(계획
    P5). 발생이 둘 이상이어도 목표는 1이어서 발생의 수가 아니라 여부를 맞히며, 구간 6 이후의 발생은 목표에 들어가지 않는다. 경기 끝 T − 30 s 뒤의 구간에는 발생이 없으므로(노출 0) 그런 행의 목표는 관측된 구간만으로
    정해진다(설계서 3.1절 경기 끝).

    근거
    ----
    계획 P5(B2의 목표), 설계서 3.1절(발생 구간과 평가 지평)과 8절(보조 결과: M2 대 B2의 발생 Brier).
    """
    onsets_row = np.asarray(s1.array('onsets/row')).astype(np.int64)
    onsets_bin = np.asarray(s1.array('onsets/bin')).astype(np.int64)
    n_rows = len(np.asarray(s1.array('frame_rows/t0')))
    rows = _int_codes('rows', rows)
    if rows.size and (rows.min() < 0 or rows.max() >= n_rows):
        raise ValueError(f'rows must lie in 0 .. {n_rows - 1} (the frame rows of the S1 store)')
    if onsets_row.size and (onsets_row.min() < 0 or onsets_row.max() >= n_rows):
        raise ValueError('onsets/row names a frame row that the S1 store does not have')
    hit = np.zeros(n_rows, dtype=np.bool_)
    hit[onsets_row[onsets_bin < B2_BINS]] = True
    return hit[rows]


class B2Rows(NamedTuple):
    """The B2 forecasts of the frame rows of a role: ``p`` the probability of an onset within 60 s, ``y`` the target, ``match`` the
    match number and ``row`` the frame row of each. ``nll()`` and ``brier()`` are the scores of every row; their match-weighted means
    are ``nll_scores.weighted_mean(values, features.match_weights(match))``.

    수식
    ----
    (B1-8)  ℓ_i = −log p_i  (y_i = 1),    ℓ_i = −log(1 − p_i)  (y_i = 0)                     ((NLL-2)에서 C = 2, c_i = y_i, 확률 (1 − p_i, p_i))
            Brier 점수 = (p_i − y_i)²

      기호 | 뜻                                          | 단위 | 범위   | 코드 변수
      p_i  | 프레임 행 i의 앞 60초 안 발생 확률(섞은 뒤)   | —    | (0, 1) | p
      y_i  | 목표 (B1-6)                                 | —    | 0, 1   | y
      ℓ_i  | 행 i의 음의 로그 확률                        | nat  | ≥ 0    | nll()

    해석
    ----
    식 (B1-8)의 ℓ_i는 맞힌 쪽 확률의 음의 로그이며 범주가 둘인 (NLL-2)이다. 프레임 행 하나가 평가 행 하나이므로 행의 점수는 v_I = ℓ_i이고((HR-5)) 경기 가중 평균 (NLL-4)와 경기 부트스트랩
    (NLL-5)로 이어진다. Brier 점수는 확률과 0 또는 1인 결과의 차이의 제곱이며 엄격한 적정 점수이다 [Brier 1950; Gneiting & Raftery 2007]. M2의 60초 발생 확률과 이 점수로 맞댄다(계획 P5).
    B2의 예측 (B1-7)은 균등 분포를 섞어 0과 1 사이의 열린 구간에 있으므로 ℓ_i가 유한하다(판정 R13).

    근거
    ----
    계획 P5(B2와 M2를 같은 사건의 Brier로 맞댄다), 판정 R13(균등 분포 섞기), 설계서 8절(보조 결과: M2 대 B2).
    Ref: Brier, G. W. (1950), Monthly Weather Review 78(1): 1-3; doi:10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2 (확률 예보의 점수).
         Gneiting, T. and Raftery, A. E. (2007), JASA 102(477): 359-378; doi:10.1198/016214506000001437 (로그 점수와 Brier 점수는 엄격한 적정 점수).
    """
    p: np.ndarray
    y: np.ndarray
    match: np.ndarray
    row: np.ndarray

    def nll(self) -> np.ndarray:
        return -np.log(np.where(self.y, self.p, 1.0 - self.p))

    def brier(self) -> np.ndarray:
        return (self.p - self.y.astype(np.float64)) ** 2

    def scores(self) -> HeadScores:
        """The NLL of the rows as ``HeadScores`` (mean, bootstrap).

        (B1-8)의 ℓ_i를 평가 행 점수 v_I로 담는다((HR-5))."""
        return HeadScores(self.nll(), self.match, self.row, 0)


class B2(_Boosted):
    """The next-minute onset classifier: a binary LightGBM model of the b1 row of a frame-clock row (``fit``, ``predict``, ``nll``,
    ``brier``). ``fit_stores`` fits it on the frame rows of the FIT role with the STOP rows for the early stopping, ``score_role`` gives
    its forecasts for a role. The probability is mixed with the uniform law over {0, 1} (ruling R13).

    수식
    ----
    (B1-7)  p^model_{i} = 1 / (1 + exp(−F(x_i))),    p_i = (1 − ν)·p^model_{i} + ν / 2
            F는 Loss = −Σ_i [y_i·log p^model_{i} + (1 − y_i)·log(1 − p^model_{i})]를 낮추는 나무의 합이다(그래디언트 부스팅).
            y_i가 모든 행에서 같은 값이면 부스터 없이 p^model_{i} = (N_1 + 1/2) / (N + 1)이다.

      기호        | 뜻                                            | 단위      | 범위     | 코드 변수
      F(x_i)      | 나무 합(원점수)                               | 로그 오즈 | 유한     | booster.predict(raw_score=True)
      p^model_{i} | 부스터가 낸 앞 60초 안 발생 확률              | —         | (0, 1)   | expit(raw), 상수 모형은 constant_
      p_i         | 섞은 뒤의 확률                                | —         | > 0      | predict의 반환값
      y_i         | 프레임 행 i의 목표 (B1-6)                     | —         | 0, 1     | y
      N_1, N      | 목표가 1인 행의 수 Σ_i y_i, 적합에 쓴 행의 수 | 행        | 0–N, ≥ 1 | y.sum(), len(y)
      ν           | 균등 분포의 몫                                | —         | 1e-6     | MIX
      Loss        | 부스팅 손실(이진 로그 손실)                   | nat       | ≥ 0      | LightGBM 목적함수 'binary'

    해석
    ----
    식 (B1-7)은 목표가 0 또는 1인 경우의 (B1-4)이다: 범주가 둘일 때 소프트맥스는 로지스틱 함수이고 목적함수는 이진 로그 손실이며, 균등 분포의 몫은 ν / C 자리에 ν / 2이다(판정 R13). 목표가 프레임 행 하나에 대한
    것이므로 입력은 b1 행뿐이고 구간이나 표지 열은 없다(계획 P5). 목표가 모든 행에서 같은 값이면 부스터를 두지 않고 상수 모형을 쓴다. (N_1 + 1/2) / (N + 1)은 Beta(1/2, 1/2) 사전분포의 사후 평균이어서 0이나
    1이 되지 않는다 [도출].

    근거
    ----
    계획 P5(B2의 목표), P9(고정 모수), 판정 R13(균등 분포 섞기), 설계서 6.4절(B2는 필수 기준 모형)과 8절(보조 결과: 발생 Brier).
    Ref: Friedman, J. H. (2001), Annals of Statistics 29(5): 1189-1232; doi:10.1214/aos/1013203451 (그래디언트 부스팅, 이진 로지스틱 우도).
         Chen, S. F. and Goodman, J. (1999), Computer Speech and Language 13(4); doi:10.1006/csla.1999.0128 (균등 분포와의 보간).
    """
    name = 'B2'
    objective = 'binary'
    metric = 'binary_logloss'

    def __init__(self, params=None):
        super().__init__('masked', params)
        self.dropped_ = None                  # the columns left out of b1 (set by ``fit_stores`` and ``load``)
        self.constant_ = None                 # the probability of a model whose target had one value only
        self.fit_roles_ = self.stop_roles_ = ()

    def fit(self, X, y, *, categorical=(), layout=None, valid=None, check=None):
        """Fit on the rows ``X`` with the 0/1 target ``y``. ``valid``: None, or ``(Xv, yv)`` the rows that early stopping reads (the
        STOP frame rows). A target with one value only gives the constant model with the smoothed frequency (N_1 + 1/2) / (N + 1).

        (B1-7)의 부스터를 맞추고, 목표가 한 값뿐이면 상수 모형을 둔다."""
        X = _design_matrix(X.take() if isinstance(X, _Design) else X)
        y = np.asarray(y)
        if y.shape != (len(X),) or y.dtype.kind not in 'biu' or (y.dtype.kind != 'b' and not np.isin(y, (0, 1)).all()):
            raise ValueError(f'{self.name}: the target must be one 0 or 1 for each row')
        y = y.astype(np.int64)
        self._set_layout(layout, categorical, X)
        self.n_fit_, self.n_valid_ = int(len(y)), 0
        vdata = None
        if valid is not None:
            if len(valid) != 2:
                raise ValueError(f'{self.name}: valid must be (X, target)')
            Xv, yv = _design_matrix(valid[0]), np.asarray(valid[1])
            if yv.shape != (len(Xv),) or Xv.shape[1] != X.shape[1] or yv.dtype.kind not in 'biu' or not np.isin(yv, (0, 1)).all():
                raise ValueError(f'{self.name}: the validation rows must have {X.shape[1]} columns and a 0 or 1 target each')
            vdata = (Xv, [(yv.astype(np.int64), None)])
            self.n_valid_ = int(len(yv))
        if y.min() == y.max():
            self.constant_ = float((y.sum() + 0.5) / (len(y) + 1.0))
            self.boosters_, self.info_ = [None], [_no_booster_info()]
            return self
        self.constant_ = None
        (booster, info), = self._boost(_Design(X), [(y, None)], vdata, check)
        self.boosters_, self.info_ = [booster], [info]
        return self

    def predict(self, X) -> np.ndarray:
        """(n,) the probability of at least one onset within 60 s, strictly between 0 and 1.

        (B1-7)의 p_i를 낸다."""
        self._need_fit()
        X = _design_matrix(X)
        if self.boosters_[0] is None:
            p = np.full(len(X), self.constant_)
        else:
            if X.shape[1] != self.n_features_:
                raise ValueError(f'{self.name}: the design must have {self.n_features_} columns, not shape {X.shape}')
            p = np.empty(len(X))
            for lo in range(0, len(X), _PREDICT_CHUNK):
                raw = self.boosters_[0].predict(X[lo:lo + _PREDICT_CHUNK], raw_score=True, num_threads=NUM_THREADS)
                p[lo:lo + _PREDICT_CHUNK] = expit(raw)
        return (1.0 - MIX) * p + MIX / 2

    def _target(self, y, n: int) -> np.ndarray:
        y = np.asarray(y)
        if y.shape != (n,) or y.dtype.kind not in 'biu' or not np.isin(y, (0, 1)).all():
            raise ValueError(f'{self.name}: the target must be one 0 or 1 for each row')
        return y.astype(np.bool_)

    def nll(self, X, y) -> np.ndarray:
        """-log p of the target of every row.

        (B1-8)의 ℓ_i이다."""
        p = self.predict(X)
        y = self._target(y, len(p))
        return -np.log(np.where(y, p, 1.0 - p))

    def brier(self, X, y) -> np.ndarray:
        """(p - y)^2 of every row.

        (B1-8)의 Brier 점수이다."""
        p = self.predict(X)
        return (p - self._target(y, len(p)).astype(np.float64)) ** 2

    # -- rows ------------------------------------------------------------------------------------------------------------------------------
    @classmethod
    def fit_stores(cls, s1, s2, dropped, roles=None, *, check=None, params=None) -> 'B2':
        """Fit on the frame rows of the role ``roles['fit']`` (default FIT) with the frame rows of ``roles['stop']`` (default STOP) for the
        early stopping. ``dropped``: the columns left out of b1. ``check``: the E6 guard, called before anything is read and before the
        booster. ValueError for a holdout role or a role that S3 does not use, before any array is read. B2 reads no label store.

        기본값은 FIT 역할의 프레임 행으로 맞추고 STOP 역할의 프레임 행으로 멈추는 것이다(판정 R21의 roles 서명). 목표 (B1-6)이 발생 표에서 나오므로 라벨 저장소를 읽지 않는다."""
        fit_role, stop_role = _fit_stop_roles(roles)
        tick = check if check is not None else (lambda: None)
        tick()
        fit_rows = HeadRows(s1, s2, None, dropped, fit_role)
        stop_rows = HeadRows(s1, s2, None, dropped, stop_role)
        fit_frame, stop_frame = fit_rows.frame(), stop_rows.frame()
        model = cls(params)
        model.dropped_ = tuple(dropped)
        layout = Layout(('feat',), fit_rows.columns, (), len(fit_rows.columns))
        tick()
        model.fit(fit_frame.feat, b2_target(s1, fit_frame.rows), layout=layout, valid=(stop_frame.feat, b2_target(s1, stop_frame.rows)),
                  check=check)
        model.fit_roles_, model.stop_roles_ = _as_tuple(fit_role), _as_tuple(stop_role)
        return model

    def score_role(self, s1, s2, role, *, check=None) -> B2Rows:
        """The forecasts of the frame rows of ``role`` (DEV, say) as ``B2Rows``.

        (B1-7)의 p_i와 (B1-6)의 y_i를 한 역할의 프레임 행에서 낸다. 평가 행은 프레임 행 하나이다((HR-5))."""
        if self.dropped_ is None:
            raise ValueError('B2.score_role needs the dropped columns: fit the model with fit_stores or load it')
        if check is not None:
            check()
        rows = HeadRows(s1, s2, None, self.dropped_, role).frame()
        return B2Rows(self.predict(rows.feat), b2_target(s1, rows.rows), rows.match, rows.rows)

    # -- storage ---------------------------------------------------------------------------------------------------------------------------
    def save(self, directory) -> Path:
        """Write ``b2.json`` and ``models/b2.txt`` into ``directory`` (created). FileExistsError if either file is there.

        적합한 B2를 모형 텍스트와 그 SHA-256을 담은 JSON 한 쌍으로 남긴다."""
        self._need_fit()
        out = Path(directory)
        if (out / B2_JSON_NAME).exists() or (out / MODEL_DIR / 'b2.txt').exists():
            raise FileExistsError(f'{out} already holds a saved B2: it is not written over')
        texts, meta = self.state()
        meta.update(constant=self.constant_, dropped=None if self.dropped_ is None else list(self.dropped_),
                    fit_roles=list(self.fit_roles_), stop_roles=list(self.stop_roles_))
        (out / MODEL_DIR).mkdir(parents=True, exist_ok=True)
        files = {}
        for part, text in texts.items():
            blob = text.encode('utf-8')
            store_core.write_atomic(out / MODEL_DIR / 'b2.txt', blob)
            files[part] = {'file': f'{MODEL_DIR}/b2.txt', 'sha256': hashlib.sha256(blob).hexdigest()}
        doc = {'schema': SCHEMA, 'lightgbm': lgb.__version__, 'model': 'B2', 'meta': meta, 'files': files}
        store_core.write_atomic(out / B2_JSON_NAME, json.dumps(doc, indent=2, sort_keys=True, allow_nan=False) + '\n')
        return out

    @classmethod
    def load(cls, directory) -> 'B2':
        """Read a B2 saved by ``save``. ValueError for another schema or a model file whose SHA-256 differs from the JSON's.

        스키마와 SHA-256이 저장 때와 같을 때만 B2를 돌려준다."""
        folder = Path(directory)
        doc = json.loads((folder / B2_JSON_NAME).read_text(encoding='utf-8'))
        if doc.get('schema') != SCHEMA or doc.get('model') != 'B2':
            raise ValueError(f'{folder / B2_JSON_NAME}: the schema is {doc.get("schema")!r}, not a B2 of {SCHEMA!r}')
        texts = {}
        for part, entry in doc['files'].items():
            path = folder / entry['file']
            if store_core.sha(path) != entry['sha256']:
                raise ValueError(f'{path}: the model file has changed (its sha256 differs from the one in the JSON)')
            texts[part] = path.read_bytes().decode('utf-8')
        meta = doc['meta']
        model = cls(meta['params'])
        model._restore(texts, meta)
        model.constant_ = meta['constant']
        model.dropped_ = None if meta['dropped'] is None else tuple(meta['dropped'])
        model.fit_roles_, model.stop_roles_ = tuple(meta['fit_roles']), tuple(meta['stop_roles'])
        return model


def _as_tuple(role) -> tuple:
    return (role,) if isinstance(role, str) else tuple(role)


def _fit_stop_roles(roles) -> tuple:
    """The role(s) to fit on and the role(s) to stop on from ``roles`` = {'fit': ..., 'stop': ...}; the default is FIT and STOP (ruling
    R21). ValueError when the two overlap: the early stopping must not read the rows that the model is fitted on. It does not forbid DEV
    as the stopping role (the early stopping would then read the rows that the gate is judged on): the stage S3 runner always gives STOP
    (``run_s3.B1_ROLES``) and a test of the runner pins that, so nothing here makes the choice."""
    if roles is None:
        roles = {'fit': 'FIT', 'stop': 'STOP'}
    if not isinstance(roles, Mapping):
        raise TypeError("roles must be a mapping {'fit': role name(s), 'stop': role name(s)}")
    if set(roles) != {'fit', 'stop'}:
        raise ValueError("roles must have exactly the keys 'fit' and 'stop'")
    if set(_as_tuple(roles['fit'])) & set(_as_tuple(roles['stop'])):
        raise ValueError("the 'stop' role must not be one of the 'fit' roles: the early stopping reads rows the model was not fitted on")
    return roles['fit'], roles['stop']


# =============================================================================
# 블록 10: 합성기 연동 프로토콜 어댑터 (_MaskedChain, ComposerHeads 구현)
# =============================================================================
class _MaskedChain:
    """The masked B1 heads as the composer reads them (``heads.ComposerHeads``): no closed form (``analytic_ok`` False), K is read by the
    object head and K and O by the conversion head (decision P6).

    각 방법은 B1의 부스터로 (HEAD-1)–(HEAD-4)의 한 고리를 낸다: ``rate``는 구간 18개마다 설계 행을 만들어 (B1-2)로, ``mark_prob``·``zone_prob``·``size_prob``는 (B1-4)로,
    ``kill_rate``는 파랑과 빨강 두 칸의 (B1-2)로, ``gold_prob``는 팀마다 그 팀 자신의 킬 수 층을 주고 (B1-4)로, ``obj_rate``는 K를 읽는 (B1-2)로, ``conv_rate``는
    ``features.m4_masked_inputs``의 행렬을 읽는 (B1-2)로 낸다."""
    feature_kind = 'b1'
    analytic_ok = False
    obj_uses_K = True
    conv_uses_KO = True

    def __init__(self, masked: Mapping, dropped: tuple):
        self._h = masked
        self._dropped = tuple(dropped)
        self._columns = features.b1_columns(self._dropped)
        self._b1 = None
        # 킬 수 법칙(결정 D5, 검토 I1): ``kill_rate``의 d 칸이 초과분의 평균 λ_d('shifted')인지 킬 수의 평균 μ('poisson')인지는 M3Kills 머리가 정한다. 합성기가 이 선언을 자기 법칙과
        # 대조하므로(``Composer``) 다른 법칙으로 맞춘 머리를 말없이 합성하지 못한다. 킬 머리가 없으면(부분 묶음) 옛 법칙이다.
        self.kill_law = getattr(masked.get('M3Kills'), 'kill_law', 'poisson')

    def bind(self, s2, views) -> None:
        """Compute and keep the b1 rows of the frame-clock views ``views`` (view numbers in any order, or a bool mask). ValueError for a
        view that is not a frame-clock view (``views/src`` 0): an event view holds the events up to the first kill or the end of the
        engagement, which a masked head must never see (A3).

        b1 행의 NaN은 채우지 않고 LightGBM이 결측으로 읽게 둔다((B1-1))."""
        b1 = features.b1(s2, views, self._dropped)
        v = np.asarray(views)
        chosen = np.flatnonzero(v) if v.dtype == np.bool_ else v.astype(np.int64)
        try:
            src = np.asarray(s2.array('views/src'))[chosen]
        except FileNotFoundError:
            raise ValueError('the S2 store has no array views/src') from None
        if (src != 0).any():
            raise ValueError(f'bind takes frame-clock views (views/src 0) only, but {int((src != 0).sum())} of the {len(chosen)} views are '
                             'event views')
        self._b1 = b1

    def _set_rows(self, row_view) -> None:
        """Use the b1 rows ``row_view`` (n, d) as the bound rows (no S2 store): the rows of a ``HeadBatch``. They are float32, as the
        rows that ``bind`` computes are."""
        A = np.asarray(row_view)
        if A.dtype.kind not in 'fiu' or A.ndim != 2 or A.shape[1] != len(self._columns):
            raise ValueError(f'row_view must be a numeric array of shape (n, {len(self._columns)}), not {A.dtype} {A.shape}')
        self._b1 = narrow('row_view', A.astype(np.float64), np.float32)

    def _rows(self, rows) -> np.ndarray:
        if self._b1 is None:
            raise RuntimeError('call bind(s2, views) before asking the heads for rows')
        r = _int_codes('rows', rows)
        if r.size and (r.min() < 0 or r.max() >= len(self._b1)):
            raise ValueError(f'rows must lie in 0 .. {len(self._b1) - 1} (the bound views)')
        return r

    def row_view(self, rows) -> np.ndarray:
        return self._b1[self._rows(rows)].copy()

    @staticmethod
    def _same_length(n: int, **named) -> None:
        for name, a in named.items():
            if len(a) != n:
                raise ValueError(f'{name} has {len(a)} entries but rows has {n}')

    @staticmethod
    def _slots(head) -> tuple:
        if head.layout_ is None:
            raise ValueError(f'{head.name} was fitted without a layout, so the composer cannot make its design')
        return head.layout_.slots

    def _predict(self, head, take, codes, K=None) -> np.ndarray:
        """The head on the designs of the bound rows ``take`` (with their raw categories and K in thousands), in chunks of design rows."""
        slots = self._slots(head)
        if len(take) == 0:
            return np.empty((0,) + _OUT_SHAPE[head.name])
        out = []
        for lo in range(0, len(take), _PREDICT_CHUNK):
            hi = lo + _PREDICT_CHUNK
            X = _assemble(slots, self._b1, take[lo:hi], {k: v[lo:hi] for k, v in codes.items()}, None if K is None else K[lo:hi])
            out.append(head.predict(X))
        return np.concatenate(out)

    def rate(self, rows) -> np.ndarray:
        r = self._rows(rows)
        slots = self._slots(self._h['M2Rate'])
        out = np.empty((len(r), N_Q))
        # 식 (HEAD-1)의 λ_q를 18구간 모두에 낸다: 한 행마다 q = 0, …, 17인 설계 행 18개를 만들어 M2Rate에 묻는다. 노출을 곱하지 않은 초당 값이다(계획 P10).
        step = max(1, _PREDICT_CHUNK // N_Q)                               # rows per call: step x 18 design rows
        for lo in range(0, len(r), step):
            sel = r[lo:lo + step]
            X = _assemble(slots, self._b1, np.repeat(sel, N_Q), {'q': np.tile(np.arange(N_Q), len(sel))})
            out[lo:lo + step] = self._h['M2Rate'].predict(X).reshape(len(sel), N_Q)
        return out

    def _codes(self, rows, **named) -> tuple:
        r = self._rows(rows)
        codes = {name: _int_codes(name, values) for name, values in named.items()}
        self._same_length(len(r), **codes)
        return r, codes

    def mark_prob(self, rows, q) -> np.ndarray:
        r, codes = self._codes(rows, q=q)
        return self._predict(self._h['M2Mark'], r, codes)

    def zone_prob(self, rows, q, m) -> np.ndarray:
        r, codes = self._codes(rows, q=q, m=m)
        return self._predict(self._h['M2Zone'], r, codes)

    def size_prob(self, rows, q, m, z) -> np.ndarray:
        r, codes = self._codes(rows, q=q, m=m, z=z)
        return self._predict(self._h['M3Size'], r, codes)

    def kill_rate(self, rows, q, m, z, s) -> np.ndarray:
        r, codes = self._codes(rows, q=q, m=m, z=z, s=s)
        return self._predict(self._h['M3Kills'], r, codes)                  # (n, 2): the blue and the red cell

    def gold_prob(self, rows, q, m, z, s, layer) -> np.ndarray:
        r, codes = self._codes(rows, q=q, m=m, z=z, s=s)
        layer = np.asarray(layer)
        if layer.shape != (len(r), 2):
            raise ValueError(f'layer must have shape ({len(r)}, 2), not {layer.shape}')
        out = np.empty((len(r), 2, N_GOLD))
        # 식 (HEAD-3)의 골드 구간 분포: 팀마다 그 팀의 모델에 그 팀의 킬 수 층 ξ를 준다.
        for team in (0, 1):                                                # the model of each team, on the kill layer of that team
            member = self._h['M3Gold'].member(team)
            out[:, team] = self._predict(member, r, {**codes, 'layer': _int_codes('layer', layer[:, team])})
        return out

    def obj_rate(self, rows, q, m, z, s, K) -> np.ndarray:
        r, codes = self._codes(rows, q=q, m=m, z=z, s=s)
        K = np.array(K, dtype=np.float64)                                  # a copy: the gold is divided by 1,000 below
        if K.shape != (len(r), N_K):
            raise ValueError(f'K must have shape ({len(r)}, {N_K}), not {K.shape}')
        # 식 (HR-4)의 K 블록은 골드를 천 골드 단위로 읽는다. 훈련 행의 K 블록과 같은 단위이다.
        K[:, 2:] /= 1000.0
        return self._predict(self._h['M3Obj'], r, codes, K=K)               # K is read: decision P6

    def conv_rate(self, m4_inputs) -> np.ndarray:
        d = len(self._columns)
        width = d + len(features.M4_EXTRA_NAMES) + N_CELLS
        A = np.asarray(m4_inputs)
        if A.ndim != 2 or A.shape[1] != width:
            raise ValueError(f'm4_inputs must have shape (n, {width}) (features.m4_masked_inputs of a b1 row), not {A.shape}')
        head = self._h['M4Conv']
        out = []
        # 식 (B1-1)의 가린 M4 설계: 합성기가 만든 행렬에서 구역 열만 z + 1로 옮긴다. 이 행렬이 K와 O를 담는다(계획 P6).
        for lo in range(0, len(A), _PREDICT_CHUNK):
            chunk = A[lo:lo + _PREDICT_CHUNK]
            copy = narrow('m4_inputs', chunk if chunk.dtype == np.float32 else chunk.astype(np.float64), np.float32)     # a copy to shift
            out.append(head.predict(_m4_design(copy, d)))                                                   # K and O are read: P6
        return np.concatenate(out) if out else np.empty((0, N_CELLS))


# =============================================================================
# 블록 11: 헤드 번들 및 직렬화 엔진 (B1Heads, save_head, load_head)
# =============================================================================
def fit_head(fit_rows: HeadRows, stop_rows: HeadRows, name: str, kind: str, *, params=None, check=None):
    """Fit the head ``name`` of ``kind`` on the rows of ``fit_rows`` with the early stopping on the rows of ``stop_rows`` (both
    ``HeadRows`` of this module, with the same dropped columns). The design is built, fitted and dropped, so that the memory is that of
    this head: the float32 design is handed to the head as a ``_Design`` and freed as soon as LightGBM has binned it, no later than the
    return. ``check`` (the E6 guard) is called first, before this head reads a single array (a runner that fits head by head, as
    ``B1Heads.fit`` does, is protected without a call of its own), and then before every booster.

    (B1-2)–(B1-4)로 머리 하나를 ``fit_rows``의 행으로 맞추고 ``stop_rows``의 행으로 조기 종료한다((B1-5)). 두 묶음의 역할은 이 함수가 검사하지 않는다(역할의 겹침은 ``B1Heads.fit``이 거부한다).
    골드 머리는 팀 표시를 함께 넘긴다(판정 R19)."""
    if check is not None:
        check()
    if tuple(fit_rows.dropped) != tuple(stop_rows.dropped):
        raise ValueError('the fit rows and the stop rows must be made with the same dropped columns')
    if fit_rows.kill_law != stop_rows.kill_law:
        raise ValueError(f'the fit rows have the kill law {fit_rows.kill_law!r} and the stop rows {stop_rows.kill_law!r}: the early stopping would score another law')
    if fit_rows.k_condition != stop_rows.k_condition:
        raise ValueError(f'the fit rows have the k_condition {fit_rows.k_condition!r} and the stop rows {stop_rows.k_condition!r}: the early stopping would score the K heads on '
                         'another input')
    if fit_rows.k_reps_sha256() != stop_rows.k_reps_sha256():                        # None on both sides under the raw gold
        raise ValueError('the fit rows and the stop rows hold different representatives of the gold bins (k_reps_sha256 differs): the early stopping would score the K heads on '
                         'another K')
    d = fit_rows.data(name, kind)
    v = stop_rows.data(name, kind)
    shifted_kills = name == 'M3Kills' and fit_rows.kill_law != 'poisson'
    head = M3Kills(kind, params, fit_rows.kill_law) if shifted_kills else HEAD_CLASSES[name](kind, params)
    design = _Design(d.X)
    d = d._replace(X=None)                                                  # the head owns the design from here on
    if name == 'M3Gold':                                                    # the team of every entry (ruling R19)
        head.fit(design, d.y, d.team, layout=d.layout, valid=(v.X, v.y, v.team), check=check)
    elif d.e is None:
        head.fit(design, d.y, layout=d.layout, valid=(v.X, v.y), check=check)
    elif shifted_kills:                                                     # (B1-2k): the labels are the excess of the first team, in the fit rows and in the STOP rows
        head.fit(design, d.y, d.e, layout=d.layout, valid=(v.X, v.y, v.e), check=check, first_team=d.first_team, valid_first_team=v.first_team)
    else:
        head.fit(design, d.y, d.e, layout=d.layout, valid=(v.X, v.y, v.e), check=check)
        # 공통 가정 (나)에서 뺀 행의 수를 머리의 기록에 합친다.
        head.n_zero_exposure_ += d.n_zero_exposure
    return head


def save_head(head, kind: str, name: str, columns, directory, *, record=None) -> Path:
    """Write one fitted head into the folder ``directory`` (created): one text file for each booster (``<part>.txt``: the
    ``model_to_string`` text of LightGBM, UTF-8), then ``head.json``, LAST, so that a folder without it is a save that was cut. The JSON
    holds the schema, the LightGBM version, the ``kind`` and ``name``, the layout (the column names of the design for the b1 ``columns``)
    and the counts of the head, the file name and SHA-256 of every text and ``record``: any JSON value that the caller keeps with the
    head (the runner: the code state and the hash of the inputs of the fit stage that the head was fitted in, and its time), None for a
    head of a saved set. The same head gives the same bytes. FileExistsError if the folder already holds a ``head.json``: a saved head is
    never written over. Returns the folder.

    적합한 머리를 실행기가 머리마다 바로 남기는 단위이다(G-S3b B3, 판정 R40)."""
    out = Path(directory)
    if (out / HEAD_JSON).exists():
        raise FileExistsError(f'{out} already holds a saved B1 head: it is not written over')
    layout = make_layout(name, kind, columns, getattr(head, 'kill_law', 'poisson'))             # only the kill head has a kill law
    texts, meta = head.state()
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for part, text in texts.items():
        blob = text.encode('utf-8')
        store_core.write_atomic(out / f'{part}.txt', blob)
        files[part] = {'file': f'{part}.txt', 'sha256': hashlib.sha256(blob).hexdigest()}
    meta['columns'] = list(layout.names)
    meta['files'] = files
    doc = {'schema': HEAD_SCHEMA, 'lightgbm': lgb.__version__, 'kind': kind, 'name': name, 'record': record,
           'head': {**meta, **head.summary()}}
    store_core.write_atomic(out / HEAD_JSON, json.dumps(doc, indent=2, sort_keys=True, allow_nan=False) + '\n')
    return out


def _read_head_doc(folder: Path) -> dict:
    doc = json.loads((folder / HEAD_JSON).read_text(encoding='utf-8'))
    if doc.get('schema') != HEAD_SCHEMA:
        raise ValueError(f'{folder / HEAD_JSON}: the schema is {doc.get("schema")!r}, not {HEAD_SCHEMA!r}')
    return doc


def read_head_record(directory):
    """The ``record`` that the head saved in ``directory`` was saved with (None when it had none); only its JSON is read. FileNotFoundError
    for a folder without ``head.json`` and ValueError for another schema.

    실행기가 이어하기 때 저장된 머리의 코드와 입력을 대조하는 기록만 읽는다(G-S3b B3)."""
    return _read_head_doc(Path(directory)).get('record')


def load_head(directory, kind: str, name: str, columns):
    """Read the head ``name`` of ``kind`` saved in ``directory`` by ``save_head``, for the b1 ``columns``. ValueError for another schema,
    for a head that is not the one asked for, for a design that is not the design of this module for ``columns`` and for a text whose
    SHA-256 differs from the one in the JSON (the bytes that are hashed are the bytes that are read); FileNotFoundError for a missing
    file. The head gives the same predictions and the same bytes when it is saved again as the head that was saved.

    스키마, 모형 파일의 SHA-256, 설계 배치가 저장 때와 같을 때만 머리를 돌려준다(G-S3b B3)."""
    folder = Path(directory)
    doc = _read_head_doc(folder)
    meta = doc['head']
    if (doc.get('kind'), doc.get('name')) != (kind, name) or meta.get('kind') != kind or meta.get('class') != name:
        raise ValueError(f'{folder / HEAD_JSON}: the saved head is the {doc.get("kind")} head {doc.get("name")!r}, '
                         f'not the {kind} head {name!r}')
    layout = make_layout(name, kind, columns, meta.get('kill_law', 'poisson'))                  # the kill head says its law; absent means the plain law
    if meta['columns'] != list(layout.names):
        raise ValueError(f'{kind} head {name!r}: the saved design is not the design of this module')
    texts = {}
    for part, entry in meta['files'].items():
        path = folder / entry['file']
        blob = path.read_bytes()
        if hashlib.sha256(blob).hexdigest() != entry['sha256']:
            raise ValueError(f'{path}: the model file has changed (its sha256 differs from the one in the JSON)')
        texts[part] = blob.decode('utf-8')
    head = HEAD_CLASSES[name].from_state(texts, meta, layout)
    for single in (head.members if name == 'M3Gold' else (head,)):
        if single.n_features_ != layout.width or single.categorical_ != layout.categorical:
            raise ValueError(f'{kind} head {name!r}: the saved design is not the design of this module')
    return head


class B1Heads:
    """The fitted B1 heads: ``masked`` (the eight heads of the composer) and ``open`` (the five M3 and M4 heads), the columns that were
    dropped from b1 and the roles that were fitted on and stopped on. Use ``B1Heads.fit`` to fit, ``load`` to read, ``save`` to write,
    ``composer_heads('masked')`` for the composer, ``nll`` to score a role, ``predict_all`` to evaluate the masked heads on a batch.

    가린 머리 8개와 열린 머리 5개를 한 묶음으로 맞추고, 쓰고, 읽는다. 식은 (B1-1)–(B1-5)이다."""

    def __init__(self, masked: Mapping, open_: Mapping, dropped, fit_roles=('FIT',), stop_roles=('STOP',), kill_law: str = 'poisson', k_condition: str = 'raw',
                 k_reps_sha256=None):
        unknown = (set(masked) - set(HEAD_NAMES)) | (set(open_) - set(OPEN_HEAD_NAMES))
        if unknown:
            raise ValueError(f'unknown head(s): {", ".join(sorted(unknown))}')
        self.kill_law = killlaw.check_law(kill_law)
        # the gold of the K block that the object and conversion heads (masked and open) were fitted on (decision D21 B16); the set says it, the heads carry no declaration of it
        self.k_condition = head_rows.check_k_condition(k_condition)
        # the sha256 of the 31 representatives of the binned K block (``value.gold_reps_sha256``; review m4): None for the raw gold, and for a binned set whose representatives are not
        # recorded (``run_inference`` refuses such a set). A lowercase 64-digit hex string or None.
        if k_reps_sha256 is not None and (self.k_condition != 'binned' or not isinstance(k_reps_sha256, str) or len(k_reps_sha256) != 64
                                          or any(c not in '0123456789abcdef' for c in k_reps_sha256)):
            raise ValueError("k_reps_sha256 must be None or a lowercase 64-digit hex sha256, and only for k_condition 'binned'")
        self.k_reps_sha256 = k_reps_sha256
        for group in (masked, open_):
            if 'M3Kills' in group and getattr(group['M3Kills'], 'kill_law', 'poisson') != self.kill_law:
                raise ValueError(f'the kill head is fitted for kill_law {getattr(group["M3Kills"], "kill_law", "poisson")!r} but the set says {self.kill_law!r}')
        # always in the order of HEAD_NAMES and OPEN_HEAD_NAMES: summary(), nll() and the files do not depend on how the set was made
        self.masked = {name: masked[name] for name in HEAD_NAMES if name in masked}
        self.open = {name: open_[name] for name in OPEN_HEAD_NAMES if name in open_}
        self.dropped = tuple(dropped)
        self.columns = features.b1_columns(self.dropped)
        self.fit_roles = _as_tuple(fit_roles)
        self.stop_roles = _as_tuple(stop_roles)

    # -- fit -------------------------------------------------------------------------------------------------------------------------------
    @classmethod
    def fit(cls, s1, s2, labels, dropped, roles=None, *, check=None, params=None, kill_law: str = 'poisson', k_condition: str = 'raw') -> 'B1Heads':
        """Fit the eight masked and the five open heads on the matches of the role ``roles['fit']`` (default FIT), one head and one
        cell at a time, with the early stopping on the matches of ``roles['stop']`` (default STOP). ``dropped``: the columns left out of
        b1, ``features.dropped_columns(s2, np.flatnonzero(features.split_views(s1, s2, 'FIT')))`` (ruling R14). ``check``: a function
        without arguments (the E6 guard), called before anything is read, before each head and before each booster. ``params``: LightGBM
        parameters that replace the fixed ones (for the tests; the runner passes none). ValueError for a holdout role or a role that S3
        does not use, before any array is read.

        기본값은 FIT 역할의 경기로 맞추고 STOP 역할의 경기로 조기 종료 (B1-5)를 정하는 것이다(판정 R21의 roles 서명). 두 역할이 겹치거나 홀드아웃 역할이거나 S3가 쓰지 않는 역할이면
        배열을 읽기 전에 거부한다(계획 전역 제약의 분할).

        ``kill_law`` (S4b 준비 M1, 결정 D5): 킬 머리 M3Kills의 킬 수 법칙. 기본값 ``'poisson'``은 이전 적합과 같다(행, 설계, 부스터, 저장 파일). ``'shifted'``는 첫 킬 팀의 킬 수를 1 + N_d로 둔
        적합이다: 두 M3Kills 머리가 초과분을 라벨로 학습하고(조기 종료의 STOP 행도 같은 라벨) 열린 머리의 설계에 범주 열 d가 더해진다(``make_layout``). 다른 머리는 그대로이다.

        ``k_condition`` (S4b 준비 M3, 결정 D21 B16): K 블록을 읽는 네 머리(가린·열린 오브젝트 머리와 전환 머리)가 학습에서 보는 골드. 기본값 ``'raw'``는 이전 적합과 같다(행, 설계, 부스터, 저장 파일).
        ``'binned'``는 라벨의 실제 연속 골드 대신 관측한 골드 구간의 대표값(``head_rows``의 (HR-9); 킬이 없는 팀은 0)을 넣는 적합으로, 합성기가 이 머리들에 넣는 값과 같은 지지에서 학습한다.
        FIT 행과 조기 종료의 STOP 행이 같은 조건으로 만들어지고, 다른 머리는 그대로이다. 이름은 ``head_rows.K_CONDITIONS``에 있어야 한다(배열을 읽기 전에 거부). 'binned'이면 묶음이 대표값 31개의
        ``k_reps_sha256``(float64 little-endian 바이트의 sha256, ``value.gold_reps_sha256``)를 기록하고 ``save``가 JSON에 적는다; 평가(``run_inference``)가 자기 대표값과 대조한다."""
        fit_role, stop_role = _fit_stop_roles(roles)
        killlaw.check_law(kill_law)
        head_rows.check_k_condition(k_condition)
        tick = check if check is not None else (lambda: None)
        tick()
        law = {} if kill_law == 'poisson' else {'kill_law': kill_law}
        condition = {} if k_condition == 'raw' else {'k_condition': k_condition}      # the raw gold is not passed: the calls are those of every earlier run
        fit_rows = HeadRows(s1, s2, labels, dropped, fit_role, **law, **condition)
        stop_rows = HeadRows(s1, s2, labels, dropped, stop_role, **law, **condition)
        fitted = {kind: {} for kind in KINDS}
        for kind, names in (('masked', HEAD_NAMES), ('open', OPEN_HEAD_NAMES)):
            for name in names:
                fitted[kind][name] = fit_head(fit_rows, stop_rows, name, kind, params=params, check=check)       # it calls check first
        recorded = {} if k_condition == 'raw' else {'k_reps_sha256': fit_rows.k_reps_sha256()}       # the 31 numbers of the binned K block (fit and STOP rows agree: ``fit_head`` checked)
        return cls(fitted['masked'], fitted['open'], dropped, _as_tuple(fit_role), _as_tuple(stop_role), **law, **condition, **recorded)

    # -- scoring ---------------------------------------------------------------------------------------------------------------------------
    def nll(self, s1, s2, labels, role, kind: str = 'masked', *, check=None) -> dict:
        """The row-wise NLL of every head of ``kind`` on the matches of ``role`` (DEV, say): {head name: HeadScores}. The rows without a
        zone (M2Zone) and the rows without exposure (cells beyond the game end, units without a conversion window) are not rows of the
        head's mean; they are counted in ``n_unscored`` (a cell without exposure adds 0 to the NLL).

        머리별 설계 행 NLL (NLL-1)–(NLL-3)을 ``HeadScores``에 담아 평가 행의 점수 (HR-5)와 평균 (NLL-4)로 이어 준다."""
        if kind not in KINDS:
            raise ValueError(f'kind must be one of {KINDS}, not {kind!r}')
        tick = check if check is not None else (lambda: None)
        tick()
        rows = HeadRows(s1, s2, labels, self.dropped, role, **({} if self.kill_law == 'poisson' else {'kill_law': self.kill_law}),
                        **({} if self.k_condition == 'raw' else {'k_condition': self.k_condition}))          # the K heads are scored on the K they were fitted on
        group = self.masked if kind == 'masked' else self.open
        out = {}
        for name, head in group.items():
            tick()
            d = rows.data(name, kind)
            # 구역 −1인 행은 평균에서 빼고 센다((NLL-2)가 읽지 않는다, 판정 R22). 노출이 없는 칸은 (NLL-1)이 0을 더하므로 센 수만 보고한다(판정 R22).
            if name == 'M2Zone':
                scored = d.y >= 0
                out[name] = HeadScores(head.nll(d.X[scored], d.y[scored]), d.match[scored], d.row[scored], int((~scored).sum()))
            elif name == 'M3Gold':
                out[name] = HeadScores(head.nll(d.X, d.y, d.team), d.match, d.row, 0)
            elif name == 'M3Kills' and self.kill_law != 'poisson':
                out[name] = HeadScores(head.nll(d.X, d.y, d.e, first_team=d.first_team), d.match, d.row, d.n_zero_exposure)       # (KL-3): the shifted score
            elif d.e is None:
                out[name] = HeadScores(head.nll(d.X, d.y), d.match, d.row, 0)
            else:
                out[name] = HeadScores(head.nll(d.X, d.y, d.e), d.match, d.row, d.n_zero_exposure)
        return out

    # -- the composer ----------------------------------------------------------------------------------------------------------------------
    def composer_heads(self, mode: str = 'masked') -> 'heads.ComposerHeads':
        """The masked heads as a ``heads.ComposerHeads`` (call ``bind`` before asking for rows). The open heads are scored, not composed.

        합성은 가린 머리만 쓰고 교전의 사건 보기를 읽는 열린 머리는 쓰지 않으므로 열린 머리는 거부한다(설계서 3.4절, 8절의 평가 층)."""
        if mode == 'open':
            raise ValueError("composer_heads('open'): the open B1 heads are scored (B1Heads.nll), not composed")
        if mode != 'masked':
            raise ValueError(f"composer_heads: mode must be 'masked', not {mode!r}")
        return _MaskedChain(self.masked, self.dropped)

    def predict_all(self, batch: HeadBatch) -> np.ndarray:
        """(n, 184) every masked head on the rows of ``batch``, side by side in the order of ``PREDICT_ALL_WIDTHS``: the rate of the 18
        bins (per second), the laws of the mark, the zone and the size class, the mean kills of the two teams, the 62 gold probabilities
        (blue 31, then red 31, for the kill layer min(max(kills, 1), 6) of each team), the expected objectives and the conversion rates
        (per second). ``batch.row_view`` holds the b1 rows. TypeError for something that is not a ``HeadBatch``, ValueError for rows out
        of range.

        (HEAD-1)–(HEAD-4)의 머리 전부를 한 배치에서 계산한다. 골드 머리에 주는 킬 수 층은 팀의 킬 수 k에서 ξ = min(max(⌊k⌉, 1), 6)이다(⌊k⌉은 가장 가까운 정수, 동점은 짝수 쪽;
        층은 1–6, 계획 P4). 킬이 0인 팀에도 층 1의 분포를 낸다."""
        if not isinstance(batch, HeadBatch):
            raise TypeError(f'predict_all takes a HeadBatch, not {type(batch).__name__}')
        chain = self.composer_heads('masked')
        chain._set_rows(batch.row_view)
        n = len(batch.q)
        rows = np.arange(n)
        K = np.asarray(batch.K, dtype=np.float64)
        if K.shape != (n, N_K) or not np.isfinite(K).all() or (K < 0).any():
            raise ValueError(f'K must be a finite, non-negative array of shape ({n}, {N_K})')
        # 식 (HR-3)의 층 ξ를 K의 킬 수 열에서 만들어 골드 머리의 조건으로 준다.
        layer = np.clip(np.rint(K[:, :2]), 1, N_LAYER).astype(np.int64)
        m4 = features.m4_masked_inputs(chain.row_view(rows), batch.q, batch.m, batch.z, K, batch.O)
        pieces = [chain.rate(rows), chain.mark_prob(rows, batch.q), chain.zone_prob(rows, batch.q, batch.m),
                  chain.size_prob(rows, batch.q, batch.m, batch.z), chain.kill_rate(rows, batch.q, batch.m, batch.z, batch.s),
                  chain.gold_prob(rows, batch.q, batch.m, batch.z, batch.s, layer).reshape(n, -1),
                  chain.obj_rate(rows, batch.q, batch.m, batch.z, batch.s, K), chain.conv_rate(m4)]
        return np.hstack(pieces)

    # -- reports and files -----------------------------------------------------------------------------------------------------------------
    def summary(self) -> dict:
        """{'masked': {head: counts}, 'open': {head: counts}} as plain JSON values (rows, left-out rows, cells without events, the rounds
        kept and the trees of every model)."""
        return {kind: {name: head.summary() for name, head in getattr(self, kind).items()} for kind in KINDS}

    def save(self, directory) -> Path:
        """Write the folder of every head (``save_head``) into ``models/<kind>.<name>/`` of ``directory`` (created) and then
        ``b1_heads.json``, the last file. FileExistsError if the JSON or the folder ``models`` is there: a saved set is never written
        over.

        적합한 머리를 합성과 보고가 다시 읽도록 머리마다 모형 텍스트의 폴더로 남기고, 그 해시를 담은 설계 설명(JSON)을 마지막에 쓴다."""
        out = Path(directory)
        if (out / JSON_NAME).exists() or (out / MODEL_DIR).exists():
            raise FileExistsError(f'{out} already holds saved B1 heads: they are not written over')
        (out / MODEL_DIR).mkdir(parents=True, exist_ok=True)
        described = {kind: {} for kind in KINDS}
        for kind in KINDS:
            for name, head in getattr(self, kind).items():
                where = f'{MODEL_DIR}/{kind}.{name}'
                folder = save_head(head, kind, name, self.columns, out / where)
                entry = dict(_read_head_doc(folder)['head'])
                entry['files'] = {part: {**f, 'file': f'{where}/{f["file"]}'} for part, f in entry['files'].items()}      # from this JSON
                entry['dir'], entry['head_sha256'] = where, store_core.sha(folder / HEAD_JSON)
                described[kind][name] = entry
        doc = {'schema': SCHEMA, 'lightgbm': lgb.__version__, 'fit_roles': list(self.fit_roles), 'stop_roles': list(self.stop_roles),
               'dropped': list(self.dropped), 'b1_columns': list(self.columns), 'fixed_params': FIXED_PARAMS, 'max_rounds': MAX_ROUNDS,
               'early_stop': EARLY_STOP, 'heads': described}
        if self.kill_law != 'poisson':
            doc['kill_law'] = self.kill_law                         # absent means the plain law: a plain set is saved as every earlier set was
        if self.k_condition != 'raw':
            doc['k_condition'] = self.k_condition                   # absent means the raw gold: a raw set is saved as every earlier set was
            if self.k_reps_sha256 is not None:
                doc['k_reps_sha256'] = self.k_reps_sha256           # the 31 representatives of the binned K block (sha256 of the float64 little-endian bytes)
        store_core.write_atomic(out / JSON_NAME, json.dumps(doc, indent=2, sort_keys=True, allow_nan=False) + '\n')
        return out

    @classmethod
    def load(cls, directory) -> 'B1Heads':
        """Read heads saved by ``save`` (every head with ``load_head``). ValueError for another schema, for a head file or a model file
        whose SHA-256 differs from the JSON's, and for heads whose design is not the design of this module for the dropped columns;
        FileNotFoundError for a missing file.

        스키마, 파일의 SHA-256, 설계 배치가 저장 때와 같을 때만 머리를 돌려준다. 이전 스키마(v1)의 저장본은 머리마다 폴더가 없어 거부한다(판정 R40)."""
        folder = Path(directory)
        doc = json.loads((folder / JSON_NAME).read_text(encoding='utf-8'))
        if doc.get('schema') != SCHEMA:
            raise ValueError(f'{folder / JSON_NAME}: the schema is {doc.get("schema")!r}, not {SCHEMA!r}')
        dropped = tuple(doc['dropped'])
        columns = features.b1_columns(dropped)
        if list(columns) != doc['b1_columns']:
            raise ValueError('the b1 columns of the saved set are not those of this module for its dropped columns')
        wanted = {'masked': HEAD_NAMES, 'open': OPEN_HEAD_NAMES}
        if any(set(doc['heads'].get(kind, {})) != set(names) for kind, names in wanted.items()):
            raise ValueError('the saved set does not hold the eight masked and the five open heads')
        kill_law = killlaw.check_law(doc.get('kill_law', 'poisson'))
        k_condition = head_rows.check_k_condition(doc.get('k_condition', 'raw'))
        groups = {}
        for kind, names in wanted.items():                                              # the order of the head names, not that of the JSON
            groups[kind] = {}
            for name in names:
                meta = doc['heads'][kind][name]
                cls_ = HEAD_CLASSES.get(meta['class'])
                if cls_ is None or cls_.name != name or meta['kind'] != kind:
                    raise ValueError(f'{kind} head {name!r}: the JSON names the class {meta["class"]!r} of kind {meta["kind"]!r}')
                layout = make_layout(name, kind, columns, kill_law)
                if meta['columns'] != list(layout.names):
                    raise ValueError(f'{kind} head {name!r}: the saved design is not the design of this module')
                where = f'{MODEL_DIR}/{kind}.{name}'
                if meta.get('dir') != where:
                    raise ValueError(f'{kind} head {name!r}: the JSON puts it in {meta.get("dir")!r}, not in {where!r}')
                if store_core.sha(folder / where / HEAD_JSON) != meta['head_sha256']:
                    raise ValueError(f'{folder / where / HEAD_JSON}: the head file has changed (its sha256 differs from the one in the '
                                     'JSON)')
                groups[kind][name] = load_head(folder / where, kind, name, columns)
        return cls(groups['masked'], groups['open'], dropped, doc['fit_roles'], doc['stop_roles'], **({} if kill_law == 'poisson' else {'kill_law': kill_law}),
                   **({} if k_condition == 'raw' else {'k_condition': k_condition, 'k_reps_sha256': doc.get('k_reps_sha256')}))
