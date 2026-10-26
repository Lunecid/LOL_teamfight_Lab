r"""The rows that the baseline heads are trained and scored on, the same for B0 and B1 (stage S3 Task 7, ruling R18).

The heads of the composer (M2Rate ... M4Conv, ``baselines_b0`` and ``baselines_b1``) are fitted on the same rows: which S2 view a row reads,
which target, which exposure, which categories. That mapping from the S1 tables and the S2 views to rows is the one place where the causal
rule A3 is kept (a masked head sees the t0 row and the categories of the onset, an open head the event view of its engagement), so it lives
here once. What differs between the models is the *feature builder* (B0: ``features.b0`` filled with the FIT means; B1: ``features.b1``
without the FIT drop list) and the *design* that is made of the rows (B0: standardised numeric columns and one-hot blocks; B1: the float32
columns and the categories as LightGBM categorical columns). ``HeadRows`` takes the builder; the design is made by the model.

``HeadRows(s1, s2, labels, role, build)``: the rows of the matches of one role (FIT, STOP or DEV) of an (S1, S2, labels) triple. The keyword ``kill_law``
(S4b preparation M1, decision D5; ``'poisson'`` by default, which gives the rows of every earlier run) names the kill law of the M3Kills head: under ``'shifted'``
the M3Kills rows carry the first-kill team d (``HeadParts.first_team``; the open rows also as the category ``codes['d']``, the masked rows have it in the mark ``m``)
and are refused when a unit's first-kill team has no kill (``units/K`` and ``units/mark`` must agree, equation (HR-8)). The keyword ``k_condition`` (S4b preparation M3,
decision D21 B16; ``'raw'`` by default, which gives the rows of every earlier run) names what the gold of the K block is: ``'raw'`` the true continuous kill gold of the
labels, ``'binned'`` the representative of the observed gold bin (``rep[labels/gold_bin]`` for a team with a kill, 0 for a team without one), exactly the K that the
composer feeds to the K-conditioned heads; the representatives are the in-scope FIT bin means of the label store (``value.fit_gold_bin_means``, ``HeadRows.k_reps``),
equation (HR-9).
``build(s2, views)`` returns the (n, d) float32 feature rows of the S2 views ``views`` (an int64 array of view numbers).
``parts(head, kind)`` gives the ``HeadParts`` of one head: the feature rows, the categories, the target, the exposure and the K and O
blocks. ``view_numbers(head, kind)`` gives, row by row in the order of ``parts``, the S2 view that each row reads (a model that reads the
S2 store itself, the EPT of stage S4a, needs the view numbers and not the feature rows). ``frame()`` gives the frame rows of the role and
their features (the rows of B2). ValueError for an S2 store or label store that
does not belong to the S1 store (counts only, never a key), for a role that is a holdout role or not used by S3, and for a unit that lacks
a view.

The rows. The masked rows are the onsets (a frame-clock row t0 and a unit with its first kill after t0); the feature row of an onset is that
of the frame-clock S2 view of the onset's row: ``views/src == 0`` and ``views/src_index == onsets/row`` (the event views have colliding
``src_index`` numbers). The M2Rate rows are the (frame row, bin) pairs with exposure. The open rows are the in-scope units; their feature
row is that of the S2 event view of the unit: ``views/src == 1`` (the view at the first kill f) for M3 and ``== 2`` (the view at delta) for
M4, ``views/src_index`` being a row of ``event_rows``. The views are found by ``views/src`` and ``views/src_index`` and nowhere else (never
by position) and are checked against S1 (duplicates, range, event kind, match number and reference time).

The inputs of a head (what a design may be made of; a model takes the ones its head conditions on, decision P6)

    feat   the feature rows; the row j of the head reads ``feat[take[j]]`` (``take`` None: ``feat[j]``), so that a design can be filled
           chunk by chunk without a second copy of the 8.2 M rows of M2Rate
    codes  the categories: q the occurrence bin, m the mark, z the zone (-1 = the first kill has no position), s the size class, layer
           the kill layer of the team of a gold entry (1 .. 6, six or more); a head has the ones it conditions on
    K      kills blue, kills red, kill gold blue and red in thousands (float64): the open M3 object and M4 heads, and the masked M3 object
           head when ``masked_obj_k`` is set (B1 conditions it on K; B0 does not, decision P6). The gold is the true gold of the labels
           (``k_condition='raw'``) or the representative of the observed gold bin (``'binned'``, equation (HR-9))
    O      the 18 objective counts of the engagement: the open M4 head
    m4     the masked M4 head only: the matrix that ``features.m4_masked_inputs`` builds from the t0 rows and the TRUE q, m, z, K and O of
           the onsets (the one way to build what the masked M4 head sees, leakage test (a)); ``feat`` is its first d columns and the
           categories q, m and z are read back from it. B0 keeps [feat, q, m, z] (P6); B1 uses the whole matrix

The targets: M2Rate the count of onsets in a (row, bin) cell; M2Mark, M2Zone and M3Size the mark, the zone and the size class of the unit of
the onset; M3Kills the kills of the blue and of the red team (a column for each, ruling R19); M3Gold the gold bin of a team entry (the team
entries with at least one kill, blue entries first, ``team`` says which); M3Obj and M4Conv the 18 objective and conversion counts. The
exposure ``e`` is in seconds: the part of the bin inside the window for M2Rate, the conversion window for M4Conv, 1 for the other Poisson
heads, None for the categorical heads. ``n_zero_exposure`` counts the cells or units that were left out for want of exposure.

The arrays that a call returns are fresh or read-only (the cached ones), so a fit that standardises a design in place cannot change what
the next call gives.

방법에서의 위치. 이 모듈은 기준 모형의 머리마다 학습과 채점에 쓰는 행을 S1 표와 S2 보기에서 골라 만든다. 어느 행이 어느 머리의 행이고 그 행의 목표, 노출, 범주가
무엇인지를 정하는 곳은 여기 한 곳이며, B0와 B1은 특징을 만드는 방법과 그 특징으로 만드는 설계만 다르다(판정 R18). 이 한 곳이 인과 규칙을 지킨다: 가린 머리의 행은
t₀ 행의 특징과 사슬의 앞 단계 값(발생의 범주 q, m, z, s, 골드 머리의 킬 수 층, 머리에 따라 실제 K와 O; 교사 강제)만, 열린 머리의 행은 그 교전의 사건 보기를
읽는다(설계서 3.3절·10절 A3, 계획 전역 제약의 인과(A3)). 이 모듈의 식은 새 통계량이 아니라 점수 (NLL-1)–(NLL-3)과 평균 (NLL-4), (NLL-5)에 무엇이 들어가는지를 정하는
행의 정의이다.

용어. 머리: 조건부 분포 하나를 예측하는 모형 부품이며 M2Rate, M2Mark, M2Zone, M3Size, M3Kills, M3Gold, M3Obj, M4Conv의 여덟 개이다. 분 프레임: 경기 상태를
1분마다 찍은 기록이고 t₀는 그 시각이다. 교전: 같은 경기에서 15초와 2,000 지도 단위 이내로 이어진 킬의 묶음이며, 범위 안 교전은
첫 킬 f가 120 s ≤ f ≤ T − 30 s인 교전이다(설계서 2절). 발생: t₀ 행과 그 뒤 구간 q(10초씩 18개)에서 첫 킬이 일어난 교전의 쌍이다(설계서 3.1절). 가린 행: 발생
하나(골드 머리는 그 발생의 팀 항목, 발생률 머리는 (t₀ 행, 구간) 쌍). 열린 행: 범위 안 교전 하나(골드 머리는 그 교전의 팀 항목). 표지 m: 첫 킬의 선제 킬 쌍(킬러
슬롯이 피해자 슬롯을 잡는 방향 쌍 50개)과 처형 2개. 킬 수 층: 킬이 하나 이상인 팀의 킬 수를 1–5와 6 이상으로 나눈 여섯 층. 오브젝트 칸: 오브젝트 종류와
가져간 팀의 쌍마다 하나(18칸). 전환 창: 교전이 끝난 뒤 60초. 노출: 그 칸을 관측한 시간(초).

식 목록. 식 번호는 (모듈 약어-번호) 꼴이며 식과 기호는 docs/superpowers/notes/equation-index.md의 식 색인과 기호 등록부에 올라 있다. 행 i는 한 머리의 설계
행이고, c는 그 머리의 출력 번호(포아송 머리의 칸, 분류 머리의 범주)이다.
    (HR-1)  M2Rate의 행 i = (t₀ 행, 구간 q) 쌍:   y_i = n_q,   e_i = Δ_q                  Δ_q > 0인 쌍만 행이다
    (HR-2)  개수 머리의 y_{ic}, e_i:   M3Kills y_{i,blue} = k_blue, y_{i,red} = k_red, e_i = 1;   M3Obj y_{ic} = O_c, e_i = 1;
            M4Conv y_{ic} = 전환 수_c, e_i = 전환 창의 길이(초; e_i > 0인 행만).   범주 머리의 c_i:  M2Mark m, M2Zone z, M3Size s
    (HR-3)  M3Gold의 행 i = 킬이 k ≥ 1인 팀 하나(파랑 먼저):   ξ_i = min(k, 6),   c_i = b_i = 그 팀의 골드 구간
    (HR-4)  K 블록 = (k_blue, k_red, g_blue / 1000, g_red / 1000)                           골드는 천 골드 단위
    (HR-5)  v_I = Σ_{i∈I} ℓ_i                                                              평가 행 I의 점수; (NLL-4)와 (NLL-5)의 입력
    (HR-8)  d_i = m_i // 25 (m_i ≤ 49),  d_i = −1 (m_i = 50, 51);   단언 k_{i,d_i} ≥ 1                  M3Kills 이동 법칙 행의 첫 킬 팀(``kill_law='shifted'``만)
    (HR-9)  g̃_{i,t} = rep_{b_{i,t}} (k_{i,t} ≥ 1),   g̃_{i,t} = 0 (k_{i,t} = 0),   K 블록 = (k_blue, k_red, g̃_blue / 1000, g̃_red / 1000)       K 조건 'binned'의 K 블록 ((HR-4)의 g 대신)

공통 가정.
    (가) 노출이 0인 칸은 (NLL-1)에 아무것도 더하지 않으므로 행으로 만들지 않고 그 수를 센다(계획 전역 제약의 확률 족, 판정 R22). 그 칸에 기록된
         개수는 자료 오류이다(검산 노트 03 C절).
    (나) 목표를 알 수 없는 행(구역이 관측되지 않은 첫 킬, z = −1)은 돌려주되 채점 평균에서 빼고 센다(판정 R17, R22).
    (다) 모든 행은 경기 번호와 평가 행 번호를 가져서, 평균과 복제가 경기를 하나의 독립 단위로 다룰 수 있다(nll 모듈 글 (나); 계획 P7, P9).
    (라) K 조건 'binned'(S4b 준비 M3, 결정 D21 B16)는 K 블록의 골드를 합성기가 넣는 값으로 맞춘다. 합성기는 팀의 킬 골드를 구간 b로만 뽑아 rep_b로 읽고 킬이 없는 팀은 0이므로,
         학습 행이 실제 연속 골드를 넣으면 K 조건 머리(오브젝트·전환)가 합성에서 한 번도 보지 않는 입력의 지지로 학습한다(검토 A3-F4). 대표값 rep_b는 FIT 전체의 범위 안
         단위에서 한 번 내며 행의 역할이나 고른 경기에 따라 바뀌지 않는다(결정 D12: 타깃의 정의는 FIT 전체의 고정 자산). 킬 수 열, 모든 목표, 골드 머리의 구간 라벨은 그대로이다.
"""
# Ref: docs/superpowers/plans/2026-10-21-c-ept-s3.md decisions P4 (the factorisation), P6 (what the masked heads are conditioned on), P9
#      (one row per training row), P10 (the exposures), Tasks 6 and 7; controller rulings R17 (the zone label), R18 (this module), R19 (a
#      column of y for each team in the kill heads) and R22 (left-out rows are counted).
# Ref: docs/superpowers/specs/2026-10-20-c-ept-design.md sections 3.2–3.3 (the masked and the open heads) and 9 (stage S3: B0 and B1)
#      and section 4 (frame-clock rows and the occurrence window).
# Ref: harness/decisions.md D21 (2026-10-26, B16: the K-conditioned heads are trained on the representatives of the observed gold bin, from the in-scope FIT units) and D12
#      (target definitions on the whole FIT), and the review finding A3-F4 (harness/reviews/EPT-T010-area-3.md), for ``k_condition``.
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np

from c_ept import config as C
from c_ept.preprocessing import features
from c_ept.models import killlaw, nll as nll_scores

# ---- the heads and their shapes --------------------------------------------------------------------------------------------------------
KINDS = ('masked', 'open')
HEAD_NAMES = ('M2Rate', 'M2Mark', 'M2Zone', 'M3Size', 'M3Kills', 'M3Gold', 'M3Obj', 'M4Conv')
OPEN_HEAD_NAMES = ('M3Size', 'M3Kills', 'M3Gold', 'M3Obj', 'M4Conv')
TEAMS = ('blue', 'red')          # team 0 and team 1: the two cells of M3Kills, the two members of M3Gold
MIX = 1e-6                       # ruling R13: weight of the uniform law in every categorical output
N_Q, N_M, N_SIZE, N_LAYER = C.N_BINS, C.N_MARKS, 4, 6          # the size classes are those of ``units.SIZE_NAMES`` (a test pins 4)
N_ZONE = features.N_ZONES        # the zones of ``terrain.ZONES``; the zone input has one more level, "no position" (-1)
N_GOLD = nll_scores.N_GOLD_BINS
N_CELLS = features.N_O           # the cells of O and of the conversions: 2 kind + taker
N_K = 4                          # kills blue, kills red, kill gold blue and red (in thousands): the K columns of the M4 input
# the width of the output of every composer method, in the order in which ``predict_all`` puts them side by side
PREDICT_ALL_WIDTHS = {'rate': N_Q, 'mark_prob': N_M, 'zone_prob': N_ZONE, 'size_prob': N_SIZE, 'kill_rate': 2, 'gold_prob': 2 * N_GOLD,
                      'obj_rate': N_CELLS, 'conv_rate': N_CELLS}
VALID_HEADS = frozenset([(name, 'masked') for name in HEAD_NAMES] + [(name, 'open') for name in OPEN_HEAD_NAMES])
K_CONDITIONS = ('raw', 'binned')  # the gold of the K block: the true kill gold of the labels (every earlier run) or the representative of the observed gold bin (decision D21 B16)


def check_k_condition(k_condition) -> str:
    """``k_condition`` itself if it is one of ``K_CONDITIONS``; ValueError naming ``k_condition`` otherwise (a short string is shown, any other value by its type only)."""
    if not isinstance(k_condition, str) or k_condition not in K_CONDITIONS:
        shown = repr(k_condition[:40]) if isinstance(k_condition, str) else type(k_condition).__name__
        raise ValueError(f'k_condition must be one of {K_CONDITIONS}, not {shown}')
    return k_condition


# ---- the training and scoring rows -------------------------------------------------------------------------------------------------------
class HeadData(NamedTuple):
    """The rows of one head as a model sees them: the design ``X`` (n, d), the target ``y`` (class numbers; counts (n,) or (n, 18)), the
    exposure ``e`` in seconds (None for a categorical head), the match number of every row and the id of the evaluation row it belongs to
    (a frame row for M2Rate, an onset for a masked head, a unit for an open head: rows with one id are scored together), the ``layout`` of
    the design (that of the model: ``baselines_b0.Layout`` or ``baselines_b1.Layout``) and ``n_zero_exposure``, the number of rows or
    cells that were left out for want of exposure (cells beyond the game end, units without a conversion window). ``team`` (0 blue, 1 red,
    one entry for each row) is given for the gold head only, whose rows are the team entries with at least one kill, blue entries first;
    M3Kills has one row per onset (or unit) and a column of ``y`` for each team. ``first_team`` (S4b preparation M1, decision D5) is given for the
    M3Kills head of the shifted kill law only: the team d that took the first kill of each row (0 blue, 1 red, −1 an execution).

    설계 ``X``는 모형이 만든다. ``y``와 ``e``는 점수 (NLL-1)–(NLL-3)의 개수·범주와 노출이고(행의 정의는 (HR-1)–(HR-4)), ``match``와 ``row``는 평균 (NLL-4)와
    복제 (NLL-5)가 쓰는 경기 번호와 평가 행 번호이다((HR-5)). ``first_team``은 이동 법칙의 첫 킬 팀 d((HR-8))이다."""
    X: np.ndarray
    y: np.ndarray
    e: np.ndarray | None
    match: np.ndarray
    row: np.ndarray
    layout: object
    n_zero_exposure: int = 0
    team: np.ndarray | None = None
    first_team: np.ndarray | None = None


class HeadParts(NamedTuple):
    """The rows of one head before a model makes a design of them (``HeadRows.parts``); see the module text for the inputs ``feat`` /
    ``take``, ``codes``, ``K``, ``O`` and ``m4``. ``y``, ``e``, ``match``, ``row``, ``n_zero_exposure`` and ``team`` are those of
    ``HeadData``.

    행의 정의 (HR-1)–(HR-4)의 개수 y_i·y_{ic}와 범주 머리의 목표 c_i는 ``y``에, 노출 e_i는 ``e``에, 조건 범주 q, m, z, s와 층 ξ_i는 ``codes``에, K 블록은 ``K``에 들어 있다.
    ``first_team``(S4b 준비 M1, 결정 D5)은 이동 법칙을 켠 M3Kills 행의 첫 킬 팀 d((HR-8))이다. 열린 M3Kills 행은 같은 값을 ``codes['d']``(머리의 입력)로도 갖고,
    가린 행은 표지 ``codes['m']``에서 d를 얻는다."""
    feat: np.ndarray
    take: np.ndarray | None
    codes: dict
    y: np.ndarray
    e: np.ndarray | None
    match: np.ndarray
    row: np.ndarray
    n_zero_exposure: int = 0
    team: np.ndarray | None = None
    K: np.ndarray | None = None
    O: np.ndarray | None = None
    m4: np.ndarray | None = None
    first_team: np.ndarray | None = None

    def gather(self) -> np.ndarray:
        """(n, d) the feature row of every row of the head (a copy if ``take`` is given, else ``feat`` itself)."""
        return self.feat if self.take is None else self.feat[self.take]


def _unselected_label_fill(dtype: np.dtype):
    """The value that stands in for the label rows of the units of matches that a ``matches=`` selection leaves out (never a valid label).

    A signed integer array gets -1, a floating array NaN, an unsigned integer array the largest value of its type. A boolean (or any
    other kind of) array has no value that a label cannot take, so TypeError: the subset path refuses it rather than fill it with False.
    The label arrays read here today are ``labels/kill_gold`` (float64) and ``labels/gold_bin`` (int8), so only the first two rules are
    used.

    표지값. 0은 유효한 골드 구간이고 유효한 킬 골드이므로 선택하지 않은 행에 0을 두면 행 선택의 실수가 그럴듯한 값으로 숨는다. 부호 있는 정수는 −1(골드 구간
    −1은 "킬이 없는 팀"의 표지이기도 하지만, 킬이 있는 팀의 항목에서 −1을 읽으면 ``_gold_rows``가 거부한다), 실수는 NaN(``_kraw``의 유한성 검사가 거부한다),
    부호 없는 정수는 그 자료형의 최댓값(구간 0–30이나 개수가 될 수 없는 값)이다. 참거짓 배열에는 라벨이 갖지 않는 값이 없어 표지를 둘 수 없으므로 거부한다.
    """
    kind = np.dtype(dtype).kind
    if kind == 'i':
        return -1
    if kind == 'f':
        return np.nan
    if kind == 'u':
        return np.iinfo(dtype).max
    raise TypeError(f'a label array of dtype {np.dtype(dtype)} has no value that marks a row as not selected; '
                    'the match-subset path takes signed integer, unsigned integer or floating label arrays')


def _freeze(obj):
    """Make every array in ``obj`` (an array, a tuple or list of them, or a namespace of them) read-only; returns ``obj``."""
    if isinstance(obj, np.ndarray):
        obj.setflags(write=False)
    elif isinstance(obj, (tuple, list)):
        for part in obj:
            _freeze(part)
    elif isinstance(obj, SimpleNamespace):
        for part in vars(obj).values():
            _freeze(part)
    return obj


class HeadRows:
    """The rows that the heads are trained or scored on, for the matches of one role of an (S1, S2, labels) triple; see the module text.

    ``build(s2, views)`` is the feature builder: the (n, d) float32 feature rows of the S2 views ``views`` (b0 with the FIT means for B0,
    b1 without the dropped columns for B1). It is the only thing that differs between the models.

    행의 정의는 식 (HR-1)–(HR-4)이고 ``parts``가 그 정의대로 행을 낸다. 역할은 FIT, STOP, DEV만 받으며 홀드아웃 역할(TEST, EXTERNAL)과 S3가 쓰지 않는
    역할은 첫 배열을 읽기 전에 거부한다(계획 전역 제약의 분할)."""

    def __init__(self, s1, s2, labels, role, build: Callable, *, matches=None, kill_law: str = 'poisson', k_condition: str = 'raw'):
        self.kill_law = killlaw.check_law(kill_law)             # before any array is read; 'poisson' gives the rows of every earlier run
        self.k_condition = check_k_condition(k_condition)       # likewise: 'raw' gives the rows of every earlier run
        if self.k_condition == 'binned' and labels is None:
            raise ValueError("k_condition 'binned' reads labels/gold_bin and the representatives of the gold bins: it needs the label store")
        self.s1, self.s2, self.labels, self.role = s1, s2, labels, role
        self.build = build
        self._frame_in = features.role_mask(s1, 'frame_rows', role)       # refuses a bad role before the first array is read
        self._onset_in = features.role_mask(s1, 'onsets', role)
        self._unit_in = features.role_mask(s1, 'units', role)
        # Select only after authoritative role validation; global row IDs remain intact.
        self.matches = None if matches is None else features.checked_matches(s1, role, matches)
        if self.matches is not None:
            frame_match = np.asarray(s1.array('frame_rows/match'))
            onset_row = np.asarray(s1.array('onsets/row')).astype(np.int64)
            self._frame_in &= np.isin(frame_match, self.matches)
            self._onset_in &= np.isin(frame_match[onset_row], self.matches)
            self._unit_in &= np.isin(np.asarray(s1.array('units/match')), self.matches)
        self._cache = {}

    # -- reading -------------------------------------------------------------------------------------------------------------------------
    def _read(self, store, name: str) -> np.ndarray:
        key = (id(store), name)
        if key not in self._cache:
            try:
                self._cache[key] = np.asarray(store.array(name))
            except FileNotFoundError:
                raise ValueError(f'the store has no array {name}') from None
        return self._cache[key]

    def _s1(self, name: str) -> np.ndarray:
        return self._read(self.s1, name)

    def _labels(self, name: str) -> np.ndarray:
        n_units = len(self._unit_in)
        if self.matches is None:
            a = self._read(self.labels, name)
        else:
            key = (id(self.labels), name)
            if key not in self._cache:
                source = self.labels.array(name)
                if source.shape != (n_units, 2):
                    raise ValueError(f'{name} must have shape ({n_units}, 2)')
                selected = np.flatnonzero(self._unit_in)
                # Preserve global unit IDs while reading selected label values only. 선택하지 않은 경기의 단위는 라벨을 읽지 않으므로 그 행을
                # 유효한 값(0은 유효한 골드 구간이자 유효한 킬 골드이다)이 아니라 표지값으로 채운다: 정수 −1, 실수 NaN. 행 선택에 실수가 있어
                # 그 행을 읽으면 그럴듯한 0으로 학습하거나 채점하지 않고, 킬 골드는 유한성 검사(_kraw)에서, 골드 구간은 −1 거부
                # (_gold_rows)에서 바로 거부된다(EPT-T010 A1-F7). 선택한 행의 값은 전체 경로와 같다.
                # K 조건 'binned'의 대표값(``_k_gold_reps``)은 이 경로가 아니라 라벨 저장소의 전체 FIT 배열을 직접 읽는다(결정 D12: 타깃의 정의는 FIT 전체의 고정 자산이므로 고른 경기가 달라도 같은 31개 수).
                a = np.full(source.shape, _unselected_label_fill(np.dtype(source.dtype)), dtype=source.dtype)
                a[selected] = np.asarray(source[selected])
                self._cache[key] = a
            a = self._cache[key]
        if a.shape != (n_units, 2):
            raise ValueError(f'{name} must have shape ({n_units}, 2), one row for each unit of the S1 store, not {a.shape}: '
                             'the labels do not belong to this S1 store')
        return a

    def _once(self, key: str, make):
        """The value of ``make()``, computed once; its arrays are made read-only, so that a caller who changes the arrays of a ``HeadData``
        (a fit with ``overwrite_X`` does) cannot change what the next call returns."""
        if key not in self._cache:
            self._cache[key] = _freeze(make())
        return self._cache[key]

    # -- the S2 views ----------------------------------------------------------------------------------------------------------------------
    def _view_maps(self) -> tuple:
        """(view of every frame row, view of every event row), -1 where the S2 store has none; the match number and the reference time of
        the views of the role are checked against S1."""
        def make():
            src, index = self._read(self.s2, 'views/src'), self._read(self.s2, 'views/src_index').astype(np.int64)
            n_rows, n_events = len(self._s1('frame_rows/t0')), len(self._s1('event_rows/kind'))
            frame_view, event_view = np.full(n_rows, -1, dtype=np.int64), np.full(n_events, -1, dtype=np.int64)
            for target, selected, limit, what in ((frame_view, src == 0, n_rows, 'frame-clock'),
                                                  (event_view, (src == 1) | (src == 2), n_events, 'event')):
                at = np.flatnonzero(selected)
                found = index[at]
                if found.size and (found.min() < 0 or found.max() >= limit):
                    raise ValueError(f'views/src_index names a {what} row that the S1 store does not have')
                if np.bincount(found, minlength=limit).max(initial=0) > 1:
                    raise ValueError(f'two {what} views of the S2 store have the same src_index')
                target[found] = at
            kind = self._s1('event_rows/kind')
            mapped = event_view >= 0
            if (src[event_view[mapped]] != kind[mapped]).any():
                raise ValueError('an event view of the S2 store has another kind (src) than its event row')
            return frame_view, event_view
        return self._once('view_maps', make)

    def _frame_features(self) -> tuple:
        """(positions of the frame rows of the role in a (n, d) float32 feature matrix, that matrix)."""
        def make():
            frame_view, _ = self._view_maps()
            rows = np.flatnonzero(self._frame_in)
            views = frame_view[rows]
            if (views < 0).any():
                raise ValueError(f'{int((views < 0).sum())} frame row(s) of the role have no frame-clock view in the S2 store')
            self._check_views(views, self._s1('frame_rows/match')[rows], self._s1('frame_rows/t0')[rows], 'frame-clock')
            pos = np.full(len(self._frame_in), -1, dtype=np.int64)
            pos[rows] = np.arange(len(rows))
            return pos, self.build(self.s2, views)
        return self._once('frame_features', make)

    def _check_views(self, views, match, ref, what) -> None:
        if (self._read(self.s2, 'views/match')[views] != match).any() or (self._read(self.s2, 'views/ref')[views] != ref).any():
            raise ValueError(f'the {what} views of the S2 store have other match numbers or reference times than the S1 rows: '
                             'the S2 store does not belong to this S1 store')

    def frame(self) -> SimpleNamespace:
        """The frame rows of the role: ``rows`` (the indices into ``frame_rows``), ``feat`` (their (n, d) float32 features, in that order)
        and ``match`` (their match numbers). These are the rows of B2. ValueError for a role without a frame row.

        발생이 없는 t₀ 행도 포함하여 역할의 모든 t₀ 분 프레임 행을 낸다. B2는 프레임 행만 쓴다(계획 Task 7)."""
        rows = np.flatnonzero(self._frame_in)
        if rows.size == 0:
            raise ValueError('the role has no frame row')
        _, feat = self._frame_features()
        return SimpleNamespace(rows=rows, feat=feat, match=self._s1('frame_rows/match')[rows].astype(np.int64))

    # -- the onsets of the role (masked rows) ----------------------------------------------------------------------------------------------
    def _onsets(self):
        def make():
            idx = np.flatnonzero(self._onset_in)
            if idx.size == 0:
                raise ValueError('the role has no onset')
            row, unit = self._s1('onsets/row')[idx].astype(np.int64), self._s1('onsets/unit')[idx].astype(np.int64)
            pos, _ = self._frame_features()
            mark = self._s1('units/mark')[unit].astype(np.int64)
            if not np.array_equal(mark, self._s1('onsets/mark')[idx]):
                raise ValueError('onsets/mark differs from the mark of the unit of the onset')
            return SimpleNamespace(idx=idx, row=row, unit=unit, bin=self._s1('onsets/bin')[idx].astype(np.int64), mark=mark,
                                   take=pos[row], zone=self._zones()[unit].astype(np.int64),
                                   size=self._s1('units/size')[unit].astype(np.int64),
                                   match=self._s1('frame_rows/match')[row].astype(np.int64))
        return self._once('onsets', make)

    def _zones(self) -> np.ndarray:
        return self._once('zones', lambda: features.unit_zone(self.s1) if self.matches is None
                          else features.unit_zone_subset(self.s1, np.flatnonzero(self._unit_in)))

    def _kcounts(self, unit) -> np.ndarray:
        return self._s1('units/K')[unit][:, :2].astype(np.int64)

    def _first_teams(self, head: str, mark, unit, what: str):
        """The first team d of every row of the M3Kills head under the shifted kill law, after the check that it has a kill; None for every other head and for the plain law.

        수식
        ----
        (HR-8)  d_i = m_i // 25  (m_i ≤ 49),   d_i = −1  (m_i = 50, 51) ;   단언: k_{i,d_i} ≥ 1  (d_i ≥ 0)

          기호  | 뜻                                                | 단위 | 범위       | 코드 변수
          m_i   | 행 i의 단위가 첫 킬에서 얻은 표지                   | —    | 0 … 51    | mark
          d_i   | 첫 킬 팀 (0 파랑, 1 빨강, −1 처형)                  | —    | −1, 0, 1  | 반환값
          k_{i,t} | 행 i의 단위에서 팀 t가 낸 킬 수 (``units/K``)      | 개   | 0, 1, …   | _kcounts

        해석: 이동 법칙은 첫 킬 팀의 킬 수를 1 + N_d로 두므로 그 팀의 킬 수가 0인 단위는 확률 0이다. 표지의 d는 킬러 칸(참가자 1–5가 팀 0)에서, ``units/K``는 팀 배열
        (``side[killer − 1]``)에서 나오므로 두 출처가 어긋나면 그 단위가 −1 칸을 만든다(검토 초점 2). 어긋난 단위의 수만 말하고 경기 번호는 말하지 않는다.
        """
        if head != 'M3Kills' or self.kill_law != 'shifted':
            return None
        first = killlaw.first_team(mark)
        champion = np.flatnonzero(first >= 0)
        bad = int((self._kcounts(unit)[champion, first[champion]] < 1).sum())
        if bad:
            raise ValueError(f'{bad} {what}(s) of the role have a first kill by a team that scored no kill (the mark and units/K disagree): the shifted kill law gives them '
                             'the probability 0')
        return first

    def k_reps(self):
        """(31,) float64, read only: the representatives rep_b of the gold bins that the K block of the rows is made of under ``k_condition='binned'`` (the in-scope FIT bin means of
        the label store, the whole FIT whatever the role of the rows and the matches they are restricted to); None under ``'raw'``, which uses none. These are the 31 numbers
        that the composer must hold (``Composer(..., gold_reps=rows.k_reps())``, ``run_inference(..., gold_reps='in_scope_fit')``) so that the heads see one support in
        training and in composition.

        (HR-9)의 rep_b이다. 값은 ``value.fit_gold_bin_means(population='in_scope')`` 한 곳에서 오고 행마다 다시 계산하지 않는다. 범위 안 FIT 단위가 하나도 없거나 그 단위에 킬이 하나도
        없으면 그 함수가 거부한다(31개가 모두 가운데 값이 되는 저장소로 학습하지 않는다)."""
        return self._k_reps() if self.k_condition == 'binned' else None

    def k_reps_fallback(self):
        """The bins of the representatives that had no in-scope FIT entry and hold the middle of their range (100 b + 50; 3,000 for the open bin), as an ascending tuple of bin
        numbers, like the ``gold_bin_fallback`` that the S3 labels named; None under ``'raw'``. A ledger that records the representatives (``k_reps_sha256``) records these as well.

        (VAL-7)의 fallback이다(판정 R12: 항목이 없는 구간은 가운데를 쓰고 그 번호를 알린다)."""
        return self._k_gold_reps().fallback if self.k_condition == 'binned' else None

    def k_reps_sha256(self):
        """The sha256 (hex) of the 31 representatives as float64 little-endian bytes in C order (``value.gold_reps_sha256``); None under ``'raw'``. The identity of the 31 numbers that
        the K block of these rows was made of: the training identity, the B1 set and the evaluation record it.

        (HR-9)의 rep_b 31개의 신원이다."""
        if self.k_condition != 'binned':
            return None
        from c_ept.valuation import value                      # not at module level, as in ``_k_gold_reps``
        return value.gold_reps_sha256(self._k_reps())

    def _k_gold_reps(self):
        def make():
            # imported here, not at module level: this module imports no label code (a test pins it), and the representatives are read through this one function
            from c_ept.valuation import value
            return value.fit_gold_bin_means(self.s1, self.labels, population='in_scope')
        return self._once('k_gold_reps', make)               # a GoldReps (means, fallback); ``_once`` makes the means read-only

    def _k_reps(self) -> np.ndarray:
        return self._k_gold_reps().means

    def _kbinned(self, unit) -> np.ndarray:
        """(n, 4) float64: kills blue, kills red and the binned gold of the two teams, g̃ = rep[gold bin] of a team with a kill and 0 for a team without one (the K of the composer).

        수식
        ----
        (HR-9)  g̃_{i,t} = rep_{b_{i,t}}  (k_{i,t} ≥ 1),   g̃_{i,t} = 0  (k_{i,t} = 0) ;   단언: k_{i,t} ≥ 1이면 0 ≤ b_{i,t} ≤ 30

          기호      | 뜻                                                      | 단위 | 범위       | 코드 변수
          k_{i,t}   | 단위 i에서 팀 t가 낸 킬 수 (``units/K``의 앞 두 열)       | 개   | 0, 1, …    | counts
          b_{i,t}   | 팀 t의 킬 골드 구간 (``labels/gold_bin``, 킬이 없으면 −1)  | —    | −1, 0 … 30 | bins
          rep_b     | 구간 b의 대표값 (범위 안 FIT 단위의 구간 평균)            | 골드 | ≥ 0        | self._k_reps()
          g̃_{i,t}   | 팀 t의 이산화한 킬 골드                                 | 골드 | ≥ 0        | 반환값의 뒤 두 열

        해석: 합성기는 킬이 하나 이상인 팀의 골드를 뽑은 구간의 대표값으로, 킬이 없는 팀은 0으로 K 조건 머리에 넣는다(``Composer._draw_row``의 ``np.where(k >= 1, reps[drawn], 0)``).
        이 함수는 학습 행의 K를 같은 규칙으로 만든다. 판정 기준은 골드 머리의 행(``_gold_rows``)과 같은 킬 수 ``units/K``이다. 킬이 하나 이상인 팀에 구간이 없으면 0이라는 그럴듯한
        값으로 학습하지 않고 거부한다(고르지 않은 경기의 단위도 구간 −1이므로 여기서 걸린다). 킬이 없는 팀에 구간이 있는 라벨은 합성기의 규칙대로 0이다. 킬이 있으나 골드가 0인
        팀은 구간 0의 대표값을 갖는다.
        """
        counts = self._kcounts(unit)
        bins = self._labels('labels/gold_bin')[unit].astype(np.int64)
        has = counts >= 1
        reps = self._k_reps()
        if (has & (bins < 0)).any():
            raise ValueError(f'{int((has & (bins < 0)).sum())} team(s) with a kill have no gold bin in labels/gold_bin for the role: the binned K has no representative for them')
        if (has & (bins >= len(reps))).any():
            raise ValueError(f'{int((has & (bins >= len(reps))).sum())} team(s) with a kill have a gold bin above {len(reps) - 1} in labels/gold_bin')
        gold = np.where(has, reps[np.clip(bins, 0, len(reps) - 1)], 0.0)
        return np.column_stack([counts.astype(np.float64), gold])

    def _kraw(self, unit) -> np.ndarray:
        """(n, 4) float64: kills blue, kills red, kill gold blue and red (the K of ``features.m4_masked_inputs``). With ``k_condition='binned'`` the gold is the representative of the
        gold bin (``_kbinned``, equation (HR-9)); with ``'raw'`` (the default) it is the true kill gold of the labels, as in every earlier run.

        (HR-4)의 K 블록을 원시 골드로 낸다. K 조건 'binned'이면 (HR-9)의 대표값 골드로 낸다."""
        if self.k_condition == 'binned':
            return self._kbinned(unit)
        gold = self._labels('labels/kill_gold')[unit]
        if not np.isfinite(gold).all():
            raise ValueError(f'labels/kill_gold is not finite for {int((~np.isfinite(gold)).any(axis=1).sum())} unit(s) of the role')
        return np.column_stack([self._kcounts(unit).astype(np.float64), gold])

    def _kblock(self, unit) -> np.ndarray:
        """(n, 4) float64: kills blue, kills red, kill gold blue and red in thousands (the K columns of the masked M4 input).

        (HR-4)의 K 블록이다."""
        K = self._kraw(unit)
        K[:, 2:] /= 1000.0
        return K

    # -- the units of the role (open rows) -------------------------------------------------------------------------------------------------
    def _event_of_unit(self, kind: int) -> np.ndarray:
        def make():
            ev_kind, ev_unit = self._s1('event_rows/kind'), self._s1('event_rows/unit').astype(np.int64)
            sel = np.flatnonzero(ev_kind == kind)
            of = np.full(len(self._unit_in), -1, dtype=np.int64)
            if np.bincount(ev_unit[sel], minlength=len(of)).max(initial=0) > 1:
                raise ValueError(f'a unit has two event rows of kind {kind}')
            of[ev_unit[sel]] = sel
            return of
        return self._once(f'event_of_unit_{kind}', make)

    def _open_units(self):
        def make():
            unit = np.flatnonzero(self._unit_in & self._s1('units/in_scope'))
            if unit.size == 0:
                raise ValueError('the role has no unit in scope')
            _, event_view = self._view_maps()
            ev3 = self._event_of_unit(1)[unit]
            if (ev3 < 0).any():
                raise ValueError(f'{int((ev3 < 0).sum())} unit(s) in scope have no M3 event row')
            views = event_view[ev3]
            if (views < 0).any():
                raise ValueError(f'{int((views < 0).sum())} M3 event row(s) of the role have no event view in the S2 store')
            match = self._s1('units/match')[unit].astype(np.int64)
            self._check_views(views, match, self._s1('event_rows/ref')[ev3], 'M3 event')
            return SimpleNamespace(unit=unit, match=match, views=views, size=self._s1('units/size')[unit].astype(np.int64))
        return self._once('open_units', make)

    def _open_feat(self) -> np.ndarray:
        """The feature rows of the M3 event views of the in-scope units (the open M4 head does not build them)."""
        return self._once('open_feat', lambda: self.build(self.s2, self._open_units().views))

    def _open_m4(self):
        """The in-scope units of the role that have an M4 event view and their feature rows; the others have no conversion window.

        (HR-2)의 열린 M4Conv 행이다(창과 보기의 일치 조건은 ``parts``의 해석)."""
        def make():
            u = self._open_units()
            ev4 = self._event_of_unit(2)[u.unit]
            has = ev4 >= 0
            exposure = self._s1('units/conv_exposure_ms')[u.unit]
            if ((exposure > 0) != has).any():
                raise ValueError('a unit has an M4 event row without a conversion window, or a window without an M4 event row')
            _, event_view = self._view_maps()
            views = event_view[ev4[has]]
            if (views < 0).any():
                raise ValueError(f'{int((views < 0).sum())} M4 event row(s) of the role have no event view in the S2 store')
            self._check_views(views, u.match[has], self._s1('event_rows/ref')[ev4[has]], 'M4 event')
            return SimpleNamespace(sel=np.flatnonzero(has), feat=self.build(self.s2, views), n_no_view=int((~has).sum()), views=views)
        return self._once('open_m4', make)

    # -- the rows of every head ------------------------------------------------------------------------------------------------------------
    def parts(self, head: str, kind: str, *, masked_obj_k: bool = False) -> HeadParts:
        """The ``HeadParts`` of the head ``head`` ('M2Rate', ... 'M4Conv') of ``kind`` ('masked' or 'open'). ``masked_obj_k``: give the
        masked M3 object head its K block (B1 conditions that head on K, decision P6; B0 does not and reads no label for it). ValueError
        for a head and kind that do not exist (M2 has no open heads).

        수식
        ----
        (HR-1)  y_i = n_q,   e_i = Δ_q                  M2Rate의 행 i = (t₀ 행, 구간 q) 쌍 (Δ_q > 0인 쌍만 행이다)
        (HR-2)  개수 머리: M3Kills y_{i,blue} = k_blue, y_{i,red} = k_red, e_i = 1;   M3Obj y_{ic} = O_c, e_i = 1;
                M4Conv y_{ic} = 전환 수_c, e_i = 전환 창의 길이 (e_i > 0인 행만).   범주 머리: c_i = m (M2Mark), z (M2Zone), s (M3Size)
        (HR-3)  ξ_i = min(k, 6),   c_i = b_i            M3Gold의 행 i = 킬이 k ≥ 1인 팀 하나 (파랑 먼저)
        (HR-4)  (k_blue, k_red, g_blue / 1000, g_red / 1000)       K 블록

          기호           | 뜻                                          | 단위  | 범위                         | 코드 변수
          y_i, y_{ic}    | 행 i(의 칸 c)에서 센 개수                   | 개    | 0, 1, …                      | ``HeadParts.y``
          c_i            | 행 i의 범주 번호(C = 범주의 수)             | —     | 0, …, C−1 (구역은 −1도)      | ``HeadParts.y`` (범주 머리)
          e_i            | 행 i의 노출                                 | s (M3는 무차원 1) | > 0              | ``HeadParts.e``
          n_q, Δ_q       | t₀ 행의 구간 q의 발생 수, 노출              | 개, s | 0, 1, …; 0–10                | ``onsets/bin`` 개수, ``exposure_ms``/1000
          k_blue, k_red  | 한 교전의 팀별 킬 수                        | 개    | 0, 1, …                      | ``units/K``의 앞 두 열
          g_blue, g_red  | 한 교전의 팀별 킬 골드(원시)                | 골드  | ≥ 0                          | ``labels/kill_gold``
          O_c, 전환 수_c | 교전의 칸 c의 오브젝트 수, 전환 수          | 개    | 0, 1, …                      | ``units/O``, ``units/conv``
          ξ_i            | 팀 항목 i의 킬 수 층                        | —     | 1, …, 6                      | ``codes['layer']``
          b_i            | 팀 항목 i의 골드 구간 번호                  | —     | 0, …, 30                     | ``HeadParts.y`` (M3Gold)

        해석
        ----
        식 (HR-1)–(HR-4)는 머리마다 한 행이 무엇이고 그 행의 개수나 범주와 노출이 어느 표에서 오는지를 정한다. 열린 행은 범위 안 교전의 사건 보기(M3는 첫 킬 f, M4는 교전
        끝 δ)를 읽는다(설계서 3.2절·3.3절의 두 보기). M4 사건 보기는 교전 끝 δ < T일 때만 있고 그때만 전환 창의 노출이 0보다 크므로, 창과 보기의 유무가 어긋나면 자료
        오류로 거부한다(설계서 3.3절). M3Gold는 킬이 0인 팀의 골드가 0으로 정해지므로 킬이 하나 이상인 팀만 항목을 갖고, 그 팀에 골드 구간이 없으면 거부한다(계획 P4).

        근거
        ----
        계획 P4(M3 분해와 골드 킬 수 층), P6(머리가 조건으로 삼는 것), P10(노출), 전역 제약의 확률 족(노출이 0인 칸은 적합에서 빼고 NLL에는 0), 판정 R17(구역 라벨),
        R18(행을 B0·B1이 공유), R19(킬 머리는 팀마다 y의 열 하나), R22(채점하지 않는 행을 센다), 설계서 3.1절(첫 킬 하나를 센다)과 3.2절·3.3절(가린·열린 보기),
        검산 노트 math-control-03 C절(노출 0인 칸)."""
        if (head, kind) not in VALID_HEADS:
            raise ValueError(f'there is no {kind} head {head!r}; the heads are {sorted(VALID_HEADS)}')
        return self._masked(head, masked_obj_k) if kind == 'masked' else self._open(head)

    def _frame_view_numbers(self) -> np.ndarray:
        """(n frame rows of the role,) int64: the frame-clock S2 view of every frame row of the role, in the order of ``frame().rows``
        (the order of the feature rows that ``_frame_features`` builds)."""
        def make():
            frame_view, _ = self._view_maps()
            return frame_view[np.flatnonzero(self._frame_in)]
        return self._once('frame_view_numbers', make)

    def view_numbers(self, head: str, kind: str) -> np.ndarray:
        """(n,) int64: the S2 view that row i of ``parts(head, kind)`` reads, in the order of the rows of ``parts`` (the order of ``take``).

        A masked head reads the frame-clock view (``views/src`` 0) of the frame row of the row (M2Rate) or of the frame row of the onset
        (the other masked heads, the masked M4 head only for the onsets that have a conversion window); an open M3 head reads the src 1
        view of its unit and the open M4 head the src 2 view. ``parts`` holds these views only as feature rows (``feat`` is built from
        them, ``take`` picks them), and a model that reads the S2 store itself, as the EPT does, needs the view numbers. They come from
        the maps ``parts`` uses, so the two cannot drift apart. ValueError as for ``parts``: a head and a kind that do not exist, a role
        without rows, a frame row or a unit without its view.

        수식
        ----
        (HR-6)  v_i = Ψ[r^X_i]   (머리에 특징 행 번호가 있을 때),   v_i = Ψ[i]   (없을 때)
        (HR-7)  v_i = Ψ_frame[r^X(ι_i)]   (가린 M4: 행이 전환 창이 있는 발생만 남은 부분 집합이라 특징 행 번호 r^X_i가 없다)

          기호      | 뜻                                                                | 단위      | 범위                    | 코드 변수
          i         | ``parts(head, kind)``의 행 번호 (기호 등록부의 행 번호 i와 같은 뜻) | 행 번호   | 0 .. 행 수 − 1          | 반환값의 위치
          v_i       | 행 i가 읽는 S2 보기의 번호                                        | 보기 번호 | 0 .. 보기 수 − 1        | 반환값[i]
          Ψ         | 특징 행을 만든 보기의 번호 열 (가린 머리는 Ψ_frame, 열린 머리는 그 머리의 사건 시계 보기) | 보기 번호 | 0 .. 보기 수 − 1 | _open_units().views, _open_m4().views
          Ψ_frame   | 역할의 프레임 행마다 그 프레임 시계 보기(``views/src`` 0)의 번호 (프레임 행 순서) | 보기 번호 | 0 .. 보기 수 − 1 | _frame_view_numbers()
          r^X_i       | 행 i가 읽는 특징 행의 번호                                        | 행 번호   | 0 .. 특징 행 수 − 1     | parts.take[i]
          r^X(ι)      | 역할의 발생 ι가 읽는 프레임 특징 행의 번호                        | 행 번호   | 0 .. 특징 행 수 − 1     | _onsets().take, ι는 _onsets().idx에서 찾는다
          ι_i       | 행 i의 평가 행 번호 (가린 M4 머리에서는 S1의 발생 번호)           | 발생 번호 | 0 .. S1 발생 수 − 1     | parts.row[i]

        해석
        ----
        이 함수는 새 수치를 만들지 않고, 행 묶음이 이미 쓰는 "행 → 보기" 대응을 번호로 내준다.
        특징 행은 보기마다 하나씩 만들어지므로 행 i가 읽는 보기의 번호는 특징 행 번호 r^X_i에 놓인 보기 번호이다 (HR-6).
        가린 M4 머리는 전환 창이 없는 발생을 뺀 부분 집합이라 r^X_i가 없으므로, 평가 행 번호 ι_i로 그 발생의 프레임 특징 행을 찾아 간다 (HR-7).
        발생 번호를 정렬된 열에서 찾으므로 부분 집합이어도 행의 순서가 행 묶음과 같고, 찾은 발생이 ι_i와 다르면 오류로 멈춘다.

        근거
        ----
        S4a 계획 Q4 (EPT는 특징 열 대신 S2 보기 번호가 필요하다; 추가만 한다), S3 판정 R18 (행과 정답은 이 모듈에서 한 번만 만든다),
        인과 규칙 A3 (가린 머리는 프레임 시계 보기, 열린 머리는 사건 시계 보기).
        """
        p = self.parts(head, kind)                                     # the checks of ``parts``, and the order of its rows
        if kind == 'open':
            views = self._open_m4().views if head == 'M4Conv' else self._open_units().views
            return views if p.take is None else views[p.take]
        frame_views = self._frame_view_numbers()
        if head != 'M4Conv':
            return frame_views[p.take]
        on = self._onsets()
        at = np.minimum(np.searchsorted(on.idx, p.row), len(on.idx) - 1)      # on.idx ascending, p.row a subset (onsets with a window)
        if (on.idx[at] != p.row).any():
            raise RuntimeError('view_numbers: a row of the masked M4 head is not an onset of the role')
        return frame_views[on.take[at]]

    def _masked(self, head: str, masked_obj_k: bool) -> HeadParts:
        if head == 'M2Rate':
            return self._m2_rate()
        on = self._onsets()
        _, feat = self._frame_features()
        codes = {'q': on.bin, 'm': on.mark, 'z': on.zone, 's': on.size}
        # 머리마다 조건으로 삼는 범주는 사슬에서 앞선 단계까지이다(q → m → z → s; (HEAD-2), (HEAD-3), 계획 P4).
        names = {'M2Mark': ('q',), 'M2Zone': ('q', 'm'), 'M3Size': ('q', 'm', 'z'), 'M3Obj': ('q', 'm', 'z', 's'),
                 'M3Kills': ('q', 'm', 'z', 's')}
        if head in ('M2Mark', 'M2Zone', 'M3Size'):
            y = {'M2Mark': on.mark, 'M2Zone': on.zone, 'M3Size': on.size}[head]
            return HeadParts(feat, on.take, {k: codes[k] for k in names[head]}, y, None, on.match, on.idx)
        if head in ('M3Obj', 'M3Kills'):
            # 식 (HR-2)의 개수 머리: 노출 e_i = 1이어서 (NLL-1)의 μ_i = λ_i·e_i가 μ_i = λ_i이다. 예측값이 곧 기대 개수이다.
            # M3Obj: the 18 objective counts; M3Kills: the kills of the two teams (a cell for each)
            y = self._s1('units/O')[on.unit].astype(np.int64) if head == 'M3Obj' else self._kcounts(on.unit)
            K = self._kblock(on.unit) if head == 'M3Obj' and masked_obj_k else None
            return HeadParts(feat, on.take, {k: codes[k] for k in names[head]}, y, np.ones(len(on.idx)), on.match, on.idx, K=K,
                             first_team=self._first_teams(head, on.mark, on.unit, 'onset'))
        if head == 'M3Gold':
            return self._gold_rows(feat, on.take, on.unit, on.idx, on.match, codes)
        # 식 (HR-2)의 M4Conv 행: 입력은 [t₀ 행 특징 | q, m, z | K | O]이고 목표는 전환 수이며 노출은 전환 창의 길이이다. 범주 q, m, z는 만든 입력의 열에서 되읽어
        # 입력과 같은 값을 쓴다.
        # M4Conv: the masked M4 input is built by the one function that builds it (leakage test (a)) from the true K and O
        m4 = features.m4_masked_inputs(feat[on.take], on.bin, on.mark, on.zone, self._kraw(on.unit), self._s1('units/O')[on.unit])
        exposure = self._s1('units/conv_exposure_ms')[on.unit] / 1000.0
        keep = exposure > 0
        d = feat.shape[1]
        m4 = m4[keep]
        mcodes = {'q': np.rint(m4[:, d]).astype(np.int64), 'm': np.rint(m4[:, d + 1]).astype(np.int64),
                  'z': np.rint(m4[:, d + 2]).astype(np.int64)}
        return HeadParts(m4[:, :d], None, mcodes, self._s1('units/conv')[on.unit][keep].astype(np.int64), exposure[keep], on.match[keep],
                         on.idx[keep], int((~keep).sum()), m4=m4)

    def _m2_rate(self) -> HeadParts:
        """(HR-1)의 M2Rate 행을 만든다: 노출이 있는 (t₀ 행, 구간) 쌍마다 발생 수 y와 노출 e(초)이다."""
        rows = np.flatnonzero(self._frame_in)
        if rows.size == 0:
            raise ValueError('the role has no frame row')
        pos, feat = self._frame_features()
        exposure = self._s1('frame_rows/exposure_ms')[rows]
        at = np.flatnonzero(self._onset_in)
        # 식 (HR-1)의 y_i = n_q: 역할의 발생을 (t₀ 행, 구간) 칸마다 센다. 한 발생은 첫 킬이 든 구간의 칸에만 속한다(설계서 3.1절).
        counts = np.bincount(pos[self._s1('onsets/row')[at].astype(np.int64)] * N_Q + self._s1('onsets/bin')[at].astype(np.int64),
                             minlength=len(rows) * N_Q).reshape(len(rows), N_Q)
        # 식 (HR-1)의 Δ_q > 0 조건.
        seen = exposure > 0
        if (counts[~seen] > 0).any():
            raise ValueError(f'{int(counts[~seen].sum())} onset(s) lie in bins without exposure')
        r, q = np.nonzero(seen)
        return HeadParts(feat, r, {'q': q}, counts[r, q].astype(np.int64), exposure[r, q] / 1000.0,
                         self._s1('frame_rows/match')[rows][r].astype(np.int64), rows[r], int((~seen).sum()))

    def _gold_rows(self, feat, take, unit, row_id, match, codes) -> HeadParts:
        """The team entries with at least one kill, blue entries first: the categories of the entry, the gold bin of the entry, and the
        team of every entry in ``team``. ``take`` indexes ``feat`` for the rows ``unit`` (the entry of a unit takes its row).

        (HR-3)의 팀 항목을 만든다."""
        K = self._kcounts(unit)
        # 식 (HR-3)의 팀 항목.
        sel = [np.flatnonzero(K[:, 0] >= 1), np.flatnonzero(K[:, 1] >= 1)]
        rep = np.concatenate(sel)
        team = np.concatenate([np.full(len(s), t) for t, s in enumerate(sel)]).astype(np.int64)
        entry = {name: codes[name][rep] for name in ('q', 'm', 'z', 's') if name in codes}
        # 식 (HR-3)의 층 ξ = min(k, 6): 킬 수를 여섯 층으로 묶어 골드 분포의 조건으로 쓴다(계획 P4).
        entry['layer'] = np.minimum(K[rep, team], N_LAYER)
        # 식 (HR-3)의 목표 b: 라벨의 골드 구간이다. 킬이 있는 팀에 구간이 없으면 거부한다.
        gold = self._labels('labels/gold_bin')[unit[rep], team].astype(np.int64)
        if (gold < 0).any():
            raise ValueError(f'labels/gold_bin has no bin for {int((gold < 0).sum())} team(s) with a kill')
        return HeadParts(feat, rep if take is None else take[rep], entry, gold, None, match[rep], row_id[rep], 0, team)

    def _open(self, head: str) -> HeadParts:
        u = self._open_units()
        if head == 'M4Conv':
            m4 = self._open_m4()
            unit = u.unit[m4.sel]
            return HeadParts(m4.feat, None, {}, self._s1('units/conv')[unit].astype(np.int64),
                             self._s1('units/conv_exposure_ms')[unit] / 1000.0, u.match[m4.sel], unit, m4.n_no_view,
                             K=self._kblock(unit), O=self._s1('units/O')[unit])
        feat = self._open_feat()
        if head == 'M3Size':
            return HeadParts(feat, None, {}, u.size, None, u.match, u.unit)
        if head == 'M3Obj':
            return HeadParts(feat, None, {'s': u.size}, self._s1('units/O')[u.unit].astype(np.int64), np.ones(len(u.unit)), u.match,
                             u.unit, K=self._kblock(u.unit))
        if head == 'M3Kills':
            codes, first = {'s': u.size}, self._first_teams(head, self._s1('units/mark')[u.unit].astype(np.int64), u.unit, 'unit')
            if first is not None:
                codes['d'] = first                                          # the open head reads d from its input (the masked head reads the mark m)
            return HeadParts(feat, None, codes, self._kcounts(u.unit), np.ones(len(u.unit)), u.match, u.unit, first_team=first)
        return self._gold_rows(feat, None, u.unit, u.unit, u.match, {'s': u.size})


# ---- scores ------------------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class HeadScores:
    """The NLL of the rows of one head: ``nll`` (n,) per design row, ``match`` (n,) their match numbers, ``row`` (n,) the id of the
    evaluation row each belongs to, and ``n_unscored``, the rows that were left out (no zone, no exposure). ``per_row()`` sums the NLL
    of the rows with one id (the 18 bins of a frame row, the two teams of an onset) and gives the value and match of each evaluation row;
    ``mean`` and ``bootstrap`` are the match-weighted mean (weights 1/n_u of ``features.match_weights``) and its match bootstrap.

    수식
    ----
    (HR-5)  v_I = Σ_{i∈I} ℓ_i        평가 행 I(같은 ``row`` 번호를 가진 설계 행의 집합)의 점수
            이 v_I가 (NLL-4)와 (NLL-5)가 평균하는 행 점수 v_i이고, 가중은 w = 1 / n_u이다(``features.match_weights``).

      기호 | 뜻                                                    | 단위 | 범위                                             | 코드 변수
      v_I  | 평가 행 I의 점수                                      | nat  | ≥ 0 (범주 점수의 허용오차 때문에 조금 작을 수 있다) | ``per_row()``의 values
      ℓ_i  | 설계 행 i의 NLL(그 행의 칸 점수 (NLL-1)–(NLL-3)의 합) | nat  | ≥ 0 (같은 이유)                                  | ``nll``의 i번째 원소
      I    | 같은 ``row`` 번호를 가진 설계 행의 집합               | —    | 1개 이상                                         | ``row``
      n_u  | 경기 u의 평가 행 수                                   | 행   | ≥ 1                                              | ``match``

    해석
    ----
    식 (HR-5)는 한 평가 행에 속한 설계 행의 NLL을 더해 그 행의 점수 하나로 만든다. 발생률 머리에서는 한 t₀ 행의 구간 행들이, 골드 머리에서는 한
    발생의 두 팀 항목이 한 평가 행이므로, 같은 t₀ 행이나 같은 발생은 평균에서 평가 행 하나로만 센다. 모형에서 한 평가 행의 칸들은 입력이 주어지면 서로 독립이므로
    칸 점수의 합은 그 평가 행의 음의 로그 우도이다. 경기를 하나의 독립 단위로 보므로(nll 모듈 글 (나)) 평가 행마다 1/n_u를 주어 경기마다 가중의 합이 1이 되게 하고,
    (NLL-5)는 그 단위인 경기를 통째로 다시 뽑는다. 채점할 칸이 하나도 없는 t₀ 행은 평가 행이 되지 않는다(판정 R22).

    근거
    ----
    설계서 8절 비교 공통 규칙(경기 가중), 계획 P7(관문 공통 조건)과 P9(학습과 평가의 가중), 판정 R22(채점하지 않는 행은 평균에서 빼고 센다)."""
    nll: np.ndarray
    match: np.ndarray
    row: np.ndarray
    n_unscored: int = 0

    def per_row(self) -> tuple:
        """(HR-5)의 v_I를 평가 행 번호마다 계산하여 그 값과 경기 번호를 낸다."""
        ids, inverse = np.unique(self.row, return_inverse=True)
        # 식 (HR-5)의 합 Σ_{i∈I} ℓ_i: 평가 행 번호가 같은 설계 행의 NLL을 더한다. 평가 행은 번호의 크기 순으로 나온다.
        values = np.bincount(inverse.reshape(-1), weights=self.nll, minlength=len(ids))
        # 평가 행마다 경기 번호 하나를 붙인다. 같은 평가 행의 설계 행은 같은 경기의 것이다(``HeadRows``가 그렇게 행을 만든다).
        match = np.zeros(len(ids), dtype=np.int64)
        match[inverse.reshape(-1)] = self.match
        return values, match

    def mean(self) -> float:
        """평가 행 점수 v_I의 경기 가중 평균, 곧 (NLL-4)이다."""
        values, match = self.per_row()
        return nll_scores.weighted_mean(values, features.match_weights(match))

    def bootstrap(self, B: int = nll_scores.BOOT_B, seed: int = nll_scores.BOOT_SEED) -> np.ndarray:
        """같은 v_I와 가중에 경기 재표본을 적용한 B개의 복제, 곧 (NLL-5)이다."""
        values, match = self.per_row()
        return nll_scores.match_bootstrap(values, match, features.match_weights(match), B, seed)


class HeadBatch(NamedTuple):
    """n rows to evaluate every masked head on, without a store: ``row_view`` (n, d) the row features (b0: 39 columns; b1: the columns
    of ``features.b1_columns``), ``q`` ``m`` ``z`` ``s`` (n,) the categories of the onset, ``K`` (n, 4) = kills blue, kills red, kill
    gold blue, kill gold red (raw gold) and ``O`` (n, 18) the objectives; a head reads K and O only if its model conditions on them (B0:
    K only through the kill layer of each team, O not at all; B1: also in the object and the conversion heads, decision P6).

    사슬 (HEAD-1)–(HEAD-4)의 한 발생에 넣는 입력(t₀ 행의 특징, 발생의 범주, 킬 K와 오브젝트 O)을 저장소 없이 모은 묶음이다. 머리가 K와 O 가운데
    무엇을 읽는지는 모형이 정한다(계획 P6)."""
    row_view: np.ndarray
    q: np.ndarray
    m: np.ndarray
    z: np.ndarray
    s: np.ndarray
    K: np.ndarray
    O: np.ndarray
