"""Declarative, hash-locked run profiles of the run kernel (S4b-prep K3, decision D11). Standard library and ``run.io`` only.

A profile is one JSON file that says what a run is, so that a run is started from a file and from nothing else (the earlier stacks took the
same settings from command-line flags, module constants and operator configs, in a different shape for each stack). There are three kinds:

* ``train``: a training run (model size, data share, seed, epoch budget, optimiser, and one switch for every behaviour that Track M adds).
* ``evaluate``: the evaluation of trained models on one role of the split (the role, the models, the exposure, the statistics, and for the role TEST ``holdout_lock_sha256``,
  the sha256 of the S5 lock file that anchors the TEST gate: required for TEST, null or absent for DEV and STOP; K6b C1, decision D28).
* ``measure``: the P6 bit-identity measurement. It names a train profile by relative path and SHA-256 (``train_profile``, ``train_sha256``) and
  says how many steps (``steps``) and how many repeated calls (``repeat``) to measure. The measured configuration is that train profile and
  nothing else, so the measurement cannot drift from the fit that it vouches for.

``load(path)`` reads the file once through ``run.io.read_json_with_sha`` (the strict reader, which also gives the SHA-256 of the bytes that it
parsed), checks every field, fills the omitted fields with their defaults and returns a frozen ``Profile`` that carries ``kind`` and
``sha256``, the SHA-256 of those bytes. The runner records that hash in the ledger, so a run names the exact file it was started from. This module imports no model code and no Track M code:
the names of the Track M settings are written here as text, and the contract test (``tests/test_run_profile.py``) pins them, so that a rename
in Track M must change both sides on purpose. ``Profile.to_train_kwargs()`` gives the settings that ``TrainConfig`` takes; K6 maps the rest
(the data preparation and the encoder) to the real constructors.

Symbols
-------
    p, b     the profile file and its bytes (the one read of ``run.io.read_json_with_sha``)
    h        the lock, ``h = SHA-256(b)`` as 64 lowercase hex digits (``Profile.sha256``)
    d        the JSON object in ``b``, as ``run.io.read_json_with_sha`` returns it
    k        the kind, ``d['kind']``, one of ``KINDS = (train, evaluate, measure)``
    F_k      the field names of kind k (``FIELDS[k]``); ``R_k`` of them are required, every other field f has the default ``v_f``
    c_f      the check of field f (a type, a range or an allowed set); it returns the normal form of the value (an int that is given for a
             real field becomes a float, a list of models becomes a tuple)
    S        the settings of the profile, ``S(f) = c_f(d[f])`` when f is in d, else ``v_f``
    q        ``fraction``, the share of the FIT matches that a train run uses
    s, s'    ``seed`` (the training seed) and ``selection_seed`` (null or a whole number); ``membership_seed() = s'`` if s' is not null, else ``s``
    r, h_r   ``train_profile`` (a relative path, resolved against the folder of p) and ``train_sha256`` of a measure profile; ``T`` the profile
             that r names (``Profile.train``)

Equations
---------
    accept(d)  <=>  d['kind'] is in KINDS  and  keys(d) is inside F_k + {'kind'}  and  R_k is inside keys(d)  and  every c_f(d[f]) passes
                    and  (k = measure  =>  T = load-without-reference(folder(p) / r) is a train profile and SHA-256(T) = h_r)
    load(p)    =  Profile(k, h, S, T)    when read_json_with_sha(p) = (d, h) and b holds no carriage return    (one read: d, h and the
                                         carriage-return test are all functions of the same b)
    omission   :  an omitted field has the value of the EPT-T002 run (or the old behaviour of the Track M switch), so the bare train profile
                  ``{"kind": "train"}`` carries the T002 settings (the batch size 32 and the device are runner constants, not profile fields)
    ranges     :  q in (0, 1];  lr, weight_decay, terrain_smooth_weight >= 0;  1 <= epochs <= 1000;  patience >= 1;  warmup, field_channels >= 0;
                  0 <= seed, selection_seed < 2^63;  1 <= bootstrap_B <= 100000;  chunk_views >= 1;  1 <= steps <= 100000;  2 <= repeat <= 10
                  composed (K6e) is null or an object whose keys sample_matches, sample_seed, draws, mc_batches, draw_seed are required and progress_rows (default 16)
                  is optional; draws is a multiple of mc_batches with at least two draws in a batch (``_composed``)
    whole numbers are JSON integers (12, not 12.0) and a flag is true or false (not 1 or 0): a bool is never a number, a float never a count
    r          :  non-empty, no leading "/", no backslash, no ":", no control character, every "/" segment non-empty and not "." or "..", no
                  segment that ends in "." or " " (Windows strips them: ".. " would be ".."), and the resolved file lies inside folder(p)

해석
----
프로파일은 실행 한 번을 해시로 잠그는 선언형 설정 파일이다. 옛 스택은 같은 설정을 명령줄 인자, 모듈 상수, 운영자 설정 파일로 따로 받았으므로
"이 실행은 어떤 설정이었나"를 한 곳에서 답할 수 없었다. 여기서는 파일 하나가 설정의 전부이고 그 바이트의 SHA-256이 잠금이다. 생략한 필드는 EPT-T002의
값(또는 Track M 스위치의 옛 동작)으로 채워지므로 ``{"kind": "train"}`` 한 줄이 T002 설정이고, 바뀌는 동작은 필드를 적어야만 켜진다. 배치 크기 32와 장치는
프로파일 필드가 아니라 실행기의 상수이다(K6이 원장에 적는다).

모르는 필드, 잘못된 형, 허용 집합 밖의 값은 필드 이름을 대며 거부한다. 거부 메시지에는 값도 모르는 필드의 이름도, 참조 경로와 해시도 싣지 않는다(경로나
경기 키일 수 있다). 모르는 필드는 개수와, 가장 비슷한 알려진 이름만 말한다(오타 안내). 실수는 정수로 써도 받지만(1 → 1.0) 정수 필드에 실수(12.0)나
논리값을 쓰면 거부한다. 논리값은 true/false만 받는다. 이렇게 엄격해야 같은 설정이 두 가지 글자로 적혀 두 해시를 갖는 일이 줄고, 숫자 12.0을 에폭으로
읽는 식의 조용한 변환이 없다. 줄 끝에 CR이 든 파일(CRLF 체크아웃)도 거부한다. 같은 내용이 체크아웃마다 다른 해시를 갖게 되기 때문이다. 에폭, 걸음 수,
반복 수, 부트스트랩 수에는 상한이 있어, 오타(10^12)가 나중에 큰 메모리 할당으로 나타나지 않는다.

해시는 읽은 바이트의 해시여야 한다. 그래서 파일을 한 번만 읽는다. ``run.io.read_json_with_sha``가 읽은 바이트 하나에서 (사전, 해시)를 함께 내고, CR 거부도 같은
바이트에 대한 검사(``check``)이다. 파싱한 내용과 해시한 내용, CR을 본 내용이 같은 바이트이므로, 읽은 뒤에 파일이 바뀌거나 바뀌었다 되돌아와도
받은 쌍은 읽은 그 내용의 쌍이다. 이전 방식은 파일을 읽고 해시하고 다시 읽어 두 파싱을 비교했는데, 그 사이에 파일이 B로 바뀌었다 A로 돌아오면 해시는 B의 바이트이고
설정은 A의 것인데도 받아들이는 잔여 위험이 있었다. 한 번 읽기는 그 위험을 없앤다. 같은 내용을 다른 공백으로 쓴 파일은 해시가 다르다. 잠금은 바이트에 대한
것이며 의미에 대한 것이 아니다(의미가 같음은 ``settings``가 같음으로 확인한다).

경기 선택(``fraction``이 1보다 작을 때)의 시드는 ``selection_seed``이다. null(기본값)이면 학습 시드를 쓴다. 이것이 T002의 규칙이다(``proxy_matches(s1,
share, seed)``). 정수이면 학습 시드와 무관하게 그 값으로 고정한다. 이것이 T009/matched의 규칙이다(``select_membership(..., selection_seed=...)``). S4b의 시드
2와 3 학습은 25 % 경기 집합이 시드마다 달라지면 안 되므로 정수를 적는다. ``Profile.membership_seed()``가 둘 중 쓸 값을 돌려준다.

``measure`` 프로파일은 설정을 따로 적지 않고 train 프로파일 하나를 상대 경로와 SHA-256으로 가리킨다. P6은 S4b 학습이 실제로 돌리는 설정(shifted, binned,
split, random_view, 매끄러움 가중)의 비트 동일성을 보여야 하므로, 필드 목록을 복사하면 어긋날 수 있지만 해시로 가리킨 파일 하나는 어긋나지 않는다. 가리킨
파일은 같은 ``load``로 읽고(measure나 evaluate이면 거부, 그래서 참조는 겹치지 않는다) 해시가 ``train_sha256``과 같아야 한다.

``Profile.to_train_kwargs()``는 TrainConfig가 받는 설정(``warmup``은 TrainConfig에서 ``warmup_steps``)만 돌려준다. 이것은 소비처의 분할이 아니다. 자료 준비도
``kill_law``, ``k_condition``, ``swap_mode``, ``column_drop``, ``stats_population``, ``seed``, ``selection_seed``를 읽으므로 나머지는 K6이 실제 생성자에 맞춘다.
통계를 만든 ``column_drop``과 ``stats_population``은 TrainConfig에도 같은 값으로 가야 한다(``train._check_inputs``가 다른 설정으로 만든 통계를 거부하므로, M5 검토 H1).
TrainConfig의 M 갈래 필드가 이 키에 빠지면 tests/test_run_profile.py의 X1 확인이 막는다.
"""
from __future__ import annotations

import difflib
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, NamedTuple

from c_ept.run import io as run_io

KINDS = ('train', 'evaluate', 'measure')
SEED_LIMIT = 1 << 63                                         # a seed is 0 <= seed < 2^63: a numpy SeedSequence and torch.manual_seed accept it
_HASH = re.compile(r'[0-9a-f]{64}')
_SUGGEST_MAX = 64                                            # an unknown name longer than this is not compared with the known names


class ProfileRefused(run_io.IORefused):
    """A profile that is not acceptable: an unknown field, a wrong type, a value outside its range or set, a missing field, a changed file.

    An ``IORefused`` (so a ``ValueError``), like the strict reader's refusals. The message names the field and the rule; it never holds a
    value, the name of an unknown field, a path or a hash. It is raised outside any ``except`` block, so ``__context__`` and ``__cause__``
    are ``None``.
    """


def _refuse(name: str, rule: str):
    """Raise ``ProfileRefused`` for the field ``name`` (a name defined by this module, so it is safe to print); ``rule`` says what is required."""
    raise ProfileRefused(f'field {name!r} {rule}')


# ---- the checks of one field: each returns the normal form of the value or refuses ----------------------------------------------------
class _Choice:
    """The check of a field with an allowed set of texts; ``allowed`` is public so that ``CHOICES`` (and the contract test) can list it."""

    def __init__(self, *allowed: str):
        self.allowed = allowed

    def __call__(self, name: str, value: Any) -> str:
        if type(value) is not str or value not in self.allowed:
            _refuse(name, 'must be one of ' + ', '.join(repr(a) for a in self.allowed))
        return value


def _whole(low: int, high: int | None = None) -> Callable[[str, Any], Any]:
    """A whole number in ``[low, high]`` (``high`` is included; none means no upper bound)."""
    rule = f'must be a whole number of at least {low}' + ('' if high is None else f' and at most {high}') + ' (an integer, not 12.0 or true)'

    def check(name, value):
        if type(value) is not int or value < low or (high is not None and value > high):
            _refuse(name, rule)
        return value
    return check


def _real(low: float, *, open_low: bool = False, high: float | None = None) -> Callable[[str, Any], Any]:
    rule = 'must be a number ' + (f'above {low}' if open_low else f'of at least {low}') + ('' if high is None else f' and at most {high}')

    def check(name, value):
        number = None
        if type(value) in (int, float):                      # a bool is not a number here
            try:
                number = float(value)
            except OverflowError:                            # an integer too large for a float
                number = None
        ok = (number is not None and math.isfinite(number) and (number > low if open_low else number >= low)
              and (high is None or number <= high))
        if not ok:
            _refuse(name, rule)
        return number
    return check


def _flag(name: str, value: Any) -> bool:
    if type(value) is not bool:
        _refuse(name, 'must be true or false')
    return value


def _or_null(check: Callable[[str, Any], Any]) -> Callable[[str, Any], Any]:
    def checked(name, value):
        return None if value is None else check(name, value)
    return checked


def _models(name: str, value: Any) -> tuple:
    if not (type(value) is list and value and all(type(m) is str and m != '' for m in value) and len(set(value)) == len(value)):
        _refuse(name, 'must be a non-empty list of distinct, non-empty model names')
    return tuple(value)


def _checkpoints(name: str, value: Any) -> Mapping:
    if not (type(value) is dict and all(type(k) is str and k != '' and type(v) is str and v != '' for k, v in value.items())):
        _refuse(name, 'must be an object that maps each model name to a non-empty path text')
    return MappingProxyType(dict(sorted(value.items())))


def _relative_path(name: str, value: Any) -> str:
    """A relative path with forward slashes that cannot leave the folder it is resolved against (the rule is in the module docstring)."""
    ok = type(value) is str
    if ok:
        ok = not ('\\' in value or ':' in value or any(ord(c) < 32 or ord(c) == 127 for c in value))
    if ok:                  # every "/" segment is non-empty (so no leading "/", no "//", no trailing "/", and the empty text is refused) and does
        #                     not end in "." or " ": that refuses "." and ".." too (they strip to nothing), as Windows would strip such an ending
        ok = all(part != '' and part == part.rstrip(' .') for part in value.split('/'))
    if not ok:
        _refuse(name, 'must be a relative path with forward slashes, without an empty, "." or ".." segment, a drive letter or a backslash')
    return value


def _digest(name: str, value: Any) -> str:
    if type(value) is not str or not _HASH.fullmatch(value):
        _refuse(name, 'must be 64 lowercase hexadecimal digits')
    return value


# ---- the composed block of an evaluate profile (S4b-prep K6e, decision D30) ------------------------------------------------------------
# The composed V_start pass on a sample of STOP matches (A7's V_start CRPS gain and D14's excluded-row share) has parameters that a later, separately approved cost probe sets: how many matches,
# which seeds, how many Monte Carlo draws and in how many batches the Monte Carlo error is estimated. They have no default: a block without one of them is refused, so that none of them is invented here.
#
#   (CB-1)  composed = null  (the pass is off, the evaluation is that of every earlier profile)   or   an object with the keys of COMPOSED_REQUIRED and, optionally, progress_rows
#   (CB-2)  S = draws, K = mc_batches:   K >= 2,   S = K x s with a whole s >= 2   (s = the draws of one batch)
#
#      기호        | 뜻                                                  | 단위 | 범위                    | 키
#      N           | 표본으로 뽑을 STOP 경기의 수                         | 경기 | 1 .. 1,000,000          | sample_matches
#      seed_N      | 경기 표본을 뽑는 시드                                | —    | 0 .. 2^63 - 1           | sample_seed
#      S           | 행마다의 몬테카를로 모의 횟수                        | 회   | 2 .. 1,000,000          | draws
#      K           | 몬테카를로 오차를 재는 배치 수(모의를 K등분)          | 개   | 2 .. 10,000             | mc_batches
#      seed_S      | 합성기의 난수 시드(행의 흐름은 [시드, 프레임 행])       | —    | 0 .. 2^63 - 1           | draw_seed
#      r           | 진행 줄 하나가 덮는 행 수                            | 행   | 1 .. 1,000,000, 기본 16  | progress_rows
#
# 해석: 합성 V_start는 비싸서 STOP 전체가 아니라 표본에서만 하고(D30), 표본의 크기와 모의 횟수는 비용 탐침이 정한다. 그래서 이 값들은 기본값이 없고, 하나라도 빠지면 거부한다. 모의 횟수 S는 K개 배치로
# 똑같이 나뉘어야 하고(한 배치에 2회 이상), 그래야 배치 평균으로 몬테카를로 표준오차를 잴 수 있다. progress_rows만 기계적인 값(합성기 한 블록 = 16행)이어서 기본값이 있다.
COMPOSED_REQUIRED = ('sample_matches', 'sample_seed', 'draws', 'mc_batches', 'draw_seed')
COMPOSED_PROGRESS_ROWS = 16                                  # the default of progress_rows: the rows of one block of the Composer (compose.DEFAULT_BLOCK); a progress line, nothing more


def _composed(name: str, value: Any) -> Mapping:
    """The composed block as an immutable mapping in the order of the keys (``progress_rows`` filled in), or ``ProfileRefused`` naming ``composed`` or ``composed.<key>`` and the rule.

    Never echoes a value or an unknown key (a key may be a path): an unknown key is counted only. Rule (CB-2) is checked after the keys, naming ``composed.draws``."""
    if type(value) is not dict:
        _refuse(name, 'must be null or an object with the keys ' + ', '.join(COMPOSED_REQUIRED) + ' (and optionally progress_rows)')
    checks = {'sample_matches': _whole(1, 1_000_000), 'sample_seed': _SEED, 'draws': _whole(2, 1_000_000), 'mc_batches': _whole(2, 10_000), 'draw_seed': _SEED,
              'progress_rows': _whole(1, 1_000_000)}
    unknown = [key for key in value if key not in checks]
    if unknown:
        raise ProfileRefused(f'field {name!r} has {len(unknown)} field(s) not defined for the block')
    out: dict[str, Any] = {}
    for key, check in checks.items():
        if key in value:
            out[key] = check(f'{name}.{key}', value[key])
        elif key in COMPOSED_REQUIRED:
            _refuse(f'{name}.{key}', 'is required in the composed block')
        else:
            out[key] = COMPOSED_PROGRESS_ROWS
    if out['draws'] % out['mc_batches'] != 0 or out['draws'] // out['mc_batches'] < 2:
        _refuse(f'{name}.draws', 'must be a multiple of mc_batches with at least two draws in each batch')
    return MappingProxyType(out)


# ---- the fields of each kind: (name, check, default); _REQUIRED has no default ------------------------------------------------------
class _Required:
    """The default of a field that must be written."""
    def __repr__(self):
        return '<required>'


_REQUIRED = _Required()


class _Spec(NamedTuple):
    name: str
    check: Callable[[str, Any], Any]
    default: Any


_SIZE = _Choice('S', 'M')
_SEED = _whole(0, SEED_LIMIT - 1)

_SPECS = MappingProxyType({
    # defaults of a train profile = EPT-T002 (training_operator.FIXED: S, 25 %, seed 20261022, 30 epochs, patience 2, lr 0.001,
    # weight decay 0.01, 10 warm-up steps) and the old behaviour of every Track M switch (the plan's Interfaces blocks M1 .. M9).
    # FIXED also holds batch_size 32 and device 'cuda': they are runner constants (K6 writes them to the ledger), not profile fields.
    'train': (
        _Spec('size', _SIZE, 'S'),
        _Spec('fraction', _real(0, open_low=True, high=1), 0.25),
        _Spec('seed', _SEED, 20261022),
        _Spec('epochs', _whole(1, 1000), 30),
        _Spec('patience', _whole(1), 2),
        _Spec('lr', _real(0), 0.001),
        _Spec('weight_decay', _real(0), 0.01),
        _Spec('warmup', _whole(0), 10),
        _Spec('kill_law', _Choice('poisson', 'shifted'), 'poisson'),                         # M1
        _Spec('k_condition', _Choice('raw', 'binned'), 'raw'),                               # M3
        _Spec('decay_groups', _Choice('all', 'split'), 'all'),                               # M4
        _Spec('terrain_smooth_weight', _real(0), 0.0),                                       # M6
        _Spec('swap_mode', _Choice('average', 'random_view', 'none'), 'average'),            # M7
        _Spec('eval_average', _flag, True),                                                  # M7
        _Spec('on_wall_cap', _Choice('fail', 'best_stop'), 'fail'),                          # M8
        _Spec('zone_mark', _flag, True),                                                     # M9
        _Spec('field_channels', _whole(0), 8),                                               # M9 (A6 = 0)
        _Spec('column_drop', _Choice('none', 'store', 'store_and_subset'), 'none'),          # M5
        _Spec('stats_population', _Choice('whole_fit', 'selected_fit'), 'whole_fit'),        # M5
        _Spec('selection_seed', _or_null(_SEED), None),                                      # null: the training seed (T002); integer: fixed (T009)
    ),
    'evaluate': (
        _Spec('role', _Choice('STOP', 'DEV', 'TEST'), _REQUIRED),
        _Spec('models', _models, _REQUIRED),
        _Spec('exposure', _Choice('observed', 'nominal'), 'observed'),                       # M12
        _Spec('winner_truth', _Choice('gold', 'binned'), 'gold'),                            # M2
        _Spec('chunk_views', _or_null(_whole(1)), None),                                     # M14
        _Spec('b0', _flag, False),                                                           # M11
        _Spec('strata', _flag, False),                                                       # M13
        _Spec('price_sensitivity', _flag, False),                                            # M13
        _Spec('b1_swap_average', _flag, False),                                              # M13
        _Spec('bootstrap_B', _whole(1, 100000), 2000),                                       # C4
        _Spec('checkpoints', _checkpoints, MappingProxyType({})),                            # shape only; K6 reads the files
        _Spec('holdout_lock_sha256', _or_null(_digest), None),                               # K6b C1 (D28): the anchor of the TEST gate; required for the role TEST (see ``_resolve``)
        _Spec('composed', _or_null(_composed), None),                                        # K6e (D30): null = no composed V_start pass; the block sets its sample and draws
    ),
    'measure': (
        _Spec('train_profile', _relative_path, _REQUIRED),                                   # the configuration that is measured
        _Spec('train_sha256', _digest, _REQUIRED),
        _Spec('steps', _whole(1, 100000), _REQUIRED),
        _Spec('repeat', _whole(2, 10), 2),                                                   # two calls at least, or there is nothing to compare
    ),
})

FIELDS = MappingProxyType({kind: tuple(spec.name for spec in specs) for kind, specs in _SPECS.items()})
CHOICES = MappingProxyType({kind: MappingProxyType({spec.name: spec.check.allowed for spec in specs if isinstance(spec.check, _Choice)})
                            for kind, specs in _SPECS.items()})                            # the allowed texts of every field that has a set

# What ``TrainConfig`` takes. ``TRAIN_RENAMES`` lists the profile fields that the TrainConfig calls differently (the profile says ``warmup``,
# ``TrainConfig.warmup_steps`` is the same number). ``TRAIN_KEYS`` are the keys of ``Profile.to_train_kwargs()``, in TrainConfig names: every
# Track M field of ``TrainConfig`` (M1, M3-M9) and lr, weight_decay, warmup_steps and patience. It is not a partition of the train fields: data
# preparation also reads kill_law, k_condition, swap_mode, column_drop, stats_population, seed and selection_seed (the statistics are made with
# column_drop and stats_population, and ``TrainConfig`` must carry the same two values, because ``train._check_inputs`` refuses statistics made
# under other settings, S4b-prep M5 review H1). A Track M field that the profile does not route is refused by tests/test_run_profile.py (X1).
TRAIN_RENAMES = MappingProxyType({'warmup': 'warmup_steps'})
TRAIN_KEYS = ('kill_law', 'k_condition', 'decay_groups', 'column_drop', 'stats_population', 'terrain_smooth_weight', 'swap_mode',
              'eval_average', 'on_wall_cap', 'zone_mark', 'lr', 'weight_decay', 'warmup_steps', 'patience')
_PROFILE_FIELD = MappingProxyType({config: profile for profile, config in TRAIN_RENAMES.items()})


def _resolve(data: Mapping) -> tuple[str, dict]:
    """The kind and the settings (every field of the kind, defaults filled) of the JSON object ``data``; ``ProfileRefused`` if not acceptable."""
    if 'kind' not in data:
        _refuse('kind', 'is missing; it must be one of ' + ', '.join(repr(k) for k in KINDS))
    kind = data['kind']
    if type(kind) is not str or kind not in KINDS:
        _refuse('kind', 'must be one of ' + ', '.join(repr(k) for k in KINDS))
    specs = _SPECS[kind]
    known = set(FIELDS[kind])
    unknown = sorted(name for name in data if name != 'kind' and name not in known)
    if unknown:
        near = sorted({close for name in unknown if len(name) <= _SUGGEST_MAX
                       for close in difflib.get_close_matches(name, FIELDS[kind], n=1)})      # known names only: the unknown ones are not echoed
        hint = f' (did you mean: {", ".join(repr(n) for n in near)}?)' if near else ''
        raise ProfileRefused(f'{len(unknown)} field(s) not defined for a {kind} profile{hint}')
    settings: dict[str, Any] = {}
    for spec in specs:
        if spec.name in data:
            settings[spec.name] = spec.check(spec.name, data[spec.name])
        elif spec.default is _REQUIRED:
            _refuse(spec.name, f'is required in a {kind} profile')
        else:
            settings[spec.name] = spec.default
    if kind == 'evaluate' and settings['role'] == 'TEST' and settings['holdout_lock_sha256'] is None:
        # K6b C1 (decision D28): the sha256 of the S5 lock file is the one fixed anchor of the TEST gate. It is part of the profile, hence of the profile without its role, so the
        # DEV rehearsal of a TEST run is made under the same lock. A TEST profile without it could be opened against any ledger.
        _refuse('holdout_lock_sha256', 'is required when the role is TEST (the sha256 of the S5 lock file, the anchor of the TEST gate)')
    return kind, settings


@dataclass(frozen=True)
class Profile:
    """A loaded profile: ``kind``, the lock ``sha256`` (of the file bytes), the resolved ``settings`` (read-only; every field of the kind) and,
    for a measure profile, ``train``: the train ``Profile`` that ``train_profile`` names (``None`` for the other kinds).

    Frozen. ``profile.epochs``, ``profile['epochs']`` and ``profile.settings['epochs']`` are the same value; an attribute of another kind
    (``profile.role`` on a train profile) is an ``AttributeError``. The constructor only checks the shape (a known kind, a 64-digit hash,
    exactly the fields of the kind, a ``train`` that matches ``train_sha256`` for a measure profile and is ``None`` otherwise); the values are
    checked by ``load``, the one door for a file.

    A ``Profile`` can be copied with ``copy.copy`` but not pickled or deep-copied (``settings`` is a ``MappingProxyType``, which CPython does
    not pickle). A child process gets the path and the ``sha256`` and calls ``load`` again, which also checks the lock a second time.
    """
    kind: str
    sha256: str
    settings: Mapping[str, Any] = field(hash=False)
    train: Any = field(default=None, hash=False)

    def __post_init__(self):
        if type(self.kind) is not str or self.kind not in KINDS:
            _refuse('kind', 'must be one of ' + ', '.join(repr(k) for k in KINDS))
        if type(self.sha256) is not str or not _HASH.fullmatch(self.sha256):
            raise ProfileRefused('sha256 must be 64 lowercase hexadecimal digits')
        if not isinstance(self.settings, Mapping) or set(self.settings) != set(FIELDS[self.kind]):
            raise ProfileRefused(f'the settings of a {self.kind} profile must have exactly its fields')
        if self.kind == 'measure':
            ok = isinstance(self.train, Profile) and self.train.kind == 'train' and self.train.sha256 == self.settings['train_sha256']
        else:
            ok = self.train is None
        if not ok:
            raise ProfileRefused('a measure profile carries the train profile that it names (matching train_sha256); the other kinds carry none')
        object.__setattr__(self, 'settings', MappingProxyType(dict(self.settings)))

    def __getitem__(self, name: str) -> Any:
        return self.settings[name]

    def __getattr__(self, name: str) -> Any:                 # only called when the normal lookup fails
        if name == 'settings':                               # a half-built object (copy) has no ``settings`` yet: ask no further, or recurse forever
            raise AttributeError(name)
        try:
            return self.settings[name]
        except KeyError:
            raise AttributeError(f'a {self.kind} profile has no field {name!r}') from None

    def as_dict(self) -> dict:
        """The resolved profile as plain JSON values: ``kind`` and every field. Written to a file it loads to the same ``settings``."""
        out: dict[str, Any] = {'kind': self.kind}
        for name in FIELDS[self.kind]:
            value = self.settings[name]
            out[name] = list(value) if isinstance(value, tuple) else dict(value) if isinstance(value, Mapping) else value
        return out

    def to_train_kwargs(self) -> dict:
        """The settings that ``TrainConfig`` takes (a new dict), for ``TrainConfig(encoder, **kwargs)``: the keys ``TRAIN_KEYS`` (the Track M
        switches, ``lr``, ``weight_decay``, ``warmup_steps`` and ``patience``). Only a train profile has them (a measure profile has them in
        ``.train``).

        This is what TrainConfig takes, not an exclusive partition of the train fields: K6 maps the rest to the real constructors, and data
        preparation also reads ``kill_law``, ``k_condition``, ``swap_mode``, ``column_drop``, ``stats_population``, ``seed`` and
        ``selection_seed``.
        """
        if self.kind != 'train':
            raise ProfileRefused(f'only a train profile has TrainConfig settings (this one is {self.kind})')
        return {key: self.settings[_PROFILE_FIELD.get(key, key)] for key in TRAIN_KEYS}

    def membership_seed(self) -> int:
        """The seed that selects the ``fraction`` of the FIT matches: ``selection_seed`` if it is not null, else the training ``seed``.

        null is the T002 rule (``proxy_matches(s1, share, seed)``); an integer is the T009/matched rule (a fixed selection for every training
        seed). Only a train profile has a selection seed.
        """
        if self.kind != 'train':
            raise ProfileRefused(f'only a train profile has a membership seed (this one is {self.kind})')
        chosen = self.settings['selection_seed']
        return self.settings['seed'] if chosen is None else chosen


def _refuse_carriage_return(raw: bytes) -> None:
    """``check`` of the read-once reader: one content, one hash. A CRLF checkout of the same file must not give a second hash, so a carriage
    return is refused. It is tested on the bytes that were parsed and hashed (K3 re-check n3)."""
    if b'\r' in raw:
        raise ProfileRefused('the profile file contains a carriage return; it must use LF line ends')


def _read_file(path: Path) -> tuple[str, str, dict]:
    """``(kind, sha256, settings)`` of one profile file; the reference of a measure profile is not followed here. One read of the file:
    the object, the hash and the carriage-return test are all taken from the same bytes (``run.io.read_json_with_sha``)."""
    data, digest = run_io.read_json_with_sha(path, check=_refuse_carriage_return)
    kind, settings = _resolve(data)
    return kind, digest, settings


def _load_reference(folder: Path, settings: Mapping) -> Profile:
    """The train ``Profile`` that the measure settings name: resolved against ``folder``, loaded without following a reference, hash checked."""
    target = folder / settings['train_profile']
    try:
        inside = target.resolve().is_relative_to(folder.resolve())       # a junction or link in the middle of the path is followed here
    except (OSError, ValueError):
        inside = False
    if not inside:
        _refuse('train_profile', 'must name a file inside the folder of the measure profile')
    reason = None
    try:
        kind, digest, train_settings = _read_file(target)
    except run_io.IORefused as error:                        # a missing file, a link or a directory too (the strict reader refuses them); the
        reason = str(error)                                  # message holds no value, key or path, and is passed on after the except block
    if reason is not None:
        _refuse('train_profile', f'does not name an acceptable profile ({reason})')
    if kind != 'train':
        _refuse('train_profile', "must name a 'train' profile (a reference does not nest)")
    if digest != settings['train_sha256']:
        _refuse('train_sha256', 'does not match the SHA-256 of the file that train_profile names')
    return Profile('train', digest, train_settings)


def load(path) -> Profile:
    """Read, check and lock the profile file ``path``; ``ProfileRefused`` (or the strict reader's ``IORefused``) if it is not acceptable.

    The file is read once with ``run.io.read_json_with_sha``: the strict object, the SHA-256 and the carriage-return refusal all come from the
    same bytes, so the settings and the lock always describe one content, whatever happens to the file after that read. The returned
    ``Profile`` holds the hash of the bytes that were read. A measure profile also loads the train profile that it names (read once as well),
    from the folder of ``path``, and checks its SHA-256 (``Profile.train``).
    """
    path = Path(path)
    kind, digest, settings = _read_file(path)
    train = _load_reference(path.parent, settings) if kind == 'measure' else None
    return Profile(kind, digest, settings, train)
