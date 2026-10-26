"""Training runner of the run kernel: the profile -> TrainConfig and data, the two stage bodies of the supervisor, the ledger records, the loss-scale
condition, the cap of the training stage and E6 before every epoch (S4b-prep K6a; decisions D11, D12, D13, D24, D27; plan Q9 and the M1-M10 hand-offs).

What it does. ``start(profile_path, run_dir, ...)`` reads a hash-locked train profile and an inputs manifest, refuses what is wrong before anything is
reserved, and hands the supervisor (``run.supervisor``) a plan of two stages, each a function ``module:function(run_dir, *texts)`` that a child process
calls:

    stats   ``stats_stage``  -> ``write_stats``: the loss scales are checked against the training condition, the standardisation statistics are made
                               (the D7 list of constant columns once, on the whole FIT store; the D12 population), ``stats/STATS.json`` and
                               ``stats/RECORD.json`` are written
    train   ``train_stage``  -> ``train``: ``TrainConfig`` and the FIT/STOP data are built from the profile, ``ept.train.train`` runs with E6 before every
                               epoch, ``train/RECORD.json`` and ``train/E6.json`` are written next to the checkpoints in ``train/ckpt``

``train(profile, run_dir, *, inputs, ...)`` is the training stage's work as one function of the profile and the run folder; it needs the opened ``Inputs``
because the stores cannot be derived from the profile (a profile holds settings only). The stage bodies open them with an inputs loader that is named
by text (``LOADER``, ``load_inputs``: the stores, the label store checked against its manifest, the terrain checked against the build of the S2 views, the
B0 loss scales); a test names its own loader.

Symbols
-------
    P, h     the train profile and its lock (``sha256`` of the file); q = ``fraction``; s = ``seed``; s' = ``selection_seed`` (null or a number)
    M_sel    the selected matches ``{FIT, STOP}`` = ``proxy_matches(s1, q, s'')``,  s'' = s' if s' is not null else s   (``Profile.membership_seed``)
    D7       the names of the columns that are constant over the views of the whole FIT store, and ``t_s`` (``data.store_constant_columns``)
    S        the statistics: FIT views of M_sel (``selected_fit``) or of the whole FIT (``whole_fit``), the columns dropped (``column_drop``)
    law, K   ``kill_law`` and ``k_condition`` of the training; rep_1..rep_31 the representatives of the gold bins (``HeadRows.k_reps``)
    T_cap    ``ept.train.WALL_CAP_SECONDS`` (24 h)
    E, W, F  the declared bound of the longest epoch, of one checkpoint write and of the finalisation (``CapAllowance``)
    C        the ``wall_cap_s`` of the training stage
    B        the batch size 32, the device and the thread count: runner constants, not profile fields (``RunnerConstants``)

Equations
---------
    (RUN-1)  M_sel = proxy_matches(s1, q, s'')                         the T002 rule for s' = null, the T009 rule for a number
    (RUN-2)  D7 = constant_columns(FIT store);  S = fit_stats(FIT views,  matches = M_sel.FIT if selected_fit else all,  drop = D7 [+ subset-constant])
    (RUN-3)  accept(scales)  <=>  law(scales) = law  and  K(scales) = K  and  (K = binned  =>  sha(rep(scales)) = sha(rep(rows)))        (D24, Q9)
    (RUN-4)  C = T_cap + E + 2 W + F    when on_wall_cap = best_stop;   C = T_cap    when it is fail
    (RUN-5)  start accepted  <=>  wall budget >= C + F_sup,  F_sup the supervisor's finalisation margin (``supervisor.FINALISATION_MARGIN_S``)
    (RUN-6)  end(head) = early_stopping | budget | max_epochs;   a head with done = True is a success whatever its end (``budget`` included)

해석
----
프로파일은 설정만 담으므로 저장소 위치는 입력 매니페스트가 정하고, 이 모듈은 프로파일 필드가 실제 학습 코드의 어느 인자로 가는지를 한 표(``ROUTES``)에 두어 호출이 그 표로만
인자를 받게 한다. 시험은 표의 이름을 실제 시그니처와 대조하고, 프로파일의 모든 필드에 소비처가 있는지 확인한다. 기본 프로파일은 T002와 같은 TrainConfig·통계·행·배치를 만든다
(통계는 전체 FIT, 행은 역할 전체를 경기 선택으로 자른 것). S4b 프로파일은 D7 목록을 전체 FIT 저장소에서 한 번만 구하고, 통계는 선택된 FIT에서 재며, K 조건의 대표값 31개는
``HeadRows``의 한 출처에서만 온다.

손실 척도(B0 NLL의 평균)는 B0가 맞춘 킬 법칙과 K 조건에 따라 값이 다르므로, 척도 파일이 학습 조건과 다르면(법칙, 조건, 구간 대표값의 sha256) 시작 전에 거부한다(종료 코드 2).
통계 단계가 행의 실제 대표값으로 한 번 더 대조한다. 학습 단계의 상한 C는 'best_stop'이면 24 h에 닿은 에폭을 끝까지 마치는 몫(한 에폭, 체크포인트 쓰기 두 번, 마무리)을
더해 선언하고, 감독자는 남은 예산이 C와 자기 여유를 담을 때만 단계를 시작한다. 그래서 학습 안의 상한이 감독자의 예산보다 먼저 작동한다. 상한에서 끝난 학습('budget')은
실패가 아니라 성공한 끝이며, 기록에 학습 시간과 상한이 함께 남는다. 에폭마다 E6를 확인하고 답을 ``train/E6.json``에 적는다(해시되는 출력).

원장에는 명시한 값이 남는다: 머리글(``Header.runner``)에 실행기 상수와 프로파일의 모든 필드(옛 기본값도 글자로), 입력 매니페스트와 묶인 파일, 척도 파일과 그 조건이 있고, 단계 기록에
D7 목록과 해시, 통계 해시, 대표값의 sha256과 대체 구간 목록, 학습기가 정체성에 적은 값(기본값은 정체성에서 빠지므로 여기서 채움), 감쇠 묶음의 규칙 해시와 머리의 배정, 끝난 이유,
체크포인트 파일마다의 sha256이 있다.

Before every epoch the training re-checks the hashes of the imported module files against the identity in the ledger header (D11; ``EpochCheck``), then E6; both answers are
written to ``train/E6.json`` (schema ``ept-run-epoch-checks-v1``), and a change ends the child with exit code 2.

Numeric policy and approval (K6c, decisions D10 P6 and D32 item 5): ``train`` sets the float32 policy of ``c_ept.run.numeric`` (highest matmul precision, TF32 off, cuDNN deterministic,
deterministic algorithms with ``CUBLAS_WORKSPACE_CONFIG=:4096:8``, ``fill_uninitialized_memory`` off, no autocast) before any model is built, reads the state of the process back and refuses
(exit code 2) when it is not the policy; the state is written to the record of the stage (``numeric``). The ledger header holds the policy that the run asks for and the text of the author's
approval as given (``approval_record``). 훈련 단계는 모형을 만들기 전에 숫자 정책을 켜고 읽어 확인하며, 켠 뒤의 상태는 단계 기록에, 요청한 정책과 승인 문구는 원장 머리글에 남는다.

The FIT and STOP batches are made ahead by the ordered prefetch of the kernel (``c_ept.run.prefetch``, S4b speed-ups SPD item 2, decision D32): one producer thread each, at most
``prefetch.DEPTH`` = 2 batches ahead, the same batches in the same order as the plain ``Batches`` (the producer of the FIT batches also does the CPU work of ``batch_prep``). The
depth is a run constant and is written down as ``prefetch`` in the ledger header and in ``train/RECORD.json``; ``train`` closes both wrappers whatever happens to the training.

The evaluation runner (the checkpoint identities, the STOP evaluation, C4, the DEV access ledger and the TEST gate; K6b) is ``c_ept.run.evaluate``; ``evaluate`` below is its
entry point under the name that the plan gives it.

The measurement runner of the plan (P6: short training calls of one setting in fresh processes, compared bit for bit; K6c) is ``c_ept.run.measure``; ``measure`` below is its entry point under the
name that the plan gives it.

The command line (``python -m c_ept.run.cli {train,evaluate,measure}``; K6c) is ``c_ept.run.cli``.

Not done here (later tasks): the equivalence run with the T002 stack (K8).
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib
import os
import re
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import numpy as np
import torch

from c_ept import runlog
from c_ept.data import store_core
from c_ept.ept import batch_prep, checkpoints, data, encoder, scales
from c_ept.ept import train as ept_train
from c_ept.preprocessing import features, terrain
from c_ept.run import guards, prefetch
from c_ept.run import identity as run_identity
from c_ept.run import io as run_io
from c_ept.run import ledger as run_ledger
from c_ept.run import numeric as run_numeric
from c_ept.run import profile as run_profile
from c_ept.run import supervisor

BATCH_SIZE = 32                                                     # engagements per batch: the T002 value (FIXED in training_operator); a runner constant, not a profile field
DEVICES = ('cpu', 'cuda')
DEFAULT_DEVICE = 'cuda'                                             # the T002 device
DEFAULT_THREADS = 8                                                 # = ept.train.THREADS: the threads of the reported runs
MAX_THREADS = 1024
TRAIN_CONFIG_RUNNER_ONLY = ('model_spec',)                          # TrainConfig fields that no train profile sets (N2's model_spec belongs to the S7 profile)
PROFILE_FIELDS_OTHERWISE_USED = ('size', 'selection_seed')          # profile fields that are no keyword of a call: the table key of the encoder, and the membership seed

ENCODER, TRAIN_CONFIG, HEAD_ROWS = 'c_ept.ept.encoder:EncoderConfig', 'c_ept.ept.train:TrainConfig', 'c_ept.ept.data:ept_head_rows'
PROXY, FIT_STATS, BATCHES, TRAIN = 'c_ept.ept.data:proxy_matches', 'c_ept.ept.data:fit_stats', 'c_ept.ept.data:Batches', 'c_ept.ept.train:train'
_TRAIN_FIELD = MappingProxyType({config: profile for profile, config in run_profile.TRAIN_RENAMES.items()})      # TrainConfig name -> profile name
ROUTES = MappingProxyType({
    ENCODER: MappingProxyType({'field_channels': 'field_channels'}),
    TRAIN_CONFIG: MappingProxyType({key: _TRAIN_FIELD.get(key, key) for key in run_profile.TRAIN_KEYS}),
    HEAD_ROWS: MappingProxyType({'kill_law': 'kill_law', 'k_condition': 'k_condition'}),
    PROXY: MappingProxyType({'share': 'fraction'}),
    FIT_STATS: MappingProxyType({'column_drop': 'column_drop', 'stats_population': 'stats_population'}),
    BATCHES: MappingProxyType({'seed': 'seed', 'swap_mode': 'swap_mode'}),
    TRAIN: MappingProxyType({'epochs': 'epochs', 'seed': 'seed'}),
})


@dataclasses.dataclass(frozen=True)
class RunnerConstants:
    """The constants of the runner that are not profile fields: the batch size (engagements per batch), the device and the thread count.

    ``threads`` is the count of the child's thread variables and of ``deterministic_cpu`` before the call. On the CPU path ``ept.train.train`` sets its own
    ``ept_train.THREADS`` (8) whatever this number is; on the CUDA path the number stays. The count that a training really used is in its record
    (``effective_threads``). K8 runs the kernel with ``threads=8``, as T002 did."""

    batch_size: int = BATCH_SIZE
    device: str = DEFAULT_DEVICE
    threads: int = DEFAULT_THREADS

    def __post_init__(self):
        for name in ('batch_size', 'threads'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise run_io.IORefused(f'runner constant {name} must be a whole number')
        if self.batch_size < 1 or not 1 <= self.threads <= MAX_THREADS:
            raise run_io.IORefused(f'runner constants out of range (batch_size >= 1, 1 <= threads <= {MAX_THREADS})')
        if not isinstance(self.device, str) or self.device not in DEVICES:
            raise run_io.IORefused(f'runner constant device must be one of {DEVICES}')

    @property
    def cuda_visible_devices(self) -> str:
        return '0' if self.device == 'cuda' else ''

    def to_record(self) -> dict:
        return {'batch_size': self.batch_size, 'device': self.device, 'threads': self.threads, 'cuda_visible_devices': self.cuda_visible_devices}


def route(profile, callee: str) -> dict:
    """The keyword arguments that the profile gives to ``callee`` (a key of ``ROUTES``): ``{parameter: profile value}``."""
    return {parameter: profile[field] for parameter, field in ROUTES[callee].items()}


@dataclasses.dataclass(frozen=True)
class Inputs:
    """What a run reads: the S1 and S2 stores, the label store, the static terrain channels (``static_source``: the record of where they come from or None),
    the thirteen B0 loss scales and the path of the scales file (None when the scales were given directly)."""

    s1: object
    s2: object
    labels: object
    static: object
    static_source: object
    scales: Mapping
    scales_path: object


# ---- the inputs of a run ---------------------------------------------------------------------------------------------------------------------
class InputsRefused(run_io.IORefused):
    """The inputs manifest or a store it names is not acceptable. A refusal (exit code 2): nothing was computed."""


class ScalesConditionRefused(run_io.IORefused):
    """The loss-scale file cannot be used for this training: it is missing or not valid, or it was made under another kill law, K condition or set of gold-bin
    representatives (decision D24, plan Q9). A refusal (exit code 2). The message names the field and, for the law and the condition, both values."""


INPUTS_SCHEMA = 'ept-run-inputs-v1'
TERRAIN_FILES = ('TERRAIN.json', 'terrain.npz', 'geodesic.npy')
_MANIFEST_FIELDS = frozenset({'schema', 's1', 's2', 'labels', 'terrain', 'scales', 'scales_provenance'})
_HEX = re.compile(r'[0-9a-f]{64}')


def _absolute(name: str, text) -> str:
    if not isinstance(text, str) or '\x00' in text or not Path(text).is_absolute() or '..' in Path(text).parts:
        raise InputsRefused(f'inputs manifest field {name} must be an absolute path without ".."')
    return text


def read_manifest(path) -> dict:
    """The inputs manifest ``path`` (see ``read_manifest_with_sha``), without the hash."""
    return read_manifest_with_sha(path)[0]


def read_manifest_with_sha(path) -> tuple:
    """``(manifest, sha256)`` of the inputs manifest ``path``, read once (strict JSON through ``run.io``: the object and the hash are those of the same bytes):
    exactly the fields ``schema`` (``INPUTS_SCHEMA``), ``s1``, ``s2``, ``labels`` (store folders),
    ``terrain`` (the terrain folder, or null: only a loader that takes the static channels from elsewhere may go without), ``scales`` (the ``B0_SCALES.json`` file)
    and ``scales_provenance`` (null, or the mapping that ``scales.load_scales(expected_provenance=)`` takes). Every path is absolute. ``InputsRefused`` otherwise."""
    document, digest = run_io.read_json_with_sha(path)
    if set(document) != _MANIFEST_FIELDS:
        raise InputsRefused('the inputs manifest must have exactly the fields ' + ', '.join(sorted(_MANIFEST_FIELDS)))
    if document['schema'] != INPUTS_SCHEMA:
        raise InputsRefused(f'the inputs manifest schema must be {INPUTS_SCHEMA}')
    for name in ('s1', 's2', 'labels', 'scales'):
        _absolute(name, document[name])
    if document['terrain'] is not None:
        _absolute('terrain', document['terrain'])
    if document['scales_provenance'] is not None and not isinstance(document['scales_provenance'], dict):
        raise InputsRefused('inputs manifest field scales_provenance must be null or an object')
    return document, digest


def manifest_files(manifest: Mapping) -> list:
    """The files that the ledger binds as inputs of a stage, as sorted absolute paths with forward slashes: ``STORE.json`` of the three stores (the manifest of a
    store holds the hash of every file of it; the 18.7 GiB payload is not hashed again), the scales file and, if there is a terrain, its three files."""
    files = [Path(manifest[name]) / 'STORE.json' for name in ('s1', 's2', 'labels')] + [Path(manifest['scales'])]
    if manifest['terrain'] is not None:
        files += [Path(manifest['terrain']) / name for name in TERRAIN_FILES]
    return sorted(path.as_posix() for path in files)


def _open_store(text: str, name: str):
    path = Path(text)
    if not (path / 'STORE.json').is_file():
        raise InputsRefused(f'the {name} store has no STORE.json')
    store = store_core.Store(path)
    if store.manifest.get('complete') is not True:
        raise InputsRefused(f'the {name} store is not complete')
    return store


def open_stores(manifest: Mapping) -> tuple:
    """``(s1, s2, labels)``: the three stores of the manifest, opened (complete manifest required; no payload is read or hashed)."""
    return tuple(_open_store(manifest[name], name) for name in ('s1', 's2', 'labels'))


def read_scales(manifest: Mapping) -> tuple:
    """``(scales, path)``: the thirteen B0 loss scales of the manifest's scales file (``scales.load_scales``, with the expected provenance if the manifest gives one)
    and the path of the file. ``ScalesConditionRefused`` for a file that is missing or not valid."""
    path = Path(manifest['scales'])
    try:
        values = scales.load_scales(path, expected_provenance=manifest['scales_provenance'])
    except ValueError as error:
        raise ScalesConditionRefused(f'the loss-scale file is not valid: {error}') from None
    except OSError as error:
        raise ScalesConditionRefused(f'the loss-scale file cannot be read ({type(error).__name__})') from None
    return values, path


def terrain_hashes(folder) -> dict:
    """``{file name: sha256}`` of the three terrain files of ``folder``; ``InputsRefused`` for a file that cannot be read."""
    try:
        return {name: run_io.sha_file(Path(folder) / name) for name in TERRAIN_FILES}
    except OSError:
        raise InputsRefused('a terrain file cannot be read') from None


# Ref: c_ept/workflows/run_s4a.py ``_require_build_terrain`` (R-S4a-18, A5-F10): the same rule, the same record (stages.views.terrain_sha256 of the S2 STORE.json,
# ``runlog.sha256_json`` of the hashes of the three terrain files); written again because that module is frozen and a command module.
def check_terrain_build(s2_manifest: Mapping, hashes: Mapping) -> None:
    """Refuse (``InputsRefused``) unless the terrain folder as it is now is the terrain that built the S2 views.

    The S2 build records ``stages.views.terrain_sha256`` = ``runlog.sha256_json`` of the hashes of the three terrain files. The static channels and the geodesic bias
    of the model come from the terrain folder now; a folder that was rebuilt would train on a terrain other than the one of the views. A store without the record
    is refused as well: nothing is guessed (review A5-F10).

    해석: S2 보기는 빌드 때의 지형으로 만든 값이다. 학습이 읽는 정적 채널과 측지 편향은 지금 폴더의 지형에서 나오므로, 폴더가 바뀌었거나 빌드 기록이 없으면 거부한다.
    """
    stages = s2_manifest.get('stages') if isinstance(s2_manifest, Mapping) else None
    views = stages.get('views') if isinstance(stages, Mapping) else None
    built = views.get('terrain_sha256') if isinstance(views, Mapping) else None
    if built is None:
        raise InputsRefused('the S2 store records no terrain hash (stages.views.terrain_sha256), so the terrain folder cannot be checked against its build')
    if not isinstance(built, str) or _HEX.fullmatch(built) is None:
        raise InputsRefused('the S2 store records no valid terrain hash (stages.views.terrain_sha256 is not 64 lowercase hex digits)')
    if runlog.sha256_json(dict(hashes)) != built:
        raise InputsRefused('the terrain folder differs from the terrain that built the S2 views (terrain_sha256 of the S2 store)')


# Ref: c_ept/workflows/run_s4a.py ``_load_inputs`` (the stores, ``labels.verify()``, ``encoder.static_channels(terrain.Terrain(...))``) and ``inspect_inputs``
# (the terrain hashes). The prices of that bundle are not read: the training does not use them. The S3-run consistency checks of ``inspect_inputs`` are
# the business of the step that writes the manifest and the scales file (the ledger binds the files by hash).
def load_inputs(manifest_path) -> Inputs:
    """The ``Inputs`` of a run, from the manifest at ``manifest_path``: the stores (the label store verified against its manifest), the terrain (checked against the
    build of the S2 views; the static channels of ``encoder.static_channels``) and the B0 scales. ``InputsRefused`` for a manifest without a terrain or a store that
    is not as the rule says."""
    manifest = read_manifest(manifest_path)
    if manifest['terrain'] is None:
        raise InputsRefused('the terrain is required: the manifest names no terrain folder')
    s1, s2, labels = open_stores(manifest)
    hashes = terrain_hashes(manifest['terrain'])
    check_terrain_build(s2.manifest, hashes)
    labels.verify()                                                  # the derived labels hold the FIT/STOP/DEV labels with sentinels for the other roles: a small store, checked fully
    values, path = read_scales(manifest)
    static = encoder.static_channels(terrain.Terrain(manifest['terrain']))
    return Inputs(s1=s1, s2=s2, labels=labels, static=static, static_source={'terrain_files_sha256': hashes}, scales=values, scales_path=path)


def check_scales_condition(profile, scales_path, k_reps_sha256=None) -> dict:
    """The condition of the loss-scale file ``scales_path`` (``scales.scales_condition``: ``kill_law``, ``k_condition``, ``k_reps_sha256`` with the defaults
    filled in), after refusing (``ScalesConditionRefused``) a file that was made under another setting than the training's.

    Rules. The file's kill law must equal the profile's, and its K condition too. Under the binned condition the file must record the sha256 of its 31
    representatives of the gold bins, and, when ``k_reps_sha256`` is given (the sha of the representatives that the rows of the training are made of), it must
    equal it. Before the stores are opened the representatives are not known: the caller then gives none, and the file's record is only required to exist.

    해석: 척도는 B0 NLL의 평균이라 B0가 맞춘 킬 법칙과 K 조건이 다르면 값이 다르다(결정 D5, D24, 계획 Q9). 이동 법칙·구간 대표값으로 학습하는 EPT가 옛 법칙으로 잰 척도로
    정규화되면 블록 사이의 무게가 틀어지므로 시작하기 전에 거부한다. 구간 대표값은 31개의 sha256으로 대조한다.
    """
    try:
        condition = scales.scales_condition(scales_path)
    except ValueError as error:
        raise ScalesConditionRefused(f'the loss-scale file is not valid: {error}') from None
    except OSError as error:
        raise ScalesConditionRefused(f'the loss-scale file cannot be read ({type(error).__name__})') from None
    for name in ('kill_law', 'k_condition'):
        if condition[name] != profile[name]:
            raise ScalesConditionRefused(f'the loss scales were made under {name} {condition[name]!r} but the training asks for {profile[name]!r}: '
                                         'the scales must be those of a B0 under the training condition (decision D24)')
    if profile['k_condition'] == 'binned':
        if condition['k_reps_sha256'] is None:
            raise ScalesConditionRefused('the loss-scale file records no representatives of the gold bins (k_reps_sha256) although the training is binned')
        if k_reps_sha256 is not None and condition['k_reps_sha256'] != k_reps_sha256:
            raise ScalesConditionRefused('the representatives of the gold bins of the loss scales differ from those of the training rows (k_reps_sha256)')
    return condition


def encoder_config(profile) -> encoder.EncoderConfig:
    return dataclasses.replace(encoder.SIZES[profile['size']], field_channels=profile['field_channels'])


def train_config(profile) -> ept_train.TrainConfig:
    return ept_train.TrainConfig(encoder_config(profile), **profile.to_train_kwargs())


def select_matches(profile, s1) -> dict:
    """``{'FIT': matches, 'STOP': matches}``: the matches of the fraction of the profile, chosen by the membership seed (``selection_seed`` if the profile has
    one, else the training seed)."""
    return data.proxy_matches(s1, seed=profile.membership_seed(), **route(profile, PROXY))


def make_stats(profile, inputs: Inputs, *, check=None) -> data.Stats:
    """The standardisation statistics of the run. ``column_drop='none'`` with ``stats_population='whole_fit'`` is the call of every earlier run (the whole FIT, no
    column dropped). Otherwise the D7 list is computed once on the views of the whole FIT store (``store_constant_columns``) when columns are dropped, and the
    statistics are measured on the selected FIT matches (``'selected_fit'``, D12) or on the whole FIT."""
    settings = route(profile, FIT_STATS)
    if settings['column_drop'] == 'none' and settings['stats_population'] == 'whole_fit':
        return data.fit_stats(inputs.s1, inputs.s2, check=check)
    store = ()
    if settings['column_drop'] != 'none':
        store = data.store_constant_columns(inputs.s2, np.flatnonzero(features.split_views(inputs.s1, inputs.s2, 'FIT')))
    matches = select_matches(profile, inputs.s1)['FIT'] if settings['stats_population'] == 'selected_fit' else None
    return data.fit_stats(inputs.s1, inputs.s2, matches=matches, check=check, store_constant=store, **settings)


# Ref: c_ept/workflows/run_s4a.py ``_train_stage`` (lines 586-592) and c_ept/workflows/run_training.py (lines 200-204): rows of the whole role cut to the proxy
# matches, batches of ``args.batch_size`` with the training seed, FIT shuffled and STOP in a fixed order, the same statistics. Written again here (the frozen
# stack is a command module that the kernel may not import, Q8); K8 compares the outputs of both.
def make_train_data(profile, inputs: Inputs, stats: data.Stats, constants: RunnerConstants) -> ept_train.TrainData:
    """The ``TrainData`` of the run, built as the T002 stack builds it: the rows of the whole role cut to the selected matches (``EngagementRows(rows, matches)``),
    FIT batches shuffled (``training=True``, made for the profile's swap mode) and STOP batches in a fixed order, both read through the same statistics.

    The rows are those of the whole role before the cut, as in T002: under ``kill_law='shifted'`` a unit whose first-kill team scored no kill anywhere in the
    role (also outside the selected matches) refuses the run. That is a data error (Review Focus 2) and fails closed.

    Both are returned inside the ordered prefetch of the kernel (``prefetch.Prefetch``, depth ``prefetch.DEPTH``; S4b speed-ups SPD item 2, decision D32): one producer thread each makes the
    next batches while the device works on the current one, in the order and with the contents of the plain ``Batches`` (``Prefetch.source``); the producer of the FIT batches also does
    the CPU work of a batch (``batch_prep.Preparation`` for the kill law and the zone mark of the profile: the coins of 'random_view' applied, the data checks of the heads and of the loss made once,
    the index lists of the loss made; S4b speed-ups item 3, D32; for a CUDA device the batch is also laid in one pinned buffer for each dtype, item 4; a batch that the checks refuse is raised in the consumer at
    its place). The wrappers forward ``rows``, ``stats``, ``len`` and the other attributes that ``ept.train`` reads.
    The caller closes them (``prefetch.close_all``) whatever happens to the training: ``train`` does.

    해석: 배치를 만드는 일(저장소 읽기, 표준화, 13블록의 행 자르기)은 시드·에폭·배치 번호만의 함수이고 전역 난수를 쓰지 않으므로, 한 스레드가 앞서 만들어 두어도 배치의 내용과 순서와 학습의 난수 흐름은
    그대로이다. 장치가 일하는 동안 다음 배치를 만들어 걸음 사이의 기다림을 숨기는 것이 전부이다."""
    selected = select_matches(profile, inputs.s1)
    rows = route(profile, HEAD_ROWS)
    grouped = {role: data.EngagementRows(data.ept_head_rows(inputs.s1, inputs.s2, inputs.labels, role, **rows), selected[role]) for role in ('FIT', 'STOP')}
    batches = route(profile, BATCHES)
    fit = data.Batches(grouped['FIT'], constants.batch_size, stats=stats, training=True, **batches)
    stop = data.Batches(grouped['STOP'], constants.batch_size, batches['seed'], stats=stats, training=False)
    fit = prefetch.Prefetch(fit, prefetch.DEPTH, transform=batch_prep.Preparation(kill_law=profile['kill_law'], zone_mark=profile['zone_mark'], device=constants.device))
    stop = prefetch.Prefetch(stop, prefetch.DEPTH)
    return ept_train.TrainData(fit, stop, dict(inputs.scales), inputs.static, static_source=inputs.static_source)


def prefetch_record() -> dict:
    """The record of how the batches were made, for the ledger header (``runner_record``) and the hashed training record: the run constant ``depth`` of the ordered prefetch (a code constant, ``prefetch.DEPTH``, in
    the hashed code of the run), one ordered producer, and what the producer of each kind does (S4b speed-ups SPD item 2)."""
    return {**prefetch.describe(prefetch.DEPTH), 'fit': 'the CPU work of batch_prep.Preparation (the frame, the data checks, the index lists, the packed pinned buffers of a CUDA device) is done in the producer thread', 'stop': 'the batches only'}


def effective_settings(profile) -> dict:
    record = {'profile': profile.as_dict(), 'profile_sha256': profile.sha256, 'membership_seed': profile.membership_seed()}
    if profile['decay_groups'] == 'split':
        record['decay_rule_sha256'] = ept_train.decay_rule_sha256()
    return record


# ---- the stage records -------------------------------------------------------------------------------------------------------------------------
STATS_FILE = 'stats/STATS.json'
STATS_RECORD_FILE = 'stats/RECORD.json'
CHECKPOINT_DIR = 'train/ckpt'
TRAIN_RECORD_FILE = 'train/RECORD.json'
E6_FILE = 'train/E6.json'
STATS_RECORD_SCHEMA = 'ept-run-stats-record-v1'
TRAIN_RECORD_SCHEMA = 'ept-run-train-record-v1'
EPOCH_CHECKS_SCHEMA = 'ept-run-epoch-checks-v1'                    # the per-epoch identity and E6 checks (``E6_FILE``)
# What the training identity (``train._identity``) leaves out when a setting has its old default ("absent means the default", plan M1): the record names every one.
IDENTITY_DEFAULTS = MappingProxyType({'kill_law': 'poisson', 'k_condition': 'raw', 'decay_groups': 'all', 'terrain_smooth_weight': 0.0, 'swap_mode': 'average',
                                      'eval_average': True, 'on_wall_cap': 'fail', 'zone_mark': True, 'column_drop': 'none', 'stats_population': 'whole_fit'})


def _scales_record(profile, inputs: Inputs, k_reps_sha256) -> dict:
    """The check of the loss-scale file against the training and the record of it: ``{'file_sha256', 'condition'}``. A run without a scales file is refused: with
    no file there is no condition to compare, and a missing comparison must not read as a passed one."""
    if inputs.scales_path is None:
        raise ScalesConditionRefused('the inputs name no scales file, so the condition of the loss scales cannot be checked')
    condition = check_scales_condition(profile, inputs.scales_path, k_reps_sha256)
    return {'file_sha256': run_io.sha_file(inputs.scales_path), 'condition': condition}


def _reps_record(rows) -> dict | None:
    """``{'sha256', 'fallback'}`` of the 31 gold-bin representatives of the rows (``HeadRows.k_reps_sha256`` / ``k_reps_fallback``, the one source), None when raw."""
    sha = rows.k_reps_sha256()
    return None if sha is None else {'sha256': sha, 'fallback': [int(b) for b in rows.k_reps_fallback()]}


def write_stats(profile, run_dir, *, inputs: Inputs, constants: RunnerConstants) -> dict:
    """The statistics stage: check the loss scales against the training condition, make the statistics, write ``stats/STATS.json`` and the explicit record
    ``stats/RECORD.json`` (the effective settings, the selected matches, the statistics, the D7 list and the dropped columns with their hashes, the
    representatives of the gold bins, the scales file) under ``run_dir``; return the record. Nothing is written when the scales are refused."""
    run_dir = Path(run_dir)
    if profile['kill_law'] == 'poisson' and profile['k_condition'] == 'raw':
        reps = None                                                  # the plain law and the raw gold have no representatives: the whole FIT rows are not built to find that out
    else:                                                            # (under 'shifted' the rows are built here: a first-kill inconsistency is found before the statistics)
        reps = _reps_record(data.ept_head_rows(inputs.s1, inputs.s2, inputs.labels, 'FIT', **route(profile, HEAD_ROWS)))
    scales_record = _scales_record(profile, inputs, None if reps is None else reps['sha256'])
    stats = make_stats(profile, inputs)
    selected = select_matches(profile, inputs.s1)
    listed = lambda names: {'columns': list(names), 'sha256': runlog.sha256_json(list(names))}                # noqa: E731 - a list of column names and its hash
    stats.save(run_dir / STATS_FILE)
    record = {'schema': STATS_RECORD_SCHEMA, 'profile_sha256': profile.sha256, 'settings': effective_settings(profile), 'runner': constants.to_record(),
              'membership': {'seed': profile.membership_seed(), 'fraction': profile['fraction'], 'matches': {role: int(len(selected[role])) for role in ('FIT', 'STOP')},
                             'selected_sha256': runlog.sha256_json({role: [int(m) for m in selected[role]] for role in ('FIT', 'STOP')})},
              'stats': {'sha256': stats.sha256, 'n_views': int(stats.n_views), 'population': stats.population, 'column_drop': stats.column_drop},
              'd7': None if profile['column_drop'] == 'none' else listed(stats.dropped_store), 'dropped': listed(stats.dropped), 'k_reps': reps, 'scales': scales_record}
    run_io.write_json_atomic(run_dir / STATS_RECORD_FILE, record)
    return record


NUMERIC_VOLATILE = ('threads',)       # the one field of the measured numeric state that may differ between the processes of one run (the trainer sets its own count; ``effective_threads``)


def held_numeric(attempts, numeric) -> list:
    """The list of the numeric states of the processes of one training run: ``attempts`` (as the E6 file holds them, oldest first) plus ``numeric`` of the process that starts now, or
    ``attempts`` unchanged when ``numeric`` is None. ``NumericRefused`` (exit code 2) when a field of the measured state other than ``NUMERIC_VOLATILE`` differs from the first
    attempt: a run that is resumed under another torch, CUDA or cuDNN version, device, device name or attention-backend flag is not one run of one numeric state (m1 of the K6c review).

    수식: attempts = [M_0, ..., M_{k-1}], 새 프로세스 M_k:  M_k \\ {threads} = M_0 \\ {threads} 이어야 한다 (모든 이름에서 같아야 이어 하기를 허용).
    해석: 24 h 학습을 이어 하는 프로세스가 처음과 다른 torch 버전·장치·SDPA 설정에서 돌면 한 번의 실행이라 부를 수 없으므로 시작 전에 거부하고, 통과한 프로세스는 모두 목록에 남긴다.
    스레드 수만 처음과 달라도 되는 것은 트레이너가 자기 값을 다시 정하기 때문이다."""
    attempts = list(attempts)
    if numeric is None:
        return attempts
    measured = dict(numeric['measured'])
    if attempts:
        first = attempts[0]['measured']
        names = sorted(name for name in set(first) | set(measured) if name not in NUMERIC_VOLATILE and first.get(name) != measured.get(name))
        if names:
            raise run_numeric.NumericRefused('environment', 'the training is resumed under another numeric state than its first process had: ' +
                                             '; '.join(f'{name} {first.get(name)!r} -> {measured.get(name)!r}' for name in names))
    attempts.append({'attempt': len(attempts), 'measured_sha256': numeric['measured_sha256'], 'measured': measured})
    return attempts


class EpochCheck:
    """The ``check`` hook of ``ept.train.train``: before every epoch the call runs, first the identity check (``verifier``, a function without arguments that
    raises ``IdentityChanged`` or ``Paused``; D11: "each epoch re-checks those hashes"), then E6 (``probe``, ``guards.e6`` unless a test gives another). Each
    answer is written down in ``E6_FILE`` as ``{'call', 'identity', 'e6'}`` (``identity`` is 'verified', 'refused' or, with no verifier, 'not_checked'); the file
    (schema ``EPOCH_CHECKS_SCHEMA``, historically the E6 file; the name ``train/E6.json`` is kept so that the declared outputs stay the same) is rewritten whole,
    atomically, after every check, and is one of the hashed outputs of the training stage. A refusal of either check is written down too and then raised, so the
    child ends with exit code 2. A second call of the stage on the same run (a resumed training) continues the list.

    해석: 에폭마다 가져온 모듈 파일의 해시가 처음 잡은 정체성과 같은지 먼저 확인하고(Git 없이 해시만 다시 계산한다), 그다음 E6를 확인한다. 24 h 학습 도중 파일이 바뀌면
    다음 에폭 경계에서 멈춘다."""

    def __init__(self, path, probe=None, verifier=None, numeric=None):
        self.path, self.probe, self.verifier = Path(path), probe, verifier
        prior = run_io.read_json(self.path) if self.path.exists() else {}
        self.checks = list(prior.get('checks', []))
        self.numeric_attempts = held_numeric(prior.get('numeric_attempts', []), numeric)

    def __call__(self) -> None:
        identity = 'not_checked'
        if self.verifier is not None:
            try:
                self.verifier()
            except (run_identity.IdentityChanged, run_identity.Paused) as refusal:
                self._write({'identity': 'refused', 'refused': str(refusal)})
                raise
            identity = 'verified'
        probe = self.probe if self.probe is not None else guards.e6
        try:
            answer = probe()
        except guards.GateRefused as refusal:
            self._write({'identity': identity, 'refused': str(refusal)})
            raise
        self._write({'identity': identity, 'e6': answer})

    def _write(self, entry: dict) -> None:
        self.checks.append({'call': len(self.checks), **entry})
        document = {'schema': EPOCH_CHECKS_SCHEMA, 'checks': self.checks}
        if self.numeric_attempts:                                                    # only a training that gave its numeric state keeps the list (a bare EpochCheck writes the old file)
            document['numeric_attempts'] = self.numeric_attempts
        run_io.write_json_atomic(self.path, document)


def check_cuda(constants: RunnerConstants) -> None:
    """``GateRefused('environment')`` (exit 2) for a CUDA run on a machine without CUDA; the runner never substitutes another device."""
    if constants.device == 'cuda' and not torch.cuda.is_available():
        raise guards.GateRefused('environment', 'CUDA is not available, and the runner never substitutes another device')


def _termination(head: dict) -> dict:
    """The end of a finished training, from its head: ``reason`` is 'early_stopping', 'budget' (the head key ``budget_end``: the wall cap ended it, a success under
    ``on_wall_cap='best_stop'``) or 'max_epochs'. ``RuntimeError`` for a head that says done and gives no reason."""
    budget = 'budget_end' in head
    if head['stopped_early']:
        reason = 'early_stopping'
    elif budget:
        reason = 'budget'
    elif head['epoch'] == head['epochs']:
        reason = 'max_epochs'
    else:
        raise RuntimeError('the final head is done but names no end (not stopped early, no budget end, not at the horizon)')
    return {'reason': reason, 'done': bool(head['done']), 'epochs_run': int(head['epoch']), 'best_epoch': int(head['best_epoch']),
            'stopped_early': bool(head['stopped_early']), 'budget_end': budget}


def train(profile, run_dir, *, inputs: Inputs, constants: RunnerConstants = RunnerConstants(), probe=None, verifier=None) -> dict:
    """The training stage: build the ``TrainConfig`` and the data from the profile, run ``ept.train.train`` into ``run_dir/train/ckpt`` with the identity check
    (``verifier``) and E6 (``probe``) before every epoch (``EpochCheck``), and write ``train/RECORD.json``; return the record. The CUDA check comes first: a CUDA
    run on a machine without CUDA is refused before any row or file is read. The numeric policy comes next (``numeric.apply``, decision D10 P6): the float32 settings are set,
    read back and written to the record as ``numeric``, and a state that is not the policy is a refusal (exit code 2) before any row is read or model built. It changes
    process-wide settings, as ``ept.train.deterministic_cpu`` does; the caller of a test gives them back. ``verifier`` None means that no identity check is made (the record
    of each epoch says 'not_checked'); the stage body ``train_stage`` always gives one.

    Needs ``stats/STATS.json`` of the statistics stage (``IORefused`` if it is missing) and the loss-scale file of ``inputs`` (its condition must be the
    training's, checked here again with the representatives the rows were really made of). A finished training whose head says done is a success whatever its
    end is, 'budget' included; a head that is not done is a ``RuntimeError`` (3). The record holds the effective values of the trainer's own identity with the old
    defaults written out, the weight-decay assignment of the head, the representatives, the statistics, the scales, the end, the training seconds against the
    24 h cap, and the sha256 of every file of the checkpoint folder.

    ``ValueError`` of the trainer for statistics or rows made under other settings than the profile's (it checks both ways before any work).
    """
    run_dir = Path(run_dir)
    check_cuda(constants)
    numeric = run_numeric.apply(constants.device)              # D10 P6 (K6c): the float32 policy is held before any row is read or any model is built (NumericRefused: exit code 2)
    watcher = EpochCheck(run_dir / E6_FILE, probe, verifier, numeric=numeric)    # m1: a resume under another numeric state than the first process had is refused here (NumericRefused), before any row is read
    config = train_config(profile)
    stats_path = run_dir / STATS_FILE
    if not stats_path.is_file():
        raise run_io.IORefused('the statistics file of the statistics stage is missing')
    stats = data.Stats.load(stats_path)
    train_data = make_train_data(profile, inputs, stats, constants)
    rows = train_data.fit.rows.head_rows
    reps = _reps_record(rows)
    scales_record = _scales_record(profile, inputs, None if reps is None else reps['sha256'])
    out_dir = run_dir / CHECKPOINT_DIR
    # Ref: c_ept/workflows/run_training.py (line 137): ``train.deterministic_cpu(threads=8, seed=args.seed)`` before the call. On the CPU path ``ept.train.train``
    # fixes the threads to ``ept_train.THREADS`` (8) itself, whatever ``constants.threads`` is; on the CUDA path the number set here stays.
    ept_train.deterministic_cpu(threads=constants.threads, seed=profile['seed'])
    try:
        ept_train.train(config, train_data, out_dir, device=constants.device, check=watcher, **route(profile, TRAIN))
    except BaseException:                                              # an error, a refusal, a KeyboardInterrupt: the producers stop, and what is leaving is never hidden (SPD item 2)
        prefetch.close_all(train_data.fit, train_data.stop, quiet=True)
        raise
    prefetch.close_all(train_data.fit, train_data.stop)                  # a finished training has used up every epoch it started: nothing is left, and a producer that is, is an error
    latest = checkpoints.latest_metadata(out_dir)
    if latest is None:
        raise RuntimeError('the training returned without a committed checkpoint')
    folder, head, history = latest
    if head.get('done') is not True:
        raise RuntimeError('the final head of the training is not done')
    termination = _termination(head)
    if history is not None and history.get('termination_reason') != termination['reason']:
        raise RuntimeError('the end named by the head differs from the epoch-metrics record')
    identity = head['identity']
    explicit = {name: identity.get(name, default) for name, default in IDENTITY_DEFAULTS.items()}
    if explicit != {name: profile[name] for name in IDENTITY_DEFAULTS}:
        raise RuntimeError('the identity of the training differs from the settings of the profile')
    seconds = float(head['seconds'])
    cap = ept_train.WALL_CAP_SECONDS
    files = {path.relative_to(run_dir).as_posix(): run_io.sha_file(path) for path in sorted(out_dir.rglob('*')) if path.is_file()}
    record = {'schema': TRAIN_RECORD_SCHEMA, 'profile_sha256': profile.sha256, 'settings': effective_settings(profile), 'runner': constants.to_record(),
              'numeric': numeric,                                    # the policy that was asked for and the state of the process that was read back after it was set (D10 P6)
              'numeric_attempts': [{'attempt': a['attempt'], 'measured_sha256': a['measured_sha256']} for a in watcher.numeric_attempts],   # m1: one entry per process of the run (the states are in the E6 file)
              'effective_threads': int(head['threads']),            # what the trainer ran with: on the CPU it is ``ept_train.THREADS`` whatever ``constants.threads`` says
              'config': {'encoder': dataclasses.asdict(config.encoder), 'lr': config.lr, 'weight_decay': config.weight_decay, 'warmup_steps': config.warmup_steps,
                         'patience': config.patience},
              'identity': explicit,
              'decay_groups': {'policy': explicit['decay_groups'], 'rule_sha256': identity.get('decay_groups_sha256'), 'head': head.get('decay_groups')},
              'k_reps': reps, 'stats': {'sha256': stats.sha256, 'dropped_store_sha256': runlog.sha256_json(list(stats.dropped_store))}, 'scales': scales_record,
              'termination': termination, 'time': {'training_seconds': seconds, 'cap_seconds': cap, 'over_cap': seconds > cap},
              'checkpoint': {'final': folder.relative_to(run_dir).as_posix(), 'head_sha256': files[f'{folder.relative_to(run_dir).as_posix()}/head.json'],
                             'best': head['best'], 'files': files},
              'e6': {'file': E6_FILE, 'checks': len(watcher.checks)}, 'prefetch': prefetch_record()}
    run_io.write_json_atomic(run_dir / TRAIN_RECORD_FILE, record)
    return record


# ---- the stages of the supervisor ----------------------------------------------------------------------------------------------------------
STATS_STAGE, TRAIN_STAGE = 'stats', 'train'
LOADER = 'c_ept.run.runners:load_inputs'
BODIES = MappingProxyType({STATS_STAGE: 'c_ept.run.runners:stats_stage', TRAIN_STAGE: 'c_ept.run.runners:train_stage'})
RUNNER_MODULES = ('c_ept.run.supervisor', 'c_ept.run.runners')              # the entry modules of the code identity: the supervisor and everything the stage bodies import
_LOADER_NAME = re.compile(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*')
_NUMBER_TEXT = re.compile(r'[0-9]{1,9}')


def _declared(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
        raise run_io.IORefused(f'cap allowance {name} must be a finite number of seconds, zero or more')
    return float(value)


@dataclasses.dataclass(frozen=True)
class CapAllowance:
    """What the wall-clock cap of the training stage holds beyond the 24 h of the trainer under ``on_wall_cap='best_stop'`` (the M8 hand-off): ``epoch_s`` the
    declared bound of the longest epoch (the trainer ends the epoch that reaches the cap, it does not cut it), ``checkpoint_write_s`` the bound of one checkpoint
    write (the epoch's own and the final one: two writes) and ``finalisation_s`` the rest: the hashes of the checkpoint files and the record, and the runner's
    preparation that comes before the trainer's clock starts (the child's start, the loader, the whole-role rows and batches). All three are
    declared by the caller from the measurement P6 or the first run, never defaulted here: a number that this module made up would be a claim about a machine.

    해석: 'best_stop' 학습은 24 h에 닿은 에폭을 끝까지 마치므로 24 h를 한 에폭과 체크포인트 쓰기 두 번, 마무리만큼 넘을 수 있다. 감독자의 여유 1 h로는 모자랄 수 있어
    학습 단계의 상한 자체에 그 몫을 더해 선언한다.
    """

    epoch_s: float
    checkpoint_write_s: float
    finalisation_s: float

    def __post_init__(self):
        for name in ('epoch_s', 'checkpoint_write_s', 'finalisation_s'):
            object.__setattr__(self, name, _declared(name, getattr(self, name)))

    def total(self) -> float:
        return self.epoch_s + 2 * self.checkpoint_write_s + self.finalisation_s

    def to_record(self) -> dict:
        return {'epoch_s': self.epoch_s, 'checkpoint_write_s': self.checkpoint_write_s, 'finalisation_s': self.finalisation_s}


def training_stage_cap(profile, allowance: CapAllowance) -> float:
    """The ``wall_cap_s`` of the training stage: ``T_cap + E + 2 W + F`` under ``on_wall_cap='best_stop'``, the trainer's own 24 h ``T_cap`` under ``'fail'`` (which
    refuses at the cap instead of finishing an epoch). The supervisor adds its finalisation margin on top (it refuses to begin a stage whose cap and margin the
    budget that is left cannot hold)."""
    cap = float(ept_train.WALL_CAP_SECONDS)
    return cap + allowance.total() if profile['on_wall_cap'] == 'best_stop' else cap


def evaluate(profile, run_dir, *, inputs, **kwargs) -> dict:
    """The evaluation runner of the plan (K6b): ``runners.evaluate(profile, run_dir, inputs=...)`` is ``c_ept.run.evaluate.evaluate`` (see there for the arguments, the roles
    and the records). The implementation lives in its own module because the evaluation (the checkpoint identities, STOP, C4, the access ledger, the TEST gate) is as large as
    this training runner; the import is made when the function is called, so that the training runner does not import the evaluation modules."""
    from c_ept.run import evaluate as run_evaluate
    return run_evaluate.evaluate(profile, run_dir, inputs=inputs, **kwargs)


def measure(profile, run_dir, *, profile_path, **kwargs) -> int:
    """The measurement runner of the plan (K6c, P6): ``runners.measure(profile, run_dir, profile_path=..., ...)`` is ``c_ept.run.measure.measure`` (see there for the arguments, the stages and the
    records). ``profile`` is the loaded measure profile and ``profile_path`` the file that it was loaded from: the children load that file and check its lock. The measurement runs ``repeat``
    short training calls, each in a fresh child process under the supervisor, and compares them bit for bit. The implementation lives in its own module; the import is made when the function is
    called, so that the training runner does not import the measurement modules."""
    from c_ept.run import measure as run_measure
    return run_measure.measure(profile, run_dir, profile_path=profile_path, **kwargs)


def _absolute_path(path) -> Path:
    return Path(os.path.abspath(path))


def stage_arguments(profile, profile_path, manifest_path, loader: str, constants: RunnerConstants) -> tuple:
    """The text arguments of both stage bodies: the profile file and its lock, the inputs manifest, the loader (``module:function``), the device, the threads
    and the batch size (they travel on the command line of the child)."""
    return (str(_absolute_path(profile_path)), profile.sha256, str(_absolute_path(manifest_path)), loader, constants.device, str(constants.threads),
            str(constants.batch_size))


def plan(profile, profile_path, manifest_path, *, constants: RunnerConstants, allowance: CapAllowance, loader: str = LOADER, bodies=None, manifest=None) -> list:
    """The stages of a training run for the supervisor: ``stats`` (the statistics, their record, the D7 list) and ``train`` (the training, its record, the E6
    checks), each ``module:function(run_dir, *texts)``.

    The inputs of a stage are what its ledger hashes before it begins and after it ends: the profile file, the inputs manifest, the files that the manifest binds
    (``manifest_files``: the manifests of the stores, the scales, the terrain), and, for the training, the two files of the statistics stage. The outputs
    are fixed files (the stage writes the list of its checkpoint files into its record). The training stage carries ``wall_cap_s`` = ``training_stage_cap``.
    ``bodies`` (None or a mapping ``{'stats': ..., 'train': ...}`` of ``module:function`` texts) replaces a default body (tests wrap the real body).
    ``manifest`` is the manifest dict that the caller has read already (``start`` reads the manifest once); None: it is read here.
    """
    chosen = {**BODIES, **(bodies or {})}
    profile_file, manifest_file = _absolute_path(profile_path), _absolute_path(manifest_path)
    bound = manifest_files(manifest if manifest is not None else read_manifest(manifest_file))
    common = [profile_file.as_posix(), manifest_file.as_posix(), *bound]
    arguments = stage_arguments(profile, profile_file, manifest_file, loader, constants)
    return [supervisor.Stage(STATS_STAGE, chosen[STATS_STAGE], args=arguments, inputs=common, outputs=(STATS_FILE, STATS_RECORD_FILE)),
            supervisor.Stage(TRAIN_STAGE, chosen[TRAIN_STAGE], args=arguments, inputs=[STATS_FILE, STATS_RECORD_FILE, *common],
                             outputs=(TRAIN_RECORD_FILE, E6_FILE), wall_cap_s=training_stage_cap(profile, allowance))]


def _whole_text(name: str, text, low: int) -> int:
    if not isinstance(text, str) or _NUMBER_TEXT.fullmatch(text) is None or int(text) < low:
        raise run_io.IORefused(f'stage argument {name} must be a whole number of at least {low}')
    return int(text)


def _loader(spec):
    if not isinstance(spec, str) or _LOADER_NAME.fullmatch(spec) is None:
        raise run_io.IORefused('the inputs loader must be named module.path:function')
    module_name, _, function = spec.partition(':')
    try:
        return getattr(importlib.import_module(module_name), function)
    except (ImportError, AttributeError):
        raise InputsRefused('the inputs loader cannot be imported') from None


def _open_stage(profile_path, profile_sha256, manifest_path, loader, device, threads, batch_size) -> tuple:
    """The profile (its lock checked), the constants and the inputs of a stage body, from the seven text arguments; ``IORefused`` for any of them that is not
    acceptable, before anything is read from the stores or written."""
    profile = run_profile.load(profile_path)
    if not isinstance(profile_sha256, str) or profile.sha256 != profile_sha256:
        raise run_io.IORefused('the profile file is not the one the run was started with (its sha256 differs)')
    if profile.kind != 'train':
        raise run_io.IORefused(f'the runner takes a train profile, not a {profile.kind} profile')
    constants = RunnerConstants(batch_size=_whole_text('batch_size', batch_size, 1), device=device, threads=_whole_text('threads', threads, 1))
    opened = _loader(loader)(manifest_path)
    if not isinstance(opened, Inputs):
        raise InputsRefused('the inputs loader did not return Inputs')
    return profile, constants, opened


def stats_stage(run_dir, profile_path, profile_sha256, manifest_path, loader, device, threads, batch_size) -> None:
    """Stage body ``stats``: ``write_stats`` for the arguments of ``stage_arguments``."""
    profile, constants, opened = _open_stage(profile_path, profile_sha256, manifest_path, loader, device, threads, batch_size)
    write_stats(profile, run_dir, inputs=opened, constants=constants)


def train_stage(run_dir, profile_path, profile_sha256, manifest_path, loader, device, threads, batch_size) -> None:
    """Stage body ``train``: ``train`` for the arguments of ``stage_arguments``. A finished training is a success whatever its end is (early stopping, the
    horizon, or the wall cap under ``best_stop``); a refusal of the trainer, of E6 or of the per-epoch identity check is exit code 2 of the child, every other
    error 3 (``supervisor.child_main``). The saved identity is read from the ledger header (read-only) and re-verified before every epoch."""
    profile, constants, opened = _open_stage(profile_path, profile_sha256, manifest_path, loader, device, threads, batch_size)
    saved = run_ledger.Ledger(run_dir).header.identity                      # read-only: the identity that the supervisor bound when it made the ledger
    train(profile, run_dir, inputs=opened, constants=constants, verifier=lambda: run_identity.verify(saved))


def approval_record(text) -> dict | None:
    """The text of the author's approval of a run as the ledger header holds it (K6c): ``{'text': text, 'sha256': the SHA-256 of its UTF-8 bytes}``, or None when no text is given. The
    text is recorded exactly as it was given (not stripped, not normalised); the hash lets a reader check that. ``IORefused`` (exit code 2) for a value that is not a text with a visible
    character (an empty or blank approval approves nothing) or that cannot be written as UTF-8.

    해석: 승인 문구는 저자가 준 글자 그대로 원장 머리글에 남기고, 그 UTF-8 바이트의 해시를 곁들인다. 글자가 보이지 않는 문구는 승인이 아니므로 거부한다."""
    if text is None:
        return None
    if not isinstance(text, str) or not text.strip():
        raise run_io.IORefused('the approval must be a text with at least one visible character (it is recorded as given)')
    try:
        digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
    except UnicodeEncodeError:
        raise run_io.IORefused('the approval text cannot be written as UTF-8') from None
    return {'text': text, 'sha256': digest}


def runner_record(profile, manifest: Mapping, manifest_sha256: str, constants: RunnerConstants, allowance: CapAllowance, scales_condition: dict, approval=None) -> dict:
    """The runner constants and the effective values for ``Header.runner`` (written once, when the ledger is made): the batch size, the device and the
    threads, every setting of the profile written out, the cap of the training stage and its allowance, the inputs manifest (its hash is that of the bytes
    that ``manifest`` was read from) with the files it binds, the loss-scale file with the condition it was made under, the numeric policy that the run asks for
    (``numeric.requested_record``: the stage record holds the state that was read back) and the text of the approval (``approval_record``: None when none was given)."""
    return {**constants.to_record(), 'prefetch': prefetch_record(), 'settings': effective_settings(profile),
            'training_stage': {'wall_cap_s': training_stage_cap(profile, allowance), 'trainer_cap_s': float(ept_train.WALL_CAP_SECONDS), 'allowance': allowance.to_record()},
            'inputs': {'manifest_sha256': manifest_sha256, 'bound': manifest_files(manifest)},
            'scales': {'file_sha256': run_io.sha_file(manifest['scales']), 'condition': scales_condition},
            'numeric': run_numeric.requested_record(), 'approval': approval_record(approval)}


def _start(profile_path, run_dir, *, manifest_path, spec_path, budget, allowance, constants, scrub_keys, loader, bodies, stream, approval, options) -> int:
    """Everything that can be refused before a run folder exists, cheaply: the kind of the profile, the approval text, CUDA, the manifest (read once), the stores (complete
    manifests), the terrain against the build of the S2 views (when the manifest names one) and the condition of the scales file."""
    profile = run_profile.load(profile_path)
    if profile.kind != 'train':
        raise run_io.IORefused(f'the runner takes a train profile, not a {profile.kind} profile')
    approval_record(approval)
    check_cuda(constants)
    manifest, manifest_sha256 = read_manifest_with_sha(manifest_path)
    condition = check_scales_condition(profile, manifest['scales'])            # the early check: the representatives of the rows are compared in the stats stage
    _, s2, _ = open_stores(manifest)
    if manifest['terrain'] is not None:
        check_terrain_build(s2.manifest, terrain_hashes(manifest['terrain']))
    stages = plan(profile, profile_path, manifest_path, constants=constants, allowance=allowance, loader=loader, bodies=bodies, manifest=manifest)
    return supervisor.run(profile, run_dir, stages=stages, spec_path=spec_path, budget=budget, cuda_visible_devices=constants.cuda_visible_devices,
                          threads=constants.threads, identity_modules=RUNNER_MODULES,
                          runner=runner_record(profile, manifest, manifest_sha256, constants, allowance, condition, approval),
                          scrub_keys=scrub_keys, stream=stream, training_cap_s=training_stage_cap(profile, allowance), **options)


def start(profile_path, run_dir, *, manifest_path, spec_path, budget, allowance: CapAllowance, constants: RunnerConstants = RunnerConstants(), scrub_keys=(),
          loader: str = LOADER, bodies=None, stream=None, approval=None, **options) -> int:
    """Start (or resume) the training run of the train profile at ``profile_path`` in ``run_dir`` and return the exit code of the process (0, 2 or 3: the
    table of ``supervisor``). Everything that is wrong with the profile, the manifest or the scales before the supervisor starts is a printed refusal (2)
    with the match keys of ``scrub_keys`` removed, and leaves no run folder. ``approval``: the text of the author's approval, recorded as given in the ledger header
    (``approval_record``; None records null). ``options`` go to ``supervisor.run`` (``e6``, ``capture``, ``verify``, ``start_gate``, ``poll_s``, ``margin_s``, ``repo``,
    ``environ``).

    The cap of the run for the supervisor's budget rule is the training stage's cap (``training_stage_cap``): the wall budget must hold it plus the
    supervisor's finalisation margin, so the wall cap of the trainer acts before the budget does.

    Importing this module imports numpy and torch. A process that has to apply the native-thread policy first (the command line of K6c) imports
    ``c_ept.run.guards``, calls ``native_policy()`` and only then imports this module; ``supervisor.run`` calls it again as its first step.
    """
    return supervisor.main_guard(lambda: _start(profile_path, run_dir, manifest_path=manifest_path, spec_path=spec_path, budget=budget, allowance=allowance,
                                                constants=constants, scrub_keys=scrub_keys, loader=loader, bodies=bodies, stream=stream, approval=approval,
                                                options=options),
                                 scrub_keys=scrub_keys, stream=stream)
