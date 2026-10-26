"""Tests of the declarative, hash-locked run profiles of the run kernel (``c_ept.run.profile``, S4b-prep K3, decision D11).

A profile is one JSON file that says what a run is: ``train`` (a training run), ``evaluate`` (an evaluation of trained models on a role of
the split) or ``measure`` (the P6 bit-identity measurement, which names a train profile by relative path and SHA-256). The loader reads it
through the strict read-once reader ``c_ept.run.io.read_json_with_sha``, checks every field, fills the omitted fields with their defaults and returns a frozen
``Profile`` that carries the SHA-256 of the file bytes. These tests pin: the refusal of every unknown field, wrong type and value outside its
allowed set or range; the defaults (they are the EPT-T002 values, so the bare ``{"kind": "train"}`` is the T002 configuration); the exact
field-name set and allowed texts of each kind (the contract with Track M and with the runners of K6); the hash lock; the reference of a
measure profile; the round trip of a written profile; and ``to_train_kwargs`` against the real ``TrainConfig`` source. They use synthetic
files in ``tmp_path`` only; nothing here imports model code.

해석: 프로파일은 실행 한 번을 해시로 잠그는 설정 파일이다. 생략한 필드는 EPT-T002의 옛 값으로 채워지므로 ``{"kind": "train"}`` 한 줄이 T002
설정이다. 모르는 필드, 잘못된 형, 허용 집합이나 범위 밖의 값은 필드 이름을 대며 거부하고 값은 메시지에 싣지 않는다. 필드 이름 집합과 허용 문자열은
계약 시험이 글자 그대로 고정하므로, M 갈래에서 이름을 바꾸면 이 시험을 일부러 고쳐야 한다. measure 프로파일은 train 프로파일을 상대 경로와 해시로
가리키고, 가리킨 파일이 바뀌면 거부한다.
"""
import ast
import copy
import dataclasses
import hashlib
import inspect
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

from c_ept.run import io as run_io
from c_ept.run import profile as run_profile
from c_ept.tests.test_run_io import watch_opens           # K5a: counts the reads of one file (and can change it right after a read)

REPO = Path(__file__).resolve().parents[2]
LEAK = 'ZZ-LEAK-4417'          # a marker that must never be echoed by a refusal

TRAIN_FIELDS = {'size', 'fraction', 'seed', 'epochs', 'patience', 'lr', 'weight_decay', 'warmup',
                'kill_law', 'k_condition', 'decay_groups', 'terrain_smooth_weight', 'swap_mode', 'eval_average', 'on_wall_cap',
                'zone_mark', 'field_channels', 'column_drop', 'stats_population', 'selection_seed'}
EVALUATE_FIELDS = {'role', 'models', 'exposure', 'winner_truth', 'chunk_views', 'b0', 'strata', 'price_sensitivity', 'b1_swap_average',
                   'bootstrap_B', 'checkpoints', 'holdout_lock_sha256', 'composed'}
MEASURE_FIELDS = {'train_profile', 'train_sha256', 'steps', 'repeat'}
TRAIN_KEYS = {'kill_law', 'k_condition', 'decay_groups', 'column_drop', 'stats_population', 'terrain_smooth_weight', 'swap_mode', 'eval_average',
              'on_wall_cap', 'zone_mark', 'lr', 'weight_decay', 'warmup_steps', 'patience'}
# Track M fields of ``TrainConfig`` that a later task of Track M has not added yet (M7 added swap_mode and eval_average, M8 added on_wall_cap and M9 added zone_mark, and each removed its names from
# this set). The profile routes them all (they are in ``TRAIN_KEYS``). The X1 check below asserts that each name in this set is NOT yet a field, so the task that adds one must remove its name here
# (and the test fails if it forgets). Integration decision D25. It is empty since M9 landed: the X1 check binds fully and ``TrainConfig(encoder, **profile.to_train_kwargs())`` builds.
PENDING_TRACK_M = frozenset()
# Track M fields of ``TrainConfig`` that the runner sets itself and the profile does not carry. The code shows none today (every field after
# ``model_spec`` is a profile field), so it is empty; a name here must be a real field that is not in ``TRAIN_KEYS``.
RUNNER_ONLY = frozenset()

# The EPT-T002 values that an omitted train field must reproduce (training_operator.FIXED, the TrainConfig defaults, the old behaviour
# of every Track M switch), written out once more here so that a change of a default is a deliberate edit of this test.
TRAIN_DEFAULTS = {
    'size': 'S', 'fraction': 0.25, 'seed': 20261022, 'epochs': 30, 'patience': 2, 'lr': 0.001, 'weight_decay': 0.01, 'warmup': 10,
    'kill_law': 'poisson', 'k_condition': 'raw', 'decay_groups': 'all', 'terrain_smooth_weight': 0.0, 'swap_mode': 'average',
    'eval_average': True, 'on_wall_cap': 'fail', 'zone_mark': True, 'field_channels': 8, 'column_drop': 'none',
    'stats_population': 'whole_fit', 'selection_seed': None,
}
EVALUATE_DEFAULTS = {'exposure': 'observed', 'winner_truth': 'gold', 'chunk_views': None, 'b0': False, 'strata': False,
                     'price_sensitivity': False, 'b1_swap_average': False, 'bootstrap_B': 2000, 'checkpoints': {},
                     'holdout_lock_sha256': None, 'composed': None}
MEASURE_DEFAULTS = {'repeat': 2}
CHOICES = {                                                  # the allowed texts of every field that has a set
    'train': {'size': ('S', 'M'), 'kill_law': ('poisson', 'shifted'), 'k_condition': ('raw', 'binned'), 'decay_groups': ('all', 'split'),
              'swap_mode': ('average', 'random_view', 'none'), 'on_wall_cap': ('fail', 'best_stop'),
              'column_drop': ('none', 'store', 'store_and_subset'), 'stats_population': ('whole_fit', 'selected_fit')},
    'evaluate': {'role': ('STOP', 'DEV', 'TEST'), 'exposure': ('observed', 'nominal'), 'winner_truth': ('gold', 'binned')},
    'measure': {},
}


def write(tmp_path: Path, obj: dict, name: str = 'profile.json') -> Path:
    path = tmp_path / name
    run_io.write_json_atomic(path, obj)
    return path


def base(tmp_path: Path, kind: str) -> dict:
    """The smallest valid file of ``kind``. A measure profile needs the train profile that it names: ``train.json`` is written next to it."""
    if kind == 'measure':
        train = write(tmp_path, {'kind': 'train'}, 'train.json')
        return {'kind': 'measure', 'train_profile': 'train.json', 'train_sha256': run_io.sha_file(train), 'steps': 3}
    return {'train': {'kind': 'train'}, 'evaluate': {'kind': 'evaluate', 'role': 'DEV', 'models': ['ept']}}[kind]


def make(tmp_path: Path, kind: str, **fields) -> run_profile.Profile:
    return run_profile.load(write(tmp_path, {**base(tmp_path, kind), **fields}))


def refused(tmp_path: Path, kind: str, **fields) -> str:
    """The message of the refusal of the base file of ``kind`` updated with ``fields`` (the test fails if there is none)."""
    with pytest.raises(run_profile.ProfileRefused) as info:
        run_profile.load(write(tmp_path, {**base(tmp_path, kind), **fields}))
    return str(info.value)


# ---- the interface -----------------------------------------------------------------------------------------------------------------
def test_the_interface_has_exactly_the_planned_names():
    assert list(inspect.signature(run_profile.load).parameters) == ['path']
    assert run_profile.KINDS == ('train', 'evaluate', 'measure')
    assert dataclasses.is_dataclass(run_profile.Profile) and run_profile.Profile.__dataclass_params__.frozen
    assert [f.name for f in dataclasses.fields(run_profile.Profile)] == ['kind', 'sha256', 'settings', 'train']
    assert issubclass(run_profile.ProfileRefused, run_io.IORefused) and issubclass(run_profile.ProfileRefused, ValueError)
    for removed in ('TRAIN_GROUPS', 'to_data_kwargs', 'to_encoder_kwargs'):          # the one-consumer partition was a false contract (I2)
        assert not hasattr(run_profile, removed) and not hasattr(run_profile.Profile, removed), removed


def test_the_module_imports_only_the_standard_library_and_run_io():
    """No Track M code, no numpy, no torch: a profile is read before any model code is loaded."""
    tree = ast.parse(Path(run_profile.__file__).read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(a.name.split('.')[0] in sys.stdlib_module_names for a in node.names), [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, 'a relative import'
            module = node.module or ''
            assert module.split('.')[0] in sys.stdlib_module_names or (module == 'c_ept.run' and [a.name for a in node.names] == ['io']), module


# ---- the contract ------------------------------------------------------------------------------------------------------------------
def test_the_contract_pins_the_exact_field_names_of_each_kind():
    """A rename in Track M (``TrainConfig``) must change this test and ``profile.FIELDS`` together, on purpose."""
    assert {kind: set(names) for kind, names in run_profile.FIELDS.items()} == \
        {'train': TRAIN_FIELDS, 'evaluate': EVALUATE_FIELDS, 'measure': MEASURE_FIELDS}
    assert all(len(names) == len(set(names)) for names in run_profile.FIELDS.values())


def test_the_contract_pins_the_allowed_values():
    """The texts that a field accepts are part of the contract too (``swap_mode`` ``random_view`` is the Track M value of M7, and so on)."""
    assert {kind: {name: tuple(allowed) for name, allowed in fields.items()} for kind, fields in run_profile.CHOICES.items()} == CHOICES


@pytest.mark.parametrize('kind', run_profile.KINDS)
def test_every_allowed_text_is_accepted_and_its_near_misses_are_not(tmp_path, kind):
    for name, allowed in CHOICES[kind].items():
        for value in allowed:
            extra = {'holdout_lock_sha256': 'ab' * 32} if (kind, name, value) == ('evaluate', 'role', 'TEST') else {}          # a TEST profile fixes the S5 lock (D28)
            assert make(tmp_path, kind, **{name: value}, **extra).settings[name] == value
            for near in (value.upper() if value != value.upper() else value.lower(), value + 'x', value[:-1], ' ' + value, value + ' '):
                assert near not in allowed
                assert f"'{name}'" in refused(tmp_path, kind, **{name: near}), (name, near)


def test_the_train_keys_are_the_names_that_trainconfig_takes():
    """The plan's Track M interfaces name these TrainConfig fields (M1, M3, M4, M5, M6, M7, M8, M9) next to the existing lr, weight_decay,
    warmup_steps and patience. ``warmup`` of the profile is ``warmup_steps`` of the TrainConfig; the other names are equal."""
    assert set(run_profile.TRAIN_KEYS) == TRAIN_KEYS and len(run_profile.TRAIN_KEYS) == len(TRAIN_KEYS)
    assert dict(run_profile.TRAIN_RENAMES) == {'warmup': 'warmup_steps'}
    inverse = {config: profile for profile, config in run_profile.TRAIN_RENAMES.items()}
    assert all(inverse.get(key, key) in run_profile.FIELDS['train'] for key in run_profile.TRAIN_KEYS)
    doc = inspect.getdoc(run_profile.Profile.to_train_kwargs)
    assert 'not an exclusive partition' in doc                # the set is what TrainConfig takes; data preparation reads more fields
    for name in ('kill_law', 'k_condition', 'swap_mode', 'seed', 'selection_seed'):
        assert name in doc, name


# ---- the keys against the real TrainConfig and EncoderConfig (read as source text, like the T002 operator constants) ----------------
_NO_DEFAULT = object()


def class_defaults(relative: str, class_name: str) -> dict:
    """``{field: default}`` of the dataclass ``class_name`` in the source file ``relative`` of this checkout (``_NO_DEFAULT`` if none).

    Read with ``ast``, nothing is imported. A default may be a literal or the name of a module-level literal constant (``patience: int =
    PATIENCE``); anything else fails loudly, so that a new kind of default is looked at instead of skipped.
    """
    tree = ast.parse((REPO / relative).read_text(encoding='utf-8'))
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                constants[node.targets[0].id] = ast.literal_eval(node.value)
            except Exception:                                  # not a literal (an expression, a call): not usable as a default here
                pass
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    out = {}
    for node in cls.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            value = node.value
            out[node.target.id] = _NO_DEFAULT if value is None else constants[value.id] if isinstance(value, ast.Name) else ast.literal_eval(value)
    return out


def test_the_trainconfig_keywords_and_the_patience_default_are_those_of_the_real_trainconfig(tmp_path):
    fields = class_defaults('c_ept/ept/train.py', 'TrainConfig')
    for key in ('lr', 'weight_decay', 'warmup_steps', 'patience'):                      # the keywords that exist today
        assert key in fields, key
    assert fields['patience'] == TRAIN_DEFAULTS['patience'] == make(tmp_path, 'train').patience
    # every key of to_train_kwargs() that TrainConfig defines with a default has the profile default as that default
    kwargs = make(tmp_path, 'train').to_train_kwargs()
    for key, default in fields.items():
        if key in kwargs and default is not _NO_DEFAULT:
            assert kwargs[key] == default and type(kwargs[key]) is type(default), key


def test_the_field_channels_default_is_that_of_the_real_encoderconfig(tmp_path):
    fields = class_defaults('c_ept/ept/encoder.py', 'EncoderConfig')
    assert fields['field_channels'] == TRAIN_DEFAULTS['field_channels'] == make(tmp_path, 'train').field_channels


def routing_problems(fields: dict, keys, kwargs: dict, pending=PENDING_TRACK_M, runner_only=RUNNER_ONLY) -> list:
    """What is wrong with the routing of the profile into ``TrainConfig`` (an empty list when nothing is). Integration decision D25 and the
    review H1 of M5 (the plan, Task X1).

    Symbols: K = ``keys`` (``profile.TRAIN_KEYS``), F = ``fields`` (``class_defaults`` of the real TrainConfig, in definition order, with the
    default of each field), d_k = ``kwargs[k]`` (the profile default of the key k), P = ``pending``, R = ``runner_only``, and
    M = the fields of F that are defined after ``model_spec`` (the Track M fields).

    Rules:  (1) k in K - P  =>  k in F, and F(k) = d_k in value and type when F(k) has a default;     (2) P and F share no name;
            (3) P is inside K;     (4) M is inside K + R;     (5) R is inside F and R and K share no name.
    Rule 1 and 2 are the old X1 check with the pending set made explicit, rule 4 is the reverse check (a Track M field that no profile key
    reaches), and 3 and 5 keep both lists honest.

    해석: 프로파일이 TrainConfig로 보내는 키 집합 K와 TrainConfig의 M 갈래 필드 M은 어긋나면 안 된다. K에 있는데 필드가 아니면
    ``TrainConfig(**kwargs)``가 실패하고, M에 있는데 K에 없으면 그 설정은 프로파일로 켤 수 없어 통계와 설정이 다른 채로 ``_check_inputs``가 모든
    S4b 프로파일을 거부한다(M5 검토 H1). 아직 오지 않은 M7-M9의 이름은 P에 적어 두므로, 그 과제가 필드를 더하면 P에서 지우지 않는 한 시험이 실패한다.
    """
    problems = []
    for key in keys:
        if key in pending:
            if key in fields:
                problems.append(f'{key}: is a TrainConfig field now; remove it from PENDING_TRACK_M')
        elif key not in fields:
            problems.append(f'{key}: is a TRAIN_KEY and not pending, but TrainConfig has no such field')
        elif fields[key] is not _NO_DEFAULT and not (fields[key] == kwargs[key] and type(fields[key]) is type(kwargs[key])):
            problems.append(f'{key}: the TrainConfig default differs from the profile default')
    problems += [f'{name}: is in PENDING_TRACK_M but not a TRAIN_KEY' for name in sorted(pending) if name not in keys]
    if 'model_spec' not in fields:
        return problems + ['model_spec: the anchor of the Track M fields is not a TrainConfig field any more']
    names = list(fields)
    problems += [f'{name}: a Track M field that is neither a TRAIN_KEY nor in RUNNER_ONLY'
                 for name in names[names.index('model_spec') + 1:] if name not in keys and name not in runner_only]
    problems += [f'{name}: is in RUNNER_ONLY but ' + ('is a TRAIN_KEY' if name in keys else 'is not a TrainConfig field')
                 for name in sorted(runner_only) if name in keys or name not in fields]
    return problems


def test_x1_every_train_key_is_a_trainconfig_field_with_the_profile_default(tmp_path):
    """Binds after X1: once Track M (M1: ``c_ept/models/killlaw.py``) is merged, every key of ``to_train_kwargs()`` that is not pending must be a
    field of the real TrainConfig with the profile default as its default (so an omitted profile field is the old TrainConfig); every pending
    key (a field of M7-M9, not merged yet) must not be a field yet; and every Track M field of TrainConfig (every field after ``model_spec``)
    is a key or listed in RUNNER_ONLY, so that no S4b setting is left out of the profile."""
    if not (REPO / 'c_ept' / 'models' / 'killlaw.py').exists():
        pytest.skip('Track M (M1, c_ept/models/killlaw.py) is not merged into this checkout yet; the check binds after X1')
    fields = class_defaults('c_ept/ept/train.py', 'TrainConfig')
    kwargs = make(tmp_path, 'train').to_train_kwargs()
    assert routing_problems(fields, run_profile.TRAIN_KEYS, kwargs) == []


def test_x1_the_reverse_check_looks_at_the_merged_track_m_fields():
    """The reverse check is not vacuous: the fields after ``model_spec`` include the six Track M fields that are merged today (M1, M3, M4, M5
    twice, M6). If ``model_spec`` moved to the end of the class, the slice would be empty and rule 4 would pass without looking at anything."""
    if not (REPO / 'c_ept' / 'models' / 'killlaw.py').exists():
        pytest.skip('Track M (M1, c_ept/models/killlaw.py) is not merged into this checkout yet; the check binds after X1')
    names = list(class_defaults('c_ept/ept/train.py', 'TrainConfig'))
    assert {'kill_law', 'k_condition', 'decay_groups', 'column_drop', 'stats_population', 'terrain_smooth_weight'} <= \
        set(names[names.index('model_spec') + 1:])


def test_x1_the_routing_check_names_each_kind_of_drift():
    """``routing_problems`` on small hand-made field tables: the clean table gives nothing, and each rule gives its own message."""
    no = _NO_DEFAULT
    fields = {'encoder': no, 'lr': no, 'patience': 2, 'model_spec': None, 'kill_law': 'poisson', 'column_drop': 'none'}
    keys = ('kill_law', 'column_drop', 'swap_mode', 'lr', 'patience')
    kwargs = {'kill_law': 'poisson', 'column_drop': 'none', 'swap_mode': 'average', 'lr': 0.001, 'patience': 2}
    pending = frozenset({'swap_mode'})

    def problems(fields=fields, keys=keys, kwargs=kwargs, pending=pending, runner_only=frozenset()):
        return routing_problems(fields, keys, kwargs, pending, runner_only)

    assert problems() == []                                                                      # encoder (before model_spec) needs no key
    assert problems(fields={**fields, 'swap_mode': 'average'}) == [                              # rule 2: a pending field has landed
        'swap_mode: is a TrainConfig field now; remove it from PENDING_TRACK_M']
    assert problems(pending=frozenset()) == ['swap_mode: is a TRAIN_KEY and not pending, but TrainConfig has no such field']   # rule 1
    assert problems(fields={k: v for k, v in fields.items() if k != 'column_drop'}) == [
        'column_drop: is a TRAIN_KEY and not pending, but TrainConfig has no such field']
    assert problems(fields={**fields, 'kill_law': 'shifted'}) == ['kill_law: the TrainConfig default differs from the profile default']
    assert problems(fields={**fields, 'patience': 2.0}) == ['patience: the TrainConfig default differs from the profile default']   # type
    assert problems(pending=frozenset({'swap_mode', 'ghost'})) == ['ghost: is in PENDING_TRACK_M but not a TRAIN_KEY']          # rule 3
    assert problems(fields={**fields, 'zz_new': 0}) == ['zz_new: a Track M field that is neither a TRAIN_KEY nor in RUNNER_ONLY']   # rule 4
    assert problems(fields={**fields, 'zz_new': 0}, runner_only=frozenset({'zz_new'})) == []
    assert problems(runner_only=frozenset({'ghost'})) == ['ghost: is in RUNNER_ONLY but is not a TrainConfig field']             # rule 5
    assert problems(runner_only=frozenset({'kill_law'})) == ['kill_law: is in RUNNER_ONLY but is a TRAIN_KEY']
    assert problems(fields={k: v for k, v in fields.items() if k != 'model_spec'}) == [
        'model_spec: the anchor of the Track M fields is not a TrainConfig field any more']


# ---- the defaults ------------------------------------------------------------------------------------------------------------------
def test_the_bare_train_profile_loads_and_equals_the_t002_values(tmp_path):
    """``{"kind": "train"}`` is valid and is the T002 configuration: the defaults are valid together."""
    path = write(tmp_path, {'kind': 'train'})
    assert path.read_bytes() == b'{\n  "kind": "train"\n}\n'
    profile = run_profile.load(path)
    assert profile.kind == 'train' and profile.train is None
    assert dict(profile.settings) == TRAIN_DEFAULTS
    for name, value in TRAIN_DEFAULTS.items():                # the type is part of the default: 0.001 is a float, 30 an int
        assert type(profile.settings[name]) is type(value), name


def test_train_defaults_equal_the_values_of_the_t002_operator():
    """The same defaults, read as source text from the frozen EPT-T002 operator (no import of it): ``FIXED`` holds the T002 profile."""
    tree = ast.parse((REPO / 'c_ept' / 'workflows' / 'training_operator.py').read_text(encoding='utf-8'))
    fixed = next(ast.literal_eval(node.value) for node in tree.body
                 if isinstance(node, ast.Assign) and [t.id for t in node.targets if isinstance(t, ast.Name)] == ['FIXED'])
    assert fixed['size'] == TRAIN_DEFAULTS['size'] and fixed['max_epochs'] == TRAIN_DEFAULTS['epochs']
    assert fixed['share'] == TRAIN_DEFAULTS['fraction'] and fixed['seed'] == TRAIN_DEFAULTS['seed']
    assert fixed['patience'] == TRAIN_DEFAULTS['patience'] and fixed['lr'] == TRAIN_DEFAULTS['lr']
    assert fixed['weight_decay'] == TRAIN_DEFAULTS['weight_decay'] and fixed['warmup_steps'] == TRAIN_DEFAULTS['warmup']
    assert set(fixed) - {'schema', 'profile', 'size', 'device', 'max_epochs', 'share', 'patience', 'batch_size', 'seed', 'lr', 'weight_decay',
                         'warmup_steps'} == set()         # batch_size and device are the runner constants outside the profile


def test_evaluate_and_measure_defaults(tmp_path):
    evaluate = make(tmp_path, 'evaluate')
    assert dict(evaluate.settings) == {**EVALUATE_DEFAULTS, 'role': 'DEV', 'models': ('ept',)} and evaluate.train is None
    assert evaluate.bootstrap_B == 2000 and type(evaluate.bootstrap_B) is int
    measure = make(tmp_path, 'measure')
    assert dict(measure.settings) == {**MEASURE_DEFAULTS, **{k: v for k, v in base(tmp_path, 'measure').items() if k != 'kind'}}


def test_omitted_defaults_and_explicit_defaults_give_the_same_settings(tmp_path):
    for kind, defaults in (('train', TRAIN_DEFAULTS), ('evaluate', EVALUATE_DEFAULTS), ('measure', MEASURE_DEFAULTS)):
        short = make(tmp_path, kind)
        explicit = run_profile.load(write(tmp_path, {**base(tmp_path, kind), **defaults}, f'{kind}-explicit.json'))
        assert short.settings == explicit.settings
        assert short.sha256 != explicit.sha256            # the lock is on the bytes of the file, not on what they mean


# ---- the selection seed (I1) -------------------------------------------------------------------------------------------------------
def test_membership_is_no_longer_a_field_and_selection_seed_is_the_checked_replacement(tmp_path):
    with pytest.raises(run_profile.ProfileRefused, match='not defined'):
        run_profile.load(write(tmp_path, {'kind': 'train', 'membership': 'share-25'}))
    assert 'selection_seed' in refused(tmp_path, 'train', selection_seed=-1)
    assert make(tmp_path, 'train', fraction=0.5).fraction == 0.5          # a fraction below 1 needs nothing else: the default rule is T002's


def test_membership_seed_follows_the_training_seed_until_a_selection_seed_is_written(tmp_path):
    """null selection_seed = the T002 rule (select with the training seed); an integer = the T009/matched rule (a fixed selection)."""
    for seed in (1, 2, 3, 20261022):
        assert make(tmp_path, 'train', seed=seed).membership_seed() == seed
    assert make(tmp_path, 'train').membership_seed() == 20261022
    for seed in (1, 2, 3, 20261022):
        assert make(tmp_path, 'train', seed=seed, selection_seed=20261022).membership_seed() == 20261022
        assert make(tmp_path, 'train', seed=seed, selection_seed=5).membership_seed() == 5
    assert make(tmp_path, 'train', seed=9, selection_seed=0).membership_seed() == 0          # 0 is a seed, not "unset"
    assert type(make(tmp_path, 'train', selection_seed=7).membership_seed()) is int


def test_membership_seed_belongs_to_train_profiles_only(tmp_path):
    for kind in ('evaluate', 'measure'):
        with pytest.raises(run_profile.ProfileRefused, match='train'):
            make(tmp_path, kind).membership_seed()


# ---- the refusals ------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('kind, extra', [
    ('train', 'epoch'), ('train', 'role'), ('train', 'steps'), ('train', 'batch_size'), ('train', 'membership'), ('train', 'train_profile'),
    ('evaluate', 'epochs'), ('evaluate', 'model'), ('evaluate', 'size'), ('evaluate', 'seed'), ('evaluate', 'selection_seed'),
    ('measure', 'lr'), ('measure', 'epochs'), ('measure', 'step'), ('measure', 'models'), ('measure', 'size'), ('measure', 'seed'),
])
def test_an_unknown_field_is_refused(tmp_path, kind, extra):
    """A field that the kind does not define is refused, even when another kind defines it."""
    with pytest.raises(run_profile.ProfileRefused, match='not defined'):
        run_profile.load(write(tmp_path, {**base(tmp_path, kind), extra: 1}))


def test_an_unknown_field_refusal_suggests_the_known_name_and_never_echoes_the_key(tmp_path):
    message = refused(tmp_path, 'train', epoch=3)
    assert "'epochs'" in message and "'epoch'" not in message
    message = refused(tmp_path, 'train', **{LEAK: 1})
    assert LEAK not in message and message.startswith('1 field(s) not defined')
    two = refused(tmp_path, 'train', **{LEAK: 1, LEAK + 'b': 2})                   # the count is reported instead of the names
    assert LEAK not in two and two.startswith('2 field(s) not defined')


def test_the_kind_is_required_and_must_be_known(tmp_path):
    with pytest.raises(run_profile.ProfileRefused, match="'kind'"):
        run_profile.load(write(tmp_path, {'epochs': 3}))
    for bad in ('trainer', 'TRAIN', '', 3, None, ['train'], True):
        with pytest.raises(run_profile.ProfileRefused, match="'kind'") as info:
            run_profile.load(write(tmp_path, {'kind': bad}, 'bad-kind.json'))
        assert info.value.__context__ is None and info.value.__cause__ is None


BAD_VALUES = [                                     # (kind, field, a value that must be refused)
    # train: the allowed sets
    *[('train', 'size', v) for v in ('L', 's', 'SM', '', 1, True, None, ['S'])],
    *[('train', 'kill_law', v) for v in ('shift', 'Poisson', '', 0, None, ['poisson'])],
    *[('train', 'k_condition', v) for v in ('bin', 'RAW', 1, None)],
    *[('train', 'decay_groups', v) for v in ('some', 'ALL', 1, None)],
    *[('train', 'swap_mode', v) for v in ('both', 'random', 'Average', 1, None)],
    *[('train', 'on_wall_cap', v) for v in ('stop', 'best', 'FAIL', 0, None)],
    *[('train', 'column_drop', v) for v in ('all', 'store_only', '', 1, None)],
    *[('train', 'stats_population', v) for v in ('fit', 'selected', 'whole', 1, None)],
    # train: numbers (a bool is not a number, a float is not a whole number, a text is not a number)
    *[('train', 'fraction', v) for v in (0, 0.0, -0.1, 1.0001, 2, True, False, '0.5', None, [0.5])],
    *[('train', 'seed', v) for v in (-1, 1.5, 2 ** 63, True, '20261022', None, 20261022.0)],
    *[('train', 'selection_seed', v) for v in (-1, 1.5, 2 ** 63, True, False, '5', [5], 5.0)],
    *[('train', 'epochs', v) for v in (0, -3, 1001, 10 ** 12, 1.0, 12.5, True, '12', None)],
    *[('train', 'patience', v) for v in (0, -1, 2.0, True, '2', None)],
    *[('train', 'lr', v) for v in (-0.001, -1, True, '1e-3', None, [0.001])],
    *[('train', 'weight_decay', v) for v in (-0.01, True, '0.01', None)],
    *[('train', 'warmup', v) for v in (-1, 1.5, 10.0, True, '10', None)],
    *[('train', 'terrain_smooth_weight', v) for v in (-1e-9, True, '0', None)],
    *[('train', 'field_channels', v) for v in (-1, 8.0, 1.5, True, '8', None)],
    # train: flags
    *[('train', 'eval_average', v) for v in (1, 0, 'true', 'yes', None, [True])],
    *[('train', 'zone_mark', v) for v in (1, 0, 'false', None)],
    # evaluate
    *[('evaluate', 'role', v) for v in ('dev', 'VAL', 'DEV ', 'CAL', 'FIT', '', 1, None, ['DEV'])],
    *[('evaluate', 'models', v) for v in ([], 'ept', ['ept', 'ept'], [''], [1], ['ept', None], None, {'ept': 1}, 7)],
    *[('evaluate', 'exposure', v) for v in ('nominal ', 'Observed', 'both', 1, None)],
    *[('evaluate', 'winner_truth', v) for v in ('bins', 'Gold', 1, None)],
    *[('evaluate', 'chunk_views', v) for v in (0, -1, 1.5, 256.0, True, '256', [256])],
    *[('evaluate', 'b0', v) for v in (1, 0, 'false', None)],
    *[('evaluate', 'strata', v) for v in (1, 'yes', None)],
    *[('evaluate', 'price_sensitivity', v) for v in (1, 'on', None)],
    *[('evaluate', 'b1_swap_average', v) for v in (1, 'on', None)],
    *[('evaluate', 'bootstrap_B', v) for v in (0, -2000, 100001, 10 ** 12, 2000.0, 1e3, True, '2000', None)],
    *[('evaluate', 'checkpoints', v) for v in ([], 'path', None, {'ept': 1}, {'ept': ''}, {'': 'x.pt'}, {'ept': None}, {'ept': ['x.pt']}, 7)],
    *[('evaluate', 'holdout_lock_sha256', v) for v in ('ab', 'AB' * 32, 'g' * 64, 'a' * 63, 'a' * 65, 'a' * 64 + '\n', 5, True, ['a' * 64])],
    # measure (the reference path forms have their own test)
    *[('measure', 'train_profile', v) for v in (5, None, True, ['train.json'], {'a': 'b'})],
    *[('measure', 'train_sha256', v) for v in ('ab', 'AB' * 32, 'g' * 64, 'a' * 63, 'a' * 65, 'a' * 64 + '\n', 5, None, ['a' * 64])],
    *[('measure', 'steps', v) for v in (0, -1, 100001, 10 ** 12, 2.0, 1.5, True, '3', None)],
    *[('measure', 'repeat', v) for v in (1, 0, -2, 11, 2.0, True, '2', None)],
]


@pytest.mark.parametrize('kind, name, value', BAD_VALUES, ids=[f'{k}-{n}-{v!r}' for k, n, v in BAD_VALUES])
def test_a_wrong_type_or_a_value_outside_its_set_is_refused_naming_the_field(tmp_path, kind, name, value):
    message = refused(tmp_path, kind, **{name: value})
    assert f"'{name}'" in message


BOUNDS = [('train', 'epochs', 1000), ('measure', 'steps', 100000), ('measure', 'repeat', 10), ('evaluate', 'bootstrap_B', 100000)]


@pytest.mark.parametrize('kind, name, top', BOUNDS)
def test_an_upper_bound_is_accepted_and_one_above_it_is_refused(tmp_path, kind, name, top):
    """A typo such as 10^12 must not become a huge allocation later: the bound itself loads, bound + 1 does not."""
    assert make(tmp_path, kind, **{name: top}).settings[name] == top
    assert f"'{name}'" in refused(tmp_path, kind, **{name: top + 1})
    low = 2 if name == 'repeat' else 1
    assert make(tmp_path, kind, **{name: low}).settings[name] == low                                # the lower end loads


# ---- the composed block of an evaluate profile (S4b-prep K6e, decision D30): the V_start pass on a STOP sample ---------------------------------------
COMPOSED_OK = {'sample_matches': 40, 'sample_seed': 11, 'draws': 200, 'mc_batches': 10, 'draw_seed': 12}


def test_the_composed_block_is_off_by_default_and_a_valid_block_is_read_back_with_the_default_progress_rows(tmp_path):
    """K6e: ``composed`` is null (the pass is off) unless the profile writes the block; the block is read back as an immutable mapping with ``progress_rows`` 16 (the block size of the Composer) when it is omitted,
    and the resolved profile is a valid file with the same settings. The sample size, the seeds and the draw counts have no default: they are the parameters of a later cost probe."""
    assert make(tmp_path, 'evaluate').settings['composed'] is None
    block = make(tmp_path, 'evaluate', composed=COMPOSED_OK).settings['composed']
    assert dict(block) == {**COMPOSED_OK, 'progress_rows': 16}
    with pytest.raises(TypeError):
        block['draws'] = 1
    profile = make(tmp_path, 'evaluate', composed=COMPOSED_OK)
    resolved = profile.as_dict()
    assert json.loads(json.dumps(resolved)) == resolved and resolved['composed'] == {**COMPOSED_OK, 'progress_rows': 16}
    assert run_profile.load(write(tmp_path, resolved, 'composed-r.json')).settings == profile.settings
    other = make(tmp_path, 'evaluate', composed={**COMPOSED_OK, 'draws': 400})
    assert other.sha256 != profile.sha256 and other.settings != profile.settings                      # a parameter of the block is part of the lock


@pytest.mark.parametrize('value', [5, 'composed', [], [COMPOSED_OK], True, 1.5, {}], ids=repr)
def test_a_composed_value_that_is_not_an_object_with_the_required_parameters_is_refused(tmp_path, value):
    assert "'composed" in refused(tmp_path, 'evaluate', composed=value)


@pytest.mark.parametrize('missing', sorted(COMPOSED_OK))
def test_every_parameter_of_the_composed_block_but_progress_rows_is_required(tmp_path, missing):
    message = refused(tmp_path, 'evaluate', composed={k: v for k, v in COMPOSED_OK.items() if k != missing})
    assert f"'composed.{missing}'" in message and 'required' in message


COMPOSED_BAD = [('sample_matches', v) for v in (0, -1, 1.0, 40.5, True, '40', None, 1_000_001)] + \
               [('sample_seed', v) for v in (-1, 1.5, 2 ** 63, True, '11', None)] + \
               [('draws', v) for v in (0, 1, 3, 1.0, True, '200', None, 1_000_001)] + \
               [('mc_batches', v) for v in (0, 1, 2.0, True, '10', None, 10_001)] + \
               [('draw_seed', v) for v in (-1, 1.5, 2 ** 63, True, '12', None)] + \
               [('progress_rows', v) for v in (0, -1, 8.0, True, '8', None, 1_000_001)]


@pytest.mark.parametrize('name, value', COMPOSED_BAD, ids=[f'{n}-{v!r}' for n, v in COMPOSED_BAD])
def test_a_composed_parameter_of_the_wrong_type_or_range_is_refused_naming_it(tmp_path, name, value):
    assert f"'composed.{name}'" in refused(tmp_path, 'evaluate', composed={**COMPOSED_OK, 'progress_rows': 8, name: value})


def test_the_composed_draws_must_split_into_whole_batches_of_at_least_two_draws_and_unknown_keys_are_refused_without_echo(tmp_path):
    """The Monte Carlo error is estimated from ``mc_batches`` batches of ``draws / mc_batches`` draws: ``draws`` must be a multiple of it, and a batch holds at least two draws (a quantile of one draw is not a
    batch estimate). A key that the block does not define is refused by count, never echoed (a key may be a path)."""
    for draws, batches in ((200, 7), (10, 10), (10, 3)):
        message = refused(tmp_path, 'evaluate', composed={**COMPOSED_OK, 'draws': draws, 'mc_batches': batches})
        assert "'composed.draws'" in message and 'batches' in message
    assert make(tmp_path, 'evaluate', composed={**COMPOSED_OK, 'draws': 20, 'mc_batches': 10}).composed['draws'] == 20          # two draws per batch is the least
    message = refused(tmp_path, 'evaluate', composed={**COMPOSED_OK, LEAK: 1})
    assert "'composed'" in message and '1 field(s)' in message and LEAK not in message


@pytest.mark.parametrize('name, top', [('sample_matches', 1_000_000), ('draws', 1_000_000), ('mc_batches', 10_000), ('progress_rows', 1_000_000)])
def test_an_upper_bound_of_a_composed_parameter_loads(tmp_path, name, top):
    """The bound itself loads (1,000,000 draws are a multiple of every ``mc_batches`` that is tried); bound + 1 is in ``COMPOSED_BAD``."""
    block = {**COMPOSED_OK, 'draws': 1_000_000, 'mc_batches': 10, name: top}
    assert make(tmp_path, 'evaluate', composed=block).composed[name] == top


@pytest.mark.parametrize('kind, name', [('evaluate', 'role'), ('evaluate', 'models'), ('measure', 'train_profile'), ('measure', 'train_sha256'),
                                        ('measure', 'steps')])
def test_a_required_field_that_is_missing_is_refused_naming_the_field(tmp_path, kind, name):
    fields = {k: v for k, v in base(tmp_path, kind).items() if k != name}
    with pytest.raises(run_profile.ProfileRefused, match=f"'{name}' is required"):          # refused as missing, not by a later check
        run_profile.load(write(tmp_path, fields))


def test_the_role_test_needs_the_hash_of_the_s5_lock_and_the_other_roles_may_leave_it_out(tmp_path):
    """K6b C1 (decision D28): ``holdout_lock_sha256`` is the anchor of the TEST gate. It is required for the role TEST (a TEST profile without it is refused as missing, naming
    the field) and may be null or absent for DEV and STOP (a DEV rehearsal of the TEST profile carries the same value, since it is part of the profile without its role)."""
    digest = 'ab' * 32
    assert make(tmp_path, 'evaluate', role='TEST', holdout_lock_sha256=digest).settings['holdout_lock_sha256'] == digest
    for role in ('DEV', 'STOP'):
        assert make(tmp_path, 'evaluate', role=role).settings['holdout_lock_sha256'] is None
        assert make(tmp_path, 'evaluate', role=role, holdout_lock_sha256=None).settings['holdout_lock_sha256'] is None
        assert make(tmp_path, 'evaluate', role=role, holdout_lock_sha256=digest).settings['holdout_lock_sha256'] == digest
    for fields in ({'role': 'TEST'}, {'role': 'TEST', 'holdout_lock_sha256': None}):
        message = refused(tmp_path, 'evaluate', **fields)
        assert "'holdout_lock_sha256'" in message and 'TEST' in message


def test_a_refusal_never_carries_a_value_or_a_key(tmp_path):
    """Values and keys may be paths, match keys or other things that a log must not hold: the message names the field and the rule only."""
    cases = [('train', {'size': LEAK}), ('train', {'epochs': LEAK}), ('train', {'lr': -4417000}), ('train', {'selection_seed': -4417000}),
             ('train', {'kill_law': [LEAK]}), ('train', {LEAK: LEAK}), ('evaluate', {'role': LEAK}), ('evaluate', {'models': [LEAK, LEAK]}),
             ('evaluate', {'models': [LEAK, 7]}), ('evaluate', {'checkpoints': {LEAK: 7}}), ('evaluate', {'checkpoints': {LEAK: ''}}),
             ('measure', {'steps': LEAK}), ('measure', {'repeat': -4417000}), ('measure', {'train_profile': LEAK + '/x.json'}),
             ('measure', {'train_profile': LEAK + '\\x.json'}), ('measure', {'train_sha256': LEAK}), ('measure', {'train_sha256': 'ab' * 32})]
    for kind, fields in cases:
        with pytest.raises(run_profile.ProfileRefused) as info:
            run_profile.load(write(tmp_path, {**base(tmp_path, kind), **fields}))
        message = str(info.value)
        assert LEAK not in message and '4417000' not in message and 'abab' not in message, message
        assert info.value.__context__ is None and info.value.__cause__ is None


def test_strict_reader_refusals_pass_through_load(tmp_path):
    """The loader reads through ``run_io.read_json_with_sha``: a duplicate key, a non-finite number, a non-object, a link or a missing file is refused."""
    path = tmp_path / 'dup.json'
    path.write_text('{"kind": "train", "epochs": 3, "epochs": 4}', encoding='utf-8')
    with pytest.raises(run_io.IORefused, match='duplicate'):
        run_profile.load(path)
    path.write_text('{"kind": "train", "lr": NaN}', encoding='utf-8')
    with pytest.raises(run_io.IORefused, match='non-finite'):
        run_profile.load(path)
    path.write_text('{"kind": "train", "lr": 1e999}', encoding='utf-8')
    with pytest.raises(run_io.IORefused, match='non-finite'):
        run_profile.load(path)
    path.write_text('["train"]', encoding='utf-8')
    with pytest.raises(run_io.IORefused, match='object'):
        run_profile.load(path)
    with pytest.raises(run_io.IORefused):
        run_profile.load(tmp_path / 'absent.json')


# ---- the normal form of the accepted values ---------------------------------------------------------------------------------------
def test_accepted_values_have_one_type_each(tmp_path):
    profile = make(tmp_path, 'train', fraction=1, lr=1, weight_decay=0, terrain_smooth_weight=2)
    assert [type(profile.settings[n]) for n in ('fraction', 'lr', 'weight_decay', 'terrain_smooth_weight')] == [float] * 4
    assert profile.fraction == 1.0 and profile.lr == 1.0 and profile.weight_decay == 0.0 and profile.terrain_smooth_weight == 2.0
    assert type(profile.seed) is int and type(profile.warmup) is int and profile.warmup == 10
    low = make(tmp_path, 'train', lr=0, warmup=0, field_channels=0, patience=1, epochs=1, seed=0, selection_seed=0)
    assert (low.lr, low.warmup, low.field_channels, low.patience, low.epochs, low.seed, low.selection_seed) == (0.0, 0, 0, 1, 1, 0, 0)
    top = make(tmp_path, 'train', seed=2 ** 63 - 1, selection_seed=2 ** 63 - 1)
    assert top.seed == 2 ** 63 - 1 and top.selection_seed == 2 ** 63 - 1


def test_models_and_checkpoints_are_immutable_in_the_profile(tmp_path):
    profile = make(tmp_path, 'evaluate', models=['b0', 'b1', 'ept'], checkpoints={'ept': 'runs/ept/best.pt', 'n2': 'runs/n2/best.pt'},
                   chunk_views=256)
    assert profile.models == ('b0', 'b1', 'ept')
    assert dict(profile.checkpoints) == {'ept': 'runs/ept/best.pt', 'n2': 'runs/n2/best.pt'}
    with pytest.raises(TypeError):
        profile.checkpoints['ept'] = 'other'
    with pytest.raises(TypeError):
        profile.settings['role'] = 'TEST'
    with pytest.raises(dataclasses.FrozenInstanceError):
        profile.kind = 'train'
    with pytest.raises(dataclasses.FrozenInstanceError):
        profile.sha256 = '0' * 64
    assert profile.chunk_views == 256 and type(profile.chunk_views) is int
    assert hash(profile) == hash(profile) and profile == make(tmp_path, 'evaluate', models=['b0', 'b1', 'ept'],
                                                              checkpoints={'ept': 'runs/ept/best.pt', 'n2': 'runs/n2/best.pt'}, chunk_views=256)


def test_attribute_access_reaches_the_settings_and_nothing_else(tmp_path):
    profile = make(tmp_path, 'train', epochs=12)
    assert profile.epochs == 12 and profile['epochs'] == 12 and profile.settings['epochs'] == 12
    with pytest.raises(AttributeError):
        profile.role                                                  # a field of another kind
    with pytest.raises(AttributeError):
        profile.no_such_field
    with pytest.raises(KeyError):
        profile['role']
    clone = copy.copy(profile)                                       # a half-built copy has no ``settings`` yet: no endless ``__getattr__``
    assert clone == profile and clone.epochs == 12 and clone.settings is profile.settings
    assert not hasattr(object.__new__(run_profile.Profile), 'epochs')


# ---- the hash lock -----------------------------------------------------------------------------------------------------------------
def test_sha256_is_the_hash_of_the_file_bytes(tmp_path):
    path = tmp_path / 'p.json'
    returned = run_io.write_json_atomic(path, {'kind': 'train', 'epochs': 12})
    profile = run_profile.load(path)
    assert profile.sha256 == hashlib.sha256(path.read_bytes()).hexdigest() == returned == run_io.sha_file(path)
    assert len(profile.sha256) == 64 and profile.sha256 == profile.sha256.lower()
    assert run_profile.load(path).sha256 == profile.sha256            # the same file gives the same lock


def test_the_hash_comes_from_the_read_once_reader_of_run_io(tmp_path, monkeypatch):
    """K5a: the lock is the hash that ``run_io.read_json_with_sha`` returns for the bytes it parsed; ``load`` hashes nothing by itself."""
    path = write(tmp_path, {'kind': 'train'})
    real = run_io.read_json_with_sha
    monkeypatch.setattr(run_io, 'read_json_with_sha', lambda target, **kwargs: (real(target, **kwargs)[0], 'ab' * 32))
    assert run_profile.load(path).sha256 == 'ab' * 32
    monkeypatch.undo()

    def no_second_hash(*args, **kwargs):
        raise AssertionError('load hashed the file again')
    monkeypatch.setattr(run_io, 'sha_file', no_second_hash)
    assert run_profile.load(path).sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_one_changed_byte_changes_the_hash(tmp_path):
    path = write(tmp_path, {'kind': 'train', 'epochs': 12})
    first = run_profile.load(path)
    data = bytearray(path.read_bytes())
    at = data.index(b'12')                                            # 12 -> 13: one byte, a different run
    data[at + 1] = ord('3')
    path.write_bytes(bytes(data))
    second = run_profile.load(path)
    assert second.epochs == 13 and second.sha256 != first.sha256
    path.write_bytes(path.read_bytes() + b' ')                       # a trailing space: same run, different bytes, different lock
    third = run_profile.load(path)
    assert third.settings == second.settings and third.sha256 != second.sha256
    assert third.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_a_file_with_a_carriage_return_is_refused(tmp_path):
    """One content, one hash: a CRLF checkout of the same profile would give a second hash, so any carriage return is refused."""
    path = write(tmp_path, {'kind': 'train', 'epochs': 12})
    good = path.read_bytes()
    assert run_profile.load(path).epochs == 12
    for bad in (good.replace(b'\n', b'\r\n'), good + b'\r', good.replace(b'{', b'{\r', 1)):
        path.write_bytes(bad)
        with pytest.raises(run_profile.ProfileRefused, match='carriage return') as info:
            run_profile.load(path)
        assert info.value.__context__ is None
    path.write_bytes(good)
    assert run_profile.load(path).epochs == 12                       # LF only: accepted again


def test_load_reads_the_profile_file_once(tmp_path, monkeypatch):
    """K5a (K3 re-check n3): one open for the object, the hash and the carriage-return check together. The loader used to read a file
    four times (parse, carriage return, hash, parse again) and could not tell that a file had been changed and changed back (A, B, A)."""
    for kind in ('train', 'evaluate'):
        path = write(tmp_path, base(tmp_path, kind), f'{kind}.json')
        opened = watch_opens(monkeypatch, path)
        profile = run_profile.load(path)
        assert opened == ['rb'], (kind, opened)                                       # exactly one read of the file, whatever it is used for
        assert profile.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
        monkeypatch.undo()


def test_a_measure_profile_reads_itself_once_and_the_train_profile_it_names_once(tmp_path, monkeypatch):
    train = write(tmp_path, S4B_LIKE, 'train.json')
    measure = measure_file(tmp_path, 'train.json', run_io.sha_file(train))
    opened_measure = watch_opens(monkeypatch, measure)
    opened_train = watch_opens(monkeypatch, train)                                  # the second spy wraps the first: both files are watched
    loaded = run_profile.load(measure)
    assert opened_measure == ['rb'] and opened_train == ['rb']
    assert loaded.sha256 == hashlib.sha256(measure.read_bytes()).hexdigest()
    assert loaded.train.sha256 == hashlib.sha256(train.read_bytes()).hexdigest() == loaded.train_sha256


def test_the_hash_and_the_settings_come_from_one_read_even_when_the_file_changes_after_it(tmp_path, monkeypatch):
    """A file rewritten right after the read (then rewritten back) cannot give the hash of one content and the settings of another: the
    pair is that of the one read. The old loader's A, B, A window is gone, so it is not refused and not accepted by luck either."""
    path = write(tmp_path, {'kind': 'train', 'epochs': 30})
    first = path.read_bytes()

    def rewrite(number):
        path.write_bytes(first.replace(b'"epochs": 30', b'"epochs": 31') if number == 1 else first)
    opened = watch_opens(monkeypatch, path, after_close=rewrite)
    profile = run_profile.load(path)
    monkeypatch.undo()                                                              # the reads below are the test's own, not the loader's
    assert opened == ['rb']
    assert profile.epochs == 30 and profile.sha256 == hashlib.sha256(first).hexdigest()
    assert path.read_bytes() != first                                               # the file on disk is now another content
    assert profile.sha256 != hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_carriage_return_check_is_made_on_the_bytes_that_were_parsed(tmp_path, monkeypatch):
    """K3 re-check n3: the CR test used to read the file separately. Now a content that is CR-free when read is accepted whatever the file
    becomes afterwards, and a content with a CR when read is refused whatever the file becomes afterwards."""
    path = write(tmp_path, {'kind': 'train', 'epochs': 12})
    lf = path.read_bytes()
    crlf = lf.replace(b'\n', b'\r\n')
    watch_opens(monkeypatch, path, after_close=lambda number: path.write_bytes(crlf))
    assert run_profile.load(path).sha256 == hashlib.sha256(lf).hexdigest()
    monkeypatch.undo()
    path.write_bytes(crlf)
    watch_opens(monkeypatch, path, after_close=lambda number: path.write_bytes(lf))
    with pytest.raises(run_profile.ProfileRefused, match='carriage return'):
        run_profile.load(path)


def test_the_read_once_reader_replaced_the_re_read_check_in_the_docstrings_and_the_readme():
    """The A->B->A residual is gone, so the module docstring, the ``load`` docstring and ``profiles/README.md`` must not describe a re-read
    check (K3 re-check n1, m3) and must say that the file is read once."""
    assert 'A→B→A' not in run_profile.__doc__ and 'Residual' not in run_profile.load.__doc__ and 'A, B, A' not in run_profile.load.__doc__
    assert 'read_json_with_sha' in run_profile.__doc__ and 'read_json_with_sha' in run_profile.load.__doc__
    assert 'once' in run_profile.load.__doc__
    readme = (REPO / 'profiles' / 'README.md').read_text(encoding='utf-8')
    assert 'while it is being read is refused' not in readme and 'changes while it is being read' not in readme
    assert 'is read once' in readme


def test_a_stable_file_is_read_without_complaint_and_through_run_io_read_json_with_sha(tmp_path, monkeypatch):
    path = write(tmp_path, base(tmp_path, 'evaluate'))
    real = run_io.read_json_with_sha
    seen = []
    monkeypatch.setattr(run_io, 'read_json_with_sha', lambda target, **kwargs: (seen.append(target), real(target, **kwargs))[1])
    assert run_profile.load(path).role == 'DEV'
    assert len(seen) == 1 and Path(seen[0]) == path


# ---- the train profile that a measure profile names (I3) ----------------------------------------------------------------------------
S4B_LIKE = {'kind': 'train', 'kill_law': 'shifted', 'k_condition': 'binned', 'decay_groups': 'split', 'swap_mode': 'random_view',
            'terrain_smooth_weight': 0.1, 'column_drop': 'store_and_subset', 'stats_population': 'selected_fit', 'on_wall_cap': 'best_stop',
            'selection_seed': 20261022, 'epochs': 12}


def measure_file(folder: Path, train_profile: str, sha: str, name: str = 'measure.json', **extra) -> Path:
    return write(folder, {'kind': 'measure', 'train_profile': train_profile, 'train_sha256': sha, 'steps': 5, **extra}, name)


def test_a_measure_profile_loads_the_train_profile_it_names(tmp_path):
    train = write(tmp_path, S4B_LIKE, 'train.json')
    digest = run_io.sha_file(train)
    measure = run_profile.load(measure_file(tmp_path, 'train.json', digest, repeat=3))
    assert measure.kind == 'measure' and measure.steps == 5 and measure.repeat == 3
    assert measure.train_profile == 'train.json' and measure.train_sha256 == digest
    assert isinstance(measure.train, run_profile.Profile) and measure.train.kind == 'train' and measure.train.sha256 == digest
    assert measure.train.settings == run_profile.load(train).settings                 # the measured configuration is the train file
    assert measure.train.kill_law == 'shifted' and measure.train.swap_mode == 'random_view' and measure.train.terrain_smooth_weight == 0.1
    assert measure.train.to_train_kwargs()['decay_groups'] == 'split' and measure.train.membership_seed() == 20261022
    assert measure.train.train is None
    assert make(tmp_path, 'train').train is None and make(tmp_path, 'evaluate').train is None


def test_the_reference_is_resolved_against_the_folder_of_the_measure_file(tmp_path, monkeypatch):
    folder = tmp_path / 'm'
    folder.mkdir()
    train = write(folder, S4B_LIKE, 'train.json')
    decoy = write(tmp_path, {'kind': 'train', 'epochs': 3}, 'train.json')            # same name in the working directory: must not be used
    monkeypatch.chdir(tmp_path)
    measure = run_profile.load(measure_file(folder, 'train.json', run_io.sha_file(train)))
    assert measure.train.epochs == 12 and measure.train.sha256 == run_io.sha_file(train) != run_io.sha_file(decoy)
    with pytest.raises(run_profile.ProfileRefused, match="'train_sha256'"):              # the decoy is a different file: its hash is not the one named
        run_profile.load(measure_file(folder, 'train.json', run_io.sha_file(decoy), 'other.json'))
    # a relative path to the measure file itself works the same way
    monkeypatch.chdir(folder)
    assert run_profile.load(Path('measure.json')).train.epochs == 12


def test_a_hash_that_does_not_match_the_named_file_is_refused(tmp_path):
    train = write(tmp_path, S4B_LIKE, 'train.json')
    digest = run_io.sha_file(train)
    wrong = ('0' if digest[0] != '0' else '1') + digest[1:]
    with pytest.raises(run_profile.ProfileRefused, match="'train_sha256'") as info:
        run_profile.load(measure_file(tmp_path, 'train.json', wrong))
    assert digest not in str(info.value) and wrong not in str(info.value) and 'train.json' not in str(info.value)
    run_profile.load(measure_file(tmp_path, 'train.json', digest))                  # the right one loads
    train.write_bytes(train.read_bytes() + b' ')                                    # the named file changed after the hash was written
    with pytest.raises(run_profile.ProfileRefused, match="'train_sha256'"):
        run_profile.load(measure_file(tmp_path, 'train.json', digest))
    train.write_bytes(train.read_bytes().replace(b'"epochs": 12', b'"epochs": 13'))   # a different run, same shape
    with pytest.raises(run_profile.ProfileRefused, match="'train_sha256'"):
        run_profile.load(measure_file(tmp_path, 'train.json', digest))


def test_a_missing_or_unacceptable_named_file_is_refused_without_its_path(tmp_path):
    with pytest.raises(run_profile.ProfileRefused, match="'train_profile'") as info:
        run_profile.load(measure_file(tmp_path, LEAK + '.json', 'ab' * 32))
    assert LEAK not in str(info.value) and info.value.__context__ is None
    (tmp_path / 'folder.json').mkdir()                                               # a directory is not a profile file
    with pytest.raises(run_profile.ProfileRefused, match="'train_profile'"):
        run_profile.load(measure_file(tmp_path, 'folder.json', 'ab' * 32))
    bad = write(tmp_path, {'kind': 'train', LEAK: 1, 'epochs': 'many'}, 'bad.json')    # the named file is refused by the loader itself
    with pytest.raises(run_profile.ProfileRefused, match="'train_profile'") as info:
        run_profile.load(measure_file(tmp_path, 'bad.json', run_io.sha_file(bad)))
    assert LEAK not in str(info.value) and info.value.__context__ is None
    (tmp_path / 'crlf.json').write_bytes(b'{\r\n  "kind": "train"\r\n}\r\n')            # the named file has CRLF line ends
    with pytest.raises(run_profile.ProfileRefused, match="'train_profile'.*carriage return"):
        run_profile.load(measure_file(tmp_path, 'crlf.json', run_io.sha_file(tmp_path / 'crlf.json')))
    (tmp_path / 'dup.json').write_text('{"kind": "train", "kind": "train"}', encoding='utf-8')     # a strict-reader refusal inside the reference
    with pytest.raises(run_profile.ProfileRefused, match="'train_profile'"):
        run_profile.load(measure_file(tmp_path, 'dup.json', run_io.sha_file(tmp_path / 'dup.json')))


@pytest.mark.parametrize('text', [
    '', '/train.json', '/abs/train.json', '//server/share/train.json', 'C:/x/train.json', 'C:train.json', 'c:\\x\\train.json',
    'sub\\train.json', '\\train.json', '..', '../train.json', 'a/../train.json', 'a/..', './train.json', 'a/./train.json', '.',
    'a//train.json', 'train.json/', '.. /train.json', '... /x.json', 'a./train.json', 'train.json ', ' ', 'x:y.json', 'a\x00b.json', 'a\nb.json',
    'a\tb.json', 'a\x7fb.json'])
def test_a_reference_path_must_be_a_relative_path_with_forward_slashes(tmp_path, text):
    """Refused: empty text, absolute path, drive letter (any colon), backslash, an empty, ``.`` or ``..`` segment, a segment that ends in a dot or
    a space (Windows strips them, so ``.. `` would be ``..``), a control character."""
    with pytest.raises(run_profile.ProfileRefused, match="'train_profile' must be a relative path with forward slashes") as info:
        run_profile.load(measure_file(tmp_path, text, 'ab' * 32))               # refused by the path rule itself, not by a later check
    assert info.value.__context__ is None


@pytest.mark.parametrize('digest', ['ab', 'AB' * 32, 'aB' + 'a' * 62, 'g' * 64, 'a' * 63, 'a' * 65, 'a' * 64 + '\n', ' ' + 'a' * 64, '', 5, None])
def test_a_malformed_digest_is_refused_by_its_format(tmp_path, digest):
    """Refused by the format check (64 lowercase hexadecimal digits), before the named file is looked at."""
    with pytest.raises(run_profile.ProfileRefused, match="'train_sha256' must be 64 lowercase hexadecimal digits"):
        run_profile.load(measure_file(tmp_path, 'train.json', digest))


def test_good_reference_paths_load(tmp_path):
    for sub in ('sub', 'sub/deeper', 'a.b/c-d_e'):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
        train = write(tmp_path / sub, {'kind': 'train'}, 'train.v2.json')
        loaded = run_profile.load(measure_file(tmp_path, f'{sub}/train.v2.json', run_io.sha_file(train)))
        assert loaded.train.sha256 == run_io.sha_file(train)


def test_a_reference_cannot_leave_the_folder_through_a_link(tmp_path):
    """A directory link (a symbolic link, or a junction on Windows) inside the folder that points outside it is not followed."""
    outside = tmp_path / 'outside'
    inside = tmp_path / 'inside'
    outside.mkdir()
    inside.mkdir()
    train = write(outside, {'kind': 'train'}, 'train.json')
    link = inside / 'link'
    try:
        os.symlink(outside, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        done = subprocess.run(['cmd', '/c', 'mklink', '/J', str(link), str(outside)], capture_output=True) if os.name == 'nt' else None
        if done is None or done.returncode != 0:
            pytest.skip('neither a symbolic link nor a junction can be created here')
    try:
        with pytest.raises(run_profile.ProfileRefused, match="'train_profile'"):
            run_profile.load(measure_file(inside, 'link/train.json', run_io.sha_file(train)))
    finally:
        if link.is_symlink() or os.path.isdir(link):
            os.rmdir(link)                                                           # removes the link, never the target


def test_a_reference_does_not_nest(tmp_path):
    """The named file must be a train profile: a measure or an evaluate profile is refused, and a measure file naming itself does not loop."""
    train = write(tmp_path, {'kind': 'train'}, 'train.json')
    inner = measure_file(tmp_path, 'train.json', run_io.sha_file(train), 'inner.json')
    assert run_profile.load(inner).train is not None                                  # the inner one is fine by itself
    with pytest.raises(run_profile.ProfileRefused, match="'train_profile'"):
        run_profile.load(measure_file(tmp_path, 'inner.json', run_io.sha_file(inner), 'outer.json'))
    with pytest.raises(run_profile.ProfileRefused, match="'train_profile'"):
        run_profile.load(measure_file(tmp_path, 'measure.json', 'ab' * 32, 'measure.json'))          # it names itself
    evaluate = write(tmp_path, base(tmp_path, 'evaluate'), 'evaluate.json')
    with pytest.raises(run_profile.ProfileRefused, match="'train_profile'"):
        run_profile.load(measure_file(tmp_path, 'evaluate.json', run_io.sha_file(evaluate)))


# ---- the round trip ----------------------------------------------------------------------------------------------------------------
EXAMPLES = {
    'train': {'kind': 'train', 'size': 'M', 'fraction': 0.5, 'seed': 7, 'epochs': 12, 'patience': 3, 'lr': 0.0005, 'weight_decay': 0.02,
              'warmup': 20, 'kill_law': 'shifted', 'k_condition': 'binned', 'decay_groups': 'split', 'terrain_smooth_weight': 0.1,
              'swap_mode': 'random_view', 'eval_average': False, 'on_wall_cap': 'best_stop', 'zone_mark': False, 'field_channels': 0,
              'column_drop': 'store_and_subset', 'stats_population': 'selected_fit', 'selection_seed': 20261022},
    'evaluate': {'kind': 'evaluate', 'role': 'STOP', 'models': ['b0', 'b1', 'ept', 'n2'], 'exposure': 'nominal', 'winner_truth': 'binned',
                 'chunk_views': 256, 'b0': True, 'strata': True, 'price_sensitivity': True, 'b1_swap_average': True, 'bootstrap_B': 5000,
                 'checkpoints': {'ept': 'runs/20261101_ept/best.pt', 'n2': 'runs/20261101_n2/best.pt'}, 'holdout_lock_sha256': 'ab' * 32,
                 'composed': {'sample_matches': 40, 'sample_seed': 11, 'draws': 200, 'mc_batches': 10, 'draw_seed': 12, 'progress_rows': 8}},
    'measure': {'kind': 'measure', 'train_profile': 'train.json', 'train_sha256': '<set by the test>', 'steps': 5, 'repeat': 3},
}
DEFAULTS_OF = {'train': TRAIN_DEFAULTS, 'evaluate': EVALUATE_DEFAULTS, 'measure': MEASURE_DEFAULTS}


@pytest.mark.parametrize('kind', run_profile.KINDS)
def test_a_written_example_profile_round_trips(tmp_path, kind):
    example = dict(EXAMPLES[kind])
    if kind == 'measure':                                             # the measure example names the train example
        example['train_sha256'] = run_io.sha_file(write(tmp_path, EXAMPLES['train'], 'train.json'))
    assert set(example) - {'kind'} == set(run_profile.FIELDS[kind])           # the example sets every field of its kind
    for name, value in example.items():
        if name in DEFAULTS_OF[kind]:
            assert value != DEFAULTS_OF[kind][name], name                     # ... and every field that has a default to a non-default value
    path = tmp_path / f'{kind}.json'
    digest = run_io.write_json_atomic(path, example)
    profile = run_profile.load(path)
    assert profile.kind == kind and profile.sha256 == digest
    for name, value in example.items():
        if name == 'kind':
            continue
        got = profile.settings[name]
        assert (list(got) if isinstance(value, list) else dict(got) if isinstance(value, dict) else got) == value, name
    if kind == 'measure':
        assert profile.train.settings == run_profile.load(tmp_path / 'train.json').settings
    # the resolved profile is itself a valid profile file with the same settings
    resolved = profile.as_dict()
    assert resolved['kind'] == kind and set(resolved) == set(run_profile.FIELDS[kind]) | {'kind'}
    again = run_profile.load(write(tmp_path, resolved, f'{kind}-resolved.json'))
    assert again.settings == profile.settings and again.kind == kind


@pytest.mark.parametrize('kind', run_profile.KINDS)
def test_the_resolved_default_profile_is_a_valid_file_with_the_same_settings(tmp_path, kind):
    short = make(tmp_path, kind)
    resolved = short.as_dict()
    assert json.loads(json.dumps(resolved)) == resolved                                           # plain JSON values only
    again = run_profile.load(write(tmp_path, resolved, f'{kind}-r.json'))
    assert again.settings == short.settings


# ---- the view for TrainConfig -------------------------------------------------------------------------------------------------------
def test_to_train_kwargs_returns_the_train_config_settings(tmp_path):
    profile = run_profile.load(write(tmp_path, EXAMPLES['train']))
    assert profile.to_train_kwargs() == {
        'kill_law': 'shifted', 'k_condition': 'binned', 'decay_groups': 'split', 'column_drop': 'store_and_subset',
        'stats_population': 'selected_fit', 'terrain_smooth_weight': 0.1, 'swap_mode': 'random_view',
        'eval_average': False, 'on_wall_cap': 'best_stop', 'zone_mark': False,
        'lr': 0.0005, 'weight_decay': 0.02, 'warmup_steps': 20, 'patience': 3}
    assert set(profile.to_train_kwargs()) == set(run_profile.TRAIN_KEYS)


def test_the_defaults_give_the_old_train_config_values(tmp_path):
    profile = make(tmp_path, 'train')
    assert profile.to_train_kwargs() == {
        'kill_law': 'poisson', 'k_condition': 'raw', 'decay_groups': 'all', 'column_drop': 'none', 'stats_population': 'whole_fit',
        'terrain_smooth_weight': 0.0, 'swap_mode': 'average',
        'eval_average': True, 'on_wall_cap': 'fail', 'zone_mark': True, 'lr': 0.001, 'weight_decay': 0.01, 'warmup_steps': 10, 'patience': 2}


def test_the_view_is_a_copy_and_belongs_to_train_profiles_only(tmp_path):
    profile = make(tmp_path, 'train')
    kwargs = profile.to_train_kwargs()
    kwargs['lr'] = 99.0
    assert profile.lr == 0.001 and profile.to_train_kwargs()['lr'] == 0.001
    for kind in ('evaluate', 'measure'):
        with pytest.raises(run_profile.ProfileRefused, match='train'):
            make(tmp_path, kind).to_train_kwargs()
    assert make(tmp_path, 'measure').train.to_train_kwargs()['lr'] == 0.001          # a measure profile has them in .train


# ---- the README of the profile folder ----------------------------------------------------------------------------------------------
README = REPO / 'profiles' / 'README.md'


def readme_rows() -> dict:
    """``{kind: {field: (allowed cell, default cell)}}`` from the tables of ``profiles/README.md`` (rows that start with a field in backticks)."""
    text = README.read_text(encoding='utf-8')
    parts = re.split(r'^### (train|evaluate|measure)[ \t]*$', text, flags=re.MULTILINE)
    rows = {}
    for kind, body in zip(parts[1::2], parts[2::2]):
        rows[kind] = {}
        for line in body.splitlines():
            match = re.match(r'^\| `(\w+)` \| (.+?) \| (.+?)(?: \| .+)? \|$', line)
            if match:
                rows[kind][match.group(1)] = (match.group(2), match.group(3))
    return rows


def test_the_readme_tables_list_exactly_the_fields_of_each_kind():
    rows = readme_rows()
    assert {kind: set(fields) for kind, fields in rows.items()} == {kind: set(names) for kind, names in run_profile.FIELDS.items()}


def test_the_readme_defaults_are_the_defaults_of_the_code(tmp_path):
    rows = readme_rows()
    for kind in run_profile.KINDS:
        loaded = make(tmp_path, kind).settings
        for name, (allowed, default) in rows[kind].items():
            if default == 'required':
                fields = {k: v for k, v in base(tmp_path, kind).items() if k != name}
                with pytest.raises(run_profile.ProfileRefused, match=f"'{name}' is required"):
                    run_profile.load(write(tmp_path, fields))
                continue
            expected = json.loads(default.strip('`'))
            got = loaded[name]
            got = dict(got) if isinstance(got, Mapping) else got
            assert type(got) is type(expected) and got == expected, (kind, name)


def test_the_readme_allowed_texts_are_the_allowed_texts_of_the_code():
    rows = readme_rows()
    for kind, fields in run_profile.CHOICES.items():
        for name, allowed in fields.items():
            assert set(re.findall(r'"([^"]+)"', rows[kind][name][0])) == set(allowed), (kind, name)


def test_the_readme_ranges_are_the_bounds_of_the_code():
    rows = readme_rows()
    for (kind, name, top) in BOUNDS:
        assert f'at most {top}' in rows[kind][name][0].replace(',', ''), (kind, name)
    assert '2^63' in rows['train']['seed'][0] and '2^63' in rows['train']['selection_seed'][0]


def test_the_readme_names_every_train_key():
    """The paragraph of ``to_train_kwargs()`` in the README lists the keys of ``TRAIN_KEYS`` (M5 review H1: it once sent ``column_drop`` and
    ``stats_population`` to the real constructors only)."""
    text = README.read_text(encoding='utf-8')
    start = text.index('`Profile.to_train_kwargs()` returns what `TrainConfig` takes')
    paragraph = text[start:text.index('\n\n', start)].replace('\n', ' ')
    keys_sentence = paragraph.split('This is not a partition')[0]                       # the sentence that lists the keys
    for key in run_profile.TRAIN_KEYS:
        assert f'`{key}`' in keys_sentence, key


def test_the_readme_examples_load(tmp_path):
    blocks = re.findall(r'```json\n(.*?)```', README.read_text(encoding='utf-8'), flags=re.DOTALL)
    assert len(blocks) >= 4
    examples = [json.loads(block) for block in blocks]
    assert [e['kind'] for e in examples] == ['train', 'train', 'evaluate', 'measure']
    assert examples[0] == {'kind': 'train'}                                             # the first example is the bare T002 profile
    train_file = write(tmp_path, examples[0], 'train.json')
    assert examples[3]['train_sha256'] == '<sha256 of train.json>'                      # the measure example names the first example
    examples[3]['train_sha256'] = run_io.sha_file(train_file)
    loaded = [run_profile.load(write(tmp_path, example, f'example-{n}.json')) for n, example in enumerate(examples)]
    assert dict(loaded[0].settings) == TRAIN_DEFAULTS                                    # ... and it reproduces the T002 run
    assert loaded[1].epochs == 12 and loaded[1].kill_law == 'shifted' and loaded[1].swap_mode == 'random_view'
    assert loaded[1].selection_seed == 20261022 and loaded[1].membership_seed() == 20261022
    assert loaded[2].role == 'STOP' and loaded[2].models == ('b0', 'b1', 'ept')
    assert loaded[3].train.settings == loaded[0].settings and loaded[3].steps == 3


# ---- the constructor ---------------------------------------------------------------------------------------------------------------
def test_a_profile_cannot_be_built_with_the_wrong_field_set_hash_or_reference(tmp_path):
    good = make(tmp_path, 'measure')
    settings = dict(good.settings)
    with pytest.raises(run_profile.ProfileRefused):
        run_profile.Profile('measure', good.sha256, {**settings, 'lr': 0.1}, good.train)            # an extra field
    with pytest.raises(run_profile.ProfileRefused):
        run_profile.Profile('measure', good.sha256, {k: v for k, v in settings.items() if k != 'steps'}, good.train)
    with pytest.raises(run_profile.ProfileRefused):
        run_profile.Profile('measure', 'not-a-hash', settings, good.train)
    with pytest.raises(run_profile.ProfileRefused):
        run_profile.Profile('trainer', good.sha256, settings, good.train)
    with pytest.raises(run_profile.ProfileRefused):
        run_profile.Profile('measure', good.sha256, settings)                                       # a measure profile needs its train profile
    with pytest.raises(run_profile.ProfileRefused):
        run_profile.Profile('measure', good.sha256, {**settings, 'train_sha256': '0' * 64}, good.train)    # ... and it must match the hash
    with pytest.raises(run_profile.ProfileRefused):
        run_profile.Profile('measure', good.sha256, settings, make(tmp_path, 'evaluate'))           # ... and be a train profile
    train = make(tmp_path, 'train')
    with pytest.raises(run_profile.ProfileRefused):
        run_profile.Profile('train', train.sha256, dict(train.settings), train)                    # the other kinds carry none
    assert run_profile.Profile('measure', good.sha256, settings, good.train) == good
    assert run_profile.Profile('train', train.sha256, dict(train.settings)) == train


def test_the_train_kwargs_of_a_profile_build_a_trainconfig_and_the_default_profile_builds_the_default_one(tmp_path):
    """M9 (the last Track M field): ``TrainConfig(encoder, **profile.to_train_kwargs())`` works. The bare train profile gives the TrainConfig whose Track M fields are all at their defaults (so its run is the
    run of every earlier task); the A7 profile (``zone_mark`` False) differs in that one setting only; the A6 profile (``field_channels`` 0 with ``terrain_smooth_weight`` 0) routes the channel count to the
    encoder and a TrainConfig that accepts the zero weight."""
    from c_ept.ept import encoder as ept_encoder
    from c_ept.ept import train as ept_train
    small = ept_encoder.EncoderConfig(d=16, layers=1, heads=2)
    default = ept_train.TrainConfig(small, **make(tmp_path, 'train').to_train_kwargs())
    assert default == ept_train.TrainConfig(small, lr=0.001, weight_decay=0.01, warmup_steps=10)                      # patience 2 and every Track M setting at its default
    assert default.zone_mark is True and routing_problems(class_defaults('c_ept/ept/train.py', 'TrainConfig'), run_profile.TRAIN_KEYS, make(tmp_path, 'train').to_train_kwargs()) == []
    assert PENDING_TRACK_M == frozenset()
    a7 = ept_train.TrainConfig(small, **make(tmp_path, 'train', zone_mark=False).to_train_kwargs())
    assert a7.zone_mark is False and a7 == ept_train.TrainConfig(small, lr=0.001, weight_decay=0.01, warmup_steps=10, zone_mark=False)
    a6 = make(tmp_path, 'train', field_channels=0, terrain_smooth_weight=0)
    assert a6.field_channels == 0 and a6.to_train_kwargs()['terrain_smooth_weight'] == 0
    assert ept_train.TrainConfig(ept_encoder.EncoderConfig(d=16, layers=1, heads=2, field_channels=a6.field_channels), **a6.to_train_kwargs()).encoder.field_channels == 0
    every = {key: value for key, value in make(tmp_path, 'train', kill_law='shifted', zone_mark=False, swap_mode='random_view', eval_average=False, on_wall_cap='best_stop').to_train_kwargs().items()}
    assert ept_train.TrainConfig(small, **every).zone_mark is False


# ---- S4b profile files (S4b-prep K9, decisions D8, D10, D13, D17) -----------------------------------------------------------------------------
S4B = REPO / 'profiles' / 's4b'
S4B_COMMON = {'size': 'S', 'patience': 2, 'weight_decay': 0.01, 'warmup': 10, 'kill_law': 'shifted', 'k_condition': 'binned', 'decay_groups': 'split',
              'column_drop': 'store_and_subset', 'stats_population': 'selected_fit', 'on_wall_cap': 'best_stop'}
# The table of plan Task K9, written out again here on purpose: a change of a value in tools/make_s4b_profiles.py must change this table too.
# Only profiles whose every value is decided now are written (training 1 at lambda_Omega 0 and its STOP evaluation); everything that depends on the best lambda_Omega, the best
# learning rate, the chosen configuration, the 50 % rule or the measurement steps is pending (PENDING.json) and is not replaced by a stand-in.
S4B_TRAIN = {
    'train_01_lambda0.json': dict(fraction=0.25, epochs=12, seed=20261022, lr=0.001, terrain_smooth_weight=0.0, swap_mode='random_view', eval_average=True,
                                  zone_mark=True, field_channels=8, selection_seed=20261022),
}
S4B_EVALUATE_MODELS = {'evaluate_01_lambda0.json': 't01'}
S4B_SHA256 = {
    'evaluate_01_lambda0.json': '79abd79da9989efa8977ba45cb1ae45c0c268d551fb9698e2a54c4bdbb58b3dc',
    'train_01_lambda0.json': '55e88291d85aedcfd0f6812cd287eec143af1ba0004edc68d28480c24793606b',
}
# What the S4b run plan (X1) still decides: the ids and statuses of PENDING.json. A status moves only when X1 passes the value to the generator.
S4B_PENDING = {'best_lambda': 'pending', 'best_lr': 'pending', 'c4_seed_group': 'pending', 'checkpoints': 'pending', 'chunk_views': 'pending',
               'lambda_omega_values': 'pending', 'measure_steps': 'pending', 'share50_membership_and_epochs': 'pending'}


def s4b_generator():
    """The generator ``tools/make_s4b_profiles.py`` loaded from its file (a command module, not imported by name)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('make_s4b_profiles_under_test', REPO / 'tools' / 'make_s4b_profiles.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_s4b_profiles_load():
    """Every written S4b profile loads, its hash is pinned and its values equal the table of plan Task K9 (decisions D8, D10, D13, D17)."""
    written = sorted(path.name for path in S4B.glob('*.json') if path.name != 'PENDING.json')
    assert written == sorted(S4B_SHA256)                                       # no file more, no file fewer than the pinned ones
    for name, digest in S4B_SHA256.items():
        path = S4B / name
        assert b'\r' not in path.read_bytes(), name                              # one content, one hash (K3 review m4); .gitattributes keeps it so
        profile = run_profile.load(path)
        assert profile.sha256 == digest, name
        assert profile.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    for name, expected in S4B_TRAIN.items():
        profile = run_profile.load(S4B / name)
        assert profile.kind == 'train'
        assert dict(profile.settings) == dict(S4B_COMMON, **expected), name
        assert profile.settings['selection_seed'] == (20261022 if profile.fraction == 0.25 else None)     # the 25 % runs share one fixed membership
    for name, model in S4B_EVALUATE_MODELS.items():
        profile = run_profile.load(S4B / name)
        assert profile.kind == 'evaluate' and profile.role == 'STOP' and profile.models == (model,), name
        assert profile.chunk_views is None and profile.holdout_lock_sha256 is None and dict(profile.checkpoints) == {}      # pending in PENDING.json
        assert profile.exposure == 'observed' and profile.winner_truth == 'gold' and profile.bootstrap_B == 2000 and not profile.b0
    assert not (S4B / 'measure_p6.json').exists()                               # the measurement steps are pending (written when X1 gives them)


def test_s4b_the_trainings_that_the_plan_leaves_to_the_run_plan_are_not_written():
    """Every training that depends on a value the plan leaves to X1 (trainings 2-13) has no file, and no evaluation of it; nothing guesses or stands in for them."""
    names = {path.name for path in S4B.glob('*.json')}
    assert names == {'train_01_lambda0.json', 'evaluate_01_lambda0.json', 'PENDING.json'}
    assert run_profile.load(S4B / 'train_01_lambda0.json').terrain_smooth_weight == 0.0


def test_s4b_the_pending_list_is_pinned():
    pending = run_io.read_json(S4B / 'PENDING.json')
    assert sorted(pending) == ['items', 'note']
    assert {item['id']: item['status'] for item in pending['items']} == S4B_PENDING
    for item in pending['items']:
        assert sorted(item) == ['affects', 'argument', 'decision', 'id', 'reason', 'status']
        assert item['status'] == 'pending' and item['reason'] and item['decision'] and item['argument'].startswith('--')
    by_id = {item['id']: item for item in pending['items']}
    assert by_id['chunk_views']['reason'].endswith('hand-off value is 256')        # the M14 value is recorded as pending, not written
    assert 'train_12_share50.json' in by_id['share50_membership_and_epochs']['affects']
    assert by_id['lambda_omega_values']['affects'] == ['train_02_lambda_a.json', 'train_03_lambda_b.json']
    assert by_id['best_lambda']['affects'][0] == 'train_04_lr_low.json' and 'train_13_share100.json' in by_id['best_lr']['affects']
    assert by_id['measure_steps']['reason'].startswith('at least warmup_steps (10) plus enough steps to cover the batch-shape variety; set at X1')


def test_s4b_the_gitattributes_keeps_profile_line_ends():
    lines = (REPO / '.gitattributes').read_text(encoding='utf-8').splitlines()
    assert 'profiles/**/*.json -text' in lines


def test_s4b_the_generator_reproduces_the_committed_files_byte_for_byte(tmp_path):
    generator = s4b_generator()
    written = generator.generate(tmp_path)
    assert written == {**S4B_SHA256, 'PENDING.json': hashlib.sha256((S4B / 'PENDING.json').read_bytes()).hexdigest()}
    for name in written:
        assert (tmp_path / name).read_bytes() == (S4B / name).read_bytes(), name


def test_s4b_the_generator_takes_the_pending_values_and_writes_the_final_files(tmp_path):
    generator = s4b_generator()
    written = generator.generate(tmp_path, lambdas=(0.01, 0.1), best_lambda=0.01, best_lr=0.002, membership_50=20261022, epochs_50=10, chunk_views=256,
                                 chosen_training=5, checkpoints={'t05': 'runs/a', 't08': 'runs/b', 't09': 'runs/c'}, measure_steps=5)
    assert sorted(n for n in range(1, 14) if generator.train_file(n) in written) == list(range(1, 14))
    train2, train3 = (run_profile.load(tmp_path / generator.train_file(n)) for n in (2, 3))
    assert (train2.terrain_smooth_weight, train3.terrain_smooth_weight) == (0.01, 0.1) and train2.lr == train3.lr == 0.001
    for number in (4, 5, 6, 7, 8, 9, 10, 11, 12, 13):
        assert run_profile.load(tmp_path / generator.train_file(number)).terrain_smooth_weight == 0.01
    assert run_profile.load(tmp_path / generator.train_file(4)).lr == 0.0005 and run_profile.load(tmp_path / generator.train_file(5)).lr == 0.002
    assert [run_profile.load(tmp_path / generator.train_file(n)).lr for n in (8, 9, 6, 13)] == [0.002] * 4
    share50 = run_profile.load(tmp_path / generator.train_file(12))
    assert (share50.fraction, share50.epochs, share50.selection_seed) == (0.5, 10, 20261022)
    c4 = run_profile.load(tmp_path / 'evaluate_c4_seeds.json')
    assert c4.models == ('t05', 't08', 't09') and dict(c4.checkpoints) == {'t05': 'runs/a', 't08': 'runs/b', 't09': 'runs/c'} and c4.chunk_views == 256
    measure = run_profile.load(tmp_path / 'measure_p6.json')
    assert measure.steps == 5 and measure.train_profile == generator.train_file(1) and measure.train.sha256 == measure.train_sha256
    assert run_io.read_json(tmp_path / 'PENDING.json')['items'] == []                  # every pending value was passed
    # "training_seed" asks for the T002 rule: the selection follows the training seed (a null selection_seed)
    only_lambda = generator.generate(tmp_path / 'lam', best_lambda=0.01)                      # a missing lr* leaves out 6-13, a missing lambda_Omega values leaves out 2 and 3
    assert sorted(name for name in only_lambda if name.startswith('train_')) == [generator.train_file(n) for n in (1, 4, 5)]
    assert not (tmp_path / 'lam' / 'measure_p6.json').exists()
    other = generator.generate(tmp_path / 'other', best_lambda=0.01, best_lr=0.001, membership_50='training_seed', epochs_50=7)
    assert run_profile.load(tmp_path / 'other' / generator.train_file(12)).selection_seed is None and generator.train_file(12) in other


def test_s4b_the_generator_removes_the_stale_files_of_an_earlier_run(tmp_path):
    generator = s4b_generator()
    generator.generate(tmp_path, lambdas=(0.01, 0.1))
    assert (tmp_path / generator.train_file(2)).exists()
    (tmp_path / 'notes.json').write_text('{}', encoding='utf-8')               # a file that is not the generator's is left alone
    generator.generate(tmp_path)
    assert not (tmp_path / generator.train_file(2)).exists() and (tmp_path / 'notes.json').exists()


def test_s4b_the_generator_refuses_a_c4_group_it_cannot_write(tmp_path):
    generator = s4b_generator()
    with pytest.raises(ValueError):
        generator.generate(tmp_path, chosen_training=6)
    with pytest.raises(ValueError):
        generator.generate(tmp_path, chosen_training=2)                        # training 2 is not written without lambdas
