"""Tests of the K condition of the K-conditioned heads (S4b preparation, task M3, decision D21 B16, review finding A3-F4).

The defect. The training rows of the heads that are conditioned on K (the object head and the conversion head, masked and open, of the EPT and of B1) carried the true continuous
kill gold of the labels, while the composer feeds them the representative ``reps[drawn gold bin]`` of a team with a kill and 0 for a team without one. ``k_condition='binned'``
makes the rows carry the composer's K: ``rep[labels/gold_bin]`` for a team with a kill, 0 for a team without one, with the 31 representatives from the in-scope FIT units of the
label store (``value.fit_gold_bin_means(population='in_scope')``, one source for the rows and for ``run_inference``). ``k_condition='raw'`` (the default) is the rows of every
earlier run and must give the bits of the code before this task.

What the "default" tests prove. A test of this file can only compare the code as it is now. Where a test says the default equals the explicit ``'raw'``, that is all it says (the
keyword changes nothing); where it compares with the oracle of ``s3_toys`` (the true gold of the labels read by plain loops) or with a second implementation written from the
text of a rule, it is a real comparison. The evidence that ``k_condition='raw'`` gives the bits of the base commit 63b454f is the set of comparison scripts of the T012 session
folder (``scratchpad\\s4bprep``): ``m1_proof_ept.py`` (loss, gradient, biases, rows digest, identity, a 3-epoch run, frame-averaged rates), ``m1_proof_compose.py`` (closed form,
draws), ``m1_proof_models.py`` (B0 and B1 saved files, scores), ``m1_proof_inference.py`` and ``m2_proof_inference.py`` (every field of ``run_inference``), ``m2_proof_value.py``
(the labels and the bin means) and ``m3_proof_rows.py`` (every field of every head of the rows of B0, B1 and the EPT), each run at the base commit and at the final commit with equal
sha256 per key, and the equivalence acceptance test K8 of the kernel track.
"""
import dataclasses
import gc
import inspect
import json
import math
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from c_ept.data import head_rows
from c_ept.ept import data, encoder, heads, loss, train
from c_ept.models import baselines_b0, baselines_b1, killlaw
from c_ept.preprocessing import features
from c_ept.tests.ept_toys import build_ept_world, toy_static  # noqa: F401  (the fixtures are imported by ruling R1)
from c_ept.tests.s3_toys import consistent_heads_world, heads_world  # noqa: F401
from c_ept.valuation import value

SMALL = {'num_leaves': 7, 'min_data_in_leaf': 5, 'num_iterations': 12, 'min_data_per_group': 5, 'cat_smooth': 1.0, 'num_threads': 1}
K_BLOCKS = [('M3Obj', 'masked'), ('M4Conv', 'masked'), ('M3Obj', 'open'), ('M4Conv', 'open')]       # the blocks whose K field is the K block of the rows


# =====================================================================================================================================
# the EPT rows and the training loop
# =====================================================================================================================================
@pytest.fixture(scope='module')
def run_world(tmp_path_factory):
    """The EPT toy world of the training tests (7 events, 5 edges) with FIT statistics."""
    folder = tmp_path_factory.mktemp('kcondition_train')
    world = build_ept_world(folder, n_events=7, n_edges=5)
    yield world, data.fit_stats(world.s1, world.s2)
    del world
    gc.collect()


def ept_rows(world, role='FIT', k_condition=None):
    kw = {} if k_condition is None else {'k_condition': k_condition}
    return data.ept_head_rows(world.s1, world.s2, world.labels, role, **kw)


def test_ept_head_rows_take_the_k_condition_and_the_engagement_blocks_carry_the_binned_gold_in_the_k_fields_only(run_world):
    world, _ = run_world
    raw_rows, binned_rows = ept_rows(world), ept_rows(world, k_condition='binned')
    assert raw_rows.k_condition == 'raw' and binned_rows.k_condition == 'binned' and raw_rows.k_reps() is None
    raw, binned = data.EngagementRows(raw_rows), data.EngagementRows(binned_rows)
    reps = binned_rows.k_reps()
    support = (np.concatenate([[0.0], reps]) / 1000.0).astype(np.float32)           # the gold in thousands in the float32 of a batch
    for key in data.ROW_FIELDS:
        a, b = raw.blocks[key], binned.blocks[key]
        assert a.fields.keys() == b.fields.keys() and np.array_equal(a.views, b.views) and np.array_equal(a.group, b.group), key
        for name in a.fields:
            same = np.array_equal(a.fields[name], b.fields[name])
            # the K field of the four K blocks is the only thing that changes; the kill counts (columns 0, 1) and the O field stay
            assert same == (not (key in K_BLOCKS and name == 'K')), (key, name)
    for key in K_BLOCKS:
        K = binned.blocks[key].fields['K']
        assert K.dtype == np.float32 and np.isin(K[:, 2:], support).all(), key
        assert np.array_equal(K[:, :2], raw.blocks[key].fields['K'][:, :2]), key
    assert not np.isin(raw.blocks[('M3Obj', 'open')].fields['K'][:, 2:], support).all()      # the control: the raw gold leaves the support
    parameters = inspect.signature(data.ept_head_rows).parameters
    names = list(parameters)                                                       # relative order, so that a later keyword after these does not break the pin (review m3)
    assert names.index('kill_law') == names.index('matches') + 1 and names.index('k_condition') == names.index('kill_law') + 1
    assert parameters['k_condition'].default == 'raw' and parameters['k_condition'].kind is inspect.Parameter.KEYWORD_ONLY
    with pytest.raises(ValueError, match='k_condition'):
        ept_rows(world, k_condition='continuous')


def test_the_rows_digest_of_the_binned_rows_differs_from_the_raw_digest_in_the_rows_hash_alone(run_world):
    """The digest of a run (``train._rows_digest``) hashes every field of every row, so rows that carry other gold are another input of the run."""
    world, stats = run_world

    def digest(k_condition):
        rows = data.EngagementRows(ept_rows(world, k_condition=k_condition))
        return train._rows_digest(data.Batches(rows, 4, 7, stats=stats, training=True))
    raw, binned, again = digest(None), digest('binned'), digest('binned')
    assert raw['rows_sha256'] != binned['rows_sha256'] and binned == again
    assert {k: v for k, v in raw.items() if k != 'rows_sha256'} == {k: v for k, v in binned.items() if k != 'rows_sha256'}
    assert digest('raw') == raw


def test_the_train_config_has_the_k_condition_after_the_kill_law_with_the_default_raw_and_refuses_another_name():
    names = [f.name for f in dataclasses.fields(train.TrainConfig)]
    # relative order (review m3): the fields of the next tasks come after these, so no pin may name the tail
    assert names.index('kill_law') == names.index('model_spec') + 1 and names.index('k_condition') == names.index('kill_law') + 1
    assert names[:6] == ['encoder', 'lr', 'weight_decay', 'warmup_steps', 'patience', 'model_spec']
    cfg = train.TrainConfig(encoder.EncoderConfig(d=16, layers=1, heads=2), 1e-3, 0.0, 1)
    assert cfg.k_condition == 'raw' and train.TrainConfig(encoder.EncoderConfig(d=16, layers=1, heads=2), 1e-3, 0.0, 1, 2, None, 'poisson').k_condition == 'raw'
    assert train.TrainConfig(encoder.EncoderConfig(d=16, layers=1, heads=2), 1e-3, 0.0, 1, k_condition='binned').k_condition == 'binned'
    for bad in ('continuous', 'Binned', None, 1):
        with pytest.raises(ValueError, match='k_condition'):
            train.TrainConfig(encoder.EncoderConfig(d=16, layers=1, heads=2), 1e-3, 0.0, 1, k_condition=bad)


def run_data(world, stats, k_condition):
    rows = {role: data.EngagementRows(ept_rows(world, role, k_condition)) for role in ('FIT', 'STOP')}
    fit = data.Batches(rows['FIT'], max(1, math.ceil(len(rows['FIT'].units) / 3)), 7, stats=stats, training=True)
    stop = data.Batches(rows['STOP'], max(1, len(rows['STOP'].units)), 7, stats=stats, training=False)
    scales = {key: 1.0 + 0.25 * i for i, key in enumerate(loss.BLOCKS)}
    return train.TrainData(fit=fit, stop=stop, scales=scales, static=toy_static())


def run_cfg(k_condition):
    kw = {} if k_condition == 'raw' else {'k_condition': k_condition}
    return train.TrainConfig(encoder=encoder.EncoderConfig(d=16, layers=2, heads=2, dropout=0.1, kappas=1, field_channels=2), lr=3e-3, weight_decay=0.01,
                             warmup_steps=2, patience=10, **kw)


def run_train(folder, world, stats, cfg, data_condition, epochs=2):
    return train.train(cfg, run_data(world, stats, data_condition), folder, device='cpu', epochs=epochs, check=lambda: None, seed=3)


def test_a_binned_run_records_the_condition_in_its_identity_and_trains_on_other_inputs_than_the_raw_run(run_world, tmp_path):
    world, stats = run_world
    binned = run_train(tmp_path / 'binned', world, stats, run_cfg('binned'), 'binned')
    raw = run_train(tmp_path / 'raw', world, stats, run_cfg('raw'), 'raw')
    assert binned.head['identity']['k_condition'] == 'binned' and 'k_condition' not in raw.head['identity']          # absent means the raw gold
    assert 'k_condition' not in json.dumps(raw.head['identity']) and 'k_reps_sha256' not in json.dumps(raw.head['identity'])
    # review m4: the 31 numbers that the K block was made of are part of the identity (as the sha256 of the float64 little-endian bytes), only under the binned condition
    assert binned.head['identity']['k_reps_sha256'] == value.gold_reps_sha256(ept_rows(world, 'FIT', 'binned').k_reps())
    assert np.isfinite(binned.stop_losses).all() and np.isfinite(binned.fit_batch_mean_losses).all() and len(binned.stop_losses) == 2
    assert binned.stop_losses != raw.stop_losses                                                                      # the input is not a label
    assert binned.head['identity']['fit']['rows_sha256'] != raw.head['identity']['fit']['rows_sha256']


def test_the_condition_of_the_rows_and_the_condition_of_the_settings_must_agree_before_any_work_in_both_directions(run_world, tmp_path):
    world, stats = run_world
    with pytest.raises(ValueError, match='k_condition'):
        run_train(tmp_path / 'a', world, stats, run_cfg('binned'), 'raw')
    with pytest.raises(ValueError, match='k_condition'):
        run_train(tmp_path / 'b', world, stats, run_cfg('raw'), 'binned')
    for name in ('a', 'b'):
        assert not (tmp_path / name).exists() or not list((tmp_path / name).glob('epoch_*'))


def test_the_fit_and_stop_rows_of_a_binned_run_must_hold_the_same_representatives_before_any_work(run_world, tmp_path):
    """Review m4: FIT and STOP rows made from different label stores would train on one K and stop on another; the run refuses before it writes an epoch."""
    world, stats = run_world
    scaled = Overlay(world.labels, {'labels/kill_gold': np.asarray(world.labels.array('labels/kill_gold')) * 1.5})
    fit = data.Batches(data.EngagementRows(ept_rows(world, 'FIT', 'binned')), 4, 7, stats=stats, training=True)
    stop_rows = data.ept_head_rows(world.s1, world.s2, scaled, 'STOP', k_condition='binned')
    assert stop_rows.k_reps_sha256() != fit.rows.head_rows.k_reps_sha256()
    stop = data.Batches(data.EngagementRows(stop_rows), 4, 7, stats=stats, training=False)
    good = run_data(world, stats, 'binned')
    with pytest.raises(ValueError, match='k_reps_sha256'):
        train.train(run_cfg('binned'), train.TrainData(fit=fit, stop=stop, scales=good.scales, static=good.static), tmp_path / 'mixed', device='cpu', epochs=1,
                    check=lambda: None, seed=3)
    assert not (tmp_path / 'mixed').exists() or not list((tmp_path / 'mixed').glob('epoch_*'))


def test_a_run_cannot_be_resumed_under_the_other_k_condition_in_either_direction(run_world, tmp_path):
    world, stats = run_world
    for first, second in (('binned', 'raw'), ('raw', 'binned')):
        run_train(tmp_path / first, world, stats, run_cfg(first), first, epochs=2)
        with pytest.raises(ValueError, match='k_condition'):                        # the message of ``_check_identity`` lists the keys that differ
            run_train(tmp_path / first, world, stats, run_cfg(second), second, epochs=2)


# =====================================================================================================================================
# B1 (the object and conversion heads read K in their designs; B0 reads K in its open heads and follows the condition since M3b, decision D24: see the end of this file)
# =====================================================================================================================================
@pytest.fixture(scope='module')
def b1_heads(heads_world):
    """B1 fitted on the heads world under both K conditions (small trees)."""
    w = heads_world
    dropped = features.dropped_columns(w.s2, w.fit_views)
    return SimpleNamespace(world=w, dropped=dropped,
                           b1={condition: baselines_b1.B1Heads.fit(w.s1, w.s2, w.labels, dropped, params=SMALL, k_condition=condition) for condition in head_rows.K_CONDITIONS})


def b1_rows(heads, role, k_condition=None, labels=None):
    kw = {} if k_condition is None else {'k_condition': k_condition}
    w = heads.world
    return baselines_b1.HeadRows(w.s1, w.s2, w.labels if labels is None else labels, heads.dropped, role, **kw)


def test_b1_heads_fitted_on_the_binned_condition_change_the_k_heads_only_and_record_the_condition(b1_heads, tmp_path):
    raw, binned = b1_heads.b1['raw'], b1_heads.b1['binned']
    assert raw.k_condition == 'raw' and binned.k_condition == 'binned'
    for kind in ('masked', 'open'):
        for name, head in getattr(binned, kind).items():
            same = head.state()[0] == getattr(raw, kind)[name].state()[0]                 # the texts of the boosters
            assert same == ((name, kind) not in K_BLOCKS), (kind, name)                   # a head is the head of the raw fit unless its design holds K
    # save and load: the set records the condition when it is not the raw gold, and a raw set is the file of every earlier set
    binned.save(tmp_path / 'binned')
    raw.save(tmp_path / 'raw')
    doc_b = json.loads((tmp_path / 'binned' / baselines_b1.JSON_NAME).read_text(encoding='utf-8'))
    doc_r = json.loads((tmp_path / 'raw' / baselines_b1.JSON_NAME).read_text(encoding='utf-8'))
    assert doc_b['k_condition'] == 'binned' and 'k_condition' not in doc_r and 'k_condition' not in json.dumps(doc_r)
    # review m4: the set records which 31 numbers its K blocks were fitted on (sha256 of the float64 little-endian bytes), only when binned
    sha = value.gold_reps_sha256(b1_rows(b1_heads, 'FIT', 'binned').k_reps())
    assert binned.k_reps_sha256 == sha == doc_b['k_reps_sha256'] and raw.k_reps_sha256 is None and 'k_reps_sha256' not in json.dumps(doc_r)
    back = baselines_b1.B1Heads.load(tmp_path / 'binned')
    assert back.k_condition == 'binned' and baselines_b1.B1Heads.load(tmp_path / 'raw').k_condition == 'raw'
    assert back.k_reps_sha256 == sha and baselines_b1.B1Heads.load(tmp_path / 'raw').k_reps_sha256 is None
    d = b1_rows(b1_heads, 'DEV', 'binned').data('M3Obj', 'open')
    assert np.array_equal(back.open['M3Obj'].predict(d.X), binned.open['M3Obj'].predict(d.X))
    doc_b['k_condition'] = 'nb'
    (tmp_path / 'binned' / baselines_b1.JSON_NAME).write_text(json.dumps(doc_b), encoding='utf-8')
    with pytest.raises(ValueError, match='k_condition'):
        baselines_b1.B1Heads.load(tmp_path / 'binned')


def test_b1_scores_the_k_heads_on_the_rows_of_the_condition_it_was_fitted_on(b1_heads):
    w = b1_heads.world
    raw, binned = b1_heads.b1['raw'], b1_heads.b1['binned']
    got = {kind: binned.nll(w.s1, w.s2, w.labels, 'DEV', kind) for kind in ('masked', 'open')}
    other_score = []
    for name, kind in K_BLOCKS:
        d = b1_rows(b1_heads, 'DEV', 'binned').data(name, kind)
        want = getattr(binned, kind)[name].nll(d.X, d.y, d.e)
        assert np.array_equal(got[kind][name].nll, want), (name, kind)
        other = b1_rows(b1_heads, 'DEV').data(name, kind)                                  # the same heads on the true gold
        other_score.append(not np.array_equal(getattr(binned, kind)[name].nll(other.X, other.y, other.e), want))
    assert any(other_score)           # the control: the condition of the scored rows matters (a head with small trees may not split on the gold columns, so not every head moves)
    # a head that does not read K is scored as the head of the raw fit
    assert np.array_equal(got['open']['M3Size'].nll, raw.nll(w.s1, w.s2, w.labels, 'DEV', 'open')['M3Size'].nll)


def test_the_fit_and_stop_rows_of_a_b1_head_must_have_the_same_k_condition_in_both_directions(b1_heads):
    for fit_condition, stop_condition in (('raw', 'binned'), ('binned', 'raw')):
        with pytest.raises(ValueError, match='k_condition'):
            baselines_b1.fit_head(b1_rows(b1_heads, 'FIT', fit_condition), b1_rows(b1_heads, 'STOP', stop_condition), 'M3Obj', 'open', params=SMALL)
    head = baselines_b1.fit_head(b1_rows(b1_heads, 'FIT', 'binned'), b1_rows(b1_heads, 'STOP', 'binned'), 'M3Obj', 'open', params=SMALL)
    assert head.state()[0] == b1_heads.b1['binned'].open['M3Obj'].state()[0]             # the same rows, the same head, bit for bit
    # review m4: the same condition from two label stores is still two K variables (other representatives): refused as well
    scaled = Overlay(b1_heads.world.labels, {'labels/kill_gold': np.asarray(b1_heads.world.labels.array('labels/kill_gold')) * 1.5})
    with pytest.raises(ValueError, match='k_reps_sha256'):
        baselines_b1.fit_head(b1_rows(b1_heads, 'FIT', 'binned'), b1_rows(b1_heads, 'STOP', 'binned', labels=scaled), 'M3Obj', 'open', params=SMALL)


def test_a_b1_set_checks_the_sha256_of_its_representatives(b1_heads):
    raw, binned = b1_heads.b1['raw'], b1_heads.b1['binned']
    sha = binned.k_reps_sha256
    assert isinstance(sha, str) and len(sha) == 64 and sha == sha.lower()
    with pytest.raises(ValueError, match='k_reps_sha256'):                                  # a raw set has no representatives
        baselines_b1.B1Heads(raw.masked, raw.open, b1_heads.dropped, k_reps_sha256=sha)
    for bad in ('abc', sha.upper(), sha[:-1], 5, ''):
        with pytest.raises(ValueError, match='k_reps_sha256'):
            baselines_b1.B1Heads(binned.masked, binned.open, b1_heads.dropped, k_condition='binned', k_reps_sha256=bad)
    unrecorded = baselines_b1.B1Heads(binned.masked, binned.open, b1_heads.dropped, k_condition='binned')              # allowed to exist; run_inference refuses it
    assert unrecorded.k_reps_sha256 is None
    parameters = inspect.signature(baselines_b1.B1Heads.__init__).parameters
    names = list(parameters)
    assert names.index('k_reps_sha256') == names.index('k_condition') + 1 and parameters['k_reps_sha256'].default is None


def test_a_b1_set_checks_the_name_of_the_k_condition_and_the_default_fit_equals_the_explicit_raw_fit(b1_heads, tmp_path):
    w = b1_heads.world
    raw = b1_heads.b1['raw']
    with pytest.raises(ValueError, match='k_condition'):
        baselines_b1.B1Heads(raw.masked, raw.open, b1_heads.dropped, k_condition='continuous')
    with pytest.raises(ValueError, match='k_condition'):
        baselines_b1.B1Heads.fit(w.s1, w.s2, w.labels, b1_heads.dropped, params=SMALL, k_condition='continuous')
    parameters = inspect.signature(baselines_b1.B1Heads.fit).parameters
    names = list(parameters)
    assert parameters['k_condition'].default == 'raw' and names.index('k_condition') == names.index('kill_law') + 1
    assert parameters['k_condition'].kind is inspect.Parameter.KEYWORD_ONLY
    # what this proves: the keyword at its default is the explicit 'raw' (the files are equal bit for bit); the comparison with the code before the task is the proof script
    default = baselines_b1.B1Heads.fit(w.s1, w.s2, w.labels, b1_heads.dropped, params=SMALL)
    default.save(tmp_path / 'default')
    raw.save(tmp_path / 'explicit')
    files = sorted(p.relative_to(tmp_path / 'default') for p in (tmp_path / 'default').rglob('*') if p.is_file())
    assert files == sorted(p.relative_to(tmp_path / 'explicit') for p in (tmp_path / 'explicit').rglob('*') if p.is_file()) and files
    for f in files:
        assert (tmp_path / 'default' / f).read_bytes() == (tmp_path / 'explicit' / f).read_bytes(), f


# =====================================================================================================================================
# run_inference: the representatives of the draws and of the binned truth are the representatives of the training rows
# =====================================================================================================================================
class Overlay:
    """Small synthetic additions to a linked toy store (as in ``tests/test_ept_dev_adapters``)."""

    def __init__(self, base, arrays):
        self.base, self.arrays = base, arrays

    def array(self, key):
        if key == 'matches/key':
            raise AssertionError('no match keys needed')
        return self.arrays[key] if key in self.arrays else self.base.array(key)


LABEL_REPS = np.arange(31, dtype=float) * 100                       # the labels/gold_reps of the toy bundle: not the means of its FIT units


@pytest.fixture(scope='module')
def inference_world(tmp_path_factory):
    """A DEV bundle on the EPT toy world, a tiny EPT, FIT statistics and B1 heads of both K conditions fitted on the world."""
    from c_ept.tests.s3_toys import toy_tier_model
    from c_ept.valuation import flowprice
    world = build_ept_world(tmp_path_factory.mktemp('kcondition_inference'), n_events=7, n_edges=5)
    n_matches = len(np.asarray(world.s1.array('matches/role')))
    s1 = Overlay(world.s1, {'matches/obj_off': np.zeros(n_matches + 1, dtype=np.int64),
                            **{f'objectives/{k}': np.empty(0, dtype=np.int64) for k in ('t', 'kind', 'sub', 'taker')}})
    n_units = len(s1.array('units/f'))
    labels = Overlay(world.labels, {'labels/f0_unit': np.resize(np.array([-1., 0., 1.]), n_units), 'labels/gold_reps': LABEL_REPS,
                                    'labels/obj_value': np.zeros((n_units, 2))})
    stats = data.fit_stats(s1, world.s2)
    torch.manual_seed(17)
    model = heads.EPT(encoder.EncoderConfig(d=16, layers=2, heads=2, dropout=.1, kappas=1, field_channels=2), toy_static())
    bundle = SimpleNamespace(s1=s1, s2=world.s2, labels=labels, role='DEV', prices={k: 100. for k in flowprice.GROUPS}, tier=toy_tier_model())
    dropped = features.dropped_columns(world.s2, np.flatnonzero(features.split_views(world.s1, world.s2, 'FIT')))
    b1 = {condition: baselines_b1.B1Heads.fit(world.s1, world.s2, world.labels, dropped, params=SMALL, k_condition=condition) for condition in head_rows.K_CONDITIONS}
    yield bundle, model, stats, b1
    del world
    gc.collect()


def test_run_inference_has_the_k_condition_and_the_gold_reps_source_after_the_winner_truth_with_the_old_defaults():
    from c_ept.evaluation import adapters
    parameters = inspect.signature(adapters.run_inference).parameters
    names = list(parameters)                                                       # relative order: a later keyword (M11) may follow these
    assert names.index('winner_truth') == names.index('kill_law') + 1 and names.index('k_condition') == names.index('winner_truth') + 1
    assert names.index('gold_reps') == names.index('k_condition') + 1
    assert parameters['k_condition'].default == 'raw' and parameters['gold_reps'].default == 'labels'
    assert parameters['k_condition'].kind is parameters['gold_reps'].kind is inspect.Parameter.KEYWORD_ONLY
    assert adapters.GOLD_REP_SOURCES == ('labels', 'in_scope_fit')


def test_run_inference_refuses_a_k_condition_or_a_representative_source_that_does_not_fit_before_any_store_is_read():
    from c_ept.evaluation.adapters import run_inference
    bundle = SimpleNamespace(role='DEV')                                                   # no stores: a read would be an AttributeError, not the ValueError asked for
    b1 = SimpleNamespace(fit_roles=('FIT',), stop_roles=('STOP',))
    common = dict(check=lambda: None, device='cpu')
    for kwargs, name in (({'k_condition': 'continuous'}, 'k_condition'), ({'k_condition': None}, 'k_condition'), ({'gold_reps': 'in_scope'}, 'gold_reps'),
                         ({'gold_reps': None}, 'gold_reps'), ({'k_condition': 'binned'}, 'gold_reps')):
        with pytest.raises(ValueError, match=name):
            run_inference(bundle, object(), object(), b1, **common, **kwargs)
    # rows trained on the representatives and draws made with other representatives would leave the support of the heads: refused, the other way round is a question of evaluation only
    with pytest.raises(ValueError, match="k_condition 'binned' needs gold_reps 'in_scope_fit'"):
        run_inference(bundle, object(), object(), b1, **common, k_condition='binned', gold_reps='labels')
    # the B1 heads say the condition they were fitted on, in both directions
    b1.k_condition = 'binned'
    with pytest.raises(ValueError, match='k_condition'):
        run_inference(bundle, object(), object(), b1, **common)                                                    # the raw gold asked of heads fitted on the binned gold
    b1.k_condition = 'raw'
    with pytest.raises(ValueError, match='k_condition'):
        run_inference(bundle, object(), object(), b1, **common, k_condition='binned', gold_reps='in_scope_fit')      # the binned gold asked of heads fitted on the raw gold
    del b1.k_condition                                                                                              # a set without the attribute is a raw set
    with pytest.raises(ValueError, match='k_condition'):
        run_inference(bundle, object(), object(), b1, **common, k_condition='binned', gold_reps='in_scope_fit')


def test_run_inference_under_the_binned_condition_scores_the_k_blocks_on_binned_rows_and_draws_and_truth_share_the_representatives_of_the_rows(inference_world, monkeypatch):
    from c_ept.evaluation import adapters
    bundle, model, stats, b1 = inference_world
    common = dict(check=lambda: None, device='cpu', draws=6, batch_size=4)
    raw = adapters.run_inference(bundle, model, stats, b1['raw'], **common)
    seen = {'draws': [], 'truth': []}
    real_sample, real_truth = adapters.sample_open_value, value.winner_truth

    def sample_spy(laws, objects, prices, gold_reps, **kw):
        seen['draws'].append(gold_reps)
        return real_sample(laws, objects, prices, gold_reps, **kw)

    def truth_spy(f0_unit, kill_gold_bins, gold_reps, obj_value, rule):
        seen['truth'].append(gold_reps)
        return real_truth(f0_unit, kill_gold_bins, gold_reps, obj_value, rule)
    monkeypatch.setattr(adapters, 'sample_open_value', sample_spy)
    monkeypatch.setattr(value, 'winner_truth', truth_spy)
    binned = adapters.run_inference(bundle, model, stats, b1['binned'], k_condition='binned', gold_reps='in_scope_fit', winner_truth='binned', **common)
    monkeypatch.undo()
    # one source: the open draws of both models, the early check and the end of the binned truth all hold ONE array (the same object), and it is the representatives of the rows
    rows = data.ept_head_rows(bundle.s1, bundle.s2, bundle.labels, 'DEV', k_condition='binned')
    reps = rows.k_reps()
    assert len(seen['draws']) > 0 and len(seen['truth']) == 2
    everything = seen['draws'] + seen['truth']
    assert all(a is everything[0] for a in everything) and np.array_equal(everything[0], reps)
    assert not np.array_equal(everything[0], LABEL_REPS)                                  # not the labels/gold_reps of the bundle
    assert np.array_equal(reps, value.fit_gold_bin_means(bundle.s1, bundle.labels, population='in_scope').means)
    import hashlib
    assert binned['provenance']['k_condition'] == 'binned' and binned['provenance']['gold_reps'] == 'in_scope_fit'
    assert binned['provenance']['gold_reps_sha256'] == hashlib.sha256(np.ascontiguousarray(reps).tobytes()).hexdigest() == value.gold_reps_sha256(reps)
    assert binned['provenance']['gold_reps_sha256'] == b1['binned'].k_reps_sha256 == rows.k_reps_sha256()               # the set, the rows and the evaluation hold the same 31 numbers
    # review m1: the bins that had no in-scope FIT entry are named (the middle of their range), as the S3 labels named theirs
    fallback = list(value.fit_gold_bin_means(bundle.s1, bundle.labels, population='in_scope').fallback)
    assert binned['provenance']['gold_reps_fallback'] == fallback == list(rows.k_reps_fallback()) and 0 < len(fallback) < 31
    assert all(isinstance(b, int) for b in binned['provenance']['gold_reps_fallback'])
    assert 'k_condition' not in raw['provenance'] and 'gold_reps' not in raw['provenance'] and 'gold_reps_sha256' not in raw['provenance']
    assert 'gold_reps_fallback' not in raw['provenance']
    by_name = lambda out: {r['name']: r for r in out['nll']}                              # noqa: E731
    a, b = by_name(raw), by_name(binned)
    assert set(a) == set(b)
    k_keys = {f'{name}/{kind}' for name, kind in K_BLOCKS}
    for key in a:
        if key in k_keys:
            assert np.isfinite(b[key]['nll_ept']).all() and np.isfinite(b[key]['nll_b1']).all() and len(b[key]['row']) > 0, key
        else:                                                                            # every other block is scored exactly as before (the EPT is the same model, the B1 heads equal)
            assert np.array_equal(a[key]['nll_ept'], b[key]['nll_ept']) and np.array_equal(a[key]['nll_b1'], b[key]['nll_b1']), key
    assert any(not np.array_equal(a[key]['nll_ept'], b[key]['nll_ept']) for key in k_keys)       # the K blocks are scored on the binned K
    assert any(not np.array_equal(a[key]['nll_b1'], b[key]['nll_b1']) for key in k_keys)
    positive = [e for e in binned['endpoints'] if e['name'] == 'M3/open/positive_value'][0]
    assert len(positive['row']) > 0 and np.isfinite(np.asarray(positive['p_ept'], dtype=float)).all()


def test_run_inference_refuses_binned_heads_fitted_on_other_representatives_or_on_unrecorded_ones_before_any_head_is_called(inference_world):
    """Review m4: a binned set fitted on one label store and evaluated on another would pass the condition check and score with other 31 numbers; the sha256 recorded in the set is
    compared with the representatives of this evaluation, before the Monte Carlo (the guard `check` is called once, at the start, and then not again)."""
    from c_ept.evaluation.adapters import run_inference
    bundle, model, stats, b1 = inference_world
    scaled = Overlay(bundle.labels, {'labels/kill_gold': np.asarray(bundle.labels.array('labels/kill_gold')) * 1.5})     # same bins, other means
    other = SimpleNamespace(**{**vars(bundle), 'labels': scaled})
    assert value.gold_reps_sha256(value.fit_gold_bin_means(other.s1, other.labels, population='in_scope').means) != b1['binned'].k_reps_sha256
    unrecorded = baselines_b1.B1Heads(b1['binned'].masked, b1['binned'].open, b1['binned'].dropped, k_condition='binned')
    for label, bundle_, heads_, said in (('other store', other, b1['binned'], 'fitted on other representatives'), ('unrecorded', bundle, unrecorded, 'record no k_reps_sha256')):
        calls = []
        with pytest.raises(ValueError, match=said):                                  # each case says what it is: a mismatch is not an unrecorded set
            run_inference(bundle_, model, stats, heads_, check=lambda: calls.append(1), device='cpu', draws=6, batch_size=4, k_condition='binned', gold_reps='in_scope_fit')
        assert len(calls) == 1, label
    # the set with its own store passes (the run itself is the test above)
    run_inference(bundle, model, stats, b1['binned'], check=lambda: None, device='cpu', draws=2, batch_size=4, k_condition='binned', gold_reps='in_scope_fit')


def test_the_representative_source_is_independent_of_the_condition_of_the_rows_and_the_defaults_change_nothing(inference_world, monkeypatch):
    """gold_reps='in_scope_fit' with the raw rows moves the draws (and the binned truth) only: the NLL rows of the raw run stay; the default call equals the explicit raw/labels call.
    (What this proves for the default: the keywords at their defaults are the explicit raw/labels; the comparison with the code before the task is the proof script.)"""
    from c_ept.evaluation import adapters
    run_inference = adapters.run_inference
    bundle, model, stats, b1 = inference_world
    common = dict(check=lambda: None, device='cpu', draws=6, batch_size=4)
    seen = []
    real_sample = adapters.sample_open_value

    def sample_spy(laws, objects, prices, gold_reps, **kw):
        seen.append(gold_reps)
        return real_sample(laws, objects, prices, gold_reps, **kw)
    monkeypatch.setattr(adapters, 'sample_open_value', sample_spy)
    default = run_inference(bundle, model, stats, b1['raw'], **common)
    explicit = run_inference(bundle, model, stats, b1['raw'], k_condition='raw', gold_reps='labels', **common)
    n_default = len(seen) // 2
    scoped = run_inference(bundle, model, stats, b1['raw'], gold_reps='in_scope_fit', **common)
    monkeypatch.undo()
    in_scope = value.fit_gold_bin_means(bundle.s1, bundle.labels, population='in_scope').means
    assert n_default > 0 and len(seen) == 3 * n_default
    assert all(np.array_equal(a, LABEL_REPS) for a in seen[:2 * n_default])                  # the default draws read labels/gold_reps, as in every earlier run
    assert all(np.array_equal(a, in_scope) for a in seen[2 * n_default:]) and not np.array_equal(in_scope, LABEL_REPS)     # the in-scope source moves the draws
    assert default['provenance'] == explicit['provenance'] and 'gold_reps' not in default['provenance'] and 'gold_reps_fallback' not in default['provenance']
    assert scoped['provenance']['gold_reps'] == 'in_scope_fit' and 'k_condition' not in scoped['provenance'] and len(scoped['provenance']['gold_reps_sha256']) == 64
    assert scoped['provenance']['gold_reps_fallback'] == list(value.fit_gold_bin_means(bundle.s1, bundle.labels, population='in_scope').fallback)
    for x, y, z in zip(default['nll'], explicit['nll'], scoped['nll']):
        assert x['name'] == y['name'] == z['name']
        assert np.array_equal(x['nll_ept'], y['nll_ept']) and np.array_equal(x['nll_b1'], y['nll_b1'])
        assert np.array_equal(x['nll_ept'], z['nll_ept']) and np.array_equal(x['nll_b1'], z['nll_b1'])         # the raw rows are scored as before whatever the draws use
    for x, y in zip(default['endpoints'], explicit['endpoints']):
        assert x['name'] == y['name'] and all(np.array_equal(np.asarray(x[k], dtype=float), np.asarray(y[k], dtype=float), equal_nan=True) for k in ('p_ept', 'p_b1') if k in x)


def test_the_identity_check_names_the_k_condition_when_it_is_the_one_key_that_differs():
    """The rows digest differs between the conditions as well, so the run test above could pass without the identity key. Here the key is the only difference: it is absent in the
    identity of a raw run and ``_check_identity`` compares the union of the keys, so an identity with the key is refused against one without it, and the other way round."""
    raw, binned = {'seed': 3, 'epochs': 2}, {'seed': 3, 'epochs': 2, 'k_condition': 'binned'}
    for saved, now in ((raw, binned), (binned, raw)):
        with pytest.raises(ValueError, match=r"\['k_condition'\] differ"):
            train._check_identity(saved, now)
    train._check_identity(binned, dict(binned))
    train._check_identity(raw, dict(raw))
    # review m4: the representatives are a key of the identity of their own: the same condition with other 31 numbers is another run
    one, two = {**binned, 'k_reps_sha256': 'a' * 64}, {**binned, 'k_reps_sha256': 'b' * 64}
    with pytest.raises(ValueError, match=r"\['k_reps_sha256'\] differ"):
        train._check_identity(one, two)


# =====================================================================================================================================
# B0 follows the K condition (S4b preparation M3b, decision D24)
# =====================================================================================================================================
# The open object head (M3Obj) and the open conversion head (M4Conv) of B0 read the K block in their designs; the masked heads of B0 do not (plan P6), so they cannot move.
# B0 has no STOP role: its fit has one set of rows, so "the fit and the STOP rows agree" holds by construction; what can disagree is the rows that a fitted set is scored on.
N_K_COLUMNS = 4                                                        # kills blue and red, kill gold blue and red (in thousands): the K block of the design


@pytest.fixture(scope='module')
def b0_heads(heads_world):
    """B0 fitted on the heads world under both K conditions."""
    w = heads_world
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')                                # the toy heads do not converge: the known warnings of the B0 tests
        fits = {condition: baselines_b0.B0Heads.fit(w.s1, w.s2, w.labels, w.fit_means, k_condition=condition) for condition in head_rows.K_CONDITIONS}
    return SimpleNamespace(world=w, b0=fits)


def b0_rows(heads, role, k_condition=None, labels=None):
    kw = {} if k_condition is None else {'k_condition': k_condition}
    w = heads.world
    return baselines_b0.HeadRows(w.s1, w.s2, w.labels if labels is None else labels, w.fit_means, role, **kw)


def same_arrays(a, b) -> bool:
    return set(a) == set(b) and all(np.array_equal(a[k], b[k]) for k in a)


def test_binned_b0_open_rows_carry_the_representative_of_the_gold_bin_and_zero_for_a_team_without_a_kill_as_the_b1_rows_do(b0_heads, b1_heads):
    """Oracle: written from the text of the rule (HR-9) on the arrays of the toy world: rep[gold bin] for a team with a kill, 0 for a team without one, in thousands."""
    w = b0_heads.world
    reps = value.fit_gold_bin_means(w.s1, w.labels, population='in_scope').means
    kills = np.asarray(w.s1.array('units/K'))[:, :2]
    bins = np.asarray(w.labels.array('labels/gold_bin'))
    for head, tail in (('M3Obj', 0), ('M4Conv', baselines_b0.N_CELLS)):
        d, raw = b0_rows(b0_heads, 'FIT', 'binned').data(head, 'open'), b0_rows(b0_heads, 'FIT').data(head, 'open')
        start = d.layout.n_numeric - tail - N_K_COLUMNS                 # the object head ends with K, the conversion head with K and O
        K = d.X[:, start:start + N_K_COLUMNS]
        want_gold = np.where(kills[d.row] >= 1, reps[np.clip(bins[d.row], 0, len(reps) - 1)], 0.0) / 1000.0
        assert np.array_equal(K[:, :2], kills[d.row]) and np.array_equal(K[:, 2:], want_gold), head
        assert (kills[d.row] == 0).any() and (want_gold[kills[d.row] == 0] == 0).all()               # the control: there are teams without a kill, and they get 0
        assert not np.array_equal(raw.X[:, start + 2:start + 4], want_gold), head                    # the control: the raw gold is the true gold, not a representative
        assert np.array_equal(raw.X[:, start:start + 2], K[:, :2]) and np.array_equal(raw.y, d.y)    # the kills and the targets do not move
        assert np.array_equal(K, b1_rows(b1_heads, 'FIT', 'binned').parts(head, 'open').K), head     # one K block for B0 and for B1


def test_the_binned_condition_changes_the_design_of_two_open_b0_heads_and_of_nothing_else(b0_heads):
    raw, binned = b0_rows(b0_heads, 'FIT'), b0_rows(b0_heads, 'FIT', 'binned')
    changed = []
    for kind, names in (('masked', head_rows.HEAD_NAMES), ('open', head_rows.OPEN_HEAD_NAMES)):
        for name in names:
            a, b = raw.data(name, kind), binned.data(name, kind)
            assert np.array_equal(a.y, b.y) and np.array_equal(a.row, b.row) and a.layout == b.layout, (name, kind)
            if not np.array_equal(a.X, b.X):
                changed.append((name, kind))
    assert changed == [('M3Obj', 'open'), ('M4Conv', 'open')]


def test_b0_heads_fitted_on_the_binned_condition_change_the_two_open_k_heads_only_and_record_the_condition(b0_heads):
    raw, binned = b0_heads.b0['raw'], b0_heads.b0['binned']
    assert raw.k_condition == 'raw' and binned.k_condition == 'binned' and raw.k_reps_sha256 is None
    for kind in ('masked', 'open'):
        for name, head in getattr(binned, kind).items():
            same = same_arrays(head.state()[0], getattr(raw, kind)[name].state()[0])
            assert same == ((name, kind) not in [('M3Obj', 'open'), ('M4Conv', 'open')]), (kind, name)         # a head is the head of the raw fit unless its design holds K
    reps = b0_rows(b0_heads, 'FIT', 'binned').k_reps()
    assert binned.k_reps_sha256 == value.gold_reps_sha256(reps) and len(binned.k_reps_sha256) == 64


def test_b0_save_and_load_round_trip_the_condition_and_the_sha256_and_record_them_only_when_binned(b0_heads, tmp_path):
    raw, binned = b0_heads.b0['raw'], b0_heads.b0['binned']
    binned.save(tmp_path / 'binned')
    raw.save(tmp_path / 'raw')
    text_r = (tmp_path / 'raw' / baselines_b0.JSON_NAME).read_text(encoding='utf-8')
    doc_b = json.loads((tmp_path / 'binned' / baselines_b0.JSON_NAME).read_text(encoding='utf-8'))
    assert 'k_condition' not in text_r and 'k_reps_sha256' not in text_r                              # absent means the raw gold: the file of every earlier set
    assert doc_b['k_condition'] == 'binned' and doc_b['k_reps_sha256'] == binned.k_reps_sha256
    back, back_raw = baselines_b0.B0Heads.load(tmp_path / 'binned'), baselines_b0.B0Heads.load(tmp_path / 'raw')
    assert (back.k_condition, back.k_reps_sha256) == ('binned', binned.k_reps_sha256) and (back_raw.k_condition, back_raw.k_reps_sha256) == ('raw', None)
    w = b0_heads.world
    for kind in ('masked', 'open'):
        got, want = back.nll(w.s1, w.s2, w.labels, 'DEV', kind), binned.nll(w.s1, w.s2, w.labels, 'DEV', kind)
        assert all(np.array_equal(got[k].nll, want[k].nll) for k in want), kind
    for change, match in (({'k_condition': 'nb'}, 'k_condition'), ({'k_reps_sha256': 'abc'}, 'k_reps_sha256')):         # another name, a malformed sha256
        (tmp_path / 'edit').mkdir(exist_ok=True)
        for name in (baselines_b0.NPZ_NAME, baselines_b0.JSON_NAME):
            (tmp_path / 'edit' / name).write_bytes((tmp_path / 'binned' / name).read_bytes())
        (tmp_path / 'edit' / baselines_b0.JSON_NAME).write_text(json.dumps({**doc_b, **change}), encoding='utf-8')
        with pytest.raises(ValueError, match=match):
            baselines_b0.B0Heads.load(tmp_path / 'edit')
    (tmp_path / 'edit' / baselines_b0.JSON_NAME).write_text(json.dumps({k: v for k, v in doc_b.items() if k != 'k_condition'}), encoding='utf-8')
    with pytest.raises(ValueError, match='k_reps_sha256'):                                             # a sha256 without the binned condition is the other direction of the same mismatch
        baselines_b0.B0Heads.load(tmp_path / 'edit')


def test_b0_scores_the_open_k_heads_on_the_rows_of_the_condition_it_was_fitted_on(b0_heads):
    w = b0_heads.world
    raw, binned = b0_heads.b0['raw'], b0_heads.b0['binned']
    got = {kind: binned.nll(w.s1, w.s2, w.labels, 'DEV', kind) for kind in ('masked', 'open')}
    other_score = []
    for name in ('M3Obj', 'M4Conv'):
        d = b0_rows(b0_heads, 'DEV', 'binned').data(name, 'open')
        want = binned.open[name].nll(d.X, d.y, d.e)
        assert np.array_equal(got['open'][name].nll, want), name
        other = b0_rows(b0_heads, 'DEV').data(name, 'open')                                            # the same head on the true gold
        other_score.append(not np.array_equal(binned.open[name].nll(other.X, other.y, other.e), want))
    assert all(other_score)                                                                           # the control: the condition of the scored rows matters
    # the heads that do not read K are the heads of the raw fit and are scored as such
    for name in ('M3Size', 'M3Kills', 'M3Gold'):
        assert np.array_equal(got['open'][name].nll, raw.nll(w.s1, w.s2, w.labels, 'DEV', 'open')[name].nll), name
    for name, scores in got['masked'].items():
        assert np.array_equal(scores.nll, raw.nll(w.s1, w.s2, w.labels, 'DEV', 'masked')[name].nll), name


def test_a_b0_set_refuses_rows_of_the_other_condition_in_both_directions_and_rows_with_other_representatives(b0_heads):
    w = b0_heads.world
    raw, binned = b0_heads.b0['raw'], b0_heads.b0['binned']
    with pytest.raises(ValueError, match='k_condition'):                                               # a raw set, rows of the binned gold
        raw._check_rows(b0_rows(b0_heads, 'DEV', 'binned'))
    with pytest.raises(ValueError, match='k_condition'):                                               # a binned set, rows of the true gold
        binned._check_rows(b0_rows(b0_heads, 'DEV'))
    binned._check_rows(b0_rows(b0_heads, 'DEV', 'binned'))
    raw._check_rows(b0_rows(b0_heads, 'DEV'))
    # the same condition from another label store is another K variable (other representatives): the set refuses it
    scaled = Overlay(w.labels, {'labels/kill_gold': np.asarray(w.labels.array('labels/kill_gold')) * 1.5})
    with pytest.raises(ValueError, match='k_reps_sha256'):
        binned._check_rows(b0_rows(b0_heads, 'DEV', 'binned', labels=scaled))
    with pytest.raises(ValueError, match='k_reps_sha256'):
        binned.nll(w.s1, w.s2, scaled, 'DEV', 'open')
    raw.nll(w.s1, w.s2, scaled, 'DEV', 'open')                                                         # the raw set has no representatives: any label store is a label store


def test_a_b0_set_checks_the_sha256_of_its_representatives_and_the_name_of_the_condition(b0_heads):
    w = b0_heads.world
    raw, binned = b0_heads.b0['raw'], b0_heads.b0['binned']
    sha = binned.k_reps_sha256
    with pytest.raises(ValueError, match='k_reps_sha256'):                                             # a raw set has no representatives
        baselines_b0.B0Heads(raw.masked, raw.open, w.fit_means, k_reps_sha256=sha)
    for bad in ('abc', sha.upper(), sha[:-1], 5, ''):
        with pytest.raises(ValueError, match='k_reps_sha256'):
            baselines_b0.B0Heads(binned.masked, binned.open, w.fit_means, k_condition='binned', k_reps_sha256=bad)
    assert baselines_b0.B0Heads(binned.masked, binned.open, w.fit_means, k_condition='binned').k_reps_sha256 is None
    with pytest.raises(ValueError, match='k_condition'):
        baselines_b0.B0Heads(raw.masked, raw.open, w.fit_means, k_condition='continuous')
    with pytest.raises(ValueError, match='k_condition'):                                               # refused before any array is read: no stores are given
        baselines_b0.B0Heads.fit(None, None, None, w.fit_means, k_condition='continuous')
    parameters = inspect.signature(baselines_b0.B0Heads.fit).parameters
    names = list(parameters)
    assert parameters['k_condition'].default == 'raw' and names.index('k_condition') == names.index('kill_law') + 1
    assert parameters['k_condition'].kind is inspect.Parameter.KEYWORD_ONLY
    parameters = inspect.signature(baselines_b0.B0Heads.__init__).parameters
    names = list(parameters)
    assert names.index('k_condition') == names.index('kill_law') + 1 and names.index('k_reps_sha256') == names.index('k_condition') + 1
    assert parameters['k_condition'].default == 'raw' and parameters['k_reps_sha256'].default is None


def test_the_default_b0_fit_equals_the_explicit_raw_fit_in_its_saved_files_and_records_no_condition(b0_heads, tmp_path):
    """What this proves: the keyword at its default is the explicit 'raw' (the files are equal bit for bit, and no key of the condition is written); the comparison with the code
    before the task is the proof script of the M3b report."""
    w = b0_heads.world
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        default = baselines_b0.B0Heads.fit(w.s1, w.s2, w.labels, w.fit_means)
    default.save(tmp_path / 'default')
    b0_heads.b0['raw'].save(tmp_path / 'explicit')
    for name in (baselines_b0.NPZ_NAME, baselines_b0.JSON_NAME):
        assert (tmp_path / 'default' / name).read_bytes() == (tmp_path / 'explicit' / name).read_bytes(), name
    assert 'k_condition' not in (tmp_path / 'default' / baselines_b0.JSON_NAME).read_text(encoding='utf-8') and default.k_condition == 'raw'


# =====================================================================================================================================
# run_inference(b0=): the NLL of B0 on exactly the rows the model is scored on (S4b preparation M11, decision D16)
# =====================================================================================================================================
STRUCTURAL = ('M2Rate/masked', 'M2Mark/masked', 'M2Zone/masked', 'M3Size/open', 'M3Kills/open', 'M3Gold/open', 'M3Obj/open', 'M4Conv/open')


@pytest.fixture(scope='module')
def inference_b0(inference_world):
    """B0 fitted on the FIT matches of the inference world under both K conditions (the means of the b0 columns of its FIT views)."""
    bundle, model, stats, b1 = inference_world
    means = features.b0_fit_means(bundle.s2, np.flatnonzero(features.split_views(bundle.s1, bundle.s2, 'FIT')))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')                                            # the toy heads do not converge: the known warnings of the B0 tests
        fits = {condition: baselines_b0.B0Heads.fit(bundle.s1, bundle.s2, bundle.labels, means, k_condition=condition) for condition in head_rows.K_CONDITIONS}
    return SimpleNamespace(b0=fits, means=means)


def same_result(a, b, path=''):
    """Equality of two ``run_inference`` results with no tolerance: arrays by dtype, shape and bytes, everything else by type and value."""
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), path
        for k in a:
            same_result(a[k], b[k], f'{path}/{k}')
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            same_result(x, y, f'{path}[{i}]')
    elif isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), path
    else:
        assert type(a) is type(b) and (a == b or (a != a and b != b)), path


def test_run_inference_has_the_b0_keyword_after_zone_mark_with_the_old_default():
    from c_ept.evaluation import adapters
    parameters = inspect.signature(adapters.run_inference).parameters
    names = list(parameters)
    assert parameters['b0'].default is None and parameters['b0'].kind is inspect.Parameter.KEYWORD_ONLY and names.index('b0') > names.index('zone_mark')


def test_without_b0_the_result_is_that_of_every_earlier_run_and_with_b0_only_the_b0_fields_are_added(inference_world, inference_b0):
    """The default is bit for bit the result before the task (the proof script compares it with the base commit): no ``nll_b0`` key in any record, no B0 in the provenance; explicit ``b0=None`` is the default.
    With B0 given, every other field (endpoints, the EPT and B1 NLL, the counts, the provenance but for its 'b0' entry) is that of the default."""
    from c_ept.evaluation import adapters
    bundle, model, stats, b1 = inference_world
    common = dict(check=lambda: None, device='cpu', draws=6, batch_size=4)
    default = adapters.run_inference(bundle, model, stats, b1['raw'], **common)
    same_result(default, adapters.run_inference(bundle, model, stats, b1['raw'], b0=None, **common))
    assert all('nll_b0' not in r for r in default['nll']) and 'b0' not in default['provenance']
    with_b0 = adapters.run_inference(bundle, model, stats, b1['raw'], b0=inference_b0.b0['raw'], **common)
    same_result(default['endpoints'], with_b0['endpoints'])
    assert [r['name'] for r in default['nll']] == [r['name'] for r in with_b0['nll']]
    for a, b in zip(default['nll'], with_b0['nll']):
        assert 'nll_b0' in b, a['name']
        same_result(a, {k: v for k, v in b.items() if k != 'nll_b0'}, a['name'])
    assert with_b0['provenance']['b0'] == {'kill_law': 'poisson', 'k_condition': 'raw', **inference_b0.b0['raw'].saved_sha256()}
    same_result(default['provenance'], {k: v for k, v in with_b0['provenance'].items() if k != 'b0'})


def test_b0_through_the_evaluation_path_scores_one_per_head(inference_world, inference_b0):
    """Review Focus of the plan, decision D16: B0 is 1 on every head. The NLL of B0 that the evaluation path writes (``nll_b0``) is, row for row, the HR-5 score of ``B0Heads.nll`` (the independent route:
    its own rows and its own HeadScores aggregation) on the rows that the model is scored on; used as the model's column it gives R_h = 1.0 exactly on every head and for the mean, and the
    real model's column gives a finite ratio that is the quotient of the two sums."""
    from c_ept.evaluation import adapters, inference, metrics
    bundle, model, stats, b1 = inference_world
    b0 = inference_b0.b0['raw']
    out = adapters.run_inference(bundle, model, stats, b1['raw'], b0=b0, check=lambda: None, device='cpu', draws=6, batch_size=4)
    by_name = {r['name']: r for r in out['nll']}
    for kind in ('masked', 'open'):
        for name, scores in b0.nll(bundle.s1, bundle.s2, bundle.labels, 'DEV', kind).items():
            record = by_name[f'{name}/{kind}']
            values, matches = scores.per_row()
            rows = np.unique(scores.row)
            keep = np.isin(rows, record['row'])                                     # the rows of B0's own aggregation that the model is scored on
            assert np.array_equal(record['row'], rows[keep]) and np.array_equal(record['match'], matches[keep]), (kind, name)
            assert len(record['nll_b0']) == len(record['row']) == len(record['nll_ept']) == len(record['nll_b1']), (kind, name)
            np.testing.assert_allclose(record['nll_b0'], values[keep], rtol=1e-12, atol=0, err_msg=f'{kind} {name}')
    ids = np.unique(np.concatenate([r['match'] for r in out['nll'] if len(r['match'])]))
    matrix = inference.resample_matrix(len(ids), B=60)
    as_model = [dict(r, nll_ept=r['nll_b0']) for r in out['nll']]
    one = metrics.relative_structural(as_model, matrix, ids=ids)
    assert one['status'] == 'available' and one['metrics']['ept']['estimate'] == 1.0 and one['metrics']['ept']['ci95'] == [1.0, 1.0]
    for name in STRUCTURAL:
        assert one['secondary']['per_head'][name]['metrics']['ept']['estimate'] == 1.0, name
    real = metrics.relative_structural(out['nll'], matrix, ids=ids)
    assert real['status'] == 'available'
    for name in STRUCTURAL:
        r = by_name[name]
        got = real['secondary']['per_head'][name]['metrics']['ept']['estimate']
        assert np.isfinite(got) and got > 0 and got != 1.0, name
        if len(ids) == 1:                                                           # one match: the weights are equal and R_h is the quotient of the two sums
            assert got == pytest.approx(r['nll_ept'].sum() / r['nll_b0'].sum(), rel=1e-12), name


def test_b0_is_scored_on_the_rows_of_the_model_including_the_zone_rows_without_a_zone_and_under_the_binned_condition(inference_world, inference_b0):
    """B0 M2Zone drops the rows without a zone (y < 0) exactly as ``loss.scored_rows`` does, and the masked M3Obj of B0 has no K (it is left out of the paired K check): the records of both models
    hold the same rows in every block, also under the binned condition with its representatives."""
    from c_ept.evaluation import adapters
    bundle, model, stats, b1 = inference_world
    for condition, kwargs in (('raw', {}), ('binned', {'k_condition': 'binned', 'gold_reps': 'in_scope_fit'})):
        out = adapters.run_inference(bundle, model, stats, b1[condition], b0=inference_b0.b0[condition], check=lambda: None, device='cpu', draws=2, batch_size=4, **kwargs)
        zone = next(r for r in out['nll'] if r['name'] == 'M2Zone/masked')
        assert len(zone['row']) > 0 and len(zone['nll_b0']) == len(zone['row'])
        for r in out['nll']:
            assert r['nll_b0'].shape == r['row'].shape and np.isfinite(r['nll_b0']).all(), (condition, r['name'])
        expected = {'kill_law': 'poisson', 'k_condition': condition, **inference_b0.b0[condition].saved_sha256()}
        if condition == 'binned':
            expected['k_reps_sha256'] = inference_b0.b0['binned'].k_reps_sha256
        assert out['provenance']['b0'] == expected


def test_run_inference_refuses_a_b0_of_another_kill_law_or_k_condition_or_unfitted_roles_before_any_store_is_read():
    """Both directions: the raw gold asked of a binned B0 and the binned gold asked of a raw B0 (the attribute absent means raw and plain), the other kill law either way; a B0 that was not fitted on FIT only
    or lacks heads is refused too. The stand-ins hold no store: any read would be an AttributeError, not the ValueError asked for."""
    from c_ept.evaluation.adapters import run_inference
    from c_ept.data.head_rows import HEAD_NAMES, OPEN_HEAD_NAMES
    bundle = SimpleNamespace(role='DEV')
    b1 = SimpleNamespace(fit_roles=('FIT',), stop_roles=('STOP',))
    common = dict(check=lambda: None, device='cpu')

    def b0(**kw):
        return SimpleNamespace(**{'fit_roles': ('FIT',), 'masked': dict.fromkeys(HEAD_NAMES), 'open': dict.fromkeys(OPEN_HEAD_NAMES), **kw})
    with pytest.raises(ValueError, match='B0.*k_condition'):
        run_inference(bundle, object(), object(), b1, b0=b0(k_condition='binned'), **common)                                        # the raw gold asked of binned B0 heads
    b1.k_condition = 'binned'
    with pytest.raises(ValueError, match='B0.*k_condition'):
        run_inference(bundle, object(), object(), b1, b0=b0(k_condition='raw'), k_condition='binned', gold_reps='in_scope_fit', **common)       # the binned gold asked of raw B0 heads
    with pytest.raises(ValueError, match='B0.*k_condition'):
        run_inference(bundle, object(), object(), b1, b0=b0(), k_condition='binned', gold_reps='in_scope_fit', **common)            # a set without the attribute is a raw set
    del b1.k_condition
    with pytest.raises(ValueError, match='B0.*kill_law'):
        run_inference(bundle, object(), object(), b1, b0=b0(kill_law='shifted'), **common)
    b1.kill_law = 'shifted'
    with pytest.raises(ValueError, match='B0.*kill_law'):
        run_inference(bundle, object(), object(), b1, b0=b0(kill_law='poisson'), kill_law='shifted', **common)
    del b1.kill_law
    with pytest.raises(ValueError, match='B0.*FIT'):
        run_inference(bundle, object(), object(), b1, b0=b0(fit_roles=('FIT', 'STOP')), **common)
    with pytest.raises(ValueError, match='B0'):
        run_inference(bundle, object(), object(), b1, b0=b0(masked={}), **common)
    with pytest.raises(ValueError, match='B0'):
        run_inference(bundle, object(), object(), b1, b0=object(), **common)


def test_run_inference_refuses_a_binned_b0_of_other_representatives_or_without_a_record_before_any_head_is_called(inference_world, inference_b0):
    """As for B1 (review m4), now for B0: the sha256 of the representatives the binned B0 was fitted on must equal that of this evaluation's ``in_scope_fit`` array, and a binned B0 that records none is refused
    (both directions); the guard ``check`` is called once, at the start."""
    from c_ept.evaluation.adapters import run_inference
    bundle, model, stats, b1 = inference_world
    good = inference_b0.b0['binned']
    scaled = Overlay(bundle.labels, {'labels/kill_gold': np.asarray(bundle.labels.array('labels/kill_gold')) * 1.5})            # same bins, other means
    other = SimpleNamespace(**{**vars(bundle), 'labels': scaled})
    wrong = baselines_b0.B0Heads(good.masked, good.open, good.fit_means, k_condition='binned', k_reps_sha256='0' * 64)
    unrecorded = baselines_b0.B0Heads(good.masked, good.open, good.fit_means, k_condition='binned')
    assert unrecorded.k_reps_sha256 is None and wrong.k_reps_sha256 != good.k_reps_sha256
    for label, bundle_, heads_, said in (('wrong sha', bundle, wrong, 'B0.*other representatives'), ('unrecorded', bundle, unrecorded, 'B0.*no k_reps_sha256')):
        calls = []
        with pytest.raises(ValueError, match=said):
            run_inference(bundle_, model, stats, b1['binned'], b0=heads_, check=lambda: calls.append(1), device='cpu', draws=6, batch_size=4, k_condition='binned', gold_reps='in_scope_fit')
        assert len(calls) == 1, label
    assert value.gold_reps_sha256(value.fit_gold_bin_means(other.s1, other.labels, population='in_scope').means) != good.k_reps_sha256        # the store of the other sha is another store
    run_inference(bundle, model, stats, b1['binned'], b0=good, check=lambda: None, device='cpu', draws=2, batch_size=4, k_condition='binned', gold_reps='in_scope_fit')


def test_a_b0_scored_on_other_rows_than_the_model_is_refused(inference_world, inference_b0, monkeypatch):
    """The rows are proved equal, not assumed: a B0 whose rows differ in a target (here one changed count) is a ValueError naming the block, as the pairing of B1 is."""
    from c_ept.evaluation import adapters
    bundle, model, stats, b1 = inference_world
    b0 = inference_b0.b0['raw']
    real = b0.head_rows

    def other_rows(*args, **kw):
        rows = real(*args, **kw)

        class Proxy:
            def __getattr__(self, name):
                return getattr(rows, name)

            def data(self, head, kind):
                d = rows.data(head, kind)
                if (head, kind) == ('M3Size', 'open'):
                    y = d.y.copy()
                    y[0] = (y[0] + 1) % 4
                    return d._replace(y=y)
                return d
        return Proxy()
    monkeypatch.setattr(b0, 'head_rows', other_rows)
    with pytest.raises(ValueError, match='M3Size/open'):
        adapters.run_inference(bundle, model, stats, b1['raw'], b0=b0, check=lambda: None, device='cpu', draws=2, batch_size=4)


def test_with_the_zone_mark_off_the_b0_zone_block_is_empty_with_the_reason_and_the_relative_metric_is_unavailable(inference_world, inference_b0):
    from c_ept.evaluation import adapters, inference, metrics
    bundle, model, stats, b1 = inference_world
    torch.manual_seed(17)
    off = heads.EPT(encoder.EncoderConfig(d=16, layers=2, heads=2, dropout=.1, kappas=1, field_channels=2), toy_static(), zone_mark=False)
    out = adapters.run_inference(bundle, off, stats, b1['raw'], b0=inference_b0.b0['raw'], zone_mark=False, check=lambda: None, device='cpu', draws=2, batch_size=4)
    zone = next(r for r in out['nll'] if r['name'] == 'M2Zone/masked')
    assert zone['counts']['unavailable_reason'] == 'zone_mark_off' and len(zone['nll_b0']) == 0 and len(zone['nll_ept']) == 0
    ids = np.unique(np.concatenate([r['match'] for r in out['nll'] if len(r['match'])]))
    rel = metrics.relative_structural(out['nll'], inference.resample_matrix(len(ids), B=20), ids=ids)
    assert rel['status'] == 'unavailable' and 'M2Zone/masked' in rel['reason'] and 'zone_mark_off' in rel['reason'] and rel['metrics'] is None
    assert rel['secondary']['per_head']['M2Zone/masked']['reason'] == 'zone_mark_off'
    assert all(rel['secondary']['per_head'][n]['status'] == 'available' for n in STRUCTURAL if n != 'M2Zone/masked')


def test_a_b0_scored_on_other_views_than_the_model_is_refused(inference_world, inference_b0, monkeypatch):
    """M11 review m5: the design of B0 is built from the S2 views of its rows, so the view numbers are proved equal to those of the model's rows, as the pairing of B1 does."""
    from c_ept.evaluation import adapters
    bundle, model, stats, b1 = inference_world
    b0 = inference_b0.b0['raw']
    real = b0.head_rows

    def other_views(*args, **kw):
        rows = real(*args, **kw)

        class Proxy:
            def __getattr__(self, name):
                return getattr(rows, name)

            def view_numbers(self, head, kind):
                v = np.array(rows.view_numbers(head, kind))
                if (head, kind) == ('M3Size', 'open'):
                    v[0] = v[-1] if v[0] != v[-1] else v[0] + 1
                return v
        return Proxy()
    monkeypatch.setattr(b0, 'head_rows', other_views)
    with pytest.raises(ValueError, match='M3Size/open.*view'):
        adapters.run_inference(bundle, model, stats, b1['raw'], b0=b0, check=lambda: None, device='cpu', draws=2, batch_size=4)
