"""EPT-T001 synthetic scalars and temporary toy checkpoints only."""
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from c_ept.ept import checkpoints as cp, epoch_metrics as em


def fixture_state(stops=(4., 3.), *, horizon=4, patience=2):
    best = min(range(len(stops)), key=stops.__getitem__)
    n = len(stops)
    early = n - best - 1 >= patience
    head = {'schema': cp.METRICS_SCHEMA, 'epoch': n, 'epochs': horizon, 'seed': 7,
            'identity': {'patience': patience, 'seed': 7, 'epochs': horizon},
            'done': early or n == horizon, 'stopped_early': early, 'best_epoch': best,
            'best': {'folder': f'epoch_{best+1:04d}', 'bytes': 4, 'sha256': cp.sha256(b'best')},
            'timing': {'steps': [2] * n, **{key: [0.] * n for key in cp.COMPONENT_FIELDS}}}
    loop = {'epoch': n, 'fit_batch_mean_losses': [-.5 - i for i in range(n)], 'stop_losses': list(stops),
            'steps': [2] * n, 'lrs': [.01, .02] * n, 'best_epoch': best,
            'bad': n-best-1, 'best_value': min(stops), 'stopped_early': early}
    return head, loop


def save(root, stops=(4., 3.), **kwargs):
    head, loop = fixture_state(stops, **kwargs)
    files = {'last.pt': b'last', 'best.pt': b'best', **em.encode(loop, head)}
    return cp.save_checkpoint(root, len(stops), files, head)


def rebind(folder, head, doc):
    raw = {em.FILES[0]: em.json_bytes(doc), em.FILES[1]: em.csv_bytes(doc['rows'])}
    for name, payload in raw.items():
        (folder/name).write_bytes(payload)
        head['epoch_metrics']['files'][name] = {'bytes': len(payload), 'sha256': cp.sha256(payload)}
    (folder/'head.json').write_bytes(cp.json_bytes(head))


def test_exact_values_csv_parity_hashes_and_head_last(tmp_path):
    head, loop = fixture_state()
    seen = []
    def writer(path, raw):
        seen.append(path.name)
        cp.write_atomic(path, raw)
    folder, head, digest = cp.save_checkpoint(tmp_path, 2, {'last.pt': b'last', **em.encode(loop, head)}, head, write=writer)
    assert seen == ['last.pt', *em.FILES, 'head.json']
    assert digest == cp.sha256((folder/'head.json').read_bytes())
    _, _, doc = cp.latest_metadata(tmp_path)
    assert doc['rows'] == [dict(zip(em.COLUMNS, (1, 0, -.5, 4., .02))), dict(zip(em.COLUMNS, (2, 1, -1.5, 3., .02)))]
    parsed = list(csv.DictReader(io.StringIO((folder/em.FILES[1]).read_text(encoding='utf-8'))))
    assert [float(row['stop_loss']) for row in parsed] == [4., 3.]
    assert doc['definitions']['fit_batch_mean_loss'].startswith('arithmetic mean of FIT batch losses in training mode')
    assert 'not a mean of batch means' in doc['definitions']['stop_loss']
    for name in em.FILES:
        payload = (folder/name).read_bytes()
        assert head['epoch_metrics']['files'][name] == {'bytes': len(payload), 'sha256': cp.sha256(payload)}


@pytest.mark.parametrize('stops,horizon,reason,best', [((4.,3.),4,'in_progress',1), ((4.,3.),2,'max_epochs',1), ((-2.,-2.,-1.),3,'early_stopping',0)])
def test_strict_improvement_patience_and_termination(stops, horizon, reason, best):
    head, loop = fixture_state(stops, horizon=horizon)
    doc = em.snapshot(loop, head)
    assert doc['termination_reason'] == reason and doc['best_epoch_index'] == best
    assert doc['best_epoch'] == best+1 and doc['done'] == (reason != 'in_progress')


@pytest.mark.parametrize('key', ['fit_batch_mean_losses','stop_losses','lrs'])
@pytest.mark.parametrize('value', [True, float('nan'), float('inf'), -float('inf')])
def test_bool_and_nonfinite_not_published(key, value):
    head, loop = fixture_state()
    loop[key][-1] = value
    with pytest.raises(RuntimeError, match='epoch metrics'):
        em.encode(loop, head)


@pytest.mark.parametrize('damage', ['missing_json','missing_csv','changed_csv','changed_json','binding','format_marker','downgrade_without_binding','traversal','identity','epoch','short_history','selection','done','lr_bool','oversize'])
def test_new_format_corruption_fails_before_payload_access(tmp_path, monkeypatch, damage):
    save(tmp_path, (4.,))
    folder, head, _ = save(tmp_path)
    doc = em.read_snapshot(folder, head)
    if damage.startswith('missing_'):
        (folder/em.FILES[0 if damage.endswith('json') else 1]).unlink()
    elif damage.startswith('changed_'):
        (folder/em.FILES[0 if damage.endswith('json') else 1]).write_bytes(b'bad')
    elif damage == 'binding':
        head.pop('epoch_metrics')
    elif damage == 'format_marker':
        head['schema'] = cp.SCHEMA
    elif damage == 'downgrade_without_binding':
        head['schema'] = cp.SCHEMA
        head.pop('epoch_metrics')
    elif damage == 'traversal':
        head['epoch_metrics']['files']['../escape'] = head['epoch_metrics']['files'].pop(em.FILES[0])
    elif damage == 'oversize':
        head['epoch_metrics']['files'][em.FILES[0]]['bytes'] = em.MAX_BYTES+1
    else:
        if damage == 'identity': doc['identity'] = {**doc['identity'], 'seed': 8}
        if damage == 'epoch': doc['rows'][-1]['epoch_index'] = True
        if damage == 'short_history': doc['rows'].pop()
        if damage == 'selection': doc['best_epoch_index'] = 0
        if damage == 'done': doc['done'] = True
        if damage == 'lr_bool': doc['rows'][-1]['lr_last_step'] = True
        rebind(folder, head, doc)
    (folder/'head.json').write_bytes(cp.json_bytes(head))
    monkeypatch.setattr(cp, 'read_verified', lambda *a: pytest.fail('payload reader reached'))
    with pytest.raises(RuntimeError):
        cp.latest_checkpoint(tmp_path)


@pytest.mark.parametrize('cut', [*em.FILES,'head.json'])
def test_torn_write_retry_publishes_once_and_old_snapshot_is_immutable(tmp_path, cut):
    old, _, _ = save(tmp_path, (4.,))
    old_bytes = {p.name:p.read_bytes() for p in old.iterdir()}
    head, loop = fixture_state()
    def writer(path, payload):
        if path.name == cut:
            raise OSError('synthetic cut')
        cp.write_atomic(path, payload)
    with pytest.raises(OSError):
        cp.save_checkpoint(tmp_path, 2, {'last.pt':b'last', **em.encode(loop,head)}, head, write=writer)
    assert cp.latest_metadata(tmp_path)[0] == old
    new, _, _ = save(tmp_path)
    assert len(cp.latest_metadata(tmp_path)[2]['rows']) == 2
    assert old_bytes == {p.name:p.read_bytes() for p in old.iterdir()}
    with pytest.raises(FileExistsError):
        save(tmp_path)
    assert (new/'head.json').is_file()


def test_metadata_reader_is_stdlib_only_and_never_opens_pt(tmp_path):
    save(tmp_path)
    code = """
import sys
from pathlib import Path
from c_ept.ept import checkpoints
real = Path.open
def no_payload(self, *args, **kwargs):
    assert self.suffix != '.pt', self.name
    return real(self, *args, **kwargs)
Path.open = no_payload
folder, head, doc = checkpoints.latest_metadata(Path(sys.argv[1]))
assert len(doc['rows']) == 2
assert not any(name in sys.modules for name in ('torch', 'numpy', 'c_ept.cohort'))
"""
    result = subprocess.run([sys.executable,'-B','-c',code,str(tmp_path)], capture_output=True,text=True,encoding='utf-8',env=os.environ.copy(),timeout=30)
    assert result.returncode == 0, result.stderr


def test_v5_history_unavailable_and_tensor_bytes_readable(tmp_path):
    folder, head, _ = save(tmp_path)
    head['schema'] = cp.SCHEMA
    head.pop('epoch_metrics')
    for name in em.FILES:
        (folder/name).unlink()
    (folder/'head.json').write_bytes(cp.json_bytes(head))
    assert cp.latest_metadata(tmp_path)[2] is None
    assert cp.latest_checkpoint(tmp_path)[2:] == (b'last', b'best')


def test_rebound_csv_must_match_json_even_with_valid_hash(tmp_path):
    folder, head, _ = save(tmp_path)
    raw = (folder/em.FILES[1]).read_bytes().replace(b'-0.5', b'-9.5')
    (folder/em.FILES[1]).write_bytes(raw)
    head['epoch_metrics']['files'][em.FILES[1]] = {'bytes':len(raw),'sha256':cp.sha256(raw)}
    (folder/'head.json').write_bytes(cp.json_bytes(head))
    with pytest.raises(RuntimeError): cp.latest_metadata(tmp_path)


@pytest.mark.parametrize('legacy_resume', [False, True])
def test_loop_exports_known_batch_mean_and_resume_finished_no_mutations(tmp_path, monkeypatch, legacy_resume):
    import torch
    from c_ept.ept import train
    from c_ept.tests import test_ept_train as helpers
    from c_ept.tests.ept_toys import build_ept_world
    world = build_ept_world(tmp_path/'toy', n_events=7, n_edges=5)
    stats = helpers.data.fit_stats(world.s1, world.s2)
    rows = {role: helpers.data.EngagementRows(helpers.data.ept_head_rows(world.s1,world.s2,world.labels,role)) for role in ('FIT','STOP')}
    data = helpers.make_data(rows,stats)
    cfg = helpers.make_cfg(patience=2)
    assert len(data.fit) == 3
    values = iter([1.,4.,7., 2.,5.,8., 3.,6.,9.])
    monkeypatch.setattr(train.training_step,'optimize_batch',lambda *a,**k: torch.tensor(next(values)))
    stops = iter([-2., -2., -1.])
    monkeypatch.setattr(train,'evaluate_stop',lambda *a,**k: next(stops))
    out = tmp_path/'run'
    checks = []
    def interrupt():
        checks.append(1)
        if len(checks) == 2: raise OSError('synthetic interruption')
    with pytest.raises(OSError): helpers.run(out,data,cfg,epochs=5,check=interrupt)
    if legacy_resume:
        first = out/'epoch_0001'
        old_head = json.loads((first/'head.json').read_bytes())
        old_head['schema'] = cp.SCHEMA
        old_head.pop('epoch_metrics')
        for name in em.FILES: (first/name).unlink()
        (first/'head.json').write_bytes(cp.json_bytes(old_head))
    old = {p.name:p.read_bytes() for p in (out/'epoch_0001').iterdir()}
    result = helpers.run(out,data,cfg,epochs=5)
    doc = cp.latest_metadata(out)[2]
    assert result.fit_batch_mean_losses == [4.,5.,6.] and result.stop_losses == [-2.,-2.,-1.]
    assert [row['fit_batch_mean_loss'] for row in doc['rows']] == [4.,5.,6.]
    assert doc['termination_reason'] == 'early_stopping' and doc['best_epoch_index'] == 0
    assert old == {p.name:p.read_bytes() for p in (out/'epoch_0001').iterdir()}
    before = {p.relative_to(out):(p.read_bytes(),p.stat().st_mtime_ns) for p in out.rglob('*') if p.is_file()}
    monkeypatch.setattr(train,'_write_atomic',lambda *a: pytest.fail('finished run write'))
    helpers.run(out,data,cfg,epochs=5)
    assert before == {p.relative_to(out):(p.read_bytes(),p.stat().st_mtime_ns) for p in out.rglob('*') if p.is_file()}
    # A historical completed v5 remains untouched, with no history backfill.
    for folder in out.glob('epoch_[0-9][0-9][0-9][0-9]'):
        head = json.loads((folder/'head.json').read_bytes())
        head['schema'] = cp.SCHEMA
        head.pop('epoch_metrics', None)
        for name in em.FILES:
            if (folder/name).exists(): (folder/name).unlink()
        (folder/'head.json').write_bytes(cp.json_bytes(head))
    before = {p.relative_to(out):p.read_bytes() for p in out.rglob('*') if p.is_file()}
    helpers.run(out,data,cfg,epochs=5)
    assert cp.latest_metadata(out)[2] is None
    assert before == {p.relative_to(out):p.read_bytes() for p in out.rglob('*') if p.is_file()}


@pytest.mark.parametrize('artifact', list(em.FILES))
def test_writer_partial_file_then_error_leaves_uncommitted_retryable_epoch(tmp_path, artifact):
    head, loop = fixture_state((4.,), horizon=2)
    def partial(path, payload):
        if path.name == artifact:
            path.write_bytes(payload[:len(payload)//2])
            raise OSError('synthetic partial write')
        cp.write_atomic(path, payload)
    with pytest.raises(OSError):
        cp.save_checkpoint(tmp_path, 1, {'last.pt':b'last', **em.encode(loop,head)}, head, write=partial)
    assert cp.latest_metadata(tmp_path) is None
    assert not (tmp_path/'epoch_0001/head.json').exists()
    save(tmp_path, (4.,), horizon=2)
    assert cp.latest_metadata(tmp_path)[2]['epoch_count'] == 1


def test_successful_writer_return_with_partial_csv_never_commits_head(tmp_path):
    head, loop = fixture_state((4.,), horizon=2)
    def partial(path, payload):
        cp.write_atomic(path, payload[:-1] if path.name == em.FILES[1] else payload)
    with pytest.raises(RuntimeError):
        cp.save_checkpoint(tmp_path, 1, {'last.pt':b'last', **em.encode(loop,head)}, head, write=partial)
    assert not (tmp_path/'epoch_0001/head.json').exists()



@pytest.mark.parametrize('damage', ['missing_fit','short_stop','nan_fit','bool_stop','nan_lr','wrong_epoch','missing_steps','missing_best','nan_best','wrong_best'])
def test_legacy_partial_resume_refuses_invalid_restored_history_before_new_epoch(tmp_path, monkeypatch, damage):
    import torch
    from c_ept.ept import train
    from c_ept.tests import test_ept_train as helpers
    from c_ept.tests.ept_toys import build_ept_world
    world = build_ept_world(tmp_path/'toy',n_events=7,n_edges=5)
    stats = helpers.data.fit_stats(world.s1,world.s2)
    rows = {role:helpers.data.EngagementRows(helpers.data.ept_head_rows(world.s1,world.s2,world.labels,role)) for role in ('FIT','STOP')}
    data = helpers.make_data(rows,stats)
    out = tmp_path/'run'
    checks = []
    def cut():
        checks.append(1)
        if len(checks) == 2: raise OSError('synthetic cut')
    with pytest.raises(OSError): helpers.run(out,data,epochs=3,check=cut)
    folder, head, _ = cp.latest_metadata(out)
    # Only freshly generated toy state is deserialized to construct the corrupt fixture.
    state = torch.load(folder/'last.pt',weights_only=True)
    loop = state['loop']
    if damage == 'missing_fit': loop.pop('fit_batch_mean_losses')
    if damage == 'short_stop': loop['stop_losses'] = []
    if damage == 'nan_fit': loop['fit_batch_mean_losses'][0] = float('nan')
    if damage == 'bool_stop': loop['stop_losses'][0] = True
    if damage == 'nan_lr': loop['lrs'][0] = float('nan')
    if damage == 'wrong_epoch': loop['epoch'] = 0
    if damage == 'missing_steps': loop.pop('steps')
    if damage == 'missing_best': loop.pop('best_value')
    if damage == 'nan_best': loop['best_value'] = float('nan')
    if damage == 'wrong_best': loop['best_value'] -= 1.
    raw = cp.torch_bytes(state)
    (folder/'last.pt').write_bytes(raw)
    head['files']['last.pt'] = {'bytes':len(raw),'sha256':cp.sha256(raw)}
    head['schema'] = cp.SCHEMA
    head.pop('epoch_metrics')
    for name in em.FILES: (folder/name).unlink()
    (folder/'head.json').write_bytes(cp.json_bytes(head))
    before = {p.name:p.read_bytes() for p in folder.iterdir()}
    monkeypatch.setattr(train.training_step,'optimize_batch',lambda *a,**k:pytest.fail('new epoch work reached'))
    with pytest.raises(RuntimeError,match='epoch metrics'): helpers.run(out,data,epochs=3)
    assert not (out/'epoch_0002').exists()
    assert before == {p.name:p.read_bytes() for p in folder.iterdir()}


@pytest.mark.parametrize('best_value', [True, float('nan'),float('inf'),-99.,999.])
def test_resume_history_rejects_inconsistent_live_best_threshold(best_value):
    head, loop = fixture_state()
    loop['best_value'] = best_value
    with pytest.raises(RuntimeError,match='epoch metrics'):
        em.snapshot(loop,head)


def test_resume_learning_rate_offsets_bound_to_head_steps():
    head,loop = fixture_state()
    loop['steps'] = [1,3]  # same count/sum, different epoch boundary
    with pytest.raises(RuntimeError,match='epoch metrics'):
        em.snapshot(loop,head)


@pytest.mark.parametrize('remove_previous_files', [False,True])
def test_later_v5_head_refused_after_any_prior_v6(tmp_path, remove_previous_files):
    prior, _, _ = save(tmp_path,(4.,))
    folder, head, _ = save(tmp_path)
    head['schema'] = cp.SCHEMA
    head.pop('epoch_metrics')
    for name in em.FILES: (folder/name).unlink()
    if remove_previous_files:
        for name in em.FILES: (prior/name).unlink()
    (folder/'head.json').write_bytes(cp.json_bytes(head))
    with pytest.raises(RuntimeError,match='downgrade'):
        cp.latest_metadata(tmp_path)


# ---- the v7 head and the epoch-metrics-v2 snapshot: one kappa column for each decay scale under decay_groups='split' (S4b preparation M4, decision D6) ----
KAPPAS = [[5.0, 20.0], [5.25, 19.5], [5.5, 19.0]]


def fixture_state_v7(stops=(4., 3.), *, kappas=KAPPAS, n2=False, horizon=4, patience=2):
    """The fixture of a run under ``decay_groups='split'`` with two decay scales: the v7 head, an identity that names the policy and a loop with the record of kappa."""
    head, loop = fixture_state(stops, horizon=horizon, patience=patience)
    head['schema'] = em.HEAD_SCHEMA_V7
    head['identity'] = {**head['identity'], 'encoder': {'kappas': 2}, 'decay_groups': 'split', 'decay_groups_sha256': 'a' * 64}
    if n2:
        head['identity']['model_spec'] = {'architecture': 'n2_teamset_v1', 'n2_hidden': 8, 'prefetch': 0, 'precision': 'float32_strict'}
    loop['kappa'] = [None if n2 else list(kappas[i]) for i in range(len(stops))]
    return head, loop


def save_v7(root, stops=(4., 3.), **kwargs):
    head, loop = fixture_state_v7(stops, **kwargs)
    files = {'last.pt': b'last', 'best.pt': b'best', **em.encode(loop, head)}
    return cp.save_checkpoint(root, len(stops), files, head)


def test_the_v6_names_and_columns_are_unchanged_and_the_v7_ones_are_new():
    assert em.HEAD_SCHEMA == 'ept_train_checkpoint_v6' and em.SCHEMA == 'epoch-metrics-v1' and em.HEAD_SCHEMA_V7 == 'ept_train_checkpoint_v7'
    assert em.COLUMNS == ('epoch', 'epoch_index', 'fit_batch_mean_loss', 'stop_loss', 'lr_last_step') and em.HEAD_SCHEMAS == (em.HEAD_SCHEMA, em.HEAD_SCHEMA_V7, em.HEAD_SCHEMA_V8)     # v8 added by S4b preparation M6
    assert em.head_schema({'patience': 2}) == em.HEAD_SCHEMA and em.head_schema({'decay_groups': 'all'}) == em.HEAD_SCHEMA and em.head_schema({'decay_groups': 'split'}) == em.HEAD_SCHEMA_V7
    head, loop = fixture_state()                                       # a v6 head: no kappa column whatever the loop holds
    loop['kappa'] = [[1.0], [2.0]]
    doc = em.snapshot(loop, head)
    assert doc['schema'] == em.SCHEMA and all(set(row) == set(em.COLUMNS) for row in doc['rows'])
    assert em.csv_bytes(doc['rows']).decode('utf-8').split('\n')[0] == ','.join(em.COLUMNS) and em.DEFINITIONS.keys() < em.DEFINITIONS_V2.keys()


def test_v7_rows_carry_one_kappa_column_for_each_decay_scale_in_json_and_csv_and_survive_the_head_last_round_trip(tmp_path):
    folder, head, _ = save_v7(tmp_path)
    found_folder, found_head, doc = cp.latest_metadata(tmp_path)
    assert found_folder == folder and found_head == head and head['schema'] == em.HEAD_SCHEMA_V7
    columns = (*em.COLUMNS, 'kappa_0', 'kappa_1')
    assert doc['schema'] == em.SCHEMA_V2 and doc['definitions'] == em.DEFINITIONS_V2 and 'kappa_i' in doc['definitions']
    assert [dict(zip(columns, (1, 0, -.5, 4., .02, 5.0, 20.0))), dict(zip(columns, (2, 1, -1.5, 3., .02, 5.25, 19.5)))] == doc['rows']
    parsed = list(csv.DictReader(io.StringIO((folder / em.FILES[1]).read_text(encoding='utf-8'))))
    assert list(parsed[0]) == list(columns) and [float(row['kappa_0']) for row in parsed] == [5.0, 5.25] and [float(row['kappa_1']) for row in parsed] == [20.0, 19.5]
    assert em.read_snapshot(folder, head) == doc and head['epoch_metrics']['schema'] == em.SCHEMA_V2 == em.snapshot_schema(head)
    assert em.columns(head) == columns and em.columns(fixture_state()[0]) == em.COLUMNS and em.snapshot_schema(fixture_state()[0]) == em.SCHEMA
    assert cp.latest_checkpoint(tmp_path)[2:] == (b'last', b'best')


def test_v7_n2_rows_have_blank_kappa_cells_and_the_blank_follows_the_architecture_of_the_identity(tmp_path):
    from c_ept.comparison import models as comparison_models
    assert em.EPT_ARCHITECTURE == comparison_models.ARCHITECTURES[0]                 # the one name that has decay scales
    folder, head, _ = save_v7(tmp_path, n2=True)
    doc = cp.latest_metadata(tmp_path)[2]
    assert all(row['kappa_0'] is None and row['kappa_1'] is None for row in doc['rows'])
    parsed = list(csv.DictReader(io.StringIO((folder / em.FILES[1]).read_text(encoding='utf-8'))))
    assert all(row['kappa_0'] == '' and row['kappa_1'] == '' for row in parsed)
    head, loop = fixture_state_v7(n2=True)
    loop['kappa'] = [list(k) for k in KAPPAS[:2]]                       # numbers for N2: it has no decay scale
    with pytest.raises(RuntimeError, match='epoch metrics'):
        em.encode(loop, head)
    head, loop = fixture_state_v7()
    loop['kappa'][1] = None                                              # blank for an EPT: the record is incomplete
    with pytest.raises(RuntimeError, match='epoch metrics'):
        em.encode(loop, head)
    head, loop = fixture_state_v7()                                      # an EPT chosen through the model specification of the comparison has its scales
    head['identity']['model_spec'] = {'architecture': 'ept_s_v1', 'n2_hidden': 8, 'prefetch': 0, 'precision': 'float32_strict'}
    assert em.snapshot(loop, head)['rows'][0]['kappa_1'] == 20.0
    loop['kappa'][0] = None
    with pytest.raises(RuntimeError, match='epoch metrics'):
        em.snapshot(loop, head)


@pytest.mark.parametrize('damage', ['no_record', 'short_record', 'long_record', 'wrong_width', 'zero', 'negative', 'nan', 'inf', 'bool', 'string', 'not_a_list', 'identity_without_split',
                                    'split_identity_in_v6', 'no_kappa_count'])
def test_v7_refuses_a_record_of_kappa_that_is_missing_or_not_a_positive_finite_number_and_a_head_that_does_not_match_its_policy(damage):
    head, loop = fixture_state_v7()
    if damage == 'no_record':
        loop.pop('kappa')
    elif damage == 'short_record':
        loop['kappa'].pop()
    elif damage == 'long_record':
        loop['kappa'].append([5.0, 20.0])
    elif damage == 'wrong_width':
        loop['kappa'][0] = [5.0]
    elif damage == 'not_a_list':
        loop['kappa'][0] = 5.0
    elif damage in ('zero', 'negative', 'nan', 'inf', 'bool', 'string'):
        loop['kappa'][1][1] = {'zero': 0.0, 'negative': -1.0, 'nan': float('nan'), 'inf': float('inf'), 'bool': True, 'string': '5'}[damage]
    elif damage == 'identity_without_split':
        head['identity'].pop('decay_groups')
    elif damage == 'split_identity_in_v6':
        head['schema'] = em.HEAD_SCHEMA
    else:
        head['identity']['encoder'] = {}
    with pytest.raises(RuntimeError, match='epoch metrics'):
        em.snapshot(loop, head)


@pytest.mark.parametrize('damage', ['doc_schema', 'definitions', 'column_removed', 'column_added', 'kappa_changed', 'v6_doc_under_v7_head', 'csv_changed'])
def test_a_v7_snapshot_that_was_edited_is_refused_when_it_is_read(tmp_path, damage):
    folder, head, _ = save_v7(tmp_path)
    doc = em.read_snapshot(folder, head)
    if damage == 'doc_schema':
        doc['schema'] = em.SCHEMA
    elif damage == 'definitions':
        doc['definitions'] = em.DEFINITIONS
    elif damage == 'column_removed':
        doc['rows'][0].pop('kappa_1')
    elif damage == 'column_added':
        doc['rows'][0]['kappa_2'] = 1.0
    elif damage == 'kappa_changed':
        doc['rows'][0]['kappa_1'] = -3.0
    elif damage == 'v6_doc_under_v7_head':
        for row in doc['rows']:
            row.pop('kappa_0'), row.pop('kappa_1')
        doc['schema'], doc['definitions'] = em.SCHEMA, em.DEFINITIONS
    if damage == 'csv_changed':
        raw = (folder / em.FILES[1]).read_bytes().replace(b'19.5', b'19.6')
        (folder / em.FILES[1]).write_bytes(raw)
        head['epoch_metrics']['files'][em.FILES[1]] = {'bytes': len(raw), 'sha256': cp.sha256(raw)}
        (folder / 'head.json').write_bytes(cp.json_bytes(head))
    else:                                                                # the json rebound with a valid hash: only the content check can refuse it
        payload = em.json_bytes(doc)
        (folder / em.FILES[0]).write_bytes(payload)
        head['epoch_metrics']['files'][em.FILES[0]] = {'bytes': len(payload), 'sha256': cp.sha256(payload)}
        (folder / 'head.json').write_bytes(cp.json_bytes(head))
    with pytest.raises(RuntimeError):
        cp.latest_metadata(tmp_path)


# ---- the v8 head and the epoch-metrics-v3 snapshot: the column terrain_penalty under a terrain smoothness weight other than 0 (S4b preparation M6, decision D8) ----
PENALTIES = [0.25, 0.5, 0.75]


def fixture_state_v8(stops=(4., 3.), *, split=False, n2=False, penalties=PENALTIES, horizon=4, patience=2):
    """The fixture of a run with ``terrain_smooth_weight`` 0.5: the v8 head, an identity that names the weight and a loop with the record of R_Omega (and of kappa under 'split')."""
    head, loop = fixture_state(stops, horizon=horizon, patience=patience)
    head['schema'] = em.HEAD_SCHEMA_V8
    head['identity'] = {**head['identity'], 'encoder': {'kappas': 2}, 'terrain_smooth_weight': 0.5}
    if split:
        head['identity'] = {**head['identity'], 'decay_groups': 'split', 'decay_groups_sha256': 'a' * 64}
        loop['kappa'] = [None if n2 else list(KAPPAS[i]) for i in range(len(stops))]
    if n2:
        head['identity']['model_spec'] = {'architecture': 'n2_teamset_v1', 'n2_hidden': 8, 'prefetch': 0, 'precision': 'float32_strict'}
    loop['terrain_penalty'] = list(penalties[:len(stops)])
    return head, loop


def save_v8(root, stops=(4., 3.), **kwargs):
    head, loop = fixture_state_v8(stops, **kwargs)
    files = {'last.pt': b'last', 'best.pt': b'best', **em.encode(loop, head)}
    return cp.save_checkpoint(root, len(stops), files, head)


def test_the_v6_and_v7_names_are_unchanged_and_the_v8_ones_are_new():
    assert em.HEAD_SCHEMA_V8 == 'ept_train_checkpoint_v8' and em.SCHEMA_V3 == 'epoch-metrics-v3' and em.TERRAIN_COLUMN == 'terrain_penalty'
    assert em.HEAD_SCHEMAS == (em.HEAD_SCHEMA, em.HEAD_SCHEMA_V7, em.HEAD_SCHEMA_V8) and cp.METRICS_SCHEMAS == em.HEAD_SCHEMAS
    assert em.HEAD_SCHEMA == 'ept_train_checkpoint_v6' and em.HEAD_SCHEMA_V7 == 'ept_train_checkpoint_v7' and em.SCHEMA == 'epoch-metrics-v1' and em.SCHEMA_V2 == 'epoch-metrics-v2'
    assert em.head_schema({'patience': 2}) == em.HEAD_SCHEMA and em.head_schema({'decay_groups': 'split'}) == em.HEAD_SCHEMA_V7
    assert em.head_schema({'terrain_smooth_weight': 0.5}) == em.HEAD_SCHEMA_V8 and em.head_schema({'decay_groups': 'split', 'terrain_smooth_weight': 0.5}) == em.HEAD_SCHEMA_V8
    head, loop = fixture_state()                                       # a v6 head: no terrain column whatever the loop holds
    loop['terrain_penalty'] = [0.25, 0.5]
    assert all(set(row) == set(em.COLUMNS) for row in em.snapshot(loop, head)['rows'])


@pytest.mark.parametrize('split', [False, True])
def test_v8_rows_carry_the_terrain_penalty_after_the_kappa_columns_in_json_and_csv_and_survive_the_head_last_round_trip(tmp_path, split):
    folder, head, _ = save_v8(tmp_path, split=split)
    found_folder, found_head, doc = cp.latest_metadata(tmp_path)
    assert found_folder == folder and found_head == head and head['schema'] == em.HEAD_SCHEMA_V8
    columns = (*em.COLUMNS, *(('kappa_0', 'kappa_1') if split else ()), 'terrain_penalty')
    assert doc['schema'] == em.SCHEMA_V3 and head['epoch_metrics']['schema'] == em.SCHEMA_V3 == em.snapshot_schema(head) and em.columns(head) == columns
    assert 'terrain_penalty' in doc['definitions'] and ('kappa_i' in doc['definitions']) is split and doc['definitions']['fit_batch_mean_loss'] == em.DEFINITIONS['fit_batch_mean_loss']
    values = ((5.0, 20.0, 0.25), (5.25, 19.5, 0.5)) if split else ((0.25,), (0.5,))
    assert doc['rows'] == [dict(zip(columns, (1, 0, -.5, 4., .02, *values[0]))), dict(zip(columns, (2, 1, -1.5, 3., .02, *values[1])))]
    parsed = list(csv.DictReader(io.StringIO((folder / em.FILES[1]).read_text(encoding='utf-8'))))
    assert list(parsed[0]) == list(columns) and [float(row['terrain_penalty']) for row in parsed] == [0.25, 0.5]
    assert em.read_snapshot(folder, head) == doc and cp.latest_checkpoint(tmp_path)[2:] == (b'last', b'best')


def test_v8_n2_rows_have_blank_kappa_and_a_number_for_the_terrain_penalty(tmp_path):
    folder, head, _ = save_v8(tmp_path, split=True, n2=True)
    doc = cp.latest_metadata(tmp_path)[2]
    assert all(row['kappa_0'] is None and row['kappa_1'] is None and isinstance(row['terrain_penalty'], float) for row in doc['rows'])


@pytest.mark.parametrize('damage', ['no_record', 'short_record', 'long_record', 'negative', 'nan', 'inf', 'bool', 'string', 'none', 'identity_without_weight', 'v6_head_with_weight', 'zero_weight_in_identity',
                                    'negative_weight', 'bool_weight', 'string_weight', 'v7_head_with_weight', 'split_without_kappa_record'])
def test_v8_refuses_a_record_that_is_missing_or_not_a_finite_number_not_below_zero_and_a_head_that_does_not_match_the_weight(damage):
    head, loop = fixture_state_v8(split=damage == 'split_without_kappa_record')
    if damage == 'no_record':
        loop.pop('terrain_penalty')
    elif damage == 'short_record':
        loop['terrain_penalty'].pop()
    elif damage == 'long_record':
        loop['terrain_penalty'].append(1.0)
    elif damage in ('negative', 'nan', 'inf', 'bool', 'string', 'none'):
        loop['terrain_penalty'][1] = {'negative': -0.5, 'nan': float('nan'), 'inf': float('inf'), 'bool': True, 'string': '5', 'none': None}[damage]
    elif damage == 'identity_without_weight':
        head['identity'].pop('terrain_smooth_weight')
    elif damage == 'v6_head_with_weight':
        head['schema'] = em.HEAD_SCHEMA
    elif damage == 'v7_head_with_weight':
        head['schema'] = em.HEAD_SCHEMA_V7
    elif damage == 'split_without_kappa_record':
        loop.pop('kappa')
    else:
        head['identity']['terrain_smooth_weight'] = {'zero_weight_in_identity': 0.0, 'negative_weight': -0.5, 'bool_weight': True, 'string_weight': '0.5'}[damage]
    with pytest.raises(RuntimeError, match='epoch metrics'):
        em.snapshot(loop, head)


def test_a_zero_value_of_the_penalty_is_a_valid_record():
    """R_Omega is 0 while the learned field is still zero (the first steps): a mean of 0 is a number, not a missing record."""
    head, loop = fixture_state_v8(penalties=[0.0, 0.0, 0.0])
    assert [row['terrain_penalty'] for row in em.snapshot(loop, head)['rows']] == [0.0, 0.0]


@pytest.mark.parametrize('damage', ['doc_schema', 'definitions', 'column_removed', 'column_added', 'penalty_changed', 'v1_doc_under_v8_head', 'csv_changed'])
def test_a_v8_snapshot_that_was_edited_is_refused_when_it_is_read(tmp_path, damage):
    folder, head, _ = save_v8(tmp_path)
    doc = em.read_snapshot(folder, head)
    if damage == 'doc_schema':
        doc['schema'] = em.SCHEMA_V2
    elif damage == 'definitions':
        doc['definitions'] = em.DEFINITIONS
    elif damage == 'column_removed':
        doc['rows'][0].pop('terrain_penalty')
    elif damage == 'column_added':
        doc['rows'][0]['kappa_0'] = 1.0
    elif damage == 'penalty_changed':
        doc['rows'][0]['terrain_penalty'] = -3.0
    elif damage == 'v1_doc_under_v8_head':
        for row in doc['rows']:
            row.pop('terrain_penalty')
        doc['schema'], doc['definitions'] = em.SCHEMA, em.DEFINITIONS
    if damage == 'csv_changed':
        raw = (folder / em.FILES[1]).read_bytes().replace(b'0.25', b'0.26')
        (folder / em.FILES[1]).write_bytes(raw)
        head['epoch_metrics']['files'][em.FILES[1]] = {'bytes': len(raw), 'sha256': cp.sha256(raw)}
        (folder / 'head.json').write_bytes(cp.json_bytes(head))
    else:                                                                # the json rebound with a valid hash: only the content check can refuse it
        payload = em.json_bytes(doc)
        (folder / em.FILES[0]).write_bytes(payload)
        head['epoch_metrics']['files'][em.FILES[0]] = {'bytes': len(payload), 'sha256': cp.sha256(payload)}
        (folder / 'head.json').write_bytes(cp.json_bytes(head))
    with pytest.raises(RuntimeError):
        cp.latest_metadata(tmp_path)


def test_a_v5_head_after_a_v8_head_is_a_downgrade(tmp_path):
    prior, _, _ = save_v8(tmp_path, (4.,))
    folder, head, _ = save_v8(tmp_path)
    head['schema'] = cp.SCHEMA
    head.pop('epoch_metrics')
    for name in em.FILES:
        (folder / name).unlink()
    (folder / 'head.json').write_bytes(cp.json_bytes(head))
    with pytest.raises(RuntimeError, match='downgrade'):
        cp.latest_metadata(tmp_path)
    assert (prior / 'head.json').is_file()


def test_a_v5_head_after_a_v7_head_is_a_downgrade(tmp_path):
    prior, _, _ = save_v7(tmp_path, (4.,))
    folder, head, _ = save_v7(tmp_path)
    head['schema'] = cp.SCHEMA
    head.pop('epoch_metrics')
    for name in em.FILES:
        (folder / name).unlink()
    (folder / 'head.json').write_bytes(cp.json_bytes(head))
    with pytest.raises(RuntimeError, match='downgrade'):
        cp.latest_metadata(tmp_path)
    assert (prior / 'head.json').is_file()


# ---- S4b preparation M8 (decision D10 P7): a run that ends at the wall cap says so --------------------------------------------------------
def budget_state(stops=(4., 3.), *, horizon=4, patience=2):
    """The state of a run that ended at the wall cap after len(stops) epochs: the identity names the setting, the head says done and budget_end."""
    head, loop = fixture_state(stops, horizon=horizon, patience=patience)
    head['identity']['on_wall_cap'] = 'best_stop'
    head['done'], head['budget_end'] = True, True
    head['caps'], head['seconds'] = {'wall_seconds': 86_400}, 86_400.5                 # the cap was reached when the epoch ended (EM-3)
    return head, loop


def test_epoch_metrics_accepts_budget_reason_and_consumers_accept_done(tmp_path):
    head, loop = budget_state()
    doc = em.snapshot(loop, head)
    assert (doc['termination_reason'], doc['done'], doc['stopped_early'], doc['best_epoch'], doc['bad_epochs']) == ('budget', True, False, 2, 0)
    files = {'last.pt': b'last', 'best.pt': b'best', **em.encode(loop, head)}
    folder, saved, _ = cp.save_checkpoint(tmp_path, 2, files, head)
    assert em.read_snapshot(folder, saved)['termination_reason'] == 'budget'
    latest_folder, latest_head, latest_doc = cp.latest_metadata(tmp_path)                    # the reader of the newest checkpoint takes it, and tells done
    assert latest_folder == folder and latest_head['done'] is True and latest_head['budget_end'] is True and latest_doc['termination_reason'] == 'budget'


def test_the_budget_end_of_the_best_epoch_that_is_not_the_last_is_a_valid_end():
    head, loop = budget_state((3., 4.))                                                      # the best epoch is the first; one epoch without a new best, under the patience
    assert em.snapshot(loop, head)['best_epoch'] == 1 and em.snapshot(loop, head)['termination_reason'] == 'budget'


@pytest.mark.parametrize('damage', ['no_setting', 'budget_false', 'budget_one', 'budget_text', 'early_stopping', 'horizon_reached', 'not_done', 'under_the_cap', 'no_seconds'])
def test_a_budget_end_is_refused_unless_it_is_the_end_that_no_other_rule_reached(damage):
    head, loop = budget_state()
    if damage == 'no_setting':
        head['identity'].pop('on_wall_cap')                                                  # 'fail' (the default) can not end a run at the cap
    elif damage == 'budget_false':
        head['budget_end'] = False
    elif damage == 'budget_one':
        head['budget_end'] = 1
    elif damage == 'budget_text':
        head['budget_end'] = 'yes'
    elif damage == 'early_stopping':
        head, loop = budget_state((-2., -2., -1.), horizon=4)                                # the patience ran out: that is the reason
    elif damage == 'horizon_reached':
        head, loop = budget_state((4., 3.), horizon=2)                                       # the horizon was reached: that is the reason
    elif damage == 'under_the_cap':
        head['seconds'] = 86_399.5                                                           # the cap was not reached: nothing ended this run
    elif damage == 'no_seconds':
        head.pop('seconds')
    else:
        head['done'] = False
    with pytest.raises(RuntimeError, match='epoch metrics'):
        em.snapshot(loop, head)


def test_without_a_budget_end_the_reasons_and_the_snapshot_are_those_of_every_earlier_run():
    head, loop = fixture_state((4., 3.), horizon=4)
    assert 'budget_end' not in head and em.snapshot(loop, head)['termination_reason'] == 'in_progress'
    head['identity']['on_wall_cap'] = 'best_stop'                                            # the setting alone changes nothing about the rows or the reason
    assert em.snapshot(loop, head)['termination_reason'] == 'in_progress' and em.snapshot(loop, head)['done'] is False
