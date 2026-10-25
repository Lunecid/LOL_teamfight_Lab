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
