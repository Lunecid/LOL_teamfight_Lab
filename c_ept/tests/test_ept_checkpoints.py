"""합성 임시 바이트만 사용하는 v5 IO 경계 테스트. tensor 역직렬화는 없다."""
import json
from pathlib import Path

import pytest

from c_ept.ept import checkpoints as cp


def save(root, epoch=1, *, best=None, write=None):
    files = {'last.pt': b'synthetic last state'}
    if best is None:
        files['best.pt'] = b'synthetic best state'
        best = {'folder': f'epoch_{epoch:04d}', 'sha256': cp.sha256(files['best.pt']), 'bytes': len(files['best.pt'])}
    meta = {'schema': cp.SCHEMA, 'epoch': epoch, 'done': True, 'best': best,
            'timing': {name: [0.] * epoch for name in cp.COMPONENT_FIELDS}}
    return cp.save_checkpoint(root, epoch, files, meta, write=write)


def test_save_and_verified_latest_round_trip_including_older_best(tmp_path):
    folder, head, digest = save(tmp_path)
    assert digest == cp.sha256((folder / 'head.json').read_bytes())
    assert cp.SCHEMA == 'ept_train_checkpoint_v5'
    found = cp.latest_checkpoint(tmp_path)
    assert found == (folder, head, b'synthetic last state', b'synthetic best state')
    second, head2, _ = save(tmp_path, 2, best=head['best'])
    assert not (second / 'best.pt').exists()
    assert cp.latest_checkpoint(tmp_path) == (second, head2, found[2], found[3])
    with pytest.raises(FileExistsError):
        save(tmp_path, 2, best=head['best'])


def test_empty_and_torn_folders_are_skipped_then_reused(tmp_path):
    assert cp.latest_checkpoint(tmp_path / 'missing') is None
    folder = tmp_path / 'epoch_0001'
    folder.mkdir()
    (folder / 'last.pt').write_bytes(b'torn')
    (folder / 'head.json.tmp').write_bytes(b'incomplete head')
    assert cp.latest_checkpoint(tmp_path) is None
    save(tmp_path)
    assert cp.latest_checkpoint(tmp_path)[2] == b'synthetic last state'
    assert not (folder / 'head.json.tmp').exists()


@pytest.mark.parametrize('file', ['last.pt', 'best.pt'])
@pytest.mark.parametrize('damage', ['same_size', 'wrong_size', 'missing'])
def test_payload_damage_refuses_instead_of_falling_back(tmp_path, file, damage):
    save(tmp_path)
    folder, _, _ = save(tmp_path, 2)
    path = folder / file
    if damage == 'missing':
        path.unlink()
    else:
        path.write_bytes(b'x' * path.stat().st_size if damage == 'same_size' else b'x')
    with pytest.raises(RuntimeError, match='not guessing an older'):
        cp.latest_checkpoint(tmp_path)


@pytest.mark.parametrize('damage', ['json', 'schema', 'epoch', 'timing_missing', 'timing_negative', 'timing_bool', 'timing_nan', 'timing_length', 'best_path'])
def test_complete_head_damage_refuses_latest(tmp_path, damage):
    save(tmp_path)
    folder, head, _ = save(tmp_path, 2)
    if damage == 'schema': head['schema'] = 'obsolete'
    if damage == 'epoch': head['epoch'] = 3
    if damage == 'timing_missing': head['timing'].pop('forward_seconds')
    if damage == 'timing_negative': head['timing']['forward_seconds'][0] = -1.
    if damage == 'timing_bool': head['timing']['forward_seconds'][0] = True
    if damage == 'timing_nan': head['timing']['forward_seconds'][0] = float('nan')
    if damage == 'timing_length': head['timing']['forward_seconds'] = []
    if damage == 'best_path': head['best']['folder'] = '../epoch_0001'
    (folder / 'head.json').write_bytes(b'{' if damage == 'json' else cp.json_bytes(head))
    with pytest.raises(RuntimeError):
        cp.latest_checkpoint(tmp_path)


def test_corrupted_historical_best_is_verified(tmp_path):
    folder, head, _ = save(tmp_path)
    save(tmp_path, 2, best=head['best'])
    (folder / 'best.pt').write_bytes(b'corrupted')
    with pytest.raises(RuntimeError, match='best.pt.*not guessing'):
        cp.latest_checkpoint(tmp_path)


def test_injected_writer_keeps_head_last_and_torn_retry(tmp_path):
    old, _, _ = save(tmp_path)
    seen = []
    def cut(path, payload):
        seen.append(path.name)
        if path.name == 'head.json':
            raise OSError('synthetic power cut')
        cp.write_atomic(path, payload)
    with pytest.raises(OSError):
        save(tmp_path, 2, write=cut)
    assert seen == ['last.pt', 'best.pt', 'head.json']
    assert cp.latest_checkpoint(tmp_path)[0] == old
    save(tmp_path, 2)
    assert cp.latest_checkpoint(tmp_path)[0].name == 'epoch_0002'


def test_atomic_rename_failure_keeps_old_complete_file(tmp_path, monkeypatch):
    path = tmp_path / 'synthetic.bin'
    cp.write_atomic(path, b'old')
    def fail(src, dst):
        raise OSError('synthetic rename failure')
    with monkeypatch.context() as patch:
        patch.setattr(cp.os, 'replace', fail)
        with pytest.raises(OSError):
            cp.write_atomic(path, b'new')
    assert path.read_bytes() == b'old'
    assert path.with_name(path.name + '.tmp').read_bytes() == b'new'
    cp.write_atomic(path, b'new')
    assert path.read_bytes() == b'new'


def test_train_wrapper_observes_write_monkeypatch_on_each_save(tmp_path):
    import ast
    path = Path(__file__).parents[1] / 'ept/train.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    wrapper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_save_checkpoint')
    seen = []
    def writer(tag):
        def write(path, payload):
            seen.append((tag, path.name))
            cp.write_atomic(path, payload)
        return write
    ns = {'Path': Path, 'checkpoints': cp, '_write_atomic': writer('first')}
    exec(compile(ast.Module(body=[wrapper], type_ignores=[]), '<train checkpoint compatibility>', 'exec'), ns)
    files = {'last.pt': b'synthetic last', 'best.pt': b'synthetic best'}
    def metadata(epoch):
        return {'schema': cp.SCHEMA, 'epoch': epoch, 'done': True,
                'best': {'folder': f'epoch_{epoch:04d}', 'bytes': len(files['best.pt']), 'sha256': cp.sha256(files['best.pt'])},
                'timing': {key: [0.] * epoch for key in cp.COMPONENT_FIELDS}}
    ns['_save_checkpoint'](tmp_path, 1, files, metadata(1))
    ns['_write_atomic'] = writer('second')
    ns['_save_checkpoint'](tmp_path, 2, files, metadata(2))
    assert seen == [(tag, name) for tag in ('first', 'second') for name in ('last.pt', 'best.pt', 'head.json')]
    assert cp.latest_checkpoint(tmp_path)[1]['epoch'] == 2


def test_the_head_schemas_are_v5_for_the_calls_and_v6_or_v7_for_a_head_with_a_metrics_snapshot():
    """v7 (S4b preparation M4) is the head of a run under ``decay_groups='split'``: it carries the same snapshot files with the kappa columns. v6 stays the head of every earlier
    run, so the names of the earlier schemas do not change."""
    assert cp.SCHEMA == 'ept_train_checkpoint_v5' and cp.METRICS_SCHEMA == 'ept_train_checkpoint_v6'
    assert cp.METRICS_SCHEMAS == ('ept_train_checkpoint_v6', 'ept_train_checkpoint_v7', 'ept_train_checkpoint_v8')          # v8 (S4b preparation M6): a run with a terrain smoothness weight
