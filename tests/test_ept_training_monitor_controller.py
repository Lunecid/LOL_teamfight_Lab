"""Fresh synthetic metadata only: no tensor files, data, GPU or real launches."""
from __future__ import annotations

import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / 'tools'))
import ept_training_monitor as m
import ept_training_controller as c
sys.path.pop(0)


def write(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')


def missing(pid):
    return {'kind': 'missing'}


def alive(pid):
    return {'kind': 'observed', 'create_time': 17.0}


@pytest.fixture
def fake_job(monkeypatch):
    """Pure fake containment; no Windows object/process operation is performed."""
    class Job:
        instances = []
        def __init__(self):
            self.events = ['created']
            self.child = None
            self.stopped = False
            self.instances.append(self)
        def assign(self, child):
            self.events.append('assigned')
            self.child = child
        def release(self, child):
            assert child is self.child
            self.events.append('released')
            if getattr(child, 'stdin', None): child.stdin.close()
        def contains(self, pid, created):
            self.events.append(('contains', pid, created))
            return True
        def stop(self, timeout=10):
            self.events.append(('stop', timeout))
            self.stopped = True
            return True
        def gone(self): return True
        def close(self): self.events.append('closed')
    monkeypatch.setattr(c, 'WindowsJob', Job)
    return Job


@pytest.fixture
def pilot(tmp_path):
    runs = tmp_path / 'runs'
    run = runs / '20261025_c_ept_ept_pilot_1'
    control = runs / 'pilot-control'
    config = {'schema': 'ept-training-operator-v1', 'profile': 's25_early_stop',
              'python': str(Path(sys.executable).resolve()), **m.PARAMETERS,
              **{k: str((tmp_path / k).resolve()) for k in ('s1', 's2', 's3', 'catalog_db')},
              'runs_root': str(runs.resolve())}
    cfg_path = control / 'operator.json'
    write(cfg_path, config)
    identity = {'args': config, 'operator_config_sha256': m.digest(config),
                'code': {'code_commit': 'a'*40, 'code_dirty': False, 'repo_root': str(SOURCE)},
                'runtime': {'dtype': 'torch.float32', 'threads': 8}, 'inputs': {'synthetic': True}}
    manifest = {'schema': m.RUN_SCHEMA, 'stage': 'ept_training_single', 'operator_api_version': 1,
                'identity': identity, 'run_identity_sha256': m.digest(identity),
                'operator_config_sha256': m.digest(config), 'code_commit': 'a'*40, 'code_dirty': False,
                'runtime': identity['runtime'], 'stages': list(m.STAGES), 'results': {}}
    write(run / 'RUN.json', manifest)
    process = {'schema': m.PROCESS_SCHEMA, 'operator_api_version': 1, 'run_path': str(run),
               **{k: manifest[k] for k in m.REF_FIELDS - {'run'}}, 'pid': 123456, 'create_time': 17.0,
               'state': 'running', 'outcome': None, 'exit_code': None, 'updated_at': '2026-10-25T00:00:00+00:00'}
    write(run / 'PROCESS.json', process)
    ref = {'run': str(run), **{k: manifest[k] for k in m.REF_FIELDS - {'run'}}}
    write(control / 'run.json', ref)
    controller = {'schema': c.CONTROLLER_SCHEMA, 'source': str(SOURCE), 'code_commit': 'a'*40,
                  'operator_config_sha256': m.digest(config), 'state': 'running',
                  'controller_pid': 234567, 'controller_create_time': 17.0,
                  'wrapper_pid': 345678, 'wrapper_create_time': 17.0}
    write(control / 'controller.json', controller)
    return SimpleNamespace(run=run, control=control, config=config, config_path=cfg_path,
                           manifest=manifest, process=process, ref=ref, controller=controller, out=tmp_path/'reports')


def epoch(p, stops=(4.,), terminal=False):
    cp = m.checkpoint_reader(SOURCE)
    n = len(stops)
    best = min(range(n), key=stops.__getitem__)
    early = n-best-1 >= 2
    identity = {**{k: p.config[k] for k in ('seed', 'lr', 'weight_decay', 'warmup_steps', 'patience', 'device')},
                'epochs': 30, 'dtype': 'torch.float32', 'threads': 8, 'deterministic_algorithms': True,
                'encoder': {'d': 128, 'layers': 4, 'heads': 4, 'dropout': .1, 'kappas': 3, 'field_channels': 8},
                **{role: {'role': role.upper(), 'training': role == 'fit', 'batch_size': 32, 'seed': 20261022} for role in ('fit','stop')}}
    head = {'schema': 'ept_train_checkpoint_v6', 'epoch': n, 'epochs': 30, 'seed': 20261022,
            'identity': identity, 'done': early, 'stopped_early': early, 'best_epoch': best,
            'best': {'folder': f'epoch_{best+1:04d}', 'bytes': 17, 'sha256': 'f'*64},
            'files': {'last.pt': {'bytes': 17, 'sha256': 'f'*64}},
            'timing': {'steps': [1]*n, **{k: [0.]*n for k in cp.COMPONENT_FIELDS}}}
    loop = {'epoch': n, 'fit_batch_mean_losses': [-.5-i for i in range(n)], 'stop_losses': list(stops),
            'steps': [1]*n, 'lrs': [.001]*n, 'best_epoch': best, 'bad': n-best-1,
            'best_value': min(stops), 'stopped_early': early}
    folder = p.run / 'train/S_cuda' / f'epoch_{n:04d}'
    folder.mkdir(parents=True, exist_ok=True)
    files = cp.epoch_metrics.encode(loop, head)
    head['epoch_metrics'] = {'schema': 'epoch-metrics-v1', 'files': {k: {'bytes': len(v), 'sha256': m.sha(v)} for k,v in files.items()}}
    for k,v in files.items():
        (folder/k).write_bytes(v)
    write(folder/'head.json', head)
    if terminal:
        pending = {'schema': 'ept_train_checkpoint_v5', 'kind': 'pending', 'call': 1,
                   'identity_sha256': m.sha(json.dumps(identity, sort_keys=True).encode()),
                   'epoch_done_start': 0, 'head_sha256_start': None, 'start_accounting': {}}
        write(p.run/'train/S_cuda/calls/call_0001.pending.json', pending)
        final = {'schema': 'ept_train_checkpoint_v5', 'kind': 'final', 'call': 1,
                 'identity_sha256': pending['identity_sha256'], 'epoch_done_start': 0,
                 'epoch_done_end': n, 'head_sha256_end': m.sha((folder/'head.json').read_bytes()),
                 'pending_sha256': m.sha((p.run/'train/S_cuda/calls/call_0001.pending.json').read_bytes()),
                 'outcome': 'completed', 'status': 'ok', 'seconds': 3.0}
        write(p.run/'train/S_cuda/calls/call_0001.json', final)
        ledger = {'schema': 's3_stages_v1', 'calls': [], 'stages': {}}
        for name in m.STAGES:
            stage_digest = m.digest({'identity': p.manifest['identity'], 'stage': name})
            data = {'synthetic': True} if name != 'train_S_cuda' else {
                'size':'S','device':'cuda','seed':20261022,'epochs_run':n,'best_epoch':best,'resumed':False,'recovered':False}
            record = {'schema':'s4a_stage_v1','stage':name,'inputs_sha256':stage_digest,'data':data,'resources':{'sampled':True}}
            write(p.run/(name+'.json'), record)
            outputs = {name+'.json': m.sha((p.run/(name+'.json')).read_bytes())}
            if name == 'train_S_cuda':
                for file in list(folder.iterdir()) + list((p.run/'train/S_cuda/calls').glob('*.json')):
                    outputs[file.relative_to(p.run).as_posix()] = m.sha(file.read_bytes())
                # A payload hash can be present without any payload read or creation.
                outputs['train/S_cuda/epoch_0001/last.pt'] = 'f'*64
            ledger['stages'][name] = {'complete':True,'inputs_sha256':stage_digest,
                                     'inputs':{'run_identity_sha256':p.manifest['run_identity_sha256']},'outputs':outputs}
            p.manifest['results'][name] = record
        write(p.run/'STAGES.json', ledger)
        write(p.run/'RUN.json', p.manifest)
        p.process.update(state='finished', outcome='completed', exit_code=0)
        write(p.run/'PROCESS.json', p.process)
        # Explicit synthetic OS-cleanup evidence; no real Job was launched here.
        p.controller.update(state='completed', tree_gone=True, job_gone_before_close=True, job_closed=True,
                            job_ownership={'kind':'windows_job_kill_on_close','assigned':True,'released':True})
        write(p.control/'controller.json', p.controller)
    return folder, head


def inspect(p, probe=alive):
    return m.inspect_run(p.run, SOURCE, p.config, control=p.control, probe=probe)


def test_active_metadata_only_quiet_then_new_completed_epoch(pilot, monkeypatch):
    p = pilot
    opened = []
    original = Path.open
    science_before = {name: sys.modules.get(name) for name in ('torch', 'numpy', 'matplotlib.pyplot')}
    def guarded(path, *a, **kw):
        assert path.suffix != '.pt'
        opened.append(path)
        return original(path, *a, **kw)
    monkeypatch.setattr(Path, 'open', guarded)
    epoch(p)
    report = inspect(p)
    assert report['state'] == 'running' and report['epoch'] == 1
    before = {str(path): path.read_bytes() for path in p.run.rglob('*') if path.is_file()}
    assert m.save_report(report, p.out)
    snapshots = sorted((p.out/'history').glob('epoch_*/snapshot.json'))
    assert len(snapshots)==1
    retained = {str(path): path.read_bytes() for path in snapshots[0].parent.iterdir()}
    assert not m.save_report(inspect(p), p.out)
    assert len(list((p.out/'history').glob('epoch_*/snapshot.json')))==1
    epoch(p, (4.,3.))
    assert m.save_report(inspect(p), p.out)
    assert m.read_json(p.out/'monitor-state.json')['epoch'] == 2
    assert len(list((p.out/'history').glob('epoch_*/snapshot.json')))==2
    for path, raw in retained.items(): assert Path(path).read_bytes()==raw
    for path, raw in before.items():
        assert Path(path).read_bytes() == raw
    assert not any(path.suffix == '.pt' for path in opened)
    assert all(sys.modules.get(name) is module for name, module in science_before.items())


@pytest.mark.parametrize('case', ['code','config','digest','args','dirty','stage_selection','source','process_code','process_ref','process_fields','process_combo','pid_bool','pid_generation'])
def test_bad_root_binding_refused(pilot, case):
    p = pilot
    if case == 'code': p.manifest['code_commit'] = 'b'*40
    if case == 'config': p.manifest['operator_config_sha256'] = 'b'*64
    if case == 'digest': p.manifest['run_identity_sha256'] = 'b'*64
    if case == 'args': p.manifest['identity']['args'] = {**p.config, 'max_epochs':2}
    if case == 'dirty': p.manifest['code_dirty'] = True
    if case == 'stage_selection': p.manifest['stages'].append('test')
    if case == 'source': p.manifest['identity']['code']['repo_root'] = str(p.control)
    if case == 'process_code': p.process['code_commit'] = 'b'*40
    if case == 'process_ref': p.process['run_path'] = str(p.control)
    if case == 'process_fields': p.process['extra'] = None
    if case == 'process_combo': p.process.update(state='finished', outcome='completed', exit_code=7)
    if case == 'pid_bool': p.process['pid'] = True
    if case == 'pid_generation': p.process['create_time'] = 17
    write(p.run/'RUN.json', p.manifest)
    write(p.run/'PROCESS.json', p.process)
    with pytest.raises((m.MetadataError, RuntimeError)):
        inspect(p)


@pytest.mark.parametrize('key,value', [('max_epochs',2),('patience',True),('size','M'),('device','cpu'),('share',1.),('seed',7),('lr',.1),('batch_size',1),('schema','other'),('profile','other'),('warmup_steps',True),('weight_decay',float('inf'))])
def test_config_tuple_is_strict(pilot, key, value):
    data = {**pilot.config, key:value}
    pilot.config_path.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(m.MetadataError):
        m.load_config(pilot.config_path)


@pytest.mark.parametrize('raw', ['{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}', '[]', '{'])
def test_corrupt_duplicate_nonfinite_json_refused(tmp_path, raw):
    path = tmp_path/'bad.json'
    path.write_text(raw, encoding='utf-8')
    with pytest.raises(m.MetadataError): m.read_json(path)


def test_read_limit_is_bounded_before_parse(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'MAX_BYTES', 16)
    path = tmp_path/'big.json'
    path.write_bytes(b' '*17)
    with pytest.raises(m.MetadataError, match='bounded'): m.read_json(path)


@pytest.mark.parametrize('probe,state', [(missing,'interrupted_unconfirmed'),(lambda pid:{'kind':'observed','create_time':18.},'interrupted_unconfirmed'),(lambda pid:{'kind':'unavailable'},'interrupted_unconfirmed')])
def test_running_owner_missing_reused_unavailable_never_success(pilot, probe, state):
    assert inspect(pilot,probe)['state'] == state


def test_process_missing_is_bootstrap_then_controller_failure(pilot):
    p=pilot
    (p.run/'PROCESS.json').unlink()
    assert inspect(p)['state'] == 'bootstrapping'
    p.controller['state']='failed'
    write(p.control/'controller.json',p.controller)
    assert inspect(p)['state'] == 'failed'


def test_terminal_success_requires_final_call_and_raw_stages(pilot):
    epoch(pilot,(4.,4.,5.),terminal=True)
    report=inspect(pilot,missing)
    assert report['state']=='completed' and report['history']['best_epoch']==1
    assert report['history']['termination_reason']=='early_stopping'
    m.render_guard(report,missing)
    assert not list(pilot.run.rglob('*.pt'))


@pytest.mark.parametrize('case',['root_hash','stage_input','stage_result','ledger_missing','head_hash','csv','json','call_identity','call_head','second_call','missing_final','missing_pending','head_param','head_batch','head_encoder'])
def test_terminal_corruption_is_not_success(pilot,case):
    p=pilot
    folder, head=epoch(p,(4.,4.,5.),terminal=True)
    if case=='root_hash': (p.run/'scales.json').write_text('{}',encoding='utf-8')
    elif case=='stage_input':
        ledger=m.read_json(p.run/'STAGES.json');ledger['stages']['stats']['inputs_sha256']='f'*64;write(p.run/'STAGES.json',ledger)
    elif case=='stage_result':
        p.manifest['results']['stats']['data']={};write(p.run/'RUN.json',p.manifest)
    elif case=='ledger_missing': (p.run/'STAGES.json').unlink()
    elif case=='head_hash': head['extra']='tampered';write(folder/'head.json',head)
    elif case in ('csv','json'): (folder/('epoch_metrics.'+case)).write_text('{}',encoding='utf-8')
    elif case.startswith('call_'):
        path=p.run/'train/S_cuda/calls/call_0001.json';doc=m.read_json(path)
        doc['identity_sha256' if case=='call_identity' else 'head_sha256_end']='f'*64;write(path,doc)
    elif case=='second_call': write(p.run/'train/S_cuda/calls/call_0002.json',{})
    elif case=='missing_final': (p.run/'train/S_cuda/calls/call_0001.json').unlink()
    elif case=='missing_pending': (p.run/'train/S_cuda/calls/call_0001.pending.json').unlink()
    else:
        if case=='head_param': head['identity']['lr']=.7
        if case=='head_batch': head['identity']['fit']['batch_size']=7
        if case=='head_encoder': head['identity']['encoder']['d']=256
        # Bind a syntactically sound metrics file to the altered identity so the
        # cross-layer training-parameter check, rather than CSV corruption, fires.
        doc=m.read_json(folder/'epoch_metrics.json');doc['identity']=head['identity']
        write(folder/'epoch_metrics.json',doc)
        raw=(folder/'epoch_metrics.json').read_bytes()
        head['epoch_metrics']['files']['epoch_metrics.json']={'bytes':len(raw),'sha256':m.sha(raw)}
        write(folder/'head.json',head)
    with pytest.raises((m.MetadataError,RuntimeError)):
        inspect(p,missing)


@pytest.mark.parametrize('probe,state',[(alive,'terminal_process_alive'),(lambda pid:{'kind':'unavailable'},'terminal_unconfirmed')])
def test_final_render_blocked_until_owner_gone(pilot,probe,state,monkeypatch):
    epoch(pilot,(4.,4.,5.),terminal=True)
    report=inspect(pilot,probe)
    assert report['state']==state
    with pytest.raises(m.MetadataError): m.final_plot(report,pilot.out,probe)
    assert 'matplotlib.pyplot' not in sys.modules


def test_render_blocks_live_wrapper_or_controller(pilot):
    epoch(pilot,(4.,4.,5.),terminal=True)
    report=inspect(pilot,missing)
    with pytest.raises(m.MetadataError):
        m.render_guard(report,lambda pid: alive(pid) if pid==345678 else missing(pid))


def test_monitor_cli_bootstrap_then_ref_and_failure_repeat_quiet(pilot,monkeypatch,capsys):
    p=pilot
    (p.control/'run.json').unlink()
    monkeypatch.setattr(m,'generation_state',lambda *a,**kw:'alive')
    argv=['--source',str(SOURCE),'--config',str(p.config_path),'--control',str(p.control),'--out',str(p.out)]
    assert m.main(argv)==0
    assert capsys.readouterr().out
    assert m.main(argv)==0 and not capsys.readouterr().out
    write(p.control/'run.json',p.ref)
    epoch(p)
    # main's default probe is captured at function definition, replace the API.
    original=m.inspect_run
    monkeypatch.setattr(m,'inspect_run',lambda *a,**kw:original(*a,**kw,probe=alive))
    assert m.main(argv)==0
    (p.run/'PROCESS.json').write_text('{',encoding='utf-8')
    assert m.main(argv)==1
    assert 'observation_error' in capsys.readouterr().out
    assert m.main(argv)==1 and not capsys.readouterr().out
    assert m.read_json(p.out/'report.json')['epoch']==1


def test_monitor_ref_drift_not_rebound(pilot):
    epoch(pilot)
    report=inspect(pilot)
    assert m.save_report(report,pilot.out)
    report={**report,'run':str(pilot.control)}
    with pytest.raises(m.MetadataError):m.save_report(report,pilot.out)


def test_controller_environment_is_process_local():
    original={'PATH':os.pathsep.join([r'C:\Codex\bin',str(c.PYTHON.parent/'Scripts'),r'C:\Windows\System32',r'C:\Git\cmd']),
              'CODEX_SECRET':'x','OPENAI_API_KEY':'x','CONDA_PREFIX':'x','PYTHONPATH':'x','GIT_DIR':'x','KEEP':'ok'}
    before=copy.deepcopy(original)
    env=c.child_environment(original)
    assert original==before
    assert not any(k in env for k in ('CODEX_SECRET','OPENAI_API_KEY','CONDA_PREFIX','PYTHONPATH','GIT_DIR'))
    assert env['KEEP']=='ok' and env['CUDA_VISIBLE_DEVICES']=='0'
    assert env['OMP_NUM_THREADS']==env['MKL_NUM_THREADS']==env['OPENBLAS_NUM_THREADS']=='8'
    assert env['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
    assert env['MKL_THREADING_LAYER']=='SEQUENTIAL' and env['KMP_DUPLICATE_LIB_OK']=='FALSE'
    assert 'Codex' not in env['PATH'] and str(c.PYTHON.parent) not in env['PATH']


def test_exact_wrapper_contract():
    command=c.wrapper_command(SOURCE,Path('config'),Path('control'),'Start')
    assert command[-6:]==['-Action','Start','-Config','config','-Control','control']
    assert str(SOURCE/'run_ept_training.ps1') in command
    if os.name=='nt': assert c.hidden_options()=={'creationflags':subprocess.CREATE_NO_WINDOW}


def test_controller_default_check_has_no_writes_or_launch(tmp_path,monkeypatch,capsys):
    control=tmp_path/'absent'
    config=tmp_path/'config.json'
    calls=[]
    monkeypatch.setattr(c,'preflight',lambda *a,**kw:(calls.append(a) or ({'synthetic':True},{})))
    monkeypatch.setattr(c,'supervise',lambda *a:pytest.fail('Start called by Check'))
    assert c.main(['--source',str(SOURCE),'--config',str(config),'--control',str(control),'--reviewed-code','a'*40])==0
    assert 'READY_WITHIN_SCOPE' in capsys.readouterr().out and calls
    assert not control.exists() and not config.exists()


@pytest.mark.parametrize('name',['launch.once','operator_start.once','run.json','controller.json'])
def test_previous_attempt_refuses_start_before_preflight(tmp_path,monkeypatch,name):
    control=tmp_path/'control';control.mkdir();(control/name).write_text('preserve')
    monkeypatch.setattr(c,'preflight',lambda *a,**kw:pytest.fail('preflight reached after used attempt'))
    with pytest.raises(m.MetadataError):c.supervise(SOURCE,tmp_path/'operator.json',control,'a'*40)
    assert (control/name).read_text()=='preserve'


def test_exclusive_controller_token_distinct_from_operator(tmp_path):
    control=tmp_path/'control'
    c.take_launch_token(control,{'test':True})
    assert (control/'launch.once').exists() and not (control/'operator_start.once').exists()
    with pytest.raises(FileExistsError):c.take_launch_token(control,{'test':False})


def test_exact_run_reference_only_no_folder_discovery(pilot):
    p=pilot
    ref,proc=c.read_bound_run(p.control,SOURCE,p.config,'a'*40)
    assert ref==p.ref and proc==p.process
    write(p.control/'run.json',{**p.ref,'run':str(p.control)})
    with pytest.raises(m.MetadataError):c.read_bound_run(p.control,SOURCE,p.config,'a'*40,ref)
    (p.control/'run.json').unlink()
    with pytest.raises(m.MetadataError):c.read_bound_run(p.control,SOURCE,p.config,'a'*40,ref)


def test_source_check_rejects_dirty_commit_or_failed_git(monkeypatch):
    for responses in ([SimpleNamespace(returncode=0,stdout='b'*40)],
                      [SimpleNamespace(returncode=0,stdout='a'*40),SimpleNamespace(returncode=0,stdout=' M file')],
                      [SimpleNamespace(returncode=1,stdout='')]):
        values=iter(responses)
        monkeypatch.setattr(c.subprocess,'run',lambda *a,**kw:next(values))
        with pytest.raises(m.MetadataError):c.source_check(SOURCE,'a'*40,{})


@pytest.mark.parametrize('failure',['popen','timeout','no_ref','wrong_process','wrapper_nonzero'])
def test_supervisor_one_launch_failure_preserved_no_retry(pilot,monkeypatch,failure,fake_job):
    p=pilot
    for name in ('controller.json','run.json'):(p.control/name).unlink()
    monkeypatch.setattr(c,'preflight',lambda *a,**kw:(p.config,{}))
    monkeypatch.setattr(c,'source_check',lambda *a:None)
    count={'launch':0,'stops':0}
    class Child:
        pid=100000
        def poll(self):return None if failure=='timeout' else 7 if failure=='wrapper_nonzero' else 0
        def wait(self,timeout):return 99 if failure=='timeout' else 0
    def popen(*args,**kwargs):
        count['launch']+=1
        if failure=='popen':raise OSError('synthetic launch failed')
        return Child()
    class Tree:
        root=(100000,17.0)
        known={root}
        referenced=set()
        def __init__(self,pid,job=None):pass
        def refresh(self):pass
        def bind_owner(self,p):return p['pid'],p['create_time']
        def stop(self):count['stops']+=1;return True
        def gone(self):return True
    monkeypatch.setattr(c.subprocess,'Popen',popen)
    monkeypatch.setattr(c,'ProcessTree',Tree)
    if failure=='timeout':
        # Preflight completes within budget, first supervisor loop exceeds it.
        ticks=iter([0.,0.,0.,0.,0.,0.,90000.,90001.,90002.])
        monkeypatch.setattr(c.time,'monotonic',lambda:next(ticks,90003.))
    if failure=='wrong_process':
        bad={**p.process,'state':'running'}
        monkeypatch.setattr(c,'read_bound_run',lambda *a:(p.ref,bad))
    elif failure=='wrapper_nonzero':
        monkeypatch.setattr(c,'read_bound_run',lambda *a:(p.ref,p.process))
    elif failure=='timeout':
        monkeypatch.setattr(c,'read_bound_run',lambda *a:(None,None))
    record=c.supervise(SOURCE,p.config_path,p.control,'a'*40)
    assert count['launch']==1 and record['state']=='failed' and record['attempts']==1
    assert (p.control/'launch.once').exists() and record['error']
    stored=m.read_json(p.control/'controller.json')
    assert stored['state']=='failed' and stored['elapsed_seconds']>=0
    with pytest.raises(m.MetadataError):c.supervise(SOURCE,p.config_path,p.control,'a'*40)
    if failure=='timeout':assert count['stops']==1 and stored['exit_code']==99


def test_controller_start_preflight_rejection_never_reserves_or_popen(pilot,monkeypatch):
    p=pilot
    for name in ('controller.json','run.json'):(p.control/name).unlink()
    monkeypatch.setattr(c,'preflight',lambda *a,**kw:(_ for _ in ()).throw(m.MetadataError('Check refused')))
    monkeypatch.setattr(c.subprocess,'Popen',lambda *a,**kw:pytest.fail('launched'))
    with pytest.raises(m.MetadataError):c.supervise(SOURCE,p.config_path,p.control,'a'*40)
    assert not (p.control/'launch.once').exists()


def test_pid_reuse_tree_never_kills_replacement(monkeypatch):
    killed=[]
    tree=object.__new__(c.ProcessTree)
    tree.root=(1,17.)
    tree.known={(1,17.),(2,17.)}
    tree.psutil=SimpleNamespace(Process=lambda pid:SimpleNamespace(kill=lambda:killed.append(pid)))
    monkeypatch.setattr(tree,'refresh',lambda:None)
    monkeypatch.setattr(m,'generation_state',lambda pid,created:'pid_reused')
    assert tree.stop() and killed==[]


def test_explicit_helper_source_rejects_other_checkout(tmp_path):
    with pytest.raises(m.MetadataError,match='source binding'):
        m.checkpoint_reader(tmp_path)


def test_stdlib_reader_subprocess_does_not_import_science(pilot):
    epoch(pilot)
    code='''import sys\nfrom pathlib import Path\nsys.path.insert(0,sys.argv[1])\nimport ept_training_monitor as m\nraw=Path.open\ndef guarded(p,*a,**kw):\n assert p.suffix != '.pt'\n return raw(p,*a,**kw)\nPath.open=guarded\nm.checkpoint_reader(Path(sys.argv[2])).latest_metadata(Path(sys.argv[3]))\nassert not any(x in sys.modules for x in ('torch','numpy','matplotlib','c_ept.cohort'))\n'''
    result=subprocess.run([sys.executable,'-B','-c',code,str(SOURCE/'tools'),str(SOURCE),str(pilot.run/'train/S_cuda')],capture_output=True,text=True,encoding='utf-8',timeout=30)
    assert result.returncode==0,result.stderr


def test_terminal_snapshot_retained_beside_prior_epoch(pilot):
    p=pilot
    epoch(p)
    assert m.save_report(inspect(p),p.out)
    first=next((p.out/'history').glob('epoch_*/snapshot.json'))
    first_bytes=first.read_bytes()
    epoch(p,(4.,4.,5.),terminal=True)
    terminal=inspect(p,missing)
    assert m.save_report(terminal,p.out)
    assert not m.save_report(terminal,p.out)
    snapshots=list((p.out/'history').glob('epoch_*/snapshot.json'))
    assert len(snapshots)==2 and first.read_bytes()==first_bytes
    state=m.read_json(p.out/'monitor-state.json')
    snapshot=m.read_json(p.out/state['snapshot']/'snapshot.json')
    assert snapshot['terminal']=='completed' and snapshot['epoch']==3
    for name,digest in snapshot['files_sha256'].items():
        assert m.sha((p.out/state['snapshot']/name).read_bytes())==digest


def test_observer_detects_interrupted_controller_even_if_owner_alive(pilot):
    report=inspect(pilot,lambda pid:missing(pid) if pid==pilot.controller['controller_pid'] else alive(pid))
    assert report['state']=='interrupted_unconfirmed'


def test_final_render_requires_tree_gone(pilot):
    epoch(pilot,(4.,4.,5.),terminal=True)
    report=inspect(pilot,missing)
    report['controller']['owned_generations']=[{'pid':456789,'create_time':17.}]
    with pytest.raises(m.MetadataError,match='owned process tree'):
        m.render_guard(report,lambda pid:alive(pid) if pid==456789 else missing(pid))


def test_metadata_symlink_refused(tmp_path):
    target=tmp_path/'target.json';write(target,{'ok':True})
    link=tmp_path/'link.json'
    try:link.symlink_to(target)
    except OSError:pytest.skip('Windows symlink privilege unavailable')
    with pytest.raises(m.MetadataError,match='symlink'):m.read_json(link)


def test_supervisor_success_one_launch_and_actual_terminal_metadata(pilot,monkeypatch,fake_job):
    p=pilot
    epoch(p,(4.,4.,5.),terminal=True)
    (p.control/'controller.json').unlink()
    (p.control/'run.json').unlink()
    monkeypatch.setattr(c,'preflight',lambda *a,**kw:(p.config,{}))
    monkeypatch.setattr(c,'source_check',lambda *a:None)
    monkeypatch.setattr(m,'generation_state',lambda *a,**kw:'gone')
    original=m.inspect_run
    monkeypatch.setattr(m,'inspect_run',lambda *a,**kw:original(*a,**kw,probe=missing))
    count=[]
    class Child:
        pid=100000
        def poll(self):return 0
    class Tree:
        root=(100000,17.)
        known={root,(123456,17.)}
        referenced=set()
        def __init__(self,pid,job=None):pass
        def refresh(self):pass
        def bind_owner(self,process):return process['pid'],process['create_time']
        def gone(self):return True
    def popen(*a,**kw):
        count.append(kw)
        assert fake_job.instances and fake_job.instances[-1].events==['created']
        assert a[0][-1]=='-AwaitJobBinding' and kw['stdin']==subprocess.PIPE
        write(p.control/'run.json',p.ref)
        return Child()
    monkeypatch.setattr(c.subprocess,'Popen',popen)
    monkeypatch.setattr(c,'ProcessTree',Tree)
    record=c.supervise(SOURCE,p.config_path,p.control,'a'*40)
    assert record['state']=='completed' and record['exit_code']==0 and len(count)==1
    assert record['wrapper_pid']==100000 and record['owner_pid']==123456
    assert record['wrapper_create_time']==record['owner_create_time']==17.
    assert record['run_ref']==p.ref and record['attempts']==1
    assert fake_job.instances[-1].events==['created','assigned','released','closed']
    assert record['job_ownership']['assigned'] and record['job_ownership']['released']
    assert record['job_closed'] and record['tree_gone'] is True


def test_preflight_self_binding_refused_before_operator_check(pilot,monkeypatch):
    monkeypatch.setattr(c.subprocess,'run',lambda *a,**kw:pytest.fail('subprocess reached before script binding check'))
    with pytest.raises(m.MetadataError,match='explicit reviewed source'):
        c.preflight(pilot.control,pilot.config_path,pilot.control,'a'*40)


def test_final_synthetic_render(pilot):
    p=pilot
    epoch(p,(4.,4.,5.),terminal=True)
    report=inspect(p,missing)
    p.out.mkdir()
    write(p.out/'synthetic-input.json',report)
    code='''import sys\nfrom pathlib import Path\nsys.path.insert(0,sys.argv[1])\nimport ept_training_monitor as m\np=Path(sys.argv[2])\nr=m.read_json(p/'synthetic-input.json')\nassert m.save_report(r,p,render=True,probe=lambda pid:{'kind':'missing'})\nassert not m.save_report(r,p,render=True,probe=lambda pid:{'kind':'missing'})\nassert (p/'loss-final.png').read_bytes().startswith(bytes([137,80,78,71]))\nassert 'torch' not in sys.modules\n'''
    result=subprocess.run([sys.executable,'-B','-c',code,str(SOURCE/'tools'),str(p.out)],capture_output=True,text=True,encoding='utf-8',timeout=60)
    assert result.returncode==0,result.stderr


@pytest.mark.parametrize('generation,expected,code', [('alive','bootstrapping',0),('gone','interrupted_unconfirmed',1),
                                                    ('pid_reused','interrupted_unconfirmed',1),('unconfirmed','interrupted_unconfirmed',1)])
def test_freeze02_bootstrap_controller_generation(pilot,monkeypatch,capsys,generation,expected,code):
    p=pilot
    (p.control/'run.json').unlink()
    monkeypatch.setattr(m,'generation_state',lambda pid,*a,**kw:generation if pid==p.controller['controller_pid'] else 'alive')
    argv=['--source',str(SOURCE),'--config',str(p.config_path),'--control',str(p.control),'--out',str(p.out)]
    assert m.main(argv)==code
    report=m.read_json(p.out/'report.json')
    assert report['state']==expected and report['controller_generation']==generation
    assert report['wrapper_generation']=='alive' and report['run'] is None and report['run_identity_sha256'] is None
    assert capsys.readouterr().out
    assert m.main(argv)==code and not capsys.readouterr().out
    assert not (p.control/'launch.once').exists()


@pytest.mark.parametrize('damage',['schema','source','config','commit'])
def test_freeze02_bootstrap_controller_binding_not_assumed(pilot,monkeypatch,damage):
    p=pilot
    (p.control/'run.json').unlink()
    if damage=='schema':p.controller['schema']='unknown'
    if damage=='source':p.controller['source']=str(p.control)
    if damage=='config':p.controller['operator_config_sha256']='b'*64
    if damage=='commit':p.controller['code_commit']='bad'
    write(p.control/'controller.json',p.controller)
    monkeypatch.setattr(m,'generation_state',lambda *a,**kw:'alive')
    assert m.main(['--source',str(SOURCE),'--config',str(p.config_path),'--control',str(p.control),'--out',str(p.out)])==1
    assert m.read_json(p.out/'report.json')['state']=='observation_error'


def test_freeze02_publication_race_reacquires_whole_snapshot(pilot,monkeypatch):
    p=pilot
    epoch(p)
    original=m.read_json
    crossed=[]
    def crossing(path):
        doc=original(path)
        if path==p.run/'RUN.json' and not crossed:
            crossed.append(True)
            epoch(p,(4.,4.,5.),terminal=True)
            p.process.update(state='running',outcome=None,exit_code=None)
            write(p.run/'PROCESS.json',p.process)
            p.controller['state']='running';write(p.control/'controller.json',p.controller)
        return doc
    monkeypatch.setattr(m,'read_json',crossing)
    report=inspect(p)
    assert report['state']=='running' and report['epoch']==3
    assert report['observation']=={'attempts':2,'consistent_snapshot':True}


@pytest.mark.parametrize('actual_failed',[False,True])
def test_freeze02_stable_corruption_not_actual_failure(pilot,monkeypatch,capsys,actual_failed):
    p=pilot
    epoch(p)
    monkeypatch.setattr(m,'generation_state',lambda *a,**kw:'alive')
    argv=['--source',str(SOURCE),'--config',str(p.config_path),'--control',str(p.control),'--out',str(p.out)]
    assert m.main(argv)==0
    epoch(p,(4.,4.,5.),terminal=True)
    p.process.update(state='failed' if actual_failed else 'running',outcome='failed' if actual_failed else None,exit_code=7 if actual_failed else None)
    write(p.run/'PROCESS.json',p.process)
    p.controller['state']='failed' if actual_failed else 'running'
    write(p.control/'controller.json',p.controller)
    (p.run/'scales.json').write_text('{}',encoding='utf-8')
    with pytest.raises(m.ObservationError) as caught:inspect(p)
    assert caught.value.attempts==3 and caught.value.stable
    assert len(caught.value.evidence['diagnostics'])==3
    assert m.main(argv)==1
    report=m.read_json(p.out/'report.json')
    assert report['state']==('failed' if actual_failed else 'observation_error')
    assert report['epoch']==1 and report['history']['epoch_count']==1
    assert report['process']['state']==p.process['state']
    assert report['observation_error']['attempts']==3
    assert report['observation_error']['stable_bytes'] is True
    with pytest.raises(m.MetadataError,match='observation error'):m.render_guard(report,missing)
    capsys.readouterr()
    before=len(list((p.out/'history').glob('epoch_*/snapshot.json')))
    assert m.main(argv)==1 and not capsys.readouterr().out
    assert len(list((p.out/'history').glob('epoch_*/snapshot.json')))==before


def test_freeze02_repeated_generation_changes_are_bounded(pilot,monkeypatch):
    p=pilot
    epoch(p)
    original=m.read_json
    reads=[]
    def changing(path):
        doc=original(path)
        if path==p.run/'RUN.json':
            reads.append(True)
            write(path,{**doc,'writer_generation':len(reads)})
        return doc
    monkeypatch.setattr(m,'read_json',changing)
    with pytest.raises(m.ObservationError) as caught:inspect(p)
    assert len(reads)==m.SNAPSHOT_ATTEMPTS==3
    assert caught.value.stable is False


@pytest.mark.parametrize('generation',['alive','unconfirmed'])
def test_freeze02_unowned_owner_never_killed_or_cleaned(monkeypatch,generation):
    import psutil
    killed=[]
    monkeypatch.setattr(psutil,'Process',lambda pid:SimpleNamespace(create_time=lambda:17.,children=lambda **kw:[],kill=lambda:killed.append(pid)))
    tree=c.ProcessTree(11111)
    monkeypatch.setattr(m,'generation_state',lambda pid,*a,**kw:'gone' if pid==11111 else generation)
    with pytest.raises(m.MetadataError,match='not in the wrapper process tree'):tree.bind_owner({'pid':22222,'create_time':18.})
    assert tree.stop() is False and tree.gone() is False
    assert killed==[] and (22222,18.) not in tree.known


def test_freeze02_job_retains_owner_after_wrapper_disappears(monkeypatch,fake_job):
    import psutil
    monkeypatch.setattr(psutil,'Process',lambda pid:SimpleNamespace(create_time=lambda:17.,children=lambda **kw:pytest.fail('sampled discovery used with job')))
    job=fake_job()
    tree=c.ProcessTree(11111,job=job)
    monkeypatch.setattr(m,'generation_state',lambda pid,*a,**kw:'gone' if pid==11111 else 'alive')
    assert tree.bind_owner({'pid':22222,'create_time':18.})==(22222,18.)
    assert ('contains',22222,18.) in job.events and (22222,18.) in tree.known
    assert tree.stop() and ('stop',10) in job.events


@pytest.mark.parametrize('contained',[False,'unavailable'])
def test_freeze02_job_refusal_never_adopts_unknown_owner(monkeypatch,fake_job,contained):
    import psutil
    monkeypatch.setattr(psutil,'Process',lambda pid:SimpleNamespace(create_time=lambda:17.))
    job=fake_job()
    def contains(*args):
        if contained=='unavailable':raise OSError('membership unavailable')
        return False
    job.contains=contains
    tree=c.ProcessTree(11111,job=job)
    monkeypatch.setattr(m,'generation_state',lambda pid,*a,**kw:'gone' if pid==11111 else 'alive')
    with pytest.raises((m.MetadataError,OSError)):tree.bind_owner({'pid':22222,'create_time':18.})
    assert (22222,18.) not in tree.known
    assert tree.stop() is False and tree.gone() is False


def test_freeze02_job_unavailable_has_no_launch_fallback(pilot,monkeypatch):
    p=pilot
    for name in ('controller.json','run.json'):(p.control/name).unlink()
    monkeypatch.setattr(c,'preflight',lambda *a,**kw:(p.config,{}))
    monkeypatch.setattr(c,'source_check',lambda *a:None)
    monkeypatch.setattr(c,'WindowsJob',lambda:(_ for _ in ()).throw(OSError('job unavailable')))
    monkeypatch.setattr(c.subprocess,'Popen',lambda *a,**kw:pytest.fail('fallback launch attempted'))
    record=c.supervise(SOURCE,p.config_path,p.control,'a'*40)
    assert record['state']=='failed' and record['wrapper_pid'] is None
    assert 'job unavailable' in record['error'] and record['tree_gone'] is not True
    assert (p.control/'launch.once').exists()


@pytest.mark.parametrize('successful_wrapper',[False,True])
def test_freeze02_job_close_error_always_finalizes(pilot,monkeypatch,fake_job,successful_wrapper):
    import psutil
    p=pilot
    if successful_wrapper:epoch(p,(4.,4.,5.),terminal=True)
    for name in ('controller.json','run.json'):(p.control/name).unlink()
    monkeypatch.setattr(c,'preflight',lambda *a,**kw:(p.config,{}))
    monkeypatch.setattr(c,'source_check',lambda *a:None)
    monkeypatch.setattr(m,'generation_state',lambda *a,**kw:'gone')
    monkeypatch.setattr(psutil,'Process',lambda *a:SimpleNamespace(create_time=lambda:17.))
    def close_error(self):
        self.events.append('close_error')
        raise OSError('synthetic CloseHandle failure; handle retained')
    monkeypatch.setattr(fake_job,'close',close_error)
    launched=[]
    class Child:
        pid=100000
        stdin=io.BytesIO()
        def poll(self):return 0
        def wait(self,timeout):return 0
    def popen(*a,**kw):
        launched.append(a)
        if successful_wrapper:write(p.control/'run.json',p.ref)
        return Child()
    monkeypatch.setattr(c.subprocess,'Popen',popen)
    record=c.supervise(SOURCE,p.config_path,p.control,'a'*40)
    stored=m.read_json(p.control/'controller.json')
    assert len(launched)==1 and record==stored and stored['state']=='failed'
    assert stored['job_closed'] is False and stored['tree_gone'] is False
    assert 'CloseHandle failure' in stored['cleanup_error'] and stored['manual_attention']
    assert (p.control/'launch.once').exists() and fake_job.instances[-1].events[-1]=='close_error'


@pytest.mark.parametrize('after',['gone','pid_reused','alive','unconfirmed'])
def test_freeze02_owner_exit_races_membership_without_adoption(monkeypatch,fake_job,after):
    import psutil
    monkeypatch.setattr(psutil,'Process',lambda pid:SimpleNamespace(create_time=lambda:17.))
    job=fake_job()
    job.contains=lambda *a:False
    tree=c.ProcessTree(11111,job=job)
    observations=iter(['alive',after])
    monkeypatch.setattr(m,'generation_state',lambda *a,**kw:next(observations,after))
    if after in ('gone','pid_reused'):
        assert tree.bind_owner({'pid':22222,'create_time':18.})==(22222,18.)
    else:
        with pytest.raises(m.MetadataError,match='not in the wrapper process tree'):
            tree.bind_owner({'pid':22222,'create_time':18.})
    assert (22222,18.) not in tree.known and (22222,18.) in tree.referenced


@pytest.mark.parametrize('field',['tree_gone','job_gone_before_close','job_closed'])
@pytest.mark.parametrize('value',['absent',False,None,'unconfirmed',1])
def test_freeze03_render_requires_affirmative_whole_job_cleanup(pilot,monkeypatch,field,value):
    p=pilot
    epoch(p,(4.,4.,5.),terminal=True)
    p.controller['state']='failed'
    if value=='absent':p.controller.pop(field)
    else:p.controller[field]=value
    write(p.control/'controller.json',p.controller)
    report=inspect(p,missing)
    assert report['state']=='failed'
    with pytest.raises(m.MetadataError,match='affirmative whole-job cleanup'):
        m.render_guard(report,missing)
    monkeypatch.setattr(m,'final_plot',lambda *a,**kw:pytest.fail('heavy render reached'))
    assert m.save_report(report,p.out,render=True,probe=missing)
    saved=m.read_json(p.out/'report.json')
    assert saved['state']=='failed' and field in saved['render_refused']
    assert (p.out/'loss-live.svg').exists() and (p.out/'epoch_metrics.csv').exists()
    assert not (p.out/'loss-final.png').exists()
    assert m.read_json(p.out/'monitor-state.json')['rendered'] is False
    assert not m.save_report(report,p.out,render=True,probe=missing)


@pytest.mark.parametrize('state',['completed','failed'])
def test_freeze03_confirmed_cleanup_allows_terminal_render_gate(pilot,state):
    epoch(pilot,(4.,4.,5.),terminal=True)
    pilot.controller['state']=state
    write(pilot.control/'controller.json',pilot.controller)
    report=inspect(pilot,missing)
    assert report['state']==state
    m.render_guard(report,missing)


@pytest.mark.parametrize('damage',['absent','kind','assigned','released'])
def test_freeze03_render_retains_launch_owned_containment_binding(pilot,damage):
    epoch(pilot,(4.,4.,5.),terminal=True)
    report=inspect(pilot,missing)
    if damage=='absent':report['controller'].pop('job_ownership')
    elif damage=='kind':report['controller']['job_ownership']['kind']='sampled_pids'
    else:report['controller']['job_ownership'][damage]=False
    with pytest.raises(m.MetadataError,match='containment binding'):m.render_guard(report,missing)


def test_freeze03_cli_render_denial_is_nonzero_and_preserves_light_report(pilot,monkeypatch):
    p=pilot
    epoch(p,(4.,4.,5.),terminal=True)
    p.controller['tree_gone']=False
    write(p.control/'controller.json',p.controller)
    monkeypatch.setattr(m,'generation_state',lambda *a,**kw:'gone')
    monkeypatch.setattr(m,'final_plot',lambda *a,**kw:pytest.fail('heavy render reached'))
    args=['--source',str(SOURCE),'--config',str(p.config_path),'--control',str(p.control),'--out',str(p.out),'--render']
    assert m.main(args)==1
    saved=m.read_json(p.out/'report.json')
    assert saved['state']=='completed' and saved['render_refused']
    assert saved['controller']['tree_gone'] is False and (p.out/'loss-live.svg').exists()
