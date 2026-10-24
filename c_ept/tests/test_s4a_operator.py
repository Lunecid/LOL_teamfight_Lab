"""Synthetic-only operator and actual PowerShell wrapper integration tests."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import types

import pytest

REPO=Path(__file__).resolve().parents[2]
SOURCE=REPO/'c_ept/workflows/s4a_operator.py'
sys.path.insert(0,str(REPO))
spec=importlib.util.spec_from_file_location('operator_candidate',SOURCE)
op=importlib.util.module_from_spec(spec)
spec.loader.exec_module(op)

@pytest.fixture(autouse=True)
def control_flow_policy(monkeypatch):
    # Native startup is exercised in fresh subprocess tests, not pytest embedding.
    monkeypatch.setattr(op,'prepare_native_runtime',lambda:{'test_scope':'control-flow-only'})


@pytest.fixture
def cfg(tmp_path,monkeypatch):
    c={'schema':'s4a-operator-v1','python':sys.executable,
       's1':str(tmp_path/'s1'),'s2':str(tmp_path/'s2'),'s3':str(tmp_path/'s3'),
       'runs_root':str(tmp_path/'runs'),'catalog_db':str(tmp_path/'catalog.sqlite'),
       'sizes':['S','M'],'devices':['cpu','cuda'],'epochs':2,'share':.25,
       'dev_sample':200,'task7_integration_verified':False}
    for k in ('s1','s2','s3','runs_root'):
        Path(c[k]).mkdir()
    for k in ('s1','s2'):
        (Path(c[k])/'STORE.json').write_text('{}')
    for name in ('RUN.json','STAGES.json'):
        (Path(c['s3'])/name).write_text('{}')
    monkeypatch.setattr(op,'load_config',lambda _,**kwargs:c)
    return c

def snapshot(root):
    return {str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}

def test_default_check_readonly_never_imports_runner(cfg,tmp_path,monkeypatch):
    monkeypatch.setattr(op,'runtime_check',lambda:None)
    monkeypatch.setattr(op,'code_check',lambda:'abc')
    monkeypatch.setattr(op,'gpu_metadata',lambda:['synthetic GPU'])
    monkeypatch.setattr(op,'e6_check',lambda:None)
    def no_import(*a):
        raise AssertionError('runner imported by Check')
    monkeypatch.setattr(op.importlib,'import_module',no_import)
    before=snapshot(tmp_path)
    cfg['task7_integration_verified']=True
    assert op.main([])==0  # green Check still never imports runner
    cfg['task7_integration_verified']=False
    assert op.main([])==2  # explicit integration blocker
    assert snapshot(tmp_path)==before
    assert not Path(cfg['catalog_db']).exists()

def test_status_pending_is_unknown_readonly(cfg,tmp_path,monkeypatch,capsys):
    run=Path(cfg['runs_root'])/'20261023_c_ept_s4a_1'
    run.mkdir()
    write_status_fixture(run,process=False)
    (run/'pending.json').write_text('{}')
    monkeypatch.setattr(op,'preflight',lambda c:pytest.fail('status ran preflight'))
    before=snapshot(tmp_path)
    assert op.main(['--action','Status','--run',str(run)])==0
    assert '"state": "unknown"' in capsys.readouterr().out
    assert snapshot(tmp_path)==before
    (run/'COST.json').write_text('{broken')
    assert op.main(['--action','Status','--run',str(run)])==2

def test_start_resume_forward_safe_args_and_exit(cfg,monkeypatch,tmp_path):
    captured=[]
    fake=types.SimpleNamespace(OPERATOR_API_VERSION=1,
        main=lambda args:captured.append(args) or 17)
    monkeypatch.setattr(op,'preflight',lambda c:'abc')
    monkeypatch.setattr(op.importlib,'import_module',lambda name:fake)
    assert op.main(['--action','Start'])==17
    assert '--operator-config-sha256' in captured[0]
    assert '--resume' not in captured[0]
    assert op.main(['--action','Resume'])==2
    run=Path(cfg['runs_root'])/'20261023_c_ept_s4a_2'
    run.mkdir()
    assert op.main(['--action','Resume','--run',str(run)])==17
    assert captured[-1][-4:-2]==['--resume',str(run)]
    fake.OPERATOR_API_VERSION=0
    assert op.main(['--action','Start'])==2

def off_probe(**updates):
    # Complete producer shape; a valid scan can contain unreadable system processes.
    result = {'running':False,'matched':[], 'bridges_ignored':1,'scanned':4,
              'cmdline_unreadable':2,'rule':'synthetic frozen producer rule'}
    result.update(updates)
    return result


BAD_PROBES = [
    None, [], {}, {'running':False},
    {'running':False,'matched':[{'pid':123,'name':'strata.exe'}],'cmdline_unreadable':0},
    {'running':False,'matched':'broken','cmdline_unreadable':None},
    off_probe(running=False,matched=[{'pid':123,'name':'strata.exe'}]),
    off_probe(running=True,matched=[]), off_probe(running=True,matched=None),
    off_probe(running=True,matched=[None]),
    off_probe(running=True,matched=[{'pid':123}]),
    off_probe(running=True,matched=[{'pid':True,'name':'strata.exe'}]),
    off_probe(running=True,matched=[{'pid':-1,'name':'strata.exe'}]),
    off_probe(running=True,matched=[{'pid':123,'name':None}]),
    off_probe(running=True,matched=[{'pid':123,'name':'strata.exe','cmdline':'must not accept'}]),
    off_probe(scanned=-1), off_probe(scanned=True), off_probe(scanned=1.5),
    off_probe(bridges_ignored=-1), off_probe(bridges_ignored=False),
    off_probe(cmdline_unreadable=-1), off_probe(cmdline_unreadable=None),
    off_probe(cmdline_unreadable=True), off_probe(cmdline_unreadable=1.5),
    off_probe(cmdline_unreadable=5), off_probe(bridges_ignored=5),
    off_probe(bridges_ignored=3,cmdline_unreadable=2),
    off_probe(running=0), off_probe(rule=None), off_probe(rule=''),
    off_probe(extra='unsupported'),
    off_probe(running=True,matched=[{'pid':123,'name':'strata.exe'}]),
    off_probe(running=True,matched=[{'pid':'123','name':'strata.exe'}]),
    off_probe(running=True,scanned=0,bridges_ignored=0,cmdline_unreadable=0,
              matched=[{'pid':123,'name':'strata.exe'}]),
    off_probe(running=True,matched=[{'pid':123,'name':'strata.exe'},
                                   {'pid':123,'name':'strata.exe'}]),
    off_probe(running=True,matched=[{'pid':124,'name':'strata.exe'},
                                   {'pid':123,'name':'strata.exe'}]),
]

BAD_PROBES += [{key:value for key,value in off_probe().items() if key != missing}
               for missing in off_probe()]

@pytest.mark.parametrize('result',BAD_PROBES)
@pytest.mark.parametrize('action',['Check','Start','Resume'])
def test_e6_bad_probe_refuses_without_dispatch(result,action,cfg,tmp_path,monkeypatch,capsys):
    cfg['task7_integration_verified']=True
    monkeypatch.setattr(op,'runtime_check',lambda:None)
    monkeypatch.setattr(op,'code_check',lambda:'abc')
    monkeypatch.setattr(op,'gpu_metadata',lambda:['synthetic GPU'])
    monkeypatch.setitem(sys.modules,'c_ept',types.SimpleNamespace(
        runlog=types.SimpleNamespace(strata_check=lambda:result)))
    monkeypatch.setattr(op.importlib,'import_module',lambda name:pytest.fail('E6 failure dispatched runner'))
    run=Path(cfg['runs_root'])/'20261023_c_ept_s4a_1'
    run.mkdir()
    args=['--action',action]
    if action=='Resume':args+=['--run',str(run)]
    before=snapshot(tmp_path)
    assert op.main(args)==2
    assert 'E6' in capsys.readouterr().err
    assert snapshot(tmp_path)==before


@pytest.mark.parametrize('action',['Check','Start','Resume'])
def test_e6_probe_exception_refuses_without_dispatch(action,cfg,tmp_path,monkeypatch):
    cfg['task7_integration_verified']=True
    monkeypatch.setattr(op,'runtime_check',lambda:None)
    monkeypatch.setattr(op,'code_check',lambda:'abc')
    monkeypatch.setattr(op,'gpu_metadata',lambda:['synthetic GPU'])
    def unavailable():raise RuntimeError('synthetic probe failure')
    monkeypatch.setitem(sys.modules,'c_ept',types.SimpleNamespace(
        runlog=types.SimpleNamespace(strata_check=unavailable)))
    monkeypatch.setattr(op.importlib,'import_module',lambda name:pytest.fail('exception dispatched runner'))
    run=Path(cfg['runs_root'])/'20261023_c_ept_s4a_1'
    run.mkdir()
    args=['--action',action]+(['--run',str(run)] if action=='Resume' else [])
    before=snapshot(tmp_path)
    assert op.main(args)==2
    with pytest.raises(op.OperatorError):op.e6_check()
    assert snapshot(tmp_path)==before


@pytest.mark.parametrize('unreadable',[0,2])
def test_e6_valid_off_diagnostic_does_not_block_start(unreadable,cfg,monkeypatch,capsys):
    cfg['task7_integration_verified']=True
    monkeypatch.setattr(op,'runtime_check',lambda:None)
    monkeypatch.setattr(op,'code_check',lambda:'abc')
    monkeypatch.setattr(op,'gpu_metadata',lambda:['synthetic GPU'])
    monkeypatch.setitem(sys.modules,'c_ept',types.SimpleNamespace(
        runlog=types.SimpleNamespace(strata_check=lambda:off_probe(cmdline_unreadable=unreadable))))
    captured=[]
    mock=types.SimpleNamespace(OPERATOR_API_VERSION=1,main=lambda args:captured.append(args) or 17)
    monkeypatch.setattr(op.importlib,'import_module',lambda name:mock)
    assert op.main(['--action','Start'])==17
    assert len(captured)==1
    output=capsys.readouterr().out
    assert 'cmdline_unreadable' in output
    assert '절대' in output


@pytest.mark.parametrize('unreadable',[0,2])
def test_e6_valid_active_operator_error(unreadable,monkeypatch):
    result=off_probe(running=True,matched=[{'pid':123,'name':'strata.exe'}],
                     cmdline_unreadable=unreadable)
    monkeypatch.setitem(sys.modules,'c_ept',types.SimpleNamespace(
        runlog=types.SimpleNamespace(strata_check=lambda:result)))
    with pytest.raises(op.OperatorError,match='123'):op.e6_check()



def test_corrupt_metadata_and_nonfinite(tmp_path):
    p=tmp_path/'RUN.json'
    for text in ('[]','{"x":NaN}','{"x":Infinity}','{"x":1e400}',
                 '{"x":[{"nested":-1e400}]}','{"x":1,"x":2}','not json'):
        p.write_text(text)
        with pytest.raises(op.OperatorError):op.read_json(p)

def test_config_strict_early_validation(tmp_path):
    p=tmp_path/'config.json'
    p.write_text('{"schema":"wrong"}')
    with pytest.raises(op.OperatorError):op.load_config(p)

def test_readonly_git_failure_and_dirty(monkeypatch):
    calls=[]
    def git(args,**kw):
        calls.append(kw)
        return types.SimpleNamespace(returncode=0,stdout='abc' if args[-1]=='HEAD' else ' M file')
    monkeypatch.setattr(op.subprocess,'run',git)
    with pytest.raises(op.OperatorError):op.code_check()
    assert all(x['env']['GIT_OPTIONAL_LOCKS']=='0' for x in calls)

def test_runtime_missing_package(monkeypatch):
    monkeypatch.setattr(op,'read_json',lambda p:{'python_version':op.platform.python_version(),'packages':{'missing-example':'1'}})
    def missing(name):raise op.importlib.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(op.importlib.metadata,'version',missing)
    with pytest.raises(op.OperatorError,match='자동 설치'):op.runtime_check()

def test_digest_tracks_effective_configuration(cfg):
    original=op.config_digest(cfg)
    assert op.config_digest(dict(reversed(list(cfg.items()))))==original
    assert op.config_digest(dict(cfg,s1=cfg['s2']))!=original

@pytest.mark.parametrize('shell',['powershell','pwsh'])
def test_actual_wrapper_quoted_paths_foreign_cwd_exit(tmp_path,shell):
    exe=shutil.which(shell)
    if not exe:pytest.skip(shell+' unavailable')
    repo=tmp_path/'repo with spaces'
    pkg=repo/'c_ept'
    pkg.mkdir(parents=True)
    wrapper=REPO/'run_s4a.ps1'
    shutil.copyfile(wrapper,repo/'run_s4a.ps1')
    # Safe mock operator. No real module/data/runner loaded.
    (pkg/'s4a_operator.py').write_text(
        'import os,sys,json\nfrom pathlib import Path\n'
        'print(json.dumps({"cwd":os.getcwd(),"args":sys.argv[1:],"no_bytecode":sys.dont_write_bytecode}))\n'
        'raise SystemExit(23)\n')
    config=tmp_path/"settings with spaces & $ O'Brien.json"
    config.write_text('{}')
    before=snapshot(tmp_path)
    result=subprocess.run([exe,'-NoProfile','-File',str(repo/'run_s4a.ps1'),
        '-Config',str(config),'-Python',sys.executable],cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==23,(result.stdout,result.stderr)
    result_data=json.loads(result.stdout.strip())
    assert Path(result_data['cwd'])==repo
    assert result_data['args']==['--action','Check','--config',str(config)]
    assert result_data['no_bytecode'] is True
    assert snapshot(tmp_path)==before



def test_gpu_unavailable_refused_without_compute(monkeypatch):
    def missing(*a,**kw):raise FileNotFoundError('nvidia-smi')
    monkeypatch.setattr(op.subprocess,'run',missing)
    with pytest.raises(op.OperatorError):op.gpu_metadata()


def test_protected_payload_never_opened_during_check(cfg,tmp_path,monkeypatch):
    monkeypatch.setattr(op,'runtime_check',lambda:None)
    monkeypatch.setattr(op,'code_check',lambda:'abc')
    monkeypatch.setattr(op,'gpu_metadata',lambda:['synthetic GPU'])
    monkeypatch.setattr(op,'e6_check',lambda:None)
    for key in ('s1','s2'):
        (Path(cfg[key])/'protected.npy').write_bytes(b'never deserialize')
    original=Path.open
    opened=[]
    def guarded(path,*a,**kw):
        opened.append(path.name)
        assert path.suffix not in ('.npy','.joblib','.pkl')
        return original(path,*a,**kw)
    monkeypatch.setattr(Path,'open',guarded)
    assert op.main([])==2
    assert 'STORE.json' in opened


def test_missing_metadata_refused(cfg,monkeypatch):
    monkeypatch.setattr(op,'runtime_check',lambda:None)
    monkeypatch.setattr(op,'code_check',lambda:'abc')
    monkeypatch.setattr(op,'gpu_metadata',lambda:['synthetic GPU'])
    cfg['s1']=str(Path(cfg['s1'])/'absent')
    assert op.main([])==2
@pytest.mark.parametrize('key,value',[('epochs',True),('epochs',3),('share',.5),
                                     ('dev_sample',False),('sizes',['M']),
                                     ('devices',['cpu']),('python','relative.exe'),
                                     ('task7_integration_verified','yes')])
def test_config_rejects_out_of_scope_values(tmp_path,key,value):
    c=json.loads((REPO/'configs/s4a.example.json').read_text())
    c['catalog_db']=str(tmp_path/'catalog.sqlite')
    c[key]=value
    p=tmp_path/'invalid.json'
    p.write_text(json.dumps(c))
    with pytest.raises(op.OperatorError):op.load_config(p)


def test_run_binding_rejects_escape_and_wrong_family(cfg,tmp_path):
    for p in (tmp_path/'20261023_c_ept_s4a_1',Path(cfg['runs_root'])/'20261023_c_ept_s3_1'):
        p.mkdir()
        with pytest.raises(op.OperatorError):op.run_path(p,cfg)


def test_finite_json_values_preserved(tmp_path):
    p=tmp_path/'finite.json'
    p.write_text('{"n":1e300,"values":[-1.5,0.0,2]}')
    assert op.read_json(p)=={'n':1e300,'values':[-1.5,0.0,2]}


def test_e6_matched_unreadable_overlap_is_not_corruption(monkeypatch):
    result=off_probe(running=True,scanned=3,bridges_ignored=1,cmdline_unreadable=2,
                     matched=[{'pid':123,'name':''},{'pid':124,'name':'strata.exe'}])
    monkeypatch.setitem(sys.modules,'c_ept',types.SimpleNamespace(
        runlog=types.SimpleNamespace(strata_check=lambda:result)))
    with pytest.raises(op.OperatorError,match='E6 거부'):op.e6_check()


def test_e6_empty_producer_scan_remains_a_limited_diagnostic(monkeypatch,capsys):
    result=off_probe(scanned=0,bridges_ignored=0,cmdline_unreadable=0)
    monkeypatch.setitem(sys.modules,'c_ept',types.SimpleNamespace(
        runlog=types.SimpleNamespace(strata_check=lambda:result)))
    assert op.e6_check()=={'running':False}
    assert '절대 부재 증명이 아닙니다' in capsys.readouterr().out


@pytest.mark.parametrize('shell',['powershell','pwsh'])
@pytest.mark.parametrize('action,expected',[('check','Check'),('cHeCk','Check'),
    ('start','Start'),('START','Start'),('resume','Resume'),('rEsUmE','Resume'),
    ('status','Status'),('sTaTuS','Status')])
def test_wrapper_canonical_action_mock_only(tmp_path,shell,action,expected):
    exe=shutil.which(shell)
    if not exe:pytest.skip(shell+' unavailable')
    repo=tmp_path/'repo with spaces'
    pkg=repo/'c_ept'
    pkg.mkdir(parents=True)
    shutil.copyfile(REPO/'run_s4a.ps1',repo/'run_s4a.ps1')
    (pkg/'s4a_operator.py').write_text(
        'import json,sys,os\n'
        'print(json.dumps({"args":sys.argv[1:],"cwd":os.getcwd(),"no_bytecode":sys.dont_write_bytecode}))\n'
        'sys.stderr.write("safe mock runner only\\n")\nraise SystemExit(23)\n')
    config=tmp_path/"settings with spaces & $ O'Brien.json"
    config.write_text('{}')
    run=tmp_path/'run with spaces'
    run.mkdir()
    args=[exe,'-NoProfile','-File',str(repo/'run_s4a.ps1'),'-Action',action,
          '-Config',str(config),'-Python',sys.executable]
    expected_args=['--action',expected,'--config',str(config)]
    if expected in ('Resume','Status'):
        args+=['-Run',str(run)]
        expected_args+=['--run',str(run)]
    before=snapshot(tmp_path)
    result=subprocess.run(args,cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==23,(result.stdout,result.stderr)
    payload=json.loads(result.stdout.strip())
    assert payload['args']==expected_args
    assert Path(payload['cwd'])==repo
    assert payload['no_bytecode'] is True
    assert 'safe mock runner only' in result.stderr
    assert snapshot(tmp_path)==before


def test_wrapper_utf8_bom():
    assert (REPO/'run_s4a.ps1').read_bytes().startswith(bytes.fromhex('efbbbf'))


@pytest.mark.parametrize('shell',['powershell','pwsh'])
def test_wrapper_native_missing_python_korean_exact(tmp_path,shell):
    exe=shutil.which(shell)
    if not exe:pytest.skip(shell+' unavailable')
    wrapper=REPO/'run_s4a.ps1'
    missing=tmp_path/'missing-s4a-python.exe'
    # Direct native error path: exits before any Python/operator/runner invocation.
    direct=subprocess.run([exe,'-NoProfile','-File',str(wrapper),'-Action','check',
                           '-Python',str(missing)],capture_output=True)
    assert direct.returncode==2,(direct.stdout,direct.stderr)
    assert direct.stderr
    # Capture ErrorRecord Unicode code points before console transcoding, with
    # no encoding/policy/environment changes. JSON output contains ASCII only.
    harness=tmp_path/'capture-error.ps1'
    harness.write_text('''param([string]$Wrapper,[string]$MissingPython)
$records = @(& $Wrapper -Action check -Python $MissingPython 2>&1)
$wrapperExit = $LASTEXITCODE
$messages = @($records | ForEach-Object { [string]$_.Exception.Message })
$points = @($messages[0].ToCharArray() | ForEach-Object { [int]$_ })
@{ exit_code = $wrapperExit; error_count = $messages.Count; codepoints = $points } | ConvertTo-Json -Compress
''',encoding='utf-8-sig')
    result=subprocess.run([exe,'-NoProfile','-File',str(harness),'-Wrapper',str(wrapper),
                           '-MissingPython',str(missing)],capture_output=True,text=True)
    assert result.returncode==0,(result.stdout,result.stderr)
    payload=json.loads(result.stdout.strip())
    assert payload['exit_code']==2
    assert payload['error_count']==1
    assert ''.join(map(chr,payload['codepoints']))=='Python 경로가 없습니다. -Python으로 기존 인터프리터를 지정하세요.'


@pytest.mark.parametrize('shell',['powershell','pwsh'])
def test_wrapper_default_config_mock_only(tmp_path,shell):
    exe=shutil.which(shell)
    if not exe:pytest.skip(shell+' unavailable')
    repo=tmp_path/'repo with spaces'
    pkg=repo/'c_ept'
    pkg.mkdir(parents=True)
    shutil.copyfile(REPO/'run_s4a.ps1',repo/'run_s4a.ps1')
    (pkg/'s4a_operator.py').write_text(
        'import json,sys\nprint(json.dumps(sys.argv[1:]))\nraise SystemExit(23)\n')
    before=snapshot(tmp_path)
    result=subprocess.run([exe,'-NoProfile','-File',str(repo/'run_s4a.ps1'),
                           '-Python',sys.executable],cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==23,(result.stdout,result.stderr)
    assert json.loads(result.stdout.strip())==['--action','Check','--config',
                                               str(repo/'configs'/'s4a.local.json')]
    assert snapshot(tmp_path)==before

def write_status_fixture(run, *, process=True):
    import datetime as dt
    import hashlib
    identity={'args':{'runs_root':str(run.parent.resolve())},
              'operator_config_sha256':'b'*64,
              'code':{'code_commit':'a'*40,'code_dirty':False,'repo_root':str(run.parent.resolve())},
              'runtime':{'fixture':'한글 runtime'},'inputs':{'fixture':'합성'}}
    digest=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    manifest={'schema':'s4a_run_v1','stage':'s4a','operator_api_version':1,
              'identity':identity,'run_identity_sha256':digest,'code_commit':'a'*40,
              'operator_config_sha256':'b'*64,'code_dirty':False,'runtime':identity['runtime']}
    owner={'schema':'s4a_process_v1','operator_api_version':1,'run_path':str(run.resolve()),
           'run_identity_sha256':digest,'code_commit':'a'*40,'operator_config_sha256':'b'*64,
           'pid':12345,'create_time':1000.25,'state':'running','outcome':None,'exit_code':None,
           'updated_at':dt.datetime.now(dt.timezone.utc).isoformat()}
    (run/'RUN.json').write_text(json.dumps(manifest),encoding='utf-8')
    if process:(run/'PROCESS.json').write_text(json.dumps(owner),encoding='utf-8')
    return manifest, owner


@pytest.mark.parametrize('kind,creation,expected',[('observed',1000.25,'running'),
    ('observed',1000.5,'unknown'),('missing',None,'interrupted_unconfirmed'),
    ('access_denied',None,'unknown'),('unavailable',None,'unknown')])
def test_status_owner_evidence_readonly(tmp_path,monkeypatch,kind,creation,expected):
    run=tmp_path/'20261023_c_ept_s4a_1'
    run.mkdir()
    write_status_fixture(run)
    (run/'pending.json').write_text('{}')
    monkeypatch.setattr(op,'observe_process',lambda pid:{'kind':kind,'create_time':creation})
    before=snapshot(tmp_path)
    result=op.status(run)
    assert result['state']==expected
    assert 'recorded_exit_code' not in result
    assert snapshot(tmp_path)==before


@pytest.mark.parametrize('state,outcome,exitcode,expected',[
    ('finished','completed',0,'recorded_completion'),('failed','failed',1,'recorded_failure'),
    ('failed','failed',130,'recorded_failure'),('failed','failed',None,'recorded_failure')])
def test_status_recorded_outcome_not_os_exit(tmp_path,monkeypatch,state,outcome,exitcode,expected):
    run=tmp_path/'20261023_c_ept_s4a_1'
    run.mkdir()
    manifest,owner=write_status_fixture(run)
    owner.update(state=state,outcome=outcome,exit_code=exitcode)
    (run/'PROCESS.json').write_text(json.dumps(owner))
    monkeypatch.setattr(op,'observe_process',lambda pid:pytest.fail('recorded outcome queried OS exit'))
    before=snapshot(tmp_path)
    result=op.status(run)
    assert result['state']==expected
    assert result['recorded_exit_code']==exitcode
    assert 'OS 종료' in result['message']
    assert snapshot(tmp_path)==before


@pytest.mark.parametrize('key,value',[
    ('schema','wrong'),('operator_api_version',True),('operator_api_version',2),
    ('run_path','relative'),('run_identity_sha256','0'*64),('code_commit','c'*40),
    ('operator_config_sha256',None),('pid',True),('pid',0),('pid',-1),('pid','12345'),
    ('create_time',True),('create_time',0.0),('create_time',-1.0),('create_time','1000.25'),
    ('updated_at','not a timestamp'),('updated_at','2026-10-23T00:00:00'),
    ('updated_at','2026-10-23T00:00:00+09:00'),('state','finished'),('outcome','completed'),
    ('exit_code',0),('current_stage','train_S_cpu')])
def test_status_corrupt_process_binding_refused(tmp_path,key,value):
    run=tmp_path/'20261023_c_ept_s4a_1'
    run.mkdir()
    manifest,owner=write_status_fixture(run)
    owner[key]=value
    (run/'PROCESS.json').write_text(json.dumps(owner))
    before=snapshot(tmp_path)
    with pytest.raises(op.OperatorError):op.status(run)
    assert snapshot(tmp_path)==before


@pytest.mark.parametrize('key,value',[
    ('schema','wrong'),('stage','s3'),('operator_api_version',True),
    ('run_identity_sha256','0'*64),('identity_sha256','alias_only'),
    ('code_commit','c'*40),('operator_config_sha256',None),('code_dirty',True),
    ('identity',{}),('runtime',{'drift':True})])
def test_status_corrupt_run_binding_refused(tmp_path,key,value):
    run=tmp_path/'20261023_c_ept_s4a_1'
    run.mkdir()
    manifest,owner=write_status_fixture(run)
    if key=='identity_sha256':
        manifest['identity_sha256']=manifest.pop('run_identity_sha256')
    else:manifest[key]=value
    (run/'RUN.json').write_text(json.dumps(manifest))
    before=snapshot(tmp_path)
    with pytest.raises(op.OperatorError):op.status(run)
    assert snapshot(tmp_path)==before


def test_status_progress_no_pending_liveness_and_cost_binding(tmp_path):
    run=tmp_path/'20261023_c_ept_s4a_1'
    run.mkdir()
    manifest,owner=write_status_fixture(run,process=False)
    (run/'STAGES.json').write_text(json.dumps({'schema':'s3_stages_v1','stages':{
        'train_S_cpu':{'complete':False}},'calls':[]}))
    (run/'COST.json').write_text(json.dumps({'schema':'s4a_cost_v1',
        'run_identity_sha256':manifest['run_identity_sha256']}))
    before=snapshot(tmp_path)
    result=op.status(run)
    assert result['state']=='unknown'
    assert result['progress']=={'recorded_stages':1,'complete_stages':0,'liveness_inferred':False}
    assert snapshot(tmp_path)==before
    (run/'COST.json').write_text(json.dumps({'schema':'s4a_cost_v1','run_identity_sha256':'0'*64}))
    with pytest.raises(op.OperatorError):op.status(run)


def test_status_skips_runtime_and_e6_with_unsafe_env(cfg,tmp_path,monkeypatch):
    run=Path(cfg['runs_root'])/'20261023_c_ept_s4a_1'
    run.mkdir()
    write_status_fixture(run,process=False)
    monkeypatch.setenv('KMP_DUPLICATE_LIB_OK','TRUE')
    monkeypatch.setenv('MKL_THREADING_LAYER','STATUS_SENTINEL')
    for name in ('prepare_native_runtime','preflight','runtime_check','e6_check'):
        monkeypatch.setattr(op,name,lambda *a,**kw:pytest.fail('Status invoked runtime/E6'))
    before=snapshot(tmp_path)
    assert op.main(['--action','Status','--run',str(run)])==0
    import os
    assert os.environ['MKL_THREADING_LAYER']=='STATUS_SENTINEL'
    assert snapshot(tmp_path)==before


@pytest.fixture
def fresh_operator_repo(tmp_path):
    repo=tmp_path/'fresh repo'
    pkg=repo/'c_ept'
    pkg.mkdir(parents=True)
    (pkg/'__init__.py').write_text('')
    for name in ('s4a_operator.py','s4a_runtime.py'):
        shutil.copyfile(REPO / ('c_ept/workflows/s4a_operator.py' if name=='s4a_operator.py' else 'c_ept/workflows/s4a_runtime.py'),pkg/name)
    (repo/'environment').mkdir()
    shutil.copyfile(REPO/'environment/s4a-runtime.json',repo/'environment/s4a-runtime.json')
    (pkg/'runlog.py').write_text(
        'import os,sys\n'
        'assert os.environ.get("MKL_THREADING_LAYER")=="SEQUENTIAL", "runlog imported before startup"\n'
        'def strata_check():\n'
        ' return {"running":False,"matched":[],"bridges_ignored":1,"scanned":4,"cmdline_unreadable":2,"rule":"fixture"}\n')
    (pkg/'run_s4a.py').write_text(
        'import os\nassert os.environ.get("MKL_THREADING_LAYER")=="SEQUENTIAL"\n'
        'OPERATOR_API_VERSION=1\n'
        'def main(argv):\n print("MOCK_RUNNER_ARGS="+repr(argv))\n return 17\n')
    config=json.loads((REPO/'configs/s4a.example.json').read_text())
    for key in ('s1','s2','s3'):
        folder=repo/key
        folder.mkdir()
        config[key]=str(folder)
    for key in ('s1','s2'):(Path(config[key])/'STORE.json').write_text('{}')
    for name in ('RUN.json','STAGES.json'):(Path(config['s3'])/name).write_text('{}')
    config['python']=sys.executable
    config['task7_integration_verified']=True  # synthetic fixture only; integration release flag is verified separately
    path=repo/'config.json'
    path.write_text(json.dumps(config))
    harness=repo/'fixture-main.py'
    harness.write_text(
        'import json,os,sys\nfrom c_ept import s4a_operator as op\n'
        'op.code_check=lambda:"a"*40\nop.gpu_metadata=lambda:["synthetic metadata"]\n'
        'original=op.load_config\n'
        'def load(*a,**kw):\n'
        ' assert os.environ.get("MKL_THREADING_LAYER")=="SEQUENTIAL", "loader before startup"\n'
        ' return original(*a,**kw)\n'
        'op.load_config=load\n'
        'code=op.main(sys.argv[1:])\n'
        'print("PHASEA_RESULT="+json.dumps({"code":code,"runner_loaded":"c_ept.run_s4a" in sys.modules, '
        '"native_loaded":[x for x in ("numpy","torch","scipy") if x in sys.modules], '
        '"mkl":os.environ.get("MKL_THREADING_LAYER")}))\n'
        'raise SystemExit(code)\n')
    return repo,path,harness


def child_environment(repo):
    import os
    env=dict(os.environ,PYTHONPATH=str(repo),PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8',
             MKL_THREADING_LAYER='INTEL',OMP_NUM_THREADS='2',MKL_NUM_THREADS='3',OPENBLAS_NUM_THREADS='4')
    env.pop('KMP_DUPLICATE_LIB_OK',None)
    return env


def test_fresh_process_default_check_startup_readonly_no_runner(fresh_operator_repo):
    repo,config,harness=fresh_operator_repo
    before=snapshot(repo)
    result=subprocess.run([sys.executable,'-B',str(harness),'--config',str(config)],
        cwd=repo,capture_output=True,text=True,encoding='utf-8',env=child_environment(repo))
    assert result.returncode==0,(result.stdout,result.stderr)
    data=json.loads(result.stdout.split('PHASEA_RESULT=')[1])
    assert data=={'code':0,'runner_loaded':False,'native_loaded':[],'mkl':'SEQUENTIAL'}
    assert '"OMP_NUM_THREADS": "2"' in result.stdout
    assert '"MKL_NUM_THREADS": "3"' in result.stdout
    assert '"OPENBLAS_NUM_THREADS": "4"' in result.stdout
    assert '"numpy_mkl_parallelism": "sequential"' in result.stdout
    assert 'cmdline_unreadable' in result.stdout
    assert snapshot(repo)==before


@pytest.mark.parametrize('value',['TRUE','1','custom'])
@pytest.mark.parametrize('action',['Check','Start','Resume'])
def test_direct_fresh_module_unsafe_kmp_before_loader(fresh_operator_repo,value,action):
    repo,config,harness=fresh_operator_repo
    env=child_environment(repo)
    env['KMP_DUPLICATE_LIB_OK']=value
    before=snapshot(repo)
    result=subprocess.run([sys.executable,'-B','-m','c_ept.s4a_operator','--action',action,
                           '--config',str(repo/'NONEXISTENT.json')],cwd=repo,
                          capture_output=True,text=True,encoding='utf-8',env=env)
    assert result.returncode==2
    assert 'KMP_DUPLICATE_LIB_OK' in result.stderr
    assert 'NONEXISTENT' not in result.stderr
    assert 'MOCK_RUNNER_ARGS' not in result.stdout
    assert snapshot(repo)==before


def test_fresh_start_mock_shared_policy_then_api_dispatch(fresh_operator_repo):
    repo,config,harness=fresh_operator_repo
    before=snapshot(repo)
    result=subprocess.run([sys.executable,'-B',str(harness),'--action','Start','--config',str(config)],
        cwd=repo,capture_output=True,text=True,encoding='utf-8',env=child_environment(repo))
    assert result.returncode==17,(result.stdout,result.stderr)
    assert 'MOCK_RUNNER_ARGS=' in result.stdout
    assert '--operator-config-sha256' in result.stdout
    assert snapshot(repo)==before

def test_actual_primary_metadata_writer_and_api_synthetic(tmp_path):
    # Read current primary source only. No execute/main/train call, source edits,
    # real input stores or active run outputs. Primary retains sole ownership.
    primary=Path(op.ROOT)/'c_ept/workflows/run_s4a.py'
    assert primary.is_file(), 'integrated package must include its own primary runner'
    script=tmp_path/'primary-contract-check.py'
    script.write_text('''import hashlib,json,sys,types
from pathlib import Path
from c_ept import s4a_operator as op
from c_ept.s4a_runtime import prepare_native_runtime
policy=prepare_native_runtime()
source=Path(sys.argv[1])
content=source.read_bytes()
runner=types.ModuleType('synthetic_primary_run_s4a')
runner.__file__=str(source)
exec(compile(content,str(source),'exec'),runner.__dict__)
assert 'numpy' not in sys.modules and 'torch' not in sys.modules
assert runner.OPERATOR_API_VERSION==1
base=Path(sys.argv[2]).resolve()
run=base/'20261023_c_ept_s4a_1'
run.mkdir()
c=json.loads((Path(op.ROOT)/'configs/s4a.example.json').read_text())
c['runs_root']=str(base)
for key in ('s1','s2','s3'):c[key]=str(base/key)
digest=op.config_digest(c)
args=runner._parser().parse_args([*op.runner_args(c),'--operator-config-sha256',digest])
normal,operator_digest=runner._normal_args(args)
assert operator_digest==digest and 'resume' not in normal and 'operator_config_sha256' not in normal
resumed=runner._parser().parse_args([*op.runner_args(c,run),'--operator-config-sha256',digest])
assert runner._normal_args(resumed)==(normal,digest)
identity={'args':normal,'operator_config_sha256':digest,
 'code':{'code_commit':'a'*40,'code_dirty':False,'repo_root':str(source.parent.parent.resolve())},
 'runtime':policy,'inputs':{'fixture':'합성 metadata only'}}
manifest={'schema':runner.RUN_SCHEMA,'stage':'s4a','operator_api_version':runner.OPERATOR_API_VERSION,
 'code_commit':'a'*40,'code_dirty':False,'identity':identity,'run_identity_sha256':runner._digest(identity),
 'operator_config_sha256':digest,'runtime':policy,'stages':[],'results':{}}
(run/'RUN.json').write_text(json.dumps(manifest),encoding='utf-8')
record=runner._process_record(run,manifest,'running')
assert 'current_stage' not in record
(run/'PROCESS.json').write_text(json.dumps(record),encoding='utf-8')
assert op.status(run)['state']=='running'  # actual live PID+psutil creation time
record['create_time']+=1
(run/'PROCESS.json').write_text(json.dumps(record),encoding='utf-8')
assert op.status(run)['state']=='unknown'
for state,outcome,code,expected in [('finished','completed',0,'recorded_completion'),('failed','failed',130,'recorded_failure')]:
 (run/'PROCESS.json').write_text(json.dumps(runner._process_record(run,manifest,state,outcome,code)),encoding='utf-8')
 before={p.name:p.read_bytes() for p in run.iterdir()}
 assert op.status(run)['state']==expected
 assert {p.name:p.read_bytes() for p in run.iterdir()}==before
assert content==source.read_bytes(), 'primary changed while this contract snapshot was tested'
print(json.dumps({'primary_source_sha256':hashlib.sha256(content).hexdigest(),
 'actual_api_parse_and_normalized_resume':'passed','actual_process_writer_live_pid_status':'passed',
 'recorded_outcomes':'passed','native_modules_imported':False}))
''',encoding='utf-8')
    repo=REPO
    result=subprocess.run([sys.executable,'-B',str(script),str(primary),str(tmp_path)],cwd=tmp_path,
        capture_output=True,text=True,encoding='utf-8',env=child_environment(repo))
    assert result.returncode==0,(result.stdout,result.stderr)
    data=json.loads(result.stdout.strip())
    (tmp_path/'primary-integration-evidence.json').write_text(json.dumps(data,indent=2))
    assert data['actual_api_parse_and_normalized_resume']=='passed'
    assert data['actual_process_writer_live_pid_status']=='passed'

def test_status_config_does_not_require_inputs_or_python_presence(tmp_path):
    c=json.loads((REPO/'configs/s4a.example.json').read_text())
    c['python']=str(tmp_path/'missing-python.exe')
    for key in ('s1','s2','s3'):c[key]=str(tmp_path/('missing-'+key))
    path=tmp_path/'status-config.json'
    path.write_text(json.dumps(c))
    before=snapshot(tmp_path)
    assert op.load_config(path,require_inputs=False)==c
    with pytest.raises(op.OperatorError):op.load_config(path)
    assert snapshot(tmp_path)==before


@pytest.mark.parametrize('filename,text',[
    ('PROCESS.json','{"create_time":1e400}'),
    ('RUN.json','{"identity":{"runtime":{"n":-1e400}}}'),
    ('COST.json','{"elapsed":NaN}')])
def test_status_nonfinite_metadata_refused_readonly(tmp_path,filename,text):
    run=tmp_path/'20261023_c_ept_s4a_1'
    run.mkdir()
    write_status_fixture(run)
    (run/filename).write_text(text)
    before=snapshot(tmp_path)
    with pytest.raises(op.OperatorError):op.status(run)
    assert snapshot(tmp_path)==before


def test_status_direct_cli_null_operator_digest_valid(tmp_path,monkeypatch):
    import hashlib
    run=tmp_path/'20261023_c_ept_s4a_1'
    run.mkdir()
    manifest,owner=write_status_fixture(run)
    manifest['identity']['operator_config_sha256']=None
    digest=hashlib.sha256(json.dumps(manifest['identity'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    manifest.update(operator_config_sha256=None,run_identity_sha256=digest)
    owner.update(operator_config_sha256=None,run_identity_sha256=digest)
    (run/'RUN.json').write_text(json.dumps(manifest))
    (run/'PROCESS.json').write_text(json.dumps(owner))
    monkeypatch.setattr(op,'observe_process',lambda pid:{'kind':'observed','create_time':owner['create_time']})
    assert op.status(run)['state']=='running'


@pytest.mark.parametrize('key,value',[
    ('state','not-a-state'),('exit_code',False),('create_time',None),
    ('run_path',str(Path('C:/')/'invalid\x00run'))])
def test_status_additional_invalid_process_fields(tmp_path,key,value):
    run=tmp_path/'20261023_c_ept_s4a_1'
    run.mkdir()
    manifest,owner=write_status_fixture(run)
    owner[key]=value
    (run/'PROCESS.json').write_text(json.dumps(owner))
    with pytest.raises(op.OperatorError):op.status(run)


def test_integrated_example_declares_verified_api_without_weakening_guards():
    config=json.loads((Path(op.ROOT)/'configs/s4a.example.json').read_text(encoding='utf-8'))
    assert config['schema']=='s4a-operator-v1'
    assert config['task7_integration_verified'] is True
    assert config['epochs']==2 and config['share']==.25 and config['dev_sample']==200
    assert config['sizes']==['S','M'] and config['devices']==['cpu','cuda']
    assert (Path(op.ROOT)/'c_ept/workflows/run_s4a.py').is_file()
    assert '/configs/s4a.local.json' in (Path(op.ROOT)/'.gitignore').read_text(encoding='utf-8').splitlines()
