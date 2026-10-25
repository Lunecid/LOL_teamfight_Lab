"""S4a synthetic runtime and execution-contract tests; no actual data/GPU."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from c_ept.workflows.s4a_runtime import RuntimeRefused, prepare_native_runtime, validate_e6_probe


@pytest.fixture(autouse=True)
def supported_runtime_test_environment(monkeypatch):
    """Safe-path tests set an explicit disabled prerequisite; restore caller state.

    sklearn may enable its default during collection. Unsafe-input tests override
    this reversible fixture themselves; production refusal is never bypassed.
    """
    monkeypatch.setenv('KMP_DUPLICATE_LIB_OK','FALSE')


def probe(**changes):
    return dict(running=False, matched=[], bridges_ignored=0, scanned=2, cmdline_unreadable=1, rule='original-rule', **changes)


@pytest.mark.parametrize('value', ['true','TRUE','1','yes','on','unexpected'])
def test_unsafe_duplicate_override_refused(monkeypatch, value):
    monkeypatch.setenv('KMP_DUPLICATE_LIB_OK', value)
    with pytest.raises(RuntimeRefused, match='KMP_DUPLICATE_LIB_OK'):
        prepare_native_runtime()


def test_native_policy_sets_only_current_process_environment(monkeypatch):
    monkeypatch.delenv('KMP_DUPLICATE_LIB_OK', raising=False)
    monkeypatch.setenv('MKL_THREADING_LAYER','SEQUENTIAL')
    rec = prepare_native_runtime()
    assert rec['mkl_threading_layer'] == 'SEQUENTIAL'
    assert rec['kmp_duplicate_lib_ok'] == 'disabled'


@pytest.mark.parametrize('change', [dict(running=0),dict(matched=None),dict(scanned=True),dict(cmdline_unreadable=-1),dict(rule=''),dict(matched=[dict(pid=True,name='x')]),dict(bridges_ignored=3)])
def test_e6_schema_refuses_invalid(change):
    rec=probe(); rec.update(change)
    with pytest.raises(RuntimeRefused, match='E6'):
        validate_e6_probe(rec)


def test_e6_unreadable_diagnostic_is_not_new_gate():
    rec = probe()
    assert validate_e6_probe(rec) is rec
    active=probe(); active.update(running=True,matched=[dict(pid=1,name='strata.exe')],scanned=1,cmdline_unreadable=1)
    with pytest.raises(RuntimeRefused,match='E6'):
        validate_e6_probe(active)
    with pytest.raises(RuntimeRefused,match='E6'):
        validate_e6_probe(None)


def test_runtime_helper_import_is_stdlib_only():
    code="import sys; import c_ept.s4a_runtime; assert not any(x in sys.modules for x in ('numpy','torch','scipy','c_ept.run_s4a','c_ept.run_s3'))"
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('result', [None,probe()])
def test_guard_optional_validator_single_probe_and_legacy_default(monkeypatch,result):
    from c_ept.workflows import run_s1, run_s3
    calls=[]
    monkeypatch.setattr(run_s1,'require_no_strata',lambda:(calls.append('probe'),result)[1])
    default=run_s3._Guard()
    assert default() is None
    received=[]
    strict=run_s3._Guard(validate=received.append)
    assert strict() is None
    assert received == [result] and calls == ['probe','probe']
    assert default.calls == strict.calls == 1


def test_guard_validator_error_keeps_finally_counters(monkeypatch):
    from c_ept.workflows import run_s1, run_s3
    monkeypatch.setattr(run_s1,'require_no_strata',lambda:None)
    g=run_s3._Guard(validate=validate_e6_probe)
    with g.window() as window:
        with pytest.raises(RuntimeRefused,match='E6'):
            g()
    assert g.calls == 1 and g.seconds >= 0 and window.counters()['check_calls'] == 1


def test_projection_arithmetic_matches_a_hand_case():
    from c_ept.workflows import run_s4a as r
    assert r.project_cost(10.,2.,4.,3,1) == 122.
    assert r.project_cost(10.,2.,4.,3,2) == 244.
    with pytest.raises(RuntimeRefused):
        r.project_cost(float('nan'),2,4,3,1)


def test_whole_repository_guard_is_readonly_and_not_c_ept_scoped(monkeypatch,tmp_path):
    from types import SimpleNamespace
    from c_ept.workflows import run_s4a as r
    calls=[]
    def git(argv,**kw):
        calls.append((argv,kw))
        if 'HEAD' in argv:
            text='a'*40
        elif '--show-toplevel' in argv:
            text=str(tmp_path)
        else:
            text=' M docs/changed.md'
        return SimpleNamespace(returncode=0,stdout=text,stderr='')
    monkeypatch.setattr(r.subprocess,'run',git)
    with pytest.raises(RuntimeRefused,match='whole-repository'):
        r.repository_state(tmp_path)
    assert all(kw['env']['GIT_OPTIONAL_LOCKS']=='0' for _,kw in calls)
    assert '--untracked-files=all' in calls[-1][0] and 'c_ept' not in calls[-1][0]


def test_resume_checks_args_runtime_config_and_input_even_with_same_digest():
    from c_ept.workflows import run_s4a as r
    base=dict(code=dict(code_commit='a'*40,code_dirty=False),args=dict(epochs=2),runtime=dict(mkl='SEQUENTIAL'),inputs=dict(s1='a'),operator_config_sha256='b'*64)
    manifest=dict(schema=r.RUN_SCHEMA,stage='s4a',operator_api_version=1,code_commit='a'*40,code_dirty=False,identity=base,run_identity_sha256=r._digest(base),operator_config_sha256=base['operator_config_sha256'])
    r.verify_resume(manifest,base)
    for key,value in [('args',dict(epochs=3)),('runtime',dict(mkl='INTEL')),('inputs',dict(s1='b')),('operator_config_sha256',None)]:
        with pytest.raises(RuntimeRefused,match='identity'):
            r.verify_resume(manifest,{**base,key:value})


def test_lock_real_process_contention_and_owner_exit_recovery(tmp_path):
    from c_ept.workflows import run_s4a as r
    owner_code='from c_ept.run_s4a import run_lock; import sys; from pathlib import Path\nwith run_lock(Path(sys.argv[1])):\n print("READY",flush=True)\n sys.stdin.readline()\n'
    contender_code='from c_ept.run_s4a import run_lock; from pathlib import Path; import sys\nwith run_lock(Path(sys.argv[1])): print("ACQUIRED")\n'
    owner=subprocess.Popen([sys.executable,'-c',owner_code,str(tmp_path)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        assert owner.stdout.readline().strip() == 'READY'
        other=subprocess.run([sys.executable,'-c',contender_code,str(tmp_path)],capture_output=True,text=True,timeout=15)
        assert other.returncode != 0 and 'another process' in other.stderr
    finally:
        owner.communicate('exit\n',timeout=15)
    assert owner.returncode == 0 and (tmp_path/'S4a.lock').is_file()
    crash=contender_code.replace('print("ACQUIRED")','__import__("os")._exit(17)')
    assert subprocess.run([sys.executable,'-c',crash,str(tmp_path)],capture_output=True,timeout=15).returncode == 17
    with r.run_lock(tmp_path):
        assert (tmp_path/'S4a.lock').is_file()


@pytest.mark.parametrize('inherited_encoding',['utf-8','cp949'])
def test_direct_cli_refuses_unsafe_override_before_native_import(monkeypatch,inherited_encoding):
    monkeypatch.setenv('PYTHONIOENCODING',inherited_encoding)
    env=dict(os.environ,KMP_DUPLICATE_LIB_OK='TRUE',PYTHONIOENCODING='utf-8')
    result=subprocess.run([sys.executable,'-m','c_ept.run_s4a'],env=env,capture_output=True,text=True,encoding='utf-8',timeout=15)
    assert result.returncode == 2 and 'KMP_DUPLICATE_LIB_OK' in result.stderr


def test_every_model_file_including_metadata_is_verified(tmp_path):
    from c_ept.workflows import run_s4a as r
    arrays=tmp_path/'b0_heads.npz'; arrays.write_bytes(b'synthetic arrays')
    meta=tmp_path/'b0_heads.json'; meta.write_text(json.dumps(dict(npz_sha256=r._sha(arrays))),encoding='utf-8')
    ledger={'fit/b0/b0_heads.json':r._sha(meta)}
    hashes=r._model_hashes(tmp_path,'b0_heads.json','b0_heads.npz',ledger,'fit/b0')
    assert len(hashes)==2
    arrays.write_bytes(b'corrupt')
    with pytest.raises(RuntimeRefused,match='arrays'):
        r._model_hashes(tmp_path,'b0_heads.json','b0_heads.npz',ledger,'fit/b0')
    meta.write_text('{}',encoding='utf-8')
    with pytest.raises(RuntimeRefused,match='metadata'):
        r._model_hashes(tmp_path,'b0_heads.json','b0_heads.npz',ledger,'fit/b0')



def test_completed_stage_revalidates_actual_output_and_skips_action(tmp_path):
    from c_ept.workflows import run_s4a as r, run_s3
    manifest={'stages':['stats'],'results':{}}
    ledger=run_s3.Ledger(tmp_path)
    artifact=tmp_path/'payload.txt'
    calls=[]
    def action(check):
        check(); calls.append(1); artifact.write_text('original')
        return {'observed':7},[artifact]
    assert r._stage(tmp_path,manifest,ledger,'stats',{},lambda:None,action)=={'observed':7}
    assert r._stage(tmp_path,manifest,ledger,'stats',{},lambda:None,action)=={'observed':7}
    assert calls==[1]
    artifact.write_text('tampered')
    with pytest.raises(RuntimeRefused,match='output'):
        r._stage(tmp_path,manifest,ledger,'stats',{},lambda:None,action)


def test_final_cost_is_integrity_bound_and_not_rewritten_on_resume(tmp_path):
    from c_ept.workflows import run_s4a as r
    manifest={'run_identity_sha256':'a'*64}
    cost={'schema':r.COST_SCHEMA,'run_identity_sha256':'a'*64,'seconds':7}
    r._finalize_cost(tmp_path,manifest,cost)
    before=(tmp_path/'COST.json').read_bytes()
    r._finalize_cost(tmp_path,manifest,{**cost,'seconds':99})
    assert (tmp_path/'COST.json').read_bytes()==before
    (tmp_path/'COST.json').write_text('{}')
    with pytest.raises(RuntimeRefused,match='COST'):
        r._finalize_cost(tmp_path,manifest,cost)


def test_process_metadata_matches_exact_run_identity_without_inventing_stage(tmp_path):
    from c_ept.workflows import run_s4a as r
    manifest={'run_identity_sha256':'a'*64,'code_commit':'b'*40,
              'identity':{'operator_config_sha256':None}}
    record=r._process_record(tmp_path,manifest,'running')
    assert record['run_identity_sha256']==manifest['run_identity_sha256']
    assert record['pid']==os.getpid() and record['create_time']>0
    assert record['state']=='running' and record['exit_code'] is None
    assert 'current_stage' not in record


def test_train_stage_executes_real_tiny_cpu_model_and_persists_components(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from c_ept.workflows import run_s4a as r
    from c_ept.ept import data,encoder,loss
    from c_ept.tests.ept_toys import build_ept_world,toy_static
    world=build_ept_world(tmp_path/'synthetic',n_events=3,n_edges=2)
    bundle=SimpleNamespace(s1=world.s1,s2=world.s2,labels=world.labels,static=toy_static(),meta={'terrain_files_sha256':{'toy':'a'*64}})
    args=SimpleNamespace(batch_size=16,seed=3,lr=.001,weight_decay=.01,warmup_steps=1,epochs=1)
    monkeypatch.setattr(r,'_model_config',lambda size:encoder.EncoderConfig(d=16,layers=2,heads=2,kappas=1,field_channels=2))
    fitted,files=r._train_stage(tmp_path,args,bundle,data.fit_stats(world.s1,world.s2),
                              {key:1. for key in loss.BLOCKS},data.proxy_matches(world.s1,1.,3),'S','cpu',lambda:None)
    assert fitted['epochs_run']==1 and not fitted['recovered']
    assert fitted['whole_call_seconds']>0 and fitted['whole_call_peak_bytes']>0
    assert fitted['timing_components']['epochs']['forward_seconds'][0]>0
    assert fitted['timing_components']['checkpoint_io_current_call']
    assert any(p.name=='head.json' for p in files)
    model=r._best_model(tmp_path/fitted['checkpoint'],'S','cpu',bundle.static)
    assert not model.training


@pytest.mark.parametrize('failure',[None,dict(running=True,matched=[dict(pid=1,name='strata')],bridges_ignored=0,scanned=1,cmdline_unreadable=0,rule='frozen')])
def test_execute_e6_refuses_before_input_or_run_mutation(tmp_path,monkeypatch,failure):
    from c_ept.workflows import run_s4a as r, run_s1
    from c_ept import runlog
    args=r._parser().parse_args(['--s1','synthetic-s1','--s2','synthetic-s2','--s3','synthetic-s3','--runs-root',str(tmp_path)])
    monkeypatch.setattr(run_s1,'require_no_strata',lambda:failure)
    def forbidden(*a,**kw):
        pytest.fail('must refuse before code/input/newrun')
    monkeypatch.setattr(r,'inspect_inputs',forbidden)
    monkeypatch.setattr(r,'repository_state',forbidden)
    monkeypatch.setattr(runlog,'new_run',forbidden)
    with pytest.raises(RuntimeRefused,match='E6'):
        r.execute(args,prepare_native_runtime())
    assert not list(tmp_path.iterdir())


def test_output_roots_cannot_mutate_sources_or_frozen_trees(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from c_ept.workflows import run_s4a as r
    allowed=tmp_path/'runs'
    monkeypatch.setattr(r,'APPROVED_RUNS_ROOT',allowed)
    args=SimpleNamespace(runs_root=str(allowed),catalog_db=str(tmp_path/'catalog.sqlite'))
    meta={'paths':{'s1':str(allowed/'source/s1'),'s2':str(allowed/'source2/s2'),'s3':str(allowed/'source3')}}
    r._output_paths(args,meta)
    args.runs_root=str(tmp_path/'arbitrary')
    with pytest.raises(RuntimeRefused,match='output'):
        r._output_paths(args,meta)
    args.runs_root=str(allowed)
    args.catalog_db=str(allowed/'source3/catalog.sqlite')
    with pytest.raises(RuntimeRefused,match='source'):
        r._output_paths(args,meta)


def test_fresh_process_shared_bootstrap_allows_torch_and_real_tier_probability():
    code='''from c_ept.s4a_runtime import prepare_native_runtime
prepare_native_runtime()
import torch
assert (torch.ones(2,2) @ torch.ones(2,2)).tolist()==[[2.,2.],[2.,2.]]
from c_ept.tests.s3_toys import toy_tier_model
value=toy_tier_model().prob([[0,0,0,0]],[0],[600.])
assert value.shape==(1,4) and abs(value.sum()-1.)<1e-12
'''
    env=dict(os.environ)
    env.pop('MKL_THREADING_LAYER',None)
    env.pop('KMP_DUPLICATE_LIB_OK',None)
    completed=subprocess.run([sys.executable,'-c',code],env=env,capture_output=True,text=True,timeout=45)
    assert completed.returncode==0,completed.stderr


def test_synthetic_cpu_pipeline_cost_schema_resume_and_protected_payloads(tmp_path,monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    from c_ept.workflows import run_s4a as r, run_s1
    from c_ept.preprocessing import features
    from c_ept import config as C
    from c_ept.ept import encoder,loss,scales
    from c_ept.tests.ept_toys import build_ept_world,toy_static
    from c_ept.tests.s3_toys import _put,toy_tier_model,TOY_COMPOSE_PRICES,TOY_GOLD_REPS
    world=build_ept_world(tmp_path/'world',roles=('FIT','FIT','STOP','DEV','CAL','AUDIT','TEST','EXTERNAL'),n_events=3,n_edges=2)
    _put(world.s1,'matches/obj_off',np.zeros(9),'i8')
    for name in ('t','kind','sub','taker'):
        _put(world.s1,'objectives/'+name,[], 'i8')
    world.s1.finalize()
    _put(world.labels,'labels/gold_reps',TOY_GOLD_REPS,'f8')
    world.labels.finalize()
    permitted=features.split_views(world.s1,world.s2,('FIT','STOP','DEV'))
    reads=[]
    class GuardedArray:
        def __init__(self,array): self.raw=array; self.shape=array.shape
        def __getitem__(self,key):
            selection=key[0] if isinstance(key,tuple) else key
            assert np.all(permitted[np.arange(len(permitted))[selection]]),'protected raw S2 payload access'
            reads.append(1)
            return self.raw[key]
        def __array__(self,*a,**kw):
            raise AssertionError('whole raw S2 materialization forbidden')
    class GuardedStore:
        def __getattr__(self,name): return getattr(world.s2,name)
        def array(self,name):
            raw=world.s2.array(name)
            return GuardedArray(raw) if name.startswith('x/') else raw
        def verify(self): raise AssertionError('whole S2 verify forbidden')
    source=tmp_path/'source3'; source.mkdir()
    toy_tier_model().save(source/'fit/tier')
    scales_path=source/scales.SCALES_NAME
    scales_path.write_text('{}')
    factors={key:1. for key in loss.BLOCKS}
    bundle=SimpleNamespace(s1=world.s1,s2=GuardedStore(),labels=world.labels,static=toy_static(),prices=TOY_COMPOSE_PRICES)
    meta={'paths':{'s1':str(world.s1.root),'s2':str(world.s2.root),'s3':str(source)},'expected_provenance':{'synthetic':True},'terrain_files_sha256':{'synthetic':'a'*64}}
    root=tmp_path/'runs'; root.mkdir()
    monkeypatch.setattr(r,'APPROVED_RUNS_ROOT',root)
    monkeypatch.setattr(r,'repository_state',lambda:dict(code_commit='a'*40,code_dirty=False,repo_root=str(tmp_path)))
    monkeypatch.setattr(r,'inspect_inputs',lambda args:meta)
    monkeypatch.setattr(r,'_load_inputs',lambda meta:bundle)
    monkeypatch.setattr(run_s1,'require_no_strata',lambda:probe())
    monkeypatch.setattr(scales,'load_scales',lambda path,expected_provenance:factors)
    monkeypatch.setattr(r,'_model_config',lambda size:encoder.EncoderConfig(d=16,layers=2,heads=2,kappas=1,field_channels=2))
    flags=['--s1','toy1','--s2','toy2','--s3',str(source),'--runs-root',str(root),'--catalog-db',str(tmp_path/'catalog.sqlite'),'--sizes','S','--devices','cpu','--epochs','1','--share','1','--dev-sample','1']
    assert r.main(flags)==0
    run=next(root.iterdir())
    cost=json.loads((run/'COST.json').read_text())
    manifest=json.loads((run/'RUN.json').read_text())
    process=json.loads((run/'PROCESS.json').read_text())
    assert cost['schema']==r.COST_SCHEMA and cost['runtime']['mkl_threading_layer']=='SEQUENTIAL'
    assert cost['run_identity_sha256']==process['run_identity_sha256']==r._digest(manifest['identity'])
    assert cost['training'][0]['epochs_run']==1 and cost['projections'][0]['available']
    assert cost['compose']['measurements'][0]['S']==200 and reads
    assert process['state']=='finished' and process['outcome']=='completed'
    before=(run/'COST.json').read_bytes()
    assert r.main(flags+['--resume',str(run)])==0
    assert (run/'COST.json').read_bytes()==before


def test_resumed_training_is_not_uninterrupted_projection_calibration():
    from c_ept.workflows import run_s4a as r
    fitted={'size':'S','device':'cpu','recovered':False,'resumed':True}
    result=r._projections([fitted])['projections'][0]
    assert result['available'] is False and 'uninterrupted' in result['reason']


def test_resume_cannot_create_lock_inside_requested_source(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from c_ept.workflows import run_s4a as r
    root=tmp_path/'runs'; root.mkdir()
    source=root/'s3'; source.mkdir()
    monkeypatch.setattr(r,'APPROVED_RUNS_ROOT',root)
    args=SimpleNamespace(runs_root=str(root),catalog_db=str(tmp_path/'catalog.sqlite'),resume=str(source))
    meta={'paths':{'s1':str(root/'s1'),'s2':str(root/'s2'),'s3':str(source)}}
    with pytest.raises(RuntimeRefused,match='source'):
        r._output_paths(args,meta)
    assert not list(source.iterdir())


def test_projection_does_not_scale_checkpoint_io_by_rows_or_double_count_components():
    from c_ept.workflows import run_s4a as r
    fitted={'size':'S','device':'cpu','recovered':False,'resumed':False,'epochs_run':2,
            'epoch_seconds':[10.,10.],'whole_call_seconds':28.,'proxy_work_items':10,'full_work_items':40,
            'step_seconds':[1000.,1000.],
            'timing_components':{'epochs':{'forward_seconds':[999.,999.]},'checkpoint_io_current_call':[{'seconds':2.},{'seconds':2.}]}}
    p=r._projections([fitted])['projections'][0]
    # S7 low horizon10:24*(10s*4rows*10epochs +2s IO*10epochs +4s residual).
    assert p['scenarios']['low']['S7_seconds']==24*(400+20+4)
    sensitivity=p['scenarios']['low']['unallocated_residual_sensitivity']['S7_seconds']
    assert sensitivity==[24*(400+20+4),24*(400+20+20)]


@pytest.mark.parametrize('case', ['nested', 'nested_with_flat_decoys', 'prices_hash', 'rules_hash',
                                  'label_prices_hash', 'label_rules_hash', 'missing_prices', 'missing_rules',
                                  'flat_decoy_prices', 'flat_decoy_rules'])
def test_input_preflight_builds_scales_expectations_from_requested_records(tmp_path,case):
    from types import SimpleNamespace
    from c_ept.workflows import run_s4a as r, run_s3
    from c_ept import runlog
    root=tmp_path
    def put(path,doc):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(doc),encoding='utf-8')
    for name in ('s1','s2'):
        put(root/name/name/'STORE.json',{'complete':True,'files':{name+'.npy':'a'*64},'arrays':{}})
    records={name:{'run':name,'files_sha256':runlog.sha256_json({name+'.npy':'a'*64})} for name in ('s1','s2')}
    s3=root/'s3'
    prices=root/'s3a/prices/PRICES.json'; rules=root/'s3a/rules/RULE_PAYOUTS.json'
    put(prices,{'toy':1}); put(rules,{'toy':2})
    put(s3/'RUN.json',{'stage':'s3',**records,'s3a':{'run':'s3a','prices_sha256':r._sha(prices),'rules_sha256':r._sha(rules)}})
    put(s3/'labels/store/STORE.json',{'complete':True,'files':{'labels/a.npy':'b'*64},'arrays':{}})
    put(s3/'labels/store/LABELS.json',{'roles':['FIT','STOP','DEV'],'s1_files_sha256':records['s1']['files_sha256'],'prices_sha256':r._sha(prices),'rules_sha256':r._sha(rules)})
    outputs={}
    for folder,stem in [('b0','b0_heads'),('tier','tower_tier')]:
        arrays=s3/f'fit/{folder}/{stem}.npz'; arrays.parent.mkdir(parents=True,exist_ok=True); arrays.write_bytes(b'synthetic arrays')
        metadata=s3/f'fit/{folder}/{stem}.json'; put(metadata,{'npz_sha256':r._sha(arrays)})
        outputs[f'fit/{folder}/{stem}.json']=r._sha(metadata)
    terrain=root/'s2/terrain'; terrain.mkdir()
    for name in ('TERRAIN.json','terrain.npz','geodesic.npy'):
        (terrain/name).write_bytes(b'synthetic static')
    ledger=run_s3.Ledger(s3)
    ledger.begin('labels',{},'a'); ledger.finish('labels',{'labels/store/':runlog.sha256_json({'labels/a.npy':'b'*64})})
    ledger.begin('fit',{},'b'); ledger.finish('fit',outputs)
    args=SimpleNamespace(s1=str(root/'s1'),s2=str(root/'s2'),s3=str(s3),runs_root=str(root))
    # Exercise the actual preflight against the producer's nested on-disk contract.
    # Root decoys must neither override nested files nor rescue absent nested files.
    if case == 'nested_with_flat_decoys':
        put(root/'s3a/PRICES.json',{'decoy':'not the recorded bytes'})
        put(root/'s3a/RULE_PAYOUTS.json',{'decoy':'not the recorded bytes'})
    elif case in ('prices_hash','rules_hash'):
        put(prices if case == 'prices_hash' else rules,{'tampered':True})
        with pytest.raises(RuntimeRefused,match='price identity' if case == 'prices_hash' else 'rules identity'):
            r.inspect_inputs(args)
        return
    elif case in ('label_prices_hash','label_rules_hash'):
        label_path=s3/'labels/store/LABELS.json'
        changed=json.loads(label_path.read_text())
        changed['prices_sha256' if case == 'label_prices_hash' else 'rules_sha256']='0'*64
        put(label_path,changed)
        with pytest.raises(RuntimeRefused,match='price identity' if case == 'label_prices_hash' else 'rules identity'):
            r.inspect_inputs(args)
        return
    elif case.startswith(('missing_','flat_decoy_')):
        nested=prices if case.endswith('prices') else rules
        if case.startswith('flat_decoy_'):
            (root/'s3a'/nested.name).write_bytes(nested.read_bytes())
        nested.unlink()
        with pytest.raises((FileNotFoundError,RuntimeRefused)):
            r.inspect_inputs(args)
        return
    verified=r.inspect_inputs(args)
    assert Path(verified['paths']['prices']) == prices
    assert Path(verified['paths']['rules']) == rules
    assert Path(verified['paths']['s3a']) == (root/'s3a').resolve()
    assert verified['prices_sha256'] == r._sha(prices)
    assert verified['rules_sha256'] == r._sha(rules)
    expected=verified['expected_provenance']
    assert set(expected)=={'s3_run','s1','s2','labels_store_sha256','model_files_sha256'}
    assert set(expected['model_files_sha256'])=={'fit/b0/b0_heads.json','fit/b0/b0_heads.npz'}
    # Expectations were made without opening a B0_SCALES file or raw .npy payload.
    assert not (s3/'B0_SCALES.json').exists()
    wrong=root/'other/s1'; put(wrong/'STORE.json',{'complete':True,'files':{'s1.npy':'a'*64}})
    args.s1=str(wrong)
    with pytest.raises(RuntimeRefused,match='requested store'):
        r.inspect_inputs(args)
    args.s1=str(root/'s1')
    changed=json.loads((s3/'labels/store/LABELS.json').read_text()); changed['roles'].append('CAL')
    put(s3/'labels/store/LABELS.json',changed)
    with pytest.raises(RuntimeRefused,match='label role'):
        r.inspect_inputs(args)


def test_store_preflight_refuses_incomplete_manifest_without_payload_read(tmp_path):
    from c_ept.workflows import run_s4a as r
    (tmp_path/'STORE.json').write_text(json.dumps({'complete':False,'files':{}}))
    with pytest.raises(RuntimeRefused,match='complete'):
        r._store_metadata(tmp_path)


def test_checkpoint_projection_component_stays_invariant_when_row_ratio_changes():
    from c_ept.workflows import run_s4a as r
    for ratio in (1.,4.):
        with_io=r.project_cost(10.,4.,ratio,10,24,checkpoint_io_seconds=2.)
        without_io=r.project_cost(10.,4.,ratio,10,24)
        assert with_io-without_io==24*2*10


@pytest.mark.parametrize('target', ['terrain_catalog','s1_resume','s2_resume','s3_resume','s3a_resume','price_catalog','rules_catalog','separate_terrain_catalog','separate_labels_catalog'])
def test_readonly_producer_and_separate_input_trees_refuse_write_targets(tmp_path,monkeypatch,target):
    from types import SimpleNamespace
    from c_ept.workflows import run_s4a as r
    root=tmp_path/'runs'; root.mkdir()
    paths={'s1':root/'source1/s1','s2':root/'source2/s2','s3':root/'source3',
           's1_producer':root/'source1','s2_producer':root/'source2','s3a':root/'source-price',
           'terrain':root/'source2/terrain','prices':root/'source-price/PRICES.json',
           'rules':tmp_path/'separate-rules/RULE_PAYOUTS.json','labels':tmp_path/'separate-labels/store'}
    monkeypatch.setattr(r,'APPROVED_RUNS_ROOT',root)
    args=SimpleNamespace(runs_root=str(root),catalog_db=str(tmp_path/'catalog.sqlite'),resume=None)
    meta={'paths':{key:str(value) for key,value in paths.items()}}
    # A common runs parent is allowed; its input producer descendants are protected.
    r._output_paths(args,meta)
    if target.endswith('_resume'):
        args.resume=str(paths[{'s1_resume':'s1_producer','s2_resume':'s2_producer','s3_resume':'s3','s3a_resume':'s3a'}[target]])
    else:
        parent={'terrain_catalog':paths['terrain'],'price_catalog':paths['prices'].parent,
                'rules_catalog':paths['rules'].parent,'separate_labels_catalog':paths['labels'],
                'separate_terrain_catalog':tmp_path/'separate-terrain'}[target]
        if target=='separate_terrain_catalog': meta['paths']['terrain']=str(parent)
        args.catalog_db=str(parent/'catalog.sqlite')
    with pytest.raises(RuntimeRefused,match='source'):
        r._output_paths(args,meta)
    assert list(root.iterdir())==[]  # No reservation, lock or catalog side effect.


@pytest.mark.parametrize('failure_type',[RuntimeRefused,ValueError,OSError])
def test_main_post_process_failure_records_actual_declared_return(tmp_path,monkeypatch,failure_type):
    from c_ept.workflows import run_s4a as r, run_s1
    root=tmp_path/'runs'; root.mkdir()
    meta={'paths':{'s1':str(root/'source1/s1'),'s2':str(root/'source2/s2'),'s3':str(root/'source3')}}
    monkeypatch.setattr(r,'APPROVED_RUNS_ROOT',root)
    monkeypatch.setattr(r,'repository_state',lambda:dict(code_commit='a'*40,code_dirty=False,repo_root=str(tmp_path)))
    monkeypatch.setattr(r,'inspect_inputs',lambda args:meta)
    monkeypatch.setattr(run_s1,'require_no_strata',lambda:probe())
    def fail_loading(meta): raise failure_type('synthetic post-PROCESS failure')
    monkeypatch.setattr(r,'_load_inputs',fail_loading)
    returned=r.main(['--s1','toy1','--s2','toy2','--s3','toy3','--runs-root',str(root),
                     '--catalog-db',str(tmp_path/'catalog.sqlite'),'--sizes','S','--devices','cpu'])
    processes=list(root.glob('*/PROCESS.json'))
    assert len(processes)==1
    record=json.loads(processes[0].read_text())
    assert returned==2 and record['state']=='failed'
    assert record['exit_code']==returned


@pytest.mark.parametrize('incoming',[None,'false'])
def test_fresh_sklearn_import_cannot_enable_override_after_bootstrap(incoming):
    code='''import json,os,sys
from c_ept.s4a_runtime import prepare_native_runtime
assert 'sklearn' not in sys.modules
first=prepare_native_runtime()
assert os.environ['KMP_DUPLICATE_LIB_OK']=='FALSE'
from sklearn.linear_model import LogisticRegression
assert os.environ['KMP_DUPLICATE_LIB_OK']=='FALSE'
assert prepare_native_runtime()==first
print(json.dumps({'policy':first['kmp_duplicate_lib_ok'],'sklearn_imported':True,'rebootstrap':'accepted'}))
'''
    env=dict(os.environ,PYTHONIOENCODING='utf-8')
    if incoming is None:env.pop('KMP_DUPLICATE_LIB_OK',None)
    else:env['KMP_DUPLICATE_LIB_OK']=incoming
    result=subprocess.run([sys.executable,'-B','-c',code],env=env,capture_output=True,text=True,encoding='utf-8',timeout=45)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['rebootstrap']=='accepted'


@pytest.mark.parametrize('incoming',['TRUE','unexpected'])
def test_fresh_bootstrap_refuses_unsafe_caller_before_import_or_rewrite(incoming):
    code='''import os,sys
from c_ept.s4a_runtime import RuntimeRefused,prepare_native_runtime
original=os.environ['KMP_DUPLICATE_LIB_OK']
try:prepare_native_runtime()
except RuntimeRefused:
 assert os.environ['KMP_DUPLICATE_LIB_OK']==original
 assert not any(name in sys.modules for name in ('sklearn','torch','numpy'))
 print('REFUSED_WITHOUT_IMPORT_OR_REWRITE')
else:raise AssertionError('unsafe incoming override was accepted')
'''
    result=subprocess.run([sys.executable,'-B','-c',code],env=dict(os.environ,KMP_DUPLICATE_LIB_OK=incoming,PYTHONIOENCODING='utf-8'),capture_output=True,text=True,encoding='utf-8',timeout=15)
    assert result.returncode==0,result.stderr
    assert result.stdout.strip()=='REFUSED_WITHOUT_IMPORT_OR_REWRITE'


def test_safe_runtime_test_fixture_restores_sklearn_collection_environment():
    code='''import os
from sklearn.linear_model import LogisticRegression
assert os.environ['KMP_DUPLICATE_LIB_OK']=='True'
import pytest
result=pytest.main(['c_ept/tests/test_run_s4a.py::test_execute_e6_refuses_before_input_or_run_mutation','-q','-p','no:cacheprovider'])
assert result==0
assert os.environ['KMP_DUPLICATE_LIB_OK']=='True', 'fixture changed incoming environment permanently'
from c_ept.s4a_runtime import RuntimeRefused,prepare_native_runtime
try:prepare_native_runtime()
except RuntimeRefused:print('COLLECTION_ENV_RESTORED_AND_STILL_REFUSED')
else:raise AssertionError('restored unsafe environment silently accepted')
'''
    env=dict(os.environ,PYTHONIOENCODING='utf-8',MKL_THREADING_LAYER='SEQUENTIAL')
    env.pop('KMP_DUPLICATE_LIB_OK',None)
    result=subprocess.run([sys.executable,'-B','-c',code],cwd=Path(__file__).resolve().parents[2],env=env,capture_output=True,text=True,encoding='utf-8',timeout=60)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'COLLECTION_ENV_RESTORED_AND_STILL_REFUSED' in result.stdout
