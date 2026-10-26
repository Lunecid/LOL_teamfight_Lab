"""Synthetic wiring/metadata tests; no scientific fits, GPU or actual stores."""
import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from c_ept.comparison import protocol as P
from c_ept.comparison import models,operator,runtime
from c_ept.comparison.evaluation import neural_pair,summarize_pairs
from c_ept.comparison.reporting import write_report
from c_ept.evaluation.adapters import STRUCTURAL_HEADS


ROOT=Path(__file__).resolve().parents[1]


def example():
    return P.read_json(ROOT/'configs/ept_comparison.example.json')


def prediction(value):
    ids=np.array([0,1],dtype=np.int64)
    endpoint={'name':'M2/60','module':'M2','layer':'whole','horizon_s':60,
              'target':'occurrence','match':ids,'row':ids,'y':np.array([0,1]),
              'p_ept':np.array([.2,.8]),'p_b1':np.array([.4,.6]),'counts':{}}
    nll=[{'name':name,'match':ids,'row':ids,'nll_ept':np.full(2,value),
          'nll_b1':np.full(2,2.),'scale':1.,'nll_ept_normalized':np.full(2,value),
          'nll_b1_normalized':np.full(2,2.),'common_head':True,'counts':{}}
         for name in sorted(STRUCTURAL_HEADS)]
    return {'endpoints':[endpoint],'nll':nll,'provenance':{'synthetic':True}}


def test_example_is_unbound_and_cannot_launch(monkeypatch):
    c=example()
    assert P.validate_config(c) is c and c['execution_authorized'] is False
    monkeypatch.setattr(P,'preflight',lambda *a,**kw:pytest.fail('Start called preflight before authorization refusal'))
    with pytest.raises(P.ComparisonRefused,match='execution_authorized=false'):
        operator.start(c,'invented-config.json')


@pytest.mark.parametrize('key,value',[('share',.5),('amp',True),('tf32',True),
    ('dtype','bfloat16'),('selection_seed',4),('b1_seed',20261022),('prefetch',4),
    ('batch_size',64),('max_epochs',31)])
def test_profile_rejects_silent_scientific_or_resource_changes(key,value):
    c=example();c['settings'][key]=value
    with pytest.raises(P.ComparisonRefused):P.validate_config(c)


def test_output_cannot_overlap_inputs_before_metadata_access(monkeypatch,tmp_path):
    c=example();c['source_commit']='a'*40;c['source_manifest_sha256']='b'*64
    c['output']=str(tmp_path/'20261025_c_ept_matched25_1')
    c['inputs']['s1']['root']=str(tmp_path)
    monkeypatch.setattr(P,'verify_source',lambda *a,**k:None)
    with pytest.raises(P.ComparisonRefused,match='overlap'):
        P.preflight(c,runs_root=tmp_path)


def test_duplicate_json_and_nonfinite_refused(tmp_path):
    p=tmp_path/'x.json';p.write_text('{"a":1,"a":2}')
    with pytest.raises(P.ComparisonRefused,match='duplicate'):P.read_json(p)
    p.write_text('{"a":NaN}')
    with pytest.raises(P.ComparisonRefused,match='nonfinite'):P.read_json(p)


def test_worker_never_dispatches_without_job_grant(monkeypatch):
    monkeypatch.setattr(runtime,'execute_worker',lambda *a:pytest.fail('ungated scientific dispatch'))
    with pytest.raises(P.ComparisonRefused,match='binding'):
        operator.worker(example(),'invented','prepare',None,binding=False,stdin=io.StringIO('G\n'))
    with pytest.raises(P.ComparisonRefused,match='grant'):
        operator.worker(example(),'invented','prepare',None,binding=True,stdin=io.StringIO(''))


def test_authorization_false_dispatch_refused_before_any_import_or_work(monkeypatch):
    monkeypatch.setattr(runtime,'prepare',lambda *a:pytest.fail('unauthorized preparation'))
    with pytest.raises(P.ComparisonRefused,match='not authorized'):
        runtime.execute_worker(example(),'prepare',None,lambda:None)


@pytest.mark.parametrize('stage,model',[('prepare',None),('fit','ept'),('fit','n2'),
    ('fit','b1'),('evaluate','ept'),('evaluate','n2'),('evaluate','b1'),('finalize',None)])
def test_worker_dispatch_wires_all_registered_stages(monkeypatch,stage,model):
    c=example();c['execution_authorized']=True;calls=[]
    def selected(*args):calls.append(args);return 'reached'
    monkeypatch.setattr(runtime,stage,selected)
    assert runtime.execute_worker(c,stage,model,lambda:None)=='reached'
    assert calls and calls[0][0] is c


def test_three_pair_structural_scores_and_model_names_are_hand_checkable(tmp_path):
    a,b=prediction(1.),prediction(1.5)
    pairs=summarize_pairs(a,b,bootstrap=5,seed=7,check=lambda:None)
    first=pairs['ept_vs_b1']['structural_normalized_nll']['metrics']
    third=pairs['ept_vs_n2']['structural_normalized_nll']['metrics']
    assert first['ept']['estimate']==1. and first['b1']['estimate']==2.
    assert third['delta']['estimate']==-.5
    assert pairs['n2_vs_b1']['model_labels']=={'ept':'N2','b1':'B1'}
    r=write_report(pairs,tmp_path/'report',provenance={'synthetic':True})
    assert r['decisions']['ept_vs_n2']['confirmed_success'] is False
    assert r['decisions']['ept_vs_b1']['difference']==-1.
    csv=(tmp_path/'report/comparison.csv').read_text(encoding='utf-8-sig')
    assert 'N2' in csv and 'EPT' in csv and 'B1' in csv
    assert ET.parse(tmp_path/'report/comparison.svg').getroot().tag.endswith('svg')


@pytest.mark.parametrize('mutation',('row','truth','reference','scale','missing'))
def test_three_model_comparison_rejects_misalignment_instead_of_joining(mutation):
    a,b=prediction(1.),prediction(1.5)
    if mutation=='row':b['endpoints'][0]['row']=np.array([0,2])
    if mutation=='truth':b['endpoints'][0]['y']=np.array([1,1])
    if mutation=='reference':b['nll'][0]['nll_b1']=np.full(2,9.)
    if mutation=='scale':b['nll'][0]['scale']=2.
    if mutation=='missing':b['nll'].pop()
    with pytest.raises(ValueError):neural_pair(a,b)


def test_private_prediction_roundtrip_and_tamper_refusal(tmp_path):
    a=prediction(1.)
    runtime.save_predictions(a,tmp_path/'private')
    b=runtime.load_predictions(tmp_path/'private')
    assert np.array_equal(a['nll'][0]['row'],b['nll'][0]['row'])
    # Windows refuses truncation while the read-only mappings remain live.
    del b
    import gc
    gc.collect()
    p=next((tmp_path/'private').glob('*.npy'))
    p.write_bytes(p.read_bytes()+b'corrupt')
    with pytest.raises(P.ComparisonRefused,match='changed'):runtime.load_predictions(tmp_path/'private')


@pytest.mark.parametrize('protected',['C_EPT/CoHoRt.py','RESEARCH_PIPELINE/cohort_policy.py',
                                    'research_pipeline/HOLDOUT_R3/x.py'])
def test_source_manifest_never_opens_protected_case_variants(tmp_path,protected):
    c=example(); c['source']=str(tmp_path);c['source_commit']='a'*40
    manifest=tmp_path/'manifest.json'
    files={protected:'a'*64,**{p:'b'*64 for p in P.SOURCE_REQUIRED}}
    manifest.write_text(json.dumps({'files':files}),encoding='utf-8')
    c['source_manifest']=str(manifest);c['source_manifest_sha256']=P.sha_file(manifest)
    with pytest.raises(P.ComparisonRefused,match='protected'):
        P.verify_source(c,source_probe=lambda *args:None)


def test_model_spec_is_explicit_and_legacy_config_identity_stays_optional():
    from c_ept.ept import encoder,train
    original=train.TrainConfig(encoder.SIZES['S'],.001,.01,10)
    assert original.model_spec is None
    spec=models.specification('n2')
    chosen=train.TrainConfig(encoder.SIZES['S'],.001,.01,10,model_spec=spec)
    assert chosen.model_spec['architecture']=='n2_teamset_v1'
    spec['architecture']='invalid'
    assert chosen.model_spec['architecture']=='n2_teamset_v1'
    with pytest.raises(ValueError):models.validate_model_spec(spec)


@pytest.mark.parametrize('value',[True,float('nan'),float('inf'),None])
def test_pilot_cannot_be_promoted_from_invalid_scores(value):
    with pytest.raises(P.ComparisonRefused):P.pilot_decision(value,[-2.,-1.],'available')


def decide(difference,ci95,ci_status='available',label='B1'):
    return P.pilot_decision(difference,ci95,ci_status,comparator_label=label)


# Rule (T008 "A CI including 0 is inconclusive, never equivalence"; EPT_MATCHED25_PROTOCOL_KO.md):
#   upper < 0 -> provisional_improvement; lower > 0 -> diagnose_before_expansion; otherwise inconclusive.
@pytest.mark.parametrize('difference,ci95,expected',[
    (-.5,[-.8,-.2],'provisional_improvement'),
    (-.5,[-.8,-1e-12],'provisional_improvement'),
    (.5,[.2,.8],'diagnose_before_expansion'),
    (.5,[1e-12,.8],'diagnose_before_expansion'),
    # the point estimate is negative but the interval reaches 0 or above: a noise-level "improvement"
    (-.5,[-.8,.1],'inconclusive'),
    (-.5,[-.8,0.],'inconclusive'),
    (.5,[0.,.8],'inconclusive'),
    (.5,[-.1,.8],'inconclusive'),
    (0.,[-.3,.3],'inconclusive')])
def test_pilot_decision_follows_the_paired_interval_not_the_point_sign(difference,ci95,expected):
    d=decide(difference,ci95)
    assert d['status']==expected
    assert d['difference']==difference and d['ci95']==ci95 and d['ci_status']=='available'


@pytest.mark.parametrize('ci95,ci_status',[
    (None,'available'),(None,'not_requested'),(None,'unavailable_invalid_replicates'),
    ([-.8,-.2],'unavailable_invalid_replicates'),([-.8,-.2],'not_requested'),([-.8,-.2],None),
    ([float('nan'),-.2],'available'),([-.8,float('inf')],'available'),([float('-inf'),-.2],'available'),
    ([-.8],'available'),([-.8,-.5,-.2],'available'),('ab','available'),([True,False],'available'),
    ([-.2,-.8],'available')])
def test_missing_nonfinite_or_invalid_interval_is_inconclusive_never_an_improvement(ci95,ci_status):
    d=decide(-.5,ci95,ci_status)
    assert d['status']=='inconclusive' and d['ci95'] is None
    assert d['difference']==-.5 and d['confirmed_success'] is False


def test_pilot_decision_keeps_the_diagnostic_labels_and_names_the_real_comparator():
    d=decide(-.5,[-.8,-.2],label='N2')
    assert d['inference']=='diagnostic_only' and d['confirmed_success'] is False
    assert d['withdrawal']=='no superiority claim from this single-seed pilot'
    assert d['comparator']=='N2' and d['contrast']=='EPT-minus-N2'
    for ci95 in ([-.8,-.2],[.2,.8],[-.1,.1],None):
        assert decide(-.5,ci95)['confirmed_success'] is False


def spread(ept_values,*,b1=2.):
    """A synthetic prediction whose structural NLL differs by match, so bootstrap intervals are not degenerate."""
    values=np.asarray(ept_values,dtype=float)
    n=len(values)
    ids=np.arange(n,dtype=np.int64)
    endpoint={'name':'M2/60','module':'M2','layer':'whole','horizon_s':60,'target':'occurrence',
              'match':ids,'row':ids,'y':ids%2,'p_ept':np.full(n,.4)+.1*(ids%2),'p_b1':np.full(n,.5),'counts':{}}
    nll=[{'name':name,'match':ids,'row':ids,'nll_ept':values,'nll_b1':np.full(n,b1),'scale':1.,
          'nll_ept_normalized':values,'nll_b1_normalized':np.full(n,b1),'common_head':True,'counts':{}}
         for name in sorted(STRUCTURAL_HEADS)]
    return {'endpoints':[endpoint],'nll':nll,'provenance':{'synthetic':True}}


def table_rows(path):
    rows={}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        cells=[c.strip() for c in line.strip('|').split('|')]
        if cells and cells[0].startswith('ept_vs_'):rows[cells[0]]=cells
    return rows


def report_of(a,b,tmp_path,*,bootstrap=400):
    pairs=summarize_pairs(a,b,bootstrap=bootstrap,seed=7,check=lambda:None)
    record=write_report(pairs,tmp_path/'report',provenance={'synthetic':True})
    return pairs,record,table_rows(tmp_path/'report/REPORT_KO.md')


@pytest.mark.parametrize('ept,n2,b1_status,n2_status',[
    ([.95,1.,1.05,1.,.97,1.03,1.,1.],[1.5,1.45,1.55,1.5,1.52,1.48,1.5,1.5],'provisional_improvement','provisional_improvement'),
    ([0.,4.,0.,4.,0.,4.,0.,4.],[0.,4.,0.,4.,0.,4.,0.,4.],'inconclusive','inconclusive'),
    ([3.,3.1,2.9,3.,3.05,2.95,3.,3.],[1.,1.,1.,1.,1.,1.,1.,1.],'diagnose_before_expansion','diagnose_before_expansion')])
def test_report_decisions_use_the_bootstrap_interval_of_the_structural_difference(tmp_path,ept,n2,b1_status,n2_status):
    pairs,record,rows=report_of(spread(ept),spread(n2),tmp_path)
    assert record['decisions']['ept_vs_b1']['status']==b1_status
    assert record['decisions']['ept_vs_n2']['status']==n2_status
    for key in ('ept_vs_b1','ept_vs_n2'):
        delta=pairs[key]['structural_normalized_nll']['metrics']['delta']
        d=record['decisions'][key]
        # the decision reads exactly the interval the pipeline computed; no new statistic
        assert d['difference']==delta['estimate'] and d['ci95']==delta['ci95'] and d['ci_status']=='available'
        lo,hi=delta['ci95']
        assert d['status']==('provisional_improvement' if hi<0 else 'diagnose_before_expansion' if lo>0 else 'inconclusive')
        assert d['confirmed_success'] is False and d['inference']=='diagnostic_only'
        # every decision row prints the difference, the interval and the status
        cells=rows[key]
        assert f"{delta['estimate']:.6g}" in cells[2] and f'{lo:.6g}' in cells[3] and f'{hi:.6g}' in cells[3]
        assert cells[-1]==d['status']


def test_inconclusive_when_the_point_difference_is_negative_but_the_interval_reaches_zero(tmp_path):
    # EPT is better on average (point difference -0.6) but the match-level spread puts 0 inside the interval.
    pairs,record,rows=report_of(spread([0.,3.,0.,3.,0.,3.,0.,3.,1.,1.]),spread([1.]*10),tmp_path)
    delta=pairs['ept_vs_b1']['structural_normalized_nll']['metrics']['delta']
    assert delta['estimate']<0 and delta['ci95'][1]>0
    assert record['decisions']['ept_vs_b1']['status']=='inconclusive'


def test_decision_names_the_n2_comparator_and_reads_the_comparator_slot_of_each_pair(tmp_path):
    # EPT 1.0, N2 0.5, B1 2.0: EPT beats B1 by -1.0 but loses to N2 by +0.5, so a mix-up of slots is visible.
    pairs,record,rows=report_of(spread([1.]*6),spread([.5]*6),tmp_path,bootstrap=50)
    b1,n2=record['decisions']['ept_vs_b1'],record['decisions']['ept_vs_n2']
    assert (b1['comparator'],b1['difference'],b1['status'])==('B1',-1.,'provisional_improvement')
    assert (n2['comparator'],n2['difference'],n2['status'])==('N2',.5,'diagnose_before_expansion')
    assert n2['contrast']=='EPT-minus-N2'
    assert rows['ept_vs_b1'][1]=='B1' and rows['ept_vs_n2'][1]=='N2'


def test_report_prints_unavailable_when_the_interval_was_not_computed(tmp_path):
    pairs,record,rows=report_of(spread([.5]*6),spread([1.]*6),tmp_path,bootstrap=0)
    for key in ('ept_vs_b1','ept_vs_n2'):
        d=record['decisions'][key]
        assert d['status']=='inconclusive' and d['ci95'] is None and d['ci_status']=='not_requested'
        assert rows[key][3]=='unavailable' and rows[key][-1]=='inconclusive'
        assert rows[key][2]!='unavailable'       # the point difference itself is still printed


def test_report_without_common_heads_is_unavailable_everywhere(tmp_path):
    a,b=spread([1.]*4),spread([1.]*4)
    for pred in (a,b):
        for row in pred['nll']:row['common_head']=False
    pairs,record,rows=report_of(a,b,tmp_path,bootstrap=10)
    for key in ('ept_vs_b1','ept_vs_n2'):
        assert record['decisions'][key]['status']=='unavailable'
        assert rows[key][2]=='unavailable' and rows[key][3]=='unavailable'


# --- matched-25% provenance text: the T003 "whole-FIT B1 / unequal scope" wording must not leak in -----------------
STALE_T003=('whole-FIT','unequal training scope','FIT/STOP-25%-EPT')
PAIR_MODELS={'ept_vs_b1':('EPT','B1'),'n2_vs_b1':('N2','B1'),'ept_vs_n2':('EPT','N2')}


def test_matched_pairs_name_the_matched_design_and_the_real_comparator(tmp_path):
    pairs,record,_=report_of(spread([1.,1.1,.9,1.]),spread([1.5,1.4,1.6,1.5]),tmp_path,bootstrap=20)
    saved=json.loads((tmp_path/'report/comparison.json').read_text(encoding='utf-8'))
    for text in (json.dumps(pairs,ensure_ascii=False),json.dumps(saved['pairs'],ensure_ascii=False)):
        assert not any(stale in text for stale in STALE_T003)
    for key,(model,comparator) in PAIR_MODELS.items():
        s=saved['pairs'][key]
        assert s['bootstrap']['delta']==f'{model}-{comparator}'
        assert f'{model} versus {comparator}' in s['comparison']
        # what is shared is the selected membership; the predictor preprocessing is fitted on the selected FIT per model family: EPT and N2 read one
        # Stats object, B1 does not use it and fits its own dropped columns (EPT-T011 group 1 review M1: "same predictor Stats for every model" was wrong)
        text = s['comparison']
        assert 'same selected 25% FIT/STOP membership for every model' in text
        assert 'one Stats shared by EPT and N2' in text and 'dropped columns of B1' in text
        assert not any(wrong in text for wrong in ('predictor Stats for every model', 'predictor Stats for all models', 'and predictor Stats'))
        assert s['limitations'] and not any(stale in line for line in s['limitations'] for stale in STALE_T003)


def test_matched_limitations_keep_the_t003_caveats_and_add_the_shared_asset_condition():
    from c_ept.evaluation.metrics import LIMITATIONS
    pairs=summarize_pairs(prediction(1.),prediction(1.5),bootstrap=3,seed=7,check=lambda:None)
    for s in pairs.values():
        assert s['limitations'][:len(LIMITATIONS)]==list(LIMITATIONS)
        assert len(s['limitations'])==len(LIMITATIONS)+1 and 'shared fixed' in s['limitations'][-1]
        last=s['limitations'][-1]                          # says what is shared exactly: B1 does not use the shared Stats
        assert 'one Stats shared by EPT and N2' in last and 'dropped columns of B1' in last and 'predictor Stats for all models' not in last


@pytest.mark.parametrize('model,label',[('ept','EPT'),('n2','N2')])
def test_matched_evaluation_stage_passes_matched_text_into_predictions_metadata(monkeypatch,tmp_path,model,label):
    import c_ept.evaluation.adapters as adapters
    from c_ept.models import baselines_b1
    c=example();c['output']=str(tmp_path/'out');(tmp_path/'out').mkdir()
    member,stats=SimpleNamespace(sha256='m'*64),SimpleNamespace(sha256='s'*64)
    monkeypatch.setattr(runtime,'_common',lambda c,check:(SimpleNamespace(static=None),member,stats,{},{}))
    monkeypatch.setattr(runtime,'_receipt',lambda *a,**k:{})
    monkeypatch.setattr(runtime,'load_neural_best',lambda *a,**k:'neural model')
    monkeypatch.setattr(runtime,'verify_bundle',lambda bundle:None)
    monkeypatch.setattr(baselines_b1.B1Heads,'load',staticmethod(lambda directory:'b1 heads'))
    seen={}
    def fake_inference(bundle,fitted,stats_,b1,**kw):
        seen.update(kw)
        out=prediction(1.)
        out['provenance']={'adapter':'synthetic','comparison':kw.get('comparison')}  # what the real adapter records
        return out
    monkeypatch.setattr(adapters,'run_inference',fake_inference)
    runtime.evaluate(c,model,lambda:None)
    text=(tmp_path/'out/evaluate'/model/'predictions/PREDICTIONS.json').read_text(encoding='utf-8')
    provenance=json.loads(text)['provenance']
    assert provenance['comparison']==seen['comparison'] and provenance['model']==model
    assert f'{label} versus B1' in provenance['comparison']
    assert not any(stale in text for stale in STALE_T003)


def test_matched_bundle_comes_from_the_one_production_builder_that_sets_the_dev_role(monkeypatch):
    from c_ept.evaluation import runtime as evaluation_runtime
    sentinel=SimpleNamespace(role='DEV')
    seen=[]
    monkeypatch.setattr(evaluation_runtime,'_bundle',lambda meta,*,check:seen.append((meta,check)) or sentinel)
    check=lambda:None
    assert runtime.load_bundle({'paths':{}},check) is sentinel
    assert seen==[({'paths':{}},check)]


def test_run_three_gives_each_pair_the_matched_text(monkeypatch):
    import c_ept.evaluation.adapters as adapters
    from c_ept.comparison.evaluation import run_three
    calls={}
    def fake_inference(bundle,model,stats,b1,**kw):
        calls[model]=kw.get('comparison')
        return prediction(1. if model=='ept' else 1.5)
    monkeypatch.setattr(adapters,'run_inference',fake_inference)
    pairs=run_three(None,'ept','n2',None,None,check=lambda:None,device='cpu',bootstrap=3)
    assert 'EPT versus B1' in calls['ept'] and 'N2 versus B1' in calls['n2']
    assert not any(stale in text for text in calls.values() for stale in STALE_T003)
    assert pairs['ept_vs_n2']['bootstrap']['delta']=='EPT-N2'


def test_architecture_bound_loader_roundtrips_only_synthetic_best_weights(tmp_path):
    import dataclasses
    import hashlib
    import torch
    from c_ept.ept.encoder import SIZES
    from c_ept.models.neural.team_set import TeamSetModel
    from c_ept.tests.ept_toys import toy_static
    c=example();c['output']=str(tmp_path);c['execution_authorized']=True
    spec=models.specification('n2',hidden=c['settings']['n2_hidden'])
    cfg=SIZES['S']
    static=toy_static()
    original=models.build_model(cfg,static,spec)
    folder=tmp_path/'fit/n2'; epoch=folder/'model/epoch_0001';epoch.mkdir(parents=True)
    best=epoch/'best.pt';torch.save(original.state_dict(),best)
    pointer={'folder':'epoch_0001','sha256':P.sha_file(best),'bytes':best.stat().st_size}
    row_identity={'role':'synthetic','stats_sha256':'c'*64}
    (tmp_path/'prepare').mkdir()
    runtime._write_receipt(c,'prepare',None,{'neural_rows':{'fit':row_identity,'stop':row_identity}},[])
    head={'done':True,'best_epoch':0,'best':pointer,'identity':{
        'model_spec':spec,'encoder':dataclasses.asdict(cfg),
        'fit':row_identity,'stop':row_identity,'epochs':30,'seed':20261022,
        'lr':.001,'weight_decay':.01,'warmup_steps':10,'patience':2,
        'dtype':'torch.float32','threads':8,'deterministic_algorithms':True,'device':'cuda',
        'static_sha256':hashlib.sha256(static.tobytes()).hexdigest()}}
    (epoch/'head.json').write_text(json.dumps(head),encoding='utf-8')
    info={'architecture':spec,'head':'model/epoch_0001/head.json',
          'head_sha256':P.sha_file(epoch/'head.json'),'best_epoch':0,'best':pointer}
    runtime._write_receipt(c,'fit','n2',info,[epoch/'head.json'])
    loaded=runtime.load_neural_best(c,'n2',static,'cpu')
    assert isinstance(loaded,TeamSetModel)
    assert all(torch.equal(value,loaded.state_dict()[key]) for key,value in original.state_dict().items())
    best.write_bytes(best.read_bytes()+b'bad')
    with pytest.raises(P.ComparisonRefused,match='best-only'):
        runtime.load_neural_best(c,'n2',static,'cpu')


# ---- EPT-T011 A5-F8: unexpected exceptions keep their class name and a key-scrubbed traceback; exit code 2 stays ------------------
def write_config(tmp_path,c):
    path=tmp_path/'comparison.json'
    path.write_text(json.dumps(c),encoding='utf-8')
    return str(path)


def test_main_names_an_unexpected_error_and_leaves_a_guard_refusal_alone(monkeypatch,tmp_path,capsys):
    path=write_config(tmp_path,example())
    def unexpected(*a,**kw):
        raise RuntimeError('synthetic CUDA out of memory')
    monkeypatch.setattr(P,'preflight',unexpected)
    assert operator.main(['--config',path])==2                                              # the exit code is unchanged
    err=capsys.readouterr().err
    assert err.startswith('Comparison refused:') and 'RuntimeError' in err and 'Traceback' in err
    def refusal(*a,**kw):
        raise P.ComparisonRefused('synthetic guard refusal')
    monkeypatch.setattr(P,'preflight',refusal)
    assert operator.main(['--config',path])==2
    assert capsys.readouterr().err=='Comparison refused: synthetic guard refusal\n'


def test_worker_error_is_scrubbed_with_the_match_keys_of_its_s1_store(monkeypatch,tmp_path,capsys):
    from c_ept.tests.test_failure_diagnostics import KEY,toy_s1
    c=example()
    c['inputs']['s1']['root']=str(toy_s1(tmp_path/'s1'))
    def broken(*a,**kw):
        raise KeyError(f'x/{KEY}')
    monkeypatch.setattr(operator,'worker',broken)
    assert operator.main(['--config',write_config(tmp_path,c),'--action','Worker','--stage','prepare','--await-job-binding'])==2
    err=capsys.readouterr().err
    assert 'KeyError' in err and 'Traceback' in err and 'TOYKEY' not in err and 'x/<key>' in err


def test_worker_runtime_refused_is_a_guard_refusal_of_one_line(monkeypatch,tmp_path,capsys):
    # The prepare worker calls run_s4a.inspect_inputs, whose RuntimeRefused (for example the terrain check) is the same one-line guard
    # refusal as in S4a and T002; it must not be labelled "not a guard refusal" or get a traceback. The exit code is unchanged.
    from c_ept.tests.test_failure_diagnostics import toy_s1
    from c_ept.workflows.s4a_runtime import RuntimeRefused
    c=example()
    c['inputs']['s1']['root']=str(toy_s1(tmp_path/'s1'))
    def refused(*a,**kw):
        raise RuntimeRefused('synthetic S4a input refusal')
    monkeypatch.setattr(operator,'worker',refused)
    assert operator.main(['--config',write_config(tmp_path,c),'--action','Worker','--stage','prepare','--await-job-binding'])==2
    assert capsys.readouterr().err=='Comparison refused: synthetic S4a input refusal\n'


def test_failed_start_records_the_exception_type_in_final_json(monkeypatch,tmp_path):
    c=example()
    c['execution_authorized']=True
    c['output']=str(tmp_path/'20261025_c_ept_matched25_1')
    monkeypatch.setattr(P,'preflight',lambda *a,**kw:{'config_sha256':P.digest(c)})
    def broken(*a,**kw):
        raise RuntimeError('synthetic guard construction failure')
    monkeypatch.setattr(operator,'IntegrityGuard',broken)
    with pytest.raises(RuntimeError,match='synthetic guard construction failure'):
        operator.start(c,write_config(tmp_path,c))
    final=json.loads((Path(c['output'])/'FINAL.json').read_text(encoding='utf-8'))
    assert final['state']=='failed' and final['exception_type']=='RuntimeError'
    assert final['error'].startswith('RuntimeError: ')                                       # the existing field is kept as it was


@pytest.mark.parametrize('setting',[{'zone_mark':False},{'kill_law':'shifted'},{'k_condition':'binned'},{'eval_average':False}])
def test_neural_loader_refuses_a_checkpoint_whose_identity_has_a_setting_the_evaluation_does_not_pass_on(tmp_path,setting):
    """S4b preparation M9 review I1: ``run_inference`` is called with its defaults and the model is built plain, so such a checkpoint would be evaluated as the other design with no error;
    the loader refuses it by name before it builds anything. The default identity loads (the round trip test above)."""
    import dataclasses
    import hashlib
    import torch
    from c_ept.ept.encoder import SIZES
    from c_ept.tests.ept_toys import toy_static
    c=example();c['output']=str(tmp_path);c['execution_authorized']=True
    spec=models.specification('ept')
    cfg=SIZES['S'];static=toy_static()
    original=models.build_model(cfg,static,spec)
    folder=tmp_path/'fit/ept'; epoch=folder/'model/epoch_0001';epoch.mkdir(parents=True)
    best=epoch/'best.pt';torch.save(original.state_dict(),best)
    pointer={'folder':'epoch_0001','sha256':P.sha_file(best),'bytes':best.stat().st_size}
    row_identity={'role':'synthetic','stats_sha256':'c'*64}
    (tmp_path/'prepare').mkdir()
    runtime._write_receipt(c,'prepare',None,{'neural_rows':{'fit':row_identity,'stop':row_identity}},[])
    head={'done':True,'best_epoch':0,'best':pointer,'identity':{
        'model_spec':spec,'encoder':dataclasses.asdict(cfg),'fit':row_identity,'stop':row_identity,'epochs':30,'seed':20261022,
        'lr':.001,'weight_decay':.01,'warmup_steps':10,'patience':2,'dtype':'torch.float32','threads':8,'deterministic_algorithms':True,'device':'cuda',
        'static_sha256':hashlib.sha256(static.tobytes()).hexdigest(),**setting}}
    (epoch/'head.json').write_text(json.dumps(head),encoding='utf-8')
    info={'architecture':spec,'head':'model/epoch_0001/head.json','head_sha256':P.sha_file(epoch/'head.json'),'best_epoch':0,'best':pointer}
    runtime._write_receipt(c,'fit','ept',info,[epoch/'head.json'])
    with pytest.raises(P.ComparisonRefused,match=next(iter(setting))):
        runtime.load_neural_best(c,'ept',static,'cpu')


def test_the_comparison_source_closure_pins_the_guard_that_its_loader_calls():
    """M9 re-check r2: ``load_neural_best`` refuses unpassed settings through ``c_ept/evaluation/guard.py``, so that file is part of the pinned source closure."""
    assert 'c_ept/evaluation/guard.py' in P.SOURCE_REQUIRED
