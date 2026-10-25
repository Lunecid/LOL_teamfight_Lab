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


@pytest.mark.parametrize('value',[True,float('nan'),float('inf')])
def test_pilot_cannot_be_promoted_from_invalid_scores(value):
    with pytest.raises(P.ComparisonRefused):P.pilot_decision(value,1.)


def test_bad_or_equal_pilot_does_not_claim_equivalence_or_success():
    d=P.pilot_decision(2.,1.)
    assert d['status']=='diagnose_before_expansion' and d['confirmed_success'] is False
    assert P.pilot_decision(1.,1.)['confirmed_success'] is False


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
