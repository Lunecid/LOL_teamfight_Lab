"""Owned-worker scientific call wiring; imported only after explicit Start gate.

This module is never executed by implementation checks against real stores.
Every output is private to the new campaign; no old run or input is mutated.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

from . import protocol as P


def atomic_json(path, value):
    from c_ept.evaluation.runtime import atomic_json as write
    write(path,value)


def _json_default(value):
    if hasattr(value,'item'):
        return value.item()
    raise TypeError('non-scalar metadata value')


def stage_path(c,stage,model=None):
    P.require(stage in P.STAGES,'unknown stage')
    if stage in ('fit','evaluate'):
        P.require(model in P.MODELS,'unknown model')
        return Path(c['output'])/stage/model
    P.require(model is None,'model not allowed for shared stage')
    return Path(c['output'])/stage


def _receipt(c,stage,model=None,*,check=None):
    path=stage_path(c,stage,model)/'stage.json'
    record=P.read_json(path)
    P.require(record['state']=='completed' and record['config_sha256']==P.digest(c),'stage binding incomplete')
    P.require(record['stage']==stage and record['model']==model,'stage identity mismatch')
    for rel,h in record['artifacts'].items():
        P.require(P.sha_file(P.inside(c['output'],rel),check=check)==h,'stage artifact changed')
    return record


def _write_receipt(c,stage,model,details,artifacts):
    root=Path(c['output']).resolve()
    hashes={str(Path(p).resolve().relative_to(root)):P.sha_file(p) for p in artifacts}
    record={'schema':'ept-comparison-stage-v1','state':'completed','stage':stage,'model':model,
            'config_sha256':P.digest(c),'details':details,'artifacts':hashes}
    atomic_json(stage_path(c,stage,model)/'stage.json',record)
    return record


def inspect_lineage(c):
    """Derive scale/target provenance independently from the signed S3 chain."""
    from c_ept.workflows.run_s4a import inspect_inputs
    args=SimpleNamespace(s3=c['inputs']['s3']['root'],s1=c['inputs']['s1']['root'],
                         s2=c['inputs']['s2']['root'],runs_root=str(Path(c['output']).parent))
    meta=inspect_inputs(args)
    for name in ('s1','s2','labels','terrain','s3'):
        P.require(Path(meta['paths'][name]).resolve()==Path(c['inputs'][name]['root']).resolve(),
                  'S3 lineage input mismatch: '+name)
    P.require(Path(meta['paths']['prices']).resolve()==Path(c['inputs']['prices']['root']).resolve()/'PRICES.json',
              'S3 price lineage mismatch')
    P.require(meta['expected_provenance']==c['expected_scale_provenance'],'scale provenance not independently bound')
    return meta


def load_bundle(meta,check):
    from c_ept.evaluation.runtime import _bundle
    return _bundle(meta,check=check)


def verify_bundle(bundle):
    for name in ('s1','s2','labels'):
        getattr(bundle,name).verify_used()


def _common(c,check):
    from .data import Membership
    from c_ept.ept.data import Stats
    from c_ept.ept.scales import load_scales
    receipt=_receipt(c,'prepare')
    folder=stage_path(c,'prepare')
    meta=P.read_json(folder/'inputs.json')
    member=Membership.from_record(P.read_json(folder/'membership.json'))
    stats=Stats.load(folder/'STATS.json')
    scales=load_scales(P.inside(c['inputs']['s3']['root'],'B0_SCALES.json'),
                       expected_provenance=c['expected_scale_provenance'])
    P.require(stats.sha256==receipt['details']['stats_sha256'],'shared Stats differs')
    bundle=load_bundle(meta,check)
    member.verify_against(bundle.s1)
    return bundle,member,stats,scales,receipt


def prepare(c,check):
    from .data import select_membership,fit_stats_subset,build_neural_data
    from c_ept.ept import train
    from c_ept.ept.scales import load_scales
    folder=stage_path(c,'prepare')
    P.require(not folder.exists(),'preparation already attempted')
    folder.mkdir(parents=True)
    check()
    meta=inspect_lineage(c)
    bundle=load_bundle(meta,check)
    member=select_membership(bundle.s1,share=P.FIXED['share'],selection_seed=P.FIXED['selection_seed'])
    member.verify_against(bundle.s1)
    stats=fit_stats_subset(bundle.s1,bundle.s2,member,check=check)
    scales=load_scales(P.inside(c['inputs']['s3']['root'],'B0_SCALES.json'),
                       expected_provenance=c['expected_scale_provenance'])
    training,_,_=build_neural_data(bundle,member,scales=scales,stats=stats,
                       batch_size=c['settings']['batch_size'],train_seed=c['settings']['neural_seed'],check=check)
    row_identities={'fit':train._rows_digest(training.fit),'stop':train._rows_digest(training.stop)}
    stats.save(folder/'STATS.json')
    atomic_json(folder/'membership.json',member.to_record())
    atomic_json(folder/'inputs.json',meta)
    verify_bundle(bundle);check()
    details={'stats_sha256':stats.sha256,'membership_sha256':member.sha256,
             'neural_rows':row_identities,
             'fit_matches':len(member.fit),'stop_matches':len(member.stop),
             'predictor_stats_population':'selected FIT only',
             'auxiliary_population':'shared original whole-FIT S3 scales, terrain and target-definition assets'}
    return _write_receipt(c,'prepare',None,details,[folder/'STATS.json',folder/'membership.json',folder/'inputs.json'])


def fit(c,model,check):
    folder=stage_path(c,'fit',model)
    P.require(not folder.exists(),'model fit already attempted; no retry/resume')
    folder.mkdir(parents=True)
    bundle,member,stats,scales,parent=_common(c,check)
    settings=c['settings']
    check()
    if model=='b1':
        from .data import fit_b1_subset
        fitted=fit_b1_subset(bundle.s1,bundle.s2,bundle.labels,member,check=check)
        P.require(getattr(fitted,'membership_sha256',None)==member.sha256,'fresh B1 subset identity missing')
        model_dir=folder/'model'
        fitted.save(model_dir)
        # Bind every model file emitted by the canonical serializer. No input
        # directories or checkpoint payloads are enumerated here.
        artifacts=sorted(p for p in model_dir.rglob('*') if p.is_file())
        P.require(artifacts and not any(p.is_symlink() for p in artifacts),'invalid B1 model artifacts')
        info={'architecture':'lightgbm_b1_original_p9','seed':1,'native_threads':8,
              'subset_fitting':fitted.comparison_receipt}
    else:
        from .data import build_neural_data
        from .models import specification,configure_float32
        from c_ept.ept import encoder,train,checkpoints
        configure_float32()
        spec=specification(model,hidden=settings['n2_hidden'],prefetch=settings['prefetch'])
        training,supplied_stats,data_receipt=build_neural_data(bundle,member,scales=scales,stats=stats,
                                   batch_size=settings['batch_size'],train_seed=settings['neural_seed'],check=check)
        P.require(supplied_stats.sha256==stats.sha256,'neural worker changed shared Stats')
        P.require(data_receipt['membership_sha256']==member.sha256,'neural worker changed membership')
        if settings['prefetch']:
            from .infrastructure import OrderedPrefetch
            training=dataclasses.replace(training,fit=OrderedPrefetch(training.fit,max_prefetch=settings['prefetch']))
        cfg=train.TrainConfig(encoder.SIZES['S'],settings['lr'],settings['weight_decay'],
                              settings['warmup_steps'],settings['patience'],model_spec=spec)
        model_dir=folder/'model'
        try:
            result=train.train(cfg,training,model_dir,device='cuda',epochs=settings['max_epochs'],check=check,seed=settings['neural_seed'])
        finally:
            if hasattr(training.fit,'close'):
                training.fit.close()
        latest=checkpoints.latest_metadata(model_dir)
        P.require(latest is not None and latest[1]['done'] is True,'nonterminal neural model')
        epoch_dir,head,_=latest
        P.require(head['identity']['model_spec']==spec,'neural architecture identity differs')
        validate_training_head(c,head,parent,model)
        info={'architecture':spec,'seed':settings['neural_seed'],'epochs_run':result.epochs_run,
              'best_epoch':result.best_epoch,'stopped_early':result.stopped_early,
              'head':str((epoch_dir/'head.json').relative_to(folder)),
              'head_sha256':P.sha_file(epoch_dir/'head.json'),'best':head['best'],
              'model_parameters':sum(p.numel() for p in result.model.parameters())}
        artifacts=[epoch_dir/'head.json',epoch_dir/'epoch_metrics.json',epoch_dir/'epoch_metrics.csv']
    verify_bundle(bundle);check()
    info.update(membership_sha256=member.sha256,stats_sha256=stats.sha256,
                scale_sha256=c['inputs']['s3']['files']['B0_SCALES.json'])
    atomic_json(folder/'result.json',info)
    artifacts.append(folder/'result.json')
    return _write_receipt(c,'fit',model,info,artifacts)


def validate_training_head(c,head,preparation,model):
    from .models import specification
    from c_ept.ept import encoder
    identity=head['identity'];s=c['settings']
    expected={'epochs':s['max_epochs'],'seed':s['neural_seed'],'lr':s['lr'],
              'weight_decay':s['weight_decay'],'warmup_steps':s['warmup_steps'],
              'patience':s['patience'],'dtype':'torch.float32','threads':s['native_threads'],
              'deterministic_algorithms':True,'device':'cuda'}
    P.require(all(identity.get(k)==v and type(identity[k]) is type(v) for k,v in expected.items()),
              'neural training settings differ from matched profile')
    P.require(identity['encoder']==dataclasses.asdict(encoder.SIZES['S']),'encoder size differs from fixed S profile')
    P.require(identity['model_spec']==specification(model,hidden=s['n2_hidden'],prefetch=s['prefetch']),
              'neural architecture differs')
    for role in ('fit','stop'):
        P.require(identity[role]==preparation['details']['neural_rows'][role],
                  'neural training rows/Stats differ from common preparation')


def load_neural_best(c,model,static,device):
    """Fresh architecture-bound best state only; never optimizer/last/RNG."""
    P.require(c['execution_authorized'] is True,'best loading requires explicit execution authorization')
    import torch
    from .models import build_model,specification,configure_float32
    from c_ept.ept import encoder
    fit_receipt=_receipt(c,'fit',model)
    info=fit_receipt['details']
    spec=specification(model,hidden=c['settings']['n2_hidden'],prefetch=c['settings']['prefetch'])
    P.require(info['architecture']==spec,'model registry identity mismatch')
    folder=stage_path(c,'fit',model)
    head_path=P.inside(folder,info['head'])
    P.require(P.sha_file(head_path)==info['head_sha256'],'final neural head changed')
    head=P.read_json(head_path)
    validate_training_head(c,head,_receipt(c,'prepare'),model)
    P.require(head['done'] is True and head['best_epoch']==info['best_epoch'] and head['best']==info['best'],
              'best pointer differs from final head')
    P.require(head['identity']['model_spec']==spec,'head architecture differs')
    from c_ept.evaluation.guard import unpassed_evaluation_settings
    unpassed=unpassed_evaluation_settings(head['identity'])      # the plain model and the default run_inference would evaluate such a checkpoint as another design
    P.require(not unpassed,f'the checkpoint was trained with settings that this evaluation does not pass on: {unpassed}')
    P.require(head['identity']['static_sha256']==hashlib.sha256(static.tobytes()).hexdigest(),'static input differs')
    best=P.inside(folder/'model',head['best']['folder'])/'best.pt'
    P.require(best.is_file() and not best.is_symlink() and best.stat().st_size==head['best']['bytes'], 'invalid best-only file')
    P.require(P.sha_file(best)==head['best']['sha256'],'best-only payload changed')
    configure_float32()
    fitted=build_model(encoder.EncoderConfig(**head['identity']['encoder']),static,spec)
    fitted.load_state_dict(torch.load(best,weights_only=True,map_location='cpu'),strict=True)
    return fitted.to(device).eval()


def save_predictions(result,folder):
    """Private derived arrays; metadata contains file identities, not row lists."""
    import numpy as np
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    meta={'schema':'ept-comparison-predictions-v1','provenance':result['provenance']}
    files=[]
    for kind in ('endpoints','nll'):
        records=[]
        for i,record in enumerate(result[kind]):
            arrays={};fields={}
            for key,value in record.items():
                if isinstance(value,np.ndarray):
                    name=f'{kind}_{i:03d}_{key}.npy'
                    P.require(key.isidentifier(),'unsafe prediction field')
                    p=folder/name;np.save(p,value,allow_pickle=False);files.append(p)
                    arrays[key]={'file':name,'sha256':P.sha_file(p),'bytes':p.stat().st_size}
                else:
                    fields[key]=value
            records.append({'fields':fields,'arrays':arrays})
        meta[kind]=records
    meta=json.loads(json.dumps(meta,default=_json_default,allow_nan=False))
    atomic_json(folder/'PREDICTIONS.json',meta);files.append(folder/'PREDICTIONS.json')
    return files


def load_predictions(folder):
    import numpy as np
    folder=Path(folder);meta=P.read_json(folder/'PREDICTIONS.json')
    P.require(meta['schema']=='ept-comparison-predictions-v1','prediction schema mismatch')
    result={'provenance':meta['provenance']}
    for kind in ('endpoints','nll'):
        records=[]
        for item in meta[kind]:
            record=dict(item['fields'])
            for key,entry in item['arrays'].items():
                p=P.inside(folder,entry['file'])
                P.require(p.stat().st_size==entry['bytes'] and P.sha_file(p)==entry['sha256'],'prediction array changed')
                record[key]=np.load(p,mmap_mode='r',allow_pickle=False)
            records.append(record)
        result[kind]=records
    return result


def evaluate(c,model,check):
    if model=='b1':
        # B1 predictions are measured on the exact rows of both neural pairs.
        # This cheap stage only verifies its reference artifact; it emits no
        # invented standalone metric or inference claim.
        folder=stage_path(c,'evaluate','b1')
        P.require(not folder.exists(),'B1 reference verification already attempted')
        folder.mkdir(parents=True)
        _receipt(c,'fit','b1');check()
        atomic_json(folder/'reference.json',{'role':'common paired reference','metrics':'measured in ept/n2 paired stages'})
        return _write_receipt(c,'evaluate','b1',{'scope':'reference artifact verification only'},[folder/'reference.json'])
    P.require(model in ('ept','n2'),'unknown neural evaluation model')
    from c_ept.models.baselines_b1 import B1Heads
    from c_ept.evaluation.adapters import run_inference
    folder=stage_path(c,'evaluate',model)
    P.require(not folder.exists(),'model evaluation already attempted')
    folder.mkdir(parents=True)
    bundle,member,stats,scales,parent=_common(c,check)
    _receipt(c,'fit','b1')
    b1=B1Heads.load(stage_path(c,'fit','b1')/'model')
    fitted=load_neural_best(c,model,bundle.static,'cuda')
    check()
    # Matched design: this model and B1 were fitted on the same selected 25% membership. The neural model read the shared Stats;
    # B1 did not (it fitted its own dropped columns on the selected FIT). The provenance text says exactly that; the T003 default
    # ("existing whole-FIT B1; unequal scope") describes another design.
    result=run_inference(bundle,fitted,stats,b1,check=check,device='cuda',
                         batch_size=c['settings']['batch_size'],draws=c['settings']['mc_draws'],
                         seed=c['settings']['mc_seed'],scales=scales,
                         comparison=P.matched_text({'ept':'EPT','n2':'N2'}[model],'B1')['comparison'])
    result['provenance'].update(model=model,membership_sha256=member.sha256,stats_sha256=stats.sha256)
    artifacts=save_predictions(result,folder/'predictions')
    verify_bundle(bundle);check()
    return _write_receipt(c,'evaluate',model,{'model':model,'membership_sha256':member.sha256},artifacts)


def finalize(c,check):
    from .evaluation import summarize_pairs
    from .reporting import write_report
    folder=stage_path(c,'finalize')
    P.require(not folder.exists(),'finalization already attempted')
    for model in ('ept','n2'):
        _receipt(c,'evaluate',model)
    a=load_predictions(stage_path(c,'evaluate','ept')/'predictions')
    b=load_predictions(stage_path(c,'evaluate','n2')/'predictions')
    summaries=summarize_pairs(a,b,bootstrap=c['settings']['bootstrap'],
                              seed=c['settings']['bootstrap_seed'],check=check)
    check()
    write_report(summaries,folder,provenance={'config_sha256':P.digest(c),
                 'source_commit':c['source_commit'],'settings':c['settings'],
                 'limits':c['limits'],'auxiliary':'shared original whole-FIT fixed assets',
                 'training_seeds':{'ept':20261022,'n2':20261022,'b1':1}})
    for model in ('ept','n2'):
        _receipt(c,'evaluate',model)
    check()
    return _write_receipt(c,'finalize',None,{'interpretation':'diagnostic_only'},list(folder.iterdir()))


def execute_worker(c,stage,model,check):
    """Production dispatch; caller must hold the owned-worker Start grant."""
    P.require(c['execution_authorized'] is True,'actual comparison execution is not authorized')
    if stage=='prepare':return prepare(c,check)
    if stage=='fit':return fit(c,model,check)
    if stage=='evaluate':return evaluate(c,model,check)
    if stage=='finalize':return finalize(c,check)
    raise P.ComparisonRefused('unknown worker stage')
