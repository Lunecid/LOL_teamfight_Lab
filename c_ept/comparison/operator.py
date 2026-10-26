"""Inert-by-default CLI for the separately authorized matched25 campaign."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

from . import protocol as P


def child_environment(original=None):
    env=dict(os.environ if original is None else original)
    env.update(PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='8',
               MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',NUMEXPR_NUM_THREADS='8',
               VECLIB_MAXIMUM_THREADS='8',MKL_THREADING_LAYER='SEQUENTIAL',
               CUBLAS_WORKSPACE_CONFIG=':4096:8',CUDA_VISIBLE_DEVICES='0')
    return env


class IntegrityGuard:
    """Bounded stat polling plus mandatory full boundary hashes; no lock claim."""
    def __init__(self,c,config_path,*,clock=time.monotonic,interval=2.):
        self.c,self.config_path,self.clock,self.interval=c,Path(config_path),clock,interval
        self.config_digest=P.digest(c)
        self.last=-float('inf')
        manifest=P.read_json(c['source_manifest'])
        paths=[Path(c['source_manifest']),self.config_path]
        paths.extend(P.inside(c['source'],p) for p in manifest['files'])
        paths.extend(P.inside(entry['root'],p) for entry in c['inputs'].values() for p in entry['files'])
        self.signatures={p:self.signature(p) for p in paths}

    @staticmethod
    def signature(path):
        P.require(path.is_file() and not path.is_symlink(),'bound file missing or linked')
        s=path.stat()
        return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns

    def __call__(self):
        now=self.clock()
        if now-self.last>=self.interval:
            P.require(P.digest(P.load_config(self.config_path))==self.config_digest,'campaign config changed')
            P.require(self.c['execution_authorized'] is True,'future explicit run authorization required')
            for p,sig in self.signatures.items():
                P.require(self.signature(p)==sig,'bound source/input file changed: '+str(p))
            self.last=now

    def boundary(self,check=None):
        if check is not None: check()
        self.last=-float('inf');self()
        P.verify_source(self.c,check=check)
        for name,entry in self.c['inputs'].items():
            for rel,h in entry['files'].items():
                P.require(P.sha_file(P.inside(entry['root'],rel),check=check)==h,'input identity changed: '+name+'/'+rel)
        if check is not None: check()


def worker_command(config,stage,model=None):
    command=[str(P.PYTHON),'-B','-m','c_ept.comparison.operator','--config',str(Path(config).resolve()),
             '--action','Worker','--stage',stage,'--await-job-binding']
    if model is not None:
        command+=['--model',model]
    return command


def start(c,config_path):
    """Future explicit Start only; never called during implementation acceptance."""
    started_at,started_wall=time.monotonic(),time.time()
    P.require(c['execution_authorized'] is True,'execution_authorized=false; implementation is not a run grant')
    def startup_check():
        elapsed=max(time.monotonic()-started_at,time.time()-started_wall)
        P.require(0<=elapsed<P.LIMITS['prepare_seconds'],'preflight exhausted preparation budget')
    checked=P.preflight(c,check=startup_check)
    from .infrastructure import CampaignLedger,CampaignBudget,ResourcePlan
    from tools.ept_comparison_controller import run_owned
    from .runtime import atomic_json,_receipt
    output=Path(c['output'])
    output.mkdir(parents=True,exist_ok=False)
    ledger=CampaignLedger(output/'CAMPAIGN.json',models=P.MODELS,
                          started_at=started_at,started_wall=started_wall)
    budget=CampaignBudget(ledger)
    state='failed'
    phase,model='prepare',None
    try:
        guard=IntegrityGuard(c,config_path)
        atomic_json(output/'RUN.json',{'schema':'ept-comparison-run-v1','config':c,
                     'config_sha256':checked['config_sha256'],'resources':ResourcePlan().describe(),
                     'state':'running','stage':'matched25','source_commit':c['source_commit']})
        for phase,model in [('prepare',None),*(('fit',m) for m in P.MODELS),
                            *(('evaluate',m) for m in P.MODELS),('finalize',None)]:
            def active_check():
                budget.check();guard()
            def before():
                guard.boundary(check=active_check)
            def accept():
                _receipt(c,phase,model,check=active_check)
                guard.boundary(check=active_check)
            logdir=output/'logs';logdir.mkdir(exist_ok=True)
            logpath=logdir/(phase+('_'+model if model else '')+'.log')
            with logpath.open('wb') as stream:
                receipt=run_owned(worker_command(config_path,phase,model),cwd=c['source'],
                                  budget=budget,phase=phase,model=model,check=guard,
                                  env=child_environment(),stdout=stream,stderr=stream,
                                  before=before,accept=accept)
            P.require(receipt['state']=='completed','owned worker failed; see phase receipt/log')
        state='completed'
        atomic_json(output/'FINAL.json',{'schema':'ept-comparison-final-v1','state':state,
                    'config_sha256':P.digest(c),'source_commit':c['source_commit'],
                    'interpretation':'diagnostic_only','actual_claim':'no single-seed confirmation',
                    'campaign_ledger_file':'CAMPAIGN.json',
                    'campaign_ledger_snapshot_scope':'before FINAL write; CAMPAIGN includes terminal marker time',
                    'campaign_ledger':ledger.snapshot()})
        budget.seal_terminal()
    except BaseException as exc:
        state='failed'
        budget.abort(f'{type(exc).__name__}: {exc}',phase=phase,model=model)
        atomic_json(output/'FINAL.json',{'schema':'ept-comparison-final-v1','state':state,
                    'config_sha256':P.digest(c),'source_commit':c['source_commit'],
                    'error':f'{type(exc).__name__}: {exc}','exception_type':type(exc).__name__,
                    'campaign_ledger':ledger.snapshot()})
        raise
    return {'state':state,'output':str(output)}


def worker(c,config_path,stage,model,*,binding=False,stdin=None):
    # Crucially no torch, data store, model or runtime import precedes G.
    P.require(binding,'workers require owned Job binding')
    P.require((stdin or sys.stdin).readline().strip()=='G','owned Job release grant missing')
    P.require(c['execution_authorized'] is True,'scientific execution not authorized')
    # The parent ledger is read-only from the worker and must name this phase.
    from .infrastructure import check_worker_budget
    guard=IntegrityGuard(c,config_path)
    def check():
        guard()
        check_worker_budget(Path(c['output'])/'CAMPAIGN.json',phase=stage,model=model)
    check()
    guard.boundary(check=check)
    import torch
    torch.set_num_threads(c['settings']['native_threads'])
    from .runtime import execute_worker
    result=execute_worker(c,stage,model,check)
    check();guard.boundary(check=check)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description='EPT/B1/N2 matched25 comparison; default is metadata-only Check.')
    parser.add_argument('--config',required=True)
    parser.add_argument('--action',choices=('Check','Start','Status','Worker'),default='Check')
    parser.add_argument('--stage',choices=P.STAGES)
    parser.add_argument('--model',choices=P.MODELS)
    parser.add_argument('--await-job-binding',action='store_true')
    args=parser.parse_args(argv)
    c=None
    try:
        c=P.load_config(args.config)
        if args.action=='Worker':
            P.require(args.stage is not None,'worker stage required')
            result=worker(c,args.config,args.stage,args.model,binding=args.await_job_binding)
        elif args.action=='Start':
            result=start(c,args.config)
        elif args.action=='Status':
            result=P.read_json(Path(c['output'])/'CAMPAIGN.json')
        else:
            result=P.preflight(c)
        print(json.dumps(result,ensure_ascii=False,allow_nan=False))
        return 0
    except (ValueError,RuntimeError,OSError,KeyError) as err:
        # ComparisonRefused and the S4a RuntimeRefused are guard refusals: the prepare stage reuses run_s4a.inspect_inputs, whose refusals (for
        # example the terrain check) are the same one-line guards as in S4a and T002. Any other error keeps its class name and a traceback (A5-F8).
        # A worker has loaded match data, so its text is scrubbed with the keys of the S1 store; Check, Start and Status read metadata only.
        from c_ept.workflows.run_s4a import failure_report
        from c_ept.workflows.s4a_runtime import RuntimeRefused
        s1=c['inputs']['s1']['root'] if c is not None and args.action=='Worker' else None
        print(failure_report('Comparison refused',err,(P.ComparisonRefused,RuntimeRefused),s1=s1),file=sys.stderr)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
