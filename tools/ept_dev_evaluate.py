"""EPT-T003 metadata Check (default) and explicit one-shot supervised Start.

No Start is performed by importing this module, the training monitor, or Check.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from c_ept.evaluation import guard
from c_ept.evaluation.runtime import PHASE_LIMITS, TOTAL_SECONDS, MEMORY_BYTES, atomic_json

SAMPLE_INTERVAL = .25
MAX_SAMPLE_GAP = 1.0
CLEANUP_SECONDS = 10.0


def child_env():
    env = dict(os.environ)
    for key in ('PYTHONPATH','PYTHONHOME'):
        env.pop(key, None)
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',PYTHONUNBUFFERED='1',
               OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',
               MKL_THREADING_LAYER='SEQUENTIAL',KMP_DUPLICATE_LIB_OK='FALSE',
               CUBLAS_WORKSPACE_CONFIG=':4096:8',CUDA_VISIBLE_DEVICES='0')
    return env


def worker(config_path):
    # Before G this child imports standard-library modules ONLY.
    guard.require(sys.stdin.buffer.read(1) == b'G', 'owned Job gate was not released')
    c = guard.load_config(config_path)
    out = Path(c['output'])
    ctl = guard.read_json(out/'controller.json')
    import psutil
    guard.require(ctl['child_pid'] == os.getpid() and ctl['config_sha256'] == guard.digest(c)
                  and ctl['assigned'] is True, 'worker launch binding mismatch')
    guard.require(ctl['controller_pid'] == os.getppid(), 'worker is not the direct supervised child')
    parent = psutil.Process(ctl['controller_pid'])
    guard.require(abs(parent.create_time()-ctl['controller_create_time']) < .00001, 'supervisor generation changed')
    guard.require(abs(psutil.Process(os.getpid()).create_time()-ctl['child_create_time']) < .00001,
                  'worker generation mismatch')
    try:
        started = time.monotonic()
        atomic_json(out/'phase.json', {'phase':'preflight', 'phase_started_monotonic':ctl['started_monotonic'],
                                      'pid':os.getpid(), 'config_sha256':guard.digest(c)})
        checked = guard.preflight(c, allow_reserved=True)
        finished = time.monotonic()
        guard.require(finished-ctl['started_monotonic'] < PHASE_LIMITS['preflight'], 'preflight budget exceeded')
        guard.require(checked['config_sha256'] == guard.digest(c), 'checked configuration mismatch')
        atomic_json(out/'CHECK.json', checked)
        receipt = {'state':'passed', 'config_sha256':guard.digest(c),
                   'check_sha256':guard.sha_file(out/'CHECK.json'),
                   'metadata_hashes_sha256':guard.digest(checked['metadata_hashes']),
                   **{k:ctl[k] for k in ('controller_pid','controller_create_time','child_pid','child_create_time')},
                   'started_monotonic':started, 'finished_monotonic':finished}
        atomic_json(out/'PREFLIGHT_RECEIPT.json', receipt)
        receipt_sha = guard.sha_file(out/'PREFLIGHT_RECEIPT.json')
        manifest = guard.read_json(out/'RUN_MANIFEST.json')
        guard.require(manifest['config_sha256'] == guard.digest(c) and manifest['config'] == c,
                      'launch manifest binding mismatch')
        manifest.update(check_sha256=receipt['check_sha256'], preflight_receipt_sha256=receipt_sha,
                        metadata_hashes_sha256=receipt['metadata_hashes_sha256'])
        atomic_json(out/'RUN_MANIFEST.json', manifest)
        _checked_receipt(out, c, ctl)
        from c_ept.evaluation.runtime import execute
        execute(c, checked)
        guard.require(guard.sha_file(out/'PREFLIGHT_RECEIPT.json') == receipt_sha, 'preflight receipt changed during evaluation')
        _checked_receipt(out, c, ctl)
        result = guard.read_json(out/'RESULT.json')
        guard.require(result['state'] == 'completed' and result['config_sha256'] == guard.digest(c)
                      and result['best'] == checked['best'], 'unbound evaluation result')
        result.update(preflight_receipt_sha256=receipt_sha, check_sha256=receipt['check_sha256'],
                      metadata_hashes_sha256=receipt['metadata_hashes_sha256'])
        atomic_json(out/'RESULT.json', result)
    except BaseException as exc:
        atomic_json(out/'WORKER_FAILURE.json', {'state':'failed','error':f'{type(exc).__name__}: {exc}',
                                       'config_sha256':guard.digest(c),'retry':False})
        raise


def _checked_receipt(out, c, ctl):
    receipt = guard.read_json(out/'PREFLIGHT_RECEIPT.json')
    guard.require(receipt['state'] == 'passed' and receipt['config_sha256'] == guard.digest(c), 'preflight receipt binding mismatch')
    for key in ('controller_pid','controller_create_time','child_pid','child_create_time'):
        guard.require(receipt[key] == ctl[key], 'preflight receipt generation mismatch')
    start, end = receipt['started_monotonic'], receipt['finished_monotonic']
    guard.require(all(type(v) in (int,float) and math.isfinite(v) for v in (start,end))
                  and ctl['started_monotonic'] <= start <= end <= time.monotonic()
                  and end-ctl['started_monotonic'] < PHASE_LIMITS['preflight'], 'preflight receipt timing mismatch')
    guard.require(guard.sha_file(out/'CHECK.json') == receipt['check_sha256'], 'checked metadata changed')
    checked = guard.read_json(out/'CHECK.json')
    guard.require(checked['config_sha256'] == guard.digest(c)
                  and guard.digest(checked['metadata_hashes']) == receipt['metadata_hashes_sha256'], 'checked metadata binding mismatch')
    return receipt, checked


def _sample_child(child, job, record, psutil):
    """Independent private-byte sample of the launch-owned child, never the host.

    Fresh generation checks bracket the memory call; Job membership is checked
    through the Job's own handle. Only NoSuchProcess plus Popen-confirmed exit
    is benign. No RSS fallback, PID-based kill, or descendant/host memory claim.
    """
    now = time.monotonic()
    previous = record.get('last_memory_sample_monotonic')
    if previous is not None:
        gap = now-previous
        record['max_memory_sample_gap_seconds'] = max(record['max_memory_sample_gap_seconds'], gap)
        guard.require(gap <= MAX_SAMPLE_GAP, 'owned child memory sampling gap exceeded')
    try:
        process = psutil.Process(child.pid)
        guard.require(abs(process.create_time()-record['child_create_time']) < .00001, 'owned child generation changed')
        guard.require(job.contains(child.pid, record['child_create_time']), 'owned child Job membership unavailable')
        private = getattr(process.memory_info(), 'private', None)
        guard.require(abs(psutil.Process(child.pid).create_time()-record['child_create_time']) < .00001,
                      'owned child generation changed during memory sample')
    except psutil.NoSuchProcess:
        guard.require(child.poll() is not None, 'owned child memory unavailable while process live')
        return
    guard.require(type(private) is int and private > 0, 'owned child private memory unavailable')
    record['memory_samples'] += 1
    record['last_memory_sample_monotonic'] = now
    record['peak_sampled_child_private_bytes'] = max(record['peak_sampled_child_private_bytes'], private)
    guard.require(private < MEMORY_BYTES, 'owned child sampled private memory cap exceeded')


def supervise(c, config_path):
    began = time.monotonic()
    # Validate only strict config/path safety before reserving. All scientific
    # metadata preflight runs in the gated Job child, under this same clock.
    guard.require(guard.load_config(config_path) == c, 'configuration changed before reservation')
    guard.require(Path(c['source']).resolve() == SOURCE, 'evaluator imported from wrong source')
    out = Path(c['output'])
    out.mkdir()  # atomic no-overwrite reservation; no retry/resume on any old output
    record={'state':'launching','controller_pid':os.getpid(),'controller_create_time':None,
            'config_sha256':guard.digest(c),'check_sha256':None,'preflight_receipt_sha256':None,'kind':'windows_job',
            'started_monotonic':began,'memory_scope':'launch-owned generation-bound evaluation child only; sampled, not continuous OS cap',
            'memory_cap_bytes':MEMORY_BYTES,'memory_sample_interval_seconds':SAMPLE_INTERVAL,
            'memory_sample_max_gap_seconds':MAX_SAMPLE_GAP,'memory_samples':0,
            'max_memory_sample_gap_seconds':0.,'peak_sampled_child_private_bytes':0,
            'child_pid':None,'assigned':False,'released':False,
            'job_gone_before_close':False,'job_closed':False,'tree_gone':False,'exit_code':None,'error':None}
    job,child=None,None
    phase,phase_start='preflight',began
    try:
        with (out/'launch.once').open('x', encoding='utf-8') as f:
            json.dump({'config_sha256':guard.digest(c)},f)
            f.flush(); os.fsync(f.fileno())
        atomic_json(out/'RUN_MANIFEST.json',{'schema':guard.SCHEMA,'config':c,'config_sha256':guard.digest(c),
                    'training_identity':guard.TRAIN_ID,'attempts':1,'retry_limit':0,'started_monotonic':began,
                    'phase_limits_seconds':PHASE_LIMITS,'total_wall_seconds':TOTAL_SECONDS,
                    'memory_scope':record['memory_scope'],'memory_cap_bytes':MEMORY_BYTES,
                    'memory_sample_max_gap_seconds':MAX_SAMPLE_GAP})
        import psutil
        sys.path.insert(0,str(SOURCE/'tools'))
        from ept_training_job import WindowsJob
        record['controller_create_time']=psutil.Process().create_time()
        job=WindowsJob()
        with (out/'stdout.log').open('xb') as stdout,(out/'stderr.log').open('xb') as stderr:
            guard.require(guard.load_config(config_path)==c,'configuration changed before launch')
            guard.clean_source(c['source'],c['reviewed_commit'])
            guard.require(time.monotonic()-began < PHASE_LIMITS['preflight'], 'prelaunch preflight budget exceeded')
            child=subprocess.Popen([str(guard.PYTHON),'-B',str(SOURCE/'tools/ept_dev_evaluate.py'),
                                    '--config',str(config_path),'--worker'],cwd=SOURCE,env=child_env(),
                                    stdin=subprocess.PIPE,stdout=stdout,stderr=stderr,
                                    creationflags=subprocess.CREATE_NO_WINDOW)
            job.assign(child)
            record.update(child_pid=child.pid,child_create_time=psutil.Process(child.pid).create_time(),assigned=True,state='running')
            _sample_child(child, job, record, psutil)
            atomic_json(out/'controller.json',record)
            job.release(child)
            record['released']=True
            atomic_json(out/'controller.json',record)
            while True:
                now=time.monotonic()
                if child.poll() is None:
                    _sample_child(child, job, record, psutil)
                if (out/'phase.json').exists():
                    p=guard.read_json(out/'phase.json')
                    guard.require(p['pid']==child.pid and p['config_sha256']==guard.digest(c),'phase binding mismatch')
                    if p['phase'] != phase:
                        names=list(PHASE_LIMITS)
                        guard.require(p['phase'] in names and names.index(p['phase'])>names.index(phase),'phase reset/refusal')
                        transition=p['phase_started_monotonic']
                        guard.require(type(transition) in (int,float) and math.isfinite(transition)
                                      and phase_start <= transition <= now, 'phase timestamp mismatch')
                        guard.require(transition-phase_start < PHASE_LIMITS[phase], 'previous phase wall limit exceeded')
                        phase,phase_start=p['phase'],transition
                guard.require(now-began<TOTAL_SECONDS and now-phase_start<PHASE_LIMITS[phase], 'supervised phase/total wall limit exceeded')
                if (out/'PREFLIGHT_RECEIPT.json').exists():
                    receipt_sha=guard.sha_file(out/'PREFLIGHT_RECEIPT.json')
                    if record['preflight_receipt_sha256'] is None:
                        receipt,_=_checked_receipt(out,c,record)
                        record.update(preflight_receipt_sha256=receipt_sha,check_sha256=receipt['check_sha256'])
                    else:
                        guard.require(record['preflight_receipt_sha256']==receipt_sha, 'preflight receipt changed')
                record.update(phase=phase,elapsed_seconds=now-began)
                atomic_json(out/'controller.json',record)
                if child.poll() is not None:
                    break
                time.sleep(max(0., SAMPLE_INTERVAL-(time.monotonic()-now)))
            record['exit_code']=child.returncode
            guard.require(child.returncode==0 and job.gone(), 'evaluation failed or descendant remains alive')
            result=guard.read_json(out/'RESULT.json')
            guard.require(result['state']=='completed' and result['config_sha256']==guard.digest(c), 'missing bound evaluation result')
            receipt,checked=_checked_receipt(out,c,record)
            guard.require(record['preflight_receipt_sha256'] is not None
                          and result['preflight_receipt_sha256']==record['preflight_receipt_sha256']
                          and result['check_sha256']==record['check_sha256']
                          and result['metadata_hashes_sha256']==receipt['metadata_hashes_sha256']
                          and result['best']==checked['best'], 'result/preflight receipt mismatch')
            manifest=guard.read_json(out/'RUN_MANIFEST.json')
            guard.require(manifest['config']==c and manifest['config_sha256']==guard.digest(c)
                          and all(manifest[k]==result[k] for k in ('check_sha256','preflight_receipt_sha256','metadata_hashes_sha256')),
                          'manifest/preflight receipt mismatch')
            guard.require(guard.load_config(config_path)==c, 'configuration changed during evaluation')
            guard.require(guard.sha_file(out/'CHECK.json')==record['check_sha256'], 'checked metadata changed during evaluation')
            record['state']='completed'
    except BaseException as exc:
        record.update(state='failed',error=f'{type(exc).__name__}: {exc}')
    finally:
        cleanup_started=time.monotonic()
        deadline=cleanup_started+CLEANUP_SECONDS
        if record['state']!='completed':
            if child is not None and child.stdin is not None and not child.stdin.closed:
                try: child.stdin.close()
                except Exception as exc: record['cleanup_error']=str(exc)
            if job is not None:
                try: record['tree_gone']=job.stop(timeout=max(0.,deadline-time.monotonic()))
                except Exception as exc: record['cleanup_error']=str(exc)
            if child is not None:
                try: record['exit_code']=child.wait(timeout=max(0.,deadline-time.monotonic()))
                except Exception as exc: record['cleanup_error']=str(exc)
        if job is not None:
            try:
                record['job_gone_before_close']=job.gone()
                record['tree_gone']=record['job_gone_before_close'] and (child is None or child.poll() is not None)
                job.close(); record['job_closed']=True
            except Exception as exc:
                record.update(cleanup_error=str(exc),tree_gone=False)
        if not all(record[k] is True for k in ('tree_gone','job_gone_before_close','job_closed')):
            record.update(state='failed',error=record['error'] or 'owned job cleanup unconfirmed')
        record['elapsed_seconds']=time.monotonic()-began
        record['cleanup_elapsed_seconds']=time.monotonic()-cleanup_started
        if record['state']=='failed':
            atomic_json(out/'FAILURE.json',{'state':'failed','error':record['error'],
                                          'config_sha256':guard.digest(c),'retry':False,'controller':record})
        atomic_json(out/'controller.json',record)
        atomic_json(out/'final-audit.json',{'controller':record,'manual_visual_review':'PENDING',
                                          'scientific_claim':'single-seed DEV diagnostic only','config_sha256':guard.digest(c)})
    return record


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--action',choices=('Check','Start'),default='Check')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args(argv)
    try:
        guard.require(Path(sys.executable).resolve()==guard.PYTHON.resolve(),'fixed Python required')
        if a.worker:
            worker(a.config); return 0
        c=guard.load_config(a.config)
        if a.action=='Check':
            checked=guard.preflight(c)
            print(json.dumps({'state':'READY_WITHIN_SCOPE','config_sha256':guard.digest(c),'best':checked['best']}))
            return 0
        result=supervise(c,a.config)
        print(json.dumps(result,ensure_ascii=False))
        return 0 if result['state']=='completed' else 1
    except Exception as exc:
        print(json.dumps({'state':'REFUSED','error':f'{type(exc).__name__}: {exc}'},ensure_ascii=False),file=sys.stderr)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
