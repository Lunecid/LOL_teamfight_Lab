"""Inference child. The production caller gates this behind verified parent completion."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

from . import guard

PHASE_LIMITS = {'preflight': 1800, 'inference': 14400, 'metrics': 7200, 'report': 5400}
TOTAL_SECONDS = 28800
MEMORY_BYTES = 40 * 1024**3


def atomic_json(path, value):
    p = Path(path)
    tmp = p.with_name(p.name + f'.{os.getpid()}.tmp')
    data = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n'
    with tmp.open('w', encoding='utf-8') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, p)


class Budget:
    def __init__(self, output, config_sha256, *, clock=time.monotonic, measure=None):
        self.output, self.config_sha256 = Path(output), config_sha256
        self.clock, self.measure = clock, measure
        self.start = self.clock()
        self.phase = None
        self.peak_private = 0
        self.peak_upper = None
        self.samples = 0

    def enter(self, name):
        names = list(PHASE_LIMITS)
        guard.require(name in names and (self.phase is None and name == names[0] or
                      self.phase is not None and names.index(name) == names.index(self.phase) + 1), 'invalid phase transition')
        self.phase, self.phase_start = name, self.clock()
        atomic_json(self.output/'phase.json', {'phase': name, 'phase_started_monotonic': self.phase_start,
                    'pid': os.getpid(), 'config_sha256': self.config_sha256})
        self.check()

    def check(self):
        now = self.clock()
        if now - self.start >= TOTAL_SECONDS or now - self.phase_start >= PHASE_LIMITS[self.phase]:
            raise TimeoutError('evaluation phase/total wall budget exceeded')
        if self.measure is not None:
            private, upper = self.measure()
            guard.require(type(private) is int and private > 0, 'private memory sample unavailable')
            self.samples += 1
            self.peak_private = max(self.peak_private, private)
            if upper is not None:
                self.peak_upper = max(self.peak_upper or 0, upper)
            guard.require(private < MEMORY_BYTES, 'sampled private memory cap exceeded')


def load_best(reference, head, static, device):
    """Read ONLY the hash-bound best weight tensor dict; never last/optimizer/RNG."""
    import torch
    from c_ept.ept import encoder, heads
    p = Path(reference['path'])
    guard.require(p.name == 'best.pt' and not p.is_symlink(), 'best-only loader path mismatch')
    raw = p.read_bytes()
    guard.require(len(raw) == reference['bytes'] and hashlib.sha256(raw).hexdigest() == reference['sha256'],
                  'best payload differs from final head')
    guard.require(hashlib.sha256(static.tobytes()).hexdigest() == head['identity']['static_sha256'], 'static terrain differs from training')
    model = heads.EPT(encoder.EncoderConfig(**head['identity']['encoder']), static)
    state = torch.load(io.BytesIO(raw), weights_only=True, map_location='cpu')
    model.load_state_dict(state, strict=True)
    del state, raw
    return model.to(device).eval()


def _bundle(meta, *, check):
    # Store constructors validate metadata; adapter restricts ALL table access to
    # DEV through existing role-selecting APIs. Never verify/read whole payloads.
    from .stores import ReadOnlyVerifiedStore
    from c_ept.preprocessing import terrain
    from c_ept.valuation import value, towertier
    from c_ept.ept import encoder
    p = meta['paths']
    for filename, expected in meta['terrain_files_sha256'].items():
        check()
        guard.require(guard.sha_file(guard.inside(p['terrain'], filename)) == expected, 'terrain payload hash mismatch')
    for rel, expected in meta['tier_files_sha256'].items():
        check()
        guard.require(guard.sha_file(guard.inside(p['s3'], rel)) == expected, 'tier artifact hash mismatch')
    return SimpleNamespace(s1=ReadOnlyVerifiedStore(p['s1'], check=check),
                           s2=ReadOnlyVerifiedStore(p['s2'], check=check),
                           labels=ReadOnlyVerifiedStore(p['labels'], check=check),
                           static=encoder.static_channels(terrain.Terrain(p['terrain'])),
                           prices=value.load_prices(p['prices']), tier=towertier.TierModel.load(Path(p['s3'])/'fit/tier'),
                           rules=guard.read_json(p['rules']), meta=meta)


def execute(c, checked):
    """Production-only: caller owns the exclusive output and gated Windows Job."""
    from c_ept.workflows.s4a_runtime import prepare_native_runtime, validate_e6_probe
    prepare_native_runtime()  # BEFORE NumPy/PyTorch/LightGBM imports
    from c_ept.workflows import run_s1
    validate_e6_probe(run_s1.require_no_strata())  # local process evidence; no Strata service calls
    import numpy as np
    import torch
    import psutil
    from c_ept.ept import data, scales, train
    from c_ept.models import baselines_b1
    from .adapters import run_inference
    from .metrics import summarize_endpoints
    from .reporting import write_report
    train.deterministic_cpu(threads=8, seed=c['seed'])
    train._deterministic_gpu(c['seed'])
    guard.require(torch.cuda.is_available(), 'CUDA required; no CPU fallback')
    proc = psutil.Process()
    def measure():
        info = proc.memory_info()
        # Windows private has a precise meaning; no RSS-as-private substitution.
        return getattr(info, 'private', None), getattr(info, 'peak_pagefile', None)
    out = Path(c['output'])
    budget = Budget(out, guard.digest(c), measure=measure)
    budget.enter('preflight')
    meta = checked['inputs']
    # Recheck every immutable metadata binding before entering actual inference.
    for path, expected in checked['metadata_hashes'].items():
        guard.require(guard.sha_file(path) == expected, 'input metadata changed before inference')
    bundle = _bundle(meta, check=budget.check)
    stats = data.Stats.load(Path(c['training_run'])/'stats/STATS.json')
    factors = scales.load_scales(Path(c['training_run'])/'B0_SCALES.json', expected_provenance=meta['expected_provenance'])
    head = checked['parent']['head']
    identity = head['identity']
    guard.require(stats.sha256 == identity['fit']['stats_sha256'] == identity['stop']['stats_sha256'],
                  'FIT statistics differ from trained model identity')
    guard.require(identity['seed'] == c['seed'] and identity['device'] == c['device']
                  and identity['dtype'] == str(torch.float32) and identity['threads'] == c['native_threads'],
                  'trained runtime profile differs from evaluation')
    scale_digest = hashlib.sha256(json.dumps({k:float(factors[k]) for k in sorted(factors)}).encode()).hexdigest()
    guard.require(scale_digest == head['identity']['scales_sha256'], 'training scales changed')
    model = load_best(checked['best'], head, bundle.static, c['device'])
    b1 = baselines_b1.B1Heads.load(Path(c['s3'])/'fit/b1')
    budget.enter('inference')
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        result = run_inference(bundle, model, stats, b1, check=budget.check, device=c['device'],
                               batch_size=c['batch_size'], draws=c['draws'], seed=c['seed'], scales=factors)
    torch.cuda.synchronize()
    gpu = {'allocated_peak_bytes':torch.cuda.max_memory_allocated(), 'reserved_peak_bytes':torch.cuda.max_memory_reserved()}
    # Private reproducible prediction evidence. Keys contain no raw identifiers;
    # numeric row/match arrays remain local and never enter cloud/report prose.
    arrays, prediction_index = {}, []
    for group in ('endpoints','nll'):
        for i, endpoint in enumerate(result[group]):
            entry = {'group': group, 'index': i, 'name': endpoint['name'], 'arrays': {}}
            for key, value in endpoint.items():
                if isinstance(value, np.ndarray):
                    archive_key = f'{group}_{i}_{key}'
                    arrays[archive_key] = value
                    entry['arrays'][key] = {'key': archive_key, 'shape': list(value.shape), 'dtype': str(value.dtype)}
            prediction_index.append(entry)
    np.savez_compressed(out/'predictions.private.npz', **arrays)
    atomic_json(out/'prediction-index.json', {'schema': guard.SCHEMA, 'config_sha256': guard.digest(c),
                'private_local_only': True, 'entries': prediction_index})
    budget.check()
    budget.enter('metrics')
    summary = summarize_endpoints(result['endpoints'], result['nll'], bootstrap=c['bootstrap'], seed=c['seed'], check=budget.check)
    budget.enter('report')
    for store in (bundle.s1, bundle.s2, bundle.labels):
        store.verify_used()
    store_evidence = {key: getattr(bundle, key).used_evidence() for key in ('s1', 's2', 'labels')}
    atomic_json(out/'used-inputs.json', {'config_sha256': guard.digest(c), 'stores': store_evidence})
    provenance = {'training_run':c['training_run'],'training_commit':guard.TRAIN_COMMIT,'evaluation_commit':c['reviewed_commit'],
                  'config_sha256':guard.digest(c),'best':checked['best'],'b1_source_commit':guard.B1_COMMIT,
                  'comparison':'EPT25% versus existing whole-FIT B1; single training seed',
                  'definition_sha256':c['definitions_sha256'],'adapter':result['provenance'],
                  'gpu_memory':gpu,'peak_sampled_private_bytes':budget.peak_private,
                  'observed_lifetime_upper_bytes':budget.peak_upper,'memory_samples':budget.samples,
                  'memory_scope':'evaluation child only; sampled, not continuous upper bound',
                  'api_tokens':None,'cost':None,'elapsed_seconds':time.monotonic()-budget.start}
    write_report(summary, out/'reports', provenance=provenance, render=True)
    budget.check()
    # No source or immutable parent/input change may silently pass at the end.
    guard.clean_source(c['source'], c['reviewed_commit'])
    guard.clean_source(c['training_source'], guard.TRAIN_COMMIT)
    guard.validate_final_audit(c)
    latest_parent = guard.parent_metadata(c)
    guard.require(latest_parent['head'] == checked['parent']['head'], 'parent final head changed during evaluation')
    guard.verify_history_hashes(c['training_control'])
    for path, expected in checked['metadata_hashes'].items():
        guard.require(guard.sha_file(path) == expected, 'input metadata changed during evaluation')
    atomic_json(out/'RESULT.json', {'state':'completed','config_sha256':guard.digest(c),'best':checked['best'],
                                   'provenance':provenance,'prediction_sha256':guard.sha_file(out/'predictions.private.npz')})
