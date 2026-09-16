"""Stages P+Z: protocol.json (once) and the frozen manifest (plan freeze; no fitting) before MAIN TEST or the extraction chunks are
opened for the full pass. --smoke writes the smoke freeze under smoke_validation_only/."""
from __future__ import annotations

import os

os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fc20260915_common as C  # noqa: E402
import vd20260916_common as V  # noqa: E402

import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    V.log_command()
    base = V.SMOKE if args.smoke else V.OUT
    fz_fc = C.read_json(V.FC / 'frozen_manifest.json')
    adapter = C.load_v_adapter(V.V_ADAPTER, fz_fc['v_final_sha256'])
    names, groups, gidx, tmask = V.column_map(adapter)
    b, b0 = V.beta(adapter)
    if not (V.OUT / 'protocol.json').exists():
        snap = V.OUT / 'integrity' / 'snapshot_before.json'
        if not snap.exists():
            raise SystemExit('integrity snapshot_before.json must precede the protocol')
        before = C.read_json(snap)
        proto = dict(version=V.VERSION, role=V.ROLE_TAG, written_at=time.strftime('%Y-%m-%d %H:%M:%S'), design='docs/CLAUDE_V_MECHANISM_20260916.md (Claude, continuing the Codex line); no fitting',
                     spec_sha256=C.sha256_file(V.SPEC), provenance_docs_sha256={p.name: C.sha256_file(p) for p in V.PROVENANCE_DOCS},
                     prior_test_exposure='Exploratory follow-up; MAIN TEST labels and the V endpoint predictions were evaluated before (full corpus run).',
                     parents=dict(read_only=['outputs/full_corpus_training_20260915', 'outputs/cohort_role_training_20260915', 'source implementations', 'raw data'],
                                  integrity_snapshot_before_sha256=C.sha256_file(snap), full_corpus_frozen_manifest_sha256=C.sha256_file(V.FC / 'frozen_manifest.json'),
                                  adapter=V.V_ADAPTER.relative_to(V.ROOT).as_posix(), adapter_sha256=fz_fc['v_final_sha256'], selection_v=C.read_json(V.FC / 'selection_v.json')['chosen'],
                                  labels_test_external_manifest_sha256=C.sha256_file(V.FC / 'labels' / 'labels_test_external_manifest.json'),
                                  labels_trainval_manifest_sha256=C.sha256_file(V.FC / 'labels' / 'labels_trainval_manifest.json'),
                                  extraction_manifest_sha256=C.sha256_file(V.FC / 'extract' / 'MAIN' / 'extraction_manifest.json'), n_chunks=len(V.chunk_files()),
                                  expected_parent_file_sha256=dict(before['sha256_full_corpus_read_targets'])),
                     model=dict(pipeline=[(n, type(s).__name__) for n, s in adapter.base.steps], calibration=adapter.calibration, C=float(adapter.base.steps[-1][1].C),
                                intercept=b0, state_columns=len(adapter.state_names), feature_columns=len(adapter.feature_names), transformed_columns=len(names),
                                transformed_names_sha256=C.sha256_json(names), group_sizes={g: int(len(gidx[g])) for g in V.GROUPS}, time_related_columns=int(tmask.sum()),
                                identity='Delta-logit_i = beta^T (t(S_e) - t(S_pre)) = sum_j c_ij; Y_h90 = 1[Delta-logit_i > 0] (raw calibrator, strictly monotone sigmoid)'),
                     rows=dict(sets=list(V.SETS), horizon=V.H, rows='all h90-valid rows (no sampling)', strata=dict(cohort='T / N', same_frame='post snapshot timestamp == pre snapshot',
                               abs_delta_V=[V.stratum_name(k) for k in range(len(V.DELTA_STRATA))], pre_time_bands=list(V.BANDS)), top_columns=V.TOP_COLUMNS),
                     checks=['p_pre / p_post recomputed from stored states equal the parent labels exactly', 'sum_k g_ik = Delta-logit_i (< 1e-9)', 'champion block = 0 exactly',
                             'logit(p_post) - logit(p_pre) = Delta-logit (< 1e-8)', 'sign(Delta-logit) = Y_h90 on every row', 'all valid rows collected once (counts)',
                             'sealed access only after this run\'s freeze', 'parent files unchanged'],
                     not_done=['V(S(L)) and the pre -> L -> e split (state at L not stored)', 'causal reading of any term', 'changes to V, labels or q'],
                     smoke=dict(rows='MAIN_VALIDATION rows of chunks 370-429 (the chunks are ordered TRAIN, VALIDATION, TEST; unsealed)'), environment=V.environment(), source_sha256=V.own_source_hashes())
        print('protocol sha256', C.write_json(V.OUT / 'protocol.json', proto))
    if V.frozen_path(base).exists():
        raise SystemExit('frozen manifest exists (never overwritten)')
    if not args.smoke:
        ct = V.OUT / 'contract_tests' / 'result.json'
        if not ct.exists() or not C.read_json(ct).get('passed'):
            raise SystemExit('contract tests must pass before the full freeze')
        if not (V.frozen_path(V.SMOKE).exists() and (V.SMOKE / 'results.json').exists() and C.read_json(V.SMOKE / 'results.json')['checks']['pass']):
            raise SystemExit('smoke must complete and pass before the full freeze')
        if (V.OUT / 'results.json').exists():
            raise SystemExit('results.json exists before freeze')
    access = [json.loads(l) for l in (base / 'access_log.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()] if (base / 'access_log.jsonl').exists() else []
    if any(a.get('sealed') for a in access):
        raise SystemExit('sealed access before freeze')
    frozen = dict(role=V.ROLE_TAG, version=V.VERSION, smoke=bool(args.smoke), protocol_sha256=C.sha256_file(V.OUT / 'protocol.json'), adapter_sha256=fz_fc['v_final_sha256'],
                  adapter_file_sha256_now=C.sha256_file(V.V_ADAPTER), transformed_names_sha256=C.sha256_json(names), beta_sha256=C.sha256_bytes(np.asarray(b, dtype=np.float64).tobytes()),
                  group_sizes={g: int(len(gidx[g])) for g in V.GROUPS}, time_related_columns=int(tmask.sum()), n_chunks=len(V.chunk_files()),
                  source_sha256=V.own_source_hashes(), frozen_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                  statement='Adapter, column map and identities are frozen here; MAIN TEST labels and the extraction chunks are opened for the full pass only after this file exists.')
    if frozen['adapter_file_sha256_now'] != frozen['adapter_sha256']:
        raise SystemExit('adapter file hash differs from the parent freeze')
    print('frozen sha256', C.write_json(V.frozen_path(base), frozen))


if __name__ == '__main__':
    main()
