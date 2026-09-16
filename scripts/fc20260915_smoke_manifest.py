"""TRAIN-only smoke layout: reference already-written MAIN extraction chunks whose matches are ALL TRAIN.

Writes outputs/full_corpus_training_20260915/smoke_train_only/extract/MAIN/extraction_manifest.json. Smoke stages
remap these TRAIN matches to pseudo roles (fc20260915_data.smoke_role); no VALIDATION/TEST match is referenced.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fc20260915_common as C  # noqa: E402


def main(n_chunks=int(sys.argv[1]) if len(sys.argv) > 1 else 15):
    src = C.OUT / 'extract' / 'MAIN'
    chunks, names, proto = [], None, None
    for c in range(100000):
        sp = src / 'states' / f'chunk_{c:05d}.npz'
        op = src / 'outcomes_SEALED' / f'chunk_{c:05d}.npz'
        if len(chunks) >= n_chunks:
            break
        if not (sp.exists() and op.exists()):
            continue
        with np.load(sp, allow_pickle=False) as z:
            if set(z['m_role'].tolist()) != {'TRAIN'}:
                continue
            names = z['names'].tolist()
            chunks.append(dict(chunk=c, matches=int(len(z['m_match'])), plan_sha256=str(z['plan_sha256']),
                               path=str(sp.relative_to(C.OUT)), outcome_path=str(op.relative_to(C.OUT)),
                               states_sha256=C.sha256_file(sp), outcomes_sha256=C.sha256_file(op)))
    man = dict(set='MAIN', smoke=True, source='references TRAIN-only chunks of extract/MAIN', protocol_sha256=C.sha256_file(C.OUT / 'protocol.json'),
               names=names, names_sha256=C.sha256_json(names), matches=sum(c['matches'] for c in chunks), chunks=chunks)
    import os
    tag = os.environ.get('FC_SMOKE_TAG', '')
    C.write_json(C.OUT / ('smoke_train_only' + (f'_{tag}' if tag else '')) / 'extract' / 'MAIN' / 'extraction_manifest.json', man)
    print(dict(chunks=len(chunks), matches=man['matches']))


if __name__ == '__main__':
    main()
