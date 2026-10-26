"""The store builder for the supervised tests of the Gate 0 probe (``c_ept/tests/test_run_gate0.py``, EPT-T013): what a child process of the supervisor imports by name.

Not a test module (no ``test_`` prefix, nothing here is collected). ``toy_bundle(meta, check)`` stands for ``comparison.runtime.load_bundle`` (the verified stores of a T009 run): it opens the toy world
whose folder the made-up inputs record names (``meta['toy']['root']``), adds the arrays that the heads world lacks (an empty objectives table, the label arrays the inference reads) and takes the static
channels from ``ept_toys.toy_static``. A run that uses it is recorded as not using the production loader and so has no decision.

해석: 감독자 시험의 자식 프로세스가 이름으로 불러 쓰는 장난감 저장소 적재기이다. 실제 T009 저장소 대신 합성 세계를 열고, 이것을 쓴 실행은 판정을 내지 못한다고 기록된다.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np


class Overlay:
    """A store with some arrays replaced or added by hand: ``array(name)`` answers from ``arrays`` first and then from ``base``."""

    def __init__(self, base, arrays):
        self.base, self.arrays = base, arrays

    def array(self, name):
        return self.arrays[name] if name in self.arrays else self.base.array(name)

    def __getattr__(self, name):
        return getattr(self.base, name)


def toy_bundle(meta, check):
    """The bundle of the toy world named by ``meta['toy']['root']`` (``ept_toys.build_ept_world``): role DEV, as the production builder declares it."""
    from c_ept.data.store_core import Store
    from c_ept.tests.ept_toys import toy_static
    root = Path(meta['toy']['root'])
    base_s1, labels, s2 = Store(root / 'base' / 's1'), Store(root / 'base' / 'labels'), Store(root / 's2')
    n_matches, n_units = len(base_s1.array('matches/role')), len(base_s1.array('units/f'))
    s1 = Overlay(base_s1, {'matches/obj_off': np.zeros(n_matches + 1, dtype=np.int64), **{f'objectives/{k}': np.empty(0, dtype=np.int64) for k in ('t', 'kind', 'sub', 'taker')}})
    labels = Overlay(labels, {'labels/f0_unit': np.resize(np.array([-1., 0., 1.]), n_units), 'labels/gold_reps': np.arange(31, dtype=float) * 100})
    return SimpleNamespace(role='DEV', s1=s1, s2=s2, labels=labels, static=toy_static(), meta=meta)
