"""Loader and backend for the evaluation-runner tests (``c_ept/tests/test_run_evaluate.py``, S4b-prep K6b): what a child process of the supervisor imports by name.

Not a test module (no ``test_`` prefix, nothing here is collected). ``toy_eval_inputs`` is an inputs loader (``module:function``, the ``loader`` argument of the stage
body) that opens the toy stores of the manifest, takes the static channels from ``ept_toys.toy_static`` instead of a terrain folder and reads the dropped columns of the
matched B1 from ``B1_DROPPED.json`` next to the manifest. ``SyntheticBackend`` is a backend (``backend`` argument) that builds no model and answers the inference with the
endpoints of three made-up seeds, so that a supervised run needs no prices, tier model or fitted B1.

해석: 감독자 시험의 자식 프로세스가 이름으로 불러 쓰는 장난감 입력 적재기와 가짜 추론 뒤꿈치이다. 모형을 만들지 않고 시드 셋의 합성 끝점으로 추론을 대신한다.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from c_ept.run import evaluate as run_evaluate
from c_ept.run import io as run_io
from c_ept.run import runners

IDS = np.arange(10, 14, dtype=np.int64)
SEEDS = (20261022, 20261023, 20261024)


def toy_eval_inputs(manifest_path) -> run_evaluate.EvalInputs:
    from c_ept.tests.ept_toys import toy_static
    manifest = run_io.read_json(manifest_path)
    s1, s2, labels = runners.open_stores(manifest)
    values, path = runners.read_scales(manifest)
    base = runners.Inputs(s1=s1, s2=s2, labels=labels, static=toy_static(), static_source=None, scales=values, scales_path=path)
    dropped = Path(manifest_path).with_name('B1_DROPPED.json')
    b1 = SimpleNamespace(dropped=tuple(run_io.read_json(dropped)['dropped'])) if dropped.exists() else None
    return run_evaluate.EvalInputs(base=base, b1=b1)


class Overlay:
    """A store with some arrays replaced or added by hand: ``array(name)`` answers from ``arrays`` first and then from ``base`` (the heads toy world has no objectives table, no
    ``labels/f0_unit``, which the real inference of the DEV rehearsal reads)."""

    def __init__(self, base, arrays):
        self.base, self.arrays = base, arrays

    def array(self, name):
        return self.arrays[name] if name in self.arrays else self.base.array(name)

    def __getattr__(self, name):                                            # ``root``, ``file``, ``manifest`` ... of the base store
        return getattr(self.base, name)


def constant_b1(dropped=()):
    """A matched-B1 stand-in with the real canonical prediction classes and a degenerate hand-made state (the constant paths of the heads, no fit, no file): the plain kill law and the raw K
    condition, fitted on FIT with selection on STOP (``fit_roles``, ``stop_roles``), with the dropped columns ``dropped``."""
    from c_ept.models import baselines_b1 as B1
    columns = B1.features.b1_columns(tuple(dropped))
    groups = {}
    for kind, names in (('masked', B1.HEAD_NAMES), ('open', B1.OPEN_HEAD_NAMES)):
        group = {}
        for name in names:
            head = getattr(B1, name)(kind)
            layout = B1.make_layout(name, kind, columns)
            for member in head.parts:
                member.layout_, member.n_features_ = layout, layout.width
                member.info_ = [{'synthetic': True}]
                if hasattr(member, 'base_'):
                    width = {'M2Rate': 1, 'M3Kills': 2, 'M3Obj': 18, 'M4Conv': 18}[name]
                    member.boosters_ = [None] * width
                    member.base_ = np.full(width, np.log(.02 if name in ('M2Rate', 'M4Conv') else .4))
                    member.squeeze_ = name == 'M2Rate'
                else:
                    member.boosters_, member.classes_ = [None], np.array([0])
            group[name] = head
        groups[kind] = group
    return B1.B1Heads(groups['masked'], groups['open'], tuple(dropped))


def real_eval_inputs(base, *, dropped=(), labels_extra=None) -> run_evaluate.EvalInputs:
    """``EvalInputs`` for the real inference on the toy world of ``base`` (``runners.Inputs``): the objectives table made empty (the heads world has none), ``labels/f0_unit`` and
    ``labels/gold_reps`` added (``labels_extra`` adds or replaces more), flat prices, the toy tower-tier model and a ``constant_b1``. Nothing is read from a file."""
    import dataclasses
    from c_ept.tests.s3_toys import toy_tier_model
    from c_ept.valuation import flowprice
    n_matches, n_units = len(base.s1.array('matches/role')), len(base.s1.array('units/f'))
    s1 = Overlay(base.s1, {'matches/obj_off': np.zeros(n_matches + 1, dtype=np.int64), **{f'objectives/{k}': np.empty(0, dtype=np.int64) for k in ('t', 'kind', 'sub', 'taker')}})
    labels = Overlay(base.labels, {'labels/f0_unit': np.resize(np.array([-1., 0., 1.]), n_units), 'labels/gold_reps': np.arange(31, dtype=float) * 100, **(labels_extra or {})})
    return run_evaluate.EvalInputs(base=dataclasses.replace(base, s1=s1, labels=labels), prices={k: 100. for k in flowprice.GROUPS}, tier=toy_tier_model(), b1=constant_b1(dropped))


TOY_LOADER = 'c_ept.tests.run_evaluate_bodies:toy_eval_inputs'
SYNTHETIC = 'c_ept.tests.run_evaluate_bodies:SyntheticBackend'


def evaluate_stage_with_toy_defaults(run_dir, *texts) -> None:
    """The stage body ``evaluate`` of a supervised toy run that stands for a run with the default loader and backend: it sets ``run_evaluate.LOADER`` and ``BACKEND`` to the toy
    ones in the child process and then runs the real stage body. The role TEST refuses the test-only switch ``allow_custom``, so a toy TEST run (and its DEV rehearsal) can only be made
    this way. It is reachable through the ``bodies`` argument of ``start`` that tests use; nothing in ``start`` or in a command line sets the two names (review r1, K6b)."""
    run_evaluate.LOADER, run_evaluate.BACKEND = TOY_LOADER, SYNTHETIC
    run_evaluate.evaluate_stage(run_dir, *texts)


def twin_stage_with_toy_defaults(run_dir, *texts) -> None:
    """``evaluate_stage_with_toy_defaults`` under another name (S4b-prep K6c, K6b review r4): a TEST run made with this body differs from a rehearsal made with the first only in the
    stage body, which the record names by module, name and the hash of the file and which the TEST gate compares."""
    evaluate_stage_with_toy_defaults(run_dir, *texts)


FLAGS_AT_LOAD: list = []                                                  # what a test reads after ``flags_probe_loader`` ran (it runs in the test's own process)


def flags_probe_loader(manifest_path):
    """An inputs loader that writes down the numeric flags that are in force when the inputs are loaded and then stops the stage (m3 of the K6c review)."""
    import torch
    FLAGS_AT_LOAD.append({'cudnn_deterministic': bool(torch.backends.cudnn.deterministic), 'deterministic_algorithms': bool(torch.are_deterministic_algorithms_enabled()),
                          'float32_matmul_precision': torch.get_float32_matmul_precision()})
    raise RuntimeError('flags_probe_loader: stop after the flags were read')


class SyntheticBackend(run_evaluate.Backend):
    """No model, a made-up split of four matches, the C4 endpoint of three seeds (EPT better than the matched B1) and a canned STOP report."""

    def evaluates(self, role):
        return True                                                    # a made-up backend answers for every role, TEST too (the default backend cannot until a TEST reader of the stores exists: S8)

    def build_model(self, trained, static, device):
        return object()

    def split_ids(self, inputs, role):
        return IDS

    def stop_report(self, model, trained, inputs, constants, check):
        return {'loss_stop': 1.5, 'blocks': {'M2Rate/masked': {'mean': 1.0, 'scale': 2.0, 'term': 0.5}, 'M2Mark/masked': {'mean': 1.0, 'scale': 2.0, 'term': 0.5}}}

    def inference(self, model, trained, inputs, *, role, profile, reps, constants, check, exposure='observed', population='all', opening=None, progress=None):
        from c_ept.evaluation import endpoints
        match = np.repeat(IDS, 3)
        y = np.array([1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0], dtype=np.int64)
        p_b1 = np.full(len(match), 0.5)
        p_ept = np.clip(p_b1 + np.where(y == 1, 0.1, -0.1) * (1 + 0.1 * (trained.seed - SEEDS[0])), 0.0, 1.0)
        found = endpoints.endpoint('M2/t0/H60/occurrence', 'M2', 't0', 60, 'first_kill_ge1', match, np.arange(len(match), dtype=np.int64), y, p_ept, p_b1, counts={})
        return {'endpoints': [found], 'nll': [], 'provenance': {'role': role, 'exposure': exposure, 'population': population}}
