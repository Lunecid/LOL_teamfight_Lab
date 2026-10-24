"""Tests of ``c_ept.run_s3``: the staged, resumable S3 runner (stage S3 Task 12).

No real run folder, raw file, catalog or index is read: every store is a toy one under pytest's folders, the s3a files and the decision
record are made here (``write_s3a``: the SHA-256 of the files are put into a SPEC file of the test), and the runner is told where to
write by ``--runs-root`` and ``--catalog-db``. Four kinds of check:

* the command line with the stages replaced by fakes (``install_fakes``): the order of the stages, the record in STAGES.json, resume,
  the refusal of other inputs, of a changed file and of an s3a file whose hash is not the one of the decision record, the E6 guard (a
  fake process list), the lock, the smoke selection;
* the stages one by one with fakes for what they call (the Composer, the models, ``value.accounting``) where the point is how the
  runner calls them: the roles that are pinned, ``check=``, the blocks and the workers of the compose calls, the R31 screen, the chunks;
* the arithmetic of the gate (``gate_results``, ``brier_comparison``, ``tower_pricing_diagnostic``, ``observed_onsets``) on arrays whose
  answers were worked out by hand or are known by construction, and the rule that ``gate_pass`` needs nothing missing (ruling R27);
* one run of all five stages with the real functions on a small random world (the module fixture ``e2e``): 48 synthetic matches sent
  through the real S1 extraction, a random S2, B0, B1, B2, the tier law, the composition with spawned workers and the gate. The numbers
  of that world mean nothing (the heads are fitted on a few hundred rows); what is checked is that every stage runs on the smoke subset,
  how it calls what it calls, and what it records.

The toy match keys (``KEY_PREFIX``) are sentinels: no file, no message and no error text that the runner writes may hold one.
"""
import ast
import collections
import contextlib
import gc
import hashlib
import io
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

from c_ept import (
    baselines_b0,
    baselines_b1,
    compose,
    features,
    flowprice,
    gate_s3,
    nll,
    rulepay,
    run_s1,
    run_s3,
    runlog,
    store_core,
    towertier,
    value,
)
from c_ept import config as C
from c_ept.store_core import Store
from c_ept.tests import s3_toys, synth
from c_ept.tests.conftest import all_imports, empty_process_list, module_level_imports
from c_ept.tests.s3_toys import (  # noqa: F401  (the fixtures are imported by ruling R1)
    KEY_PREFIX,
    TOY_ONSET_BINS,
    TOY_PRICES,
    build_value_s1,
    onset_toy_s1,
    random_value_matches,
    random_value_record,
    toy_key,
    toy_s1,
    toy_s2,
)

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / 'c_ept' / 'workflows' / 'run_s3.py'
STAGES = run_s3.STAGES
P3_BASE = tuple(300 + 10 * max(0, level - 6) for level in range(1, 19))
CONSTANT_IN_FIT = 'rule_uncertain_structures'              # an S2 summary column that b0 does not read: constant in the FIT views


# ---- the files and stores the runner is given --------------------------------------------------------------------------------------------
def rules_document() -> dict:
    """A ``rule_payouts_v2`` table that says what decision P3 says (every f0 rule verified for 15.18 - 15.22)."""
    def entry(v):
        return {'patch_range': ['15.18', '15.22'], 'value': v, 'unit': 'gold', 'status': 'verified', 'sources': ['toy'], 'note': 'toy'}
    rules = {'champion_kill_base_bounty': [entry({'level_1_to_6': 300, 'by_level': list(P3_BASE)})],
             'assist_bounty': [entry({'pool_share': 0.5, 'cap_share_of_base': 0.5})],
             'assist_bounty_early_game_multiplier': [entry({'flat_until_s': 90, 'full_at_s': 210, 'start_multiplier': 0.5,
                                                            'end_multiplier': 1.0})],
             'first_blood_bonus': [entry({'killer_bonus': 0, 'assist_pool_bonus': 0})]}
    return {'schema': 'rule_payouts_v2', 'created': '2026-10-21', 'rules': rules, 'sources': {'toy': 'toy'}}


def sha_of(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_spec(path: Path, prices_sha: str, rules_sha: str) -> Path:
    path.write_text(
        json.dumps({'decisions': {'S3_prices': {'prices': {'sha256': prices_sha}, 'rules': {'sha256': rules_sha}}}}), encoding='utf-8'
    )
    return path


def write_s3a(root: Path, name: str = '20261021_c_ept_s3a_1', bump: float = 0.0) -> types.SimpleNamespace:
    """An s3a run folder with the toy prices (each plus ``bump``) and the P3 rule table, and a SPEC file whose ``decisions.S3_prices`` holds
    their hashes."""
    run = root / name
    (run / 'prices').mkdir(parents=True)
    (run / 'rules').mkdir(parents=True)
    prices, rules = run / 'prices' / 'PRICES.json', run / 'rules' / 'RULE_PAYOUTS.json'
    prices.write_text(json.dumps({g: {'value': v + bump} for g, v in TOY_PRICES.items()}), encoding='utf-8')
    rules.write_text(json.dumps(rules_document()), encoding='utf-8')
    spec = write_spec(root / 'SPEC.json', sha_of(prices), sha_of(rules))
    return types.SimpleNamespace(run=run, prices=prices, rules=rules, spec=spec)


def build_run_s2(root: Path, s1, seed: int = 7) -> Store:
    """An S2 store for any S1 store: a frame-clock view for every frame row and an event view for every event row (shuffled together, as
    the real one is), random features (the game time is the reference time), and one frame-clock view of every FIT match whose cross-team
    distances are all missing (so that b0 has cells to fill)."""
    g = lambda n: np.asarray(s1.array(n))                                                          # noqa: E731
    n_frame, n_event = len(g('frame_rows/t0')), len(g('event_rows/kind'))
    src = np.concatenate([np.zeros(n_frame), g('event_rows/kind')])
    index = np.concatenate([np.arange(n_frame), np.arange(n_event)])
    match = np.concatenate([g('frame_rows/match'), g('units/match')[g('event_rows/unit')]])
    ref = np.concatenate([g('frame_rows/t0'), g('event_rows/ref')])
    t0 = np.concatenate([g('frame_rows/t0'), g('event_rows/t0')])
    frame_k = np.concatenate([g('frame_rows/k'), g('event_rows/frame_k')])
    rng = np.random.default_rng(seed)
    from c_ept import tokens
    n_views = n_frame + n_event
    order = rng.permutation(n_views)
    s2 = Store(root)
    for name, arr, dtype in (('src', src, 'i1'), ('src_index', index, 'i4'), ('match', match, 'i4'), ('ref', ref, 'i8'), ('t0', t0, 'i8'),
                             ('frame_k', frame_k, 'i2')):
        s3_toys._put(s2, f'views/{name}', arr[order], dtype)
    summary = rng.integers(0, 6, (n_views, len(tokens.SUMMARY))).astype(np.float32)
    summary[:, tokens.SUMMARY.index('game_time_s')] = ref[order] / 1000.0
    fit_matches = np.flatnonzero(g('matches/role') == C.ROLES.index('FIT'))
    summary[np.isin(match[order], fit_matches), tokens.SUMMARY.index(CONSTANT_IN_FIT)] = 1.0      # one value in every FIT view only
    players = rng.normal(5.0, 3.0, (n_views, 10, len(tokens.PLAYER_NUM))).astype(np.float32)
    geo = (rng.integers(1, 90, (n_views, 10, 10)) * 100).astype(np.float16)
    geo[:, np.arange(10), np.arange(10)] = 0
    match_shown, src_shown = match[order], src[order]
    for m in np.flatnonzero(g('matches/role') == C.ROLES.index('FIT')):  # a frame-clock view of every FIT match without distances
        hit = np.flatnonzero((match_shown == m) & (src_shown == 0))
        if hit.size:
            geo[hit[0], :5, 5:] = np.nan
            geo[hit[0], 5:, :5] = np.nan
    s3_toys._put(s2, 'x/summary', summary, 'f4')
    s3_toys._put(s2, 'x/players_num', players, 'f4')
    s3_toys._put(s2, 'x/geo_pp', geo, 'f2')
    s2.finalize()
    return s2


def release(*stores) -> None:
    """Close the memory maps (Windows does not delete a folder that holds a mapped file)."""
    for s in stores:
        s.maps.clear()
    gc.collect()


PINNED_CODE_STATE = {'code_commit': 'c' * 40, 'code_dirty': False}


@pytest.fixture(scope='module', autouse=True)
def _pinned_code_state():
    """The code state that RUN.json, the head records and the progress log hold, pinned for the module: a resume is refused when the code
    differs from the one the run began on or when the tree is dirty (G-S3b B3), so no test may depend on the state of the working tree
    that it runs in. A test that needs another state patches ``runlog.code_state`` itself."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(runlog, 'code_state', lambda: dict(PINNED_CODE_STATE))
        yield


@pytest.fixture(scope='module')
def toy_runs(toy_s1, toy_s2, tmp_path_factory):
    """The toy S1 and S2 stores of ``s3_toys`` as the two run folders the command line takes (the store folders themselves are accepted)
    and the s3a files; the stores are read only."""
    root = tmp_path_factory.mktemp('run_s3_toy_inputs')
    s3a = write_s3a(root)
    return types.SimpleNamespace(s1=toy_s1.root, s2=toy_s2.root, s3a=s3a.run, spec=s3a.spec, root=root, files=s3a)


class World:
    """Names for the random world of ``world``: the S1 run folder, the S2 run folder, the s3a files."""

    def __init__(self, root: Path, s3a: types.SimpleNamespace):
        self.root, self.s3a = root, s3a
        self.s1, self.s2, self.spec = root / '20261020_c_ept_s1_1', root / '20261021_c_ept_s2_1', s3a.spec
        self.s3a_run = s3a.run


@pytest.fixture(scope='module')
def world(tmp_path_factory):
    """48 random matches (roles FIT, FIT, STOP, DEV, CAL, AUDIT in turn) through the real S1 extraction, a random S2 and the s3a files."""
    root = tmp_path_factory.mktemp('run_s3_world')
    matches = random_value_matches(48, first=2000)
    s1 = build_value_s1(root / '20261020_c_ept_s1_1' / 's1', matches=matches, records=random_value_record)
    s2 = build_run_s2(root / '20261021_c_ept_s2_1' / 's2', s1)
    release(s1, s2)
    w = World(root, write_s3a(root))
    yield w
    gc.collect()
    shutil.rmtree(root, ignore_errors=True)


def cli_args(tmp_path: Path, s1, s2, s3a, spec, *extra) -> list:
    return ['--s1', str(s1), '--s2', str(s2), '--s3a', str(s3a), '--spec', str(spec), '--runs-root', str(tmp_path / 'runs'),
            '--catalog-db', str(tmp_path / 'catalog.sqlite'), *extra]


def toy_cli(tmp_path, toy_runs, *extra) -> int:
    return run_s3.main(cli_args(tmp_path, toy_runs.s1, toy_runs.s2, toy_runs.s3a, toy_runs.spec, *extra))


def catalog_rows(tmp_path):
    db = tmp_path / 'catalog.sqlite'
    if not db.exists():
        return []
    con = sqlite3.connect(db)
    try:
        return con.execute('SELECT run, stage FROM runs').fetchall()
    finally:
        con.close()


def only_run(tmp_path) -> Path:
    runs = sorted((tmp_path / 'runs').iterdir())
    assert len(runs) == 1
    return runs[0]


def read_json(path) -> dict:
    return json.loads(Path(path).read_text(encoding='utf-8'))


# ---- fakes for the stages ---------------------------------------------------------------------------------------------------------------
def make_fake(stage: str, calls: list, seen=None):
    """A stand-in for a stage function: it records its name (and the context, when ``seen`` is given), writes the output files that
    ``STAGE_OUTPUTS`` names for the stage (a store for the folders) and returns a small result."""
    def fake(ctx):
        calls.append(stage)
        if seen is not None:
            seen[stage] = ctx
        for rel in run_s3.STAGE_OUTPUTS[stage]:
            path = ctx.run / rel
            if rel.endswith('/'):
                s = Store(path)
                s3_toys._put(s, 'labels/x', [1.0, 2.0], 'f8')
                s.finalize()
                release(s)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f'{stage}:{rel}'.encode('utf-8'))
        return {'gate_pass': True} if stage == 'gate' else {'stage': stage}
    return fake


@pytest.fixture
def fakes(monkeypatch):
    """The five stage functions replaced by ``make_fake``; ``fakes.calls`` is the list of the stages that ran, in order."""
    calls = []
    for stage in STAGES:
        monkeypatch.setitem(run_s3.STAGE_FUNCTIONS, stage, make_fake(stage, calls))
    return types.SimpleNamespace(calls=calls)


# ---- units: numbers, JSON, memory, time -------------------------------------------------------------------------------------------------
def test_the_json_of_a_run_holds_plain_values_and_no_nan():
    text = run_s3.render(
        {
            'a': np.float64(1.5),
            'b': np.int64(3),
            'c': np.array([1, 2]),
            'd': float('nan'),
            'e': [np.float32(np.inf), np.bool_(True)],
            'f': {'z': 1, 'y': None},
            'g': Path('x'),
        }
    )
    assert json.loads(text) == {'a': 1.5, 'b': 3, 'c': [1, 2], 'd': None, 'e': [None, True], 'f': {'y': None, 'z': 1}, 'g': 'x'}
    assert text.endswith('\n') and text.index('"a"') < text.index('"b"')                                        # sorted keys


def test_a_peak_in_bytes_becomes_gib_and_nothing_measured_becomes_none():
    assert run_s3._gib(2 ** 30) == 1.0 and run_s3._gib(3 * 2 ** 29) == 1.5 and run_s3._gib(0) is None


def test_the_sampler_gives_each_scope_its_own_peak_and_scopes_nest():
    values = iter([10, 50, 20, 5])
    sampler = run_s3._Sampler(interval=1e9, measure=lambda: next(values))                  # no thread: the scopes sample at their edges
    with sampler.scope() as outer:
        with sampler.scope() as inner:
            pass
    assert (inner.peak, outer.peak, sampler.peak, sampler.samples) == (50, 50, 50, 4)
    assert inner.wall >= 0 and outer.wall >= inner.wall
    after = iter([7, 3])
    sampler2 = run_s3._Sampler(interval=1e9, measure=lambda: next(after))
    with sampler2.scope() as only:
        pass
    assert only.peak == 7 and sampler2.peak == 7


def test_the_sampler_thread_samples_while_the_call_runs_and_stops():
    readings = []

    def measure():
        readings.append(1)
        return 2 ** 30
    with run_s3._Sampler(interval=0.01, measure=measure) as sampler:
        time.sleep(0.15)
    assert sampler.samples >= 5 and sampler.peak == 2 ** 30
    n = len(readings)
    time.sleep(0.05)
    assert len(readings) == n                                                               # the thread is stopped


def test_the_real_sampler_reads_the_private_memory_of_the_process_tree():
    assert run_s3._tree_private_bytes() > 10 * 2**20  # a python process with numpy: tens of MiB at least


def test_the_private_memory_of_a_child_process_is_in_the_sum_of_the_tree():
    """The reviewer of Task 12 found no test that the sum covers the children (the compose workers and the LightGBM threads' helpers)."""
    code = ('import sys\n'
            'block = bytearray(200 * 2**20)\n'
            'block[::4096] = bytes([1]) * len(range(0, len(block), 4096))\n'                      # every page is touched
            'print("ready", flush=True)\n'
            'sys.stdin.readline()\n')
    child = subprocess.Popen([sys.executable, '-c', code], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == 'ready'
        with_child = run_s3._tree_private_bytes()
    finally:
        child.stdin.write('\n')
        child.stdin.flush()
        child.wait(timeout=60)
    assert with_child - run_s3._tree_private_bytes() >= 100 * 2**20                                # the child holds 200 MiB of it


# ---- G-S3b B3 item 4: the peak memory also has an upper bound (the peak_pagefile of every process of the tree) --------------------------
MemoryInfo = collections.namedtuple('MemoryInfo', 'private rss peak_pagefile')
MemoryInfoWithoutPeak = collections.namedtuple('MemoryInfoWithoutPeak', 'private rss')


class FakeProcess:
    """What ``_tree_memory`` asks of a process: ``memory_info()``."""

    def __init__(self, info=None, error=None):
        self.info, self.error = info, error

    def memory_info(self):
        if self.error is not None:
            raise self.error
        return self.info


def test_the_memory_of_a_list_of_processes_is_the_sum_of_their_private_bytes_and_of_their_peak_pagefile():
    private, upper = run_s3._tree_memory([FakeProcess(MemoryInfo(100, 200, 300)), FakeProcess(MemoryInfo(50, 60, 70))])
    assert (private, upper) == (150, 370)
    assert run_s3._tree_memory([FakeProcess(MemoryInfo(None, 40, 90))]) == (40, 90)                       # no private bytes: the rss


def test_a_process_without_a_peak_pagefile_leaves_the_upper_bound_unknown_and_a_process_that_is_gone_is_skipped():
    psutil = pytest.importorskip('psutil')
    assert run_s3._tree_memory([FakeProcess(MemoryInfoWithoutPeak(100, 200))]) == (100, None)              # not Windows
    assert run_s3._tree_memory([FakeProcess(MemoryInfo(100, 200, 300)), FakeProcess(MemoryInfoWithoutPeak(5, 6))]) == (105, None)
    gone = FakeProcess(error=psutil.NoSuchProcess(4242))
    assert run_s3._tree_memory([FakeProcess(MemoryInfo(100, 200, 300)), gone]) == (100, 300)
    assert run_s3._tree_memory([gone]) == (0, None)                                                        # nothing could be read


def test_the_real_tree_memory_has_an_upper_bound_of_at_least_the_private_bytes_on_windows():
    private, upper = run_s3._tree_memory()
    assert private == run_s3._tree_private_bytes() or abs(private - run_s3._tree_private_bytes()) < 64 * 2**20
    assert upper is None or upper >= private - 64 * 2**20                       # a lifetime peak is never below the present


def test_a_scope_records_the_largest_upper_bound_beside_its_peak_and_scopes_nest():
    pairs = iter([(10, 100), (50, 200), (20, 300), (5, 400)])
    sampler = run_s3._Sampler(interval=1e9, measure=lambda: next(pairs))
    with sampler.scope() as outer:
        with sampler.scope() as inner:
            pass
    assert (inner.peak, inner.peak_upper) == (50, 300)                                                     # the samples at its two edges
    assert (outer.peak, outer.peak_upper) == (50, 400) and (sampler.peak, sampler.peak_upper) == (50, 400)


def test_a_measure_that_gives_one_number_leaves_the_upper_bound_of_every_scope_unknown():
    values = iter([10, 20, 5])
    sampler = run_s3._Sampler(interval=1e9, measure=lambda: next(values))
    with sampler.scope() as scope:
        pass
    assert scope.peak == 20 and scope.peak_upper is None and sampler.peak_upper is None
    mixed = iter([(3, None), (4, 9), (2, None)])
    sampler = run_s3._Sampler(interval=1e9, measure=lambda: next(mixed))
    with sampler.scope() as scope:
        pass
    assert scope.peak == 4 and scope.peak_upper == 9                                                 # a sample without it adds nothing


def test_the_sampler_thread_takes_the_pairs_too():
    with run_s3._Sampler(interval=0.01, measure=lambda: (2**30, 3 * 2**30)) as sampler:
        time.sleep(0.1)
    assert sampler.samples >= 3 and (sampler.peak, sampler.peak_upper) == (2**30, 3 * 2**30)


def test_every_place_that_records_a_peak_records_its_upper_bound_beside_it(e2e):
    manifest, ledger = run_json(e2e), read_json(e2e.run / 'STAGES.json')

    def check(record, where):
        assert 'peak_private_upper_gib' in record, where
        peak, upper = record['peak_private_gib'], record['peak_private_upper_gib']
        assert peak is not None and (upper is None or upper >= peak - 0.002), (where, peak, upper)       # the bound is never below the peak
    for stage in STAGES:
        check(manifest[f'{stage}_result'], f'{stage} result')
        check(ledger['stages'][stage], f'{stage} record')
    fit = manifest['fit_result']
    for model in ('tier', 'b0', 'b1', 'b2', 'dev'):
        check(fit[model], model)
    for model in ('b0', 'b1'):
        for key, unit in fit[model]['heads'].items():
            check(unit, f'{model} {key}')
    check(manifest['compose_result'], 'compose')
    check(read_json(e2e.run / 'gate' / 'GATE.json'), 'gate')
    check(ledger['calls'][0], 'call')
    s3 = manifest['s3_result']
    assert set(s3['stage_peak_private_upper_gib']) == set(STAGES) and 'peak_private_upper_gib' in s3
    assert s3['peak_private_upper_gib'] is None or s3['peak_private_upper_gib'] >= s3['peak_private_gib'] - 0.002
    for line in (e2e.run / 'fit' / 'PROGRESS.jsonl').read_text(encoding='utf-8').splitlines():
        entry = json.loads(line)
        if 'peak_private_gib' in entry:
            check(entry, entry['unit'])
    for kind, name in run_s3._head_names():
        check(baselines_b1.read_head_record(e2e.run / 'fit' / 'b1_heads' / f'{kind}.{name}'), f'record {kind}.{name}')
    if run_s3.psutil is not None and hasattr(run_s3.psutil.Process().memory_info(), 'peak_pagefile'):         # Windows: the bound is there
        assert s3['peak_private_upper_gib'] is not None and all(v is not None for v in s3['stage_peak_private_upper_gib'].values())


def test_the_guard_counts_and_times_the_e6_check_and_lets_a_refusal_through(fake_processes):
    guard = run_s3._Guard()
    guard()
    guard()
    assert guard.calls == 2 and guard.seconds >= 0
    before = guard.snapshot()
    fake_processes([synth.STRATA_SERVER])
    with pytest.raises(run_s1.StrataRunning):
        guard()
    delta = guard.since(before)
    assert delta['check_calls'] == 1 and delta['check_s_mean'] is not None and guard.calls == 3


def test_the_guard_looks_up_the_e6_check_at_the_time_of_the_call(monkeypatch):
    seen = []
    monkeypatch.setattr(run_s1, 'require_no_strata', lambda: seen.append(1))
    run_s3._Guard()()
    assert seen == [1]


def test_the_guard_remembers_the_longest_of_its_checks(monkeypatch):
    guard = run_s3._Guard()
    monkeypatch.setattr(run_s1, 'require_no_strata', lambda: time.sleep(0.05))
    guard()
    monkeypatch.setattr(run_s1, 'require_no_strata', lambda: None)
    guard()
    assert guard.calls == 2 and 0.04 <= guard.longest <= guard.seconds                    # (a sleep can end a tick early on Windows)


def test_the_tick_timer_runs_the_guard_at_every_call_and_times_the_units_between_two_calls(monkeypatch):
    seen = []
    monkeypatch.setattr(run_s1, 'require_no_strata', lambda: seen.append(1))
    ctx = run_s3.Context(
        run=Path('x'), s1=None, s2=None, prices={}, rules=None, hashes={}, sampler=run_s3._Sampler(interval=1e9, measure=lambda: 0)
    )
    timer = run_s3._TickTimer(ctx)
    for _ in range(4):
        timer()
    timer.finish()
    assert len(seen) == 4 and len(timer.segments) == 4 and all(s['wall_s'] >= 0 for s in timer.segments)


def test_a_b0_fit_that_fails_or_is_refused_leaves_no_timing_scope_open(tmp_path, monkeypatch):
    monkeypatch.setattr(run_s1, 'require_no_strata', lambda: None)
    ctx = run_s3.Context(
        run=tmp_path, s1=None, s2=None, prices={}, rules=None, hashes={}, sampler=run_s3._Sampler(interval=1e9, measure=lambda: 0)
    )

    held = []                                                # the timer stays alive: a timer that is dropped closes its scope by itself

    def failing_fit(cls, s1, s2, labels, means, roles=None, *, check=None):
        held.append(check)
        check()                                              # the setup
        check()                                              # the first head
        raise RuntimeError('a head failed')

    monkeypatch.setattr(baselines_b0.B0Heads, 'fit', classmethod(failing_fit))
    with pytest.raises(RuntimeError, match='a head failed'):
        run_s3._fit_b0(ctx, tmp_path, run_s3._ProgressLog(tmp_path / 'PROGRESS.jsonl'), None, np.zeros(3))
    assert held and ctx.sampler._scopes == [] and ctx.guard.calls == 2              # the unit that was running was closed too
    assert len(held[0].segments) == 2                                                  # the setup and the first head


# ---- units: the smoke selection ---------------------------------------------------------------------------------------------------------
def test_the_smoke_selection_takes_n_fit_a_fifth_of_n_stop_and_three_tenths_of_n_dev_as_ceilings(world):
    s1 = run_s1.load_s1(world.s1, verify=False)
    role = np.asarray(s1.array('matches/role'))
    for n, fit, stop, dev in ((1, 1, 1, 1), (4, 4, 1, 2), (10, 10, 2, 3), (12, 12, 3, 4), (6, 6, 2, 2)):
        sel = run_s3.select_smoke(s1, n)
        assert {k: len(v) for k, v in sel['matches'].items()} == {'FIT': fit, 'STOP': stop, 'DEV': dev}, n
        assert sel['asked'] == {'FIT': fit, 'STOP': stop, 'DEV': dev}
        for name, rows in sel['matches'].items():
            assert rows == sorted(rows) and len(set(rows)) == len(rows)
            assert all(type(r) is int for r in rows)                                       # match rows, plain integers, never keys
            assert {C.ROLES[r] for r in role[rows].tolist()} == {name}
    release(s1)


def test_the_asked_counts_are_the_exact_ceilings_of_n_a_fifth_of_n_and_three_tenths_of_n(world):
    from fractions import Fraction
    s1 = run_s1.load_s1(world.s1, verify=False)
    role = np.asarray(s1.array('matches/role'))
    for n in list(range(1, 60)) + [999, 1000, 1001, 1667, 2000]:
        asked = run_s3.select_smoke(s1, n)['asked']
        assert asked == {'FIT': n, 'STOP': math.ceil(Fraction(1, 5) * n), 'DEV': math.ceil(Fraction(3, 10) * n)}, n
    assert (role == C.ROLES.index('FIT')).sum() == 16  # asked more than there is: all
    release(s1)


def test_the_smoke_selection_is_seeded_nested_and_fixed(world):
    s1 = run_s1.load_s1(world.s1, verify=False)
    a, b = run_s3.select_smoke(s1, 8), run_s3.select_smoke(s1, 8)
    assert a == b and a['seed'] == 20261021 and a['selection_sha256'] == runlog.sha256_json(a['matches'])
    small, big = run_s3.select_smoke(s1, 4), run_s3.select_smoke(s1, 12)
    for name in ('FIT', 'STOP', 'DEV'):
        assert set(small['matches'][name]) <= set(big['matches'][name])                    # a smaller n lies inside a larger one
    other = run_s3.select_smoke(s1, 8, seed=1)
    assert other['matches'] != a['matches'] and other['selection_sha256'] != a['selection_sha256']
    release(s1)


def test_a_role_with_fewer_matches_than_asked_gives_all_of_them_and_a_missing_role_is_refused(world):
    s1 = run_s1.load_s1(world.s1, verify=False)
    sel = run_s3.select_smoke(s1, 1000)
    role = np.asarray(s1.array('matches/role'))
    assert sel['matches']['FIT'] == np.flatnonzero(role == C.ROLES.index('FIT')).tolist() and sel['asked']['FIT'] == 1000
    assert sel['matches']['DEV'] == np.flatnonzero(role == C.ROLES.index('DEV')).tolist()
    for bad in (0, -3, True, 2.5):
        with pytest.raises(run_s3.RunRefused, match='--smoke'):
            run_s3.select_smoke(s1, bad)
    release(s1)


def test_a_smoke_run_needs_all_three_roles(toy_runs):
    s1 = run_s1.load_s1(toy_runs.s1, verify=False)

    def dev_read_as_fit(name):
        values = np.asarray(s1.array(name))
        return np.where(values == C.ROLES.index('DEV'), C.ROLES.index('FIT'), values)

    view = types.SimpleNamespace(array=dev_read_as_fit)
    with pytest.raises(run_s3.RunRefused, match='no match with the role DEV'):
        run_s3.select_smoke(view, 3)
    release(s1)


def test_the_subset_store_shows_the_matches_outside_the_selection_as_unused_and_nothing_else_changes(world):
    s1 = run_s1.load_s1(world.s1, verify=False)
    sel = run_s3.select_smoke(s1, 6)
    view = run_s3.SubsetStore(s1, sel)
    base, shown = np.asarray(s1.array('matches/role')), np.asarray(view.array('matches/role'))
    chosen = np.zeros(len(base), dtype=bool)
    for rows in sel['matches'].values():
        chosen[rows] = True
    assert np.array_equal(shown[chosen], base[chosen])
    assert (shown[~chosen] == C.ROLES.index('UNUSED')).all() and (base[~chosen] != C.ROLES.index('UNUSED')).any()
    assert not shown.flags.writeable and (np.asarray(s1.array('matches/role')) == base).all()                    # the store is not changed
    assert np.array_equal(np.asarray(view.array('units/f')), np.asarray(s1.array('units/f')))
    assert view.manifest is s1.manifest and view.root == s1.root                                                  # the rest is the store
    assert runlog.sha256_json(view.manifest['files']) == runlog.sha256_json(s1.manifest['files'])
    import copy
    assert copy.copy(view).array('matches/role') is view.array('matches/role')                                    # a copy does not recurse
    with pytest.raises(AttributeError):
        view._nope
    release(s1)


def test_the_s3_code_sees_only_the_smoke_subset_through_the_subset_store(world):
    """The labels, the tier law, the rows of the heads and of the composition select matches by role alone, so the view is the mask."""
    s1 = run_s1.load_s1(world.s1, verify=False)
    s2 = flowprice._open_s2(world.s2, False)
    sel = run_s3.select_smoke(s1, 6)
    view = run_s3.SubsetStore(s1, sel)
    keys = np.asarray(s1.array('matches/key')).tolist()
    wanted = {name: {keys[r].decode('ascii') for r in rows} for name, rows in sel['matches'].items()}
    asked = []

    def records(key, patch):
        asked.append(key)
        return random_value_record(key, patch)
    levels = value.kill_levels(view, run_s3.LABEL_ROLES, workers=1, records=records)
    assert set(asked) == set().union(*wanted.values()) and len(asked) == 10  # 6 + 2 + 2 matches, no others
    prices = dict(TOY_PRICES)
    uv = value.unit_values(view, prices, rulepay.P3_RULES, levels)
    unit_match = np.asarray(s1.array('units/match'))
    chosen = {r for rows in sel['matches'].values() for r in rows}
    assert np.array_equal(uv.valued, np.isin(unit_match, sorted(chosen)))
    acc = value.accounting(view, uv, prices, rulepay.P3_RULES, levels)
    assert acc['n_valued_matches'] == 10 and acc['max_abs_error'] == 0.0
    full = value.unit_values(
        s1, prices, rulepay.P3_RULES, value.kill_levels(s1, run_s3.LABEL_ROLES, workers=1, records=random_value_record)
    )
    assert full.valued.sum() > uv.valued.sum() and np.array_equal(
        full.f0_unit[uv.valued], uv.f0_unit[uv.valued]
    )  # the same labels, fewer units
    rows = compose.row_table(view, s2, ('DEV',))
    assert set(np.unique(rows.match).tolist()) == set(sel['matches']['DEV'])
    assert towertier.tier_rows(view, ['FIT']).n_matches == 6
    frame_match = np.asarray(s1.array('frame_rows/match'))
    assert set(frame_match[features.split_rows(view, 'FIT')].tolist()) == set(sel['matches']['FIT'])
    assert set(np.asarray(s2.array('views/match'))[features.split_views(view, s2, 'STOP')].tolist()) == set(sel['matches']['STOP'])
    release(s1, s2)


# ---- units: the decision record and the s3a files ---------------------------------------------------------------------------------------
def test_the_decision_record_gives_the_two_hashes_of_decisions_s3_prices(tmp_path):
    spec = write_spec(tmp_path / 'spec.json', 'a' * 64, 'b' * 64)
    got = run_s3.read_decisions(spec)
    assert got == {'prices_sha256': 'a' * 64, 'rules_sha256': 'b' * 64, 'spec_sha256': sha_of(spec)}


@pytest.mark.parametrize(
    'doc',
    [
        {},
        {'decisions': {}},
        {'decisions': {'S3_prices': {}}},
        {'decisions': {'S3_prices': {'prices': {'sha256': 'a' * 64}}}},
        {'decisions': {'S3_prices': {'prices': {'sha256': 'xyz'}, 'rules': {'sha256': 'b' * 64}}}},
        {'decisions': {'S3_prices': {'prices': {'sha256': 'A' * 64}, 'rules': {'sha256': 'b' * 64}}}},
        [1, 2],
    ],
)
def test_a_decision_record_without_the_two_hashes_is_refused(tmp_path, doc):
    path = tmp_path / 'spec.json'
    path.write_text(json.dumps(doc), encoding='utf-8')
    with pytest.raises(run_s3.RunRefused, match='S3_prices|SHA-256'):
        run_s3.read_decisions(path)


def test_a_decision_record_that_is_not_there_or_not_json_is_refused(tmp_path):
    with pytest.raises(run_s3.RunRefused, match='cannot be read'):
        run_s3.read_decisions(tmp_path / 'nope.json')
    bad = tmp_path / 'bad.json'
    bad.write_text('{"decisions": ', encoding='utf-8')
    with pytest.raises(run_s3.RunRefused, match='S3_prices'):
        run_s3.read_decisions(bad)


def test_the_s3a_files_are_accepted_when_their_hashes_are_those_of_the_record(tmp_path):
    files = write_s3a(tmp_path)
    got = run_s3.check_s3a(files.run, run_s3.read_decisions(files.spec))
    assert got['prices_sha256'] == sha_of(files.prices) and got['rules_sha256'] == sha_of(files.rules)
    assert got['prices'] == files.prices and got['rules'] == files.rules and got['run'] == files.run.name


@pytest.mark.parametrize('which', ['prices', 'rules'])
def test_an_s3a_file_whose_sha256_differs_from_the_decision_record_is_refused(tmp_path, which):
    files = write_s3a(tmp_path)
    path = files.prices if which == 'prices' else files.rules
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(run_s3.RunRefused) as raised:
        run_s3.check_s3a(files.run, run_s3.read_decisions(files.spec))
    assert path.name in str(raised.value) and 'decisions.S3_prices' in str(raised.value)


def test_a_missing_s3a_file_is_refused(tmp_path):
    files = write_s3a(tmp_path)
    files.rules.unlink()
    with pytest.raises(run_s3.RunRefused, match='RULE_PAYOUTS'):
        run_s3.check_s3a(files.run, run_s3.read_decisions(files.spec))


# ---- units: the lock -------------------------------------------------------------------------------------------------------------------
def test_the_lock_is_exclusive_and_released_and_never_deleted_for_the_user(tmp_path):
    run = tmp_path / '20261021_c_ept_s3_1'
    lock = run_s3._acquire_lock(run)
    assert lock == run / 'S3.lock' and json.loads(lock.read_text(encoding='utf-8'))['pid'] > 0
    with pytest.raises(run_s3.RunLocked) as raised:
        run_s3._acquire_lock(run)
    assert 'S3.lock' in str(raised.value) and 'delete that file yourself' in str(raised.value) and lock.is_file()
    run_s3._release_lock(lock)
    assert not lock.exists()
    run_s3._release_lock(lock)                                                              # a second release is not an error
    run_s3._release_lock(run_s3._acquire_lock(run))


# ---- units: the model folders ------------------------------------------------------------------------------------------------------------
def test_a_model_folder_counts_as_saved_only_when_its_json_is_there(tmp_path):
    messages = []
    ctx = run_s3.Context(run=tmp_path, s1=None, s2=None, prices={}, rules=None, hashes={}, out=lambda m, flush=True: messages.append(m))
    root = tmp_path / 'fit'
    root.mkdir()
    assert run_s3._model_saved(ctx, root / 'tier', 'tower_tier.json', root) is False                  # nothing there
    (root / 'tier').mkdir()
    (root / 'tier' / 'tower_tier.npz').write_bytes(b'npz without its json')  # a save that was cut after the array file
    assert run_s3._model_saved(ctx, root / 'tier', 'tower_tier.json', root) is False
    assert not (root / 'tier').exists() and 'incomplete save' in messages[-1]
    (root / 'b1' / 'models').mkdir(parents=True)
    (root / 'b1' / 'models' / 'masked.M2Rate.c00.txt').write_bytes(b'a booster')  # model files of B1, the JSON is written last
    assert run_s3._model_saved(ctx, root / 'b1', 'b1_heads.json', root) is False and not (root / 'b1').exists()
    (root / 'b0').mkdir()
    (root / 'b0' / 'b0_heads.npz').write_bytes(b'x')
    (root / 'b0' / 'b0_heads.json').write_bytes(b'{}')
    assert run_s3._model_saved(ctx, root / 'b0', 'b0_heads.json', root) is True and (root / 'b0' / 'b0_heads.npz').exists()


def test_a_folder_outside_the_run_is_never_removed(tmp_path):
    inside, outside = tmp_path / 'run', tmp_path / 'elsewhere'
    inside.mkdir()
    outside.mkdir()
    (outside / 'keep.txt').write_text('keep', encoding='utf-8')
    with pytest.raises(RuntimeError, match='not removed'):
        run_s3._remove_tree(outside, inside)
    with pytest.raises(RuntimeError, match='not removed'):
        run_s3._remove_tree(inside, inside)
    assert (outside / 'keep.txt').exists() and inside.exists()


def test_a_stage_needs_the_stages_before_it_to_be_complete_or_asked_for_too():
    problem = run_s3.prerequisite_problem
    assert problem(STAGES, set()) is None and problem(('levels',), set()) is None and problem(('levels', 'labels'), set()) is None
    assert 'fit needs the stage levels' in problem(('fit',), set()) and 'labels needs the stage levels' in problem(('labels',), set())
    assert 'gate needs the stage compose' in problem(('levels', 'labels', 'fit', 'gate'), set())
    assert problem(('fit', 'compose'), {'levels', 'labels'}) is None and 'compose needs the stage fit' in problem(
        ('compose',), {'levels', 'labels'}
    )


# ---- the command line with fake stages -------------------------------------------------------------------------------------------------
def test_the_stages_run_in_the_order_levels_labels_fit_compose_gate_and_each_is_recorded(tmp_path, toy_runs, fakes, capsys):
    assert toy_cli(tmp_path, toy_runs) == 0
    assert fakes.calls == ['levels', 'labels', 'fit', 'compose', 'gate']
    run = only_run(tmp_path)
    assert re.fullmatch(r'\d{8}_c_ept_s3_1', run.name) and run.parent == tmp_path / 'runs'
    ledger, manifest = read_json(run / 'STAGES.json'), read_json(run / 'RUN.json')
    assert ledger['schema'] == 's3_stages_v1' and set(ledger['stages']) == set(STAGES)
    for stage in STAGES:
        rec = ledger['stages'][stage]
        assert rec['complete'] is True and rec['inputs_sha256'] == runlog.sha256_json(rec['inputs'])
        assert sorted(rec['outputs']) == sorted(run_s3.STAGE_OUTPUTS[stage]) and rec['wall_s'] >= 0 and 'peak_private_gib' in rec
        assert f'{stage}_result' in manifest
    assert ledger['stages']['labels']['inputs']['kill_levels_sha256'] == ledger['stages']['levels']['outputs']['levels/kill_levels.npy']
    assert ledger['stages']['fit']['inputs']['labels_store'] == ledger['stages']['labels']['outputs']['labels/store/']
    assert ledger['stages']['compose']['inputs']['fit']['b0'] == ledger['stages']['fit']['outputs']['fit/b0/b0_heads.json']
    assert ledger['stages']['gate']['inputs']['compose']['b1'] == ledger['stages']['compose']['outputs']['compose/b1_dev.npz']
    assert len(ledger['calls']) == 1 and ledger['calls'][0]['stages_run'] == list(STAGES) and ledger['calls'][0]['stages_skipped'] == []
    assert manifest['stage'] == 's3' and manifest['run'] == run.name and manifest['s3_result']['complete'] is True
    assert catalog_rows(tmp_path) == [(run.name, 's3')] and not (run / 'S3.lock').exists()
    out = capsys.readouterr()
    assert str(run) in out.out and KEY_PREFIX not in out.out + out.err


def test_stages_given_in_another_order_run_in_the_order_of_the_plan(tmp_path, toy_runs, fakes):
    assert toy_cli(tmp_path, toy_runs, '--stages', 'gate', 'compose', 'fit', 'labels', 'levels') == 0
    assert fakes.calls == list(STAGES)


@pytest.mark.parametrize('stages, text', [(['fit'], 'needs the stage levels'), (['levels', 'fit'], 'fit needs the stage labels'),
                                          (['gate'], 'needs the stage levels'), (['nope'], 'unknown stage'),
                                          (['levels', 'levels'], 'named twice')])
def test_a_new_run_with_stages_that_cannot_run_is_refused_before_a_run_folder_is_made(tmp_path, toy_runs, fakes, capsys, stages, text):
    assert toy_cli(tmp_path, toy_runs, '--stages', *stages) == 2
    assert text in capsys.readouterr().err
    assert not (tmp_path / 'runs').exists() and fakes.calls == [] and catalog_rows(tmp_path) == []


@pytest.mark.parametrize('flag, value', [('--workers', '0'), ('--workers', '62'), ('--workers', '-3'), ('--smoke', '0'), ('--smoke', '-1')])
def test_bad_numbers_are_refused_before_a_run_folder_is_made(tmp_path, toy_runs, fakes, capsys, flag, value):
    assert toy_cli(tmp_path, toy_runs, flag, value) == 2
    assert flag in capsys.readouterr().err and not (tmp_path / 'runs').exists() and fakes.calls == []


def test_a_run_may_be_made_in_steps_and_is_registered_only_when_all_five_stages_are_complete(tmp_path, toy_runs, fakes, capsys):
    assert toy_cli(tmp_path, toy_runs, '--stages', 'levels', 'labels') == 0
    run = only_run(tmp_path)
    assert fakes.calls == ['levels', 'labels'] and catalog_rows(tmp_path) == []
    assert 'not every stage is complete yet' in capsys.readouterr().out
    manifest = read_json(run / 'RUN.json')
    assert 'levels_result' in manifest and 'fit_result' not in manifest and 's3_result' not in manifest
    assert toy_cli(tmp_path, toy_runs, '--resume', str(run), '--stages', 'fit', 'compose', 'gate') == 0
    assert fakes.calls == list(STAGES) and catalog_rows(tmp_path) == [(run.name, 's3')]
    assert read_json(run / 'RUN.json')['s3_result']['calls'] == 2


def test_a_resume_skips_the_stages_that_are_complete(tmp_path, toy_runs, fakes, capsys):
    assert toy_cli(tmp_path, toy_runs, '--stages', 'levels', 'labels') == 0
    run = only_run(tmp_path)
    before = (run / 'levels' / 'kill_levels.npy').read_bytes()
    capsys.readouterr()
    assert toy_cli(tmp_path, toy_runs, '--resume', str(run)) == 0
    assert fakes.calls == ['levels', 'labels', 'fit', 'compose', 'gate']                    # levels and labels did not run again
    out = capsys.readouterr().out
    assert 'stage levels: complete in an earlier call, skipped' in out and 'stage labels: complete in an earlier call, skipped' in out
    ledger = read_json(run / 'STAGES.json')
    assert [c['stages_run'] for c in ledger['calls']] == [['levels', 'labels'], ['fit', 'compose', 'gate']]
    assert [c['stages_skipped'] for c in ledger['calls']] == [[], ['levels', 'labels']] and [c['resumed'] for c in ledger['calls']] == [
        False,
        True,
    ]
    assert (run / 'levels' / 'kill_levels.npy').read_bytes() == before
    again = fakes.calls[:]
    assert toy_cli(tmp_path, toy_runs, '--resume', str(run)) == 0 and fakes.calls == again          # everything is complete: nothing runs


def test_a_resume_needs_a_run_folder_made_by_this_runner_and_refuses_one_that_is_not(tmp_path, toy_runs, fakes, capsys):
    for bad in (tmp_path / 'nowhere', tmp_path):
        assert toy_cli(tmp_path, toy_runs, '--resume', str(bad)) == 2
        assert 'RUN.json' in capsys.readouterr().err
    other = tmp_path / '20261021_c_ept_s2_9'
    other.mkdir()
    (other / 'RUN.json').write_text(json.dumps({'stage': 's2'}), encoding='utf-8')
    assert toy_cli(tmp_path, toy_runs, '--resume', str(other)) == 2 and 'not an S3 run folder' in capsys.readouterr().err
    assert fakes.calls == []


def test_a_resume_refuses_other_stores_other_s3a_files_and_another_smoke_selection(tmp_path, toy_runs, fakes, capsys):
    assert toy_cli(tmp_path, toy_runs, '--stages', 'levels') == 0
    run = only_run(tmp_path)
    capsys.readouterr()
    other1 = s3_toys.build_toy_s1(tmp_path / 'other' / '20261020_c_ept_s1_2' / 's1', s3_toys.TOY_MATCHES[:5])
    other2 = s3_toys.build_toy_s2(tmp_path / 'other' / '20261021_c_ept_s2_2' / 's2', other1)
    release(other1, other2)
    cases = [('another S1 store', dict(s1=tmp_path / 'other' / '20261020_c_ept_s1_2'), 'S1 store'),
             ('another S2 store', dict(s2=tmp_path / 'other' / '20261021_c_ept_s2_2'), 'S2 store')]
    for name, change, text in cases:
        args = {'s1': toy_runs.s1, 's2': toy_runs.s2, 's3a': toy_runs.s3a, 'spec': toy_runs.spec, **change}
        assert run_s3.main(cli_args(tmp_path, args['s1'], args['s2'], args['s3a'], args['spec'], '--resume', str(run))) == 2, name
        assert text in capsys.readouterr().err
    bump = write_s3a(tmp_path / 'other_s3a', bump=1.0)  # approved too, but not the prices this run was made with
    assert run_s3.main(cli_args(tmp_path, toy_runs.s1, toy_runs.s2, bump.run, bump.spec, '--resume', str(run))) == 2
    assert 'PRICES.json' in capsys.readouterr().err
    assert toy_cli(tmp_path, toy_runs, '--resume', str(run), '--smoke', '3') == 2
    assert 'smoke selection' in capsys.readouterr().err
    assert fakes.calls == ['levels'] and (run / 'STAGES.json').is_file() and not (run / 'S3.lock').exists() and catalog_rows(tmp_path) == []
    assert toy_cli(tmp_path, toy_runs, '--resume', str(run)) == 0 and fakes.calls[1:] == ['labels', 'fit', 'compose', 'gate']


def test_a_resume_of_a_smoke_run_needs_the_same_smoke_number(tmp_path, toy_runs, fakes, capsys):
    assert toy_cli(tmp_path, toy_runs, '--stages', 'levels', '--smoke', '3') == 0
    run = only_run(tmp_path)
    capsys.readouterr()
    for extra in ([], ['--smoke', '2']):
        assert toy_cli(tmp_path, toy_runs, '--resume', str(run), *extra) == 2 and 'smoke selection' in capsys.readouterr().err
    assert toy_cli(tmp_path, toy_runs, '--resume', str(run), '--smoke', '3') == 0 and fakes.calls == list(STAGES)


def test_a_changed_output_of_a_complete_stage_is_refused_and_nothing_runs(tmp_path, toy_runs, fakes, capsys):
    assert toy_cli(tmp_path, toy_runs, '--stages', 'levels', 'labels') == 0
    run = only_run(tmp_path)
    path = run / 'levels' / 'kill_levels.npy'
    path.write_bytes(path.read_bytes() + b'\0')
    capsys.readouterr()
    assert toy_cli(tmp_path, toy_runs, '--resume', str(run)) == 2
    err = capsys.readouterr().err
    assert 'complete stage levels' in err and 'kill_levels.npy' in err and 'start a new run folder' in err
    assert fakes.calls == ['levels', 'labels'] and catalog_rows(tmp_path) == [] and not (run / 'S3.lock').exists()


def test_a_changed_input_of_a_complete_stage_is_refused_by_its_input_hash(tmp_path, toy_runs, fakes, capsys):
    """The output of the stage before changed and its record was brought along: the stage that was made from the old file is refused."""
    assert toy_cli(tmp_path, toy_runs, '--stages', 'levels', 'labels') == 0
    run = only_run(tmp_path)
    path = run / 'levels' / 'kill_levels.npy'
    path.write_bytes(path.read_bytes() + b'\0')
    ledger = read_json(run / 'STAGES.json')
    ledger['stages']['levels']['outputs']['levels/kill_levels.npy'] = store_core.sha(path)
    (run / 'STAGES.json').write_text(json.dumps(ledger), encoding='utf-8')
    capsys.readouterr()
    assert toy_cli(tmp_path, toy_runs, '--resume', str(run)) == 2
    err = capsys.readouterr().err
    assert 'inputs of the complete stage labels' in err and 'kill_levels_sha256' in err
    assert fakes.calls == ['levels', 'labels']


def test_a_changed_output_of_a_later_stage_is_refused_too(tmp_path, toy_runs, fakes, capsys):
    assert toy_cli(tmp_path, toy_runs, '--stages', 'levels', 'labels') == 0
    run = only_run(tmp_path)
    (run / 'labels' / 'ACCOUNTING.json').write_text('changed', encoding='utf-8')
    capsys.readouterr()
    assert toy_cli(tmp_path, toy_runs, '--resume', str(run)) == 2
    err = capsys.readouterr().err
    assert 'complete stage labels' in err and 'ACCOUNTING.json' in err


def test_a_missing_output_of_a_complete_stage_is_refused(tmp_path, toy_runs, fakes, capsys):
    assert toy_cli(tmp_path, toy_runs, '--stages', 'levels', 'labels') == 0
    run = only_run(tmp_path)
    (run / 'levels' / 'LEVELS.json').unlink()
    capsys.readouterr()
    assert toy_cli(tmp_path, toy_runs, '--resume', str(run)) == 2 and 'LEVELS.json' in capsys.readouterr().err


def test_a_stage_that_is_complete_after_one_that_is_not_is_an_inconsistent_record(tmp_path, toy_runs, fakes, capsys):
    assert toy_cli(tmp_path, toy_runs, '--stages', 'levels', 'labels') == 0
    run = only_run(tmp_path)
    ledger = read_json(run / 'STAGES.json')
    ledger['stages']['levels']['complete'] = False
    (run / 'STAGES.json').write_text(json.dumps(ledger), encoding='utf-8')
    capsys.readouterr()
    assert toy_cli(tmp_path, toy_runs, '--resume', str(run)) == 2 and 'not consistent' in capsys.readouterr().err


def test_an_s3a_file_that_is_not_the_approved_one_is_refused_before_a_run_folder_is_made(tmp_path, toy_runs, fakes, capsys):
    other = write_s3a(tmp_path / 'b')
    other.prices.write_bytes(other.prices.read_bytes() + b' ')                              # the file changed after the record was made
    assert run_s3.main(cli_args(tmp_path, toy_runs.s1, toy_runs.s2, other.run, other.spec)) == 2
    err = capsys.readouterr().err
    assert 'PRICES.json' in err and 'decisions.S3_prices' in err and 'SHA-256' in err
    assert not (tmp_path / 'runs').exists() and fakes.calls == [] and catalog_rows(tmp_path) == []
    third = write_s3a(tmp_path / 'c')
    third.rules.write_bytes(third.rules.read_bytes() + b' ')
    assert run_s3.main(cli_args(tmp_path, toy_runs.s1, toy_runs.s2, third.run, third.spec)) == 2
    assert 'RULE_PAYOUTS.json' in capsys.readouterr().err and not (tmp_path / 'runs').exists()


def test_a_decision_record_without_decisions_s3_prices_refuses_the_run(tmp_path, toy_runs, fakes, capsys):
    spec = tmp_path / 'SPEC.json'
    spec.write_text(json.dumps({'decisions': {'S3_plan': {}}}), encoding='utf-8')
    assert run_s3.main(cli_args(tmp_path, toy_runs.s1, toy_runs.s2, toy_runs.s3a, spec)) == 2
    assert 'S3_prices' in capsys.readouterr().err and not (tmp_path / 'runs').exists()


def test_prices_or_rules_that_cannot_be_used_refuse_the_run_before_a_run_folder_is_made(tmp_path, toy_runs, fakes, capsys):
    files = write_s3a(tmp_path / 'c')
    prices = json.loads(files.prices.read_text(encoding='utf-8'))
    del prices['plate']                                                                     # approved by hash, but a group has no price
    files.prices.write_text(json.dumps(prices), encoding='utf-8')
    write_spec(files.spec, sha_of(files.prices), sha_of(files.rules))
    assert run_s3.main(cli_args(tmp_path, toy_runs.s1, toy_runs.s2, files.run, files.spec)) == 2
    assert 'plate' in capsys.readouterr().err and not (tmp_path / 'runs').exists()
    bad = write_s3a(tmp_path / 'd')
    doc = rules_document()
    doc['rules']['assist_bounty'][0]['status'] = 'documented'                              # not verified: f0 uses verified rules only
    bad.rules.write_text(json.dumps(doc), encoding='utf-8')
    write_spec(bad.spec, sha_of(bad.prices), sha_of(bad.rules))
    assert run_s3.main(cli_args(tmp_path, toy_runs.s1, toy_runs.s2, bad.run, bad.spec)) == 2
    assert 'verified' in capsys.readouterr().err and not (tmp_path / 'runs').exists() and fakes.calls == []


def test_a_missing_or_incomplete_store_is_refused_before_a_run_folder_is_made(tmp_path, toy_runs, fakes, capsys):
    assert (
        run_s3.main(cli_args(tmp_path, tmp_path / 'nope', toy_runs.s2, toy_runs.s3a, toy_runs.spec)) == 2
        and 'S1' in capsys.readouterr().err
    )
    assert (
        run_s3.main(cli_args(tmp_path, toy_runs.s1, tmp_path / 'nope', toy_runs.s3a, toy_runs.spec)) == 2
        and 'S2' in capsys.readouterr().err
    )
    half = s3_toys.build_toy_s1(tmp_path / 'half' / 's1', s3_toys.TOY_MATCHES)
    half.manifest['complete'] = False
    half.save()
    release(half)
    assert run_s3.main(cli_args(tmp_path, tmp_path / 'half', toy_runs.s2, toy_runs.s3a, toy_runs.spec)) == 2
    assert 'not complete' in capsys.readouterr().err and not (tmp_path / 'runs').exists() and fakes.calls == []


def test_verify_checks_the_files_of_both_stores_first(tmp_path, toy_runs, fakes, capsys):
    assert toy_cli(tmp_path, toy_runs, '--verify') == 0
    s1 = s3_toys.build_toy_s1(tmp_path / 'bad' / 's1', s3_toys.TOY_MATCHES)
    release(s1)
    victim = tmp_path / 'bad' / 's1' / 'kills' / 'killer.npy'
    victim.write_bytes(victim.read_bytes() + b'\0')
    assert run_s3.main(cli_args(tmp_path / 'again', tmp_path / 'bad', toy_runs.s2, toy_runs.s3a, toy_runs.spec, '--verify')) == 2
    assert 'does not match STORE.json' in capsys.readouterr().err and not (tmp_path / 'again' / 'runs').exists()


# ---- E6 --------------------------------------------------------------------------------------------------------------------------------
def test_a_running_model_server_stops_the_run_before_a_store_is_opened_and_a_run_folder_is_made(
    tmp_path, toy_runs, fakes, fake_processes, capsys
):
    fake_processes([synth.STRATA_SERVER, synth.STRATA_BRIDGE])
    assert toy_cli(tmp_path, toy_runs) == 2
    err = capsys.readouterr().err
    assert 'E6' in err and 'does not stop' in err and f'pid {synth.STRATA_SERVER[0]}' in err and 'strata_worker_mcp' not in err
    assert not (tmp_path / 'runs').exists() and fakes.calls == [] and catalog_rows(tmp_path) == []


def test_e6_is_checked_before_every_stage(tmp_path, toy_runs, monkeypatch):
    order = []
    real = run_s1.require_no_strata
    monkeypatch.setattr(run_s1, 'require_no_strata', lambda: (order.append('e6'), real())[1])
    for stage in STAGES:
        monkeypatch.setitem(run_s3.STAGE_FUNCTIONS, stage, make_fake(stage, order))
    assert toy_cli(tmp_path, toy_runs) == 0
    assert order[0] == 'e6'                                                                 # before the first read
    for i, name in enumerate(order):
        if name in STAGES:
            assert order[i - 1] == 'e6', order
    assert [n for n in order if n in STAGES] == list(STAGES)


def test_a_model_server_that_starts_between_the_stages_stops_the_run_cleanly_and_it_resumes(tmp_path, toy_runs, fakes, monkeypatch, capsys):
    real, calls = run_s1.require_no_strata, []

    def fails_on_the_fourth_call():
        calls.append(1)
        if len(calls) == 4:                                    # 1: the start, 2-4: before levels, labels and fit
            raise run_s1.StrataRunning('E6: a Strata process started meanwhile')
        return real()
    monkeypatch.setattr(run_s1, 'require_no_strata', fails_on_the_fourth_call)
    assert toy_cli(tmp_path, toy_runs) == 2
    run = only_run(tmp_path)
    err = capsys.readouterr().err
    assert 'E6' in err and '--resume' in err and str(run) in err and fakes.calls == ['levels', 'labels']
    ledger = read_json(run / 'STAGES.json')
    assert ledger['stages']['labels']['complete'] and 'fit' not in ledger['stages'] and ledger['calls'][0]['stopped_by'] == 'StrataRunning'
    assert not (run / 'S3.lock').exists() and catalog_rows(tmp_path) == []
    monkeypatch.setattr(run_s1, 'require_no_strata', real)
    assert toy_cli(tmp_path, toy_runs, '--resume', str(run)) == 0
    assert fakes.calls == ['levels', 'labels', 'fit', 'compose', 'gate'] and catalog_rows(tmp_path) == [(run.name, 's3')]


# ---- the lock ----------------------------------------------------------------------------------------------------------------------------
def test_a_run_folder_that_another_call_holds_is_refused_and_the_lock_is_not_deleted(tmp_path, toy_runs, fakes, capsys):
    assert toy_cli(tmp_path, toy_runs, '--stages', 'levels') == 0
    run = only_run(tmp_path)
    (run / 'S3.lock').write_text(json.dumps({'pid': 4242, 'created': '2026-10-21T00:00:00+00:00'}), encoding='utf-8')
    capsys.readouterr()
    assert toy_cli(tmp_path, toy_runs, '--resume', str(run)) == 2
    err = capsys.readouterr().err
    assert 'lock' in err and '4242' in err and (run / 'S3.lock').is_file() and fakes.calls == ['levels']


def test_the_lock_is_released_when_a_stage_fails(tmp_path, toy_runs, fakes, monkeypatch, capsys):
    def fails(ctx):
        raise run_s3.StageFailed('the accounting identity is not exact')
    monkeypatch.setitem(run_s3.STAGE_FUNCTIONS, 'labels', fails)
    assert toy_cli(tmp_path, toy_runs) == 2
    run = only_run(tmp_path)
    assert 'not exact' in capsys.readouterr().err and not (run / 'S3.lock').exists() and fakes.calls == ['levels']
    ledger = read_json(run / 'STAGES.json')
    assert ledger['stages']['levels']['complete'] and ledger['stages']['labels']['complete'] is False       # begun, not complete
    assert catalog_rows(tmp_path) == [] and 's3_result' not in read_json(run / 'RUN.json')


def test_a_stage_whose_peak_memory_is_over_the_cap_is_reported_and_marked_not_hidden(tmp_path, toy_runs, fakes, monkeypatch, capsys):
    monkeypatch.setattr(run_s3, '_tree_memory', lambda: (50 * 2**30, 60 * 2**30))   # the machine says 50 GiB (the bound 60), the cap is 40
    assert toy_cli(tmp_path, toy_runs) == 0                                         # a report, not a stop: the author decides
    err = capsys.readouterr().err
    run = only_run(tmp_path)
    stages = read_json(run / 'STAGES.json')['stages']
    assert all(stages[s]['peak_private_gib'] == 50.0 and stages[s]['within_memory_cap'] is False for s in STAGES)
    assert all(stages[s]['peak_private_upper_gib'] == 60.0 for s in STAGES)         # the bound is beside the peak; the cap is the peak's
    assert err.count('above the cap of 40 GiB') == len(STAGES) and 'WARNING' in err
    s3 = read_json(run / 'RUN.json')['s3_result']
    assert s3['peak_private_gib'] == 50.0 and s3['within_memory_cap'] is False and s3['memory_cap_gib'] == 40.0


def test_an_unexpected_error_prints_its_traceback_without_the_match_keys_and_returns_1(tmp_path, toy_runs, fakes, monkeypatch, capsys):
    s1 = run_s1.load_s1(toy_runs.s1, verify=False)
    key = bytes(np.asarray(s1.array('matches/key'))[0]).decode('ascii')
    release(s1)

    def breaks(ctx):
        raise KeyError(f'the match {key} is broken, and its id is {key.rsplit("_", 1)[1]}')
    monkeypatch.setitem(run_s3.STAGE_FUNCTIONS, 'levels', breaks)
    assert toy_cli(tmp_path, toy_runs) == 1
    err = capsys.readouterr().err
    assert 'KeyError' in err and 'is broken' in err and KEY_PREFIX not in err and '<key>' in err
    assert not (only_run(tmp_path) / 'S3.lock').exists()


def first_key_of(toy_runs) -> str:
    s1 = run_s1.load_s1(toy_runs.s1, verify=False)
    key = bytes(np.asarray(s1.array('matches/key'))[0]).decode('ascii')
    release(s1)
    return key


def test_a_match_key_in_the_result_of_a_stage_is_an_error_and_run_json_does_not_get_it(tmp_path, toy_runs, fakes, monkeypatch, capsys):
    key = first_key_of(toy_runs)
    fake_levels = make_fake('levels', [])

    def leaks(ctx):
        return {**fake_levels(ctx), 'worst_match': key}
    monkeypatch.setitem(run_s3.STAGE_FUNCTIONS, 'levels', leaks)
    assert toy_cli(tmp_path, toy_runs) == 1
    err = capsys.readouterr().err
    assert 'match keys occur' in err and KEY_PREFIX not in err
    run = only_run(tmp_path)
    manifest = (run / 'RUN.json').read_text(encoding='utf-8')
    assert KEY_PREFIX not in manifest and 'levels_result' not in manifest and not (run / 'S3.lock').exists()


# ---- G-S3b B3 item 2: a stage is complete only after its result is key-scanned and recorded -------------------------------------------
def test_the_result_of_a_stage_is_in_run_json_when_the_stage_is_marked_complete(tmp_path, toy_runs, fakes, monkeypatch):
    seen, real = [], run_s3.Ledger.finish

    def finish(self, name, outputs, **facts):
        manifest = read_json(self.path.parent / 'RUN.json')
        seen.append((name, f'{name}_result' in manifest))                                  # RUN.json as it is the moment before the mark
        return real(self, name, outputs, **facts)
    monkeypatch.setattr(run_s3.Ledger, 'finish', finish)
    assert toy_cli(tmp_path, toy_runs) == 0
    assert seen == [(stage, True) for stage in STAGES]


def test_a_stage_whose_result_holds_a_match_key_is_not_marked_complete_and_leaves_no_result(tmp_path, toy_runs, fakes, monkeypatch, capsys):
    key = first_key_of(toy_runs)
    fake_levels = make_fake('levels', [])
    monkeypatch.setitem(run_s3.STAGE_FUNCTIONS, 'levels', lambda ctx: {**fake_levels(ctx), 'worst_match': key})
    assert toy_cli(tmp_path, toy_runs, '--stages', 'levels') == 1
    run = only_run(tmp_path)
    assert 'match keys occur' in capsys.readouterr().err
    ledger = read_json(run / 'STAGES.json')
    assert ledger['stages']['levels']['complete'] is False and 'outputs' not in ledger['stages']['levels']
    assert 'levels_result' not in read_json(run / 'RUN.json')


def test_a_rerun_of_an_incomplete_stage_replaces_its_own_result_and_no_other_key_of_run_json(tmp_path, toy_runs, fakes, monkeypatch):
    real, state = run_s3.Ledger.finish, {'cut': True}

    def finish_that_is_cut_once(self, name, outputs, **facts):
        if name == 'levels' and state['cut']:
            state['cut'] = False
            raise RuntimeError('the call was cut between the result and the mark')
        return real(self, name, outputs, **facts)
    monkeypatch.setattr(run_s3.Ledger, 'finish', finish_that_is_cut_once)
    assert toy_cli(tmp_path, toy_runs, '--stages', 'levels') == 1
    run = only_run(tmp_path)
    before = read_json(run / 'RUN.json')
    assert before['levels_result']['stage'] == 'levels' and read_json(run / 'STAGES.json')['stages']['levels']['complete'] is False
    fake_levels = make_fake('levels', [])
    monkeypatch.setitem(run_s3.STAGE_FUNCTIONS, 'levels', lambda ctx: {**fake_levels(ctx), 'rerun': True})
    assert toy_cli(tmp_path, toy_runs, '--resume', str(run), '--stages', 'levels') == 0
    after = read_json(run / 'RUN.json')
    assert after['levels_result']['rerun'] is True and 'rerun' not in before['levels_result']              # replaced ...
    others = lambda manifest: {k: v for k, v in manifest.items() if k != 'levels_result'}         # noqa: E731
    assert others(after) == others(before)                                                      # ... alone
    assert read_json(run / 'STAGES.json')['stages']['levels']['complete'] is True


def test_the_result_of_a_stage_that_is_complete_is_never_replaced(tmp_path, toy_runs, fakes):
    assert toy_cli(tmp_path, toy_runs, '--stages', 'levels') == 0
    run = only_run(tmp_path)
    with pytest.raises(ValueError, match='never overwrites'):
        runlog.amend_manifest(run, {'levels_result': {'stage': 'other'}})                    # the write-once rule of the manifest stays
    assert read_json(run / 'RUN.json')['levels_result']['stage'] == 'levels'
    before = read_json(run / 'RUN.json')
    assert toy_cli(tmp_path, toy_runs, '--resume', str(run), '--stages', 'levels') == 0       # complete: skipped, not run again
    assert read_json(run / 'RUN.json') == before and fakes.calls == ['levels']


def test_a_resume_refuses_a_complete_stage_whose_result_is_missing_from_run_json(tmp_path, toy_runs, fakes, capsys):
    assert toy_cli(tmp_path, toy_runs) == 0
    run = only_run(tmp_path)
    manifest = read_json(run / 'RUN.json')
    del manifest['labels_result']
    (run / 'RUN.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    calls, _ = fakes.calls[:], capsys.readouterr()
    assert toy_cli(tmp_path, toy_runs, '--resume', str(run)) == 2
    error = capsys.readouterr().err
    assert 'labels' in error and 'labels_result' in error and 'not consistent' in error
    assert fakes.calls == calls and not (run / 'S3.lock').exists()


# ---- G-S3b B3 item 3: nothing is reused across other inputs or other code ----------------------------------------------------------
def test_a_stage_that_is_not_complete_is_not_begun_again_on_other_inputs(tmp_path):
    ledger = run_s3.Ledger(tmp_path)
    ledger.begin('fit', {'a': 1, 'b': 2}, 'one')
    ledger.begin('fit', {'a': 1, 'b': 2}, 'one')                                              # the same inputs: the stage is run again
    with pytest.raises(run_s3.RunRefused) as raised:
        ledger.begin('fit', {'a': 1, 'b': 3, 'c': 4}, 'two')
    text = str(raised.value)
    assert 'fit' in text and 'changed: b, c' in text and 'new run folder' in text
    assert ledger.stage('fit')['inputs_sha256'] == 'one' and ledger.stage('fit')['inputs'] == {'a': 1, 'b': 2}     # the record is as it was
    ledger.begin('labels', {'x': 1}, 'three')                                                 # another stage has no earlier record


def test_a_resume_refuses_an_incomplete_stage_begun_on_other_inputs_and_runs_nothing(tmp_path, toy_runs, fakes, monkeypatch, capsys):
    def cut(ctx):
        raise RuntimeError('the call was cut')
    monkeypatch.setitem(run_s3.STAGE_FUNCTIONS, 'levels', cut)
    assert toy_cli(tmp_path, toy_runs, '--stages', 'levels') == 1
    run = only_run(tmp_path)
    ledger = read_json(run / 'STAGES.json')
    assert ledger['stages']['levels']['complete'] is False
    ledger['stages']['levels']['inputs']['roles'] = ['FIT']                                   # the stage was begun on other inputs
    ledger['stages']['levels']['inputs_sha256'] = runlog.sha256_json(ledger['stages']['levels']['inputs'])
    (run / 'STAGES.json').write_text(json.dumps(ledger), encoding='utf-8')
    monkeypatch.setitem(run_s3.STAGE_FUNCTIONS, 'levels', make_fake('levels', fakes.calls))
    capsys.readouterr()
    assert toy_cli(tmp_path, toy_runs, '--resume', str(run)) == 2
    error = capsys.readouterr().err
    assert 'levels' in error and 'changed: roles' in error and 'new run folder' in error
    assert fakes.calls == [] and read_json(run / 'STAGES.json')['stages']['levels']['complete'] is False and not (run / 'S3.lock').exists()


def started_run(tmp_path, toy_runs, monkeypatch, state=None):
    """A run whose first call (the stage levels) ran on the code state ``state`` (default: the pinned one)."""
    if state is not None:
        monkeypatch.setattr(runlog, 'code_state', lambda: dict(state))
    assert toy_cli(tmp_path, toy_runs, '--stages', 'levels') == 0
    return only_run(tmp_path)


@pytest.mark.parametrize('now, text', [({'code_commit': 'd' * 40, 'code_dirty': False}, 'commit'),
                                       ({'code_commit': 'c' * 40, 'code_dirty': True}, 'dirty'),
                                       ({'code_commit': 'c' * 40, 'code_dirty': None}, 'dirty')])
def test_a_resume_on_other_code_is_refused_and_runs_nothing(tmp_path, toy_runs, fakes, monkeypatch, capsys, now, text):
    run = started_run(tmp_path, toy_runs, monkeypatch)
    ledger_before, calls = (run / 'STAGES.json').read_bytes(), fakes.calls[:]
    monkeypatch.setattr(runlog, 'code_state', lambda: dict(now))
    capsys.readouterr()
    assert toy_cli(tmp_path, toy_runs, '--resume', str(run)) == 2
    error = capsys.readouterr().err
    assert text in error and 'cccccccccccc' in error and 'new run folder' in error                # it names both states, short
    assert fakes.calls == calls and not (run / 'S3.lock').exists()
    assert (run / 'STAGES.json').read_bytes() == ledger_before                                  # not even a record of the refused call


def test_a_resume_on_a_dirty_tree_is_refused_also_when_the_run_began_on_the_same_dirty_tree(tmp_path, toy_runs, fakes, monkeypatch, capsys):
    dirty = {'code_commit': 'c' * 40, 'code_dirty': True}
    run = started_run(tmp_path, toy_runs, monkeypatch, dirty)                                 # a first call on a dirty tree is allowed ...
    assert read_json(run / 'RUN.json')['code_dirty'] is True and fakes.calls == ['levels']
    capsys.readouterr()
    assert toy_cli(tmp_path, toy_runs, '--resume', str(run)) == 2                             # ... a resume is not
    assert 'dirty' in capsys.readouterr().err and fakes.calls == ['levels']


def test_a_resume_on_the_same_clean_code_runs_the_rest(tmp_path, toy_runs, fakes, monkeypatch):
    run = started_run(tmp_path, toy_runs, monkeypatch)
    assert toy_cli(tmp_path, toy_runs, '--resume', str(run)) == 0 and fakes.calls == list(STAGES)


def test_every_line_of_the_progress_log_records_the_code_state(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, 'code_state', lambda: {'code_commit': 'a' * 40, 'code_dirty': True})
    log = run_s3._ProgressLog(tmp_path / 'PROGRESS.jsonl')
    log.add('tier', wall_s=1.0)
    monkeypatch.setattr(runlog, 'code_state', lambda: {'code_commit': 'b' * 40, 'code_dirty': False})
    log.add('b0', wall_s=2.0)
    assert log.last('tier')['code_commit'] == 'a' * 40 and log.last('tier')['code_dirty'] is True and log.last('tier')['wall_s'] == 1.0
    assert (log.last('b0')['code_commit'], log.last('b0')['code_dirty']) == ('b' * 40, False)       # the state at the time of each line


def test_the_progress_log_of_a_run_records_the_code_state_in_every_line(e2e):
    lines = [json.loads(line) for line in (e2e.run / 'fit' / 'PROGRESS.jsonl').read_text(encoding='utf-8').splitlines()]
    state = (PINNED_CODE_STATE['code_commit'], PINNED_CODE_STATE['code_dirty'])
    assert len(lines) >= 16 and all((entry['code_commit'], entry['code_dirty']) == state for entry in lines)


def test_a_match_key_in_the_arguments_that_run_json_records_is_an_error_before_any_stage_runs(tmp_path, toy_runs, fakes, capsys):
    key = first_key_of(toy_runs)
    spec = tmp_path / f'{key}.json'                                                    # --spec is recorded in RUN.json
    shutil.copy(toy_runs.spec, spec)
    assert run_s3.main(cli_args(tmp_path, toy_runs.s1, toy_runs.s2, toy_runs.s3a, spec)) == 1
    err = capsys.readouterr().err
    assert 'match keys occur' in err and KEY_PREFIX not in err and fakes.calls == []
    assert not (only_run(tmp_path) / 'S3.lock').exists()


# ---- the smoke run, with fake stages ---------------------------------------------------------------------------------------------------
def test_a_smoke_run_records_the_selection_as_match_rows_and_every_stage_sees_the_subset(tmp_path, toy_runs, monkeypatch):
    calls, seen = [], {}
    for stage in STAGES:
        monkeypatch.setitem(run_s3.STAGE_FUNCTIONS, stage, make_fake(stage, calls, seen))
    assert toy_cli(tmp_path, toy_runs, '--smoke', '3') == 0
    run = only_run(tmp_path)
    smoke = read_json(run / 'RUN.json')['smoke']
    s1 = run_s1.load_s1(toy_runs.s1, verify=False)
    role = np.asarray(s1.array('matches/role'))
    assert smoke['n'] == 3 and smoke['seed'] == 20261021 and smoke['asked'] == {'FIT': 3, 'STOP': 1, 'DEV': 1}
    assert {k: len(v) for k, v in smoke['matches'].items()} == {'FIT': 3, 'STOP': 1, 'DEV': 1}
    assert all(isinstance(r, int) for rows in smoke['matches'].values() for r in rows)                  # row numbers, never keys
    assert smoke['matches']['STOP'] == [2] and smoke['matches']['DEV'] == [3] and set(smoke['matches']['FIT']) <= {0, 1, 4, 5}
    assert smoke['selection_sha256'] == runlog.sha256_json(smoke['matches'])
    chosen = {r for rows in smoke['matches'].values() for r in rows}
    for stage in STAGES:
        view = seen[stage].s1
        assert isinstance(view, run_s3.SubsetStore) and seen[stage].smoke == smoke
        shown = np.asarray(view.array('matches/role'))
        assert all((shown[m] == role[m]) == (m in chosen) for m in range(len(role)))
    ledger = read_json(run / 'STAGES.json')
    assert all(ledger['stages'][s]['inputs']['smoke_sha256'] == smoke['selection_sha256'] for s in STAGES)
    assert read_json(run / 'RUN.json')['s3_result']['smoke'] is True
    release(s1)


def test_a_full_run_passes_the_s1_store_itself_and_records_no_smoke_selection(tmp_path, toy_runs, monkeypatch):
    calls, seen = [], {}
    for stage in STAGES:
        monkeypatch.setitem(run_s3.STAGE_FUNCTIONS, stage, make_fake(stage, calls, seen))
    assert toy_cli(tmp_path, toy_runs) == 0
    assert all(type(ctx.s1).__name__ == 'Store' and ctx.smoke is None for ctx in seen.values())
    assert read_json(only_run(tmp_path) / 'RUN.json')['smoke'] is None
    assert read_json(only_run(tmp_path) / 'STAGES.json')['stages']['levels']['inputs']['smoke_sha256'] is None


# ---- what RUN.json holds -----------------------------------------------------------------------------------------------------------------
def test_run_json_holds_the_sources_the_hashes_the_arguments_and_the_constants(tmp_path, toy_runs, fakes):
    assert toy_cli(tmp_path, toy_runs, '--workers', '3') == 0
    run = only_run(tmp_path)
    m = read_json(run / 'RUN.json')
    s1, s2 = Store(toy_runs.s1), Store(toy_runs.s2)
    assert m['s1']['files_sha256'] == runlog.sha256_json(s1.manifest['files']) and m['s2']['files_sha256'] == runlog.sha256_json(
        s2.manifest['files']
    )
    assert m['s1']['run'] == toy_runs.s1.parent.name and m['s2']['run'] == toy_runs.s2.parent.name
    assert m['s3a'] == {
        'run': toy_runs.s3a.name,
        'prices_sha256': sha_of(toy_runs.files.prices),
        'rules_sha256': sha_of(toy_runs.files.rules),
    }
    assert m['spec'] == {'sha256': sha_of(toy_runs.spec), 'decision': 'decisions.S3_prices'}
    assert m['args'] == {'smoke': None, 'workers': 3, 'stages': list(STAGES), 'verify': False, 'spec': str(toy_runs.spec)}
    assert m['roles'] == {'tier': ['FIT'], 'b0': {'fit': 'FIT'}, 'b1': {'fit': 'FIT', 'stop': 'STOP'}, 'valued': ['FIT', 'STOP', 'DEV']}
    assert m['compose_constants'] == run_s3.compose_constants() and m['memory_cap_gib'] == 40.0
    for key in ('code_commit', 'code_dirty', 'config_sha256', 'spec_sha256', 'legacy_head', 'strata_running', 'python', 'numpy', 'cpu'):
        assert key in m
    assert m['code_commit'] == runlog.code_state()['code_commit'] or m['code_commit'] is None
    release(s1, s2)


def test_run_json_records_the_blocks_of_the_compose_calls_and_the_sizes_of_the_run():
    c = run_s3.compose_constants()
    assert c['blocks'] == {'expected_onsets': 4096, 'b0_simulate': 512, 'b0_expect': 64, 'b1_simulate': 256}
    assert (c['S_b0'], c['S_b1'], c['seed'], c['b1_dev_matches'], c['r31_limit'], c['chunk_rows']) == (1000, 200, 20261021, 500, 50.0, 8192)


# ---- one run of all five stages with the real functions ----------------------------------------------------------------------------------
class Spy:
    """What the e2e run recorded about its calls (the real functions run behind every spy)."""

    def __init__(self):
        self.kill_levels, self.unit_values, self.accounting, self.gold_bin = [], [], [], []
        self.tier_fit, self.b0_fit, self.b2_fit, self.fit_head, self.rows = [], [], [], [], []
        self.compose, self.e6 = [], []


def spy_on(mp, owner, name, record):
    real = getattr(owner, name)

    def wrapper(*args, **kwargs):
        record.append((args, kwargs))
        return real(*args, **kwargs)
    mp.setattr(owner, name, wrapper)


def install_spies(mp, spy: Spy) -> None:
    spy_on(mp, value, 'kill_levels', spy.kill_levels)
    spy_on(mp, value, 'unit_values', spy.unit_values)
    spy_on(mp, value, 'accounting', spy.accounting)
    spy_on(mp, value, 'gold_bin_means', spy.gold_bin)
    spy_on(mp, towertier.TierModel, 'fit', spy.tier_fit)
    spy_on(mp, baselines_b0.B0Heads, 'fit', spy.b0_fit)
    spy_on(mp, baselines_b1.B2, 'fit_stores', spy.b2_fit)
    real_head = baselines_b1.fit_head

    def fit_head(fit_rows, stop_rows, name, kind, **kwargs):
        spy.fit_head.append((fit_rows.role, stop_rows.role, name, kind, kwargs.get('check')))
        return real_head(fit_rows, stop_rows, name, kind, **kwargs)
    mp.setattr(baselines_b1, 'fit_head', fit_head)

    class RecordingRows(baselines_b1.HeadRows):
        def __init__(self, s1, s2, labels, dropped, role):
            spy.rows.append(role)
            super().__init__(s1, s2, labels, dropped, role)
    mp.setattr(baselines_b1, 'HeadRows', RecordingRows)
    for method in ('expected_onsets', 'expect', 'simulate'):
        def make(real, method):
            def wrapper(self, idx, block=compose.DEFAULT_BLOCK, workers=1, check=None):
                spy.compose.append({'method': method, 'kind': self.heads.feature_kind, 'rows': len(idx), 'block': block, 'workers': workers,
                                    'check': check, 'S': self.S, 'seed': self.seed})
                return real(self, idx, block=block, workers=workers, check=check)
            return wrapper
        mp.setattr(compose.Composer, method, make(getattr(compose.Composer, method), method))

    def never(self, row):
        raise AssertionError('Composer.simulate_onsets has no workers: it computes in the parent, so a reported value must not use it')
    mp.setattr(compose.Composer, 'simulate_onsets', never)
    real_e6 = run_s1.require_no_strata
    mp.setattr(run_s1, 'require_no_strata', lambda: (spy.e6.append(1), real_e6())[1])


@pytest.fixture(scope='module')
def e2e(world, tmp_path_factory):
    """All five stages on the smoke subset (``--smoke 12``) of the random world, with the real functions, two spawned workers for the
    composition and the spies of ``install_spies``. B0 is fitted with a strong ridge (``ALPHA`` 1) so that its heads on a few hundred rows
    do not predict absurd rates."""
    out = tmp_path_factory.mktemp('run_s3_e2e')
    mp, spy = pytest.MonkeyPatch(), Spy()
    install_spies(mp, spy)
    mp.setattr(baselines_b0, 'ALPHA', 1.0)
    stdout, stderr = io.StringIO(), io.StringIO()
    try:
        with empty_process_list(), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            rc = run_s3.main(cli_args(out, world.s1, world.s2, world.s3a_run, world.spec, '--smoke', '12', '--workers', '1'),
                             records=random_value_record)
    finally:
        mp.undo()
    runs = sorted((out / 'runs').iterdir()) if (out / 'runs').exists() else []
    yield types.SimpleNamespace(rc=rc, run=runs[0] if runs else None, out=out, spy=spy, stdout=stdout.getvalue(), stderr=stderr.getvalue(),
                                world=world)
    gc.collect()
    shutil.rmtree(out, ignore_errors=True)


def run_json(e2e) -> dict:
    return read_json(e2e.run / 'RUN.json')


def test_all_five_stages_run_on_the_smoke_subset_and_the_run_is_complete_and_registered(e2e):
    assert e2e.rc == 0, e2e.stderr[-3000:]
    ledger, manifest = read_json(e2e.run / 'STAGES.json'), run_json(e2e)
    assert all(ledger['stages'][s]['complete'] for s in STAGES) and manifest['s3_result']['complete'] is True
    assert [c['stages_run'] for c in ledger['calls']] == [list(STAGES)] and ledger['calls'][0]['stopped_by'] is None
    assert catalog_rows(e2e.out) == [(e2e.run.name, 's3')] and not (e2e.run / 'S3.lock').exists()
    for stage in STAGES:                                                                        # what was recorded is what is on the disk
        assert run_s3.stage_outputs(e2e.run, stage) == ledger['stages'][stage]['outputs']
    for rel in sum((list(v) for v in run_s3.STAGE_OUTPUTS.values()), []):
        assert (e2e.run / rel).exists(), rel
    assert 'S3 done' in e2e.stdout and 'gate_pass' in e2e.stdout


def test_the_smoke_selection_of_the_run_is_what_every_stage_worked_on(e2e):
    manifest = run_json(e2e)
    smoke = manifest['smoke']
    assert smoke['asked'] == {'FIT': 12, 'STOP': 3, 'DEV': 4} and {k: len(v) for k, v in smoke['matches'].items()} == {
        'FIT': 12,
        'STOP': 3,
        'DEV': 4,
    }
    assert manifest['levels_result']['report'] == {
        'matches': 19,
        'unreadable': 0,
        'kills_read': manifest['levels_result']['n_valued_kills'],
        'kills_unread': 0,
    }
    assert manifest['fit_result']['n_fit_matches'] == 12 and manifest['compose_result']['matches'] == 4
    assert manifest['labels_result']['n_valued_units'] < manifest['labels_result']['n_units']
    s1 = run_s1.load_s1(e2e.world.s1, verify=False)
    keys = np.asarray(s1.array('matches/key')).tolist()
    chosen = {keys[r].decode('ascii') for rows in smoke['matches'].values() for r in rows}
    assert len(chosen) == 19
    release(s1)


def test_the_roles_are_pinned_the_tier_law_and_b0_on_fit_b1_and_b2_on_fit_with_the_early_stopping_on_stop_and_never_dev(e2e):
    spy = e2e.spy
    assert len(spy.tier_fit) == 1 and spy.tier_fit[0][1]['roles'] == ('FIT',) and callable(spy.tier_fit[0][1]['check'])
    assert len(spy.b0_fit) == 1 and spy.b0_fit[0][1]['roles'] == {'fit': 'FIT'} and type(spy.b0_fit[0][1]['check']).__name__ == '_TickTimer'
    assert len(spy.b2_fit) == 1 and spy.b2_fit[0][1]['roles'] == {'fit': 'FIT', 'stop': 'STOP'} and callable(spy.b2_fit[0][1]['check'])
    assert len(spy.fit_head) == 13  # eight masked and five open heads, one by one
    assert {(fit, stop) for fit, stop, _, _, _ in spy.fit_head} == {('FIT', 'STOP')}                # stop is STOP, never DEV
    assert [(kind, name) for _, _, name, kind, _ in spy.fit_head] == run_s3._head_names() and spy.fit_head[0][2] == 'M2Rate'
    assert all(callable(check) for *_, check in spy.fit_head)
    assert spy.rows[:4] == ['FIT', 'STOP', 'FIT', 'STOP'] and set(spy.rows[4:]) == {'DEV'}  # B1's rows, then B2's; DEV only for scoring
    assert run_json(e2e)['roles'] == {
        'tier': ['FIT'],
        'b0': {'fit': 'FIT'},
        'b1': {'fit': 'FIT', 'stop': 'STOP'},
        'valued': ['FIT', 'STOP', 'DEV'],
    }


def test_only_fit_stop_and_dev_are_valued_and_read(e2e):
    spy = e2e.spy
    assert len(spy.kill_levels) == 1 and spy.kill_levels[0][1]['roles'] == ('FIT', 'STOP', 'DEV')
    assert spy.unit_values[0][1]['roles'] == ('FIT', 'STOP', 'DEV') and spy.accounting[0][1]['roles'] == ('FIT', 'STOP', 'DEV')
    assert spy.kill_levels[0][1]['workers'] == 1 and spy.kill_levels[0][1]['records'] is random_value_record
    assert run_s3.LABEL_ROLES == value.LABEL_ROLES == flowprice.S3_ROLES


def test_the_gold_bin_means_of_the_labels_are_made_from_the_units_of_fit_alone(e2e):
    """The means go to the composition as the gold of a simulated kill: STOP and DEV units must not be in them."""
    (args, _), = e2e.spy.gold_bin
    s1 = run_s1.load_s1(e2e.world.s1, verify=False)
    view = run_s3.SubsetStore(s1, run_json(e2e)['smoke'])
    fit_units, every_unit = value.role_units(view, ('FIT',)), value.role_units(view, run_s3.LABEL_ROLES)
    assert fit_units.sum() < every_unit.sum() and np.array_equal(np.asarray(args[1]), fit_units)
    release(s1)


def test_every_fit_function_and_every_compose_call_gets_the_e6_guard_as_check(e2e, monkeypatch):
    spy = e2e.spy
    seen = []
    monkeypatch.setattr(run_s1, 'require_no_strata', lambda: seen.append(1))
    for check in (
        spy.tier_fit[0][1]['check'],
        spy.b2_fit[0][1]['check'],
        spy.fit_head[0][4],
        spy.fit_head[-1][4],
        spy.b0_fit[0][1]['check'],
    ):
        before = len(seen)
        check()
        assert len(seen) == before + 1                                                          # it reaches run_s1.require_no_strata
    assert len(spy.compose) == 5 and all(callable(c['check']) for c in spy.compose)
    before = len(seen)
    spy.compose[0]['check']()
    assert len(seen) == before + 1
    assert len(spy.e6) >= 6 + len(STAGES)  # the start, every stage, the raw reads, the models


def test_the_compose_calls_have_the_blocks_the_workers_and_the_simulation_counts_of_the_plan_and_the_screen_comes_first(e2e):
    calls = e2e.spy.compose
    assert [(c['kind'], c['method']) for c in calls] == [('b0', 'expected_onsets'), ('b1', 'expected_onsets'), ('b0', 'expect'),
                                                         ('b0', 'simulate'), ('b1', 'simulate')]
    blocks = {(c['kind'], c['method']): c['block'] for c in calls}
    assert blocks == {('b0', 'expected_onsets'): run_s3.BLOCK_EXPECTED_ONSETS, ('b1', 'expected_onsets'): run_s3.BLOCK_EXPECTED_ONSETS,
                      ('b0', 'expect'): run_s3.BLOCK_B0_EXPECT, ('b0', 'simulate'): run_s3.BLOCK_B0_SIMULATE,
                      ('b1', 'simulate'): run_s3.BLOCK_B1_SIMULATE}
    assert (run_s3.BLOCK_EXPECTED_ONSETS, run_s3.BLOCK_B0_SIMULATE, run_s3.BLOCK_B0_EXPECT, run_s3.BLOCK_B1_SIMULATE) == (
        4096,
        512,
        64,
        256,
    )
    assert all(c['workers'] >= 2 for c in calls)  # every compose value is computed in workers (R30)
    assert {(c['kind'], c['S']) for c in calls} == {('b0', 1000), ('b1', 200)} and {c['seed'] for c in calls} == {20261021}
    assert {c['rows'] for c in calls if c['kind'] == 'b0'} == {run_json(e2e)['compose_result']['rows']}


def test_compose_stage_result_records_every_call_with_its_blocks_its_time_and_the_cost_of_its_checks(e2e):
    result = run_json(e2e)['compose_result']
    assert result['blocks'] == run_s3.compose_constants()['blocks'] and result['chunk_rows'] == 8192 and result['workers'] == 2
    assert [(c['head_set'], c['method']) for c in result['calls']] == [
        ('B0', 'expected_onsets'),
        ('B1', 'expected_onsets'),
        ('B0', 'expect'),
        ('B0', 'simulate'),
        ('B1', 'simulate'),
    ]
    for c in result['calls']:
        assert c['workers'] == 2 and c['n_blocks'] == -(-c['rows'] // c['block']) and c['wall_s'] > 0
        assert c['seconds_per_block_wall'] == pytest.approx(c['wall_s'] / c['n_blocks'], abs=1e-3) and c['call_check_calls'] >= 1
        assert 'call_check_s_mean' in c and 'call_check_budget_s' in c and 'block_compute_s_estimate' in c       # per-call names (G-S3b B3)
    assert result['r31']['limit'] == 50.0 and result['r31']['max_expected_onsets_180_b0'] < 50.0
    assert result['b1_sample'] == 'all the smoke DEV matches' and len(result['b1_matches']) == result['matches'] == 4
    assert result['check_calls'] >= 5 and result['check_s_longest'] >= result['check_s_mean'] >= 0


def test_the_compose_files_hold_one_row_per_dev_frame_row_with_the_arrays_of_the_plan(e2e):
    s1 = run_s1.load_s1(e2e.world.s1, verify=False)
    smoke = run_json(e2e)['smoke']
    frame_match = np.asarray(s1.array('frame_rows/match'))
    expected_rows = np.flatnonzero(np.isin(frame_match, smoke['matches']['DEV']))
    with np.load(e2e.run / 'compose' / 'b0_dev.npz') as z:
        b0 = {k: z[k] for k in z.files}
    with np.load(e2e.run / 'compose' / 'b1_dev.npz') as z:
        b1 = {k: z[k] for k in z.files}
    n = len(expected_rows)
    assert set(b0) == {'S', 'expected', 'expected_onsets', 'frame_row', 'mc_mean', 'mc_sd', 'match', 'q19', 'seed'}
    assert set(b1) == {'S', 'expected_onsets', 'frame_row', 'mc_mean', 'mc_sd', 'match', 'q19', 'seed'}            # B1 has no closed form
    assert np.array_equal(b0['frame_row'], expected_rows) and np.array_equal(b0['match'], frame_match[expected_rows])
    assert b0['expected'].shape == b0['mc_mean'].shape == b0['mc_sd'].shape == b0['expected_onsets'].shape == (n, 3)
    assert b0['q19'].shape == (n, 3, 19) and run_s3._scalar(b0['S']) == 1000 and run_s3._scalar(b0['seed']) == 20261021
    assert (
        np.array_equal(b1['frame_row'], expected_rows) and b1['q19'].shape == (n, 3, 19) and run_s3._scalar(b1['S']) == 200
    )  # all 4 smoke DEV matches
    for a in (b0['expected'], b0['mc_mean'], b0['mc_sd'], b0['q19'], b1['mc_mean'], b1['expected_onsets']):
        assert np.isfinite(a).all()
    assert (np.diff(b0['q19'], axis=2) >= 0).all() and (b0['mc_sd'] >= 0).all()                             # quantiles in order
    assert not (e2e.run / 'compose' / 'parts').exists()  # the chunks are in the two files
    release(s1)


def test_the_labels_stage_checks_the_accounting_and_writes_the_store_the_summary_and_the_provenance(e2e):
    manifest = run_json(e2e)
    result = manifest['labels_result']
    assert result['max_abs_error'] == 0.0 and result['n_valued_units'] > 0 and 0 <= result['draw'] <= 1
    assert result['winner_blue'] + result['winner_red'] + result['draw'] == pytest.approx(1.0)
    store = Store(e2e.run / 'labels' / 'store')
    store.verify()
    labels = read_json(e2e.run / 'labels' / 'store' / 'LABELS.json')
    s1 = run_s1.load_s1(e2e.world.s1, verify=False)
    assert labels['prices_sha256'] == manifest['s3a']['prices_sha256'] and labels['rules_sha256'] == manifest['s3a']['rules_sha256']
    assert labels['s1_files_sha256'] == manifest['s1']['files_sha256'] and set(labels['code_state']) == {'code_commit', 'code_dirty'}
    assert labels['gold_bin_fallback'] == result['gold_bin_fallback'] and labels['roles'] == ['FIT', 'STOP', 'DEV']
    assert labels['kill_levels_sha256'] == manifest['levels_result']['kill_levels_sha256']
    assert np.asarray(store.array('labels/gold_reps')).shape == (31,) and np.asarray(store.array('labels/v_start')).shape[1] == 3
    accounting = read_json(e2e.run / 'labels' / 'ACCOUNTING.json')
    assert accounting['max_abs_error'] == 0.0 and accounting['n_valued_matches'] == 19
    release(store, s1)


def test_the_fit_stage_records_the_views_the_dropped_columns_the_means_and_the_fill_counts_of_the_fit_matches(e2e):
    result = run_json(e2e)['fit_result']
    s1 = run_s1.load_s1(e2e.world.s1, verify=False)
    s2 = flowprice._open_s2(e2e.world.s2, False)
    view = run_s3.SubsetStore(s1, run_json(e2e)['smoke'])
    fit_views = np.flatnonzero(features.split_views(view, s2, 'FIT'))
    means = features.b0_fit_means(s2, fit_views)
    fill = np.zeros(features.N_B0, dtype=np.int64)
    features.b0(s2, fit_views, means, fill_counts=fill)
    assert result['n_fit_views'] == len(fit_views) > 0 and result['dropped'] == list(features.dropped_columns(s2, fit_views))
    assert 't_s' in result['dropped'] and np.array_equal(np.asarray(result['b0_fit_means']), means) and len(result['b0_fit_means']) == 39
    every_view = np.arange(len(np.asarray(s2.array('views/match'))))
    assert CONSTANT_IN_FIT in result['dropped']                                                    # judged by the FIT views alone (R14)
    assert CONSTANT_IN_FIT not in features.dropped_columns(s2, every_view)
    assert result['b0_fill_counts'] == fill.tolist() and sum(result['b0_fill_counts']) > 0                         # views without distances
    nan_columns = [features.B0_NAMES.index('geo_min'), features.B0_NAMES.index('geo_mean')]
    assert fill[nan_columns[0]] == fill[nan_columns[1]] == 12 and fill.sum() == 24  # one view of each of the 12 FIT matches
    assert read_json(e2e.run / 'fit' / 'FIT_INPUTS.json')['b0_names'] == list(features.B0_NAMES)
    release(s1, s2)


def test_the_fit_stage_records_time_peak_memory_convergence_and_rounds_of_every_model_and_head(e2e):
    result = run_json(e2e)['fit_result']
    assert (
        result['tier']['summary']['converged'] in (True, False) and 'n_iter' in result['tier']['summary'] and result['tier']['wall_s'] > 0
    )
    names = [f'{kind}.{name}' for kind, name in run_s3._head_names()]
    assert sorted(result['b0']['heads']) == sorted(names) and sorted(result['b1']['heads']) == sorted(names)
    for key, unit in result['b0']['heads'].items():
        assert unit['wall_s'] >= 0 and unit['peak_private_gib'] > 0, key
    assert result['b0']['setup']['wall_s'] >= 0
    for kind in ('masked', 'open'):
        for name, summary in result['b0']['summary'][kind].items():
            assert 'converged' in summary and 'n_iter' in summary, name
    for key, unit in result['b1']['heads'].items():
        s = unit['summary']
        assert unit['wall_s'] > 0 and unit['peak_private_gib'] > 0 and {'best_iteration', 'num_trees', 'rounds_run'} <= set(s), key
    assert result['b1']['heads']['masked.M2Rate']['summary']['n_valid'] > 0                       # the early stopping read STOP rows
    assert result['b2']['summary']['n_valid'] > 0 and result['b2']['roles'] == {'fit': 'FIT', 'stop': 'STOP'}
    assert result['peak_private_gib'] > 0 and result['wall_s'] > 0 and result['dev']['n_rows'] == run_json(e2e)['compose_result']['rows']
    log = [json.loads(line) for line in (e2e.run / 'fit' / 'PROGRESS.jsonl').read_text(encoding='utf-8').splitlines()]
    units = [entry['unit'] for entry in log]
    assert units[:3] == ['tier', 'b0', 'b1.masked.M2Rate'] and units[-3:] == ['b1', 'b2', 'dev']          # M2Rate first, as the plan says


def test_the_dev_scores_and_predictions_of_the_fit_stage_cover_every_head_of_b0_b1_and_b2(e2e):
    doc = read_json(e2e.run / 'fit' / 'DEV_NLL.json')
    assert sorted(doc) == ['b0', 'b1', 'b2'] and sorted(doc['b0']) == ['masked', 'open'] == sorted(doc['b1'])
    assert sorted(doc['b0']['masked']) == sorted(n for k, n in run_s3._head_names() if k == 'masked')
    for model in ('b0', 'b1'):
        for kind in ('masked', 'open'):
            for name, d in doc[model][kind].items():
                assert {'n_rows', 'n_design_rows', 'n_unscored', 'nll', 'nll_ci'} <= set(d), name
                if d['n_rows']:
                    assert d['nll'] is not None and d['nll_ci'][0] <= d['nll'] <= d['nll_ci'][1] or d['nll_ci'][0] is None
    assert (
        doc['b2']['n_rows'] > 0
        and 0 <= doc['b2']['brier'] <= 1
        and doc['b2']['brier_ci'][0] <= doc['b2']['brier'] <= doc['b2']['brier_ci'][1]
    )
    with np.load(e2e.run / 'fit' / 'DEV_PRED.npz') as z:
        pred = {k: z[k] for k in z.files}
    n = doc['b2']['n_rows']
    assert (
        sorted(pred) == ['b2_p', 'frame_row', 'match', 'mu_b0', 'mu_b1'] and pred['mu_b1'].shape == (n, 3) and pred['mu_b0'].shape == (n, 3)
    )
    assert (pred['mu_b1'][:, :-1] <= pred['mu_b1'][:, 1:]).all() and (
        (pred['b2_p'] > 0) & (pred['b2_p'] < 1)
    ).all()  # more time, more onsets
    with np.load(e2e.run / 'compose' / 'b0_dev.npz') as z:
        assert np.allclose(pred['mu_b0'], z['expected_onsets'], rtol=1e-9)  # the parent and the workers agree on the rate head


def test_the_gate_file_has_the_four_criteria_the_overdispersion_the_brier_scores_the_tower_diagnostic_and_the_pinned_constants(e2e):
    gate = read_json(e2e.run / 'gate' / 'GATE.json')
    assert {'results', 'verdict', 'gate_pass', 'brier', 'tower_pricing', 'constants'} <= set(gate)
    assert sorted(gate['results']) == ['accounting', 'mc', 'occurrence', 'overdispersion', 'value'] and gate['verdict']['missing'] == []
    assert gate['gate_pass'] == (gate['verdict']['all_pass'] and not gate['verdict']['missing']) and isinstance(gate['gate_pass'], bool)
    c, n = gate['constants'], gate['n_dev_matches_with_rows']
    assert (c['B'], c['seed'], c['S_b0'], c['n_matches']) == (2000, 20261021, 1000, n) and n == 4 == gate['n_dev_matches']
    for name in ('occurrence', 'value'):
        r = gate['results'][name]
        assert (r['B'], r['seed'], r['n_matches']) == (2000, 20261021, n) and r['n_rows'] == c['n_rows']
    assert (
        gate['results']['mc']['S'] == 1000
        and gate['results']['overdispersion']['B'] == 2000
        and gate['results']['accounting']['max_abs_error'] == 0.0
    )
    assert sorted(gate['brier']) == ['B', 'b2', 'difference', 'm2_b0', 'm2_b1', 'n_matches', 'n_rows', 'positive_share', 'seed']
    assert sorted(gate['brier']['difference']) == ['m2_b0_minus_b2', 'm2_b0_minus_m2_b1', 'm2_b1_minus_b2'] and gate['brier']['B'] == 2000
    assert gate['tower_pricing']['window_s'] == 60.0 and gate['tower_pricing']['n_rows'] == c['n_rows']
    assert gate['m2_b0_workers_against_parent_max_relative_difference'] < 1e-9 and gate['wall_s'] >= 0
    assert run_json(e2e)['gate_result']['gate_pass'] == gate['gate_pass'] and run_json(e2e)['s3_result']['gate_pass'] == gate['gate_pass']


def test_the_peak_private_memory_of_every_stage_and_of_the_run_is_recorded_and_under_the_cap(e2e):
    manifest, ledger = run_json(e2e), read_json(e2e.run / 'STAGES.json')
    for stage in STAGES:
        rec = ledger['stages'][stage]
        assert (
            rec['peak_private_gib'] > 0
            and rec['within_memory_cap'] is True
            and manifest[f'{stage}_result']['peak_private_gib'] == rec['peak_private_gib']
        )
    s3 = manifest['s3_result']
    assert (
        s3['peak_private_gib'] == max(s3['stage_peak_private_gib'].values()) > 0
        and s3['within_memory_cap'] is True
        and s3['memory_cap_gib'] == 40.0
    )
    call = ledger['calls'][0]
    assert call['peak_private_gib'] >= s3['peak_private_gib'] - 1e-3 and call['memory_samples'] > 5 and call['check_calls'] > 20


def test_no_match_key_is_in_any_file_of_the_run_or_in_anything_that_was_printed(e2e):
    for path in e2e.run.rglob('*'):
        if path.is_file():
            assert KEY_PREFIX.encode() not in path.read_bytes(), path.name
    assert KEY_PREFIX not in e2e.stdout + e2e.stderr


def copy_of_the_run(e2e, tmp_path) -> Path:
    """A copy of the finished run to damage or resume (the module's run stays as it was)."""
    copy = tmp_path / 'copy' / e2e.run.name
    shutil.copytree(e2e.run, copy)
    return copy


def resume_cli(e2e, tmp_path, run, *extra, records=random_value_record) -> int:
    return run_s3.main(
        cli_args(tmp_path, e2e.world.s1, e2e.world.s2, e2e.world.s3a_run, e2e.world.spec, '--smoke', '12', '--resume', str(run), *extra),
        records=records,
    )


def test_a_resume_of_the_finished_run_runs_nothing_and_changes_no_file(e2e, tmp_path, monkeypatch):
    run = copy_of_the_run(e2e, tmp_path)

    def boom(*args, **kwargs):
        raise AssertionError('a stage ran again')
    for owner, name in (
        (value, 'kill_levels'),
        (value, 'unit_values'),
        (towertier.TierModel, 'fit'),
        (baselines_b0.B0Heads, 'fit'),
        (baselines_b1, 'fit_head'),
        (baselines_b1.B2, 'fit_stores'),
        (compose.Composer, 'simulate'),
        (compose.Composer, 'expect'),
    ):
        monkeypatch.setattr(owner, name, boom)
    before = {p.relative_to(run).as_posix(): store_core.sha(p) for p in run.rglob('*') if p.is_file() and p.name != 'STAGES.json'}
    assert resume_cli(e2e, tmp_path, run) == 0
    after = {p.relative_to(run).as_posix(): store_core.sha(p) for p in run.rglob('*') if p.is_file() and p.name != 'STAGES.json'}
    assert before == after
    calls = read_json(run / 'STAGES.json')['calls']
    assert len(calls) == 2 and calls[1]['stages_run'] == [] and calls[1]['stages_skipped'] == list(STAGES) and calls[1]['resumed'] is True


# ---- a cut during the fit stage --------------------------------------------------------------------------------------------------------
def cut_the_run_before(run: Path, first_stage: str) -> None:
    """Make a copy of a finished run look like a run that was cut while ``first_stage`` was working: that stage and the later ones are not
    complete (their records are gone), the folders of the later ones and the results in RUN.json are removed (the folder of ``first_stage``
    keeps what the cut left, which the caller then damages)."""
    ledger = read_json(run / 'STAGES.json')
    later = STAGES[STAGES.index(first_stage):]
    for stage in later:
        ledger['stages'].pop(stage, None)
    (run / 'STAGES.json').write_text(json.dumps(ledger), encoding='utf-8')
    manifest = read_json(run / 'RUN.json')
    for stage in later:
        manifest.pop(f'{stage}_result', None)
    manifest.pop('s3_result', None)
    (run / 'RUN.json').write_text(json.dumps(manifest), encoding='utf-8')
    for stage in later[1:]:
        shutil.rmtree(run / stage, ignore_errors=True)


def fit_spies(monkeypatch) -> Spy:
    spy = Spy()
    spy_on(monkeypatch, towertier.TierModel, 'fit', spy.tier_fit)
    spy_on(monkeypatch, baselines_b0.B0Heads, 'fit', spy.b0_fit)
    spy_on(monkeypatch, baselines_b1.B2, 'fit_stores', spy.b2_fit)
    real_head = baselines_b1.fit_head
    monkeypatch.setattr(baselines_b1, 'fit_head', lambda *a, **k: (spy.fit_head.append(a[2:4]), real_head(*a, **k))[1])
    monkeypatch.setattr(baselines_b0, 'ALPHA', 1.0)                                          # as the run that is resumed was made
    return spy


def test_a_cut_fit_keeps_the_tier_law_and_refits_b0_whose_npz_has_no_json_and_everything_after_it(e2e, tmp_path, monkeypatch):
    run = copy_of_the_run(e2e, tmp_path)
    cut_the_run_before(run, 'fit')
    shutil.rmtree(run / 'fit' / 'b1')
    shutil.rmtree(run / 'fit' / 'b1_heads')                                                   # no head of B1 was checkpointed either
    shutil.rmtree(run / 'fit' / 'b2')
    (run / 'fit' / 'b0' / 'b0_heads.json').unlink()  # the cut came between the array file and the JSON
    for name in ('FIT_FACTS.json', 'DEV_NLL.json', 'DEV_PRED.npz'):
        (run / 'fit' / name).unlink()
    kept = ('fit/tier/tower_tier.json', 'fit/b0/b0_heads.json', 'fit/b0/b0_heads.npz', 'fit/b1/b1_heads.json', 'fit/b2/b2.json',
            'fit/FIT_INPUTS.json')
    original = {rel: store_core.sha(e2e.run / rel) for rel in kept}
    spy = fit_spies(monkeypatch)
    assert (run / 'fit' / 'b0' / 'b0_heads.npz').exists()
    assert resume_cli(e2e, tmp_path, run, '--stages', 'levels', 'labels', 'fit') == 0
    assert len(spy.tier_fit) == 0                                                           # saved: loaded, not fitted again
    assert len(spy.b0_fit) == 1 and len(spy.fit_head) == 13 and len(spy.b2_fit) == 1        # the NPZ alone did not count
    assert {rel: store_core.sha(run / rel) for rel in original} == original                  # the same fits, byte for byte
    result = read_json(run / 'RUN.json')['fit_result']
    assert result['tier']['loaded'] is True and result['b0']['loaded'] is False and result['b1']['loaded'] is False
    assert result['tier']['wall_s'] > 0 and result['tier']['summary']['classes']                    # its facts come from PROGRESS.jsonl
    ledger = read_json(run / 'STAGES.json')
    assert (
        ledger['stages']['fit']['complete']
        and 'compose' not in ledger['stages']
        and [c['stages_run'] for c in ledger['calls']][-1] == ['fit']
    )


def test_a_cut_fit_keeps_b0_when_its_json_is_there_and_fits_only_what_is_missing(e2e, tmp_path, monkeypatch):
    run = copy_of_the_run(e2e, tmp_path)
    cut_the_run_before(run, 'fit')
    shutil.rmtree(run / 'fit' / 'b1')  # B1 was being fitted (its models are saved last)
    shutil.rmtree(run / 'fit' / 'b1_heads')                                                   # before its first head was checkpointed
    shutil.rmtree(run / 'fit' / 'b2')
    for name in ('FIT_FACTS.json', 'DEV_NLL.json', 'DEV_PRED.npz'):
        (run / 'fit' / name).unlink()
    kept = ('fit/b0/b0_heads.json', 'fit/b1/b1_heads.json', 'fit/b2/b2.json', 'fit/DEV_PRED.npz', 'fit/DEV_NLL.json')
    original = {rel: store_core.sha(e2e.run / rel) for rel in kept}
    spy = fit_spies(monkeypatch)
    assert resume_cli(e2e, tmp_path, run, '--stages', 'levels', 'labels', 'fit') == 0
    assert len(spy.tier_fit) == 0 and len(spy.b0_fit) == 0 and len(spy.fit_head) == 13 and len(spy.b2_fit) == 1
    assert {rel: store_core.sha(run / rel) for rel in original} == original
    result = read_json(run / 'RUN.json')['fit_result']
    assert result['b0']['loaded'] is True and sorted(result['b0']['heads']) == sorted(f'{k}.{n}' for k, n in run_s3._head_names())
    assert result['b0']['heads']['masked.M2Rate']['wall_s'] >= 0 and result['b0']['summary']['masked']  # facts from the log


def test_a_cut_fit_removes_a_b1_without_its_json_and_makes_the_set_again_from_the_head_checkpoints(e2e, tmp_path, monkeypatch):
    run = copy_of_the_run(e2e, tmp_path)
    cut_the_run_before(run, 'fit')
    (run / 'fit' / 'b1' / 'b1_heads.json').unlink()  # the booster texts are there, the JSON (last) is not
    shutil.rmtree(run / 'fit' / 'b2')
    spy = fit_spies(monkeypatch)
    assert any((run / 'fit' / 'b1' / 'models').iterdir())
    assert resume_cli(e2e, tmp_path, run, '--stages', 'levels', 'labels', 'fit') == 0
    assert len(spy.b0_fit) == 0 and (run / 'fit' / 'b1' / 'b1_heads.json').is_file()
    assert len(spy.fit_head) == 0                                                           # every head had its checkpoint: none is fitted
    assert store_core.sha(run / 'fit' / 'b1' / 'b1_heads.json') == store_core.sha(e2e.run / 'fit' / 'b1' / 'b1_heads.json')


# ---- G-S3b B3 item 1: the B1 fit is checkpointed head by head ----------------------------------------------------------------------
REAL_B1_FIT_HEAD = baselines_b1.fit_head                                          # before any test replaces it


def b1_fit_spy(monkeypatch, fail_at=None) -> list:
    """Replace ``baselines_b1.fit_head`` by a wrapper that records the (kind, name) of every head that is fitted and, with ``fail_at``,
    raises when that many heads were fitted (the cut). B0 keeps the ridge of the run that is resumed."""
    seen = []

    def wrapper(fit_rows, stop_rows, name, kind, **kwargs):
        if fail_at is not None and len(seen) == fail_at:
            raise RuntimeError('the fit was cut')
        seen.append((kind, name))
        return REAL_B1_FIT_HEAD(fit_rows, stop_rows, name, kind, **kwargs)
    monkeypatch.setattr(baselines_b1, 'fit_head', wrapper)
    monkeypatch.setattr(baselines_b0, 'ALPHA', 1.0)
    return seen


def cut_the_b1_fit(run: Path, keep_heads: bool = False) -> None:
    """Make a copy of a finished run look cut while B1 was being fitted: no B1 set, nothing after it, and no head checkpoint unless
    ``keep_heads``."""
    cut_the_run_before(run, 'fit')
    for name in ('b1', 'b2') + (() if keep_heads else ('b1_heads',)):
        shutil.rmtree(run / 'fit' / name)
    for name in ('FIT_FACTS.json', 'DEV_NLL.json', 'DEV_PRED.npz'):
        (run / 'fit' / name).unlink()


def b1_files(run: Path) -> dict:
    """{relative path: bytes} of every file of the B1 set that a run saved."""
    root = run / 'fit' / 'b1'
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob('*')) if p.is_file()}


FIT_ARGS = ('--stages', 'levels', 'labels', 'fit')


def test_every_b1_head_is_checkpointed_with_the_code_state_the_inputs_hash_of_the_fit_stage_and_its_time(e2e):
    inputs = read_json(e2e.run / 'STAGES.json')['stages']['fit']['inputs_sha256']
    state = runlog.code_state()
    for kind, name in run_s3._head_names():
        folder = e2e.run / 'fit' / 'b1_heads' / f'{kind}.{name}'
        record = baselines_b1.read_head_record(folder)
        assert (record['code_commit'], record['code_dirty'], record['inputs_sha256']) == (state['code_commit'], state['code_dirty'], inputs)
        assert record['wall_s'] > 0 and record['peak_private_gib'] > 0, (kind, name)
        assert (folder / baselines_b1.HEAD_JSON).is_file()
    assert sorted(p.name for p in (e2e.run / 'fit' / 'b1_heads').iterdir()) == sorted(f'{k}.{n}' for k, n in run_s3._head_names())


def test_a_cut_b1_fit_keeps_the_heads_that_were_saved_fits_only_the_rest_and_ends_in_the_files_of_an_uncut_run(e2e, tmp_path, monkeypatch):
    names, k = run_s3._head_names(), 5
    run = copy_of_the_run(e2e, tmp_path)
    cut_the_b1_fit(run)
    seen = b1_fit_spy(monkeypatch, fail_at=k)
    assert resume_cli(e2e, tmp_path, run, *FIT_ARGS) == 1                                      # the sixth head raises: an unexpected error
    assert seen == names[:k] and not (run / 'fit' / 'b1').exists()
    saved = sorted(p.name for p in (run / 'fit' / 'b1_heads').iterdir() if (p / baselines_b1.HEAD_JSON).is_file())
    assert saved == sorted(f'{kind}.{name}' for kind, name in names[:k])                          # a checkpoint after every head
    seen_again = b1_fit_spy(monkeypatch)
    assert resume_cli(e2e, tmp_path, run, *FIT_ARGS) == 0
    assert seen_again == names[k:]                                                            # the heads 1 .. k are not fitted again
    assert b1_files(run) == b1_files(e2e.run)                                                 # the folder of an uncut run, byte for byte
    heads = read_json(run / 'RUN.json')['fit_result']['b1']['heads']
    assert [heads[f'{kind}.{name}']['loaded'] for kind, name in names] == [True] * k + [False] * (len(names) - k)


@pytest.mark.parametrize('field, other', [('inputs_sha256', 'f' * 64), ('code_commit', 'e' * 40), ('code_dirty', 'unknown')])
def test_a_b1_head_saved_by_other_code_or_inputs_is_refused_and_nothing_is_fitted_again(e2e, tmp_path, monkeypatch, capsys, field, other):
    run = copy_of_the_run(e2e, tmp_path)
    cut_the_b1_fit(run, keep_heads=True)
    path = run / 'fit' / 'b1_heads' / 'masked.M2Mark' / baselines_b1.HEAD_JSON
    doc = read_json(path)
    doc['record'][field] = other
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    seen = b1_fit_spy(monkeypatch)
    assert resume_cli(e2e, tmp_path, run, *FIT_ARGS) == 2
    error = capsys.readouterr().err
    assert 'masked.M2Mark' in error and field in error and 'new run folder' in error
    assert seen == [] and not (run / 'fit' / 'b1').exists()


def test_a_head_is_logged_before_its_checkpoint_is_written_so_that_a_checkpoint_always_has_its_log_line(e2e, tmp_path, monkeypatch):
    names, k = run_s3._head_names(), 3
    run = copy_of_the_run(e2e, tmp_path)
    cut_the_b1_fit(run)
    log = run / 'fit' / 'PROGRESS.jsonl'
    old_lines = len(log.read_text(encoding='utf-8').splitlines())                          # the lines of the run it was copied from
    real, saved = baselines_b1.save_head, []

    def save_head(head, kind, name, *args, **kwargs):
        if len(saved) == k:
            raise RuntimeError('the call was cut while the fourth checkpoint was being written')
        saved.append((kind, name))
        return real(head, kind, name, *args, **kwargs)
    monkeypatch.setattr(baselines_b1, 'save_head', save_head)
    seen = b1_fit_spy(monkeypatch)
    assert resume_cli(e2e, tmp_path, run, *FIT_ARGS) == 1
    assert seen == names[:k + 1] and saved == names[:k]                                    # the fourth head was fitted, its checkpoint cut
    new_units = [json.loads(line)['unit'] for line in log.read_text(encoding='utf-8').splitlines()[old_lines:]]
    assert [u for u in new_units if u.startswith('b1.')] == [f'b1.{kind}.{name}' for kind, name in names[:k + 1]]      # logged, then cut
    done = sorted(p.name for p in (run / 'fit' / 'b1_heads').iterdir() if (p / baselines_b1.HEAD_JSON).is_file())
    assert done == sorted(f'{kind}.{name}' for kind, name in names[:k])                    # a checkpoint has its line, a line need not
    monkeypatch.setattr(baselines_b1, 'save_head', real)
    seen = b1_fit_spy(monkeypatch)
    assert resume_cli(e2e, tmp_path, run, *FIT_ARGS) == 0 and seen == names[k:]
    assert b1_files(run) == b1_files(e2e.run)


def test_a_b1_head_checkpoint_without_its_json_is_removed_and_only_that_head_is_fitted_again(e2e, tmp_path, monkeypatch):
    run = copy_of_the_run(e2e, tmp_path)
    cut_the_b1_fit(run, keep_heads=True)
    victim = run / 'fit' / 'b1_heads' / 'open.M4Conv'
    (victim / baselines_b1.HEAD_JSON).unlink()                                                # the cut came between the texts and the JSON
    assert any(victim.iterdir())
    seen = b1_fit_spy(monkeypatch)
    assert resume_cli(e2e, tmp_path, run, *FIT_ARGS) == 0
    assert seen == [('open', 'M4Conv')] and (victim / baselines_b1.HEAD_JSON).is_file()
    assert b1_files(run) == b1_files(e2e.run)


# ---- the stages on their own, with fakes for what they call ----------------------------------------------------------------------------
@contextlib.contextmanager
def context_for(run: Path, world, smoke=12, workers=8, records=None):
    """A ``Context`` on the random world (the smoke subset of ``smoke``, or the whole store) for a run folder."""
    s1 = run_s1.load_s1(world.s1, verify=False)
    s2 = flowprice._open_s2(world.s2, False)
    selection = run_s3.select_smoke(s1, smoke) if smoke else None
    s3a = run_s3.check_s3a(world.s3a_run, run_s3.read_decisions(world.spec))
    ctx = run_s3.Context(run=run, s1=run_s3.SubsetStore(s1, selection) if selection else s1, s2=s2, prices=value.load_prices(s3a['prices']),
                         rules=rulepay.RuleTable.load(s3a['rules'], patches=rulepay.P3_PATCHES),
                         hashes=run_s3._base_hashes(s1, s2, s3a, selection), smoke=selection, workers=workers, records=records,
                         out=lambda message, flush=True: None)
    try:
        yield ctx
    finally:
        release(s1, s2)


def test_write_json_looks_for_match_keys_in_the_text_and_writes_nothing_when_one_is_there(world, tmp_path):
    with context_for(tmp_path / 'run', world, smoke=None) as ctx:
        key = bytes(np.asarray(ctx.s1.array('matches/key'))[0]).decode('ascii')
        path = tmp_path / 'run' / 'x.json'
        with pytest.raises(RuntimeError, match='match keys occur') as raised:
            ctx.write_json(path, {'note': f'see {key}'})
        assert key not in str(raised.value) and not path.exists()
        digest = ctx.write_json(path, {'note': 'no key in this text'})
        assert read_json(path) == {'note': 'no key in this text'} and digest == store_core.sha(path)


def test_the_levels_stage_reads_fit_stop_and_dev_with_the_workers_and_the_records_it_is_given_and_saves_the_levels_with_their_hash(
        world, tmp_path, monkeypatch):
    asked = []

    def fake(s1, roles=None, workers=8, *, records=None, report=None):
        asked.append((roles, workers, records))
        report.update(matches=3, unreadable=0, kills_read=5, kills_unread=0)
        return np.array([-1, 3, 18, 1, -1], dtype=np.int16)
    monkeypatch.setattr(value, 'kill_levels', fake)
    with context_for(tmp_path / 'run', world, smoke=None, workers=5, records=random_value_record) as ctx:
        result = run_s3.stage_levels(ctx)
    assert asked == [(('FIT', 'STOP', 'DEV'), 5, random_value_record)]
    path = tmp_path / 'run' / 'levels' / 'kill_levels.npy'
    levels = np.load(path)
    assert levels.dtype == np.int16 and levels.tolist() == [-1, 3, 18, 1, -1]
    doc = read_json(tmp_path / 'run' / 'levels' / 'LEVELS.json')
    assert doc['kill_levels_sha256'] == store_core.sha(path) == result['kill_levels_sha256'] and doc['n_valued_kills'] == 3
    assert (doc['level_min'], doc['level_max'], doc['n_kills']) == (1, 18, 5) and doc['roles'] == ['FIT', 'STOP', 'DEV']


def test_the_levels_stage_without_records_uses_the_real_reader_and_an_unreadable_file_fails_the_stage(world, tmp_path, monkeypatch):
    seen = []

    def fake(s1, roles=None, workers=8, **kwargs):
        seen.append(kwargs)
        kwargs['report'].update(matches=4, unreadable=2, kills_read=1, kills_unread=7)
        return np.array([1], dtype=np.int16)
    monkeypatch.setattr(value, 'kill_levels', fake)
    with context_for(tmp_path / 'run', world, smoke=None) as ctx:
        with pytest.raises(run_s3.StageFailed, match='2 of 4 raw match files'):
            run_s3.stage_levels(ctx)
    assert 'records' not in seen[0] and not (tmp_path / 'run' / 'levels' / 'kill_levels.npy').exists()


def made_levels(world, tmp_path, smoke=12) -> Path:
    """A run folder whose ``levels`` stage ran with the real functions."""
    run = tmp_path / 'run'
    with context_for(run, world, smoke=smoke, workers=1, records=random_value_record) as ctx:
        run_s3.stage_levels(ctx)
    return run


def test_the_labels_stage_runs_unit_values_v_start_accounting_gold_bin_means_and_write_labels_in_that_order(world, tmp_path, monkeypatch):
    order = []

    def spy_name(name):
        real = getattr(value, name)
        monkeypatch.setattr(value, name, lambda *a, **k: (order.append(name), real(*a, **k))[1])
    for name in ('unit_values', 'v_start', 'accounting', 'gold_bin_means', 'write_labels'):
        spy_name(name)
    run = made_levels(world, tmp_path)
    with context_for(run, world, smoke=12) as ctx:
        run_s3.stage_labels(ctx)
    assert order == ['unit_values', 'v_start', 'accounting', 'gold_bin_means', 'write_labels']


def test_the_labels_stage_fails_when_the_accounting_identity_is_not_exactly_zero_and_writes_no_store(world, tmp_path, monkeypatch):
    real = value.accounting

    def inexact(*args, **kwargs):
        acc = real(*args, **kwargs)
        acc['max_abs_error'] = 1.0
        return acc
    monkeypatch.setattr(value, 'accounting', inexact)
    run = made_levels(world, tmp_path)
    with context_for(run, world, smoke=12) as ctx:
        with pytest.raises(run_s3.StageFailed, match='not exact'):
            run_s3.stage_labels(ctx)
    assert not (run / 'labels' / 'store').exists() and not (run / 'labels' / 'ACCOUNTING.json').exists()


def test_a_tiny_error_in_the_accounting_is_not_zero_either(world, tmp_path, monkeypatch):
    real = value.accounting
    monkeypatch.setattr(value, 'accounting', lambda *a, **k: {**real(*a, **k), 'max_abs_error': 1e-12})
    run = made_levels(world, tmp_path)
    with context_for(run, world, smoke=12) as ctx:
        with pytest.raises(run_s3.StageFailed, match='not exact'):
            run_s3.stage_labels(ctx)


def test_a_match_key_in_the_provenance_of_the_labels_is_an_error_and_nothing_is_written(world, tmp_path, monkeypatch):
    s1 = run_s1.load_s1(world.s1, verify=False)
    key = bytes(np.asarray(s1.array('matches/key'))[3]).decode('ascii')
    release(s1)
    monkeypatch.setattr(runlog, 'code_state', lambda: {'code_commit': f'commit of {key}', 'code_dirty': False})
    run = made_levels(world, tmp_path)
    with context_for(run, world, smoke=12) as ctx:
        with pytest.raises(RuntimeError, match='1 match keys occur') as raised:
            run_s3.stage_labels(ctx)
    assert KEY_PREFIX not in str(raised.value) and not (run / 'labels' / 'store').exists()


def test_the_labels_stage_replaces_a_half_written_store_of_an_earlier_call(world, tmp_path):
    run = made_levels(world, tmp_path)
    half = run / 'labels' / 'store'
    (half / 'labels').mkdir(parents=True)
    (half / 'labels' / 'kill_gold.npy').write_bytes(b'half an array')
    (half / 'STORE.json').write_text(
        json.dumps({'version': 1, 'stages': {}, 'arrays': {}, 'files': {}, 'complete': False}), encoding='utf-8'
    )
    with context_for(run, world, smoke=12) as ctx:
        result = run_s3.stage_labels(ctx)
    store = Store(half)
    store.verify()
    assert store.manifest['complete'] and result['max_abs_error'] == 0.0
    release(store)


class FakeComposer:
    """Stands in for ``compose.Composer``: values are functions of the frame row, every call is recorded and the E6 ``check`` is called
    before it, as the real one calls it before each block. ``scale[kind]`` multiplies the expected onsets (kind: 'b0' or 'b1')."""
    made = []
    scale = {'b0': 1.0, 'b1': 1.0}

    def __init__(self, heads, prices, tier_model, gold_reps, rows, s2, S=1000, seed=20261021):
        self.heads, self.prices, self.tier_model, self.gold_reps, self.rows, self.S, self.seed = (
            heads,
            prices,
            tier_model,
            gold_reps,
            rows,
            S,
            seed,
        )
        self.calls = []
        FakeComposer.made.append(self)

    def _log(self, method, idx, block, workers, check):
        if check is not None:
            check()
        self.calls.append({'method': method, 'idx': np.asarray(idx), 'block': block, 'workers': workers, 'check': check})
        return np.asarray(self.rows.frame_row)[np.asarray(idx)].astype(np.float64)

    def expected_onsets(self, idx, block=16, workers=1, check=None):
        base = self._log('expected_onsets', idx, block, workers, check)
        return np.column_stack([base * 1e-3 * k for k in (1, 2, 3)]) * FakeComposer.scale[self.heads.feature_kind]

    def expect(self, idx, block=16, workers=1, check=None):
        base = self._log('expect', idx, block, workers, check)
        return np.column_stack([base + k for k in (0, 1, 2)])

    def simulate(self, idx, block=16, workers=1, check=None):
        base = self._log('simulate', idx, block, workers, check)
        mean = np.column_stack([base + 0.5 * k for k in (0, 1, 2)])
        return compose.SimResult(mean=mean, sd=np.ones_like(mean), q19=mean[:, :, None] + np.arange(19)[None, None, :] / 10.0, S=self.S)


@pytest.fixture
def fake_composer(monkeypatch):
    FakeComposer.made, FakeComposer.scale = [], {'b0': 1.0, 'b1': 1.0}
    monkeypatch.setattr(compose, 'Composer', FakeComposer)
    return FakeComposer


def test_the_compose_stage_screens_b0_then_b1_then_composes_b0_then_b1_with_the_blocks_the_workers_and_the_guard_of_the_plan(
        e2e, tmp_path, fake_composer):
    run = copy_of_the_run(e2e, tmp_path)
    with context_for(run, e2e.world, smoke=12, workers=8) as ctx:
        result = run_s3.stage_compose(ctx)
        guard = ctx.guard
    b0, b1 = fake_composer.made
    assert (
        b0.heads.feature_kind == 'b0' and b1.heads.feature_kind == 'b1' and (b0.S, b1.S) == (1000, 200) and b0.seed == b1.seed == 20261021
    )
    assert [c['method'] for c in b0.calls] == ['expected_onsets', 'expect', 'simulate'] and [c['method'] for c in b1.calls] == [
        'expected_onsets',
        'simulate',
    ]
    assert [c['block'] for c in b0.calls] == [4096, 64, 512] and [c['block'] for c in b1.calls] == [4096, 256]
    assert all(c['workers'] == 8 and c['check'] is guard for c in b0.calls + b1.calls)
    assert [(c['head_set'], c['method']) for c in result['calls']] == [
        ('B0', 'expected_onsets'),
        ('B1', 'expected_onsets'),
        ('B0', 'expect'),
        ('B0', 'simulate'),
        ('B1', 'simulate'),
    ]
    n = len(b0.rows)
    assert all(len(c['idx']) == n for c in b0.calls) and np.array_equal(b0.calls[0]['idx'], np.arange(n))
    assert np.array_equal(b0.rows.match, b1.rows.match) and len(b1.rows) == n                      # every smoke DEV match is composed by B1


@pytest.mark.parametrize('given, used', [(1, 2), (2, 2), (3, 3), (8, 8), (61, 61)])
def test_the_compose_stage_always_computes_in_at_least_two_workers(e2e, tmp_path, fake_composer, given, used):
    run = copy_of_the_run(e2e, tmp_path)
    with context_for(run, e2e.world, smoke=12, workers=given) as ctx:
        assert ctx.compose_workers == used
        run_s3.stage_compose(ctx)
    assert {c['workers'] for comp in fake_composer.made for c in comp.calls} == {used}


def test_the_compose_stage_never_calls_simulate_onsets_for_a_reported_value(e2e, tmp_path, fake_composer, monkeypatch):
    def never(self, row):
        raise AssertionError('simulate_onsets has no workers')
    monkeypatch.setattr(FakeComposer, 'simulate_onsets', never, raising=False)
    run = copy_of_the_run(e2e, tmp_path)
    with context_for(run, e2e.world, smoke=12) as ctx:
        run_s3.stage_compose(ctx)
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr == 'simulate_onsets']


def test_the_r31_screen_stops_the_run_when_a_row_has_more_than_50_expected_onsets_in_180_s_and_names_the_head_set_and_the_rows(
        e2e, tmp_path, fake_composer):
    FakeComposer.scale['b0'] = 40.0
    run = copy_of_the_run(e2e, tmp_path)
    with context_for(run, e2e.world, smoke=12) as ctx:
        rows = compose.row_table(ctx.s1, ctx.s2, ('DEV',))
        over = int(((np.asarray(rows.frame_row) * 1e-3 * 3 * 40.0) > 50.0).sum())
        assert 0 < over < len(rows)
        with pytest.raises(run_s3.StageFailed) as raised:
            run_s3.stage_compose(ctx)
    text = str(raised.value)
    assert 'R31' in text and 'head set B0' in text and f'on {over} of {len(rows)} rows' in text and 'nothing is composed' in text
    assert [c['method'] for c in fake_composer.made[0].calls] == ['expected_onsets'] and len(
        fake_composer.made
    ) == 1  # nothing else was asked


def test_the_r31_screen_looks_at_b1_too(e2e, tmp_path, fake_composer):
    FakeComposer.scale['b1'] = 1e6
    run = copy_of_the_run(e2e, tmp_path)
    with context_for(run, e2e.world, smoke=12) as ctx:
        with pytest.raises(run_s3.StageFailed, match='head set B1'):
            run_s3.stage_compose(ctx)
    assert [c['method'] for c in fake_composer.made[0].calls] == ['expected_onsets'] and [
        c['method'] for c in fake_composer.made[1].calls
    ] == ['expected_onsets']


def test_the_r31_limit_is_a_strict_50_in_180_s(e2e, tmp_path, fake_composer):
    run = copy_of_the_run(e2e, tmp_path)
    with context_for(run, e2e.world, smoke=12) as ctx:
        rows = compose.row_table(ctx.s1, ctx.s2, ('DEV',))
        top = float(np.asarray(rows.frame_row).max()) * 1e-3 * 3
        FakeComposer.scale['b0'] = FakeComposer.scale['b1'] = 50.0 / top                       # the largest row is exactly at the limit
        run_s3.stage_compose(ctx)                                                               # not more than 50: no stop
        FakeComposer.scale['b0'] = FakeComposer.scale['b1'] = 50.0 / top * (1 + 1e-9)
        with pytest.raises(run_s3.StageFailed, match='R31'):
            run_s3.stage_compose(ctx)


def test_the_compose_stage_saves_chunks_and_a_resume_keeps_the_ones_that_are_saved(e2e, tmp_path, fake_composer, monkeypatch):
    monkeypatch.setattr(run_s3, 'CHUNK_ROWS', 40)
    clean = copy_of_the_run(e2e, tmp_path / 'clean')
    with context_for(clean, e2e.world, smoke=12) as ctx:
        run_s3.stage_compose(ctx)
    n = len(fake_composer.made[0].rows)
    n_chunks = -(-n // 40)
    assert n_chunks >= 3 and [c['method'] for c in fake_composer.made[0].calls].count('expect') == n_chunks
    with np.load(clean / 'compose' / 'b0_dev.npz') as z:
        reference = {k: z[k] for k in z.files}
    assert not (clean / 'compose' / 'parts').exists()
    fake_composer.made.clear()
    cut = copy_of_the_run(e2e, tmp_path / 'cut')
    state = {'checks': 0}
    real_e6 = run_s1.require_no_strata

    def guard_that_fails_in_the_third_chunk():
        state['checks'] += 1
        if state['checks'] == 2 + 2 * 2 + 1:  # two screens, then B0 chunk 0 (expect, simulate), chunk 1 (expect, simulate), then chunk 2
            raise run_s1.StrataRunning('E6: a Strata process started meanwhile')
    with context_for(cut, e2e.world, smoke=12) as ctx:
        monkeypatch.setattr(run_s1, 'require_no_strata', guard_that_fails_in_the_third_chunk)
        with pytest.raises(run_s1.StrataRunning):
            run_s3.stage_compose(ctx)
    saved = sorted(p.name for p in (cut / 'compose' / 'parts').iterdir())
    assert saved == ['B0_00000.npz', 'B0_00001.npz']                                          # two chunks of B0 were saved before the stop
    monkeypatch.setattr(run_s1, 'require_no_strata', real_e6)
    fake_composer.made.clear()
    with context_for(cut, e2e.world, smoke=12) as ctx:
        run_s3.stage_compose(ctx)
    b0 = fake_composer.made[0]
    asked = [c for c in b0.calls if c['method'] in ('expect', 'simulate')]
    assert (
        min(int(c['idx'].min()) for c in asked) == 80 and sum(c['method'] == 'expect' for c in asked) == n_chunks - 2
    )  # chunks 0 and 1: kept
    with np.load(cut / 'compose' / 'b0_dev.npz') as z:
        assert all(np.array_equal(z[k], reference[k]) for k in reference)                    # the same as a run that was not cut
    assert not (cut / 'compose' / 'parts').exists()


# ---- G-S3b B3 item 5: the counters of COMPOSE.json cover the stage and the work of this call ---------------------------------------
CALL_CHECK_KEYS = ('call_check_calls', 'call_check_s_total', 'call_check_s_mean', 'call_check_s_longest', 'call_check_budget_s')
OLD_CALL_KEYS = ('check_calls', 'check_s_total', 'check_s_mean', 'check_budget_s')
COUNTERS = ('chunks_total', 'chunks_loaded', 'chunks_computed', 'rows_computed')


def test_a_clean_compose_computes_every_chunk_and_counts_the_chunks_and_rows_by_head_set(e2e, tmp_path, fake_composer, monkeypatch):
    monkeypatch.setattr(run_s3, 'CHUNK_ROWS', 40)
    run = copy_of_the_run(e2e, tmp_path)
    with context_for(run, e2e.world, smoke=12) as ctx:
        result = run_s3.stage_compose(ctx)
    b0, b1 = fake_composer.made
    chunks = {'B0': -(-len(b0.rows) // 40), 'B1': -(-len(b1.rows) // 40)}
    rows = {'B0': len(b0.rows), 'B1': len(b1.rows)}
    assert chunks['B0'] >= 3
    by_set = {name: {'chunks_total': chunks[name], 'chunks_loaded': 0, 'chunks_computed': chunks[name], 'rows_computed': rows[name]}
              for name in ('B0', 'B1')}
    assert result['chunks_by_head_set'] == by_set
    assert {k: result[k] for k in COUNTERS} == {'chunks_total': sum(chunks.values()), 'chunks_loaded': 0,
                                                'chunks_computed': sum(chunks.values()), 'rows_computed': sum(rows.values())}
    assert read_json(run / 'compose' / 'COMPOSE.json')['chunks_by_head_set'] == by_set


def test_a_resumed_compose_reports_the_chunks_it_loaded_and_counts_only_the_work_it_did(e2e, tmp_path, fake_composer, monkeypatch):
    monkeypatch.setattr(run_s3, 'CHUNK_ROWS', 40)
    cut = copy_of_the_run(e2e, tmp_path)
    state, real_e6 = {'checks': 0}, run_s1.require_no_strata

    def guard_that_fails_in_the_third_chunk():
        state['checks'] += 1
        if state['checks'] == 2 + 2 * 2 + 1:                         # two screens, B0 chunks 0 and 1 (expect, simulate each), then chunk 2
            raise run_s1.StrataRunning('E6: a Strata process started meanwhile')
    with context_for(cut, e2e.world, smoke=12) as ctx:
        monkeypatch.setattr(run_s1, 'require_no_strata', guard_that_fails_in_the_third_chunk)
        with pytest.raises(run_s1.StrataRunning):
            run_s3.stage_compose(ctx)
    assert sorted(p.name for p in (cut / 'compose' / 'parts').iterdir()) == ['B0_00000.npz', 'B0_00001.npz']
    fake_composer.made.clear()

    def slow_check():                                                  # a check before the stage that is slower than any in the stage
        time.sleep(0.05)
    with context_for(cut, e2e.world, smoke=12) as ctx:
        monkeypatch.setattr(run_s1, 'require_no_strata', slow_check)
        ctx.guard()
        ctx.guard()
        monkeypatch.setattr(run_s1, 'require_no_strata', real_e6)
        before, seconds_before = ctx.guard.calls, ctx.guard.seconds
        result = run_s3.stage_compose(ctx)
        made = ctx.guard.calls - before
    b0, b1 = fake_composer.made
    chunks = {'B0': -(-len(b0.rows) // 40), 'B1': -(-len(b1.rows) // 40)}
    assert result['chunks_by_head_set'] == {
        'B0': {'chunks_total': chunks['B0'], 'chunks_loaded': 2, 'chunks_computed': chunks['B0'] - 2, 'rows_computed': len(b0.rows) - 80},
        'B1': {'chunks_total': chunks['B1'], 'chunks_loaded': 0, 'chunks_computed': chunks['B1'], 'rows_computed': len(b1.rows)}}
    assert result['chunks_loaded'] == 2 and result['chunks_loaded'] + result['chunks_computed'] == result['chunks_total']
    assert result['rows_computed'] == len(b0.rows) - 80 + len(b1.rows)
    assert result['check_calls'] == made and 0 < made < ctx.guard.calls                         # this stage only, not the two before it
    assert result['check_s_longest'] < 0.04 <= ctx.guard.longest                               # nor the slow check of the call before
    assert result['check_s_total'] == pytest.approx(ctx.guard.seconds - seconds_before, abs=1e-3)
    assert result['check_s_mean'] == pytest.approx(result['check_s_total'] / made, abs=1e-3)


def test_every_compose_call_keeps_its_own_counters_under_per_call_names(e2e, tmp_path, fake_composer, monkeypatch):
    run = copy_of_the_run(e2e, tmp_path)
    with context_for(run, e2e.world, smoke=12) as ctx:
        result = run_s3.stage_compose(ctx)
    assert len(result['calls']) == 5
    for call in result['calls']:
        assert all(key in call for key in CALL_CHECK_KEYS) and not any(key in call for key in OLD_CALL_KEYS), call
        assert call['call_check_calls'] >= 1 and call['call_check_s_longest'] <= call['call_check_s_total'] + 1e-4
    assert sum(call['call_check_calls'] for call in result['calls']) <= result['check_calls']      # and checks between the calls
    assert result['check_s_longest'] >= result['check_s_mean'] >= 0 and result['check_calls'] >= 5


def test_a_scoped_guard_window_counts_the_checks_made_while_it_is_open_only(monkeypatch):
    guard = run_s3._Guard()
    monkeypatch.setattr(run_s1, 'require_no_strata', lambda: time.sleep(0.03))
    guard()
    with guard.window() as outer:
        guard()
        with guard.window() as inner:
            guard()
        monkeypatch.setattr(run_s1, 'require_no_strata', lambda: None)
        guard()
    guard()
    assert guard.calls == 5
    assert outer.counters()['check_calls'] == 3 and inner.counters()['check_calls'] == 1
    assert 0.025 <= inner.counters()['check_s_longest'] and outer.counters()['check_s_longest'] >= inner.counters()['check_s_longest']
    assert outer.counters('call_')['call_check_calls'] == 3 and guard._windows == []                    # frozen when closed, none left open
    with run_s3._Guard().window() as empty:
        assert empty.counters()['check_s_mean'] is None and empty.counters()['check_calls'] == 0         # no check: no mean


def test_a_saved_chunk_made_with_another_simulation_count_seed_or_rows_is_not_used(tmp_path):
    parts = tmp_path / 'parts'
    parts.mkdir()
    frame_row = np.arange(100, 140)
    good = {
        'frame_row': frame_row,
        'S': np.array(1000),
        'seed': np.array(20261021),
        'expected': np.zeros((40, 3)),
        'mc_mean': np.zeros((40, 3)),
        'mc_sd': np.zeros((40, 3)),
        'q19': np.zeros((40, 3, 19)),
    }
    ctx = run_s3.Context(run=tmp_path, s1=None, s2=None, prices={}, rules=None, hashes={})
    load = run_s3._load_chunk
    ctx.write_npz(parts / 'B0_00000.npz', good)
    assert load(parts / 'B0_00000.npz', frame_row, 1000, 20261021) is not None
    for field, bad in (('S', np.array(500)), ('seed', np.array(7)), ('frame_row', frame_row + 1)):
        ctx.write_npz(parts / 'B0_00000.npz', {**good, field: bad})
        assert load(parts / 'B0_00000.npz', frame_row, 1000, 20261021) is None, field
    ctx.write_npz(parts / 'B0_00000.npz', {k: v for k, v in good.items() if k != 'S'})
    assert load(parts / 'B0_00000.npz', frame_row, 1000, 20261021) is None
    (parts / 'B0_00000.npz').write_bytes(b'not a zip')
    assert load(parts / 'B0_00000.npz', frame_row, 1000, 20261021) is None
    assert load(parts / 'missing.npz', frame_row, 1000, 20261021) is None


def test_the_chunks_of_rows_cover_every_row_once(monkeypatch):
    assert run_s3._chunk_ranges(0) == [] and run_s3._chunk_ranges(5, 8) == [(0, 5)] and run_s3._chunk_ranges(16, 8) == [(0, 8), (8, 16)]
    assert run_s3._chunk_ranges(17, 8) == [(0, 8), (8, 16), (16, 17)]
    monkeypatch.setattr(run_s3, 'CHUNK_ROWS', 3)
    assert run_s3._chunk_ranges(7) == [(0, 3), (3, 6), (6, 7)]


def test_a_saved_array_of_one_element_is_read_as_a_whole_number_and_anything_else_is_none(recwarn):
    assert run_s3._scalar(np.array(1000)) == 1000 and run_s3._scalar(np.array([1000])) == 1000 and run_s3._scalar(np.int64(7)) == 7
    assert (
        run_s3._scalar(None) is None and run_s3._scalar(np.array([1, 2])) is None and run_s3._scalar(np.array([], dtype=np.int64)) is None
    )
    assert run_s3._scalar(np.array([1.0])) is None and run_s3._scalar(np.array(['a'])) is None
    assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]               # int() of a 1-d array is deprecated


def test_the_npz_writer_saves_a_zero_dimensional_array_with_the_shape_one_and_scalar_reads_it(tmp_path):
    path = tmp_path / 'x.npz'
    path.write_bytes(run_s3._npz_bytes({'S': np.array(1000), 'rows': np.arange(3)}))
    with np.load(path, allow_pickle=False) as z:
        assert z['S'].shape == (1,) and run_s3._scalar(z['S']) == 1000                           # the reason for ``_scalar``


def test_the_b1_sample_is_all_the_dev_matches_in_a_smoke_run_and_a_seeded_500_otherwise():
    rows = types.SimpleNamespace(match=np.repeat(np.arange(700), 3))
    small = types.SimpleNamespace(match=np.repeat(np.arange(300), 2))
    assert run_s3.b1_sample(small, smoke=False).tolist() == list(range(300)) and run_s3.b1_sample(rows, smoke=True).tolist() == list(
        range(700)
    )
    pick = run_s3.b1_sample(rows, smoke=False)
    assert len(pick) == 500 and pick.tolist() == sorted(pick.tolist()) and len(set(pick.tolist())) == 500 and pick.max() < 700
    assert pick.tolist() == run_s3.b1_sample(rows, smoke=False).tolist()                      # fixed by the seed
    assert pick.tolist() != run_s3.b1_sample(rows, smoke=False, seed=1).tolist() and pick.tolist() != list(range(500))
    assert run_s3.b1_sample(types.SimpleNamespace(match=np.repeat(np.arange(500), 2)), smoke=False).tolist() == list(range(500))


def test_b1_is_composed_on_the_seeded_sample_when_there_are_more_dev_matches_than_the_limit(e2e, tmp_path, fake_composer, monkeypatch):
    run = copy_of_the_run(e2e, tmp_path)
    monkeypatch.setattr(run_s3, 'B1_DEV_MATCHES', 2)                                    # the limit is read when the stage runs
    with context_for(run, e2e.world, smoke=None, workers=2) as ctx:                     # a full run: the DEV matches are not a smoke subset
        result = run_s3.stage_compose(ctx)
    b0, b1 = fake_composer.made
    dev = np.unique(b0.rows.match)
    sample = np.unique(b1.rows.match)
    assert len(dev) > 2 and result['matches'] == len(dev) and result['rows'] == len(b0.rows)
    assert len(sample) == 2 and np.isin(sample, dev).all() and np.array_equal(result['b1_matches'], sample)
    assert len(b1.rows) == int(np.isin(b0.rows.match, sample).sum()) < len(b0.rows) == result['rows'] and result['b1_rows'] == len(b1.rows)
    assert result['b1_sample'] == 'a seeded sample of 2 DEV matches'
    assert all(len(c['idx']) == len(b1.rows) for c in b1.calls) and all(len(c['idx']) == len(b0.rows) for c in b0.calls)
    with np.load(run / 'compose' / 'b1_dev.npz') as z:                                  # the file holds the rows of the sample only
        assert np.array_equal(z['frame_row'], np.asarray(b1.rows.frame_row)) and np.array_equal(np.unique(z['match']), sample)


def test_subset_rows_keeps_the_rows_where_the_mask_is_true_in_their_order_with_every_field():
    rows = s3_toys.toy_row_table()
    keep = np.array([True, False, True, True, False])
    sub = run_s3.subset_rows(rows, keep)
    assert len(sub) == 3 and sub.roles == rows.roles
    for name in ('frame_row', 'match', 't0_ms', 'exposure_ms', 'T_ms', 'views', 'destroyed'):
        assert np.array_equal(getattr(sub, name), getattr(rows, name)[keep]), name


# ---- the gate: arrays in, the verdict of D6 out ----------------------------------------------------------------------------------------
def gate_inputs(
    seed=3, n_matches=1500, rows=6, occ_scale=1.0, value_bias=0.0, mc_shift=0.0, od_scale=1.0, od_noise=0.0, **override
) -> dict:
    """Arrays for ``gate_results`` that satisfy the four criteria of P7 by construction (observed Poisson with the expected mean, a
    realised value that is the expected one plus noise, a simulation that agrees with the closed form to within its standard error, no
    accounting error); the arguments bend one of them."""
    rng = np.random.default_rng(seed)
    match = np.repeat(np.arange(n_matches), rows)
    n = len(match)
    mu = rng.uniform(0.5, 3.0, n)                                   # the expected onsets of B0 in the first 60 s
    luck = rng.gamma(1 / od_noise, od_noise, n) if od_noise else np.ones(n)             # a gamma mixture makes the counts over-dispersed
    y = rng.poisson(mu * occ_scale * luck)
    ev = rng.normal(0.0, 300.0, n)                                  # the closed-form expected value of B0
    sd = np.full(n, 500.0)
    out = dict(match=match, y60=y, expected_onsets60=mu, expected_value60=ev, realised_value60=ev + value_bias + rng.normal(0.0, 800.0, n),
               mc_mean60=ev + mc_shift + rng.normal(0.0, 1.0, n) * sd / np.sqrt(1000.0), mc_sd60=sd, mu_b1_60=mu * od_scale, mu_b0_60=mu,
               b2_p=np.clip(1 - np.exp(-mu), 0.01, 0.99), accounting_error=0.0, n_matches=n_matches)
    out.update(override)
    return out


@pytest.fixture(scope='module')
def passing_gate():
    return run_s3.gate_results(**gate_inputs())


def test_gate_pass_needs_every_criterion_judged_to_pass_and_none_of_the_four_missing_ruling_r27():
    assert run_s3.gate_pass({'all_pass': True, 'missing': []}) is True
    assert run_s3.gate_pass({'all_pass': True, 'missing': ['mc']}) is False  # all_pass is only over the criteria that are there
    assert (
        run_s3.gate_pass({'all_pass': False, 'missing': []}) is False
        and run_s3.gate_pass({'all_pass': False, 'missing': ['value']}) is False
    )


def test_arrays_that_satisfy_the_four_criteria_pass_the_gate(passing_gate):
    g = passing_gate
    assert g['gate_pass'] is True and g['verdict']['all_pass'] is True and g['verdict']['missing'] == []
    assert sorted(g['results']) == ['accounting', 'mc', 'occurrence', 'overdispersion', 'value']
    for name in ('occurrence', 'value', 'accounting', 'mc'):
        assert g['verdict'][name]['pass'] is True, name
    assert g['verdict']['overdispersion']['overdispersed'] is False and 0.9 < g['verdict']['overdispersion']['phi'] < 1.1
    assert g['constants'] == {'B': 2000, 'seed': 20261021, 'S_b0': 1000, 'n_matches': 1500, 'n_rows': 9000,
                              'rule': 'gate_pass = all_pass and nothing missing (R27)'}


@pytest.mark.parametrize(
    'bend, failing',
    [
        (dict(occ_scale=1.3), 'occurrence'),
        (dict(value_bias=80.0), 'value'),
        (dict(mc_shift=600.0), 'mc'),
        (dict(accounting_error=1.0), 'accounting'),
    ],
)
def test_one_criterion_that_fails_fails_the_gate(bend, failing):
    g = run_s3.gate_results(**gate_inputs(**bend))
    assert g['gate_pass'] is False and g['verdict'][failing]['pass'] is False
    assert all(g['verdict'][name]['pass'] for name in ('occurrence', 'value', 'accounting', 'mc') if name != failing)


def test_the_gate_uses_the_weights_one_over_the_rows_of_the_match_and_the_closed_form_of_b0_for_both_calibrations(monkeypatch):
    seen = {}
    real_occ, real_val, real_mc, real_od = (
        gate_s3.occurrence_calibration,
        gate_s3.value_calibration,
        gate_s3.mc_agreement,
        gate_s3.overdispersion,
    )
    monkeypatch.setattr(gate_s3, 'occurrence_calibration', lambda *a, **k: (seen.update(occ=(a, k)), real_occ(*a, **k))[1])
    monkeypatch.setattr(gate_s3, 'value_calibration', lambda *a, **k: (seen.update(val=(a, k)), real_val(*a, **k))[1])
    monkeypatch.setattr(gate_s3, 'mc_agreement', lambda *a, **k: (seen.update(mc=(a, k)), real_mc(*a, **k))[1])
    monkeypatch.setattr(gate_s3, 'overdispersion', lambda *a, **k: (seen.update(od=(a, k)), real_od(*a, **k))[1])
    inputs = gate_inputs(n_matches=60, rows=5)
    inputs['match'] = np.repeat(np.arange(60), 5)[:-3]                                              # one match has two rows only
    for key in ('y60', 'expected_onsets60', 'expected_value60', 'realised_value60', 'mc_mean60', 'mc_sd60', 'mu_b1_60', 'mu_b0_60', 'b2_p'):
        inputs[key] = inputs[key][:-3]
    run_s3.gate_results(**inputs)
    weights = features.match_weights(inputs['match'])
    (e, o, m, w), k = seen['occ']
    assert (
        np.array_equal(e, inputs['expected_onsets60'])
        and np.array_equal(o, inputs['y60'])
        and np.array_equal(w, weights)
        and k == {'B': 2000, 'seed': 20261021}
    )
    (ev, rv, m2, w2), _ = seen['val']
    assert np.array_equal(ev, inputs['expected_value60']) and np.array_equal(rv, inputs['realised_value60']) and np.array_equal(w2, weights)
    (mean, sd, S, analytic), _ = seen['mc']
    assert (
        np.array_equal(mean, inputs['mc_mean60'])
        and np.array_equal(sd, inputs['mc_sd60'])
        and S == 1000
        and np.array_equal(analytic, inputs['expected_value60'])
    )
    assert w.min() == 0.2 and w.max() == 0.5 and sorted(set(w.tolist())) == [0.2, 0.5]


def test_the_overdispersion_uses_the_b1_rate_on_the_rows_with_a_positive_rate_and_one_over_n_m_recomputed_over_those_rows(monkeypatch):
    seen = {}
    real = gate_s3.overdispersion
    monkeypatch.setattr(gate_s3, 'overdispersion', lambda *a, **k: (seen.update(args=a, kwargs=k), real(*a, **k))[1])
    inputs = gate_inputs(n_matches=40, rows=4, seed=5)
    mu_b1 = inputs['mu_b1_60'].copy()
    mu_b1[[0, 1, 5]] = 0.0  # rows without exposure: the first match keeps 2, the second 3
    inputs['mu_b1_60'] = mu_b1
    inputs['y60'] = np.where(mu_b1 > 0, inputs['y60'], 0)
    g = run_s3.gate_results(**inputs)
    y, mu, match, w = seen['args']
    keep = mu_b1 > 0
    assert np.array_equal(mu, mu_b1[keep]) and np.array_equal(y, inputs['y60'][keep]) and np.array_equal(match, inputs['match'][keep])
    assert (
        np.array_equal(w, features.match_weights(inputs['match'][keep])) and w[0] == 0.5 and w[2] == pytest.approx(1 / 3)
    )  # 1 / n_m over the kept rows
    assert g['results']['overdispersion']['n_zero_mu'] == 0 and g['results']['overdispersion']['n_rows'] == int(keep.sum())


def test_counts_that_vary_more_than_their_mean_make_the_overdispersion_flag_true_and_it_is_reported_not_judged():
    g = run_s3.gate_results(**gate_inputs(od_noise=0.6))
    od = g['verdict']['overdispersion']
    assert od['overdispersed'] is True and od['phi_ci'][0] > 1.1 and 'pass' not in od
    assert g['gate_pass'] == all(
        g['verdict'][name]['pass'] for name in ('occurrence', 'value', 'accounting', 'mc')
    )  # the flag is no criterion


def test_the_gate_pass_rule_is_applied_to_the_verdict_it_gets_even_when_all_pass_is_true_and_a_criterion_is_missing(monkeypatch):
    real = gate_s3.verdict
    monkeypatch.setattr(gate_s3, 'verdict', lambda results: {**real(results), 'all_pass': True, 'missing': ['mc']})
    g = run_s3.gate_results(**gate_inputs(n_matches=200))
    assert g['verdict']['all_pass'] is True and g['verdict']['missing'] == ['mc'] and g['gate_pass'] is False


def test_the_gate_refuses_a_number_of_matches_that_is_not_the_number_of_dev_matches_with_rows():
    inputs = gate_inputs(n_matches=300, rows=4)
    with pytest.raises(run_s3.StageFailed, match='DEV matches'):
        run_s3.gate_results(**{**inputs, 'n_matches': 301})  # an interval made from fewer matches than there are


@pytest.mark.parametrize('bad', [dict(B=1000), dict(seed=7), dict(S=500)])
def test_the_gate_is_pinned_to_b_2000_the_seed_20261021_and_s_1000(bad):
    with pytest.raises(run_s3.StageFailed, match='pinned to B = 2000'):
        run_s3.gate_results(**{**gate_inputs(n_matches=50), **bad})


def test_the_pinned_constants_are_those_of_the_plan_and_of_the_modules_they_must_agree_with():
    assert (run_s3.BOOT_B, run_s3.SEED, run_s3.S_B0, run_s3.S_B1, run_s3.B1_DEV_MATCHES, run_s3.R31_LIMIT) == (
        2000,
        20261021,
        1000,
        200,
        500,
        50.0,
    )
    assert run_s3.BOOT_B == nll.BOOT_B == flowprice.BOOT_B and run_s3.SEED == nll.BOOT_SEED == flowprice.BOOT_SEED == compose.DEFAULT_SEED
    assert run_s3.S_B0 == compose.DEFAULT_S and run_s3.MAX_WORKERS == compose.MAX_WORKERS == 61 and run_s3.MEMORY_CAP_GIB == 40.0
    assert (
        run_s3.B2_BINS == baselines_b1.B2_BINS == 6
        and run_s3.LABEL_ROLES == value.LABEL_ROLES
        and run_s3.STAGES == ('levels', 'labels', 'fit', 'compose', 'gate')
    )
    assert run_s3.TIER_ROLES == ('FIT',) and run_s3.B0_ROLES == {'fit': 'FIT'} and run_s3.B1_ROLES == {'fit': 'FIT', 'stop': 'STOP'}
    assert 'DEV' not in run_s3.B1_ROLES.values() and 'DEV' not in run_s3.B0_ROLES.values() and 'DEV' not in run_s3.TIER_ROLES


def test_the_brier_scores_of_m2_and_b2_by_hand():
    """Three rows, two matches (weights 1/n_m = 1/2, 1/2, 1): observed counts 0, 2, 1 (so y = 0, 1, 1), M2 of B0 with p = 1/2, 0, 1/2
    (1 - exp(-mu), mu = ln 2, 0, ln 2), M2 of B1 with p = 1/2, 1/2, 0, B2 with p = 1/4, 3/4, 1/2. (p - y)^2: B0 1/4, 1, 1/4;
    B1 1/4, 1/4, 1; B2 1/16, 1/16, 1/4.
    """
    ln2 = np.log(2.0)
    got = run_s3.brier_comparison(
        mu_b0=[ln2, 0.0, ln2], mu_b1=[ln2, ln2, 0.0], p_b2=[0.25, 0.75, 0.5], observed=[0, 2, 1], match=[0, 0, 1], B=200, seed=4
    )
    assert got['m2_b0']['brier'] == pytest.approx((0.5 * 0.25 + 0.5 * 1 + 1 * 0.25) / 2)
    assert got['m2_b0']['brier'] == pytest.approx(0.4375)
    assert got['m2_b1']['brier'] == pytest.approx(0.625) and got['b2']['brier'] == pytest.approx(0.15625)
    d = got['difference']
    assert d['m2_b0_minus_b2']['value'] == pytest.approx(0.28125) and d['m2_b1_minus_b2']['value'] == pytest.approx(0.46875)
    assert d['m2_b0_minus_m2_b1']['value'] == pytest.approx(-0.1875)
    assert (
        got['positive_share'] == pytest.approx(0.75)
        and got['n_rows'] == 3
        and got['n_matches'] == 2
        and (got['B'], got['seed']) == (200, 4)
    )
    values = {'m2_b0': np.array([0.25, 1, 0.25]), 'm2_b1': np.array([0.25, 0.25, 1]), 'b2': np.array([1 / 16, 1 / 16, 0.25])}
    w = np.array([0.5, 0.5, 1.0])
    reps = {k: nll.match_bootstrap(v, np.array([0, 0, 1]), w, 200, 4) for k, v in values.items()}
    for k in values:
        assert got[k]['ci'] == pytest.approx(list(np.percentile(reps[k], [2.5, 97.5])))
    paired = list(np.percentile(reps['m2_b0'] - reps['b2'], [2.5, 97.5]))                       # the same matches are drawn for every score
    assert d['m2_b0_minus_b2']['ci'] == pytest.approx(paired)


def test_the_m2_forecast_of_at_least_one_onset_is_one_minus_exp_of_minus_the_expected_count():
    got = run_s3.brier_comparison(
        mu_b0=[0.0, 1.0, 3.0], mu_b1=[0.0, 1.0, 3.0], p_b2=[0.5, 0.5, 0.5], observed=[0, 1, 5], match=[0, 1, 2], B=20, seed=1
    )
    p = 1 - np.exp(-np.array([0.0, 1.0, 3.0]))
    assert (
        got['m2_b0']['brier'] == pytest.approx(float(np.mean((p - np.array([0, 1, 1])) ** 2)))
        and got['m2_b0']['brier'] == got['m2_b1']['brier']
    )
    assert got['difference']['m2_b0_minus_m2_b1']['value'] == 0.0


def test_the_observed_onsets_of_the_first_60_s_are_the_onsets_of_the_bins_0_to_5_of_each_row(onset_toy_s1):
    """The onset toy has the bins of every row written out: [3, 9], [6], [], [5], [0, 0], [17], [2], [], [7, 12], [5, 5], [8]."""
    got = run_s3.observed_onsets(onset_toy_s1, np.arange(11))
    assert (
        got.tolist() == [1, 0, 0, 1, 2, 0, 1, 0, 0, 2, 0] and [len([b for b in bins if b < 6]) for bins in TOY_ONSET_BINS] == got.tolist()
    )
    assert run_s3.observed_onsets(onset_toy_s1, np.array([9, 4, 4])).tolist() == [2, 2, 2]
    assert run_s3.observed_onsets(onset_toy_s1, np.arange(11), n_bins=18).tolist() == [len(b) for b in TOY_ONSET_BINS]
    assert run_s3.observed_onsets(onset_toy_s1, np.array([], dtype=np.int64)).tolist() == []


class StubTierModel:
    """The tier law as a table: a tower taken by blue is outer or inner with probability 1/2, one taken by red inner 1/4 or base 3/4."""

    def __init__(self):
        self.calls = []

    def prob(self, destroyed_victim, taker, t_s):
        self.calls.append((np.array(destroyed_victim), np.array(taker), np.array(t_s)))
        out = np.zeros((len(taker), 4))
        out[np.asarray(taker) == 0] = [0.5, 0.5, 0.0, 0.0]
        out[np.asarray(taker) == 1] = [0.0, 0.25, 0.75, 0.0]
        return out


def tower_world():
    """One DEV match with seven objectives (kind 6 is the tower; sub is the tier, -1 none; taker 0 blue, 1 red):
        t 50 s   outer, red     (before the first row: no window, but it counts in the state)
        t 100 s  outer, blue    row t0 60 s     victim red, nothing lost yet before 60 s
        t 130 s  dragon         no tower
        t 170 s  tower without a tier
        t 200 s  inner, red     row t0 180 s    victim blue, who lost an outer at 50 s
        t 250 s  base, blue     row t0 240 s    victim red, who lost an outer at 100 s
        t 330 s  outer, blue    past the window (240 s + 60 s = 300 s)"""
    arrays = {'matches/obj_off': np.array([0, 7]), 'objectives/t': np.array([50, 100, 130, 170, 200, 250, 330]) * 1000,
              'objectives/kind': np.array([6, 6, 0, 6, 6, 6, 6]), 'objectives/sub': np.array([0, 0, -1, -1, 1, 2, 0]),
              'objectives/taker': np.array([1, 0, 0, 0, 1, 0, 0])}
    return types.SimpleNamespace(array=lambda name: arrays[name])


def test_the_tower_pricing_diagnostic_compares_the_tier_price_of_each_tower_in_the_window_with_the_price_the_t0_state_expects():
    prices = {'tower_outer': 100.0, 'tower_inner': 200.0, 'tower_base': 400.0, 'tower_nexus': 800.0}
    tier = StubTierModel()
    got = run_s3.tower_pricing_diagnostic(
        tower_world(),
        frame_row=[0, 1, 2, 3],
        match=[0, 0, 0, 0],
        t0_ms=[60_000, 120_000, 180_000, 240_000],
        tier_model=tier,
        prices=prices,
        B=50,
        seed=3,
    )
    # the three towers in a window: outer by blue at 100 s (expected 150), inner by red at 200 s (350), base by blue at 250 s (150)
    assert got['n_events'] == 3 and got['n_rows'] == 4 and got['n_matches'] == 1 and got['window_s'] == 60.0
    assert got['mean_realised'] == pytest.approx((100 + 200 + 400) / 3) and got['mean_expected'] == pytest.approx((150 + 350 + 150) / 3)
    assert got['mean_difference'] == pytest.approx((-50 - 150 + 250) / 3) and got['sum_realised'] == 700.0 and got['sum_expected'] == 650.0
    assert got['mean_difference_ci'] == pytest.approx([got['mean_difference']] * 2)  # one match: every replicate is the same
    assert got['by_tier'] == {'tower_outer': {'n': 1, 'mean_expected_at_t0': 150.0}, 'tower_inner': {'n': 1, 'mean_expected_at_t0': 350.0},
                              'tower_base': {'n': 1, 'mean_expected_at_t0': 150.0}, 'tower_nexus': {'n': 0, 'mean_expected_at_t0': None}}
    (state, taker, t_s), = tier.calls
    assert state.tolist() == [[0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]  # the towers the victim had lost before t0 of the row
    assert taker.tolist() == [0, 1, 0] and t_s.tolist() == [100.0, 200.0, 250.0]


def test_the_window_of_a_row_is_open_at_t0_and_closed_at_t0_plus_60_s():
    """A row with t0 = 60 s sees the towers in (60 s, 120 s]: one at t0 itself and one a millisecond after the end are not in it."""
    arrays = {'matches/obj_off': np.array([0, 4]), 'objectives/t': np.array([60_000, 100_000, 120_000, 120_001]),
              'objectives/kind': np.array([6, 6, 6, 6]), 'objectives/sub': np.array([0, 0, 0, 0]),
              'objectives/taker': np.array([0, 0, 0, 0])}
    got = run_s3.tower_pricing_diagnostic(
        types.SimpleNamespace(array=lambda name: arrays[name]),
        frame_row=[0],
        match=[0],
        t0_ms=[60_000],
        tier_model=StubTierModel(),
        prices={'tower_outer': 100.0, 'tower_inner': 200.0, 'tower_base': 400.0, 'tower_nexus': 800.0},
        B=10,
        seed=1,
    )
    assert got['n_events'] == 2 and got['by_tier']['tower_outer']['n'] == 2


def test_the_tower_diagnostic_with_no_tower_in_a_window_reports_nothing_and_counts_the_rows():
    got = run_s3.tower_pricing_diagnostic(
        tower_world(),
        frame_row=[7],
        match=[0],
        t0_ms=[1_000_000],
        tier_model=StubTierModel(),
        prices={'tower_outer': 1.0, 'tower_inner': 2.0, 'tower_base': 3.0, 'tower_nexus': 4.0},
        B=10,
        seed=1,
    )
    assert got['n_events'] == 0 and got['mean_difference'] is None and got['mean_difference_ci'] == [None, None] and got['by_tier'] == {}
    with pytest.raises(ValueError, match='not ordered'):
        run_s3.tower_pricing_diagnostic(
            tower_world(),
            frame_row=[0, 1],
            match=[0, 0],
            t0_ms=[120_000, 60_000],
            tier_model=StubTierModel(),
            prices={'tower_outer': 1.0, 'tower_inner': 2.0, 'tower_base': 3.0, 'tower_nexus': 4.0},
            B=10,
            seed=1,
        )


def test_the_score_summary_of_a_head_is_the_match_weighted_mean_with_its_interval_and_the_rows_left_out():
    from c_ept.head_rows import HeadScores
    scores = HeadScores(nll=np.array([1.0, 3.0, 2.0, 2.0]), match=np.array([0, 0, 1, 1]), row=np.array([0, 0, 1, 2]), n_unscored=5)
    got = run_s3.score_summary(scores, B=100, seed=2)
    assert got['n_rows'] == 3 and got['n_design_rows'] == 4 and got['n_unscored'] == 5
    assert (
        got['nll'] == pytest.approx(3.0) and got['nll'] == scores.mean()
    )  # rows: 4 (match 0, weight 1), 2 and 2 (match 1, weight 1/2 each)
    assert got['nll_ci'] == pytest.approx(list(np.percentile(scores.bootstrap(100, 2), [2.5, 97.5])))
    empty = run_s3.score_summary(
        HeadScores(nll=np.empty(0), match=np.empty(0, dtype=np.int64), row=np.empty(0, dtype=np.int64), n_unscored=2)
    )
    assert empty == {'n_rows': 0, 'n_design_rows': 0, 'n_unscored': 2, 'nll': None, 'nll_ci': [None, None]}


def test_the_gate_stage_reads_the_files_of_the_stages_before_and_writes_gate_json(e2e, tmp_path):
    run = copy_of_the_run(e2e, tmp_path)
    (run / 'gate' / 'GATE.json').unlink()
    with context_for(run, e2e.world, smoke=12) as ctx:
        result = run_s3.stage_gate(ctx)
    now, before = read_json(run / 'gate' / 'GATE.json'), read_json(e2e.run / 'gate' / 'GATE.json')
    for key in ('results', 'verdict', 'gate_pass', 'brier', 'tower_pricing', 'constants'):
        assert now[key] == before[key], key  # a pure function of the files: the same answer
    assert result['gate_pass'] == now['gate_pass'] and result['missing'] == []


def test_the_gate_stage_gives_the_gate_the_60_s_column_of_every_array_it_reads(e2e, tmp_path, monkeypatch):
    run = copy_of_the_run(e2e, tmp_path)
    seen, real = [], run_s3.gate_results
    monkeypatch.setattr(run_s3, 'gate_results', lambda **kwargs: (seen.append(kwargs), real(**kwargs))[1])
    with context_for(run, e2e.world, smoke=12) as ctx:
        run_s3.stage_gate(ctx)
        (kw,) = seen
        with np.load(run / 'compose' / 'b0_dev.npz') as z:
            b0 = {k: z[k] for k in z.files}
        with np.load(run / 'fit' / 'DEV_PRED.npz') as z:
            pred = {k: z[k] for k in z.files}
        labels = store_core.Store(run / 'labels' / 'store')
        v_start = np.asarray(labels.array('labels/v_start'))
        rows = b0['frame_row']
        for name in ('expected', 'expected_onsets', 'mc_mean', 'mc_sd'):                  # the horizons differ: a wrong column shows
            assert not np.array_equal(b0[name][:, 0], b0[name][:, 1]), name
        assert not np.array_equal(v_start[rows, 0], v_start[rows, 1]) and not np.array_equal(pred['mu_b1'][:, 0], pred['mu_b1'][:, 1])
        assert np.array_equal(kw['realised_value60'], v_start[rows, 0]) and np.array_equal(kw['expected_value60'], b0['expected'][:, 0])
        assert np.array_equal(kw['expected_onsets60'], b0['expected_onsets'][:, 0])
        assert np.array_equal(kw['mu_b0_60'], b0['expected_onsets'][:, 0])
        assert np.array_equal(kw['mc_mean60'], b0['mc_mean'][:, 0]) and np.array_equal(kw['mc_sd60'], b0['mc_sd'][:, 0])
        assert np.array_equal(kw['mu_b1_60'], pred['mu_b1'][:, 0]) and np.array_equal(kw['b2_p'], pred['b2_p'])
        assert np.array_equal(kw['match'], b0['match']) and np.array_equal(kw['y60'], run_s3.observed_onsets(ctx.s1, rows))
        assert kw['accounting_error'] == read_json(run / 'labels' / 'ACCOUNTING.json')['max_abs_error'] == 0.0
        assert (kw['B'], kw['seed'], kw['S']) == (2000, 20261021, 1000) and kw['n_matches'] == len(np.unique(b0['match'])) == 4
        release(labels)


def test_the_gate_stage_refuses_files_of_the_composition_and_the_fit_that_are_not_for_the_same_rows(e2e, tmp_path):
    run = copy_of_the_run(e2e, tmp_path)
    with np.load(run / 'fit' / 'DEV_PRED.npz') as z:
        pred = {k: z[k] for k in z.files}
    pred['frame_row'] = pred['frame_row'] + 1
    store_core.write_atomic(run / 'fit' / 'DEV_PRED.npz', run_s3._npz_bytes(pred))
    with context_for(run, e2e.world, smoke=12) as ctx:
        with pytest.raises(run_s3.StageFailed, match='differ'):
            run_s3.stage_gate(ctx)


def test_a_label_store_that_was_changed_is_refused_by_the_stages_that_read_it(e2e, tmp_path):
    run = copy_of_the_run(e2e, tmp_path)
    victim = run / 'labels' / 'store' / 'labels' / 'v_start.npy'
    victim.write_bytes(victim.read_bytes() + b'\0')
    with context_for(run, e2e.world, smoke=12) as ctx:
        with pytest.raises(run_s3.RunRefused, match='does not match STORE.json'):
            run_s3.open_labels(ctx)
        for stage in (run_s3.stage_fit, run_s3.stage_compose, run_s3.stage_gate):
            with pytest.raises(run_s3.RunRefused):
                stage(ctx)


# =====================================================================================================================================
# the structure of the module: what it imports, how it starts, the command line as a process
# =====================================================================================================================================
LIGHT_C_EPT = {'c_ept', 'c_ept.config', 'c_ept.features', 'c_ept.nll', 'c_ept.runlog', 'c_ept.store_core', 'c_ept.runtime'}
MODEL_LIBRARIES = {'lightgbm', 'sklearn', 'torch', 'pandas', 'joblib', 'pickle', 'psycopg', 'psycopg2', 'numba'}
STAGE_MODULES = (
    'c_ept.run_s1',
    'c_ept.run_s2',
    'c_ept.cohort',
    'c_ept.store_build',
    'c_ept.value',
    'c_ept.compose',
    'c_ept.baselines_b0',
    'c_ept.baselines_b1',
    'c_ept.towertier',
    'c_ept.gate_s3',
    'c_ept.rulepay',
    'c_ept.heads',
    'c_ept.head_rows',
    'c_ept.report_s2',
)


def run_module(*args, timeout=180):
    """``python -m c_ept.run_s3 ARGS`` as a process of its own (the repository is the working folder)."""
    return subprocess.run([sys.executable, '-m', 'c_ept.run_s3', *args], cwd=REPO, capture_output=True, text=True, encoding='utf-8',
                          errors='replace', timeout=timeout, check=False)


def test_the_module_level_imports_are_light_and_hold_no_s1_runner_no_stage_module_and_no_model_library():
    imports = module_level_imports(SOURCE)
    assert {n for n in imports if n.split('.')[0] == 'c_ept'} <= LIGHT_C_EPT
    assert {'c_ept.config', 'c_ept.runlog', 'c_ept.store_core', 'c_ept.runtime'} <= imports
    assert not {n.split('.')[0] for n in imports} & MODEL_LIBRARIES
    assert not imports & set(STAGE_MODULES)


def test_the_s1_runner_is_imported_inside_functions_only_and_no_reader_cohort_or_s2_runner_is_imported_at_all():
    assert 'c_ept.run_s1' not in module_level_imports(SOURCE)
    assert 'c_ept.run_s1' in all_imports(SOURCE)                                         # the E6 guard and the S1 loader come from it
    assert not all_imports(SOURCE) & {'c_ept.run_s2', 'c_ept.reader', 'c_ept.cohort', 'c_ept.terrain', 'c_ept.targets'}
    assert not {n for n in all_imports(SOURCE) if any(w in n for w in ('pickle', 'joblib', 'dill', 'shelve', 'marshal'))}


def test_importing_the_module_loads_no_stage_module_no_s1_runner_and_no_model_library():
    code = ('import json, sys\n'
            'import c_ept.run_s3\n'
            f'stage = {list(STAGE_MODULES)!r}\n'
            f'libraries = {sorted(MODEL_LIBRARIES - {"pickle"})!r}\n'                          # numpy itself imports pickle
            'print(json.dumps(sorted(m for m in sys.modules if m in stage or m.split(".")[0] in libraries)))\n')
    proc = subprocess.run([sys.executable, '-c', code], cwd=REPO, capture_output=True, text=True, encoding='utf-8', errors='replace',
                          timeout=180, check=False)
    assert proc.returncode == 0, proc.stderr[-400:]
    assert json.loads(proc.stdout.strip().splitlines()[-1]) == []


def test_main_is_called_only_under_the_entry_guard_and_nothing_else_runs_when_the_module_is_imported():
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    guards = [n for n in tree.body if isinstance(n, ast.If) and ast.unparse(n.test).replace('"', "'") == "__name__ == '__main__'"]
    assert len(guards) == 1 and tree.body[-1] is guards[0]
    assert ast.unparse(guards[0].body[0]) == 'raise SystemExit(main())'
    inside = {id(n) for n in ast.walk(guards[0])}
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, 'id', None) == 'main']
    assert calls and all(id(c) in inside for c in calls)
    statements = [n for n in tree.body if isinstance(n, ast.Expr) and not isinstance(n.value, ast.Constant)]
    assert not statements                                                               # no call at module level (only the docstring)


def test_the_parent_process_sets_one_environment_variable_and_only_in_its_own_function():
    """The parent sets no MKL, OpenMP or other variable (ruling R30: the workers get ``compose.WORKER_ENVIRONMENT``). The one thing that it
    sets is LOKY_MAX_CPU_COUNT, by ``setdefault`` in ``_set_loky_cpu_count`` (G-S3b B3), which ``main`` calls first."""
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.stack, self.sites = [], []

        def visit_FunctionDef(self, node):
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_Attribute(self, node):
            if node.attr in ('environ', 'putenv', 'unsetenv'):
                self.sites.append((self.stack[-1] if self.stack else '<module>', node.attr))
            self.generic_visit(node)

    visitor = Visitor()
    visitor.visit(tree)
    assert visitor.sites and {where for where, _ in visitor.sites} == {'_set_loky_cpu_count'}
    assert {attr for _, attr in visitor.sites} == {'environ'}
    function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_set_loky_cpu_count')
    text = ast.unparse(function)
    assert "os.environ.setdefault('LOKY_MAX_CPU_COUNT'" in text and text.count('LOKY_MAX_CPU_COUNT') >= 2
    main = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'main')
    assert ast.unparse(main.body[1]) == '_set_loky_cpu_count()'                      # after the docstring, before the arguments are read


def forget_loky_cpu_count(monkeypatch) -> None:
    """Remove LOKY_MAX_CPU_COUNT for the rest of the test and put it back as it was afterwards (``monkeypatch.delenv`` of a variable that
    is not there records nothing, and a later ``setdefault`` would leak into the other tests)."""
    monkeypatch.setenv('LOKY_MAX_CPU_COUNT', 'x')
    monkeypatch.delenv('LOKY_MAX_CPU_COUNT')


def test_loky_max_cpu_count_is_set_to_the_physical_cores_when_it_is_not_set(monkeypatch):
    pytest.importorskip('psutil')
    forget_loky_cpu_count(monkeypatch)
    asked = []
    monkeypatch.setattr(run_s3.psutil, 'cpu_count', lambda logical=True: (asked.append(logical), 6)[1])
    run_s3._set_loky_cpu_count()
    assert os.environ['LOKY_MAX_CPU_COUNT'] == '6' and asked == [False]


def test_loky_max_cpu_count_falls_back_to_the_logical_cores_when_the_physical_ones_are_not_known(monkeypatch):
    pytest.importorskip('psutil')
    forget_loky_cpu_count(monkeypatch)
    monkeypatch.setattr(run_s3.psutil, 'cpu_count', lambda logical=True: None)
    run_s3._set_loky_cpu_count()
    assert os.environ['LOKY_MAX_CPU_COUNT'] == str(os.cpu_count() or 1)
    forget_loky_cpu_count(monkeypatch)
    monkeypatch.setattr(run_s3, 'psutil', None)                                          # no psutil at all
    run_s3._set_loky_cpu_count()
    assert os.environ['LOKY_MAX_CPU_COUNT'] == str(os.cpu_count() or 1)


def test_loky_max_cpu_count_is_left_alone_when_the_user_set_it(monkeypatch):
    pytest.importorskip('psutil')
    monkeypatch.setenv('LOKY_MAX_CPU_COUNT', '3')
    asked = []
    monkeypatch.setattr(run_s3.psutil, 'cpu_count', lambda logical=True: (asked.append(logical), 6)[1])
    run_s3._set_loky_cpu_count()
    assert os.environ['LOKY_MAX_CPU_COUNT'] == '3' and asked == []                         # the machine is not even asked


def test_main_sets_loky_max_cpu_count_before_it_reads_its_arguments(monkeypatch, capsys):
    forget_loky_cpu_count(monkeypatch)
    with pytest.raises(SystemExit) as raised:
        run_s3.main(['--help'])
    assert raised.value.code == 0 and 'usage' in capsys.readouterr().out
    assert os.environ['LOKY_MAX_CPU_COUNT'].isdigit() and int(os.environ['LOKY_MAX_CPU_COUNT']) >= 1


def test_loky_max_cpu_count_is_set_before_sklearn_or_joblib_is_imported():
    code = ('import os, sys\n'
            'from c_ept import run_s3\n'
            'try:\n'
            '    run_s3.main(["--help"])\n'
            'except SystemExit:\n'
            '    pass\n'
            'print(os.environ.get("LOKY_MAX_CPU_COUNT"), "sklearn" in sys.modules, "joblib" in sys.modules)\n')
    env = {k: v for k, v in os.environ.items() if k != 'LOKY_MAX_CPU_COUNT'}
    proc = subprocess.run([sys.executable, '-c', code], cwd=REPO, capture_output=True, text=True, encoding='utf-8', errors='replace',
                          timeout=180, check=False, env=env)
    assert proc.returncode == 0, proc.stderr[-400:]
    count, sklearn, joblib = proc.stdout.strip().splitlines()[-1].split()
    assert count.isdigit() and int(count) >= 1 and (sklearn, joblib) == ('False', 'False')


def test_the_key_array_is_read_in_two_places_only_the_key_cache_and_the_scrub_of_an_unexpected_error():
    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.stack, self.sites = [], []

        def visit_FunctionDef(self, node):
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_Constant(self, node):
            if node.value == 'matches/key':
                self.sites.append('.'.join(self.stack))

    visitor = Visitor()
    visitor.visit(ast.parse(SOURCE.read_text(encoding='utf-8')))
    assert sorted(visitor.sites) == ['keys', 'main']                                    # Context.keys and the scrub in main


def test_help_prints_every_option_and_exits_zero():
    proc = run_module('--help')
    assert proc.returncode == 0 and 'Traceback' not in proc.stderr
    for option in (
        '--s1',
        '--s2',
        '--s3a',
        '--spec',
        '--stages',
        '--smoke',
        '--resume',
        '--workers',
        '--verify',
        '--runs-root',
        '--catalog-db',
    ):
        assert option in proc.stdout
    for stage in STAGES:
        assert stage in proc.stdout


@pytest.mark.parametrize('extra, message', [(['--stages', 'levels', 'nonsense'], 'unknown stage'),
                                            (['--stages', 'fit'], 'needs the stage levels'),
                                            (['--stages', 'levels', 'levels'], 'twice'),
                                            (['--workers', '0'], '--workers must be'),
                                            (['--smoke', '0'], '--smoke must be at least 1')])
def test_a_refusal_ends_the_process_with_status_two_a_message_on_stderr_and_no_run_folder(tmp_path, extra, message):
    runs = tmp_path / 'runs'
    proc = run_module('--s1', str(tmp_path / 's1'), '--s2', str(tmp_path / 's2'), '--s3a', str(tmp_path / 's3a'), '--runs-root', str(runs),
                      '--catalog-db', str(tmp_path / 'catalog.db'), *extra)
    assert proc.returncode == 2 and message in proc.stderr
    assert proc.stdout == '' and not runs.exists() and not (tmp_path / 'catalog.db').exists()


def test_a_missing_required_option_is_the_usage_error_of_argparse():
    proc = run_module('--s1', 'x')
    assert proc.returncode == 2 and 'required' in proc.stderr
