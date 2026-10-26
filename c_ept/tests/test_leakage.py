"""The three masking-leakage tests of the composition path (stage S3 Task 11, gate G9b of the design).

The composed value of the engagements after t0 may use the information at t0 and what the heads sample, and nothing else (constraint
A3). The three tests, with the one rule behind each of them:

    (a) structure   the builder of the masked M4 input takes the t0 row view, q, m, z, K and O and nothing else; the views of the row
                    table are the frame-clock views at t0; compose.py names no array of the engagements, reads no S1 table outside the
                    t0 ones and gets the tower state through ``towertier`` only, asked for once for each row, at the t0 of that row
    (b) attach      the true (K, O) of the rows are attached to the row table and to the heads (under every name a lookup would try) and
                    are present in the S1 store as arrays: the composition (``simulate``, and ``expect`` where the heads have a closed
                    form) is bit for bit what it is without them
    (c) perturb     the toy S1 store is rebuilt on disk with the arrays units/{delta, last, K, O, conv, conv_exposure_ms, mark, size}
                    replaced by noise that changes every entry: the row table and the composition are bit for bit the same

Each test calls a check (``check_a``, ``check_b``, ``check_c``) that raises ``Leak``, an AssertionError whose message says what depends on
what it must not. A check that cannot fail proves nothing, so each one is also run on deliberately leaking variants of the composition path
and must fail on every one of them (the RED tests, in the second half of this file). The variants are made inside these tests, with
``monkeypatch`` (and, for the text of compose.py, a changed copy of the text); no production code is touched.

    (a) the builder of the M4 input accepts the true delta, or adds a column; the composer hands the head one more column; the source of
        compose.py names an array of the engagements (or a table that is not a t0 one); the row table reads the objectives itself, reads
        an engagement table, or picks an event view (a view at an event time, after t0) for a row; the row table asks towertier for
        the state at another time than t0 (at t0 + 60 s, at the middle of each bin, or at the t0 of the next row of the match) and the
        composer uses it (these three also move the composed values, which the RED tests show)
    (b) a leak is a pair (medium, mechanism): the true K and O are found on the row table, on the heads or in the S1 store; the composer
        feeds them to the object and conversion heads instead of its samples, prices the true kill gold in the value of an onset, or adds
        the true kill gold to the closed form (the last two patch the private ``_draw_row`` and ``_expect_row`` of the composer)
    (c) the composer reads the S1 K and O (the same three mechanisms); the row table depends on each of the eight perturbed arrays

The stores are the toy S1 of the value toys (made by the real extraction, so its units hold true K and O) with its toy S2, and the toy
heads of ``s3_toys``: ``ToyHeads`` (closed form) and ``KOToyHeads`` (reads K and O, the case of B1, so that a leak of K or O shows).
The tower prices and the tier law of the composers are the local ``PRICES`` and ``state_tier_model``, which depend on the state (the
toy of ``s3_toys`` prices every tier at 400, so that a tower state taken at the wrong time would move nothing). No run folder, raw
file or catalog is read.
"""
# Ref: docs/superpowers/specs/2026-10-20-c-ept-design.md G9b (the three masking-leakage tests) and
#      docs/superpowers/plans/2026-10-21-c-ept-s3.md Task 11, the global constraint A3 (the masked input is the t0 row and (q, m, z); K and
#      O are samples) and P10 (the conversion exposure uses the estimated delta, never the true one).
import ast
import dataclasses
import gc
import inspect
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest

from c_ept.valuation import compose
from c_ept.preprocessing import features
from c_ept.valuation import towertier
from c_ept import config as C
from c_ept.data.store_core import Store
from c_ept.tests.s3_toys import (TOY_COMPOSE_PRICES, TOY_GOLD_REPS, KOToyHeads, ToyHeads, build_toy_s2, toy_tier_model,  # noqa: F401
                                 toy_value_s1)

SOURCE = Path(compose.__file__)
FORBIDDEN = ('units/', 'onsets/', 'kills/', 'objectives/', 'event_rows/')        # the arrays of the engagements: compose.py names none
# the tables hidden from the store in the "without" runs of (b); objectives/ stays, the tower state of a row is read from it by towertier
ENGAGEMENT = ('units/', 'onsets/', 'kills/', 'event_rows/')
READABLE = ('matches/', 'frame_rows/', 'views/')                                 # the arrays that compose.py itself may name
M4_PARAMETERS = ['row_view', 'q', 'm', 'z', 'K', 'O']
PERTURBED = ('units/delta', 'units/last', 'units/K', 'units/O', 'units/conv', 'units/conv_exposure_ms', 'units/mark', 'units/size')
MILLISECONDS = ('units/delta', 'units/last', 'units/conv_exposure_ms')
K_NAMES, O_NAMES = ('true_K', 'K_true', 'K', 'kills'), ('true_O', 'O_true', 'O', 'objectives')    # where (b) attaches the true K and O
ROLES = ('FIT', 'STOP', 'DEV')                                                   # the 22 frame rows of the matches 0, 1 and 2 of the toy
S_SIM, SEED = 200, 20261021
ALL_ROWS = list(range(22))
# the rows of the closed form: 0 (full exposure), 1 and 7 (lost towers; 7 has 13 bins of exposure), 9 (one bin), 10 and 17 (other matches)
EXPECT_ROWS = [0, 1, 7, 9, 10, 17]
HEADS = {'closed': ToyHeads, 'ko': KOToyHeads}
# The tower prices of the toy depend on the state, so that a tower state taken at another time than t0 changes the composed values: the four
# tiers are worth 300 .. 900, and the tier law makes the next tier of the victim the likely one. (``s3_toys`` prices every tier at 400.)
PRICES = {**TOY_COMPOSE_PRICES, 'tower_outer': 300.0, 'tower_inner': 500.0, 'tower_base': 700.0, 'tower_nexus': 900.0}


def state_tier_model():
    """The tier law of the composers of this file: the logit of tier j is -2 D_j + D_(j-1) and a little of the taker and the minutes, D_j
    the number of towers of tier j that the victim has lost (the features of ``towertier``: D_0 .. D_3, the taker, the minutes)."""
    coef = [[-2.0, 0.0, 0.0, 0.0, 0.0, -0.02], [1.0, -2.0, 0.0, 0.0, 0.3, 0.0], [0.0, 1.0, -2.0, 0.0, 0.0, 0.01],
            [0.0, 0.0, 1.0, -2.0, 0.0, 0.02]]
    return towertier.TierModel([0, 1, 2, 3], coef, [1.0, 0.0, -1.0, -2.0], np.zeros(towertier.N_FEATURES), np.ones(towertier.N_FEATURES))


class Leak(AssertionError):
    """What the composition returns depends on something it must not see; the message says which output (a failure of a check, as opposed
    to a failure of its set-up, which is a plain AssertionError)."""


@dataclass(frozen=True, eq=False)
class World:
    """The stores of the toy and the true outcome of its rows: ``s1`` and ``s2``, ``perturbed`` (the S1 store rebuilt with the arrays of
    ``PERTURBED`` replaced by noise), the row table ``rows`` of ``s1``, and ``K`` (n, 4) and ``O`` (n, 18), the true outcome of each row."""
    s1: object
    s2: object
    perturbed: object
    rows: compose.RowTable
    K: np.ndarray
    O: np.ndarray


class ViewStore:
    """A store as a test wants to see it: the arrays whose name starts with one of ``hide`` are not there (FileNotFoundError, as in a store
    that was never written), and every array that is asked for is recorded in ``reads``, in order."""

    def __init__(self, base, hide=()):
        self.base, self.hide, self.reads = base, tuple(hide), []

    def array(self, name):
        self.reads.append(name)
        if name.startswith(self.hide):
            raise FileNotFoundError(f'no array {name}')
        return self.base.array(name)


# =====================================================================================================================================
# the truth of the toy, and the helpers that attach it, perturb it or build composers
# =====================================================================================================================================
def first_unit(s1, rows) -> np.ndarray:
    """(n,) the first engagement (the smallest onset bin) that starts in the window of each row of the table, -1 for a row without one."""
    onset_row, onset_bin, onset_unit = (np.asarray(s1.array(f'onsets/{c}')) for c in ('row', 'bin', 'unit'))
    out = np.full(len(rows), -1, dtype=np.int64)
    for i, row in enumerate(rows.frame_row):
        hit = np.flatnonzero(onset_row == row)
        if hit.size:
            out[i] = onset_unit[hit[np.argmin(onset_bin[hit])]]
    return out


def true_ko(s1, rows) -> tuple:
    """The outcome that the composition predicts and may not use: ((n, 4) kills of blue and red and their kill gold (recorded bounty +
    shutdown bounty), (n, 18) objectives), each summed over the engagements that start in the window of a row of the table (``onsets``).
    Reads ``onsets/row``, ``onsets/unit``, ``units/K`` and ``units/O``; FileNotFoundError where the store lacks them."""
    onset_row, onset_unit = np.asarray(s1.array('onsets/row')), np.asarray(s1.array('onsets/unit'))
    k, o = np.asarray(s1.array('units/K')).astype(np.float64), np.asarray(s1.array('units/O')).astype(np.float64)
    unit_k = np.column_stack([k[:, 0], k[:, 1], k[:, 2] + k[:, 4], k[:, 3] + k[:, 5]])
    K, O = np.zeros((len(rows), 4)), np.zeros((len(rows), o.shape[1]))
    for i, row in enumerate(rows.frame_row):
        for u in onset_unit[onset_row == row]:
            K[i] += unit_k[u]
            O[i] += o[u]
    return K, O


def perturbed_arrays(s1, seed: int = 9) -> dict:
    """{name: array} for every name of ``PERTURBED``: the stored array plus a seeded noise that is never zero, so that every entry changes
    (a time by 1 .. 60,000 ms, a count by 1 .. 6; a mark and a size go to another class, modulo the number of the classes)."""
    rng = np.random.default_rng(seed)
    out = {}
    for name in PERTURBED:
        a = np.asarray(s1.array(name))
        if name == 'units/mark':
            new = (a + rng.integers(1, C.N_MARKS, a.shape)) % C.N_MARKS
        elif name == 'units/size':
            new = (a + rng.integers(1, 4, a.shape)) % 4
        else:
            new = a + rng.integers(1, 60_000 if name in MILLISECONDS else 7, a.shape)
        out[name] = new.astype(a.dtype)
    return out


def attached(rows, **extra):
    """A copy of the row table with more attributes (it is a frozen dataclass: ``object.__setattr__``): what a leak could look for."""
    copy = dataclasses.replace(rows)
    for name, value in extra.items():
        object.__setattr__(copy, name, value)
    return copy


def make_composer(s1, s2, heads, truth=None, tier_model=None, prices=None):
    """The composer of the toy rows of the store ``s1``, with the state-sensitive tower prices of this file (``tier_model`` and ``prices``
    replace them). ``truth`` = (K, O) attaches them, under the names of ``K_NAMES`` and ``O_NAMES``, to the row table and to the heads."""
    rows = compose.row_table(s1, s2, ROLES)                       # looked up now: a leaking variant replaces ``compose.row_table``
    if truth is not None:
        names = {**{name: truth[0] for name in K_NAMES}, **{name: truth[1] for name in O_NAMES}}
        rows = attached(rows, **names)
        for name, value in names.items():
            setattr(heads, name, value)
    return compose.Composer(heads, PRICES if prices is None else prices, state_tier_model() if tier_model is None else tier_model,
                            TOY_GOLD_REPS, rows, s2, S=S_SIM, seed=SEED)


def masked_by_hand(row_view, q, m, z, K, O) -> np.ndarray:
    """What ``features.m4_masked_inputs`` must give, written out again: the row view, q, m, z, K (gold in thousands) and O, as float32."""
    K = np.array(K, dtype=np.float64)
    K[:, 2:] /= 1000.0
    return np.column_stack([row_view, q, m, z, K, O]).astype(np.float32)


class Recording:
    """Mixin for toy heads: the matrix that the conversion head is handed is kept in ``seen``, one entry for each call."""

    def conv_rate(self, m4_inputs):
        self.seen.append(np.array(m4_inputs))
        return super().conv_rate(m4_inputs)


class RecordingKO(Recording, KOToyHeads):
    def __init__(self):
        super().__init__()
        self.seen = []


class RecordingClosed(Recording, ToyHeads):
    def __init__(self):
        super().__init__()
        self.seen = []


def array_names(source: str) -> list:
    """The string constants of ``source`` that look like the name of a store array, ``table/column`` (not the docstrings, which are prose;
    the pieces of an f-string count)."""
    tree = ast.parse(source)
    scopes = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    docstrings = {id(n.body[0].value) for n in ast.walk(tree) if isinstance(n, scopes) and n.body and isinstance(n.body[0], ast.Expr)
                  and isinstance(n.body[0].value, ast.Constant) and isinstance(n.body[0].value.value, str)}
    return sorted({n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
                   and id(n) not in docstrings and re.match(r'[a-z_]+/', n.value)})


# =====================================================================================================================================
# the three checks
# =====================================================================================================================================
def same(what: str, a, b) -> None:
    """``Leak`` unless the two arrays are equal bit for bit."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or not np.array_equal(a, b):
        worst = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.shape == b.shape else float('nan')
        raise Leak(f'{what} is not bit-identical (largest difference {worst:.3g})')


def same_tables(what: str, a, b) -> None:
    """``Leak`` unless two row tables hold the same t0 information: the table is the one channel from S1 to the composer."""
    for field in dataclasses.fields(compose.RowTable):
        if field.name != 'roles':
            same(f'{what}: the row table field {field.name}', getattr(a, field.name), getattr(b, field.name))


def compare_composers(what: str, plain, other, closed_form: bool) -> None:
    """``Leak`` unless ``simulate`` (mean, sd, the 19 quantiles) and, for heads with a closed form, ``expect`` of the two composers are the
    same bit for bit."""
    a, b = plain.simulate(ALL_ROWS), other.simulate(ALL_ROWS)
    assert (a.mean != 0).any() and (a.q19 != 0).any(), 'set-up: the toy simulation is empty, so that equality would prove nothing'
    for field in ('mean', 'sd', 'q19'):
        same(f'{what}: simulate {field}', getattr(a, field), getattr(b, field))
    if closed_form:
        e = plain.expect(EXPECT_ROWS)
        assert (e != 0).any(), 'set-up: the toy closed form is zero, so that equality would prove nothing'
        same(f'{what}: expect', e, other.expect(EXPECT_ROWS))


def check_a(world: World, source: str | None = None) -> None:
    """(a) structure: the masked M4 input is the t0 row view, q, m, z, K and O of the draw and nothing else, and compose.py reads t0
    information only. ``source`` is the text of compose.py (default: the file)."""
    # 1. the builder: its parameters, and what it returns
    parameters = inspect.signature(features.m4_masked_inputs).parameters
    if list(parameters) != M4_PARAMETERS or any(p.kind is not p.POSITIONAL_OR_KEYWORD or p.default is not p.empty
                                                for p in parameters.values()):
        raise Leak(f'the parameters of m4_masked_inputs are {list(parameters)}, not exactly the required {M4_PARAMETERS}')
    rng = np.random.default_rng(3)
    row_view = rng.normal(size=(6, 5)).astype(np.float32)
    q, m, z = np.array([0, 3, 17, 5, 9, 11]), np.array([0, 51, 7, 20, 33, 1]), np.array([-1, 0, 9, 4, 2, 6])
    K, O = rng.integers(0, 6, (6, 4)).astype(np.float64) * 100.0, rng.integers(0, 4, (6, 18))
    got = features.m4_masked_inputs(row_view, q, m, z, K, O)
    want = masked_by_hand(row_view, q, m, z, K, O)
    if got.dtype != want.dtype or got.shape != want.shape or not np.array_equal(got, want):
        raise Leak(f'm4_masked_inputs builds {got.shape} {got.dtype}, not the row view, q, m, z, K and O: {want.shape} {want.dtype}')
    # 2. compose.py, by its text: no array of the engagements, and no array outside the three t0 families
    text = SOURCE.read_text(encoding='utf-8') if source is None else source
    named = [w for w in FORBIDDEN if w in text]
    if named:
        raise Leak(f'compose.py names an array of the engagements: {named}')
    outside = [n for n in array_names(text) if not n.startswith(READABLE)]
    if outside:
        raise Leak(f'compose.py names an S1 or S2 array outside the t0 tables {READABLE}: {outside}')
    # 3. the row table, by what it reads and when: S1 tables of t0 information only; the objectives only inside towertier.destroyed_before,
    #    which is asked for once for each row, at the t0 of that row and at no other time (the state at a later time is the future); and
    #    the frame-clock view of each row at its t0 (the event views are taken at event times, after t0)
    store, views = ViewStore(world.s1), ViewStore(world.s2)
    real, windows, asked = towertier.destroyed_before, [], []       # windows: the reads (from, to) of each call; asked: its (match, t_ms)

    def destroyed_before(s1, match, t_ms):
        start = len(store.reads)
        out = np.asarray(real(s1, match, t_ms))
        windows.append((start, len(store.reads)))
        asked.extend(zip(np.ravel(match).tolist(), np.ravel(t_ms).tolist()))
        return out + 1                                                       # marked: the table must hold what towertier says

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(towertier, 'destroyed_before', destroyed_before)
        rows = compose.row_table(store, views, ROLES)
    wanted = set(zip(rows.match.tolist(), rows.t0_ms.tolist()))                 # the (match, t0) of the rows, one pair for each row
    other = [pair for pair in asked if pair not in wanted]
    if other:
        raise Leak(f'the tower state was asked for at {len(other)} (match, time) pairs that are not the (match, t0) of a row: only the '
                   'state before t0 itself may be used (a later time is the future)')
    if sorted(asked) != sorted(wanted):                       # no pair of a row twice (t0 + 60 s of a row is the t0 of the next one)
        raise Leak(f'the tower state was asked for {len(asked)} times, not once for each of the {len(wanted)} rows at its t0')
    if not np.array_equal(rows.destroyed, real(world.s1, rows.match, rows.t0_ms) + 1):
        raise Leak('the tower state of the row table is not what towertier.destroyed_before says (and only that)')
    stray = sorted({n for i, n in enumerate(store.reads)
                    if n.startswith('objectives/') and not any(lo <= i < hi for lo, hi in windows)})
    if stray:
        raise Leak(f'the row table reads {stray} outside towertier.destroyed_before')
    tables = {n.split('/')[0] + '/' for n in store.reads}
    if not tables <= {'matches/', 'frame_rows/', 'objectives/'} or not {n.split('/')[0] + '/' for n in views.reads} <= {'views/'}:
        raise Leak(f'the row table reads the S1 tables {sorted(tables)} and the S2 arrays {sorted(set(views.reads))}: t0 tables only')
    src, ref, match = (np.asarray(world.s2.array(f'views/{c}')) for c in ('src', 'ref', 'match'))
    if (src[rows.views] != 0).any() or not np.array_equal(ref[rows.views], rows.t0_ms) or not np.array_equal(match[rows.views], rows.match):
        raise Leak('the views of the row table are not the frame-clock views of its rows at their t0')
    # 4. the composer: what the conversion head is handed in a simulation, and in the closed form, is the masked input and nothing else
    heads = RecordingKO()
    d = make_composer(world.s1, world.s2, heads).simulate_onsets(0)
    assert len(d.q) > 0 and heads.seen, 'set-up: the toy row 0 must have onsets'
    seen = np.vstack(heads.seen)
    want = masked_by_hand(heads.row_view(np.zeros(len(d.q), dtype=np.int64)), d.q, d.m, d.z, np.column_stack([d.k, d.gold]), d.O)
    if seen.shape != want.shape or not np.array_equal(seen, want):
        raise Leak(f'the conversion head is handed {seen.shape}, not the row view, q, m, z, K and O of the draw: {want.shape}')
    heads = RecordingClosed()
    make_composer(world.s1, world.s2, heads).expect([0])
    X, d0 = np.vstack(heads.seen), heads.N_VIEW_COLUMNS
    if X.shape[1] != d0 + 25 or (X[:, d0 + 3:] != 0).any() or not np.array_equal(X[:, :d0], np.tile(heads.row_view([0]), (len(X), 1))):
        raise Leak('the conversion head of the closed form is handed more than the row view, q, m, z and K = O = 0')


def check_b(make, closed_form: bool) -> None:
    """(b) ``make(False)`` builds a composer in which the true (K, O) cannot be found, ``make(True)`` one in which they can; the
    composition must not know the difference."""
    plain, with_truth = make(False), make(True)
    same_tables('(b) the true K and O attached', plain.rows, with_truth.rows)
    compare_composers('(b) the true K and O attached', plain, with_truth, closed_form)


def check_c(world: World, make, closed_form: bool) -> None:
    """(c) ``make(store)`` builds a composer from an S1 store; the composition from ``world.perturbed``, the store whose arrays of
    ``PERTURBED`` are noise, must be the one from the stored arrays."""
    for name in PERTURBED:
        old, new = np.asarray(world.s1.array(name)), np.asarray(world.perturbed.array(name))
        assert new.shape == old.shape and (new != old).all(), f'set-up: the perturbation must change every entry of {name}'
    base, other = make(world.s1), make(world.perturbed)
    same_tables('(c) the perturbed S1', base.rows, other.rows)
    compare_composers('(c) the perturbed S1', base, other, closed_form)


def real_b(world: World, kind: str, attach: bool = True):
    """The ``make`` of ``check_b`` for the real code: without, the engagement tables are not in the store and nothing is attached; with,
    they are in the store and (``attach``) the true K and O are attributes of the row table and of the heads."""
    def make(with_truth: bool):
        s1 = world.s1 if with_truth else ViewStore(world.s1, hide=ENGAGEMENT)
        return make_composer(s1, world.s2, HEADS[kind](), truth=(world.K, world.O) if with_truth and attach else None)
    return make


def real_c(world: World, kind: str):
    """The ``make`` of ``check_c`` for the real code."""
    def make(s1):
        return make_composer(s1, world.s2, HEADS[kind]())
    return make


# =====================================================================================================================================
# the fixtures
# =====================================================================================================================================
@pytest.fixture(scope='module')
def leak_s2(tmp_path_factory, toy_value_s1):
    """The toy S2 of the value toys: one frame-clock view for each frame row and two event views for each match."""
    folder = tmp_path_factory.mktemp('leak_s2')
    s = build_toy_s2(folder / 's2', toy_value_s1)
    yield s
    del s
    gc.collect()                                   # Windows does not delete a folder with open memory maps
    shutil.rmtree(folder, ignore_errors=True)


@pytest.fixture(scope='module')
def perturbed_s1(tmp_path_factory, toy_value_s1):
    """The toy S1 store rebuilt on disk: a copy of its folder in which the arrays of ``PERTURBED`` hold the noise of ``perturbed_arrays``
    (every other file is the same), hashed again, so that it is a complete store of its own."""
    folder = tmp_path_factory.mktemp('perturbed_s1')
    root = folder / 's1'
    shutil.copytree(toy_value_s1.root, root)
    for name, new in perturbed_arrays(toy_value_s1).items():
        on_disk = np.load(root / f'{name}.npy', mmap_mode='r+')
        on_disk[...] = new
        on_disk.flush()
        del on_disk
    s = Store(root)
    s.finalize()
    s.verify()
    yield s
    del s
    gc.collect()                                   # Windows does not delete a folder with open memory maps
    shutil.rmtree(folder, ignore_errors=True)


@pytest.fixture(scope='module')
def world(toy_value_s1, leak_s2, perturbed_s1):
    rows = compose.row_table(toy_value_s1, leak_s2, ROLES)
    K, O = true_ko(toy_value_s1, rows)
    return World(s1=toy_value_s1, s2=leak_s2, perturbed=perturbed_s1, rows=rows, K=K, O=O)


def test_the_truth_of_the_toy_is_there_to_be_leaked(world):
    """Without true outcomes, and perturbed arrays that differ, the tests below would compare nothing."""
    assert len(world.rows) == len(ALL_ROWS) and world.K.shape == (22, 4) and world.O.shape == (22, 18)
    assert (world.K[:, :2].sum(axis=1) > 0).sum() >= 8 and (world.O.sum(axis=1) > 0).sum() >= 4           # rows with kills and objectives
    assert world.K[:, 2:].max() >= 900 and world.rows.destroyed.any()
    src = np.asarray(world.s2.array('views/src'))
    assert (src != 0).any() and (src[world.rows.views] == 0).all()           # the toy S2 has event views, and the table does not use them
    same_match = world.rows.match[1:] == world.rows.match[:-1]
    assert same_match.sum() == 19 and (np.diff(world.rows.t0_ms)[same_match] == 60_000).all()       # t0 + 60 s of a row: the t0 of the next


def test_the_perturbed_store_differs_from_the_toy_in_every_entry_of_the_eight_arrays_and_in_no_other(world):
    assert set(world.perturbed.manifest['arrays']) == set(world.s1.manifest['arrays'])
    for name in world.s1.manifest['arrays']:
        a, b = np.asarray(world.s1.array(name)), np.asarray(world.perturbed.array(name))
        assert a.shape == b.shape and a.dtype == b.dtype, name
        assert (a != b).all() if name in PERTURBED else a.tobytes() == b.tobytes(), name          # (a NaN is not equal to itself)


# =====================================================================================================================================
# the three tests of the brief, on the real code
# =====================================================================================================================================
def test_a_masked_inputs_are_t0_row_and_sampled_outcomes_only(world):                       # (a) structure
    check_a(world)


@pytest.mark.parametrize('kind', ['closed', 'ko'])
def test_b_true_K_O_attached_leaves_composition_bit_identical(world, kind):                  # (b) the composition path
    check_b(real_b(world, kind), closed_form=kind == 'closed')


@pytest.mark.parametrize('kind', ['closed', 'ko'])
def test_c_perturbed_delta_K_O_leave_composition_unchanged(world, kind):                     # (c) invariance
    check_c(world, real_c(world, kind), closed_form=kind == 'closed')


# =====================================================================================================================================
# RED: each check fails on deliberately leaking variants (monkeypatch inside the tests, never in production code)
# =====================================================================================================================================
def from_rows(comp):
    """The true (K, O) as attributes of the row table of ``comp``, or None where there are none."""
    return (comp.rows.true_K, comp.rows.true_O) if hasattr(comp.rows, 'true_K') else None


def from_heads(comp):
    """The true (K, O) as attributes of the heads of ``comp``, or None where there are none."""
    return (comp.heads.true_K, comp.heads.true_O) if hasattr(comp.heads, 'true_K') else None


def reading_the_store(monkeypatch) -> None:
    """The row table also reads the tables of the engagements when the store has them, and attaches the true K and O to what it returns:
    the medium ``store`` (what it finds there it leaves on the table, where ``from_rows`` looks)."""
    real = compose.row_table

    def row_table(s1, s2, roles):
        rows = real(s1, s2, roles)
        try:
            K, O = true_ko(s1, rows)
        except FileNotFoundError:
            return rows
        return attached(rows, true_K=K, true_O=O)

    monkeypatch.setattr(compose, 'row_table', row_table)


def feeding_the_heads(monkeypatch, comp, find) -> None:
    """The composer hands the object head and the conversion head the TRUE K and O of the row (the pair ``find(comp)`` returns; nothing
    is leaked while it returns None) instead of what it sampled."""
    heads = comp.heads
    obj_rate, conv_rate = heads.obj_rate, heads.conv_rate

    def leaking_obj_rate(rows, q, m, z, s, K):
        truth = find(comp)
        return obj_rate(rows, q, m, z, s, K if truth is None else truth[0][np.asarray(rows)])

    def leaking_conv_rate(m4_inputs):
        truth = find(comp)
        if truth is None:
            return conv_rate(m4_inputs)
        x = np.array(m4_inputs)
        row, d = x[:, 0].astype(np.int64), x.shape[1] - 25                 # the toy row view starts with the row; then q, m, z, K, O
        x[:, d + 3:d + 5], x[:, d + 5:d + 7], x[:, d + 7:] = truth[0][row, :2], truth[0][row, 2:] / 1000.0, truth[1][row]
        return conv_rate(x)

    monkeypatch.setattr(heads, 'obj_rate', leaking_obj_rate)
    monkeypatch.setattr(heads, 'conv_rate', leaking_conv_rate)


def valuing_the_true_gold(monkeypatch, comp, find) -> None:
    """The value of an onset takes the true kill gold of the row instead of the gold that the composer sampled."""
    draw_row = comp._draw_row

    def leaking_draw_row(i):
        d, truth = draw_row(i), find(comp)
        if truth is None:
            return d
        sampled = d.gold[:, 0] - d.gold[:, 1]
        return dataclasses.replace(d, value=d.value - sampled + (truth[0][i, 2] - truth[0][i, 3]))

    monkeypatch.setattr(comp, '_draw_row', leaking_draw_row)


def expecting_the_true_gold(monkeypatch, comp, find) -> None:
    """The closed form adds the true kill gold of the row."""
    expect_row = comp._expect_row

    def leaking_expect_row(i):
        truth = find(comp)
        return expect_row(i) + (0.0 if truth is None else truth[0][i, 2] - truth[0][i, 3])

    monkeypatch.setattr(comp, '_expect_row', leaking_expect_row)


FIND = {'rows': from_rows, 'heads': from_heads, 'store': from_rows}


class Mechanism(NamedTuple):
    """How a leak uses the true (K, O): ``use(monkeypatch, comp, find)`` patches the composer; ``kind`` the toy heads it is shown with (the
    feed needs heads that read K and O); ``closed_form`` whether the check also compares ``expect``; ``output`` the word that the message
    of the ``Leak`` must hold: the first output that differs."""
    use: object
    kind: str
    closed_form: bool
    output: str


MECHANISMS = {'feed': Mechanism(feeding_the_heads, 'ko', False, 'simulate'),
              'value': Mechanism(valuing_the_true_gold, 'closed', True, 'simulate'),
              'closed form': Mechanism(expecting_the_true_gold, 'closed', True, 'expect')}


def leaking(monkeypatch, make, medium: str, mechanism: str):
    """``make`` of a check, whose composers leak: the true (K, O) that they find in ``medium`` go where ``mechanism`` puts them."""
    def leaking_make(arg):
        comp = make(arg)
        MECHANISMS[mechanism].use(monkeypatch, comp, FIND[medium])
        return comp
    return leaking_make


def depending_on(monkeypatch, name: str) -> None:
    """The row table depends on the S1 array ``name``: the end of the game of a row is moved by that array's numbers for the first
    engagement that starts in its window (a position-weighted sum, so that no permutation of the entries goes unnoticed)."""
    real = compose.row_table

    def row_table(s1, s2, roles):
        rows = real(s1, s2, roles)
        a = np.asarray(s1.array(name)).astype(np.float64)
        shift = [0 if u < 0 else int(np.dot(np.ravel(a[u]), 1 + np.arange(np.size(a[u])))) for u in first_unit(s1, rows)]
        return dataclasses.replace(rows, T_ms=rows.T_ms + np.array(shift, dtype=np.int64))

    monkeypatch.setattr(compose, 'row_table', row_table)


def asking_for_other_times(monkeypatch, kind: str) -> None:
    """The row table also asks ``towertier.destroyed_before`` for the tower state at another time than t0 and keeps the answer on the
    table (the variants M1c and M1d of the review of Task 11, and one more):

        later      M1c: the state at t0 + 60 s, the time of the next frame, as ``later`` (n, 2, 4)
        bins       M1d: the state at the middle of each occurrence bin, t0 + (q + 0.5) 10 s, as ``by_bin`` (n, 18, 2, 4)
        successor  the state at the t0 of the next row of the match, as ``later``: every pair that it asks for is itself the (match, t0) of
                   a row (the rows are 60 s apart), but each of them is asked for twice
    """
    real = compose.row_table

    def row_table(s1, s2, roles):
        rows = real(s1, s2, roles)
        if kind == 'later':
            return attached(rows, later=towertier.destroyed_before(s1, rows.match, rows.t0_ms + 60_000))
        if kind == 'bins':
            middle = C.BIN_MS * np.arange(C.N_BINS) + C.BIN_MS // 2
            return attached(rows, by_bin=np.stack([towertier.destroyed_before(s1, rows.match, rows.t0_ms + m) for m in middle], axis=1))
        has_next = np.flatnonzero(rows.match[1:] == rows.match[:-1])                       # the rows with a next row in their match
        later = np.zeros_like(rows.destroyed)
        later[has_next] = towertier.destroyed_before(s1, rows.match[has_next], rows.t0_ms[has_next + 1])
        return attached(rows, later=later)

    monkeypatch.setattr(compose, 'row_table', row_table)


def using_the_future_state(monkeypatch, comp, kind: str) -> None:
    """The composer uses the tower state that the leaking row table kept. ``later`` and ``successor``: 100 gold for each tower that red
    has lost and minus 100 for each that blue has lost, at that later time, are added to the value of every onset (M1c; it patches the
    private ``_draw_row``). ``bins``: a predicted tower is priced with the state at the middle of its bin (M1d; it patches the private
    ``_cell_values``, which the committed composer and the one that prices the towers row by row both have)."""
    tower = 2 * C.O_KINDS.index('tower')
    if kind == 'bins':
        cell_values = comp._cell_values

        def leaking_cell_values(i):
            v = np.array(cell_values(i))
            t_s = (comp.rows.t0_ms[i] + C.BIN_MS * np.arange(C.N_BINS) + C.BIN_MS // 2) / 1000.0
            for taker in (0, 1):
                who = np.full(C.N_BINS, taker)
                prob = comp.tier_model.prob(towertier.victim_counts(comp.rows.by_bin[i], who), who, t_s)
                v[:, tower + taker] = (1 - 2 * taker) * towertier.expected_tower_price(prob, comp.prices)      # blue +, red -
            return v

        monkeypatch.setattr(comp, '_cell_values', leaking_cell_values)
        return
    draw_row = comp._draw_row

    def leaking_draw_row(i):
        d, later = draw_row(i), comp.rows.later[i]
        return dataclasses.replace(d, value=d.value + 100.0 * (later[1].sum() - later[0].sum()))

    monkeypatch.setattr(comp, '_draw_row', leaking_draw_row)


# ---- (a) ------------------------------------------------------------------------------------------------------------------------------
def test_red_a_builder_that_accepts_the_true_delta_is_caught(monkeypatch, world):
    real = features.m4_masked_inputs

    def with_delta(row_view, q, m, z, K, O, delta=None):
        x = real(row_view, q, m, z, K, O)
        return x if delta is None else np.hstack([x, np.asarray(delta, dtype=np.float32)[:, None]])

    monkeypatch.setattr(features, 'm4_masked_inputs', with_delta)
    with pytest.raises(Leak, match='parameters'):
        check_a(world)


def test_red_a_builder_that_adds_a_column_is_caught(monkeypatch, world):
    real = features.m4_masked_inputs

    def with_column(row_view, q, m, z, K, O):
        x = real(row_view, q, m, z, K, O)
        return np.hstack([x, np.full((len(x), 1), 55_000.0, dtype=np.float32)])          # the column that a lookup of the true delta fills

    monkeypatch.setattr(features, 'm4_masked_inputs', with_column)
    with pytest.raises(Leak, match='m4_masked_inputs builds'):
        check_a(world)


def test_red_a_composer_that_hands_the_conversion_head_one_more_column_is_caught(monkeypatch, world):
    real = Recording.conv_rate

    def with_delta(self, m4_inputs):
        return real(self, np.hstack([m4_inputs, np.full((len(m4_inputs), 1), 99_000.0, dtype=np.float32)]))

    monkeypatch.setattr(Recording, 'conv_rate', with_delta)
    with pytest.raises(Leak, match='conversion head is handed'):
        check_a(world)


@pytest.mark.parametrize('name,message', [
    ('units/delta', 'names an array of the engagements'), ('onsets/row', 'names an array of the engagements'),
    ('kills/t', 'names an array of the engagements'), ('objectives/t', 'names an array of the engagements'),
    ('event_rows/ref', 'names an array of the engagements'), ('frames/num', 'outside the t0 tables')])
def test_red_a_source_that_names_an_array_that_is_not_a_t0_one_is_caught(world, name, message):
    leaking_source = SOURCE.read_text(encoding='utf-8') + f'\n_LEAK = {name!r}\n'
    with pytest.raises(Leak, match=message):
        check_a(world, source=leaking_source)


@pytest.mark.parametrize('name,message', [('objectives/t', 'outside towertier.destroyed_before'), ('units/delta', 't0 tables only')])
def test_red_a_row_table_that_reads_another_table_than_the_t0_ones_is_caught(monkeypatch, world, name, message):
    real = compose.row_table

    def row_table(s1, s2, roles):
        rows = real(s1, s2, roles)
        s1.array(name)                                    # the objectives not through towertier.destroyed_before, or an engagement table
        return rows

    monkeypatch.setattr(compose, 'row_table', row_table)
    with pytest.raises(Leak, match=message):
        check_a(world)


def test_red_a_row_table_that_picks_an_event_view_is_caught(monkeypatch, world):
    real = compose.row_table

    def row_table(s1, s2, roles):
        rows = real(s1, s2, roles)
        src, match = np.asarray(s2.array('views/src')), np.asarray(s2.array('views/match'))
        views = [int(np.flatnonzero((src != 0) & (match == m))[0]) for m in rows.match]         # a view at an event time, after t0
        return dataclasses.replace(rows, views=views)

    monkeypatch.setattr(compose, 'row_table', row_table)
    with pytest.raises(Leak, match='not the frame-clock views'):
        check_a(world)


@pytest.mark.parametrize('kind,message', [('later', 'that are not the'), ('bins', 'that are not the'), ('successor', 'not once for each')])
def test_red_a_row_table_that_asks_for_the_tower_state_at_another_time_than_t0_is_caught(monkeypatch, world, kind, message):
    honest = make_composer(world.s1, world.s2, ToyHeads())                          # built before the leak is put in
    asking_for_other_times(monkeypatch, kind)
    leaking_composer = make_composer(world.s1, world.s2, ToyHeads())
    using_the_future_state(monkeypatch, leaking_composer, kind)
    # the variant is a leak of something: with the state-sensitive tower prices the composed values move (the spy is not the only witness)
    assert not np.array_equal(honest.simulate(ALL_ROWS).mean, leaking_composer.simulate(ALL_ROWS).mean), 'set-up: the variant moves nothing'
    if kind == 'bins':
        assert not np.array_equal(honest.expect(EXPECT_ROWS), leaking_composer.expect(EXPECT_ROWS)), 'set-up: the closed form is not moved'
    with pytest.raises(Leak, match=message):
        check_a(world)


def test_the_tower_prices_of_the_composers_depend_on_the_state_the_taker_and_the_minute():
    model = state_tier_model()

    def price(destroyed, taker=0, minutes=5.0):
        return float(towertier.expected_tower_price(model.prob(np.array([destroyed]), [taker], [60.0 * minutes]), PRICES)[0])

    start = price([0, 0, 0, 0])
    assert price([1, 0, 0, 0]) - start > 50.0 and price([1, 1, 0, 0]) - price([1, 0, 0, 0]) > 50.0          # the next tier follows
    assert abs(price([0, 0, 0, 0], taker=1) - start) > 3.0 and abs(price([0, 0, 0, 0], minutes=15.0) - start) > 3.0


def test_with_the_constant_tower_prices_of_s3_toys_a_tower_state_at_the_wrong_time_moves_no_value(monkeypatch, world):
    """Why the composers of this file have their own tower prices: with the 400 of every tier of ``s3_toys`` the variant of the bins leaks
    the state of the future and the composed values do not show it (to rounding), so that only the spy could."""
    constant = dict(tier_model=toy_tier_model(), prices=TOY_COMPOSE_PRICES)
    honest = make_composer(world.s1, world.s2, ToyHeads(), **constant)
    asking_for_other_times(monkeypatch, 'bins')
    leaking_composer = make_composer(world.s1, world.s2, ToyHeads(), **constant)
    using_the_future_state(monkeypatch, leaking_composer, 'bins')
    assert np.allclose(honest.simulate(ALL_ROWS).mean, leaking_composer.simulate(ALL_ROWS).mean, rtol=1e-12, atol=1e-9)
    assert np.allclose(honest.expect(EXPECT_ROWS), leaking_composer.expect(EXPECT_ROWS), rtol=1e-12, atol=1e-9)


# ---- (b) ------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('medium,mechanism', [('rows', 'feed'), ('heads', 'feed'), ('store', 'feed'), ('rows', 'value'),
                                              ('rows', 'closed form')])
def test_red_b_composer_that_uses_the_true_K_O_is_caught(monkeypatch, world, medium, mechanism):
    if medium == 'store':
        reading_the_store(monkeypatch)
    how = MECHANISMS[mechanism]
    make = leaking(monkeypatch, real_b(world, how.kind, attach=medium != 'store'), medium, mechanism)      # 'store': only the arrays differ
    with pytest.raises(Leak, match=how.output):
        check_b(make, closed_form=how.closed_form)


# ---- (c) ------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('mechanism', ['feed', 'value', 'closed form'])
def test_red_c_composer_that_reads_the_s1_K_and_O_is_caught(monkeypatch, world, mechanism):
    reading_the_store(monkeypatch)
    how = MECHANISMS[mechanism]
    with pytest.raises(Leak, match=how.output):
        check_c(world, leaking(monkeypatch, real_c(world, how.kind), 'store', mechanism), closed_form=how.closed_form)


@pytest.mark.parametrize('name', PERTURBED)
def test_red_c_row_table_that_depends_on_any_perturbed_array_is_caught(monkeypatch, world, name):
    depending_on(monkeypatch, name)
    with pytest.raises(Leak, match='the row table field T_ms'):
        check_c(world, real_c(world, 'closed'), closed_form=False)


# S4a Task6: real EPT adapter arm; existing S3 arms above are unchanged.
@pytest.fixture(scope='module')
def ept_leak_world(tmp_path_factory):
    from c_ept.tests.test_ept_compose import build_compose_world
    w = build_compose_world(tmp_path_factory.mktemp('ept_leakage'))
    altered = perturbed_arrays(w.s1)
    class Perturbed:
        def array(self, name):
            return altered[name] if name in altered else w.s1.array(name)
    rows = compose.row_table(w.s1, w.s2, ROLES)
    K, O = true_ko(w.s1, rows)
    assert len(rows) == len(ALL_ROWS) and (K[:, 2:] > 0).any()
    return World(w.s1, w.s2, Perturbed(), rows, K, O)


@pytest.fixture(scope='module')
def ept_leak_stats(ept_leak_world):
    from c_ept.ept import data
    return data.fit_stats(ept_leak_world.s1, ept_leak_world.s2)


def ept_leak_heads(stats):
    import torch
    from c_ept.ept.compose_heads import EPTComposerHeads
    from c_ept.tests.test_ept_compose import model
    m = model(seed=17)
    # Controlled synthetic rates keep the simulation nonvacuous and bounded.
    # Real K/O conditioning weights remain active, so feed mutants change outputs.
    with torch.no_grad():
        m.heads.rate_net.out.weight.zero_()
        m.heads.rate_net.out.bias.fill_(np.log(0.002))
        m.heads.obj_net.out.bias.fill_(-2.)
        m.heads.conv_net.out.bias.fill_(-5.)
    return EPTComposerHeads(m, stats)


def ept_b(world, stats):
    def make(with_truth):
        s1 = world.s1 if with_truth else ViewStore(world.s1, hide=ENGAGEMENT)
        return make_composer(s1, world.s2, ept_leak_heads(stats), truth=(world.K, world.O) if with_truth else None)
    return make


def ept_c(world, stats):
    def make(s1):
        return make_composer(s1, world.s2, ept_leak_heads(stats))
    return make


def test_ept_a_masked_inputs_are_only_t0_handles_and_sampled_conditions(ept_leak_world, ept_leak_stats, monkeypatch):
    assert list(inspect.signature(features.m4_masked_inputs).parameters) == M4_PARAMETERS
    world = ept_leak_world
    h = ept_leak_heads(ept_leak_stats)
    seen = []
    original = h.conv_rate
    def recording(X):
        seen.append(X.copy())
        return original(X)
    monkeypatch.setattr(h, 'conv_rate', recording)
    comp = make_composer(world.s1, world.s2, h)
    d = comp.simulate_onsets(0)
    assert len(d.q) > 0 and seen
    expected = features.m4_masked_inputs(h.row_view(np.zeros(len(d.q), dtype=np.int64)), d.q, d.m, d.z, np.column_stack([d.k, d.gold]), d.O)
    np.testing.assert_array_equal(np.vstack(seen), expected)
    assert expected.shape[1] == 27


def test_ept_b_true_K_O_attached_leaves_composition_bit_identical(ept_leak_world, ept_leak_stats):
    check_b(ept_b(ept_leak_world, ept_leak_stats), closed_form=False)


def test_ept_c_perturbed_delta_K_O_leave_composition_bit_identical(ept_leak_world, ept_leak_stats):
    check_c(ept_leak_world, ept_c(ept_leak_world, ept_leak_stats), closed_form=False)


@pytest.mark.parametrize('medium', ['rows', 'heads'])
def test_ept_red_b_actual_true_K_O_feed_is_caught(monkeypatch, ept_leak_world, ept_leak_stats, medium):
    with pytest.raises(Leak, match='simulate'):
        check_b(leaking(monkeypatch, ept_b(ept_leak_world, ept_leak_stats), medium, 'feed'), closed_form=False)


def test_ept_red_c_true_K_O_from_store_is_caught(monkeypatch, ept_leak_world, ept_leak_stats):
    reading_the_store(monkeypatch)
    with pytest.raises(Leak, match='simulate'):
        check_c(ept_leak_world, leaking(monkeypatch, ept_c(ept_leak_world, ept_leak_stats), 'store', 'feed'), closed_form=False)


def test_ept_red_c_true_delta_dependency_is_caught(monkeypatch, ept_leak_world, ept_leak_stats):
    depending_on(monkeypatch, 'units/delta')
    with pytest.raises(Leak, match='the row table field T_ms'):
        check_c(ept_leak_world, ept_c(ept_leak_world, ept_leak_stats), closed_form=False)


def test_ept_red_b_true_gold_value_is_caught(monkeypatch, ept_leak_world, ept_leak_stats):
    # Reuse the value mutant without the unrelated toy closed-form expectation.
    make = ept_b(ept_leak_world, ept_leak_stats)
    def mutant(arg):
        comp = make(arg)
        valuing_the_true_gold(monkeypatch, comp, from_rows)
        return comp
    with pytest.raises(Leak, match='simulate'):
        check_b(mutant, closed_form=False)


# =====================================================================================================================================
# (d) the end of the game T (S4b preparation M12, decision D14): a t0 forecast is T-blind
# =====================================================================================================================================
# D14: T is allowed in the training censoring (P10) and in the filter of the primary population (t0 + H <= T - 30 s), nowhere else. The composed V_start and the occurrence probabilities of the evaluation use
# the nominal exposures (10 s per occurrence bin, 60 s of conversion window) and no head and no composer reads T. The check (d) takes two composers of the same rows whose tables differ in T (and so in
# the observed exposure) and asks for the same composition bit for bit on the primary rows of both; the RED variants below (a head that reads T, a nominal composer that reads T or the observed
# exposure, and the observed composer itself) must fail it.
T_NEAR_S, T_FAR_S = 120, 200                # T = t0 + 120 s / t0 + 200 s: both primary at H = 60 s (hi - t0 = 90 s / 170 s); the conversion windows of the later bins are cut by the first and not by the second


def with_game_end(rows, after_s: int):
    """A copy of the row table whose game ends ``after_s`` seconds after each t0, with the exposure of the occurrence bins that ``frame_rows`` gives for it (hi = min(t0 + 180 s, T - 30 s))."""
    T = rows.t0_ms + 1000 * after_s
    hi = np.minimum(rows.t0_ms + C.H_OCC_MS, T - C.END_CUT_MS)
    exposure = np.clip(hi[:, None] - (rows.t0_ms[:, None] + C.BIN_MS * np.arange(C.N_BINS)[None, :]), 0, C.BIN_MS)
    return dataclasses.replace(rows, T_ms=T, exposure_ms=exposure)


def primary_both(a, b, H: int = 60) -> np.ndarray:
    """The rows (numbers in the table) that are primary at the horizon ``H`` under both tables: hi - t0 >= H x 1000 with hi = min(t0 + 180 s, T - 30 s)."""
    from c_ept.evaluation.endpoints import primary_rows
    masks = [primary_rows(r.t0_ms, np.minimum(r.t0_ms + C.H_OCC_MS, r.T_ms - C.END_CUT_MS), H) for r in (a, b)]
    return np.flatnonzero(masks[0] & masks[1])


def check_d(plain, other, closed_form: bool, H: int = 60) -> None:
    """(d) ``plain`` and ``other`` compose the same rows, their tables differ in the end of the game T (and in the occurrence exposure that follows from it): on the rows that are primary at the horizon ``H`` in
    both, the composition (``simulate``: mean, sd, 19 quantiles; the raw draws of the first rows; ``expected_onsets``; ``expect`` for heads with a closed form) is bit for bit the same, or ``Leak``."""
    idx = primary_both(plain.rows, other.rows, H)
    assert len(idx) >= 6 and not np.array_equal(plain.rows.T_ms, other.rows.T_ms) and not np.array_equal(plain.rows.exposure_ms, other.rows.exposure_ms), 'set-up: the two tables must differ in T on primary rows'
    a, b = plain.simulate(idx), other.simulate(idx)
    assert (a.mean != 0).any() and (a.q19 != 0).any(), 'set-up: the simulation is empty, so that equality would prove nothing'
    for field in ('mean', 'sd', 'q19'):
        same(f'(d) T changed: simulate {field}', getattr(a, field), getattr(b, field))
    same('(d) T changed: expected onsets', plain.expected_onsets(idx), other.expected_onsets(idx))
    for i in idx[:3]:
        da, db = plain.simulate_onsets(int(i)), other.simulate_onsets(int(i))
        for field in ('sim', 'q', 'm', 'z', 's', 'k', 'gold', 'O', 'conv', 'value'):
            same(f'(d) T changed: the draws of the row {i}: {field}', getattr(da, field), getattr(db, field))
    if closed_form:
        e = plain.expect(idx)
        assert (e != 0).any(), 'set-up: the closed form is zero, so that equality would prove nothing'
        same('(d) T changed: expect', e, other.expect(idx))


def t_blind_pair(world, kind: str, exposure: str, heads=None):
    """Two composers of the toy rows of ``world`` with the games ending 120 s and 200 s after t0, composed with the given exposure; ``heads(rows)`` makes the heads of a table (default the toy heads of ``kind``)."""
    out = []
    for after in (T_NEAR_S, T_FAR_S):
        rows = with_game_end(world.rows, after)
        h = HEADS[kind]() if heads is None else heads(rows)
        out.append(compose.Composer(h, PRICES, state_tier_model(), TOY_GOLD_REPS, rows, world.s2, S=S_SIM, seed=SEED, exposure=exposure))
    return out


class TReadingHeads(ToyHeads):
    """A head set that reads the end of the game of its table: the onset rate grows with T (a leak that D14 forbids: no model reads T)."""

    def __init__(self, T_ms):
        super().__init__()
        self.T = np.asarray(T_ms, dtype=np.float64)

    def rate(self, rows) -> np.ndarray:
        return super().rate(rows) * (1.0 + self.T[self._rows(rows)][:, None] / 1e7)


@pytest.mark.parametrize('kind', ['closed', 'ko'])
def test_nominal_composition_is_bit_identical_when_T_changes_on_primary_rows(world, kind):
    check_d(*t_blind_pair(world, kind, 'nominal'), closed_form=kind == 'closed')


def test_nominal_composition_with_the_real_ept_heads_is_bit_identical_when_T_changes_on_primary_rows(ept_leak_world, ept_leak_stats):
    world = ept_leak_world
    pair = []
    for after in (T_NEAR_S, T_FAR_S):
        pair.append(compose.Composer(ept_leak_heads(ept_leak_stats), PRICES, state_tier_model(), TOY_GOLD_REPS, with_game_end(world.rows, after), world.s2, S=S_SIM, seed=SEED, exposure='nominal'))
    check_d(*pair, closed_form=False)


@pytest.mark.parametrize('kind', ['closed', 'ko'])
def test_red_the_observed_composition_depends_on_T_and_the_check_catches_it(world, kind):
    """Why D14 exists: the observed exposure cuts the conversion windows at T, so the observed composition of a primary row moves with T and the check fails."""
    with pytest.raises(Leak, match='T changed'):
        check_d(*t_blind_pair(world, kind, 'observed'), closed_form=kind == 'closed')


def test_red_variant_head_reading_T_is_caught(world):
    """A head that reads T makes the nominal composition depend on T: the check fails on the simulation. (Only the nominal case says something about the head: the observed composer already fails the check
    by itself, ``test_red_the_observed_composition_depends_on_T_and_the_check_catches_it``, so an observed variant would not isolate the head; M12 review m5.)"""
    with pytest.raises(Leak, match='T changed'):
        check_d(*t_blind_pair(world, 'closed', 'nominal', heads=lambda rows: TReadingHeads(rows.T_ms)), closed_form=True)
    check_d(*t_blind_pair(world, 'closed', 'nominal'), closed_form=True)                    # control: the same pair with heads that do not read T passes


def test_red_a_nominal_composer_that_reads_T_for_the_conversion_window_is_caught(monkeypatch, world):
    def reads_T(self, i):
        return compose.conv_exposure_s(self.rows.t0_ms[i], np.arange(compose.N_Q), self.rows.T_ms[i])
    monkeypatch.setattr(compose.Composer, '_conv_exposure', reads_T)
    with pytest.raises(Leak, match='T changed'):
        check_d(*t_blind_pair(world, 'ko', 'nominal'), closed_form=False)


def test_red_a_nominal_composer_that_reads_the_observed_occurrence_exposure_is_caught(monkeypatch, world):
    def reads_exposure(self, i):
        return self.rows.exposure_ms[i] / 1000.0
    monkeypatch.setattr(compose.Composer, '_seconds', reads_exposure)
    with pytest.raises(Leak, match='T changed'):
        check_d(*t_blind_pair(world, 'closed', 'nominal'), closed_form=True)


def test_T_is_read_by_the_composer_only_in_the_two_observed_exposure_methods_behind_the_nominal_branch():
    """Structure: in the class ``Composer`` the attributes ``T_ms`` and ``exposure_ms`` of the table are read in ``_seconds`` and ``_conv_exposure`` only, and each of the two returns before the read when the
    exposure is nominal. Nothing else of the composer, the draw, the closed form or the heads' binding, can read T."""
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    composer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Composer')
    readers = {}
    for method in (n for n in composer.body if isinstance(n, ast.FunctionDef)):
        names = {a.attr for a in ast.walk(method) if isinstance(a, ast.Attribute) and a.attr in ('T_ms', 'exposure_ms')}
        if names:
            readers[method.name] = (names, method)
    assert set(readers) == {'_seconds', '_conv_exposure'}, sorted(readers)
    for name, (names, method) in readers.items():
        body = [s for s in method.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]            # without the docstring
        guard = body[0]
        assert isinstance(guard, ast.If) and isinstance(guard.body[-1], ast.Return) and ast.unparse(guard.test) == "self.exposure == 'nominal'", name
        assert not {a.attr for stmt in guard.body for a in ast.walk(stmt) if isinstance(a, ast.Attribute)} & {'T_ms', 'exposure_ms'}, name


def test_the_evaluation_adapters_name_the_game_end_nowhere_and_read_the_window_end_in_the_population_filter_only():
    """The population filter reads ``frame_rows/t0`` and ``frame_rows/hi`` (hi = min(t0 + 180 s, T - 30 s)) in ``_primary_masks`` and nowhere else; no module of ``c_ept/evaluation`` (every ``*.py`` of the folder,
    also a future one) names ``T_ms``, and none but ``adapters.py`` names the two frame-row arrays (M12 review m6)."""
    root = Path(compose.__file__).parent.parent / 'evaluation'
    modules = sorted(root.glob('*.py'))
    assert {m.name for m in modules} >= {'adapters.py', 'endpoints.py', 'metrics.py', 'inference.py', 'guard.py', 'runtime.py', 'stores.py', 'proper.py', 'reporting.py'}
    users = set()
    for module in modules:
        tree = ast.parse(module.read_text(encoding='utf-8'))
        constants = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        assert not any('T_ms' in c for c in constants) and not any(isinstance(n, ast.Attribute) and n.attr == 'T_ms' for n in ast.walk(tree)), module.name
        names = lambda n: isinstance(n, ast.Constant) and n.value in ('frame_rows/hi', 'frame_rows/t0')                  # noqa: E731
        users |= {(module.name, f.name) for f in ast.walk(tree) if isinstance(f, ast.FunctionDef) and any(names(n) for n in ast.walk(f))}
        if any(names(n) for n in ast.walk(tree)) and not any(u[0] == module.name for u in users):
            users.add((module.name, '<module level>'))
    assert users == {('adapters.py', '_primary_masks')}, users
