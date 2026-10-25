"""Tests of the EPT losses (stage S4a Task 4): ``ept.loss`` and ``ept.scales``.

No GPU is used, no raw file, catalog or real run folder is read (ruling E6: Strata is up while this task is coded). The B0 scales are tried on
a toy run built in a temporary folder (tiny B0 heads, the toy stores of ``s3_toys`` and hashed model files). The numbers of the hand cases are
those of math-control note 04 (``docs/superpowers/notes/2026-10-23-math-control-04.md``, recomputed by ``toy_checks_04.py``).

    equality with nll.py   the torch NLL of every head equals the numpy score of ``c_ept.nll`` on the same rows (rtol 1e-6), for random inputs in
                           float32 and float64, for the outputs of the real heads, and under ruling R-S4a-10 (the head's mean already holds the
                           exposure); a row reads exactly its own cells of the output (the gradient shows which)
    rows that add nothing  a cell without exposure and a category -1 add 0 and get no gradient, whatever the junk in the output; data errors
                           and bad laws are refused
    the mean and the sum   the mean over the rows of a head, the size-normalised mean of the M3 experts, the plain mean of the size gate, the
                           rows without a category or exposure left out of the mean, the sum divided by the B0 scales (note 04, B and C)
    B0 scales              the scale of a head from its B0 rows, and ``b0_scales`` on a toy run: FIT rows only, hashed model files, a run
                           whose files are not the recorded ones is refused, the file is not written over, the command line
    module text            no GPU, no cohort module, no holdout switch, no key
"""
import dataclasses
import gc
import hashlib
import json
import math
import shutil
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from c_ept.data import head_rows
from c_ept.models import nll
from c_ept import runlog
from c_ept.data import store_core
from c_ept.ept import data, encoder, heads, loss
from c_ept.tests.conftest import all_imports, module_level_imports
from c_ept.tests.ept_toys import build_ept_world, encoder_batch, toy_static
from c_ept.tests.s3_toys import KEY_PREFIX, heads_world  # noqa: F401  (the fixture is imported by ruling R1)

HEAD_NAMES = head_rows.HEAD_NAMES
N_Q, N_M, N_ZONE, N_SIZE, N_LAYER, N_GOLD, N_CELLS = (head_rows.N_Q, head_rows.N_M, head_rows.N_ZONE, head_rows.N_SIZE, head_rows.N_LAYER,
                                                      head_rows.N_GOLD, head_rows.N_CELLS)
F64 = torch.float64

# ---- the numbers of math-control note 04 -------------------------------------------------------------------------------------------------
# A. four cells of one frame row (the other 14 cells have no exposure and no count); the head's mean is mu = lambda e, the third is junk
TOY_Y = (1, 0, 0, 2)
TOY_MU = (0.2, 0.2, 123.0, 0.5)
TOY_E = (10.0, 10.0, 0.0, 10.0)
TOY_ROW = 4.5888795                         # = TOY_POISSON_NLL of s3_toys: the same cells in the units of note 03 C
TOY_ROW_TWICE = 5.781124                    # the row if the exposure were multiplied in once more (wrong)
TOY_GOLD_P, TOY_GOLD_NLL = 0.4, 0.9162907   # a gold bin of probability 0.4
# B1. one M3 expert in a batch: the row scores and the true sizes (T, T, S, S, S, P: no ASYM row); an ASYM row of score 6 can be added
TOY_B1_SCORES, TOY_B1_SIZES = (2.0, 4.0, 1.0, 1.0, 4.0, 10.0), (0, 0, 1, 1, 1, 3)
TOY_B1_MEAN, TOY_B1_PLAIN, TOY_B1_SCALE, TOY_B1_TERM = 5.0, 22 / 6, 2.5, 2.0
TOY_B1_ASYM_MEAN, TOY_B1_ASYM_PLAIN = 5.25, 4.0
# B2. the size gate: three rows of size T, one of size S; (plain mean, size-normalised mean) of the law (p_T, p_S)
TOY_B2_SIZES = (0, 0, 0, 1)
TOY_B2 = {(0.75, 0.25): (0.5623352, 0.8369883), (0.5, 0.5): (0.6931472, 0.6931472)}
# B3. a zone head: the rows with the zone -1 are not rows of the mean
TOY_B3_Y, TOY_B3_P = (3, -1, 0, -1, 7), (0.5, 0.25, 0.125)
TOY_B3_MEAN, TOY_B3_WRONG = 2 * math.log(2), 0.8317766
# C. two heads: the means, the B0 scales, the loss; both heads 4 % better
TOY_C_MEANS, TOY_C_SCALES, TOY_C_LOSS, TOY_C_AFTER = (6.0, 0.9), (8.0, 1.2), 1.5, 1.44


# =====================================================================================================================================
# helpers: random rows of every head, and the reference score written from the table of ept/heads.py with the numpy scores of nll.py
# =====================================================================================================================================
def random_block(head: str, n: int = 9, seed: int = 0, dtype=F64):
    """``(out, target, exposure)`` for ``n`` made-up rows of ``head`` in the layout of ``ept/heads.py`` (Task 3 report, section 2): ``out``
    the output of the head for the rows (laws are log-softmax mixtures, means are positive), ``target`` the block fields (``y`` and the
    index fields, int64), ``exposure`` the seconds (M2Rate (n, 18), M4Conv (n,)) or None. Some cells have no exposure (and count 0) and
    their mean is junk (123.0); the first rows have the category -1 (zone, gold) or no exposure at all."""
    g = torch.Generator().manual_seed(1000 * seed + sum(map(ord, head)))
    f64 = dict(generator=g, dtype=F64)

    def ints(low, high, *shape):
        return torch.randint(low, high, shape, generator=g)

    def law(*shape):
        return heads.mixed_log_softmax(torch.randn(*shape, **f64) * 1.5).to(dtype)

    def counts(mean):
        return torch.poisson(mean, generator=g).long()

    exposure, target = None, {}
    if head == 'M2Rate':
        e = torch.rand(n, N_Q, **f64) * 9 + 1
        e[torch.rand(n, N_Q, generator=g) < 0.3] = 0.0
        e[0] = 0.0                                                   # a frame row without any exposure
        mu = torch.rand(n, N_Q, **f64) * 0.6 * e
        out = torch.where(e > 0, mu, torch.full_like(mu, 123.0)).to(dtype)
        target, exposure = {'y': counts(mu)}, e.to(dtype)
    elif head == 'M4Conv':
        e = torch.rand(n, **f64) * 59 + 1
        e[1] = 0.0                                                   # a row without a conversion window
        mu = torch.rand(n, N_CELLS, **f64) * 0.05 * e[:, None]
        out = torch.where((e > 0)[:, None], mu, torch.full_like(mu, 123.0)).to(dtype)
        target, exposure = {'y': counts(mu)}, e.to(dtype)
    elif head == 'M3Obj':
        out = (torch.rand(n, N_CELLS, **f64) * 0.8 + 0.01).to(dtype)
        target = {'s': ints(0, N_SIZE, n), 'y': counts(out.double())}
    elif head == 'M3Kills':
        out = (torch.rand(n, N_SIZE, 2, **f64) * 4 + 0.05).to(dtype)
        s = ints(0, N_SIZE, n)
        target = {'s': s, 'y': counts(out.double()[torch.arange(n), s])}
    elif head == 'M2Mark':
        out = law(n, N_Q, N_M)
        target = {'q': ints(0, N_Q, n), 'y': ints(0, N_M, n)}
    elif head == 'M2Zone':
        out = law(n, N_Q, N_M, N_ZONE)
        y = ints(-1, N_ZONE, n)
        y[0] = -1
        target = {'q': ints(0, N_Q, n), 'm': ints(0, N_M, n), 'y': y}
    elif head == 'M3Size':
        out = law(n, N_SIZE)
        target = {'y': ints(0, N_SIZE, n)}
    elif head == 'M3Gold':
        out = law(n, N_SIZE, 2, N_LAYER, N_GOLD)
        y = ints(-1, N_GOLD, n)
        y[0] = -1
        target = {'s': ints(0, N_SIZE, n), 'team': ints(0, 2, n), 'layer': ints(1, N_LAYER + 1, n), 'y': y}
    else:
        raise ValueError(head)
    return out, target, exposure


def reference_rows(head: str, out, target, exposure=None) -> np.ndarray:
    """The NLL of every row by plain loops over the rows and the cells, with the numpy scores of ``c_ept.nll`` on the rows that the table of
    ``ept/heads.py`` names for the head. A Poisson head gives the mean mu = lambda e, so a cell scores ``nll.poisson(y, mu / e, e)``."""
    out = out.detach().double().numpy()
    t = {k: v.numpy() for k, v in target.items()}
    e = None if exposure is None else exposure.double().numpy()
    rows = np.zeros(out.shape[0])
    for i in range(len(rows)):
        if head in ('M2Rate', 'M4Conv', 'M3Kills', 'M3Obj'):
            if head == 'M2Rate':
                mu_i, e_i = out[i], e[i]
            elif head == 'M4Conv':
                mu_i, e_i = out[i], np.full(N_CELLS, e[i])
            elif head == 'M3Kills':
                mu_i, e_i = out[i, t['s'][i]], np.ones(2)
            else:
                mu_i, e_i = out[i], np.ones(N_CELLS)
            for y_c, mu_c, e_c in zip(t['y'][i], mu_i, e_i):
                if e_c > 0:
                    rows[i] += float(nll.poisson(y_c, mu_c / e_c, e_c))
        else:
            law = {'M2Mark': lambda: out[i, t['q'][i]], 'M2Zone': lambda: out[i, t['q'][i], t['m'][i]], 'M3Size': lambda: out[i],
                   'M3Gold': lambda: out[i, t['s'][i], t['team'][i], t['layer'][i] - 1]}[head]()
            score = nll.gold if head == 'M3Gold' else nll.categorical
            rows[i] = float(score(np.array([t['y'][i]]), np.exp(law)[None, :])[0])
    return rows


def read_cells(head: str, shape, target, exposure=None) -> torch.Tensor:
    """A bool mask of the shape of the output: the cells that the rows read (the table of ``ept/heads.py``), written as loops."""
    mask = torch.zeros(shape, dtype=torch.bool)
    for i in range(shape[0]):
        t = {k: int(v[i]) for k, v in target.items() if v.ndim == 1}
        if head == 'M2Rate':
            mask[i] = exposure[i] > 0
        elif head == 'M4Conv':
            mask[i] = bool(exposure[i] > 0)
        elif head == 'M3Obj':
            mask[i] = True
        elif head == 'M3Kills':
            mask[i, t['s']] = True
        elif head == 'M2Mark':
            mask[i, t['q'], t['y']] = True
        elif head == 'M2Zone' and t['y'] >= 0:
            mask[i, t['q'], t['m'], t['y']] = True
        elif head == 'M3Size':
            mask[i, t['y']] = True
        elif head == 'M3Gold' and t['y'] >= 0:
            mask[i, t['s'], t['team'], t['layer'] - 1, t['y']] = True
    return mask


# =====================================================================================================================================
# equality with nll.py
# =====================================================================================================================================
@pytest.mark.parametrize('dtype', [F64, torch.float32], ids=['float64', 'float32'])
@pytest.mark.parametrize('head', HEAD_NAMES)
def test_torch_nll_equals_numpy_nll_for_every_head(head, dtype):
    """Design 7 and 8, plan P4 (F3: one score for every model): on random rows the torch NLL of every head equals the numpy NLL of ``nll.py``
    (relative error at most 1e-6; the rows that score exactly 0, with no exposure or the category -1, are exactly 0 in both). The outputs
    are float32 or float64; the cells without exposure hold junk."""
    for seed in range(3):
        out, target, exposure = random_block(head, seed=seed, dtype=dtype)
        got = loss.head_nll(head, out, target, exposure)
        want = reference_rows(head, out, target, exposure)
        assert got.dtype == F64 and tuple(got.shape) == want.shape
        assert (want > 0).sum() >= 4, 'the test rows must mostly have a score'
        np.testing.assert_allclose(got.numpy(), want, rtol=1e-6, atol=0)


def test_the_exposure_may_come_with_the_block():
    """The block of M2Rate and M4Conv holds the exposure as its field ``e``; ``head_nll`` takes it from there when none is given, and a target
    that is only the tensor ``y`` works for a head that needs no other field."""
    for head in ('M2Rate', 'M4Conv'):
        out, target, exposure = random_block(head)
        np.testing.assert_array_equal(loss.head_nll(head, out, {**target, 'e': exposure}).numpy(),
                                      loss.head_nll(head, out, target, exposure).numpy())
    out, target, _ = random_block('M3Size')
    np.testing.assert_array_equal(loss.head_nll('M3Size', out, target['y']).numpy(), loss.head_nll('M3Size', out, target).numpy())


def test_the_head_mean_already_holds_the_exposure_and_the_loss_does_not_multiply_it_again():
    """Ruling R-S4a-10 on the hand case of note 04, A: the head gives mu = lambda e, so a cell scores mu - y log mu + log y! and the exposure only
    says whether the cell was observed. Multiplying by the exposure once more would give 5.78112 and not the 4.58888 of note 03, C."""
    y, mu, e = (torch.zeros(1, N_Q, dtype=torch.long), torch.full((1, N_Q), 123.0, dtype=F64), torch.zeros(1, N_Q, dtype=F64))
    y[0, :4], mu[0, :4], e[0, :4] = torch.tensor(TOY_Y), torch.tensor(TOY_MU, dtype=F64), torch.tensor(TOY_E, dtype=F64)
    mu[0, 4:] = float('nan')                                   # the cells beyond the four are not observed: nothing in them is read
    got = loss.head_nll('M2Rate', mu, {'y': y}, e)
    assert got.item() == pytest.approx(TOY_ROW, abs=1e-6) and got.item() == pytest.approx(TOY_ROW, rel=1e-7)
    assert got.item() == pytest.approx(float(nll.poisson(list(TOY_Y), np.array([0.02, 0.02, 0.02, 0.05]), list(TOY_E)).sum()), rel=1e-12)
    twice = loss.head_nll('M2Rate', torch.where(e > 0, mu * e, mu), {'y': y}, e)
    assert twice.item() == pytest.approx(TOY_ROW_TWICE, abs=1e-6) and abs(twice.item() - got.item()) > 1.0


def test_a_gold_bin_of_probability_0_4_scores_minus_ln_0_4():
    p = torch.full((N_GOLD,), 0.6 / (N_GOLD - 1), dtype=F64)
    p[3] = TOY_GOLD_P
    out = torch.log(p).expand(1, N_SIZE, 2, N_LAYER, N_GOLD)
    target = {'s': torch.tensor([2]), 'team': torch.tensor([1]), 'layer': torch.tensor([6]), 'y': torch.tensor([3])}
    assert loss.head_nll('M3Gold', out, target).item() == pytest.approx(TOY_GOLD_NLL, abs=1e-7)


@pytest.fixture(scope='module')
def tiny():
    """A tiny model in evaluation mode (the same weights for every test of the module) and the encoding of six random views."""
    torch.manual_seed(0)
    model = heads.EPT(encoder.EncoderConfig(d=16, layers=2, heads=2, dropout=0.0), toy_static()).eval()
    with torch.no_grad():
        enc = model.encode(encoder_batch(views=6, events=9, edges=7, seed=1))
    return model, enc


def real_conditions(n: int = 6, seed: int = 4) -> dict:
    g = torch.Generator().manual_seed(seed)
    c = {'q': torch.randint(0, N_Q, (n,), generator=g), 'm': torch.randint(0, N_M, (n,), generator=g),
         'z': torch.randint(-1, N_ZONE, (n,), generator=g), 's': torch.randint(0, N_SIZE, (n,), generator=g),
         'K': torch.rand(n, 4, generator=g) * 5, 'O': torch.randint(0, 3, (n, N_CELLS), generator=g).float(),
         'exposure': 1 + torch.rand(n, N_Q, generator=g) * 9, 'window': 1 + torch.rand(n, generator=g) * 59,
         'team': torch.randint(0, 2, (n,), generator=g), 'layer': torch.randint(1, N_LAYER + 1, (n,), generator=g)}
    c['exposure'][0, :5] = 0.0
    c['window'][1] = 0.0
    return c


def real_outputs(model, enc, c: dict) -> dict:
    h = model.heads
    with torch.no_grad():
        m2, m3 = h.m2(enc, c['exposure']), h.m3(enc, c['q'], c['m'], c['z'])
        return {'M2Rate': m2.rate, 'M2Mark': m2.mark_logp, 'M2Zone': m2.zone_logp, 'M3Size': m3.size_logp, 'M3Kills': m3.kill_rate,
                'M3Gold': m3.gold_logp, 'M3Obj': h.m3_obj(enc, c['q'], c['m'], c['z'], c['s'], c['K']),
                'M4Conv': h.m4(enc, c['q'], c['m'], c['z'], c['K'], c['O'], c['window'])}


def test_the_outputs_of_the_real_heads_score_like_nll_on_the_same_rows(tiny):
    """The axes of the output tables are those of ``ept/heads.py``: on the outputs of the real heads (one row for each view) the torch NLL
    equals the numpy NLL of the rows that the table names, for all eight heads."""
    model, enc = tiny
    c = real_conditions()
    outputs = real_outputs(model, enc, c)
    g = torch.Generator().manual_seed(9)
    n = 6
    y = {'M2Rate': torch.poisson(outputs['M2Rate'].double(), generator=g).long(), 'M2Mark': torch.randint(0, N_M, (n,), generator=g),
         'M2Zone': c['z'], 'M3Size': c['s'], 'M3Kills': torch.poisson(outputs['M3Kills'].double()[torch.arange(n), c['s']], generator=g).long(),
         'M3Gold': torch.randint(0, N_GOLD, (n,), generator=g), 'M3Obj': torch.poisson(outputs['M3Obj'].double(), generator=g).long(),
         'M4Conv': torch.poisson(outputs['M4Conv'].double(), generator=g).long()}
    fields = {'M2Mark': ('q',), 'M2Zone': ('q', 'm'), 'M3Kills': ('s',), 'M3Gold': ('s', 'team', 'layer'), 'M3Obj': ('s',)}
    exposure = {'M2Rate': c['exposure'], 'M4Conv': c['window']}
    for head in HEAD_NAMES:
        target = {'y': y[head], **{f: c[f] for f in fields.get(head, ())}}
        got = loss.head_nll(head, outputs[head], target, exposure.get(head))
        want = reference_rows(head, outputs[head], target, exposure.get(head))
        assert (want > 0).any(), head
        np.testing.assert_allclose(got.numpy(), want, rtol=1e-6, atol=0, err_msg=head)


def test_the_loss_of_the_real_heads_under_ruling_r_s4a_10(tiny):
    """The heads' ``rate`` and ``conv_rate`` are the means mu = lambda e. The per-second lambda comes from the same heads with the exposure 1,
    and the loss of mu with the exposure e equals ``nll.poisson(y, lambda, e)`` (the two ways the ruling names): M2Rate and M4Conv."""
    model, enc = tiny
    c = real_conditions()
    h = model.heads
    with torch.no_grad():
        lam2, mu2 = h.m2(enc, 1.0).rate.double().numpy(), h.m2(enc, c['exposure']).rate
        lam4 = h.m4(enc, c['q'], c['m'], c['z'], c['K'], c['O'], 1.0).double().numpy()
        mu4 = h.m4(enc, c['q'], c['m'], c['z'], c['K'], c['O'], c['window'])
    g = torch.Generator().manual_seed(2)
    y2 = torch.poisson(torch.from_numpy(lam2) * c['exposure'].double() + 0.5, generator=g).long()
    y2[c['exposure'] == 0] = 0
    y4 = torch.poisson(torch.from_numpy(lam4) * c['window'].double()[:, None] * 40 + 0.2, generator=g).long()
    y4[c['window'] == 0] = 0
    want2 = nll.poisson(y2.numpy(), lam2, c['exposure'].double().numpy()).sum(axis=1)
    want4 = nll.poisson(y4.numpy(), lam4, c['window'].double().numpy()[:, None]).sum(axis=1)
    np.testing.assert_allclose(loss.head_nll('M2Rate', mu2, {'y': y2}, c['exposure']).numpy(), want2, rtol=1e-6, atol=0)
    np.testing.assert_allclose(loss.head_nll('M4Conv', mu4, {'y': y4}, c['window']).numpy(), want4, rtol=1e-6, atol=0)
    assert want2[0] > 0 and want4[1] == 0.0                             # the first row has a few bins without exposure; the second has no window


@pytest.mark.parametrize('head', HEAD_NAMES)
def test_a_row_reads_exactly_its_own_cells(head):
    """The gradient of the sum of the row scores shows which cells of the output a row reads: only the cells of the table of
    ``ept/heads.py``, and for a category head the gradient there is exactly -1 (d(-log p)/d log p). The row of the category -1 and the cells
    without exposure have the gradient 0 exactly."""
    out, target, exposure = random_block(head, n=7, seed=5)
    out = out.clone().requires_grad_()
    loss.head_nll(head, out, target, exposure).sum().backward()
    want = read_cells(head, out.shape, target, exposure)
    assert want.any() and (head == 'M3Obj' or not want.all())                       # the objective head reads all of its 18 cells
    assert torch.equal(out.grad != 0, want)
    assert torch.isfinite(out.grad).all()
    if head in ('M2Mark', 'M2Zone', 'M3Size', 'M3Gold'):
        assert (out.grad[want] == -1.0).all()


def test_the_poisson_gradient_is_one_minus_y_over_mu():
    """d(mu - y log mu)/d mu = 1 - y / mu: the gradient of the objective head on cells with counts, and zero beyond the observed cells."""
    out, target, _ = random_block('M3Obj', n=6, seed=2)
    out = out.clone().requires_grad_()
    loss.head_nll('M3Obj', out, target).sum().backward()
    assert (target['y'] > 0).any()
    torch.testing.assert_close(out.grad, 1 - target['y'].double() / out.detach(), rtol=1e-12, atol=1e-12)


# =====================================================================================================================================
# rows and cells that add nothing
# =====================================================================================================================================
@pytest.mark.parametrize('head', ['M2Rate', 'M4Conv'])
def test_zero_exposure_cells_add_nothing_and_get_no_gradient(head):
    """Review focus 4: a cell without exposure adds 0 whatever its mean holds (NaN, infinity, a negative number, junk) and gets the gradient
    0 exactly, never NaN; the score of the other cells and their gradient are those of a loss that never saw the unobserved cells."""
    out, target, exposure = random_block(head, n=8, seed=3)
    e = exposure.double()
    seen = (e > 0) if head == 'M2Rate' else (e > 0)[:, None].expand(8, N_CELLS)
    assert (~seen).any() and seen.any()
    clean = out.clone().requires_grad_()
    rows_clean = loss.head_nll(head, clean, target, exposure)
    rows_clean.sum().backward()
    for junk in (float('nan'), float('inf'), -float('inf'), -5.0, 1e300):
        dirty = torch.where(seen, out, torch.full_like(out, junk)).requires_grad_()
        rows = loss.head_nll(head, dirty, target, exposure)
        assert torch.isfinite(rows).all()
        assert torch.equal(rows, rows_clean.detach())                  # not one bit of the junk is in the score
        rows.sum().backward()
        assert torch.equal(dirty.grad[~seen], torch.zeros_like(dirty.grad[~seen]))
        assert torch.equal(dirty.grad[seen], clean.grad[seen])
    assert (rows_clean[~seen.any(dim=1)] == 0).all() and (~seen.any(dim=1)).any()        # a row without any observed cell scores exactly 0


def test_a_count_in_a_cell_without_exposure_is_a_data_error():
    out, target, exposure = random_block('M2Rate', n=5, seed=1)
    target['y'] = target['y'].clone()
    target['y'][exposure == 0] = 1
    with pytest.raises(ValueError, match='no exposure'):
        loss.head_nll('M2Rate', out, target, exposure)
    out, target, exposure = random_block('M4Conv', n=5, seed=1)
    target['y'] = target['y'].clone()
    target['y'][exposure == 0] = 1
    with pytest.raises(ValueError, match='no exposure'):
        loss.head_nll('M4Conv', out, target, exposure)


@pytest.mark.parametrize('head', ['M2Zone', 'M3Gold'])
def test_the_category_minus_one_adds_nothing_and_its_row_is_not_read(head):
    """Note 04, A and B3: a zone without a position and a gold bin of a team without a kill (category -1) score exactly 0 and their law is not
    read: a law of NaN in that row changes nothing, and the gradient of the row is 0."""
    out, target, _ = random_block(head, n=7, seed=4)
    minus = target['y'] == -1
    assert minus.any() and (~minus).any()
    clean = loss.head_nll(head, out, target)
    assert (clean[minus] == 0).all() and (clean[~minus] > 0).all()
    own = {'M2Zone': lambda i: (i, target['q'][i], target['m'][i]), 'M3Gold': lambda i: (i, target['s'][i], target['team'][i], target['layer'][i] - 1)}[head]
    dirty = out.clone()
    for i in torch.nonzero(minus).flatten().tolist():
        dirty[own(i)] = float('nan')
    dirty.requires_grad_()
    rows = loss.head_nll(head, dirty, target)
    assert torch.equal(rows, clean)
    rows.sum().backward()
    assert torch.isfinite(dirty.grad).all()


def test_a_law_that_gives_the_category_probability_zero_has_the_infinite_score():
    """As ``nll.categorical``: a probability 0 for the category that happened is the true score, +inf, and it is returned."""
    out = torch.log_softmax(torch.zeros(1, N_SIZE, dtype=F64), -1)
    out[0, 2] = -float('inf')
    out[0, :2] = out[0, 3] = math.log(1 / 3)
    assert loss.head_nll('M3Size', out, {'y': torch.tensor([2])}).item() == float('inf')
    assert loss.head_nll('M3Size', out, {'y': torch.tensor([0])}).item() == pytest.approx(math.log(3))
    mu = torch.zeros(1, N_CELLS, dtype=F64)                              # a Poisson mean of 0 for a count of 1
    assert loss.head_nll('M3Obj', mu, {'s': torch.tensor([0]), 'y': torch.ones(1, N_CELLS, dtype=torch.long)}).item() == float('inf')


def mutate(field, change):
    return lambda o, t, e: (o, {**t, field: change(t[field])}, e)


REFUSED = [
    ('M2Zone', lambda o, t, e: (o[:, :, :, :9], t, e), 'shape'),
    ('M2Zone', lambda o, t, e: (o[:4], t, e), 'rows'),
    ('M2Zone', lambda o, t, e: (o.reshape(-1), t, e), 'shape'),
    ('M2Zone', mutate('q', lambda x: x + N_Q), r'q must lie'),
    ('M2Zone', mutate('m', lambda x: x - N_M), r'm must lie'),
    ('M2Zone', mutate('y', lambda x: x + N_ZONE), 'category'),
    ('M2Zone', mutate('y', lambda x: x - 2), 'category'),
    ('M2Zone', lambda o, t, e: (o, {k: v for k, v in t.items() if k != 'q'}, e), "needs the field 'q'"),
    ('M2Zone', lambda o, t, e: (o, {k: v for k, v in t.items() if k != 'y'}, e), "needs the field 'y'"),
    ('M2Zone', mutate('y', lambda x: x.double()), 'integer'),
    ('M2Zone', mutate('q', lambda x: x.to(torch.uint8)), 'integer'),
    ('M2Zone', lambda o, t, e: (o + 1.0, t, e), 'add up to 1'),
    ('M2Zone', lambda o, t, e: (torch.where(t['y'][:, None, None, None] >= 0, torch.full_like(o, float('nan')), o), t, e), 'NaN'),
    ('M2Zone', lambda o, t, e: (o.long(), t, e), 'floating'),
    ('M2Zone', lambda o, t, e: (o, t, torch.ones(5)), 'exposure'),
    ('M2Rate', lambda o, t, e: (o, t, None), 'needs the exposure'),
    ('M2Rate', lambda o, t, e: (o, t, -e), 'not negative'),
    ('M2Rate', lambda o, t, e: (o, t, torch.where(e > 3, torch.full_like(e, float('nan')), e)), 'finite'),
    ('M2Rate', lambda o, t, e: (o, t, e[:, :17]), 'exposure'),
    ('M2Rate', mutate('y', lambda x: -x - 1), 'whole numbers'),
    ('M2Rate', mutate('y', lambda x: x.double() + 0.5), 'whole numbers'),
    ('M2Rate', lambda o, t, e: (torch.where(e > 0, -o - 1.0, o), t, e), 'mean'),
    ('M2Rate', lambda o, t, e: (torch.where(e > 0, torch.full_like(o, float('nan')), o), t, e), 'mean'),
    ('M4Conv', lambda o, t, e: (o, t, e[:, None].expand(5, N_CELLS)), 'R-S4a-11'),
    ('M4Conv', lambda o, t, e: (o, t, e[:3]), 'exposure'),
    ('M3Gold', mutate('layer', lambda x: x * 0), r'layer must lie'),
    ('M3Gold', mutate('layer', lambda x: x + N_LAYER), r'layer must lie'),
    ('M3Gold', mutate('team', lambda x: x + 2), r'team must lie'),
    ('M3Gold', mutate('s', lambda x: x + N_SIZE), r's must lie'),
    ('M3Kills', mutate('s', lambda x: x - N_SIZE), r's must lie'),
    ('M3Kills', lambda o, t, e: (o, t, torch.ones(5)), 'exposure'),
    ('M3Obj', lambda o, t, e: (o, {**t, 'y': t['y'][:, :17]}, e), 'shape'),
]


@pytest.mark.parametrize('case', range(len(REFUSED)), ids=[f'{h}-{i}' for i, (h, _, _) in enumerate(REFUSED)])
def test_head_nll_refuses_inputs_that_are_not_those_of_the_head(case):
    """Every refusal is a ValueError that names the argument (a TypeError for a category that is not an integer, as ``nll.categorical``)."""
    head, change, match = REFUSED[case]
    out, target, exposure = random_block(head, n=5, seed=11)
    out, target, exposure = change(out, dict(target), exposure)
    with pytest.raises((ValueError, TypeError), match=match):
        loss.head_nll(head, out, target, exposure)


@pytest.mark.parametrize('head, classes', [('M2Mark', N_M), ('M2Zone', N_ZONE), ('M3Size', N_SIZE), ('M3Gold', N_GOLD)])
def test_the_last_class_is_scored_and_the_first_number_beyond_it_is_refused(head, classes):
    """The categories run from -1 (none) to C - 1: the last class is a class, the number C is a data error (as in ``nll.categorical``), and so is -2."""
    out, target, _ = random_block(head, n=5, seed=12)
    assert (loss.head_nll(head, out, {**target, 'y': torch.full_like(target['y'], classes - 1)}) > 0).all()
    for bad in (classes, -2):
        with pytest.raises(ValueError, match='category'):
            loss.head_nll(head, out, {**target, 'y': torch.full_like(target['y'], bad)})


def test_a_seen_cell_with_the_mean_0_and_the_count_0_scores_0_with_a_finite_gradient():
    """0 log 0 = 0 (as in ``nll.poisson``): an observed cell whose mean is exactly 0 and whose count is 0 scores 0, not NaN, and its gradient is
    d(mu - y log mu)/d mu = 1."""
    mu = torch.zeros(2, N_CELLS, dtype=F64, requires_grad=True)
    rows = loss.head_nll('M3Obj', mu, {'s': torch.zeros(2, dtype=torch.long), 'y': torch.zeros(2, N_CELLS, dtype=torch.long)})
    assert torch.equal(rows, torch.zeros(2, dtype=F64))
    rows.sum().backward()
    assert torch.equal(mu.grad, torch.ones(2, N_CELLS, dtype=F64))


def test_an_unknown_head_is_refused():
    out, target, _ = random_block('M3Size')
    with pytest.raises(ValueError, match='head'):
        loss.head_nll('M5Foo', out, target)
    with pytest.raises(ValueError, match='head'):
        loss.scored_rows('M5Foo', target)


def test_scored_rows_are_the_rows_with_an_observed_cell_or_a_category():
    """The rows of the mean (note 04, B3): a frame row with an observed bin, a conversion window above 0, a category from 0 on; the kills
    and the objectives are always observed."""
    out, target, exposure = random_block('M2Rate')
    assert loss.scored_rows('M2Rate', target, exposure).tolist() == [False] + [True] * 8
    out, target, exposure = random_block('M4Conv')
    assert loss.scored_rows('M4Conv', {**target, 'e': exposure}).tolist() == [True, False] + [True] * 7
    out, target, _ = random_block('M2Zone')
    assert loss.scored_rows('M2Zone', target).tolist() == (target['y'] >= 0).tolist() and not loss.scored_rows('M2Zone', target)[0]
    for head in ('M3Kills', 'M3Obj'):
        assert loss.scored_rows(head, random_block(head)[1]).all()
        meta = torch.empty(5, 2 if head == 'M3Kills' else N_CELLS, dtype=torch.long, device='meta')           # a tensor that is on no real device: the mask follows it
        assert loss.scored_rows(head, {'y': meta}).device.type == 'meta'


# =====================================================================================================================================
# the mean of a head and the sum of the heads
# =====================================================================================================================================
def block_target(key: str, n: int, sizes=None, y=None) -> dict:
    """The fields of a block that ``total_loss`` reads, for ``n`` rows, made of ones and zeros: the sizes ``s``, the category ``y`` of a
    category head (0 unless given) and the exposure ``e`` of M2Rate and M4Conv (1)."""
    head = key.split('/')[0]
    t = {}
    if head in loss.SIZE_NORMALISED:
        t['s'] = torch.tensor(sizes if sizes is not None else [0] * n)
    if head in ('M2Mark', 'M2Zone', 'M3Size', 'M3Gold'):
        t['y'] = torch.tensor(y if y is not None else [0] * n)
    if head == 'M2Rate':
        t['e'] = torch.ones(n, N_Q, dtype=F64)
    if head == 'M4Conv':
        t['e'] = torch.ones(n, dtype=F64)
    return t


def test_the_blocks_are_the_thirteen_heads_of_the_plan():
    keys = {f'{h}/{k}' for h, k in head_rows.VALID_HEADS}
    assert set(loss.BLOCKS) == keys and len(loss.BLOCKS) == 13 == len(set(loss.BLOCKS))
    assert loss.BLOCKS[:8] == tuple(f'{h}/masked' for h in HEAD_NAMES) and loss.BLOCKS[8:] == tuple(f'{h}/open' for h in head_rows.OPEN_HEAD_NAMES)
    assert loss.split_key('M3Gold/open') == ('M3Gold', 'open') and loss.block_key('M3Gold', 'open') == 'M3Gold/open'
    assert loss.SIZE_NORMALISED == {'M3Kills', 'M3Gold', 'M3Obj'}


@pytest.mark.parametrize('key', ['M3Kills/masked', 'M3Kills/open', 'M3Gold/masked', 'M3Gold/open', 'M3Obj/masked', 'M3Obj/open'])
def test_size_normalised_m3_matches_the_hand_case(key):
    """Note 04, B1: the three M3 experts are averaged over the sizes: the rows (2, 4, 1, 1, 4, 10) of the sizes (T, T, S, S, S, P) have the
    means 3, 2 and 10 and the head mean 5.0 (not the row mean 3.667); the B0 scale 2.5 makes the term 2.0. A size that does not occur (ASYM)
    is left out of the average; with an ASYM row of score 6 the mean over four sizes is 5.25 (row mean 4.0)."""
    rows = torch.tensor(TOY_B1_SCORES, dtype=F64)
    target = {key: block_target(key, 6, list(TOY_B1_SIZES))}
    assert loss.head_losses({key: rows}, target)[key].item() == pytest.approx(TOY_B1_MEAN, abs=1e-12)
    assert loss.total_loss({key: rows}, {key: TOY_B1_SCALE}, target).item() == pytest.approx(TOY_B1_TERM, abs=1e-12)
    assert rows.mean().item() == pytest.approx(TOY_B1_PLAIN)                             # the row mean is another number
    more = torch.cat([rows, torch.tensor([6.0], dtype=F64)])
    target = {key: block_target(key, 7, [*TOY_B1_SIZES, 2])}
    assert loss.head_losses({key: more}, target)[key].item() == pytest.approx(TOY_B1_ASYM_MEAN, abs=1e-12)
    assert more.mean().item() == pytest.approx(TOY_B1_ASYM_PLAIN)
    try_it = torch.cat([rows, torch.tensor([6.0], dtype=F64)])                           # "try it": one more T row of score 6
    target = {key: block_target(key, 7, [*TOY_B1_SIZES, 0])}
    assert loss.head_losses({key: try_it}, target)[key].item() == pytest.approx(16 / 3, abs=1e-12)


def test_the_size_normalised_mean_leaves_out_the_rows_without_a_category():
    """The rows that do not count (a gold entry of the category -1 scores 0) are not rows of their size: the sizes (T, T, S, S, S, P, P) with the
    rows 3 and 6 uncounted have the means 3, (1 + 4) / 2 = 2.5 and 10, so (3 + 2.5 + 10) / 3; a size that has only uncounted rows (ASYM here) does not
    occur."""
    key = 'M3Gold/masked'
    rows = torch.tensor([2.0, 4.0, 1.0, 0.0, 4.0, 10.0, 0.0, 0.0], dtype=F64)
    target = {key: block_target(key, 8, [0, 0, 1, 1, 1, 3, 3, 2], y=[0, 0, 0, -1, 0, 0, -1, -1])}
    assert loss.head_losses({key: rows}, target)[key].item() == pytest.approx((3.0 + 2.5 + 10.0) / 3, abs=1e-12)


def test_the_size_gate_and_the_other_heads_are_plain_means():
    """Note 04, B2: the size gate P(s | x) is a marginal law; weighting its rows by size would move its optimum from the frequency of the
    sizes to the uniform law. Its loss is the row mean, as the loss of the heads that carry no size (M2Mark, M4Conv). With the law
    (0.75, 0.25) the mean is 0.5623352 (the size-normalised number 0.8369883 is not used) and the row mean prefers the frequencies."""
    sizes = torch.tensor(TOY_B2_SIZES)
    best = {}
    for p, (plain, normalised) in TOY_B2.items():
        rows = -torch.log(torch.tensor(p, dtype=F64))[sizes]
        got = loss.head_losses({'M3Size/masked': rows}, {'M3Size/masked': {'y': sizes}})['M3Size/masked'].item()
        assert got == pytest.approx(plain, abs=1e-7)
        best[p] = got
        by_size = torch.stack([rows[sizes == k].mean() for k in (0, 1)]).mean().item()
        assert by_size == pytest.approx(normalised, abs=1e-7)                            # what a size-normalised gate would minimise
    assert best[(0.75, 0.25)] < best[(0.5, 0.5)]
    mean = loss.head_losses({'M2Mark/masked': torch.tensor([1.0, 2.0, 6.0], dtype=F64), 'M4Conv/masked': torch.tensor([1.0, 3.0], dtype=F64)},
                            {'M2Mark/masked': {'y': torch.tensor([0, 1, 2])}, 'M4Conv/masked': {'e': torch.ones(2, dtype=F64)}})
    assert mean['M2Mark/masked'].item() == pytest.approx(3.0) and mean['M4Conv/masked'].item() == pytest.approx(2.0)


def test_rows_without_a_category_or_exposure_are_not_rows_of_the_mean():
    """Note 04, B3: the zone -1 rows add 0 but do not count: the mean of (y = 3, -1, 0, -1, 7) is over the three rows with a zone, 2 ln 2 = 1.38629
    (not 0.83178). The same for a frame row without exposure (M2Rate) and a row without a conversion window (M4Conv)."""
    rows = torch.zeros(5, dtype=F64)
    rows[[0, 2, 4]] = -torch.log(torch.tensor(TOY_B3_P, dtype=F64))
    key = 'M2Zone/masked'
    got = loss.head_losses({key: rows}, {key: {'y': torch.tensor(TOY_B3_Y)}})[key].item()
    assert got == pytest.approx(TOY_B3_MEAN, abs=1e-12) and rows.sum().item() / 5 == pytest.approx(TOY_B3_WRONG)
    two = -torch.log(torch.tensor([0.5, 0.5, 0.25], dtype=F64))
    assert (two.mean().item()) == pytest.approx(0.92420, abs=5e-6)                       # "try it" of note 04, B3
    e = torch.ones(3, N_Q, dtype=F64)
    e[1] = 0.0
    got = loss.head_losses({'M2Rate/masked': torch.tensor([2.0, 0.0, 6.0], dtype=F64)}, {'M2Rate/masked': {'e': e}})['M2Rate/masked']
    assert got.item() == pytest.approx(4.0)
    got = loss.head_losses({'M4Conv/open': torch.tensor([2.0, 0.0, 6.0], dtype=F64)}, {'M4Conv/open': {'e': torch.tensor([5.0, 0.0, 9.0], dtype=F64)}})
    assert got['M4Conv/open'].item() == pytest.approx(4.0)


def test_total_loss_is_the_sum_of_the_head_means_over_their_scales():
    """Note 04, C: the means 6.0 and 0.9 over the scales 8.0 and 1.2 give 0.75 + 0.75 = 1.5; both 4 % better give 1.44 (each head lowers the
    loss by 0.03: a relative gain weighs the same, whatever the unit); a model that equals B0 on all thirteen heads has the loss 13."""
    keys = ('M2Rate/masked', 'M3Size/masked')
    scales = dict(zip(keys, TOY_C_SCALES))

    def run(factor):
        nlls = {k: torch.full((3,), m * factor, dtype=F64) for k, m in zip(keys, TOY_C_MEANS)}
        return loss.total_loss(nlls, scales, {'M2Rate/masked': block_target('M2Rate/masked', 3), 'M3Size/masked': block_target('M3Size/masked', 3)})

    assert run(1.0).item() == pytest.approx(TOY_C_LOSS, abs=1e-12) and run(0.96).item() == pytest.approx(TOY_C_AFTER, abs=1e-12)
    nlls = {k: torch.full((3,), m * (0.96 if k == 'M3Size/masked' else 1.0), dtype=F64) for k, m in zip(keys, TOY_C_MEANS)}
    only_head_2 = loss.total_loss(nlls, scales, {'M2Rate/masked': block_target('M2Rate/masked', 3), 'M3Size/masked': block_target('M3Size/masked', 3)})
    assert only_head_2.item() == pytest.approx(1.47, abs=1e-12)                           # "try it" of note 04, C
    scale_all = {k: 1.0 + i for i, k in enumerate(loss.BLOCKS)}
    nlls = {k: torch.full((4,), scale_all[k], dtype=F64) for k in loss.BLOCKS}
    targets = {k: block_target(k, 4) for k in loss.BLOCKS}
    assert loss.total_loss(nlls, scale_all, targets).item() == pytest.approx(13.0, abs=1e-12)


def test_the_gradient_of_a_row_is_one_over_the_rows_and_the_scale():
    """Note 04, C: the weight of one row of a plain head is 1 / (N B0): 4 rows and the scale 8 give 0.03125; for a size-normalised head it is
    1 / (n_size N_s B0)."""
    rows = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=F64, requires_grad=True)
    loss.total_loss({'M2Mark/masked': rows}, {'M2Mark/masked': 8.0}, {'M2Mark/masked': {'y': torch.zeros(4, dtype=torch.long)}}).backward()
    assert torch.equal(rows.grad, torch.full((4,), 0.03125, dtype=F64))
    rows = torch.tensor(TOY_B1_SCORES, dtype=F64, requires_grad=True)
    loss.total_loss({'M3Kills/open': rows}, {'M3Kills/open': 2.0}, {'M3Kills/open': block_target('M3Kills/open', 6, list(TOY_B1_SIZES))}).backward()
    n_per_size = {0: 2, 1: 3, 3: 1}
    want = [1 / (3 * n_per_size[s] * 2.0) for s in TOY_B1_SIZES]
    torch.testing.assert_close(rows.grad, torch.tensor(want, dtype=F64), rtol=1e-14, atol=0)
    single = torch.tensor(TOY_B1_SCORES, dtype=torch.float32, requires_grad=True)                 # scores of a float32 run: the gradient reaches them in float32
    loss.total_loss({'M3Kills/open': single}, {'M3Kills/open': 2.0}, {'M3Kills/open': block_target('M3Kills/open', 6, list(TOY_B1_SIZES))}).backward()
    assert single.grad.dtype == torch.float32
    torch.testing.assert_close(single.grad.double(), torch.tensor(want, dtype=F64), rtol=1e-6, atol=0)


def test_total_loss_refuses_what_it_cannot_scale_or_average():
    row, key = torch.ones(3, dtype=F64), 'M2Mark/masked'
    target = {key: {'y': torch.zeros(3, dtype=torch.long)}}
    with pytest.raises(ValueError, match='scale'):
        loss.total_loss({key: row}, {}, target)
    for bad in (0.0, -1.0, float('nan'), float('inf'), 'two', None):
        with pytest.raises(ValueError, match='scale'):
            loss.total_loss({key: row}, {key: bad}, target)
    assert loss.total_loss({key: row}, {key: np.float32(2.0)}, target).item() == 0.5            # a numpy number is a scale too
    assert loss.total_loss({key: row}, {key: torch.tensor(2.0)}, target).item() == 0.5          # and a 0-d tensor
    with pytest.raises(ValueError, match='block'):
        loss.total_loss({'M2Mark/open': row}, {'M2Mark/open': 1.0}, {'M2Mark/open': target[key]})        # M2 has no open head
    with pytest.raises(ValueError, match='block'):
        loss.total_loss({'M2Mark': row}, {'M2Mark': 1.0}, target)
    with pytest.raises(ValueError, match='target'):
        loss.total_loss({key: row}, {key: 1.0}, {})
    with pytest.raises(ValueError, match="needs the field 's'"):
        loss.head_losses({'M3Obj/open': row}, {'M3Obj/open': {}})
    with pytest.raises(ValueError, match='shape'):
        loss.head_losses({key: row}, {key: {'y': torch.zeros(4, dtype=torch.long)}})
    with pytest.raises(ValueError, match='size'):
        loss.head_losses({'M3Obj/open': row}, {'M3Obj/open': {'s': torch.tensor([0, 1, 4])}})
    with pytest.raises(ValueError, match='row'):
        loss.total_loss({}, {}, {})
    with pytest.raises(ValueError, match='row'):                                                         # no counted row in any block
        loss.total_loss({key: row}, {key: 1.0}, {key: {'y': torch.full((3,), -1)}})


def test_a_block_without_a_counted_row_adds_nothing_while_another_block_counts():
    """A batch may hold no row of a head, or only rows without a category: that block adds 0 (no NaN from a mean of nothing)."""
    nlls = {'M2Zone/masked': torch.zeros(2, dtype=F64, requires_grad=True), 'M3Size/masked': torch.tensor([1.0, 3.0], dtype=F64, requires_grad=True),
            'M4Conv/open': torch.zeros(0, dtype=F64)}
    targets = {'M2Zone/masked': {'y': torch.tensor([-1, -1])}, 'M3Size/masked': {'y': torch.tensor([0, 1])}, 'M4Conv/open': {'e': torch.zeros(0, dtype=F64)}}
    scales = {k: 2.0 for k in nlls}
    total = loss.total_loss(nlls, scales, targets)
    assert total.item() == pytest.approx(1.0)
    total.backward()
    grad = nlls['M2Zone/masked'].grad
    assert grad is None or torch.equal(grad, torch.zeros(2, dtype=F64))                   # nothing flows to a block that adds nothing
    assert set(loss.head_losses(nlls, targets)) == {'M3Size/masked'}


def test_the_order_of_the_blocks_changes_no_bit():
    """The terms are added in the order of ``BLOCKS``, whatever the order of the mappings."""
    g = torch.Generator().manual_seed(6)
    nlls = {k: 1.0 + torch.rand(5, generator=g, dtype=F64) * 9 for k in loss.BLOCKS}
    targets = {k: block_target(k, 5, [0, 1, 2, 3, 0]) for k in loss.BLOCKS}
    scales = {k: 10.0 ** (6 * torch.rand(1, generator=g, dtype=F64).item() - 3) for k in loss.BLOCKS}         # terms from 1e-3 to 1e4: the order matters
    a = loss.total_loss(nlls, scales, targets)
    b = loss.total_loss(dict(reversed(nlls.items())), dict(reversed(scales.items())), dict(reversed(targets.items())))
    assert torch.equal(a, b)
    means = loss.head_losses(nlls, targets)

    def left_to_right(keys):
        total = None
        for key in keys:
            term = means[key] / scales[key]
            total = term if total is None else total + term
        return total

    assert not torch.equal(left_to_right(loss.BLOCKS), left_to_right(reversed(loss.BLOCKS)))   # on these numbers the order changes the last bits: the test can fail
    assert torch.equal(a, left_to_right(loss.BLOCKS))                                      # and the order is the one that the docstring names: left to right over BLOCKS


def test_the_head_means_follow_the_reference_rows():
    """The whole chain on random blocks: the torch rows, the means (size-normalised for the three experts) and the sum over the scales equal
    a numpy computation from the reference rows."""
    nlls, targets, want = {}, {}, 0.0
    scales = {}
    for i, head in enumerate(HEAD_NAMES):
        key = f'{head}/masked'
        out, target, exposure = random_block(head, n=12, seed=7)
        rows = loss.head_nll(head, out, target, exposure)
        ref = reference_rows(head, out, target, exposure)
        nlls[key], targets[key] = rows, {**target, **({} if exposure is None else {'e': exposure})}
        if head in ('M3Kills', 'M3Obj'):
            counted = np.ones(12, dtype=bool)                                       # the kills and the objectives are always observed
        elif head == 'M2Rate':
            counted = (exposure.numpy() > 0).any(axis=-1)                           # a frame row with an observed bin
        elif head == 'M4Conv':
            counted = exposure.numpy() > 0                                          # a conversion window
        else:
            counted = (target['y'] >= 0).numpy()                                    # a category
        if head in loss.SIZE_NORMALISED:
            s = target['s'].numpy()
            mean = np.mean([ref[counted & (s == k)].mean() for k in range(N_SIZE) if (counted & (s == k)).any()])
        else:
            mean = ref[counted].mean()
        scales[key] = 1.5 + i
        want += mean / scales[key]
    got = loss.total_loss(nlls, scales, targets)
    assert got.dtype == F64 and got.item() == pytest.approx(want, rel=1e-12)


@pytest.fixture(scope='module')
def small_world(tmp_path_factory):
    """The EPT toy world with 7 event tokens and 5 edges per view (the real store has 128 and 128): S1, S2 and label stores with the real links."""
    folder = tmp_path_factory.mktemp('loss_world')
    world = build_ept_world(folder, n_events=7, n_edges=5)
    yield world
    del world
    gc.collect()                                                                              # Windows does not delete a folder with open memory maps
    shutil.rmtree(folder, ignore_errors=True)


def block_outputs(fa, enc, batch) -> dict:
    """The output of the head of each of the 13 blocks of a ``Batches`` batch, for the rows of the block (``pos``) and the conditions that the block holds."""
    outs = {}
    for head, kind in data.ROW_FIELDS:
        b = data.block(batch, head, kind)
        q, m, z = b.get('q'), b.get('m'), b.get('z')                                          # the open blocks have none: the open reading
        if head in ('M2Rate', 'M2Mark', 'M2Zone'):
            m2 = fa.m2(enc, b['e'] if head == 'M2Rate' else 1.0, pos=b['pos'])
            out = {'M2Rate': m2.rate, 'M2Mark': m2.mark_logp, 'M2Zone': m2.zone_logp}[head]
        elif head in ('M3Size', 'M3Kills', 'M3Gold'):
            m3 = fa.m3(enc, q, m, z, pos=b['pos'])
            out = {'M3Size': m3.size_logp, 'M3Kills': m3.kill_rate, 'M3Gold': m3.gold_logp}[head]
        elif head == 'M3Obj':
            out = fa.m3_obj(enc, q, m, z, b['s'], b['K'], pos=b['pos'])
        else:
            out = fa.m4(enc, q, m, z, b['K'], b['O'], b['e'], pos=b['pos'])
        outs[loss.block_key(head, kind)] = out
    return outs


def test_a_batch_of_the_data_layer_runs_through_the_frame_averaged_heads_and_the_loss(tiny, small_world):
    """The pieces fit together: a batch of ``data.Batches`` (Task 1) on the toy world, the blocks' own fields as the conditions of the frame-averaged
    heads (Task 3), ``head_nll`` on the block, the thirteen scores equal to the numpy scores of the rows, and a ``total_loss`` with a finite
    backward pass. Every block of the plan has rows in the batch, and the rows without a category or exposure are in it too."""
    model, _ = tiny
    model.eval()
    world = small_world
    stats = data.fit_stats(world.s1, world.s2)
    rows = data.EngagementRows(data.ept_head_rows(world.s1, world.s2, world.labels, 'FIT'))
    batch = next(iter(data.Batches(rows, 100, 0, stats=stats).epoch(0)))
    fa = heads.FrameAveraged(model)
    model.zero_grad(set_to_none=True)
    enc = fa.encode(batch)
    outs = block_outputs(fa, enc, batch)
    nlls, targets = {}, {}
    for key, out in outs.items():
        head, kind = loss.split_key(key)
        targets[key] = data.block(batch, head, kind)
        assert len(targets[key]['y']) > 0, key
        nlls[key] = loss.head_nll(head, out, targets[key])
        want = reference_rows(head, out, targets[key], targets[key].get('e'))
        np.testing.assert_allclose(nlls[key].detach().numpy(), want, rtol=1e-6, atol=0, err_msg=key)
    assert set(nlls) == set(loss.BLOCKS)
    assert (targets['M2Zone/masked']['y'] == -1).any() and (~loss.scored_rows('M2Zone', targets['M2Zone/masked'])).any()
    total = loss.total_loss(nlls, {key: 1.0 + i for i, key in enumerate(loss.BLOCKS)}, targets)
    total.backward()
    assert torch.isfinite(total) and total.item() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    model.zero_grad(set_to_none=True)


def test_backward_through_the_real_heads_is_finite_and_bit_repeatable(tiny):
    """The path of a training step on a tiny model: the encoder, the eight heads, the scored rows and the sum over the scales, then a backward
    pass, with the deterministic algorithms of PyTorch switched on (a loss op without a deterministic CPU kernel would raise). The gradients
    are finite and a second pass gives the same bits."""
    model, _ = tiny
    c = real_conditions()
    batch = encoder_batch(views=6, events=9, edges=7, seed=1)

    def step():
        model.zero_grad()
        enc = model.encode(batch)
        h = model.heads
        m2, m3 = h.m2(enc, c['exposure']), h.m3(enc, c['q'], c['m'], c['z'])
        out = {'M2Rate': m2.rate, 'M2Mark': m2.mark_logp, 'M2Zone': m2.zone_logp, 'M3Size': m3.size_logp, 'M3Kills': m3.kill_rate,
               'M3Gold': m3.gold_logp, 'M3Obj': h.m3_obj(enc, c['q'], c['m'], c['z'], c['s'], c['K']),
               'M4Conv': h.m4(enc, c['q'], c['m'], c['z'], c['K'], c['O'], c['window'])}
        y = {'M2Rate': torch.ones(6, N_Q, dtype=torch.long) * (c['exposure'] > 0), 'M2Mark': c['m'], 'M2Zone': c['z'], 'M3Size': c['s'],
             'M3Kills': torch.ones(6, 2, dtype=torch.long), 'M3Gold': torch.full((6,), 7), 'M3Obj': torch.ones(6, N_CELLS, dtype=torch.long),
             'M4Conv': torch.ones(6, N_CELLS, dtype=torch.long) * (c['window'] > 0)[:, None]}
        extra = {'M2Mark': ('q',), 'M2Zone': ('q', 'm'), 'M3Kills': ('s',), 'M3Gold': ('s', 'team', 'layer'), 'M3Obj': ('s',)}
        exposure = {'M2Rate': c['exposure'], 'M4Conv': c['window']}
        nlls, targets = {}, {}
        for head in HEAD_NAMES:
            key = f'{head}/masked'
            targets[key] = {'y': y[head], **{f: c[f] for f in extra.get(head, ())}, **({'e': exposure[head]} if head in exposure else {})}
            nlls[key] = loss.head_nll(head, out[head], targets[key])
        total = loss.total_loss(nlls, {k: 2.0 for k in nlls}, targets)
        total.backward()
        return total.detach().clone(), [p.grad.clone() for p in model.parameters() if p.grad is not None]

    before = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        first, grads = step()
        again, grads2 = step()
    finally:
        torch.use_deterministic_algorithms(before)
        model.zero_grad(set_to_none=True)
    assert torch.isfinite(first) and first.item() > 0 and grads
    assert all(torch.isfinite(t).all() for t in grads)
    assert torch.equal(first, again) and all(torch.equal(a, b) for a, b in zip(grads, grads2))


# =====================================================================================================================================
# the B0 scales
# =====================================================================================================================================
def head_scores(values, row=None):
    values = np.asarray(values, dtype=np.float64)
    return head_rows.HeadScores(nll=values, match=np.zeros(len(values), dtype=np.int64),
                                row=np.arange(len(values)) if row is None else np.asarray(row, dtype=np.int64))


def test_the_scale_of_a_head_is_the_mean_over_the_rows_of_the_head():
    """Note 04, C and plan Q8: the scale of a head is the mean of the B0 NLL over the rows that the EPT trains on: the frame rows for M2Rate
    (the cells of one frame row added), the design rows for the others, and for the three M3 experts the mean over the sizes of the mean
    over the rows of a size."""
    from c_ept.ept import scales
    cells = head_scores([1.0, 2.0, 3.0, 4.0, 5.0], row=[10, 10, 11, 12, 12])                 # frame rows 10, 11, 12: 3, 3, 9
    value, n, by_size = scales.head_scale('M2Rate', cells)
    assert (value, n, by_size) == (5.0, 3, None)
    value, n, by_size = scales.head_scale('M4Conv', head_scores([1.0, 2.0, 6.0]))
    assert (value, n, by_size) == (3.0, 3, None)
    value, n, by_size = scales.head_scale('M3Gold', head_scores(TOY_B1_SCORES), sizes=np.array(TOY_B1_SIZES))
    assert value == pytest.approx(TOY_B1_MEAN) and n == 6 and by_size == pytest.approx({'T': 3.0, 'S': 2.0, 'P': 10.0})
    value, _, by_size = scales.head_scale('M3Size', head_scores(TOY_B1_SCORES), sizes=np.array(TOY_B1_SIZES))
    assert value == pytest.approx(TOY_B1_PLAIN) and by_size is None                           # the gate: the plain mean
    with pytest.raises(ValueError, match='sizes'):
        scales.head_scale('M3Kills', head_scores(TOY_B1_SCORES))
    with pytest.raises(ValueError, match='sizes'):
        scales.head_scale('M3Obj', head_scores(TOY_B1_SCORES), sizes=np.array([0, 1]))
    with pytest.raises(ValueError, match='scale'):
        scales.head_scale('M2Mark', head_scores([0.0, 0.0]))                                  # a head whose B0 NLL is 0 cannot scale a loss
    with pytest.raises(ValueError, match='row'):
        scales.head_scale('M2Mark', head_scores([]))


def write_toy_run(runs, s1, s2, labels, b0, names):
    """A toy S3 run folder under ``runs``: the S1, S2 and label stores under the names of RUN.json, the fitted B0 heads ``b0`` saved in ``fit/b0``, RUN.json and
    STAGES.json with the hashes of the model JSON and of the label store. The stores are copied, never changed."""
    shutil.copytree(s1.root, runs / names.s1 / 's1')
    shutil.copytree(s2.root, runs / names.s2 / 's2')
    run = runs / names.s3
    (run / 'fit').mkdir(parents=True)
    shutil.copytree(labels.root, run / 'labels' / 'store')
    b0.save(run / 'fit' / 'b0')
    manifest = {'stage': 's3', 'run': names.s3,
                's1': {'run': names.s1, 'files_sha256': runlog.sha256_json(s1.manifest['files'])},
                's2': {'run': names.s2, 'files_sha256': runlog.sha256_json(s2.manifest['files'])}}
    (run / 'RUN.json').write_text(json.dumps(manifest), encoding='utf-8')
    from c_ept.workflows import run_s3
    ledger = run_s3.Ledger(run)
    ledger.doc['stages']['fit'] = {'complete': True, 'outputs': {'fit/b0/b0_heads.json': store_core.sha(run / 'fit' / 'b0' / 'b0_heads.json')}}
    ledger.doc['stages']['labels'] = {'complete': True, 'outputs': {'labels/store/': run_s3.identity(run, 'labels/store/')}}
    ledger.save()
    return run


@pytest.fixture(scope='module')
def toy_run(heads_world, tmp_path_factory):                                                  # noqa: F811
    """A toy S3 run folder in a temporary runs root: the S1, S2 and label stores of the heads world under the names of RUN.json, the B0
    heads fitted on its FIT matches (``fit/b0``), RUN.json and STAGES.json with the hashes of the model JSON and of the label store."""
    from sklearn.exceptions import ConvergenceWarning

    from c_ept.models import baselines_b0
    runs = tmp_path_factory.mktemp('b0_scales_runs') / 'runs'
    names = SimpleNamespace(s1='20261020_c_ept_s1_1', s2='20261021_c_ept_s2_3', s3='20261022_c_ept_s3_3')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)                                  # the toy heads do not converge: the known warnings of the B0 tests
        b0 = baselines_b0.B0Heads.fit(heads_world.s1, heads_world.s2, heads_world.labels, heads_world.fit_means)
    run = write_toy_run(runs, heads_world.s1, heads_world.s2, heads_world.labels, b0, names)
    yield SimpleNamespace(runs=runs, run=run, names=names, b0=b0, world=heads_world)
    shutil.rmtree(runs.parent, ignore_errors=True)


def sizes_of(world, kind, rows):
    """The size class of the evaluation rows of a head: an onset (masked) or a unit (open), found in the S1 arrays by plain indexing."""
    size = np.asarray(world.s1.array('units/size'))
    unit = np.asarray(world.s1.array('onsets/unit'))[rows] if kind == 'masked' else rows
    return size[unit].astype(np.int64)


def expected_scale(world, b0, head, kind):
    """``(scale, rows, means by size)`` of a head recomputed in the test: the mean (size-normalised for the three experts) of the B0 NLL over the
    FIT rows."""
    scores = b0.nll(world.s1, world.s2, world.labels, 'FIT', kind)[head]
    values = scores.per_row()[0] if head == 'M2Rate' else scores.nll
    if head in ('M3Kills', 'M3Gold', 'M3Obj'):
        size = sizes_of(world, kind, scores.row)
        by_size = {name: float(values[size == k].mean()) for k, name in enumerate(('T', 'S', 'ASYM', 'P')) if (size == k).any()}
        return float(np.mean(list(by_size.values()))), len(values), by_size
    return float(values.mean()), len(values), None


@pytest.mark.parametrize('kind', ['masked', 'open'])
@pytest.mark.parametrize('head', ['M3Kills', 'M3Gold', 'M3Obj'])
def test_the_size_that_the_scale_finds_for_a_scored_row_is_the_size_of_the_row_in_head_rows(toy_run, head, kind):
    """``scales._size_of_rows`` finds the size of a B0 score from its evaluation row (an onset for a masked head, a unit for an open head). It must
    equal the size class that the rows themselves carry (``HeadRows.parts(...).codes['s']``, the rows of the EPT too), design row by design row."""
    from c_ept.ept import scales
    w = toy_run.world
    scores = toy_run.b0.nll(w.s1, w.s2, w.labels, 'FIT', kind)[head]
    parts = data.ept_head_rows(w.s1, w.s2, w.labels, 'FIT').parts(head, kind)
    assert len(scores.nll) == len(parts.codes['s']) > 0
    assert np.array_equal(scales._size_of_rows(w.s1, kind, np.asarray(scores.row)), np.asarray(parts.codes['s']))


def test_b0_scales_are_computed_on_fit_rows_and_hashed(toy_run, tmp_path, monkeypatch):
    """Plan Q8 and Task 4: ``B0_SCALES.json`` holds the thirteen scales, the S3 run name and the SHA-256 of the model files (the JSON and the
    arrays of ``fit/b0``); the scale of a head is the mean of the B0 NLL over the FIT rows of that head, computed here again from
    ``B0Heads.nll``; B0 is asked for the role FIT only, masked and open (STOP and DEV rows are never scored: the first column of the data
    rules). No key is written."""
    from c_ept.models import baselines_b0
    from c_ept.ept import scales
    asked = []
    real = baselines_b0.B0Heads.nll

    def spy(self, s1, s2, labels, role, kind='masked', **kw):
        asked.append((role, kind))
        return real(self, s1, s2, labels, role, kind, **kw)

    monkeypatch.setattr(baselines_b0.B0Heads, 'nll', spy)
    got = scales.b0_scales(toy_run.run, tmp_path, runs_root=toy_run.runs)
    assert sorted(asked) == [('FIT', 'masked'), ('FIT', 'open')]
    assert list(got) == list(loss.BLOCKS) and all(isinstance(v, float) and math.isfinite(v) and v > 0 for v in got.values())
    monkeypatch.undo()
    expected = {key: expected_scale(toy_run.world, toy_run.b0, *loss.split_key(key)) for key in loss.BLOCKS}
    for key, (value, _, _) in expected.items():
        assert got[key] == pytest.approx(value, rel=1e-12), key
    path = tmp_path / 'B0_SCALES.json'
    text = path.read_text(encoding='utf-8')
    doc = json.loads(text)
    folder = toy_run.run / 'fit' / 'b0'
    files = {f'fit/b0/{name}': hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in ('b0_heads.json', 'b0_heads.npz')}
    assert doc['s3_run'] == toy_run.names.s3 and doc['role'] == 'FIT' and doc['model_files_sha256'] == files
    assert doc['scales'] == got and doc['scales_sha256'] == runlog.sha256_json(got) and 'FIT rows' in doc['rule']
    assert set(doc['n_rows']) == set(loss.BLOCKS) == set(doc['n_unscored']) and doc['n_rows']['M2Mark/masked'] > 0
    for key, (_, n_rows, by_size) in expected.items():                                       # the rows of the mean and the left-out rows are those of B0
        head, kind = loss.split_key(key)
        assert doc['n_rows'][key] == n_rows and isinstance(doc['n_unscored'][key], int), key
        assert doc['n_unscored'][key] == toy_run.b0.nll(toy_run.world.s1, toy_run.world.s2, toy_run.world.labels, 'FIT', kind)[head].n_unscored, key
        if by_size is None:
            assert key not in doc['by_size'], key
        else:
            assert doc['by_size'][key] == pytest.approx(by_size, rel=1e-12), key
    assert doc['size_normalised'] == sorted(f'{h}/{k}' for h in loss.SIZE_NORMALISED for k in ('masked', 'open'))
    assert doc['s1'] == {'run': toy_run.names.s1, 'files_sha256': json.loads((toy_run.run / 'RUN.json').read_text(encoding='utf-8'))['s1']['files_sha256']}
    assert KEY_PREFIX not in text and 'matches/key' not in text
    assert scales.load_scales(path) == got and scales.load_scales(tmp_path) == got


def test_b0_scales_are_the_same_numbers_on_every_call(toy_run, tmp_path):
    """Same run, same numbers to the last bit, with or without a file (the scale constants are fixed once and hashed)."""
    from c_ept.ept import scales
    first = scales.b0_scales(toy_run.run, tmp_path, runs_root=toy_run.runs)
    assert first == scales.b0_scales(toy_run.run, None, runs_root=toy_run.runs)
    assert (tmp_path / 'B0_SCALES.json').is_file()


# ---- a wholly unobserved frame: the units of n_rows and n_unscored (fix round 2 of Task 4) -------------------------------------------------
UNSEEN_BIN = 3                                   # the one observed time bin of the first FIT frame row of the unobserved-frame worlds


@pytest.fixture(scope='module')
def unobserved_frame_worlds(toy_run, tmp_path_factory):
    """Two isolated synthetic S1/S2 pairs, built here from the heads world generator with its own seed and never written into a shared fixture.

    ``with_frame``: one FIT match with exactly two frame rows. Row A has one observed bin (``UNSEEN_BIN``; its other 17 bins have no exposure) and
    row B has no exposure at all. The FIT onsets are only the ones in row A and bin ``UNSEEN_BIN`` (an onset in a bin without exposure is a data
    error). The other roles (STOP, DEV, CAL) and the open heads keep the generator's data.
    ``without_frame``: the same arrays with one change: the frame row B belongs to the STOP match (its ``frame_rows/match`` and the ``views/match`` of its
    frame-clock view), so it is not a FIT row. B keeps its zero exposure, so only the role of B differs. The labels and the B0 heads (fitted on the
    full heads world) are shared; nothing is fitted on these tiny worlds."""
    from c_ept.data import frame_rows as frame_rows_mod
    from c_ept.data.store_core import Store
    from c_ept.tests import s3_toys
    root = tmp_path_factory.mktemp('unobserved_frame')
    real_times, real_rows, real_onsets = s3_toys.frame_times, frame_rows_mod.frame_rows, frame_rows_mod.onsets

    def two_rows(rec):                                          # frames 0 .. 3 only: the rows k = 2, 3 (A and B); only the bin UNSEEN_BIN of A is observed
        rows = real_rows(rec)
        exposure = np.zeros_like(rows.exposure_ms)
        exposure[0, UNSEEN_BIN] = rows.exposure_ms[0, UNSEEN_BIN]
        return dataclasses.replace(rows, exposure_ms=exposure)

    def seen_onsets(rows, units):
        found = real_onsets(rows, units)
        keep = (found.row == 0) & (found.bin == UNSEEN_BIN)
        return frame_rows_mod.Onsets(row=found.row[keep], bin=found.bin[keep], mark=found.mark[keep], unit=found.unit[keep])

    s3_toys.frame_times = lambda n: real_times(min(n, 4))
    frame_rows_mod.frame_rows, frame_rows_mod.onsets = two_rows, seen_onsets
    try:
        world = s3_toys.build_heads_world(root / 'with', roles=('FIT', 'STOP', 'DEV', 'CAL'))
    finally:
        s3_toys.frame_times, frame_rows_mod.frame_rows, frame_rows_mod.onsets = real_times, real_rows, real_onsets
    frame_match = np.asarray(world.s1.array('frame_rows/match'))
    from c_ept import config as C
    assert frame_match[:2].tolist() == [0, 0] and np.asarray(world.s1.array('matches/role'))[:2].tolist() == [C.ROLES.index('FIT'), C.ROLES.index('STOP')]   # rows A, B: FIT match 0; match 1 is STOP
    shutil.copytree(world.s1.root, root / 'without' / 's1')
    shutil.copytree(world.s2.root, root / 'without' / 's2')
    s1, s2 = Store(root / 'without' / 's1'), Store(root / 'without' / 's2')
    frame_view = np.flatnonzero((np.asarray(s2.array('views/src')) == 0) & (np.asarray(s2.array('views/src_index')) == 1))
    assert frame_view.size == 1
    for store, name, at in ((s1, 'frame_rows/match', 1), (s2, 'views/match', frame_view[0])):
        array = np.load(store.path(name), mmap_mode='r+')
        array[at] = 1
        array.flush()
        del array
        store.maps.pop(name, None)
        store.finalize()                                        # the new hashes of the changed copy
    yield SimpleNamespace(with_frame=SimpleNamespace(s1=world.s1, s2=world.s2), without_frame=SimpleNamespace(s1=s1, s2=s2), labels=world.labels, b0=toy_run.b0,
                          root=root)
    del world, s1, s2
    gc.collect()                                                # Windows does not delete a folder with open memory maps
    # Preserve this task-created fixture directory; pytest manages its temporary root.


def test_a_wholly_unobserved_frame_has_no_row_but_its_18_bins_are_unscored_cells_and_the_scale_does_not_move(unobserved_frame_worlds, tmp_path):
    """Real path ``HeadRows`` -> ``B0Heads.nll`` -> ``b0_scales`` (nothing is faked). One FIT frame row with a single observed bin gives one row of the M2Rate mean
    and 17 left-out cells; adding a frame row without any observed cell adds no row to ``n_rows`` but its 18 time-bin cells to ``n_unscored`` (so 35, in the unit
    ``time_bin_cell``, not frames), and the M2Rate scale, like the other twelve, stays the same bit for bit. Held fixed: the S1 and S2 arrays, the labels and the
    B0 heads; the only difference is whether the frame row B is a FIT row (see the fixture)."""
    from c_ept.ept import scales
    worlds = unobserved_frame_worlds
    got = {}
    for name in ('without_frame', 'with_frame'):
        w = getattr(worlds, name)
        names = SimpleNamespace(s1='20261020_c_ept_s1_1', s2='20261021_c_ept_s2_3', s3='20261022_c_ept_s3_3')
        runs = tmp_path / name / 'runs'
        run = write_toy_run(runs, w.s1, w.s2, worlds.labels, worlds.b0, names)
        out = tmp_path / name / 'out'
        scale = scales.b0_scales(run, out, runs_root=runs)
        got[name] = (scale, json.loads((out / scales.SCALES_NAME).read_text(encoding='utf-8')))
        scores = worlds.b0.nll(w.s1, w.s2, worlds.labels, 'FIT', 'masked')['M2Rate']        # the HeadScores that ``b0_scales`` saw
        assert scores.per_row()[0].size == 1                                              # one frame row (A) is scored, whatever the number of FIT frame rows
    (scale_without, doc_without), (scale_with, doc_with) = got['without_frame'], got['with_frame']
    key = 'M2Rate/masked'
    assert (doc_without['n_rows'][key], doc_without['n_unscored'][key]) == (1, head_rows.N_Q - 1)      # row A: one observed bin, 17 cells without exposure
    assert (doc_with['n_rows'][key], doc_with['n_unscored'][key]) == (1, 2 * head_rows.N_Q - 1) == (1, 35)   # + the 18 cells of the unobserved frame B
    assert (doc_with['n_rows_unit'][key], doc_with['n_unscored_unit'][key]) == ('frame_row', 'time_bin_cell')
    assert scale_with[key] == scale_without[key] > 0
    assert scale_with == scale_without                                                    # the other twelve blocks do not read the frame row B either
    assert {k: v for k, v in doc_with['n_rows'].items()} == doc_without['n_rows']
    assert scales.load_scales(tmp_path / 'with_frame' / 'out') == scale_with


def copy_runs(toy_run, tmp_path):
    shutil.copytree(toy_run.runs, tmp_path / 'runs')
    return tmp_path / 'runs', tmp_path / 'runs' / toy_run.names.s3


def test_b0_scales_refuse_a_run_whose_files_are_not_the_recorded_ones(toy_run, tmp_path):
    """A run folder is never changed (plan, run records): a model file that is not the one STAGES.json recorded, a store that is not the
    one RUN.json recorded, a label store that changed, a run that is not an S3 run or has no B0 are refused before a score is computed."""
    from c_ept.ept import scales

    def refuse(match, mutate):
        runs, run = copy_runs(toy_run, tmp_path / match.replace(' ', '_')[:20])
        mutate(runs, run)
        with pytest.raises(ValueError, match=match):
            scales.b0_scales(run, None, runs_root=runs)

    refuse('recorded', lambda runs, run: (run / 'fit' / 'b0' / 'b0_heads.json').write_text(
        (run / 'fit' / 'b0' / 'b0_heads.json').read_text(encoding='utf-8') + ' ', encoding='utf-8'))

    def other_s1(runs, run):
        doc = json.loads((run / 'RUN.json').read_text(encoding='utf-8'))
        doc['s1']['files_sha256'] = '0' * 64
        (run / 'RUN.json').write_text(json.dumps(doc), encoding='utf-8')

    refuse('S1 store', other_s1)

    def other_label_store(runs, run):
        ledger = json.loads((run / 'STAGES.json').read_text(encoding='utf-8'))
        ledger['stages']['labels']['outputs']['labels/store/'] = '0' * 64
        (run / 'STAGES.json').write_text(json.dumps(ledger), encoding='utf-8')

    refuse('label store', other_label_store)

    def no_fit_stage(runs, run):
        ledger = json.loads((run / 'STAGES.json').read_text(encoding='utf-8'))
        ledger['stages']['fit']['complete'] = False
        (run / 'STAGES.json').write_text(json.dumps(ledger), encoding='utf-8')

    refuse('fit stage', no_fit_stage)

    def fitted_on_dev(runs, run):                              # a B0 that was fitted on DEV, with the new hash of its JSON recorded: only the role is wrong
        path = run / 'fit' / 'b0' / 'b0_heads.json'
        doc = json.loads(path.read_text(encoding='utf-8'))
        doc['fit_roles'] = ['DEV']
        path.write_text(json.dumps(doc, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        ledger = json.loads((run / 'STAGES.json').read_text(encoding='utf-8'))
        ledger['stages']['fit']['outputs']['fit/b0/b0_heads.json'] = store_core.sha(path)
        (run / 'STAGES.json').write_text(json.dumps(ledger), encoding='utf-8')

    refuse('fitted on', fitted_on_dev)

    def not_s3(runs, run):
        doc = json.loads((run / 'RUN.json').read_text(encoding='utf-8'))
        doc['stage'] = 's2'
        (run / 'RUN.json').write_text(json.dumps(doc), encoding='utf-8')

    refuse('S3 run', not_s3)
    with pytest.raises(ValueError, match='S3 run'):
        scales.b0_scales(tmp_path / 'nothing_here', None, runs_root=toy_run.runs)


def test_b0_scales_call_the_e6_check_before_the_reads_and_between_the_heads(toy_run, tmp_path):
    from c_ept.ept import scales
    calls = []
    scales.b0_scales(toy_run.run, None, runs_root=toy_run.runs, check=lambda: calls.append(1))
    assert len(calls) >= 1 + len(head_rows.HEAD_NAMES) + len(head_rows.OPEN_HEAD_NAMES)

    def refuse():
        raise RuntimeError('E6: stop')

    with pytest.raises(RuntimeError, match='E6'):
        scales.b0_scales(toy_run.run, tmp_path, runs_root=toy_run.runs, check=refuse)
    assert not (tmp_path / 'B0_SCALES.json').exists()


def test_the_scales_file_is_not_written_over_and_load_checks_its_hash(toy_run, tmp_path):
    from c_ept.ept import scales
    scales.b0_scales(toy_run.run, tmp_path, runs_root=toy_run.runs)
    path = tmp_path / 'B0_SCALES.json'
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        scales.b0_scales(toy_run.run, tmp_path, runs_root=toy_run.runs)
    assert path.read_bytes() == before and sorted(p.name for p in tmp_path.iterdir()) == ['B0_SCALES.json']
    doc = json.loads(before)
    doc['scales']['M2Mark/masked'] += 1e-9                                                    # a value that was edited by hand
    path.write_text(json.dumps(doc), encoding='utf-8')
    with pytest.raises(ValueError, match='hash'):
        scales.load_scales(path)
    doc = json.loads(before)
    doc['schema'] = 'b0_scales_v0'
    path.write_text(json.dumps(doc), encoding='utf-8')
    with pytest.raises(ValueError, match='schema'):
        scales.load_scales(path)
    with pytest.raises(FileNotFoundError):
        scales.load_scales(tmp_path / 'nope')


def hand_values() -> dict:
    return {key: 1.5 + i for i, key in enumerate(loss.BLOCKS)}


def hand_doc(**changes) -> dict:
    """A complete, valid ``B0_SCALES.json`` record written by hand: the thirteen blocks, the role FIT, the rule of the module, full provenance
    (S1, S2, label store, two model files) and the units of the counts. ``changes`` replace top-level fields (the hash follows ``scales``)."""
    from c_ept.ept import scales
    keys = list(loss.BLOCKS)
    doc = {'schema': 'b0_scales_v1', 's3_run': '20261022_c_ept_s3_3', 'role': 'FIT', 'rule': scales.RULE,
           's1': {'run': '20261020_c_ept_s1_1', 'files_sha256': '1' * 64}, 's2': {'run': '20261021_c_ept_s2_3', 'files_sha256': '2' * 64},
           'labels_store_sha256': '3' * 64, 'model_files_sha256': {'fit/b0/b0_heads.json': '4' * 64, 'fit/b0/b0_heads.npz': '5' * 64},
           'scales': hand_values(), 'n_rows': {k: 10 for k in keys}, 'n_unscored': {k: 0 for k in keys}, 'by_size': {},
           'n_rows_unit': {k: {'M2Rate': 'frame_row', 'M3Gold': 'team_entry'}.get(k.split('/')[0], 'design_row') for k in keys},
           'n_unscored_unit': {k: {'M2Rate': 'time_bin_cell', 'M3Gold': 'team_entry'}.get(k.split('/')[0], 'design_row') for k in keys},
           'size_normalised': sorted(f'{h}/{k}' for h in loss.SIZE_NORMALISED for k in ('masked', 'open'))}
    doc.update(changes)
    doc.setdefault('scales_sha256', hashlib.sha256(json.dumps(doc['scales'], sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest())
    return doc


def write_hand(tmp_path, doc=None, drop=(), **changes) -> Path:
    doc = hand_doc(**changes) if doc is None else doc
    for name in drop:
        doc.pop(name)
    path = tmp_path / 'B0_SCALES.json'
    path.write_text(json.dumps(doc), encoding='utf-8')
    return path


def test_load_scales_checks_schema_hash_keys_and_values_of_a_hand_written_file(tmp_path):
    """``load_scales`` on a complete file written here by hand (the hash is computed with hashlib from the text of the rule: sorted compact JSON):
    it is read back as it is; a changed value, another schema, a key that is not a block, a scale that is not a positive finite number (a bool
    is not a number here) and a missing file are refused."""
    from c_ept.ept import scales
    assert scales.load_scales(write_hand(tmp_path)) == hand_values()
    with pytest.raises(ValueError, match='hash'):
        scales.load_scales(write_hand(tmp_path, scales={**hand_values(), 'M2Mark/masked': 9.9}, scales_sha256=hashlib.sha256(b'x').hexdigest()))
    with pytest.raises(ValueError, match='schema'):
        scales.load_scales(write_hand(tmp_path, schema='other'))
    with pytest.raises(ValueError, match='block'):
        scales.load_scales(write_hand(tmp_path, scales={**hand_values(), 'M2Mark': 1.5}))
    for bad in (0.0, -2.0, True, False, float('nan'), float('inf'), '1.5', None):
        with pytest.raises(ValueError, match='scale'):
            scales.load_scales(write_hand(tmp_path, scales={**hand_values(), 'M2Mark/masked': bad}))
    with pytest.raises(FileNotFoundError):
        scales.load_scales(tmp_path / 'missing.json')


def test_load_scales_needs_exactly_the_thirteen_blocks(tmp_path):
    """Review P2: an empty mapping with a matching hash, a missing block and an extra block are refused (the loss needs one scale per block)."""
    from c_ept.ept import scales
    with pytest.raises(ValueError, match='block'):
        scales.load_scales(write_hand(tmp_path, scales={}))
    missing = hand_values()
    missing.pop('M2Mark/masked')
    with pytest.raises(ValueError, match='block'):
        scales.load_scales(write_hand(tmp_path, scales=missing))
    with pytest.raises(ValueError, match='block'):
        scales.load_scales(write_hand(tmp_path, scales={**hand_values(), 'M2Rate/open': 1.0}))
    with pytest.raises(ValueError, match='block'):
        scales.load_scales(write_hand(tmp_path, scales=[1.0] * 13))


def test_load_scales_needs_the_role_fit_the_rule_and_the_size_normalised_blocks(tmp_path):
    from c_ept.ept import scales
    for field, bad, match in (('role', 'DEV', 'role'), ('role', None, 'role'), ('rule', 'a plain mean', 'rule'), ('size_normalised', [], 'size_normalised'),
                              ('size_normalised', ['M3Gold/masked'], 'size_normalised'), ('n_rows', {}, 'count'), ('n_unscored', {k: True for k in loss.BLOCKS}, 'count'), ('n_rows_unit', {}, 'unit'), ('n_unscored_unit', None, 'unit')):
        with pytest.raises(ValueError, match=match):
            scales.load_scales(write_hand(tmp_path, **{field: bad}))
    wrong_unit = hand_doc()
    wrong_unit['n_unscored_unit']['M2Rate/masked'] = 'frame_row'                             # the cells of M2Rate are not rows
    with pytest.raises(ValueError, match='unit'):
        scales.load_scales(write_hand(tmp_path, wrong_unit))


@pytest.mark.parametrize('field', ['s3_run', 's1', 's2', 'labels_store_sha256', 'model_files_sha256'])
def test_load_scales_needs_every_provenance_field(tmp_path, field):
    from c_ept.ept import scales
    with pytest.raises(ValueError, match='provenance'):
        scales.load_scales(write_hand(tmp_path, drop=(field,)))


@pytest.mark.parametrize('changes', [
    {'s3_run': ''}, {'s3_run': 5}, {'s1': {'run': '', 'files_sha256': '1' * 64}}, {'s1': {'run': 'r'}}, {'s2': {'run': 'r', 'files_sha256': 'xyz'}},
    {'s2': 'r'}, {'labels_store_sha256': 'abc'}, {'labels_store_sha256': None}, {'model_files_sha256': {}},
    {'model_files_sha256': {'fit/b0/b0_heads.json': '4' * 64}}, {'model_files_sha256': {'fit/b0/b0_heads.json': '4' * 64, 'fit/b0/other.npz': '5' * 64}},
    {'model_files_sha256': {'fit/b0/b0_heads.json': '4' * 64, 'fit/b0/b0_heads.npz': 'zz'}}])
def test_load_scales_refuses_empty_or_malformed_provenance(tmp_path, changes):
    from c_ept.ept import scales
    with pytest.raises(ValueError, match='provenance'):
        scales.load_scales(write_hand(tmp_path, **changes))


def expected_of(doc: dict) -> dict:
    return {name: doc[name] for name in ('s3_run', 's1', 's2', 'labels_store_sha256', 'model_files_sha256')}


def test_load_scales_binds_the_artifact_to_the_expected_provenance(tmp_path):
    """Review P2: the recovery of Task 7 passes the sources of the S3 run it was asked for. A complete matching mapping is accepted; a wrong run,
    store, label store or model file, an incomplete mapping and an unknown field are refused. Without a mapping only the structure is checked."""
    from c_ept.ept import scales
    path = write_hand(tmp_path)
    doc = hand_doc()
    assert scales.load_scales(path, expected_provenance=expected_of(doc)) == hand_values()
    assert scales.load_scales(path, expected_provenance=None) == hand_values()
    other = {'s3_run': 'other_run', 's1': {'run': 'x', 'files_sha256': '1' * 64}, 's2': {'run': '20261021_c_ept_s2_3', 'files_sha256': '9' * 64},
             'labels_store_sha256': '9' * 64, 'model_files_sha256': {'fit/b0/b0_heads.json': '9' * 64, 'fit/b0/b0_heads.npz': '5' * 64}}
    for name, wrong in other.items():
        with pytest.raises(ValueError, match='expected'):
            scales.load_scales(path, expected_provenance={**expected_of(doc), name: wrong})
    for name in expected_of(doc):
        partial = expected_of(doc)
        partial.pop(name)
        with pytest.raises(ValueError, match='expected'):
            scales.load_scales(path, expected_provenance=partial)
    with pytest.raises(ValueError, match='expected'):
        scales.load_scales(path, expected_provenance={**expected_of(doc), 'code': {}})
    with pytest.raises(ValueError, match='expected'):
        scales.load_scales(path, expected_provenance={})


def test_the_units_of_the_counts_follow_the_block(toy_run, tmp_path):
    """Review P2: ``n_rows`` and ``n_unscored`` keep their values and record their units: M2Rate counts frame rows and left-out time-bin cells,
    M3Gold counts team entries, the other heads design rows. The scale values are those of the plain mean (unchanged by the units)."""
    from c_ept.ept import scales
    got = scales.b0_scales(toy_run.run, tmp_path, runs_root=toy_run.runs)
    doc = json.loads((tmp_path / 'B0_SCALES.json').read_text(encoding='utf-8'))
    assert set(doc['n_rows_unit']) == set(doc['n_unscored_unit']) == set(loss.BLOCKS)
    assert doc['n_rows_unit']['M2Rate/masked'] == 'frame_row' and doc['n_unscored_unit']['M2Rate/masked'] == 'time_bin_cell'
    assert doc['n_rows_unit']['M3Gold/masked'] == doc['n_unscored_unit']['M3Gold/masked'] == 'team_entry'
    assert doc['n_rows_unit']['M2Mark/masked'] == doc['n_unscored_unit']['M2Mark/masked'] == 'design_row'
    assert {scales.block_units(key) for key in loss.BLOCKS} <= {('frame_row', 'time_bin_cell'), ('team_entry', 'team_entry'), ('design_row', 'design_row')}
    assert scales.load_scales(tmp_path) == got


def test_a_written_artifact_is_read_back_only_with_the_sources_of_its_own_run(toy_run, tmp_path):
    """The expected sources are built here from the run folder (RUN.json, the ledger, the model files), not from the artifact; the artifact of
    this run is accepted and the same artifact against another recorded source is refused."""
    from c_ept.workflows import run_s3
    from c_ept.ept import scales
    scales.b0_scales(toy_run.run, tmp_path, runs_root=toy_run.runs)
    run_doc = json.loads((toy_run.run / 'RUN.json').read_text(encoding='utf-8'))
    expected = {'s3_run': toy_run.names.s3, 's1': run_doc['s1'], 's2': run_doc['s2'], 'labels_store_sha256': run_s3.identity(toy_run.run, 'labels/store/'),
                'model_files_sha256': {f'fit/b0/{name}': store_core.sha(toy_run.run / 'fit' / 'b0' / name) for name in ('b0_heads.json', 'b0_heads.npz')}}
    assert scales.load_scales(tmp_path, expected_provenance=expected) == scales.load_scales(tmp_path)
    for name, wrong in (('s3_run', '20261022_c_ept_s3_9'), ('s1', {**run_doc['s1'], 'run': 'elsewhere'}), ('labels_store_sha256', '0' * 64),
                        ('model_files_sha256', {**expected['model_files_sha256'], 'fit/b0/b0_heads.npz': '0' * 64})):
        with pytest.raises(ValueError, match='expected'):
            scales.load_scales(tmp_path, expected_provenance={**expected, name: wrong})


def test_partial_and_empty_exposure_in_the_counts_of_m2rate():
    """Review P2 toys: a frame with 1 observed of 18 cells is one included row and 17 left-out cells, not 17 left-out rows; ``head_scale`` counts
    the frame rows that have a score (its second value) and never the cells. A frame with no observed cell leaves no score at all, so no row
    can be counted for it from the scores: the scores of a head with no observed cell have no row and are refused."""
    from c_ept.ept import scales
    one_cell = head_rows.HeadScores(nll=np.array([2.0]), match=np.zeros(1, dtype=np.int64), row=np.array([7]), n_unscored=17)
    value, n_rows, _ = scales.head_scale('M2Rate', one_cell)
    assert (value, n_rows, one_cell.n_unscored) == (2.0, 1, 17)
    two_frames = head_rows.HeadScores(nll=np.array([2.0, 4.0]), match=np.zeros(2, dtype=np.int64), row=np.array([7, 8]), n_unscored=34)
    assert scales.head_scale('M2Rate', two_frames)[:2] == (3.0, 2)
    nothing = head_rows.HeadScores(nll=np.empty(0), match=np.empty(0, dtype=np.int64), row=np.empty(0, dtype=np.int64), n_unscored=18)
    with pytest.raises(ValueError, match='row'):
        scales.head_scale('M2Rate', nothing)
    assert scales.block_units('M2Rate/masked') == ('frame_row', 'time_bin_cell')


def test_the_command_line_refuses_while_strata_runs_and_otherwise_writes_the_file(toy_run, tmp_path, monkeypatch, capsys):
    """E6 for the one-off computation: with a model server running the command refuses (exit code 2, nothing written); otherwise it writes the
    file and prints the scales (counts and numbers, never a key)."""
    from c_ept.workflows import run_s1
    from c_ept.ept import scales
    args = ['--s3', str(toy_run.run), '--out', str(tmp_path / 'out'), '--runs-root', str(toy_run.runs)]

    def running():
        raise run_s1.StrataRunning('E6: a model server runs')

    monkeypatch.setattr(run_s1, 'require_no_strata', running)
    assert scales.main(args) == 2
    assert 'E6' in capsys.readouterr().err and not (tmp_path / 'out' / 'B0_SCALES.json').exists()
    monkeypatch.setattr(run_s1, 'require_no_strata', lambda: None)
    assert scales.main(args) == 0
    out = capsys.readouterr().out
    assert (tmp_path / 'out' / 'B0_SCALES.json').is_file() and 'M2Rate/masked' in out and KEY_PREFIX not in out
    assert scales.main(args) == 2                                                             # the file is there: not written over
    assert 'already' in capsys.readouterr().err


# =====================================================================================================================================
# the module text
# =====================================================================================================================================
LOSS_SOURCE = Path(loss.__file__)


def test_the_loss_module_reads_no_gpu_no_cohort_no_holdout_no_key():
    """The loss module is pure arithmetic on tensors: no device name, no reader, no runner, no cohort module, no raw location, no switch of the
    holdout gate and no match key (its module text and its imports)."""
    text = LOSS_SOURCE.read_text(encoding='utf-8')
    assert 'cuda' not in text and 'allow_holdout' not in text and 'matches/key' not in text and KEY_PREFIX not in text
    assert not all_imports(LOSS_SOURCE) & {'c_ept.cohort', 'c_ept.reader', 'c_ept.run_s1', 'c_ept.run_s2', 'c_ept.run_s3', 'c_ept.store_core',
                                           'pickle', 'lightgbm', 'sklearn'}


def test_the_scales_module_keeps_the_heavy_and_the_runner_imports_inside_its_functions():
    """``scales`` opens runs, so it uses the runner helpers (E6 guard, store openers, the ledger), but only inside functions, as ``run_s3`` does;
    the module level stays light. It names no raw location, opens no gate and never imports the cohort module."""
    from c_ept.ept import scales
    path = Path(scales.__file__)
    top = module_level_imports(path)
    assert not {name for name in top if name.split('.')[0] in ('sklearn', 'lightgbm', 'joblib', 'pickle')}
    assert not top & {'c_ept.run_s1', 'c_ept.run_s2', 'c_ept.run_s3', 'c_ept.baselines_b0', 'c_ept.baselines_b1', 'c_ept.reader', 'c_ept.cohort',
                      'c_ept.flowprice'}
    text = path.read_text(encoding='utf-8')
    assert 'allow_holdout' not in text and 'matches/key' not in text and 'cuda' not in text and KEY_PREFIX not in text
    assert not all_imports(path) & {'c_ept.cohort', 'c_ept.reader', 'pickle'}


def test_the_constants_agree_with_nll_and_head_rows():
    assert loss.ROW_SUM_TOLERANCE == nll._ROW_SUM_TOLERANCE
    assert loss.N_SIZE == N_SIZE == 4 and loss.BLOCKS == tuple(f'{h}/{k}' for k, names in (('masked', HEAD_NAMES), ('open', head_rows.OPEN_HEAD_NAMES))
                                                               for h in names)
    assert set(heads.HEAD_OUTPUT) == set(HEAD_NAMES)
