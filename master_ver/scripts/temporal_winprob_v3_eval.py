"""P2 evaluation helpers: match-weighted metrics with explicit NA cells, paired match bootstrap, strata, plots.

Pure functions; imported by scripts/run_temporal_winprob_v3.py. Metric semantics follow P1
(scripts/train_independent_v2.py): equal total weight per match within each cell, ECE over 10 fixed bins,
logistic recalibration intercept (slope fixed at 1) and slope (Van Calster et al. 2019).
"""
from __future__ import annotations

import math

import numpy as np

MIN_CELL_MATCHES = 30
MIN_CLASS_MATCHES = 5
TIME_BANDS = ((2, 10), (10, 20), (20, 30), (30, 1000))
SIX_SOULS = ('AIR', 'EARTH', 'FIRE', 'WATER', 'HEXTECH', 'CHEMTECH')
COLORS = {'A': '#2a78d6', 'B': '#eb6834', 'ink': '#0b0b0b', 'ink2': '#52514e', 'muted': '#898781',
          'grid': '#e1e0d9', 'axis': '#c3c2b7', 'surface': '#fcfcfb'}


def band_label(lo, hi):
    return f'{lo}-{hi}' if hi < 1000 else f'{lo}+'


def match_weights(groups):
    _, ix, counts = np.unique(groups, return_inverse=True, return_counts=True)
    w = 1. / counts[ix]
    return w / w.mean()


def ece_from(y, p, w, bins=10):
    edges = np.linspace(0., 1., bins + 1)
    total, out = w.sum(), 0.
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi + (1e-10 if hi >= 1. else 0.))
        if mask.any():
            ww = w[mask]
            out += float(ww.sum() / total) * abs(float(np.average(y[mask], weights=ww)) - float(np.average(p[mask], weights=ww)))
    return float(out)


def calibration_line(y, p, w):
    from scipy.special import expit
    q = np.clip(p, 1e-10, 1 - 1e-10)
    lp = np.log(q / (1 - q))

    def newton(Z, offset):
        beta = np.zeros(Z.shape[1])
        for _ in range(200):
            mu = expit(offset + Z @ beta)
            grad = Z.T @ (w * (y - mu))
            hess = Z.T @ (Z * (w * mu * (1 - mu))[:, None])
            step = np.linalg.solve(hess + 1e-12 * np.eye(len(beta)), grad)
            beta = beta + step
            if np.max(np.abs(step)) < 1e-10:
                return beta, True
        return beta, False

    a, ok_a = newton(np.ones((len(y), 1)), lp)
    b, ok_b = newton(np.column_stack([np.ones(len(y)), lp]), np.zeros(len(y)))
    return {'calibration_intercept': float(a[0]), 'calibration_slope': float(b[1]),
            'slope_model_intercept': float(b[0]), 'calibration_line_converged': bool(ok_a and ok_b)}


def reliability_bins(y, p, w, groups, bins=10):
    edges = np.linspace(0., 1., bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi + (1e-10 if hi >= 1. else 0.))
        cell = {'lo': float(lo), 'hi': float(hi), 'rows': int(mask.sum()), 'matches': int(len(set(groups[mask].tolist())))}
        if mask.any():
            ww = w[mask]
            cell.update(weight_share=float(ww.sum() / w.sum()), predicted=float(np.average(p[mask], weights=ww)),
                        observed=float(np.average(y[mask], weights=ww)))
        out.append(cell)
    return out


def class_matches(y, groups):
    return len(set(groups[y == 1].tolist())), len(set(groups[y == 0].tolist()))


def cell_status(y, groups):
    matches = len(set(groups.tolist()))
    pos, neg = class_matches(y, groups)
    if matches < MIN_CELL_MATCHES:
        return 'NA_low_count', matches, pos, neg
    if min(pos, neg) < MIN_CLASS_MATCHES:
        return 'NA_one_class_or_low_class_count', matches, pos, neg
    return 'ok', matches, pos, neg


def cell_metrics(y, p, groups):
    """Match-weighted metrics; explicit NA for low-count cells (all metrics) and one-class cells (AUC/calibration)."""
    from train.state_value_experiment import metrics
    y = np.asarray(y).astype(np.int64)
    p = np.asarray(p, dtype=np.float64)
    groups = np.asarray(groups)
    status, matches, pos, neg = cell_status(y, groups)
    cell = {'status': status, 'rows': int(len(y)), 'matches': matches, 'positive_matches': pos, 'negative_matches': neg,
            'min_cell_matches': MIN_CELL_MATCHES, 'min_class_matches': MIN_CLASS_MATCHES}
    keys = ('auc', 'brier', 'log_loss', 'ece', 'calibration_intercept', 'calibration_slope')
    if status == 'NA_low_count':
        cell.update({k: None for k in keys})
        return cell
    m = metrics(y, p, groups)
    w = match_weights(groups)
    cell.update(positive_rate_match_weighted=m['positive_rate'], brier=m['brier'], log_loss=m['log_loss'],
                ece=ece_from(y, p, w), mean_prediction_match_weighted=float(np.average(p, weights=w)),
                reliability_bins=reliability_bins(y, p, w, groups))
    if status == 'ok':
        cell['auc'] = m['auc']
        cell.update(calibration_line(y.astype(float), p, w))
    else:
        cell.update(auc=None, calibration_intercept=None, calibration_slope=None)
    return cell


class SortedAUC:
    """Weighted Mann-Whitney AUC (ties count one half) with a fixed score order; equals sklearn roc_auc_score."""

    def __init__(self, y, p):
        self.order = np.argsort(p, kind='mergesort')
        ps = np.asarray(p)[self.order]
        self.ys = np.asarray(y, dtype=np.float64)[self.order]
        self.starts = np.flatnonzero(np.r_[True, ps[1:] != ps[:-1]])

    def __call__(self, w):
        ws = np.asarray(w, dtype=np.float64)[self.order]
        wp = np.add.reduceat(ws * self.ys, self.starts)
        wn = np.add.reduceat(ws * (1 - self.ys), self.starts)
        den = wp.sum() * wn.sum()
        if den <= 0:
            return float('nan')
        below = np.cumsum(wn) - wn
        return float(((wp * below).sum() + .5 * (wp * wn).sum()) / den)


def row_losses(y, p):
    eps = np.finfo(np.float64).eps
    q = np.clip(p, eps, 1 - eps)
    return (p - y) ** 2, -(y * np.log(q) + (1 - y) * np.log(1 - q))


def paired_bootstrap(y, p_new, p_ref, groups, n_boot, seed, new='B', ref='A'):
    """Resample matches with replacement; models fixed. Returns new minus ref for AUC, Brier, log loss."""
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y).astype(np.float64)
    groups = np.asarray(groups)
    status, matches, pos, neg = cell_status(y.astype(int), groups)
    if status != 'ok':
        return {'status': status, 'matches': matches, 'rows': int(len(y))}
    unique, inverse = np.unique(groups, return_inverse=True)
    base = match_weights(groups)
    bn, ln = row_losses(y, p_new)
    br, lr = row_losses(y, p_ref)
    auc_n, auc_r = SortedAUC(y, p_new), SortedAUC(y, p_ref)
    check = max(abs(auc_n(base) - roc_auc_score(y, p_new, sample_weight=base)),
                abs(auc_r(base) - roc_auc_score(y, p_ref, sample_weight=base)))
    if check > 1e-9:
        raise ValueError(f'fast AUC differs from sklearn by {check}')

    def stats(w):
        return (auc_n(w) - auc_r(w), float(np.average(bn - br, weights=w)), float(np.average(ln - lr, weights=w)))

    point = stats(base)
    rng = np.random.default_rng(seed)
    draws, skipped = [], 0
    for _ in range(n_boot):
        mult = np.bincount(rng.integers(0, len(unique), size=len(unique)), minlength=len(unique))
        w = base * mult[inverse]
        if len(set(y[w > 0].tolist())) < 2:
            skipped += 1
            continue
        draws.append(stats(w))
    D = np.asarray(draws)
    out = {'status': 'ok', 'difference': f'{new} minus {ref}', 'matches': int(len(unique)), 'rows': int(len(y)),
           'n_boot_requested': int(n_boot), 'n_boot_valid': int(len(D)), 'skipped_single_class': skipped, 'seed': int(seed),
           'fast_auc_max_abs_diff_vs_sklearn': float(check),
           'better_direction': {'auc': 'positive', 'brier': 'negative', 'log_loss': 'negative'},
           'uncertainty_scope': 'matches resampled with replacement (equal total weight per match); fitted models, '
                                'calibrators and selections held fixed'}
    for j, name in enumerate(('auc', 'brier', 'log_loss')):
        col = D[:, j]
        better = col > 0 if name == 'auc' else col < 0
        out[name] = {'point': float(point[j]), 'ci95': np.quantile(col, [.025, .975]).tolist(),
                     'bootstrap_se': float(col.std(ddof=1)), f'fraction_replicates_{new}_better': float(better.mean())}
    return out


def objective_strata(X, names, unassigned):
    """Identical definitions to P1 (scripts/train_independent_v2.py objective_strata)."""
    ix = {n: i for i, n in enumerate(names)}

    def col(n):
        return X[:, ix[n]]

    def either(n):
        return (col('blue_' + n) > 0) | (col('red_' + n) > 0)

    owned = either('soul_event_recorded')
    unassigned = np.asarray(unassigned) > 0
    strata = {
        'no_baron_elder_or_soul_history': ~(either('baron_ever') | either('elder_ever') | owned | unassigned),
        'dragons_any': either('dragons'),
        'dragon_count_diff_ge2': np.abs(col('blue_dragons') - col('red_dragons')) >= 2,
        'herald_or_grubs_any': either('herald_ever') | either('horde_ever'),
        'atakhan_any': either('atakhan_ever'),
        'inhibitor_kill_any': either('inhibitor_kills'),
        'baron_ever_any': either('baron_ever'),
        'baron_acquired_last_180s_any': either('baron_acquired_last_180s'),
        'elder_ever_any': either('elder_ever'),
        'elder_acquired_last_180s_any': either('elder_acquired_last_180s'),
        'owned_soul_any': owned,
        'unassigned_soul_teamId0_any_DIAGNOSTIC': unassigned,
        'unassigned_soul_teamId0_without_owned_soul_DIAGNOSTIC': unassigned & ~owned,
        'unknown_objective_team_count_positive': col('unknown_objective_team_count') > 0,
    }
    for d in SIX_SOULS:
        strata[f'owned_soul_{d}'] = either(f'soul_{d}')
    return strata


def agreement(p_new, p_ref):
    d = np.abs(p_new - p_ref)
    return {'pearson_r': float(np.corrcoef(p_new, p_ref)[0, 1]), 'mean_abs_diff': float(d.mean()),
            'abs_diff_quantiles_50_90_99': np.quantile(d, [.5, .9, .99]).tolist(),
            'fraction_abs_diff_gt_0.05': float((d > .05).mean()), 'fraction_abs_diff_gt_0.10': float((d > .10).mean())}


def compare_cell(y, groups, preds, n_boot, seed, pair=('B', 'A')):
    cell = {'rows': int(len(y)), 'matches': int(len(set(np.asarray(groups).tolist())))}
    for k, p in preds.items():
        cell[k] = cell_metrics(y, p, groups)
    if n_boot and all(k in preds for k in pair):
        cell[f'paired_bootstrap_{pair[0]}_minus_{pair[1]}'] = paired_bootstrap(y, preds[pair[0]], preds[pair[1]], groups,
                                                                               n_boot, seed, *pair)
    return cell


# ----------------------------------------------------------------------------- plots
def _style(ax):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(COLORS['axis'])
    ax.tick_params(colors=COLORS['ink2'], labelsize=8)
    ax.grid(axis='y', color=COLORS['grid'], lw=.6)
    ax.set_axisbelow(True)


EVENT_LETTER = {'dragon': 'D', 'elder': 'E', 'baron': 'B', 'herald': 'H', 'horde': 'G', 'atakhan': 'A',
                'tower': 'T', 'inhibitor': 'I', 'soul_owned': 'S', 'soul_unassigned': 's', 'plate': 'p'}


def plot_trajectories_png(path, matches, has_b):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n = len(matches)
    cols = 3 if n > 4 else max(1, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 3.1 * rows), squeeze=False, facecolor=COLORS['surface'])
    for ax, m in zip(axes.ravel(), matches):
        _style(ax)
        x = np.asarray(m['query_ms']) / 60000.
        for f in m['frames_ms']:
            ax.axvline(f / 60000., color=COLORS['grid'], lw=.5, zorder=0)
        ax.axhline(.5, color=COLORS['axis'], lw=.6, zorder=0)
        ax.step(x, m['p_A'], where='post', color=COLORS['A'], lw=1.3, label='A snapshot (frozen P1)')
        if has_b and m.get('p_B') is not None:
            ax.step(x, m['p_B'], where='post', color=COLORS['B'], lw=1.3, label='B history SimpleRNN')
        for e in m['events']:
            if e['category'] == 'champion_kill':
                ax.plot([e['ts'] / 60000.] * 2, [0, .025], color=COLORS['muted'], lw=.5)
            elif e['category'] in EVENT_LETTER:
                ax.text(e['ts'] / 60000., .045, EVENT_LETTER[e['category']], fontsize=6, color=COLORS['ink2'],
                        ha='center', va='bottom')
        ax.set_ylim(0, 1)
        ax.set_xlim(x.min(), x.max())
        ax.set_title(f"{m['match']}  final Blue win={m['winner_blue']}", fontsize=9, color=COLORS['ink'], loc='left')
        ax.set_xlabel('game time (min)', fontsize=8, color=COLORS['ink2'])
        ax.set_ylabel('p(final Blue win)', fontsize=8, color=COLORS['ink2'])
    for ax in axes.ravel()[n:]:
        ax.axis('off')
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', frameon=False, fontsize=9, ncol=2, bbox_to_anchor=(.995, .995))
    fig.suptitle('p(final Blue win) at query times: A frozen snapshot vs B history SimpleRNN', fontsize=11,
                 color=COLORS['ink'], x=.01, ha='left')
    fig.text(.01, .005, 'Evaluated at a 10 s query cadence plus 1 ms before/at every raw frame and predictor event; steps hold '
                        'the last evaluated value (no smoothing). Grey verticals: raw frame timestamps (~60 s cadence). Letters: '
                        'D dragon, E elder, B baron, H herald, G grubs, A atakhan, T tower, I inhibitor, S/s soul (owned/teamId 0), '
                        'p plate; ticks: champion kills.', fontsize=8, color=COLORS['ink2'], ha='left', va='bottom', wrap=True)
    fig.tight_layout(rect=(0, .05, 1, .95))
    fig.savefig(path, dpi=130, facecolor=COLORS['surface'])
    plt.close(fig)


def plot_trajectories_html(path, matches, has_b, note):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    n = len(matches)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=False, vertical_spacing=.25 / max(n, 1),
                        subplot_titles=[f"{m['match']} - final Blue win={m['winner_blue']}" for m in matches])
    for i, m in enumerate(matches, start=1):
        x = np.asarray(m['query_ms']) / 60000.
        hover = [f"query {q} ms<br>snapshot {s} ms (age {a:.1f} s)<br>valid history positions {v}/9<br>queried for: {r}"
                 for q, s, a, v, r in zip(m['query_ms'], m['snapshot_ms'], m['snapshot_age_s'], m['valid_positions'], m['reasons'])]
        fig.add_trace(go.Scatter(x=x, y=m['p_A'], mode='lines+markers', line_shape='hv', name='A snapshot (frozen P1)',
                                 line=dict(color=COLORS['A'], width=1.6), marker=dict(size=3), text=hover,
                                 hovertemplate='A=%{y:.4f}<br>%{text}<extra></extra>', legendgroup='A', showlegend=i == 1),
                      row=i, col=1)
        if has_b and m.get('p_B') is not None:
            fig.add_trace(go.Scatter(x=x, y=m['p_B'], mode='lines+markers', line_shape='hv', name='B history SimpleRNN',
                                     line=dict(color=COLORS['B'], width=1.6), marker=dict(size=3), text=hover,
                                     hovertemplate='B=%{y:.4f}<br>%{text}<extra></extra>', legendgroup='B', showlegend=i == 1),
                          row=i, col=1)
        ev = [e for e in m['events']]
        if ev:
            fig.add_trace(go.Scatter(
                x=[e['ts'] / 60000. for e in ev], y=[.03 if e['category'] == 'champion_kill' else .08 for e in ev],
                mode='markers', marker=dict(size=[4 if e['category'] == 'champion_kill' else 8 for e in ev],
                                            color=COLORS['muted'], symbol=['line-ns-open' if e['category'] == 'champion_kill' else 'diamond' for e in ev]),
                text=[f"{e['ts']} ms {e['type']} {e['detail']}<br>events at this timestamp: {e['n_events_same_ts']} "
                      f"(predictor events {e['n_predictor_events_same_ts']}); frame at timestamp: {e['frame_at_ts']}" for e in ev],
                hovertemplate='%{text}<extra></extra>', name='observed events', legendgroup='ev', showlegend=i == 1),
                row=i, col=1)
        for f in m['frames_ms']:
            fig.add_vline(x=f / 60000., line=dict(color=COLORS['grid'], width=1), row=i, col=1)
        fig.update_yaxes(range=[0, 1], title_text='p(final Blue win)', row=i, col=1, gridcolor=COLORS['grid'])
        fig.update_xaxes(title_text='game time (min)', row=i, col=1, gridcolor=COLORS['grid'])
    fig.update_layout(height=300 * n + 120, width=1150, plot_bgcolor=COLORS['surface'], paper_bgcolor=COLORS['surface'],
                      font=dict(family='system-ui, -apple-system, Segoe UI, sans-serif', color=COLORS['ink']),
                      title=dict(text=note, font=dict(size=12, color=COLORS['ink2'])), hovermode='closest')
    fig.write_html(path, include_plotlyjs=True, full_html=True)


def plot_minute_bands(path, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ok = [r for r in rows if r['status'] == 'ok']
    if not ok:
        return False
    x = np.asarray([r['minute'] for r in ok])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True, facecolor=COLORS['surface'])
    for ax in (a1, a2):
        _style(ax)
    a1.plot(x, [r['log_loss_A'] for r in ok], color=COLORS['A'], lw=2, label='A snapshot (frozen P1)')
    if ok[0].get('log_loss_B') is not None:
        a1.plot(x, [r['log_loss_B'] for r in ok], color=COLORS['B'], lw=2, label='B history SimpleRNN')
        d = np.asarray([r['log_loss_diff_B_minus_A'] for r in ok])
        lo = np.asarray([r['log_loss_diff_ci_lo'] for r in ok], dtype=float)
        hi = np.asarray([r['log_loss_diff_ci_hi'] for r in ok], dtype=float)
        a2.fill_between(x, lo, hi, color=COLORS['grid'], lw=0, label='exploratory 95% match-bootstrap interval')
        a2.plot(x, d, color=COLORS['ink2'], lw=1.6, label='B minus A')
        a2.axhline(0, color=COLORS['axis'], lw=.8)
        a2.set_ylabel('log loss difference', fontsize=9, color=COLORS['ink2'])
        a2.legend(frameon=False, fontsize=8)
    a1.set_ylabel('match-weighted log loss', fontsize=9, color=COLORS['ink2'])
    a1.legend(frameon=False, fontsize=8)
    a2.set_xlabel('stored test grid minute (one row per match per minute; NA cells omitted)', fontsize=9, color=COLORS['ink2'])
    fig.suptitle('Full stored test grid by one-minute band (exploratory; no multiple-testing correction)', fontsize=10,
                 color=COLORS['ink'], x=.01, ha='left')
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=COLORS['surface'])
    plt.close(fig)
    return True


def plot_endpoint_deltas(path, dA, dB):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.2, 6), facecolor=COLORS['surface'])
    _style(ax)
    lim = float(np.quantile(np.abs(np.r_[dA, dB]), .995))
    hb = ax.hexbin(dA, dB, gridsize=70, extent=(-lim, lim, -lim, lim), bins='log', cmap='Blues', mincnt=1)
    ax.plot([-lim, lim], [-lim, lim], color=COLORS['ink2'], lw=.8)
    ax.axhline(0, color=COLORS['axis'], lw=.8)
    ax.axvline(0, color=COLORS['axis'], lw=.8)
    ax.set_xlabel('delta A = p_post - p_pre (frozen snapshot)', fontsize=9, color=COLORS['ink2'])
    ax.set_ylabel('delta B = p_post - p_pre (frozen history RNN)', fontsize=9, color=COLORS['ink2'])
    ax.set_title('DIAGNOSTIC engagement endpoint transport (rows; log colour scale)', fontsize=9, color=COLORS['ink'], loc='left')
    fig.colorbar(hb, ax=ax, label='rows (log)')
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=COLORS['surface'])
    plt.close(fig)
