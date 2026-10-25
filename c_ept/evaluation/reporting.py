"""Local, aggregate-only JSON/CSV/Korean/SVG reporting for EPT DEV diagnostics.

No inference, data access, remote requests, or plotting-library import on the
default path. render=True is for a caller that has already verified terminal
training state; lifecycle/permission enforcement belongs to the lead's operator.
"""
from __future__ import annotations

import csv
from html import escape
import json
from pathlib import Path

from .metrics import _json


def _number(value):
    return 'unavailable' if value is None else f'{value:.6g}'


def _interval(value):
    return 'unavailable' if value is None else f'[{_number(value[0])}, {_number(value[1])}]'


def _rows(summary):
    for kind in ('endpoints', 'nll'):
        for entry in summary[kind]:
            for metric, values in entry['metrics'].items():
                yield kind, entry, metric, values
    structural = summary['structural_normalized_nll']
    if structural['metrics'] is not None:
        yield 'structural', {'name': 'explicit_common_heads', 'heads': structural['heads']}, 'normalized_nll', structural['metrics']


def _flat_rows(summary):
    result = []
    for kind, entry, metric, values in _rows(summary):
        row = {'kind': kind, 'name': entry['name'], 'module': entry.get('module'),
               'layer': entry.get('layer'), 'horizon_s': entry.get('horizon_s'),
               'target': entry.get('target'), 'metric': metric, 'scale': entry.get('scale'),
               'common_heads': '|'.join(entry.get('heads', [])),
               'n_rows': entry.get('counts', {}).get('n_rows'),
               'n_matches': entry.get('counts', {}).get('n_matches'),
               'valid_replicates': values['valid_replicates'],
               'invalid_replicates': values['invalid_replicates'],
               'ci_status': values['ci_status'], 'reason': values['reason']}
        for model in ('ept', 'b1', 'delta'):
            value = values[model]
            row[model] = value['estimate']
            for interval in ('ci95', 'conditional_ci95'):
                bounds = value[interval] or [None, None]
                row[f'{model}_{interval}_low'], row[f'{model}_{interval}_high'] = bounds
        result.append(row)
    return result


def _csv(path, rows, empty_fields):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else empty_fields)
        writer.writeheader()
        for row in rows:
            # Explicit missing marker: unavailable is never rendered as numeric zero.
            writer.writerow({k: 'unavailable' if v is None else v for k, v in row.items()})


def _markdown(summary):
    conditional = []
    lines = ['# EPT-T003 DEV 모듈 진단', '',
             'EPT는 FIT/STOP 25% 단일 pilot, B1은 기존 전체 FIT 모델이다. 학습량을 통제한 아키텍처 비교가 아니다.', '',
             '경기 균형 가중은 endpoint의 최종 eligible 행에서 w=1/n_match_rows로 다시 계산한다. '
             'AUC는 pooled 가중 양성·음성 쌍의 순위 비교이며 동점은 0.5다. 경기별 AUC 평균이나 within-match AUC가 아니다.', '',
             f"Paired 경기 bootstrap {summary['bootstrap']['replicates']}회, seed={summary['bootstrap']['seed']}, percentile 95% CI. "
             'Δ=EPT−B1이며 AUC는 양수, Brier/NLL은 음수가 유리하다.', '',
             '단일 학습 seed의 CI는 경기 표본 변동만 반영하며 학습 seed 변동을 포함하지 않는다. '
             'M3/M4는 열린 조건부 층의 성능이고 전체 t0·합성 가치 목표의 달성을 의미하지 않는다.', '',
             'unavailable은 미측정 또는 정의상 미정이며 0이 아니다. AUC 무효 반복이 하나라도 있으면 주 CI는 unavailable이다. '
             '조건부 CI는 유효 반복만의 진단이며 주 CI를 대신하지 않는다. ROC origin의 null threshold는 최대 점수 초과를 뜻한다.', '',
             '| Endpoint / 층 | 지표 | EPT [주 95% CI] | B1 [주 95% CI] | Δ [주 95% CI] | 유효/무효 반복 |',
             '|---|---|---|---|---|---|']
    safe = lambda text: str(text).replace('|', '\\|').replace('\n', ' ')
    for _, entry, metric, values in _rows(summary):
        cells = [safe(f"{entry['name']} / {entry.get('layer', 'common head')}"), metric]
        for model in ('ept', 'b1', 'delta'):
            item = values[model]
            cells.append(f"{_number(item['estimate'])} {_interval(item['ci95'])}")
        cells.append(f"{values['valid_replicates']}/{values['invalid_replicates']}")
        lines.append('| ' + ' | '.join(cells) + ' |')
        if values['invalid_replicates']:
            conditional.append(f"- 조건부 진단 ({safe(entry['name'])}, {metric}): " + '; '.join(
                f"{m} {_interval(values[m]['conditional_ci95'])}" for m in ('ept', 'b1', 'delta')))
    if conditional:
        lines.extend(['', *conditional])
    lines.extend(['', '## 분모·무승부·Monte Carlo 오차', ''])
    for entry in summary['endpoints']:
        c = entry['counts']
        lines.append(f"- {safe(entry['name'])}: {safe(entry['module'])}/{safe(entry['layer'])}, " +
                     ('실제 교전/전환 관측창, ' if entry['horizon_s'] is None else f"H={entry['horizon_s']}s, ") +
                     f"target={safe(entry['target'])}; "
                     f"행 {c['n_rows']}, 경기 {c['n_matches']}, 양성 {c['n_positive']}, 음성 {c['n_negative']}, "
                     f"가중 합 {_number(c['weight_sum'])}. 선택/제외: "
                     + ('unavailable' if c['selection'] is None else safe(json.dumps(c['selection'], ensure_ascii=False))))
        draw = entry['draw']
        lines.append('  무승부: ' + ('unavailable' if draw is None else
                     f"{draw['n_rows']}행, 가중 비율 {_number(draw['weighted_fraction'])}, "
                     f"EPT p0 {_number(draw['mean_probability']['ept'])}, B1 p0 {_number(draw['mean_probability']['b1'])}"))
        lines.append('  MC 표준오차: ' + ('unavailable' if entry['mc_uncertainty'] is None else
                     safe(json.dumps(entry['mc_uncertainty'], ensure_ascii=False))) + '. 경기 bootstrap CI와 별개다.')
    structural = summary['structural_normalized_nll']
    lines.extend(['', '## 구조 NLL의 범위', '',
                  '명시된 공통 head: ' + (', '.join(map(safe, structural['heads'])) or 'unavailable'),
                  '고정 B0 scale로 정규화한 공통 head의 동일 가중 평균만 구조 요약에 포함한다. '
                  '모든 학습 13블록 합이 아니며, 평균 개선을 모든 head의 개선으로 해석하지 않는다.', '',
                  'JSON/CSV/SVG는 같은 요약을 사용한다. 확률 bin은 [0,.1), …, [.9,1]이며 빈 bin의 관측률은 unavailable이다. '
                  '이 보고 함수의 성공은 실제 DEV 평가 실행·입력 identity 검증·독립 검토 완료를 뜻하지 않는다.', ''])
    return '\n'.join(lines)


def _svg(summary):
    """Lightweight local charts, including genuine unavailable states."""
    endpoints = summary['endpoints']
    nll_rows = [r for r in _rows(summary) if r[0] != 'endpoints']
    height = 110 + 470 * len(endpoints) + max(1, len(nll_rows)) * 75 + 60
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="{height}" viewBox="0 0 1100 {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="Malgun Gothic, sans-serif" fill="#18212b" font-size="13">']

    def text(x, y, value, size=13):
        parts.append(f'<text x="{x}" y="{y}" font-size="{size}">{escape(str(value))}</text>')

    def plot(x, y, title, series, diagonal=False):
        width, h = 320, 230
        text(x, y - 12, title, 16)
        parts.append(f'<path d="M{x},{y} V{y+h} H{x+width}" fill="none" stroke="#687685"/>')
        if diagonal:
            parts.append(f'<path d="M{x},{y+h} L{x+width},{y}" stroke="#aab2bc" stroke-dasharray="4 4"/>')
        for val in (0, .5, 1):
            text(x - 25, y + h * (1 - val) + 4, val)
            text(x + width * val - 5, y + h + 18, val)
        for model, xs, ys, color in series:
            if xs is None:
                text(x + 15, y + 35 + 22 * (model == 'B1'), f'{model}: unavailable')
                continue
            points = [(x + width * a, y + h * (1 - b)) for a, b in zip(xs, ys) if a is not None and b is not None]
            if len(points) > 500:
                # Display-only thinning keeps this SVG small; JSON keeps every
                # weighted ROC tie threshold used for the exact AUC.
                points = [points[i * (len(points) - 1) // 499] for i in range(500)]
            if points:
                joined = ' '.join(f'{a:.3f},{b:.3f}' for a, b in points)
                parts.append(f'<polyline points="{joined}" fill="none" stroke="{color}" stroke-width="2"/>')
                for a, b in points:
                    parts.append(f'<circle cx="{a:.3f}" cy="{b:.3f}" r="2.5" fill="{color}"/>')
            else:
                text(x + 15, y + 35 + 22 * (model == 'B1'), f'{model}: unavailable')

    text(30, 32, 'EPT-T003: endpoint별 DEV 진단', 23)
    text(30, 57, 'EPT 25% pilot vs B1 whole FIT | EPT: blue / B1: orange | CI: 단일 seed 경기 bootstrap')
    text(30, 80, 'M3/M4 조건부 층. unavailable ≠ 0. ROC 표시 최대 500점; 전체 좌표와 조건부 CI는 JSON/CSV 참조.')
    colors = {'ept': '#1769aa', 'b1': '#c45b14'}
    y = 120
    for entry in endpoints:
        c = entry['counts']
        window = 'observed event window' if entry['horizon_s'] is None else f"H={entry['horizon_s']}s"
        text(30, y, f"{entry['name']} | {entry['module']} / {entry['layer']} | {window} | {entry['target']}", 18)
        text(30, y + 23, f"rows={c['n_rows']}; matches={c['n_matches']}; positive={c['n_positive']}; negative={c['n_negative']}; weight={_number(c['weight_sum'])}")
        for offset, (name, result) in enumerate(entry['metrics'].items()):
            text(30, y + 46 + offset * 22, name + ': ' + ' | '.join(
                f"{m}={_number(result[m]['estimate'])} CI {_interval(result[m]['ci95'])}" for m in ('ept', 'b1', 'delta')))
        series = [(m.upper(), entry['roc'][m]['fpr'], entry['roc'][m]['tpr'], colors[m]) for m in colors]
        plot(65, y + 140, 'ROC: x=FPR, y=TPR', series, True)
        series = [(m.upper(), [b['mean_prediction'] for b in entry['reliability'][m]],
                   [b['observed_fraction'] for b in entry['reliability'][m]], colors[m]) for m in colors]
        plot(600, y + 140, 'Reliability: x=예측확률, y=관측률', series, True)
        for offset, m in enumerate(colors):
            text(550, y + 410 + offset * 20, f'{m} bin rows: ' + ', '.join(str(b['n_rows']) for b in entry['reliability'][m]))
        y += 470
    text(30, y, '공통 head NLL (scale은 사전 고정, 전체 학습 loss 합 아님)', 18)
    y += 30
    if not nll_rows:
        text(30, y, 'NLL: unavailable')
    for _, entry, metric, values in nll_rows:
        text(30, y, f"{entry['name']} / {metric}")
        text(30, y + 23, ' | '.join(f"{m}={_number(values[m]['estimate'])} CI {_interval(values[m]['ci95'])}" for m in ('ept', 'b1', 'delta')))
        y += 75
    parts.append('</g></svg>')
    return '\n'.join(parts)


def _png(summary, path):
    # Only reached by explicit opt-in after the operator's terminal-state gate.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    rows = list(_rows(summary))
    fig = Figure(figsize=(12, max(3, len(rows) * 1.1)), layout='constrained')
    FigureCanvasAgg(fig)
    axes = fig.subplots(max(1, len(rows)), 1, squeeze=False).ravel()
    if not rows:
        axes[0].text(.5, .5, 'unavailable', ha='center')
        axes[0].set_axis_off()
    for ax, (_, entry, metric, result) in zip(axes, rows):
        for position, model, color in ((0, 'ept', '#1769aa'), (1, 'b1', '#c45b14'), (2, 'delta', '#555555')):
            item = result[model]
            if item['estimate'] is None:
                ax.text(.02, position, f'{model}: unavailable', transform=ax.get_yaxis_transform())
                continue
            ax.plot(item['estimate'], position, 'o', color=color)
            if item['ci95'] is not None:
                ax.plot(item['ci95'], [position, position], color=color)
            else:
                ax.annotate(' CI unavailable', (item['estimate'], position), fontsize=8)
        ax.set_yticks([0, 1, 2], ['EPT', 'B1', 'delta'])
        ax.set_ylim(-.7, 2.7)
        ax.set_title(f"{entry['name']} / {metric}", fontsize=10)
        ax.grid(axis='x', alpha=.2)
    fig.savefig(path, dpi=140)


def write_report(summary, out, *, provenance=None, render=False):
    """Write local aggregates and return absolute artifact paths.

    provenance must itself contain only approved aggregate metadata. No source
    files are opened. render defaults to False and never checks/starts services.
    The caller owns guards and must verify terminal state before render=True.
    """
    summary = _json(summary)
    provenance = _json(provenance)
    if not isinstance(render, bool):
        raise ValueError('render must be boolean')
    # Validate and prepare content before any output files are written.
    payload = json.dumps({**summary, 'provenance': provenance}, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    metric_rows = _flat_rows(summary)
    markdown, svg = _markdown(summary), _svg(summary)
    calibration_rows, count_rows = [], []
    for entry in summary['endpoints']:
        for model, bins in entry['reliability'].items():
            calibration_rows.extend({'name': entry['name'], 'model': model, **b} for b in bins)
        count_rows.append({'name': entry['name'], **{k: v for k, v in entry['counts'].items() if k != 'selection'},
                           'selection': json.dumps(entry['counts']['selection'], ensure_ascii=False) if entry['counts']['selection'] is not None else None,
                           'draw_rows': None if entry['draw'] is None else entry['draw']['n_rows'],
                           'mc_uncertainty': json.dumps(entry['mc_uncertainty'], ensure_ascii=False) if entry['mc_uncertainty'] is not None else None})
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    names = {'metrics_json': 'metrics.json', 'metrics_csv': 'metrics.csv',
             'reliability_csv': 'reliability.csv', 'counts_csv': 'counts.csv',
             'report': 'FINAL_REPORT_KO.md', 'svg': 'diagnostics.svg'}
    paths = {key: out / filename for key, filename in names.items()}
    paths['metrics_json'].write_text(payload, encoding='utf-8')
    paths['report'].write_text(markdown, encoding='utf-8')
    paths['svg'].write_text(svg, encoding='utf-8')
    _csv(paths['metrics_csv'], metric_rows, ['kind', 'name', 'metric', 'ept', 'b1', 'delta'])
    _csv(paths['reliability_csv'], calibration_rows, ['name', 'model', 'left', 'right', 'n_rows', 'weight_sum', 'mean_prediction', 'observed_fraction'])
    _csv(paths['counts_csv'], count_rows, ['name', 'n_rows', 'n_matches'])
    if render:
        paths['png'] = out / 'metric_intervals.png'
        _png(summary, paths['png'])
    return {key: str(path) for key, path in paths.items()}
