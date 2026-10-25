"""Aggregate-only comparison JSON/CSV/SVG; labels never disguise N2 as EPT."""
from __future__ import annotations

import csv
from html import escape
import json
from pathlib import Path

from .protocol import pilot_decision


def rows(pairs):
    for pair,summary in pairs.items():
        labels=summary['model_labels']
        for kind in ('endpoints','nll'):
            for entry in summary[kind]:
                for metric,values in entry['metrics'].items():
                    for internal in ('ept','b1','delta'):
                        value=values[internal]
                        yield {'pair':pair,'kind':kind,'name':entry['name'],'metric':metric,
                               'model':labels.get(internal,'difference'),
                               'estimate':value['estimate'],'ci95':value['ci95'],
                               'layer':entry.get('layer'),'horizon_s':entry.get('horizon_s'),
                               'status':values['ci_status'],'reason':values['reason']}
        structural=summary['structural_normalized_nll']
        if structural['metrics'] is not None:
            for internal in ('ept','b1','delta'):
                value=structural['metrics'][internal]
                yield {'pair':pair,'kind':'structural','name':'common8','metric':'normalized_nll',
                       'model':labels.get(internal,'difference'),'estimate':value['estimate'],
                       'ci95':value['ci95'],'layer':'M2 masked + M3/M4 open',
                       'horizon_s':None,'status':structural['metrics']['ci_status'],
                       'reason':structural['reason']}


def write_report(pairs, output, *, provenance):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    flat=list(rows(pairs))
    decisions={}
    for pair in ('ept_vs_b1','ept_vs_n2'):
        m=pairs[pair]['structural_normalized_nll']['metrics']
        if m is None or any(m[k]['estimate'] is None for k in ('ept','b1')):
            decisions[pair]={'status':'unavailable','inference':'diagnostic_only','confirmed_success':False}
        else:
            decisions[pair]=pilot_decision(m['ept']['estimate'],m['b1']['estimate'])
    record={'schema':'ept-matched25-report-v1','pairs':pairs,'provenance':provenance,
            'decisions':decisions,'interpretation':'diagnostic_only',
            'claims':'Single neural seed and fixed B1 seed1; no confirmatory architecture claim.'}
    (output/'comparison.json').write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    columns=('pair','kind','name','metric','model','estimate','ci95','layer','horizon_s','status','reason')
    with (output/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(flat)
    lines=['# EPT·B1·N2 matched25 비교','',
           '단일 seed 진단이다. 같은25% predictor fit과 공유 whole-FIT 고정 assets 조건부 비교이며 최종 논문 성공 판정이 아니다.','',
           '| 비교 | 공통8-head 정규화 NLL 차이(EPT−대조) | 진단 |','|---|---:|---|']
    for key,d in decisions.items():
        lines.append(f"| {key} | {d.get('difference','unavailable')} | {d['status']} |")
    lines += ['', '경기 가중 pooled AUC·Brier/조건부 NLL은 CSV/JSON의 각 module/layer/endpoint를 확인한다.',
              '음의 손실 차이는 EPT에 유리하다. 유의성·동등성·인과 효과·최종 수렴을 주장하지 않는다.',
              '학습 loss와 STOP 평가의 정의를 구분하고, missing/unavailable을0으로 바꾸지 않는다.']
    (output/'REPORT_KO.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    points=[r for r in flat if r['kind']=='structural' and r['model']!='difference']
    height=80+28*len(points)
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{height}" viewBox="0 0 1000 {height}">',
         '<rect width="100%" height="100%" fill="white"/>',
         '<text x="20" y="26" font-family="sans-serif" font-size="17">Matched25 common-head normalized NLL — single-seed diagnostic</text>']
    finite=[float(r['estimate']) for r in points if r['estimate'] is not None]
    lower=min(finite+[0.]);upper=max(finite+[0.])
    span=max(upper-lower,1.e-12)
    zero=340+450*(0.-lower)/span
    svg.append(f'<path d="M {zero:.3f},40 V {height-10}" stroke="#555"/>')
    for i,r in enumerate(points):
        y=60+i*28;v=r['estimate'];label=f"{r['pair']} / {r['model']}"
        svg.append(f'<text x="20" y="{y}" font-family="sans-serif" font-size="12">{escape(label)}</text>')
        if v is not None:
            end=340+450*(float(v)-lower)/span
            svg.append(f'<rect x="{min(zero,end):.3f}" y="{y-13}" width="{abs(end-zero):.3f}" height="16" fill="#276b8e"/>')
        value='unavailable' if v is None else f'{float(v):.6g}'
        svg.append(f'<text x="820" y="{y}" font-family="sans-serif" font-size="12">{value}</text>')
    svg.append('</svg>')
    (output/'comparison.svg').write_text('\n'.join(svg)+'\n',encoding='utf-8')
    return record
