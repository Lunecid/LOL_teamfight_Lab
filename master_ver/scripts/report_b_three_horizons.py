from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/b_boundary_60_90_120'
def main():
    r=json.loads((OUT/'results.json').read_text());d=pd.read_csv(OUT/'dynamic_values.csv');old=pd.read_csv(ROOT/'outputs/event_boundary_cif_20260914/dynamic_values.csv')
    j=d[d.h.isin([90,120])].merge(old,on=['match','s','h'],suffixes=('_new','_old'),validate='one_to_one')
    checks=dict(r['checks'],previous_endpoint_mismatch=int((j.endpoint_new!=j.endpoint_old).sum()),previous_p_maxerror=float((j.p_post_new-j.p_post_old).abs().max()),duplicate_keys=int(d.duplicated(['match','s','h']).sum()))
    p=d.pivot(index=['match','s'],columns='h',values='p_post');e=d.pivot(index=['match','s'],columns='h',values='endpoint')
    lab=pd.read_csv(ROOT/'outputs/window_validation_20260914/labels.csv');pre=lab[lab.h==-1].set_index(['match','s']).expanded.reindex(p.index)
    first=pd.read_csv(OUT/'first_events.csv');first=first[(first['mode']=='B_next_kill')&(first.anchor_ambiguous==0)]
    rng=np.random.default_rng(7)
    def ci(values):
        g=values.groupby(level=0).agg(['sum','count']);s=g['sum'].to_numpy();n=g['count'].to_numpy();boots=[]
        for _ in range(500):
            ix=rng.integers(0,len(g),len(g));boots.append(100*s[ix].sum()/n[ix].sum())
        return np.quantile(boots,[.025,.975]).tolist()
    pairs=[]
    for a,b in [(60,90),(90,120),(60,120)]:
        valid=p[[a,b]].notna().all(axis=1)&pre.notna();flip=((p.loc[valid,a]>pre[valid])!=(p.loc[valid,b]>pre[valid])).astype(int);same=e.loc[valid,a]==e.loc[valid,b]
        gain=((first.cause=='objective')&(first.time_s>a)&(first.time_s<=b)).astype(int);gain.index=first.match
        pairs.append(dict(a=a,b=b,n=len(flip),flip_count=int(flip.sum()),flip_pct=float(flip.mean()*100),flip_ci95=ci(flip),same_endpoint_pct=float(same.mean()*100),different_endpoint_n=int((~same).sum()),different_endpoint_flip_pct=float(flip[~same].mean()*100),objective_gain_count=int(gain.sum()),objective_gain_pp=float(gain.mean()*100),objective_gain_ci95=ci(gain)))
    out=dict(checks=checks,pairs=pairs);(OUT/'paired_comparison.json').write_text(json.dumps(out,indent=2))
    lines=['# 동일 B 종료 규칙: 최대60·90·120초 비교','', '2026-09-14. 다음 추가 킬/다음 적격 교전/경기 종료 직전에 끊는 규칙은 동일하게 유지하고 최대 상한만 바꿨다. 가변 구간이며 아래 시간은 마지막 킬 이후다.','', '9,198경기26,693교전의 고정 가치 모델 평가. 공간 CIF만 앵커 불명117교전을 제외한26,576교전. 이전90/120초 endpoint와 예측 확률 재현 검사 포함.','', '| 최대 상한 | 종료 전 첫 주변 획득 CIF | 실제 길이 평균 | 중앙값 | 상한까지 도달 | 개선 라벨 비율 |','|---|---:|---:|---:|---:|---:|']
    for h in [60,90,120]:
        s=next(a for a in r['dynamic_labels'] if a['h']==h);f=next(a for a in r['cif'] if a['mode']=='B_next_kill' and a['h']==h)
        lines.append(f"| {h}초 | {f['objective_cif']*100:.2f}% | {s['mean_duration_s']:.2f}초 | {s['median_duration_s']:.2f}초 | {s['full_horizon_pct']:.2f}% | {s['positive']*100:.2f}% |")
    lines+=['','| 비교 | 추가 획득 교전 | CIF 증가 %p (95% CI) | 라벨 반전 % (95% CI) | 동일 endpoint % | endpoint가 다른 행의 반전 % |','|---|---:|---:|---:|---:|---:|']
    for a in pairs:
        c=a['objective_gain_ci95'];f=a['flip_ci95'];lines.append(f"| {a['a']}→{a['b']} | {a['objective_gain_count']:,} | {a['objective_gain_pp']:.3f} ({c[0]:.3f}–{c[1]:.3f}) | {a['flip_pct']:.3f} ({f[0]:.3f}–{f[1]:.3f}) | {a['same_endpoint_pct']:.2f} | {a['different_endpoint_flip_pct']:.2f} |")
    lines+=['','CI는 경기 단위 bootstrap500회(seed7). 고정 V 조건의 표집 불확실성이며 모델 학습 불확실성은 아니다. 전체 교전 가중 통계이며 경기별 동일 가중 통계와 다르다.','',
    '## 해석','', '- 세 상한을 직접 비교한 결과이며90초를 미리 정답으로 가정하지 않는다.', '- 긴 상한은 추가 획득을 포함하지만 다른 평가 상태와 라벨도 만든다. 종료 사건이 먼저 발생한 행은 상한을 늘려도 endpoint가 같다.', '- 추가 킬 미포함은 B의 구성상 성질이다. 추격 킬과 새 전투를 의미적으로 구분했다는 증거가 아니다.', '- 오브젝트 포함률과 라벨 안정성만으로 최적 시간을 선언하지 않는다. 최종 선택에는 유지할 이득의 범위와 별도 의미 타당성 판단이 필요하다.', '- q 예측 모델의 세 라벨별 성능이나 SHAP 결과를 비교한 실험은 아니다. 이 문서는 표적 정의의 민감도 실험이다.','',
    '## 검증','', '```json',json.dumps(checks,indent=2),'```','', '- [사전 명세](B_BOUNDARY_60_90_120_PROTOCOL_20260914.md)','- [실험 코드](../scripts/event_boundary_cif.py)','- 실행: `python scripts/event_boundary_cif.py --out outputs/b_boundary_60_90_120`','- [보고 코드](../scripts/report_b_three_horizons.py)','- [결과](../outputs/b_boundary_60_90_120/results.json)','- [쌍별 비교와 CI](../outputs/b_boundary_60_90_120/paired_comparison.json)']
    (ROOT/'docs/B_BOUNDARY_60_90_120_RESULTS_20260914.md').write_text('\n'.join(lines),encoding='utf-8');print(json.dumps(out));print(json.dumps(r['dynamic_labels']))
if __name__=='__main__':main()
