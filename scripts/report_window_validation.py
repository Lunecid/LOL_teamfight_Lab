from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs/window_validation_20260914'
def main():
    r=json.loads((OUT/'results.json').read_text());d=pd.read_csv(OUT/'labels.csv')
    p=d.pivot(index=['match','s'],columns='h',values='expanded');delta=p[[0,30,60,90,120,180]].subtract(p[-1],axis=0)
    ep=d.pivot(index=['match','s'],columns='h',values='endpoint')
    lasts=d.drop_duplicates(['match','s']).set_index(['match','s']).L
    rng=np.random.default_rng(7);checks=[]
    for a,b in [(60,90),(90,120),(120,180),(60,120)]:
        for subset in ['paired_valid','full_horizon_only']:
            good=delta[[a,b]].dropna()
            if subset=='full_horizon_only':
                good=good[(ep.loc[good.index,b]-lasts.loc[good.index])==b*1000]
            flip=((good[a]>0)!=(good[b]>0)).astype(int)
            per=flip.groupby(level='match').agg(['sum','count']);s=per['sum'].to_numpy();n=per['count'].to_numpy()
            boots=[]
            for _ in range(500):
                ix=rng.integers(0,len(per),len(per));boots.append(s[ix].sum()/n[ix].sum())
            checks.append(dict(a=a,b=b,subset=subset,n=len(good),flip=float(flip.mean()),ci95=np.quantile(boots,[.025,.975]).tolist(),same_endpoint=float((ep.loc[good.index,a]==ep.loc[good.index,b]).mean())))
    lines=['# 윈도우 및 가치 라벨 정당화 실험 결과','', '2026-09-14. 결과를 보기 전에 WINDOW_VALIDATION_PROTOCOL_20260914.md에 비교를 기록한 탐색 실험이다. 120초를 정당화하기 위해 유리한 결과만 선택하지 않는다.','',
    f"가치 모델 독립 engagement 파티션 {r['target_matches']:,}경기 중 기존 탐지 교전이 있는 {r['processed_matches']:,}경기/{r['engagements']:,}교전을 재평가했다. fit/calibrate/select/test와 경기 교집합 모두 0. 처리 오류 {len(r['errors'])}, 미래 snapshot 위반 {r['future_snapshot_violations']}. 교전 없는 대상 경기는 이 라벨 실험에 행을 만들지 않는다.", '',
    '## 전체 코퍼스: 획득 포함과 관찰 단축','', '| h 초 | 공간 조건 내 첫 획득 교전 | 전장 전체 첫 획득 교전 | h까지 완전 추적 불가 비율 |','|---|---:|---:|---:|']
    for a in r['full_corpus']:
        lines.append(f"| {a['h']} | {a['spatial_before_next']:,} | {a['before_next']:,} | {(1-a['full_followup']/a['total'])*100:.2f}% |")
    lines+=['','전장 전체와 공간 조건의 차이는 원거리 사건 연결의 대리 진단이다. 실제 무관한 사건의 정답이 없으므로 인과적 혼입률로 부르지 않는다.','', '## Expanded 가치 모델: 라벨 반전','', '| 비교 초 | 집합 | n | 반전율 | 경기 bootstrap 95% CI | 동일 endpoint 비율 |','|---|---|---:|---:|---:|---:|']
    for a in checks:
        lines.append(f"| {a['a']}→{a['b']} | {a['subset']} | {a['n']:,} | {a['flip']*100:.2f}% | {a['ci95'][0]*100:.2f}–{a['ci95'][1]*100:.2f}% | {a['same_endpoint']*100:.2f}% |")
    lines+=['','paired_valid는 양쪽 평가가 가능한 동일 교전이고 full_horizon_only는 긴 후보 시간까지 실제로 추적된 사례다. 후자는 생존/다음 교전 조건에 의해 선택된 하위집합이므로 일반 전체 결과와 구분한다. 신뢰구간은 경기 단위 bootstrap 500회, seed7이다.','', '## 두 가치 모델의 방향 일치','', '| h 초 | 공통 n | Expanded / Maymin 계열 방향 일치율 |','|---|---:|---:|']
    for a in r['model_agreement']:lines.append(f"| {a['h']} | {a['n']:,} | {a['agreement']*100:.2f}% |")
    lines+=['','두 모델은 이미 저장된 구현을 고정해 사용했다. 원 논문 모델의 새 재현 또는 최적화 완료를 주장하지 않는다. 모델 간 일치는 실제 교전 정답에 대한 정확도가 아니다.','',
    '## 해석과 남은 검증','',
    '- 더 긴 시간은 획득을 더 포함하지만 관찰 단축과 표적 변화도 증가시킨다. 포함률만으로 최적 h를 정할 수 없다.',
    '- 120초는 계속 후보로 둔다. 반전이 작아야 한다는 사전 승인 기준도 없으므로 이번 결과만으로 최종 채택을 선언하지 않는다.',
    '- 실제 오브젝트 귀속은 독립적인 전문가 판정 또는 별도 관측 근거가 필요하다. 시간/공간 필터만으로 인과 귀속을 확정할 수 없다.',
    '- V가 달라질 때 라벨이 달라지는 정도를 공개해야 한다. 보정·희소 오브젝트 조건별 타당성 검증은 별도로 필요하다.',
    '- 미래 패치가 새로 확보되지 않았으므로 외부 일반화 검증 완료가 아니다. 새 q 학습 및 SHAP의 최종 타당화도 이 실험의 결과가 아니다.','',
    '## 재현','', '- [실행 전 명세](WINDOW_VALIDATION_PROTOCOL_20260914.md)','- [실험 코드](../scripts/validate_windows.py)','- [보고 코드](../scripts/report_window_validation.py)','- [결과 및 소스·모델 해시](../outputs/window_validation_20260914/results.json)','- [각 시점 확률·관측 시각](../outputs/window_validation_20260914/labels.csv)']
    (OUT/'paired_bootstrap.json').write_text(json.dumps(checks,indent=2))
    (ROOT/'docs/WINDOW_VALIDATION_RESULTS_20260914.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(checks,indent=2));print(json.dumps(r['model_agreement']))
if __name__=='__main__':main()
