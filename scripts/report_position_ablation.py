from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,brier_score_loss,log_loss
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/position_ablation_20260914'
def main():
    r=json.loads((OUT/'results.json').read_text());p=pd.read_csv(OUT/'predictions.csv',dtype={'patch':str});names=json.loads((OUT/'feature_names.json').read_text())
    checks=dict(duplicate_rows=int(p.duplicated(['match','s']).sum()),multi_patch_matches=int((p.groupby('match').patch.nunique()>1).sum()),common_base_features=names['baseline']==names['position'][:len(names['baseline'])],age_in_both='snapshot_age_s' in names['baseline'],future_snapshot_violations=r['future_snapshot_violations'])
    test=p[p.patch=='15.16'].copy();test['age_s']=(test.s-1-test.snapshot)/1000;strata=[]
    for lo,hi in [(0,15),(15,30),(30,45),(45,61)]:
        a=test[(test.age_s>=lo)&(test.age_s<hi)]
        if len(a)==0 or a.y.nunique()<2:continue
        counts=a.groupby('match').match.transform('size');w=1/counts
        strata.append(dict(age=f'{lo}–{hi}',n=len(a),baseline_auc=float(roc_auc_score(a.y,a.baseline,sample_weight=w)),position_auc=float(roc_auc_score(a.y,a.position,sample_weight=w))))
    lines=['# 最後 관측 위치 추가: 교전 예측의 간단한 대조 실험'.replace('最後','마지막'),'', '2026-09-14. B+90초 라벨을 고정하고 같은 교전·분할·관측 나이·학습 설정에서 실제 마지막 관측 좌표20열만 추가했다. 위치 추정값이나 미래 첫 킬 중심은 넣지 않았다.','',
    f"총 {r['rows']:,}교전. 기준 {r['features']['baseline']}열, 위치 추가 {r['features']['position']}열. 기존StateBuilder 기반의 간단한 모델이며 기존7116열 원고 모델을 재현한 실험은 아니다. 패치15.14학습/15.15개발 확인/15.16탐색 평가. 세 seed(7,42,123),조건별 고정250trees. 튜닝과 early stopping은 하지 않았다.",'',
    '## 분할','', '| 패치 | 경기 수 | 교전 수 |','|---|---:|---:|']
    for k,v in r['splits'].items():lines.append(f"| {k} | {v['matches']:,} | {v['rows']:,} |")
    lines+=['','## 15.16 평가','', '| 조건 | seed | AUC | Brier | Log loss |','|---|---|---:|---:|---:|']
    for a in r['results']:
        if a['split']=='test_1516':lines.append(f"| {a['condition']} | {a['seed']} | {a['auc']:.5f} | {a['brier']:.5f} | {a['logloss']:.5f} |")
    lines+=['','## 3seed평균 확률의 위치 추가−기준 차이','', '| 지표 | 차이 | 경기 bootstrap95% CI |','|---|---:|---:|']
    for k,a in r['paired_ensemble_test'].items():lines.append(f"| {k} | {a['difference']:+.5f} | {a['ci95'][0]:+.5f}–{a['ci95'][1]:+.5f} |")
    lines+=['','AUC는 증가, Brier/logloss는 감소가 개선이다. 경기별 총가중치 동일; CI는 평가 경기bootstrap500회(seed7), 고정 학습 모델 조건이며 전체 재학습 불확실성은 아니다.','',
    '## 관측 나이별 보조 AUC','', '| 관측 나이 초 | 교전 수 | 기준 | 위치 추가 |','|---|---:|---:|---:|']
    for a in strata:lines.append(f"| {a['age']} | {a['n']:,} | {a['baseline_auc']:.4f} | {a['position_auc']:.4f} |")
    lines+=['','이 보조 계층은 사후 진단이며 최적 입력 선택이나 유의성 주장의 근거로 단독 사용하지 않는다.','',
    '## 범위와 검증','', '- 15.16은 이전 연구에서 이미 활용했으므로 새 외부 검증이 아니다.', '- 위치의 일반적 유용성, 모든 학습기의 성능, 코칭 효과 또는 인과효과를 입증하는 결과가 아니다.', '- 고정 역할 슬롯의 사전 가용성 가정은 두 조건 공통이며 이번 코드 검사만으로 입증되지 않는다.', '- 전체참가자10명 사용. 위치는s−1ms 이하 프레임에서 직접 읽고 관측 나이를 두 조건에 공통 포함했다.', '- V 학습·보정·선택 경기 교집합0 확인. 추가된20열 외 기준 특징 순서 동일.','', '```json',json.dumps(checks,indent=2),'```','',
    '## 재현','', '- [실행 전 명세](POSITION_ABLATION_PROTOCOL_20260914.md)','- [학습/평가 코드](../scripts/ablate_observed_position.py)','- [보고 코드](../scripts/report_position_ablation.py)','- [전체 결과](../outputs/position_ablation_20260914/results.json)','- [입력 열 목록](../outputs/position_ablation_20260914/feature_names.json)','- [평균 확률 예측](../outputs/position_ablation_20260914/predictions.csv)']
    (ROOT/'docs/POSITION_ABLATION_RESULTS_20260914.md').write_text('\n'.join(lines),encoding='utf-8');(OUT/'verification.json').write_text(json.dumps(dict(checks=checks,age_strata=strata),indent=2));print(json.dumps(dict(checks=checks,age_strata=strata)))
if __name__=='__main__':main()
