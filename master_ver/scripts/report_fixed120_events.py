from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/fixed120_events_20260914'
def main():
    r=json.loads((OUT/'results.json').read_text());d=pd.read_csv(OUT/'engagements.csv');ev=pd.read_csv(OUT/'event_responses.csv')
    clean=ev[(ev.valid==1)&(ev.frame_changed==0)&(ev.same_timestamp_events==1)]
    rng=np.random.default_rng(7);cis=[]
    for cat,a in clean.groupby('category'):
        g=a.groupby('match').response.agg(['sum','count']);s=g['sum'].to_numpy();n=g['count'].to_numpy();boot=[]
        for _ in range(500):
            ix=rng.integers(0,len(g),len(g));boot.append(s[ix].sum()/n[ix].sum()*100)
        cis.append(dict(category=cat,ci95=np.quantile(boot,[.025,.975]).tolist()))
    labels=pd.read_csv(ROOT/'outputs/window_validation_20260914/labels.csv');base=labels[(labels.h==120)&(labels.valid==1)]
    check=dict(n=len(d),duplicate_rows=int(d.duplicated(['match','s']).sum()),endpoint_after_120=int((d.endpoint>d.L+120000).sum()),row_key_difference=len(set(zip(d.match,d.s))^set(zip(base.match,base.s))),ambiguous_anchor_rows=int(d.anchor_ambiguous.sum()),unknown_position_events=int(d.unknown_position_objectives.sum()),invalid_response_rows=int((ev.valid==0).sum()),mixed_timestamp_rows=int((ev.same_timestamp_events!=1).sum()),frame_change_response_rows=int((ev.frame_changed==1).sum()))
    names={'all':'전체','local_objective':'주변 오브젝트 획득 있음','remote_only_objective':'원거리 오브젝트만 있음','no_objective':'오브젝트 획득 없음','additional_kills':'추가 킬 있음','no_additional_kills':'추가 킬 없음','frame_changed':'분 프레임 갱신 있음','no_frame_change':'분 프레임 갱신 없음','no_tracked_events':'추적한 킬/오브젝트/구조물 사건 없음'}
    lines=['# 120초 고정 후속 구간: 사건 연관 및 가치 모델 반응','', '2026-09-14. 사용자 승인에 따라 최대 120초를 운영 기준으로 고정한 두 번째 실험. 최적 윈도우 입증은 아니다. 다음 탐지 교전/경기 종료 직전 제한을 유지한다.','',f"독립 engagement 파티션의 {r['matches']:,}경기, 이전 실험에서 유효했던 {len(d):,}교전을 분석했다. 이 실험의 라벨 반전은 **마지막 킬 평가와 후속120초 평가 사이의 반전**이며 이전의60→120초 비교와 다르다.",'','| 조건 | 교전 수 | 마지막 킬→후속 구간 라벨 반전율 |','|---|---:|---:|']
    for a in r['groups']:lines.append(f"| {names[a['group']]} | {a['n']:,} | {a['flip_rate']*100:.2f}% |")
    lines+=['','조건별 행은 중복된다. 같은 경기/교전 맥락이 통제된 인과 비교가 아니다. 오브젝트가 없는 사례에도 프레임 갱신·시간 경과·그 외 상태 변화가 있다. 원거리 사건은 공간적 분류이지 무관함의 정답이 아니다.', '',
    '## 오브젝트 시각 직전/직후의 고정 모델 응답','',
    f"서로 다른 획득 연결 {r['events']:,}개 중 프레임 갱신·동일 timestamp 복수 가치 사건·무효 상태를 제외한 {r['isolated_events']:,}개를 요약했다. 획득 팀 기준 V(t)−V(t−1ms)이며 인과효과가 아니다. 다른 경기 상태와 모델 계수에 조건부인 반응으로, 고정 가중치로 재사용하지 않는다. 같은 timestamp 영혼 부여가 있으면 주 요약에서 제외되므로 아래 일반 드래곤 수치는 영혼 결합 효과를 대표하지 않는다.",'',
    '| 유형 | 사건 수 | 평균 응답 %p | 평균 95% CI | 중앙값 %p | 음의 응답 비율 |','|---|---:|---:|---:|---:|---:|']
    for a in r['event_response']:
        ci=next(c['ci95'] for c in cis if c['category']==a['category'])
        lines.append(f"| {a['category']} | {a['n']:,} | {a['mean_pp']:.2f} | {ci[0]:.2f}–{ci[1]:.2f} | {a['median_pp']:.2f} | {a['negative_pct']:.2f}% |")
    lines+=['','CI는 해당 유형 획득이 관측된 경기 단위 bootstrap 500회(seed7)이다. 기존 모델을 고정한 조건부 평균의 표집 변동이며 모델 학습 불확실성은 포함하지 않는다. 장로 표본은 작다.','',
    '## 판정','',
    '- 120초 후속 구간에는 상당수 추가 킬이 포함된다. 다음 *적격 탐지 교전*에서 끊는 규칙이 모든 새로운 전투를 제거하지 못한다. 기존 참여 게이트를 통과하지 않는 싸움도 있을 수 있다.',
    '- 오브젝트 획득이 없는 구간에서도 라벨 반전이 발생한다. 후속 Δp를 오브젝트만의 효과로 설명할 수 없다.',
    '- 고정 가치 모델은 주요 오브젝트 획득에 반응한다. 그러나 이것만으로 확률 보정 또는 실제 전략적 가치의 정확성을 입증하지는 않는다.',
    '- 유충 음의 반응은 무조건 수정할 오류로 단정하지 않는다. 스택·시간·팀별 조건과 모델 계수를 추가로 검사할 감사 항목이다.',
    '- 연구 명칭은 교전 연관 후속 상태의 가치 개선으로 유지한다. 순수 교전 인과효과라고 부르지 않는다. 추가 킬을 무조건 첫 시점에서 끊으면 추격전 이득도 제거할 수 있으므로 이 역시 별도 운영 기준 비교가 필요하다.',
    '- 다음 실험 후보: 추가 킬 없는 집합/전체의 라벨·예측 비교, 유충 및 드래곤 조건별 V 응답과 확률 보정 검증. 현재 주 예측기 학습·SHAP 완료를 의미하지 않는다.','',
    '## 검증','', '```json',json.dumps(check,indent=2),'```','',
    '첫 킬 timestamp가 중복되어 앵커를 단일하게 복원할 수 없는 행은 원근 분류에서 불명으로 처리했다. 이 때문에 범주별 수치가 원래 앵커를 저장한 탐지기 전체 집계와 약간 다를 수 있다.','',
    '## 재현','', '- [실행 전 명세](FIXED120_EVENT_PROTOCOL_20260914.md)','- [실험 코드](../scripts/analyze_fixed120_events.py)','- [보고 코드](../scripts/report_fixed120_events.py)','- [원시 집계](../outputs/fixed120_events_20260914/results.json)','- [교전별 사건](../outputs/fixed120_events_20260914/engagements.csv)','- [사건별 모델 응답](../outputs/fixed120_events_20260914/event_responses.csv)']
    (ROOT/'docs/FIXED120_EVENT_RESULTS_20260914.md').write_text('\n'.join(lines),encoding='utf-8')
    (OUT/'verification.json').write_text(json.dumps(dict(checks=check,conditional_mean_ci=cis,hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'scripts/analyze_fixed120_events.py',ROOT/'docs/FIXED120_EVENT_PROTOCOL_20260914.md',ROOT/'worktrees/engagement-state-value/outputs/temporal_winprob_v3_buckets/expanded_model.joblib']}),indent=2))
    print(json.dumps(check));print(json.dumps(cis))
if __name__=='__main__':main()
