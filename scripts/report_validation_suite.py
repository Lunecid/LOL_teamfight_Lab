from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/validation_suite_20260914'
def read(n):return json.loads((OUT/n).read_text(encoding='utf-8'))
def metricrow(name,a):return f"| {name} | {a['auc']:.5f} | {a['brier']:.5f} | {a['logloss']:.5f} |"
def main():
    q=read('q_results.json');v=read('v_results.json');label=read('label_results.json');ext=read('external_full/results.json');pilot=read('external/results.json');ab=read('group_ablation.json');inc=read('incremental_baselines.json')
    test={a['model']:a for a in q['metrics'] if a['split']=='test_exploratory'}
    vr=next(a for a in v['results'] if a.get('model')=='expanded' and a['stratum']=='original_one_per_match')
    fig,axs=plt.subplots(1,2,figsize=(10,4.3))
    for ax in axs:ax.plot([0,1],[0,1],'--',color='gray',linewidth=1);ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted probability',ylabel='Observed fraction');ax.grid(alpha=.15)
    for key,name in [('full_raw','Before: full tree'),('full_sigmoid','Tree + sigmoid'),('ridge_sigmoid','Selected: ridge + sigmoid')]:
        bins=test[key]['bins'];axs[0].plot([b['predicted'] for b in bins],[b['observed'] for b in bins],marker='o',label=name,markersize=4)
        for b in bins:
            if b['n']<50:axs[0].annotate('n='+str(b['n']),(b['predicted'],b['observed']),xytext=(3,7),textcoords='offset points',fontsize=7)
    axs[0].set_title('q: patch 15.16 (exploratory)');axs[0].legend(fontsize=8)
    for key,name in [('frozen','Frozen transfer'),('adapted','Early-period sigmoid')]:
        bins=ext['q'][key]['bins'];axs[1].plot([b['predicted'] for b in bins],[b['observed'] for b in bins],marker='o',label=name,markersize=4)
        for b in bins:
            if b['n']<50:axs[1].annotate('n='+str(b['n']),(b['predicted'],b['observed']),xytext=(3,7),textcoords='offset points',fontsize=7)
    axs[1].set_title('q: 26.13 late-period evaluation');axs[1].legend(fontsize=8)
    fig.suptitle('Calibration diagnostics: fixed 0.1 bins; sparse bins may be unstable',fontsize=10);fig.tight_layout();fig.savefig(OUT/'calibration_comparison.png',dpi=180);fig.savefig(OUT/'calibration_comparison.pdf');plt.close(fig)
    shap=pd.read_csv(OUT/'selected_model_shap.csv',dtype={'patch':str});a=shap[shap.patch=='15.16'].nlargest(12,'mean_abs_shap').sort_values('mean_abs_shap')
    fig,ax=plt.subplots(figsize=(8,5));ax.barh(a.feature,a.mean_abs_shap,color='#27658d');ax.set(xlabel='Mean absolute contribution (calibrated log-odds)',title='Selected ridge model: exploratory patch 15.16');fig.tight_layout();fig.savefig(OUT/'selected_shap.png',dpi=180);fig.savefig(OUT/'selected_shap.pdf');plt.close(fig)
    lines=['# 통합 검증 결과와 진행 전후 기록','', '2026-09-14. 검토 판정: **논문 주장은 보완 필요(Needs revision)**. 수치 실험·재현·설명 자료는 완료했으나 독립 사람 검토, 실제 시점의 역할 정보 가용성, 관객/코칭 효용은 미검증이다.','',
    '## 전후 상황','', '| 항목 | 진행 전 | 진행 후 |','|---|---|---|',
    '| q 최소 기준선 | 위치 유무만 비교, 확률 오차가 상수보다 큼 | 상수·사전 승률·경제·전체 특징 및 모델/보정 대조 완료 |',
    '| V 품질 | 기존 전체 평가 존재 | 원래 4,986경기 평가 수치 재현; 시간·오브젝트·용 원소/영혼 이력별 보정과 경기 bootstrap Brier CI 추가 |',
    '| 교전 라벨 | B+90 작업 기준, 60/120 민감도 존재 | B90에서 대안 V, 작은 변화, 골드·킬 방향 불일치 및 사례 120건 기록 |',
    '| 패치 이전 | 15.16 재사용 탐색 평가 | 2026 예비 100경기 이후 나머지 9,964경기 처리, 시간 분리 적응/평가 완료 |',
    '| SHAP | 새 q 설명 미완료 | 기존 TreeSHAP과 선택 보정 선형모델의 기여 계산, 합계 검증, 특징군 제거 완료 |',
    '| 사람 검증 | 미실시 | 모델 출력 가림 검토 자료 준비; 실제 평가 미실시 |','',
    '## 고정 계약과 수행 중 수정','',
    '- 기존 kill-conditioned localized engagement 모집단을 유지했다. 추가 킬 직전 중단 B+90 라벨, 60/120 민감도를 유지했다. V 및 원래 라벨을 교체하지 않았다.',
    '- 최초 보정/선택 공용 개발 표본 실험은 `q_results_pilot_shared_development.json`에 보존했다. 방법상 낙관성을 줄이기 위해 15.15를 보정 1,669경기/4,879교전과 선택 1,656경기/4,922교전으로 분리했다. 15.14만 기저 모델 학습에 사용했다.',
    '- 후보 선택은 분리된 개발 표본의 Brier, 동률 log loss 순서다. 선택 모델은 ridge_sigmoid다. 15.16을 이미 관찰한 뒤의 추가 분석이므로 전향 사전등록/미사용 시험으로 부르지 않는다.',
    '- 후속 시즌 표본은 원래 100경기 예비 평가를 제외한 complete 26.13 전수다. 생성 시간순 앞 30%에서만 sigmoid 적응, 뒤 70% 평가. 경계 시점 이전에 종료되지 않았을 수 있는 적응 경기는 보수적으로 제외했다. q의 정답은 적응 전 V로 고정했다.',
    '- 원본 자료나 Claude 작업 저장소를 수정하지 않았다. 원본 raw 파일의 SHA-256을 수집 DB/기존 표본 명세와 대조했다. 외부 캐시는 별도 산출물에 생성했다. 초기 UTF-8 디코딩 오류는 명시적 인코딩으로 수정한 뒤 완료했다.','',
    '## 15.16: 같은 7,664교전 / 2,655경기에서 비교','', '| 모델 | AUC ↑ | Brier ↓ | Log loss ↓ |','|---|---:|---:|---:|']
    for key in ['constant','full_raw','position_raw','p_pre_logistic','p_pre_spline','economic_raw','economic_sigmoid','full_ridge','full_sigmoid','position_sigmoid','ridge_sigmoid']:lines.append(metricrow(key,test[key]))
    lines+=['', '모델 설정: 기저 트리는 기존 250 trees/15 leaves/학습률 .04/최소 leaf 100/seed 7,42,123 평균. 경제 대조는 같은 설정의 입력 축소. Ridge는 champion_id를 제외한 수치 특징, 표준화와 L2 로지스틱 C=.01. 모든 q 학습 손실·평가는 경기별 총가중치 동일. 사전 승률 기준선은 그 값으로 Y를 학습하는 별도 분류기다.','',
    '| 선택 모델 − 기준 | 지표 | 차이 | 경기 bootstrap 95% CI |','|---|---|---:|---|']
    for pair in q['paired_bootstrap'][:2]:
        for k,x in pair['delta'].items():lines.append(f"| {pair['a']} − {pair['b']} | {k} | {x['estimate']:+.5f} | {x['ci95'][0]:+.5f} ~ {x['ci95'][1]:+.5f} |")
    for key,x in inc.items():
        a=x['auc'];lines.append(f"| ridge_sigmoid − {key} | AUC | {a['estimate']:+.5f} | {a['ci95'][0]:+.5f} ~ {a['ci95'][1]:+.5f} |")
    lines+=['', '500회 경기 bootstrap, 학습/선택 모델을 고정한 조건부 불확실성이다. 전체 재학습·모델 선택·가치 라벨 불확실성을 포함하지 않는다. 보정 절편/기울기는 평가 진단으로 계산했으며 평가 자료에서 재보정한 확률을 성능으로 보고하지 않았다.',
    f"기존 q의 평가 보정 기울기는 {test['full_raw']['slope']:.3f}, 선택 모델은 {test['ridge_sigmoid']['slope']:.3f}다. 확률의 과도한 극단성이 줄었으나 완전한 보정은 아니다. Brier는 보정만의 지표가 아니며 판별과 함께 해석한다.",'',
    '## 후속 시즌 26.13 이전 및 제한적 보정','',
    f"완료 캐시 10,064경기 중 예비 100경기를 제외한 {ext['total_matches']:,}경기를 처리했다. 유효 교전 {ext['valid_engagements']:,}개, 제외 교전 {ext['invalid_engagements']:,}개다. q 적응 {ext['q_adaptation_matches']:,}경기, 후기 평가 {ext['q_evaluation_matches']:,}경기/{ext['q_evaluation_rows']:,}교전이다. 경계 중첩 가능 적응 제외 {len(ext['purged_adaptation_matches'])}경기. V는 교전 여부와 독립적으로 후기 {ext['v_evaluation_matches']:,}경기에서 경기당 한 시점을 평가했다.",'',
    '| 후기 평가 q | AUC ↑ | Brier ↓ | Log loss ↓ |','|---|---:|---:|---:|']
    for k in ['constant_2025','constant_early2026','frozen','adapted']:lines.append(metricrow(k,ext['q'][k]))
    lines+=['', '| 비교 | 지표 | 차이 | 경기 bootstrap 95% CI |','|---|---|---:|---|']
    for key,vals in ext['q_paired'].items():
        for k,a in vals.items():lines.append(f"| {key} | {k} | {a['estimate']:+.5f} | {a['ci95'][0]:+.5f} ~ {a['ci95'][1]:+.5f} |")
    lines+=['', '무수정과 보정 결과를 모두 보존한다. 추가 보정의 채택 여부는 후기 결과를 보고 새 모델을 선택하는 별도 의사결정이다. 26.13 하나의 후속 시즌 결과만으로 모든 패치·지역·숙련도 일반화를 주장하지 않는다. 예비 평가 이후 확대했다는 이력과 기존 표본의 detector 기술 검토 이력을 공개한다.','',
    '## V 감사','',f"원래 경기당 1시점 4,986경기 평가를 재현했다: expanded AUC {vr['auc']:.5f}, Brier {vr['brier']:.5f}, log loss {vr['logloss']:.5f}; 보정 절편 {vr['intercept']:.3f}, 기울기 {vr['slope']:.3f}. 원래 값과 최대 수치 차이는 1e-10 미만이다.",'',
    '| 후속 시즌 V | AUC ↑ | Brier ↓ | Log loss ↓ |','|---|---:|---:|---:|']
    for k in ['frozen','adapted']:lines.append(metricrow(k,ext['v'][k]))
    lines+=['', 'V 오브젝트 층화는 해당 시점까지 기록된 획득 이력의 유무다. 영혼 이벤트는 팀 식별 기록(soul_team_recorded)과 미식별 기록(soul_unassigned_event)을 분리했으며 후자는 팀의 영혼 획득이 아니다. 활성 버프 여부, 획득 인과효과, 획득당 고정 승률 상승치가 아니다. 원소 종류별 용 획득과 영혼 이벤트 팀 식별 상태, 표본 수 및 Brier CI는 v_results.json에 모두 보관했다. 희소 장로 이력은 92경기/336시점으로 별도 주의가 필요하다. 경기 단계와 오브젝트 이력의 상관 때문에 층 간 AUC를 오브젝트 효과로 해석하지 않는다.','',
    '## 라벨 의미와 종료 경계','',
    f"B90 공통 26,693교전에서 expanded와 Maymin 가치 모델의 방향 불일치는 {label['value_disagreement']*100:.3f}%다. 전체 교전을 분모로, 골드 차이 변화가 0이 아니면서 ΔV와 방향이 다른 비율은 {label['gold_disagreement']*100:.3f}%, 킬 차이 변화는 {label['kills_disagreement']*100:.3f}%다. 이것은 오라벨률이 아니다.",
    f"|ΔV|≤1%p는 {label['small_delta']['0.01']['n']:,}교전({label['small_delta']['0.01']['fraction']*100:.3f}%)이다. 0.1/0.5/1/2%p를 기술적으로 보고했으며, 분류 성능을 위해 라벨을 제거하거나 중립 구간을 새로 채택하지 않았다.",'',
    'B60→90 반전 1.409%, B90→120 반전 0.618%는 기존 [경계 민감도 실험](B_BOUNDARY_60_90_120_RESULTS_20260914.md)을 재사용한다. B90/120에서 동일 종료점이 많다는 구조적 이유를 유지한다. 새 킬은 반드시 독립된 새 한타라는 정답이 아니며 추격이 잘릴 수 있다.','',
    '## SHAP과 특징군 제거','',
    '- 기존 트리: TreeSHAP을 seed별, 15.15/15.16에서 계산. 척도는 미보정 log-odds. 트리 경로 학습 cover에 따른 기준 분포. 기여 합계와 raw margin 최대 오차 4.22e-15.',
    '- 선택 모델: 15.14 경기 가중 평균을 기준으로 한 선형 interventional Shapley 기여. sigmoid의 선형 logit 변환까지 반영한 **보정 log-odds** 척도. 최대 합계 오차 4.89e-15. 상관 특징을 조건부로 모델링한 설명이 아니다.',
    '- 트리의 개별 특징 중요도 순위 Spearman은 seed/패치 비교에서 약 .879~1.000이다. 높은 순위 안정성이 타당한 인과 설명을 뜻하지 않는다. 선택 모델의 역할별 경험치·레벨·골드 기여는 강하게 상관될 수 있다.',
    '- 아래 특징군 제거는 동일 Ridge 설정 및 분리 보정을 사용한 사후 탐색이다. 주모델 재선택에 사용하지 않았다.','',
    '| 제거 특징군 | AUC | Brier | 전체 대비 AUC 차이 (95% CI) |','|---|---:|---:|---|']
    for a in ab['results']:
        d=a['minus_full']['auc'];lines.append(f"| {a['removed']} | {a['test']['auc']:.5f} | {a['test']['brier']:.5f} | {d['estimate']:+.5f} ({d['ci95'][0]:+.5f} ~ {d['ci95'][1]:+.5f}) |")
    lines+=['', '직접 오브젝트 특징 제거에도 p_pre/골드 등 간접 정보는 남는다. 따라서 해당 대조는 게임에서 오브젝트가 불필요한지 검증하는 실험이 아니다. 개별 SHAP 순위만으로 특정 역할·변수가 한타 승리의 원인이라고 주장하지 않는다.','',
    '## 남은 조건과 리뷰 대응','',
    '- 라벨/경계 의미 검토: 120개 층화 사례와 가림 평가 양식·사건 기록을 생성했다. 실제 독립 평가자 결과가 없으므로 사람 타당도와 평가자 일치도는 미완료다.',
    '- **영혼 스키마 허점:** 원본 DRAGON_SOUL_GIVEN은 `name: Cloud` 등을 사용하지만 StateBuilder는 dragonSoul/soulType만 읽는다. 사전 26,693행에서 원소별 soul 특징은 모두 0, blue_soul_OTHER는 200행, red_soul_OTHER는 409행이다. teamId=0 이벤트도 있어 단순 이벤트 존재를 실제 팀 영혼 획득으로 해석하면 안 된다. 이번 V 감사에서 팀 식별 이벤트와 미식별 이벤트를 분리하도록 수정했다. 동결 V/q에는 원소별 영혼 구분이 제대로 반영되지 않았으며, 코드 스키마 수정 후 새 버전 V/라벨/예측기 재학습이 필요하다. 기존 결과를 새 스키마 결과처럼 쓰지 않는다. [원본 사례와 검사](../outputs/validation_suite_20260914/soul_schema_finding.json).',
    '- 실시간 정보 가용성: StateBuilder의 프레임·사건 cutoff는 검증했지만, 역할 슬롯은 Match-V5 사후 detail의 teamPosition/individualPosition에서 추출한다. 당시 가용 역할과의 일치가 입증되지 않았으므로 엄격한 실시간 입력 계약의 미해결 조건이다. 과거 입력으로 관측한 연구라는 주장은 이 가정을 명시해야 한다.',
    '- 모델 공정성: 이번에는 고정 예산의 단순 기준선을 보강했다. 새 Y에 대한 FT-Transformer/TabNet/SAINT 등 전체 아키텍처 대조나 기존 원고 7,116열 재현을 완료한 것은 아니다.',
    '- 오류·관측 해상도: 이번 사례/위치/SHAP 결과를 보강 근거로 사용할 수 있으나 독립 사람 오류 분류와 더 촘촘한 실제 좌표 정답은 여전히 필요하다.',
    '- 공개/재현: 로컬 코드·분할·모델·예측·해시를 보관했다. 공개 배포/라이선스 검토나 camera-ready 제출을 수행한 것은 아니다.',
    '- 실제 교전 인과효과, 관객 이해도, 코칭 성과는 별도 연구이며 이번 결과로 입증하지 않는다.','',
    '## 검증 결과','', '후속 시즌 전체 유효 교전의 중복 0, 마지막 킬 이후 종료점까지 추가 킬 포함 0, 평가 시작 이후에야 알 수 있는 승패를 적응에 사용한 사례 0을 원본 이벤트로 재검사했다. 후기 frozen/adapted Brier와 log loss를 별도 수식으로 재계산해 1e-12 이내로 일치했다. 보정 그림과 SHAP PNG를 시각적으로 확인했고 희소 확률 구간에는 표본 수를 표시했다. [독립 검사](../outputs/validation_suite_20260914/external_full/independent_checks.json). 이는 역할 메타데이터의 실시간 가용성이나 교전 인과효과 검증은 아니다.','',
    '## 근거 문헌과 인용 범위','',
    '1. Maymin (2021), *Smart kills and worthless deaths*, JQAS. [DOI 10.1515/jqas-2019-0096](https://doi.org/10.1515/jqas-2019-0096). LoL 상태 승리 확률을 통한 행동 가치의 선례. 90초 경계나 우리 예측기의 성능을 보증하지 않는다.',
    '2. Decroos et al. (2019), *Actions Speak Louder than Goals*, KDD. [원문](https://www.janvanhaaren.be/assets/papers/kdd-2019-vaep.pdf). 구성 확률·설계 대조·사례를 분리하고 시간 순서로 평가하는 방법론. 축구 득점/실점과 LoL 최종 승패의 차이를 명시한다.',
    '3. Van Calster et al. (2019), *Calibration: the Achilles heel of predictive analytics*. [원문](https://link.springer.com/article/10.1186/s12916-019-1466-7). 확률 보정 진단과 갱신 근거. 의료 표본 수 기준을 교전에 대입하지 않는다.',
    '4. Jacobs & Wallach (2021), *Measurement and Fairness*. [원문](https://arxiv.org/abs/1912.05511). 운영 측정과 개념 타당도 구분. 우리 라벨의 객관적 정답 근거가 아니다.',
    '5. Lundberg & Lee (2017), *A Unified Approach to Interpreting Model Predictions*. [원문](https://arxiv.org/abs/1705.07874). 모델 예측 기여 설명. 인과효과로 해석하지 않는다.',
    '6. Austin, Lee & Fine (2016), competing risks tutorial. [원문](https://pmc.ncbi.nlm.nih.gov/articles/PMC4741409/). 기존 경계 CIF 분석의 방법 근거. 90초 숫자를 제공하지 않는다.','',
    '## 산출물과 재현','',
    '- [실행 전 명세와 수정 이력](VALIDATION_SUITE_BEFORE_20260914.md)',
    '- [통합 실행 코드](../scripts/run_validation_suite.py), [보조 대조/독립 계산](../scripts/validation_supplements.py)',
    '- [예비 외부 준비](../scripts/prepare_external_validation.py), [예비 평가·선택 모델 설명](../scripts/evaluate_external_and_explain.py)',
    '- [외부 전수 준비](../scripts/prepare_external_full.py), [시간 분리 외부 평가](../scripts/evaluate_external_full.py)',
    '- [q 상세 수치](../outputs/validation_suite_20260914/q_results.json), [V 상세 수치](../outputs/validation_suite_20260914/v_results.json), [후속 시즌 결과](../outputs/validation_suite_20260914/external_full/results.json)',
    '- [라벨 감사](../outputs/validation_suite_20260914/label_results.json), [특징군 제거](../outputs/validation_suite_20260914/group_ablation.json), [가림 양식](../outputs/validation_suite_20260914/review_blinded_form.csv)',
    '- [보정 그림](../outputs/validation_suite_20260914/calibration_comparison.png), [선택 모델 SHAP](../outputs/validation_suite_20260914/selected_shap.png)','',
    'Python: C:/Users/todtj/anaconda3/python.exe. 순서: run_validation_suite.py features → q → v → labels → shap; evaluate_external_and_explain.py는 예비 prepare_external_validation.py 뒤에 실행; validation_supplements.py; prepare_external_full.py → evaluate_external_full.py; report_validation_suite.py. 원본 입력·의존 코드/모델 해시는 provenance.json에 기록한다. 대용량 캐시를 포함한 전체 재실행은 원본 D: 드라이브 자료가 필요하다.']
    (ROOT/'docs/VALIDATION_SUITE_AFTER_20260914.md').write_text('\n'.join(lines),encoding='utf-8')
    # Traceable hashes for analysis code, models and primary inputs (not every derived cache).
    files=list((ROOT/'scripts').glob('*validation*.py'))+[ROOT/'scripts/evaluate_external_and_explain.py',ROOT/'scripts/evaluate_external_full.py',ROOT/'scripts/prepare_external_full.py']
    files+=list(OUT.glob('*.joblib'))+list(OUT.glob('*.json'))+[ROOT/'outputs/position_ablation_20260914/predictions.csv',ROOT/'outputs/b_boundary_60_90_120/dynamic_values.csv',OUT/'external_full/frozen_manifest.json',OUT/'external_full/preparation.json',OUT/'external_full/results.json']
    files+=[ROOT/'worktrees/engagement-state-value/outputs/temporal_winprob_v3_buckets'/n for n in ['expanded_model.joblib','maymin_model.joblib','protocol.json','independent_time_curves.npz']]
    files+=list((ROOT/'outputs/position_ablation_20260914').glob('*.joblib'))
    files+=[ROOT/'worktrees/engagement-state-value'/n for n in ['gameplay/state_value.py','core/roles.py','core/config.py','train/temporal_winprob.py']]
    files=[p for p in set(files) if p.name!='provenance.json']
    (OUT/'provenance.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2),encoding='utf-8')
    print('report complete',flush=True)

if __name__=='__main__':main()
