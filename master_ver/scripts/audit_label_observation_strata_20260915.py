"""Independent descriptive audit on frozen labels; no training or raw data upload."""
from pathlib import Path
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'outputs/full_corpus_training_20260915'
COHORT=ROOT/'outputs/cohort_role_training_20260915/cohorts'
OUT=ROOT/'outputs/label_observation_audit_20260915'
OUT.mkdir(exist_ok=True)
result={'scope':'Saved-label diagnostics only; row rates and equal-match rates explicitly separated; no construct-validity claim','sets':{}}
for path in sorted((PARENT/'labels').glob('*_labels.npz')):
    name=path.name.removesuffix('_labels.npz')
    with np.load(path,allow_pickle=False) as z, np.load(COHORT/f'{name}_cohort.npz',allow_pickle=False) as co:
        assert np.array_equal(z['match'],co['match']) and np.array_equal(z['s'],co['s'])
        valid=z['valid_h90'].astype(bool)
        stale=z['post_snapshot_h90']<=z['L']
        small=abs(z['delta_h90'])<=.01
        sets={}
        for label,c in [('E',np.ones(len(valid),bool)),('T',co['cohort']==1),('N',co['cohort']==0)]:
            for freshness,ff in [('all',np.ones(len(valid),bool)),('no_new_frame_after_L',stale),('new_frame_after_L',~stale)]:
                m=valid&c&ff
                if not m.any():continue
                ids,idx,cnt=np.unique(z['match'][m],return_inverse=True,return_counts=True)
                w=1/cnt[idx]
                def rates(a):return {'row_rate':float(np.mean(a[m])),'equal_match_rate':float(np.average(a[m],weights=w))}
                sets[f'{label}/{freshness}']={'rows':int(m.sum()),'matches':len(ids),'small_delta_le_1pp':rates(small),'no_new_frame_after_L':rates(stale),'positive':rates(z['Y_h90']),
                    'h60_h90_sign_disagreement':rates(z['Y_h60']!=z['Y_h90']),
                    'h90_h120_sign_disagreement':rates(z['Y_h90']!=z['Y_h120']),
                    'duration_mean_s':float(np.mean((z['endpoint_h90'][m]-z['L'][m])/1000))}
        result['sets'][name]=sets
(OUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
lines=['# 한타 규모별 관측 공백 감사','', '기존 동결 라벨에서 재계산한 h90 기술 통계. 아래 비율은 행 기준이며 JSON에는 각 셀 내부 경기별 동일 가중치 비율도 있다. 소규모 Δ는 V의 불확실성 구간이 아니며, 관측 공백과의 연관성은 인과관계가 아니다.','', '| 세트 | 코호트 | 행 | L 이후 새 프레임 없음 | abs(Δ)≤1%p | 새 프레임 없는 행의 작은 Δ | 새 프레임 있는 행의 작은 Δ |','|---|---|---:|---:|---:|---:|---:|']
for name,cells in result['sets'].items():
    for c in ('T','N'):
        a=cells[f'{c}/all']; b=cells[f'{c}/no_new_frame_after_L']; d=cells[f'{c}/new_frame_after_L']
        lines.append(f"| {name} | {c} | {a['rows']:,} | {100*a['no_new_frame_after_L']['row_rate']:.2f}% | {100*a['small_delta_le_1pp']['row_rate']:.2f}% | {100*b['small_delta_le_1pp']['row_rate']:.2f}% | {100*d['small_delta_le_1pp']['row_rate']:.2f}% |")
lines+=['','현재 모델에 의한 측정값의 분포를 확인한 결과다. 실제 한타 승리 정답, 실제 금전 변화 또는 추가 V 비교의 결과로 해석하지 않는다. 인간 검토와 대체 V 검증은 별도 단계다.']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
print('\n'.join(lines))
