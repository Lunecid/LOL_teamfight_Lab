"""Verify completed full-corpus scan and write an inspectable Korean report."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/postkill_objective_delay_full'
def main():
    run=json.loads((OUT/'results.json').read_text())
    d=pd.read_csv(OUT/'first_acquisitions.csv',dtype={'patch':str})
    ex=pd.read_csv(OUT/'exposures.csv',dtype={'patch':str})
    joined=d.merge(ex,on=['match','patch','s'],validate='many_to_one')
    checks=dict(run['checks'],next_boundary_violations=int((joined.event_ts>=joined.next_start).sum()),game_end_violations=int((joined.event_ts>joined.end).sum()),delay_max_error=float((joined.delay_s-(joined.event_ts-joined.L)/1000).abs().max()))
    with np.load('D:/LOL_Project/fusion_2615/features/tog_revision/A3-temporal-windows/fight_index.npz',allow_pickle=False) as z:
        old=set(zip(z['match_id'].astype(str),z['t_start_ts'].astype(int),z['last_kill_ts'].astype(int)))
        old_mids=set(z['match_id'].astype(str))
    shared=ex[ex.match.isin(old_mids)]
    now=set(zip(shared.match,shared.s,shared.L))
    checks.update(reference_index_rows=len(old),recomputed_shared_rows=len(now),reference_rows_missing=len(old-now),additional_rows_in_shared_matches=len(now-old))
    strict=d[d['mode']=='spatial_before_next']
    mapping={'any':'전체: 첫 획득','baron':'바론','dragon':'일반 드래곤','elder':'장로','herald':'전령','horde':'유충 첫 획득','atakhan':'아타칸'}
    lines=['# 전체 캐시 마지막 킬 이후 오브젝트 지연 측정','', '2026-09-14. 전체 패치를 포함한 기술 통계이며 종료 유예를 확정하거나 예측 모델을 학습한 결과가 아니다.','',
           f"캐시 {run['attempted_matches']:,}경기를 전부 처리했다. 로드 성공 {run['stats'].get('loaded_matches',0):,}경기, 로드/스키마 거부 {run['stats'].get('unreadable_or_schema_rejected',0):,}경기, 처리 예외 {run['stats'].get('errors',0):,}경기다. 기존 v3.3 탐지 규칙에서 {len(ex):,}교전이 추출되었고 {ex.match.nunique():,}경기에 속한다. 경기별 성공/거부와 패치 수는 결과 JSON을 참조한다.", '',
           '## 60초가 최대 지연이라는 해석은 틀림','',
           '마지막 킬 이후 다음 탐지 교전 시작 또는 경기 종료까지 추적했다. 고정 초 단위 상한을 두지 않았다. 기존 첫 킬 앵커에서 4,264u 이내의 각 오브젝트 유형 첫 획득을 집계한다. 아래의 60초 초과 비율 분모는 **획득이 관찰된 교전**이다. 미획득 교전을 0초로 넣지 않는다. 유형별 표본은 중복될 수 있다.','',
           '| 유형 | 획득 교전 수 | 평균 초 | 중앙값 초 | P90 초 | 60초 초과 | 관측 최대 초 |','|---|---:|---:|---:|---:|---:|---:|']
    for cat,name in mapping.items():
        v=strict[strict.category==cat].delay_s
        lines.append(f'| {name} | {len(v):,} | {v.mean():.1f} | {v.median():.1f} | {v.quantile(.9):.1f} | {(v>60).mean()*100:.1f}% | {v.max():.1f} |')
    lines+=['','위 최대값은 이번 자료와 귀속 규칙에서의 관측 최대이며 게임의 절대 상한이 아니다. 긴 지연은 다음 적격 교전이 드물거나 멀리 이동한 후 획득한 사건을 포함할 수 있다. 그 한타의 인과적 성과라는 뜻도 아니다.','', '## 이전 120초 관찰과 같은 조건의 비교','', '| 유형 | 120초 이내 획득 교전 | 평균 초 | 중앙값 초 | 그중 60초 초과 |','|---|---:|---:|---:|---:|']
    conditional=[]
    for cat,name in mapping.items():
        v=strict[(strict.category==cat)&(strict.delay_s<=120)].delay_s
        rec=dict(category=cat,n=len(v),mean=float(v.mean()),median=float(v.median()),after60=int((v>60).sum()),after60_pct=float((v>60).mean()*100));conditional.append(rec)
        lines.append(f"| {name} | {len(v):,} | {v.mean():.1f} | {v.median():.1f} | {(v>60).mean()*100:.1f}% |")
    anyd=strict[strict.category=='any']
    lines+=['','## 포함률과 미획득/관찰 종료','', '| 유예 후보 초 | 그 시간 이내 첫 획득 교전 | 전체 교전 대비 | 해당 시간까지 완전 추적 가능 |','|---|---:|---:|---:|']
    for h in [30,60,90,120,180,300]:
        n=int((anyd.delay_s<=h).sum());follow=np.minimum(ex.next_start,ex.end)-ex.L
        lines.append(f'| {h} | {n:,} | {n/len(ex)*100:.2f}% | {int((follow>=h*1000).sum()):,} |')
    lines+=['','- 관찰 종료까지 적격 획득이 없는 교전도 존재한다. 이것은 이후에도 영원히 획득하지 않는다는 뜻이 아니다.',
            '- 다음 교전은 같은 경기 전체를 기준으로 한다. 원격 교전도 관찰을 끊으며, 탐지되지 않는 무킬 전투 등은 경계가 되지 않는다.',
            '- 드래곤 원소/이전 스택을 원시 결과에 보존했다. 영혼을 별도 획득으로 중복 집계하지 않았고 전령 사용이나 오브젝트의 승률 효과는 측정하지 않았다.',
            '- 마지막 킬과 같은 timestamp 획득은 양의 지연 통계에서 제외한다.',
            '- 평균만으로 종료 경계를 정하지 않는다. 획득 포함률과 이후 사건의 혼입, 라벨 안정성을 함께 비교해야 한다.',
            '- 사용자의 전체 데이터 측정 요청에 따라 미래 평가용 패치도 이 기술 통계에 포함했다. 이를 보고 종료 규칙을 고르면 해당 패치를 완전히 미사용 테스트라고 주장할 수 없다.','', '## 검증','', '```json',json.dumps(checks,ensure_ascii=False,indent=2),'```','',
            '## 재현 파일','', '- [측정 스크립트](../scripts/measure_postkill_full.py)','- [보고서 생성 및 검증](../scripts/report_postkill_full.py)','- [전체 결과/소스 해시](../outputs/postkill_objective_delay_full/results.json)','- [첫 획득 연결](../outputs/postkill_objective_delay_full/first_acquisitions.csv)','- [교전별 관찰 경계](../outputs/postkill_objective_delay_full/exposures.csv)']
    (ROOT/'docs/POSTKILL_OBJECTIVE_DELAY_FULL_20260914.md').write_text('\n'.join(lines),encoding='utf-8')
    (OUT/'verification.json').write_text(json.dumps(dict(checks=checks,conditional120=conditional),indent=2),encoding='utf-8')
    print(json.dumps(dict(checks=checks,conditional120=conditional),indent=2))
if __name__=='__main__':main()
