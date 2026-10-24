from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, html
from receipts import verify_aggregation, digest
HERE=Path(__file__).resolve().parent;OUT=HERE/'review_02';RUN=HERE/'full_02'
def main():
 s=json.loads((OUT/'SUMMARY.json').read_text(encoding='utf-8'));v=json.loads((OUT/'VALIDATION.json').read_text(encoding='utf-8'));ex=json.loads((OUT/'AMBIGUOUS_KILL_AUDIT.json').read_text(encoding='utf-8'));inv=json.loads((RUN/'INVENTORY.json').read_text(encoding='utf-8'));rare=json.loads((OUT/'OBJECTIVE_DISTRIBUTIONS.json').read_text(encoding='utf-8'))
 assert v['passed'] and v['aggregation_receipt_sha256']==verify_aggregation()
 assert ex['audit_source_sha256']==digest(RUN/'MATCH_AUDIT.jsonl')
 assert ex['completed'] and ex['records_scanned']==v['audit_records'] and ex['run_complete_sha256']==digest(RUN/'RUN_COMPLETE.json') and v['exception_audit_sha256']==digest(OUT/'AMBIGUOUS_KILL_AUDIT.json')
 assert inv['canonical_duplicate_groups']==0 and ex['global_soul_conflict_masks']==0 and ex['auditor_errors']==0
 roster=json.loads((OUT/'CHAMPION_DENOMINATORS.json').read_text(encoding='utf-8'))
 n=s['known_fit_candidates'];kept=s['retained'];un=s['unresolved_scope'];qu=s['known_fit_quarantine'];short=s['population_excluded']
 figures=[('01_population','모집단과 처분',f'실제 hash 배정으로 확인된 FIT {n:,}경기 중 {kept:,}경기를 보존합니다. 기존 모집단 기준 제외 {short:,}, FIT 원본 격리 {qu:,}입니다. 패치 판독 불가 {un:,}건은 FIT 수에 섞지 않고 별도로 둡니다.'),
 ('02_duration_kills','경기 길이와 킬','검증된 종료 시각과 원본 킬 사건 수의 전체 범위를 비교합니다. 사건 중복이 의심되는 경우 자동 보정하지 않았으므로 원본 킬 기록 수는 확정된 의미상 킬 수와 구별해야 합니다.'),
 ('03_gold_extremes','골드 변화의 꼬리','최종 관측 팀 골드 차이와 인접 프레임 변화의 전체 경험 누적 분포입니다. 60/120/180초 교전 라벨 분포가 아닙니다. 큰 값 자체로 삭제한 경기는 없습니다.'),
 ('04_objectives','오브젝트 보존','원소 용·장로·바론·아타칸·전령·유충의 획득 기록 보유 경기 비율과 실제 제외된 보유 경기 수를 함께 표시합니다. 귀속 미확인 사건이 있는 경우 획득 없음으로 확정하지 않습니다. 영혼 소유와 소환 조건은 별도의 사건 규칙으로 검사했습니다.'),
 ('05_masks','필드와 사건 미지원','왼쪽은 보존 경기의 실제 원본 frame×선수 셀 분모입니다. 오른쪽은 원본 사건 전체에서 미지원 사유가 한 번이라도 있는 경기 수입니다. 이것을 예측창의 미지원율로 해석하면 안 됩니다. 경기 시작의 미귀속 아이템 사건과 누락된 어시스트 목록 때문에 대부분의 경기가 RETAIN_MASKED일 수 있습니다.'),
 ('06_features','입력 18개 분포','이전 FIT 감사에서 만든 동일 bin을 사용합니다. 표시 범위 밖 값도 histogram의 별도 tail bin으로 전부 보존했으며 정확한 수는 FEATURE_TAIL_COUNTS.json에 있습니다. 전체 원본 frame×선수 분포라 이전 교전 선별 질의의 분포와 다릅니다.'),
 ('07_champions','챔피언별 보존','전체 챔피언을 포함해 출현 수와 제외 비율을 비교합니다. 희귀 챔피언이라는 이유로 제외하지 않았습니다. 제외는 경기별 처분에 따른 것이며 챔피언에 대한 인과 효과로 해석하지 않습니다.')]
 figures[-1]=(figures[-1][0],figures[-1][1],figures[-1][2]+f" 단, 명단을 읽을 수 있는 경기만의 조건부 분포입니다. 명단 미관측 처분별 경기 수: {roster['missing_roster_matches_by_disposition']}.")
 report=f'''# 전체 FIT 원본 감사 · 새 원장 생성 전 점검

기존 hash 규칙으로 실제 FIT 원본 **{n:,}경기**를 확인했고, 승인한 기준에 따라 **{kept:,}경기를 보존 대상으로 분류**했다. 기존 모집단 조건 제외는 {short:,}경기, 확정 FIT의 원본 손상·핵심 모순 격리는 {qu:,}경기다. 패치를 판독하지 못한 {un:,}건은 FIT라고 단정하지 않고 별도로 격리했다. 원본은 이동·삭제하지 않았다.

이것은 원본 감사의 완료이며 새 학습 원장의 완료가 아니다. 시간창·교전 질의별 타깃과 마스크를 새로 생성하지 않았고 모델도 학습하지 않았다. `MATCH_AUDIT.jsonl`은 원본 hash·판정·마스크 좌표를 담은 감사 기록이다.

## 배정과 범위

파일 목록은 detail {inv['detail_files']:,}개, timeline {inv['timeline_files']:,}개다. canonical ID의 합집합에 기존 salt/bucket을 적용해 hash 적격 {inv['hash_eligible_unique_keys']:,}개를 확인하고, 버전으로 15.18–15.20 FIT만 본문 감사했다. 나머지 패치는 범위 판별에만 사용했다. 이전 160,304경기는 패치별 수의 80%를 반올림한 추정치였으며 이번 수는 실제 경기 ID로 배정했다.

| 패치 | FIT 원본 | 보존 | 모집단 제외 | 격리 |
|---|---:|---:|---:|---:|
'''
 for p,d in s['by_patch'].items():report+=f"| {p} | {sum(d.values()):,} | {d.get('RETAIN',0)+d.get('RETAIN_MASKED',0):,} | {d.get('EXCLUDE_POPULATION',0):,} | {d.get('QUARANTINE',0):,} |\n"
 report+='\n격리·제외 사유는 중복될 수 있으므로 최종 처분 수와 사유별 수를 혼용하지 않는다. 아래는 사유별 경기 수다.\n\n'
 for k,c in s['reasons_match_counts_overlapping'].items():report+=f'- `{k}`: {c:,}\n'
 report+=f'''\n## 미지원과 예외

어시스트 목록 누락은 명시적 빈 목록과 구별했다. 공식 API의 생략 의미를 확인하기 전까지 해당 사건의 어시스트를 0으로 확정하지 않는다. 이런 목록 생략이 있는 경기 전체를 제거하지 않는다. 미귀속 구매·와드·전령·유충 사건도 해당 작업/시간 구간의 지원 여부로 처리한다. 마스크는 원본 ordinal 또는 frame×선수 좌표이므로 다음 생산자는 각 입력 prefix와 출력 구간에 교차 적용해야 한다.

같은 시각·피해자의 복수 킬이 있는 **{ex['ambiguous_matches_reopened']:,}경기**는 원본을 모두 다시 열어 확인했다. 관련 파일 hash도 감사 시점과 일치했다. 아래 분포는 피해자 챔피언별 모호한 사건 그룹 수다. 독립 킬 수나 교전 질의 수가 아니다.

'''
 for k,c in ex['counts'].items():
  if k.startswith('group:'):report+=f'- {k[6:]}: {c:,}그룹\n'
 report+='\n같은 payload가 반복된 경우와 킬러·보상이 다른 경우를 구분해 `AMBIGUOUS_KILL_AUDIT.json`에 기록했다. 챔피언 예외인지 API 표현 문제인지 이번 감사만으로 확정하지 않는다. 자동 dedup이나 경기 전체 제외 대신 해당 킬·어시스트 타깃 및 그 사건으로 정의된 교전 경계의 지원 여부를 새 생산자에서 다뤄야 한다. 사후 상세 경기의 deaths 비교는 감사용이며 모델 입력이 아니다.\n\n'
 report+='음수 저항력·큰 스탯·긴 경기·큰 골드 변화는 보존했다. 원본 `attackSpeed` 등의 단위도 그대로 두었다. 관측 규칙과의 모순은 원본 획득 삭제가 아니라 파생 availability의 미지원으로 구분했다. K3 의미 태그, 실제 버프 보유자·쿨다운·복제 스킬, 지형의 실시간 상태는 이번 감사로 새롭게 관측 가능해진 것이 아니다.\n\n'
 for stem,title,caption in figures:report+=f'## {title}\n\n{caption}\n\n![{title}]({(OUT/(stem+".png")).as_posix()})\n\n'
 report+=f'''## 검증과 한계

기존 FIT 배정 {s['existing_fit_assignments_found']:,}경기가 이번 감사 명단에 모두 포함되었다. 기존 r2의 원본 hash와 비교 가능한 {s['old_r2_source_hashes_compared']:,}경기는 모두 일치했다. 전체 감사 기록의 ID·bucket·마스크 좌표·histogram 분모를 독립 검산했고, 시작/종료 시 원본 파일명 목록 hash도 일치했다. 감사 코드 내부 오류는 {ex['auditor_errors']}건이다.

초기 full_01은 독립 검토 후 중단했으며 기록과 실행 코드를 남겼다. 최종 수치는 full_02에서만 가져온다. 실행 코드의 사본과 hash가 저장되어 있다. 검토에서 지적된 canonical 중복 그룹과 양 팀 영혼의 전역 모순 분기는 실제 전체 감사에서 각각 {inv['canonical_duplicate_groups']}건, {ex['global_soul_conflict_masks']}건이어서 이번 결과에 영향을 주지 않았다. 이 두 미발생 분기는 이번 실행에서 동작을 검증한 것이 아니며, 해당 유형의 원본이 추가될 경우 재사용 전에 수정·회귀 검증해야 한다.

이 검사는 스키마·원본 정합성과 관측 가능성 감사다. 새 교전 선별 후의 표본 수, 예측창별 지원율, 모델 성능, 인과적 오브젝트 가치와 잔여 가치 U는 아직 결과가 없다. 일반 시점의 교전 발생 예측 모집단으로 확장하지 않았다.

## 다음 단계의 검토안

이번 격리·모집단 제외 명단은 새 원장 생산 후보에서 빼고, 보존 경기에는 필드/사건 마스크를 적용한다. 모호한 킬을 사용하는 교전 경계는 자동 확정하지 않으며, 원장의 count와 focal kill 타깃을 해당 ordinal에 연결해야 한다. 입력 마스크는 t0 prefix만으로 결정하고 출력 마스크는 해당 예측창 안에서 결정한다.

사용자가 위 제외 전후 분포와 미지원 처리 방식을 점검한 뒤 새 원장을 별도 버전으로 만든다. 그때 60·120·180초별 목표 지원율, 교전 수·경기 수, 희귀 사건의 유효 수를 다시 시각화한다. 이 단계에서는 새 원장을 만들지 않았다.
'''
 (OUT/'AUDIT_REPORT_KO.md').write_text(report,encoding='utf-8')
 nav=''.join(f'<a href="#{stem}">{i+1}. {html.escape(title)}</a>' for i,(stem,title,c) in enumerate(figures))
 body=''.join(f'<section id="{stem}"><h2>{i+1}. {html.escape(title)}</h2><p>{html.escape(c)}</p><a href="{stem}.png" target="_blank"><img src="{stem}.png" loading="lazy" alt="{html.escape(title)}"></a></section>' for i,(stem,title,c) in enumerate(figures))
 page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>전체 FIT 원본 감사</title><style>body{font-family:Malgun Gothic,system-ui,sans-serif;background:#f5f6f8;color:#20313e;margin:0;line-height:1.8}main{max-width:1120px;margin:auto;padding:30px 24px}h1{font-size:30px}h2{font-size:21px}section{padding:24px;background:white;border:1px solid #dbe0e5;margin:22px 0}img{width:100%;height:auto}nav{display:flex;flex-wrap:wrap;gap:10px 20px}a{color:#235c88}.note{padding:18px;background:white;border-left:4px solid #287d76}.numbers{font-size:19px}footer{font-size:13px}</style><main>'''+f'''<h1>전체 FIT 원본 감사</h1><p class="numbers">FIT {n:,}경기 → 보존 {kept:,}경기</p><p>모집단 제외 {short:,} · FIT 격리 {qu:,} · 패치 미확인 별도 {un:,}</p><div class="note">원본 전수 감사와 검산을 마쳤습니다. 새 학습 원장 생성 전 점검 자료입니다. 부분 미지원은 마스크로 남기고, 희귀 사건·극단값만을 이유로 버리지 않았습니다.</div>'''+nav+body+f'''<section><h2>점검할 사항</h2><p>격리·제외 명단과 사건/필드 마스크를 새 생산자에 연결하는 안입니다. 모호한 킬과 생략된 어시스트 목록을 자동으로 확정값으로 바꾸지 않습니다. 이 기준으로 새 원장을 만들기 전 확인을 기다립니다.</p><p><a href="AUDIT_REPORT_KO.md">전체 감사 보고서</a> · <a href="MATCH_DISPOSITIONS.csv">경기별 처분</a> · <a href="QUARANTINE_AND_UNRESOLVED.json">격리 근거</a> · <a href="AMBIGUOUS_KILL_AUDIT.json">모호한 킬 재확인</a> · <a href="VALIDATION.json">검산 결과</a></p></section><footer>출처: full_02/MATCH_AUDIT.jsonl 및 FEATURE_HISTOGRAMS.npz. 원본 경로·SHA256는 경기별 감사 기록에 있습니다. 그림의 분모는 경기 또는 실제 frame×선수 셀입니다. 60/120/180초 교전 라벨이나 모델 성능 자료가 아닙니다.</footer></main></html>'''
 (OUT/'index.html').write_text(page,encoding='utf-8')
 manifest={}
 for base in [RUN,OUT]:
  for p in base.iterdir():
   if p.is_file() and p.name!='ARTIFACT_MANIFEST.json':
    h=hashlib.sha256()
    with p.open('rb') as f:
     for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    manifest[str(p.relative_to(HERE))]={'sha256':h.hexdigest(),'bytes':p.stat().st_size}
 (OUT/'ARTIFACT_MANIFEST.json').write_text(json.dumps({'status':'AUDIT_COMPLETE_AWAITING_REVIEW_BEFORE_NEW_LEDGER','generated_utc':datetime.now(timezone.utc).isoformat(),'files':manifest,'new_training_ledger_created':False},ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps({'files':len(manifest),'known_fit':n,'retained':kept,'review':str(OUT/'index.html')},ensure_ascii=False))
if __name__=='__main__':main()
