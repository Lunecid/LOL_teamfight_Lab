from pathlib import Path
import json,collections
R=Path(__file__).parent
def read(n):return json.loads((R/n).read_text(encoding='utf-8'))
def main():
 report=read('semantic_validation_report.json');rows=read('ability_semantics.json')['records'];exceptions=read('exception_registry.json')['records'];vocab=read('semantic_feature_catalog.json')
 groups=collections.defaultdict(list)
 for r in rows:groups[r['skill_key'].rsplit('.',1)[0]].append(r)
 status_names={'SOURCE_EXPLICIT':'명시','SOURCE_CONDITIONAL':'조건부','AMBIGUOUS':'모호'}
 lines=['# 전체 챔피언 스킬 의미·예외 추출 보고서 v2','',
 f"15.14.1의 **{report['champions']}종·{report['skills']}개 패시브/QWER 항목 전수**를 읽고 구조화했다. 총 {report['claims']:,}개 의미 주장, 정규화 후 {report['tags']}개 태그 식별자다. 비어 있는 스킬 레코드는 없다.",'',
 '범위는 Data Dragon의 기본 카탈로그 항목이다. 모든 변신형·하위 스킬·서버 스크립트를 완전히 복원했다는 뜻은 아니다. 설명과 툴팁을 모두 읽었으며 각 주장에 적용 조건·근거 구간·현재 상태 의존성을 연결했다.','',
 '## 1. 출처 근거와 현재 상태','',
 '| 출처 지원 | 주장 수 | 처리 |','|---|---:|---|']
 for s,n in report['source_support_counts'].items():lines.append(f"| {status_names[s]} ({s}) | {n:,} | "+('주 입력의 확정 태그에서 제외·마스크' if s=='AMBIGUOUS' else '정적 능력/조건 지식으로 보존; 현재 발동은 별개')+' |')
 lines += ['',f"{report['claims']:,}개 근거 구간이 정규화된 원본의 정확한 부분 문자열인지 검사했고, 171개 챔피언 원본과 전체 205개 출처 해시를 대조했다. 독립 의미 검토는 NOT_DONE이다. 원문 구간 검사 통과는 자동으로 모든 해석의 정확성이나 실제 서버 동작을 증명하지 않는다.",'',
 'SOURCE_CONDITIONAL에는 적중·시전·대상 상태 같은 일반 조건도 포함한다. 이 수가 모두 특수한 변신/부활 예외라는 뜻은 아니다. 태그 미언급은 false가 아니라 UNKNOWN이다. 현재 시전, 쿨다운, 충전, 표식, 변신, 복제 대상 등은 경기 스키마에서 직접 관측되지 않으므로 정적 태그를 current_active=true로 승격하지 않는다.','',
 '기본 패키지의 mechanic_facts는 독립 검토 전 UNVERIFIED이며, 세부 source_support는 ability_semantics.json에 별도 저장한다. 이것은 원문 근거를 못 찾았다는 뜻이 아니라 출처 해석·독립 검토·현재 상태의 상태값을 혼용하지 않기 위한 것이다.','',
 '## 2. 상태 엔진에 중요한 예외','',
 '| 계열 | 대표 대상 | 확인한 능력 | 현재 불명인 상태 / 엔진 처리 |','|---|---|---|---|',
 '| 변신·상태 전환 | Elise, Gnar, Jayce, Nidalee, Shyvana, Yunara 등 | 형태에 따라 스킬/수치가 달라짐 | 현재 형태·전환 시전·지속시간이 없으면 단일 키트로 확정하지 않음 |',
 '| 무기·다중 스킬 선택 | Aphelios, Hwei, Udyr 등 | 기본 슬롯 아래 무기·선택·각성 분기 | 현재 무기/선택/각성 및 별도 하위 스킬 커버리지는 따로 표시 |',
 '| 아군 부활 | Akshan, Zilean | 표식/보호와 조건을 만족하면 부활 | 대상 표식·보호 버프·발동 여부를 알 수 없으면 부활을 확정하지 않음 |',
 '| 자기 부활 계열 | Anivia, Zac | 알·분열체 등의 생존/복귀 과정 | source에 생략된 생존·방해 조건을 추측으로 완성하지 않음 |',
 '| 사망 지연·방지 | Renata, Tryndamere | 조건부 사망 지연 또는 일정 시간 죽음 방지 | 실제 CHAMPION_KILL과 치명 피해 추론을 구분 |',
 '| 사망 후 활동 | Karthus, Sion | 죽은 뒤에도 일정한 행동/효과 가능 | 생존·행동 가능·타깃 가능 상태를 분리하고 노드를 삭제하지 않음 |',
 '| 빙의 | Viego | 다른 챔피언의 몸/능력 상태를 일시적으로 사용 | 실제 빙의 대상·종료·유효 스킬을 champion_id만으로 결정하지 않음 |',
 '| 복제·도용 | Sylas, Zoe, LeBlanc 등 | 궁극기/주문·아이템/자기 스킬 복제는 서로 다른 범위 | 무엇을 복제했는지와 잔여 사용 가능 상태는 미관측 |',
 '| 위장·분신 | Neeko, Shaco, Wukong 등 | 겉모습 또는 분신이 실제 정체성과 다름 | 실제 참가자 ID를 덮어쓰거나 분신 위치를 만들어 넣지 않음 |',
 '| 원격 기점·부착 | Orianna, Azir, Yuumi 등 | 공·병사·부착 대상이 효과 기점을 결정 | 선수 XY를 효과 기점의 실시간 좌표로 대체하지 않음 |','',
 f"exception_registry.json에는 {len(exceptions)}개 조건/예외 설명이 있으며 {len({x['champion'] for x in exceptions})}개 챔피언을 참조한다. 기본 조건 묶음과 여러 스킬을 잇는 상세 설명이 겹칠 수 있으므로, 독립된 예외 현상 {len(exceptions)}개나 중복 지급할 효과 {len(exceptions)}개로 해석하지 않는다.",'',
 '`UNKNOWN_UNLESS_OBSERVED`는 필요한 상태 의존성의 근거가 없을 때 불명으로 두라는 정책이다. 승인된 결정적 규칙과 충분히 관측된 전제가 있으면 RULE_DERIVED 상태를 만들 수 있지만, 이번 패키지는 실행 규칙을 자동 활성화하지 않는다.','',
 '## 3. 모호한 주장 전체 목록','',
 '| 스킬 | 태그 | 남긴 조건/충돌 |','|---|---|---|']
 for r in rows:
  for c in r['claims']:
   if c['assertion']=='AMBIGUOUS':lines.append('| '+r['skill_key']+' | '+c['tag']+' | '+(c['condition'] or '').replace('|','/')+' |')
 lines += ['', '## 4. 챔피언별 전수 커버리지','', '| 챔피언 | 항목 | 주장 | 모호 | 조건/예외 참조 |','|---|---:|---:|---:|---:|']
 for ch,rs in sorted(groups.items()):lines.append(f"| {ch} | {len(rs)} | {sum(len(r['claims']) for r in rs)} | {sum(c['assertion']=='AMBIGUOUS' for r in rs for c in r['claims'])} | {sum(e['champion']==ch for e in exceptions)} |")
 lines += ['', '## 5. 모델 투입 규정','',
 f"{report['tags']}개는 추출된 세부 태그 vocabulary이며 18개 기본 관측 피처를 대체하지 않는다. 같은 뜻의 표현과 공격적 디버프/방어 효과의 명칭은 semantic_normalization.json에서 정규화했고 원래 shard는 보존했다. 이후 ontology 수정도 버전과 매핑을 남긴다.",'',
 'K3에서는 능력 존재·지원 마스크·조건 임베딩을 함께 받는다. 미언급을 0/false로 학습시키지 않는다. 현재 조건이 충족됐다고 할 수 없으면 동적 활성도는 UNKNOWN이다. 역할별 임의 합산으로 공성력·조합 점수를 만들지 않는다. 같은 지식 표현을 tree/attention/temporal/graph 비교군에 제공한다.','',
 'K2 텍스트 임베딩과 K3 태그 임베딩 벡터는 아직 생성하지 않았다. 전수 추출이 끝났다는 이유만으로 예측 성능이나 오브젝트 가치 타당성이 확인된 것은 아니다. 모델·상태 엔진의 실행은 END_TO_END_DESIGN_KO.md의 별도 인수 조건을 따른다.']
 (R/'SEMANTICS_AND_EXCEPTIONS_KO.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 readme=f'''# 15.14 전수 스킬 의미 패키지 v2와 통합 설계서

**171종·855개 카탈로그 항목 전수 추출 완료.** 원문 설명·툴팁에서 의미 주장 {report['claims']:,}개를 추출했고, 모호한 주장 {report['source_support_counts']['AMBIGUOUS']}개를 분리했다. 조건/예외 레코드는 {len(exceptions)}개다. 원본 패키지 v1은 보존했다.

우선 읽을 문서:

1. [데이터→입력→예측→출력 통합 정의·설계서](END_TO_END_DESIGN_KO.md)
2. [전체 스킬 의미·예외·171종 커버리지 보고서](SEMANTICS_AND_EXCEPTIONS_KO.md)
3. [기계 판독 파이프라인 계약](PIPELINE_CONTRACT_V4.json)

실제 데이터 파일:

| 파일 | 내용 |
|---|---|
| bundle.json | 챔피언·수치·오브젝트·규칙·전수 의미를 결합한 패키지 |
| ability_semantics.json | 855개 스킬의 태그·조건·원문 구간·출처 포인터·런타임 의존성 |
| exception_registry.json | 변신·부활·사망 후 활동·복제·빙의·위장·소환·원격 기점 등 |
| observability_contract.json | 정적으로 아는 사실 / 사건·프레임에서 아는 상태 / 현재 미관측 상태 |
| semantic_feature_catalog.json | 정규화된 태그 vocabulary와 지원·불명 마스크 규칙 |
| semantic_normalization.json | 용어 통합 이력; 원래 shard와 대응 |
| runtime_policy.json | 규칙 자동 실행 금지, 미지원 상태 처리, 연구용 지식 arm의 범위 |
| sources/, source_locations.json | 205개 원본·출처 위치·무결성 근거. v2에 필요한 원본을 함께 복사 |
| shards/ | 3개 독립 분담 결과와 재구성 스크립트 |

**완료와 미완료를 구분한다.** 원문 전수 추출·정적 포인터/해시/커버리지 검증과 설계 계약 검사는 완료했다. 독립 의미 정확성 검토, 실제 경기 상태 replay, 전수 피처·라벨 생산, 임베딩 생성·학습, 인과 가치 추정은 실행하지 않았다. 카탈로그 855개는 모든 서버 스크립트/하위·변신 스킬까지 완전 복원했다는 의미가 아니다.

`SOURCE_EXPLICIT`/`SOURCE_CONDITIONAL`은 제공된 원문에 근거한 기능 해석이다. 현재 스킬 사용 가능·버프 발동·변신·부활·복제 상태가 관측됐다는 뜻이 아니다. `AMBIGUOUS`는 확정 태그에서 제외하며, 미언급은 false로 바꾸지 않는다. 독립 검토 상태는 별도 NOT_DONE으로 남겼다.

검증:

- 스킬 의미: {report['quote_checks']:,}개 원문 구간, 챔피언 171종, 슬롯 855개, 누락/중복 없음.
- core bundle: {report['core_package_checks']['evidence_references_checked']:,}개 출처 참조와 {report['core_package_checks']['facts_checked']:,}개 사실 레코드, 205개 원본 해시.
- 설계 계약: 시간·분할·Q/P 목표 구분·골드 정산·지원 상태 등 13개 일관성 검사.
- 숫자는 정적 산출물 감사 수치이며 모델 성능이나 경기 실험 결과가 아니다.

단독 패키지 검사(표준 라이브러리만 필요):

```powershell
python validate_export.py
```

재생성은 부모 v1 패키지에서 `integrate_semantics.py`를 실행한다. 설계 계약 재생성은 원 연구 명세를 읽으므로 별도 프로젝트 경로가 필요하다. 배포된 자료의 검증에는 원 연구 저장소나 raw 경기가 필요하지 않다. `artifact_manifest.json`과 ZIP 해시는 완성 시점의 무결성을 기록한다.

예측 설계는 관측 후속 성과가 주 기준이다. 잔여 자산 U 확장은 기존 우세·반사실·지원 문제가 해결될 때까지 비활성이다. 기존 연구 데이터·동결 라벨·모델은 변경하지 않았다.
'''
 (R/'README_KO.md').write_text(readme,encoding='utf-8')
 print('Readable reports written with full 171-champion coverage table and all ambiguous claims.')
if __name__=='__main__':main()
