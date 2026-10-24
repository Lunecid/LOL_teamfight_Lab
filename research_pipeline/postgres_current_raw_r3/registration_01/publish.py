"""Publish only after database and preservation acceptance passes."""
from pathlib import Path
from collections import Counter
import json,html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from register import HERE,SCHEMA,read,dump,sha

def main():
    verified=read(HERE/'VERIFICATION.json');assert verified['pass']
    expected=read(HERE/'FILE_EXPECTED.json');actual=read(HERE/'SQL_SUPPORT_SUMMARY.json')
    staging=read(HERE/'STAGING.json');review=HERE/'review';review.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False})
    categories=['고유 감사 경기','유지 원장 경기','교전 행','60·120·180초 행']
    tables=['matches','sources','queries','horizons']
    counts=[expected['counts'][t] for t in tables]
    db_counts=[verified['tables'][t]['n'] for t in tables]
    fig,ax=plt.subplots(figsize=(11,4.7));x=np.arange(4)
    ax.bar(x-.18,counts,.36,label='파일 기반',color='#517e9b');ax.bar(x+.18,db_counts,.36,label='PostgreSQL',color='#358579')
    ax.set_xticks(x,categories);ax.set_ylim(0,max(counts)*1.23);ax.legend();ax.set_title('기존 파일과 PostgreSQL 등록 결과 — 전수 일치')
    for i,n in enumerate(counts):ax.text(i,n+max(counts)*.035,f'{n:,}\n차이 0',ha='center',fontsize=10)
    ax.ticklabel_format(axis='y',style='plain');ax.spines[['top','right']].set_visible(False);fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(review/f'01_counts.{ext}',dpi=150)
    plt.close(fig)
    roles=sorted({r['split_role'] for r in actual});grid=np.array([[next(r['economic_supported']/r['queries']*100 for r in actual if r['split_role']==role and r['horizon_s']==h) for h in (60,120,180)] for role in roles])
    fig,ax=plt.subplots(figsize=(8,5.6));im=ax.imshow(grid,vmin=99.9,vmax=100,cmap='YlGnBu',aspect='auto')
    ax.set_xticks(range(3),['60초','120초','180초']);ax.set_yticks(range(len(roles)),roles);ax.set_title('SQL로 집계한 순골드 라벨 지원율\n모든 패치·분할·예측창에서 파일 집계와 차이 0')
    for i in range(len(roles)):
        for j in range(3):ax.text(j,i,f'{grid[i,j]:.4f}%',ha='center',va='center',color='white' if grid[i,j]>99.95 else 'black')
    fig.colorbar(im,ax=ax,label='지원율 (%) — 색 범위 99.9~100');fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(review/f'02_support.{ext}',dpi=150)
    plt.close(fig)
    usage=Counter();calls=[]
    for p in HERE.glob('strata_*.json'):
        r=read(p);u=r.get('usage',{});usage.update({k:u.get(k,0) for k in ('prompt_tokens','completion_tokens','total_tokens')});calls.append({'file':p.name,'finish_reason':r.get('finish_reason'),'usage':u})
    dump(HERE/'STRATA_USAGE.json',{'local_tokens':dict(usage),'calls':calls,'cloud_quota_savings_measured':False,'review_limit':'Import review truncated; no retry loop. Bytea/bit-order suggestions rejected by direct PostgreSQL encoding probe; all acceptance relies on executed checks.'})
    disposition=Counter()
    for r in expected['dispositions']:disposition[r['disposition']]+=r['n']
    report=f'''# 현재 원본 PostgreSQL 등록 및 대조

등록 DB: `lol_project_raw_catalog_20261018`  
새 스키마: `{SCHEMA}`  
검증 결과: **통과**. 기존 `raw_cache` 데이터와 기존 원장 제어 파일·모든 NPZ shard 해시가 전후 동일하다. 원본 이동·삭제, 분할 재배정, 원장 재생성, 학습은 실행하지 않았다.

## 등록 범위

- 고유 경기 {expected['counts']['matches']:,}개: 유지 {disposition['RETAIN_MASKED']:,}, 기존 모집단 제외 {disposition['EXCLUDE_POPULATION']:,}, 격리 {disposition['QUARANTINE']:,}.
- 감사 이력 {expected['counts']['audits']:,}개: 두 감사에 중복된 패치 미확인 40경기의 이력을 모두 보존했다. canonical 경기 목록은 최신 HOLDOUT 감사를 연결하며 패치·분할을 추정하지 않았다.
- detail/timeline 파일 참조 {expected['counts']['files']:,}개. 실제 없는 파일 {expected['missing_files']:,}개도 `present=false`로 남긴다. 현재 존재하지만 감사 SHA-256이 없는 참조는 {expected['present_files_without_audit_hash']:,}개다.
- 원장 경기 {expected['counts']['sources']:,}개, 교전 {expected['counts']['queries']:,}행, 60·120·180초 {expected['counts']['horizons']:,}행.
- `UNUSED_RESERVED` 26,415경기는 등록 대상에서 제외했다. 학습 모집단으로 편입하지 않았다.

## 구조 및 활용

`catalog_info`에는 입력 계약·마스크 차원·소스 manifest를 저장했다. `audits`는 원본 감사 JSON 텍스트를 그대로 보존하고 JSONB로 조회할 수 있게 했다. `matches`는 고정 분할·패치·처분을 연결한다. `files`에는 파일 경로·존재·현재 크기/mtime과 감사 당시 해시/크기를 구별했다.

`sources` → `queries` → `horizons`가 경기·교전·예측창을 연결한다. `shards`와 `shard_row`로 기존 NPZ의 정확한 행을 찾는다. 입력 및 타깃 마스크는 C-order, little-bit-order bytea로 저장했다. `get_bit`으로 팀·필드별 지원 여부를 조회할 수 있다. 축 순서·필드 이름은 `catalog_info`와 `MASK_LAYOUT.json`에 있다.

원본 감사의 `field_masks`/`event_masks`는 원래 inclusive 좌표를 그대로 보존한다. `rule_masks`가 존재하면 원래 onset 정보도 그대로 남는다. 전체 경기 감사·격리 정보는 입력 피처가 아니며, 쿼리별 입력 마스크와 예측창별 정답 마스크를 서로 대체하지 않는다. 현재 학습용 NPZ loader는 그대로 사용한다.

경제 부호 원본의 미지원 sentinel은 `-9`다. DB는 `economic_supported=false`와 `economic_sign=-9`를 함께 보존하며, 이를 RED 승리로 해석하지 않는다. 지원된 행의 부호만 -1/0/1이다. 첫 적재에서 이 sentinel을 빠뜨린 제약이 실패했으며, 스키마 전체 롤백 및 기존 raw_cache 보존을 확인한 후 제약을 수정해 재등록했다. 실패 로그와 `ROLLBACK_VERIFIED.json`도 남겼다.

## 검증 범위

1. 기존 검증 영수증의 모든 NPZ hash와 생산 요약·원본 감사 hash를 확인한 뒤 staging을 생성했다.
2. 신규 스키마와 모든 COPY를 하나의 트랜잭션으로 처리했다. 기존 스키마 이름이 있으면 교체하지 않고 실패한다. PK/FK·소스별 교전 수·shard별 행 수·행당 세 예측창·미지원 선택행의 타깃 마스크를 커밋 전에 검사했다.
3. 7개 데이터 테이블의 모든 저장 열을 CSV로 재추출했다. 행 수·열 이름·각 행 SHA-256의 multiset 합을 staging과 비교해 전수 일치를 확인했다. JSONB 생성 열은 원본 JSON 텍스트에서 DB가 생성한다. catalog_info도 의미적 JSON 동등성을 확인했다.
4. 독립 SQL 집계와 NumPy 집계의 {verified['support_groups']}개 분할·패치·예측창 그룹이 일치한다. 순골드·종료·파레토·전체 사건·6개 에픽 공통 지원 및 팀별 20개 사건 지원을 비교했다. 기존에 전달한 지원율 CSV의 {verified['prior_published_support_rows_matched']}개 행도 hazard 위험집합 지원을 포함해 분자·분모가 일치한다. 동결 분할 SQL 교차검사 불일치 0개다.
5. 기존 raw_cache 두 테이블의 정렬된 전체 데이터 export 해시가 전후 일치했다. 기존 제어 파일·원장·모든 NPZ shard의 SHA-256도 전후 일치했다.

**해석 한계:** raw 원본 내용 전체를 이번에 다시 읽어 해싱하지 않았다. 저장한 `audit_sha256`은 기존 원본 감사의 해시다. 이번 등록은 현재 파일 존재·기존 감사 크기와의 일치 및 기존 원장/검증 산출물의 보존을 확인했다. 따라서 이를 새로운 원본 내용 재감사로 표현하지 않는다. DB 등록 통과가 전체 모델·지형·가치 추정 파이프라인의 학습 준비 완료를 의미하지 않는다.

## 조회 예제와 근거

- `EXAMPLE_QUERIES.sql`: split/patch 조건으로 원본 찾기, 격리 사유, 지원율, shard 행 연결, 바론 지원 비트.
- `VERIFICATION.json`: 전수 DB 대조 및 보존 영수증.
- `SQL_SUPPORT_SUMMARY.json`: DB 지원율의 정확한 분자와 분모.
- `SOURCE_MANIFEST.json`, `STAGING.json`, `CATALOG_METADATA.json`: 소스·등록 provenance.
- `ENCODING_PROBE.json`: 실제 PostgreSQL CSV NULL·bytea·bit order 시험.

Strata 로컬 사용량은 {usage['total_tokens']:,}토큰이다. 적재 검토 응답 하나는 길이 제한으로 잘렸으며, 제기된 bytea/bit 순서 주장은 실제 DB 시험으로 확인해 잘못된 제안을 채택하지 않았다. 구독/클라우드 사용량 절감은 측정하지 않았다.
'''
    (HERE/'REPORT_KO.md').write_text(report,encoding='utf-8')
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>현재 원본 DB 등록 검증</title><style>body{margin:0;background:#f3f5f6;color:#223746;font:16px/1.8 Malgun Gothic,system-ui}main{max-width:1120px;margin:auto;padding:28px}section{background:white;padding:22px;margin:24px 0;border:1px solid #ddd}img{width:100%}a{color:#226985}</style><main><h1>현재 원본 PostgreSQL 등록 — 검증 통과</h1>'''
    page+=f'<p>별도 스키마 <code>{SCHEMA}</code>. 기존 DB·원장·분할 보존. 학습 미실행.</p>'
    for image,title in [('01_counts','등록 규모와 파일 대조'),('02_support','분할·패치·예측창별 지원율')]:page+=f'<section><h2>{title}</h2><img src="{image}.png" alt="{title}"></section>'
    page+='<section><h2>검증 근거와 사용법</h2><p>현재 파일 존재·크기를 확인했습니다. 원본 내용 해시는 기존 감사값이며 이번 전체 재해싱 결과가 아닙니다.</p><p><a href="../REPORT_KO.md">전체 보고서</a> · <a href="../VERIFICATION.json">전수 검증</a> · <a href="../EXAMPLE_QUERIES.sql">SQL 조회 예제</a> · <a href="../MASK_LAYOUT.json">마스크 구조</a> · <a href="../SQL_SUPPORT_SUMMARY.json">지원 분자·분모</a></p></section></main></html>'
    (review/'index.html').write_text(page,encoding='utf-8')
    files=[p for p in HERE.iterdir() if p.is_file() and p.suffix in ('.py','.sql','.json','.md') and p.name!='DELIVERY_MANIFEST.json']+list(review.iterdir())
    dump(HERE/'DELIVERY_MANIFEST.json',{'files':{str(p.relative_to(HERE)):sha(p) for p in files},'training':False})
    print(json.dumps({'pass':True,'schema':SCHEMA,'counts':expected['counts'],'local_tokens':usage['total_tokens']}))
if __name__=='__main__':main()
