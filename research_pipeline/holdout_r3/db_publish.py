"""Publish catalog-scope comparison without rebuilding scientific artifacts."""
from pathlib import Path
from collections import Counter
import json, re, hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from db_reconcile import sha,dump,H
V=H/'production_01/review'

def main():
    r=json.loads((H/'DB_INDEX_RECONCILIATION.json').read_text(encoding='utf-8'))
    assert r['catalog_scan_ids_counts_and_bytes_match']
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False})
    fig,ax=plt.subplots(figsize=(11,4.4))
    labels=['DB 가공 캐시','현재 원본 detail','현재 유지 원장 (FIT + held-out)','DB ∩ 현재 유지 원장']
    values=[r['db_unique_matches'],r['raw_detail_ids'],r['retained_ledger_matches'],r['db_intersect_retained_ledger']]
    bars=ax.barh(labels,values,color=['#b17b32','#4a7e96','#2a6661','#ac4343'])
    ax.invert_yaxis();ax.set_xlim(0,360000);ax.set_xlabel('고유 경기 수 (교전 행 수가 아님)')
    for bar,value in zip(bars,values):ax.text(value+3500,bar.get_y()+bar.get_height()/2,f'{value:,}',va='center')
    ax.set_title('PostgreSQL 인덱스는 현재 학습 원본과 별도 모집단입니다',pad=18)
    ax.spines[['top','right']].set_visible(False)
    fig.text(.02,.02,'경기 ID 전체 집합 대조 · 내용 동일성/캐시 유효성 검증 아님 · DB 미등록을 이유로 제외하지 않음',fontsize=10)
    fig.tight_layout(rect=[0,.07,1,1])
    for suffix in ('png','svg'):fig.savefig(V/f'11_db_scope.{suffix}',dpi=150)
    plt.close(fig)
    lines=['# PostgreSQL 인덱스 대조 결과','',
      '접속과 대조를 완료했다. 읽기 전용 REPEATABLE READ 트랜잭션으로 메타데이터를 확보했으며 DB·원본·원장·분할을 변경하지 않았다. 학습은 실행하지 않았다.','',
      f"DB는 `{r['snapshot']['scans'][0]['root_path']}`의 가공 캐시다. {r['db_files']:,}개 파일, {r['db_unique_matches']:,}경기다. 파일 종류별 수는 {r['db_file_kinds']}이며 스캔별 파일 수·경기 수·바이트 합을 등록 메타데이터와 대조했다.",'',
      f"현재 detail 원본 {r['raw_detail_ids']:,}경기 및 유지 원장 {r['retained_ledger_matches']:,}경기와 DB의 ID 교집합: detail {r['db_intersect_raw_detail']:,}경기, 유지 원장 {r['db_intersect_retained_ledger']:,}경기. 원본·원장 ID 형식 검증 여부는 {r['id_namespace_verified']}다.",'',
      '| 분할/패치 | 유지 경기 | DB 포함 경기 | 교전 행 |','|---|---:|---:|---:|']
    for key,c in r['coverage'].items():lines.append(f"| {key} | {c['retained_matches']:,} | {c['indexed_matches']:,} | {c['query_rows']:,} |")
    lines+=['','DB에는 패치·원본 내용 해시가 없다. 크기와 수정 시각은 가공 캐시 파일의 값이므로 현재 detail/timeline 원본의 값과 비교할 수 없다. 이 대조는 캐시 파일의 현존 여부·mtime·내용을 다시 검사한 결과가 아니다. 기존 원장 내용의 근거는 독립 원본 감사와 원장 재계산 영수증이다.','',
      '현재 DB를 이용해 학습 원본을 선택하면 전체 학습 모집단을 잃는다. 따라서 이 DB의 등록 여부를 새 제외·격리 조건으로 쓰지 않는다. 기존 분할·마스크·라벨·원장은 보존했다. 현재 DB의 다른 캐시를 혼합하는 대신, 현 학습 원본용 별도 catalog scan이 필요하다. 이번 작업에서는 새 DB 테이블이나 인덱스를 생성하지 않았다.','',
      '로컬 Strata의 코드 검토에서 ID 형식 확인과 등록되지 않은 scan_id 검사 제안을 반영했다. 반면 FIT/held-out 중복이 허용될 수 있다는 제안은 누출 방지 계약과 충돌하여 채택하지 않았다. ID·분모·바이트 수 검증은 직접 실행했다.','',
      '근거: DB_SCHEMA.json, DB_METADATA_SNAPSHOT.txt, DB_LEDGER_ID_COVERAGE.csv, DB_INDEX_RECONCILIATION.json. 인증 내용은 이 산출물들에 포함하지 않는다.']
    (H/'DB_RECONCILIATION_KO.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    p=H/'END_TO_END_REVIEW_KO.md';s=p.read_text(encoding='utf-8')
    marker='\n## 10. PostgreSQL 인덱스 범위 확인'
    s=s.split(marker)[0]+marker+'\n\n2026-10-18 후속 대조는 실행 시점의 등록 캐시·detail 원본·유지 원장 분모와 ID 교집합을 계산해 보고한다. 산출된 대조 결과를 원본·분할·마스크의 변경 근거로 사용하지 않는다. 현재 원본용 catalog scan이 필요하면 별도 절차로 수행한다. 상세 근거는 [DB_RECONCILIATION_KO.md](DB_RECONCILIATION_KO.md)를 참조한다.\n'
    p.write_text(s,encoding='utf-8')
    p=V/'index.html';s=p.read_text(encoding='utf-8')
    if 'id="11_db_scope"' not in s:
        s=s.replace('</main>','<section id="11_db_scope"><h2>DB 인덱스 범위 대조</h2><p>DB 캐시, 현재 detail 원본 및 유지 원장의 경기 수와 ID 교집합을 실행 시점에 계산해 표시합니다. 다른 모집단일 경우 제외 기준을 자동 변경하지 않습니다.</p><img src="11_db_scope.png" alt="DB와 현재 원본 및 원장의 경기 수 비교"><p><a href="../../DB_RECONCILIATION_KO.md">DB 대조 해석과 한계</a></p></section></main>')
    p.write_text(s,encoding='utf-8')
    usage=Counter();calls=[]
    for p in sorted(H.glob('strata_*.json')):
        x=json.loads(p.read_text(encoding='utf-8'));u=x.get('usage',{})
        usage.update({k:u.get(k,0) for k in ('prompt_tokens','completion_tokens','total_tokens')})
        calls.append({'file':p.name,'usage':u,'finish_reason':x.get('finish_reason')})
    dump(H/'STRATA_USAGE.json',{'local_usage':dict(usage),'calls':calls,'measured_cloud_subscription_savings':None,'note':'Local API tokens only; character reduction is not quota or billing savings.'})
    dump(V/'MANIFEST.json',{'files':{p.name:sha(p) for p in V.iterdir() if p.is_file() and p.name!='MANIFEST.json'},'training':False})
    path=H/'production_01/DELIVERY_MANIFEST.json';m=json.loads(path.read_text(encoding='utf-8'))
    added=['DB_SCHEMA.json','db_reconcile.py','db_publish.py','DB_METADATA_SNAPSHOT.txt','DB_LEDGER_ID_COVERAGE.csv','DB_IDS_OUTSIDE_CURRENT_RAW.txt','DB_RECONCILIATION_KO.md','strata_db_review.json']
    allowed={'DB_INDEX_RECONCILIATION.json','END_TO_END_REVIEW_KO.md','STRATA_USAGE.json','production_01\\review\\MANIFEST.json'}
    for name,old in m['files'].items():
        new=sha(H/name)
        assert old==new or name in allowed, f'Unexpected modification: {name}'
        m['files'][name]=new
    for name in added:m['files'][name]=sha(H/name)
    dump(path,m)
    for link in re.findall(r'(?:href|src)="([^"]+)"',s):
        if not link.startswith(('#','http')):assert (V/link).is_file(),link
    print(json.dumps({'delivery_files':len(m['files']),'local_tokens':usage['total_tokens'],'unexpected_modified_delivery_files':0,'training':False}))
if __name__=='__main__':main()
