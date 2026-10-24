"""Create inspectable notebook and source receipt from recorded inventory, no data scan."""
import json,pathlib
ROOT=pathlib.Path(__file__).resolve().parent
def load(n):return json.loads((ROOT/n).read_text(encoding='utf-8'))
def write(n,d):(ROOT/n).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
a=load('MULTIPATCH_DATA_AUDIT.json');s=load('MULTIPATCH_CACHE_SUPPORT_REVIEW.json');r=load('MULTIPATCH_EXPERIMENT_REGISTRY_V5.json')
rows=[{'source':name,'patch':patch,'indexed_matches':g['indexed_matches'],'complete_triplets':g['complete_file_triplets']} for name,v in a['indexed_caches'].items() for patch,g in v['groups'].items()]
receipt={'schemaVersion':1,'items':[
 {'id':'patch-holdings','title':'패치별 캐시와 원본 경로 보유 확인','queries':[{'id':'current-index-census','source':{'label':'현재 패치 인덱스·파일명 감사','files':[{'label':'MULTIPATCH_DATA_AUDIT.json'},{'label':'audit_multipatch.py'}],'executedAt':a['captured_utc'],'metricDefinitions':[{'label':'캐시 보유 경기','definition':'현재 인덱스의 경기 수와 meta/events/NPZ 세 파일이 모두 존재하는 경기 수.'}],'filters':['기존 감사에서 확인한 네 개 캐시 인덱스 전 항목.','알려진 아홉 개 원본 detail/timeline 경로의 파일명.'],'caveats':['파일 존재는 실제 피처·라벨 적격성이나 내용 무결성을 보증하지 않는다.','이번에는 NPZ와 원본 경기 payload를 열지 않았고 meta는 패치별 세 개만 추가 점검했다.']},'columns':['source','patch','indexed_matches','complete_triplets'],'rows':rows,'methods':[{'language':'python','code':"complete = sum(all(mid+'.'+ext in filenames for ext in ['meta.json','events.json','npz']) for mid in indexed_match_ids)"}]}]},
 {'id':'readiness-exposure','title':'학습 준비와 미노출 시험은 아직 확인되지 않음','queries':[{'id':'source-acceptance','source':{'label':'기존 캐시 지원 보고서와 평가 노출 장부','files':[{'label':'MULTIPATCH_CACHE_SUPPORT_REVIEW.json'},{'label':'EXTERNAL_DATA_INVENTORY_20261009.json'}],'metricDefinitions':[{'label':'새 설계 준비 상태','definition':'보유 자료와 별개로 원본 시각·누락 provenance, 새 producer 및 패치별 지식·지형 게이트가 필요한 상태.'}],'caveats':['40,000개 변환 캐시의 기존 감사에서 source clock provenance가 UNKNOWN이고 preparation training_ready는 false다.','15.22 및 16.x는 이전 학습 또는 평가 사용 이력이 있어 새 분할에서도 pristine test로 부르지 않는다.']},'rows':[{'finding':'15.18–15.22 cache source clock unknown','matches':s['cache_audit']['reason_counts']['source_clock_provenance_unknown']},{'finding':'현재 설계에서 실행한 모델 학습','matches':0}],'columns':['finding','matches']}]}]}
write('multipatch-answer-sources.json',receipt)
code="""import json, hashlib
from pathlib import Path
root = Path.cwd()  # run with this notebook's folder as working directory
audit = json.loads((root/'MULTIPATCH_DATA_AUDIT.json').read_text(encoding='utf-8'))
rows = []
for name, source in audit['indexed_caches'].items():
    for patch, group in source['groups'].items():
        rows.append((name, patch, group['indexed_matches'], group['complete_file_triplets']))
for row in rows:
    print(*row, sep=' | ')
print('Rows above are file inventory, not training-eligible sample sizes.')
"""
display='\n'.join(' | '.join(str(x) for x in row.values()) for row in rows)+'\nRows above are file inventory, not training-eligible sample sizes.\n'
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3'}},'cells':[
 {'cell_type':'markdown','id':'scope','metadata':{},'source':['# 다중 패치 인벤토리 검토\n','읽기 전용 감사 결과를 재검토한다. 원본 payload/NPZ를 읽거나 학습하지 않는다. 전체 파일명 감사 코드는 audit_multipatch.py이며, 이 노트북은 기록된 JSON의 집계만 표시한다.\n','아래 출력은 동일 JSON에서 생성한 기록 요약이다. 새 감사가 필요할 때만 audit_multipatch.py를 별도 실행한다.']},
 {'cell_type':'code','id':'inventory','metadata':{},'execution_count':None,'source':code.splitlines(True),'outputs':[{'output_type':'stream','name':'stdout','text':display.splitlines(True)}]},
 {'cell_type':'markdown','id':'limits','metadata':{},'source':['## 해석 한계\n','캐시40k의 원본 시각/정적ID provenance는 기존 감사에서 UNKNOWN이었다. 새 producer·전체 스키마·지식/지형·split/matching 게이트는 미완료다. 15.22 및 16.x에는 이전 연구 노출이 있다. 상세 설계는 MULTIPATCH_PROTOCOL_KO.md를 따른다.']} ]}
write('MULTIPATCH_AUDIT_REVIEW.ipynb',nb)
print('Receipt and notebook generated from recorded audit; no data scan or training.')
