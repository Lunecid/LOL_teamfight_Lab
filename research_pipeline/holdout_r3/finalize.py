"""Package verified held-out ledgers and explicit remaining training gates."""
from pathlib import Path
import json,hashlib,re,importlib.util
from collections import Counter
from split import roles_by_macro,fit_role_names,heldout_role_names
import numpy as np
import dataset
H=Path(__file__).resolve().parent;R=H.parent;O=H/'production_01';V=O/'review'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def main():
 v=json.loads((O/'VALIDATION.json').read_text(encoding='utf-8'));assert v['pass'] and v['full_population']
 c=json.loads((R/'multiwindow_r3/TRAINING_DATA_CONTRACT.json').read_text(encoding='utf-8'))
 groups=roles_by_macro()
 c.update(id='MW60_FRAME_r3_holdout',fit_roles=groups.get('FIT',[]),stop_roles=groups.get('STOP',[]),dev_roles=groups.get('DEV',[]),cal_roles=groups.get('CAL',[]),audit_roles=groups.get('AUDIT',[]),test_roles=groups.get('TEST',[]),source_audit='audit_01/MATCH_AUDIT.jsonl',source_validation_sha256=sha(O/'VALIDATION.json'),holdout_note='Held-out roles follow the local split policy; this dataset is not used for training or tuning.',fit_companion='../multiwindow_r3/TRAINING_DATA_CONTRACT.json',readiness_scope='Observed-target data only. Shared feature adapters, trainer, scientific ModelCards and semantic/terrain heads remain pending.')
 dump(H/'TRAINING_DATA_CONTRACT.json',c)
 counts=Counter();rows=0
 for x,y,m in dataset.iter_shards(O,sorted(heldout_role_names()),L=3):
  assert set(x)==set(dataset.FEATURE_KEYS)|{'history_present'}
  assert x['player_raw'].shape[1:]==(3,10,18)
  assert not {'endpoint_ms','actual_elapsed_ms','selection_supported','first_kill_ms','last_kill_ms','patch','split_role','match_id','component_id'}&set(x)
  assert np.array_equal(y['economic_class'][y['economic_supported']],y['economic_sign'][y['economic_supported']]+1)
  assert (y['economic_class'][~y['economic_supported']]==-1).all()
  counts.update(m['split_role'].tolist());rows+=len(m['row_id'])
 for length in (1,5):
  x,y,m=next(dataset.iter_shards(O,list(counts),L=length));assert x['player_raw'].shape[1]==length
 try:next(dataset.iter_shards(O,sorted(fit_role_names())[:1]))
 except ValueError:pass
 else:raise AssertionError('Held-out reader accepted a FIT role')
 spec=importlib.util.spec_from_file_location('prior_fit_loader',R/'multiwindow_r3/dataset.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 try:next(mod.iter_shards(R/'multiwindow_r3/production_01',sorted(heldout_role_names())[:1]))
 except ValueError:pass
 else:raise AssertionError('Original FIT reader accepted TEST')
 res=json.loads((V/'RESULTS.json').read_text(encoding='utf-8'));assert rows==v['queries_checked']
 for r in res['support']:
  assert 0<=r['supported']<=r['denominator'];assert abs(r['pct']-100*r['supported']/r['denominator'])<1e-10
 for r in res['rare_events']:assert r['positive_unique_matches']<=r['positive_windows']<=r['supported_windows']<=r['all_windows']
 db=H/'DB_INDEX_RECONCILIATION.json'
 if not db.exists():dump(db,{'status':'PENDING_LOCAL_AUTHENTICATION','database':'configured_local_catalog','host':'configured_by_operator','port':None,'used_for_index_selection':False,'source_of_ledger':'read-only raw filesystem inventory with immutable source hashes','note':'No DB comparison is claimed unless the operator supplies a local authenticated connection.'})
 links=[]
 for link in re.findall(r'(?:href|src)="([^"]+)"',(V/'index.html').read_text(encoding='utf-8')):
  if link.startswith(('#','http')):continue
  assert (V/link).is_file(),link;links.append(link)
 usage=Counter();calls=[]
 for p in sorted(H.glob('strata_*.json')):
  x=json.loads(p.read_text(encoding='utf-8'));u=x.get('usage',{});usage.update({k:u.get(k,0) for k in ('prompt_tokens','completion_tokens','total_tokens')});calls.append({'file':p.name,'usage':u,'finish_reason':x.get('finish_reason')})
 dump(H/'STRATA_USAGE.json',{'local_usage':dict(usage),'calls':calls,'measured_cloud_subscription_savings':None,'note':'Local API tokens only; character reduction is not quota or billing savings.'})
 checked={'pass':True,'all_loaded_rows':rows,'roles':dict(counts),'history_lengths_checked':[1,3,5],'original_fit_loader_rejects_test':True,'holdout_loader_rejects_fit':True,'outcome_metadata_excluded_from_inputs':True,'visual_local_links':len(links),'distribution_denominators_checked':True,'training':False}
 dump(O/'LOADER_DELIVERY_CHECKS.json',checked)
 text=['# 검증·시험 원장 생성 및 검토 결과','',f"원본 독립 검산: {v['matches_covered']:,}경기 / {v['queries_checked']:,}교전. 기존 원장 변경 없음. 학습 미실행.",'','| 분할 | 원본 확인 경기 | 유지 경기 | 교전 경기 | 교전 행 |','|---|---:|---:|---:|---:|']
 for g,p in res['population'].items():
  if g.startswith('FIT_'):continue
  text.append(f"| {g} | {p['raw_known_matches']:,} | {p['retained_matches']:,} | {p['query_matches']:,} | {p['query_rows']:,} |")
 text+=['','지원율은 SUPPORT_BY_ROLE_PATCH_HORIZON.csv, 희귀 사건의 창·고유 경기 분모는 EVENT_SUPPORT_AND_RARITY.csv에서 확인한다. 원본의 제외 전후 희귀 사건 수도 RESULTS.json에 별도로 있다.','',f"기존 r2 겹치는 표본의 교전 경계: {v['prior_r2_holdout_queries_exactly_preserved']:,}행 유지. 누락/추가 {v['prior_query_missing']}/{v['prior_query_extra']}.",'','의미 태그·지형·정확한 버프·인과 추가 가치 U는 활성화하지 않았다. 전체 설계의 구현 공백과 실행 전 잠금 항목은 ../../END_TO_END_REVIEW_KO.md를 참조한다.','',f"Strata 로컬 사용량: {usage['total_tokens']:,} tokens. 구독 절감량은 측정하지 않았다."]
 (V/'RESULTS_KO.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
 files=[p for p in H.iterdir() if p.is_file() and p.suffix in ('.py','.md','.json')]+[O/n for n in ['BUILD_SUMMARY.json','SOURCE_LEDGER.jsonl','VALIDATION.json','ORIGINAL_HASHES_BEFORE.json','ORIGINAL_PRESERVATION.json','LEAKAGE_CHECKS.json','LOADER_DELIVERY_CHECKS.json']]+[H/'audit_01/MATCH_AUDIT.jsonl',H/'audit_01/VALIDATION.json']
 dump(V/'MANIFEST.json',{'files':{p.name:sha(p) for p in V.iterdir() if p.is_file() and p.name!='MANIFEST.json'},'training':False})
 files += [V/'MANIFEST.json']
 dump(O/'DELIVERY_MANIFEST.json',{'files':{str(p.relative_to(H)):sha(p) for p in files},'training_performed':False,'full_pipeline_ready':False,'review_required_before_training':True})
 print(json.dumps(checked,ensure_ascii=False));print('local_tokens',usage['total_tokens'])
if __name__=='__main__':main()
