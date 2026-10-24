"""Validated FIT/holdout audit comparisons. Descriptive only, no training/tuning."""
from pathlib import Path
from collections import Counter,defaultdict
import json,hashlib,csv,html,os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from split import roles,bucket
H=Path(__file__).resolve().parent;R=H.parent;O=H/'production_01';V=O/'review'
LABELS=['FIT','STOP','DEV','CAL','AUDIT','TEST'];COLORS=['#277e75','#bb8135','#487fa8','#9b6496','#6c7841','#b35e52']
EVENTS=['kills','assists','elem_dragon','elder_dragon','baron','atakhan','herald','horde','outer_turret','inner_turret','base_turret','nexus_turret','inhibitor','plate','item_purchased','item_sold','item_undo','ward_placed','ward_kill','level_up']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def lines(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines()]
def load(run):
 v=json.loads((run/'VALIDATION.json').read_text(encoding='utf-8'));assert v['pass'] and v['full_population'];assert sha(run/'VALIDATED_DISTRIBUTIONS.npz')==v['distributions_sha256']
 with np.load(run/'VALIDATED_DISTRIBUTIONS.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
 if 'macro_role' not in d:d['macro_role']=np.full(len(d['row_id']),'FIT');d['split_role']=np.char.add('FIT_',d['patch'])
 return d,v
def main():
 V.mkdir(exist_ok=True);fit,fv=load(R/'multiwindow_r3/production_01');hold,hv=load(O)
 assert EVENTS==json.loads((O/'BUILD_SUMMARY.json').read_text(encoding='utf-8'))['event_fields']
 d={k:np.concatenate([fit[k],hold[k]]) for k in hold}
 oldaudit=lines(R/'full_fit_raw_audit_20261018/full_02/MATCH_AUDIT.jsonl');newaudit=lines(H/'audit_01/MATCH_AUDIT.jsonl')
 oldids={r['match_id']:r for r in oldaudit}
 for r in newaudit:
  if r['match_id'] in oldids:
   assert roles(r.get('patch'),r['bucket'])[1]=='OTHER' and roles(oldids[r['match_id']].get('patch'),r['bucket'])[1]=='OTHER'
 audit=list({r['match_id']:r for r in oldaudit+newaudit}.values())
 source=lines(R/'multiwindow_r3/production_01/SOURCE_LEDGER.jsonl')+lines(O/'SOURCE_LEDGER.jsonl')
 known={r['match_id']:r for r in audit if roles(r.get('patch'),r['bucket'])[1] in LABELS}
 assert len(source)==len({r['match_id'] for r in source})
 violations=[];payloads={k:{} for k in ('detail_sha256','timeline_sha256')}
 for r in source:
  role=roles(r['patch'],bucket(r['match_id']))[1];assert role in LABELS
  for k,seen in payloads.items():
   key=r[k]
   if key in seen and seen[key][1]!=role:violations.append([k,seen[key],[r['match_id'],role]])
   seen[key]=(r['match_id'],role)
 assert not violations
 match_overlap=len(set(fit['match_id'])&set(hold['match_id']))
 query_unique=len(set(d['row_id']))==len(d['row_id'])
 leakage={'passed':match_overlap==0 and not violations and query_unique,'match_id_overlap':match_overlap,'cross_split_identical_raw_payloads':len(violations),'split_assignment':'frozen salt/hash/patch mapping; partition validation is recomputed from supplied ledgers','query_identity_unique':query_unique,'input_time':'all emitted queries raw-recomputed; history and event prefix <=t0','heldout_fit_prohibition':'no scaler, vocabulary fit, model, threshold or calibration fitted in this task','test_exposure':'Outcome/support summaries are exploratory, not an untouched/unseen TEST claim','player_disjoint':'not asserted: this is match- and patch-split, repeated players may occur','population_limit':'retrospectively selected fresh first-kill components; not online fight occurrence detection'}
 assert leakage['query_identity_unique'];dump(O/'LEAKAGE_CHECKS.json',leakage)
 pop={};support=[];rare=[];classes=[];events=[];timing=[]
 masks={'gold_both':d['gold_supported'].all(2),'epics_all':d['cum_event_supported'][:,:, :,2:8].all((2,3)),'all_20_events':d['cum_event_supported'].all((2,3)),'terminal':d['terminal_supported'],'hazard_at_risk':d['hazard_at_risk_supported'],'pareto':d['pareto_supported']}
 # The combined all-20 mask is diagnostic; never the global training row filter.
 groups=[(lab,None) for lab in LABELS]+[('FIT',p) for p in ['15.18','15.19','15.20']]
 for lab,patch in groups:
  key=lab if patch is None else lab+'_'+patch
  aa=[r for r in known.values() if roles(r['patch'],r['bucket'])[1]==lab and (patch is None or r['patch']==patch)]
  ss=[r for r in source if roles(r['patch'],bucket(r['match_id']))[1]==lab and (patch is None or r['patch']==patch)]
  sel=(d['macro_role']==lab)&((d['patch']==patch) if patch else True);n=int(sel.sum());disp=Counter(r['disposition'] for r in aa)
  pop[key]={'patch':patch or sorted({r['patch'] for r in aa}),'raw_known_matches':len(aa),'retained_matches':len(ss),'no_query_matches':sum(r['n_queries']==0 for r in ss),'query_matches':len(set(d['match_id'][sel])),'query_rows':n,'selection_supported_rows':int(d['selection_supported'][sel].sum()),'dispositions':dict(disp)}
  for h in range(3):
   for target,mm in masks.items():
    count=int(mm[sel,h].sum());support.append({'group':key,'patch':patch or str(pop[key]['patch']),'horizon_s':60*(h+1),'target':target,'supported':count,'denominator':n,'pct':100*count/n if n else None})
   for j,name in enumerate(EVENTS):
    ok=sel&d['cum_event_supported'][:,h,:,j].all(1);positive=ok&(np.nansum(d['cum_team_events'][:,h,:,j],axis=1)>0);nn=int(ok.sum());pp=int(positive.sum())
    rec={'group':key,'horizon_s':60*(h+1),'event':name,'supported_windows':nn,'all_windows':n,'support_pct':100*nn/n if n else None,'positive_windows':pp,'positive_pct':100*pp/nn if nn else None,'positive_unique_matches':len(set(d['match_id'][positive])),'supported_unique_matches':len(set(d['match_id'][ok]))}
    events.append(rec)
    if j in range(2,8):rare.append(rec)
   for label,k,valid in [('economic','economic_sign',masks['gold_both']),('terminal','terminal_class',masks['terminal']),('pareto','pareto_class',masks['pareto'])]:
    v=d[k][sel&valid[:,h],h];classes.append({'group':key,'horizon_s':60*(h+1),'label':label,'supported':len(v),'unsupported':n-len(v),'counts':{str(x):int((v==x).sum()) for x in np.unique(v)}})
   ev=d['actual_elapsed_ms'][sel,h];ok=ev>=0;timing.append({'group':key,'horizon_s':60*(h+1),'actual_elapsed_ms_quantiles':np.quantile(ev[ok],[0,.25,.5,.75,1]).tolist() if ok.any() else [],'early_terminal_windows':int((ev[ok]<60000*(h+1)).sum()),'late_frame_alignment_windows':int((ev[ok]>60000*(h+1)).sum())})
 def table(name,rows):
  with (V/name).open('w',encoding='utf-8-sig',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 table('SUPPORT_BY_ROLE_PATCH_HORIZON.csv',support);table('EVENT_SUPPORT_AND_RARITY.csv',events)
 table('DISPOSITIONS.csv',[{'match_id':r['match_id'],'patch':r.get('patch'),'split_role':roles(r.get('patch'),r['bucket'])[0],'disposition':r['disposition'],'reasons':';'.join(r['hard_reasons']+r['population_reasons'])} for r in audit if roles(r.get('patch'),r['bucket'])[1]!='FIT'])
 raw_epic={}
 mask_summary={}
 for lab in LABELS:
  aa=[r for r in known.values() if roles(r['patch'],r['bucket'])[1]==lab]
  fields=Counter();event_masks=Counter();rule_masks=Counter();diagnostics=Counter()
  for r in aa:
   if not r['disposition'].startswith('RETAIN'):continue
   for key,rr in r['field_masks'].items():fields[key]+=sum(b-a+1 for a,b in rr)
   for key,rr in r['event_masks'].items():event_masks[key]+=sum(b-a+1 for a,b in rr)
   for rm in r.get('rule_masks',[]):rule_masks[rm['reason']]+=1
   diagnostics.update(r['diagnostics'])
  mask_summary[lab]={'field_cells_by_reason':dict(fields),'event_ordinals_by_reason':dict(event_masks),'rule_onsets_by_reason':dict(rule_masks),'diagnostics':dict(diagnostics),'note':'Reasons can overlap; sums are not unique affected cells/events/matches.'}
  before=[r for r in aa if 'epic_acquisition_counts' in r['metrics']];after=[r for r in before if r['disposition'].startswith('RETAIN')]
  raw_epic[lab]={'available_before_matches':len(before),'available_after_matches':len(after),'unknown_before_matches':len(aa)-len(before),'before_counts':np.sum([r['metrics']['epic_acquisition_counts'] for r in before],axis=0).astype(int).tolist() if before else [0]*6,'after_counts':np.sum([r['metrics']['epic_acquisition_counts'] for r in after],axis=0).astype(int).tolist() if after else [0]*6}
  for tag,rr in [('before',before),('after',after)]:raw_epic[lab][tag+'_unattributed_event_counts']=np.sum([r['metrics']['unknown_event_counts'][2:8] for r in rr],axis=0).astype(int).tolist() if rr else [0]*6
 result={'population':pop,'support':support,'rare_events':rare,'label_distributions':classes,'timing':timing,'raw_epic_before_after':raw_epic,'mask_summary':mask_summary,'leakage':leakage,'unknown_scope_records':sum(roles(r.get('patch'),r['bucket'])[1]=='OTHER' for r in audit),'training_performed':False,'fit_validation_sha256':sha(R/'multiwindow_r3/production_01/VALIDATION.json'),'holdout_validation_sha256':sha(O/'VALIDATION.json')}
 result['label_codebook']={'economic_sign':{'-1':'RED_GAIN','0':'TIE','1':'BLUE_GAIN','-9':'UNKNOWN'},'economic_class_in_loader':{'0':'RED_GAIN','1':'TIE','2':'BLUE_GAIN','-1':'UNKNOWN'},'terminal':{'-1':'UNKNOWN','0':'ONGOING','1':'BLUE_END','2':'RED_END'},'pareto':{'-1':'UNKNOWN','0':'BLUE_DOMINATES','1':'RED_DOMINATES','2':'TRADEOFF','3':'NO_DIFFERENCE'}}
 dump(V/'RESULTS.json',result)
 plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'font.size':10})
 figs=[]
 def save(fig,name,title,note):
  fig.savefig(V/(name+'.png'),dpi=150,bbox_inches='tight');fig.savefig(V/(name+'.svg'),bbox_inches='tight');plt.close(fig);figs.append((name,title,note))
 fig,axs=plt.subplots(1,2,figsize=(13,4.6));x=np.arange(6)
 for label,offset,color in [('raw_known_matches',-.2,'#a5b1ba'),('retained_matches',.2,'#277e75')]:axs[0].bar(x+offset,[pop[g][label] for g in LABELS],.4,label=label,color=color)
 axs[0].set_xticks(x,LABELS);axs[0].set_title('원본 확인 경기와 유지 경기');axs[0].legend(fontsize=8)
 axs[1].bar(x,[pop[g]['query_rows'] for g in LABELS],color=COLORS);axs[1].set_xticks(x,LABELS);axs[1].set_title('교전 후보 행 수 (경기 수와 별개)')
 save(fig,'01_population','분할별 원본 → 유지 → 교전','짧은 경기 등 기존 모집단 제외와 손상 격리를 구분합니다. 교전이 없는 유지 경기도 원장에 남습니다.')
 fig,axs=plt.subplots(2,3,figsize=(13,8))
 for ax,target in zip(axs.flat,masks):
  m=np.array([[next(r['pct'] for r in support if r['group']==g and r['target']==target and r['horizon_s']==hh) for hh in (60,120,180)] for g in LABELS]);ax.imshow(m,vmin=0,vmax=100,cmap='YlGnBu',aspect='auto');ax.set_xticks(range(3),['60s','120s','180s']);ax.set_yticks(range(6),LABELS);ax.set_title(target)
  for i in range(6):
   for j in range(3):ax.text(j,i,f'{m[i,j]:.3f}%',ha='center',va='center',fontsize=8,color='white' if m[i,j]>65 else 'black')
 save(fig,'02_support','분할·예측창별 지원율','분모=각 분할의 모든 교전 후보. hazard는 관측 품질 외에 구간 시작 시 미종결 조건을 포함합니다. all_20_events는 진단용이며 전역 삭제 기준이 아닙니다.')
 fig,axs=plt.subplots(1,3,figsize=(16,7))
 for ax,hh in zip(axs,(60,120,180)):
  m=np.array([[next(r['support_pct'] for r in events if r['group']==g and r['event']==e and r['horizon_s']==hh) for g in LABELS] for e in EVENTS]);im=ax.imshow(m,vmin=0,vmax=100,cmap='YlGnBu',aspect='auto');ax.set_xticks(range(6),LABELS,rotation=45);ax.set_yticks(range(20),EVENTS if hh==60 else ['']*20);ax.set_title(f'{hh}초 사건별 지원율')
 fig.colorbar(im,ax=axs.ravel().tolist(),label='%');save(fig,'03_event_support','사건별 마스크','각 사건·팀·창의 지원 마스크를 사용합니다. 어시스트와 아이템 미지원이 골드·오브젝트 학습 행을 제거하지 않습니다.')
 fig,axs=plt.subplots(2,3,figsize=(14,9))
 for ax,e in zip(axs.flat,EVENTS[2:8]):
  m=np.array([[next(r['positive_pct'] for r in rare if r['group']==g and r['event']==e and r['horizon_s']==hh) for hh in (60,120,180)] for g in LABELS]);ax.imshow(m,cmap='YlOrBr',aspect='auto');ax.set_title(e+' 양성 창 비율');ax.set_xticks(range(3),['60s','120s','180s']);ax.set_yticks(range(6),LABELS)
  for i in range(6):
   for j in range(3):ax.text(j,i,f'{m[i,j]:.2f}%',ha='center',va='center',fontsize=8,color='white' if (m[i,j]-m.min())/max(1e-9,m.max()-m.min())>.6 else 'black')
 save(fig,'04_rare','희귀 사건과 오브젝트','분모=양 팀이 해당 사건을 지원하는 창. 색 범위는 오브젝트별로 다르므로 표시된 비율로 비교합니다. 창은 겹칠 수 있으므로 독립 사건 수가 아닙니다. CSV에 양성 창 수와 고유 경기 수를 함께 기록했습니다.')
 fig,axs=plt.subplots(1,3,figsize=(14,4.5))
 for h,ax in enumerate(axs):
  for g,c in zip(LABELS,COLORS):
   v=np.sort(d['net_gold'][(d['macro_role']==g)&masks['gold_both'][:,h],h]);ax.plot(v,np.arange(1,len(v)+1)/len(v),label=g,color=c)
  ax.set_title(f'{60*(h+1)}초 순골드 ECDF');ax.set_xlabel('BLUE - RED 추가 골드');ax.set_ylabel('누적 비율');ax.grid(alpha=.15)
 axs[0].legend(fontsize=8);save(fig,'05_gold','경제 이득 라벨 분포','전 범위 ECDF이며 극단값을 절단하지 않았습니다. 실제 골드 증가량의 팀 간 차이이지 오브젝트 인과 가치가 아닙니다.')
 fig,axs=plt.subplots(1,3,figsize=(14,4.5))
 for ax,field in zip(axs,['detail_duration_s','n_kills','final_gold_difference']):
  for g,c in zip(LABELS,COLORS):
   aa=[r for r in known.values() if roles(r['patch'],r['bucket'])[1]==g]
   for keep,ls in [(False,'--'),(True,'-')]:
    vals=np.sort([r['metrics'][field] for r in aa if r['metrics'].get(field) is not None and (not keep or r['disposition'].startswith('RETAIN'))]);
    if len(vals):ax.plot(vals,np.arange(1,len(vals)+1)/len(vals),color=c,ls=ls,label=g+(' 유지' if keep else ' 전'))
  ax.set_title(field+' 전후 ECDF');ax.grid(alpha=.15)
 axs[0].legend(fontsize=7,ncol=2);save(fig,'06_curation','제외 전후 분포','점선=측정 가능한 원본, 실선=유지 경기. 손상으로 측정 불가능한 값은 0으로 대체하지 않습니다. 극단값·희귀 사건 기반 제거는 없습니다.')
 fig,axs=plt.subplots(1,3,figsize=(14,4.5))
 for ax,hh in zip(axs,(60,120,180)):
  base=np.zeros(6)
  for code,name,color in [('0','ONGOING','#9ba9b3'),('1','BLUE_END','#4782ab'),('2','RED_END','#bc6f60')]:
   values=[]
   for g in LABELS:
    rr=next(r for r in classes if r['group']==g and r['horizon_s']==hh and r['label']=='terminal');values.append(100*rr['counts'].get(code,0)/rr['supported'] if rr['supported'] else 0)
   ax.bar(LABELS,values,bottom=base,label=name,color=color);base+=values
  ax.set_title(f'{hh}초 종료 상태');ax.tick_params(axis='x',rotation=45);ax.set_ylim(0,100)
 axs[0].legend(fontsize=8);save(fig,'07_terminal','종료 라벨','분모=종료 라벨 지원 창. 진행 중 0과 미확인 -1을 구분하며, 확인된 종결 뒤 구간은 hazard 위험집합에서 제외합니다.')
 pg=['FIT_15.18','FIT_15.19','FIT_15.20','STOP','DEV','CAL','AUDIT','TEST']
 fig,axs=plt.subplots(1,3,figsize=(15,6))
 for ax,target in zip(axs,['gold_both','epics_all','all_20_events']):
  m=np.array([[next(r['pct'] for r in support if r['group']==g and r['target']==target and r['horizon_s']==hh) for hh in (60,120,180)] for g in pg]);ax.imshow(m,vmin=0,vmax=100,cmap='YlGnBu',aspect='auto');ax.set_xticks(range(3),['60s','120s','180s']);ax.set_yticks(range(len(pg)),pg);ax.set_title(target)
  for i in range(len(pg)):
   for j in range(3):ax.text(j,i,f'{m[i,j]:.3f}%',ha='center',va='center',fontsize=8,color='white' if m[i,j]>65 else 'black')
 save(fig,'08_patch_support','패치별 지원율','FIT 15.18/15.19/15.20을 분리했습니다. STOP=15.20, DEV/CAL/AUDIT=15.21, TEST=15.22이며 분할 간 표본 재배정은 없습니다.')
 fig,axs=plt.subplots(2,3,figsize=(14,8))
 for j,ax in enumerate(axs.flat):
  for field,offset,color in [('before',-.18,'#a5b1ba'),('after',.18,'#277e75')]:
   vals=[1000*raw_epic[g][field+'_counts'][j]/max(1,raw_epic[g]['available_'+field+'_matches']) for g in LABELS];ax.bar(np.arange(6)+offset,vals,.36,label=field,color=color)
  ax.set_xticks(range(6),LABELS,rotation=35);ax.set_title(EVENTS[j+2]);ax.set_ylabel('관측 가능 원본 1,000경기당 획득 수')
 axs.flat[0].legend();save(fig,'09_raw_rare','원본 희귀 사건의 제외 전후','팀 귀속이 확인된 원본 획득 사건 집계이며 겹치는 교전 창의 합이 아닙니다. 귀속 불명 사건 수는 RESULTS.json에 별도로 기록했습니다. 분모는 사건 로그를 감사한 경기이며 파싱 불가 파일은 별도 격리 명단에 남습니다.')
 fig,axs=plt.subplots(1,2,figsize=(13,4.5))
 for g,c in zip(LABELS,COLORS):
  sel=d['macro_role']==g
  axs[0].hist((d['tau_ms'][sel]-d['t0_ms'][sel])/1000,bins=np.linspace(0,10,21),density=True,histtype='step',color=c,label=g)
  axs[1].hist((d['first_kill_ms'][sel]%60000)/1000,bins=np.arange(0,61),density=True,histtype='step',color=c,label=g)
 axs[0].set_title('선별된 교전의 프레임 나이');axs[0].set_xlabel('tau - t0 (초)');axs[1].set_title('선별된 첫 킬의 분 내 위상');axs[1].set_xlabel('first_kill mod 60s');axs[0].legend(fontsize=8)
 save(fig,'10_selection','fresh 조건과 시간 위상','모든 분할에서 같은 10초 이내 조건을 적용했습니다. 이 그림은 선별된 표본의 분포이며 전체 임의 시점 모집단의 분포로 일반화하지 않습니다.')
 header='<h1>검증·시험 r3 — 학습 전 점검</h1><p>기존 FIT와 분할을 보존했습니다. 원본 감사·원장 전수 재계산·누출 점검 결과이며 모델 학습은 수행하지 않았습니다.</p>'
 tablehtml='<table><tr><th>분할</th><th>원본 경기</th><th>유지 경기</th><th>교전 경기</th><th>교전 행</th></tr>'+''.join(f'<tr><td>{g}</td>'+''.join(f'<td>{pop[g][k]:,}</td>' for k in ['raw_known_matches','retained_matches','query_matches','query_rows'])+'</tr>' for g in LABELS)+'</table>'
 body=''.join(f'<section id="{n}"><h2>{title}</h2><p>{note}</p><img src="{n}.png" alt="{title}"></section>' for n,title,note in figs)
 links='<section><h2>근거 및 설계 검토</h2><p><a href="RESULTS.json">분모·라벨·희귀 사건 상세</a> · <a href="SUPPORT_BY_ROLE_PATCH_HORIZON.csv">분할·패치·창별 지원율</a> · <a href="EVENT_SUPPORT_AND_RARITY.csv">사건별 지원·양성 고유 경기</a> · <a href="DISPOSITIONS.csv">격리·제외 명단</a> · <a href="../LEAKAGE_CHECKS.json">누출 점검</a> · <a href="../VALIDATION.json">전수 재계산</a> · <a href="../../END_TO_END_REVIEW_KO.md">전체 설계 재검토</a> · <a href="../../DB_INDEX_RECONCILIATION.json">DB 대조</a></p></section>'
 page='<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Holdout r3 review</title><style>body{margin:0;background:#f3f5f6;color:#223746;font:16px/1.8 Malgun Gothic,system-ui}main{max-width:1180px;margin:auto;padding:26px}section{background:white;padding:20px;margin:24px 0;border:1px solid #dce2e6}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;background:white}td,th{padding:9px;text-align:right;border-bottom:1px solid #ddd}td:first-child,th:first-child{text-align:left}a{color:#226985}</style><main>'+header+tablehtml+body+links+'</main></html>'
 (V/'index.html').write_text(page,encoding='utf-8')
 dump(V/'MANIFEST.json',{'files':{p.name:sha(p) for p in V.iterdir() if p.is_file() and p.name!='MANIFEST.json'},'figures':len(figs),'training':False})
 print(json.dumps(pop,ensure_ascii=False))
if __name__=='__main__':main()
