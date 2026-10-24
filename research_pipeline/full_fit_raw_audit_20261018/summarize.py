"""Summarize completed full source audit; keep input sources and raw files unchanged."""
from pathlib import Path
from collections import Counter, defaultdict
import json, csv, hashlib, os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;RUN=ROOT/'full_02';OUT=ROOT/'review_02'
FIELDS=['level','xp','currentGold','totalGold','minionsKilled','jungleMinionsKilled','health_fraction','power_fraction','health','healthMax','attackDamage','abilityPower','armor','magicResist','attackSpeed','movementSpeed','x','y']
EVENTS=['kills','assists','elem_dragon','elder_dragon','baron','atakhan','herald','horde','outer_turret','inner_turret','base_turret','nexus_turret','inhibitor','plate','item_purchased','item_sold','item_undo','ward_placed','ward_kill','level_up']
PATCHES=['15.18','15.19','15.20'];C0='#a8afb8';C1='#287d76';CQ='#b45750'
plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':10,'axes.spines.right':False,'axes.spines.top':False,'savefig.dpi':150})
def dump(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def ranges_n(rs):return sum(b-a+1 for a,b in rs if a>=0)
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def save(fig,name):fig.savefig(OUT/(name+'.png'),bbox_inches='tight');fig.savefig(OUT/(name+'.svg'),bbox_inches='tight');plt.close(fig)
def main():
 complete=json.loads((RUN/'RUN_COMPLETE.json').read_text(encoding='utf-8'));assert complete['completed'] and complete['all_hash_eligible_scanned'] and not complete['limited_smoke']
 OUT.mkdir(exist_ok=True);inventory=json.loads((RUN/'INVENTORY.json').read_text(encoding='utf-8'));progress=json.loads((RUN/'PROGRESS.json').read_text(encoding='utf-8'))
 rows=[];known=[];unresolved=[];dispositions=Counter();reasons=Counter();diag=Counter();fieldmatches=Counter();eventmatches=Counter();fieldcells=Counter();eventcells=Counter();champbefore=Counter();champafter=Counter();ambigchamp=Counter();mids=set();sources=0
 roster_missing=Counter();roster_observed=Counter()
 old={}
 src=ROOT.parent/'multiwindow_r2/production_01/SOURCE_LEDGER.jsonl'
 for s in src.open(encoding='utf-8'):
  r=json.loads(s);old[r['match_id']]=(r['detail_sha256'],r['timeline_sha256'])
 oldcompared=0;oldmismatch=[]
 with (OUT/'MATCH_DISPOSITIONS.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.writer(f);w.writerow(['match_id','patch','scope','disposition','hard_reasons','population_reasons','field_mask_reason_count','event_mask_reason_count','terminal_verified','duration_s'])
  for s in (RUN/'MATCH_AUDIT.jsonl').open(encoding='utf-8'):
   r=json.loads(s);mid=r['match_id'];assert mid not in mids; mids.add(mid);assert 0<=r['bucket']<=79
   dispositions[r['disposition']]+=1;reasons.update(r['hard_reasons']+r['population_reasons']);diag.update(r['diagnostics']);sources+=r.get('detail_bytes',0)+r.get('timeline_bytes',0)
   if mid in old and 'detail_sha256' in r and 'timeline_sha256' in r:
    oldcompared+=1
    if old[mid]!=(r['detail_sha256'],r['timeline_sha256']):oldmismatch.append(mid)
   fieldmatches.update(r['field_masks'].keys());eventmatches.update(r['event_masks'].keys())
   for k,v in r['field_masks'].items():fieldcells[k]+=ranges_n(v)
   for k,v in r['event_masks'].items():eventcells[k]+=ranges_n(v)
   if r.get('scope')=='UNRESOLVED':unresolved.append(r)
   else:
    assert r['patch'] in PATCHES
    known.append(r)
    cc=[str(c) for c in r.get('champion_ids',[])];champbefore.update(cc)
    (roster_observed if len(cc)==10 else roster_missing).update([r['disposition']])
    if r['disposition'].startswith('RETAIN'):champafter.update(cc)
    # Do not infer victim from broad champion roster: identify precisely in follow-up source examples.
    if 'kills:ambiguous_same_timestamp_victim' in r['event_masks']:ambigchamp.update(cc)
   m=r['metrics'];w.writerow([mid,r['patch'],r.get('scope'),r['disposition'],'|'.join(r['hard_reasons']),'|'.join(r['population_reasons']),len(r['field_masks']),len(r['event_masks']),m.get('terminal_verified'),m.get('duration_s')])
 retained=[r for r in known if r['disposition'].startswith('RETAIN')];excluded=[r for r in known if not r['disposition'].startswith('RETAIN')]
 # Validate previous sample membership independently of the expected rounded full-corpus estimate.
 assignments=[json.loads(s) for s in Path(os.environ['LOL_SPLIT_ASSIGNMENTS_PATH']).open(encoding='utf-8')]
 oldfit={r['match_id'] for r in assignments if r['v5_split_role'].startswith('FIT_')};absent=sorted(oldfit-mids)
 assert not absent and not oldmismatch
 scanned=len(known)+len(unresolved)+sum(progress['other_patches'].values());assert scanned==inventory['hash_eligible_unique_keys']
 bypatch={p:dict(Counter(r['disposition'] for r in known if r['patch']==p)) for p in PATCHES}
 metrics={}
 for name in ['duration_s','n_kills','final_gold_difference','max_abs_frame_gold_change']:
  metrics[name]={}
  for label,rr in [('before',known),('after',retained)]:
   v=np.array([r['metrics'][name] for r in rr if r['metrics'].get(name) is not None],float)
   metrics[name][label]={'n':len(v),'quantiles':{str(q):float(np.quantile(v,q)) for q in [0,.01,.25,.5,.75,.99,1]} if len(v) else {}}
 counts={'known_fit_candidates':len(known),'unresolved_scope':len(unresolved),'retained':len(retained),'known_fit_quarantine':sum(r['disposition']=='QUARANTINE' for r in known),'population_excluded':sum(r['disposition']=='EXCLUDE_POPULATION' for r in known),'dispositions':dict(dispositions),'by_patch':bypatch,'reasons_match_counts_overlapping':dict(reasons),'diagnostic_cell_or_event_counts':dict(diag),'field_mask_match_counts':dict(fieldmatches),'event_mask_match_counts':dict(eventmatches),'field_mask_cells':dict(fieldcells),'event_mask_events':dict(eventcells),'source_bytes_read':sources,'old_r2_source_hashes_compared':oldcompared,'old_r2_source_mismatches':oldmismatch,'existing_fit_assignments_found':len(oldfit),'existing_fit_assignments_absent':absent,'metrics':metrics,'historical_count_is_rounded_estimate':True,'new_training_ledger_created':False}
 dump('SUMMARY.json',counts);dump('QUARANTINE_AND_UNRESOLVED.json',[r for r in known+unresolved if r['disposition']=='QUARANTINE']);dump('SOURCE_CHECKS.json',{'all_scope_counts_reconcile':True,'canonical_ids_unique':True,'all_buckets_fit':True,'old_fit_membership_preserved':True,'old_r2_hash_matches':oldcompared,'old_r2_hash_mismatches':oldmismatch,'complete_run':complete})
 fig,axs=plt.subplots(1,2,figsize=(12,4.7),layout='constrained');x=np.arange(3)
 nb=[sum(r['patch']==p for r in known) for p in PATCHES];na=[sum(r['patch']==p for r in retained) for p in PATCHES]
 for off,vals,lab,col in [(-.18,nb,'감사 전 FIT 후보',C0),(.18,na,'격리·모집단 제외 후',C1)]:
  b=axs[0].bar(x+off,vals,.36,label=lab,color=col);axs[0].bar_label(b,fmt='%d',fontsize=8)
 axs[0].set(xticks=x,xticklabels=PATCHES,ylabel='경기 수');axs[0].legend(fontsize=8,loc='lower left')
 labs=['보존','모집단 제외','확정 FIT 격리','패치 미확인'];vv=[len(retained),counts['population_excluded'],counts['known_fit_quarantine'],len(unresolved)]
 b=axs[1].barh(labs,vv,color=[C1,C0,CQ,'#b68a35']);axs[1].bar_label(b,fmt='%d',padding=4);axs[1].set(xlabel='경기 수');axs[1].set_xlim(0,max(vv)*1.18)
 fig.suptitle('01 | 전체 FIT 원본: 실제 hash 배정과 처분',fontsize=15);save(fig,'01_population')
 fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
 for ax,code,title in [(axs[0],'duration_s','검증된 종료 시각 (분)'),(axs[1],'n_kills','원본 CHAMPION_KILL 기록 수')]:
  for rr,lab,col in [(known,'제외 전',C0),(retained,'제외 후',C1)]:
   a=np.sort(np.array([r['metrics'][code] for r in rr if r['metrics'].get(code) is not None],float));a=a/60 if code=='duration_s' else a;ax.plot(a,np.arange(1,len(a)+1)/len(a),label=f'{lab} n={len(a):,}',color=col,lw=2)
  ax.set(xlabel=title,ylabel='경기 누적 비율');ax.legend(fontsize=8)
 fig.suptitle('02 | 길이·킬 분포 전후 — 전체 범위, 극단값 절단 없음',fontsize=15);save(fig,'02_duration_kills')
 fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
 for ax,code,title in [(axs[0],'final_gold_difference','마지막 관측 프레임 팀 골드 차이'),(axs[1],'max_abs_frame_gold_change','인접 관측 프레임의 최대 골드 차 변화 절댓값')]:
  for rr,lab,col in [(known,'제외 전',C0),(retained,'제외 후',C1)]:
   a=np.sort([r['metrics'][code] for r in rr if r['metrics'].get(code) is not None]);ax.plot(a,np.arange(1,len(a)+1)/len(a),label=f'{lab} n={len(a):,}',color=col,lw=2)
  ax.set(xlabel=title,ylabel='경기 누적 비율');ax.legend(fontsize=8)
 fig.suptitle('03 | 골드 분포 전후 — 교전 라벨이 아닌 원본 경기 진단',fontsize=15);save(fig,'03_gold_extremes')
 epicnames=['원소 용','장로','바론','아타칸','전령','유충'];rare={}
 fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained');xx=np.arange(6)
 for off,rr,lab,col in [(-.18,known,'제외 전',C0),(.18,retained,'제외 후',C1)]:
  support=[r for r in rr if 'epic_acquisition_counts' in r['metrics']]
  pos=np.array([sum(r['metrics']['epic_acquisition_counts'][j]>0 for r in support) for j in range(6)])
  rates=pos/max(1,len(support))*100;rare[lab]={'denominator':len(support),'positive_matches':pos.tolist(),'rate_pct':rates.tolist()}
  axs[0].bar(xx+off,rates,.36,color=col,label=f'{lab} n={len(support):,}')
 axs[0].set(xticks=xx,xticklabels=epicnames,ylabel='귀속 확인 획득 기록이 있는 경기 (%)');axs[0].legend(fontsize=8)
 removed=np.array(rare['제외 전']['positive_matches'])-np.array(rare['제외 후']['positive_matches']);b=axs[1].bar(xx,removed,color=CQ);axs[1].bar_label(b,fmt='%d');axs[1].set(xticks=xx,xticklabels=epicnames,ylabel='제외된 경기 중 획득 기록 보유 수')
 axs[1].set_ylim(0,max(1,float(removed.max())*1.2))
 fig.suptitle('04 | 희귀 오브젝트 보존 — 미귀속 사건은 획득 없음으로 확정하지 않음',fontsize=14);save(fig,'04_objectives');dump('OBJECTIVE_DISTRIBUTIONS.json',rare)
 features={};fig,axs=plt.subplots(1,2,figsize=(13,6),layout='constrained')
 rr=[r for r in retained if 'supported_player_cells' in r['metrics']];den=sum(r['metrics']['player_cells_per_feature'] for r in rr);nums=np.sum([r['metrics']['supported_player_cells'] for r in rr],axis=0);missing=100*(1-nums/den)
 b=axs[0].barh(FIELDS,missing,color='#b68a35');axs[0].bar_label(b,fmt='%.3f',fontsize=7);axs[0].invert_yaxis();axs[0].set(xlabel=f'미지원 셀 (%) / 필드당 {den:,} 셀',xlim=(0,max(missing)*1.35+.1))
 ee=eventmatches.most_common(10);b=axs[1].barh([x[0] for x in ee],[x[1] for x in ee],color='#637d97');axs[1].bar_label(b,fmt='%d',fontsize=8);axs[1].invert_yaxis();axs[1].set(xlabel='원본 전체에서 해당 사유가 한 번 이상 있는 경기 수')
 fig.suptitle('05 | 미지원은 경기 삭제와 다름 — 사건 마스크는 향후 창과 교차 적용',fontsize=14);save(fig,'05_masks')
 dump('PLAYER_SUPPORT.json',{'retained_matches_with_frames':len(rr),'denominator_per_feature':den,'supported_cells':dict(zip(FIELDS,nums.astype(int).tolist()))})
 hh=np.load(RUN/'FEATURE_HISTOGRAMS.npz');fig,axs=plt.subplots(6,3,figsize=(13,19),layout='constrained');tail={}
 for j,(name,ax) in enumerate(zip(FIELDS,axs.flat)):
  edges=hh[f'edges_{j}'];low=edges[1];high=edges[-2];finitecenters=(edges[1:-2]+edges[2:-1])/2
  tail[name]={}
  for prefix,lab,col in [('before','전',C0),('after','후',C1)]:
   c=hh[f'{prefix}_{j}'];total=int(c.sum());shown=c[1:-1].copy();boundary=0;upper=int(c[-1])
   if name in ('health_fraction','power_fraction') and high==1:
    boundary=upper;shown[-1]+=boundary;upper=0
   ax.stairs(shown/max(1,total)*100,edges[1:-1],color=col,label=lab,lw=1.5)
   tail[name][prefix]={'total_supported_cells':total,'below_display_range':int(c[0]),'unplotted_upper_bin':upper,'upper_bin_definition':'value >= upper edge; fractions equal to 1 folded into last visible bin','boundary_one_included':boundary,'range':[float(low),float(high)]}
  ax.set(title=f'{name} | 후 별도 bin {tail[name]["after"]["below_display_range"]+tail[name]["after"]["unplotted_upper_bin"]:,}',ylabel='전체 지원 셀 중 bin 비율 (%)');ax.legend(fontsize=7)
 fig.suptitle('06 | 입력 18개: 동일 bin의 전후 비교\n별도 bin은 하한 미만·상한 이상 값; 비율 1은 마지막 표시 bin에 포함. 이상치 판정 아님.',fontsize=14);save(fig,'06_features');dump('FEATURE_TAIL_COUNTS.json',tail)
 dump('CHAMPION_DENOMINATORS.json',{'observed_roster_matches_by_disposition':dict(roster_observed),'missing_roster_matches_by_disposition':dict(roster_missing),'conditional_on_roster_observed':True})
 # Champion-level exclusion rates: conditional on observed rosters, no outcome threshold.
 cmap={}
 for pp in PATCHES:
  dd=json.loads((ROOT.parent/f'multiwindow_r2/knowledge_r2/{pp}/sources/championFull_{pp}.1.json').read_bytes())
  cmap.update({str(v['key']):v['id'] for v in dd['data'].values()})
 champ=[]
 for c,n in champbefore.items():champ.append({'champion_id':c,'name':cmap.get(c,c),'before_slots':n,'after_slots':champafter[c],'removed_slots':n-champafter[c],'removed_pct':100*(n-champafter[c])/n})
 champ.sort(key=lambda x:x['removed_pct'],reverse=True);dump('CHAMPION_RETENTION.json',champ)
 fig,ax=plt.subplots(figsize=(11,5),layout='constrained');xx=[r['before_slots'] for r in champ];yy=[r['removed_pct'] for r in champ];ax.scatter(xx,yy,s=20,color=C1,alpha=.7)
 for r in champ[:7]:ax.annotate(r['name'],(r['before_slots'],r['removed_pct']),fontsize=8)
 ax.set(xlabel='제외 전 챔피언 출현 슬롯 수 (동일 경기 1 슬롯)',ylabel='제외된 슬롯 비율 (%)',title='07 | 희귀 챔피언도 포함한 전후 보존 감사');save(fig,'07_champions')
 print(json.dumps({k:counts[k] for k in ['known_fit_candidates','retained','known_fit_quarantine','population_excluded','unresolved_scope','reasons_match_counts_overlapping','existing_fit_assignments_found','old_r2_source_hashes_compared']},ensure_ascii=False))
 inputs={str(p.relative_to(ROOT)):digest(p) for p in [RUN/'MATCH_AUDIT.jsonl',RUN/'FEATURE_HISTOGRAMS.npz',RUN/'RUN_COMPLETE.json',Path(__file__)]}
 names=['SUMMARY.json','MATCH_DISPOSITIONS.csv','QUARANTINE_AND_UNRESOLVED.json','SOURCE_CHECKS.json','OBJECTIVE_DISTRIBUTIONS.json','PLAYER_SUPPORT.json','FEATURE_TAIL_COUNTS.json','CHAMPION_RETENTION.json','CHAMPION_DENOMINATORS.json']
 for stem in ['01_population','02_duration_kills','03_gold_extremes','04_objectives','05_masks','06_features','07_champions']:
  names.extend([stem+'.png',stem+'.svg'])
 dump('AGGREGATION_COMPLETE.json',{'complete':True,'inputs':inputs,'outputs':{n:digest(OUT/n) for n in names}})
if __name__=='__main__':main()
