"""Render only validated FIT results with explicit window and match denominators."""
from pathlib import Path
from collections import Counter
import os,json,html,hashlib,argparse,csv
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from validate import sha,dump
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':10,'axes.spines.right':False,'axes.spines.top':False,'savefig.dpi':150})
COLORS=['#287d76','#5b7999','#bd8743'];H=['60초','120초','180초'];PATCHES=['15.18','15.19','15.20'];EPICS=[2,3,4,5,6,7];NAMES=['원소 용','장로','바론','아타칸','전령','유충']

def main(run):
 val=json.loads((run/'VALIDATION.json').read_text(encoding='utf-8'));assert val['pass'] and val['full_population']
 assert sha(run/'BUILD_SUMMARY.json')==val['build_summary_sha256'] and sha(run/'VALIDATED_DISTRIBUTIONS.npz')==val['distributions_sha256']
 out=run/'review';out.mkdir(exist_ok=True);summ=json.loads((run/'BUILD_SUMMARY.json').read_text(encoding='utf-8'));raw=json.loads((ROOT/'full_fit_raw_audit_20261018/review_02/SUMMARY.json').read_text(encoding='utf-8'))
 with np.load(run/'VALIDATED_DISTRIBUTIONS.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
 n=len(d['row_id']);by_patch={p:{'retained_matches':sum(raw['by_patch'][p].get(k,0) for k in ('RETAIN','RETAIN_MASKED')),'query_matches':len(set(d['match_id'][d['patch']==p])),'queries':int(np.sum(d['patch']==p)),'selection_supported_queries':int(np.sum(d['selection_supported'][d['patch']==p]))} for p in PATCHES}
 events=summ['event_fields'];es=d['cum_event_supported'].all(axis=2);gold=d['gold_supported'].all(axis=2);terminal=d['terminal_supported'];hazard=d['hazard_at_risk_supported'];pareto=d['pareto_supported'];epics=es[:,:,EPICS].all(axis=2)
 supports={'골드 차':gold,'6 오브젝트 전체':epics,'20 사건 전체':es.all(axis=2),'종료 상태':terminal,'종료 hazard 위험집합':hazard,'Pareto':pareto}
 support={name:{H[k]:{'supported':int(mask[:,k].sum()),'denominator':n,'pct':float(mask[:,k].mean()*100)} for k in range(3)} for name,mask in supports.items()}
 event_support={events[j]:{H[k]:{'supported':int(es[:,k,j].sum()),'denominator':n,'pct':float(es[:,k,j].mean()*100)} for k in range(3)} for j in range(len(events))}
 rare={};labels={};time_summary={};patch_support={}
 for k,label in enumerate(H):
  rare[label]={};labels[label]={};time_summary[label]={}
  for j,name in zip(EPICS,NAMES):
   supported=es[:,k,j];positive=supported&(d['cum_team_events'][:,k,:,j].sum(axis=1)>0)
   rare[label][name]={'supported_windows':int(supported.sum()),'positive_windows':int(positive.sum()),'unique_matches_with_positive_window':len(set(d['match_id'][positive])),'positive_pct_of_supported':float(positive.sum()/max(1,supported.sum())*100),'unsupported_windows':int((~supported).sum())}
  known=np.isfinite(d['net_gold'][:,k]);g=d['net_gold'][known,k]
  labels[label]['economic']={'RED_GAIN':int(np.sum(g<0)),'TIE':int(np.sum(g==0)),'BLUE_GAIN':int(np.sum(g>0)),'UNKNOWN':int((~known).sum())}
  labels[label]['gold_quantiles']={str(q):float(np.quantile(g,q)) for q in [0,.01,.25,.5,.75,.99,1]}
  labels[label]['terminal']={str(x):int(np.sum(d['terminal_class'][:,k]==x)) for x in (-1,0,1,2)}
  labels[label]['pareto']={str(x):int(np.sum(d['pareto_class'][:,k]==x)) for x in (-1,0,1,2,3)}
  elapsed=d['actual_elapsed_ms'][:,k];ok=elapsed>=0;nominal=(k+1)*60000
  time_summary[label]={'endpoint_supported':int(ok.sum()),'shorter_than_nominal':int(np.sum(ok&(elapsed<nominal))),'equal_nominal':int(np.sum(elapsed==nominal)),'later_within_tolerance':int(np.sum(elapsed>nominal)),'max_elapsed_ms':int(elapsed.max())}
 for p in PATCHES:
  sel=d['patch']==p;patch_support[p]={name:[{'supported':int(mask[sel,k].sum()),'denominator':int(sel.sum())} for k in range(3)] for name,mask in supports.items()}
 result={'queries':n,'query_matches':len(set(d['match_id'])),'retained_matches':summ['audit_retained_matches'],'no_query_matches':val['retained_sources_without_queries'],'selection_supported_queries':int(d['selection_supported'].sum()),'ambiguous_selection_queries':int((~d['selection_supported']).sum()),'ambiguous_selection_matches':len(set(d['match_id'][~d['selection_supported']])),'by_patch':by_patch,'support':support,'event_support':event_support,'patch_support':patch_support,'rare_events':rare,'labels':labels,'actual_elapsed':time_summary,'soul_input_supported_rows':int((d['soul_source_status']>0).all(1).sum()),'focal_kills_supported_rows':int(d['focal_kills_supported'].all(1).sum()),'source_validation_sha256':sha(run/'VALIDATION.json'),'training_performed':False}
 dump(out/'RESULTS.json',result)
 with (out/'HORIZON_SUPPORT.csv').open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.writer(f);w.writerow(['target','horizon','supported','all_query_rows','support_pct'])
  for name,hh in support.items():
   for h,r in hh.items():w.writerow([name,h,r['supported'],r['denominator'],r['pct']])
 def save(fig,name):
  fig.savefig(out/(name+'.png'),bbox_inches='tight');fig.savefig(out/(name+'.svg'),bbox_inches='tight');plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained');x=np.arange(3)
 for off,key,title,col in [(-.18,'retained_matches','보존 경기','#b1b8bd'),(.18,'query_matches','교전 행 보유 경기',COLORS[0])]:
  b=axs[0].bar(x+off,[by_patch[p][key] for p in PATCHES],.36,label=title,color=col);axs[0].bar_label(b,fontsize=8)
 axs[0].set(xticks=x,xticklabels=PATCHES,ylabel='경기 수');axs[0].legend(fontsize=8)
 b=axs[1].bar(PATCHES,[by_patch[p]['queries'] for p in PATCHES],color=COLORS);axs[1].bar_label(b);axs[1].set(ylabel='고유 교전 후보 행 수')
 fig.suptitle('01 | 전체 FIT 보존 경기 → 기존 정의의 교전 후보',fontsize=15);save(fig,'01_population')
 fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained');qc=np.array(list(Counter(d['match_id']).values()));axs[0].hist(qc,bins=np.arange(qc.max()+2)-.5,color=COLORS[0]);axs[0].set(xlabel='교전 행 보유 경기당 교전 후보 수',ylabel='경기 수')
 ages=(d['tau_ms']-d['t0_ms'])/1000;axs[1].hist(ages,bins=np.linspace(0,10,21),color=COLORS[1]);axs[1].set(xlabel='tau - t0 (초)',ylabel='교전 행 수')
 fig.suptitle('02 | 선별 분포 — 프레임 신선도 0–10초 조건 유지',fontsize=15);save(fig,'02_selection')
 fig,ax=plt.subplots(figsize=(12,5.4),layout='constrained');keys=list(supports);xx=np.arange(len(keys))
 for k in range(3):
  vv=[supports[key][:,k].mean()*100 for key in keys];b=ax.bar(xx+(k-1)*.25,vv,.25,label=H[k],color=COLORS[k]);ax.bar_label(b,fmt='%.3f',fontsize=7)
 ax.set(xticks=xx,xticklabels=keys,ylabel=f'전체 {n:,}행 중 지원 (%)',ylim=(0,113));ax.legend();fig.suptitle('03 | 예측창별 지원율 — hazard는 관측 지원과 위험집합 조건의 교집합',fontsize=14);save(fig,'03_support')
 fig,ax=plt.subplots(figsize=(8,10),layout='constrained');rates=np.array([[es[:,k,j].mean()*100 for k in range(3)] for j in range(20)]);im=ax.imshow(rates,vmin=0,vmax=100,cmap='YlGnBu',aspect='auto')
 for j in range(20):
  for k in range(3):ax.text(k,j,f'{rates[j,k]:.3f}%',ha='center',va='center',color='white' if rates[j,k]>55 else '#20313e',fontsize=9)
 ax.set(xticks=range(3),xticklabels=H,yticks=range(20),yticklabels=events);fig.colorbar(im,ax=ax,label='양 팀 모두 지원 (%)');fig.suptitle('04 | 사건 채널별 지원율 — 미지원은 0회가 아님',fontsize=15);save(fig,'04_event_support')
 fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
 for k in range(3):
  a=np.sort(d['net_gold'][np.isfinite(d['net_gold'][:,k]),k]);axs[0].plot(a,np.arange(1,len(a)+1)/len(a),label=f'{H[k]} n={len(a):,}',color=COLORS[k])
 axs[0].set(xlabel='BLUE 추가 골드 - RED 추가 골드',ylabel='지원 교전 행 누적 비율');axs[0].legend(fontsize=8)
 bottom=np.zeros(3)
 for name,col in [('RED_GAIN','#5b7999'),('TIE','#aaa'),('BLUE_GAIN','#287d76'),('UNKNOWN','#c77b63')]:
  vv=np.array([labels[h]['economic'][name] for h in H])/n*100;axs[1].bar(H,vv,bottom=bottom,label=name,color=col);bottom+=vv
 axs[1].set(ylabel='전체 교전 행 (%)');axs[1].legend(fontsize=8);fig.suptitle('05 | 관측 순골드 라벨 — 전체 범위 보존, 인과효과 해석 없음',fontsize=14);save(fig,'05_economic')
 fig,axs=plt.subplots(1,2,figsize=(13,5.3),layout='constrained');xx=np.arange(6)
 for k,h in enumerate(H):
  for ax,key,title in [(axs[0],'positive_windows','지원 양성 교전 창 수'),(axs[1],'unique_matches_with_positive_window','양성 창을 가진 고유 경기 수')]:
   vv=[rare[h][name][key] for name in NAMES];b=ax.bar(xx+(k-1)*.25,vv,.25,label=h,color=COLORS[k]);ax.bar_label(b,fontsize=6,rotation=70,padding=2);ax.set(xticks=xx,xticklabels=NAMES,yscale='log',ylabel=title);ax.set_ylim(.8,max(vv)*5)
 axs[0].legend();fig.suptitle('06 | 희귀 사건 — 겹치는 창 수와 고유 경기 수를 분리 (로그 축)',fontsize=14);save(fig,'06_rare_events')
 fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained');bottom=np.zeros(3)
 for code,lab,col in [(-1,'UNKNOWN','#b5b5b5'),(0,'진행 중','#d7dfe3'),(1,'BLUE 승리',COLORS[0]),(2,'RED 승리',COLORS[1])]:
  vv=np.array([labels[h]['terminal'][str(code)] for h in H])/n*100;axs[0].bar(H,vv,bottom=bottom,label=lab,color=col);bottom+=vv
 axs[0].set(ylabel='전체 교전 행 (%)');axs[0].legend(fontsize=8)
 for k in range(3):
  vv=d['actual_elapsed_ms'][:,k];vv=np.sort(vv[vv>=0]/1000);axs[1].plot(vv,np.arange(1,len(vv)+1)/len(vv),color=COLORS[k],label=H[k])
 axs[1].set(xlabel='실제 endpoint - t0 (초)',ylabel='종점 지원 행 누적 비율');axs[1].legend();fig.suptitle('07 | 종료·실제 예측 길이 — 종료 후 반복 종점은 흡수 상태',fontsize=14);save(fig,'07_terminal')
 notes=[('01_population','모집단과 교전 수','교전이 없는 경기도 SOURCE_LEDGER에 남아 있습니다. 경기 수와 교전 행 수를 합산하지 않습니다.'),('02_selection','교전 선별과 프레임 나이','기존 fresh 조건을 유지한 조건부 모집단입니다. 전체 임의 시점의 교전 발생 예측 자료가 아닙니다.'),('03_support','60·120·180초 지원율','분모는 모든 후보 행입니다. 모호한 경계의 행은 타깃 미지원으로 포함됩니다. hazard는 이미 종료된 행을 위험집합에서 제외합니다.'),('04_event_support','사건별 지원율','20개 사건 전체를 동시에 요구하는 것은 권장 loss 규칙이 아닙니다. 각 타깃별 지원 마스크를 사용해야 합니다.'),('05_economic','순골드 라벨','원본에서 관측된 미래 추가 순골드입니다. 승률 가치나 오브젝트의 인과적 추가 가치가 아닙니다.'),('06_rare_events','희귀 사건','양성은 양 팀의 사건 수가 모두 지원되는 창에서 계산합니다. 겹치는 창의 수를 독립 사건 수로 해석하지 않습니다. 상세 지원 분모와 양성률은 RESULTS.json에 있습니다.'),('07_terminal','종료와 실제 길이','60·120·180초는 명목 길이입니다. 조기 종료 또는 5초 내 프레임 정렬에 따른 실제 길이를 별도 저장했습니다.')]
 lines=[f'# 전체 FIT 학습 원장 r3 검증 결과\n',f'보존 경기 {summ["audit_retained_matches"]:,}개를 처리해 {result["query_matches"]:,}경기에서 교전 후보 {n:,}행을 생성했다. 교전 0개 경기 {result["no_query_matches"]:,}개도 소스 원장에 보존했다. 모호한 경계 {result["ambiguous_selection_queries"]:,}행은 모든 타깃을 미지원으로 표시했다. 모델 학습은 하지 않았다.\n','| 타깃 | 60초 지원 | 120초 지원 | 180초 지원 |','|---|---:|---:|---:|']
 for name in supports:lines.append('| '+name+' | '+' | '.join(f'{support[name][h]["supported"]:,} ({support[name][h]["pct"]:.3f}%)' for h in H)+' |')
 lines.extend(['',f'분모는 각 예측창 모두 {n:,}행이다. 기존 r2 FIT {val["prior_r2_fit_query_matches"]:,}경기·{val["prior_r2_fit_queries_exactly_preserved"]:,}교전의 ID와 경계를 전부 대조했다. 원본 재계산은 모든 {val["queries_checked"]:,}교전·{val["checks"]["raw_horizon_recalculations"]:,}예측창에 수행했다. 기존 파일 변경은 0건이다.','', '원소 용·장로·바론·아타칸·전령·유충은 획득 타깃이다. 영혼은 기존 입력 소유 상태이며, 별도의 영혼 획득 타깃은 새로 정의하지 않았다. 규칙 기반 가용성, 실제 버프 보유자, 직접 보상 분해, 잔여 가치 U, K3 및 지형은 이번 원장 생성으로 새로 지원된다고 주장하지 않는다.','', '어시스트 목록 생략의 의미는 미확인이다. 모호한 동시 킬과 해당 경계, 입력 스킬의 예외 슬롯도 관측 마스크를 유지한다. 전역 complete-case 필터로 경기/행 전체를 버리지 말고 타깃별 지원 마스크로 loss를 계산해야 한다.'])
 for stem,title,note in notes:lines.extend(['',f'## {title}',note,f'![{title}]({(out/(stem+".png")).as_posix()})'])
 lines.extend(['','## 학습 전 점검','본 결과는 FIT 자료 생산·검산의 완료다. 검증/시험 원장의 갱신, 실제 학습, 성능 평가 및 인과적 가치 검증은 수행하지 않았다. 승인된 기존 입력창·예측창·분할을 유지했고, 새 원장과 그림을 사용자에게 보여주는 단계에서 멈춘다.'])
 (out/'RESULTS_KO.md').write_text('\n'.join(lines),encoding='utf-8')
 body=''.join(f'<section id="{stem}"><h2>{html.escape(title)}</h2><p>{html.escape(note)}</p><a href="{stem}.png"><img src="{stem}.png" alt="{html.escape(title)}" loading="lazy"></a></section>' for stem,title,note in notes)
 nav=''.join(f'<a href="#{stem}">{html.escape(title)}</a>' for stem,title,_ in notes)
 page='<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FIT r3 학습 전 점검</title><style>body{font-family:Malgun Gothic,system-ui;background:#f4f6f7;color:#20313e;line-height:1.8;margin:0}main{max-width:1120px;margin:auto;padding:28px}section{background:white;border:1px solid #dbe0e5;margin:24px 0;padding:22px}img{width:100%;height:auto}nav{display:flex;flex-wrap:wrap;gap:15px}a{color:#235c88}.note{border-left:4px solid #287d76;padding:15px;background:white}</style><main>'+f'<h1>전체 FIT r3 · 학습 전 점검</h1><p>보존 {summ["audit_retained_matches"]:,}경기 → 교전 보유 {result["query_matches"]:,}경기 → {n:,}교전 후보</p><div class="note">원본 재계산과 기존 교전 경계 대조를 마쳤습니다. 모호한 경계 {result["ambiguous_selection_queries"]:,}행은 타깃 미지원입니다. 기존 원장을 보존했고 모델은 학습하지 않았습니다.</div><nav>'+nav+'</nav>'+body+'<section><h2>근거와 검토</h2><a href="RESULTS_KO.md">전체 보고서</a> · <a href="HORIZON_SUPPORT.csv">예측창별 지원율</a> · <a href="RESULTS.json">분모·희귀 사건 상세</a> · <a href="../VALIDATION.json">전수 검산</a><p>타깃별 마스크를 유지한 학습 입력 후보입니다. 학습 시작 전에 위 분포와 미지원 처리를 점검해 주세요.</p></section></main></html>'
 (out/'index.html').write_text(page,encoding='utf-8')
 manifest={}
 for p in sorted(out.iterdir()):
  if p.is_file() and p.name!='MANIFEST.json':manifest[p.name]={'sha256':sha(p),'bytes':p.stat().st_size}
 dump(out/'MANIFEST.json',{'source_validation_sha256':sha(run/'VALIDATION.json'),'files':manifest,'training_performed':False})
 print(json.dumps({k:result[k] for k in ['queries','query_matches','retained_matches','no_query_matches','ambiguous_selection_queries','support','rare_events']},ensure_ascii=False))

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('run',type=Path);main(a.parse_args().run)
