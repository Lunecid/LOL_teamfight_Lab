"""Paired 60/90/120s subsequent-kill proxy and objective coverage; no fitting."""
from pathlib import Path
import json,time,hashlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];CACHE=Path('D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13');OUT=ROOT/'outputs/compare90_contamination'
def one(task):
    mid,refs=task;events=json.loads((CACHE/(mid+'.events.json')).read_text());kills=[e for e in events if e.get('type')=='CHAMPION_KILL'];byts={int(e['timestamp']):e for e in kills};counts=Counter(int(e['timestamp']) for e in kills);out=[]
    objs=[e for e in events if e.get('type')=='ELITE_MONSTER_KILL' and e.get('monsterType') in ['DRAGON','BARON_NASHOR','RIFTHERALD','HORDE','ATAKHAN']]
    for r in refs:
        kt=r['s']+15000;anchor=byts[kt]['position'];ambiguous=counts[kt]>1
        ks=[int(e['timestamp']) for e in kills if r['L']<int(e['timestamp'])<=r['endpoint']];first=min(ks) if ks else None
        local=[];remote=0;unknown=0
        for ev in objs:
            t=int(ev['timestamp'])
            if not r['L']<t<=r['endpoint']:continue
            pos=ev.get('position') or {}
            if ambiguous or 'x' not in pos or 'y' not in pos:unknown+=1
            elif (pos['x']-anchor['x'])**2+(pos['y']-anchor['y'])**2<=4264**2:local.append(t)
            else:remote+=1
        out.append(dict(match=mid,s=r['s'],h=r['h'],additional_kills=len(ks),first_additional_delay=(first-r['L'])/1000 if first else None,local_objectives=len(local),remote_objectives=remote,unknown_objectives=unknown,local_before_additional=sum(first is None or t<first for t in local),endpoint=r['endpoint']))
    return out
def main():
    OUT.mkdir(parents=True,exist_ok=True);start=time.time()
    labels=pd.read_csv(ROOT/'outputs/window_validation_20260914/labels.csv');d=labels[labels.h.isin([60,90,120])&(labels.valid==1)].copy()
    shared=d.groupby(['match','s']).h.nunique();keys=shared[shared==3].index;d=d.set_index(['match','s']).loc[keys].reset_index()
    rows=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i,r in enumerate(pool.map(one,[(mid,g.to_dict('records')) for mid,g in d.groupby('match')]),1):
            rows.extend(r)
            if i%3000==0:print(i,flush=True)
    df=pd.DataFrame(rows);df.to_csv(OUT/'rows.csv',index=False);summary=[]
    for h,a in df.groupby('h'):
        summary.append(dict(h=int(h),n=len(a),additional_kill_rows=int((a.additional_kills>0).sum()),additional_kill_pct=float((a.additional_kills>0).mean()*100),local_objective_rows=int((a.local_objectives>0).sum()),local_objective_pct=float((a.local_objectives>0).mean()*100),local_objective_without_additional=int(((a.local_objectives>0)&(a.additional_kills==0)).sum()),local_objective_before_first_additional=int((a.local_before_additional>0).sum())))
    old=pd.read_csv(ROOT/'outputs/fixed120_events_20260914/engagements.csv');j=df[df.h==120].merge(old,on=['match','s'],suffixes=('_new','_old'),validate='one_to_one')
    checks={k:int((j[k+'_new']!=j[k+'_old']).sum()) for k in ['additional_kills','local_objectives','remote_objectives','endpoint']}
    w=df.pivot(index=['match','s'],columns='h',values='additional_kills');paired=(w[120]>0).astype(int)-(w[90]>0).astype(int)
    g=paired.groupby(level='match').agg(['sum','count']);rng=np.random.default_rng(7);boots=[];s=g['sum'].to_numpy();n=g['count'].to_numpy()
    for _ in range(500):
        ix=rng.integers(0,len(g),len(g));boots.append(s[ix].sum()/n[ix].sum()*100)
    out=dict(summary=summary,checks_120_reproduction=checks,reduction90_vs120_pp=float(paired.mean()*100),reduction_ci95=np.quantile(boots,[.025,.975]).tolist(),seconds=time.time()-start,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'results.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
    lines=['# 90초 대 120초: 추가 킬 지표와 오브젝트 포함률','', '동일 독립 engagement 집합에서 공통 유효 교전만 비교했다. 추가 킬은 보수적 오염 대리 지표이며 새 한타의 정답 판정이 아니다. 다음 적격 교전/경기 종료 제한은 유지한다.','', '| 후속 초 | 교전 수 | 추가 킬 있는 교전 | 비율 | 주변 오브젝트 획득 교전 | 추가 킬 없이 주변 획득 | 첫 추가 킬 전 주변 획득 |','|---|---:|---:|---:|---:|---:|---:|']
    for a in summary:lines.append(f"| {a['h']} | {a['n']:,} | {a['additional_kill_rows']:,} | {a['additional_kill_pct']:.2f}% | {a['local_objective_rows']:,} | {a['local_objective_without_additional']:,} | {a['local_objective_before_first_additional']:,} |")
    lines+=['',f"120→90초의 추가 킬 노출 감소는 {out['reduction90_vs120_pp']:.2f}%p (경기 bootstrap500회95% CI {out['reduction_ci95'][0]:.2f}–{out['reduction_ci95'][1]:.2f}%p).",'',
    '90초는 오염 지표를 낮추지만 제거하지 않는다. 다음 추가 킬 직전에서 끊는 동적 경계를 별도 후보로 평가할 수 있다. 이 경우 추가 킬이 없다는 것은 설계상 보장되므로 독립 검증 성과로 주장하면 안 된다. 추격전 이득을 제외하는 문제, 가변 구간과 정보성 관찰 종료, 무킬 전투·원거리 성장의 혼입은 남는다.','',
    '앵커 timestamp가 모호한 사례는 공간 분류 불명으로 유지했다. 첫 추가 킬과 같은 timestamp 오브젝트는 그 킬 이전 획득으로 세지 않았다.','', '120초 기존 결과 재현 검사: '+json.dumps(checks),'', '- [재현 코드](../scripts/compare90_contamination.py)','- [수치와 신뢰구간](../outputs/compare90_contamination/results.json)']
    (ROOT/'docs/COMPARE90_CONTAMINATION_20260914.md').write_text('\n'.join(lines),encoding='utf-8')
if __name__=='__main__':main()
