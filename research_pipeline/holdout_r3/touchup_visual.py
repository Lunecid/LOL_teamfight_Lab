"""Rerender only the reviewed rare-event chart; statistics and ledgers unchanged."""
import json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent;V=H/'production_01/review';r=json.loads((V/'RESULTS.json').read_text(encoding='utf-8'))
labels=['FIT','STOP','DEV','CAL','AUDIT','TEST'];events=['elem_dragon','elder_dragon','baron','atakhan','herald','horde']
plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'font.size':10})
fig,axs=plt.subplots(2,3,figsize=(14,9))
for ax,e in zip(axs.flat,events):
 m=np.array([[next(x['positive_pct'] for x in r['rare_events'] if x['group']==g and x['event']==e and x['horizon_s']==hh) for hh in (60,120,180)] for g in labels]);ax.imshow(m,cmap='YlOrBr',aspect='auto');ax.set_title(e+' 양성 창 비율');ax.set_xticks(range(3),['60s','120s','180s']);ax.set_yticks(range(6),labels)
 for i in range(6):
  for j in range(3):ax.text(j,i,f'{m[i,j]:.2f}%',ha='center',va='center',fontsize=8,color='white' if (m[i,j]-m.min())/max(1e-9,m.max()-m.min())>.6 else 'black')
fig.savefig(V/'04_rare.png',dpi=150,bbox_inches='tight');fig.savefig(V/'04_rare.svg',bbox_inches='tight');plt.close(fig)
p=V/'index.html';s=p.read_text(encoding='utf-8');s=s.replace('분모=양 팀이 해당 사건을 지원하는 창. 창은','분모=양 팀이 해당 사건을 지원하는 창. 색 범위는 오브젝트별로 다르므로 표시된 비율로 비교합니다. 창은');p.write_text(s,encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
mp=V/'MANIFEST.json';d=json.loads(mp.read_text(encoding='utf-8'));d['files']={p.name:sha(p) for p in V.iterdir() if p.is_file() and p.name!='MANIFEST.json'};dump(mp,d)
dp=H/'production_01/DELIVERY_MANIFEST.json';d=json.loads(dp.read_text(encoding='utf-8'))
for p in [H/'compare.py',Path(__file__),mp]:d['files'][str(p.relative_to(H))]=sha(p)
dump(dp,d)
print('Rare-event contrast fixed; no source statistics or ledger changed.')
