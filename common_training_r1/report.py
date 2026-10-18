"""Implementation review, support limitations and descriptive STOP smoke figures."""
from collections import Counter
import json,platform,re
import joblib,numpy as np,torch,lightgbm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import softmax
from common import *
from models import derived
from train import score

def main():
    acc=read(HERE/'ACCEPTANCE.json');assert acc['pass'];r=read(HERE/'RESULTS.json');d=joblib.load(HERE/'DATA.joblib');spec=read(HERE/'LOSS_SPEC.json');card=read(HERE/'CARD.json')
    stop=d['meta']['macro_role']=='STOP';fit=~stop;y=subset(d['y'],stop);m=subset(d['meta'],stop);n=int(stop.sum());p=np.load(HERE/'PREDICTIONS.npz',allow_pickle=False)
    initial={'gold':np.tile(spec['initial']['gold'],(n,1)),'count':np.tile(spec['initial']['count'],(n,1)),'hazard':np.tile(softmax(np.array(spec['initial']['hazard']).reshape(3,3),axis=1),(n,1,1))}
    base=derived([initial,initial],spec);baseline=score(base,[initial,initial],y,m,spec);dump(HERE/'BASELINE.json',{'description':'FIT weighted mean initialization; no additional model fit','metrics':baseline})
    all_metrics={**r['metrics'],'fit_constant':baseline};review=HERE/'review';review.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False})
    def save(fig,name):
        fig.tight_layout()
        for ext in ('png','svg'):fig.savefig(review/f'{name}.{ext}',dpi=150)
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.7));xx=np.arange(3)
    for i,(name,color) in enumerate([('fit_constant','#959da2'),('tree','#477d9b'),('hierarchy','#358579')]):
        vals=[z['net_gold_mae'] for z in all_metrics[name]];ax.bar(xx+(i-1)*.24,vals,.24,label=name,color=color)
        for j,v in enumerate(vals):ax.text(j+(i-1)*.24,v+20,f'{v:.0f}',ha='center',fontsize=9)
    ax.set_xticks(xx,['60초','120초','180초']);ax.set_ylabel('경기 동등 가중 순골드 MAE (gold)');ax.set_title(f"STOP 진단 — n={len(set(m['match_id']))}, 단일 seed·L={card['L']}, 우열 검정 아님");ax.legend();ax.set_ylim(0,max(z['net_gold_mae'] for rows in all_metrics.values() for z in rows)*1.2);save(fig,'01_gold')
    trace=read(HERE/'TRAIN_TRACE.json');fig,ax=plt.subplots(figsize=(10,4.5));ax.plot([t['epoch'] for t in trace],[t['training_minibatch_objective'] for t in trace],label='FIT minibatch 추정값');ax.plot([t['epoch'] for t in trace],[t['stop_objective'] for t in trace],label='STOP epoch 말 목적함수');ax.axvline(r['hierarchy_info']['best_epoch'],ls='--',color='#888',label='선택 epoch');ax.set_xlabel('epoch');ax.set_ylabel('정규화된 공통 손실');ax.set_title(f"계층 모델 실행 — 최대 {card['epochs']} epochs");ax.legend();save(fig,'02_loss')
    coverage={};labels=['base18','개인 K/D/A','슬롯 업그레이드 기록','아이템 보유 상태','팀 합계 8개','관계 거리']
    for role,ix in [('FIT',fit),('STOP',stop)]:
        sums=np.zeros(6);den=np.zeros(6)
        for index in np.where(ix)[0]:
            obs=d['observed'][index]
            for j in range(5-card['L'],5):
                if not obs['present'][j]:continue
                vals=[obs['player'][j,:,:18],obs['player'][j,:,18:21],obs['player'][j,:,21:25],obs['items'][j][1],obs['team'][j,:,:8],obs['relation'][j,:,:,2]]
                for k,v in enumerate(vals):sums[k]+=np.sum(v if k==3 else np.isfinite(v));den[k]+=v.size
        coverage[role]={'supported':sums.astype(int).tolist(),'denominator':den.astype(int).tolist(),'pct':(100*sums/den).tolist()}
    dump(HERE/'FEATURE_COVERAGE.json',{'labels':labels,'groups':coverage,'denominator':'observed history cells; includes repeated frame usage across eligible queries, not unique raw frames','complete_inventory_supported':bool(coverage['FIT']['pct'][3] == 100 and coverage['STOP']['pct'][3] == 100),'items_reason':'Per-player inventory support follows the observed attribution masks; unsupported item ownership is not imputed.'})
    fig,ax=plt.subplots(figsize=(11,4.8));pos=np.arange(6)
    for i,(role,color) in enumerate([('FIT','#477d9b'),('STOP','#358579')]):
        ax.bar(pos+(i-.5)*.36,coverage[role]['pct'],.36,label=role,color=color)
        for j,v in enumerate(coverage[role]['pct']):ax.text(j+(i-.5)*.36,v+2,f'{v:.1f}%',ha='center',fontsize=9)
    ax.set_xticks(pos,labels,rotation=12);ax.set_ylim(0,115);ax.set_ylabel('지원 비율 (%)');ax.set_title('입력 지원 감사 — 미지원 항목 때문에 교전 행 전체를 버리지 않음');ax.legend();save(fig,'03_input_support')
    fig,axes=plt.subplots(1,2,figsize=(11,4.8));h=1;mask=np.isfinite(y['net_gold'][:,h]);actual=y['net_gold'][mask,h]
    for ax,name in zip(axes,['tree','hierarchy']):
        pred=p[name+'_net_gold'][mask,h];ax.scatter(actual,pred,s=12,alpha=.5);lo=min(actual.min(),pred.min());hi=max(actual.max(),pred.max());ax.plot([lo,hi],[lo,hi],'--',color='#777');ax.set_title(f'{name}: 120초');ax.set_xlabel('관측 순골드');ax.set_ylabel('예측 순골드')
    fig.suptitle('STOP 전체 지원 교전 — 극단값 제외 없음');save(fig,'04_scatter')
    # Positive support denominators, not a rare-event performance ranking.
    fig,ax=plt.subplots(figsize=(10,4.7));events=EVENTS[2:8];rows=r['metrics']['tree'][2]['events'];positive=[sum(e['positive_rows'] for e in rows if e['event']==name) for name in events];ax.bar(events,positive,color='#ab8042')
    for i,v in enumerate(positive):ax.text(i,v+.5,str(v),ha='center')
    ax.set_ylabel('180초 양성 팀×교전 창 수 (중첩 가능)');ax.set_title('희귀 사건 지원 — 이 작은 실행으로 희귀 사건 일반화를 판단하지 않음');save(fig,'05_rare_support')
    fig,ax=plt.subplots(figsize=(9,4.5))
    for name in ('tree','hierarchy','fit_constant'):ax.plot([60,120,180],[v['terminal_brier'] for v in all_metrics[name]],marker='o',label=name)
    ax.set_xlabel('명목 예측창 (초)');ax.set_ylabel('종료 3-class Brier');ax.set_title('STOP 종료 확률 진단');ax.legend();save(fig,'06_terminal')
    usage=Counter();reviews=[]
    for f in HERE.glob('strata_*.json'):
        q=read(f);usage.update({k:q.get('usage',{}).get(k,0) for k in ('prompt_tokens','completion_tokens','total_tokens')});reviews.append({'file':f.name,'finish_reason':q.get('finish_reason')})
    dump(HERE/'STRATA_USAGE.json',{'local_usage':dict(usage),'reviews':reviews,'measured_subscription_savings':None})
    # Independently recalculate headline errors from saved predictions.
    for name in ('tree','hierarchy'):
        for h in range(3):
            mask=np.isfinite(y['net_gold'][:,h]);w=weights(mask[:,None],m['match_id'],m['patch'])[:,0];v=float(np.sum(w[mask]*np.abs(p[name+'_net_gold'][mask,h]-y['net_gold'][mask,h])));assert abs(v-r['metrics'][name][h]['net_gold_mae'])<1e-7
    ready={'implementation_smoke_passed':bool(acc.get('pass',False)),'common_full_inventory_ready':bool(r.get('common_full_inventory_ready',False)),'confirmatory_training_started':bool(r.get('full_training',False)),'saved_prediction_metrics_recomputed':True,'user_review_required_before_full_training':True,'remaining':['Review local item attribution support before complete inventory claims.','Validate the declared slot proxy before strongest-tree claims.','Lock the feature card and undertake full FIT streaming/cache capacity checks before confirmatory runs.','K3/terrain/buff/respawn/U remain separately gated extensions.']}
    dump(HERE/'READINESS.json',ready)
    table='\n'.join(f"| {z['horizon_s']}초 | {r['metrics']['tree'][j]['net_gold_mae']:.1f} | {r['metrics']['hierarchy'][j]['net_gold_mae']:.1f} | {baseline[j]['net_gold_mae']:.1f} |" for j,z in enumerate(baseline))
    text=f'''# DB 기반 공통 입력·학습 파이프라인 소규모 검증

이 문서는 현재 실행 폴더의 산출물을 요약한다. 구현 점검 통과는 본 학습 준비 완료나 확증적 성능을 뜻하지 않는다. 전체 학습·다중 seed·L 비교 여부는 실행 산출물에서 확인한다.

## 실제 실행

- DB `{SCHEMA}`에서 CARD에 지정된 FIT/STOP 표본 수를 역할별로 조회했다. FIT {r['fit_rows']:,}행, STOP {r['stop_rows']:,}행이며 교전 행 선택에 미래 라벨 값은 사용하지 않았다. 원래 교전 코호트 자체는 첫 킬을 이용해 정의한 회고적 모집단이다.
- L=3, seed={card['seed']}, CPU {card['torch_threads']} threads. DEV/CAL/AUDIT/TEST는 읽지 않았다. 기존 DB·원장·분할은 변경하지 않았다.
- 계층 모델은 {r['hierarchy_info']['parameters']:,}개 파라미터, 최대 {card['epochs']} epochs 중 STOP 목적함수로 epoch {r['hierarchy_info']['best_epoch']}를 선택했다. 트리는 {r['tree_info']['head_slots']}개 head slot 중 {r['tree_info']['trained_estimators']}개 추정기를 실제 적합했고, 트리 수는 {r['tree_info']['actual_trees']}개다. 상수/미지원 head는 별도로 기록했다.
- 모델별 flat 정보량은 동일한 {r['input_flat_dimension']:,}차원 텐서에서 출발한다. GRU/MLP·트리의 구조와 최적화·규제·퇴화 head 처리가 달라 이것만으로 순수 구조 효과를 주장하지 않는다.

## 공통 입력과 모델

`data.py`는 원본 해시 및 DB/NPZ 마스크를 확인하고, base18 + 개인 K/D/A + 슬롯별 스킬 업그레이드 사건 수 + 마지막 사망 나이 + 아이템 사건 잔액 + 팀 집계 + 팀 사건/마지막 시각 + 구조물별 파괴 기록 + 협곡 공지 + 공간 관계를 생성한다. 스킬 업그레이드 수는 예외 챔피언의 정확한 rank/사용 가능 여부가 아니다. 구조물 파괴 기록은 억제기의 영구 부재 상태가 아니다.

각 실제 이력 프레임마다 그 시각까지의 prefix를 다시 만들었다. 현재 prefix를 과거에 복제하지 않는다. 90초 이내 고정 위치·CS·강타 및 시작 챔피언/주문으로 canonical 슬롯을 정한다. 모호한 경우 챔피언/주문으로 고정하며 역할 정답으로 취급하지 않는다. 사후 role·아이템·win·킬 결과 필드는 읽지 않는다. 실제 프레임의 좌표·HP 지원으로 공간 활동을 판정하며 사망한 개인 토큰은 유지한다.

전처리·범주 vocab·골드 스케일·count dispersion은 FIT만 사용한다. 미지원 값은 변환 후 0과 별도 support bit로 표현하며, 이는 실제 0 관측과 구분된다. 트리는 동일한 allowlist 텐서를 flatten하고 계층 모델은 relation→player GRU→team pooling 및 team GRU→rift GRU 융합을 거친다. player/team/rift/relation 경로 제거 스위치는 구현했지만 별도 ablation 실행은 하지 않았다.

**아이템 제한:** 완전한 아이템 보유 상태의 지원 여부는 로컬 입력의 귀속 마스크를 따르며, 미지원 소유권을 추정하거나 다른 선수에게 임의 배분하지 않는다. 본 파이프라인의 아이템 지원 분포는 실행 시 생성되는 지원 표에서 확인한다.

## 목표·손실·평가

동일 t0에서 세 구간의 팀 골드(6), 사건 count(84), 종료 hazard(3×3)를 병렬 출력한다. 실제 60초 결과를 다음 출력의 입력으로 쓰지 않는다. 골드는 FIT scale Huber(delta=1), count는 고정 FIT dispersion NB NLL, 종료는 위험집합의 CE다. 트리 Huber Hessian에는 0.01 하한을 쓰는 최적화 근사가 있다. group 평균 가중은 경제적 오브젝트 가격이 아니다.

각 head에서 지원 query→경기→패치 중첩 평균을 계산한다. 두 팀 방향을 증강한 모집단에서 가중을 다시 계산하고, 균일 query minibatch에서는 전체 정규화 가중에 N/batch_size를 곱한다. 미지원 타깃은 산술 전에 제외한다. 전체 미지원 head를 0-loss 경기처럼 분모에 넣지 않는다.

예측 골드/사건 평균은 구간 합으로 누적한다. count P(0)는 조건부 구간 독립 가정에서 구간 P(0)의 곱이고, 두 팀 방향의 누적 분포를 만든 뒤 1/2 혼합한다. 종료는 생존×위험으로 누적해 단조성·확률합을 검사한다. 팀 간 획득 배타성, 종결 후 골드/사건 흡수, 남은 구조물 수는 자동 보장하지 않는다. 누적 count NLL은 보고하지 않으며 NB NLL은 구간별이다.

## STOP 진단값

| 예측창 | Tree 순골드 MAE | Hierarchy 순골드 MAE | FIT 상수 참고 |
|---|---:|---:|---:|
{table}

STOP은 epoch 선택에 사용하므로 이 표는 선택에 노출된 진단값이다. 단일 seed·작은 표본에서 구조 우열/일반화/희귀 오브젝트 가치 타당성을 판단하지 않는다. 사건별 양성 창·고유 경기·지원 분모와 AP undefined 조건은 로컬 RESULTS.json에 기록한다. 경제 예측은 관측 후속 골드이지 교전의 인과적 추가 이득이 아니다.

## 검증과 수정 이력

{read(HERE/'TEST_RESULTS.json')['tests']}개 검사가 통과했다: shared flat roundtrip, FIT-only guard, L별 현재 입력 일치, 미래 사건 변조 불변, 시각 경계 포함, 미지원 발생의 과거 역전파 방지, 협곡 이름/unknown 처리, canonical 순열, 사후 결과 필드 무시, 팀 합계/공간 활동, 계층 가중·minibatch 기대값, masked NaN/극단값 손실, NB 수치 미분, 선수 순열·팀 교환·누적 확률 일관성.

두 모델을 파일에서 다시 읽은 예측의 최대 차이는 {r['reload_errors']}다. 저장 예측으로 순골드 MAE를 독립 재계산했다. 선택한 원본 파일과 기존 shard 해시는 실행 전후 동일하다.

실행별 수정 이력과 로그는 로컬에 보존한다. 팀 교환과 순열 가정은 포함된 단위 검사로 점검한다.

STOP은 선택에 사용된 진단 집합이며 독립된 최종 시험 성능으로 해석하지 않는다. 외부 API·에이전트 사용량은 이 실행 보고서에서 평가하지 않는다.

## 다음 승인 전 남은 항목

1. 아이템 상태를 기본 COMMON에서 명시적으로 제외할지, 귀속 불명 이벤트를 설명할 근거를 확보할지 결정. 현재 보수적 마스크 유지.
2. 90초 슬롯 프록시의 전술적 안정성 및 강한 트리 기준선 적합성 점검.
3. 본실험 카드·예산 고정과 전체 FIT에 대한 스트리밍/캐시 메모리 검증. 현재 smoke runner는 선택 표본을 메모리에 올린다. 전체 규모 자동 실행으로 확장하지 않는다.
4. K3·지형 geometry·정확한 버프/부활·잔여 가치 U는 별도 지원 게이트.

재현: CARD와 DB_SELECTION, DATA_MANIFEST, LOSS_SPEC, TEST_RESULTS, ACCEPTANCE, READINESS, 모델 checkpoint와 PREDICTIONS.npz를 함께 보존한다. DB 접속은 호출 환경의 PGPASSFILE을 사용하며 인증 내용은 코드·모델·보고서에 저장하지 않는다.
'''
    (HERE/'REPORT_KO.md').write_text(text,encoding='utf-8')
    titles={'01_gold':'순골드 예측 진단','02_loss':'학습 곡선','03_input_support':'입력 지원율','04_scatter':'120초 예측 분포','05_rare_support':'희귀 사건 표본 수','06_terminal':'종료 확률 진단'}
    page='<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>공통 입력·학습 smoke</title><style>body{margin:0;background:#f3f5f6;color:#223746;font:16px/1.8 Malgun Gothic,system-ui}main{max-width:1130px;margin:auto;padding:28px}section{background:white;border:1px solid #ddd;padding:22px;margin:24px 0}img{width:100%}a{color:#226985}.notice{border-left:5px solid #ad7937;padding:15px;background:#fff8e9}</style><main><h1>공통 입력·학습 파이프라인 — 실행 요약</h1><p>FIT __FIT_MATCHES__ matches / STOP __STOP_MATCHES__ matches · L=__HISTORY_LENGTH__ · seed __SEED__ · full training __FULL_TRAINING__</p><p class="notice">이 페이지의 표와 그림은 현재 실행 폴더 산출물에서 생성합니다. STOP은 epoch 선택에 사용한 진단셋이며 독립 시험 성능이나 최종 일반화 성능으로 해석할 수 없습니다.</p>'
    page=page.replace('__FIT_MATCHES__',str(r['fit_matches'])).replace('__STOP_MATCHES__',str(r['stop_matches'])).replace('__HISTORY_LENGTH__',str(card['L'])).replace('__SEED__',str(card['seed'])).replace('__FULL_TRAINING__',str(r['full_training']))
    for name,title in titles.items():page+=f'<section><h2>{title}</h2><img src="{name}.png" alt="{title}"></section>'
    page+='<section><a href="../REPORT_KO.md">설계·실행 보고서</a> · <a href="../RESULTS.json">지표·지원 분모</a> · <a href="../TEST_RESULTS.json">검사 결과</a> · <a href="../READINESS.json">남은 게이트</a> · <a href="../CARD.json">고정 실행 카드</a></section></main></html>'
    (review/'index.html').write_text(page,encoding='utf-8')
    dump(HERE/'ENVIRONMENT.json',{'python':platform.python_version(),'torch':torch.__version__,'lightgbm':lightgbm.__version__,'numpy':np.__version__,'device':'cpu','threads':4})
    files=[q for q in HERE.iterdir() if q.is_file() and q.suffix in ('.py','.json','.md','.pt','.joblib','.npz') and q.name!='DELIVERY_MANIFEST.json']+list(review.iterdir())
    dump(HERE/'DELIVERY_MANIFEST.json',{'files':{str(q.relative_to(HERE)):sha(q) for q in files},'full_training':False})
    print(json.dumps({'tests':read(HERE/'TEST_RESULTS.json'),'MAE_120':{k:v[1]['net_gold_mae'] for k,v in all_metrics.items()},'local_tokens':usage['total_tokens'],'implementation_smoke_passed':True,'full_inventory_ready':False}))
if __name__=='__main__':main()
