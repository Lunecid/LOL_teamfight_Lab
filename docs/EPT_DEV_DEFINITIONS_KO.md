# EPT-T003 DEV 추론 정의와 통합 계약

2026-10-25. 이 문서는 별도 `ept-dev-evaluation` 개발 트리의 구현 정의다. 학습 트리 `ept-s25-pilot`은 수정하지 않는다. 저자의 이번 지시가 이전 계획의 M2 censoring·M3 draw·M4 taxonomy 결정 대기 항목을 아래와 같이 해소한다. 실제 DEV payload 읽기, 학습, production Check/Start, 재시도, 공개 게시를 수행했다는 문서가 아니다. 구현자 합성 검증과 독립 reviewer의 수락은 별도다.

## 1. API와 책임 경계

```python
run_inference(bundle, model, stats, b1, *, check, device='cuda',
              batch_size=32, draws=1000, seed=20261022, scales=None)
# -> {'endpoints': list[dict], 'nll': list[dict], 'provenance': dict}
```

`bundle.s1/s2/labels`, `bundle.prices`, `bundle.tier`를 사용한다. `static`, `rules`, `meta`를 함께 제공해도 무방하나 adapter는 이를 읽거나 재학습하지 않는다. 원래 global 숫자 row/match ID를 유지하는 DEV 제한 store를 caller가 제공한다. payload 파일 경로를 받거나 직접 open/load하지 않는다. `check()`는 첫 store 접근 전, head·배치·MC 목적물 chunk와 B1 열린 법칙의 호출 사이에 실행된다. False 반환 방식이 아닌 **예외를 발생시키는 guard**를 제공한다. 성공·실패 시 모형의 모든 원래 train/eval mode를 복구한다. 매개변수·장치를 변경하지 않고 float32 지정 장치의 EPT 또는 FrameAveraged(EPT)만 받는다.

Caller/lead가 먼저 학습 정상 종료 및 종료 세대 확인, verified best-only model 로딩, 코드·입력·parent 통계/scale identity, B1 FIT/STOP 이력, DEV 접근 승인·role scoping·예산을 검증한다. 현재 실행 중 트리나 training operator를 호출하지 않는다. 실제 실행에서는 `draws=1000`, `seed=20261022`, 승인 batch/device를 고정한다. 합성 fixture는 작은 draw 수와 CPU를 명시해 사용하며 이를 실측 결과로 보고하지 않는다.

정확한 canonical 로더:

| 산출물 | 로더 | 원문 근거 |
|---|---|---|
| 부모 FIT 통계 | `c_ept.ept.data.Stats.load(path)` | `c_ept/ept/data.py:263–312` |
| 부모 whole-FIT B0 scale | `c_ept.ept.scales.load_scales(path, expected_provenance=...)` | `c_ept/ept/scales.py:243` 이후 |
| 기존 B1 | `c_ept.models.baselines_b1.B1Heads.load(directory)` | `c_ept/models/baselines_b1.py:1919` 이후 |
| 고정 flow 가격 | `c_ept.valuation.value.load_prices(path)` | `c_ept/valuation/value.py:127–145` |
| 고정 tower tier | `c_ept.valuation.towertier.TierModel.load(directory)` | `c_ept/valuation/towertier.py:599–619` |

이 로더들은 실제 평가 승인 이후 caller가 호출한다. `primitive._load_inputs` 또는 모든 role을 훑는 `labels.verify()`를 사용하지 않는다. 새 stats/scales fit, B1 fit, best 재선택은 없다.

추가 label 배열은 `labels/f0_unit`, `labels/gold_reps`, `labels/kill_gold`, `labels/gold_bin`이다. `f0_unit`은 고정 가격과 도움 풀·실제 tower tier가 이미 반영된 저장 정답을 쓴다. rules로 다시 계산하지 않는다. M3 예측의 tower 상태는 DEV `units/f`, `matches/obj_off`, `objectives/t,kind,sub,taker`를 기존 `destroyed_before`에 전달해 계산한다. canonical HeadRows가 DEV 행을 선택한 뒤 해당 행만 추론·채점한다. 저장 배열은 여러 role이 공유하는 전처리 파일이며, read-only memmap과 요청된 배열 파일의 해시 검증은 파일 전체 바이트를 접근할 수 있다. 따라서 DEV만 추론한다는 계약과 다른 role의 바이트를 OS 수준에서 읽지 못하게 격리한다는 주장을 구분한다. 후자의 격리를 구현했다고 주장하지 않는다. 원시 파일·보호된 분할 본문은 수동으로 열지 않는다.

## 2. endpoint 스키마

각 dict의 필수 필드: `name,module,layer,horizon_s,target,match,row,y,p_ept,p_b1`. 배열은 모두 동일 길이의 1차원이며 `match,row`는 정수다. `row`는 M2의 frame row, M3/M4 OPEN의 unit row이고 endpoint 내에서 유일하다. labels나 경기 키 문자열은 공개 보고에 포함하지 않는다.

| name | layer | horizon_s | target |
|---|---|---|---|
| `M2/t0/H60/occurrence` | t0 | 60 | first_kill_ge1 |
| `M2/t0/H120/occurrence` | t0 | 120 | first_kill_ge1 |
| `M2/t0/H180/occurrence` | t0 | 180 | first_kill_ge1 |
| `M3/open/positive_value` | open | None | f0_gt0 |
| `M4/open/{dragon,baron,tower}/{blue,red}` | open | None | `{kind}_{team}_ge1` |

None은 고정 H초가 아닌 실제 교전/전환 관측창임을 뜻한다. M3 추가 배열은 `draw_truth,p_draw_ept,p_draw_b1,mc_se_ept,mc_se_b1,mc_se_draw_ept,mc_se_draw_b1`이다. `counts`에는 후보·eligible·제외·양성·음성·경기 수와 경기 가중 합, M3 무승부 수가 들어간다. 제외의 단위(시간 bin cell인지 unit인지)는 NLL count 필드에서 구분한다. 법적으로 빈 population은 길이 0의 배열과 사유를 갖는다. 예측하지 않은 행에 가짜 0 확률을 넣지 않는다.

## 3. M2: 종료로 잘린 관측창의 발생

H=60/120/180 각각 첫 H/10개 bin에서 `p = 1-exp(-sum(rate_q * observed_exposure_q))`, `y = 1[sum(count_q)>0]`. 노출 합이 양수인 행을 포함하며 완전 H초 관측 필터를 추가하지 않는다. 관측된 짧은 음성도 포함한다. 발생 정의의 기존 경계는 **T−30초**이며 단순한 T까지의 창과 구별한다. 실제 저장 노출을 사용하고 다시 window를 정의하지 않는다. 0노출 cell에서 양성 count가 있으면 오류다.

근거: `c_ept/data/head_rows.py:491–508`은 노출 양수 bin과 해당 발생수를 구성한다. `c_ept/models/nll.py:110–167`은 발생 T−30초/전환 T 경계와 rate×exposure를 설명·구현한다. `c_ept/models/baselines_b1.py:1232–1265`의 B2 목표도 관측 bin 안 발생이다. EPT `FrameAveraged.m2_rate` 출력은 이미 exposure를 곱한 평균 개수이므로 초당 rate로 되돌린 뒤 기존 노출을 정확히 한 번 곱한다 (`c_ept/ept/heads.py:641–661`).

## 4. M3 OPEN: 양의 f0 여부와 별도 무승부

M3 event-clock의 첫 킬 시각 f에서 `p+=P(f0(K,O)>0)`, `y=1[f0_unit>0]`를 보고한다. 음성에는 f0=0 무승부가 포함된다. `draw_truth=1[f0_unit==0]`, `p_draw=P(f0==0)` 및 각각의 MC SE를 별도 보고한다. 무승부 제외 조건화나 `p+/(p++p-)`로 바꾸지 않는다. `sign(E[f0])`는 p+가 아니다.

예측 사슬은 `size -> Poisson kills -> own-team kill-layer gold bin -> generated K -> Poisson O`다. size를 관측 크기로 고정하지 않는다. 킬0 팀의 골드는0, 그 외는 `min(k,6)`층의 고정 gold 대표값을 쓴다. Obj 입력의 K는 이 사슬에서 생성한 값만이고 골드 열을 천 단위로 한 번 변환한다. **실제 미래 K/O 또는 실제 f0는 p+/draw 예측 입력으로 사용하지 않는다.** 반면 NLL의 `P(O | K,s,x)` 항은 관측 K/s에서 관측 O의 조건부 우도를 평가하는 기존 head 정의이며 별도 경로다. 이는 미래 K를 주변 p+ 예측에 제공하는 것과 다르다.

예측 f0는 기존 Composer의 `gold_blue-gold_red + sum(O_cell * signed_price_cell)`를 사용한다. conversion은 포함하지 않고, 도움 풀은 gold에 이미 포함되므로 다시 더하지 않는다. tower의 예측 가격은 기존 tier 법의 기대가격이다. 열린 M3는 정확히 알려진 f 시점에서 `destroyed_before(f)`, `victim_counts`, `tier.prob(..., f/1000)`, `expected_tower_price`를 사용한다. 실제 미래 tower tier를 예측 가격에 넣지 않는다. 정답은 저장된 실제 tier 가격의 f0다.

근거: `c_ept/data/event_rows.py:101–115`(M3 ref=f, M4 ref=delta), `c_ept/valuation/compose.py:745–779`(생성 K, 골드층·대표값·f0의 계산), `c_ept/valuation/compose.py:536–560`(tier 기대가격), `c_ept/valuation/towertier.py:162–232,235,631`(과거 tower 상태·희생팀·기대가격), `c_ept/valuation/value.py:631–634,916–917`(실제 f0와 저장 labels). B1 `composer_heads('open')`는 명시적으로 미지원이다 (`c_ept/models/baselines_b1.py:1850–1858`). adapter는 `open` head의 `predict_blocks`와 canonical design을 사용하며 masked 대체가 없다 (`같은 파일:181–195,558–586,1087` 이후).

## 5. MC 근사와 대칭·전처리

K에 조건부인 O를 포함한 무한 Poisson 결합법의 정확 합산 대신 행당1000 draw, seed20261022를 사용한다. NumPy `default_rng(SeedSequence([seed,match,unit]))`로 행마다 초기화해 처리 순서와 batch 경계에 의한 RNG 변화를 제거한다. 두 모델은 같은 초기 stream을 쓰지만 Poisson rejection 소비가 달라질 수 있으므로 엄격한 common-random-number coupling이라고 주장하지 않는다. MC chunk 크기를 줄여 draw 수를 줄이지 않는다. guard 실패 시 예외로 끝난다.

MC SE는 각 Bernoulli indicator의 sample-variance 기준 `sqrt(p_hat*(1-p_hat)/(S-1))`다. p+와 draw에 각각 반환한다. 경계의 SE=0은 정확 확률의 증명이 아니며, 이는 경기 bootstrap CI와 별개의 수치 근사오차다. 경기 bootstrap은 고정 MC 예측을 대상으로 하므로 MC와 학습 seed 변동을 포함하는 CI가 아니다. float 연산의 합 순서도 기존 Composer의 cell-wise 곱·합을 사용한다.

EPT는 `ViewTensors(..., parent_stats)`와 `FrameAveraged`를 통해 원본/교환 frame의 law를 평균한다. 가중치나 통계는 수정하지 않는다. 원본 source: `c_ept/ept/data.py:493–545,603` 이후, `c_ept/ept/heads.py:625–690`. B1는 원래 dropped column과 `_assemble/_m4_design`을 사용해 학습 때의 category/골드 단위를 보존한다 (`c_ept/models/baselines_b1.py:304–360,1193–1224`).

## 6. M4 OPEN: delta에서의 실제 K/O 조건부 전환

실제 교전 결과 K/O가 알려진 delta의 M4 보기에서 종류×팀별 `y=1[count>=1]`, `p=1-exp(-rate*actual_exposure)`를 계산한다. exposure>0 unit만 포함한다. 게임 종료 T로 잘린 노출을 쓰며 완전 conversion window 필터를 추가하지 않는다. 현재 원본 HeadRows 계약에 별도 object-up 필터는 없다. 계약 외의 새 가용성/위험집합 mask를 임의로 추가하지 않는다.

`C.O_KINDS=('dragon','elder','grubs','herald','baron','atakhan','tower','inhibitor','plate')`, cell=`2*kind_index+team`, blue=0/red=1. 따라서 dragon=0/1, baron=8/9, tower=12/13이다 (`c_ept/config.py:56`). elder를 dragon에 합치지 않으며 tower count의 tier도 새 endpoint로 분해하지 않는다. 단일 cell의 Poisson 확률만 사용하고 종류/팀 확률을 임의로 합치지 않는다.

근거: `c_ept/data/head_rows.py:342–360,529–548`은 양의 conversion 노출과 M4 event view 유무 일치, 실제 K/O 조건을 보장한다. `c_ept/ept/heads.py:682–690`은 FrameAveraged M4를 적용한다. M4 Brier는 이번 탐색 진단이며 원 사전등록 주 지표를 교체하지 않는다.

## 7. NLL·common_head·가중

NLL record는 `name,match,row,nll_ept,nll_b1,common_head,counts`이며 y는 없다. name은 canonical `Head/kind`이다. raw NLL을 보존하며 scales가 있으면 `scale,nll_ept_normalized,nll_b1_normalized`를 추가한다. 각 head에서 양 모델의 row/match/target/exposure/team/K/O/codes/view를 대조하고, scored mask 뒤의 평가 row/match를 다시 대조한다. 불일치를 inner join으로 버리지 않는다. 원래 조건부 likelihood를 사용하고 M2 bin·M3Gold team 항은 HR-5에 따라 같은 평가 row 안에서 합한다 (`c_ept/data/head_rows.py:550–595`, `c_ept/ept/loss.py:235–312`).

**common_head=True 조건:** 위 계약 비교를 통과한 같은 head·같은 layer의 쌍이며 parent-run 고정 scale이 제공되어야 하고, 원 설계8절 G9b 구조 비교의 **전체 층 M2 + 열린 보기 M3/M4**에 속해야 한다. 정확한8개는 `M2Rate/masked`, `M2Mark/masked`, `M2Zone/masked`, `M3Size/open`, `M3Kills/open`, `M3Gold/open`, `M3Obj/open`, `M4Conv/open`이다. 나머지5개 masked M3/M4 block의 raw/정규화 NLL은 개별 진단으로 보존하되 구조 평균에는 넣지 않는다. 근거 원본은 `C:/Users/todtj/PycharmProjects/LOL_teamfight_new/docs/superpowers/specs/2026-10-20-c-ept-design.md:285,299`의 평가 층과 구조 판정 행이다. `common_head_basis`와 provenance의 `structural_heads`가 선택 근거를 기록한다. scale 없는 raw-only 호출에서는 False로 반환한다. scale을 DEV에서 새로 계산하지 않는다.

각 endpoint의 최종 eligible 행 집합에서 경기 m의 행마다 `w=1/n_m`, 동일 가중으로 EPT/B1 비교다. AUC는 이 가중을 사용한 pooled 양성-음성 쌍 비교이며 경기별 AUC 평균 또는 within-match AUC가 아니다. Brier는 `sum(w*(p-y)^2)/sum(w)`. NLL은 HR-5 row 합 이후 같은 경기 균형으로 요약한다. 구조 정규화 NLL은 고정 scale로 나눈 명시 공통 head들의 경기 가중 평균을 다시 동일 head 가중으로 평균한다. 이 값은 학습의 13개 block 합이나 M3 expert의 규모별 평균 training/STOP loss와 같지 않다. 일부 head가 비면 해당 head를 평균에서 조용히 빼지 않고 전체 구조 비교도 unavailable로 남긴다.

## 8. 합법 빈 집합과 불완전 입력의 구별

DEV canonical frame/onset/in-scope-unit mask 자체가 비거나, 유효한 HeadRows가 exposure/kill-team 조건에 따라 빈 행을 반환하면 합법 빈 집합이다. adapter는 model prediction을 생략하고 길이0 배열, eligible0, `unavailable_reason`을 반환한다. M4가 모두 종료 직전 교전이라 conversion window가 없다면6개 M4 endpoint 모두 unavailable이다. 한 클래스만 남는 경우는 빈 집합이 아니므로 Brier는 가능하고 AUC만 unavailable이다.

필수 array 부재, 양의 exposure인데 event/view가 없음, 0exposure인데 M4 view가 있음, 범위/target/condition/row 불일치, 잘못된 확률, 비유한 NLL은 예외다. 예외 문자열을 catch해서 빈 집합으로 바꾸지 않는다. 빈 head의 common_head/scale 표시는 계약의 동일성만 나타내며 실제 측정 가능성을 만들어내지 않는다. metrics는 빈 head의 NLL과 해당 구조 평균을 null/unavailable로 낸다.

## 9. 검증·해석·다음 담당자

합성 tests는 real EPT/FrameAveraged, real B1의 상수 fallback prediction classes와 canonical HeadRows/ViewTensors를 연결한다. 모델 fit/load/optimizer step/composer fallback은 sentinel로 금지한다. 합성 FIT 입력에서 fixture stats만 생성하며 실제 parent stats나 DEV payload를 읽지 않는다. M2 censoring, M4 taxonomy, 생성 K·own-team 골드층과 단위, draw/MC, 실제 미래 K/O 변경에 대한 M3 예측 불변성, B1.nll HR-5 원본 결과와13개 head 일치, mode/parameter 보존, guard 선행 실행, 합법 empty M4와 불완전 view의 구분을 검사한다. `run_inference -> summarize_endpoints -> write_report(render=False)` 연결은 합성 임시 디렉터리에서 JSON/CSV/SVG를 만든다.

명령(환경은 각 PowerShell 변수에 명시):

```powershell
$env:CUDA_VISIBLE_DEVICES=''
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONUTF8='1'
& 'C:/Users/todtj/anaconda3/python.exe' -B -m pytest tests/test_ept_dev_adapters.py tests/test_ept_dev_metrics.py -q -p no:cacheprovider
```

개발 위치는 `C:/Users/todtj/.codex/worktrees/ept-dev-evaluation/LOL_teamfight`다. 첫 시도는 sandbox 임시폴더 권한 실패였으며 정상 임시폴더 접근으로 합성 검사를 수행했다. 초기 empty 통합 assertion 한 건은 metrics의 실제 unavailable 표현(`estimate=None`, `reason=no_rows`, ROC status)에 맞게 수정했다. 최종 실행 결과는 구현자 반환 메시지에 기록한다. 독립 reviewer는 네 파일 diff와 별도 synthetic probes를 검토해야 한다. production Check/Start와 실제 모델·dataset payload 검증은 수행하지 않았다.

이번 비교는 **기존 whole-FIT B1 대 FIT/STOP25% EPT 단일 pilot**이다. 자료량을 통제한 architecture 비교, 학습 seed 변동을 포함한 일반화 증거, TEST 성능, G1–G5·전체 설계 성공 또는 인과 효과로 해석하지 않는다. runtime/guard/best-only/loader는 lead 소유, metrics/report는 별도 담당자 소유다. 승인된 실제 실행 전에 independent review, code/config/decision identity와 종료 gate를 모두 연결해야 한다.
