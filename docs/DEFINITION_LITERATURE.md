# 정의 방법론 문헌 분석 (원문 정독, 2026-09-08)

`DEFINITION_EVIDENCE.md` 13절의 인용 묶음을 원문으로 확인하고, 각 논문이 **정확히
무엇을 했는지**, 우리 방법과 **어디가 같고 어디가 다른지**, **어느 문장에 어떻게
인용할지**를 적는다. 인용문은 원문 그대로다. PDF 사본은
`D:/LOL_Project/references/`에 있고 BibTeX는 `docs/references/definition_refs.bib`.

| 논문 | 사본 | 정독 범위 |
|---|---|---|
| Halfaker et al. 2015 WWW | Halfaker2015_UserSessionIdentification_WWW.pdf | 전체 |
| Mehrzadi & Feitelson 2012 SYSTOR | MehrzadiFeitelson2012_SessionData_SYSTOR.pdf | 전체 |
| Zaliapin & Ben-Zion 2021 SRL (2013 JGR의 방법 요약) | ZaliapinBenZion2021_ClusteringDeclusteringPerspectives_SRL.pdf | 1~12쪽 |
| Pedrassoli Chitayat et al. 2023 AIIDE | PedrassoliChitayat2023_BeyondTheMeta_AIIDE.pdf | 전체 |
| Jacobs & Wallach 2021 FAccT | JacobsWallach2021_MeasurementAndFairness_FAccT.pdf | 전체 |
| Berman et al. 2014 J R Soc Interface | Berman2014_StereotypedBehaviour_JRSInterface.pdf | 1~7쪽 |
| Schubert 2016, Ke 2022, Tot 2021, Kleinman 2020, Yang 2014 | 같은 폴더 | 9/7 정독 (메모리 `teamfight-definition-literature`) |

---

## 1. Halfaker et al. 2015 — 활동 간 시간의 규칙성에서 세션 경계

**무엇을 했나.** 7개 시스템 12개 데이터셋(AOL 검색, Wikimedia 페이지뷰·편집, OSM,
MovieLens, Stack Overflow, **League of Legends 경기 간 시간**)에서 사용자 활동 간
시간을 log 척도 히스토그램으로 그리고, EM으로 가우시안 혼합을 맞춘 뒤 두 성분이
등확률인 점을 세션 임계값으로 잡았다.

- 방법 (§3.1): "we plot a histogram based on the logarithmically scaled inter-activity
  time and look for evidence of a valley." → "we try to fit a two component gaussian
  mixture model using expectation maximization" → "we identify a theoretically optimal
  inter-activity threshold ... by finding the point where inter-activity time is equally
  likely to be within the gaussians fit with sub-hour means (within-session) and
  gaussians fit with means beyond an hour (between-session)."
- 결과 (§4.1, Table 1): 대부분 쌍봉, within 성분 모드 약 1분·between 성분 모드 약 1일,
  교차점 29~115분 → 경험칙 1시간. **LoL 경기 간 시간은 예외**(§4.4): 매우 좁은 5분
  성분과 넓은 5시간 성분, 교차 약 14분. 큐 시스템이라는 "system's limitations on user
  behavior"가 분포 모양을 만든다고 해석.
- 실패 사례 정직 보고 (§4.4): Stack Overflow는 "does not suggest a clear valley".
- 권고 (§5): "we still advise that any new application ... is preceded by a plot of a
  histogram of log-scaled inter-activity times and visual inspection for a natural valley."
- 일반성 주장의 형식 (§5): "(1) grounded in empirical observations of a natural valley
  ... and (2) holds constant across a wide range of systems and activity types."

**우리와의 대응.**
| Halfaker | 우리 |
|---|---|
| log 활동 간 시간 히스토그램 | log 연속 킬 간격 밀도 |
| 2성분 가우시안 혼합, 등확률점 | 혼합으로 성분 위치, KDE 최소값(`kde_valley`), 혼합 교차점도 보고 |
| 도메인별 재추정, 예외 도메인 명시 | 패치별·단계별 재추정, 후반 단봉 명시 |
| 경험칙 1시간 + "먼저 히스토그램을 그려라" | spec + drift 판정 자동화 |

**차이.** 그들은 혼합 성분의 교차점을 쓰고 우리는 성분 사이의 비모수 골짜기를 쓴다.
우리 3,000경기 이하 표본에서 봉우리 높이로 고르는 방식이 깨졌던 경험이 그 이유이고,
두 값을 모두 보고한다(13.3 vs 14.3 s).

**인용 자리.** 시간 경계 절 첫 문장. LoL 데이터가 그들 논문에 이미 있다는 점을 한
줄 언급하면 좋다(경기 *간* 시간이지 킬 간격은 아님).

---

## 2. Mehrzadi & Feitelson 2012 — 전역 임계값의 인공물과 개별 임계값

**무엇을 했나.** AOL 2006 검색 로그에서 전역 세션 임계값(10·20·60분)이 만드는 인공물을
보이고, 사용자별 간격 히스토그램(log2 구간)에서 자연 골짜기를 찾는 알고리즘을 제안,
사람 판단·주제 전환과 비교했다.

- 핵심 관찰 (§3): 전체 간격 분포는 "there is no unique spot that would appear to be a
  natural threshold value"; 임계값을 쓰면 "The distribution of resulting session
  durations has a pronounced break at the point of the threshold used to create it."
- 결론 (§7): "the global distribution of inter-activity intervals is typically smooth,
  with no natural threshold value. Worse, using a global threshold may lead to
  artifacts that directly reflect the chosen threshold value." 반면 개별 사용자 분포는
  "often do display a structure including a natural threshold."
- 알고리즘 (Fig. 4): 2의 거듭제곱 구간 히스토그램, 후보 구간을 512~8192 s로 제한,
  양쪽 최대값 대비 깊이로 점수, 최고점 선택, 동점이면 1200 s에 가까운 쪽.
- 평가의 순환성 경고 (§6): "This risks a circular argument, where breaks in activity are
  used both to find session boundaries and to evaluate whether the found boundaries are
  reasonable." → 주제(n-gram) 전환이라는 독립 신호로 검증.
- 사람 판단 비교 (§5): 4,992 간격 중 1,334 합의, 알고리즘 정밀도 83%, 사람은 1~2시간
  임계값을 더 일관되게 씀.

**우리와의 대응.** 후보 구간을 제한하고 그 안에서 깊이로 최소값을 고르는 것은 우리
`kde_valley`의 "혼합 성분 사이에서 최소" 탐색과 같은 구조다. 순환성 경고는 우리에게
그대로 적용된다: 킬 간격으로 경계를 정하고 킬 간격으로 평가하면 순환이다. 우리는
**참여 챔피언 공유**라는 독립 신호로 공간 경계를 검증했고, 시간 경계에는 ARI 평원과
패치 안정성을 썼다. 이 점을 논문에 명시할 것.

**차이.** 그들은 사용자별 임계값을 권한다. 우리의 대응물은 "게임 단계별 G"인데,
후반은 단봉이라 개별 임계값이 정의되지 않는다. 그래서 풀링 값 + 평원 보고를 택했다.

**인용 자리.** 시간 경계 절, "고정 임계값의 인공물"과 "독립 신호로 검증" 두 문장.

---

## 3. Zaliapin & Ben-Zion 2013 JGR / 2021 SRL — 최근접 시공간 근접도의 쌍봉

**무엇을 했나.** 각 지진에서 이전 지진까지의 근접도를 η = Δt · r^d (Δt 사건 간 시간,
r 지표 거리, d 진앙 분포의 프랙탈 차원 ≈ 1.6)로 정의. 최근접 근접도 분포가 쌍봉이고
낮은 모드가 군집(여진·군발), 높은 모드가 배경 사건. 2013 JGR: 남캘리포니아
111,981개 사건을 41,393개 통계적으로 유의한 군집으로 분리, 방법의 수치 매개변수·
규모 하한·카탈로그 불완전성·위치 오차에 대해 안정.

- SRL §2.1 (l.143–147): "The proximity from any earthquake to an earlier event is
  defined as a product of the interevent time Δt and surface distance r raised to the
  fractal dimension d of the events, η = Δt r^d. The distribution of the nearest-neighbor
  proximity is prominently bimodal in various observed and model catalogs."
- 매개변수 무관성 (l.151–152): "Here we use d = 1.6; but the bimodal nature of the
  proximity is independent of the parameter selection (Zaliapin and Ben-Zion, 2013)."
- 시공간 분리 (l.154–156): "The two modes are also well separated in space-time. The
  cluster mode (lower proximity values) is mainly comprised of aftershock sequences and
  swarms."
- 경계의 부드러움 (l.157–159): "The declustering involves stochastic thinning based on
  the proximity values; this helps avoiding a sharp cut along the model separation line,
  which might result in artificial inhomogeneities."
- 겸손한 결론 (Abstract): "Declustering remains at present an exploratory tool, rather
  than a rigorous optimization problem, and selecting an appropriate declustering method
  should depend on the data and problem at hand."
- 안정성 보고 방식 (l.265–273): Gini 계수가 규모 하한 2단위, 사건 수 2자릿수, 시간
  이산화 2자릿수, 공간 셀 1자릿수 변화에 대해 ±0.1 안.

**우리와의 대응과 실제 검증.** 우리는 (Δt, Δd)를 직사각형으로 자르고 그들은 곱
η로 스칼라화한다. 같은 킬 쌍 데이터(세 패치, Δt ≤ 120 s)에 η = Δt·Δd^d를 적용해
보았다(`features/fight_boundary/eta_proximity.json`):

| d | log η 쌍봉 | 골짜기 깊이 | 같은 싸움(공유) 구분 AUC |
|---|---|---|---|
| 0 (시간만) | 예 | 0.87 | 0.631 |
| 1.0 | 예 | 0.93 | 0.764 |
| 1.5 | 예 | 0.93 | 0.795 |
| 2.0 | 예 | 0.93 | 0.814 |
| 직사각형 13.6 s × 4,263 u | | | 연결 시 공유 0.937 / 분리 시 0.513 |
| 스칼라 d=2, 같은 연결률 | | | 연결 시 0.933 / 분리 시 0.517, 직사각형과 92% 일치 |

**지진학의 스칼라 근접도를 우리 데이터에 그대로 적용해도 쌍봉이 나오고, 결과
분할은 우리 직사각형과 92% 일치하며 직사각형이 근소하게 낫다.** 이는 방법의
동형성을 실증하는 문장으로 쓸 수 있다.

**차이와 배울 점.** (1) 그들은 최근접 이웃(가장 가까운 이전 사건)을 쓰고 우리는
연속 쌍을 쓴다. 동시 다발 싸움에서는 최근접 이웃이 더 맞을 수 있다. (2) 경계에서
확률적 씨닝을 써서 날카로운 절단의 인공물을 피한다. 우리 ARI 평원 보고는 같은
걱정의 다른 답이다. (3) 안정성을 "매개변수를 자릿수 단위로 흔들어도 결과가 ±ε"로
보고하는 형식이 우리 부트스트랩·대역폭 스윕·패치 안정성 표와 같다.

**인용 자리.** 시간·공간 경계 절의 방법 문장, 그리고 η 실험을 한 단락으로.

---

## 4. Pedrassoli Chitayat, Block, Walker, Drachen 2023 AIIDE — 패치와 모델 수명

**무엇을 했나.** Dota 2 패치 7.27~7.33의 게임 디자인 파라미터(OpenDota constants
저장소의 과거 버전)를 파싱해 능력(ability) 속성표를 만들고, K-means(K=68, 엘보+
실루엣)로 능력을 군집화한 뒤 캐릭터를 군집 카운트 벡터로 표현. 새 캐릭터(7.32)와
새 속성 유형(7.33) 패치를 홀드아웃해, 캐릭터 ID one-hot(NN2)은 깨지지만 군집 표현
(NN3)은 정확도를 유지함을 보임(61,254 프로 경기).

- 문제 진술 (Abstract): "Compared to traditional sports, esport titles change rapidly,
  in terms of mechanics as well as rules. Due to these frequent changes to the
  parameters of the game, esport analytics models can have a short life-spam, a problem
  which is largely ignored within the literature."
- 변화의 세 종류 (p.5): breaking(아키텍처를 바꿔야 함) / impactful(성능에 실질 영향)
  / unimpactful. "differentiating between impactful and unimpactful changes may require
  analysis."
- 재군집 조건 (p.21): "significant changes to the core game design ... could impact the
  cluster themselves and change the density distribution of the clusters. In this case
  new clustering may be required."
- 자료 공개: github.com/ChitaAPC/Dota2CCR.

**우리와의 대응.** 그들의 "impactful vs unimpactful 판별에는 분석이 필요하다"가 우리
drift_decision이다. 그들의 게임 디자인 파라미터 파싱은 우리 Data Dragon 사거리
커버리지 점검과 같은 자료원 활용이다. 우리는 표현이 아니라 **정의(단위)**의 패치
불변성을 다룬다는 점이 새로움이고, 그것을 이 논문의 확장으로 자리매김할 수 있다.

**인용 자리.** 파이프라인 절 도입, "패치마다 정의를 재추정하고 판정한다"는 문장.

---

## 5. Jacobs & Wallach 2021 FAccT — 측정 모형과 타당도

**무엇을 했나.** 사회과학의 측정 모형(measurement modeling)을 전산 시스템에 도입.
관측 불가한 구성개념(construct)을 관측 속성으로 조작화(operationalize)할 때 생기는
불일치를 신뢰도·타당도로 검사하자는 제안.

- 핵심 (Abstract): "This process, which necessarily involves making assumptions,
  introduces the potential for mismatches between the theoretical understanding of the
  construct purported to be measured and its operationalization."
- 검사의 두 축 (§3, Quinn et al. 인용): "The evaluation of any measurement is generally
  based on its reliability (can it be repeated?) and validity (is it right?)."
- 신뢰도 (§3.1.1): test–retest — 구성개념이 변하지 않았다는 가정 하에 다른 시점의
  측정이 같은가.
- 타당도 7측면 (§3.2): face(그럴듯한가) / content(계약성·실질·구조: 구성개념을 온전히
  담는가) / convergent(기존 측정과 상관) / discriminant(다른 구성개념과 부당하게
  상관하지 않는가) / predictive(관련 속성을 예측하는가) / hypothesis(실질적 가설을
  지지하는가) / consequential(사용의 결과).
- 정도의 문제 (§3.2): "construct validity is always a matter of degree, to be supported
  by critical reasoning."
- 순환 경고 (§3.2.3): 검증 안 된 다른 측정과의 수렴은 "a false sense of security".

**우리와의 대응 (정의 절의 뼈대).**
| 타당도 | 우리 검사 |
|---|---|
| face | 미니맵 오버레이: R은 둥지 크기, D는 인접 타워까지 |
| content | 커뮤니티 정의("entire team")·Schubert/Ke 정의와의 대응; 킬 없는 encounter를 명시적으로 제외 |
| convergent | Ke 2022 team fight와 동형; Zaliapin η 분할과 92% 일치 |
| discriminant | 공유율: 연결 시 0.94 vs 분리 시 0.51 — 다른 싸움을 같은 싸움으로 묶지 않음 |
| predictive | 라벨 AUC .746; 경기 승자 정보량(M0/M1/M2, 예정) |
| hypothesis | 후반 AUC 상승, 규모별 차이 |
| consequential | 킬 없는 한타 제외가 결론을 바꾸는가(4.9% 상한) |
| test–retest | 패치 3개, 시드, 부트스트랩, ARI 평원 |

**인용 자리.** 정의 절 도입 문단과 마지막 "validity summary" 표.

---

## 6. Berman, Choi, Bialek, Shaevitz 2014 — 데이터가 정하는 행동 범주

**무엇을 했나.** 초파리 자세 시계열 → 웨이블릿 스펙트로그램 → t-SNE 2차원 임베딩 →
가우시안 평활 밀도 → watershed로 122개 봉우리 영역. 각 영역이 사람이 알아보는 행동에
대응. 봉우리 근처의 "pause"가 정형 행동, 빠른 이동이 비정형.

- 문제 제기 (§I): 기존 방법은 "assume, a priori, that stereotyped classes of behavior
  exist without first showing, from the data, that an organism's actions can be
  meaningfully categorised in a discrete manner."; "Ideally, a behavioural description
  should manifest itself directly from the data, based upon clearly-stated assumptions,
  each with testable consequences."
- 결과 (§IV.B): "the segmentation of the movies into behavioural categories has emerged
  from the data itself, not through a priori definitions."
- **쌍봉으로 상태 구분** (§IV.A, Fig. 5b): 임베딩 공간 속도 분포가 "well-described by a
  two-component log-normal mixture model in which the two peaks are separated by almost
  two orders of magnitude" → pause/move. 우리 킬 간격 쌍봉과 같은 논리.
- 안정성 (§IV.D): "results are consistent across a large range of the behavioural-map
  smoothing parameter σ" — 우리 대역폭 스윕과 동일.
- 검증의 한계 인정 (§IV.D): "Inherent in any unsupervised classification method is the
  question of how to validate its accuracy. Here, there is no ground truth ... a
  significant aim of our work is to dispense with a priori behavioural definitions."

**우리와의 대응.** 규모 클래스를 참여 인원 분포의 봉우리(2v2, 5v5)에서 읽고, 골짜기가
없는 축(min 2 vs 3)의 컷은 관례로 명시한 것. 그리고 "정답이 없는 비지도 정의를
어떻게 검증하는가"에 대한 그들의 답(개체 간 일관성, 평활 매개변수 불변성, 사람이
알아보는 범주와의 부합)이 우리 답(패치 간 일관성, 대역폭 불변성, 커뮤니티 용어와의
부합)과 같은 구조.

**인용 자리.** 규모 클래스 절, 그리고 "비지도 정의의 검증" 문장.

---

## 7. 그 밖의 확인 사항

- **McQueen, Wiens, Guttag 2014 (MIT SSAC)**: 온볼 스크린을 규칙 기반 후보 위에
  학습 분류기로 인식, 민감도 82%·PPV 80%, 14경기 252분. 사건 정의를 데이터로 배운
  스포츠 선례. 규모 클래스 절이나 관련 연구에 한 줄.
- **Kulldorff 2001 (JRSS A)**: 원통형 시공간 스캔 통계, 고정 창 없이 우도 최대 창을
  찾고 Monte Carlo로 유의성. 우리 직사각형의 형식적 대안으로 한 줄.
- **Catledge & Pitkow 1995**: Halfaker §2.2 요약 그대로 — 평균 9.3분 + 1.5σ = 25.5분이
  30분으로 굳음. "데이터에서 나온 값이 재추정 없이 관례가 된" 반례.
- **Schubert 2016**: combat range = 관측 피해거리 평균+1SD(약 85%), encounter의 81.1%
  킬 없음, decisive encounter만 분석 (9/7 정독).

---

## 8. 정의 절에 실제로 넣을 문장 (초안은 `paper/tog/definition_section_draft.tex`)

1. 도입: 측정 모형 틀 [Jacobs & Wallach].
2. 시간 경계: 쌍봉 골짜기 [Halfaker; Mehrzadi & Feitelson; Zaliapin & Ben-Zion];
   순환성 회피를 위한 독립 신호 [Mehrzadi & Feitelson §6].
3. 공간 경계: 사거리 분위 [Schubert]; 공유율 50% = 베이즈 결정점; η 검증 [Zaliapin].
4. 안정성: 매개변수 흔들기 [Zaliapin 2013], 평활 불변 [Berman], ARI 평원 [Campello].
5. 패치: 재추정과 판정 [Pedrassoli Chitayat].
6. 규모 클래스: 봉우리 = 범주, 골짜기 없는 축은 관례 [Berman].
7. 제외: 킬 없는 encounter [Schubert 81.1%; 우리 4.9%].
