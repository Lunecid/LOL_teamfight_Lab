"""EPT의 학습 고리 (S4a 과제 5): AdamW와 예열·코사인 학습률, FIT 주변 분포로 시작하는 출력 편향, 평가 행·경기 가중의 STOP 값과 조기 종료, 하드 중단에서도 이어지는 체크포인트, 시간·메모리 회계.

이 모듈은 과제 5 작업서(AdamW, 예열·코사인, 인내, STOP으로만 조기 종료), 계획 Q8(손실)·Q10(비용 점검)·Q3(CPU 결정적 보고 값; 지침 A6a), 설계서 7절(운영 상한, 체크포인트·이어하기)과 계약 v4를 구현한다.
이 모듈은 킬이 하나 이상 난 교전을 다룬다 (G-S3c C3). "블록"은 ``loss.BLOCKS``의 열세 개이다.
방법. FIT 배치마다 열세 블록의 행 점수를 ``loss.total_loss``로 합쳐 (LOSS-4) 한 번 갱신한다 (TRN-7). 한 에폭 뒤 STOP을 평가 모드로 한 번 훑어 STOP 값 (TRN-2)을 구하고, 가장 낮았던 에폭이 최선이며
인내 에폭 동안 낮아지지 않으면 멈춘다 (TRN-3). 반환 모형은 최선 에폭의 것이고 이어하기는 마지막 에폭의 모형·최적화기·난수 상태에서 한다. 전원 차단으로 닫히지 못한 호출은 다음 호출이 박동 줄에서 헤아려 입양한다 (TRN-6).

식 (식 번호는 ``(TRN-n)``; 기호 표는 식을 계산하는 함수의 문서 문자열에 있다)

    (TRN-1)  η_t = η_max (t + 1) / n_w  (t < n_w),   η_t = η_max · ½ [1 + cos(π (t − n_w) / max(1, n_T − n_w))]  (t ≥ n_w)
    (TRN-2)  Ω_{h,u,j} = 평균_{i ∈ I⁺} ℓ_i ;   L_{h,s} = (1 / M_{h,s}) Σ_u [Σ_j Ω_{h,u,j}] / n_{h,s,u} ;   L_h = 나온 s에 대한 L_{h,s}의 평균 ;   Loss_STOP = Σ_h L_h / ℓ̄^B0_h
    (TRN-3)  최선 갱신 ⇔ Loss_STOP < 지금까지의 최소 ;   개선 없는 에폭 수 n_bad ← 0 (갱신), n_bad + 1 (아니면) ;   n_bad ≥ n_pat이면 종료
    (TRN-4)  γ_c = log[(n_c + ζ₀) / (N + C ζ₀)]  (범주 머리)
    (TRN-5)  γ = log max(Σ y / Σ e, λ_min)  (포아송 머리; 규모·팀별 킬은 그 규모의 평균)
    (TRN-6)  Z_n^sec = Z_{n−1}^sec + beat_n^sec + Δ_beat ;   Z_n^peak = max(Z_{n−1}^peak, beat_n^peak)  (닫히지 못한 호출 n; 박동이 없으면 beat_n = 0)
    (TRN-7)  AdamW 한 걸음: θ ← θ (1 − η_t ϑ) − η_t · A₁ / (√A₂ + ε_A),  A₁, A₂는 기울기와 그 제곱의 편향 보정한 지수이동평균
    (TRN-8)  매개변수마다의 감쇠: θ_i ← θ_i (1 − η_t ϑ_i) − …,  ϑ_i = ϑ (모든 i; decay_groups = 'all')  또는  ϑ_i = 0 (i ∈ N), ϑ (i ∈ D) (decay_groups = 'split')
    (TRN-9)  감쇠 척도 기록: κ_r = exp(log κ_r), 에폭의 마지막 걸음 뒤 (decay_groups = 'split'일 때 에폭마다; epoch_metrics의 열 kappa_r)
    (TRN-10) 학습의 틀 (swap_mode): 'average'  L = ℓ(ȳ(b), y), ȳ는 두 틀의 평균 출력 (EH-9) ;  'random_view'  L = ℓ(ŷ(F_c(b)), F_c(y)), c_v ~ Bernoulli(½) 보기마다 (DAT-11), 걸음마다 한 틀 ;
             'none'  L = ℓ(ŷ(b), y). 평가는 eval_average = True이면 ȳ, False이면 ŷ(b)
    (TRN-11) 상한 동작 (on_wall_cap): 'fail'  Z_k^sec > W 이면 ResourceBoundError (이전 실행) ;  'best_stop'  예산 B_k = 1[ Z_k^sec ≥ W ∧ ¬(n_bad ≥ n_pat) ∧ k < K ] 이면 k번째 에폭을 마치고 done = 참, 최선 STOP 체크포인트는 그대로,
             머리에 budget_end (에폭 지표의 termination_reason = 'budget'). 메모리 상한·측정 불가·정체성·무결성 위반은 두 설정 모두 실패

      기호      | 뜻                                                          | 단위 | 범위      | 코드 변수
      Z_k^sec   | 에폭 k의 STOP 측정 직후까지 모든 호출의 시간 합              | s    | ≥ 0       | res.seconds()
      W         | 벽시계 상한 (24 h)                                          | s    | 86,400    | WALL_CAP_SECONDS
      K         | 일정의 에폭 수 (코사인이 끝나는 곳, 에폭 예산)               | 에폭 | ≥ 1       | epochs
      B_k       | 에폭 k 뒤에 예산으로 끝내는가                                | —    | {0, 1}    | budget

    해석: 'best_stop'에서 상한은 걸음 사이에서 끊지 않고 에폭이 끝난 뒤에 본다. 체크포인트·최선 STOP 상태·원장(호출 표지의 에폭 수)이 모두 에폭 단위여서 에폭 한가운데서 끊으면 버려야 하는 일이 생기고, 끝난 에폭의
    STOP 값은 이미 쟀기 때문이다. 그래서 시간 합은 상한을 한 에폭(FIT, STOP, 그 체크포인트 쓰기)에 앞 에폭의 체크포인트 쓰기 한 번을 더한 만큼까지 넘을 수 있고 그 여유는 감독자가 채운다(시작 때 남은 예산 ≥ 상한 + 마무리 여유, K5). 다른 규칙이 같은 에폭에 닿았으면(조기 종료, k = K) 그 이유를 쓴다. 경계: 'fail'은 정확히 W까지 허용하고(Z > W만 거부), 'best_stop'은 Z ≥ W에서 끝낸다(상한에 닿은 것이지 넘은 것이 아니므로 거부가 아니라 끝이다).

가정과 선택

    · 가중치 감쇠의 묶음 (S4b 준비 M4, 저자 결정 D6, 검토 A2-F2): N = {log κ, 모든 편향(Linear 편향, 머리의 은닉·출력 편향, type_bias), LayerNorm의 이득·편향, Embedding 표},
      D = {Linear 가중 행렬, 공동 참여 가중 w·w_back, 학습 지형장 Ω_learn}. 'all'(기본값)은 이전 실행의 한 묶음이다 (``param_groups``, ``TrainConfig.decay_groups``).
    · STOP 값 (R-S4a-16, 수정): 평가 행은 점수 있는 항목의 평균이고(M3Gold의 팀 항목도 평균이어서 B0 척도와 같은 항목 단위이다), 경기는 규모 층 안에서 1/n_m로 가중하며, 층 평균은 전문가 셋(``SIZE_NORMALISED``)만 낸다.
      배치 평균의 평균이 아니어서 배치를 어떻게 자르든 값이 같고, 학습 손실과 비교할 수 없으며 새로운 B0 상대 주장이 아니다. 학습 행 가중은 같다 (계획 P9).
    · 출력 편향은 FIT 주변 분포에서 시작한다. 표지 머리의 스칼라 하나는 소프트맥스에서 상쇄되므로 바꾸지 않는다 (``fit_biases``).
    · 학습의 틀 (S4b 준비 M7, 저자 결정 D10 P1·D17, 계획 Q3): swap_mode 'average'(기본값)는 보기마다 두 틀을 평균해 부호화기를 두 번 돌리고, 'random_view'는 보기마다 확률 ½로 고른 한 틀만 돌리며(입력과 그 보기를 읽는
      행의 조건·정답을 함께 맞바꾼다; 동전은 배치가 시드·에폭·배치 번호에서 뽑고 전역 난수는 쓰지 않는다), 'none'은 기록된 틀만 쓴다. 평가(STOP 값 포함)는 eval_average=True(기본값)이면 모든 swap_mode에서 두 틀의
      평균이고, False이면 기록된 틀만 쓴다(D17의 대칭 없음 점검은 'none'과 False). 출력 편향의 시작값은 swap_mode와 상관없이 기록된 틀의 FIT 주변 분포이다(시작값일 뿐이며 학습이 고친다).
      'random_view'의 틀은 CPU 배치에서 먼저 적용하고 복사한다 (S4b 속도 개선 SPD 항목 1, 결정 D32, ``batch_prep.prepare_batch``): 모형은 같은 텐서를 비트까지 같게 읽고(틀의 모든 연산이 정확한 연산이며 난수를 쓰지 않는다)
      장치 경로에서 틀을 맞바꾸는 작은 연산과 동기화가 사라진다.
      걸음의 점검도 같은 방식으로 나눈다 (항목 3, ``batch_prep.Preparation``, ``step_guard``): 자료 점검(범위·유한성·부호·모양)은 배치를 만들 때 CPU에서 한 번, 머리와 손실의 점검 함수 그대로 거치고(``validate_rows``),
      평균에 드는 행의 번호 목록(``loss.counting_plan``)과 공유 M2 호출의 보기 합집합을 함께 만든다. 걸음은 그 배치를 믿고(``step_guard.trust``) 장치에서 입력 값을 읽어 되묻지 않으며, 모형 점검(평균이 음수이거나 유한하지
      않음, 로그 법칙이 합 1이 아님)은 칸 수를 미뤄 두었다가 걸음당 한 번, 역전파 앞에서 손실의 유한성과 함께 읽는다(``training_step.optimize_batch``). 막히는 조건·오류·문구·시점(가중이 바뀌기 전)은 그대로이고 값도
      그대로이다. STOP 평가와 이 모듈 밖에서 부르는 점검은 믿지 않고 이전과 같이 모두 점검한다.
    · 보고 값은 CPU 결정적 float32 실행이다 (계획 Q3). 이어하기 정체성은 설정, 입력 행, S2 ``STORE.json``의 해시 (R-S4a-18, 수정), 장치·dtype·환경·torch 버전에 묶이고 코드 버전은 아니다 (``PRECONDITION``).
    · 상한 (R43, R-S4a-17): 표본 최댓값은 40 GiB 미만(같은 값도 거부), 이어하기 구간을 합한 시간은 24 h 이하(정확히 24 h는 허용). 측정 없음(0 바이트)은 측정된 0이 아니라 거부이다. 호출의 회계는 마지막 표지를 쓰기 직전의 스냅샷이며, 표지 자신의 쓰기와 호출 전체의 시간·최댓값은 과제 7이 밖에서 잰다.

입력과 출력

    train(cfg, data, out_dir, *, device, epochs, check, seed) → TrainResult      evaluate_stop(model, stop, scales, device) → float       fit_biases(rows) → (편향, 기록)


구현 경계: 체크포인트 IO는 checkpoints, 한 배치의 계산은 training_step, 복사 앞의 배치 준비(틀 적용)는 batch_prep, 자원 측정은 c_ept.runtime.resources에 위임한다. 에폭·조기 종료·재개·회계 조정은 이 모듈이 맡는다.

참조 경로 확인(2026-10-24): 위에서 인용한 역사적 설계서·S4a 계획·Task 5 계약서 중 해당 경로는 이 checkout에 미동봉되어 있다. 인용은 과거 설계 출처이며 이번 검토에서 그 원문까지 확인했다는 뜻이 아니다. docs/superpowers/notes/equation-index.md는 이 checkout에 존재하지만, 파일 존재만으로 모든 수식의 정확성을 인증하지 않는다."""
# Ref: docs/superpowers/plans/2026-10-22-c-ept-s4a.md decisions Q8 (the loss, equal rows in training, 1/n_m in evaluation), Q10 (early stopping on STOP, what the cost check measures) and Q3
#      (reported values on a deterministic CPU run); docs/superpowers/plans/2026-10-21-c-ept-s3.md P9; docs/superpowers/specs/2026-10-20-c-ept-design.md sections 7 and 8; the handoff
#      2026-10-23 section 5.2; controller rulings R14 (FIT only), R22, R43 (sampled private memory), R-S4a-10, R-S4a-16 to 20; the contract ``.superpowers/sdd/2026-10-22-c-ept-s4a/task-5-contract.md``.
# Ref: Loshchilov, I. and Hutter, F. (2019), "Decoupled weight decay regularization", ICLR; arXiv:1711.05101 (AdamW). Kingma, D. P. and Ba, J. (2015), "Adam: a method for
#      stochastic optimization", ICLR; arXiv:1412.6980. Loshchilov, I. and Hutter, F. (2017), "SGDR: stochastic gradient descent with warm restarts", ICLR; arXiv:1608.03983
#      (the cosine decay). Goyal, P. et al. (2017), "Accurate, large minibatch SGD: training ImageNet in 1 hour"; arXiv:1706.02677 (the linear warm-up).
# Ref: Krichevsky, R. E. and Trofimov, V. K. (1981), "The performance of universal encoding", IEEE Transactions on Information Theory 27(2): 199-207; doi:10.1109/TIT.1981.1056331
#      (bibliographic metadata only; the pseudocount 0.5 is an operational smoothing choice, neither mandated by that paper nor shown to be optimal).
from __future__ import annotations

import dataclasses
import functools
import hashlib
import io
import json
import math
import os
import random
import re
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

try:
    import psutil
except ImportError:      # the memory is then reported as unavailable (``_memory_sample``)
    psutil = None

from c_ept.preprocessing import terrain
from c_ept import runlog, runtime
from c_ept.data import head_rows
from c_ept.ept import data as ept_data
from c_ept.ept import encoder, heads, loss, checkpoints, training_step, epoch_metrics, batch_prep, step_guard
from c_ept.models import killlaw

# ---- the operational constants (every one has a name; the contract gives the reasons) ----------------------------------------------------
PATIENCE = 2                                     # epochs without a new best STOP loss that end the run
THREADS = 8                                      # the threads of the reported CPU runs
CATEGORY_PSEUDOCOUNT = 0.5                       # ζ₀: the pseudocount of a categorical start bias
RATE_FLOOR = 1e-6                                # λ_min: the lowest rate that a start bias may encode
MEMORY_CAP_BYTES = 40 * 2 ** 30                  # the sampled peak is strictly under this: the value itself is refused
WALL_CAP_SECONDS = 24 * 3600                     # the seconds of all calls of a run: under 'fail' exactly this is allowed and more is refused; under 'best_stop' the run ends at the end of the epoch that reaches it (TRN-11, `>=`)
MEMORY_INTERVAL_S = 0.1                          # the period of the sampling thread
BEAT_INTERVAL_S = 60.0                           # Δ_beat: the heartbeat period, and the tail that a call without an end is charged beyond its last heartbeat
SCHEMA = checkpoints.SCHEMA
COMPONENT_FIELDS = checkpoints.COMPONENT_FIELDS
CALLS_DIR = 'calls'                              # call_<n>.pending.json, call_<n>.beat and call_<n>.json of every call (new files only)
CALL_NAME = re.compile(r'call_(\d{4})(\.pending)?\.json')
METRICS = ('private_bytes_windows', 'rss_bytes_fallback')
ACCOUNTING_KEYS = ('seconds', 'peak_bytes', 'peak_upper_bytes', 'peak_rss_bytes', 'metric')
FAILURE_GLOB = 'resource_failure_*.json'
FILES = checkpoints.FILES                   # best.pt only in the folder of an epoch that improved; head.json is written after them
STATIC_FUNCTION = 'c_ept.ept.encoder.static_channels'
PRECONDITION = 'standalone resume holds for the same implementation, the same environment and float32; the code version is not in the identity (Task 7 runs clean code at an exact commit)'
MEMORY_SCOPE = 'this process and its children, sampled by a thread: a spike between two samples is not seen (a sample, not a proof of a continuous bound)'
LOSSES_NOTE = {'fit_batch_mean_losses': 'mean of batch losses of an epoch (training mode): a diagnostic, not an evaluation',
               'stop_losses': 'STOP evaluation, R-S4a-16 (evaluation mode): not comparable with a training loss'}
_OUTCOMES = {'completed': ('ok',), 'interrupted': ('ok',), 'hard_interrupted': ('ok', 'exceeded'), 'exceeded': ('exceeded',), 'unmeasured': ('unmeasured',)}
PENDING_KEYS = frozenset({'schema', 'kind', 'call', 'identity_sha256', 'epoch_done_start', 'head_sha256_start', 'start_accounting'})
CALL_KEYS = frozenset({'schema', 'kind', 'call', 'identity_sha256', 'pending_sha256', 'epoch_done_start', 'epoch_done_end', 'head_sha256_end', 'outcome', 'status', 'reason', 'evidence',
                       *ACCOUNTING_KEYS})
AXES = {'rate_net.out.bias': 'M2Rate: one scalar shared by all 18 bins and every row (the pooled rate of the observed cells)',
        'zone_net.out.bias': 'M2Zone: one 10-vector over the zones',
        'size_out.bias': 'M3Size: one 4-vector over the sizes; masked and open FIT rows pooled',
        'kill_out.bias': 'M3Kills: 8 outputs = size x team (index 2 s + t); masked and open FIT rows pooled',
        'gold_out.bias': 'M3Gold: one law over 31 bins shared by 48 slots (all sizes, teams and kill layers); masked and open FIT entries pooled',
        'obj_net.out.bias': 'M3Obj: 18 cells shared by all sizes; masked and open FIT rows pooled',
        'conv_net.out.bias': 'M4Conv: 18 cells; masked and open FIT rows with a conversion window pooled',
        'mark_net.out.bias': 'M2Mark: one scalar added to every mark score: it cancels in the softmax over the marks'}
SHIFTED_KILL_BIAS_NOTE = ('shifted kill law (decision D5): the start of each (size, team) slot is the log of the mean of the effective counts of that slot, the count minus 1 '
                          'where the team is the first-kill team of the row (the excess N_d) and the count where it is not; the slot is shared by both kinds of rows, and this mean is '
                          'the maximum-likelihood constant of the shared slot')
WALL_CAP_MODES = ('fail', 'best_stop')           # what a run does at the wall cap (S4b preparation M8, decision D10 P7): the earlier refusal, or the end with the best STOP checkpoint
DECAY_GROUPS = ('all', 'split')                  # the policies of the weight decay (S4b preparation M4, decision D6): one group of every parameter (all earlier runs), or the groups of ``param_groups``
_CLOCK = time.perf_counter


def _clock() -> float:
    """The clock of the time bound and of the heartbeat (a test replaces it)."""
    return _CLOCK()


# ---- the settings ----------------------------------------------------------------------------------------------------------------------
def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed % 2 ** 32)
    random.seed(seed)


def deterministic_cpu(threads: int = THREADS, seed: int | None = None) -> None:
    """Fix the CPU run: ``threads`` intra-op threads (default 8), ``torch.use_deterministic_algorithms(True)`` and, with a ``seed``, the torch, numpy and python streams. ValueError for
    ``threads`` that is not an integer of at least 1.

    해석: 같은 입력이 같은 비트를 내려면 합산 순서와 난수 흐름이 고정되어야 하므로 스레드 수, 결정적 알고리즘, 시드를 고정한다 (계획 Q3). 전역 설정을 바꾸므로 호출한 쪽이 되돌린다.
    """
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError(f'threads must be an integer of at least 1, not {threads!r}')
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    if seed is not None:
        _seed_all(int(seed))


def _deterministic_gpu(seed: int) -> None:
    """The GPU run (a cost measurement, not a reported value): deterministic algorithms and seeds. cuBLAS needs ``CUBLAS_WORKSPACE_CONFIG`` before its first call, so it is set here
    (in a process that already used CUDA the caller must have set it)."""
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    _seed_all(int(seed))


@dataclass(frozen=True)
class TrainConfig:
    """The settings of a run: the ``encoder`` size (``EncoderConfig``), the peak learning rate ``lr`` (not negative; 0 freezes the weights), the AdamW ``weight_decay``,
    the number of warm-up steps ``warmup_steps`` and the early-stopping ``patience`` (default 2). ValueError for a value outside those ranges.

    해석: 가중치 감쇠는 모든 매개변수에 같은 값으로 걸고(``decay_groups='all'``, 기본값) 기울기 자르기는 하지 않는다 (계획에 없는 선택이며 보고서에 적는다).

    ``kill_law`` (S4b 준비 M1, 저자 결정 D5): M3Kills의 킬 수 법칙. ``'poisson'``(기본값)은 이전 실행의 법칙이고 손실, 편향, 기울기, 체크포인트가 비트까지 같다. ``'shifted'``는 첫 킬 팀의 킬 수를
    1 + N_d로 두는 이동 법칙이다(손실 (LOSS-1k), 시작 편향 (TRN-5k)). 이동 법칙의 행은 ``ept_head_rows(..., kill_law='shifted')``로 만든 것이어야 하고(첫 킬 팀 d를 담는다), 이름은 ``killlaw.LAWS``에
    있어야 한다. 기본값이 아닐 때만 실행의 정체성(``_identity``)에 적으므로 정체성에 없으면 옛 법칙이다.

    ``k_condition`` (S4b 준비 M3, 저자 결정 D21 B16): K 조건 머리(가린·열린 오브젝트 머리와 전환 머리)가 학습에서 보는 K 블록의 골드. ``'raw'``(기본값)는 라벨의 실제 연속 킬 골드로 이전 실행의
    행, 손실, 기울기, 체크포인트가 비트까지 같다. ``'binned'``는 관측한 골드 구간의 대표값(킬이 없는 팀은 0)으로, 합성기가 K 조건 머리에 넣는 값과 같은 지지에서 학습한다. 'binned' 행은
    ``ept_head_rows(..., k_condition='binned')``로 만든 것이어야 하고(``_check_inputs``가 양쪽 방향으로 거부한다), 이름은 ``head_rows.K_CONDITIONS``에 있어야 한다. 기본값이 아닐 때만 실행의
    정체성(``_identity``)에 적으므로 정체성에 없으면 'raw'이다. 'binned'이면 정체성에 대표값 31개의 ``k_reps_sha256``(float64 little-endian 바이트의 sha256, ``value.gold_reps_sha256``)도 적고,
    FIT와 STOP 행의 대표값이 다르면 일을 시작하기 전에 거부한다.

    ``decay_groups`` (S4b 준비 M4, 저자 결정 D6, 검토 A2-F2): AdamW 가중치 감쇠의 묶음 정책. ``'all'``(기본값)은 이전 실행의 호출 하나(모든 매개변수, 감쇠 ``weight_decay``)이고 최적화기 상태, 체크포인트,
    정체성, 에폭 지표가 비트까지 같다. ``'split'``은 log κ, 모든 편향, LayerNorm, Embedding에는 감쇠를 걸지 않고 Linear 가중 행렬, 공동 참여 가중 w·w_back, 학습 지형장에는 ``weight_decay``를 건다
    (``param_groups``). ``'split'``이면 에폭 지표에 감쇠 척도 열 kappa_r이 더해지고 머리 스키마는 v7이다 (``epoch_metrics``). 이름은 ``DECAY_GROUPS``에 있어야 한다. 기본값이 아닐 때만 실행의 정체성
    (``_identity``)에 정책과 묶음 규칙의 해시를 적으므로 정체성에 없으면 'all'이다. 정책이 다른 체크포인트에서 이어하는 것은 양쪽 방향 모두 거부한다(최적화기 상태가 묶음 안의 위치로 번호가 매겨지기 때문).

    ``column_drop``과 ``stats_population`` (S4b 준비 M5, 저자 결정 D7·D12): 입력 통계(``data.Stats``)가 어떻게 만들어졌어야 하는지에 대한 실행의 설정이다. ``column_drop``은 'none'(기본값; 뺀 열
    없음, 이전 실행), 'store'(전체 FIT 저장소에서 상수인 열, D7 목록), 'store_and_subset'(D7 목록과 그 실행의 선택된 FIT에서만 상수인 열)이고, ``stats_population``은 'whole_fit'(기본값)
    또는 'selected_fit'(그 실행이 고른 FIT 경기)이다. 통계가 다른 설정으로 만들어졌으면 일을 시작하기 전에 거부한다(양쪽 방향; ``_check_inputs``). 기본값이 아닐 때만 정체성에 적으므로
    (뺀 열의 목록은 해시로) 정체성에 없으면 기본값이다. 열을 빼는 일 자체(입력 열을 0으로 고정)는 ``data.ViewTensors``가 통계의 목록으로 한다. 입력 폭은 그대로여서 모형과 체크포인트의 모양이 같다.

    ``terrain_smooth_weight`` (S4b 준비 M6, 저자 결정 D8, 검토 A2-F5): 지형 매끄러움 가중 λ_Ω (유한한 0 이상의 수). 학습 손실은 L = Loss_EPT + λ_Ω · R_Ω이고 R_Ω는 학습 지형장의 이웃 칸 차이 제곱 평균(ENC-7,
    ``encoder.TerrainField.penalty``)이다(``training_step.optimize_batch``). 기본값 0은 이전 실행이며 벌점을 계산하지 않아 손실, 기울기, 최적화기 상태, 체크포인트, 에폭 지표가 비트까지 같다. 0이 아니면 에폭 지표에
    열 ``terrain_penalty``(걸음별 R_Ω의 에폭 평균, 가중 없음)가 더해지고 머리 스키마는 v8이다 (``epoch_metrics``). STOP 값(조기 종료와 최선 체크포인트)에는 벌점이 들어가지 않는다. 기본값이 아닐 때만 실행의
    정체성에 적으므로 정체성에 없으면 0이고, 가중이 다른 체크포인트에서 이어하는 것은 양쪽 방향 모두 거부한다. 학습 채널이 하나도 없는 모형(``encoder.field_channels`` 0)에 0이 아닌 가중을 거는 것은 설정 실수여서
    일을 시작하기 전에 거부한다.

    ``swap_mode`` (S4b 준비 M7, 저자 결정 D10 P1·D17, 계획 Q3, 검토 A2-F4): 학습 배치를 어느 틀에서 쓰는가. ``'average'``(기본값)는 이전 실행이다: 모든 보기를 기록된 틀과 맞바꾼 틀에서 부호화해 평균한다(``heads.FrameAveraged``, 보기마다
    부호화기 두 번). ``'random_view'``는 보기마다 확률 ½로 맞바꾼 한 틀만 부호화한다(부호화기 한 번; 맞바꾼 보기를 읽는 행은 조건과 정답이 함께 맞바뀐다; 동전은 ``Batches``가 시드·에폭·배치 번호에서 뽑는다). ``'none'``은 교환 없이
    기록된 틀 하나만 쓴다(D17의 대칭 없음 점검). 배치는 이 설정으로 만든 것이어야 한다(``Batches(..., swap_mode=cfg.swap_mode)``; ``_check_inputs``가 양쪽 방향으로 거부한다). 기본값이 아닐 때만 실행의 정체성(``_identity``)에 적으므로
    정체성에 없으면 ``'average'``이고, 설정이 다른 체크포인트에서 이어하는 것은 양쪽 방향 모두 거부한다.

    ``eval_average`` (같은 과제): 평가(조기 종료의 STOP 값과 이후의 평가)가 두 틀의 평균을 쓰는가. True(기본값)는 이전 실행이고 모든 ``swap_mode``에서 정확한 팀 교환 불변을 낸다(``heads.FrameAveraged``의 평균). False는 기록된 틀 하나만
    평가한다(D17의 점검: ``swap_mode='none', eval_average=False``). STOP 값은 이 평가 경로로 잰다(``evaluate_stop``). 기본값이 아닐 때만 정체성에 적는다(없으면 True). 평가 쪽은 모형 객체가 정체성을 갖지 않으므로 이 값을
    호출하는 쪽(K6)이 정체성에서 읽어 ``adapters.run_inference(..., eval_average=...)``로 넘긴다.

    ``on_wall_cap`` (S4b 준비 M8, 저자 결정 D10 P2·P7): 모든 호출의 시간 합이 24 h 상한(``WALL_CAP_SECONDS``)에 닿았을 때의 동작. ``'fail'``(기본값)은 이전 실행이다: 상한을 넘기면 ``ResourceBoundError``이고 실패 기록이
    남으며 그 실행은 끝난다. ``'best_stop'``은 상한에 닿은 에폭을 끝까지 마치고(체크포인트의 단위가 에폭이다) ``done=True``로 끝내며 최선 STOP 체크포인트를 그대로 두고 머리에 ``budget_end``를 적는다
    (에폭 지표의 ``termination_reason``은 ``'budget'``, 조기 종료나 일정 끝이 같은 에폭에 닿았으면 그 이유). 메모리 상한 위반, 측정 불가, 정체성·무결성 위반은 두 설정 모두 실패다. 에폭 예산은 이미 있는 ``epochs``
    인자이고 코사인 일정은 ``epochs × len(fit)`` 걸음에서 끝난다(프로파일이 예산을 정한다). 기본값이 아닐 때만 정체성에 적으므로 정체성에 없으면 ``'fail'``이고, 설정이 다른 체크포인트에서 이어하는 것은 양쪽 방향
    모두 거부한다. 판정(리드): `'budget'`은 상한이 *끝난 에폭의 경계*에서 닿았다는 뜻이다. `'best_stop'` 실행이 상한을 넘긴 뒤 그 에폭의 체크포인트를 쓰기 전에 죽거나 중단되면(예외, 하드 중단) 예산 종료가 아니라 저장된 실패로
    끝난다(닫힌 채 실패): 다음 호출은 24 h `ResourceBoundError`로 거부되고 일은 하지 않으며, 앞 에폭들의 최선 STOP 체크포인트는 디스크에 그대로 남는다. 동작 변경은 없고 이 문장이 그 규칙을 적는다.

    ``zone_mark`` (S4b 준비 M9, 저자 결정 D9): 구역 표지(첫 킬의 규칙 구역 z)를 모형의 한 부분으로 두는가. True(기본값)는 이전 실행이다. False는 A7 설계 점검 "구역 표지 없음"이다: M2Zone 블록을 학습하지
    않고(손실에도 STOP 값에도 들지 않는다: ``score_blocks``), M3·오브젝트·M4의 z 조건 입력 Γ_z를 0으로 두며(``heads.EPTHeads``), 합성기는 z를 뽑지 않는다(``compose.Composer(zone_mark=False)``).
    구역 표(zone_emb)와 M2Zone 머리는 모듈에 남아 기울기를 받지 않으므로 값이 시작 값 그대로이다. 기본값이 아닐 때만 실행의 정체성(``_identity``)에 False로 적으므로 정체성에 없으면 True이고, 설정이 다른
    체크포인트에서 이어하는 것은 양쪽 방향 모두 거부한다. 에폭 지표의 열과 머리 스키마는 바뀌지 않는다(새 열 없음). D9의 비교에 쓰는 공통 M2Rate·M2Mark 블록 값은 끝난 모형에 ``evaluate_stop_blocks``를
    불러 얻는다. 평가 쪽은 모형 객체가 정체성을 갖지 않으므로(모형은 자기 설정 ``heads.zone_mark``만 안다) 이 값을 호출하는 쪽(K6)이 정체성에서 읽어 ``adapters.run_inference(..., zone_mark=...)``와
    ``Composer(..., zone_mark=...)``로 넘기고, 모형이 선언한 값과 다르면 두 곳이 거부한다. N2 모형은 구역 표지를 끌 수 없다(``model_spec``과 함께 False이면 일을 시작하기 전에 거부한다).
    """
    encoder: encoder.EncoderConfig
    lr: float
    weight_decay: float
    warmup_steps: int
    patience: int = PATIENCE
    model_spec: Mapping | None = None
    kill_law: str = 'poisson'
    k_condition: str = 'raw'
    decay_groups: str = 'all'
    column_drop: str = 'none'
    stats_population: str = 'whole_fit'
    terrain_smooth_weight: float = 0.0
    swap_mode: str = 'average'
    eval_average: bool = True
    on_wall_cap: str = 'fail'
    zone_mark: bool = True

    def __post_init__(self):
        killlaw.check_law(self.kill_law)
        if not isinstance(self.zone_mark, bool):
            raise ValueError(f'zone_mark must be True (the zone mark is part of the model) or False (A7: no zone mark), not {self.zone_mark!r}')
        head_rows.check_k_condition(self.k_condition)
        check_decay_groups(self.decay_groups)
        ept_data.check_swap_mode(self.swap_mode)
        if not isinstance(self.eval_average, bool):
            raise ValueError(f'eval_average must be True (the two-frame average) or False (the recorded frame only), not {self.eval_average!r}')
        if not isinstance(self.on_wall_cap, str) or self.on_wall_cap not in WALL_CAP_MODES:
            raise ValueError(f'on_wall_cap must be one of {WALL_CAP_MODES} (stop with an error at the wall cap, or end the run with its best STOP checkpoint), not {self.on_wall_cap!r}')
        if not isinstance(self.column_drop, str) or self.column_drop not in ept_data.COLUMN_DROPS:
            raise ValueError(f'column_drop must be one of {ept_data.COLUMN_DROPS}, not {self.column_drop!r}')
        if not isinstance(self.stats_population, str) or self.stats_population not in ept_data.POPULATIONS:
            raise ValueError(f'stats_population must be one of {ept_data.POPULATIONS}, not {self.stats_population!r}')
        if self.model_spec is not None:
            from c_ept.comparison.models import validate_model_spec
            object.__setattr__(self, 'model_spec', validate_model_spec(self.model_spec))
        if not isinstance(self.encoder, encoder.EncoderConfig):
            raise ValueError('encoder must be an EncoderConfig')
        for name in ('lr', 'weight_decay', 'terrain_smooth_weight'):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                raise ValueError(f'{name} must be a finite number that is not negative, got {v!r}')
        for name, low in (('warmup_steps', 0), ('patience', 1)):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < low:
                raise ValueError(f'{name} must be a whole number of at least {low}, got {v!r}')


@dataclass(frozen=True)
class TrainData:
    """The data of a run: ``fit`` (``Batches``, role FIT, ``training=True``), ``stop`` (``Batches``, role STOP, ``training=False``), ``scales`` (the 13 B0 scales of
    ``loss.BLOCKS``) and ``static`` (``encoder.static_channels(Terrain)``), with ``static_source`` (a JSON-able mapping the caller wants recorded, for example the hash of the
    terrain manifest; None). There is no slot for any other role. ``train`` checks all of it before any work."""
    fit: object
    stop: object
    scales: Mapping
    static: np.ndarray
    static_source: Mapping | None = None


@dataclass
class TrainResult:
    """What a run returns. ``model``: the EPT with the weights of the best STOP epoch (evaluation mode). ``epochs_run``, ``best_epoch`` (0-based), ``stopped_early``;
    ``fit_batch_mean_losses`` (the mean of the batch losses of an epoch in training mode: a diagnostic), ``stop_losses`` (the STOP evaluation, one for each epoch run), ``lrs``
    (one for each optimizer step). Per epoch: ``epoch_seconds`` and its parts ``load_seconds`` (inside ``next()`` of the FIT batches; with the kernel's prefetch this is the wait for the producer, not the assembly), ``step_seconds`` (transfer, forward, backward, optimizer
    step; ``transfer_seconds`` of a run on plain batches includes the CPU preparation of ``batch_prep.Preparation``, S4b speed-ups, D32) over ``steps`` steps, and ``stop_seconds``. v5 ``timing_components`` records persisted forward/loss, backward, optimizer and transfer spans, plus current-call checkpoint IO only (old-attempt IO unavailable). ``seconds`` and the memory (``memory_metric``, the sampled ``peak_memory_bytes``, the Windows lifetime upper bound
    ``peak_upper_bytes`` or None, ``peak_rss_bytes``) are the accounting of all calls of the run, up to the cutoff of ``resource``, the final call marker of this call (the last
    one of a run that was already finished, which this call does not change). ``checkpoint_bytes``: the last.pt of the newest checkpoint plus the best.pt it names; ``checkpoint_dir``
    and ``head``: that checkpoint and its head.json."""
    model: torch.nn.Module
    epochs_run: int
    best_epoch: int
    stopped_early: bool
    fit_batch_mean_losses: list
    stop_losses: list
    lrs: list
    seconds: float
    epoch_seconds: list
    load_seconds: list
    step_seconds: list
    steps: list
    stop_seconds: list
    memory_metric: str
    peak_memory_bytes: int
    peak_upper_bytes: int | None
    peak_rss_bytes: int
    checkpoint_bytes: int
    checkpoint_dir: Path
    head: dict = field(default_factory=dict)
    resource: dict = field(default_factory=dict)
    timing_components: dict = field(default_factory=dict)


# ---- the parameter groups of the weight decay (S4b preparation M4, decision D6, review finding A2-F2) ------------------------------------------
# Ref: Loshchilov, I. and Hutter, F. (2019), "Decoupled weight decay regularization", ICLR; arXiv:1711.05101: the decay is a term of the update that is separate from the gradient step
#      (TRN-7), so it can be set for each group of parameters; which parameters decay is the author's decision D6 (docs: harness/decisions.md), not a rule of that paper.
_DECAY_CLASSES = {'*': object, 'LayerNorm': torch.nn.LayerNorm, 'Embedding': torch.nn.Embedding, 'Linear': torch.nn.Linear,
                  'Coparticipation': encoder.Coparticipation, 'TerrainField': encoder.TerrainField}
# the rule table of decision D6, first match wins: (class of the module that owns the parameter, name of the parameter itself, kind of its decay); the name is the last part of the
# qualified name ('bias' of 'encoder.blocks.0.qkv.bias'), not a part of its path: the distance and geodesic networks sit in the attribute ``encoder.bias`` and their weight matrices decay
DECAY_RULES = (('LayerNorm', '*', 'zero'), ('Embedding', '*', 'zero'),
               ('*', 'log_kappa', 'zero'), ('*', 'type_bias', 'zero'), ('*', 'bias', 'zero'),
               ('Linear', 'weight', 'weight_decay'), ('Coparticipation', 'w', 'weight_decay'), ('Coparticipation', 'w_back', 'weight_decay'), ('TerrainField', 'learned', 'weight_decay'))


def check_decay_groups(name):
    """``name`` if it is one of ``DECAY_GROUPS``; ValueError naming ``decay_groups`` otherwise (before anything is built)."""
    if not isinstance(name, str) or name not in DECAY_GROUPS:
        raise ValueError(f'decay_groups must be one of {DECAY_GROUPS}, not {name!r}')
    return name


def _decay_kinds(model: torch.nn.Module) -> dict:
    """``{qualified name: 'zero' | 'weight_decay'}`` of every parameter of ``model`` in the order of ``model.named_parameters()``, from ``DECAY_RULES``; ValueError naming a parameter that
    no rule matches (the names are code, not data). A parameter that several modules share is one entry under its first name."""
    kinds = {}
    for name, _parameter in model.named_parameters():
        owner, _, leaf = name.rpartition('.')
        module = model.get_submodule(owner)
        for module_class, parameter_name, kind in DECAY_RULES:
            if parameter_name in ('*', leaf) and isinstance(module, _DECAY_CLASSES[module_class]):
                kinds[name] = kind
                break
        else:
            raise ValueError(f'the parameter {name!r} ({type(module).__name__}.{leaf}) matches no rule of DECAY_RULES: add its kind to the table (decision D6)')
    return kinds


def decay_assignment(model: torch.nn.Module, weight_decay: float) -> dict:
    """``{qualified name: decay}`` that the policy ``'split'`` gives to every parameter of ``model``: 0.0 for the kind 'zero', ``weight_decay`` for the kind 'weight_decay'.

    해석: 이름 하나에 감쇠 하나이고 모든 매개변수가 한 번씩 나온다(``named_parameters``). 묶음 해시와 헤드 기록이 이 사상에서 나온다."""
    return {name: 0.0 if kind == 'zero' else weight_decay for name, kind in _decay_kinds(model).items()}


def param_groups(model: torch.nn.Module, weight_decay: float, policy: str) -> list:
    """The ``params`` groups of the optimizer for the policy ``'all'`` (one group of every parameter with ``weight_decay``, the groups of every earlier run) or ``'split'``
    (two groups: the weight matrices, ``w``, ``w_back`` and the learned terrain field with ``weight_decay``, then log κ, every bias, LayerNorm and Embedding with 0.0). Every
    parameter of ``model`` lands in exactly one group. ValueError for a policy that is not in ``DECAY_GROUPS`` and, under 'split', for a parameter that no rule of ``DECAY_RULES``
    matches (named in the message).

    수식
    ----
    (TRN-8)  θ_i ← θ_i (1 − η_t ϑ_i) − η_t · A₁ᵢ / (√A₂ᵢ + ε_A),   ϑ_i = ϑ (모든 i; 'all'),   ϑ_i = 0 (i ∈ N),  ϑ_i = ϑ (i ∈ D)  ('split')
             N = {log κ, 편향, type_bias, LayerNorm의 이득·편향, Embedding 표},   D = {Linear 가중 행렬, w, w_back, Ω_learn}

      기호    | 뜻                                              | 단위 | 범위     | 코드 변수
      ϑ_i     | 매개변수 i의 가중치 감쇠                        | —    | ≥ 0      | 묶음의 weight_decay
      ϑ       | 설정한 가중치 감쇠 (0.01)                       | —    | ≥ 0      | weight_decay
      N, D    | 감쇠 없는 묶음, 감쇠 있는 묶음                  | —    | 서로소   | 반환값의 두 묶음 (D 먼저)
      η_t     | 걸음 t의 학습률 (TRN-1)                         | —    | ≥ 0      | 묶음의 lr (걸음마다 모든 묶음에 쓴다)

    해석
    ----
    감쇠는 매개변수를 0 쪽으로 당기는 사전 분포(L2)의 구실을 한다. 가중 행렬에는 그 구실이 알맞지만 log κ에서는 초 단위 척도를 1 s 쪽으로 당기는 단위에 매인 사전 분포가 되고(걸음 4,511개의
    에폭 18번이면 log κ에 0.543을 곱해 80 s가 10.8 s가 된다), FIT 한계 분포로 시작한 출력 편향은 시작점에서 0 쪽으로 끌려가며, LayerNorm의 이득·편향과 Embedding 표는 척도 자체를 나른다
    (결정 D6, 검토 A2-F2). 학습 지형장은 0을 중심으로 한 L2 사전 분포로 감쇠를 유지한다(매끄러움은 따로 정한다). 규칙은 매개변수 이름의 마지막 조각과 그것을 가진 모듈의 종류로 읽는다: 차원 수로는
    가를 수 없는 것(type_bias, w, field.learned)이 있고, 이름 전체에 'bias'가 든다고 감쇠를 끄면 부호화기의 ``bias`` 속성 아래 있는 거리·측지 망의 가중 행렬까지 꺼 버린다.
    경계 사례: 가중치 감쇠가 0이어도 정책 'split'은 두 묶음을 낸다(둘 다 0.0). 규칙이 없는 새 매개변수는 조용히 감쇠되지 않고 오류로 멈춘다.

    근거
    ----
    저자 결정 D6(2026-10-25, B10), 검토 A2-F2, Ref: Loshchilov and Hutter 2019 (분리된 가중치 감쇠).
    """
    check_decay_groups(policy)
    if policy == 'all':
        return [{'params': list(model.parameters()), 'weight_decay': weight_decay}]
    named, kinds = dict(model.named_parameters()), _decay_kinds(model)
    groups = [{'params': [named[name] for name, kind in kinds.items() if kind == 'weight_decay'], 'weight_decay': weight_decay},
              {'params': [named[name] for name, kind in kinds.items() if kind == 'zero'], 'weight_decay': 0.0}]
    return [group for group in groups if group['params']]


def decay_rule_sha256() -> str:
    """The SHA-256 of the rule table that ``'split'`` applies (``DECAY_RULES``): the run identity holds it, so a change of the rule is another run."""
    return _sha(json.dumps({'policy': 'split', 'rules': [list(rule) for rule in DECAY_RULES]}, sort_keys=True).encode())


def _optimizer(model: torch.nn.Module, cfg: TrainConfig) -> torch.optim.AdamW:
    """The AdamW of a run. The policy 'all' is the call of every earlier run, written as it was (the optimizer state and its bytes are those of the code before this task)."""
    if cfg.decay_groups == 'all':
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    return torch.optim.AdamW(param_groups(model, cfg.weight_decay, cfg.decay_groups), lr=cfg.lr, weight_decay=cfg.weight_decay)


def _decay_record(model: torch.nn.Module, cfg: TrainConfig) -> dict | None:
    """What the head records about the groups of a run under a policy other than 'all' (None under 'all'): the policy, the hash of the rule table, the SHA-256 of the sorted
    (name, decay) list that the rule gave to the parameters of this model, and the size of each group. A resume compares it with the head of the checkpoint."""
    if cfg.decay_groups == 'all':
        return None
    assignment = decay_assignment(model, cfg.weight_decay)
    groups = param_groups(model, cfg.weight_decay, cfg.decay_groups)
    return json.loads(json.dumps({'policy': cfg.decay_groups, 'rule_sha256': decay_rule_sha256(),
                                  'assignment_sha256': _sha(json.dumps([[name, decay] for name, decay in sorted(assignment.items())]).encode()),
                                  'groups': [{'weight_decay': g['weight_decay'], 'tensors': len(g['params']), 'elements': sum(p.numel() for p in g['params'])} for g in groups]}))


def _kappa_record(model: torch.nn.Module):
    """The decay scales κ_r = exp(log κ_r) of the co-participation bias of ``model`` as a list of floats (seconds), or None for a model without them (N2).

    수식
    ----
    (TRN-9)  κ_r = exp(log κ_r),  r = 0 .. K − 1

      기호    | 뜻                                | 단위 | 범위     | 코드 변수
      κ_r     | r번째 감쇠 척도                   | s    | > 0      | Coparticipation.kappa
      log κ_r | 학습하는 매개변수 (감쇠 없음, D6) | 로그 s | 실수   | Coparticipation.log_kappa

    해석: 결정 D6이 감쇠를 끄는 대신 요구한 기록이다. 척도가 에폭마다 어디로 가는지를 초 단위로 적어 두면 감쇠가 없을 때 척도가 시작 구간(5–80 s)을 벗어나는지 지표만으로 읽을 수 있다.
    float32 값을 그대로 float로 넓히므로 같은 모형은 같은 수를 낸다."""
    scales = [module for module in model.modules() if isinstance(module, encoder.Coparticipation)]
    if not scales:
        return None
    if len(scales) != 1:
        raise ValueError(f'the model has {len(scales)} co-participation modules; the record of kappa expects one')
    return [float(x) for x in scales[0].kappa.detach().cpu().tolist()]


# ---- the learning rate -----------------------------------------------------------------------------------------------------------------
def lr_at(step: int, total: int, warmup: int, peak: float) -> float:
    """The learning rate of the optimizer step ``step`` (0-based) of a run of ``total`` steps: a linear rise over ``warmup`` steps to ``peak``, then a cosine decay.

    수식
    ----
    (TRN-1)  η_t = η_max (t + 1) / n_w    (t < n_w),    η_t = η_max · ½ [1 + cos(π (t − n_w) / max(1, n_T − n_w))]    (t ≥ n_w)

      기호   | 뜻                                    | 단위 | 범위   | 코드 변수
      η_t    | 걸음 t의 학습률                       | —    | ≥ 0    | 반환값
      η_max  | 최대 학습률                           | —    | ≥ 0    | peak
      t      | 최적화 걸음 번호 (0부터)              | 걸음 | ≥ 0    | step
      n_w, n_T | 예열 걸음 수, 전체 걸음 수 (에폭 수 × 에폭당 배치 수) | 걸음 | ≥ 0, ≥ 1 | warmup, total

    해석
    ----
    예열은 첫 갱신에서 AdamW의 2차 모멘트가 아직 믿을 만하지 않을 때 큰 걸음을 막고, 코사인 감소는 끝으로 갈수록 걸음을 줄인다.
    값은 걸음 번호와 전체 걸음 수만의 함수이므로 끊었다 이어도 같은 값을 낸다.
    마지막으로 쓰는 걸음은 n_T − 1이고 식이 n_T − n_w로 나누므로 그 값은 0이 아니다. 전체 걸음이 예열보다 짧아도 분모는 1 아래로 내려가지 않는다.

    근거
    ----
    과제 5 작업서 (AdamW, 예열·코사인), Ref: Loshchilov and Hutter 2017 (코사인), Goyal et al. 2017 (선형 예열).
    """
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return peak * 0.5 * (1.0 + math.cos(math.pi * progress))


# ---- the blocks of a batch -------------------------------------------------------------------------------------------------------------
def score_blocks(fa, enc, batch, kill_law: str = 'poisson', zone_mark: bool = True) -> dict:
    """배치 내 행이 존재하는 블록들에 대해 각 머리 출력과 음의 로그우도(NLL)를 계산하여 {block_key: (head, target, nll)} 사전을 반환한다.

    매개변수:
    - fa (FrameAveraged): 프레임 평균화된 머리 래퍼.
    - enc: 배치 전체에 대해 1회 생성되어 공유되는 부호화 상태.
    - batch: 학습 또는 검증 데이터 배치.
    - kill_law: M3Kills의 킬 수 법칙 (결정 D5). 'poisson'(기본값)은 이전 호출과 같고, 'shifted'는 ``loss.head_nll``의 이동 법칙 (LOSS-1k)으로 M3Kills 블록을 매긴다. 기본값일 때는
      ``loss.head_nll``을 이전과 같은 인자로 부른다.
    - zone_mark: 구역 표지를 학습하는가 (결정 D9, S4b 준비 M9). True(기본값)는 이전 호출과 같다. False이면 M2Zone/masked 블록을 매기지 않는다: 반환값에 없으므로 손실(LOSS-4)과 STOP 값(TRN-2)에 들지 않고,
      공유 M2 호출은 M2Mark가 읽는 보기만 부른다. 나머지 열두 블록은 그대로이다.

    수식
    ----
    (LOSS-4z)  Loss_EPT = Σ_{h ∈ H} ℓ̄_h / ℓ̄^B0_h ,   H = 열세 블록 (zone_mark=True) ;   H = 열세 블록 − {M2Zone/masked} (zone_mark=False)

      기호          | 뜻                                              | 단위 | 범위    | 코드 변수
      H             | 점수를 매기는 블록의 집합                       | —    | —       | 반환 사전의 키
      ℓ̄_h, ℓ̄^B0_h   | 블록 h의 평균 점수, 같은 블록의 B0 평균 NLL    | nat  | ≥ 0, > 0 | loss.total_loss의 항

    해석: M2Zone의 점수 −Σ log ρ(z | q, m)는 52표지 모형에 없는 항이어서 전체 모형의 M2 NLL에 구역 항이 실려 있다. 구역 표지를 뺀 설계는 그 항을 목적에서 빼고, 남은 열두 블록의 합을 최소화한다.

    반환값:
    - out (dict): 유효한 블록별 머리 이름, 대상 데이터 딕셔너리, loss.head_nll 결과 튜플의 매핑.

    동작 및 최적화:
    - M2Mark와 M2Zone은 유효한 마스킹 행들의 고유 뷰(torch.unique) 합집합에 대해 fa.m2를 단 1회 호출하여 계산 결과를 공유한다 (R-S4a-20).
    - M2Rate 블록은 callable인 fa.m2_rate가 있으면 학습/평가 모두 그 API를 호출하며, 이 API는 두 모드 모두 표지·구역 분기를 건너뛴 rate 전용 경로를 쓴다 (학습 모드에서도 값과 기울기는 전체 m2의 rate와 비트 단위로 같다; EPT-T011 A2-F10; CPU에서 시험했으며 GPU는 시험하지 않았다). API가 없거나 callable이 아니면 fa.m2(...).rate로 호환한다. API 호출 중 예외를 숨기고 fallback하지 않는다.
    - 행이 없는 빈 블록(len(y) == 0)은 연산에서 제외된다.
    - 한 틀의 부호화(``heads.FrameAveraged(model, average=False)``, S4b 준비 M7)가 동전대로 맞바꾼 행을 ``enc.rows``에 담아 주면 블록의 행과 정답은 배치가 아니라 그것에서 읽는다: 맞바꾼 보기를 읽는 행은
      조건과 정답이 그 보기의 틀에 있다. 담아 주지 않으면(평균, 동전 없는 한 틀, 목 부호화) 배치 그대로이고 호출은 이전과 같다."""
    framed = getattr(enc, 'rows', None)
    batch = batch if framed is None else framed
    blocks = {key: ept_data.block(batch, *key) for key in ept_data.ROW_FIELDS}
    trusted = step_guard.trusted()
    if trusted:
        # (SPD-3), S4b speed-ups item 3, D32: the index lists that ``batch_prep.Preparation`` made on the CPU go with the blocks (``loss.head_losses`` averages ``rows.index_select(0, list)``)
        for key, b in blocks.items():
            for name in loss.PLAN_NAMES:
                plan = loss.plan_key(*key, name)
                if plan in batch:
                    b[plan] = batch[plan]
    law = {} if kill_law == 'poisson' else {'kill_law': kill_law}                      # the plain law is not passed: the call is that of every earlier run
    unscored = frozenset() if zone_mark else frozenset({'M2Zone'})                     # (LOSS-4z): the heads that are not scored (none by default)
    pos = {h: blocks[(h, 'masked')]['pos'] for h in ('M2Mark', 'M2Zone') if h not in unscored and len(blocks[(h, 'masked')]['y'])}
    shared = {}
    if pos:
        listed = batch_prep.m2_key(zone_mark, 'views') in batch if trusted else False
        if listed:                                                                     # (SPD-4): the union of the views, made once on the CPU (the setting is in the key)
            views, inverse = batch[batch_prep.m2_key(zone_mark, 'views')], batch[batch_prep.m2_key(zone_mark, 'inverse')]
        else:
            views, inverse = torch.unique(torch.cat(list(pos.values())), return_inverse=True)
        m2, start = fa.m2(enc, 1.0, pos=views), 0
        for h, p in pos.items():
            shared[h] = (m2.mark_logp if h == 'M2Mark' else m2.zone_logp).index_select(0, inverse[start:start + len(p)])
            start += len(p)
    out = {}
    for head, kind in ept_data.ROW_FIELDS:
        b = blocks[(head, kind)]
        if len(b['y']) == 0 or head in unscored:
            continue
        q, m, z = b.get('q'), b.get('m'), b.get('z')
        if head in shared:
            value = shared[head]
        elif head == 'M2Rate':
            # rate 전용 경로(표지·구역 분기 생략)의 선택은 FrameAveraged 소유이며 학습·평가에 공통이다. m2만 가진 기존 mock도 받는다 (TRN-7, EH-1).
            rate = getattr(fa, 'm2_rate', None)
            value = rate(enc, b['e'], pos=b['pos']) if callable(rate) else fa.m2(enc, b['e'], pos=b['pos']).rate
        elif head in ('M3Size', 'M3Kills', 'M3Gold'):
            m3 = fa.m3(enc, q, m, z, pos=b['pos'])
            value = {'M3Size': m3.size_logp, 'M3Kills': m3.kill_rate, 'M3Gold': m3.gold_logp}[head]
        elif head == 'M3Obj':
            value = fa.m3_obj(enc, q, m, z, b['s'], b['K'], pos=b['pos'])
        else:
            value = fa.m4(enc, q, m, z, b['K'], b['O'], b['e'], pos=b['pos'])
        out[loss.block_key(head, kind)] = (head, b, loss.head_nll(head, value, b, **law))
    return out


def _to(batch: Mapping, device: torch.device):
    return batch if device.type == 'cpu' else {k: v.to(device) for k, v in batch.items()}


# ---- the STOP number -------------------------------------------------------------------------------------------------------------------
class StopMeter:
    """The pool of the STOP evaluation over any number of batches: ``add(block, nll, scored, match, row=None, size=None)`` for each block of each batch, ``value(scales)``
    at the end. ``block`` is a key of ``loss.BLOCKS``; ``nll`` (n,) float scores of the design entries, ``scored`` (n,) bool (``loss.scored_rows``), ``match`` (n,) integer
    match numbers, ``row`` (n,) the id of the evaluation row of each entry inside its match (None: every entry is its own row), ``size`` (n,) the size 0 .. 3 for the three
    size-normalised experts and only for them. Only the sum and the number of the scored entries of an evaluation row are kept. ValueError for a block that does not exist,
    arrays of another length or type, a size given or missing for the wrong block or outside 0 .. 3, and two sizes inside one evaluation row.

    수식
    ----
    (TRN-2)  Ω_{h,u,j} = (1 / |I⁺|) Σ_{i ∈ I⁺} ℓ_i                  I⁺ = 블록 h, 경기 u, 평가 행 j의 점수 있는 항목 (등록부의 I에서 점수 있는 것)
             n_{h,s,u} = 층 s 안에서 점수 있는 항목이 있는 경기 u의 평가 행 수      (묶음·점수 뒤에, 모든 배치에 걸쳐 센다)
             L_{h,s} = (1 / M_{h,s}) Σ_u [Σ_j Ω_{h,u,j}] / n_{h,s,u}                 (층 안의 경기 가중 평균)
             L_h = 나온 층 s의 L_{h,s}의 평균 ;   Loss_STOP = Σ_h L_h / ℓ̄^B0_h

      기호        | 뜻                                              | 단위 | 범위      | 코드 변수
      ℓ_i         | 설계 항목 i의 점수 (``loss.head_nll``)          | nat  | ≥ 0       | nll[i]
      Ω_{h,u,j}   | 평가 행의 값: 점수 있는 항목의 평균 (B0 척도와 같은 항목 단위) | nat | ≥ 0 | 합 / 개수
      h, s        | 블록, 규모 층 (전문가 셋만; 그 밖은 층 하나)    | —    | —         | block, size
      u, j        | 경기, 그 경기 안의 평가 행                      | —    | —         | match, row
      n_{h,s,u}   | 층 안에서 경기 u의 평가 행 수                   | 행   | ≥ 1       | 묶음 수
      M_{h,s}     | 층 s에 평가 행이 있는 경기 수                   | 경기 | ≥ 1       | len(strata[s])
      L_{h,s}, L_h | 층 평균, 블록 평균                             | nat  | ≥ 0       | means, term
      ℓ̄^B0_h      | 블록 h의 B0 FIT 평균 NLL (항목당)               | nat  | > 0       | scales[block]

    해석
    ----
    조기 종료는 모형 선택에 쓰는 평가이므로 학습과 달리 경기 하나가 행 수와 상관없이 같은 몫을 갖는다 (계획 Q8).
    경기 정규화 n은 묶음과 점수를 매긴 뒤에 보고하는 층 안에서 세고, 평균은 전문가 셋에만 쓴다: 이것이 R-S4a-16 (수정)의 선택이며 Q8이 유일하게 정한 것이 아니다.
    M3Gold의 팀 항목 평균은 B0 척도와 항목 단위를 맞추지만 FIT 척도와 STOP의 행·경기 가중이 다르므로 B0 예측이어도 비가 1이라고 보장하지 않으며, 이 값은 학습 손실이 아닌 평가값이다.
    점수 없는 항목은 합과 개수에 들지 않고, 항목이 모두 점수 없는 평가 행은 세지 않으며, 점수 있는 행이 없는 블록은 합에서 빠진다 (판정 R22).
    합계와 개수만 쌓으므로 배치를 어떻게 자르든 값이 같다.

    근거
    ----
    계획 Q8·P9, 설계서 8절, 판정 R22, R-S4a-16, ``head_rows.HeadScores`` (S3는 항목을 합한다).
    """

    def __init__(self):
        self._groups: dict = {}                  # block -> {(match, row): [sum of the scored scores, number of them, stratum]}
        self._auto = 0                           # ids for the entries that come without a row id

    def add(self, block: str, nll, scored, match, row=None, size=None) -> None:
        head, _ = loss.split_key(block)
        nll, scored, match = np.asarray(nll), np.asarray(scored), np.asarray(match)
        if nll.ndim != 1 or not np.issubdtype(nll.dtype, np.floating):
            raise ValueError('nll must be a 1-d float array')
        if scored.shape != nll.shape or scored.dtype != np.bool_:
            raise ValueError('scored must be a bool array with the shape of nll')
        if match.shape != nll.shape or not np.issubdtype(match.dtype, np.integer):
            raise ValueError('match must be an integer array with the shape of nll')
        if row is None:
            row = np.arange(self._auto, self._auto + len(nll), dtype=np.int64)
            self._auto += len(nll)
        row = np.asarray(row)
        if row.shape != nll.shape or not np.issubdtype(row.dtype, np.integer):
            raise ValueError('row must be an integer array with the shape of nll')
        expert = head in loss.SIZE_NORMALISED
        if expert != (size is not None):
            raise ValueError(f'size is given for the experts {sorted(loss.SIZE_NORMALISED)} and only for them, not for {block}')
        if expert:
            size = np.asarray(size)
            if size.shape != nll.shape or not np.issubdtype(size.dtype, np.integer) or (size.size and (size.min() < 0 or size.max() >= loss.N_SIZE)):
                raise ValueError(f'size must be an integer array with the shape of nll and values in 0 .. {loss.N_SIZE - 1}')
        else:
            size = np.zeros(nll.shape, dtype=np.int64)
        take = np.flatnonzero(scored)
        if not take.size:
            return
        keys, inverse = np.unique(np.stack([match[take], row[take]], axis=1).astype(np.int64), axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        sums = np.bincount(inverse, weights=nll[take].astype(np.float64), minlength=len(keys))
        counts = np.bincount(inverse, minlength=len(keys))
        low, high = np.full(len(keys), loss.N_SIZE, dtype=np.int64), np.full(len(keys), -1, dtype=np.int64)
        np.minimum.at(low, inverse, size[take].astype(np.int64))
        np.maximum.at(high, inverse, size[take].astype(np.int64))
        if (low != high).any():
            raise ValueError('the entries of one evaluation row must have one size')
        book = self._groups.setdefault(block, {})
        for (u, r), total, count, s in zip(keys.tolist(), sums.tolist(), counts.tolist(), low.tolist()):
            cell = book.setdefault((u, r), [0.0, 0, s])
            if cell[2] != s:
                raise ValueError('the entries of one evaluation row must have one size')
            cell[0] += total
            cell[1] += count

    @staticmethod
    def _scale_of(key: str, scales: Mapping) -> float:
        scale = float(scales.get(key, float('nan')))
        if not (math.isfinite(scale) and scale > 0):
            raise ValueError(f'the scale of the block {key} must be finite and positive, not {scales.get(key)!r}')
        return scale

    @staticmethod
    def _block_mean(book: dict) -> float:
        """L_h of (TRN-2) for the entries ``book`` of one block: the mean over the size strata of the stratum mean of the match-weighted evaluation rows."""
        strata: dict = {}
        for (u, _), (row_sum, count, s) in book.items():
            cell = strata.setdefault(s, {}).setdefault(u, [0.0, 0])
            cell[0] += row_sum / count
            cell[1] += 1
        means = [sum(t / n for t, n in strata[s].values()) / len(strata[s]) for s in sorted(strata)]
        return sum(means) / len(means)

    def value(self, scales: Mapping) -> float:
        """``Loss_STOP`` (TRN-2) of everything added: the blocks in the order of ``loss.BLOCKS``, each divided by its scale and added. ValueError for a missing or non-positive
        scale of a block that has rows, and when no block has a scored row."""
        total = None
        for key in loss.BLOCKS:
            book = self._groups.get(key)
            if not book:
                continue
            term = self._block_mean(book) / self._scale_of(key, scales)
            total = term if total is None else total + term
        if total is None:
            raise ValueError('no block has a scored row: there is nothing to average')
        return total

    def block_values(self, scales: Mapping) -> dict:
        """``{block: {'mean': L_h, 'scale': ℓ̄^B0_h, 'term': L_h / ℓ̄^B0_h}}`` of every block that has a scored row, in the order of ``loss.BLOCKS`` (S4b preparation M9, decision D9).

        수식
        ----
        (TRN-2b)  L_h = 블록 평균 (TRN-2의 L_h),   term_h = L_h / ℓ̄^B0_h ,   Loss_STOP = Σ_h term_h   (같은 순서로 더한 합, 비트까지 ``value``와 같다)

          기호            | 뜻                                         | 단위 | 범위  | 코드 변수
          L_h             | 블록 h의 STOP 평균 NLL (경기 가중, 규모 층 평균) | nat  | ≥ 0   | 'mean'
          ℓ̄^B0_h          | 블록 h의 B0 평균 NLL                       | nat  | > 0   | 'scale'
          term_h          | Loss_STOP에 더해지는 몫                    | —    | ≥ 0   | 'term'

        해석: 블록마다 따로 보고하면 구역 항이 M2 NLL에 얼마나 실렸는지(결정 D9가 확인하지 못한 것)와, 구역 표지의 유무가 공유 블록 M2Rate·M2Mark의 값을 움직였는지를 같은 잣대(항목당 nat 또는 B0 척도로 나눈 몫)로
        견줄 수 있다. 두 블록은 서로 다른 행 집합(M2Rate는 프레임 행, M2Mark는 발생 행)의 평균이어서 평균끼리의 합은 (발생, 표지)의 결합 NLL이 아니므로 여기서는 더하지 않는다. 합산 방식은 B4 결정의 몫이고, 이 함수는 몫(term)의 합이 Loss_STOP인 것만 보장한다."""
        out = {}
        for key in loss.BLOCKS:
            book = self._groups.get(key)
            if not book:
                continue
            scale = self._scale_of(key, scales)
            mean = self._block_mean(book)
            out[key] = {'mean': mean, 'scale': scale, 'term': mean / scale}
        return out


def evaluate_stop(model, stop, scales: Mapping, device='cpu', guard=None, kill_law: str = 'poisson', eval_average: bool = True, zone_mark: bool = True) -> float:
    """stop 데이터셋의 전체 배치(epoch 0)에 대해 검증 손실 Loss_STOP(TRN-2)을 측정하여 반환한다.

    매개변수:
    - model: 검증할 EPT 모델.
    - stop (Batches): 조기 종료 검증용 배치 컬렉션 (epoch 0만 순회).
    - scales (Mapping): 손실 정규화에 쓰는 B0 척도 매핑.
    - device: 연산 대상 디바이스 (기본값 'cpu').
    - guard (선택): 각 배치 평가 직전에 호출되는 인자 없는 감시 콜백 함수.
    - kill_law: M3Kills의 킬 수 법칙 (결정 D5; ``score_blocks``에 그대로 넘긴다). 기본값 'poisson'은 이전 평가와 같다.
    - eval_average: True(기본값)는 이전 평가이다: 두 틀의 평균(``heads.FrameAveraged``)이어서 모든 ``swap_mode``의 모형에서 정확히 팀 교환 불변이다. False는 기록된 틀 하나만 평가한다(S4b 준비 M7, 결정 D17의
      대칭 없음 점검; 부호화기 한 번). bool이 아니면 ValueError.
    - zone_mark: True(기본값)는 이전 평가이다. False이면 M2Zone/masked 블록을 STOP 값에 넣지 않는다 (``score_blocks(zone_mark=False)``, 결정 D9, S4b 준비 M9). bool이 아니면 ValueError.

    반환값:
    - loss (float): StopMeter로 집계된 가중치 적용 손실 값.

    모드 제어 및 부작용 보장:
    - 호출 전 model.training 상태를 기록한 뒤 model.eval()로 전환하고, FrameAveraged를 eval 모드로 래핑하며 torch.no_grad() 환경에서 실행한다.
    - 기본 EPT 평가 경로는 드롭아웃을 끄고 난수 추출을 하지 않는다. 사용자 주입 guard나 배치 공급자의 부작용까지 보장하지는 않는다. 평가용 프레임 평균의 팀 교환 불변성 계약을 따른다.
    - 예외 발생 여부와 무관하게 finally 블록을 통해 model.train(was_training)으로 최상위 training 상태를 복원한다. 하위 모듈마다 달랐던 혼합 모드를 개별 복원하지는 않는다.
    loss.head_nll과 loss.scored_rows를 블록별로 계산하고 경기/행 식별자 및 해당되는 규모를 StopMeter에 전달한다. 최상위 모드 전환과 래퍼 생성은 try 진입 이전에 수행된다."""
    return _stop_meter(model, stop, device, guard, kill_law, eval_average, zone_mark).value(scales)


def evaluate_stop_blocks(model, stop, scales: Mapping, device='cpu', guard=None, kill_law: str = 'poisson', eval_average: bool = True, zone_mark: bool = True) -> dict:
    """The STOP evaluation of ``evaluate_stop`` with the value of every block reported separately (S4b preparation M9, decision D9): ``{'loss_stop': Loss_STOP, 'blocks': StopMeter.block_values(scales)}``.
    The arguments and the errors are those of ``evaluate_stop``; ``loss_stop`` is its value bit for bit. It is a call on a model (the finished run's best weights, ``TrainResult.model``), not a
    column of the epoch record: the metric files, the head schema and the checkpoint bytes of every run are unchanged by this task.

    해석: 결정 D9는 구역 표지를 공통 M2Rate + M2Mark NLL의 비열등과 V_start CRPS·가린 M3 NLL의 이득으로 판정한다. 전체 모형의 M2 NLL에는 구역 항이 실려 있어서 그대로 견주면 구역 표지가 늘 불리하므로, 두 블록의
    값을 따로 얻어 ``shared_m2_blocks``로 고른다. T002는 블록별 STOP 값을 기록하지 않아 구역 항의 크기를 확인하지 못했다."""
    meter = _stop_meter(model, stop, device, guard, kill_law, eval_average, zone_mark)
    return {'loss_stop': meter.value(scales), 'blocks': meter.block_values(scales)}


SHARED_M2_BLOCKS = ('M2Rate/masked', 'M2Mark/masked')        # the blocks that the zone term does not enter: the common NLL of decision D9


def shared_m2_blocks(report: Mapping) -> dict:
    """``{'M2Rate/masked': ..., 'M2Mark/masked': ...}``, the two separate entries (``mean``, ``scale``, ``term``) of ``report['blocks']`` (the result of ``evaluate_stop_blocks``) that decision D9 compares
    between the design with the zone mark and the one without it. They are not added here: the two are means over different row sets (the M2Rate rows are frame rows, the M2Mark rows are onsets, with different match and stratum weights), so their sum is not the joint NLL of occurrence and mark; how D9's "M2Rate + M2Mark" is formed is for the B4 decision (the per-row NLLs of ``run_inference`` carry the per-match values). ValueError naming the block when the report lacks one."""
    blocks = report['blocks']
    missing = [key for key in SHARED_M2_BLOCKS if key not in blocks]
    if missing:
        raise ValueError(f'the report has no value for the shared M2 blocks {missing}: the STOP rows held no scored entry for them')
    return {key: dict(blocks[key]) for key in SHARED_M2_BLOCKS}


def _stop_meter(model, stop, device, guard, kill_law: str, eval_average: bool, zone_mark: bool) -> StopMeter:
    """The pooled ``StopMeter`` of the STOP evaluation (the body of ``evaluate_stop``): see there for the modes and the guarantees."""
    if not isinstance(eval_average, bool):
        raise ValueError(f'eval_average must be True (the two-frame average) or False (the recorded frame only), not {eval_average!r}')
    if not isinstance(zone_mark, bool):
        raise ValueError(f'zone_mark must be True (M2Zone is scored) or False (A7: no zone block), not {zone_mark!r}')
    if zone_mark and getattr(getattr(model, 'heads', None), 'zone_mark', True) is False:         # the model without the zone mark has an untrained M2Zone head (review m2); the other direction is the D9 comparison
        raise ValueError('zone_mark is True but the model was built without the zone mark: its M2Zone block is not trained and would enter the STOP value; ask for zone_mark=False')
    zone = {} if zone_mark else {'zone_mark': False}                    # not passed by default: the call is that of every earlier evaluation
    device = torch.device(device)
    was_training = model.training
    model.eval()
    fa, meter = (heads.FrameAveraged(model) if eval_average else heads.FrameAveraged(model, average=False)).eval(), StopMeter()
    try:
        with torch.no_grad():
            for batch in stop.epoch(0):
                if guard is not None:
                    guard()
                batch = _to(batch, device)
                enc = fa.encode(batch)
                for key, (head, target, nll) in score_blocks(fa, enc, batch, kill_law, **zone).items():
                    ok = loss.scored_rows(head, target).cpu().numpy()
                    size = target['s'].cpu().numpy() if head in loss.SIZE_NORMALISED else None
                    meter.add(key, nll.cpu().numpy(), ok, target['match'].cpu().numpy(), target['row'].cpu().numpy(), size)
    finally:
        model.train(was_training)
    return meter


# ---- the start of the output biases ----------------------------------------------------------------------------------------------------
def _pool(f: dict, head: str, name: str) -> np.ndarray:
    """The field ``name`` of the masked and the open rows of ``head`` as one array (the shared output layer reads both views)."""
    return np.concatenate([np.asarray(f[(head, kind)][name]) for kind in ('masked', 'open') if (head, kind) in f])


def _rate(count: float, exposure: float) -> tuple:
    """``(bias, fallback)`` of (TRN-5): log max(count / exposure, λ_min); 0 and True when there is no exposure."""
    if exposure <= 0:
        return 0.0, True
    return math.log(max(count / exposure, RATE_FLOOR)), False


def _category(y: np.ndarray, classes: int) -> tuple:
    """``(bias vector, fallback)`` of (TRN-4) from the categories ``y`` (-1 and other negatives do not count); no observation gives the uniform law and True."""
    y = y[y >= 0]
    n = np.bincount(y.astype(np.int64), minlength=classes).astype(np.float64)
    return np.log((n + CATEGORY_PSEUDOCOUNT) / (n.sum() + classes * CATEGORY_PSEUDOCOUNT)), y.size == 0


def _first_teams_of_the_kill_rows(f: dict) -> np.ndarray:
    """The first-kill team d of the pooled M3Kills rows, in the order of ``_pool`` (the masked rows, then the open rows): the team of the mark ``m`` of a masked row (``killlaw.first_team``) and
    the field ``d`` of an open row. ValueError when an open row has no ``d`` (the rows were not made for the shifted kill law).

    (TRN-5k)의 d_i를 모은다. 열린 행의 d는 ``ept_head_rows(..., kill_law='shifted')``가 적은 필드이다."""
    parts = []
    for kind in ('masked', 'open'):
        if ('M3Kills', kind) not in f:
            continue
        block = f[('M3Kills', kind)]
        if kind == 'masked':
            parts.append(killlaw.first_team(np.asarray(block['m'], dtype=np.int64)) if len(block['y']) else np.zeros(0, dtype=np.int64))
        elif len(block['y']):
            if 'd' not in block:
                raise ValueError("the open M3Kills rows have no field 'd' (the first-kill team): make the rows with kill_law='shifted'")
            parts.append(np.asarray(block['d'], dtype=np.int64))
        else:
            parts.append(np.zeros(0, dtype=np.int64))
    return np.concatenate(parts)


def fit_biases(rows, kill_law: str = 'poisson') -> tuple:
    """``(biases, info)`` for the output layers of the heads from the FIT rows ``rows`` (an ``EngagementRows``; ValueError if its role is not FIT): ``biases`` maps the name of a
    bias (relative to ``model.heads``) to a float64 array of the shape of the bias, ``info`` records the rows used and whether a fallback was taken. The mark head's shared
    scalar is not in ``biases`` and ``info`` says why. The two views of a head are pooled. ``kill_law`` (decision D5): ``'poisson'`` (the default) is the bias of every earlier run,
    ``'shifted'`` starts the kill bias at the mean of the effective counts of the shifted law (see (TRN-5k)); only ``kill_out.bias`` and its record change.

    수식
    ----
    (TRN-4)  γ_c = log[(n_c + ζ₀) / (N + C ζ₀)]            (M2Zone C = 10, M3Size C = 4, M3Gold C = 31)
    (TRN-5)  γ = log max(Σ_i y_i / Σ_i e_i, λ_min)         (M2Rate: 관측한 칸의 합, 한 개의 스칼라; M4Conv: 칸 c마다 Σ y_{i,c} / Σ e_i, e_i > 0인 행;
             M3Obj: 칸 c마다 Σ y_{i,c} / N, 노출 1;  M3Kills: (규모 s, 팀 t)마다 Σ_{i: s_i = s} y_{i,t} / N_s, 없는 규모는 팀의 전체 평균, 그것도 없으면 0)
    (TRN-5k) M3Kills, 이동 법칙:  y*_{i,t} = y_{i,t} − 1[t = d_i],   γ_{s,t} = log max(Σ_{i: s_i = s} y*_{i,t} / N_s, λ_min)       (없는 규모는 팀의 y*의 전체 평균)
             d_i = 행 i의 첫 킬 팀 (표지 m_i // 25, 처형이면 −1)

      기호     | 뜻                                              | 단위       | 범위    | 코드 변수
      n_c, N   | 범주 c의 FIT 행 수, 범주가 있는 행의 수 (두 보기 합) | 행    | ≥ 0     | bincount, y.size
      C        | 머리의 범주 수                                  | —          | ≥ 1     | classes
      ζ₀       | 범주 가상 개수 (운영 선택)                      | 행         | > 0     | CATEGORY_PSEUDOCOUNT
      y, e     | 개수, 노출 (M2Rate 칸별 초, M4Conv 행별 창 초)  | 개, s      | ≥ 0     | fields['y'], fields['e']
      λ_min    | 편향이 담는 가장 낮은 율                        | 1/s 또는 개 | > 0   | RATE_FLOOR
      γ_c, γ   | 출력층 편향 한 칸                               | —          | 유한    | 반환값

    해석
    ----
    이 편향은 첫 출력의 중심을 FIT의 주변 분포 쪽에 놓을 뿐, 출력 가중치가 작은 무작위 값이므로 첫 예측이 그 분포와 같다고 주장하지 않는다.
    범주는 가상 개수를 더한 로그 빈도라서 나오지 않은 범주도 유한하고, 행이 없으면 균등 법칙이 된다.
    율은 개수와 노출 합의 비이며 개수가 0이어도 하한으로 유한하고, 노출이 없으면 구성 때의 0을 두고 기록한다.
    구간·표지가 나눠 쓰는 스칼라와 (규모, 팀, 층) 자리가 나눠 쓰는 골드 편향은 구조상 자리별 주변을 담지 못하고, 표지 편향은 소프트맥스에서 상쇄되므로 바꾸지 않는다.
    STOP과 DEV의 정답은 읽지 않는다 (판정 R14).
    이동 법칙 (TRN-5k)에서 킬 머리의 출력 칸은 팀 t이고 값의 뜻은 행마다 다르다: 그 행의 첫 킬 팀이면 초과분 N_d의 평균 λ_d, 아니면 킬 수의 평균 μ이다. 같은 칸을 두 종류의 행이 나눠 쓰므로
    시작 편향은 칸을 읽는 모든 행의 유효 개수 y*(첫 킬 팀이면 y − 1, 아니면 y)의 평균의 로그이다. 이것은 상수 하나로 포아송 우도를 최대화하는 값이고 d 칸만의 평균 log mean(y − 1)과
    다른 칸만의 평균 log mean(y)의 가중 평균(행 비율)이다. 모든 행이 한 팀이 먼저 낸 교전이면 그 팀의 칸은 log mean(y − 1), 상대 칸은 log mean(y)로 갈라진다. 가중치가 작은 무작위
    값이므로 첫 예측이 이 평균과 같다고 주장하지 않는 점은 같다.

    근거
    ----
    인계 5.2 (출력 편향을 FIT 주변 분포로), 계획 P4·P10 (칸별 포아송, 노출), 판정 R14·R22, 저자 결정 D5 (이동 법칙). Ref: Krichevsky and Trofimov 1981 (가상 개수; 이 정책의 근거가 아니라 서지 정보).
    """
    killlaw.check_law(kill_law)
    if getattr(rows, 'role', None) != 'FIT':
        raise ValueError(f'the start biases are computed on FIT rows only (ruling R14), not on {getattr(rows, "role", None)!r}')
    f = {key: blk.fields for key, blk in rows.blocks.items()}
    out, info = {}, {}

    def note(name, rows_used, fallback, why=None):
        info[name] = {'rows': int(rows_used), 'fallback': bool(fallback), 'shared_axes': AXES[name], **({'reason': why} if why else {})}

    rate = f[('M2Rate', 'masked')]
    e, y = np.asarray(rate['e'], dtype=np.float64), np.asarray(rate['y'], dtype=np.float64)
    seen = e > 0
    value, fb = _rate(float(y[seen].sum()), float(e[seen].sum()))
    out['rate_net.out.bias'] = np.array([value])
    note('rate_net.out.bias', seen.any(axis=1).sum(), fb, 'no observed second' if fb else None)
    for name, head, classes in (('zone_net.out.bias', 'M2Zone', loss.N_ZONE), ('size_out.bias', 'M3Size', loss.N_SIZE), ('gold_out.bias', 'M3Gold', loss.N_GOLD)):
        y = _pool(f, head, 'y')
        out[name], fb = _category(y, classes)
        note(name, (y >= 0).sum(), fb, 'no observation: uniform law' if fb else None)
    s, y = _pool(f, 'M3Kills', 's').astype(np.int64), _pool(f, 'M3Kills', 'y').astype(np.float64)
    if kill_law == 'shifted':
        y = killlaw.excess(y.astype(np.int64), _first_teams_of_the_kill_rows(f), kill_law).astype(np.float64)       # the effective counts y* of (TRN-5k)
    kill, fb = np.zeros((loss.N_SIZE, 2)), False
    for team in range(2):
        overall = float(y[:, team].sum()) / len(y) if len(y) else None
        for size in range(loss.N_SIZE):
            n = int((s == size).sum())
            mean = float(y[s == size, team].sum()) / n if n else overall
            if mean is None:
                kill[size, team], fb = 0.0, True
            else:
                kill[size, team] = math.log(max(mean, RATE_FLOOR))
                fb = fb or n == 0
    out['kill_out.bias'] = kill.reshape(-1)
    note('kill_out.bias', len(y), fb, 'a size without rows takes the mean of the team over all sizes (0 if there is none)' if fb else None)
    if kill_law == 'shifted':
        info['kill_out.bias'].update(kill_law='shifted', kill_law_note=SHIFTED_KILL_BIAS_NOTE)
    y = _pool(f, 'M3Obj', 'y').astype(np.float64)
    out['obj_net.out.bias'] = np.zeros(loss.N_CELLS) if not len(y) else np.log(np.maximum(y.sum(0) / len(y), RATE_FLOOR))
    note('obj_net.out.bias', len(y), not len(y), 'no row: 0' if not len(y) else None)
    y, e = _pool(f, 'M4Conv', 'y').astype(np.float64), _pool(f, 'M4Conv', 'e').astype(np.float64)
    live = e > 0
    out['conv_net.out.bias'] = np.log(np.maximum(y[live].sum(0) / e[live].sum(), RATE_FLOOR)) if live.any() else np.zeros(loss.N_CELLS)
    note('conv_net.out.bias', live.sum(), not live.any(), 'no conversion window: 0' if not live.any() else None)
    info['mark_net.out.bias'] = {'rows': 0, 'fallback': False, 'unchanged': True, 'shared_axes': AXES['mark_net.out.bias'],
                                 'reason': 'mark_net has a shared scalar output bias: it is added to the score of every mark, cancels in the softmax over the marks and cannot encode a mark-specific marginal'}
    return out, info


def _apply_biases(model, biases: Mapping) -> None:
    """Copy ``biases`` into the output biases of ``model.heads``; ValueError for a name that is not a parameter or a shape that differs."""
    params = dict(model.heads.named_parameters())
    with torch.no_grad():
        for name, value in biases.items():
            if name not in params or tuple(params[name].shape) != tuple(np.shape(value)):
                raise ValueError(f'{name}: no output bias of that shape in the heads')
            params[name].copy_(torch.as_tensor(np.asarray(value), dtype=params[name].dtype))


# ---- the identity of a run and the checks of the inputs ---------------------------------------------------------------------------------
def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _rows_digest(batches) -> dict:
    """The identity of a ``Batches``: role, order settings and a SHA-256 over every row field and view number of its rows (so a change of the rows is seen on resume)."""
    h, rows = hashlib.sha256(), batches.rows
    for key in sorted(rows.blocks):
        blk = rows.blocks[key]
        h.update(repr(key).encode())
        for name in sorted(blk.fields):
            a = np.ascontiguousarray(blk.fields[name])
            h.update(f'{name}|{a.dtype.str}|{a.shape}'.encode())
            h.update(a.reshape(-1).view(np.uint8))
        for name, a in (('views', blk.views), ('group', blk.group)):
            a = np.ascontiguousarray(a)
            h.update(f'{name}|{a.dtype.str}|{a.shape}'.encode())
            h.update(a.reshape(-1).view(np.uint8))
    return {'role': rows.role, 'training': bool(batches.training), 'batch_size': int(batches.batch_size), 'seed': int(batches.seed), 'batches': len(batches),
            'stats_sha256': batches.stats.sha256, 'rows_sha256': h.hexdigest()}


def _store_json_sha(data: TrainData) -> str:
    """The SHA-256 of the ``STORE.json`` bytes of the S2 store that the FIT and STOP rows read (R-S4a-18, modified). The manifest holds the file hashes of a complete store, so a rebuilt
    or replaced store changes it; no view and no file of the store is read, hashed or listed (no ``Store.verify``), and an in-place edit of a payload that leaves STORE.json alone is not
    seen. ValueError for an incomplete manifest and for FIT and STOP rows that read different stores."""
    shas = []
    for batches in (data.fit, data.stop):
        s2 = batches.rows.head_rows.s2
        if not isinstance(getattr(s2, 'manifest', None), Mapping) or not s2.manifest.get('complete'):
            raise ValueError('the S2 store of the batches has no complete manifest: refusing')
        shas.append(_sha(Path(s2.file).read_bytes()))
    if shas[0] != shas[1]:
        raise ValueError('the FIT and STOP batches read different S2 stores')
    return shas[0]


def _identity(cfg: TrainConfig, data: TrainData, static: np.ndarray, epochs: int, seed: int, store_sha: str, dev: torch.device) -> dict:
    """What a resumed run must equal: horizon, seed, settings, encoder size, static channels, scales, the two data identities, the S2 store and the execution environment (device,
    default dtype, deterministic algorithms, threads, torch version). Not the version of the code (``PRECONDITION``). JSON round trip, so a comparison after reading ``head.json`` is exact."""
    doc = {'epochs': epochs, 'seed': seed, 'lr': cfg.lr, 'weight_decay': cfg.weight_decay, 'warmup_steps': cfg.warmup_steps, 'patience': cfg.patience,
           'encoder': dataclasses.asdict(cfg.encoder), 'static_sha256': _sha(static.tobytes()),
           'scales_sha256': _sha(json.dumps({k: float(data.scales[k]) for k in sorted(data.scales)}).encode()),
           'fit': _rows_digest(data.fit), 'stop': _rows_digest(data.stop), 's2_store_json_sha256': store_sha,
           'device': str(dev), 'dtype': str(torch.get_default_dtype()), 'deterministic_algorithms': bool(torch.are_deterministic_algorithms_enabled()),
           'threads': int(torch.get_num_threads()), 'torch': torch.__version__}
    if cfg.model_spec is not None:
        from c_ept.comparison.models import validate_model_spec
        doc['model_spec'] = validate_model_spec(cfg.model_spec)
    if cfg.kill_law != 'poisson':
        doc['kill_law'] = cfg.kill_law                           # absent means the plain law: a run of the plain law has the identity of every earlier run
    if cfg.k_condition != 'raw':
        doc['k_condition'] = cfg.k_condition                     # absent means the raw gold: a run of the raw condition has the identity of every earlier run
        # the 31 numbers the K block was made of (sha256 of the float64 little-endian bytes, ``value.gold_reps_sha256``): the rows digest holds their effect, this key their identity
        # and a resume or a comparison with other representatives is refused by name (``_check_inputs`` has checked that the FIT and STOP rows hold the same ones)
        doc['k_reps_sha256'] = data.fit.rows.head_rows.k_reps_sha256()
    if cfg.decay_groups != 'all':
        doc['decay_groups'] = cfg.decay_groups                   # absent means 'all': a run with one group has the identity of every earlier run
        doc['decay_groups_sha256'] = decay_rule_sha256()         # the rule table of the groups: a resume under another policy or another rule is refused by name (the optimizer state is indexed by group)
    if cfg.column_drop != 'none':
        doc['column_drop'] = cfg.column_drop                     # absent means 'none': a run that drops no column has the identity of every earlier run
        # the lists of the dropped columns (the D7 list and the whole list) as one hash: the statistics hash in the rows digest holds them too, this key names them
        stats = data.fit.stats
        doc['column_drop_sha256'] = runlog.sha256_json({'dropped': list(stats.dropped), 'dropped_store': list(stats.dropped_store)})
    if cfg.stats_population != 'whole_fit':
        doc['stats_population'] = cfg.stats_population          # absent means 'whole_fit'
    if cfg.terrain_smooth_weight != 0:
        doc['terrain_smooth_weight'] = float(cfg.terrain_smooth_weight)       # absent means 0 (no penalty): a run without the penalty has the identity of every earlier run (S4b preparation M6, decision D8)
    if cfg.swap_mode != 'average':
        doc['swap_mode'] = cfg.swap_mode                         # absent means 'average' (both frames averaged in training): the identity of every earlier run (S4b preparation M7, decisions D10 P1, D17)
    if not cfg.eval_average:
        doc['eval_average'] = False                              # absent means True (the two-frame average in the evaluation): a resume under the other setting is refused both ways
    if cfg.on_wall_cap != 'fail':
        doc['on_wall_cap'] = cfg.on_wall_cap                     # absent means 'fail' (the earlier refusal at the wall cap): a resume under the other setting is refused both ways (S4b preparation M8, decision D10 P7)
    if not cfg.zone_mark:
        doc['zone_mark'] = False                                 # absent means True (the zone mark is part of the model): a resume under the other setting is refused both ways (S4b preparation M9, decision D9)
    return json.loads(json.dumps(doc))


def _check_identity(saved: dict, now: dict) -> None:
    """ValueError naming what differs when a resumed run is not the run of the checkpoint (before any work)."""
    differs = sorted(k for k in set(saved) | set(now) if saved.get(k) != now.get(k))
    if differs:
        raise ValueError(f'this call is not the run of the checkpoint: {differs} differ(s); use another out_dir for another run')


def _check_batches(name: str, batches, role: str, training: bool) -> None:
    got = getattr(getattr(batches, 'rows', None), 'role', None)
    if got != role or bool(getattr(batches, 'training', None)) is not training:
        raise ValueError(f'data.{name} must be the batches of the {role} rows with training={training} (ruling R14, plan Q10), not role {got!r} with '
                         f'training={getattr(batches, "training", None)!r}')
    if getattr(batches, 'stats', None) is None:
        raise ValueError(f'data.{name} has no FIT statistics (Batches(..., stats=...)): the model needs the view tensors')


def _check_inputs(cfg, data, epochs: int, seed: int) -> None:
    if not isinstance(cfg, TrainConfig) or not isinstance(data, TrainData):
        raise ValueError('cfg must be a TrainConfig and data a TrainData')
    for name, v, low in (('epochs', epochs, 1), ('seed', seed, 0)):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < low:
            raise ValueError(f'{name} must be a whole number of at least {low}, got {v!r}')
    _check_batches('fit', data.fit, 'FIT', True)
    _check_batches('stop', data.stop, 'STOP', False)
    for name, batches in (('fit', data.fit), ('stop', data.stop)):
        # the rows hold the first-kill team d only when they were made for the shifted kill law (a mock without ``head_rows`` is the plain law)
        made_for = getattr(getattr(batches.rows, 'head_rows', None), 'kill_law', 'poisson')
        if made_for != cfg.kill_law:
            raise ValueError(f'data.{name} holds the rows of the kill law {made_for!r} but the settings ask for kill_law {cfg.kill_law!r}: '
                             "make the rows with ept_head_rows(..., kill_law=cfg.kill_law)")
        # the K block of the rows holds the true gold or the representative of the gold bin: the other condition is another input (a mock without ``head_rows`` is the raw gold)
        made_with = getattr(getattr(batches.rows, 'head_rows', None), 'k_condition', 'raw')
        if made_with != cfg.k_condition:
            raise ValueError(f'data.{name} holds the rows of the k_condition {made_with!r} but the settings ask for k_condition {cfg.k_condition!r}: '
                             "make the rows with ept_head_rows(..., k_condition=cfg.k_condition)")
    # the FIT batches carry the coins of the random view swap only when they were made for it (S4b preparation M7): the mode of the batches and the mode of the settings are one claim (a mock
    # without the record is the plain average). The STOP batches are never swapped at random, so they have no claim to check.
    made_for_swap = getattr(data.fit, 'swap_mode', 'average')
    if made_for_swap != cfg.swap_mode:
        raise ValueError(f'data.fit holds batches made for swap_mode {made_for_swap!r} but the settings ask for swap_mode {cfg.swap_mode!r}: '
                         'make the batches with data.Batches(..., swap_mode=cfg.swap_mode)')
    # the statistics must have been made under the settings of the run (decisions D7 and D12): the dropped columns are an input of the model, the population is a claim of the record
    # (a mock of the statistics without the record is the plain statistics of every earlier run)
    for name, batches in (('fit', data.fit), ('stop', data.stop)):
        made = getattr(batches.stats, 'column_drop', 'none')
        if made != cfg.column_drop:
            raise ValueError(f'data.{name} reads statistics made with column_drop {made!r} but the settings ask for column_drop {cfg.column_drop!r}: '
                             'make the statistics with data.fit_stats(..., column_drop=cfg.column_drop, store_constant=...)')
        population = getattr(batches.stats, 'population', 'whole_fit')
        if population != cfg.stats_population:
            raise ValueError(f'data.{name} reads statistics of the population {population!r} but the settings ask for stats_population {cfg.stats_population!r}: '
                             'make the statistics with data.fit_stats(..., stats_population=cfg.stats_population)')
    # the zone mark can be taken off the EPT only (decision D9): the N2 model has its own heads, whose conditions this switch does not reach (a refusal before any work, not a quiet change of the claim)
    if not cfg.zone_mark and cfg.model_spec is not None and cfg.model_spec['architecture'] != 'ept_s_v1':
        raise ValueError(f"zone_mark is False but model_spec names the architecture {cfg.model_spec['architecture']!r}: the zone mark can be taken off the EPT ('ept_s_v1') only")
    # a terrain smoothness weight needs a learned terrain channel to smooth (the EPT and N2 both build their field from ``encoder.field_channels``): none is a mistake of the profile, not a quiet 0
    if cfg.terrain_smooth_weight != 0 and cfg.encoder.field_channels == 0:
        raise ValueError(f'terrain_smooth_weight is {cfg.terrain_smooth_weight!r} but encoder.field_channels is 0: the model has no learned terrain channel to smooth; '
                         'set terrain_smooth_weight to 0 or field_channels to at least 1')
    # FIT and STOP rows of the binned condition must hold the same 31 representatives (rows made from two label stores would train on one K and stop on another)
    if cfg.k_condition == 'binned' and data.fit.rows.head_rows.k_reps_sha256() != data.stop.rows.head_rows.k_reps_sha256():
        raise ValueError('the FIT and STOP rows hold different representatives of the gold bins (k_reps_sha256 differs): the early stopping would score the K heads on another K')
    if data.fit.stats.sha256 != data.stop.stats.sha256:
        raise ValueError('the FIT and STOP batches must use the same FIT statistics (Stats.sha256 differs): the two sides would read different inputs')
    want = (terrain.GRID, terrain.GRID, len(encoder.STATIC_CHANNELS))
    if np.shape(data.static) != want:
        raise ValueError(f'data.static must be the array of {STATIC_FUNCTION} with the shape {want}, not {np.shape(data.static)}')
    if not isinstance(data.scales, Mapping) or set(data.scales) != set(loss.BLOCKS):
        raise ValueError(f'scales must have exactly the {len(loss.BLOCKS)} keys of loss.BLOCKS')
    for key, v in data.scales.items():
        try:
            ok = math.isfinite(float(v)) and float(v) > 0
        except (TypeError, ValueError):
            ok = False
        if not ok:
            raise ValueError(f'the B0 scale of {key} must be finite and positive, not {v!r}')


# ---- files and checkpoints -------------------------------------------------------------------------------------------------------------
# 기존 호출/monkeypatch 경계만 남긴다. IO 구현은 checkpoints가 소유한다.
_write_atomic = checkpoints.write_atomic
_torch_bytes = checkpoints.torch_bytes
_json_bytes = checkpoints.json_bytes


def _rng_state(device: torch.device) -> dict:
    py, np_state = random.getstate(), np.random.get_state()
    state = {'torch': torch.get_rng_state(), 'numpy': {'key': torch.from_numpy(np_state[1].astype(np.int64)), 'pos': int(np_state[2]), 'has_gauss': int(np_state[3]),
                                                       'cached': float(np_state[4])},
             'python': {'version': py[0], 'state': list(py[1]), 'gauss': py[2]}}
    if device.type == 'cuda':
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: dict) -> None:
    torch.set_rng_state(state['torch'])
    n = state['numpy']
    np.random.set_state(('MT19937', n['key'].numpy().astype(np.uint32), n['pos'], n['has_gauss'], n['cached']))
    p = state['python']
    random.setstate((p['version'], tuple(p['state']), p['gauss']))
    if 'cuda' in state:
        torch.cuda.set_rng_state_all(state['cuda'])


_read_verified = checkpoints.read_verified
_latest_checkpoint = checkpoints.latest_checkpoint


def _save_checkpoint(out_dir: Path, epoch_done: int, files: dict, meta: dict) -> tuple:
    """현재 `_write_atomic` 교체를 저장마다 반영하는 호환 경계."""
    return checkpoints.save_checkpoint(out_dir, epoch_done, files, meta, write=_write_atomic)


# ---- the memory, the clock and the resource guard --------------------------------------------------------------------------------------
_PROCESS = None


def _metric_name(mem) -> str:
    """``private_bytes_windows`` where psutil reports private bytes; otherwise ``rss_bytes_fallback`` (RSS is not private memory)."""
    return 'private_bytes_windows' if getattr(mem, 'private', None) is not None else 'rss_bytes_fallback'


def _memory_sample() -> tuple:
    """One sample of this process and its children: ``(bytes of the metric, observed lifetime upper bound or None, rss bytes, metric name)`` from the public runtime tree sampler;
    ``(0, None, 0, 'unavailable')`` when nothing could be read, which the guard refuses (a test replaces this function)."""
    global _PROCESS
    if psutil is None:
        return 0, None, 0, 'unavailable'
    try:
        if _PROCESS is None:
            _PROCESS = psutil.Process()
        procs = [_PROCESS, *_PROCESS.children(recursive=True)]
        used, upper = runtime.tree_memory(procs, process_api=psutil)
        rss, name = 0, 'unavailable'
        for p in procs:
            try:
                mem = p.memory_info()
            except psutil.Error:
                continue
            rss, name = rss + int(mem.rss), _metric_name(mem)
    except psutil.Error:
        return 0, None, 0, 'unavailable'
    return int(used), (None if upper is None else int(upper)), rss, name


def _violation(peak: int, metric: str, blind: bool, seconds: float, wall: bool = True):
    """The reason (a sentence) that a bound is broken, or None: the memory was not measured, the peak is not strictly under 40 GiB (R43), or the seconds are above 24 h. ``wall`` False leaves
    the seconds out (the run asked to end at the wall cap with its best STOP checkpoint, ``TrainConfig.on_wall_cap``: the time is then acted on at the end of an epoch, not refused)."""
    if blind or metric == 'unavailable':
        return 'could not measure the memory (a sample of 0 bytes is not a measured 0): refusing to run unmeasured'
    if peak >= MEMORY_CAP_BYTES:
        return f'the sampled {metric} peak {peak / 2 ** 30:.2f} GiB is not under the 40 GiB bound (R43)'
    if wall and seconds > WALL_CAP_SECONDS:
        return f'the training time {seconds / 3600:.2f} h is above the 24 h bound (all calls of the run)'
    return None


class ResourceBoundError(RuntimeError):
    """The sampled memory peak is not under 40 GiB, the memory could not be measured, the seconds are above 24 h (R43, design 7), or the run folder holds the record of an earlier failure
    or of a crash loop. ``record`` False: the refusal is not stored again in the run folder."""

    def __init__(self, message: str, record: bool = True):
        super().__init__(message)
        self.record = record


class _Resources:
    """The clock, the memory sampler and the heartbeat of one ``train()`` call, started at its first line: the sampled peaks, the seconds and the accounting adopted from the earlier
    calls. ``guard()`` reads the sampler's peak and the clock between batches without a sample of its own and raises ``ResourceBoundError`` when a bound is broken; ``boundary()`` takes
    an explicit sample first (epoch boundaries and checkpoint writes). The sampling thread and the main thread both call ``_measure``, which the lock serialises."""

    def __init__(self):
        self.lock = threading.Lock()
        self.track = {'rss': 0, 'metric': 'unavailable', 'blind': False}
        self.prior = {'seconds': 0.0, 'peak_bytes': 0, 'peak_upper_bytes': None, 'peak_rss_bytes': 0}
        self.epoch_done, self.head_sha = 0, None
        self.wall_soft = False                                   # True under on_wall_cap='best_stop' (set by `_run`): the guards leave the seconds out, the end of an epoch acts on them (TRN-11)
        self.beat_path, self.beat_origin, self.beat_at, self.beat_error = None, 0.0, 0.0, None
        self.sampler = runtime.ProcessTreeSampler(MEMORY_INTERVAL_S, self._measure)
        self.started = _clock()

    def _measure(self) -> tuple:
        used, upper, rss, name = _memory_sample()
        with self.lock:
            self.track['rss'], self.track['metric'] = max(self.track['rss'], int(rss)), name
            if used <= 0:
                self.track['blind'] = True
            self._beat(int(used))
        return used, upper

    def _beat(self, used: int) -> None:
        """Append and fsync one line ``{"seconds": the seconds of this call since its pending marker, "peak_bytes": the peak so far}`` when ``BEAT_INTERVAL_S`` has passed since the
        last one (the lock is held). A write that fails is kept and raised by the next ``guard()``: a call without its heartbeat is not charged safely."""
        if self.beat_path is None or self.beat_error is not None or _clock() - self.beat_at < BEAT_INTERVAL_S:
            return
        self.beat_at = now = _clock()
        line = json.dumps({'seconds': now - self.beat_origin, 'peak_bytes': max(int(self.prior['peak_bytes']), int(self.sampler.peak), used)}) + '\n'
        try:
            with open(self.beat_path, 'ab') as handle:
                handle.write(line.encode('utf-8'))
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as err:
            self.beat_error = err

    def arm_beat(self, path: Path, origin: float) -> None:
        with self.lock:
            self.beat_path, self.beat_origin = path, origin
            self.beat_at = self.beat_origin

    def adopt(self, record: dict) -> None:
        self.prior = {k: record[k] for k in self.prior}
        if self.track['metric'] == 'unavailable' and record['metric'] in METRICS:
            self.track['metric'] = record['metric']

    def seconds(self) -> float:
        return self.prior['seconds'] + (_clock() - self.started)

    def peak(self) -> int:
        return max(int(self.prior['peak_bytes']), int(self.sampler.peak))

    def snapshot(self, *, now: float | None = None) -> dict:
        a, b = self.prior['peak_upper_bytes'], self.sampler.peak_upper
        with self.lock:
            rss, metric = max(int(self.prior['peak_rss_bytes']), self.track['rss']), self.track['metric']
        return {'seconds': self.prior['seconds'] + ((_clock() if now is None else now) - self.started), 'peak_bytes': self.peak(), 'peak_upper_bytes': b if a is None else a if b is None else max(a, b), 'peak_rss_bytes': rss, 'metric': metric}

    def violation(self):
        return _violation(self.peak(), self.track['metric'], self.track['blind'], self.seconds(), wall=not self.wall_soft)

    def guard(self) -> None:
        if self.beat_error is not None:
            raise self.beat_error
        reason = self.violation()
        if reason:
            raise ResourceBoundError(reason, record=not (self.track['blind'] or self.track['metric'] == 'unavailable'))

    def boundary(self) -> None:
        self.sampler.sample()
        self.guard()


# ---- the records: failures, call markers, orphans --------------------------------------------------------------------------------------
def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_seconds(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x >= 0


def _check_accounting(rec: dict, where: str) -> None:
    """RuntimeError unless ``rec`` has all the ``ACCOUNTING_KEYS`` with finite non-negative seconds, whole non-negative byte counts (``peak_upper_bytes`` may be None) and a known metric."""
    try:
        seconds, peak, upper, rss, metric = (rec[k] for k in ACCOUNTING_KEYS)
        ok = (_is_seconds(seconds) and all(_is_int(v) and v >= 0 for v in (peak, rss)) and (upper is None or (_is_int(upper) and upper >= 0)) and metric in (*METRICS, 'unavailable'))
    except (KeyError, TypeError):
        ok = False
    if not ok:
        raise RuntimeError(f'{where}: the accounting is missing a field or holds a number that is not finite and non-negative, or an unknown metric; not guessing')


def _merge_accounting(a: dict | None, b: dict | None) -> dict | None:
    """The elementwise maximum of two accounting records (None counts as absent); the metric of the first that has a measured one."""
    if a is None or b is None:
        return None if a is None and b is None else {k: (a or b)[k] for k in ACCOUNTING_KEYS}
    uppers = [x for x in (a['peak_upper_bytes'], b['peak_upper_bytes']) if x is not None]
    return {'seconds': max(a['seconds'], b['seconds']), 'peak_bytes': max(a['peak_bytes'], b['peak_bytes']), 'peak_upper_bytes': max(uppers) if uppers else None,
            'peak_rss_bytes': max(a['peak_rss_bytes'], b['peak_rss_bytes']), 'metric': next((r['metric'] for r in (a, b) if r['metric'] in METRICS), a['metric'])}


def _head_accounting(meta: dict) -> dict:
    """The accounting that a head.json recorded just before it was written."""
    m = meta['memory']
    return {'seconds': meta['seconds'], 'peak_bytes': m['peak_bytes'], 'peak_upper_bytes': m['peak_upper_bytes'], 'peak_rss_bytes': m['peak_rss_bytes'], 'metric': m['metric']}


def _refuse_if_failed(out_dir: Path) -> None:
    """ResourceBoundError (not stored again) when the folder holds the record of an earlier failure: a run that broke a bound is over, a new ``out_dir`` is a new run."""
    if out_dir.is_dir() and any(out_dir.glob(FAILURE_GLOB)):
        raise ResourceBoundError('an earlier call of this run broke a resource bound (a resource_failure_*.json is in the run folder): this run is over, use another out_dir',
                                 record=False)


def _record_failure(out_dir: Path, res: _Resources, reason: str) -> None:
    """Store a new ``resource_failure_<epochs done>.json`` with the accounting at the failure (never over an existing one)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'resource_failure_{res.epoch_done:04d}.json'
    if not path.exists():
        _write_atomic(path, _json_bytes({'schema': SCHEMA, 'epoch_done': res.epoch_done, 'reason': reason, **res.snapshot()}))


def _read_beats(path: Path) -> tuple:
    """``(number of complete heartbeat lines, the last one or None)`` of a ``call_<n>.beat`` file (``(0, None)`` without the file). Text after the last newline is an append that was cut
    and is left out; a complete line that is not ``{"seconds", "peak_bytes"}`` with finite non-negative seconds that do not go down and a whole non-negative peak is damage (RuntimeError)."""
    try:
        raw = Path(path).read_bytes()
    except FileNotFoundError:
        return 0, None
    lines, last = raw.split(b'\n')[:-1], None
    for i, line in enumerate(lines, 1):
        try:
            beat = json.loads(line.decode('utf-8'))
            ok = (isinstance(beat, dict) and set(beat) == {'seconds', 'peak_bytes'} and _is_seconds(beat['seconds']) and _is_int(beat['peak_bytes']) and beat['peak_bytes'] >= 0
                  and (last is None or beat['seconds'] >= last['seconds']))
        except ValueError:
            ok = False
        if not ok:
            raise RuntimeError(f'{Path(path).name}: heartbeat line {i} is not valid; not guessing')
        last = beat
    return len(lines), last


def _final_record(call: int, identity_sha: str, pending_sha: str, epochs: tuple, head_sha, outcome: str, status: str, reason, evidence, acct: dict) -> dict:
    return {'schema': SCHEMA, 'kind': 'final', 'call': call, 'identity_sha256': identity_sha, 'pending_sha256': pending_sha, 'epoch_done_start': epochs[0], 'epoch_done_end': epochs[1],
            'head_sha256_end': head_sha, 'outcome': outcome, 'status': status, 'reason': reason, 'evidence': evidence, **{k: acct[k] for k in ACCOUNTING_KEYS}}


def _load_calls(out_dir: Path, identity_sha: str) -> tuple:
    """``(final markers, orphan)`` of the calls of this run, after every marker was checked for its exact keys, its run (identity hash), its number (1, 2, ... without a gap, a pending
    marker for every final one, the pending hash inside the final marker) and its accounting. ``orphan`` is None or the last call, which has a pending marker and no final marker:
    ``{'call', 'pending', 'pending_sha', 'beats', 'beat'}`` with its last complete heartbeat. A final marker with the status ``exceeded`` ends the run (``ResourceBoundError``)."""
    folder = out_dir / CALLS_DIR
    pend, fin = {}, {}
    for p in (folder.iterdir() if folder.is_dir() else []):
        m = CALL_NAME.fullmatch(p.name)
        if m:
            (pend if m.group(2) else fin)[int(m.group(1))] = p
    numbers = sorted(set(pend) | set(fin))
    if numbers != list(range(1, len(numbers) + 1)):
        raise RuntimeError('the call markers of this run are not numbered 1, 2, ...; not guessing')
    finals, orphan = [], None
    for n in numbers:
        where = f'call marker {n}'
        if n not in pend:
            raise RuntimeError(f'{where}: a final marker without its pending marker; not guessing')
        raw = pend[n].read_bytes()
        try:
            doc = json.loads(raw.decode('utf-8'))
            sound = isinstance(doc, dict) and set(doc) == PENDING_KEYS and doc['schema'] == SCHEMA and doc['kind'] == 'pending' and doc['call'] == n
        except ValueError:
            sound = False
        if not sound or doc['identity_sha256'] != identity_sha:
            raise RuntimeError(f'{where}: the pending marker is unreadable or belongs to another run; not guessing')
        _check_accounting(doc['start_accounting'], f'{where} (pending)')
        if n not in fin:
            if n != numbers[-1]:
                raise RuntimeError(f'{where}: a pending marker without a final marker before a later call; not guessing')
            beats, beat = _read_beats(folder / f'call_{n:04d}.beat')
            orphan = {'call': n, 'pending': doc, 'pending_sha': _sha(raw), 'beats': beats, 'beat': beat}
            continue
        try:
            rec = json.loads(fin[n].read_text(encoding='utf-8'))
            sound = (isinstance(rec, dict) and set(rec) == CALL_KEYS and rec['schema'] == SCHEMA and rec['kind'] == 'final' and rec['call'] == n and rec['identity_sha256'] == identity_sha
                     and rec['pending_sha256'] == _sha(raw) and rec['outcome'] in _OUTCOMES and rec['status'] in _OUTCOMES[rec['outcome']])
        except (OSError, ValueError):
            sound = False
        if not sound:
            raise RuntimeError(f'{where}: the final marker is unreadable, has other keys or is not the record of this pending marker and run; not guessing')
        _check_accounting(rec, f'{where} (final)')
        finals.append(rec)
    if any(rec['status'] == 'exceeded' for rec in finals):
        raise ResourceBoundError('an earlier call of this run ended over a resource bound (a final call marker says exceeded): this run is over, use another out_dir', record=False)
    return finals, orphan


def _adopt_orphan(out_dir: Path, orphan: dict, finals: list, base, res: _Resources, identity_sha: str) -> dict:
    """Write the missing final marker of a call that a hard interrupt (a power cut, a kill) left without one, and return it. The call is charged the seconds of its pending start + its
    last heartbeat + one period, and the larger of the peaks it recorded (the replay of what it lost is sampled again by the new call); ``base`` is the accounting of the newest
    checkpoint, which the charge never goes below. The marker is a new file with the outcome ``hard_interrupted`` and the evidence. ResourceBoundError (the failure is stored) when the
    charge breaks a bound; ResourceBoundError, without a marker, for a crash loop: two consecutive hard interrupts and no epoch finished by the second, which every later call finds again.

    수식
    ----
    (TRN-6)  Z_n^sec = Z_{n−1}^sec + beat_n^sec + Δ_beat ;   Z_n^peak = max(Z_{n−1}^peak, beat_n^peak)           (beat_n = 0 without a heartbeat)

      기호        | 뜻                                                               | 단위 | 범위 | 코드 변수
      Z_{n−1}     | 호출 n이 시작할 때의 누적 회계 (pending 표지의 start_accounting) | s, 바이트 | ≥ 0 | start
      beat_n      | 호출 n의 마지막 완전한 박동 줄 (표지 이후 초, 지금까지의 최댓값) | s, 바이트 | ≥ 0 | beat
      Δ_beat      | 박동 주기: 마지막 박동 뒤에 기록되지 않았을 수 있는 꼬리        | s    | > 0  | BEAT_INTERVAL_S
      Z_n         | 호출 n이 끝난 것으로 청구한 누적 회계                            | s, 바이트 | ≥ 0 | charged

    해석
    ----
    박동 초는 pending 회계와 같은 쓰기 전 시각에서 시작하여 pending 쓰기 시간도 포함한다. 기록되지 않은 꼬리의 주기 하나 상한은 마지막 박동 시각부터 중단까지가 한 주기 이내라는 조건에 한한다. 정전 시간은 청구하지 않으며, 스케줄러 지연이나 fsync 정지가 이 조건을 깨면 무조건적인 시간 상한은 보장하지 않는다.
    같은 곳에서 두 번 연달아 죽는 실행은 이어가도 나아가지 않으므로 거부한다. 검출한 초과는 실패 기록으로 남아 이후 호출도 거부한다(닫힌 채 실패).

    근거
    ----
    R-S4a-17 (대체), 설계서 7절, 계약 v4 3절.
    """
    pending, n = orphan['pending'], orphan['call']
    if finals and finals[-1]['outcome'] == 'hard_interrupted' and pending['epoch_done_start'] == res.epoch_done:
        raise ResourceBoundError('the last two calls were hard interrupts and the second finished no epoch (two consecutive hard interrupts without a new epoch: a crash loop): '
                                 'this run is not resumed again, find the cause and use another out_dir', record=False)
    start, beat = pending['start_accounting'], orphan['beat']
    claim = {**start, 'seconds': start['seconds'] + (beat['seconds'] if beat else 0.0) + BEAT_INTERVAL_S, 'peak_bytes': max(start['peak_bytes'], beat['peak_bytes'] if beat else 0)}
    charged = _merge_accounting(base, claim)
    reason = _violation(charged['peak_bytes'], charged['metric'], False, charged['seconds'])
    evidence = {'basis': 'beat' if beat else 'pending_start', 'beats': orphan['beats'], 'beat': beat, 'tail_seconds': BEAT_INTERVAL_S}
    rec = _final_record(n, identity_sha, orphan['pending_sha'], (pending['epoch_done_start'], res.epoch_done), res.head_sha, 'hard_interrupted', 'exceeded' if reason else 'ok', reason,
                        evidence, charged)
    _write_atomic(out_dir / CALLS_DIR / f'call_{n:04d}.json', _json_bytes(rec))
    if reason:
        res.adopt(charged)
        _record_failure(out_dir, res, reason)
        raise ResourceBoundError(reason, record=False)
    return rec


class _Call:
    """The pending and the final marker of one ``train()`` call (new files in ``calls/``). ``begin`` writes the pending marker, with the accounting at the start, and starts the
    heartbeat; ``finish`` writes the final marker with the accounting at that moment: the snapshot is taken before the marker's own write, which it excludes."""

    def __init__(self, out_dir: Path):
        self.dir, self.pending_written, self.final_written = out_dir / CALLS_DIR, False, False
        self.index, self.identity_sha, self.pending_sha, self.start_epoch = 0, '', '', 0

    def begin(self, identity_sha: str, res: _Resources) -> None:
        numbers = [int(m.group(1)) for p in (self.dir.iterdir() if self.dir.is_dir() else []) if (m := CALL_NAME.fullmatch(p.name))]
        self.index, self.identity_sha, self.start_epoch = max(numbers, default=0) + 1, identity_sha, res.epoch_done
        self.dir.mkdir(parents=True, exist_ok=True)
        origin = _clock()  # the pending snapshot and beat elapsed time share this instant, before the pending write
        payload = _json_bytes({'schema': SCHEMA, 'kind': 'pending', 'call': self.index, 'identity_sha256': identity_sha, 'epoch_done_start': res.epoch_done,
                               'head_sha256_start': res.head_sha, 'start_accounting': res.snapshot(now=origin)})
        _write_atomic(self.dir / f'call_{self.index:04d}.pending.json', payload)
        self.pending_sha, self.pending_written = _sha(payload), True
        res.arm_beat(self.dir / f'call_{self.index:04d}.beat', origin)

    def finish(self, res: _Resources, outcome: str, reason) -> dict:
        res.beat_path = None
        snap = res.snapshot()
        found = _violation(snap['peak_bytes'], snap['metric'], outcome == 'unmeasured', snap['seconds'], wall=not res.wall_soft)
        if outcome in ('completed', 'interrupted') and found:
            outcome, reason = 'exceeded', found
        rec = _final_record(self.index, self.identity_sha, self.pending_sha, (self.start_epoch, res.epoch_done), res.head_sha, outcome, _OUTCOMES[outcome][0], reason, None, snap)
        _write_atomic(self.dir / f'call_{self.index:04d}.json', _json_bytes(rec))
        self.final_written = True
        return rec


def _abort(ctx: _Call, res: _Resources, out_dir: Path, err: BaseException) -> None:
    """Finalize the call marker of a call that ends with an exception: ``interrupted`` (under the bounds; the run can be resumed), ``exceeded`` (a failure file is stored first) or
    ``unmeasured``. A marker that cannot be written leaves the call without an end, which the next call adopts as it does after a hard interrupt."""
    if not ctx.pending_written or ctx.final_written:
        return
    try:
        res.sampler.sample()
        blind = res.track['blind'] or res.track['metric'] == 'unavailable'
        reason = res.violation()
        if blind:
            outcome = 'unmeasured'
        elif isinstance(err, ResourceBoundError) or reason:
            outcome, reason = 'exceeded', reason or str(err)
        else:
            outcome = 'interrupted'
        if outcome == 'exceeded':
            _record_failure(out_dir, res, reason)
        ctx.finish(res, outcome, reason)
    except Exception as err2:
        err.add_note(f'the call marker could not be finalized ({err2.__class__.__name__}): the next call adopts the call as a hard interrupt')


# ---- the run ---------------------------------------------------------------------------------------------------------------------------
def train(cfg: TrainConfig, data: TrainData, out_dir, *, device, epochs: int, check, seed: int) -> TrainResult:
    """Train the EPT on FIT with early stopping on STOP and return the best-STOP model. ``out_dir`` holds one folder ``epoch_<k>`` per finished epoch and a ``calls/`` folder with the
    markers of every call. A call on a folder with a complete checkpoint resumes from it: the call that a hard interrupt left without an end is adopted and charged first, and a run that
    is already finished returns its result without training and without a new call. ``epochs`` is the total horizon of the schedule, ``check`` the E6 guard (no argument; once for
    every epoch the call runs, the first before the preparation), ``device`` a torch device name (reported values: ``'cpu'``). The clock, the sampler and the heartbeat run from the first
    line until the final marker is written. ValueError before any work for a default dtype other than float32, for a role, order, statistics, scales or static array that is not the
    contract's, and for a resume whose horizon, seed, settings, static channels, scales, rows, S2 store or execution environment differ from the checkpoint's; RuntimeError for a damaged
    or inconsistent record; ``ResourceBoundError`` for the memory and time bounds, for a crash loop and for an earlier stored failure; FloatingPointError for a loss that is not finite.

    수식
    ----
    (TRN-3)  Loss_STOP,k < min_{k' < k} Loss_STOP,k' ⇒ 최선 = k, n_bad ← 0 ;   아니면 n_bad ← n_bad + 1 ;   n_bad ≥ n_pat이면 종료하고 최선 에폭의 모형을 돌려준다
    (TRN-7)  θ_{t+1} = θ_t (1 − η_t ϑ) − η_t · A₁ / (√A₂ + ε_A),   A₁ ← β₁ A₁ + (1 − β₁) ∇L_t,   A₂ ← β₂ A₂ + (1 − β₂) (∇L_t)²   (편향 보정 생략 표기)

      기호            | 뜻                                              | 단위   | 범위     | 코드 변수
      Loss_STOP,k     | 에폭 k 끝의 STOP 값 (TRN-2)                     | —      | ≥ 0      | stop_value
      k, n_bad, n_pat | 에폭 번호(0부터), 개선 없는 에폭 수, 인내       | 에폭   | ≥ 0, ≥ 0, ≥ 1 | epoch, loop['bad'], cfg.patience
      η_t, ϑ          | 학습률 (TRN-1), 가중치 감쇠 (매개변수마다: TRN-8) | —    | ≥ 0      | lr_at, cfg.weight_decay, param_groups
      θ, ∇L_t         | 매개변수, FIT 배치의 총손실 (LOSS-4)의 기울기   | —      | 실수     | model, total.backward
      A₁, A₂          | 기울기와 그 제곱의 지수이동평균 (AdamW 상태)    | —      | 실수     | 최적화기 상태
      β₁, β₂, ε_A     | AdamW 기본값                                    | —      | —        | torch.optim.AdamW

    해석
    ----
    기울기는 FIT 배치에서만 나오고 STOP은 최선 에폭과 멈춤만 정하므로, STOP이 달라져도 앞 에폭들의 ``last.pt``는 비트 같다.
    이어하기는 마지막 에폭의 모형·최적화기·난수 상태를 읽어 드롭아웃 마스크까지 끊지 않은 실행과 같고, 체크포인트는 파일을 쓴 뒤 ``head.json``을 마지막에 써서 끊긴 쓰기를 알아본다.
    최선 상태는 개선된 에폭의 폴더에만 쓰고 ``head.json``이 그 폴더를 가리키며, 끊긴 폴더를 다시 쓰면 그 안의 파일이 바뀐다.
    ``fit_batch_mean_losses``는 학습 모드 배치 손실의 에폭 평균이며 자료 전체의 손실도 평가도 아니다.
    ``cfg.decay_groups='split'``(결정 D6)이면 감쇠 ϑ는 매개변수마다 다르고(TRN-8) 에폭마다 감쇠 척도 κ를 기록하며(TRN-9) 머리 스키마가 v7이다. 정책이 다른 체크포인트에서의 이어하기는 정체성이 달라서 거부되고,
    같은 정책이어도 묶음이 매개변수에 준 감쇠(헤드의 ``decay_groups`` 기록)가 다르면 최적화기 상태를 읽기 전에 거부한다.
    ``cfg.on_wall_cap``(결정 D10 P7)이 ``'best_stop'``이면 시간 합이 24 h에 닿은 에폭을 끝까지 마치고 ``done=True``로 끝낸다(TRN-11): 최선 STOP 체크포인트와 그 상태가 반환 모형이고, 머리의 ``budget_end``와 에폭 지표의
    ``termination_reason='budget'``이 이유를 적으며, 호출 표지는 정상 ``completed``이고 실패 기록은 없다. 끝난 실행에 다시 부르면 일을 하지 않는다. 기본값 ``'fail'``에서는 아무것도 달라지지 않는다.
    ``cfg.terrain_smooth_weight``(결정 D8)가 0이 아니면 기울기의 손실은 Loss_EPT + λ_Ω · R_Ω(LOSS-5)이고, 에폭마다 가중 없는 R_Ω의 걸음 평균을 지표 열 terrain_penalty에 적으며(EM-2) 머리 스키마가 v8이다.
    ``fit_batch_mean_losses``는 벌점이 없는 데이터 손실 Loss_EPT의 평균이고 STOP 값에도 벌점이 들어가지 않으므로 최선 에폭과 멈춤의 규칙은 가중과 상관없이 같다. 가중이 다른 체크포인트에서의 이어하기는 정체성이 달라 거부된다.

    근거
    ----
    과제 5 작업서 (AdamW, 예열·코사인, 인내), 계획 Q8·Q10, 설계서 7절 (운영 상한, 체크포인트·이어하기), 판정 R14·R43, R-S4a-16..20, 인계 5.2, 계약 v4, Ref: Loshchilov and Hutter 2019 (AdamW).
    """
    res, out_dir = _Resources(), Path(out_dir)
    ctx = _Call(out_dir)
    with res.sampler:
        try:
            return _run(cfg, data, out_dir, device, epochs, check, seed, res, ctx)
        except BaseException as err:
            if isinstance(err, ResourceBoundError) and err.record and not ctx.final_written:
                _record_failure(out_dir, res, str(err))
            _abort(ctx, res, out_dir, err)
            raise


def _head_record(loop: dict, identity: dict, static_record: dict, epochs: int, seed: int, done: bool, snap: dict, samples: int, decay: dict | None = None, budget: bool = False) -> dict:
    """The head.json of an epoch. Its schema is v6 (every earlier run) or, for a policy other than 'all', v7 (``epoch_metrics.head_schema``); ``decay`` (``_decay_record``) is written
    only then. `budget` (the run ends at the wall cap, `TrainConfig.on_wall_cap`) adds the key `budget_end` True, which no other head has (`epoch_metrics` validates it)."""
    head = {'schema': epoch_metrics.head_schema(identity), 'epoch': loop['epoch'], 'epochs': epochs, 'seed': seed, 'done': done, 'stopped_early': loop['stopped_early'], 'best_epoch': loop['best_epoch'],
            'best': loop['best'], 'identity': identity, 'static': static_record, 'bias_init': loop['bias_init'], 'threads': torch.get_num_threads(), 'seconds': snap['seconds'],
            'caps': {'memory_bytes': MEMORY_CAP_BYTES, 'wall_seconds': WALL_CAP_SECONDS},
            'memory': {'metric': snap['metric'], 'peak_bytes': snap['peak_bytes'], 'peak_upper_bytes': snap['peak_upper_bytes'], 'peak_rss_bytes': snap['peak_rss_bytes'],
                       'sampled': True, 'interval_s': MEMORY_INTERVAL_S, 'samples': samples, 'scope': MEMORY_SCOPE},
            'timing': {k: list(loop[k]) for k in ('load_seconds', 'step_seconds', 'steps', 'stop_seconds', *COMPONENT_FIELDS)}, 'losses': LOSSES_NOTE, 'precondition': PRECONDITION}
    if decay is not None:
        head['decay_groups'] = decay                             # the groups this run trained with (policy, rule hash, assignment hash, sizes); no key under 'all'
    if budget:
        head['budget_end'] = True                                # no key for any other end: the head of every earlier run is unchanged (S4b preparation M8, decision D10 P7)
    return head


def _component_clock(device):
    """해석: CUDA 작업 완료를 기다린 host 시각이며 CPU에서는 원래 시계만 읽는다."""
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    return _clock()


def _result(model, loop: dict, folder: Path, meta: dict, acct: dict, call: dict, checkpoint_writes=()) -> TrainResult:
    return TrainResult(model=model, epochs_run=int(loop['epoch']), best_epoch=int(loop['best_epoch']), stopped_early=bool(loop['stopped_early']),
                       fit_batch_mean_losses=list(loop['fit_batch_mean_losses']), stop_losses=list(loop['stop_losses']), lrs=list(loop['lrs']), seconds=float(acct['seconds']),
                       epoch_seconds=list(loop['epoch_seconds']), load_seconds=list(loop['load_seconds']), step_seconds=list(loop['step_seconds']), steps=list(loop['steps']),
                       stop_seconds=list(loop['stop_seconds']), memory_metric=str(acct['metric']), peak_memory_bytes=int(acct['peak_bytes']), peak_upper_bytes=acct['peak_upper_bytes'],
                       peak_rss_bytes=int(acct['peak_rss_bytes']), checkpoint_bytes=int(meta['checkpoint_bytes']), checkpoint_dir=Path(folder), head=meta, resource=call,
                       timing_components={'epochs': {k: list(loop[k]) for k in COMPONENT_FIELDS},
                                          'checkpoint_io_current_call': list(checkpoint_writes),
                                          'checkpoint_io_scope': 'current call only; previous-attempt writes unavailable',
                                          'component_scope': 'host-clock synchronized CUDA spans; forward includes loss; excludes LR/zero_grad/bookkeeping'})


def _run(cfg: TrainConfig, data: TrainData, out_dir: Path, device, epochs: int, check, seed: int, res: _Resources, ctx: _Call) -> TrainResult:
    check()                                                      # E6 for the first epoch of this call (or for a finished run): before any preparation
    _check_inputs(cfg, data, epochs, seed)
    if torch.get_default_dtype() != torch.float32:
        raise ValueError(f'the default dtype must be torch.float32 (the reported runs are float32; another default is another run), not {torch.get_default_dtype()}')
    _refuse_if_failed(out_dir)
    dev = torch.device(device)
    checkpoint_writes = []
    deterministic_cpu(seed=int(seed)) if dev.type == 'cpu' else _deterministic_gpu(int(seed))
    static = np.ascontiguousarray(data.static, dtype=np.float32)
    identity = _identity(cfg, data, static, int(epochs), int(seed), _store_json_sha(data), dev)
    identity_sha = _sha(json.dumps(identity, sort_keys=True).encode())
    found, base = _latest_checkpoint(out_dir), None
    if found is not None:
        _check_identity(found[1]['identity'], identity)
        res.epoch_done, res.head_sha = int(found[1]['epoch']), _sha((found[0] / 'head.json').read_bytes())
        base = _head_accounting(found[1])
    finals, orphan = _load_calls(out_dir, identity_sha)
    if orphan is not None:
        finals.append(_adopt_orphan(out_dir, orphan, finals, base, res, identity_sha))
    prior = _merge_accounting(base, finals[-1] if finals else None)
    if prior is not None:
        res.adopt(prior)
    finished = found is not None and bool(found[1]['done'])
    best_stop = cfg.on_wall_cap == 'best_stop'
    res.wall_soft = best_stop and finished                       # a run that has ended is read, not charged again: over the cap by the epoch that reached it is its normal state under 'best_stop'
    res.guard()                                                  # a run that has not ended and is over the cap already is refused (no clean end behind it), under both settings
    res.wall_soft = best_stop                                    # from here the seconds are acted on at the end of an epoch (TRN-11), not between batches
    if not finished:
        ctx.begin(identity_sha, res)                             # from here the call has a pending marker: it ends finalized, or a later call adopts it
    zone = {} if cfg.zone_mark else {'zone_mark': False}        # the zone mark is taken off the model, the blocks and the STOP value only when asked: the default calls are those of every earlier run
    if cfg.model_spec is None:
        model = heads.EPT(cfg.encoder, static, **zone).to(dev)
    else:
        from c_ept.comparison.models import build_model, configure_float32
        configure_float32()
        model = build_model(cfg.encoder, static, cfg.model_spec, **zone).to(dev)
    optimizer = _optimizer(model, cfg)
    decay_record = _decay_record(model, cfg)
    if found is not None and decay_record is not None and found[1].get('decay_groups') != decay_record:
        # the identity holds the policy and the hash of the rule table; the head holds what the rule gave to the parameters of this model. Another assignment (the same table read
        # in another way) means that the groups of the saved optimizer state are not these groups: refused before the state is loaded into them
        raise ValueError('the weight-decay groups of this call (the decay_groups assignment of the parameters) are not those of the checkpoint; use another out_dir for another run')
    static_record = {'function': STATIC_FUNCTION, 'shape': list(static.shape), 'channels': list(encoder.STATIC_CHANNELS), 'sha256': identity['static_sha256'],
                     'source': None if data.static_source is None else json.loads(json.dumps(dict(data.static_source)))}
    steps_per_epoch = len(data.fit)
    total_steps = int(epochs) * steps_per_epoch

    law = {} if cfg.kill_law == 'poisson' else {'kill_law': cfg.kill_law}    # the plain law is not passed: the calls are those of every earlier run
    smooth = {} if cfg.terrain_smooth_weight == 0 else {'smooth_weight': float(cfg.terrain_smooth_weight)}      # λ_Ω of (LOSS-5): not passed under 0, so the step is the call of every earlier run
    if found is None:
        biases, bias_info = fit_biases(data.fit.rows, **law)
        _apply_biases(model, biases)
        loop = {'epoch': 0, 'best_value': math.inf, 'best_epoch': -1, 'best': None, 'bad': 0, 'stopped_early': False, 'bias_init': json.loads(json.dumps(bias_info)),
                **{k: [] for k in ('fit_batch_mean_losses', 'stop_losses', 'lrs', 'epoch_seconds', 'load_seconds', 'step_seconds', 'steps', 'stop_seconds', *COMPONENT_FIELDS)}}
        if cfg.decay_groups != 'all':
            loop['kappa'] = []                                   # κ at the end of each epoch (TRN-9, epoch_metrics); no such key under 'all': the last.pt is that of every earlier run
        if smooth:
            loop['terrain_penalty'] = []                         # the mean R_Ω of each epoch (EM-2, epoch_metrics); no such key under a weight of 0: the last.pt is that of every earlier run
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    else:
        last = torch.load(io.BytesIO(found[2]), weights_only=True, map_location='cpu')
        model.load_state_dict(last['model'])
        optimizer.load_state_dict(last['optimizer'])
        _set_rng_state(last['rng'])
        loop, best_state = last['loop'], torch.load(io.BytesIO(found[3]), weights_only=True, map_location='cpu')
        if loop['best'] != found[1]['best']:
            raise RuntimeError(f'checkpoint {found[0].name}: the best state in last.pt is not the one that head.json names; not guessing')
    if found is not None and (not finished or found[1]['schema'] in checkpoints.METRICS_SCHEMAS):
        # Legacy resume exports only real, complete restored loop history in a NEW epoch.
        restored_history = epoch_metrics.snapshot(loop, found[1])
        if found[1]['schema'] in checkpoints.METRICS_SCHEMAS and restored_history != epoch_metrics.read_snapshot(found[0], found[1]):
            raise RuntimeError('checkpoint epoch metrics disagree with cumulative loop; not guessing')
    res.boundary()                                               # the preparation and the restore are inside the measured scope

    fa = heads.FrameAveraged(model) if cfg.swap_mode == 'average' else heads.FrameAveraged(model, average=False)      # 'random_view' and 'none': one frame of every view (EH-10), the coins come with the batches
    evaluation = {} if cfg.eval_average else {'eval_average': False}      # the evaluation averages both frames unless asked not to: the default call is the call of every earlier run
    score = functools.partial(score_blocks, **law, **zone) if (law or zone) else score_blocks       # the plain law with the zone mark passes ``score_blocks`` itself
    preparation = batch_prep.Preparation(kill_law=cfg.kill_law, zone_mark=cfg.zone_mark, device=dev)       # the CPU work of a FIT batch: the frame, the data checks, the index lists, the packing (settings of this run)
    final_folder, final_meta, first = (found[0], found[1], loop['epoch']) if found is not None else (None, None, 0)
    for epoch in range(first, 0 if finished else int(epochs)):
        if epoch > first:
            check()                                              # E6 before the work of every later epoch
        res.boundary()
        began = _clock()
        model.train()
        fit_sum, load_s, step_s, n_steps, step = 0.0, 0.0, 0.0, 0, epoch * steps_per_epoch
        components = {key: 0.0 for key in COMPONENT_FIELDS}
        batches = iter(data.fit.epoch(epoch))
        while True:
            t0 = _clock()
            batch = next(batches, None)
            t1 = _clock()
            if batch is None:
                break
            load_s += t1 - t0
            res.guard()                                          # the resource guard between batches (not the E6 check)
            component_start = _component_clock(dev)
            # (SPD-1, S4b speed-ups D32 item 1): the frame of every view is applied on the CPU batch, before the copy, so that the device runs the model and not the swap (the model reads the
            # same tensors bit for bit: every operation of the frame is exact and uses no random number). A batch without coins (the modes 'average' and 'none') comes back as it is.
            # (SPD-2), item 3: the data checks of the heads and of the loss are made here once, on the CPU, by the very functions that made them on the device (``batch_prep.validate_rows``), and the
            # index lists of the loss are made; the step then runs TRUSTED: it does not read the range or the finiteness of an input back from the device, and the model checks are read once before backward.
            # (SPD-5), item 4: a device that copies gets the batch as one packed pinned buffer for each dtype (one ``non_blocking`` copy for each, the tensors as views of the copies); the CPU copies nothing.
            batch = batch_prep.to_device(preparation(batch), dev)
            components['transfer_seconds'] += _component_clock(dev) - component_start
            lr = lr_at(step, total_steps, cfg.warmup_steps, cfg.lr)
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            guards = step_guard.StepGuards()
            with step_guard.trust(guards):
                total = training_step.optimize_batch(fa, batch, optimizer, data.scales, step, components,
                                                     score=score, clock=lambda: _component_clock(dev), guards=guards, **smooth)
            loop['lrs'].append(lr)
            fit_sum += float(total.detach())
            step += 1
            n_steps += 1
            step_s += _clock() - t1
        t3 = _clock()
        stop_value = evaluate_stop(model, data.stop, data.scales, dev, guard=res.guard, **law, **evaluation, **zone)
        stop_s = _clock() - t3
        if not math.isfinite(stop_value):
            raise FloatingPointError(f'the STOP loss of epoch {epoch} is not finite')
        for key, value in (('fit_batch_mean_losses', fit_sum / steps_per_epoch), ('stop_losses', stop_value), ('epoch_seconds', _clock() - began), ('load_seconds', load_s),
                           ('step_seconds', step_s), ('steps', n_steps), ('stop_seconds', stop_s)):
            loop[key].append(value)
        for key in COMPONENT_FIELDS:
            loop[key].append(components[key])
        if cfg.decay_groups != 'all':
            loop['kappa'].append(_kappa_record(model))           # (TRN-9): the decay scales after the last step of this epoch (the model of the epoch, not the best state)
        if smooth:
            loop['terrain_penalty'].append(components['terrain_penalty'] / n_steps)      # (EM-2): the mean over the steps of this epoch of the unweighted R_Ω; it is not in ``stop_value``, which was measured above
        improved = stop_value < loop['best_value']               # (TRN-3): a strict improvement
        if improved:
            loop['best_value'], loop['best_epoch'], loop['bad'] = stop_value, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            loop['bad'] += 1
        loop['stopped_early'] = loop['bad'] >= cfg.patience
        loop['epoch'] = epoch + 1
        # (TRN-11): the wall cap is reached and no other rule ends the run here, so this epoch (finished, its STOP value measured, its best state kept) is the last one: the checkpoint of an epoch is the unit of
        # the ledger and the best STOP state, so the run is not cut between batches. The seconds are read before the write of this checkpoint (its own write is charged to the call as always).
        budget = best_stop and not loop['stopped_early'] and loop['epoch'] < int(epochs) and res.seconds() >= WALL_CAP_SECONDS
        done = loop['stopped_early'] or loop['epoch'] == int(epochs) or budget
        files = {}
        if improved:                                             # best.pt is written only when the best changed; head.json names the folder that holds it
            files['best.pt'] = _torch_bytes(best_state)
            loop['best'] = {'folder': f'epoch_{loop["epoch"]:04d}', 'sha256': _sha(files['best.pt']), 'bytes': len(files['best.pt'])}
        files['last.pt'] = _torch_bytes({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'rng': _rng_state(dev), 'loop': loop})
        res.boundary()                                           # the serialised state is in memory: a bound broken here stops before the write
        head = _head_record(loop, identity, static_record, int(epochs), int(seed), done, res.snapshot(), res.sampler.samples, decay_record, budget)
        head['checkpoint_bytes'] = len(files['last.pt']) + loop['best']['bytes']
        files.update(epoch_metrics.encode(loop, head))
        checkpoint_start = _component_clock(dev)
        final_folder, final_meta, res.head_sha = _save_checkpoint(out_dir, loop['epoch'], files, head)
        checkpoint_writes.append({'epoch': loop['epoch'], 'seconds': _component_clock(dev) - checkpoint_start})
        res.epoch_done = loop['epoch']
        res.boundary()                                           # the write is inside the measured scope: a bound broken by it is stored and the checkpoint is kept
        if loop['stopped_early'] or budget:                      # (TRN-3): no epoch after the patience ran out; (TRN-11): none after the epoch that reached the wall cap
            break
    model.load_state_dict(best_state)                            # the best state goes into the model that trained (no second model); the last state is in last.pt
    model.eval()
    res.boundary()
    if finished:
        return _result(model, loop, final_folder, final_meta, prior, finals[-1] if finals else {}, checkpoint_writes)       # nothing is added to the accounting of a run that was already finished
    call = ctx.finish(res, 'completed', None)                    # the accounting cutoff: the snapshot excludes the write of this marker
    res.sampler.sample()                                         # the sampler and the guard stay active through that write
    reason = call['reason'] if call['status'] != 'ok' else res.violation()
    if reason:
        _record_failure(out_dir, res, reason)
        raise ResourceBoundError(reason, record=False)
    return _result(model, loop, final_folder, final_meta, call, call, checkpoint_writes)
