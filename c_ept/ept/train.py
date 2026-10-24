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

가정과 선택

    · STOP 값 (R-S4a-16, 수정): 평가 행은 점수 있는 항목의 평균이고(M3Gold의 팀 항목도 평균이어서 B0 척도와 같은 항목 단위이다), 경기는 규모 층 안에서 1/n_m로 가중하며, 층 평균은 전문가 셋(``SIZE_NORMALISED``)만 낸다.
      배치 평균의 평균이 아니어서 배치를 어떻게 자르든 값이 같고, 학습 손실과 비교할 수 없으며 새로운 B0 상대 주장이 아니다. 학습 행 가중은 같다 (계획 P9).
    · 출력 편향은 FIT 주변 분포에서 시작한다. 표지 머리의 스칼라 하나는 소프트맥스에서 상쇄되므로 바꾸지 않는다 (``fit_biases``).
    · 보고 값은 CPU 결정적 float32 실행이다 (계획 Q3). 이어하기 정체성은 설정, 입력 행, S2 ``STORE.json``의 해시 (R-S4a-18, 수정), 장치·dtype·환경·torch 버전에 묶이고 코드 버전은 아니다 (``PRECONDITION``).
    · 상한 (R43, R-S4a-17): 표본 최댓값은 40 GiB 미만(같은 값도 거부), 이어하기 구간을 합한 시간은 24 h 이하(정확히 24 h는 허용). 측정 없음(0 바이트)은 측정된 0이 아니라 거부이다. 호출의 회계는 마지막 표지를 쓰기 직전의 스냅샷이며, 표지 자신의 쓰기와 호출 전체의 시간·최댓값은 과제 7이 밖에서 잰다.

입력과 출력

    train(cfg, data, out_dir, *, device, epochs, check, seed) → TrainResult      evaluate_stop(model, stop, scales, device) → float       fit_biases(rows) → (편향, 기록)


구현 경계: 체크포인트 IO는 checkpoints, 한 배치의 계산은 training_step, 자원 측정은 c_ept.runtime.resources에 위임한다. 에폭·조기 종료·재개·회계 조정은 이 모듈이 맡는다.

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

from c_ept import terrain, runtime
from c_ept.ept import data as ept_data
from c_ept.ept import encoder, heads, loss, checkpoints, training_step

# ---- the operational constants (every one has a name; the contract gives the reasons) ----------------------------------------------------
PATIENCE = 2                                     # epochs without a new best STOP loss that end the run
THREADS = 8                                      # the threads of the reported CPU runs
CATEGORY_PSEUDOCOUNT = 0.5                       # ζ₀: the pseudocount of a categorical start bias
RATE_FLOOR = 1e-6                                # λ_min: the lowest rate that a start bias may encode
MEMORY_CAP_BYTES = 40 * 2 ** 30                  # the sampled peak is strictly under this: the value itself is refused
WALL_CAP_SECONDS = 24 * 3600                     # the seconds of all calls of a run: exactly this is allowed, more is refused
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

    해석: 가중치 감쇠는 모든 매개변수에 같은 값으로 걸고 기울기 자르기는 하지 않는다 (계획에 없는 선택이며 보고서에 적는다).
    """
    encoder: encoder.EncoderConfig
    lr: float
    weight_decay: float
    warmup_steps: int
    patience: int = PATIENCE

    def __post_init__(self):
        if not isinstance(self.encoder, encoder.EncoderConfig):
            raise ValueError('encoder must be an EncoderConfig')
        for name in ('lr', 'weight_decay'):
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
    (one for each optimizer step). Per epoch: ``epoch_seconds`` and its parts ``load_seconds`` (inside ``next()`` of the FIT batches), ``step_seconds`` (forward, backward, optimizer
    step) over ``steps`` steps, and ``stop_seconds``. v5 ``timing_components`` records persisted forward/loss, backward, optimizer and transfer spans, plus current-call checkpoint IO only (old-attempt IO unavailable). ``seconds`` and the memory (``memory_metric``, the sampled ``peak_memory_bytes``, the Windows lifetime upper bound
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
def score_blocks(fa, enc, batch) -> dict:
    """배치 내 행이 존재하는 블록들에 대해 각 머리 출력과 음의 로그우도(NLL)를 계산하여 {block_key: (head, target, nll)} 사전을 반환한다.

    매개변수:
    - fa (FrameAveraged): 프레임 평균화된 머리 래퍼.
    - enc: 배치 전체에 대해 1회 생성되어 공유되는 부호화 상태.
    - batch: 학습 또는 검증 데이터 배치.

    반환값:
    - out (dict): 유효한 블록별 머리 이름, 대상 데이터 딕셔너리, loss.head_nll 결과 튜플의 매핑.

    동작 및 최적화:
    - M2Mark와 M2Zone은 유효한 마스킹 행들의 고유 뷰(torch.unique) 합집합에 대해 fa.m2를 단 1회 호출하여 계산 결과를 공유한다 (R-S4a-20).
    - M2Rate 블록은 callable인 fa.m2_rate가 있으면 학습/평가 모두 그 API를 호출한다. 이 API가 학습 시 전체 M2, 평가 시 rate 전용 경로를 선택한다. API가 없거나 callable이 아니면 fa.m2(...).rate로 호환한다. API 호출 중 예외를 숨기고 fallback하지 않는다.
    - 행이 없는 빈 블록(len(y) == 0)은 연산에서 제외된다."""
    blocks = {key: ept_data.block(batch, *key) for key in ept_data.ROW_FIELDS}
    pos = {h: blocks[(h, 'masked')]['pos'] for h in ('M2Mark', 'M2Zone') if len(blocks[(h, 'masked')]['y'])}
    shared = {}
    if pos:
        views, inverse = torch.unique(torch.cat(list(pos.values())), return_inverse=True)
        m2, start = fa.m2(enc, 1.0, pos=views), 0
        for h, p in pos.items():
            shared[h] = (m2.mark_logp if h == 'M2Mark' else m2.zone_logp).index_select(0, inverse[start:start + len(p)])
            start += len(p)
    out = {}
    for head, kind in ept_data.ROW_FIELDS:
        b = blocks[(head, kind)]
        if len(b['y']) == 0:
            continue
        q, m, z = b.get('q'), b.get('m'), b.get('z')
        if head in shared:
            value = shared[head]
        elif head == 'M2Rate':
            # 평가 전용 생략은 FrameAveraged 소유다. m2만 가진 기존 mock도 받는다 (TRN-7, EH-1).
            rate = getattr(fa, 'm2_rate', None)
            value = rate(enc, b['e'], pos=b['pos']) if callable(rate) else fa.m2(enc, b['e'], pos=b['pos']).rate
        elif head in ('M3Size', 'M3Kills', 'M3Gold'):
            m3 = fa.m3(enc, q, m, z, pos=b['pos'])
            value = {'M3Size': m3.size_logp, 'M3Kills': m3.kill_rate, 'M3Gold': m3.gold_logp}[head]
        elif head == 'M3Obj':
            value = fa.m3_obj(enc, q, m, z, b['s'], b['K'], pos=b['pos'])
        else:
            value = fa.m4(enc, q, m, z, b['K'], b['O'], b['e'], pos=b['pos'])
        out[loss.block_key(head, kind)] = (head, b, loss.head_nll(head, value, b))
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

    def value(self, scales: Mapping) -> float:
        """``Loss_STOP`` (TRN-2) of everything added: the blocks in the order of ``loss.BLOCKS``, each divided by its scale and added. ValueError for a missing or non-positive
        scale of a block that has rows, and when no block has a scored row."""
        total = None
        for key in loss.BLOCKS:
            book = self._groups.get(key)
            if not book:
                continue
            scale = float(scales.get(key, float('nan')))
            if not (math.isfinite(scale) and scale > 0):
                raise ValueError(f'the scale of the block {key} must be finite and positive, not {scales.get(key)!r}')
            strata: dict = {}
            for (u, _), (row_sum, count, s) in book.items():
                cell = strata.setdefault(s, {}).setdefault(u, [0.0, 0])
                cell[0] += row_sum / count
                cell[1] += 1
            means = [sum(t / n for t, n in strata[s].values()) / len(strata[s]) for s in sorted(strata)]
            term = (sum(means) / len(means)) / scale
            total = term if total is None else total + term
        if total is None:
            raise ValueError('no block has a scored row: there is nothing to average')
        return total


def evaluate_stop(model, stop, scales: Mapping, device='cpu', guard=None) -> float:
    """stop 데이터셋의 전체 배치(epoch 0)에 대해 검증 손실 Loss_STOP(TRN-2)을 측정하여 반환한다.

    매개변수:
    - model: 검증할 EPT 모델.
    - stop (Batches): 조기 종료 검증용 배치 컬렉션 (epoch 0만 순회).
    - scales (Mapping): 손실 정규화에 쓰는 B0 척도 매핑.
    - device: 연산 대상 디바이스 (기본값 'cpu').
    - guard (선택): 각 배치 평가 직전에 호출되는 인자 없는 감시 콜백 함수.

    반환값:
    - loss (float): StopMeter로 집계된 가중치 적용 손실 값.

    모드 제어 및 부작용 보장:
    - 호출 전 model.training 상태를 기록한 뒤 model.eval()로 전환하고, FrameAveraged를 eval 모드로 래핑하며 torch.no_grad() 환경에서 실행한다.
    - 기본 EPT 평가 경로는 드롭아웃을 끄고 난수 추출을 하지 않는다. 사용자 주입 guard나 배치 공급자의 부작용까지 보장하지는 않는다. 평가용 프레임 평균의 팀 교환 불변성 계약을 따른다.
    - 예외 발생 여부와 무관하게 finally 블록을 통해 model.train(was_training)으로 최상위 training 상태를 복원한다. 하위 모듈마다 달랐던 혼합 모드를 개별 복원하지는 않는다.
    loss.head_nll과 loss.scored_rows를 블록별로 계산하고 경기/행 식별자 및 해당되는 규모를 StopMeter에 전달한다. 최상위 모드 전환과 래퍼 생성은 try 진입 이전에 수행된다."""
    device = torch.device(device)
    was_training = model.training
    model.eval()
    fa, meter = heads.FrameAveraged(model).eval(), StopMeter()
    try:
        with torch.no_grad():
            for batch in stop.epoch(0):
                if guard is not None:
                    guard()
                batch = _to(batch, device)
                enc = fa.encode(batch)
                for key, (head, target, nll) in score_blocks(fa, enc, batch).items():
                    ok = loss.scored_rows(head, target).cpu().numpy()
                    size = target['s'].cpu().numpy() if head in loss.SIZE_NORMALISED else None
                    meter.add(key, nll.cpu().numpy(), ok, target['match'].cpu().numpy(), target['row'].cpu().numpy(), size)
    finally:
        model.train(was_training)
    return meter.value(scales)


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


def fit_biases(rows) -> tuple:
    """``(biases, info)`` for the output layers of the heads from the FIT rows ``rows`` (an ``EngagementRows``; ValueError if its role is not FIT): ``biases`` maps the name of a
    bias (relative to ``model.heads``) to a float64 array of the shape of the bias, ``info`` records the rows used and whether a fallback was taken. The mark head's shared
    scalar is not in ``biases`` and ``info`` says why. The two views of a head are pooled.

    수식
    ----
    (TRN-4)  γ_c = log[(n_c + ζ₀) / (N + C ζ₀)]            (M2Zone C = 10, M3Size C = 4, M3Gold C = 31)
    (TRN-5)  γ = log max(Σ_i y_i / Σ_i e_i, λ_min)         (M2Rate: 관측한 칸의 합, 한 개의 스칼라; M4Conv: 칸 c마다 Σ y_{i,c} / Σ e_i, e_i > 0인 행;
             M3Obj: 칸 c마다 Σ y_{i,c} / N, 노출 1;  M3Kills: (규모 s, 팀 t)마다 Σ_{i: s_i = s} y_{i,t} / N_s, 없는 규모는 팀의 전체 평균, 그것도 없으면 0)

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

    근거
    ----
    인계 5.2 (출력 편향을 FIT 주변 분포로), 계획 P4·P10 (칸별 포아송, 노출), 판정 R14·R22. Ref: Krichevsky and Trofimov 1981 (가상 개수; 이 정책의 근거가 아니라 서지 정보).
    """
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


def _violation(peak: int, metric: str, blind: bool, seconds: float):
    """The reason (a sentence) that a bound is broken, or None: the memory was not measured, the peak is not strictly under 40 GiB (R43), or the seconds are above 24 h."""
    if blind or metric == 'unavailable':
        return 'could not measure the memory (a sample of 0 bytes is not a measured 0): refusing to run unmeasured'
    if peak >= MEMORY_CAP_BYTES:
        return f'the sampled {metric} peak {peak / 2 ** 30:.2f} GiB is not under the 40 GiB bound (R43)'
    if seconds > WALL_CAP_SECONDS:
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
        return _violation(self.peak(), self.track['metric'], self.track['blind'], self.seconds())

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
        found = _violation(snap['peak_bytes'], snap['metric'], outcome == 'unmeasured', snap['seconds'])
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
      η_t, ϑ          | 학습률 (TRN-1), 가중치 감쇠                     | —      | ≥ 0      | lr_at, cfg.weight_decay
      θ, ∇L_t         | 매개변수, FIT 배치의 총손실 (LOSS-4)의 기울기   | —      | 실수     | model, total.backward
      A₁, A₂          | 기울기와 그 제곱의 지수이동평균 (AdamW 상태)    | —      | 실수     | 최적화기 상태
      β₁, β₂, ε_A     | AdamW 기본값                                    | —      | —        | torch.optim.AdamW

    해석
    ----
    기울기는 FIT 배치에서만 나오고 STOP은 최선 에폭과 멈춤만 정하므로, STOP이 달라져도 앞 에폭들의 ``last.pt``는 비트 같다.
    이어하기는 마지막 에폭의 모형·최적화기·난수 상태를 읽어 드롭아웃 마스크까지 끊지 않은 실행과 같고, 체크포인트는 파일을 쓴 뒤 ``head.json``을 마지막에 써서 끊긴 쓰기를 알아본다.
    최선 상태는 개선된 에폭의 폴더에만 쓰고 ``head.json``이 그 폴더를 가리키며, 끊긴 폴더를 다시 쓰면 그 안의 파일이 바뀐다.
    ``fit_batch_mean_losses``는 학습 모드 배치 손실의 에폭 평균이며 자료 전체의 손실도 평가도 아니다.

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


def _head_record(loop: dict, identity: dict, static_record: dict, epochs: int, seed: int, done: bool, snap: dict, samples: int) -> dict:
    return {'schema': SCHEMA, 'epoch': loop['epoch'], 'epochs': epochs, 'seed': seed, 'done': done, 'stopped_early': loop['stopped_early'], 'best_epoch': loop['best_epoch'],
            'best': loop['best'], 'identity': identity, 'static': static_record, 'bias_init': loop['bias_init'], 'threads': torch.get_num_threads(), 'seconds': snap['seconds'],
            'caps': {'memory_bytes': MEMORY_CAP_BYTES, 'wall_seconds': WALL_CAP_SECONDS},
            'memory': {'metric': snap['metric'], 'peak_bytes': snap['peak_bytes'], 'peak_upper_bytes': snap['peak_upper_bytes'], 'peak_rss_bytes': snap['peak_rss_bytes'],
                       'sampled': True, 'interval_s': MEMORY_INTERVAL_S, 'samples': samples, 'scope': MEMORY_SCOPE},
            'timing': {k: list(loop[k]) for k in ('load_seconds', 'step_seconds', 'steps', 'stop_seconds', *COMPONENT_FIELDS)}, 'losses': LOSSES_NOTE, 'precondition': PRECONDITION}


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
    res.guard()
    finished = found is not None and bool(found[1]['done'])
    if not finished:
        ctx.begin(identity_sha, res)                             # from here the call has a pending marker: it ends finalized, or a later call adopts it
    model = heads.EPT(cfg.encoder, static).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    static_record = {'function': STATIC_FUNCTION, 'shape': list(static.shape), 'channels': list(encoder.STATIC_CHANNELS), 'sha256': identity['static_sha256'],
                     'source': None if data.static_source is None else json.loads(json.dumps(dict(data.static_source)))}
    steps_per_epoch = len(data.fit)
    total_steps = int(epochs) * steps_per_epoch

    if found is None:
        biases, bias_info = fit_biases(data.fit.rows)
        _apply_biases(model, biases)
        loop = {'epoch': 0, 'best_value': math.inf, 'best_epoch': -1, 'best': None, 'bad': 0, 'stopped_early': False, 'bias_init': json.loads(json.dumps(bias_info)),
                **{k: [] for k in ('fit_batch_mean_losses', 'stop_losses', 'lrs', 'epoch_seconds', 'load_seconds', 'step_seconds', 'steps', 'stop_seconds', *COMPONENT_FIELDS)}}
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    else:
        last = torch.load(io.BytesIO(found[2]), weights_only=True, map_location='cpu')
        model.load_state_dict(last['model'])
        optimizer.load_state_dict(last['optimizer'])
        _set_rng_state(last['rng'])
        loop, best_state = last['loop'], torch.load(io.BytesIO(found[3]), weights_only=True, map_location='cpu')
        if loop['best'] != found[1]['best']:
            raise RuntimeError(f'checkpoint {found[0].name}: the best state in last.pt is not the one that head.json names; not guessing')
    res.boundary()                                               # the preparation and the restore are inside the measured scope

    fa = heads.FrameAveraged(model)
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
            batch = _to(batch, dev)
            components['transfer_seconds'] += _component_clock(dev) - component_start
            lr = lr_at(step, total_steps, cfg.warmup_steps, cfg.lr)
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            total = training_step.optimize_batch(fa, batch, optimizer, data.scales, step, components,
                                                 score=score_blocks, clock=lambda: _component_clock(dev))
            loop['lrs'].append(lr)
            fit_sum += float(total.detach())
            step += 1
            n_steps += 1
            step_s += _clock() - t1
        t3 = _clock()
        stop_value = evaluate_stop(model, data.stop, data.scales, dev, guard=res.guard)
        stop_s = _clock() - t3
        if not math.isfinite(stop_value):
            raise FloatingPointError(f'the STOP loss of epoch {epoch} is not finite')
        for key, value in (('fit_batch_mean_losses', fit_sum / steps_per_epoch), ('stop_losses', stop_value), ('epoch_seconds', _clock() - began), ('load_seconds', load_s),
                           ('step_seconds', step_s), ('steps', n_steps), ('stop_seconds', stop_s)):
            loop[key].append(value)
        for key in COMPONENT_FIELDS:
            loop[key].append(components[key])
        improved = stop_value < loop['best_value']               # (TRN-3): a strict improvement
        if improved:
            loop['best_value'], loop['best_epoch'], loop['bad'] = stop_value, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            loop['bad'] += 1
        loop['stopped_early'] = loop['bad'] >= cfg.patience
        loop['epoch'] = epoch + 1
        done = loop['stopped_early'] or loop['epoch'] == int(epochs)
        files = {}
        if improved:                                             # best.pt is written only when the best changed; head.json names the folder that holds it
            files['best.pt'] = _torch_bytes(best_state)
            loop['best'] = {'folder': f'epoch_{loop["epoch"]:04d}', 'sha256': _sha(files['best.pt']), 'bytes': len(files['best.pt'])}
        files['last.pt'] = _torch_bytes({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'rng': _rng_state(dev), 'loop': loop})
        res.boundary()                                           # the serialised state is in memory: a bound broken here stops before the write
        head = _head_record(loop, identity, static_record, int(epochs), int(seed), done, res.snapshot(), res.sampler.samples)
        head['checkpoint_bytes'] = len(files['last.pt']) + loop['best']['bytes']
        checkpoint_start = _component_clock(dev)
        final_folder, final_meta, res.head_sha = _save_checkpoint(out_dir, loop['epoch'], files, head)
        checkpoint_writes.append({'epoch': loop['epoch'], 'seconds': _component_clock(dev) - checkpoint_start})
        res.epoch_done = loop['epoch']
        res.boundary()                                           # the write is inside the measured scope: a bound broken by it is stored and the checkpoint is kept
        if loop['stopped_early']:                                # (TRN-3): no epoch after the patience ran out
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
