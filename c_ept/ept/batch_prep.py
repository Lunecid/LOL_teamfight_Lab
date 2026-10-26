"""The CPU side of a training step: what is done to a batch on the host, once, before it is moved to the device (EPT-T012, S4b speed-ups SPD, author decision D32).

A training step of the EPT is dominated by per-operation cost on the device (kernel launches, host synchronisations), not by arithmetic. Everything that can be decided from the data alone is
therefore decided here, on the CPU, where a batch is made, and the device only runs the model. This module grows item by item; each item changes no number:

    item 1  the frame of every view (``prepare_batch``): the coins of ``swap_mode='random_view'`` are applied on the CPU batch, so that the transfer carries the batch in its frames;
    item 3  the data checks, once per batch (``validate_rows``) and the index lists that replace the boolean masks of the loss (``build_plan``), made by ``Preparation`` and carried by a
            ``PreparedBatch``; the step that reads such a batch runs trusted (``step_guard``): it does not read the range or the finiteness of an input back from the device.
    item 4  the transfer (``pack``, ``Packed``, ``to_device``): for a CUDA device the batch is laid in one pinned host buffer for each dtype and moved with one ``non_blocking`` copy for each buffer
            (about 130 separate copies before), the tensors being views of the copies; (SPD-5) in ``pack``. For the CPU there is nothing to copy and nothing is packed.

방법. ``Batches`` (``data``) makes a batch as it was recorded, and, in the mode 'random_view', one coin for each distinct view (``swap_view``, plan Q3). ``FrameAveraged.encode`` used to
apply the coins itself, on the device, with ``data.swap_views``: about a thousand small operations and two hundred host synchronisations of the device queue for each step (toy count of the
audit: 1,157 operations, 220 synchronisations for a batch of 102 views). ``prepare_batch`` applies the same function to the CPU batch, before the copy. In the same way the heads and the loss
used to check every input they were given on the device (``int(t.min())``, ``bool(torch.isfinite(t).all())``, the masks of the blocks): about 300 more synchronisations. ``validate_rows`` makes
those checks once, on the CPU, with the very functions that made them on the device, and ``build_plan`` makes the index lists once.

식 (식 번호는 ``(SPD-n)``; ``(DAT-n)`` 은 ``data``의 식)

    (SPD-1)  b' = F_c(b) = swap_views(b)  (b has the key swap_view) ;   b' = b  (b has none) ;   T(F_c(b)) = F_c(T(b))  bit for bit
    (SPD-2)  accept(b') ⇔ for every block h with rows:  heads-checks(h, b') ∧ head_nll(h, ŷ₀, b') defined ∧ total_loss(...) defined      with ŷ₀ a dummy output that passes every model check
    (SPD-3)  P_h = { i : row i of the block h counts } (ascending) ;   P_{h,k} = { i ∈ P_h : s_i = k }, k = 0 .. 3  (the three experts) ;   mean_h = mean( rows[P_h] ) = mean( rows[mask_h] )
    (SPD-4)  (V, J) = unique(cat of the views of the rows of M2Mark and M2Zone) with the inverse J of each row: the shared M2 call of ``train.score_blocks``

      기호    | 뜻                                                              | 단위 | 범위            | 코드 변수
      b       | 배치: 보기 텐서와 13개 블록의 행 필드 (기록된 틀)                | —    | 텐서의 사전      | batch
      c_v     | 보기 v의 동전 (1이면 맞바꾼 틀), (DAT-11)                        | —    | 0 또는 1        | batch['swap_view']
      F_c     | 동전대로 보기와 그 보기를 읽는 행을 함께 맞바꾸는 변환 (DAT-11)   | —    | —               | data.swap_views
      T       | 장치로의 복사 (CPU 배치를 장치 텐서로)                           | —    | —               | train._to
      b'      | 모형과 손실이 읽는 배치 (동전이 이미 적용되어 swap_view가 없다)   | —    | —               | 반환값
      ŷ₀      | 점검만을 위한 가짜 모형 출력 (율 1, 로그 법칙 −log C)             | —    | —               | _dummy_output
      P_h     | 블록 h에서 평균에 드는 행의 번호 (오름차순)                       | 행   | 0 .. n − 1      | plan 'counted'
      P_{h,k} | 그 가운데 규모 k의 행의 번호 (규모별 전문가 셋만)                  | 행   | 0 .. n − 1      | plan 'size0' .. 'size3'
      V, J    | 공유 M2 호출의 보기와 행마다의 위치                               | —    | 0 .. 보기 수 − 1 | plan 'plan/m2/zone*/views', 'inverse'

해석
----
(SPD-1): 동전을 장치에서 적용하든 CPU에서 먼저 적용하고 복사하든 모형이 보는 텐서는 비트까지 같다: ``F_c``가 하는 일은 모두 정확한 연산이어서(수집 ``index_select``와 표 찾기, 선택 ``torch.where``, 복사, 한 번의
IEEE 뺄셈 ``0.0 − x``·``W − x``·``1 − t``, 정수 나머지 ``(ξ + 5) mod 10``) CPU와 장치에서 같은 값을 내고, 복사는 값을 바꾸지 않으므로 ``T``와 ``F_c``의 순서를 바꿔도 같다. 동전은 배치를 만들 때 정해지고
(``Batches.assemble``) 이 함수는 난수를 쓰지 않으므로 난수 흐름과 드롭아웃 마스크도 그대로이다. 이미 틀이 적용된 배치(동전 키가 없다)에는 다시 불러도 같은 배치를 돌려준다.

(SPD-2): 자료 점검(범위, 유한성, 모양, 형)은 머리와 손실의 점검 함수 그대로 CPU 배치에 한 번 한다. 가짜 출력으로 ``head_nll``과 ``total_loss``를 불러 손실의 자료 점검(행 수, 범주 범위, 노출, 개수)을 같은 순서로
거치고, 머리의 점검(보기 번호, 조건 q·m·z, 규모, K·O, 전환 창)은 머리가 부르는 함수를 같은 순서로 부른다. 그래서 점검이 거부하는 것과 그 오류의 종류와 문구는 이전과 같다(시험이 칸마다 망가뜨려 견준다). 이 점검을
통과한 배치만 ``PreparedBatch``가 되고, 그 배치를 읽는 걸음만 신뢰된다(``step_guard.trust``). 두 곳이 동시에 잘못된 배치(자료 점검과 모형 점검)에서 어느 오류가 먼저 나오는지는 이전과 다를 수 있다(둘 다 같은 걸음에서
닫힌다).

(SPD-3): 손실은 평균에 드는 행만 불리언 마스크로 골라(``rows[mask]``) 평균했고, 마스크마다 장치에서 값을 읽어 와야(``nonzero``) 했다. 번호 목록 ``rows.index_select(0, P)``는 같은 원소를 같은 순서로 같은 모양의
연속 텐서에 모으므로 ``.mean()``과 기울기(같은 칸에 한 번씩 쓰는 합)가 비트까지 같다. 목록의 길이는 장치에서 읽지 않아도 아는 모양이다(블록에 점수 있는 행이 없으면 목록이 비어 있다).

(SPD-4): ``score_blocks``는 M2Mark·M2Zone이 읽는 보기의 합집합을 ``torch.unique``로 장치에서 구했다(길이가 자료에 달려 있어 장치를 기다린다). 같은 식을 CPU 배치에서 한 번 하고 결과를 싣는다.
구역 표지를 끈 실행(M2Zone을 점수 매기지 않는다)은 합집합이 달라서 열쇠에 설정이 들어 있다.

근거
----
저자 결정 D32 (수치를 바꾸지 않는 학습 속도 개선 1–4), 성능 감사 perf-audit.md 항목 1과 3 (장난감: 손실과 모든 기울기가 같다), 결정 D10 P1·계획 Q3 (동전의 단위와 출처).
"""
# Ref: docs/superpowers/plans/2026-10-26-c-ept-s4b-prep.md Task M7 (the random view swap: the coins of SeedSequence([seed, epoch, batch, 0x5157]), the rows of a view follow its frame).
# Ref: the performance audit of the S4b hot path (perf-audit.md, items 1, 3 and 4): the swap on the CPU batch before the transfer, the data checks once per batch on the CPU, one packed pinned copy for each dtype; toy checks: loss and every gradient equal.
from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from c_ept.data.head_rows import N_CELLS, N_K, N_Q, N_SIZE
from c_ept.ept import data, heads, loss, step_guard
from c_ept.models import killlaw

PLAN_PREFIX = 'plan/'                                   # the keys that ``build_plan`` adds to a batch (no field of a block and no view key starts so)


def m2_key(zone_mark: bool, name: str) -> str:
    """The key of the shared M2 call in the plan: ``name`` 'views' or 'inverse', for the setting ``zone_mark`` (the union of the views differs without the zone block)."""
    return f'plan/m2/zone{int(zone_mark)}/{name}'


def prepare_batch(batch: Mapping) -> Mapping:
    """The batch as the model reads it, made on the CPU: ``data.swap_views(batch)`` when the batch carries the coins ``swap_view`` of the mode 'random_view' (every view in the frame that its
    coin chose, the rows that read a swapped view swapped with it, ``swap_view`` consumed), else ``batch`` itself (the modes 'average' and 'none', the STOP batches, a batch that was framed
    already: the same object comes back). The input is never changed. ValueError, naming the key, for a ``swap_view`` or a block that ``data.swap_views`` refuses.

    수식
    ----
    (SPD-1)  b' = swap_views(b) (swap_view ∈ b),  b' = b (그 밖) ;  T(b') = b'의 복사 = F_c(T(b))   (``data``의 (DAT-11)과 같은 변환을 복사 앞에서 한다)

    해석
    ----
    장치에서 틀을 고르면 걸음마다 보기 수에 비례하는 작은 연산과 동기화가 장치 경로에 들어간다. CPU에서 먼저 하면 그 일은 장치 경로에서 사라지고, 모형이 읽는 텐서와 행의 조건·정답은 비트까지 같다
    (모든 연산이 정확한 연산이고 난수를 쓰지 않는다). 멱등이다: 틀이 적용된 배치에는 동전이 없으므로 다시 불러도 같은 객체를 돌려준다.
    """
    if 'swap_view' not in batch:
        return batch
    return data.swap_views(batch)


# ---- the data checks, once per batch ----------------------------------------------------------------------------------------------------------------
def _like() -> torch.Tensor:
    """The stand-in for the model output that the check functions take for the dtype and the device of what they convert to: float32 on the CPU (the heads' output is float32)."""
    return torch.zeros(1, dtype=torch.float32)


def _dummy_output(head: str, rows: int) -> torch.Tensor:
    """A model output of ``rows`` rows that passes every model check of ``head_nll`` (a rate of 1 for a Poisson head, the log of the uniform law for a category head), as a view without memory: the dry run
    of ``validate_rows`` needs a tensor of the right shape and dtype, not values."""
    tail = loss._OUT_TAIL[head]
    value = 1.0 if head in loss._COUNT_WIDTH else -math.log(tail[-1])
    return torch.full((), value, dtype=torch.float32).expand(rows, *tail)


def _heads_checks(head: str, b: Mapping, views: int, like: torch.Tensor) -> None:
    """The checks that the heads make of the rows of one block, in the order of the calls of ``score_blocks`` (``FrameAveraged.m3`` ... then ``EPTHeads.m3``: the second call repeats the first on what
    the first has cast): the view numbers of the rows, then the conditions q, m, z, the size, K, O and the window or the exposure."""
    pos = b['pos']
    if head not in ('M2Mark', 'M2Zone'):                  # these two read their views through the shared M2 call: the union of their views is checked once, in ``validate_rows``
        heads._view_index(pos, views, like)
    n = pos.shape[0]
    if head == 'M2Rate':
        heads._exposure(b['e'], n, N_Q, like)
    elif head in ('M3Size', 'M3Kills', 'M3Gold'):
        heads._reading(n, like, b.get('q'), b.get('m'), b.get('z'))
    elif head == 'M3Obj':
        heads._reading(n, like, b.get('q'), b.get('m'), b.get('z'))
        heads._whole('s', b['s'], n, 0, N_SIZE - 1, like)
        heads._counts('K', b['K'], n, N_K, like)
    elif head == 'M4Conv':
        heads._reading(n, like, b.get('q'), b.get('m'), b.get('z'))
        heads._counts('K', b['K'], n, N_K, like)
        heads._counts('O', b['O'], n, N_CELLS, like)
        heads._window(b['e'], n, like)


def m2_views(batch: Mapping, zone_mark: bool):
    """``(views, inverse)`` of the shared M2 call of ``train.score_blocks`` -- the distinct views of the rows of M2Mark (and of M2Zone when it is scored), in the order of ``torch.unique``, and the position of each
    row in them -- or None when neither block has a row. The expression of ``score_blocks``, on the CPU batch."""
    pos = [batch[data.row_key(h, 'masked', 'pos')] for h in ('M2Mark', 'M2Zone') if (h != 'M2Zone' or zone_mark) and len(batch[data.row_key(h, 'masked', 'y')])]
    if not pos:
        return None
    return torch.unique(torch.cat(pos), return_inverse=True)


def validate_rows(batch: Mapping, *, kill_law: str = 'poisson', zone_mark: bool = True) -> None:
    """Make on the CPU, once, every check of the data-derived inputs that the heads and the loss make of this batch on the device in one training step: ValueError or TypeError with the very message of the check
    that refuses, None when the batch is acceptable.

    The checks are those functions themselves, called in the order of the step: the shared M2 call (``heads._view_index`` of the distinct views), then for each block that has rows (the order of ``data.ROW_FIELDS``,
    M2Zone left out when ``zone_mark`` is False) the checks of the heads of that block (``_heads_checks``) and ``loss.head_nll`` on a dummy model output that passes the model checks (so every data check of the loss runs
    in its own order), and at the end ``loss.total_loss`` (the means, the sizes, the scales). The dry run is made outside any trusted scope.

    수식
    ----
    (SPD-2)  accept(b') ⇔ ∀ 블록 h (행이 있는): heads-checks(h, b') ∧ head_nll(h, ŷ₀, b') ∧ … ∧ total_loss(·)   (모형 점검은 ŷ₀가 통과한다)

    해석
    ----
    점검을 새로 쓰지 않고 같은 함수를 같은 순서로 부르므로 거부하는 것, 오류의 종류, 문구가 같다. 거울이 어긋나면(머리나 손실의 호출 순서가 바뀌면) 시험이 칸마다 값을 망가뜨려 장치 경로와 견주어 알려 준다.
    """
    killlaw.check_law(kill_law)
    law = {} if kill_law == 'poisson' else {'kill_law': kill_law}
    missing = [k for k in data.VIEW_KEYS if k != 'view' and k not in batch]
    if missing:
        raise ValueError(f'the batch lacks the view tensors {missing}')                                 # the text of ``EPTEncoder.encode``, which the step would have raised
    views, like = int(batch['ev_valid'].shape[0]), _like()                                              # the number of views the encoder encodes (``h_pool.shape[0]``)
    nlls, targets = {}, {}
    with step_guard.untrusted():
        shared = m2_views(batch, zone_mark)
        if shared is not None:
            heads._view_index(shared[0], views, like)
        for head, kind in data.ROW_FIELDS:
            if head == 'M2Zone' and not zone_mark:
                continue
            b = data.block(batch, head, kind)
            if len(b['y']) == 0:
                continue
            _heads_checks(head, b, views, like)
            key = loss.block_key(head, kind)
            nlls[key], targets[key] = loss.head_nll(head, _dummy_output(head, b['pos'].shape[0]), b, **law), b
        loss.total_loss(nlls, {key: 1.0 for key in nlls}, targets)


def build_plan(batch: Mapping, *, zone_mark: bool = True) -> dict:
    """``{key: int64 tensor}``, the index lists of the trusted step (module text, (SPD-3) and (SPD-4)): for every block that has rows and is scored, ``loss.counting_plan`` (the rows that count, and for the three
    experts the rows of each size), and for the shared M2 call its views and the position of each row (``m2_key``)."""
    plan = {}
    shared = m2_views(batch, zone_mark)
    if shared is not None:
        plan[m2_key(zone_mark, 'views')], plan[m2_key(zone_mark, 'inverse')] = shared
    for head, kind in data.ROW_FIELDS:
        if head == 'M2Zone' and not zone_mark:
            continue
        b = data.block(batch, head, kind)
        if len(b['y']):
            plan.update(loss.counting_plan(head, kind, b))
    return plan


# ---- item 4: the packed transfer ------------------------------------------------------------------------------------------------------------------------
ALIGN_BYTES = 512        # the alignment of every block of the caching allocator of the device: a slot starts where a tensor of its own would start


def _is_cuda(device) -> bool:
    return device is not None and torch.device(device).type == 'cuda'


def should_pack(device) -> bool:
    """True for a CUDA device: only there are about 130 small copies slower than a few large ones. The CPU copies nothing (``train._to`` returns the batch itself), so there is nothing to pack for it."""
    return _is_cuda(device)


class Packed:
    """A batch in a few buffers: ``buffers`` = ``{dtype: 1-d host tensor}`` (pinned when made for a CUDA device), ``slots`` = ``{key: (dtype, offset, numel, shape)}`` for the contiguous tensors that lie in them, ``loose`` =
    ``{key: tensor}`` for the others, ``order`` = the keys of the batch in their order. ``views()`` is the batch as host tensors (views of the buffers), ``to(device)`` the batch on the device: one copy for each buffer
    and the tensors as views of the copies (module text, (SPD-5)). Made by ``pack``."""
    __slots__ = ('buffers', 'slots', 'loose', 'order')

    def __init__(self, buffers: dict, slots: dict, loose: dict, order: list):
        self.buffers, self.slots, self.loose, self.order = buffers, slots, loose, order

    def _unpack(self, buffers: Mapping, loose) -> dict:
        out = {}
        for key in self.order:
            if key in self.slots:
                dtype, offset, numel, shape = self.slots[key]
                out[key] = buffers[dtype][offset:offset + numel].view(shape)
            else:
                out[key] = loose(self.loose[key])
        return out

    def views(self) -> dict:
        """The batch on the host: every tensor a view of its slot (the loose ones as they are)."""
        return self._unpack(self.buffers, lambda t: t)

    def to(self, device) -> dict:
        """The batch on ``device``: ONE ``Tensor.to`` for each buffer -- ``non_blocking`` when the buffer is pinned -- and a view of the copy for each tensor; a loose tensor is copied as ``train._to`` copies it.

        A copy from pinned memory returns before the data has arrived; it is ordered on the stream of the caller before everything that is queued after it (the model reads the tensors on that stream), and the
        caching host allocator does not hand the pinned buffer out again before the copy has finished, so the host batch may be dropped at once. A pageable buffer is copied the plain way."""
        moved = {dtype: buffer.to(device, non_blocking=buffer.is_pinned()) for dtype, buffer in self.buffers.items()}
        return self._unpack(moved, lambda t: t.to(device))


def pack(tensors: Mapping, *, pin: bool) -> Packed:
    """Lay the tensors of a batch in one buffer for each dtype, each in a slot that starts at a multiple of ``ALIGN_BYTES`` (an empty tensor takes no room). Only a contiguous strided tensor is laid in a buffer: it has
    the strides that its own copy would have; every other tensor stays loose and is copied alone, with the strides that ``train._to`` gives it, so that no kernel meets another layout. ``pin``: pinned buffers (CUDA).

    수식
    ----
    (SPD-5)  slot(k) = [o_k, o_k + n_k) in the buffer of dtype(k) ;   o_k = ALIGN · ⌈ (o_{k-1} + n_{k-1}) / ALIGN ⌉ (elements of ALIGN_BYTES / itemsize) ;   T(pack(b))[k] = view(T(buffer)[o_k : o_k + n_k], shape_k) = T(b[k])

      기호        | 뜻                                                | 단위   | 범위          | 코드 변수
      k, n_k      | 텐서의 열쇠, 그 원소 수                            | —, 개  | ≥ 0           | key, numel
      o_k         | 그 텐서의 칸이 시작하는 원소 번호                  | 개     | ALIGN의 배수  | offset
      ALIGN       | 칸 시작의 정렬 (512 바이트를 원소 수로 환산)       | 개     | 512/itemsize  | ALIGN_BYTES // element_size

    해석
    ----
    텐서마다 따로 복사하든 dtype별 한 덩어리로 복사해 덩어리 위의 보기로 쓰든 장치가 읽는 값은 같다(복사는 바이트를 옮길 뿐이다). 칸을 512바이트 배수에 두는 것은 캐싱 할당자가 텐서마다 주는 정렬과 같은 정렬에서
    커널이 시작하게 하려는 것이다(벡터화한 읽기의 선택이 주소 정렬에 달려 있을 수 있어서 정렬이 다르면 같은 값이 다른 경로로 계산될 위험이 있다). 연속이 아닌 텐서는 덩어리에 넣지 않고 이전처럼 따로 복사해서 걸음(stride)이
    바뀌지 않는다. 이 배치는 호스트에서 이미 CPU 텐서의 사전이므로 호스트 쪽 복사는 한 번 늘지만(생산 스레드에서) 장치로 가는 복사 수는 약 130에서 dtype 수(셋)로 줄어든다.
    """
    sizes, slots, loose = {}, {}, {}
    for key, t in tensors.items():
        if t.layout != torch.strided or not t.is_contiguous():
            loose[key] = t
            continue
        step = max(1, ALIGN_BYTES // t.element_size())
        offset = sizes.get(t.dtype, 0)
        slots[key] = (t.dtype, offset, t.numel(), tuple(t.shape))
        sizes[t.dtype] = offset + -(-t.numel() // step) * step
    buffers = {dtype: torch.zeros(size, dtype=dtype, pin_memory=pin) for dtype, size in sizes.items()}
    for key, (dtype, offset, numel, shape) in slots.items():
        buffers[dtype][offset:offset + numel].copy_(tensors[key].reshape(-1))
    return Packed(buffers, slots, loose, list(tensors))


def to_device(batch: Mapping, device) -> dict:
    """The batch on ``device``: a packed ``PreparedBatch`` by ``Packed.to`` (one copy for each dtype); any other batch as ``train._to`` moved it (the CPU: the batch itself; a device: one copy for each tensor)."""
    packed = getattr(batch, 'packed', None)
    if packed is not None:
        return packed.to(device)
    if torch.device(device).type == 'cpu':
        return batch
    return {k: v.to(device) for k, v in batch.items()}


class PreparedBatch(dict):
    """A training batch that is ready for a trusted step: a ``dict`` of CPU tensors (the batch in its frames, ``swap_view`` consumed, plus the index lists of ``build_plan`` under the keys ``plan/...``), made by
    ``Preparation`` after the data checks passed, the ``settings`` it was prepared for and, when it was packed for the transfer (item 4), its ``packed`` buffers (the tensors of the dict are views of them).
    Do not give it to ``data.swap`` (the plan keys are no fields of a batch)."""
    __slots__ = ('settings', 'packed')

    def __init__(self, tensors: Mapping, settings: Mapping, packed: Packed | None = None):
        super().__init__(tensors)
        self.settings = dict(settings)
        self.packed = packed


class Preparation:
    """``Preparation(kill_law='poisson', zone_mark=True, pack=None, device=None)``: the CPU work of one training batch as a function (``prepare(batch) -> PreparedBatch``): the frame (``prepare_batch``), the data checks
    (``validate_rows``), the plan (``build_plan``) and, for a device that copies, the packing of the transfer (``pack``). The settings are those of the training that will read the batch (``TrainConfig.kill_law`` and
    ``zone_mark``); a ``PreparedBatch`` made for other settings is refused, not trusted. A ``PreparedBatch`` passes through unchanged. ValueError for a setting that is not valid, and for a batch that the checks
    refuse (the message of the check).

    ``device``: where the batch will be moved (None: nowhere yet). ``pack``: None = ``should_pack(device)`` (only a CUDA device), True or False = as asked (the proof and the tests pack on the CPU too). The buffers are
    pinned for a CUDA device."""

    def __init__(self, *, kill_law: str = 'poisson', zone_mark: bool = True, pack: bool | None = None, device=None):
        killlaw.check_law(kill_law)
        if not isinstance(zone_mark, bool):
            raise ValueError(f'zone_mark must be True or False, not {zone_mark!r}')
        self.settings = {'kill_law': kill_law, 'zone_mark': zone_mark}
        self.pack, self.device = pack, device

    @property
    def packed_for(self) -> bool:
        """Whether this preparation packs the transfer of the batches it makes."""
        return should_pack(self.device) if self.pack is None else bool(self.pack)

    def __call__(self, batch: Mapping) -> PreparedBatch:
        if isinstance(batch, PreparedBatch):
            if batch.settings != self.settings:
                raise ValueError(f'the batch was prepared for the settings {batch.settings} but the training asks for {self.settings}')
            return batch
        framed = prepare_batch(batch)
        validate_rows(framed, **self.settings)
        tensors = dict(framed)
        tensors.update(build_plan(framed, zone_mark=self.settings['zone_mark']))
        if not self.packed_for:
            return PreparedBatch(tensors, self.settings)
        packed = pack(tensors, pin=_is_cuda(self.device))                         # pinned only for a CUDA device: a packed batch on the CPU is for the proof and the tests
        return PreparedBatch(packed.views(), self.settings, packed)
