"""EPT의 가린 머리를 합성기의 수치 배열 규약에 연결한다.

킬이 하나 이상 난 교전의 프레임 평균 예측을 기존 Composer에 전달한다.
프레임 시계 보기만 한 번 부호화하며, 참 δ나 사건 보기는 합성 입력이 아니다.
통계는 FIT에서 고정한 Stats이고, 모형·통계·장치는 바인딩 동안 불변이어야 한다.
이 어댑터는 재바인딩을 검출하지만 매개변수의 임의 변경까지 검출하지 않는다.

식 목록
  (EC-1) handle_i = (i, bind_id): 로컬 행과 바인딩 세대의 정확한 float32 참조.
  (EC-2) λ_q = M2(노출 1), π = exp(log π), ρ = exp(log ρ).
  (EC-3) P(s), μ_{s,t}, p(b | s,t,ξ): 규모·팀·자기 팀 킬 층 선택.
  (EC-4) K' = (k_B,k_R,g_B/1000,g_R/1000): 목적물 조건의 단위 변환.
  (EC-5) r = M4(q,m,z,K',O,노출 1): 전환의 초당 발생률.
  (EC-6) C_c = H(Enc(X[v_{cB}..v_{(c+1)B−1}])), cache = concat_c C_c : 보기를 B개씩 묶어 부호화하고 머리가 읽는 텐서 H만 남긴다.
         H(E) = (h_tok[:, 0..9], h_pool, side) ;  원 틀과 맞바꾼 틀 둘 다(프레임 평균일 때).

      기호  | 뜻                                                         | 단위 | 범위          | 코드 변수
      B     | 한 묶음의 보기 수(평가 정체성에 기록하는 고정 상수)        | 보기 | 양의 정수 | chunk_views
      v_i   | 바인딩이 받은 i번째 보기(요청한 순서)                     | —    | S2 보기 번호(views/src = 0) | idx
      X[·]  | 그 보기들의 표준화한 입력 텐서                             | —    | 실수      | data.ViewTensors
      Enc   | 부호화기(원 틀과 맞바꾼 틀 각각의 호출)                   | —    | —         | frame.encode
      H     | 머리가 읽는 상태: 선수 토큰 10개, 풀링 상태, 진영 표시    | —    | 실수      | _head_inputs
      C_c   | 묶음 c의 캐시 조각                                         | —    | 실수      | parts

해석: 부호화기는 보기마다 독립이어서 B개씩 나눠 부호화해도 같은 보기의 상태는 같은 식의 값이다. 다만 묶음이 달라지면 같은 보기가 놓이는 자리와 한 번에 처리하는 행렬의 모양이 달라져
부동소수점 반올림이 마지막 자리 수준에서 달라질 수 있다(실제 규모 B = 256에서 상대 1.15e-7, 계획 Q6의 T011 측정; 장난감 모형 B = 1에서 5.7e-7, 이 과제의 시험; 문서화한 한계는 B ≥ 4에서 1e-6, B ≤ 2에서 2e-6).
그래서 같은 B에서는 비트 단위로 같고 B가 다르면 한계 안에서만 같으며, B는 평가 정체성에 묶는다. 합성기의 뽑기와 모의 값은 균등 난수를 누적 확률과 견주므로 머리 출력의 이 한계보다 크게 달라질 수 있다(뽑기 하나가 뒤집힌다).
머리가 읽는 것은 선수 토큰 10개의 상태와 풀링 상태뿐이므로(heads._read) 토큰 전체(N_tok개) 대신 이 둘만 남기면 보기당 상태가 N_tok/11배 줄고(요약 토큰의 상태 h_pool은 h_tok의 마지막 토큰이어서 따로 세지 않는다) 마스크도 없어진다(사건 상한 128이면 N_tok = 173, 173/11 = 15.7배, 계획 Q6의 수치와 같다).
묶음 크기가 보기 수 이상이면 한 번의 부호화와 같다. 기본값 None은 지금의 한 번 부호화 그대로이며 캐시도 전체 부호화를 담는다.

handle_i는 features.m4_masked_inputs의 float32 행렬을 통과하므로 두 정수 모두
2^24 이하로 제한하고 초과는 거부한다. 첫 열은 기존 누설 변형의 로컬 행
해석을 보존한다. 참조는 어댑터 내부용이며 다른 인스턴스 간 인증 표지가 아니다.
빈 입력의 출력 모양도 규약을 따르며 추론은 기울기 없이 평가 모드로 한다.
모형의 기존 모듈별 학습 모드는 성공·실패 뒤 모두 복구한다.
O가 K에 조건부이므로 닫힌 꼴을 제공하지 않고 모의 합성만 허용한다.

근거: 계획 Q7·Q9·Q12, 작업서 Task6, heads.ComposerHeads, (EH-9).
"""
from contextlib import contextmanager

import numpy as np
import torch

from c_ept.ept import data
from c_ept.ept.encoder import Encoded, token_layout
from c_ept.ept.heads import EPT, FrameAveraged, FrameEncoded

MAX_HANDLE = 2**24


def _indices(name, values, n, low, high, *, packed=False):
    a = np.asarray(values)
    if a.ndim != 1 or len(a) != n:
        raise ValueError(f'{name} must be a 1-d array of length {n}')
    if not packed and a.size and a.dtype.kind not in 'iu':
        raise TypeError(f'{name} must contain integers')
    if a.size and (a.dtype.kind not in 'fiu' or not np.isfinite(a).all() or (a != np.floor(a)).any() or a.min() < low or a.max() > high):
        raise ValueError(f'{name} must contain finite integers in {low} .. {high}')
    return a.astype(np.int64)


def _counts(name, values, n, width, *, dtype=np.float32):
    a = np.asarray(values)
    if a.shape != (n, width) or a.dtype.kind not in 'fiu' or not np.isfinite(a).all() or (a < 0).any() or (a > np.finfo(np.float32).max).any():
        raise ValueError(f'{name} must be a finite nonnegative ({n}, {width}) numeric array representable in float32')
    return a.astype(dtype, copy=True)


def _chunk_views(value):
    """``chunk_views`` as a Python int, or None; a ValueError (that does not repeat the value) for anything that is not None or a positive integer (a bool, a float and a string are not)."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError('chunk_views must be a positive integer (views per encoding chunk) or None (one encoding pass)')
    return int(value)


def _head_inputs(encoded: Encoded) -> Encoded:
    """The part of one encoding that the heads read, as new tensors that own their memory (S4b preparation M14, EC-6).

    ``heads._read`` takes the states of the ten player tokens (``token_layout(0).players``) and the pooled state; ``_row_count`` takes the number of views of the pool. Nothing reads the other tokens or the
    mask, so the mask is an empty (views, 0) tensor. A slice of the encoder output would keep the whole output alive: ``clone`` copies, and the cache of a chunk no longer references the chunk.
    """
    pool = encoded.h_pool.clone()
    return Encoded(encoded.h_tok[:, token_layout(0).players].clone(memory_format=torch.contiguous_format), pool, torch.zeros((pool.shape[0], 0), dtype=torch.bool, device=pool.device))


def _chunk_cache(encoded: FrameEncoded) -> FrameEncoded:
    """``encoded`` reduced to what the heads read in each frame (``_head_inputs``); the swapped frame stays None for one frame (``FrameAveraged(average=False)``), the side marker is kept, ``rows`` is None."""
    return FrameEncoded(_head_inputs(encoded.original), None if encoded.swapped is None else _head_inputs(encoded.swapped), encoded.side.clone())


def _join(parts: list) -> FrameEncoded:
    """The caches of the chunks put one after the other along the view axis, in the order of the chunks."""
    def cat(frame):
        return Encoded(*(torch.cat([getattr(frame(p), field) for p in parts], 0) for field in Encoded._fields))
    swapped = None if parts[0].swapped is None else cat(lambda p: p.swapped)
    return FrameEncoded(cat(lambda p: p.original), swapped, torch.cat([p.side for p in parts], 0))


class EPTComposerHeads:
    """Frame-only EPT adapter with the public ComposerHeads methods.

    해석: 바인딩은 프레임 시계 입력만 고정하고 합성 사슬에서 뽑은 조건만 각 머리에 전달한다.

    ``chunk_views`` (keyword only, default None; S4b preparation M14, decision Q6 = A, T011 A2-F6): None encodes all bound views in one pass and keeps the whole encoding, as before. A positive integer B
    encodes the views in chunks of B views, in the requested order, and keeps only what the heads read (EC-6), so the cache of a very large evaluation set fits in memory (about 15.7 times smaller). The
    outputs of the same B are bit-identical across runs; another B (or None) moves them at the last digits of float32 (real scale B = 256: relative 1.15e-7, plan Q6 / T011; toy model B = 1: 5.7e-7, the tests of M14;
    documented bound 1e-6 for B >= 4 and 2e-6 for B <= 2), so the evaluation identity records B (``chunk_views``, read by the run profile); a B of at least the number of views is the one pass. Composer draws and
    simulated values compare uniforms with cumulative probabilities and may differ by more than that head-output bound (a draw can flip). ``strata.flow_composer`` therefore requires the same B in both chains. ValueError for another value. The adapter itself has no provenance record: B is recorded by the profile, and only when it is not None.
    """
    feature_kind = 'ept'
    analytic_ok = False
    obj_uses_K = True
    conv_uses_KO = True

    def __init__(self, model, stats, *, chunk_views=None):
        self._chunk_views = _chunk_views(chunk_views)
        if isinstance(model, FrameAveraged):
            self.frame, self.model = model, model.model
        elif isinstance(model, EPT):
            self.model, self.frame = model, FrameAveraged(model)
        else:
            raise TypeError('model must be EPT or FrameAveraged(EPT)')
        if not isinstance(self.model, EPT):
            raise TypeError('the frame model must be EPT')
        self.stats = stats
        self._generation, self._n, self._enc = 0, None, None

    @property
    def chunk_views(self) -> int | None:
        """The number of views encoded together (S4b preparation M14), or None for one pass; the evaluation identity records it when it is not None."""
        return self._chunk_views

    @property
    def zone_mark(self) -> bool:
        """Whether the model was built with the zone mark (S4b preparation M9, decision D9): the setting of its heads (``EPTHeads.zone_mark``). ``Composer`` compares it with its own."""
        return self.model.heads.zone_mark

    @contextmanager
    def _evaluation(self):
        modes = [(module, module.training) for module in self.frame.modules()]
        try:
            self.frame.eval()
            with torch.no_grad():
                yield
        finally:
            for module, mode in modes:
                module.training = mode

    @property
    def _device(self):
        return next(self.model.parameters()).device

    def _tensor(self, values):
        return torch.as_tensor(values, device=self._device)

    def bind(self, s2, views):
        """Cache encoded frame-clock views in their requested order; refuse event views.

        해석: 입력 번호와 보기 종류를 먼저 검사하므로 거부할 사건 보기의 특징은 읽지 않는다.
        부호화가 끝난 뒤에만 새 캐시를 채택하여 실패한 바인딩은 기존 참조를 보존한다.
        chunk_views가 정수 B이면 (EC-6) 보기를 요청한 순서대로 B개씩 부호화해 머리가 읽는 텐서만 이어 붙이므로, 한 번에 메모리에 있는 완전한 부호화는 묶음 하나이고 중간 묶음이 실패해도 기존 캐시는 그대로다.
        """
        src, chosen = np.asarray(s2.array('views/src')), np.asarray(views)
        if src.ndim != 1:
            raise ValueError('views/src must be 1-d')
        if chosen.dtype == np.bool_:
            if chosen.shape != src.shape:
                raise ValueError('a view mask must cover all views')
            idx = np.flatnonzero(chosen)
        else:
            idx = _indices('views', chosen, len(chosen) if chosen.ndim else -1, 0, len(src) - 1)
        if (src[idx] != 0).any():
            raise ValueError('bind accepts frame-clock views (views/src == 0) only')
        generation = self._generation + 1
        if generation > MAX_HANDLE:
            raise ValueError('binding generation exceeds exact float32 integer range')
        if len(idx) > MAX_HANDLE + 1:
            raise ValueError('bound row handles exceed exact float32 integer range')
        if any(p.dtype != torch.float32 for p in self.model.parameters()):
            raise ValueError('EPT composition requires float32 model parameters')
        enc = None
        if len(idx) and self._chunk_views is None:
            batch = {k: v.to(self._device) for k, v in data.ViewTensors(s2, idx, self.stats).items()}
            with self._evaluation():
                enc = self.frame.encode(batch)
        elif len(idx):
            parts = []
            with self._evaluation():
                for lo in range(0, len(idx), self._chunk_views):                                     # (EC-6): C_c = H(Enc(X[chunk c])), the chunk is released before the next is encoded
                    batch = {k: v.to(self._device) for k, v in data.ViewTensors(s2, idx[lo:lo + self._chunk_views], self.stats).items()}
                    parts.append(_chunk_cache(self.frame.encode(batch)))
                    del batch
            enc = _join(parts)
        self._enc, self._n, self._generation = enc, len(idx), generation

    def _rows(self, rows):
        if self._n is None:
            raise RuntimeError('call bind before reading EPT heads')
        a = np.asarray(rows)
        return _indices('rows', a, len(a) if a.ndim else -1, 0, self._n - 1)

    def row_view(self, rows):
        """Return copied numeric handles (local row, binding generation), shape(n,2).

        수식
        ----
        (EC-1) handle_i = (i, bind_id).
          기호 | 뜻 | 단위 | 범위 | 코드 변수
          handle_i | 캐시 참조 두 열 | — | 정확한 float32 정수 | 반환 배열
          i | 현재 바인딩 안의 행 | — | 0..n−1, ≤2^24 | rows
          bind_id | 성공한 바인딩의 세대 | — | 1..2^24 | _generation

        해석
        ----
        일반 수치 행렬을 거쳐도 두 정수를 잃지 않도록 참조 범위를 제한한다.
        세대는 같은 수의 보기에 다시 바인딩해도 바뀌므로 오래된 합성기와 M4 참조를 구별한다.
        빈 행 선택은 두 열의 빈 배열이며 원본 캐시는 외부에 노출하지 않는다.

        근거
        ----
        계획 Q9, Task6, compose.Composer._check_bound.
        """
        r = self._rows(rows)
        return np.column_stack((r, np.full(len(r), self._generation))).astype(np.float32)

    def _m2(self, rows):
        return self.frame.m2(self._enc, torch.ones((len(rows), 18), device=self._device), pos=self._tensor(rows))

    def rate(self, rows):
        """주어진 행들에 대한 18개 구간별 초당 첫 킬 발생률 λ_{i,q}를 반환한다 (EC-2).

        매개변수:
        - rows: 평가 대상 행 번호 목록 또는 인덱스 컬렉션.

        반환값:
        - rate (ndarray): 초당 발생률 (1/s), float32 형상 (n, 18).

        수식 및 동작:
        (EC-2) λ_{i,q} = M2(handle_i, Δ_q = 1).rate_q
        - 신경망 출력은 노출이 곱해진 기대 개수이므로, 단위 노출(Δ_q = 1초) 텐서를 전달하여 초당 발생률을 얻는다. 실제 관측 창 길이는 합성기가 사후에 1회 곱한다.
        - self._evaluation() 문맥 내에서 frame.m2_rate를 호출하여 표지·구역 분기 연산을 생략한다.
        - rows가 비어 있는 경계 조건에서는 신경망 연산 없이 형상 (0, 18)의 빈 float32 ndarray를 즉시 반환한다.
        기호 | 뜻 | 단위/범위 | 코드 변수
        λ_{i,q} | 첫 킬 발생률 | 1/s, ≥0 | 반환 배열
        i,q | 행과 앞 시간 구간 | i<n, q=0..17 | rows, 출력 열
        Δ_q | 머리에 전달하는 노출 | 1 s | ones 텐서
        근거: 계획 Q7·Q9, (EH-1), ComposerHeads.rate."""
        r = self._rows(rows)
        if not len(r):
            return np.empty((0, 18), dtype=np.float32)
        with self._evaluation():
            return self.frame.m2_rate(self._enc, torch.ones((len(r), 18), device=self._device),
                                      pos=self._tensor(r)).cpu().numpy()

    def mark_prob(self, rows, q):
        """(n,52) mark probabilities given row and bin.

        수식
        ----
        (EC-2) π_{i,q,m} = exp(M2(handle_i,1).mark_logp_{q,m}).
          기호 | 뜻 | 단위 | 범위 | 코드 변수
          π | 표지 확률 | — | >0, 표지 합1 | 반환 배열
          i,q,m | 행, 앞 시간 구간, 표지 | — | i<n, 0..17, 0..51 | rows,q,출력 열

        해석
        ----
        프레임 평균한 로그 확률을 확률로 되돌리고 합성에서 뽑은 구간만 고른다.
        균등 혼합 바닥과 정규화는 원래 머리에 그대로 맡긴다.
        빈 행은 열52의 빈 배열이다.

        근거
        ----
        계획 Q7·Q9, (EH-9), ComposerHeads.mark_prob.
        """
        r = self._rows(rows)
        q = _indices('q', q, len(r), 0, 17)
        if not len(r):
            return np.empty((0, 52), dtype=np.float32)
        with self._evaluation():
            out = self._m2(r).mark_logp.exp()
            return out[self._tensor(np.arange(len(r))), self._tensor(q)].cpu().numpy()

    def zone_prob(self, rows, q, m):
        """(n,10) zone probabilities given row, bin and mark.

        수식
        ----
        (EC-2) ρ_{i,q,m,z} = exp(M2(handle_i,1).zone_logp_{q,m,z}).
          기호 | 뜻 | 단위 | 범위 | 코드 변수
          ρ | 규칙 구역 확률 | — | >0, 구역 합1 | 반환 배열
          i,q,m,z | 행, 구간, 표지, 구역 | — | i<n,0..17,0..51,0..9 | rows,q,m,출력 열

        해석
        ----
        합성 사슬의 구간과 표지에 해당하는 열 구역의 법칙을 선택한다.
        위치 없음은 이 법칙의 범주가 아니며 빈 행에는 열10의 빈 배열을 준다.

        근거
        ----
        계획 Q7·Q9, (EH-5)·(EH-9), ComposerHeads.zone_prob.
        """
        r = self._rows(rows)
        q, m = _indices('q', q, len(r), 0, 17), _indices('m', m, len(r), 0, 51)
        if not len(r):
            return np.empty((0, 10), dtype=np.float32)
        with self._evaluation():
            out = self._m2(r).zone_logp.exp()
            return out[self._tensor(np.arange(len(r))), self._tensor(q), self._tensor(m)].cpu().numpy()

    def _conditions(self, rows, q, m, z):
        r = self._rows(rows)
        n = len(r)
        return r, tuple(self._tensor(_indices(name, x, n, lo, hi)) for name, x, lo, hi in (('q', q, 0, 17), ('m', m, 0, 51), ('z', z, -1, 9)))

    def _m3(self, r, codes):
        return self.frame.m3(self._enc, *codes, pos=self._tensor(r))

    def size_prob(self, rows, q, m, z):
        """(n,4) size probabilities for the sampled onset conditions.

        수식
        ----
        (EC-3) P_i(s | q,m,z) = exp(M3(handle_i,q,m,z).size_logp_s).
          기호 | 뜻 | 단위 | 범위 | 코드 변수
          P_i | 규모 법칙 | — | >0, 규모 합1 | 반환 배열
          i,q,m,z,s | 행, 구간, 표지, 구역, 규모 | — | i<n,0..17,0..51,−1..9,0..3 | rows,q,m,z,출력 열

        해석
        ----
        결과 전문가에 앞서는 규모 관문을 그대로 전달한다.
        구역 −1은 위치 없는 교전의 조건이고 빈 행에는 열4의 빈 배열을 준다.

        근거
        ----
        계획 Q7·Q9, (EH-6)·(EH-9).
        """
        r, codes = self._conditions(rows, q, m, z)
        if not len(r):
            return np.empty((0, 4), dtype=np.float32)
        with self._evaluation():
            return self._m3(r, codes).size_logp.exp().cpu().numpy()

    def kill_rate(self, rows, q, m, z, s):
        """(n,2) Poisson kill means for the sampled size.

        수식
        ----
        (EC-3) μ_{i,t} = M3(handle_i,q,m,z).kill_rate_{s,t}.
          기호 | 뜻 | 단위 | 범위 | 코드 변수
          μ | 팀별 킬 기대 개수 | 개 | ≥0 | 반환 배열
          i,q,m,z,s | 행과 가린 조건 | — | size_prob 표와 같음 | rows,q,m,z,s
          t | 팀 | — | 0파랑,1빨강 | 출력 열

        해석
        ----
        합성에서 뽑은 규모 전문가의 두 팀 포아송 평균을 선택한다.
        이동 킬 법칙(결정 D5, K_d = 1 + N_d)에서는 첫 킬 팀 d = m // 25의 칸이 초과분 N_d의 평균 λ_d이고 뜻의 해석은 합성기(``Composer(kill_law=)``)가 맡는다; 이 머리는 출력만 낸다.
        이 머리는 교전당 개수이므로 시간 노출을 다시 곱하지 않는다.
        빈 행은 열2의 빈 배열이다.

        근거
        ----
        계획 Q7·Q9, (EH-6), ComposerHeads.kill_rate.
        """
        r, codes = self._conditions(rows, q, m, z)
        s = _indices('s', s, len(r), 0, 3)
        if not len(r):
            return np.empty((0, 2), dtype=np.float32)
        with self._evaluation():
            out = self._m3(r, codes).kill_rate
            return out[self._tensor(np.arange(len(r))), self._tensor(s)].cpu().numpy()

    def gold_prob(self, rows, q, m, z, s, layer):
        """(n,2,31) gold-bin probabilities conditioned on each team's own kill layer.

        수식
        ----
        (EC-3) p_{i,t,b} = exp(M3(handle_i,q,m,z).gold_logp_{s,t,ξ_{i,t}−1,b}).
          기호 | 뜻 | 단위 | 범위 | 코드 변수
          p | 골드 구간 법칙 | — | >0, b합1 | 반환 배열
          i,q,m,z,s | 행과 가린 조건 | — | size_prob 표와 같음 | rows,q,m,z,s
          t,ξ,b | 팀, 자기 팀 킬 층, 골드 구간 | — | 0..1,1..6,0..30 | 출력 축,layer,출력 축

        해석
        ----
        각 팀의 법칙은 상대 팀이 아닌 자기 팀의 킬 층으로 선택한다.
        킬 0의 골드를 0으로 만드는 일은 합성기가 하며 층6은 여섯 킬 이상을 뜻한다.
        빈 출력의 모양은 (0,2,31)이다.

        근거
        ----
        계획 Q7·Q9, P4, (EH-6), ComposerHeads.gold_prob.
        """
        r, codes = self._conditions(rows, q, m, z)
        s = _indices('s', s, len(r), 0, 3)
        a = np.asarray(layer)
        if a.shape != (len(r), 2):
            raise ValueError('layer must have shape (n, 2)')
        layers = _indices('layer', a.ravel(), len(r) * 2, 1, 6).reshape(-1, 2)
        if not len(r):
            return np.empty((0, 2, 31), dtype=np.float32)
        with self._evaluation():
            out = self._m3(r, codes).gold_logp.exp()
            rr, ss, ll = self._tensor(np.arange(len(r))), self._tensor(s), self._tensor(layers - 1)
            return torch.stack([out[rr, ss, t, ll[:, t]] for t in (0, 1)], dim=1).cpu().numpy()

    def obj_rate(self, rows, q, m, z, s, K):
        """(n,18) objective means given sampled kills and raw kill gold.

        수식
        ----
        (EC-4) K' = (k_B,k_R,g_B/1000,g_R/1000) ; μ^O = M3Obj(handle_i,q,m,z,s,K').
          기호 | 뜻 | 단위 | 범위 | 코드 변수
          K,K' | 두 팀 킬 수와 킬 골드 | 개,골드 → 개,천 골드 | ≥0 | K,kills
          μ^O | 목적물 칸별 평균 | 개 | ≥0 | 반환 배열
          i,q,m,z,s | 행과 가린 조건 | — | size_prob 표와 같음 | rows,q,m,z,s

        해석
        ----
        합성기가 표본으로 만든 킬과 골드만 목적물 조건으로 사용한다.
        원시 골드를 학습 입력과 같은 천 골드 단위로 한 번만 바꾸며 호출자 배열은 보존한다.
        합성기가 넣는 골드는 뽑은 구간의 대표값(킬이 없는 팀은 0)이다. K 조건 'binned'로 학습한 머리(S4b 준비 M3, 결정 D21 B16)는 학습 행의 골드도 같은 값 집합 {0} ∪ {rep_b}에서
        왔으므로 학습과 합성의 입력 지지가 같고, 'raw'(기본값)로 학습한 머리는 연속 골드를 보고 학습했다. 어느 쪽인지는 이 머리가 모르며 학습 정체성(``k_condition``)이 말한다.
        빈 행에는 열18의 빈 배열을 준다.

        근거
        ----
        계획 Q7·Q9, data.ROW_FIELDS, (EH-7).
        """
        r, codes = self._conditions(rows, q, m, z)
        s = _indices('s', s, len(r), 0, 3)
        kills = _counts('K', K, len(r), 4, dtype=np.float64)
        kills[:, 2:] /= 1000.  # match features.m4_masked_inputs: convert units before float32 narrowing
        kills = kills.astype(np.float32)
        if not len(r):
            return np.empty((0, 18), dtype=np.float32)
        with self._evaluation():
            return self.frame.m3_obj(self._enc, *codes, self._tensor(s), self._tensor(kills), pos=self._tensor(r)).cpu().numpy()

    def conv_rate(self, m4_inputs):
        """(n,18) conversion rates per second from features.m4_masked_inputs.

        수식
        ----
        (EC-5) r^C = M4(handle_i,q,m,z,K',O,Δ=1).
          기호 | 뜻 | 단위 | 범위 | 코드 변수
          handle_i | EC-1의 현재 세대 캐시 참조 | — | 두 정수 열 | x[:,:2]
          q,m,z | 앞 시간 구간, 표지, 구역 | — | 0..17,0..51,−1..9 | x[:,2:5]
          K',O | 표본 킬·천 골드와 목적물18칸 | 개,천 골드 | ≥0 | x[:,5:9],x[:,9:]
          Δ,r^C | 머리 노출, 전환 발생률 | s,1/s | 1,≥0 | ones,반환 배열

        해석
        ----
        일반 특징 함수가 골드를 이미 천 단위로 바꿨으므로 여기서 다시 나누지 않는다.
        실제 전환 창은 합성기가 곱하며 이 머리는 노출1로 초당 발생률만 낸다.
        참 δ와 사건 보기는 입력에 없고 오래된 바인딩 참조는 거부한다.
        빈 행에는 열18의 빈 배열을 준다.

        근거
        ----
        계획 Q7·Q9, features.m4_masked_inputs, (EH-8)·(EH-9).
        """
        if self._n is None:
            raise RuntimeError('call bind before reading EPT heads')
        x = np.asarray(m4_inputs)
        if x.ndim != 2 or x.shape[1] != 27 or x.dtype.kind not in 'fiu':
            raise ValueError('m4_inputs must be a numeric (n, 27) matrix')
        n = len(x)
        r = _indices('row handle', x[:, 0], n, 0, self._n - 1, packed=True)
        gen = _indices('generation handle', x[:, 1], n, 1, MAX_HANDLE, packed=True)
        if (gen != self._generation).any():
            raise ValueError('stale binding generation in M4 input')
        codes = tuple(self._tensor(_indices(name, x[:, col], n, lo, hi, packed=True)) for name, col, lo, hi in (('q', 2, 0, 17), ('m', 3, 0, 51), ('z', 4, -1, 9)))
        K, O = _counts('K', x[:, 5:9], n, 4), _counts('O', x[:, 9:], n, 18)
        if not n:
            return np.empty((0, 18), dtype=np.float32)
        with self._evaluation():
            return self.frame.m4(self._enc, *codes, self._tensor(K), self._tensor(O), torch.ones(n, device=self._device), pos=self._tensor(r)).cpu().numpy()
