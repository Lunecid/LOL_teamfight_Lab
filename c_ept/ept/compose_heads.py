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
from c_ept.ept.heads import EPT, FrameAveraged

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


class EPTComposerHeads:
    """Frame-only EPT adapter with the public ComposerHeads methods.

    해석: 바인딩은 프레임 시계 입력만 고정하고 합성 사슬에서 뽑은 조건만 각 머리에 전달한다.
    """
    feature_kind = 'ept'
    analytic_ok = False
    obj_uses_K = True
    conv_uses_KO = True

    def __init__(self, model, stats):
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
        if len(idx):
            batch = {k: v.to(self._device) for k, v in data.ViewTensors(s2, idx, self.stats).items()}
            with self._evaluation():
                enc = self.frame.encode(batch)
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
