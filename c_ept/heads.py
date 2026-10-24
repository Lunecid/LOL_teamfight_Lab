r"""The head interface that the composer reads (stage S3 Task 6, ruling R3).

A *head* is one conditional law of the model: the rate of the first kills in a bin (M2), the mark and the zone of an engagement (M2), its
size, its kills, its kill gold and its objectives (M3), and the objectives converted after it (M4). The composer (``compose``, Task 9)
draws an engagement from these laws, or computes their expectation, and it must not care which model sits behind them. The B0 GLMs
(``baselines_b0``, Task 6), the B1 LightGBM boosters (``baselines_b1``, Task 7) and the later EPT all implement ``ComposerHeads`` and
``compose`` imports it from here (and re-exports it).

Rows. ``bind(s2, views)`` computes and keeps the row features of the composer's t0 frame-clock views, in the order in which the views are
given (``views``: view numbers of the S2 store, or a bool mask over them); every other method takes ``rows``, an int array (n,) that
indexes those bound rows, and the categories of an onset as int arrays (n,): ``q`` the occurrence bin (0 .. 17), ``m`` the mark of the
first kill (0 .. 51), ``z`` the zone of the first kill (-1 .. 9; -1 = no position), ``s`` the size class (0 .. 3). ``row_view(rows)`` gives
the bound features back, so that the composer builds the M4 input with ``features.m4_masked_inputs(heads.row_view(rows), q, m, z, K, O)``
and passes it to ``conv_rate``; the heads never see anything but what the composer hands them (decision P6: the masked M3 object head
does not get K, the masked M4 head gets neither K nor O).

Outputs, one row per input row (the last dimensions are in ``OUTPUT_WIDTHS``)

    rate(rows)                           (n, 18)      the first-kill rate of each bin, per second
    mark_prob(rows, q)                   (n, 52)      the law of the mark, rows add up to 1
    zone_prob(rows, q, m)                (n, 10)      the law of the zone
    size_prob(rows, q, m, z)             (n, 4)       the law of the size class
    kill_rate(rows, q, m, z, s)          (n, 2)       the mean number of kills of the blue and of the red team (Poisson)
    gold_prob(rows, q, m, z, s, layer)   (n, 2, 31)   the law of the kill-gold bin of each team given its kill layer; ``layer`` is (n, 2)
                                                      with values 1 .. 6 (the kills of the team, 6 = six or more)
    obj_rate(rows, q, m, z, s, K)        (n, 18)      the expected objectives of each cell; ``K`` is (n, 4) = kills blue, kills red,
                                                      kill gold blue, kill gold red (raw gold); a head with ``obj_uses_K`` False ignores it
    conv_rate(m4_inputs)                 (n, 18)      the conversion rate of each cell, per second; ``m4_inputs`` is the matrix of
                                                      ``features.m4_masked_inputs``; a head with ``conv_uses_KO`` False ignores K and O

Every categorical output is a full-width probability row that is strictly positive (ruling R13: p = (1 - 1e-6) p_model + 1e-6 / C).

Flags. ``feature_kind`` names the row features ('b0' or 'b1'). ``analytic_ok`` is True when the heads make the expectation of the value
of an engagement closed (B0: the masked M3 object head does not read K and the masked M4 head reads neither K nor O); the gate then
checks the simulation against the closed form (decision P6). ``obj_uses_K`` and ``conv_uses_KO`` say whether ``obj_rate`` reads K and
``conv_rate`` reads K and O.

방법에서의 위치. 이 모듈은 교전 과정 모형(EPT)이 예측하는 조건부 분포 하나하나(머리)를 합성기가 같은 호출로 받도록 입출력 규약을 정한다. 기준 모형 B0와 B1은
이 규약을 따르고(판정 R3) 이후의 EPT는 S4a 계획 Q9의 어댑터로 따르므로 합성기는 어느 모형이 뒤에 있는지 모른다. 교전은 같은 경기에서 15초와 2,000 지도 단위 이내로
이어진 킬의 묶음이고(설계서 2절), 발생은 분 프레임의 기준 시각 t₀ 뒤 구간 q에서 한 교전의 첫 킬이 일어난 것이다. 설계서 3절은 한 발생을 표지 m과 구역 z(3.1절),
규모 s, 팀별 킬 K, 오브젝트 칸별 개수 O(3.2절의 분해 P(s)·P(K | s)·P(O | K, s); 계획 P4), 전환(3.3절)의 차례로 쪼개고, 머리 하나가 이 차례의 한 고리를 맡는다.
이 규약은 t₀ 행에 묶인 가린 머리만 다룬다
(``bind``가 합성기의 t₀ 분 프레임 보기를 묶는다). 이 모듈은 분포를 계산하지 않고 분포가 오가는 모양과 그 조건만 정한다.

용어. 분 프레임: 경기 상태를 1분마다 찍은 기록이고 t₀는 그 시각이다. 구간 q: t₀ 뒤 10초씩 18개(0–17). 표지 m: 첫 킬의 선제 킬 쌍(킬러 슬롯이 피해자
슬롯을 잡는 방향 쌍 50개)과 처형 2개를 합친 52가지. 킬 수 층: 킬이 하나 이상인 팀의 킬 수를 1–5와 6 이상으로 나눈 여섯 층. 오브젝트 칸: 오브젝트 종류와
가져간 팀의 쌍마다 하나(18칸). 전환: 교전이 끝난 뒤 60초 안에 어느 교전에도 붙지 않은 오브젝트(설계서 3.3절). 가린 머리: 합성기가 줄 수 있는 입력(t₀ 행의 특징과
사슬의 앞 단계 값: 범주 q, m, z, s, 킬 수 층, 머리에 따라 K와 O)만 읽는 머리이고, 열린 머리는 교전의 사건 보기를 읽는다(설계서 3.2절·3.3절).

식 목록. 구간의 발생 수와 한 발생의 사슬이다. 식 번호는 (모듈 약어-번호) 꼴이며 식과 기호는 docs/superpowers/notes/equation-index.md의 식 색인과
기호 등록부에 올라 있다.
    (HEAD-1)  n_q ~ Poisson(λ_q·Δ_q)                                  구간 q의 발생 수 (λ_q = ``rate``의 q열, Δ_q = 노출(초))
    (HEAD-2)  P(m, z, s | q) = π(m | q)·ρ(z | q, m)·P(s | q, m, z)      표지·구역·규모 (``mark_prob``, ``zone_prob``, ``size_prob``)
    (HEAD-3)  k_blue ~ Poisson(μ_blue),  k_red ~ Poisson(μ_red)         팀별 킬 수 (μ = ``kill_rate``)
              킬이 k ≥ 1인 팀의 골드 구간 ~ ``gold_prob``(킬 수 층 ξ = min(k, 6));  k = 0이면 골드 0
    (HEAD-4)  O_c ~ Poisson(μ_c);  c칸의 전환 수 ~ Poisson(λ_c·e)       (μ_c = ``obj_rate``, λ_c = ``conv_rate``의 초당 값, e = 전환 노출(초))
    (HEAD-5)  p = (1 − ν)·p^model + ν / C,   ν = 1e-6                   범주 출력의 섞기(C = 범주의 수; 판정 R13)

공통 가정.
    (가) 개수는 포아송이다. 발생 수는 설계서 3.1절, 팀별 킬 수와 오브젝트 칸은 계획 P4, 전환 칸은 계획 전역 제약의 확률 족을 따른다.
    (나) 한 발생의 각 단계는 사슬의 앞 단계 값에 조건부이다(설계서 3절의 분해 P(s)·P(K | s)·P(O | K, s), 계획 P4). 어느 값을 읽는지는 모형이 정하며 B0의 가린 오브젝트·전환
         머리는 K와 O를 읽지 않는다(계획 P6). 서로 다른 발생과 두 팀의 킬 수는 입력이 주어지면 독립으로 뽑는다(합성기의 모의가 그렇게 뽑는다).
    (다) 발생률과 전환율은 초당 값이다. 노출은 경기 끝에서 자른 관측 시간이며 모형의 입력이 아니어서 호출한 쪽이 곱한다(계획 P10).
    (라) (HEAD-5)로 확률이 0인 범주가 없으므로 범주 점수 (NLL-2)와 (NLL-3)이 유한하다(판정 R13).
"""
# Ref: docs/superpowers/plans/2026-10-21-c-ept-s3.md decisions P4 (the factorisation P(s) P(K | s) P(O | K, s) and the probability family),
#      P6 (what the masked heads are conditioned on) and Task 9 (the composer); controller ruling R3 (this module) and R13 (full-width,
#      strictly positive categorical outputs).
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from c_ept import config as C

FEATURE_KINDS = ('b0', 'b1')
# the last dimensions of the outputs of the methods; the zone, size, gold and cell counts are checked against ``features``, ``units``,
# ``nll`` and ``targets`` by a test (this module imports none of them, so that every head module can import it)
OUTPUT_WIDTHS = {'rate': (C.N_BINS,), 'mark_prob': (C.N_MARKS,), 'zone_prob': (10,), 'size_prob': (4,), 'kill_rate': (2,),
                 'gold_prob': (2, 31), 'obj_rate': (2 * len(C.O_KINDS),), 'conv_rate': (2 * len(C.O_KINDS),)}


@runtime_checkable
class ComposerHeads(Protocol):
    """The heads of one model as the composer reads them (see the module text for the shapes).

    ``isinstance(x, ComposerHeads)`` is True for an object that has all the members (the check is by name, not by signature).

    방법 하나가 사슬의 한 고리이다: ``rate``는 (HEAD-1), ``mark_prob``·``zone_prob``·``size_prob``는 (HEAD-2), ``kill_rate``·``gold_prob``는 (HEAD-3),
    ``obj_rate``·``conv_rate``는 (HEAD-4)이다. 이름만 검사하므로 출력의 모양과 확률의 합은 이 규약이 검사하지 않는다."""
    feature_kind: str
    analytic_ok: bool
    obj_uses_K: bool
    conv_uses_KO: bool

    def bind(self, s2, views) -> None:
        """Compute and keep the row features of the t0 frame-clock views ``views`` (numbers of S2 views, or a bool mask), in that order.

        사슬 전체의 입력인 t₀ 행의 특징을 한 번 계산해 둔다. 사건 보기는 첫 킬이나 교전 끝까지의 사건을 담으므로 가린 머리가 읽어서는 안 된다(설계서 10절 A3)."""
        ...

    def row_view(self, rows) -> np.ndarray:
        """(n, d) the bound row features of ``rows`` (int array (n,) into the bound views).

        합성기가 전환 머리의 입력을 만들 때 ``features.m4_masked_inputs``에 넘기는 값이다(설계서 3.3절 누설 시험 (a))."""
        ...

    def rate(self, rows) -> np.ndarray:
        """(n, 18) the rate of the first kills of each occurrence bin, per second.

        식 (HEAD-1)의 λ_q를 18구간 모두에 낸다."""
        ...

    def mark_prob(self, rows, q) -> np.ndarray:
        """(n, 52) the law of the mark of the first kill given the row and the bin q.

        식 (HEAD-2)의 π(m | q)이다. 조건은 t₀ 행과 구간 q뿐이다."""
        ...

    def zone_prob(self, rows, q, m) -> np.ndarray:
        """(n, 10) the law of the zone of the first kill given the row, q and the mark m.

        식 (HEAD-2)의 ρ(z | q, m)이다. 구역이 관측되지 않은 첫 킬(z = −1)은 이 분포의 결과가 아니어서 구역 10개만 낸다(계획 전역 제약의 구역 z).
        합성에서 이런 발생을 어떻게 다룰지는 S8에서 정한다."""
        ...

    def size_prob(self, rows, q, m, z) -> np.ndarray:
        """(n, 4) the law of the size class given the row, q, m and the zone z (-1 = no position).

        식 (HEAD-2)의 P(s | q, m, z)이다. 입력에서 구역이 관측되지 않은 경우(z = −1)는 한 수준으로 읽는다(계획 전역 제약의 구역 z)."""
        ...

    def kill_rate(self, rows, q, m, z, s) -> np.ndarray:
        """(n, 2) the mean number of kills of the blue team and of the red team given the row, q, m, z and the size class s.

        식 (HEAD-3)의 포아송 평균 μ_blue, μ_red를 한 호출에 낸다. 규모 s는 팀별 킬 수의 조건이다."""
        ...

    def gold_prob(self, rows, q, m, z, s, layer) -> np.ndarray:
        """(n, 2, 31) the law of the kill-gold bin of each team; ``layer`` (n, 2), values 1 .. 6, is the kill layer of each team.

        식 (HEAD-3)의 골드 구간 분포이다. 팀마다 자기 킬 수 층으로 읽는다(계획 P4)."""
        ...

    def obj_rate(self, rows, q, m, z, s, K) -> np.ndarray:
        """(n, 18) the expected number of objectives of each cell; ``K`` is (n, 4): kills blue, kills red, gold blue, gold red.

        식 (HEAD-4)의 포아송 평균 μ_c이다. 개수의 평균이어서 노출을 곱하지 않는다(M3 머리의 노출은 1, head_rows 모듈 글)."""
        ...

    def conv_rate(self, m4_inputs) -> np.ndarray:
        """(n, 18) the rate of the objectives converted in each cell, per second, from the input of ``features.m4_masked_inputs``.

        식 (HEAD-4)의 전환율 λ_c이다."""
        ...
