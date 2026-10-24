r"""S3 runner: the five stages of stage S3 (victim levels, labels, baseline fits, composition, gate) in one resumable command.

    python -m c_ept.run_s3 --s1 <S1 run> --s2 <S2 run> --s3a <s3a run> --smoke 1000 --workers 8
    python -m c_ept.run_s3 --s1 <S1 run> --s2 <S2 run> --s3a <s3a run> --stages levels labels fit compose gate
    python -m c_ept.run_s3 --s1 <S1 run> --s2 <S2 run> --s3a <s3a run> --resume <S3 run folder> [--smoke N]

Inputs and the decision record. ``--s3a`` is the s3a run folder (``prices\PRICES.json``, ``rules\RULE_PAYOUTS.json``). The SHA-256 of both
files must equal the ones that ``decisions.S3_prices`` of the decision record (``--spec``, default ``config.SPEC_PATH``) holds; a run on
any other file is refused before a run folder is made. The prices are read with ``value.load_prices`` and the rules with
``rulepay.RuleTable.load(path, patches=rulepay.P3_PATCHES)`` (ruling R10: any other patch raises when it is priced).

The stages (the run folder is ``<runs root>\<YYYYMMDD>_c_ept_s3_<n>``, ``runlog.new_run('s3')``)

    levels    ``levels\kill_levels.npy``  the victim level of every kill of FIT, STOP and DEV (``value.kill_levels``: the raw files, read
              by ``reader.read_match`` only), and ``LEVELS.json``.
    labels    ``labels\store\``  the label store (``value.unit_values``, ``v_start``, ``accounting``, ``gold_bin_means``,
              ``write_labels``), ``labels\ACCOUNTING.json`` and ``labels\LABELS_SUMMARY.json``. The accounting identity must be exactly 0
              (ruling R11) or the stage fails. LABELS.json is scanned for match keys before it is written.
    fit       ``fit\``  the tower tier law (``tier\``), B0 (``b0\``), B1 (``b1\``) and B2 (``b2\``), each saved as soon as it is fitted
              (B1 head by head: every head goes to ``b1_heads\<kind>.<name>\`` when it is fitted, with the code state and the input hash
              of the stage in its record, and ``b1\`` is made from the 13 heads at the end),
              the facts of the fit (``FIT_FACTS.json``: the dropped columns, the FIT means, the fill counts, the time, the peak memory,
              ``converged`` / ``n_iter`` and the rounds of every head), the DEV scores of every head (``DEV_NLL.json``) and the DEV
              predictions that the gate needs (``DEV_PRED.npz``: the M2 expected counts of B0 and B1 and the probability of B2).
    compose   ``compose\b0_dev.npz`` (every DEV row: the closed form, the mean, the standard deviation and the 19 quantiles of S = 1,000
              simulations, the expected onsets) and ``compose\b1_dev.npz`` (a seeded sample of 500 DEV matches, S = 200, simulation only).
    gate      ``gate\GATE.json``: the four criteria of P7 and the over-dispersion of P8, the Brier scores of M2 (B0, B1) and B2 and the
              tower-pricing diagnostic.

The roles are pinned in the module (``TIER_ROLES``, ``B0_ROLES``, ``B1_ROLES``): the tier law and B0 are fitted on FIT; B1 and B2 on FIT
with the early stopping on STOP, never on DEV. These runner stages select FIT, STOP and DEV through the upstream role-selecting APIs,
leaving CAL and AUDIT outside their work and relying on those APIs to refuse unsupported roles such as TEST and EXTERNAL.

Resume. ``STAGES.json`` records, for every stage, whether it is complete, the hashes of its inputs (the stores, the s3a files, the
smoke selection and the output files of the stage before) and the SHA-256 of its output files. A resumed run (``--resume RUN``, the
same ``--s1``, ``--s2``, ``--s3a`` and ``--smoke``) skips the stages that are complete and refuses (exit code 2, nothing is run) when
an input or an output of a complete stage is not what was recorded, when a complete stage has no ``<stage>_result`` in RUN.json (a stage
is marked complete only after its result is key-scanned and recorded there; the rerun of a stage that is not complete replaces its own
result and no other key), when a stage that was begun and cut is begun again on other inputs (its recorded ``inputs_sha256`` differs),
and when the code is not the one the run began on (the commit or the dirty flag differs from RUN.json) or the tree is dirty: a run folder
never mixes the output of two codes or two sets of inputs. A stage that was cut is run again; ``fit`` keeps the models that
were saved (a model counts as saved only when its JSON, the last file of every ``save``, is there: an NPZ or a model folder without it
is removed and fitted again; B1 is saved head by head, and a head whose record is not the code and inputs of this call is refused) and
``compose`` keeps the chunks of rows that were written (``compose\parts``). The run folder holds
``S3.lock`` while a call works on it (exclusive; a lock that a power cut left behind is deleted by hand, the message says so).

Smoke (``--smoke N``). A seeded selection of N FIT, ceil(0.2 N) STOP and ceil(0.3 N) DEV matches (``select_smoke``; the match rows go to
RUN.json, never the keys). Every stage then works on that subset through ``SubsetStore``: the S1 store in which the matches outside the
selection have the role UNUSED. All the S3 code selects matches by ``matches/role`` alone, so the labels, the fits (also the tier law and
the early stopping on STOP) and the composition see the subset without a signature changing; the full run passes the S1 store itself.

E6 (no Strata). ``run_s1.require_no_strata()`` (imported inside functions, never at module level) is called before the first read, before
every stage and, through ``check=``, by every fit function and every compose call between their units of work (heads, boosters, blocks).
The check costs about 0.4 s: ``_Guard`` counts its calls and seconds, and the compose blocks are sized so that a block outlasts
``workers`` checks (below). A model server that starts stops the run cleanly (exit code 2, the message gives the command that resumes).

Memory. The peak of the private memory of this process and its children (``_Sampler``, as ``run_s1._PeakMemory`` sums it) is recorded for
every stage, every head, every call of the runner and the run; it must stay under ``MEMORY_CAP_GIB`` = 40 GiB (a stage over it is
reported, not hidden). Beside every ``peak_private_gib`` the record holds ``peak_private_upper_gib``: the largest sampled sum, over the
tree, of the ``peak_pagefile`` of each process (Windows: its lifetime peak of private commit), a bound for the lifetime private commit of the successfully read observed process set. Omitted or terminated children can be
missed; it is not a per-stage peak and may include pre-scope history. A sample without the field yields None, while a previously
observed non-None maximum is retained. The cap is judged on the sampled peak.

Composition (rulings R29-R32). The compose stage computes every value in spawned workers (``workers`` is at least 2, so the calling
process computes no compose value, R30) and never calls ``Composer.simulate_onsets``. Before anything is composed the expected onsets of
every row are screened through the composer (R31): a row with more than 50 expected onsets in 180 s stops the run and names the head set.
``Composer.check`` is called by the parent before each block, so a block must compute for well over ``workers`` times the check time or
the run is bound by the parent (the review of Task 9: about 3.6 s at 8 workers). The block of each method has a named constant
(``BLOCK_EXPECTED_ONSETS``, ``BLOCK_B0_SIMULATE``, ``BLOCK_B0_EXPECT``, ``BLOCK_B1_SIMULATE``); RUN.json records them, and the time of every
compose call and of every check, so that the full run can be tuned from the smoke run. COMPOSE.json counts the checks of the stage from
its start (``check_calls``, ``check_s_total``, ``check_s_mean``, ``check_s_longest``) and those of each call under ``call_`` names, and
the work of the call that wrote it (``chunks_total``, ``chunks_loaded``, ``chunks_computed``, ``rows_computed``, also by head set).

Environment. ``main`` sets ``LOKY_MAX_CPU_COUNT`` (the physical cores) unless the user did, before scikit-learn or joblib is imported:
without it loky looks for ``wmic``, which a current Windows lacks, and prints a warning with a traceback.

Privacy. This runner reads stored match keys for output scanning and error scrubbing. The levels stage also delegates raw-record
lookup by stored key to ``value.kill_levels``; that path scrubs the key, game id and exception cause chain with the helpers of
``store_build``. The JSON output helper scans its rendered text before writing; records use counts, hashes and match rows.


방법
----
이 모듈은 S3의 피해자 레벨·라벨·적합·합성·관문을 연결하며 입력·출력 해시와 단계 완료 기록으로 이어하기를 대조한다.
질문 행은 정규 프레임의 실제 시각이고, 발생은 그 뒤 교전 첫 킬을 센 값이며, 가치는 그 발생에 연결된 골드 단위 결과이다.
실제 관문 계산과 보고용 Brier·포탑·부모/작업자 진단을 구분하고 식은 결과를 보유하는 공개 문서에 둔다(S3 R69).
여기서 소규모 점검 실행(smoke)은 FIT·STOP·DEV의 작은 경기 표본이며, 모형의 식 자체를 바꾸는 별도 실험을 뜻하지 않는다.

식 목록
-------
(RUN3-1) 역할별 정수 올림 표본
(RUN3-2) 관측 프로세스 메모리와 기록 변환
(RUN3-3) E6 검사 범위 계수
(RUN3-4) 머리 점수와 백분위 구간
(RUN3-5) B1 합성 경기 표본
(RUN3-6) 저장 발생의 행별 합
(RUN3-7) 발생 Brier와 짝지은 차이
(RUN3-8) 사건별 포탑 가격 진단
(RUN3-9) 관문 집단과 양의 평균 행 가중치
(RUN3-10) 값이 있는 단위의 요약 비율
(RUN3-11) 합성 호출의 비용·작업량 기록
(RUN3-12) 부모·작업자 발생 예측 차이
(RUN3-13) 단계 한도와 최종 실행 요약

공통 가정
---------
저장소의 행·경기 참조는 0 기반이며 참가자 ID를 다루는 하위 모듈의 1 기반 ID와 다르다.
저장 시각은 ms, 모형 시간 인자는 s로 변환하며, 확률·유한값·행 대응의 검사 범위는 각 함수의 계약을 따른다.
역할 선택과 이어하기 대조는 실행 경로와 상위 API의 계약이며 각 유틸리티가 그 조건을 모두 독립적으로 검사하지는 않는다.
E6는 실제 실행에서 저장소 읽기 전에 확인하고 단계·작업 사이에도 검사하며, 인자 검증은 그보다 앞설 수 있다(S3 Task12).
이 모듈의 개수 회계 관문과 보고 모듈의 골드 회계 확인을 구분한다(S3 as-built 회계 항).
"""
# Ref: docs/superpowers/plans/2026-10-21-c-ept-s3.md Task 12 (the stages, STAGES.json, the smoke run), decisions P7-P11 (the gate and its
#      pinned constants, B = 2,000, the seed, S = 1,000 / 200) and the rulings R11 (accounting is exact on counts), R14 (the FIT-only drop
#      list), R21 (the roles mappings), R27 (gate_pass = all_pass and nothing missing), R29-R32 (the compose workers, the R31 screen, the
#      composer signature) of the S3 ledger.
# Ref: Brier, G. W. (1950), "Verification of forecasts expressed in terms of probability", Monthly Weather Review 78(1): 1-3 (the Brier
#      score of the forecasts of at least one onset within 60 s, M2 against B2).
# Ref: Efron, B. (1979), "Bootstrap methods: another look at the jackknife", Annals of Statistics 7(1): 1-26, and Cameron, A. C., Gelbach,
#      J. B. and Miller, D. L. (2008), Review of Economics and Statistics 90(3): 414-427 (matches are resampled whole; the paired
#      intervals of the Brier differences use the draw of ``nll.match_bootstrap``, so the same matches are drawn for every score).
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import gc
import hashlib
import io
import json
import math
import os
import shutil
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import psutil
except ImportError:      # the peak memory is then not recorded
    psutil = None

from c_ept import config as C
from c_ept import features, nll, runlog, store_core, runtime

# ---- the stages and the constants of the run ---------------------------------------------------------------------------------------
STAGES = ('levels', 'labels', 'fit', 'compose', 'gate')
SCHEMA = 's3_stages_v1'
LOCK_NAME = 'S3.lock'                          # in the run folder: one call at a time
MAX_WORKERS = 61                               # the limit of a process pool on Windows (``compose.MAX_WORKERS``)

LABEL_ROLES = ('FIT', 'STOP', 'DEV')           # the roles stage S3 values (``value.LABEL_ROLES``); CAL and AUDIT are never read
FIT_ROLE, STOP_ROLE, DEV_ROLE = 'FIT', 'STOP', 'DEV'
TIER_ROLES = (FIT_ROLE,)                       # ``TierModel.fit(roles=...)``
B0_ROLES = {'fit': FIT_ROLE}                   # ``B0Heads.fit(roles=...)`` (ruling R21)
B1_ROLES = {'fit': FIT_ROLE, 'stop': STOP_ROLE}    # B1 and ``B2.fit_stores(roles=...)``: the early stopping reads STOP, never DEV

SEED = 20261021                                # the base seed of stage S3 (``compose.DEFAULT_SEED``, ``nll.BOOT_SEED``)
BOOT_B = 2000                                  # match-bootstrap replicates of every interval (``nll.BOOT_B``)
S_B0, S_B1 = 1000, 200                         # simulations of a row: B0 on every DEV row, B1 on its sample (decision P6)
B1_DEV_MATCHES = 500                           # the seeded sample of DEV matches that B1 is composed on
R31_LIMIT = 50.0                               # more expected onsets than this in 180 s on one row: a broken head (ruling R31)
MEMORY_CAP_GIB = 40.0                          # sampled-peak reporting/warning threshold; a recorded stage can exceed it
SMOKE_SHARE = {'FIT': (1, 1), 'STOP': (1, 5), 'DEV': (3, 10)}       # the share of N of each role, as an integer fraction (a ceiling)

# The units of work that the compose methods hand to a worker (``block``), sized for the cost of the E6 check (module text): the closed
# form costs about 0.25 s a row, a B0 simulation (S = 1,000) about 0.03-0.05 s, a B1 simulation (S = 200) about 0.1 s, and the expected
# onsets about a millisecond. The smoke run measures the time of a block and of a check; RUN.json records both.
BLOCK_EXPECTED_ONSETS = 4096
BLOCK_B0_SIMULATE = 512
BLOCK_B0_EXPECT = 64
BLOCK_B1_SIMULATE = 256
CHUNK_ROWS = 8192                              # rows of one saved chunk of the compose stage (a resume keeps the chunks that are saved)
CHUNK_COUNTERS = ('chunks_total', 'chunks_loaded', 'chunks_computed', 'rows_computed')     # what COMPOSE.json counts of the chunks
FILL_CHUNK = 1 << 18                           # FIT views per call of ``features.b0`` when the filled cells are counted
B2_BINS = C.EVAL_H_MS[0] // C.BIN_MS           # the bins 0 .. 5: the first 60 s
PERCENTILES = (2.5, 97.5)

# The files that identify the output of a stage (the SHA-256 of each is in STAGES.json); a name that ends in a slash is a store folder,
# identified by the hashes of its STORE.json. A model is identified by its JSON, which holds the hashes of its array and model files.
STAGE_OUTPUTS = {
    'levels': ('levels/kill_levels.npy', 'levels/LEVELS.json'),
    'labels': ('labels/store/', 'labels/ACCOUNTING.json', 'labels/LABELS_SUMMARY.json'),
    'fit': ('fit/tier/tower_tier.json', 'fit/b0/b0_heads.json', 'fit/b1/b1_heads.json', 'fit/b2/b2.json', 'fit/FIT_INPUTS.json',
            'fit/FIT_FACTS.json', 'fit/DEV_NLL.json', 'fit/DEV_PRED.npz'),
    'compose': ('compose/b0_dev.npz', 'compose/b1_dev.npz', 'compose/COMPOSE.json'),
    'gate': ('gate/GATE.json',),
}


class RunRefused(Exception):
    """A refusal that the caller can put right (a bad option, other inputs than the run was made with, a stage whose earlier stage is not
    complete): ``main`` prints the message and returns 2. The message holds counts, names and hashes, never a match key.

    실행 전제의 위반을 상위 실행기의 거부 경로로 전달한다(S3 Task12).
    """


class StageFailed(RunRefused):
    """A check of a stage failed (the accounting identity, the R31 screen, a pinned constant): the stage is not marked complete.

    단계 결과가 실행 계약을 만족하지 못했음을 나타낸다(S3 Task12).
    """


class RunLocked(RuntimeError):
    """Another call holds ``S3.lock`` of the run folder, or a call that was cut left it behind. It holds no match key.

    같은 실행 폴더에 대한 동시 변경을 잠금 거부로 전달한다(S3 Task12).
    """


# ---- small helpers -----------------------------------------------------------------------------------------------------------------
def _refuse(message: str) -> int:
    print(message, file=sys.stderr, flush=True)
    return 2


def _clean(obj):
    """``obj`` as plain JSON values: numpy numbers and arrays become Python ones and a number that is not finite becomes None."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, os.PathLike):
        return os.fspath(obj)
    return obj


def render(obj) -> str:
    """``obj`` as the text of a JSON file of this module: sorted keys, two-space indent, UTF-8, no NaN.

    배열과 비유한 수를 JSON 값으로 정리하여 기록 형식을 통일한다(S3 Task12).
    """
    return json.dumps(_clean(obj), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n'


def _gib(n_bytes) -> float | None:
    """Bytes as GiB (3 decimals), or None for a falsy n_bytes (including 0, as without psutil).

    식 (RUN3-2)의 해당 항을 계산한다.
    """
    return round(n_bytes / 2 ** 30, 3) if n_bytes else None


def _peaks(where) -> dict:
    """The two peaks that a scope (or the sampler) saw, as every record of the run holds them: ``peak_private_gib``, the largest sampled
    sum of the private memory of the tree, and ``peak_private_upper_gib``, the largest sampled sum of the peak_pagefile of its processes,
    a lifetime bound for the successfully read observed process set, not an unconditional bound for historical children or a stage.
    An unavailable sample does not erase an earlier non-None maximum; an all-unavailable upper record is None.

    식 (RUN3-2)의 해당 항을 계산한다.
    """
    return {'peak_private_gib': _gib(where.peak), 'peak_private_upper_gib': _gib(where.peak_upper)}


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')


def _npz_bytes(arrays: dict) -> bytes:
    """A stored (uncompressed) .npz without time stamps (the writer of the B0 files: the same arrays give the same bytes)."""
    from c_ept.baselines_b0 import _write_npz_bytes
    return _write_npz_bytes(arrays)


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(array), allow_pickle=False)
    return buffer.getvalue()


def _remove_tree(folder: Path, root: Path) -> None:
    """Remove ``folder`` (a partial output of this run) when it lies inside ``root``; anything else is never removed."""
    if not folder.resolve().is_relative_to(root.resolve()) or folder.resolve() == root.resolve():
        raise RuntimeError(f'{folder.name} is not a folder of this run: it is not removed')
    shutil.rmtree(folder)


# ---- memory, time and the E6 check -------------------------------------------------------------------------------------------------
def _tree_memory(processes=None) -> tuple:
    """기존 psutil 교체 지점을 보존하는 공용 트리 측정 어댑터 (RUN3-2)."""
    return runtime.tree_memory(processes, process_api=psutil)


def _tree_private_bytes() -> int:
    """기존 private-byte 투영을 보존한다 (RUN3-2)."""
    return _tree_memory()[0]


_larger = runtime.larger
_Scope = runtime.MemoryScope
_GuardWindow = runtime.GuardWindow


class _Sampler(runtime.ProcessTreeSampler):
    """기존 `_tree_memory` monkeypatch를 생성 시 반영한다 (RUN3-2)."""

    def __init__(self, interval: float = 0.1, measure=None):
        super().__init__(interval, _tree_memory if measure is None else measure)


def _require_no_strata():
    from c_ept import run_s1
    return run_s1.require_no_strata()


class _Guard(runtime.E6Guard):
    """기존 E6 probe를 호출 시 조회하는 호환 어댑터 (RUN3-3)."""

    def __init__(self, *, validate=None):
        super().__init__(_require_no_strata, validate=validate)


class _TickTimer:
    """The ``check=`` of ``B0Heads.fit``: the E6 guard, and the clock and the memory peak of each unit between two calls. ``B0Heads.fit``
    calls its check once before it reads anything and once before each of its 13 heads, so the unit after the first call is the setup and
    the unit after call k + 1 is head k; ``finish()`` closes the last one."""

    def __init__(self, ctx):
        self.ctx, self.segments = ctx, []
        self._cm, self._scope = None, None

    def _close(self) -> None:
        """식 (RUN3-2)의 해당 항을 계산한다."""
        if self._cm is not None:
            self._cm.__exit__(None, None, None)
            self.segments.append({'wall_s': round(self._scope.wall, 3), **_peaks(self._scope)})
            self._cm = self._scope = None

    def __call__(self) -> None:
        self.ctx.guard()
        self._close()
        self._cm = self.ctx.sampler.scope()
        self._scope = self._cm.__enter__()

    def finish(self) -> None:
        self._close()


# ---- the smoke selection -----------------------------------------------------------------------------------------------------------
def select_smoke(s1, n: int, seed: int = SEED) -> dict:
    """The smoke selection of ``n``: ``ceil(share x n)`` matches of each of FIT (n), STOP (ceil(0.2 n)) and DEV (ceil(0.3 n)), chosen by
    a permutation of the matches of the role that is seeded by (seed, role number); the first k of the permutation, sorted, are the
    selection, so the selection of a smaller n lies inside that of a larger one. A role with fewer matches than asked gives all of them.

    Returns ``{'n', 'seed', 'asked': {role: k}, 'matches': {role: [S1 match rows]}, 'selection_sha256'}``; the rows are S1 match numbers
    (``runlog.sha256_json`` of ``matches`` is the hash). RunRefused for a role without a match: early stopping, the labels and the DEV rows
    need all three.

    수식
    ----
    (RUN3-1)  asked_role = ⌈n · num_role / den_role⌉,
             chosen_role = sort(cand_role[perm_role[:asked_role]]),
             perm_role = RNG([seed, role_code]).permutation(#cand_role).

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      n | 요청한 FIT 경기 수 | 경기 | 정수 ≥ 1, bool 제외 | n
      num_role, den_role | 역할별 비율의 분자·분모 | — | FIT 1/1, STOP 1/5, DEV 3/10 | SMOKE_SHARE
      asked_role, chosen_role | 요청 수와 선택된 저장소 경기 행 | 경기; — | 양의 정수; 0 기반 int64 행 | asked, chosen
      cand_role, perm_role | 역할의 경기 행과 그 위치의 순열 | — | 0 기반 | candidates, order
      seed, role_code | 난수 시드와 역할 코드 | — | 난수 생성기가 받는 정수; C.ROLES의 위치 | seed, code

    해석
    ----
    역할별 요청 수는 부동소수점 곱 대신 정수 나눗셈으로 올림한다(S3 as-built의 점검 표본).
    같은 시드와 같은 후보 순서이면 작은 요청의 선택은 큰 요청에 포함된다.
    후보가 요청보다 적으면 있는 경기를 모두 쓰되, 후보가 전혀 없는 역할은 영어 거부 조건을 따른다(S3 Task12).

    근거
    ----
    S3 계획 Task12(점검 표본); S3 as-built 인터페이스 표; S3 판정 R33(역할 가림).
    """
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise RunRefused(f'--smoke must be a whole number of at least 1, got {n!r}')
    role = np.asarray(s1.array('matches/role'))
    asked, chosen = {}, {}
    for name, (num, den) in SMOKE_SHARE.items():
        code = C.ROLES.index(name)
        asked[name] = -(-int(n) * num // den)                       # the ceiling of num / den x n, in integers: no float to round
        candidates = np.flatnonzero(role == code)
        if candidates.size == 0:
            raise RunRefused(f'the S1 store has no match with the role {name}: a smoke run needs FIT, STOP and DEV matches')
        order = np.random.default_rng([int(seed), code]).permutation(candidates.size)
        chosen[name] = np.sort(candidates[order[:asked[name]]]).astype(np.int64).tolist()
    return {'n': int(n), 'seed': int(seed), 'asked': asked, 'matches': chosen, 'selection_sha256': runlog.sha256_json(chosen)}


class SubsetStore:
    """A read-only view of an S1 store in which only the matches of a smoke selection keep their role: ``matches/role`` of every other match
    reads UNUSED, so every function that selects matches by role (and all of S3 does) sees the subset. Everything else is the S1 store
    itself (``array``, ``manifest``, ``maps`` ...).

    소규모 점검 실행에서 선택하지 않은 경기의 역할만 UNUSED로 가리며 다른 배열은 원 저장소에서 읽는다(S3 R33; as-built).
    """

    def __init__(self, base, selection: dict):
        role = np.array(base.array('matches/role'))
        keep = np.zeros(len(role), dtype=bool)
        for rows in selection['matches'].values():
            keep[np.asarray(rows, dtype=np.int64)] = True
        role[~keep] = C.ROLES.index('UNUSED')
        role.setflags(write=False)
        self._base, self._role = base, role

    def array(self, name):
        """역할 열은 가린 사본으로, 나머지 열은 원 저장소에서 제공한다(S3 R33)."""
        return self._role if name == 'matches/role' else self._base.array(name)

    def __getattr__(self, name):            # only what this class does not have: the manifest, the maps, the root ...
        if name.startswith('_'):            # (never look up our own two fields through the base: a copy of the view would recurse)
            raise AttributeError(name)
        return getattr(self._base, name)


# ---- the decision record and the s3a files -----------------------------------------------------------------------------------------
def read_decisions(spec_path) -> dict:
    """The two hashes that ``decisions.S3_prices`` of the decision record holds, ``{'prices_sha256', 'rules_sha256', 'spec_sha256'}`` (the
    last is the hash of the record itself; the record grows, so it is only recorded). RunRefused when the record or the block is missing.

    승인된 S3 가격·실행 결정이 있는지 확인하여 실행의 기준으로 읽는다(S3 Task12).
    """
    path = Path(spec_path)
    try:
        raw = path.read_bytes()
        doc = json.loads(raw.decode('utf-8'))
        block = doc['decisions']['S3_prices']
        prices, rules = block['prices']['sha256'], block['rules']['sha256']
    except OSError as e:
        raise RunRefused(f'the decision record cannot be read ({type(e).__name__}): {path.name}') from None
    except (ValueError, KeyError, TypeError):
        raise RunRefused(f'{path.name} has no decisions.S3_prices with prices.sha256 and rules.sha256') from None
    for name, digest in (('prices', prices), ('rules', rules)):
        if not (isinstance(digest, str) and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest)):
            raise RunRefused(f'decisions.S3_prices.{name}.sha256 of {path.name} is not a SHA-256')
    return {'prices_sha256': prices, 'rules_sha256': rules, 'spec_sha256': hashlib.sha256(raw).hexdigest()}


def check_s3a(s3a, decisions: dict) -> dict:
    """The paths and hashes of the s3a files, after both hashes are compared with the decision record: RunRefused when a file is missing or
    its SHA-256 differs. Returns ``{'run', 'prices', 'rules', 'prices_sha256', 'rules_sha256'}`` (the paths as ``Path``).

    가격과 규칙 파일의 해시를 승인 결정과 대조하여 다른 입력의 실행을 거부한다(S3 Task12).
    """
    s3a = Path(s3a)
    out = {'run': s3a.name}
    for name, rel in (('prices', Path('prices') / 'PRICES.json'), ('rules', Path('rules') / 'RULE_PAYOUTS.json')):
        path = s3a / rel
        if not path.is_file():
            raise RunRefused(f'the s3a run {s3a.name} has no {rel}')
        digest, want = store_core.sha(path), decisions[f'{name}_sha256']
        if digest != want:
            raise RunRefused(f'{rel} of the s3a run {s3a.name} has the SHA-256 {digest[:12]}..., but decisions.S3_prices records '
                             f'{want[:12]}...: the labels are made from the approved prices and rules only')
        out[name], out[f'{name}_sha256'] = path, digest
    return out


# ---- the lock ----------------------------------------------------------------------------------------------------------------------
def _lock_message(path: Path) -> str:
    try:
        held = json.loads(path.read_text(encoding='utf-8'))
        pid, created = held.get('pid'), held.get('created')
    except (OSError, ValueError, AttributeError):
        pid = created = None
    who = f'made {created} by pid {pid}' if pid is not None else 'its content cannot be read'
    hint = ''
    if isinstance(pid, int) and psutil is not None:
        try:
            hint = ', a process with that pid is running now' if psutil.pid_exists(pid) else ', no process with that pid is running now'
        except Exception:
            hint = ''
    return (f'another call holds the lock {path} ({who}{hint}). Only one call may work on a run folder at a time. If none is running '
            '(a power cut or a killed process leaves the lock behind), delete that file yourself and resume the run; this command never '
            'deletes it.')


def _acquire_lock(run: Path) -> Path:
    """Create ``<run>/S3.lock`` exclusively (the system lets one process win) and return its path; RunLocked when it is there. The file
    holds the pid and the time and nothing that names a match. (``store_build`` and ``flowprice`` have the same lock under their own
    names and messages.)"""
    run.mkdir(parents=True, exist_ok=True)
    path = run / LOCK_NAME
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, 'O_BINARY', 0))
    except FileExistsError:
        raise RunLocked(_lock_message(path)) from None
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as f:
            f.write(json.dumps({'pid': os.getpid(), 'created': _now()}) + '\n')
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def _release_lock(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError as e:
        print(f'WARNING: the lock {path} could not be removed ({type(e).__name__}); delete it before the next call on this run',
              file=sys.stderr, flush=True)


# ---- the context of a call ---------------------------------------------------------------------------------------------------------
@dataclass
class Context:
    """What the stages of one call share. ``s1`` is the S1 store the stages read (the store itself, or the ``SubsetStore`` of a smoke run),
    ``s2`` the S2 store, ``prices`` the ``{group: value}`` dict and ``rules`` the ``rulepay.RuleTable`` of the approved s3a run. ``hashes``:
    ``s1_files_sha256``, ``s2_files_sha256``, ``prices_sha256``, ``rules_sha256`` and ``smoke_sha256`` (None for a full run). ``records``
    replaces ``reader.read_match`` in ``value.kill_levels`` (tests); None is the real reader.

    수식
    ----
    (RUN3-2)  used_sample = Σ_{proc∈read_set} (private_proc or rss_proc),
             upper_sample = Σ_{proc∈read_set} lifetime_proc
                 (read_set ≠ ∅이고 모두 peak_pagefile을 보고할 때), 그 밖에는 None.
             peak = max(0, 범위 안 used_sample),
             peak_upper = max(범위 안 None이 아닌 upper_sample), 없으면 None.
             gib(bytes) = round(bytes / 2^30, 3) (bytes가 참인 값), 그 밖에는 None.
             wall = perf_counter_end − perf_counter_start.

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      proc, read_set | 한 표본에서 memory_info를 읽은 프로세스와 그 집합 | — | 현재 열거·읽기에 성공한 대상 | p, processes, read
      private_proc, rss_proc, lifetime_proc | private, RSS, 수명 전체 peak_pagefile | byte | 보고된 비음수 값, private가 없거나 0이면 RSS | mi.private, mi.rss, peak
      used_sample, upper_sample | 표본 합과 수명 최대값의 합 | byte | ≥ 0; ≥ 0 또는 None | _tree_memory 반환값
      peak, peak_upper | 범위에 들어온 표본의 최대값 | byte | ≥ 0; ≥ 0 또는 None | sampler/scope.peak, .peak_upper
      bytes, gib(bytes) | 기록할 바이트와 반올림한 메모리 | byte; GiB | 0이면 None | _gib, _peaks
      wall | 범위를 열고 닫은 경과 시간 | s | 비음수, 단조 시계 가정 | scope.wall

    해석
    ----
    이 문맥이 보유한 표본기와 범위 객체는 관측된 프로세스의 메모리와 경과 시간을 기록한다(S3 G-S3b B3).
    수명 최대값의 합은 성공적으로 읽은 프로세스 집합의 private commit에 대한 상한이며, 누락되거나 종료된 자식의 과거 메모리는 보장하지 않는다(S3 R79).
    범위 이전의 수명 이력이 포함될 수 있으므로 범위 자체의 최대 메모리로 해석하지 않는다.
    나중 표본의 상한이 None이어도 앞서 얻은 최대값은 유지하고, psutil이 없으면 표본 합은 0·상한은 None이다(S3 R79).
    한도 판단은 수명 상한이 아닌 표본 최대값을 사용한다(S3 R38, R43).

    근거
    ----
    S3 G-S3b B3 메모리 항; S3 판정 R38·R43(표본 한도), R79(관측 범위).

    수식
    ----
    (RUN3-3)  check_count = calls_end − calls_start, check_time = seconds_end − seconds_start,
             check_mean = round(check_time / check_count, 4) (check_count > 0), 그 밖에는 None,
             check_total = round(check_time, 4), check_longest = round(max(0, 범위 안 spent), 4).
             실행 호출의 check_mean은 round(seconds / max(1, calls), 4)로 기록한다.

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      calls_start, calls_end, check_count | 범위 앞뒤 누적 검사 수와 차 | 회 | 비음수 정수 | _calls0, _calls1, calls
      seconds_start, seconds_end, check_time | 누적 검사 시간과 차 | s | 비음수 | _seconds0, _seconds1, seconds
      spent, check_mean, check_total, check_longest | 각 검사 시간·평균·합·최대 | s | 비음수, 범위 평균은 None 가능 | _Guard, _GuardWindow.counters

    해석
    ----
    검사 호출은 실패하더라도 finally에서 시간과 횟수를 누적하므로, 검사 비용과 실제 계산 시간을 구분할 수 있다(S3 Task12, G-S3b B3).
    닫힌 범위는 종료 시점의 누적값을 고정하고, 아직 열린 범위는 조회 시점까지의 값을 쓴다.
    범위에 검사가 없으면 평균은 None이지만 실행 호출 전체의 기록은 분모를 보정하여 0으로 남긴다.

    근거
    ----
    S3 계획 Task12(E6 기록); S3 G-S3b B3 검사 계수 항; _run 호출 기록.
    """
    run: Path
    s1: Any
    s2: Any
    prices: dict
    rules: Any
    hashes: dict
    smoke: dict | None = None
    workers: int = 8
    records: Any = None
    guard: _Guard = field(default_factory=_Guard)
    sampler: _Sampler = field(default_factory=_Sampler)
    out: Any = print
    cached_keys: Any = field(default=None, init=False, repr=False)
    stage: str | None = field(default=None, init=False)             # the stage that ``run_stages`` is running ...
    inputs_sha256: str | None = field(default=None, init=False)     # ... and the hash of its inputs (the one STAGES.json records)

    def say(self, message: str) -> None:
        """실행 메시지를 지정 출력으로 즉시 전달한다(S3 Task12)."""
        self.out(message, flush=True)

    @property
    def compose_workers(self) -> int:
        """The compose stage computes every value in spawned workers (ruling R30): at least two, whatever ``--workers`` says.

        부모가 합성값을 계산하지 않도록 작업자 수의 하한을 적용한다(S3 R30).
        """
        return max(2, self.workers)

    def keys(self) -> list:
        """The match keys of the S1 store as text (read once; used only to scan what this module writes and to scrub a message).

        출력 검사와 오류 문구 가림에 쓸 식별자를 한 번 읽어 보유한다(S3 Task12 개인정보 규칙).
        """
        if self.cached_keys is None:
            self.cached_keys = [k.decode('ascii') for k in np.asarray(self.s1.array('matches/key')).tolist() if k]
        return self.cached_keys

    def assert_no_keys(self, text: str) -> None:
        """RuntimeError when the key of any S1 match occurs in ``text`` (as ``report_s2.assert_no_keys``): the number of hits is named,
        never a key.

        출력 문구에서 저장소 식별자가 발견되면 내용 대신 개수만 포함한 오류로 거부한다(S3 Task12).
        """
        hits = sum(1 for k in self.keys() if k in text)
        if hits:
            raise RuntimeError(f'{hits} match keys occur in the text that was to be written')

    def write_json(self, path: Path, obj) -> str:
        """Write ``obj`` as the JSON file ``path`` (atomically, after the keys were looked for in its text) and return its SHA-256.

        문구의 식별자 검사를 통과한 JSON을 원자적 쓰기에 넘기고 파일 해시를 반환한다(S3 Task12).
        """
        text = render(obj)
        self.assert_no_keys(text)
        path.parent.mkdir(parents=True, exist_ok=True)
        store_core.write_atomic(path, text)
        return store_core.sha(path)

    def write_npz(self, path: Path, arrays: dict) -> None:
        """수치 배열을 압축 기록으로 묶어 원자적 쓰기에 넘긴다(S3 Task12)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        store_core.write_atomic(path, _npz_bytes(arrays))


def identity(run: Path, rel: str) -> str:
    """The SHA-256 that stands for the output ``rel`` of a stage: the file's own, or for a store folder (``rel`` ends in a slash) the hash
    of the recorded hashes of its files (``STORE.json``). RunRefused when the output is not there.

    일반 파일은 내용 해시를, 저장소 폴더는 기록된 파일 해시 사전의 해시를 반환한다(S3 Task12).
    """
    path = run / rel
    if rel.endswith('/'):
        if not (path / 'STORE.json').is_file():
            raise RunRefused(f'the output {rel} of this run is missing')
        return runlog.sha256_json(store_core.Store(path).manifest['files'])
    if not path.is_file():
        raise RunRefused(f'the output {rel} of this run is missing')
    return store_core.sha(path)


def stage_outputs(run: Path, stage: str) -> dict:
    """``{output: SHA-256}`` of the files that identify ``stage`` (``STAGE_OUTPUTS``), read from the run folder now.

    현재 단계의 출력 식별 파일들을 해시 사전으로 모은다(S3 Task12).
    """
    return {rel: identity(run, rel) for rel in STAGE_OUTPUTS[stage]}


def stage_inputs(ctx: Context, stage: str) -> dict:
    """What the stage ``stage`` is made from, as hashes: the stores and the smoke selection, the s3a files (labels, composition), the output
    files of the stage before, and the constants that decide the numbers (the roles, S, the seed, B). The SHA-256 of this dict is the input
    hash that STAGES.json records. The output files of the earlier stages are hashed from the disk now, so a file that changed (or went
    away) since is seen. Block sizes and worker counts decide no value and are not in it.

    단계가 사용하는 입력 해시와 수치 결정 상수를 모아 이어하기 대조에 제공한다(S3 Task12; G-S3b B3).
    """
    h = ctx.hashes
    base = {'s1_files_sha256': h['s1_files_sha256'], 'smoke_sha256': h['smoke_sha256']}
    if stage == 'levels':
        return {**base, 'roles': list(LABEL_ROLES)}
    levels = identity(ctx.run, 'levels/kill_levels.npy')
    if stage == 'labels':
        return {**base, 'prices_sha256': h['prices_sha256'], 'rules_sha256': h['rules_sha256'], 'kill_levels_sha256': levels,
                'roles': list(LABEL_ROLES)}
    labels = identity(ctx.run, 'labels/store/')
    s2 = {'s2_files_sha256': h['s2_files_sha256']}
    if stage == 'fit':
        return {**base, **s2, 'labels_store': labels, 'roles': {'tier': list(TIER_ROLES), 'b0': B0_ROLES, 'b1': B1_ROLES, 'b2': B1_ROLES}}
    fit_files = (('tier', 'fit/tier/tower_tier.json'), ('b0', 'fit/b0/b0_heads.json'), ('b1', 'fit/b1/b1_heads.json'),
                 ('b2', 'fit/b2/b2.json'), ('dev_pred', 'fit/DEV_PRED.npz'))
    fit = {name: identity(ctx.run, rel) for name, rel in fit_files}
    if stage == 'compose':
        return {**base, **s2, 'labels_store': labels, 'fit': fit, 'prices_sha256': h['prices_sha256'],
                'constants': {'S_b0': S_B0, 'S_b1': S_B1, 'seed': SEED, 'b1_dev_matches': B1_DEV_MATCHES}}
    compose = {name: identity(ctx.run, rel) for name, rel in (('b0', 'compose/b0_dev.npz'), ('b1', 'compose/b1_dev.npz'))}
    if stage == 'gate':
        return {**base, 'labels_store': labels, 'fit': fit, 'compose': compose, 'prices_sha256': h['prices_sha256'],
                'constants': {'B': BOOT_B, 'seed': SEED, 'S_b0': S_B0}}
    raise ValueError(f'unknown stage {stage!r}')


# ---- the stage ledger (STAGES.json) ------------------------------------------------------------------------------------------------
class Ledger:
    """``<run>\\STAGES.json``: for every stage whether it is complete, the hashes of its inputs, the SHA-256 of its outputs, its time and
    its peak memory, and one record for each call of the runner. Written atomically after every change.

    단계와 호출의 기록을 보유하고 원자적 쓰기로 갱신한다(S3 Task12).
    완료 단계의 불변성 검사는 바깥 이어하기 검증이 담당한다.
    """

    def __init__(self, run: Path):
        self.path = run / 'STAGES.json'
        if self.path.is_file():
            self.doc = json.loads(self.path.read_text(encoding='utf-8'))
            if self.doc.get('schema') != SCHEMA:
                raise RunRefused(f'{self.path.name} has the schema {self.doc.get("schema")!r}, not {SCHEMA!r}')
        else:
            self.doc = {'schema': SCHEMA, 'stages': {}, 'calls': []}

    def save(self) -> None:
        """현재 장부를 JSON으로 기록한다(S3 Task12)."""
        store_core.write_atomic(self.path, render(self.doc))

    def stage(self, name: str) -> dict | None:
        """단계 기록을 조회하며 아직 없으면 None을 반환한다(S3 Task12)."""
        return self.doc['stages'].get(name)

    def complete(self, name: str) -> bool:
        """해당 단계 기록의 완료 표시를 확인한다(S3 Task12)."""
        rec = self.stage(name)
        return bool(rec and rec.get('complete'))

    def begin(self, name: str, inputs: dict, digest: str) -> None:
        """Start the stage ``name``, or start it again. RunRefused when the stage was begun by an earlier call that did not finish it and on
        other inputs (the recorded ``inputs_sha256`` is not ``digest``): what that call left in the run folder (a model, a chunk of rows)
        was made from those inputs. This incomplete-stage digest check relies on the outer resume code, input and output checks for the
        full no-mixing contract; a direct call may replace a completed record.

        미완료 단계의 기존 digest만 이번 인자와 대조하며, digest 계산은 호출자가 맡는다(S3 G-S3b B3).
        직접 호출하면 완료 기록도 새 미완료 기록으로 덮으므로 전체 이어하기 검증은 완료 기록·코드·입출력 대조를 포함하는 바깥 경로와 함께 사용한다.
        """
        old = self.doc['stages'].get(name)
        if old and not old.get('complete') and old.get('inputs_sha256') != digest:
            before = old.get('inputs') or {}
            changed = sorted(key for key in set(inputs) | set(before) if inputs.get(key) != before.get(key))
            raise RunRefused(f'the stage {name} was begun by an earlier call on other inputs than this call has (changed: '
                             f'{", ".join(changed) or "the hash only"}): what that call left in this run folder was made from them, and '
                             'a run folder never mixes two sets of inputs. Start a new run folder.')
        self.doc['stages'][name] = {'complete': False, 'started': _now(), 'inputs': inputs, 'inputs_sha256': digest}
        self.save()

    def finish(self, name: str, outputs: dict, **facts) -> None:
        """이미 시작한 단계에 출력 해시와 완료 사실을 기록한다(S3 Task12)."""
        rec = self.doc['stages'][name]
        rec.update(complete=True, finished=_now(), outputs=outputs, **facts)
        self.save()

    def begin_call(self, record: dict) -> dict:
        """실행 호출 기록을 장부에 추가하고 같은 기록 객체를 반환한다(S3 Task12)."""
        self.doc['calls'].append(record)
        self.save()
        return record


def check_ledger(ctx: Context, ledger: Ledger) -> None:
    """RunRefused (and nothing is run) unless every stage that STAGES.json calls complete still has the inputs and the outputs it was made
    with, has its result in RUN.json (``<stage>_result``: a stage is marked complete only after its result is recorded), and no stage is
    complete after one that is not. The hash of an input that is a file of an earlier stage is read from the disk.

    완료 단계가 순서의 앞부분을 이루는지와 현재 입력·출력 해시 및 결과 기록을 대조한다(S3 Task12; G-S3b B3).
    미완료 단계의 재시작 입력 검사는 Ledger.begin이 맡는다.
    """
    earlier_open, manifest = None, None
    for stage in STAGES:
        if not ledger.complete(stage):
            earlier_open = earlier_open or stage
            continue
        if earlier_open:
            raise RunRefused(f'STAGES.json calls {stage} complete but {earlier_open} is not: the record of this run is not consistent')
        if manifest is None:
            manifest = json.loads((ctx.run / 'RUN.json').read_text(encoding='utf-8'))
        if f'{stage}_result' not in manifest:
            raise RunRefused(f'STAGES.json calls {stage} complete but RUN.json has no {stage}_result: the record of this run is not '
                             'consistent, so start a new run folder')
        rec = ledger.stage(stage)
        now = stage_inputs(ctx, stage)
        if runlog.sha256_json(now) != rec['inputs_sha256']:
            changed = sorted(k for k in set(now) | set(rec['inputs']) if now.get(k) != rec['inputs'].get(k))
            raise RunRefused(f'the inputs of the complete stage {stage} are not what it was made from (changed: {", ".join(changed)}): '
                             'a run folder is never changed, so start a new run folder')
        outputs = stage_outputs(ctx.run, stage)
        if outputs != rec['outputs']:
            changed = sorted(k for k in set(outputs) | set(rec['outputs']) if outputs.get(k) != rec['outputs'].get(k))
            raise RunRefused(f'the outputs of the complete stage {stage} changed since it finished ({", ".join(changed)}): '
                             'a run folder is never changed, so start a new run folder')


# ---- stage: levels -----------------------------------------------------------------------------------------------------------------
def stage_levels(ctx: Context) -> dict:
    """The victim level of every kill of FIT, STOP and DEV (``value.kill_levels``; the raw files, ``reader.read_match`` only). An unreadable
    file fails the stage: ``unit_values`` would refuse the kills without a level anyway.

    원시 읽기 경로에서 피해자 레벨을 만들고 읽지 못한 파일이 있으면 라벨 단계로 넘기지 않는다(S3 Task12).
    """
    from c_ept import value
    out = ctx.run / 'levels'
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    extra = {} if ctx.records is None else {'records': ctx.records}
    levels = value.kill_levels(ctx.s1, roles=LABEL_ROLES, workers=ctx.workers, report=report, **extra)
    levels = np.asarray(levels)
    if report.get('unreadable'):
        raise StageFailed(f'{report["unreadable"]} of {report.get("matches")} raw match files could not be read: their kills have no level')
    path = out / 'kill_levels.npy'
    store_core.write_atomic(path, _npy_bytes(levels))
    known = levels[levels >= 0]
    facts = {'kill_levels_sha256': store_core.sha(path), 'dtype': str(levels.dtype), 'n_kills': int(levels.size),
             'n_valued_kills': int(known.size), 'level_min': int(known.min()) if known.size else None,
             'level_max': int(known.max()) if known.size else None, 'roles': list(LABEL_ROLES), 'report': report}
    ctx.write_json(out / 'LEVELS.json', facts)
    return facts


# ---- stage: labels -----------------------------------------------------------------------------------------------------------------
def stage_labels(ctx: Context) -> dict:
    """The label store: ``unit_values`` -> ``v_start`` -> ``accounting`` (must be exactly 0, ruling R11) -> ``gold_bin_means`` ->
    ``write_labels``. ``LABELS.json`` (the provenance) is scanned for match keys before the store is written.

    수식
    ----
    (RUN3-10)  share_flag = count_flag / valued_count (valued_count > 0), 그 밖에는 None,
             count_flag = Σ_{unit_idx: valued} 1[flag_unit_idx],
             value_row_count = Σ_row 1[isfinite(v_start_{row,0})].

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      unit_idx, row | 교전 단위 행과 프레임 행 | — | 0 기반 | uv, v
      valued_count, count_flag, share_flag | 평가한 단위 수·표시 수·비율 | 개; 개; — | ≥ 0; 비율은 None 가능 | n_valued, _share, summary
      flag_unit_idx | 파랑 승리·빨강 승리·무승부 또는 확실성 표시 | — | winner = 1, −1, 0; sure | uv.winner, uv.sure
      v_start_{row,0}, value_row_count | 첫 지평 가치와 유한 행 수 | 골드; 행 | 저장 가치 계약; ≥ 0 | v[:, 0], n_rows_with_value

    해석
    ----
    가치와 회계를 계산한 뒤, 값이 있는 단위만 분모로 삼아 요약한다(S3 Task12).
    개수 회계 오차가 정확히 0이 아니면 저장 전에 단계가 실패하며, 골드 오차는 별도 필드로 보고한다(S3 R11; as-built 회계 항).
    평가 단위가 없으면 비율은 None이고 골드 구간 대표값의 대체 여부는 반환된 fallback 목록으로 남긴다.

    근거
    ----
    S3 계획 Task12(라벨); S3 판정 R11(개수 회계); S3 as-built 회계 항.
    """
    from c_ept import value
    out = ctx.run / 'labels'
    out.mkdir(parents=True, exist_ok=True)
    levels = np.load(ctx.run / 'levels' / 'kill_levels.npy', allow_pickle=False)
    s1 = ctx.s1
    uv = value.unit_values(s1, ctx.prices, ctx.rules, levels, roles=LABEL_ROLES)
    v = value.v_start(s1, uv)
    acc = value.accounting(s1, uv, ctx.prices, ctx.rules, levels, roles=LABEL_ROLES)
    error = float(acc['max_abs_error'])
    if error != 0.0:
        raise StageFailed(f'the accounting identity is not exact: max_abs_error is {error!r}, not 0 (ruling R11); the labels are not '
                          'written')
    reps = value.gold_bin_means(uv, value.role_units(s1, (FIT_ROLE,)))
    provenance = {'prices_sha256': ctx.hashes['prices_sha256'], 'rules_sha256': ctx.hashes['rules_sha256'],
                  's1_files_sha256': ctx.hashes['s1_files_sha256'],
                  'kill_levels_sha256': store_core.sha(ctx.run / 'levels' / 'kill_levels.npy'), 'code_state': runlog.code_state(),
                  'gold_bin_fallback': [int(b) for b in reps.fallback], 'roles': list(LABEL_ROLES),
                  'smoke_sha256': ctx.hashes['smoke_sha256']}
    ctx.assert_no_keys(render(provenance))                  # the text that write_labels will write: a key in it is an error
    store_dir = out / 'store'
    if store_dir.exists():                                  # a store that a cut left half written (it has no finished STORE.json)
        _remove_tree(store_dir, out)
    value.write_labels(store_dir, s1, uv, v, reps.means, provenance)
    valued = np.asarray(uv.valued)
    winner = np.asarray(uv.winner)[valued]
    n_valued = int(valued.sum())
    summary = {'n_units': int(valued.size), 'n_valued_units': n_valued, 'n_rows': int(len(v)),
               'n_rows_with_value': int(np.isfinite(v[:, 0]).sum()),
               'winner_blue': _share((winner == 1).sum(), n_valued), 'winner_red': _share((winner == -1).sum(), n_valued),
               'draw': _share((winner == 0).sum(), n_valued), 'sure': _share(np.asarray(uv.sure)[valued].sum(), n_valued),
               'gold_bin_means': reps.means, 'gold_bin_fallback': [int(b) for b in reps.fallback]}
    ctx.write_json(out / 'ACCOUNTING.json',
                   {k: acc[k] for k in ('max_abs_error', 'max_abs_value_error', 'n_matches', 'n_valued_matches', 'total')})
    ctx.write_json(out / 'LABELS_SUMMARY.json', summary)
    return {'max_abs_error': error, 'max_abs_value_error': float(acc['max_abs_value_error']),
            **{k: val for k, val in summary.items() if k != 'gold_bin_means'}}


def _share(count, total: int) -> float | None:
    """식 (RUN3-10)의 해당 항을 계산한다."""
    return float(count) / total if total else None


def open_labels(ctx: Context):
    """The label store of this run, complete and with every file at its recorded hash (RunRefused otherwise).

    앞 단계의 라벨 저장소를 읽어 후속 단계에 제공한다(S3 Task12).
    """
    root = ctx.run / 'labels' / 'store'
    store = store_core.Store(root)
    if not store.manifest.get('complete'):
        raise RunRefused('the label store of this run is not complete')
    try:
        store.verify()
    except ValueError as e:
        raise RunRefused(str(e)) from None
    return store


# ---- stage: fit --------------------------------------------------------------------------------------------------------------------
class _ProgressLog:
    """``fit\\PROGRESS.jsonl``: one line for each unit of the fit (a model, a head, the DEV scores) when it is finished, appended and
    fsynced. The facts of a model that a resumed call loads instead of fitting come from here. Every line also records the code state
    (``code_commit`` and ``code_dirty``, from ``runlog.code_state()`` at the time of the line): the log says which code made which model."""

    def __init__(self, path: Path):
        self.path = path

    def add(self, unit: str, **facts) -> None:
        line = json.dumps(_clean({'t': _now(), 'unit': unit, **runlog.code_state(), **facts}), sort_keys=True, allow_nan=False)
        with open(self.path, 'a', encoding='utf-8', newline='\n') as f:
            f.write(line + '\n')
            f.flush()
            os.fsync(f.fileno())

    def last(self, unit: str) -> dict | None:
        found = None
        if self.path.is_file():
            for line in self.path.read_text(encoding='utf-8').splitlines():
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue                                     # a line that a cut left half written
                if entry.get('unit') == unit:
                    found = entry
        return found


def _b0_fill_counts(ctx: Context, views: np.ndarray, means: np.ndarray) -> np.ndarray:
    """How many cells of each of the 39 b0 columns the FIT means fill, over the FIT views of the three sources (ruling R2)."""
    fill = np.zeros(features.N_B0, dtype=np.int64)
    for lo in range(0, len(views), FILL_CHUNK):
        features.b0(ctx.s2, views[lo:lo + FILL_CHUNK], means, fill_counts=fill)
        ctx.guard()
    return fill


def _fit_inputs(ctx: Context, out: Path) -> dict:
    """The FIT views, the columns to drop (ruling R14: judged by the FIT views alone), the 39 FIT means of b0 and the cells they fill: made
    once and kept in ``fit\\FIT_INPUTS.json`` (a resumed call reads them)."""
    path = out / 'FIT_INPUTS.json'
    if path.is_file():
        return json.loads(path.read_text(encoding='utf-8'))
    views = np.flatnonzero(features.split_views(ctx.s1, ctx.s2, FIT_ROLE))
    ctx.guard()
    dropped = features.dropped_columns(ctx.s2, views)
    ctx.guard()
    means = features.b0_fit_means(ctx.s2, views)
    ctx.guard()
    fill = _b0_fill_counts(ctx, views, means)
    role = np.asarray(ctx.s1.array('matches/role'))
    doc = {'n_fit_views': int(len(views)), 'n_fit_matches': int((role == C.ROLES.index(FIT_ROLE)).sum()), 'dropped': list(dropped),
           'n_dropped': len(dropped), 'b0_names': list(features.B0_NAMES), 'b0_fit_means': means, 'b0_fill_counts': fill,
           'fill_counts_over': 'the FIT views of the frame-clock, M3 and M4 sources'}
    ctx.write_json(path, doc)
    return doc


def _model_saved(ctx: Context, folder: Path, json_name: str, root: Path) -> bool:
    """True when a complete model is saved in ``folder``. Every ``save`` writes its JSON last, so a folder without it (an NPZ alone, or the
    model files of a booster set) is a save that was cut: it counts as incomplete, is removed and the model is fitted again."""
    if (folder / json_name).is_file():
        return True
    if folder.exists():
        _remove_tree(folder, root)
        ctx.say(f'  {folder.name}: an incomplete save was found and removed; the model is fitted again')
    return False


def _head_names(kinds=None) -> list:
    from c_ept.head_rows import HEAD_NAMES, OPEN_HEAD_NAMES
    return [(kind, name) for kind, names in (('masked', HEAD_NAMES), ('open', OPEN_HEAD_NAMES)) for name in names
            if kinds is None or kind in kinds]


def _fit_tier(ctx: Context, out: Path, log: _ProgressLog) -> tuple:
    from c_ept import towertier
    folder = out / 'tier'
    if _model_saved(ctx, folder, towertier.JSON_NAME, out):
        model = towertier.TierModel.load(folder)
        return model, {**(log.last('tier') or {}), 'loaded': True, 'summary': model.summary()}
    with ctx.sampler.scope() as scope:
        model = towertier.TierModel.fit(ctx.s1, roles=TIER_ROLES, check=ctx.guard)
    model.save(folder)
    facts = {'wall_s': round(scope.wall, 3), **_peaks(scope), 'summary': model.summary(), 'roles': list(TIER_ROLES)}
    log.add('tier', **facts)
    ctx.say(f'  tier law: {facts["wall_s"]} s, converged {model.converged_}')
    return model, {**facts, 'loaded': False}


def _fit_b0(ctx: Context, out: Path, log: _ProgressLog, labels, means: np.ndarray) -> tuple:
    from c_ept import baselines_b0
    folder = out / 'b0'
    if _model_saved(ctx, folder, baselines_b0.JSON_NAME, out):
        model = baselines_b0.B0Heads.load(folder)
        return model, {**(log.last('b0') or {}), 'loaded': True, 'summary': model.summary()}
    timer = _TickTimer(ctx)
    with ctx.sampler.scope() as scope:
        try:
            model = baselines_b0.B0Heads.fit(ctx.s1, ctx.s2, labels, means, roles=B0_ROLES, check=timer)
        finally:
            timer.finish()                                    # the unit that was running when a check refused or a head failed ends too
    model.save(folder)
    names = _head_names()
    if len(timer.segments) == 1 + len(names):                # the setup, then the 13 heads in the order of HEAD_NAMES and OPEN_HEAD_NAMES
        heads = {f'{kind}.{name}': seg for (kind, name), seg in zip(names, timer.segments[1:])}
    else:
        heads = {'unmapped': timer.segments}
    facts = {'wall_s': round(scope.wall, 3), **_peaks(scope), 'setup': timer.segments[0] if timer.segments else None,
             'heads': heads, 'summary': model.summary(), 'roles': dict(B0_ROLES)}
    log.add('b0', **facts)
    ctx.say(f'  B0: {facts["wall_s"]} s, peak {facts["peak_private_gib"]} GiB')
    return model, {**facts, 'loaded': False}


def _fit_identity(ctx: Context) -> dict:
    """What a checkpoint of the fit stage records and a resume compares it with: the code state and the hash of the inputs of the fit
    stage (the one STAGES.json holds; computed from the run folder when the stage was not started by ``run_stages``)."""
    digest = ctx.inputs_sha256 if ctx.stage == 'fit' and ctx.inputs_sha256 else runlog.sha256_json(stage_inputs(ctx, 'fit'))
    return {**runlog.code_state(), 'inputs_sha256': digest}


def _check_head_record(kind: str, name: str, record, identity: dict) -> None:
    """RunRefused when the checkpoint of the B1 head ``kind`` ``name`` was made with other code or other inputs than this call: a run
    folder never mixes models of two codes, so the head is not fitted again behind the user's back."""
    have = {key: (record or {}).get(key) for key in identity}
    differ = sorted(key for key in identity if have[key] != identity[key])
    if differ:
        shown = ', '.join(f'{key} (saved {str(have[key])[:12]}, now {str(identity[key])[:12]})' for key in differ)
        raise RunRefused(f'the saved B1 head {kind}.{name} was fitted with other {shown} than this call has: a run folder never mixes the '
                         'models of two codes or two inputs. Start a new run folder.')


def _fit_b1(ctx: Context, out: Path, log: _ProgressLog, labels, dropped: tuple) -> tuple:
    """B1, head by head: every head is saved to ``b1_heads\\<kind>.<name>\\`` (its JSON last) as soon as it is fitted, with the code state
    and the input hash of the stage as its record. A resumed call loads the heads whose JSON is there (RunRefused when the record of one
    is not the identity of this call) and fits the rest; ``b1\\`` is then made from the 13 heads."""
    from c_ept import baselines_b1
    folder = out / 'b1'
    if _model_saved(ctx, folder, baselines_b1.JSON_NAME, out):
        model = baselines_b1.B1Heads.load(folder)
        heads = {f'{kind}.{name}': log.last(f'b1.{kind}.{name}') for kind, name in _head_names()}
        return model, {**(log.last('b1') or {}), 'loaded': True, 'heads': heads, 'summary': model.summary()}
    identity, columns, root = _fit_identity(ctx), features.b1_columns(dropped), out / 'b1_heads'
    fit_rows = baselines_b1.HeadRows(ctx.s1, ctx.s2, labels, dropped, FIT_ROLE)
    stop_rows = baselines_b1.HeadRows(ctx.s1, ctx.s2, labels, dropped, STOP_ROLE)        # the early stopping reads STOP and never DEV
    fitted, heads = {'masked': {}, 'open': {}}, {}
    with ctx.sampler.scope() as total:
        for kind, name in _head_names():                      # M2Rate first: the largest design shows its peak and its time early
            where = root / f'{kind}.{name}'
            if _model_saved(ctx, where, baselines_b1.HEAD_JSON, root):          # a checkpoint of an earlier call (a cut one is removed)
                ctx.guard()
                record = baselines_b1.read_head_record(where)
                _check_head_record(kind, name, record, identity)
                head = baselines_b1.load_head(where, kind, name, columns)
                unit = {'wall_s': record.get('wall_s'), 'peak_private_gib': record.get('peak_private_gib'),
                        'peak_private_upper_gib': record.get('peak_private_upper_gib'), 'loaded': True}
                ctx.say(f'  B1 {kind} {name}: loaded from its checkpoint')
            else:
                with ctx.sampler.scope() as scope:
                    head = baselines_b1.fit_head(fit_rows, stop_rows, name, kind, check=ctx.guard)
                measured = {'wall_s': round(scope.wall, 3), **_peaks(scope)}
                unit = {**measured, 'loaded': False}
                log.add(f'b1.{kind}.{name}', **measured, best_iteration=head.summary().get('best_iteration'),
                        rounds_run=head.summary().get('rounds_run'))               # the log line first: a checkpoint always has its line
                baselines_b1.save_head(head, kind, name, columns, where, record={**identity, **measured})
                ctx.say(f'  B1 {kind} {name}: {unit["wall_s"]} s, peak {unit["peak_private_gib"]} GiB')
            fitted[kind][name] = head
            heads[f'{kind}.{name}'] = {**unit, 'summary': head.summary()}
    del fit_rows, stop_rows
    gc.collect()
    model = baselines_b1.B1Heads(fitted['masked'], fitted['open'], dropped, (FIT_ROLE,), (STOP_ROLE,))
    model.save(folder)
    facts = {'wall_s': round(total.wall, 3), **_peaks(total), 'heads': heads, 'roles': dict(B1_ROLES)}
    log.add('b1', **{k: v for k, v in facts.items() if k != 'heads'})
    return model, {**facts, 'loaded': False, 'summary': model.summary()}


def _fit_b2(ctx: Context, out: Path, log: _ProgressLog, dropped: tuple) -> tuple:
    from c_ept import baselines_b1
    folder = out / 'b2'
    if _model_saved(ctx, folder, baselines_b1.B2_JSON_NAME, out):
        model = baselines_b1.B2.load(folder)
        return model, {**(log.last('b2') or {}), 'loaded': True, 'summary': model.summary()}
    with ctx.sampler.scope() as scope:
        model = baselines_b1.B2.fit_stores(ctx.s1, ctx.s2, dropped, roles=B1_ROLES, check=ctx.guard)
    model.save(folder)
    facts = {'wall_s': round(scope.wall, 3), **_peaks(scope), 'roles': dict(B1_ROLES)}
    log.add('b2', **facts)
    ctx.say(f'  B2: {facts["wall_s"]} s')
    return model, {**facts, 'loaded': False, 'summary': model.summary()}


def _percentile_ci(replicates) -> list:
    """식 (RUN3-4)의 해당 항을 계산한다."""
    reps = np.asarray(replicates, dtype=np.float64)
    if reps.size == 0 or np.isnan(reps).any():
        return [None, None]
    lo, hi = np.percentile(reps, PERCENTILES)
    return [float(lo), float(hi)]


def score_summary(scores, B: int = BOOT_B, seed: int = SEED) -> dict:
    """The DEV score of one head (``head_rows.HeadScores``): the match-weighted mean NLL with its percentile interval over ``B``
    match-bootstrap replicates, the evaluation rows and the rows that were left out (no zone, no exposure). A head with no row has no
    mean (None).

    수식
    ----
    (RUN3-4)  score = Σ_row weight_row · loss_row / Σ_row weight_row,
             weight_row = 1/n_u, u = match_row,
             ci = [Pct₂.₅(replicates), Pct₉₇.₅(replicates)].
             행이 없으면 score = None; 복제가 비거나 NaN이 하나라도 있으면 ci = [None, None].

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      row, u, n_u | 평가 행·그 경기·경기 안 평가 행 수 | —; —; 행 | 0 기반 행; 저장 경기 행; ≥ 1 | per_row(), match
      loss_row, weight_row, score | 행 NLL·평가 가중치·평균 | nat; —; nat | nll 모듈의 유한 점수 계약 | values, scores.mean()
      replicates, ci | 경기 단위 재표본 점수와 백분위 구간 | nat | B개; 2개 또는 None | scores.bootstrap(B, seed), _percentile_ci
      B, seed, Pct | 복제 수·시드·numpy 백분위 연산 | 회; —; — | 기본 2000; 기본 20261021 | B, seed, PERCENTILES

    해석
    ----
    경기별 평가 행의 수를 보정한 NLL과 경기 전체를 재표집한 구간을 함께 기록한다(S3 P7, P9; NLL-4·5).
    구간은 복제의 백분위이며 신뢰수준이 실제로 달성된다는 보장은 아니다.
    관측할 행이 없는 머리는 평균과 구간을 비워 두며, 구간을 요약하는 단계는 NaN만 검사하고 무한값까지 별도로 거부하지는 않는다.

    근거
    ----
    S3 계획 P7·P9; (NLL-4)·(NLL-5); _percentile_ci(빈 복제).
    """
    values, match = scores.per_row()
    out = {'n_rows': int(len(values)), 'n_design_rows': int(len(scores.nll)), 'n_unscored': int(scores.n_unscored), 'nll': None,
           'nll_ci': [None, None]}
    if len(values):
        out['nll'] = scores.mean()
        out['nll_ci'] = _percentile_ci(scores.bootstrap(B, seed))
    return out


def _dev_predictions(ctx: Context, out: Path, log: _ProgressLog, labels, models: dict) -> dict:
    """The DEV scores of every head of B0 and B1 (masked and open), the M2 expected counts of both on the DEV frame rows and the probability
    that B2 gives: ``DEV_NLL.json`` and ``DEV_PRED.npz``."""
    from c_ept import compose
    s1, s2, guard = ctx.s1, ctx.s2, ctx.guard
    nll_doc = {}
    with ctx.sampler.scope() as scope:
        for label in ('b0', 'b1'):
            nll_doc[label] = {}
            for kind in ('masked', 'open'):
                guard()
                scores = models[label].nll(s1, s2, labels, DEV_ROLE, kind, check=guard)
                nll_doc[label][kind] = {name: score_summary(sc) for name, sc in scores.items()}
        guard()
        b2_rows = models['b2'].score_role(s1, s2, DEV_ROLE, check=guard)
        brier_values = b2_rows.brier()
        match, row = np.asarray(b2_rows.match), np.asarray(b2_rows.row)
        weights = features.match_weights(match)
        nll_doc['b2'] = {'nll': score_summary(b2_rows.scores()),
                         'brier': nll.weighted_mean(brier_values, weights),
                         'brier_ci': _percentile_ci(nll.match_bootstrap(brier_values, match, weights, BOOT_B, SEED)),
                         'n_rows': int(len(row))}
        guard()
        rows = compose.row_table(s1, s2, (DEV_ROLE,))
        if len(rows) == 0:
            raise StageFailed('there is no DEV frame row: the gate has nothing to judge')
        if not np.array_equal(np.asarray(rows.frame_row), row):
            raise StageFailed('the DEV frame rows of B2 and of the row table differ')
        seconds = np.asarray(rows.exposure_ms, dtype=np.float64) / 1000.0
        pred = {'frame_row': np.asarray(rows.frame_row, dtype=np.int64), 'match': np.asarray(rows.match, dtype=np.int64),
                'b2_p': np.asarray(b2_rows.p, dtype=np.float64)}
        for label in ('b0', 'b1'):
            chain = models[label].composer_heads('masked')
            chain.bind(s2, rows.views)
            rate = np.asarray(chain.rate(np.arange(len(rows))), dtype=np.float64)                 # (n, 18) first kills per second
            mass = rate * seconds                                                                 # the expected first kills of each bin
            pred[f'mu_{label}'] = np.cumsum(mass, axis=1)[:, [h // C.BIN_MS - 1 for h in C.EVAL_H_MS]]       # up to 60, 120 and 180 s
            del chain
    ctx.write_json(out / 'DEV_NLL.json', nll_doc)
    ctx.write_npz(out / 'DEV_PRED.npz', pred)
    log.add('dev', wall_s=round(scope.wall, 3), **_peaks(scope))
    ctx.say(f'  DEV scores: {round(scope.wall, 1)} s')
    return {'wall_s': round(scope.wall, 3), **_peaks(scope), 'n_rows': int(len(rows)), 'nll': nll_doc}


def stage_fit(ctx: Context) -> dict:
    """The tower tier law, B0, B1 and B2 on the FIT matches (B1 and B2 stop early on STOP), and the DEV predictions of all of them. Each
    model is saved when it is fitted, with its time and peak memory in ``fit\\PROGRESS.jsonl``; a resumed call keeps the models that are
    saved.

    머리별 저장을 이용해 기준 모형 적합과 DEV 예측을 연결한다(S3 Task12; G-S3b B3).
    """
    out = ctx.run / 'fit'
    out.mkdir(parents=True, exist_ok=True)
    labels = open_labels(ctx)
    log = _ProgressLog(out / 'PROGRESS.jsonl')
    inputs = _fit_inputs(ctx, out)
    dropped, means = tuple(inputs['dropped']), np.asarray(inputs['b0_fit_means'], dtype=np.float64)
    ctx.say(f'  FIT views {inputs["n_fit_views"]}, {len(dropped)} columns dropped: {", ".join(dropped) if dropped else "none"}')
    models, facts = {}, {}
    models['tier'], facts['tier'] = _fit_tier(ctx, out, log)
    models['b0'], facts['b0'] = _fit_b0(ctx, out, log, labels, means)
    models['b1'], facts['b1'] = _fit_b1(ctx, out, log, labels, dropped)
    models['b2'], facts['b2'] = _fit_b2(ctx, out, log, dropped)
    dev = _dev_predictions(ctx, out, log, labels, models)
    result = {'n_fit_views': inputs['n_fit_views'], 'n_fit_matches': inputs['n_fit_matches'], 'dropped': inputs['dropped'],
              'b0_fit_means': inputs['b0_fit_means'], 'b0_fill_counts': inputs['b0_fill_counts'], **facts,
              'dev': {k: v for k, v in dev.items() if k != 'nll'}}
    ctx.write_json(out / 'FIT_FACTS.json', result)
    return result


# ---- stage: compose ----------------------------------------------------------------------------------------------------------------
def subset_rows(rows, keep: np.ndarray):
    """The ``RowTable`` of the rows of ``rows`` where the bool mask ``keep`` is true, in the same order (``compose.RowTable`` has no
    selection of its own).

    선택된 행의 순서를 유지하여 합성용 표를 만든다(S3 Task12).
    """
    from c_ept import compose
    return compose.RowTable(frame_row=rows.frame_row[keep], match=rows.match[keep], t0_ms=rows.t0_ms[keep],
                            exposure_ms=rows.exposure_ms[keep], T_ms=rows.T_ms[keep], views=rows.views[keep],
                            destroyed=rows.destroyed[keep], roles=rows.roles)


def b1_sample(rows, smoke: bool, seed: int = SEED, size: int | None = None) -> np.ndarray:
    """The match numbers (sorted) of the DEV matches that B1 is composed on: a seeded sample of ``size`` (default: ``B1_DEV_MATCHES``, read
    when called), or all of them when there are no more than ``size`` and in a smoke run (where the DEV matches are the smoke selection).

    수식
    ----
    (RUN3-5)  chosen_matches = unique(match) (smoke 또는 #unique(match) ≤ size),
             그 밖에는 sort(unique(match)[perm[:size]]),
             perm = RNG([seed, DEV_code, size]).permutation(#unique(match)).

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      match, chosen_matches | 합성 행의 경기 행과 선택된 경기 행 | — | 0 기반 int64 저장 행 | rows.match, 반환값
      size, smoke | 요청한 경기 수와 소규모 점검 실행 여부 | 경기; — | 기본 500; bool | size, B1_DEV_MATCHES, smoke
      perm, seed, DEV_code | 위치 순열·시드·DEV 역할 코드 | — | 0 기반; 난수 생성기 계약 | order, seed, C.ROLES.index(DEV_ROLE)

    해석
    ----
    B1의 합성 대상은 행을 따로 뽑지 않고 경기별로 고른다(S3 Task12).
    소규모 점검 실행 또는 요청 이하의 경기 수이면 난수를 만들기 전에 전부 반환한다.
    다른 경로의 요청 경기 수 유효성은 이 함수가 직접 검증하지 않으므로 호출 계약이며, 순열 시드에 요청 경기 수도 들어가 표본의 포함 관계를 보장하지 않는다.

    근거
    ----
    S3 계획 Task12(B1 합성 표본); b1_sample(호출 계약).
    """
    size = B1_DEV_MATCHES if size is None else size
    matches = np.unique(np.asarray(rows.match, dtype=np.int64))
    if smoke or len(matches) <= size:
        return matches
    order = np.random.default_rng([int(seed), C.ROLES.index(DEV_ROLE), int(size)]).permutation(len(matches))
    return np.sort(matches[order[:size]])


def _timed_call(ctx: Context, calls: list, head_set: str, comp, method: str, idx: np.ndarray, block: int, workers: int):
    """One compose call (``expected_onsets``, ``expect`` or ``simulate``) with the block size and the workers of this stage and the E6
    guard as ``check``; its time, its blocks and the checks it made go into ``calls`` (COMPOSE.json). The counters of the checks are
    those of this call alone and carry the prefix ``call_`` (``call_check_calls``, ``call_check_s_total``, ``call_check_s_mean``,
    ``call_check_s_longest`` and ``call_check_budget_s``: workers x the mean check); the counters of the stage have no prefix.

    식 (RUN3-11)의 해당 항을 계산한다.
    """
    t0 = time.perf_counter()
    with ctx.guard.window() as checks:
        result = getattr(comp, method)(idx, block=block, workers=workers, check=ctx.guard)
    wall = time.perf_counter() - t0
    # 식 (RUN3-11)의 동시 작업 수 보정은 블록 계산시간의 추정 척도이며 CPU 실측이 아니다.
    n_blocks = max(1, -(-len(idx) // block))
    counters = checks.counters('call_')
    record = {'head_set': head_set, 'method': method, 'rows': int(len(idx)), 'block': int(block), 'workers': int(workers),
              'n_blocks': int(n_blocks), 'wall_s': round(wall, 3), 'seconds_per_block_wall': round(wall / n_blocks, 4),
              'block_compute_s_estimate': round(wall * min(workers, n_blocks) / n_blocks, 3), **counters}
    mean = counters['call_check_s_mean']
    record['call_check_budget_s'] = None if mean is None else round(workers * mean, 3)
    calls.append(record)
    return result


def _screen(ctx: Context, calls: list, comp, head_set: str, workers: int) -> np.ndarray:
    """The expected onsets of every row (60, 120, 180 s) through the composer, with workers: ruling R31 stops the run when a row has more
    than ``R31_LIMIT`` in 180 s, naming the head set and the number of rows.

    식 (RUN3-11)의 해당 항을 계산한다.
    """
    n = len(comp.rows)
    onsets = np.asarray(_timed_call(ctx, calls, head_set, comp, 'expected_onsets', np.arange(n), BLOCK_EXPECTED_ONSETS, workers))
    over = int((onsets[:, 2] > R31_LIMIT).sum()) if n else 0
    if over:
        raise StageFailed(f'R31: the head set {head_set} (its M2 rate head) gives more than {R31_LIMIT:g} expected onsets in 180 s on '
                          f'{over} of {n} rows (the largest {float(onsets[:, 2].max()):.1f}; real rates are about 3.6): the head is '
                          'broken, nothing is composed')
    return onsets


def _chunk_ranges(n: int, size: int | None = None) -> list:
    """The consecutive row ranges (lo, hi) of at most ``size`` rows (default: ``CHUNK_ROWS``, read when called) that cover ``n`` rows.

    식 (RUN3-11)의 해당 항을 계산한다.
    """
    size = CHUNK_ROWS if size is None else size
    return [(lo, min(lo + size, n)) for lo in range(0, n, size)]


def _scalar(saved) -> int | None:
    """The whole number in a saved array of one element, or None for anything else (nothing, a longer array, a non-integer). The npz
    writer (``np.ascontiguousarray``) saves a 0-d array with shape (1,), and ``int()`` of a 1-d array is deprecated in NumPy."""
    if saved is None:
        return None
    saved = np.asarray(saved)
    if saved.size != 1 or saved.dtype.kind not in 'iu':
        return None
    return int(saved.reshape(-1)[0])


def _load_chunk(path: Path, frame_row: np.ndarray, S: int, seed: int):
    """The arrays of a saved chunk, or None when there is none, it cannot be read, it holds other rows than ``frame_row`` or it was made
    with another number of simulations or another seed than ``S`` and ``seed``."""
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            arrays = {name: z[name] for name in z.files}
    except (OSError, ValueError):
        return None
    same = (np.array_equal(arrays.get('frame_row'), frame_row) and _scalar(arrays.get('S')) == S
            and _scalar(arrays.get('seed')) == seed)
    return arrays if same else None


def _compose_head_set(ctx: Context, calls: list, comp, head_set: str, onsets: np.ndarray, closed_form: bool, parts: Path,
                      workers: int, stats: dict | None = None) -> dict:
    """The closed form (when the heads have one) and the simulation of every row of ``comp.rows``, one chunk of ``CHUNK_ROWS`` rows at a
    time; a chunk is saved as soon as it is computed and a resumed call keeps the chunks that are there. Returns the arrays of the whole
    table. ``stats`` (a dict, filled in place) gets the work of this call: ``chunks_total``, ``chunks_loaded`` (found saved by an earlier
    call and used), ``chunks_computed`` and ``rows_computed`` (the rows of the chunks that this call computed).

    식 (RUN3-11)의 해당 항을 계산한다.
    """
    rows, n = comp.rows, len(comp.rows)
    sim_block = BLOCK_B0_SIMULATE if closed_form else BLOCK_B1_SIMULATE
    ranges = _chunk_ranges(n)
    stats = {} if stats is None else stats
    stats.update(chunks_total=len(ranges), chunks_loaded=0, chunks_computed=0, rows_computed=0)
    pieces = []
    for i, (lo, hi) in enumerate(ranges):
        path = parts / f'{head_set}_{i:05d}.npz'
        arrays = _load_chunk(path, rows.frame_row[lo:hi], comp.S, comp.seed)
        if arrays is None:
            idx = np.arange(lo, hi)
            arrays = {'frame_row': np.asarray(rows.frame_row[lo:hi], dtype=np.int64), 'S': np.array(comp.S), 'seed': np.array(comp.seed)}
            if closed_form:
                arrays['expected'] = np.asarray(_timed_call(ctx, calls, head_set, comp, 'expect', idx, BLOCK_B0_EXPECT, workers))
            sim = _timed_call(ctx, calls, head_set, comp, 'simulate', idx, sim_block, workers)
            arrays.update(mc_mean=np.asarray(sim.mean), mc_sd=np.asarray(sim.sd), q19=np.asarray(sim.q19))
            ctx.write_npz(path, arrays)
            stats['chunks_computed'] += 1
            stats['rows_computed'] += hi - lo
            ctx.say(f'  {head_set}: rows {lo}-{hi} of {n} composed')
        else:
            stats['chunks_loaded'] += 1
            ctx.say(f'  {head_set}: rows {lo}-{hi} of {n} were saved by an earlier call')
        pieces.append(arrays)
    whole = {name: np.concatenate([p[name] for p in pieces]) for name in pieces[0] if name not in ('S', 'seed')}
    whole.update(match=np.asarray(rows.match, dtype=np.int64), expected_onsets=onsets, S=np.array(comp.S), seed=np.array(comp.seed))
    return whole


def stage_compose(ctx: Context) -> dict:
    """The composition of V_start on the DEV rows (decision P6): B0 on every DEV row (the expected onsets screened first, R31; the closed
    form and S = 1,000 simulations, every value computed in workers, R30) and B1 on a seeded sample of 500 DEV matches (S = 200, simulation
    only). Per row: the frame row, the match, the closed form (B0), the mean, the standard deviation and the 19 quantiles of the simulations
    and the expected onsets. COMPOSE.json holds the counters of the E6 checks of THIS stage (from its start: ``check_calls``,
    ``check_s_total``, ``check_s_mean``, ``check_s_longest``), those of every compose call under ``call_`` names in ``calls``, and the
    work of this call: ``chunks_total``, ``chunks_loaded`` (saved by an earlier call and used), ``chunks_computed`` and ``rows_computed``,
    also for each head set (``chunks_by_head_set``).

    수식
    ----
    (RUN3-11)  block_count = max(1, ⌈row_count / block_size⌉),
             seconds_per_block_wall = round(call_wall / block_count, 4),
             block_compute_s_estimate = round(call_wall · min(worker_count, block_count) / block_count, 3),
             call_check_budget_s = round(worker_count · rounded_check_mean, 3) (평균이 None이면 None).
             chunk_total = chunk_loaded + chunk_computed,
             rows_computed = Σ_{이번 호출에서 계산한 chunk} (chunk_hi − chunk_lo).
             단계 거부 ⇔ #{row : expected_onsets[row, 2] > 50} > 0.

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      row_count, block_size, block_count | 호출 행 수·블록 상한·기록 블록 수 | 행; 행; 개 | ≥ 0; 양수 계약; ≥ 1 | len(idx), block, n_blocks
      call_wall, seconds_per_block_wall, block_compute_s_estimate | 경과 시간·블록당 벽시계·계산시간 추정 | s | 비음수, 실측 CPU 시간 아님 | wall, record
      worker_count, rounded_check_mean, call_check_budget_s | 작업자 수·반올림된 검사 평균·비용 기준 | 개; s; s | 합성은 ≥ 2; None 가능 | workers, counters, mean
      chunk_total, chunk_loaded, chunk_computed, rows_computed | 묶음 수와 새 계산 행 수 | 개; 개; 개; 행 | 비음수 | stats, CHUNK_COUNTERS
      chunk_lo, chunk_hi | 묶음의 시작·배타적 끝 행 | — | 0 기반, 길이 ≤ CHUNK_ROWS | _chunk_ranges, lo, hi
      expected_onsets | 지평별 발생 기대 개수 | 개 | 열 2는 180 s, 유한·비음수는 상위 계약 | _screen의 onsets

    해석
    ----
    블록 경과 시간과 검사 비용을 함께 남겨 부모 프로세스의 검사 비용이 계산을 지배하는지 검토한다(S3 R29~R31).
    블록당 계산시간은 동시 작업 수를 반영한 추정값이며 직접 측정한 CPU 시간이 아니다.
    이어하기에서 읽은 묶음은 새 계산 행 수에 넣지 않고, 묶음 계수는 이번 호출에 대해서만 합산한다(S3 G-S3b B3).
    검사 평균은 이미 소수 네 자리로 반올림된 값을 사용하며, 검사가 없으면 검사 비용 기준도 None이다.
    발생 기대값이 유한·비음수라는 것은 상위 계산의 계약으로 이 상한 검사가 검증하지 않으며, 초과 행이 있으면 행을 버리지 않고 실행을 멈춘다(S3 R31).

    근거
    ----
    S3 계획 Task12; S3 판정 R29~R32; S3 G-S3b B3 계수 항.
    """
    with ctx.guard.window() as checks:
        return _stage_compose(ctx, checks)


def _stage_compose(ctx: Context, checks: _GuardWindow) -> dict:
    """식 (RUN3-11)의 해당 항을 계산한다."""
    from c_ept import baselines_b0, baselines_b1, compose, towertier
    out = ctx.run / 'compose'
    parts = out / 'parts'
    out.mkdir(parents=True, exist_ok=True)
    parts.mkdir(parents=True, exist_ok=True)
    labels = open_labels(ctx)
    gold_reps = np.array(labels.array('labels/gold_reps'), dtype=np.float64)
    folder = ctx.run / 'fit'
    tier = towertier.TierModel.load(folder / 'tier')
    b0 = baselines_b0.B0Heads.load(folder / 'b0')
    b1 = baselines_b1.B1Heads.load(folder / 'b1')
    rows = compose.row_table(ctx.s1, ctx.s2, (DEV_ROLE,))
    if len(rows) == 0:
        raise StageFailed('there is no DEV frame row to compose')
    workers = ctx.compose_workers
    calls = []
    smoke = ctx.smoke is not None
    sample = b1_sample(rows, smoke)
    rows_b1 = subset_rows(rows, np.isin(rows.match, sample))
    with ctx.sampler.scope() as scope:
        comp0 = compose.Composer(b0.composer_heads('masked'), ctx.prices, tier, gold_reps, rows, ctx.s2, S=S_B0, seed=SEED)
        onsets0 = _screen(ctx, calls, comp0, 'B0', workers)
        comp1 = compose.Composer(b1.composer_heads('masked'), ctx.prices, tier, gold_reps, rows_b1, ctx.s2, S=S_B1, seed=SEED)
        onsets1 = _screen(ctx, calls, comp1, 'B1', workers)
        ctx.say(f'  R31 screen: the largest expected onsets in 180 s are {float(onsets0[:, 2].max()):.2f} (B0) and '
                f'{float(onsets1[:, 2].max()):.2f} (B1)')
        by_set = {'B0': {}, 'B1': {}}                             # the chunks of each head set: the work of this call
        whole0 = _compose_head_set(ctx, calls, comp0, 'B0', onsets0, True, parts, workers, by_set['B0'])
        whole1 = _compose_head_set(ctx, calls, comp1, 'B1', onsets1, False, parts, workers, by_set['B1'])
    ctx.write_npz(out / 'b0_dev.npz', whole0)
    ctx.write_npz(out / 'b1_dev.npz', whole1)
    _remove_tree(parts, out)                                  # the chunks are in the two files now
    n_dev = len(np.unique(rows.match))
    how = ('all the smoke DEV matches' if smoke else 'all the DEV matches' if len(sample) == n_dev
           else f'a seeded sample of {B1_DEV_MATCHES} DEV matches')
    facts = {'rows': int(len(rows)), 'matches': int(n_dev), 'workers': workers, 'S_b0': S_B0, 'S_b1': S_B1, 'seed': SEED,
             'b1_rows': int(len(rows_b1)), 'b1_matches': sample, 'b1_sample': how,
             'r31': {'limit': R31_LIMIT, 'max_expected_onsets_180_b0': float(onsets0[:, 2].max()),
                     'max_expected_onsets_180_b1': float(onsets1[:, 2].max())},
             'blocks': compose_constants()['blocks'], 'chunk_rows': CHUNK_ROWS, 'calls': calls, 'wall_s': round(scope.wall, 3),
             **_peaks(scope), **checks.counters(),
             **{key: sum(stats[key] for stats in by_set.values()) for key in CHUNK_COUNTERS}, 'chunks_by_head_set': by_set}
    ctx.write_json(out / 'COMPOSE.json', facts)
    return facts


def compose_constants() -> dict:
    """The sizes of the compose units of work, as RUN.json records them.

    블록·묶음·표본 상수를 실행 기록에 제공한다(S3 Task12).
    """
    return {'blocks': {'expected_onsets': BLOCK_EXPECTED_ONSETS, 'b0_simulate': BLOCK_B0_SIMULATE, 'b0_expect': BLOCK_B0_EXPECT,
                       'b1_simulate': BLOCK_B1_SIMULATE}, 'chunk_rows': CHUNK_ROWS, 'S_b0': S_B0, 'S_b1': S_B1, 'seed': SEED,
            'b1_dev_matches': B1_DEV_MATCHES, 'r31_limit': R31_LIMIT}


# ---- stage: gate -------------------------------------------------------------------------------------------------------------------
def observed_onsets(s1, frame_row: np.ndarray, n_bins: int = B2_BINS) -> np.ndarray:
    """The number of onsets in the bins 0 .. ``n_bins`` - 1 of each of the frame rows ``frame_row`` (the first kills of engagements within
    the first 60 s for 6 bins), from the S1 ``onsets`` table. The runner reads it, and not ``compose``, which never reads an array of the
    engagements.

    수식
    ----
    (RUN3-6)  count_row = Σ_ev 1[onset_row_ev = row ∧ onset_bin_ev < n_bins],
             반환 = count[frame_row].

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      row, ev, frame_row | 전체 프레임 행·발생 표 행·요청 행 | — | 모두 0 기반 유효 저장 행 | onsets_row, counts, frame_row
      onset_bin_ev, n_bins | 발생의 구간과 포함할 구간 수 | — | 계약상 구간 ≥ 0; 기본 6 | onsets_bin, n_bins
      count_row | 선택한 앞 구간의 첫 킬 수 | 개 | 비음수 정수 | np.bincount 결과

    해석
    ----
    합성 모듈에 실제 교전 표를 넘기지 않고 실행기가 별도로 평가용 발생 수를 읽는다(S3 Task12).
    구간 번호가 비음수라는 것은 S1 표의 계약이며, 여기의 선택식 자체는 상한만 검사한다.
    저장 행과 구간은 int64로 바꾸며, 요청 프레임 행이 유효한지도 호출자가 지킨다(S1–S2 Task5 발생 표).

    근거
    ----
    S3 계획 Task12(관문 관측값); S1–S2 계획 Task5(발생 표).
    """
    onsets_row = np.asarray(s1.array('onsets/row')).astype(np.int64)
    onsets_bin = np.asarray(s1.array('onsets/bin')).astype(np.int64)
    n_rows = len(np.asarray(s1.array('frame_rows/t0')))
    counts = np.bincount(onsets_row[onsets_bin < n_bins], minlength=n_rows)
    return counts[np.asarray(frame_row, dtype=np.int64)]


def gate_pass(verdict: dict) -> bool:
    """Ruling R27: the gate passes when every criterion that was judged passes (``all_pass``) and none of the four is missing.

    판정된 기준의 전체 통과와 빠진 기준의 부재를 함께 확인한다(S3 R27).
    """
    return bool(verdict['all_pass']) and not verdict['missing']


def brier_comparison(mu_b0, mu_b1, p_b2, observed, match, B: int = BOOT_B, seed: int = SEED) -> dict:
    """The Brier scores of the forecasts of at least one onset in the first 60 s: M2 of B0 and of B1, 1 - exp(-sum_{q<6} rate_q
    exposure_q), and B2. Each is the match-weighted mean of (p - y)^2 (weights 1/n_u) with its percentile interval; the differences are
    intervals of the differences of the paired bootstrap replicates (the same matches are drawn for every score). Reported, not judged.

    수식
    ----
    (RUN3-7)  truth_row = 1[observed_row ≥ 1], p_{row,head} = −expm1(−μ_{row,head}) (M2),
             p_{row,B2} = p_b2_row, square_{row,head} = (p_{row,head} − truth_row)^2,
             score_head = Σ_row weight_row · square_{row,head} / Σ_row weight_row,
             weight_row = 1/n_u, positive_share = Σ_row weight_row · truth_row / Σ_row weight_row.
             difference_{head,other} = score_head − score_other,
             difference_ci = [Pct₂.₅, Pct₉₇.₅](replicates_head − replicates_other),
             score_ci = [Pct₂.₅, Pct₉₇.₅](replicates_head), 같은 B, seed의 (NLL-5).

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      row, head, other | 평가 행과 비교할 예측 이름 | — | 0 기반; m2_b0, m2_b1, b2 | values의 행과 키
      observed_row, truth_row | 발생 수와 한 번 이상 표시 | 개; — | 계약상 ≥ 0; 0 또는 1 | observed, y
      μ_{row,head}, p_{row,head} | 기대 개수와 한 번 이상 확률 | 개; — | 계약상 유한 ≥ 0; [0,1] | mu_b0, mu_b1, forecasts
      square_{row,head}, score_head | 행 제곱오차와 경기 가중 평균 | — | 유효 확률이면 [0,1] | values, out[*].brier
      weight_row, n_u, u | 행 가중치·경기별 행 수·경기 | —; 행; — | 1/n_u; ≥ 1; 저장 경기 행 | w, match
      replicates_head, B, seed, Pct | 복제·복제 수·시드·백분위 | — | (NLL-5), (RUN3-4) | reps, B, seed, _percentile_ci
      difference_{head,other}, difference_ci, score_ci, positive_share | 점수 차·차이 구간·점수 구간·양성 비율 | — | 유한 입력 계약, 구간은 None 가능 | out

    해석
    ----
    포아송 기대 개수를 한 번 이상 발생할 확률로 바꾸어 직접 확률을 내는 이항 기준 모형과 같은 목표를 채점한다(S3 Task12).
    같은 경기 배열·시드·복제 수를 사용하므로 모형 간 차이의 구간은 같은 경기 재표본끼리 뺀 값으로 계산한다(S3 R26; NLL-5).
    입력의 기대 개수와 확률 범위는 이 함수에서 별도로 검사하지 않으며, 빈 입력이나 비유한 점수는 공통 점수 함수의 유효성 검사를 따른다.
    점수와 차이는 보고용이며 관문 합격 조건에 더하지 않는다.

    근거
    ----
    S3 계획 Task12(Brier 비교); S3 판정 R26(공유 재표집); (NLL-4)·(NLL-5).
    """
    # 식 (RUN3-7)의 이항 목표와 포아송 발생 확률을 같은 행에서 비교한다(S3 Task12).
    y = (np.asarray(observed) >= 1).astype(np.float64)
    forecasts = {'m2_b0': -np.expm1(-np.asarray(mu_b0, dtype=np.float64)), 'm2_b1': -np.expm1(-np.asarray(mu_b1, dtype=np.float64)),
                 'b2': np.asarray(p_b2, dtype=np.float64)}
    w = features.match_weights(match)
    values = {k: (p - y) ** 2 for k, p in forecasts.items()}
    # 식 (RUN3-7)의 차이 구간은 같은 시드의 경기 재표본끼리 짝지어 계산한다(S3 R26).
    reps = {k: nll.match_bootstrap(v, match, w, B, seed) for k, v in values.items()}
    out = {k: {'brier': nll.weighted_mean(v, w), 'ci': _percentile_ci(reps[k])} for k, v in values.items()}
    out['difference'] = {f'{a}_minus_{b}': {'value': out[a]['brier'] - out[b]['brier'], 'ci': _percentile_ci(reps[a] - reps[b])}
                         for a, b in (('m2_b0', 'b2'), ('m2_b1', 'b2'), ('m2_b0', 'm2_b1'))}
    out.update(n_rows=int(len(y)), n_matches=int(len(np.unique(match))), positive_share=float(w @ y / w.sum()), B=int(B), seed=int(seed))
    return out


def tower_pricing_diagnostic(s1, frame_row, match, t0_ms, tier_model, prices, B: int = BOOT_B, seed: int = SEED,
                             window_ms: int = 60_000) -> dict:
    """Report only (Task 8): the towers that were taken in (t0, t0 + 60 s] of the DEV frame rows, priced by their real tier, against the
    expected price that the tier law gives from the tower state of t0 (the state the composer knows), with its time argument at the tower
    event time. For each such tower: the price of its tier (``realised``), the expected price from that state and event time (``expected``)
    and their difference; the means, the interval of the mean difference over
    ``B`` match-bootstrap replicates and the counts by tier. ``frame_row``, ``match``, ``t0_ms``: the DEV rows in S1 order.

    수식
    ----
    (RUN3-8)  key_row = match_row · 2^40 + t0_ms_row,
             pick_ev = searchsorted(key_row, match_ev · 2^40 + time_ms_ev, left) − 1,
             유지 ⇔ pick_ev ≥ 0 ∧ match_pick = match_ev ∧ time_ms_ev ≤ t0_ms_pick + window_ms.
             prob_ev = tier_model(victim_counts(state_at_t0, taker_ev), taker_ev, time_ms_ev/1000),
             expected_ev = Σ_tier prob_{ev,tier} · price_tier, realised_ev = price_{tier_ev},
             gap_ev = realised_ev − expected_ev, mean_gap = Σ_ev gap_ev / event_count,
             mean_realised = mean(realised), mean_expected = mean(expected),
             sum_realised = Σ_ev realised_ev, sum_expected = Σ_ev expected_ev.
             gap_ci = [Pct₂.₅, Pct₉₇.₅]((NLL-5)(gap, ev_match, 모두 1인 가중치)).

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      row, ev, tier | DEV 행·선택한 포탑 사건·포탑 단계 | — | 0 기반; 단계 0..3 | frame_row, events, tiers
      match_row, match_ev, ev_match | 행과 포탑의 저장 경기 행 | — | 0 기반 | match, obj_match
      key_row, pick_ev | 정렬용 복합 키와 직전 행 위치 | — | int64 범위·시간 비충돌 가정; −1은 선행 행 없음 | row_key, pos
      t0_ms_row, time_ms_ev, window_ms | 기준·사건 시각과 창 길이 | ms | 호출 계약; 기본 60000 | t0_ms, t[events], window_ms
      state_at_t0, taker_ev, prob_ev | 기준 시각 파괴 상태·가져간 팀·단계 확률 | 개; —; — | 상태 계약; 0/1; 확률 4열 | state, who, probability
      price_tier, realised_ev, expected_ev, gap_ev | 단계 가격·실현 가격·기대 가격·차 | 골드 | 유한 입력 가정, 부호 가능 | prices, realised, expected, gap
      event_count, mean_gap, mean_realised, mean_expected, sum_realised, sum_expected, gap_ci | 사건 수·평균·합·차이 구간 | 개; 골드 | 사건 수 > 0일 때 | out
      B, seed, Pct | 경기 복제 수·시드·백분위 | 회; —; — | (RUN3-4)·(NLL-5) | B, seed, _percentile_ci

    해석
    ----
    선택된 각 포탑은 같은 경기에서 사건보다 엄격히 앞선 마지막 DEV 행 하나에만 대응시킨다(S3 Task8 진단).
    상태는 기준 시각에서 읽지만 단계 분포의 시간 인자는 실제 사건 시각의 초 단위 값이다(S3 as-built의 열린 항목).
    평균은 사건을 합쳐 낸 평균이며 구간도 사건 가중치를 모두 1로 둔 경기 군집 부트스트랩이므로 경기별 평균의 단순 평균이 아니다.
    보고한 경기 수는 입력 행의 경기 수라서 실제 사건 재표본의 경기 수보다 클 수 있고, 선택 사건이 없으면 평균·구간을 비우고 단계 표는 빈 사전이다.
    복합 키의 비충돌·유효 범위와 인덱싱할 비어 있지 않은 행 입력은 가정이며, 이 함수는 키의 비감소 순서만 명시적으로 검사한다.

    근거
    ----
    S3 계획 Task8·Task12(가격 진단); S3 as-built 열린 항목; (NLL-5).
    """
    from c_ept import flowprice, towertier
    frame_row, match, t0_ms = (np.asarray(a, dtype=np.int64) for a in (frame_row, match, t0_ms))
    off = np.asarray(s1.array('matches/obj_off')).astype(np.int64)
    t, kind, sub, taker = (np.asarray(s1.array(f'objectives/{n}')) for n in ('t', 'kind', 'sub', 'taker'))
    first = flowprice.GROUPS.index('tower_outer')
    tier = flowprice.group_of(kind, sub).astype(np.int64) - first
    obj_match = np.repeat(np.arange(len(off) - 1, dtype=np.int64), np.diff(off))
    counted = (tier >= 0) & (tier < towertier.N_TIERS) & ((taker == 0) | (taker == 1)) & np.isin(obj_match, np.unique(match))
    events = np.flatnonzero(counted)
    span = 1 << 40
    row_key = match * span + t0_ms
    if len(row_key) > 1 and (np.diff(row_key) < 0).any():
        raise ValueError('the DEV rows are not ordered by match and t0')
    event_key = obj_match[events] * span + t[events].astype(np.int64)
    pos = np.searchsorted(row_key, event_key, side='left') - 1                  # the last row with t0 < the time of the tower
    ok = pos >= 0
    pos = np.where(ok, pos, 0)
    inside = ok & (match[pos] == obj_match[events]) & (t[events].astype(np.int64) <= t0_ms[pos] + window_ms)
    events, pos = events[inside], pos[inside]
    group_price = np.array([prices[g] for g in towertier.TIER_GROUPS], dtype=np.float64)
    out = {'window_s': window_ms / 1000.0, 'n_events': int(len(events)), 'n_rows': int(len(frame_row)),
           'n_matches': int(len(np.unique(match))), 'B': int(B), 'seed': int(seed)}
    if not len(events):
        return {**out, 'mean_realised': None, 'mean_expected': None, 'mean_difference': None, 'mean_difference_ci': [None, None],
                'by_tier': {}}
    who, tiers = taker[events].astype(np.int64), tier[events]
    # 식 (RUN3-8): 상태는 질문 시각, 분포의 시간 인자는 사건 시각이다(S3 as-built).
    state = towertier.destroyed_before(s1, match[pos], t0_ms[pos])                     # the state at t0 of the row of each tower
    probability = tier_model.prob(towertier.victim_counts(state, who), who, t[events].astype(np.float64) / 1000.0)
    expected = towertier.expected_tower_price(probability, prices)
    realised = group_price[tiers]
    gap = realised - expected
    ev_match = obj_match[events]
    out.update(mean_realised=float(realised.mean()), mean_expected=float(expected.mean()), mean_difference=float(gap.mean()),
               mean_difference_ci=_percentile_ci(nll.match_bootstrap(gap, ev_match, np.ones(len(gap)), B, seed)),
               sum_realised=float(realised.sum()), sum_expected=float(expected.sum()),
               by_tier={towertier.TIER_GROUPS[j]: {'n': int((tiers == j).sum()), 'mean_expected_at_t0': float(expected[tiers == j].mean())
                                                   if (tiers == j).any() else None} for j in range(towertier.N_TIERS)})
    return out


def gate_results(*, match, y60, expected_onsets60, expected_value60, realised_value60, mc_mean60, mc_sd60, mu_b1_60, mu_b0_60, b2_p,
                 accounting_error: float, n_matches: int, S: int = S_B0, B: int = BOOT_B, seed: int = SEED) -> dict:
    """The gate of D6 on the DEV rows (decisions P7 and P8), from arrays alone. ``match`` the match of each row, ``y60`` the onsets in the
    first 60 s, ``expected_onsets60`` and ``expected_value60`` the closed forms of B0 (H = 60 s), ``realised_value60``
    ``labels/v_start[:, 0]``, ``mc_mean60`` and ``mc_sd60`` the simulation of B0, ``mu_b1_60`` the M2 expected count of B1, ``mu_b0_60``
    that of B0 (for the Brier score), ``b2_p`` the probability of B2, ``accounting_error`` the labels' ``max_abs_error``. ``n_matches`` is
    the number of DEV matches that have a frame row (the sample the intervals must have been made from). The weights are 1/n_u.
    StageFailed when a pinned constant (B, the seed, the number of matches, S) is not what the plan fixes.

    Returns ``{'results', 'verdict', 'gate_pass', 'brier', 'constants'}``; ``gate_pass`` is ruling R27 (``gate_pass()``).

    수식
    ----
    (RUN3-9)  retained = {row : μ_{row,B1} > 0},
             retained_n_u = #{row∈retained : match_row = u},
             retained_weight_row = 1/retained_n_u,
             φ = (GATE-6)(y60[retained], μ_B1[retained], retained_weight).
             gate_pass = verdict.all_pass ∧ (verdict.missing가 비어 있음).

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      row, retained, u | DEV 행·양의 평균 행 집합·그 경기 | — | 0 기반 저장 행 | keep, match
      μ_{row,B1}, y60 | B1 기대 개수와 관측 발생 수 | 개 | μ > 0인 행만; 계약상 ≥ 0 | mu_b1, y60
      retained_n_u, retained_weight_row, φ | 남은 경기별 행 수·재계산 가중치·분산비 | 행; —; — | ≥ 1; > 0; (GATE-6) | features.match_weights(match[keep]), overdispersion
      gate_pass | 빠짐없이 통과한 관문 여부 | — | bool | gate_pass(verdict)
      B, seed, S, n_matches | 복제 수·시드·모의 수·DEV 유효 경기 수 | 회; —; 회; 경기 | 2000, 20261021, 1000, 호출값 | B, seed, S, n_matches

    해석
    ----
    발생·가치 보정과 모의 일치는 공통 관문 통계의 식으로 계산하고, 분산비만 양의 B1 평균을 갖는 행에서 다시 경기 가중치를 만든다(S3 P7·P8, R26).
    발생·가치 구간은 입력 DEV 경기 수와 정확히 같아야 하지만 분산비의 경기 수는 그 이하이면 허용한다(이 함수의 구간 검증).
    복제 수·시드·모의 수는 고정값과 대조하고, 구간 부적합이나 빈 표본의 처리는 각 관문 통계의 계약을 따른다.
    관문은 네 기준에 빠진 항목이 없는지도 확인하며, 과산포와 Brier는 별도 진단이다(S3 R27; GATE-7).

    근거
    ----
    S3 계획 P7·P8; S3 판정 R26·R27; (GATE-1)~(GATE-7).
    """
    from c_ept import gate_s3
    if (B, seed, S) != (nll.BOOT_B, nll.BOOT_SEED, S_B0) or (BOOT_B, SEED) != (nll.BOOT_B, nll.BOOT_SEED):
        raise StageFailed(f'the gate is pinned to B = {nll.BOOT_B}, the seed {nll.BOOT_SEED} and S = {S_B0}; got B = {B}, seed {seed}, '
                          f'S = {S}')
    match = np.asarray(match, dtype=np.int64)
    w = features.match_weights(match)
    occurrence = gate_s3.occurrence_calibration(expected_onsets60, y60, match, w, B=B, seed=seed)
    value = gate_s3.value_calibration(expected_value60, realised_value60, match, w, B=B, seed=seed)
    mc = gate_s3.mc_agreement(mc_mean60, mc_sd60, S, expected_value60)
    mu_b1 = np.asarray(mu_b1_60, dtype=np.float64)
    # 식 (RUN3-9)의 양의 평균 행에서 경기별 분모를 다시 만든다(S3 P8·R26).
    keep = mu_b1 > 0                                              # 1 / n_u over the rows that count, as HeadScores has it (Task 10)
    overdispersion = gate_s3.overdispersion(np.asarray(y60)[keep], mu_b1[keep], match[keep], features.match_weights(match[keep]),
                                            B=B, seed=seed)
    results = {'occurrence': occurrence, 'value': value, 'mc': mc, 'overdispersion': overdispersion,
               'accounting': {'max_abs_error': float(accounting_error)}}
    verdict = gate_s3.verdict(results)
    for name in ('occurrence', 'value'):                    # the intervals are judged as intervals only when they were made as planned
        r = results[name]
        if (r['B'], r['seed']) != (nll.BOOT_B, nll.BOOT_SEED) or r['n_matches'] != n_matches:
            raise StageFailed(f'the {name} interval was made from B = {r["B"]}, seed {r["seed"]}, {r["n_matches"]} matches; the plan '
                              f'fixes B = {nll.BOOT_B}, seed {nll.BOOT_SEED} and the {n_matches} DEV matches')
    pinned = (overdispersion['B'], overdispersion['seed']) == (nll.BOOT_B, nll.BOOT_SEED) and overdispersion['n_matches'] <= n_matches
    if not pinned or mc['S'] != S_B0:
        raise StageFailed('the over-dispersion interval or the simulation count is not the one the plan fixes')
    brier = brier_comparison(mu_b0_60, mu_b1, b2_p, y60, match, B=B, seed=seed)
    return {'results': results, 'verdict': verdict, 'gate_pass': gate_pass(verdict), 'brier': brier,
            'constants': {'B': B, 'seed': seed, 'S_b0': S, 'n_matches': int(n_matches), 'n_rows': int(len(match)),
                          'rule': 'gate_pass = all_pass and nothing missing (R27)'}}


def stage_gate(ctx: Context) -> dict:
    """The gate of D6 and the report-only items, from the files of the stages before: ``gate\\GATE.json``.

    수식
    ----
    (RUN3-12)  cross = max_row[abs(parent_mu_row − worker_mu_row) / max(1, abs(worker_mu_row))].

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      row | 대응이 확인된 DEV 프레임 행 | — | 0 기반, 비어 있지 않음 가정 | frame_row
      parent_mu_row, worker_mu_row | 부모와 작업자의 첫 지평 발생 기대 개수 | 개 | 유한 입력 가정 | pred['mu_b0'][:, 0], on60
      cross | 분모를 하한으로 보정한 최대 차이 | — | ≥ 0 (유한 입력일 때) | cross

    해석
    ----
    적합 단계와 합성 단계의 행 대응을 먼저 확인한 뒤 부모와 작업자의 발생 예측 차이를 보고한다(S3 Task12).
    분모 하한은 작은 기대값에서 비율이 과도하게 커지는 것을 막는 구현 척도이며, 이 진단 자체에는 합격 문턱을 적용하지 않는다.
    실제 관문은 별도의 통계·판정 단계가 계산하며, 이 필드는 그 판정에 더하지 않는다.

    근거
    ----
    S3 계획 Task12(관문 기록); stage_gate(진단 필드).
    """
    from c_ept import towertier
    out = ctx.run / 'gate'
    out.mkdir(parents=True, exist_ok=True)
    labels = open_labels(ctx)
    with np.load(ctx.run / 'compose' / 'b0_dev.npz', allow_pickle=False) as z:
        b0 = {name: z[name] for name in z.files}
    with np.load(ctx.run / 'fit' / 'DEV_PRED.npz', allow_pickle=False) as z:
        pred = {name: z[name] for name in z.files}
    frame_row, match = b0['frame_row'], b0['match']
    if not (np.array_equal(pred['frame_row'], frame_row) and np.array_equal(pred['match'], match)):
        raise StageFailed('the DEV rows of the composition and of the fit stage differ')
    with open(ctx.run / 'labels' / 'ACCOUNTING.json', encoding='utf-8') as f:
        accounting = json.load(f)
    realised = np.asarray(labels.array('labels/v_start'))[frame_row, 0]
    y60 = observed_onsets(ctx.s1, frame_row)
    row_off = np.asarray(ctx.s1.array('matches/row_off')).astype(np.int64)
    role = np.asarray(ctx.s1.array('matches/role'))
    dev = np.flatnonzero(role == C.ROLES.index(DEV_ROLE))
    n_matches = int((np.diff(row_off)[dev] > 0).sum())                # the DEV matches that have a frame row
    with ctx.sampler.scope() as scope:
        gate = gate_results(match=match, y60=y60, expected_onsets60=b0['expected_onsets'][:, 0], expected_value60=b0['expected'][:, 0],
                            realised_value60=realised, mc_mean60=b0['mc_mean'][:, 0], mc_sd60=b0['mc_sd'][:, 0],
                            mu_b1_60=pred['mu_b1'][:, 0], mu_b0_60=b0['expected_onsets'][:, 0], b2_p=pred['b2_p'],
                            accounting_error=accounting['max_abs_error'], n_matches=n_matches, S=_scalar(b0['S']), B=BOOT_B, seed=SEED)
        tier = towertier.TierModel.load(ctx.run / 'fit' / 'tier')
        t0 = np.asarray(ctx.s1.array('frame_rows/t0'))[frame_row]
        tower = tower_pricing_diagnostic(ctx.s1, frame_row, match, t0, tier, ctx.prices)
    on60 = b0['expected_onsets'][:, 0]
    # 식 (RUN3-12)의 분모 하한은 보고 척도이며 별도의 관문 문턱이 아니다(stage_gate).
    cross = float(np.max(np.abs(pred['mu_b0'][:, 0] - on60) / np.maximum(1.0, np.abs(on60))))
    facts = {**gate, 'tower_pricing': tower, 'n_dev_matches_with_rows': n_matches, 'n_dev_matches': int(len(dev)),
             'm2_b0_workers_against_parent_max_relative_difference': cross, 'wall_s': round(scope.wall, 3), **_peaks(scope)}
    ctx.write_json(out / 'GATE.json', facts)
    criteria = ', '.join(f'{k} {"pass" if gate["verdict"][k]["pass"] else "FAIL"}' for k in ('occurrence', 'value', 'accounting', 'mc'))
    ctx.say(f'  gate {"PASS" if gate["gate_pass"] else "FAIL"}: {criteria}')
    return {'gate_pass': gate['gate_pass'], 'verdict_all_pass': gate['verdict']['all_pass'], 'missing': gate['verdict']['missing'],
            'wall_s': facts['wall_s'], 'peak_private_gib': facts['peak_private_gib'],
            'peak_private_upper_gib': facts['peak_private_upper_gib']}


STAGE_FUNCTIONS = {'levels': stage_levels, 'labels': stage_labels, 'fit': stage_fit, 'compose': stage_compose, 'gate': stage_gate}


# ---- the run -----------------------------------------------------------------------------------------------------------------------
def prerequisite_problem(requested, complete) -> str | None:
    """The sentence that refuses ``requested`` (stage names) when a stage in it needs an earlier stage that is neither in ``complete`` nor
    in ``requested`` too (a stage works on the output files of the stage before it); None when every stage has what it needs.

    요청 단계가 필요로 하는 앞 단계가 준비되었는지 확인한다(S3 Task12).
    """
    wanted = [s for s in STAGES if s in requested]
    for stage in wanted:
        missing = [s for s in STAGES[:STAGES.index(stage)] if s not in complete and s not in wanted]
        if missing:
            return f'the stage {stage} needs the stage {missing[0]} to be complete first: add it to --stages'
    return None


def _record_stage_result(run: Path, stage: str, record: dict) -> None:
    """Put ``record`` into RUN.json as ``<stage>_result``. The key is added with ``runlog.amend_manifest``, which never changes a key that
    is there. The one exception is made here: a call that was cut between this write and the mark in STAGES.json left the result of a stage
    that is not complete, and the rerun of that stage replaces it. No other key of RUN.json is touched."""
    key = f'{stage}_result'
    path = run / 'RUN.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if key not in manifest:
        runlog.amend_manifest(run, {key: record})
        return
    manifest[key] = record
    store_core.write_atomic(path, json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + '\n')


def run_stages(ctx: Context, ledger: Ledger, requested: tuple) -> dict:
    """Run the stages of ``requested`` (in the order of ``STAGES``) that are not complete: E6 before each, its result in RUN.json
    (``<stage>_result``: the result is key-scanned, then recorded) and only then its mark in STAGES.json, so that a stage that is marked
    complete always has its result. Returns ``{'run': [...], 'skipped': [...]}``. RunRefused when a stage needs one that is not complete
    or was begun on other inputs.

    수식
    ----
    (RUN3-13)  stage_within_cap = (gib(peak) ≤ 40) (gib(peak)가 None이면 None),
             run_wall = round(Σ_stage (stage_wall or 0), 1),
             run_peak = max(None이 아닌 stage_peak), 없으면 None,
             run_upper = max(None이 아닌 stage_upper), 없으면 None,
             run_within_cap = (run_peak ≤ 40) (run_peak가 None이면 None).
             stage_wall = round(scope.wall, 3); 호출 wall_s = round(call_wall, 1).

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      stage, stage_wall, run_wall, call_wall | 단계·단계/전체/호출 경과 시간 | —; s | 다섯 단계; 비음수 | STAGES, wall_s, scope.wall
      peak, gib(peak), stage_peak, run_peak | 바이트 표본 최대·반올림 GiB·단계/전체 최대 | byte; GiB | (RUN3-2), None 가능 | _peaks, recs
      stage_upper, run_upper | 단계에 기록한 수명 상한과 그 최대 | GiB | (RUN3-2), None 가능 | peak_private_upper_gib
      stage_within_cap, run_within_cap | 기록된 한도 이내 서술 표시 | — | bool 또는 None, 재점검 관문과 구별 | within_memory_cap

    해석
    ----
    단계 경과 시간과 메모리는 단계 범위의 값을 우선하며, 한도 초과는 경고로 기록하고 그 자체로 예외를 던지지 않는다(S3 R42·R43).
    모든 단계가 완료되고 최종 결과가 아직 없을 때 저장된 단계 기록을 합쳐 최종 실행 결과로 등록한다.
    합산 시간은 각 단계에 마지막으로 기록된 시간의 합이므로 중단·재시도한 모든 호출의 누적 시간과 다르다(S3 Task12).
    within_memory_cap은 반올림된 표본 최대값의 40 GiB 이하 여부인 서술 표시이며, 재점검 관문의 40 GiB 미만 조건을 대신하지 않는다(S3 R38·R43).
    이 서술 표시와 재점검 관문은 모두 표본 최대값을 사용하며 수명 상한으로 판단하지 않는다(S3 R43·R79).

    근거
    ----
    S3 계획 Task12(단계 기록); S3 G-S3b B3; S3 판정 R38·R42·R43·R79; _finish(최종 집계).
    """
    wanted = [s for s in STAGES if s in requested]
    problem = prerequisite_problem(requested, {s for s in STAGES if ledger.complete(s)})
    if problem:
        raise RunRefused(problem)
    ran, skipped = [], []
    for stage in wanted:
        if ledger.complete(stage):
            ctx.say(f'stage {stage}: complete in an earlier call, skipped')
            skipped.append(stage)
            continue
        ctx.guard()                                               # E6 before every stage
        inputs = stage_inputs(ctx, stage)
        digest = runlog.sha256_json(inputs)
        ledger.begin(stage, inputs, digest)                       # RunRefused here: the stage was begun on other inputs and not finished
        ctx.stage, ctx.inputs_sha256 = stage, digest
        ctx.say(f'stage {stage}: started')
        before = ctx.guard.snapshot()
        with ctx.sampler.scope() as scope:
            result = STAGE_FUNCTIONS[stage](ctx)
        outputs = stage_outputs(ctx.run, stage)
        peak = _gib(scope.peak)
        facts = {'wall_s': round(scope.wall, 3), **_peaks(scope),
                 'within_memory_cap': None if peak is None else peak <= MEMORY_CAP_GIB,
                 **{f'e6_{k}': v for k, v in ctx.guard.since(before).items()}}
        record = _clean({**result, **facts})                      # the time and the peaks of the stage are those of the ledger
        ctx.assert_no_keys(render(record))                        # a match key in the result is an error: nothing is recorded ...
        _record_stage_result(ctx.run, stage, record)              # ... the result goes into RUN.json first ...
        ledger.finish(stage, outputs, **facts)                    # ... and only then is the stage marked complete
        ctx.say(f'stage {stage}: done, {facts["wall_s"]} s, peak {peak} GiB')
        if peak is not None and peak > MEMORY_CAP_GIB:
            print(f'WARNING: the peak private memory of the stage {stage} was {peak} GiB, above the cap of {MEMORY_CAP_GIB:g} GiB',
                  file=sys.stderr, flush=True)
        ran.append(stage)
    return {'run': ran, 'skipped': skipped}


def _base_hashes(s1, s2, s3a: dict, smoke: dict | None) -> dict:
    return {'s1_files_sha256': runlog.sha256_json(s1.manifest['files']), 's2_files_sha256': runlog.sha256_json(s2.manifest['files']),
            'prices_sha256': s3a['prices_sha256'], 'rules_sha256': s3a['rules_sha256'],
            'smoke_sha256': None if smoke is None else smoke['selection_sha256']}


def _check_resumable(run: Path, hashes: dict, smoke: dict | None, a) -> dict:
    """RUN.json of the run to resume, after it was compared with the inputs of this call: RunRefused when the stores, the s3a files or the
    smoke selection differ from the ones the run was made with."""
    path = run / 'RUN.json'
    if not path.is_file():
        raise RunRefused(f'{run.name} has no RUN.json: it is not an S3 run folder')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('stage') != 's3':
        raise RunRefused(f'{run.name} is not an S3 run folder')
    s1, s2, s3a, smoke_ = (manifest.get(k) or {} for k in ('s1', 's2', 's3a', 'smoke'))
    recorded = {'s1_files_sha256': s1.get('files_sha256'), 's2_files_sha256': s2.get('files_sha256'),
                'prices_sha256': s3a.get('prices_sha256'), 'rules_sha256': s3a.get('rules_sha256'),
                'smoke_sha256': smoke_.get('selection_sha256')}
    differ = sorted(k for k in recorded if recorded[k] != hashes[k])
    if differ:
        what = {'s1_files_sha256': 'the S1 store', 's2_files_sha256': 'the S2 store', 'prices_sha256': 'PRICES.json',
                'rules_sha256': 'RULE_PAYOUTS.json', 'smoke_sha256': 'the smoke selection (--smoke)'}
        raise RunRefused(f'{run.name} was made from other inputs than this call gives: {", ".join(what[k] for k in differ)} differ. '
                         'A run folder holds one set of inputs; give the same ones, or start a new run folder.')
    return manifest


def check_resume_code(manifest: dict, now: dict) -> None:
    """RunRefused (nothing is run, nothing is recorded) when a resume would run other code than the run began on, or code that cannot be
    named. ``manifest`` is RUN.json of the run (``code_commit``, ``code_dirty`` of its first call), ``now`` the ``runlog.code_state()`` of
    this call. A resume is refused when the commit differs, when the dirty flag differs, and whenever the tree is dirty now (``code_dirty``
    is True, or None because git cannot answer): the models and chunks that a cut call left were made by the code of RUN.json, and a run
    folder never mixes the output of two codes. A first call on a dirty tree is allowed (its state is recorded), a resume of it is not.

    이어하기에서 원 실행의 코드 상태와 현재 상태를 대조하고 더러운 작업 트리를 거부한다(S3 G-S3b B3).
    """
    began = (manifest.get('code_commit'), manifest.get('code_dirty'))
    problems = []
    if now['code_commit'] != began[0]:
        problems.append('the commit is another')
    if now['code_dirty'] != began[1]:
        problems.append('the dirty flag is another')
    if now['code_dirty'] is not False:
        problems.append('the tree is dirty (uncommitted changes in c_ept/, or git cannot say), so the code cannot be named')
    if problems:
        raise RunRefused(f'a resume must run the code the run began on, from a clean tree, but {"; ".join(problems)} (the run began on '
                         f'commit {str(began[0])[:12]}, dirty {began[1]}; this call runs on commit {str(now["code_commit"])[:12]}, '
                         f'dirty {now["code_dirty"]}). Commit the changes and resume on that commit, or start a new run folder.')


def _resume_hint(a, run: Path) -> str:
    parts = ['python', '-m', 'c_ept.run_s3', '--s1', str(a.s1), '--s2', str(a.s2), '--s3a', str(a.s3a), '--resume', str(run),
             '--workers', str(a.workers)]
    if a.smoke is not None:
        parts += ['--smoke', str(a.smoke)]
    if Path(a.spec) != Path(C.SPEC_PATH):
        parts += ['--spec', str(a.spec)]
    if Path(a.catalog_db) != Path(C.CATALOG_DB):
        parts += ['--catalog-db', str(a.catalog_db)]
    return ' '.join(parts)


def build_parser() -> argparse.ArgumentParser:
    """단계·이어하기·소규모 점검 실행의 호출 인자를 정의한다(S3 Task12)."""
    ap = argparse.ArgumentParser(prog='python -m c_ept.run_s3',
                                 description='S3: victim levels, labels, baseline fits, composition and gate.')
    ap.add_argument('--s1', type=Path, required=True, help='the S1 run folder (or its s1 store folder); read only')
    ap.add_argument('--s2', type=Path, required=True, help='the S2 run folder (or its s2 store folder); read only')
    ap.add_argument('--s3a', type=Path, required=True,
                    help='the s3a run folder (prices\\PRICES.json and rules\\RULE_PAYOUTS.json); their SHA-256 must be those of '
                         'decisions.S3_prices')
    ap.add_argument('--spec', type=Path, default=C.SPEC_PATH, help='the decision record that holds decisions.S3_prices (read only)')
    ap.add_argument('--stages', nargs='+', default=list(STAGES), metavar='STAGE',
                    help=f'the stages to run, from {" ".join(STAGES)} (default: all)')
    ap.add_argument('--smoke', type=int, default=None, metavar='N',
                    help='a smoke run on N FIT, ceil(0.2 N) STOP and ceil(0.3 N) DEV matches (a seeded selection; the rows go to RUN.json)')
    ap.add_argument('--resume', type=Path, default=None, metavar='RUN',
                    help='continue this S3 run folder (the same --s1, --s2, --s3a and --smoke): the stages that are complete are skipped')
    ap.add_argument('--workers', type=int, default=8,
                    help=f'processes of the raw reads and of the composition (1 to {MAX_WORKERS}; the composition always uses at least 2)')
    ap.add_argument('--verify', action='store_true', help='check every file of both stores against its recorded SHA-256 first (slow)')
    ap.add_argument('--runs-root', type=Path, default=C.RUNS_ROOT)
    ap.add_argument('--catalog-db', type=Path, default=C.CATALOG_DB)
    return ap


def _validate(a) -> tuple:
    """The requested stages in the order of ``STAGES``; RunRefused for an unknown or repeated stage or a bad number."""
    bad = [s for s in a.stages if s not in STAGES]
    if bad:
        raise RunRefused(f'unknown stage(s) {bad}: the stages are {", ".join(STAGES)}')
    if len(set(a.stages)) != len(a.stages):
        raise RunRefused('a stage is named twice in --stages')
    if not 1 <= a.workers <= MAX_WORKERS:
        raise RunRefused(f'--workers must be 1 to {MAX_WORKERS} (the limit of the process pool on Windows), got {a.workers}')
    if a.smoke is not None and a.smoke < 1:
        raise RunRefused(f'--smoke must be at least 1, got {a.smoke}')
    return tuple(s for s in STAGES if s in a.stages)


def _call_record(a, requested: tuple, resumed: bool) -> dict:
    return {'started': _now(), 'stages_requested': list(requested), 'workers': a.workers, 'resumed': resumed, 'smoke': a.smoke,
            **runlog.code_state()}


def _first_manifest(a, s3a: dict, decisions: dict, smoke: dict | None, hashes: dict, requested: tuple) -> dict:
    """What the first call writes into RUN.json beside the provenance fields of ``runlog.write_manifest``: the arguments, the S1 and S2
    stores (name and the hash of their files), the s3a files and the decision record (hashes), the smoke selection (the match rows) and
    the constants of the composition."""
    return {'args': {'smoke': a.smoke, 'workers': a.workers, 'stages': list(requested), 'verify': bool(a.verify), 'spec': str(a.spec)},
            's1': {'run': _run_name(a.s1), 'files_sha256': hashes['s1_files_sha256']},
            's2': {'run': _run_name(a.s2), 'files_sha256': hashes['s2_files_sha256']},
            's3a': {'run': s3a['run'], 'prices_sha256': hashes['prices_sha256'], 'rules_sha256': hashes['rules_sha256']},
            'spec': {'sha256': decisions['spec_sha256'], 'decision': 'decisions.S3_prices'},
            'smoke': smoke, 'compose_constants': compose_constants(),
            'roles': {'tier': list(TIER_ROLES), 'b0': B0_ROLES, 'b1': B1_ROLES, 'valued': list(LABEL_ROLES)},
            'memory_cap_gib': MEMORY_CAP_GIB}


def _run_name(path) -> str:
    """The run folder name of a store: the folder itself, or the parent of a store folder (the one that holds STORE.json)."""
    path = Path(path)
    return path.parent.name if (path / 'STORE.json').is_file() else path.name


def _run(a, records, out) -> int:
    from c_ept import rulepay, value
    from c_ept.flowprice import _open_s2
    from c_ept.run_s1 import StrataRunning, load_s1, require_no_strata
    requested = _validate(a)
    if a.resume is None:                                          # a new run has no stage complete: the first stage asked for is levels
        problem = prerequisite_problem(requested, ())
        if problem:
            raise RunRefused(problem)
    try:
        require_no_strata()                                       # E6: before a store is opened and before the run folder is made
    except StrataRunning as e:
        return _refuse(f'{e} (The E6 guard of the S1 runner stops the S3 run too; no run folder was made.)')
    decisions = read_decisions(a.spec)
    s3a = check_s3a(a.s3a, decisions)
    try:
        s1 = load_s1(a.s1, verify=a.verify)
    except (FileNotFoundError, ValueError) as e:
        return _refuse(f'the S1 run cannot be used: {e}')
    try:
        s2 = _open_s2(a.s2, a.verify)
    except (FileNotFoundError, ValueError) as e:
        return _refuse(f'the S2 run cannot be used: {e}')
    lock = run = None
    try:
        try:
            prices = value.load_prices(s3a['prices'])
            rules = rulepay.RuleTable.load(s3a['rules'], patches=rulepay.P3_PATCHES)
        except (OSError, ValueError, TypeError) as e:
            return _refuse(f'the approved prices or rules cannot be used: {e}')
        smoke = None if a.smoke is None else select_smoke(s1, a.smoke)
        hashes = _base_hashes(s1, s2, s3a, smoke)
        resumed = a.resume is not None
        if resumed:
            run = Path(a.resume)
            _check_resumable(run, hashes, smoke, a)
        else:
            run = runlog.new_run('s3', root=a.runs_root)
        view = s1 if smoke is None else SubsetStore(s1, smoke)
        ctx = Context(run=run, s1=view, s2=s2, prices=prices, rules=rules, hashes=hashes, smoke=smoke, workers=a.workers,
                      records=records, out=out)
        lock = _acquire_lock(run)
        ctx.say(f'S3 run folder: {run}{" (resumed)" if resumed else ""}; stages {" ".join(requested)}'
                f'{"; smoke " + str(a.smoke) if smoke else ""}; {a.workers} workers')
        if not resumed:
            info = _first_manifest(a, s3a, decisions, smoke, hashes, requested)
            ctx.assert_no_keys(render(info))
            runlog.write_manifest(run, info)
        ledger = Ledger(run)
        if resumed:
            check_resume_code(json.loads((run / 'RUN.json').read_text(encoding='utf-8')), runlog.code_state())
        check_ledger(ctx, ledger)
        call = ledger.begin_call(_call_record(a, requested, resumed))
        t0, done, error = time.perf_counter(), None, None
        try:
            with ctx.sampler:
                done = run_stages(ctx, ledger, requested)
        except BaseException as e:                                # the call is recorded whatever stopped it; the error goes on
            error = e
            raise
        finally:
            call.update(finished=_now(), wall_s=round(time.perf_counter() - t0, 1), **_peaks(ctx.sampler),
                        memory_samples=ctx.sampler.samples, check_calls=ctx.guard.calls,
                        check_s_mean=round(ctx.guard.seconds / max(1, ctx.guard.calls), 4), check_s_longest=round(ctx.guard.longest, 4),
                        stages_run=None if done is None else done['run'], stages_skipped=None if done is None else done['skipped'],
                        stopped_by=None if error is None else type(error).__name__)
            ledger.save()
        return _finish(ctx, ledger, a, run, done)
    except StrataRunning as e:                                    # E6 again between the stages and the units of work
        return _refuse(f'{e} The run folder {run} can be resumed: after the model server is stopped, run  {_resume_hint(a, run)}')
    except RunLocked as e:
        return _refuse(str(e))
    except RunRefused as e:
        return _refuse(str(e))
    finally:
        for store in (s1, s2):
            store.maps.clear()
        gc.collect()
        if lock is not None:
            _release_lock(lock)


def _finish(ctx: Context, ledger: Ledger, a, run: Path, done: dict) -> int:
    """After the stages: when all five are complete, the record of the whole run goes into RUN.json (``s3_result``) and the run is
    registered in the run index (as ``run_s2`` does it only for a finished build).

    식 (RUN3-13)의 해당 항을 계산한다.
    """
    complete = all(ledger.complete(s) for s in STAGES)
    manifest = json.loads((run / 'RUN.json').read_text(encoding='utf-8'))
    if complete and 's3_result' not in manifest:
        recs = ledger.doc['stages']
        peaks = [recs[s].get('peak_private_gib') for s in STAGES if recs[s].get('peak_private_gib') is not None]
        peak = max(peaks) if peaks else None
        uppers = [recs[s].get('peak_private_upper_gib') for s in STAGES if recs[s].get('peak_private_upper_gib') is not None]
        passed = (manifest.get('gate_result') or {}).get('gate_pass')
        runlog.amend_manifest(run, {'s3_result': {
            'complete': True, 'smoke': ctx.smoke is not None, 'wall_s': round(sum(recs[s].get('wall_s') or 0.0 for s in STAGES), 1),
            'stage_wall_s': {s: recs[s].get('wall_s') for s in STAGES},
            'stage_peak_private_gib': {s: recs[s].get('peak_private_gib') for s in STAGES}, 'peak_private_gib': peak,
            'stage_peak_private_upper_gib': {s: recs[s].get('peak_private_upper_gib') for s in STAGES},
            'peak_private_upper_gib': max(uppers) if uppers else None,
            'memory_cap_gib': MEMORY_CAP_GIB, 'within_memory_cap': None if peak is None else peak <= MEMORY_CAP_GIB,
            'calls': len(ledger.doc['calls']), 'gate_pass': passed, 'code_state_at_end': runlog.code_state()}})
        runlog.register(run, json.loads((run / 'RUN.json').read_text(encoding='utf-8')), a.catalog_db)
        ctx.say(f'S3 done: all five stages complete, gate_pass {passed}, peak {peak} GiB; the run is registered')
    else:
        ctx.say(f'S3 call done: ran {" ".join(done["run"]) or "nothing"}, skipped {" ".join(done["skipped"]) or "nothing"}; '
                f'{"every stage is complete" if complete else "not every stage is complete yet"}')
    return 0


def _scrub_text(text: str, keys) -> str:
    """``text`` with every match key replaced by <key> and every game id by <id> (the scrub of ``store_build``)."""
    from c_ept import store_build
    for key in keys:
        text = store_build._scrub(text, key)
    return text


def _set_loky_cpu_count() -> None:
    """Set ``LOKY_MAX_CPU_COUNT`` to the number of physical cores (the logical ones when psutil cannot say) unless the user set it. Without
    it joblib's loky asks the machine for the physical cores with a tool (``wmic``) that a current Windows does not have, and every first
    use of scikit-learn that goes through joblib prints a warning with a traceback. It is the one variable that the parent sets; it is set
    by ``main`` before anything imports scikit-learn or joblib (this module imports neither at module level)."""
    if 'LOKY_MAX_CPU_COUNT' in os.environ:                       # the setting of the user wins, and the machine is not even asked
        return
    physical = psutil.cpu_count(logical=False) if psutil is not None else None
    os.environ.setdefault('LOKY_MAX_CPU_COUNT', str(physical or os.cpu_count() or 1))


def main(argv=None, *, records=None) -> int:
    """The command line (see the module text). ``records`` replaces ``reader.read_match`` in the ``levels`` stage (tests). Returns 0, or
    2 for a refusal or a failed stage (nothing is registered); an unexpected error prints its traceback with the match keys removed and
    returns 1.

    인자 검증 뒤 저장소를 열기 전에 E6를 확인하는 실행 경로로 들어간다(S3 Task12).
    실행 거부와 예외는 정해진 종료 경로로 전달한다.
    """
    _set_loky_cpu_count()
    a = build_parser().parse_args(argv)
    try:
        return _run(a, records, print)
    except RunRefused as e:                                       # before a run folder was made
        return _refuse(str(e))
    except Exception:                                             # noqa: BLE001 - the traceback, scrubbed, is the message
        text = traceback.format_exc()
        try:
            from c_ept.run_s1 import load_s1
            keys = np.asarray(load_s1(a.s1, verify=False).array('matches/key')).tolist()
            text = _scrub_text(text, [k.decode('ascii') for k in keys if k])
        except Exception:                                         # noqa: BLE001 - no store to take the keys from: the text goes out as is
            pass
        print(text, file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
