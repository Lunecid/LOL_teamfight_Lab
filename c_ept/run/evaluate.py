"""Evaluation runner of the run kernel: the identity of every checkpoint, the STOP evaluation, C4, the DEV access ledger and the TEST gate, the stage body of the
supervisor and the ledger records (S4b-prep K6b; decisions D4, D9, D13, D14, D15, D16, D20, D24, D27; plan Q9 and the K3, M1-M12 hand-offs).

What it does. ``start(profile_path, run_dir, ...)`` reads a hash-locked evaluate profile and an evaluation inputs manifest, refuses what is wrong before anything is
reserved and hands the supervisor (``run.supervisor``) a plan of one stage, ``evaluate``: ``evaluate_stage`` -> ``evaluate`` -> ``evaluate/RECORD.json`` (what was evaluated,
under which settings, with which hashes) and ``evaluate/REPORT.json`` (the numbers, and what is unavailable with the reason). ``evaluate(profile, run_dir, *, inputs, ...)`` is
the work as one function of the profile and the run folder (``runners.evaluate`` is its entry point under the name of the plan).

* The identity of a checkpoint is read from its training run (K6a), through the ledger (``Ledger.stage``: both stages finished, every declared output still with the recorded
  hash), the record ``train/RECORD.json`` and the final ``head.json`` (``load_trained``): the eleven settings of ``SETTING_DEFAULTS`` with the old defaults for what the identity
  leaves out. ``check_models`` refuses a mixture (the models of one evaluation are the seeds of one configuration: the whole training identity is equal except the seed and the
  execution keys device and threads), static channels, loss scales or an S2 store that are not the training's, representatives of the gold bins that differ from the evaluation's,
  a loss-scale file of another kill law or K condition (D24), dropped columns that differ between the models and a matched B1 whose dropped columns differ (M5 H3). ``Backend`` hands each model its own kill law, K condition, frame average and zone mark (the model object carries no identity).
* STOP (D13): the whole STOP split, each fit's best checkpoint. The S4b quantities that this runner makes there are the closed-form ones (decision D30): the STOP loss and the per-head blocks of
  ``train.evaluate_stop_blocks`` with the two shared M2 blocks and the masked M3 blocks (D9); the per-head NLL records and the endpoints of the DEV adapter on the STOP rows
  (``adapters.run_inference(role='STOP', monte_carlo=False)``), hence the 60 s onset probability for C4 (D13, with the primary and secondary passes of D14 under the nominal exposure) and the relative
  structural metric R (D16). No Monte Carlo is drawn on STOP (the open M3 value, its sign endpoint, ``draws``): the DEV rehearsal and TEST keep it. THE COMPOSED V_START PASS (K6e, D30) is the profile
  block ``composed`` and is off by default (null): then the evaluation, its record and its report are those of every earlier profile, bit for bit. With the block, on the role STOP and the nominal exposure only,
  every checkpoint gets one Composer pass over the rows of a seeded sample of STOP matches (``vstart.sample_matches``), the nominal one over every row and the observed one over the rows that a horizon leaves out
  (``composed_v_start``), in blocks with one progress line each; ``vstart.summarize`` scores them (the CRPS of the primary and of the secondary population with the weights 1/n_m, the D14 excluded-row share) with
  the Monte Carlo error (batch means over the draws) and the match bootstrap (a prefix of the common resample matrix); ``vstart.occurrence_calibration`` gives the calibration-in-the-large of the occurrence
  probability next to them (the models under-predict occurrence); the per-row losses go to the rows file ``vstart.ROWS_FILE``, by which ``vstart.paired_contrast`` makes the A7 contrast (zone mark on minus off, the
  same rows) of two records. Its parameters (sample size and seeds, draws, batches) are profile values with no default (a later cost probe sets them), and ``chunk_views`` is read by this pass only. The fields
  ``strata``, ``price_sensitivity`` and ``b1_swap_average`` are a task before S8 (M17) and are refused (``UNWIRED_FIELDS``) until they are wired. Each arm used the STOP rows that its own training run recorded for early stopping; this runner
  cannot see B1's row set (a matched B1 stops on the same matched subset as the EPT, ``B1Heads`` records the role only) and records the EPT's STOP row digest from its training identity
  (``selection_data``): the numbers are the decision data of S4b and not a held-out estimate.
* Progress (``StageLog``): every stage of ``evaluate`` (the identity checks, the access, per model the model, the STOP report, each inference pass with one ``block`` stage for each of its 13 head blocks, the
  structural metric, and C4 and the records) writes a flushed ``begin`` and ``end`` line to ``PROGRESS_FILE`` with the stage, the model key, the attempt number, the wall start and end, the elapsed seconds, the
  status and counts (rows, views, draws, matches); no metric is ever written in it. A refusal inside the identity or the access stage leaves the file; a refusal before them writes nothing.
* C4 (D13, ``c4_statistic``): the seed values 20261022, +1, +2 paired by equal value, the mean of the paired differences to the matched 25 % B1 in the 60 s onset Brier, a
  95 % match bootstrap on the first ``bootstrap_B`` rows of the common B = 10,000 matrix over the split's sorted match list, and the stop rule.
* DEV (D4, D20): every DEV evaluation writes one entry to the ``AccessLedger`` before it makes a number; the roles DEV and TEST take the default loader and backend (a test-only
  switch ``allow_custom`` is recorded). TEST (D20): ``HoldoutGate`` needs the S5 lock file with its fixed hash, a finished DEV rehearsal of the same profile (the ``role`` aside), the
  same code, the same loader, backend and inputs manifest and the same training records, whose own ``dev_access`` entry is in the access ledger, and a ledger in which TEST was
  never opened; it opens TEST once, and only for a backend that can evaluate TEST.

Symbols
-------
    P, h     the evaluate profile and its lock (``sha256`` of the file); C(P) = ``profile_without_role_sha256`` (canonical hash of ``as_dict()`` without ``role``)
    M_i      the models of P, i = 1 .. m, with the training seed s_i; T_i the ``TrainedRun`` of M_i (``load_trained``)
    Z_i      the eleven settings of T_i (``settings_of``: kill_law, k_condition, decay_groups, terrain_smooth_weight, swap_mode, eval_average, on_wall_cap, zone_mark, column_drop,
             stats_population, k_reps_sha256)
    R        the 31 representatives of the gold bins of the evaluation (``evaluation_reps``); rep(T_i) the ``k_reps_sha256`` of the training
    I, L     the code identity of the run and of the DEV rehearsal (``run.identity.Identity``), C(I) its canonical hash
    ids, c   the sorted match list of the judging split and the common resample matrix (B x n), c_{b,m} the count of match m in resample b
    src      the sources of a run: the inputs loader and the backend (``module:name``), the sha256 of the inputs manifest, the test-only switch ``allow_custom`` and the stage body
             (``body_record``: its module, its name and the sha256 of its file; K6c, K6b review r4)
    rec(T_i) ``train_record_sha256``: the hash of the training record of checkpoint i;   entry(R) the ``dev_access`` entry that the rehearsal R wrote (seq and hash)

Equations
---------
    (EV-1)  accept(P)  <=>  keys(checkpoints) = models  and  every T_i is read through a finished ledger  and  Z_1 = ... = Z_m  and  enc_1 = ... = enc_m
    (EV-2)  accept(models, inputs)  <=>  (EV-1)  and  static/scales hashes equal the identity's  and  (k_condition = binned => rep(T_i) = sha(R))
                                         and  condition(scales file) = (kill_law, k_condition, sha(R))  and  (B1 given => dropped(B1) = dropped(T_i) for every i)
    (EV-3)  DEV:  the entry ``dev_access`` is appended before the first model is built
    (EV-4)  TEST: open(P, I)  <=>  backend.evaluates(TEST)  and  sha(lock) = P.holdout_lock_sha256  and  lock names the ledger (folder, created entry)  and  finished(rehearsal, role DEV)  and  C(P) = C(P_rehearsal)  and  C(I) = C(L)
                  and  src = src(rehearsal)  and  rec(T_i) = rec(rehearsal's M_i) for every i  and  entry(rehearsal) in the ledger  and  not opened;   TEST is never opened
                  twice (an exclusive marker file as well as the ledger entry), and only a backend that can evaluate TEST may open it (the opening is not spent on nothing)
    (EV-5)  C4 = Delta_bar = (1/3) sum_s Delta_s with Delta_s the match-balanced Brier(EPT_s) - Brier(B1) (see ``c4_statistic``); the matrix rows used are c[:B], B <= 10,000
    (EV-6)  passes(P) = [(nominal, primary), (observed, secondary)] if exposure(P) = nominal else [(observed, all)]: C4 and the structural metric read the first pass, the second
            is reported next to it (D14)
    (EV-7)  accept(models) additionally  <=>  id(T_i) \\ {seed, device, threads} equal (the FIT and STOP row digests without their batch seed)  and  s2_store(T_i) = sha(STORE.json
            of the inputs)  and  dropped(T_i) equal;   src = (loader, backend, sha(inputs manifest), allow_custom, body), body = (module, name, sha(file of the module)) of the stage
            body that the supervisor starts; the TEST gate requires a named body and src(TEST) = src(rehearsal), the child checks that the file still has the hash (``verify_body``)
    (EV-8)  apply(numeric policy) before any model is built (D10 P6, K6c): the state read back is the policy or the run is refused (exit code 2); it is written to the record as ``numeric``

해석
----
평가는 체크포인트의 학습 정체성을 폴더를 훑어서가 아니라 학습 실행의 원장 단계 기록과 K6a 기록(RECORD.json)에서 읽는다. 어느 파일이든 원장이 해시한 뒤 바뀌었으면 거부한다. 한 평가의 모형들은
한 설정의 시드들이어야 하므로 설정이 섞이면 거부한다(구성이 다른 후보의 비교는 프로파일을 따로 두고 기록끼리 견준다). 킬 법칙과 K 조건은 B1·B0·손실 척도·합성기가 한 조건으로만 만들어져
있어서 섞일 수 없고, 구간 대표값은 학습 행이 쓴 31개와 평가가 넣는 31개가 같아야 한다. 모든 평가 설정(킬 법칙, K 조건, 프레임 평균, 구역 표지)은 모형 객체가 스스로 알려 주지 않으므로
정체성에서 읽은 값을 `run_inference`와 모형 생성에 넘기고, 원장에는 정체성이 생략하는 기본값까지 명시한 값으로 적는다.

DEV는 탐색용이다. 그 평가는 숫자를 내기 전에 열람 원장에 한 줄을 남기고(계산했을 뿐 읽지는 않음: `opened` False), 사람이 그 숫자를 읽으면 하네스 `DEV_ACCESS_LEDGER.md`에 행을 더한다
(코드는 그 파일을 고치지 않는다). TEST는 잠금이 가리키는 원장 안에서 한 번만 열린다. 관문은 TEST 프로파일이 고정한 잠금 파일(`holdout_lock_sha256`; 원장 폴더와 created 줄의 해시를
적은 파일, 결정 D28), 같은 프로파일(role만 다름)·같은 코드·같은 로더와 백엔드와 입력 목록·같은 학습 기록의
체크포인트로 끝난 DEV 리허설(원장에 그 리허설 자신의 줄이 있어야 한다), 열린 적 없는 원장이 모두 있을 때만 열고, 연 뒤에는 도중에 실패해도 열린 것으로 센다. 기본 백엔드가 TEST를
평가할 수 없으면(TEST 읽는 곳이 생기는 S8까지) 관문을 열기 전에 거부하므로 단 한 번의 개봉이 아무것도 못 만드는 실행에 쓰이지 않는다. 이 모듈은 관문과 그 시험만 구현한다: 실제 TEST 자료는 없고 읽지도 않는다.
새 원장, 개봉 전에 복사한 원장 폴더, 원장과 표지를 지우고 다시 만든 원장은 잠금과 맞지 않아 거부된다(폴더와 created 줄의 해시). 남는 허점(코드가 막지 못한다): 옛 원장 파일을 같은 자리에
되돌려 놓고 표지 파일만 지우는 것(그 원장은 잠금이 적은 폴더와 created 줄을 그대로 가진다), 그리고 해시 사슬을 다시 계산한 기록자가 만든 줄. 그러므로 `test_opened` 줄의 해시를 하네스(git)에 커밋해
두는 것을 권고한다.

한 평가의 모형들은 시드(와 device, threads)만 다르고 학습 정체성의 나머지(lr, 가중치 감쇠, 워밍업, 인내, 에폭, 모형 규격, 감쇠·열 제거 해시, FIT·STOP 행 요약에서 배치 시드를 뺀 것, S2 저장소
해시, 환경)가 같아야 한다. 구성이 다른 후보(λ, lr, A6, A7, P1)의 비교는 기록끼리 한다. 명목 노출(nominal)은 두 번의 추론을 한다: 1차(명목, 주 인구)와 2차(관측, 나머지 행; D14).

못 만드는 것은 이유와 함께 '없음'으로 보고한다. `run_inference`는 role을 받으므로(DEV 기본, STOP 허용, TEST는 관문이 준 증표가 있어야 한다) STOP의 끝점·NLL 기록과 그 위의 C4·상대 구조 지표를
전체 STOP에서 만든다. STOP 경기는 두 쪽 모두의 조기 종료 자료이므로(EPT는 그 부분집합, B1은 `fit_head(fit_rows, stop_rows)`) 기록에 그 사실을 적는다(`selection_data`). STOP에서는 닫힌 꼴 값만 만들고
몬테카를로는 하지 않는다(결정 D30: 열린 M3 값의 추첨은 DEV 예행과 TEST에만 남는다). 합성된 V_start(Composer)는 프로파일의 `composed` 칸이 있을 때만, STOP과 명목 노출에서, 값이 매겨진 STOP 경기의 시드 표본에서 한다(K6e, D30; A7의 V_start CRPS 이득과 D14의 제외 행 몫). 칸이 꺼져(null) 있으면
이 실행기의 기록과 보고는 그 칸이 생기기 전과 비트까지 같다. 표본 크기·시드·모의 횟수·배치 수는 비용 탐침이 정하므로 기본값이 없고, 몬테카를로 오차는 모의를 K배치로 나눈 배치 평균으로, 경기 표본의 불확실성은 공통 재표본
행렬의 앞 B행으로 따로 낸다. 읽는 곳이 아직 없는 프로파일 칸(strata, price_sensitivity, b1_swap_average)은 기본값이 아니면 거부하고, chunk_views는 합성 칸 없이 적으면 거부한다(조용히 무시하지 않는다). 재표본에서 분모가 0인 비는 정의되지 않으므로 그
행은 만들지 않고 이유를 적는다(M10 n1). 단계마다 걸린 시간은 `PROGRESS_FILE`에 한 줄씩 남긴다(`StageLog`: 단계 이름, 모형, 시작·끝 시각, 경과 초, 개수만; 지표 값은 없다).

K6c: the stage body is named in the sources (``body_record``) and compared by the TEST gate, the numeric policy is held before any model is built, and the command line (``c_ept.run.cli``) refuses
a stage body other than the default one for the roles DEV and TEST. The measurement runner is ``c_ept.run.measure``.

Not done here (later tasks): the wiring of the profile fields strata, price_sensitivity and b1_swap_average (M17, before S8), the equivalence run (K8), the profile files (K9) and the values of the composed block (a cost probe
that needs its own approval; X1). The role TEST of the default backend stays closed (``Backend.evaluates``) until a TEST reader of the stores exists (S8): the stores and the row views refuse a holdout role, with or without the opening of the gate.
"""
from __future__ import annotations

import contextlib
import copy
import dataclasses
import hashlib
import importlib
import importlib.util
import json
import os
import re
import time
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import torch

from c_ept.ept import checkpoints, compose_heads, data, encoder, heads
from c_ept.ept import train as ept_train
from c_ept.evaluation import adapters
from c_ept.evaluation import guard as eval_guard
from c_ept.evaluation import inference, metrics, proper
from c_ept.models import baselines_b0, baselines_b1
from c_ept.preprocessing import features, terrain
from c_ept.run import identity as run_identity
from c_ept.run import io as run_io
from c_ept.run import ledger as run_ledger
from c_ept.run import numeric as run_numeric
from c_ept.run import profile as run_profile
from c_ept.run import runners, supervisor, vstart
from c_ept.valuation import compose, towertier, value

STAGE = 'evaluate'
RECORD_FILE = 'evaluate/RECORD.json'
REPORT_FILE = 'evaluate/REPORT.json'
RECORD_SCHEMA = 'ept-run-evaluate-record-v2'               # v2 (K6d): + agreed.stop_sha256, selection_data, inference_constants.monte_carlo (draws are 0 when none was made)
REPORT_SCHEMA = 'ept-run-evaluate-report-v1'
ACCESS_SCHEMA = 'ept-run-access-ledger-v1'
ACCESS_FILE = 'ACCESS_LEDGER.json'
OPENED_FILE = 'TEST_OPENED.json'
ACCESS_LOCK = 'ACCESS_LEDGER.lock'
ACCESS_EVENTS = ('created', 'dev_access', 'test_opened')
APPENDED_EVENTS = ('dev_access', 'test_opened')
_ENTRY_FIELDS = frozenset({'seq', 'event', 'when', 'previous_sha256', 'record'})


class EvaluationRefused(run_io.IORefused):
    """The evaluation cannot be made as the profile asks: a refusal (exit code 2), nothing was computed."""


class AccessRefused(EvaluationRefused):
    """The DEV access ledger or the TEST gate refuses: a ledger that is missing, malformed or busy, a TEST that was opened already, a gate whose evidence is
    missing. A refusal (exit code 2). The message names the rule, never a path or a key."""


def profile_record(profile) -> dict:
    """``profile.as_dict()`` as the records show it: without the ``composed`` key when the block is null (K6e). A profile that does not ask for the composed pass is then written to the record, the report
    header and the ledger header exactly as it was before the field existed (bit for bit), and a profile that asks for it shows the block.

    해석: 합성 칸이 꺼진 프로파일은 그 칸이 생기기 전과 글자 하나까지 같게 기록한다."""
    return {key: value for key, value in profile.as_dict().items() if not (key == 'composed' and value is None)}


def profile_without_role_sha256(profile) -> str:
    """The canonical hash of ``profile.as_dict()`` without its ``role``: what a DEV rehearsal and the TEST run must share (D20: "the same code and profiles").

    The composed block (K6e) is part of it when it is set. A null block is left out, so that a profile that does not ask for the composed pass has the hash that it had before the field existed
    (the gate compares a rehearsal with its TEST run under one code, and older records stay comparable).

    해석: 리허설(DEV)과 본 판정(TEST)은 role만 다르므로 파일 해시가 같을 수 없다. role을 뺀 정규 해시가 같아야 같은 프로파일이다. 합성 칸이 꺼져(null) 있으면 해시에서 뺀다. 그 칸이 생기기 전의 해시와 같게 하기 위해서이다."""
    if getattr(profile, 'kind', None) != 'evaluate':
        raise EvaluationRefused('the evaluation takes an evaluate profile')
    return run_io.canonical_sha256({key: value for key, value in profile_record(profile).items() if key != 'role'})


# ---- the access ledger (D4, D20) ---------------------------------------------------------------------------------------------------------------
class AccessLedger:
    """The ledger of the accesses to DEV and TEST: one JSON file ``ACCESS_FILE`` in a folder, outside the run folders (a run folder can be deleted, this must not).

    Entries ``{seq, event, when, previous_sha256, record}`` in a chain: ``previous_sha256`` is the canonical hash of the entry before it. ``created`` is the first
    entry; ``dev_access`` is written once by every DEV evaluation before it computes a number; ``test_opened`` is written once before a TEST number is
    computed, in this ledger. TEST is also marked by the file ``OPENED_FILE``, created exclusively before the entry is written, so that cutting the last entry out of the
    ledger file does not make TEST unopened again. A second writer is refused by the lock file ``ACCESS_LOCK`` (created exclusively; a lock that a crash left is
    deleted by hand). The harness file ``DEV_ACCESS_LEDGER.md`` is a record for people and is never edited by code.

    Anchor and residual (decision D28, review C1, m3): the ledger by itself is not the anchor. The TEST gate (``HoldoutGate``) opens TEST only in the ledger that the S5 lock file names,
    by folder and by the hash of the ``created`` entry (the lock's sha256 is ``holdout_lock_sha256`` of the TEST profile). A new ledger, a copy of the folder made before the opening and
    a ledger that was deleted with its marker and made again are therefore refused (the folder or the created entry, which holds the time of its making, differs). What code cannot stop:
    the old ledger file put back in its place with the marker file deleted (it has the folder and the created entry that the lock names), and a chain that someone recomputes (it has no
    value fixed outside it). Commit the hash of the ``test_opened`` entry to the harness (git) as the out-of-band record.

    해석: 열람 원장은 실행 폴더가 지워져도 남아야 하므로 실행 폴더 밖의 한 폴더에 둔다. DEV 평가는 숫자를 내기 전에 한 줄을 남기고, TEST는 이 원장에서 한 번만 열린다. 마지막 줄을
    잘라내도 TEST가 열린 적 없는 것으로 돌아가지 않도록 배타적으로 만든 표지 파일을 따로 둔다. 새 원장·복사한 폴더·지웠다 다시 만든 원장은 관문이 잠금으로 거부한다. 옛 원장 파일을 되돌려 놓고
    표지만 지우는 길은 코드가 막지 못하므로 `test_opened` 줄의 해시를 하네스에 커밋해 밖의 기록으로 삼는다."""

    def __init__(self, folder):
        self.folder = Path(folder)
        self.path = self.folder / ACCESS_FILE
        self.marker = self.folder / OPENED_FILE
        self.lock = self.folder / ACCESS_LOCK

    @classmethod
    def create(cls, folder, *, clock=time.time) -> AccessLedger:
        """Make the ledger in ``folder`` (made if missing) with its ``created`` entry; ``AccessRefused`` if the folder has a ledger or a TEST marker already."""
        ledger = cls(folder)
        if os.path.lexists(ledger.path) or os.path.lexists(ledger.marker):
            raise AccessRefused('an access ledger already exists in this folder; it is never overwritten')
        entry = {'seq': 0, 'event': 'created', 'when': float(clock()), 'previous_sha256': None, 'record': {}}
        run_io.write_json_atomic(ledger.path, {'schema': ACCESS_SCHEMA, 'entries': [entry]})
        return ledger

    def entries(self) -> list:
        """The verified entries (new dicts); ``AccessRefused`` for a missing ledger, another schema or fields, a first entry that is not ``created``, an event, a number
        or a chain that does not fit."""
        if not os.path.lexists(self.path):
            raise AccessRefused('no access ledger in this folder (a ledger is made once, by hand, before the first DEV evaluation)')
        refused = False
        try:
            document = run_io.read_json(self.path)
        except run_io.IORefused:
            refused = True
        if refused:
            raise AccessRefused('the access ledger is not a readable strict JSON file')
        if set(document) != {'schema', 'entries'}:
            raise AccessRefused('the access ledger has other fields than schema and entries')
        if document['schema'] != ACCESS_SCHEMA:
            raise AccessRefused('the access ledger has another schema')
        entries = document['entries']
        if not isinstance(entries, list) or not entries:
            raise AccessRefused('the access ledger holds no entries; the first must be created')
        previous = None
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict) or set(entry) != _ENTRY_FIELDS:
                raise AccessRefused(f'access ledger entry {index} has other fields than {sorted(_ENTRY_FIELDS)}')
            if entry['event'] not in ACCESS_EVENTS:
                raise AccessRefused(f'access ledger entry {index} has an event that is none of {ACCESS_EVENTS}')
            if (index == 0) != (entry['event'] == 'created'):
                raise AccessRefused('the first entry of the access ledger, and only it, must be created')
            if entry['seq'] != index or isinstance(entry['seq'], bool):
                raise AccessRefused(f'access ledger entry {index} has the wrong seq')
            if isinstance(entry['when'], bool) or not isinstance(entry['when'], (int, float)) or not isinstance(entry['record'], dict):
                raise AccessRefused(f'access ledger entry {index} has a malformed time or record')
            if entry['previous_sha256'] != (None if previous is None else run_io.canonical_sha256(previous)):
                raise AccessRefused(f'the chain of the access ledger is broken at entry {index}')
            previous = entry
        return copy.deepcopy(entries)

    def test_opened(self) -> bool:
        """True if TEST was opened: the marker file exists or an entry says so (``AccessRefused`` if the ledger does not verify)."""
        entries = self.entries()
        return os.path.lexists(self.marker) or any(entry['event'] == 'test_opened' for entry in entries)

    @contextlib.contextmanager
    def _locked(self):
        try:
            descriptor = os.open(self.lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            raise AccessRefused('the access ledger is locked by another writer (ACCESS_LEDGER.lock). If none is running, a crash left the lock: delete that file '
                                'yourself; the code never deletes it') from None
        try:
            with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
                handle.write(f'pid={os.getpid()}\n')
            yield
        finally:
            with contextlib.suppress(OSError):
                self.lock.unlink()

    def append(self, event: str, record: Mapping, *, clock=time.time) -> dict:
        """Append one entry of ``event`` (``dev_access`` or ``test_opened``) with ``record`` (a strict JSON object) and return it. The ledger must verify; for
        ``test_opened`` TEST must not have been opened (``AccessRefused`` otherwise)."""
        if event not in APPENDED_EVENTS:
            raise AccessRefused(f'only the events {APPENDED_EVENTS} are appended, not {event!r}')
        if not isinstance(record, Mapping):
            raise AccessRefused('the record of an access entry must be an object')
        if not os.path.lexists(self.path):
            raise AccessRefused('no access ledger in this folder (a ledger is made once, by hand, before the first DEV evaluation)')
        with self._locked():
            entries = self.entries()
            entry = {'seq': len(entries), 'event': event, 'when': float(clock()), 'previous_sha256': run_io.canonical_sha256(entries[-1]), 'record': dict(record)}
            run_io.canonical_sha256(entry)                                           # strict JSON, found out before anything is marked
            if event == 'test_opened':
                if self.test_opened():
                    raise AccessRefused('TEST was opened already; it is never opened a second time')
                try:
                    with open(self.marker, 'x', encoding='utf-8') as handle:         # exclusive: the first opener wins, and the marker outlives a cut-back ledger file
                        handle.write('opened\n')
                except FileExistsError:
                    raise AccessRefused('TEST was opened already; it is never opened a second time') from None
            run_io.write_json_atomic(self.path, {'schema': ACCESS_SCHEMA, 'entries': [*entries, entry]})
        return entry


# ---- the gate of the TEST role (D20, Review Focus 5) ------------------------------------------------------------------------------------------
def _is_digest(text) -> bool:
    return isinstance(text, str) and len(text) == 64 and all(c in '0123456789abcdef' for c in text)


# The S5 lock file (decision D28, review C1): strict JSON that names the access ledger that TEST may be opened in, by its folder and by the hash of its ``created`` entry. The sha256
# of this file is the field ``holdout_lock_sha256`` of the TEST profile (hence of the profile without its role, which the DEV rehearsal shares), so the lock is fixed by the profile and
# not by whoever starts the run.
LOCK_SCHEMA = 'ept-run-holdout-lock-v1'
_LOCK_FIELDS = frozenset({'schema', 'access_dir', 'created_entry_sha256'})


def _folder_text(folder) -> str:
    return Path(os.path.abspath(folder)).as_posix()


def holdout_lock_document(access: AccessLedger) -> dict:
    """The content of the S5 lock file for the ledger ``access`` (which must exist and verify): ``{'schema', 'access_dir', 'created_entry_sha256'}`` with the absolute folder (forward
    slashes) and the canonical hash of the ``created`` entry. The operator writes it as strict JSON once, before the first DEV evaluation, and puts the sha256 of the file into the TEST
    profile as ``holdout_lock_sha256``.

    해석: 열람 원장은 만든 순간의 시각이 들어 있는 첫 줄(created)로 한 번만 만들어진다. 그 줄의 해시와 폴더를 잠금 파일에 적고 그 파일의 해시를 프로파일에 고정하면, 새로 만들거나 복사하거나
    지웠다 다시 만든 원장은 이 잠금과 맞지 않는다."""
    return {'schema': LOCK_SCHEMA, 'access_dir': _folder_text(access.folder), 'created_entry_sha256': run_io.canonical_sha256(access.entries()[0])}


@dataclasses.dataclass(frozen=True)
class HoldoutGate:
    """What the role TEST needs before it is opened (decision D20, anchored by D28), and the one opening.

    ``lock_path``: where the S5 lock file is (its hash is not an argument: the TEST profile fixes it as ``holdout_lock_sha256``). ``rehearsal_run``: the run folder of a finished DEV
    evaluation of the same profile (``role`` aside, so under the same lock) with the same code, the same loader, backend and inputs manifest and the training records of the same
    checkpoints. ``access``: the ``AccessLedger`` that the lock names, in which TEST was never opened and which holds that rehearsal's own ``dev_access`` entry.

    Equations
    ---------
        gate(P, I, src, rec)  <=>  role(P) = TEST  and  sha(lock) = P.holdout_lock_sha256  and  schema(lock) ok  and  folder(access) = lock.access_dir
                         and  C(access.entries[0]) = lock.created_entry_sha256  and  finished(R)  and  role(R) = DEV  and  C(P \\ role) = C(R's profile \\ role)
                         and  C(I) = C(identity of R's header)  and  src = src(R) and not allow_custom(src)  and  rec = rec(R)
                         and  entry(R) = access.entries[seq(R)]  (the very line, by hash)  and  not opened(access)
        open(P, I, src, rec)  =  gate(P, I, src, rec) and then one ``test_opened`` entry (exclusive: the second opening is refused)

      기호 | 뜻
      P, I   | 평가 프로파일, 이 실행의 코드 정체성(Identity)
      R      | DEV 리허설 실행 폴더(원장의 evaluate 단계가 끝난 것)
      C(x)   | 정규 JSON 해시(run.io.canonical_sha256)
      src    | (로더, 백엔드, 입력 목록 sha256, allow_custom, 단계 본문: 모듈·이름·파일 해시): 리허설과 본 실행이 같아야 하고 TEST는 allow_custom을 받지 않으며 본문을 적어야 한다(K6c)
      rec    | {모형: 학습 기록 해시}: 리허설과 본 실행의 체크포인트가 같은 학습 실행에서 나왔는지
      lock   | S5 잠금 파일(``holdout_lock_document``): 원장 폴더와 created 줄의 해시

    해석: 잠금 파일의 해시는 호출자가 넘기지 않고 TEST 프로파일이 고정한다(role을 뺀 프로파일 해시에 들어가므로 DEV 리허설도 같은 잠금 아래에서 만들어진다). 잠금은 원장 폴더와 그 원장의
    created 줄의 해시를 적으므로, 새로 만든 원장, 개봉 전에 복사한 원장 폴더, 원장과 표지를 지우고 다시 만든 원장은 모두 거부된다. 같은 프로파일(role만 다름)·같은 코드·같은 로더·백엔드·입력
    목록·같은 학습 기록의 체크포인트로 DEV 리허설이 끝났고 열람 원장에 그 리허설 자신의 줄이 남아 있고, 원장에 TEST를 연 적이 없을 때만 연다. 같은 두 해시를 가진 아무 줄이나 리허설로 치지
    않는다. 어느 하나라도 없으면 거부하고 아무것도 열지 않는다. 연 다음에는 숫자를 내는 도중에 실패해도 다시 열지 않는다(열린 것으로 센다).

    남는 허점(코드가 막지 못한다): 옛 원장 파일을 같은 자리에 되돌려 놓고 표지 파일만 지우는 것. 그 원장은 잠금이 적은 폴더와 created 줄을 그대로 가진다. 그래서 `test_opened` 줄의 해시를
    하네스(git)에 커밋해 밖의 기록으로 삼고, 개봉 전후 원장 폴더를 사람이 확인하는 것을 권고한다."""

    lock_path: Path
    rehearsal_run: Path
    access: AccessLedger

    def __post_init__(self):
        if not isinstance(self.access, AccessLedger):
            raise AccessRefused('the gate needs an AccessLedger')
        object.__setattr__(self, 'lock_path', Path(self.lock_path))
        object.__setattr__(self, 'rehearsal_run', Path(self.rehearsal_run))

    def _lock(self, profile) -> tuple:
        """``(sha256 of the lock file, its document)`` after checking the file against ``profile['holdout_lock_sha256']`` and its fields; ``AccessRefused`` otherwise."""
        wanted = profile['holdout_lock_sha256']
        if not _is_digest(wanted):
            raise AccessRefused('the TEST profile fixes no S5 lock (holdout_lock_sha256 is null): the lock is the anchor of the gate')
        try:
            actual = run_io.sha_file(self.lock_path)
        except OSError:
            raise AccessRefused('the S5 lock file cannot be read') from None
        if actual != wanted:
            raise AccessRefused('the S5 lock file differs from the holdout_lock_sha256 that the profile fixes')
        refused = False
        try:
            document = run_io.read_json(self.lock_path)
        except run_io.IORefused:
            refused = True
        if refused or not isinstance(document, dict) or set(document) != _LOCK_FIELDS or document['schema'] != LOCK_SCHEMA:
            raise AccessRefused('the S5 lock file is not a lock document (schema, access_dir and created_entry_sha256 only)')
        if not isinstance(document['access_dir'], str) or not _is_digest(document['created_entry_sha256']):
            raise AccessRefused('the S5 lock file has a malformed access_dir or created_entry_sha256')
        return actual, document

    def _anchored(self, document: Mapping) -> list:
        """The verified entries of the access ledger after checking that it is the ledger that the lock names: the folder, and the hash of the ``created`` entry (a ledger that was
        made again, or another one, has a created entry of another time)."""
        if os.path.normcase(_folder_text(self.access.folder)) != os.path.normcase(document['access_dir']):
            raise AccessRefused('the access ledger folder is not the one that the S5 lock names')
        entries = self.access.entries()                                              # a missing or broken ledger refuses: "never opened" must be provable
        if run_io.canonical_sha256(entries[0]) != document['created_entry_sha256']:
            raise AccessRefused('the created entry of the access ledger is not the one that the S5 lock names (the ledger was made again or is another one)')
        return entries

    def _rehearsal(self, profile) -> dict:
        refused = None
        try:
            ledger = run_ledger.Ledger(self.rehearsal_run)
            stage = ledger.stage(STAGE)
        except (run_ledger.LedgerRefused, run_io.IORefused):
            refused = 'the DEV rehearsal run has no readable ledger with an evaluate stage'
        if refused is not None:
            raise AccessRefused(refused)
        if stage.state != 'finished' or {RECORD_FILE, REPORT_FILE} - set(stage.outputs):
            raise AccessRefused('the DEV rehearsal is not a finished evaluate stage with its record and its report')
        for name, recorded in stage.outputs.items():
            try:
                actual = run_io.sha_file(self.rehearsal_run / name)
            except OSError:
                actual = None
            if actual != recorded:
                raise AccessRefused('a file of the DEV rehearsal changed after the ledger hashed it')
        try:
            record = run_io.read_json(self.rehearsal_run / RECORD_FILE)
        except run_io.IORefused:
            raise AccessRefused('the record of the DEV rehearsal is not readable') from None
        if record.get('schema') != RECORD_SCHEMA or record.get('role') != 'DEV':
            raise AccessRefused('the rehearsal must be an evaluation of the role DEV')
        wanted = profile_without_role_sha256(profile)
        if record.get('profile_without_role_sha256') != wanted:
            raise AccessRefused('the DEV rehearsal ran another profile (the profile without its role differs)')
        models = record.get('models')
        if not isinstance(models, dict) or not all(isinstance(entry, dict) for entry in models.values()):
            raise AccessRefused('the record of the DEV rehearsal names no training records of its models')
        return {'profile_without_role_sha256': wanted, 'identity_sha256': run_io.canonical_sha256(ledger.header.identity.to_record()),
                'record_sha256': stage.outputs[RECORD_FILE], 'report_sha256': stage.outputs[REPORT_FILE], 'sources': record.get('sources'),
                'models': {name: entry.get('train_record_sha256') for name, entry in models.items()}, 'access': record.get('access')}

    @staticmethod
    def _body_named(sources: Mapping) -> None:
        """``AccessRefused`` unless ``sources['body']`` is a record of a stage body (``BODY_FIELDS``: module, name and the sha256 of the file). The role TEST compares the stage body with the
        rehearsal's, and two runs that name none would be equal by omission: the gate wants it named (K6b review r4)."""
        body = sources.get('body')
        ok = (isinstance(body, Mapping) and set(body) == BODY_FIELDS and isinstance(body['module'], str) and body['module'] != '' and isinstance(body['name'], str)
              and body['name'] != '' and _is_digest(body['file_sha256']))
        if not ok:
            raise AccessRefused('the role TEST compares the stage body with its rehearsal: this run names no stage body that is acceptable (sources.body: module, name and the sha256 of '
                                'its file); a body that is not named is not equal to another')

    @staticmethod
    def _same_evaluation(rehearsal: Mapping, sources: Mapping, models: Mapping) -> None:
        """``AccessRefused`` unless this run has the loader, the backend, the inputs manifest and the stage body of the rehearsal (``sources``, key by key: the stage body is the record of
        its module, name and file hash, K6c) and the training records of its checkpoints (``models``: ``{name: train_record_sha256}``, the same names and the same hashes). Review I2 and m4."""
        if sources.get('allow_custom'):
            raise AccessRefused('the role TEST never takes the test-only switch allow_custom')
        recorded = rehearsal['sources'] if isinstance(rehearsal['sources'], Mapping) else {}
        differ = sorted(key for key in set(recorded) | set(sources) if recorded.get(key) != sources.get(key))
        if differ or not recorded:
            raise AccessRefused('this run and the DEV rehearsal differ in ' + (', '.join(differ) or 'sources') + ' (the loader, the backend, the inputs manifest and the switch '
                                'allow_custom of a TEST run are those of its rehearsal)')
        theirs, ours = rehearsal['models'], dict(models)
        wrong = sorted(name for name in set(theirs) | set(ours) if theirs.get(name) != ours.get(name))
        if wrong:
            raise AccessRefused('the checkpoints of this run are not those of the DEV rehearsal (the training records differ for models: ' + ', '.join(wrong) + ')')

    @staticmethod
    def _logged(entries: list, rehearsal: Mapping) -> None:
        """``AccessRefused`` unless the record of the rehearsal names an entry of this ledger: the entry at its ``seq`` is a ``dev_access`` with the recorded hash and with the
        profile and code hashes of the rehearsal. A ``dev_access`` line of another run for the same profile and code is not the rehearsal's own line."""
        access = rehearsal['access']
        seq = access.get('seq') if isinstance(access, Mapping) else None
        valid = (isinstance(seq, int) and not isinstance(seq, bool) and 0 <= seq < len(entries) and isinstance(access.get('entry_sha256'), str)
                 and access.get('event') == 'dev_access')
        if valid:
            entry = entries[seq]
            valid = (entry['event'] == 'dev_access' and run_io.canonical_sha256(entry) == access['entry_sha256']
                     and entry['record'].get('profile_without_role_sha256') == rehearsal['profile_without_role_sha256']
                     and entry['record'].get('identity_sha256') == rehearsal['identity_sha256'])
        if not valid:
            raise AccessRefused('the DEV rehearsal was not logged in this access ledger (its own dev_access entry, named by its record, is not an entry of the ledger)')

    def check_evidence(self, profile, *, sources: Mapping, models: Mapping) -> dict:
        """The evidence for the profile (a TEST profile) of a run with ``sources`` and the training records ``models``: ``{'lock_sha256', 'rehearsal', 'never_opened'}``;
        ``AccessRefused`` for any gap. Nothing is written."""
        if getattr(profile, 'kind', None) != 'evaluate' or profile['role'] != 'TEST':
            raise AccessRefused('the gate is for the role TEST of an evaluate profile')
        lock, document = self._lock(profile)
        entries = self._anchored(document)
        rehearsal = self._rehearsal(profile)
        if self.access.test_opened():
            raise AccessRefused('TEST was opened already; it is never opened a second time')
        self._body_named(sources)
        self._same_evaluation(rehearsal, sources, models)
        self._logged(entries, rehearsal)
        return {'lock_sha256': lock, 'rehearsal': rehearsal, 'never_opened': True}

    @staticmethod
    def check_identity(evidence: Mapping, identity) -> None:
        """``AccessRefused`` unless the code identity ``identity`` of this run equals the identity of the rehearsal's header (the commit and the hash of every imported file)."""
        if not isinstance(identity, run_identity.Identity):
            raise AccessRefused('the code identity of the run must be an Identity')
        if run_io.canonical_sha256(identity.to_record()) != evidence['rehearsal']['identity_sha256']:
            raise AccessRefused('the code identity of this run differs from the identity of the DEV rehearsal')

    def open(self, profile, identity, *, sources: Mapping, models: Mapping, clock=time.time) -> adapters.HoldoutOpening:
        """Check everything and write the ``test_opened`` entry; return it as the ``adapters.HoldoutOpening`` (the entry as a dict, and the capability that ``run_inference`` demands for the role
        TEST: only this function can make it). After this call TEST counts as opened whatever happens next.

        해석: 열린 줄이 곧 TEST를 평가할 수 있다는 증표이다. 증표는 이 함수 안에서만 만들어지고, 평가 함수는 증표 없이는 TEST 행을 읽지 않는다."""
        evidence = self.check_evidence(profile, sources=sources, models=models)
        self.check_identity(evidence, identity)
        entry = self.access.append('test_opened', {
            'profile_sha256': profile.sha256, 'profile_without_role_sha256': evidence['rehearsal']['profile_without_role_sha256'], 'lock_sha256': evidence['lock_sha256'],
            'rehearsal_record_sha256': evidence['rehearsal']['record_sha256'], 'identity_sha256': evidence['rehearsal']['identity_sha256'],
            'identity_commit': identity.commit, 'sources': dict(sources), 'models': dict(models), 'access_dir': _folder_text(self.access.folder),
            'created_entry_sha256': run_io.canonical_sha256(self.access.entries()[0])}, clock=clock)
        return adapters.HoldoutOpening(entry)


# ---- the trained runs: the identity of every checkpoint, read through the ledger and the record of its training run (K6a) ---------------------
class CheckpointRefused(EvaluationRefused):
    """A checkpoint, its training run or the set of checkpoints cannot be evaluated: a ledger or a file that does not agree, an identity that is not classified, models
    that do not fit together. A refusal (exit code 2). The message names the model and the rule, never a path or a key."""


class MixedSettings(CheckpointRefused):
    """The models that are evaluated together differ in a setting of their training identity (the models of one evaluation are the seeds of one configuration)."""


# What the training identity (``train._identity``) leaves out when a setting has its old default ("absent means the default", plan M1): the ten settings that
# ``runners.IDENTITY_DEFAULTS`` names, and the sha256 of the representatives of the gold bins that only a binned run has.
SETTING_DEFAULTS = MappingProxyType({**runners.IDENTITY_DEFAULTS, 'k_reps_sha256': None})
STAGE_OUTPUTS = MappingProxyType({'stats': (runners.STATS_FILE, runners.STATS_RECORD_FILE), 'train': (runners.TRAIN_RECORD_FILE, runners.E6_FILE)})


def settings_of(identity: Mapping) -> dict:
    """The eleven settings of a training identity with the old defaults filled in for what the identity leaves out (``SETTING_DEFAULTS``).

    해석: 학습 정체성은 기본값인 설정을 적지 않는다("없음 = 기본값"). 평가는 모든 설정을 값으로 읽어 모형 생성, run_inference, 합성기에 넘기고, 원장에는 명시한 값을 적는다."""
    return {name: identity.get(name, default) for name, default in SETTING_DEFAULTS.items()}


def _posix(path) -> str:
    return Path(os.path.abspath(path)).as_posix()


@dataclasses.dataclass(frozen=True)
class TrainedRun:
    """One trained model as its training run (K6a) records it: ``settings`` (``settings_of`` the training identity of the final head), ``head`` (the final ``head.json``,
    whose ``identity`` is the training identity), ``record`` (``train/RECORD.json``) and its hash, ``best_path`` (the ``best.pt`` that the head names; its hash was
    checked), ``stats`` (the statistics of the run, ``stats/STATS.json``) and ``seed``."""

    name: str
    folder: Path
    seed: int
    settings: Mapping = dataclasses.field(hash=False)
    head: Mapping = dataclasses.field(hash=False)
    record: Mapping = dataclasses.field(hash=False)
    record_sha256: str
    best_path: Path
    stats: object = dataclasses.field(hash=False)

    @property
    def identity(self) -> Mapping:
        return self.head['identity']

    @property
    def encoder(self) -> Mapping:
        return self.head['identity']['encoder']

    def bound_files(self) -> list:
        """The files the evaluation reads from this run, as sorted absolute paths with forward slashes: the training record, the statistics, the final head and
        the best weights. The stage of an evaluation run lists them among its inputs, so the ledger hashes them before and after the stage."""
        final = self.folder / self.record['checkpoint']['final'] / 'head.json'
        return sorted(_posix(path) for path in (self.folder / runners.TRAIN_RECORD_FILE, self.folder / runners.STATS_FILE, final, self.best_path))


def _refuse_ledger(name: str, why: str):
    raise CheckpointRefused(f'model {name!r}: the training run {why}')


def _stage_outputs(name: str, folder: Path, ledger) -> dict:
    """``{relative path: sha256}`` of the outputs of the stages ``stats`` and ``train`` of a training run, after checking that both stages are finished with
    their declared outputs, that the stage file agrees with the ledger and that every output still has the hash that the ledger recorded."""
    outputs = {}
    for stage, wanted in STAGE_OUTPUTS.items():
        try:
            record = ledger.stage(stage)
        except run_ledger.LedgerRefused:
            _refuse_ledger(name, f'has no {stage} stage in its ledger')
        if record.state != 'finished' or set(wanted) - set(record.outputs):
            _refuse_ledger(name, f'has no finished {stage} stage with its declared outputs')
        refused = None
        try:
            stage_file = run_io.read_json(folder / run_ledger.STAGE_FOLDER / f'{stage}.json')
        except run_io.IORefused:
            refused = 'the stage file is missing or not readable'
        if refused is None and (stage_file.get('stage') != stage or stage_file.get('outputs') != dict(record.outputs) or stage_file.get('inputs') != dict(record.inputs)):
            refused = 'the stage file does not agree with the ledger'
        if refused is not None:
            _refuse_ledger(name, f'has a {stage} stage whose record is not sound ({refused}); ledger refused')
        for relative in wanted:
            try:
                actual = run_io.sha_file(folder / relative)
            except OSError:
                actual = None
            if actual != record.outputs[relative]:
                _refuse_ledger(name, f'has a file ({relative}) that changed after the ledger hashed it')
            outputs[relative] = actual
    return outputs


def load_trained(name: str, folder) -> TrainedRun:
    """The ``TrainedRun`` of the training run in ``folder`` (K6a: ``runners.start``), read through its ledger and its records, nothing guessed from the folder.

    Checks (``CheckpointRefused`` for each): the folder has a readable ledger with the finished stages ``stats`` and ``train`` whose declared outputs still have the
    hashes the ledger recorded; ``train/RECORD.json`` is a K6a record; the newest checkpoint is the record's final one, its ``head.json`` has the hash the record
    names, the head says ``done``; every key of the head's identity is classified for the evaluation (``evaluation.guard.CLASSIFIED_IDENTITY_KEYS``: the guard is closed,
    a setting that a later task adds is refused until someone decides); the model is an EPT; the identity agrees with the explicit identity of the record; the
    ``best.pt`` named by the head has the recorded size and hash and is listed with that hash in the record; the statistics file agrees with the record.

    해석: 체크포인트의 정체성은 폴더를 훑어서가 아니라 학습 실행의 원장 단계 기록과 K6a 기록(RECORD.json)에서 읽는다. 어느 한 파일이라도 원장이 해시한 뒤 바뀌었으면 거부한다."""
    folder = Path(folder)
    refused = None
    try:
        ledger = run_ledger.Ledger(folder)
    except (run_ledger.LedgerRefused, run_io.IORefused):
        refused = 'has no readable ledger'
    if refused is not None:
        _refuse_ledger(name, refused)
    outputs = _stage_outputs(name, folder, ledger)
    try:
        record, record_sha = run_io.read_json_with_sha(folder / runners.TRAIN_RECORD_FILE)
        checkpoint = record['checkpoint']
        explicit = record['identity']
        stats_summary = record['stats']
    except (run_io.IORefused, KeyError, TypeError):
        _refuse_ledger(name, 'has a training record that is not readable')
    if record_sha != outputs[runners.TRAIN_RECORD_FILE] or record.get('schema') != runners.TRAIN_RECORD_SCHEMA:
        _refuse_ledger(name, 'has a record that is not a K6a training record')
    out_dir = folder / runners.CHECKPOINT_DIR
    try:
        found = checkpoints.latest_metadata(out_dir)
    except (RuntimeError, OSError, ValueError, KeyError, TypeError):
        found = None
    if found is None:
        _refuse_ledger(name, 'has no readable committed checkpoint')
    final, head, _ = found
    head_relative = f"{checkpoint['final']}/head.json"
    if final.relative_to(folder).as_posix() != checkpoint['final']:
        _refuse_ledger(name, 'has a newest checkpoint other than the final one of its record (head)')
    try:
        head_sha = run_io.sha_file(final / 'head.json')
    except OSError:
        head_sha = None
    if head_sha is None or head_sha != checkpoint['head_sha256'] or checkpoint['files'].get(head_relative) != head_sha:
        _refuse_ledger(name, 'has a head.json that is not the one of its record (head)')
    if head.get('done') is not True:
        _refuse_ledger(name, 'has a final head that is not done')
    identity = head['identity']
    unknown = sorted(set(identity) - eval_guard.CLASSIFIED_IDENTITY_KEYS)
    if unknown:
        _refuse_ledger(name, f'has identity keys that are not classified for the evaluation: {unknown}')
    spec = identity.get('model_spec')
    if spec is not None and spec.get('architecture') != 'ept_s_v1':
        _refuse_ledger(name, 'is not an EPT (the evaluation of other architectures belongs to S7)')
    settings = settings_of(identity)
    if explicit != {key: settings[key] for key in runners.IDENTITY_DEFAULTS}:
        _refuse_ledger(name, 'has an identity that differs from the explicit identity of its record')
    best = head['best']
    best_path = out_dir / best['folder'] / 'best.pt'
    try:
        best_sha, best_bytes = run_io.sha_file(best_path), best_path.stat().st_size
    except OSError:
        best_sha = best_bytes = None
    if best_sha != best['sha256'] or best_bytes != best['bytes'] or checkpoint['files'].get(f"{runners.CHECKPOINT_DIR}/{best['folder']}/best.pt") != best['sha256']:
        _refuse_ledger(name, 'has a best.pt that is not the one of its head and its record (best)')
    try:
        stats = data.Stats.load(folder / runners.STATS_FILE)
    except (ValueError, OSError, KeyError, TypeError):
        _refuse_ledger(name, 'has a statistics file that cannot be read')
    if stats.sha256 != stats_summary.get('sha256'):
        _refuse_ledger(name, 'has statistics other than the ones of its record')
    return TrainedRun(name=name, folder=folder, seed=int(identity['seed']), settings=MappingProxyType(settings), head=head, record=record, record_sha256=record_sha,
                      best_path=best_path, stats=stats)


def load_models(profile) -> list:
    """The ``TrainedRun`` of every model of the evaluate profile, ordered by training seed. ``profile['checkpoints']`` maps each model name to the folder of its
    training run (K6a); the keys are matched against ``profile['models']`` in both directions (a model without a checkpoint, a checkpoint without a model), and two models
    of one training seed are refused (the seed values pair the arms)."""
    if getattr(profile, 'kind', None) != 'evaluate':
        raise EvaluationRefused('the evaluation takes an evaluate profile')
    models, folders = list(profile['models']), dict(profile['checkpoints'])
    for model in models:
        if model not in folders:
            raise CheckpointRefused(f'model {model!r} has no checkpoint in the profile')
    for key in sorted(folders):
        if key not in models:
            raise CheckpointRefused(f'checkpoint {key!r} belongs to no model of the profile')
    trained = sorted((load_trained(model, folders[model]) for model in models), key=lambda run: run.seed)
    if len({run.seed for run in trained}) != len(trained):
        raise CheckpointRefused('two models have the same training seed; the seed values 20261022, +1, +2 pair the arms')
    return trained


# ---- the inputs of an evaluation and the checks of the models against them -----------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class EvalInputs:
    """What an evaluation reads besides the checkpoints: ``base`` (``runners.Inputs``: the stores, the static channels, the thirteen loss scales and the path of the
    scales file), and for the inference of the DEV adapter ``prices`` (``value.load_prices``), ``tier`` (``towertier.TierModel``), ``b1`` (a fitted ``B1Heads``, the matched
    25 % B1 of C4), ``b1_files`` (``{relative name: path}`` of the files of its folder, hashed from the disk for the record: the B1 folder hash beside the C4 row, review m7), ``b0`` (a
    fitted ``B0Heads``) and ``b0_files`` (``{name: path}`` of the saved B0 files, hashed directly from the disk for the ledger, M11 m6)."""

    base: object
    prices: object = None
    tier: object = None
    b1: object = None
    b0: object = None
    b0_files: Mapping = dataclasses.field(default_factory=dict, hash=False)
    b1_files: Mapping = dataclasses.field(default_factory=dict, hash=False)


def evaluation_reps(inputs: EvalInputs) -> dict:
    """``{'sha256', 'fallback', 'array'}``: the 31 representatives of the gold bins as the evaluation makes them, ``value.fit_gold_bin_means(population='in_scope')`` of the
    stores of ``inputs`` (the one function that also makes the K block of the binned training rows), their sha256, the bins that had no in-scope FIT entry, and the array."""
    found = value.fit_gold_bin_means(inputs.base.s1, inputs.base.labels, population='in_scope')
    return {'sha256': value.gold_reps_sha256(found.means), 'fallback': [int(bin_) for bin_ in found.fallback], 'array': found.means}


MAY_DIFFER = ('seed',)                                      # the one key in which the models of one evaluation differ ...
EXECUTION_KEYS = ('device', 'threads')                      # ... and the execution keys that may also differ (listed in the record; the lead ruling on review I5)


def comparable_identity(identity: Mapping) -> dict:
    """The training identity without the keys in which the seeds of one configuration differ: ``seed``, ``device``, ``threads``, and the batch seed inside the row digests ``fit`` and
    ``stop`` (``train._rows_digest`` holds the shuffle seed, which is the training seed). Everything else must be equal: the learning rate, the weight decay, the warmup, the patience,
    the epochs, the model spec, the hashes of the decay groups and the dropped columns, the membership (the row digests), the S2 store, the dtype, the deterministic algorithms and torch.

    해석: 한 구성의 세 시드는 시드(와 배치 섞기 시드, 실행 장치·스레드 수)만 다르다. 학습률이나 FIT 구성원이 다른 실행을 시드 평균에 섞으면 C4가 다른 구성의 평균이 된다."""
    skip = set(MAY_DIFFER) | set(EXECUTION_KEYS)
    out = {key: val for key, val in identity.items() if key not in skip}
    for key in ('fit', 'stop'):
        if isinstance(out.get(key), Mapping):
            out[key] = {name: val for name, val in out[key].items() if name != 'seed'}
    return out


def check_agreement(trained: list) -> TrainedRun:
    """The first model of ``trained`` after checking that the models are the seeds of one configuration: the training identities are equal apart from ``MAY_DIFFER`` and
    ``EXECUTION_KEYS`` (``comparable_identity``), which includes the eleven settings of ``SETTING_DEFAULTS`` (with the old defaults) and the encoder; ``MixedSettings`` names every key
    that differs; ``CheckpointRefused`` for no model. Needs no inputs, so a command can refuse before it reserves anything."""
    if not trained:
        raise CheckpointRefused('there is no model to evaluate')
    first = trained[0]
    differ = [name for name in SETTING_DEFAULTS if any(run.settings[name] != first.settings[name] for run in trained)]
    reference = comparable_identity(first.identity)
    others = [comparable_identity(run.identity) for run in trained]
    for key in sorted({key for item in [reference, *others] for key in item}):
        if key not in differ and any(item.get(key, _ABSENT) != reference.get(key, _ABSENT) for item in others):
            differ.append(key)
    if differ:
        raise MixedSettings('the models differ in ' + ', '.join(differ) + ': the models of one evaluation are the seeds of one configuration (the training identity is equal apart '
                            'from the seed and the execution keys; evaluate another configuration with its own profile)')
    return first


_ABSENT = object()                                          # a key that one identity has and another leaves out is a difference, whatever its value


def check_models(trained: list, inputs: EvalInputs) -> dict:
    """Refuse (``CheckpointRefused``; the scales through ``runners.ScalesConditionRefused``) what does not fit together, return what the models agree on.

    1. The models are the seeds of one configuration: the training identities are equal apart from the seed, ``device`` and ``threads`` (``check_agreement``; ``MixedSettings``
       names every key that differs); the execution keys are listed in the result.
    2. The static channels, the loss scales and the S2 store (``STORE.json``) of ``inputs`` are those of the training (their hashes in the training identity).
    3. Under the binned K condition the representatives of the training (``k_reps_sha256`` of the identity) are those of the evaluation (``evaluation_reps``).
    4. The loss-scale file was made under the training's kill law and K condition and, when binned, with these representatives (decision D24, plan Q9).
    5. The dropped columns of every model's statistics are equal, and equal to those of a matched B1 when there is one (M5 hand-off H3).

    Returns ``{'settings', 'encoder', 'execution' ({model: {device, threads}}), 'fit_sha256' (the FIT row digest without its batch seed, one for all models), 's2_store_json_sha256',
    'reps' (or None), 'scales': {'file_sha256', 'condition'}, 'dropped': {'columns', 'sha256'}}``.

    해석: 한 평가의 모형들은 한 설정의 시드들이어야 한다. 킬 법칙과 K 조건이 섞이면 B1·B0·합성기·척도가 한 조건으로만 만들어져 있으므로 거부한다. 학습의 구간 대표값이 평가의
    대표값과 다르면 K 머리가 학습에서 본 값의 집합과 평가가 넣는 값의 집합이 어긋난다. 평가가 읽는 S2 저장소가 학습이 읽은 것과 다르면 같은 행이라는 보장이 없으므로 거부한다."""
    first = check_agreement(trained)
    base = inputs.base
    static_sha = hashlib.sha256(np.asarray(base.static).tobytes()).hexdigest()
    scales_sha = hashlib.sha256(json.dumps({key: float(base.scales[key]) for key in sorted(base.scales)}).encode()).hexdigest()
    try:
        store_sha = run_io.sha_file(Path(base.s2.file))                  # the same bytes that ``train._store_json_sha`` hashed
    except (OSError, AttributeError, TypeError):
        raise CheckpointRefused('the S2 store file (STORE.json) of the inputs cannot be read, so it cannot be compared with the one of the training (S2 store)') from None
    for run in trained:
        if run.identity['static_sha256'] != static_sha:
            raise CheckpointRefused(f'model {run.name!r}: the static channels of the inputs are not those of its training (static)')
        if run.identity['scales_sha256'] != scales_sha:
            raise CheckpointRefused(f'model {run.name!r}: the loss scales of the inputs are not those of its training (scales)')
        if run.identity['s2_store_json_sha256'] != store_sha:
            raise CheckpointRefused(f'model {run.name!r}: the S2 store of the inputs is not the one of its training (S2 store)')
    reps = None
    if first.settings['k_condition'] == 'binned':
        reps = evaluation_reps(inputs)
        for run in trained:
            if run.settings['k_reps_sha256'] != reps['sha256']:
                raise CheckpointRefused(f'model {run.name!r}: the representatives of the gold bins of its training (k_reps_sha256) differ from those of the evaluation')
    if base.scales_path is None:
        raise runners.ScalesConditionRefused('the inputs name no scales file, so the condition of the loss scales cannot be checked')
    condition = runners.check_scales_condition({'kill_law': first.settings['kill_law'], 'k_condition': first.settings['k_condition']}, base.scales_path,
                                               None if reps is None else reps['sha256'])
    wanted = list(first.stats.dropped)
    for run in trained:
        if list(run.stats.dropped) != wanted:
            raise CheckpointRefused(f'model {run.name!r}: the dropped columns of its statistics differ from those of the other models (dropped)')
    if inputs.b1 is not None and list(inputs.b1.dropped) != wanted:
        raise CheckpointRefused(f'model {first.name!r}: the dropped columns of its statistics differ from those of the matched B1 (dropped)')
    fit, stop = (comparable_identity(first.identity).get(key) for key in ('fit', 'stop'))
    return {'settings': dict(first.settings), 'encoder': dict(first.encoder), 'reps': reps, 's2_store_json_sha256': store_sha,
            'execution': {run.name: {key: run.identity.get(key) for key in EXECUTION_KEYS} for run in trained},
            'fit_sha256': None if fit is None else run_io.canonical_sha256(fit), 'stop_sha256': None if stop is None else run_io.canonical_sha256(stop),
            'stop_rows': None if stop is None else {key: stop[key] for key in ('role', 'batches', 'rows_sha256') if key in stop},
            'scales': {'file_sha256': run_io.sha_file(base.scales_path), 'condition': condition}, 'dropped': {'columns': wanted, 'sha256': run_io.canonical_sha256(wanted)}}


# ---- C4 (decision D13) -------------------------------------------------------------------------------------------------------------------------
class Unavailable(Exception):
    """A part of the evaluation that cannot be made here. ``reason`` goes into the report next to the item (a text without a path or a key); it is not an error of the
    run: the other parts are still reported."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


C4_ENDPOINT = 'M2/t0/H60/occurrence'                    # the 60 s onset probability: D13 "C4 ... on 60 s onset Brier"
C4_BOOTSTRAP_B = 2000                                   # D13: the first 2,000 rows of the common B = 10,000 matrix
C4_ADJUSTED_NOTE = 'outside the D15/D20 Holm family: C4 is one decision of D13 (the stop rule), not a row of the seven; p is the raw two-sided p and there is no family-adjusted p'
C4_RULES = MappingProxyType({'stop': 'the point estimate of the EPT is not better than the matched B1 (difference >= 0): stop and report to the author',
                             'proceed': 'the interval lies below 0: proceed',
                             'proceed_inconclusive': 'the point estimate is better but the interval contains 0: proceed with the inconclusive flag'})


def c4_statistic(found: Mapping, ids, matrix) -> dict:
    """The C4 row of decision D13: the mean over the seed values 20261022, +1, +2 of the paired difference EPT minus the matched B1 in the 60 s onset Brier, with a match
    bootstrap on ``matrix`` and the stop rule.

    ``found`` maps each seed value of ``inference.SEEDS`` to the endpoint record ``M2/t0/H60/occurrence`` of that seed's model (``run_inference`` ``endpoints``: ``match``,
    ``row``, ``y``, ``p_ept``, ``p_b1``); the three records are on the same rows and carry the same B1 probabilities (the one matched B1; ValueError otherwise). ``ids`` is the
    sorted match list of the judging split, ``matrix`` the first rows of the common resample matrix (``inference.resample_matrix(len(ids))``). ``Unavailable`` when a
    resample has no present match (the ratio is undefined there; M10 n1).

    수식
    ----
    (C4-1)  a_{s,m} = (1/n_m) Σ_{r in m} [(p^EPT_{s,r} − y_r)² − (p^B1_r − y_r)²],   d_m = [n_m ≥ 1],   Δ_s = Σ_m a_{s,m} / Σ_m d_m
    (C4-2)  Δ̄ = (Δ_1 + Δ_2 + Δ_3)/3,   Δ̄*_b = mean_s ( Σ_m c_{b,m} a_{s,m} / Σ_m c_{b,m} d_m ),   ci95 = quantiles(Δ̄*, 0.025, 0.975),   p = two-sided p of Δ̄ against Δ̄*
    (C4-3)  rule = stop if Δ̄ ≥ 0;  proceed if ci95_high < 0;  else proceed_inconclusive

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      s       | 시드 값(20261022, +1, +2), 같은 값끼리 짝 | — | 3개 | inference.SEEDS
      Δ_s     | 시드 s의 EPT − B1 브라이어 차이(음수가 EPT에 유리) | 확률² | 유한 | per_seed[s]['diff']
      c_{b,m} | 공통 재표본 행렬에서 경기 m이 뽑힌 횟수 | 개 | 0..n | matrix

    해석
    ----
    매칭된 25 % B1은 한 모형이므로 세 시드 모두 같은 B1과 짝지어진다(같은 시드 값끼리의 짝). 경기 안 행 평균(가중 1/n_m)을 경기별 분자로, 있음 표지를 분모로 하는 비 추정량을
    공통 행렬에서 재표본마다 다시 합한다. 점 추정이 B1보다 나쁘거나 같으면(차이 ≥ 0) 멈추고 저자에게 보고하며(D13), 구간은 항상 함께 보고한다. C4는 D15/D20의 Holm 가족(일곱 줄) 밖의
    하나의 판정이므로 가족 조정 p는 없다(`adjusted_p` null과 이유 `adjusted_p_note`): `p`는 조정하지 않은 양쪽 p이다. D13의 판정은 `rule`이고, D15의 가족 행에 쓰는 주장·철회 낱말(`verdict`)은 이 행에 두지 않는다. 재표본 행 수가 D13이 정한 2,000이 아니면 `bootstrap_flag`가 그것을 알린다."""
    seeds = inference.SEEDS
    if not isinstance(found, Mapping) or set(found) != set(seeds):
        raise ValueError(f'C4 needs the endpoint of exactly the seed values {seeds}, paired by equal value')
    first = found[seeds[0]]
    for seed in seeds:
        record = found[seed]
        if record['name'] != C4_ENDPOINT:
            raise ValueError(f'C4 reads the endpoint {C4_ENDPOINT}')
        if any(not np.array_equal(record[key], first[key]) for key in ('match', 'row', 'y')):
            raise ValueError('the arms of C4 are not on the same rows (match, row and outcome must be equal for the three seeds)')
        if not np.array_equal(record['p_b1'], first['p_b1']):
            raise ValueError('the B1 probabilities differ between the seeds: C4 compares the three seeds with the one matched B1')
    match, y = np.asarray(first['match']), np.asarray(first['y'], dtype=np.float64)
    _, inverse, counts = np.unique(match, return_inverse=True, return_counts=True)
    weight = 1.0 / counts[inverse.reshape(-1)]                                              # 1/n_m for every row (D15)
    present = inference.match_presence(match, ids)
    ones = np.ones((1, len(present)))
    p_b1 = np.asarray(first['p_b1'], dtype=np.float64)
    per_seed, draws = {}, []
    for seed in seeds:
        p_ept = np.asarray(found[seed]['p_ept'], dtype=np.float64)
        numerator = inference.match_aggregate(match, (p_ept - y) ** 2 - (p_b1 - y) ** 2, ids)[1]
        point = float(inference.bootstrap_stat(numerator, present, ones)[0])                  # (C4-1)
        try:
            draws.append(inference.bootstrap_stat(numerator, present, matrix))
        except ValueError as error:
            raise Unavailable(f'C4 is not computed: {error} (a resample without a present match, M10 n1)') from None
        per_seed[seed] = {'ept': proper.brier(p_ept, y, weight), 'b1': proper.brier(p_b1, y, weight), 'diff': point}
    delta = inference.paired_seed_mean({seed: per_seed[seed]['diff'] for seed in seeds})     # (C4-2)
    mean_draws = inference.paired_seed_mean(draws)
    ci = [float(v) for v in np.quantile(mean_draws, [0.025, 0.975])]
    p = inference.two_sided_p(mean_draws, delta)
    rule = 'stop' if delta >= 0 else 'proceed' if ci[1] < 0 else 'proceed_inconclusive'     # (C4-3)
    rows = int(matrix.shape[0])
    return {'name': 'C4', 'endpoint': C4_ENDPOINT, 'metric': 'brier', 'direction': 'EPT minus matched B1; below 0 favours the EPT',
            'n_matches': int(len(ids)), 'n_present_matches': int(present.sum()), 'n_rows': int(len(match)), 'bootstrap_B': rows,
            'bootstrap_flag': None if rows == C4_BOOTSTRAP_B else f'bootstrap_B is {rows}, not the first {C4_BOOTSTRAP_B:,} rows of the common matrix that decision D13 fixes for C4',
            'per_seed': {str(seed): {key: float(val) for key, val in per_seed[seed].items()} for seed in seeds},
            'delta': float(delta), 'ci95': ci, 'p': float(p), 'adjusted_p': None, 'adjusted_p_note': C4_ADJUSTED_NOTE,
            'rule': rule, 'rule_text': C4_RULES[rule]}


# ---- the backend: the heavy side of an evaluation (replaceable in a test) ------------------------------------------------------------------------
MC_DRAWS = 1000                                          # the Monte Carlo draws of the open M3 value: the default of run_inference, a runner constant (not a profile field)
MC_SEED = inference.SEED                                 # and its seed
COMPARISON_TEXT = 'S4b kernel evaluation: the seeds of one EPT configuration versus the matched 25% B1 on the same rows (decision D13)'
TEST_UNAVAILABLE = ('the role TEST is evaluated only with the opening that HoldoutGate.open returns (the one test_opened entry of the access ledger); this call has none, so the endpoints, the '
                    'NLL records and the quantities built on them (C4, the relative structural metric) are not made')
V_START_UNAVAILABLE = ('the composed V_start (a Composer pass over the rows of the role with its prices and tier model) is not made by this runner: it is a later task on a sample of STOP '
                       '(decision D30); composer_settings gives the settings such a pass takes from the checkpoint')


def _no_check() -> None:
    return None


class Backend:
    """The parts of an evaluation that need the heavy objects (the model, the batches, ``run_inference``). The defaults are the real ones; a test hands in a subclass.

    ``build_model``: the EPT of the checkpoint (the encoder and the zone mark of its training, the static channels of the inputs) with the weights of the ``best.pt`` that
    the head names, in evaluation mode. ``split_ids``: the sorted match numbers of the judging split of ``role``. ``stop_report``: the blocks of ``train.evaluate_stop_blocks``
    on the whole STOP split. ``inference``: ``adapters.run_inference`` of the role (DEV or STOP; TEST only with the ``opening`` that the gate returned, ``Unavailable`` without it) with every
    setting of the checkpoint (kill law, K condition, frame average, zone mark) and of the profile (winner truth, B0) and the ``exposure`` and
    ``population`` of the pass. ``v_start``: ``composed_v_start`` when the profile has a ``composed`` block, else ``Unavailable`` (see ``V_START_UNAVAILABLE``). ``evaluates(role)``: whether this backend can evaluate the role at all; the default
    answers for DEV and STOP and not for TEST, so that the one opening of TEST is never spent on a run that cannot make a number (review I1). TEST stays False for the default backend
    until the stores have a TEST reader (S8): the row views refuse a holdout role even when the gate has opened it.

    해석: 백엔드는 자기가 평가할 수 없는 role을 스스로 밝힌다. TEST는 한 번만 열리므로 숫자를 못 만드는 백엔드로는 열지 않는다."""

    def evaluates(self, role: str) -> bool:
        return role in ('DEV', 'STOP')

    def build_model(self, trained: TrainedRun, static, device):
        config = encoder.EncoderConfig(**trained.encoder)
        zone_mark = trained.settings['zone_mark']
        model = heads.EPT(config, static, **({} if zone_mark else {'zone_mark': False}))              # the model is built the way it was trained (decision D9)
        model.load_state_dict(torch.load(trained.best_path, weights_only=True, map_location='cpu'), strict=True)       # the best weights only: never last.pt or the optimizer
        return model.to(device).eval()

    def split_ids(self, inputs: EvalInputs, role: str):
        refused = None
        try:
            frame_rows = features.split_rows(inputs.base.s1, role)
        except ValueError:
            refused = f'the matches of the role {role} cannot be listed from the stores here'
        if refused is not None:
            raise Unavailable(refused)
        return np.unique(np.asarray(inputs.base.s1.array('frame_rows/match'))[frame_rows]).astype(np.int64)

    def stop_report(self, model, trained: TrainedRun, inputs: EvalInputs, constants, check) -> dict:
        base, settings = inputs.base, trained.settings
        rows = data.ept_head_rows(base.s1, base.s2, base.labels, 'STOP', kill_law=settings['kill_law'], k_condition=settings['k_condition'])
        stop = data.Batches(data.EngagementRows(rows), constants.batch_size, trained.seed, stats=trained.stats, training=False)          # all STOP matches: no selection (D13)
        report = ept_train.evaluate_stop_blocks(model, stop, base.scales, device=constants.device, guard=check, kill_law=settings['kill_law'],
                                                eval_average=settings['eval_average'], zone_mark=settings['zone_mark'])
        return {'loss_stop': float(report['loss_stop']), 'blocks': {key: {name: float(val) for name, val in entry.items()} for key, entry in report['blocks'].items()}}

    def inference(self, model, trained: TrainedRun, inputs: EvalInputs, *, role: str, profile, reps, constants, check, exposure: str = 'observed', population: str = 'all',
                  opening=None, progress=None) -> dict:
        """``adapters.run_inference`` on the rows of ``role`` (K6d). The bundle declares the same role, and the role TEST also hands over ``opening``, the ``HoldoutOpening`` that the gate
        returned (``Unavailable`` without it). ``Unavailable`` for a role that is none of DEV, STOP, TEST. ``progress``: the callback of ``run_inference`` for the head blocks (None: none).

        해석: 평가 함수가 정한 role의 행만 읽는다. 번들은 같은 role을 선언하고, TEST이면 관문이 준 증표를 함께 넘기며 증표가 없으면 '없음'으로 보고한다."""
        if role not in ('DEV', 'STOP', 'TEST'):
            raise Unavailable(f'the inference of the DEV adapter reads the roles DEV and STOP (and TEST through the gate), not {role}')
        if role == 'TEST' and opening is None:
            raise Unavailable(TEST_UNAVAILABLE)
        monte_carlo = role != 'STOP'                                  # D30: the S4b decisions on STOP are closed-form quantities; the open M3 value is drawn on the DEV rehearsal and on TEST only
        if inputs.b1 is None:
            raise Unavailable('the inputs hold no matched B1, which the inference of the DEV adapter needs')
        if monte_carlo and (inputs.prices is None or inputs.tier is None):
            raise Unavailable('the inputs hold no prices or no tier model, which the Monte Carlo of the open M3 value needs')
        base, settings = inputs.base, trained.settings
        bundle = SimpleNamespace(role=role, s1=base.s1, s2=base.s2, labels=base.labels, static=base.static, prices=inputs.prices, tier=inputs.tier)
        return adapters.run_inference(bundle, model, trained.stats, inputs.b1, check=check if check is not None else _no_check, device=constants.device,
                                      batch_size=constants.batch_size, draws=MC_DRAWS, seed=MC_SEED, scales=base.scales, comparison=COMPARISON_TEXT,
                                      kill_law=settings['kill_law'], winner_truth=profile['winner_truth'], k_condition=settings['k_condition'],
                                      gold_reps='in_scope_fit' if settings['k_condition'] == 'binned' else 'labels', eval_average=settings['eval_average'],
                                      zone_mark=settings['zone_mark'], b0=inputs.b0 if profile['b0'] else None, exposure=exposure, population=population,
                                      role=role, holdout=opening, monte_carlo=monte_carlo, progress=progress)

    def v_start(self, model, trained: TrainedRun, inputs: EvalInputs, *, role: str, profile, reps, constants, check, progress=None) -> dict:
        """The composed V_start pass on a sample of STOP matches (``composed_v_start``) when the profile has a ``composed`` block, else ``Unavailable`` as in every earlier profile
        (``V_START_UNAVAILABLE``). ``progress`` is the optional block callback of the pass (the runner hands ``BlockStages``). The pass refuses a role other than STOP.

        해석: 합성 칸이 없으면 예전처럼 '없음'이다. 칸이 있으면 STOP 표본에서만 합성 V_start를 계산한다."""
        if profile['composed'] is None:
            raise Unavailable(V_START_UNAVAILABLE)
        return composed_v_start(model, trained, inputs, role=role, profile=profile, reps=reps, constants=constants, check=check, progress=progress)


def composer_settings(settings: Mapping, exposure: str) -> dict:
    """The keyword arguments of ``compose.Composer`` for the heads of a checkpoint of these settings: ``kill_law``, ``zone_mark`` (the EPT compose heads declare their zone
    mark but not their law, so the law comes from the training identity) and the ``exposure`` of the evaluation ('nominal' on the primary population, D14). The gold
    representatives are the evaluation's (``evaluation_reps``) when binned, ``labels/gold_reps`` otherwise.

    해석: 합성기에 넘기는 킬 법칙과 구역 표지는 모형이 스스로 알려 주지 않으므로(법칙) 학습 정체성에서 읽은 값을 넘긴다. 구간 대표값은 학습 행이 쓴 31개와 같아야 한다."""
    return {'kill_law': settings['kill_law'], 'zone_mark': settings['zone_mark'], 'exposure': exposure}


def reps_record(winner_truth: str, checked: Mapping, inputs: EvalInputs):
    """The representatives of the gold bins that an evaluation hangs on, for the ledger: ``{'source', 'sha256', 'fallback'}``, or None when nothing hangs on them.

    The binned K condition uses the in-scope FIT means (``source`` 'in_scope_fit': the training's, with the bins that had no entry). The binned winner truth of a raw run
    reads ``labels/gold_reps`` (``source`` 'labels', no fallback list). With the raw K condition and the gold truth no representatives are read (None).

    해석: 승자 참값이 'binned'이면 참값이 구간 대표값에 달려 있으므로 실제로 쓴 배열의 해시를 남긴다. K 조건이 binned이면 학습과 같은 대표값(범위 안 FIT 평균)을 쓴다."""
    if checked['reps'] is not None:
        return {'source': 'in_scope_fit', 'sha256': checked['reps']['sha256'], 'fallback': list(checked['reps']['fallback'])}
    if winner_truth == 'binned':
        reps = np.asarray(inputs.base.labels.array('labels/gold_reps'), dtype=np.float64)
        return {'source': 'labels', 'sha256': value.gold_reps_sha256(reps), 'fallback': None}
    return None


# ---- the composed V_start pass on a sample of STOP matches (K6e; decisions D9, D14, D30; the hand-offs of M11, M12 and M14) ----------------------------------------------
#
#   (CV-1)  pool    P = { m in STOP : y_r finite for every row r of m }         sample  A = vstart.sample_matches(P, N, seed_N)         rows  R = { r : m(r) in A }
#   (CV-2)  nominal pass:   Q_nom(r) = Composer(chain, exposure='nominal') over every row r of R         (T-blind: no model or composer reads T)
#   (CV-3)  observed pass:  Q_obs(r) = Composer(chain', exposure='observed') over the rows with t0_r + 180 s > T - 30 s      (the primary sets are nested: these are the rows that some horizon leaves out)
#   (CV-4)  the rows are scored as vstart.summarize says: the nominal CRPS on P_H, the observed CRPS on the rest, the D14 share; the errors are vstart's (batch means, match bootstrap)
#
#      기호        | 뜻                                                                                          | 코드 변수
#      N, seed_N   | 표본의 경기 수와 시드(프로파일 composed의 sample_matches, sample_seed)                           | block
#      S, K, seed  | 행마다의 모의 횟수, 몬테카를로 배치 수, 합성기의 시드(draws, mc_batches, draw_seed)             | block
#      chain       | 체크포인트의 EPT 합성 머리, EPTComposerHeads(model, stats, chunk_views=프로파일의 chunk_views) | ``ept_chain``
#      R_gold      | 합성기에 넘기는 31개 구간 대표 골드(이진: 평가의 범위 안 FIT 평균, 원시: labels/gold_reps)         | ``composer_gold_reps``
#
# 해석: 합성 V_start는 STOP 전체가 아니라 값이 매겨진 경기의 표본에서만 한다(D30). 체크포인트마다 합성기를 표본 행 전체에 명목 노출로 한 번, 지평이 경기 끝 T − 30 s에 잘리는 행에 관측 노출로 한 번 돌린다. 모형 객체는
# 정체성을 지니지 않으므로 킬 법칙, 구역 표지, 프레임 평균, K 조건은 학습 정체성에서 읽은 값을 넘긴다. 구간 대표값이 학습의 것이 아니면(k_reps_sha256) 합성기를 만들기 전에 거부한다. 합성 사슬은 합성기마다
# 새로 만든다(한 사슬은 한 합성기의 행에만 묶인다). 묶음 크기 chunk_views는 프로파일에서 오며 추론의 배치 크기와는 다른 것이고, GPU 평가이면 장치와 결정성 설정을 기록에 적는다(M14 m7).
def execution_record(constants) -> dict:
    """The execution of an evaluation for the record: ``{'device'}`` on the CPU and, on a CUDA device, also the determinism settings that torch has in this process (``deterministic_algorithms``,
    ``cudnn_deterministic``, ``cudnn_benchmark``, ``float32_matmul_precision``) and the torch version. Nothing is changed: it only reads.

    해석: GPU 평가는 같은 입력에서도 마지막 비트가 장치와 결정성 설정에 따라 달라질 수 있으므로(M14 m7) 읽은 설정을 기록에 남긴다. CPU는 장치 이름만 적는다."""
    record = {'device': constants.device}
    if torch.device(constants.device).type == 'cuda':
        record.update(deterministic_algorithms=bool(torch.are_deterministic_algorithms_enabled()), cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
                      cudnn_benchmark=bool(torch.backends.cudnn.benchmark), float32_matmul_precision=torch.get_float32_matmul_precision(), torch=torch.__version__)
    return record


def composer_gold_reps(settings: Mapping, reps, labels) -> tuple:
    """``(array, source)`` of the 31 representatives of the gold bins that the Composer of this checkpoint gets. A binned checkpoint (``k_condition`` 'binned') gets the evaluation's representatives
    ``reps`` (``evaluation_reps``: the in-scope FIT means) and ``EvaluationRefused`` if there are none, if their sha256 is not the ``k_reps_sha256`` of the training or if the array does not have the sha256
    that it carries: heads trained on other representatives would be fed another K. A raw checkpoint gets ``labels/gold_reps`` as the label store holds it.

    해석: 이진 K 조건으로 학습한 머리는 학습 때 본 31개 값만 K로 읽었으므로 합성기에도 같은 값을 넘기고, 다르면 거부한다. 원시 조건은 라벨 저장소의 값을 그대로 쓴다."""
    if settings['k_condition'] == 'binned':
        if not isinstance(reps, Mapping) or settings['k_reps_sha256'] is None or reps.get('sha256') != settings['k_reps_sha256']:
            raise EvaluationRefused('the representatives of the gold bins of the evaluation are not the ones of the training (k_reps_sha256): the composer would feed the heads another K')
        array = np.asarray(reps['array'], dtype=np.float64)
        if value.gold_reps_sha256(array) != reps['sha256']:
            raise EvaluationRefused('the representatives of the gold bins carry a sha256 that they do not have: the composer would feed the heads another K')
        return array, 'in_scope_fit'
    return np.asarray(labels.array('labels/gold_reps'), dtype=np.float64), 'labels'


def ept_chain(model, trained: TrainedRun, profile):
    """A new composer chain of the EPT of ``trained``: ``compose_heads.EPTComposerHeads(model, stats, chunk_views=profile['chunk_views'])``; a checkpoint that is evaluated without the two-frame average
    (``eval_average`` False) is wrapped as ``FrameAveraged(model, average=False)``, as ``run_inference`` does. Every chain that a composer is bound to is made here (the nominal and the observed pass), so they
    all have the one ``chunk_views`` of the profile.

    해석: 합성기마다 새 사슬을 만든다(한 사슬은 한 합성기의 행에만 묶인다). 묶음 크기는 프로파일에서 오며 추론의 배치 크기와는 다른 것이다."""
    frame = model if trained.settings['eval_average'] else heads.FrameAveraged(model, average=False)
    return compose_heads.EPTComposerHeads(frame, trained.stats, chunk_views=profile['chunk_views'])


def subset_rows(rows: compose.RowTable, mask) -> compose.RowTable:
    """The ``compose.RowTable`` of the rows of ``rows`` where ``mask`` (bool, or the numbers of the rows) holds, in the same order and with the same roles."""
    return dataclasses.replace(rows, **{name: getattr(rows, name)[mask] for name in ('frame_row', 'match', 't0_ms', 'exposure_ms', 'T_ms', 'views', 'destroyed')})


def check_composed(profile) -> None:
    """``EvaluationRefused`` for a profile whose composed block cannot be run as asked: the pass runs on the role STOP only (DEV and TEST stay closed to it in S4b, decision D30) and under the nominal
    exposure (the T-blind forecast of D14, which is what A7 and the D14 check are about); and ``chunk_views``, whose only reader is this pass, set without the block (it would be recorded and used by nothing).

    해석: 합성 V_start 표본 계산은 STOP과 명목 노출에서만 한다. 읽는 곳이 이 계산뿐인 chunk_views는 합성 칸 없이 적으면 기록에는 남고 쓰이지 않으므로 거부한다."""
    block = profile['composed']
    if block is None:
        if profile['chunk_views'] is not None:
            raise EvaluationRefused('the profile sets chunk_views, which only the composed V_start pass reads (the block composed); it is refused rather than recorded and used by nothing')
        return
    if profile['role'] != 'STOP':
        raise EvaluationRefused(f"the composed V_start pass runs on the role STOP only, not on {profile['role']} (decision D30: DEV and TEST stay closed to it in S4b)")
    if profile['exposure'] != 'nominal':
        raise EvaluationRefused('the composed V_start pass needs the exposure "nominal": A7 and the D14 check are about the T-blind forecast of the primary population (decision D14)')


def stage_outputs(profile) -> tuple:
    """The outputs that the stage ``evaluate`` declares for ``profile``: the record and the report and, with a composed block, the rows file of the pass (``vstart.ROWS_FILE``)."""
    return (RECORD_FILE, REPORT_FILE) + ((vstart.ROWS_FILE,) if profile['composed'] is not None else ())


def composed_v_start(model, trained: TrainedRun, inputs: EvalInputs, *, role: str, profile, reps, constants, check, progress=None) -> dict:
    """The composed V_start of the checkpoint on a sample of STOP matches: the per-row losses of the nominal pass over every row of the sample and of the observed pass over the rows that the longest horizon
    leaves out, each from all S draws and from each of the K batches (``vstart.simulate_losses``), with the sample and the settings.

    Returns ``{'match', 'frame_row', 'primary' (n, 3), 'loss', 'loss_batches', 'loss_observed', 'loss_observed_batches' (the fields of ``vstart.ROW_FIELDS``), 'sample' (``vstart.sample_record``),
    'settings'}``. ``EvaluationRefused`` for a role other than STOP, representatives that are not the training's (``composer_gold_reps``, before any composer) and a sample larger than the pool of valued
    matches; ``Unavailable`` for inputs without prices, tier model or ``labels/v_start``, or a split without rows.

    해석: 체크포인트마다 표본 행 전체에 명목 합성기를 한 번, 지평이 경기 끝에 잘리는 행에 관측 합성기를 한 번 돌려 행별 손실을 낸다. 점수와 오차는 vstart.summarize가 만든다."""
    block = profile['composed']
    if block is None or role != 'STOP':
        raise EvaluationRefused(f'the composed V_start pass needs the composed block of the profile and the role STOP, not {role} (decision D30)')
    if inputs.prices is None:
        raise Unavailable('the inputs hold no prices, which the composed V_start needs')
    if inputs.tier is None:
        raise Unavailable('the inputs hold no tier model, which the composed V_start needs')
    base = inputs.base
    try:
        label = np.asarray(base.labels.array('labels/v_start'), dtype=np.float64)
    except (FileNotFoundError, KeyError):
        raise Unavailable('the label store holds no labels/v_start, the realised V_start that the composed forecasts are scored against') from None
    gold, source = composer_gold_reps(trained.settings, reps, base.labels)                            # refuses before a composer is made
    table = compose.row_table(base.s1, base.s2, ('STOP',))
    if not len(table):
        raise Unavailable('the role STOP holds no frame-clock row to compose')
    if label.ndim != 2 or label.shape[1] != len(vstart.HORIZONS_S) or len(label) <= int(table.frame_row.max()):
        raise EvaluationRefused('labels/v_start must hold one row of three horizons for every frame row of the S1 store')
    every = label[table.frame_row]
    pool = vstart.valued_matches(table.match, every)
    try:
        sample = vstart.sample_matches(pool, block['sample_matches'], block['sample_seed'])
    except vstart.VStartRefused as error:
        raise EvaluationRefused(str(error)) from None
    sampled = np.isin(table.match, sample)
    rows, realized = subset_rows(table, sampled), every[sampled]
    masks = adapters.primary_masks(base.s1, rows.frame_row)                                           # the one population filter of the t0 forecasts (D14)
    primary = np.column_stack([masks[h] for h in vstart.HORIZONS_S])

    def simulate(table_, y, exposure):
        """The losses of one pass; ``exposure`` ('nominal' or 'observed') also names its progress blocks (``bind/<exposure>``, then ``<exposure>/<first row>``)."""
        if progress is not None:
            progress('begin', f'bind/{exposure}', {})
        composer = compose.Composer(ept_chain(model, trained, profile), inputs.prices, inputs.tier, gold, table_, base.s2, S=block['draws'], seed=block['draw_seed'],
                                    **composer_settings(trained.settings, exposure))                  # the one expensive step: the heads are bound (the views are encoded) here
        if progress is not None:
            progress('end', f'bind/{exposure}', {'rows': len(table_), 'views': len(table_)})
        return vstart.simulate_losses(composer, y, batches=block['mc_batches'], block=block['progress_rows'], progress=progress, check=check, label=exposure)
    loss, loss_b = simulate(rows, realized, 'nominal')
    cut = np.flatnonzero(~primary[:, -1])                                                             # the primary sets are nested: left out at some horizon = left out at the longest
    loss_observed, loss_observed_b = np.full(loss.shape, np.nan), np.full(loss_b.shape, np.nan)
    if len(cut):
        cut_loss, cut_loss_b = simulate(subset_rows(rows, cut), realized[cut], 'observed')
        loss_observed[cut], loss_observed_b[:, cut] = cut_loss, cut_loss_b
    settings = {'kill_law': trained.settings['kill_law'], 'zone_mark': trained.settings['zone_mark'], 'eval_average': trained.settings['eval_average'], 'k_condition': trained.settings['k_condition'],
                'gold_reps': {'source': source, 'sha256': value.gold_reps_sha256(gold)}, **execution_record(constants)}
    if profile['chunk_views'] is not None:
        settings['chunk_views'] = int(profile['chunk_views'])                                         # recorded only when it is set (M14); never the batch size of the inference
    return {'match': np.asarray(rows.match), 'frame_row': np.asarray(rows.frame_row), 'primary': primary, 'loss': loss, 'loss_batches': loss_b, 'loss_observed': loss_observed,
            'loss_observed_batches': loss_observed_b, 'sample': vstart.sample_record(pool, sample, block['sample_matches'], block['sample_seed']), 'settings': settings}


# ---- the durable progress and timing record (K6d, the lead's addition after the T009 evaluation that left a 0-byte log) -------------------------------------
PROGRESS_FILE = 'evaluate/PROGRESS.jsonl'
PROGRESS_SCHEMA = 'ept-run-evaluate-progress-v1'


def _whole_counts(values: Mapping) -> dict:
    """A copy of ``values`` in which every entry is a whole number of at least 0 (``ValueError`` naming the key otherwise): a progress line holds counts of rows, draws and models, and never the
    number of a metric."""
    out = {}
    for key, val in values.items():
        if isinstance(val, bool) or not isinstance(val, (int, np.integer)) or val < 0:
            raise ValueError(f'a progress count must be a whole number of at least 0: {key!r} is not')
        out[str(key)] = int(val)
    return out


def _plain_detail(values: Mapping) -> dict:
    """A copy of ``values`` in which every entry is a text or a whole number (``ValueError`` naming the key otherwise): the role, the pass, the exposure and the population of a stage."""
    out = {}
    for key, val in values.items():
        if isinstance(val, bool) or not isinstance(val, (str, int, np.integer)):
            raise ValueError(f'a progress detail must be a text or a whole number: {key!r} is not')
        out[str(key)] = val if isinstance(val, str) else int(val)
    return out


class StageLog:
    """The progress and timing record of an evaluation: one strict-JSON line (``ept-run-evaluate-progress-v1``) at the begin and one at the end of every stage, appended and flushed (and synced) at
    once, so that a run that is killed, hangs or is stopped by the wall cap still leaves how far it came and how long each finished stage took. The T009 evaluation of the old stack ended at its
    phase cap with a 0-byte log, so nobody could tell what used the 8 hours.

    Every line carries ``attempt``: the log made for the next attempt of a run folder is numbered one more than the number of ``evaluate`` begin lines that the file holds (a resumed or repeated run appends
    to one file; its ``evaluate`` begin line names the profile and the role).

    ``with log.stage(name, model, detail=..., soft=(...), **counts) as counts:`` writes the ``begin`` line (``stage``, ``model`` key or None, ``detail``, wall ``start``, ``pid``), runs the body, and
    writes the ``end`` line (the same fields, the wall ``end``, ``elapsed_s`` from the monotonic clock, ``status``, ``error`` and ``counts``). ``counts`` (whole numbers only) may be filled in by the
    body. ``status`` is ``ok``, ``unavailable`` (an exception of a class in ``soft``: a part that cannot be made, which the caller handles) or ``failed`` (any other exception, also an interrupt);
    ``error`` is the class name of the exception, never its message (a message can name a path or a key). The exception goes on. Nothing of a metric is written: counts and details are whole
    numbers and texts and nothing else.

    해석: 단계가 시작할 때와 끝날 때 한 줄씩 곧바로 파일에 쓰고 비운다. 도중에 죽거나 멈춰도 어느 단계까지 왔는지와 끝난 단계가 얼마나 걸렸는지가 남는다. 줄에는 행·추첨·모형의 개수만 있고 지표의 값은 없다.
    """

    def __init__(self, path, *, wall=time.time, monotonic=time.monotonic):
        self.path, self._wall, self._monotonic = Path(path), wall, monotonic
        self.attempt = 1 + self._attempts_before()

    def _attempts_before(self) -> int:
        """The number of attempts that the file already holds: its ``begin`` lines of the stage ``evaluate``. A log made for the next attempt of a run folder numbers its lines ``attempt`` = this + 1
        (review M8: one file holds the attempts of a folder, and the counter tells the lines of one attempt from another's). Text that is not a line of this log is not counted."""
        try:
            lines = self.path.read_text(encoding='utf-8').splitlines()
        except OSError:
            return 0
        count = 0
        for line in lines:
            try:
                item = json.loads(line)
            except ValueError:
                continue
            if isinstance(item, dict) and item.get('schema') == PROGRESS_SCHEMA and item.get('stage') == 'evaluate' and item.get('event') == 'begin':
                count += 1
        return count

    def _append(self, line: Mapping) -> None:
        text = json.dumps(line, sort_keys=True, allow_nan=False, separators=(',', ':'))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, 'a', encoding='utf-8', newline='\n') as handle:
            handle.write(text + '\n')
            handle.flush()
            os.fsync(handle.fileno())

    @contextlib.contextmanager
    def stage(self, name: str, model=None, *, detail=None, soft: tuple = (), **counts):
        counts = _whole_counts(counts)
        base = {'schema': PROGRESS_SCHEMA, 'stage': str(name), 'model': None if model is None else str(model), 'detail': _plain_detail(detail or {}), 'pid': os.getpid(),
                'attempt': self.attempt}
        start, began = float(self._wall()), self._monotonic()
        self._append({**base, 'event': 'begin', 'start': start})
        status, error, raised = 'ok', None, None
        try:
            yield counts
        except soft as caught:
            status, error, raised = 'unavailable', type(caught).__name__, caught
        except BaseException as caught:
            status, error, raised = 'failed', type(caught).__name__, caught
        try:
            final = _whole_counts(counts)
        except ValueError as bad:                                                                  # a count that is not a whole number: the stage failed, and its line says so
            final = {}
            if raised is None:
                status, error, raised = 'failed', type(bad).__name__, bad
        end = float(self._wall())
        self._append({**base, 'event': 'end', 'start': start, 'end': end, 'elapsed_s': float(self._monotonic() - began), 'status': status, 'error': error, 'counts': final})
        if raised is not None:
            raise raised


class BlockStages:
    """The ``progress`` callback of ``adapters.run_inference`` that writes one ``block`` stage to a ``StageLog`` for each head block (begin line at ``'begin'``, end line with the counts of the block at
    ``'end'``). Used as a context manager around the call: if the call raises while a block is open, the line of that block is closed as failed (or unavailable) with the exception, so that the log shows
    which block the call died in. Nothing but whole-number counts and the block name is written.

    해석: 추론 호출이 머리 블록마다 알려 주는 시작과 끝을 진행 기록의 한 단계로 바꾼다. 호출이 블록 도중에 죽으면 그 블록의 줄이 실패로 닫혀 어느 블록에서 죽었는지 남는다."""

    def __init__(self, log: StageLog, model, detail: Mapping):
        self.log, self.model, self.detail, self.current = log, model, dict(detail), None

    def __call__(self, event: str, block: str, counts: Mapping) -> None:
        if event == 'begin':
            context = self.log.stage('block', self.model, detail={**self.detail, 'block': block})
            self.current = (context, context.__enter__())
        elif event == 'end' and self.current is not None:
            context, held = self.current
            self.current = None
            held.update(counts)
            context.__exit__(None, None, None)

    def __enter__(self):
        return self

    def __exit__(self, kind, error, trace):
        if self.current is not None:
            context, _ = self.current
            self.current = None
            if error is None:
                context.__exit__(None, None, None)
            else:
                try:
                    context.__exit__(kind, error, trace)
                except BaseException:                                                              # the line is written first; the caller's exception is the one that goes on
                    pass
        return False


# ---- the evaluation ---------------------------------------------------------------------------------------------------------------------------------
def _plain(obj):
    """A strict-JSON copy of ``obj``: numpy scalars and arrays become numbers and lists, mappings and sequences are copied."""
    if isinstance(obj, Mapping):
        return {str(key): _plain(val) for key, val in obj.items()}
    if isinstance(obj, np.ndarray):
        return _plain(obj.tolist())
    if isinstance(obj, (list, tuple)):
        return [_plain(val) for val in obj]
    if isinstance(obj, np.generic):
        return _plain(obj.item())
    return obj


def _sha_of_array(array) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _hash_files(files: Mapping) -> dict:
    """``{name: sha256}`` of files that are hashed directly from the disk (never from a copy held in memory; M11 m6); ``EvaluationRefused`` for one that cannot be read."""
    out = {}
    for name in sorted(files):
        try:
            out[str(name)] = run_io.sha_file(files[name])
        except OSError:
            raise EvaluationRefused(f'the file {name!r} of the inputs cannot be read') from None
    return out


class _Bootstrap:
    """The common resample matrix of the judging split, made once and only if a bootstrap is needed (D15: one matrix, B = 10,000; D13's B = 2,000 are its first rows)."""

    def __init__(self, backend, inputs, role, bootstrap_b):
        self.backend, self.inputs, self.role, self.b = backend, inputs, role, bootstrap_b
        self.ids = self.full = None

    def get(self):
        if self.full is None:
            ids = np.asarray(self.backend.split_ids(self.inputs, self.role))
            if ids.ndim != 1 or not len(ids) or (len(ids) > 1 and not (ids[1:] > ids[:-1]).all()):
                raise EvaluationRefused('the match list of the judging split must be a non-empty, sorted, unique vector')
            self.ids, self.full = ids, inference.resample_matrix(len(ids), B=max(inference.B_DEFAULT, self.b))
        return self.ids, self.full[:self.b]

    def record(self):
        if self.full is None:
            return None
        used = self.full[:self.b]
        return {'numpy': np.__version__, 'seed': inference.SEED, 'matrix_rows': int(self.full.shape[0]), 'matrix_sha256': _sha_of_array(self.full), 'B': int(self.b),
                'used_sha256': _sha_of_array(used), 'split': {'n_matches': int(len(self.ids)), 'ids_sha256': _sha_of_array(self.ids.astype('<i8'))}}


# ---- what the runner refuses before it starts (review I2, I6, m5) ------------------------------------------------------------------------------------
# Profile fields that the K3 profile accepts and that no code of this runner reads yet, with the default that is not a request and the task that wires them. A value other than
# the default is refused: a number computed without what the profile asked for, under a record that shows the flag set, would read as if it had been applied.
UNWIRED_FIELDS = (('strata', False, 'M17 (the M13 inputs, a task before S8)'), ('price_sensitivity', False, 'M17 (the M13 inputs, a task before S8)'),
                  ('b1_swap_average', False, 'M17 (the M13 inputs, a task before S8)'))               # chunk_views was wired by K6e: it is read by the composed pass (``check_composed`` refuses it without the pass)


def check_wired_fields(profile) -> None:
    """``EvaluationRefused`` for a field of ``UNWIRED_FIELDS`` that is not at its default, naming the task that wires it.

    해석: 프로파일이 받아들이지만 이 실행기가 아직 읽지 않는 칸이 켜져 있으면 그 칸 없이 계산한 숫자가 켜진 것처럼 기록된다. 조용히 무시하지 않고 거부한다."""
    for name, default, task in UNWIRED_FIELDS:
        if profile[name] != default:
            raise EvaluationRefused(f'the profile asks for {name}, which nothing in this runner reads yet (it is wired by {task}); it is refused rather than evaluated without it')


def check_bootstrap_rows(profile) -> None:
    """``EvaluationRefused`` for a ``bootstrap_B`` above the ``inference.B_DEFAULT`` = 10,000 rows of the common matrix: the prefix of a matrix drawn with more rows is not shown to
    be the B = 10,000 matrix of decision D15.

    해석: C4의 재표본은 공통 B = 10,000 행렬의 앞 B행이다. 10,000을 넘기면 같은 행렬의 앞부분이라는 보장이 없으므로 거부한다."""
    if profile['bootstrap_B'] > inference.B_DEFAULT:
        raise EvaluationRefused(f"bootstrap_B is {profile['bootstrap_B']}, more than the {inference.B_DEFAULT:,} rows of the common resample matrix (decision D15)")


BODY_FIELDS = frozenset({'module', 'name', 'file_sha256'})
_BODY_NAME = re.compile(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*')            # the form of ``supervisor.Stage.body``: module.path:qualified.name


def body_record(spec) -> dict:
    """The stage body ``spec`` (``module.path:qualified.name``: the text that ``supervisor.Stage`` carries and that the child imports) as the record names it, ``{'module', 'name',
    'file_sha256'}``: the SHA-256 of the source file of the module, found without importing the module (its parent packages are imported). ``EvaluationRefused`` for a text that is not of
    that form, a module that cannot be found or has no readable source file.

    K6b review r4 (K6c): the code identity names the repository modules that the supervisor process loaded; a stage body that only the child imports (a body given to ``start`` by
    ``bodies=``) was not in it, so a DEV rehearsal and a TEST run could differ in the body and still pass the gate. The record is made by the parent from the text that it hands the supervisor,
    written to the record, the DEV access line and the ledger header, checked again by the child (``verify_body``) and compared by the TEST gate.

    해석: 단계 본문은 자식 프로세스가 이름으로 불러 쓰는 함수이므로 코드 정체성에 들지 않을 수 있다. 그래서 본문을 모듈 이름, 함수 이름, 그 모듈 파일의 해시로 적어 기록과 열람 원장 줄과
    원장 머리글에 남기고, TEST 관문은 리허설과 같은 본문일 때만 연다."""
    if not isinstance(spec, str) or _BODY_NAME.fullmatch(spec) is None:
        raise EvaluationRefused('the stage body must be named module.path:function')
    module, _, name = spec.partition(':')
    try:
        found = importlib.util.find_spec(module)
    except (ImportError, ValueError, AttributeError):
        found = None
    origin = getattr(found, 'origin', None)
    if not isinstance(origin, str) or not Path(origin).is_file():
        raise EvaluationRefused('the module of the stage body cannot be found or has no source file')
    try:
        digest = run_io.sha_file(origin)
    except OSError:
        raise EvaluationRefused('the source file of the stage body cannot be read') from None
    return {'module': module, 'name': name, 'file_sha256': digest}


def verify_body(record) -> None:
    """``EvaluationRefused`` unless ``record`` is a record of a stage body (``BODY_FIELDS``, all texts) and the module file that it names has the hash that it names now: what the supervisor
    hashed when it started the run is the code that the child runs.

    해석: 부모가 본문 파일의 해시를 적은 뒤 자식이 시작하기 전에 그 파일이 바뀌었다면 거부한다."""
    if not (isinstance(record, Mapping) and set(record) == BODY_FIELDS and all(isinstance(record[key], str) for key in BODY_FIELDS)):
        raise EvaluationRefused('the stage body record of the ledger header is not a record of a stage body (module, name and file_sha256)')
    if body_record(f"{record['module']}:{record['name']}") != dict(record):
        raise EvaluationRefused('the file of the stage body changed (or is another file) since the supervisor hashed it: the stage body is the code that the run was started with')


def sources_of(loader, backend, manifest_sha256, allow_custom, body=None) -> dict:
    """The sources of a run as the record and the gate see them: the inputs loader and the backend (``module:name`` texts or None), the sha256 of the inputs manifest, the
    test-only switch and, when it is known, the stage body (``body_record``; the key is absent for a call of ``evaluate`` that was given none, and the role TEST refuses that)."""
    sources = {'loader': loader, 'backend': backend, 'manifest_sha256': manifest_sha256, 'allow_custom': bool(allow_custom)}
    if body is not None:
        sources['body'] = dict(body)
    return sources


def check_sources(role: str, sources: Mapping) -> None:
    """``EvaluationRefused`` for the roles DEV and TEST unless the loader and the backend are the defaults (``LOADER``, ``BACKEND``). DEV may set the test-only switch ``allow_custom``
    (a fixture of the tests; it is recorded); TEST never may, so a TEST run is always made with the default loader and backend (review r1).

    The code identity names the repository modules that the supervisor process loaded; a loader or backend that only the child imports is not part of it. A DEV rehearsal of one
    backend and a TEST run of another would therefore pass for the same code. The default loader and backend are modules of this repository, so their identity is the code identity.

    해석: DEV 리허설과 TEST가 같은 코드라는 것은 코드 정체성이 보장하는데, 그 정체성은 자식 프로세스만 읽는 로더·백엔드를 담지 못한다. 그래서 두 role은 기본 로더·백엔드만 받는다. DEV만 시험 장치
    (allow_custom)로 다른 것을 쓸 수 있고 그 사실이 기록된다. TEST는 장치를 받지 않는다."""
    if role == 'TEST' and sources['allow_custom']:
        raise EvaluationRefused('the role TEST never takes the test-only switch allow_custom: it is made with the default inputs loader and backend')
    if role in ('DEV', 'TEST') and not sources['allow_custom'] and (sources['loader'] != LOADER or sources['backend'] != BACKEND):
        raise EvaluationRefused(f'the role {role} takes the default inputs loader {LOADER} and the default backend {BACKEND}; another loader or backend is a test fixture and needs the '
                                'test-only switch allow_custom (DEV only; recorded in the record)')


def _backend_name(backend) -> str:
    return f'{type(backend).__module__}:{type(backend).__qualname__}'


def passes_of(exposure: str) -> tuple:
    """The passes of the inference as ``(name, exposure, population)``: under the nominal exposure the primary pass (nominal exposure, the rows whose horizon the game end does not
    cut) and the secondary pass (the observed exposure, the remaining rows; decision D14), otherwise the one legacy pass (observed exposure, every row). C4 and the structural metric
    read the first pass.

    해석: 명목 노출의 주 결과는 T를 모르는 예측(주 인구)이고, D14는 나머지 행의 관측 노출 결과를 함께 보고하라고 한다. 프로파일에 칸을 더하지 않고 명목 노출이면 항상 두 번 추론한다."""
    if exposure == 'nominal':
        return (('primary', 'nominal', 'primary'), ('secondary', 'observed', 'secondary'))
    return (('all', 'observed', 'all'),)


def evaluate(profile, run_dir, *, inputs: EvalInputs, constants: runners.RunnerConstants = runners.RunnerConstants(), backend=None, access=None, gate=None, identity=None,
             check=None, clock=time.time, loader=None, backend_name=None, manifest_sha256=None, allow_custom=False, body=None, progress=None) -> dict:
    """Evaluate the models of the evaluate profile and write ``evaluate/RECORD.json`` (what was evaluated, under which settings, with which hashes) and
    ``evaluate/REPORT.json`` (the numbers, and what is unavailable with the reason) under ``run_dir``; return ``{'record', 'report'}``.

    Roles. ``STOP`` (decision D13): the whole STOP split with each fit's best checkpoint; per model the blocks of ``train.evaluate_stop_blocks``, the two shared M2 blocks (D9), the
    endpoints and NLL records of the DEV adapter on the STOP rows (C4, the relative structural metric); the composed V_start is made with the profile block ``composed`` only (K6e: a sample of STOP matches, the role STOP and the nominal exposure), else it is reported as unavailable. ``DEV`` (exploratory, D4, D20): needs ``access`` (the
    ``AccessLedger``) and the code ``identity`` of the run; the access is logged before any DEV number is made. ``TEST`` (decision D20): needs ``gate`` (``HoldoutGate``) and ``identity``
    and a backend that can evaluate TEST (``Backend.evaluates``); the gate opens TEST once, after the models passed their checks and before any TEST row is read, and the
    ``HoldoutOpening`` that it returns goes to the backend with every inference of the run. Whatever a role cannot make here is reported as unavailable with its reason.

    ``loader``, ``backend_name`` (default: the class of ``backend``), ``manifest_sha256``, ``allow_custom`` and ``body`` (the record of the stage body, ``body_record``; None for a call
    that is not made by a stage body) are what the run says about its sources (``sources_of``): the roles DEV and TEST accept the default loader and backend only unless ``allow_custom``
    (a test fixture switch, recorded) is set, and the gate compares them with the rehearsal's; the role TEST needs a ``body``.

    The numeric policy (``numeric.apply``, decision D10 P6, K6c) is set and read back after the early refusals and before the first stage, the first model or the first line of the
    progress file; a state that is not the policy is ``NumericRefused`` (exit code 2) and the record gets the state as ``numeric``. It changes process-wide settings, as ``train`` does.

    For every model the identity of the checkpoint is read from the training run (``load_models``), a mixture of settings and everything that does not fit is refused
    (``check_models``) and each model's own kill law, K condition, frame average and zone mark are passed on (the model object carries no identity). The seed values
    pair the arms: with the three seeds 20261022, +1, +2 and a matched B1 the C4 row is made on the first ``bootstrap_B`` rows of the common matrix over the split's match list.
    The inference makes the passes of ``passes_of(exposure)``.

    ``progress`` (a ``StageLog``, default one at ``run_dir / PROGRESS_FILE``) receives a flushed ``begin`` and ``end`` line for the whole run and for every stage after the early refusals; a refusal before
    the first model is read writes nothing. The log holds counts and timing only (``StageLog``).

    해석: 평가는 체크포인트의 학습 정체성을 읽어 섞임을 거부하고, 모형마다 자기 설정(킬 법칙, K 조건, 프레임 평균, 구역 표지)을 넘긴다. DEV는 숫자를 내기 전에 열람 원장에 한 줄을 남기고,
    TEST는 관문(잠금 해시, 같은 프로파일·코드·로더·백엔드·입력·학습 기록의 DEV 리허설, 열린 적 없음)을 통과한 뒤 한 번만 연다. 못 만드는 것은 이유와 함께 '없음'으로 보고한다."""
    if getattr(profile, 'kind', None) != 'evaluate':
        raise EvaluationRefused('the evaluation takes an evaluate profile')
    run_dir, role = Path(run_dir), profile['role']
    backend = backend if backend is not None else Backend()
    # 1. what the role needs, before anything is read
    if role == 'DEV' and not isinstance(access, AccessLedger):
        raise AccessRefused('a DEV evaluation needs the access ledger: every access to DEV results is logged (decision D4)')
    if role == 'TEST' and not isinstance(gate, HoldoutGate):
        raise AccessRefused('the role TEST needs the gate: the S5 lock, a DEV rehearsal and a ledger in which TEST was never opened (decision D20)')
    if role in ('DEV', 'TEST') and not isinstance(identity, run_identity.Identity):
        raise AccessRefused(f'the role {role} needs the code identity of the run (an Identity)')
    check_wired_fields(profile)
    check_composed(profile)
    check_bootstrap_rows(profile)
    if not backend.evaluates(role):                                                                   # before the gate opens anything: TEST is opened once (review I1)
        raise EvaluationRefused(f'the backend cannot evaluate the role {role} here (the default backend evaluates DEV and STOP; TEST waits for a TEST reader of the stores, S8); nothing was opened or logged')
    sources = sources_of(loader, backend_name if backend_name is not None else _backend_name(backend), manifest_sha256, allow_custom, body)
    check_sources(role, sources)
    if profile['b0'] and inputs.b0 is None:
        raise EvaluationRefused('the profile asks for the relative structural metric (b0) but the inputs hold no B0 heads')
    numeric = run_numeric.apply(constants.device)                    # D10 P6 (K6c): the float32 policy is held before any stage, any model and any line of the progress file
    log = progress if progress is not None else StageLog(run_dir / PROGRESS_FILE)                     # from here on every stage leaves a flushed line (a refusal above wrote nothing)
    with log.stage('evaluate', detail={'role': role, 'profile_sha256': profile.sha256}, models=len(profile['models'])):
        return _evaluate_stages(profile, run_dir, role, inputs=inputs, constants=constants, backend=backend, access=access, gate=gate, identity=identity, check=check,
                                clock=clock, sources=sources, log=log, numeric=numeric)


def _evaluate_stages(profile, run_dir: Path, role: str, *, inputs: EvalInputs, constants, backend, access, gate, identity, check, clock, sources: Mapping, log: StageLog,
                     numeric: Mapping) -> dict:
    """Steps 2 to 5 of ``evaluate``: the models and their checks, the access, the evaluation of every model, C4 and the records, each as a stage of ``log`` (``StageLog``): ``identity``, ``access``
    (DEV and TEST), then per model ``build_model``, ``stop_report`` (STOP), ``inference`` (one per pass), ``structural`` (with B0) and ``v_start``, then ``c4`` and ``records``."""
    # 2. the models and what must fit together (no DEV or TEST row is read here)
    with log.stage('identity', models=len(profile['models'])):
        trained = load_models(profile)
        model_hashes = {run.name: run.record_sha256 for run in trained}
        if role == 'TEST':
            gate.check_evidence(profile, sources=sources, models=model_hashes)                            # an early refusal; nothing is written
        checked = check_models(trained, inputs)
    # 3. the access: logged (DEV) or opened (TEST) before the first number is made
    opened = None
    if role in ('DEV', 'TEST'):
        with log.stage('access', detail={'role': role}):
            if role == 'DEV':
                opened = access.append('dev_access', {
                    'role': 'DEV', 'profile_sha256': profile.sha256, 'profile_without_role_sha256': profile_without_role_sha256(profile),
                    'identity_sha256': run_io.canonical_sha256(identity.to_record()), 'identity_commit': identity.commit, 'opened': False, 'sources': sources,
                    'models': {run.name: {'seed': run.seed, 'train_record_sha256': run.record_sha256} for run in trained},
                    'note': 'computed by this run; whoever reads its numbers adds a row to harness/DEV_ACCESS_LEDGER.md (decision D4)'}, clock=clock)
            else:
                opened = gate.open(profile, identity, sources=sources, models=model_hashes, clock=clock)
    # 4. the evaluation
    bootstrap = _Bootstrap(backend, inputs, role, profile['bootstrap_B'])
    passes = passes_of(profile['exposure'])
    unavailable, models, endpoints_by_seed, drawn = [], {}, {}, []                                       # drawn: (monte_carlo, draws) as each inference call says it in its provenance
    vrows = {}                                                                                           # the composed pass (K6e): model name -> the parts that its rows file holds
    for run in trained:
        with log.stage('build_model', run.name):
            model = backend.build_model(run, inputs.base.static, constants.device)
        part = {'seed': run.seed, 'stop': None, 'structural': None, 'v_start': None, 'inference_provenance': None, 'passes': {}}
        if role == 'STOP':
            with log.stage('stop_report', run.name) as counts:
                stop = backend.stop_report(model, run, inputs, constants, check)
                counts['blocks'] = len(stop['blocks'])
            try:
                stop['shared_m2'] = ept_train.shared_m2_blocks({'blocks': stop['blocks']})                   # D9: the two blocks that the zone term does not enter, separately
            except ValueError as error:
                stop['shared_m2'] = None
                unavailable.append({'item': 'shared_m2', 'model': run.name, 'reason': str(error)})
            stop['masked_m3'] = {key: entry for key, entry in stop['blocks'].items() if key.startswith('M3') and key.endswith('/masked')}        # D9: the masked M3 NLL
            if not stop['masked_m3']:
                unavailable.append({'item': 'masked_m3', 'model': run.name, 'reason': 'the STOP rows held no scored entry for a masked M3 block'})
            part['stop'] = stop
        results = {}
        for name, exposure, population in passes:
            try:
                with log.stage('inference', run.name, detail={'role': role, 'pass': name, 'exposure': exposure, 'population': population}, soft=(Unavailable,)) as counts:
                    with BlockStages(log, run.name, {'role': role, 'pass': name}) as blocks:                  # one line pair for each head block of the call (review M3)
                        results[name] = backend.inference(model, run, inputs, role=role, profile=profile, reps=checked['reps'], constants=constants, check=check,
                                                          exposure=exposure, population=population, progress=blocks, **({'opening': opened} if role == 'TEST' else {}))
                    said = results[name].get('provenance') or {}                                              # whether and how much the call drew is what it says, not what the role would suggest (M2)
                    monte_carlo = bool(said.get('monte_carlo', True))
                    draws = 0 if not monte_carlo else said.get('draws')
                    drawn.append((monte_carlo, draws))
                    counts.update(endpoint_rows=sum(len(ep['row']) for ep in results[name]['endpoints']), block_records=len(results[name]['nll']), **({} if draws is None else {'draws': draws}))
            except Unavailable as error:
                unavailable.append({'item': 'inference', 'model': run.name, 'pass': name, 'reason': error.reason})
        for name, exposure, population in passes:
            if name in results:
                part['passes'][name] = _pass_part(results[name], exposure, population)
        result = results.get(passes[0][0])                                                                # C4 and the structural metric read the first pass
        if result is not None:
            part['inference_provenance'] = _plain(result.get('provenance'))
            endpoints_by_seed[run.seed] = next((ep for ep in result['endpoints'] if ep['name'] == C4_ENDPOINT), None)
            if profile['b0']:
                try:
                    with log.stage('structural', run.name, soft=(Unavailable, ValueError)) as counts:
                        ids, matrix = bootstrap.get()
                        part['structural'] = _plain(metrics.relative_structural(result['nll'], matrix, ids=ids))
                        counts.update(matches=len(ids), bootstrap_rows=len(matrix))
                except (Unavailable, ValueError) as error:
                    unavailable.append({'item': 'structural', 'model': run.name, 'reason': getattr(error, 'reason', str(error))})
        elif profile['b0']:
            unavailable.append({'item': 'structural', 'model': run.name, 'reason': 'the relative structural metric needs the NLL records of the inference, which is unavailable'})
        try:
            with log.stage('v_start', run.name, detail={'role': role}, soft=(Unavailable,)):
                if profile['composed'] is None:                                                           # every earlier profile: the call, the part and the stage lines are unchanged
                    vs = backend.v_start(model, run, inputs, role=role, profile=profile, reps=checked['reps'], constants=constants, check=check)
                    part['v_start'] = _v_start_part(vs)
                else:
                    with BlockStages(log, run.name, {'role': role, 'pass': 'v_start'}) as blocks:           # a begin and an end line for the bind and for every block of rows
                        vs = backend.v_start(model, run, inputs, role=role, profile=profile, reps=checked['reps'], constants=constants, check=check, progress=blocks)
                    ids, matrix = bootstrap.get()
                    part['v_start'] = vstart.summarize(vs, ids, matrix)
                    if result is None:
                        part['v_start']['occurrence'] = None
                        unavailable.append({'item': 'occurrence', 'model': run.name, 'reason': 'the calibration of the occurrence probability needs the primary pass of the inference, which is unavailable'})
                    else:
                        part['v_start']['occurrence'] = vstart.occurrence_calibration(result['endpoints'], np.unique(vs['match']), ids, matrix)
                    vrows[run.name] = {'seed': run.seed, 'parts': vs}
        except Unavailable as error:
            unavailable.append({'item': 'v_start', 'model': run.name, 'reason': error.reason})
        models[run.name] = part
    b1_record = None                                                                                    # no matched B1, no record of it (review r2)
    if inputs.b1 is not None:
        b1_hashes = _hash_files(inputs.b1_files)
        b1_record = {'files_sha256': b1_hashes, 'folder_sha256': run_io.canonical_sha256(b1_hashes)}
    with log.stage('c4', bootstrap_rows=int(profile['bootstrap_B'])) as counts:
        c4 = _c4_part(trained, inputs, endpoints_by_seed, bootstrap, unavailable, checked, b1_record)
        counts['matches'] = 0 if bootstrap.ids is None else int(len(bootstrap.ids))
    rows = [row for row in (c4,) if row is not None]
    # 5. the records
    with log.stage('records'):
        return _write_records(profile, run_dir, role, constants=constants, sources=sources, checked=checked, inputs=inputs, trained=trained, models=models, passes=passes, bootstrap=bootstrap,
                              opened=opened, unavailable=unavailable, c4=c4, rows=rows, b1_record=b1_record, drawn=drawn, numeric=numeric, vrows=vrows)


def write_composed(profile, run_dir: Path, vrows: Mapping) -> tuple:
    """Write the rows file of the composed pass (``vstart.ROWS_FILE``: the per-row losses of every model, with the sample, the draws and the training seeds as its meta) and return ``(report entry, record
    entry)``: the file named by its content hash, and the sample, the parameters and the settings of every model. ``vrows`` maps the model name to ``{'seed', 'parts'}`` (``composed_v_start``); it may be
    empty (every model unavailable): the file is then written without rows, because the stage declares it. ``EvaluationRefused`` for models that were sampled differently.

    해석: A7의 대조는 두 실행의 같은 행을 짝지어야 하므로 행별 손실을 파일로 남기고 보고서에는 그 내용 해시만 적는다. 경기 번호는 파일에만 있고 기록에는 해시만 있다."""
    block = profile['composed']
    samples = {item['parts']['sample']['ids_sha256'] for item in vrows.values()}
    if len(samples) > 1:
        raise EvaluationRefused('the models of one evaluation were sampled differently: the composed pass takes one sample for all of them')
    sample = next(iter(vrows.values()))['parts']['sample'] if vrows else None
    settings = {name: dict(item['parts']['settings']) for name, item in vrows.items()}
    meta = {'sample': sample, 'draws': int(block['draws']), 'mc_batches': int(block['mc_batches']), 'draw_seed': int(block['draw_seed']),
            'models': {name: {'seed': int(item['seed']), **{key: settings[name][key] for key in ('kill_law', 'zone_mark', 'k_condition', 'eval_average')}} for name, item in vrows.items()}}
    digest = vstart.write_rows(run_dir / vstart.ROWS_FILE, {name: {field: item['parts'][field] for field in vstart.ROW_FIELDS} for name, item in vrows.items()}, meta)
    s, k = int(block['draws']), int(block['mc_batches'])
    record = {'sample': sample, 'draws': s, 'mc_batches': k, 'draw_seed': int(block['draw_seed']), 'progress_rows': int(block['progress_rows']),
              'exposure': {'primary': 'nominal', 'secondary': 'observed'}, 'mc_error': f'batch means: sd of the statistic over the {k} batches of {s // k} draws / sqrt({k})',
              'bootstrap_B': int(profile['bootstrap_B']), 'models': settings}
    return {'file': vstart.ROWS_FILE, 'sha256': digest, 'models': sorted(vrows)}, record


def _write_records(profile, run_dir: Path, role: str, *, constants, sources, checked, inputs, trained, models, passes, bootstrap, opened, unavailable, c4, rows, b1_record, drawn,
                   numeric: Mapping, vrows: Mapping | None = None) -> dict:
    """Step 5 of ``evaluate``: ``evaluate/RECORD.json`` and ``evaluate/REPORT.json``; returns ``{'record', 'report'}``. ``inference_constants`` holds the batch size and, as the inference calls said it in
    their provenance (``drawn``), ``monte_carlo`` (True, False, or None when no call was made or the calls differ) and ``draws`` (the draws per unit that were made, 0 when none; None when no call was made
    or none said); ``seed`` is the seed of the draws (``MC_SEED``) when at least one inference call had the Monte Carlo on (``monte_carlo=True``), and ``None`` when no call was made or none had it on. It
    says which seed the calls were given, not that a draw happened: a call with the Monte Carlo on and no unit in its scope still carries it (K6d re-check R2, minor m10 of the Gate 0 review).

    해석: seed는 몬테카를로를 켠 호출이 하나라도 있을 때만 적는 추첨 씨앗(MC_SEED)이다. 호출이 없었거나 몬테카를로를 켠 호출이 없으면 None이다. 켠 호출에 실제로 뽑힌 단위가 있었다는 뜻은 아니다."""
    profile_hash = profile_without_role_sha256(profile)
    modes, counts_seen = {monte for monte, _ in drawn}, [draws for _, draws in drawn if draws is not None]
    record = {'schema': RECORD_SCHEMA, 'role': role, 'profile_sha256': profile.sha256, 'profile_without_role_sha256': profile_hash, 'profile': profile_record(profile),
              'sources': sources, 'runner': constants.to_record(), 'numeric': numeric,
              'inference_constants': {'draws': counts_seen[0] if counts_seen and len(set(counts_seen)) == 1 else None, 'seed': MC_SEED if True in modes else None,
                                      'batch_size': constants.batch_size, 'monte_carlo': next(iter(modes)) if len(modes) == 1 else None},
              'winner_truth': profile['winner_truth'],
              'exposure': {'exposure': profile['exposure'], 'population': 'primary' if profile['exposure'] == 'nominal' else 'all',
                           'passes': [{'name': name, 'exposure': exposure, 'population': population} for name, exposure, population in passes]},
              'models': {run.name: {'seed': run.seed, 'settings': dict(run.settings), 'train_record_sha256': run.record_sha256, 'best': dict(run.head['best']),
                                    'decay_groups': _plain(run.record['decay_groups']), 'stats_sha256': run.stats.sha256,
                                    'termination': _plain(run.record['termination'])} for run in trained},          # a training that ended in budget is evaluated at its best checkpoint
              'agreed': {'settings': checked['settings'], 'encoder': checked['encoder'], 'execution': checked['execution'], 'fit_sha256': checked['fit_sha256'],
                         'stop_sha256': checked['stop_sha256'], 's2_store_json_sha256': checked['s2_store_json_sha256']},
              'selection_data': selection_record(role, checked, inputs),
              'reps': reps_record(profile['winner_truth'], checked, inputs), 'scales': checked['scales'], 'dropped': checked['dropped'], 'b1': b1_record,
              'b0': None if not profile['b0'] else {'files_sha256': _hash_files(inputs.b0_files),
                                                   'in_memory': next((m['inference_provenance'].get('b0') for m in models.values() if m['inference_provenance']), None)},
              'bootstrap': bootstrap.record(), 'access': None if opened is None else {'event': opened['event'], 'seq': opened['seq'], 'entry_sha256': run_io.canonical_sha256(opened)},
              'unavailable': unavailable}
    report = {'schema': REPORT_SCHEMA, 'role': role, 'models': models, 'rows': rows, 'c4': c4}
    if profile['composed'] is not None:                                                                  # K6e: the rows file first (atomic), then the records that name it by its content hash
        report['v_start_rows'], record['composed'] = write_composed(profile, run_dir, vrows or {})
    record, report = _plain(record), _plain(report)
    run_io.write_json_atomic(run_dir / RECORD_FILE, record)
    run_io.write_json_atomic(run_dir / REPORT_FILE, report)
    return {'record': record, 'report': report}


def selection_record(role: str, checked: Mapping, inputs: EvalInputs):
    """What the rows of the evaluated role were used for before the evaluation, for the record, or None for a role that took no part in the selection of the checkpoints: on STOP each arm used the
    STOP rows that its own training run recorded for early stopping. For the EPT the record holds the row digest of its training identity (``early_stopping_rows``: role, number of batches, sha256 of the
    rows; ``stop_sha256`` is the hash of the digest without its batch seed). This runner cannot see the rows of the matched B1: ``B1Heads`` records the role only (``stop_roles``), and a matched B1 stops on
    the same matched subset (``membership.stop``) as the EPT whereas an unmatched ``B1Heads.fit`` stops on the whole STOP, so the record says neither.

    해석: STOP은 판정의 자료이면서 두 쪽의 조기 종료 자료이다. 그 숫자는 따로 떼어 둔 자료의 추정이 아니므로 기록에 그 사실과 증거(EPT의 행 요약, B1이 기록한 역할)를 적는다(결정 D13). B1이 어느
    행으로 멈췄는지는 이 실행기가 볼 수 없으므로 추측해 적지 않는다."""
    if role != 'STOP':
        return None
    stop_roles = getattr(inputs.b1, 'stop_roles', None) if inputs.b1 is not None else None
    return {'role': 'STOP', 'note': 'each arm used the STOP rows that its own training run recorded for early stopping; this runner cannot see the row set of the matched B1; the numbers are the '
                                     'decision data of S4b (decision D13), not a held-out estimate',
            'ept': {'early_stopping_rows': checked['stop_rows'], 'stop_sha256': checked['stop_sha256']},
            'b1': None if inputs.b1 is None else {'early_stopping_rows': 'the stop_rows of its fit_head: STOP matches, the matched subset of a matched B1 or all of STOP; this runner cannot see which '
                                                                         '(B1Heads records the role only)', 'stop_roles': None if stop_roles is None else list(stop_roles)}}


def _pass_part(result: Mapping, exposure: str, population: str) -> dict:
    """What one pass of the inference reports next to the main numbers: its ``exposure`` and ``population``, its provenance, the counts of the C4 endpoint (for the primary and the
    secondary pass ``population_candidate`` and ``population_selected``) and the match-balanced Brier of the EPT and of the matched B1 on the rows of that endpoint with their
    difference (``None`` where the endpoint has no row). It makes no decision: the rule of C4 reads the first pass only.

    해석: 두 번째 추론(관측 노출, 나머지 행)은 판정에 쓰지 않고 곁에 보고한다. 후보 행 수와 선택 행 수가 함께 있어야 제외된 행의 몫을 읽을 수 있다(D14)."""
    found = next((ep for ep in result['endpoints'] if ep['name'] == C4_ENDPOINT), None)
    part = {'exposure': exposure, 'population': population, 'provenance': _plain(result.get('provenance')), 'counts': None, 'brier': None}
    if found is None:
        return part
    part['counts'] = _plain(found['counts']) if 'counts' in found else None
    match, y = np.asarray(found['match']), np.asarray(found['y'], dtype=np.float64)
    if len(match):
        _, inverse, counts = np.unique(match, return_inverse=True, return_counts=True)
        weight = 1.0 / counts[inverse.reshape(-1)]                                                    # 1/n_m, as in C4
        ept, b1 = proper.brier(np.asarray(found['p_ept'], dtype=np.float64), y, weight), proper.brier(np.asarray(found['p_b1'], dtype=np.float64), y, weight)
        part['brier'] = {'ept': ept, 'b1': b1, 'diff': ept - b1}
    return part


def _v_start_part(parts: Mapping) -> dict:
    """The V_start CRPS of one model from the rows the backend gives (``q19`` (n, 19), ``realized`` (n,), ``match`` (n,)), match-balanced (weights 1/n_m)."""
    match = np.asarray(parts['match'])
    _, inverse, counts = np.unique(match, return_inverse=True, return_counts=True)
    weight = 1.0 / counts[inverse.reshape(-1)]
    return {'crps19': proper.crps19(parts['q19'], parts['realized'], weight), 'n_rows': int(len(match)), 'n_matches': int(len(counts))}


def _c4_part(trained, inputs, endpoints_by_seed, bootstrap, unavailable, checked, b1_record):
    """The C4 row with the FIT row digest of the models and the hash of the matched B1's files beside it, or None with a reason in ``unavailable``.

    해석: C4는 같은 FIT 구성원으로 학습한 세 시드와 한 B1의 비교이므로 행 옆에 그 FIT 행 요약과 B1 폴더 해시를 남긴다(25 % 구성원 자체는 B1Heads가 기록하지 않아 검사하지 못한다)."""
    def refuse(reason):
        unavailable.append({'item': 'C4', 'model': None, 'reason': reason})
        return None
    if {run.seed for run in trained} != set(inference.SEEDS):
        return refuse(f'C4 needs the three seed values {inference.SEEDS} (the models have {[run.seed for run in trained]})')
    if inputs.b1 is None:
        return refuse('C4 needs the matched 25% B1 (the inputs hold none)')
    if any(endpoints_by_seed.get(seed) is None for seed in inference.SEEDS):
        return refuse('C4 needs the endpoint ' + C4_ENDPOINT + ' of the inference of every seed, which is unavailable')
    try:
        ids, matrix = bootstrap.get()
    except Unavailable as error:
        return refuse(error.reason)
    try:
        row = c4_statistic(endpoints_by_seed, ids, matrix)
    except Unavailable as error:
        return refuse(error.reason)
    except ValueError as error:
        raise EvaluationRefused(f'C4 cannot be made from the endpoints: {error}') from None
    row['fit_sha256'] = checked['fit_sha256']
    row['b1_folder_sha256'] = None if b1_record is None else b1_record['folder_sha256']
    return row


# ---- the inputs manifest of an evaluation and its loader ---------------------------------------------------------------------------------------
EVAL_INPUTS_SCHEMA = 'ept-run-eval-inputs-v1'
_EVAL_FIELDS = frozenset({'schema', 's1', 's2', 'labels', 'terrain', 'scales', 'scales_provenance', 'prices', 'tier', 'b1', 'b0'})
_REQUIRED_PATHS = ('s1', 's2', 'labels', 'scales')
_OPTIONAL_PATHS = ('terrain', 'prices', 'tier', 'b1', 'b0')


def _absolute(name: str, text) -> str:
    if not isinstance(text, str) or '\x00' in text or not Path(text).is_absolute() or '..' in Path(text).parts:
        raise runners.InputsRefused(f'evaluation inputs manifest field {name} must be an absolute path without ".."')
    return text


def read_eval_manifest(path) -> tuple:
    """``(manifest, sha256)`` of the evaluation inputs manifest ``path``, read once (strict JSON through ``run.io``): exactly the fields of the training manifest
    (``schema`` ``EVAL_INPUTS_SCHEMA``, ``s1``, ``s2``, ``labels``, ``terrain``, ``scales``, ``scales_provenance``) and ``prices`` (the ``PRICES.json`` file), ``tier`` (the
    tower-tier model folder), ``b1`` (the matched B1 folder) and ``b0`` (the B0 folder), each of the last four an absolute path or null (an evaluation that makes no
    inference needs none of them). ``runners.InputsRefused`` otherwise."""
    document, digest = run_io.read_json_with_sha(path)
    if set(document) != _EVAL_FIELDS:
        raise runners.InputsRefused('the evaluation inputs manifest must have exactly the fields ' + ', '.join(sorted(_EVAL_FIELDS)))
    if document['schema'] != EVAL_INPUTS_SCHEMA:
        raise runners.InputsRefused(f'the evaluation inputs manifest schema must be {EVAL_INPUTS_SCHEMA}')
    for name in _REQUIRED_PATHS:
        _absolute(name, document[name])
    for name in _OPTIONAL_PATHS:
        if document[name] is not None:
            _absolute(name, document[name])
    if document['scales_provenance'] is not None and not isinstance(document['scales_provenance'], dict):
        raise runners.InputsRefused('evaluation inputs manifest field scales_provenance must be null or an object')
    return document, digest


def manifest_bound_files(manifest: Mapping) -> list:
    """The files that the ledger binds as inputs of the evaluation stage, as sorted absolute paths with forward slashes: those of ``runners.manifest_files`` (the manifests of
    the stores, the scales, the terrain), the prices file and every file under the tier, B1 and B0 folders that the manifest names."""
    files = set(runners.manifest_files(manifest))
    if manifest['prices'] is not None:
        files.add(Path(manifest['prices']).as_posix())
    for name in ('tier', 'b1', 'b0'):
        if manifest[name] is not None:
            files |= {path.as_posix() for path in Path(manifest[name]).rglob('*') if path.is_file()}
    return sorted(files)


def load_extras(manifest: Mapping) -> dict:
    """The prices, the tier model, the matched B1 and the B0 heads of the manifest, by the loaders of the earlier stacks, and ``b0_files`` (``{name: path}`` of the two files
    of the saved B0 set, which the ledger hashes directly from the disk) and ``b1_files`` (``{relative name: path}`` of every file under the B1 folder, hashed for the record: the B1
    folder hash beside the C4 row). A name that the manifest leaves null is None."""
    extras = {'prices': None, 'tier': None, 'b1': None, 'b0': None, 'b0_files': {}, 'b1_files': {}}
    if manifest['prices'] is not None:
        extras['prices'] = value.load_prices(manifest['prices'])
    if manifest['tier'] is not None:
        extras['tier'] = towertier.TierModel.load(Path(manifest['tier']))
    if manifest['b1'] is not None:
        extras['b1'] = baselines_b1.B1Heads.load(Path(manifest['b1']))
        folder = Path(manifest['b1'])
        extras['b1_files'] = {path.relative_to(folder).as_posix(): path for path in sorted(folder.rglob('*')) if path.is_file()} if folder.is_dir() else {}
    if manifest['b0'] is not None:
        extras['b0'] = baselines_b0.B0Heads.load(Path(manifest['b0']))
        extras['b0_files'] = {name: Path(manifest['b0']) / name for name in (baselines_b0.JSON_NAME, baselines_b0.NPZ_NAME)}
    return extras


# Ref: c_ept/run/runners.py ``load_inputs`` (itself after c_ept/workflows/run_s4a.py ``_load_inputs``): the same stores, the label store verified against its manifest, the terrain
# checked against the build of the S2 views, the static channels, the B0 loss scales; here with the extras of an evaluation.
def load_eval_inputs(manifest_path) -> EvalInputs:
    """The ``EvalInputs`` of an evaluation run, from the manifest at ``manifest_path``. ``runners.InputsRefused`` for a manifest without a terrain or a store that is not as
    the rule says (the same rules as the training loader)."""
    manifest, _ = read_eval_manifest(manifest_path)
    if manifest['terrain'] is None:
        raise runners.InputsRefused('the terrain is required: the manifest names no terrain folder')
    s1, s2, labels = runners.open_stores(manifest)
    runners.check_terrain_build(s2.manifest, runners.terrain_hashes(manifest['terrain']))
    labels.verify()
    values, path = runners.read_scales(manifest)
    static = encoder.static_channels(terrain.Terrain(manifest['terrain']))
    base = runners.Inputs(s1=s1, s2=s2, labels=labels, static=static, static_source=None, scales=values, scales_path=path)
    return EvalInputs(base=base, **load_extras(manifest))


# ---- the stage body, the plan and the start ----------------------------------------------------------------------------------------------------------
BODY = 'c_ept.run.evaluate:evaluate_stage'
LOADER = 'c_ept.run.evaluate:load_eval_inputs'
BACKEND = 'c_ept.run.evaluate:Backend'
EVALUATE_MODULES = ('c_ept.run.supervisor', 'c_ept.run.evaluate')       # the entry modules of the code identity: the supervisor and everything the stage body imports
NONE_TEXT = '-'                                                          # the text of an optional stage argument that is not given
_NAME = re.compile(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*')


def _by_name(spec, what: str):
    """The object that the text ``module.path:name`` names; ``IORefused`` for a text that is not of that form, ``runners.InputsRefused`` for one that cannot be imported."""
    if not isinstance(spec, str) or _NAME.fullmatch(spec) is None:
        raise run_io.IORefused(f'the {what} must be named module.path:name')
    module_name, _, name = spec.partition(':')
    try:
        return getattr(importlib.import_module(module_name), name)
    except (ImportError, AttributeError):
        raise runners.InputsRefused(f'the {what} cannot be imported') from None


def _whole(name: str, text, low: int) -> int:
    if not isinstance(text, str) or not text.isascii() or not text.isdigit() or len(text) > 9 or int(text) < low:
        raise run_io.IORefused(f'stage argument {name} must be a whole number of at least {low}')
    return int(text)


def _optional(name: str, text):
    if not isinstance(text, str) or text == '' or '\x00' in text:
        raise run_io.IORefused(f'stage argument {name} must be a text or {NONE_TEXT!r}')
    return None if text == NONE_TEXT else text


def _gate_and_access(profile, access_dir, lock_path, rehearsal_run):
    """``(access ledger or None, gate or None)`` for the role of the profile; ``AccessRefused`` when the role needs what is not given (DEV: the access ledger; TEST: the ledger
    and the three things of the gate)."""
    role = profile['role']
    access = None if access_dir is None else AccessLedger(access_dir)
    if role in ('DEV', 'TEST') and access is None:
        raise AccessRefused(f'the role {role} needs the access ledger (decisions D4, D20): the folder of ACCESS_LEDGER.json')
    gate = None
    if role == 'TEST':
        if lock_path is None or rehearsal_run is None:
            raise AccessRefused('the role TEST needs the gate: the S5 lock file (its hash is the holdout_lock_sha256 of the profile) and the folder of the finished DEV rehearsal (decision D20)')
        gate = HoldoutGate(lock_path, rehearsal_run, access)
    return access, gate


ALLOW_CUSTOM_TEXT = 'allow_custom'                                       # the text of the stage argument that sets the test-only switch (NONE_TEXT: not set)


def stage_arguments(profile_path, profile, manifest_path, loader: str, backend: str, constants: runners.RunnerConstants, access_dir=None, lock_path=None,
                    rehearsal_run=None, allow_custom=False) -> tuple:
    """The twelve texts of the stage body: the profile file and its lock, the evaluation inputs manifest, the loader and the backend (``module:name``), the device, the threads and
    the batch size (they travel on the command line of the child), the access ledger folder, the S5 lock file and the DEV rehearsal run (``NONE_TEXT`` when not given) and the
    test-only switch (``ALLOW_CUSTOM_TEXT`` or ``NONE_TEXT``). The hash of the lock file is not a text: the TEST profile fixes it (``holdout_lock_sha256``)."""
    given = lambda value_: NONE_TEXT if value_ is None else str(_abspath(value_))             # noqa: E731
    return (str(_abspath(profile_path)), profile.sha256, str(_abspath(manifest_path)), loader, backend, constants.device, str(constants.threads),
            str(constants.batch_size), given(access_dir), given(lock_path), given(rehearsal_run),
            ALLOW_CUSTOM_TEXT if allow_custom else NONE_TEXT)


def _abspath(path) -> Path:
    return Path(os.path.abspath(path))


def plan(profile, profile_path, manifest_path, *, trained: list, manifest: Mapping, constants: runners.RunnerConstants, loader: str = LOADER, backend: str = BACKEND,
         access_dir=None, lock_path=None, rehearsal_run=None, allow_custom=False, bodies=None) -> list:
    """The stage of an evaluation run for the supervisor: ``evaluate``, a ``module:function(run_dir, *texts)`` body (``bodies`` replaces it in a test).

    The inputs that its ledger hashes before and after the stage are the profile file, the inputs manifest and the files it binds (``manifest_bound_files``), the files of every
    checkpoint that is read (``TrainedRun.bound_files``) and, for the role TEST, the S5 lock file and the record and the report of the DEV rehearsal (the evidence of the gate). The
    outputs are the record and the report."""
    profile_file, manifest_file = _abspath(profile_path), _abspath(manifest_path)
    inputs = {profile_file.as_posix(), manifest_file.as_posix(), *manifest_bound_files(manifest)}
    for run in trained:
        inputs |= set(run.bound_files())
    if lock_path is not None:
        inputs.add(_abspath(lock_path).as_posix())
    if rehearsal_run is not None:
        inputs |= {(_abspath(rehearsal_run) / name).as_posix() for name in (RECORD_FILE, REPORT_FILE)}
    arguments = stage_arguments(profile_file, profile, manifest_file, loader, backend, constants, access_dir, lock_path, rehearsal_run, allow_custom)
    return [supervisor.Stage(STAGE, (bodies or {}).get(STAGE, BODY), args=arguments, inputs=sorted(inputs), outputs=stage_outputs(profile))]


def evaluate_stage(run_dir, profile_path, profile_sha256, manifest_path, loader, backend, device, threads, batch_size, access_dir, lock_path, rehearsal_run,
                   allow_custom) -> None:
    """Stage body ``evaluate``: ``evaluate`` for the arguments of ``stage_arguments``. The profile (its lock checked, an evaluate profile), the constants and the texts are
    checked before anything is read from the stores or written; the code identity that the gate and the DEV line name is the one that the supervisor bound in the ledger header
    (read-only), and so is the record of the stage body (``body_record`` of the text that the supervisor started: ``verify_body`` finds out here, before the inputs are loaded, that its file
    is still the one that was hashed). A refusal of the evaluation, of the access ledger or of the gate is exit code 2 of the child, every other error 3 (``supervisor.child_main``)."""
    profile = run_profile.load(profile_path)
    if not isinstance(profile_sha256, str) or profile.sha256 != profile_sha256:
        raise run_io.IORefused('the profile file is not the one the run was started with (its sha256 differs)')
    if profile.kind != 'evaluate':
        raise EvaluationRefused('the evaluation takes an evaluate profile, not a ' + profile.kind + ' profile')
    constants = runners.RunnerConstants(batch_size=_whole('batch_size', batch_size, 1), device=device, threads=_whole('threads', threads, 1))
    access_dir, lock_path, rehearsal_run = (_optional(name, text) for name, text in (('access_dir', access_dir), ('lock_path', lock_path), ('rehearsal_run', rehearsal_run)))
    if allow_custom not in (NONE_TEXT, ALLOW_CUSTOM_TEXT):
        raise run_io.IORefused(f'stage argument allow_custom must be {ALLOW_CUSTOM_TEXT!r} or {NONE_TEXT!r}')
    run_numeric.apply(constants.device)                                  # m3 of the K6c review: the policy is held before the inputs (the fitted B1/B0/tier objects) are loaded; ``evaluate`` sets and records it again
    load, make = _by_name(loader, 'inputs loader'), _by_name(backend, 'backend')
    access, gate = _gate_and_access(profile, access_dir, lock_path, rehearsal_run)
    _, manifest_sha256 = read_eval_manifest(manifest_path)              # what the record and the gate say about the inputs: the hash of the manifest this run was started with
    header = run_ledger.Ledger(run_dir).header                          # read-only: the identity and the stage body that the supervisor bound when it made the ledger
    evaluation = header.runner.get('evaluation')
    named = evaluation.get('sources') if isinstance(evaluation, Mapping) else None
    body = named.get('body') if isinstance(named, Mapping) else None
    if body is not None:
        verify_body(body)
        body = dict(body)
    inputs = load(manifest_path)
    if not isinstance(inputs, EvalInputs):
        raise runners.InputsRefused('the inputs loader did not return EvalInputs')
    evaluate(profile, run_dir, inputs=inputs, constants=constants, backend=make(), access=access, gate=gate, identity=header.identity, loader=loader, backend_name=backend,
             manifest_sha256=manifest_sha256, allow_custom=allow_custom == ALLOW_CUSTOM_TEXT, body=body)


def runner_record(profile, manifest_sha256: str, manifest: Mapping, constants: runners.RunnerConstants, trained: list, sources: Mapping, approval=None) -> dict:
    """The constants and the effective values for ``Header.runner`` (written once, when the ledger is made): the runner constants, the role, the profile as it is (every field
    written out) with its two hashes, the models with their seeds, settings and the hash of their training record, the ``sources`` (loader, backend, inputs manifest hash, the
    test-only switch and the stage body) and the inputs manifest with the files it binds, the numeric policy that the run asks for (``numeric.requested_record``) and the text of the
    approval (``runners.approval_record``: None when none was given)."""
    return {**constants.to_record(),
            'evaluation': {'role': profile['role'], 'profile': profile_record(profile), 'profile_sha256': profile.sha256,
                           'profile_without_role_sha256': profile_without_role_sha256(profile), 'sources': dict(sources),
                           'models': {run.name: {'seed': run.seed, 'settings': dict(run.settings), 'train_record_sha256': run.record_sha256} for run in trained},
                           'inputs': {'manifest_sha256': manifest_sha256, 'bound': manifest_bound_files(manifest)}},
            'numeric': run_numeric.requested_record(), 'approval': runners.approval_record(approval)}


def _start(profile_path, run_dir, *, manifest_path, spec_path, budget, constants, access_dir, lock_path, rehearsal_run, allow_custom, scrub_keys, loader, backend,
           bodies, stream, approval, options, allow_custom_body=False) -> int:
    """Everything that can be refused before a run folder exists, cheaply: the kind of the profile, the approval text, the profile fields that nothing reads yet, CUDA, the manifest
    (read once), that the backend can evaluate the role and that a DEV or TEST run takes the default loader and backend, the stage body (its record is a source of the run: the TEST gate
    compares it), what the role needs (the access ledger, the gate and its evidence), the checkpoints and that they are the seeds of one configuration, the complete stores and the
    condition of the scales file."""
    profile = run_profile.load(profile_path)
    if profile.kind != 'evaluate':
        raise EvaluationRefused('the evaluation takes an evaluate profile, not a ' + profile.kind + ' profile')
    runners.approval_record(approval)
    check_wired_fields(profile)
    check_composed(profile)
    check_bootstrap_rows(profile)
    runners.check_cuda(constants)
    manifest, manifest_sha256 = read_eval_manifest(manifest_path)
    role = profile['role']
    if not _by_name(backend, 'backend')().evaluates(role):
        raise EvaluationRefused(f'the backend cannot evaluate the role {role} here (the default backend evaluates DEV and STOP; TEST waits for a TEST reader of the stores, S8); nothing was opened or logged')
    stage_body = supervisor.Stage(STAGE, (bodies or {}).get(STAGE, BODY)).body                  # the text that the supervisor will start (a function is named by its module and qualified name)
    if role in ('DEV', 'TEST') and stage_body != BODY and not allow_custom_body:
        raise EvaluationRefused(f'the role {role} takes the default stage body {BODY} only: a run with another stage body is not the run that its rehearsal was (decision D20, K6b review r4); '
                                'the test-only switch allow_custom_body lets a fixture name another one; nothing was started')
    body = body_record(stage_body)
    sources = sources_of(loader, backend, manifest_sha256, allow_custom, body)
    check_sources(role, sources)
    access, gate = _gate_and_access(profile, access_dir, lock_path, rehearsal_run)
    trained = load_models(profile)
    if gate is not None:
        gate.check_evidence(profile, sources=sources, models={run.name: run.record_sha256 for run in trained})
    elif access is not None:
        access.entries()                                                # a DEV ledger must exist and verify
    first = check_agreement(trained)
    runners.check_scales_condition({'kill_law': first.settings['kill_law'], 'k_condition': first.settings['k_condition']}, manifest['scales'])
    _, s2, _ = runners.open_stores(manifest)
    if manifest['terrain'] is not None:
        runners.check_terrain_build(s2.manifest, runners.terrain_hashes(manifest['terrain']))
    stages = plan(profile, profile_path, manifest_path, trained=trained, manifest=manifest, constants=constants, loader=loader, backend=backend, access_dir=access_dir,
                  lock_path=lock_path, rehearsal_run=rehearsal_run, allow_custom=allow_custom, bodies=bodies)
    return supervisor.run(profile, run_dir, stages=stages, spec_path=spec_path, budget=budget, cuda_visible_devices=constants.cuda_visible_devices, threads=constants.threads,
                          identity_modules=EVALUATE_MODULES, runner=runner_record(profile, manifest_sha256, manifest, constants, trained, sources, approval), scrub_keys=scrub_keys,
                          stream=stream, **options)


def start(profile_path, run_dir, *, manifest_path, spec_path, budget, constants: runners.RunnerConstants = runners.RunnerConstants(), access_dir=None, lock_path=None,
          rehearsal_run=None, allow_custom=False, scrub_keys=(), loader: str = LOADER, backend: str = BACKEND, bodies=None, stream=None, approval=None,
          allow_custom_body=False, **options) -> int:
    """Start (or resume) the evaluation run of the evaluate profile at ``profile_path`` in ``run_dir`` and return the exit code of the process (0, 2 or 3: the table of
    ``supervisor``). Everything that is wrong with the profile, the manifest, the checkpoints, the access ledger or the gate before the supervisor starts is a printed refusal (2)
    with the match keys of ``scrub_keys`` removed, and leaves no run folder. ``approval``: the text of the author's approval, recorded as given in the ledger header
    (``runners.approval_record``; None records null). ``options`` go to ``supervisor.run`` (``e6``, ``capture``, ``verify``, ``start_gate``, ``poll_s``,
    ``margin_s``, ``repo``, ``environ``). ``access_dir`` is the folder of the access ledger (roles DEV and TEST), ``lock_path`` the S5 lock file (its hash is not an argument: the
    TEST profile fixes it as ``holdout_lock_sha256``) and ``rehearsal_run`` the run folder of the finished DEV rehearsal (role TEST). The roles DEV and TEST take the default ``loader`` and
    ``backend`` only; ``allow_custom`` is the test-only switch of DEV for a made-up loader or backend (it is recorded; the role TEST never takes it). A TEST run needs a
    backend that can evaluate TEST (``Backend.evaluates``), else it is refused here, before the one opening is spent. The roles DEV and TEST also take the default stage body
    (``BODY``) only, as the command line does (K6b review r4, m4 of the K6c review): ``bodies`` that names another one for the stage is a refusal unless the test-only switch
    ``allow_custom_body`` is set (a fixture; it is not recorded, the body itself is, and the TEST gate compares it with the rehearsal's).

    Importing this module imports numpy and torch. A process that has to apply the native-thread policy first (the command line of K6c) imports ``c_ept.run.guards``, calls
    ``native_policy()`` and only then imports this module; ``supervisor.run`` calls it again as its first step.
    """
    return supervisor.main_guard(lambda: _start(profile_path, run_dir, manifest_path=manifest_path, spec_path=spec_path, budget=budget, constants=constants,
                                                access_dir=access_dir, lock_path=lock_path, rehearsal_run=rehearsal_run, allow_custom=allow_custom,
                                                scrub_keys=scrub_keys, allow_custom_body=allow_custom_body,
                                                loader=loader, backend=backend, bodies=bodies, stream=stream, approval=approval, options=options),
                                 scrub_keys=scrub_keys, stream=stream)
