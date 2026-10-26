r"""EPT의 B0 척도 (S4a 과제 4): 손실 (LOSS-4)의 분모 ℓ̄^B0_head를 S3 전체 실행의 B0 머리로 FIT 행에서 한 번 재어, 모형 파일의 해시와 함께 ``B0_SCALES.json``에 고정한다.

이 모듈은 설계서 7절(머리별 NLL ÷ 그 머리의 B0 NLL)과 계획 Q8(B0 NLL 상수는 S3 전체 실행의 B0 머리로 FIT 행에서 한 번 계산해 해시와 함께 고정한다)·P9(학습 행 가중은 같다)·Q12(범위 문구),
판정 R14(학습 통계는 FIT만)·R22를 구현한다. 이 모듈은 킬이 하나 이상 난 교전을 다룬다 (G-S3c C3). 블록(머리와 보기 종류의 한 쌍)은 ``loss.BLOCKS``의 열세 개이다.
방법. S3 실행 폴더의 ``fit/b0``에서 B0 머리를 읽고 (STAGES.json이 적은 해시와 같은 파일만 읽는다), FIT 경기의 행에서 가린 머리 여덟과 열린 머리 다섯의 NLL을 ``B0Heads.nll``로 구한다.
척도는 EPT가 학습하는 행과 같은 행 위의 평균이다 (SCALE-1): M2Rate는 프레임 행(한 프레임 행의 열여덟 구간 점수의 합), 다른 머리는 B0의 설계 행(M3Gold는 팀 항목 하나)이고,
M3Kills·M3Gold·M3Obj는 규모마다 행 평균을 낸 뒤 규모끼리 평균한다 (``loss.SIZE_NORMALISED``). 평균에 드는 행은 B0가 점수에 올린 행이고, 구역 −1 행과 노출 없는 행은 점수에서 빠져 있다 (판정 R22). 빠진 수 ``n_unscored``의 단위는 머리마다 다르다: M2Rate는 노출 없는 구간 칸이고 (행이 아니다), M3Gold는 팀 항목, 그 밖은 설계 행이다 (``block_units``). 관측이 한 칸도 없는 프레임은 `n_rows`에 포함되지 않지만, 그 프레임의 18개 구간 칸은 `n_unscored`에 포함된다. `n_unscored`는 제외된 프레임 수가 아니라 무노출 칸 수이다.
S3 실행 폴더는 읽기만 한다. 결과 파일은 지정한 폴더 ``out``에 쓰고, 이미 있는 파일은 덮어쓰지 않는다. 실제 실행은 제어자가 Strata를 끈 뒤에만 한다 (E6).

식 (식 번호는 ``(SCALE-n)``)

    (SCALE-1)  ℓ̄^B0_head = (1 / N) Σ_{i ∈ FIT의 점수가 있는 행} ℓ^B0_i ;   M3Kills·M3Gold·M3Obj: (1 / n_size) Σ_{s ∈ 나온 규모} (1 / N_s) Σ_{i: s_i = s} ℓ^B0_i       ((LOSS-3)과 같은 평균)

가정과 선택

    · 척도는 학습 행 가중(같은 가중, 계획 P9)의 평균이다. 경기 가중 1/n_u는 평가에만 쓰므로 척도에 넣지 않는다. 평가는 DEV의 경기 가중 평균을 같은 상수로 나눈다 (설계서 8절: 학습 손실과 같은 척도).
    · 역할은 FIT뿐이다. STOP, DEV와 그 밖의 역할은 점수를 구하지 않고 (``ROLE``), 학습 통계는 FIT로만 잰다 (판정 R14).
    · B0는 FIT에서 맞춘 모형이므로 척도는 B0의 학습 표본 안 NLL이다. 표본 밖 점수가 아니다.
    · 척도가 0이거나 유한하지 않은 머리는 손실의 분모가 될 수 없으므로 거부한다.

입력과 출력

    b0_scales(run_s3, out=None, *, runs_root, check) → {블록: 척도}, ``out``에 ``B0_SCALES.json``       load_scales(path, expected_provenance=None) → {블록: 척도}, 열세 블록·역할·해시·출처 구조를 확인한다 (출처가 요청한 실행의 것인지는 expected_provenance를 줄 때만 대조한다)
    head_scale(head, scores, sizes=None) → (척도, 행 수, 규모별 평균)
    scales_condition(path) → {kill_law, k_condition, k_reps_sha256}: 척도를 만든 B0의 조건 (S4b 준비 M3b, 결정 D24). 파일은 기본값('poisson', 'raw', 없음)이 아닌 필드만 적으므로 옛 파일은 기본값으로 읽힌다
    ``python -m c_ept.ept.scales --s3 <S3 실행> --out <폴더>``: E6를 확인한 뒤 한 번 실행한다
"""
# Ref: docs/superpowers/specs/2026-10-20-c-ept-design.md section 7 (the loss: NLL divided by the B0 NLL) and 8 (the scores);
#      docs/superpowers/plans/2026-10-22-c-ept-s4a.md decision Q8 (the B0 constants are computed once on the FIT rows of the full S3 run and fixed with a hash)
#      and Q12; docs/superpowers/plans/2026-10-21-c-ept-s3.md decision P9; controller rulings R14 (statistics on FIT only), R22 (left-out rows are counted) and E6
#      (no run while Strata is up); math-control note 04, C.
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from c_ept import config as C
from c_ept import runlog
from c_ept.data import store_core
from c_ept.ept import loss
from c_ept.data.head_rows import HEAD_NAMES, check_k_condition
from c_ept.models import killlaw

SCALES_NAME = 'B0_SCALES.json'
SCHEMA = 'b0_scales_v1'
ROLE = 'FIT'                                            # the one role that is scored here
KINDS = ('masked', 'open')
SIZE_NAMES = ('T', 'S', 'ASYM', 'P')                    # the size classes 0 .. 3 (``units.SIZE_NAMES``; a test pins the equality)
MODEL_FILES = ('b0_heads.json', 'b0_heads.npz')         # the files of a saved ``B0Heads`` (``baselines_b0.JSON_NAME`` and ``NPZ_NAME``; a test pins them)
MODEL_DIR = 'fit/b0'                                    # in the S3 run folder
RULE = ('mean of the B0 NLL over the FIT rows of the block (M2Rate: over the frame rows, the 18 bins of a frame row added; M3Gold: over the team entries); '
        'M3Kills, M3Gold and M3Obj: the mean over the sizes that occur of the mean over the rows of a size; the rows that B0 leaves out of its scores do not count '
        '(for M2Rate the left-out units are time-bin cells, not rows); every row has the same weight (plan P9)')
PROVENANCE_FIELDS = ('s3_run', 's1', 's2', 'labels_store_sha256', 'model_files_sha256')   # what ``load_scales(expected_provenance=)`` binds
CONDITION_FIELDS = ('kill_law', 'k_condition', 'k_reps_sha256')   # the condition of the B0 that made the scales (S4b 준비 M3b, 결정 D24); written only when not the default
CONDITION_DEFAULTS = {'kill_law': 'poisson', 'k_condition': 'raw', 'k_reps_sha256': None}      # what a file without the field says (every file of an earlier run)


def block_units(key: str) -> tuple:
    """``(unit of n_rows, unit of n_unscored)`` of a block. M2Rate: the rows of the mean are frame rows, but ``HeadScores.n_unscored`` counts the time-bin cells without exposure
    (``head_rows``). A wholly unobserved frame contributes no row to ``n_rows``, but its 18 time-bin cells contribute to ``n_unscored``. M3Gold: team entries. The other heads: design rows."""
    head = loss.split_key(key)[0]
    if head == 'M2Rate':
        return 'frame_row', 'time_bin_cell'
    return ('team_entry', 'team_entry') if head == 'M3Gold' else ('design_row', 'design_row')


def head_scale(head: str, scores, sizes=None) -> tuple:
    """``(scale, n_rows, by_size)`` of one head from its B0 scores on the FIT rows: ``scores`` is the ``head_rows.HeadScores`` of ``B0Heads.nll``, ``sizes`` the size class (0 .. 3)
    of every design row of ``scores`` (required for M3Kills, M3Gold and M3Obj, which are averaged over the sizes). ``n_rows`` counts the rows of the mean (the frame rows for M2Rate),
    ``by_size`` is ``{name: mean}`` of the sizes that occur (None for the other heads). ValueError for a head without a row, a size-averaged head that is given no sizes or sizes
    of another length or outside 0 .. 3, and a scale that is zero or not finite.

    수식
    ----
    (SCALE-1)  ℓ̄^B0_head = (1 / N) Σ_{i} ℓ^B0_i ;    M3Kills·M3Gold·M3Obj:  ℓ̄^B0_head = (1 / n_size) Σ_{s ∈ 나온 규모} ℓ̄^B0_{head,s}

      기호             | 뜻                                                          | 단위 | 범위  | 코드 변수
      ℓ^B0_i           | B0가 행 i에서 낸 NLL (M2Rate는 프레임 행의 구간 점수 합)     | nat  | ≥ 0   | scores.nll, per_row()의 값
      N                | 평균의 행의 수 (M2Rate는 프레임 행, 그 밖에는 설계 행)       | 행   | ≥ 1   | 반환값의 두 번째
      ℓ̄^B0_{head,s}    | 규모 s의 행의 평균, n_size는 나온 규모의 수                  | nat  | > 0   | by_size, 길이

    해석
    ----
    척도는 B0가 EPT와 같은 행에서 낸 평균 NLL이므로, 손실의 비 ℓ̄_head / ℓ̄^B0_head는 B0 수준에서 1이다. 평균은 EPT의 학습 평균 (LOSS-3)과 같은 규칙이다: 점수가 있는 행만, 전문가 셋은 규모별 평균의 평균이다.
    합은 ``math.fsum``으로 정확히 더해 행의 순서가 결과를 바꾸지 않는다. B0는 이 행에서 맞춘 모형이라 척도는 학습 표본 안 값이다.

    근거
    ----
    설계서 7절, 계획 Q8·P9, 판정 R22, 검산 노트 04 C절.
    """
    if head not in HEAD_NAMES:
        raise ValueError(f'unknown head {head!r}')
    values = np.asarray(scores.per_row()[0] if head == 'M2Rate' else scores.nll, dtype=np.float64)
    if values.size == 0:
        raise ValueError(f'the B0 scores of {head} have no row')
    by_size = None
    if head in loss.SIZE_NORMALISED:
        size = None if sizes is None else np.asarray(sizes)
        if size is None or size.shape != np.asarray(scores.nll).shape or size.dtype.kind not in 'iu' or (size < 0).any() or (size >= len(SIZE_NAMES)).any():
            raise ValueError(f'{head} is averaged over the sizes: it needs the sizes, one class 0 .. {len(SIZE_NAMES) - 1} for each of its {np.asarray(scores.nll).size} rows')
        by_size = {SIZE_NAMES[k]: math.fsum(values[size == k]) / int((size == k).sum()) for k in range(len(SIZE_NAMES)) if (size == k).any()}
        value = math.fsum(by_size.values()) / len(by_size)
    else:
        value = math.fsum(values) / values.size
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f'the B0 scale of {head} is {value}: it cannot be the denominator of a loss')
    return value, int(values.size), by_size


def _size_of_rows(s1, kind: str, rows: np.ndarray) -> np.ndarray:
    """The size class of the evaluation rows ``rows`` of ``HeadScores.row``: the onset's unit for a masked head, the unit itself for an open head."""
    size = np.asarray(s1.array('units/size'))
    unit = np.asarray(s1.array('onsets/unit'))[rows] if kind == 'masked' else np.asarray(rows)
    return size[unit].astype(np.int64)


def condition_record(b0) -> dict:
    """The condition fields of ``B0_SCALES.json`` for the B0 set ``b0`` (a ``B0Heads``): ``{}`` for a B0 of the defaults, and otherwise only the fields that are not the default:
    ``kill_law`` when it is not ``'poisson'``, ``k_condition`` when it is not ``'raw'`` and, for ``'binned'``, the ``k_reps_sha256`` of its 31 representatives when the set records them.
    A default B0 therefore gives the bytes of every earlier file.

    해석: 척도는 B0 NLL의 평균이므로 B0가 맞춘 킬 법칙과 K 조건(결정 D5, D24)에 따라 값이 달라진다. 파일이 어느 조건의 척도인지 적어 두어야 K6가 학습 조건과 대조할 수 있다.
    모든 새 필드의 규칙("없음 = 기본값", 계획 M1 구현 기록)을 따라 기본값이면 쓰지 않는다."""
    record = {}
    if b0.kill_law != CONDITION_DEFAULTS['kill_law']:
        record['kill_law'] = b0.kill_law
    if b0.k_condition != CONDITION_DEFAULTS['k_condition']:
        record['k_condition'] = b0.k_condition
        if b0.k_reps_sha256 is not None:
            record['k_reps_sha256'] = b0.k_reps_sha256
    return record


def _run_record(run_s3, runs_root) -> tuple:
    """``(folder, RUN.json)`` of the S3 run (a path, or a name under ``runs_root``); ValueError unless it is an S3 run folder."""
    run = Path(run_s3)
    if not run.is_absolute() and not run.is_dir():
        run = Path(runs_root) / run
    path = run / 'RUN.json'
    if not path.is_file():
        raise ValueError(f'no S3 run folder at {run.name}: it has no RUN.json')
    doc = json.loads(path.read_text(encoding='utf-8'))
    if doc.get('stage') != 's3':
        raise ValueError(f'{run.name} is not an S3 run folder (its stage is {doc.get("stage")!r})')
    return run, doc


def _open_inputs(run: Path, doc: dict, runs_root) -> tuple:
    """The S1 and S2 stores of ``doc`` (opened by name under ``runs_root``) and the label store of the run, each checked against the hash that the run recorded."""
    from c_ept.valuation.flowprice import _open_s2
    from c_ept.workflows.run_s1 import load_s1
    stores = []
    for label, opener in (('S1', load_s1), ('S2', _open_s2)):
        record = doc.get(label.lower()) or {}
        if not record.get('run') or not record.get('files_sha256'):
            raise ValueError(f'RUN.json of {run.name} does not name its {label} store')
        store = opener(Path(runs_root) / record['run'], False)
        if runlog.sha256_json(store.manifest['files']) != record['files_sha256']:
            raise ValueError(f'the {label} store {record["run"]} is not the one that RUN.json of {run.name} recorded')
        stores.append(store)
    return tuple(stores)


def _open_labels(run: Path, ledger):
    from c_ept.workflows import run_s3 as s3_runner
    recorded = ((ledger.stage('labels') or {}).get('outputs') or {}).get('labels/store/')
    if not (run / 'labels' / 'store' / 'STORE.json').is_file() or not (ledger.stage('labels') or {}).get('complete'):
        raise ValueError(f'the label store of {run.name} is not complete')
    identity = s3_runner.identity(run, 'labels/store/')
    if identity != recorded:
        raise ValueError(f'the label store of {run.name} is not the one that STAGES.json recorded')
    labels = store_core.Store(run / 'labels' / 'store')
    labels.verify()
    return labels, identity


def b0_scales(run_s3, out=None, *, runs_root=C.RUNS_ROOT, check=None) -> dict:
    """The B0 scales of the full S3 run ``run_s3`` (a run folder, or its name under ``runs_root``): ``{block: scale}`` for the thirteen blocks of ``loss.BLOCKS`` (floats), computed
    on the FIT rows with the B0 heads of ``fit/b0`` (``head_scale`` for each). With ``out`` (a folder, made when needed) the file ``B0_SCALES.json`` is written there: the scales,
    the S3 run, the SHA-256 of the model files and of the stores, the number of rows and of left-out units (R22; the units are named in ``n_rows_unit`` and ``n_unscored_unit``: M2Rate leaves out cells, not rows) and the means by size, the code state and the hash of the scales.
    ``check``: a function without
    arguments (the E6 guard), called before anything is read and between the heads. The S3 run is read only. ValueError for a folder that is not an S3 run, a fit or label stage that
    is not complete, a model file, store or label store that is not the one the run recorded, a B0 that was not fitted on FIT only, and a scale that is zero or not finite;
    FileExistsError when ``out`` already holds the file (it is computed once and not written over).

    해석: 식은 ``head_scale``가 계산한다. 이 함수는 기록된 파일만 읽도록 모형·저장소·라벨의 해시를 대조하고, 역할을 FIT로 고정해 열세 척도를 모아 해시와 함께 한 번 쓴다.
    """
    from c_ept.models import baselines_b0
    from c_ept.workflows import run_s3 as s3_runner
    tick = check if check is not None else (lambda: None)
    tick()
    path = None if out is None else Path(out) / SCALES_NAME
    if path is not None and path.exists():
        raise FileExistsError(f'{path} already exists: the B0 scales are computed once and never written over')
    run, doc = _run_record(run_s3, runs_root)
    ledger = s3_runner.Ledger(run)
    if not ledger.complete('fit'):
        raise ValueError(f'the fit stage of {run.name} is not complete in STAGES.json: there is no B0 to read')
    folder, recorded = run / MODEL_DIR, (ledger.stage('fit').get('outputs') or {})
    hashes = {f'{MODEL_DIR}/{name}': store_core.sha(folder / name) for name in MODEL_FILES}
    if hashes[f'{MODEL_DIR}/{MODEL_FILES[0]}'] != recorded.get(f'{MODEL_DIR}/{MODEL_FILES[0]}'):
        raise ValueError(f'the model file {MODEL_DIR}/{MODEL_FILES[0]} of {run.name} is not the one that STAGES.json recorded')
    s1, s2 = _open_inputs(run, doc, runs_root)
    labels, labels_sha = _open_labels(run, ledger)
    b0 = baselines_b0.B0Heads.load(folder)
    if tuple(b0.fit_roles) != (ROLE,):
        raise ValueError(f'the B0 heads of {run.name} were fitted on {tuple(b0.fit_roles)}, not on {ROLE} only')
    scale, n_rows, n_unscored, by_size = {}, {}, {}, {}
    for kind in KINDS:
        tick()
        for head, scores in b0.nll(s1, s2, labels, ROLE, kind, check=check).items():
            key = loss.block_key(head, kind)
            sizes = _size_of_rows(s1, kind, np.asarray(scores.row)) if head in loss.SIZE_NORMALISED else None
            scale[key], n_rows[key], per_size = head_scale(head, scores, sizes)
            n_unscored[key] = int(scores.n_unscored)
            if per_size is not None:
                by_size[key] = per_size
    if set(scale) != set(loss.BLOCKS):
        raise ValueError(f'B0 scored {sorted(scale)}, not the thirteen blocks {loss.BLOCKS}')
    scale = {key: scale[key] for key in loss.BLOCKS}
    if path is not None:
        record = {'schema': SCHEMA, 's3_run': run.name, 'role': ROLE, 'rule': RULE,
                  's1': {k: doc['s1'][k] for k in ('run', 'files_sha256')}, 's2': {k: doc['s2'][k] for k in ('run', 'files_sha256')},
                  'labels_store_sha256': labels_sha, 'model_files_sha256': hashes, 'scales': scale, 'n_rows': {key: n_rows[key] for key in loss.BLOCKS},
                  'n_unscored': {key: n_unscored[key] for key in loss.BLOCKS}, 'by_size': by_size,
                  'n_rows_unit': {key: block_units(key)[0] for key in loss.BLOCKS}, 'n_unscored_unit': {key: block_units(key)[1] for key in loss.BLOCKS},
                  'size_normalised': sorted(key for key in loss.BLOCKS if loss.split_key(key)[0] in loss.SIZE_NORMALISED),
                  'code': runlog.code_state(), 'scales_sha256': runlog.sha256_json(scale), **condition_record(b0)}
        path.parent.mkdir(parents=True, exist_ok=True)
        store_core.write_atomic(path, json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + '\n')
    return scale


def _is_hex(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _check_condition(name: str, doc: dict) -> None:
    """ValueError unless the condition fields of ``doc`` that are present are in the form ``condition_record`` writes: a kill law of ``killlaw.LAWS``, a K condition of
    ``head_rows.K_CONDITIONS`` and a lowercase 64-digit ``k_reps_sha256`` only together with the condition ``'binned'``. A field that is absent is the default and is not checked."""
    try:
        killlaw.check_law(doc.get('kill_law', CONDITION_DEFAULTS['kill_law']))
        condition = check_k_condition(doc.get('k_condition', CONDITION_DEFAULTS['k_condition']))
    except ValueError as e:
        raise ValueError(f'{name}: {e}') from None
    if 'k_reps_sha256' in doc and not (condition == 'binned' and _is_hex(doc['k_reps_sha256'])):
        raise ValueError(f"{name}: k_reps_sha256 must be a SHA-256 (64 lowercase hex digits) and only with the k_condition 'binned'")


def _check_provenance(name: str, doc: dict) -> None:
    """ValueError ('provenance') unless ``doc`` holds the five source fields in the form that ``b0_scales`` writes: a non-empty run name, the SHA-256 (64 hex digits) of the
    files of S1 and S2 and of the label store, and the SHA-256 of exactly the two model files."""
    def refuse(field, why):
        raise ValueError(f'{name}: the provenance field {field} {why}')
    run = doc.get('s3_run')
    if not (isinstance(run, str) and run):
        refuse('s3_run', 'must be the name of the S3 run')
    for field in ('s1', 's2'):
        record = doc.get(field)
        if not (isinstance(record, dict) and set(record) == {'run', 'files_sha256'} and isinstance(record['run'], str) and record['run'] and _is_hex(record['files_sha256'])):
            refuse(field, 'must be {run: a name, files_sha256: SHA-256}')
    if not _is_hex(doc.get('labels_store_sha256')):
        refuse('labels_store_sha256', 'must be a SHA-256')
    model = doc.get('model_files_sha256')
    if not (isinstance(model, dict) and set(model) == {f'{MODEL_DIR}/{file}' for file in MODEL_FILES} and all(_is_hex(h) for h in model.values())):
        refuse('model_files_sha256', f'must be the SHA-256 of {sorted(f"{MODEL_DIR}/{file}" for file in MODEL_FILES)}')


def load_scales(path, expected_provenance=None) -> dict:
    """The scales of a ``B0_SCALES.json`` (the file, or the folder that holds it): ``{block: scale}``. ValueError for another schema, scales that do not match their hash,
    scales that are not exactly the thirteen blocks of ``loss.BLOCKS``, a scale that is not a positive finite number (a bool is not), a role other than FIT, another rule or other
    size-normalised blocks, counts or units that are not those of ``b0_scales``, and an empty or malformed provenance (S3 run, S1 and S2 stores, label store, the two model files);
    FileNotFoundError for a missing file.

    ``expected_provenance``: None, or a complete mapping with exactly the fields ``PROVENANCE_FIELDS`` (``s3_run``, ``s1``, ``s2``, ``labels_store_sha256``, ``model_files_sha256``,
    in the form of the file). Every field must equal the file's; a missing or extra field, or a difference, is a ValueError ('expected'). A caller that reuses a file of an earlier
    run (Task 7: the file is there, the stage is not recorded) must build the mapping from the S3 run it was asked for (its RUN.json, STAGES.json and model files), never from
    the artifact. Without the mapping this function checks the structure and the internal consistency of the file only; it cannot know which run the file belongs to.

    해석: 학습은 이 함수가 돌려준 상수만 쓴다. 파일의 값을 손으로 고치면 해시가 달라져 읽기가 멈춘다. 해시는 값만 덮으므로 출처는 ``expected_provenance``로 대조한다.
    """
    path = Path(path)
    if path.is_dir():
        path = path / SCALES_NAME
    doc = json.loads(path.read_text(encoding='utf-8'))
    name = path.name
    if doc.get('schema') != SCHEMA:
        raise ValueError(f'{name}: the schema is {doc.get("schema")!r}, not {SCHEMA!r}')
    scale = doc.get('scales')
    if not isinstance(scale, dict):
        raise ValueError(f'{name}: the scales must map the thirteen blocks to their scale')
    for key, value in scale.items():
        if isinstance(value, bool) or not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0):
            raise ValueError(f'{name}: the scale of the block {key} must be finite and positive, not {value!r}')
    if runlog.sha256_json(scale) != doc.get('scales_sha256'):
        raise ValueError(f'{name}: the scales do not match their hash (scales_sha256)')
    if set(scale) != set(loss.BLOCKS):
        raise ValueError(f'{name}: the scales hold the blocks {sorted(scale)}, not exactly the thirteen blocks {list(loss.BLOCKS)}')
    if doc.get('role') != ROLE:
        raise ValueError(f'{name}: the role is {doc.get("role")!r}, not {ROLE!r}')
    if doc.get('rule') != RULE:
        raise ValueError(f'{name}: the rule is not the rule of this module')
    if doc.get('size_normalised') != sorted(key for key in loss.BLOCKS if loss.split_key(key)[0] in loss.SIZE_NORMALISED):
        raise ValueError(f'{name}: size_normalised is not the list of the size-normalised blocks')
    for field in ('n_rows', 'n_unscored'):
        counts = doc.get(field)
        if not (isinstance(counts, dict) and set(counts) == set(loss.BLOCKS)
                and all(isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in counts.values())):
            raise ValueError(f'{name}: {field} must hold one count (an int >= 0) for each of the thirteen blocks')
    for field, index in (('n_rows_unit', 0), ('n_unscored_unit', 1)):
        if doc.get(field) != {key: block_units(key)[index] for key in loss.BLOCKS}:
            raise ValueError(f'{name}: {field} is not the unit of the counts of each block')
    _check_provenance(name, doc)
    _check_condition(name, doc)
    if expected_provenance is not None:
        if not isinstance(expected_provenance, Mapping) or set(expected_provenance) != set(PROVENANCE_FIELDS):
            raise ValueError(f'{name}: the expected provenance must be a mapping with exactly the fields {list(PROVENANCE_FIELDS)}')
        for field in PROVENANCE_FIELDS:
            if doc[field] != expected_provenance[field]:
                raise ValueError(f'{name}: the field {field} is not the expected one: the file is not the scales of the requested S3 run')
    return {key: float(value) for key, value in scale.items()}


def scales_condition(path) -> dict:
    """The condition of the B0 that made the scales of a ``B0_SCALES.json`` (the file, or the folder that holds it), with the defaults filled in: ``{'kill_law', 'k_condition',
    'k_reps_sha256'}``. A field that the file does not hold is the default (``'poisson'``, ``'raw'``, ``None``), so the file of every earlier run reads as the defaults. The file is
    checked as ``load_scales`` checks it (without a provenance mapping); ValueError for a file that is not valid, FileNotFoundError for a missing one.

    해석: K6는 이 값을 학습 정체성의 ``kill_law``·``k_condition``·``k_reps_sha256``과 대조해 다르면 거부한다(결정 D24: B0의 척도는 학습과 같은 조건에서 잰 것이어야 한다)."""
    load_scales(path)
    path = Path(path)
    doc = json.loads((path / SCALES_NAME if path.is_dir() else path).read_text(encoding='utf-8'))
    return {name: doc.get(name, default) for name, default in CONDITION_DEFAULTS.items()}


def main(argv=None) -> int:
    """The command line (see the module text): the B0 scales of one S3 run into ``--out``. Returns 0, or 2 for a refusal (a model server runs: E6; a run or a file that is not
    what it should be; a file that is there already). It prints the scales and the number of rows, never a match key."""
    ap = argparse.ArgumentParser(prog='python -m c_ept.ept.scales', description='The B0 scales of a full S3 run (once, on the FIT rows).')
    ap.add_argument('--s3', type=Path, required=True, help='the S3 run folder (or its name under --runs-root); read only')
    ap.add_argument('--out', type=Path, required=True, help='the folder that gets B0_SCALES.json')
    ap.add_argument('--runs-root', type=Path, default=C.RUNS_ROOT)
    a = ap.parse_args(argv)
    from c_ept.workflows import run_s1
    try:
        scale = b0_scales(a.s3, a.out, runs_root=a.runs_root, check=run_s1.require_no_strata)         # E6: the check is the first call of ``b0_scales``, before anything is read
    except run_s1.StrataRunning as e:
        print(f'{e} (No scale was computed.)', file=sys.stderr, flush=True)
        return 2
    except (ValueError, FileExistsError, FileNotFoundError) as e:
        print(str(e), file=sys.stderr, flush=True)
        return 2
    doc = json.loads((a.out / SCALES_NAME).read_text(encoding='utf-8'))
    for key, value in scale.items():
        print(f'{key:16s} {value:12.6f}  {doc["n_rows"][key]:9d} rows')
    print(f'{SCALES_NAME}: {a.out / SCALES_NAME}  scales_sha256 {doc["scales_sha256"]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
