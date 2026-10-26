r"""[모듈 개요 및 하네스 위치]
이 모듈은 EPT 신경망 학습 과정에서 에폭별 학습 및 검증 지표를 누적 집계하고,
불변 스냅샷 파일(epoch_metrics.json, epoch_metrics.csv)로 보존하는 진단 엔진이다 (S4a 과제 5; v6 스키마, S4b 준비 M4에서 v7 스키마, M6에서 v8 스키마 추가).

A. 하네스 및 파이프라인 내 위치:
   - 외부 의존성 없는 표준 라이브러리(stdlib) 전용 구현으로, 대용량 텐서/데이터 로딩 없이 가볍고 안전하게 작동한다.
   - 단일 진실 공급원(Single Source of Truth): 최신 완료된 `epoch_XXXX/head.json`에 의해
     파일 크기(bytes) 및 SHA-256 해시가 검증 결합된다.
   - 에폭 표기는 1-based(사용자 친화적), 에폭 인덱스는 0-based(코드 루프)로 명확히 분리 관리.

B. 지표 정의 및 평가 규약:
   1) fit_batch_mean_loss: 학습 모드(training mode)에서의 FIT 배치 손실 산술 평균 (학습 진행 진단용, 공식 평가 아님).
   2) stop_loss: 평가 모드(evaluation mode)에서 풀링된 점수 행/경기 가중/규모 층 손실 (판정 R-S4a-16).
      배치 평균의 평균이 아니며, 학습 손실과 직접 비교할 수 없는 정밀 조기 종료 판정 기준.
   3) lr_last_step: 해당 에폭의 마지막 옵티마이저 스텝에서 적용된 학습률.
   4) kappa_0 .. kappa_{K-1} (v7, 저자 결정 D6): 에폭 끝 모형의 공동 참여 감쇠 척도. 아래 D절.
   5) terrain_penalty (v8, 저자 결정 D8): 에폭의 걸음별 지형 매끄러움 벌점 R_Ω의 평균(가중 없음). 아래 E절.

C. 주요 함수 및 안전성 보장:
   - bounded_read: 최대 읽기 바이트(MAX_BYTES, 16MB) 제한을 두어 메모리 초과 공격 및 파손 파일 방지.
   - read_snapshot / write_snapshot: 불변 스냅샷 입출력 및 무결성 검증.

D. v7 스냅샷 (S4b 준비 M4, 저자 결정 D6, 검토 A2-F2):
   수식   (EM-1)  κ_{e,r} = exp(log κ_r)  (에폭 e의 마지막 최적화 걸음 뒤의 값; r = 0 .. K − 1)

     기호      | 뜻                                          | 단위 | 범위        | 코드 변수
     κ_{e,r}   | 에폭 e 끝의 r번째 감쇠 척도                 | s    | > 0, 유한   | 행의 kappa_r (loop['kappa'][e][r])
     log κ_r   | 학습하는 매개변수(감쇠 없음, 결정 D6)       | 로그 s | 실수      | encoder.bias.co.log_kappa
     K         | 감쇠 척도의 수                              | —    | ≥ 1         | identity['encoder']['kappas']

   해석: 결정 D6은 log κ에 가중치 감쇠를 걸지 않고(decay_groups='split'), 감쇠 척도가 에폭마다 어디로 가는지를 기록하라고 정했다. 척도는 초 단위 값이어서 로그가
   아니라 exp(log κ)로 적는다. 행은 K개의 열 kappa_0 .. kappa_{K−1}을 더 갖고, 머리 스키마는 v7, 스냅샷 문서 스키마는 epoch-metrics-v2이다. 공동 참여 편향이 없는 N2
   (``EPT_ARCHITECTURE``가 아닌 구조)는 척도가 없으므로 같은 열을 두되 값은 비운다(JSON null, CSV 빈 칸). 이 열과 v7은 decay_groups가 'all'이 아닐 때에만 생기고,
   'all'(기본값)의 머리·행·CSV는 이전 실행의 v6와 바이트까지 같다. 머리 스키마는 실행의 정책에서 정해지며(``head_schema``) 둘이 맞지 않는 머리는 거부한다.

E. v8 스냅샷 (S4b 준비 M6, 저자 결정 D8, 검토 A2-F5):
   수식   (EM-2)  R̄_e = (1 / n_e) Σ_{t ∈ 에폭 e의 걸음} R_Ω(θ_t)   (R_Ω: ENC-7, 가중 없는 벌점; 손실 L = Loss_EPT + λ_Ω R_Ω는 LOSS-5)

     기호      | 뜻                                                | 단위     | 범위     | 코드 변수
     R̄_e       | 에폭 e의 걸음별 벌점 평균                         | 표의 값² | ≥ 0      | 행의 terrain_penalty (loop['terrain_penalty'][e])
     R_Ω(θ_t)  | 걸음 t의 손실을 계산할 때의 지형장 벌점 (갱신 전)  | 표의 값² | ≥ 0      | components['terrain_penalty']의 걸음별 값
     n_e       | 에폭 e의 최적화 걸음 수                           | 걸음     | ≥ 1      | loop['steps'][e]
     λ_Ω       | 지형 매끄러움 가중 (정체성의 terrain_smooth_weight) | —        | > 0      | identity['terrain_smooth_weight']

   해석: 결정 D8은 학습 손실에 λ_Ω R_Ω를 더하고 R_Ω를 에폭마다 기록하라고 정했다. 기록하는 값은 가중을 곱하지 않은 R_Ω의 에폭 평균이어서, 다른 λ_Ω로 학습한 실행을 같은 눈금으로 읽는다.
   학습 지형장이 아직 0인 첫 걸음들의 값은 0이므로 0은 유효한 기록이다(음수와 비유한은 아니다). v8은 정체성에 terrain_smooth_weight가 있을 때에만, 곧 가중이 0이 아닐 때에만 생기며,
   머리 스키마 v8·스냅샷 문서 스키마 epoch-metrics-v3이다. 가중이 0(기본값)인 실행의 머리·행·CSV는 이전 실행의 v6(또는 'split'이면 v7)와 바이트까지 같다. v7과 겹치는 실행('split'이면서 가중이 있음)은
   v8이고 행에 kappa_r 열이 먼저, 그 뒤에 terrain_penalty가 온다. STOP 값(stop_loss)에는 벌점이 들어가지 않는다.

F. 예산 종료 (S4b 준비 M8, 저자 결정 D10 P7):
   수식   (EM-3)  B = 1[ 머리에 budget_end 키가 있음 ] ;  B = 1 이면  budget_end = True ∧ identity.on_wall_cap = 'best_stop' ∧ ¬조기 종료 ∧ n < H ∧ 머리의 시간 합 ≥ 상한 ;  done = 조기 종료 ∨ n = H ∨ B
          termination_reason = 'early_stopping' (조기 종료) ▷ 'budget' (B) ▷ 'max_epochs' (n = H) ▷ 'in_progress'   (앞의 것이 우선; B = 1이면 n < H여서 'max_epochs'와 겹치지 않는다)

     기호          | 뜻                                                      | 단위 | 범위       | 코드 변수
     B             | 이 체크포인트가 벽시계 상한에서 끝난 실행의 것인가      | —    | {0, 1}     | budget
     n, H          | 끝난 에폭 수, 일정의 에폭 수(코사인이 끝나는 곳)        | 에폭 | 1 ≤ n ≤ H  | head['epoch'], head['epochs']
     on_wall_cap   | 상한에 닿았을 때의 동작 설정 (기본값 'fail')            | —    | fail, best_stop | identity['on_wall_cap']

   해석: 벽시계 상한에 닿은 실행은 그 에폭을 끝까지 마치고(체크포인트 단위가 에폭이어서 에폭 한가운데는 이어할 수 없다) 최선 STOP 체크포인트를 그대로 둔 채 끝난다. 이 끝은 조기 종료도 일정 끝도 아닌
   '남은 규칙이 없는 끝'이므로 머리의 키 하나(budget_end, 이 경우에만 있다)로 적고, 다른 규칙이 먼저 닿았으면(조기 종료, n = H) 그 규칙의 이유를 쓴다. 'fail'(기본값)은 상한에서 끝나지 않고 실패하므로 이 키가
   있는 머리는 정체성에 'best_stop'이 있고 머리의 시간 합(head['seconds'])이 머리가 적은 상한(caps.wall_seconds) 이상이어야 하며, 없거나 거짓이거나 다른 규칙과 겹치면 거부한다. 이 키가 없는 머리의 스냅샷은 이전 실행과 바이트까지 같고 머리 스키마(v6, v7, v8)는 바뀌지 않는다.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from pathlib import Path

SCHEMA = 'epoch-metrics-v1'
SCHEMA_V2 = 'epoch-metrics-v2'                   # the snapshot of a v7 head: the rows have the kappa columns
SCHEMA_V3 = 'epoch-metrics-v3'                   # the snapshot of a v8 head: the rows have the column terrain_penalty (after the kappa columns when the run has them)
HEAD_SCHEMA = 'ept_train_checkpoint_v6'
HEAD_SCHEMA_V7 = 'ept_train_checkpoint_v7'       # the head of a run under decay_groups='split' (S4b preparation M4, decision D6)
HEAD_SCHEMA_V8 = 'ept_train_checkpoint_v8'       # the head of a run with a terrain smoothness weight other than 0 (S4b preparation M6, decision D8); it also carries the kappa columns under 'split'
HEAD_SCHEMAS = (HEAD_SCHEMA, HEAD_SCHEMA_V7, HEAD_SCHEMA_V8)
FILES = ('epoch_metrics.json', 'epoch_metrics.csv')
MAX_BYTES = 16 * 1024 * 1024
COLUMNS = ('epoch', 'epoch_index', 'fit_batch_mean_loss', 'stop_loss', 'lr_last_step')
KAPPA_COLUMN = 'kappa_{}'
TERRAIN_COLUMN = 'terrain_penalty'
EPT_ARCHITECTURE = 'ept_s_v1'                    # the only architecture of ``c_ept.comparison.models.ARCHITECTURES`` with decay scales (a test pins the name); no import: stdlib only
DEFINITIONS = {
    'fit_batch_mean_loss': 'arithmetic mean of FIT batch losses in training mode; diagnostic, not evaluation',
    'stop_loss': 'existing evaluate_stop pooled scored-row/match/size-stratum loss (R-S4a-16), evaluation mode; not a mean of batch means',
    'lr_last_step': 'learning rate used by the last optimizer step of this epoch',
    'epoch': '1-based completed epoch label',
    'epoch_index': '0-based loop index; best_epoch_index also 0-based',
}
DEFINITIONS_V2 = {
    **DEFINITIONS,
    'kappa_i': 'decay scale i of the co-participation bias, exp(log_kappa[i]) in seconds, after the last optimizer step of this epoch (decision D6); null (empty in the csv) for N2, which has none',
}
DEFINITIONS_V3 = {
    **DEFINITIONS,
    'terrain_penalty': 'mean over the optimizer steps of this epoch of the unweighted smoothness penalty R_Omega of the learned terrain field (ENC-7), as computed for the loss L = Loss_EPT + lambda_Omega * R_Omega '
                       '(decision D8); not in the STOP value, and not in fit_batch_mean_loss',
}
DEFINITIONS_V3_SPLIT = {**DEFINITIONS_V2, 'terrain_penalty': DEFINITIONS_V3['terrain_penalty']}       # v8 under decay_groups='split': the kappa columns too
MONITOR = {'name': 'STOP', 'mode': 'min', 'improvement': 'strict', 'min_delta': 0.0}


def _require(ok):
    """조건이 참이 아닐 경우 체크포인트 에러를 발생시킨다."""
    if not ok:
        raise RuntimeError('checkpoint epoch metrics missing or invalid; not guessing an older checkpoint')


def _integer(value, low=0):
    """값이 특정 하한선 이상의 정수형인지 검증한다."""
    return type(value) is int and value >= low


def _number(value):
    """값이 유한한 실수 또는 정수인지 검증한다."""
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def head_schema(identity):
    """The schema of the head of a run with this ``identity``: v8 when the run has a terrain smoothness weight (the key ``terrain_smooth_weight``, which the identity holds only when the weight is not 0),
    else v7 when it groups its parameters by a policy other than 'all' (``decay_groups``), else v6.

    해석: 키가 없으면 'all'이고 가중 0이다(이전 실행의 정체성에는 키가 없다). 머리의 스키마가 정책과 가중에서 나오므로 스냅샷 열의 유무와 정체성이 서로 어긋날 수 없다 (``validate``가 확인한다).
    v8은 v7을 대신하지 않고 겹친다: 'split'이면서 가중이 있는 실행은 v8이고 그 행에 감쇠 척도 열도 있다(열 순서: 기본 열, kappa_r, terrain_penalty)."""
    if 'terrain_smooth_weight' in identity:
        return HEAD_SCHEMA_V8
    return HEAD_SCHEMA_V7 if identity.get('decay_groups', 'all') != 'all' else HEAD_SCHEMA


def _layout(head):
    """(split, smooth) of the head: whether its rows have the kappa columns (v7, or v8 of a run under 'split') and the column terrain_penalty (v8)."""
    smooth = head['schema'] == HEAD_SCHEMA_V8
    return head['schema'] == HEAD_SCHEMA_V7 or (smooth and head['identity'].get('decay_groups', 'all') != 'all'), smooth


def _format(head):
    """(snapshot schema, columns, definitions) of the snapshot that ``head`` binds: v6 for a v5/v6 head, v7 with one kappa column for each decay scale of the identity, v8 with the column
    terrain_penalty after them."""
    split, smooth = _layout(head)
    if not (split or smooth):
        return SCHEMA, COLUMNS, DEFINITIONS
    names = COLUMNS
    if split:
        scales = head['identity']['encoder']['kappas']
        _require(_integer(scales, 1))
        names += tuple(KAPPA_COLUMN.format(i) for i in range(scales))
    if smooth:
        return SCHEMA_V3, names + (TERRAIN_COLUMN,), DEFINITIONS_V3_SPLIT if split else DEFINITIONS_V3
    return SCHEMA_V2, names, DEFINITIONS_V2


def snapshot_schema(head):
    """The schema name of the snapshot that ``head`` binds (``SCHEMA`` for a v5/v6 head, ``SCHEMA_V2`` for v7): the name that ``head['epoch_metrics']['schema']`` holds."""
    try:
        return _format(head)[0]
    except (KeyError, TypeError):
        _require(False)


def columns(head):
    """The columns of the rows of the snapshot that ``head`` binds (the csv header): ``COLUMNS``, and for a v7 head the kappa columns after them."""
    try:
        return _format(head)[1]
    except (KeyError, TypeError):
        _require(False)


def _has_scales(identity):
    """True when the model of the run has decay scales (the EPT), False for N2: a blank kappa is then the rule, a number is refused."""
    spec = identity.get('model_spec')
    return spec is None or spec.get('architecture') == EPT_ARCHITECTURE


def json_bytes(doc):
    """메타데이터 딕셔너리를 정렬된 JSON 포맷 UTF-8 바이트로 직렬화한다."""
    return (json.dumps(doc, sort_keys=True, indent=1, allow_nan=False) + '\n').encode('utf-8')


def bounded_read(path):
    """최대 메모리 버짓(MAX_BYTES) 내에서 파일 바이트를 안전하게 읽는다."""
    try:
        with Path(path).open('rb') as handle:
            raw = handle.read(MAX_BYTES + 1)
        _require(len(raw) <= MAX_BYTES)
        return raw
    except OSError as err:
        raise RuntimeError('checkpoint metadata cannot be read; not guessing an older checkpoint') from err


def csv_bytes(rows, columns=COLUMNS):
    """행 리스트를 규격 CSV 포맷 UTF-8 바이트로 직렬화한다 (``columns``: v7 머리는 ``columns(head)``; 값이 None인 칸은 빈 칸)."""
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8')


def validate(doc, head):
    """모델 페이로드 없이 스칼라 지표로부터 조기 종료 및 최적 에폭 메타데이터를 검증한다."""
    try:
        n, horizon, seed, identity = head['epoch'], head['epochs'], head['seed'], head['identity']
        patience = identity['patience']
        schema, names, definitions = _format(head)
        expected = head_schema(identity)
        _require(head['schema'] == expected or (expected == HEAD_SCHEMA and head['schema'] not in HEAD_SCHEMAS))      # the head is the one of the policy and the weight of the run (v8 iff a weight, else v7 iff not 'all'); a v5 head (a run resumed from the old format) has the plain rows of v6
        if 'terrain_smooth_weight' in identity:
            _require(_number(identity['terrain_smooth_weight']) and identity['terrain_smooth_weight'] > 0)      # the key is there only for a weight other than 0
        _require(_integer(n, 1) and n <= 9999 and _integer(horizon, 1) and n <= horizon)
        _require(_integer(seed) and _integer(patience, 1))
        _require(type(identity['epochs']) is int and identity['epochs'] == horizon)
        _require(type(identity['seed']) is int and identity['seed'] == seed)
        _require(doc['schema'] == schema and doc['identity'] == identity and doc['seed'] == seed)
        _require(doc['epoch_count'] == n and type(doc['epoch_count']) is int)
        _require(doc['definitions'] == definitions and doc['monitor'] == MONITOR and doc['patience'] == patience)
        rows = doc['rows']
        _require(isinstance(rows, list) and len(rows) == n)
        scales = _has_scales(identity)
        best, best_index, bad = math.inf, -1, 0
        for index, row in enumerate(rows):
            _require(isinstance(row, dict) and set(row) == set(names))
            _require(type(row['epoch']) is int and row['epoch'] == index + 1)
            _require(type(row['epoch_index']) is int and row['epoch_index'] == index)
            _require(all(_number(row[key]) for key in COLUMNS[2:]) and row['lr_last_step'] >= 0)
            for key in names[len(COLUMNS):]:
                if key == TERRAIN_COLUMN:                                          # the penalty of a v8 row: a finite number that is not negative (R_Omega >= 0; 0 while the field is still 0), for every model
                    _require(_number(row[key]) and row[key] >= 0)
                else:                                                              # the kappa columns of a v7 row: positive finite numbers, or all blank for N2
                    _require(_number(row[key]) and row[key] > 0 if scales else row[key] is None)
            if row['stop_loss'] < best:
                best, best_index, bad = row['stop_loss'], index, 0
            else:
                bad += 1
            _require(index == n - 1 or bad < patience)
        early = bad >= patience
        budget = 'budget_end' in head                                           # the run ended at the wall cap (S4b preparation M8, decision D10 P7); no such key in the head of any other run
        if budget:                                                              # (EM-3): the budget end is the end that no other rule reached, and only a run that was asked to end there can have it
            _require(head['budget_end'] is True and identity.get('on_wall_cap') == 'best_stop' and not early and n < horizon)
            _require(_number(head['seconds']) and _number(head['caps']['wall_seconds']) and head['seconds'] >= head['caps']['wall_seconds'])      # the cap was reached when the epoch ended: the head is written after the read of the seconds
        done = early or n == horizon or budget
        reason = 'early_stopping' if early else 'budget' if budget else 'max_epochs' if n == horizon else 'in_progress'
        for key, value in {'best_epoch_index': best_index, 'best_epoch': best_index + 1,
                           'bad_epochs': bad, 'done': done, 'stopped_early': early,
                           'termination_reason': reason}.items():
            _require(type(doc[key]) is type(value) and doc[key] == value)
        _require(type(head['best_epoch']) is int and head['best_epoch'] == best_index)
        _require(type(head['done']) is bool and head['done'] == done)
        _require(type(head['stopped_early']) is bool and head['stopped_early'] == early)
        _require(head['best']['folder'] == f'epoch_{best_index + 1:04d}')
    except (KeyError, TypeError, ValueError, OverflowError):
        _require(False)
    return doc


def snapshot(loop, head):
    """루프 상태나 난수 스트림을 변경하지 않고 누적된 스칼라 지표 딕셔너리를 추출한다."""
    try:
        n = head['epoch']
        schema, names, definitions = _format(head)
        split, smooth = _layout(head)
        _require(type(loop['epoch']) is int and loop['epoch'] == n)
        _require(len(loop['fit_batch_mean_losses']) == len(loop['stop_losses']) == len(loop['steps']) == n)
        _require(all(_integer(step, 1) for step in loop['steps']))
        _require(isinstance(head['timing']['steps'], list) and len(head['timing']['steps']) == n)
        _require(all(_integer(step, 1) for step in head['timing']['steps']))
        _require(loop['steps'] == head['timing']['steps'])
        _require(sum(loop['steps']) == len(loop['lrs']))
        _require(all(_number(lr) and lr >= 0 for lr in loop['lrs']))
        _require(_number(loop['best_value']) and loop['best_value'] == min(loop['stop_losses']))
        if split:
            _require(isinstance(loop['kappa'], list) and len(loop['kappa']) == n)       # one record of kappa for every epoch (a list of K numbers, or None for N2)
        if smooth:
            _require(isinstance(loop['terrain_penalty'], list) and len(loop['terrain_penalty']) == n)      # one mean R_Omega for every epoch
        kappa_names = names[len(COLUMNS):len(names) - smooth]
        rows, offset = [], 0
        for index, count in enumerate(loop['steps']):
            offset += count
            row = dict(zip(COLUMNS, (index + 1, index, loop['fit_batch_mean_losses'][index],
                                     loop['stop_losses'][index], loop['lrs'][offset - 1])))
            if split:
                record = loop['kappa'][index]
                _require(record is None or (isinstance(record, list) and len(record) == len(kappa_names)))
                row.update(zip(kappa_names, [None] * len(kappa_names) if record is None else record))
            if smooth:
                row[TERRAIN_COLUMN] = loop['terrain_penalty'][index]
            rows.append(row)
        early, done, budget = loop['stopped_early'], head['done'], 'budget_end' in head      # validate checks that the key means what it says (EM-3)
        doc = {'schema': schema, 'identity': head['identity'], 'seed': head['seed'], 'epoch_count': n,
               'definitions': definitions, 'monitor': MONITOR, 'patience': head['identity']['patience'],
               'best_epoch_index': loop['best_epoch'], 'best_epoch': loop['best_epoch'] + 1,
               'bad_epochs': loop['bad'], 'done': done, 'stopped_early': early,
               'termination_reason': 'early_stopping' if early else 'budget' if budget else 'max_epochs' if done else 'in_progress', 'rows': rows}
        return validate(doc, head)
    except (KeyError, TypeError, IndexError):
        _require(False)


def encode(loop, head):
    """스냅샷 결과를 JSON 및 CSV 바이트 딕셔너리로 인코딩하여 반환한다."""
    doc = snapshot(loop, head)
    return dict(zip(FILES, (json_bytes(doc), csv_bytes(doc['rows'], columns(head)))))


def read_snapshot(folder, head):
    """고정된 JSON/CSV 스냅샷 파일들을 읽어 크기 및 SHA-256 해시를 검증하고 반환한다."""
    try:
        binding = head['epoch_metrics']
        _require(head['schema'] in HEAD_SCHEMAS and binding['schema'] == _format(head)[0])
        _require(set(binding['files']) == set(FILES))
        raw = {}
        for name in FILES:
            entry = binding['files'][name]
            _require(set(entry) == {'bytes', 'sha256'} and _integer(entry['bytes'], 1) and entry['bytes'] <= MAX_BYTES)
            _require(isinstance(entry['sha256'], str) and re.fullmatch('[0-9a-f]{64}', entry['sha256']) is not None)
            raw[name] = bounded_read(Path(folder) / name)
            _require(len(raw[name]) == entry['bytes'] and hashlib.sha256(raw[name]).hexdigest() == entry['sha256'])
        doc = validate(json.loads(raw[FILES[0]]), head)
        _require(raw[FILES[1]] == csv_bytes(doc['rows'], columns(head)))
        return doc
    except (KeyError, TypeError, ValueError, OverflowError):
        _require(False)
