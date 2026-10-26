"""S4a 비용 점검 실행기: 자료·모형 정의는 기존 모듈을 재사용한다.

킬이 하나 이상 난 교전의 EPT 비용을 FIT/STOP 대리 과제와 DEV 모의에서 잰다.
새 프로세스의 native 정책은 heavy import 전에 적용하며 실제 작업은 E6와
깨끗한 전체 저장소, 입력 정체성, 실행별 OS 잠금 뒤에만 진행한다.
Status action은 c_ept.s4a_operator가 제공한다. 이 실행기 CLI의 action이 아니다.
상태 표지의 pending만으로 프로세스가 살아 있다고 단정하지 않는다.

(COST-1) 학습호출 예상 = 반복 수 · (잔여 가정 + 에폭 비용 · 행 비 · 에폭 수 + 저장 비용 · 에폭 수).
  기호 | 뜻 | 단위 | 범위 | 코드 변수
  학습호출 예상 | 학습 호출만의 조건부 시간 | s | ≥0 | project_cost 반환값
  반복 수,에폭 수 | 명시한 실행 수와 학습 길이 | 개 | 양수 | fits,epochs
  잔여 가정 | 전체 호출에서 분리 식별하지 못한 잔여의 배분 가정 | s | ≥0 | fixed_seconds
  에폭 비용,저장 비용 | pre-checkpoint span과 저장 IO의 에폭당 평균 | s | ≥0 | epoch_seconds,checkpoint_io_seconds
  행 비 | 전체/대리 입력의 행 수 비 | — | >0 | row_ratio
(COST-2) 단계 시간 = 끝 host 시각 − 시작 host 시각; peak는 표본 최댓값.

외삽은 가정이지 실험 결과가 아니며 회복 박동 청구는 정확한 성능 측정이 아니다.
체크포인트의 마지막 쓰기 시간은 같은 쓰기 전 기록에 넣을 수 없으므로 현재
호출 반환값에 따로 담고 과거 중단 시도의 그 성분은 미측정으로 남긴다.
계획 Q1..Q12, Task7, R-S4a-16..20, 승인된 operator API1을 따른다."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

from c_ept import __path__ as _package_path
from c_ept.workflows.s4a_runtime import EXCEPTION_TYPE_PATTERN, RuntimeRefused, prepare_native_runtime, validate_e6_probe

OPERATOR_API_VERSION = 1
RUN_SCHEMA = 's4a_run_v1'
COST_SCHEMA = 's4a_cost_v1'
PROCESS_SCHEMA = 's4a_process_v1'
MAX_JSON_BYTES = 4 * 1024 * 1024
REPO = Path(_package_path[0]).resolve().parent
MEMORY_CAP = 40 * 2**30
APPROVED_RUNS_ROOT = Path(r'D:\LOL_Data\runs')
SEED = 20261022
PROJECTION_EPOCHS = {'low': 10, 'mid': 30, 'high': 100}


def _digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise RuntimeRefused('duplicate JSON field')
            result[key] = value
        return result
    def number(value):
        result = float(value)
        if not math.isfinite(result):
            raise RuntimeRefused('nonfinite JSON number')
        return result
    def invalid(value):
        raise RuntimeRefused('nonfinite JSON constant')
    path = Path(path)
    if path.stat().st_size > MAX_JSON_BYTES:
        raise RuntimeRefused('metadata exceeds 4 MiB bound')
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=pairs, parse_float=number, parse_constant=invalid)


def _write(path, obj):
    from c_ept.data.store_core import write_atomic
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(path, json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + '\n')


def repository_state(repo=REPO):
    """Read-only whole-repository clean identity; ignored local configs are not dirty.

    해석: S3의 c_ept 한정 정의는 바꾸지 않고 S4의 실행 바깥에서 전체 저장소를 확인한다.
    Git 확인 불가는 깨끗함으로 해석하지 않는다.
    """
    env = dict(os.environ, GIT_OPTIONAL_LOCKS='0')
    def git(*args):
        proc = subprocess.run(['git', '-C', str(repo), *args], env=env, capture_output=True, text=True, timeout=30)
        if proc.returncode:
            raise RuntimeRefused('whole-repository Git identity unavailable')
        return proc.stdout.strip()
    head = git('rev-parse', 'HEAD')
    top = Path(git('rev-parse', '--show-toplevel')).resolve()
    if top != Path(repo).resolve() or not re.fullmatch('[0-9a-f]{40}', head):
        raise RuntimeRefused('Git repository/HEAD identity unavailable')
    if git('status', '--porcelain', '--untracked-files=all'):
        raise RuntimeRefused('S4a requires a clean whole-repository checkout, including docs/wrapper/environment files')
    return {'code_commit': head, 'code_dirty': False, 'repo_root': str(top)}


@contextmanager
def run_lock(run):
    """Own byte0 of a persistent S4a.lock; handle close/crash releases, no deletion.

    해석: 새 디렉터리 예약은 가능하지만 상태 쓰기는 이 잠금 뒤에만 한다.
    잠금 파일의 존재만으로 실행 중이라고 판단하지 않는다.
    """
    path = Path(run) / 'S4a.lock'
    with path.open('a+b') as f:
        f.seek(0, 2)
        if f.tell() == 0:
            f.write(b'0'); f.flush()
        f.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as err:
            raise RuntimeRefused('another process owns this S4a run') from err
        try:
            yield
        finally:
            f.seek(0)
            if os.name == 'nt':
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def project_cost(epoch_seconds, fixed_seconds, row_ratio, epochs, fits, *, checkpoint_io_seconds=0.):
    """Conditional scenario seconds, not an observed full-run result.

    수식
    ----
    (COST-1) 학습호출 예상 = fits · (fixed_seconds + epoch_seconds · row_ratio · epochs + checkpoint_io_seconds · epochs).
      기호 | 뜻 | 단위 | 범위 | 코드 변수
      학습호출 예상 | 학습 호출만의 조건부 시간 | s | ≥0 | 반환값
      잔여 가정 | 분리 식별하지 못한 잔여의 배분 가정 | s | ≥0 | fixed_seconds
      에폭 비용,저장 비용 | pre-checkpoint 및 저장 IO의 에폭당 평균 | s | ≥0 | epoch_seconds,checkpoint_io_seconds
      행 비 | 목표/대리 행 수 | — | >0 | row_ratio
      반복 수,에폭 수 | 명시한 시나리오 | 개 | 양수 | fits,epochs

    해석
    ----
    pre-checkpoint 성분만 행에 비례시키고 저장 IO는 에폭 수에만 비례시키며 잔여는 명시한 가정으로 배분한다.
    학습 곡선과 장치 병목이 같다는 가정이 깨지면 외삽도 달라진다.
    회복 추정 총시간을 이 식의 정확한 측정치로 넣지 않는다.

    근거
    ----
    계획 Q10, Task7.
    """
    values = (epoch_seconds, fixed_seconds, row_ratio, epochs, fits, checkpoint_io_seconds)
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in values) or min(epoch_seconds, fixed_seconds, checkpoint_io_seconds) < 0 or min(row_ratio, epochs, fits) <= 0:
        raise RuntimeRefused('invalid projection assumptions')
    return fits * (fixed_seconds + epoch_seconds * row_ratio * epochs + checkpoint_io_seconds * epochs)


def verify_resume(manifest, identity):
    """Refuse schema/config/args/code/runtime/input drift before any resume writes."""
    from c_ept.workflows.run_s3 import check_resume_code
    if manifest.get('schema') != RUN_SCHEMA or manifest.get('stage') != 's4a' or manifest.get('operator_api_version') != OPERATOR_API_VERSION:
        raise RuntimeRefused('unsupported S4a RUN schema/API')
    check_resume_code(manifest, identity['code'])
    if manifest.get('identity') != identity or manifest.get('run_identity_sha256') != _digest(identity) or manifest.get('operator_config_sha256') != identity['operator_config_sha256'] or manifest.get('code_commit') != identity['code']['code_commit']:
        raise RuntimeRefused('resume effective args/code/runtime/input/operator config identity differs')


def _process_record(run, manifest, state, outcome=None, exit_code=None, exception_type=None):
    """The PROCESS.json record. ``exception_type`` (the class name of the exception that ended a failed run) is an additive field: it is
    written only when given and only when the name matches ``s4a_runtime.EXCEPTION_TYPE_PATTERN``, the form that the PROCESS validators accept.
    So a running or finished record keeps exactly its earlier field set, and a class name of another form (non-ASCII, dotted, longer than 128
    characters) leaves the field out instead of turning a recorded failure into a metadata refusal.

    해석: 같은 종료 코드(2)가 정책 거부와 계산 오류에 모두 쓰이므로, 실패한 실행의 예외 클래스 이름을 따로 남겨 기록만으로 둘을 가른다.
    실행 정체성에는 들어가지 않는 기록 필드이다(이어하기 판정은 바뀌지 않는다). 이름이 검사기가 받는 형식(EXCEPTION_TYPE_PATTERN)이 아니면
    필드만 빼고 실패 기록은 그대로 남긴다. 이름 하나를 못 적는 편이, 실패 기록 전체를 읽지 못하게 되는 것보다 낫다.
    """
    import psutil
    record = {'schema': PROCESS_SCHEMA, 'operator_api_version': OPERATOR_API_VERSION,
              'run_path': str(Path(run).resolve()), 'run_identity_sha256': manifest['run_identity_sha256'],
              'code_commit': manifest['code_commit'], 'operator_config_sha256': manifest['identity']['operator_config_sha256'],
              'pid': os.getpid(), 'create_time': psutil.Process().create_time(), 'state': state,
              'outcome': outcome, 'exit_code': exit_code, 'updated_at': dt.datetime.now(dt.timezone.utc).isoformat()}
    if exception_type is not None and re.fullmatch(EXCEPTION_TYPE_PATTERN, exception_type):
        record['exception_type'] = exception_type
    return record


def _store_dir(path, kind):
    path = Path(path).resolve()
    return path / kind if (path / kind / 'STORE.json').is_file() else path


def _store_metadata(path):
    from c_ept import runlog
    doc = _read(Path(path) / 'STORE.json')
    if doc.get('complete') is not True or not isinstance(doc.get('files'),dict):
        raise RuntimeRefused('store metadata is not complete or lacks file identity')
    for rel in doc['files']:
        if Path(rel).is_absolute() or '..' in Path(rel).parts or ':' in rel:
            raise RuntimeRefused('unsafe store manifest path')
    return doc, runlog.sha256_json(doc['files'])


def _model_hashes(folder, json_name, npz_name, outputs, rel):
    meta_path, arrays_path = folder / json_name, folder / npz_name
    meta_sha = _sha(meta_path)
    if meta_sha != outputs.get(f'{rel}/{json_name}'):
        raise RuntimeRefused('model metadata differs from S3 fit ledger')
    doc = _read(meta_path)
    array_sha = _sha(arrays_path)
    expected = outputs.get(f'{rel}/{npz_name}', doc.get('npz_sha256'))
    if array_sha != expected or array_sha != doc.get('npz_sha256'):
        raise RuntimeRefused('model arrays differ from ledger-bound model metadata')
    return {f'{rel}/{json_name}': meta_sha, f'{rel}/{npz_name}': array_sha}


def _require_build_terrain(s2_doc, terrain_sha):
    """Refuse unless the terrain folder next to the S2 store is the terrain that built the store's views.

    ``s2_doc`` is the parsed STORE.json of the S2 store; ``terrain_sha`` maps TERRAIN.json, terrain.npz and geodesic.npy to the SHA-256 of
    the file in the folder now. The build (``store_build.build_s2``) records ``stages.views.terrain_sha256`` in that STORE.json as
    ``runlog.sha256_json`` of the same three file hashes, and this check uses the same function on the folder as it is now.

    해석: S2 보기는 빌드 때 쓴 지형(보행 격자, 측지 거리표)으로 만든 값이다. 학습이 읽는 정적 채널과 측지 편향은 지금 폴더의 지형에서 나오므로,
    폴더가 다시 만들어졌거나 바뀌었다면 보기와 다른 지형으로 학습하게 된다. 빌드 기록이 없거나 형식이 틀려도 같은 이유로 거부하며
    기록이 없는 저장소를 통과시키지 않는다(추측하지 않는다). 수치식이 아니므로 새 식 번호를 만들지 않는다.
    근거: EPT-T010 영역 5 발견 A5-F10, 설계서 5.3절·9절, R-S4a-18.
    """
    from c_ept import runlog
    views = s2_doc.get('stages', {}).get('views') if isinstance(s2_doc.get('stages'), dict) else None
    built = views.get('terrain_sha256') if isinstance(views, dict) else None
    if built is None:
        raise RuntimeRefused('S2 store records no terrain hash (STORE.json stages.views.terrain_sha256), so the terrain folder cannot be checked against its build')
    if not isinstance(built, str) or not re.fullmatch('[0-9a-f]{64}', built):
        raise RuntimeRefused('S2 store records no valid terrain hash (STORE.json stages.views.terrain_sha256 is not 64 lower-case hex digits), so the terrain folder cannot be checked against its build')
    if runlog.sha256_json(terrain_sha) != built:
        raise RuntimeRefused('terrain folder differs from the terrain that built the S2 views (terrain_sha256 in the S2 STORE.json)')


def inspect_inputs(args):
    """Verify requested metadata and every model-file hash; do not verify raw payload stores."""
    from c_ept.workflows import run_s3
    s3 = Path(args.s3).resolve()
    if not s3.is_dir():
        s3 = Path(args.runs_root) / args.s3
    doc = _read(s3 / 'RUN.json')
    if doc.get('stage') != 's3':
        raise RuntimeRefused('requested --s3 is not an S3 run')
    ledger = run_s3.Ledger(s3)
    if not ledger.complete('fit') or not ledger.complete('labels'):
        raise RuntimeRefused('S3 fit/label stages incomplete')
    stores, producers, record, store_docs = {}, {}, {'s3_run': s3.name}, {}
    for name in ('s1', 's2'):
        supplied = Path(getattr(args, name))
        if not supplied.is_absolute() and not supplied.exists():
            supplied = Path(args.runs_root) / supplied
        store = _store_dir(supplied, name)
        expected_store = _store_dir(Path(args.runs_root) / doc[name]['run'], name)
        store_docs[name], sha = _store_metadata(store)
        if store.resolve() != expected_store.resolve() or sha != doc[name]['files_sha256']:
            raise RuntimeRefused('requested store differs from verified S3 input identity')
        stores[name] = str(store)
        producers[name+'_producer'] = str((Path(args.runs_root) / doc[name]['run']).resolve())
        record[name] = {k: doc[name][k] for k in ('run', 'files_sha256')}
    labels = s3 / 'labels/store'
    _, label_sha = _store_metadata(labels)
    if label_sha != ledger.stage('labels')['outputs'].get('labels/store/'):
        raise RuntimeRefused('derived labels differ from S3 label ledger')
    label_prov = _read(labels / 'LABELS.json')
    if label_prov.get('roles') != ['FIT', 'STOP', 'DEV'] or label_prov.get('s1_files_sha256') != record['s1']['files_sha256']:
        raise RuntimeRefused('derived label role/source provenance invalid')
    fit_outputs = ledger.stage('fit')['outputs']
    b0 = _model_hashes(s3 / 'fit/b0', 'b0_heads.json', 'b0_heads.npz', fit_outputs, 'fit/b0')
    tier = _model_hashes(s3 / 'fit/tier', 'tower_tier.json', 'tower_tier.npz', fit_outputs, 'fit/tier')
    record.update(labels_store_sha256=label_sha, model_files_sha256=b0)
    s3a = Path(args.runs_root) / doc['s3a']['run']
    price_sha = _sha(s3a / 'prices' / 'PRICES.json')
    if price_sha != doc['s3a']['prices_sha256'] or label_prov.get('prices_sha256') != price_sha:
        raise RuntimeRefused('price identity differs from S3/labels')
    rules_sha = _sha(s3a / 'rules' / 'RULE_PAYOUTS.json')
    if rules_sha != doc['s3a']['rules_sha256'] or label_prov.get('rules_sha256') != rules_sha:
        raise RuntimeRefused('rules identity differs from S3/labels')
    terrain_path = Path(stores['s2']).parent / 'terrain'
    terrain_sha = {name: _sha(terrain_path / name) for name in ('TERRAIN.json', 'terrain.npz', 'geodesic.npy')}
    _require_build_terrain(store_docs['s2'], terrain_sha)
    return {'paths': {**stores, **producers, 's3': str(s3), 's3a': str(s3a.resolve()), 'labels': str(labels), 'terrain': str(terrain_path), 'prices': str(s3a / 'prices' / 'PRICES.json'), 'rules': str(s3a / 'rules' / 'RULE_PAYOUTS.json')},
            'expected_provenance': record, 'tier_files_sha256': tier, 'terrain_files_sha256': terrain_sha,
            'source_json_sha256': {name: _sha(path) for name, path in (('S3RUN', s3/'RUN.json'), ('S3STAGES', s3/'STAGES.json'), ('S1STORE',Path(stores['s1'])/'STORE.json'), ('S2STORE',Path(stores['s2'])/'STORE.json'))},
            'prices_sha256': price_sha, 'rules_sha256': rules_sha, 'label_provenance_sha256': _sha(labels/'LABELS.json')}


def _output_paths(args, meta):
    """Keep production writes in approved runs root and outside immutable sources."""
    root = Path(args.runs_root).resolve()
    if root != APPROVED_RUNS_ROOT.resolve():
        raise RuntimeRefused('output runs root differs from approved D:/LOL_Data/runs')
    catalog = Path(args.catalog_db).resolve()
    protected = []
    for key, value in meta['paths'].items():
        path = Path(value).resolve()
        # Store leaves may have read-only siblings; retain explicit recorded producer
        # directories too, without protecting their ordinary shared runs parent.
        if key in ('prices','rules') or (key in ('s1','s2') and path.name.casefold() == key):
            path = path.parent
        protected.append(path)
    protected += [Path(path).resolve() for path in (r'D:\LOL_Project',r'D:\LOL_Training_20261004',r'D:\LOL_External_20261009')]
    for source in protected:
        resumed = getattr(args,'resume',None)
        if root.is_relative_to(source) or catalog.is_relative_to(source) or (resumed is not None and Path(resumed).resolve().is_relative_to(source)):
            raise RuntimeRefused('output/catalog overlaps read-only source or frozen tree')


def _parser():
    p = argparse.ArgumentParser(description='S4a cost runner; explicit manual execution only')
    for key in ('s1','s2','s3'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--runs-root', default=r'D:\LOL_Data\runs')
    p.add_argument('--catalog-db', default=r'D:\LOL_Data\catalog\runs.sqlite')
    p.add_argument('--sizes', nargs='+', choices=('S','M'), default=['S','M'])
    p.add_argument('--devices', nargs='+', choices=('cpu','cuda'), default=['cpu','cuda'])
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--share', type=float, default=.25)
    p.add_argument('--dev-sample', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--weight-decay', type=float, default=.01)
    p.add_argument('--warmup-steps', type=int, default=10)
    p.add_argument('--seed', type=int, default=SEED)
    p.add_argument('--resume')
    p.add_argument('--operator-config-sha256')
    return p


def _normal_args(args):
    out = vars(args).copy()
    out.pop('resume')
    digest = out.pop('operator_config_sha256')
    if digest is not None and not re.fullmatch('[0-9a-fA-F]{64}', digest):
        raise RuntimeRefused('operator-config-sha256 must be a SHA256')
    if args.epochs < 1 or args.dev_sample < 1 or args.batch_size < 1 or not 0 < args.share <= 1 or args.warmup_steps < 0 or args.seed < 0 or not all(math.isfinite(x) and x >= 0 for x in (args.lr,args.weight_decay)):
        raise RuntimeRefused('invalid S4a effective parameters')
    for name in ('s1','s2','s3','runs_root','catalog_db'):
        out[name] = str(Path(out[name]).resolve())
    if len(set(args.sizes)) != len(args.sizes) or len(set(args.devices)) != len(args.devices):
        raise RuntimeRefused('duplicate size/device selection')
    return out, None if digest is None else digest.lower()


def _runtime_fingerprint(policy):
    import torch
    return {**policy, 'numpy': str(sys.modules['numpy'].__version__), 'torch': str(torch.__version__), 'cuda': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version(), 'threads': torch.get_num_threads(),
            'interop_threads': torch.get_num_interop_threads(), 'dtype': str(torch.get_default_dtype()),
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}


def provenance_record():
    """Record-only provenance that the S4a and T002 manifests lacked. ``runlog.write_manifest`` keeps the decision-record hash (``spec_sha256``)
    and the GPU name and memory (``gpu``) in the RUN.json of S1-S3, but no driver; this record adds the hash and the GPU name and driver.

    Returns ``{'spec_sha256': ..., 'gpu_devices': ...}``. ``spec_sha256`` is what ``config.check_spec()`` returns, the SHA-256 of the decision
    record after the code constants were checked against it. ``gpu_devices`` is a list of ``{'name', 'driver_version'}``, one per line of the
    operator's nvidia-smi query (``s4a_operator.gpu_metadata``). A value that cannot be established is None, never a guess: an unreadable or
    mismatching decision record, no nvidia-smi or no GPU, a line that does not read ``name, driver``.

    해석: S1–S3 매니페스트는 결정 기록 해시(spec_sha256)와 GPU 이름·메모리(gpu)를 담지만 드라이버는 담지 않고, S4a·T002 매니페스트에는 앞의 둘이
    모두 없었다. 이 기록은 결정 기록 해시와 GPU 이름·드라이버를 더한다.
    기록 필드일 뿐이므로 실행 정체성(identity)에도, 이어하기 동등 검사에도 넣지 않는다. 그래서 이 값이 없거나 이어하기 때 다르게 나와도
    이어하기 판정은 바뀌지 않는다. 확인하지 못한 값은 추측하지 않고 null로 남기며, 결정 기록 불일치는 조용히 지나가지 않도록 stderr에 한 줄 남긴다. 수치식이 아니다.
    근거: EPT-T010 영역 5 발견 A5-F4, 설계서 9절(실행 기록), 지침 A6a(GPU 실행은 장치와 드라이버를 기록한다).
    """
    from c_ept import config as C
    # No failure of this probe may end a run (the T002 pilot has one attempt), so every error is caught and leaves the value None.
    try:
        spec = C.check_spec()
    except Exception as err:                                      # noqa: BLE001 - unreadable record, JSON error, SpecMismatch, ...
        print(f'Run provenance: the decision-record hash is not recorded ({type(err).__name__}: {err})', file=sys.stderr)
        spec = None
    try:
        from c_ept.workflows import s4a_operator
        devices = []
        for line in s4a_operator.gpu_metadata():
            name, comma, driver = line.rpartition(',')
            if not (comma and name.strip() and driver.strip()):
                raise ValueError('nvidia-smi line is not "name, driver"')
            devices.append({'name': name.strip(), 'driver_version': driver.strip()})
    except Exception:                                             # noqa: BLE001 - no nvidia-smi or GPU (OperatorError) or a garbled line
        devices = []
    return {'spec_sha256': spec, 'gpu_devices': devices or None}


def _e6(guard):
    try:
        guard()
    except RuntimeRefused:
        raise
    except Exception as err:
        raise RuntimeRefused('E6 unavailable or active: '+type(err).__name__) from err


_REFUSAL_ERRORS = (RuntimeRefused, ValueError, OSError)


def _failure_exit_code(error):
    """Recorded runner return outcome, not an independently observed OS exit."""
    return 130 if isinstance(error, KeyboardInterrupt) else 2 if isinstance(error, _REFUSAL_ERRORS) else 1


def _match_keys(s1, runs_root=None):
    """The match keys of the S1 store at ``s1`` as text, or None when they cannot be loaded; it never raises (a diagnostic must not fail).

    ``s1`` is a run folder or its ``s1`` store folder, as ``--s1`` takes it; a relative name that does not exist is read under ``runs_root``.
    해석: 오류 문구를 가리려고 S1 저장소의 경기 키 배열 하나만 읽는다. 읽지 못하면 None을 돌려주고 호출자가 그 사실을 문구에 밝힌다.
    """
    try:
        import numpy as np
        from c_ept.workflows import run_s1
        path = Path(s1)
        if runs_root is not None and not path.is_absolute() and not path.exists():
            path = Path(runs_root) / path
        store = run_s1.load_s1(path, verify=False)
        keys = [k.decode('ascii') for k in np.asarray(store.array('matches/key')).tolist() if k]
        del store
        return keys
    except Exception:                                             # noqa: BLE001 - no store to take the keys from
        return None


def failure_report(label, error, guard_types, *, s1=None, runs_root=None):
    """The text that a runner prints on stderr for one caught failure. The caller keeps the exit code and the refusal classification.

    A guard refusal (``error`` is one of ``guard_types``) keeps its one line ``<label>: <message>``, exactly as before. Any other exception
    prints that line, then its class name and its traceback, with every match key of the S1 store at ``s1`` replaced by <key> and the game
    id inside it by <id> (the rule of ``run_s3._scrub_text``, which ``run_s3.main`` applies to its own tracebacks). ``s1=None`` is for a
    metadata-only caller that loads no match data, so there is nothing to remove. When the keys cannot be loaded the text says so at its end.

    Which exceptions reach this function is each caller's choice, and it is unchanged. The S4a runner and the T002 runner catch the guard types
    plus ValueError and OSError (a NumPy ValueError is an example); these keep exit 2 and get the class name and the scrubbed traceback. Any other
    exception (a RuntimeError such as a CUDA out-of-memory error or ``train.ResourceBoundError``, a KeyError, a TypeError, an IndexError, an
    AttributeError, an AssertionError, a MemoryError, a ``subprocess.TimeoutExpired``) is not caught there: it ends with exit 1 and Python's own
    traceback, which is NOT scrubbed. The comparison operator also catches RuntimeError and KeyError (exit 2, scrubbed in a worker).

    해석: 정책 거부는 의도한 거부이므로 한 줄로 충분하다. 실행기가 잡는 그 밖의 예외(S4a·T002는 NumPy ValueError와 OSError, 비교 오퍼레이터는
    RuntimeError·KeyError까지)는 같은 종료 코드(2)를 받더라도 버그일 수 있어, 클래스 이름과 traceback을 남겨 다시 돌리지 않고도 원인을 볼 수 있게
    한다. traceback에는 경기 키가 들어갈 수 있으므로 run_s3.main과 같은 가림 규칙을 쓴다. 키를 읽지 못하면 가리지 못했다는 사실을 문구 끝에 적어,
    가리지 않은 문구를 가린 문구로 오해하지 않게 한다. S4a·T002에서 잡히지 않은 예외(CUDA 메모리 부족 같은 RuntimeError, KeyError, TypeError 등 무엇이든)는
    종료 코드 1과 파이썬 기본 traceback으로 끝나며 이 traceback은 가려지지 않는다(분류와 종료 코드는 이전 그대로이고, 바꾸려면 저자 결정이 필요하다).
    근거: EPT-T010 영역 5 발견 A5-F8 (예외 분류가 계산 오류의 traceback을 숨김). 수치식이 아니므로 새 식 번호를 만들지 않는다.
    """
    message = f'{label}: {error}'
    if isinstance(error, guard_types):
        return message
    detail = ''.join(traceback.format_exception(type(error), error, error.__traceback__))
    text = f'{message}\n{type(error).__name__} is not a guard refusal; the exit code is unchanged. Traceback:\n{detail}'
    if s1 is None:
        return text
    keys = _match_keys(s1, runs_root)
    if keys is not None:
        try:
            from c_ept.workflows import run_s3
            return run_s3._scrub_text(text, keys)
        except Exception:                                         # noqa: BLE001 - fall through to the unscrubbed notice
            pass
    return text + '(the S1 match keys could not be loaded, so none was removed from this text)\n'


def main(argv=None):
    """One foreground CLI path shared by direct -m and operator; returns exit status."""
    args = None
    try:
        policy = prepare_native_runtime()
        args = _parser().parse_args(argv)
        return execute(args, policy)
    except _REFUSAL_ERRORS as err:
        print(failure_report('S4a refused', err, (RuntimeRefused,), s1=getattr(args, 's1', None),
                             runs_root=getattr(args, 'runs_root', None)), file=sys.stderr)
        return _failure_exit_code(err)




def _load_inputs(meta):
    from types import SimpleNamespace
    from c_ept.valuation import value
    from c_ept.preprocessing import terrain
    from c_ept.data.store_core import Store
    from c_ept.ept import encoder
    paths = meta['paths']
    labels = Store(paths['labels'])
    labels.verify()  # derived FIT/STOP/DEV labels with other-role sentinels, not raw S2
    return SimpleNamespace(s1=Store(paths['s1']), s2=Store(paths['s2']), labels=labels,
                           static=encoder.static_channels(terrain.Terrain(paths['terrain'])),
                           prices=value.load_prices(paths['prices']))


def _model_config(size):
    from c_ept.ept.encoder import EncoderConfig
    dims = {'S': (128, 4, 4), 'M': (256, 6, 8)}
    d, layers, heads = dims[size]
    return EncoderConfig(d=d, layers=layers, heads=heads)


def _cuda_sync(device):
    if device == 'cuda':
        import torch
        torch.cuda.synchronize()


def _stage(run, manifest, ledger, name, identity, check, action):
    from c_ept.runtime import ProcessTreeSampler
    check()
    digest = _digest({'identity': identity, 'stage': name})
    old = ledger.stage(name)
    if ledger.complete(name):
        if old.get('inputs_sha256') != digest or name not in manifest['results']:
            raise RuntimeRefused('completed stage lacks matching inputs/RUN result')
        if not old.get('outputs') or any(Path(rel).is_absolute() or '..' in Path(rel).parts or ':' in rel or _sha(run / rel) != sha for rel, sha in old['outputs'].items()):
            raise RuntimeRefused('completed stage output missing or changed')
        result = _read(run / f'{name}.json')
        if result.get('inputs_sha256') != digest or result != manifest['results'][name]:
            raise RuntimeRefused('completed stage result binding differs')
        return result['data']
    if any(ledger.complete(x) for x in manifest['stages'][manifest['stages'].index(name)+1:]):
        raise RuntimeRefused('complete stage follows an incomplete stage')
    if (run / f'{name}.json').exists():
        raise RuntimeRefused('unfinalized stage result requires a new run; no deletion/overwrite recovery')
    ledger.begin(name, {'run_identity_sha256': _digest(identity)}, digest)
    with ProcessTreeSampler() as sampler:
        def bounded():
            if sampler.peak <= 0 or sampler.peak >= MEMORY_CAP:
                raise RuntimeRefused('process-tree sampled memory unavailable or exceeds strict 40 GiB bound')
        bounded()
        with sampler.scope() as scope:
            result, output_files = action(bounded)
        bounded()
    record = {'schema': 's4a_stage_v1', 'stage': name, 'inputs_sha256': digest, 'data': result,
              'resources': {'wall_seconds': scope.wall, 'peak_bytes': scope.peak, 'peak_upper_bytes': scope.peak_upper,
                            'metric': 'process_tree_private_bytes_windows_else_rss', 'sampled': True,
                            'scope': 'action only; excludes stage ledger/result finalization'}}
    path = run / f'{name}.json'
    _write(path, record)
    outputs = {str(Path(p).relative_to(run)).replace('\\','/'): _sha(p) for p in [path, *output_files]}
    manifest['results'][name] = record
    _write(run / 'RUN.json', manifest)
    ledger.finish(name, outputs)
    return result


def _best_model(folder, size, device, static):
    import io
    import torch
    from c_ept.ept import heads, checkpoints
    checkpoint = checkpoints.latest_checkpoint(folder)
    if checkpoint is None or not checkpoint[1]['done']:
        raise RuntimeRefused('inference requires a completed verified best STOP checkpoint')
    model = heads.EPT(_model_config(size), static)
    model.load_state_dict(torch.load(io.BytesIO(checkpoint[3]), weights_only=True, map_location='cpu'))
    return model.to(device).eval()


def _train_stage(run, args, bundle, stats, factors, proxy, size, device, check):
    import torch
    from c_ept.ept import data, train
    from c_ept.runtime import ProcessTreeSampler
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeRefused('requested CUDA device unavailable; no silent device substitution')
    rows = {role: data.ept_head_rows(bundle.s1, bundle.s2, bundle.labels, role) for role in ('FIT','STOP')}
    full = data.EngagementRows(rows['FIT'])
    grouped = {role: data.EngagementRows(rows[role], proxy[role]) for role in ('FIT','STOP')}
    batches = {role: data.Batches(grouped[role], args.batch_size, args.seed, stats=stats, training=role == 'FIT') for role in ('FIT','STOP')}
    training = train.TrainData(batches['FIT'], batches['STOP'], factors, bundle.static,
                              static_source={'terrain_files_sha256': bundle.meta['terrain_files_sha256']})
    cfg = train.TrainConfig(_model_config(size), args.lr, args.weight_decay, args.warmup_steps)
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    folder = run / 'train' / f'{size}_{device}'
    resumed = folder.exists() and any(folder.iterdir())
    with ProcessTreeSampler() as sampler:
        with sampler.scope() as scope:
            _cuda_sync(device)
            began = time.perf_counter()
            result = train.train(cfg, training, folder, device=device, epochs=args.epochs, check=check, seed=args.seed)
            _cuda_sync(device)
            wall = time.perf_counter() - began
    if scope.peak <= 0 or scope.peak >= MEMORY_CAP:
        raise RuntimeRefused('whole train-call sampled memory unavailable or over bound')
    finals = [_read(path) for path in sorted((folder/'calls').glob('call_*.json')) if not path.name.endswith('.pending.json')]
    recovered = any(r.get('outcome') == 'hard_interrupted' for r in finals)
    fields = {'size': size, 'device': device, 'seed': args.seed, 'epochs_run': result.epochs_run,
              'best_epoch': result.best_epoch, 'whole_call_seconds': wall, 'epoch_seconds': result.epoch_seconds,
              'load_seconds': result.load_seconds, 'step_seconds': result.step_seconds, 'stop_seconds': result.stop_seconds,
              'timing_components': result.timing_components, 'budget_seconds': result.seconds,
              'recovered': recovered, 'resumed': resumed, 'budget_seconds_estimated': recovered,
              'accounting_scope': 'cumulative budget; recovered beat+one-period tail conditional, not exact performance',
              'whole_call_peak_bytes': scope.peak, 'peak_upper_bytes': scope.peak_upper,
              'peak_rss_bytes': result.peak_rss_bytes, 'sampled': True,
              'checkpoint_bytes': sum(p.stat().st_size for p in folder.rglob('*') if p.is_file()),
              'checkpoint_byte_scope': 'all files of this training run, including calls and older epochs',
              'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else None,
              'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved() if device == 'cuda' else None,
              'proxy_work_items': len(grouped['FIT'].units)+len(grouped['FIT'].frames),
              'full_work_items': len(full.units)+len(full.frames), 'checkpoint': str(folder.relative_to(run)),
              'residual_cost': 'M2Rate training retains full mark/zone RNG; evaluation uses rate-only; M2Mark+Zone union shared',
              'epoch_scope': 'pre-checkpoint host span excludes checkpoint preparation and save; STEP overlaps transfer/forward/backward/optimizer and includes guard/LR/bookkeeping',
              'checkpoint_io_scope': 'current-call save only; excludes best-state copy and _torch_bytes serialization; prior-attempt writes unavailable',
              'whole_call_scope': 'outer host span around train includes setup, checkpoint preparation/save and final marker; separate from cumulative budget; do not sum overlapping fields',
              'measurement_overhead': 'host clocks and CUDA synchronization at component boundaries; affects measured throughput'}
    return fields, sorted(p for p in folder.rglob('*') if p.is_file())


def _dev_selection(bundle, args):
    import numpy as np
    from c_ept.valuation import compose
    rows = compose.row_table(bundle.s1, bundle.s2, ('DEV',))
    if not len(rows):
        raise RuntimeRefused('DEV has no frame-clock rows')
    idx = np.sort(np.random.default_rng(args.seed).choice(len(rows), min(args.dev_sample, len(rows)), replace=False))
    return rows, idx


def _inference_stage(run, args, bundle, stats, trains, check, bounded, composition=False):
    import torch
    from c_ept.valuation import compose, towertier
    from c_ept.ept.compose_heads import EPTComposerHeads
    rows, idx = _dev_selection(bundle, args)
    records = []
    for fitted in trains:
        check(); bounded()
        device, size = fitted['device'], fitted['size']
        model = _best_model(run/fitted['checkpoint'], size, device, bundle.static)
        adapter = EPTComposerHeads(model, stats)
        _cuda_sync(device)
        started = time.perf_counter()
        if composition:
            from dataclasses import fields
            sampled = compose.RowTable(**{field.name:(getattr(rows,field.name) if field.name == 'roles' else getattr(rows,field.name)[idx]) for field in fields(rows)})
            tier = towertier.TierModel.load(Path(bundle.meta['paths']['s3'])/'fit/tier')
            engine = compose.Composer(adapter, bundle.prices, tier, bundle.labels.array('labels/gold_reps'), sampled, bundle.s2, S=200, seed=args.seed)
        else:
            adapter.bind(bundle.s2, rows.views[idx])
        _cuda_sync(device)
        bind_seconds = time.perf_counter()-started
        started = time.perf_counter()
        if composition:
            engine.simulate(list(range(len(idx))), check=lambda:(check(), bounded()))
        else:
            with torch.no_grad():
                adapter.rate(list(range(len(idx))))
        _cuda_sync(device)
        seconds = time.perf_counter()-started
        bounded()
        records.append({'size':size,'device':device,'rows':len(idx),'sample_sha256':_digest(idx.tolist()),
                        'seed':args.seed,'S':200 if composition else None,'bind_seconds':bind_seconds,
                        'seconds':seconds,'seconds_per_row':seconds/len(idx),
                        'scope':'sampled DEV composition' if composition else 'DEV encode plus M2Rate; head-only seconds excludes encode',
                        'encoding_seconds_per_row':bind_seconds/len(idx) if not composition else None})
        if composition:
            del engine
        del model, adapter
    return {'measurements':records}, []


def _projections(trains):
    """Separate row-scaled epoch work, epoch-only IO and unresolved residual sensitivity."""
    result = []
    for fitted in trains:
        if fitted['recovered'] or fitted.get('resumed',False):
            result.append({'size':fitted['size'],'device':fitted['device'],'available':False,
                           'reason':'recovered or resumed attempt is not uninterrupted performance calibration'})
            continue
        observed_epochs = fitted['epochs_run']
        epoch_total = sum(fitted['epoch_seconds'])
        io_total = sum(v['seconds'] for v in fitted['timing_components']['checkpoint_io_current_call'])
        epoch_mean, io_mean = epoch_total/observed_epochs, io_total/observed_epochs
        residual = max(0.,fitted['whole_call_seconds']-epoch_total-io_total)
        ratio = fitted['full_work_items']/fitted['proxy_work_items']
        scenarios = {}
        for label, horizon in PROJECTION_EPOCHS.items():
            def estimate(row_ratio, fits, epoch_scaled_residual=False):
                unallocated = residual*horizon/observed_epochs if epoch_scaled_residual else residual
                return project_cost(epoch_mean,unallocated,row_ratio,horizon,fits,checkpoint_io_seconds=io_mean)
            def values(epoch_scaled=False):
                return {'S4b_seconds':sum(estimate(ratio*fraction,9,epoch_scaled) for fraction in (.25,.5,1.)),
                        'S6_seconds':estimate(1.,30,epoch_scaled),'S7_seconds':estimate(ratio,24,epoch_scaled)}
            fixed_assumption, epoch_assumption = values(), values(True)
            scenarios[label] = {'epochs':horizon, **fixed_assumption,
                                'unallocated_residual_sensitivity':{key:sorted([value,epoch_assumption[key]]) for key,value in fixed_assumption.items()}}
        result.append({'size':fitted['size'],'device':fitted['device'],'available':True,'scenarios':scenarios,
                       'pre_checkpoint_epoch_seconds_mean':epoch_mean,'checkpoint_io_seconds_mean_current_call':io_mean,
                       'unallocated_full_call_residual_seconds':residual,'row_ratio':ratio,
                       'assumptions':{'S4b':{'fractions':[.25,.5,1.],'seeds':3,'additional_design_checks':2,'extra_assumption':'baseline and both checks repeated at EVERY fraction and seed; 27 training calls under this planning assumption, not a finalized S4b design'},
                                      'S6':'30proxytrials','S7':'24fullfits','epochs':PROJECTION_EPOCHS,
                                      'scope':'training calls only; excludes scales/statistics preprocessing, standalone inference, composition, orchestration and catalog stages; measured stage costs reported separately, no new evaluation row counts assumed',
                                      'row_scaling':'pre-checkpoint epoch work only; FIT unit+frame ratio is a proxy assuming STOP and other workload scale similarly',
                                      'checkpoint_io':'current-call save IO only, epoch-scaled, never row-scaled; previous-attempt writes unavailable',
                                      'residual':'checkpoint preparation, best-state copies, tensor serialization, setup and final marker are not separately identified; primary assumes fixed per fit; sensitivity treats all residual as per observed epoch without row scaling',
                                      'interpretation':'conditional sensitivity, not empirical confidence bounds or measured end-to-end epoch; bottlenecks, IO size and setup may change',
                                      'nonoverlap':'epoch_seconds overlaps step and components; only epoch total plus disjoint save IO used; components never added again'}})
    return {'projections':result}


def _finalize_cost(run, manifest, cost):
    """Bind final COST bytes once; resume verifies and preserves recorded measurements."""
    path = run/'COST.json'
    recorded = manifest.get('cost_sha256')
    if recorded is not None:
        if not path.is_file() or _sha(path) != recorded:
            raise RuntimeRefused('COST missing or differs from RUN-bound bytes')
        old = _read(path)
        if old.get('schema') != COST_SCHEMA or old.get('run_identity_sha256') != manifest['run_identity_sha256']:
            raise RuntimeRefused('COST schema or run identity differs')
        return
    if path.exists():
        raise RuntimeRefused('orphan COST lacks RUN integrity binding; use a new run')
    _write(path, cost)
    manifest['cost_sha256'] = _sha(path)
    _write(run/'RUN.json', manifest)


def execute(args, policy):
    """Run verified stages after native bootstrap; caller uses main for supported CLI."""
    from c_ept import runlog
    from c_ept.workflows import run_s3, run_s1
    from c_ept.runtime import E6Guard
    from c_ept.ept import data, train, scales
    normal, operator_digest = _normal_args(args)
    last_probe = {}
    def validated_probe(record):
        validate_e6_probe(record)
        last_probe.clear()
        last_probe.update(record)
    e6_guard = E6Guard(lambda: run_s1.require_no_strata(), validate=validated_probe)
    _e6(e6_guard)
    code = repository_state()
    meta = inspect_inputs(args)
    _output_paths(args, meta)
    train.deterministic_cpu(threads=8, seed=args.seed)
    if 'cuda' in args.devices:
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (None, ':4096:8'):
            raise RuntimeRefused('incompatible CUBLAS_WORKSPACE_CONFIG')
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    runtime = {**_runtime_fingerprint(policy), 'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG')}
    identity = {'args':normal,'operator_config_sha256':operator_digest,'code':code,'runtime':runtime,'inputs':meta}
    def check():
        _e6(e6_guard)
        run_s3.check_resume_code(code, repository_state())
        current = {**_runtime_fingerprint(prepare_native_runtime()),'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG')}
        if current != runtime or inspect_inputs(args) != meta:
            raise RuntimeRefused('runtime or independently verified inputs changed during run')
    if args.resume:
        run = Path(args.resume).resolve()
        if run.parent != Path(args.runs_root).resolve():
            raise RuntimeRefused('resume must belong to requested runs root')
    else:
        run = runlog.new_run('s4a', Path(args.runs_root))
    with run_lock(run):
        stages = ['scales','stats'] + [f'train_{size}_{device}' for size in args.sizes for device in args.devices] + ['infer','compose','project']
        if args.resume:
            manifest = _read(run/'RUN.json')
            verify_resume(manifest, identity)
            if manifest.get('stages') != stages:
                raise RuntimeRefused('stage selection changed')
        else:
            manifest = {'schema':RUN_SCHEMA,'stage':'s4a','operator_api_version':1,'code_commit':code['code_commit'],
                        'code_dirty':False,'created':dt.datetime.now(dt.timezone.utc).isoformat(),'identity':identity,
                        'run_identity_sha256':_digest(identity),'operator_config_sha256':operator_digest,'runtime':runtime,
                        'stages':stages,'results':{},'stats_population':'whole FIT; train/STOP proxy25% default',
                        **provenance_record()}      # record fields beside the identity, never inside it (A5-F4)
            _write(run/'RUN.json', manifest)
        ledger = run_s3.Ledger(run)
        _write(run/'PROCESS.json',_process_record(run,manifest,'running'))
        try:
            bundle = _load_inputs(meta)
            bundle.meta = meta
            def scale_action(bounded):
                path = run/scales.SCALES_NAME
                if not path.exists():
                    source = Path(meta['paths']['s3'])/scales.SCALES_NAME
                    if source.exists():
                        scales.load_scales(source, expected_provenance=meta['expected_provenance'])
                        from c_ept.data.store_core import write_atomic
                        write_atomic(path, source.read_bytes())
                    else:
                        scales.b0_scales(meta['paths']['s3'],out=run,runs_root=args.runs_root,check=check)
                value = scales.load_scales(path, expected_provenance=meta['expected_provenance'])
                return {'values':value,'provenance':meta['expected_provenance']}, [path]
            factors = _stage(run,manifest,ledger,'scales',identity,check,scale_action)['values']
            def stats_action(bounded):
                path = run/'stats/STATS.json'
                if path.exists():
                    raise RuntimeRefused('orphan statistics require a new run; cannot certify source binding')
                fitted = data.fit_stats(bundle.s1,bundle.s2)
                fitted.save(path)
                return {'sha256':fitted.sha256,'role':'FIT','population':'whole FIT'}, [path]
            _stage(run,manifest,ledger,'stats',identity,check,stats_action)
            stats = data.Stats.load(run/'stats/STATS.json')
            proxy = data.proxy_matches(bundle.s1,args.share,args.seed)
            trains = []
            for size in args.sizes:
                for device in args.devices:
                    name = f'train_{size}_{device}'
                    action = lambda bounded,size=size,device=device: _train_stage(run,args,bundle,stats,factors,proxy,size,device,check)
                    trains.append(_stage(run,manifest,ledger,name,identity,check,action))
            infer = _stage(run,manifest,ledger,'infer',identity,check,lambda bounded:_inference_stage(run,args,bundle,stats,trains,check,bounded))
            comp = _stage(run,manifest,ledger,'compose',identity,check,lambda bounded:_inference_stage(run,args,bundle,stats,trains,check,bounded,True))
            def project_action(bounded):
                return _projections(trains), []
            projection = _stage(run,manifest,ledger,'project',identity,check,project_action)
            cost = {'schema':COST_SCHEMA,'run_identity_sha256':manifest['run_identity_sha256'],'runtime':runtime,
                    'training':trains,'infer':infer,'compose':comp,**projection,
                    'limits':{'memory':'sampled private <40GiB; observed lifetime upper separate, not a continuous proof',
                              'budget':'trainer cumulative <=24h; recovered tail assumes timely durable heartbeat',
                              'interpretation':'uninterrupted component costs separate from estimated recovery; no B0=1 claim'},
                    'e6':{'calls':e6_guard.calls,'seconds':e6_guard.seconds,'last_probe':last_probe,
                          'visibility':'unreadable command lines limit detection; accepted off is not absolute absence proof'}}
            _finalize_cost(run,manifest,cost)
            runlog.register(run,manifest,Path(args.catalog_db))
            _write(run/'PROCESS.json',_process_record(run,manifest,'finished','completed',0))
        except BaseException as err:
            _write(run/'PROCESS.json',_process_record(run,manifest,'failed','failed',_failure_exit_code(err),type(err).__name__))
            raise
    print(json.dumps({'run':str(run),'cost':str(run/'COST.json'),'schema':COST_SCHEMA}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
