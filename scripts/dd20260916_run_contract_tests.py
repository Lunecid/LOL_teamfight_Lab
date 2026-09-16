"""Run the contract tests and record contract_tests/result_run<N>.json (result.json only when passed)."""
from __future__ import annotations

import os

os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '4'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from pathlib import Path  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fc20260915_common as C  # noqa: E402
import dd20260916_common as DD  # noqa: E402

TESTS = ['tests/test_dd20260916_contracts.py']


def main():
    DD.log_command()
    out = DD.OUT / 'contract_tests'
    out.mkdir(parents=True, exist_ok=True)
    run = 1 + len(list(out.glob('result_run*.json')))
    cmd = [sys.executable, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', f'--basetemp={out / f"pytest_basetemp_run{run}"}'] + TESTS
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(DD.ROOT), capture_output=True, text=True, encoding='utf-8', errors='replace')
    log = proc.stdout + '\n' + proc.stderr
    (out / f'run{run}.txt').write_bytes(log.encode('utf-8'))
    m = re.search(r'(\d+) passed', log)
    f = re.search(r'(\d+) failed', log)
    e = re.search(r'(\d+) error', log)
    s = re.search(r'(\d+) skipped', log)
    res = dict(run=run, command=' '.join(cmd[1:]), returncode=proc.returncode, passed_count=int(m.group(1)) if m else 0, failed_count=int(f.group(1)) if f else 0,
               error_count=int(e.group(1)) if e else 0, skipped_count=int(s.group(1)) if s else 0, passed=proc.returncode == 0 and not (s and int(s.group(1))),
               seconds=round(time.time() - t0, 1), at=time.strftime('%Y-%m-%d %H:%M:%S'), test_sha256={t: C.sha256_file(DD.ROOT / t) for t in TESTS},
               source_sha256=DD.own_source_hashes(), log=f'contract_tests/run{run}.txt')
    C.write_json(out / f'result_run{run}.json', res)
    if res['passed']:
        C.write_json(out / 'result.json', res)
    else:
        DD.log_failure('contract_tests', f'run {run} failed', returncode=proc.returncode, tail=log[-3000:])
    print({k: v for k, v in res.items() if k not in ('source_sha256', 'test_sha256')})
    return 0 if res['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
