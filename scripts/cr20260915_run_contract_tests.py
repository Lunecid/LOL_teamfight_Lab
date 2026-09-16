"""Run the TRAIN-only contract tests and record the gate file contract_tests/result.json (checked before full fits)."""
from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fc20260915_common as C  # noqa: E402
import cr20260915_common as K  # noqa: E402

TESTS = ['tests/test_cr20260915_contracts.py', 'tests/test_state_value_v2_contract.py', 'tests/test_fc20260915_contracts.py']


def main():
    run = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    out = K.OUT / 'contract_tests'
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', f'--basetemp={K.OUT / "probes" / f"pytest_basetemp_run{run}"}'] + TESTS
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(K.ROOT), capture_output=True, text=True, encoding='utf-8', errors='replace')
    log = proc.stdout + '\n' + proc.stderr
    (out / f'run{run}.txt').write_bytes(log.encode('utf-8'))
    m = re.search(r'(\d+) passed', log)
    f = re.search(r'(\d+) failed', log)
    e = re.search(r'(\d+) error', log)
    s = re.search(r'(\d+) skipped', log)
    res = dict(run=run, command=' '.join(cmd[1:]), returncode=proc.returncode, passed_count=int(m.group(1)) if m else 0,
               failed_count=int(f.group(1)) if f else 0, error_count=int(e.group(1)) if e else 0, skipped_count=int(s.group(1)) if s else 0,
               passed=proc.returncode == 0, seconds=round(time.time() - t0, 1), at=time.strftime('%Y-%m-%d %H:%M:%S'),
               test_sha256={t: C.sha256_file(K.ROOT / t) for t in TESTS},
               script_sha256={p.name: C.sha256_file(p) for p in sorted((K.ROOT / 'scripts').glob('cr20260915_*.py'))},
               log=f'contract_tests/run{run}.txt', scope='synthetic and TRAIN-only fixtures; no TEST/external rows or labels')
    C.write_json(out / f'result_run{run}.json', res)
    if res['passed']:
        C.write_json(out / 'result.json', res)
    print({k: v for k, v in res.items() if k not in ('script_sha256', 'test_sha256')})
    return 0 if res['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
