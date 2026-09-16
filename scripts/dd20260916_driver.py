"""Sequential stage driver (one child process at a time). Stops at the first failing stage; child stdout/stderr kept in logs/."""
from __future__ import annotations

import os

os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dd20260916_common as DD  # noqa: E402

EXT_SETS = [f'EXT_{s}' for s in DD.EXT]
PLANS = {
    'detect': [['dd20260916_redetect.py', '--set', 'MAIN', '--definition', 'frozen', '--limit', str(DD.FROZEN_CHECK_MATCHES)],
               ['dd20260916_redetect.py', '--set', 'MAIN', '--definition', 'dev']]
              + [['dd20260916_redetect.py', '--set', s, '--definition', 'dev'] for s in EXT_SETS] + [['dd20260916_census.py']],
    'rebuild_dev': [['dd20260916_rebuild.py', '--fixture', str(DD.FIXTURE_MATCHES)], ['dd20260916_rebuild.py', '--set', 'MAIN_TRAIN'], ['dd20260916_rebuild.py', '--set', 'MAIN_VALIDATION']],
    'fit': [['dd20260916_fit_freeze.py']],
    'sealed': [['dd20260916_rebuild.py', '--set', 'MAIN_TEST']] + [['dd20260916_rebuild.py', '--set', s] for s in EXT_SETS] + [['dd20260916_evaluate.py']],
    'post': [['dd20260916_snapshot.py', '--phase', 'after'], ['dd20260916_postrun_checks.py'], ['dd20260916_report.py']],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', choices=sorted(PLANS), required=True)
    args = ap.parse_args()
    DD.log_command()
    st = DD.Status(f'driver_{args.plan}')
    steps = PLANS[args.plan]
    for i, step in enumerate(steps):
        name = '_'.join(s.lstrip('-') for s in [step[0].replace('dd20260916_', '').replace('.py', '')] + step[1:] if s not in ('--set', '--definition', '--limit', '--phase', '--fixture'))
        st.update('running', name, processed=i, total=len(steps), next_step=name)
        t0 = time.time()
        (DD.OUT / 'logs').mkdir(parents=True, exist_ok=True)
        with open(DD.OUT / 'logs' / f'{name}.out', 'ab') as fo, open(DD.OUT / 'logs' / f'{name}.err', 'ab') as fe:
            proc = subprocess.run([sys.executable, '-B', str(DD.ROOT / 'scripts' / step[0])] + step[1:], cwd=str(DD.ROOT), stdout=fo, stderr=fe)
        st.log(f'{name}: returncode={proc.returncode} seconds={time.time() - t0:.1f}')
        if proc.returncode != 0:
            DD.log_failure(f'driver_{args.plan}', f'{name} returned {proc.returncode}')
            st.update('failed', name, processed=i, total=len(steps), error=f'{name} returncode {proc.returncode}', next_step='inspect logs; failure retained')
            return proc.returncode
    st.update('complete', args.plan, processed=len(steps), total=len(steps), next_step='next plan')
    return 0


if __name__ == '__main__':
    sys.exit(main())
