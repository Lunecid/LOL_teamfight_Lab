"""Sequential stage driver (one child process at a time). Stops at the first failing stage; each child's stdout/stderr is
kept in logs/<stage>.out|err (appended, attempts retained)."""
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
import ta20260916_common as T  # noqa: E402

FIT_ORDER = [(c, f) for c in T.COHORTS for f in T.FAMILIES]
PLANS = {
    'smoke': [['ta20260916_fit.py', '--smoke', '--cohort', c, '--family', f] for c, f in FIT_ORDER]
             + [['ta20260916_freeze.py', '--smoke'], ['ta20260916_evaluate.py', '--smoke']],
    'full': [['ta20260916_fit.py', '--cohort', c, '--family', f] for c, f in FIT_ORDER],
    'freeze_eval': [['ta20260916_freeze.py'], ['ta20260916_evaluate.py']],
    'post': [['ta20260916_snapshot.py', '--phase', 'after'], ['ta20260916_postrun_checks.py'], ['ta20260916_report.py']],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', choices=sorted(PLANS), required=True)
    args = ap.parse_args()
    T.log_command()
    st = T.Status(f'driver_{args.plan}')
    steps = PLANS[args.plan]
    for i, step in enumerate(steps):
        name = '_'.join(s.lstrip('-') for s in [step[0].replace('ta20260916_', '').replace('.py', '')] + step[1:] if s not in ('--cohort', '--family', '--phase'))
        st.update('running', name, processed=i, total=len(steps), next_step=name)
        t0 = time.time()
        (T.OUT / 'logs').mkdir(parents=True, exist_ok=True)
        with open(T.OUT / 'logs' / f'{name}.out', 'ab') as fo, open(T.OUT / 'logs' / f'{name}.err', 'ab') as fe:
            proc = subprocess.run([sys.executable, '-B', str(T.ROOT / 'scripts' / step[0])] + step[1:], cwd=str(T.ROOT), stdout=fo, stderr=fe)
        st.log(f'{name}: returncode={proc.returncode} seconds={time.time() - t0:.1f}')
        if proc.returncode != 0:
            T.log_failure(f'driver_{args.plan}', f'{name} returned {proc.returncode}')
            st.update('failed', name, processed=i, total=len(steps), error=f'{name} returncode {proc.returncode}', next_step='inspect logs; failure retained')
            return proc.returncode
    st.update('complete', args.plan, processed=len(steps), total=len(steps), next_step='next plan')
    return 0


if __name__ == '__main__':
    sys.exit(main())
