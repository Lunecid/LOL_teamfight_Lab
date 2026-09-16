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
import hs20260916_common as HS  # noqa: E402

FIT_ORDER = [(h, c, f) for h in HS.HORIZONS for c in HS.COHORTS for f in HS.FAMILIES]
PLANS = {
    'smoke': [['hs20260916_fit.py', '--smoke', '--horizon', '60', '--cohort', c, '--family', f] for c in HS.COHORTS for f in HS.FAMILIES]
             + [['hs20260916_freeze.py', '--smoke'], ['hs20260916_evaluate.py', '--smoke']],
    'full': [['hs20260916_fit.py', '--horizon', str(h), '--cohort', c, '--family', f] for h, c, f in FIT_ORDER],
    'freeze_eval': [['hs20260916_freeze.py'], ['hs20260916_evaluate.py']],
    'post': [['hs20260916_snapshot.py', '--phase', 'after'], ['hs20260916_postrun_checks.py'], ['hs20260916_report.py']],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', choices=sorted(PLANS), required=True)
    args = ap.parse_args()
    HS.log_command()
    st = HS.Status(f'driver_{args.plan}')
    steps = PLANS[args.plan]
    for i, step in enumerate(steps):
        name = '_'.join(s.lstrip('-') for s in [step[0].replace('hs20260916_', '').replace('.py', '')] + step[1:] if s not in ('--cohort', '--family', '--phase', '--horizon'))
        st.update('running', name, processed=i, total=len(steps), next_step=name)
        t0 = time.time()
        (HS.OUT / 'logs').mkdir(parents=True, exist_ok=True)
        with open(HS.OUT / 'logs' / f'{name}.out', 'ab') as fo, open(HS.OUT / 'logs' / f'{name}.err', 'ab') as fe:
            proc = subprocess.run([sys.executable, '-B', str(HS.ROOT / 'scripts' / step[0])] + step[1:], cwd=str(HS.ROOT), stdout=fo, stderr=fe)
        st.log(f'{name}: returncode={proc.returncode} seconds={time.time() - t0:.1f}')
        if proc.returncode != 0:
            HS.log_failure(f'driver_{args.plan}', f'{name} returned {proc.returncode}')
            st.update('failed', name, processed=i, total=len(steps), error=f'{name} returncode {proc.returncode}', next_step='inspect logs; failure retained')
            return proc.returncode
    st.update('complete', args.plan, processed=len(steps), total=len(steps), next_step='next plan')
    return 0


if __name__ == '__main__':
    sys.exit(main())
