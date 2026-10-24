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
import bsh20260916_common as B  # noqa: E402

PLANS = {
    'smoke': [['bsh20260916_freeze.py', '--smoke'], ['bsh20260916_shap.py', '--smoke']],
    'full': [['bsh20260916_freeze.py'], ['bsh20260916_shap.py']],
    'post': [['bsh20260916_snapshot.py', '--phase', 'after'], ['bsh20260916_postrun_checks.py'], ['bsh20260916_report.py']],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', choices=sorted(PLANS), required=True)
    args = ap.parse_args()
    B.log_command()
    st = B.Status(f'driver_{args.plan}')
    steps = PLANS[args.plan]
    for i, step in enumerate(steps):
        name = '_'.join(s.lstrip('-') for s in [step[0].replace('bsh20260916_', '').replace('.py', '')] + step[1:] if s not in ('--phase',))
        st.update('running', name, processed=i, total=len(steps), next_step=name)
        t0 = time.time()
        (B.OUT / 'logs').mkdir(parents=True, exist_ok=True)
        with open(B.OUT / 'logs' / f'{name}.out', 'ab') as fo, open(B.OUT / 'logs' / f'{name}.err', 'ab') as fe:
            proc = subprocess.run([sys.executable, '-B', str(B.ROOT / 'scripts' / step[0])] + step[1:], cwd=str(B.ROOT), stdout=fo, stderr=fe)
        st.log(f'{name}: returncode={proc.returncode} seconds={time.time() - t0:.1f}')
        if proc.returncode != 0:
            B.log_failure(f'driver_{args.plan}', f'{name} returned {proc.returncode}')
            st.update('failed', name, processed=i, total=len(steps), error=f'{name} returncode {proc.returncode}', next_step='inspect logs; failure retained')
            return proc.returncode
    st.update('complete', args.plan, processed=len(steps), total=len(steps), next_step='next plan')
    return 0


if __name__ == '__main__':
    sys.exit(main())
