"""Sequential driver for the full pre-freeze fits (one CPU-bounded process at a time) and the freeze.

Steps: fit_cohort_q (A, h90/h60/h120) -> role_models (B) -> fit_arms (C, h90, T and N + T ablations) -> freeze.
Stops at the first failing step; every step logs to logs/run_<step>.out/.err and commands.txt.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cr20260915_common as K  # noqa: E402

STEPS = [('fit_cohort_q', ['scripts/cr20260915_fit_cohort_q.py']),
         ('role_models', ['scripts/cr20260915_role_models.py']),
         ('fit_arms', ['scripts/cr20260915_fit_arms.py']),
         ('freeze', ['scripts/cr20260915_freeze.py'])]


def main():
    only = sys.argv[1].split(',') if len(sys.argv) > 1 else [s for s, _ in STEPS]
    st = K.Status('driver_fits')
    for name, args in STEPS:
        if name not in only:
            continue
        cmd = [sys.executable] + args
        K.log_command(' '.join(['python'] + args) + f' (driver; logs/run_{name}.out)')
        st.update('running', name, next_step='wait')
        t0 = time.time()
        with open(K.OUT / 'logs' / f'run_{name}.out', 'ab') as fo, open(K.OUT / 'logs' / f'run_{name}.err', 'ab') as fe:
            rc = subprocess.run(cmd, cwd=str(K.ROOT), stdout=fo, stderr=fe).returncode
        K.log_command(f'  -> {name} returncode {rc} in {round(time.time() - t0, 1)} s')
        if rc != 0:
            st.update('failed', name, error=f'returncode {rc}', next_step='inspect logs, fix, resume with remaining steps')
            return rc
    st.update('complete', 'driver_fits', next_step='evaluation')
    return 0


if __name__ == '__main__':
    sys.exit(main())
