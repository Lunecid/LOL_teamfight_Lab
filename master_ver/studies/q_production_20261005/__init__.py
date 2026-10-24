"""q stage: engagement-outcome direction model (reports/architecture_20261005/ARCHITECTURE_KO.md sections 4-7).

Modules: rows (q row table), spatial (XY, sigma, S200, edges, A, M), features (q inputs and
preprocessing), qmodels, qtrain, qselect, qbaselines, qcalibrate, qeval, jobs.  V code is imported,
never edited (its files are bound to the live V release).
"""
import os

# torch and MKL/scikit-learn each ship an OpenMP runtime; without this the second one aborts the
# process inside scipy's L-BFGS-B (seen 2026-10-05).  The V worker sets the same flag.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
