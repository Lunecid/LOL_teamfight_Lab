# c_ept module guide

Implementation source lives in the domain packages below. The original root modules
are small compatibility aliases; each algorithm has one implementation file.

| Package | Implementation modules | Responsibility |
| --- | --- | --- |
| `data` | `casts`, `reader`, `store_core`, `store_build`, `event_rows`, `frame_rows`, `head_rows` | Checked conversion, records, stores, row construction |
| `preprocessing` | `features`, `terrain`, `tokens`, `traits` | Model inputs, map geometry, token and trait preparation |
| `domain` | `units`, `targets`, `state`, `symmetry` | Engagement definitions, targets, state and symmetry |
| `valuation` | `flowprice`, `rulepay`, `towertier`, `value`, `compose` | Prices, payout rules, values and composition |
| `models` | `baselines_b0`, `baselines_b1`, `heads`, `nll` | Baselines, head interfaces and likelihood scores |

`ept/` and `runtime/` retain their existing organization. `config.py`, `legacy.py`,
`legacy_pins.json`, `runlog.py` and protected `cohort.py` stay at the root, as do
`run_s*`, `report_s*`, `gate_s3.py`, `s4a_operator.py`, `s4a_runtime.py` and `checks/`.
These retained paths preserve configuration, resource, runner and CLI contracts.

## Migrating imports

Existing imports continue to work:

```python
from c_ept import reader
from c_ept.reader import MatchRecord
```

New code can use the grouped implementation path:

```python
from c_ept.data import reader
from c_ept.data.reader import MatchRecord
```

After either import order, both paths resolve to the same module object and both
parent packages expose that object. Patching either module changes the globals used
by its functions. Group `__init__.py` files contain only a docstring, so importing a
group does not eagerly import its implementations. There are no import finders or
`sys.path` modifications.

The implementation bodies and their existing absolute imports are byte-identical
to the pre-move source. Their `__file__`, `__name__`, `__package__`, and `__spec__`
now describe the grouped path. Code that locates the repository from an implementation
file must instead use a retained location anchor such as `c_ept.config.__file__`.
Static source checks should inspect the imported module's `__file__`, not the root
alias. Existing source-inspection tests follow this convention.

Old pickle references such as `c_ept.reader.MatchRecord` still load through the root
alias and resolve to the moved class. New pickles record the grouped class path and
require this layout (they are not promised to load in older checkouts). No class
`__module__` is rewritten by the implementation. Hash-bound scientific runs remain
subject to their existing source and resume contracts.

`python -m c_ept.flowprice` delegates to the moved `main`; runner and operator CLI
modules retain their root paths. No defaults, scientific calculations, data, splits,
model parameters or checkpoint schemas are changed by the move.

## Bounded verification

`tests/test_module_layout.py` uses fresh interpreters to test all 24 modules in both
import orders, parent attributes, function globals, legacy and new type pickle
references, synthetic instance pickle loading, monkeypatch execution and the legacy
flowprice dispatcher. It also checks that group imports add no path hooks or eager
implementation imports. It does not read research inputs or train models.

The regression command from the repository root is:

```powershell
python -B -m pytest -q -p no:cacheprovider --tb=short c_ept/tests/test_module_layout.py
```

Set `CUDA_VISIBLE_DEVICES=-1`, `MKL_CBWR=COMPATIBLE`, and the native thread counts
(`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`,
`NUMBA_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `BLIS_NUM_THREADS`) to `1` first.
These regression tests need neither Git history nor `work/` files. Their dependencies
are the package source and installed test/runtime libraries (pytest, NumPy, SciPy,
LightGBM and the existing runner utilities' optional dependencies).

The separate implementation audit, full command list, source hashes, changed paths,
and test results are recorded in `work/c_ept-grouping-result.md` for the handoff.
