# EPT component map

The visible `c_ept/` root now contains five Python files: `__init__.py`, `config.py`,
`legacy.py`, `runlog.py`, and the protected `cohort.py`. The root also retains
`legacy_pins.json` and this guide. Thirty-four former root files have moved.

| Package | Modules | Responsibility |
| --- | --- | --- |
| `data/` | casts, reader, store_core, store_build, event_rows, frame_rows, head_rows | Records, stores and training rows |
| `preprocessing/` | features, terrain, tokens, traits | Model input preparation |
| `domain/` | units, targets, state, symmetry | Engagement definitions, targets and symmetry |
| `valuation/` | flowprice, rulepay, towertier, value, compose | Prices, payouts and composed values |
| `models/` | baselines_b0, baselines_b1, heads, nll | Baselines and likelihood interfaces |
| `models/neural/` | team_set | N2 team-set encoder with shared EPT heads, explicit architecture identity |
| `ept/` | Encoder, heads, training, checkpoints, epoch_metrics and composition | Transformer and payload-free epoch diagnostics |
| `evaluation/` | Endpoint adapters, metrics, reporting, guards, read-only stores and runtime | Separately gated DEV diagnostics; no implicit training or evaluation on import |
| `comparison/` | protocol, data, models, infrastructure, operator, runtime, evaluation, reporting | Separately gated matched25 EPT/B1/N2 pilot; no implicit fit or evaluation |
| `workflows/` | run_s1, run_s2, run_s3, run_s4a, s4a_operator, s4a_runtime, training_operator, run_training | Frozen cost workflow, separate single-training profile and native startup policy |
| `reporting/` | report_s1, report_s2, report_s3 | Existing report builders |
| `checks/` | gate_s3 and existing diagnostic scripts | Stage acceptance and diagnostics |
| `runtime/` | Existing resources and public exports | Shared resource measurement |
| `_compat/` | 34 small forwarding modules | Legacy import and module CLI compatibility |
| `tests/` | Synthetic tests and separately gated research-data tests | Verification |

## Imports and entrypoints

Use grouped paths for new code:

```python
from c_ept.data.reader import MatchRecord
from c_ept.workflows import run_s4a
from c_ept.reporting import report_s3
```

Active implementation and ordinary tests also use these physical grouped paths.
The T005 cleanup migrated the remaining internal legacy imports without changing
their bound names or numerical bodies. `_compat/` serves external callers,
historical module entrypoints and the explicit migration tests; it is not an
internal dependency layer. An unreferenced public re-export, CLI entrypoint or
serialization name is not automatically dead code.

Existing callers continue to work:

```python
from c_ept import reader, run_s4a
from c_ept.reader import MatchRecord
```

Both import orders resolve to the same implementation module. Parent attributes,
function globals and monkeypatches share that object. Old class pickle references
such as `c_ept.reader.MatchRecord` and `c_ept.s4a_runtime.RuntimeRefused` resolve
through the compatibility file. New pickles use canonical grouped names and require
this layout; loading them in an older checkout is not promised.

The original root **filesystem files** no longer exist. `c_ept.__path__` keeps the
real package directory first and appends `_compat/` for ordinary Python lookup.
There is no global `sys.path` change, custom import finder or eager implementation
import. `_compat/__init__.py` makes its files visible to setuptools package
collection, while its empty `__path__` keeps recursive discovery from advertising
a second `c_ept._compat.reader` namespace. `pkgutil.iter_modules(c_ept.__path__)`
continues to find each public legacy name once.

Both CLI spellings work, for example:

```powershell
python -B -m c_ept.run_s4a --help
python -B -m c_ept.workflows.run_s4a --help
python -B -m c_ept.report_s3 --help
python -B -m c_ept.reporting.report_s3 --help
```

`run_s4a.ps1` and the `lol-teamfight-ept` console entry keep their existing module
names. No active direct-file launcher of the removed root files was found in the
bounded source/PowerShell/TOML scan. Untracked external callers using a literal
`c_ept/run_s4a.py` path must switch to `python -m`; direct-file paths are not retained.

## Anchors and safety contracts

`config`, `legacy`, the pin resource, `runlog` and protected `cohort` retain their
locations. Operator `ROOT` and runner `REPO` use the first package path, which is the
same repository anchor after the move. The small `s4a_runtime` helper belongs in
`workflows/`: importing it does not import `psutil`, NumPy, Torch or SciPy. Placing it
under `runtime/` would execute that package's existing eager resource exports.

S1 extraction provenance uses a fixed path inventory covering its canonical
implementations, their forwarding files, package initializers and unchanged root
anchors. Changed canonical sources cannot hide behind an unchanged shim. Git
failure remains unknown, never clean. S4a whole-repository cleanliness, identity,
input hashes, E6 and resume checks are preserved. Existing scientific run identities
are not migrated or made resumable by this file move.

Historical equation and source inventories retain their original paths. The current
lookup is `docs/module-layout-20261024.json`: `moved_implementations` records the
first grouping; `ept_visible_root_cleanup` records the additional moves and shim
locations. Static source checks now point to implementations, not forwarding files.

## Bounded verification and limits

The migration suites test all 34 legacy/canonical pairs in both cold import orders,
type pickles, synthetic instance pickles, monkeypatch identity, module main dispatch,
help-only CLIs, package discovery, resource anchors, bootstrap imports and extraction
source coverage. The existing grouped package initializer simplicity test is retained.

An offline wheel check stages allowlisted source, substitutes a raising sentinel
for protected `cohort.py` without opening or copying its body, verifies all `_compat`
files are in the wheel, and checks installed legacy/canonical operator and runner
help commands outside the source tree. The existing `pyproject.toml` requires no
change. This remains a source-backed operator: a wheel alone does not provide the
external inputs, repository environment/config files or legacy assets/pin resources.
The staged sentinel wheel is a disposable test artifact, not a distribution.

Run from the repository root with GPU disabled, one native thread per library,
`PYTHONDONTWRITEBYTECODE=1`, `PYTHONUTF8=1`, `PYTHONIOENCODING=utf-8`,
`MKL_THREADING_LAYER=SEQUENTIAL`, `MKL_CBWR=COMPATIBLE` and
`KMP_DUPLICATE_LIB_OK=FALSE`:

```powershell
python -B -m pytest -q -p no:cacheprovider --tb=short c_ept/tests/test_module_layout.py c_ept/tests/test_visible_root.py c_ept/tests/test_s4a_operator.py tests/test_project_boundaries.py
```

This is synthetic software verification, not a scientific run or a complete
research-data test suite. No live run, training, inference, protected source body,
raw input, weight payload, commit or push is part of this change. Exact results,
source audit exceptions and remaining limitations are in
`work/ept-root-cleanup-result.md`.

Epoch FIT/STOP JSON/CSV diagnostics and the metadata-only reader are documented in [the epoch metrics guide](../docs/EPT_EPOCH_METRICS_KO.md). This is instrumentation, not a change to the loss or early-stopping rule.

## Single pilot workflow (EPT-T002)

`run_ept_training.ps1` provides Check/Start/Status for the separately approved CUDA S 25% early-stop pilot. Its strict profile does not loosen the old two-epoch cost operator. `tools/ept_training_controller.py` supervises one attempt; `tools/ept_training_monitor.py` reads bounded scalar metadata and prepares epoch reports/visuals. See [pilot protocol](../docs/EPT_SINGLE_PILOT_KO.md).
