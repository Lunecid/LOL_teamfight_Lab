# Pure components validation — 2026-10-24

Scope: isolated copied branch `codex/modularize-three-projects`. This report artifact describes the working-tree implementation, not code already contained in the baseline commit. **Commit disclosure:** this worker previously created `32871ea` containing only the initial report; the lead has been informed. The updates in this report and implementation/tests remain uncommitted for lead integration and Zeno review. No additional worker commit or push is authorized or performed. Original workspaces, data, checkpoints and published runs were untouched.

## Identity

Pinned baseline: `282dcc36f2e175b33782fc0b4bcb31581e663612`. SHA-256 below hashes exact `git show <baseline>:<path>` bytes:

| Baseline source | SHA-256 |
| --- | --- |
| `COG/runner.py` | `19fb519c265a76297730c542c7084fb624467ac85d12acda9104210b4a314108` |
| `master_ver/studies/v_production_20261004/train.py` | `708987e3ad8bf6960464688e3c1fbae797dce117b0b5002c666d5aa8e0313673` |
| `c_ept/ept/heads.py` | `2822ab1b20cd60a336c84af91fff52fbaeb990ce9f8ca970723f4a54b7b71dfa` |
| `c_ept/ept/compose_heads.py` | `8cafc24af3b8db47a153a97274441c74352287daeb14cae4ad3662405e1afece` |

Final validated snapshot (Gemini fusion/metrics and corrected runner documentation present; uniform docstring filter and separately authorized failure-exit fix applied):

| Validated working-tree source/test | SHA-256 |
| --- | --- |
| `COG/runner.py` | `4a162d4734203b318f441b97598e48fecdfc566e5284da6b9306a25e888d26fc` |
| `COG/app/fusion_spec.py` | `62a9bcf53e3cca9c97190bff6f547893af7082a3b242ff2653392925f3932ced` |
| `master_ver/studies/v_production_20261004/train.py` | `3d3dc1b6bd17435d4103f9afa23018da52873101d29baf184497db9d1c981682` |
| `master_ver/studies/v_production_20261004/metrics.py` | `ea9af11ac638040bb18d96a08a80f782599dcc26bd4b244ad3e8b916b35059c8` |
| `COG/tests/test_fusion_spec.py` | `884d8d506e3b9746529e4e09ed67c4ad95980b6adbd503124906e05418008a5e` |
| `master_ver/studies/v_production_20261004/tests/test_metrics.py` | `a2201661e0f4fc3b48618742ca16f020e20fa0fc90174f93b35446746afec7db` |

These six hashes were unchanged across the final test process. `EPTHeads.__init__` AST SHA-256 was identical to baseline during the initial M2 check and the later 09:58 UTC static check: `d10fc9608d75a9825e4c0fad66a1ba55b9cb8d81dbffd6417d0d273d4fcfc8e4` (`sha256(ast.dump(node).encode())`, default dump options). This establishes unchanged constructor/module registration in those checks. A later concurrent change to `c_ept/ept/heads.py` was observed; the historical M2 runtime result below is not a revalidation of that later file revision.

## Commands and bounded results

Working directory: isolated checkout root. Initial M2 command/environment (PowerShell):

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:PYTHONDONTWRITEBYTECODE='1'
python -B -c "import torch, pytest; torch.set_num_threads(1); torch.set_num_interop_threads(1); raise SystemExit(pytest.main(['-q', '-p', 'no:cacheprovider', 'c_ept/tests/test_ept_m2_components.py']))"
```

Historical initial console output, exit 0 (not rerun for this report):

```text
.............................................................            [100%]
61 passed in 1.54s
```

No separate initial pytest process wall duration was recorded. The three M2 files also passed static parsing in the original check.

For the final pure-component run, retain the four environment settings above and add:

```powershell
$env:NUMEXPR_NUM_THREADS='1'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
$env:PYTHONIOENCODING='utf-8'
python -B -m pytest -q -o addopts= -p no:cacheprovider COG/tests/test_fusion_spec.py master_ver/studies/v_production_20261004/tests/test_metrics.py
```

Final output, after the uniform comparison fix, completed runner documentation correction and authorized child-failure fix (67 existing cases + 4 new mocked cases):

```text
.......................................................................  [100%]
71 passed in 0.71s
exit_code: 0
process_wall_seconds: 1.024875
```

Started `2026-10-24T10:03:36.942721+00:00`; process timing excludes preceding hash/static-parse checks. Six pure-component source/test files passed `ast.parse`. Detailed local log: `work/modularization_validation/final_child_exit_20261024T100336942721Z.log`. Essential outputs and hashes are copied here because work logs may be ignored.

Earlier runs: pre-Gemini recheck `67 passed in 0.57s`, process wall `0.890834 s`; first post-Gemini check `67 passed in 0.76s`, process wall `1.092560 s`. The uniform-filter run then recorded `67 passed in 0.71s`, process wall `1.035372 s`. The final 71-case run additionally verifies the completed runner doc correction and four new child-exit guard cases. Docstring-filter self-checks are embedded in existing cases, so the final count is 71, not 69. Repeated runs are not additional scientific observations.

## Equivalence findings

- **COG:** fusion constants, aliases, helpers and their `_parse_model_list` dependency moved to `app/fusion_spec.py`; runner names are re-exported. Pinned-source AST and synthetic checks cover alias/prefix normalization, Cartesian and deduplication order, canonical strings, CLI help/defaults/presets, results and exception messages. Remaining runner executable AST, including preset/split-mode handling, is unchanged except extraction/import changes and the exact authorized `raise SystemExit(1)` described below. No heavy runner or split implementation was executed.
- **Master arithmetic:** float64 conversion, flattening, sigmoid branches, formula/reduction order, conditional clipping and validation/errors are preserved. Tiny-array checks cover exact sigmoid bytes/shape, strided/extreme/nonfinite inputs, and unit-row mean log loss. `PositiveSlopeCalibrator` remains defined in the original `train.py` with unchanged executable AST, preserving its defining-module location. The three old train names are import re-exports; moved functions' defining-module metadata naturally changes.
- **M2 arithmetic/gradients:** initial tests compared full outputs/gradients with frozen pre-split operations, rate-only formula/gradient parity in float32/float64, zero exposure, position validation, masks/empty inputs and exact evaluated frame-rate symmetry. Eval skips unused mark/zone invocations; composer uses that path. Legacy FrameAveraged heads without `m2_rate` retain full-M2 fallback.
- **M2 RNG:** training rate requests still use full M2. Synthetic dropout hooks checked discarded-branch order, RNG state, subsequent random draws and gradients, including both frame calls. Optimization is evaluation-only; no encoder forward or actual training trajectory was run.
- **M2 state:** focused tests retained parameter identities/names/order, state-dict key order/values across calls, and strict in-memory reload. Constructor AST equality independently supports unchanged registration. No historical checkpoint payload was read or validated.

## Documentation is reviewed separately

Gemini's fusion/metrics docstrings were checked separately against parsing/order/error behavior, sigmoid nonfinite handling, unit-row weighting and conditional probability clipping. No production file was edited in this validation follow-up. Numerical/structural tests are not a substitute for prose review.

All production AST-equivalence comparisons now use the same normalization, including the entire remaining runner/train bodies and the calibrator class. On a copy, it removes **only a leading string-expression docstring** from each module, class, function or async function. Every other node is retained: decorators, annotations, defaults, return literals and later standalone string expressions. Embedded synthetic checks cover both whole modules and body lists, prove docstring-only changes compare equal, and reject changed non-docstring strings. Equality does not assert identical `__doc__`, source bytes or all introspection metadata.

The lead/Gemini correction to `_run_isolated_per_model` is present in the final tested snapshot. It describes separate Python processes/state and explicitly disclaims physical RAM/GPU isolation and resource caps; the unsupported address-space-intersection claim was removed. The worker preserved that docstring while adding the separately authorized failure-exit statement. All final AST comparisons ignore only docstrings, including the complete remaining runner body.

## Explicit behavior fix, separate from pure extraction

Previously, failed child return codes were printed but the parent function returned normally, allowing the outer CLI to exit 0. The only authorized executable addition is `raise SystemExit(1)` immediately after the existing failed-model summary. Successful execution still returns `None`; all scheduled children/batches complete before any failure exit, and every original per-child return-code summary is preserved.

Four AST-extracted-function cases mock serial and parallel subprocesses for all-success and mixed failure (codes 3 and -9), with five synthetic children so later batches must still execute. They assert scheduling/wait order, every summary entry, unchanged success behavior and final `SystemExit(1)` after failures. No real child process/model is launched.

The baseline test retains the **whole** runner function. It deep-copies the baseline AST and adds exactly `raise SystemExit(1)` to the existing `if failed` branch after its original summary print, then compares all remaining executable AST. It does not ignore the function or permit other changes. This is an intentional behavior change, not a claim of complete baseline equivalence. Scientific loss/population/equation definitions are unchanged. Tests are frozen at the hashes recorded above after this final run.

## Limits and scientific run identity

Checks used tiny synthetic inputs and one-thread CPU numeric runtimes; M2 also fixed Torch intra/inter-op threads to one. No real training, heavy encoder forward, GPU test, benchmark, raw/cohort/split/holdout data, checkpoint or published artifact was used. Passing tests establish the bounded code checks described above, not measured speed gains, full training-trajectory equivalence, full checkpoint compatibility, scientific replication or historical-run acceptance. Independent review remains separate.

This branch is a **new software revision**. Frozen scientific definitions are preserved, but changed source hashes require **new-run identity**. It is not eligible to resume previous scientific runs and does not establish old-run validation. Legacy entrypoint paths and references pinned to the original base remain valid. No resume was attempted and no protocol, population or old run was relabeled.
