# Independent review: EPT physical root cleanup

Decision: **ACCEPT**

Reviewed the frozen working tree against `611a4a4ef092890fca52e00250608e70bc407e81`. Implementation and review were separate. The reviewer changed only this report and ignored `work/ept-root-review*` evidence; no implementation fix, commit, push, training, benchmark, inference, or scientific report generation was performed. Strata remained off.

## Findings and resolution

No open P1/P2/P3 finding remains in the reviewed scope.

- **Resolved P2 — S3 test repository anchor.** The initial `test_run_s3.py:75` retained the old implementation-depth calculation, producing `c_ept/c_ept/workflows/run_s3.py` and an incorrect subprocess cwd. This was reported immediately. The implementer changed the anchor to the test file's repository root. The independent anchor probe and seven selected import/main/source checks passed; no reviewer implementation edit was made.
- **Justified test correction — shared runtime allowlist.** Baseline and current S3 both contain `from c_ept import features, nll, runlog, store_core, runtime` (`workflows/run_s3.py:149`), and the entire moved runner AST is identical. Its adapters actually use runtime memory scopes, sampler and E6 guard. `runtime/__init__.py` and `runtime/resources.py` are unchanged and do not launch work. Adding this one existing dependency to `LIGHT_C_EPT` and the required-import set fixes a stale assertion. `MODEL_LIBRARIES`, `STAGE_MODULES`, and the prohibited stage/model import predicates remain unchanged; the source test now reads the implementation.

## Specification verification

- **Physical inventory:** 39 root Python files became exactly five: `__init__`, `config`, `legacy`, `runlog`, and protected `cohort`. All 34 removed root paths and their registry targets were checked. Ten real implementations now occupy workflows/reporting/checks; 24 previously grouped implementations are unchanged.
- **Bounded source proof:** seven of the ten moves have exactly equal ASTs. The other three are equal after only enumerated exceptions: the fixed 24-path S1 provenance inventory plus its selection/docstring, and the stdlib package-path import/repository anchor in operator and S4a runner. No remaining AST differences exist. All 24 prior implementation bodies equal baseline after EOL normalization. The 24 relocated old shims equal baseline text and also match exact Git-filtered baseline checkout bytes under current `core.autocrlf=true`.
- **Aliases and discovery:** `_compat` is appended only to `c_ept.__path__`; the real package remains first. No global `sys.path` or meta-path mutation is added. The storage package's empty `__path__` hides duplicate discovery/import namespaces. Both legacy/canonical import orders, parent attributes, shared globals and monkeypatch state, old class references, instance pickles and legacy main dispatch are covered by the passing collection and separate probes.
- **Startup and anchors:** both startup-helper paths stay free of psutil/NumPy/SciPy/Torch; putting the helper in docstring-only workflows avoids the existing eager runtime resource export. Config/legacy pins/runlog retain their anchors. Operator ROOT and runner REPO resolve to the repository via the first package path. No other moved implementation retains a file-relative source anchor.
- **Provenance and guards:** all 24 fixed extraction paths exist and are unique. Independent probes verify every canonical/glue/anchor path is supplied to Git status, dirty paths refuse cleanliness, and Git failure remains unknown. S4 whole-repository cleanliness, input/run identity, E6 and resume contracts have no semantic source change. Scientific definitions, split selection, evaluation and numerical bodies were not weakened by this cleanup.
- **Packaging/CLI:** unchanged setuptools discovery uses `namespaces=false`; the disposable offline wheel includes all 34 forwarding files and canonical modules. Installed legacy/canonical operator and runner help commands run outside the source tree; console metadata and synthetic PowerShell wrapper integration pass. The test wheel substitutes a raising cohort sentinel without reading/copying real cohort source. Removed root filesystem `.py` launch paths are intentionally unsupported; module and wrapper contracts remain. The bounded active-caller scan did not establish external/untracked caller compatibility.

## Standards and working-guidance review

The frozen plan, component guide, current navigation, path registry and lead result summary agree with the implementation. Local link targets were checked. Protected anchor bodies were excluded from content/AST inspection; a separate hash-only check found the protected cohort/policy/holdout source anchors unchanged (23 files).

The consolidated `docs/WORKING_GUIDELINES_KO.md` and root AGENTS preamble preserve the relevant boundaries:

- Strata stays OFF and neither health checks nor fallback permit automatic startup (guide line 10).
- Protected source/data/weights and external-model transmission restrictions remain explicit (24-28).
- Synthetic CPU1/GPU-off tests are distinguished from scientific execution contracts (43).
- Clean dedicated execution checkouts, whole-repository guards, failed-run identity preservation and read-only default Check remain required (51-56).
- G-S4a remains incomplete; S4b/S6/S7 do not auto-start. GPU-first or shorter operator modes require a reviewed plan/config contract and cannot bypass the operator through the runner (60-67).
- Publication remains a bounded source snapshot on existing public ancestry, excluding private history, fixed membership, payloads and credentials; no force push or general hook-disabling permission is created (84-87). This review does not itself perform or verify publication.
- The explicitly authorized Master-derived module-layout exception prevents forcing sync into the derived tree and retains the original workspace as numerical authority (28). The entire original CoG AGENTS suffix is preserved verbatim after EOL normalization.

No global memory/config change was made by this reviewer. No guidance clause grants a new scientific execution budget.

## Independent test evidence

Final frozen-code command, executed through the reviewer's protected-cohort stub and import/open-denial harness:

```powershell
python -B work/ept-root-review-probes.py --pytest -q -p no:cacheprovider --tb=short --junitxml=work/ept-root-review-final-357.xml c_ept/tests/test_module_layout.py c_ept/tests/test_visible_root.py c_ept/tests/test_s4a_operator.py tests/test_project_boundaries.py
```

Environment: fixed `C:/Users/todtj/anaconda3/python.exe`; `PYTHONUTF8=1`, `PYTHONIOENCODING=utf-8`, `PYTHONDONTWRITEBYTECODE=1`, `CUDA_VISIBLE_DEVICES=-1`, native library thread limits and `LOKY_MAX_CPU_COUNT=1`, `MKL_THREADING_LAYER=SEQUENTIAL`, `MKL_CBWR=COMPATIBLE`, `KMP_DUPLICATE_LIB_OK=FALSE`; third-party pytest autoload disabled.

**357 passed in 62.09 seconds in one final run; zero failures, errors, skips, or reported warnings.** XML breakdown: module_layout 52; visible_root 50; s4a_operator 244; project_boundaries 11. Frozen implementation/test/package hashes were identical before and after this run and at final capture.

Separate reviewer probes: **36/36 child processes passed**. Twenty individually cold processes cover both orders of the ten new moves; ten grouped processes cover the 24 older alias pairs in both orders; four check startup/discovery; one checks all 24 provenance paths; one checks S3 test anchors. Probes are independently written, rather than only reusing the implementation tests. An earlier additional focused selection had 81 passes and one reviewer-harness refusal when the overly broad file guard blocked writing the disposable sentinel; the guard was narrowed to real protected source paths and the wheel check passed in the final collection. A preliminary probe setup also incorrectly expected an unimported legacy parent attribute; explicit legacy import corrected that probe assumption. Neither was an implementation defect.

`python -B tools/check_project_boundaries.py` also passed (four packaging declarations, 19 allowlisted AST files); `git diff --check` passed.

Failure history is retained: worker main regression 355 passed/one Windows cp949 decoder failure; worker focused run 33 passed/one stale runtime-allowlist failure; corrected focused reruns passed. The worker's cumulative 357 evidence is distinct from this reviewer's final single 357-pass result; these counts must not be added. Earlier worker/contextmanager and encoding failures remain described in the implementation handoff. Logs and exact final case IDs remain under ignored work paths.

## Limits

Acceptance covers this physical layout, compatibility, documented authorization boundaries and bounded synthetic verification. The full research-data/integration suite and package-wide reader scans that open protected sources were not run. No real input, weight/checkpoint payload or protected split body was inspected. No training, inference, reporting workload, new cost run, Start/Resume, or scientific result equivalence was tested or authorized.

The wheel remains source-backed: external legacy assets/pins, repository config/environment and research inputs are not made self-contained by wheel import/help success. New grouped pickle names require this layout; loading them in older checkouts is not promised. No removed direct-file CLI path is restored. Existing failed-run identity and incomplete G-S4a status remain unchanged. The required public code-only parent is `a90ab4`; this worktree ancestry must not be published. Public history parent/fast-forward and published blob checks belong to the subsequent publisher; this reviewer made no commit/push and did not verify a remote snapshot.

## Frozen hash evidence

Captured at `2026-10-24T21:57:16+09:00`. Baseline and current HEAD are `611a4a4ef092890fca52e00250608e70bc407e81`; reviewed changes are uncommitted working-tree changes. The full **174-path** scoped raw-SHA256/Git-filtered-blob inventory is `work/ept-root-review-hashes.json` (raw SHA256 `df0a230e77670810dbe7d17fbc19c0373642d6bc6f5d942ff6bc8629d5ba79e6`). It contains only scope paths and hashes, never source bodies. The review file itself is excluded to avoid a self-referential digest.

The table below binds changed/relevant scope files and retained anchors to **RAW filesystem SHA256**. These are not Git blob IDs. The manifest separately records actual `git hash-object` identities after repository filters. Git baseline blobs contain normalized line endings; exact raw hash equality with such blobs is not expected. The 24-shim preservation check explicitly distinguishes normalized text and Git-filtered checkout bytes. Hash-only protected-anchor evidence is in `work/ept-root-review-preservation.json`.

| Scope path | RAW SHA256 |
| --- | --- |
| `AGENTS.md` | `673bb0337185365a405f3df0aa3c147f82893ef316e7947d0dbffc041d137754` |
| `MODULES.md` | `230ba1cf6692a755488ca2a258e93767bb28ed4ff3d6fd37d3a1a194114cf02b` |
| `README.ko.md` | `08ada013aa272a0cf8574105feaf3fcc05e2606f7b0eb24ca390f2b88264d5a4` |
| `README.md` | `5459b34beef60a5cb077b56ce802d0302e846e9d54229e82da9056500fe635fa` |
| `c_ept/MODULES.md` | `a9d1ce1e60bef45c1d36ace29aadf423031600384d5f5e10391164fb204d3a5b` |
| `c_ept/__init__.py` | `38ee5093acbcca89ce45d0c8b26b95ada351c219a6907c5aa49e786fee770c0e` |
| `c_ept/_compat/__init__.py` | `fcc66dce9fc0edf51dc143e0bf00b4bf1ea75ee470673dce0e0670742d26777a` |
| `c_ept/_compat/baselines_b0.py` | `4e214b83f563178eb346924e4847db4aca90ae436f73e6699d7d956f40d79c21` |
| `c_ept/_compat/baselines_b1.py` | `67463f155cf55149f8179aa2a427f4341cf2924367c73120bdb2b47ead6ca7d1` |
| `c_ept/_compat/casts.py` | `a2d0f79b63440647f2b77d60402333db50ac5911d3c4d6b35d5e72d79e8063d1` |
| `c_ept/_compat/compose.py` | `ca5749faadc118122d33bafbdfd188d236b3cbf35e67ab07fef8245a286f52e3` |
| `c_ept/_compat/event_rows.py` | `01455a621113be851c4cbe9d9a6c68fd257f3f276c47076e354ac4b48e67c443` |
| `c_ept/_compat/features.py` | `ddfa359396c749b20c2cbe1d45369c5f27b8d949e071ea61fb8497b168999787` |
| `c_ept/_compat/flowprice.py` | `25efc970382497fe9e0fcdb918bb89f5db2e4c042e06872848e6f12d703af049` |
| `c_ept/_compat/frame_rows.py` | `f4aa550b4fb7c2ce68b41b112bba20881b3e8dc18d0e12282f2ffca100b68a25` |
| `c_ept/_compat/gate_s3.py` | `6f0e96b15287b88a1ac89cbbbf5cdd82ac8fbd212e2e767ce53a261f797ddf0a` |
| `c_ept/_compat/head_rows.py` | `2bfac1064d61416520c67dcca797e9d96d54a8fa35126cfa0d23a78d7c69eebd` |
| `c_ept/_compat/heads.py` | `82e43451f3973bab2b6e9b7b77be16592d36336387818585678e1f2d8040c01f` |
| `c_ept/_compat/nll.py` | `1a55bd2038d630f93af0c9000c5f5ef9219b72aaf2ac3ba43914afbf8163218f` |
| `c_ept/_compat/reader.py` | `7659398a178716b9fa7a722dc3c97a9989c566c5de8a8b522ec92ad2b856f777` |
| `c_ept/_compat/report_s1.py` | `761c1b25882d69464b45d2a094ad0532aa5d9cd4941b1472b23a24abc33d3f91` |
| `c_ept/_compat/report_s2.py` | `e2c37166055512a68a7a470622a9680e5eaa970c422bfb6490fc799167ec3f0e` |
| `c_ept/_compat/report_s3.py` | `11fc62576aa86df46d0b5fe692758b979ca4b12ddc81b0d2d98b296392fe73fc` |
| `c_ept/_compat/rulepay.py` | `da90ef2254c54595c8a9a7951cd074004021d273eab4befe5085695f76022940` |
| `c_ept/_compat/run_s1.py` | `8f153108e8230c39a56f9ed949f975bc8179fa4a12383061d8f73b4437860f64` |
| `c_ept/_compat/run_s2.py` | `dcf5065263b2e82091d1ebf176aad411de4778ab3e458c5426346637ac87f5b5` |
| `c_ept/_compat/run_s3.py` | `16621bbee8378ca7d87f640a46d8b5b0c14b0f3d8e6432512acde37bfcf6e6d7` |
| `c_ept/_compat/run_s4a.py` | `5e73017b493675055a7284ddbda35ed9557beab479bf20ed1dc51a1bf0d1b665` |
| `c_ept/_compat/s4a_operator.py` | `54909d9e4af9cbd6c1545a9b1bd23ff6456984079e0e9294f1bb7583397f3838` |
| `c_ept/_compat/s4a_runtime.py` | `ef83f48cc6dbe1b5d0a0e80e358abe01c12e3a2a2804b0e51433fad19f97e58a` |
| `c_ept/_compat/state.py` | `331407d82fd52099327590d381a786bb899b771de68b8ec382f810cd6f01a0e6` |
| `c_ept/_compat/store_build.py` | `0fb12d82d01ed4a514d059cd7cea8d6d3505aa404efffc21e129c4f9924d6aa7` |
| `c_ept/_compat/store_core.py` | `a5c0406dbd011bb6b6f2a00d300a92b47733c7140a79c7f74d2451bc46929762` |
| `c_ept/_compat/symmetry.py` | `ab9361737653be386bef28ae404c7e41333eef08e92f494474cbbc9eb3fb85e3` |
| `c_ept/_compat/targets.py` | `3583a9d1628fa57bfffbb0dc32509349bba4bac92b88c05d409b45da9fa7efbd` |
| `c_ept/_compat/terrain.py` | `c2c1f022104bd7b3a1685b28f63dd2a8ad7b1d3ec78a4b03c38cd69db66315b4` |
| `c_ept/_compat/tokens.py` | `1d01b3e42dac171bffe5257e0c188548031076e5fefde3d6b2ec79606bcd58e7` |
| `c_ept/_compat/towertier.py` | `fc54dcb180ad2e1e48ace1d465bdebd4558ddf9d0463d7e499563d34c0572903` |
| `c_ept/_compat/traits.py` | `3a04af77b88c79b9ee3575cd4a0e5b728536e415d42f0c4fcc8c8d9f6dfe6373` |
| `c_ept/_compat/units.py` | `73b92010ea3f8092e5f4a1a8e105555830400be2d9b6cd4772bb3458af4281f4` |
| `c_ept/_compat/value.py` | `54e3b3c39dcdc6b7ee095d82d85d69ee563f9f33f707df16743768386f869583` |
| `c_ept/checks/__init__.py` | `a0ebef513cf12b9e802e76449d0b086842b08dce32198649e1987808642cfb31` |
| `c_ept/checks/common.py` | `3d58e9408979fd8bd190a89017f192dd6b787e56fee4af877cacda3792c80d81` |
| `c_ept/checks/explain_mismatch.py` | `bd771bb14cf7e022d71b4d2dfae72f2191d790adc03952397cfc693a69d9eab5` |
| `c_ept/checks/gate_s3.py` | `a67ed9a59425d66ad439b87b9de05fb8183a00fe7baa422d1c00d84d566a2241` |
| `c_ept/checks/measure_levels.py` | `889e3fde12977e64f628f7487f4f9d51f89e7d93be1d43f1e237b5a450523c6e` |
| `c_ept/checks/measure_levels_wide.py` | `4580bd1b79a02b10b4f6fbec10a366feec202f511f5f5ec9ddd4bf4dd92a9099` |
| `c_ept/checks/sweep_s1.py` | `82e3d8257ef9f086b58abbdbaba61be609d5e4ce71ca7d99d9506ab4f5da49c1` |
| `c_ept/checks/sweep_state.py` | `1f528fc25eb3dfbd7eb68cfb740b3de4f0e75be7bf1915f7781adf5d1be6f784` |
| `c_ept/cohort.py` | `d6ddc46320d609cfe9c562586d39321b17f5e0c1119fac8c198032ebcebaa808` |
| `c_ept/config.py` | `d4064dca3e0483ddeddf80d7dc4b68cc31878c2ff254472ef11e7490683d7ca5` |
| `c_ept/data/__init__.py` | `c580bc650b9c209be0cc6e4e1504a03c7adc3de59300dcc73607e07b9eada08d` |
| `c_ept/data/casts.py` | `22674e1355b9b819fa40d898fed10eaab00c998cbc3e734b74e7fe1295d435ed` |
| `c_ept/data/event_rows.py` | `b991de70fa30bee2eba2cdf4140ed3b004d3b65693f48f3ff8684d339919e027` |
| `c_ept/data/frame_rows.py` | `6b0239a2cc7ff918fcb9a0df4a56328e9d49cd389aa9d9ae4cb7d66e1759a1a7` |
| `c_ept/data/head_rows.py` | `4dec9dcd902ede72bf575ae9defac116df6a4c9093abe7b02de356719c8c6d30` |
| `c_ept/data/reader.py` | `9d6ec393d7c680c83d936516d9ee29c91fa71d8246b77dfcac6689682a7ba10f` |
| `c_ept/data/store_build.py` | `d5742444f4ab9865a3561a883c0856bbc75ee3cc877ca1a151184116816dbb64` |
| `c_ept/data/store_core.py` | `564d6c3fd6967b84fd1c257c40c85a7735fd436d7550a33cd3bb6a0b3e984fc6` |
| `c_ept/domain/__init__.py` | `ee0cb261d8cab686f4f73ebe6311036434c6af7c9c3e860036d217161cb7bd6a` |
| `c_ept/domain/state.py` | `46b4accc337a2ec5fa76b588f5f2ef215492ecbd8b6643360a189599336c36c9` |
| `c_ept/domain/symmetry.py` | `b110737754bd2c519ca2d90b615ad4a1a19a7f3a3adca5fa718837e461b67a5b` |
| `c_ept/domain/targets.py` | `77b0082ed8c1bd57df379de098895310e40aa74cfe3d2c2fdc3c26a34aa0848e` |
| `c_ept/domain/units.py` | `a42310d610283e7751f21a828df60081c16bb8cd394808f8c4251c2093305ffe` |
| `c_ept/ept/__init__.py` | `eb7e309ec4ca766a53b8cff82a894837e00ea8443bc7322720529c84a45e6ca9` |
| `c_ept/ept/checkpoints.py` | `6ca70b7dddbeca3f1e83481cded9270f7384c971d8f7d7e9fde60336ad6d952e` |
| `c_ept/ept/compose_heads.py` | `e620d55d488ca43df84d6ac0c219803ef413e52a7ad4ae0408fdc20c61d6091c` |
| `c_ept/ept/data.py` | `7621236fe6a1c6edced346ade6e6e70d025b4347bc3cb9e173fbec0b55b3f120` |
| `c_ept/ept/encoder.py` | `f026808922071a27f7e0bd88abf67e18687d2f7e7430c838f22c70a67a967bca` |
| `c_ept/ept/heads.py` | `c9484a7fde8b1a27927aac4e9139c8625d23da235af3690cabc8730e7df17322` |
| `c_ept/ept/loss.py` | `3e15e715eb3c18da933749bc352225e4bb9241934e89fab51c1623a683c9bd87` |
| `c_ept/ept/scales.py` | `f2597b4b8d320d10b6c2003fde1329a28f3cf3349f8c8068c1ea7a3aaac39bd8` |
| `c_ept/ept/train.py` | `9d53ceba09788cc562b69740f6b1bfc9c06b771d3c5992b013c8b270f0ef61aa` |
| `c_ept/ept/training_step.py` | `584bb824ce7f0e62c97e82bf75aa00c52c419680a34d735335642023454a434a` |
| `c_ept/legacy.py` | `91f1adb4eed7a1f822c5a3c7aa3636afc1a934a16ea7ace282c0febaf967df07` |
| `c_ept/legacy_pins.json` | `616c05be297d5b3d003d2572f2b12b594427c2dd0d7b03fb8175be9a6e4b95c5` |
| `c_ept/models/__init__.py` | `25be85082ff01a7ec080e8c9162d992e83c6f7564a7094a7e7adbd47d2f17f99` |
| `c_ept/models/baselines_b0.py` | `32924ca3897fe77e4fd8dd4e7b07e7321005745bfd0d830e40cceeeab96a40dd` |
| `c_ept/models/baselines_b1.py` | `9ca557d5a93151f3a60f4813390dd91c12881c33df43d80303dfb84297fbd979` |
| `c_ept/models/heads.py` | `675bb60f16ac092d0a0dc53f369644e40447be22b48e2c2b8c36597a0425e872` |
| `c_ept/models/nll.py` | `f9982abf2178de6718be99f4de8e37bfd94d1ab35db8bc9bbd094a2dcbea8d13` |
| `c_ept/preprocessing/__init__.py` | `7195525f3dbb731a4432c04bad63ff98836e0fc02b210cffdbe958f7edccdad2` |
| `c_ept/preprocessing/features.py` | `b3d6464e4e61de7e525e75b62612b92f4d603d752deb678d7d537fc34c0798cf` |
| `c_ept/preprocessing/terrain.py` | `fe1015ad944d5704f2e15c7cb9e1b362644ada164af3e6d82a59a72c5639fca5` |
| `c_ept/preprocessing/tokens.py` | `12ed5fbdc0b721e2222e99664284d4a99675a3106e929b2529f3f91d405ec141` |
| `c_ept/preprocessing/traits.py` | `0423009513bd3a8bd37fd231fcb6da01c126cb73e00c93c1f6a4b6997be6a36e` |
| `c_ept/reporting/__init__.py` | `5bc79824d7e585afe217dba390a565d5ee3b5371dfd5bccf2a145cf97e0af2c3` |
| `c_ept/reporting/report_s1.py` | `282fef60ffbeadf5b4dfe5a0bc815a6f5e92502e39b49aaee9724cee37c973c4` |
| `c_ept/reporting/report_s2.py` | `7bcf69fb761184a24dc46f619b94ae434dafb9dc965253a37817639fdbb23721` |
| `c_ept/reporting/report_s3.py` | `054c0be386b4758e2a29cbdaf9e7130cf574d0c63d8f4e131b9e05f48e21d44e` |
| `c_ept/runlog.py` | `293f16b74792ef6b47e5ea2257c7ce11628aee2d61d0923e2968750d1574be44` |
| `c_ept/runtime/__init__.py` | `5150a2e6e5b2f8c724052eb04109045bbcfd5e5d1640df073f3cfe3b8d267e00` |
| `c_ept/runtime/resources.py` | `c9bfe82bf58e20edbdc9b60bcdcf549b1a5cc2f2257d04ceded8b79d3921a14b` |
| `c_ept/tests/test_reader.py` | `a977d5bb884fa88ee5657a6024a98e6a023d0a1631ee384f776881fc06c96618` |
| `c_ept/tests/test_run_s1.py` | `0b870f2c1742740708f742f910a973b93d6107cddea63de2c7689d7f343bfe2a` |
| `c_ept/tests/test_run_s3.py` | `5adbe1ee4eaa5574bf41cd74ffd1101fd1933df5cc08832e9fd2970534b3f4b8` |
| `c_ept/tests/test_runtime_resources.py` | `ecabe3b4568e914b93c661220f512daadef62e8584107f09acf7e1b384eaa828` |
| `c_ept/tests/test_s4a_operator.py` | `49188e1041ea404af42fb529b3a4c8318a7432e7275b58562d65ef0b5fbcf426` |
| `c_ept/tests/test_store_build.py` | `f90bba101e4aee184f2b344e4268241f6e8b343834151c44f59058fecaf255bb` |
| `c_ept/tests/test_visible_root.py` | `9cd63d52e199de39dcee90ca318cb097561cf28640cea35689234076aa45fb5f` |
| `c_ept/valuation/__init__.py` | `80efa0fa558bb70aa0df8dc05d76c18e4b4e32f98bd43c9acfde38bab081c837` |
| `c_ept/valuation/compose.py` | `af624793716a0b5b318c0f4fa601d47d9cb93c5f9ebc60c203b05714b51f1ca8` |
| `c_ept/valuation/flowprice.py` | `80ee9c4909d0cb7362b9ece02f975a50b414ccd8922fef376a056b258b5104f9` |
| `c_ept/valuation/rulepay.py` | `3c426b547234c53be537bb889c34886a1559fc2617e3467193f59057319fb8dc` |
| `c_ept/valuation/towertier.py` | `c2972697b55d0127642bf4f58321539b57290edd47e529eaf21a177763e02ea1` |
| `c_ept/valuation/value.py` | `545cc1ff8e1f83631e58bb406e438296c73ad7d1cf01df9e0ecf830796ef5b05` |
| `c_ept/workflows/__init__.py` | `c899b723e38eac7aeafda4b2b1107d5030e9568a15a85215a91e33cff037b772` |
| `c_ept/workflows/run_s1.py` | `1bea46d6047a9e21be53dd158d950ae7030d210c7415bbd2f67ad37d3752a348` |
| `c_ept/workflows/run_s2.py` | `6e873b32fa085b117e6294e9c7bb1126a396461e12e4266d8e2218a83cdeae6e` |
| `c_ept/workflows/run_s3.py` | `7a09bb81b27fd2e4b2ac0e3ebcb2e93c5a8af6c86a39aaa2570391758b4f32ed` |
| `c_ept/workflows/run_s4a.py` | `fe50a5daef6dae84a892dafef78d807d23a5ee25ba894e836f1a327d60f20db2` |
| `c_ept/workflows/s4a_operator.py` | `6fc6a1d0dde81183c2b03caad9b76a3769cf42f2f0b83628e84767828516162d` |
| `c_ept/workflows/s4a_runtime.py` | `31822adf644e8cbefc202f3d032afc630e87406774fdd8ae3ba7b8686e9b6b3f` |
| `docs/PUBLICATION_20261024.md` | `fce666cf4699b77547e956c67fa366972fb5ecac3180062550e0cc5893afc0c0` |
| `docs/S4A_NEXT_STEPS_KO.md` | `cea4b568ce78691df7abfb992de81c4c5510774ffaf61a8c4e40a4c9a75a2c85` |
| `docs/WORKING_GUIDELINES_KO.md` | `147ed525af58a40ea6d1088e105669b12fd1e0a9ae396e7c688f1288248d03de` |
| `docs/module-layout-20261024.json` | `c8ac87a18cbf2bd7f45074db11918593497df111f56a8da4d6e6051877ce4d27` |
| `docs/superpowers/plans/2026-10-24-ept-root-cleanup.md` | `a6771915c328e04dc9a69eae68655d97553b787da58191255beea4c6bf4cb2c1` |
| `docs/superpowers/reports/2026-10-24-ept-root-cleanup-result-ko.md` | `d7992e02b919da704cf0e641ab3d51e28223a2181022d48b65dd1ec66793d90e` |
| `pyproject.toml` | `5e6f340b29b636500729fd1f5824dad009e394133526f242f6c6e1c634e1c424` |
| `run_s4a.ps1` | `1ca4adbc2e12e7aeb93bb4de46e2bb1fdc13fcbb9be2db83707664979d3c6399` |
| `tests/test_project_boundaries.py` | `5c76ac3d7edf71481c2e26e3e58f01f3adae46fed81cbcd490cf5a5d5ff417bb` |
| `tools/check_project_boundaries.py` | `e8e7a118d9384976b7e74ea951a92d11aa423f5a6a3f25bc3b3da2493e697041` |

Local evidence: `work/ept-root-review-final-357.log`, `work/ept-root-review-final-357.xml`, `work/ept-root-review-probes.py`, `work/ept-root-review-probes.json`, `work/ept-root-review-ast-proof.json`, `work/ept-root-review-preservation.json`, `work/ept-root-review-before-tests-hashes.json`, and `work/ept-root-review-post-tests.json`. Implementation evidence was read as evidence, not substituted for independent verification.
