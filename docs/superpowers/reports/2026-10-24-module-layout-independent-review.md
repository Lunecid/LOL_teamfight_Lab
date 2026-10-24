# Independent review: physical module layout, 2026-10-24

Decision: **ACCEPT within the reviewed physical module-layout scope**. The prior P2 CoG namespace regression is resolved by the frozen correction and independent delta verification. No actionable findings remain in this reviewed scope. This is separate from author self-verification; it does not certify the full research projects, performance, or numerical results.

## Resolved finding and correction verification

The initial frozen helper hardcoded `analysis.<leaf>` and `sys.modules['analysis']`. Independent baseline/current source snapshots demonstrated that four previously valid `COG.analysis.*` imports failed with KeyError and that mixed-root imports replaced an already-held source-backed alias. That evidence supported the original REQUEST_CHANGES decision.

The corrected `COG/analysis/fight_boundaries/__init__.py:10-16` derives the legacy name from the actual canonical name and obtains its parent package from that derived name. It preserves historical function/class metadata separately within `analysis.*` and `COG.analysis.*`. It does not unify those namespaces or overwrite one namespace's saved aliases/types/globals with the other's.

Delta review found exactly four changed inputs from the prior 118-file inventory: the CoG registration helper, organization test, module guide, and author handoff record. All 39 canonical implementations, every legacy shim, Master/EPT sources and tests, lead-owned integration files, and the finalized historical-manifest clarification in the publication note retain their previously reviewed raw hashes. The author's frozen 17-file CoG source/document SHA-256 inventory matches actual worktree bytes.

The reviewer reran only the four corrected CoG organization tests: all passed in 4.760 s, including eight fresh-process combinations (source-backed alone, qualified alone, mixed source-first, mixed qualified-first; each legacy-first and canonical-first). The seven algorithm fingerprints and seven exact legacy-alias ASTs still pass unchanged.

Eight separately authored fresh-process probes also passed. Unlike the author's per-leaf interleaving, these finish one namespace before loading the second. They recheck all saved references after both namespaces and transitive imports load, derive the expected top-level symbols from the unchanged canonical ASTs, and verify every symbol rather than selecting only symbols that already have expected metadata. Checks cover all seven modules per root: old/new identity within a root; namespace isolation; canonical __name__, __spec__, __package__ and file; parent bindings; function globals; 48 historical function/class roots per namespace; constructed old GLOBAL pickle lookups; new type serialization using namespace-correct historical roots; and three synthetic dataclass instances per namespace with protocol 0 and the highest protocol. No historical payload was opened. Both roots remain separate module/type identities, and importing only one does not register the other. Search paths/hooks remain unchanged and torch is not imported.

The original qualified-import and mixed-alias failure conditions no longer reproduce. **P2 is closed.** The expanded test and guide were reviewed for this namespace contract; no implementation edit was made by the reviewer. Master/EPT tests, scientific-unit cases, reports, training and research results were not rerun for this delta.

## Exact scope and fixed inputs

Implementation baseline: `d9d7ac8dbd5c389c8bf70cbd4c0ed1ee8be99e4a`. Declared public parent: `938574066dc0763a42748617d7006cb42b82538c`. The public parent was not used as an algorithm-equivalence baseline and no remote repository or publication was inspected.

The 39 mappings in `docs/module-layout-20261024.json` resolve to 39 unique existing canonical files: EPT 24, CoG 7, Master 8. The eight Master mappings agree exactly with `master_ver/module-layout.json`. The hash inventory below identifies the final reviewed RAW worktree bytes, not a future commit or normalized clone. It now includes the two completed acceptance-status documents. It includes source/alias files, package helpers, changed tests and fixtures, scope documentation, and the three handoff records. Handoff claims were checked independently where stated below; reading a handoff is not itself verification.

### Master

Reviewed all eight old `master_ver/scripts/report_*.py` entry points and their exact eight mapped canonical implementations under `scripts/reporting/`, the package initializer, marker, module guide, organization test, and tracked baseline hash fixture. The moved canonical bodies were parsed locally in full for comparison, without running report mains.

Independent comparison against actual baseline Git blobs verified all eight baseline source SHA-256 pins, all eight complete baseline AST pins, and the legacy provenance-assignment AST pin. Thus the tracked fixture was not accepted merely because current code agreed with a fixture produced by the same worker.

Canonical ASTs equal those actual baseline ASTs after precisely the declared exceptions: one top-level `ROOT` expression per implementation changes `parents[1]` to `parents[2]`; validation-suite `main` adds exactly the eight canonical source paths immediately after the original nonrecursive validation-source inventory. Existing result/model/input expressions and later hashing behavior remain present. The synthetic inventory test executes only the two selected statements against a temporary scripts tree; no real output, model, input, or provenance record is enumerated or opened.

The actual wrappers and dependency-stubbed canonical imports pass both import orders for `master_ver.scripts`, `scripts`, and bare modules with `scripts/` as the import root. Additional independent probes, beyond the shipped assertions, confirmed canonical `__spec__.name`, `__package__`, parent-package bindings, and `main.__module__` in all six root/order combinations. Shared globals and ROOT remain consistent.

Synthetic CLI tests execute copies of the actual wrappers against inert canonical fixtures. Argument values after argv[0] and cwd are preserved; runpy intentionally substitutes canonical argv[0] during execution and restores the caller's value afterwards. This is documented behavior, not byte-for-byte argv[0] identity. The canonical source main guards remain present and match baseline. Actual reports were never executed.

### CoG

Reviewed the seven mapped old aliases and canonical implementations under `COG/analysis/fight_boundaries/`, their registration helper, organization tests, module guide, and the two existing package initializers relevant to repository-qualified imports.

Independent comparison against actual baseline Git blobs verified all seven complete AST fingerprints after exactly six absolute-to-relative sibling-import changes. Each canonical module adds only the exact helper import and registration call. The tracked fingerprints match those independently reconstructed baseline ASTs. Each old path has only the documented alias AST.

Within the intended `analysis.*` root, both import orders share module objects, parent attributes, canonical files/specifications, and function globals. The helper preserves historical metadata for locally defined top-level functions/classes. The tests exercise 48 historical globals, constructed old GLOBAL pickle lookups, new serialization roots, and three synthetic dataclass round trips without opening old payloads. The correction-specific qualified/mixed coverage above closes the original namespace defect; the initial source-backed checks alone had not established that compatibility.

### EPT

Reviewed all 24 mappings in the global manifest, the 24 exact root alias ASTs, all five new docstring-only package initializers, the new migration test, module guide, and the diffs in eight existing test files: `test_flowprice.py`, `test_reader.py`, `test_rulepay.py`, `test_state.py`, `test_store_build.py`, `test_terrain.py`, `test_tokens.py`, and `test_value.py`.

Independent full-AST comparison against actual `d9d7ac8` Git blobs passed for 24/24 moved implementations. Their source texts also match after line-ending normalization. Pre-move worktree byte identity is an author-reported check; the reviewer independently pins the present raw bytes below and verifies baseline source semantics/text. No moved implementation uses `__file__`, `__package__`, or `__spec__`. The five group initializers contain only a string docstring. Root aliases exactly import their mapped canonical module and assign it into sys.modules; the flowprice alias additionally dispatches its main guard.

The new tests cover all 24 pairs in both import orders, parent bindings, canonical metadata, function globals after inspect.unwrap where appropriate, old type GLOBAL lookups, canonical type pickle round trips, synthetic old/new instance payloads loaded in a fresh interpreter, a real small monkeypatch execution, and the inert flowprice dispatcher. EPT intentionally keeps new class metadata canonical and makes new pickles depend on the grouped layout; the guide states this direction rather than promising loading by old checkouts.

The eight existing test diffs were reviewed by predicate, not assertion count. Source inspection now follows imported canonical `__file__` values; runner/check paths retain their existing anchors. Rulepay's exact newly-imported-module allowlist adds only `c_ept.valuation` and `c_ept.valuation.rulepay`. Reader package scans anchor to retained config's package root rather than the newly nested reader directory. Its raw-location exemptions remain exact relative-path equality and add only `data/reader.py`; there is no suffix or substring exemption. An independent synthetic tree confirmed root reader/config and canonical data/reader are exempt while `sub/reader.py`, `sub/config.py`, and `nested/data/reader.py` remain offenders. This probe extracted only the relevant scanner functions/constants and operated entirely on temporary synthetic source files. No real package-wide protected-source scan was run.

### Lead-owned integration and documentation

The sync guard executes before source enumeration for both `master_ver --apply` and `all --apply`. Its three synthetic tests inspect that early refusal, permitted dry-run/other-target modes, and an unmarked mirror. Protection relies on the marker file; this review does not claim safety if that marker is missing or replaced. No sync command was executed.

The static checker selects `c_ept/valuation/compose.py` and directly reads/parses that source through SOURCE_MODULES, so it follows the canonical implementation rather than checking the old shim alone. The full checker was not executed, because this review kept its source reads within the authorized scope.

The global mapping, root module guide, bilingual README files, common-training README, and root `.gitignore` addition were reviewed. The referenced publication note was also read for scope/interpretation limits. `/work/` is a root-only ignore addition; it does not itself untrack existing files or prove a future publication payload clean. The navigation guides describe physical responsibilities, retained dated studies, and source/pickle compatibility limits without claiming regenerated scientific results. Historical hash-bound runs remain subject to their original source/resume contracts. Common training retains its bounded sibling workflow. The publication note linked by both README files is present at final inspection. Its source-backed runtime and private-history test caveats were read; historical publication/payload assertions were not independently audited here.

## Verification actually executed by this reviewer

| Check | Result |
| --- | --- |
| Actual fixed-Git-blob AST/source/fixture audit | Master 8/8, CoG 7/7, EPT 24/24 passed after only declared exceptions |
| Master organization suite | 8 tests passed, 2.567 s |
| Additional Master spec/package/parent/function-metadata probes | 6 root/order subprocesses, each covering 8 modules, passed |
| Corrected CoG organization suite | 4 tests passed, 4.760 s; 8 fresh-process namespace/order combinations |
| CoG namespace/pickle delta probes | 8 separately authored fresh-process combinations passed; original qualified/mixed failure conditions resolved |
| EPT migration-only pytest file, with --noconftest and external plugins disabled | 52 tests passed, 28.43 s |
| Independent reader exact-exemption synthetic tree | Passed; nested lookalikes remain flagged |
| Relative navigation destination checks | All relative destinations in both README files and four module guides exist at final inspection |
| Reviewed-byte delta and freeze check | Exactly the four declared CoG correction inputs changed; all 39 canonical bodies unchanged; 17/17 author frozen pins verified |

The user relayed a separate lead precommit check: 113 immutable code/document files staged, all 99 staged Python files parsed, and `git -c core.whitespace=cr-at-eol diff --cached --check` exited 0. The lead reported no source-byte changes and deferred the three status documents until reviewer finalization. These are lead-provenance integration results, not independently rerun reviewer checks; staging and remote publication were not audited by this reviewer. No commit or push was reported.

The initial review ran 64 organization/migration test cases, plus additional focused probes. This correction review reran only the four updated CoG organization cases (eight fresh-process scenarios), and ran eight separately authored namespace/pickle probes; these reruns are not added to a misleading distinct-case total. The author's EPT 71+163=234 distinct bounded cases, 52 portability cases, CoG seven small scientific-unit cases and Git-free portability checks, and Master's public-export checks were read as handoff evidence, not claimed as independently rerun. Existing development-history/private-baseline tests and external-data/golden integration tests remain limited by their documented prerequisites. This review does not endorse the entire repository as error-free.

All reviewer execution used one-thread native-library settings, bytecode disabled, and CUDA hidden. No Strata/model call, network operation, scientific experiment, training, benchmark, report main, real-data enumeration, historical payload/checkpoint load, protected cohort/holdout/split body read, external original Master access, or live Task8 access was performed. Scratch outputs were kept outside the repository. Reviewer repository writes are limited to this requested report, `docs/module-layout-validation-20261024.json`, and the final acceptance paragraph of `docs/superpowers/reports/2026-10-24-module-layout-result-ko.md`. No implementation edits, commits, push, or merge occurred.

## Acceptance boundary and source transport

The 39 mapped moves and reviewed compatibility/integration changes are accepted at the final raw-byte inventory below. No open actionable findings remain after the frozen CoG P2 correction. The two authorized status documents now record the same scoped acceptance and actual independent verification. The previously validated Master/EPT conclusions carry forward because their reviewed bytes have not changed.

Acceptance does not certify whole-project correctness, scientific-result equivalence, operational resumption of historical hash-bound runs, or the remote publication payload. After any further implementation change, freeze and review the corresponding delta before carrying this decision forward.

This inventory hashes RAW bytes in the reviewed worktree. Git core.autocrlf may normalize line endings, so identical raw SHA-256 values are not promised for a fresh clone or remote checkout. The publisher will verify committed Git blob identities in the public tree and retain algorithm AST checks; that publication transport verification was not performed by this reviewer.

## Reviewed file SHA-256 inventory

Paths are relative to the isolated repository root. This table pins 120 reviewed source/document/test/handoff inputs, including both completed status documents. The report does not hash itself. Hashes identify bytes and do not by themselves establish scientific equivalence. Existing package initializer/conftest entries were examined only for relevant namespace/test-safety context; no payload fixtures were executed.

| Reviewed path | SHA-256 |
| --- | --- |
| `.gitignore` | `0566cf96506294c8df5c14b98c08ff7c9f061f9fbf02a5d9f91b0eaf01d33805` |
| `COG/MODULES.md` | `a8350664fdda880a7d2e4f9447d5b4f185a9085ea15e2e843ba1c56e22eaf601` |
| `COG/__init__.py` | `21e0441edb1d6db3c7bb945d4ba3c06bb188a3bcf84a4ee084d19a3cfb05cd10` |
| `COG/analysis/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `COG/analysis/ability_range.py` | `7c6b73e41aa68fba38f0541dbb8e997654d8d376c23f851eacc5f8ee42fcc78d` |
| `COG/analysis/boundary_spec.py` | `7e9252a2ff1a817d3b0abf1daeb816c8684f50d42de9f8483fe3a68fd43783d1` |
| `COG/analysis/fight_boundaries/__init__.py` | `35f4c3282257131fba864d9af9aa3f11c15c3959de63652f4a8afa3cd0c656cd` |
| `COG/analysis/fight_boundaries/ability_range.py` | `027344aa0b37ce5acd6deb1f51aeed63c3f9d9c01c47822d5001bd70322f3b81` |
| `COG/analysis/fight_boundaries/boundary_spec.py` | `eac69c13ffa9ea619cba78f02dd0e19019a9f170adcbfb39f7c8f236db42526e` |
| `COG/analysis/fight_boundaries/fight_boundary_pipeline.py` | `5ffe124dc398c4066a5ae3a8500a86764f3870ec206e6f6ac3a798b7641a5392` |
| `COG/analysis/fight_boundaries/kill_pairs.py` | `fd2c7f89015652fab3ed9a09b73671af46da09a9e5bed86235b7edc7501103de` |
| `COG/analysis/fight_boundaries/map_regions.py` | `88dc89c0118f954cc2174a87412185da0bfa66fa1163a0ac9108bb8d01c56350` |
| `COG/analysis/fight_boundaries/spatial_boundary.py` | `126b0873bd0d3c841c485476f54fdc2279c3d9e3525ce9aeeae842d0ee40bfd1` |
| `COG/analysis/fight_boundaries/temporal_boundary.py` | `2dd33a7530d5cf2d5cc0e67a0ddcfd03b73e6c9346a1a6d03f3e784a2529c511` |
| `COG/analysis/fight_boundary_pipeline.py` | `5c800fee27b449382c00fdeaa7d165aa74db75fbf463a68648115f7db4934eed` |
| `COG/analysis/kill_pairs.py` | `d267e261a94a8a302289d3529ca97bac8aa478d3565557224a6abb47db1f0b4a` |
| `COG/analysis/map_regions.py` | `33f3d82eafad4c2434e5b53759a2cdccc33e26b2339ab8e35b244ef8b4d30181` |
| `COG/analysis/spatial_boundary.py` | `35833f098d786e644674023c7744e4826456df773dec44c0abc8d9c5f90310e7` |
| `COG/analysis/temporal_boundary.py` | `ca30568625baf973231fb079045bcfaccd9dfcec520ca75039618a5b87fbf89f` |
| `COG/organization_tests/test_fight_boundary_layout.py` | `ec166a133da83282ce0c27ffa303ac94fcbac1d4f246f0425a9137cc258b5469` |
| `MODULES.md` | `497df4124dfc1d9f0f701803dcf90f7d3cd6099bebf6804444c7621fb2b0533d` |
| `README.ko.md` | `5f9de6480e3777b343bc4c6a4e0bc1bc710bb51883c0c40b8cba2f349b861ff4` |
| `README.md` | `306b5c6949b4001a5c71852425c5026b3026365d5056caeef71cf53ac200eb73` |
| `c_ept/MODULES.md` | `820bf67fe0f71c9f30192b02372406cfa85b998fffcfac660797b05d6d334b34` |
| `c_ept/baselines_b0.py` | `4e214b83f563178eb346924e4847db4aca90ae436f73e6699d7d956f40d79c21` |
| `c_ept/baselines_b1.py` | `67463f155cf55149f8179aa2a427f4341cf2924367c73120bdb2b47ead6ca7d1` |
| `c_ept/casts.py` | `a2d0f79b63440647f2b77d60402333db50ac5911d3c4d6b35d5e72d79e8063d1` |
| `c_ept/compose.py` | `ca5749faadc118122d33bafbdfd188d236b3cbf35e67ab07fef8245a286f52e3` |
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
| `c_ept/event_rows.py` | `01455a621113be851c4cbe9d9a6c68fd257f3f276c47076e354ac4b48e67c443` |
| `c_ept/features.py` | `ddfa359396c749b20c2cbe1d45369c5f27b8d949e071ea61fb8497b168999787` |
| `c_ept/flowprice.py` | `25efc970382497fe9e0fcdb918bb89f5db2e4c042e06872848e6f12d703af049` |
| `c_ept/frame_rows.py` | `f4aa550b4fb7c2ce68b41b112bba20881b3e8dc18d0e12282f2ffca100b68a25` |
| `c_ept/head_rows.py` | `2bfac1064d61416520c67dcca797e9d96d54a8fa35126cfa0d23a78d7c69eebd` |
| `c_ept/heads.py` | `82e43451f3973bab2b6e9b7b77be16592d36336387818585678e1f2d8040c01f` |
| `c_ept/models/__init__.py` | `25be85082ff01a7ec080e8c9162d992e83c6f7564a7094a7e7adbd47d2f17f99` |
| `c_ept/models/baselines_b0.py` | `32924ca3897fe77e4fd8dd4e7b07e7321005745bfd0d830e40cceeeab96a40dd` |
| `c_ept/models/baselines_b1.py` | `9ca557d5a93151f3a60f4813390dd91c12881c33df43d80303dfb84297fbd979` |
| `c_ept/models/heads.py` | `675bb60f16ac092d0a0dc53f369644e40447be22b48e2c2b8c36597a0425e872` |
| `c_ept/models/nll.py` | `f9982abf2178de6718be99f4de8e37bfd94d1ab35db8bc9bbd094a2dcbea8d13` |
| `c_ept/nll.py` | `1a55bd2038d630f93af0c9000c5f5ef9219b72aaf2ac3ba43914afbf8163218f` |
| `c_ept/preprocessing/__init__.py` | `7195525f3dbb731a4432c04bad63ff98836e0fc02b210cffdbe958f7edccdad2` |
| `c_ept/preprocessing/features.py` | `b3d6464e4e61de7e525e75b62612b92f4d603d752deb678d7d537fc34c0798cf` |
| `c_ept/preprocessing/terrain.py` | `fe1015ad944d5704f2e15c7cb9e1b362644ada164af3e6d82a59a72c5639fca5` |
| `c_ept/preprocessing/tokens.py` | `12ed5fbdc0b721e2222e99664284d4a99675a3106e929b2529f3f91d405ec141` |
| `c_ept/preprocessing/traits.py` | `0423009513bd3a8bd37fd231fcb6da01c126cb73e00c93c1f6a4b6997be6a36e` |
| `c_ept/reader.py` | `7659398a178716b9fa7a722dc3c97a9989c566c5de8a8b522ec92ad2b856f777` |
| `c_ept/rulepay.py` | `da90ef2254c54595c8a9a7951cd074004021d273eab4befe5085695f76022940` |
| `c_ept/state.py` | `331407d82fd52099327590d381a786bb899b771de68b8ec382f810cd6f01a0e6` |
| `c_ept/store_build.py` | `0fb12d82d01ed4a514d059cd7cea8d6d3505aa404efffc21e129c4f9924d6aa7` |
| `c_ept/store_core.py` | `a5c0406dbd011bb6b6f2a00d300a92b47733c7140a79c7f74d2451bc46929762` |
| `c_ept/symmetry.py` | `ab9361737653be386bef28ae404c7e41333eef08e92f494474cbbc9eb3fb85e3` |
| `c_ept/targets.py` | `3583a9d1628fa57bfffbb0dc32509349bba4bac92b88c05d409b45da9fa7efbd` |
| `c_ept/terrain.py` | `c2c1f022104bd7b3a1685b28f63dd2a8ad7b1d3ec78a4b03c38cd69db66315b4` |
| `c_ept/tests/conftest.py` | `b29f165cc4210800472733ae4ccb50fc24c1c6dc0785c1b1d5b31b97d161aaef` |
| `c_ept/tests/test_flowprice.py` | `17176a179caf65e4373285b2b16cf23e24a3773a30a7b50a5b80ff1d67b60885` |
| `c_ept/tests/test_module_layout.py` | `854a24b68b544843affb1196a92dec37121c50cc6938e3966bd7f0e0f541f8bf` |
| `c_ept/tests/test_reader.py` | `73a1e40d51c2799b084c5aa8d5440bf5f3cbd3ac3b37b21f50c1b5a0addad2fe` |
| `c_ept/tests/test_rulepay.py` | `22d666db7488af198457fac1512d995bed37291d00a5d66fdc6b4a9125db3f66` |
| `c_ept/tests/test_state.py` | `28d7e35e08a63595eb893e37a03ab6aca10408c401e4c3a44c5fc0c0e53d159a` |
| `c_ept/tests/test_store_build.py` | `237849b4d11e4c27edab24a9599fe658a29673ea533f11677dd6590de88330f0` |
| `c_ept/tests/test_terrain.py` | `3b48f78e5a9ebd089d6535c71cbce67da7cc4a25f6b78f345b98106f2c44ed53` |
| `c_ept/tests/test_tokens.py` | `9ee7c09ec85f7366571ea8fdd9ce7a5f80a44daae977d21f6e33e5b3e50c922a` |
| `c_ept/tests/test_value.py` | `c1a385f4390b7d0faf4770a97849326695223ed8c8c36949d9031b8f1ebff79e` |
| `c_ept/tokens.py` | `1d01b3e42dac171bffe5257e0c188548031076e5fefde3d6b2ec79606bcd58e7` |
| `c_ept/towertier.py` | `fc54dcb180ad2e1e48ace1d465bdebd4558ddf9d0463d7e499563d34c0572903` |
| `c_ept/traits.py` | `3a04af77b88c79b9ee3575cd4a0e5b728536e415d42f0c4fcc8c8d9f6dfe6373` |
| `c_ept/units.py` | `73b92010ea3f8092e5f4a1a8e105555830400be2d9b6cd4772bb3458af4281f4` |
| `c_ept/valuation/__init__.py` | `80efa0fa558bb70aa0df8dc05d76c18e4b4e32f98bd43c9acfde38bab081c837` |
| `c_ept/valuation/compose.py` | `af624793716a0b5b318c0f4fa601d47d9cb93c5f9ebc60c203b05714b51f1ca8` |
| `c_ept/valuation/flowprice.py` | `80ee9c4909d0cb7362b9ece02f975a50b414ccd8922fef376a056b258b5104f9` |
| `c_ept/valuation/rulepay.py` | `3c426b547234c53be537bb889c34886a1559fc2617e3467193f59057319fb8dc` |
| `c_ept/valuation/towertier.py` | `c2972697b55d0127642bf4f58321539b57290edd47e529eaf21a177763e02ea1` |
| `c_ept/valuation/value.py` | `545cc1ff8e1f83631e58bb406e438296c73ad7d1cf01df9e0ecf830796ef5b05` |
| `c_ept/value.py` | `54e3b3c39dcdc6b7ee095d82d85d69ee563f9f33f707df16743768386f869583` |
| `common_training_r1/README.md` | `f51ff3efbec4fe42a9aa5b526dd015ebc28ec7c19b36f6e879497b222cc6f9dd` |
| `docs/PUBLICATION_20261024.md` | `4393df94a95f85b756ebc7409d4dd53bac57e26fa95ae7e93ee13bb61dcd8beb` |
| `docs/module-layout-20261024.json` | `f02196bf8ca3f0ec6c3c5199dcaa1728f3cd62df7987e7289f2aca17b0e00255` |
| `docs/module-layout-validation-20261024.json` | `7a72e1b1192fecf58ed675880b583563f2985987d58eb75803286a9e62bdda84` |
| `docs/superpowers/plans/2026-10-24-module-layout.md` | `4f88502cefc4bf6d860bc2d34117336cd9dc10cac9c281d678f24890a23210c8` |
| `docs/superpowers/reports/2026-10-24-module-layout-result-ko.md` | `ef3147615e2c1dacf8d5830762b5877cafbdec116216de6b9dd5e2223c058a1b` |
| `master_ver/MODULES.md` | `84dfe5e9836c7ce0b957f4bd9761d127a60553f864a26db4d5db42a937607aa2` |
| `master_ver/module-layout.json` | `b50c1d84bb80d706fe36bc983a2fba12477fe42936c1f686d38f13c6b4f5b4ee` |
| `master_ver/organization_tests/baseline_ast_sha256.json` | `bd02f97ef818eef9756544389e6358c661dcbdc5817cb8254fde275eabdf65f4` |
| `master_ver/organization_tests/test_reporting_layout.py` | `14c774ff758105fdf6b44fe78aa6f0bfca486cb982fe30b1282061f78d648430` |
| `master_ver/scripts/report_b_three_horizons.py` | `2df6b1e03da257619dca29159548d7c94ce85950bfa3c6a9908566e2d3bff2d9` |
| `master_ver/scripts/report_causal_position.py` | `c74605f515576af57a04081458cae7080714034216fe9bf686f72a5127591465` |
| `master_ver/scripts/report_event_boundary_cif.py` | `0869b68aff5e0b1bc44be16e175e5e049efed545aa2a284ad8dd1ad69d9261f0` |
| `master_ver/scripts/report_fixed120_events.py` | `24bfebec0d5ada0f86cf56da9715a85f0f4da471a49713e84f449d1fcc1c147e` |
| `master_ver/scripts/report_position_ablation.py` | `5eccd8b3342e679dd743b87be34e7aa7eeba64ce826b1e62f32df316cfd210b3` |
| `master_ver/scripts/report_postkill_full.py` | `63ab816ba6a1a7bbbd338477d585052c03bf092141d903e84aaeab85387899fb` |
| `master_ver/scripts/report_validation_suite.py` | `9161f5d2cc9c32d6345b9474db1d34ae8b2bcaff68c47e7da5c6266486bb2cfe` |
| `master_ver/scripts/report_window_validation.py` | `15547258fffde9e5ddb1aa1d2dcc776683ff0abd3c4658b64ba01d024041c052` |
| `master_ver/scripts/reporting/__init__.py` | `2f0f6aeed382144110d48d0b9535a869f1060ad54f896ad099822d821c0d6a17` |
| `master_ver/scripts/reporting/report_b_three_horizons.py` | `cda1bfe4932d5cbaed670699518a95dc3e7cd9a674f58eaa9305bad10de1e68d` |
| `master_ver/scripts/reporting/report_causal_position.py` | `38d5b2036b228487ba18cbeb8ddfb699689806befe628a507df3f988e7ebfe38` |
| `master_ver/scripts/reporting/report_event_boundary_cif.py` | `7cbfc49bf6916be390d2c9209cbcdc71979e8729259ed75bd7b8a81cb7056126` |
| `master_ver/scripts/reporting/report_fixed120_events.py` | `d7f63583f2a709f0b792121b4f9e91fafc9a0862313cb3c21b8fe90fc8a79cee` |
| `master_ver/scripts/reporting/report_position_ablation.py` | `73c0c087af0772ffc65e879db660746fe3a1a518672dbc296f15d1f454baac9f` |
| `master_ver/scripts/reporting/report_postkill_full.py` | `c6a2725e1e3db3677fba2d76275a0343d8337596f3d9dd3c1291fd2b76deb98f` |
| `master_ver/scripts/reporting/report_validation_suite.py` | `89ee69567af4e7880fc01796d8ac0b6fe32f238f35fccac4c2baf43eba5eaf9b` |
| `master_ver/scripts/reporting/report_window_validation.py` | `15140b565cea441d9490b58d64e5b1797c71bcd0fa3b0baa75b0ddfab3756b89` |
| `tests/test_sync_overlay.py` | `aecc02fd8e9391473a71c773d96623e5575b3cdec8a56b155be2169e15011565` |
| `tools/check_project_boundaries.py` | `5a024de9f85d693a9d75fcb135e8c15efefa88048c28d8cce17891b78375f64a` |
| `tools/sync_from_sources.py` | `497b9684a9d83734f677da47a24f5c02feb636f60d82b7af695e3d5af390053e` |
| `work/c_ept-grouping-result.md` | `b7f50ac0c24f92077668cb2bb4e327d1afca792598474bd1f65752b99acf7c09` |
| `work/cog-grouping-result.md` | `90e58cec507d48551775464b6910d49dc36691cc72eb3d8d68b661fee8145ded` |
| `work/master-grouping-result.md` | `f135e4a0fee4194afd64e3f93a0f25600f40cd6fcdbb1cd4b8727d40a86c9b63` |
