# Independent Task 8 GPU profile pre-launch review

**Final decision: ACCEPT** (2026-10-25 KST). F1, F2, F3 and residual F3-R are closed. No remaining actionable finding in the requested bounded review. Acceptance applies to the frozen bytes listed below, the approved S/M CUDA cost scope and unchanged preflight gates. It is not a claim of actual GPU execution or scientific result validation.

## Final residual verification

The shared `verify_historical()` checks the preserved metadata reference using bounded 4 MiB reads per historical JSON. Both normal reports and no-RUN failure reports use it. The failure branch verifies hashes after unchanged-event suppression and before creating/publishing its report folder or recording report completion.

Only the two authorized final cases were run:

- `test_helper_failure_checks_historical_hashes`: changed synthetic historical RUN is refused before a report is published — passed.
- `test_helper_failed_start_without_run_produces_failure_report`: unchanged six historical hashes permit Korean failure report, null run identity, unavailable CSV, valid SVG/gallery, and unchanged subsequent failure stays quiet — passed.

Result: **2 passed, 68 deselected in 0.17s**. Log: `work/review-f3r-final.log`. Exact invocation: `C:/Users/todtj/anaconda3/python.exe -B -m pytest work/review_contract_probes.py -q -p no:cacheprovider --tb=short --basetemp=work/review-f3r-final-temp -k 'helper_failure_checks_historical_hashes or helper_failed_start_without_run_produces_failure_report'`.

No 321/357/full-suite repeat. All real controller/helper subprocesses were mocked. CUDA disabled, native threads 1, SEQUENTIAL MKL, UTF-8, bytecode/plugins/cache disabled. This reviewer performed no actual launch, GPU calculation, source edit, commit, push, input-store access, protected cohort/holdout read, weight/checkpoint inspection or Strata use. Writes were limited to this report and ignored work/review* artifacts in the explicit modularize development checkout. The new clean execution checkout was not touched.

## Previously completed evidence retained

- Operator v1/v2 relevant suite: **321 passed in 14.16s** before source freeze; accepted profile source/test/example and scope documents remain byte-for-byte unchanged. v1 canonical digest/exact CPU+CUDA contract and cuda-only refusal remain intact. Explicit v2 gpu_cost selects only S/M CUDA, 2 epochs, .25 share and DEV 200; schema/profile and selected args bind to identity. Resume/native/E6/whole-Git-clean and scientific runner/trainer remain unchanged.
- F1/F2/F3 re-review: all **three original reproductions and three required positive cases passed** with full realistic RUN/STAGES fixtures. F1 discovers before wait and after exit; normal Start/exclusive launch.once/runtime environment passed. F2 rejects cross-identity stages and accepts a valid full-schema GPU snapshot with seven-stage SVG. F3 reports failed no-RUN Start and suppresses unchanged repeats; normal Korean reports bind commit and observed run identity. These six passing cases were not repeated for F3-R.
- F2 frozen SHA256 is `c75ba8a2a6ef0a871ec62bba15e76717abcae903b3b7eb36d44ce637ebcb2e33`, matching the implementer's supplied value. Worker-reported 94 tests are not substituted for independent probes.
- All six real historical S CPU root metadata byte hashes were rechecked and still match. Old S CPU remains a different-revision reference; M CPU remains unmeasured. Missing measurements are unavailable, not zero. New profile has seven stages and only S_cuda/M_cuda combinations.
- Realistic reviewer fixtures use actual unchanged runner parser/_normal_args/digest definitions without main/execute or real data. They contain schema/stage/API, full args/code/runtime/inputs identity, matching config and run digests, seven selected stages and real ledger structure. The cross-identity reproduction was retained, with a self-consistent changed RUN identity but foreign stage input binding.

Detailed rejected-state audit history: `work/review-initial-report.md`, `work/review-before-f3r-accept.md`. Focused re-review log: `work/review-fixes-final.log`; initial relevant suite log: `work/review-relevant-tests.log`. Counts from overlapping runs are not additive.

## Launch handoff boundaries

Lead owns final commit, advance/preparation of the separate clean measurement checkout and external settings/config identity, then unchanged actual preflight and the authorized Start. These values were intentionally not required before acceptance. Preserve historical S CPU, perform only the approved GPU cost run, report each completion/failure, and stop at G-S4a. No M CPU or S4b/S6/S7 execution is included. Junk cleanup stays in the separate development tree. This independent reviewer has not committed or launched anything.

## Accepted raw byte SHA256 registry

Captured UTC: 2026-10-24T15:04:38.772726+00:00. Report self excluded. Machine registry: `work/review-raw-hashes-accepted.json`. Only report_once.py changed since the immediately preceding frozen review; other 21 reviewed files match. Final helper SHA256: `cb9a0243b4c075ad4a22c5f551c9f2b3fdb1359a414f9de4e325b5f5795081cd`.

| Reviewed file | Bytes | Raw SHA256 |
|---|---:|---|
| `c_ept/workflows/s4a_operator.py` | 19657 | `6664f92e60037d45f025c87ec1da7d9843bbd2bc4036be55b1897bf27e785b83` |
| `c_ept/tests/test_gpu_cost_profile.py` | 9748 | `9283531af2b33a24ff2c0bc29081b49ffeb00f88f879efa9d7be149c068b931e` |
| `configs/s4a.gpu-cost.example.json` | 553 | `9f0354f546b84384000593758cd75ff432c99fd234974a6bdaa4a875a0034ae4` |
| `docs/WORKING_GUIDELINES_KO.md` | 15629 | `12521ef998ead2ec9bbca39ba25bdcbe781deba72747e6c8b7e9cf32db81fa70` |
| `docs/S4A_NEXT_STEPS_KO.md` | 3150 | `9dc030b16c83c9e33e199d04154b41bcad234e6f61d52066c50452005f0b261f` |
| `docs/superpowers/plans/2026-10-24-task8-gpu-cost.md` | 3554 | `b1d6bee432f04641ca300cd1733bdd8785b6626cb2851c2ccb96512da831abc8` |
| `D:\LOL_Data\runs\20261024_c_ept_task8_gpu_control_1\tools\controller.py` | 6749 | `7da8326841123b25622aa44dca8243642154eb0aacb4e953ab85b534bc81e14f` |
| `D:\LOL_Data\runs\20261024_c_ept_task8_gpu_control_1\tools\watch_once.py` | 1986 | `bab7336b96dfbbb735155302d51b5fa0542322ef781e4a6bc086fde91f05e49e` |
| `D:\LOL_Data\runs\20261024_c_ept_task8_gpu_control_1\tools\reporting.py` | 21449 | `c75ba8a2a6ef0a871ec62bba15e76717abcae903b3b7eb36d44ce637ebcb2e33` |
| `D:\LOL_Data\runs\20261024_c_ept_task8_gpu_control_1\tools\report_once.py` | 10678 | `cb9a0243b4c075ad4a22c5f551c9f2b3fdb1359a414f9de4e325b5f5795081cd` |
| `c_ept/workflows/run_s4a.py` | 40869 | `fe50a5daef6dae84a892dafef78d807d23a5ee25ba894e836f1a327d60f20db2` |
| `c_ept/workflows/s4a_runtime.py` | 4266 | `31822adf644e8cbefc202f3d032afc630e87406774fdd8ae3ba7b8686e9b6b3f` |
| `c_ept/ept/train.py` | 83673 | `9d53ceba09788cc562b69740f6b1bfc9c06b771d3c5992b013c8b270f0ef61aa` |
| `c_ept/tests/test_s4a_operator.py` | 38422 | `49188e1041ea404af42fb529b3a4c8318a7432e7275b58562d65ef0b5fbcf426` |
| `c_ept/tests/conftest.py` | 14999 | `b29f165cc4210800472733ae4ccb50fc24c1c6dc0785c1b1d5b31b97d161aaef` |
| `c_ept/__init__.py` | 706 | `38ee5093acbcca89ce45d0c8b26b95ada351c219a6907c5aa49e786fee770c0e` |
| `configs/s4a.example.json` | 527 | `3d415a5677ba5cddff2be5116c03f74f34a0fb6989253a98c49788c8746bbab9` |
| `run_s4a.ps1` | 1695 | `1ca4adbc2e12e7aeb93bb4de46e2bb1fdc13fcbb9be2db83707664979d3c6399` |
| `environment/s4a-runtime.json` | 1091 | `8888786acc50134959c437b2d337f113a190e9a1dab47e2d11379b9911294347` |
| `.gitignore` | 7994 | `0566cf96506294c8df5c14b98c08ff7c9f061f9fbf02a5d9f91b0eaf01d33805` |
| `work/gpu-profile-result.md` | 3706 | `f21cb6199dc14978a55a2a394dd5938816ae6b0ebb6a236954ed4e56754f250b` |
| `D:\LOL_Data\runs\20261024_c_ept_task8_gpu_control_1\historical-s-cpu-reference.json` | 846 | `282a7513959ae46f0435004bd52c5e8e2ce06429f8be707b7cfb6a4e43895208` |

### Preserved historical S CPU root metadata

- `RUN.json`: `9923d2488fe346dcd1e1a965c00df42eccf549b2a3bf9f9585a4862ea0fc987a` (matches preserved reference).
- `STAGES.json`: `9ac59f8c09b1202db419dedb4deb95c68063d2aa2cf10ab2434f653ec8171350` (matches preserved reference).
- `PROCESS.json`: `faae772900e9ac3aef425d2904594ceae977b6eef397127c92d9f3b500b9c826` (matches preserved reference).
- `scales.json`: `96d7ae5bbe734a1e2ac10dd410230e7c9ef627cb01c5d2d0b7ee07fd840d11be` (matches preserved reference).
- `stats.json`: `408372d2afc18a6d6ae2ad937a2a31d6547bdffb1f73cfa6c70ebb512285ffe9` (matches preserved reference).
- `train_S_cpu.json`: `d107593fa237a207d0d70c6e93929653261eaf1123609fc6e5bc61c2628d05ff` (matches preserved reference).
