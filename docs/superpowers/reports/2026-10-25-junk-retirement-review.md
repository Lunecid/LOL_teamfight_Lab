# Independent optional agent-tooling retirement review

Final decision: **ACCEPT** (2026-10-25 KST), frozen optional-tooling retirement only. All requested bounded checks pass; no actionable finding. Worker freeze was explicitly confirmed before final hashes. This document is a new retirement review; prior GPU profile review and measured checkout remain untouched.

## Scope and interpretation

This deliberately retires the repository-local optional developer installer plus its entire vendor input bundle. It does not claim the upstream projects are globally dead, unused by external users, or unnecessary to existing global installations. Recovery is pinned to `4bdc77953f3bfbaa852f954f1846f907384383a3`; current DEV HEAD is `f867a51e582bd5f4011e6cc52b6f3476deb1b75a`. The ignored preparation plan's WAIT-GO language is historical; the worker result records explicit revised GO, consistent with the current author's authorized retirement scope.

## Independent evidence

- Exact diff: all 47 planned paths absent, all under DEV `master_ver/tooling`, all known stage-0 tracked regular-file entries, all 47 original Git blob identities equal the recovery parent. No symlink/junction traversal in retained parent directories. Total removed Git blob size: 1,604,815 bytes. No recursive deletion was performed by this reviewer.
- Remaining tracked delta is only `master_ver/AGENTS.md`; four untracked archive docs are exactly the planned README, historical installation manifest and two complete licenses. No staged change. No retained Python/scientific implementation, compatibility module, studies or frozen result was edited.
- All three preservation copies match recorded pre-deletion raw SHA256/byte counts and complete original Git blob contents after CRLF normalization. Raw-byte and Git-normalized identities are distinguished. Both upstream source revisions and the Git recovery parent are described in the README; the historical manifest is not presented as a current installation/validation result.
- Replacing only the approved new AGENTS sentence with the exact old sentence reconstructs the original raw SHA256 `7539516810921d4a5e91140e396662b83be602e5d1b2400c204a50962ef18222`. Thus all other original text, including research requirements, remains byte-identical. CLAUDE/PROMPT_TEMPLATES and four packaging files are unchanged.
- Independent fixed-string rg scan: 1204 safe tracked code/config files, 0 removed-bundle references. Selected live entry docs also have 0 such references. Protected cohort/holdout bodies, data/payloads, ignored/user/global configuration and historical frozen documents were excluded. Static scanning cannot disprove dynamic/external manual use; that limitation is consistent with the explicit capability-retirement decision.
- Registered entrypoints and package discovery boundaries were inspected statically. Existing command targets resolve; tiny AST parsing of command sources passed without importing/executing them. EPT packaging includes only c_ept; Master/CoG use their src packages and shared runtime. The retired tooling is outside these roots and dependencies. Local links in changed documents resolve.

## Activity and limits

No pytest/321/357 rerun, GPU/CPU training, model call, actual Start/Resume, source fix, Git staging, commit or push. Only a stdlib one-process static audit and Git/rg metadata checks ran. Protected bodies, raw data, weights/checkpoint payloads, measured checkout and original Master workspace were not accessed. Strata remained OFF. No user/global tools were modified; no global installation audit is claimed.

Reviewer observation began 2026-10-24T15:09:26.7364224Z. Bounded audit was pinned to one available CPU (affinity mask 1; native thread env 1): 2026-10-24T15:11:22.919898+00:00 through 2026-10-24T15:11:24.583991+00:00 UTC, 1.664 seconds. An earlier ctypes-affinity setup attempt failed before audit checks; the reviewer-only handle declaration was corrected. Lightweight read/coordination time outside the audit will be included in the final lead activity interval; no measured-run metadata was changed by the reviewer.

Evidence: ignored `work/reviewjunk_checks.py`, `work/reviewjunk-checks.json`, `work/reviewjunk-checks.log`. Implementation evidence: `work/agent-tooling-removal-manifest.json`, `work/agent-tooling-cleanup-result.json`, `work/agent-tooling-cleanup-inventory.md`, and `work/junk-inventory.*` (inventory read selectively).

## Exact deletion whitelist

| Former tracked path | Preserved Git blob SHA1 |
|---|---|
| `master_ver/tooling/agent_tooling_install_manifest.json` | `1ce5b49a71e0962dd8bfa0a8b193bdc6feca7287` |
| `master_ver/tooling/install_agent_tooling.py` | `c93284104d42a5f56f18d0ec6c6bdd0d2b5fc462` |
| `master_ver/tooling/vendor/Minimize-Cursor-Cost/CONTRIBUTING.md` | `006db7422f07e862ede16c5aa2bf674d9b1073c2` |
| `master_ver/tooling/vendor/Minimize-Cursor-Cost/LICENSE` | `261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64` |
| `master_ver/tooling/vendor/Minimize-Cursor-Cost/README.md` | `46813a749e60c1adeea3bf32d6e76a604af48cc4` |
| `master_ver/tooling/vendor/Minimize-Cursor-Cost/assets/Main-Diagram.png` | `1476e7b13eca48aa633a0c39d54c059930417904` |
| `master_ver/tooling/vendor/Minimize-Cursor-Cost/benchmarks/README.md` | `ff6094d5e550a953ffb3c8ceb611320062ed6813` |
| `master_ver/tooling/vendor/Minimize-Cursor-Cost/benchmarks/scenarios.json` | `fefaeefe8a941fa03db87e9559452ebf60618f37` |
| `master_ver/tooling/vendor/Minimize-Cursor-Cost/benchmarks/summarize.py` | `a6d1adb6076c51bf64d8f913ad056e012e7adc7c` |
| `master_ver/tooling/vendor/Minimize-Cursor-Cost/install.ps1` | `eb297f93024470b0c6c534183498499d7697e141` |
| `master_ver/tooling/vendor/Minimize-Cursor-Cost/install.sh` | `c7ad8676fb5297c8a707194c58c001cb824d864f` |
| `master_ver/tooling/vendor/Minimize-Cursor-Cost/lean-cursor/.cursorindexingignore.example` | `d719b46376b267176c0d0361465bca95626410fd` |
| `master_ver/tooling/vendor/Minimize-Cursor-Cost/lean-cursor/.cursorrules` | `0ac901b196999af55abfc13a51773f610076dc17` |
| `master_ver/tooling/vendor/Minimize-Cursor-Cost/lean-cursor/AGENTS.md` | `d1e16dc92bc68718a0a6bcfaa247e93dc043bb16` |
| `master_ver/tooling/vendor/Minimize-Cursor-Cost/lean-cursor/CLAUDE.md` | `bba510691bc5b108d790701ebb1e1a987d52d90a` |
| `master_ver/tooling/vendor/Minimize-Cursor-Cost/lean-cursor/PROMPT_TEMPLATES.md` | `2298fbf22d1f9177906a430a549d2314deb140c6` |
| `master_ver/tooling/vendor/Minimize-Cursor-Cost/lean-cursor/SETUP_GUIDE.md` | `b84db0f76162ec27ff23ac8568521fac51b8903b` |
| `master_ver/tooling/vendor/Minimize-Cursor-Cost/tools/gen-claude-rules.py` | `c3e67779fc0698fe9ce10c9885b0259156997ea0` |
| `master_ver/tooling/vendor/codex-astra-luna-orchestrator/.github/FUNDING.yml` | `304282db2242544a6998d3f2aeeb089b98e62dd7` |
| `master_ver/tooling/vendor/codex-astra-luna-orchestrator/.gitignore` | `5dfc3d3eaf8619ecedff723012554c416abeac00` |
| `master_ver/tooling/vendor/codex-astra-luna-orchestrator/AGENTS.md` | `c9328d14d6b887e72783ae9a238873b3f68f2069` |
| `master_ver/tooling/vendor/codex-astra-luna-orchestrator/LICENSE` | `d645695673349e3947e8e5ae42332d0ac3164cd7` |
| `master_ver/tooling/vendor/codex-astra-luna-orchestrator/README.md` | `b0d4d66d0239b67ed188523a9ffd06060140e044` |
| `master_ver/tooling/vendor/codex-astra-luna-orchestrator/guides/complex-repo-work.md` | `e854441c83bf068716b86ec991b30c8f9cd61f06` |
| `master_ver/tooling/vendor/codex-astra-luna-orchestrator/guides/fast-iteration.md` | `abd5a075e9e4d2ce3a1ad59be84257881b9ae3d0` |
| `master_ver/tooling/vendor/codex-astra-luna-orchestrator/guides/full-orchestration.md` | `a5b7603852426422da57c76ac6bc3b16958ee440` |
| `master_ver/tooling/vendor/codex-astra-luna-orchestrator/guides/plus-plan.md` | `63a7415d2ef2d8ca97c84103655e9cefc115364f` |
| `master_ver/tooling/vendor/codex-astra-luna-orchestrator/guides/routine-coding.md` | `b88b64b23195d7db9a9d57a6cc7c861568a02e8d` |
| `master_ver/tooling/vendor/codex-astra-luna-orchestrator/guides/token-usage.md` | `0359ccee47d9a54acaa493e7a4032b32952caf30` |
| `master_ver/tooling/vendor/codex-astra-luna-orchestrator/profiles/plus/agents/skills/astra-orchestrator/SKILL.md` | `9862da73dfa6fbc2babcfeac6c8c5dfad7765552` |
| `master_ver/tooling/vendor/codex-astra-luna-orchestrator/profiles/plus/codex/agents/explorer.toml` | `afc5f28501aed98f0a9df964ff2dee2d77db3275` |
| `master_ver/tooling/vendor/codex-astra-luna-orchestrator/profiles/plus/codex/agents/researcher.toml` | `f289d1113ef48186640fc9cc140824eb572b84a1` |
| `master_ver/tooling/vendor/codex-astra-luna-orchestrator/profiles/plus/codex/agents/reviewer.toml` | `f3684fafc7561dc4b776d324a81e23950ca831b7` |
| `master_ver/tooling/vendor/codex-astra-luna-orchestrator/profiles/plus/codex/agents/tester.toml` | `f50cbcc51523a3a5b672bbb72bc600a60828eb64` |
| `master_ver/tooling/vendor/codex-astra-luna-orchestrator/profiles/plus/codex/agents/worker.toml` | `f30a4bd6dda5b7c5b8d2dd68f84297fa5725900a` |
| `master_ver/tooling/vendor/codex-astra-luna-orchestrator/profiles/plus/codex/config.toml` | `b047e6bf00c0e1c14e8fa1ff29cb45b0e84e6df8` |
| `master_ver/tooling/vendor/codex-astra-luna-orchestrator/profiles/pro/agents/skills/astra-orchestrator/SKILL.md` | `712ae32de0e99233b7af992f4d01c6c7945de81a` |
| `master_ver/tooling/vendor/codex-astra-luna-orchestrator/profiles/pro/codex/agents/explorer.toml` | `57b932e122a5340ef67caba0c1f6e66e75cd72a3` |
| `master_ver/tooling/vendor/codex-astra-luna-orchestrator/profiles/pro/codex/agents/researcher.toml` | `74758216b729ecdd723fd8e2e60cd53db75d00f1` |
| `master_ver/tooling/vendor/codex-astra-luna-orchestrator/profiles/pro/codex/agents/reviewer.toml` | `f3684fafc7561dc4b776d324a81e23950ca831b7` |
| `master_ver/tooling/vendor/codex-astra-luna-orchestrator/profiles/pro/codex/agents/tester.toml` | `60c2e3f3311d88a8e9d9b7a914ea39f72e7bb86b` |
| `master_ver/tooling/vendor/codex-astra-luna-orchestrator/profiles/pro/codex/agents/worker.toml` | `8c53f47960d0fb16e58a50e573675a87be574633` |
| `master_ver/tooling/vendor/codex-astra-luna-orchestrator/profiles/pro/codex/config.toml` | `efd11eeb7a029f23558dc9bbafcd11830dbd243b` |
| `master_ver/tooling/vendor/codex-astra-luna-orchestrator/scripts/token_usage.py` | `9537ed0d80d570f5e4556a4c41396a7203e0e1ec` |
| `master_ver/tooling/vendor/codex-astra-luna-orchestrator/setup.ps1` | `cfeeb0f2872c283d48d964ba48cb0c3c01244771` |
| `master_ver/tooling/vendor/codex-astra-luna-orchestrator/setup.sh` | `ca06c206be7086130d98485daa5ef6471659b9af` |
| `master_ver/tooling/vendor/codex-astra-luna-orchestrator/tests/test_token_usage.py` | `01f5d612067b59014315301673e63b0e9c63abff` |

## Final frozen acceptance and byte accounting

No implementation drift after the preliminary checks. Final tracked diff remains exactly 47 deletions plus one AGENTS routing-line edit, all unstaged. The four preserved archive docs and routing edit are the implementation acceptance scope. The lead's separate frozen Korean result document was read and hashed below; this review does not treat it as a scientific code change.

Removed raw worktree file bytes 1,610,649 minus preservation additions 27,090 minus AGENTS growth 10 equals **1,583,549 net raw worktree file bytes reclaimed**. Deleted Git blob bytes are **1,604,815**. The net raw figure is not Git object-store savings, filesystem allocation recovery, runtime speedup or scientific improvement. Recovery remains possible from the pinned Git parent.

The Korean result document accurately distinguishes byte accounting, preserved legacy research, optional capability retirement and the separate GPU checkout. Its statements about the live measurement checkout are lead-reported; the independent reviewer deliberately did not access that checkout. No current run identity or training result is certified here.

### Final raw SHA256 (review self excluded)

| Accepted implementation file | Bytes | Raw SHA256 |
|---|---:|---|

| `master_ver/AGENTS.md` | 5951 | `6cb09fc3ebc8d54c691a1297ba8f70f41fe4d16c1f700b66c3d44b2c68a929bb` |
| `docs/retired-agent-tooling/README.md` | 1841 | `c96d3bbb3a0e395a2361275a83ff2dc89a6993a3a874146536f90612e9790f79` |
| `docs/retired-agent-tooling/agent_tooling_install_manifest.json` | 2131 | `8bade3ae7b4f80c4e108d093cd63abf6401bde476b05caad7bb4de365eebd6ea` |
| `docs/retired-agent-tooling/LICENSE.Minimize-Cursor-Cost` | 11558 | `1eb85fc97224598dad1852b5d6483bbcf0aa8608790dcc657a5a2a761ae9c8c6` |
| `docs/retired-agent-tooling/LICENSE.codex-astra-luna-orchestrator` | 11560 | `3ddf9be5c28fe27dad143a5dc76eea25222ad1dd68934a047064e56ed2fa40c5` |

Separate lead result document: `docs/superpowers/reports/2026-10-25-junk-retirement-result-ko.md`, 2406 bytes, SHA256 `5add7031501bc9fc079c2854b67843f5251ee509d3c7b026ff395315aca281c1`.

Full final registry, exact original deletion Git objects and source-evidence hashes: `work/reviewjunk-final-hashes.json`.

Final one-CPU metadata/hash check: 2026-10-24T15:16:24.765148+00:00 through 2026-10-24T15:16:25.068025+00:00, 0.303 seconds. Entire reviewer I/O/coordination wall interval for the lead's activity ledger: **2026-10-24T15:09:26.7364224Z through 2026-10-24T15:16:25.068025+00:00**; this includes waiting, not continuous CPU load. Earlier active audit interval was 2026-10-24T15:11:22.919898+00:00 through 2026-10-24T15:11:24.583991+00:00 (1.664 seconds). No developer-activity.json write was made by this reviewer; lead can append these precise intervals.

Standards: pass for the documented preservation, scope and review workflow. Spec: pass for the explicitly approved optional developer capability retirement. **ACCEPT**, with no broad scientific validity, global dead-code or performance claim. No stage/commit/push, user IDE sync or run operation was performed by this reviewer. Prior GPU profile review and measured checkout remain untouched.
