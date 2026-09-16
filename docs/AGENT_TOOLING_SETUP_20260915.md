# Agent tooling installation — 2026-09-15

Installed in `C:/Users/todtj/문서/LOL_Teamfight`, not globally. No experiment was launched or changed by this installation.

## Sources and installed components

- [codex-astra-luna-orchestrator](https://github.com/donvito/codex-astra-luna-orchestrator), commit `642b16074ba8973d4f920ad8cbfb542bde3b4682`: Pro topology, project `.codex/config.toml`, five `.codex/agents/*.toml` roles, and `.agents/skills/astra-orchestrator/SKILL.md` installed using the Codex skill installer at the pinned commit.
- [Minimize-Cursor-Cost](https://github.com/inboxpraveen/Minimize-Cursor-Cost), commit `bf3319fb41ca4de9619df066e7c3e44b7843aa40`: adapted project `AGENTS.md`, `CLAUDE.md`, Python/data-science/tests Claude rules and `PROMPT_TEMPLATES.md`. Cursor-specific rules were not installed because this workflow uses Codex and Claude.
- Reviewed source snapshots remain under `tooling/vendor/`. Installed files are independent copies; upstream updates are not automatic. Review and merge later updates against project overrides.
- `tooling/agent_tooling_install_manifest.json` records source revisions and installed-file SHA-256 hashes. `tooling/install_agent_tooling.py` documents the initial installation and refuses overwriting existing files.

## Workflow

Root: Astra, medium reasoning. Explorer/worker/tester/researcher: Luna, max reasoning. Reviewer: Astra, low reasoning. Actual availability, explicit user settings and host limits take precedence. Configuration permits up to four subagents; this current session has four total slots, including root.

Use bounded task contracts and separate file ownership. Delegate only useful independent work; keep simple changes local. Codex retains design/audit/integration responsibility. Existing user preference that Claude implement and run research experiments remains in force; avoid duplicate dispatches.

For future sessions opened in this project, the project instructions and skill are available for discovery. Start a new session if the current client has cached configuration. The current root model is not switched by writing config. Project configuration loading remains subject to Codex project trust and higher-priority session settings.

## Research-specific adaptations

Cost rules reduce duplicated context, broad searches and oversized chat outputs. They do not relax frozen patch splits, test isolation, full-corpus computations, immutable results, source citations or reproducibility. Small printed samples are not sampling authorization for the experiment. Seeds and necessary documentation must be recorded even where older code omitted them. Complete logs/results stay on disk.

Global provider, credentials and approval settings were not edited. Root approval/sandbox values from the upstream example were deliberately omitted to inherit host settings. Role sandbox declarations retain upstream restrictions, subject to host permissions.

## Validation

- Six TOML files parsed during installation.
- Official Codex configuration and custom-agent documentation were checked for the installed standalone role format and `agents` defaults: [configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference), [subagents](https://learn.chatgpt.com/docs/agent-configuration/subagents).
- Codex CLI `0.154.0-alpha.6.2`, user-context `codex --strict-config doctor --summary --no-color --ascii`: configuration loaded, authentication configured, desktop app-server initialized, provider reachable and WebSocket connected. Overall command exit code was 1 because `TERM=dumb` fails the terminal check; this is not a configuration failure. Sandbox-account diagnostics initially lacked the user's credentials; the user-context check resolved that discrepancy without changing authentication.
- `codex --strict-config features list` is unsupported by this CLI; it is not used as evidence of validation.
- An explicitly requested Luna/max subagent was dispatched for an independent review of the lean rules. See the review completion record below.

Token savings have not been measured. No percentage reduction is claimed. Compare recorded usage and accepted-change quality over subsequent real tasks before estimating savings.

### Independent review completion

Luna/max reviewer completed read-only review. Its recommendations on protocol-level acceptance, provenance, immutable artifacts, source checks for documentation and necessary research logs/tests were integrated into AGENTS.md and CLAUDE.md. No research code or results were modified. This demonstrates in-session delegation; future automatic model routing depends on the loaded client configuration.
