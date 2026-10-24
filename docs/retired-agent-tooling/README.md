# Retired optional agent tooling

The project has retired its repository-local agent-tooling installer and vendored setup bundle as an optional developer capability outside the research runtime. This decision does not claim that the upstream projects are globally unused.

Use the existing user-installed OpenCodex and Sol tools under the [project working guidelines](../WORKING_GUIDELINES_KO.md). This retirement does not install, uninstall, or change user configuration, global agents, or existing astra installations.

The former installer wrote project agent configuration and rules from vendored templates. It is no longer shipped or supported here. Master AGENTS.md, CLAUDE.md, PROMPT_TEMPLATES.md, and all scientific instructions remain in place; the optional skill-routing sentence now points to the common guidelines.

Preserved provenance and attribution:

- [Original installation manifest](agent_tooling_install_manifest.json): exact historical copy, including source revisions, original scope, adaptations and recorded installation hashes. These hashes describe the historical installation; they are not validation of the current project or evidence that global tools were changed.
- [Minimize-Cursor-Cost license](LICENSE.Minimize-Cursor-Cost), from inboxpraveen/Minimize-Cursor-Cost, revision bf3319fb41ca4de9619df066e7c3e44b7843aa40.
- [codex-astra-luna-orchestrator license](LICENSE.codex-astra-luna-orchestrator), from donvito/codex-astra-luna-orchestrator, revision 642b16074ba8973d4f920ad8cbfb542bde3b4682.

The removed installer, vendor tools, artwork, setup documentation and profiles are recoverable from Git parent 4bdc77953f3bfbaa852f954f1846f907384383a3 under master_ver/tooling/. Research scripts, compatibility entrypoints, data, frozen records, and historical studies are outside this retirement.
