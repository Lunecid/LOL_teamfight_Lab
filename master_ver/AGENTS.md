# AGENTS.md — Lean Agent Rules

Optimize cost per accepted change, not response length. Preserve correctness
while avoiding duplicated context, speculative work, and unnecessary prose.

## Response and Scope

- Start with the answer or edit. Keep responses concise; retain required progress updates and evidence.
- Show patches with 3–5 context lines. Show a full file only when new, under
  40 lines, or explicitly requested.
- Ask one blocking question only when a fact cannot be inferred or checked.
- Do not expand beyond the user's scope. Execute explicitly large work in
  bounded phases.
- Present at most two alternatives and recommend one.

## Silent Reuse-First Gate

Before tools or edits, check whether the supplied context is sufficient and
whether an existing helper, type, pattern, dependency, fixture, or file can be
reused. Prefer the smallest safe edit and cheapest acceptance check. Do not
narrate this checklist or ask the user to make decisions that code can answer.

## Evidence and Tools

- Start from the named file, symbol, or exact error.
- Search exact strings before semantic exploration; read matching ranges, not
  whole large files.
- Reuse session context; batch independent tools; do not re-read by habit.
- Limit search and command output to relevant failures and stack frames.
- Exclude generated, vendored, minified, lock, build, and cache files.
- Stop widening when repeated searches produce no new evidence.

## Code and Verification

- Match nearby language, naming, formatting, error handling, types, and
  abstractions.
- Do not add dependencies, files, tests, logging, comments, or refactors beyond
  the task or surrounding baseline.
- Text/comment/format-only edits need no check unless requested.
- Normal code gets the cheapest targeted check when available.
- Bug fixes, public APIs, auth, migrations, and config require the smallest
  relevant test, lint, parse, or validation check.
- Stop when the acceptance check passes and no unrelated code changed. Report a
  repeated failure without new evidence instead of looping.

## Cache and Model Use

Keep this file stable and task-specific context late; a stable prefix is what
makes prompt caching cheap. Use a fast model for deterministic bounded work and
stronger reasoning for ambiguous architecture or novel debugging; a cheap model
that causes retries is not cheaper.

---

# Codex project instructions

For complex coding tasks, follow the project-wide [working guidelines](../docs/WORKING_GUIDELINES_KO.md).

The root agent owns architecture, decomposition, integration, and final verification.
Prefer specialized subagents for bounded exploration, implementation, testing, review, and technical research.

Do not delegate trivial work merely for parallelism.
Do not let multiple implementation agents edit the same files without explicit ownership boundaries.
User instructions always take precedence over this orchestration policy.



## Research project requirements (override generic cost-saving rules)

- User instructions and host safety/runtime instructions take precedence. Keep required progress updates.
- Codex owns research design, audit and integration; Claude implements/runs experiments from approved specifications. Luna assists bounded exploration, implementation when authorized, and verification; do not duplicate an active Claude experiment.
- Definition/citation edits require consistency and source checks even when text-only. For experiments, run all gates required by the approved protocol (including split/leakage, OOF exclusions, reload and independent metric checks where applicable); a preflight or running process is not completion. Preserve failed runs and allow necessary entry points, run-local logging and scientific test documentation. Never edit frozen notebooks in place.
- Preserve protocol definitions, citations, reproducibility and necessary validation. Cost reduction means less duplicated context, not less evidence.
- Frozen main split: patch 15.14 train, 15.15 validation, 15.16 test; newer and regional cohorts are external tests. Never tune on test data, leak future information into predictors, or replace full-corpus validation with a sample.
- Do not silently change populations, labels, features, endpoints, or follow-up windows. Check the relevant current protocol before experiments; distinguish planned from executed results.
- Treat existing raw data, trained models and completed experiment outputs as immutable. Write new runs to separate named directories. Do not modify the original repository or external data stores without explicit scope.
- Read generated results/manifests/cache evidence when required for an audit. Exclude them only from unrelated broad searches. Small displayed samples are for inspection, not substitutes for complete computations.
- Record seeds, environment, row counts, split keys, exclusions, hashes and metrics as appropriate even if older scripts omitted them. Add documentation or meaningful tests needed by the research task.
- Keep complete experiment logs and result artifacts on disk; return concise summaries and paths. Do not discard evidence to reduce chat tokens.
- Use existing Python/scikit-learn tooling and focused pytest checks where applicable. Do not launch training to validate a documentation-only edit.
- Respect an overall CPU budget of four workers and avoid nested training/agent oversubscription; GPU use requires explicit scope.
- Never include credentials or account identifiers in external model payloads. Existing Claude authorization covers specifications, relevant code, aggregates and limited timelines, not whole raw datasets.
- Start current research status discovery from docs/TOG_END_TO_END_READINESS_AUDIT_20260915.md or docs/OBJECTIVE_CHANNEL_ABLATION_FINDINGS_20260915.md when relevant; verify newer evidence before relying on them.
