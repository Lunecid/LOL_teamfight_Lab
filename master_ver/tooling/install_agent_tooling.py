"""Install reviewed, project-scoped agent configuration; never overwrite existing files."""
from pathlib import Path
import hashlib
import json
import tomllib

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / 'tooling/vendor'
ORCH = VENDOR / 'codex-astra-luna-orchestrator'
LEAN = VENDOR / 'Minimize-Cursor-Cost/lean-cursor'

NOTES = '''
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
'''

files = {}
def add(rel, body):
    path = ROOT / rel
    if not path.resolve().is_relative_to(ROOT):
        raise RuntimeError(f'Outside project: {path}')
    if path.exists():
        raise RuntimeError(f'Refusing to overwrite: {path}')
    files[path] = body

config = '''# Project model routing; host approval, sandbox, provider and credentials remain inherited.
model = "gpt-6-astra"
model_reasoning_effort = "medium"

[agents]
enabled = true
max_concurrent_threads_per_session = 4
default_subagent_model = "gpt-5.6-luna"
default_subagent_reasoning_effort = "max"
'''
tomllib.loads(config)
add('.codex/config.toml', config)
for src in sorted((ORCH / 'profiles/pro/codex/agents').glob('*.toml')):
    content = src.read_text(encoding='utf-8')
    tomllib.loads(content)
    add('.codex/agents/' + src.name, content)

for name in ('AGENTS.md', 'CLAUDE.md'):
    core = (LEAN / name).read_text(encoding='utf-8').split('## PROJECT-SPECIFIC NOTES')[0].rstrip()
    core = core.replace('No preamble, affirmation, recap, or sign-off.', 'Keep responses concise; retain required progress updates and evidence.')
    if name == 'AGENTS.md':
        core += '\n\n' + (ORCH / 'AGENTS.md').read_text(encoding='utf-8')
    add(name, core + '\n\n' + NOTES)

for name in ('python.md', 'data-science.md', 'tests.md'):
    body = (LEAN / '.claude/rules' / name).read_text(encoding='utf-8')
    body = body.replace('Set random seeds only if the surrounding code does.', 'Record random seeds for stochastic research computations, even when older code omitted them.')
    body = body.replace("Don't add markdown cells unless asked.", 'Add concise markdown cells when needed to document research definitions, assumptions or evidence.')
    body = body.replace('Match existing error handling style (bare except vs specific exceptions)', 'Match existing conventions without reproducing unsafe exception handling; preserve failures needed for audit.')
    body += '\n\nProject research requirements in CLAUDE.md take precedence over these generic output shortcuts. Preserve full artifacts and necessary reproducibility checks. Output abbreviations must never become placeholders in executable code.\n'
    add('.claude/rules/' + name, body)
add('PROMPT_TEMPLATES.md', (LEAN / 'PROMPT_TEMPLATES.md').read_text(encoding='utf-8'))

for path, body in files.items():
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding='utf-8', newline='\n')

skill = ROOT / '.agents/skills/astra-orchestrator/SKILL.md'
assert skill.is_file()
manifest = {
    'date': '2026-09-15', 'scope': str(ROOT),
    'sources': {
        'https://github.com/donvito/codex-astra-luna-orchestrator': '642b16074ba8973d4f920ad8cbfb542bde3b4682',
        'https://github.com/inboxpraveen/Minimize-Cursor-Cost': 'bf3319fb41ca4de9619df066e7c3e44b7843aa40',
    },
    'adaptations': ['Project-only Pro topology', 'Host security and credentials inherited', 'Research rigor overrides generic output minimization', 'Python/data-science/tests Claude scopes only'],
    'sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [*files, skill]},
}
(ROOT / 'tooling/agent_tooling_install_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(f'Installed {len(files)} configuration/rule files plus previously installed skill; TOML parse passed.')
