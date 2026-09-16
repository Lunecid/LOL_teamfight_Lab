import sys, time
sys.path.insert(0, 'scripts')
import fc20260915_common as C
import oc20260915_common as K
item = dict(
    id='ER1', type='documentation_semantics_correction',
    found_by='Codex source review (CODEX_SOURCE_SEMANTICS_NOTE.md), verified by Claude in source',
    affected_text=['protocol.json analysis.unknown_objective_team_count.semantics', 'feature_evidence.json unknown_objective_team_count.semantics',
                   'scripts/oc20260915_common.py UNKNOWN_SEMANTICS'],
    incorrect='listed DRAGON_SOUL_GIVEN with teamId not 100/200 as incrementing the counter',
    correct=('in StateV2 the wrapper (gameplay/state_value_v2.py StateBuilder.__init__) removes DRAGON_SOUL_GIVEN events whose teamId is not '
             '100/200 before the legacy builder and counts them separately as unassigned_soul_events (not a model input); the V2 counter '
             'therefore counts only unidentified-team ELITE_MONSTER_KILL and BUILDING_KILL/TURRET_PLATE_DESTROYED events'),
    source=dict(state_value_v2=str(C.WT / 'gameplay' / 'state_value_v2.py'), state_value_v2_sha256=C.sha256_file(C.WT / 'gameplay' / 'state_value_v2.py'),
                state_value=str(C.WT / 'gameplay' / 'state_value.py'), state_value_sha256=C.sha256_file(C.WT / 'gameplay' / 'state_value.py')),
    impact=('none on the frozen 176/185 feature decision, retention of the column, models, labels, metrics or validation; frozen files kept '
            'unchanged for provenance; REPORT.md and DEFINITION_AND_EVIDENCE.md use the corrected V2 description'))
print(C.write_json(K.OUT / 'errata.json', dict(created_at=time.strftime('%Y-%m-%d %H:%M:%S'), items=[item])))
