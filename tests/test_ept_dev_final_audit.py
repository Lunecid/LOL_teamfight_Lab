"""Synthetic final-audit binding; no actual run or payload access."""
import json
from pathlib import Path
import pytest
from c_ept.evaluation import guard


@pytest.fixture
def completed(tmp_path):
    reports = tmp_path/'reports'; reports.mkdir()
    for name, content in [('FINAL_REPORT_KO.md', 'synthetic report'), ('curve.png', 'synthetic image stub'),
                          ('history.csv', 'epoch,loss\n1,1\n'), ('history.json', '{"synthetic": true}')]:
        (reports/name).write_text(content, encoding='utf-8')
    c = dict(training_control=str(tmp_path), training_source=str((tmp_path/'source').resolve()),
             training_run=str((tmp_path/'run').resolve()), final_report_sha256=guard.sha_file(reports/'FINAL_REPORT_KO.md'))
    audit = dict(schema='ept-pilot-final-audit-v1', state='completed', code_commit=guard.TRAIN_COMMIT,
                 operator_config_sha256=guard.TRAIN_CONFIG, run_identity_sha256=guard.TRAIN_ID,
                 run=c['training_run'], source=c['training_source'], manual_visual_review='verified_against_json_csv',
                 final_report_sha256=c['final_report_sha256'],
                 reviewed_artifacts={name:guard.sha_file(reports/name) for name in ('curve.png','history.csv','history.json')})
    def write():
        (reports/'final-audit.json').write_text(json.dumps(audit), encoding='utf-8')
        c['final_audit_sha256'] = guard.sha_file(reports/'final-audit.json')
    write()
    return c, audit, write


def test_bound_completed_review(completed):
    c, _, _ = completed
    assert len(guard.validate_final_audit(c)['reviewed_artifacts']) == 3


@pytest.mark.parametrize('key,value', [('schema','wrong'), ('state','failed'), ('state','running'),
    ('manual_visual_review','PENDING'), ('code_commit','0'*40), ('run_identity_sha256','0'*64),
    ('operator_config_sha256','0'*64), ('run','C:/wrong/run'), ('source','C:/wrong/source'),
    ('final_report_sha256','0'*64), ('reviewed_artifacts',{})])
def test_wrong_semantics_even_when_hash_matches(completed, key, value):
    c, audit, write = completed
    audit[key] = value; write()
    with pytest.raises(guard.EvaluationRefused): guard.validate_final_audit(c)


@pytest.mark.parametrize('key', ['schema','state','manual_visual_review','run_identity_sha256','reviewed_artifacts'])
def test_missing_field(completed, key):
    c, audit, write = completed
    del audit[key]; write()
    with pytest.raises(guard.EvaluationRefused): guard.validate_final_audit(c)


def test_numerical_artifact_tamper(completed):
    c, _, _ = completed
    (Path(c['training_control'])/'reports/history.csv').write_text('changed', encoding='utf-8')
    with pytest.raises(guard.EvaluationRefused, match='artifact hash'): guard.validate_final_audit(c)


def test_requires_actual_png_reference(completed):
    c, audit, write = completed
    del audit['reviewed_artifacts']['curve.png']; write()
    with pytest.raises(guard.EvaluationRefused, match='PNG'): guard.validate_final_audit(c)


def test_dirty_source_never_executes_parent_probe(tmp_path):
    c = dict(source=str(Path(guard.__file__).resolve().parents[2]), training_source=str(tmp_path), reviewed_commit='a'*40)
    called=[]
    def reject(*args): raise guard.EvaluationRefused('dirty')
    with pytest.raises(guard.EvaluationRefused, match='dirty'):
        guard.preflight(c, source_probe=reject, parent_probe=lambda c:called.append(c))
    assert called == []
