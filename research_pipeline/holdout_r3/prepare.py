"""Versioned, narrowly transformed r3 holdout producers; prior artifacts untouched."""
from pathlib import Path
import hashlib,json,shutil
H=Path(__file__).resolve().parent;R=H.parent
def replace(s,a,b):
    assert a in s,a
    return s.replace(a,b)
def write(n,s):(H/n).write_text(s,encoding='utf-8')
split='''import hashlib
ROLES={'STOP_15.20','DEV_15.21','CAL_15.21','AUDIT_15.21','TEST_15.22'}
def bucket(mid):
    mid=mid.upper() if '_' in mid else 'KR_'+mid
    return int(hashlib.sha256(('rift-v5-multipatch-r1:'+mid).encode()).hexdigest()[:16],16)%100
def roles(patch,b):
    if patch in ('15.18','15.19'):return (('FIT_'+patch,'FIT') if b<=79 else ('UNUSED_RESERVED_'+patch,'UNUSED'))
    if patch=='15.20':return ('FIT_15.20','FIT') if b<=79 else ('STOP_15.20','STOP')
    if patch=='15.21':
        m='DEV' if b<=49 else ('CAL' if b<=74 else 'AUDIT')
        return m+'_15.21',m
    if patch=='15.22':return 'TEST_15.22','TEST'
    return 'OTHER','OTHER'
'''
write('split.py',split)
s=(R/'full_fit_raw_audit_20261018/audit_raw.py').read_text(encoding='utf-8')
s=replace(s,"PATCHES=('15.18','15.19','15.20')","PATCHES=('15.20','15.21','15.22')\nfrom split import roles")
# Scope selection uses raw header patch and the unchanged hash; no outcomes.
s=replace(s,"if patch not in PATCHES:return {'match_id':mid,'patch':patch,'scope':'OTHER_PATCH'},None","if patch not in PATCHES or roles(patch,bucket(mid))[1] not in ('STOP','DEV','CAL','AUDIT','TEST'):return {'match_id':mid,'patch':patch,'scope':'OTHER_PATCH'},None")
s=replace(s,"if actual not in PATCHES:","if actual not in PATCHES or roles(actual,bucket(mid))[1] not in ('STOP','DEV','CAL','AUDIT','TEST'):")
s=replace(s,"r['scope']='FIT_CANDIDATE'","r['scope']='HOLDOUT_CANDIDATE'")
s=replace(s,"r['scope']='FIT_CANDIDATE' if","r['scope']='HOLDOUT_CANDIDATE' if") if "r['scope']='FIT_CANDIDATE' if" in s else s
# A global -1 mask would reveal a future soul conflict in earlier X. Record its first event instead.
s=replace(s,"if soul[z]==4 and soul_time is None:soul_time=t","if soul[z]==4 and soul_time is None:soul_time=t\n    if all(x>=4 for x in soul) and not em['soul:both_teams_have_four_dragons']:em['soul:both_teams_have_four_dragons'].append(ordinal)")
s=replace(s," if all(x>=4 for x in soul):em['soul:both_teams_have_four_dragons'].append(-1)","")
# Main is supplied separately, keeping the audited analyze/audit_file implementation intact.
s=s[:s.index('def main():')]
write('audit_core.py',s)
s=(R/'multiwindow_r3/producer.py').read_text(encoding='utf-8')
s=replace(s,"AUDIT = ROOT/'full_fit_raw_audit_20261018/full_02/MATCH_AUDIT.jsonl'","AUDIT = HERE/'audit_01/MATCH_AUDIT.jsonl'\nfrom split import roles, bucket")
s=replace(s,"split_role='FIT_'+patch,macro_role='FIT'","split_role=roles(patch,bucket(mid))[0],macro_role=roles(patch,bucket(mid))[1]")
s=replace(s,"'patch':patch,'detail_sha256':dh","'patch':patch,'split_role':roles(patch,bucket(mid))[0],'macro_role':roles(patch,bucket(mid))[1],'detail_sha256':dh")
s=replace(s,"paths += [AUDIT,ROOT/","paths += [p for p in (ROOT/'multiwindow_r3').rglob('*') if p.is_file() and '__pycache__' not in p.parts]\n    paths += [AUDIT,ROOT/")
s=replace(s,"manifest=json.loads((ROOT/'full_fit_raw_audit_20261018/review_02/ARTIFACT_MANIFEST.json').read_text(encoding='utf-8'))","manifest=json.loads((HERE/'audit_01/VALIDATION.json').read_text(encoding='utf-8'))")
s=replace(s,"assert sha(AUDIT)==manifest['files'][str(Path('full_02')/'MATCH_AUDIT.jsonl')]['sha256']","assert manifest['passed'] and sha(AUDIT)==manifest['audit_sha256']")
s=replace(s,"audit_validation=json.loads((ROOT/'full_fit_raw_audit_20261018/review_02/VALIDATION.json').read_text(encoding='utf-8'));assert audit_validation['passed']","assert manifest['passed']")
s=replace(s,"'profile':'MW60_FRAME_r3'","'profile':'MW60_FRAME_r3_holdout'")
write('producer.py',s)
shutil.copyfile(R/'multiwindow_r3/reference.py',H/'reference.py')
s=(R/'multiwindow_r3/validate.py').read_text(encoding='utf-8')
s=replace(s,"from reference import validate_match","from reference import validate_match\nfrom split import roles,bucket,ROLES")
s=replace(s,"assert str(r['split_role'])=='FIT_'+aud[mid]['patch'] and str(r['macro_role'])=='FIT'","assert (str(r['split_role']),str(r['macro_role']))==roles(aud[mid]['patch'],bucket(mid)) and str(r['split_role']) in ROLES")
s=replace(s,"assert int(hashlib.sha256(('rift-v5-multipatch-r1:'+mid).encode()).hexdigest()[:16],16)%100<=79","assert aud[mid]['bucket']==bucket(mid)")
s=replace(s,"selected=('row_id'","selected=('split_role','macro_role','row_id'")
s=replace(s,"auditpath=ROOT/'full_fit_raw_audit_20261018/full_02/MATCH_AUDIT.jsonl'","auditpath=HERE/'audit_01/MATCH_AUDIT.jsonl'")
s=replace(s,"np.char.startswith(z['split_role'],'FIT_')","np.isin(z['split_role'],list(ROLES))")
s=s.replace('prior_r2_fit_','prior_r2_holdout_')
write('validate.py',s)
s=(R/'multiwindow_r3/stream_validate.py').read_text(encoding='utf-8').replace("ROOT/'full_fit_raw_audit_20261018/full_02/MATCH_AUDIT.jsonl'","ROOT/'holdout_r3/audit_01/MATCH_AUDIT.jsonl'")
write('stream_validate.py',s)
s=(R/'multiwindow_r3/dataset.py').read_text(encoding='utf-8').replace("{'FIT_15.18','FIT_15.19','FIT_15.20'}","{'STOP_15.20','DEV_15.21','CAL_15.21','AUDIT_15.21','TEST_15.22'}").replace('explicit FIT roles only','explicit held-out roles only')
write('dataset.py',s)
write('PREPARATION.json',json.dumps({'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [R/'multiwindow_r3/producer.py',R/'multiwindow_r3/reference.py',R/'multiwindow_r3/validate.py',R/'full_fit_raw_audit_20261018/audit_raw.py']},'unchanged_policy':['split salt and ranges','population selection','frame freshness','mask policy','endpoints','targets'],'implementation_fix':'future soul conflict mask now starts at first conflicting original ordinal; zero instances in FIT','training':False},indent=2))
print('Prepared separately versioned holdout scripts.')
