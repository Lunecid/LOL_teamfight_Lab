"""Bind validated output, preserve legacy loader interface and record local usage."""
from pathlib import Path
import json,argparse
from validate import sha,dump
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
def main(run):
 v=json.loads((run/'VALIDATION.json').read_text(encoding='utf-8'));assert v['pass'] and v['full_population']
 old=ROOT/'multiwindow_r2';contract=json.loads((old/'TRAINING_DATA_CONTRACT.json').read_text(encoding='utf-8'))
 contract.update(id='MW60_FRAME_r3',production=run.name,validation=str(run.name+'/VALIDATION.json'),knowledge_index='../multiwindow_r2/knowledge_r2/INDEX.json',source_audit='../full_fit_raw_audit_20261018/full_02/MATCH_AUDIT.jsonl',source_validation_sha256=sha(run/'VALIDATION.json'),stop_roles=[],dev_roles=[],cal_roles=[],audit_roles=[],test_roles=[],matched_size_experiment_flags_source=None,no_automatic_redefinition_of_ablation_cohorts=True,full_v5_pipeline_ready=False,core_observed_outcomes_ready=True,training_performed=False,review_required_before_training=True,selection_supported_role='audit metadata only; never model input; unsupported boundary rows have all supervised target masks false',horizon_ms=[60000,120000,180000],event_interval='(start, endpoint]',population_function='studies.observable_state_20261002.population.build_population',holdout_note='FIT-only generation. Existing r2 holdout datasets preserved; no claim of newly audited holdouts or redefined matched-N ablations.')
 dump(HERE/'TRAINING_DATA_CONTRACT.json',contract)
 loader=(old/'dataset.py').read_text(encoding='utf-8')
 loader=loader.replace('import json','import json\nimport hashlib\nfrom holdout_r3.split import fit_role_names')
 loader=loader.replace("    if not validation['pass']:raise ValueError('Ledger validation did not pass')", "    if not validation['pass'] or not validation['full_population']:raise ValueError('Full ledger validation required')\n    if hashlib.sha256((directory/'BUILD_SUMMARY.json').read_bytes()).hexdigest()!=validation['build_summary_sha256']:raise ValueError('Build summary changed after validation')")
 loader=loader.replace("    if not requested or any(role.startswith('UNUSED') for role in requested):raise ValueError('Explicit active split roles required')", "    if not requested or not requested<=fit_role_names():raise ValueError('This ledger contains explicit FIT roles only')")
 loader=loader.replace("        with np.load(directory/'shards'/sn,allow_pickle=False) as z:", "        if hashlib.sha256((directory/'shards'/sn).read_bytes()).hexdigest()!=validation['shard_sha256'][sn]:raise ValueError('Shard changed after validation: '+sn)\n        with np.load(directory/'shards'/sn,allow_pickle=False) as z:")
 loader=loader.replace("'endpoint_ms','actual_elapsed_ms')", "'endpoint_ms','actual_elapsed_ms','selection_supported')")
 (HERE/'dataset.py').write_text(loader,encoding='utf-8')
 usage=[]
 for p in sorted(HERE.glob('strata_*review.json')):
  r=json.loads(p.read_text(encoding='utf-8'));usage.append({'file':p.name,'usage':r.get('usage',{}),'finish_reason':r.get('finish_reason',r.get('metrics',{}).get('finish_reason'))})
 totals={k:sum(x['usage'].get(k,0) for x in usage) for k in ('prompt_tokens','completion_tokens','total_tokens')}
 dump(HERE/'STRATA_USAGE.json',{'calls':usage,'local_token_totals':totals,'cloud_subscription_savings_measured':False,'decisions':['Reject remove-limit recommendation because it conflicts with the frozen cohort contract.','Reject future-mask/ordinal claims after raw ordinal call-site check; local narrowed review retracted.','Reject reference review suggestions that change agreed semantics: terminal 0 means ongoing; transition and boundary uncertainty masks are approved policy.','One length-truncated review narrowed once; no model/config changes, no raw source games sent.']})
 evidence={str(p.relative_to(HERE)):sha(p) for p in HERE.glob('*.py')}
 evidence.update({str(p.relative_to(HERE)):sha(p) for p in HERE.glob('*.json')})
 dump(run/'DELIVERY_MANIFEST.json',{'status':'VALIDATED_FIT_LEDGER_AWAITING_USER_REVIEW_BEFORE_TRAINING','validation_sha256':sha(run/'VALIDATION.json'),'review_manifest_sha256':sha(run/'review/MANIFEST.json'),'loader_checks_sha256':sha(run/'LOADER_AND_DELIVERY_CHECKS.json') if (run/'LOADER_AND_DELIVERY_CHECKS.json').exists() else None,'code_and_contract_sha256':evidence,'training_performed':False})
 print(json.dumps({'local_tokens':totals,'ledger':str(run),'ready_for_user_review':True},ensure_ascii=False))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('run',type=Path);main(a.parse_args().run)
