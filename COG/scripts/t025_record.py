from t025_common import *
import shutil
out=provenance()
for n in ['information_T','information_S','definition','integrity','assist_outcome','verification']:
 out[n]=json.loads((OUT/(n+'.json')).read_text(encoding='utf-8'))
out['unseen']=dict(status='BLOCKED',http_status=401,first_attempt='Sandbox network retries exhausted',network_retry='AuthenticationRejected HTTP 401',collection_matches=0,score_status='NOT_RUN')
out['execution_notes']=['Definition smoke initially used wrong endpoint_validity positional order and rejected all rows; fixed before main execution; no failed smoke metrics used.','Initial information-group order differed from frozen numeric order and gave small solver differences. Preserved initial outputs, restored original relative column order for every subset, reran both cohorts; final full-arm probabilities match frozen q exactly.','No TEST-based candidate selection; four declared tree candidates selected by Q_SELECT. These are post-TEST-exposure exploratory analyses.']
out['sources']={p.name:sha(p) for p in sorted((REPO/'scripts').glob('t025_*.py'))}
(REPO/'docs/THESIS_FOLLOWUP_20260923.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('aggregate ledger saved')
