"""Baseline import resolution/re-export checks; no scientific inputs or fits."""
import ast
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('name', ['baselines_b0', 'baselines_b1'])
def test_baselines_reference_concrete_grouped_import_paths(name):
    tree = ast.parse((ROOT/'c_ept/models'/f'{name}.py').read_text(encoding='utf-8'))
    references = {(node.module, alias.name) for node in tree.body if isinstance(node, ast.ImportFrom)
                  for alias in node.names}
    assert {('c_ept.data.casts','narrow'), ('c_ept.data.head_rows','HeadData'),
            ('c_ept.data.head_rows','HeadScores'), ('c_ept.data.head_rows','HEAD_NAMES'),
            ('c_ept.data','head_rows'), ('c_ept.data','store_core'),
            ('c_ept.preprocessing','features'), ('c_ept.models','heads'),
            ('c_ept.models','nll')} <= references
    assert all(module != 'c_ept' or symbol == 'config' for module,symbol in references)
    assert not any(module in ('c_ept.casts','c_ept.head_rows') for module,_ in references)


@pytest.mark.parametrize('canonical_first', [False, True])
def test_baseline_public_reexports_keep_canonical_identity(canonical_first):
    script = r'''
import importlib,sys,types
class DenyProtected:
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='research_pipeline.cohort_policy' or fullname.startswith('research_pipeline.holdout_r3'):
            raise AssertionError('protected import')
sys.meta_path.insert(0,DenyProtected())
fake=types.ModuleType('c_ept.cohort')
def refuse(*args,**kwargs): raise AssertionError('actual cohort operation forbidden')
fake.role_of=fake.match_list=refuse
sys.modules['c_ept.cohort']=fake
from c_ept.data import casts,head_rows
names='HEAD_NAMES KINDS MIX N_CELLS N_GOLD N_K N_LAYER N_M N_Q N_SIZE N_ZONE OPEN_HEAD_NAMES PREDICT_ALL_WIDTHS TEAMS HeadBatch HeadData HeadScores'.split()
for name in ('baselines_b0','baselines_b1'):
    old='c_ept.'+name; canonical='c_ept.models.'+name
    order=(canonical,old) if sys.argv[1]=='True' else (old,canonical)
    a,b=map(importlib.import_module,order)
    assert a is b
    assert a.narrow is casts.narrow
    assert all(getattr(a,n) is getattr(head_rows,n) for n in names)
    assert a.head_rows is head_rows
    assert a.nll_scores is importlib.import_module('c_ept.models.nll')
print('baseline modules and 17 shared exports preserve identity')
'''
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',CUDA_VISIBLE_DEVICES='',
             OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
             MKL_THREADING_LAYER='SEQUENTIAL',KMP_DUPLICATE_LIB_OK='FALSE')
    result=subprocess.run([sys.executable,'-B','-c',script,str(canonical_first)],cwd=ROOT,
                          env=env,capture_output=True,text=True,encoding='utf-8',timeout=60)
    assert result.returncode==0,result.stdout+result.stderr
