"""Static active-source import hygiene; deliberate compatibility APIs remain."""
import ast
import importlib.util
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
COMPAT_TESTS={'c_ept/tests/test_module_layout.py','c_ept/tests/test_visible_root.py'}


def legacy_names():
    return {p.stem for p in (ROOT/'c_ept/_compat').glob('*.py') if p.stem!='__init__'}


def old_references(tree, package):
    aliases=legacy_names()
    result=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.ImportFrom):
            module=node.module or ''
            if node.level:
                module=importlib.util.resolve_name('.'*node.level+module,package)
            if module=='c_ept':
                result.extend((node.lineno,a.name) for a in node.names if a.name in aliases)
            elif module.startswith('c_ept.') and module.split('.')[1] in aliases:
                result.append((node.lineno,module))
        elif isinstance(node,ast.Import):
            result.extend((node.lineno,a.name) for a in node.names
                          if a.name.startswith('c_ept.') and a.name.split('.')[1] in aliases)
    return result


def test_active_ept_code_and_ordinary_callers_use_canonical_import_paths():
    violations=[]
    for folder in ('c_ept','tools','tests','common_training_r1'):
        for path in (ROOT/folder).rglob('*.py'):
            rel=path.relative_to(ROOT).as_posix()
            if rel=='c_ept/cohort.py' or rel in COMPAT_TESTS or any(
                part in path.parts for part in ('_compat','__pycache__','.venv')):
                continue
            # Never inspect protected cohort/split sources or research inputs.
            tree=ast.parse(path.read_text(encoding='utf-8-sig'))
            package='.'.join(path.relative_to(ROOT).parts[:-1])
            violations.extend((rel,line,name) for line,name in old_references(tree,package))
    assert not violations,violations


@pytest.mark.parametrize('code,package', [
    ('from c_ept import head_rows','c_ept.ept'),
    ('from c_ept.casts import narrow','c_ept.ept'),
    ('import c_ept.reader','c_ept.ept'),
    ('from .. import terrain','c_ept.ept'),
])
def test_hygiene_guard_detects_legacy_absolute_and_relative_paths(code,package):
    assert old_references(ast.parse(code),package)


@pytest.mark.parametrize('code', ['from c_ept import config',
    'from c_ept.data.casts import narrow','from c_ept.preprocessing import terrain',
    'from . import checkpoints'])
def test_hygiene_guard_accepts_canonical_and_unmoved_anchors(code):
    assert not old_references(ast.parse(code),'c_ept.ept')
