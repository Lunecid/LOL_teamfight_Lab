import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from c_ept import config as C
from c_ept import legacy


def _pycache_state():
    d = C.LEGACY_ROOT / 'gameplay' / '__pycache__'
    return sorted((p.name, p.stat().st_mtime_ns) for p in d.glob('*')) if d.is_dir() else []


def test_pins_match_current_files(tmp_path, monkeypatch):
    before = _pycache_state()
    ns = legacy.load(); assert ns.head.startswith('5cf5cac')
    assert len(ns.head) == 40 and ns is legacy.load()          # imported once per process, cached
    # the four modules are the pinned gameplay files, and loading wrote no bytecode into the read-only LEGACY tree
    for name in ('event_survival', 'objective_timers', 'item_state', 'champion_attributes'):
        assert getattr(ns, name).__file__.replace('\\', '/').endswith(f'gameplay/{name}.py')
    assert callable(ns.event_survival.death_intervals) and callable(ns.objective_timers.feature_names)
    assert callable(ns.item_state.item_vectors_at) and ns.champion_attributes is not None
    assert _pycache_state() == before
    # the pin list is exactly the planned set: 6 gameplay modules, 5 rule files, champions/items for each patch
    files = legacy.pins()['files']
    assert len(files) == 6 + 5 + 2 * len(C.PATCHES) and sorted(files) == legacy.expected_files()
    # the pin-file digest that goes into run manifests survives a checkout with CRLF line endings
    crlf = tmp_path / 'legacy_pins.json'
    crlf.write_bytes(legacy.PINS_PATH.read_bytes().replace(b'\n', b'\r\n'))
    digest = legacy.pins_sha256()
    monkeypatch.setattr(legacy, 'PINS_PATH', crlf)
    assert b'\r\n' in crlf.read_bytes() and legacy.pins_sha256() == digest


def test_pin_mismatch_refuses(tmp_path, monkeypatch):
    real = C.LEGACY_ROOT
    skip = shutil.ignore_patterns('__pycache__')
    shutil.copytree(C.LEGACY_ROOT / 'gameplay', tmp_path / 'gameplay', ignore=skip)
    shutil.copytree(C.LEGACY_ROOT / 'config', tmp_path / 'config', ignore=skip)
    # a byte-identical copy elsewhere passes the pins and is what gets imported; importing writes no bytecode (the tree is read only)
    code = ('import sys; from pathlib import Path; from c_ept import config as C, legacy; C.LEGACY_ROOT = Path(sys.argv[1]); '
            'ns = legacy.load(); print(ns.head, ns.event_survival.__file__)')
    env = {k: v for k, v in os.environ.items() if k != 'PYTHONDONTWRITEBYTECODE'}
    r = subprocess.run([sys.executable, '-c', code, str(tmp_path)], cwd=Path(C.__file__).parents[1], env=env,
                       capture_output=True, text=True, encoding='utf-8')
    assert r.returncode == 0, r.stderr
    head, loaded_from = r.stdout.strip().split(' ', 1)
    assert head == legacy.pins()['head'] and Path(loaded_from).resolve().is_relative_to(tmp_path.resolve())
    assert not list(tmp_path.rglob('__pycache__'))
    (tmp_path / 'gameplay' / 'respawn_rules.py').write_text('# changed\n', encoding='utf-8')
    monkeypatch.setattr(C, 'LEGACY_ROOT', tmp_path)
    with pytest.raises(legacy.LegacyPinError, match='respawn_rules.py'): legacy.load()
    # a missing pinned file is refused by name too, and every offender is listed
    (tmp_path / 'config' / 'game_rules' / 'objective_rules.json').unlink()
    with pytest.raises(legacy.LegacyPinError, match='objective_rules.json.*missing|missing.*objective_rules.json'): legacy.load()
    with pytest.raises(legacy.LegacyPinError, match='respawn_rules.py'): legacy.verify()
    # a refusal does not poison the cache: the real tree loads again
    monkeypatch.setattr(C, 'LEGACY_ROOT', real)
    assert legacy.load().head.startswith('5cf5cac')
