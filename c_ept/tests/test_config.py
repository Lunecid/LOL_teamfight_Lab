import hashlib
import json
import os
import sys

import pytest

from c_ept import config as C


def test_constants_match_spec():
    sha = C.check_spec()
    assert len(sha) == 64
    assert (C.G_MS, C.L_U, C.TAU_LEAD_MS, C.DELTA_AFTER_F_MS) == (15_000, 2_000.0, 15_000, 20_000)
    assert (C.SCOPE_FIRST_MS, C.END_CUT_MS, C.H_OCC_MS, C.BIN_MS, C.N_BINS, C.CONV_MS) == (120_000, 30_000, 180_000, 10_000, 18, 60_000)
    assert C.EVAL_H_MS == (60_000, 120_000, 180_000) and (C.RECENT_MS, C.OLD_MS) == (120_000, 600_000)
    assert (C.MAX_EVENTS, C.MAX_EDGES) == (128, 128)            # S2 gate S2G-1, 2026-10-21: the event cap was 80
    assert (C.W, C.N_MARKS, len(C.O_KINDS), len(C.FRAME_FIELDS)) == (14_875.0, 52, 9, 33)
    # the bin grid, the occurrence window and the evaluation horizons are one grid
    assert C.N_BINS * C.BIN_MS == C.H_OCC_MS == C.EVAL_H_MS[-1]
    assert all(h % C.BIN_MS == 0 for h in C.EVAL_H_MS)
    # name lists, spelled out independently of the module
    assert C.PATCHES == ('15.18', '15.19', '15.20', '15.21', '15.22', '16.13')
    assert C.ROLES == ('FIT', 'UNUSED', 'STOP', 'DEV', 'CAL', 'AUDIT', 'TEST', 'EXTERNAL')
    assert C.O_KINDS == ('dragon', 'elder', 'grubs', 'herald', 'baron', 'atakhan', 'tower', 'inhibitor', 'plate')
    assert C.EVENT_TYPES == ('CHAMPION_KILL', 'ELITE_MONSTER_KILL', 'BUILDING_KILL', 'TURRET_PLATE_DESTROYED',
                             'ITEM_PURCHASED', 'LEVEL_UP', 'WARD_PLACED', 'WARD_KILL')
    assert len(C.CHAMPION_STATS_KEYS) == 25 and list(C.CHAMPION_STATS_KEYS) == sorted(C.CHAMPION_STATS_KEYS)
    assert (C.CHAMPION_STATS_KEYS[0], C.CHAMPION_STATS_KEYS[-1]) == ('abilityHaste', 'spellVamp')
    assert C.FRAME_FIELDS[:8] == ('x', 'y', 'level', 'xp', 'currentGold', 'totalGold', 'minionsKilled', 'jungleMinionsKilled')
    assert C.FRAME_FIELDS[8:] == tuple('cs_' + k for k in C.CHAMPION_STATS_KEYS)
    # path defaults point at the real layout (patch folders hold raw and cache), and the catalog index file name is fixed
    assert (C.RAW_ROOT / '15.18' / 'raw' / 'kr' / 'timeline').is_dir()
    assert (C.CACHE_ROOT / '15.18' / 'cache' / 'ext_kr_15_18_22').is_dir()
    assert C.RUNS_ROOT.name == 'runs' and C.CATALOG_DB.name == 'runs.sqlite' and C.CATALOG_DB.parent.name == 'catalog'
    assert (C.LEGACY_ROOT / 'gameplay').is_dir() and (C.SPLIT_DIR / 'split.py').is_file() and C.SPEC_PATH.is_file()
    # the constants dict that runlog hashes holds numbers and name lists only (no paths)
    consts = C.constants()
    assert consts['G_MS'] == 15_000 and consts['FRAME_FIELDS'] == list(C.FRAME_FIELDS)
    assert not {'RAW_ROOT', 'CACHE_ROOT', 'RUNS_ROOT', 'CATALOG_DB', 'LEGACY_ROOT', 'SPLIT_DIR', 'SPEC_PATH'} & set(consts)
    json.dumps(consts)


def test_check_spec_detects_changed_constant(tmp_path):
    base = json.loads(C.SPEC_PATH.read_text(encoding='utf-8'))

    def write(spec, name='spec.json'):
        p = tmp_path / name
        p.write_text(json.dumps(spec), encoding='utf-8')
        return p

    s = json.loads(json.dumps(base))
    s['operating_point']['G_ms'] = 13000
    with pytest.raises(C.SpecMismatch):
        C.check_spec(write(s))

    # every other compared field is detected as well
    def set_op(key, value):
        t = json.loads(json.dumps(base))
        t['operating_point'][key] = value
        return t
    for bad in (set_op('L_u', 1500.0), set_op('tau_lead_ms', 13000), set_op('presence_gate', 4)):
        with pytest.raises(C.SpecMismatch):
            C.check_spec(write(bad))
    t = json.loads(json.dumps(base))
    del t['operating_point']['presence_gate']
    with pytest.raises(C.SpecMismatch):
        C.check_spec(write(t))
    t = json.loads(json.dumps(base))
    del t['operating_point']['size_classes']['ASYM']
    with pytest.raises(C.SpecMismatch, match='ASYM'):
        C.check_spec(write(t))
    with pytest.raises(C.SpecMismatch):
        C.check_spec(write({'no': 'operating_point'}))

    # the record keeps growing (more decisions, more keys): equal constants still pass, and the hash is that of the file bytes
    grown = json.loads(json.dumps(base))
    grown['G99_later_decision'] = {'anything': [1, 2, 3]}
    grown['operating_point']['extra_note'] = 'x'
    p = write(grown, 'grown.json')
    assert C.check_spec(p) == hashlib.sha256(p.read_bytes()).hexdigest()


def test_importing_the_split_module_writes_no_bytecode(tmp_path, monkeypatch):
    """The split folder is read only and its module holds the salt as a literal, so a .pyc there would hold it too. A dummy ``split.py``
    in tmp_path stands in for the real module (whose text is never read)."""
    (tmp_path / 'split.py').write_text('def bucket(key):\n    return 7\n\n\ndef roles(patch, bucket):\n    return "X_" + patch, "FIT"\n',
                                       encoding='utf-8')
    monkeypatch.setattr(C, 'SPLIT_DIR', tmp_path)
    monkeypatch.setattr(C, '_SPLIT', None)                          # forget the module that is cached
    monkeypatch.setattr(sys, 'path', list(sys.path))                # the insertion of SPLIT_DIR is undone with the test
    monkeypatch.setattr(sys, 'dont_write_bytecode', False)          # python without -B (the suite is usually started with -B)
    monkeypatch.setattr(sys, 'pycache_prefix', None)                # a prefix would send the .pyc elsewhere and hide the defect
    saved = sys.modules.pop('split', None)
    try:
        mod = C.split_module()
        assert os.path.samefile(mod.__file__, tmp_path / 'split.py') and mod.bucket('k') == 7
        assert not (tmp_path / '__pycache__').exists() and not list(tmp_path.rglob('*.pyc'))
        assert sys.dont_write_bytecode is False                     # the flag is put back as it was
        assert C.split_module() is mod                              # the import is still cached
    finally:
        sys.modules.pop('split', None)
        if saved is not None:
            sys.modules['split'] = saved


def test_the_bytecode_flag_is_put_back_when_the_import_fails(tmp_path, monkeypatch):
    (tmp_path / 'split.py').write_text('raise RuntimeError("the dummy module fails to import")\n', encoding='utf-8')
    monkeypatch.setattr(C, 'SPLIT_DIR', tmp_path)
    monkeypatch.setattr(C, '_SPLIT', None)
    monkeypatch.setattr(sys, 'path', list(sys.path))
    monkeypatch.setattr(sys, 'dont_write_bytecode', False)
    monkeypatch.setattr(sys, 'pycache_prefix', None)
    saved = sys.modules.pop('split', None)
    try:
        with pytest.raises(RuntimeError, match='fails to import'):
            C.split_module()
        assert sys.dont_write_bytecode is False and C._SPLIT is None     # the flag is back, nothing is cached
        assert not (tmp_path / '__pycache__').exists()                   # the bytecode of a module is written before it runs
    finally:
        sys.modules.pop('split', None)
        if saved is not None:
            sys.modules['split'] = saved
