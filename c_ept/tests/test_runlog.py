import datetime
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from c_ept import config as C
from c_ept import runlog
from c_ept.tests import synth


def test_new_run_increments(tmp_path):
    a = runlog.new_run('s1', root=tmp_path, today='20261020')
    b = runlog.new_run('s1', root=tmp_path, today='20261020')
    assert (a.name, b.name) == ('20261020_c_ept_s1_1', '20261020_c_ept_s1_2')
    assert a.is_dir() and b.is_dir() and a.parent == tmp_path
    # numbering continues after the highest existing number; other stages and other days count separately
    (tmp_path / '20261020_c_ept_s1_7').mkdir()
    assert runlog.new_run('s1', root=tmp_path, today='20261020').name == '20261020_c_ept_s1_8'
    assert runlog.new_run('s2', root=tmp_path, today='20261020').name == '20261020_c_ept_s2_1'
    assert runlog.new_run('s1', root=tmp_path, today='20261021').name == '20261021_c_ept_s1_1'
    for bad in ('../x', 'a b', ''):
        with pytest.raises(ValueError):
            runlog.new_run(bad, root=tmp_path, today='20261020')
    # without `today` the folder carries today's date
    assert runlog.new_run('s3', root=tmp_path).name == f'{datetime.date.today():%Y%m%d}_c_ept_s3_1'


def test_manifest_fields(tmp_path):
    m = runlog.write_manifest(tmp_path, {'args': {}})
    assert {'code_commit', 'code_dirty', 'config_sha256', 'spec_sha256', 'legacy_head', 'legacy_pins_sha256',
            'split_policy_sha256', 'python', 'numpy', 'cpu', 'ram_gib', 'gpu', 'strata_running', 'strata_detail'} <= set(m)
    assert (tmp_path / 'RUN.json').exists()
    # the file is exactly what was returned, and the caller's own fields ride along
    assert json.loads((tmp_path / 'RUN.json').read_text(encoding='utf-8')) == m and m['args'] == {}
    assert len(m['code_commit']) == 40 and isinstance(m['code_dirty'], bool)
    assert all(len(m[k]) == 64 for k in ('config_sha256', 'spec_sha256', 'legacy_pins_sha256', 'split_policy_sha256'))
    assert m['spec_sha256'] == C.check_spec() and m['legacy_head'].startswith('5cf5cac')
    assert isinstance(m['strata_running'], bool) and m['ram_gib'] > 0 and m['cpu'] and m['numpy']
    assert m['gpu'] is None or isinstance(m['gpu'], str)
    # the split-policy hash is a pure function of the role table: stable, and the split module is imported once
    assert runlog.write_manifest(tmp_path, {'args': {}})['split_policy_sha256'] == m['split_policy_sha256']
    assert C.split_module() is C.split_module()
    # provenance fields are not the caller's to set
    with pytest.raises(ValueError):
        runlog.write_manifest(tmp_path, {'code_commit': 'x'})


# ---- the public provenance helpers: one definition for RUN.json, TERRAIN.json and the S2 builder (Task 11 fix round 1) ---------------
def test_sha256_json_is_the_hash_of_the_sorted_compact_json():
    want = hashlib.sha256(b'{"a":{"x":null,"y":true},"b":[1,2],"c":"\\u00fc"}').hexdigest()          # ensure_ascii stays on
    assert runlog.sha256_json({'b': [1, 2], 'a': {'y': True, 'x': None}, 'c': '\u00fc'}) == want
    assert runlog.sha256_json({'c': '\u00fc', 'a': {'x': None, 'y': True}, 'b': [1, 2]}) == want      # the key order does not matter
    assert runlog.sha256_json([1, 2]) != runlog.sha256_json([2, 1])                                    # the order of a list does


def fake_git(head, status, calls=None):
    """A stand-in for ``runlog._git``: ``rev-parse`` gives ``head``, ``status`` gives ``status`` (None: git cannot answer)."""
    def git(*args):
        if calls is not None:
            calls.append(args)
        return head if args[0] == 'rev-parse' else status
    return git


@pytest.mark.parametrize('head, status, want', [
    ('0123abc\n', '', {'code_commit': '0123abc', 'code_dirty': False}),
    ('0123abc\n', ' M c_ept/terrain.py\n?? c_ept/new.py\n', {'code_commit': '0123abc', 'code_dirty': True}),
    ('\n', '  \n', {'code_commit': None, 'code_dirty': False}),          # an empty answer is no commit; a blank status is clean
    (None, None, {'code_commit': None, 'code_dirty': None}),              # git cannot answer: unknown, not clean
], ids=['clean', 'dirty', 'empty-answers', 'git-absent'])
def test_code_state_is_the_commit_and_the_dirty_flag_of_the_package(monkeypatch, head, status, want):
    calls = []
    monkeypatch.setattr(runlog, '_git', fake_git(head, status, calls))
    assert runlog.code_state() == want
    assert sorted(calls) == [('rev-parse', 'HEAD'), ('status', '--porcelain', '--', 'c_ept')]


def test_the_manifest_takes_the_commit_and_the_flag_from_code_state(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, '_git', fake_git('c' * 40 + '\n', ' M c_ept/x.py\n'))
    m = runlog.write_manifest(tmp_path, {'args': {}})
    assert (m['code_commit'], m['code_dirty']) == ('c' * 40, True) == tuple(runlog.code_state().values())
    monkeypatch.setattr(runlog, '_git', fake_git(None, None))
    m = runlog.write_manifest(tmp_path / 'other', {'args': {}})
    assert (m['code_commit'], m['code_dirty']) == (None, None)


def test_the_manifest_fields_that_the_helpers_made_are_unchanged(tmp_path):
    """The formulas that RUN.json used before the helpers existed, written out again: the same bytes, the same git answers."""
    def git(*args):
        done = subprocess.run(['git', '-C', str(runlog.HUB), *args], capture_output=True, text=True, encoding='utf-8', check=True)
        return done.stdout
    # Other commits land in this worktree while the suite runs (the author and parallel tasks commit here), so one commit cannot be
    # compared with a second `git rev-parse` taken later. The manifest's commit is bracketed by the HEAD before and the HEAD after the
    # write instead: it is one of the two, and the test cannot flake unless two commits land within the few seconds of the write.
    before = git('rev-parse', 'HEAD').strip()
    m = runlog.write_manifest(tmp_path, {'args': {}})
    after = git('rev-parse', 'HEAD').strip()
    assert m['code_commit'] in {before, after} and len(m['code_commit']) == 40 and isinstance(m['code_dirty'], bool)
    constants_json = json.dumps(C.constants(), sort_keys=True, separators=(',', ':')).encode('utf-8')
    assert m['config_sha256'] == hashlib.sha256(constants_json).hexdigest()
    assert m['config_sha256'] == runlog.sha256_json(C.constants())
    split = C.split_module()                                                      # only roles() is called
    table = {p: [list(split.roles(p, b)) for b in range(100)] for p in C.PATCHES}
    assert m['split_policy_sha256'] == hashlib.sha256(json.dumps(table, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()


# ---- RUN.json is replaced atomically and durably (final review M2) -------------------------------------------------------------------
def spy_on_disk_calls(monkeypatch):
    """Record every ``os.fsync`` and ``os.replace`` of the test: ``('fsync', size of the file)`` and ``('replace', source name, target
    name, the bytes of the source at that moment)``. The calls go through to the real functions."""
    events, real_fsync, real_replace = [], os.fsync, os.replace

    def fsync(fd):
        events.append(('fsync', os.fstat(fd).st_size))
        return real_fsync(fd)

    def replace(src, dst, *args, **kwargs):
        events.append(('replace', Path(src).name, Path(dst).name, Path(src).read_bytes()))
        return real_replace(src, dst, *args, **kwargs)
    monkeypatch.setattr(os, 'fsync', fsync)
    monkeypatch.setattr(os, 'replace', replace)
    return events


def assert_fsynced_then_renamed(events, run):
    """RUN.json came from a temp file whose data was fsynced first (the fsync saw the whole file), then renamed over RUN.json."""
    renames = [i for i, e in enumerate(events) if e[0] == 'replace' and e[2] == 'RUN.json']
    assert len(renames) == 1, events
    i = renames[0]
    _, source, _, data = events[i]
    assert source == 'RUN.json.tmp' and data.endswith(b'}\n') and b'\r' not in data               # the same bytes as before: LF, one final newline
    assert ('fsync', len(data)) in events[:i], 'the temp file was renamed over RUN.json before its data was fsynced'
    assert (Path(run) / 'RUN.json').read_bytes() == data and not (Path(run) / 'RUN.json.tmp').exists()


def test_write_manifest_fsyncs_the_temp_file_before_it_replaces_run_json(tmp_path, monkeypatch):
    """A power cut right after the rename must not leave a RUN.json whose data never reached the disk (final review M2)."""
    events = spy_on_disk_calls(monkeypatch)
    m = runlog.write_manifest(tmp_path, {'args': {}})
    assert_fsynced_then_renamed(events, tmp_path)
    assert json.loads((tmp_path / 'RUN.json').read_text(encoding='utf-8')) == m


def test_amend_manifest_fsyncs_the_temp_file_before_it_replaces_run_json(tmp_path, monkeypatch):
    runlog.write_manifest(tmp_path, {'args': {}})
    events = spy_on_disk_calls(monkeypatch)
    amended = runlog.amend_manifest(tmp_path, {'s2_result': {'complete': True}})
    assert_fsynced_then_renamed(events, tmp_path)
    assert json.loads((tmp_path / 'RUN.json').read_text(encoding='utf-8')) == amended and amended['s2_result'] == {'complete': True}


def test_the_durable_write_is_local_to_runlog_and_loads_no_store_layer():
    """``terrain`` imports ``runlog``, and a worker that loads the terrain must not load the store layer (test_terrain and test_tokens
    pin that for their modules): so RUN.json is written by a helper of this module and not by ``store_core.write_atomic``."""
    code = 'import sys, c_ept.runlog, c_ept.terrain; sys.stdout.write(str("c_ept.store_core" in sys.modules))'
    done = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True, cwd=runlog.HUB, timeout=300)
    assert (done.returncode, done.stdout) == (0, 'False'), done.stderr[-300:]


def test_a_cut_before_the_rename_leaves_the_old_run_json_whole(tmp_path, monkeypatch):
    runlog.write_manifest(tmp_path, {'args': {}})
    old = (tmp_path / 'RUN.json').read_bytes()

    def cut(src, dst, *args, **kwargs):
        raise OSError('power cut before the rename')
    monkeypatch.setattr(os, 'replace', cut)
    with pytest.raises(OSError, match='power cut'):
        runlog.amend_manifest(tmp_path, {'extra': 1})
    with pytest.raises(OSError, match='power cut'):
        runlog.write_manifest(tmp_path, {'args': {'again': True}})
    assert (tmp_path / 'RUN.json').read_bytes() == old                    # the reader sees the old file, never half of the new one


def test_register_upsert(tmp_path):
    db = tmp_path / 'runs.sqlite'
    m = runlog.write_manifest(tmp_path, {'args': {}})
    runlog.register(tmp_path, m, db)
    runlog.register(tmp_path, m, db)
    assert sqlite3.connect(db).execute('select count(*) from runs').fetchone()[0] == 1
    # a changed manifest replaces the row, a second run adds one; the real catalog is never touched here
    other = tmp_path / 'second_run'
    other.mkdir()
    runlog.register(tmp_path, {**m, 'stage': 's9'}, db)
    runlog.register(other, runlog.write_manifest(other, {'stage': 's2'}), db)
    con = sqlite3.connect(db)
    rows = {r[0]: r for r in con.execute('select run, stage, created, manifest from runs')}
    assert set(rows) == {tmp_path.name, 'second_run'} and rows[tmp_path.name][1] == 's9' and rows['second_run'][1] == 's2'
    assert json.loads(rows[tmp_path.name][3])['stage'] == 's9' and rows[tmp_path.name][2] == m['created']
    con.close()


# ---- E6: the Strata server is found by its command line (Ruling 33) -----------------------------------------------------------------
SERVER_PID = synth.STRATA_SERVER[0]


def test_strata_check_finds_the_server_by_its_command_line():
    rec = runlog.strata_check([synth.UNRELATED_PYTHON, synth.STRATA_SERVER, synth.STRATA_BRIDGE])
    assert rec['running'] is True and rec['matched'] == [{'name': 'python.exe', 'pid': SERVER_PID}]
    assert (rec['scanned'], rec['bridges_ignored'], rec['cmdline_unreadable']) == (3, 1, 0)


def test_strata_check_ignores_the_idle_mcp_bridges():
    bridges = [(5000 + i, 'python.exe', synth.STRATA_BRIDGE[2]) for i in range(6)]
    rec = runlog.strata_check([*bridges, synth.UNRELATED_PYTHON])
    assert rec['running'] is False and rec['matched'] == [] and rec['bridges_ignored'] == 6 and rec['scanned'] == 7


@pytest.mark.parametrize('process', [synth.LLAMA_SERVER, synth.OLLAMA_APP, (4110, 'Strata-Helper.EXE', ['strata-helper.exe'])],
                         ids=['llama-server', 'ollama', 'strata-in-the-name'])
def test_strata_check_keeps_the_name_rule(process):
    rec = runlog.strata_check([synth.UNRELATED_PYTHON, process])
    assert rec['running'] is True and rec['matched'] == [{'name': process[1], 'pid': process[0]}] and rec['bridges_ignored'] == 0


@pytest.mark.parametrize('cmdline', [
    [r'D:\Strata\.venv\Scripts\python.exe', r'D:\Strata\serve\server.py', '--port', '8080'],      # backslashes, as the launcher starts it
    ['PYTHON.EXE', 'd:/STRATA/Serve/SERVER.PY'],                                                   # forward slashes, other case
    r'"D:\Strata\.venv\Scripts\python.exe" D:\Strata\serve\server.py --engine strata',             # one string
], ids=['backslashes', 'slashes-and-case', 'one-string'])
def test_strata_check_normalises_slashes_and_case(cmdline):
    rec = runlog.strata_check([(77, 'python.exe', cmdline)])
    assert rec['running'] is True and rec['matched'] == [{'name': 'python.exe', 'pid': 77}]


@pytest.mark.parametrize('cmdline', [
    ['python.exe', r'D:\Other\app\server.py', '--port', '9000'],      # a server.py that is not Strata's
    ['python.exe', r'D:\Strata\notes.py'],                            # Strata's folder, not its server
    ['python.exe', '-m', 'pytest', 'c_ept/tests/test_runlog.py'],     # unrelated python
    None, [], '',                                                     # a protected process: the command line cannot be read
], ids=['other-server', 'strata-without-server', 'pytest', 'none', 'empty-list', 'empty-string'])
def test_strata_check_does_not_flag_what_is_not_the_server(cmdline):
    rec = runlog.strata_check([(78, 'python.exe', cmdline)])
    assert rec['running'] is False and rec['matched'] == []
    assert rec['cmdline_unreadable'] == (0 if cmdline else 1)


def test_the_bridge_exclusion_wins_over_both_rules():
    """A command line that has the bridge's name is never counted, even if it also holds 'strata' and 'server.py' or the process name says
    strata."""
    both = (80, 'python.exe', ['python.exe', 'strata_worker_mcp.py', r'D:\Strata\serve\server.py'])
    named = (81, 'strata-worker.exe', ['strata-worker.exe', 'strata_worker_mcp.py'])
    rec = runlog.strata_check([both, named])
    assert rec['running'] is False and rec['matched'] == [] and rec['bridges_ignored'] == 2


def test_the_record_lists_names_and_pids_in_pid_order_and_no_command_line():
    table = [synth.OLLAMA_APP, synth.STRATA_BRIDGE, synth.LLAMA_SERVER, synth.STRATA_SERVER]
    rec = runlog.strata_check(table)
    assert [(m['pid'], m['name']) for m in rec['matched']] == [(4101, 'python.exe'), (4103, 'llama-server.exe'), (4104, 'ollama.exe')]
    assert set(rec) == {'running', 'matched', 'bridges_ignored', 'scanned', 'cmdline_unreadable', 'rule'}
    assert rec['rule'] == runlog.STRATA_RULE                      # the only free text is the fixed rule, never part of a command line
    text = json.dumps({k: v for k, v in rec.items() if k != 'rule'})
    for token in (*synth.STRATA_SERVER[2], *synth.STRATA_BRIDGE[2], *synth.LLAMA_SERVER[2][1:]):
        assert token not in text, 'a command line leaked into the record'


def test_strata_check_without_psutil_is_unknown(monkeypatch):
    monkeypatch.setattr(runlog, 'psutil', None)
    assert runlog.strata_check() is None
    assert runlog.strata_check([synth.STRATA_SERVER])['running'] is True        # an injected list needs no psutil


def test_the_default_scan_reads_psutil_with_pid_name_and_cmdline(fake_processes):
    fake_processes([synth.STRATA_SERVER, synth.STRATA_BRIDGE, synth.UNRELATED_PYTHON])
    rec = runlog.strata_check()
    assert rec['running'] is True and rec['matched'] == [{'name': 'python.exe', 'pid': SERVER_PID}] and rec['bridges_ignored'] == 1


def test_manifest_records_the_server_that_only_the_command_line_shows(tmp_path, fake_processes):
    """The server's process name is only python.exe: a test of the name cannot see it, the command line shows it."""
    fake_processes([synth.STRATA_SERVER, synth.STRATA_BRIDGE, synth.UNRELATED_PYTHON])
    m = runlog.write_manifest(tmp_path, {'args': {}})
    assert m['strata_running'] is True
    assert m['strata_detail']['matched'] == [{'name': 'python.exe', 'pid': SERVER_PID}] and m['strata_detail']['bridges_ignored'] == 1
    assert json.loads((tmp_path / 'RUN.json').read_text(encoding='utf-8')) == m


def test_manifest_with_only_bridges_says_not_running_and_how_many_were_ignored(tmp_path, fake_processes):
    fake_processes([synth.STRATA_BRIDGE, (4106, 'python.exe', synth.STRATA_BRIDGE[2]), synth.UNRELATED_PYTHON])
    m = runlog.write_manifest(tmp_path, {'args': {}})
    assert m['strata_running'] is False and m['strata_detail']['matched'] == [] and m['strata_detail']['bridges_ignored'] == 2
    assert m['strata_detail']['scanned'] == 3 and 'rule' in m['strata_detail']


def test_manifest_without_psutil_records_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, 'psutil', None)
    m = runlog.write_manifest(tmp_path, {'args': {}})
    assert m['strata_running'] is None and m['strata_detail'] is None


def test_strata_detail_is_a_provenance_field_the_caller_cannot_set(tmp_path):
    with pytest.raises(ValueError, match='strata_detail'):
        runlog.write_manifest(tmp_path, {'strata_detail': {'running': False}})
    with pytest.raises(ValueError, match='strata_running'):
        runlog.write_manifest(tmp_path, {'strata_running': False})


def test_the_real_process_scan_works_on_this_machine(monkeypatch):
    """The fakes cannot show that the real psutil call is right: this test lists the real processes (it does not assert on Strata)."""
    monkeypatch.undo()                  # the autouse fixture gave every test an empty process list
    rec = runlog.strata_check()
    assert rec is not None and rec['scanned'] > 10 and isinstance(rec['running'], bool) and rec['running'] == bool(rec['matched'])
    assert all(isinstance(m['pid'], int) and isinstance(m['name'], str) for m in rec['matched'])
    assert 0 <= rec['cmdline_unreadable'] <= rec['scanned'] and rec['bridges_ignored'] >= 0
