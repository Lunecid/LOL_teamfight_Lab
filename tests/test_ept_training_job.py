"""Harmless real Windows containment tests; never invoke the EPT operator."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

import psutil
import pytest

from tools.ept_training_job import WindowsJob

pytestmark = pytest.mark.skipif(os.name != 'nt', reason='approved runtime uses Windows Job Objects')
ROOT = Path(__file__).resolve().parents[1]
HELPER = """import json,subprocess,sys,time,psutil
from pathlib import Path
if sys.stdin.buffer.read(1)!=b'G': sys.exit(125)
p=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(60)'],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
Path(sys.argv[1]).write_text(json.dumps({'pid':p.pid,'create_time':psutil.Process(p.pid).create_time()}),encoding='utf-8')
"""


def wait_ref(path):
    end = time.monotonic()+10
    while time.monotonic()<end:
        if path.exists():
            return json.loads(path.read_text(encoding='utf-8'))
        time.sleep(.03)
    raise AssertionError('harmless process reference not published')


def same_alive(ref):
    try:
        p=psutil.Process(ref['pid'])
        return p.is_running() and abs(p.create_time()-ref['create_time'])<.00001
    except psutil.NoSuchProcess:
        return False


def assert_gone(ref):
    end=time.monotonic()+10
    while time.monotonic()<end:
        if not same_alive(ref):
            return
        time.sleep(.03)
    assert not same_alive(ref), 'owned harmless descendant survived cleanup'


@pytest.mark.parametrize('mode', ['terminate', 'close'])
def test_parent_gone_before_sample_remains_contained(tmp_path, mode):
    ref=tmp_path/'child.json'
    job=WindowsJob()
    p=None
    try:
        p=subprocess.Popen([sys.executable,'-B','-c',HELPER,str(ref)],stdin=subprocess.PIPE,
                           stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                           creationflags=subprocess.CREATE_NO_WINDOW)
        job.assign(p)
        time.sleep(.1)
        assert not ref.exists() and job.active_count()>=1
        assert job.contains(p.pid,psutil.Process(p.pid).create_time())
        assert not job.contains(p.pid,psutil.Process(p.pid).create_time()+10)
        job.release(p)
        child=wait_ref(ref)
        assert p.wait(timeout=10)==0
        assert same_alive(child) and job.contains(**child)
        if mode=='terminate':
            assert job.stop() and job.gone()
        else:
            job.close()
        assert_gone(child)
    finally:
        if p and p.stdin and not p.stdin.closed:
            p.stdin.close()
        if job.handle:
            job.stop()
            job.close()
        if p:
            p.wait(timeout=10)


def test_supervisor_death_closes_noninherited_job(tmp_path):
    ref=tmp_path/'child.json'
    supervisor="""import importlib.util,subprocess,sys,time
s=importlib.util.spec_from_file_location('job',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
j=m.WindowsJob()
p=subprocess.Popen([sys.executable,'-B','-c',sys.argv[2],sys.argv[3]],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
j.assign(p);j.release(p)
time.sleep(60)
"""
    p=subprocess.Popen([sys.executable,'-B','-c',supervisor,str(ROOT/'tools/ept_training_job.py'),HELPER,str(ref)],
                       stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                       creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        child=wait_ref(ref)
        assert same_alive(child)
        p.kill()  # retained Popen launch handle, never an arbitrary PID
        p.wait(timeout=10)
        assert_gone(child)
    finally:
        if p.poll() is None:
            p.kill()
            p.wait(timeout=10)


@pytest.mark.parametrize('token,exit_code', [(b'',125),(b'X',125),(b'G',0)])
def test_exact_powershell_gate_prefix_without_operator(tmp_path,token,exit_code):
    prefix=(ROOT/'run_ept_training.ps1').read_text(encoding='utf-8').split('$canonicalActions =',1)[0]
    script=tmp_path/'gate-only.ps1'
    script.write_text(prefix+'\n[Console]::Out.Write("gate_passed")\nexit 0\n',encoding='utf-8')
    ps=Path(os.environ['SystemRoot'])/'System32/WindowsPowerShell/v1.0/powershell.exe'
    command=[str(ps),'-NoLogo','-NoProfile','-NonInteractive','-File',str(script),'-Action','Start',
             '-AwaitJobBinding','-Config',str(tmp_path/'unused.json')]
    p=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                       creationflags=subprocess.CREATE_NO_WINDOW)
    job=WindowsJob()
    try:
        job.assign(p)
        if token==b'G':
            job.release(p)
            p.stdin=None
            out,err=p.communicate(timeout=10)
        else:
            out,err=p.communicate(input=token,timeout=10)
        assert p.returncode==exit_code,err
        assert (b'gate_passed' in out)==(exit_code==0)
        assert job.gone()
    finally:
        job.stop()
        job.close()
        p.wait(timeout=10)


def test_close_failure_retains_owned_handle_and_raises():
    import ctypes
    from types import SimpleNamespace
    from tools.ept_training_job import JobError
    job=WindowsJob.__new__(WindowsJob)
    job.handle=12345  # fake only: no kernel calls or process ownership
    job.k=SimpleNamespace(CloseHandle=lambda handle: 0)
    ctypes.set_last_error(5)
    with pytest.raises(JobError, match='CloseHandle'):
        job.close()
    assert job.handle==12345
    job.k.CloseHandle=lambda handle: 1
    job.close()
    assert job.handle is None
