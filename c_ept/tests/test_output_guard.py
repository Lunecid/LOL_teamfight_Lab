"""The conftest guard that keeps match keys out of the test output (Rulings 14 and 33).

pytest prints the arguments of the failing test in its default traceback (``--tb=auto``, ``--tb=long``) and the locals with
``--showlocals``; both can hold a match key (``raw_fixture`` is one). ``c_ept/tests/conftest.py`` therefore turns these two styles into
``--tb=short`` and refuses ``--showlocals``. The checks below call the hook with a stand-in configuration and, end to end, run a small
failing test in a child pytest that uses a copy of the conftest.
"""
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from c_ept.tests import conftest as guard

REPO = Path(__file__).resolve().parents[2]
SECRET = 'KR_0000000000_SECRET'           # stands for a match key; the child's failing test holds it as a fixture value
CHILD_TEST = f'''import pytest


@pytest.fixture
def match_key():
    return {SECRET!r}


def test_fails(match_key):
    ok = False
    assert ok
'''


def stand_in(tbstyle, showlocals=False):
    """The part of a pytest ``Config`` that the hook reads and writes."""
    option = types.SimpleNamespace(tbstyle=tbstyle, showlocals=showlocals, fulltrace=False)
    return types.SimpleNamespace(option=option, getoption=lambda name: getattr(option, name))


@pytest.mark.parametrize('before, after', [('auto', 'short'), ('long', 'short'), ('short', 'short'), ('line', 'line'),
                                           ('native', 'native'), ('no', 'no')])
def test_the_hook_turns_auto_and_long_tracebacks_into_short_ones(before, after):
    config = stand_in(before)
    guard.pytest_configure(config)
    assert config.option.tbstyle == after


def test_the_hook_refuses_showlocals():
    with pytest.raises(pytest.UsageError, match='showlocals'):
        guard.pytest_configure(stand_in('short', showlocals=True))
    with pytest.raises(pytest.UsageError, match='showlocals'):
        guard.pytest_configure(stand_in('auto', showlocals=True))


def test_this_session_runs_with_safe_options(request):
    assert request.config.option.tbstyle not in ('auto', 'long')
    assert not request.config.option.showlocals


def run_child(tmp_path, *options):
    """A child pytest in an empty folder with a copy of the real conftest and one failing test that holds SECRET as a fixture value."""
    (tmp_path / 'conftest.py').write_bytes(Path(guard.__file__).read_bytes())
    (tmp_path / 'test_child.py').write_text(CHILD_TEST, encoding='utf-8')
    env = {k: v for k, v in os.environ.items() if k != 'PYTEST_ADDOPTS'}
    env['PYTHONPATH'] = str(REPO)                                # the copy of conftest imports c_ept
    done = subprocess.run([sys.executable, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', *options], cwd=tmp_path, env=env,
                          capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=120)
    return done.returncode, done.stdout + done.stderr


@pytest.mark.parametrize('options', [(), ('--tb=auto',), ('--tb=long',)], ids=['default', 'auto', 'long'])
def test_a_failing_test_does_not_print_its_fixture_values(tmp_path, options):
    code, output = run_child(tmp_path, *options)
    printed = SECRET in output
    failed_as_designed = code == 1 and '1 failed' in output
    assert failed_as_designed and not printed


@pytest.mark.parametrize('flag', ['--showlocals', '-l'])
def test_showlocals_is_refused_before_any_test_runs(tmp_path, flag):
    code, output = run_child(tmp_path, flag)
    printed = SECRET in output
    assert code == 4 and 'showlocals' in output and not printed          # pytest's usage-error exit code; no test was run
    assert 'passed' not in output and 'failed' not in output


def test_full_trace_is_refused_before_any_test_runs(tmp_path):
    code, output = run_child(tmp_path, '--full-trace')
    printed = SECRET in output
    assert code == 4 and 'full-trace' in output and not printed          # pytest's usage-error exit code; no test was run
    assert 'passed' not in output and 'failed' not in output
