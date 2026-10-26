"""Command line of the run kernel: ``python -m c_ept.run.cli {train,evaluate,measure} --profile <path> --run-dir <path> [--approval <text>] ...`` (S4b-prep K6c; decisions D10 P6, D11, D20;
K6b review r4).

One command starts (or resumes) one run of the kernel from a hash-locked profile file and nothing else: ``train`` (``runners.start``), ``evaluate`` (``evaluate.start``) and ``measure``
(``measure.start``, the P6 bit-identity check). The command goes through the supervisor of the kernel (``run.supervisor``: one Windows-Job-owned child per stage) and the gates of K4 (the native-thread
policy, E6, the child environment, the decision record, the start gate of the code identity and the budget). The text of the author's approval (``--approval``) is recorded as given in the ledger
header. The options that the profile does not hold are the inputs manifest (``--inputs``), the decision record (``--spec``), the budget (``--wall-hours``, ``--memory-gib``), the runner constants
(``--device``, ``--threads``, ``--batch-size``), for ``train`` the allowance of the wall-cap stage (``--allowance``), and for ``evaluate`` the access ledger folder, the S5 lock file and the DEV
rehearsal (``--access-dir``, ``--lock``, ``--rehearsal-run``).

Exit codes (Q7; the table of the supervisor)
---------------------------------------------
    0   the run finished (a measurement that finds a difference has finished, too: read ``identical`` in ``MEASURE.json`` and the line that is printed)
    2   a refusal: a wrong command line (the usage is printed), a profile of another kind than the command, a GPU run without the text of an approval, a stage body other than the default one for the
        evaluation of the role DEV or TEST, or any refusal of the gates and runners (nothing was reserved or computed that is lost)
    3   a computation that did not finish (a stage failed, the budget ran out during the run)

Symbols
-------
    c        the command, one of ``COMMANDS``;  P the profile at ``--profile`` and k its kind;  R the run folder ``--run-dir``;  d the device, ``cuda`` (default) or ``cpu``
    A        the text of the approval (``--approval``), recorded as given with its hash in ``Header.runner['approval']`` (None: null)
    W, G     ``--wall-hours`` and ``--memory-gib``: the budget ``Budget(3600 W seconds, G GiB)``
    S        the stage body that ``bodies`` names for the evaluation, ``BODY`` of ``run.evaluate`` by default

Equations
---------
    (CLI-1)  accept  <=>  k = c  and  (d = cpu  or  A has a visible character)  and  (c = train and P.on_wall_cap = best_stop  =>  --allowance is given)
                          and  (c = evaluate and role(P) in {DEV, TEST}  =>  S is the default stage body)
    (CLI-2)  scrub keys = the match keys of the S1 store of the inputs manifest (``matches/key``): a run that cannot scrub does not start (as Gate 0)
    (CLI-3)  exit code = 2 for every refusal above and for a wrong command line, else the exit code of the supervisor (0 or 2 or 3)

해석
----
명령줄은 설정을 프로파일 파일 하나로 받고 나머지(입력 목록, 결정 기록, 예산, 실행기 상수, 평가의 열람 원장)만 옵션으로 받는다. 장치 기본값이 GPU(``cuda``)이므로 GPU 실행은 저자의 승인 문구를
함께 받아야 시작한다(문구는 원장 머리글에 글자 그대로 남는다). CPU 실행(시험)은 문구 없이도 된다. DEV와 TEST 평가는 기본 단계 본문(``evaluate.BODY``)만 받는다: 리허설과 본 실행이 같은 코드라는
보증(관문이 단계 본문의 모듈·이름·파일 해시를 견준다)이 명령줄에서는 처음부터 깨지지 않게 한다. 오류 글에서는 입력 목록이 가리키는 S1 저장소의 경기 키를 가린다(키를 읽을 수 없으면 시작하지 않는다).
이 모듈은 표준 라이브러리와 가벼운 커널 모듈만 불러오고, 학습·평가·측정 모듈(numpy와 torch)은 네이티브 스레드 정책을 먼저 적용한 뒤에 불러온다. ``main``의 키워드 ``seams``는 시험이 장난감 적재기,
본문, 감독자의 관문을 넣는 자리이고, 프로세스 진입(``python -m c_ept.run.cli``)은 아무것도 넘기지 않는다.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from c_ept.run import guards
from c_ept.run import io as run_io
from c_ept.run import profile as run_profile
from c_ept.run import supervisor

COMMANDS = ('train', 'evaluate', 'measure')
SEAMS = frozenset({'bodies', 'loader', 'backend', 'allow_custom', 'scrub_keys', 'budget', 'e6', 'capture', 'verify', 'start_gate', 'poll_s', 'margin_s', 'repo', 'environ'})
OPTION_SEAMS = ('e6', 'capture', 'verify', 'start_gate', 'poll_s', 'margin_s', 'repo', 'environ')          # the seams that go to ``supervisor.run`` as keywords
DEFAULT_MEMORY_GIB = 40.0                                                                               # = ``ept.train.MEMORY_CAP_BYTES``, the memory bound of the trainer and of Gate 0
ROLES_WITH_DEFAULT_BODY = ('DEV', 'TEST')                                                               # K6b review r4: the roles whose stage body must be the default one


class _Parser(argparse.ArgumentParser):
    """An ``ArgumentParser`` that writes the usage, the help and the errors to the streams of the command (not to the process's) and ends a wrong command line with ``SystemExit(2)``."""

    def __init__(self, *args, stream, out, **kwargs):
        super().__init__(*args, **kwargs)
        self._stream, self._out = stream, out

    def print_usage(self, file=None):
        super().print_usage(self._stream if file is None or file is sys.stderr else file)

    def print_help(self, file=None):
        super().print_help(self._out if file is None or file is sys.stdout else file)

    def error(self, message):
        self.print_usage(self._stream)
        self._stream.write(f'{self.prog}: error: {message}\n')
        raise SystemExit(2)

    def exit(self, status=0, message=None):
        if message:
            self._stream.write(message)
        raise SystemExit(status)


def _positive(name: str):
    def parse(text: str) -> float:
        try:
            value = float(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f'{name} must be a number, not {text!r}') from None
        if not math.isfinite(value) or value <= 0:
            raise argparse.ArgumentTypeError(f'{name} must be a positive finite number, not {text!r}')
        return value
    return parse


def _allowance(text: str) -> tuple:
    parts = text.split(',')
    try:
        values = tuple(float(part) for part in parts)
    except ValueError:
        values = ()
    if len(values) != 3 or not all(math.isfinite(v) and v >= 0 for v in values):
        raise argparse.ArgumentTypeError('--allowance is three numbers of seconds, zero or more, joined by commas: epoch_s,checkpoint_write_s,finalisation_s')
    return values


def build_parser(stream, out) -> argparse.ArgumentParser:
    parser = _Parser(prog='python -m c_ept.run.cli', stream=stream, out=out, allow_abbrev=False,
                     description='Start or resume one run of the kernel from a hash-locked profile: train, evaluate or measure (the P6 bit-identity check).')
    parser.add_argument('command', choices=COMMANDS, help='the kind of run; the profile must be of the same kind')
    parser.add_argument('--profile', required=True, help='the profile file (its SHA-256 is the lock of the run)')
    parser.add_argument('--run-dir', required=True, help='the run folder (a new folder, or the folder of the run to resume)')
    parser.add_argument('--inputs', required=True, help='the inputs manifest (train and measure: ept-run-inputs-v1; evaluate: ept-run-eval-inputs-v1)')
    parser.add_argument('--approval', default=None, help='the text of the approval of the author, recorded as given in the ledger header (required for a GPU run)')
    parser.add_argument('--spec', default=None, help='the decision record (default: the one of c_ept.config)')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda', help='the device of the run (default: cuda, the T002 device)')
    parser.add_argument('--threads', type=int, default=8, help='the thread count of the children (default 8, the T002 value)')
    parser.add_argument('--batch-size', type=int, default=32, help='engagements per batch (default 32, the T002 value)')
    parser.add_argument('--wall-hours', type=_positive('--wall-hours'), required=True, help='the wall-clock budget of the whole run in hours')
    parser.add_argument('--memory-gib', type=_positive('--memory-gib'), default=DEFAULT_MEMORY_GIB, help=f'the memory budget in GiB (default {DEFAULT_MEMORY_GIB:g})')
    parser.add_argument('--allowance', type=_allowance, default=None,
                        help='train only: epoch_s,checkpoint_write_s,finalisation_s that the wall-cap stage holds beyond the 24 h of the trainer (required when the profile ends at the wall cap with its best STOP)')
    parser.add_argument('--access-dir', default=None, help='evaluate DEV and TEST: the folder of the access ledger')
    parser.add_argument('--lock', default=None, help='evaluate TEST: the S5 lock file (the profile fixes its sha256)')
    parser.add_argument('--rehearsal-run', default=None, help='evaluate TEST: the run folder of the finished DEV rehearsal')
    return parser


def make_budget(wall_hours, memory_gib) -> guards.Budget:
    """``guards.Budget(3600 * wall_hours seconds, memory_gib GiB)``; ``IORefused`` (exit code 2) for a number that is not a positive finite number or a memory below one byte."""
    for name, value in (('wall hours', wall_hours), ('memory GiB', memory_gib)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise run_io.IORefused(f'the {name} of the budget must be a positive finite number')
    memory = int(memory_gib * 2 ** 30)
    if memory < 1:
        raise run_io.IORefused('the memory budget is below one byte')
    return guards.Budget(wall_hours * 3600.0, memory)


def match_keys(manifest_path) -> list:
    """The match keys of the S1 store that the inputs manifest names (``matches/key``, as text; the empty entries are left out): what every text that the run prints or records is scrubbed of.
    ``IORefused`` when the manifest names no S1 store or the keys cannot be read: a run that cannot scrub does not start (the rule of Gate 0).

    해석: 오류 글에서 경기 키를 가리려면 키 목록이 필요하다. 입력 목록이 가리키는 S1 저장소에서 읽고, 읽을 수 없으면 시작하지 않는다."""
    s1 = run_io.read_json(manifest_path).get('s1')
    if not isinstance(s1, str) or not s1:
        raise run_io.IORefused('the inputs manifest names no S1 store, so the match keys that the output must not show cannot be read: nothing was started')
    failed = False
    try:
        import numpy as np
        from c_ept.data import store_core
        found = np.asarray(store_core.Store(Path(s1)).array('matches/key')).tolist()
    except Exception:                                                                       # noqa: BLE001 - any failure to read the keys is the same refusal
        failed = True
    if failed:
        raise run_io.IORefused('the match keys of the S1 store cannot be read, so the output of the run cannot be scrubbed of them: nothing was started')
    return [key.decode('ascii') if isinstance(key, bytes) else str(key) for key in found if key]


def _check_approval(args) -> None:
    if args.device == 'cuda' and not (isinstance(args.approval, str) and args.approval.strip()):
        raise run_io.IORefused('a GPU run (--device cuda, the default) needs the text of the approval of the author (--approval "<text>"), which is recorded as given; '
                               'nothing was started. A CPU run (--device cpu) does not need one.')


def _check_kind(command: str, profile) -> None:
    if profile.kind != command:
        raise run_io.IORefused(f'the {command} command takes a {command} profile, not a {profile.kind} profile')


APPLIES_TO = {'allowance': ('train',), 'access_dir': ('evaluate',), 'lock': ('evaluate',), 'rehearsal_run': ('evaluate',)}      # the options that only one command reads


def _check_options(args) -> None:
    """``IORefused`` for an option that the command does not read (it would be ignored without a word, and a run that looks as if it had the option is a wrong record)."""
    for name, commands in APPLIES_TO.items():
        if getattr(args, name) is not None and args.command not in commands:
            raise run_io.IORefused(f"the option --{name.replace('_', '-')} is read by the {' and '.join(commands)} command only, not by {args.command}; nothing was started")


def _check_body(profile, bodies) -> None:
    """K6b review r4: for the roles DEV and TEST the stage body must be the default one (``evaluate.BODY``). A body that ``bodies`` names for the stage ``evaluate`` and that is another text (or
    is not a text that a child can import) is refused; a body for another stage is none of the evaluation's."""
    from c_ept.run import evaluate as run_evaluate
    role = profile['role']
    chosen = (bodies or {}).get(run_evaluate.STAGE)
    if role not in ROLES_WITH_DEFAULT_BODY or chosen is None:
        return
    try:
        text = supervisor.Stage(run_evaluate.STAGE, chosen).body
    except run_io.IORefused:
        text = None
    if text != run_evaluate.BODY:
        raise run_io.IORefused(f'the role {role} takes the default stage body {run_evaluate.BODY} only: a run with another stage body is not the run that its rehearsal was (decision D20, K6b '
                               'review r4), and the command line does not take one; nothing was started')


def _allowance_of(args, profile, runners):
    if args.allowance is not None:
        return runners.CapAllowance(*args.allowance)
    if profile['on_wall_cap'] == 'best_stop':
        raise run_io.IORefused("the profile ends at the wall cap with its best STOP checkpoint (on_wall_cap 'best_stop'): --allowance epoch_s,checkpoint_write_s,finalisation_s is required "
                               '(the bounds that the stage holds beyond the 24 h of the trainer; they are declared, never defaulted)')
    return runners.CapAllowance(0.0, 0.0, 0.0)                                              # under 'fail' the trainer never goes past its cap: no allowance is added to the stage's cap


def _options(seams: dict) -> dict:
    return {name: seams[name] for name in OPTION_SEAMS if name in seams}


def _spec_path(args):
    if args.spec is not None:
        return args.spec
    from c_ept import config
    return str(config.SPEC_PATH)


def _start(args, profile, seams: dict, keys: list, stream, out) -> int:
    """The run of the command (after the checks of ``main``): the runner of the kernel for ``args.command`` with the arguments of the command line and the seams."""
    from c_ept.run import runners
    constants = runners.RunnerConstants(batch_size=args.batch_size, device=args.device, threads=args.threads)
    budget = seams['budget'] if 'budget' in seams else make_budget(args.wall_hours, args.memory_gib)
    common = dict(manifest_path=args.inputs, spec_path=_spec_path(args), budget=budget, constants=constants, scrub_keys=keys, bodies=seams.get('bodies'), stream=stream,
                  approval=args.approval, **_options(seams))
    if args.command == 'train':
        return runners.start(args.profile, args.run_dir, allowance=_allowance_of(args, profile, runners), loader=seams.get('loader', runners.LOADER), **common)
    if args.command == 'measure':
        from c_ept.run import measure as run_measure
        return run_measure.start(args.profile, args.run_dir, loader=seams.get('loader', run_measure.LOADER), out=out, **common)
    from c_ept.run import evaluate as run_evaluate
    return run_evaluate.start(args.profile, args.run_dir, access_dir=args.access_dir, lock_path=args.lock, rehearsal_run=args.rehearsal_run, allow_custom=seams.get('allow_custom', False),
                              loader=seams.get('loader', run_evaluate.LOADER), backend=seams.get('backend', run_evaluate.BACKEND), **common)


def main(argv=None, *, stream=None, out=None, **seams) -> int:
    """Run the command line ``argv`` (default: ``sys.argv[1:]``) and return the exit code (0, 2 or 3). Errors and the usage go to ``stream`` (default standard error), the help, the line of the
    measurement and the closing line of the command go to ``out`` (default standard output).

    ``seams``: the keywords that a test gives for the machine it runs on and that the process entry never passes: ``bodies`` (replacement stage bodies), ``loader``, ``backend``,
    ``allow_custom`` (the test-only switch of the evaluation), ``scrub_keys`` (instead of reading the S1 store), ``budget`` (a ``guards.Budget`` instead of the one made from the options) and the
    options of ``supervisor.run`` (``e6``, ``capture``, ``verify``, ``start_gate``, ``poll_s``, ``margin_s``, ``repo``, ``environ``). ``TypeError`` for another keyword.

    The native-thread policy (``guards.native_policy``) is applied first; the modules that import numpy and torch are imported after it. Everything that can be refused before the supervisor
    starts (the command line, the kind of the profile, the approval of a GPU run, the stage body of a DEV or TEST evaluation, the allowance, the match keys, the runner constants and the budget)
    is a printed refusal with exit code 2 and reserves nothing."""
    unknown = sorted(set(seams) - SEAMS)
    if unknown:
        raise TypeError(f'main() got unknown seams {unknown}')
    stream, out = (sys.stderr if stream is None else stream), (sys.stdout if out is None else out)
    prepared: dict = {}

    def prepare() -> None:
        guards.native_policy()                                                              # before anything can import numpy (the runners do, and so does the match-key reader)
        args = build_parser(stream, out).parse_args(argv)
        _check_approval(args)
        _check_options(args)
        profile = run_profile.load(args.profile)
        _check_kind(args.command, profile)
        if args.command == 'evaluate':
            _check_body(profile, seams.get('bodies'))
        prepared.update(args=args, profile=profile, keys=list(seams['scrub_keys']) if 'scrub_keys' in seams else match_keys(args.inputs))
    try:
        status = supervisor.main_guard(prepare, stream=stream)
    except SystemExit as stop:                                                              # a wrong command line, or --help: the parser has printed the text
        return stop.code if isinstance(stop.code, int) else supervisor.EXIT_REFUSED         # a code that is not a number (a message) is no success
    if status != supervisor.EXIT_OK:
        print(f'cli: exit code {status} (0 finished, 2 refused, 3 failed)', file=out, flush=True)
        return status
    args, keys = prepared['args'], prepared['keys']
    code = supervisor.main_guard(lambda: _start(args, prepared['profile'], seams, keys, stream, out), scrub_keys=keys, stream=stream)
    print(f'cli {args.command}: exit code {code} (0 finished, 2 refused, 3 failed)', file=out, flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
