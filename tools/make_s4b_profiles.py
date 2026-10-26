"""Writes the S4b run-kernel profiles into ``profiles/s4b/`` (S4b-prep K9; decisions D8, D10, D13, D17, D30; plan Task K9).

Deterministic: the same arguments give the same bytes. Every file goes through ``c_ept.run.io.write_json_atomic`` (strict JSON, sorted keys, two-space
indent, LF line ends, written all or nothing) and is read back with ``c_ept.run.profile.load``, so a profile that the kernel would refuse is never left
behind. Run it from the repository root: ``python -m tools.make_s4b_profiles --help``.

What it writes (the table of plan Task K9; N is the number of the training)
---------------------------------------------------------------------------
    train_NN_<tag>.json      the thirteen S4b trainings. Every field of a train profile is written, defaults included, so the file does not depend on
                             a default of ``profile.py`` that may change later.
    evaluate_NN_<tag>.json   the role STOP evaluation (decision D13: the whole STOP split, each fit's best checkpoint) of training N, one model ``tNN``.
    evaluate_c4_seeds.json   the three seeds of the chosen configuration against the matched B1 (C4), only when ``chosen_training`` is given.
    measure_p6.json          the P6 bit-identity measurement, written only when the steps are given. It names ``train_01_lambda0.json`` by relative path and sha256 (never a copy).
    PENDING.json             what the S4b run plan (X1, author approval) still has to decide, with the reason and the decision that says so.

Symbols
-------
    lam_k, lr_k, s_k, f_k   the smoothness weight lambda_Omega (profile field ``terrain_smooth_weight``), the learning rate, the training seed and the data share
                            of training k
    q                       ``SELECTION_SEED`` = 20261022, the fixed 25 % membership (``selection_seed``; decision D13: C4 compares seeds 1-3 with the B1 that
                            trained on the same FIT membership)
    s_1, s_2, s_3           20261022, 20261023, 20261024 (``inference.SEEDS``: the seed values that C4 pairs)
    lam*, lr*               the best lambda_Omega and the best learning rate on STOP (decision D13), known only after trainings 1-5 have run

Equations
---------
    (K9-1)  all trainings: kill_law = shifted, k_condition = binned, decay_groups = split, column_drop = store_and_subset, stats_population = selected_fit,
            on_wall_cap = best_stop, patience 2, weight decay 0.01, warm-up 10
    (K9-2)  25 % trainings (1-11): fraction 0.25, epochs 12 (D10 P2), selection_seed = q; training 13: fraction 1, epochs 8 (D10 P2: "start 8")
    (K9-3)  1-3: lam in {0, lam_a, lam_b}, lr 1e-3, seed s_1 | 4, 5: lam*, lr in {5e-4, 2e-3}, s_1 | 6: A6 field_channels 0 | 7: A7 zone_mark false
            8, 9: lam*, lr*, seed s_2, s_3 | 10: swap_mode average | 11: swap_mode none and eval_average false | 12: fraction 0.5 | 13: fraction 1
            swap_mode is random_view in every training except 10 and 11, which carry their own mode.
    (K9-4)  a value that lam_a, lam_b, lam*, lr*, the 50 % membership rule, the 50 % epochs, chunk_views or the measurement steps would give is never guessed and
            never replaced by a stand-in: while it is not an argument, every file that depends on it is not written and PENDING.json names the dependency.

해석
----
S4b의 학습 13개는 계획 Task K9의 표를 그대로 옮긴 것이다. 공통 설정은 이동 킬 법칙, 구간화한 K 조건, 분리한 감쇠 묶음, 저장소와 부분집합의 열 제거, 선택한 FIT로 낸
통계, 최선 STOP 종료(24 h 상한에서)이다. 값이 아직 정해지지 않은 것(λ_Ω 잠정값 둘, STOP 최선 λ와 학습률, 50 % 소속 규칙과 에폭, chunk_views, C4가 쓸 설정, 측정 걸음 수)은 이 파일이 정하지 않는다. 그런 값은 S4b 실행 계획(X1)에서
저자 승인으로 정해지며, 인자로 받아 같은 코드로 최종 파일을 쓴다. 값이 없으면 그 값에 달린 파일은 대리값 없이 쓰지 않고 PENDING.json에 의존하는 값과 근거 결정을 남긴다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from c_ept.run import io as run_io
from c_ept.run import profile as run_profile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / 'profiles' / 's4b'
SEEDS = {1: 20261022, 2: 20261023, 3: 20261024}              # inference.SEEDS: the seed values that C4 pairs (D13)
SELECTION_SEED = 20261022                                    # the fixed 25 % membership (plan K9)
EPOCHS_25 = 12                                               # D10 P2
EPOCHS_100 = 8                                               # D10 P2: "the 100% budget is set from the S4b 100% fit (start 8)"
LR_DEFAULT = 0.001
LR_LOW, LR_HIGH = 0.0005, 0.002                              # D10 P3: the three-point learning-rate check (5e-4, 1e-3, 2e-3)
MEASURE_TRAIN = 1                                            # the measure profile names this training (plan K9: training 1)
CHUNK_VIEWS_HAND_OFF = 256                                   # M14 hand-off value, refused by the runner until K6e: only recorded as pending

COMMON = {'kill_law': 'shifted', 'k_condition': 'binned', 'decay_groups': 'split', 'column_drop': 'store_and_subset',
          'stats_population': 'selected_fit', 'on_wall_cap': 'best_stop', 'patience': 2, 'weight_decay': 0.01, 'warmup': 10, 'size': 'S',
          'eval_average': True, 'zone_mark': True, 'field_channels': 8}
TAGS = {1: 'lambda0', 2: 'lambda_a', 3: 'lambda_b', 4: 'lr_low', 5: 'lr_high', 6: 'a6_no_field', 7: 'a7_no_zone_mark', 8: 'seed2', 9: 'seed3',
        10: 'two_frame', 11: 'no_symmetry', 12: 'share50', 13: 'share100'}


def train_file(number: int) -> str:
    return f'train_{number:02d}_{TAGS[number]}.json'


def evaluate_file(number: int) -> str:
    return f'evaluate_{number:02d}_{TAGS[number]}.json'


def model_name(number: int) -> str:
    return f't{number:02d}'


def _train(number: int, *, lam: float, lr: float, seed: int, fraction: float, epochs: int, selection, swap_mode: str = 'random_view',
           eval_average: bool = True, zone_mark: bool = True, field_channels: int = 8) -> dict:
    """Every field of a train profile (K9-1, K9-2); ``selection`` is ``SELECTION_SEED`` or None."""
    out = dict(COMMON, kind='train', fraction=fraction, epochs=epochs, seed=seed, lr=lr, terrain_smooth_weight=lam, swap_mode=swap_mode,
               selection_seed=selection)
    out.update(eval_average=eval_average, zone_mark=zone_mark, field_channels=field_channels)
    return out


def trainings(*, lambdas=None, best_lambda=None, best_lr=None, membership_50=None, epochs_50=None) -> dict:
    """``{number: profile dict}`` of the trainings whose every value is decided by the arguments (K9-3, K9-4). Nothing is replaced by a stand-in.

    ``lambdas`` is ``(lam_a, lam_b)`` for trainings 2 and 3, ``best_lambda`` and ``best_lr`` are lam* and lr*, ``membership_50`` is the selection seed of
    the 50 % training (an integer, or ``'training_seed'`` for the T002 rule) and ``epochs_50`` its epoch budget. Training 1 needs nothing. 2 and 3 need
    ``lambdas``; 4 and 5 need lam*; 6-11 and 13 need lam* and lr*; 12 needs lam*, lr*, ``membership_50`` and ``epochs_50``."""
    s1 = SEEDS[1]
    base = dict(fraction=0.25, epochs=EPOCHS_25, selection=SELECTION_SEED)
    out = {1: _train(1, lam=0.0, lr=LR_DEFAULT, seed=s1, **base)}
    if lambdas is not None:
        out[2] = _train(2, lam=lambdas[0], lr=LR_DEFAULT, seed=s1, **base)
        out[3] = _train(3, lam=lambdas[1], lr=LR_DEFAULT, seed=s1, **base)
    if best_lambda is not None:
        out[4] = _train(4, lam=best_lambda, lr=LR_LOW, seed=s1, **base)
        out[5] = _train(5, lam=best_lambda, lr=LR_HIGH, seed=s1, **base)
    if best_lambda is not None and best_lr is not None:
        chosen = dict(lam=best_lambda, lr=best_lr, **base)
        out[6] = _train(6, seed=s1, field_channels=0, **chosen)
        out[7] = _train(7, seed=s1, zone_mark=False, **chosen)
        out[8] = _train(8, seed=SEEDS[2], **chosen)
        out[9] = _train(9, seed=SEEDS[3], **chosen)
        out[10] = _train(10, seed=s1, swap_mode='average', **chosen)
        out[11] = _train(11, seed=s1, swap_mode='none', eval_average=False, **chosen)
        if membership_50 is not None and epochs_50 is not None:
            selection = None if membership_50 == 'training_seed' else membership_50
            out[12] = _train(12, lam=best_lambda, lr=best_lr, seed=s1, fraction=0.5, epochs=epochs_50, selection=selection)
        out[13] = _train(13, lam=best_lambda, lr=best_lr, seed=s1, fraction=1.0, epochs=EPOCHS_100, selection=None)
    return dict(sorted(out.items()))

def _evaluate(models: list, checkpoints: dict, chunk_views) -> dict:
    return {'kind': 'evaluate', 'role': 'STOP', 'models': models, 'exposure': 'observed', 'winner_truth': 'gold', 'chunk_views': chunk_views,
            'b0': False, 'strata': False, 'price_sensitivity': False, 'b1_swap_average': False, 'bootstrap_B': 2000,
            'checkpoints': {name: checkpoints[name] for name in models if name in checkpoints}, 'holdout_lock_sha256': None}


def pending(*, lambdas, best_lambda, best_lr, membership_50, epochs_50, chunk_views, chosen_training, checkpoints, measure_steps) -> list:
    """The list of ``PENDING.json``: what X1 still decides, with the dependency of each file. An entry is dropped when its value is an argument."""
    items = []

    def add(item_id, affects, reason, decision, argument):
        items.append({'id': item_id, 'status': 'pending', 'affects': affects, 'reason': reason, 'decision': decision, 'argument': argument})
    if lambdas is None:
        add('lambda_omega_values', [train_file(2), train_file(3)],
            'lambda_Omega is a three-point grid: 0 and two provisional values that the S4b run plan sets with reasons; only 0 can be written now (training 1)',
            'D8 (author, B9): "the two provisional values are set with reasons in the S4b plan"', '--lambdas A B')
    if best_lambda is None:
        add('best_lambda', [train_file(n) for n in (4, 5, 6, 7, 8, 9, 10, 11, 12, 13)],
            'depends on the lambda_Omega that is best on STOP (D13), known only after trainings 1-3 have run: trainings 4-13 are not written',
            'D8 and D13 (lambda_Omega selected on the whole STOP split)', '--best-lambda L')
    if best_lr is None:
        add('best_lr', [train_file(n) for n in (6, 7, 8, 9, 10, 11, 12, 13)],
            'depends on the best learning rate of the three-point check (5e-4, 1e-3, 2e-3), known only after trainings 1, 4 and 5 have run: trainings 6-13 are not written',
            'D10 P3 and D13', '--best-lr R')
    if membership_50 is None or epochs_50 is None:
        add('share50_membership_and_epochs', [train_file(12)],
            'the membership rule of the 50 % training (a fixed selection or the training seed) and its epoch budget are left to the S4b run plan; D10 P2 gives a budget only '
            'for 25 % (12) and 100 % (start 8); training 12 also depends on best_lambda and best_lr',
            'plan Task K9: "the membership rule of the 50 % training is set in the S4b run plan"; D10 P2, P4', '--membership-50 N|training_seed --epochs-50 E')
    if chunk_views is None:
        add('chunk_views', ['evaluate_*.json'],
            f'the evaluate profiles keep chunk_views null (the default): the runner refuses it until K6e wires the composed pass; the M14 hand-off value is {CHUNK_VIEWS_HAND_OFF}',
            'M14 record (plan Task M14), UNWIRED_FIELDS in c_ept/run/evaluate.py', '--chunk-views N')
    if not checkpoints:
        add('checkpoints', ['evaluate_*.json'],
            'the evaluate profiles name their models but not the training run folders: they exist only after the trainings have run (the runner needs one folder '
            'for each model)', 'K6b (c_ept/run/evaluate.py: keys(checkpoints) = models)', '--checkpoints FILE.json')
    if chosen_training is None:
        add('c4_seed_group', ['evaluate_c4_seeds.json'],
            'C4 pairs seeds 1-3 of the chosen configuration; which of trainings 1-5 is the chosen one is known only after STOP selection',
            'D13 (C4)', '--chosen-training N')
    if measure_steps is None:
        add('measure_steps', ['measure_p6.json'],
            'at least warmup_steps (10) plus enough steps to cover the batch-shape variety; set at X1 (the size is approved together with the GPU measurement); '
            'measure_p6.json is not written until it is given',
            'D10 P6 and plan Task K6c; c_ept/run/measure.py', '--measure-steps N')
    return sorted(items, key=lambda item: item['id'])

def _clean(out_dir: Path, keep: set) -> None:
    """Removes the files of an earlier run of this generator that this run does not write (a stale file must not look current)."""
    for path in out_dir.glob('*.json'):
        name = path.name
        generated = name.startswith(('train_', 'evaluate_', 'measure_')) or name == 'PENDING.json'
        if generated and name not in keep:
            path.unlink()


def generate(out_dir=DEFAULT_OUT, *, lambdas=None, best_lambda=None, best_lr=None, membership_50=None, epochs_50=None, chunk_views=None,
             chosen_training=None, checkpoints=None, measure_steps=None) -> dict:
    """Writes the profiles; returns ``{file name: sha256}`` of every file written (``PENDING.json`` included)."""
    out_dir = Path(out_dir)
    checkpoints = dict(checkpoints or {})
    plan = trainings(lambdas=lambdas, best_lambda=best_lambda, best_lr=best_lr, membership_50=membership_50, epochs_50=epochs_50)
    files = {train_file(n): profile for n, profile in plan.items()}
    for n in plan:
        files[evaluate_file(n)] = _evaluate([model_name(n)], checkpoints, chunk_views)
    if chosen_training is not None:
        if chosen_training not in (1, 2, 3, 4, 5) or chosen_training not in plan or 8 not in plan or 9 not in plan:
            raise ValueError('chosen_training must be one of the written trainings 1-5, and trainings 8 and 9 must be written (best_lambda and best_lr)')
        files['evaluate_c4_seeds.json'] = _evaluate([model_name(chosen_training), model_name(8), model_name(9)], checkpoints, chunk_views)
    out_dir.mkdir(parents=True, exist_ok=True)
    _clean(out_dir, set(files) | {'PENDING.json'} | ({'measure_p6.json'} if measure_steps is not None else set()))
    written = {name: run_io.write_json_atomic(out_dir / name, body) for name, body in files.items()}
    if measure_steps is not None:
        written['measure_p6.json'] = run_io.write_json_atomic(out_dir / 'measure_p6.json', {
            'kind': 'measure', 'train_profile': train_file(MEASURE_TRAIN), 'train_sha256': written[train_file(MEASURE_TRAIN)], 'steps': measure_steps, 'repeat': 2})
    items = pending(lambdas=lambdas, best_lambda=best_lambda, best_lr=best_lr, membership_50=membership_50, epochs_50=epochs_50, chunk_views=chunk_views,
                    chosen_training=chosen_training, checkpoints=checkpoints, measure_steps=measure_steps)
    written['PENDING.json'] = run_io.write_json_atomic(out_dir / 'PENDING.json', {
        'note': 'Values that the S4b run plan (X1, author approval) still has to decide. Generated by tools/make_s4b_profiles.py; do not edit by hand.',
        'items': items})
    for name in sorted(written):
        if name != 'PENDING.json':
            loaded = run_profile.load(out_dir / name)
            if loaded.sha256 != written[name]:
                raise RuntimeError(f'{name} was read back with another hash than the one written')
    return dict(sorted(written.items()))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Write the S4b profiles (profiles/s4b). Values the S4b run plan has not decided stay pending.')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--lambdas', type=float, nargs=2, metavar=('A', 'B'), help='lambda_Omega of trainings 2 and 3 (D8: set in the S4b plan)')
    parser.add_argument('--best-lambda', type=float, help='the lambda_Omega that is best on STOP (trainings 4-13)')
    parser.add_argument('--best-lr', type=float, help='the best learning rate (trainings 6-13)')
    parser.add_argument('--membership-50', help='selection seed of the 50 %% training: an integer, or training_seed for the T002 rule')
    parser.add_argument('--epochs-50', type=int, help='epoch budget of the 50 %% training')
    parser.add_argument('--chunk-views', type=int, help='chunk_views of the evaluate profiles (refused by the runner until K6e)')
    parser.add_argument('--chosen-training', type=int, help='which of trainings 1-5 is the chosen configuration (seed 1 of C4)')
    parser.add_argument('--checkpoints', type=Path, help='JSON file: model name (t01 ...) to training run folder')
    parser.add_argument('--measure-steps', type=int, help='steps of the P6 measurement (P6 measurement; measure_p6.json is written only when given)')
    args = parser.parse_args(argv)
    membership = args.membership_50
    if membership is not None and membership != 'training_seed':
        membership = int(membership)
    checkpoints = run_io.read_json(args.checkpoints) if args.checkpoints else None
    written = generate(args.out, lambdas=tuple(args.lambdas) if args.lambdas else None, best_lambda=args.best_lambda, best_lr=args.best_lr,
                       membership_50=membership, epochs_50=args.epochs_50, chunk_views=args.chunk_views, chosen_training=args.chosen_training,
                       checkpoints=checkpoints, measure_steps=args.measure_steps)
    for name, digest in written.items():
        print(f'{digest}  {name}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
