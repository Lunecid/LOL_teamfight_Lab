r"""S2 runner: the views of an S1 run -> one S2 store under a new run folder ``<runs root>\<YYYYMMDD>_c_ept_s2_<n>``.

    python -m c_ept.run_s2 --s1 <S1 run folder> --build-terrain --macros FIT --limit 1000 --workers 8
    python -m c_ept.run_s2 --s1 <S1 run folder> --terrain <terrain folder> --macros FIT STOP DEV CAL AUDIT --workers 8

``--macros`` names the S1 macro roles to build, taken from the S1 store (``matches/role``): TEST and EXTERNAL are refused, as in
``c_ept.run_s1``; the matches are the S1 matches of those roles in S1 order, the first ``--limit`` of them when it is given. Exactly one
terrain option: ``--terrain PATH`` an existing terrain folder (read only; it needs TERRAIN.json, its files must match it, and it must
have been built with the parameters of this code -- the walkable threshold ``min_count``, the grid and the zone rules --, so the terrain of
the first smoke run is refused before a run folder is made) or ``--build-terrain``, which builds the terrain of the FIT matches of the S1
run into ``<new run folder>\terrain`` first (``terrain.build_terrain``). The store is ``<new run folder>\s2`` (``store_build.build_s2``);
the run is registered in the run index (``--catalog-db``) only when the build is complete. After a power cut, ``--resume <run folder>
--terrain <terrain folder>`` with the same ``--s1``, ``--macros``, ``--limit`` and ``--block-matches`` continues in that run folder
(``build_s2`` refuses any other plan, and a run folder with a store and no RUN.json). Only one build works on a run folder: while
``<run folder>\S2_BUILD.lock`` exists a second build or resume is refused, and a lock that a power cut left behind has to be deleted by
hand (the message says so; nothing deletes it).

The E6 check comes first (no Strata / llama / ollama server may run: the message says what was found, nothing is stopped), before a run
folder is made, and again after every finished block of the build: a model server that starts meanwhile stops the build cleanly after a
marked block, and the message gives the command that resumes the run once the server is stopped (exit code 2, nothing is registered).
Nothing is printed that names a match. This module imports ``c_ept.run_s1`` only inside ``main`` (``store_build`` says why).


방법에서의 위치. 이 모듈은 S1의 선택 경기와 지형을 S2 보기 저장소 빌더에 연결하는 명령행 진입점이다.
보기는 프레임 행 또는 교전 사건 행의 관측 입력이고, 실제 특징·통계 식은 토큰·지형·빌더 모듈이 소유한다.
입력은 S1 실행 경로·역할·지형 선택·재개 인자이고, 출력은 S2 실행 경로와 완료 기록이다.
식 목록. 여기서는 선택·검사·호출만 수행하며 별도 수학식을 정의하지 않는다.
공통 가정. 역할로 고른 경기 행은 S1의 0부터 시작하는 순서를 유지하며 limit는 그 앞부분에 적용한다.
지형을 새로 만들거나 검증된 기존 지형을 쓰는 두 경로를 허용한다(Task 13; S1–S2 판정 R60).
재개 때는 기존 지형과 실행 기록을 요구하며, 목록·블록·출처의 실제 재개 검사는 build_s2가 맡는다.
E6 시작 거부와 블록 사이 거부는 종료·재개 안내를 제공하며 여기서 서버를 조작하지 않는다(지침 E6).
"""
# Ref: docs/superpowers/plans/2026-10-20-c-ept-s1-s2.md (S2, tasks 13 and 14: the command line of the S2 build and its smoke run) and
#      c_ept/run_s1.py (the S1 command line this one follows: macros, E6 before new_run, registration at the end).
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from c_ept import config as C
from c_ept import runlog, store_build, terrain


def _refuse(message: str) -> int:
    print(message, file=sys.stderr, flush=True)
    return 2


def _resume_hint(a, run: Path, terrain_path: Path) -> str:
    """How to go on after a model server stopped the build: the command that resumes the run folder (the store is there once a block was
    marked), or the plain advice to run the command again when the stop came before any store was begun."""
    if not (run / store_build.STORE_DIR / 'STORE.json').is_file():
        return f'No S2 store was begun in {run}: stop the model server and run the command again.'
    parts = ['python', '-m', 'c_ept.run_s2', '--s1', str(a.s1), '--resume', str(run), '--terrain', str(terrain_path), '--macros', *a.macros,
             '--block-matches', str(a.block_matches), '--workers', str(a.workers)]
    if a.limit is not None:
        parts += ['--limit', str(a.limit)]
    if Path(a.catalog_db) != Path(C.CATALOG_DB):
        parts += ['--catalog-db', str(a.catalog_db)]
    return f'The run folder {run} can be resumed: after the model server is stopped, run  {" ".join(parts)}'


def main(argv=None) -> int:
    """S1–S2 계획 Task 13과 판정 R60에 따라 지형 선택·재개 조건을 검사하고 빌더에 넘긴다.
    완료 결과가 기록된 실행만 색인에 등록하며, 이 함수의 사전 검사만으로 모든 빌더 검사가 끝난 것은 아니다."""
    from c_ept.run_s1 import StrataRunning, load_s1, require_no_strata       # here and not at module level (store_build says why)
    ap = argparse.ArgumentParser(prog='python -m c_ept.run_s2', description='S2: the views of an S1 run -> one S2 store.')
    ap.add_argument('--s1', type=Path, required=True, help='the S1 run folder (or its s1 store folder); read only')
    ap.add_argument('--macros', nargs='+', required=True, metavar='ROLE',
                    help='S1 macro roles to build, e.g. FIT STOP DEV CAL AUDIT (TEST and EXTERNAL are refused)')
    ap.add_argument('--limit', type=int, default=None, help='build only the first N of the matches of those roles (S1 order)')
    ap.add_argument('--terrain', type=Path, default=None, help='an existing terrain folder with TERRAIN.json (read only)')
    ap.add_argument('--build-terrain', action='store_true', help='build the terrain of the S1 FIT matches into the new run folder')
    ap.add_argument('--resume', type=Path, default=None, metavar='RUN',
                    help='continue an interrupted build in this S2 run folder instead of making a new one (needs --terrain and the same '
                         '--macros, --limit and --block-matches; the plan check of build_s2 refuses any difference)')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--block-matches', type=int, default=250)
    ap.add_argument('--runs-root', type=Path, default=C.RUNS_ROOT)
    ap.add_argument('--catalog-db', type=Path, default=C.CATALOG_DB)
    a = ap.parse_args(argv)
    bad = [m for m in a.macros if m in store_build.REFUSED_ROLES or m not in C.ROLES]
    if bad:
        return _refuse(f'macro role(s) {bad} refused: S2 builds S1 roles of {C.ROLES} except {store_build.REFUSED_ROLES}')
    for flag, value in (('--workers', a.workers), ('--block-matches', a.block_matches), ('--limit', a.limit)):
        if value is not None and value < 1:
            return _refuse(f'{flag} must be at least 1, got {value}')
    if a.workers > store_build.MAX_WORKERS:              # the pool refuses more on Windows, but only after the run folder is made
        return _refuse(f'--workers must be at most {store_build.MAX_WORKERS} (the limit of the process pool on Windows), got {a.workers}')
    if (a.terrain is None) == (not a.build_terrain):
        return _refuse('give either --terrain PATH (an existing terrain folder, read only) or --build-terrain (builds terrain\\ into the '
                       'new run folder), not both and not neither')
    if a.resume is not None and a.build_terrain:
        return _refuse('--resume continues an existing run: give --terrain (the terrain of that run), not --build-terrain')
    try:
        require_no_strata()                                  # E6: before the S1 store is opened and before the run folder is made
    except StrataRunning as e:                               # the guard text is the S1 runner's: say that it holds for S2 as well
        return _refuse(f'{e} (The E6 guard of the S1 runner stops the S2 build too; no run folder was made.)')
    if a.terrain is not None:                                # build_s2 checks the files; this saves the run folder
        refusal = store_build.terrain_refusal(a.terrain)     # no TERRAIN.json, or built with other parameters than this code uses
        if refusal is not None:
            return _refuse(f'{refusal}. No run folder was made.')
    if a.resume is not None:
        if not (a.resume / store_build.STORE_DIR / 'STORE.json').is_file():
            return _refuse(f'there is no S2 store to resume under {a.resume}')
        try:
            store_build.require_run_record(a.resume)         # a store without RUN.json is refused by build_s2 as well
        except ValueError as e:
            return _refuse(str(e))
    try:
        s1 = load_s1(a.s1, verify=False)                     # build_s2 checks every file against STORE.json; this only picks the matches
    except (FileNotFoundError, ValueError) as e:
        return _refuse(f'the S1 run cannot be used: {e}')
    selected = np.flatnonzero(np.isin(np.asarray(s1.array('matches/role')), [C.ROLES.index(m) for m in a.macros]))
    s1.maps.clear()
    if a.limit is not None:
        selected = selected[:a.limit]
    if not selected.size:
        return _refuse(f'no S1 matches with the macro roles {a.macros}')
    matches = [int(i) for i in selected]
    run = a.resume if a.resume is not None else runlog.new_run('s2', root=a.runs_root)
    print(f'S2 run folder: {run}{" (resumed)" if a.resume is not None else ""}; {len(matches)} matches, {a.workers} workers, '
          f'blocks of {a.block_matches}', flush=True)
    t0 = time.perf_counter()

    def progress(done, total):
        print(f'  block {done}/{total} done, {time.perf_counter() - t0:.0f} s', flush=True)
    terrain_path = terrain.build_terrain(a.s1, run) if a.build_terrain else a.terrain
    if a.build_terrain:
        print(f'terrain built: {terrain_path}, {time.perf_counter() - t0:.0f} s', flush=True)
    try:
        store_build.build_s2(a.s1, terrain_path, matches, run, a.workers, a.block_matches, info={'macros': list(a.macros), 'limit': a.limit},
                             progress=progress)
    except store_build.RunLocked as e:                       # another build, or a lock that a cut left: the message says what to delete
        return _refuse(str(e))
    except StrataRunning as e:                               # E6 again between the blocks (or a race at the start)
        return _refuse(f'{e} {_resume_hint(a, run, terrain_path)}')
    manifest = json.loads((run / 'RUN.json').read_text(encoding='utf-8'))
    r = manifest.get('s2_result') or {}
    if not r.get('complete'):
        return _refuse(f'the build of {run} did not finish: the run is not registered')
    runlog.register(run, manifest, a.catalog_db)
    print(f'S2 done: {r.get("views")} views of {r.get("matches")} matches, {r.get("bytes_per_view")} bytes per view, {r.get("wall_s")} s, '
          f'peak {r.get("peak_private_gib")} GiB private (largest worker {r.get("peak_private_worker_gib")} GiB)', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
