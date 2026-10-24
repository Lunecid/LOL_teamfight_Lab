r"""Measurement scripts behind the counts that code and tests of c_ept pin (S2, Task 9).

Run one from the repository folder as ``python -B -m c_ept.checks.<script> <S1 run folder> [--out counts.json]``. A script learns from
the S1 run (its store, or its golden lock) which matches to read, reads them with ``reader.read_match`` and its default root, refuses to
start while a Strata server runs (E6), prints counts or per-cell numbers (never a match key or any other identifier) and, with ``--out``,
writes the same as JSON. It never writes into the S1 run folder. Nothing runs when a module is imported.

    measure_levels        the level clock against the frame level on the 20 golden matches
    measure_levels_wide   the same on a sample of 200 S1 matches, and how close LEVEL_UPs lie to a frame
    explain_mismatch      the cause of every difference in that sample
    sweep_s1              the level clock, the traits and the item vectors over every S1 match


방법에서의 위치. S1–S2 판정 R37의 진단 스크립트 이름공간이며 실제 진단은 각 main 호출 때 수행한다.
식 목록. 패키지 표식에는 수학식이 없으며, 진단의 수치는 그 계산 함수에 둔다.
"""
