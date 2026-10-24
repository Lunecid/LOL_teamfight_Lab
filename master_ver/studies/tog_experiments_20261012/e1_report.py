"""T1.5 part B documents: the Korean E1 report and the E1 rows of the ToG ledger, rendered from the records of the single
E1 TEST step (PLAN_E1_E5_KO.md Task 16; SDD Rulings 56 and 60).

Reads records only, never a sealed file: tog_e1_1/E1_REPORT.json (the TEST step's record; it holds the TEST summaries
opened in that step), E1_QCAL.json, E1_LOCK.json, E1_TRAIN.json, E1_SELECTION.json, the public BALANCE.json and
WINDOWS_h90.json, the TEST step's two logs, and reports/tog_experiments_20261012/OPENINGS.json; E1_OPENING.json and the
TEST score files are only hashed.  Writes:
  reports/tog_experiments_20261012/E1_REPORT_KO.md, once.  The only way to replace it is --replace-report, under the last
    record of REVISIONS: the record's previous_sha256 must equal the file on disk, and every record is printed in the
    report's header (reason, previous hash, commit).
  the e1_ rows of reports/tog_paper_20261011/ledger/TOG_NUMBERS.csv: all at once on the first run (refused when an e1_ key
    already exists); with --replace-report only the missing keys are appended, and an existing e1_ row that differs from
    its generated row, or an e1_ row the code does not make, refuses the run (rows are never edited).
Ledger rows follow the thesis schema key,printed,kind,source,locator,scale,dataset,note with kind json, a '/'-joined
locator into an absolute source path, printed values with U+2212 and thousands separators, and the sign convention in the
note of every signed row; no note contains the substring that marks a magnitude-only row.  Windows paths are built from
Path objects and chr(92), and a written text holding a TAB or CR character is refused.  Rendering stops (ValueError) when
the records disagree: a failed check after the opening, a lock cross-check that is not true, an environment or code hash
that differs from the lock, openings before E1 that differ from the TEST record, or logs that do not match the record.

Usage: python -B -m studies.tog_experiments_20261012.e1_report [--replace-report]
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path, PureWindowsPath
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studies.tog_experiments_20261012 import e1_run as e1  # noqa: E402
from studies.tog_experiments_20261012.common import TOG_E1, sha256_file  # noqa: E402

MINUS = "−"
SIGN_NOTE = "G = Brier(PT_flex) − Brier(LGBM), 양수면 LightGBM 우위"
INTERVAL_NOTE = "95% percentile interval, joint match bootstrap 2,000 draws shared by the four arrays, PCG64 seed 20261011"
MAIN_INTERVAL_NOTE = "95% percentile interval, paired match bootstrap of the two main cells, 2,000 draws, PCG64 seed 20261011"
LEDGER_COLUMNS = ("key", "printed", "kind", "source", "locator", "scale", "dataset", "note")
CODE_KEY = "studies/tog_experiments_20261012/e1_run.py"
STDOUT_LOG, STDERR_LOG = "e1_run_test.stdout.log", "e1_run_test.stderr.log"
FEATURE_NAME_WARNING = "UserWarning: X does not have valid feature names, but LGBMClassifier was fitted with feature names"
# q_run_1 TEST_REPORT.json bootstrap.contrasts.P1.point: the thesis's full engagement TEST contrast, sign LightGBM - PT_flex
# (PRESPEC_E1 line 6; checked in prespec-reaudit-r2-fable.md R-1 and task-4-report.md:275)
THESIS_P1_POINT = -0.0138566147497243
QUANTITY_LABEL = {"G_eng": "G_eng_size (모델 (나), 짝 맞은 교전)", "G_plc": "G_plc (모델 (가), 위약 창)",
                  "L1": "L₁ = G_plc − 0.25·G_eng_size", "L2": "L₂ = G_plc − 0.75·G_eng_size",
                  "net": "순이득 = G_eng_size − G_plc", "r": "r = G_plc / G_eng_size (기술만)"}
KILL_GAP_BINS = ("< 30초", "30–60초", "60–90초", "90–180초", "180초 이상이거나 앞선 킬 없음")
# key stems whose rows carry the sign convention (G-type differences, their linear contrasts and ratios)
SIGNED_KEYS = frozenset({"g_eng", "g_plc", "l1", "l2", "net", "r", "r001_g_eng", "r001_g_plc", "r001_r", "objfree_g_eng",
                         "objfree_g_plc", "objfree_r", "objfree_l2", "dedup_r", "rstrat", "g_eng_full", "gplc_ratio",
                         "main_matched", "main_unmatched", "main_all"})
CARRY_FORWARD = (  # SDD ledger progress.md, line "Carry-forward to Task 15/16" (Q_DEV, design stage, no V, label or W)
    "짝 맞은 교전의 D 중앙값 29.0초, 짝 없는 교전 52.7초.",
    "D 4분위별 짝 비율: 가장 짧은 4분위 53.7%, 가장 긴 4분위 11.8%.",
    "직전 킬 뒤 30초 안에 시작: 짝 맞은 교전 71.5%, Q_DEV 교전 전체 49.4%. 90초 안: 93.8%, 87.4%.",
    "창 안 프레임 0개(n_bound 0): 짝 맞은 교전 84.2%, Q_DEV 교전 전체 63.2%.",
    "골드 차 양 끝 칸(0과 4): 짝 맞은 교전 47.8%, Q_DEV 교전 전체 40.5%.",
    "25분 이후(time_bin 2): 짝 맞은 교전 10.9%, Q_DEV 교전 전체 13.7%.",
    "SMD: 직전 킬 뒤 시간 −0.058, 시작 나이 −0.004, 골드 +0.001, 경기 시간 −0.064.",
    "위약 시작의 56%가 τ 앞이다. |s − τ| 중앙값은 178초다.",
    "위약 창의 92.9%가 q 교전이 아닌 성분의 결과 구간과 만난다(60.7%는 완전히 안).",
    "위약 창의 14.1%가 시작이 다른 위약 창과 겹친다.",
    "설계 모의의 남은 불균형: 직전 성분 크기 SMD +0.14(10-12판 칸 0 쌍), 목표물 포함 짝 맞은 교전 48.7% 대 위약 42.8%.",
)


# The recorded replacements of E1_REPORT_KO.md (write_report(replace=True), --replace-report).  A new record names the
# SHA-256 of the version it replaces; the report prints every record in its header.
REVISIONS: tuple[dict[str, str], ...] = (
    {"date": "2026-10-13", "label": "Task 16b 수정 1회차",
     "reason": "Task 16b 검토(Fable) Important 1에 따라 §9에 목표물 없는 짝 민감도의 Q_CAL 값과 TEST 값을 나란히 적었다. 같은 "
               "검토의 경미 2(§4의 '95% 구간 기준')와 경미 3(쓰지 않던 경로, 지수 표기), §9의 Ruling 61 문장, 원장 행 셋"
               "(e1_qcal_objfree_r) 덧붙임을 함께 반영했다. 판정, 기록, 기존 원장 행은 바꾸지 않았다.",
     "previous_sha256": "0ba0ee8ad4073bd2652d30e7cae78dc77331003e62ddb094cc5e426182253ee4", "previous_commit": "51b760b"},
    {"date": "2026-10-13", "label": "최종 검토 수정",
     "reason": "최종 검토(Opus I1, M2, M3)에 따라 §9 한계 3에 \"keep\"에 불리한 쪽의 남은 불균형(직전 성분 크기 SMD +0.14, Q_DEV "
               "설계 모의)과 자료를 보기 전 두 검토의 판단을 더해 두 방향을 적었다. §11에 판정 칸 "
               "placebo_gain_exceeds_engagement_gain이 계산되는 범위와 원장 note의 부호 문장을 읽는 법을 적었다. 판정, 기록, "
               "원장 행은 바꾸지 않았다.",
     "previous_sha256": "e62c298517be55b7785993f1588b6881e5ef5ba34323bd78f226be31767cf278", "previous_commit": "e312b99"},
)


@dataclass(frozen=True)
class Paths:
    tog: Path = TOG_E1
    report_md: Path = ROOT / "reports" / "tog_experiments_20261012" / "E1_REPORT_KO.md"
    ledger: Path = ROOT / "reports" / "tog_paper_20261011" / "ledger" / "TOG_NUMBERS.csv"
    openings: Path = e1.OPENINGS

    def record(self, name: str) -> Path:
        return self.tog / name


RECORDS = {"report": e1.REPORT_NAME, "qcal": e1.QCAL_NAME, "lock": e1.LOCK_NAME, "train": e1.TRAIN_NAME,
           "selection": e1.SELECTION_NAME, "balance": "BALANCE.json", "windows": "WINDOWS_h90.json"}


# ----------------------------------------------------------------------------- formats
def num(value: float, digits: int) -> str:
    """Fixed decimals with U+2212 for a negative value."""

    return f"{float(value):.{digits}f}".replace("-", MINUS)


def count(value: float) -> str:
    return f"{int(value):,}"


def pct(value: float, digits: int = 2) -> str:
    return num(100.0 * float(value), digits)


def ci(interval: Sequence[float], digits: int) -> str:
    return f"[{num(interval[0], digits)}, {num(interval[1], digits)}]"


def sci(value: float, digits: int = 1) -> str:
    """Scientific notation with U+2212 for a negative mantissa or exponent (any exponent length)."""

    mantissa, exponent = f"{float(value):.{digits}e}".split("e")
    return f"{mantissa.replace('-', MINUS)}e{exponent.replace('-', MINUS)}"


def win(*parts: str) -> str:
    """A Windows path from its parts joined by chr(92) (no backslash escape in a literal)."""

    return chr(92).join(parts)


def zulu(iso: str) -> str:
    return datetime.fromisoformat(iso).strftime("%Y-%m-%d %H:%M:%SZ")


def get(record: Any, locator: str) -> Any:
    for part in locator.split("/"):
        record = record[int(part)] if isinstance(record, list) else record[part]
    return record


def load(paths: Paths) -> dict[str, Any]:
    out = {name: json.loads(paths.record(file).read_text(encoding="utf-8")) for name, file in RECORDS.items()}
    out["openings"] = json.loads(Path(paths.openings).read_text(encoding="utf-8"))
    return out


def label_name(path: str) -> str:
    """The last folder and the file name of a recorded Windows path (two sealed files share a name)."""

    return win(*PureWindowsPath(path).parts[-2:])


def step_logs(paths: Paths, outcome: str) -> dict[str, int]:
    """The TEST step's logs: stdout must end with the summary of E1_REPORT.json and its outcome; stderr lines are counted
    as the LightGBM feature-name warning, its 'warnings.warn(' echo, or other."""

    lines = paths.record(STDOUT_LOG).read_text(encoding="utf-8", errors="replace").strip().splitlines()
    done = json.loads(lines[-1]) if lines else {}
    if done.get("outcome") != outcome or PureWindowsPath(done.get("done", "")).name != e1.REPORT_NAME:
        raise ValueError(f"{STDOUT_LOG} does not end with the summary of {e1.REPORT_NAME} ({outcome})")
    errors = [line for line in paths.record(STDERR_LOG).read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    warnings = sum(FEATURE_NAME_WARNING in line for line in errors)
    other = sum(FEATURE_NAME_WARNING not in line and line.strip() != "warnings.warn(" for line in errors)
    return {"feature_name_warnings": warnings, "other_stderr_lines": other}


def consistency(records: Mapping[str, Any]) -> dict[str, Any]:
    """Cross-checks between the records; any disagreement stops the rendering."""

    report, lock = records["report"], records["lock"]
    failed = sorted(name for name, value in report["result"]["checks"].items() if value is not True)
    lock_failed = sorted(name for name, value in lock["cross_checks"].items() if value is not True)
    code = report["run"]["code_sha256"]
    code_differs = sorted(name for name, digest in code.items() if lock["code"].get(name, {}).get("sha256") != digest)
    openings = records["openings"]["openings"]
    at = [k for k, entry in enumerate(openings) if entry.get("experiment") == "E1"]
    before = [(entry["experiment"], entry["utc"], entry["lock_sha256"]) for entry in openings[:at[0]]] if len(at) == 1 else None
    recorded = [(entry["experiment"], entry["utc"], entry["lock_sha256"]) for entry in report["openings_before"]]
    problems = {"checks_after_opening": failed, "lock_cross_checks": lock_failed, "code_differs_from_lock": code_differs,
                "environment_differs_from_lock": report["environment"] != lock["environment"],
                "openings_before_differ": before != recorded,
                "opening_differs": len(at) != 1 or openings[at[0]]["utc"] != report["opening"]["utc"]
                                   or openings[at[0]]["lock_sha256"] != report["lock"]["sha256"]}
    if any(problems.values()):
        raise ValueError(f"the E1 records disagree: {problems}")
    return {"opening_index": at[0], "checks": len(report["result"]["checks"]), "lock_cross_checks": len(lock["cross_checks"]),
            "code_files": len(code)}


# ----------------------------------------------------------------------------- the ledger rows
def dataset_key(dataset: str) -> str:
    return {"test": "test", "q_cal": "qcal", "q_dev": "qdev"}[dataset]


def _spec() -> list[tuple[str, str, str, str, int, float, str, str]]:
    """(key, record, locator, format, digits, scale, dataset, note) of every E1 ledger row."""

    rows: list[tuple[str, str, str, str, int, float, str, str]] = []
    labels = {"G_eng": "G_eng_size, model (나) on the matched engagements", "G_plc": "G_plc, model (가) on the placebo windows",
              "L1": f"L1 = G_plc {MINUS} 0.25 G_eng_size", "L2": f"L2 = G_plc {MINUS} 0.75 G_eng_size",
              "net": f"net gain = G_eng_size {MINUS} G_plc", "r": "r = G_plc / G_eng_size, descriptive only"}
    split_name = {"test": "TEST", "q_cal": "Q_CAL"}

    def interval_rows(stem: str, record: str, base: str, ci_base: str, digits: int, dataset: str, what: str,
                      interval_note: str = INTERVAL_NOTE) -> None:
        for suffix, locator, part in (("", base, "point"), ("_lo", f"{ci_base}/0", "lower 95% bound"),
                                      ("_hi", f"{ci_base}/1", "upper 95% bound")):
            prefix = "" if what.startswith("E1 ") else "E1 (PRESPEC_E1) "
            rows.append((f"e1_{dataset_key(dataset)}_{stem}{suffix}", record, locator, "num", digits, 1.0, dataset,
                         f"{prefix}{what}, {split_name[dataset]} {part}; {SIGN_NOTE}; match-weighted raw Brier; {interval_note}"))

    for dataset, record in (("test", "report"), ("q_cal", "qcal")):
        for name, stem in (("G_eng", "g_eng"), ("G_plc", "g_plc"), ("L1", "l1"), ("L2", "l2"), ("net", "net"), ("r", "r")):
            what = labels[name] + (" (decision quantity, PRESPEC_E1 section 5)" if name in ("G_eng", "L1", "L2", "net") else "")
            if dataset == "q_cal":
                what += "; Q_CAL is development evidence, not used for the decision (PRESPEC_E1 section 6 step 5)"
            interval_rows(stem, record, f"result/primary/point/{name}", f"result/primary/{name}_ci", 3 if name == "r" else 5,
                          dataset, what)
        rows.append((f"e1_{dataset_key(dataset)}_draws_g_eng_le_0", record, "result/primary/draws_G_eng_le_0", "count", 0, 1.0,
                     dataset, "E1 joint bootstrap: draws of the 2,000 with G_eng_size at or below 0, counted and kept "
                              "(PRESPEC_E1 section 4)"))
        rows.append((f"e1_{dataset_key(dataset)}_pairs", record, "result/rows", "count", 0, 1.0, dataset,
                     "E1 matched engagement and placebo window pairs of the role (placebo row i = engagement row i)"))
        rows.append((f"e1_{dataset_key(dataset)}_matches", record, "result/matches", "count", 0, 1.0, dataset,
                     "E1 matches holding the pairs of the role (the four arrays' match count)"))
    rows.append(("e1_test_match_rate", "report", "result/summaries/windows/match_rate", "pct", 2, 100.0, "test",
                 "E1 TEST match rate, percent (scale 100): matched TEST engagements over TEST q engagements (h90 valid); sealed "
                 "window summary opened in the TEST step"))
    shown = {"lightgbm": "LightGBM", "PT_flex": "PT_flex"}
    for arm in e1.ARMS:
        for family in e1.FAMILIES:
            rows.append((f"e1_test_brier_{arm}_{family.lower()}", "report", f"result/cells/{arm}/{family}", "num", 5, 1.0, "test",
                         f"E1 TEST match-weighted raw Brier of the selected {shown[family]} cell of run {arm} on its pairs"))
            rows.append((f"e1_qdev_brier_{arm}_{family.lower()}", "selection", f"selection/{arm}/{family}/brier", "num", 6, 1.0,
                         "q_dev", f"E1 Q_DEV raw match-weighted Brier of the selected {shown[family]} card of run {arm} "
                                  "(qselect.select_within)"))
    sens = "result/sensitivities"
    for stem, name, digits in (("r001_g_eng", "G_eng", 5), ("r001_g_plc", "G_plc", 5), ("r001_r", "r", 3)):
        interval_rows(stem, "report", f"{sens}/r_0.01/point/{name}", f"{sens}/r_0.01/{name}_ci", digits, "test",
                      f"E1 sensitivity r_0.01 (rows with |SVI| and |W| at or above 0.01; matches with rows in all four arrays), {name}")
    rows.append(("e1_test_r001_matches", "report", f"{sens}/r_0.01/matches", "count", 0, 1.0, "test",
                 "E1 sensitivity r_0.01: matches with rows in all four arrays after the 0.01 filters"))
    rows.append(("e1_test_r001_rows_eng", "report", f"{sens}/r_0.01/rows_eng", "count", 0, 1.0, "test",
                 "E1 sensitivity r_0.01: engagement rows inside the kept matches (count after the four-array intersection)"))
    rows.append(("e1_test_r001_rows_plc", "report", f"{sens}/r_0.01/rows_plc", "count", 0, 1.0, "test",
                 "E1 sensitivity r_0.01: placebo rows inside the kept matches (count after the four-array intersection)"))
    for stem, name, digits in (("objfree_g_eng", "G_eng", 5), ("objfree_g_plc", "G_plc", 5), ("objfree_r", "r", 3),
                               ("objfree_l2", "L2", 5)):
        interval_rows(stem, "report", f"{sens}/objective_free/point/{name}", f"{sens}/objective_free/{name}_ci", digits, "test",
                      f"E1 sensitivity: pairs where neither window includes an objective, {name}")
    rows.append(("e1_test_objfree_pairs", "report", f"{sens}/objective_free/pairs", "count", 0, 1.0, "test",
                 "E1 sensitivity: pairs where neither window includes an objective (diagnostics.py:152-158)"))
    rows.append(("e1_test_rstrat", "report", f"{sens}/stratified_r/r", "num", 3, 1.0, "test",
                 f"E1 stratified r over V(tau) quintiles, computed whatever its trigger says; {SIGN_NOTE}"))
    interval_rows("dedup_r", "report", f"{sens}/deduplicated_r/point/r", f"{sens}/deduplicated_r/r_ci", 3, "test",
                  "E1 de-duplicated r (one pair per reused placebo window), computed whatever its trigger says")
    rows.append(("e1_test_dedup_dropped", "report", f"{sens}/deduplicated_r/pairs_dropped", "count", 0, 1.0, "test",
                 "E1 de-duplicated r: pairs dropped (reused placebo windows)"))
    aux = "result/auxiliary"
    for arm in e1.ARMS:
        for family in e1.FAMILIES:
            rows.append((f"e1_test_auc_{arm}_{family.lower()}", "report", f"{aux}/auc/{arm}/auc/{family}", "num", 4, 1.0, "test",
                         f"E1 auxiliary: match-weighted AUC of the selected {shown[family]} cell of run {arm}, TEST"))
    interval_rows("g_eng_full", "report", f"{aux}/G_eng_full/point", f"{aux}/G_eng_full/interval", 5, "test",
                  "E1 auxiliary G_eng_full: q_run_1 main models (lgbm_l15_m200, ptflex_k8_C0.001) on the matched engagements",
                  MAIN_INTERVAL_NOTE)
    interval_rows("gplc_ratio", "report", f"{aux}/G_plc_over_main_test_gain/point", f"{aux}/G_plc_over_main_test_gain/interval", 3,
                  "test", "E1 auxiliary G_plc / 0.01386 (the full engagement TEST gain printed in PRESPEC_E1)")
    for stem, part, scope in (("main_matched", "matched", "the matched"), ("main_unmatched", "unmatched", "the unmatched"),
                              ("main_all", "all", "all")):
        interval_rows(stem, "report", f"{aux}/main_matched_vs_unmatched/{part}/point",
                      f"{aux}/main_matched_vs_unmatched/{part}/interval", 5, "test",
                      f"E1 auxiliary: q_run_1 main-model gain on {scope} TEST engagements", MAIN_INTERVAL_NOTE)
    balance = "result/summaries/balance"
    for stem, field in (("smd_v_start", "v_start"), ("smd_game_time", "game_time_min"), ("smd_kill_gap", "time_since_last_kill_s"),
                        ("smd_gold", "gold_difference"), ("smd_start_age", "start_frame_age_s")):
        rows.append((f"e1_test_{stem}", "report", f"{balance}/smd/{field}/smd", "num", 4, 1.0, "test",
                     f"E1 TEST balance (sealed summary opened in the TEST step): SMD of {field}, placebo minus engagement over "
                     "the pooled SD, row weighted"))
    for stem, arm in (("obj_share_eng", "engagement"), ("obj_share_plc", "placebo")):
        rows.append((f"e1_test_{stem}", "report", f"{balance}/objective_share/{arm}/share", "pct", 2, 100.0, "test",
                     f"E1 TEST share of {arm} windows that include an objective, percent (scale 100); sealed summary opened in "
                     "the TEST step"))
    # Task 16b fix round 1: appended after the first 107 rows (append_missing_rows), so they come last
    interval_rows("objfree_r", "qcal", f"{sens}/objective_free/point/r", f"{sens}/objective_free/r_ci", 3, "q_cal",
                  "E1 sensitivity: pairs where neither window includes an objective, r; Q_CAL is development evidence, not used "
                  "for the decision (PRESPEC_E1 section 6 step 5)")
    return rows


def ledger_rows(paths: Paths = Paths()) -> list[dict[str, str]]:
    """The E1 ledger rows with their printed values taken from the records (each must round-trip the thesis verifier)."""

    records = load(paths)
    out = []
    for key, record, locator, form, digits, scale, dataset, note in _spec():
        value = float(get(records[record], locator)) * scale
        printed = count(value) if form == "count" else num(value, digits)
        out.append({"key": key, "printed": printed, "kind": "json", "source": str(paths.record(RECORDS[record])), "locator": locator,
                    "scale": "1" if scale == 1.0 else f"{scale:g}", "dataset": dataset, "note": note})
    return out


def append_rows(ledger: Path, rows: Sequence[Mapping[str, str]]) -> None:
    """Append the rows once (LF endings, the ledger's header order); an existing e1_ key refuses the whole append."""

    ledger = Path(ledger)
    with ledger.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != LEDGER_COLUMNS:
            raise ValueError(f"{ledger} does not have the header {','.join(LEDGER_COLUMNS)}")
        present = {row["key"] for row in reader}
    clash = sorted(present & {row["key"] for row in rows}) + sorted(key for key in present if key.startswith("e1_"))
    if clash:
        raise FileExistsError(f"{ledger} already holds E1 rows: {clash[:5]}")
    data = ledger.read_bytes()
    if not data.endswith(b"\n") or b"\r" in data:
        raise ValueError(f"{ledger} must use LF line endings and end with a newline")
    with ledger.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        for row in rows:
            writer.writerow([row[name] for name in LEDGER_COLUMNS])


def _missing_rows(ledger: Path, rows: Sequence[Mapping[str, str]]) -> list[Mapping[str, str]]:
    """The rows whose keys the ledger lacks.  Every e1_ row already there must equal its generated row field for field, and
    no e1_ row may exist that the code does not make (ValueError otherwise: rows are never edited or dropped)."""

    ledger = Path(ledger)
    with ledger.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != LEDGER_COLUMNS:
            raise ValueError(f"{ledger} does not have the header {','.join(LEDGER_COLUMNS)}")
        present = {row["key"]: row for row in reader}
    generated = {row["key"]: row for row in rows}
    differs = sorted(key for key, row in present.items() if key.startswith("e1_") and (
        key not in generated or any(row[name] != generated[key][name] for name in LEDGER_COLUMNS)))
    if differs:
        raise ValueError(f"{ledger} holds e1_ rows that differ from the generated rows or that the code does not make: {differs[:5]}")
    data = ledger.read_bytes()
    if not data.endswith(b"\n") or b"\r" in data:
        raise ValueError(f"{ledger} must use LF line endings and end with a newline")
    return [row for row in rows if row["key"] not in present]


def append_missing_rows(ledger: Path, rows: Sequence[Mapping[str, str]]) -> list[str]:
    """Append only the rows whose keys the ledger lacks (LF, the ledger's header order); returns their keys."""

    missing = _missing_rows(ledger, rows)
    with Path(ledger).open("a", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        for row in missing:
            writer.writerow([row[name] for name in LEDGER_COLUMNS])
    return [row["key"] for row in missing]


# ----------------------------------------------------------------------------- the Korean report
def _quantity_table(primary: Mapping[str, Any]) -> list[str]:
    lines = ["| 판정량 | 점 | 95% 구간 |", "|---|---|---|"]
    for name in e1.QUANTITIES:
        digits = 3 if name == "r" else 5
        point, interval = num(primary["point"][name], digits), ci(primary[f"{name}_ci"], digits)
        if name in ("G_eng", "L1", "L2", "net"):
            point, interval = f"**{point}**", f"**{interval}**"
        lines.append(f"| {QUANTITY_LABEL[name]} | {point} | {interval} |")
    return lines


def _meaning(label: str, primary: Mapping[str, Any]) -> str:
    """The section 5 reading in words; every number sits in parentheses after its noun, so the particle follows the noun."""

    g, l1, l2, net, r = (primary[f"{name}_ci"] for name in ("G_eng", "L1", "L2", "net", "r"))
    positive = f"G_eng_size 구간의 하한({num(g[0], 5)})이 0보다 커서 크기를 맞춘 교전 모델의 판세 대비 이득은 있다."
    texts = {
        "no_gain": f"G_eng_size 구간의 상한({num(g[1], 5)})이 0보다 작다. 크기를 맞춘 교전 모델에 판세 대비 이득이 없다.",
        "inconclusive": f"G_eng_size 구간({ci(g, 5)})이 0을 포함한다.",
        "keep": f"{positive} L₁의 97.5 백분위({num(l1[1], 5)})가 0보다 작다. 95% 구간 기준으로 위약 이득은 교전 이득의 4분의 1보다 "
                "작다.",
        "rewrite": (f"{positive} L₁의 97.5 백분위({num(l1[1], 5)})가 0보다 작지 않아 \"keep\"이 아니고, L₂의 2.5 백분위"
                    f"({num(l2[0], 5)})가 0보다 커서 \"rewrite\"다. 95% 구간 기준으로 위약 이득이 교전 이득의 4분의 3보다 크다. "
                    f"r 구간의 하한"
                    f"({num(r[0], 3)})도 0.75보다 커서 r 읽기도 같은 판정이다(§4 동치)."
                    + (f" 순이득 구간의 상한({num(net[1], 5)})이 0보다 작다. 95% 구간 기준으로 짝이 맞은 집합에서 위약 창의 "
                       "LightGBM 이득이 교전의 이득보다 크다." if net[1] < 0 else "")),
        "net_gain": f"{positive} L₁·L₂ 규칙에 걸리지 않고 순이득 구간의 하한({num(net[0], 5)})이 0보다 크다.",
        "indistinguishable": f"{positive} L₁·L₂ 규칙에 걸리지 않고 순이득 구간의 하한({num(net[0], 5)})이 0보다 크지 않다.",
    }
    return texts[label]


def _openings_before(records: Mapping[str, Any], index: int) -> str:
    out = []
    for k, entry in enumerate(records["openings"]["openings"][:index]):
        files = ", ".join(f"`{label_name(label)}`" for label in entry.get("labels") or [])
        out.append(f"{entry['experiment']}({zulu(entry['utc'])}, `openings[{k}]`{'; ' + files if files else ''})")
    return "; ".join(out)


def _l2_reading(block: Mapping[str, Any]) -> str:
    low, high = block["L2_ci"]
    if low > 0:
        return "L₂ 구간이 0보다 크다."
    if high < 0:
        return "L₂ 구간이 0보다 작다."
    return "L₂ 구간이 0을 포함해 이 민감도는 결론이 나지 않는다(미결)."


def _r_reading(block: Mapping[str, Any]) -> str:
    """Where the r interval lies against 0.75 (the rewrite line, PRESPEC section 4) and 1 (equal gains)."""

    low, high = block["r_ci"]
    if low > 1:
        return f"r 구간의 하한({num(low, 3)})이 1보다 크다."
    if high < 0.75:
        return f"r 구간의 상한({num(high, 3)})이 0.75보다 작다."
    spans = [mark for mark in (0.75, 1.0) if low <= mark <= high]
    return {(0.75, 1.0): "r 구간이 0.75와 1을 모두 포함한다.", (1.0,): "r 구간이 1을 포함한다.",
            (0.75,): "r 구간이 0.75를 포함한다."}.get(tuple(spans), "r 구간이 0.75와 1 사이에 있다.")


def _limitations(windows: Mapping[str, Any], shares: Mapping[str, Any], objective: Mapping[str, Any], free: Mapping[str, Any],
                 qfree: Mapping[str, Any], primary: Mapping[str, Any]) -> list[tuple[str, list[str]]]:
    """Section 9 as (key, lines): the first line is numbered when printed, the rest are its sub-bullets."""

    def side(name: str, block: Mapping[str, Any]) -> str:
        return (f"   - {name}(짝 {count(block['pairs'])}개): r {num(block['point']['r'], 3)} {ci(block['r_ci'], 3)}, L₂ "
                f"{num(block['point']['L2'], 5)} {ci(block['L2_ci'], 5)}. {_l2_reading(block)} {_r_reading(block)}")

    q_r, t_r = qfree["point"]["r"], free["point"]["r"]
    objective_free = ["목표물 없는 짝 민감도의 Q_CAL 값과 TEST 값(보고만, 판정에 쓰지 않음; §3, §5).", side("Q_CAL", qfree),
                      side("TEST", free)]
    if (q_r < 1) != (t_r < 1):
        objective_free.append(f"   - r의 점이 Q_CAL에서는 1보다 {'작고' if q_r < 1 else '크고'}, TEST에서는 1보다 "
                              f"{'작다' if t_r < 1 else '크다'}.")
    objective_free.append(f"   - 저자의 TEST 전 결정 행(`reports/thesis_notes_20261005/DECISION_LOG_KO.md` 10-13, 사전 등록대로 TEST "
                          f"실행)은 Q_CAL의 목표물 없는 짝 r({num(q_r, 2)})을 인용했다.")
    if free["r_ci"][0] > 1:
        objective_free.append(f"   - TEST에서는 목표물 없는 짝에서도 r 구간이 1보다 크다(짝 맞은 전체의 r {num(primary['point']['r'], 3)} "
                              f"{ci(primary['r_ci'], 3)}).")
    return [
        ("short", [f"짧고 킬 직후인 창, 좁은 짝 집합. TEST 짝 비율은 {pct(windows['match_rate'])}%다.",
                   "   - 짝 맞은 교전은 짝 없는 교전보다 짧다(Q_DEV D 중앙값 29.0초 대 52.7초).",
                   f"   - 짝 맞은 교전은 대부분 직전 킬 뒤 30초 안에 시작한다(TEST {pct(shares['share_of_matched'], 1)}%, Q_DEV 71.5%).",
                   "   - 결론은 짝이 맞은 교전 집합에 적용된다(§1)."]),
        ("lag", ["직전 킬에 대한 늦은 반응. 위약의 표류에는 직전 킬에 대한 측정기의 늦은 반응이 들어 있을 수 있다(§1 한정 문장).",
                 "   - 짝 맞은 교전도 같은 조건에서 시작한다."]),
        ("imbalance", ["남은 불균형(두 방향).",
                       f"   - \"keep\" 쪽: 위약 창은 목표물을 덜 포함한다(TEST 교전 {pct(objective['engagement']['share'])}%, 위약 "
                       f"{pct(objective['placebo']['share'])}%; PRESPEC §3 배제).",
                       "   - \"keep\"에 불리한 쪽: 위약은 조금 더 큰 교전 뒤에 놓인다(직전 성분 크기 SMD +0.14, 짝 맞은 10-12판 칸 0 "
                       "쌍; Q_DEV 설계 모의이고 TEST 값은 없다).",
                       "   - 자료를 보기 전 두 검토(Task 13)는 종합해 남은 편향이 주로 \"keep\"에 불리한 쪽이라고 보았다(PRESPEC_E1 "
                       "§9 표의 10-13 배제 행).",
                       "   - 그래서 \"rewrite\"를 불균형을 거슬러 나온 보수적 판정으로 읽지 않는다.",
                       f"   - 위약 창의 {pct(windows['component_windows']['share'])}%는 q 교전이 아닌 성분의 결과 구간과 만난다."]),
        ("objective_free", objective_free),
        ("repeat", ["주 TEST 라벨의 반복 열람. 15.16 TEST 라벨은 E1 전에 여러 번 열렸다(2026-10-08 첫 열람, ToG E2·E4 재열람).",
                    "   - 전체 목록은 `EXPOSURE_KO.md`에 있다.",
                    "   - 위약 TEST 라벨과 TEST 창의 V(s + D)는 이번에 처음 열었다."]),
        ("seed", ["한 시드. q 카드는 한 시드이고, 구간은 경기 재표집만 반영한다."]),
        ("precision", ["정밀도. 위약 V는 CPU, 교전 라벨은 CUDA로 채점했다(§8).",
                       "   - TEST의 정확히 0과 1e\u22126 미만 크기의 수는 계산하지 않았다(SDD 원장 Ruling 61). 사전 등록된 TEST 단계가 "
                       "내지 않았고, 내려면 봉인 라벨을 다시 읽어야 한다."]),
        ("level", ["구간 수준. 학위논문의 전체 교전 P1 구간은 97.5%이고 E1은 95%다."]),
        ("report_only", ["민감도와 보조는 보고만 한다. 판정은 §5의 TEST 구간 규칙 하나다."]),
    ]


def render(paths: Paths = Paths(), revisions: Sequence[Mapping[str, str]] = REVISIONS) -> str:
    """E1_REPORT_KO.md from the records (module docstring); the revision records are printed in the header."""

    records = load(paths)
    facts = consistency(records)
    report, qcal, lock, train = records["report"], records["qcal"]["result"], records["lock"], records["train"]
    result = report["result"]
    primary, decision = result["primary"], result["decision"]
    windows, balance = result["summaries"]["windows"], result["summaries"]["balance"]
    sens, aux = result["sensitivities"], result["auxiliary"]
    selection = records["selection"]["selection"]
    logs = step_logs(paths, decision["outcome"])
    tog, sealed, index = paths.tog, lock["sealed_files"], facts["opening_index"]
    code = lock["code"][CODE_KEY]
    rows = ledger_rows(paths)
    objective = balance["objective_share"]
    shares = windows["matched_vs_unmatched_by_kill_gap_bin"]["0"]
    limits = _limitations(windows, shares, objective, sens["objective_free"], qcal["sensitivities"]["objective_free"], primary)
    repeat_item = 1 + [key for key, _ in limits].index("repeat")
    lines: list[str] = []
    add = lines.append

    add("# E1 결과: 위약 구간 대조(측정기 표류) (ToG 투고 전 실험)")
    add("")
    add("- 작성 2026-10-13 KST, 구현자(Opus, 계획서 Task 16 B단계). 이 문서와 원장 행은 `studies/tog_experiments_20261012/e1_report.py`가 "
        "`E1_REPORT.json`과 공개 기록(잠금, 학습, 선정, Q_CAL, 균형, 창, 로그, `OPENINGS.json`)에서 만들었다. 봉인 파일은 다시 읽지 않았다.")
    if revisions:
        add("- 고친 판 기록(쓰기 한 번 규칙의 예외는 `e1_report.py`의 `REVISIONS`에 적은 것뿐이다; `--replace-report`):")
        for record in revisions:
            add(f"  - {record['date']} {record['label']}: {record['reason']} 이전 판 SHA-256 `{record['previous_sha256']}`"
                f"(커밋 {record['previous_commit']}).")
    add(f"- 사전 명세: `reports/tog_experiments_20261012/PRESPEC_E1_KO.md`(2026-10-13 개정, 저자 재승인 커밋 {lock['prespec']['commit'][:7]}, "
        f"SHA-256 `{lock['prespec']['sha256']}`).")
    amendment = lock["amendment"]
    add(f"  - 규약 개정: `AMENDMENT_E1_KO.md`(커밋 {amendment['commit'][:7]}, SHA-256 `{amendment['sha256']}`).")
    if amendment.get("committed_sha256", amendment["sha256"]) != amendment["sha256"]:
        add(f"    - 위 해시는 잠금이 읽은 작업 사본(줄 끝 CRLF)의 것이다. 커밋된 내용의 SHA-256은 `{amendment['committed_sha256']}`이고, "
            "잠금은 두 값을 모두 적었다.")
    add(f"  - PRESPEC 해시가 bd300bf 판의 `{e1.PRESPEC_AT_BD300BF[:8]}…`(Task 13 r1 기록이 인용)와 다른 까닭은 덧붙인 재승인 문장 하나다"
        "(SDD 원장 Ruling 50, 잠금의 `prespec_note`).")
    add(f"- 코드: `studies/tog_experiments_20261012/e1_run.py`(커밋 {code['commit'][:7]}, SHA-256 `{code['sha256']}`; 학습, 선정, Q_CAL "
        "예행, 잠금, TEST 단계).")
    add("  - 위약 창은 `e1_candidates.py`(232e990), 상태와 위약 V는 `e1_states.py`(f7f9dc8), 행 표·원 입력 저장소·균형표는 "
        "`e1_rows.py`(12a9577)가 만들었다.")
    add(f"- 잠금: `{report['lock']['path']}`(SHA-256 `{report['lock']['sha256']}`, {zulu(report['lock']['locked_utc'])}, 파일 "
        f"{count(lock['hashing']['files'])}개).")
    add("  - A단계 검토(Fable, Opus)는 Critical 0, Important 0으로 통과했다(`.superpowers/sdd/PLAN_E1_E5_KO/task-16a-review-fable.md`, "
        "`task-16a-review-opus.md`).")
    add(f"- 봉인 열람: 한 번, {zulu(report['opening']['utc'])}(`reports/tog_experiments_20261012/OPENINGS.json` `openings[{index}]`). "
        "연 파일 다섯의 해시는 잠금과 같다.")
    add(f"  - 위약 TEST 라벨 `{sealed['plc_labels']['path']}`(SHA-256 `{sealed['plc_labels']['sha256']}`): 처음 열었다. 만들 때 바로 "
        "봉인했다(Task 15).")
    add(f"  - 위약 TEST 창의 끝 V `{sealed['pred_end']['path']}`(`{sealed['pred_end']['sha256']}`): 처음 열었다(Task 14).")
    add(f"  - 봉인 요약 `{sealed['windows_summary']['path']}`(`{sealed['windows_summary']['sha256']}`)와 "
        f"`{sealed['balance_summary']['path']}`(`{sealed['balance_summary']['sha256']}`): 처음 열었다.")
    add(f"  - 주 TEST 라벨 `{sealed['eng_labels']['path']}`(`{sealed['eng_labels']['sha256']}`): 모델 (나)의 라벨로 다시 열었다"
        f"(반복 열람, §9 한계 {repeat_item}).")
    add(f"- 결과 JSON: `{paths.record(e1.REPORT_NAME)}`(SHA-256 `{sha256_file(paths.record(e1.REPORT_NAME))}`).")
    add(f"  - 열람 기록 `{paths.record(e1.OPENING_NAME)}`(`{sha256_file(paths.record(e1.OPENING_NAME))}`)은 봉인 파일을 읽기 전에 썼다.")
    add(f"- 부호: {SIGN_NOTE}. G는 경기 가중 원 확률 Brier의 차다. L₁, L₂, 순이득도 같은 부호다.")
    add("  - E2–E4의 \"LightGBM − PT_flex\"(음수면 LightGBM이 나음)와 반대다.")
    add("- 구간: 백분위 95%, `numpy.quantile` linear, 경기 단위 공유 재표집 2,000회, PCG64 시드 20261011(`qeval.paired_match_bootstrap`).")
    add("  - 네 배열(교전 LightGBM, 교전 PT_flex, 위약 LightGBM, 위약 PT_flex의 경기 평균 손실)이 같은 재표집을 쓴다.")
    add("  - L₁, L₂, 순이득, r은 같은 회차의 G에서 낸다.")
    add(f"- E1 전에 열린 ToG TEST 결과(TEST 단계 직전 `OPENINGS.json`; 노출 장부 규칙 2): {_openings_before(records, index)}.")
    add("  - `E1_REPORT.json` `openings_before`의 목록과 같다.")
    add(f"- 자료: TEST 15.16의 짝 맞은 교전 {count(result['rows'])}개, 경기 {count(result['matches'])}개다.")
    add(f"  - TEST q 교전 {count(windows['engagements'])}개 가운데 {pct(windows['match_rate'])}%다.")
    add("")

    add("## 1. 질문과 범위")
    add("")
    add("PRESPEC_E1 §1(그대로 옮김):")
    add("")
    add("> 킬이 없는 구간에서도 교전 직전 상세 상태가 판세(교전 전 승리 확률과 경기 시간)보다 승리 확률 변화의 방향을 잘 맞히는가. 그렇다면 "
        "교전 이득의 일부는 측정기의 예측 가능한 표류다. E1은 킬이 없을 때의 표류만 잰다. 교전 창 안의 킬에 대한 측정기 반응의 편향은 재지 "
        "못하며, 그 몫은 킬 우위 라벨 결과(승리 확률 모델을 쓰지 않음)가 받친다. 다만 2026-10-13 개정 뒤 위약 창은 교전과 같은 직전 킬 뒤 "
        "시간 칸에서 시작하므로(§3 kill_gap_bin), 위약의 표류에는 직전 킬에 대한 측정기의 늦은 반응(예: 창 안의 분 프레임 갱신으로 뒤늦게 "
        "들어오는 몫)이 들어 있을 수 있고, 짝이 맞은 교전도 같은 조건에서 시작한다. E1의 결론은 짝이 맞은 교전 집합에 적용된다(§3 "
        "배제·kill_gap_bin). 이 범위는 한계에 적는다.")
    add("")
    add("- 두 모델은 PRESPEC §3 모델의 카드 16개(LightGBM 4, PT_flex 12)를 같은 동결 작업자로 학습하고, 계열마다 Q_DEV 원 Brier로 골랐다.")
    add("  - (가)는 위약 창(킬 없는 창)으로 학습했다. 입력은 위약 시작 s의 상태와 F(s)다.")
    add("  - (나)는 짝이 맞은 교전만으로 학습한 크기를 맞춘 교전 모델이다(계획서 D9).")
    add("")

    add("## 2. 짝 맞추기 설계 (2026-10-13 개정, 저자 선택 ①)")
    add("")
    add("- 후보: 교전마다 τ 기준 격자 s = τ + 1,000·k(|k| ≤ 600)에서 위약 시작 s를 찾는다. s < 120초인 점은 미리 뺀다. 창 길이 D는 "
        "교전의 것을 쓴다.")
    add("- 배제는 다섯 가지다(처음 걸린 규칙 하나에만 센다).")
    add("  - start_age_out: 시작 나이가 [0, 10초] 밖.")
    add("  - end_range: s + D가 마지막 프레임 시각보다 뒤이거나 경기 끝 이후.")
    add("  - kill_inside: [s, s + D]에 킬이 있음.")
    add("  - outcome_overlap: [s, s + D]가 같은 경기 **q 교전의 결과 구간** [f − 10초, e_h90](자기 교전 포함)과 겹침. q 교전이 "
        "아닌 성분으로는 배제하지 않는다. 개정 전에는 은행 성분 표 전체로 배제했다.")
    add("  - keys_differ: 짝 특성 여섯 가운데 하나라도 다름.")
    add("- 짝 특성 여섯:")
    add("  - 시작 나이 칸, 창 안 프레임 수(n_bound), 끝 나이 칸, 경기 시간 칸.")
    add("  - **직전 킬 뒤 시간 다섯 칸**: " + ", ".join(KILL_GAP_BINS) + ". 개정 전에는 세 칸이었다.")
    add("  - 골드 차 5분위: time_bin마다 TRAIN만으로 정한 경계(PRESPEC 부록 A).")
    add("- 선택: 남은 후보 가운데 SHA-256(engagement_key:s)이 가장 작은 것. 재사용은 허용한다. 짝이 없는 교전은 E1에서 빠지고 주 분석에는 "
        "남는다.")
    add(f"- 관문: 원 규칙의 시도 r0은 Q_DEV 짝 비율 3.79%로 30% 관문을 넘지 못했다. 저자가 위 개정을 골랐고, 시도 r1은 "
        f"{pct(records['windows']['gate_record']['q_dev_match_rate'])}%로 통과했다(`WINDOWS_h90.json` `gate_record`).")
    add("")

    add("## 3. 개발 자료 Q_CAL (먼저 보고, 판정에 쓰지 않음)")
    add("")
    add("- PRESPEC §6 5단계대로 잠금 전에 Q_CAL(선정에 쓰지 않은 15.15)에서 먼저 냈다. 개발 근거이며 판정과 설계에 쓰지 않았다.")
    add("- Q_CAL 결과를 본 뒤 정의를 바꾸지 않았다(§6 개발 결과 정책). 저자는 10-13에 개정 없이 사전 등록대로 TEST를 열기로 정했다.")
    add("")
    lines += _quantity_table(qcal["primary"])
    add("")
    add(f"- 짝 {count(qcal['rows'])}개, 경기 {count(qcal['matches'])}개다.")
    add(f"- G_eng_size ≤ 0인 회차는 2,000회 가운데 {count(qcal['primary']['draws_G_eng_le_0'])}회다.")
    add(f"- Q_CAL에서 decide()가 주는 꼬리표는 \"{qcal['decision']['outcome']}\"다. 판정은 아니다.")
    qs = qcal["sensitivities"]
    add("- Q_CAL 민감도의 r(보고만):")
    add(f"  - r₀.₀₁: {num(qs['r_0.01']['point']['r'], 3)} {ci(qs['r_0.01']['r_ci'], 3)}(경기 {count(qs['r_0.01']['matches'])}개).")
    add(f"  - 목표물 없는 짝: {num(qs['objective_free']['point']['r'], 3)} {ci(qs['objective_free']['r_ci'], 3)}"
        f"(짝 {count(qs['objective_free']['pairs'])}개).")
    add(f"  - 층화 r: {num(qs['stratified_r']['r'], 3)}. 중복 제거 r: {num(qs['deduplicated_r']['point']['r'], 3)}"
        f"(뺀 짝 {count(qs['deduplicated_r']['pairs_dropped'])}개).")
    add(f"- 출처: `{paths.record(e1.QCAL_NAME)}` `result`(SHA-256 `{sha256_file(paths.record(e1.QCAL_NAME))}`).")
    add("- 카드 선정(Q_DEV 원 Brier, `qselect.select_within`; 실행마다 16개 모두 완료):")
    add("")
    add("| 실행 | LightGBM | Q_DEV Brier | PT_flex | Q_DEV Brier |")
    add("|---|---|---|---|---|")
    for arm, name in ((e1.ENG, "eng (나)"), (e1.PLC, "plc (가)")):
        lgbm, pt = selection[arm]["lightgbm"], selection[arm]["PT_flex"]
        add(f"| {name} | {lgbm['selected']} | {num(lgbm['brier'], 6)} | {pt['selected']} | {num(pt['brier'], 6)} |")
    add("")

    add("## 4. 판정 (TEST 15.16)")
    add("")
    lines += _quantity_table(primary)
    add("")
    cells = result["cells"]
    add(f"- 셀별 경기 가중 Brier: 교전 LightGBM {num(cells['eng']['lightgbm'], 5)}, PT_flex {num(cells['eng']['PT_flex'], 5)}; "
        f"위약 LightGBM {num(cells['plc']['lightgbm'], 5)}, PT_flex {num(cells['plc']['PT_flex'], 5)}.")
    equivalence = primary["equivalence"]
    add(f"- G_eng_size ≤ 0인 회차는 2,000회 가운데 {count(primary['draws_G_eng_le_0'])}회다(빼지 않음).")
    add(f"  - L₁ ≥ 0인 회차는 {count(primary['draws_L1_ge_0'])}회, L₂ ≤ 0인 회차는 {count(primary['draws_L2_le_0'])}회다.")
    add(f"  - 모든 회차에서 G_eng_size > 0: {'예' if equivalence['every_draw_G_eng_positive'] else '아니오'}.")
    add(f"  - L 읽기와 r 읽기의 일치(§4 동치): {'예' if equivalence['readings_agree'] else '아니오'}.")
    add(f"- decide(): **\"{decision['outcome']}\"**. §5 규칙 원문(그대로 옮김):")
    add("")
    add(f"> {e1.DECISIONS[decision['outcome']]}")
    add("")
    add(f"- 뜻: {_meaning(decision['outcome'], primary)}")
    if decision.get("note"):
        add(f"- §5 규칙 3c(기록 `{decision['note']}`): 순이득 구간의 상한({num(primary['net_ci'][1], 5)})이 0보다 작다. 위약 이득이 "
            "교전 이득보다 크다는 것도 적는다.")
    if decision.get("default_handling"):
        add(f"- 기본 처리(§5 규칙 4): {decision['default_handling']}")
    add("- DECISION_LOG 행은 컨트롤러가 쓴다.")
    add("")

    add("## 5. 민감도 (보고만, 재학습 없음, 판정에 쓰지 않음)")
    add("")
    r001, free, dedup, strat = sens["r_0.01"], sens["objective_free"], sens["deduplicated_r"], sens["stratified_r"]

    def triple(block: Mapping[str, Any]) -> str:
        return (f"{num(block['point']['G_eng'], 5)} {ci(block['G_eng_ci'], 5)} | {num(block['point']['G_plc'], 5)} "
                f"{ci(block['G_plc_ci'], 5)} | {num(block['point']['r'], 3)} {ci(block['r_ci'], 3)}")

    add("| 민감도 | 범위 | G_eng_size | G_plc | r |")
    add("|---|---|---|---|---|")
    add(f"| r₀.₀₁ (\\|교전 가치\\| ≥ 0.01, \\|W\\| ≥ 0.01) | 네 배열 모두에 행이 있는 경기 {count(r001['matches'])}개(빠진 경기 "
        f"{count(r001['matches_dropped'])}개); 그 경기 안 교전 행 {count(r001['rows_eng'])}, 위약 행 {count(r001['rows_plc'])} | {triple(r001)} |")
    add(f"| 목표물 없는 짝 | 짝 {count(free['pairs'])}개, 경기 {count(free['matches'])}개 | {triple(free)} |")
    add(f"| 중복 제거 r | 짝 {count(dedup['pairs_kept'])}개(뺀 짝 {count(dedup['pairs_dropped'])}개) | {triple(dedup)} |")
    add(f"| 층화 r | V(τ) 5분위, 층 가중 = 교전 몫 | — | — | {num(strat['r'], 3)} |")
    add("")
    add("- r₀.₀₁의 행 수는 교차 뒤 값이다. 문턱을 넘은 행 가운데 네 배열 모두에 행이 있는 경기, 곧 교집합 경기 안의 행만 센다"
        "(Task 16a 검토 Fable 2, Opus 5).")
    add(f"  - 문턱만 넘은 행의 수는 `E1_REPORT.json`에 없다. 문턱을 넘은 행이 있는 경기는 교전 {count(r001['matches_eng'])}개, 위약 "
        f"{count(r001['matches_plc'])}개다.")
    add(f"- 목표물 없는 짝: L₂ {num(free['point']['L2'], 5)} {ci(free['L2_ci'], 5)}, 순이득 {num(free['point']['net'], 5)} "
        f"{ci(free['net_ci'], 5)}.")
    trig_s, trig_d = balance["triggers"]["stratified_r"], balance["triggers"]["deduplicated_r"]
    add(f"- 층화 r의 촉발 조건(V(시작)이나 경기 시간의 |SMD| > 0.1)은 TEST에서 {'참' if strat['trigger'] else '거짓'}이다.")
    add(f"  - 봉인 요약 `BALANCE_TEST_SEALED.json`의 |SMD|: V(시작) {num(trig_s['abs_smd_v_start'], 4)}, 경기 시간 "
        f"{num(trig_s['abs_smd_game_time'], 4)}.")
    add("  - 촉발과 상관없이 계산했다.")
    add(f"  - 5분위 경계(짝 맞은 TEST 교전의 V(τ)): {', '.join(num(edge, 4) for edge in strat['edges'])}.")
    add("  - 층별 r: " + ", ".join(f"층 {k} {num(v['r'], 3)}" for k, v in strat["strata"].items() if v["r"] is not None) + ".")
    add(f"- 중복 제거 r의 촉발 조건(재사용 위약 창의 행이 위약 행의 5% 초과)은 TEST에서 {'참' if dedup['trigger'] else '거짓'}이다.")
    add(f"  - 재사용 창의 행 비율은 {num(100 * trig_d['share_rows_in_reused_windows'], 4)}%다. 촉발과 상관없이 계산했다.")
    add("")

    add("## 6. 보조 (보고만, 판정에 쓰지 않음)")
    add("")
    for arm, name in ((e1.ENG, "모델 (나) 교전"), (e1.PLC, "모델 (가) 위약")):
        auc = aux["auc"][arm]
        add(f"- 경기 가중 AUC, {name}: LightGBM {num(auc['auc']['lightgbm'], 4)}, PT_flex {num(auc['auc']['PT_flex'], 4)}.")
        add(f"  - 차(LightGBM − PT_flex): {num(auc['auc_difference']['point'], 4)} {ci(auc['auc_difference']['interval'], 4)}.")
    add("- AUC 차의 구간은 사전 명세 밖이다(E2, E4와 같은 방식으로 같은 재표집에 덧붙임).")
    full = aux["G_eng_full"]
    add(f"- G_eng_full: {num(full['point'], 5)} {ci(full['interval'], 5)}.")
    add("  - 주 모델은 q_run_1의 고른 LightGBM lgbm_l15_m200 셀과 PT_flex ptflex_k8_C0.001 셀이다. 범위는 짝 맞은 TEST 교전이다.")
    add("  - 아래 \"짝 있음\"과 같은 양이다.")
    ratio = aux["G_plc_over_main_test_gain"]
    add(f"- G_plc / 0.01386 = {num(ratio['point'], 3)} {ci(ratio['interval'], 3)}.")
    add("  - 0.01386은 전체 교전 TEST 이득이다(PRESPEC 첫머리; 그 구간은 97.5%).")
    split = aux["main_matched_vs_unmatched"]
    add("- 주 모델 이득, 짝 있음 대 짝 없음(기술용, 경기 재표집은 각각):")
    for part, name in (("matched", "짝 있음"), ("unmatched", "짝 없음"), ("all", "전체")):
        add(f"  - {name}: {num(split[part]['point'], 5)} {ci(split[part]['interval'], 5)}(행 {count(split[part]['rows'])}, "
            f"경기 {count(split[part]['matches'])}).")
    add(f"  - 짝 있음 − 짝 없음 = {num(split['difference_point'], 5)}(점).")
    gap = split["all"]["point"] + THESIS_P1_POINT
    add(f"  - 전체 점은 학위논문 P1 점(`{win('q_run_1', 'TEST_REPORT.json')}` `bootstrap.contrasts.P1.point` "
        f"{num(THESIS_P1_POINT, 16)})의 부호를 바꾼 값과 " + ("같다." if gap == 0.0 else f"{gap:.1e}만큼 다르다.")
        + " P1 구간은 97.5%다.")
    add("")

    add("## 7. 짝 맞은 집합의 대표성")
    add("")
    add("TEST(봉인 요약을 이번 TEST 단계에서 처음 열었다; `E1_REPORT.json` `result.summaries`):")
    add("")
    gap_bins = windows["bins"]["kill_gap_bin"]
    add(f"- 짝 비율: {pct(windows['match_rate'])}%({count(windows['matched'])}/{count(windows['engagements'])}).")
    add(f"  - 짝 맞은 경기는 {count(windows['matched_matches'])}/{count(windows['matches'])}개다.")
    add("  - 직전 킬 뒤 시간 칸별 짝 비율: " + ", ".join(
        f"{name} {pct(gap_bins[str(b)]['match_rate'], 1)}%" for b, name in enumerate(KILL_GAP_BINS)) + ".")
    add(f"  - 직전 킬 뒤 30초 안에 시작한 교전은 짝 맞은 교전의 {pct(shares['share_of_matched'], 1)}%, 짝 없는 교전의 "
        f"{pct(shares['share_of_unmatched'], 1)}%다.")
    smd = balance["smd"]
    add(f"- SMD(위약 − 교전, 행 가중, 합동 SD): V(시작) {num(smd['v_start']['smd'], 4)}, 경기 시간 {num(smd['game_time_min']['smd'], 4)}, "
        f"골드 차 {num(smd['gold_difference']['smd'], 4)}, 시작 프레임 나이 {num(smd['start_frame_age_s']['smd'], 4)}, 직전 킬 뒤 시간 "
        f"{num(smd['time_since_last_kill_s']['smd'], 4)}.")
    add(f"  - 직전 킬 뒤 시간은 두 창 모두 앞선 킬이 있는 짝 {count(smd['time_since_last_kill_s']['pairs'])}개로 냈다(빠진 짝 "
        f"{count(smd['time_since_last_kill_s']['pairs_dropped_without_earlier_kill'])}개).")
    add(f"- 목표물 포함 비율: 교전 {pct(objective['engagement']['share'])}%, 위약 {pct(objective['placebo']['share'])}%.")
    add(f"  - 두 창 모두 목표물이 없는 짝은 {count(objective['pairs_without_objective_in_either'])}개다.")
    add(f"- 겹침: 위약 창의 {pct(windows['overlap']['share'])}%가 시작이 다른 위약 창과 겹친다.")
    add(f"  - {pct(windows['component_windows']['share'])}%는 q 교전이 아닌 성분의 결과 구간과 만난다(배제 대상 아님).")
    add(f"- 재사용 창의 행은 {count(windows['reuse']['rows_in_reused_windows'])}개다. F(s) XY가 유한하지 않은 창은 "
        f"{count(windows['xy']['nonfinite_f_s_xy'])}개다.")
    add("")
    add("비TEST(`BALANCE.json`, `WINDOWS_h90.json`; 잠금 전 공개 기록):")
    add("")
    add("| 역할 | 짝 | 짝 비율 | SMD V(시작) | SMD 경기 시간 | SMD 직전 킬 뒤 시간 | 목표물 포함 교전 / 위약 | D 4분위별 짝 비율 | "
        "D 중앙값 짝 / 없음 |")
    add("|---|---|---|---|---|---|---|---|---|")
    for role in ("TRAIN", "Q_DEV", "Q_CAL"):
        block = records["balance"]["roles"][role]
        quartiles = block["extras"]["d_quartiles"]["quartiles"]
        medians = block["extras"]["medians"]["D_s"]
        add(f"| {role} | {count(block['pairs'])} | {pct(records['windows']['roles'][role]['match_rate'])}% | "
            f"{num(block['smd']['v_start']['smd'], 4)} | {num(block['smd']['game_time_min']['smd'], 4)} | "
            f"{num(block['smd']['time_since_last_kill_s']['smd'], 4)} | {pct(block['objective_share']['engagement']['share'])}% / "
            f"{pct(block['objective_share']['placebo']['share'])}% | "
            + " / ".join(f"{pct(quartiles[str(k)]['match_rate'], 1)}%" for k in range(4))
            + f" | {num(medians['matched'], 1)} / {num(medians['unmatched'], 1)}초 |")
    add("")
    add("설계 단계의 사실(Q_DEV; SDD 원장 `.superpowers/sdd/PLAN_E1_E5_KO/progress.md`의 \"Carry-forward to Task 15/16\" 줄; 승리 확률·"
        "라벨·W를 보지 않은 값):")
    add("")
    for fact in CARRY_FORWARD:
        add(f"- {fact}")
    add("")

    add("## 8. 정밀도 공개")
    add("")
    add("- 위약 V는 CPU로 채점했다(Task 14: `score.LockedV`, device cpu, 1,024행 정규 묶음).")
    add("  - CPU에서 상태의 V는 묶음 구성에 따라 약 3e−7까지 달라진다(Task 14 보고서 우려 1).")
    add("- 교전 라벨은 CUDA로 채점한 주 라벨(`labels_1`)이다.")
    add("- 재현 관문(SDD 원장 Ruling 51): 비TEST 1,400행에서 CPU로 다시 낸 V와 저장된 pt_p_pre(CUDA)의 차가 모두 1e−5 이하였다.")
    add("  - 최대 차는 2.755e−7이고, 1e−6을 넘는 행은 없었다(Task 14 보고서).")
    add("- 정확히 0과 1e−6 미만의 크기(비TEST, `BALANCE.json` `precision`):")
    add("")
    add("| 역할 | 위약 W 정확히 0 | 위약 \\|W\\| < 1e−6 (0 아닌 최소 \\|W\\|) | 교전 Δ 정확히 0 | 교전 \\|Δ\\| < 1e−6 "
        "(0 아닌 최소 \\|Δ\\|) |")
    add("|---|---|---|---|---|")
    for role in ("TRAIN", "Q_DEV", "Q_CAL"):
        precision = records["balance"]["roles"][role]["precision"]
        plc, eng = precision["placebo"], precision["engagement"]
        add(f"| {role} | {count(plc['exact_zero'])} | {count(plc['abs_lt_1e-6'])} ({sci(plc['min_nonzero_abs'])}) | "
            f"{count(eng['exact_zero'])} | {count(eng['abs_lt_1e-6'])} ({sci(eng['min_nonzero_abs'])}) |")
    add("")
    add("- TEST의 같은 수는 사전 등록된 TEST 단계가 계산하지 않았다(`E1_REPORT.json`에 없음).")
    add("  - 봉인 라벨을 다시 읽지 않으려고 여기서도 내지 않았다(SDD 원장 Ruling 61).")
    add("")

    add("## 9. 한계")
    add("")
    for number, (_, item) in enumerate(limits, start=1):
        add(f"{number}. {item[0]}")
        lines += item[1:]
    add("")

    add("## 10. 절차와 점검 (A·B단계)")
    add("")
    add("- A단계(잠금까지; `.superpowers/sdd/PLAN_E1_E5_KO/task-16a-report.md`):")
    wall = train["wall"]
    launches = sum(int(wall[arm]["launches"]) for arm in e1.ARMS)
    add(f"  - 카드 16개 × 두 실행 = {count(len(train['jobs']))}개가 모두 끝났다(띄운 작업 {count(launches)}개). 작업자는 최대 넷, "
        "LightGBM은 한 번에 하나였고 CPU만 썼다.")
    add(f"  - 학습 벽시계: 전체 {train['timing']['wall_seconds']:,.1f}초(eng {wall['eng']['seconds']:,.1f}초, plc "
        f"{wall['plc']['seconds']:,.1f}초, 두 실행이 겹쳐 돎).")
    reproduce = records["qcal"].get("rehearsal", {}).get("checks", {}).get("scores_reproduce_the_stored_q_cal_predictions", {})
    add("  - Q_CAL 예행: TEST 단계의 평가 함수(`evaluate_opened`)를 비TEST 대역 파일로 끝까지 돌렸다(Ruling 55). 결과는 §3이다.")
    if reproduce:
        add(f"    - 고른 네 셀을 `qtest.LoadedCell`로 다시 불러 낸 Q_CAL 로짓과 저장값의 최대 차: {max(reproduce.values())}.")
    add(f"  - 잠금은 파일 {count(lock['hashing']['files'])}개를 묶었고 교차 점검 {facts['lock_cross_checks']}개가 모두 참이었다.")
    add(f"  - 봉인 요약 {count(len(lock['sealed_summaries']['files']))}개를 경로 규칙으로 찾아 해시로 묶었다(Ruling 58).")
    env = report["environment"]
    add("- B단계(이번 TEST 단계), 실행 전 점검:")
    add("  - 잠금 해시가 위 값과 같다.")
    add(f"  - 환경(Python {env['python']}, numpy {env['numpy']}, scipy {env['scipy']}, lightgbm {env['lightgbm']}, sklearn "
        f"{env['sklearn']})이 잠금의 기록과 같다.")
    add(f"  - TEST 단계가 기록한 코드 해시 {facts['code_files']}개가 모두 잠금과 같다(`e1_run.py` 포함).")
    add("  - `verify_lock`에 문제가 없다. 봉인 라벨을 여는 다른 프로세스가 없다.")
    add("  - 잠금 뒤 커밋 04a8562는 `EXPOSURE_KO.md`만 바꿨다. 잠금은 이 문서를 묶지 않고 기록만 한다.")
    add("- B단계 실행:")
    add(f"  - 명령: `python -B -m studies.tog_experiments_20261012.e1_run test --confirm-single-opening --lock-sha256 "
        f"{report['lock']['sha256']}`.")
    add(f"  - 로그: `{paths.record(STDOUT_LOG)}`, `{paths.record(STDERR_LOG)}`. 종료 코드 0.")
    add(f"  - stdout의 마지막 줄은 `E1_REPORT.json`의 요약이고 결과는 \"{decision['outcome']}\"다.")
    add(f"  - stderr에는 sklearn의 특성 이름 경고(UserWarning) {logs['feature_name_warnings']}건만 있다(그 밖의 줄 "
        f"{logs['other_stderr_lines']}개).")
    reload = sorted({meta["reload_max_abs_logit_diff"] for scores in report["scores"].values() for meta in scores.values()})
    reload_rows = sorted({meta["reload_rows"] for scores in report["scores"].values() for meta in scores.values()})
    add("  - 순서:")
    add("    1. 잠금 재확인(파일, 규칙, 환경, q 릴리스).")
    add("    2. 두 행 표의 TEST 봉인 확인.")
    add(f"    3. 네 셀 채점(`e4_q.score_cell`, 동결 `qtest.LoadedCell`; 행 표의 TEST y = −1). 저장된 Q_DEV 앞 "
        f"{', '.join(count(v) for v in reload_rows)}행 재채점 차는 {', '.join(str(v) for v in reload)}이다.")
    add("    4. 네 셀의 TEST 점수 파일.")
    add("    5. `E1_OPENING.json`, 이어서 `OPENINGS.json` 덧붙임.")
    add("    6. 봉인 파일 다섯 한 번 읽기.")
    add("    7. 평가.")
    add("    8. `E1_REPORT.json`.")
    add(f"  - 열람 뒤 점검 {facts['checks']}개가 모두 참이었다. 예:")
    add("    - 위약 라벨이 V(s + D) − V(s)와 같다.")
    add("    - 교전 라벨이 y = 1[SVI > 0]을 따른다.")
    add("    - 봉인 요약이 짝을 설명한다(짝 수, 교전 수, 목표물 수, 재사용).")
    add("    - 점수가 행 표의 TEST 행이다.")
    test = report["stages"]["test"]
    parts = test["parts"]
    add(f"  - 단계 시간은 {num(test['wall_seconds'], 1)}초다(잠금 확인 {num(parts['verify_lock']['wall_seconds'], 1)}, 채점 "
        f"{num(parts['score']['wall_seconds'], 1)}, 평가 {num(parts['evaluate']['wall_seconds'], 1)}).")
    add(f"  - 최대 메모리는 {num(test['peak_memory_bytes_main_process'] / 2**30, 2)} GiB였다. 프로세스는 하나였고 GPU는 쓰지 않았다.")
    add("- `qtest.evaluate`, `qeval.kill_count_strata`, `qeval.credit_strata`는 부르지 않았다(Ruling 55).")
    add("- 실패 기록(E1_FAILED_*, E1_BLOCKED_*)은 없다. Ruling 31에 따른 재실행은 없었다.")
    add("")

    add("## 11. 산출물과 해시")
    add("")
    add("| 파일 | SHA-256 |")
    add("|---|---|")
    for name in (e1.REPORT_NAME, e1.OPENING_NAME, e1.LOCK_NAME, e1.QCAL_NAME, e1.SELECTION_NAME, e1.TRAIN_NAME, STDOUT_LOG, STDERR_LOG):
        add(f"| `{paths.record(name)}` | `{sha256_file(paths.record(name))}` |")
    for path in sorted(tog.glob(f"*/{e1.SCORES_DIR}/*")):
        if path.is_file():
            add(f"| `{path}` | `{sha256_file(path)}` |")
    add("")
    add(f"- 열람 기록: `reports/tog_experiments_20261012/OPENINGS.json` `openings[{index}]`(TEST 단계가 씀), `EXPOSURE_KO.md` "
        "덧붙임 행.")
    add(f"- 원장: `reports/tog_paper_20261011/ledger/TOG_NUMBERS.csv`, `e1_` 행 {count(len(rows))}개. 출처는 절대 경로다.")
    add(f"  - 부호가 있는 행의 note에 \"{SIGN_NOTE}\"를 적었다.")
    add("  - 마지막 세 행(`e1_qcal_objfree_r`, `_lo`, `_hi`)은 Task 16b 수정 1회차에 없는 키만 덧붙인 것이다. 앞의 행은 고치지 않았다.")
    add("  - `dataset` 칸은 `q_cal`, `q_dev`로 적었다. e2_·e4_ 행은 `qdev`를 쓴다. 기존 행을 고치지 않으려고 그대로 둔다.")
    add(f"  - 원장 note의 부호 문장 \"{SIGN_NOTE}\"는 G(G_eng_size, G_plc와 그 민감도·보조 이득)에 대한 것이다. G의 선형 대비와 "
        "비는 아래처럼 읽는다. 원장 행은 덧붙이기만 하므로 note는 고치지 않는다.")
    add("    - L₁ = G_plc − 0.25·G_eng_size: 양수면 위약 이득이 교전 이득의 4분의 1보다 크다(`e1_*_l1`).")
    add("    - L₂ = G_plc − 0.75·G_eng_size: 양수면 위약 이득이 교전 이득의 4분의 3보다 크다(`e1_*_l2`, `e1_test_objfree_l2`).")
    add("    - 순이득 = G_eng_size − G_plc: 양수면 교전 이득이 더 크고, 음수면 위약 이득이 더 크다(`e1_*_net`).")
    add("    - r = G_plc / G_eng_size(기술만): G_eng_size가 양수일 때 1보다 크면 위약 이득이 더 크다(`e1_*_r`, `e1_test_r001_r`, "
        "`e1_test_objfree_r`, `e1_test_dedup_r`, `e1_test_rstrat`, `e1_qcal_objfree_r`).")
    flag, net = bool(decision.get("placebo_gain_exceeds_engagement_gain")), primary["net_ci"]
    add(f"- 판정 칸 `placebo_gain_exceeds_engagement_gain`(`E1_REPORT.json` `result.decision`)의 값은 {'참' if flag else '거짓'}이다.")
    add("  - 이 칸은 §5 규칙 3c, 곧 \"indistinguishable\" 분기에서만 계산된다(`e1_run.py`의 `decision()`; 잠금된 코드라 고치지 "
        "않는다).")
    if decision["outcome"] != "indistinguishable":
        add(f"  - TEST 판정(\"{decision['outcome']}\")에서는 이 칸을 계산하지 않는다. 거짓은 \"위약 이득이 교전 이득보다 크지 "
            "않다\"는 뜻이 아니다.")
    if net[1] < 0:
        add(f"  - TEST의 순이득 구간({ci(net, 5)})은 모두 0보다 작다. 위약 이득이 교전 이득보다 크다.")
    elif net[0] > 0:
        add(f"  - TEST의 순이득 구간({ci(net, 5)})은 모두 0보다 크다. 교전 이득이 위약 이득보다 크다.")
    else:
        add(f"  - TEST의 순이득 구간({ci(net, 5)})은 0을 포함한다.")
    add("- DECISION_LOG 행은 컨트롤러가 쓴다.")
    add("")
    return "\n".join(lines)


def write_report(paths: Paths = Paths(), *, replace: bool = False, revisions: Sequence[Mapping[str, str]] = REVISIONS) -> Path:
    """E1_REPORT_KO.md written once (UTF-8, LF); a text holding a TAB or CR character is refused.

    replace=True is the one recorded way to write it again: the file must exist and its SHA-256 must equal the
    previous_sha256 of the last revision record, which the new text prints with its reason."""

    target = Path(paths.report_md)
    if target.exists() and not replace:
        raise FileExistsError(f"{target} exists; it is written once (a replacement needs replace=True and a revision record)")
    if replace:
        if not revisions:
            raise ValueError("a replacement needs a revision record (REVISIONS)")
        if not target.exists():
            raise FileNotFoundError(f"{target} does not exist; there is nothing to replace")
        current = sha256_file(target)
        if current != revisions[-1]["previous_sha256"]:
            raise ValueError(f"{target} is {current}, not the version the last revision record replaces "
                             f"({revisions[-1]['previous_sha256']})")
    text = render(paths, revisions)
    bad = sorted({repr(ch) for ch in text if ch in (chr(9), chr(13))})
    if bad:
        raise ValueError(f"the report holds control characters {bad}")
    target.write_bytes(text.encode("utf-8"))
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replace-report", action="store_true",
                        help="replace E1_REPORT_KO.md under the last REVISIONS record and append only the missing e1_ ledger rows")
    args = parser.parse_args(argv)
    paths = Paths()
    rows = ledger_rows(paths)
    if args.replace_report:
        _missing_rows(paths.ledger, rows)                      # refuse before writing anything when an existing row differs
        target = write_report(paths, replace=True)
        appended = append_missing_rows(paths.ledger, rows)
    else:
        target = write_report(paths)
        append_rows(paths.ledger, rows)
        appended = [row["key"] for row in rows]
    print(json.dumps({"report": str(target), "sha256": sha256_file(target), "ledger_rows": len(rows), "appended": appended}),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
