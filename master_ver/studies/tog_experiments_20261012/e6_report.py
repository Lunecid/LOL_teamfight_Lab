"""E6 part B documents: the Korean E6 report and the e6_ rows of the ToG ledger, rendered from the records of the single
E6 test step (PRESPEC_E6_KO.md section 8.3 B3 and section 12; SDD Ruling 60: the locked e6_run.py stays byte-identical).

Reads records only, never a sealed file: tog_e6_1/E6_REPORT.json (the test step's record; it holds the results opened in
that step), E6_OPENING.json, the lock the report names, the test step's logs (logs/E6_TEST.stdout.log, .stderr.log,
.exit.txt) and reports/tog_experiments_20261012/OPENINGS.json.  Writes, once each:
  reports/tog_experiments_20261012/E6_REPORT_KO.md (refused when it exists);
  the e6_ rows of reports/tog_paper_20261011/ledger/TOG_NUMBERS.csv (refused when an e6_ key exists; rows are never
    edited).
Ledger rows follow the thesis schema key,printed,kind,source,locator,scale,dataset,note with kind json, a '/'-joined
locator into the absolute path of E6_REPORT.json, dataset external_e6, printed values with U+2212 and thousands
separators, the sign convention, interval method and status in the note (PRESPEC_E6 section 12), and no note holding the
substring that marks a magnitude-only row.  Windows paths come from Path objects and chr(92); a text holding a TAB or CR
character is refused.  Rendering stops (ValueError) when the records disagree (check()).

Usage: python -B -m studies.tog_experiments_20261012.e6_report
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studies.tog_experiments_20261012 import e6_run as E6  # noqa: E402  (constants and readers only)

MINUS = "−"
LEDGER_COLUMNS = ("key", "printed", "kind", "source", "locator", "scale", "dataset", "note")
DATASET = "external_e6"
FIXED_STDOUT = "E6 TEST written (attempt 1)\n"
CELL_TAGS = {E6.P1: "p1", E6.PT: "pt", E6.S10: "s10", E6.S00: "s00", E6.S11: "s11", E6.S01: "s01"}
CELL_NAMES = {E6.P1: "P1 (LightGBM 전체 입력, LGBM_S1_A1_M1_E1_C1_R1_D1)", E6.PT: "PT_flex (판세 기준)",
              E6.S10: "S1D0", E6.S00: "S0D0", E6.S11: "S1D1", E6.S01: "S0D1"}
STRATA = (("1", "k1", "1킬"), ("2-3", "k23", "2–3킬"), ("4+", "k4", "4+킬"))
DECISION_WORD = {"confirmed": "확증", "unresolved": "미결", "opposite": "반대 방향"}

SIGN = "LightGBM − PT_flex, negative = LightGBM better"
S_SIGN = ("S = 1/2[Brier(S1D0) − Brier(S0D0)] + 1/2[Brier(S1D1) − Brier(S0D1)], negative = the position input "
          "lowers the Brier")
TWOWAY = "95% player-match two-way interval theta -/+ 1.96 sqrt(V), V = max(V1, VU)"
BOOT = "95% percentile interval, paired match bootstrap 2,000, seed 20261011"
STRATA_BOOT = "95% percentile interval, paired match bootstrap inside the stratum, 2,000, seed 20261011"
STATUS = "pre-registered confirmatory test E6 (PRESPEC_E6_KO.md)"
SECONDARY = "pre-registered secondary (PRESPEC_E6_KO.md section 7)"
STRATA_STATUS = ("descriptive, kill-record strata (PRESPEC_E6_KO.md section 7 (c); KA ties dropped, so 2-3 and 4+ leave "
                 "out even trades)")
SCOPE = ("external KR 15.18-15.22 slice never processed by this lineage (40,000 selected), non-tie rows of the KA_v1 "
         "kill-advantage label, q_ka_1 cells")


@dataclass(frozen=True)
class Paths:
    out: Path = E6.OUT_DEFAULT
    openings: Path = ROOT / "reports" / "tog_experiments_20261012" / "OPENINGS.json"
    ledger: Path = ROOT / "reports" / "tog_paper_20261011" / "ledger" / "TOG_NUMBERS.csv"
    report_md: Path = ROOT / "reports" / "tog_experiments_20261012" / "E6_REPORT_KO.md"

    @property
    def report_json(self) -> Path:
        return Path(self.out) / "E6_REPORT.json"

    @property
    def opening_json(self) -> Path:
        return Path(self.out) / "E6_OPENING.json"

    @property
    def logs(self) -> Path:
        return Path(self.out) / "logs"


# ----------------------------------------------------------------------------- small formatting helpers
def num(value: float, digits: int) -> str:
    """Fixed decimals with U+2212 for a negative value."""

    return f"{float(value):.{digits}f}".replace("-", MINUS)


def signed(value: float, digits: int) -> str:
    """Fixed decimals with an explicit + for a positive value and U+2212 for a negative one."""

    text = num(value, digits)
    return text if text.startswith(MINUS) else "+" + text


def count(value: float) -> str:
    return f"{int(value):,}"


def ci(interval: Sequence[float], digits: int, plus: bool = False) -> str:
    fmt = signed if plus else num
    return f"[{fmt(interval[0], digits)}, {fmt(interval[1], digits)}]"


def zulu(iso: str) -> str:
    return datetime.fromisoformat(iso).strftime("%Y-%m-%d %H:%M:%SZ")


def win(*parts: str) -> str:
    """A Windows path from its parts joined by chr(92)."""

    return chr(92).join(parts)


def get(record: Any, locator: str) -> Any:
    for part in locator.split("/"):
        record = record[int(part)] if isinstance(record, list) else record[part]
    return record


# ----------------------------------------------------------------------------- records
def load(paths: Paths = Paths()) -> dict[str, Any]:
    """Every record the documents use (no sealed file; E6.read_json refuses SEALED names)."""

    report = E6.read_json(paths.report_json)
    lock_path = Path(report["lock"]["path"])
    logs = paths.logs
    stdout = logs / "E6_TEST.stdout.log"
    stderr = logs / "E6_TEST.stderr.log"
    exit_file = logs / "E6_TEST.exit.txt"
    out = Path(paths.out)
    after = sorted(p.name for p in out.iterdir() if p.is_file() and (
        p.name.startswith(("E6_FAILED", "TEST_BLOCKED", "E6_OPENING_a", "E6_REPORT_a")) or "_SEALED_" in p.name))
    return {"paths": paths, "report": report, "opening": E6.read_json(paths.opening_json),
            "openings": E6.read_json(paths.openings), "lock": E6.read_json(lock_path), "lock_path": lock_path,
            "lock_sha256": E6.sha256_file(lock_path), "report_sha256": E6.sha256_file(paths.report_json),
            "opening_sha256": E6.sha256_file(paths.opening_json),
            "logs": {"stdout_path": str(stdout), "stderr_path": str(stderr),
                     "stdout": stdout.read_text(encoding="utf-8") if stdout.is_file() else None,
                     "stderr_bytes": stderr.stat().st_size if stderr.is_file() else None,
                     "exit": exit_file.read_text(encoding="utf-8").strip() if exit_file.is_file() else None},
            "after_opening_files": after}


ENTRY_FIELDS = ("experiment", "attempt", "utc", "lock", "lock_sha256", "prespec_sha256", "labels", "sealed_sha256", "note")


def check(records: Mapping[str, Any]) -> dict[str, Any]:
    """The records agree (ValueError otherwise): a completed attempt 1, the same opening entry in the report, in
    E6_OPENING.json and once in OPENINGS.json, the lock on disk is the one the opening names, the code the report ran is
    the locked code, the pre-opening checks held, and the recorded decisions follow PRESPEC section 6."""

    report, opening = records["report"], records["opening"]
    problems = []
    if report.get("status") != "completed" or report.get("experiment") != "E6" or report.get("attempt") != 1:
        problems.append("the report is not a completed E6 attempt 1")
    entry = report.get("opening") or {}
    if any(entry.get(k) != opening.get(k) for k in ENTRY_FIELDS):
        problems.append("the report's opening differs from E6_OPENING.json")
    e6 = [o for o in (records["openings"].get("openings") or []) if o.get("experiment") == "E6"]
    if len(e6) != 1 or any(e6[0].get(k) != entry.get(k) for k in ENTRY_FIELDS):
        problems.append("OPENINGS.json does not hold exactly this E6 opening")
    if report.get("openings_before") != opening.get("openings_before"):
        problems.append("openings_before differs between the report and E6_OPENING.json")
    if not (records["lock_sha256"] == (report.get("lock") or {}).get("sha256") == entry.get("lock_sha256")):
        problems.append("the lock on disk is not the lock of the opening")
    locked_code = ((records["lock"].get("verify") or {}).get(f"code:{E6.CODE_REL}") or {}).get("sha256")
    if ((report.get("run") or {}).get("code_sha256") or {}).get(E6.CODE_REL) != locked_code:
        problems.append("the code the test step ran is not the locked code")
    checks = report.get("pre_exposure_checks") or {}
    if checks.get("lock") != "held" or not (checks.get("reproduction") or {}).get("equal"):
        problems.append("the pre-opening checks did not all hold")
    result = report["result"]
    if result["decision"] != E6.decide(result["primary"]["interval"]):
        problems.append("the recorded decision does not follow the primary interval")
    if result["decisions_by_interval"].get("match_bootstrap") != E6.decide(result["bootstrap"]["interval"]):
        problems.append("the bootstrap reading does not follow its interval")
    if result["twoway"].get("available") and result["decisions_by_interval"].get("twoway") != E6.decide(result["twoway"]["interval"]):
        problems.append("the two-way reading does not follow its interval")
    if problems:
        raise ValueError("; ".join(problems))
    return {"entries": len(e6), "lock_sha256": records["lock_sha256"]}


# ----------------------------------------------------------------------------- ledger rows
def _spec(report: Mapping[str, Any]) -> list[tuple[str, str, str, int, str]]:
    """(key, locator, form, digits, note) for every e6_ row (PRESPEC_E6 section 12)."""

    result = report["result"]
    two = bool(result["twoway"].get("available"))
    decision = result["decision"]
    primary_method = TWOWAY if two else BOOT
    branch = result["branch"]["branch"]
    rows: list[tuple[str, str, str, int, str]] = []
    base = f"E6 theta = Brier(P1) − Brier(PT_flex), match-weighted on the non-tie rows; {SIGN}; P1 = LightGBM full input " \
           f"(LGBM_S1_A1_M1_E1_C1_R1_D1); {SCOPE}; {STATUS}"
    rows += [("e6_p1", "result/primary/point", "num", 5, f"point: {base}; decision {decision} under the primary interval"),
             ("e6_p1_lo", "result/primary/interval/0", "num", 5,
              f"lower bound of the primary interval (branch {branch}): {primary_method}; {base}"),
             ("e6_p1_hi", "result/primary/interval/1", "num", 5,
              f"upper bound of the primary interval (branch {branch}): {primary_method}; {base}")]
    if two:
        rows += [("e6_p1_boot_lo", "result/bootstrap/interval/0", "num", 5,
                  f"lower bound of the second interval reported with the primary (branch {branch}): {BOOT}; {base}"),
                 ("e6_p1_boot_hi", "result/bootstrap/interval/1", "num", 5,
                  f"upper bound of the second interval reported with the primary (branch {branch}): {BOOT}; {base}"),
                 ("e6_p1_de", "result/twoway/design_effect", "num", 3,
                  f"design effect sqrt(V / V1) of the primary two-way interval against match-only; {SIGN}; {STATUS}"),
                 ("e6_players", "result/twoway/players/players", "count", 0,
                  f"identified players in the matches with non-tie rows (two-way interval, branch {branch}); {STATUS}")]
    rows += [("e6_brier_p1", f"result/cells/{E6.P1}/brier", "num", 5,
              f"match-weighted Brier of the P1 cell (LightGBM full input) on the non-tie rows, lower is better; {SCOPE}; {STATUS}"),
             ("e6_brier_pt", f"result/cells/{E6.PT}/brier", "num", 5,
              f"match-weighted Brier of the PT_flex cell on the non-tie rows, lower is better; {SCOPE}; {STATUS}"),
             ("e6_selected", "result/selected", "count", 0, f"matches selected for the slice (PRESPEC_E6 section 2.2); {STATUS}"),
             ("e6_q_rows", "result/scale/q_rows", "count", 0, f"E6 q rows (valid h90 engagements of the slice); {STATUS}"),
             ("e6_rows", "result/scale/non_tie_rows", "count", 0, f"non-tie rows evaluated (KA not 0); {STATUS}"),
             ("e6_ties", "result/scale/ties", "count", 0, f"tie rows (KA = 0), left out of every denominator; {STATUS}"),
             ("e6_matches", "result/scale/matches", "count", 0, f"matches with non-tie rows (M); {STATUS}")]
    a = result["secondary"]["a_position_share"]
    if a.get("scored"):
        s_base = f"secondary (a) LightGBM position share; {S_SIGN}; {SECONDARY}"
        s_primary = "twoway" if two else "bootstrap"
        rows += [("e6_s_meand", "result/secondary/a_position_share/point", "num", 5, f"point: {s_base}"),
                 ("e6_s_meand_lo", f"result/secondary/a_position_share/{s_primary}/interval/0", "num", 5,
                  f"lower bound, interval of the primary branch ({primary_method}): {s_base}"),
                 ("e6_s_meand_hi", f"result/secondary/a_position_share/{s_primary}/interval/1", "num", 5,
                  f"upper bound, interval of the primary branch ({primary_method}): {s_base}")]
        if two:
            rows += [("e6_s_meand_boot_lo", "result/secondary/a_position_share/bootstrap/interval/0", "num", 5,
                      f"lower bound, second interval ({BOOT}): {s_base}"),
                     ("e6_s_meand_boot_hi", "result/secondary/a_position_share/bootstrap/interval/1", "num", 5,
                      f"upper bound, second interval ({BOOT}): {s_base}")]
    for cell in E6.CELLS:
        if cell in result["secondary"]["b_auc"]:
            rows.append((f"e6_auc_{CELL_TAGS[cell]}", f"result/secondary/b_auc/{cell}", "num", 4,
                         f"secondary (b) match-weighted AUC of {cell} on the non-tie rows, point only, no decision; {SECONDARY}"))
    for stratum, tag, _ in STRATA:
        block = result["secondary"]["c_kill_strata"]["strata"][stratum]
        loc = f"result/secondary/c_kill_strata/strata/{stratum}"
        k_base = f"kill-count stratum {stratum} kills; {SIGN}; {STRATA_BOOT}; {STRATA_STATUS}"
        if block.get("p1") is not None:
            rows += [(f"e6_{tag}_p1", f"{loc}/p1/point", "num", 5, f"point: {k_base}"),
                     (f"e6_{tag}_p1_lo", f"{loc}/p1/interval/0", "num", 5, f"lower 95% bound: {k_base}"),
                     (f"e6_{tag}_p1_hi", f"{loc}/p1/interval/1", "num", 5, f"upper 95% bound: {k_base}")]
        rows += [(f"e6_{tag}_rows", f"{loc}/rows", "count", 0, f"non-tie rows: kill-count stratum {stratum}; {STRATA_STATUS}"),
                 (f"e6_{tag}_ties", f"{loc}/ties", "count", 0,
                  f"tie rows in the stratum (not in any denominator): kill-count stratum {stratum}; {STRATA_STATUS}")]
    return rows


def ledger_rows(records: Mapping[str, Any]) -> list[dict[str, str]]:
    """The e6_ rows with printed values taken from E6_REPORT.json (each must round-trip the thesis verifier)."""

    report = records["report"]
    source = str(Path(records["paths"].report_json))
    out = []
    for key, locator, form, digits, note in _spec(report):
        if "abs" in note:
            raise ValueError(f"{key}: a note must not hold the magnitude-only marker")
        value = float(get(report, locator))
        printed = count(value) if form == "count" else num(value, digits)
        out.append({"key": key, "printed": printed, "kind": "json", "source": source, "locator": locator, "scale": "1",
                    "dataset": DATASET, "note": note})
    return out


def append_rows(ledger: Path, rows: Sequence[Mapping[str, str]]) -> None:
    """Append the rows once (LF endings, the ledger's header order); an existing e6_ key refuses the whole append."""

    ledger = Path(ledger)
    with ledger.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != LEDGER_COLUMNS:
            raise ValueError(f"{ledger} does not have the header {','.join(LEDGER_COLUMNS)}")
        present = {row["key"] for row in reader}
    clash = sorted(present & {row["key"] for row in rows}) + sorted(key for key in present if key.startswith("e6_"))
    if clash:
        raise FileExistsError(f"{ledger} already holds E6 rows: {clash[:5]}")
    data = ledger.read_bytes()
    if not data.endswith(b"\n") or b"\r" in data:
        raise ValueError(f"{ledger} must use LF line endings and end with a newline")
    for row in rows:
        if any(ch in value for value in row.values() for ch in (chr(9), chr(13))):
            raise ValueError(f"{row['key']}: a TAB or CR character")
    with ledger.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        for row in rows:
            writer.writerow([row[name] for name in LEDGER_COLUMNS])


# ----------------------------------------------------------------------------- the Korean report
def _decision_sentence(decision: str) -> str:
    if decision == "confirmed":
        return ("주장 1(교전 직전 상세 상태가 판세보다 교전 결과의 방향을 더 잘 예측한다)을, 사전 등록한 새 자료에서 측정기 표류가 "
                "라벨에 들어가지 않는 라벨(승리 확률 모델을 쓰지 않는 킬 우위 라벨)로 확증했다.")
    if decision == "unresolved":
        return ("새 자료에서는 방향을 확인하지 못했다(미결). 주장 1의 킬 우위 근거는 사후 분석으로 남고, 원고 문구는 저자가 정해 "
                "DECISION_LOG에 적는다.")
    return ("새 자료에서 판세 기준의 Brier가 더 낮았다(반대 방향). 투고 전에 주장 1을 저자와 다시 정한다(DECISION_LOG).")


def _reading(decision: str) -> str:
    return {"confirmed": "상한 < 0", "unresolved": "구간이 0을 포함", "opposite": "하한 > 0"}[decision]


def render(records: Mapping[str, Any]) -> str:
    report, lock = records["report"], records["lock"]
    paths = records["paths"]
    res = report["result"]
    entry = report["opening"]
    two = bool(res["twoway"].get("available"))
    decision = res["decision"]
    word = DECISION_WORD[decision]
    branch = res["branch"]
    inc = branch.get("incomplete_identification") or {}
    scale = res["scale"]
    cells = res["cells"]
    sec = res["secondary"]
    a = sec["a_position_share"]
    out_dir = Path(paths.out)
    lines: list[str] = []
    add = lines.append

    add("# E6 결과 보고서: 처리한 적 없는 외부 KR 15.18–15.22 조각에서 킬 우위 라벨로 한 사전 등록 확증 검정")
    add("")
    add("- 작성 2026-10-14 KST, 구현자(Opus, E6 Part B). 이 문서와 원장 `e6_*` 행은 `studies/tog_experiments_20261012/e6_report.py`가 "
        "`E6_REPORT.json`과 공개 기록(열람 기록, 잠금, `OPENINGS.json`, 시험 단계 로그)만 읽어 썼다. 잠긴 `e6_run.py`는 바꾸지 "
        "않았다(Ruling 60).")
    add("- 사전 등록: `reports/tog_experiments_20261012/PRESPEC_E6_KO.md`(승인본 d11d63c, 승인 줄 c4d4ee8, 2026-10-14 운영 정정 "
        "Ruling 69 86e1541). 판정 규칙의 원본은 그 문서 §6이다.")
    add(f"- 결과 기록: `{win(str(out_dir), 'E6_REPORT.json')}`(SHA-256 `{records['report_sha256']}`), 열람 기록 "
        f"`{win(str(out_dir), 'E6_OPENING.json')}`(SHA-256 `{records['opening_sha256']}`), 잠금 `{records['lock_path']}`"
        f"(SHA-256 `{records['lock_sha256']}`).")
    add("- 수치는 원장 `reports/tog_paper_20261011/ledger/TOG_NUMBERS.csv`의 `e6_*` 행과 같다(같은 기록에서 같은 반올림).")
    add("")
    add("## 1. 판정")
    add("")
    add(f"- 판정: **{word}** ({_reading(decision)}; PRESPEC §6, 주 구간 하나로 판정).")
    add(f"- 주 양 θ = Brier(P1) − Brier(PT_flex): 동률 아닌 행의 경기 가중 Brier 차, 음수면 P1(LightGBM)이 낫다. "
        f"θ̂ = {num(res['primary']['point'], 5)}.")
    if two:
        add(f"- 주 구간(갈래 (가), 선수·경기 이중 군집 95%, θ̂ ± 1.96√V, V = max(V₁, V_U)): {ci(res['twoway']['interval'], 5)}.")
        add(f"- 함께 보고하는 구간(경기 짝 부트스트랩 95%, 2,000회, 시드 20261011): {ci(res['bootstrap']['interval'], 5)}. "
            f"이 구간의 읽기도 {DECISION_WORD[res['decisions_by_interval']['match_bootstrap']]}"
            f"({_reading(res['decisions_by_interval']['match_bootstrap'])})이다"
            + ("." if not res["intervals_disagree"] else "; 두 구간의 읽기가 달라 주 구간이 정한다(§6)."))
        tw = res["twoway"]
        add(f"- 설계 효과 √(V/V₁) = {num(tw['design_effect'], 3)}. 보수적 변형 V₂로 잡은 구간 "
            f"{ci(tw['conservative_V2']['interval'], 5)}(설계 효과 {num(tw['conservative_V2']['design_effect'], 3)})은 기록만 한다.")
    else:
        add(f"- 주 구간(갈래 ({branch['branch']}), 경기 짝 부트스트랩 95%, 2,000회, 시드 20261011): "
            f"{ci(res['bootstrap']['interval'], 5)}. 이중 군집 구간은 계산하지 않았다(사유 "
            f"`{branch.get('twoway_unavailable_reason')}`, §5.3).")
    add(f"- 셀의 경기 가중 Brier: P1 {num(cells[E6.P1]['brier'], 5)}, PT_flex {num(cells[E6.PT]['brier'], 5)}. "
        f"AUC: {num(cells[E6.P1]['auc'], 4)} 대 {num(cells[E6.PT]['auc'], 4)}(보조 (b)).")
    add(f"- 문구(§6): {_decision_sentence(decision)}")
    add("- 범위: 외부 KR 15.18–15.22(2025 시즌, 한 지역), 이 계보가 처리하지 않은 경기, 동률 제외 행, 15.14에서 KA로 다시 학습한 "
        "q_ka_1의 P1·PT_flex 셀. 일반화는 이 범위까지다.")
    add("- 읽을 때 지킬 것(§6 문구 규칙)")
    add("  - \"승리 확률 모델을 쓰지 않음\"은 라벨에만 해당한다. PT_flex의 입력 p_pre는 동결 V의 교전 전 확률이고, 팀 킬 수는 V와 "
        "q의 입력이기도 하다.")
    add("  - E6는 15.16 사후 결과를 다시 보인 실험이 아니다. E6의 θ̂와 15.16 사후 추정(−0.0200, 원장 `kap_test_p1`)의 차이를 라벨·"
        "시점·패치의 효과로 읽지 않는다(자료와 경기가 다름). E6가 확인하는 것은 방향이지 15.16 크기가 아니다(§1).")
    add("  - 구간 수준은 95%다. 97.5% 구간과 폭을 견주지 않는다.")
    add("  - \"확증\"과 \"사전 등록 확증 검정\"은 E6 판정에만 쓴다. 15.15–15.16의 킬 우위 결과에는 계속 \"사후\"를 붙인다.")
    add("")
    add("## 2. 보조 질문 (사전 등록, 주 판정과 별도, 다중 비교 보정 없음)")
    add("")
    add("### (a) LightGBM 위치 몫 S")
    add("")
    if a.get("scored"):
        add("- S = ½[Brier(S1D0) − Brier(S0D0)] + ½[Brier(S1D1) − Brier(S0D1)]. 음수면 위치를 켠 쪽이 낫다. 네 S 셀 모두 Part A의 "
            "재적재 점검과 채점을 통과했다.")
        if two:
            add(f"- Ŝ = {num(a['point'], 5)}. 주 구간과 같은 갈래(이중 군집 95%): {ci(a['twoway']['interval'], 5)}. 경기 부트스트랩 "
                f"95%: {ci(a['bootstrap']['interval'], 5)}.")
        else:
            add(f"- Ŝ = {num(a['point'], 5)}. 경기 부트스트랩 95%: {ci(a['bootstrap']['interval'], 5)}.")
        reading = a["reading"]
        text = {"confirmed": "새 자료에서도 위치를 켠 LightGBM의 Brier가 낮았다(사전 등록 보조).",
                "unresolved": "구간이 0을 포함해 미결이다(사전 등록 보조).",
                "opposite": "새 자료에서 위치를 켠 쪽의 Brier가 더 높았다(반대 방향, 사전 등록 보조)."}[reading]
        add(f"- 읽기(§7 (a), {_reading(reading)}): {text} 위치 켬·끔은 위치 입력이 쓰이는지 확인한 비교로만 쓴다.")
    else:
        add(f"- 채점 불가: {a.get('reason')}")
    add("")
    add("### (b) AUC (경기 가중, 점만, 판정 없음)")
    add("")
    add("| 셀 | AUC | Brier |")
    add("|---|---|---|")
    for cell in E6.CELLS:
        if cell in sec["b_auc"]:
            add(f"| {CELL_NAMES[cell]} | {num(sec['b_auc'][cell], 4)} | {num(cells[cell]['brier'], 5)} |")
    add("")
    add("### (c) 킬 수 층 (킬 기록으로 고른 층, 기술만)")
    add("")
    add("- 층은 결과 구간의 킬 기록(`eval_n_kills`)으로 고른 것이다. 예측 시점 규모의 효과로 읽지 않는다. 2–3킬·4+킬 층은 동률"
        "(맞교환)이 빠진 모집단이다. 층마다 그 층의 경기만 따로 재표집했다(경기 짝 부트스트랩 2,000회, 시드 20261011, 95%).")
    add("")
    add("| 층 | 동률 아닌 행 | 경기 | 동률 행 | P1 − PT_flex [95%] | Brier P1 / PT_flex | AUC P1 / PT_flex |")
    add("|---|---|---|---|---|---|---|")
    for stratum, _, label in STRATA:
        b = sec["c_kill_strata"]["strata"][stratum]
        if b.get("p1") is None:
            add(f"| {label} | {count(b['rows'])} | {count(b['matches'])} | {count(b['ties'])} | 없음 | 없음 | 없음 |")
            continue
        add(f"| {label} | {count(b['rows'])} | {count(b['matches'])} | {count(b['ties'])} | {signed(b['p1']['point'], 5)} "
            f"{ci(b['p1']['interval'], 5, plus=True)} | {num(b['brier'][E6.P1], 5)} / {num(b['brier'][E6.PT], 5)} | "
            f"{num(b['auc'][E6.P1], 4)} / {num(b['auc'][E6.PT], 4)} |")
    add("")
    positive = [label for stratum, _, label in STRATA
                if sec["c_kill_strata"]["strata"][stratum].get("p1") and sec["c_kill_strata"]["strata"][stratum]["p1"]["interval"][0] > 0]
    if positive:
        add(f"- {', '.join(positive)} 층에서는 P1 대비 구간이 0 위에 있다. 그 층에서는 PT_flex의 Brier가 더 낮았다. 킬 기록으로 고른 "
            "층의 기술이며 판정이 아니다.")
        add("")
    add("## 3. 주 구간 갈래 (§5.3)")
    add("")
    if two:
        players = res["twoway"]["players"]
        add(f"- 갈래 (가): 선수 행렬을 만들었고, 불완전 식별 경기 {count(inc.get('incomplete_matches', 0))} / q 행 경기 "
            f"{count(inc.get('q_row_matches', 0))}(문턱 1%). 주 구간은 이중 군집, 함께 보고하는 구간은 경기 부트스트랩이다. 갈래는 Part A"
            "(A7)에서 어떤 라벨도 읽기 전에 정했다.")
        add(f"- 동률 아닌 행이 있는 경기 {count(res['matches'])}개의 식별 선수 {count(players['players'])}명(식별 칸 "
            f"{count(players['identified_slots'])}, 미식별 칸 {count(players.get('unidentified_slots', 0))}). 공유 선수 쌍 "
            f"{count(res['twoway']['pairs_c_ge_1'])}, 그 가운데 c ≥ 2 비율 {num(100 * res['twoway']['share_c_ge_2'], 2)}%. "
            f"교차항 부호 {'+' if res['twoway']['cross_U_sign'] > 0 else MINUS}.")
    else:
        add(f"- 갈래 ({branch['branch']}): 경기 부트스트랩이 주 구간이고 유일한 구간이다(사유 `{branch.get('twoway_unavailable_reason')}`).")
    add("")
    add("## 4. 규모")
    add("")
    counts = res.get("stage_counts") or {}
    build = counts.get("build") or {}
    audit = counts.get("audit") or {}
    bank = counts.get("bank") or {}
    e2c = counts.get("e2c") or {}
    add(f"- 선정 {count(res['selected'])}경기(패치별 7,514 / 10,075 / 9,738 / 8,271 / 4,402). 빌드 {count(build.get('built', 0))}, "
        f"빌드 실패 {count(build.get('failed', 0))}; 감사 {count(audit.get('audited', 0))}, 감사 탈락 "
        f"{count(audit.get('audit_fail', 0))}; 은행 {count(bank.get('bank_matches', 0))}, 은행 제외 {count(bank.get('excluded', 0))}.")
    add(f"- q 행 {count(scale['q_rows'])}(q 행 경기 {count(e2c.get('q_row_matches', inc.get('q_row_matches', 0)))}), 동률 "
        f"{count(scale['ties'])}, 동률 아닌 행 {count(scale['non_tie_rows'])}, 경기 M {count(scale['matches'])}.")
    half = (res["primary"]["interval"][1] - res["primary"]["interval"][0]) / 2
    add(f"- 설계 참고(§5.4, 판정에 안 씀): 예상 동률 아닌 행 약 13만 2천, M 약 38,000, 주 구간 반폭 약 0.0008. 실제 주 구간 반폭 "
        f"{num(half, 6)}.")
    add("")
    add("## 5. 노출과 열람 기록")
    add("")
    add("- 이 열람 직전의 `OPENINGS.json` 목록(`E6_OPENING.json` `openings_before`):")
    for o in report["openings_before"]:
        attempt = "" if o.get("attempt") is None else f" attempt {o['attempt']}"
        add(f"  - {o.get('experiment')}{attempt}, {zulu(o['utc'])}")
    add("- PRESPEC §2.4가 적은 이 검정과 상관된 노출: 같은 라벨의 15.15 Q_DEV·15.16 TEST 결과(2026-10-09, KA 사후 2026-10-13), "
        "같은 패치 다른 경기의 외부 4만 SVI 라벨 두 번(2026-10-09, E3 2026-10-12), 구 계보 v4의 같은 경기 다른 라벨 채점"
        "(2026-09-28/29), INFRA의 파일 수 세기(내용 읽지 않음).")
    add(f"- E6 열람 항목: experiment E6, attempt {entry['attempt']}, {zulu(entry['utc'])}, 잠금 `{entry['lock']}`(SHA-256 "
        f"`{entry['lock_sha256']}`), PRESPEC blob SHA-256 `{entry['prespec_sha256']}`, 라벨 `{entry['labels'][0]}`(SHA-256 "
        f"`{entry['sealed_sha256']}`).")
    logs = records["logs"]
    stdout_text = "고정 문장 한 줄(\"E6 TEST written (attempt 1)\")만" if logs["stdout"] == FIXED_STDOUT \
        else f"{len((logs['stdout'] or '').encode('utf-8'))}바이트(고정 문장과 다름)"
    stderr_text = "0바이트" if logs["stderr_bytes"] == 0 else f"{logs['stderr_bytes']}바이트"
    captured = report.get("captured_output_after_opening") or ""
    add(f"- 노출 상태: 라벨은 열람 항목을 디스크에 저장한 뒤 한 번 읽었다. 라벨에서 나온 값은 계획된 결과 파일 `E6_REPORT.json`에만 "
        f"들어갔다. 라벨을 읽은 뒤의 Python 출력 포착 버퍼는 {'비어 있다' if not captured else '비어 있지 않다'}. 표준 출력 파일"
        f"(`{logs['stdout_path']}`)에는 {stdout_text} 있고, 표준 오류 파일(`{logs['stderr_path']}`)은 {stderr_text}다. 종료 기록: "
        f"`{logs['exit']}`.")
    after = records["after_opening_files"]
    add(f"- 실패 기록·TEST_BLOCKED·SEALED 상세 파일: {'없음' if not after else ', '.join(after)}. 열지 않은 두 파일: "
        "`EXTERNAL_LABELS_SEALED_kr_e6.npz`(고정 조립이 쓴 SVI 봉인 라벨), `PRED_final`과 그 조각.")
    checks = report["pre_exposure_checks"]
    held = "유지(held)" if checks["lock"] == "held" else str(checks["lock"])
    add(f"- 열람 전 점검(B0): 잠금 {held}, 라벨 없는 입력 점검 통과(q 행 {count(checks['inputs']['rows'])}), 두 예행을 지금 "
        f"코드로 다시 계산한 결과가 잠금에 묶인 파일과 같음(Q_DEV 예행은 KA_REPORT와도 같음), 잠금 밖 모듈 없음.")
    add(f"- 시험 단계 시간: 시작 {zulu(report['started_utc'])}, 끝 {zulu(report['finished_utc'])}, 벽시계 "
        f"{num(report['wall_seconds'], 1)}초, 최대 메모리 {num(report['peak_memory_bytes'] / 1e9, 2)} GB. 한 번의 호출이었고, 그 동안 "
        "`OPENINGS.json`의 유일한 작성자였다(`OPENINGS.json`에는 다중 프로세스 잠금이 없다).")
    add("")
    add("## 6. 사건과 PRESPEC에서 벗어난 점")
    add("")
    prev = lock.get("previous_locks") or []
    add("- 2026-10-14 운영 정정(Ruling 69, PRESPEC 머리와 §4에 적음): score-v 환경 허용 목록에 USERNAME을 더했다(동결 V 경로의 "
        "`getpass.getuser()`가 승인 환경에서 실패). 잠금 전에 죽은 단계 시도는 새 시도로 다시 돌릴 수 있게 했다(실제 데이터 단계의 "
        "재시도는 없었다).")
    lock_lines = ", ".join(f"`{Path(p['path']).name}`(SHA-256 `{p['sha256'][:12]}…`)" for p in prev)
    add(f"- 잠금 판: {lock_lines}는 어떤 열람도 하기 전에 바뀌었다(`previous_locks`의 opened false). 바꾼 까닭은 시험 단계 코드의 "
        f"고침뿐이다(Astra 검토 고침 1차: 열람 뒤 모든 단계의 실패 처리와 노출 기록, 고침 2차: 기록기 전체 보호와 표준 출력·오류 "
        f"파일 증가의 파일 노출 반영). 자료와 라벨은 그대로이고 데이터 단계를 다시 돌리지 않았다. 열람에 쓴 잠금은 "
        f"`{records['lock_path'].name}`이다.")
    add("- 잠금 전 코드 고침: 첫 잠금 시도는 잠금 파일을 쓰지 않고 BLOCKED로 멈췄다(단계 사이 e6_run.py 판 비교, 커밋 ad0bbed로 "
        "단계 시도 안으로 고침). 단계마다의 코드 판은 단계가 시작할 때의 커밋으로 적는다(커밋 fe7e367). bank 단계는 81fbb37로 "
        "돌았고, 그 모듈 기록에는 단계 끝 파일의 86e1541 해시가 들어 있다(`edited_during_stage`).")
    add("- 운영 시험: Part A 전에 이미 처리·노출된 기존 외부 표본 25경기로 세션 임시 폴더에서 절차 전체를 시험했다. 어떤 라벨도 "
        "읽지 않았다.")
    add("- 남겨 둔 검토 항목(코드 바꾸지 않음): `OPENINGS.json`의 다중 프로세스 잠금 없음(Part B는 한 번의 호출, 유일한 작성자), "
        "기본 폴더가 아닌 곳의 재시도 문구, 단독 score-v 호출의 환경, 가중 불일치를 거부하지 않고 기록함, 회귀 시험 범위의 작은 "
        "공백(SEALED 쓰기 실패 주입, 표준 오류 증가, 예외 경로의 표준 출력 증가 각각의 전용 시험 없음; 공용 구현이 같은 경로를 덮음).")
    add("- 감사 상태: 40,000경기 모두 'unknown'(외부 경로의 고정 챔피언 지도가 15.14–15.16만 다룸; 치명 아님, 탈락 0).")
    add(f"- §8.2 검토 비율(판정 조건 아님): q 행 경기 / 선정 = {num(100 * inc.get('q_row_matches', 0) / res['selected'], 2)}%"
        f"(기존 4만 96.1%), q 행 경기당 행 {num(scale['q_rows'] / max(inc.get('q_row_matches', 1), 1), 2)}(기존 3.82).")
    add("")
    add("## 7. 남는 한계 (§2.5)")
    add("")
    add("- \"아무도 본 적 없는 자료\"(등급 0)가 아니라 \"이 라벨로, 이 계보에서 처음 보는 자료\"(등급 1)다.")
    add("- 같은 패치·같은 수집 틀의 다른 경기에서 SVI 결과를 봤고, 같은 라벨의 결과는 학습 시즌의 다른 패치에서 봤다.")
    add("- 새 조각과 기존 4만은 같은 수집 틀이라 선수가 겹친다. 설계에 영향을 준 앞선 결과와 E6 결과를 서로 독립이라고 할 수 없다. "
        "이중 군집 구간은 E6 안의 선수 의존만 다루고, 앞선 결과와 E6 사이의 의존은 다루지 않는다.")
    add("")
    return "\n".join(lines)


def write_report(target: Path, text: str) -> Path:
    """E6_REPORT_KO.md written once (UTF-8, LF); a text holding a TAB or CR character is refused."""

    target = Path(target)
    bad = sorted({repr(ch) for ch in text if ch in (chr(9), chr(13))})
    if bad:
        raise ValueError(f"the report holds control characters {bad}")
    if target.exists():
        raise FileExistsError(f"{target} exists; it is written once")
    target.write_bytes(text.encode("utf-8"))
    return target


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0]).parse_args(argv)
    paths = Paths()
    records = load(paths)
    check(records)
    rows = ledger_rows(records)
    text = render(records)
    target = write_report(paths.report_md, text)
    append_rows(paths.ledger, rows)
    print(json.dumps({"report": str(target), "sha256": E6.sha256_file(target), "ledger_rows": len(rows)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
