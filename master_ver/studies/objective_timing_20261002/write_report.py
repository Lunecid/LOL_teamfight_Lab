"""Korean report for objective timing diagnostic."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from . import config as C
from .io_util import file_sha256, load_json, write_json


def _fmt(x: Any, nd: int = 3) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "NA"
    if v != v:
        return "NA"
    return f"{v:.{nd}f}"


def _row(agg_cat: dict[str, Any]) -> list[str]:
    d = agg_cat.get("delays", {})
    ew = d.get("event_weighted", {})
    mw = d.get("equal_match_weighted", {})
    # Conditional CDF among observed delays
    prop = ew.get("prop_le_conditional_on_observed") or ew.get("prop_le", {})
    # Cohort-denominator threshold coverage
    cov = (agg_cat.get("threshold_coverage_cohort") or {}).get("event_weighted", {})
    return [
        str(agg_cat.get("n_candidates", 0)),
        str(agg_cat.get("n_observed_acquisition", 0)),
        _fmt(agg_cat.get("coverage_observed"), 4),
        str(agg_cat.get("n_no_acquisition", 0)),
        str(agg_cat.get("n_unknown_observation", 0)),
        str(agg_cat.get("n_neutral_in_interval", 0)),
        str(agg_cat.get("n_no_interval", 0)),
        _fmt(ew.get("mean_s")),
        _fmt(ew.get("median_s")),
        _fmt(mw.get("mean_s")),
        _fmt(mw.get("median_s")),
        _fmt(cov.get("30"), 4),
        _fmt(cov.get("60"), 4),
        _fmt(cov.get("90"), 4),
        _fmt(cov.get("120"), 4),
        _fmt(cov.get("180"), 4),
        _fmt(prop.get("30"), 4),
        _fmt(prop.get("60"), 4),
        _fmt(prop.get("90"), 4),
    ]


def write_report(manifest: dict[str, Any], report_dir: Path | None = None) -> Path:
    report_dir = report_dir or C.report_dir()
    report_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(manifest["run_dir"])
    aggregates = load_json(run_dir / "aggregates.json")
    census = load_json(run_dir / "schema_census.json")
    reload = load_json(run_dir / "reload_verification.json")
    gates = manifest.get("gates", {})

    isol = aggregates["cohorts"][C.COHORT_FULL_ISOL]
    isol_size = aggregates["cohort_sizes"][C.COHORT_FULL_ISOL]

    def block(interval: str, cat: str) -> dict[str, Any]:
        return isol[interval][cat]

    lines: list[str] = [
        "# 교전 마지막 킬 → 오브젝트 획득 지연 (TRAIN 15.14 기술통계)",
        "",
        "## 상태",
        "",
        f"- complete: `{manifest.get('complete')}`",
        f"- run_dir: `{manifest.get('run_dir')}`",
        f"- gates.all_ok: `{gates.get('all_ok')}`",
        f"- reload.all_ok: `{reload.get('all_ok')}`",
        f"- n_queries: {manifest.get('n_queries')} / query_wall_s: {_fmt(manifest.get('query_wall_s'), 2)}",
        "",
        "## 질문과 답",
        "",
        "사용자 질문: 참여 반경 진단 방향과 기존 전역 격리 지표를 유지한 채, "
        "**최종 멤버 킬(L)에서 첫 확정 팀 오브젝트 획득까지** 지연의 평균·중앙값은 얼마인가?",
        "",
        "주 코호트: inclusive δ∈[10,000,20,000]ms **그리고** "
        "`foreign_kills_in_tau_L==0 AND other_anchors_in_tau_L==0` "
        f"(컴포넌트 {isol_size['n_components']:,}, 매치 {isol_size['n_matches']:,}). "
        "valid_h90 게이트는 적용하지 않았다. 참여 반경은 미결정이며 본 결과는 인과 전환 시간이 아니다.",
        "",
        "### 주표 — 격리 δ10–20 코호트, 미제한(GAME_END까지)",
        "",
        "평균·중앙값·조건부 CDF는 **확정 획득(팀 100/200)이 관측된 행에 조건부**다. "
        "≤30/60/90/120/180 **코호트 커버리지**는 전체 적격 후보 분모다 "
        "(무획득·불확실·구간없음은 분모에 포함, 분자에서 제외). "
        "명시적 중립(killerTeamId=300)은 팀 신용 부재이며 불확실 관측이 아니다.",
        "",
        "| 범주 | 후보 | 확정획득 | 획득비율 | 무획득 | 불확실 | 중립구간 | 구간없음 | 평균(s)·사건 | 중앙(s)·사건 | 평균(s)·균등매치 | 중앙(s)·균등매치 | cov≤30 | cov≤60 | cov≤90 | cov≤120 | cov≤180 | 조건부≤30 | 조건부≤60 | 조건부≤90 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    main_cats = (
        "any_objective",
        "any_epic_monster",
        "any_building",
        "any_nonterminal_building",
        "dragon_elemental",
        "dragon_elder",
        "dragon_unknown_subtype",
        "baron",
        "herald",
        "horde",
        "atakhan",
        "tower",
        "inhibitor",
        "building_terminal",
    )
    for cat in main_cats:
        r = _row(block(C.INTERVAL_UNCAPED, cat))
        lines.append("| " + " | ".join([cat] + r) + " |")

    lines.extend(
        [
            "",
            "### 제한 구간 — 다음 킬/앵커 이전 (strict pre-next-fight)",
            "",
            "구간: `L < t_obj < min(next_kill_after_L, next_component_tau, GAME_END)`. "
            "저장 sentinel(-1/-2)은 실시각이 아니다. `next_component_tau<=L`이면 "
            "이미 활성/겹치는 다음 컴포넌트로 **NO_INTERVAL**이며, "
            "다가오는 사건이 없다고 해석하지 않는다.",
            "",
            "| 범주 | 후보 | 확정획득 | 획득비율 | 무획득 | 불확실 | 중립구간 | 구간없음 | 평균(s)·사건 | 중앙(s)·사건 | 평균(s)·균등매치 | 중앙(s)·균등매치 | cov≤30 | cov≤60 | cov≤90 | cov≤120 | cov≤180 | 조건부≤30 | 조건부≤60 | 조건부≤90 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for cat in ("any_objective", "any_epic_monster", "any_building"):
        r = _row(block(C.INTERVAL_PRE_NEXT, cat))
        lines.append("| " + " | ".join([cat] + r) + " |")

    ao = block(C.INTERVAL_UNCAPED, "any_objective")
    lines.extend(
        [
            "",
            "## 핵심 수치 (격리 δ10–20, any_objective, uncapped)",
            "",
            f"- 후보 컴포넌트: **{ao['n_candidates']:,}**",
            f"- 확정 획득 관측: **{ao['n_observed_acquisition']:,}** "
            f"(비율 {_fmt(ao['coverage_observed'], 4)})",
            f"- 무획득: {ao['n_no_acquisition']:,} / 불확실 관측: {ao['n_unknown_observation']:,} "
            f"/ 중립 구간: {ao.get('n_neutral_in_interval', 0):,} "
            f"/ 구간없음(해당 간격): {ao['n_no_interval']:,}",
            f"- 사건가중 평균·중앙(초): **{_fmt(ao['delays']['event_weighted']['mean_s'])}** / "
            f"**{_fmt(ao['delays']['event_weighted']['median_s'])}**",
            f"- 균등매치 평균·중앙(초): **{_fmt(ao['delays']['equal_match_weighted']['mean_s'])}** / "
            f"**{_fmt(ao['delays']['equal_match_weighted']['median_s'])}**",
            f"- 코호트 분모 커버리지 ≤30/60/90/120/180s (사건가중): "
            + " / ".join(
                _fmt(
                    ao.get("threshold_coverage_cohort", {})
                    .get("event_weighted", {})
                    .get(str(t)),
                    4,
                )
                for t in C.DELAY_THRESHOLDS_S
            ),
            f"- 조건부 CDF ≤30/60/90/120/180s (확정획득 관측만): "
            + " / ".join(
                _fmt(
                    (
                        ao["delays"]["event_weighted"].get("prop_le_conditional_on_observed")
                        or ao["delays"]["event_weighted"].get("prop_le", {})
                    ).get(str(t)),
                    4,
                )
                for t in C.DELAY_THRESHOLDS_S
            ),
            f"- 동일 첫 오브젝트 다중 사용: unique_events={ao.get('unique_first_objective_events')}, "
            f"query/unique={_fmt(ao.get('query_count_over_unique_events'))}",
            "",
            "## 획득 증거 센서스 (전체 처리 매치)",
            "",
            f"- 확정(100/200): {census.get('n_acq_confirmed', 0):,}",
            f"- 중립(killerTeamId=300): {census.get('n_acq_neutral', 0):,}",
            f"- 팀 불명: {census.get('n_acq_unknown', 0):,}",
            f"- 모순 증거: {census.get('n_acq_contradictory', 0):,}",
            "",
            "주 타이밍은 **첫 확정 팀 획득**이다. 명시적 중립만 있는 구간은 `NO_ACQ`이고, "
            "불명/모순 증거만 있는 구간은 `UNKNOWN_OBS`다.",
            "",
            "## 오염/겹침 분류 (uncapped any_objective, 전체 컴포넌트)",
            "",
        ]
    )
    cont = census.get("contamination_uncapped_any_objective", {})
    for k, v in sorted(cont.items(), key=lambda kv: (-int(kv[1]), kv[0])):
        lines.append(f"- {k}: {int(v):,}")

    lines.extend(
        [
            "",
            "## 코호트 크기",
            "",
        ]
    )
    for name, sz in aggregates["cohort_sizes"].items():
        lines.append(f"- {name}: components={sz['n_components']:,}, matches={sz['n_matches']:,}")

    lines.extend(
        [
            "",
            "## 해석 한계 (SVI follow-up 선택)",
            "",
            "- 본 값은 기술통계이며 h90/반경을 자동 채택하지 않는다.",
            "- 컴포넌트 가중 평균은 고유 오브젝트 기준 전환 시간과 다르다(다중성 보고).",
            "- 제한 구간은 개입 킬/앵커를 배제한 기술 구간일 뿐 인과 증거가 아니다.",
            "- 이벤트 관측 지지선은 저장 이벤트와 권위 GAME_END이며, 메타 duration/마지막 프레임으로 대체하지 않았다.",
            "",
            "## 재현",
            "",
            f"- aggregates reload primary mean ok: `{reload.get('reload_mean_ok')}`",
            f"- source components sha256: `{manifest.get('source', {}).get('components_sha256')}`",
            f"- anchor manifest sha256: `{manifest.get('source', {}).get('anchor_manifest_sha256')}`",
            "",
        ]
    )

    out = report_dir / "OBJECTIVE_TIMING_REPORT_KO.md"
    text = "\n".join(lines) + "\n"
    out.write_text(text, encoding="utf-8")
    meta = {
        "report_path": str(out),
        "sha256": file_sha256(out),
        "run_dir": str(run_dir),
        "complete": bool(manifest.get("complete")),
    }
    write_json(report_dir / "report_meta.json", meta)
    return out
