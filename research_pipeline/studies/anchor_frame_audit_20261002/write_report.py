"""Write the Korean audit report under reports/feature_control_study_20261001/anchor_frame_audit_20261002/."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import config as C
from .io_source import file_sha256


def write_report(manifest: dict[str, Any], code_audit: dict[str, Any], report_dir: Path | None = None) -> Path:
    report_dir = report_dir or C.report_dir()
    report_dir.mkdir(parents=True, exist_ok=True)
    agg = manifest["aggregates"]
    gates = manifest["gates"]
    storage = manifest.get("storage_estimate", {})
    lines = [
        "# 앵커-프레임 정렬 및 측정 감사 (2026-10-02)",
        "",
        "## 상태",
        "",
        f"- complete: `{manifest.get('complete')}`",
        f"- 채택 설정: `{manifest.get('adopted_cfg')}` (eps_t=14000ms, eps_s=2000, min_samples=1)",
        f"- run_dir: `{manifest.get('run_dir')}`",
        f"- gates.all_ok: `{gates.get('all_ok')}`",
        f"- terminal_crosscheck_ok: `{gates.get('terminal_crosscheck_ok')}`",
        f"- reload_reconcile_ok: `{gates.get('reload_reconcile_ok')}`",
        "",
        "## 센서스 (TRAIN 15.14 관측 캐시)",
        "",
        f"- 컴포넌트: **{agg['n_components']}** (기대 {C.EXPECTED_MIN1_COMPONENTS})",
        f"- 컴포넌트가 있는 매치: {agg['n_matches_with_components']}",
        f"- 제로킬 매치(터미널): {gates.get('zero_kill_matches')} (기대 {C.EXPECTED_ZERO_KILL})",
        f"- 싱글톤 비율: {agg['singleton_share']}",
        f"- span_ms_max: {agg['span_ms_max']} (ST 기대 {C.EXPECTED_SPAN_MAX})",
        f"- diameter_max: {agg['diameter_max']} (ST 기대 ≈{C.EXPECTED_DIAMETER_MAX})",
        "",
        "## Inclusive vs strict 정렬",
        "",
        "Inclusive 계약: τ=f−10000(비클램프), F=max minute_ts≤τ, δ=f−F. "
        "정확(exact)은 δ=10000 및 F=τ(앵커 시각 프레임 포함). "
        "이는 역사적 strict 컷오프 t_in=τ−1과 **호환되지 않음**.",
        "",
        f"- exact δ=10000 컴포넌트: **{agg['exact_10000']['n_components']}** "
        f"(매치 {agg['exact_10000']['n_matches']})",
        f"- strict 적격 컴포넌트: {agg['strict_eligible']['n_components']}",
        f"- exact∧strict: {agg['inclusive_vs_strict']['exact_and_strict']}",
        f"- exact∧¬strict: {agg['inclusive_vs_strict']['exact_not_strict']}",
        "",
        "### 주 요청 지연 구간: 닫힌 구간 [10000, 20000] ms",
        "",
        "첫 킬 시각 − 프레임 시각(δ). 정수 등호, 닫힌 20,000ms 경계. "
        "25,000/30,000ms는 보조 진단.",
        "",
        "| 폭 w (ms) | 누적 구간 | 컴포넌트 | 매치 | 추가 | exact 부분집합 |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for w in C.INCLUSIVE_WIDTHS_MS:
        row = agg["cumulative_inclusive"][f"w_{w}"]
        lines.append(
            f"| {w} | [{C.ANCHOR_OFFSET_MS},{C.ANCHOR_OFFSET_MS + w}] | "
            f"{row['n_components']} | {row['n_matches']} | "
            f"{row.get('added_components')} | {row['n_exact_subset']} |"
        )
    prim = agg.get("primary_lag_window", {})
    lines.extend([
        "",
        f"- 주 구간 [10000,20000] 컴포넌트: **{prim.get('n_components')}** "
        f"(매치 {prim.get('n_matches')})",
        "",
        "### 서로소 δ 밴드 [lo,hi)",
        "",
    ])
    for k, v in agg["disjoint_delta_bands"].items():
        lines.append(f"- {k}: components={v['n_components']}, matches={v['n_matches']}")
    lines.extend(["", "### 상한 갭 보조 진단 (25k/30k)", ""])
    for k, v in agg["upper_gap_diagnostics"].items():
        lines.append(f"- {k}: components={v['n_components']}, matches={v['n_matches']}")

    # Per-window contracted summary for w=10000 (10–20s)
    w20 = agg["cumulative_inclusive"].get("w_10000", {})
    contracted = w20.get("contracted", {})
    lines.extend([
        "",
        "### [10000,20000] 계약 분포 요약 (이벤트 가중; 균등-매치 가중 병기)",
        "",
        f"- 싱글톤 비율(이벤트): {contracted.get('singleton_share_event')}",
        f"- 싱글톤 비율(균등-매치): {contracted.get('singleton_share_equal_match')}",
        f"- 외래 킬 존재 비율(이벤트): {contracted.get('foreign_kill_any_share_event')}",
        f"- 다른 앵커 존재 비율(이벤트): {contracted.get('other_anchor_any_share_event')}",
        f"- 겹침 스팬 비율(이벤트): {contracted.get('overlap_span_any_share_event')}",
        f"- C0/C_all 주의: {contracted.get('note_C0_Call')}",
        f"- C0 scale hist: `{json.dumps(contracted.get('C0_scale_max_side_hist_event', {}), ensure_ascii=False)}`",
        f"- Call scale hist: `{json.dumps(contracted.get('Call_scale_max_side_hist_event', {}), ensure_ascii=False)}`",
    ])
    for h in C.HORIZONS_S:
        hh = contracted.get(f"h{h}", {})
        lines.append(
            f"- h{h}: valid={hh.get('n_valid')}, invalid={hh.get('n_invalid')}, "
            f"full_horizon={hh.get('n_full_horizon')}, followup_p50={hh.get('followup_ms_p50_valid')}"
        )

    term = agg["terminal"]
    lines.extend([
        "",
        "## GAME_END vs 마지막 프레임",
        "",
        f"- GAME_END known: {term['n_game_end_known']} / {term['n_matches']}",
        f"- GAME_END unknown: {term['n_game_end_unknown']} (unknown 유지; last_frame으로 대체하지 않음)",
        f"- multiple GAME_END events: {term.get('n_game_end_multiple_events')}",
        f"- duplicate identical timestamps: {term.get('n_game_end_duplicate_ts')}",
        f"- ambiguous distinct timestamps: {term.get('n_game_end_ambiguous_distinct')}",
        f"- malformed timestamps 포함 매치: {term.get('n_game_end_malformed_any')}",
        f"- terminal 교차검증 불일치: {term.get('n_terminal_crosscheck_mismatch')} (완료 게이트 실패 조건)",
        f"- GE−last_frame 음수: {term.get('n_ge_minus_last_frame_negative')}",
        f"- meta duration ≠ last_frame: {term.get('n_meta_duration_disagrees_last_frame')}",
        f"- meta duration ≠ GAME_END: {term.get('n_meta_duration_disagrees_game_end')}",
        f"- 프레임 무결성 실패: {term.get('n_frame_integrity_fail')} "
        f"(nonmonotone={term.get('n_frame_nonmonotone')}, negative={term.get('n_frame_negative_ts')})",
        f"- missing minute_ts npz: {term['n_missing_npz']}",
        f"- missing events.json: {term['n_missing_events']}",
        "",
        "## Follow-up / isolation 진단 (V/SVI 라벨 없음)",
        "",
        "터미널 지원이 없거나 무효하면 엔드포인트를 권위적으로 valid로 표기하지 않음. 후보는 유지.",
        "",
    ])
    for h in C.HORIZONS_S:
        fu = agg["followup"][f"h{h}"]
        lines.append(
            f"- h={h}s: valid={fu['n_valid']}, invalid={fu['n_invalid']}, "
            f"full_horizon={fu['n_full_horizon']}, followup_p50_ms={fu['followup_ms_p50_valid']}"
        )
    chain = agg.get("chaining_neighbor_summaries", {})
    lines.extend([
        "",
        f"- [τ,L] 내 외래 킬 있는 컴포넌트: {agg['n_foreign_any']}",
        f"- 겹치는 스팬 컴포넌트: {agg['n_overlap_span_any']}",
        f"- 동일 ms 첫 멤버(>1): {agg['n_same_ms_first_gt1']}",
        f"- span>14000: {chain.get('n_span_gt_14000')} / diameter>2000: {chain.get('n_diameter_gt_2000')}",
        f"- 해석: {chain.get('note')}",
        "",
        "## 연결 경계 케이스북",
        "",
        "- 익명 케이스북: 런 디렉터리 `casebook.json` (실제 assignment 멤버·상대시각·좌표·크레딧·이웃 엣지).",
        "- 이웃 반경 2000은 진단용 거리이며 전체 클러스터 직경 상한이 아님.",
        "- 이 단계에서는 차트/이미지 없음.",
        "",
        "## V/OOF 및 베이스라인 입력 감사 (계획 vs 완료 구분)",
        "",
        "### 완료된 진단",
        "",
        "- 구 OOF(`ev4_03b_oof_v.py`)는 전체 TRAIN 하이퍼/NaN-fill/bin을 재사용; 가중치만 OOF이며 "
        "새 nested fold-local 계약(FULL_STUDY_PROTOCOL.md §8)을 구현하지 않음.",
        "- 동결 PT_flex raw: p_pre, time_minutes (code_audit JSON).",
        "- M0는 시간만(time_minutes, time_minutes_sq); PT와 구분.",
        "- 행렬에 p_pre가 보조로 저장될 수 있으나, fit 호출을 추적하지 않고 소비된다고 단정하지 않음.",
        "",
        "### 계획만 있고 이 감사에서 수행하지 않음",
        "",
        "- V/q 학습, 예측, 테스트 스코어링, 새 인구/반경 채택, 피처 행렬 구축, 그림/원고 수정.",
        "- q538/V180/1531 옵티마이저 패스는 계획 예산이며 여기서 재측정하지 않음.",
        f"- nested 신규 OOF 미구현: "
        f"`{code_audit.get('unimplemented_new_nested_oof', code_audit.get('baselines', {}).get('unimplemented_new_nested_oof'))}`",
        "",
        "## 자원",
        "",
        f"- wall_s: {manifest['machine'].get('wall_s')}",
        f"- extraction_wall_s: {manifest['machine'].get('extraction_wall_s')}",
        f"- peak_parent_rss_bytes: {manifest['machine'].get('peak_parent_rss_bytes')}",
        f"- peak_process_tree_rss_bytes: {manifest['machine'].get('peak_process_tree_rss_bytes')}",
        f"- RSS 주의: {manifest['machine'].get('peak_rss_bytes_note')}",
        f"- cpu_logical: {manifest['machine'].get('cpu_logical')}",
        f"- ram_available_bytes: {manifest['machine'].get('ram_available_bytes')}",
        f"- nvidia_smi: {manifest['machine'].get('nvidia_smi')}",
        f"- gpu_used_this_run: {manifest['machine'].get('gpu_used_this_run')}",
        "- 저장 추정은 고유 `(익명 매치 키, query 시각)` 기준; 행렬 미구축. 상세는 RUN_MANIFEST `storage_estimate`.",
        f"- w_10000 unique pre(strict τ−1): "
        f"{(storage.get('windows') or {}).get('w_10000', {}).get('unique_pre_strict_tau_minus_1')}",
        f"- w_10000 unique pre(inclusive τ): "
        f"{(storage.get('windows') or {}).get('w_10000', {}).get('unique_pre_inclusive_tau')}",
        f"- w_10000 unique post h60/90/120 valid: "
        f"{(storage.get('windows') or {}).get('w_10000', {}).get('unique_post_h60_valid')}/"
        f"{(storage.get('windows') or {}).get('w_10000', {}).get('unique_post_h90_valid')}/"
        f"{(storage.get('windows') or {}).get('w_10000', {}).get('unique_post_h120_valid')}",
        f"- w_10000 unique pre/post union: "
        f"{(storage.get('windows') or {}).get('w_10000', {}).get('unique_pre_post_union_all')}",
        "",
        "## 게이트",
        "",
        "```json",
        json.dumps(gates, indent=2, ensure_ascii=False),
        "```",
        "",
        "## 과학적 한계",
        "",
    ])
    for s in manifest.get("scientific_limits", []):
        lines.append(f"- {s}")
    lines.extend([
        "",
        "## 산출물",
        "",
        f"- RUN_MANIFEST: `{manifest.get('run_dir')}/RUN_MANIFEST.json`",
        f"- components.npz sha256: `{manifest.get('components_sha256')}`",
        f"- terminals sha256: `{manifest.get('terminals_sha256')}`",
        f"- code_sha256: `{manifest.get('code_sha256')}`",
        f"- reload_reconcile: `{manifest.get('run_dir')}/reload_reconcile.json`",
        "",
    ])
    report_path = report_dir / "REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    (report_dir / "RUN_MANIFEST_pointer.json").write_text(
        json.dumps({
            "run_dir": manifest.get("run_dir"),
            "complete": manifest.get("complete"),
            "components_sha256": manifest.get("components_sha256"),
            "code_sha256": manifest.get("code_sha256"),
            "report_sha256": file_sha256(report_path),
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    (report_dir / "code_audit.json").write_text(
        json.dumps(code_audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report_path
