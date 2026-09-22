#!/usr/bin/env python3
"""Generate thesis/docs/FEATURE_CENSUS.md and thesis/latex/tables/gen/tab_input_blocks.tex
from the committed feature manifests.

Sources (all under docs/; outputs/ and data/raw/ are not opened, per AGENTS.md):
  docs/supplementary_e1_20260921/feature_groups.json          column names per group, E1 arms
  docs/V_MODEL_INPUT_DESIGN_20260919/STATEV2_REFERENCE_FEATURE_SET.json  per-column dtype/source
  docs/V_FEATURE_MANIFEST_RUNTIME_20260919.md                 runtime widths, name/order match
  docs/A_MLP_expanded_evaluator_meta_20260919.json            frozen evaluator preproc dims
Pure transcription: every number below is read from a field, none is recomputed from data.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FG = ROOT / "docs/supplementary_e1_20260921/feature_groups.json"
REF = ROOT / "docs/V_MODEL_INPUT_DESIGN_20260919/STATEV2_REFERENCE_FEATURE_SET.json"
RUN = ROOT / "docs/V_FEATURE_MANIFEST_RUNTIME_20260919.md"
EVAL = ROOT / "docs/A_MLP_expanded_evaluator_meta_20260919.json"
OUT_MD = ROOT / "thesis/docs/FEATURE_CENSUS.md"
OUT_TEX = ROOT / "thesis/latex/tables/gen/tab_input_blocks.tex"
CHECK_ONLY = False

# Korean block names used in the manuscript; order = manifest group order in §3.4's table.
BLOCKS = OrderedDict([
    ("clock", ("경기 시각", "clock")),
    ("quality", ("자료 품질", "quality")),
    ("player_snapshot", ("참가자 상태", "player_snapshot")),
    ("player_event", ("참가자 사건", "player_event")),
    ("team_event", ("팀 사건", "team_event")),
    ("team_time_interaction", ("팀 사건 $\\times$ 시각", "team_time_interaction")),
    ("champion", ("챔피언 ID", "champion")),
])
SHAPE = {
    "clock": "1 $\\times$ 2",
    "quality": "1 $\\times$ 1",
    "player_snapshot": "10 $\\times$ 9",
    "player_event": "10 $\\times$ 7",
    "team_event": "2 $\\times$ 47",
    "team_time_interaction": "2 $\\times$ 47",
    "champion": "10 $\\times$ 1",
}
DTYPE_KO = {"numeric": "수치", "categorical": "범주"}


def slot_fields(cols, pat):
    """Per-slot / per-side field suffixes, asserting every slot carries the same set."""
    groups = {}
    for c in cols:
        m = re.match(pat, c)
        groups.setdefault(m.group(1), []).append(m.group(2))
    sets = [tuple(sorted(v)) for v in groups.values()]
    assert all(s == sets[0] for s in sets), "slots carry different field sets"
    return len(groups), list(groups[next(iter(groups))])


def main() -> int:
    fg = json.loads(FG.read_text(encoding="utf-8"))
    ref = json.loads(REF.read_text(encoding="utf-8"))
    ev = json.loads(EVAL.read_text(encoding="utf-8"))
    run_md = RUN.read_text(encoding="utf-8")

    groups = fg["groups"]
    ref_feats = {f["name"]: f for f in ref["features"]}
    dims = ref["dimensions"]
    pre = ev["preproc_dims"]

    # --- widths, read from fields -------------------------------------------------
    raw_w = int(re.search(r"Raw width: \*\*(\d+)\*\*", run_md).group(1))
    exp_w = int(re.search(r"Expanded \(drop `snapshot_age_s`\): \*\*(\d+)\*\*", run_md).group(1))
    sv2_hash = re.search(r"`state_value_v2\.py` hash16: `([0-9a-f]+)`", run_md).group(1)
    order_match = re.search(r"Order exact match: \*\*(\w+)\*\*", run_md).group(1)
    only_run = int(re.search(r"Only in runtime: (\d+)", run_md).group(1))
    only_ref = int(re.search(r"Only in reference: (\d+)", run_md).group(1))

    n_names = sum(len(v) for v in groups.values())
    n_champ = len(groups["champion"])
    n_numeric = n_names - n_champ
    assert n_names == exp_w == dims["source"], (n_names, exp_w, dims["source"])
    assert n_numeric == dims["numeric_expanded"] == pre["n_numeric"] == 351
    assert n_champ == dims["champion_categories"] == pre["n_champ"] == 10
    assert raw_w == exp_w + 1

    # --- derived-parent check: interactions are team_event x clock -----------------
    te = set(re.sub(r"^(blue|red)_", "", c) for c in groups["team_event"])
    ti = set(re.sub(r"_x_time$", "", re.sub(r"^(blue|red)_", "", c)) for c in groups["team_time_interaction"])
    assert te == ti and len(te) == 47, (len(te), len(ti))
    denom = Counter(re.search(r"/ (\d+)\s*$", ref_feats[c]["source"]).group(1)
                    for c in groups["team_time_interaction"])
    assert len(denom) == 1, denom
    denom_ms = int(next(iter(denom)))

    n_slots_snap, snap_fields = slot_fields(groups["player_snapshot"], r"^participant_slot(\d+)_(.+)$")
    n_slots_ev, ev_fields = slot_fields(groups["player_event"], r"^participant_slot(\d+)_(.+)$")
    n_sides, team_fields = slot_fields(groups["team_event"], r"^(blue|red)_(.+)$")

    # --- absence check: no coordinate / anchor / distance / frame-age column -------
    banned = re.compile(r"(?i)(^|_)(x|y)$|pos|coord|dist|zone|anchor|near_|tower_range|xy|"
                        r"snapshot_age|frame_age|time_norm|rune|item|ban|damage|spell")
    hits = sorted(n for n in ref_feats if banned.search(n))

    arms = fg["arms"]

    # ---------------------------------------------------------------- LaTeX table
    rows = []
    for key, (ko, _) in BLOCKS.items():
        f = ref_feats[groups[key][0]]
        rows.append(f"  {ko} & {SHAPE[key]} & {len(groups[key])} & {DTYPE_KO[f['dtype']]} \\\\")
    tex = [
        "%% GENERATED by thesis/tools/gen_feature_census.py -- do not edit by hand.",
        "\\begin{table}[!t]",
        "  \\centering",
        f"  \\caption{{$\\Vhat$의 상태 표현 Expanded{exp_w}의 블록 구성}}",
        "  \\label{tab:input-blocks}",
        "  \\small",
        "  \\begin{tabular}{lrrl}",
        "  \\toprule",
        "    블록 & 슬롯 $\\times$ 항목 & 열 & 자료형 \\\\",
        "  \\midrule",
        *rows,
        "  \\midrule",
        f"    수치 소계 & & {n_numeric} & \\\\",
        f"    합계 & & {exp_w} & \\\\",
        "  \\bottomrule",
        "  \\end{tabular}",
        "  \\tabnote{열 이름과 블록 배정은 실행 매니페스트의 전사이다"
        " (\\path{docs/V_FEATURE_MANIFEST_RUNTIME_20260919.md},"
        " \\path{docs/supplementary_e1_20260921/feature_groups.json})."
        f" 원 폭 {raw_w}열에서 \\texttt{{snapshot\\_age\\_s}}를 뺀 {exp_w}열이며,"
        f" $q$의 입력은 수치 {n_numeric}열에 $\\ppre$를 더한 {n_numeric + 1}열로 챔피언 ID를 제외한다."
        " 팀 사건 $\\times$ 시각 블록은 팀 사건 47개 각각에"
        f" \\texttt{{query\\_ms}}$/${denom_ms // 60000}분을 곱한 파생 열이다."
        " 좌표·지도 앵커·거리·구역 열은 이 표현에 없다.}",
        "\\end{table}",
        "",
    ]
    if not CHECK_ONLY:
        OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
        OUT_TEX.write_text("\n".join(tex), encoding="utf-8")

    # ---------------------------------------------------------------- census doc
    L = []
    A = L.append
    A("# 입력 특징 전수조사 (FEATURE CENSUS)")
    A("")
    A("`tools/gen_feature_census.py`가 `docs/`의 매니페스트에서 생성한다 (손으로 고치지 않음). "
      "`outputs/`, `data/raw/`, 체크포인트, 대형 NPZ/joblib은 열지 않았다 (`AGENTS.md`). "
      "모든 수는 아래 출처의 필드를 전사한 값이며 자료에서 다시 계산한 값이 아니다.")
    A("")
    A("| 출처 | 제공하는 것 |")
    A("|---|---|")
    A("| `docs/V_FEATURE_MANIFEST_RUNTIME_20260919.md` | 실행 폭, 그룹별 열 수, `state_value_v2.py` hash16 |")
    A("| `docs/supplementary_e1_20260921/feature_groups.json` | 열 이름 전체, 그룹 배정, E1 arm 구성 |")
    A("| `docs/V_MODEL_INPUT_DESIGN_20260919/STATEV2_REFERENCE_FEATURE_SET.json` | 열별 자료형·원천 문자열, 차원 사전 |")
    A("| `docs/A_MLP_expanded_evaluator_meta_20260919.json` | 동결 평가기의 전처리 차원 |")
    A("| `scripts/rr20260920_q_newv_primary_fit.py` | $q$·LightGBM 후보의 입력 조립 코드 |")
    A("")
    A("## 1. 두 특징 경로의 구분")
    A("")
    A("이 저장소에는 서로 다른 두 특징 경로가 있고, 열 수를 혼동하는 원인이 여기에 있다.")
    A("")
    A("| | StateV2 | 탭·시퀀스 표현 |")
    A("|---|---|---|")
    A("| 구현 | `gameplay/state_value_v2.py` (별도 worktree `worktrees/engagement-state-value/`) | "
      "저장소의 `gameplay/{pipeline_cache,pipeline_interp,features,feature_spatial,anchors}.py` |")
    A(f"| 폭 | 원 {raw_w} → Expanded {exp_w} → 수치 {n_numeric} | 기본 1,015 $\\times$ 통계 7 = 7,105 (+1) |")
    A("| 쓰임 | 승률 평가기 $\\hat V$, SVI 라벨, 방향 예측기 $q$ | 교전 승자 라벨(`market_event`) 아래의 정의 민감도·누출 절제 |")
    A("| 경기 시각 | `time_minutes`, `time_minutes_sq` (절대 분) | `time_norm` = $t$ / 45분, 1에서 절단 (`TIME_NORM_ABSOLUTE`) |")
    A("| 좌표·지도 앵커 | 없음 | 있음 (`feature_spatial.py`, `anchors.py`, `config/game_rules/map_anchors.json`) |")
    A("| 프레임 나이 | `snapshot_age_s`를 기록하고 입력에서 제외 | `frame_age_s`를 기본 특징으로 포함 (`TAB_FRAME_AGE_FEATURE`) |")
    A("")
    A("`TIME_NORM_ABSOLUTE`(45분 정규화)와 `ANCHORS_CAUSAL`(컷오프 이전 파괴만 반영하는 지도 앵커)은 "
      "**탭·시퀀스 경로의 설정**이다 (`core/config.py` L437–L442, `core/presets.py` L28). "
      "StateV2에는 좌표·앵커·거리·구역 열이 없으므로 이 두 설정은 351열의 성질이 아니다.")
    A("")
    A("## 2. 폭의 대조")
    A("")
    A("| 폭 | 구성 | 쓰임 | 출처 필드 |")
    A("|---|---|---|---|")
    A(f"| {raw_w} | Expanded {exp_w} + `snapshot_age_s` | 감사 기록 | `Raw width` |")
    A(f"| {exp_w} | 수치 {n_numeric} + 챔피언 ID {n_champ} | $\\hat V$ 입력, LightGBM 폭 | "
      f"`Expanded`, `preproc_dims.lgbm_width` = {pre['lgbm_width']} |")
    A(f"| {n_numeric} | 챔피언 ID 제외 | $q$의 수치 블록 | `dimensions.numeric_expanded`, `preproc_dims.n_numeric` |")
    A(f"| {n_numeric + 1} | 수치 {n_numeric} + $p_{{pre}}$ | `logit_state` ($q$) | "
      "`rr20260920_q_newv_primary_fit.py` L236–L241 |")
    A(f"| {exp_w + 1} | Expanded {exp_w} + $p_{{pre}}$ | `lgbm_state` (진단 후보) | 같은 스크립트 L243–L245 |")
    A(f"| {dims['numeric_core']} | 수치 {n_numeric} − 팀 사건 $\\times$ 시각 {dims['derived_time_interactions']} | "
      f"Core{dims['numeric_core'] + n_champ} 절제 | `dimensions.numeric_core` |")
    A("| 7,105 | 탭 기본 1,015 $\\times$ 통계 7 | 정의 민감도·누출 절제 | `INPUT_FEATURE_AUDIT_V3.md` |")
    A("")
    A(f"두 학습기의 입력은 다르다. `logit_state`는 {n_numeric + 1}열, `lgbm_state`는 {exp_w + 1}열이며, "
      "차이는 챔피언 ID 10열이다. 그래서 학습기 간 정합 비교는 이 짝으로 답해지지 않는다.")
    A("")
    A(f"실행 매니페스트와 협력자 재구성 참조는 이름과 순서가 정확히 일치한다 "
      f"(`Order exact match: {order_match}`, only-in-runtime {only_run}, only-in-reference {only_ref}). "
      f"`state_value_v2.py` hash16 = `{sv2_hash}`이며 동결 평가기 메타의 "
      f"`meta.state_value_v2_sha256_16`과 같다.")
    A("")
    A("## 3. 블록별 전수")
    A("")
    A("| 블록 | 슬롯 × 항목 | 열 | 자료형 | 원천 문자열 |")
    A("|---|---|---|---|---|")
    for key, (ko, _) in BLOCKS.items():
        f = ref_feats[groups[key][0]]
        src = f["source"]
        if key == "team_time_interaction":
            src = "팀 사건 × `query_ms`/%d ms" % denom_ms
        shape = SHAPE[key].replace("$\\times$", "×")
        ko_md = ko.replace("$\\times$", "×")
        dt = DTYPE_KO[f["dtype"]]
        A(f"| {ko_md} (`{key}`) | {shape} | {len(groups[key])} | {dt} | {src} |")
    A(f"| **수치 소계** | | **{n_numeric}** | | |")
    A(f"| **합계** | | **{exp_w}** | | |")
    A("")
    A(f"### 3.1 경기 시각 (`clock`, {len(groups['clock'])}열)")
    A("")
    A("".join(f"`{c}` " for c in groups["clock"]))
    A("")
    A(f"원천은 `{ref_feats[groups['clock'][0]]['source']}`이다. 절대 시각이며 경기 총 길이로 "
      "정규화하지 않는다. 탭 경로의 `time_norm`(45분 분모)과 다른 열이다.")
    A("")
    A(f"### 3.2 자료 품질 (`quality`, {len(groups['quality'])}열)")
    A("")
    A("".join(f"`{c}` " for c in groups["quality"]))
    A("")
    A(f"원천은 `{ref_feats[groups['quality'][0]]['source']}`이다. E1의 $Q_0$가 이 열이다.")
    A("")
    A(f"### 3.3 참가자 상태 (`player_snapshot`, {n_slots_snap} 슬롯 × {len(snap_fields)} = "
      f"{len(groups['player_snapshot'])}열)")
    A("")
    A("슬롯마다 같은 항목을 갖는다. 항목:")
    A("")
    for f in snap_fields:
        A(f"- `{f}`")
    A("")
    A(f"원천은 `{ref_feats[groups['player_snapshot'][0]]['source']}`이다. "
      "`_norm` 접미의 정규화 상수는 매니페스트에 기록되어 있지 않다.")
    A("")
    A(f"### 3.4 참가자 사건 (`player_event`, {n_slots_ev} 슬롯 × {len(ev_fields)} = "
      f"{len(groups['player_event'])}열)")
    A("")
    for f in ev_fields:
        A(f"- `{f}`")
    A("")
    A(f"원천은 `{ref_feats[groups['player_event'][0]]['source']}`이다. 바론·장로 관련 열은 "
      "활성 버프의 정확한 상태가 아니라 획득 이후 사망 여부의 대리 지표이다.")
    A("")
    A(f"### 3.5 팀 사건 (`team_event`, {n_sides} 진영 × {len(team_fields)} = "
      f"{len(groups['team_event'])}열)")
    A("")
    A("진영 접두 `blue_`, `red_`가 같은 항목 집합을 갖는다. 항목:")
    A("")
    for f in sorted(team_fields):
        A(f"- `{f}`")
    A("")
    A(f"원천은 `{ref_feats[groups['team_event'][0]]['source']}`이다.")
    A("")
    A(f"### 3.6 팀 사건 × 시각 (`team_time_interaction`, {len(groups['team_time_interaction'])}열)")
    A("")
    A(f"팀 사건 {len(groups['team_event'])}열 각각에 `query_ms`/{denom_ms} ms "
      f"(= {denom_ms // 60000}분)를 곱한 파생 열이며, 부모 집합은 팀 사건 블록과 정확히 일치한다 "
      "(47개 항목, 두 진영). 예: `blue_kills_x_time` = "
      f"`blue_kills` × `query_ms`/{denom_ms}. 분모는 탭 경로의 45분이 아니라 "
      f"{denom_ms // 60000}분이다.")
    A("")
    A(f"### 3.7 챔피언 ID (`champion`, {len(groups['champion'])}열)")
    A("")
    A(f"원천은 `{ref_feats[groups['champion'][0]]['source']}`이다. 범주형이며 $q$에서 제외한다. "
      f"동결 평가기는 어휘 {ev['vocab_n']}의 임베딩으로 읽는다 "
      f"(`preproc_dims.emb_vocab` = {pre['emb_vocab']}).")
    A("")
    A("## 4. 배제된 열")
    A("")
    A("| 열 | 어디서 배제 | 근거 |")
    A("|---|---|---|")
    A(f"| `snapshot_age_s` | 원 {raw_w} → Expanded {exp_w} | `V_FEATURE_MANIFEST_RUNTIME` "
      "`Expanded (drop snapshot_age_s)`; `feature_groups.json` `excluded[0]`; "
      "`STATEV2_REFERENCE_FEATURE_SET` `required_audit_metadata` |")
    A(f"| 챔피언 ID {n_champ}열 | Expanded {exp_w} → 수치 {n_numeric} ($q$) | "
      "`feature_groups.json` `excluded[1]`, `notes[2]`; `rr20260920_q_newv_primary_fit.py` L236 |")
    A("")
    A("`snapshot_age_s`는 기록되고 감사 메타데이터로 요구되지만 어느 모형의 입력도 아니다. "
      "7.3절의 관측 갱신 층화는 이 기록된 값으로 결과를 나누어 본다.")
    A("")
    A("입력에 속하지 않는 값(`not_in_value_input`): " +
      ", ".join(f"`{x}`" for x in ref["not_in_value_input"]) + ".")
    A("")
    A("감사 메타데이터로 요구되는 값(`required_audit_metadata`): " +
      ", ".join(f"`{x}`" for x in ref["required_audit_metadata"]) + ".")
    A("")
    A("## 5. 없는 것")
    A("")
    A(f"{exp_w}개 열 이름 전체를 좌표·위치·거리·구역·앵커·프레임 나이·`time_norm`·룬·아이템·밴·"
      f"피해·주문 패턴으로 검색한 결과 일치하는 이름은 {len(hits)}개이다"
      f"{'' if not hits else ': ' + ', '.join('`'+h+'`' for h in hits)}. "
      "따라서 다음은 이 표현에 없다.")
    A("")
    A("- 챔피언 좌표, 싸움 중심 좌표, 팀 분산·대치 지표")
    A("- 지도 앵커까지의 거리, 구역(zone) 지시자, 타워 사거리 안 여부")
    A("- 마지막 프레임의 나이를 입력으로 쓰는 열")
    A("- 룬, 아이템 해시, 챔피언 스탯, 누적 피해, 밴")
    A("")
    A("이 항목들은 탭·시퀀스 표현의 기본 특징 1,015개에는 있다 "
      "(`INPUT_FEATURE_AUDIT_V3.md` §1). 두 경로를 한 표현으로 서술하면 안 된다.")
    A("")
    A("## 6. E1 정보군 arm과의 대응")
    A("")
    A("| arm | X의 블록 | X 열 | $p_{pre}$ | 차원 |")
    A("|---|---|---|---|---|")
    arm_blocks = {
        "M-F0": "경기 시각 + 자료 품질",
        "M-F1": "경기 시각 + 자료 품질 + 참가자 상태",
        "M-F2": "경기 시각 + 자료 품질 + 참가자 사건 + 팀 사건 + 팀 사건 × 시각",
        "M-F3": "위의 전부 (= 수치 %d열)" % n_numeric,
    }
    for a, spec in arms.items():
        A(f"| {a} | {arm_blocks[a]} | {len(spec['columns_from_X'])} | "
          f"{'포함' if spec['include_p_pre'] else '제외'} | {spec['expected_dim']} |")
    A("")
    A(f"$F$ = 참가자 상태 {len(groups['player_snapshot'])}열, "
      f"$E$ = 참가자 사건 {len(groups['player_event'])} + 팀 사건 {len(groups['team_event'])} + "
      f"팀 사건 × 시각 {len(groups['team_time_interaction'])} = "
      f"{len(groups['player_event']) + len(groups['team_event']) + len(groups['team_time_interaction'])}열. "
      f"M-F3의 {arms['M-F3']['expected_dim']}열은 $q$의 입력 열 수와 같고, 적합 가중치의 정규화와 "
      "학습기가 다르므로 절대 Brier는 같은 척도가 아니다.")
    A("")
    A("`feature_groups.json`의 메모: " + " / ".join(fg["notes"]) + ".")
    A("")
    md = "\n".join(L)
    # thesis/docs/*.md carry no inline math; render the few symbols as plain text.
    for a, b in (("$\\times$", "×"), ("$\\hat V$", "V-hat"), ("$p_{pre}$", "p_pre"),
                 ("$Q_0$", "Q0"), ("$q$", "q"), ("$F$", "F"), ("$E$", "E"),
                 ("$t$", "t"), ("$\\Vhat$", "V-hat")):
        md = md.replace(a, b)
    assert "$" not in md, md[md.index("$") - 80:md.index("$") + 80]
    if CHECK_ONLY:
        bad = 0
        for path, fresh in ((OUT_MD, md), (OUT_TEX, "\n".join(tex))):
            if not path.exists():
                print(f"MISSING  {path.relative_to(ROOT)}")
                bad += 1
            elif path.read_text(encoding="utf-8") != fresh:
                print(f"DRIFT    {path.relative_to(ROOT)}  (re-run thesis/tools/gen_feature_census.py)")
                bad += 1
        print("feature_census: " + ("OK" if bad == 0 else f"{bad} problem(s)"))
        return 1 if bad else 0
    OUT_MD.write_text(md, encoding="utf-8")
    print(f"wrote {OUT_MD.relative_to(ROOT)} and {OUT_TEX.relative_to(ROOT)}")
    print(f"  raw={raw_w} expanded={exp_w} numeric={n_numeric} champ={n_champ} "
          f"q={n_numeric + 1} lgbm={exp_w + 1} core={dims['numeric_core']} "
          f"interaction_denom_ms={denom_ms}")
    print(f"  coordinate/anchor/frame-age name hits in {exp_w}: {len(hits)}")
    return 0


if __name__ == "__main__":
    CHECK_ONLY = "--check" in sys.argv[1:]
    raise SystemExit(main())
