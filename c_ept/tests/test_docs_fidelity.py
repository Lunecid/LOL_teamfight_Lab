"""Tests that pin the documentation fidelity fixes of EPT-T011 group 2 (findings A1-F5, A2-F7, A3-F5 of EPT-T010).

The strings below are contracts, not style: a reader of the module header must not be told that tau is a model input time, that the
size classes are kill-count bins, that the evaluation horizons are 30 ... 180 s, or that the (VAL-n) numbers differ from the equation
registry and from the function docstrings. No store, no run folder and no model is read.
"""
import inspect
import re

import pytest

from c_ept import config as C
from c_ept.domain import units
from c_ept.models import baselines_b0
from c_ept.models import heads as model_heads
from c_ept.valuation import towertier, value


def _source(module) -> str:
    return inspect.getsource(module)


# ---- A1-F5: tau only defines the outcome and aftermath windows ------------------------------------------------------------------------
@pytest.mark.parametrize('module', [C, units])
def test_no_comment_calls_tau_a_model_input_time(module):
    text = _source(module)
    for stale in ('입력 snapshot', '상태 관측 기준점', '예측 모델의 입력 기준점', '입력 기준점', '직전 스냅샷'):
        assert stale not in text, stale


def test_the_tau_comments_say_that_tau_only_starts_the_attribution_window():
    assert '귀속 창' in _source(units) and '입력 시점이 아님' in _source(units)
    assert '귀속 창 [tau, delta)' in _source(C) and '관측 시점이 아니다' in _source(C)
    assert '입력의 snapshot 시점이 아니다' in units.UnitTable.__doc__


# ---- A3-F5: towertier keeps the I13 equation, interpretation and evidence sections ---------------------------------------------------
TIER_EQUATIONS = [(towertier.destroyed_before, 'TIER-1'), (towertier.victim_counts, 'TIER-2'), (towertier.TierModel.fit, 'TIER-3'),
                  (towertier.TierModel.prob, 'TIER-4'), (towertier.expected_tower_price, 'TIER-5')]


@pytest.mark.parametrize('function, label', TIER_EQUATIONS)
def test_every_towertier_equation_has_the_three_sections(function, label):
    doc = inspect.getdoc(function)
    assert f'({label})' in doc
    for section in ('수식', '해석', '근거'):
        assert re.search(rf'^{section}\n-+$', doc, flags=re.M), (label, section)


def test_the_towertier_module_text_lists_its_equations_and_cites_the_references():
    doc = towertier.__doc__
    for n in range(1, 6):
        assert f'(TIER-{n})' in doc
    assert _source(towertier).count('# Ref:') >= 6


def test_towertier_does_not_call_the_tier_prices_flow_prices():
    # 포탑 단계 가격은 가격표의 value이다: 억제기 포탑과 넥서스 포탑은 늘 즉시값이고 나머지는 채택된 쪽이다(흐름 가격 고정이 아니다).
    assert 'Flow Price' not in _source(towertier)
    assert 'FIXED_IMMEDIATE' in towertier.__doc__


# ---- A3-F5: value.py header agrees with config and with the equation registry ---------------------------------------------------------
def _header_line(n: int) -> str:
    lines = [ln for ln in value.__doc__.splitlines() if ln.strip().startswith(f'(VAL-{n})')]
    assert len(lines) == 1, (n, lines)
    return lines[0]


def test_value_header_horizons_are_the_evaluation_horizons_of_the_config():
    horizons = ', '.join(str(h // 1000) for h in C.EVAL_H_MS)          # 60, 120, 180
    assert f'H = {horizons}' in value.__doc__
    assert '30, 60, 90' not in value.__doc__


# 등록부(equation-index)의 (VAL-n) 뜻: 1 피해자 레벨, 2 킬 골드, 3 오브젝트 가격, 4 f₀, 5 승자, 6 골드 구간, 7 구간 대표값, 8 V_start, 9 보존 회계.
HEADER_WORDS = {1: '피해자 레벨', 2: '킬 골드', 3: '오브젝트 가격', 4: '교전 단위 가치', 5: '승자', 6: '골드 구간', 7: '구간 대표값', 8: 'V_start', 9: '보존 회계'}
FUNCTION_OF = {1: value.kill_levels_for_record, 2: value.unit_values, 3: value.unit_values, 4: value.unit_values, 5: value.winner_of,
               6: value.gold_bin, 7: value.gold_bin_means, 8: value.v_start, 9: value.accounting}


@pytest.mark.parametrize('n', range(1, 10))
def test_value_header_equation_numbers_match_the_registry_and_the_function_docstrings(n):
    assert HEADER_WORDS[n] in _header_line(n)
    assert f'(VAL-{n})' in FUNCTION_OF[n].__doc__, (n, FUNCTION_OF[n].__name__)


def test_value_block_banners_use_the_registry_numbers():
    text = _source(value)
    expected = {'블록 3': '식 VAL-5 ~ VAL-7', '블록 4': '식 VAL-1', '블록 5': '식 VAL-2 ~ VAL-6', '블록 6': '식 VAL-8', '블록 7': '식 VAL-9'}
    for block, numbers in expected.items():
        banners = re.findall(rf'^# {block}:.*$', text, flags=re.M)
        assert len(banners) == 1 and numbers in banners[0], (block, banners)


# ---- group 1 follow-up: the matched-25% protocol text states the CI rule that pilot_decision implements -----------------------------------
def test_the_matched25_protocol_text_states_the_ci_rule_not_the_point_sign_rule():
    from pathlib import Path
    from c_ept.comparison import protocol
    text = (Path(__file__).resolve().parents[2] / 'docs' / 'EPT_MATCHED25_PROTOCOL_KO.md').read_text(encoding='utf-8')
    assert '음수는 잠정 개선' not in text                                                  # the point-sign rule is gone
    rule = next(line for line in text.splitlines() if line.startswith('2. pilot 손실 차이'))
    for word in ('hi < 0', 'lo > 0', 'provisional_improvement', 'diagnose_before_expansion', 'inconclusive'):
        assert word in rule, word
    # the three status names of the text are exactly the ones the function returns for the three interval positions
    statuses = {protocol.pilot_decision(-.5, [-.8, -.2], 'available')['status'], protocol.pilot_decision(.5, [.2, .8], 'available')['status'],
                protocol.pilot_decision(-.5, [-.8, .2], 'available')['status']}
    assert statuses == {'provisional_improvement', 'diagnose_before_expansion', 'inconclusive'}


# ---- A3-F5: the B0 header names the real size classes --------------------------------------------------------------------------------
def test_the_b0_header_names_the_size_classes_t_s_asym_p_and_not_matchups():
    header = baselines_b0.__doc__
    assert all(f'{i}: {name}' in header for i, name in enumerate(units.SIZE_NAMES))
    assert '2v2, 3v3 등' not in header


# ---- EPT-T011 group 4, item 5: the minors M1, M2, M3 and M5 of the group 2 review ------------------------------------------------------
# M2: tau starts the attribution window [tau, delta) of the outcome O; the aftermath (conversion) window is another one, (delta, delta + CONV_MS].
def test_the_tau_comments_keep_the_attribution_window_apart_from_the_conversion_window():
    window = f'(delta, delta + {C.CONV_MS // 1000}초]'                       # config.CONV_MS; domain.targets.conversions counts in (delta, delta + CONV_MS]
    for module in (C, units):
        text = _source(module)
        assert window in text, module.__name__
        assert '뒤 흐름 귀속 창' not in text, module.__name__                  # "[tau, delta)" was called the window of the aftermath
    assert '귀속 창 [tau, delta)' in units.UnitTable.__doc__ and window in units.UnitTable.__doc__
    assert '귀속 창 [tau, delta)' in units.build_units.__doc__ and window in units.build_units.__doc__


# M5: the config header names the anchor, not an observation time.
def test_the_config_header_does_not_list_an_observation_time_among_the_definition_constants():
    assert '관측 시점 등' not in C.__doc__ and '규칙 앵커 tau 등' in C.__doc__


# M3: a tier price is the flow price only where the run used it (pre-trend adopted and the flow fit estimable); the lines are not longer than 140.
def test_towertier_says_when_a_tier_price_is_the_flow_price():
    module_text = towertier.__doc__
    assert "``used == 'flow'``" in module_text and 'pre-trend' in module_text and 'estimable' in module_text
    assert 'where the run adopted it' not in module_text
    price_text = towertier.expected_tower_price.__doc__
    assert 'the S3a run used for the group' in price_text and 'adopted' not in price_text


def test_towertier_lines_stay_within_140_characters_and_the_first_paragraph_of_the_price_function_is_wrapped_as_prose():
    assert max(len(line) for line in _source(towertier).splitlines()) <= 140
    first = inspect.getdoc(towertier.expected_tower_price).split('\n\n')[0].splitlines()
    assert all(len(line) >= 100 for line in first[:-1])                      # no short line in the middle of the paragraph ("The weights need not")


# M1: FEATURE_KINDS is declared by the implementations and read only by tests; the comment says so and keeps no history.
def test_the_feature_kinds_comment_says_who_declares_and_who_reads_the_field():
    text = _source(model_heads)
    start = text.index('# 머리 구현체가 선언하는 행 특징 종류')
    comment = text[start:text.index('FEATURE_KINDS = ', start)]
    assert '선언하며 실행 코드는 읽지 않는다' in comment and '시험만' in comment
    assert '합성기가 기록·구분에만 쓴다' not in comment and '바로잡았다' not in comment and 'EPT-T010' not in comment
