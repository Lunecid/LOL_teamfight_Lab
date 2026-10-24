"""Tests of ``c_ept.heads``: the head interface that the composer reads (stage S3 Task 6, ruling R3).

``ComposerHeads`` is a ``typing.Protocol``; B0 (Task 6), B1 (Task 7) and the later EPT implement it and ``compose`` (Task 9) reads it. The
tests pin the member list of the ruling, check that ``isinstance`` (the protocol is ``runtime_checkable``) tells a complete implementation
from one with a missing member, and keep the module free of anything but numpy and typing, so that every head module can import it.
No store, no run folder and no model is read: the classes below are made up here.
"""
import inspect
import typing
from pathlib import Path

import numpy as np
import pytest

from c_ept import heads
from c_ept.heads import ComposerHeads
from c_ept.tests.conftest import all_imports

SOURCE = Path(heads.__file__)
METHODS = ('bind', 'row_view', 'rate', 'mark_prob', 'zone_prob', 'size_prob', 'kill_rate', 'gold_prob', 'obj_rate', 'conv_rate')
ATTRIBUTES = ('feature_kind', 'analytic_ok', 'obj_uses_K', 'conv_uses_KO')


class Complete:
    """The smallest class that has every member of the protocol."""
    feature_kind = 'b0'
    analytic_ok = True
    obj_uses_K = False
    conv_uses_KO = False

    def bind(self, s2, views): ...
    def row_view(self, rows): ...
    def rate(self, rows): ...
    def mark_prob(self, rows, q): ...
    def zone_prob(self, rows, q, m): ...
    def size_prob(self, rows, q, m, z): ...
    def kill_rate(self, rows, q, m, z, s): ...
    def gold_prob(self, rows, q, m, z, s, layer): ...
    def obj_rate(self, rows, q, m, z, s, K): ...
    def conv_rate(self, m4_inputs): ...


def without(name: str):
    """A copy of ``Complete`` that lacks the member ``name``."""
    body = {k: v for k, v in vars(Complete).items() if k != name and not k.startswith('__')}
    return type('Incomplete', (), body)()


def test_a_class_with_every_member_is_a_composer_heads():
    assert isinstance(Complete(), ComposerHeads)


@pytest.mark.parametrize('missing', METHODS + ATTRIBUTES)
def test_a_class_that_lacks_any_one_member_is_not(missing):
    assert not isinstance(without(missing), ComposerHeads)


def test_an_object_with_nothing_is_not():
    assert not isinstance(object(), ComposerHeads) and not isinstance(None, ComposerHeads)


def test_the_members_are_exactly_those_of_the_ruling():
    assert set(typing.get_protocol_members(ComposerHeads)) == set(METHODS + ATTRIBUTES)


def test_the_protocol_is_runtime_checkable_and_is_a_protocol():
    assert getattr(ComposerHeads, '_is_protocol', False) and getattr(ComposerHeads, '_is_runtime_protocol', False)


@pytest.mark.parametrize('method, params', [
    ('bind', ['self', 's2', 'views']), ('row_view', ['self', 'rows']), ('rate', ['self', 'rows']),
    ('mark_prob', ['self', 'rows', 'q']), ('zone_prob', ['self', 'rows', 'q', 'm']), ('size_prob', ['self', 'rows', 'q', 'm', 'z']),
    ('kill_rate', ['self', 'rows', 'q', 'm', 'z', 's']), ('gold_prob', ['self', 'rows', 'q', 'm', 'z', 's', 'layer']),
    ('obj_rate', ['self', 'rows', 'q', 'm', 'z', 's', 'K']), ('conv_rate', ['self', 'm4_inputs'])])
def test_the_method_parameters_are_those_of_the_ruling(method, params):
    assert list(inspect.signature(getattr(ComposerHeads, method)).parameters) == params


def test_feature_kinds_are_b0_and_b1():
    assert heads.FEATURE_KINDS == ('b0', 'b1')


def test_the_output_widths_are_the_ones_of_the_ruling():
    from c_ept import config as C
    assert heads.OUTPUT_WIDTHS == {'rate': (C.N_BINS,), 'mark_prob': (C.N_MARKS,), 'zone_prob': (10,), 'size_prob': (4,),
                                   'kill_rate': (2,), 'gold_prob': (2, 31), 'obj_rate': (18,), 'conv_rate': (18,)}
    assert heads.OUTPUT_WIDTHS['rate'] == (18,) and heads.OUTPUT_WIDTHS['mark_prob'] == (52,)


def test_the_output_widths_agree_with_the_other_modules():
    from c_ept import features, nll, targets, units
    assert heads.OUTPUT_WIDTHS['zone_prob'] == (features.N_ZONES,)
    assert heads.OUTPUT_WIDTHS['size_prob'] == (len(units.SIZE_NAMES),)
    assert heads.OUTPUT_WIDTHS['gold_prob'] == (2, nll.N_GOLD_BINS)
    assert heads.OUTPUT_WIDTHS['obj_rate'] == (targets.N_CELLS,) == heads.OUTPUT_WIDTHS['conv_rate']


def test_the_module_imports_only_numpy_typing_and_the_config():
    imported = {name.split('.')[0] for name in all_imports(SOURCE)}
    assert imported <= {'__future__', 'typing', 'numpy', 'c_ept'}, imported
    assert not {n for n in all_imports(SOURCE) if n.startswith('c_ept.')} - {'c_ept.config'}


def test_the_module_text_names_no_store_array():
    text = SOURCE.read_text(encoding='utf-8')
    assert 'matches/key' not in text and 'allow_holdout' not in text
    assert np.__name__ == 'numpy'
