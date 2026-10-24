import json
from collections import Counter

import pytest

from c_ept import cohort
from c_ept import config as C


def spec_counts():
    return json.loads(C.SPEC_PATH.read_text(encoding='utf-8'))['G8_training']['split_counts']


def test_cache_role_counts_equal_spec():
    spec = spec_counts()
    for p in ('15.18', '15.19', '15.20', '15.21', '15.22'):
        got = Counter(cohort.role_of(k, p)[1] for k in cohort.cache_keys(p))
        assert dict(got) == spec[p]


def test_external_16_13_is_named_like_the_decision_record():
    # the split module names the NA1 matches 'OTHER'; the decision record calls them 'external', config.ROLES 'EXTERNAL'
    got = Counter(cohort.role_of(k, '16.13')[1] for k in cohort.cache_keys('16.13'))
    assert dict(got) == {'EXTERNAL': spec_counts()['16.13_NA1']['external']}


def test_macro_role_outside_roles_is_an_error(monkeypatch):
    class Odd:                                              # a split module that returns a macro name unknown to config.ROLES
        @staticmethod
        def bucket(key):
            return 0

        @staticmethod
        def roles(patch, bucket):
            return 'X_15.18', 'BOGUS'

    monkeypatch.setattr(C, 'split_module', lambda: Odd)
    with pytest.raises(ValueError, match='BOGUS'):
        cohort.role_of('SYN_0', '15.18')
    with pytest.raises(ValueError, match='15.14'):
        cohort.cache_keys('15.14')                                                       # not a patch of this study


def test_match_list_order_and_filter():
    fit = cohort.match_list(['15.19', '15.18'], ['FIT'])                                  # patch order is that of config.PATCHES
    spec = spec_counts()
    assert len(fit) == spec['15.18']['FIT'] + spec['15.19']['FIT']
    assert [p for p, _, _ in fit] == ['15.18'] * spec['15.18']['FIT'] + ['15.19'] * spec['15.19']['FIT']
    assert all(m == 'FIT' for _, _, m in fit)
    for p in ('15.18', '15.19'):
        keys = [k for q, k, _ in fit if q == p]
        in_order, in_cache = keys == sorted(keys), set(keys) <= set(cohort.cache_keys(p))        # booleans: a failure must not print keys
        assert in_order and in_cache, p
    both = cohort.match_list(('15.20',), ('FIT', 'STOP'))
    assert Counter(m for _, _, m in both) == Counter(spec['15.20'])                              # role names and counts: no key
    both_keys = [k for _, k, _ in both]
    keys_in_order = both_keys == sorted(both_keys)
    assert keys_in_order
    with pytest.raises(ValueError):
        cohort.match_list(['15.18'], ['FIT', 'NOT_A_ROLE'])
    with pytest.raises(ValueError):
        cohort.match_list(['15.99'], ['FIT'])
