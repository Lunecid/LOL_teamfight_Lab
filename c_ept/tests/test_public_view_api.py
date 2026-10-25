"""Synthetic public API and compatibility regressions; no store payloads."""
import numpy as np
import pytest
from c_ept.preprocessing import features
from c_ept.domain import symmetry

def test_public_names_keep_legacy_identity():
    assert features.view_index is features._view_index
    assert symmetry.SUMMARY_SWAP is symmetry._SUMMARY_SWAP
    perm = np.asarray(symmetry.SUMMARY_SWAP)
    np.testing.assert_array_equal(perm[perm], np.arange(len(perm)))

@pytest.mark.parametrize('value,expected', [([2,0,2],[2,0,2]), ([False,True,False],[1]), ([],[]), (np.empty((0,2)),[])])
def test_public_view_index_preserves_order_mask_and_empty_contract(value,expected):
    out = features.view_index(3,value)
    assert out.dtype == np.int64
    assert out.ndim == 1
    np.testing.assert_array_equal(out,expected)

@pytest.mark.parametrize('value,error', [([1.0],TypeError), ([[1]],ValueError), ([-1],ValueError), ([3],ValueError), ([True],ValueError)])
def test_public_view_index_preserves_refusals(value,error):
    with pytest.raises(error): features.view_index(3,value)
