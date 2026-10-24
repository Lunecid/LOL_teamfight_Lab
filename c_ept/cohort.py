"""Cohort match lists: which matches of a patch are cached, and which role the cohort split gives each of them.

The split module (config.split_module) is the only authority on roles: this module calls only its ``bucket()`` and ``roles()``.
The cached matches of the five KR patches and of the NA1 external patch are the cohort; their counts per role must equal the
decision record's G8_training.split_counts (see the tests).
"""
# Ref: docs/superpowers/specs/2026-10-20-c-ept-design.md section 11, item 15 (cohort split; counts equal the decision record).
# Ref: D:\LOL_Data\runs\20261019_constant_drift_fit_1\SPEC_C_DEFINITION.json, key G8_training.split_counts.
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from c_ept import config as C

# Name correspondence between the split module and config.ROLES / the decision record, keyed by (patch, name the split module gives).
# The split module knows only the KR holdout design of 15.18-15.22 and names every other match 'OTHER'; the decision record calls the
# 10,000 cached NA1 matches of 16.13 'external' and config.ROLES has 'EXTERNAL'. The patch is part of the key so that an 'OTHER'
# anywhere else still raises. (The names of 15.18-15.22 need no alias: they equal the decision record's.)
MACRO_ALIASES = {('16.13', 'OTHER'): 'EXTERNAL'}

_META_SUFFIX = '.meta.json'


def _check_patch(patch: str) -> None:
    if patch not in C.CACHE_SETS:
        raise ValueError(f'unknown patch {patch!r}: not one of {C.PATCHES}')


def cache_keys(patch: str) -> list[str]:
    """Keys of the cached matches of ``patch`` (names of the ``*.meta.json`` files without the suffix), sorted."""
    _check_patch(patch)
    d: Path = C.CACHE_ROOT / patch / 'cache' / C.CACHE_SETS[patch]
    if not d.is_dir():
        raise FileNotFoundError(f'cache folder of patch {patch} not found: {d}')
    return sorted(p.name[:-len(_META_SUFFIX)] for p in d.glob('*' + _META_SUFFIX))


def role_of(key: str, patch: str) -> tuple[str, str]:
    """``(split_role, macro_role)`` of a match: the split module's ``roles(patch, bucket(key))``, the macro name checked against
    ``config.ROLES`` (after ``MACRO_ALIASES``); ValueError when it is not one of them."""
    _check_patch(patch)
    split = C.split_module()
    split_role, macro = split.roles(patch, split.bucket(key))
    macro = MACRO_ALIASES.get((patch, macro), macro)
    if macro not in C.ROLES:
        raise ValueError(f'macro role {macro!r} of a match of patch {patch} is not in config.ROLES {C.ROLES}')
    return split_role, macro


def _names(x: str | Iterable[str]) -> tuple[str, ...]:
    return (x,) if isinstance(x, str) else tuple(x)


def match_list(patches: str | Iterable[str], macros: str | Iterable[str]) -> list[tuple[str, str, str]]:
    """``(patch, key, macro)`` of every cached match of ``patches`` whose macro role is in ``macros``.

    Order: the patches in the order of ``config.PATCHES`` (whatever the order of the argument), keys sorted within a patch, so that
    the list does not depend on how it was asked for. Unknown patch or role names raise ValueError.
    """
    patches, macros = _names(patches), _names(macros)
    for p in patches:
        _check_patch(p)
    for m in macros:
        if m not in C.ROLES:
            raise ValueError(f'unknown macro role {m!r}: not one of {C.ROLES}')
    out = []
    for p in (q for q in C.PATCHES if q in patches):
        for key in cache_keys(p):
            macro = role_of(key, p)[1]
            if macro in macros:
                out.append((p, key, macro))
    return out
