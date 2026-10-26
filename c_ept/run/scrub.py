"""Removal of match keys from the text a run prints or records (S4b-prep K5b, decision D11, Q7: "every traceback is scrubbed of match keys").

A match key is ``<platform>_<game id>`` (``KR_1234567890``). It must reach neither a terminal nor a log nor a ledger record: a traceback can carry
one in an exception message, in a path (``.../KR_1234567890/timeline.json``) or in a quoted value of a data file. The S1-S3 stack has the rule in two
places, the frozen ``workflows.run_s3._scrub_text`` (a loop over the keys of the S1 store) and ``data.store_build._scrub`` (one key). The earlier
stacks are frozen and ``run_s3`` imports numpy through ``data.store_build``, so the kernel keeps the same rule here, with the standard library only,
and a test (``test_run_supervisor.py``) runs both on the same inputs. The keys come from the caller (the supervisor is given them; it reads no data).

Symbols
-------
    K       the keys given by the caller, in order; k one key
    n(k)    the game id of k: the digits after the last underscore (``(?:^|_)(\\d+)$``), or none; for a key of digits only, n(k) = k
    P(k)    the parts of k, longest first: () for the empty key; (k) if n(k) is none or n(k) = k; else (k, n(k))
    L       the labels ``('<key>', '<id>')``; the first part gets <key>, the second <id>
    S(t, k) = t with, for (part, label) in zip(P(k), L): every occurrence of part replaced by label (in this order, so the key goes first and the id
              inside what is left is still found)
    scrub(t, K) = S(...S(S(t, k_1), k_2)..., k_m)

해석
----
경기 키는 ``<플랫폼>_<게임 번호>``이다. 오류 메시지, 경로, 데이터 파일에서 따온 값 어디에든 들어갈 수 있으므로, 화면과 기록에 나가기 전에 키를 ``<key>``로, 키 안의
게임 번호를 ``<id>``로 바꾼다. 키마다 차례로 바꾸므로 키 목록의 순서가 결과에 영향을 줄 수 있다(키가 다른 키의 일부일 때). 옛 규칙과 똑같이 하기 위해 순서를 그대로 둔다.
키가 빈 글자이면 아무것도 바꾸지 않는다.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

LABELS = ('<key>', '<id>')
_GAME_ID = re.compile(r'(?:^|_)(\d+)$')


# Ref: c_ept/data/store_build.py ``_key_parts`` and ``_scrub`` (the rule) and c_ept/workflows/run_s3.py ``_scrub_text`` (the loop over the keys of the
# S1 store). Copied, not imported: both frozen modules import numpy. Same output for the same input (agreement test in test_run_supervisor.py).
def _key_parts(key: str) -> tuple:
    """What names a match in a text: the key and, when the key ends in a number after an underscore, the game id on its own; longest first, empty for
    an empty key."""
    if not key:
        return ()
    number = _GAME_ID.search(key)
    return (key,) if number is None or number.group(1) == key else (key, number.group(1))


def scrub_text(text: str, keys: Iterable[str]) -> str:
    """``text`` with every match key in ``keys`` replaced by ``<key>`` and every game id inside a key by ``<id>``.

    The output is the same as that of ``run_s3._scrub_text(text, keys)``. ``keys`` is read once, in order (any iterable of text); an empty key changes
    nothing; a key that is a non-empty value other than text is a ``TypeError``.
    """
    for key in keys:
        for part, label in zip(_key_parts(key), LABELS):
            text = text.replace(part, label)
    return text
