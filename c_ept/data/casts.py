"""The cast rule of S1 and S2: ``narrow(name, a, dtype)`` changes a dtype with the checks described in ``narrow``.

A change to a different dtype is checked and raises, naming the array, for an integer out of range or a float that becomes
infinite. NaN may stay NaN only where the design says so (a missing position, a previous position that is not there, the traits of an
unknown champion). [S2-4]

The function was moved here unchanged from ``c_ept/run_s1.py``, so that a module that is not the S1 runner (``traits``, the S2 modules)
can use it without importing the runner; ``run_s1.narrow`` is this same function. This module imports numpy and nothing else.

방법에서의 역할
---------------
이 모듈은 S1과 S2의 dtype 변환을 한 함수 `narrow`로 모은다. 같은 dtype이면 어떤 검사도 없이 복사하고, 다른 dtype이면 지원하는 변환(부호 있는 정수
대상에는 정수·부호 없는 정수·bool 입력, float 대상에는 float 입력)만 검사한다. 범위를 벗어난 정수와 원래 유한했는데 무한대가 된 float 값은
배열 이름을 단 ValueError로, 그 밖의 변환은 TypeError로 멈춘다 [S1–S2 판정 S2-4]. 이 모듈에는 식이 없고 허용 조건 둘만 있으며 `narrow`의
설명에 적었다.

가정
----
- NaN을 허용하는 곳은 호출하는 쪽이 정하며 이 모듈은 그 허용을 검사하지 않는다. 같은 dtype 복사는 NaN이나 무한대를 포함해 값을 검사하지 않는다.
- float 변환은 원래 유한한 값이 유한하게 남는지만 본다(원래 NaN은 허용하고 원래 무한대는 거부한다). 가수부 반올림과 아주 작은 값이 0이 되는 것은
  거부하지 않으므로 변환 결과가 원래 값과 같다는 보장은 하지 않는다.
"""
from __future__ import annotations

import numpy as np


def narrow(name: str, a, dtype) -> np.ndarray:
    """``a`` as a new C-contiguous array of ``dtype``, checked as described below; exactness of the values is not promised.

    For a different dtype, integer, unsigned-integer or bool input may be cast to a signed integer target if the values fit its range
    (np.iinfo); otherwise ValueError names ``name``. Floating input may be cast to a floating target if each originally non-NaN value is
    finite afterward; original infinities are rejected and original NaNs remain NaN. The same dtype is only copied, before any check. All
    other dtype changes raise TypeError. Rounding and underflow are not checked.

    같은 dtype이면 검사 없이 새 C 연속 복사본을 돌려준다. 다른 dtype에서 부호 있는 정수 대상은 정수·부호 없는 정수·bool 입력을 받고, 안전한 변환이 아니면
    (빈 배열은 제외) 최솟값과 최댓값이 대상의 `np.iinfo` 범위 안이어야 하며 아니면 ValueError이다. float 대상은 float 입력만 받아 원래 NaN이 아닌 값마다
    결과가 유한해야 하고 아니면(원래 무한대 포함) ValueError이다. 그 밖의 변환은 TypeError이다(부호 없는 정수 대상, float에서 정수, 정수에서 float 등).
    가수부 반올림과 0으로의 언더플로는 검사하지 않는다.
    """
    a = np.asarray(a)
    dtype = np.dtype(dtype)
    if a.dtype == dtype:
        return np.array(a, order='C', copy=True)
    # 다른 dtype의 허용 변환: (부호 없는) 정수·bool에서 부호 있는 정수로는 범위(np.iinfo), float에서 float로는 원래 NaN이 아닌 값이 변환 뒤에도 유한함.
    src, dst = a.dtype.kind, dtype.kind
    if dst == 'i' and src in 'iub':
        if a.size and not np.can_cast(a.dtype, dtype, 'safe'):
            info, lo, hi = np.iinfo(dtype), int(a.min()), int(a.max())
            if lo < info.min or hi > info.max:
                raise ValueError(f'{name}: values {lo} .. {hi} do not fit {dtype.name} ({info.min} .. {info.max})')
        return a.astype(dtype, order='C')
    if dst == 'f' and src == 'f':
        with np.errstate(over='ignore', invalid='ignore'):
            out = a.astype(dtype, order='C')
        bad = ~np.isnan(a) & ~np.isfinite(out)
        if bad.any():
            raise ValueError(f'{name}: {int(bad.sum())} value(s) do not fit {dtype.name}: a float must stay finite (NaN is allowed)')
        return out
    raise TypeError(f'{name}: cannot change {a.dtype} to {dtype.name} without a loss')
