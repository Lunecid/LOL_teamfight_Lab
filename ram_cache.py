from __future__ import annotations

from typing import Dict, Any, Optional

try:
    from config import cfg
except ImportError:
    from config import cfg
from collections import OrderedDict

# ---------------------------------------------------------------------
# [P1-2 FIX] Pure OrderedDict LRU cache for load_match_cache
#
# 수학적 배경:
#   LRU(Least Recently Used) 정책은 temporal locality가 있을 때
#   최적에 가까운 캐시 히트율 h를 제공:
#
#     h_LRU ≥ h_FIFO  (일반적으로)
#
#   기존 구현은 OrderedDict(LRU 순서) + List(FIFO eviction)를 혼용하여:
#     - _ram_get()에서 move_to_end() → OrderedDict는 LRU 순서
#     - _ram_put()에서 _RAM_CACHE_ORDER.pop(0) → List는 FIFO 순서
#     - 결과: eviction이 FIFO이므로 최근 접근된 항목이 제거될 수 있음
#     - 추가: list.remove(mid) = O(n) 오버헤드
#
#   수정 후:
#     - OrderedDict 단일 자료구조로 통합
#     - popitem(last=False) = O(1) — 가장 오래된(LRU) 항목 제거
#     - move_to_end(mid) = O(1) — 접근 시 최신으로 이동
#     - 전체 연산 복잡도: get=O(1), put=O(1), evict=O(1)
# ---------------------------------------------------------------------
_RAM_CACHE: OrderedDict[str, Dict[str, Any]] = OrderedDict()


def _ram_cache_enabled() -> bool:
    return bool(getattr(cfg, "CACHE_IN_RAM", False)) or bool(getattr(cfg, "CACHE_MATCH_PACKS_IN_RAM", False))


def _ram_cache_max() -> int:
    return int(getattr(cfg, "CACHE_RAM_MAX_MATCHES", 256))


def _ram_get(mid: str) -> Optional[Dict[str, Any]]:
    """O(1) LRU lookup: 존재하면 move_to_end로 최근 사용 표시."""
    if mid in _RAM_CACHE:
        _RAM_CACHE.move_to_end(mid)  # O(1) — mark as most recently used
        return _RAM_CACHE[mid]
    return None


def _ram_put(mid: str, pack: Dict[str, Any]) -> None:
    """O(1) LRU insert + eviction.

    이미 존재하는 키면 move_to_end로 갱신하고 값을 덮어씀.
    캐시가 가득 차면 popitem(last=False)로 LRU 항목 제거.

    Invariant:
        OrderedDict 순서 = 접근 시간 오름차순
        → popitem(last=False) = LRU eviction
    """
    if mid in _RAM_CACHE:
        _RAM_CACHE.move_to_end(mid)  # O(1) — update access order
    _RAM_CACHE[mid] = pack

    mx = _ram_cache_max()
    while len(_RAM_CACHE) > mx:
        _RAM_CACHE.popitem(last=False)  # O(1) — evict least recently used


def _ram_clear() -> None:
    """캐시 전체 초기화."""
    _RAM_CACHE.clear()