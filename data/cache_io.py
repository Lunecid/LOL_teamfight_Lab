from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from core.common import Any, Dict, List, Optional, Tuple, np
from core.config import CACHE_DIR, F_GLOBAL, F_NODE, cfg
from core.utils import read_json, write_log

logger = logging.getLogger(__name__)

# [P1-2 FIX] _RAM_CACHE_ORDER 제거됨 — OrderedDict 단일 구조로 LRU 통합
from data.ram_cache import _ram_cache_enabled, _ram_get, _ram_put, _RAM_CACHE
from data.events_index import _attach_event_index_inplace
from gameplay.pipeline import parse_timeline_to_minute_cache
from gameplay.fights import build_anchors_from_events, normalize_patch
from data.events_index import _safe_int_dict
from core.roles import get_role_slots_from_detail


# =========================================================
# Paths
# =========================================================
def cache_paths(match_id: str) -> Tuple[Path, Path, Path]:
    return (
        CACHE_DIR / f"{match_id}.npz",
        CACHE_DIR / f"{match_id}.events.json",
        CACHE_DIR / f"{match_id}.meta.json",
    )


# =========================================================
# Helpers: static metadata extraction (champ/runes/bans)
# =========================================================
def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _extract_static_meta_from_detail(detail: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract static participant/team meta from match detail:
      - championId per participantId
      - championName per participantId
      - summoner spell ids (summoner1Id/summoner2Id)
      - rune style ids (primary/sub)
      - rune perks (primary 1..4, sub 1..2, stat perks)
      - team bans
    Stored into meta.json for reproducibility / later feature attachment.

    This does NOT change node_minute shape by itself.
    """
    out: Dict[str, Any] = {}

    info = detail.get("info", {}) if isinstance(detail, dict) else {}
    parts = info.get("participants", [])
    if not isinstance(parts, list):
        parts = []

    champ_by_pid: Dict[str, int] = {}
    champ_name_by_pid: Dict[str, str] = {}
    runes_by_pid: Dict[str, Dict[str, int]] = {}
    summoner_spells_by_pid: Dict[str, Dict[str, int]] = {}

    name_vocab = int(getattr(cfg, "CHAMPION_NAME_VOCAB", 4096))

    def _stable_name_id(name: Any) -> int:
        s = str(name or "").strip().lower()
        if not s:
            return 0
        v = int(hashlib.blake2b(s.encode("utf-8"), digest_size=4).hexdigest(), 16)
        return int(v % max(1, name_vocab - 1)) + 1

    for p in parts:
        if not isinstance(p, dict):
            continue
        pid = _safe_int(p.get("participantId", 0), 0)
        if pid <= 0:
            continue

        champ_name = str(p.get("championName", "") or "")
        champ_name_id = _stable_name_id(champ_name)
        champ_id = _safe_int(p.get("championId", 0), 0)
        if champ_id <= 0:
            champ_id = champ_name_id
        champ_by_pid[str(pid)] = champ_id
        champ_name_by_pid[str(pid)] = champ_name

        # perks 구조: perks.styles[0].selections[0..3].perk (primary 4)
        #          perks.styles[1].selections[0..1].perk (sub 2)
        #          perks.statPerks.{offense, flex, defense}
        perks = p.get("perks", {}) or {}
        styles = perks.get("styles", []) or []
        stat_perks = perks.get("statPerks", {}) or {}

        def _perk_at(style_idx: int, sel_idx: int) -> int:
            try:
                style = styles[style_idx]
                sels = style.get("selections", [])
                return _safe_int(sels[sel_idx].get("perk", 0), 0)
            except Exception:
                return 0

        def _style_at(style_idx: int) -> int:
            try:
                style = styles[style_idx]
                return _safe_int(style.get("style", 0), 0)
            except Exception:
                return 0

        s1 = _safe_int(p.get("summoner1Id", 0), 0)
        s2 = _safe_int(p.get("summoner2Id", 0), 0)
        summoner_spells_by_pid[str(pid)] = {
            "summoner_spell_1_id": s1,
            "summoner_spell_2_id": s2,
        }

        r = {
            "champion_name_id": champ_name_id,
            "summoner_spell_1_id": s1,
            "summoner_spell_2_id": s2,
            "primary_style_id": _style_at(0),
            "sub_style_id": _style_at(1),
            "primary_rune_1": _perk_at(0, 0),
            "primary_rune_2": _perk_at(0, 1),
            "primary_rune_3": _perk_at(0, 2),
            "primary_rune_4": _perk_at(0, 3),
            "sub_rune_1": _perk_at(1, 0),
            "sub_rune_2": _perk_at(1, 1),
            "stat_perk_offense": _safe_int(stat_perks.get("offense", 0), 0),
            "stat_perk_flex": _safe_int(stat_perks.get("flex", 0), 0),
            "stat_perk_defense": _safe_int(stat_perks.get("defense", 0), 0),
        }
        runes_by_pid[str(pid)] = r

    # bans
    bans = {"blue": [0, 0, 0, 0, 0], "red": [0, 0, 0, 0, 0]}
    teams = info.get("teams", [])
    if isinstance(teams, list) and len(teams) >= 2:
        for t in teams:
            if not isinstance(t, dict):
                continue
            tid = _safe_int(t.get("teamId", 0), 0)  # 100/200
            ban_list = t.get("bans", [])
            if not isinstance(ban_list, list):
                ban_list = []

            ids = []
            for b in ban_list[:5]:
                if isinstance(b, dict):
                    ids.append(_safe_int(b.get("championId", 0), 0))
                else:
                    ids.append(0)
            ids += [0] * (5 - len(ids))

            if tid == 100:
                bans["blue"] = ids
            elif tid == 200:
                bans["red"] = ids

    out["champion_by_pid"] = champ_by_pid
    out["champion_name_by_pid"] = champ_name_by_pid
    out["summoner_spells_by_pid"] = summoner_spells_by_pid
    out["runes_by_pid"] = runes_by_pid
    out["bans"] = bans
    return out


def _interp_policy_snapshot() -> Dict[str, Any]:
    """
    Save the intended interpolation policy into meta for reproducibility.
    (Actual interpolation is applied in pipeline/bin builder.)
    """
    return {
        "frame_ms": int(getattr(cfg, "FRAME_MS", 60000)),
        "bin_ms": int(getattr(cfg, "BIN_MS", 5000)),
        # legacy switches
        "interp_method": str(getattr(cfg, "INTERP_METHOD", "linear")),
        "interp_xy": bool(getattr(cfg, "INTERP_XY", True)),
        "interp_scalars": bool(getattr(cfg, "INTERP_SCALARS", True)),
        # recommended split policies (if you added them)
        "interp_xy_method": str(getattr(cfg, "INTERP_XY_METHOD", "linear_guard_midstep")),
        "interp_scalars_method": str(getattr(cfg, "INTERP_SCALARS_METHOD", "ffill")),
        "xy_discont_dist_raw": float(getattr(cfg, "XY_DISCONT_DIST_RAW", 7000.0)),
        "xy_discont_use_alive": bool(getattr(cfg, "XY_DISCONT_USE_ALIVE", True)),
        "xy_guard_mode": str(getattr(cfg, "XY_GUARD_MODE", "midstep")),
    }


def _parse_timeline_to_minute_cache_compat(
    tl: Dict[str, Any],
    tm: Dict[int, int],
    detail: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Backward/forward compatible wrapper:
      - New pipeline may accept (tl, tm, detail=..., cfg=...)
      - Old pipeline may accept (tl, tm)
    """
    # best-effort: newest signature
    try:
        return parse_timeline_to_minute_cache(tl, tm, detail=detail, cfg_in=cfg)  # type: ignore
    except TypeError:
        pass

    # mid signature
    try:
        return parse_timeline_to_minute_cache(tl, tm, detail=detail)  # type: ignore
    except TypeError:
        pass

    # oldest signature
    try:
        return parse_timeline_to_minute_cache(tl, tm)
    except Exception:
        return None


# =========================================================
# Load cache
# =========================================================
def load_match_cache(match_id: str) -> Optional[Dict[str, Any]]:
    """
    Load cache pack:
      {minute_ts, node_minute, global_minute, gold_team_minute, events, meta, (optional) xy_raw_minute}
    + (NEW) optional RAM LRU caching when cfg.CACHE_IN_RAM=True
    """
    if _ram_cache_enabled():
        hit = _ram_get(match_id)
        if hit is not None:
            return hit

    npz, evj, meta = cache_paths(match_id)
    if not (npz.exists() and evj.exists() and meta.exists()):
        return None

    try:
        arr = np.load(npz, allow_pickle=False)
        m = read_json(meta)
        events = read_json(evj)

        if not isinstance(m, dict):
            return None

        # Guard against stale caches from older feature schemas.
        if str(m.get("feature_version", "")) != str(getattr(cfg, "FEATURE_VERSION", "")):
            return None

        if not isinstance(events, list):
            events = []
        events = [e for e in events if isinstance(e, dict)]

        node = arr["node_minute"].astype(np.float32)
        glob = arr["global_minute"].astype(np.float32)
        if node.ndim != 3 or int(node.shape[-1]) != int(F_NODE):
            return None
        if glob.ndim != 2 or int(glob.shape[-1]) != int(F_GLOBAL):
            return None

        team_map = _safe_int_dict(m.get("team_map", None)) or {}
        role_slots = _safe_int_dict(m.get("role_slots", None))
        anchors = m.get("anchors", None)
        if not isinstance(anchors, dict):
            anchors = None

        meta_obj = {
            **m,
            "team_map": team_map,
            "role_slots": role_slots,
            "anchors": anchors,
            "anchor_is_norm": bool(m.get("anchor_is_norm", False)),
        }

        pack: Dict[str, Any] = {
            "minute_ts": arr["minute_ts"].astype(np.int64),
            "node_minute": node,
            "global_minute": glob,
            "gold_team_minute": arr["gold_team_minute"].astype(np.float32),
            "events": events,
            "meta": meta_obj,
        }

        # IMPORTANT: keep xy_raw_minute for safe bin interpolation / jump guard later
        if "xy_raw_minute" in arr.files:
            pack["xy_raw_minute"] = arr["xy_raw_minute"].astype(np.float32)

        _attach_event_index_inplace(pack)

        if _ram_cache_enabled():
            _ram_put(match_id, pack)

        return pack

    except Exception as e:
        logger.debug("Failed to load cache for match %s: %s", match_id, e)
        return None


# =========================================================
# Pair builder
# =========================================================
def build_match_pairs(detail_dir: Path, timeline_dir: Path) -> List[Tuple[str, Path, Path]]:
    dmap = {p.stem: p for p in detail_dir.glob("*.json")}
    pairs: List[Tuple[str, Path, Path]] = []
    for p in timeline_dir.glob("*.json"):
        mid = p.stem
        if mid in dmap:
            pairs.append((mid, dmap[mid], p))

    # deterministic ordering (helps reproducibility/logging)
    pairs.sort(key=lambda x: x[0])
    return pairs


# =========================================================
# Cache prebuilder
# =========================================================
def prebuild_cache(pairs: List[Tuple[str, Path, Path]], log_fp: Optional[Path] = None):
    n_ok, n_skip, n_fail, t0 = 0, 0, 0, time.time()
    log_every = int(getattr(cfg, "CACHE_LOG_EVERY", 5000))

    for i, (mid, dpath, tpath) in enumerate(pairs):
        if getattr(cfg, "MAX_MATCHES", None) and i >= int(cfg.MAX_MATCHES):
            break

        npz, evj, meta = cache_paths(mid)

        # validate existing
        if getattr(cfg, "CACHE_VALIDATE_EXISTING", False) and npz.exists() and evj.exists() and meta.exists():
            if load_match_cache(mid):
                n_skip += 1
                continue
            elif not getattr(cfg, "CACHE_REBUILD_CORRUPT", True):
                n_fail += 1
                continue

        try:
            detail = read_json(dpath)
            tl = read_json(tpath)

            if not isinstance(detail, dict) or not isinstance(tl, dict):
                n_skip += 1
                continue

            parts = detail.get("info", {}).get("participants", None)
            if not isinstance(parts, list) or len(parts) < 10:
                n_skip += 1
                continue

            # team map (participantId -> teamId)
            tm: Dict[int, int] = {}
            okp = 0
            for p in parts:
                if not isinstance(p, dict):
                    continue
                pid = int(p.get("participantId", 0) or 0)
                tid = int(p.get("teamId", 0) or 0)
                if pid > 0 and tid in (100, 200):
                    tm[pid] = tid
                    okp += 1
            if okp < 10:
                n_skip += 1
                continue

            # ---- parse timeline into minute cache ----
            cache = _parse_timeline_to_minute_cache_compat(tl, tm, detail=detail)
            if not cache:
                n_skip += 1
                continue

            # anchors from events
            anchors = build_anchors_from_events(cache["events"])

            # ensure mandatory arrays exist
            if "minute_ts" not in cache or "node_minute" not in cache or "global_minute" not in cache:
                n_fail += 1
                if getattr(cfg, "CACHE_LOG_ERRORS", False):
                    write_log(f"[CACHE][ERR] mid={mid} missing mandatory arrays", log_fp)
                continue

            # IMPORTANT: xy_raw_minute should exist if you want safe jump-guarded XY binning later
            # If missing, we still save, but downstream should handle it.
            xy_raw = cache.get("xy_raw_minute", None)

            # ---- save npz ----
            if xy_raw is not None:
                np.savez_compressed(
                    npz,
                    minute_ts=cache["minute_ts"],
                    node_minute=cache["node_minute"],
                    global_minute=cache["global_minute"],
                    gold_team_minute=cache["gold_team_minute"],
                    xy_raw_minute=xy_raw,
                )
            else:
                np.savez_compressed(
                    npz,
                    minute_ts=cache["minute_ts"],
                    node_minute=cache["node_minute"],
                    global_minute=cache["global_minute"],
                    gold_team_minute=cache["gold_team_minute"],
                )

            # ---- save events ----
            with open(evj, "w", encoding="utf-8") as f:
                json.dump(cache["events"], f, ensure_ascii=False)

            # ---- meta ----
            static_meta = _extract_static_meta_from_detail(detail)
            interp_meta = _interp_policy_snapshot()

            with open(meta, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "match_id": mid,
                        "patch_full": detail.get("info", {}).get("gameVersion", "") or "",
                        "patch": normalize_patch(detail.get("info", {}).get("gameVersion", "") or ""),
                        "team_map": tm,
                        "role_slots": get_role_slots_from_detail(detail),
                        "anchors": anchors,
                        "anchor_is_norm": False,
                        "static_meta": static_meta,
                        "interp_policy": interp_meta,
                        "feature_version": str(getattr(cfg, "FEATURE_VERSION", "")),
                    },
                    f,
                    ensure_ascii=False,
                )

            # if RAM caching enabled, clear corrupted stale entries
            # [P1-2 FIX] OrderedDict 단일 구조 — pop만으로 충분
            if _ram_cache_enabled():
                _RAM_CACHE.pop(mid, None)

            n_ok += 1

        except Exception as e:
            n_fail += 1
            if getattr(cfg, "CACHE_LOG_ERRORS", False):
                write_log(f"[CACHE][ERR] mid={mid} err={e}", log_fp)

        if (i + 1) % log_every == 0:
            write_log(f"[CACHE] i={i+1} ok={n_ok} skip={n_skip} fail={n_fail}", log_fp)

    write_log(f"[CACHE DONE] ok={n_ok} skip={n_skip} fail={n_fail} time={time.time()-t0:.1f}s", log_fp)
