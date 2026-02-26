from __future__ import annotations

from typing import Any, Dict, Optional


def _norm_layered_global_kind(v: str) -> Optional[str]:
    t = (v or "").lower().strip()
    if not t:
        return None
    if t.startswith("rnn_"):
        t = t[4:]
    alias = {
        "gru": "ugru",
        "ugru": "ugru",
        "bigru": "bigru",
        "lstm": "ulstm",
        "ulstm": "ulstm",
        "bilstm": "bilstm",
        "transformer": "transformer",
        "tcn": "tcn",
        "mamba": "mamba",
    }
    return alias.get(t, None)


def _norm_layered_gnn_kind(v: str) -> Optional[str]:
    t = (v or "").lower().strip()
    if not t:
        return None
    if t.startswith("gnn_"):
        t = t[4:]
    alias = {
        "gcn": "gcn",
        "graphsage": "graphsage",
        "sage": "graphsage",
        "gnnsage": "graphsage",
        "graphtransformer": "graphtransformer",
        "gat": "gatv2",
        "gatv2": "gatv2",
        "mpnn": "mpnn",
    }
    return alias.get(t, None)


def _norm_layered_event_kind(v: str) -> Optional[str]:
    t = (v or "").lower().strip()
    if not t:
        return None
    alias = {
        "attn": "attn",
        "xattn": "xattn",
        "event_xattn": "xattn",
        "mean": "mean",
        "avg": "mean",
        "pool": "mean",
    }
    return alias.get(t, None)


def _parse_bool_optional(v: str) -> Optional[bool]:
    t = (v or "").lower().strip()
    if t in ("1", "true", "on", "yes", "y"):
        return True
    if t in ("0", "false", "off", "no", "n"):
        return False
    return None


def parse_layered_fusion_spec(model_name: str) -> Optional[Dict[str, Any]]:
    """Parse layered-fusion inline alias spec."""
    m_raw = (model_name or "").strip()
    if not m_raw:
        return None
    m = m_raw.lower()

    bases = ("layered_fusion", "fusion_layered", "fusion_layered_gnn_bigru_xattn")
    base = None
    rest = ""
    for b in bases:
        if m == b:
            base = b
            break
        tag = b + "@"
        if m.startswith(tag):
            base = b
            rest = m[len(tag):]
            break
    if base is None:
        return None

    out: Dict[str, Any] = {}
    if not rest:
        return out

    parts = [p.strip() for p in rest.replace(",", "+").split("+") if p.strip()]
    for part in parts:
        if "=" in part:
            k, v = part.split("=", 1)
        elif ":" in part:
            k, v = part.split(":", 1)
        else:
            k, v = "", part

        key = (k or "").lower().strip()
        val = (v or "").lower().strip()

        if key in ("global", "rnn", "g"):
            gk = _norm_layered_global_kind(val)
            if gk is not None:
                out["global_kind"] = gk
            continue
        if key in ("gnn", "graph", "n"):
            nk = _norm_layered_gnn_kind(val)
            if nk is not None:
                out["gnn_kind"] = nk
            continue
        if key in ("event", "attn", "e"):
            ek = _norm_layered_event_kind(val)
            if ek is not None:
                out["event_kind"] = ek
            continue
        if key in ("logit", "lgbm", "use_logit"):
            bv = _parse_bool_optional(val)
            if bv is not None:
                out["use_lgbm_logit"] = bool(bv)
            continue

        gk = _norm_layered_global_kind(val)
        if gk is not None:
            out["global_kind"] = gk
            continue
        nk = _norm_layered_gnn_kind(val)
        if nk is not None:
            out["gnn_kind"] = nk
            continue
        ek = _norm_layered_event_kind(val)
        if ek is not None:
            out["event_kind"] = ek
            continue
        bv = _parse_bool_optional(val)
        if bv is not None and key in ("", "logit", "lgbm"):
            out["use_lgbm_logit"] = bool(bv)

    return out
