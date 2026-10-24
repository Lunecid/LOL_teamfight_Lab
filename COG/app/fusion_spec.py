"""런타임 종속성 없이 퓨전 모델 CLI 문자열 명세를 파싱 및 확장하는 순수 유틸리티 모듈.

모델 아키텍처, 프리셋, 데이터 분할 런타임 import를 완전히 배제하고, 축별 별칭 정규화 및 데카르트 곱 조합을 통해 layered_fusion 명세 문자열 목록을 결정론적으로 생성합니다."""
from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Tuple


def _parse_model_list(x) -> List[str]:
    """문자열, 시퀀스 또는 반복 가능한 입력을 개별 모델 명세 토큰 문자열 목록으로 정규화합니다.

    None은 빈 리스트를 반환합니다. 일반 iterable 변환 경로의 예외만 빈 리스트로 처리하며, list/tuple 요소의 문자열 변환 예외까지 모두 숨기지는 않습니다. 문자열은 세미콜론(;)을 쉼표(,)로 변환 후 분할하며, 공백 토큰은 제거하고 원본 순서를 유지합니다."""
    if x is None:
        return []
    if isinstance(x, str):
        x = x.replace(";", ",")
        return [t.strip() for t in x.split(",") if t.strip()]
    if isinstance(x, (list, tuple)):
        out = []
        for t in x:
            if t is None:
                continue
            s = str(t).strip()
            if s:
                out.append(s)
        return out
    try:
        return [str(t).strip() for t in list(x) if str(t).strip()]
    except Exception:
        return []


_FUSION_GLOBAL_ORDER: Tuple[str, ...] = ("ugru", "bigru", "ulstm", "bilstm", "transformer", "tcn", "mamba")


_FUSION_GNN_ORDER: Tuple[str, ...] = ("gcn", "graphsage", "graphtransformer", "gatv2", "mpnn")


_FUSION_EVENT_ORDER: Tuple[str, ...] = ("attn", "xattn", "mean")


_FUSION_LOGIT_ORDER: Tuple[str, ...] = ("1", "0")


_FUSION_GLOBAL_ALIASES: Dict[str, str] = {
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


_FUSION_GNN_ALIASES: Dict[str, str] = {
    "gcn": "gcn",
    "graphsage": "graphsage",
    "sage": "graphsage",
    "gnnsage": "graphsage",
    "graphtransformer": "graphtransformer",
    "gat": "gatv2",
    "gatv2": "gatv2",
    "mpnn": "mpnn",
}


_FUSION_EVENT_ALIASES: Dict[str, str] = {
    "attn": "attn",
    "xattn": "xattn",
    "event_xattn": "xattn",
    "mean": "mean",
    "avg": "mean",
    "pool": "mean",
}


_FUSION_LOGIT_ALIASES: Dict[str, str] = {
    "1": "1",
    "true": "1",
    "on": "1",
    "yes": "1",
    "y": "1",
    "with": "1",
    "use": "1",
    "0": "0",
    "false": "0",
    "off": "0",
    "no": "0",
    "n": "0",
    "without": "0",
    "drop": "0",
}


def _stable_unique(xs: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in xs:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _axis_key_from_token(tok: str) -> Optional[Tuple[str, str]]:
    t = (tok or "").strip().lower()
    if not t:
        return None

    if "=" in t:
        k, v = t.split("=", 1)
        return k.strip(), v.strip()
    if ":" in t:
        k, v = t.split(":", 1)
        return k.strip(), v.strip()

    if t in ("rnn_all", "global_all"):
        return "rnn", "all"
    if t in ("gnn_all", "graph_all"):
        return "gnn", "all"
    if t in ("event_all",):
        return "event", "all"
    if t in ("logit_all", "lgbm_all"):
        return "logit", "all"

    if t.startswith(("rnn_", "global_")):
        return "rnn", t
    if t.startswith("gnn_"):
        return "gnn", t
    if t.startswith(("event_", "attn_")):
        return "event", t
    if t.startswith(("logit_", "lgbm_")):
        return "logit", t

    if t in _FUSION_GLOBAL_ALIASES:
        return "rnn", t
    if t in _FUSION_GNN_ALIASES:
        return "gnn", t
    if t in _FUSION_EVENT_ALIASES:
        return "event", t
    if t in _FUSION_LOGIT_ALIASES:
        return "logit", t

    return None


def _normalize_fusion_axis_key(k: str) -> Optional[str]:
    key = (k or "").strip().lower()
    if key in ("rnn", "global", "g"):
        return "rnn"
    if key in ("gnn", "graph", "n"):
        return "gnn"
    if key in ("event", "attn", "e"):
        return "event"
    if key in ("logit", "lgbm", "use_logit", "use_lgbm_logit", "l"):
        return "logit"
    return None


def _expand_fusion_axis_values(axis: str, raw_val: str) -> List[str]:
    v = (raw_val or "").strip().lower()
    if axis == "rnn":
        if v in ("all", "rnn_all", "global_all"):
            return list(_FUSION_GLOBAL_ORDER)
        if v.startswith("global_"):
            v = v[len("global_") :]
        if v.startswith("rnn_"):
            v = v[len("rnn_") :]
        x = _FUSION_GLOBAL_ALIASES.get(v)
        if x is not None:
            return [x]
        raise ValueError(f"Unsupported fusion rnn/global token: {raw_val!r}")

    if axis == "gnn":
        if v in ("all", "gnn_all", "graph_all"):
            return list(_FUSION_GNN_ORDER)
        if v.startswith("gnn_"):
            v = v[len("gnn_") :]
        x = _FUSION_GNN_ALIASES.get(v)
        if x is not None:
            return [x]
        raise ValueError(f"Unsupported fusion gnn token: {raw_val!r}")

    if axis == "event":
        if v in ("all", "event_all"):
            return list(_FUSION_EVENT_ORDER)
        if v.startswith("event_"):
            x = _FUSION_EVENT_ALIASES.get(v)
            if x is not None:
                return [x]
            v = v[len("event_") :]
        if v.startswith("attn_"):
            v = v[len("attn_") :]
        x = _FUSION_EVENT_ALIASES.get(v)
        if x is not None:
            return [x]
        raise ValueError(f"Unsupported fusion event token: {raw_val!r}")

    if axis == "logit":
        if v in ("all", "logit_all", "lgbm_all"):
            return list(_FUSION_LOGIT_ORDER)
        if v.startswith("logit_"):
            v = v[len("logit_") :]
        elif v.startswith("lgbm_"):
            v = v[len("lgbm_") :]
        x = _FUSION_LOGIT_ALIASES.get(v)
        if x is not None:
            return [x]
        raise ValueError(f"Unsupported fusion logit token: {raw_val!r}")

    raise ValueError(f"Unsupported fusion axis: {axis!r}")


def _expand_single_fusion_expr(expr: str) -> List[str]:
    """단일 퓨전 표현식('+'-연결 토큰)을 파싱하여 축별 데카르트 곱으로 확장된 고유 layered_fusion 명세를 생성합니다.

    rnn, gnn, event, logit 축 토큰을 식별·검증하고 각 축의 데카르트 곱 조합을 layered_fusion@... 형식으로 정규화합니다. 공백뿐인 식은 빈 리스트를 반환하며, 해석할 수 없는 토큰이나 지원하지 않는 축/값은 ValueError를 발생시킵니다. 유효 조합의 중복은 최초 순서를 유지해 제거합니다."""
    parts = [p.strip() for p in str(expr or "").split("+") if p.strip()]
    if not parts:
        return []

    axis_vals: Dict[str, List[str]] = {"rnn": [], "gnn": [], "event": [], "logit": []}

    for p in parts:
        kv = _axis_key_from_token(p)
        if kv is None:
            raise ValueError(f"Could not parse fusion token: {p!r}")
        raw_k, raw_v = kv
        axis = _normalize_fusion_axis_key(raw_k)
        if axis is None:
            raise ValueError(f"Unsupported fusion axis key: {raw_k!r}")
        vals = _expand_fusion_axis_values(axis, raw_v)
        for v in vals:
            if v not in axis_vals[axis]:
                axis_vals[axis].append(v)

    rnn_vals = axis_vals["rnn"] or [None]
    gnn_vals = axis_vals["gnn"] or [None]
    event_vals = axis_vals["event"] or [None]
    logit_vals = axis_vals["logit"] or [None]

    out: List[str] = []
    for rnn_v, gnn_v, event_v, logit_v in itertools.product(rnn_vals, gnn_vals, event_vals, logit_vals):
        spec_parts: List[str] = []
        if rnn_v is not None:
            spec_parts.append(f"global=rnn_{rnn_v}")
        if gnn_v is not None:
            spec_parts.append(f"gnn=gnn_{gnn_v}")
        if event_v is not None:
            event_token = "event_xattn" if event_v == "xattn" else event_v
            spec_parts.append(f"event={event_token}")
        if logit_v is not None:
            spec_parts.append(f"logit={logit_v}")

        if spec_parts:
            out.append("layered_fusion@" + "+".join(spec_parts))
        else:
            out.append("layered_fusion")

    return _stable_unique(out)


def _expand_fusion_exprs(raw: str) -> List[str]:
    """다중 퓨전 표현식 문자열을 파싱하고 각각을 직교 축으로 확장하여 중복 없는 전체 퓨전 명세 목록을 반환합니다.

    _parse_model_list로 분할된 각 표현식에 _expand_single_fusion_expr를 적용하며, 축 조합 확장 후에도 최초 등장 순서를 유지하여 안정적인 고유 리스트를 구성합니다."""
    exprs = _parse_model_list(raw)
    out: List[str] = []
    for expr in exprs:
        out.extend(_expand_single_fusion_expr(expr))
    return _stable_unique(out)
