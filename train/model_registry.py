from __future__ import annotations

from dataclasses import dataclass
from typing import Callable as _Callable
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

from core.config import cfg
from train.layered_spec import parse_layered_fusion_spec as _parse_layered_fusion_spec


@dataclass
class ModelSpec:
    name: str
    feature_set: str = "full"


def infer_dims_from_batch(batch: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    f_node = -1
    d_macro = -1
    if batch.get("node_seq", None) is not None:
        f_node = int(batch["node_seq"].shape[-1])

    priority = getattr(cfg, "TEMPORAL_SEQ_PRIORITY",
                       ("macro_seq", "x_seq", "extra_seq"))
    for k in priority:
        if batch.get(k, None) is not None:
            d_macro = int(batch[k].shape[-1])
            break

    return f_node, d_macro


MODEL_REGISTRY: Dict[str, _Callable[..., nn.Module]] = {}


def _M():
    import train.models as _models

    return _models


def register_model(*aliases: str):
    def decorator(factory_fn: _Callable[..., nn.Module]):
        for alias in aliases:
            MODEL_REGISTRY[alias.lower().strip()] = factory_fn
        return factory_fn

    return decorator


@register_model("lgbm", "lightgbm", "baseline_lgbm", "tab_lgbm", "tab_logit", "tab")
def _build_tab(f_node, d_seq, use_lgbm_logit):
    return _M().TabLogitModel(prefer_key="lgbm_logit", allow_missing=True)


@register_model("rnn_ugru", "ugru")
def _build_ugru(f_node, d_seq, use_lgbm_logit):
    return _M().RNNOnlyModel("ugru", d_in=d_seq)


@register_model("rnn_bigru", "bigru")
def _build_bigru(f_node, d_seq, use_lgbm_logit):
    return _M().RNNOnlyModel("bigru", d_in=d_seq)


@register_model("rnn_ulstm", "ulstm")
def _build_ulstm(f_node, d_seq, use_lgbm_logit):
    return _M().RNNOnlyModel("ulstm", d_in=d_seq)


@register_model("rnn_bilstm", "bilstm")
def _build_bilstm(f_node, d_seq, use_lgbm_logit):
    return _M().RNNOnlyModel("bilstm", d_in=d_seq)


@register_model("rnn_transformer", "transformer")
def _build_transformer(f_node, d_seq, use_lgbm_logit):
    return _M().RNNOnlyModel("transformer", d_in=d_seq)


@register_model("rnn_tcn", "tcn")
def _build_tcn(f_node, d_seq, use_lgbm_logit):
    return _M().RNNOnlyModel("tcn", d_in=d_seq)


@register_model("rnn_mamba", "mamba")
def _build_mamba(f_node, d_seq, use_lgbm_logit):
    return _M().RNNOnlyModel("mamba", d_in=d_seq)


@register_model("hybrid_bigru", "rnn_hybrid_bigru")
def _build_hybrid_bigru(f_node, d_seq, use_lgbm_logit):
    return _M().HybridRNNModel("bigru", d_in=d_seq)


@register_model("hybrid_bilstm", "rnn_hybrid_bilstm")
def _build_hybrid_bilstm(f_node, d_seq, use_lgbm_logit):
    return _M().HybridRNNModel("bilstm", d_in=d_seq)


@register_model("hybrid_ugru", "rnn_hybrid_ugru")
def _build_hybrid_ugru(f_node, d_seq, use_lgbm_logit):
    return _M().HybridRNNModel("ugru", d_in=d_seq)


@register_model("gnn_gcn", "gcn")
def _build_gcn(f_node, d_seq, use_lgbm_logit):
    return _M().GNNOnlyModel("gcn", f_node=f_node)


@register_model("gnn_graphsage", "graphsage")
def _build_graphsage(f_node, d_seq, use_lgbm_logit):
    return _M().GNNOnlyModel("graphsage", f_node=f_node)


@register_model("gnn_graphtransformer", "graphtransformer")
def _build_graphtransformer(f_node, d_seq, use_lgbm_logit):
    return _M().GNNOnlyModel("graphtransformer", f_node=f_node)


@register_model("gnn_gatv2", "gatv2", "gat")
def _build_gatv2(f_node, d_seq, use_lgbm_logit):
    return _M().GNNOnlyModel("gatv2", f_node=f_node)


@register_model("gnn_mpnn", "mpnn")
def _build_mpnn(f_node, d_seq, use_lgbm_logit):
    return _M().GNNOnlyModel("mpnn", f_node=f_node)


@register_model("gnn_stgnn", "stgnn")
def _build_stgnn(f_node, d_seq, use_lgbm_logit):
    return _M().STGNNModel(
        f_node=f_node,
        gnn_kind=str(getattr(cfg, "STGNN_GNN_KIND", "graphsage")),
        multiscale_adj=bool(getattr(cfg, "STGNN_MULTISCALE_ADJ", False)),
    )


@register_model("gnn_stgcn", "stgcn")
def _build_stgcn(f_node, d_seq, use_lgbm_logit):
    return _M().STGCNModel(
        f_node=f_node,
        gnn_kind=str(getattr(cfg, "STGCN_GNN_KIND", "graphsage")),
        multiscale_adj=bool(getattr(cfg, "STGCN_MULTISCALE_ADJ", False)),
    )


@register_model("edge_stgnn", "stgnn_edge", "stgnn_mpnn", "stgnn_edge_mpnn")
def _build_edge_stgnn(f_node, d_seq, use_lgbm_logit):
    return _M().EdgeSTGNNModel(
        f_node=f_node,
        multiscale_adj=bool(getattr(cfg, "EDGE_STGNN_MULTISCALE_ADJ", False)),
    )


@register_model("ms_stgcn", "multiscale_stgcn", "ms_dyngraph")
def _build_ms_stgcn(f_node, d_seq, use_lgbm_logit):
    return _M().STGCNModel(
        f_node=f_node,
        gnn_kind=str(getattr(cfg, "STGCN_GNN_KIND", "graphsage")),
        multiscale_adj=True,
    )


@register_model("ms_stgnn", "multiscale_stgnn")
def _build_ms_stgnn(f_node, d_seq, use_lgbm_logit):
    return _M().EdgeSTGNNModel(
        f_node=f_node,
        multiscale_adj=True,
    )


@register_model("stgnn_mamba", "st_mamba", "stmamba")
def _build_stmamba(f_node, d_seq, use_lgbm_logit):
    return _M().STMambaModel(
        f_node=f_node,
        gnn_kind=str(getattr(cfg, "STMAMBA_GNN_KIND", "graphsage")),
        multiscale_adj=bool(getattr(cfg, "STMAMBA_MULTISCALE_ADJ", False)),
    )


@register_model("event_xattn", "xattn", "traj_event_xattn")
def _build_xattn(f_node, d_seq, use_lgbm_logit):
    return _M().EventXAttnSTModel(
        f_node=f_node,
        gnn_kind=str(getattr(cfg, "XATTN_GNN_KIND", "mpnn")),
        multiscale_adj=bool(getattr(cfg, "XATTN_MULTISCALE_ADJ", False)),
    )


@register_model("fusion_layered_gnn_bigru_xattn", "layered_fusion", "fusion_layered")
def _build_layered_fusion(f_node, d_seq, use_lgbm_logit):
    return _M().LayeredFusionGNNBiGRUXAttn(
        f_node=f_node,
        d_seq=d_seq,
        use_lgbm_logit=use_lgbm_logit,
    )


@register_model("fusion_gated_gnn_bigru", "lgbm_dual_gnn_bigru", "fusion")
def _build_fusion(f_node, d_seq, use_lgbm_logit):
    return _M().FusionGatedGNNBiGRU(f_node=f_node, d_macro=d_seq, use_lgbm_logit=use_lgbm_logit)


def build_model(model_name: str, f_node: int, d_seq: int, use_lgbm_logit: bool = True) -> nn.Module:
    m = (model_name or "").lower().strip()

    layered_spec = _parse_layered_fusion_spec(m)
    if layered_spec is not None:
        use_logit_eff = bool(layered_spec.get("use_lgbm_logit", use_lgbm_logit))
        return _M().LayeredFusionGNNBiGRUXAttn(
            f_node=f_node,
            d_seq=d_seq,
            use_lgbm_logit=use_logit_eff,
            global_kind=layered_spec.get("global_kind", None),
            gnn_kind=layered_spec.get("gnn_kind", None),
            event_kind=layered_spec.get("event_kind", None),
        )

    if m not in MODEL_REGISTRY:
        available = sorted(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model_name='{model_name}'. "
            f"Available ({len(available)}): {available}"
        )
    return MODEL_REGISTRY[m](f_node, d_seq, use_lgbm_logit)


def sanitize_feature_name(name: str) -> str:
    bad = ["{", "}", "[", "]", ":", '"', "'", "\\", "\n", "\r", "\t"]
    out = name
    for b in bad:
        out = out.replace(b, "_")
    return out


def sanitize_feature_names(names: Sequence[str]) -> List[str]:
    return [sanitize_feature_name(str(n)) for n in names]

