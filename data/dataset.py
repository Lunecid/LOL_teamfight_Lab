"""dataset.py — InMemoryFightDataset + collate_batch  (v2 optimised)

Performance-critical changes
============================

[OPT-1]  Match-grouped preloading
    Original: for *each* ref r_i, call load_match_cache(r_i.match_id)
    independently.  If M unique matches produce N refs (avg k̄ = N/M
    refs per match), the original performs N cache lookups, each hitting
    the LRU list with O(cache_size) removal cost.

    Fix: group refs by match_id → load each pack *once* → process all
    co-located refs → release.  Total I/O: O(N) → O(M).

    ┌─────────────────────────────────────────────────────────────┐
    │  Old:  r₁→load(m₁)  r₂→load(m₁)  r₃→load(m₁)  r₄→load(m₂)  │
    │        ─────────────────────────────────────────────────────│
    │  New:  load(m₁)→[r₁,r₂,r₃]  load(m₂)→[r₄]                │
    └─────────────────────────────────────────────────────────────┘

[OPT-2]  Zero-copy __getitem__ (pre-injected ref_key)
    Original __getitem__ copies the entire sample dict every call
    solely to inject "ref_key".  With N=200k, epoch=10, that's
    2M unnecessary dict copies.

    Fix: inject ref_key at preload time.  __getitem__ returns a
    direct reference.  Training never mutates dict values in-place
    (gradients operate on cloned parameter tensors), so aliasing
    is safe.

[OPT-3]  Progress reporting with ETA
    200k samples × ~5ms/sample ≈ 16 min silent wait.
    Now prints ~5% interval progress with elapsed time.

[OPT-4]  Factored _build_sample(pack, tm, r)
    Extracts the per-sample kernel from _load so that grouped
    loading can reuse the same pack across sibling refs without
    re-entering load_match_cache.
"""
from __future__ import annotations

import time
from collections import defaultdict

from core.common import Any, Dict, List, Optional, Tuple, np
from core.config import NODE_IDX, cfg

import torch
from torch.utils.data import Dataset

from core.fight_types import FightRef, PruneSpec, ref_key
from data.cache_io import load_match_cache
from gameplay.pipeline import build_ms_sequence
from data.logits import _normalize_logit_maps, _cfg_wants_logits
from core.contract import SPATIAL_NAMES
from gameplay.features import build_sequence_features, get_extra_feature_names, get_xseq_feature_names


# =====================================================================
# Event-token keys (consumed by EventXAttnSTModel)
#   pipeline.py::build_event_tokens_for_xattn() produces 5 tensors:
#     event_type   (K,)    int64   — event category hash
#     event_actor  (K,)    int64   — acting participantId
#     event_team   (K,)    int64   — team (0=blue, 1=red, 2=unk)
#     event_cont   (K, 5)  float32 — continuous [t_rel, dt_end, x, y, val]
#     event_mask   (K,)    float32 — padding mask (1=real, 0=pad)
# =====================================================================
_EVENT_TOKEN_KEYS = ("event_type", "event_actor", "event_team", "event_cont", "event_mask")

_EVENT_TOKEN_DTYPES = {
    "event_type":  torch.long,
    "event_actor": torch.long,
    "event_team":  torch.long,
    "event_cont":  torch.float32,
    "event_mask":  torch.float32,
}


class InMemoryFightDataset(Dataset):
    """Fight-level dataset with optional RAM preloading.

    Mathematical contract
    ---------------------
    Each sample s_i is a dict of tensors representing one teamfight:

        s_i = { node_seq  ∈ ℝ^{T × N × F_node},       (GNN/RNN input)
                macro_seq ∈ ℝ^{T × D_macro}  | extra_seq | x_seq,
                y         ∈ {0,1}^{1×1},                (label)
                ref_key   ∈ str,                         (alignment key)
                [opt]  *_logit ∈ ℝ^{1×1} }              (teacher logits)

    where  T = context_ms / bin_ms  (temporal bins),
           N = 10  (players),
           F_node = |NODE_FEATURE_NAMES|.
    """

    def __init__(
        self,
        refs: List[FightRef],
        feature_set: str,
        model_name: str,
        prune: Optional[PruneSpec] = None,
        scaler=None,
        lgbm_logit_map: Optional[Dict[str, float]] = None,
        cache_in_ram: bool = True,
        logit_maps: Optional[Dict[str, Dict[str, float]]] = None,
        force_emit_logits: Optional[bool] = None,
    ):
        self.refs_all = list(refs)
        self.refs = list(refs)

        self.feature_set = feature_set
        self.model_name = model_name
        self.prune = prune or PruneSpec()
        self.scaler = scaler

        # ---- logit maps ------------------------------------------------
        self.lgbm_logit_map = lgbm_logit_map if lgbm_logit_map is not None else None

        self.logit_maps: Dict[str, Dict[str, float]] = _normalize_logit_maps(
            lgbm_logit_map=lgbm_logit_map,
            logit_maps=logit_maps,
        )

        mn = str(model_name or "").lower()
        cfg_wants = _cfg_wants_logits(model_name)

        if force_emit_logits is not None:
            self.emit_logits = bool(force_emit_logits)
        else:
            provided_any = (lgbm_logit_map is not None) or (logit_maps is not None)
            looks_logit_model = ("logit" in mn) or (mn in ("tablogit", "tab_logit", "tablogitmodel")) or cfg_wants
            self.emit_logits = bool(provided_any or looks_logit_model)

        if self.emit_logits:
            if "lgbm_logit" not in self.logit_maps:
                self.logit_maps["lgbm_logit"] = {}
            if bool(getattr(cfg, "LOGIT_WARN_EMPTY", True)):
                if all((not mp) for mp in self.logit_maps.values()):
                    print("[WARN] emit_logits=True but all logit maps are empty -> batch logits will be zeros.")

        self.cache_in_ram = bool(cache_in_ram)

        # [OPT-4] Hoist cfg lookups out of per-sample hot path
        self.exclude_prefixes = tuple(getattr(cfg, "SCALER_EXCLUDE_PREFIXES", ("x_", "y_", "pos_", "dist_", "angle_")))
        self.spatial_names = set(SPATIAL_NAMES)
        self._log_dropped = bool(getattr(cfg, "LOG_DROPPED_REFS", True))

        # ---- storage ---------------------------------------------------
        self.samples: List[Dict[str, Any]] = []
        self.dropped_refs: List[FightRef] = []
        self.dropped_keys: List[str] = []

        # ---- preload ---------------------------------------------------
        if self.cache_in_ram:
            self._preload_grouped()

    # =================================================================
    # [OPT-1] Match-grouped preloading
    # =================================================================
    def _preload_grouped(self) -> None:
        """Load samples grouped by match_id to minimise cache I/O.

        Complexity analysis
        -------------------
        Let  N = |refs|,  M = |unique match IDs|,  k̄ = N / M.

        Original:
            T_old = Σ_{i=1}^{N}  [ C_load(m_i) + C_build(r_i) ]
                  = N · C̄_load  +  N · C̄_build

        Optimised (grouped):
            T_new = Σ_{m=1}^{M}  [ C_load(m) + k_m · C_build ]
                  = M · C̄_load  +  N · C̄_build

        Savings:  ΔT = (N − M) · C̄_load  =  N · (1 − 1/k̄) · C̄_load

        With k̄ ≈ 4 and C_load ≈ 2ms (disk) or 0.5ms (RAM-LRU):
            ΔT ≈ 200k × 0.75 × 2ms ≈ 300s  (disk)
            ΔT ≈ 200k × 0.75 × 0.5ms ≈ 75s  (RAM-cached)
        """
        t0 = time.time()
        n_total = len(self.refs)
        print(f"[RAM] Loading {n_total} samples (match-grouped)...")

        # Step 1: group ref indices by match_id  — O(N)
        #   groups[match_id] = [(original_index, ref), ...]
        groups: Dict[str, List[Tuple[int, FightRef]]] = defaultdict(list)
        for i, r in enumerate(self.refs):
            groups[r.match_id].append((i, r))

        n_matches = len(groups)

        # Pre-allocate result slots (indexed by original position for order preservation)
        results: List[Optional[Dict[str, Any]]] = [None] * n_total

        n_ok = 0
        n_fail = 0
        log_interval = max(1, n_matches // 20)    # print ~5% increments

        for m_idx, (match_id, idx_ref_pairs) in enumerate(groups.items()):
            # ── single disk read per match ──
            pack = load_match_cache(match_id)

            if pack is None:
                n_fail += len(idx_ref_pairs)
                continue

            tm = pack["meta"]["team_map"]
            role_slots = pack["meta"].get("role_slots", None)

            # ── process all sibling refs from this match ──
            for orig_idx, r in idx_ref_pairs:
                sample = self._build_sample(pack, tm, role_slots, r)
                if sample is not None:
                    # [OPT-2] Pre-inject ref_key → no copy needed in __getitem__
                    sample["ref_key"] = ref_key(r)
                    results[orig_idx] = sample
                    n_ok += 1
                else:
                    n_fail += 1

            # Release pack reference (helps GC when not RAM-cached)
            del pack

            # [OPT-3] Progress reporting
            if (m_idx + 1) % log_interval == 0:
                elapsed = time.time() - t0
                pct = 100.0 * (m_idx + 1) / n_matches
                eta = elapsed / (m_idx + 1) * (n_matches - m_idx - 1)
                print(
                    f"[RAM]  {pct:5.1f}%  matches={m_idx+1}/{n_matches}  "
                    f"ok={n_ok}  fail={n_fail}  "
                    f"elapsed={elapsed:.1f}s  ETA={eta:.0f}s"
                )

        # Step 2: compact — keep successful samples, preserve original order
        kept_refs: List[FightRef] = []
        kept_samples: List[Dict[str, Any]] = []

        for i, r in enumerate(self.refs):
            s = results[i]
            if s is not None:
                kept_refs.append(r)
                kept_samples.append(s)
            else:
                self.dropped_refs.append(r)
                try:
                    self.dropped_keys.append(ref_key(r))
                except Exception:
                    self.dropped_keys.append("")

        self.refs = kept_refs
        self.samples = kept_samples

        elapsed = time.time() - t0
        print(
            f"[RAM] Done: kept={len(self.refs)}  dropped={len(self.dropped_refs)}  "
            f"matches={n_matches}  time={elapsed:.1f}s"
        )
        if self.dropped_refs and self._log_dropped:
            show = self.dropped_keys[:10]
            print(f"[RAM] Dropped keys (first 10): {show}")

    # =================================================================
    # [OPT-4] Factored sample builder
    #     Separated from _load() so that grouped loading can pass
    #     the same pre-loaded pack to multiple sibling refs.
    # =================================================================
    def _build_sample(
        self,
        pack: Dict[str, Any],
        tm: Dict[int, int],
        role_slots,
        r: FightRef,
    ) -> Optional[Dict[str, Any]]:
        """Build one training sample from a pre-loaded cache pack.

        Data flow
        ---------
            cache_pack × FightRef
                → build_ms_sequence       (raw temporal sequence)
                → build_sequence_features (feature extraction)
                → tensor construction     (node/macro/x layout)
                → logit injection         (optional teacher signals)
                → sample dict
        """
        # 1) Build raw ms-level sequence
        try:
            label_end_ts = int(getattr(r, "label_end_ts", -1))
        except Exception:
            label_end_ts = -1

        if r.t_start_ts >= 0:
            raw = build_ms_sequence(
                pack,
                tm,
                -1,
                engage_ts=r.t_start_ts,
                label_end_ts=(label_end_ts if label_end_ts >= 0 else None),
            )
        else:
            raw = build_ms_sequence(pack, tm, r.t_start)

        if not raw:
            return None

        # 2) Feature extraction
        feats = build_sequence_features(raw, tm, role_slots, self.feature_set)
        y = torch.tensor([[float(feats["y"])]], dtype=torch.float32)

        def _inject_aux_targets(dst: Dict[str, Any]) -> None:
            for k in ("y_kill_diff", "y_gold_diff", "y_obj_diff"):
                try:
                    v = float(feats.get(k, 0.0))
                except Exception:
                    v = 0.0
                dst[k] = torch.tensor([[v]], dtype=torch.float32)

        # 3) Tensor construction — three possible layouts:
        #    (a) node_seq + macro_seq + tab_x  (MacroFusion models)
        #    (b) node_seq + extra_seq           (GNN/RNN models)
        #    (c) x_seq only                     (flat sequence models)
        if "macro_seq" in feats and "tab_x" in feats:
            node_np = feats.get("node_seq", None)
            macro_np = feats.get("macro_seq", None)
            tab_np = feats.get("tab_x", None)

            node_ts = torch.from_numpy(node_np).float() if node_np is not None else None
            macro_ts = torch.from_numpy(macro_np).float() if macro_np is not None else None

            macro_names = get_extra_feature_names(self.feature_set)
            node_ts, macro_ts = self._apply_scaler_and_restore_coords(node_ts, macro_ts, seq_names=macro_names)

            if self.prune.extra_keep is not None and macro_ts is not None:
                macro_ts = macro_ts[:, self.prune.extra_keep]

            tab_ts = torch.from_numpy(tab_np).float().view(1, -1)
            out = {"node_seq": node_ts, "macro_seq": macro_ts, "tab_x": tab_ts, "y": y}

        elif "node_seq" in feats and "extra_seq" in feats:
            node_np = feats.get("node_seq", None)
            extra_np = feats.get("extra_seq", None)

            node_ts = torch.from_numpy(node_np).float() if node_np is not None else None
            extra_ts = torch.from_numpy(extra_np).float() if extra_np is not None else None

            extra_names = get_extra_feature_names(self.feature_set)
            node_ts, extra_ts = self._apply_scaler_and_restore_coords(node_ts, extra_ts, seq_names=extra_names)

            if self.prune.extra_keep is not None and extra_ts is not None:
                extra_ts = extra_ts[:, self.prune.extra_keep]

            out = {"node_seq": node_ts, "extra_seq": extra_ts, "y": y}

        else:
            x_np = feats.get("x_seq", None)
            if x_np is None:
                return None
            x_ts = torch.from_numpy(x_np).float()

            x_names = get_xseq_feature_names(self.feature_set)
            _, x_ts = self._apply_scaler_and_restore_coords(None, x_ts, seq_names=x_names)

            if self.prune.x_keep is not None:
                x_ts = x_ts[:, self.prune.x_keep]

            out = {"x_seq": x_ts, "y": y}

        _inject_aux_targets(out)

        # 4) Event tokens (for EventXAttnSTModel)
        self._inject_event_tokens(out, raw)

        # 5) Logit passthrough (teacher signals)
        self._inject_logits(out, r)

        return out

    # =================================================================
    # Public interface
    # =================================================================
    def __len__(self):
        return len(self.refs)

    def __getitem__(self, idx):
        r = self.refs[idx]

        if self.cache_in_ram:
            # [OPT-2] ref_key already injected at preload → return directly
            #
            # Safety note: PyTorch DataLoader workers receive pickled copies
            # of the dataset, so in-process aliasing doesn't cause races.
            # Training never mutates tensor *values* in-place (autograd
            # operates on parameter tensors, not input tensors).
            return self.samples[idx]

        # Lazy-loading path (cache_in_ram=False)
        s = self._load(r)
        if s is None:
            return None

        if isinstance(s, dict):
            s["ref_key"] = ref_key(r)
            return s

        if isinstance(s, (tuple, list)) and len(s) == 2:
            x, y = s
            return {"x": x, "y": y, "ref_key": ref_key(r)}

        return {"x": s, "ref_key": ref_key(r)}

    # =================================================================
    # Legacy single-ref loader (fallback for cache_in_ram=False)
    # =================================================================
    def _load(self, r: FightRef) -> Optional[Dict[str, Any]]:
        """Load one sample via single-ref path (backward-compatible)."""
        pack = load_match_cache(r.match_id)
        if not pack:
            return None
        return self._build_sample(
            pack,
            pack["meta"]["team_map"],
            pack["meta"].get("role_slots", None),
            r,
        )

    # =================================================================
    # Scaler / coordinate restoration
    # =================================================================
    def _restore_no_scale_cols(
            self,
            seq_out: torch.Tensor,
            seq_orig: torch.Tensor,
            seq_names: List[str],
    ) -> torch.Tensor:
        if seq_out is None or seq_orig is None:
            return seq_out
        if not isinstance(seq_names, list) or len(seq_names) != int(seq_out.shape[-1]):
            return seq_out

        restore_idx = []
        for i, n in enumerate(seq_names):
            s = str(n)
            if s in self.spatial_names:
                restore_idx.append(i)
                continue
            for pref in self.exclude_prefixes:
                if s.startswith(pref):
                    restore_idx.append(i)
                    break

        if restore_idx:
            seq_out[..., restore_idx] = seq_orig[..., restore_idx]
        return seq_out

    def _apply_scaler_and_restore_coords(
            self,
            node_ts: Optional[torch.Tensor],
            seq_ts: Optional[torch.Tensor],
            seq_names: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.scaler is None:
            return node_ts, seq_ts

        node_orig = node_ts.clone() if node_ts is not None else None
        seq_orig = seq_ts.clone() if seq_ts is not None else None

        node_out, seq_out = self.scaler.transform(node_ts, seq_ts)

        if node_out is not None and node_orig is not None:
            xi = NODE_IDX.get("x_norm", None)
            yi = NODE_IDX.get("y_norm", None)
            if xi is not None:
                node_out[..., xi] = node_orig[..., xi]
            if yi is not None:
                node_out[..., yi] = node_orig[..., yi]

        if seq_out is not None and seq_orig is not None and seq_names is not None:
            seq_out = self._restore_no_scale_cols(seq_out, seq_orig, seq_names)

        return node_out, seq_out

    # =================================================================
    # Logit injection
    # =================================================================
    def _inject_logits(self, out: Dict[str, Any], r: FightRef) -> None:
        if not self.emit_logits:
            return

        rk = ref_key(r)

        if "lgbm_logit" in self.logit_maps:
            try:
                lg = float(self.logit_maps["lgbm_logit"].get(rk, 0.0))
            except Exception:
                lg = 0.0
            out["lgbm_logit"] = torch.tensor([[lg]], dtype=torch.float32)
        else:
            out["lgbm_logit"] = torch.tensor([[0.0]], dtype=torch.float32)

        for k, mp in self.logit_maps.items():
            if k == "lgbm_logit":
                continue
            if not isinstance(k, str) or not k.endswith("_logit"):
                continue
            try:
                v = float(mp.get(rk, 0.0))
            except Exception:
                v = 0.0
            out[k] = torch.tensor([[v]], dtype=torch.float32)

    # =================================================================
    # Event token injection
    # =================================================================
    @staticmethod
    def _inject_event_tokens(
        out: Dict[str, Any],
        raw: Dict[str, Any],
    ) -> None:
        """Transfer event tokens from raw pipeline output to sample dict.

        Each event token e_k is a tuple:
          e_k = (type_k ∈ ℤ_V,  actor_k ∈ {0..10},  team_k ∈ {0,1,2},
                 c_k ∈ ℝ^5,  m_k ∈ {0,1})
        padded to K = max_tokens fixed-length tensors.
        """
        for ek in _EVENT_TOKEN_KEYS:
            v = raw.get(ek, None)
            if v is None:
                continue
            if isinstance(v, np.ndarray):
                out[ek] = torch.from_numpy(v).to(dtype=_EVENT_TOKEN_DTYPES.get(ek, torch.float32))
            elif isinstance(v, torch.Tensor):
                out[ek] = v.to(dtype=_EVENT_TOKEN_DTYPES.get(ek, torch.float32))
            else:
                out[ek] = torch.as_tensor(v, dtype=_EVENT_TOKEN_DTYPES.get(ek, torch.float32))


# =====================================================================
# Collate function
# =====================================================================
def collate_batch(batch: List[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """Collate a list of samples into a batched dict.

    Stacking strategy:
        node_seq  → torch.stack → (B, T, N, F)
        macro_seq → torch.stack → (B, T, D)
        tab_x     → torch.cat   → (B, D_tab)
        y         → torch.cat   → (B, 1)
        *_logit   → torch.cat   → (B, 1)
        event_*   → torch.stack → (B, K, ...)

    Alignment invariant:
        Every sample carries "ref_key" ∈ str.  This is the *only*
        mechanism to align predictions ↔ FightRefs.  Positional
        alignment is intentionally forbidden (shuffle + drop can
        silently permute ordering).
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    # ---- ref_key integrity check ----
    ref_keys = [b.get("ref_key") for b in batch]
    if any(k is None for k in ref_keys):
        raise RuntimeError("collate_batch: missing ref_key in some samples. "
                           "Make sure __getitem__ injects sample['ref_key'].")

    y = torch.cat([b["y"] for b in batch], dim=0)
    out: Dict[str, Any] = {"y": y, "ref_key": [str(k) for k in ref_keys]}

    aux_target_keys = sorted({
        k for b in batch
        for k in b.keys()
        if isinstance(k, str) and k.startswith("y_")
    })
    for k in aux_target_keys:
        vals = []
        for b in batch:
            if k in b:
                vals.append(_as_1x1(b[k]))
            else:
                vals.append(torch.zeros((1, 1), dtype=torch.float32))
        out[k] = torch.cat(vals, dim=0)

    # ---- feature tensors ----
    if all(("macro_seq" in b) for b in batch):
        out["node_seq"] = torch.stack([b["node_seq"] for b in batch], dim=0)
        out["macro_seq"] = torch.stack([b["macro_seq"] for b in batch], dim=0)
        out["tab_x"] = torch.cat([b["tab_x"] for b in batch], dim=0)

    elif all(("node_seq" in b) and ("extra_seq" in b) for b in batch):
        out["node_seq"] = torch.stack([b["node_seq"] for b in batch], dim=0)
        out["extra_seq"] = torch.stack([b["extra_seq"] for b in batch], dim=0)

    else:
        out["x_seq"] = torch.stack([b["x_seq"] for b in batch], dim=0)

    # ---- event tokens ----
    for ek in _EVENT_TOKEN_KEYS:
        if all(ek in b for b in batch):
            out[ek] = torch.stack([b[ek] for b in batch], dim=0)

    # ---- logit features ----
    logit_keys = sorted({
        k for b in batch
        for k in b.keys()
        if isinstance(k, str) and k.endswith("_logit")
    })

    for k in logit_keys:
        vals = []
        for b in batch:
            if k in b:
                vals.append(_as_1x1(b[k]))
            else:
                vals.append(torch.zeros((1, 1), dtype=torch.float32))
        out[k] = torch.cat(vals, dim=0)  # (B,1)

    # ---- lgbm_logit fallback ----
    if _cfg_wants_logits("") and ("lgbm_logit" not in out):
        out["lgbm_logit"] = torch.zeros((len(batch), 1), dtype=torch.float32)

    return out


def _as_1x1(t) -> torch.Tensor:
    """Coerce any scalar-like tensor to shape (1, 1)."""
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=torch.float32)
    t = t.to(dtype=torch.float32)
    if t.dim() == 0:
        return t.view(1, 1)
    if t.dim() == 1:
        return t.view(1, 1)
    if t.dim() >= 2:
        return t.reshape(-1)[0].view(1, 1)
    return t.view(1, 1)
