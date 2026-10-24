"""q selection on Q_DEV raw match-weighted Brier (ARCHITECTURE_KO.md section 4.6; EXECUTION_CARDS.json
``q.common_search``, ``q.full_family_selection``; DECISIONS D-Q1, D-Q9).

  - common card: one card for all 132 cells; a card is eligible only when all 10 anchor x G fits
    completed on identical Q_DEV keys; its score is the mean of the 10 Q_DEV Briers (fixed
    denominator 10); ties go to card order; no eligible card -> search incomplete;
  - within a family (auxiliary families, references, pure-set per S): lowest Q_DEV Brier among
    completed candidates, ties by card order;
  - families: lowest Q_DEV Brier of each family's full-input result; families within
    best + 0.0005 (inclusive) are tied and resolved by the fixed simplicity order; complete when
    every family succeeded, except that a non-converged logistic is a recorded terminal failure
    excluded from selection (D-Q1).  FT-Transformer was removed from the set before any q fit
    after its cost gate (D-Q15), so seven families are selected among.
Results are dicts as written by the worker: status, scores.q_dev.raw_match_brier, q_dev_keys_sha256.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

FAMILY_ORDER = ("logistic", "lightgbm", "plain_mlp", "shared_slot_mlp", "team_set_pair_readout", "graphsage", "ft_transformer", "relational_mpnn")
# Removed before any q fit after its cost gate (DECISIONS_20261005.json D-Q15); the declared order is kept for the record.
EXCLUDED_FAMILIES = {"ft_transformer": "cost gate: 1,942 tokens, 0.8-1.6 h per FIT epoch at batch 16 on the RTX 5080 (D-Q15)"}
SELECTION_ORDER = tuple(family for family in FAMILY_ORDER if family not in EXCLUDED_FAMILIES)
FAMILY_TIE = 0.0005


def _brier(result: Mapping[str, Any] | None) -> float | None:
    if not result or result.get("status") != "completed":
        return None
    return float(result["scores"]["q_dev"]["raw_match_brier"])


def select_common(results: Mapping[tuple[str, str], Mapping[str, Any]], anchors: Sequence[str], card_order: Sequence[str]) -> dict[str, Any]:
    """``results[(condition_id, card_id)]`` over the anchor x G conditions."""

    table = {}
    for card in card_order:
        cells = [results.get((condition, card)) for condition in anchors]
        briers = [_brier(cell) for cell in cells]
        keys = {cell.get("q_dev_keys_sha256") for cell in cells if cell}
        eligible = all(b is not None for b in briers) and len(keys) == 1 and len(briers) == len(anchors)
        table[card] = {"eligible": eligible, "fits": len([b for b in briers if b is not None]),
                       "score": sum(briers) / len(anchors) if eligible else None}
    eligible = [card for card in card_order if table[card]["eligible"]]
    if not eligible:
        return {"complete": False, "selected": None, "cards": table, "reason": "no card has all anchor fits on identical Q_DEV keys"}
    best = min(eligible, key=lambda card: (table[card]["score"], list(card_order).index(card)))
    return {"complete": True, "selected": best, "cards": table, "denominator": len(anchors)}


def select_within(results: Mapping[str, Mapping[str, Any]], card_order: Sequence[str]) -> dict[str, Any]:
    """``results[card_id]`` for one family or reference."""

    table = {card: {"status": (results.get(card) or {}).get("status", "missing"), "brier": _brier(results.get(card))} for card in card_order}
    done = [card for card in card_order if table[card]["brier"] is not None]
    if not done:
        return {"complete": False, "selected": None, "cards": table}
    best = min(done, key=lambda card: (table[card]["brier"], list(card_order).index(card)))
    return {"complete": len(done) == len(card_order), "selected": best, "cards": table, "brier": table[best]["brier"]}


def select_family(representatives: Mapping[str, Mapping[str, Any] | None], logistic_attempts: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """``representatives[family]``: the family's full-input result (None when it never produced one).

    Families in EXCLUDED_FAMILIES (D-Q15) are not candidates and do not count as missing."""

    briers = {family: _brier(representatives.get(family)) for family in SELECTION_ORDER}
    keys = {r.get("q_dev_keys_sha256") for f, r in representatives.items() if f in briers and r and r.get("status") == "completed"}
    logistic_terminal = briers["logistic"] is None and bool(logistic_attempts) and all(
        a.get("status") == "failed" and a.get("reason") == "nonconverged" for a in logistic_attempts)
    missing = [f for f in SELECTION_ORDER if briers[f] is None and not (f == "logistic" and logistic_terminal)]
    candidates = [f for f in SELECTION_ORDER if briers[f] is not None]
    excluded = dict(EXCLUDED_FAMILIES)
    if not candidates:
        return {"complete": False, "selected": None, "briers": briers, "missing": missing, "excluded_families": excluded}
    best_value = min(briers[f] for f in candidates)
    tied = [f for f in candidates if briers[f] <= best_value + FAMILY_TIE]
    selected = min(tied, key=SELECTION_ORDER.index)
    return {"complete": not missing and len(keys) == 1, "selected": selected, "briers": briers, "tied": tied,
            "tie_tolerance": FAMILY_TIE, "missing": missing, "identical_q_dev_keys": len(keys) == 1,
            "logistic_terminal_nonconvergence": logistic_terminal, "excluded_families": excluded}
