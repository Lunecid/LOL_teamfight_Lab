"""q job plan from the hash-locked cards and comparison matrix (ARCHITECTURE_KO.md section 4.5).

Search (first seed; EXECUTION_CARDS.json planned_counts 148, 140 after D-Q15):
  common card: 5 anchors x G {0,1} x 4 q_shared cards (x2 passes)            = 80
  auxiliary families on full input: logistic 4 (x1) + 4 families x 4 cards (x2) = 36
  references: time_only 4 + p_only 4 + PT_linear 4 + PT_flex 12 (x1)          = 24
Final (one seed): 155 module cells (FULL_LOGISTIC x1, others x2) = 309, tied-message 2 x2 = 4,
references 4 logistic-type x1 + B / B_minus_map LightGBM x2 = 8, sensitivity variants x full_G0 /
full_G1 / full_LightGBM x2.  ``old_V_h90`` is recorded as not executable (DECISIONS D-Q5), so the
sensitivity block is 4 x 3 x 2 = 24 and the final total 345 (353 in the cards).  FT-Transformer
(4 search cards and the FULL_FT_TRANSFORMER cell) was removed before any q fit after its cost gate
(DECISIONS D-Q15) and is listed in NOT_EXECUTED_FAMILIES.  Pure-set auxiliary:
2 S conditions x 4 cards x 2 = 16 (research_execution CONTRACT pure_set_auxiliary).
Final jobs carry ``selected:<family>`` card placeholders until selection; a final job whose resolved
identity equals a search job reuses that result (identity deduplication, research_execution CONTRACT).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from studies.q_production_20261005.qselect import EXCLUDED_FAMILIES

ROOT = Path(__file__).resolve().parents[2]
CARDS = ROOT / "reports/feature_control_study_20261001/EXECUTION_CARDS.json"
MATRIX = ROOT / "reports/feature_control_study_20261001/CARTESIAN_COMPARISON_MATRIX.json"
CARDS_SHA256 = "5dcf8209c0801661bb52b338ef794ad5de942a59626c610aef9eda9d3044b34a"
MATRIX_SHA256 = "7fe1f9af6a7e8b0a457d4dcbe73ed156a707293dcade069f2e7fc46217eb414f"
SEED = 20261001
VERSION = "q_jobs_v1_20261005"
FULL = {"S": 1, "A": 1, "M": 1, "E": 1, "C": 1, "R": 1, "D": 1}
FAMILY_ALIASES = {"legacy_shared_encoder_mlp_new_adapter": "shared_slot_mlp", "heterogeneous_relational_mpnn": "relational_mpnn"}
FULL_CONDITIONS = {"logistic": "FULL_LOGISTIC", "lightgbm": "LGBM_S1_A1_M1_E1_C1_R1_D1", "plain_mlp": "FULL_PLAIN_MLP",
                   "shared_slot_mlp": "FULL_LEGACY_SHARED_ENCODER_MLP_NEW_ADAPTER", "graphsage": "SAGE_S1_A1_M1_E1_C1_R1_D1",
                   "ft_transformer": "FULL_FT_TRANSFORMER"}
SENSITIVITY_VARIANTS = ("new_V_h60", "new_V_h120", "exclude_near_zero_h90", "exclude_zero_coordinate_h90")
NOT_EXECUTABLE = {"old_V_h90": "its V is the superseded StateV3 lineage; recorded as not run (DECISIONS_20261005.json D-Q5)"}
NOT_EXECUTED_FAMILIES = {family: f"{reason}; its search cards and full-input cell are not run" for family, reason in EXCLUDED_FAMILIES.items()}
PURE_SET_CARDS = ({"id": "pure_h64_wd0.001", "hidden_width": 64, "weight_decay": 0.001}, {"id": "pure_h64_wd0.0001", "hidden_width": 64, "weight_decay": 0.0001},
                  {"id": "pure_h128_wd0.001", "hidden_width": 128, "weight_decay": 0.001}, {"id": "pure_h128_wd0.0001", "hidden_width": 128, "weight_decay": 0.0001})


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_sources() -> tuple[dict[str, Any], dict[str, Any]]:
    for path, expected in ((CARDS, CARDS_SHA256), (MATRIX, MATRIX_SHA256)):
        if _sha256(path) != expected:
            raise ValueError(f"{path.name} differs from its recorded hash")
    return json.loads(CARDS.read_text(encoding="utf-8")), json.loads(MATRIX.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Job:
    job_id: str
    stage: str                 # search | final | pure_set
    block: str                 # common | auxiliary | reference | module | tied | sensitivity | pure_set
    condition_id: str
    family: str
    card_id: str               # a card id, "selected:<family>" before resolution, or "closed_form"
    card: tuple                # sorted (key, value) pairs
    factors: tuple             # sorted (factor, level) pairs; G may be None
    passes: int                # optimizer passes counted by the protocol
    variant: str = "main_h90"
    inputs: str = "condition"  # condition | B | B_minus_map | pt
    reuse_of: str | None = None

    @property
    def identity(self) -> str:
        payload = {"condition": self.condition_id, "family": self.family, "card": dict(self.card), "factors": dict(self.factors),
                   "variant": self.variant, "inputs": self.inputs, "seed": SEED}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out.update(card=dict(self.card), factors=dict(self.factors), identity=self.identity)
        return out


def _job(stage, block, condition, family, card, factors, passes, **extra) -> Job:
    card_id = card if isinstance(card, str) else card["id"]
    card_items = () if isinstance(card, str) else tuple(sorted((k, v) for k, v in card.items() if k != "id"))
    variant = extra.get("variant", "main_h90")
    job_id = f"{stage}/{block}/{condition}/{family}/{card_id}" + ("" if variant == "main_h90" else f"/{variant}")
    return Job(job_id, stage, block, condition, family, card_id, card_items, tuple(sorted(factors.items())), passes, **extra)


def _neural_conditions(matrix: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {c["condition_id"]: c for c in matrix["neural_conditions"]}


def _condition_id(g: int, factors: Mapping[str, int]) -> str:
    return "N_G{G}_S{S}_A{A}_M{M}_E{E}_C{C}_R{R}_D{D}".format(G=g, **factors)


def plan() -> dict[str, list[Job]]:
    cards, matrix = load_sources()
    q = cards["q"]
    neural = _neural_conditions(matrix)
    aux = {c["condition_id"]: c for c in matrix["auxiliary_conditions"]}
    if len(neural) != 132 or len(aux) != 24:
        raise ValueError("matrix counts differ from 132 neural / 24 auxiliary conditions")
    family_cards = {FAMILY_ALIASES.get(f["family"], f["family"]): f["cards"] for f in q["auxiliary_families"]}
    graph_family = {0: "team_set_pair_readout", 1: "relational_mpnn"}
    search: list[Job] = []
    for anchor in q["common_search"]["anchors"]:
        factors = {"E": 1, "C": 1, "R": 1, **{k: int(v) for k, v in anchor.items()}}
        for g in (0, 1):
            condition = _condition_id(g, factors)
            if condition not in neural:
                raise ValueError(f"anchor {condition} is not a valid matrix cell")
            for card in q["common_search"]["cards"]:
                search.append(_job("search", "common", condition, graph_family[g], card, {"G": g, **factors}, 2))
    for family, condition in FULL_CONDITIONS.items():
        if family in EXCLUDED_FAMILIES:
            continue
        if condition not in aux:
            raise ValueError(f"{condition} missing from the auxiliary conditions")
        for card in family_cards[family]:
            search.append(_job("search", "auxiliary", condition, family, card, {"G": None, **FULL}, 1 if family == "logistic" else 2))
    references = q["references"]
    for name in ("time_only", "p_only", "PT_linear"):
        for card in references[name]["cards"]:
            search.append(_job("search", "reference", name.upper(), name, card, {}, 1, inputs="pt"))
    for knots in references["PT_flex"]["p_knots"]:
        for c in references["PT_flex"]["Cs"]:
            card = {"id": f"ptflex_k{knots}_C{c:g}", "p_knots": int(knots), "time_knots": int(references["PT_flex"]["time_knots"]),
                    "degree": int(references["PT_flex"]["degree"]), "C": float(c)}
            search.append(_job("search", "reference", "PT_FLEX", "PT_flex", card, {}, 1, inputs="pt"))
    final: list[Job] = []
    for condition, cell in neural.items():
        factors = {k: (None if v is None else int(v)) for k, v in cell["factors"].items()}
        family = FAMILY_ALIASES.get(cell["algorithm"], cell["algorithm"])
        final.append(_job("final", "module", condition, family, "selected:common", factors, 2))
    for condition, cell in aux.items():
        family = FAMILY_ALIASES.get(cell["algorithm"], cell["algorithm"])
        if family in EXCLUDED_FAMILIES:
            continue
        factors = {k: (None if v is None else int(v)) for k, v in cell["factors"].items()} if cell.get("factors") else {"G": None, **FULL}
        final.append(_job("final", "module", condition, family, f"selected:{family}", factors, 1 if family == "logistic" else 2))
    for s in (0, 1):
        factors = {"G": 1, "S": s, "A": 0, "M": 0, "E": 1, "C": 1, "R": 1, "D": 1}
        final.append(_job("final", "tied", f"TIED_S{s}_A0_M0_E1_C1_R1_D1", "tied_message", "selected:common", factors, 2))
    for name in ("time_only", "p_only", "PT_linear", "PT_flex"):
        final.append(_job("final", "reference", name.upper(), name, f"selected:{name}", {}, 1, inputs="pt"))
    final.append(_job("final", "reference", "CONSTANT", "constant", "closed_form", {}, 0, inputs="pt"))
    for inputs in ("B", "B_minus_map"):
        final.append(_job("final", "reference", f"{inputs}_LIGHTGBM", "lightgbm", "selected:lightgbm", {}, 2, inputs=inputs))
    sensitivity_models = {"full_G0": ("team_set_pair_readout", _condition_id(0, FULL), "selected:common"),
                          "full_G1": ("relational_mpnn", _condition_id(1, FULL), "selected:common"),
                          "full_LightGBM": ("lightgbm", FULL_CONDITIONS["lightgbm"], "selected:lightgbm")}
    declared = [v["id"] for v in cards["sensitivities"]["variants"]]
    if set(declared) != set(SENSITIVITY_VARIANTS) | set(NOT_EXECUTABLE):
        raise ValueError(f"sensitivity variants differ from the cards: {declared}")
    for variant in SENSITIVITY_VARIANTS:
        for model, (family, condition, card) in sensitivity_models.items():
            g = 0 if model == "full_G0" else 1 if model == "full_G1" else None
            final.append(_job("final", "sensitivity", condition, family, card, {"G": g, **FULL}, 2, variant=variant))
    pure = []
    for s in (0, 1):
        factors = {"G": None, "S": s, "A": 0, "M": 0, "E": 1, "C": 1, "R": 1, "D": 1}
        for card in PURE_SET_CARDS:
            pure.append(_job("pure_set", "pure_set", f"PURE_S{s}_E1_C1_R1_D1", "pure_set", card, factors, 2))
    jobs = {"search": search, "final": final, "pure_set": pure}
    ids = [job.job_id for group in jobs.values() for job in group]
    if len(set(ids)) != len(ids):
        raise ValueError("job ids are not unique")
    return jobs


def counts(jobs: Mapping[str, list[Job]]) -> dict[str, int]:
    return {name: sum(job.passes for job in group) for name, group in jobs.items()}


def resolve_final(jobs: Mapping[str, list[Job]], selected_cards: Mapping[str, Mapping[str, Any]]) -> list[Job]:
    """Fill ``selected:<family>`` placeholders and mark exact-identity reuse of search results."""

    by_identity = {job.identity: job.job_id for job in jobs["search"]}
    resolved = []
    for job in jobs["final"]:
        if job.card_id.startswith("selected:"):
            card = dict(selected_cards[job.card_id.split(":", 1)[1]])
            job = replace(job, card_id=card["id"], card=tuple(sorted((k, v) for k, v in card.items() if k != "id")))
        resolved.append(replace(job, reuse_of=by_identity.get(job.identity)))
    return resolved
