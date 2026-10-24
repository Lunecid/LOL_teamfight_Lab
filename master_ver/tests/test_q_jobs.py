"""q job plan: protocol pass counts, unique ids, valid anchors, identity reuse after selection."""
from studies.q_production_20261005 import jobs as qjobs, qselect


def test_pass_counts_match_the_protocol_with_old_v_and_ft_removed():
    plan = qjobs.plan()
    # 148 / 353 in the cards; old_V_h90 not executable (D-Q5: -6), FT removed after its cost gate (D-Q15: -8 search, -2 final)
    assert qjobs.counts(plan) == {"search": 148 - 8, "final": 353 - 6 - 2, "pure_set": 16}
    assert len([j for j in plan["search"] if j.block == "common"]) == 40
    assert len([j for j in plan["final"] if j.block == "module"]) == 155
    assert not [j for group in plan.values() for j in group if j.family in qselect.EXCLUDED_FAMILIES]
    assert set(qjobs.NOT_EXECUTED_FAMILIES) == {"ft_transformer"}


def test_final_jobs_reuse_search_results_by_exact_identity():
    plan = qjobs.plan()
    selected = {"common": {"id": "q_shared_h64_wd0.001", "hidden_width": 64, "weight_decay": 0.001}}
    for job in plan["search"]:
        if job.block == "auxiliary" and job.card_id.endswith(("0.001", "m100", "_l2")):
            selected.setdefault(job.family, {"id": job.card_id, **dict(job.card)})
    for name, card_id in (("time_only", "lr_C0.01"), ("p_only", "lr_C0.01"), ("PT_linear", "lr_C0.01")):
        selected[name] = {"id": card_id, "C": 0.01}
    selected["PT_flex"] = {"id": "ptflex_k6_C0.01", "p_knots": 6, "time_knots": 4, "degree": 3, "C": 0.01}
    resolved = qjobs.resolve_final(plan, selected)
    reused = [j for j in resolved if j.reuse_of]
    # 10 anchor cells of the common card, 5 full-input auxiliary results (FT removed), 4 reference cards
    assert len(reused) == 19
    assert all(not j.card_id.startswith("selected:") for j in resolved)
    assert sum(j.passes for j in resolved if j.reuse_of is None) == 345 - sum(j.passes for j in reused)
