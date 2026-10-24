import os
"""Validate all artifacts in multipatch_v5_prep and ensure 100% consistency with integrated_design_v5."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PREP_ROOT = Path(os.environ["LOL_ARTIFACT_ROOT"]) / 'multipatch_v5_prep'
KNOW_DIR = PREP_ROOT / "knowledge_multipatch_v5"
DATA_DIR = PREP_ROOT / "data_curation_v5"
DIAG_DIR = PREP_ROOT / "architecture_diagnostics_v5"


def read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    checks = 0

    def check(cond: bool, msg: str) -> None:
        nonlocal checks
        if not cond:
            raise AssertionError(f"CHECK FAILED: {msg}")
        checks += 1

    # 1. Required files exist
    req_files = [
        KNOW_DIR / "PATCH_KNOWLEDGE_INDEX_V5.json",
        KNOW_DIR / "CROSS_PATCH_DIFF_LEDGER_V5.json",
        KNOW_DIR / "SEMANTIC_PROPAGATION_15_18_22.json",
        KNOW_DIR / "P05_DYNAMIC_RANK_EXCEPTION_RULES.json",
        KNOW_DIR / "bundles" / "bundle_15_18_v2.json",
        KNOW_DIR / "bundles" / "bundle_15_19_v2.json",
        KNOW_DIR / "bundles" / "bundle_15_20_v2.json",
        KNOW_DIR / "bundles" / "bundle_15_21_v2.json",
        KNOW_DIR / "bundles" / "bundle_15_22_v2.json",
        KNOW_DIR / "bundles" / "bundle_16_13_v2.json",
        KNOW_DIR / "bundles" / "bundle_16_14_v2.json",
        KNOW_DIR / "bundles" / "bundle_16_15_v2.json",
        DATA_DIR / "CURATED_DATASET_MANIFEST_V5.json",
        DATA_DIR / "CURATED_40K_SPLIT_ASSIGNMENTS.jsonl",
        DIAG_DIR / "ARCHITECTURE_DIAGNOSTICS_V5.json",
        PREP_ROOT / "FULL_PREP_AND_DIAGNOSTIC_REPORT_KO.md",
        PREP_ROOT / "PREP_ARTIFACT_MANIFEST.json",
    ]
    for rf in req_files:
        check(rf.exists() and rf.stat().st_size > 0, f"Required file exists: {rf.name}")

    # 2. Knowledge package checks (15.18-15.22 & 16.13-16.15)
    p_idx = read_json(KNOW_DIR / "PATCH_KNOWLEDGE_INDEX_V5.json")
    diff_led = read_json(KNOW_DIR / "CROSS_PATCH_DIFF_LEDGER_V5.json")
    sem_prop = read_json(KNOW_DIR / "SEMANTIC_PROPAGATION_15_18_22.json")
    p05 = read_json(KNOW_DIR / "P05_DYNAMIC_RANK_EXCEPTION_RULES.json")

    for p in ["15.18", "15.19", "15.20", "15.21", "15.22"]:
        check(p in p_idx["ddragon_sources_by_patch"], f"DataDragon source indexed for {p}")
        check(p_idx["ddragon_sources_by_patch"][p]["canonical_champion_entries"] == 171, f"171 canonical champions in {p}")
        b_meta = p_idx["generated_v2_bundles_15_18_to_15_22"][p]
        b_path = PREP_ROOT / b_meta["path"]
        check(sha256_file(b_path) == b_meta["sha256"], f"Bundle sha256 matches for {p}")
        b_doc = read_json(b_path)
        check(len(b_doc["champions"]) == 171, f"Bundle {p} has 171 champions")
        check(sum(len(c["skills"]) for c in b_doc["champions"]) == 855, f"Bundle {p} has 855 skills")

        skill_map = {s["skill_key"]: s for c in b_doc["champions"] for s in c["skills"]}
        aatrox_p_keys = [m["key"] for m in skill_map["Aatrox.PASSIVE"]["mechanic_facts"]]
        briar_q_keys = [m["key"] for m in skill_map["Briar.Q"]["mechanic_facts"]]
        azir_w_keys = [m["key"] for m in skill_map["Azir.W"]["mechanic_facts"]]
        reksai_e_keys = [m["key"] for m in skill_map["RekSai.E"]["mechanic_facts"]]
        reksai_r_keys = [m["key"] for m in skill_map["RekSai.R"]["mechanic_facts"]]
        graves_e_keys = [m["key"] for m in skill_map["Graves.E"]["mechanic_facts"]]

        check("semantic.damage.magic.0" in aatrox_p_keys and "semantic.damage.physical.0" not in aatrox_p_keys, f"Aatrox.PASSIVE magic damage updated in {p}")
        check("semantic.damage.physical.2" in briar_q_keys and "semantic.damage.magic.2" not in briar_q_keys, f"Briar.Q physical damage updated in {p}")
        check("semantic.attack_modifier.attack_speed.5" not in azir_w_keys, f"Azir.W attack_speed removed in {p}")
        check("semantic.damage.true.1" in reksai_e_keys and "semantic.scaling.target_max_health.2" not in reksai_e_keys, f"RekSai.E true damage updated in {p}")
        check("semantic.scaling.target_max_health.6" in reksai_r_keys and "semantic.resource.cooldown_reset.7" in reksai_r_keys, f"RekSai.R max_health + cooldown_reset updated in {p}")
        if p in ("15.21", "15.22"):
            check("semantic.defense.magic_resist.5" in graves_e_keys, f"Graves.E magic_resist added in {p}")
        else:
            check("semantic.defense.magic_resist.5" not in graves_e_keys, f"Graves.E magic_resist not present in {p}")

        sp = sem_prop["by_patch"][p]
        check(sp["evaluated_skills"] == 855, f"855 skills evaluated in semantic propagation for {p}")
        check(sp["total_claims"] == 3901, f"3901 claims evaluated in semantic propagation for {p}")
        check(sp["verbatim_quote_survival_rate"] > 0.98, f"Quote survival rate >98% in {p}")

    for p16 in ["16.13", "16.14", "16.15"]:
        check(p16 in p_idx["ddragon_sources_by_patch"], f"DataDragon source indexed for {p16}")
        check(p_idx["ddragon_sources_by_patch"][p16]["canonical_champion_entries"] == 173, f"173 canonical champions in {p16}")
        b_meta16 = p_idx["generated_v2_bundles_16_13_to_16_15"][p16]
        b_path16 = PREP_ROOT / b_meta16["path"]
        check(sha256_file(b_path16) == b_meta16["sha256"], f"Bundle sha256 matches for {p16}")
        b_doc16 = read_json(b_path16)
        check(len(b_doc16["champions"]) == 173, f"Bundle {p16} has 173 champions (+Locke, +Zaahen)")
        check(sum(len(c["skills"]) for c in b_doc16["champions"]) == 865, f"Bundle {p16} has 865 skills")
        skill_map16 = {s["skill_key"]: s for c in b_doc16["champions"] for s in c["skills"]}
        check("Locke.R" in skill_map16 and "Zaahen.R" in skill_map16, f"Locke and Zaahen skills present in {p16}")
        check("semantic.sustain.omnivamp.5" in [m["key"] for m in skill_map16["LeeSin.W"]["mechanic_facts"]], f"LeeSin.W omnivamp updated in {p16}")
        check("semantic.cc.fear.5" in [m["key"] for m in skill_map16["Shyvana.R"]["mechanic_facts"]], f"Shyvana.R fear updated in {p16}")

    # Verify 60,063-match P05 census
    check(len(p05["deterministic_producer_rules"]) == 7, "7 deterministic P05 rules locked")
    c60k = p05["empirical_discovery_summary"]["full_60063_match_pid0_census"]
    check(c60k["kr_15_18_22"]["total_matches"] == 40000, "40,000 matches in kr_15_18_22 P05 census")
    check(c60k["kr_15_18_22"]["matches_with_pid0_skill_level_up"] == 17688, "17,688 matches with pid=0 in kr_15_18_22")
    check(c60k["kr_15_18_22"]["unexplained_matches"] == 0, "0 unexplained matches in kr_15_18_22")
    check(c60k["kr_15_18_22"]["affected_queries_unk_entries"] == 366192, "366,192 unk_entries in kr_15_18_22")
    check(c60k["kr_15_18_22"]["affected_queries_unk_weighted"] == 608123, "608,123 unk_weighted in kr_15_18_22")
    check(c60k["kr_16_13"]["unexplained_matches"] == 0 and c60k["kr_16_13"]["affected_queries_unk_weighted"] == 191072, "kr_16_13 pid=0 census exact")
    check(c60k["na1_16_13"]["unexplained_matches"] == 0 and c60k["na1_16_13"]["affected_queries_unk_weighted"] == 108645, "na1_16_13 pid=0 census exact")
    check(diff_led["training_variation_audit_15_18_to_15_20"]["maxrank_changes_across_15_14_to_15_22"] == 0, "0 maxrank changes across 15.14-15.22")

    # 3. Data curation & E1 provenance & IL-1..5 checks
    cur = read_json(DATA_DIR / "CURATED_DATASET_MANIFEST_V5.json")
    e1_prov = cur["cache_provenance_audit_e1"]
    check(e1_prov["T_equals_raw_frames_pass"] == 40000, "40,000/40,000 T_equals_raw_frames pass")
    check(e1_prov["rehash_matches_build_manifest"] == 120000, "120,000/120,000 rehash matches")
    check(e1_prov["determinism_fresh_rebuild_25_of_25_bytes_equal"] is True, "25/25 rebuild byte determinism")
    check(e1_prov["identifier_token_hits"] == 0, "0 identifier token hits")

    l1 = cur["level_1_full_raw_corpus_curation_15_18_to_15_22"]
    l2 = cur["level_2_indexed_40k_bank_curation_15_18_to_15_22"]

    check(l1["raw_detail_files_on_disk"] == 301744, "301,744 raw detail files audited")
    check(l1["clean_trainable_matches_ge_300s_total"] == 298218, "298,218 clean raw matches >=300s")
    check(l2["indexed_matches_total"] == 40000, "40,000 indexed matches audited")
    check(l2["excluded_remake_or_early_surrender_lt_300s_total"] + l2["structural_valid_ge_300s_total"] == 40000, "443 + 39,557 == 40,000")
    check(l2["clean_combat_trainable_matches_allphase"] == 39557, "39,557 clean trainable matches >=300s")
    check(l2["clean_q_fresh_10s_eligible_matches"] == 39367, "39,367 clean Q_fresh_10s matches")
    check(l2["clean_q_fresh_isolated_engagement_eligible_matches"] == 38410, "38,410 clean isolated engagement matches")

    il_aud = l2["item_ledger_audit"]
    check(il_aud["unassigned_item_events_exact_identity"]["kr_15_18_22_unassigned_events"] == 80000, "80,000 unassigned item events")
    check(il_aud["unassigned_item_events_exact_identity"]["100pct_single_event_signature"]["itemId"] == 3865, "itemId 3865 World Atlas")
    check(il_aud["empirical_500_match_replay_ablation"]["il1_to_il4_supported_blocks"] == 148177, "148,177 / 148,249 blocks supported under IL-1..IL-4")
    check(il_aud["empirical_500_match_replay_ablation"]["il1_to_il5_with_2055_clamp_supported_blocks"] == 148249, "148,249 / 148,249 blocks supported under IL-1..IL-5")
    check(il_aud["empirical_500_match_replay_ablation"]["viego_possession_standalone_destroy_underflows_in_500_matches"] == 6741, "6,741 Viego possession underflows quantified")

    # Verify JSONL row count and split partition completeness
    jsonl_path = DATA_DIR / "CURATED_40K_SPLIT_ASSIGNMENTS.jsonl"
    check(sha256_file(jsonl_path) == l2["assignments_jsonl_sha256"], "JSONL sha256 matches manifest")
    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    check(len(rows) == 40000, "40,000 rows in CURATED_40K_SPLIT_ASSIGNMENTS.jsonl")
    check(len({r["match_id"] for r in rows}) == 40000, "40,000 unique match_ids in JSONL")
    check(sum(1 for r in rows if r["is_clean_combat_trainable"]) == 39557, "JSONL clean_combat_trainable count == 39,557")
    check(sum(1 for r in rows if r["in_matched_N_allphase_multi"]) == 6321, "JSONL matched_N_allphase_multi == 6,321 (2,107 * 3)")
    check(sum(1 for r in rows if r["in_matched_N_allphase_single"]) == 6321, "JSONL matched_N_allphase_single == 6,321")
    check(sum(1 for r in rows if r["in_matched_N_fresh10s_multi"]) == 6300, "JSONL matched_N_fresh10s_multi == 6,300 (2,100 * 3)")
    check(sum(1 for r in rows if r["in_matched_N_fresh10s_single"]) == 6300, "JSONL matched_N_fresh10s_single == 6,300")
    check(sum(1 for r in rows if r["in_matched_N_fresh_isolated_multi"]) == 6144, "JSONL matched_N_fresh_isolated_multi == 6,144 (2,048 * 3)")
    check(sum(1 for r in rows if r["in_matched_N_fresh_isolated_single"]) == 6144, "JSONL matched_N_fresh_isolated_single == 6,144")

    # 4. Architecture diagnostics checks
    sc = l2["v5_split_counts"]
    check(sum(sc["queries_allphase_by_split"].values()) == sum(r["n_queries_allphase"] for r in rows if r["is_valid_structural"]) == 833153, "clean all-phase queries == 833,153")
    check(sum(sc["queries_fresh_10s_by_split"].values()) == 250215, "clean fresh10s queries == 250,215")
    check(sum(sc["supported_primary_engagements_by_split"].values()) == 146831, "clean isolated engagements == 146,831")
    check(sc["clean_q_fresh_10s_eligible"]["FIT_15.20"] == 6301 and sc["clean_q_fresh_10s_eligible"]["TEST_15.22"] == 7864, "per-split fresh10s eligibility matches report")
    pdl = diff_led["pairwise_diffs"]["15.18_to_15.22"]
    check(pdl["n_champions_stat_changed"] == 23 and pdl["n_spells_numeric_changed"] == 11, "15.18->15.22 diff == 23 champs / 11 spells")
    fv = diff_led["training_variation_audit_15_18_to_15_20"]
    check(fv["n_champions_with_stat_variation_in_fit"] == 10 and fv["n_spells_with_numeric_variation_in_fit"] == 5, "FIT variation == 10 champs / 5 spells")
    check("73 spell" not in json.dumps(diff_led), "no stale 73-spell claim in ledger")

    report = (PREP_ROOT / "FULL_PREP_AND_DIAGNOSTIC_REPORT_KO.md").read_text(encoding="utf-8")
    for token in [
        "298,218", "39,557", "833,153", "250,215", "146,831",
        "6,321", "6,300", "6,144", "90.44%", "60,063", "366,192",
        "608,123", "3865", "6,741", "99.9514%", "-0.00180", "+0.00357",
    ]:
        check(token in report, f"report contains verified number {token}")

    diag = read_json(DIAG_DIR / "ARCHITECTURE_DIAGNOSTICS_V5.json")
    check("73 spell" not in json.dumps(diag), "no stale 73-spell claim in diagnostics")
    check(len(diag["six_layer_architecture_diagnosis"]) == 6, "6 architectural layers diagnosed")
    l6 = diag["six_layer_architecture_diagnosis"][5]
    check(len(l6["required_upgrades_before_training"]) == 5, "5 required prototype upgrades documented")
    check(all(v is True for v in l6["code_inspection_checks"].values()), "All 10 code_inspection_checks on model.py & spatial.py evaluate to True")

    # 5. Manifest check
    prep_man = read_json(PREP_ROOT / "PREP_ARTIFACT_MANIFEST.json")
    for f_entry in prep_man["files"]:
        fp = PREP_ROOT / f_entry["path"]
        check(fp.exists(), f"Manifest file exists: {f_entry['path']}")
        check(sha256_file(fp) == f_entry["sha256"], f"Manifest sha256 matches: {f_entry['path']}")

    print(f"ALL {checks} MULTIPATCH V5 PREP CHECKS PASSED.")


if __name__ == "__main__":
    main()
