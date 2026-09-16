"""Verify saved experiment identities, splits, labels, metrics and objective coverage."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.state_value_experiment import metrics


def audit(dataset, output, pilot):
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    manifest, report = read(dataset / "manifest.json"), read(output / "results.json")
    assert manifest["status"] == report["status"] == "complete"
    selected = read(dataset / "selection.json")["match_ids"]
    assert len(selected) == len(set(selected))
    assert not set(selected) & set(read(pilot / "selection.json")["match_ids"])
    splits = read(output / "match_splits.json")
    all_ids = [mid for ids in splits.values() for mid in ids]
    assert len(all_ids) == len(set(all_ids))
    assert set(all_ids) == set(manifest["completed"])
    root = Path(__file__).resolve().parents[1]
    config = read(output / "run_configuration.json")
    for mapping in (manifest["settings"]["source_sha256"], config["source_sha256"]):
        for name, digest in mapping.items():
            assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
    model_id = hashlib.sha256((output / "value_model.joblib").read_bytes()).hexdigest()
    assert model_id == report["value_model_id"]
    cv = read(output / "value_regularization_selection.json")
    assert cv["chosen_C"] == min(cv["scores"], key=lambda s: s["log_loss"])["C"]
    assert cv["chosen_C"] == report["value_regularization"]["C"]
    seen = []
    for f in cv["folds"]:
        tr, va = set(f["train_matches"]), set(f["validation_matches"])
        assert not tr & va and tr | va == set(splits["value_train"])
        seen.extend(va)
    assert Counter(seen) == Counter(splits["value_train"])
    labels = {}
    for role in ("predict_train", "predict_test"):
        with np.load(output / (role + "_labels.npz"), allow_pickle=False) as z:
            labels[role] = {k: z[k] for k in z.files}
        z = labels[role]
        assert str(z["model_id"]) == model_id and str(z["split_id"]) == role
        assert set(z["groups"]).issubset(splits[role])
        assert len(set(z["engagement_id"])) == len(z["engagement_id"])
        assert np.array_equal(z["delta"], z["value_post"] - z["value_pre"])
        assert np.all(z["pre_snapshot"] <= z["cutoff"])
        assert np.all(z["post_snapshot"] <= z["post_time"])
        assert np.all(z["post_time"] > z["cutoff"])
        ok = z["y"] >= 0
        assert np.all(z["reason"][ok] == "labelled")
        assert np.array_equal(z["y"][ok], (z["delta"][ok] > 0).astype(int))
    tr, te = labels["predict_train"], labels["predict_test"]
    with np.load(output / "predictions.npz", allow_pickle=False) as p:
        assert np.array_equal(p["train_id"], tr["engagement_id"][tr["y"] >= 0])
        assert np.array_equal(p["test_id"], te["engagement_id"][te["y"] >= 0])
        assert np.array_equal(p["test_y"], te["y"][te["y"] >= 0])
        assert np.array_equal(p["test_match_win"], te["winner"][te["y"] >= 0])
        groups = tr["groups"][tr["y"] >= 0]
        folds = read(output / "oof_splits.json")
        seen = []
        assert np.isfinite(p["train_engagement_oof"]).all()
        assert set(p["train_oof_fold"]) == {f["fold"] for f in folds}
        for f in folds:
            a, b = set(f["train_matches"]), set(f["validation_matches"])
            assert not a & b and a | b == set(groups)
            assert set(groups[p["train_oof_fold"] == f["fold"]]) == b
            seen.extend(b)
        assert Counter(seen) == Counter(set(groups))
        checks = {"engagement_test": (p["test_y"], p["test_engagement"])}
        for key in ("A_pre_information", "B_plus_predicted_engagement", "C_plus_realized_label_RETROSPECTIVE"):
            checks[key] = p["test_match_win"], p[key]
        for key, (y, prob) in checks.items():
            calculated = metrics(y, prob, p["test_groups"])
            for metric in ("auc", "brier", "log_loss"):
                assert abs(calculated[metric] - report[key][metric]) < 1e-12
    names = read(dataset / "schema.json")["state_names"]
    coverage = {}
    for role in ("value_train", "value_validation"):
        counts = Counter()
        for mid in splits[role]:
            with np.load(dataset / "matches" / (mid + ".npz"), allow_pickle=False) as z:
                states = np.concatenate([z["value_states"], z["pre"], z["post"]])
            for kind in ("baron", "elder", "dragons", "soul_event_recorded"):
                if np.any(states[:, [names.index(t + kind) for t in ("blue_", "red_")]] > 0):
                    counts[kind] += 1
        coverage[role] = dict(counts)
    result = {"status": "passed", "selected_matches": len(selected), "completed_matches": len(all_ids),
              "exclusions": dict(Counter(manifest["excluded"].values())), "split_matches": {k: len(v) for k,v in splits.items()},
              "pilot_overlap": 0, "value_model_sha256": model_id, "source_hashes_match": True,
              "group_cv_and_oof": "disjoint; each held-out match exactly once",
              "labels_timestamps_and_saved_metrics": "verified", "objective_distinct_matches_either_team": coverage}
    (output / "artifact_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--pilot-dataset", type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(audit(args.dataset, args.out_dir, args.pilot_dataset), indent=2))
