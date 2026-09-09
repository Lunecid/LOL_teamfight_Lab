"""Disjoint valuation and engagement learning, followed by OOF stacking."""
from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from gameplay.state_value import value_labels


def match_weights(groups):
    _, ix, counts = np.unique(groups, return_inverse=True, return_counts=True)
    w = 1. / counts[ix]
    return w / w.mean()


def metrics(y, p, groups):
    w = match_weights(groups)
    calibration = []
    for lo in np.arange(0., 1., .1):
        mask = (p >= lo) & (p < lo + .1 + (1e-10 if lo > .89 else 0))
        if mask.any():
            calibration.append(dict(n=int(mask.sum()), predicted=float(np.average(p[mask], weights=w[mask])),
                                    observed=float(np.average(y[mask], weights=w[mask]))))
    return dict(n=len(y), matches=len(set(groups)), positive_rate=float(np.average(y, weights=w)),
                auc=float(roc_auc_score(y, p, sample_weight=w)) if len(set(y)) == 2 else None,
                brier=float(brier_score_loss(y, p, sample_weight=w)),
                log_loss=float(log_loss(y, p, labels=[0, 1], sample_weight=w)), calibration=calibration)


def assert_disjoint(*groups):
    sets = [set(x) for x in groups]
    for i, group in enumerate(sets):
        for other in sets[i+1:]:
            if group & other:
                raise ValueError("match leakage between partitions")


def logistic(names, C=1.):
    cats = [i for i, k in enumerate(names) if k.endswith("champion_id")]
    nums = [i for i in range(len(names)) if i not in cats]
    prep = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), nums),
        ("champions", OneHotEncoder(handle_unknown="ignore"), cats)])
    return Pipeline([("preprocess", prep), ("model", LogisticRegression(C=C, solver="liblinear", max_iter=1500, random_state=7))])


def select_value_regularization(X, y, groups, names, candidates=(.0001, .001, .01, .1, 1.), folds=3):
    """Choose C using only match-grouped folds within the value TRAIN partition."""
    splits = list(GroupKFold(folds).split(X, y, groups))
    scores, audits = [], []
    for tr, va in splits:
        assert_disjoint(groups[tr], groups[va])
        audits.append({"train_matches": sorted(set(groups[tr])), "validation_matches": sorted(set(groups[va]))})
    for C in candidates:
        p = np.full(len(y), np.nan)
        for tr, va in splits:
            model = logistic(names, C=C)
            model.fit(X[tr], y[tr], model__sample_weight=match_weights(groups[tr]))
            p[va] = model.predict_proba(X[va])[:, 1]
        scores.append({"C": C, **metrics(y, p, groups)})
        print(f"[valuation CV] C={C:g}, log loss={scores[-1]['log_loss']:.4f}", flush=True)
    chosen = min(scores, key=lambda row: row["log_loss"])["C"]
    return chosen, {"selection_partition": "value_train_only", "criterion": "match_weighted_log_loss", "chosen_C": chosen, "scores": scores, "folds": audits}


def learner(seed=7, trees=250):
    return LGBMClassifier(n_estimators=trees, num_leaves=15, max_depth=-1, learning_rate=.04,
                         min_child_samples=40, reg_lambda=1., colsample_bytree=.9,
                         random_state=seed, n_jobs=4, verbosity=-1)


def fit_engagement_oof(X, y, groups, *, folds=5, trees=250):
    if len(set(groups)) < folds:
        raise ValueError("not enough training matches for OOF")
    pred = np.full(len(y), np.nan)
    fold_ids = np.full(len(y), -1)
    audits = []
    for fold, (tr, va) in enumerate(GroupKFold(folds).split(X, y, groups)):
        assert_disjoint(groups[tr], groups[va])
        if len(set(y[tr])) < 2:
            raise ValueError("single-class OOF training fold; enlarge sample")
        model = learner(trees=trees)
        model.fit(X[tr], y[tr], sample_weight=match_weights(groups[tr]))
        pred[va] = model.predict_proba(X[va])[:, 1]
        fold_ids[va] = fold
        audits.append({"fold": fold, "train_matches": sorted(set(groups[tr])), "validation_matches": sorted(set(groups[va]))})
    if not np.isfinite(pred).all() or (fold_ids < 0).any():
        raise ValueError("incomplete OOF predictions")
    return pred, fold_ids, audits


def paired_bootstrap(y, a, b, groups, n_boot=300):
    unique, inverse = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(7)
    base = match_weights(groups)
    draws = []
    for _ in range(n_boot):
        multiplicity = np.bincount(rng.integers(0, len(unique), size=len(unique)), minlength=len(unique))
        w = base * multiplicity[inverse]
        if len(set(y[w > 0])) < 2:
            continue
        draws.append([float(np.average((y-a)**2 - (y-b)**2, weights=w)),
                      float(roc_auc_score(y, b, sample_weight=w)-roc_auc_score(y, a, sample_weight=w))])
    if not draws:
        return {"n_boot": 0, "reason": "insufficient_classes"}
    quantiles = np.quantile(draws, [.025, .975], axis=0)
    return {"n_boot": len(draws), "positive_means_B_better": True,
            "brier_improvement_ci95": quantiles[:, 0].tolist(), "auc_improvement_ci95": quantiles[:, 1].tolist()}


def run_experiment(dataset: Path, out: Path, *, folds=5, trees=250):
    if (out / "results.json").exists():
        raise ValueError("completed results already exist; use a new output directory")
    started = time.time()
    print("[evaluation] loading per-match checkpoints", flush=True)
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("dataset build incomplete")
    if len(manifest["completed"]) != len(set(manifest["completed"])):
        raise ValueError("duplicate match files")
    schema = json.loads((dataset / "schema.json").read_text(encoding="utf-8"))
    names = schema["state_names"]
    by_role = {r: [] for r in ("value_train", "value_validation", "predict_train", "predict_test")}
    for mid in manifest["completed"]:
        with np.load(dataset / "matches" / f"{mid}.npz", allow_pickle=False) as f:
            role = str(f["role"])
            # Valuation has its own compact state schema; do not materialize
            # unused 7,106-column engagement inputs for those matches.
            unused = {"X"} if role.startswith("value_") else {"value_states", "value_times"}
            record = {k: f[k] for k in f.files if k not in unused}
        if str(record["match_id"]) != mid:
            raise ValueError("manifest/record mismatch")
        by_role[str(record["role"])].append(record)
    groups_by_role = {r: [str(z["match_id"]) for z in rows] for r, rows in by_role.items()}
    assert_disjoint(*groups_by_role.values())
    if any(not rows for rows in by_role.values()):
        raise ValueError("empty experimental partition; enlarge sample")
    out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    configuration = {"dataset": str(dataset.resolve()), "folds": folds, "trees": trees,
                     "dataset_settings": manifest["settings"],
                     "source_sha256": {p: hashlib.sha256((root / p).read_bytes()).hexdigest()
                                       for p in ("train/state_value_experiment.py", "gameplay/state_value.py")},
                     "versions": {p: version(p) for p in ("numpy", "scikit-learn", "lightgbm", "joblib")}}
    (out / "run_configuration.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    (out / "match_splits.json").write_text(json.dumps(groups_by_role, indent=2), encoding="utf-8")

    def valuation_data(rows):
        matrices, labels, groups = [], [], []
        for z in rows:
            x = np.concatenate([z["value_states"], z["pre"], z["post"]])
            matrices.append(x); labels.extend([int(z["winner"])] * len(x)); groups.extend([str(z["match_id"])] * len(x))
        return np.concatenate(matrices), np.array(labels), np.array(groups)
    vx, vy, vg = valuation_data(by_role["value_train"])
    cx, cy, cg = valuation_data(by_role["value_validation"])
    if len(set(vy)) < 2:
        raise ValueError("single-class valuation training set")
    chosen_C, selection = select_value_regularization(vx, vy, vg, names)
    (out / "value_regularization_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    value = logistic(names, C=chosen_C)
    value.fit(vx, vy, model__sample_weight=match_weights(vg))
    print(f"[evaluation] value model fitted ({time.time()-started:.0f}s)", flush=True)
    value_path = out / "value_model.joblib"
    joblib.dump(value, value_path)
    model_id = hashlib.sha256(value_path.read_bytes()).hexdigest()
    report = {"protocol": "disjoint valuation + match-grouped engagement OOF + held-out match stacking",
              "status": "evaluating", "value_model_id": model_id, "input": manifest["settings"]["input"],
              "label_semantics": "frozen model value increase, not causal engagement effect",
              "value_validation": metrics(cy, value.predict_proba(cx)[:, 1], cg),
              "value_regularization": {"C": chosen_C, "selection": "3-fold grouped CV on value_train only"},
              "value_validation_training_prior_baseline": metrics(cy, np.repeat(np.average(vy, weights=match_weights(vg)), len(cy)), cg),
              "objective_representation": "event counts, explicit soul events, acquisition ages, deaths since acquisition; not exact active buffs"}
    objective_cols = [i for i, n in enumerate(names) if any(k in n for k in ("baron", "elder", "dragon", "soul_"))]
    report["objective_state_coverage"] = {n: {"train_nonzero": int((vx[:, i] != 0).sum()), "validation_nonzero": int((cx[:, i] != 0).sum()),
                                             "train_matches": len(set(vg[vx[:, i] != 0])), "validation_matches": len(set(cg[cx[:, i] != 0]))}
                                          for i, n in enumerate(names) if n in ("blue_baron", "red_baron", "blue_elder", "red_elder", "blue_dragons", "red_dragons", "blue_soul_event_recorded", "red_soul_event_recorded")}
    # Small diagnostic ablation: never use it to choose labels on the test set.
    keep = [i for i in range(len(names)) if i not in objective_cols]
    no_obj = logistic([names[i] for i in keep], C=chosen_C)
    no_obj.fit(vx[:, keep], vy, model__sample_weight=match_weights(vg))
    report["value_validation_without_objectives"] = metrics(cy, no_obj.predict_proba(cx[:, keep])[:, 1], cg)
    del vx, cx
    by_role["value_train"].clear()
    by_role["value_validation"].clear()
    tables = {}
    for role in ("predict_train", "predict_test"):
        pieces = []
        for z in by_role[role]:
            if not len(z["pre"]):
                continue
            a, b = value.predict_proba(z["pre"])[:, 1], value.predict_proba(z["post"])[:, 1]
            y, delta, reason = value_labels(a, b)
            pieces.append(dict(X=z["X"], pre=z["pre"], y=y, delta=delta, reason=reason,
                value_pre=a, value_post=b, groups=np.repeat(str(z["match_id"]), len(y)),
                winner=np.repeat(z["winner"], len(y)), engagement_id=z["engagement_id"],
                cutoff=z["cutoff"], post_time=z["post_time"], pre_snapshot=z["pre_snapshot"], post_snapshot=z["post_snapshot"],
                scale=z["scale"], y_market_event=z["y_market_event"], y_attention_value_win=z["y_attention_value_win"]))
        if not pieces:
            raise ValueError("no engagements in prediction partition")
        tab = {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
        if len(set(tab["engagement_id"])) != len(tab["engagement_id"]):
            raise ValueError("duplicate engagement rows")
        valid = tab["y"] >= 0
        report[role + "_labels"] = {"total": len(valid), "labelled": int(valid.sum()), "ties_or_invalid": int((~valid).sum())}
        report[role + "_labels"]["small_change_fraction"] = {
            str(eps): float((np.abs(tab["delta"][valid]) < eps).mean()) for eps in (.001, .01, .05)}
        np.savez_compressed(out / f"{role}_labels.npz", model_id=model_id, split_id=role, **{k: v for k, v in tab.items() if k not in ("X", "pre")})
        tables[role] = {k: v[valid] for k, v in tab.items()}
        by_role[role].clear()
    del pieces, tab, by_role
    tr, te = tables["predict_train"], tables["predict_test"]
    assert_disjoint(tr["groups"], te["groups"], vg, cg)
    if len(set(tr["y"])) < 2:
        raise ValueError("single-class engagement labels")
    oof, fold_ids, audits = fit_engagement_oof(tr["X"], tr["y"], tr["groups"], folds=folds, trees=trees)
    print(f"[evaluation] engagement OOF complete ({time.time()-started:.0f}s)", flush=True)
    (out / "oof_splits.json").write_text(json.dumps(audits, indent=2), encoding="utf-8")
    engagement = learner(trees=trees)
    engagement.fit(tr["X"], tr["y"], sample_weight=match_weights(tr["groups"]))
    p = engagement.predict_proba(te["X"])[:, 1]
    joblib.dump(engagement, out / "engagement_model.joblib")
    report["engagement_oof"] = metrics(tr["y"], oof, tr["groups"])
    report["engagement_test"] = metrics(te["y"], p, te["groups"])
    shortcut = learner(trees=trees)
    shortcut.fit(tr["value_pre"].reshape(-1, 1), tr["y"], sample_weight=match_weights(tr["groups"]))
    report["engagement_value_pre_only_baseline"] = metrics(te["y"], shortcut.predict_proba(te["value_pre"].reshape(-1, 1))[:, 1], te["groups"])
    report["engagement_by_scale"] = {}
    for name, mask in (("pick", te["scale"] <= 1), ("skirmish", (te["scale"] >= 2) & (te["scale"] <= 3)), ("teamfight", te["scale"] >= 4)):
        if mask.any():
            report["engagement_by_scale"][name] = metrics(te["y"][mask], p[mask], te["groups"][mask])
    report["label_agreement_test"] = {}
    for old in ("y_market_event", "y_attention_value_win"):
        mask = te[old] >= 0
        report["label_agreement_test"][old] = {"n": int(mask.sum()), "agreement": float((te[old][mask] == te["y"][mask]).mean()) if mask.any() else None}
    # Fair input-matched control: the same pre-cutoff X and learner in A/B/C.
    match_preds = {}
    for name, add_train, add_test in (("A_pre_information", None, None), ("B_plus_predicted_engagement", oof, p),
                                       ("C_plus_realized_label_RETROSPECTIVE", tr["y"], te["y"])):
        a = tr["X"] if add_train is None else np.column_stack([tr["X"], add_train])
        b = te["X"] if add_test is None else np.column_stack([te["X"], add_test])
        model = learner(trees=trees)
        model.fit(a, tr["winner"], sample_weight=match_weights(tr["groups"]))
        match_preds[name] = model.predict_proba(b)[:, 1]
        report[name] = metrics(te["winner"], match_preds[name], te["groups"])
        joblib.dump(model, out / f"{name}.joblib")
        print(f"[evaluation] {name} complete ({time.time()-started:.0f}s)", flush=True)
    report["B_minus_A"] = paired_bootstrap(te["winner"], match_preds["A_pre_information"], match_preds["B_plus_predicted_engagement"], te["groups"])
    np.savez_compressed(out / "predictions.npz", train_id=tr["engagement_id"], train_engagement_oof=oof,
                        train_oof_fold=fold_ids, test_id=te["engagement_id"], test_groups=te["groups"],
                        test_engagement=p, test_y=te["y"], test_match_win=te["winner"], **match_preds)
    report["status"] = "complete"
    report["elapsed_seconds"] = round(time.time()-started, 2)
    report["learner_settings"] = {"folds": folds, "trees": trees, "num_leaves": 15, "learning_rate": .04, "seed": 7}
    (out / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
