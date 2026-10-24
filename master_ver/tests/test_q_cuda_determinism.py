"""q neural families on CUDA: two complete fits from the same spec are bitwise identical (ARCHITECTURE_KO.md 6.4)."""
import numpy as np
import pytest
import torch

from studies.q_production_20261005 import q_worker, qtrain
from tests.test_q_worker import GRAPH, _spec, run_dir  # noqa: F401 - run_dir is a fixture

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
if torch.cuda.is_available():
    # scripts/run_q_v3_baselines.py sets CUDA_VISIBLE_DEVICES='' when tests/test_q_v3_baselines.py imports it; initializing
    # CUDA at collection keeps the device visible to these tests in a full-suite run.
    torch.cuda.init()

FAMILIES = [
    ("relational_mpnn", {"hidden_width": 16, "weight_decay": 0.001}, GRAPH),
    ("team_set_pair_readout", {"hidden_width": 16, "weight_decay": 0.001}, dict(GRAPH, G=0)),
    ("tied_message", {"hidden_width": 16, "weight_decay": 0.001}, dict(GRAPH, A=0, M=0)),
    ("graphsage", {"hidden_width": 16, "weight_decay": 0.001}, dict(GRAPH, G=None)),
    ("shared_slot_mlp", {"hidden_width": 16, "weight_decay": 0.001}, dict(GRAPH, G=None)),
    ("pure_set", {"hidden_width": 16, "weight_decay": 0.001}, {"G": None, "S": 1, "A": 0, "M": 0, "E": 1, "C": 1, "R": 1, "D": 1}),
    ("plain_mlp", {"hidden": [16, 8], "weight_decay": 0.001}, dict(GRAPH, G=None)),
    ("ft_transformer", {"d_token": 8, "layers": 1, "heads": 2}, dict(GRAPH, G=None)),
]


@pytest.mark.parametrize("family,card,factors", FAMILIES, ids=[f[0] for f in FAMILIES])
def test_two_cuda_fits_are_bitwise_identical(run_dir, tmp_path, monkeypatch, family, card, factors):  # noqa: F811
    if torch.cuda.device_count() == 0:
        pytest.skip("CUDA hidden in this process; run tests/test_q_cuda_determinism.py on its own")
    root, _, _, inputs = run_dir
    monkeypatch.setattr(q_worker.qtrain, "neural_rules", lambda f, wd=None: qtrain.NeuralRules(batch_size=16, max_epochs=3, patience=3,
                                                                                                 weight_decay=float(wd or 0.0)))
    outputs = []
    for k in (1, 2):
        out = tmp_path / f"fit{k}"
        spec = dict(_spec(root, f"cuda/{family}", family, card, factors, inputs, out), device="cuda:0")
        result = q_worker.run(spec, out)
        assert result["status"] == "completed", result
        state = torch.load(out / "model_refit.pt", map_location="cpu", weights_only=False)["model"]
        outputs.append((np.load(out / "pred_q_dev.npz")["logit"], np.load(out / "pred_q_cal.npz")["logit"], state, result))
    (dev1, cal1, s1, r1), (dev2, cal2, s2, r2) = outputs
    assert np.array_equal(dev1, dev2) and np.array_equal(cal1, cal2)
    assert s1.keys() == s2.keys() and all(torch.equal(s1[k], s2[k]) for k in s1)
    assert r1["best_epoch"] == r2["best_epoch"] and r1["model_sha256"] == r2["model_sha256"]
