"""두 합성 점수 행의 AdamW 닫힌 계산과 직렬화 후 재개를 검산한다."""
import io
import itertools
import math
from types import SimpleNamespace


def test_numeric_adamw_and_checkpointed_optimizer_resume(tmp_path):
    import torch
    from c_ept.ept import checkpoints, train, training_step

    # 실제 loss.total_loss가 두 행을 평균하므로 gradient = weight - desired.
    # 실제 머리의 NLL이나 연구 데이터를 재정의/평가하는 테스트는 아니다.
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        weights = torch.nn.Parameter(torch.tensor([.25, -.75], dtype=torch.float64))
        optimizer = torch.optim.AdamW([weights], lr=.01, weight_decay=.1, foreach=False)
        reference, first, second = [.25, -.75], [0., 0.], [0., 0.]
        resumed = None
        ticks = itertools.count().__next__
        def score(fa, encoded, desired):
            target = {'y': torch.tensor([0, 1])}
            return {'M2Mark/masked': ('M2Mark', target, (encoded - desired).square())}
        def update(parameter, opt, desired, lr, step):
            opt.param_groups[0]['lr'] = lr
            opt.zero_grad(set_to_none=True)
            fa = SimpleNamespace(encode=lambda batch: parameter)
            components = dict(transfer_seconds=0., forward_seconds=0., backward_seconds=0., optimizer_seconds=0.)
            total = training_step.optimize_batch(fa, desired, opt, {'M2Mark/masked': 1.}, step, components, score=score, clock=ticks)
            assert components == dict(transfer_seconds=0., forward_seconds=1., backward_seconds=1., optimizer_seconds=1.)
            return total
        for step, desired_values in enumerate(([1., -.5], [-.25, .5], [.75, -.25])):
            desired = torch.tensor(desired_values, dtype=torch.float64)
            lr = train.lr_at(step, 4, 2, .01)
            expected_loss = sum((x-y)**2 for x,y in zip(reference, desired_values))/2
            total = update(weights, optimizer, desired, lr, step)
            assert math.isclose(float(total.detach()), expected_loss, rel_tol=1e-12, abs_tol=1e-14)
            for i, target in enumerate(desired_values):
                gradient = reference[i] - target
                first[i] = .9 * first[i] + .1 * gradient
                second[i] = .999 * second[i] + .001 * gradient**2
                mean = first[i] / (1 - .9**(step+1))
                variance = second[i] / (1 - .999**(step+1))
                reference[i] = reference[i] * (1 - lr * .1) - lr * mean / (math.sqrt(variance) + 1e-8)
            torch.testing.assert_close(weights.detach(), torch.tensor(reference, dtype=torch.float64), rtol=1e-12, atol=1e-14)
            if resumed is not None:
                rw, ro = resumed
                update(rw, ro, desired, lr, step)
                assert torch.equal(weights.detach(), rw.detach())
                for key in ('step', 'exp_avg', 'exp_avg_sq'):
                    assert torch.equal(optimizer.state[weights][key], ro.state[rw][key])
            if step == 0:
                payload = checkpoints.torch_bytes({'weights': weights.detach().clone(), 'optimizer': optimizer.state_dict()})
                path = tmp_path / 'synthetic-optimizer.pt'
                checkpoints.write_atomic(path, payload)
                verified = checkpoints.read_verified(path, {'bytes': len(payload), 'sha256': checkpoints.sha256(payload)}, 'synthetic')
                state = torch.load(io.BytesIO(verified), weights_only=True, map_location='cpu')
                rw = torch.nn.Parameter(state['weights'].clone())
                ro = torch.optim.AdamW([rw], lr=.01, weight_decay=.1, foreach=False)
                ro.load_state_dict(state['optimizer'])
                resumed = rw, ro
    finally:
        torch.set_num_threads(previous_threads)
