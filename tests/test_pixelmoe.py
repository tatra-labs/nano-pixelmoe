import torch

from pixelmoe import Config, MoE, PixelDiT, flow

TINY = dict(dim=64, depth=3, heads=4, dense_blocks=1, ffn_hidden=128, experts=8, active=2, routed_hidden=32,
            shared_hidden=64, head_dim=96, head_depth=1)


def test_moe_routes_every_token_to_a_experts_and_balances():
    torch.manual_seed(0)
    moe = MoE(16, experts=8, active=2, routed_hidden=8, shared_hidden=8, gamma=0.1)
    x = torch.randn(4, 50, 16)
    assert moe(x).shape == x.shape
    assert torch.isclose(moe.load.sum(), torch.tensor(1.0))
    with torch.no_grad():  # make expert 0 everyone's favourite
        moe.router.weight[0] = 5 * x.reshape(-1, 16).mean(0)
    moe(x)
    before = moe.load[0].item()
    for _ in range(50):
        moe(x)
        moe.update_bias()
    assert moe.bias[0] < 0 and moe.load[0].item() < before  # the bias steers tokens away


def test_dense_and_moe_twins_have_matched_active_parameters():
    dense = PixelDiT(Config(experts=0))
    moe = PixelDiT(Config())
    assert abs(moe.active_params() / dense.active_params() - 1) < 0.05
    assert sum(p.numel() for p in moe.parameters()) > 2 * sum(p.numel() for p in dense.parameters())


def test_one_model_reads_both_resolutions_and_trains():
    torch.manual_seed(0)
    model = PixelDiT(Config(**TINY))
    for size in (16, 32):
        x = torch.randn(2, 3, size, size)
        assert model(x, torch.rand(2), torch.tensor([1, 10])).shape == x.shape
    loss = flow.loss(model, torch.rand(8, 3, 32, 32) * 2 - 1, torch.randint(0, 10, (8,)))
    loss.backward()
    assert torch.isfinite(loss) and model.blocks[-1].ffn.router.weight.grad is not None


def test_sampler_returns_full_resolution_images():
    model = PixelDiT(Config(**TINY)).eval()
    x = flow.sample(model, torch.tensor([0, 3]), 32, steps=4)
    assert x.shape == (2, 3, 32, 32) and x.abs().max() <= 1


def test_stage_transition_noise_matches_the_stage1_distribution():
    torch.manual_seed(0)
    z = torch.randn(20000, 1, 2, 2)
    z = (z - flow.up(torch.nn.functional.avg_pool2d(z, 2))) * 2 / 3 ** 0.5  # eps' of the paper
    flat = z.reshape(-1, 4)
    cov = flat.T @ flat / len(flat)
    assert torch.allclose(cov.diag(), torch.ones(4), atol=0.03)
    assert torch.allclose(cov[0, 1:], torch.full((3,), -1 / 3), atol=0.03)
    # Upsampled Stage 0 noise (scaled) plus this noise has the Stage 1 covariance t_s1^2 I.
    beta, alpha = (1 - flow.T_S1) / (1 - flow.T_E0) * flow.T_E0, 3 ** 0.5 / 2 * flow.T_S1
    total = beta ** 2 * torch.ones(4, 4) + alpha ** 2 * cov
    assert torch.allclose(total, flow.T_S1 ** 2 * torch.eye(4), atol=0.03)
