import torch

from eideticnet_training.prune import TaylorScorer


def _initialize_masks(module):
    # Set the first neuron to 0 to simulate that it's already been pruned in
    # this round of iterative pruning.
    module.weight_mask = torch.ones_like(module.weight)
    module.weight_mask[0] = 0
    # Set the second neuron to 0 to simulate that it's allocated to a previous
    # task.
    module.weight_excess_capacity = torch.ones_like(module.weight)
    module.weight_excess_capacity[1] = 0


def test_smoke_neuron_scorer_linear():
    linear = torch.nn.Linear(5, 3, bias=True)
    _initialize_masks(linear)
    modules = [linear]
    scorer = TaylorScorer(modules)
    scorer.register_hooks()
    x = torch.randn(4, 5)
    optimizer = torch.optim.SGD(linear.parameters(), lr=1.0)
    y = linear(x)
    target = torch.tensor([1.0])
    loss = torch.nn.functional.mse_loss(y.sum(), target)
    loss.backward()
    optimizer.step()
    for hook in scorer.hooks:
        hook.remove()
    scorer.partition_scores()
    scorer.normalize_scores()
    tuple(
        [
            module.normalized_scores
            for module in scorer.modules
            if hasattr(module, "normalized_scores")
        ]
    )


def test_smoke_neuron_scorer_conv2d():
    conv2d = torch.nn.Conv2d(
        out_channels=2, in_channels=1, kernel_size=3, bias=True
    )
    _initialize_masks(conv2d)
    modules = [conv2d]
    scorer = TaylorScorer(modules)
    scorer.register_hooks()
    x = torch.randn(1, 1, 3, 3)
    optimizer = torch.optim.SGD(conv2d.parameters(), lr=1.0)
    y = conv2d(x)
    target = torch.tensor([1.0])
    loss = torch.nn.functional.mse_loss(y.sum(), target)
    loss.backward()
    optimizer.step()
    for hook in scorer.hooks:
        hook.remove()
    scorer.partition_scores()
    scorer.normalize_scores()
    tuple(
        [
            module.normalized_scores
            for module in scorer.modules
            if hasattr(module, "normalized_scores")
        ]
    )
