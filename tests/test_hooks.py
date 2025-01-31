import torch
import torch.nn.utils.prune as prune

from eideticnet_training.prune import hooks, masks


def test_freeze_assigned_params():
    """Freeze the weight_orig parameters assigned to previous tasks."""
    linear = torch.nn.Linear(10, 2)
    prune.identity(linear, "weight")
    linear.weight_excess_capacity = masks.UNASSIGNED_PARAMS * torch.ones_like(
        linear.weight
    )
    # Mark the first neuron as assigned to the first task.
    linear.weight_excess_capacity[0] = 0
    filter_func = hooks._freeze_assigned_params(linear.weight_excess_capacity)
    linear.weight_orig.register_post_accumulate_grad_hook(filter_func)
    weight_orig = linear.weight_orig.clone()

    optimizer = torch.optim.SGD(linear.parameters(), lr=0.1)
    data = torch.randn(3, 10)
    labels = torch.multinomial(torch.tensor([0.5, 0.5]), 3, replacement=True)
    y = linear(data)
    loss = torch.nn.functional.cross_entropy(y, labels)
    loss.backward()
    optimizer.step()
    assert torch.all(linear.weight_orig[0] == weight_orig[0])
    assert torch.all(linear.weight_orig[1:] != weight_orig[1:])


def test_bn_hooks():
    bn = torch.nn.BatchNorm2d(10)
    prune.identity(bn, "weight")
    prune.identity(bn, "bias")
    bn.weight_excess_capacity = masks.UNASSIGNED_PARAMS * torch.ones_like(
        bn.weight
    )
    bn.bias_excess_capacity = masks.UNASSIGNED_PARAMS * torch.ones_like(
        bn.bias
    )
    # Mark the first unit as assigned to the first task.
    bn.weight_excess_capacity[0] = 0
    bn.bias_excess_capacity[0] = 0
    bn.weight_orig.register_post_accumulate_grad_hook(
        hooks._freeze_assigned_params(bn.weight_excess_capacity)
    )
    bn.bias_orig.register_post_accumulate_grad_hook(
        hooks._freeze_assigned_params(bn.bias_excess_capacity)
    )
    bn.register_forward_hook(hooks._bn_compute_frozen_running_stats)
    bn.register_forward_pre_hook(hooks._bn_preserve_frozen_running_stats)
    weight_orig = bn.weight_orig.clone()
    bias_orig = bn.bias_orig.clone()
    optimizer = torch.optim.SGD(bn.parameters(), lr=0.1)
    data = torch.randn(3, 10, 10, 3)
    target = torch.randn(10, 10, 3)
    bn.train()
    y = bn(data)
    loss = torch.nn.functional.mse_loss(y.mean(dim=0), target)
    loss.backward()
    optimizer.step()
    # Verify that the weight and bias of a frozen unit do not change during
    # training.
    assert torch.all(bn.weight_orig[0] == weight_orig[0])
    assert torch.all(bn.bias_orig[0] == bias_orig[0])
    assert torch.all(bn.weight_orig[1:] != weight_orig[1:])
    assert torch.all(bn.bias_orig[1:] != bias_orig[1:])
    # Verify that the first unit output (including running mean and var) does
    # not change during training.
    data2 = torch.randn(3, 10, 10, 3)
    y1 = bn(data2)
    y2 = bn(data2)
    assert torch.all(y1[0] == y2[0])
    bn.eval()
    data3 = torch.randn(3, 10, 10, 3)
    y3 = bn(data3)
    y4 = bn(data3)
    assert torch.all(y4 == y3)


def apply_finetuning_hooks(model):
    raise NotImplementedError()
