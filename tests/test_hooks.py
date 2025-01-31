import torch
import torch.nn.functional as F
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


def test_bn_compute_frozen_running_stats():
    """Verify that frozen running stats are computed correctly."""

    def verify_hook_output(hook_output, output):
        assert hook_output is not None
        assert torch.is_tensor(hook_output)
        assert hook_output.shape == output.shape
        assert torch.all(hook_output == output)

    #######################################################################
    # Set up conditions for testing that frozen running stats are computed
    # correctly as units are assigned to tasks and become frozen.
    #######################################################################

    data = torch.randn(2, 3, 1, 1)
    bn = torch.nn.BatchNorm2d(3)

    # Update running stats once.
    bn.train()
    bn(data)

    # Copy running stats here and use them as the running stats for tasks when
    # we freeze them later in the test.
    frozen_running_mean = bn.running_mean.clone()
    frozen_running_var = bn.running_var.clone()
    bn._frozen_running_mean = frozen_running_mean.clone()
    bn._frozen_running_var = frozen_running_var.clone()

    frozen_output = F.batch_norm(
        data,
        bn._frozen_running_mean,
        bn._frozen_running_var,
        weight=bn.weight,
        bias=bn.bias,
        # Even though the module is in training mode, set training to false
        # here because we don't want the frozen running stats tensors to be
        # updated.
        training=False,
        momentum=bn.momentum,
        eps=bn.eps,
    )

    # Having done that, also create the masks that track which units have been
    # assigned to tasks.
    bn.weight_excess_capacity = masks.UNASSIGNED_PARAMS * torch.ones_like(
        bn.weight
    )
    bn.bias_excess_capacity = masks.UNASSIGNED_PARAMS * torch.ones_like(
        bn.bias
    )

    # Update the running stats again. These will be the running stats for
    # normalization units that have not yet been assigned to a task.
    bn.train()
    bn(data)

    bn.eval()
    non_frozen_output = bn(data)
    bn.train()

    # Verify that frozen and non-frozen running stats are everywhere different.
    assert not torch.any(frozen_output == non_frozen_output)

    #######################################################################
    # Verify the output is the same when no tasks have been assigned.
    #######################################################################

    args = [data]
    hook_output_no_tasks = hooks._bn_compute_frozen_running_stats(
        bn, args, non_frozen_output
    )
    assert torch.all(bn.running_mean != bn._frozen_running_mean)
    assert torch.all(bn.running_var != bn._frozen_running_var)
    assert torch.all(bn._frozen_running_mean == frozen_running_mean)
    assert torch.all(bn._frozen_running_var == frozen_running_var)
    verify_hook_output(hook_output_no_tasks, non_frozen_output)

    #######################################################################
    # Verify the output does not change when one task has been assigned.
    #######################################################################

    # Assign the first weight and bias units to task 0.
    bn.weight_excess_capacity[0] = 0
    bn.bias_excess_capacity[0] = 0

    hook_output_one_task = hooks._bn_compute_frozen_running_stats(
        bn, args, non_frozen_output
    )
    assert torch.all(bn.running_mean != bn._frozen_running_mean)
    assert torch.all(bn.running_var != bn._frozen_running_var)
    assert torch.all(bn._frozen_running_mean == frozen_running_mean)
    assert torch.all(bn._frozen_running_var == frozen_running_var)

    verify_hook_output(hook_output_one_task[:, :1], frozen_output[:, :1])
    verify_hook_output(hook_output_one_task[:, 1:], non_frozen_output[:, 1:])

    #######################################################################
    # Verify the output does not change when two tasks have been assigned.
    #######################################################################

    # Assign the second weight and bias units to task 1.
    bn.weight_excess_capacity[1] = 1
    bn.bias_excess_capacity[1] = 1

    hook_output_two_tasks = hooks._bn_compute_frozen_running_stats(
        bn, args, non_frozen_output
    )
    assert torch.all(bn.running_mean != bn._frozen_running_mean)
    assert torch.all(bn.running_var != bn._frozen_running_var)
    assert torch.all(bn._frozen_running_mean == frozen_running_mean)
    assert torch.all(bn._frozen_running_var == frozen_running_var)
    verify_hook_output(hook_output_two_tasks[:, :2], frozen_output[:, :2])
    verify_hook_output(hook_output_two_tasks[:, 2:], non_frozen_output[:, 2:])


def test_bn_preserve_frozen_running_stats_train_mode():
    """Verify behavior of batch norm hooks in train mode."""
    data = torch.randn(3, 10, 10, 3)
    bn = torch.nn.BatchNorm2d(10)

    # After switching from eval to training mode, the frozen stat tensors
    # should be created on the first call only.
    bn.train()
    bn(data)
    hooks._bn_preserve_frozen_running_stats(bn, None)
    assert hasattr(bn, "_frozen_running_mean")
    assert hasattr(bn, "_frozen_running_var")
    mean_id = id(bn._frozen_running_mean)
    var_id = id(bn._frozen_running_var)
    frozen_running_mean_copy = bn._frozen_running_mean.clone()
    frozen_running_var_copy = bn._frozen_running_var.clone()
    bn(data)
    assert id(bn._frozen_running_mean) == mean_id
    assert id(bn._frozen_running_var) == var_id
    assert torch.all(bn._frozen_running_mean == frozen_running_mean_copy)
    assert torch.all(bn._frozen_running_var == frozen_running_var_copy)


def test_bn_preserve_frozen_running_stats_eval_mode():
    """Verify behavior of batch norm hooks in eval mode."""
    data = torch.randn(3, 10, 10, 3)
    bn = torch.nn.BatchNorm2d(10)

    bn(data)
    bn.eval()

    # Initialize mask no parameters assigned to a task.
    bn.weight_excess_capacity = masks.UNASSIGNED_PARAMS * torch.ones_like(
        bn.weight
    )

    # After switching from training to eval mode, if no parameters have been
    # assigned to a task, the running stats should not change.
    running_mean = bn.running_mean.clone()
    running_var = bn.running_var.clone()
    # Dummy running mean and var. Because no tasks have been assigned, the
    # values from the dummies should not end up in the real running mean or
    # var.
    bn._frozen_running_mean = 42 * torch.ones_like(bn.running_mean)
    bn._frozen_running_var = 42 * torch.ones_like(bn.running_var)
    hooks._bn_preserve_frozen_running_stats(bn, None)
    assert torch.all(bn.running_mean == running_mean)
    assert torch.all(bn.running_var == running_var)
    # These attributes are deleted the first time through in eval mode.
    assert not hasattr(bn, "_frozen_running_mean")
    assert not hasattr(bn, "_frozen_running_var")

    # Assign a unit to a task. In this case, we expect the frozen running
    # stats of that unit to be copied from the frozen stats.
    bn.weight_excess_capacity[0] = 0
    # Set up the dummy running mean and var again.
    bn._frozen_running_mean = 42 * torch.ones_like(bn.running_mean)
    bn._frozen_running_var = 42 * torch.ones_like(bn.running_var)
    hooks._bn_preserve_frozen_running_stats(bn, None)
    assert bn.running_mean[0] == 42
    assert bn.running_var[0] == 42
    assert torch.all(bn.running_mean[1:] == running_mean[1:])
    assert torch.all(bn.running_var[1:] == running_var[1:])


def test_bn_hooks_all():
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
