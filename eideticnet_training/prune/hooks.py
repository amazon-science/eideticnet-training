import warnings
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch import nn

from .masks import assigned_params_mask, unassigned_params_mask


def attr_is_tensor(module, name):
    return hasattr(module, name) and getattr(module, name) is not None


def should_register_post_accumulate_grad_hook(module, name):
    return (
        attr_is_tensor(module, f"{name}_excess_capacity")
        and getattr(module, name).requires_grad
    )


def _freeze_assigned_params(weight_param_phase):
    """
    Freeze the parameters that have been assigned to previously-trained tasks.
    """

    def _filter(tensor):
        fraction_frozen = torch.count_nonzero(
            assigned_params_mask(weight_param_phase)
        ).item() / torch.numel(weight_param_phase)
        if fraction_frozen == 1:
            msg = f"All parameters of module are frozen ({tensor.shape})."
            warnings.warn(msg)
        tensor.grad[assigned_params_mask(weight_param_phase)] = 0

    return _filter


def _bn_compute_frozen_running_stats(module, args, output):
    """
    Ensure that the running mean and variance of the previously-trained units
    of a batch normalization layer remain constant during training. Register
    this function as a forward pre-hook on the layer. This function must be
    used in conjunction with `_bn_preserve_frozen_running_stats`.
    """
    if not module.training or not hasattr(module, "_frozen_running_mean"):
        return None
    x = args[0]

    if not attr_is_tensor(module, "weight_excess_capacity"):
        return
    mask = unassigned_params_mask(module.weight_excess_capacity)

    view = list(module.running_mean.shape)
    if isinstance(module, nn.BatchNorm2d):
        view += [1, 1]
    frozen = F.batch_norm(
        x,
        module._frozen_running_mean,
        module._frozen_running_var,
        weight=module.weight,
        bias=module.bias,
        # Even though the module is in training mode, set training to false
        # here because we don't want the frozen running stats tensors to be
        # updated. We could defensively copy the frozen running stats tensors
        # when caling this function, but setting training to false is more
        # efficient.
        training=False,
        momentum=module.momentum,
        eps=module.eps,
    )
    if module.weight is not None:
        return torch.where(
            mask.view(view),
            output,
            frozen * module.weight.view(view) + module.bias.view(view),
        )
    else:
        return torch.where(mask.view(view), output, frozen)


@torch.no_grad
def _bn_preserve_frozen_running_stats(module, args):
    """
    Ensure that the running mean and variance of the previously-trained units
    of a batch normalization layer remain constant during training. Register
    this function as a forward pre-hook on the layer. This function must be
    used in conjunction with `_bn_compute_frozen_running_stats`.

    When starting to train a network on a task, this function copies the
    running mean and variance to out-of-band buffers. These buffers are
    used by `_bn_compute_frozen_running_stats` such that the mean and variance of
    previously-trained (frozen) units remain constant. When the network
    switches to eval model, this function copies the mean and variance of
    frozen units over the original tensors.

    The running mean and variance are normally updated on an ongoing basis
    during training in the forward pass. This is independent of the zeroing out
    of the gradients of the weight and bias of the units of previously-trained
    tasks in the backward pass via the post-accumulate-grad hook.
    """
    # This should enter first mini batch when starting fine-tuning.
    if not hasattr(module, "_frozen_running_mean") and module.training:
        if attr_is_tensor(module, "running_mean"):
            module._frozen_running_mean = module.running_mean.clone()
        if attr_is_tensor(module, "running_var"):
            module._frozen_running_var = module.running_var.clone()

    # This should enter first eval minibatch after fine-tuning.
    if hasattr(module, "_frozen_running_mean") and not module.training:
        mask = assigned_params_mask(module.weight_excess_capacity)
        module.running_mean[mask] = module._frozen_running_mean[mask]
        module.running_var[mask] = module._frozen_running_var[mask]
        del module._frozen_running_mean
        del module._frozen_running_var


def register_eidetic_hooks(model):
    handles = defaultdict(dict)
    for module_name, module in model.named_modules():
        if should_register_post_accumulate_grad_hook(module, "weight"):
            handle = module.weight_orig.register_post_accumulate_grad_hook(
                _freeze_assigned_params(module.weight_excess_capacity)
            )
            handles[module_name]["freeze_weights"] = handle
        if should_register_post_accumulate_grad_hook(module, "bias"):
            handle = module.bias_orig.register_post_accumulate_grad_hook(
                _freeze_assigned_params(module.bias_excess_capacity)
            )
            handles[module_name]["freeze_biases"] = handle
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            handle = module.register_forward_hook(
                _bn_compute_frozen_running_stats
            )
            handles[module_name]["compute_frozen_bn_running_stats"] = handle
            handle = module.register_forward_pre_hook(
                _bn_preserve_frozen_running_stats
            )
            handles[module_name]["preserve_frozen_bn_running_stats"] = handle
    return handles


def unregister_eidetic_hooks(handles):
    for module_name in handles.keys():
        for hook_name in handles[module_name].keys():
            handles[module_name][hook_name].remove()
