import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from .masks import assigned_params_mask
from .threshold import thresholded_score_structured


@torch.no_grad
def get_pruning_importance(module: nn.Module, param: str):
    importance = getattr(module, param).clone()

    # Don't prune what has already been pruned while training the current task.
    if hasattr(module, f"{param}_mask"):
        importance[
            getattr(module, f"{param}_mask").bool().logical_not()
        ] = np.inf

    # Don't prune what has already been trained in previous tasks.
    # Previously-trained neurons are frozen and must not be modified.
    if hasattr(module, f"{param}_excess_capacity"):
        mask = assigned_params_mask(
            getattr(module, f"{param}_excess_capacity")
        )
        importance[mask] = np.inf

    return importance


def extract_first_two_dims(tensor):
    if tensor.dim() < 2:
        raise ValueError("Tensor must have at least 2 dimensions")
    return tensor[(slice(None), slice(None)) + (0,) * (tensor.dim() - 2)]


def propagate_mask(layer, next_layer, current_bn=None):
    """Propagates pruning masks through connected layers in a neural network.

    This function ensures consistency in pruning by propagating masks from one
    layer to connected components:

        1. Masks the bias of the current layer if pruned neurons exist
        2. Propagates masks to the batch normalization layer if present
        3. Propagates masks to the input dimensions of the following layer

    Args:
        layer (torch.nn.Module): The current layer being pruned
        next_layer (Optional[torch.nn.Module]): The subsequent layer in the
            network
        current_bn (Optional[torch.nn.Module]): Optional batch normalization
            layer associated with the current layer

    Note:
        Handles special cases like spatial pooling between Conv2D and Linear
            layers
        by adjusting mask propagation accordingly.
    """
    mask = (layer.weight_mask.flatten(1) == 0).all(1)

    # Prune the bias based on the weight mask to prevent bias units from being
    # added to the outputs of pruned neurons.
    if layer.bias is not None:
        layer.bias_mask[mask] = 0

    # Propagate the weight masking to the normalization layer (if any)
    has_bn = current_bn is not None and not isinstance(
        current_bn, torch.nn.Identity
    )
    if has_bn:
        current_bn.weight_mask[mask] = 0
        current_bn.bias_mask[mask] = 0

    # Propagate the weight masking to the following layer's input dim
    if next_layer is not None:
        # make sure we handle spatial pooling (e.g. conv2d-global pool-linear)
        if isinstance(layer, torch.nn.Conv2d) and isinstance(
            next_layer, torch.nn.Linear
        ):
            spatial_pooling = next_layer.weight.size(1) // layer.weight.size(0)
            mask = torch.repeat_interleave(mask, spatial_pooling)
        next_layer.weight_mask[:, mask] = 0


@torch.no_grad
def bridge_prune(
    current_layer,
    current_bn,
    following_layer,
    percentage=0.5,
    pruning_type="l2",
    score_threshold=None,
):
    if pruning_type == "random":
        prune.random_structured(current_layer, "weight", percentage, dim=0)
    elif pruning_type == "taylor":
        if not hasattr(current_layer, "weight_mask"):
            raise ValueError(
                "Modules should always have a weight_mask buffer when pruning"
            )
        if not hasattr(current_layer, "normalized_scores"):
            raise ValueError(
                "Normalized scores must already be computed when using Taylor "
                "scoring."
            )
        if score_threshold is None:
            raise ValueError(
                "Score threshold is required when using Taylor scoring."
            )

        # FIXME current_layer should always have a weight_mask, so don't make
        # this conditional and add a input validation that raises an exception
        # if the attribute is missing.
        neuron_scores = current_layer.normalized_scores
        dims = [-x for x in range(1, current_layer.weight_mask.ndim)]
        pruned_mask = current_layer.weight_mask.sum(dim=dims) == 0

        if torch.all(~pruned_mask):
            unpruned_neuron_scores = neuron_scores
        else:
            unpruned_neuron_scores = neuron_scores[~pruned_mask]

        thresholded_score_structured(
            current_layer,
            "weight",
            unpruned_neuron_scores,
            score_threshold,
            dim=0,
        )
    else:
        importance = get_pruning_importance(current_layer, "weight")
        if isinstance(pruning_type, int):
            norm = pruning_type
        elif isinstance(pruning_type, str):
            norm = int(pruning_type.lower().replace("l", ""))
        else:
            raise ValueError(f"Unexpected pruning norm: {pruning_type}.")
        prune.ln_structured(
            current_layer,
            "weight",
            percentage,
            n=norm,
            dim=0,
            importance_scores=importance,
        )

    propagate_mask(current_layer, following_layer, current_bn=current_bn)


@torch.no_grad
def bridge_prune_residual(in_layer, out_layer, res_layer):
    assert hasattr(in_layer, "weight_mask")
    assert hasattr(out_layer, "weight_mask")
    if not isinstance(res_layer, nn.Conv2d):
        raise ValueError("Only implement with conv residual layer for now")
    assert res_layer.bias is None
    # make sure a mask is present for the propagation to occur
    in_masked = (in_layer.weight == 0).flatten(2).all(-1)
    out_masked = (out_layer.weight == 0).flatten(2).all(-1)
    res_layer.weight_mask[:, in_masked.all(0)] = 0
    res_layer.weight_mask[out_masked.all(1)] = 0
