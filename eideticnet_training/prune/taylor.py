from collections.abc import Iterable

import torch

from .masks import assigned_params_mask


def get_already_trained_neurons(weight_excess_capacity, dims=None):
    """
    Return a boolean vector identifying which neurons or output channels have
    already been identified as important to a previous task.
    """
    if dims is None:
        dims = [-x for x in range(1, weight_excess_capacity.ndim)]
    # Element-wise mask in which any parameter that was
    # important for a previous task is true.
    excap_mask = assigned_params_mask(weight_excess_capacity)
    # This mask is true for any output channel that contains a
    # true value, which means it corresponds to an important
    # parameter for some previous task.
    return excap_mask.any(dim=dims)


def get_already_pruned_neurons(weight_mask, dims=None):
    """
    Return a boolean vector identifying which neurons or output channels have
    already been pruned for the current task.
    """
    if dims is None:
        dims = [-x for x in range(1, weight_mask.ndim)]
    already_pruned = (~weight_mask.bool()).all(dim=dims)
    return already_pruned


class TaylorScorer:
    """
    An implementation of neuron importance estimation based on the
    first-order Taylor expansion of the change in loss when pruning a
    particular neuron.

    See https://openreview.net/forum?id=SJGCiw5gl for original paper.
    """

    def __init__(self, modules):
        if not isinstance(modules, Iterable):
            raise ValueError("Modules must be iterable")

        self.modules = modules
        self.assert_masks_exist = False
        self.assert_already_trained_nonzero = False

    def register_hooks(self):
        self.hooks = []
        for module in self.modules:
            self.hooks.append(module.register_forward_hook(self.forward_hook))
            self.hooks.append(
                module.register_full_backward_hook(self.backward_hook)
            )

    def unregister_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def remove_buffers(self):
        for module in self.modules:
            if hasattr(module, "raw_scores"):
                delattr(module, "raw_scores")
            if hasattr(module, "normalized_scores"):
                delattr(module, "normalized_scores")

    def zero_buffers(self):
        for module in self.modules:
            if hasattr(module, "raw_scores"):
                # no_longer_prunable = module.raw_scores.isinf()
                module.raw_scores.fill_(0)
                # module.raw_scores[no_longer_prunable] = torch.inf
            if hasattr(module, "normalized_scores"):
                # no_longer_prunable = module.normalized_scores.isinf()
                module.normalized_scores.fill_(0)
                # module.normalized_scores[no_longer_prunable] = torch.inf

    def forward_hook(self, module, input, output):
        module.output = output

    def backward_hook(self, module, grad_input, grad_output):
        """
        For reference implementation, see:

            https://github.com/NVlabs/Taylor_pruning/blob/master/pruning_engine.py
        """
        if not hasattr(module, "raw_scores"):
            num_neurons = module.weight.shape[0]
            module.register_buffer(
                "raw_scores", torch.zeros(num_neurons).to(module.weight.device)
            )

        raw_scores = (grad_output[0] * module.output).abs()
        if len(grad_output[0].shape) == 4:
            # Aggregate over spatial dimensions and the batch to get the
            # average score for the output channel dimension.
            raw_scores = raw_scores.mean(-1).mean(-1).mean(0)
        else:
            raw_scores = raw_scores.mean(0)

        module.raw_scores += raw_scores

    def partition_scores(self):
        """
        Partition the scores such that neurons that were previously pruned or
        have been frozen due to their importance to a previous task always have
        the highest score and hence will not be pruned again.
        """
        for module in self.modules:
            if self.assert_masks_exist:
                assert hasattr(module, "weight_mask")
                assert hasattr(module, "weight_excess_capacity")

            if hasattr(module, "weight_excess_capacity"):
                # Prevent pruning of neurons that are frozen because they are
                # important to a previously-trained task.
                already_trained = get_already_trained_neurons(
                    module.weight_excess_capacity
                )
                module.raw_scores[already_trained] = torch.inf

            if hasattr(module, "weight_mask"):
                # Prevent pruning of neurons that have already been pruned.
                # A neuron has been pruned in this round of iterative pruning
                # when all elements of the mask for the output neuron/channel
                # are 0.
                already_pruned = get_already_pruned_neurons(module.weight_mask)
                module.raw_scores[already_pruned] = torch.inf

    def normalize_scores(self):
        """
        Compute the normalized scores by dividing the scores by the global norm
        (i.e. the L2 norm of all scores).
        """

        def get_global_norm():
            # Return the L2 norm of all scores, excluding those set to inf.
            to_cat = [
                module.raw_scores
                for module in self.modules
                if hasattr(module, "raw_scores")
            ]
            accumulator = torch.cat(to_cat)
            # Exclude scores of neurons that should be ignored.
            accumulator = accumulator[~accumulator.isinf()]
            return (accumulator**2).sum().sqrt()

        global_norm = get_global_norm()

        for module in self.modules:
            if hasattr(module, "raw_scores"):
                raw_scores = module.raw_scores.clone()
                module.normalized_scores = raw_scores / global_norm

    def clear(self):
        self.modules.clear()
