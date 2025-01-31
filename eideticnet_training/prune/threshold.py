import torch
import torch.nn.utils.prune as torch_prune


class ThresholdScoreStructuredPruning(torch_prune.BasePruningMethod):
    PRUNING_TYPE = "structured"

    def __init__(self, scores, threshold, dim):
        """
        TODO document how this class is used in the framework.

        scores (torch.Tensor): The scores of the neurons in this module.
        threshold (int): The threshold is the kth smallest score.
        """
        self.dim = dim
        # If scores is 0-length, no capacity is remaining in this layer to be
        # pruned now. It has either been allocated to a previous task or it has
        # been pruned in this task. In that case, in compute_mask, return the
        # default mask.
        self.scores = scores.clone()
        self.threshold = threshold

    def make_mask(self, t, dim, indices):
        mask = torch.zeros_like(t)
        to_keep = [slice(None)] * len(t.shape)
        to_keep[dim] = indices
        mask[to_keep] = 1
        return mask

    def compute_mask(self, tensor, default_mask):
        """
        The scores for this tensor are pre-computed and the tensor is only used
        for shape checking.
        """
        if len(self.scores):
            indices = self.scores > self.threshold
            mask = self.make_mask(tensor, self.dim, indices)
            mask *= default_mask.to(dtype=mask.dtype)
        else:
            # Return the default mask if all neurons have allocated to a
            # previous task or already pruned in this task.
            mask = default_mask
        return mask


def thresholded_score_structured(module, name, scores, threshold, dim):
    ThresholdScoreStructuredPruning.apply(module, name, scores, threshold, dim)
    return module
