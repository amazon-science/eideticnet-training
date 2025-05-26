from collections import OrderedDict

import torch

from eideticnet_training.prune import TaylorScorer
from eideticnet_training.prune.threshold import thresholded_score_structured
from eideticnet_training.prune import bridge_prune
from eideticnet_training.prune import bridge_prune_residual


# FIXME this is a duplicate of what's in eideticnet_training.utils.prune.
def _extract_first_two_dims(tensor):
    if tensor.dim() < 2:
        raise ValueError("Tensor must have at least 2 dimensions")
    return tensor[(slice(None), slice(None)) + (0,) * (tensor.dim() - 2)]


def reshape_scores_like_tensor(scores, tensor):
    if scores.shape != tensor.shape:
        # Reshape is needed, but check whether the shapes are compatible.
        ndim = scores.ndim
        if scores.shape[:ndim] != tensor.shape[:ndim]:
            raise ValueError(
                f"The score tensor dimensions ({scores.shape}) "
                "differ from the leading dimensions of the parameter "
                f"tensor ({tensor.shape[ndim]}), so the scores cannot "
                "be broadcast along the trailing dimensions."
            )
        for _ in range(tensor.ndim - 1):
            scores.unsqueeze_(-1)
        scores = scores.repeat(1, *tensor.shape[ndim:])
    return scores


def _verify_pruning(model, scores, score_threshold, dim):
    for module_name, score_tensor in scores.items():
        module = getattr(model, module_name)

        if hasattr(module, "weight_mask"):
            mask_with_leading_dims = _extract_first_two_dims(module.weight_mask)
            scores_of_not_pruned = score_tensor[
                mask_with_leading_dims[:, 0].bool()
            ]
        else:
            scores_of_not_pruned = score_tensor

        thresholded_score_structured(
            module, "weight", scores_of_not_pruned, score_threshold, dim
        )

        # Verify torch's pruning added the weight pruning mask.
        assert hasattr(module, "weight_mask")

        # Verify that right neurons were pruned.
        expected_weight_mask = (score_tensor > score_threshold).to(float)
        expected_weight_mask = reshape_scores_like_tensor(
            expected_weight_mask, module.weight
        )
        assert torch.all(module.weight_mask == expected_weight_mask)

        assert torch.all(module.weight[~module.weight_mask.bool()] == 0.0)
        assert torch.all(module.weight[module.weight_mask.bool()] != 0.0)


def test_thresholded_score_structured_linear():
    """
    Verify ThresholdedScoreStructuredPruning prunes the right Linear neurons.
    """
    model = torch.nn.Sequential(
        OrderedDict(
            [
                ("linear1", torch.nn.Linear(5, 5, bias=False)),
                ("linear2", torch.nn.Linear(5, 5, bias=False)),
                ("linear3", torch.nn.Linear(5, 5, bias=False)),
            ]
        )
    )

    modules = list(model.children())

    scorer = TaylorScorer(modules)

    model.linear1.raw_scores = torch.tensor([15, 13, 11, 9, 7])
    model.linear2.raw_scores = torch.tensor([14, 12, 10, 8, 6])
    model.linear3.raw_scores = torch.tensor([50, 50, 50, 50, 50])

    scorer.normalize_scores()

    scores = {
        "linear1": model.linear1.normalized_scores,
        "linear2": model.linear2.normalized_scores,
        "linear3": model.linear3.normalized_scores,
    }
    all_scores = torch.cat(tuple(scores.values())).flatten()

    # Prune the 3 least-important neurons. With Taylor pruning, the scores
    # of more important neurons are higher.
    score_threshold, _ = torch.kthvalue(all_scores, k=3)
    # The normalized score corresponding to the raw_score of 8 above.
    assert score_threshold == model.linear2.normalized_scores[3]

    _verify_pruning(model, scores, score_threshold, dim=0)
    # And verify that only linear1 and linear2 are pruned.
    assert not torch.all(model.linear1.weight_mask == 1)
    assert not torch.all(model.linear2.weight_mask == 1)
    assert torch.all(model.linear3.weight_mask == 1)

    # Prune the next 3 least-important neurons.
    score_threshold, _ = torch.kthvalue(all_scores, k=6)
    # The normalized score corresponding to the raw_score of 11 above.
    assert score_threshold == model.linear1.normalized_scores[2]

    _verify_pruning(model, scores, score_threshold, dim=0)
    # And verify that only linear1 and linear2 are pruned.
    assert not torch.all(model.linear1.weight_mask == 1)
    assert not torch.all(model.linear2.weight_mask == 1)
    assert torch.all(model.linear3.weight_mask == 1)


def test_thresholded_score_structured_conv2d():
    """
    Verify ThresholdedScoreStructuredPruning prunes the right Conv2d neurons.
    """
    model = torch.nn.Sequential(
        OrderedDict(
            [
                ("conv1", torch.nn.Conv2d(5, 5, 3, bias=False)),
                ("conv2", torch.nn.Conv2d(5, 5, 3, bias=False)),
                ("conv3", torch.nn.Conv2d(5, 5, 3, bias=False)),
            ]
        )
    )

    modules = list(model.children())

    scorer = TaylorScorer(modules)

    model.conv1.raw_scores = torch.tensor([15, 13, 11, 9, 7])
    model.conv2.raw_scores = torch.tensor([14, 12, 10, 8, 6])
    model.conv3.raw_scores = torch.tensor([50, 50, 50, 50, 50])

    scorer.normalize_scores()

    scores = {
        "conv1": model.conv1.normalized_scores,
        "conv2": model.conv2.normalized_scores,
        "conv3": model.conv3.normalized_scores,
    }
    all_scores = torch.cat(tuple(scores.values())).flatten()

    # Prune the 3 least-important neurons/channels. With Taylor pruning, the
    # scores of more important neurons are higher.
    score_threshold, _ = torch.kthvalue(all_scores, k=3)
    # The normalized score corresponding to the raw_score of 8 above.
    assert score_threshold == model.conv2.normalized_scores[3]

    _verify_pruning(model, scores, score_threshold, dim=0)
    # And verify that only conv1 and conv2 are pruned.
    assert not torch.all(model.conv1.weight_mask == 1)
    assert not torch.all(model.conv2.weight_mask == 1)
    assert torch.all(model.conv3.weight_mask == 1)

    # Prune the next 3 least-important neurons/channels.
    score_threshold, _ = torch.kthvalue(all_scores, k=6)
    # The normalized score corresponding to the raw_score of 11 above.
    assert score_threshold == model.conv1.normalized_scores[2]

    _verify_pruning(model, scores, score_threshold, dim=0)
    # And verify that only conv1 and conv2 are pruned.
    assert not torch.all(model.conv1.weight_mask == 1)
    assert not torch.all(model.conv2.weight_mask == 1)
    assert torch.all(model.conv3.weight_mask == 1)


def test_smoke_bridge_prune():
    """
    Smoke test of bridge_prune.
    """
    current_layer = torch.nn.Linear(10, 10, bias=True)
    current_bn = torch.nn.BatchNorm1d(10)
    following_layer = torch.nn.Linear(10, 3, bias=True)

    current_layer.bias_mask = torch.zeros_like(current_layer.bias)
    current_bn.weight_mask = torch.zeros_like(current_bn.weight)
    current_bn.bias_mask = torch.zeros_like(current_bn.bias)
    following_layer.weight_mask = torch.zeros_like(following_layer.weight)

    bridge_prune(
        current_layer=current_layer,
        current_bn=current_bn,
        following_layer=following_layer,
        percentage=0.5,
        pruning_type=2,
    )


def test_smoke_bridge_prune_residual():
    """
    Smoke test of bridge_prune_residual with Conv2d layers, mimicking ResNet's usage pattern.
    """
    # Setup layers matching ResNet's block structure
    in_layer = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
    out_layer = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1)
    res_layer = torch.nn.Conv2d(64, 256, kernel_size=1, bias=False)

    # Initialize weight masks as required by bridge_prune_residual
    in_layer.weight_mask = torch.ones_like(in_layer.weight)
    out_layer.weight_mask = torch.ones_like(out_layer.weight)
    res_layer.weight_mask = torch.ones_like(res_layer.weight)

    # Test the residual pruning
    bridge_prune_residual(in_layer, out_layer, res_layer)
