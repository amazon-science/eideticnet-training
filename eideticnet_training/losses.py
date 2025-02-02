import warnings

import torch
import torch.nn.functional as F


def soft_orthogonality_loss(weight, reduction="mean", normalize=True):
    """Maximize the orthogonality of neurons in a weight matrix.

    The name "soft orthogonality" (SO) comes from [1].

        [1] https://proceedings.neurips.cc/paper/2018/hash/bf424cb7b0dea050a42b9739eb261a3a-Abstract.html  # noqa: E501

    Arguments
    -----------
    weight (torch.Tensor): the weight matrix over which to compute the SO loss
    reduction (None, str): the reduction to apply to the output: 'none' |
        'mean' | 'sum'. 'none': no reduction will be applied, 'mean': the sum
        of the output will be divided by the number of elements in the output,
        'sum': the output will be summed. Note: size_average and reduce are in
        the process of being deprecated, and in the meantime, specifying either
        of those two args will override reduction. Default: 'mean'.
    normalize (bool): whether to normalize the rows of `weight` to unit length
        before computing the loss. Default: False.

    Returns
    -----------
    loss (float): the SO loss of the weight matrix
    """
    # View Nd weight tensors as 2d.
    if weight.dim() > 2:
        weight = weight.view(weight.size(0), -1)

    if normalize:
        weight_ = F.normalize(weight, p=2, dim=-1)
    else:
        weight_ = weight

    cosine = torch.matmul(weight_, weight_.t())
    cosine = cosine - 2.0 * torch.diag(torch.diag(cosine))

    # Optionally reduction.
    if reduction is None:
        loss = cosine
    elif reduction == "mean":
        loss = cosine.mean()
    elif reduction == "sum":
        loss = cosine.sum()
    else:
        raise ValueError(f"unknown reduction argument value {reduction}")

    # Adjust range of data so the minimum is 0.
    if reduction is None:
        pass
    elif reduction == "mean":
        loss = loss + 1 / weight.shape[0]
    elif reduction == "sum":
        loss = loss + weight.shape[0]
    else:
        raise ValueError(f"unknown reduction argument value {reduction}")

    return loss


def double_soft_orthogonality_loss(weight, reduction="mean", normalize=True):
    """Minimize the similarity among neurons in a weight matrix.

    The name "double soft orthgonality" (DSO) comes from [1]. The deficiency of
    the more conventional ("soft") orthogonality loss is evident when the
    weight matrix is over- or under-complete. In those cases, the soft
    orthogonality loss of the weight matrix can be minimized while the soft
    orthogonality of its transpose remains high. (This is not possible when the
    weight matrix is square.)

        [1] https://proceedings.neurips.cc/paper/2018/hash/bf424cb7b0dea050a42b9739eb261a3a-Abstract.html  # noqa: E501

    Arguments
    -----------
    weight (torch.Tensor): the weight matrix over which to compute the DSO loss
    reduction (None, str): the reduction to apply to the output: 'none' |
        'mean' | 'sum'. 'none': no reduction will be applied, 'mean': the sum
        of the output will be divided by the number of elements in the output,
        'sum': the output will be summed. Note: size_average and reduce are in
        the process of being deprecated, and in the meantime, specifying either
        of those two args will override reduction. Default: 'mean'.
    normalize (bool): whether to normalize the rows of `weight` to unit length
        before computing the loss. Default: False.

    Returns
    -----------
    loss (float): the DSO loss of the weight matrix
    """
    # View Nd weight tensors as 2d.
    if weight.dim() > 2:
        weight = weight.view(weight.size(0), -1)

    so_loss = soft_orthogonality_loss
    kwargs = dict(reduction=reduction, normalize=normalize)
    return so_loss(weight, **kwargs) + so_loss(weight.t(), **kwargs)


def mutual_coherence_loss(weight, dim=None, normalize=True):
    """Minimize the mutual coherence loss.

    Mutual coherence (MC) loss is defined in [1]. It is the infinity norm of
    the soft orthogonality loss (prior to reduction).

        [1] https://proceedings.neurips.cc/paper/2018/hash/bf424cb7b0dea050a42b9739eb261a3a-Abstract.html  # noqa:   E501

    Arguments
    -----------
    weight (torch.Tensor): the weight matrix over which to compute the MC loss

    Returns
    -----------
    loss (float): the MC loss of the weight matrix
    """
    so_loss = soft_orthogonality_loss(
        weight, reduction=None, normalize=normalize
    )
    if dim is int:
        _, mc_loss_indices = so_loss.max(dim=dim)
        return so_loss[mc_loss_indices].mean()
    else:
        return so_loss.max()


def maximize_minimal_angle_loss(weight, normalize=True):
    """Maximize the minimal angles (MMA) among neurons in a weight matrix.

    Compute the MMA loss over the neurons in the given weight tensor. MMA loss
    was defined in [1]. Code is available at [2].

        [1] https://proceedings.neurips.cc/paper/2020/hash/dcd2f3f312b6705fb06f4f9f1b55b55c-Abstract.html  # noqa: E501
        [2] https://github.com/wznpub/MMA_Regularization

    Arguments
    -----------
    weight (torch.Tensor): the weight matrix over which to compute the MMA loss

    Returns
    -----------
    loss (float): the MMA loss of the weight matrix
    """

    # View Nd weight tensors as 2d.
    if weight.dim() > 2:
        weight = weight.view(weight.size(0), -1)

    # Set the norm of the rows to unit norm before computing similarity.
    if normalize:
        weight_ = F.normalize(weight, p=2, dim=1)
    else:
        weight_ = weight
    cosine = torch.matmul(weight_, weight_.t())

    # Remove the contribution of the diagonal from the loss.
    cosine = cosine - 2.0 * torch.diag(torch.diag(cosine))

    # Maximize the minimal angle between the neurons.
    loss = -torch.acos(cosine.max(dim=1)[0].clamp(-0.99999, 0.99999)).mean()

    return loss


def gini_loss(x, reduction="mean", dim=None, use_abs=True):
    """Compute Gini coefficient.

    The Gini coefficient [1] measures income inequality. The range of the
    function is [0, 1]. The most equal distribution of income has a Gini
    coefficient of 0; conversely, the most unequal has a Gini coefficient of 1.
    Specifically, inequality is minimized by the Dirac delta function (all
    members of the population have the same amount) and maximized by a bi-modal
    distribution in which one member of the populuation has all of the wealth.

    This implementation is based on that found in [2].

    [1] https://en.wikipedia.org/wiki/Gini_coefficient
    [2] https://stackoverflow.com/a/61154922

    Arguments
    ---------
    x (torch.Tensor): the input to the function

    Returns
    --------
    coefficient (float): the Gini coefficient of the input

    """

    if x.ndim not in {1, 2}:
        raise ValueError("Input 'x' must be 1- or 2-dimensional")

    if torch.any(x < 0.0):
        warnings.warn(
            "Gini coeffficient with negative inputs may be incorrect",
            category=UserWarning,
        )
        if x.ndim == 1:
            if use_abs:
                x = x.abs()
            else:
                x = x - x.min()
        elif x.ndim == 2:
            if use_abs:
                x = x.abs()
            else:
                x = x - x.min(dim=1, keepdim=True)[0]

    def compute_gini_1d(x):
        assert x.ndim == 1
        diffsum = 0
        for i, xi in enumerate(x[:-1], 1):
            diffsum += torch.sum(torch.abs(xi - x[i:]))
        coef = diffsum / (len(x) ** 2 * torch.mean(x))
        return coef

    def compute_gini_2d(x):
        assert x.ndim == 2
        coefs = []
        for i in range(len(x)):
            coefs.append(compute_gini_1d(x[i]))
        return torch.tensor(coefs)

    if x.ndim == 1:
        return compute_gini_1d(x)
    else:
        loss = compute_gini_2d(x)
        if reduction is None:
            return loss
        elif reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        else:
            raise ValueError(f"unknown reduction argument value {reduction}")


def compute_moments(x, dim=None, custom_moment=5.0):
    """Compute the first four moments of a tensor.

    If dim is None (the default), the moments are computed across all
    dimensions of the tensor and scalar moments are returned. Otherwise, the
    moments are computed across the specified dimension(s).

    Arguments
    ---------
    x (torch.Tensor): the tensor
    dim (int, tuple of ints, or None): the dimension(s) across which to compute
        moments.
    """
    kwargs = dict() if dim is None else dict(dim=dim, keepdim=True)
    mean = torch.mean(x.detach(), **kwargs)
    diffs = x - mean
    var = torch.mean(torch.pow(diffs.detach(), 2.0), **kwargs)
    std = torch.pow(var, 0.5)
    zscores = diffs / std
    skewness = torch.mean(torch.pow(zscores, 3.0), **kwargs)
    kurtosis = torch.mean(torch.pow(zscores, 4.0), **kwargs)
    custom = torch.mean(torch.pow(zscores, custom_moment), **kwargs)
    return mean, std, skewness, kurtosis, custom


def kurtosis_loss(x, dim=None, excess_kurtosis=0):
    _, _, _, kurtosis, _ = compute_moments(x, dim=dim)
    return kurtosis - excess_kurtosis
