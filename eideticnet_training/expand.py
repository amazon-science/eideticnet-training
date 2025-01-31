import torch

from .prune.masks import UNASSIGNED_PARAMS


def expand_1d(module, add_out_features, param_name="bias", initial_value=0):
    """
    Given a torch module, add -- in place -- the specified number of output
    dimensions to the 1D parameter with the specified name.

    Args:
        module (torch.nn.Module): The module of which to expand the bias.
        add_out_features (int): The number of output features to add.
        param_name (str): The name of the parameter to expand.
        initial_value (int): The initial value of the expanded elements.
    """
    bias = getattr(module, param_name)
    new_bias = torch.empty(bias.shape[0] + add_out_features).to(bias.device)
    new_bias.fill_(initial_value)
    new_bias = new_bias.to(bias.dtype)
    new_bias[: bias.shape[0]] = bias.data
    new_bias.requires_grad = bias.requires_grad
    if isinstance(bias, torch.nn.Parameter):
        new_param = torch.nn.Parameter(new_bias)
        with torch.no_grad():
            setattr(module, param_name, new_param)
    else:
        setattr(module, param_name, new_bias)


def maybe_expand_1d(module, add_num_features, param_name, initial_value=0):
    """
    Conditionally call `expand_1d` if the named parameter exists in the module.
    """
    if hasattr(module, param_name):
        expand_1d(
            module,
            add_num_features,
            param_name=param_name,
            initial_value=initial_value,
        )


def expand_2d(
    module,
    add_in_features,
    add_out_features,
    param_name="weight",
    initial_value=0,
):
    """
    Given a torch module, add -- in place -- the specified number of input
    and/or output dimensions to the 2D parameter with the specified name.

    Args:
        module (torch.nn.Module): The module of which to expand the weight
            matrix.
        add_in_features (int): The number of input features to add.
        add_out_features (int): The number of output features to add.
        param_name (str): The name of the param
        initial_value (int): The initial value of the expanded elements.
    """
    weight = getattr(module, param_name)
    out_features, in_features = weight.shape
    new_weight = torch.empty(
        out_features + add_out_features, in_features + add_in_features
    ).to(weight.device)
    new_weight.fill_(initial_value)
    new_weight = new_weight.to(weight.dtype)
    new_weight[:out_features, :in_features] = weight.data
    new_weight.requires_grad = weight.requires_grad
    if isinstance(weight, torch.nn.Parameter):
        new_param = torch.nn.Parameter(new_weight)
        with torch.no_grad():
            setattr(module, param_name, new_param)
    else:
        setattr(module, param_name, new_weight)


def maybe_expand_2d(
    module, add_in_features, add_out_features, param_name, initial_value=0
):
    """
    Conditionally call `expand_2d` if the named parameter exists in the module.
    """
    if hasattr(module, param_name):
        expand_2d(
            module,
            add_in_features,
            add_out_features,
            param_name=param_name,
            initial_value=initial_value,
        )


def get_optional_params(param_name):
    optional_params = {
        # The mask of newly-added parameters should be 1, to indicate that they
        # have been pruned and can be reinitialized.
        f"{param_name}_mask": 0,
        f"{param_name}_orig": 0,
        f"{param_name}_excess_capacity": UNASSIGNED_PARAMS,
    }

    return optional_params


def update_trace_dict(module, trace_dict=None, add_in=None, add_out=None):
    if trace_dict is None:
        return
    trace_dict[module] = {}
    trace_dict[module]["add_in"] = 0 if add_in is None else add_in
    trace_dict[module]["add_out"] = 0 if add_out is None else add_out


def expand_linear(linear, add_in_features, add_out_features, trace_dict={}):
    """
    Given a torch Linear module, add -- in place -- the specified number of
    input and/or output features to the parameters. This function supports
    modules that are being pruned via torch.nn.utils.prune.

    Args:
        linear (torch.nn.Linear): The linear layer to expand.
        add_in_features (int): The number of input dimensions to add.
        add_out_features (int): The number of output neurons to add.
    """
    out_features, in_features = linear.weight.shape
    expand_2d(linear, add_in_features, add_out_features)

    for param_name, initial_value in get_optional_params("weight").items():
        maybe_expand_2d(
            linear,
            add_in_features,
            add_out_features,
            param_name=param_name,
            initial_value=initial_value,
        )

    if linear.bias is not None:
        expand_1d(linear, add_out_features)
        optional_bias_params = get_optional_params("bias")
        for param_name, initial_value in optional_bias_params.items():
            maybe_expand_1d(
                linear,
                add_out_features,
                param_name=param_name,
                initial_value=initial_value,
            )

    update_trace_dict(
        linear, trace_dict, add_in=add_in_features, add_out=add_out_features
    )
    linear.out_features = out_features + add_out_features
    linear.in_features = in_features + add_in_features


def expand_Nd(
    module,
    add_in_channels,
    add_out_channels,
    param_name="weight",
    initial_value=0,
):
    weight = getattr(module, param_name)
    out_channels, in_channels = weight.shape[:2]
    new_weight = torch.empty(
        out_channels + add_out_channels,
        in_channels + add_in_channels,
        *weight.shape[2:],
    ).to(weight.device)
    new_weight.fill_(initial_value)
    new_weight[:out_channels, :in_channels] = weight.data
    new_weight = new_weight.to(weight.dtype)
    new_weight.requires_grad = weight.requires_grad
    if isinstance(weight, torch.nn.Parameter):
        new_param = torch.nn.Parameter(new_weight)
        with torch.no_grad():
            setattr(module, param_name, new_param)
    else:
        setattr(module, param_name, new_weight)


def maybe_expand_Nd(
    module, add_in_channels, add_out_channels, param_name, initial_value=0
):
    """
    Conditionally call `expand_Nd` if the named parameter exists in the module.
    """
    if hasattr(module, param_name):
        expand_Nd(
            module,
            add_in_channels,
            add_out_channels,
            param_name=param_name,
            initial_value=initial_value,
        )


def expand_conv2d(conv2d, add_in_channels, add_out_channels, trace_dict=None):
    """
    Given a torch Conv2d module, add -- in place -- the specified number of
    input and/or output features to the parameters. This function supports
    modules that are being pruned via torch.nn.utils.prune.

    Args:
        conv2d (torch.nn.Conv2d): The conv2d layer to expand.
        add_in_features (int): The number of input dimensions to add.
        add_out_features (int): The number of output neurons to add.
    """
    out_channels, in_channels = conv2d.weight.shape[:2]
    expand_Nd(conv2d, add_in_channels, add_out_channels)
    for param_name, initial_value in get_optional_params("weight").items():
        maybe_expand_Nd(
            conv2d,
            add_in_channels,
            add_out_channels,
            param_name=param_name,
            initial_value=initial_value,
        )

    if conv2d.bias is not None:
        expand_1d(conv2d, add_out_channels)
        optional_bias_params = get_optional_params("bias")
        for param_name, initial_value in optional_bias_params.items():
            maybe_expand_1d(
                conv2d,
                add_out_channels,
                param_name=param_name,
                initial_value=initial_value,
            )
    update_trace_dict(
        conv2d, trace_dict, add_in=add_in_channels, add_out=add_out_channels
    )
    conv2d.out_channels = out_channels + add_out_channels
    conv2d.in_channels = in_channels + add_in_channels


def expand_bn(bn, add_num_features, trace_dict=None):
    """
    Given a torch BatchNorm module, add -- in place -- the specified number of
    output features to the parameters. This function supports modules that are
    being pruned via torch.nn.utils.prune.

    Args:
        module (torch.nn.BatchNorm*d): The batch norm layer to expand.
        add_num_features (int): The number of output features to add.
    """
    num_features = bn.num_features

    if bn.affine:
        expand_1d(bn, add_num_features, param_name="weight")
        expand_1d(bn, add_num_features, param_name="bias")
        for param_name, initial_value in get_optional_params("weight").items():
            maybe_expand_1d(
                bn,
                add_num_features,
                param_name=param_name,
                initial_value=initial_value,
            )
        for param_name, initial_value in get_optional_params("bias").items():
            maybe_expand_1d(
                bn,
                add_num_features,
                param_name=param_name,
                initial_value=initial_value,
            )

    if bn.track_running_stats:
        expand_1d(bn, add_num_features, param_name="running_mean")
        expand_1d(bn, add_num_features, param_name="running_var")
        maybe_expand_1d(
            bn, add_num_features, param_name="_frozen_running_mean"
        )
        maybe_expand_1d(bn, add_num_features, param_name="_frozen_running_var")

    bn.num_features = num_features + add_num_features

    update_trace_dict(bn, trace_dict, add_out=add_num_features)
