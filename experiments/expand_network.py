import torch
import torch.nn as nn

from eideticnet_training import expand


def get_amount(*, amount=None, module=None, weight_mask=None):
    if amount is None:
        raise ValueError("'amount' is required")
    if 0 < amount < 1:
        return int(amount * module.weight.shape[0])
    elif amount >= 1:
        return int(amount)
    else:
        raise ValueError(f"'amount' must be a positive number")


def get_num_neurons_assigned(*, module=None, weight_mask=None):
    """
    A policy for expanding the modules of an EideticNet such that, when
    training a new task, there is always the same amount of free, unallocated
    capacity as the network had before training any task.

    Returns the number of neurons that were frozen when training the previous
    task.
    """
    if weight_mask is None:
        if module is None:
            raise ValueError(
                "Either 'module' or 'weight_mask' must be provided."
            )
        if hasattr(module, "weight_mask"):
            weight_mask = module.weight_mask
        else:
            raise ValueError("Module does not have a 'weight_mask' buffer.")
    if weight_mask.ndim == 2:
        neurons_to_be_frozen = torch.all(weight_mask == 1, dim=1)
    elif weight_mask.ndim == 4:
        neurons_to_be_frozen = torch.all(weight_mask == 1, dim=(1, 2, 3))
    else:
        raise ValueError("This function requires a 2D or 4D input.")
    num_neurons_to_add = neurons_to_be_frozen.sum()
    return num_neurons_to_add


def expand_mlp(*, model, add_out_func):
    """
    User-defined policy for expanding the capacity of the layers of our
    MLP. After training the network on a task, and applying iterative
    pruning, the policy adds to a layer exactly the number of neurons that
    were not pruned. The rationale of this policy is that when training a
    new task, there should always be the same amount of free, unallocated
    capacity as the network started with.
    """
    add_in_features = 0
    trace_dict = {}

    for i in range(len(model.blocks)):
        # Expand the Linear layer. Here we add as many neurons as remain
        # after pruning, because those neurons will be frozen and not
        # available for optimizing a downstream task, but this is a policy
        # choice. We should allow the user to design their own policy and
        # choose by how much to expand a layer.
        linear = model.blocks[i][0]
        add_out_features = add_out_func(module=linear)
        expand.expand_linear(
            linear, add_in_features, add_out_features, trace_dict
        )
        maybe_bn = model.blocks[i][1]
        if isinstance(maybe_bn, torch.nn.BatchNorm1d):
            expand.expand_bn(maybe_bn, add_out_features, trace_dict)

        add_in_features = add_out_features

    if add_in_features:
        for i in range(len(model.classifiers)):
            expand.expand_linear(
                model.classifiers[i], add_in_features, 0, trace_dict
            )

    return trace_dict


def expand_convnet(*, model, add_out_func):
    """
    User-defined policy for expanding the capacity of the layers of our
    MLP. After training the network on a task, and applying iterative
    pruning, the policy adds to a layer exactly the number of neurons that
    were not pruned. The rationale of this policy is that when training a
    new task, there should always be the same amount of free, unallocated
    capacity as the network started with.
    """
    add_in_features = 0
    trace_dict = {}

    for i in range(len(model.blocks)):
        conv2d = model.blocks[i][0]
        add_out_features = add_out_func(module=conv2d)
        expand.expand_conv2d(
            conv2d, add_in_features, add_out_features, trace_dict
        )

        maybe_bn = model.blocks[i][1]
        if isinstance(maybe_bn, torch.nn.BatchNorm2d):
            expand.expand_bn(maybe_bn, add_out_features, trace_dict)

        add_in_features = add_out_features
    if add_in_features:
        for i in range(len(model.classifiers)):
            expand.expand_linear(
                model.classifiers[i], add_in_features, 0, trace_dict
            )


def expand_resnet(*, model, add_out_func):
    """
    User-defined procedure for expanding the layers of our ResNet. After
    training the network on a task, and applying iterative pruning, the
    procedure traverses the modules in the network and inspects each one with
    `add_out_func` to determine by how much to expand it.

    """

    def is_conv_or_bn(m):
        return isinstance(m, (nn.Conv2d, nn.BatchNorm2d))

    add_out_channels = 0
    add_in_channels = 0
    trace_dict = {}

    for i in range(len(model.blocks)):
        print(f"expand_resnet block {i}")
        res_add_in_channels = add_in_channels
        block = model.blocks[i]

        # Expand each conv and its subsequent batch norm layer.
        convs_and_bns = list(filter(is_conv_or_bn, block.children()))
        for j, layer in enumerate(convs_and_bns):
            if isinstance(layer, nn.Conv2d):
                add_out_channels = add_out_func(module=layer)
                print(
                    f"expand_resnet block {i} "
                    f"add_out_channnels {add_out_channels}"
                )
                expand.expand_conv2d(
                    layer, add_in_channels, add_out_channels, trace_dict
                )
                if j + 1 < len(convs_and_bns):
                    bn = convs_and_bns[j + 1]
                    if isinstance(bn, torch.nn.BatchNorm2d):
                        expand.expand_bn(bn, add_out_channels, trace_dict)
                add_in_channels = add_out_channels

        # Expand the residual conv by the number of new input channels of the
        # entire block's ingress layer (res_add_in_channels) and the number of
        # new output channels of the block's egress layer (add_out_channels, at
        # this point).
        residual_conv = model.res[i]
        res_add_out_channels = add_out_channels
        expand.expand_conv2d(
            residual_conv, res_add_in_channels, res_add_out_channels, trace_dict
        )

    if add_in_channels:
        for i in range(len(model.classifiers)):
            if isinstance(model.classifiers[i], nn.Linear):
                expand.expand_linear(
                    model.classifiers[i], add_in_channels, 0, trace_dict
                )

    return trace_dict
