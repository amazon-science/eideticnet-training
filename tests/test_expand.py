import math
import os
from itertools import product

import numpy as np
import pytest
import torch
import torch.nn as nn
import wandb
from torch.nn.utils import prune

import eideticnet_training as eideticnet
import eideticnet_training.expand as expand
from eideticnet_training import EideticNetwork
from eideticnet_training.prune.hooks import (
    register_eidetic_hooks,
    unregister_eidetic_hooks,
)

BATCH_SIZE = 8
SEED = 1


os.environ["WANDB_MODE"] = "disabled"
wandb.init()


@torch.no_grad
def initialize_expanded_weights(module, add_in, add_out, seed=1):
    torch.manual_seed(seed)
    placeholder = module.weight.clone()
    torch.nn.init.kaiming_uniform_(placeholder, a=math.sqrt(5))
    index = np.index_exp[-add_in:, -add_out:]
    module.weight_orig[index] = placeholder[index]


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
        neurons_to_be_frozen = ~torch.all(weight_mask == 0, dim=1)
    elif weight_mask.ndim == 4:
        neurons_to_be_frozen = ~torch.all(weight_mask == 0, dim=(1, 2, 3))
    else:
        raise ValueError("This function requires a 2D or 4D input.")
    num_neurons_to_add = neurons_to_be_frozen.sum()
    return num_neurons_to_add


@pytest.mark.parametrize(
    "dtype", (torch.float64, torch.float32, torch.bfloat16)
)
def test_expand_matmul(dtype):
    """
    Test numerical stability of matmul when matrix expands.

    This test indicates that expanding a small matrix with 0s can change the
    results of matrix multiplications. This may be a factor in the instability
    we observe when expanding EideticNets.
    """
    batch_size = 2
    in_features = 2
    out_features = 1
    x = torch.randn(batch_size, in_features).to(dtype)
    w = torch.randn(out_features, in_features).to(dtype)
    w2 = torch.zeros(out_features + 1, in_features).to(dtype)
    w2[0] = w.flatten()
    y = x @ w.T
    y2 = (x @ w2.T)[:, :1]
    assert torch.all(y == y2)


def _test_expand_linear(model, x, y):
    """Verify expanding Linear under different settings."""
    network_expansions = [
        # Add one intermediate feature.
        ((0, 1), (1, 0)),
        # Add one input feature.
        ((1, 0), (0, 0)),
        # Add two input and one intermediate feature.
        ((2, 1), (1, 0)),
        # Add three input and one output feature.
        ((3, 0), (0, 1)),
    ]

    for network_expansion in network_expansions:
        # Expand all of the layers of the network.
        for i, layer_expansion in enumerate(network_expansion):
            out_features, in_features = (
                model[i].out_features,
                model[i].in_features,
            )
            expand.expand_linear(model[i], *layer_expansion)
            assert model[i].in_features == in_features + layer_expansion[0]
            assert model[i].out_features == out_features + layer_expansion[1]

        print(model[0].weight, model[0].bias)
        print(model[1].weight, model[1].bias)

        # Verify the outputs have not changed.
        x_expanded = torch.zeros(BATCH_SIZE, model[0].in_features)
        x_expanded[:, : x.shape[1]] = x.clone()
        y_expanded = model(x_expanded)
        if isinstance(model, EideticNetwork):
            assert isinstance(y_expanded, list)
            y_expanded = y_expanded[0]
        assert torch.all(y_expanded[:, : y.shape[1]] == y)


@pytest.mark.parametrize(
    "in_features, hidden_features, with_bias",
    product((1, 3), (1, 3), (True, False)),
)
def test_expand_linear(in_features, hidden_features, with_bias):
    """Verify expanding Linear."""
    model = nn.Sequential(
        nn.Linear(in_features, hidden_features, bias=with_bias),
        nn.Linear(hidden_features, 1, bias=with_bias),
    )
    x = torch.randn(BATCH_SIZE, in_features)
    y = model(x)
    _test_expand_linear(model, x, y)


@pytest.mark.parametrize("with_bias", (True, False))
def test_expand_linear_torch_prune(with_bias):
    """Verify expanding Linear while using torch prune."""
    model = nn.Sequential(
        nn.Linear(10, 10, bias=with_bias), nn.Linear(10, 1, bias=with_bias)
    )
    prune.identity(model[0], "weight")
    prune.identity(model[1], "weight")
    if with_bias:
        prune.identity(model[0], "bias")
        prune.identity(model[1], "bias")
    model[0].weight_mask[:2] = 0
    model[1].weight_mask[-3:] = 0
    x = torch.randn(BATCH_SIZE, 10)
    y = model(x)
    _test_expand_linear(model, x, y)


@pytest.mark.parametrize("with_bias", (True, False))
def test_expand_linear_eideticnet(with_bias):
    """Verify expanding Linear EideticNet."""

    class EideticLinear(EideticNetwork):
        def __init__(self, with_bias):
            super().__init__()
            self.blocks = nn.ModuleList(
                [
                    nn.Linear(10, 10, bias=with_bias),
                    nn.Linear(10, 1, bias=with_bias),
                ]
            )
            self.classifiers = [nn.Identity()]

        def __getitem__(self, index):
            return self.blocks[index]

        def forward(self, x):
            for block in self.blocks:
                x = block(x)
            return [c(x) for c in self.classifiers]

    model = EideticLinear(with_bias)
    model.set_phase(0)
    model[0].weight_mask[:2] = 0
    model[1].weight_mask[-3:] = 0
    x = torch.randn(BATCH_SIZE, 10)
    y = model(x)[0]
    _test_expand_linear(model, x, y)


def test_equivalence_of_expanding_and_pruning_linear_eideticnet():
    """
    Verify the network output is the same after expanding as it is after
    the equivalent pruning.
    """
    seed = 1

    class LinearEideticNet(EideticNetwork):
        def __init__(self, in_features, out_features):
            super().__init__()
            torch.manual_seed(seed)
            self.linear = nn.Linear(in_features, out_features, bias=False)
            self.classifiers = [nn.Identity(), nn.Identity()]

        def forward(self, x):
            x = self.linear(x)
            return [c(x) for c in self.classifiers]

    torch.manual_seed(seed)

    x = torch.randn(1, 1)

    ###########################################################################
    # In the baseline setting, we prune a 2-neuron network.
    ###########################################################################
    baseline = LinearEideticNet(in_features=1, out_features=2)
    baseline.prepare_for_task(0)
    baseline.linear.weight_mask[1] = 0

    y_phase0 = baseline(x)[0]
    no_change0, should_change0 = y_phase0[0]
    assert no_change0 != 0
    assert should_change0 == 0

    baseline_mask = baseline.linear.weight_mask.clone()
    baseline_excess_capacity = baseline.linear.weight_excess_capacity.clone()

    # Set the RNG seed before initializing unallocated parameters.
    torch.manual_seed(seed)
    baseline.prepare_for_task(1)
    y_phase1 = baseline(x)[0]
    no_change1, should_change1 = y_phase1[0]
    assert no_change1 == no_change0
    assert should_change1 != 0
    assert should_change1 != should_change0

    ###########################################################################
    # In the expanding setting, we expand a 1-neuron network to have 2.
    ###########################################################################
    expander = LinearEideticNet(in_features=1, out_features=1)
    expander.prepare_for_task(0)
    trace_dict = {}
    expand.expand_linear(expander.linear, 0, 1, trace_dict=trace_dict)
    y_expand_phase0 = expander(x)[0]
    expand_no_change0, expand_should_change0 = y_expand_phase0[0]
    assert expand_no_change0 == no_change0
    assert expand_should_change0 == should_change0

    # The torch pruning mask of the newly-added neurons should be the
    # same as those of pruned neurons.
    assert torch.all(expander.linear.weight_mask == baseline_mask)
    assert torch.all(
        expander.linear.weight_excess_capacity == baseline_excess_capacity
    )

    # Set the RNG seed before initializing unallocated parameters.
    torch.manual_seed(seed)
    expander.prepare_for_task(1)

    # Achieving the equivalence requires random initialization of the
    # newly-added parameters in *_orig. Ideally, the `expand` module will
    # always add randomly-initialized weight_orig parameters. We'll need a
    # way for users to specify their own initialization function.
    initialize_expanded_weights(expander.linear, 0, 1, seed=seed)

    y_expand_phase1 = expander(x)[0]
    expand_no_change1, expand_should_change1 = y_expand_phase1[0]
    assert expand_no_change1 == no_change0
    assert expand_should_change1 == should_change1


def _test_expand_conv2d(model, x, y):
    """Verify expanding Linear under different settings."""
    network_expansions = [
        # Add one input, two intermediate channels, and one output channel.
        ((1, 2), (2, 1)),
        # Add four input, one intermediate, and zero output channels.
        ((4, 1), (1, 0)),
        # Add one input, zero intermediate, and one output channel.
        ((1, 0), (0, 1)),
    ]

    for network_expansion in network_expansions:
        # Expand all of the layers of the network.
        for i, layer_expansion in enumerate(network_expansion):
            out_channels, in_channels = (
                model[i].out_channels,
                model[i].in_channels,
            )
            expand.expand_conv2d(model[i], *layer_expansion)
            assert model[i].in_channels == in_channels + layer_expansion[0]
            assert model[i].out_channels == out_channels + layer_expansion[1]

        # Verify the output has not changed.
        x_expanded = torch.zeros(BATCH_SIZE, model[0].in_channels, 4, 4)
        x_expanded[:, : x.shape[1]] = x.clone()
        assert torch.all(x_expanded[:, x.shape[1] :] == 0)

        y_expanded = model(x_expanded)
        if isinstance(model, EideticNetwork):
            assert isinstance(y_expanded, list)
            y_expanded = y_expanded[0]
        assert y_expanded[:, : y.shape[1]].shape == y.shape
        assert torch.allclose(y_expanded[:, : y.shape[1]], y)


@pytest.mark.parametrize("with_bias", (False, True))
def test_expand_conv2d(with_bias):
    """Verify expanding Conv2d."""
    model = nn.Sequential(
        nn.Conv2d(3, 3, kernel_size=2, bias=with_bias),
        nn.Conv2d(3, 1, kernel_size=2, bias=with_bias),
    )
    x = torch.randn(BATCH_SIZE, 3, 4, 4)
    y = model(x)

    _test_expand_conv2d(model, x, y)


@pytest.mark.parametrize("with_bias", (False, True))
def test_expand_conv2d_torch_prune(with_bias):
    """Verify expanding Conv2d while using torch prune."""
    model = nn.Sequential(
        nn.Conv2d(3, 3, kernel_size=2, bias=with_bias),
        nn.Conv2d(3, 1, kernel_size=2, bias=with_bias),
    )
    prune.identity(model[0], "weight")
    prune.identity(model[1], "weight")
    if with_bias:
        prune.identity(model[0], "bias")
        prune.identity(model[1], "bias")
    model[0].weight_mask[:1] = 0
    model[1].weight_mask[-1:] = 0

    x = torch.randn(BATCH_SIZE, 3, 4, 4)
    y = model(x)

    _test_expand_conv2d(model, x, y)


@pytest.mark.parametrize("with_bias", (True, False))
def test_expand_conv2d_eideticnet(with_bias):
    """Verify expanding Linear EideticNet."""

    class EideticConv2d(EideticNetwork):
        def __init__(self, with_bias):
            super().__init__()
            self.blocks = nn.ModuleList(
                [
                    nn.Conv2d(3, 3, kernel_size=2, bias=with_bias),
                    nn.Conv2d(3, 1, kernel_size=2, bias=with_bias),
                ]
            )
            self.classifiers = [nn.Identity()]

        def __getitem__(self, index):
            return self.blocks[index]

        def forward(self, x):
            for block in self.blocks:
                x = block(x)
            return [c(x) for c in self.classifiers]

    model = EideticConv2d(with_bias)
    model.set_phase(0)
    model[0].weight_mask[:1] = 0
    model[1].weight_mask[-1:] = 0
    x = torch.randn(BATCH_SIZE, 3, 4, 4)
    y = model(x)[0]
    _test_expand_conv2d(model, x, y)


def _test_expand_bn(model, x, y):
    """Verify expanding BatchNorm under different settings."""
    network_expansion = [
        # Add 0 input and 3 output features to the first layer.
        (0, 3),
        # Add 3 output features to the second (bn) layer.
        (3,),
        # Add 3 input and 2 output features to the third layer.
        (3, 2),
        # Add 2 output features to the second (bn) layer.
        (2,),
        # Add 2 input and 0 output features to the last layer.
        (2, 0),
    ]

    for i, layer_expansion in enumerate(network_expansion):
        if isinstance(model[i], nn.Linear):
            out_features = model[i].out_features
            expand.expand_linear(model[i], *layer_expansion)
            assert model[i].out_features == out_features + layer_expansion[1]
        elif isinstance(model[i], nn.BatchNorm1d):
            num_features = model[i].num_features
            expand.expand_bn(model[i], *layer_expansion)
            assert model[i].num_features == num_features + layer_expansion[0]

    y_expanded = model(x)
    if isinstance(model, EideticNetwork):
        assert isinstance(y_expanded, list)
        y_expanded = y_expanded[0]

    assert torch.all(y_expanded == y)


@pytest.mark.parametrize(
    "affine, track_running_stats",
    [(True, True), (False, True), (True, False), (False, False)],
)
def test_expand_bn(affine, track_running_stats):
    model = nn.Sequential(
        nn.Linear(10, 16),
        nn.BatchNorm1d(
            16, affine=affine, track_running_stats=track_running_stats
        ),
        nn.Linear(16, 4),
        nn.BatchNorm1d(
            4, affine=affine, track_running_stats=track_running_stats
        ),
        nn.Linear(4, 2),
    )
    x = torch.randn(BATCH_SIZE, 10)
    y = model(x)
    _test_expand_bn(model, x, y)


@pytest.mark.parametrize(
    "affine, track_running_stats",
    [(True, True), (False, True), (True, False), (False, False)],
)
def test_expand_bn_torch_prune(affine, track_running_stats):
    """Verify expanding BatchNorm while using torch prune."""
    model = nn.Sequential(
        nn.Linear(10, 16),
        nn.BatchNorm1d(
            16, affine=affine, track_running_stats=track_running_stats
        ),
        nn.Linear(16, 4),
        nn.BatchNorm1d(
            4, affine=affine, track_running_stats=track_running_stats
        ),
        nn.Linear(4, 2),
    )
    prune.identity(model[0], "weight")
    prune.identity(model[0], "bias")
    if affine:
        prune.identity(model[1], "weight")
        prune.identity(model[1], "bias")
    prune.identity(model[2], "weight")
    prune.identity(model[2], "bias")
    if affine:
        prune.identity(model[3], "weight")
        prune.identity(model[3], "bias")
    x = torch.randn(BATCH_SIZE, 10)
    y = model(x)
    _test_expand_bn(model, x, y)


@pytest.mark.parametrize(
    "affine, track_running_stats",
    [(True, True), (False, True), (True, False), (False, False)],
)
def test_expand_bn_eideticnet(affine, track_running_stats):
    class EideticBatchNorm(EideticNetwork):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList(
                [
                    nn.Linear(10, 16),
                    nn.BatchNorm1d(
                        16,
                        affine=affine,
                        track_running_stats=track_running_stats,
                    ),
                    nn.Linear(16, 4),
                    nn.BatchNorm1d(
                        4,
                        affine=affine,
                        track_running_stats=track_running_stats,
                    ),
                    nn.Linear(4, 2),
                ]
            )
            self.classifiers = [nn.Identity()]

        def __getitem__(self, index):
            return self.blocks[index]

        def forward(self, x):
            for block in self.blocks:
                x = block(x)
            return [c(x) for c in self.classifiers]

    model = EideticBatchNorm()
    model.set_phase(0)
    model[0].weight_mask[:3] = 0
    if affine:
        model[1].weight_mask[5] = 0
    model[2].weight_mask[-2:] = 0
    if affine:
        model[3].weight_mask[1] = 0
    model[4].weight_mask[0] = 0
    x = torch.randn(BATCH_SIZE, 10)
    y = model(x)[0]
    _test_expand_bn(model, x, y)


def _test_expected_expansion(layer, expansion):
    if isinstance(layer, torch.nn.Linear):
        assert layer.in_features == expansion["in_features"]
        assert layer.out_features == expansion["out_features"]
        expected_weight_shape = (
            expansion["out_features"],
            expansion["in_features"],
        )
        expected_bias_shape = (expansion["out_features"],)
        assert layer.weight.shape == expected_weight_shape
        for param_name in expand.get_optional_params("weight").keys():
            assert getattr(layer, param_name).shape == expected_weight_shape
        if hasattr(layer, "bias") and layer.bias is not None:
            assert layer.bias.shape == expected_bias_shape
            bias_params = expand.get_optional_params("bias")
            for param_name in bias_params.keys():
                assert getattr(layer, param_name).shape == expected_bias_shape
    elif isinstance(layer, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
        assert layer.num_features == expansion["num_features"]
        expected_shape = (expansion["num_features"],)
        if layer.affine:
            assert layer.weight.shape == expected_shape
            weight_params = expand.get_optional_params("weight")
            for param_name in weight_params.keys():
                assert getattr(layer, param_name).shape == expected_shape

            assert layer.bias.shape == expected_shape
            bias_params = expand.get_optional_params("bias")
            for param_name in bias_params.keys():
                assert getattr(layer, param_name).shape == expected_shape
        if layer.track_running_stats:
            running_stats_params = (
                "running_mean",
                "_frozen_running_mean",
                "running_var",
                "_frozen_running_var",
            )
            for param_name in running_stats_params:
                if hasattr(layer, param_name):
                    assert getattr(layer, param_name).shape == expected_shape
    elif isinstance(layer, torch.nn.Conv2d):
        assert layer.in_channels == expansion["in_channels"]
        assert layer.out_channels == expansion["out_channels"]
        expected_weight_shape = (
            expansion["out_channels"],
            expansion["in_channels"],
            *layer.weight.shape[2:],
        )
        expected_bias_shape = (expansion["out_channels"],)
        assert layer.weight.shape == expected_weight_shape
        for param_name in expand.get_optional_params("weight").keys():
            assert getattr(layer, param_name).shape == expected_weight_shape
        if hasattr(layer, "bias") and layer.bias is not None:
            assert layer.bias.shape == expected_bias_shape
            bias_params = expand.get_optional_params("bias")
            for param_name in bias_params.keys():
                assert getattr(layer, param_name).shape == expected_bias_shape
    elif isinstance(layer, nn.Identity):
        return
    else:
        raise ValueError(f"Unexpected type of module for unit test: {layer}")


def _test_expected_expansions(expected_expansions):
    for layer, expansion in expected_expansions.items():
        try:
            _test_expected_expansion(layer, expansion)
        except Exception as e:
            raise AssertionError(
                f"When verifying expansion {expansion} on module {layer}, "
                f"caught exception {e}."
            )


def expand_mlp(model, get_add_out_func, trace_dict={}):
    """
    User-defined policy for expanding the capacity of the layers of our
    MLP. After training the network on a task, and applying iterative
    pruning, the policy adds to a layer exactly the number of neurons that
    were not pruned. The rationale of this policy is that when training a
    new task, there should always be the same amount of free, unallocated
    capacity as the network started with.
    """
    add_in_features = 0

    for i in range(len(model.blocks)):
        # Expand the Linear layer. Here we add as many neurons as remain
        # after pruning, because those neurons will be frozen and not
        # available for optimizing a downstream task, but this is a policy
        # choice. We should allow the user to design their own policy and
        # choose by how much to expand a layer.
        linear = model.blocks[i][0]
        add_out_features = get_add_out_func(module=linear)
        expand.expand_linear(
            linear, add_in_features, add_out_features, trace_dict=trace_dict
        )

        maybe_bn = model.blocks[i][1]
        if isinstance(maybe_bn, torch.nn.BatchNorm1d):
            expand.expand_bn(maybe_bn, add_out_features, trace_dict=trace_dict)

        add_in_features = add_out_features

    if add_in_features:
        for i in range(len(model.classifiers)):
            expand.expand_linear(
                model.classifiers[i], add_in_features, 0, trace_dict=trace_dict
            )


@pytest.mark.parametrize("with_bn", (False, True))
def test_expand_mlp_eideticnet(with_bn):
    # Network with three layers, two hidden.
    num_hidden = 2

    model = eideticnet.MLP(
        in_features=10,
        num_classes=[2],
        num_layers=num_hidden,
        width=100,
        bn=with_bn,
    )

    x = torch.randn(BATCH_SIZE, 10)

    model.set_phase(0)

    # Prune the first ten out of the 100 neurons, so there are 90 neurons
    # remaining, and those 90 will be frozen, so using the simple policy
    # here in `expand_mlp`, we should add 90 more neurons.
    model.blocks[0][0].weight_mask[:10] = 0
    # Prune the last five out of the 100 neurons, so we should add 95 more
    # neurons.
    model.blocks[1][0].weight_mask[-5:] = 0
    # Prune no neurons of the last hidden block.
    pass

    y = model(x)[0]

    expected_expansions = {
        model.blocks[0][0]: {"in_features": 10, "out_features": 190},
        model.blocks[0][1]: {"num_features": 190},
        model.blocks[1][0]: {"in_features": 190, "out_features": 195},
        model.blocks[1][1]: {"num_features": 195},
    }

    expand_mlp(model, get_num_neurons_assigned)

    _test_expected_expansions(expected_expansions)

    y_expanded = model(x)[0]
    assert y_expanded.shape == y.shape
    assert torch.allclose(y_expanded, y)


@pytest.mark.parametrize("with_bn", (True, False))
def test_expand_mlp_eideticnet_smoke_backward(with_bn):
    """Verify an expanded EideticNet can be optimized."""
    # Network with three layers, two hidden.
    num_hidden = 2

    model = eideticnet.MLP(
        in_features=10,
        num_classes=[2],
        num_layers=num_hidden,
        width=100,
        bn=with_bn,
    )

    x = torch.randn(BATCH_SIZE, 10)
    target = torch.randint(low=0, high=2, size=(BATCH_SIZE,))

    model.set_phase(0)

    # Prune the first ten out of the 100 neurons, so there are 90 neurons
    # remaining, and those 90 will be frozen, so using the simple policy
    # here in `expand_mlp`, we should add 90 more neurons.
    model.blocks[0][0].weight_mask[:10] = 0
    # Prune the last five out of the 100 neurons, so we should add 95 more
    # neurons.
    model.blocks[1][0].weight_mask[-5:] = 0
    # Prune no neurons of the last hidden block.
    pass

    optimizer = torch.optim.Adam(model.parameters(), lr=0)
    y = model(x)[0]
    loss = torch.nn.functional.cross_entropy(y, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    expected_expansions = {
        model.blocks[0][0]: {"in_features": 10, "out_features": 190},
        model.blocks[0][1]: {"num_features": 190},
        model.blocks[1][0]: {"in_features": 190, "out_features": 195},
        model.blocks[1][1]: {"num_features": 195},
    }

    expand_mlp(model, get_num_neurons_assigned)

    _test_expected_expansions(expected_expansions)

    optimizer = torch.optim.Adam(model.parameters(), lr=0)
    y_expanded = model(x)[0]
    assert y_expanded.shape == y.shape
    # Because the model is running in train mode, the batch norm stats get
    # updated. So use a more generous atol than the default.
    assert torch.allclose(y_expanded, y, atol=1e-6)
    loss = torch.nn.functional.cross_entropy(y_expanded, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


"""
FAILED test_expand.py::test_expand_and_train_mlp_does_not_forget_first_task[0-True-True] - assert False
FAILED test_expand.py::test_expand_and_train_mlp_does_not_forget_first_task[0-True-False] - assert False
FAILED test_expand.py::test_expand_and_train_mlp_does_not_forget_first_task[0-False-True] - assert False
FAILED test_expand.py::test_expand_and_train_mlp_does_not_forget_first_task[0-False-False] - assert False
FAILED test_expand.py::test_expand_and_train_mlp_does_not_forget_first_task[1-True-True] - assert False
FAILED test_expand.py::test_expand_and_train_mlp_does_not_forget_first_task[1-False-True] - assert False
FAILED test_expand.py::test_expand_and_train_mlp_does_not_forget_first_task[1-False-False] - assert False
FAILED test_expand.py::test_expand_and_train_mlp_does_not_forget_first_task[2-True-False] - assert False
"""


@pytest.mark.parametrize(
    "num_hidden, with_forward_transfer, with_bn",
    product(range(3), (True, False), (True, False)),
)
def test_expand_and_train_mlp_does_not_forget_first_task(
    num_hidden, with_forward_transfer, with_bn
):
    """
    Verify that MLP does not forget task 0 after training task 1.
    """

    def train(model, x, target, num_epochs):
        is_eval = not model.training
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        for i in range(num_epochs):
            optimizer.zero_grad()
            y = model(x)[model.phase]
            loss = torch.nn.functional.cross_entropy(y, target)
            loss.backward()
            optimizer.step()
        if is_eval:
            model.eval()

    in_features = 2
    num_tasks = 2
    num_classes = 2
    width = 1
    network = eideticnet.MLP(
        in_features=in_features,
        num_classes=[num_classes] * num_tasks,
        num_layers=num_hidden,
        width=width,
        bn=with_bn,
    )

    batch_size = 2
    inputs = [torch.randn(batch_size, in_features) for _ in range(num_tasks)]
    targets = [torch.randint(0, 2, (batch_size,)) for _ in range(num_tasks)]

    network.eval()

    # Train the first task.
    network.prepare_for_task(0, forward_transfer=with_forward_transfer)
    train(network, inputs[0], targets[0], 10)
    outputs_after_training = network(inputs[0])[0]

    def add_neuron_to_first_layer(*, module=None, weight_mask=None):
        if module == network.blocks[0][0]:
            return 1
        else:
            return 0

    unregister_eidetic_hooks(network.eidetic_handles)
    expand_mlp(network, add_neuron_to_first_layer)
    network.eidetic_handles = register_eidetic_hooks(network)

    print(network.phase)
    for module in network.modules():
        if hasattr(module, "weight_excess_capacity"):
            print(module.weight_excess_capacity)

    network.prepare_for_task(1, forward_transfer=with_forward_transfer)

    print(network.phase)
    for module in network.modules():
        if hasattr(module, "weight_excess_capacity"):
            print(module.weight_excess_capacity)

    train(network, inputs[1], targets[1], 10)

    # Run one batch to restore batch norm running mean and variance.
    network(inputs[0])

    # Verify that the output on the first task has not changed after training
    # the second task.
    assert torch.allclose(network(inputs[0])[0], outputs_after_training)

    def add_neuron_to_first_layer(*, module=None, weight_mask=None):
        if module == network.blocks[0][0]:
            return 1
        else:
            return 0

    unregister_eidetic_hooks(network.eidetic_handles)
    expand_mlp(network, add_neuron_to_first_layer)
    network.eidetic_handles = register_eidetic_hooks(network)

    network.prepare_for_task(1, forward_transfer=with_forward_transfer)

    train(network, inputs[1], targets[1], 10)

    # Run one batch to restore batch norm running mean and variance.
    network(inputs[0])

    # Verify that the model remembers the first task.
    assert torch.allclose(network(inputs[0])[0], outputs_after_training)


@pytest.mark.parametrize(
    "num_tasks, num_hidden, with_forward_transfer, with_bn",
    product([3, 4, 5], range(3), (True, False), (True, False)),
)
def test_expand_and_train_mlp_does_not_forget_multiple_tasks(
    num_tasks, num_hidden, with_forward_transfer, with_bn
):
    """
    Verify that MLP does not forget task t after training task t+1.
    """

    def train(model, x, target, num_epochs):
        is_eval = not model.training
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        for i in range(num_epochs):
            optimizer.zero_grad()
            y = model(x)[model.phase]
            loss = torch.nn.functional.cross_entropy(y, target)
            loss.backward()
            optimizer.step()
        if is_eval:
            model.eval()

    in_features = 2
    num_classes = 2
    width = 1
    network = eideticnet.MLP(
        in_features=in_features,
        num_classes=[num_classes] * num_tasks,
        num_layers=num_hidden,
        width=width,
        bn=with_bn,
    )

    batch_size = 2
    inputs = [torch.randn(batch_size, in_features) for _ in range(num_tasks)]
    targets = [torch.randint(0, 2, (batch_size,)) for _ in range(num_tasks)]

    def add_one_neuron(*, module=None, weight_mask=None):
        return 1

    network.eval()
    outputs_after_training = []

    # Train each task and expand the network.
    for task_id in range(num_tasks):
        network.prepare_for_task(
            task_id, forward_transfer=with_forward_transfer
        )
        train(network, inputs[task_id], targets[task_id], 10)
        outputs_after_training.append(network(inputs[task_id])[task_id])

        if task_id < num_tasks - 1:
            unregister_eidetic_hooks(network.eidetic_handles)
            expand_mlp(network, add_one_neuron)
            network.eidetic_handles = register_eidetic_hooks(network)

    # Run one batch to restore batch norm running mean and variance.
    network(inputs[0])

    # Verify that the output on each task has not changed.
    for task_id in range(num_tasks):
        assert torch.allclose(
            network(inputs[task_id])[task_id], outputs_after_training[task_id]
        )


def expand_convnet(model, get_add_out_func):
    """
    User-defined policy for expanding the capacity of the layers of our
    MLP. After training the network on a task, and applying iterative
    pruning, the policy adds to a layer exactly the number of neurons that
    were not pruned. The rationale of this policy is that when training a
    new task, there should always be the same amount of free, unallocated
    capacity as the network started with.
    """
    add_in_features = 0
    for i in range(len(model.blocks)):
        conv2d = model.blocks[i][0]
        add_out_features = get_add_out_func(module=conv2d)
        expand.expand_conv2d(conv2d, add_in_features, add_out_features)

        maybe_bn = model.blocks[i][1]
        if isinstance(maybe_bn, torch.nn.BatchNorm2d):
            expand.expand_bn(maybe_bn, add_out_features)

        add_in_features = add_out_features
    if add_in_features:
        for i in range(len(model.classifiers)):
            expand.expand_linear(model.classifiers[i], add_in_features, 0)


@pytest.mark.parametrize("with_bn", (False, True))
def test_expand_convnet_eideticnet(with_bn):
    num_hidden = 2
    num_channels = 100

    model = eideticnet.ConvNet(
        in_channels=3,
        num_classes=[2],
        num_layers=num_hidden,
        width=num_channels,
        bn=with_bn,
    )

    x = torch.randn(BATCH_SIZE, 3, 32, 32)

    model.set_phase(0)

    num_blocks = 3 * num_hidden

    # Prune a fixed number of output channels per layer.
    num_pruned = 10

    in_channels = 3
    expected_expansions = {}
    for i in range(num_blocks):
        conv2d = model.blocks[i][0]
        # When pruning the fixed number of output channels, prune them from
        # different parts of the weight tensor.
        if i % 2 == 0:
            conv2d.weight_mask[:num_pruned] = 0
        else:
            conv2d.weight_mask[-num_pruned:] = 0
        out_channels = conv2d.out_channels + (conv2d.out_channels - num_pruned)
        expected_expansions[conv2d] = {
            "in_channels": in_channels,
            "out_channels": out_channels,
        }
        if with_bn:
            bn = model.blocks[i][1]
            expected_expansions[bn] = {"num_features": out_channels}
        in_channels = out_channels

    y = model(x)[0]

    expand_convnet(model, get_num_neurons_assigned)

    _test_expected_expansions(expected_expansions)

    y_expanded = model(x)[0]
    assert y_expanded.shape == y.shape
    assert torch.allclose(y_expanded, y)


def setup_resnet_test(model, num_pruned, with_bn, in_channels=3):
    expected_expansions = {}
    for i in range(len(model.blocks)):
        residual_in_channels = in_channels
        residual_conv = model.res[i]
        block = model.blocks[i]
        for i, layer in enumerate(block):
            if isinstance(layer, nn.Conv2d):
                # When pruning the fixed number of output channels, prune them
                # from different parts of the weight tensor.
                layer.weight_mask[:num_pruned] = 0
                out_channels = layer.out_channels + (
                    layer.out_channels - num_pruned
                )
                expected_expansions[layer] = {
                    "in_channels": in_channels,
                    "out_channels": out_channels,
                }
                if with_bn:
                    bn = block[i + 1]
                    expected_expansions[bn] = {"num_features": out_channels}
                in_channels = out_channels
        residual_conv.weight_mask[:num_pruned] = 0
        expected_expansions[residual_conv] = {
            "in_channels": residual_in_channels,
            "out_channels": (
                residual_conv.out_channels
                + (residual_conv.out_channels - num_pruned)
            ),
        }
    return expected_expansions


def expand_resnet(model, get_add_out_func):
    """
    User-defined procedure for expanding the layers of our ResNet. After
    training the network on a task, and applying iterative pruning, the
    procedure traverses the modules in the network and inspects each one with
    `get_add_out_func` to determine by how much to expand it.

    """

    def is_conv_or_bn(m):
        return isinstance(m, (nn.Conv2d, nn.BatchNorm2d))

    add_out_channels = 0
    add_in_channels = 0

    for i in range(len(model.blocks)):
        res_add_in_channels = add_in_channels
        block = model.blocks[i]

        # Expand each conv and its subsequent batch norm layer.
        convs_and_bns = list(filter(is_conv_or_bn, block.children()))
        for j, layer in enumerate(convs_and_bns):
            if isinstance(layer, nn.Conv2d):
                add_out_channels = get_add_out_func(module=layer)
                expand.expand_conv2d(layer, add_in_channels, add_out_channels)
                if j + 1 < len(convs_and_bns):
                    bn = convs_and_bns[j + 1]
                    if isinstance(bn, torch.nn.BatchNorm2d):
                        expand.expand_bn(bn, add_out_channels)
                add_in_channels = add_out_channels

        # Expand the residual conv by the number of new input channels of the
        # entire block's ingress layer (res_add_in_channels) and the number of
        # new output channels of the block's egress layer (add_out_channels, at
        # this point).
        residual_conv = model.res[i]
        res_add_out_channels = add_out_channels
        expand.expand_conv2d(
            residual_conv, res_add_in_channels, res_add_out_channels
        )

    if add_in_channels:
        for i in range(len(model.classifiers)):
            if isinstance(model.classifiers[i], nn.Linear):
                expand.expand_linear(model.classifiers[i], add_in_channels, 0)


@pytest.mark.parametrize("with_bn", (False, True))
def test_expand_baby_resnet_eideticnet(with_bn):
    class BabyResNet(EideticNetwork):
        def __init__(self):
            super().__init__()
            self.res = nn.ModuleList(
                [nn.Conv2d(3, 64, kernel_size=1, bias=False, stride=1)]
            )
            self.blocks = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size=1), nn.BatchNorm2d(64)
                    )
                ]
            )
            self.formatter = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten()
            )
            self.classifiers = nn.ModuleList([nn.Linear(64, 2)])

        def forward(self, x):
            for b, r in zip(self.blocks, self.res):
                x = b(x) + r(x)
            flat = self.formatter(x)
            return [c(flat) for c in self.classifiers]

    model = BabyResNet()
    model.set_phase(0)

    x = torch.randn(BATCH_SIZE, 3, 32, 32)

    num_pruned = 10
    expected_expansions = setup_resnet_test(model, num_pruned, with_bn)

    y = model(x)[0]

    expand_resnet(model, get_num_neurons_assigned)

    _test_expected_expansions(expected_expansions)

    y_expanded = model(x)[0]
    assert y_expanded.shape == y.shape
    assert torch.allclose(y_expanded, y)


@pytest.mark.parametrize("with_bn", (True, False))
def test_expand_resnet_eideticnet(with_bn):
    model = eideticnet.ResNet(
        in_channels=3,
        num_classes=[2],
        n_blocks=[2, 2, 2, 2],
        expansion=1,
        bn=with_bn,
        low_res=True,
    )

    x = torch.randn(BATCH_SIZE, 3, 32, 32)

    model.set_phase(0)

    # Prune a fixed number of output channels per layer.
    num_pruned = 10
    expected_expansions = setup_resnet_test(model, num_pruned, with_bn)

    y = model(x)[0]

    expand_resnet(model, get_num_neurons_assigned)

    _test_expected_expansions(expected_expansions)

    y_expanded = model(x)[0]
    assert y_expanded.shape == y.shape
    assert torch.allclose(y_expanded, y)
