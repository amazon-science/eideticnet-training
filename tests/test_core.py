import os

import pytest
import torch
import wandb

import eideticnet_training as eideticnet
from eideticnet_training.prune.masks import UNASSIGNED_PARAMS

os.environ["WANDB_MODE"] = "disabled"
wandb.init()


def get_tasks(module):
    tasks = set(
        [task.item() for task in module.weight_excess_capacity.unique()]
    )
    return tasks


@pytest.mark.parametrize("with_forward_transfer", [True, False])
def test_prepare_for_task(with_forward_transfer):
    """
    Verify that prepare_for_task updates masks correctly.

    To understand how parameters are assigned to a task in the current
    implementation of EideticNetwork, imagine that prepare_for_task(0) has been
    called, and the network has been trained on the data for task 0, and
    pruned. When prepare_for_task(1) is called, parameters are assigned as
    follows:

        * The parameters (both neurons and input dimensions) of each feature
          extraction layer that were not pruned during task 0 are assigned to
          task 0.
        * The parameters (input dimensions only) of the task-0 classifier head
          that were not pruned during task 0 are assigned to task 0.
        * The parameters of all other classifier heads are not assigned.

    Now imagine that the network s trained on the data for task 1, and pruned.
    When prepare_for_task(2) is called, the parameters are assigned as follows:

        * The parameters (both neurons and input dimensions) of each feature
          extraction layer that were not pruned are assigned to task 1. The
          parameters that were assigned to task 0 remain assigned to task 0.
        * The parameters (input dimensions only) of the task-1 classifier that
          were not pruned during task 1 are assigned to task 1.
        * The parameters of all other classifier heads, including the task-0
          classifier head, are not assigned.
    """

    class TestNet(eideticnet.EideticNetwork):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4, bias=False)
            self.classifiers = torch.nn.ModuleList(
                [torch.nn.Linear(4, 1, bias=False) for _ in range(3)]
            )
            for c in self.classifiers:
                c.last_layer = True

        def forward(self, x):
            if self.phase is None:
                raise RuntimeError("Need to call `prepare_for_task` first.")
            x = x.flatten(1)
            x = self.linear(x)
            return [c(x) for c in self.classifiers]

    net = TestNet()

    net.prepare_for_task(0)
    assert net.classifiers[0].weight_orig.requires_grad
    assert not net.classifiers[1].weight_orig.requires_grad
    assert not net.classifiers[2].weight_orig.requires_grad

    # Initially, the linear classifier layers should have no assigned
    # parameters.
    assert torch.all(net.linear.weight_excess_capacity == UNASSIGNED_PARAMS)
    assert torch.all(
        net.classifiers[0].weight_excess_capacity == UNASSIGNED_PARAMS
    )
    assert torch.all(
        net.classifiers[1].weight_excess_capacity == UNASSIGNED_PARAMS
    )
    assert torch.all(
        net.classifiers[2].weight_excess_capacity == UNASSIGNED_PARAMS
    )

    # Prune three neurons of the 4-neuron linear layer, and one dimension of the
    # current classifier head. The unpruned neuron -- and its corresponding
    # input dimension in the classifier head -- should be assigned to task 0.
    net.linear.weight_mask[:3] = 0
    net.classifiers[0].weight_mask[:, :3] = 0

    # Preparing for the next task entails assigning unpruned capacity to the
    # previous task, so we expect all parameters to be either unassigned or
    # assigned to task 0.
    net.prepare_for_task(1)
    assert not net.classifiers[0].weight_orig.requires_grad
    assert net.classifiers[1].weight_orig.requires_grad
    assert not net.classifiers[2].weight_orig.requires_grad

    expected_linear_weight_excess_capacity = torch.tensor(
        (
            (-1, -1, -1, -1),
            (-1, -1, -1, -1),
            (-1, -1, -1, -1),
            (0, 0, 0, 0),
        )
    )
    assert torch.all(
        net.linear.weight_excess_capacity
        == expected_linear_weight_excess_capacity
    )
    expected_classifier_weight_excess_capacity = torch.tensor(
        ((-1, -1, -1, 0),)
    )
    assert torch.all(
        net.classifiers[0].weight_excess_capacity
        == expected_classifier_weight_excess_capacity
    )

    assert get_tasks(net.classifiers[1]) == set([UNASSIGNED_PARAMS])
    assert get_tasks(net.classifiers[2]) == set([UNASSIGNED_PARAMS])

    # Prune two neurons of the 4-neuron linear layer, and two dimensions of the
    # current classifier head. One unpruned neuron -- and its corresponding
    # input dimension in the classifier head -- should be assigned to task 1.
    # The other unpruned neuron is already assigned to task 0.
    net.linear.weight_mask[:2] = 0
    net.classifiers[1].weight_mask[:, :2] = 0

    net.prepare_for_task(2)
    assert not net.classifiers[0].weight_orig.requires_grad
    assert not net.classifiers[1].weight_orig.requires_grad
    assert net.classifiers[2].weight_orig.requires_grad

    expected_linear_weight_excess_capacity = torch.tensor(
        (
            (-1, -1, -1, -1),
            (-1, -1, -1, -1),
            (1, 1, 1, 1),
            (0, 0, 0, 0),
        )
    )
    assert torch.all(
        net.linear.weight_excess_capacity
        == expected_linear_weight_excess_capacity
    )
    expected_classifier_weight_excess_capacity = torch.tensor(
        ((-1, -1, 1, 1),)
    )
    assert torch.all(
        net.classifiers[1].weight_excess_capacity
        == expected_classifier_weight_excess_capacity
    )

    assert get_tasks(net.classifiers[2]) == set([UNASSIGNED_PARAMS])

    # Prune one neuron of the 4-neuron linear layer, and one dimension of the
    # current classifier head. One unpruned neuron -- and its corresponding
    # input dimension in the classifier head -- should be assigned to task 1.
    # The other unpruned neurons are already assigned to tasks 0 and 1.
    net.linear.weight_mask[:1] = 0
    net.classifiers[2].weight_mask[:, :1] = 0

    net.prepare_for_task(3)

    expected_linear_weight_excess_capacity = torch.tensor(
        (
            (-1, -1, -1, -1),
            (2, 2, 2, 2),
            (1, 1, 1, 1),
            (0, 0, 0, 0),
        )
    )
    assert torch.all(
        net.linear.weight_excess_capacity
        == expected_linear_weight_excess_capacity
    )
    expected_classifier_weight_excess_capacity = torch.tensor(((-1, 2, 2, 2),))
    assert torch.all(
        net.classifiers[2].weight_excess_capacity
        == expected_classifier_weight_excess_capacity
    )
