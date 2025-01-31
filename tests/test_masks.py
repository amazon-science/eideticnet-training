import pytest
import torch

from eideticnet_training.prune.masks import (
    assign_1d_params_to_task,
    assign_Xd_params_to_task_forward_transfer,
    assign_Xd_params_to_task_last_layer,
    assign_Xd_params_to_task_no_forward_transfer,
)


@pytest.mark.parametrize(
    "weight_excess_capacity, masked, phase, expected",
    [
        (
            torch.tensor([[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]),
            # Only the input dimensions of a last layer are pruned, so the
            # pruning mask's sparsity here differs from that we see in
            # non-last-layer scenarios. Here, the first two output neurons of
            # the previous layer have been pruned, and the third output neuron
            # has to be assigned to task 0.
            torch.tensor(
                [[True, True, False], [True, True, False], [True, True, False]]
            ),
            0,
            torch.tensor([[-1, -1, 0], [-1, -1, 0], [-1, -1, 0]]),
        ),
        (
            # Starting with the mask from the previous test, prune only the
            # first output neuron of the previous layer.
            torch.tensor([[-1, -1, 0], [-1, -1, 0], [-1, -1, 0]]),
            torch.tensor(
                [
                    [True, False, False],
                    [True, False, False],
                    [True, False, False],
                ]
            ),
            1,
            torch.tensor([[-1, 1, 0], [-1, 1, 0], [-1, 1, 0]]),
        ),
        (
            # The second and third neurons of the previous layer have been
            # assigned to tasks 1 and 0, respectively. Now no input dimensions
            # of this layer have been pruned, so the task assignment mask
            # should not have any unassigned parameters.
            # pruned
            torch.tensor([[-1, 1, 0], [-1, 1, 0], [-1, 1, 0]]),
            torch.tensor(
                [
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                ]
            ),
            2,
            torch.tensor([[2, 1, 0], [2, 1, 0], [2, 1, 0]]),
        ),
    ],
)
def test_assign_Xd_params_to_task_last_layer(
    weight_excess_capacity, masked, phase, expected
):
    """
    Verify that parameters are assigned correctly to each task and are
    preserved across tasks.
    """
    assign_Xd_params_to_task_last_layer(weight_excess_capacity, masked, phase)
    assert torch.all(weight_excess_capacity == expected)


@pytest.mark.parametrize(
    "weight_excess_capacity, masked, phase, expected",
    [
        (
            torch.tensor([[-1, -1], [-1, -1], [-1, -1]]),
            torch.tensor([[False, False], [True, True], [True, True]]),
            0,
            torch.tensor([[0, 0], [-1, -1], [-1, -1]]),
        ),
        (
            torch.tensor([[0, 0], [-1, -1], [-1, -1]]),
            torch.tensor([[False, False], [False, False], [True, True]]),
            1,
            torch.tensor([[0, 0], [1, 1], [-1, -1]]),
        ),
        (
            torch.tensor([[0, 0], [1, 1], [-1, -1]]),
            torch.tensor([[False, False], [False, False], [False, False]]),
            2,
            torch.tensor([[0, 0], [1, 1], [2, 2]]),
        ),
    ],
)
def test_assign_Xd_params_to_task_forward_transfer(
    weight_excess_capacity, masked, phase, expected
):
    """
    Verify that parameters are assigned correctly to each task and are
    preserved across tasks.
    """
    assign_Xd_params_to_task_forward_transfer(
        weight_excess_capacity, masked, phase
    )
    assert torch.all(weight_excess_capacity == expected)


@pytest.mark.parametrize(
    "weight_excess_capacity, masked, phase, expected",
    [
        (
            # Initial capacity mask.
            torch.tensor(
                [
                    [-1, -1, -1, -1],
                    [-1, -1, -1, -1],
                    [-1, -1, -1, -1],
                    [-1, -1, -1, -1],
                ]
            ),
            # In this first pass, the first and third neurons of this layer
            # were pruned. In the previous layer, the second and fourth neurons
            # were pruned.
            torch.tensor(
                [
                    [True, True, True, True],
                    [False, True, False, True],
                    [True, True, True, True],
                    [False, True, False, True],
                ]
            ),
            0,  # Assigning to task 0.
            # Since forward transfer is disabled, the neurons in this layer
            # that were pruned (1 and 3) will only be able to receive input
            # from the second and fourth neurons of the previous layer. All
            # other parameters are marked as assigned to task 0.
            torch.tensor(
                [[0, -1, 0, -1], [0, 0, 0, 0], [0, -1, 0, -1], [0, 0, 0, 0]]
            ),
        ),
        (
            # Capacity mask from previous test.
            torch.tensor(
                [[0, -1, 0, -1], [0, 0, 0, 0], [0, -1, 0, -1], [0, 0, 0, 0]]
            ),
            # In this second pass, the first neuron neuron of this layer was
            # pruned. In the previous layer, the second neuron was pruned.
            torch.tensor(
                [
                    [True, True, True, True],
                    [False, True, False, False],
                    [False, True, False, False],
                    [False, True, False, False],
                ]
            ),
            1,  # Assigning to task 1.
            # Since forward transfer is disabled, the neurons in this layer
            # that were pruned (1) will only be able to receive input from the
            # neuron in the previous layer that were pruned. All other
            # parameters will be marked as assigned to task 0 or task 1.
            torch.tensor(
                [[0, -1, 0, 1], [0, 0, 0, 0], [0, 1, 0, 1], [0, 0, 0, 0]]
            ),
        ),
    ],
)
def test_assign_Xd_params_to_task_no_forward_transfer(
    weight_excess_capacity, masked, phase, expected
):
    """
    Verify that parameters are assigned correctly to each task and are
    preserved across tasks when forward transfer is disabled. In this setting,
    any neurons that were not pruned in the previous layer (as indicated by an
    input dimension being masked as True across all neurons in this layer)
    must be prevented from contributing to any neurons pruned in this layer in
    any subsequent task. To accomplish this, the input dimension of the current
    weight is marked as being assigned to the current task and is frozen by the
    gradient accumulation hooks.

    To test this with multiple passes, we use a larger implied weight matrix.
    Whereas the test with forward transfer uses a 3x2 (3 neurons, 2 input
    dimension) weight matrix, this test uses a 4x4 matrix.
    """
    assign_Xd_params_to_task_no_forward_transfer(
        weight_excess_capacity, masked, phase
    )
    assert torch.all(weight_excess_capacity == expected)


@pytest.mark.parametrize(
    "weight_excess_capacity, masked, phase, expected",
    [
        # End of training first task (phase=0).
        (
            torch.tensor([-1, -1, -1]),
            torch.tensor([False, True, True]),
            0,
            # The first (unpruned) element is assigned to phase 0.
            torch.tensor([0, -1, -1]),
        ),
        # End of training second task (phase=1).
        (
            torch.tensor([0, -1, -1]),
            torch.tensor([False, False, True]),
            1,
            # The second (unpruned) element is assigned to phase 1.
            torch.tensor([0, 1, -1]),
        ),
        (
            torch.tensor([0, 1, -1]),
            torch.tensor([False, False, False]),
            2,
            # The third (unpruned) element is assigned to phase 2.
            torch.tensor([0, 1, 2]),
        ),
    ],
)
def test_assign_1d_params_to_task(
    weight_excess_capacity, masked, phase, expected
):
    """
    Verify that parameters are assigned correctly to each task and are
    preserved across tasks.
    """
    assign_1d_params_to_task(weight_excess_capacity, masked, phase)
    assert torch.all(weight_excess_capacity == expected)
