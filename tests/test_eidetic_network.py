import os

import pytest

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy

import wandb

import eideticnet_training as eideticnet
from eideticnet_training.utils.data import create_sequential_classification
from eideticnet_training import EideticNetwork
from eideticnet_training.prune import bridge_prune


os.environ["WANDB_MODE"] = "disabled"
wandb.init()


class EideticTestNet(EideticNetwork):
    """
    A simple example of a subclass of EideticNetwork.
    """

    def __init__(self, num_tasks):
        super().__init__()

        self.linear1 = nn.Linear(3 * 32 * 32, 12)
        self.linear2 = nn.Linear(12, 13)
        self.classifiers = nn.ModuleList(
            [nn.Linear(13, 10) for _ in range(num_tasks)]
        )
        self.set_input_layer(self.linear1)
        for c in self.classifiers:
            self.set_output_layer(c)

    def forward(self, x):
        x = x.flatten(1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        return [c(x) for c in self.classifiers]

    def _bridge_prune(self, pct, pruning_type=2, score_threshold=None):
        """ """
        bridge_prune(
            current_layer=self.linear1,
            current_bn=None,
            following_layer=self.linear2,
            percentage=pct,
            pruning_type=pruning_type,
            score_threshold=score_threshold,
        )
        bridge_prune(
            current_layer=self.linear2,
            current_bn=None,
            # Having the client refer to self.phase here is another example of
            # leakage from the superclass to the subclass.
            following_layer=self.classifiers[self.phase],
            percentage=pct,
            pruning_type=pruning_type,
            score_threshold=score_threshold,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("pruning_type", (1, 2, "l1", "l2", "taylor"))
def test_test_net_mnist_5(pruning_type, lr=0.1):

    num_tasks = 2
    model = EideticTestNet(num_tasks=num_tasks)
    model.to("cuda")

    dataset_path = os.path.expanduser("~/datasets")
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    train_dataset, test_dataset = eideticnet.data.torchvision_dataset(
        "MNIST", path=dataset_path
    )

    train_tasks = create_sequential_classification(train_dataset, 5)
    test_tasks = create_sequential_classification(
        test_dataset,
        5,
    )
    train_tasks = train_tasks[:num_tasks]
    test_tasks = test_tasks[:num_tasks]

    metrics = [MulticlassAccuracy(num_classes=10).to("cuda")] * num_tasks
    class_metrics = [
        MulticlassAccuracy(num_classes=10, average="none").to("cuda")
    ] * num_tasks

    for t, train_task in enumerate(train_tasks):
        train_loader = torch.utils.data.DataLoader(
            train_task,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
            num_workers=1,
            batch_size=256,
        )

        #######################################################################
        # Note that calling prepare_for_task(t) is a convenience function that
        # calls the following (if t > 0):
        #
        #     activate_excess_capacity
        #     random_initialize_excess_capacity.
        #
        # To enable more flexible use cases, we should still allow the client
        # to call set_phase(t) if they want to.
        #
        # To enable more flexible use cases, we should allow the sharing
        # argument of activate_excess_capacity to be specified on a per layer
        # and/or per task basis.
        #
        # TODO: Rename random_initialize_excess_capacity to
        # recycle_excess_capacity.
        #######################################################################
        model.prepare_for_task(t)

        optimizer = optim.AdamW(model.parameters(), lr)

        model.train_task(
            train_loader,
            metric=MulticlassAccuracy(num_classes=10).to("cuda"),
            optimizer=optimizer,
            pruning_step=0.1,
            tolerance=0.0,
            early_stopping_patience=1,
            pruning_type=pruning_type,
            validation_tasks=test_tasks,
            validation_metrics=metrics,
            validation_class_metrics=class_metrics,
            test_batch_size=256,
            last_task=t == (len(train_tasks) - 1),
        )


@pytest.mark.parametrize(
    "with_forward_transfer, with_bn",
    [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ],
)
def test_simple_eideticnet_does_not_forget(with_forward_transfer, with_bn):
    """
    Verify that outputs are unchanged on task 0 after training on task 1.
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

    @torch.no_grad
    def prepare_for_pruning(baseline):
        # Prepare to prune the second neuron of the first layer by ensuring
        # that the L2 norm is smaller than the first neuron.
        baseline.blocks[0][0].weight_orig[1] = (
            baseline.blocks[0][0].weight_orig[0] / 2
        )

    baseline = eideticnet.MLP(
        in_features=2,
        num_classes=[2, 2],
        num_layers=0,
        width=2,
        bn=with_bn,
    )

    x0 = torch.randn(2, 2)
    x1 = torch.randn(2, 2)
    target0 = torch.randint(0, 2, (2,))
    target1 = torch.randint(0, 2, (2,))

    baseline.eval()

    baseline.prepare_for_task(0, forward_transfer=with_forward_transfer)
    train(baseline, x0, target0, 50)
    prepare_for_pruning(baseline)
    baseline._bridge_prune(pct=0.5, pruning_type=2)
    output_after_train_and_prune = baseline(x0)[0]

    baseline.prepare_for_task(1, forward_transfer=with_forward_transfer)
    assert torch.all(baseline(x0)[0] == output_after_train_and_prune)
    train(baseline, x1, target1, 50)
    assert torch.all(baseline(x0)[0] == output_after_train_and_prune)
