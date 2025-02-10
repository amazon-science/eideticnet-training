import argparse
import math
import os
import random
from pathlib import Path

import neural_collapse_metrics as nc
import numpy as np
import torch
import torch.nn.utils.prune
import torch.optim as optim
import torchvision as tvis
import wandb
from adversarial_robustness import test_adversarial_robustness
from torchmetrics.classification import MulticlassAccuracy

import eideticnet_training as eideticnet
import eideticnet_training.prune.bridge as bridge
from eideticnet_training.prune.hooks import unregister_eidetic_hooks
from eideticnet_training.utils.data import RGBTransform

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_optimizer(name, parameters, **kwargs):
    if name != "SGD":
        if "momentum" in kwargs:
            del kwargs["momentum"]
    return getattr(optim, name)(parameters, **kwargs)


def make_random_subsets(dataset, num_subsets):
    task_ids = torch.randint(0, num_subsets, size=(len(dataset),))
    subsets = []
    for task in range(num_subsets):
        indices = torch.where(task_ids == task)[0]
        subset = torch.utils.data.Subset(dataset, indices)
        subsets.append(subset)
    return subsets


def get_transforms():
    train_transform = tvis.transforms.Compose(
        [
            RGBTransform(),
            tvis.transforms.RandomResizedCrop(32, scale=(0.4, 1)),
            tvis.transforms.ToTensor(),
        ]
    )
    test_transform = tvis.transforms.Compose(
        [
            RGBTransform(),
            tvis.transforms.Resize(32),
            tvis.transforms.ToTensor(),
        ]
    )
    return train_transform, test_transform


def load_tasks(args, sample_train_randomly=True):
    path = Path("~/datasets/MNIST").expanduser()
    train_transform, test_transform = get_transforms()
    train_dataset = tvis.datasets.MNIST(
        path, train=True, download=True, transform=train_transform
    )
    test_dataset = tvis.datasets.MNIST(
        path, train=False, download=True, transform=test_transform
    )
    if sample_train_randomly:
        train_datasets = make_random_subsets(
            train_dataset, args.num_training_subsets
        )
        train_tasks = train_datasets
        test_tasks = [test_dataset] * args.num_training_subsets
    else:
        train_tasks = [train_dataset]
        test_tasks = [test_dataset]
    return train_tasks, test_tasks


def make_metrics(args, num_tasks):
    metrics = [
        MulticlassAccuracy(num_classes=args.num_classes).to(DEVICE)
    ] * num_tasks
    class_metrics = [
        MulticlassAccuracy(num_classes=args.num_classes, average="none").to(
            DEVICE
        )
    ] * num_tasks
    return metrics, class_metrics


def hold_out_training_as_test(
    *, args, train_tasks, test_tasks, metrics, num_classes
):
    # Evaluate hyperparameters on 10% of training set on the first task.
    assert len(test_tasks) == len(train_tasks)
    print(
        f"hold_out_training_as_test: train {len(train_tasks)} test "
        f"{len(test_tasks)} hold_out_num_tasks {args.hold_out_num_tasks}."
    )
    try:
        # Needed for holding out during training final MNIST classifier head.
        test_transforms = [tt.dataset.transform for tt in test_tasks]
    except Exception:
        # Needed for holding out during Eidetic training with MNIST.
        test_transforms = [tt.transform for tt in test_tasks]
    args.num_tasks = args.hold_out_num_tasks
    train_tasks = train_tasks[: args.hold_out_num_tasks]
    test_tasks = test_tasks[: args.hold_out_num_tasks]
    # Copy transforms from test set, because otherwise data augmentation is
    # done on the test set.
    for i in range(args.hold_out_num_tasks):
        test_tasks[i].transform = test_transforms[i]
    num_classes = num_classes[: args.hold_out_num_tasks]
    metrics = metrics[: args.hold_out_num_tasks]

    return train_tasks, test_tasks, metrics, num_classes


def _bridge_prune(self, pct, pruning_type=2, score_threshold=None):
    for i in range(len(self.blocks) - 1):
        bridge.bridge_prune(
            self.blocks[i][0],
            self.blocks[i][1],
            self.blocks[i + 1][0],
            pct,
            pruning_type=pruning_type,
            score_threshold=score_threshold,
        )
    for i in range(self.phase, len(self.classifiers)):
        bridge.bridge_prune(
            self.blocks[-1][0],
            self.blocks[-1][1],
            self.classifiers[i],
            pct,
            pruning_type=pruning_type,
            score_threshold=score_threshold,
        )


def make_model(args, num_classes):
    if args.arch == "MLP-small":
        # Hard-coded width for PMNIST for partity with CLNP
        # https://arxiv.org/abs/1903.04476
        width = 2000 if args.task == "PMNIST" else 512
        num_layers = 2 if args.task == "PMNIST" else 2
        # Monkey patch _bridge prune to ensure current *and* subsequent
        # classifier heads are pruned.
        eideticnet.networks.MLP._bridge_prune = _bridge_prune

        model = eideticnet.networks.MLP(
            args.in_channels * 32 * 32,
            num_classes,
            num_layers=num_layers,
            width=width,
            bn=False,
            dropout=False,
        ).to(DEVICE)
    elif args.arch == "MLP-big":
        model = eideticnet.networks.MLP(
            args.in_channels * 32 * 32,
            num_classes,
            bn=args.bn,
            num_layers=5,
            width=4096,
        ).to(DEVICE)
    elif args.arch == "ConvNet-small":
        model = eideticnet.networks.ConvNet(
            args.in_channels,
            num_classes,
            bn=args.bn,
            num_layers=2,
            width=8,
        ).to(DEVICE)
    elif args.arch == "ConvNet-big":
        model = eideticnet.networks.ConvNet(
            args.in_channels,
            num_classes,
            bn=args.bn,
            num_layers=4,
            width=64,
        ).to(DEVICE)
    elif args.arch == "ResNet18":
        print("num_classes", num_classes)
        model = eideticnet.networks.ResNet(
            args.in_channels,
            num_classes,
            [2, 2, 2, 2],
            expansion=1,
            low_res="imagenet" not in args.task.lower(),
        ).to(DEVICE)
    elif args.arch == "ResNet34":
        model = eideticnet.networks.ResNet(
            args.in_channels,
            num_classes,
            [3, 4, 6, 3],
            expansion=1,
            low_res="imagenet" not in args.task.lower(),
        ).to(DEVICE)
    elif args.arch == "ResNet50":
        model = eideticnet.networks.ResNet(
            args.in_channels,
            num_classes,
            [3, 4, 6, 3],
            expansion=4,
            low_res="imagenet" not in args.task.lower(),
        ).to(DEVICE)
    return model


def run(args):
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.random.manual_seed(args.random_seed)

    train_tasks, test_tasks = load_tasks(args, sample_train_randomly=True)

    metrics, class_metrics = make_metrics(args, args.num_training_subsets)

    num_classes = [args.num_classes] * args.num_training_subsets

    if args.hold_out:
        (
            train_tasks,
            test_tasks,
            metrics,
            num_classes,
        ) = hold_out_training_as_test(
            args=args,
            train_tasks=train_tasks,
            test_tasks=test_tasks,
            num_classes=num_classes,
            metrics=metrics,
        )
        print(f"Num hold-out tasks {len(train_tasks)}")

    model = make_model(args, num_classes)

    print(model)

    def cache_hidden_states_hook(self, input, output):
        nc.HiddenStates.value = input[0].clone()

    for classifier in model.classifiers:
        classifier.register_forward_hook(cache_hidden_states_hook)

    # FIXME deprecate 'tolerance'.
    tolerance = args.stop_pruning_threshold

    hyperparams = {
        "max_interpruning_epochs": args.max_recovery_epochs,
        "max_epochs": args.max_epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "pruning_step": args.pruning_step_size,
        "pruning_step_size_is_constant": args.pruning_step_size_is_constant,
        "reduce_learning_rate_before_pruning": args.reduce_learning_rate_before_pruning,  # noqa: E501
        "tolerance": tolerance,
        "pruning_type": args.pruning_type,
        "test_batch_size": args.test_batch_size,
        "test_frequently": args.test_frequently,
        "validation_tasks": test_tasks,
        "validation_metrics": metrics,
        "validation_class_metrics": class_metrics,
    }

    print(f"Number of classifier heads {len(model.classifiers)}")
    print(f"Number of validation tasks {len(test_tasks)}")

    for t, train_task in enumerate(train_tasks):
        train_loader = torch.utils.data.DataLoader(
            train_task,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
            num_workers=args.num_train_workers,
            batch_size=args.batch_size,
        )

        print(f"Training task {t} on {len(train_task)} samples.")

        model.prepare_for_task(t, forward_transfer=args.forward_transfer)

        parameters = model.parameters()
        optimizer = make_optimizer(
            args.optimizer,
            parameters,
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum if args.momentum else 0,
        )

        hyperparams["validation_task_nums"] = list(range(t + 1))

        model.train_task(
            train_loader,
            metrics[0],
            optimizer=optimizer,
            **hyperparams,
        )

        nc.log_neural_collapse_metrics(
            model=model,
            classifier=model.classifiers[t],
            train_loader=train_loader,
            validation_tasks=test_tasks,
            test_batch_size=args.test_batch_size,
        )

        print(f"== Task assignments after training task {t}. ==")
        for module_name, module in model.named_modules():
            if hasattr(module, "weight_excess_capacity"):
                print(module_name, module.weight_excess_capacity.unique())
            if hasattr(module, "bias_excess_capacity"):
                print(module_name, module.bias_excess_capacity.unique())
        print()

    # Assign capacity of final task.
    model.prepare_for_task(t + 1, forward_transfer=args.forward_transfer)

    def disable_pruning(model):
        """
        Remove pruning masks from all modules. The task assignment masks will
        remain and can be used during optimization.
        """
        for module_name, module in model.named_modules():
            if hasattr(module, "weight_orig"):
                torch.nn.utils.prune.remove(module, "weight")
                print(f"Removed weight pruning from {module}")
            if hasattr(module, "bias_orig"):
                torch.nn.utils.prune.remove(module, "bias")
                print(f"Removed bias pruning from {module}")
            if hasattr(module, "raw_scores"):
                del module.raw_scores

    disable_pruning(model)

    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(state_dict, os.path.join(wandb.run.dir, "model.pt"))

    def reinitialize_classifier_head(head):
        """Randomly initialize and finetune the first head only."""
        torch.nn.init.kaiming_uniform_(head.weight, a=math.sqrt(5))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(head.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        torch.nn.init.uniform_(head.bias, -bound, bound)

    # Reload model, reinitialize the first classifier head, and train just the
    # first classifier head on the entire training set with part of the
    # training set held out as validation.
    model = make_model(args, num_classes)
    model.to(DEVICE)
    model.prepare_for_task(0)
    disable_pruning(model)
    unregister_eidetic_hooks(model.eidetic_handles)
    model.load_state_dict(state_dict)
    reinitialize_classifier_head(model.classifiers[0])
    parameters = model.classifiers[0].parameters()
    optimizer = make_optimizer(
        args.optimizer,
        parameters,
        lr=args.lr,
        weight_decay=args.classifier_weight_decay,
        momentum=args.momentum if args.momentum else 0,
    )

    train_tasks, test_tasks = load_tasks(args, sample_train_randomly=False)

    if args.hold_out:
        # We're training on the entire dataset now, so the number of hold-out
        # tasks is just one.
        args.hold_out_num_tasks = 1
        (
            train_tasks,
            test_tasks,
            metrics,
            num_classes,
        ) = hold_out_training_as_test(
            args=args,
            train_tasks=train_tasks,
            test_tasks=test_tasks,
            num_classes=num_classes,
            metrics=metrics,
        )

    train_loader = torch.utils.data.DataLoader(
        train_tasks[0],  # Complete MNIST training set
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_train_workers,
        batch_size=args.batch_size,
    )

    # Train the first classifier head using the metric on the held-out portion
    # of the training set for early stopping.
    best_metric = 0
    best_epoch = 0
    current_epoch = 0

    while True:
        model.train_one_epoch(train_loader, metrics[0], optimizer)
        model.test(
            test_tasks[0],
            metrics[0],
            class_metrics[0],
            0,
            args.test_batch_size,
        )
        current_metric = metrics[0].compute().item()
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = current_epoch
        if (current_epoch - best_epoch) > args.early_stopping_patience:
            break
        current_epoch += 1

    test_loader = torch.utils.data.DataLoader(
        test_tasks[0],  # Complete MNIST test set
        num_workers=8,
        shuffle=False,
        drop_last=False,
        batch_size=args.test_batch_size,
    )

    adversarial_results = test_adversarial_robustness(model, test_loader)
    wandb.log(adversarial_results)

    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(state_dict, os.path.join(wandb.run.dir, "final-model.pt"))


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--arch",
        type=str,
        choices=[
            "MLP-small",
            "MLP-big",
            "ConvNet-small",
            "ConvNet-big",
            "ResNet18",
            "ResNet34",
            "ResNet50",
        ],
        default="MLP-small",
    )
    #######################################################################
    # Optimization arguments
    #######################################################################
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        metavar="N",
        help="Training batch size",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=2048,
        metavar="N",
        help="Test batch size",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        choices=[
            name
            for name in dir(optim)
            if name[0].isupper() and name != "Optimizer"
        ],
        help="The optimizer to use during training.",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, metavar="LR", help="Learning rate"
    )
    parser.add_argument(
        "--momentum", type=float, default=0, help="Momentum when using SGD"
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=math.inf,
        help=(
            "Number of epochs to train. This overrides early stopping patience."
        ),
    )
    parser.add_argument(
        "--early-stopping-patience",
        default=5,
        type=int,
        help=(
            "Stop training if training set accuracy has not improved in this "
            "number of epochs. After early stopping patience has been "
            "exceeded, iterative pruning begins."
        ),
    )
    #######################################################################
    # Regularization arguments
    #######################################################################
    parser.add_argument(
        "--weight-decay", default=0, type=float, help="L2 penalty on weights"
    )
    parser.add_argument(
        "--classifier-weight-decay",
        default=0,
        type=float,
        help="L2 penalty on weights of (active) classifier head",
    )
    #######################################################################
    # Pruning arguments
    #######################################################################
    parser.add_argument(
        "--pruning-type",
        nargs="+",
        default=["l2"],
        choices=["l1", "1", "l2", "2", "taylor", "random"],
        type=str,
        help=(
            "The criterion to use for selecting neurons during iterative "
            "pruning. Multiple criteria can be provided, and iterative "
            "pruning will cycle through the pruning criteria within each "
            "task. Multiple pruning step sizes can also be provided, and "
            "they will be paired with the criterion during iterative pruning."
        ),
    )
    parser.add_argument(
        "--pruning-step-size",
        nargs="+",
        default=[0.05],
        type=float,
        help=(
            "Fraction of neurons to prune in each step. To prune 10 percent "
            "of them in each step, use e.g. 0.1. Multiple step sizes can be "
            "provided, and iterative pruning will cycle through the step "
            "sizes within each task. (See --pruning-type for interactions.) "
            "With L1, L2, or random pruning, this is (currently) a function "
            "of the total number of neurons. For Taylor pruning, this is a "
            "function of either the number of unpruned neurons (the default "
            "for Taylor) or of the total number of neurons. When using "
            "Taylor pruning, to make this a function of the total number of "
            "neurons, use --pruning-step-size-is-constant."
        ),
    )
    parser.add_argument(
        "--stop-pruning-threshold",
        required=True,
        type=float,
        help=(
            "Stop pruning a task when accuracy on training set drops below "
            "the best accuracy * threshold. Set to e.g. 0.01 for a 1 percent "
            "drop."
        ),
    )
    parser.add_argument(
        "--pruning-step-size-is-constant",
        action="store_true",
        help=(
            "If true, pruning steps are a fixed percentage of the number of "
            "neurons. If false, then pruning steps are a percentage of the "
            "number of neurons that remain to be pruned."
        ),
    )
    parser.add_argument(
        "--max-recovery-epochs",
        type=int,
        required=False,
        default=5,
        help="Maximum number of epochs to run when recovering from pruning.",
    )
    parser.add_argument(
        "--reduce-learning-rate-before-pruning",
        default=0,
        type=float,
        help="Divide learning rate by this amount before pruning begins",
    )
    parser.add_argument(
        "--expand-amount",
        type=float,
        metavar="AMOUNT",
        help=(
            "Add neurons to every layer after training every task. The "
            "amount argument is a fraction if in the range (0, 1) and a "
            "number if > 1."
        ),
    )
    #######################################################################
    # EideticNet arguments
    #######################################################################
    parser.add_argument(
        "--num-training-subsets",
        type=int,
        metavar="N",
        required=True,
        help=(
            "The number of training samples into which to split the training "
            "set."
        ),
    )
    parser.add_argument(
        "--forward-transfer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Forward-transfer features learned in a previous task to "
            "subsequent tasks"
        ),
    )
    #######################################################################
    # Arguments for hyperparameter searching.
    #######################################################################
    parser.add_argument(
        "--hold-out",
        action="store_true",
        help=(
            "Hold out 10 percent of training set to evaluate hyperparameters"
        ),
    )
    parser.add_argument(
        "--hold-out-num-tasks",
        type=int,
        default=1,
        help="The number of tasks to train when using --hold-out",
    )
    #######################################################################
    # Miscellaneous arguments
    #######################################################################
    parser.add_argument(
        "--test-frequently",
        action=argparse.BooleanOptionalAction,
        help="Run test sets more frequently at the cost of slower training",
    )
    parser.add_argument(
        "--num-train-workers",
        default=os.cpu_count(),
        type=int,
        help=(
            "Number of worker processes of the train loader. Defaults to the "
            "number of CPUs."
        ),
    )
    parser.add_argument(
        "--in-channels",
        default=1,
        type=int,
        help="Number of input channels of images in dataset.",
    )
    parser.add_argument(
        "--random-seed",
        default=0,
        type=int,
        help="The seed of the random number generator",
    )
    return parser


def initialize_wandb(args):
    if not args.entity:
        os.environ["WANDB_MODE"] = "disabled"
        wandb.init()
    else:
        wandb.init(
            entity=args.entity,
            project="bagging",
            config=args,
        )


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    args.num_classes = 10
    args.num_tasks = 1
    args.task = "MNIST"
    args.entity = "excap"
    initialize_wandb(args)
    run(args)
    wandb.finish()
