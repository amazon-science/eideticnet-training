import argparse
import os
import random
from functools import partial

import expand_network
import numpy as np
import torch
import torch.optim as optim
import wandb
from torchmetrics.classification import MulticlassAccuracy

import eideticnet_training as eideticnet
from eideticnet_training.utils.data import make_task_classifier_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_optimizer(name, parameters, **kwargs):
    return getattr(optim, name)(parameters, **kwargs)


def load_tasks(args):
    if args.task in {
        "MNIST-5",
        "CIFAR10-5",
        "CIFAR100-5",
        "IMAGENETTE-5",
        "IMAGEWOOF-5",
        "CIFAR100-10",
        "CIFAR100-20",
    }:
        train_dataset, test_dataset = eideticnet.data.torchvision_dataset(
            args.task.split("-")[0],
            path="~/datasets",
        )
        train_tasks = eideticnet.data.create_sequential_classification(
            train_dataset, args.num_tasks
        )
        test_tasks = eideticnet.data.create_sequential_classification(
            test_dataset, args.num_tasks
        )
    elif set(args.task.split("-")) == {"MNIST", "SVHN", "CIFAR10"}:
        train_mnist, test_mnist = eideticnet.data.torchvision_dataset(
            "MNIST", path="~/datasets"
        )
        train_svhn, test_svhn = eideticnet.data.torchvision_dataset(
            "SVHN", path="~/datasets"
        )
        train_cifar10, test_cifar10 = eideticnet.data.torchvision_dataset(
            "CIFAR10", path="~/datasets"
        )
        train_names = args.task.lower().split("-")
        datasets = {
            "train_mnist": train_mnist,
            "test_mnist": test_mnist,
            "train_svhn": train_svhn,
            "test_svhn": test_svhn,
            "train_cifar10": train_cifar10,
            "test_cifar10": test_cifar10,
        }
        train_tasks = [datasets[f"train_{name}"] for name in train_names]
        test_tasks = [test_mnist, test_svhn, test_cifar10]
    elif args.task == "PMNIST":
        train_tasks, test_tasks = eideticnet.data_pmnist.get_pmnist(args)

    return train_tasks, test_tasks, test_dataset


def make_metrics(args):
    metrics = [
        MulticlassAccuracy(num_classes=args.num_classes).to(DEVICE)
    ] * args.num_tasks
    class_metrics = [
        MulticlassAccuracy(num_classes=args.num_classes, average="none").to(
            DEVICE
        )
    ] * args.num_tasks
    return metrics, class_metrics


def hold_out_training_as_test(
    *, args, train_tasks, test_tasks, metrics, num_classes
):
    # Evaluate hyperparameters on 10% of training set on the first task.
    assert len(test_tasks) == len(train_tasks)
    for i, test_task in enumerate(test_tasks):
        print(i, dir(test_task.dataset))
    test_transforms = [tt.dataset.transform for tt in test_tasks]
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


def add_task_classification_task(
    args, train_tasks, test_tasks, metrics, class_metrics, num_classes
):
    # When instantiating the model, all of the classifier heads are
    # instantiated. So if we're going to train a final task classifier
    # after training all of the individual tasks, include that here.
    # This also requires a different metric for evaluation.
    print(
        "Will use early stopping patience of "
        f"{args.task_classifier_early_stopping_patience} when training "
        "the task classifier."
    )
    task_classifier_num_classes = args.num_tasks
    num_classes.append(task_classifier_num_classes)
    metrics.append(MulticlassAccuracy(num_classes=args.num_tasks).to(DEVICE))
    class_metrics.append(
        MulticlassAccuracy(num_classes=args.num_tasks, average="none").to(
            DEVICE
        )
    )
    task_classifier_train_task = make_task_classifier_dataset(train_tasks)
    train_tasks.append(task_classifier_train_task)

    task_classifier_test_task = make_task_classifier_dataset(test_tasks)
    test_tasks.append(task_classifier_test_task)


def make_model(args, num_classes):
    if args.arch == "MLP-small":
        # Hard-coded width for PMNIST for partity with CLNP
        # https://arxiv.org/abs/1903.04476
        width = 2000 if args.task == "PMNIST" else 512
        num_layers = 2 if args.task == "PMNIST" else 2
        model = eideticnet.networks.MLP(
            args.in_channels * 32 * 32,
            num_classes,
            num_layers=num_layers,
            width=width,
            bn=args.bn,
            dropout=args.dropout,
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


def null_expander(model):
    pass


def network_expander(arch, amount):
    add_out_func = partial(expand_network.get_amount, amount=amount)
    if "MLP" in args.arch:
        return partial(expand_network.expand_mlp, add_out_func=add_out_func)
    elif "ConvNet" in args.arch:
        return partial(
            expand_network.expand_convnet, add_out_func=add_out_func
        )
    elif "ResNet" in args.arch:
        return partial(expand_network.expand_resnet, add_out_func=add_out_func)
    else:
        raise ValueError(f"Unexpected network architecture '{arch}'")


def make_expander(args):
    if args.expand_amount is None:
        return null_expander
    else:
        return network_expander(args.arch, args.expand_amount)


def run(args):
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.random.manual_seed(args.random_seed)

    print("run #1 args.num_classes", args.num_classes)

    train_tasks, test_tasks, test_dataset = load_tasks(args)

    metrics, class_metrics = make_metrics(args)

    print("run #2 args.num_classes", args.num_classes)
    num_classes = [args.num_classes] * args.num_tasks
    print("run #3 num_classes", args.num_classes)

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

    print("run #4 (post hold_out_training_as_test) num_classes", num_classes)

    if args.train_task_classifier:
        add_task_classification_task(
            args, train_tasks, test_tasks, metrics, class_metrics, num_classes
        )

    model = make_model(args, num_classes)
    expander = make_expander(args)

    # FIXME deprecate 'tolerance'.
    tolerance = args.stop_pruning_threshold

    hyperparams = {
        "max_interpruning_epochs": args.max_recovery_epochs,
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
        "lower_bound": args.lower_bound,
    }

    for t, train_task in enumerate(train_tasks):
        train_loader = torch.utils.data.DataLoader(
            train_task,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
            num_workers=args.num_train_workers,
            batch_size=args.batch_size,
        )

        print(f"Training task {t}")

        if not args.lower_bound:
            if args.reset:
                # Reset the weights of the network for each task to obtain a
                # baseline performance.
                model.reset_capacity_buffers()
                model.random_initialize_excess_capacity()
            else:
                model.prepare_for_task(
                    t, forward_transfer=args.forward_transfer
                )

            if args.classifier_weight_decay:
                parameters = [
                    {"params": model.blocks.parameters()},
                    {
                        "params": model.classifiers[t].parameters(),
                        "weight_decay": args.classifier_weight_decay,
                    },
                ]
            else:
                parameters = model.parameters()

            optimizer = make_optimizer(
                args.optimizer,
                parameters,
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
            is_last_task = t == len(test_tasks) - 1
        elif t == 0:
            model.set_phase(t)
            for i in range(1, len(model.classifiers)):
                model.classifiers[i] = model.classifiers[0]
            optimizer = make_optimizer(
                args.optimizer,
                model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )

            is_last_task = True

        if args.train_task_classifier and is_last_task:
            # Convergence takes longer for the task classifier, because it is
            # the concatenation of the datasets for all tasks, so a larger
            hyperparams[
                "early_stopping_patience"
            ] = args.task_classifier_early_stopping_patience

        hyperparams["validation_task_nums"] = list(range(t + 1))

        model.train_task(
            train_loader,
            metrics[t],
            optimizer=optimizer,
            last_task=is_last_task,
            **hyperparams,
        )
        expander(model=model)
        print(model)

    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(state_dict, os.path.join(wandb.run.dir, "model.pt"))
    wandb.save(os.path.join(wandb.run.dir, "model.pt"))

    if args.train_task_classifier:
        # FIXME refactor into a reusable function.
        class_metric = MulticlassAccuracy(
            num_classes=args.total_num_classes, average="none"
        ).to(DEVICE)
        oracle_class_metric = MulticlassAccuracy(
            num_classes=args.total_num_classes, average="none"
        ).to(DEVICE)

        for preds, targets in model.predict(test_dataset, 128):
            task_prediction = preds[-1].argmax(1)
            eideticnet.utils.utils.update_class_predictions_metrics(
                preds[:-1],
                task_prediction,
                targets,
                args,
                class_metric,
                oracle_class_metric,
            )

        m = class_metric.compute()
        om = oracle_class_metric.compute()
        print("FINAL PER CLASS ACCURACY:", m)
        print("FINAL (oracle) PER CLASS ACCURACY:", om)

        wandb.log(
            {
                "eval/class_metric_final": m.tolist(),
                "eval/class_metric_final_oracle": om.tolist(),
            }
        )

    if args.lower_bound:
        # FIXME refactor into a reusable function.
        class_metric = MulticlassAccuracy(
            num_classes=args.total_num_classes, average="none"
        ).to(DEVICE)
        for preds, targets in model.predict(test_dataset, 128):
            class_metric.update(preds[0][:, : args.total_num_classes], targets)

        m = class_metric.compute()
        print("FINAL PER CLASS ACCURACY:", m)
        wandb.log(
            {
                "eval/class_metric_final": m.tolist(),
            }
        )


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--task",
        choices=[
            "PMNIST",
            "MNIST-5",
            "CIFAR10-5",
            "CIFAR100-5",
            "CIFAR100-10",
            "CIFAR100-20",
            "IMAGENETTE-5",
            "IMAGEWOOF-5",
            "MNIST-SVHN-CIFAR10",
            "MNIST-CIFAR10-SVHN",
            "SVHN-CIFAR10-MNIST",
            "SVHN-MNIST-CIFAR10",
            "CIFAR10-MNIST-SVHN",
            "CIFAR10-SVHN-MNIST",
        ],
        help="The sequential classification task",
    )
    #######################################################################
    # Network arguments
    #######################################################################
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
    parser.add_argument(
        "--bn",
        action="store_true",
        help=(
            "Use batch normalization. ResNets always use batch normalization. "
            "This argument only applies to MLP and smaller convolutional "
            "networks."
        ),
    )
    parser.add_argument(
        "--dropout",
        default=0,
        type=float,
        help="Use dropout with this probability. Only applies to MLP.",
    )
    parser.add_argument(
        "--in-channels",
        default=3,
        choices=[3, 1],
        type=int,
        help="Number of channels of input data.",
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
        "--forward-transfer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Forward-transfer features learned in a previous task to "
            "subsequent tasks"
        ),
    )
    parser.add_argument(
        "--train-task-classifier",
        action="store_true",
        help=(
            "After training all tasks, train an additional classifier head "
            "that predicts the task to which task a given input belongs."
        ),
    )
    parser.add_argument(
        "--task-classifier-early-stopping-patience",
        default=10,
        type=int,
        help=(
            "When training the task classifier, stop training if training set "
            "accuracy has not improved in this number of epochs. Convergence "
            "takes longer for the task classifier, because it is the "
            "concatenation of the datasets for all tasks, so its best to use "
            "a larger number for the task classifier than the individual "
            "tasks."
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
        default=3,
        help="The number of tasks to train when using --hold-out",
    )
    #######################################################################
    # Arguments for baseline training
    #######################################################################
    parser.add_argument("--upper-bound", action="store_true")
    parser.add_argument("--lower-bound", action="store_true")
    parser.add_argument(
        "--reset",
        "--reinitialize-after-each-task",
        action="store_true",
        help=(
            "After training each task, reset the excess capacity and "
            "reinitalize all of the weights of the network"
        ),
    )
    #######################################################################
    # Miscellaneous arguments
    #######################################################################
    parser.add_argument(
        "--test-frequently",
        action="store_true",
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
        "--random-seed",
        default=0,
        type=int,
        help="The seed of the random number generator",
    )
    parser.add_argument(
        "--entity", required=True, help="The wandb entity for logging."
    )
    parser.add_argument(
        "--venue",
        required=False,
        help="Add a venue tag for conference/workshop submissions.",
    )
    return parser


def initialize_wandb(args):
    if not args.entity:
        os.environ["WANDB_MODE"] = "disabled"
        wandb.init()
    else:
        wandb.init(
            entity=args.entity,
            project="single_dataset_sequential",
            config=args,
        )


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    initialize_wandb(args)

    if args.upper_bound:
        args.stop_pruning_threshold = -1
        if args.task in {
            "MNIST-5",
            "CIFAR10-5",
            "IMAGENETTE-5",
            "IMAGEWOOF-5",
        }:
            args.num_tasks = 1
            args.num_classes = 10
            args.num_classes_per_task = 2
        elif args.task in {"CIFAR100-5", "CIFAR100-10", "CIFAR100-20"}:
            args.num_tasks = 1
            args.num_classes = 100
    elif args.task in {
        "MNIST-5",
        "CIFAR10-5",
        "CIFAR100-5",
        "IMAGENETTE-5",
        "IMAGEWOOF-5",
    }:
        args.num_tasks = 5
        args.num_classes = 100
        if args.task == "CIFAR100-5":
            args.num_classes_per_task = 20
            args.total_num_classes = 100
        else:
            args.total_num_classes = 10
            args.num_classes_per_task = 2
    elif args.task == "CIFAR100-10":
        args.total_num_classes = 100
        args.num_tasks = 10
        args.num_classes_per_task = 10
        args.num_classes = 100
    elif args.task == "CIFAR100-20":
        args.total_num_classes = 100
        args.num_tasks = 20
        args.num_classes = 100
        args.num_classes_per_task = 5
    elif args.task == "PMNIST":
        args.total_num_classes = 10
        args.num_tasks = 10
        args.num_classes = 10
        args.num_classes_per_task = 10
    else:
        args.num_tasks = 3
        args.num_classes = 100
        args.num_classes_per_task = 10

    run(args)
    wandb.finish()
