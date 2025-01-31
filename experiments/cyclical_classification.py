import argparse
import math

import torch
import torch.optim as optim
from torchmetrics.classification import MulticlassAccuracy
import wandb

import eideticnet_training as eideticnet
import eideticnet_training.utils.data as data
from eideticnet_training.utils.data import make_task_classifier_dataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_fraction_zero(tensor):
    return (tensor == 0).sum() / tensor.numel()


def reinitialize_pruned_classifier_head_weights(*, model, cycle, task_id):
    """
    To enable cyclical learning, allow the classifier head of an
    already-trained head to be trained on additional (unallocated) neurons.
    Do this by reinitialize the synaptic weights of the classifier head that
    have been set to 0.

    This function assumes that the classifier head comprises a single linear
    layer. In a setting where the classifier head is more than that, this
    function will not work.
    """
    head = model.classifiers[task_id].weight

    previous_layer = model.blocks[-1][0]
    fraction_previous_zero = get_fraction_zero(previous_layer.weight)

    print(
        f"Cycle {cycle}, task {task_id} - fraction of weights in prev layer "
        f"that are 0 is {fraction_previous_zero}"
    )

    placeholder = head.clone()
    torch.nn.init.kaiming_uniform_(placeholder, a=math.sqrt(5))

    # Make mask for selecting from the placeholder just those parameters that
    # correspond to excess capacity and should be reinitialized.
    excess_neurons = previous_layer.weight_excess_capacity.all(1)
    classifier_head_neurons = head.shape[0]

    # When making the mask the same shape as the classifier head, neurons from
    # the excess capacity of the previous layer become input dimensions.
    params_to_reinitialize = excess_neurons.unsqueeze(0).repeat(
        classifier_head_neurons, 1
    )

    model.classifiers[model.phase].weight = placeholder[params_to_reinitialize]

    # Can we skip reinitializing the bias?
    """
    classifier_bias = model.classifiers[model.phase].bias
    if classifier_bias is not None:
        # Reinitialize all elements of the bias.
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(
                head
            )
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(classifier_bias, -bound, bound)
    """


def full_training(args):
    torch.random.manual_seed(0)

    if args.task in ["MNIST-5", "CIFAR10-5", "CIFAR100-5", "CIFAR100-20"]:
        train_dataset, test_dataset = data.torchvision_dataset(
            args.task.split("-")[0],
            path="datasets",
        )
        train_tasks = data.create_sequential_classification(
            train_dataset, args.num_tasks
        )
        test_tasks = data.create_sequential_classification(
            test_dataset, args.num_tasks
        )
    elif args.task == "MNIST-SVHN-CIFAR10":
        train_mnist, test_mnist = data.torchvision_dataset(
            "MNIST", path="datasets"
        )
        train_svhn, test_svhn = data.torchvision_dataset(
            "SVHN", path="datasets"
        )
        train_cifar10, test_cifar10 = data.torchvision_dataset(
            "CIFAR10", path="datasets"
        )
        train_tasks = [train_mnist, train_svhn, train_cifar10]
        test_tasks = [test_mnist, test_svhn, test_cifar10]

    """
    # Make each training task a subset of the total training task.
    for i, train_task in enumerate(train_tasks):
        fractions = [1 / args.num_cycles] * args.num_cycles
        # The earlier subsets should be smaller than the later ones.
        fractions = torch.arange(1, args.num_cycles + 1)
        # And the fractions should sum to 1.
        fractions = fractions / fractions.sum()
        print(f"training set sizes over cycles: {fractions}")

        partial_train_tasks = torch.utils.data.random_split(
            train_task, fractions
        )
        train_tasks[i] = partial_train_tasks
    """

    metrics = [
        MulticlassAccuracy(num_classes=args.num_classes).to(DEVICE)
    ] * args.num_tasks
    class_metrics = [
        MulticlassAccuracy(num_classes=args.num_classes, average="none").to(
            DEVICE
        )
    ] * args.num_tasks
    num_classes = [args.num_classes] * args.num_tasks

    print("train_tasks")
    print(train_tasks)

    if args.train_task_classifier:
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
        metrics.append(
            MulticlassAccuracy(num_classes=args.num_tasks).to(DEVICE)
        )
        class_metrics.append(
            MulticlassAccuracy(num_classes=args.num_tasks, average="none").to(
                DEVICE
            )
        )
        task_classifier_train_task = make_task_classifier_dataset(train_tasks)
        train_tasks.append(task_classifier_train_task)

        task_classifier_test_task = make_task_classifier_dataset(test_tasks)
        test_tasks.append(task_classifier_test_task)

    if args.arch == "MLP-small":
        model = eideticnet.networks.MLP(
            args.in_channels * 32 * 32,
            [args.num_classes] * args.num_tasks,
            num_layers=2,
            width=4096,
            bn=args.bn,
            dropout=args.dropout,
        ).to(DEVICE)
    elif args.arch == "MLP-big":
        model = eideticnet.networks.MLP(
            args.in_channels * 32 * 32,
            [args.num_classes] * args.num_tasks,
            bn=args.bn,
            num_layers=5,
            width=4096,
            dropout=args.dropout,
        ).to(DEVICE)
    elif args.arch == "ConvNet-small":
        raise NotImplementedError(
            "For now, only MLP is supported for the back-transfer experiment."
        )
        model = eideticnet.networks.ConvNet(
            args.in_channels,
            [args.num_classes] * args.num_tasks,
            bn=args.bn,
            num_layers=2,
            width=8,
        ).to(DEVICE)
    elif args.arch == "ConvNet-big":
        raise NotImplementedError(
            "For now, only MLP is supported for the back-transfer experiment."
        )
        model = eideticnet.networks.ConvNet(
            args.in_channels,
            [args.num_classes] * args.num_tasks,
            bn=args.bn,
            num_layers=4,
            width=64,
        ).to(DEVICE)
    elif args.arch == "ResNet":
        raise NotImplementedError(
            "For now, only MLP is supported for the back-transfer experiment."
        )
        model = eideticnet.networks.ResNet(
            args.in_channels,
            [args.num_classes] * args.num_tasks,
            [2, 2, 2, 2],
            expansion=1,
        ).to(DEVICE)

    metrics = [
        MulticlassAccuracy(num_classes=args.num_classes).to(DEVICE)
    ] * args.num_tasks
    class_metrics = [
        MulticlassAccuracy(num_classes=args.num_classes, average="none").to(
            DEVICE
        )
    ] * args.num_tasks

    if not args.stop_pruning_threshold:
        # FIXME Defaults from original experiments. These were set for
        # convenience/speed. For best results with more tasks we should use a
        # tighter threshold. Delete this later.
        tolerance = 0.01 if args.num_tasks <= 5 else 0.04
    else:
        tolerance = args.stop_pruning_threshold

    hyperparams = {
        "max_interpruning_epochs": args.max_recovery_epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "pruning_step": args.pruning_step_size,
        "pruning_step_size_is_constant": args.pruning_step_size_is_constant,
        "reduce_learning_rate_before_pruning": args.reduce_learning_rate_before_pruning,  # noqa: E501
        "tolerance": tolerance,
        "pruning_norm": args.pruning_norm,
        "random_pruning": args.random_pruning,
        "test_batch_size": args.test_batch_size,
        "test_frequently": args.test_frequently,
        "validation_tasks": test_tasks,
        "validation_metrics": metrics,
        "validation_class_metrics": class_metrics,
    }

    for cycle in range(args.num_cycles):
        # We're going to run through all tasks more than once. Each time we
        # train on a task T, the following occurs:
        # - The classifier head for T. If we've trained on T before, the same
        #   classifier head as before is used and all of its weights are
        #   retrained. Note that its synaptic connections to previously-trained
        #   neurons can be re-weighted, but the hidden (representation space)
        #   neurons that were previously trained for the task remain frozen.
        # - If we've trained on T before, all excess capacity can be used to
        #   refine the model's performance on T.
        for t in range(len(train_tasks)):
            train_task = train_tasks[t][cycle]
            train_loader = torch.utils.data.DataLoader(
                train_task,
                pin_memory=True,
                shuffle=True,
                drop_last=True,
                num_workers=10,
                batch_size=args.batch_size,
            )

            print(f"Training task {t}")
            model.set_phase(t)
            if t or cycle:
                # On the first cycle, excess capacity is only allocated to
                # tasks t > 0. On the second cycle, excess capacity should be
                # allcated even for task t = 0.
                model.prepare_for_task(
                    t, forward_transfer=args.forward_transfer
                )
                # These can be removed after rebuttal period.
                if hasattr(model, "scorer"):
                    model.scorer.assert_masks_exist = True
                    model.scorer.assert_already_trained_nonzero = True

            if cycle:
                reinitialize_pruned_classifier_head_weights(
                    model=model, cycle=cycle, task_id=t
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

            optimizer = optim.AdamW(
                parameters, lr=args.lr, weight_decay=args.weight_decay
            )

            model.train_task(
                train_loader,
                metrics[t],
                optimizer=optimizer,
                last_task=False,
                **hyperparams,
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=2048,
        metavar="N",
        help="Test batch size",
    )
    parser.add_argument(
        "--test-frequently",
        action="store_true",
        help="Run test sets more frequently at the cost of slower training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        metavar="N",
        help="number of epochs to train",
    )
    parser.add_argument(
        "--task",
        choices=[
            "MNIST-5",
            "CIFAR10-5",
            "CIFAR100-5",
            "CIFAR100-20",
            "MNIST-SVHN-CIFAR10",
        ],
        help="the classification task",
    )
    parser.add_argument(
        "--lr", type=float, default=0.0001, metavar="LR", help="learning rate"
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
    parser.add_argument(
        "--arch",
        type=str,
        choices=[
            "MLP-small",
            "MLP-big",
            "ConvNet-small",
            "ConvNet-big",
            "ResNet",
        ],
        default="MLP-small",
    )
    parser.add_argument("--bn", action="store_true")
    parser.add_argument(
        "--dropout", default=0, type=float, help="The dropout probabiliity."
    )
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--random-pruning", action="store_true")
    parser.add_argument(
        "--pruning-norm",
        default=2,
        type=lambda x: int(x) if x in ["1", "2"] else x,
    )
    parser.add_argument(
        "--num-cycles",
        default=1,
        type=int,
        help="Number of times to cycle through all tasks",
    )
    parser.add_argument(
        "--weight-decay", default=0, type=float, help="L2 penalty on weights"
    )
    parser.add_argument(
        "--classifier-weight-decay",
        default=0,
        type=float,
        help="L2 penalty on weights of (active) classifier head",
    )
    parser.add_argument(
        "--pruning-step-size",
        default=0.05,
        type=float,
        help=(
            "Fraction of neurons to prune in each step. To prune 10 percent "
            "of them in each step, use e.g. 0.1. With L1, L2, or random "
            "pruning, this is (currently) a function of the total number of "
            "neurons. For Taylor pruning, this is a function of either the "
            "number of unpruned neurons (the default for Taylor) or of the "
            "total number of neurons. When using Taylor pruning, to make this "
            "a function of the total number of neurons, use "
            "--pruning-step-size-is-constant."
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
        "--train-task-classifier",
        action="store_true",
        help=(
            "After training all tasks, train an additional classifier head "
            "that predicts the task to which task a given input belongs."
        ),
    )
    parser.add_argument(
        "--in-channels",
        default=3,
        choices=[3, 1],
        type=int,
        help="Number of channels of input data.",
    )

    args = parser.parse_args()

    wandb.init(
        project="single_dataset_sequential_with_backtransfer", config=args
    )

    args.num_classes = 100
    if args.task in ["MNIST-5", "CIFAR10-5", "CIFAR100-5"]:
        args.num_tasks = 5
    elif args.task == "CIFAR100-20":
        args.num_tasks = 20
    else:
        args.num_tasks = 3
    full_training(args)
    wandb.finish()
