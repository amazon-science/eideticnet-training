import argparse
import os
from functools import partial

import robustbench.model_zoo.architectures.resnet
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import wandb
from robustbench.data import load_cifar10
from robustbench.eval import benchmark
from robustbench.utils import clean_accuracy
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import eideticnet_training.losses

hidden_state_losses = {}


def hidden_state_losses_hook(
    module, args, output, module_name=None, orthogonality_loss_func=None
):
    hidden_state_losses[module_name] = orthogonality_loss_func(output)


def compute_orthogonality_loss(args, model, orthogonality_loss_func=None):
    if args.regularize == "weights":
        # Sum orthogonality loss for all weight matrices
        ortho_loss = 0
        num_modules = 0
        for module_name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                num_modules += 1
                ortho_loss += orthogonality_loss_func(module.weight)
        return ortho_loss / num_modules
    else:  # hidden states
        # Sum orthogonality loss for all hidden states
        ortho_loss = 0
        num_modules = 0
        for module_name, hidden_state_loss in hidden_state_losses.items():
            num_modules += 1
            ortho_loss += hidden_state_loss
        return ortho_loss / num_modules


def train(args, epoch, device):
    model.train()
    total_classification_loss = 0
    total_ortho_loss = 0
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)

        # Compute main loss
        batch_classification_loss = criterion(output, target)

        # Add orthogonality regularization
        if args.orthogonality_loss:
            batch_ortho_loss = compute_orthogonality_loss(
                args, model, args.orthogonality_loss_func
            )
        else:
            batch_ortho_loss = torch.zeros(1).to(output.device)
        batch_loss = (
            batch_classification_loss + args.lambda_ortho * batch_ortho_loss
        )

        batch_loss.backward()
        optimizer.step()

        wandb.log(
            {
                "batch/classification_loss": batch_classification_loss.item(),
                "batch/ortho_loss": batch_ortho_loss.item(),
                "batch/loss": batch_loss.item(),
            }
        )

        total_classification_loss += batch_classification_loss.item()
        total_ortho_loss += batch_ortho_loss.item()
        total_loss += batch_loss.item()

    num_batches = len(train_loader)

    return (
        x / num_batches
        for x in (total_loss, total_classification_loss, total_ortho_loss)
    )


# Command line arguments
def get_parser():
    parser = argparse.ArgumentParser(
        description="CIFAR-10 MLP with orthogonality regularization"
    )
    parser.add_argument(
        "--regularize",
        choices=["weights", "hidden"],
        required=False,
        help="What to regularize: weights or hidden states",
    )
    parser.add_argument(
        "--orthogonality-loss",
        choices=[
            func_name
            for func_name in dir(eideticnet_training.losses)
            if "loss" in func_name
        ],
        required=False,
        help="Orthogonality regularlizer from eideitcnet_training.losses",
    )
    parser.add_argument(
        "--lambda-ortho",
        type=float,
        default=0.1,
        help="Orthogonality regularization strength",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size"
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        dest="lr",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="WandB entity name",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="WandB project name",
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="WandB run name",
    )
    return parser


def initialize_wandb(args):
    if not args.wandb_entity:
        os.environ["WANDB_MODE"] = "disabled"
        wandb.init()
    else:
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            config=args,
        )


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_transforms = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
            ),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transforms
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )

    x_test, y_test = load_cifar10()
    x_test = torch.FloatTensor(x_test).to(device)
    y_test = torch.LongTensor(y_test).to(device)

    args.dataset = "CIFAR-10"
    args.arch = "MLP"

    initialize_wandb(args)

    model = robustbench.model_zoo.architectures.resnet.ResNet18().to(device)

    if args.orthogonality_loss:
        args.orthogonality_loss_func = getattr(
            eideticnet_training.losses, args.orthogonality_loss
        )
        for module_name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                hook = partial(
                    hidden_state_losses_hook,
                    module_name=module_name,
                    orthogonality_loss_func=args.orthogonality_loss_func,
                )
                module.register_forward_hook(hook)

    wandb.watch(model, log="all")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_losses = []
    test_accuracies = []

    for epoch in tqdm.trange(1, args.epochs + 1):
        train_loss, classification_loss, ortho_loss = train(
            args, epoch, device
        )
        clean_test_accuracy = clean_accuracy(
            model, x_test, y_test, batch_size=args.batch_size
        )

        results = {
            "epoch/train/loss": train_loss,
            "epoch/train/classification_loss": classification_loss,
            "epoch/train/orthogonality_loss": ortho_loss,
            "epoch/test/accuracy": clean_test_accuracy,
        }
        wandb.log(results)

    adversarial_hyperparams = (
        # ("Linf", 0.1),
        ("Linf", 0.3),
        # ("Linf", 1.0),
        # ("Linf", 3.0),
        # ("L2", 0.1),
        # ("L2", 0.3),
        # ("L2", 1.0),
        # ("L2", 3.0),
    )

    adversarial_results = {}
    for norm, eps in adversarial_hyperparams:
        clean_accuracy, adversarial_accuracy = benchmark(
            model,
            dataset="cifar10",
            threat_model=norm,
            eps=eps,
            batch_size=args.batch_size,
            device=device,
        )
        key = f"adversarial/accuracy_{norm}_{eps}"
        wandb.log({key: adversarial_accuracy})
