
# Copyright (C) 2022 Jae-Won Chung <jwnchung@umich.edu>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example script for running Zeus on a CIFAR100 job."""

import random
import argparse

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ZEUS
from zeus2.profile_dataloader import ProfileDataLoader

# from examples.cifar100.models import all_models, get_model
from shufflenetv2 import shufflenetv2
StopIteration



def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arch",
        metavar="ARCH",
        default="shufflenetv2",
        # choices=all_models,
        # help="Model architecture: " + " | ".join(all_models),
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Maximum number of epochs to train."
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers in dataloader."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed to use for training."
    )
    parser.add_argument(
        "--power_limit", type=int, default=0, help="Desired power limit, in mW."
    )

    parser.add_argument("--learning_rate", type=float, default=0.001, help="Default learning rate")
    parser.add_argument("--dropout_rate", type=float, default=0.2, help="Default dropout rate")

    # ZEUS
    runtime_mode = parser.add_mutually_exclusive_group()
    runtime_mode.add_argument(
        "--zeus", action="store_true", help="Whether to run Zeus."
    )
    runtime_mode.add_argument(
        "--profile", action="store_true", help="Whether to just profile power."
    )

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Run the main training routine."""
    # Set random seed.
    if args.seed is not None:
        set_seed(args.seed)

    # Prepare model.
    # NOTE: Using torchvision.models would be also straightforward. For example:
    #       model = vars(torchvision.models)[args.arch](num_classes=100)

    # we don't want to recreate the model each time
    model = shufflenetv2(args.dropout_rate)

    # Prepare datasets.
    train_dataset = datasets.CIFAR100(
        root="data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                    std=(0.2673342858792401, 0.2564384629170883, 0.27615047132568404),
                ),
            ]
        ),
    )
    val_dataset = datasets.CIFAR100(
        root="data",
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                    std=(0.2673342858792401, 0.2564384629170883, 0.27615047132568404),
                ),
            ]
        ),
    )

    print(f"power limit arg: {args.power_limit}")
    print(f"learning rate arg: {args.learning_rate}")
    print(f"batch size arg: {args.batch_size}")

    train_loader = ProfileDataLoader(
        train_dataset,
        split="train",
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        power_limit=args.power_limit,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = ProfileDataLoader(
        val_dataset,
        split="eval",
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        power_limit=args.power_limit,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Send model to CUDA.
    model = model.cuda()

    # Prepare loss function and optimizer.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=args.learning_rate)
    # optimizer = optim.Adadelta(model.parameters())

    # ZEUS
    # ZeusDataLoader may early stop training when the cost is expected
    # to exceed the cost upper limit or the target metric was reached.
    # if args.zeus:
    #     assert isinstance(train_loader, PowerOptimizerDataLoader)
    #     epoch_iter = train_loader.epochs()
    # else:
        
    epoch_iter = range(args.epochs)


    # Main training loop.
    for epoch in epoch_iter:
        # the StopIteration exception will automatically stop the train loop, it doesn't need to be explicitly caught
        train(train_loader, model, criterion, optimizer, epoch, args)
        acc = validate(val_loader, model, criterion, epoch, args)
        train_loader.calculate_cost(acc)
        if acc > 0.01:
            

def train(train_loader, model, criterion, optimizer, epoch, args):
    """Train the model for one epoch."""
    model.train()
    num_samples = len(train_loader) * args.batch_size

    for batch_index, (images, labels) in enumerate(train_loader):
        labels = labels.cuda()
        images = images.cuda()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        print(
            f"Training Epoch: {epoch} [{(batch_index + 1) * args.batch_size}/{num_samples}]"
            f"\tLoss: {loss.item():0.4f}"
        )


@torch.no_grad()
def validate(val_loader, model, criterion, epoch, args):
    """Evaluate the model on the validation set."""
    model.eval()

    test_loss = 0.0
    correct = 0
    num_samples = len(val_loader) * args.batch_size

    for images, labels in val_loader:
        images = images.cuda()
        labels = labels.cuda()

        outputs = model(images)
        loss = criterion(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()

    print(
        f"Validation Epoch: {epoch}, Average loss: {test_loss / num_samples:.4f}"
        f", Accuracy: {correct / num_samples:.4f}"
    )

    return correct / num_samples


def set_seed(seed: int) -> None:
    """Set random seed for reproducible results."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main(parse_args())