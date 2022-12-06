
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

# Stitched together with code from https://github.com/ngduyanhece/neuMF/blob/master/NeuMF.py 

"""Example script for running Zeus on a CIFAR100 job."""

import os
import argparse

import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import model_names

import evaluate
import data_utils
import config
import numpy as np

# ZEUS
from zeus.run import ZeusDataLoader 
from zeus.profile.torch import ProfileDataLoader

from model.neumf import NCF


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arch",
        metavar="ARCH",
        default="NeuMF",
        choices=model_names,
        help="model architecture: " + " | ".join(model_names) + " (default: neumf)"
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

    parser.add_argument("--top_k", type=int, default=10, help="compute metrics@top_k")
    parser.add_argument("--factor_num", type=int, default=32, help="predictive factors numbers in the model")
    parser.add_argument("--num_layers", type=int, default=3, help="number of layers in MLP model")

    parser.add_argument("--num_ng", type=int, default=4, help="sample negative items for training")

    parser.add_argument("--learning_rate", type=float, default=0.001, help="Default learning rate")
    parser.add_argument("--dropout_rate", type=float, default=0.0, help="Default dropout rate")

    # ZEUS
    runtime_mode = parser.add_mutually_exclusive_group()
    runtime_mode.add_argument(
        "--zeus", action="store_true", help="Whether to run Zeus."
    )
    runtime_mode.add_argument(
        "--profile", action="store_true", help="Whether to just profile power."
    )

    return parser.parse_args()

train_data, test_data, user_num, item_num, train_mat = data_utils.load_all()


def main(args: argparse.Namespace) -> None:
    """Run the main training routine."""
    # Prepare model.
    # NOTE: Using torchvision.models would be also straightforward. For example:
    #       model = vars(torchvision.models)[args.arch](num_classes=100)
    if args.arch == "neumf":
        model = NCF(user_num, item_num, args.factor_num, args.num_layers, args.dropout_rate, config.model, None, None)
    else:
        model = None

    # Prepare datasets.
    train_dataset = data_utils.NCFData(train_data, item_num, train_mat, args.num_ng, True)
    val_dataset = data_utils.NCFData(test_data, item_num, train_mat, 0, False)

    # ZEUS
    # Prepare dataloaders.
    if args.zeus:
        # Zeus
        train_loader = ZeusDataLoader(
            train_dataset,
            max_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            shuffle=True,
            num_workers=args.num_workers,
        )
        val_loader = ZeusDataLoader(
            val_dataset,
            batch_size=args.batch_size, # this was test_num_neg in their original
            learning_rate=args.learning_rate,
            shuffle=False,
            num_workers=args.num_workers,
        )
    elif args.profile:
        print(f"power limit arg: {args.power_limit}")
        train_loader = ProfileDataLoader(
            train_dataset,
            split="train",
            batch_size=args.batch_size,
            power_limit=args.power_limit,
            shuffle=True,
            num_workers=args.num_workers,
        )
        val_loader = ProfileDataLoader(
            val_dataset,
            split="eval",
            batch_size=args.batch_size,
            power_limit=args.power_limit,
            shuffle=False,
            num_workers=args.num_workers,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

    # Send model to CUDA.
    model = model.cuda()

    # Prepare loss function and optimizer.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=args.learning_rate)
    # optimizer = optim.Adadelta(model.parameters())

    # ZEUS
    # ZeusDataLoader may early stop training when the cost is expected
    # to exceed the cost upper limit or the target metric was reached.
    if args.zeus:
        assert isinstance(train_loader, ZeusDataLoader)
        epoch_iter = train_loader.epochs()
    else:
        epoch_iter = range(args.epochs)

    best_hr = 0

    # Main training loop.
    for epoch in epoch_iter:
        start_time = time.time()
        train(train_loader, model, criterion, optimizer, epoch, args)
        HR, NDCG, epoch = validate(val_loader, model, criterion, epoch, args, start_time)

        if HR > best_hr:
            best_hr, best_ndcg, best_epoch = HR, NDCG, epoch

        # ZEUS
        if args.zeus:
            assert isinstance(train_loader, ZeusDataLoader)
            train_loader.report_metric(best_ndcg, higher_is_better=True)
        elif args.profile:
            if train_loader.reached_target_metric(best_ndcg):
                break

def train(train_loader, model, criterion, optimizer, epoch, args):
    """Train the model for one epoch."""
    model.train()
    num_samples = len(train_loader) * args.batch_size

    train_loader.dataset.ng_sample()

    for batch_index, (user, item, label) in enumerate(train_loader):
        user = user.cuda()
        item = item.cuda()
        label = label.float().cuda()

        model.zero_grad()
        prediction = model(user, item)
        loss = criterion(prediction, label)
        loss.backward()
        optimizer.step()
		# writer.add_scalar('data/loss', loss.item(), count)

        print(
            f"Training Epoch: {epoch} [{(batch_index + 1) * args.batch_size}/{num_samples}]"
            f"\tLoss: {loss.item():0.4f}"
        )


@torch.no_grad()
def validate(val_loader, model, criterion, epoch, args, start_time):
    """Evaluate the model on the validation set."""
    model.eval()

    num_samples = len(val_loader) * args.batch_size

    HR, NDCG, test_loss = evaluate.metrics(model, criterion, val_loader, args.top_k)

    elapsed_time = time.time() - start_time
    print("The time elapse of epoch {:03d}".format(epoch) + " is: " + 
			time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
    print("Validation Epoch: {epoch}, HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))
    print(f"\tAverage Loss:{test_loss / num_samples:.4f}")

    if HR > best_hr:
        best_hr, best_ndcg, best_epoch = HR, NDCG, epoch

    return best_hr, best_ndcg, best_epoch


if __name__ == "__main__":
    main(parse_args())
