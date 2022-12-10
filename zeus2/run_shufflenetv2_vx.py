
"""Script for running Zeus on the CIFAR100 dataset to train Shufflenet V2."""

import argparse
import sys
from pathlib import Path

from zeus2.job import Job 
from zeus2.profiler import Profiler
from zeus.util import FileAndConsole

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms



def parse_args() -> argparse.Namespace:
    """Parse commandline arguments."""
    parser = argparse.ArgumentParser()

    # This random seed is a legacy from Zeus. It was originally used for
    # 1. Multi-Armed Bandit inside PruningGTSBatchSizeOptimizer, and
    # 2. Providing random seeds for training.
    # Especially for 2, the random seed given to the nth recurrence job is args.seed + n.
    parser.add_argument("--seed", type=int, default=1, help="Random seed")

    # Default ML parameters.
    # The first recurrence uses these parameters, and it must reach the target metric.
    parser.add_argument("--b_0", type=int, default=1024, help="Default batch size")
    parser.add_argument("--lr_0", type=float, default=0.001, help="Default learning rate")
    parser.add_argument("--dropout_0", type=float, default=1.0, help="Default dropout rate")

    # The range of batch sizes to consider. The example script generates a list of power-of-two
    # batch sizes, but batch sizes need not be power-of-two for Zeus.
    parser.add_argument(
        "--b_min", type=int, default=8, help="Smallest batch size to consider"
    )
    parser.add_argument(
        "--b_max", type=int, default=4096, help="Largest batch size to consider"
    )

    # The range of learning rates to consider. 
    parser.add_argument(
        "--lr_min", type=float, default=0.8, help="Smallest learning rate multiplier to consider"
    )
    parser.add_argument(
        "--lr_max", type=float, default=1.2, help="Largest learning rate multiplier to consider"
    )
    parser.add_argument(
        "--num_lr", type=int, default=5, help="Number of learning rates values to consider"
    )

    # The \\eta knob trades off time and energy consumption. See Equation 2 in the paper.
    # The \\beta knob defines the early stopping threshold. See Section 4.4 in the paper.
    parser.add_argument(
        "--eta_knob", type=float, default=0.5, help="TTA-ETA tradeoff knob"
    )
    parser.add_argument(
        "--beta_knob", type=float, default=2.0, help="Early stopping threshold"
    )

    # Jobs are terminated when one of the three happens:
    # 1. The target validation metric is reached.
    # 2. The number of epochs exceeds the maximum number of epochs set.
    # 3. The cost of the next epoch is expected to exceed the early stopping threshold.
    parser.add_argument(
        "--target_metric", type=float, default=0.50, help="Target validation metric"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=100, help="Max number of epochs to train"
    )
    parser.add_argument("--warmup_iters", type=int, default=3)
    parser.add_argument("--profile_iters", type=int, default=10)
    parser.add_argument('--acc_thresholds', type=float, nargs='+', default=[], help="Accuracy thresholds to profile on.")
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[], help="Batch sizes to profile over.")
    parser.add_argument('--learning_rates', type=float, nargs='+', default=[], help="Learning rates to profile over.")
    parser.add_argument('--dropout_rates', type=float, nargs='+', default=[], help="Dropout rates to profile over.")

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Run automatic-bassoon on CIFAR100."""

    # The top-level class for running automatic-bassoon.
    # - Paths (log_base and monitor_path) assume our Docker image's directory structure.
    master = Profiler(
        log_base="/workspace/zeus_logs",
        seed=args.seed,
        monitor_path="/workspace/zeus/zeus_monitor/zeus_monitor",
        observer_mode=False,
        profile_warmup_iters=args.warmup_iters,
        profile_measure_iters=args.profile_iters,
    )

    # Definition of the CIFAR100 job.
    # The `Job` class encloses all information needed to run training. The `command` parameter is
    # a command template. Curly-braced parameters are recognized by Zeus and automatically filled.
    job = Job(
        dataset="cifar100",
        network="shufflenetv2",
        optimizer="adam",
        target_metric=args.target_metric,
        max_epochs=args.max_epochs,
        default_bs=args.b_0,
        default_lr=args.lr_0,  
        default_dropout=args.dropout_0,
        workdir="/workspace/zeus/zeus2",
        # fmt: off
        command=[
            "python",
            "train_lr.py",
            "--zeus",
            "--arch", "shufflenetv2",
            "--batch_size", "{batch_size}",
            "--epochs", "{epochs}",
            "--seed", "{seed}",
            "--learning_rate", "{learning_rate}",
            "--dropout_rate", "{dropout_rate}",
            "--power_limit", "{power_limit}",
        ],
        # fmt: on
    )

    # Generate a list of batch sizes with only power-of-two values.

    master_logdir = master.build_logdir(
        job=job,
        eta_knob=args.eta_knob,
        beta_knob=args.beta_knob,
        exist_ok=False,  # Should err if this directory exists.
    )

    # Overwrite the stdout file descriptor with an instance of `FileAndConsole`, so that
    # all calls to `print` will write to both the console and the master log file.
    sys.stdout = FileAndConsole(Path(master_logdir) / "master.log")

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
    
    master.train_with_profiling(
        job=job, 
        eta_knob=args.eta_knob, 
        beta_knob=args.beta_knob, 
        train_dataset=train_dataset, 
        val_dataset=val_dataset,
        target_acc=args.target_metric,
        n_epochs=args.max_epochs,
        acc_thresholds=args.acc_thresholds,
        batch_sizes=args.batch_sizes,
        learning_rates=args.learning_rates,
        dropout_rates=args.dropout_rates
    )


if __name__ == "__main__":
    main(parse_args())
