
"""Script for running Zeus on the CIFAR100 dataset to train Shufflenet V2."""

import argparse
import sys
from pathlib import Path

from zeus2.job import Job 
from zeus2.profiler import Profiler
from zeus.util import FileAndConsole


def parse_args() -> argparse.Namespace:
    """Parse commandline arguments."""
    parser = argparse.ArgumentParser()

    # This random seed is used for
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
        "--lr_min", type=int, default=0.001, help="Smallest learning rate to consider"
    )
    parser.add_argument(
        "--lr_max", type=int, default=0.1, help="Largest learning rate to consider"
    )
    
    # # The range of power limits to consider. 
    # parser.add_argument(
    #     "--pl_min", type=int, default=100000, help="Smallest power limit to consider (in mW)"
    # )
    # parser.add_argument(
    #     "--pl_max", type=int, default=200000, help="Largest power limit to consider (in mW)"
    # )

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

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Run Zeus on CIFAR100."""

    # The top-level class for running Zeus.
    # - The batch size optimizer is desinged as a pluggable policy.
    # - Paths (log_base and monitor_path) assume our Docker image's directory structure.
    # - For CIFAR100, one epoch of training may not take even ten seconds (especially for
    #   small models like squeezenet). ZeusDataLoader automatically handles profiling power
    #   limits over multiple training epochs such that the profiling window of each power
    #   limit fully fits in one epoch. However, the epoch duration may be so short that
    #   profiling even one power limit may not fully fit in one epoch. In such cases,
    #   ZeusDataLoader raises a RuntimeError, and the profiling window should be narrowed
    #   by giving smaller values to profile_warmup_iters and profile_measure_iters in the
    #   constructor of ZeusMaster.
    master = Profiler(
        log_base="/workspace/zeus_logs",
        seed=args.seed,
        monitor_path="/workspace/zeus/zeus_monitor/zeus_monitor",
        observer_mode=False,
        profile_warmup_iters=10,
        profile_measure_iters=40,
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
        workdir="/workspace/zeus/examples/cifar100",
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
            # "--dropout_rate", "{dropout_rate}"
        ],
        # fmt: on
    )

    # Generate a list of batch sizes with only power-of-two values.
    batch_sizes = [args.b_min]
    while (bs := batch_sizes[-1] * 2) <= args.b_max:
        batch_sizes.append(bs)

    # Create a designated log directory inside `args.log_base` just for this run of Zeus.
    # Six types of outputs are generated.
    # 1. Power monitor ouptut (`bs{batch_size}+e{epoch_num}.power.log`):
    #      Raw output of the Zeus power monitor.
    # 2. Profiling results (`bs{batch_size}.power.json`):
    #      Train-time average power consumption and throughput for each power limit,
    #      the optimal power limit determined from the result of profiling, and
    #      eval-time average power consumption and throughput for the optimal power limit.
    # 3. Training script output (`rec{recurrence_num}+try{trial_num}.train.log`):
    #      The raw output of the training script. `trial_num` exists because the job
    #      may be early stopped and re-run with another batch size.
    # 4. Training result (`rec{recurrence_num}+try{trial_num}+bs{batch_size}.train.json`):
    #      The total energy, time, and cost consumed, and the number of epochs trained
    #      until the job terminated. Also, whether the job reached the target metric at the
    #      time of termination. Early-stopped jobs will not have reached their target metric.
    # 5. ZeusMaster output (`master.log`): Output from ZeusMaster, including MAB outputs.
    # 6. Job history (`history.py`):
    #      A python file holding a list of `HistoryEntry` objects. Intended use is:
    #      `history = eval(open("history.py").read())` after importing `HistoryEntry`.
    master_logdir = master.build_logdir(
        job=job,
        eta_knob=args.eta_knob,
        beta_knob=args.beta_knob,
        exist_ok=False,  # Should err if this directory exists.
    )

    # Overwrite the stdout file descriptor with an instance of `FileAndConsole`, so that
    # all calls to `print` will write to both the console and the master log file.
    sys.stdout = FileAndConsole(Path(master_logdir) / "master.log")

    # Run Zeus!
    bs, lr, pl = master.profile(
        job=job,
        learning_rates=[0.9, 0.9, 1.0, 1.1, 1.2],
        batch_sizes=batch_sizes,
        beta_knob=args.beta_knob,
        eta_knob=args.eta_knob,
    )

    print(f"optimized batch size: {bs}, learning rate: {lr}, power limit: {pl}")


if __name__ == "__main__":
    main(parse_args())
