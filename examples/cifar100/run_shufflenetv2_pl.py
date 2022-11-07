
"""Script for running Zeus on the CIFAR100 dataset to train Shufflenet V2."""

import argparse
import json
import pprint
import sys
from pathlib import Path
import os
import subprocess
from copy import deepcopy
from time import sleep

from zeus.job import Job
from zeus.analyze import HistoryEntry


def parse_args() -> argparse.Namespace:
    """Parse commandline arguments."""
    parser = argparse.ArgumentParser()

    # This random seed is used for
    # 1. Multi-Armed Bandit inside PruningGTSBatchSizeOptimizer, and
    # 2. Providing random seeds for training.
    # Especially for 2, the random seed given to the nth recurrence job is args.seed + n.
    parser.add_argument("--seed", type=int, default=1, help="Random seed")

    # Default batch size and learning rate.
    # The first recurrence uses these parameters, and it must reach the target metric.
    parser.add_argument("--b_0", type=int, default=1024, help="Default batch size")
    parser.add_argument("--lr_0", type=float, default=0.001, help="Default learning rate")
    parser.add_argument("--pl_0", type=int, default=175000, help="Default power limit (mW)")

    # The range of batch sizes to consider. The example script generates a list of power-of-two
    # batch sizes, but batch sizes need not be power-of-two for Zeus.
    parser.add_argument(
        "--b_min", type=int, default=8, help="Smallest batch size to consider"
    )
    parser.add_argument(
        "--b_max", type=int, default=4096, help="Largest batch size to consider"
    )

    # The range of power limits to consider. Generates a list of power limits of every 25 W
    parser.add_argument(
        "--pl_min", type=int, default=100000, help="Smallest power limit to consider, in mW"
    )
    parser.add_argument(
        "--pl_max", type=int, default=175000, help="Largest power limit to consider, in mW"
    )

    # The total number of recurrences.
    parser.add_argument(
        "--num_recurrence", type=int, default=100, help="Total number of recurrences"
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

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Run Profiler on CIFAR100."""

    # Definition of the CIFAR100 job.
    # The `Job` class encloses all information needed to run training. The `command` parameter is
    # a command template. Curly-braced parameters are recognized by Zeus and automatically filled.
    batch_size = args.b_0
    power_limit = args.pl_0

    job = Job(
        dataset="cifar100",
        network="shufflenetv2",
        optimizer="adam",
        target_metric=args.target_metric,
        max_epochs=args.max_epochs,
        default_bs=args.b_0,
        default_lr=args.lr_0,  # where does this end up getting used? in the lr scaler?
        workdir="/workspace/zeus/examples/cifar100",
        # fmt: off
        command=[
            "python",
            "train.py",
            "--profile",
            "--arch", "shufflenetv2",
            "--batch_size", "{batch_size}",
            "--epochs", "{epochs}",
            "--seed", "{seed}",
            "--power_limit", "{power_limit}"
        ],
        # fmt: on
    )
    
    # Run the job! Code hacked together from master.py, run_job()

    # Generate job command
    command = job.gen_command(batch_size, args.lr_0, power_limit, args.seed, 0)

    # Set environment variables
    job_id = f"cifar100+shufflenetv2+bs{batch_size}+pl{power_limit}"
    logdir = "/workspace/examples/cifar100"
    zeus_env = dict(
        ZEUS_LOG_PREFIX=str(job_id),
        ZEUS_TARGET_METRIC=str(job.target_metric),
        ZEUS_MONITOR_PATH="/workspace/zeus/zeus_monitor/zeus_monitor",
        ZEUS_MONITOR_SLEEP_MS="100"
    )
    env = deepcopy(os.environ)
    env.update(zeus_env)

    # Training script output captured by the master.
    job_output = f"{job_id}.train.log"

    # Training stats (energy, time, reached, end_epoch) written by ZeusDataLoader.
    # This file being found means that the training job is done.
    train_json = Path(f"{logdir}/{job_id}.train.json")

    # Job history list to return.
    history: list[HistoryEntry] = []

    # Save job history to this file, continuously.
    history_file = f"{logdir}/history+{job_id}.py"

    # Reporting
    print(f"[run job] Launching job with BS {batch_size}:")
    print(f"[run job] {zeus_env=}")
    if job.workdir is not None:
        print(f"[run job] cwd={job.workdir}")
    print(f"[run job] {command=}")
    # print(f"[run job] {cost_ub=}")
    print(f"[run job] Job output logged to '{job_output}'")

    # Run the job.
    with open(job_output, "w") as f:
        # Launch subprocess.
        # stderr is redirected to stdout, and stdout to the job_output file.
        proc = subprocess.Popen(
            command,
            cwd=job.workdir,
            stderr=subprocess.STDOUT,
            stdout=f,
        )

        # Check if training is done.
        with open(job_output, "r") as jobf:
            while proc.poll() is None:
                print(jobf.read(), end="")
                sleep(1.0)

            # Print out the rest of the script output.
            f.flush()
            print(jobf.read())

            # Report exitcode.
            exitcode = proc.poll()
            print(f"[run job] Job terminated with exit code {exitcode}.")

        # `train_json` must exist at this point.
        if not train_json.exists():
            raise RuntimeError(f"{train_json} does not exist.")

    # Read `train_json` for the training stats.
    with open(train_json, "r") as f:
        stats = json.load(f)
        print(f"[run job] {stats=}")
    # train_json.close()

    # Casting
    if not isinstance(stats["reached"], bool):
        stats["reached"] = stats["reached"].lower() == "true"

    # Record history for visualization.
    history.append(HistoryEntry(args.b_0, args.pl_0, float(stats["energy"]), stats["reached"], float(stats["time"])))
    with open(history_file, "w") as f:
        # Intended use:
        #
        # ```python
        # from zeus.analyze import HistoryEntry
        # history = eval(open(history_file).read())
        # ```
        f.write(pprint.pformat(history) + "\n")

    if stats["reached"] == True:
        print("Reached target metric")
    # history_file.close()

    # return float(stats["energy"]), float(stats["time"]), stats["reached"]


    # Generate a list of batch sizes with only power-of-two values.
    # batch_sizes = [args.b_min]
    # while (bs := batch_sizes[-1] * 2) <= args.b_max:
    #     batch_sizes.append(bs)

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
    # master_logdir = master.build_logdir(
    #     job=job,
    #     num_recurrence=args.num_recurrence,
    #     eta_knob=args.eta_knob,
    #     beta_knob=args.beta_knob,
    #     exist_ok=False,  # Should err if this directory exists.
    # )

    # Overwrite the stdout file descriptor with an instance of `FileAndConsole`, so that
    # all calls to `print` will write to both the console and the master log file.
    # sys.stdout = FileAndConsole(Path(master_logdir) / "master.log")



    # Run Zeus!
    # master.run(
    #     job=job,
    #     num_recurrence=args.num_recurrence,
    #     batch_sizes=batch_sizes,
    #     beta_knob=args.beta_knob,
    #     eta_knob=args.eta_knob,
    # )


if __name__ == "__main__":
    main(parse_args())
