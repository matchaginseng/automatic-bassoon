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

"""Defines the Profiler class. The core of our project."""

from __future__ import annotations

import json
import logging
import os
import pprint
import subprocess
import random
from copy import deepcopy
from shufflenetv2 import shufflenetv2
from pathlib import Path
from time import localtime, sleep, strftime, monotonic
import copy

import numpy as np
import pynvml

from zeus2.analyze import HistoryEntry
from zeus2.job import Job
from zeus2.metric import epoch_cost
from zeus.util import get_env

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from zeus2.profile_dataloader import ProfileDataLoader

# JIT profiling states
NOT_PROFILING = "NOT_PROFILING"
WARMING_UP = "WARMING_UP"
PROFILING = "PROFILING"

# Config logging
LOG = logging.Logger(__name__)
LOG.setLevel(logging.INFO)
LOG_HANDLER = logging.StreamHandler()
LOG_FORMATTER = logging.Formatter("%(asctime)s %(message)s")
LOG_HANDLER.setFormatter(LOG_FORMATTER)
LOG.addHandler(LOG_HANDLER)


class Profiler:
    """Profiles and optimizes GPU power limit.
    """
    def __init__(
        self,
        log_base: str,
        monitor_path: str,
        seed: int = 123456,
        observer_mode: bool = False,
        profile_warmup_iters: int = 10,
        profile_measure_iters: int = 40) -> None:
        """Initialize the profiler.

        Args:
            log_base: Absolute path where logs will be stored. A separate directory
                will be created inside, whose name is determined by the job and current time.
            monitor_path: Absolute path to the power monitor binary.
            seed: The random seed. Every invocation of the [`run`][zeus.run.ZeusMaster.run]
                method in this class is deterministic given the random seed, because the
                internal RNG states are deepcopied before servicing jobs.
            observer_mode: When Observer Mode is on, the maximum power limit is
                always used instead of the optimal power limit. However, internal time and
                energy accounting will be done as if the cost-optimal power limit is used.
            profile_warmup_iters: Number of iterations to warm up on a specific power limit.
                This is passed to the [`ZeusDataLoader`][zeus.run.ZeusDataLoader]. # TODO: FIX THIS NAME
            profile_measure_iters: Number of iterations to measure on a specific power limit.
                This is passed to the [`ZeusDataLoader`][zeus.run.ZeusDataLoader]. # TODO: FIX THIS NAME
        """
        # Check if monitor_path is absolute.
        # This is needed since we may change the cwd based on the job's workdir.
        if not Path(monitor_path).is_absolute():
            raise ValueError("monitor_path must be specified as an absolute path.")

        # Log base directory.
        # Needs to be absolute because the training job script may have a different
        # current working directory (when job.workdir is not None).
        if not Path(log_base).is_absolute():
            raise ValueError("log_base must be specified as an absolute path.")
        os.makedirs(log_base, exist_ok=True)
        self.log_base = log_base

        # Save arguments.
        self.seed = seed
        self.monitor_path = monitor_path
        self.observer_mode = observer_mode
        self.profile_warmup_iters = profile_warmup_iters
        self.profile_measure_iters = profile_measure_iters
        

        # Query the max power limit of the GPU.
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.min_pl, self.max_pl = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)  # unit: mW
        self.min_pl //= 1000  # unit: W
        self.max_pl //= 1000 # unit: W
        print(
            f"[Zeus Master] Max power limit of {pynvml.nvmlDeviceGetName(handle)}: {self.max_pl}W"
        )
        pynvml.nvmlShutdown()

        self.power_limits = list(range(self.max_pl, self.min_pl - 25, -25))
        # self.power_limits = [175000, 150000, 125000, 100000]
        print(f"[Power Profiler] Power limits: {self.power_limits}")

    def build_logdir(
        self,
        job: Job,
        eta_knob: float,
        beta_knob: float,
        exist_ok: bool = True,
    ) -> str:
        r"""Build the `ZEUS_LOG_DIR` string and create the directory.

        Args:
            job: Job to run.
            eta_knob: $\eta$ used in the cost metric.
                $\textrm{cost} = \eta \cdot \textrm{ETA} + (1 - \eta) \cdot \textrm{MaxPower} \cdot \textrm{TTA}$
            beta_knob: `beta_knob * min_cost` is the early stopping cost threshold.
                Set to `np.inf` to disable early stopping.
            exist_ok: Passed to `os.makedirs`. If `False`, will err if the directory
                already exists.
        """
        print(job)
        now = strftime("%Y%m%d%H%M%s", localtime())
        logdir = (
            job.to_logdir() + f"+eta{eta_knob}+beta{beta_knob}+{now}"
        )
        logdir = f"{self.log_base}/{logdir}"
        os.makedirs(logdir, exist_ok=exist_ok)
        return logdir

    # def run_job(
    #     self,
    #     job: Job,
    #     batch_size: int,
    #     learning_rate: float,
    #     dropout_rate: float,
    #     power_limit: int,
    #     seed: int,
    #     logdir: str,
    #     eta_knob: float,
    #     cost_ub: float,
    # ) -> tuple[float, float, bool]:
    #     r"""Launch the training job.

    #     Args:
    #         job: The job to run.
    #         batch_size: The batch size to use.
    #         learning_rate: The learning rate to use, scaled based on `batch_size`.
    #         dropout_rate: The dropout rate to use
    #         seed: The random seed to use for training.
    #         logdir: Directory to store log files in.
    #         eta_knob: $\eta$ used in the cost metric.
    #             $\textrm{cost} = \eta \cdot \textrm{ETA} + (1 - \eta) \cdot \textrm{MaxPower} \cdot \textrm{TTA}$
    #         cost_ub: Cost upper bound. The job is terminated when the next epoch is going
    #             to exceed the cost upper bound.

    #     Returns:
    #         A tuple of energy consumption, time consumption, and whether the job reached the target metric.
    #     """
    #     # Generate job command
    #     command = job.gen_command(batch_size, learning_rate, power_limit, dropout_rate, seed)

    #     # Set environment variables
    #     # TODO: incorporate dropout rate
    #     job_id = f"bs{batch_size}+lr{learning_rate:.5f}+pl{power_limit}"
    #     zeus_env = dict(
    #         ZEUS_LOG_DIR=logdir,
    #         ZEUS_JOB_ID=job_id,
    #         ZEUS_COST_THRESH="inf" if cost_ub == np.inf else str(cost_ub),
    #         ZEUS_BATCH_SIZE=str(batch_size),
    #         ZEUS_LEARNING_RATE=str(learning_rate),
    #         ZEUS_ETA_KNOB=str(eta_knob),
    #         ZEUS_POWER_LIMIT=str(power_limit),
    #         # ZEUS_TARGET_METRIC=str(job.target_metric),
    #         ZEUS_MONITOR_PATH=self.monitor_path,
    #         ZEUS_PROFILE_PARAMS=f"{self.profile_warmup_iters},{self.profile_measure_iters}",
    #         ZEUS_LOG_PREFIX="/workspace/zeus_logs",
    #         # ZEUS_USE_OPTIMAL_PL=str(not self.observer_mode),
    #     )
    #     env = deepcopy(os.environ)
    #     env.update(zeus_env)

    #     # Training script output captured by the master.
    #     job_output = f"{logdir}/{job_id}.train.log"

    #     # Training stats (energy, time, reached, end_epoch) written by ZeusDataLoader.
    #     # This file being found means that the training job is done.
    #     # train_json = Path(f"{logdir}/{job_id}.train.json")

    #     # File that got written to in the profiling dataloader.
    #     history_json = Path(f"{logdir}/{job_id}.history_all.py")

    #     # Reporting
    #     print(f"[run job] Launching job with BS {batch_size}: and LR: {learning_rate} and PL: {power_limit} and DR: {dropout_rate}")
    #     print(f"[run job] {zeus_env=}")
    #     if job.workdir is not None:
    #         print(f"[run job] cwd={job.workdir}")
    #     print(f"[run job] {command=}")
    #     print(f"[run job] {cost_ub=}")
    #     print(f"[run job] Job output logged to '{job_output}'")

    #     # Run the job.
    #     with open(job_output, "w") as f:
    #         # Launch subprocess.
    #         # stderr is redirected to stdout, and stdout to the job_output file.
    #         proc = subprocess.Popen(
    #             command,
    #             cwd=job.workdir,
    #             stderr=subprocess.STDOUT,
    #             stdout=f,
    #         )

    #         # Check if training is done.
    #         with open(job_output, "r") as jobf:
    #             while proc.poll() is None:
    #                 print(jobf.read(), end="")
    #                 sleep(1.0)

    #             # Print out the rest of the script output.
    #             f.flush()
    #             print(jobf.read())

    #             # Report exitcode.
    #             exitcode = proc.poll()
    #             print(f"[run job] Job terminated with exit code {exitcode}.")

    #         # `history_json` must exist at this point.
    #         if not history_json.exists():
    #             raise RuntimeError(f"{history_json} does not exist.")

    #     # TODO: extract the cost from the file that got written out in the dataloader <- did i do it right? <- yas
    #     with open(history_json, "r") as histf:
    #         stats = json.load(histf)
    #         print(f"[run job] {stats=}")

    #     # Read `train_json` for the training stats.
    #     # with open(train_json, "r") as f:
    #     #     stats = json.load(f)
    #     #     print(f"[run job] {stats=}")

    #     # Casting
    #     # if not isinstance(stats["reached"], bool):
    #     #     stats["reached"] = stats["reached"].lower() == "true"

    #     return float(stats["energy"]), float(stats["time"]), float(stats["accuracy"]), float(stats["total_cost"])

    #     # return float(stats["energy"]), float(stats["time"]), float(cost["cost"]), stats["reached"]
    #     # not sure about how exactly to index into cost

    def _save_train_results(
        self, energy: float, time_: float, cost: float, reached: bool
    ) -> None:
        """Write the job training results to `train_json`."""
        # Sanity check.
        # Only load power results at master process.
        assert self.rank == 0

        train_result = dict(
            energy=energy,
            time=time_,
            cost=cost,  # Not used. Just for reference.
            num_epochs=self.epoch_num,  # Not used. Just for reference.
            reached=reached,
        )
        with open(self.train_json, "w") as f:
            json.dump(train_result, f)
        with open(self.train_json, "r") as f:
            self._log("Training done.")
            self._log(f"Saved {self.train_json}: {f.read()}")

    # def profile(
    #     self, 
    #     job: Job,
    #     eta_knob: float,
    #     beta_knob: float,
    #     batch_sizes: list,
    #     dropout_rates: list,
    #     learning_rates: list) -> tuple[int, float, int]:
    #     """Runs a job. Returns a tuple (bs, lr, pl) that minimizes our epoch cost

    #     Args:
    #         job: The job to run.
    #         batch_sizes: List of feasible batch sizes.
    #         beta_knob: `beta_knob * min_eta` is the early stopping cost threshold.
    #             Set to `np.inf` to disable early stopping.
    #         eta_knob: $\eta$ used in the cost metric.
    #             $\textrm{cost} = \eta \cdot \textrm{ETA} + (1 - \eta) \cdot \textrm{MaxPower} \cdot \textrm{TTA}$
    #     """
        
    #     if eta_knob < 0.0 or eta_knob > 1.0:
    #         raise ValueError("eta_knob must be in [0.0, 1.0].")

    #     # Copy all internal state so that simulation does not modify any
    #     # internal state and is deterministic w.r.t. the random seed.
    #     seed = self.seed

    #     # ZEUS_LOG_DIR: Where all the logs and files are stored for this run.
    #     logdir = self.build_logdir(job, eta_knob, beta_knob)

    #     # Job history list to return.
    #     history: list[HistoryEntry] = []

    #     # Save job history to this file, continuously.
    #     history_file = f"{logdir}/history.py"

    #     # beta_knob * min_cost is the early stopping cost threshold.
    #     min_cost = np.inf

    #     # list of (bs, lr) batch size tuples to try
    #     bs_lr_dr = []

    #     # dict of (bs, lr) opt power limits
    #     opt_pl = {}

    #     # batch_sizes is a list of all batch sizes the user wants us to try
    #     for bs in batch_sizes:
    #         # for lr in [job.scale_lr(bs * factor) for factor in [0.8, 0.9, 1, 1.1, 1.2]] :
    #         for lr in [job.scale_lr(bs * factor) for factor in learning_rates]:
    #             for dr in dropout_rates:
    #                 bs_lr_dr.append((bs, lr, dr))
    #                 opt_pl[(bs, lr, dr)] = 0 # initialize

    #     profile_time = 0.

    #     #
    #     # 2-lvl optimization
    #     for i in range(1, len(bs_lr_dr) + 1):
    #         bs, lr, dr = bs_lr_dr[i - 1]
    #         print(f"\n[Power Profiler] with batch size {bs} and learning rate {lr} and dropout rate {dr}")

    #         min_cost = float("inf")

    #         # initialize best pl for this combo
    #         best_pl = -1

    #         for pl in self.power_limits:
    #             # cost_acc = 0.0
            
    #             # Launch the job.
    #             # Early stops based on cost_ub.
    #             job_start_time = monotonic()

    #             # we don't want to run job here we want to do the profiling
    #             energy, time, accuracy, total_cost = self.run_job(
    #                 job=job,
    #                 batch_size=bs,
    #                 learning_rate=lr,
    #                 dropout_rate=dr,
    #                 power_limit=pl,
    #                 seed=seed,
    #                 logdir=logdir,
    #                 eta_knob=eta_knob,
    #                 cost_ub=beta_knob * min_cost,
    #             )
    #             job_end_time = monotonic()

    #             profile_time += job_end_time - job_start_time
    #             # The random seed will be unique for each run, but still jobs will be
    #             # deterministic w.r.t. each call to `run`.
    #             # seed += 1

    #             # Compute the cost of this try.
    #             # num_gpus = torch.cuda.device_count()

    #             # cost = epoch_cost(energy, time, eta_knob, self.max_pl * num_gpus)
    #             # print(f"[Zeus Master] {cost=}")

    #             if total_cost < min_cost:
    #                 min_cost = total_cost
    #                 best_pl = pl 
    #                 opt_pl[(bs, lr, dr)] = best_pl

    #             # Record history for visualization. TODO: change variables. the functions processing this may be total nonsense RN
    #             history.append(HistoryEntry(bs, pl, energy, time, accuracy, total_cost))
    #             with open(history_file, "w") as f:
    #                 # Intended use:
    #                 #
    #                 # ```python
    #                 # from zeus.analyze import HistoryEntry
    #                 # history = eval(open(history_file).read())
    #                 # ```
    #                 f.write(pprint.pformat(history) + "\n")


    #     print(f"[Power Profiler]\n{history}")

    #     profiler_info = dict(
    #         total_time=profile_time,
    #         opt_bs=opt_bs,
    #         opt_lr=opt_lr,
    #         opt_pl=opt_pl[((opt_bs, opt_lr))]
    #     )
        
    #     with open(f"{logdir}/profiler_info.json", "w") as f:
    #         json.dump(profiler_info, f)

    #     # find optimal setting to return: get argmin
    #     opt_bs, opt_lr, opt_dr = min(opt_pl, key=opt_pl.get)

    #     # return the optimal setting
    #     return (opt_bs, opt_lr, opt_dr, opt_pl[(opt_bs, opt_lr, opt_dr)])

    def set_seed(seed: int) -> None:
        """Set random seed for reproducible results."""
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def train_with_profiling(
        self, 
        job: Job,
        eta_knob: float,
        beta_knob: float,
        train_dataset,
        val_dataset,
        target_acc: float,
        n_epochs: int,
        acc_thresholds: list[float], # list of accuracy thresholds that trigger parameter profiling
        batch_sizes: list[int],
        learning_rates: list[float],
        dropout_rates: list[float],
        seed=None) -> tuple[int, float, int]:
        
        if seed is not None:
            self.set_seed(seed)

        logdir = self.build_logdir(job, eta_knob, beta_knob)
        os.environ["ZEUS_MONITOR_PATH"] = self.monitor_path
        os.environ["ZEUS_LOG_PREFIX"] = "/workspace/zeus_logs"
        os.environ["ZEUS_LOG_DIR"] = logdir

        print(f"[Training Loop] Testing batch sizes: {batch_sizes}")
        print(f"[Training Loop] Testing power limits: {self.power_limits}")
        print(f"[Training Loop] Testing learning rates: {learning_rates}")
        print(f"[Training Loop] Testing dropout rates: {dropout_rates}")
        print(f"[Training Loop] Reprofiling at accuracy thresholds {acc_thresholds}")

        model = shufflenetv2(0.0)
        # Send model to CUDA.
        model = model.cuda()

        # Prepare loss function and optimizer.
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(params=model.parameters(), lr=0.01)
        # optimizer = optim.Adadelta(model.parameters())

        epoch_iter = range(n_epochs)
        threshold_acc = 0.0
        curr_acc = 0.0
        epoch = 0
        # main training loop
        while epoch < epoch_iter:
            if curr_acc >= threshold_acc:
                print(f"[Training Loop] Model's accuracy {curr_acc} surpasses threshold {threshold_acc}! Reprofiling...")
                bs_lr_dr = []
                opt_pl = {}
                for bs in batch_sizes:
                    for lr in learning_rates:
                        for dr in dropout_rates:
                            bs_lr_dr.append((bs, lr, dr))
                            opt_pl[(bs, lr, dr)] = 0

                profile_start_time = monotonic()
                for i in range(1, len(bs_lr_dr) + 1):
                    bs, lr, dr = bs_lr_dr[i - 1]
                    # initialize best pl for this combo
                    best_pl = -1
                    min_cost = float("inf")
                    for pl in self.power_limits:
                        print(f"[Training Loop] Profiling with batch size {bs} learning rate {lr} dropout rate {dr} power limit {pl}")
                        # set the batch size, learning rate, dropout rate, and power limit of train and val dataloaders
                        train_loader = ProfileDataLoader(
                            train_dataset,
                            batch_size=bs,
                            learning_rate=lr,
                            dropout_rate=dr,
                            power_limit=pl,
                            split="train",
                            profile=True,
                            shuffle=True,
                            warmup_iters=self.profile_warmup_iters,
                            measure_iters=self.profile_measure_iters,
                            num_workers=4, # TODO: this is the default value but maybe pass in as an arg
                        )
                        val_loader = ProfileDataLoader(
                            val_dataset,
                            batch_size=bs,
                            learning_rate=lr,
                            dropout_rate=dr,
                            power_limit=pl,
                            split="eval",
                            profile=False,
                            shuffle=False,
                            num_workers=4,
                        )

                        # deepcopy the model for profiling
                        model_copy = copy.deepcopy(model)
                        model_copy = model_copy.cuda()
                        optimizer_copy = optim.Adam(model_copy.parameters(), lr=lr)

                        self.train(train_loader, model_copy, criterion, optimizer_copy, epoch, bs, True)
                        acc = self.validate(val_loader, model_copy, criterion, epoch, bs)
                        cost = train_loader.calculate_cost(acc, threshold_acc)
                        
                        if cost < min_cost:
                            min_cost = cost
                            best_pl = pl 
                            opt_pl[(bs, lr, dr)] = best_pl

                profile_end_time = monotonic()

                # find optimal setting to return: get argmin
                opt_bs, opt_lr, opt_dr = min(opt_pl, key=opt_pl.get)
                opt_pl = opt_pl[(opt_bs, opt_lr, opt_dr)]
                print(f"[Training Loop] The optimal parameters are lr: {opt_lr} dr: {opt_dr} pl: {opt_pl}")

                profiler_info = dict(
                    threshold=threshold_acc,
                    total_time=profile_end_time - profile_start_time,
                    opt_bs=opt_bs,
                    opt_lr=opt_lr,
                    opt_dr=opt_dr,
                    opt_pl=opt_pl
                )

                with open(f"{logdir}/profiler_info.json", "a") as f:
                    if epoch == 0:
                        f.write("[")
                    json.dump(profiler_info, f)
                    f.write(",")
                
                # set optimal hyperparameters for train and val dataloaders
                train_loader = ProfileDataLoader(
                    train_dataset,
                    batch_size=opt_bs,
                    learning_rate=opt_lr,
                    dropout_rate=opt_dr,
                    power_limit=opt_pl,
                    split="train",
                    profile=False,
                    shuffle=True,
                    num_workers=4,
                )
                val_loader = ProfileDataLoader(
                    val_dataset,
                    batch_size=opt_bs,
                    learning_rate=opt_lr,
                    dropout_rate=opt_dr,
                    power_limit=opt_pl,
                    split="eval",
                    profile=False,
                    shuffle=False,
                    num_workers=4,
                )

                # update the epoch of the trainloader for bookkeeping purposes
                train_loader.epoch = epoch
                # close out the list in the history_all file
                history_all = f"{self.logdir}/threshold{threshold_acc}.history_all.py"
                with open(history_all, "a") as f:
                    f.write("]")

                # set threshold_acc to the next threshold accuracy (or 1 if there are no more)
                threshold_acc = acc_thresholds.pop() if acc_thresholds else 1.0

                # change the learning rate (https://stackoverflow.com/questions/48324152/pytorch-how-to-change-the-learning-rate-of-an-optimizer-at-any-given-moment-no)
                for g in optimizer.param_groups:
                    g['lr'] = opt_lr
                
                # change the dropout rate (https://stackoverflow.com/questions/65813108/changing-dropout-value-during-training)
                model.dropout.p = opt_dr
            else:
                self.train(train_loader, model, criterion, optimizer, epoch, opt_bs)
                curr_acc = self.validate(val_loader, model, criterion, epoch, opt_bs)
                epoch += 1
                if curr_acc >= target_acc:
                    print(f"[Training Loop] Target accuracy {target_acc} reached!")
                    break
            
        with open(f"{logdir}/profiler_info.json", "a") as f:
            f.write("]")
        
        print("[Training Loop] Training done")
        
    def train(self, train_loader, model, criterion, optimizer, epoch, batch_size, profile=False):
        """Train the model for one epoch."""
        model.train()
        num_samples = len(train_loader) * batch_size

        for batch_index, (images, labels) in enumerate(train_loader):
            labels = labels.cuda()
            images = images.cuda()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if profile:
                print(f"Profiling... ")
            print(
                f"Training Epoch: {epoch} [{(batch_index + 1) * batch_size}/{num_samples}]"
                f"\tLoss: {loss.item():0.4f}"
            )

    @torch.no_grad()
    def validate(self, val_loader, model, criterion, epoch, batch_size):
        """Evaluate the model on the validation set."""
        model.eval()

        test_loss = 0.0
        correct = 0
        num_samples = len(val_loader) * batch_size

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
