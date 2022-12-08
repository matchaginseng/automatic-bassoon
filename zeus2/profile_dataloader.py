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

"""Defines the ProfileDataLoader class."""

from __future__ import annotations

import atexit
import pynvml
import os
import subprocess
import time
from typing import Literal
from functools import cached_property

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

from zeus.util import get_env
from zeus import analyze

# JIT profiling states
NOT_PROFILING = "NOT_PROFILING"
WARMING_UP = "WARMING_UP"
PROFILING = "PROFILING"


class ProfileDataLoader(DataLoader):
    """A DataLoader class that profiles power and time.

    `ProfileDataLoader` acts just like an ordinary
    [`DataLoader`][torch.utils.data.DataLoader] while profiling power
    consumption and epoch latency under the hood. Power profiling is done by
    spawning the Zeus power monitor as a subprocess. The latency of each epoch
    will be printed out to stdout.

    `ProfileDataLoader` interfaces with the outside world with environment variables.

    - `ZEUS_LOG_PREFIX`      : Prefix for power and time log files.
                               Power log: `f"{log_prefix}+gpu{index}.power.csv"`
                               Time log : `f"{log_prefix}.time.csv"`
    - `ZEUS_MONITOR_PATH`    : Path to the Zeus power monitor.
                               (Default: `"/workspace/zeus/zeus_monitor/zeus_monitor"`)
    - `ZEUS_MONITOR_SLEEP_MS`: How many milliseconds to sleep after measuring power.
                               This is passed to the monitor. (Default: `"100"`)

    `ProfileDataLoader` supports training on only a subset of the dataset and
    scaling time measurements as if trained on the entire dataset.

    `ProfileDataLoader` will assume that training is happening on all GPUs visible
    and spawn one Zeus power monitor process for each GPU. If this is not what you
    want, set `CUDA_VISIBLE_DEVICES` or spawn a Docker container that only mounts
    the GPUs that you would like to use.
    """

    # Power monitor processes
    monitor: list[subprocess.Popen] | None = None

    def __init__(
        self,
        *args,
        profile: bool = False,
        batch_size: int = 128,
        learning_rate: float = 0.01,
        power_limit: int = 100,
        dropout_rate: float = 1.0,
        warmup_iters: int = 10,
        measure_iters: int = 40,
        split: Literal["train", "eval"],
        subset_proportion: float = 1.0,
        # eat_batch_size: bool = False,
        only_scale_time: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the ProfileDataLoader.

        Args:
            batch_size: Batch size to use.
            split: Dataset split. Used when printing out epoch latency.
            subset_proportion: Should be between 0.0 and 1.0. When specified,
                only that proportion of the dataset will be used and the dataloader will
                stop early. Then, the measured epoch latency will be scaled as if the
                whole datset was used.
            only_scale_time: If True, the whole dataset will be used for training, but
                the measured epoch latency will still be scaled based on the value of
                `subset_proportion`. This is useful when you already manually scaled down
                the size of an existing dataset, but still want to simulate training the
                original large dataset.
            eat_batch_size: If True, does not pass the `batch_size` argument to the
                constructor of DataLoader. You won't usually need this.
            
        """
        # Assumes one epoch per invocation of __iter__.
        self.epoch = 0
        self.profile = profile
        if split not in ["train", "eval"]:
            raise ValueError("split should be either 'train' or 'eval'.")
        self.split = split
        self.scaling_factor = 1.0
        self.start1 = None
        self.start2 = None
        self.warmup_iter = warmup_iters
        self.profile_iter = measure_iters
        self.prof_state = NOT_PROFILING
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate

        # Call the constructor of DataLoader.
        super().__init__(*args, batch_size=batch_size, **kwargs)
        print("ProfileDataLoader constructor: called constructor of DataLoader!")

        # Retrieve environment variables needed.
        self.logdir = get_env("ZEUS_LOG_DIR", str, default="zeus_log")
        self.monitor_path = get_env("ZEUS_MONITOR_PATH", str)
        self.monitor_sleep_ms = get_env("ZEUS_MONITOR_SLEEP_MS", int, default=100)
        self.log_prefix = get_env("ZEUS_LOG_PREFIX", str)
        self.eta_knob = get_env("ZEUS_ETA_KNOB", float, default=0.5)
        # self.target_metric = get_env("ZEUS_TARGET_METRIC", float)
        
        self.power_limit = power_limit * 1000 # in mW

        # Train-time power profiling result. Maps power limit to avg_power & throughput.
        self.train_power_result: float = 0.
        self.train_tput_result: float = 0.
        self.num_samples = len(self)
        # self.num_samples = len(self)//self.batch_size #TODO: change this to not be hardcoded but i got issues doing len(self)??


        # Eval-time power profiling result. Maps power limit to avg_power & throughput.
        self.eval_power_result: float = 0.
        self.eval_tput_result: float = 0.

        # single GPU setting
        self.rank = 0

        # Check if the Zeus power monitor is executable.
        if not os.access(self.monitor_path, os.X_OK):
            raise RuntimeError(f"'{self.monitor_path}' is not executable")

        # Create time.csv and write header.
        if self.split == "train":
            ProfileDataLoader.time_file = open(self.log_prefix + ".time.csv", "w")
            self.time_file.write("epoch,split,time\n")
            self.time_file.flush()
        
        # self.power_limit is in mW...
        # job_id = f"bs{batch_size}+lr{learning_rate:.5f}+pl{self.power_limit // 1000}+thresh{self.threshold}"

        # Initialize NVML and get GPU handle or each GPU at the master process.
        self.gpu_handles = []
        self.world_size = 1
        pynvml.nvmlInit()
        for index in range(self.world_size):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            # Set persistent mode.
            pynvml.nvmlDeviceSetPersistenceMode(handle, pynvml.NVML_FEATURE_ENABLED)
            self.gpu_handles.append(handle)

        # Query NVML for the possible power limit range. Unit is mW.
        # Default power limit is the max.
        # Query NVML for the possible power limit range. Unit is mW.
        _, self.max_pl = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(
            self.gpu_handles[0]
        )
        
        # If the number of iterations in one epoch (`num_samples`) is smaller than or equal
        # to one profile window (`warmup_iters + profile_iters`), we will not be able to
        # profile for any power limit. So, we scale the profile window to fit in one epoch.
        # We also avoid using the last batch of one epoch, because when `drop_last == True`,
        # the last batch will be smaller. This usually happens with large batch size on
        # small datasets, eg. CIFAR100.
        if self._is_train and self.warmup_iter + self.profile_iter >= self.num_samples:
            print(
                f"[Profile DataLoader] The profile window takes {self.warmup_iter + self.profile_iter}"
                f" iterations ({self.warmup_iter} for warmup + {self.profile_iter}"
                f" for profile) and exceeds the number of iterations ({self.num_samples})"
                f" in one epoch. Scaling the profile window to fit in one epoch..."
            )
            scaling_factor = (self.num_samples - 1) / (
                self.warmup_iter + self.profile_iter
            )
            self.warmup_iter = int(self.warmup_iter * scaling_factor)
            self.profile_iter = int(self.profile_iter * scaling_factor)
            if self.warmup_iter == 0 or self.profile_iter == 0:
                raise RuntimeError(
                    f"Number of iterations in one epoch is {self.num_samples} and"
                    " is too small for applying the scaling. Please consider using"
                    " a smaller batch size. If you are running `run_zeus.py`, please"
                    " pass a smaller value to `--b_max`."
                )
            print(
                f"[Profile DataLoader] Scaling done! New profile window takes {self.warmup_iter + self.profile_iter}"
                f" iterations ({self.warmup_iter} for warmup + {self.profile_iter} for profile)."
            )

        # Slice out subset of dataset if subset_proportion is given.
        dataset = kwargs["dataset"] if "dataset" in kwargs else args[0]
        if subset_proportion > 1.0 or subset_proportion <= 0.0:
            raise ValueError("subset_proportion should be > 0.0 and <= 1.0.")
        if subset_proportion < 1.0:
            subset_indices = list(range(0, len(dataset), round(1 / subset_proportion)))  # type: ignore
            # See note in __next__ for more about scaling.
            self.scaling_factor = len(dataset) / (len(subset_indices) - batch_size)
            if not only_scale_time:
                subset = Subset(dataset, subset_indices)
                if "dataset" in kwargs:
                    kwargs["dataset"] = subset
                else:
                    args = (subset, *args[1:])

    def _power_log_path(self, rank: int) -> str:
        """Build the path for the power monitor log file at the GPU with rank."""
        return self.log_prefix + f"+gpu{rank}.power.csv" 


    @cached_property
    def _is_train(self) -> bool:
        """Return whether this dataloader is for training."""
        return self.split == "train"

    def __iter__(self):
        """Wrap the original `__iter__`, but with power profiling."""
        # On the first epoch, start the Zeus power monitors for each GPU.
        if ProfileDataLoader.monitor is None:
            if (count := torch.cuda.device_count()) == 1 or dist.get_rank() == 0:
                ProfileDataLoader.monitor = []
                for index in range(count):
                    monitor_cmd = [
                        self.monitor_path,
                        self.log_prefix + f"+gpu{index}.power.csv",  # Power log file
                        "0",  # Duration
                        str(self.monitor_sleep_ms),  # Monitor sleep time (ms)
                        str(index),  # GPU_id
                    ]
                    print(f"Launching Zeus monitor {index}...")
                    ProfileDataLoader.monitor.append(
                        subprocess.Popen(
                            monitor_cmd,
                            stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                        )
                    )
                atexit.register(kill_monitor)
        # pylint: disable=attribute-defined-outside-init
        self.iter = super().__iter__()
        self.epoch += 1
        self.start1 = None
        self.start2 = None
        # Tracks the number of iterations processed in this epoch
        self.sample_num = 0
        # Start epoch timer.
        self.epoch_start_time = time.monotonic()
        return self

    def __next__(self):
        """Wrap the original `__next__`, but with power profiling."""


        # try:
            # Special treatment for the first batch.
            # Data loading takes significantly more time for the first batch. Thus, if we
            # simply measure the first ~ last batch latency of the subset of the dataset
            # and multiply 1/subset_proportion, we end up overestimating time_per_epoch.
            # Thus, we isolate the processing time of the first batch (start2 - start1),
            # scale up second ~ last batch latency with the adjusted scaling factor, and
            # later add the processing time of the first batch.
            #
            # Strange if nest to make the common case number of if statement executions 1.
        if self.start2 is None:
            if self.start1 is None:
                self.start1 = time.time()
            else:
                self.start2 = time.time()
        data = self.iter.__next__()

        if self._is_train and self.profile:
            # We need to start warming up
            # We weren't doing anything. Start warming up if the iterations left in
            # the current epoch can accommodate at least one profile window.
            if (
                self.prof_state == NOT_PROFILING
            ):
                self._start_warmup()
            # We're done warming up. Start the actual profiling window.
            elif (
                self.prof_state == WARMING_UP
                and self.sample_num - self.warmup_start_sample == self.warmup_iter
            ):
                self._start_prof()
            elif (
                self.prof_state == PROFILING
                and self.sample_num - self.warmup_iter == (self.profile_iter)
            ):
                self._end_prof()
                end = time.time()
                if self.start1 and self.start2:
                    scaled_time = (
                        self.scaling_factor * (end - self.start2)
                        + self.start2
                        - self.start1
                    )
                    if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
                        self.time_file.write(f"{self.epoch},{self.split},{scaled_time}\n")
                        self.time_file.flush()
                        print(
                            f"epoch {self.epoch} {self.split} time consumed: {scaled_time:.2f}s"
                        )
    
                raise StopIteration
        
        self.sample_num += 1
        return data

    def _set_gpu_power_limit(self, power_limit: int) -> None:
        """Set the GPU's power limit using NVML.

        `power_limits` must be in mW.

        Only works for single-GPU case right now.

        Args:
            power_limit: Power limit to set.
        """
        # Sanity check.
        # Only set power limit at master process.
        # assert self.rank == 0
        assert len(self.gpu_handles) == self.world_size

        # Set power limit for GPU
        for index in range(self.world_size):
            pynvml.nvmlDeviceSetPowerManagementLimit(
                self.gpu_handles[index], power_limit
            )
            print(f"[GPU_{index}] Set GPU power limit to {power_limit//1000}W.")

            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            # Set persistent mode.
            pynvml.nvmlDeviceSetPersistenceMode(handle, pynvml.NVML_FEATURE_ENABLED)
    

    def calculate_cost(self, acc: float, threshold: float) -> None:
        print(f"[ProfileDataLoader] writing cost to history_all.py")
        history_all = f"{self.logdir}/threshold{threshold}.history_all.py"
        frac_epochs = (self.warmup_iter + self.profile_iter) / self.num_samples

        # TODO: change this cost fn!!
        # total_cost = (frac_epochs / acc) * ((self.eta_knob * self.train_power_result + (1 - self.eta_knob) * self.max_pl / self.train_tput_result))
        total_cost = (self.eta_knob * self.train_power_result + (1 - self.eta_knob) * self.max_pl) * self.time_consumed/acc
        with open(history_all, "a") as f:
            content = f'''
                    {{
                        "profile_threshold":{threshold},
                        "bs": {self.batch_size},
                        "pl": {self.power_limit},
                        "lr": {self.learning_rate},
                        "dr": {self.dropout_rate},
                        "energy": {self.train_power_result},
                        "time": {self.time_consumed},
                        "accuracy": {acc},
                        "total_cost": {total_cost}
                    }}
                    '''
            print(content)
            f.write(content)

        return total_cost

    
    def _start_warmup(self) -> None:
        """Let the GPU run for some time with the power limit to profile."""
        # TODO: Sanity checks.
        assert self._is_train, f"start_warmup: {self._is_train=}"
        # Sanity check that this profile window ends before the end of the current epoch.
        assert (
            self.sample_num + self.warmup_iter + self.profile_iter < self.num_samples
        ), (
            "start_warmup: "
            f"end_of_this_profile_window {self.sample_num + self.warmup_iter + self.profile_iter} "
            f"< end_of_this_epoch {self.num_samples}"
        )

        # Call cudaSynchronize to make sure this is the iteration boundary.
        torch.cuda.synchronize()

        # Change power limit.
        if self.rank == 0:
            # power_limit = self.power_limits[self.prof_pl_index]
            self._set_gpu_power_limit(self.power_limit)

            print(f"Warm-up started with power limit {self.power_limit//1000}W")

        self.warmup_start_sample = self.sample_num

        # Set profiling state.
        self.prof_state = WARMING_UP


    def _start_prof(self) -> None:
        """Start profiling power consumption for the current power limit."""
        # Sanity checks.
        assert self._is_train, f"start_prof: {self._is_train=}"
        # Sanity check that this profile window ends before the end of the current epoch.
        assert self.sample_num + self.profile_iter < self.num_samples, (
            "start_prof: "
            f"end_of_this_profile_window {self.sample_num + self.profile_iter} "
            f"< end_of_this_epoch {self.num_samples}"
        )

        # Start profile timer.
        self.prof_start_time = time.monotonic()

        # Set the sample number when we started profiling.
        self.prof_start_sample = self.sample_num

        # Set profiling state.
        self.prof_state = PROFILING

        # self._log(f"Profile started with power limit {self.current_gpu_pl//1000}W")


    def _end_prof(self) -> None:
        """End profiling power consumption for this power limit.

        Raises:
            ValueError: ValueError raised by sklearn.metrics.auc in analyze.avg_power,
                might due to profile window too small. In this case, user should consider
                increasing profile window.
        """
        # Sanity checks.
        assert self._is_train, f"end_prof: {self._is_train=}"
        # Sanity check that this profile window ends before the end of the current epoch.
        assert self.sample_num < self.num_samples, (
            "end_prof: "
            f"end_of_this_profile_window {self.sample_num} "
            f"< end_of_this_epoch {self.num_samples}"
        )

        # Set profiling state.
        self.prof_state = NOT_PROFILING

        # Call cudaSynchronize to make sure this is the iteration boundary.
        torch.cuda.synchronize()

        # Freeze time.
        now = time.monotonic()

        # Summing up the average power on all GPUs.
        self.sum_avg_power = 0

        # Compute and save average power.
        # The monitor is still running, so we just integrate from the beginning
        # of this profiling window (of course exclude warmup) up to now.
        # The power log file only records for the current epoch,
        # so we compute an offset.
        try:
            avg_power = analyze.avg_power(
                self._power_log_path(0),
                start=self.prof_start_time - self.epoch_start_time,
            )
        except ValueError:
            # self._log(
            #     "ValueError from analyze.avg_power, please consider increasing self.profile_iter.",
            #     logging.ERROR,
            # )
            raise
        
        self.train_power_result = avg_power

        # Compute and save throughput. We use the time at the master process.
        self.time_consumed = now - self.prof_start_time
        samples_processed = self.sample_num - self.prof_start_sample
        throughput = samples_processed / self.time_consumed
        self.train_tput_result = throughput

        print(f"Profile done with power limit {self.power_limit//1000}W")
    
    def set_power_limit(self, new_pl):
        print(f"[ProfileDataLoader] set power limit to {new_pl}")
        self.power_limit = new_pl * 1000
    
    def set_learning_rate(self, new_lr):
        print(f"[ProfileDataLoader] set learning rate to {new_lr}")
        self.learning_rate = new_lr
    
    # def set_batch_size(self, new_bs):
    #     print(f"[ProfileDataLoader] set batch size to {new_bs}")
    #     self.batch_size = new_bs
    
    def set_dropout_rate(self, new_dr):
        print(f"[ProfileDataLoader] set learning rate to {new_dr}")
        self.dropout_rate = new_dr

def kill_monitor():
    """Kill all Zeus power monitors."""
    monitor = ProfileDataLoader.monitor
    if monitor is not None:
        for i, proc in enumerate(monitor):
            proc.kill()
            print(f"Stopped Zeus monitor {i}.")