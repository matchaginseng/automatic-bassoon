<div align="center">
<h1>Automatic Bassoon: Zeus 2 Electric Boogaloo</h1>
</div>

[![Zeus's arXiv](https://custom-icon-badges.herokuapp.com/badge/ID-2208.06102-b31b1b.svg?logo=arxiv-white&logoWidth=35)](https://arxiv.org/abs/2208.06102)
[![Zeus's Docker Hub](https://img.shields.io/badge/Docker-SymbioticLab%2FZeus-blue.svg?logo=docker&logoColor=white)](https://hub.docker.com/r/symbioticlab/zeus)
[![Zeus's Homepage build](https://github.com/SymbioticLab/Zeus/actions/workflows/deploy_homepage.yaml/badge.svg)](https://github.com/SymbioticLab/Zeus/actions/workflows/deploy_homepage.yaml)
[![Zeus's Apache-2.0 License](https://custom-icon-badges.herokuapp.com/github/license/SymbioticLab/Zeus?logo=law)](/LICENSE)

## About Automatic-Bassoon

automatic-bassoon builds off of Zeus's profiler. See [our writeup](https://www.overleaf.com/project/6383ae2ca2b1544e6b589cc5) for more details.

## About Zeus

Zeus automatically optimizes the **energy and time** of training a DNN to a target validation metric by finding the optimal **batch size** and **GPU power limit**.

For more details, refer to the Zeus [NSDI’23 publication](https://arxiv.org/abs/2208.06102) for details.
Check out [Overview](https://ml.energy/zeus/overview/) for a summary. You can also access the [Zeus repo](https://github.com/SymbioticLab/Zeus) for a high-level overview of the Zeus repo organization.

Zeus is part of [The ML.ENERGY Initiative](https://ml.energy).

## Getting Started

We ran automatic-bassoon on an Ubuntu 20.04 AWS instance using one p2.xlarge GPU and four vCPUs. Zeus provides a Docker image fully equipped with all dependencies and environments, and automatic-bassoon uses the same one. 

The steps are:

1. Initialize and ssh into an AWS instance
2. Install Docker (step 1 of [these instructions](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-20-04))
3. Install nvidia-docker2 (step 2 of [these instructions](https://www.ibm.com/docs/en/maximo-vi/8.2.0?topic=planning-installing-docker-nvidia-docker2#install_ub))
4. `sudo systemctl status docker` to check Docker is running
5. `nvidia-smi` to check that you have a GPU driver installed; if not, follow [these instructions](https://levelup.gitconnected.com/how-to-install-an-nvidia-gpu-driver-on-an-aws-ec2-instance-20185c1c578c)
6. `git clone https://github.com/matchaginseng/automatic-bassoon`
7. `cd automatic-bassoon`
8. `sudo docker run -it --gpus all --cap-add SYS_ADMIN --shm-size 64G -v $(pwd):/workspace/zeus symbioticlab/zeus:latest bash`

This will spawn the Docker image. The `-v` command mounts the repo at `/workspace/zeus` in the Docker container. Our code lives in the `zeus/zeus2` folder.

Then we want to make the power monitor, so run:
```
cd zeus/zeus_monitor
cmake .
make
```

The resulting power monitor binary is `/workspace/zeus/zeus_monitor/zeus_monitor` inside the Docker container.

For more information about the Docker image, refer to Zeus's [environment setup instructions](https://ml.energy/zeus/getting_started/environment/). To work with Zeus, refer to Zeus's [getting started instructions](https://ml.energy/zeus/getting_started) for complete instructions on environment setup, installation, and integration.

## Running a Thing

We provide one working example of our profiler on ShuffleNetV2 with the [CIFAR100](../examples/cifar100) dataset. To integrate the profiler with other models, you can follow a similar structure.

To run our profiler, make sure the Docker image is up and running and that you're in the `workspace/zeus/zeus2` folder inside the Docker container. An example command is:

```
# All arguments shown are default values, except for `acc_thresholds`, `batch_sizes`, 
# `learning_rates`, and `dropout_rates`, which default to empty lists. 
python run_shufflenet_vx.py \
    --seed 1 \
    --b_0 1024 \
    --b_min 8 \
    --b_max 4096 \
    --num_recurrence 100 \
    --eta_knob 0.5 \
    --target_metric 0.50 \
    --warmup_iters 3 \
    --profile_iters 10 \
    --acc_thresholds 0.5 0.4 0.3 \ # accuracy thresholds to re-profile on
    --batch_sizes 128 256 512 1024 \ # batch sizes to profile over
    --learning_rates 0.001 0.005 0.01 \ # learning rates to profile over
    --dropout_rates 0.0 0.25 0.5 \ # dropout rates to profile over
    --max_epochs 100
```

The `eta_knob` trades off time and energy consumption. See the equation in the writeup.

Alos note that `acc_thresholds` must be specified in decreasing order because the code pops off the last threshold in the list to use as the accuracy threshold for profiling. For example, if we want to profile at accuracy thresholds 0.3, 0.4, and 0.5, the argument must be written as `--acc_thresholds 0.5 0.4 0.3`, and not `--acc_thresholds 0.3 0.4 0.5`. 


## Contact
Charumathi Badrinath (charumathibadrinath@college.harvard.edu), Helen Cho (hcho@college.harvard.edu), Vicki Xu (vickixu@college.harvard.edu)
