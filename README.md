<div align="center">

# TARC for DSRL-π₀: Time-Adaptive Replanning for Steered Diffusion Policies

## [[website](https://arnavsukhija.github.io/projects/tarc/)]      [[paper](https://arxiv.org/abs/2510.23176)]      [[TARC main repo](https://github.com/arnavsukhija/TARC)]

</div>

## Overview
This repository applies **TARC (Time-Adaptive Robotic Control)** to a pre-trained generalist policy, [π₀](https://github.com/Physical-Intelligence/openpi), steered with **DSRL** (Diffusion Steering via Reinforcement Learning). It is the π₀ / LIBERO experiment of the TARC paper, and is included as a submodule of the [TARC repository](https://github.com/arnavsukhija/TARC).

π₀ predicts an action chunk (up to 50 steps), and DSRL learns a latent-noise policy that steers it. Standard DSRL re-queries π₀ at a fixed interval (`--query_freq`). TARC instead lets the RL agent choose, at every decision, **both the steering noise and how many steps of the chunk to execute before re-querying π₀**. Each re-query pays a switching cost, so the agent only replans when it is worth the expensive π₀ forward pass.

| Mode | Flags | Behaviour |
|---|---|---|
| DSRL baseline | `--switch_cost_wrapper 0 --query_freq K` | Re-query π₀ every `K` steps |
| TARC | `--switch_cost_wrapper 1 --switch_cost c --min_time_between_switches a --max_time_between_switches b` | Agent outputs noise and a replanning time in `[a, b]`; every re-query costs `c` |

This code builds on the [original DSRL implementation](https://github.com/ajwagen/dsrl_pi0) (Wagenmaker et al., CoRL 2025), which in turn builds on [jaxrl2](https://github.com/ikostrikov/jaxrl2) and [PTR](https://github.com/Asap7772/PTR). Supported environments are LIBERO and Aloha in simulation, plus a Franka real-robot setup.

## Installation
1. Create a conda environment:
```
conda create -n dsrl_pi0 python=3.11.11
conda activate dsrl_pi0
```

2. Clone this repo with all submodules (or, from within the TARC repo, run `git submodule update --init --recursive`)
```
git clone git@github.com:arnavsukhija/dsrl_pi0.git --recurse-submodules
cd dsrl_pi0
```

3. Install all packages and dependencies
```
pip install -e .
pip install -r requirements.txt
pip install "jax[cuda12]==0.5.0"

# install openpi
pip install -e openpi
pip install -e openpi/packages/openpi-client

# install Libero
pip install -e LIBERO
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu # needed for libero

# If LIBERO imports fail because of package nesting, use the local workaround:
mv LIBERO libero
find . \( -name '*.py' -o -name '*.sh' -o -name '*.yaml' -o -name '*.yml' -o -name '*.txt' \) -type f \
  -exec perl -pi -e 's/libero\.libero(?!\.libero)/libero.libero.libero/g' {} +
```

No GPU? See [Running LIBERO on a CPU-only machine](#running-libero-on-a-cpu-only-machine-no-gpu) below.

## Training (Simulation)

**TARC on LIBERO** (adaptive replanning):
```
SWITCH_COST=20 MIN_K=5 MAX_K=50 SEED=0 bash examples/scripts/run_libero_tarc.sh
```

**DSRL baseline on LIBERO** (fixed query frequency):
```
QUERY_FREQ=20 SEED=0 bash examples/scripts/run_libero_baseline.sh
```

Both scripts read `TASK_ID` (default 57, the LIBERO-90 task) and log to Weights & Biases. Other entry points:

| Script | Purpose |
|---|---|
| `run_libero.sh`, `run_aloha.sh` | Original DSRL on LIBERO / Aloha |
| `run_libero_ablation.sh` | TARC ablation (evaluation mode, pseudo-time init bias, `K` range) |
| `submit_libero_tarc_sweep.sh`, `submit_libero_baseline_sweep.sh`, `submit_libero_ablation_sweep.sh` | Slurm sweeps (ETH Euler). Use `DRY=1` to print the `sbatch` commands only |
| `submit_libero_new_tasks.sh` | TARC and baselines on additional LIBERO-90 tasks (multi-task result) |
| `smoke_libero_tarc.sh` | Quick smoke test |

Plots for the paper are generated from the W&B runs by the scripts in `plotting/dsrl_pi0/` of the TARC repo.

### Training Logs
Sample W&B runs from the original DSRL release: https://wandb.ai/mitsuhiko/DSRL_pi0_public

## Training (Real)
For real-world experiments, we use the remote hosting feature from pi0 (see [here](https://github.com/Physical-Intelligence/openpi/blob/main/docs/remote_inference.md)) which enables us to host the pi0 model on a higher-spec remote server, in case the robot's client machine is not powerful enough. 

0. Setup Franka robot and install DROID package [[link](https://github.com/droid-dataset/droid.git)]

1. [On the remote server] Host pi0 droid model on your remote server
```
cd openpi && python scripts/serve_policy.py --env=DROID
```
2. [On your robot client machine] Run DSRL
```
bash examples/scripts/run_real.sh
```

## Citation
If you use this code, please cite both TARC and DSRL:

```bibtex
@misc{sukhija2026tarctimeadaptiveroboticcontrol,
      title={TARC: Time-Adaptive Robotic Control}, 
      author={Arnav Sukhija and Lenart Treven and Jin Cheng and Florian Dörfler and Stelian Coros and Andreas Krause},
      year={2026},
      eprint={2510.23176},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2510.23176}, 
}

@article{wagenmaker2025steering,
  author    = {Andrew Wagenmaker and Mitsuhiko Nakamoto and Yunchu Zhang and Seohong Park and Waleed Yagoub and Anusha Nagabandi and Abhishek Gupta and Sergey Levine},
  title     = {Steering Your Diffusion Policy with Latent Space Reinforcement Learning},
  journal   = {Conference on Robot Learning (CoRL)},
  year      = {2025},
}
```

## Credits
Built on the [DSRL](https://github.com/ajwagen/dsrl_pi0) codebase by Wagenmaker, Nakamoto et al., which uses [jaxrl2](https://github.com/ikostrikov/jaxrl2) and [PTR](https://github.com/Asap7772/PTR). Some of the early adaptive-chunking and explorer-learner work in this repository's history is by Klemens Iten. TARC-specific changes are by Arnav Sukhija.


## Running LIBERO on a CPU-only machine (no GPU)

This documents the setup/debugging steps needed to run the LIBERO experiments in `dsrl_pi0` on a machine without a CUDA-capable GPU.

---

### 1. Create a clean Conda environment

A broken / contaminated Conda environment caused issues initially (ROS packages leaking in, missing Python binary).

Create a fresh env:

```bash
conda create -n dsrl python=3.11.11 -y
conda activate dsrl
```

Verify:

```bash
which python
which pip
python -V
python -m pip list
```

Expected:
- Python from `~/miniconda3/envs/dsrl/...`
- Pip from same env
- only a few base packages installed

---

### 2. Install project dependencies

From repo root:

```bash
cd ~/git/dsrl_pi0
python -m pip install -e .
# Dont forget to install JAX as cpu only
pip install "jax[cpu]==0.5.0"
```

---

### 3. Fix LIBERO import path

LIBERO package structure requires adding the local `LIBERO/` directory to `PYTHONPATH`.

From repo root:

```bash
export PYTHONPATH="$PWD/LIBERO:$PYTHONPATH"
```

Without this, imports like:

```python
from libero.libero import benchmark
```

fail.

---

### 4. Install LIBERO editable

```bash
cd ~/git/dsrl_pi0/LIBERO
python -m pip install -e .
cd ..
```

---

### 5. CPU-only rendering setup (important)

Default script assumed EGL / GPU rendering:

```bash
MUJOCO_GL=egl
```

This fails on machines without GPU support.

#### Use OSMesa instead

```bash
unset CUDA_VISIBLE_DEVICES
unset MUJOCO_EGL_DEVICE_ID

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
```

---

### 6. Install OSMesa / Mesa dependencies

Needed for CPU offscreen rendering:

```bash
sudo apt update
sudo apt install \
    libosmesa6 \
    libosmesa6-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev
```

### 7. Required environment variables for launch

From repo root:

```bash
export PYTHONPATH="$PWD/LIBERO:$PYTHONPATH"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

unset CUDA_VISIBLE_DEVICES
unset MUJOCO_EGL_DEVICE_ID

export EXP=./logs/DSRL_pi0_Libero
export OPENPI_DATA_HOME=./openpi

mkdir -p "$EXP"
```

---

### 8. Launch training manually

```bash
python -m examples.launch_train_sim \
  --algorithm pixel_sac \
  --env libero \
  --prefix dsrl_pi0_libero \
  --wandb_project DSRL_pi0_Libero \
  --batch_size 256 \
  --discount 0.999 \
  --seed 0 \
  --max_steps 500000 \
  --eval_interval 10000 \
  --log_interval 500 \
  --eval_episodes 10 \
  --multi_grad_step 20 \
  --start_online_updates 500 \
  --resize_image 64 \
  --action_magnitude 1.0 \
  --query_freq 20 \
  --hidden_dims 128
```

---

### 11. Recommended fixes for `examples/scripts/run_libero.sh`

Update script to include:

```bash
export PYTHONPATH="$PWD/LIBERO:$PYTHONPATH"

unset CUDA_VISIBLE_DEVICES
unset MUJOCO_EGL_DEVICE_ID

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

export OPENPI_DATA_HOME=./openpi
export EXP=./logs/$proj_name
mkdir -p "$EXP"
```

Also prefer:

```bash
python -m examples.launch_train_sim ...
```

instead of:

```bash
python3 examples/launch_train_sim.py
```

---

### 12. Remaining note: LIBERO datasets

Warning observed:

```bash
[Warning]: datasets path ... does not exist!
```

This may require downloading LIBERO datasets separately depending on experiment.

Check LIBERO documentation if dataset-related errors appear next.
