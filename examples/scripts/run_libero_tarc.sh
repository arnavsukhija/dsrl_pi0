#!/bin/bash
#SBATCH --job-name=dsrl_tarc
#SBATCH --time=23:59:00
#SBATCH --mem-per-cpu=8192
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=logs/dsrl_tarc_%j.out
#SBATCH --error=logs/dsrl_tarc_%j.err
#SBATCH --export=ALL

# TARC-style adaptive replanning for DSRL on LIBERO.
# Submit on Euler with:   sbatch examples/scripts/run_libero_tarc.sh
# Or run locally with:    bash   examples/scripts/run_libero_tarc.sh
#
# Override hyperparameters via env vars before submission, e.g.:
#   SWITCH_COST=0.5 SEED=1 sbatch examples/scripts/run_libero_tarc.sh

proj_name=${PROJ_NAME:-DSRL_pi0_Libero_TARC}
device_id=${DEVICE_ID:-0}

# TARC sweep knobs
SWITCH_COST=${SWITCH_COST:-0.1}
MIN_K=${MIN_K:-1}
MAX_K=${MAX_K:-20}
SEED=${SEED:-0}

# --- Euler-specific bootstrap (no-op when running locally) ---
if [ -n "${SLURM_JOB_ID:-}" ]; then
  echo "[slurm] job_id=${SLURM_JOB_ID} on node ${SLURMD_NODENAME} gpu ${CUDA_VISIBLE_DEVICES}"
  # GPU compute nodes have no direct internet; load Euler's proxy to reach S3 / W&B / HF Hub
  module load eth_proxy 2>/dev/null || echo "[slurm] WARN: 'module load eth_proxy' failed"
  # Activate the dsrl_pi0 conda env; assumes miniconda3 is installed in $HOME
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate dsrlpi0_tarc
  # Prevent ~/.local from leaking into sys.path
  export PYTHONNOUSERSITE=1
  echo "[slurm] python=$(which python)"
  # If the user passed a working dir, cd there; otherwise assume we're already in dsrl_pi0
  cd "${DSRL_PI0_DIR:-$HOME/TARC/dsrl_pi0}"
fi

mkdir -p logs

# Rendering / JAX env vars
export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-$device_id}

export PYTHONPATH="$PWD/LIBERO:$PYTHONPATH"
# Cache pi0 checkpoints on /cluster/scratch to avoid $HOME quota (~5 GB per checkpoint)
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/cluster/scratch/$USER/openpi_data}"
mkdir -p "$OPENPI_DATA_HOME"
export EXP=./logs/$proj_name
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Only set CUDA_VISIBLE_DEVICES locally; Slurm sets it for us on the cluster
if [ -z "${SLURM_JOB_ID:-}" ]; then
  export CUDA_VISIBLE_DEVICES=$device_id
fi

mkdir -p "$EXP"

python -m examples.launch_train_sim \
  --algorithm pixel_sac \
  --env libero \
  --prefix dsrl_pi0_libero_tarc \
  --suffix "sc${SWITCH_COST}_K${MIN_K}-${MAX_K}" \
  --wandb_project ${proj_name} \
  --batch_size 256 \
  --discount 0.999 \
  --seed ${SEED} \
  --max_steps 500000 \
  --eval_interval 10000 \
  --log_interval 500 \
  --eval_episodes 10 \
  --multi_grad_step 20 \
  --start_online_updates 500 \
  --resize_image 64 \
  --action_magnitude 1.0 \
  --hidden_dims 128 \
  --switch_cost_wrapper 1 \
  --switch_cost ${SWITCH_COST} \
  --min_time_between_switches ${MIN_K} \
  --max_time_between_switches ${MAX_K} \
  --time_as_part_of_state 1
