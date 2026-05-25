#!/bin/bash
#SBATCH --job-name=dsrl_baseline
#SBATCH --time=23:59:00
#SBATCH --mem-per-cpu=8192
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=logs/dsrl_baseline_%j.out
#SBATCH --error=logs/dsrl_baseline_%j.err
#SBATCH --export=ALL

# Fixed-query-freq DSRL baseline on LIBERO (no TARC, switch_cost_wrapper=0).
# Submit on Euler with:   sbatch examples/scripts/run_libero_baseline.sh
# Or run locally with:    bash   examples/scripts/run_libero_baseline.sh
#
# Sweep dimensions via env vars:
#   QUERY_FREQ=10 SEED=0 sbatch examples/scripts/run_libero_baseline.sh

proj_name=${PROJ_NAME:-DSRL_pi0_Libero_baseline}
device_id=${DEVICE_ID:-0}
QUERY_FREQ=${QUERY_FREQ:-20}
SEED=${SEED:-0}
# Eval mode: hybrid (DSRL default, stochastic noise) or fully_deterministic (apples-to-apples vs TARC det eval).
# For baselines fully_random == hybrid (no pseudo_time dim to randomize).
EVAL_MODE=${EVAL_MODE:-hybrid}

# --- Euler-specific bootstrap (no-op when running locally) ---
if [ -n "${SLURM_JOB_ID:-}" ]; then
  echo "[slurm] job_id=${SLURM_JOB_ID} on node ${SLURMD_NODENAME} gpu ${CUDA_VISIBLE_DEVICES}"
  module load eth_proxy 2>/dev/null || echo "[slurm] WARN: 'module load eth_proxy' failed"
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate dsrlpi0_tarc
  export PYTHONNOUSERSITE=1
  echo "[slurm] python=$(which python)"
  cd "${DSRL_PI0_DIR:-$HOME/TARC/dsrl_pi0}"
fi

mkdir -p logs

export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-$device_id}

export PYTHONPATH="$PWD/LIBERO:$PYTHONPATH"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/cluster/scratch/$USER/openpi_data}"
mkdir -p "$OPENPI_DATA_HOME"
export EXP=./logs/$proj_name
export XLA_PYTHON_CLIENT_PREALLOCATE=false
if [ -z "${SLURM_JOB_ID:-}" ]; then
  export CUDA_VISIBLE_DEVICES=$device_id
fi

mkdir -p "$EXP"

python -m examples.launch_train_sim \
  --algorithm pixel_sac \
  --env libero \
  --prefix dsrl_pi0_libero_baseline \
  --suffix "qf${QUERY_FREQ}_eval-${EVAL_MODE}" \
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
  --query_freq ${QUERY_FREQ} \
  --switch_cost_wrapper 0 \
  --eval_mode ${EVAL_MODE}
