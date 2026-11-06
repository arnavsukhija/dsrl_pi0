#!/bin/bash
#SBATCH --job-name=dsrl_abl
#SBATCH --time=23:59:00
#SBATCH --mem-per-cpu=8192
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=logs/dsrl_abl_%j.out
#SBATCH --error=logs/dsrl_abl_%j.err
#SBATCH --export=ALL

# TARC ablation runs for DSRL pi0 on LIBERO.
# Supports the 3 new variants from the eval-mode + init-bias ablation:
#   variant 3 (det eval, no bias)        : EVAL_MODE=fully_deterministic, BIAS=0
#   variant 4 (hybrid eval, bias, minK=1): EVAL_MODE=hybrid,              BIAS=2.0
#   variant 5 (det eval, bias, minK=1)   : EVAL_MODE=fully_deterministic, BIAS=2.0
#
# Submit on Euler with:
#   EVAL_MODE=fully_deterministic BIAS=2.0 MIN_K=1 SWITCH_COST=20 SEED=0 sbatch examples/scripts/run_libero_ablation.sh

proj_name=${PROJ_NAME:-DSRL_pi0_Libero_TARC_ablation}
device_id=${DEVICE_ID:-0}

# Ablation knobs
EVAL_MODE=${EVAL_MODE:-hybrid}              # hybrid | fully_deterministic | fully_random
BIAS=${BIAS:-0.0}                           # pseudo_time_init_bias (0.0 = off, 2.0 = recommended)
SWITCH_COST=${SWITCH_COST:-20}
MIN_K=${MIN_K:-1}
MAX_K=${MAX_K:-50}
SEED=${SEED:-0}
TASK_ID=${TASK_ID:-57}

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

# Build a descriptive suffix so the W&B run name encodes the ablation cell.
SUFFIX="task${TASK_ID}_sc${SWITCH_COST}_K${MIN_K}-${MAX_K}_eval-${EVAL_MODE}_bias${BIAS}"

python -m examples.launch_train_sim \
  --algorithm pixel_sac \
  --env libero \
  --prefix dsrl_pi0_libero_abl \
  --suffix "${SUFFIX}" \
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
  --time_as_part_of_state 1 \
  --eval_mode ${EVAL_MODE} \
  --pseudo_time_init_bias ${BIAS} \
  --libero_task_id ${TASK_ID}
