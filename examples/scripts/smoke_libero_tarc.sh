#!/bin/bash
#SBATCH --job-name=dsrl_tarc_smoke
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=8192
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=logs/dsrl_tarc_smoke_%j.out
#SBATCH --error=logs/dsrl_tarc_smoke_%j.err
#SBATCH --export=ALL

# Smoke test for TARC adaptive replanning on LIBERO.
# Submit on Euler with:   sbatch examples/scripts/smoke_libero_tarc.sh
# Or run locally with:    bash   examples/scripts/smoke_libero_tarc.sh
#
# Goal: verify the new code path runs end-to-end with no crashes, before launching real sweeps.
# Runs short: 1 rollout, ~500 gradient steps, one eval pass. Should finish in well under an hour.

proj_name=${PROJ_NAME:-DSRL_pi0_Libero_TARC_smoke}
device_id=${DEVICE_ID:-0}

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
  --prefix dsrl_pi0_libero_tarc_smoke \
  --wandb_project ${proj_name} \
  --batch_size 64 \
  --discount 0.999 \
  --seed 0 \
  --max_steps 500 \
  --eval_interval 200 \
  --log_interval 50 \
  --eval_episodes 2 \
  --multi_grad_step 5 \
  --start_online_updates 1 \
  --resize_image 64 \
  --action_magnitude 1.0 \
  --hidden_dims 128 \
  --switch_cost_wrapper 1 \
  --switch_cost 0.1 \
  --min_time_between_switches 1 \
  --max_time_between_switches 20 \
  --time_as_part_of_state 1
