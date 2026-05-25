#!/bin/bash
# Sweep the noBias min_K=1 cell at higher switch cost(s), across all 3 eval modes.
# Tests whether the K-attractor at min_K=1 closes to the min_K=5 result when sc is raised,
# and whether eval-mode choice matters at this sc.
#
# Usage:
#   bash examples/scripts/submit_libero_k1_higher_cost.sh           # 3 eval modes x 3 seeds = 9 jobs (sc=100)
#   DRY=1 bash examples/scripts/submit_libero_k1_higher_cost.sh     # print sbatch commands only
#
# Override sweep:
#   SWITCH_COSTS="100 200" EVAL_MODES="fully_deterministic fully_random hybrid" SEEDS="0 1 2" \
#     bash examples/scripts/submit_libero_k1_higher_cost.sh

set -e

SWITCH_COSTS=${SWITCH_COSTS:-"100"}
EVAL_MODES=${EVAL_MODES:-"fully_deterministic fully_random hybrid"}
SEEDS=${SEEDS:-"0 1 2"}
SCRIPT=${SCRIPT:-examples/scripts/run_libero_ablation.sh}
PROJ_NAME=${PROJ_NAME:-DSRL_pi0_Libero_TARC_ablation}

# Fixed: noBias, min_K=1, max_K=50  (results land in the same wandb project so the
# existing plot pipeline picks them up automatically).
BIAS=0.0
MIN_K=1
MAX_K=50

n=0
for SC in $SWITCH_COSTS; do
  for EVAL_MODE in $EVAL_MODES; do
    for SEED in $SEEDS; do
      n=$((n+1))
      CMD="PROJ_NAME=$PROJ_NAME EVAL_MODE=$EVAL_MODE BIAS=$BIAS MIN_K=$MIN_K MAX_K=$MAX_K SWITCH_COST=$SC SEED=$SEED sbatch $SCRIPT"
      if [ "${DRY:-0}" = "1" ]; then
        echo "[$n] [sc=$SC eval=$EVAL_MODE seed=$SEED] $CMD"
      else
        echo "[$n] [sc=$SC eval=$EVAL_MODE seed=$SEED] $CMD"
        eval "$CMD"
      fi
    done
  done
done

echo "Submitted $n noBias_K1 jobs at switch_costs={$SWITCH_COSTS}, eval_modes={$EVAL_MODES} to '$PROJ_NAME'."
