#!/bin/bash
# Submit the 3-variant ablation sweep on Euler.
#
# Variants (4 cells total, all under wandb project $PROJ_NAME):
#   noBias_detEval      : fully-det eval, no bias       -> swept over min_K in {1, 5}
#                           - min_K=5 = sanity check for "does pi0 need stochastic noise" at the good config
#                           - min_K=1 = no-intervention reference for the bias intervention below
#   bias_hybridEval_K1  : hybrid eval, init-bias, min_K=1   -> does bias fix K-attractor under principled eval?
#   bias_detEval_K1     : det eval,    init-bias, min_K=1   -> deployment-ready config if hybrid works
#
# Usage:
#   bash examples/scripts/submit_libero_ablation_sweep.sh           # submit all
#   DRY=1 bash examples/scripts/submit_libero_ablation_sweep.sh     # print sbatch commands only
#
# Default sweep:
#   SWITCH_COSTS="20 50"  SEEDS="0 1 2"  ->  2 sc x 3 seeds x 4 cells = 24 jobs

set -e

SWITCH_COSTS=${SWITCH_COSTS:-"20 50"}
SEEDS=${SEEDS:-"0 1 2"}
SCRIPT=${SCRIPT:-examples/scripts/run_libero_ablation.sh}
PROJ_NAME=${PROJ_NAME:-DSRL_pi0_Libero_TARC_ablation}

# variant: "label EVAL_MODE BIAS MIN_K MAX_K"
VARIANTS=(
  "noBias_detEval_K5      fully_deterministic 0.0 5 50"
  "noBias_detEval_K1      fully_deterministic 0.0 1 50"
  "bias_hybridEval_K1     hybrid              2.0 1 50"
  "bias_detEval_K1        fully_deterministic 2.0 1 50"
)

n=0
for variant_line in "${VARIANTS[@]}"; do
  # split on whitespace
  read -r label eval_mode bias min_k max_k <<< "$variant_line"
  for SC in $SWITCH_COSTS; do
    for SEED in $SEEDS; do
      n=$((n+1))
      CMD="PROJ_NAME=$PROJ_NAME EVAL_MODE=$eval_mode BIAS=$bias MIN_K=$min_k MAX_K=$max_k SWITCH_COST=$SC SEED=$SEED sbatch $SCRIPT"
      if [ "${DRY:-0}" = "1" ]; then
        echo "[$n] [$label] $CMD"
      else
        echo "[$n] [$label] $CMD"
        eval "$CMD"
      fi
    done
  done
done

echo "Submitted $n ablation jobs to project '$PROJ_NAME'."
