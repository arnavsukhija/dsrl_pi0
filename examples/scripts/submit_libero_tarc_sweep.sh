#!/bin/bash
# Submit the full TARC switch-cost sweep on Euler.
# Usage:
#   bash examples/scripts/submit_libero_tarc_sweep.sh           # submits real
#   DRY=1 bash examples/scripts/submit_libero_tarc_sweep.sh     # prints sbatch commands only
#
# Each job runs run_libero_tarc.sh with different env vars (Slurm exports them via --export=ALL).

set -e

SWITCH_COSTS=${SWITCH_COSTS:-"0.0 0.01 0.1 0.5 1.0 5.0"}
SEEDS=${SEEDS:-"0 1 2"}
MIN_K=${MIN_K:-1}
MAX_K=${MAX_K:-20}
SCRIPT=${SCRIPT:-examples/scripts/run_libero_tarc.sh}

n=0
for SC in $SWITCH_COSTS; do
  for SEED in $SEEDS; do
    n=$((n+1))
    CMD="SWITCH_COST=$SC SEED=$SEED MIN_K=$MIN_K MAX_K=$MAX_K sbatch $SCRIPT"
    if [ "${DRY:-0}" = "1" ]; then
      echo "$CMD"
    else
      echo "[$n] $CMD"
      eval "$CMD"
    fi
  done
done

echo "Submitted $n jobs."
