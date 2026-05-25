#!/bin/bash
# Submit the fixed-query-freq baseline sweep on Euler.
# These provide the (success vs num_pi0_calls) curve that TARC's adaptive runs are compared against.
#
# Usage:
#   bash examples/scripts/submit_libero_baseline_sweep.sh           # submit
#   DRY=1 bash examples/scripts/submit_libero_baseline_sweep.sh     # print sbatch commands only
#
# Sweep:
#   QUERY_FREQS="1 5 10 20 50" SEEDS="0 1 2"   (default — 15 jobs)

set -e

QUERY_FREQS=${QUERY_FREQS:-"1 5 10 20 50"}
SEEDS=${SEEDS:-"0 1 2"}
SCRIPT=${SCRIPT:-examples/scripts/run_libero_baseline.sh}
EVAL_MODE=${EVAL_MODE:-hybrid}                      # hybrid (DSRL default) or fully_deterministic
PROJ_NAME=${PROJ_NAME:-DSRL_pi0_Libero_baseline}    # override to e.g. ..._baseline_deteval

n=0
for QF in $QUERY_FREQS; do
  for SEED in $SEEDS; do
    n=$((n+1))
    CMD="PROJ_NAME=$PROJ_NAME QUERY_FREQ=$QF SEED=$SEED EVAL_MODE=$EVAL_MODE sbatch $SCRIPT"
    if [ "${DRY:-0}" = "1" ]; then
      echo "$CMD"
    else
      echo "[$n] $CMD"
      eval "$CMD"
    fi
  done
done

echo "Submitted $n baseline jobs."
