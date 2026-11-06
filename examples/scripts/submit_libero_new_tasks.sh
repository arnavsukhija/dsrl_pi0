#!/bin/bash
# Submit TARC (sc=20, K=5-50, det eval) + fixed-freq baselines (QF=20 and QF=32, det eval)
# for the two additional LIBERO-90 tasks used in the multi-task generalisation result:
#
#   Task  0: KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet
#   Task 86: STUDY_SCENE4_pick_up_the_book_in_the_middle_and_place_it_on_the_cabinet_shelf
#
# QF=32 is the matched-frequency baseline (= measured converged mean K of the TARC policy on
# task 57).  Running QF=20 and QF=32 guards against the reviewer objection that QF=20 was
# unfairly low; if TARC still wins at the matched freq the result is airtight.
#
# Usage:
#   bash examples/scripts/submit_libero_new_tasks.sh                  # submit all for tasks 0+86
#   DRY=1 bash examples/scripts/submit_libero_new_tasks.sh            # dry-run (print only)
#
# Fine-grained control:
#   TASK_IDS="86"        bash ...  # only task 86 (fresh, no prior runs)
#   TASK_IDS="0"  SKIP_TARC=1 QUERY_FREQS="32" bash ...  # add QF=32 for already-running task 0
#   TASK_IDS="57" SKIP_TARC=1 QUERY_FREQS="32" bash ...  # add QF=32 for original task 57

set -e

TASK_IDS=${TASK_IDS:-"0 86"}
SEEDS=${SEEDS:-"0 1 2"}
# Both a standard (QF=20, published DSRL value) and a matched-frequency (QF=32 = mean TARC K)
# baseline are submitted by default.  Override to submit only one:  QUERY_FREQS="32"
QUERY_FREQS=${QUERY_FREQS:-"20 32"}
# Set SKIP_TARC=1 when the TARC run for a task is already submitted (avoids duplicates).
SKIP_TARC=${SKIP_TARC:-0}

# TARC settings (same as the winning cell on task 57)
TARC_SCRIPT=${TARC_SCRIPT:-examples/scripts/run_libero_ablation.sh}
TARC_PROJ=${TARC_PROJ:-DSRL_pi0_Libero_TARC_ablation}
SWITCH_COST=20
MIN_K=5
MAX_K=50
EVAL_MODE=fully_deterministic
BIAS=0.0

# Baseline settings (apples-to-apples: det eval)
BASE_SCRIPT=${BASE_SCRIPT:-examples/scripts/run_libero_baseline.sh}
BASE_PROJ=${BASE_PROJ:-DSRL_pi0_Libero_baseline_deteval}

n_tarc=0
n_base=0
for TASK_ID in $TASK_IDS; do
  for SEED in $SEEDS; do
    if [ "${SKIP_TARC}" = "0" ]; then
      n_tarc=$((n_tarc+1))
      TARC_CMD="PROJ_NAME=$TARC_PROJ TASK_ID=$TASK_ID SWITCH_COST=$SWITCH_COST MIN_K=$MIN_K MAX_K=$MAX_K EVAL_MODE=$EVAL_MODE BIAS=$BIAS SEED=$SEED sbatch $TARC_SCRIPT"
      if [ "${DRY:-0}" = "1" ]; then
        echo "[TARC       task=$TASK_ID seed=$SEED] $TARC_CMD"
      else
        echo "[TARC       task=$TASK_ID seed=$SEED] submitting..."
        eval "$TARC_CMD"
      fi
    fi
    for QF in $QUERY_FREQS; do
      n_base=$((n_base+1))
      BASE_CMD="PROJ_NAME=$BASE_PROJ TASK_ID=$TASK_ID QUERY_FREQ=$QF EVAL_MODE=$EVAL_MODE SEED=$SEED sbatch $BASE_SCRIPT"
      if [ "${DRY:-0}" = "1" ]; then
        echo "[BASE-QF${QF}  task=$TASK_ID seed=$SEED] $BASE_CMD"
      else
        echo "[BASE-QF${QF}  task=$TASK_ID seed=$SEED] submitting..."
        eval "$BASE_CMD"
      fi
    done
  done
done

echo "Submitted ${n_tarc} TARC + ${n_base} baseline jobs for tasks={$TASK_IDS}, seeds={$SEEDS}."
