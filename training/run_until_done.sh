#!/bin/zsh
# Unattended training: runs wesep_retrain.py (the notebook's code) and restarts it after any crash,
# resuming from the last checkpoint each time, until RUN/DONE exists. Keeps the Mac awake (caffeinate).
#
#   cd training && TSE_MODE=finetune_research PYTHON=/path/to/python ./run_until_done.sh
#
# Stops for good if training diverged (RUN/DIVERGED) or after MAX_ATTEMPTS restarts.
set -u
PY=${PYTHON:-python3}
MODE=${TSE_MODE:-finetune_research}
WORK=${TSE_WORK:-$HOME/tse_training}
RUN_DIR="$WORK/runs/$([ "${TSE_SMOKE:-0}" = 1 ] && echo "smoke_$MODE" || echo "$MODE")"
mkdir -p "$RUN_DIR"
attempt=0
until [ -f "$RUN_DIR/DONE" ]; do
  if [ -f "$RUN_DIR/DIVERGED" ]; then echo "training diverged (see $RUN_DIR/DIVERGED); not restarting"; exit 1; fi
  attempt=$((attempt + 1))
  if [ "$attempt" -gt "${MAX_ATTEMPTS:-50}" ]; then echo "giving up after $((attempt - 1)) attempts"; exit 1; fi
  echo "=== attempt $attempt, $(date) ===" | tee -a "$RUN_DIR/console.log"
  MPLBACKEND=Agg TSE_MODE="$MODE" TSE_WORK="$WORK" TSE_INSTALL="${TSE_INSTALL:-0}" \
    caffeinate -dims "$PY" "${0:A:h}/wesep_retrain.py" 2>&1 | tee -a "$RUN_DIR/console.log"
  [ -f "$RUN_DIR/DONE" ] || sleep 30
done
echo "training done: $RUN_DIR"
