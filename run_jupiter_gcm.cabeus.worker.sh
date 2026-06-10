#!/bin/bash

set -euo pipefail

cd "${PBS_O_WORKDIR}"
source "${HOME}/pyenv/bin/activate"

HOSTNAME_SHORT="$(hostname -s)"
IFS=, read -ra NODES <<< "$NODELIST"

NODE_RANK=-1
for i in "${!NODES[@]}"; do
  if [[ "$HOSTNAME_SHORT" == "${NODES[$i]%%.*}" ]]; then
    NODE_RANK=$i
    break
  fi
done

if [[ "$NODE_RANK" -lt 0 ]]; then
  echo "ERROR: hostname $HOSTNAME_SHORT is not in NODELIST=$NODELIST" >&2
  exit 1
fi

echo "Starting torchrun on hostname=$HOSTNAME_SHORT NODE_RANK=$NODE_RANK"

exec torchrun \
  --nnodes="$NNODES" \
  --nproc-per-node="$GPUS_PER_NODE" \
  --node-rank="$NODE_RANK" \
  --rdzv-id="$RDZV_ID" \
  --rdzv-backend=c10d \
  --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
  run_jupiter_moist.py --config "${EXP_INP}" \
                       --output-dir "${EXP_ROOT}/${EXP_DIR}"
