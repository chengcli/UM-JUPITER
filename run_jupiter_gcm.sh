#! /bin/bash

NODE=dungeon1
EXP_ROOT=/data00/2026.JUPITER_GCM/

# case1
EXP_NAME=gcm_H2O-NH3_F10
EXP_INP=jup_${EXP_NAME}.yaml
EXP_DIR=jup_${EXP_NAME}
RESTAET=jup_${EXP_NAME}.init.restart

paddle submit ${NODE} \
  --log "${EXP_ROOT}/log.${EXP_DIR}" \
  --file run_jupiter_moist.py \
  --file ${EXP_INP} \
  -- \
  torchrun \
    --nproc_per_node 2 \
    run_jupiter_moist.py \
    -c "${EXP_INP}" \
    -r "${RESTAET}" \
    --output-dir "${EXP_ROOT}/${EXP_DIR}"
