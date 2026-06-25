#! /bin/bash

NODE=dungeon1
EXP_ROOT=/data00/2026.JUPITER_CRM/

# case1
EXP_NAME=crm3d_H2O-NH3_F10_nu0.1
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

#################

NODE=dungeon2
EXP_ROOT=/data01/2026.JUPITER_CRM/

# case1
EXP_NAME=crm3d_H2O-NH3_F10_nu0.01
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

#################

NODE=dungeon3
EXP_ROOT=/data02/2026.JUPITER_CRM/

# case1
EXP_NAME=crm3d_H2O-NH3_F10_nu1.0
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
