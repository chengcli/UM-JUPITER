#! /bin/bash

NODE=dart5
EXP_ROOT=/mnt/dart2/2026.JupiterCRM/

# case1
EXP_NAME=crm2d_H2O-NH3_F100_nu50.0
EXP_INP=jup_${EXP_NAME}.yaml
EXP_DIR=jup_${EXP_NAME}
RESTAET=jup_${EXP_NAME}.final.restart \

paddle submit ${NODE} \
  --log "${EXP_ROOT}/log.${EXP_DIR}" \
  --file run_jupiter_moist.py \
  --file ${EXP_INP} \
  -- \
  MASTER_PORT=29500 \
  DEVICE_ID=0 \
  HDF5_USE_FILE_LOCKING=FALSE \
  python -u run_jupiter_moist.py \
  -c "${EXP_INP}" \
  --output-dir "${EXP_ROOT}/${EXP_DIR}"

# case2
#EXP_NAME=crm2d_H2O-NH3_F10_nu0.01
#EXP_INP=jup_${EXP_NAME}.yaml
#EXP_DIR=jup_${EXP_NAME}
#    #-r jup_${EXP_NAME}.final.restart \
#DEVICE_ID=1 python -u run_jupiter_moist.py \
#    -c ${EXP_INP} --output-dir ${EXP_ROOT}/${EXP_DIR} &>> ${EXP_ROOT}/log.${EXP_DIRkk} &
