#! /bin/bash

EXP_ROOT=${HOME}/data/2026.JupiterCRM/

export HDF5_USE_FILE_LOCKING=FALSE

# case1
EXP_NAME=crm2d_H2O-NH3_F10_nu100.0
EXP_INP=jup_${EXP_NAME}.yaml
EXP_DIR=jup_${EXP_NAME}
    #-r jup_${EXP_NAME}.final.restart \
MASTER_PORT=29500 DEVICE_ID=0 python -u run_jupiter_moist.py \
    -c ${EXP_INP} --output-dir ${EXP_ROOT}/${EXP_DIR} &>> ${EXP_ROOT}/log.${EXP_DIR} &

# case2
EXP_NAME=crm2d_H2O-NH3_F10_nu1000.0
EXP_INP=jup_${EXP_NAME}.yaml
EXP_DIR=jup_${EXP_NAME}
    #-r jup_${EXP_NAME}.final.restart \
DEVICE_ID=1 python -u run_jupiter_moist.py \
    -c ${EXP_INP} --output-dir ${EXP_ROOT}/${EXP_DIR} &>> ${EXP_ROOT}/log.${EXP_DIR} &
