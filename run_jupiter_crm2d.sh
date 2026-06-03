#! /bin/bash

DATE_STR=$(date +%y%m%d)
EXP_ROOT=${HOME}/data/2026.JupiterCRM/

# case1
EXP_NAME=crm2d_H2O-NH3_F100_nu1.0
EXP_INP=jup_${EXP_NAME}.yaml
EXP_DIR=${DATE_STR}_${EXP_NAME}
MASTER_PORT=29500 DEVICE_ID=0 python -u run_jupiter_moist.py -c ${EXP_INP} --output-dir ${EXP_ROOT}/${EXP_DIR} &> ${EXP_ROOT}/log.${EXP_DIR} &

# case2
EXP_NAME=crm2d_H2O-NH3_F100_nu10.0
EXP_INP=jup_${EXP_NAME}.yaml
EXP_DIR=${DATE_STR}_${EXP_NAME}
DEVICE_ID=1 python -u run_jupiter_moist.py -c ${EXP_INP} --output-dir ${EXP_ROOT}/${EXP_DIR} &> ${EXP_ROOT}/log.${EXP_DIR} &
