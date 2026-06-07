#! /bin/bash

EXP_ROOT=/mnt/dart2/2026.JupiterCRM/

#################

NODE=dart2
EXP_INP=jup_crm2d_H2O-NH3_F10_nu0.01.yaml
EXP_DIR=jup_crm2d_H2O-NH3_F100_nu0.01_T1000_F10
RESTART=jup_crm2d_H2O-NH3_F100_nu0.01.final.restart 

paddle submit ${NODE} \
  --log "${EXP_ROOT}/log.${EXP_DIR}" \
  --file run_jupiter_moist.py \
  --file ${EXP_INP} \
  -- \
  MASTER_PORT=29501 \
  DEVICE_ID=1 \
  HDF5_USE_FILE_LOCKING=FALSE \
  python -u run_jupiter_moist.py \
  -r "${RESTART}" \
  -c "${EXP_INP}" \
  --output-dir "${EXP_ROOT}/${EXP_DIR}"

#################

NODE=dart3
EXP_INP=jup_crm2d_H2O-NH3_F10_nu0.1.yaml
EXP_DIR=jup_crm2d_H2O-NH3_F100_nu0.1_T1000_F10
RESTART=jup_crm2d_H2O-NH3_F100_nu0.1.final.restart 

paddle submit ${NODE} \
  --log "${EXP_ROOT}/log.${EXP_DIR}" \
  --file run_jupiter_moist.py \
  --file ${EXP_INP} \
  -- \
  MASTER_PORT=29500 \
  DEVICE_ID=0 \
  HDF5_USE_FILE_LOCKING=FALSE \
  python -u run_jupiter_moist.py \
  -r "${RESTART}" \
  -c "${EXP_INP}" \
  --output-dir "${EXP_ROOT}/${EXP_DIR}"

#################

NODE=dart10
EXP_INP=jup_crm2d_H2O-NH3_F10_nu1.0.yaml
EXP_DIR=jup_crm2d_H2O-NH3_F100_nu1.0_T1000_F10
RESTART=jup_crm2d_H2O-NH3_F100_nu1.0.final.restart 

paddle submit ${NODE} \
  --log "${EXP_ROOT}/log.${EXP_DIR}" \
  --file run_jupiter_moist.py \
  --file ${EXP_INP} \
  -- \
  MASTER_PORT=29501 \
  DEVICE_ID=1 \
  HDF5_USE_FILE_LOCKING=FALSE \
  python -u run_jupiter_moist.py \
  -r "${RESTART}" \
  -c "${EXP_INP}" \
  --output-dir "${EXP_ROOT}/${EXP_DIR}"
