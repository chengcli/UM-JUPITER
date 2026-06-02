#! /bin/bash

# H2O-NH3 moist 2d + 3d
# dart9
MASTER_PORT=29500 DEVICE_ID=0 python -u run_jupiter_moist.py -c jupiter_crm2d_H2O-NH3_F100_nu0.01.yaml --output-dir ./260602_crm2d_H2O-NH3_F100_nu0.01 &> log.260602_crm2d_H2O-NH3_F100_nu0.01 &
#DEVICE_ID=1 python -u run_jupiter_moist.py -c jupiter_crm3d_H2O-NH3_F100_nu0.01.yaml --output-dir ./260602_crm3d_H2O-NH3_F100_nu0.01 &> log.260602_crm3d_H2O-NH3_F100_nu0.01 &

# dart10
MASTER_PORT=29500 DEVICE_ID=0 python -u run_jupiter_moist.py -c jupiter_crm2d_H2O-NH3_F100_nu0.1.yaml --output-dir ./260602_crm2d_H2O-NH3_F100_nu0.1 &> log.260602_crm2d_H2O-NH3_F100_nu0.1 &
#DEVICE_ID=1 python -u run_jupiter_moist.py -c jupiter_crm3d_H2O-NH3_F100_nu0.1.yaml --output-dir ./260602_crm3d_H2O-NH3_F100_nu0.1 &> log.260602_crm3d_H2O-NH3_F100_nu0.1 &

# dart11
MASTER_PORT=29500 DEVICE_ID=0 python -u run_jupiter_moist.py -c jupiter_crm2d_H2O-NH3_1000.yaml --output-dir ./260518_crm2d_H2O-NH3_1000 &> log.260518_crm2d_H2O-NH3_1000 &
#DEVICE_ID=1 python -u run_jupiter_moist.py -c jupiter_crm3d_H2O-NH3_1000.yaml --output-dir ./260518_crm3d_H2O-NH3_1000 &> log.260518_crm3d_H2O-NH3_1000 &

# H2O-NH3-H2S moist + dry
# dungeon1
#DEVICE_ID=0 python -u run_jupiter_crm.py -r jupiter_crm_7.5.final.restart -c jupiter_crm_7.5.yaml --output-dir ./260520_crm_7.5 &> log.260520_crm_7.5 &
#DEVICE_ID=1 python -u run_jupiter_dry.py -r jupiter_dry_7.5.final.restart -c jupiter_dry_7.5.yaml --output-dir ./260520_dry_7.5 &> log.260520_dry_7.5 &
#torchrun --nproc-per-node 2 run_jupiter_moist.py -c jupiter_gcm_H2O-NH3-H2S_100.yaml --output-dir ./260601_gcm_100 &> log.260601_gcm_100 &
#torchrun --nproc-per-node 2 run_jupiter_moist.py -c jupiter_gcm_H2O-NH3-H2S_1000.yaml --output-dir ./260528_gcm_1000 &> log.260528_gcm_1000 &
#torchrun --nproc-per-node 2 run_jupiter_crm.py -c jupiter_crm3d_F100_nu0.1.yaml --output-dir ./260602_crm3d_F100_nu0.1 &> log.260602_crm3d_F100_nu0.1 &

# dungeon2
#DEVICE_ID=1 python -u run_jupiter_crm.py -c jupiter_crm3d_F100_nu0.01.yaml --output-dir ./260602_crm3d_F100_nu0.01 &> log.260602_crm3d_F100_nu0.01 &
#torchrun --nproc-per-node 2 run_jupiter_crm.py -c jupiter_crm3d_F100_nu0.01.yaml --output-dir ./260602_crm3d_F100_nu0.01 &> log.260602_crm3d_F100_nu0.01 &

#DEVICE_ID=0 python -u run_jupiter_crm.py -r jupiter_crm_100.final.restart -c jupiter_crm_100.yaml --output-dir ./260520_crm_100 &> log.260520_crm_100 &
#DEVICE_ID=1 python -u run_jupiter_crm.py -r jupiter_crm_7.5.final.restart -c jupiter_crm_7.5.yaml --output-dir ./260520_crm_7.5 &> log.260520_crm_7.5 &
#DEVICE_ID=1 python -u run_jupiter_dry.py -r jupiter_dry_100.final.restart -c jupiter_dry_100.yaml --output-dir ./260520_dry_100 &> log.260520_dry_100 &

# dungeon3
#torchrun --nproc-per-node 2 run_jupiter_crm.py -c jupiter_crm3d_F100_nu1.0.yaml --output-dir ./260602_crm3d_F100_nu1.0 &> log.260602_crm3d_F100_nu1.0 &
#DEVICE_ID=0 python -u run_jupiter_crm.py -r jupiter_crm_1000.final.restart -c jupiter_crm_1000.yaml --output-dir ./260520_crm_1000 &> log.260520_crm_1000 &
#DEVICE_ID=1 python -u run_jupiter_dry.py -r jupiter_dry_1000.final.restart -c jupiter_dry_1000.yaml --output-dir ./260520_dry_1000 &> log.260520_dry_1000 &
