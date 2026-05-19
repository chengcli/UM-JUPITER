#! /bin/bash

# H2O-NH3 moist
# dart9
#DEVICE_ID=0 python -u run_jupiter_crm.py -c jupiter_crm2d_H2O-NH3_7.5.yaml --output-dir ./260518_crm2d_H2O-NH3_7.5 &> log.260518_crm2d_H2O-NH3_7.5 &
#DEVICE_ID=1 python -u run_jupiter_crm.py -c jupiter_crm3d_H2O-NH3_7.5.yaml --output-dir ./260518_crm3d_H2O-NH3_7.5 &> log.260518_crm3d_H2O-NH3_7.5 &

# dart10
#DEVICE_ID=0 python -u run_jupiter_crm.py -c jupiter_crm2d_H2O-NH3_100.yaml --output-dir ./260518_crm2d_H2O-NH3_100 &> log.260518_crm2d_H2O-NH3_100 &
#DEVICE_ID=1 python -u run_jupiter_crm.py -c jupiter_crm3d_H2O-NH3_100.yaml --output-dir ./260518_crm3d_H2O-NH3_100 &> log.260518_crm3d_H2O-NH3_100 &

# dart11
#DEVICE_ID=0 python -u run_jupiter_crm.py -c jupiter_crm2d_H2O-NH3_1000.yaml --output-dir ./260518_crm2d_H2O-NH3_1000 &> log.260518_crm2d_H2O-NH3_1000 &
#DEVICE_ID=1 python -u run_jupiter_crm.py -c jupiter_crm3d_H2O-NH3_1000.yaml --output-dir ./260518_crm3d_H2O-NH3_1000 &> log.260518_crm3d_H2O-NH3_1000 &

# H2O-NH3-H2S moist
# dungeon
#DEVICE_ID=0 python run_jupiter_crm.py -c jupiter_crm_7.5.yaml --output-dir ./output_260518_moist_7.5 > log.260518_moist_7.5 &
#DEVICE_ID=1 python run_jupiter_crm.py -c jupiter_crm_1000.yaml --output-dir ./output_260518_moist_1000 > log.260518_moist_1000 &

# dry with tracer
#DEVICE_ID=1 python run_jupiter_dry.py -c jupiter_dry.yaml --output-dir ./output_260513_dry > log.260513_dry &
#DEVICE_ID=0 python run_jupiter_dry.py -c jupiter_dry_7.5.yaml --output-dir ./output_260513_dry_7.5 > log.260513_dry_7.4 &
#DEVICE_ID=1 python run_jupiter_dry.py -c jupiter_dry_1000.yaml --output-dir ./output_260513_dry_1000 > log.260513_dry_1000 &
