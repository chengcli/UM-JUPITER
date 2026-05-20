#! /bin/bash

# H2O-NH3 moist 2d + 3d
# dart9
#DEVICE_ID=0 python -u run_jupiter_crm.py -c jupiter_crm2d_H2O-NH3_7.5.yaml --output-dir ./260518_crm2d_H2O-NH3_7.5 &> log.260518_crm2d_H2O-NH3_7.5 &
#DEVICE_ID=1 python -u run_jupiter_crm.py -c jupiter_crm3d_H2O-NH3_7.5.yaml --output-dir ./260518_crm3d_H2O-NH3_7.5 &> log.260518_crm3d_H2O-NH3_7.5 &

# dart10
#DEVICE_ID=0 python -u run_jupiter_crm.py -c jupiter_crm2d_H2O-NH3_100.yaml --output-dir ./260518_crm2d_H2O-NH3_100 &> log.260518_crm2d_H2O-NH3_100 &
#DEVICE_ID=1 python -u run_jupiter_crm.py -c jupiter_crm3d_H2O-NH3_100.yaml --output-dir ./260518_crm3d_H2O-NH3_100 &> log.260518_crm3d_H2O-NH3_100 &

# dart11
#DEVICE_ID=0 python -u run_jupiter_crm.py -c jupiter_crm2d_H2O-NH3_1000.yaml --output-dir ./260518_crm2d_H2O-NH3_1000 &> log.260518_crm2d_H2O-NH3_1000 &
#DEVICE_ID=1 python -u run_jupiter_crm.py -c jupiter_crm3d_H2O-NH3_1000.yaml --output-dir ./260518_crm3d_H2O-NH3_1000 &> log.260518_crm3d_H2O-NH3_1000 &

# H2O-NH3-H2S moist + dry
# dungeon1
#DEVICE_ID=0 python -u run_jupiter_crm.py -c jupiter_crm_7.5.yaml --output-dir ./output_260520_crm_7.5 &> log.260520_crm_7.5 &
#DEVICE_ID=1 python -u run_jupiter_dry.py -c jupiter_dry_7.5.yaml --output-dir ./output_260520_dry_7.5 &> log.260520_dry_7.5 &

# dungeon2
#DEVICE_ID=0 python -u run_jupiter_crm.py -c jupiter_crm_100.yaml --output-dir ./output_260520_crm_100 &> log.260520_crm_100 &
#DEVICE_ID=1 python -u run_jupiter_dry.py -c jupiter_dry_100.yaml --output-dir ./output_260520_dry_100 &> log.260520_dry_100 &

# dungeon3
DEVICE_ID=0 python -u run_jupiter_crm.py -c jupiter_crm_1000.yaml --output-dir ./output_260520_crm_1000 &> log.260520_crm_1000 &
DEVICE_ID=1 python -u run_jupiter_dry.py -c jupiter_dry_1000.yaml --output-dir ./output_260520_dry_1000 &> log.260520_dry_1000 &
