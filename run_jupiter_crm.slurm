#!/bin/bash
#SBATCH --job-name=jcrm
#SBATCH --account=atreya0
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=48:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# ---- User settings ----
PORT=29500
source ${HOME}/pyenv/bin/activate

# ---- Derive rendezvous info from Slurm allocation ----
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="${PORT}"
export NNODES="${SLURM_JOB_NUM_NODES}"
export GPUS_PER_NODE=1
export WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "NNODES=${NNODES}  GPUS_PER_NODE=${GPUS_PER_NODE}  WORLD_SIZE=${WORLD_SIZE}"
echo "NODELIST=${SLURM_JOB_NODELIST}"

# ---- Helpful NCCL/transport knobs (adjust to your cluster) ----
# export NCCL_DEBUG=INFO
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_IB_DISABLE=1                 # uncomment if no InfiniBand
# export NCCL_SOCKET_IFNAME=ib0            # or eno1/bond0/etc
# export GLOO_SOCKET_IFNAME=ib0

# ---- Launch one torchrun per node; torchrun spawns 2 procs on that node ----
srun torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${GPUS_PER_NODE}" \
  --node_rank=${SLURM_PROCID} \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  run_jupiter_moist.py \
    --config jupiter_crm2d_H2O-NH3_F100_nu1000.0.yaml \
    --output-dir /home/chengcli/chengcli0/chengcli/2026.JupiterCRM/260602_crm2d_F100_nu1000.0
