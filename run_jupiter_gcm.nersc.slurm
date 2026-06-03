#!/bin/bash
#SBATCH --job-name=jup
#SBATCH --account=m1411_g
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --time=4:00:00
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
export GPUS_PER_NODE=2
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

# ---- Launch one torchrun per node ----
DATE_STR=$(date +%y%m%d)
EXP_ROOT=${HOME}/scratch/2026.JupiterCRM/
EXP_NAME=gcm_H2O-NH3_F100
EXP_INP=${SLURM_JOB_NAME}_${EXP_NAME}.yaml
EXP_DIR=${DATE_STR}_${EXP_NAME}

cleanup() {
  echo "Moving job input/output files to ${EXP_ROOT}/${EXP_DIR}"
  mv "${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" "${EXP_ROOT}/${EXP_DIR}/" 2>/dev/null || true
  mv "${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err" "${EXP_ROOT}/${EXP_DIR}/" 2>/dev/null || true
  cp "${EXP_INP}" "${EXP_DIR}/" 2>/dev/null || true
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

srun torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${GPUS_PER_NODE}" \
  --node_rank=${SLURM_PROCID} \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  run_jupiter_moist.py --config ${EXP_INP} --output-dir ${EXP_ROOT}/${EXP_DIR}
