#!/bin/bash
#SBATCH --job-name=jup
#SBATCH --account=atreya0
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=24:00:00
#SBATCH --requeue
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
DATE_STR=$(date +%y%m%d)
EXP_ROOT=/home/chengcli/chengcli0/chengcli/2026.JupiterCRM/
EXP_NAME=crm2d_H2O-NH3_F100_nu1000.0
EXP_INP=${SLURM_JOB_NAME}_${EXP_NAME}.yaml
EXP_DIR=${DATE_STR}_${EXP_NAME}

REQUEUE_REQUESTED=0

final_cleanup() {
  status=$?

  echo "Final cleanup with exit status ${status}"

  echo "Moving job input/output files to ${EXP_ROOT}/${EXP_DIR}"
  mkdir -p "${EXP_ROOT}/${EXP_DIR}"

  mv "${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" \
     "${EXP_ROOT}/${EXP_DIR}/" 2>/dev/null || true

  mv "${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err" \
     "${EXP_ROOT}/${EXP_DIR}/" 2>/dev/null || true

  cp "${EXP_INP}" "${EXP_ROOT}/${EXP_DIR}/" 2>/dev/null || true

  if [[ "${REQUEUE_REQUESTED}" -eq 1 ]]; then
    echo "Job was interrupted; saving checkpoint and requeuing..."

    # Put your checkpoint command here if needed.
    # Example:
    # python save_checkpoint.py

    scontrol requeue "${SLURM_JOB_ID}"
  else
    echo "Job exited normally or with non-requeue failure; not requeuing."
  fi

  exit "${status}"
}

handle_term() {
  echo "Received TERM signal"
  REQUEUE_REQUESTED=1
  exit 143
}

handle_int() {
  echo "Received INT signal"
  exit 130
}

trap final_cleanup EXIT
trap handle_int INT
trap handle_term TERM

echo "Restart count: ${SLURM_RESTART_COUNT:-0}"

srun torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${GPUS_PER_NODE}" \
  --node_rank=${SLURM_PROCID} \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  run_jupiter_moist.py --config ${EXP_INP} \
                       --restart ${SLURM_JOB_NAME}_${EXP_NAME}.final.restart \
                       --output-dir ${EXP_ROOT}/${EXP_DIR}
