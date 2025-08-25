#!/usr/bin/env bash
set -euo pipefail

pip install -r requirements.txt

# --- Read SageMaker env (present on all hosts) ---
HOSTS_JSON="${SM_HOSTS:-[]}"
CURRENT_HOST="${SM_CURRENT_HOST:-localhost}"
NUM_GPUS="${SM_NUM_GPUS:-1}"

# --- Derive topology robustly ---
readarray -t HOSTS < <(python - <<'PY'
import os, json
hosts = json.loads(os.environ.get("SM_HOSTS","[]"))
for h in (hosts or []):
    print(h)
PY
)

if [[ ${#HOSTS[@]} -eq 0 ]]; then
  NNODES=1
  MASTER_ADDR="127.0.0.1"
  NODE_RANK=0
else
  NNODES=${#HOSTS[@]}
  MASTER_ADDR="${HOSTS[0]}"
  # find index of CURRENT_HOST
  NODE_RANK=$(python - <<'PY'
import os, json, sys
hosts = json.loads(os.environ.get("SM_HOSTS","[]"))
cur = os.environ.get("SM_CURRENT_HOST","localhost")
print(hosts.index(cur) if cur in hosts else 0)
PY
)
fi

# GPUs per node
NPROC_PER_NODE="${NUM_GPUS}"

# Networking knobs (safe defaults)
export MASTER_PORT="${MASTER_PORT:-29500}"
export NCCL_SOCKET_IFNAME="${SM_NETWORK_INTERFACE_NAME:-eth0}"
export GLOO_SOCKET_IFNAME="${SM_NETWORK_INTERFACE_NAME:-eth0}"
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONUNBUFFERED=1
export TORCH_NCCL_BLOCKING_WAIT=0
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

echo "[launcher] hosts=${HOSTS_JSON} current=${CURRENT_HOST} nnodes=${NNODES} node_rank=${NODE_RANK} gpus_per_node=${NPROC_PER_NODE} master=${MASTER_ADDR}:${MASTER_PORT}"

# Sanity checks
if [[ -z "${MASTER_ADDR}" || -z "${MASTER_PORT}" ]]; then
  echo "[launcher][fatal] MASTER_ADDR/PORT not set"; exit 2
fi
if ! [[ "${NNODES}" =~ ^[0-9]+$ && "${NODE_RANK}" =~ ^[0-9]+$ && "${NPROC_PER_NODE}" =~ ^[0-9]+$ ]]; then
  echo "[launcher][fatal] bad topology: nnodes=${NNODES} node_rank=${NODE_RANK} nproc_per_node=${NPROC_PER_NODE}"; exit 3
fi

# Single-node fast path (helps local debugging)
if [[ "${NNODES}" -eq 1 ]]; then
  exec torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    main.py "$@"
else
  exec torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    main.py "$@"
fi
