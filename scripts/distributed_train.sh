#!/bin/bash
set -ex

# SageMaker injects these env vars automatically
NNODES=$SM_NUM_NODES
NODE_RANK=$SM_CURRENT_NODE_RANK
NPROC_PER_NODE=$SM_NUM_GPUS
MASTER_ADDR=$SM_MASTER_ADDR
MASTER_PORT=$SM_MASTER_PORT

torchrun \
  --nproc_per_node=$NPROC_PER_NODE \
  --nnodes=$NNODES \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  main.py "$@"
