#!/bin/bash

# Required arguments
DEPTH=$1           # Depth of the model, e.g. 16, 24
BS=$2              # Batch size
NPROC_PER_NODE=$3  # Number of processes per node
VQVAE_PATH=$4      # Path to VQVAE checkpoint
OUT_DIR=$5         # Output directory path
DATA_DIR=$6        # Dataset directory path

# Check if all required arguments are provided
if [ -z "$DEPTH" ] || [ -z "$BS" ] || [ -z "$NPROC_PER_NODE" ] || [ -z "$VQVAE_PATH" ] || [ -z "$OUT_DIR" ] || [ -z "$DATA_DIR" ]; then
    echo "Usage: $0 DEPTH BS NPROC_PER_NODE VQVAE_PATH OUT_DIR DATA_DIR"
    exit 1
fi

torchrun --nproc_per_node=$NPROC_PER_NODE --nnodes=1 --rdzv-endpoint=localhost:2980 \
    --rdzv_backend=c10d train.py \
  --depth=$DEPTH --bs=$BS --ep=150 --fp16=2 --alng=1e-4 --wpe=0.1 \
  --vqvae_pretrained_path $VQVAE_PATH \
  --local_out_dir_path $OUT_DIR \
  --data_dir $DATA_DIR \
  --tblr 1e-4 \
  --num_workers 3