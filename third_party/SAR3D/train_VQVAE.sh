#!/bin/bash

lpips_lambda=0.8
ssim_lambda=0.
l1_lambda=1.
l2_lambda=0

patchgan_disc_factor=0.025
patchgan_disc_g_weight=0.025

image_size=256
num_workers=6
image_size_encoder=256
patch_size=14
kl_lambda=0

patch_rendering_resolution=130

# External data directory, number of GPUs, batch size, and log directory input
data_dir=$1
NUM_GPUS=$2
batch_size=$3
logdir=$4

# this should be smaller if OOM
microbatch=12

DATASET_FLAGS="
 --data_dir ${data_dir} \
 --eval_data_dir ${data_dir} \
"

conv_lr=1e-4
lr=8e-5

vit_decoder_lr=$lr
encoder_lr=${conv_lr}
triplane_decoder_lr=$conv_lr
super_resolution_lr=$conv_lr

LR_FLAGS="--encoder_lr $encoder_lr \
--vit_decoder_lr $vit_decoder_lr \
--triplane_decoder_lr $triplane_decoder_lr \
--super_resolution_lr $super_resolution_lr \
--lr $lr"

TRAIN_FLAGS="--iterations 10001 --anneal_lr False \
 --batch_size $batch_size --save_interval 10000 \
 --microbatch ${microbatch} \
 --image_size_encoder $image_size_encoder \
 --dino_version mv-sd-dit-dynaInp-trilatent \
 --sr_training False \
 --cls_token False \
 --weight_decay 0.05 \
 --image_size $image_size \
 --kl_lambda ${kl_lambda} \
 --no_dim_up_mlp True \
 --uvit_skip_encoder False \
 --fg_mse True \
 --bg_lamdba 1.0 \
 --lpips_delay_iter 100 \
 --sr_delay_iter 25000 \
 --kl_anneal True \
 --symmetry_loss False \
 --vae_p 2 \
 --plucker_embedding True \
 --encoder_in_channels 10 \
 --arch_dit_decoder DiT2-B/2 \
 --sd_E_ch 64 \
 --sd_E_num_res_blocks 1 \
 --lrm_decoder True \
 "

SR_TRAIN_FLAGS_v1_2XC="
--decoder_in_chans 32 \
--out_chans 96 \
--alpha_lambda 1.0 \
--vq_loss_lambda 1.0 \
--logdir $logdir \
--arch_encoder vits \
--arch_decoder vitb \
--vit_decoder_wd 0.001 \
--encoder_weight_decay 0.001 \
--color_criterion mse \
--decoder_output_dim 3 \
--ae_classname vit.vit_triplane.ft \
"

SR_TRAIN_FLAGS=${SR_TRAIN_FLAGS_v1_2XC}

mkdir -p "$logdir"/
cp "$0" "$logdir"/

export LC_ALL=en_US.UTF-8
export OPENCV_IO_ENABLE_OPENEXR=1
export OMP_NUM_THREADS=12
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_GID_INDEX=3

MASTER_PORT=29500

NUM_NODES=1

export LAUNCHER="torchrun --nproc_per_node=$NUM_GPUS \
    --nnodes=$NUM_NODES \
    --rdzv-endpoint=localhost:2950 \
    --rdzv_backend=c10d \
    "
export SCRIPT="train_VQVAE.py"
export SCRIPT_ARGS=" \
    --trainer_name nv_rec_patch_mvE_disc \
    --num_workers ${num_workers} \
    ${TRAIN_FLAGS}  \
    ${SR_TRAIN_FLAGS} \
    ${DATASET_FLAGS} \
    --lpips_lambda $lpips_lambda \
    --overfitting False \
    --load_pretrain_encoder False \
    --iterations 5000001 \
    --save_interval 5000 \
    --eval_interval 250000000 \
    --log_interval 50 \
    --decomposed True \
    --logdir $logdir \
    --decoder_load_pretrained False \
    --cfg objverse_tuneray_aug_resolution_64_64_auto \
    --patch_size ${patch_size} \
    --use_amp True \
    --eval_batch_size 1 \
    ${LR_FLAGS} \
    --l1_lambda ${l1_lambda} \
    --l2_lambda ${l2_lambda} \
    --ssim_lambda ${ssim_lambda} \
    --depth_smoothness_lambda 0 \
    --use_conf_map False \
    --objv_dataset True \
    --depth_lambda 0.5 \
    --patch_rendering_resolution ${patch_rendering_resolution} \
    --use_lmdb_compressed False \
    --use_lmdb False \
    --mv_input True \
    --split_chunk_input True \
    --append_depth True \
    --patchgan_disc_factor ${patchgan_disc_factor} \
    --patchgan_disc_g_weight ${patchgan_disc_g_weight} \
    --use_wds False \
    "

export CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS" 
$CMD

# Usage example:
# . train_VAE.sh /path/to/data 4 1 ./log/