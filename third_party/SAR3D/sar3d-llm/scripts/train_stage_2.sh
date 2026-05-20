master_port=$((RANDOM % (65535 - 49152 + 1) + 49152))
filename="stage_2"

model_name_or_path=/mnt/ywchen/logs/PointLLM/release/ # Path to the output dir of stage 1 training
data_path=/mnt/ywchen/data/sar3d-dataset
anno_path=/cpfs01/user/lanyushi.p/Repo/cyw/log/PointLLM_brief_description_660K_filtered.json 
output_dir=/mnt/ywchen/logs/PointLLM/$filename




GPU_NUM=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
torchrun --nnodes=1 --nproc_per_node=5 --master_port=$master_port pointllm/train/train_mem.py \
    --model_name_or_path $model_name_or_path \
    --data_path $data_path \
    --anno_path $anno_path \
    --output_dir $output_dir \
    --version v1 \
    --model_max_length 2048 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 100 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --eval_steps 100 \
    --save_strategy "steps" \
    --save_steps 2400 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --bf16 True \
    --fix_llm False \
    --fix_pointnet True \
    --report_to wandb \
    --run_name $filename \
    --gradient_checkpointing True \
    --stage_2 True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --conversation_types "detailed_description" "single_round" "multi_round" \
    --use_color True