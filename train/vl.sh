uid="$(date +%Y%m%d_%H%M%S)"
base_model="Qwen/Qwen2.5-VL-7B-Instruct"
lr=1e-5
min_lr=0
epochs=1
micro_batch_size=2 # -> batch_size will be 16 if 8 gpus
push_to_hub=false
gradient_accumulation_steps=4
max_steps=-1
gpu_count=$(nvidia-smi -L | wc -l)

export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_P2P_DISABLE=1

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml \
    train/sft.py \
    --per_device_train_batch_size=${micro_batch_size} \
    --per_device_eval_batch_size=${micro_batch_size} \
    --gradient_accumulation_steps=${gradient_accumulation_steps} \
    --num_train_epochs=${epochs} \
    --max_steps=${max_steps} \
    --train_file_path="./data/train.jsonl" \
    --model_name=${base_model} \
    --warmup_ratio=0.05 \
    --bf16=True \
    --eval_strategy="epoch" \
    --logging_steps=1 \
    --lr_scheduler_type="cosine" \
    --learning_rate=${lr} \
    --weight_decay=1e-4 \
    --adam_beta1=0.9 \
    --adam_beta2=0.95 \
    --output_dir="ckpts/flex_vl_7b_${uid}" \
    --save_only_model=True \
    --gradient_checkpointing=True \
    --save_strategy=no \
    --dataset_text_field="text" \
    --block_size 4096