deepspeed --master_port=61601 \
  fastchat/train/train_mem.py \
  --model_name_or_path lmsys/vicuna-13b-v1.5 \
  --data_path "your data_path" \
  --bf16 True \
  --output_dir "your ckpt path" \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 2 \
  --save_strategy "steps" \
  --max_steps 500000 \
  --save_steps 200 \
  --save_total_limit 3 \
  --learning_rate 2e-4 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --deepspeed ds_config/stage3.json \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --lazy_preprocess True

