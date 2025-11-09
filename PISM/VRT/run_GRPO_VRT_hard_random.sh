WANDB_MODE=offline MAX_PIXELS=1254400 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
WANDB_API_KEY=53ee94cdd1d6d0e4f4aa9e171c9da9169ab6c1db \
NPROC_PER_NODE=8 \
swift rlhf \
    --rlhf_type grpo \
    --model /mnt/tenant-home_speed/ywr/LLamafactory/LLaMA-Factory-0.9.3/saves/qwen2.5_vl_VRT_difficult_0724/full/sft \
    --external_plugins /mnt/tenant-home_speed/ywr/Token_mask/code/plugin.py \
    --reward_funcs external_r1v_acc \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset /mnt/tenant-home_speed/ywr/Token_mask/Datasets_0724/VRT_GRPO/random_VRT_GRPO_1k.jsonl \
    --max_length 2048 \
    --max_completion_length 1024 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --learning_rate 5e-7 \
    --gradient_accumulation_steps 2 \
    --eval_steps 100 \
    --save_steps 20 \
    --save_total_limit 40 \
    --logging_steps 1 \
    --output_dir /mnt/tenant-home_speed/ywr/Token_mask/models/RL_models \
    --warmup_ratio 0.01 \
    --dataloader_num_workers 0 \
    --num_generations 8 \
    --temperature 1.0 \
    --system 'You are a helpful assistant specialized in visual perception tasks. Always begin by thinking through the problem step-by-step and explain your reasoning clearly inside <think> and </think> tags. Then, present your final answer strictly within <answer> and </answer> tags. Do not include any other text outside these tags.' \
    --deepspeed zero3 \
    --log_completions true \
    --report_to wandb \
    --beta 0.001 \
    --num_iterations 1