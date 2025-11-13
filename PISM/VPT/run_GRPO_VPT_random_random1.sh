WANDB_MODE=offline MAX_PIXELS=1254400 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
WANDB_API_KEY= \
NPROC_PER_NODE=8 \
swift rlhf \
    --rlhf_type grpo \
    --model path/to/your/sft \
    --external_plugins path/to/your/plugin.py \
    --reward_funcs visual_perception_accuracy \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset path/to/your/random_VPT_GRPO_4_9k.jsonl \
    --max_length 2048 \
    --max_completion_length 1024 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --learning_rate 5e-7 \
    --gradient_accumulation_steps 2 \
    --eval_steps 500 \
    --save_steps 100 \
    --save_total_limit 40 \
    --logging_steps 1 \
    --output_dir path/to/your/RL_models \
    --warmup_ratio 0.01 \
    --dataloader_num_workers 0 \
    --num_generations 8 \
    --temperature 1.0 \
    --system 'You are a helpful assistant specialized in visual reasoning tasks. Always begin by thinking through the problem step-by-step and explain your reasoning clearly inside <think> and </think> tags. Then, present your final answer strictly within <answer> and </answer> tags. Do not include any other text outside these tags.' \
    --deepspeed zero3 \
    --log_completions true \
    --report_to wandb \
    --beta 0.001 \
    --num_iterations 1