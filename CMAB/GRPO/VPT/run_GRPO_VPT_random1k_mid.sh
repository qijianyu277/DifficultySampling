WANDB_MODE=offline MAX_PIXELS=1254400 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
WANDB_API_KEY=580cc68602c01fb2029b4c1e32de7e7477733f36 \
NPROC_PER_NODE=8 \
swift rlhf \
    --rlhf_type grpo \
    --model /mnt/tenant-home_speed/qjy/LLaMA-Factory/saves/SFT_VPT_random1k/full/sft/checkpoint-40/ \
    --external_plugins /mnt/tenant-home_speed/qjy/ms-swift/examples/train/grpo/plugin/plugin.py \
    --reward_funcs visual_perception_accuracy \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset /mnt/tenant-home_speed/qjy/attention_VPT_eval/swift_output_grpo/VPT_0.1-0.4_and_1.6-1.9.jsonl \
    --max_length 2048 \
    --max_completion_length 1024 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --learning_rate 5e-7 \
    --gradient_accumulation_steps 2 \
    --eval_steps 100 \
    --save_steps 40 \
    --save_total_limit 30 \
    --logging_steps 1 \
    --output_dir output/sft_grpo/random1k_mid \
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