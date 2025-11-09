MASTER_PORT=29501 WANDB_MODE=offline MAX_PIXELS=3120320 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
WANDB_API_KEY=580cc68602c01fb2029b4c1e32de7e7477733f36 \
NPROC_PER_NODE=8 \
swift rlhf \
    --rlhf_type grpo \
    --model /mnt/shared/wireless_robot/zd/qjy/SFT_GRPO/VRT/saves/SFT_VRT_hard/checkpoint-85 \
    --external_plugins /mnt/tenant-home_speed/qjy/ms-swift/examples/train/grpo/plugin/plugin.py \
    --reward_funcs external_r1v_acc \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset /mnt/shared/wireless_robot/zd/qjy/attention_VRT_eval/swift_output_grpo/VRT_0.1-0.4_and_1.6-1.9.jsonl  \
    --max_length 2048 \
    --max_completion_length 1024 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 2 \
    --eval_steps 200 \
    --save_steps 30 \
    --save_total_limit 30 \
    --logging_steps 1 \
    --output_dir output/mid_hard \
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