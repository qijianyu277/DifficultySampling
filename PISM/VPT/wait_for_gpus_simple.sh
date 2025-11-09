#!/bin/bash

# ==============================================================================
# 配置部分：请根据您的实际情况修改以下变量
# ==============================================================================

# 目标脚本的完整路径，当GPU空闲时将执行此脚本
TARGET_SCRIPT="/mnt/tenant-home_speed/ywr/Token_mask/code/run_GRPO_VPT_random_random2.sh"

# 需要空闲的GPU数量
REQUIRED_GPUS=8

# 检查GPU状态的间隔时间（秒）
CHECK_INTERVAL_SECONDS=30

# 日志文件路径，用于记录本监控脚本的运行状态
# 建议将其放在一个持久化的位置，以便查看历史记录
LOG_FILE="/mnt/tenant-home_speed/ywr/Token_mask/code/gpu_wait_and_run_simple.log"

# 判断GPU是否空闲的内存使用阈值（MiB）。
# 如果GPU利用率为0%且内存使用低于此阈值，则认为GPU空闲。
MEMORY_THRESHOLD_MB=100

# ==============================================================================
# 脚本逻辑开始
# ==============================================================================

# 函数：用于记录带时间戳的消息到日志文件
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
}

log_message "GPU等待脚本启动。"

# 主循环：持续检查GPU状态
while true; do
    log_message "正在检查GPU状态..."
    
    # 获取所有GPU的利用率和内存使用情况
    gpu_status_output=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>&1)
    
    # 检查nvidia-smi命令是否成功执行
    if [ $? -ne 0 ]; then
        log_message "错误: 无法执行 nvidia-smi。请检查NVIDIA驱动和工具是否安装正确。错误信息: $gpu_status_output"
        sleep "$CHECK_INTERVAL_SECONDS"
        continue
    fi

    idle_gpus=0
    total_gpus=0

    IFS=$'\n'
    for line in $gpu_status_output; do
        total_gpus=$((total_gpus + 1))
        utilization=$(echo "$line" | cut -d',' -f1 | xargs)
        memory_used=$(echo "$line" | cut -d',' -f2 | sed 's/MiB//' | xargs)

        if (( $(echo "$utilization == 0" | bc -l) )) && (( $(echo "$memory_used < $MEMORY_THRESHOLD_MB" | bc -l) )); then
            idle_gpus=$((idle_gpus + 1))
        fi
    done
    unset IFS

    log_message "当前可用GPU数量: $idle_gpus / $total_gpus"

    # 判断是否满足启动条件
    if [ "$idle_gpus" -ge "$REQUIRED_GPUS" ]; then
        log_message "检测到 $idle_gpus 个空闲GPU，满足 $REQUIRED_GPUS 个的需求。正在启动目标脚本..."
        
        # 执行目标脚本
        nohup "$TARGET_SCRIPT" > "${TARGET_SCRIPT%.sh}.log" 2>&1 &
        
        log_message "目标脚本 '$TARGET_SCRIPT' 已在后台启动。其输出将记录到 '${TARGET_SCRIPT%.sh}.log'。"
        break # 启动后，退出本监控脚本
    else
        log_message "空闲GPU数量不足。等待 $CHECK_INTERVAL_SECONDS 秒后再次检查..."
        sleep "$CHECK_INTERVAL_SECONDS"
    fi
done

log_message "GPU等待脚本结束。"
