import os
import re

# === 配置 ===
folder_path = "C:/Users/15196/Desktop/Paper/AAAI 2026/code_update/DifficultySampling/CMAB/GRPO/VRT"  # 👈 替换为你的 .sh 文件所在文件夹路径

# 正则表达式：匹配绝对路径（以 / 开头，包含多个 / 的字符串）
path_pattern = re.compile(r'(/\S+?)(?=\s|$)')

def replace_path(match):
    full_path = match.group(1)
    basename = os.path.basename(full_path)
    return f"path/to/your/{basename}"

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Step 1: 替换所有路径
    content = path_pattern.sub(replace_path, content)

    # Step 2: 强制设置 MAX_PIXELS=1254400
    content = re.sub(r'MAX_PIXELS=\d+', 'MAX_PIXELS=1254400', content)

    # Step 3: 清空 WANDB_API_KEY 的值
    content = re.sub(r'(WANDB_API_KEY=)\S*', r'\1', content)

    # 写回文件
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

def main():
    for filename in os.listdir(folder_path):
        if filename.endswith('.sh'):
            filepath = os.path.join(folder_path, filename)
            print(f"Processing: {filename}")
            process_file(filepath)
    print("✅ All .sh files processed.")

if __name__ == "__main__":
    main()