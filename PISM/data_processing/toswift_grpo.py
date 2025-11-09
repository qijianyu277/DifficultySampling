import json
import os

def convert_to_swift_format(input_file, output_file):
    """
    将原始数据集转换为 Swift 框架要求的格式。
    
    :param input_file: 输入文件路径（.jsonl）
    :param output_file: 输出文件路径（.jsonl）
    """
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:

        for line in fin:
            try:
                data = json.loads(line.strip())

                # 构建 messages 字段
                messages = [{
                    "role": "user",
                    "content": data.get("prompt", "").strip()
                }]

                # 构建最终输出字典
                converted = {
                    "images": data.get("image_paths", []),
                    "messages": messages,
                    "solution": data.get("ground_truth", "").strip()
                }

                # 写入新文件
                fout.write(json.dumps(converted, ensure_ascii=False) + '\n')

            except Exception as e:
                print(f"❌ 转换失败：{str(e)}")
                continue

    print(f"✅ 转换完成，已写入：{output_file}")


def batch_convert_directory(input_dir, output_dir):
    """
    批量处理指定目录下的所有 .jsonl 文件，转换为 Swift 格式。
    
    :param input_dir: 输入目录（包含多个 jsonl 文件）
    :param output_dir: 输出目录（保持相同文件名结构）
    """
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"输入路径不是有效目录：{input_dir}")

    # 收集所有 .jsonl 文件
    jsonl_files = [f for f in os.listdir(input_dir) if f.endswith('.jsonl')]
    if not jsonl_files:
        print(f"⚠️ 目录中没有找到任何 .jsonl 文件：{input_dir}")
        return

    print(f"\n🔍 正在批量处理目录：{input_dir}")
    print(f"📊 共发现 {len(jsonl_files)} 个 JSONL 文件。\n")

    # 遍历每个文件进行转换
    for filename in jsonl_files:
        input_path = os.path.join(input_dir, filename)
        output_filename = filename  # 保留原文件名
        output_path = os.path.join(output_dir, output_filename)

        print(f"🔄 正在处理文件：{filename}")
        convert_to_swift_format(input_path, output_path)

    print("\n🎉 所有文件转换完成！")


if __name__ == '__main__':
    # 📁 设置输入目录和输出目录
    input_directory = "/mnt/tenant-home_speed/ywr/Token_mask/Datasets_0724/VPT"      # 替换为你自己的目录
    output_directory = "/mnt/tenant-home_speed/ywr/Token_mask/Datasets_0724/VPT_GRPO"        # 输出目录会自动创建

    if not os.path.exists(input_directory):
        print(f"❌ 错误：输入目录 '{input_directory}' 不存在")
    else:
        batch_convert_directory(input_directory, output_directory)