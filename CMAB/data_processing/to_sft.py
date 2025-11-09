import os
import json

def convert_single_file(input_path, output_dir):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成输出文件名
    output_filename = os.path.basename(input_path)
    output_path = os.path.join(output_dir, output_filename)

    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            data = json.loads(line.strip())

            # 提取 prompt 和 solution 字段
            prompt = data.get("prompt", "")
            solution = data.get("solution", "")

            # 提取 image_paths 并映射到 images
            images = data.get("image_paths", [])

            # 构造 messages
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": solution}
            ]

            # 构建最终输出字典
            output = {
                "messages": messages,
            }

            # 如果有图像路径，加上 images 字段
            if images:
                output["images"] = images

            # 写入文件
            fout.write(json.dumps(output, ensure_ascii=False) + "\n")

    print(f"文件 {input_path} 转换完成，结果已写入 {output_path}")


def batch_convert(input_dir, output_dir):
    """
    遍历指定目录下的所有 .jsonl 文件并进行格式转换。
    """
    # 检查输入目录是否存在
    if not os.path.isdir(input_dir):
        print(f"错误：输入目录 {input_dir} 不存在")
        return
    
    # 获取目录中所有的 .jsonl 文件
    files = [f for f in os.listdir(input_dir) if f.endswith('.jsonl')]
    
    if not files:
        print("没有找到任何 .jsonl 文件")
        return
    
    # 对每个文件进行转换
    for file_name in files:
        input_path = os.path.join(input_dir, file_name)
        convert_single_file(input_path, output_dir)


if __name__ == "__main__":
    input_directory = "output_selected"       # 替换为输入目录路径
    output_directory = "swift_output_sft"   # 输出目录路径

    batch_convert(input_directory, output_directory)