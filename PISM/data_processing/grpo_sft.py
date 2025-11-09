import json
import os
import re
import glob # 导入 glob 模块用于文件路径匹配

def convert_single_jsonl_file(input_filepath: str, output_dir: str, start_id_counter: int = 0):
    """
    将单个特定格式的JSONL文件转换为目标格式。
    
    Args:
        input_filepath (str): 输入JSONL文件的完整路径。
        output_dir (str): 输出JSONL文件应保存的目录。
        start_id_counter (int): 用于生成ID的起始计数器，以便在处理多个文件时保持ID的唯一性。
                                 每次处理新文件时，计数器会从这个值开始递增。
    
    Returns:
        tuple: (list: 转换后的样本列表, int: 最终的ID计数器值)
    """
    converted_samples = []
    
    try:
        with open(input_filepath, 'r', encoding='utf-8') as infile:
            for line_num, line in enumerate(infile, 1):
                try:
                    original_data = json.loads(line.strip())

                    # 生成一个递增的ID
                    # 这里使用传入的 start_id_counter，并在每次循环中递增
                    generated_id = f"vrt_sft_{start_id_counter}"

                    # 提取原始 images 路径
                    images = original_data.get("images", [])

                    # 构建 conversations 列表
                    conversations = []

                    # 处理 user 消息
                    original_messages = original_data.get("messages", [])
                    for msg in original_messages:
                        if msg.get("role") == "user":
                            user_content = msg.get("content", "")
                            # 确保 <image> 标签位于开头
                            if "<image>" not in user_content:
                                user_content = "<image> " + user_content
                            else:
                                user_content = "<image> " + user_content.replace("<image>", "").strip()
                            
                            conversations.append({
                                "from": "human",
                                "value": user_content
                            })

                    # 处理 solution 作为 gpt 的回复
                    solution_content = original_data.get("solution", "")
                    # 移除 \boxed{} 结构
                    match = re.search(r'\\boxed{(.*?)}', solution_content)
                    if match:
                        gpt_value = match.group(1).strip()
                    else:
                        gpt_value = solution_content.strip() # 如果没有 \boxed{}，则直接用原始solution

                    conversations.append({
                        "from": "gpt",
                        "value": gpt_value
                    })

                    # 构建最终的字典
                    converted_sample = {
                        "id": generated_id,
                        "conversations": conversations,
                        "images": images
                    }
                    converted_samples.append(converted_sample)
                    start_id_counter += 1 # 每次成功处理一个样本，递增计数器

                except json.JSONDecodeError as e:
                    print(f"警告: 文件 '{input_filepath}' 第 {line_num} 行 JSON 解析失败: {e}. 跳过此行。")
                except KeyError as e:
                    print(f"警告: 文件 '{input_filepath}' 第 {line_num} 行缺少预期的键: {e}. 跳过此行。")
                except Exception as e:
                    print(f"警告: 文件 '{input_filepath}' 第 {line_num} 行处理时发生未知错误: {e}. 跳过此行。")

    except FileNotFoundError:
        print(f"错误: 输入文件未找到 -> {input_filepath}")
        return [], start_id_counter # 返回空列表和当前计数器
    except Exception as e:
        print(f"错误: 读取或处理文件 '{input_filepath}' 时发生问题: {e}")
        return [], start_id_counter # 返回空列表和当前计数器

    return converted_samples, start_id_counter


def convert_jsonl_folder(input_folder: str, output_dir: str):
    """
    遍历输入文件夹中的所有JSONL文件，将它们转换并保存到指定输出目录。

    Args:
        input_folder (str): 包含JSONL文件的输入文件夹路径。
        output_dir (str): 转换后的JSONL文件应保存的目录。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    all_jsonl_files = glob.glob(os.path.join(input_folder, "*.jsonl"))
    if not all_jsonl_files:
        print(f"在 '{input_folder}' 中未找到任何 .jsonl 文件。")
        return
    # all_jsonl_files = ["/mnt/tenant-home_speed/ywr/Token_mask/Datasets/VPT_GRPO/random_VPT_GRPO.jsonl"]
    # 初始化一个全局的ID计数器，确保所有文件中的ID是连续且唯一的
    global_id_counter = 0
    total_converted_count = 0
    # all_jsonl_files = ["/mnt/tenant-home_speed/ywr/Token_mask/Datasets/VPT_GRPO/random_VPT_GRPO_2.jsonl"]
    for input_file in all_jsonl_files:
        print(f"\n正在处理文件: {input_file}")
        
        # 为当前文件生成输出文件名
        output_filename = os.path.basename(input_file).replace('.jsonl', '_converted.jsonl')
        current_output_filepath = os.path.join(output_dir, output_filename)

        # 调用转换单个文件的函数，并传递当前的全局ID计数器
        converted_data_for_file, global_id_counter = convert_single_jsonl_file(input_file, output_dir, global_id_counter)
        
        if converted_data_for_file:
            try:
                with open(current_output_filepath, 'w', encoding='utf-8') as outfile:
                    for sample in converted_data_for_file:
                        outfile.write(json.dumps(sample, ensure_ascii=False) + '\n')
                print(f"成功转换并保存 {len(converted_data_for_file)} 条数据到:\n{current_output_filepath}")
                total_converted_count += len(converted_data_for_file)
            except Exception as e:
                print(f"错误: 无法写入输出文件 '{current_output_filepath}': {e}")
        else:
            print(f"文件 '{input_file}' 未生成任何可转换数据。")

    print(f"\n--- 所有文件处理完成。总共转换了 {total_converted_count} 条数据。---")


if __name__ == "__main__":
    # 定义输入文件夹路径和输出目录
    input_folder_path = "/mnt/tenant-home_speed/ywr/Token_mask/Datasets_0724/VPT_GRPO" # <--- 这里是输入文件夹
    output_directory = "/mnt/tenant-home_speed/ywr/Token_mask/Datasets_0724/VPT_SFT" # <--- 这里是输出文件夹

    convert_jsonl_folder(input_folder_path, output_directory)