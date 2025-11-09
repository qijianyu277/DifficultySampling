from transformers import LlavaOnevisionProcessor, LlavaOnevisionForConditionalGeneration, GenerationConfig
import json
import os
import numpy as np
import math
import torch
import torch.nn as nn
from PIL import Image
import requests
import torch.nn.init as init
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from safetensors.torch import load_file
import os
from datasets import load_dataset
import pandas as pd
import io
import re
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from qwen_vl_utils import process_vision_info
import argparse
from tqdm import tqdm
from torchvision import transforms
import PIL
from PIL import Image
import matplotlib.pyplot as plt

eval_prompt_template = '''Please help me judge the correctness of the generated answer and the corresponding rationale. 
Question: {}
Ground truth answer: {}
Generated rationale and answer: {}
Your output should only be one sentence: the generated answer is true or false.'''

few_shot_cot_prompt = '''Answer the question **step by step** and provide the final answer at the end, each step should end with **<end>** and put your final answer within $\boxed{}$. Below are two examples:
Question: BoatsRUs built 7 canoes in January of this year and then each subsequent calendar month they built twice the number of canoes they had built the previous month. How many total canoes were built by BoatsRUs by the end of May of this year?
### Step1: To find the result of the total number of canoes built by BoatsRUs by the end of May, I need to find the number of canoes built in each month from January to May and then add them up. <end>
### Step2: To find the number of canoes built in each month, I need to use the formula for the number of canoes built in a given month, which is the number of canoes built in the previous month times 2. <end>
### Step3: So, the number of canoes built in January is 7, the number of canoes built in February is 7 times 2, which is 14, the number of canoes built in March is 14 times 2, which is 28, the number of canoes built in April is 28 times 2, which is 56, and the number of canoes built in May is 56 times 2, which is 112. <end>
### Step4: Now, I can add up these numbers to get the total number of canoes built by BoatsRUs by the end of May: 7 plus 14 plus 28 plus 56 plus 112, which is 217. <end>
### Final Answer: The answer is: $boxed{217}$.
Question: Find the number of blue circles in the figure.
### Step 1: To find the result of the number of blue circles, I need to interpret the figure. The figure is a Venn diagram with two labeled sets: - One set labeled "blue" contains all the shapes that are blue in color. - The other set labeled "circle" contains all the shapes that are circular in shape. The overlapping region of the Venn diagram contains shapes that are both blue and circular. <end>
### Step 2: The overlapping region contains shapes that meet both criteria: Blue color and Circle shape. From the diagram: - There is **one blue circle** in the overlapping region. <end>
### Final Answer: The answer is: $boxed{1}$.
Remember to answer the question **step by step**! Here is your question:
'''


def read_all_parquet_to_list(directory: str):
    parquet_files = [
        f for f in os.listdir(directory) if f.endswith(".parquet")
    ]

    df_list = []

    for parquet_file in parquet_files:
        file_path = os.path.join(directory, parquet_file)
        df = pd.read_parquet(file_path)
        df_list.append(df)

    if df_list:
        combined_df = pd.concat(df_list, ignore_index=True)
    else:
        return []

    data_list = combined_df.to_dict(orient='records')

    return data_list

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def dump_to_jsonl(obj: list[dict], path: str):
    with open(path, 'w') as file:
        file.writelines([json.dumps(x) + '\n' for x in obj])

class State:

    def __init__(self, image_feat, text_context, solution_steps=None):
        self.image_feat = image_feat
        self.text_context = text_context
        self.solution_steps = solution_steps if solution_steps else []
        self.is_terminal = False

    def copy(self):
        new_state = State(
            image_feat=self.image_feat,
            text_context=self.text_context,
            solution_steps=self.solution_steps.copy()
        )
        new_state.is_terminal = self.is_terminal
        return new_state

    def __repr__(self):
        return f"<State steps={len(self.solution_steps)}, terminal={self.is_terminal}>"


class Action:

    def __init__(self, text):
        self.text = text

    def __repr__(self):
        return f"<Action: {self.text}>"


from PIL import Image
import torch
from torchvision import transforms


def sanitize_filename(filename):
    """确保文件名中只包含安全字符"""
    return "".join([c if c.isalnum() or c in (' ', '.', '_') else '_' for c in filename]).rstrip()

class VLM_get_attn_ratio:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        # 假设每张图片会被编码成固定数量的 tokens, 这个值取决于模型配置
        self.VISION_TOKEN_LENGTH = 576  # 示例值，请根据你的模型调整, 万一之法, 后面并没有用到

    def vlm_get(self, image_feat, text_context, generation_config, history=None):
        prompt = text_context
        message = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image_feat}, 
            ],
        }]
        if history:
            message.append({
                "role": "assistant",
                "content": [{"type": "text", "text": "".join(history)}],
            })

        # 构建文本输入
        # text = self.processor.apply_chat_template(
        #     message, tokenize=False, add_generation_prompt=True
        # )[:-32]

        text = self.processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        
        # print("text: ", text)
        image_feat, video_inputs = process_vision_info(message)
        # 处理图像输入
        if isinstance(image_feat, list) and len(image_feat) > 0 and isinstance(image_feat[0], Image.Image):
            image_tensor = self.transform(image_feat[0]).unsqueeze(0)  # 添加批次维度
            # print("Transform to Tensor")
        else:
            raise ValueError("Expected image_feat to be a non-empty list of PIL.Images.")

        inputs = self.processor(
            text=[text],
            images=image_tensor,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        input_ids = inputs["input_ids"]

        # 将 input_ids 转换为 tokens
        input_tokens = self.processor.tokenizer.convert_ids_to_tokens(input_ids[0])

        # 打印每个 token 和对应的 ID
        # print("\n【输入 token 信息】")
        # for i, (token_id, token) in enumerate(zip(input_ids[0], input_tokens)):
        #     print(f"Position {i}: ID={token_id}, Token={token}")


        question_input_length = input_ids.shape[1]  # 整个上下文 token 长度
        # print("question_input_length: ", question_input_length)

        # 获取 vision_patch_token_id (如果适用)
        vision_patch_token_id = None
        special_tokens = self.processor.tokenizer.special_tokens_map
        vocab = self.processor.tokenizer.get_vocab()

        for token in ["<|image_pad|>"]:
            if token in vocab:
                # print("Find it！The token is: ", token)
                vision_patch_token_id = vocab[token]
                break
            elif token in special_tokens:
                # print("find it2")
                vision_patch_token_id = self.processor.tokenizer.convert_tokens_to_ids(special_tokens[token])
                break
        # print("vision_patch_token_id", vision_patch_token_id)
        image_token_indices = []
        if vision_patch_token_id is not None:
            # print("获取位置矩阵...")
            image_token_positions = (input_ids == vision_patch_token_id).nonzero(as_tuple=True)
            # print("image_token_positions", image_token_positions)
            image_token_indices = image_token_positions[1].tolist()
            # print("image_token_indices is not None: ", image_token_indices)
        
        if not image_token_indices:
            print("image_token_indices: ", image_token_indices)
            vision_output_length = self.VISION_TOKEN_LENGTH  # 根据模型确定图像token的数量
            image_start_idx = question_input_length - vision_output_length
            image_token_indices = list(range(image_start_idx, question_input_length))
        else:
            image_start_idx = image_token_indices[0]
            vision_output_length = len(image_token_indices)

        text_token_indices = list(range(0, image_start_idx))
        text_token_length = len(text_token_indices)

        # 模型生成
        generated_outputs = self.model.generate(
            **inputs,
            generation_config=generation_config,
            stop_strings=['<end>'],
            max_new_tokens=2048,
            tokenizer=self.processor.tokenizer,
            output_attentions=True,
            return_dict_in_generate=True
        )

        generated_ids = generated_outputs.sequences
        attentions = generated_outputs.attentions  # tuple of tensors: (layer, batch, head, gen_len, ctx_len)
        # print("attentions:", attentions)
        # print("attentions[0]", attentions[0])
        # print("attentions[0][1]: ", attentions[0][1])
        # print("attentions[0][1].shape: ", attentions[0][1].shape)
        # print("attentions[8] ", attentions[8])
        context_info = {
            "input_ids": input_ids,
            "text_token_indices": text_token_indices,
            "text_token_length": text_token_length,
            "image_token_indices": image_token_indices,
            "image_start_idx": image_start_idx,
            "vision_output_length": vision_output_length,
            "total_input_length": question_input_length,
            "generated_ids": generated_ids,
        }

        return attentions, context_info

    def analyze_attention_per_token(self, image_feat, text_context, generation_config):
        
        attentions, context_info = self.vlm_get(image_feat, text_context, generation_config)


        # print("attentions: ", attentions)
        # print("type(attentions)", type(attentions))
        # print("len(attentions): ", len(attentions))  # 生成的总token数
        # first_layer_output = attentions[0] # ([layer1, head, gen, clx], [layer2, head, gen, clx])这是第几个token
        # print("len(first_token_output)(层数): ", len(first_layer_output))  # 打印发现是28层 0-27
        
        # first_layer_output = attentions[0][15] # 重要！！！！！！！！这是第几个生成token的第几层,第0个生成token的第15层
        # print("(tensor)first_layer_output[0][0].shape: ", attentions[0][0].shape)
        # print("(tensor)first_layer_output[0][1].shape: ", attentions[0][1].shape)  # [1, head, gen, clx],注意第0个token的尺寸不是[1, head, 1, clx] 即：[层数，头数， gen， 列数（相对于当前生成token的输入序列的长度） ]，它的倒数第二维是等于列数的gen
        # print("(tensor)first_layer_output[0][2].shape: ", attentions[0][2].shape)
        # print("(tensor)first_layer_output[1][3].shape: ", attentions[1][3].shape)  # [1, head, 1, clx+1]
        # print("(tensor)first_layer_output[1][4].shape: ", attentions[1][4].shape)
        # print("(tensor)first_layer_output[1][5].shape: ", attentions[1][5].shape)
        # print("(tensor)first_layer_output[2][6].shape: ", attentions[2][6].shape)  # [1, head, 1, clx+2]
        # print("(tensor)first_layer_output[2][7].shape: ", attentions[2][7].shape)
        # print("(tensor)first_layer_output[2][8].shape: ", attentions[2][8].shape)
        # print("(tensor)first_layer_output[2][9].shape: ", attentions[2][9].shape)
        # print("(tensor)first_layer_output[0][10].shape: ", attentions[0][10].shape)
        # print("(tensor)first_layer_output[0][11].shape: ", attentions[0][11].shape)
        # print("(tensor)first_layer_output[0][12].shape: ", attentions[0][12].shape)
        # print("(tensor)first_layer_output[1][13].shape: ", attentions[1][13].shape)
        # print("(tensor)first_layer_output[1][14].shape: ", attentions[1][14].shape)
        # print("(tensor)first_layer_output[1][15].shape: ", attentions[1][15].shape)
        # print("(tensor)first_layer_output[2][16].shape: ", attentions[2][16].shape)
        # print("(tensor)first_layer_output[2][17].shape: ", attentions[2][17].shape)
        # print("(tensor)first_layer_output[2][18].shape: ", attentions[2][18].shape)
        # print("(tensor)first_layer_output[0][19].shape: ", attentions[0][19].shape)
        # print("(tensor)first_layer_output[0][20].shape: ", attentions[0][20].shape)
        # print("(tensor)first_layer_output[0][21].shape: ", attentions[0][21].shape)
        # print("(tensor)first_layer_output[1][22].shape: ", attentions[1][22].shape)
        # print("(tensor)first_layer_output[1][23].shape: ", attentions[1][23].shape)
        # print("(tensor)first_layer_output[1][24].shape: ", attentions[1][24].shape)
        # print("(tensor)first_layer_output[2][25].shape: ", attentions[2][25].shape)
        # print("(tensor)first_layer_output[2][26].shape: ", attentions[2][26].shape)
        # print("(tensor)first_layer_output[2][27].shape: ", attentions[2][27].shape)

        image_token_indices = context_info["image_token_indices"]
        text_token_indices = context_info["text_token_indices"]
        image_start_idx = context_info["image_start_idx"]

        total_image_text_attn_ratio = 0.0
        total_image_attn = 0.0
        total_text_attn = 0.0
        # print("len(attentions): ", len(attentions))
        visual_map = []  # 能用list来存储每个token在指定层的attention
        for token in range(1, len(attentions)):
            attn_weights = attentions[token] # 第几个token,包含了这个token在28个层里每个层的attention，每一层的尺寸是[1, head, 1, input_length+i]
            for layer in range(20, 21):
                attn_weight = attn_weights[layer]
                attn_weight = torch.mean(attn_weight, dim=1)[0][0] # [1, head, 1, input_length+i]先变成[1, 1, 1, input_length+i]再变成[input_length+i]
                visual_map.append(attn_weight.unsqueeze(0))
                image_attn = attn_weight[image_token_indices].sum().item()/len(image_token_indices) # 第token个token在第layer层对输入的视觉token的平均注意力分数
                text_attn = attn_weight[text_token_indices].sum().item()/len(text_token_indices) # 类似上面
                if text_attn == 0:
                    print("text_attn == 0")
                    continue
                image_text_attn_ratio = image_attn / text_attn
                # print(f"Text Attention={text_attn:f}, Image Attention={image_attn:f}, Image_text_ratio={image_text_attn_ratio:f}")

                total_image_attn += image_attn
                total_text_attn += text_attn
        total_image_text_attn_ratio = total_image_attn / total_text_attn
        # print("所有token在每个层上的视觉注意力分数相加，除以在每个层上的文本注意力总和，得到total_image_text_attn_ratio： ", total_image_text_attn_ratio)
        
        return total_image_text_attn_ratio, visual_map, image_start_idx



def pad_tensor(tensor, max_len, pad_value=1.0):
    """
    对单个 tensor 在 dim=1 上进行填充，使其长度为 max_len
    """
    current_len = tensor.size(1)
    if current_len < max_len:
        padding = torch.full((tensor.size(0), max_len - current_len),
                             fill_value=pad_value,
                             dtype=tensor.dtype,
                             device=tensor.device)
        return torch.cat([tensor, padding], dim=1)
    return tensor

def normalize_tensor(tensor):
    """
    将张量进行归一化处理，使得最小值变为0，最大值变为1。
    """
    min_val = tensor.min()
    max_val = tensor.max()
    if min_val == max_val:  # 避免除以零
        return torch.zeros_like(tensor)
    return (tensor - min_val) / (max_val - min_val)


def plot_attention_heatmap(visual_map, data_id, image_start_idx):
    if not visual_map:
        print("visual_map 为空，无法绘制热力图")
        return

    # 步骤 1: 确保所有 tensor 都在 CPU 上，并且进行归一化
    visual_map_cpu_normalized = [normalize_tensor(t.cpu()) for t in visual_map]

    # 步骤 2: 找最大长度
    max_len = max(t.size(1) for t in visual_map_cpu_normalized)

    # 步骤 3: 填充所有 tensor 到统一长度
    padded_tensors = [pad_tensor(t, max_len, pad_value=0.0) for t in visual_map_cpu_normalized]

    # 步骤 4: 合并 tensor
    attention_tensor = torch.cat(padded_tensors, dim=0)  # shape: (n_tokens, max_len)

    # 步骤 5: 转为 numpy 用于绘图
    attention_array = attention_tensor.numpy()

    # 步骤 6: 构建保存路径
    safe_data_id = sanitize_filename(data_id)
    output_dir = "/mnt/tenant-home_speed/qjy/attention_com/results_map"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{safe_data_id}_attn_heatmap.png")

    # 步骤 7: 绘图
    plt.figure(figsize=(12, 8))
    plt.imshow(attention_array, cmap='viridis', aspect='auto', interpolation='nearest', vmin=0.0, vmax=1.0)
    plt.colorbar(label='Attention Weight')
    plt.title("Normalized Attention Heatmap Across Tokens")
    plt.xlabel("Input Position")
    plt.ylabel("Token Index")

    # 使用 image_start_idx 在图像中标识出文本和图片的部分
    plt.axvline(x=image_start_idx, color='r', linestyle='--')  # 添加一条红色虚线
    plt.text(image_start_idx + 1, -2, 'Image Start', color='red', verticalalignment='center',
             horizontalalignment='left')  # 在虚线旁添加文本说明

    # 步骤 8: 保存图像
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"热力图已保存至: {output_path}")


# def plot_attention_heatmap(visual_map, data_id, image_start_idx):
#     if not visual_map:
#         print("visual_map 为空，无法绘制热力图")
#         return
#
#     # Step 1: 确保所有 tensor 都在 CPU 上，并且进行归一化
#     visual_map_cpu_normalized = [normalize_tensor(t.cpu()) for t in visual_map]
#
#     # Step 2: 找最大长度
#     max_len = max(t.size(1) for t in visual_map_cpu_normalized)
#
#     # Step 3: 填充所有 tensor 到统一长度
#     padded_tensors = [pad_tensor(t, max_len, pad_value=0.0) for t in visual_map_cpu_normalized]  # 注意这里我们使用0作为填充值
#
#     # Step 4: 合并 tensor
#     attention_tensor = torch.cat(padded_tensors, dim=0)  # shape: (n_tokens, max_len)
#
#     # Step 5: 转为 numpy 用于绘图
#     attention_array = attention_tensor.numpy()
#
#     # Step 6: 构建保存路径
#     safe_data_id = sanitize_filename(data_id)
#     output_dir = "/mnt/tenant-home_speed/qjy/attention_com/results_map"
#     os.makedirs(output_dir, exist_ok=True)
#     output_path = os.path.join(output_dir, f"{safe_data_id}_attn_heatmap.png")
#
#     # Step 7: 绘图
#     plt.figure(figsize=(12, 8))
#     plt.imshow(attention_array, cmap='viridis', aspect='auto', interpolation='nearest', vmin=0.0, vmax=1.0)
#     plt.colorbar(label='Attention Weight')
#     plt.title("Normalized Attention Heatmap Across Tokens")
#     plt.xlabel("Input Position")
#     plt.ylabel("Token Index")
#
#     # Step 8: 保存图像
#     plt.savefig(output_path, bbox_inches='tight', dpi=300)
#     plt.close()
#
#     print(f"热力图已保存至: {output_path}")




         
def Get_attention_ratio(image_data, text_prompt, model, generation_config, processor):

    image_feat = image_data
    vlm_attn = VLM_get_attn_ratio(model, processor)
    image_text_attention_ratio, visual_map, image_start_idx = vlm_attn.analyze_attention_per_token(image_feat, text_prompt, generation_config)

    return image_text_attention_ratio, visual_map, image_start_idx



class VisionLanguageModel:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    def _run_vlm(self, image_feat, text_context, generation_config, history=None):

        prompt = text_context
        message = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image_feat},
            ],
        }]
        if history:
            message.append({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "".join(history)}, ],
            })

        text = self.processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )[:-32]
        # image_inputs = Image.open(io.BytesIO(image_feat))
        image_inputs, video_inputs = process_vision_info(message)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        question_input_length = inputs['input_ids'].shape[1]

        generated_ids = self.model.generate(**inputs, generation_config=generation_config, stop_strings=['<end>'],
                                       max_new_tokens=2048, tokenizer=self.processor.tokenizer)
        output = self.processor.decode(
            generated_ids[0][question_input_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output


    def propose_actions(self, state, generation_config, top_k=3):

        actions = []
        for i in range(top_k):
            llama_output = self._run_vlm(
                image_feat=state.image_feat,
                text_context=state.text_context,
                generation_config=generation_config,
                history=state.solution_steps
            )
            action_text = llama_output
            prob = 1.0 / top_k
            actions.append((Action(action_text), prob))
        return actions

    def transition(self, state, action):
        next_state = state.copy()
        next_state.solution_steps.append(action.text)

        if len(next_state.solution_steps) >= 10 or "Final Answer: " in next_state.solution_steps[-1]:
            next_state.is_terminal = True
        return next_state

    def evaluate_terminal_state(self, state, eval_llm, eval_llm_tokenizer, question, answer):
        if state.is_terminal:
            simulation_response = "".join(state.solution_steps)
            prompt = eval_prompt_template.format(question, answer, simulation_response)

            messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            text = eval_llm_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = eval_llm_tokenizer([text], return_tensors="pt").to(eval_llm.device)

            generated_ids = eval_llm.generate(
                **model_inputs,
                max_new_tokens=512
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = eval_llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            if 'true' in response.split('.')[0]:
                return 1.0
            else:
                return 0.0
        return 0.0


class MCTSNode:
    def __init__(self, state):
        self.state = state
        self.children = {}  # dict(action -> MCTSNode)
        self.visit_count = 0
        self.value_sum = 0.0
        self.parent = None
        self.action_from_parent = None

    @property
    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


def ucb_score(parent, child, c_puct=1.0):
    if child.visit_count == 0:
        return float('inf')
    return (child.value
            + c_puct * math.sqrt(math.log(parent.visit_count) / (child.visit_count)))


def select_child(node, c_puct=1.0):
    best_score = -float('inf')
    best_action = None
    best_child = None
    for action, child in node.children.items():
        score = ucb_score(node, child, c_puct)
        if score > best_score:
            best_score = score
            best_action = action
            best_child = child
    return best_action, best_child


def expand(node, vlm, generation_config, top_k=3):
    if node.state.is_terminal:
        return
    actions_probs = vlm.propose_actions(node.state, generation_config, top_k)
    for action, prob in actions_probs:
        if action not in node.children:
            next_state = vlm.transition(node.state, action)
            child_node = MCTSNode(next_state)
            child_node.parent = node
            child_node.action_from_parent = action
            node.children[action] = child_node


def simulate(state, vlm, eval_llm, eval_llm_tokenizer, question, answer, generation_config, rollout_limit=10):
    temp_state = state.copy()
    steps = 0
    while not temp_state.is_terminal and steps < rollout_limit:
        actions_probs = vlm.propose_actions(temp_state, generation_config, top_k=1)
        action, prob = random.choice(actions_probs)
        temp_state = vlm.transition(temp_state, action)
        steps += 1

    return vlm.evaluate_terminal_state(temp_state, eval_llm, eval_llm_tokenizer, question, answer), temp_state


def backpropagate(node, reward):
    cur = node
    while cur is not None:
        cur.visit_count += 1
        cur.value_sum += reward
        cur = cur.parent

def mcts_search(root_state, vlm, eval_llm, eval_llm_tokenizer, question, answer, generation_config, n_iterations,
                c_puct=1.0, top_k=3):
    root_node = MCTSNode(root_state)
    solution = None

    for iter in range(n_iterations):
        node = root_node
        while not node.state.is_terminal and len(node.children) > 0:
            _, child = select_child(node, c_puct)
            node = child

        if not node.state.is_terminal:
            expand(node, vlm, generation_config, top_k=top_k)
            if len(node.children) > 0:
                action = random.choice(list(node.children.keys()))
                node = node.children[action]


        reward, simulate_state = simulate(node.state, vlm, eval_llm, eval_llm_tokenizer, question, answer,
                                          generation_config, rollout_limit=10)
        if reward == 1:
            solution = simulate_state
            break

        backpropagate(node, reward)

    best_path = []
    current = root_node
    while not current.state.is_terminal and len(current.children) > 0:
        best_child = max(current.children.values(), key=lambda c: c.visit_count)
        best_path.append(best_child.action_from_parent.text)
        current = best_child
    return root_node, best_path, solution, iter

def solve_math_reasoning_vlm(image_data, text_prompt, model, generation_config, processor, eval_llm,
                                eval_llm_tokenizer, question, answer, n_iterations):
    image_feat = image_data

    init_state = State(
        image_feat=image_feat,
        text_context=text_prompt,
        solution_steps=[]
    )

    vlm = VisionLanguageModel(model, processor)

    root, steps, solution, n_iter = mcts_search(
        root_state=init_state,
        vlm=vlm,
        eval_llm=eval_llm,
        eval_llm_tokenizer=eval_llm_tokenizer,
        question=question,
        answer=answer,
        generation_config=generation_config,
        n_iterations=n_iterations,
        c_puct=1.0,
        top_k=3
    )
    return root, steps, solution, n_iter


def main(args):
    device = "cuda:{}".format(args.gpu_id)
    generation_config = GenerationConfig(
        temperature=0.5,
        do_sample=True,
        top_p=0.9,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=torch.float16, device_map=device
    )
    processor = AutoProcessor.from_pretrained(args.model_id)

    eval_llm = AutoModelForCausalLM.from_pretrained(
        args.eval_model_name,
        torch_dtype="auto",
        device_map=device,
    )
    eval_llm_tokenizer = AutoTokenizer.from_pretrained(args.eval_model_name)


    df = pd.read_json(args.data_pths, lines=True)  # 读取 JSON Lines 文件
    datas = df.to_dict(orient='records')


    data_chunk = get_chunk(datas, args.num_chunks, args.chunk_idx)
    # 以追加模式打开文件
    with open(args.output_file, 'a', encoding='utf-8') as f:
        for data in tqdm(data_chunk, desc="MCTS Progress"):
            image_data = data['image_paths'][0]
            question = data['prompt'].split('<image>')[1]
            answer = data['answer']
            text_prompt = few_shot_cot_prompt + '{}'.format(question)

            root, solution_steps, solution, n_iter = solve_math_reasoning_vlm(
                image_data=image_data,
                text_prompt=text_prompt,
                model=model,
                generation_config=generation_config,
                processor=processor,
                eval_llm=eval_llm,
                eval_llm_tokenizer=eval_llm_tokenizer,
                question=question,
                answer=answer,
                n_iterations=args.max_num_iterations,
            )

            # image_text_attention_ratio = Get_attention_ratio(image_data, text_prompt, model, generation_config, processor)

            if solution is not None:
                try:
                    image_text_attention_ratio, visual_map, image_start_idx = Get_attention_ratio(image_data, text_prompt, model, generation_config, processor)
                    data['solution'] = ''.join(solution.solution_steps)
                    data['iters'] = n_iter
                    data['image_text_attention_ratio'] = image_text_attention_ratio
                    # 将处理后的单条数据转换为 JSON 字符串并写入文件
                    json.dump(data, f, ensure_ascii=False)
                    f.write('\n')  # 写入换行符，保证每条数据占据一行
                    print("写入完成一条data：", data, flush=True)
                    # 画热力图
                    # print("开始画图")   为了节约时间，先不画
                    # plot_attention_heatmap(visual_map, data['id'], image_start_idx)
                except Exception as e:
                    print(f"遇到异常: {e}", flush=True)
                    continue
            else:
                # 记录失败情况
                data['solution'] = "处理失败"
                data['iters'] = -1
                json.dump(data, f, ensure_ascii=False)
                f.write('\n')
                print(f"处理失败,Solution is None: ID={data.get('id', '未知')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="/mnt/tenant-home_speed/AIM/model/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--eval_model_name", type=str, default="/mnt/tenant-home_speed/AIM/model/Qwen2.5-7B-Instruct")
    parser.add_argument("--data_pths", type=str, default="/mnt/tenant-home_speed/qjy/attention_com/VPT_total_done.jsonl")
    parser.add_argument("--output_file", type=str, default="answer.jsonl")
    parser.add_argument("--max_num_iterations", type=int, default=50)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()

    main(args)
