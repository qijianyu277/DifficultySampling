import os
import json
import random
import asyncio
import aiohttp
from tqdm.asyncio import tqdm_asyncio
from PIL import Image, ImageDraw
import base64
from io import BytesIO
import logging
import matplotlib.pyplot as plt
import requests
from typing import List, Dict, Tuple, Optional
import re
import json
from typing import Union, Optional, List, Dict
from difflib import SequenceMatcher
from sympy import sympify, Eq
import re
import ast
from collections import defaultdict
from difflib import SequenceMatcher

import asyncio
import re
import textwrap
from copy import deepcopy
from typing import Dict, List, Optional
from typing import List, Dict, Tuple, Union


import json
import torch

from swift.llm import PtEngine, RequestConfig, Template, to_device
from swift.llm.infer.protocol import ChatCompletionResponse
from swift.plugin import ORM, orms, rm_plugins
from swift.plugin.rm_plugin import DefaultRMPlugin
from swift.utils import get_logger

import ast
from word2number import w2n
import regex as regext
from collections import defaultdict
# from sympy.core.exceptions import SympifyError
# from sympy.parsing.sympy_parser import SympifyError
from sympy import SympifyError
import ast
def is_valid(result, task_type):
    """判断提取出的内容是否符合对应任务类型的要求"""
    if result is None:
        return False

    if task_type in ['cv_detection', 'cv_grounding']:
        # 检查是否为包含 bbox_2d 字段的字典列表
        return (
            isinstance(result, list) and len(result) > 0 and all(
                isinstance(box, dict) and 'bbox_2d' in box and
                isinstance(box['bbox_2d'], list) and len(box['bbox_2d']) == 4 and
                all(isinstance(coord, (int, float)) for coord in box['bbox_2d'])
                for box in result
            )
        )

    elif task_type == 'ocr':
        # OCR 要求是非空字符串
        return isinstance(result, str) and len(result.strip()) > 0

    elif task_type == 'counting':
        # Counting 要求是整数或者可转为整数的字符串
        if isinstance(result, int):
            return True
        elif isinstance(result, str):
            return result.strip().isdigit()
        return False

    elif task_type == 'math':
        # Math 要求是字符串且非空
        return isinstance(result, str) and len(result.strip()) > 0

    else:
        # 不支持的任务类型直接返回 False
        return False
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("experiment.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
from swift.plugin import ORM, orms, rm_plugins
class VisualPerceptionAccuracy(ORM):
    def __init__(self):
        self.iou_threshold = 0.2
        self.ignore_case = True
        self.fuzzy_match_ratio = 0.5
        self.match_valid_only = False

    def __call__(self,
                 completions, solution,task_type,
                 **kwargs):
   
        pred = self.extract_answer(completions, task_type)
        correct = self.extract_answer(solution, task_type)

        # print(f"pred = {pred}, correct = {correct}")

        # Step 6: 计算奖励分数
        if is_valid(pred, task_type) and is_valid(correct, task_type):
            reward = self.calculate_reward(pred, correct, task_type)
        else:
            print(f"Sample: Missing valid prediction or ground truth.")
            reward = 0.0

        return reward

    @staticmethod
    def extract_answer(content: Union[str, dict, list], task_type: str) -> Union[List[Dict], str, int, None]:
        if isinstance(content, str):
            content_str = content
        else:
            try:
                content_str = json.dumps(content)
            except Exception as e:
                print(f"Failed to convert content to string: {str(e)}")
                return ""

        if task_type in ['cv_grounding', 'cv_detection']:
            return VisualPerceptionAccuracy._extract_bbox(content_str)
        elif task_type == 'ocr':
            return VisualPerceptionAccuracy._extract_ocr(content_str)
        elif task_type == 'counting':
            return VisualPerceptionAccuracy._extract_counting(content_str)
        elif task_type == 'math':
            return VisualPerceptionAccuracy._extract_math(content_str)
        else:
            print(f"Unknown task type: {task_type}")
            return ""

    @staticmethod
    def _extract_bbox(content: str) -> List[Dict]:
    
        match = re.search(r'<answer>\s*(.*?)\s*</answer>', content, re.DOTALL)
        if match:
            raw_content = match.group(1).strip()
            try:
                parsed = ast.literal_eval(raw_content)
                if isinstance(parsed, list):
                    return parsed
            except (SyntaxError, ValueError) as e:
                print(f"解析失败: {e}")
                return []
        return []

    @staticmethod
    def _extract_ocr(content: str) -> str:
        """
        按照指定优先级提取 OCR 答案：
        1. 提取 \boxed{...} 中的完整 LaTeX 公式；
        2. 提取 <answer>...</answer> 内容；
        3. 判断是否有数字（阿拉伯或英文），有则转为阿拉伯数字字符串；
        4. 否则返回提取内容的小写形式。
        """

        # Step 1: 尝试提取 \boxed{...} 内容（支持嵌套）
        boxed_match = regext.search(r"\\boxed$\s*({(?:[^{}]|(?R)|\s)*})\s*$", content, re.DOTALL)
        if boxed_match:
            return boxed_match.group(1).strip()

        # Step 2: 尝试提取 <answer> 标签内容
        answer_match = re.search(r"<answer.*?>([\s\S]*?)</answer>", content, re.DOTALL)
        if answer_match:
            answer_text = answer_match.group(1).strip()
        else:
            answer_text = content.strip()

        # Step 3: 判断是否有阿拉伯数字或英文数字
        

        if digit_match or words_match:
            # Step 4: 存在数字，提取并转换为阿拉伯数字字符串
            digit_str = VisualPerceptionAccuracy.extract_number(answer_text)
            if digit_str:
                return digit_str

        # Step 5: 没有发现数字，返回小写形式
        return answer_text.lower()

    @staticmethod
    def extract_number(content: str) -> Optional[str]:
        """
        从 content 中提取第一个出现的数字，支持：
        - 阿拉伯数字（如 "89"）
        - 英文数字（如 "three", "five"）
        返回提取到的数字字符串，未找到则返回 None
        """
        # 尝试匹配阿拉伯数字
        digit_match = re.search(r"\d+", content)
        if digit_match:
            return digit_match.group(0)

        # 尝试匹配英文数字
        words = re.findall(
            r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|"
            r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
            r"eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
            r"eighty|ninety|hundred|thousand)\b",
            content, re.IGNORECASE
        )
        if not words:
            return None

        try:
            return str(w2n.word_to_num(' '.join(words)))
        except Exception:
            return None

    @staticmethod
    def _extract_counting(content: str) -> Optional[int]:
        """从字符串中提取计数结果（整数）"""
       
        answer_match = re.search(r"<answer.*?>([\s\S]*?)</answer>", content, re.DOTALL)
        if answer_match:
            answer_text = answer_match.group(1).strip()

            # 从文本中提取第一个数字
            number_match = re.search(r"\d+", answer_text)
            if number_match:
                return int(number_match.group(0))

        return None

    @staticmethod
    def _extract_math(content: str) -> str:
        """LaTeX 公式提取"""
        print("latex 提取中...")
        match = re.search(r'<answer>\s*(.*?)\s*</answer>', content, re.DOTALL)
        if match:
            return match.group(1).strip()

        boxed_match = re.search(r'\\boxed\{([^}]+)\}', content)
        if boxed_match:
            return boxed_match.group(1).strip()

        return content.strip()

    def calculate_reward(self,
                         preds,
                         gts,
                         task_type: str) -> float:
        """
        根据预测结果和真实值计算奖励。
        """
        if preds is None or gts is None:
            return 0.0

        if task_type in ['cv_grounding', 'cv_detection']:
            return self._calculate_bbox_reward(preds, gts)
        elif task_type == 'ocr':
            return self._calculate_ocr_reward(preds, gts)
        elif task_type == 'counting':
            return self._calculate_counting_reward(preds, gts)
        elif task_type == 'math':
            return self._calculate_math_reward(preds, gts)
        else:
            return 0.0

    def _calculate_bbox_reward(self, preds: List[Dict], gts: List[Dict]) -> float:
        

        # Step 3: 返回整体奖励
        overall_reward = total_reward / total_weight if total_weight > 0 else 0.0
        print(f"\n【最终整体加权奖励】{overall_reward:.4f}")
        return overall_reward

    def _match_and_calculate_weighted_iou(self, preds: List[Dict], gts: List[Dict]) -> Tuple[float, float]:
        matched_preds = set()
        total_iou = 0.0
        total_weight = 0.0

        for gt in gts:
            best_iou = 0.0
            best_pred_idx = -1
            gt_area = (gt['bbox_2d'][2] - gt['bbox_2d'][0]) * (gt['bbox_2d'][3] - gt['bbox_2d'][1])

            for i, pred in enumerate(preds):
                if i in matched_preds:
                    continue
                
                current_iou = self.calculate_iou(pred, gt)
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_pred_idx = i

            if best_pred_idx != -1:
                matched_preds.add(best_pred_idx)
                total_iou += best_iou * gt_area
                total_weight += gt_area

        return total_iou, total_weight

    @staticmethod
    def calculate_iou(boxA: Dict, boxB: Dict) -> float:
        xA = max(boxA['bbox_2d'][0], boxB['bbox_2d'][0])
        yA = max(boxA['bbox_2d'][1], boxB['bbox_2d'][1])
        xB = min(boxA['bbox_2d'][2], boxB['bbox_2d'][2])
        yB = min(boxA['bbox_2d'][3], boxB['bbox_2d'][3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        boxAArea = (boxA['bbox_2d'][2] - boxA['bbox_2d'][0] + 1) * (boxA['bbox_2d'][3] - boxA['bbox_2d'][1] + 1)
        boxBArea = (boxB['bbox_2d'][2] - boxB['bbox_2d'][0] + 1) * (boxB['bbox_2d'][3] - boxB['bbox_2d'][1] + 1)

        iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0.0
        return iou

    def _calculate_ocr_reward(self, pred: str, gt: str) -> float:
        # 统一清洗函数：去除 \boxed{} 和换行符，并做基本清理
        def clean_text(s: str) -> str:
            # 去除 \boxed{}
            boxed_match = re.search(r'\\boxed\s*{\s*(.*?)\s*}', s)
            if boxed_match:
                s = boxed_match.group(1).strip()
            else:
                s = s.strip()

            # 去除所有换行符和多余空白
            s = s.replace('\n', '').replace('\\n', '')  # 删除显式换行
            s = ' '.join(s.split())  # 合并多个空格为一个
            return s

        # 清洗 pred 和 gt
        pred = clean_text(pred)
        gt = clean_text(gt)
        print(f"now, pred = {pred}, gt = {gt}")
        if not isinstance(pred, str) or not isinstance(gt, str):
            return 0.0

        if self.ignore_case:
            pred = pred.lower()
            gt = gt.lower()

        if pred == gt:
            return 1.0

        # 新增：子串匹配逻辑
        if pred in gt or gt in pred:
            return 1.0

        ratio = SequenceMatcher(None, pred, gt).ratio()
        return 1.0 if ratio >= self.fuzzy_match_ratio else 0.0

    def _calculate_counting_reward(self, pred: int, gt: int) -> float:
        return 1.0 if pred == gt else 0.0

def set_seed(seed: int = 42) -> None:
    """设置随机种子以保证实验可重复性"""
    random.seed(seed)


def simple_mask(
    image: Image.Image,
    mask_ratio: float = 0.3,
    fill: Tuple[int, int, int] = (0, 0, 0),
    shape: str = 'rectangle'
) -> Image.Image:
    """
    对图像应用简单遮罩
    
    Args:
        image: 输入PIL图像
        mask_ratio: 遮罩面积占图像总面积的比例
        fill: 遮罩填充颜色的RGB元组
        shape: 遮罩形状（rectangle, ellipse, circle）
    
    Returns:
        带遮罩的图像
    """
    width, height = image.size
    # 基于面积比例计算遮罩尺寸，确保遮罩大小更一致
    area_ratio = mask_ratio
    w = int(width * area_ratio ** 0.5)
    h = int(height * area_ratio ** 0.5)
    w = max(1, min(w, width))
    h = max(1, min(h, height))


    masked_image = image.copy()
    draw = ImageDraw.Draw(masked_image)

    if shape == 'rectangle':
        draw.rectangle((x, y, x + w, y + h), fill=fill)
    elif shape == 'ellipse':
        draw.ellipse((x, y, x + w, y + h), fill=fill)
    elif shape == 'circle':
        radius = min(w, h) // 2
        cx = x + w // 2
        cy = y + h // 2
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=fill)
    else:  # 默认使用矩形
        draw.rectangle((x, y, x + w, y + h), fill=fill)
    return masked_image


def image_to_base64(image: Image.Image) -> str:
    """将PIL图像转换为Base64字符串"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")  # 使用PNG以获得更好的质量和alpha通道支持
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode()


def encode_image_to_base64(image_path: str) -> str:
    """从文件路径将图像编码为Base64"""
    file_format = image_path.split('.')[-1].lower()
    if file_format not in ['png', 'jpg', 'jpeg']:
        raise ValueError(f"不支持的图像格式: {file_format}")
    
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return f"data:image/{file_format};base64,{encoded_string}"


async def query_api(
    session: aiohttp.ClientSession,
    url: str,
    image_b64: str,
    prompt: str,
    retry: int = 3
) -> Optional[str]:
    """
    异步调用API，获取单次模型输出
    
    Args:
        session: aiohttp客户端会话
        url: API端点URL
        image_b64: Base64编码的图像字符串
        prompt: 模型的文本提示
        retry: API调用的重试次数
    
    Returns:
        模型输出字符串，若所有重试失败则返回None
    """
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_b64}}
                ]
            }
        ],
        "temperature": 0.7,
        "max_tokens": 1024
    }
    
    for attempt in range(retry):
        try:
            async with session.post(url, json=payload, timeout=600) as response:
                response.raise_for_status()  # 对HTTP错误(4xx或5xx)抛出异常
                result = await response.json()
                return result['choices'][0]["message"]["content"]
        except Exception as e:
            logger.warning(f"API调用尝试 {attempt+1}/{retry} 失败: {str(e)}")
            if attempt < retry - 1:
                await asyncio.sleep(1)  # 重试前短暂等待
    return None


async def process_sample(
    session: aiohttp.ClientSession,
    url: str,
    sample: Dict,
    mask_ratio: float,
    semaphore: asyncio.Semaphore
) -> Tuple[bool, Dict]:
    """
    异步处理单个样本，应用遮罩并调用API
    
    Args:
        session: aiohttp客户端会话
        url: API端点URL
        sample: 包含图像路径、提示和答案的样本数据
        mask_ratio: 遮罩比例
        semaphore: 限制并发API请求的信号量
    
    Returns:
        元组 (是否正确, 样本数据)
    """
    async with semaphore:  # 控制并发数量
        try:
            # 验证样本数据
            
            # 验证图像文件
            if not os.path.exists(image_path):
                logger.error(f"图像文件不存在: {image_path}, ID: {sample['id']}")
                return False, sample
            if os.path.getsize(image_path) == 0:
                logger.error(f"图像文件为空: {image_path}, ID: {sample['id']}")
                return False, sample

            # 加载并处理图像
            
            # 准备图像数据
            current_image_b64 = None
            if mask_ratio == 0.0:
                current_image_b64 = encode_image_to_base64(image_path)
            else:
                masked_image = simple_mask(img, mask_ratio=mask_ratio)
                current_image_b64 = image_to_base64(masked_image)

            # 调用API获取结果
            model_output = await query_api(session, url, current_image_b64, sample["prompt"])
            if model_output is None:
                logger.warning(f"API调用失败，样本ID: {sample['id']}")
                return False, sample
            task_type = sample["ability"]
            # 验证结果
            
            return is_correct_flag, sample

        except Exception as e:
            logger.error(f"处理样本时发生意外错误 ID: {sample.get('id', '未知')}: {str(e)}", exc_info=True)
            return False, sample


async def run_iteration(
    session: aiohttp.ClientSession,
    url: str,
    remaining_samples: List[Dict],
    mask_ratio: float,
    max_concurrent: int
) -> Tuple[List[Dict], List[Dict]]:
    """
    运行单次迭代，处理剩余样本并返回新正确的样本和仍剩余的样本
    
    Args:
        session: aiohttp客户端会话
        url: API端点URL
        remaining_samples: 上一轮未正确回答的样本
        mask_ratio: 遮罩比例
        max_concurrent: 最大并发请求数
    
    Returns:
        元组 (本轮正确的样本, 仍剩余的样本)
    """
    


async def run_experiment(
    samples: List[Dict],
    url: str,
    mask_ratio: float,
    num_iterations: int,
    max_concurrent: int = 10
) -> Tuple[List[Dict], List[Dict]]:
    """
    运行完整实验，多次迭代处理样本
    
    Args:
        samples: 要处理的样本列表
        url: API端点URL
        mask_ratio: 遮罩比例
        num_iterations: 迭代次数
        max_concurrent: 最大并发请求数
    
    Returns:
        元组 (所有正确的样本, 最终仍错误的样本)
    """
    logger.info(f"开始实验，遮罩比例: {mask_ratio}, 迭代次数: {num_iterations}")
    
    all_correct = []
    remaining_samples = samples.copy()
    
    # 创建aiohttp会话，优化连接池
    connector = aiohttp.TCPConnector(limit_per_host=max_concurrent, ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        for iteration in range(1, num_iterations + 1):
            if not remaining_samples:
                logger.info("所有样本已正确回答，提前结束迭代")
                break
                
            logger.info(f"开始迭代 {iteration}/{num_iterations}")
            new_correct, remaining_samples = await run_iteration(
                session, url, remaining_samples, mask_ratio, max_concurrent
            )
            all_correct.extend(new_correct)
    
    total_correct = len(all_correct)
    total_samples = len(samples)
    logger.info(
        f"实验完成 - 遮罩比例: {mask_ratio}, "
        f"总正确率: {total_correct}/{total_samples} ({total_correct/total_samples:.2%})"
    )
    return all_correct, remaining_samples


def load_dataset(path: str) -> List[Dict]:
    """
    从JSONL文件加载数据集
    
    Args:
        path: JSONL文件路径
    
    Returns:
        数据集样本列表
    """
    dataset = []
    try:
        with open(path, "r", encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    dataset.append({
                        "id": item.get("id", f"line_{line_num}"),
                        "ability": item.get("ability", ""),
                        "prompt": item.get("prompt", ""),
                        "ground_truth": item.get("ground_truth", ""),
                        "image_paths": item.get("image_paths", [])
                    })
                except json.JSONDecodeError as e:
                    logger.warning(f"解析JSONL文件第{line_num}行失败: {str(e)}，跳过该行")
    except FileNotFoundError:
        logger.error(f"数据集文件未找到: {path}")
        exit(1)
    except Exception as e:
        logger.error(f"加载数据集失败: {str(e)}")
        exit(1)
        
    logger.info(f"从 {path} 加载了 {len(dataset)} 个样本")
    return dataset


def save_samples(samples: List[Dict], path: str) -> None:
    """保存样本到JSONL文件"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    logger.info(f"已保存 {len(samples)} 个样本到 {path}")


if __name__ == '__main__':
    # 配置参数
    config = {
        "dataset_path": "/your dataset path/.jsonl",
        "url_pool": ["http://localhost:your port/v1/chat/completions"],
        "num_iterations": 10,  # 迭代次数（原num_generations）
        "seed": 42,
        "output_dir": "your output path",
        "max_concurrent": 10,  # 最大并发请求数
        "mask_ratios": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # 要测试的遮罩比例
    }

    # 初始化
    set_seed(config["seed"])
    os.makedirs(config["output_dir"], exist_ok=True)
    api_url = config["url_pool"][0]  # 使用第一个API URL

    # 步骤1: 获取遮罩比例为0时能正确回答的样本
    zero_mask_file = os.path.join(config["output_dir"], "correct_samples_mask_0.jsonl")
    correct_samples_zero_mask = []


    # 步骤2: 对不同遮罩比例运行实验
    results = {}
    base_count = len(correct_samples_zero_mask)


    # 步骤3: 保存最终结果
    
    # 步骤4: 绘制准确率曲线