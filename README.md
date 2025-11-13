# Revisiting the Data Sampling in Multimodal Post-training from a Difficulty-Distinguish View
## Introduction
We deeply investigate the difficulty-aware data sampling in multimodal post-training, mainly for answering the following questions:

1. Can difficulty-distinguish sampling strategies effectively optimize multimodal reasoning and perception capabilities without relying on supervised fine-tuning (SFT)?
2. Is it feasible to apply difficulty-aware sampling to various multimodal tasks beyond mathematical reasoning, such as visual perception, chart interpretation, and complex reasoning? If so, what specific improvements can be achieved across different task types?
3. For multimodal large language models (MLLMs) with varying scales and baseline capabilities, can the proposed difficulty-distinguish framework be universally applicable?
<img width="1472" height="804" alt="fig1" src="https://github.com/user-attachments/assets/d4dc43fa-62ea-4e0b-92a9-5af73452e62c" />

## Models & Data Checkpoints
We provide the GRPO-only training weights on the visual reasoning task (VRT) and visual perception task (VPT) in the following table.

| Algorithm | Tasks Type | Checkpoint  | 
|------|-------|-----------------------|
| PISM | VRT   |    To_be_uploaded     | 
| PISM | VPT   |    To_be_uploaded     | 
| CMAB | VRT   |    To_be_uploaded     |
| CMAB | VPT   |    To_be_uploaded     |
## Data
The data we use comes from the paper [One RL to See Them All](https://arxiv.org/abs/2505.18129), and through the algorithms we provide below, you can easily obtain the data for model post-training.
## Requirement
This project utilizes a dual-framework training architecture:\
1. We perform GRPO based on [MS-Swift](https://github.com/modelscope/ms-swift) for specifically we use ms-swift 3.5.0:
```
   conda create -n swift python=3.10
   conda activate swift
   pip install ms-swift==3.7.2
```
2. We implete SFT training based on [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory) for specifically we use LlamaFactory 0.9.3:
```
   conda create -n llf python=3.10
   conda activate llf
   git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
   cd LLaMA-Factory
   pip install -e ".[torch,metrics]" --no-build-isolation
```
---
## Data Sampling
### 1. PISM Algorithm Execution
#### Execute the following command:
```bash
cd /PISM/main
python main_pism_async.py
```
#### Key Parameters Configuration:
```python
config ={
"dataset _path":"/your/dataset/path.jsonl", # Dataset source path
"num_iterations": 10, # Total execution cycles
"seed": 42, # Random seed for reproducibility
"output_dir": "output/path", # Results storage directory
"max_concurrent": 10, # Maximum concurrent requests
"mask_ratios": [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9] # Masking ratios to select samples
}
```
---
### 2. CMAB Algorithm Execution
#### Execute the following command:
```bash
cd ./CMAB/main
python attention_ratio.py
```
#### Visualization:
Uncomment plotting functions in the code to generate PNG visualizations of results.
```
def plot_attention_heatmap(visual_map, data_id, image_start_idx):
    if not visual_map:
        print("visual_map is empty; cannot plot heatmap")
        return

    # Step 1: Ensure all tensors are on CPU and normalized
    visual_map_cpu_normalized = [normalize_tensor(t.cpu()) for t in visual_map]

    # Step 2: Find the maximum sequence length
    max_len = max(t.size(1) for t in visual_map_cpu_normalized)

    # Step 3: Pad all tensors to the same length
    padded_tensors = [pad_tensor(t, max_len, pad_value=0.0) for t in visual_map_cpu_normalized]

    # Step 4: Concatenate tensors
    # shape: (n_tokens, max_len)
    attention_tensor = torch.cat(padded_tensors, dim=0)

    # Step 5: Convert to numpy for plotting
    attention_array = attention_tensor.numpy()

    # Step 6: Construct output path
    safe_data_id = sanitize_filename(data_id)
    output_dir = "/path/to/your/map_images/"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{safe_data_id}_attn_heatmap.png")

    # Step 7: Plot
    plt.figure(figsize=(12, 8))
    plt.imshow(attention_array, cmap='viridis', aspect='auto', interpolation='nearest', vmin=0.0, vmax=1.0)
    plt.colorbar(label='Attention Weight')
    plt.title("Normalized Attention Heatmap Across Tokens")
    plt.xlabel("Input Position")
    plt.ylabel("Token Index")

    # Step 8: Save figure
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Heatmap saved to: {output_path}")
```
## Data Convert
### Preprocessing Workflow:
1. Put the data sampled by PISM and CMAB algorithms into the `path/to/your/sampled_data` file directory.
2. Execute format conversion scripts:
```bash
cd path/to/your/sampled_data
python to_grpo.py
python to_sft.py
```
---
## Training
### 1. GRPO Scropt Execution
We have provided shell scripts for all GRPO-only training strategies, and you only need to change the specific model and data path to directly execute them. The following "to_be_added" refers to the specific names of different training strategies. 
#### Training Command:
```bash
cd ./run sh/GRPO/VPT or   
cd ./run sh/GRPO/VRT
bash run GRPO_VPT_to_be_added.sh or 
bash run GRPO_VRT_to_be_added.sh
```
The general content of the script is
```
MASTER_PORT=29501 WANDB_MODE=offline MAX_PIXELS=1254400 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
WANDB_API_KEY= \
NPROC_PER_NODE=8 \
swift rlhf \
    --rlhf_type grpo \
    --model path/to/your/checkpoint-85 \
    --external_plugins path/to/your/plugin.py \
    --reward_funcs external_r1v_acc \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset path/to/your/VRT_0.1-0.4_and_1.6-1.9.jsonl  \
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
    --output_dir outputpath/to/your/mid_hard \
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
```
#### Data Format Specification:
```json
{
"images": ["/path/to/image.png"],"messages":[{"role": "user", "content": "<image>\nWhere is the picture from?"}], "solution": "\\boxed{Google}"
}
```
---
### 2. SFT Script Execution
We also provide shell scripts for GRPO+FFT training. Simply update the model and data paths in qwen2.5vls_sft.yaml and run them in the LLaMA Factory environment with the predefined configuration.
#### Training Command:
```bash
llamafactory-cli train examples/train full/qwen2.5vl_sft.yaml
```
#### Data Format Specification
```json
{
"messages":[{"role": "user", "content": "<image>\nExtract the formula in LaTeX format..."}],{"role": "assistant", "content": "end>\nbox> -1 + 1 =0 </box>end>"},"images": ["img data/sample_image.png"]
}
```
---
## Evalution
### Jsonl Data for Evaluation
| Training paradigm       | MathVista | MMVet  | OCRBench | HBench  | MMMU   | MMStar  |
|-------------------------|-----------|--------|----------|---------|--------|---------|
| GRPO-only(fullset)      | 70.000    | 51.147 | 77.200   | 68.034  | 0.557  | 0.615   |
| GRPO-only(unsolved)     | 68.700    | 54.541 | 77.700   | 69.085  | 0.536  | 0.615   |
| SFT(mid)+GRPO(hard)     | 66.800    | 41.239 | 75.100   | 68.244  | 0.503  | 0.627*  |
| SFT(mid)+GRPO(rand_h)   | 66.500    | 42.431 | 75.200   | 67.823  | 0.499  | 0.626   |
| SFT(rand_m)+GRPO(hard)  | 67.800    | 36.514 | 75.100   | 68.875  | 0.499  | 0.625   |
| SFT(rand_m)+GRPO(rand_h)| 67.500    | 42.798 | 75.000   | 68.769  | 0.496  | 0.623   |
| SFT(hard)+GRPO(mid)     | 67.400    | 34.037 | 75.200   | 68.244  | 0.501  | 0.618   |
| SFT(hard)+GRPO(rand_m)  | 67.900    | 48.945 | 75.900   | 67.718  | 0.538  | 0.609   |
| SFT(rand_h)+GRPO(mid)   | 68.100*   | 49.500 | 76.500   | 68.454  | 0.534  | 0.607   |
| SFT(rand_h)+GRPO(rand_m)| 67.600    | 50.321*| 77.500   | 68.980* | 0.526  | 0.610   |
| GRPO-only(random)       | 67.700    | 45.550 | 76.900*  | **69.401**  | 0.545* | 0.625   |
| GRPO-only(mid+hard)     | **68.300**| **50.367** | 76.800 | 68.244 | **0.550** | **0.629** |
### Evaluation Scripts
To be added.
### Results Display
To be added.
## Institution
- ZTE-AIM
- School of Computer Science, Central South University, Changsha, Hunan, China
## Citation
@misc{qi2025revisitingdatasamplingmultimodal,
      title={Revisiting the Data Sampling in Multimodal Post-training from a Difficulty-Distinguish View}, 
      author={Jianyu Qi and Ding Zou and Wenrui Yan and Rui Ma and Jiaxu Li and Zhijie Zheng and Zhiguo Yang and Rongchang Zhao},
      year={2025},
      eprint={2511.06722},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.06722}, 
}





