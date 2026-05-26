import unsloth
from unsloth import FastLanguageModel

import os
import json
import re
import math
import torch
from collections import Counter
from datetime import datetime
from transformers import GenerationConfig
from tqdm import tqdm

# 导入 utils 中的依赖
from utils import *

def generate_and_save_trajectories_arcc_tharp(
    model_path: str,
    adapter_path: str,
    data_path: str,
    temperature: float,
    k: int = 32,
    batch_size: int = 8,
    num_samples: int = None,
    max_new_tokens: int = 1024,
    soft_penalty: float = 0.0,
    soft_top_k: int = 10,
    output_jsonl_path: str = "trajectories.jsonl"
):
    """第一步：生成 THARP 模型在 ARCC 上的轨迹，提取 Soft Token 和 Hard Token 长度并实时落盘"""
    
    # 1. 检查是否已经生成完毕，支持断点续传
    existing_count = 0
    if os.path.exists(output_jsonl_path):
        with open(output_jsonl_path, 'r', encoding='utf-8') as f:
            existing_count = sum(1 for _ in f)
        print(f"Found existing trajectories file with {existing_count} records. Resuming...")

    # 2. 读取本地 ARCC JSON 数据集
    print(f"Loading ARCC dataset from: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
        
    if num_samples and len(dataset) > num_samples:
        import random
        random.seed(42)
        dataset = random.sample(dataset, num_samples)
        
    total_samples = len(dataset)
    print(f"Total target samples: {total_samples}")
    
    if existing_count >= total_samples:
        print("All trajectories already generated. Skipping generation phase.")
        return output_jsonl_path

    # 3. 加载 Base Model 和 Adapter
    print(f"Loading base model: {model_path}")
    print(f"Loading adapter for generation: {adapter_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_new_tokens * 2,
        load_in_4bit=False,
        fast_inference=False, 
    )
    model.answer_start = ANSWER_START
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model.load_adapter(adapter_path)
    model = FastLanguageModel.for_inference(model)

    # 准备 EOS IDs
    eos_ids = model.generation_config.eos_token_id
    if eos_ids is None:
        eos_ids = [tokenizer.eos_token_id]
    elif not isinstance(eos_ids, list):
        eos_ids = [eos_ids]
        
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None:
        eos_ids.append(im_end_id)
    eos_ids_tensor = torch.tensor(list(set(eos_ids)), device=model.device)

    # 4. 开始生成
    progress_bar = tqdm(total=total_samples, initial=existing_count, desc="Generating", dynamic_ncols=True)
    
    with open(output_jsonl_path, 'a', encoding='utf-8') as f_out:
        for i in range(existing_count, total_samples, batch_size):
            batch_data = dataset[i:i + batch_size]
            current_batch_size = len(batch_data)

            # 解析 ARCC 数据结构
            questions = [item["prompt"][0]["value"] for item in batch_data]
            true_answers = [str(item["final_answer"]).strip().upper() for item in batch_data]

            prompts = [
                [
                    {'role': 'system', 'content': QA_SYSTEM_PROMPT},
                    {'role': 'user', 'content': q.strip()},
                ] for q in questions
            ]

            formatted_prompts = [
                tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
                for p in prompts
            ]

            prompt_inputs = tokenizer(
                formatted_prompts, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
            )
            prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
            
            prompt_ids = prompt_ids.repeat_interleave(k, dim=0).to(model.device)
            prompt_mask = prompt_mask.repeat_interleave(k, dim=0).to(model.device)
            prompt_length = prompt_ids.size(1)

            # THARP 核心生成逻辑
            generated_sequences, generated_soft_mask, generated_plain_texts = model.generate(
                prompt_ids, attention_mask=prompt_mask, 
                generation_config=GenerationConfig(
                    do_sample=True,  
                    temperature=temperature,
                    max_new_tokens=max_new_tokens, 
                ),
                processing_class=tokenizer,
                is_inference=True,
                use_mixed_rollout=True,
                action_temperature=1.0, 
                action_soft_penalty=soft_penalty, 
                soft_top_k=soft_top_k,
                is_greedy=False, 
            )

            for j in range(current_batch_size):
                question = questions[j]
                true_answer = true_answers[j]

                raw_samples = []
                for sample_idx in range(k):
                    idx = j * k + sample_idx
                    gen_ids = generated_sequences[idx][prompt_length:]
                    
                    # 使用 tensor 向量化找多类型 EOS
                    eos_indices = (gen_ids.unsqueeze(1) == eos_ids_tensor).any(dim=1).nonzero(as_tuple=True)[0]
                    seq_len = eos_indices[0].item() + 1 if len(eos_indices) > 0 else len(gen_ids)

                    effective_tokens = gen_ids[:seq_len]
                    response = tokenizer.decode(effective_tokens, skip_special_tokens=True)
                    
                    # === 核心：分离 THARP 的 Soft Token 与 Hard Token ===
                    effective_len = seq_len
                    if generated_soft_mask is not None:
                        completion_mask = generated_soft_mask[idx]
                        valid_mask = completion_mask[:seq_len]
                        num_soft_tokens = valid_mask.float().sum().item()
                        effective_len = seq_len - int(num_soft_tokens)
                    
                    raw_samples.append({
                        'full_response': response,
                        'generated_length': seq_len,
                        'effective_length': effective_len
                    })

                record = {
                    'question': question,
                    'true_answer': true_answer,
                    'samples': raw_samples
                }
                
                f_out.write(json.dumps(record) + "\n")
                f_out.flush() 

            progress_bar.update(current_batch_size)
            
    progress_bar.close()
    
    del model
    del tokenizer
    torch.cuda.empty_cache()
    
    return output_jsonl_path


def evaluate_saved_trajectories_arcc(jsonl_path: str, k: int, output_dir: str):
    """第二步：离线读取轨迹文件，计算方差、置信区间及 ARCC 标准评估指标和长度指标"""
    print(f"Starting offline evaluation on {jsonl_path}...")
    
    results = []
    question_avg_accs = []
    
    total_questions = 0
    total_pass_at_k = 0
    total_maj_at_k = 0
    
    # 双长度累加器
    total_length = 0
    total_effective_length = 0
    total_samples_parsed = 0

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Evaluating Metrics", dynamic_ncols=True):
            if not line.strip(): continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print("\nWarning: Skipped a corrupted line in JSONL file.")
                continue
                
            question = data['question']
            true_answer = data['true_answer']
            
            correct_count = 0
            extracted_answers = []
            sample_evals = []

            for sample in data['samples']:
                response = sample['full_response']
                gen_len = sample['generated_length']
                eff_len = sample.get('effective_length', gen_len)
                
                # 提取并验证答案 (使用 ARCC/GPQA 专用的 A/B/C/D 提取逻辑)
                extracted = extract_from_response(response)
                generated_answer = process_gpqa_answer(extracted)
                extracted_answers.append(generated_answer)

                is_correct = (generated_answer == true_answer)
                if is_correct:
                    correct_count += 1
                
                # 长度累加
                total_length += gen_len
                total_effective_length += eff_len
                total_samples_parsed += 1

                sample_evals.append({
                    'generated_answer': generated_answer,
                    'correct': is_correct,
                    'generated_length': gen_len,
                    'effective_length': eff_len
                })

            avg_acc = correct_count / k
            pass_at_k = 1 if correct_count > 0 else 0
            
            valid_answers = [ans for ans in extracted_answers if ans in ["A", "B", "C", "D"]]
            maj_at_k = 0
            if valid_answers:
                most_common_answer = Counter(valid_answers).most_common(1)[0][0]
                if most_common_answer == true_answer:
                    maj_at_k = 1

            question_avg_accs.append(avg_acc)
            total_pass_at_k += pass_at_k
            total_maj_at_k += maj_at_k
            total_questions += 1

            results.append({
                'question': question,
                'true_answer': true_answer,
                'metrics': {
                    'avg_acc': avg_acc,
                    'pass@k': pass_at_k,
                    'maj@k': maj_at_k
                },
                'sample_evals': sample_evals
            })

    if total_questions == 0:
        print("No valid evaluation records found.")
        return
        
    mean_avg_acc = sum(question_avg_accs) / total_questions
    
    variance = sum((x - mean_avg_acc) ** 2 for x in question_avg_accs) / (total_questions - 1) if total_questions > 1 else 0
    std_dev = math.sqrt(variance)
    
    margin_of_error = 1.96 * (std_dev / math.sqrt(total_questions)) if total_questions > 0 else 0
    ci_lower = mean_avg_acc - margin_of_error
    ci_upper = mean_avg_acc + margin_of_error

    metrics = {
        f'avg@{k}': mean_avg_acc,
        f'pass@{k}': total_pass_at_k / total_questions,
        f'maj@{k}': total_maj_at_k / total_questions,
        'variance': variance,
        'std_dev': std_dev,
        '95_CI_lower': ci_lower,
        '95_CI_upper': ci_upper,
        'total_questions': total_questions,
        'avg_total_len': total_length / total_samples_parsed if total_samples_parsed > 0 else 0,
        'avg_effective_len': total_effective_length / total_samples_parsed if total_samples_parsed > 0 else 0,
        'avg_soft_ratio': (total_length - total_effective_length) / total_length if total_length > 0 else 0,
        'timestamp': datetime.now().isoformat()
    }

    os.makedirs(output_dir, exist_ok=True)
    final_save_path = os.path.join(output_dir, "eval_results.json")
    with open(final_save_path, 'w', encoding='utf-8') as f:
        json.dump({'metrics': metrics, 'results': results}, f, indent=2, ensure_ascii=False)
        
    print("\n" + "="*50)
    print("🎯 THARP ARCC EVALUATION SUMMARY")
    print("="*50)
    print(f" Avg@{k}:      {metrics[f'avg@{k}']:.4f}  [95% CI: {ci_lower:.4f} ~ {ci_upper:.4f}]")
    print(f" Pass@{k}:     {metrics[f'pass@{k}']:.4f}")
    print(f" Maj@{k}:      {metrics[f'maj@{k}']:.4f}")
    print(f" Variance:    {variance:.4f}")
    print("-" * 50)
    print(f" Avg Total Len: {metrics['avg_total_len']:.1f} tokens")
    print(f" Avg Hard Len:  {metrics['avg_effective_len']:.1f} tokens")
    print(f" Soft Ratio:    {metrics['avg_soft_ratio']*100:.2f}%")
    print("="*50)
    print(f"Results saved to: {final_save_path}\n")

    # ==========================================
    # 补充：制表符纯净打印 (方便一键复制到 Excel / LaTeX)
    # ==========================================
    avg_k_pct = metrics[f'avg@{k}'] * 100
    pass_k_pct = metrics[f'pass@{k}'] * 100
    maj_k_pct = metrics[f'maj@{k}'] * 100
    avg_generated_tokens = metrics['avg_total_len']
    
    print("Avg@k(%)\tPass@k(%)\tMaj@k(%)\tAvg_Tokens")
    print(f"{avg_k_pct:.2f}\t\t{pass_k_pct:.2f}\t\t{maj_k_pct:.2f}\t\t{avg_generated_tokens:.1f}")
    
    print("\n[Copy to Excel (纯数据)]:")
    print(f"{avg_k_pct:.2f}\t{pass_k_pct:.2f}\t{maj_k_pct:.2f}\t{avg_generated_tokens:.1f}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/arcc/arcc_test.json", help="Path to ARCC JSON")
    parser.add_argument("--k", type=int, default=32, help="Number of samples to generate per question")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for parallel processing")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to THARP adapter checkpoint")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    args = parser.parse_args()

    max_new_tokens = args.max_new_tokens

    # 自动推断 base_model (适配给出的路径模式)
    base_model = None
    checkpoint_path = args.checkpoint_path
    base_models = [
        "/workplace/nankai/liting_space/LLM/Qwen2.5-1.5B-Instruct", 
        "/workplace/nankai/liting_space/LLM/Qwen2.5-3B-Instruct", 
        "/workplace/nankai/liting_space/LLM/Qwen2.5-7B-Instruct",
    ]

    for model in base_models:
        if model.split('/')[-1] in checkpoint_path:
            base_model = model

    if base_model is None:
        raise ValueError(f"Could not infer base model from checkpoint_path: {checkpoint_path}")

    # 解析 THARP 特有的超参
    temp_match = re.search(r'-temp([\d\.]+)', checkpoint_path)
    temperature = float(temp_match.group(1)) if temp_match else 0.5

    soft_match = re.search(r'-penalty([\d\.]+)', checkpoint_path)
    soft_penalty = float(soft_match.group(1)) if soft_match else 0.0

    soft_top_k_match = re.search(r'-topk([\d\.]+)', checkpoint_path)
    soft_top_k = int(soft_top_k_match.group(1)) if soft_top_k_match else 10

    print(f"Inferred base_model: {base_model}")
    print(f"checkpoint_path: {checkpoint_path}")
    print(f"data_path: {args.data_path}")
    print(f"temperature: {temperature}")
    print(f"soft_penalty: {soft_penalty}")
    print(f"soft_top_k: {soft_top_k}")
    print(f"k: {args.k}")
    print(f"max_new_tokens: {max_new_tokens}")

    # 构建输出目录
    eval_dir = f"{checkpoint_path}/eval_arcc_tharp_k{args.k}_batch{args.batch_size}_{max_new_tokens}"
    os.makedirs(eval_dir, exist_ok=True)
    jsonl_file = os.path.join(eval_dir, "raw_trajectories.jsonl")

    # ==========================
    # STEP 1: GENERATION
    # ==========================
    print("\n" + ">"*15 + " STEP 1: GENERATION " + "<"*15)
    generate_and_save_trajectories_arcc_tharp(
        model_path=base_model,
        adapter_path=checkpoint_path,
        data_path=args.data_path,
        temperature=temperature,
        k=args.k,
        batch_size=args.batch_size,
        num_samples=None,
        max_new_tokens=max_new_tokens,
        soft_penalty=soft_penalty,
        soft_top_k=soft_top_k,
        output_jsonl_path=jsonl_file
    )

    # ==========================
    # STEP 2: EVALUATION
    # ==========================
    print("\n" + ">"*15 + " STEP 2: EVALUATION " + "<"*15)
    evaluate_saved_trajectories_arcc(
        jsonl_path=jsonl_file,
        k=args.k,
        output_dir=eval_dir
    )

