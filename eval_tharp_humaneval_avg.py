import unsloth
from unsloth import FastLanguageModel

import os
import json
import re
import math
import torch
import numpy as np
from collections import Counter
from datetime import datetime
from transformers import GenerationConfig
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from humanevaleval import evaluate_functional_correctness, estimate_pass_at_k

# ==========================================
# Constants and Prompt Definitions
# ==========================================

ANSWER_START = "```python"

CODE_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user provides a python function signature and docstring, "
    "and the assistant completes the function. "
    "The assistant MUST first think about the reasoning process, algorithm design, and edge cases in the mind. "
    "After the reasoning is complete, the assistant MUST provide the final valid python code enclosed "
    "EXACTLY within " + ANSWER_START + " and ``` tags."
)

# ==========================================
# Utility Functions
# ==========================================

def extract_python_code(text: str) -> str:
    """Extract the Python code block from the full model response."""
    matches = re.findall(r"```python\n(.*?)\n?```", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return text.strip()

# ==========================================
# Step 1: Generate Trajectories and Save to Disk (THARP Logic)
# ==========================================

def generate_and_save_trajectories_tharp_code(
    model_path: str,
    adapter_path: str,
    data_path: str,
    temperature: float,
    k: int,
    batch_size: int,
    soft_penalty: float = 0.0,
    soft_top_k: int = 10,
    num_samples: int = None,
    output_jsonl_path: str = "trajectories_humaneval.jsonl"
):
    """Generate trajectories using THARP mixed rollout with Soft/Hard token separation."""

    existing_count = 0
    if os.path.exists(output_jsonl_path):
        with open(output_jsonl_path, 'r', encoding='utf-8') as f:
            existing_count = sum(1 for _ in f)
        print(f"Found existing trajectories file with {existing_count} records. Resuming...")

    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    if num_samples and len(dataset) > num_samples:
        dataset = dataset[:num_samples]

    total_samples = len(dataset)
    print(f"Total target samples: {total_samples}")

    if existing_count >= total_samples:
        print("All trajectories already generated. Skipping generation phase.")
        return output_jsonl_path

    print(f"Loading model for generation: {adapter_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=4096,
        load_in_4bit=False,
        fast_inference=False,
    )
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model.answer_start = ANSWER_START

    model.load_adapter(adapter_path)
    model = FastLanguageModel.for_inference(model)

    eos_ids = model.generation_config.eos_token_id
    if not isinstance(eos_ids, list): eos_ids = [eos_ids] if eos_ids else [tokenizer.eos_token_id]
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None: eos_ids.append(im_end_id)
    eos_ids_tensor = torch.tensor(list(set(eos_ids)), device=model.device)

    progress_bar = tqdm(total=total_samples, initial=existing_count, desc="Generating Code", dynamic_ncols=True)

    with open(output_jsonl_path, 'a', encoding='utf-8') as f_out:
        for i in range(existing_count, total_samples, batch_size):
            batch_data = dataset[i:i + batch_size]
            current_batch_size = len(batch_data)

            questions = [item["prompt"][0]["value"] for item in batch_data]
            ground_truths = [item["final_answer"] for item in batch_data]

            prompts = [
                [
                    {'role': 'system', 'content': CODE_SYSTEM_PROMPT},
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

            # THARP mixed rollout generation
            generated_sequences, generated_soft_mask, generated_plain_texts = model.generate(
                prompt_ids, attention_mask=prompt_mask,
                generation_config=GenerationConfig(
                    do_sample=True,
                    temperature=temperature,
                    max_new_tokens=2048,
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
                gt_dict = ground_truths[j]

                raw_samples = []
                for sample_idx in range(k):
                    idx = j * k + sample_idx
                    gen_ids = generated_sequences[idx][prompt_length:]

                    eos_indices = (gen_ids.unsqueeze(1) == eos_ids_tensor).any(dim=1).nonzero(as_tuple=True)[0]
                    seq_len = eos_indices[0].item() + 1 if len(eos_indices) > 0 else len(gen_ids)

                    effective_tokens = gen_ids[:seq_len]
                    response = tokenizer.decode(effective_tokens, skip_special_tokens=True)

                    # Separate THARP Soft Tokens from Hard Tokens
                    effective_len = seq_len
                    if generated_soft_mask is not None:
                        completion_mask = generated_soft_mask[idx]
                        valid_mask = completion_mask[:seq_len]
                        num_soft_tokens = valid_mask.float().sum().item()
                        effective_len = seq_len - int(num_soft_tokens)

                    raw_samples.append({
                        'full_response': response,
                        'extracted_code': extract_python_code(response),
                        'generated_length': seq_len,
                        'effective_length': effective_len
                    })

                record = {
                    'question': question,
                    'ground_truth': gt_dict,
                    'samples': raw_samples
                }

                f_out.write(json.dumps(record) + "\n")
                f_out.flush()

            progress_bar.update(current_batch_size)

    progress_bar.close()
    del model, tokenizer
    torch.cuda.empty_cache()
    return output_jsonl_path


# ==========================================
# Step 2: Offline Sandbox Evaluation and Dual-Length Metric Aggregation
# ==========================================

def evaluate_saved_trajectories_tharp_code(jsonl_path: str, k: int, output_dir: str):
    """Execute generated code offline and compute THARP-specific metrics including Soft Ratio."""
    print(f"Starting offline code execution on {jsonl_path}...")

    executor = ThreadPoolExecutor(max_workers=8)

    results = []
    question_avg_accs = []

    total_questions = 0
    total_pass_at_k_strict = 0
    total_maj_at_k = 0

    # THARP dual-length accumulators
    total_length = 0
    total_effective_length = 0
    total_samples_parsed = 0

    all_total_samples = []
    all_correct_samples = []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Executing Code", dynamic_ncols=True):
        if not line.strip(): continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        ground_truth = data['ground_truth']
        task_id = ground_truth['task_id']

        original_prompt = data['question']
        import_lines = [
            line for line in original_prompt.split('\n')
            if line.strip().startswith('import ') or line.strip().startswith('from ')
        ]
        safe_prompt = "\n".join(import_lines) + "\n"

        mock_problems = {
            task_id: {
                "task_id": task_id,
                "prompt": safe_prompt,
                "test": ground_truth["test"],
                "entry_point": ground_truth["entry_point"]
            }
        }

        correct_count = 0
        sample_evals = []

        extracted_codes = []
        code_to_correctness = {}

        for sample in data['samples']:
            code_solution = sample['extracted_code']
            gen_len = sample['generated_length']
            eff_len = sample.get('effective_length', gen_len)

            total_length += gen_len
            total_effective_length += eff_len
            total_samples_parsed += 1

            extracted_codes.append(code_solution)

            pass_at_k, judge_info = evaluate_functional_correctness(
                task_id=task_id,
                completion=code_solution,
                problems=mock_problems,
                k=1,
                executor=executor
            )

            is_correct = False
            execution_result = "failed parsing"

            if judge_info and task_id in judge_info:
                is_correct = judge_info[task_id][-1][1]['passed']
                execution_result = judge_info[task_id][-1][1]['result']

            code_to_correctness[code_solution] = is_correct

            if is_correct:
                correct_count += 1

            sample_evals.append({
                'extracted_code': code_solution,
                'correct': is_correct,
                'execution_result': execution_result,
                'generated_length': gen_len,
                'effective_length': eff_len
            })

        current_k = len(data['samples'])
        avg_acc = correct_count / current_k if current_k > 0 else 0
        question_avg_accs.append(avg_acc)

        pass_at_k_strict = 1 if correct_count > 0 else 0
        total_pass_at_k_strict += pass_at_k_strict

        maj_at_k = 0
        if extracted_codes:
            most_common_code = Counter(extracted_codes).most_common(1)[0][0]
            if code_to_correctness.get(most_common_code, False):
                maj_at_k = 1
        total_maj_at_k += maj_at_k

        all_total_samples.append(current_k)
        all_correct_samples.append(correct_count)
        total_questions += 1

        results.append({
            'task_id': task_id,
            'metrics': {
                'correct_count': correct_count,
                'avg_acc': avg_acc,
                'pass@k': pass_at_k_strict,
                'maj@k': maj_at_k
            },
            'sample_evals': sample_evals
        })

    if total_questions == 0:
        print("No valid evaluation records found.")
        return

    mean_avg_acc = sum(question_avg_accs) / total_questions
    global_pass_at_k = total_pass_at_k_strict / total_questions
    global_maj_at_k = total_maj_at_k / total_questions

    variance = sum((x - mean_avg_acc) ** 2 for x in question_avg_accs) / (total_questions - 1) if total_questions > 1 else 0
    std_dev = math.sqrt(variance)
    margin_of_error = 1.96 * (std_dev / math.sqrt(total_questions)) if total_questions > 0 else 0
    ci_lower = mean_avg_acc - margin_of_error
    ci_upper = mean_avg_acc + margin_of_error

    total_arr = np.array(all_total_samples)
    correct_arr = np.array(all_correct_samples)
    unbiased_k_list = [1, 5, 8] if k >= 8 else [1]
    unbiased_metrics = {f'unbiased_pass@{cur_k}': estimate_pass_at_k(total_arr, correct_arr, cur_k).mean() for cur_k in unbiased_k_list if cur_k <= k}

    avg_generated_tokens = total_length / total_samples_parsed if total_samples_parsed > 0 else 0
    avg_effective_tokens = total_effective_length / total_samples_parsed if total_samples_parsed > 0 else 0
    avg_soft_ratio = (total_length - total_effective_length) / total_length if total_length > 0 else 0

    metrics = {
        f'avg@{k}': mean_avg_acc,
        f'pass@{k}': global_pass_at_k,
        f'maj@{k}': global_maj_at_k,
        'variance': variance,
        'std_dev': std_dev,
        '95_CI_lower': ci_lower,
        '95_CI_upper': ci_upper,
        'total_questions': total_questions,
        'total_samples_generated': total_samples_parsed,
        'avg_total_len': avg_generated_tokens,
        'avg_effective_len': avg_effective_tokens,
        'avg_soft_ratio': avg_soft_ratio,
        'timestamp': datetime.now().isoformat()
    }

    metrics.update(unbiased_metrics)

    os.makedirs(output_dir, exist_ok=True)
    final_save_path = os.path.join(output_dir, "tharp_humaneval_eval_results.json")
    with open(final_save_path, 'w', encoding='utf-8') as f:
        json.dump({'metrics': metrics, 'results': results}, f, indent=2, ensure_ascii=False)

    print("\n" + "="*50)
    print("🎯 THARP HUMANEVAL EVALUATION SUMMARY")
    print("="*50)
    print(f" Avg@{k}:      {metrics[f'avg@{k}']:.4f}  [95% CI: {ci_lower:.4f} ~ {ci_upper:.4f}]")
    print(f" Pass@{k}:     {metrics[f'pass@{k}']:.4f}")
    print(f" Maj@{k}:      {metrics[f'maj@{k}']:.4f}")
    print(f" Variance:    {variance:.4f}")
    print("-" * 50)
    for mk, mv in unbiased_metrics.items():
        print(f" {mk.capitalize().replace('_', ' ')}: {mv:.4f}")
    print("-" * 50)
    print(f" Avg Total Len: {metrics['avg_total_len']:.1f} tokens")
    print(f" Avg Hard Len:  {metrics['avg_effective_len']:.1f} tokens")
    print(f" Soft Ratio:    {metrics['avg_soft_ratio']*100:.2f}%")
    print("="*50)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=8, help="Number of samples per question")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True, help="Path to your custom HumanEval JSON file")
    args = argparse.parse_args()

    base_model = None
    checkpoint_path = args.checkpoint_path
    base_models = [
        "path/to/Qwen2.5-1.5B-Instruct",
        "path/to/Qwen2.5-3B-Instruct",
        "path/to/Qwen2.5-7B-Instruct",
    ]

    for model in base_models:
        if model.split('/')[-1] in checkpoint_path:
            base_model = model

    if base_model is None:
        base_model = "path/to/Qwen2.5-3B-Instruct"
        print(f"Warning: Could not infer base model from checkpoint, defaulting to {base_model}")

    temp_match = re.search(r'-temp([\d\.]+)', checkpoint_path)
    temperature = float(temp_match.group(1)) if temp_match else 0.5

    soft_match = re.search(r'-penalty([\d\.]+)', checkpoint_path)
    soft_penalty = float(soft_match.group(1)) if soft_match else 0.0

    soft_top_k_match = re.search(r'-topk([\d\.]+)', checkpoint_path)
    soft_top_k = int(soft_top_k_match.group(1)) if soft_top_k_match else 30

    eval_dir = f"{args.checkpoint_path}/eval_humaneval_k{args.k}_batch{args.batch_size}"
    os.makedirs(eval_dir, exist_ok=True)
    jsonl_file = os.path.join(eval_dir, "raw_trajectories_code.jsonl")

    print("\n" + ">"*15 + " STEP 1: THARP GENERATION " + "<"*15)
    generate_and_save_trajectories_tharp_code(
        model_path=base_model,
        adapter_path=args.checkpoint_path,
        data_path=args.data_path,
        temperature=temperature,
        k=args.k,
        batch_size=args.batch_size,
        soft_penalty=soft_penalty,
        soft_top_k=soft_top_k,
        output_jsonl_path=jsonl_file
    )

    print("\n" + ">"*15 + " STEP 2: EVALUATION (EXECUTION) " + "<"*15)
    evaluate_saved_trajectories_tharp_code(
        jsonl_path=jsonl_file,
        k=args.k,
        output_dir=eval_dir
    )
