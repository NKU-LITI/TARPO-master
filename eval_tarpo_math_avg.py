import unsloth
from unsloth import FastLanguageModel

import os
import json
import re
import math
import torch
import gc
from collections import Counter
from datetime import datetime
from datasets import load_dataset, Dataset
from transformers import GenerationConfig
from tqdm import tqdm

from utils import *


def preprocess_math(split="train", chunk_size=1000, root='data/MATH') -> Dataset:
    problems, solutions = [], []
    for folder in os.listdir(os.path.join(root, split)):
        for file in os.listdir(os.path.join(root, split, folder)):
            if file.endswith('.json'):
                with open(os.path.join(root, split, folder, file), 'r') as f:
                    entry = json.load(f)
                problems.append(entry['problem'])
                solutions.append(entry['solution'])

    dataset = Dataset.from_dict({
        'problem': problems,
        'solution': solutions,
    })
    return dataset.map(process_math, batched=True,
                       batch_size=chunk_size, load_from_cache_file=False)


def generate_and_save_trajectories(
    model_path: str,
    adapter_path: str,
    temperature: float,
    soft_top_k: int,
    k: int = 8,
    batch_size: int = 4,
    num_samples: int = None,
    output_jsonl_path: str = "trajectories.jsonl"
):
    """
    Step 1: Generate TARPO trajectories on MATH / MATH-500, extract soft/hard token
    lengths, and incrementally write results to disk.
    """

    # Resume from existing output if available
    existing_count = 0
    if os.path.exists(output_jsonl_path):
        with open(output_jsonl_path, 'r', encoding='utf-8') as f:
            existing_count = sum(1 for _ in f)
        print(f"Found existing trajectories file with {existing_count} records. Resuming...")

    dataset = preprocess_math('test', chunk_size=500)
    math500 = load_dataset('HuggingFaceH4/MATH-500')['test']
    math500_problems_set = set(p.strip() for p in math500['problem'])

    if num_samples and len(dataset) > num_samples:
        dataset = dataset.shuffle(seed=42).select(range(num_samples))
    total_samples = len(dataset)
    print(f"Total target samples: {total_samples}")

    if existing_count >= total_samples:
        print("All trajectories already generated. Skipping generation phase.")
        return output_jsonl_path

    print(f"Loading model for generation: {adapter_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=1536,
        load_in_4bit=False,
        fast_inference=False,
    )
    model.answer_start = ANSWER_START
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model.load_adapter(adapter_path)
    model = FastLanguageModel.for_inference(model)

    eos_ids = model.generation_config.eos_token_id
    if eos_ids is None:
        eos_ids = [tokenizer.eos_token_id]
    elif not isinstance(eos_ids, list):
        eos_ids = [eos_ids]

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None:
        eos_ids.append(im_end_id)
    eos_ids_tensor = torch.tensor(list(set(eos_ids)), device=model.device)

    progress_bar = tqdm(total=total_samples, initial=existing_count,
                        desc="Generating Trajectories", dynamic_ncols=True)

    with open(output_jsonl_path, 'a', encoding='utf-8') as f_out:
        for i in range(existing_count, total_samples, batch_size):
            batch_data = dataset[i:i + batch_size]
            current_batch_size = len(batch_data['problem'])

            problems = batch_data['problem']
            prompts = [
                [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': q.strip()},
                ]
                for q in problems
            ]

            formatted_prompts = [
                tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
                for p in prompts
            ]

            prompt_inputs = tokenizer(
                formatted_prompts,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            )
            prompt_ids = prompt_inputs["input_ids"]
            prompt_mask = prompt_inputs["attention_mask"]

            # Truncate long prompts to avoid OOM on MATH problems
            prompt_ids = prompt_ids[:, -512:]
            prompt_mask = prompt_mask[:, -512:]

            prompt_ids = prompt_ids.repeat_interleave(k, dim=0).to(model.device)
            prompt_mask = prompt_mask.repeat_interleave(k, dim=0).to(model.device)
            prompt_length = prompt_ids.size(1)

            with torch.no_grad():
                generated_sequences, generated_soft_masks, _ = model.generate(
                    prompt_ids,
                    attention_mask=prompt_mask,
                    generation_config=GenerationConfig(
                        do_sample=True,
                        temperature=temperature,
                        max_new_tokens=1024,
                    ),
                    processing_class=tokenizer,
                    is_inference=True,
                    use_mixed_rollout=True,
                    action_temperature=1.0,
                    soft_top_k=soft_top_k,
                    is_greedy=False,
                )

            for j in range(current_batch_size):
                question = problems[j].strip()
                true_answer = extract_boxed_answer(batch_data['solution'][j])
                true_answer = process_math_answer(true_answer)
                is_math500 = question in math500_problems_set

                raw_samples = []
                for sample_idx in range(k):
                    idx = j * k + sample_idx
                    gen_ids = generated_sequences[idx][prompt_length:]

                    eos_indices = (
                        (gen_ids.unsqueeze(1) == eos_ids_tensor).any(dim=1).nonzero(as_tuple=True)[0]
                    )
                    seq_len = (
                        eos_indices[0].item() + 1 if len(eos_indices) > 0
                        else generated_soft_masks.size(1)
                    )

                    effective_tokens = gen_ids[:seq_len]
                    response = tokenizer.decode(effective_tokens, skip_special_tokens=True)

                    valid_mask = generated_soft_masks[idx, :seq_len]
                    soft_count = valid_mask.sum().item()
                    hard_count = seq_len - soft_count

                    raw_samples.append({
                        'full_response': response,
                        'generated_length': seq_len,
                        'soft_count': soft_count,
                        'hard_count': hard_count,
                    })

                record = {
                    'question': question,
                    'true_answer': true_answer,
                    'is_math500': is_math500,
                    'samples': raw_samples,
                }

                f_out.write(json.dumps(record) + "\n")
                f_out.flush()

            progress_bar.update(current_batch_size)

            if 'prompt_ids' in locals(): del prompt_ids
            if 'prompt_mask' in locals(): del prompt_mask
            if 'generated_sequences' in locals(): del generated_sequences
            if 'generated_soft_masks' in locals(): del generated_soft_masks
            gc.collect()
            torch.cuda.empty_cache()

    progress_bar.close()
    return output_jsonl_path


def evaluate_saved_trajectories(jsonl_path: str, k: int, output_dir: str):
    """
    Step 2: Load saved trajectories offline, compute accuracy metrics, variance,
    confidence intervals, and soft-token ratio statistics for MATH and MATH-500.
    """
    print(f"Starting offline evaluation on {jsonl_path}...")

    results = []
    all_question_avg_accs = []
    math500_question_avg_accs = []

    total_questions = 0
    total_pass_at_k = 0
    total_maj_at_k = 0
    total_length = 0
    total_soft_count = 0
    total_samples_parsed = 0

    total_math500 = 0
    total_math500_pass_at_k = 0
    total_math500_maj_at_k = 0
    total_math500_length = 0
    total_math500_soft_count = 0
    total_math500_samples_parsed = 0

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Evaluating Metrics", dynamic_ncols=True):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print("\nWarning: Skipped a corrupted line in JSONL file.")
                continue

            question = data['question']
            true_answer = data['true_answer']
            is_math500 = data.get('is_math500', False)

            correct_count = 0
            extracted_answers = []
            sample_evals = []

            for sample in data['samples']:
                response = sample['full_response']
                gen_len = sample['generated_length']
                soft_count = sample.get('soft_count', 0)
                hard_count = sample.get('hard_count', gen_len)

                extracted = extract_from_response(response)
                generated_answer = process_math_answer(extracted)
                extracted_answers.append(generated_answer)

                is_correct = generated_answer == true_answer
                if is_correct:
                    correct_count += 1

                total_length += gen_len
                total_soft_count += soft_count
                total_samples_parsed += 1

                if is_math500:
                    total_math500_length += gen_len
                    total_math500_soft_count += soft_count
                    total_math500_samples_parsed += 1

                sample_evals.append({
                    'generated_answer': generated_answer,
                    'correct': is_correct,
                    'generated_length': gen_len,
                    'soft_count': soft_count,
                    'hard_count': hard_count,
                })

            avg_acc = correct_count / k
            pass_at_k = 1 if correct_count > 0 else 0

            valid_answers = [ans for ans in extracted_answers if ans is not None and ans != ""]
            maj_at_k = 0
            if valid_answers:
                most_common_answer = Counter(valid_answers).most_common(1)[0][0]
                if most_common_answer == true_answer:
                    maj_at_k = 1

            all_question_avg_accs.append(avg_acc)
            total_pass_at_k += pass_at_k
            total_maj_at_k += maj_at_k
            total_questions += 1

            if is_math500:
                math500_question_avg_accs.append(avg_acc)
                total_math500_pass_at_k += pass_at_k
                total_math500_maj_at_k += maj_at_k
                total_math500 += 1

            results.append({
                'question': question,
                'true_answer': true_answer,
                'is_math500': is_math500,
                'metrics': {
                    'avg_acc': avg_acc,
                    'pass@k': pass_at_k,
                    'maj@k': maj_at_k,
                },
                'sample_evals': sample_evals,
            })

    def calculate_stats(accs, n):
        if n == 0:
            return 0, 0, 0, 0, 0
        mean_acc = sum(accs) / n
        variance = sum((x - mean_acc) ** 2 for x in accs) / (n - 1) if n > 1 else 0
        std_dev = math.sqrt(variance)
        margin_of_error = 1.96 * (std_dev / math.sqrt(n)) if n > 0 else 0
        ci_lower = mean_acc - margin_of_error
        ci_upper = mean_acc + margin_of_error
        return mean_acc, variance, std_dev, ci_lower, ci_upper

    all_mean, all_var, all_std, all_ci_lower, all_ci_upper = calculate_stats(
        all_question_avg_accs, total_questions
    )
    m500_mean, m500_var, m500_std, m500_ci_lower, m500_ci_upper = calculate_stats(
        math500_question_avg_accs, total_math500
    )

    avg_generated_tokens = total_length / total_samples_parsed if total_samples_parsed > 0 else 0
    avg_effective_tokens = (total_length - total_soft_count) / total_samples_parsed if total_samples_parsed > 0 else 0
    global_soft_ratio = total_soft_count / total_length if total_length > 0 else 0.0

    math500_avg_generated_tokens = total_math500_length / total_math500_samples_parsed if total_math500_samples_parsed > 0 else 0
    math500_avg_effective_tokens = (total_math500_length - total_math500_soft_count) / total_math500_samples_parsed if total_math500_samples_parsed > 0 else 0
    math500_global_soft_ratio = total_math500_soft_count / total_math500_length if total_math500_length > 0 else 0.0

    metrics = {
        'total_questions': total_questions,
        f'avg@{k}': all_mean,
        f'pass@{k}': total_pass_at_k / total_questions if total_questions > 0 else 0,
        f'maj@{k}': total_maj_at_k / total_questions if total_questions > 0 else 0,
        'variance': all_var,
        'std_dev': all_std,
        '95_CI_lower': all_ci_lower,
        '95_CI_upper': all_ci_upper,
        'total_samples_generated': total_samples_parsed,
        'avg_generated_tokens': avg_generated_tokens,
        'avg_effective_tokens': avg_effective_tokens,
        'global_soft_ratio': global_soft_ratio,
        'total_math500': total_math500,
        f'math500_avg@{k}': m500_mean,
        f'math500_pass@{k}': total_math500_pass_at_k / total_math500 if total_math500 > 0 else 0,
        f'math500_maj@{k}': total_math500_maj_at_k / total_math500 if total_math500 > 0 else 0,
        'math500_variance': m500_var,
        'math500_std_dev': m500_std,
        'math500_95_CI_lower': m500_ci_lower,
        'math500_95_CI_upper': m500_ci_upper,
        'math500_total_samples_generated': total_math500_samples_parsed,
        'math500_avg_generated_tokens': math500_avg_generated_tokens,
        'math500_avg_effective_tokens': math500_avg_effective_tokens,
        'math500_global_soft_ratio': math500_global_soft_ratio,
        'timestamp': datetime.now().isoformat(),
    }

    os.makedirs(output_dir, exist_ok=True)
    final_save_path = os.path.join(output_dir, "eval_results.json")
    with open(final_save_path, 'w', encoding='utf-8') as f:
        json.dump({'metrics': metrics, 'results': results}, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 50)
    print("TARPO MATH EVALUATION SUMMARY")
    print("=" * 50)
    print("[All MATH]")
    print(f" Total:         {total_questions}")
    print(f" Avg@{k}:        {metrics[f'avg@{k}']:.4f}  [95% CI: {all_ci_lower:.4f} ~ {all_ci_upper:.4f}]")
    print(f" Pass@{k}:       {metrics[f'pass@{k}']:.4f}")
    print(f" Maj@{k}:        {metrics[f'maj@{k}']:.4f}")
    print(f" Variance:      {all_var:.4f}")
    print(f" Avg Tokens:    {avg_generated_tokens:.1f} (Full) / {avg_effective_tokens:.1f} (Effective)")
    print(f" Soft Ratio:    {global_soft_ratio:.2%}")
    print("-" * 50)
    print("[MATH-500 Subset]")
    print(f" Total:         {total_math500}")
    print(f" Avg@{k}:        {metrics[f'math500_avg@{k}']:.4f}  [95% CI: {m500_ci_lower:.4f} ~ {m500_ci_upper:.4f}]")
    print(f" Pass@{k}:       {metrics[f'math500_pass@{k}']:.4f}")
    print(f" Maj@{k}:        {metrics[f'math500_maj@{k}']:.4f}")
    print(f" Avg Tokens:    {math500_avg_generated_tokens:.1f} (Full) / {math500_avg_effective_tokens:.1f} (Effective)")
    print(f" Soft Ratio:    {math500_global_soft_ratio:.2%}")
    print("=" * 50)
    print(f"Results saved to: {final_save_path}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=8, help="Number of samples per question")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for parallel processing")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    args = parser.parse_args()

    SUPPORTED_BASE_MODELS = [
        "Qwen2.5-1.5B-Instruct",
        "Qwen2.5-3B-Instruct",
        "Qwen2.5-7B-Instruct",
    ]

    base_model = None
    checkpoint_path = args.checkpoint_path
    for model_name in SUPPORTED_BASE_MODELS:
        if model_name in checkpoint_path:
            base_model = model_name
            break

    if base_model is None:
        raise ValueError(
            f"Could not infer base model from checkpoint_path: {checkpoint_path}. "
            f"Supported models: {SUPPORTED_BASE_MODELS}"
        )

    temp_match = re.search(r'-temp([\d\.]+)', checkpoint_path)
    temperature = float(temp_match.group(1)) if temp_match else 0.5

    soft_top_k_match = re.search(r'-topk([\d\.]+)', checkpoint_path)
    soft_top_k = int(soft_top_k_match.group(1)) if soft_top_k_match else 30

    print(f"Checkpoint:    {checkpoint_path}")
    print(f"Base model:    {base_model}")
    print(f"Temperature:   {temperature} | Soft top-k: {soft_top_k}")

    eval_dir = f"{checkpoint_path}/eval_math_k{args.k}_batch{args.batch_size}"
    os.makedirs(eval_dir, exist_ok=True)
    jsonl_file = os.path.join(eval_dir, "raw_trajectories.jsonl")

    print("\n" + ">" * 15 + " STEP 1: GENERATION " + "<" * 15)
    generate_and_save_trajectories(
        model_path=base_model,
        adapter_path=checkpoint_path,
        temperature=temperature,
        soft_top_k=soft_top_k,
        k=args.k,
        batch_size=args.batch_size,
        num_samples=None,
        output_jsonl_path=jsonl_file,
    )

    print("\n" + ">" * 15 + " STEP 2: EVALUATION " + "<" * 15)
    evaluate_saved_trajectories(
        jsonl_path=jsonl_file,
        k=args.k,
        output_dir=eval_dir,
    )