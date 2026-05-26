import unsloth
from unsloth import FastLanguageModel

import os
import json
import re
import math
import torch
from collections import Counter
from datetime import datetime
from datasets import load_dataset
from transformers import GenerationConfig
from tqdm import tqdm

from utils import *


def generate_and_save_trajectories(
    model_path: str,
    adapter_path: str,
    temperature: float,
    k: int,
    batch_size: int,
    num_samples: int = None,
    soft_top_k: int = 10,
    output_jsonl_path: str = "trajectories.jsonl"
):
    """
    Step 1: Generate TARPO model trajectories on GSM8K, extract soft/hard token lengths,
    and incrementally write results to disk.
    """

    # Resume from existing output if available
    existing_count = 0
    if os.path.exists(output_jsonl_path):
        with open(output_jsonl_path, 'r', encoding='utf-8') as f:
            existing_count = sum(1 for _ in f)
        print(f"Found existing trajectories file with {existing_count} records. Resuming...")

    dataset = load_dataset('openai/gsm8k', 'main')['test']
    if num_samples and len(dataset) > num_samples:
        dataset = dataset.shuffle(seed=42).select(range(num_samples))
    total_samples = len(dataset)

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

    progress_bar = tqdm(total=total_samples, initial=existing_count, desc="Generating")

    with open(output_jsonl_path, 'a', encoding='utf-8') as f_out:
        for i in range(existing_count, total_samples, batch_size):
            batch_data = dataset[i:i + batch_size]
            current_batch_size = len(batch_data['question'])

            prompts = [
                [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': q.strip()},
                ]
                for q in batch_data['question']
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

            prompt_ids = prompt_ids.repeat_interleave(k, dim=0).to(model.device)
            prompt_mask = prompt_mask.repeat_interleave(k, dim=0).to(model.device)
            prompt_length = prompt_ids.size(1)

            generated_sequences, generated_soft_mask, generated_plain_texts, action_prob = model.generate(
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
                question = batch_data['question'][j]
                true_answer = extract_hash_answer(batch_data['answer'][j])
                true_answer = process_gsm8k_answer(true_answer)

                raw_samples = []
                for sample_idx in range(k):
                    idx = j * k + sample_idx
                    gen_ids = generated_sequences[idx][prompt_length:]
                    eos_indices = (gen_ids == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
                    seq_len = eos_indices[0].item() if len(eos_indices) > 0 else len(gen_ids)

                    response = tokenizer.decode(gen_ids[:seq_len])

                    effective_len = seq_len
                    if generated_soft_mask is not None:
                        completion_mask = generated_soft_mask[idx]
                        valid_mask = completion_mask[:seq_len]
                        num_soft_tokens = valid_mask.float().sum().item()
                        effective_len = seq_len - int(num_soft_tokens)

                    raw_samples.append({
                        'full_response': response,
                        'generated_length': seq_len,
                        'effective_length': effective_len,
                    })

                record = {
                    'question': question,
                    'true_answer': true_answer,
                    'samples': raw_samples,
                }

                f_out.write(json.dumps(record) + "\n")
                f_out.flush()

            progress_bar.update(current_batch_size)

    progress_bar.close()

    del model
    del tokenizer
    torch.cuda.empty_cache()

    return output_jsonl_path


def evaluate_saved_trajectories(jsonl_path: str, k: int, output_dir: str):
    """
    Step 2: Load saved trajectories offline, compute accuracy metrics, variance,
    confidence intervals, and soft-token ratio statistics for GSM8K.
    """
    print(f"Starting offline evaluation on {jsonl_path}...")

    results = []
    question_avg_accs = []

    total_questions = 0
    total_pass_at_k = 0
    total_maj_at_k = 0

    total_length = 0
    total_effective_length = 0
    total_samples_parsed = 0

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Evaluating Metrics"):
            data = json.loads(line)
            question = data['question']
            true_answer = data['true_answer']

            correct_count = 0
            extracted_answers = []
            sample_evals = []

            for sample in data['samples']:
                response = sample['full_response']
                gen_len = sample['generated_length']
                eff_len = sample.get('effective_length', gen_len)

                extracted = extract_from_response(response)
                generated_answer = process_gsm8k_answer(extracted)
                extracted_answers.append(generated_answer)

                is_correct = generated_answer == true_answer
                if is_correct:
                    correct_count += 1

                total_length += gen_len
                total_effective_length += eff_len
                total_samples_parsed += 1

                sample_evals.append({
                    'generated_answer': generated_answer,
                    'correct': is_correct,
                    'generated_length': gen_len,
                    'effective_length': eff_len,
                })

            avg_acc = correct_count / k
            pass_at_k = 1 if correct_count > 0 else 0

            valid_answers = [ans for ans in extracted_answers if ans is not None and ans != ""]
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
                    'maj@k': maj_at_k,
                },
                'sample_evals': sample_evals,
            })

    if total_questions == 0:
        print("No valid evaluation records found.")
        return

    mean_avg_acc = sum(question_avg_accs) / total_questions
    variance = (
        sum((x - mean_avg_acc) ** 2 for x in question_avg_accs) / (total_questions - 1)
        if total_questions > 1
        else 0
    )
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
        'timestamp': datetime.now().isoformat(),
    }

    os.makedirs(output_dir, exist_ok=True)
    final_save_path = os.path.join(output_dir, "eval_metrics_final.json")
    with open(final_save_path, 'w', encoding='utf-8') as f:
        json.dump({'metrics': metrics, 'results': results}, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 50)
    print("TARPO GSM8K EVALUATION SUMMARY")
    print("=" * 50)
    print(f" Avg@{k}:      {metrics[f'avg@{k}']:.4f}  [95% CI: {ci_lower:.4f} ~ {ci_upper:.4f}]")
    print(f" Pass@{k}:     {metrics[f'pass@{k}']:.4f}")
    print(f" Maj@{k}:      {metrics[f'maj@{k}']:.4f}")
    print(f" Variance:    {variance:.4f}")
    print("-" * 50)
    print(f" Avg Total Len: {metrics['avg_total_len']:.1f} tokens")
    print(f" Avg Hard Len:  {metrics['avg_effective_len']:.1f} tokens")
    print(f" Soft Ratio:    {metrics['avg_soft_ratio'] * 100:.2f}%")
    print("=" * 50)
    print(f"Full metrics saved to: {final_save_path}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=16, help="Number of samples per question")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for parallel processing")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()

    SUPPORTED_BASE_MODELS = [
        "Qwen2.5-1.5B-Instruct",
        "Qwen2.5-3B-Instruct",
        "Qwen2.5-7B-Instruct",
        "Llama-3.1-8B-Instruct",
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

    eval_dir = f"{checkpoint_path}/eval_gsm8k_k{args.k}_batch{args.batch_size}"
    os.makedirs(eval_dir, exist_ok=True)
    jsonl_file = os.path.join(eval_dir, "raw_trajectories.jsonl")

    print("\n" + ">" * 15 + " STEP 1: GENERATION " + "<" * 15)
    generate_and_save_trajectories(
        model_path=base_model,
        adapter_path=checkpoint_path,
        temperature=temperature,
        k=args.k,
        batch_size=args.batch_size,
        num_samples=None,
        soft_top_k=soft_top_k,
        output_jsonl_path=jsonl_file,
    )

    print("\n" + ">" * 15 + " STEP 2: EVALUATION " + "<" * 15)
    evaluate_saved_trajectories(
        jsonl_path=jsonl_file,
        k=args.k,
        output_dir=eval_dir,
    )