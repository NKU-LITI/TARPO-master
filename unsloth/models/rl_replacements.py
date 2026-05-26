# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__all__ = [
    "RL_EXTRA_ARGS",
    "RL_FUNCTIONS",
    "RL_PRE_ITEMS",
    "RL_CONFIG_CHANGES",
    "RL_METRICS_CHANGES",
]

import re
import torch
import inspect
from collections import defaultdict
from unsloth_zoo.rl_replacements import RL_REPLACEMENTS

import torch._dynamo
torch._dynamo.config.cache_size_limit = 1000000
torch._dynamo.config.accumulated_cache_size_limit = 1000000

RL_EXTRA_ARGS      = defaultdict(list)
RL_FUNCTIONS       = defaultdict(list)
RL_PRE_ITEMS       = defaultdict(list)
RL_CONFIG_CHANGES  = defaultdict(list)
RL_METRICS_CHANGES = defaultdict(list)

torch_compile_options = {
    "epilogue_fusion"   : True,
    "max_autotune"      : True,
    "shape_padding"     : True,
    "trace.enabled"     : False,
    "triton.cudagraphs" : False,
}

# Check untrained tokens
def sft_trainer_fix_untrained_tokens(call_args, extra_args):
    if "model" in call_args and "train_dataset" in call_args:
        fix_tokenizer = \
        "IGNORED_TOKENIZER_NAMES = os.environ.get('UNSLOTH_IGNORED_TOKENIZER_NAMES', '').split('\\n')\n"\
        "from unsloth_zoo.tokenizer_utils import fix_untrained_tokens\n"\
        "from unsloth_zoo.training_utils  import fix_zero_training_loss\n"\
        "if 'tokenizer' not in locals(): tokenizer = processing_class\n"\
        "fix_untrained_tokens(model, tokenizer, train_dataset, IGNORED_TOKENIZER_NAMES, eps = 1e-16)\n"\
        "fix_zero_training_loss(model, tokenizer, train_dataset)\n"
        return fix_tokenizer
    return ""
pass
RL_EXTRA_ARGS["sft_trainer"].append(sft_trainer_fix_untrained_tokens)


# Remove DPO columns which might randomnly be tokenized
def dpo_trainer_fix_columns(call_args, extra_args):
    if "model" in call_args and "train_dataset" in call_args:
        fix_dpo = \
        "if hasattr(train_dataset, 'column_names'):\n"\
        "    column_names = set(train_dataset.column_names)\n"\
        "    check = ['chosen', 'rejected', 'prompt', 'chosen_input_ids', 'chosen_attention_mask',\n"\
        "             'chosen_labels', 'rejected_input_ids', 'rejected_attention_mask', 'rejected_labels',\n"\
        "             'prompt_input_ids', 'prompt_attention_mask']\n"\
        "    if all(x in column_names for x in check):\n"\
        "        train_dataset = train_dataset.remove_columns(['chosen', 'rejected', 'prompt'])\n"\
        "    del check, column_names\n"
        return fix_dpo
    return ""
pass
RL_EXTRA_ARGS["dpo_trainer"].append(dpo_trainer_fix_columns)


# Fix tokenizer double BOS
def sft_trainer_prepare_dataset(function_name, function):
    if  function_name != "_prepare_non_packed_dataloader" and \
        function_name != "_prepare_dataset": return function

    fast_sft_prepare_dataset = RL_REPLACEMENTS.get("sft_prepare_dataset", None)
    if fast_sft_prepare_dataset is not None:
        params = inspect.signature(fast_sft_prepare_dataset).parameters.keys()
        params = ".*?".join(params)
        matched = re.match(
            r"[\s]{0,}def _prepare_dataset\(.*?" + params + r".*?\)",
            function,
            flags = re.MULTILINE | re.DOTALL,
        )
        if matched:
            # Use fast version!
            function = inspect.getsource(fast_sft_prepare_dataset)
            function = function.split("\n")
            function = "\n".join(" "*4 + x for x in function)
            function = function.replace("def sft_prepare_dataset", "def _prepare_dataset")
            return function
        pass
    pass

    check_text = \
    "if 'tokenizer'          not in locals(): tokenizer = processing_class\n"\
    "if 'formatting_func'    not in locals(): raise RuntimeError('Unsloth: Please file a bug report - `formatting_func` does not exist!')\n"\
    "if 'dataset_text_field' not in locals() and 'args' in locals(): dataset_text_field = args.dataset_text_field\n"\
    "if 'dataset_text_field' not in locals(): raise RuntimeError('Unsloth: Please file a bug report - `dataset_text_field` does not exist!')\n"\
    "test_text = dataset[0][dataset_text_field] if (formatting_func is None and dataset_text_field is not None) else formatting_func(dataset[0])[0]\n"\
    "chat_template = getattr(tokenizer, 'chat_template', None)\n"\
    "chat_template = '' if chat_template is None else chat_template\n"\
    "has_bos_token_already = (test_text.startswith(tokenizer.bos_token) or tokenizer.bos_token in chat_template) "\
    "if getattr(tokenizer, 'bos_token', None) is not None else False\n"\
    "if 'add_special_tokens' not in locals() and has_bos_token_already:\n"\
    "    from functools import partial\n"\
    "    tokenizer_call = tokenizer.__call__\n"\
    "    tokenizer.__call__ = partial(tokenizer_call, add_special_tokens = False)\n"\
    "    processing_class = tokenizer\n"\
    "else:\n"\
    "    tokenizer_call = None\n"\
    "    add_special_tokens = False if has_bos_token_already else locals().get('add_special_tokens', False)\n"

    check_text = check_text.split("\n")
    check_text = "\n".join(" "*8 + x for x in check_text)
    check_text = check_text.rstrip() + "\n"

    # .*? matches first match. .+? matches final match.
    replacer = re.findall(
        r"def " + function_name + r"\(.*?\).*?\:\n",
        function,
        flags = re.MULTILINE | re.DOTALL,
    )
    if len(replacer) != 0:
        replacer = replacer[0]
        function = function.replace(replacer, replacer + check_text)
    pass

    # Return tokenizer's original state
    return_state = "if tokenizer_call is not None: tokenizer.__call__ = tokenizer_call\n"
    function = re.sub(
        r"\n([ ]{4,})(return .*?[\s]{0,})$",
        rf"\1{return_state}\1\2",
        function,
    )
    return function
pass
RL_FUNCTIONS["sft_trainer"].append(sft_trainer_prepare_dataset)


# Ignore mean_token_accuracy since it needs logits
# We override it directly with our version
def sft_trainer_compute_loss(function_name, function):
    if  function_name != "compute_loss": return function

    def compute_loss(self, model, inputs, return_outputs = False, num_items_in_batch = None):
        outputs = super().compute_loss(
            model,
            inputs,
            return_outputs = return_outputs,
            num_items_in_batch = num_items_in_batch,
        )
        return outputs
    pass

    function = inspect.getsource(compute_loss)
    return function
pass
RL_FUNCTIONS["sft_trainer"].append(sft_trainer_compute_loss)


# Autocast precision for GRPO
def grpo_trainer__prepare_inputs(function_name, function):
    if  function_name != "_prepare_inputs": return function

    if "with torch.inference_mode()" not in function: return function

    # Add mixed precision training
    function = function.replace(
        "with torch.inference_mode():",

        "with torch.inference_mode(), "\
        "torch.amp.autocast(device_type = 'cuda', "\
        "dtype = ((torch.float16 if os.environ.get('ACCELERATE_MIXED_PRECISION', 'fp16') == 'fp16' else torch.bfloat16) "\
        "if not torch.is_autocast_enabled('cuda') else nullcontext())"\
        "if os.environ.get('UNSLOTH_FORCE_FLOAT32', '0') == '0' else torch.float16):",
    )

    # Disable attaching a float32 conversion hook which upcasts logits to FP32
    function = function.replace(
        "self.accelerator.unwrap_model(self.model)",
        "self.accelerator.unwrap_model(self.model, keep_fp32_wrapper = False)",
    )
    return function
pass
RL_FUNCTIONS["grpo_trainer"].append(grpo_trainer__prepare_inputs)


# Remove _move_model_to_vllm
def grpo_trainer__move_model_to_vllm(function_name, function):
    if  function_name != "_move_model_to_vllm": return function

    def _move_model_to_vllm(self, *args, **kwargs): return None

    function = inspect.getsource(_move_model_to_vllm)
    return function
pass
RL_FUNCTIONS["grpo_trainer"].append(grpo_trainer__move_model_to_vllm)


# Edit _get_per_token_logps to handle mixed precision
def grpo_trainer__get_per_token_logps(function_name, function):
    if  function_name != "_get_per_token_logps": return function
    
    def _get_per_token_logps(self, model, input_ids, inputs_embeds, attention_mask, logits_to_keep, use_soft_input_mask=None):
        if os.environ.get('UNSLOTH_USE_NEW_MODEL', '0') == '0':
            return None
        if not hasattr(self, '_autocast_dtype'):
            self._autocast_dtype = torch.float16 if os.environ.get('ACCELERATE_MIXED_PRECISION', 'fp16') == 'fp16' else torch.bfloat16
            if os.environ.get('UNSLOTH_FORCE_FLOAT32', '0') == '1': self._autocast_dtype = torch.float16
        
        with torch.amp.autocast(device_type = 'cuda', dtype = self._autocast_dtype):
            outputs = model(
                input_ids=input_ids, 
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask, 
                logits_to_keep=logits_to_keep + 1,
                use_soft_input_mask=use_soft_input_mask
            )

            logits = outputs.logits
            logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

            input_ids = input_ids[:, -logits_to_keep:]
            # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
            # See https://github.com/huggingface/trl/issues/2770
            logits = logits[:, -logits_to_keep:]

            action_logits = getattr(outputs, 'action_logits', None)
            if action_logits is not None:
                action_logits = action_logits[:, :-1, :]
                action_logits = action_logits[:, -logits_to_keep:]

            return logits, action_logits
        pass
    pass

    function = inspect.getsource(_get_per_token_logps)
    return function
pass
RL_FUNCTIONS["grpo_trainer"].append(grpo_trainer__get_per_token_logps)


# Custom compiled GRPO loss - creates 3 Triton kernels
# def grpo_compute_loss(old_logits, new_logits, input_ids, mask, beta, advantages):
def grpo_compute_loss(old_logits, new_logits, 
    old_action_logits, new_action_logits,
    input_ids, mask, action_mask,
    beta, advantages, action_advantages, action_loss_weight=0.1
):
    # All Unsloth Zoo code licensed under LGPLv3
    old_logits = old_logits.to(torch.float32)
    new_logits = new_logits.to(torch.float32)
    input_ids  = input_ids.unsqueeze(-1)

    if action_mask is not None:
        is_soft = action_mask.float()
        is_hard = 1.0 - is_soft
    else:
        is_soft = torch.zeros_like(mask, dtype=torch.float32)
        is_hard = torch.ones_like(mask, dtype=torch.float32)

    old_x = torch.gather(old_logits, dim = -1, index = input_ids).squeeze(-1)
    new_x = torch.gather(new_logits, dim = -1, index = input_ids).squeeze(-1)
    old = old_x - torch.logsumexp(old_logits, dim = -1) 
    new = new_x - torch.logsumexp(new_logits, dim = -1)

    # Reverse KL
    kl_token = torch.exp(old - new) - (old - new) - 1.0
    ratio_token = torch.exp(new - new.detach())
    
    token_loss = ratio_token * advantages.unsqueeze(1) - beta * kl_token 

    new_probs = torch.softmax(new_logits, dim=-1)
    new_log_probs = torch.log_softmax(new_logits, dim=-1)
    step_token_entropy = -torch.sum(new_probs * new_log_probs, dim=-1)

    if old_action_logits is not None and new_action_logits is not None and action_mask is not None:

        new_act_probs = torch.softmax(new_action_logits, dim=-1)
        new_act_log_probs = torch.log_softmax(new_action_logits, dim=-1)
        step_action_entropy = -torch.sum(new_act_probs * new_act_log_probs, dim=-1)

        action_labels = action_mask.long().unsqueeze(-1)

        old_act_x = torch.gather(old_action_logits, dim=-1, index=action_labels).squeeze(-1)
        new_act_x = torch.gather(new_action_logits, dim=-1, index=action_labels).squeeze(-1)
        
        old_act_logp = old_act_x - torch.logsumexp(old_action_logits, dim=-1)
        new_act_logp = new_act_x - torch.logsumexp(new_action_logits, dim=-1)

        # Action Policy Ratio
        ratio_action = torch.exp(new_act_logp - new_act_logp.detach())
        kl_action = torch.exp(old_act_logp - new_act_logp) - (old_act_logp - new_act_logp) - 1.0
        action_loss = action_loss_weight * ratio_action * action_advantages.unsqueeze(1) - beta * kl_action

        kl_total = kl_token
    
    else:
        action_loss = 0.0
        kl_total = kl_token
        step_action_entropy = torch.zeros_like(step_token_entropy)

    loss_i = -(token_loss + action_loss)

    # === masking & reduction ===
    mask = mask.to(torch.float32)
    n_mask_per_reward = torch.clamp(mask.sum(1), min=1.0)

    # See https://github.com/huggingface/trl/pull/2881
    loss_per_reward = (loss_i * mask).sum(1) / n_mask_per_reward
    loss = loss_per_reward.mean()  # per-response loss
    
    # Get metrics as well which are folded
    with torch.inference_mode():
        completion_length = n_mask_per_reward.mean()
        mean_kl_per_reward = (kl_total * mask).sum(1) / n_mask_per_reward
        mean_kl = mean_kl_per_reward.mean()

        mean_token_entropy_per_reward = (step_token_entropy * mask).sum(1) / n_mask_per_reward
        mean_token_entropy = mean_token_entropy_per_reward.mean()
        mean_action_entropy_per_reward = (step_action_entropy * mask).sum(1) / n_mask_per_reward
        mean_action_entropy = mean_action_entropy_per_reward.mean()
    pass
    return loss, completion_length, mean_kl, mean_token_entropy, mean_action_entropy
pass
RL_REPLACEMENTS["grpo_compute_loss"]      = grpo_compute_loss
RL_REPLACEMENTS["grpo_compute_loss_slow"] = \
    f"@torch.compile(dynamic = True, fullgraph = True, options = torch_compile_options)\n"\
    f"{inspect.getsource(grpo_compute_loss)}"
RL_REPLACEMENTS["grpo_compute_loss_slow"] = \
    RL_REPLACEMENTS["grpo_compute_loss_slow"].replace(
        "def grpo_compute_loss",
        "def grpo_compute_loss_slow",
)

# Unsloth's memory efficient GRPO implementation
class UnslothEfficientGRPO(torch.autograd.Function):
    # All Unsloth Zoo code licensed under LGPLv3
    @staticmethod
    def forward(ctx, _new_hidden_states, _old_hidden_states, lm_head, _input_ids, _mask, _advantages, _action_advantages, beta, scaler = None, n_chunks = 1,
                _new_action_logits = None, _old_action_logits = None, _action_mask = None, action_loss_weight=0.1):
        def compute_loss(new_hidden_states, old_hidden_states, input_ids, mask, advantages, action_advantages, scaling, 
                        new_action_logits, old_action_logits, action_mask, act_loss_weight_j):

            new_logits = torch.matmul(new_hidden_states, lm_head.t())
            new_logits = new_logits[:, :-1, :] # exclude the last logit: it corresponds to the next token pred
            old_logits = torch.matmul(old_hidden_states, lm_head.t())
            old_logits = old_logits[:, :-1, :] # exclude the last logit: it corresponds to the next token pred
            
            if new_action_logits is not None:
                new_act_logits = new_action_logits[:, :-1, :]
                old_act_logits = old_action_logits[:, :-1, :]
                act_mask = action_mask
            else:
                new_act_logits, old_act_logits, act_mask = None, None, None
            
            loss, completion_length, mean_kl, mean_token_entropy, mean_action_entropy = grpo_compute_loss(
                old_logits, new_logits, 
                old_act_logits, new_act_logits,
                input_ids, mask, act_mask,
                beta, advantages, 
                action_advantages,
                act_loss_weight_j,
            )
            # Scale loss if needed for mixed precision training
            scaled_loss = loss * scaling
            # Must add .loss.detach otherwise autograd uses 2x VRAM
            return scaled_loss, (loss.detach(), completion_length, mean_kl, mean_token_entropy, mean_action_entropy,)
        pass

        device =_new_hidden_states.device
        grad_inputs = torch.empty_like(_new_hidden_states)

        if _new_action_logits is not None:
            grad_action_inputs = torch.empty_like(_new_action_logits)
        else:
            grad_action_inputs = None

        accumulated_loss              = torch.zeros(1, device = device)
        accumulated_completion_length = torch.zeros(1, device = device)
        accumulated_mean_kl           = torch.zeros(1, device = device)
        accumulated_mean_token_entropy = torch.zeros(1, device = device)
        accumulated_mean_action_entropy = torch.zeros(1, device = device)

        def accumulate_chunk(new_hidden_states_j, old_hidden_states_j, new_action_logits_j, old_action_logits_j, 
                            input_ids_j, mask_j, action_mask_j, advantages_j, action_advantages_j, scaling, act_loss_weight_j):

            (chunk_grad_input,chunk_grad_action,), (chunk_loss, (unscaled_loss, chunk_completion_length, chunk_mean_kl, chunk_mean_token_entropy, chunk_mean_action_entropy,)) = torch.func.grad_and_value(
                compute_loss,
                argnums = (0,7,), 
                has_aux = True,
            )( 
                new_hidden_states_j,
                old_hidden_states_j,
                input_ids_j,
                mask_j,
                advantages_j,
                action_advantages_j,
                scaling,
                new_action_logits_j,
                old_action_logits_j,
                action_mask_j,
                act_loss_weight_j,
            )
            accumulated_loss             .add_(unscaled_loss)
            accumulated_completion_length.add_(chunk_completion_length)
            accumulated_mean_kl          .add_(chunk_mean_kl)
            accumulated_mean_token_entropy.add_(chunk_mean_token_entropy)
            accumulated_mean_action_entropy.add_(chunk_mean_action_entropy)
            return chunk_grad_input, chunk_grad_action
        pass

        accumulate_chunk = torch.compile(
            accumulate_chunk,
            fullgraph = True,
            options = torch_compile_options,
        )

        grad_inputs_chunks = torch.chunk(grad_inputs,        chunks = n_chunks, dim = 0)
        new_hidden_states  = torch.chunk(_new_hidden_states, chunks = n_chunks, dim = 0)
        old_hidden_states  = torch.chunk(_old_hidden_states, chunks = n_chunks, dim = 0)
        input_ids          = torch.chunk(_input_ids,         chunks = n_chunks, dim = 0)
        mask               = torch.chunk(_mask,              chunks = n_chunks, dim = 0)
        advantages         = torch.chunk(_advantages,        chunks = n_chunks, dim = 0)
        action_advantages  = torch.chunk(_action_advantages, chunks = n_chunks, dim = 0)

        if _new_action_logits is not None:
            _action_mask = _action_mask * _mask
            new_action_logits  = torch.chunk(_new_action_logits, chunks = n_chunks, dim = 0) 
            old_action_logits  = torch.chunk(_old_action_logits, chunks = n_chunks, dim = 0) 
            action_mask        = torch.chunk(_action_mask,       chunks = n_chunks, dim = 0) 
            grad_action_inputs_chunks = torch.chunk(grad_action_inputs, chunks = n_chunks, dim = 0)
        else:
            new_action_logits, old_action_logits, action_mask = [None]*n_chunks, [None]*n_chunks, [None]*n_chunks
            grad_action_inputs_chunks = [None] * n_chunks

        device = _new_hidden_states.device
        act_loss_weight_t = torch.tensor(action_loss_weight, device=device)

        # Get mixed precision scaling if seen
        scaling = scaler.get_scale() if scaler is not None else 1.0

        # Force torch.compile to use dynamic shapes for seqlen dim
        # mark_dynamic = lambda x: torch._dynamo.mark_dynamic(x, 1)
        mark_dynamic = lambda x: torch._dynamo.mark_dynamic(x, 1) if x is not None else None

        for (grad_inputs_j, grad_action_inputs_j, new_hidden_states_j, old_hidden_states_j, new_action_logits_j, old_action_logits_j, input_ids_j, mask_j, action_mask_j, advantages_j, action_advantages_j,) in \
            zip(grad_inputs_chunks, grad_action_inputs_chunks, new_hidden_states, old_hidden_states, new_action_logits, old_action_logits, input_ids, mask, action_mask, advantages, action_advantages): 

            mark_dynamic(new_hidden_states_j)
            mark_dynamic(old_hidden_states_j)
            mark_dynamic(new_action_logits_j) 
            mark_dynamic(old_action_logits_j) 
            mark_dynamic(input_ids_j)
            mark_dynamic(mask_j)
            mark_dynamic(action_mask_j)

            chunk_grad_hidden, chunk_grad_action = accumulate_chunk(
                new_hidden_states_j, old_hidden_states_j, new_action_logits_j, old_action_logits_j, 
                input_ids_j, mask_j, action_mask_j, advantages_j, action_advantages_j, scaling,
                act_loss_weight_t
            )
            grad_inputs_j.copy_(chunk_grad_hidden)
            if grad_action_inputs_j is not None and chunk_grad_action is not None:
                grad_action_inputs_j.copy_(chunk_grad_action)

        pass

        grad_inputs                  .div_(n_chunks)
        if grad_action_inputs is not None: 
            grad_action_inputs       .div_(n_chunks)
            action_norm = torch.norm(grad_action_inputs, p=2)
        else:
            action_norm = torch.tensor(0.0, device=device)
        
        hidden_norm = torch.norm(grad_inputs, p=2) 
        accumulated_loss             .div_(n_chunks)
        accumulated_completion_length.div_(n_chunks)
        accumulated_mean_kl          .div_(n_chunks)
        accumulated_mean_token_entropy.div_(n_chunks)
        accumulated_mean_action_entropy.div_(n_chunks)
        ctx.save_for_backward(grad_inputs, grad_action_inputs)

        return (
            accumulated_loss,
            accumulated_completion_length,
            accumulated_mean_kl,
            accumulated_mean_token_entropy,
            accumulated_mean_action_entropy,
            hidden_norm,
            action_norm,
        )
    pass

    @staticmethod
    def backward(ctx, grad_output, dcompletion_length, dmean_kl, dmean_token_entropy, dmean_action_entropy, dhidden_norm, daction_norm):
        (grad_input,grad_action,) = ctx.saved_tensors
        return ( 
            grad_input,  # 0: _new_hidden_states
            None,         # 1: _old_hidden_states
            None,         # 2: lm_head
            None,         # 3: _input_ids
            None,         # 4: _mask
            None,         # 5: _advantages
            None,         # 6: _action_advantages
            None,         # 7: beta
            None,         # 8: scaler
            None,         # 9: n_chunks
            grad_action,  # 10: _new_action_logits
            None,         # 11: _old_action_logits
            None,         # 12: _action_mask
            None,         # 13: action_loss_weight
        )
    pass
pass
RL_REPLACEMENTS["UnslothEfficientGRPO"] = UnslothEfficientGRPO

import os
import numpy as np
def grpo_accumulated_loss(
    trainer,
    input_ids,
    mixed_input_embeds, 
    use_soft_input_mask, 
    soft_topk_ids, 
    soft_topk_weights, 
    logits_to_keep,
    completion_mask,
    advantages,
    action_advantages,
    n_chunks = -1,
    action_loss_weight = 0.1, 
):
    # All Unsloth Zoo code licensed under LGPLv3
    bsz, qlen = input_ids.shape
    # Find closest multiple
    factors = [i for i in range(1, bsz + 1) if bsz % i == 0]
    if n_chunks == -1: n_chunks = bsz
    n_chunks = factors[min(np.searchsorted(factors, n_chunks), len(factors)-1)]

    mixed_dtype = torch.float16 if os.environ.get('ACCELERATE_MIXED_PRECISION', 'fp16') == 'fp16' else torch.bfloat16
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"

    completion_input_ids = input_ids[:, -logits_to_keep:]
    lm_head = trainer.model.get_output_embeddings().weight

    with torch.amp.autocast(device_type = "cuda", dtype = mixed_dtype):
        with torch.inference_mode(), trainer.accelerator.unwrap_model(trainer.model, keep_fp32_wrapper = False).disable_adapter():
            old_outputs = trainer.model(
                input_ids = input_ids, 
                logits_to_keep = logits_to_keep + 1
            )
            old_hidden_states = old_outputs.logits
            old_action_logits = getattr(old_outputs, "action_logits", None) 
        pass

        if mixed_input_embeds is not None: mixed_input_embeds = mixed_input_embeds.clone()
        if use_soft_input_mask is not None: use_soft_input_mask = use_soft_input_mask.clone()
        
        new_output = trainer.model(
            input_ids = input_ids,
            mixed_input_embeds = mixed_input_embeds,
            use_soft_input_mask = use_soft_input_mask,
            soft_topk_ids = soft_topk_ids, 
            soft_topk_weights = soft_topk_weights, 
            logits_to_keep = logits_to_keep + 1
        )
        new_hidden_states = new_output.logits
        new_action_logits = getattr(new_output, "action_logits", None)

        loss, completion_length, mean_kl, mean_token_entropy, mean_action_entropy, hidden_norm, action_norm = UnslothEfficientGRPO.apply( 
            new_hidden_states, old_hidden_states, lm_head,
            completion_input_ids, completion_mask, advantages, action_advantages, trainer.beta,
            trainer.accelerator.scaler,
            n_chunks, 
            new_action_logits, old_action_logits, use_soft_input_mask, 
            action_loss_weight
        )
        return loss, completion_length, mean_kl, mean_token_entropy, mean_action_entropy, hidden_norm, action_norm

        # Old non efficient code path
        new_logits = torch.matmul(new_hidden_states, lm_head.t())
        new_logits = new_logits[:, :-1, :] # exclude the last logit: it corresponds to the next token pred
        old_logits = torch.matmul(old_hidden_states, lm_head.t())
        old_logits = old_logits[:, :-1, :] # exclude the last logit: it corresponds to the next token pred
        loss, completion_length, mean_kl = grpo_compute_loss(
            old_logits, new_logits, completion_input_ids, completion_mask, trainer.beta, advantages,
        )
        return loss, completion_length, mean_kl
    pass
pass
RL_REPLACEMENTS["grpo_accumulated_loss"] = grpo_accumulated_loss

grpo_compute_loss      = RL_REPLACEMENTS["grpo_compute_loss"]
grpo_compute_loss_slow = RL_REPLACEMENTS["grpo_compute_loss_slow"]
UnslothEfficientGRPO   = RL_REPLACEMENTS["UnslothEfficientGRPO"]
grpo_accumulated_loss  = RL_REPLACEMENTS["grpo_accumulated_loss"]
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(grpo_compute_loss))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(UnslothEfficientGRPO))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(grpo_accumulated_loss))
RL_PRE_ITEMS["grpo_trainer"].append(grpo_compute_loss_slow)

# Edit _get_per_token_logps to handle mixed precision
def grpo_trainer_compute_loss(function_name, function):
    if  function_name != "compute_loss": return function

    def compute_loss(self, model, inputs, return_outputs = False, num_items_in_batch = None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]

        mixed_input_embeds = inputs.get("mixed_input_embeds", None) 
        use_soft_input_mask = inputs.get("use_soft_input_mask", None)
        soft_topk_ids = inputs.get("soft_topk_ids", None)
        soft_topk_weights = inputs.get("soft_topk_weights", None)
        
        topk_entropy = inputs.get("topk_entropy", None) 
        embed_dist = inputs.get("embed_dist", None)

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        # bsz, qlen = input_ids.shape
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        _input_ids = input_ids
        
        # per_token_logps = self._get_per_token_logps(model, input_ids, thinking_embeds, attention_mask, logits_to_keep)
        per_outputs = self._get_per_token_logps(model, input_ids, mixed_input_embeds, attention_mask, logits_to_keep, use_soft_input_mask)
        per_token_logps = per_outputs[0] if isinstance(per_outputs, tuple) else per_outputs

        # Compute the KL divergence between the model and the reference model
        ref_per_token_logps = inputs["ref_per_token_logps"]

        # x - x.detach() allows for preserving gradients from x
        advantages = inputs["advantages"]
        action_advantages = inputs["action_advantages"]

        action_loss_weight = getattr(self.args, "action_loss_weight", 1.0)

        input_ids = input_ids[:, -logits_to_keep:]
        if per_token_logps is not None:
            loss, completion_length, mean_kl = grpo_compute_loss_slow(
                ref_per_token_logps, per_token_logps, input_ids, completion_mask, self.beta, advantages,
            )
            grad_hidden_norm = torch.tensor(0.0)
            grad_action_norm = torch.tensor(0.0)
        else:
            loss, completion_length, mean_kl, mean_token_entropy, mean_action_entropy, grad_hidden_norm, grad_action_norm = grpo_accumulated_loss( 
                self, _input_ids, mixed_input_embeds, use_soft_input_mask, soft_topk_ids, soft_topk_weights, logits_to_keep, completion_mask, advantages,
                n_chunks = self.args.unsloth_num_chunks,
                action_advantages = action_advantages,
                action_loss_weight = action_loss_weight,
            )

        total_len = completion_mask.float().sum(1)
        if use_soft_input_mask is not None:
            valid_soft = (use_soft_input_mask.float() * completion_mask.float()).sum(1)
            mean_soft_ratio = (valid_soft / (total_len)).mean().item()
            mean_hard_ratio = 1.0 - mean_soft_ratio

            hard_len = total_len - valid_soft
            mean_hard_len = hard_len.mean().item()

            mean_topk_entropy = ((topk_entropy * completion_mask).sum() / total_len.sum()).item() 
            valid_soft_tokens = valid_soft.sum()
            mean_dist = ((embed_dist * use_soft_input_mask.float() * completion_mask).sum() / valid_soft_tokens).item() if valid_soft_tokens > 0 else 0.0

        else:
            mean_soft_ratio = 0.0
            mean_hard_ratio = 1.0
            mean_hard_len = total_len.mean().item()
            mean_topk_entropy = 0.0
            mean_dist = 0.0

        if "train" in self._metrics:
            mode = "eval" if self.control.should_evaluate else "train"
            self._metrics[mode]["completion_length"].append(completion_length.item())
            self._metrics[mode]["valid_completion_length"].append(mean_hard_len) 
            self._metrics[mode]["kl"].append(mean_kl.item())
            self._metrics[mode]['soft_ratio'].append(mean_soft_ratio)
            self._metrics[mode]['hard_ratio'].append(mean_hard_ratio)
            self._metrics[mode]['topk_entropy'].append(mean_topk_entropy) 
            self._metrics[mode]['embed_dist'].append(mean_dist)
            self._metrics[mode]['grad_hidden_norm'].append(grad_hidden_norm.item()) 
            self._metrics[mode]['grad_action_norm'].append(grad_action_norm.item()) 
            self._metrics[mode]['token_entropy'].append(mean_token_entropy.item()) 
            self._metrics[mode]['action_entropy'].append(mean_action_entropy.item())
        else:
            self._metrics["completion_length"].append(completion_length.item())
            self._metrics["valid_completion_length"].append(mean_hard_len) 
            self._metrics["kl"].append(mean_kl.item())
            self._metrics['soft_ratio'].append(mean_soft_ratio)
            self._metrics['hard_ratio'].append(mean_hard_ratio)
            self._metrics['topk_entropy'].append(mean_topk_entropy)  
            self._metrics['embed_dist'].append(mean_dist)
            self._metrics['grad_hidden_norm'].append(grad_hidden_norm.item()) 
            self._metrics['grad_action_norm'].append(grad_action_norm.item()) 
            self._metrics['token_entropy'].append(mean_token_entropy.item()) 
            self._metrics['action_entropy'].append(mean_action_entropy.item())
        return loss
    pass

    function = inspect.getsource(compute_loss)
    return function
pass
RL_FUNCTIONS["grpo_trainer"].append(grpo_trainer_compute_loss)


# https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py#L356
# TRL warns if batch size is not a multiple of num_generations -> fix this.
def grpo_trainer_fix_batch_size(RLTrainer_source, RLConfig_source):
    if "divisible by the number of generations" not in RLTrainer_source: return ""
    if "num_generations" not in RLConfig_source: return ""

    check_batch_size = \
    "div = per_device_train_batch_size // num_generations\n"\
    "if div * num_generations != per_device_train_batch_size:\n"\
    "    print('Unsloth: We now expect `per_device_train_batch_size` to be a multiple of `num_generations`.\\n"\
               "We will change the batch size of ' + str(per_device_train_batch_size) + ' to the `num_generations` of ' + str(num_generations))\n"\
    "    per_device_train_batch_size = num_generations\n"
    return check_batch_size
pass
RL_CONFIG_CHANGES["grpo_trainer"].append(grpo_trainer_fix_batch_size)


# Add other reward function names
def grpo_trainer_metrics(RLTrainer_source, RLConfig_source):
    if "reward_funcs" not in RLTrainer_source: return ""

    log_metrics = \
    "if not isinstance(reward_funcs, list): _reward_funcs = [reward_funcs]\n"\
    "else: _reward_funcs = reward_funcs\n"\
    "for reward_func in _reward_funcs:\n"\
    "    try:\n"\
    "        reward_func_name = reward_func.__name__\n"\
    "        other_metrics.append(f'rewards/{reward_func_name}')\n"\
    "    except: pass\n"
    return log_metrics
pass
RL_METRICS_CHANGES["grpo_trainer"].append(grpo_trainer_metrics)
