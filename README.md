# TARPO: Token-Wise Latent-Explicit Reasoning via Action-Routing Policy Optimization

<!-- [![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/Anonymous)
[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](#)
[![Code](https://img.shields.io/badge/Code-Anonymous_4open-blue.svg)](https://github.com/NKU-LITI/TARPO-master.git) -->

<p align="center">
  <a href="https://github.com/Anonymous">
    <img src="https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg" alt="Code License">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/Paper-PDF-red.svg" alt="Paper">
  </a>
  <a href="https://github.com/NKU-LITI/TARPO-master.git">
    <img src="https://img.shields.io/badge/Code-Anonymous_4open-blue.svg" alt="Code">
  </a>
</p>

<!-- **Authors:** Liting Zhang, Shiwan Zhao, Xuyang Zhao, Zichen Xu, Jianye Wang, Qicheng Li
**Affiliation:** TMCC, College of Computer Science, Nankai University, Tianjin, China -->

---

<!-- ## 💡 Introduction

Latent reasoning has emerged as a promising alternative to discrete Chain-of-Thought (CoT) in large language models (LLMs), enabling more expressive reasoning by operating over continuous representations. However, the inherently deterministic nature of continuous representations limits policy exploration in reinforcement learning (RL).

To address this, we propose **TARPO** (Token-Wise Latent-Explicit Reasoning via Action-Routing Policy Optimization), a pure RL framework that adaptively switches between discrete token generation and continuous latent reasoning at each step. TARPO introduces a lightweight action head router that observes the current hidden state and samples a routing decision from a binary mode-selection space, preserving the stochasticity of discrete token sampling from the vocabulary. The LLM backbone and router are jointly optimized end-to-end with a shared group-relative advantage signal. -->

<!-- ## 🖼️ Overview -->

**TARPO** is a pure reinforcement learning framework that adaptively switches between discrete token generation and continuous latent reasoning at each step. By introducing a lightweight action head router, it overcomes the deterministic bottleneck of continuous representations and preserves policy exploration stochasticity, jointly optimizing the LLM backbone and router end-to-end to consistently outperform existing RL baselines.

<!-- ![Overview of the TARPO framework](path/to/your/Figure1.png) -->
<p align="center">
  <!-- 左图 (a) -->
  <img src="assert/a.png" width="49%" alt="Token-Wise Action Routing" />
  <!-- 右图 (b) -->
  <img src="assert/b.png" width="47%" alt="Action-Routing Policy Optimization" />
</p>
<p align="center">
  <em>Figure 1: Overview of the TARPO framework. (a) During reasoning, a lightweight Action Head receives the current hidden state and routes the next step to either discrete token generation (hard) or continuous latent reasoning (soft). (b) The framework is trained end-to-end with a shared group-relative advantage signal, which jointly updates the LLM backbone and the action head from sampled hybrid rollouts.</em>
</p>
<!-- 
*Figure 1: Overview of the TARPO framework. (a) During reasoning, a lightweight Action Head receives the current hidden state and routes the next step to either discrete token generation (hard) or continuous latent reasoning (soft)[cite: 81]. (b) The framework is trained end-to-end with a shared group-relative advantage signal, which jointly updates the LLM backbone and the action head from sampled hybrid rollouts[cite: 82].* -->

<!-- ## ✨ Main Contributions

* **Token-Wise Adaptive Routing:** We propose TARPO, a pure reinforcement learning framework for token-wise latent-explicit reasoning, which adaptively switches between discrete token generation and continuous latent reasoning.
* **End-to-End Optimization:** We formulate reasoning-mode selection as a learnable action-routing policy and introduce a joint optimization objective that end-to-end trains the router and the LLM backbone.
* **Superior Performance:** Extensive experiments and analyses demonstrate that TARPO consistently outperforms existing explicit and latent reasoning RL baselines, while learning adaptive switching behaviors across diverse benchmarks. -->

## 🚀 Training

To train the TARPO model on the MATH dataset, run the following command. The action bias is initialized to prioritize hard token generation slightly, as described in the methodology.

```bash
python tarpo_math.py \
    --group_size 8 \
    --gradient_accumulation_steps 1 \
    --per_device_train_batch_size 64 \
    --action_bias 4.6 0.0
```

## 📊 Evaluation

To evaluate the trained TARPO model on the MATH benchmark, use the following command. Please ensure you replace `${EXP_PATH}` with the actual path to your saved model checkpoints.

```bash
CUDA_VISIBLE_DEVICES=6 python eval_tarpo_math_avg.py \
  --checkpoint_path "${EXP_PATH}" \
  --batch_size 2 \
  --k 32
```

## 📝 Citation

If you find TARPO useful in your research, please consider citing our paper:

<!-- ```bibtex
@inproceedings{tarpo,
  title={TARPO: Task-Aware Routing for Prompt-based Reasoning},
  author={Zhang, Yifan and Wang, Zhihao and Chen, Yujia and Chen, Yilun and Liu, Zhiyuan and Sun, Maosong},
  booktitle={International Conference on Learning Representations},
  year={2024}, url={https://openreview.net/forum?id=8j6z2X1qg9} } -->