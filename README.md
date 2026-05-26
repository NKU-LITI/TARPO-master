# TARPO: Token-Wise Latent-Explicit Reasoning via Action-Routing Policy Optimization

*This repository is anonymized for double-blind review.*

This repository contains the official code, training scripts, and evaluation pipelines for **TARPO**.

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

To evaluate the trained TARPO model on the MATH500 benchmark via majority voting (Maj@32), use the following command. Please ensure you replace `${EXP_PATH}` with the actual path to your saved model checkpoints.

```bash
CUDA_VISIBLE_DEVICES=6 python eval_tharp_math_avg.py \
  --data_path "data/math/math500.json" \
  --batch_size 2 \
  --k 32 \
  --checkpoint_path "${EXP_PATH}"
```
