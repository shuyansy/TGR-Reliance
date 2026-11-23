# When Does Temporal Grounding Help Video Reasoning?

Code and data for the paper *When Does Temporal Grounding Help Video Reasoning?* (under review).

Temporally grounded reasoning (TGR) models predict the video span that supports their answer and then answer.
We ask whether the answer actually **relies** on that span. This repository provides the reliance metric **ReGAP**,
the controlled benchmark **ReliaNeedle**, the evidence-certified training data **EviCe**, and our training recipes and models.

| Component | Status |
|---|---|
| ReGAP metric (`regap/`) | available |
| Per-item ReliaNeedle results of the 12 models in Table 1 and of our two models (`evaluation/examples/relianeedle/`) | available |
| ReliaNeedle evaluation for new models (`evaluation/eval_relianeedle.py`) | available |
| ReliaNeedle annotations (`relianeedle/relianeedle.json`) | available |
| ReliaNeedle videos | coming soon |
| EviCe training data | coming soon |
| SFT and GRPO training code (Qwen2.5-VL-7B, Qwen3-VL-8B; single-turn and multi-turn) | coming soon |
| Model weights (SFT + GRPO with the grounding reward) | coming soon |
| Video-Oasis evaluation | coming soon |

## ReGAP

For every item the model answers once with the full video $v$. Let $p(\cdot)$ be its probability for its own answer
(renormalised over the options), $\hat s$ its predicted span and $r_1,\dots,r_K$ random contiguous blocks with the same
number of frames, placed outside $\hat s$. We delete each of them separately, keep the prompt fixed, and compute

$$c = \operatorname{logit}\Big(\tfrac1K\sum_k p(v\setminus r_k)\Big) - \operatorname{logit}\big(p(v\setminus \hat s)\big),\qquad
\mathrm{ReGAP} = \frac{\mathbb E[c]}{\tau},\quad \tau = \mathbb E\big[|\operatorname{logit} p(v)|\big].$$

ReGAP = 0: the predicted span matters no more than random frames. ReGAP = 1: deleting it typically brings the answer
probability to about 0.5. Conventions (clipping, refusals, wide spans) are documented in [`regap/metric.py`](regap/metric.py)
and [`evaluation/utils/deletion.py`](evaluation/utils/deletion.py).

```python
import json, regap
items = [json.loads(l) for l in open("evaluation/examples/relianeedle/qwen2.5-vl-7b.jsonl")]
print(regap.summarize(items))   # acc, r_at_0.3, regap, regap_hit, regap_miss, ...
print(regap.bootstrap(items))   # 95% CI with tau re-estimated in every resample
```

## Reproduce Table 1 (no GPU)

```bash
python evaluation/compute_regap.py evaluation/examples/relianeedle/*.jsonl
```

| Model | Acc. | R@0.3 (%) | ReGAP All | Hit | Miss |
|---|---|---|---|---|---|
| Qwen2.5-VL-7B | 57.2 | 46.0 | 0.598 | 0.945 | 0.302 |
| Qwen3-VL-4B | 57.4 | 55.5 | 0.711 | 1.026 | 0.320 |
| VisionCoach | 65.8 | 62.3 | 0.468 | 0.763 | −0.019 |
| VideoChat-R1.5 | 61.7 | 47.1 | 0.438 | 0.863 | 0.060 |
| VideoChat-R1 | 52.6 | 46.8 | 0.375 | 0.709 | 0.081 |
| Open-o3-Video | 66.8 | 42.9 | 0.271 | 0.764 | −0.100 |
| Video-Thinker | 54.1 | 17.5 | −0.011 | 0.384 | −0.095 |
| Video-o3 | 67.3 | 64.7 | 0.611 | 0.700 | 0.448 |
| LOVE-R1 | 58.5 | 67.1 | 0.493 | 0.628 | 0.218 |
| Conan | 60.0 | 19.6 | 0.240 | 0.489 | 0.180 |
| VideoZoomer | 64.3 | 7.7 | 0.195 | 0.693 | 0.153 |
| Open-o3-Video-v2 | 65.5 | 77.4 | 0.597 | 0.651 | 0.413 |
| Ours, Qwen2.5-VL-7B (SFT + GRPO, grounding reward) | 60.8 | 81.1 | 1.147 | 1.210 | 0.874 |
| Ours, Qwen3-VL-8B (SFT + GRPO, grounding reward) | 64.6 | 75.7 | 1.225 | 1.265 | 1.101 |

Each line of the example files is one item: `id`, `gt_span`, `pred_spans`, `correct`, `n_options`, `p_full`,
`p_span_deleted` (null if the model gave no option letter after the deletion), `p_random_deleted` and `status`.
The same format is written by the evaluation script below, so you can add your own model to the table.

## Evaluate a model on ReliaNeedle

Requirements: Python ≥ 3.10, `vllm==0.11.0`, `transformers==4.57.1`, `torch==2.8.0`, `torchvision`, `decord`, `opencv-python`, `Pillow`, `numpy`.

```bash
# backbones of Table 1: 1 fps, at most 64 frames, long side 768 px, time-span prompt, at most 64 new tokens
python evaluation/eval_relianeedle.py --model Qwen/Qwen2.5-VL-7B-Instruct --sampler grid --prompt tga --max_tokens 64 \
    --annotations relianeedle/relianeedle.json --video_root /path/to/ReliaNeedle --out results/qwen2.5-vl-7b.jsonl
# our models: official Qwen sampler (1 fps, at most 128 frames), training prompt, at most 256 new tokens
python evaluation/eval_relianeedle.py --model /path/to/ours-qwen2.5-vl-7b --sampler qwen25 --prompt train --max_tokens 256 \
    --annotations relianeedle/relianeedle.json --video_root /path/to/ReliaNeedle --out results/ours.jsonl
python evaluation/compute_regap.py results/*.jsonl
```

`--num_shards N --shard_id i` splits the items over GPUs (concatenate the outputs), and `--resume` continues an interrupted run.
For a new model, pass its prompt with `--system` / `--suffix`. The script expects the predicted span as `[start, end]`
(seconds) inside `<think></think>` and the option letter inside `<answer></answer>`; for other output formats adapt
`evaluation/utils/parsing.py` (span and answer parsing, and the position of the answer token whose probabilities are read).

## ReliaNeedle

725 four-option questions. Each video inserts a real action clip (the *needle*, from STAR / Charades) into an unrelated
Video-MME video (the *haystack*); the question can only be answered from the needle. `relianeedle/relianeedle.json` lists
for every item the question, options, answer letter, needle span (seconds) and, for 646 of the 725 items, the source videos
and insertion position. The videos will be released after the Video-MME terms are cleared.

## License

Code: Apache License 2.0 (see `LICENSE`); `evaluation/utils/vision_process.py` comes from Video-Oasis (MIT) and Qwen
(Apache-2.0), see `THIRD_PARTY_NOTICES.md`. ReliaNeedle and EviCe are for research use only; the terms of their source
datasets (Video-MME, Charades / STAR, and the EviCe sources) apply.
