# Your Large Vision-Language Model Only Needs A Few Attention Heads For Visual Grounding [CVPR 2025 Highlight]
## Table of Contents

- [Your Large Vision-Language Model Only Needs A Few Attention Heads For Visual Grounding \[CVPR 2025 Highlight\]](#your-large-vision-language-model-only-needs-a-few-attention-heads-for-visual-grounding-cvpr-2025-highlight)
  - [Table of Contents](#table-of-contents)
  - [TL;DR](#tldr)
  - [Highlights](#highlights)
  - [Experiment Dataset](#experiment-dataset)
  - [Requirements](#requirements)
  - [Quick Start](#quick-start)
  - [Configuration (Hydra)](#configuration-hydra)
  - [Commands](#commands)
    - [Experiments for batch (We had 10 trials of experiments with a set consisting of 1,000 samples.)](#experiments-for-batch-we-had-10-trials-of-experiments-with-a-set-consisting-of-1000-samples)
    - [Single image/query](#single-imagequery)
  - [Input JSONL Schema](#input-jsonl-schema)
  - [Outputs](#outputs)
    - [Evaluate predicted boxes](#evaluate-predicted-boxes)
  - [How Attention Is Collected](#how-attention-is-collected)
  - [Repository Layout](#repository-layout)
  - [Tips \& Troubleshooting](#tips--troubleshooting)
  - [License](#license)

## TL;DR
This repository provides a one-shot evaluation protocol designed to support the discovery and validation of our primary contribution "Localization Heads" in Large Vision-Language Models.

This repository contains tools for finding and analyzing localization heads in multimodal LLMs. The tools help identify which attention heads in a model are most responsible for localizing objects in images.

**Paper**: [https://arxiv.org/abs/2503.06287](https://arxiv.org/abs/2503.06287)

A simple, training-free pipeline to discover and use "localization heads" in LVLMs (e.g., LLaVA) for visual grounding. This overhaul removes the need to fork Transformers, uses standard Hugging Face APIs to collect attentions, and provides a clean Hydra-driven workflow for collection → analysis → visualization → bbox/mask outputs.

## Highlights

- No local Transformers fork: uses `output_attentions=True` to capture attention
- Eager attention only: stable attention tensors from standard HF backends
- Two criteria head selection:
  - Criterion-1: Value-based elbow (chord method) on head-wise image-attention sums
  - Criterion-2: Spatial Entropy (lower is better) with bottom-row focus filtering
- Combine top-K heads → smoothing → binary mask → bbox (xyxy)
- Hydra config groups: `model`, `logic`, `data` with minimal options
- Single-file and batch processing with consistent outputs

## Experiment Dataset

For our experiments, we prepared 1,000 data samples from the RefCOCO training set. The RefCOCO dataset contains images with referring expressions that uniquely identify specific objects in the images. This selected subset allowed us to comprehensively evaluate the localization capabilities of various attention heads in the model. For more detailed information about the dataset preparation and experimental setup, please refer to our paper.

## Requirements

- Python 3.9+
- GPU recommended (CUDA), but CPU is supported for smaller tests
- Install Python packages:

```
pip install -r requirements.txt
```

Notes:
- We keep `transformers` unpinned to follow the latest stable. If you hit regressions, pin a known good version (e.g., `>=4.52.3`).
- We require eager attention (no Flash Attention/SDPA for attention outputs).
- `hydra-colorlog` is used to enable the `hydra/job_logging: colorlog` config.

## Quick Start

```
# Demo or Debug: Single image + query (full pipeline: collect + analyze + viz + bbox/mask)
python pipeline.py \
  stage=pipeline \
  data.image_file=examples/images/bird.png \
  data.query="birds."
```

Optional: choose cache directory for model downloads

```
python pipeline.py \
  stage=pipeline \
  data.image_file=examples/images/cat.png \
  data.query="a cat on the floor" \
  model.cache_dir=/your/hf/cache
```

Optional: capture generated text and use attentions from the first generated token (falls back to forward attentions if unavailable)

```
python pipeline.py \
  stage=pipeline \
  data.image_file=examples/images/cat.png \
  data.query="a cat on the floor" \
  model.use_generate=true
```

## Configuration (Hydra)

Configs live under `conf/` and are composed in `conf/config.yaml`.

- `conf/model/llava15_7b.yaml`
  - `name`: Hugging Face repo id, e.g., `liuhaotian/llava-v1.5-7b`
  - `cache_dir`: optional HF cache dir (also exported to `TRANSFORMERS_CACHE`, `HF_HOME`, `HF_HUB_CACHE`)
  - `device`: `auto` | `cpu` | `cuda:<id>` (use `device_id` when `auto`)
  - `device_id`: GPU index used when `device=auto`
  - `conv_mode`: prompt template key (default: `referseg`)
  - `max_new_tokens`, `do_sample`, `num_beams`: used only if `use_generate=true`
  - `use_generate`: false by default (forward-only attentions)
  - `use_flash_attn`: false (keep eager)

- `conf/logic/selection_v1.yaml`
  - `top_k`: number of heads to combine for visualization/mask
  - `threshold.method`: `chord` (value-based elbow)
  - `threshold.min_keep`: ensure at least N heads remain after criterion-1
  - `entropy.binarize_threshold`: threshold to build components after ReLU(mean-centered)
  - `smoothing.sigma`: Gaussian sigma before combining heads
  - `mask.method`: `mean_relu` (fixed in this version)
  - `manual_heads`: optional ordered list of `{layer, head}` pairs; entries must still pass the threshold/entropy filters, otherwise the automatic selector is used

- `conf/data/local_examples.yaml`
  - `image_file`, `query`, `attention_file`
  - `data_file`: JSONL path for batch mode
  - `process_all`, `start_index`, `end_index`
  - `output_dir`: outputs root (default `_overhaul/outputs/localization_heads`)
  - `visualize_batch`: save figures for each batch item
  - `batch_attention_dir`: optional override when running `stage=batch_analyze`

## Commands

### Experiments for batch (We had 10 trials of experiments with a set consisting of 1,000 samples.)
- Batch mode (JSONL schema below)

```python
python pipeline.py stage=batch \
  data.data_file=examples/localization_data.jsonl \
  data.process_all=true
```

- Analyze existing batch outputs (summarize selected heads for paper tables)

```python
python pipeline.py stage=batch_analyze \
  data.batch_attention_dir=outputs/results/liuhaotian-llava-v1.5-7b
```

### Single image/query
- Full pipeline 

```pyhton
python pipeline.py stage=pipeline \
  data.image_file=examples/images/bird.png \
  data.query="the bird on the branch"
```

- Collect only 

```python
python pipeline.py stage=collect \
  data.image_file=examples/images/dog.png \
  data.query="a small dog wearing a collar"
```

- Analyze an existing attention file

```python
python pipeline.py stage=analyze \
  data.attention_file=_overhaul/outputs/localization_heads/liuhaotian-llava-v1.5-7b/sample.pkl
```

- Visualize an existing attention file (also writes bbox/mask)

```python
python pipeline.py stage=visualize \
  data.attention_file=_overhaul/outputs/localization_heads/liuhaotian-llava-v1.5-7b/sample.pkl
```

## Input JSONL Schema

Each line is a JSON object:

```
{"id": "example_1", "prompt": "a black cat on the sofa", "image_path": "examples/images/cat.png"}
```

Required keys: `id`, `prompt`, `image_path`.

## Outputs

Under `data.output_dir/<model_name_sanitized>/` for single items (or per-id for batch):

- `<id>.pkl.gz`: compressed attention dict with `attn` tensor `[L, H, 1, V]` (float16 by default) and `meta`
- `<id>_analysis.pkl`: ranked head list (top by spatial entropy)
- `<id>_topK.png`: image + top-K attention maps
- `<id>_mask.png`: binary pseudo-mask at image resolution
- `<id>_bbox.json`: bbox (xyxy), image size, and selected head details
- `<id>_meta.json`: convenience metadata dump (set `storage.attention.keep_meta_pickle=true` to keep `_meta.pkl`)

`meta` includes: `image_file`, `query`, `model_name`, `image_size`, `vis_len`, `patch_size`, `num_layers`, `num_heads`, `attn_dtype`, and optionally `generated_text` when `use_generate=true`.

### Evaluate predicted boxes

1. Prepare a JSONL with ground-truth boxes (`bbox` as `[x, y, width, height]`) and include a stable identifier per row (e.g. `id` or `question_id`).
2. Run the batch pipeline to generate predictions:

   ```python
   python pipeline.py stage=batch \
     data.data_file=/path/to/your.jsonl \
     data.output_dir=outputs/results/ \
     storage.attention.compress=true
   ```

3. Evaluate IoU:

   ```python
   python evaluate_boxes.py \
     --dataset-jsonl /path/to/your.jsonl \
     --pred-root outputs/results/liuhaotian-llava-v1.5-7b \
     --id-field question_id
   ```

The script reports mean/median IoU and success rates for thresholds (defaults: 0.25, 0.50, 0.75). Adjust `--iou-thresholds` as needed.

## How Attention Is Collected

- Forward mode (default):
  - Call `model(..., output_attentions=True)`
  - Take the last token attention to visual token range: `[L, H, 1, V]`
- Generate mode (`model.use_generate=true`):
  - Call `generate(..., return_dict_in_generate=True, output_attentions=True)`
  - Use attentions from the first generated step (shape `[L, H, 1, src_len]`) sliced to visual tokens; falls back to forward if not present

## Repository Layout

```
.
├─ pipeline.py                 # Hydra entrypoint
├─ collector.py                # Attention collection (forward/generate)
├─ analyze.py                  # Elbow (value) + spatial entropy head selection
├─ bbox.py                     # Combine heads → mask → bbox
├─ viz.py                      # Image + top-K attention plots
├─ requirements.txt            # Minimal runtime dependencies
├─ conf/
│  ├─ config.yaml              # Hydra root
│  ├─ model/llava15_7b.yaml    # Model + runtime options
│  ├─ logic/selection_v1.yaml  # Head selection + post-processing
│  └─ data/local_examples.yaml # IO + batch options
├─ llava/                      # Minimal LLaVA components (no Transformers fork)
│  ├─ model/ ...               # Builder + vision tower wiring
│  ├─ conversation.py          # Prompt templates
│  ├─ constants.py             # Special tokens, log dir
│  └─ mm_utils.py              # Tokenization + image utils
├─ lab/                        # Lightweight stations (token segmentation metadata)
│  └─ stations.py
└─ examples/                   # Example images + JSONL
```

## Tips & Troubleshooting

- Eager attention: keep `use_flash_attn=false` so attention tensors are returned
- Cache directory: prefer `model.cache_dir`, which is passed to all `from_pretrained(...)` calls; env vars (`TRANSFORMERS_CACHE`, `HF_HOME`, `HF_HUB_CACHE`) are also set when provided
- If `hydra/job_logging: colorlog` not found, ensure `pip install hydra-colorlog`
- If you see memory errors, reduce image size, lower `top_k`, or try CPU for small tests

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
