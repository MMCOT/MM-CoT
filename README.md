# MM-CoT: A Benchmark for Probing Visual Chain-of-Thought Reasoning in Multimodal Models

---

## Overview

MM-CoT is a diagnostic benchmark that evaluates multimodal Chain-of-Thought (CoT) reasoning as a **discriminative verification** task. Instead of generating free-form explanations, models must select the sole event chain that satisfies two orthogonal constraints:

1. **Visual Consistency** — every step is anchored in observable evidence.
2. **Logical Coherence** — causal and temporal transitions are valid.

Adversarial distractors are engineered to violate exactly one of these constraints, enabling fine-grained diagnosis of reasoning failures.

### Benchmark at a Glance

| Modality | Source          | Instances | Distractors per item |
|----------|-----------------|-----------|----------------------|
| Image    | Flickr30k       | 5,615     | K = 3                |
| Video    | ShareGPT4Video  | 2,100     | K = 4                |

---

## Repository Structure

```
MM-CoT/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── data/
│   ├── image/
│   │   └── img_data_5615.json         # Image benchmark data (5,615 instances)
│   └── video/
│       └── video_data_2100.json       # Video benchmark data (2,100 instances)
├── src/
│   ├── image_pipeline/                # Image data construction pipeline
│   │   ├── generate_chains_api.py     # Step 1: Generate A→B→C chains (Together API)
│   │   ├── generate_chains_local.py   # Step 1: Generate chains (local Qwen model)
│   │   ├── generate_chains_openrouter.py  # Step 1: Generate chains (OpenRouter API)
│   │   ├── filter_quality_api.py      # Step 2: Quality filtering (Together API)
│   │   ├── filter_quality_local.py    # Step 2: Quality filtering (local model)
│   │   ├── correct_answer_order.py    # Step 3: Correct answer ordering
│   │   └── shuffle_options.py         # Step 4: Shuffle Set_B and Set_C options
│   ├── video_pipeline/                # Video data construction pipeline
│   │   ├── annotate_videos.py         # Step 1: Counterfactual annotation
│   │   └── filter_annotated_videos.py # Step 2: Quality filtering and audit
│   └── evaluation/                    # Model evaluation scripts
│       ├── eval_image_openrouter.py   # Evaluate models on image benchmark
│       ├── eval_video_openrouter.py   # Evaluate models on video benchmark (OpenRouter)
│       ├── eval_video_together.py     # Evaluate models on video benchmark (Together)
│       ├── compute_image_accuracy.py  # Compute image-level accuracy metrics
│       └── compute_video_accuracy.py  # Compute video-level accuracy metrics
```

---

## Getting Started

### 1. Environment Setup

```bash
pip install -r requirements.txt
```

### 2. Data Preparation

**Image data:** Download [Flickr30k](http://shannon.cs.illinois.edu/DenotationGraph/) images and place them under `./data/flickr30k/`.

**Video data:** Download [ShareGPT4Video](https://huggingface.co/datasets/ShareGPT4Video/ShareGPT4Video) and place videos under `./data/sharegpt4video/`.

The benchmark annotations are already provided in `data/image/img_data_5615.json` and `data/video/video_data_2100.json`.

### 3. Evaluate a Model

To evaluate a model on the **image** benchmark via OpenRouter:

```bash
python src/evaluation/eval_image_openrouter.py \
    --start 0 --end 100
```

To compute accuracy after evaluation:

```bash
python src/evaluation/compute_image_accuracy.py
```

For **video** evaluation:

```bash
python src/evaluation/eval_video_openrouter.py
```

> **Note:** Set your API key in the script or via environment variable before running. The placeholder `YOUR_API_KEY_HERE` must be replaced.

---

## Data Construction Pipeline

To reproduce the benchmark data from scratch, follow these steps.

### Image Pipeline

| Step | Script                              | Description                                           |
|------|-------------------------------------|-------------------------------------------------------|
| 1    | `generate_chains_*.py`              | Generate triadic chains (A→B→C) with distractors      |
| 2    | `filter_quality_*.py`               | Score and filter chains (threshold: 4.0/5.0)          |
| 3    | `correct_answer_order.py`           | Normalize answer key ordering (A1, A2, A3)            |
| 4    | `shuffle_options.py`                | Randomize option order in Set_B and Set_C             |

```bash
# Example: Generate chains using Together API
python src/image_pipeline/generate_chains_api.py

# Filter by quality
python src/image_pipeline/filter_quality_api.py \
    --input-file ./data/raw_chains.json \
    --output-file ./data/filtered_chains.json \
    --api-keys-file ./api_keys.txt \
    --quality-threshold 4.0

# Correct and shuffle
python src/image_pipeline/correct_answer_order.py
python src/image_pipeline/shuffle_options.py
```

### Video Pipeline

| Step | Script                            | Description                                         |
|------|-----------------------------------|-----------------------------------------------------|
| 1    | `annotate_videos.py`              | Generate counterfactual annotations (local Qwen)    |
| 2    | `filter_annotated_videos.py`      | Audit and filter annotations (local Qwen)           |

```bash
# Annotate videos (requires local Qwen2.5-VL-72B deployment)
python src/video_pipeline/annotate_videos.py

# Filter annotations
python src/video_pipeline/filter_annotated_videos.py
```

---

## Data Format

### Image Data (`img_data_5615.json`)

Each entry contains:

```json
{
  "set_A": "A1: <Physical event>\nA2: <Human action>\nA3: <External interaction>",
  "set_B": "B1: <Mediator 1>\nB2: ...\nB3: ...\nB4: <Distractor>\nB5: ...\nB6: ...",
  "set_C": "C1: <Outcome 1>\nC2: ...\nC3: ...\nC4: <Distractor>\nC5: ...\nC6: ...",
  "Right_answer": "A1->B2->C3, A2->B1->C2, A3->B3->C1",
  "img_path": "./data/flickr30k/<image_id>.jpg"
}
```

- `set_A`: Three initiating conditions (Physical / Behavioral / External).
- `set_B`: Six mediating events (3 valid + 3 distractors).
- `set_C`: Six outcomes (3 valid + 3 distractors).
- `Right_answer`: Ground-truth chain assignments.

### Video Data (`video_data_2100.json`)

Each entry contains:

```json
{
  "id": "<video_hash>",
  "input_video": "<relative_path_to_video>.mp4",
  "Key Elements": "[elem1], [elem2], ...",
  "Counterfactual Elements": "[cf_elem1], [cf_elem2], ...",
  "Counterfactual Outcome Pool": "[outcome1], [outcome2], ...",
  "Critical Conditions": "[elem_a], [elem_b]",
  "Counterfactual Causal Chain": "[cf_a], [cf_b] -> [outcome_x]"
}
```

---

## Supported Models

The evaluation scripts support any model accessible via OpenRouter or Together AI, including:

| Type        | Models                                                                |
|-------------|-----------------------------------------------------------------------|
| Proprietary | GPT-5, Gemini-2.5-Pro, Claude-Sonnet-4, Grok-2-Vision-1212          |
| Open-source | Qwen2.5-VL-72B, LLaMA-3.2-90B, GLM-4.5V, InternVL3-8B, Ovis-2.5   |

To evaluate a different model, change the `MODEL_NAME` variable in the evaluation script.

---

## Evaluation Metrics

**Primary metric:** Chain selection accuracy — an instance is correct only if the model selects the unique valid chain.

**Diagnostic dimensions:**
- *Visual grounding verification* — robustness against visually inconsistent distractors.
- *Logical coherence verification* — rejection of causally flawed but visually compatible distractors.

---

## Citation

This work is under anonymous review. Citation information will be provided upon acceptance.

---

## License

This benchmark is released for **research purposes only**. The image data originates from [Flickr30k](http://shannon.cs.illinois.edu/DenotationGraph/) and the video data from [ShareGPT4Video](https://huggingface.co/datasets/ShareGPT4Video/ShareGPT4Video). Please refer to their respective licenses for usage terms.
