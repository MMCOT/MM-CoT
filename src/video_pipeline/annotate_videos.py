import os
import json
from typing import Dict, List
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"  # Adjust to your GPU setup

VIDEO_ROOT = "./data/sharegpt4video"
INPUT_JSON = f"./data/sharegpt4video_40k.jsonl"
OUTPUT_JSON = (
    f"./data/annotated_sharegpt4video_40k.json"
)
OUTPUT_IMAGE_DIR = "./tmp_frames"
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)


model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-72B-Instruct", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct")
fps = 1.0


QWEN_SYSTEM_PROMPT = """You are a counterfactual reasoning expert. 
Your task is to analyze the key causal elements of a video scenario and construct a plausible counterfactual causal chain.

You MUST return a JSON object in the following structure:
{
  "Key Elements": "[key_elementA], [key_elementB], [key_elementC], ...",  # a single string of key elements from the video that are causally essential to the outcome
  "Counterfactual Elements": "[counterfactual_elementA], [counterfactual_elementB], [counterfactual_elementC], ...",  # a single string of counterfactual elements that could replace key elements
  "Counterfactual Outcome Pool": "[counterfactual_outcome1], [counterfactual_outcome2], ...",  # a single string of complete and specific outcome sentences that would occur if the Counterfactual Elements were true, based on the video context
  "Critical Conditions": "[key_elementA], [key_elementC]",  # select two **logically or causally linked** key elements from "Key Elements" that jointly contribute to the final outcome in the video. These elements must form a coherent causal pair whose counterfactual versions would lead to a plausible change in the final outcome.
  "Counterfactual Causal Chain": "[counterfactual_elementA], [counterfactual_elementC] -> [counterfactual_outcome2]"  # use counterfactual counterparts of the two Critical Conditions to construct the counterfactual causal chain. The final outcome MUST be exactly one item from the Counterfactual Outcome Pool.
}
⚠️ Important rules:
- The output must be **valid JSON** (not a string).
- Use **double quotes** (") for all string values.
- For all fields, the value must be a **single string** (not a JSON list).
- Inside the string, each item must be enclosed in square brackets, like "[item1], [item2]".
- Do **not** include any natural language explanation before or after the JSON.
"""


def load_json(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_json(data: List[Dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_existing_results(path: str) -> Dict[str, Dict]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        try:
            items = json.load(f)
            return {item["id"]: item for item in items if "id" in item}
        except json.JSONDecodeError:
            return {}


def clean_response_text(response_text: str) -> str:
    response_text = response_text.strip()
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    elif response_text.startswith("```"):
        response_text = response_text[3:]
    if response_text.endswith("```"):
        response_text = response_text[:-3]
    return response_text.strip()


def generate_cf_annotation(entry: Dict, video_path: str) -> Dict:
    template = {
        "Key Elements": "",
        "Counterfactual Elements": "",
        "Counterfactual Outcome Pool": "",
        "Critical Conditions": "",
        "Counterfactual Causal Chain": "",
    }

    example = {
        "Key Elements": "[dog], [high speed], [slippery rock], [big waves], [sunny day], [ocean], [ship]",
        "Counterfactual Elements": "[cat], [slow speed], [dry rock], [small waves], [rainy day], [grassland], [wooden boat]",
        "Counterfactual Outcome Pool": "[the puppy didn’t fall into the sea], [the puppy stared at the sea], [suddenly it started raining], [just as the puppy reached the edge, thunder struck, and it turned around and ran], [the puppy stopped and howled at the distant ship]",
        "Critical Conditions": "[high speed], [slippery rock]",
        "Counterfactual Causal Chain": "[slow speed], [dry rock] -> [the puppy didn’t fall into the sea]",
    }

    context = (
        f"You are a video counterfactual expert. Based on the video and its descriptions, "
        f"you need to extract the following fields and fill in the JSON template. "
        f"Do NOT output any text except the completed JSON.\n\n"
        f"Example:\n{json.dumps(example, indent=2)}\n\n"
        f"JSON Template:\n{json.dumps(template, indent=2)}\n\n"
        f"Video description:\n{entry.get('description','N/A')}\n\n"
        f"Now fill in the template with actual values.\n\n"
    )

    conversation = [
        {"role": "system", "content": QWEN_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 360 * 420,
                    "fps": 1.0,
                },
                {"type": "text", "text": context},
            ],
        },
    ]

    text = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        conversation, return_video_kwargs=True
    )
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        # fps=fps,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    ).to(model.device)

    # Inference
    print(f"[INFO] Now processing video: {video_path}")
    generated_ids = model.generate(**inputs, max_new_tokens=2048)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    try:
        return json.loads(clean_response_text(output_text[0].strip()))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format from model: {output_text[0]}") from e


def main():
    data = load_json(INPUT_JSON)
    existing_results = load_existing_results(OUTPUT_JSON)
    results = list(existing_results.values())
    processed_ids = set(existing_results.keys())

    for entry in data:
        entry["id"] = entry.get("video_id", "unknown")
        if entry["id"] in processed_ids:
            print(f"[SKIP] ID {entry['id']} already processed.")
            continue

        entry["input_video"] = entry.get("video_path", "")
        captions = entry.get("captions", [])
        if (
            isinstance(captions, list)
            and len(captions) > 0
            and isinstance(captions[-1], dict)
        ):
            entry["description"] = captions[-1].get("content", "")
        else:
            entry["description"] = ""

        video_path = os.path.join(VIDEO_ROOT, entry["input_video"])
        if not os.path.exists(video_path):
            print(f"[WARN] Video not found: {video_path}")
            continue

        try:
            cf = generate_cf_annotation(entry, video_path)
            cf_full = {
                "id": entry["id"],
                "input_video": entry["input_video"],
                "Key Elements": cf["Key Elements"],
                "Counterfactual Elements": cf["Counterfactual Elements"],
                "Counterfactual Outcome Pool": cf["Counterfactual Outcome Pool"],
                "Critical Conditions": cf["Critical Conditions"],
                "Counterfactual Causal Chain": cf["Counterfactual Causal Chain"],
            }
            results.append(cf_full)
            print(f"✅ ID {entry['id']} processed.")

            if len(results) % 2 == 0:
                save_json(results, OUTPUT_JSON)
                print(f"💾 Progress saved: {len(results)} samples written.")

        except Exception as e:
            print(f"\033[91m[ERROR] ID {entry['id']} failed: {e}\033[0m")
            continue

    save_json(results, OUTPUT_JSON)
    print(f"✅ All done. Results saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
