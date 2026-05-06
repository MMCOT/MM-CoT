import os
import re
import json
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

VIDEO_ROOT = "./data/sharegpt4video"
OUTPUT_JSON = (
    "./data/annotated_sharegpt4video_40k.json"
)
FILTERED_OUTPUT_JSON = "./data/filtered_annotated.json"
REJECTED_OUTPUT_JSON = "./data/rejected_annotated.json"


model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-72B-Instruct", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct")
fps = 1.0


def clean_response_text(text):
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def is_valid_output(entry):
    required_keys = [
        "Key Elements",
        "Counterfactual Elements",
        "Counterfactual Outcome Pool",
        "Critical Conditions",
        "Counterfactual Causal Chain",
    ]

    for key in required_keys:
        if (
            key not in entry
            or not isinstance(entry[key], str)
            or not entry[key].strip()
        ):
            print(f"[INVALID] Missing or empty key: {key}")
            return False

    def has_bracketed_items(s):
        return bool(re.findall(r"\[[^\[\]]+\]", s))

    if not has_bracketed_items(entry["Key Elements"]):
        print(f"[INVALID] Key Elements format error")
        return False
    if not has_bracketed_items(entry["Counterfactual Elements"]):
        print(f"[INVALID] Counterfactual Elements format error")
        return False
    if not has_bracketed_items(entry["Counterfactual Outcome Pool"]):
        print(f"[INVALID] Outcome Pool format error")
        return False

    if len(re.findall(r"\[[^\[\]]+\]", entry["Critical Conditions"])) != 2:
        print(f"[INVALID] Critical Conditions must have exactly 2 elements")
        return False

    if not re.match(
        r"^\[[^\[\]]+\], \[[^\[\]]+\] -> \[[^\[\]]+\]$",
        entry["Counterfactual Causal Chain"].strip(),
    ):
        print(f"[INVALID] Counterfactual Causal Chain format error")
        return False

    return True


def make_filter_prompt(entry):
    return f"""Check the following counterfactual annotation for a video:

            "Key Elements": {entry["Key Elements"]}
            "Counterfactual Elements": {entry["Counterfactual Elements"]}
            "Counterfactual Outcome Pool": {entry["Counterfactual Outcome Pool"]}
            "Critical Conditions": {entry["Critical Conditions"]}
            "Counterfactual Causal Chain": {entry["Counterfactual Causal Chain"]}

            Decide if this output is high quality.
            """


AUDIT_SYSTEM_PROMPT = """You are a counterfactual reasoning data auditor.

Your task is to examine a generated counterfactual reasoning output for a video and determine whether it is of **high quality**, based on both the **video content** and the provided annotation.

⚠️ Important: You must carefully analyze the visual content of the video to verify whether the key elements, causal links, and counterfactual reasoning are grounded in what is shown in the video.

The counterfactual annotation is considered high-quality only if it meets **all** of the following criteria:

1. ✅ The "Key Elements" are correctly identified from the **video** and are essential to the outcome.
2. ✅ The "Counterfactual Elements" are plausible substitutions for the Key Elements and could realistically be true **in the given video context**.
3. ✅ The "Counterfactual Outcome Pool" contains reasonable and logically possible consequences **if** the counterfactual elements were true.
4. ✅ The "Critical Conditions" are a reasonable pair of key elements from the video that causally influence the final outcome.
5. ✅ The "Counterfactual Causal Chain" must describe a **plausible causal path** under the assumption that the two counterfactual elements (on the left of the arrow "->") are true. It is acceptable if some parts of the video context change **as a natural consequence** of the counterfactual conditions — as long as the overall reasoning remains grounded in what is seen in the video. The resulting outcome (on the right of the arrow) must logically follow from the counterfactual setup and appear in the Counterfactual Outcome Pool.

You must return a JSON object using this exact format:


```json
{
  "is_high_quality": "YES" or "NO",
  "reason": "Brief explanation why it is acceptable or not (1-2 sentences)."
}
Do not output anything else.
"""


def audit_entry(entry, video_path):
    context = make_filter_prompt(entry)
    conversation = [
        {
            "role": "system",
            "content": AUDIT_SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 360 * 420,
                    "fps": 1.0
                },
                {
                    "type": "text",
                    "text": context
                }
            ]
        }
    ]
    
    text = processor.apply_chat_template(
        conversation,
        tokenize=False, 
        add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(conversation, return_video_kwargs=True)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        # fps=fps,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    ).to(model.device)
    
    generated_ids = model.generate(**inputs, max_new_tokens=2048)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    try:
        audit_result = json.loads(clean_response_text(output_text[0].strip()))
        return audit_result
    except Exception as e:
        print(f"[ERROR] Failed to parse audit result: {output_text[0].strip()}")
        audit_result = {
            "is_high_quality": "NO",
            "reason": "Failed to parse audit result."
        }
        return audit_result


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_existing_json(path):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to read {path}: {e}")
    return []


def main():
    with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    filtered = load_existing_json(FILTERED_OUTPUT_JSON)
    rejected = load_existing_json(REJECTED_OUTPUT_JSON)

    processed_ids = set(entry["id"] for entry in filtered + rejected if "id" in entry)
    print(f"🔄 Resuming: {len(processed_ids)} entries already processed.")

    counter = 0

    for entry in data:
        entry_id = entry.get("id")
        if not entry_id:
            print("[SKIP] Entry without 'id'")
            continue
        if entry_id in processed_ids:
            print(f"[SKIP] ID {entry_id} already processed.")
            continue
        video_path = os.path.join(
            VIDEO_ROOT,
            entry["input_video"]
        )
        if not os.path.exists(video_path):
            print(f"[WARN] Video not found: {video_path}")
            continue
        counter += 1
        if not is_valid_output(entry):
            print(f"[❌] ID {entry_id} - Invalid structure.")
            rejected.append(entry)
        else:
            audit_result = audit_entry(entry, video_path)
            if audit_result.get("is_high_quality", "").strip().upper() == "YES":
                print(f"[✅] ID {entry_id} passed.")
                filtered.append(entry)
            else:
                print(f"[❌] ID {entry_id} failed.")
                entry["reason"] = audit_result.get("reason", "No reason provided.")
                rejected.append(entry)

        if counter % 2 == 0:
            save_json(filtered, FILTERED_OUTPUT_JSON)
            save_json(rejected, REJECTED_OUTPUT_JSON)
            print(f"[SAVE] Total: {len(filtered)} ✅ | {len(rejected)} ❌ | Rate: {len(filtered) / (len(filtered) + len(rejected)):.2%} 👍")

    save_json(filtered, FILTERED_OUTPUT_JSON)
    save_json(rejected, REJECTED_OUTPUT_JSON)
    print(
        f"🎉 Done. {len(filtered)} high-quality samples saved to {FILTERED_OUTPUT_JSON}"
    )
    print(f"📉 {len(rejected)} low-quality samples saved to {REJECTED_OUTPUT_JSON}")


if __name__ == "__main__":
    main()
