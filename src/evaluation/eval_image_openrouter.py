import json
import requests
import re
import os
import argparse
from tqdm import tqdm

API_KEY = "YOUR_API_KEY_HERE"
MODEL_NAME = "google/gemini-2.5-pro"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

cf_prompt_en = """You are a causal reasoning expert. 
You can understand image content and logically infer the most plausible causal chains.
You must only output the causal chains strictly in the format A#->B#->C# separated by commas.
Do NOT provide any explanation, reasoning, or other text.
Your answer must ONLY contain the chains.
"""


def load_json_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_existing_results(output_path):
    if not os.path.exists(output_path):
        return {}

    processed = {}
    with open(output_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                processed[item["id"]] = item
    return processed

from PIL import Image
from io import BytesIO
import base64

def image_to_base64(img_path):
    with Image.open(img_path) as img:
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

def generate_reasoning(item):
    img_path = item["img_path"]
    set_A = item["set_A"]
    set_B = item["set_B"]
    set_C = item["set_C"]

    try:
        image_b64 = image_to_base64(img_path)
    except Exception as e:
        print(f"Error reading image for ID {item['id']}: {e}")
        return "ERROR_LOADING_IMAGE"

    user_prompt_text = (
        f"Please strictly infer what is likely to happen next based on the image content.\n"
        f"You need to complete **3 full causal chain combinations**, each formed by selecting one item from each of the following three sets:\n\n"
        f"- Choose one condition from Set A: {set_A}\n"
        f"- Choose one intermediate reasoning step from Set B: {set_B}\n"
        f"- Choose one outcome from Set C: {set_C}\n\n"
        f"Note: You should use hidden information in the image to assist in choosing A->B, then continue the causal chain from A->B to correctly select C. That is, the causal chains you generate should represent the most likely subsequent developments under the image context and Set_A condition.\n"
        f"Requirements:\n"
        f"1. Each causal chain must follow the format A#->B#->C#, e.g., A2->B1->C3.\n"
        f"2. Each A, B, C item can only be used once, no repetition.\n"
        f"3. Output a total of 3 causal chains, covering all A, B, C items.\n"
        f"4. Only output the causal chains, **do not include any explanations, reasoning, or extra text**.\n"
        f"5. The final answer should only contain these causal chains, separated by commas.\n"
        f"6. Strictly follow the format; any content not matching the format will be rejected.\n\n"
        f"Please output the answer directly, for example: A1->B3->C1, A2->B1->C3, A3->B2->C2"
    )

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": cf_prompt_en},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    {"type": "text", "text": user_prompt_text}
                ]
            }
        ],
        "temperature": 0
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()

    except Exception as e:
        print(f"Error generating for ID {item['id']}: {e}")
        return "ERROR"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0, help="Start sample index (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="End sample index (exclusive)")
    args = parser.parse_args()

    input_path = "./data/flick30k_data_final.json"

    safe_model_name = re.sub(r'[^A-Za-z0-9_\-]', '_', MODEL_NAME)
    range_str = f"{args.start}_{args.end if args.end is not None else 'end'}"
    output_path = f"./result/{safe_model_name}_result_{range_str}_final.jsonl"

    data = load_json_data(input_path)

    if args.end is not None:
        data = data[args.start:args.end]
    else:
        data = data[args.start:]

    processed_items = load_existing_results(output_path)
    processed_ids = set(processed_items.keys())

    print(f"✅ Processing sample range this time: [{args.start}, {args.end if args.end is not None else len(data)}), detected {len(processed_ids)} already processed samples, skipping them")
    print(f"✅ Output file: {output_path}")

    with open(output_path, 'a', encoding='utf-8') as f_out:
        for local_idx, item in tqdm(enumerate(data), total=len(data), desc="Processing"):
            global_idx = args.start + local_idx
            item_id = str(global_idx)
            if item_id in processed_ids:
                continue

            item["id"] = item_id
            result = generate_reasoning(item)
            tqdm.write(f"ID: {item_id} → Predicted Answer: {result}")
            item["Predicted_Answer"] = result

            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
            f_out.flush()

    print(f"✅ Processing complete, results saved to: {output_path}")

if __name__ == "__main__":
    main()

