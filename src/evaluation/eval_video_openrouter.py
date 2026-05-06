import os
import json
import base64
from typing import Dict, List
import cv2
import re
from PIL import Image
from openai import OpenAI
from calculate_acc import compute_accuracy

MODEL_NAME = "anthropic/claude-sonnet-4"
API_KEY = "YOUR_API_KEY_HERE"
input_path = "video_data_2100.json"
output_path = input_path.replace(".json", f"_{MODEL_NAME.split('/')[-1]}_result.json")
OUTPUT_IMAGE_DIR = "./video_test_tmp_frames"
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
video_path = "./data/sharegpt4video"

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY)


cf_prompt_en = """"You are a counterfactual reasoning expert. 
Your task is to analyze the key causal elements of a video scenario and construct a plausible counterfactual causal chain.

You MUST return a JSON object in the following structure:
{
  "chosen_conditions": "[key_elementA], [key_elementB]",                  # select two **logically or causally linked** key elements from "Key Elements" that jointly contribute to the final outcome in the video. These elements must form a coherent causal pair whose counterfactual versions would lead to a plausible change in the final outcome.
  "counterfactual_conditions": "[counterfactual_elementA], [counterfactual_elementB]",          # select matching replacements from Counterfactual Elements
  "counterfactual_result": "[counterfactual_outcome1]",              # select a plausible outcome from Counterfactual Outcome Pool that would occur if both counterfactual_conditions were true, based on the video context
  "counterfactual_chain": "[counterfactual_elementA], [counterfactual_elementB] -> [counterfactual_outcome1]"    # format the above into a concise causal chain
}
Only include the JSON. Do not explain your reasoning in natural language.
"""


def load_json_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_user_prompt(item):
    template = {
        "chosen_conditions": "",
        "counterfactual_conditions": "",
        "counterfactual_result": "",
        "counterfactual_chain": "",
    }

    context = (
        f"You are a video counterfactual expert. Based on the video and its descriptions, "
        f"you need to extract the following fields and fill in the JSON template. "
        f"Do NOT output any text except the completed JSON.\n\n"
        f"JSON Template:\n{json.dumps(template, indent=2)}\n\n"
        f"Video information: \n"
        f"- 1. Here are the Key Elements of the video:\n{item['Key Elements']}\n\n"
        f"- 2. Below are the Counterfactual Elements (do not repeat key elements):\n{item['Counterfactual Elements']}\n\n"
        f"- 3. Here is a set of possible counterfactual outcomes:\n{item['Counterfactual Outcome Pool']}\n\n"
        f"Your task:\n"
        f"- First, identify the key causal conditions from Key Elements.\n\n"
        f"- Then, choose their counterfactual replacements from Counterfactual Elements.\n\n"
        f"- Finally, infer a reasonable outcome from Counterfactual Outcome Pool, and organize them into a causal chain.\n\n"
        f"Now fill in the template with actual values.\n"
    )
    return context


def load_existing_results(path: str) -> Dict[str, Dict]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        try:
            items = json.load(f)
            return {item["id"]: item for item in items if "id" in item}
        except json.JSONDecodeError:
            return {}


def extract_keyframes(video_path: str, num_frames: int = 5):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    image_paths = []

    if total_frames <= 0:
        print(f"[WARN] Unable to read total frame count for {video_path}")
        cap.release()
        return []

    selected_indices = [
        int(total_frames * (i + 1) / (num_frames + 1)) for i in range(num_frames)
    ]

    for idx in selected_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        img_path = os.path.join(
            OUTPUT_IMAGE_DIR, f"{os.path.basename(video_path)}_frame_{idx}.jpg"
        )
        pil_img.save(img_path)
        image_paths.append(img_path)

    cap.release()
    return image_paths


def image_to_base64(path, max_width=640, max_height=640, quality=50) -> str:
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            width, height = img.size

            if width > max_width or height > max_height:
                scale = min(max_width / width, max_height / height)
                new_size = (int(width * scale), int(height * scale))
                img = img.resize(new_size)

            from io import BytesIO

            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=quality)
            b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return b64
    except Exception as e:
        print(f"[ERROR] Image processing failed: {path}, error message: {e}")
        return None


def clean_response_text(response_text: str) -> str:
    response_text = response_text.strip()
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    elif response_text.startswith("```"):
        response_text = response_text[3:]
    if response_text.endswith("```"):
        response_text = response_text[:-3]

    match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if match:
        return match.group(0).strip()
    else:
        return response_text.strip()


def make_multimodal_messages(
    system_prompt: str, text: str, image_paths: List[str]
) -> List[Dict]:
    messages = [{"role": "system", "content": system_prompt}]
    user_content = []

    for path in image_paths:
        b64 = image_to_base64(path)

        user_content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        )

    user_content.append({"type": "text", "text": text})
    messages.append({"role": "user", "content": user_content})
    return messages


def generate_cf_reasoning(item, image_paths):
    user_prompt = build_user_prompt(item)
    messages = make_multimodal_messages(cf_prompt_en, user_prompt, image_paths)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        extra_headers={},
        extra_body={},
    )
    return clean_response_text(response.choices[0].message.content.strip())


def main():
    data = load_json_data(input_path)
    existing_results = load_existing_results(output_path)
    results = list(existing_results.values())
    processed_ids = set(existing_results.keys())

    for item in data:
        try:
            if item["id"] in processed_ids:
                print(f"[SKIP] ID {item['id']} already processed.")
                continue

            video_path = os.path.join(video_path, item["input_video"])
            image_paths = extract_keyframes(video_path)
            result = generate_cf_reasoning(item, image_paths)
            print(f"ID: {item['id']} → Model Output: {result}")
            item["Predicted_CF_Chain"] = result

            results.append(item)
            print(f"✅ ID {item['id']} processed.")

            if len(results) % 2 == 0:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"💾 Progress saved: {len(results)} samples written.")

            if len(results) % 10 == 0:
                compute_accuracy(results)
        except Exception as e:
            print(f"\033[91m[ERROR] ID {item['id']} failed: {e}\033[0m")
            continue

    compute_accuracy(results)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
