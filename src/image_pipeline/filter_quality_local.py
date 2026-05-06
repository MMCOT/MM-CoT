import os
import json
import glob
from tqdm import tqdm
import time
import argparse
import re

from modelscope import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

class AutomatedDataEvaluatorLocal:
    def __init__(self):
        print("🤖 Initializing the local Qwen2.5-VL model for evaluation...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-72B-Instruct",
            torch_dtype="auto",
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct")
        print("✅ Local model initialized successfully.")

    def _get_evaluation_prompt_template(self) -> str:
        # Prompt template remains unchanged
        return """
You are a highly experienced and critically minded data quality analyst. Your task is to rigorously and precisely evaluate a data sample generated for a visual causal reasoning task.
Your core principle is: **better none than subpar**. Only data that excels in logic, visual grounding, and creativity, with almost no flaws, can receive a high score. You are extremely strict and cannot tolerate even minor logical inconsistencies or vague descriptions.

---
### **Data to Evaluate:**

* **Image**: [Provided]
* **Set_A (Trigger Conditions)**:
{set_A}
* **Set_B & Set_C (Correct Causal Chain)**:
{correct_B}
{correct_C}
* **Set_B & Set_C (Distractors)**:
{distractor_B}
{distractor_C}
* **Right_answer (Correct Matching Answer)**: {right_answer}

---
### **Evaluation Checklist:**
#### **Part 1: Evaluate the Quality of `Set_A` (Trigger Conditions)**
1. **Visual Grounding**: Do the conditions clearly reference elements **actually visible** in the image?
2. **Counterfactuality**: Are the described events **not currently happening**?
3. **Causal Significance**: Do the conditions have the potential to **change the scene dynamics**?
4. **Category Diversity**: Do A1, A2, A3 respectively satisfy the categories of **environment, human, external**?
5. **Conciseness & Neutrality**: Are they concise one-sentence descriptions and **not describing outcomes**?

#### **Part 2: Evaluate the Quality of the Correct Causal Chain (`A→B→C`)**
1. **Logical Coherence**: Is the causal chain provided in `Right_answer` **logically flawless**?
2. **Visual Consistency**: Are the descriptions of B and C **fully consistent** with the original image?
3. **Inference Validity**: Is B truly a **reasoning step**, rather than a paraphrased outcome?

#### **Part 3: Evaluate the Quality of the Distractors (`B4-B6` and `C4-C6`)**
1. **Visual Inconsistency**: Do the distractors clearly **contradict the visual facts** in the image?
2. **Logical Plausibility**: Would the distractor causal chains **sound reasonable** if the image were not seen?
3. **Uniqueness**: Are the distractors **essentially distinct** from the correct chain, not just simple negations or rewordings?

---
### **Final Evaluation Conclusion and Scoring Rubric:**

Before scoring, check for any of the following **“disqualifying conditions”**. If any apply, the score **must not exceed 2.5**:
- **Major logical error**: The correct causal chain (A→B→C) contains an obvious logical contradiction.
- **Major visual error**: The description of B or C in the correct chain **directly conflicts** with the image.
- **Set_A not grounded**: At least one trigger condition is purely imagined and not based on image content.

Assign a score with one decimal place based on the following rubric:

- **5.0 (Perfect)**:
  - **All** checklist criteria are met **with high quality**.
  - All three Set_A conditions are insightful and precisely grounded.
  - The causal chain is airtight with no ambiguity.
  - **All** distractors are clever, misleading, and clearly contradict the image.

- **4.0–4.5 (Excellent)**:
  - **All major criteria** are satisfied with no logical or visual flaws.
  - May contain very minor imperfections, such as:
    - A distractor with slightly less creativity.
    - A description whose phrasing could be smoother.
  - Overall data quality is very high and ready for use.

- **3.0–3.5 (Passable)**:
  - **Core logic is correct**, meaning the causal chain (A→B→C) is sound and matches the image.
  - However, some **non-core criteria** have issues, such as:
    - Set_A lacks category diversity (e.g., A1 and A3 both describe human actions).
    - Distractors are mediocre or insufficiently distinct.
    - A trigger condition is too broad or not specific enough.

- **2.0–2.5 (Poor)**:
  - At least **one disqualifying condition** is present.
  - Or multiple non-core issues significantly reduce quality.
  - Some parts may still be usable (e.g., one Set_A condition).

- **1.0–1.5 (Severely Defective)**:
  - **Multiple** disqualifying conditions occur.
  - The data is logically incoherent or completely detached from the image.
  - The data is entirely unusable.

Now, synthesize all analyses above, paying special attention to the scoring rubric, and provide a score **with exactly one decimal place**. Your response **must and can only be** a strictly valid JSON object.

**Please output strictly in the following format (e.g., 4.5):**
`{{"score": <a decimal between 1.0 and 5.0>}}`
"""

    def _prepare_evaluation_prompt(self, data_item: dict) -> str:
        try:
            def split_sets_safely(combined_str: str):
                if not isinstance(combined_str, str) or not combined_str.strip(): return "", ""
                lines = [line.strip() for line in combined_str.strip().split('\n')]
                correct_part, distractor_part = "\n".join(lines[:3]), "\n".join(lines[3:])
                return correct_part, distractor_part
            set_b_str, set_c_str = data_item.get('set_B', ''), data_item.get('set_C', '')
            set_a_str, right_answer_str = data_item.get('set_A', 'N/A'), data_item.get('Right_answer', 'N/A')
            correct_b, distractor_b = split_sets_safely(set_b_str)
            correct_c, distractor_c = split_sets_safely(set_c_str)
            template = self._get_evaluation_prompt_template()
            return template.format(set_A=set_a_str, correct_B=correct_b, correct_C=correct_c, distractor_B=distractor_b, distractor_C=distractor_c, right_answer=right_answer_str)
        except Exception as e:
            tqdm.write(f"CRITICAL ERROR in _prepare_evaluation_prompt: {e}")
            tqdm.write(f"Problematic data_item: {data_item}")
            raise ValueError("Failed to prepare prompt due to malformed data.")

    def get_evaluation_score(self, image_path: str, data_item: dict) -> float:
        prompt = self._prepare_evaluation_prompt(data_item)
        
        conversation = [{
            "role": "system",
            "content": "You are an expert data quality analyst. Respond only with a valid JSON object like {\"score\": 4.5}."
        }, {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt}
            ]
        }]

        text = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(conversation, return_video_kwargs=False)
        inputs = self.processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(**inputs, max_new_tokens=50, temperature=0.1)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, outputs)]
        raw_content = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
        
        tqdm.write(f"DEBUG: Model returned raw content: '{raw_content}'")

        try:
            score_data = json.loads(raw_content)
            if 'score' in score_data and isinstance(score_data['score'], (int, float)):
                return max(1.0, min(5.0, float(score_data['score'])))
        except (json.JSONDecodeError, TypeError):
            tqdm.write(f"INFO: Could not parse as JSON, attempting regex fallback.")
            pass
        try:
            match = re.search(r'\d+\.?\d*', raw_content)
            if match:
                score = float(match.group(0))
                tqdm.write(f"INFO: Successfully parsed score '{score}' using regex fallback.")
                return max(1.0, min(5.0, score))
        except (ValueError, TypeError): pass
        tqdm.write(f"ERROR: Could not parse a valid score from API output. Defaulting to 0.0.")
        return 0.0

    def process_and_filter_file(self, input_json_path: str, output_json_path: str, quality_threshold: float):
        if not os.path.exists(input_json_path):
            print(f"❌ Error: Input file not found at {input_json_path}")
            return
            
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data_to_evaluate = json.load(f)
            
        print(f"🔍 Found {len(data_to_evaluate)} data points to evaluate in {input_json_path}.")
        print(f"📈 Quality threshold set to: score >= {quality_threshold}")

        temp_output_path = output_json_path + ".tmp"
        kept_count = 0

        with open(temp_output_path, "w", encoding="utf-8") as f:
            f.write("[\n")
            first = True
            for item in tqdm(data_to_evaluate, desc="Evaluating and Filtering"):
                retry_count = 0
                while retry_count < 3:
                    try:
                        img_path = item['img_path']
                        if not os.path.exists(img_path):
                            tqdm.write(f"⚠️ Image not found, skipping: {img_path}")
                            break
                        
                        score = self.get_evaluation_score(img_path, item)

                        if score >= quality_threshold:
                            if not first: f.write(",\n")
                            else: first = False
                            json.dump(item, f, ensure_ascii=False, indent=2)
                            f.flush()
                            tqdm.write(f"✅ Kept: {os.path.basename(img_path)} (Score: {score})")
                            kept_count += 1
                        else:
                            tqdm.write(f"❌ Discarded: {os.path.basename(img_path)} (Score: {score} < {quality_threshold})")
                        break
                    except Exception as e:
                        tqdm.write(f"⚠️ Error occurred during evaluation {os.path.basename(item.get('img_path', 'N/A'))}: {e}")
                        retry_count += 1
                        tqdm.write(f"Retrying ({retry_count}/3)...")
                        time.sleep(2)
                
                if retry_count == 3:
                    tqdm.write(f"🚫 Skipped after 3 failed attempts: {os.path.basename(item.get('img_path', 'N/A'))}.")

            f.write("\n]")
        
        os.rename(temp_output_path, output_json_path)
        print(f"\n🎉 Filtering complete!\nTotal kept: {kept_count} / {len(data_to_evaluate)}\nHigh-quality dataset saved to: {output_json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automatically evaluate and filter visual causal chains using the local Qwen2.5-VL model.")
    parser.add_argument("--input-file", type=str, default="./data/input.json", required=True, help="Path to the generated JSON file to be evaluated.")
    parser.add_argument("--output-file", type=str, default="./data/output_filtered.json", required=True, help="Path to save the filtered high-quality new JSON file.")
    parser.add_argument("--quality-threshold", type=float, default=4.0, help="Minimum score for keeping data (1.0-5.0). Default is 4.0.")
    args = parser.parse_args()

    evaluator = AutomatedDataEvaluatorLocal()
    
    evaluator.process_and_filter_file(
        args.input_file, 
        args.output_file,
        args.quality_threshold
    )
