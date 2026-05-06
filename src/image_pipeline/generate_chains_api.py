import os
import json
import glob
import base64
from pydantic import BaseModel, Field
from tqdm import tqdm
from together import Together


def is_credit_limit_error(e):
    return "credit limit" in str(e).lower() or "Error code: 402" in str(e)


def load_api_keys(apikey_file="./apikeys.txt"):
    if not os.path.exists(apikey_file):
        raise FileNotFoundError(f"API Key file {apikey_file} not found!")
    with open(apikey_file, "r") as f:
        keys = [line.strip() for line in f if line.strip()]
    if not keys:
        raise ValueError("No API keys found in apikey.txt")
    return keys


class CausalChains(BaseModel):
    set_A: str = Field(description="Trigger conditions")
    set_B: str = Field(description="Reactions")
    set_C: str = Field(description="Outcomes")
    Right_answer: str = Field(description="Causal links")
    img_path: str = Field(description="Image path")


class UnifiedCausalChainGenerator:
    def __init__(self, api_key: str):
        self.client = Together(api_key=api_key)
        self.model_set_A = "Qwen/Qwen2.5-VL-72B-Instruct"  # Model for Set_A generation
        self.model_set_BC = (
            "Qwen/Qwen2.5-VL-72B-Instruct"  # Model for Set_B and Set_C generation
        )
        self.model_distractors = (
            "Qwen/Qwen2.5-VL-72B-Instruct"  # Model for generating distractors
        )

    # Define the prompt for generating Set_A
    def get_prompt_set_A(self) -> str:
        return """
    "task_description": "
\"You are a highly creative visual analyst. Your task is to meticulously examine the image and invent three **unique and non-obvious** potential events (Set_A).

## Core Requirements for ALL Conditions:
1.  **Visually Grounded**: Each event MUST originate from a specific, tangible object or element visible in the image.
2.  **Trigger-Only**: Describe only the event itself, NOT the result.
3.  **Maximum Diversity**: The three events (A1, A2, A3) must be completely different in nature from each other.

## **BANNED CONCEPTS** for Condition A1:
- **DO NOT use generic, universal environmental changes.**
- **BANNED**: wind, sunlight, light, shadows, rain, temperature changes.
- **Your task is to find something MORE SPECIFIC AND UNIQUE to this particular image.**

## Step-by-Step Instructions for A1:
1.  **Identify a non-human, non-animal element** in the background or foreground (e.g., a piece of furniture, a tool, a decorative item, a structural element).
2.  **Imagine a plausible physical change or interaction** involving ONLY that element (e.g., it starts to creak, tip over, leak, or a hidden part of it becomes visible).
3.  Describe this specific event as A1.

## Category-Specific Requirements:
- **A1 (Specific Physical Event)**: Follow the steps above. Must be unique to the scene.
- **A2 (Human Intention)**: A person in the image decides to perform a specific, non-trivial action.
- **A3 (External Interference)**: An unexpected and tangible object or entity (not a sound or light) enters the scene and directly interacts with an element.
\"",
    "generate_format": {
        "set_A": "A1: <A highly specific physical event, NOT weather/light related>\\nA2: <A specific human action>\\nA3: <A tangible external interference>"
    }
    """

    # Define the prompt for generating correct Set_B and Set_C based on Set_A
    def get_prompt_set_BC(self, set_A: str) -> str:
        return (
            """
    "task_description": ""You are a visual reasoning expert. Given an image and a set of three hypothetical conditions (Set_A: A1, A2, A3), your task is to generate corresponding causal reasoning steps (Set_B) and outcomes (Set_C) based on each condition.\n\n"
"For each condition in Set_A:\n"
"- Generate a corresponding B (reasoning step) that explains how the visual context and the trigger condition jointly influence the situation.Do not introduce wind as an element!!!\n"
"- Generate a corresponding C (outcome) that logically follows from the reasoning in B, grounded in what’s plausible within the scene.Do not introduce wind as an element!!!\n\n"
"Requirements:\n"
"1. Each B must integrate specific visual evidence from the image and details from the condition in A. It should be a reasoning step — not an outcome.\n"
"2. Each C must describe the plausible result that directly follows from B in the visual context.\n"
"3. The causal chain A → B → C must be logically coherent, grounded in the image, and unique.\n"
"4. Do NOT invent fantastical or physically implausible events unless the image explicitly suggests such a theme.\n\n"",
    The response must return strictly valid and accurate JSON format.
    "generate_format": {
        "set_A": "%s",  # Insert set_A dynamically here
        "set_B": "B1: <Reasoning Step 1 based on Set_A and the image>\nB2: <Reasoning Step 2 based on Set_A and the image>\nB3: <Reasoning Step 3 based on Set_A and the image>",
        "set_C": "C1: <Outcome 1 based on B1>\nC2: <Outcome 2 based on B2>\nC3: <Outcome 3 based on B3>",
        "Right_answer": "A1->B2->C3, A2->B1->C2, A3->B3->C1"
    }
    """
            % set_A
        )  # Insert the Set_A variable here.

    # Define the prompt for generating distractor Set_B and Set_C
    def get_prompt_distractors(self, set_A: str, set_B: str, set_C: str) -> str:
        return """
    "task_description": "You will now generate distractor options for Set_B and Set_C. These are plausible but incorrect reasoning steps and outcomes.

You will receive:
- The original image,
- Set_A (hypothetical conditions),
- Set_B and Set_C (correct reasoning and outcomes).

Your task is to generate three **distractor** reasoning steps (B4–B6) and corresponding **distractor** outcomes (C4–C6) that meet the following criteria:

---

🧠 **Additional Step Before You Begin**:
First, **imagine a different environment** than what is shown in the image. This environment should still make Set_A plausible, but **must differ clearly from the actual image** (e.g., different location, objects, lighting, activities, number of people, or mood).

Then, based on **your imagined scene** and the given Set_A, construct the distractors.

---

### 📌 Requirements:

1. **Visual Inconsistency**: The distractors (B4–B6 and C4–C6) must not match the actual content of the image — they should reference objects or actions **not** present or plausible in the given image.
2. **Linguistic Plausibility**: Each distractor must still logically follow from Set_A when considered **in general** (e.g., a gust of wind might still cause something to fall, even if that thing isn't in the image).
3. **Contrast with Set_B/C**: Your distractors must not repeat or paraphrase any of the correct Set_B or Set_C items.
4. **Grounded Causality**: Even though incorrect visually, the distractors must follow a causally reasonable pattern — like distraction, accident, chain reaction, etc.
5. **Creativity Encouraged**: Use imagination to explore alternate but plausible consequences in your imagined setting.
6. **Do not introduce wind as an element!!!**
---",

    "generate_format": {
        "set_A": "%s",
        "set_B_correct": "%s",
        "set_C_correct": "%s",
        "set_B": "B4: <Distractor Reasoning Step 1>\\nB5: <Distractor Reasoning Step 2>\\nB6: <Distractor Reasoning Step 3>",
        "set_C": "C4: <Distractor Outcome 1>\\nC5: <Distractor Outcome 2>\\nC6: <Distractor Outcome 3>"
    }
    """ % (
            set_A.replace('"', '\\"'),
            set_B.replace('"', '\\"'),
            set_C.replace('"', '\\"'),
        )

    def generate_set_A(self, image_path: str) -> str:
        # Load image and encode to base64
        with open(image_path, "rb") as f:
            base64_img = base64.b64encode(f.read()).decode("utf-8")

        response = self.client.chat.completions.create(
            model=self.model_set_A,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in generating plausible but visually incorrect causal distractors. Respond only in JSON.Please only return a strictly valid JSON object with all fields completed exactly as defined.The JSON must be:syntactically correct (no missing commas, colons, or quotes),fully parseable by json.loads(),contain only the required JSON object, with no extra commentary, explanation, or markdown formatting.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.get_prompt_set_A()},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_img}"
                            },
                        },
                    ],
                },
            ],
            response_format={
                "type": "json_schema",
                "schema": CausalChains.model_json_schema(),
            },
            temperature=0.3,
            top_p=0.7,
            max_tokens=1024,
        )

        raw = json.loads(response.choices[0].message.content)
        print(raw["set_A"])  # Printing Set_A to check the result
        return raw["set_A"]

    def generate_set_BC(self, image_path: str, set_A: str) -> dict:
        # Load image and encode to base64
        with open(image_path, "rb") as f:
            base64_img = base64.b64encode(f.read()).decode("utf-8")

        response = self.client.chat.completions.create(
            model=self.model_set_BC,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in generating plausible but visually incorrect causal distractors. Respond only in JSON.Please only return a strictly valid JSON object with all fields completed exactly as defined.The JSON must be:syntactically correct (no missing commas, colons, or quotes),fully parseable by json.loads(),contain only the required JSON object, with no extra commentary, explanation, or markdown formatting.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.get_prompt_set_BC(set_A)},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_img}"
                            },
                        },
                    ],
                },
            ],
            response_format={
                "type": "json_schema",
                "schema": CausalChains.model_json_schema(),
            },
            temperature=0.3,
            top_p=0.7,
            max_tokens=2048,
        )

        raw = json.loads(response.choices[0].message.content)
        print(raw)
        print(f"❗❗generate by model B\n\n")
        return raw

    def generate_distractors(
        self, image_path: str, set_A: str, set_B: str, set_C: str
    ) -> dict:
        # Load image and encode to base64
        with open(image_path, "rb") as f:
            base64_img = base64.b64encode(f.read()).decode("utf-8")

        response = self.client.chat.completions.create(
            model=self.model_distractors,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in generating plausible but visually incorrect causal distractors. Respond only in JSON.Please only return a strictly valid JSON object with all fields completed exactly as defined.The JSON must be:syntactically correct (no missing commas, colons, or quotes),fully parseable by json.loads(),contain only the required JSON object, with no extra commentary, explanation, or markdown formatting.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.get_prompt_distractors(set_A, set_B, set_C),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_img}"
                            },
                        },
                    ],
                },
            ],
            response_format={
                "type": "json_schema",
                "schema": CausalChains.model_json_schema(),
            },
            temperature=0.5,
            top_p=0.8,
            max_tokens=1024,
        )

        raw = json.loads(response.choices[0].message.content)
        print(raw)
        return raw

    def generate_causal_chain(self, image_path: str) -> dict:
        # Generate Set_A using the image
        set_A = self.generate_set_A(image_path)

        # Generate Set_B, Set_C, and Right_answer based on Set_A and the image
        set_BC_result = self.generate_set_BC(image_path, set_A)
        set_B = set_BC_result["set_B"]
        set_C = set_BC_result["set_C"]
        Right_answer = set_BC_result["Right_answer"]

        # Generate distractors for Set_B and Set_C
        distractor_result = self.generate_distractors(image_path, set_A, set_B, set_C)
        distractor_B = distractor_result["set_B"]
        distractor_C = distractor_result["set_C"]

        # Combine the correct and distractor options for Set_B and Set_C
        set_B_combined = set_B + "\n" + distractor_B
        set_C_combined = set_C + "\n" + distractor_C

        # Create the final JSON structure
        result = {
            "set_A": set_A,
            "set_B": set_B_combined,
            "set_C": set_C_combined,
            "Right_answer": Right_answer,
            "img_path": image_path,
        }
        return result

    def process_folder(
        self, input_folder: str, output_json_path: str, api_keys: list[str] = None
    ):
        import time

        all_images = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
            all_images.extend(glob.glob(os.path.join(input_folder, ext)))

        print(f"🔍 Found {len(all_images)} images in {input_folder}")

        if api_keys is None:
            api_keys = [self.client.api_key]

        key_index = 0
        current_key = api_keys[key_index]

        temp_output_path = output_json_path + ".tmp"

        with open(temp_output_path, "w", encoding="utf-8") as f:
            f.write("[\n")
            first = True

            for img_path in tqdm(all_images, desc="Processing images"):
                retry_count = 0
                while retry_count < 5:
                    try:
                        result = self.generate_causal_chain(img_path)
                        json.dumps(result)

                        if not first:
                            f.write(",\n")
                        else:
                            first = False

                        json.dump(result, f, ensure_ascii=False, indent=2)
                        f.flush()
                        tqdm.write(f"✅ Processed: {os.path.basename(img_path)}")
                        break

                    except Exception as e:
                        msg = str(e)

                        if any(
                            keyword in msg.lower()
                            for keyword in [
                                "credit limit",
                                "insufficient quota",
                                "error code: 402",
                                '"type_": "credit_limit"',
                                "payment required",
                                "quota exceeded",
                            ]
                        ):
                            tqdm.write(f"💳 API quota exhausted for key {current_key}")
                            key_index += 1
                            if key_index >= len(api_keys):
                                tqdm.write("❌ All API keys exhausted. Stopping...")
                                f.write("\n]")
                                f.close()
                                os.rename(temp_output_path, output_json_path)
                                return
                            current_key = api_keys[key_index]
                            self.client = Together(api_key=current_key)
                            tqdm.write(f"🔁 Switched to new API key.")
                            continue

                        if "Expecting" in msg and "delimiter" in msg:
                            tqdm.write(
                                f"⚠️ JSON parse error. Retrying {os.path.basename(img_path)}..."
                            )
                            retry_count += 1
                            time.sleep(1)
                            continue

                        tqdm.write(f"❌ Failed: {os.path.basename(img_path)} - {e}")
                        break

            f.write("\n]")
            f.flush()

        os.rename(temp_output_path, output_json_path)
        tqdm.write(f"\n✅ All results saved to: {output_json_path}")


if __name__ == "__main__":
    API_KEYS = load_api_keys("./openrouter.txt")
    INPUT_FOLDER = (
        "./data/flickr30k"
    )
    OUTPUT_PATH = "./data/output.json"

    generator = UnifiedCausalChainGenerator(api_key=API_KEYS[0])
    generator.process_folder(INPUT_FOLDER, OUTPUT_PATH, api_keys=API_KEYS)
