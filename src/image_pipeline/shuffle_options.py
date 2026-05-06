import json
import random
import re
from typing import List, Tuple

def parse_items(text: str, prefix: str) -> List[Tuple[str, str]]:
    lines = text.strip().split("\n")
    pattern = re.compile(rf"({prefix}\d+):\s*(.*)")
    return [(match.group(1), match.group(2)) for line in lines if (match := pattern.match(line))]

def shuffle_content_keep_labels(items: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], dict]:
    labels = [label for label, _ in items]
    contents = [content for _, content in items]
    shuffled = random.sample(contents, len(contents))
    new_items = list(zip(labels, shuffled))
    content_to_new_label = {content: label for label, content in new_items}
    return new_items, content_to_new_label

def rebuild_text(items: List[Tuple[str, str]]) -> str:
    return "\n".join(f"{label}: {content}" for label, content in items)

def update_right_answer(text: str, old_items: List[Tuple[str, str]], content_to_label: dict, prefix: str) -> str:
    content_lookup = {label: content for label, content in old_items}
    updated_links = []
    for link in text.split(", "):
        parts = link.split("->")
        for i in range(len(parts)):
            if parts[i].startswith(prefix):
                old_label = parts[i]
                content = content_lookup[old_label]
                parts[i] = content_to_label.get(content, old_label)
        updated_links.append("->".join(parts))
    return ", ".join(updated_links)

def process_entry(entry: dict) -> dict:
    # Process set_B
    print(entry["img_path"])
    old_B = parse_items(entry["set_B"], "B")
    new_B, b_map = shuffle_content_keep_labels(old_B)
    entry["set_B"] = rebuild_text(new_B)

    # Process set_C
    old_C = parse_items(entry["set_C"], "C")
    new_C, c_map = shuffle_content_keep_labels(old_C)
    entry["set_C"] = rebuild_text(new_C)

    # Update Right_answer
    entry["Right_answer"] = update_right_answer(entry["Right_answer"], old_B, b_map, "B")
    entry["Right_answer"] = update_right_answer(entry["Right_answer"], old_C, c_map, "C")

    return entry

def process_file(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed_data = [process_entry(entry) for entry in data]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Finished processing. Output saved to: {output_path}")

# Example usage
if __name__ == "__main__":
    process_file("./data/flick30k_data_corrected.json", "./data/flick30k_data_shuff.json")

