import json
import os
import re

def count_matching_answers(jsonl_file_path, output_dir=None):
    def parse_chain(s: str):
        mapping = {}
        if not isinstance(s, str):
            return mapping
        for chunk in s.split(","):
            seg = chunk.strip()
            if not seg:
                continue
            m = re.match(r'^(A[123])\s*->\s*(B\d+)\s*->\s*(C\d+)$', seg)
            if m:
                a, b, c = m.groups()
                mapping[a] = (b, c)
        return mapping

    try:
        total_samples = 0
        fully_correct = 0

        slot_keys = ["A1", "A2", "A3"]
        slot_totals = {k: 0 for k in slot_keys}
        slot_correct = {k: 0 for k in slot_keys}

        with open(jsonl_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)

                total_samples += 1

                right_answer = item.get("Right_answer", "")
                pred_answer = item.get("Predicted_Answer", "")

                right_map = parse_chain(right_answer)
                pred_map  = parse_chain(pred_answer)

                if right_map == pred_map and len(right_map) > 0:
                    fully_correct += 1

                for k in slot_keys:
                    if k in right_map:
                        slot_totals[k] += 1
                        if k in pred_map and pred_map[k] == right_map[k]:
                            slot_correct[k] += 1

        sample_acc = (fully_correct / total_samples) if total_samples > 0 else 0.0
        slot_acc = {
            "A1": (slot_correct["A1"] / slot_totals["A1"]) if slot_totals["A1"] > 0 else 0.0,
            "A2": (slot_correct["A2"] / slot_totals["A2"]) if slot_totals["A2"] > 0 else 0.0,
            "A3": (slot_correct["A3"] / slot_totals["A3"]) if slot_totals["A3"] > 0 else 0.0,
        }

        print(f"Fully Matched Samples:{fully_correct}")
        print(f"Total Samples:{total_samples}")
        print(f"Sample-level Accuracy (All 3 Chains Correct): {sample_acc:.4f}")
        print(f"（A1）: {slot_acc['A1']:.4f}")
        print(f"（A2）: {slot_acc['A2']:.4f}")
        print(f"（A3）: {slot_acc['A3']:.4f}")

        if output_dir:
            base_name = os.path.splitext(os.path.basename(jsonl_file_path))[0]
            output_file_name = f"{base_name}_match_accuracy.json"
            output_path = os.path.join(output_dir, output_file_name)
            os.makedirs(output_dir, exist_ok=True)

            result_obj = {
                "total_samples": total_samples,
                "correct_samples_fully_matched": fully_correct,
                "sample_accuracy_full_match": sample_acc,
                "per_slot": {
                    "A1": {
                        "name": "A1",
                        "total": slot_totals["A1"],
                        "correct": slot_correct["A1"],
                        "accuracy": slot_acc["A1"],
                    },
                    "A2": {
                        "name": "A2",
                        "total": slot_totals["A2"],
                        "correct": slot_correct["A2"],
                        "accuracy": slot_acc["A2"],
                    },
                    "A3": {
                        "name": "A3",
                        "total": slot_totals["A3"],
                        "correct": slot_correct["A3"],
                        "accuracy": slot_acc["A3"],
                    },
                }
            }

            with open(output_path, 'w', encoding='utf-8') as out_file:
                json.dump(result_obj, out_file, ensure_ascii=False, indent=4)
            print(f"Results Saved To: {output_path}")

        return fully_correct, sample_acc

    except FileNotFoundError:
        print(f"File Not Found {jsonl_file_path} ")
        return 0, 0.0
    except json.JSONDecodeError:
        print("JSON Decoding Error")
        return 0, 0.0
    except Exception as e:
        print(f"Unknown Error:{e}")
        return 0, 0.0



jsonl_file = "..."
output_dir = "..."

count_matching_answers(jsonl_file, output_dir)