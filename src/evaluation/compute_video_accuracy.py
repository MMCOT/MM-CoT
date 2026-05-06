import json
from collections import defaultdict

INPUT = "./data/video_result.json"  # Update to your result file


def parse_list(text):
    return [item.strip() for item in text.strip("[]").split("], [")]


def parse_cf_chain(text):
    cause_str, result_str = text.split("->")
    causes = parse_list(cause_str.strip())
    result = result_str.strip()
    return causes, result


def check_prediction(entry):
    try:
        gt_chosen = set(parse_list(entry["Critical Conditions"]))
        gt_causes, gt_result = parse_cf_chain(entry["Counterfactual Causal Chain"])

        pred = json.loads(entry["Predicted_CF_Chain"])
        pred_chosen = set(parse_list(pred["chosen_conditions"]))
        pred_causes = parse_list(pred["counterfactual_conditions"])
        pred_result = pred["counterfactual_result"].strip()
        pred_chain = pred["counterfactual_chain"].strip()

        expected_chain_1 = f"[{'], ['.join(pred_causes)}] -> {pred_result.strip()}"
        expected_chain_2 = f"[{'], ['.join(pred_causes[::-1])}] -> {pred_result.strip()}"

        return (
            pred_chosen == gt_chosen
            and set(pred_causes) == set(gt_causes)
            and pred_result == gt_result
            and (pred_chain == expected_chain_1 or pred_chain == expected_chain_2)
        )

    except Exception:
        return False


def compute_accuracy(data):
    correct = sum(check_prediction(entry) for entry in data)
    total = len(data)
    acc = correct / total if total > 0 else 0.0
    print(f"Overall Accuracy: {acc:.4f} ({correct}/{total})")
    return acc


def compute_accuracy_by_length_and_difficulty(data):
    stats = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))

    for entry in data:
        length = entry.get("length_category", None)
        diff = entry.get("difficulty", None)
        if length is None or diff is None:
            continue

        stats[length][diff]["total"] += 1
        if check_prediction(entry):
            stats[length][diff]["correct"] += 1

    print("\n📊 Accuracy by Length Category and Difficulty:")
    for length in sorted(stats.keys()):
        print(f"\nLength category: {length}")
        for diff in sorted(stats[length].keys()):
            c = stats[length][diff]["correct"]
            t = stats[length][diff]["total"]
            acc = c / t if t > 0 else 0.0
            print(f"  Difficulty {diff}: {acc:.4f} ({c}/{t})")


if __name__ == "__main__":
    with open(INPUT, "r", encoding="utf-8") as f:
        data = json.load(f)

    compute_accuracy(data)
    compute_accuracy_by_length_and_difficulty(data)
