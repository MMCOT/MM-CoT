import json

def parse_right_answer(answer_str):
    chains = answer_str.split(', ')
    A_order = [chain.split('->')[0] for chain in chains]
    return A_order, chains

def correct_right_answer(entry):
    A_order, chains = parse_right_answer(entry["Right_answer"])
    if A_order != ["A1", "A2", "A3"]:
        try:
            a1_index = next(i for i, chain in enumerate(chains) if chain.startswith("A1->"))
            a2_index = next(i for i, chain in enumerate(chains) if chain.startswith("A2->"))
            chains[a1_index], chains[a2_index] = chains[a2_index], chains[a1_index]
            entry["Right_answer"] = ', '.join(chains)
            # entry["_corrected"] = True
        except StopIteration:
            entry["_error"] = "Missing A1 or A2 in Right_answer"

    return entry

def process_json_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    corrected_data = [correct_right_answer(entry) for entry in data]

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(corrected_data, f, indent=2, ensure_ascii=False)

    print(f"Processing complete: {len(data)} records in total, saved to {output_path}")

if __name__ == "__main__":
    input_file = "./data/flick30k_data.json"
    output_file = "./data/flick30k_data_corrected.json"
    process_json_file(input_file, output_file)
