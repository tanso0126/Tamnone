import json

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def merge_jsonl_files(file1, file2, output_path, deduplicate=True):
    data1 = load_jsonl(file1)
    data2 = load_jsonl(file2)
    merged = data1 + data2

    if deduplicate:
        seen = set()
        unique = []
        for item in merged:
            key = (item["input"], item["output"])
            if key not in seen:
                seen.add(key)
                unique.append(item)
        merged = unique

    save_jsonl(merged, output_path)
    print(f"{len(merged)}개 항목이 병합되어 저장되었습니다: {output_path}")

if __name__ == "__main__":
    merge_jsonl_files(
        "train_data.jsonl",
        "train_data_augmented.jsonl",
        "train_data_merged.jsonl"
    )
