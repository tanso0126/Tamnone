import json
import glob

# ✅ 유해 발화에 대해 자동으로 조언 생성하는 템플릿
def generate_output(text):
    return (
        f"이 발언은 동료를 외모로 비하하고, 도둑질을 암시하는 심각한 모욕적 표현입니다. "
        f"직장 내 갈등을 유발할 수 있으므로, 변화에 대해 긍정적으로 바라보는 태도와 존중이 필요합니다."
    )

# ✅ JSON 파일들 변환
def convert_to_instruction_format(files):
    results = []
    for path in files:
        with open(path, encoding="utf-8") as f:
            j = json.load(f)
            for entry in j["data"]:
                result = {
                    "instruction": "갈등 발화를 분석하고, 갈등이 해결될 수 있도록 조언을 해주세요.",
                    "input": entry["instruct_text"],
                    "output": generate_output(entry["instruct_text"])
                }
                results.append(result)
    return results

# ✅ JSONL 저장
def save_as_jsonl(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

# ✅ 실행
if __name__ == "__main__":
    input_files = glob.glob("30_02_03_01_20231006_201*.json")
    converted = convert_to_instruction_format(input_files)
    save_as_jsonl(converted, "train_data.jsonl")
    print(f"{len(converted)}개 항목이 변환되어 train_data.jsonl로 저장되었습니다.")
