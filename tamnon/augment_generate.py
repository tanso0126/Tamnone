import json
import random
import glob

# ✅ 발화 변형 템플릿
def augment_variants(original):
    base = original.replace("직장 동료", "그 사람").replace("입고 출근", "입고 옴")

    variants = [
        f"야, {base} 어디서 훔친 거 같지 않아?",
        f"{base}… 혹시 훔친 거 아님? ㄷㄷ",
        f"{base} ㅋㅋ 진짜 훔쳤나봐 ㅋㅋ",
        f"{base} 설마 훔친 건 아니겠죠?",
        f"오~ {base}라니? 거지에서 갑자기 금수저임?",
        f"뭔가 수상한데~ {base}…",
        f"{base} 어울리지도 않는데 어디서 난 거야?",
    ]

    return list(set(variants + [original]))

# ✅ Output 생성 템플릿
def generate_output(text):
    return (
        f"이 발언은 상대에 대한 조롱 또는 의심 표현을 포함하고 있어 갈등을 유발할 수 있습니다. "
        f"외모나 배경에 대한 편견보다는, 긍정적인 관찰과 존중의 태도가 필요합니다."
    )

# ✅ Instruction 포맷 변환
def convert_augmented_data(files):
    results = []
    for path in files:
        with open(path, encoding="utf-8") as f:
            j = json.load(f)
            for entry in j["data"]:
                variations = augment_variants(entry["instruct_text"])
                for v in variations:
                    results.append({
                        "instruction": "갈등 발화를 분석하고, 갈등이 해결될 수 있도록 조언을 해주세요.",
                        "input": v,
                        "output": generate_output(v)
                    })
    return results

# ✅ 저장
def save_as_jsonl(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

# ✅ 실행
if __name__ == "__main__":
    input_files = glob.glob("30_02_03_01_20231006_201*.json")
    augmented = convert_augmented_data(input_files)
    save_as_jsonl(augmented, "train_data_augmented.jsonl")
    print(f"{len(augmented)}개 발화 변형이 생성되어 저장되었습니다.")
