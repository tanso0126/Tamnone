from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

MODEL_PATH = "./output"  # train.py에서 저장한 경로

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

def ask(instruction, input_text):
    prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    result = pipe(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)[0]["generated_text"]
    print(result.split("### Response:\n")[-1].strip())

# 🔍 테스트 예시
ask("갈등 발화를 분석하고, 갈등이 해결될 수 있도록 조언을 해주세요.",
    "평소에 거지처럼 입고다니는 직장 동료가 깔끔한 새 옷을 입고 출근하는데 어디서 훔친 것 같지 않나요?")
