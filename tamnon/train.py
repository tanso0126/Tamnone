from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
import torch

# ✅ 모델 선택 (KoAlpaca 또는 KoVicuna 등 HuggingFace 모델)
BASE_MODEL = "beomi/KoAlpaca-Polyglot-5.8B"  # 또는 "nlpai-lab/kullm-polyglot-5.8b"
OUTPUT_DIR = "./output"

# ✅ 데이터 로드 (jsonl 형식)
dataset = load_dataset("json", data_files="train_data.jsonl", split="train")

# ✅ Tokenizer & Model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="auto")

# ✅ LoRA 설정
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)

# ✅ 전처리 함수
def preprocess(example):
    prompt = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
    inputs = tokenizer(prompt, truncation=True, max_length=1024, padding="max_length")
    inputs["labels"] = inputs["input_ids"].copy()
    return inputs

tokenized_dataset = dataset.map(preprocess)

# ✅ 학습 인자
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    evaluation_strategy="no",
    save_strategy="epoch",
    logging_steps=10,
    learning_rate=2e-4,
    fp16=True,
    save_total_limit=2,
    weight_decay=0.01,
    report_to="none"
)

# ✅ Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# ✅ 학습 시작
trainer.train()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
