import os
import json
import torch
from typing import Dict, List
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    TaskType
)
import wandb
from tqdm import tqdm

def load_model_and_tokenizer(model_path: str):
    """모델과 토크나이저를 로드합니다."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_8bit=True
    )
    
    # 8비트 학습을 위한 모델 준비
    model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer

def create_lora_config():
    """LoRA 설정을 생성합니다."""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,  # LoRA 랭크
        lora_alpha=32,  # LoRA 알파
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],  # 타겟 모듈
        bias="none"
    )

def format_prompt(example: Dict) -> str:
    """학습 데이터 포맷을 생성합니다."""
    system_prompt = """<|system|>
You are a professional English teacher. Your task is to generate multiple-choice questions (MCQs) based on the user's input text.
You must strictly follow these rules:

1. The passage MUST contain exactly three blanks marked as (A), (B), and (C).
2. Each blank should be placed in a contextually appropriate position.
3. The options should be challenging but fair, with only one clearly correct answer.
4. You must ONLY use words and phrases that appear in the original input text for the blanks and options.
5. DO NOT add any new content or context to the original text.
</|system|>"""

    user_prompt = f"<|user|>\n{example['input']}\n</|user|>"
    assistant_prompt = f"<|assistant|>\n{json.dumps(example['output'], ensure_ascii=False)}\n</|assistant|>"
    
    return f"{system_prompt}\n{user_prompt}\n{assistant_prompt}"

def prepare_dataset(dataset_path: str, tokenizer):
    """데이터셋을 준비합니다."""
    dataset = load_dataset("json", data_files={
        "train": os.path.join(dataset_path, "train.jsonl"),
        "validation": os.path.join(dataset_path, "val.jsonl")
    })
    
    def tokenize_function(examples):
        texts = [format_prompt(example) for example in examples]
        return tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=2048,
            return_tensors="pt"
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    return tokenized_dataset

def train():
    """LoRA 파인튜닝을 실행합니다."""
    # 설정
    model_path = "models/llama-2-7b-chat.Q4_K_M.gguf"  # 실제 모델 경로로 수정 필요
    dataset_path = "finetune/data"
    output_dir = "finetune/output"
    
    # wandb 초기화
    wandb.init(project="llama-mcq-finetune")
    
    # 모델과 토크나이저 로드
    model, tokenizer = load_model_and_tokenizer(model_path)
    
    # LoRA 설정
    lora_config = create_lora_config()
    model = get_peft_model(model, lora_config)
    
    # 데이터셋 준비
    dataset = prepare_dataset(dataset_path, tokenizer)
    
    # 학습 인자 설정
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        report_to="wandb",
        fp16=True
    )
    
    # 데이터 콜레이터
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # 트레이너 설정
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator
    )
    
    # 학습 실행
    trainer.train()
    
    # 모델 저장
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # wandb 종료
    wandb.finish()

if __name__ == "__main__":
    train() 