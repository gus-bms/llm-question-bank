import json
import os
from typing import List, Dict
import random
from pathlib import Path

def create_training_example(input_text: str) -> Dict:
    """주어진 입력 텍스트로 학습 예제를 생성합니다."""
    # 실제로는 여기서 모델을 사용하여 예제를 생성하거나,
    # 미리 준비된 예제 데이터를 사용할 수 있습니다.
    return {
        "input": input_text,
        "output": {
            "question": "문맥에 맞는 낱말로 가장 적절한 것을 고르세요.",
            "passage": f"{input_text[:len(input_text)//3]} (A) {input_text[len(input_text)//3:2*len(input_text)//3]} (B) {input_text[2*len(input_text)//3:]} (C)",
            "options": "option1 - option2 - option3//option4 - option5 - option6//option7 - option8 - option9//option10 - option11 - option12//option13 - option14 - option15",
            "answer": random.randint(1, 5),
            "explanation": "정답에 대한 설명"
        }
    }

def prepare_dataset(input_texts: List[str], output_path: str, split: float = 0.9):
    """데이터셋을 준비하고 train/val로 분할합니다."""
    examples = [create_training_example(text) for text in input_texts]
    random.shuffle(examples)
    
    # train/val 분할
    split_idx = int(len(examples) * split)
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]
    
    # 디렉토리 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # train 데이터 저장
    train_path = os.path.join(os.path.dirname(output_path), "train.jsonl")
    with open(train_path, "w", encoding="utf-8") as f:
        for example in train_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    # val 데이터 저장
    val_path = os.path.join(os.path.dirname(output_path), "val.jsonl")
    with open(val_path, "w", encoding="utf-8") as f:
        for example in val_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    print(f"데이터셋 준비 완료:")
    print(f"- 학습 데이터: {len(train_examples)}개")
    print(f"- 검증 데이터: {len(val_examples)}개")

if __name__ == "__main__":
    # 예시 입력 텍스트들
    sample_texts = [
        "You would be confused momentarily, but laugh when you learned that the term 'robot' also means 'traffic light' in South Africa.",
        "The ancient city of Rome was built on seven hills, which made it easier to defend against invaders.",
        "Scientists have discovered that honey never spoils, as it contains natural preservatives that prevent bacterial growth.",
        # 더 많은 예시 텍스트 추가 필요
    ]
    
    # 데이터셋 준비
    prepare_dataset(
        input_texts=sample_texts,
        output_path="finetune/data/dataset.jsonl"
    ) 