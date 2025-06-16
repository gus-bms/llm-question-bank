import os
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline, StoppingCriteria, StoppingCriteriaList
import logging
from dotenv import load_dotenv
import uvicorn
import platform
import signal
import sys
from contextlib import asynccontextmanager, contextmanager
import gc
import time
from datetime import datetime
import psutil
import asyncio
from watchfiles import awatch
from pathlib import Path
from fastapi.middleware.gzip import GZipMiddleware
import json
import transformers
from utils import should_retry_response, extract_json_from_response

# CUDA 메모리 할당 설정
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 환경 변수 로드
load_dotenv()

# CPU 최적화
num_cpus = psutil.cpu_count(logical=True)
torch.set_num_threads(num_cpus)
os.environ["OMP_NUM_THREADS"] = str(num_cpus)
os.environ["MKL_NUM_THREADS"] = str(num_cpus)
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# uvloop 사용 (서버 성능 향상)
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

# 서버 설정
DEV_MODE = False
PORT = int(os.getenv("API_PORT", "8004"))  
HOST = os.getenv("API_HOST", "0.0.0.0")

# Mac 환경 감지
IS_MAC = platform.system() == "Darwin"
IS_APPLE_SILICON = IS_MAC and platform.machine() == "arm64"

# 메모리 최적화 설정
MAX_MEMORY = 16 * 1024 * 1024 * 1024  
TORCH_MEMORY_FRACTION = 0.9

# GPU 설정
USE_GPU = torch.cuda.is_available()  
DEVICE = "cuda" if USE_GPU else "cpu"
DEVICE_MAP = "auto"
TORCH_DTYPE = torch.float16 if USE_GPU else torch.float32

# 모델 설정
MODEL_PATH = "models/hyperclovax-1_5b"
CONTEXT_LENGTH = 2048
MAX_NEW_TOKENS = 2048

# 시스템 프롬프트 설정
SYSTEM_PROMPT_FILE = "system_prompt.txt"
SYSTEM_PROMPT = ""

# 로깅 설정
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{log_dir}/api_server_hyper_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 모델/토크나이저 상태
model = None
tokenizer = None
model_loaded = False

class Timer:
    def __init__(self, name: str):
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"[{self.name}] 시작!")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        logger.info(f"[{self.name}] 완료 - 소요시간: {duration:.2f}초")


def load_system_prompt():
    global SYSTEM_PROMPT
    try:
        if os.path.exists(SYSTEM_PROMPT_FILE):
            with open(SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
                SYSTEM_PROMPT = f.read().strip()
            logger.info("시스템 프롬프트 로드 완료")
        else:
            logger.warning(f"시스템 프롬프트 파일이 없습니다: {SYSTEM_PROMPT_FILE}")
            SYSTEM_PROMPT = ""
    except Exception as e:
        logger.error(f"시스템 프롬프트 로드 오류: {e}")
        SYSTEM_PROMPT = ""

async def watch_system_prompt():
    async for _ in awatch(SYSTEM_PROMPT_FILE):
        logger.info("시스템 프롬프트 변경 감지, 재로드...")
        load_system_prompt()

def load_model():
    global model, tokenizer, model_loaded
    with Timer("모델 로드"):
        if USE_GPU:
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(TORCH_MEMORY_FRACTION)
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.deterministic = False

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            use_fast=True,
            padding_side="left",
            truncation_side="left"
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=TORCH_DTYPE,
            device_map=DEVICE_MAP,
            low_cpu_mem_usage=True,
            max_memory={0: f"{int(MAX_MEMORY * TORCH_MEMORY_FRACTION / 1024**3)}GB"} if USE_GPU else None,
            trust_remote_code=True,
            use_cache=True,
            attn_implementation="eager"
        )
        model_loaded = True
    return model, tokenizer

class GenerateRequest(BaseModel):
    text: str
    system_prompt: str = None

class GenerateResponse(BaseModel):
    result: str
    tokens: int
    time_taken: float

class StopOnStrings(StoppingCriteria):
    def __init__(self, stop_list, tokenizer):
        self.stop_ids = [tokenizer.encode(s, add_special_tokens=False) for s in stop_list]
        self.tokenizer = tokenizer  # tokenizer 참조 저장
        
    def __call__(self, input_ids, scores, **kwargs):
        for stop in self.stop_ids:
            if len(input_ids[0]) >= len(stop):  # 입력 길이 체크 추가
                if input_ids[0][-len(stop):].tolist() == stop:
                    return True
        return False

app = FastAPI(title="HyperCLOVAX API Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

@app.on_event("startup")
async def startup_event():
    global model, tokenizer, model_loaded
    logger.info("서버 시작 중...")
    load_system_prompt()
    asyncio.create_task(watch_system_prompt())
    model, tokenizer = load_model()
    logger.info("모델 로드 완료")

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    global model, tokenizer
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다.")

    start_time = time.time()
    
    # 요청에 system_prompt가 있으면 우선 사용
    system_prompt = request.system_prompt if request.system_prompt else SYSTEM_PROMPT

    # 채팅 형식으로 입력 구성
    chat = [
        {"role": "tool_list", "content": ""},
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": request.text}
    ]
    
    # Chat 템플릿 적용
    prompt = tokenizer.apply_chat_template(
        chat,
        add_generation_prompt=True,
        tokenize=False
    )
    
    # TextGenerationPipeline 생성
    pipe = TextGenerationPipeline(
        model=model,
        tokenizer=tokenizer,
        return_full_text=False  # prompt를 출력에서 제외
    )
    
    # 생성
    stop_criteria = StoppingCriteriaList([StopOnStrings(["<|endofturn|>", "<|stop|>"], tokenizer)])
    with torch.no_grad():
        try:
            generated = pipe(
                prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.0,
                top_p=1.0,
                top_k=50,
                repetition_penalty=1.05,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                stopping_criteria=stop_criteria,
            )[0]["generated_text"]
        except Exception as e:
            logger.error(f"생성 중 오류 발생: {str(e)}")
            raise HTTPException(status_code=500, detail=f"생성 중 오류 발생: {str(e)}")
    
    logger.info("=== 모델 원본 출력 시작 ===")
    logger.info(generated)
    logger.info("=== 모델 원본 출력 끝 ===")
    
    try:
        data = json.loads(generated)
    except json.JSONDecodeError:
        data = {"text": generated}

    # 응답 반환
    time_taken = time.time() - start_time
    response = GenerateResponse(
        result=data.get("text", generated),
        tokens=len(tokenizer.encode(generated)),
        time_taken=time_taken
    )
    
    logger.info(f"응답({time_taken:.2f}s): {response.result}")
    return response

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model_loaded}

if __name__ == "__main__":
    uvicorn.run(
        "api_server_hyper:app",
        host=HOST,
        port=PORT,
        reload=True,
        workers=num_cpus,
        loop="uvloop",
        http="httptools"
    )
