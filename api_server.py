import os
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import logging
from dotenv import load_dotenv
import uvicorn
import platform
import signal
import sys
from contextlib import asynccontextmanager, contextmanager
import gc  # 가비지 컬렉션을 위한 모듈 추가
import time
from datetime import datetime
import psutil
from llama_cpp import Llama
import asyncio
from watchfiles import awatch
from pathlib import Path
from fastapi.middleware.gzip import GZipMiddleware
import json

# 환경 변수 로드
load_dotenv()

# 서버 설정
DEV_MODE = os.getenv("DEV_MODE", "true").lower() == "true"
PORT = int(os.getenv("API_PORT", "8000"))
HOST = os.getenv("API_HOST", "0.0.0.0")

# Mac 환경 감지
IS_MAC = platform.system() == "Darwin"
IS_APPLE_SILICON = IS_MAC and platform.machine() == "arm64"

# 메모리 최적화 설정
MAX_MEMORY = 4 * 1024 * 1024 * 1024
TORCH_MEMORY_FRACTION = 0.2

# 강제로 CPU 모드 사용
USE_GPU = False
DEVICE = "cpu"
DEVICE_MAP = "cpu"
TORCH_DTYPE = torch.float32

# 오프로딩 설정
OFFLOAD_FOLDER = "offload"
OFFLOAD_INDEX = 0

# 로깅 설정
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)  # logs 디렉토리가 없으면 생성

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{log_dir}/api_server_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 전역 변수
model = None
model_lock = asyncio.Lock()
file_watcher_task: Optional[asyncio.Task] = None
should_reload = False
model_loaded = False
model_load_error: Optional[str] = None

# 시그널 핸들러
def signal_handler(signum, frame):
    logger.info(f"시그널 수신: {signum}")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
if hasattr(signal, "SIGBREAK"):
    signal.signal(signal.SIGBREAK, signal_handler)

# 시스템 프롬프트 파일 경로
SYSTEM_PROMPT_FILE = "system_prompt.txt"

def load_system_prompt() -> str:
    """시스템 프롬프트를 파일에서 로드합니다."""
    try:
        with open(SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"시스템 프롬프트 로드 실패: {str(e)}")
        raise

def format_prompt(input_text: str) -> str:
    """입력 텍스트를 프롬프트 형식으로 변환합니다."""
    system_prompt = load_system_prompt()
    return f"{system_prompt}\n\n<|user|>\n{input_text}\n\n<|assistant|>\n"

def validate_json_response(response: str) -> bool:
    """JSON 응답의 유효성을 검증합니다."""
    try:
        data = json.loads(response)
        required_keys = ["question", "passage", "options", "answer", "explanation"]
        
        # 필수 키 확인
        if not all(key in data for key in required_keys):
            logger.warning("응답에 필수 키가 누락됨")
            return False
            
        # 빈칸 개수 확인
        blanks = ["(A){{|bold-underline|}}", "(B){{|bold-underline|}}", "(C){{|bold-underline|}}"]
        blank_count = sum(1 for blank in blanks if blank in data["passage"])
        if blank_count < 2:  # 최소 2개의 빈칸은 있어야 함
            logger.warning(f"빈칸 개수가 부족함: {blank_count}")
            return False
            
        # 옵션 형식 확인
        options = data["options"].split("//")
        if len(options) < 3:  # 최소 3개의 옵션은 있어야 함
            logger.warning(f"옵션 개수가 부족함: {len(options)}")
            return False
            
        for option in options:
            phrases = option.split(" - ")
            if len(phrases) < 2:  # 최소 2개의 구문은 있어야 함
                logger.warning(f"옵션 내 구문 개수가 부족함: {len(phrases)}")
                return False
                
        return True
    except json.JSONDecodeError:
        logger.warning("JSON 파싱 실패")
        return False
    except Exception as e:
        logger.warning(f"응답 검증 중 예외 발생: {str(e)}")
        return False

def generate_text(prompt: str, max_retries: int = 3) -> str:
    """텍스트를 생성하고 유효성을 검증합니다."""
    for attempt in range(max_retries):
        try:
            logger.info(f"생성 시도 {attempt + 1}/{max_retries}")
            logger.info(f"프롬프트 길이: {len(prompt)}")
            
            response = model.create_completion(
                prompt=prompt,
                max_tokens=MAX_NEW_TOKENS,
                temperature=0.1,
                top_p=0.99,
                repeat_penalty=1.0,
                stop=["<|user|>", "<|system|>"],
                echo=False
            )
            
            # 응답 전체 로깅
            logger.info(f"모델 응답: {response}")
            
            if not response or "choices" not in response:
                logger.error("모델이 응답을 반환하지 않음")
                continue
                
            if not response["choices"]:
                logger.error("응답 choices가 비어있음")
                continue
                
            # 응답에서 JSON 부분만 추출
            response_text = response["choices"][0]["text"].strip()
            logger.info(f"추출된 텍스트: {response_text}")
            
            if not response_text:
                logger.warning(f"빈 응답 (시도 {attempt + 1}/{max_retries})")
                continue
                
            # JSON 유효성 검증
            if validate_json_response(response_text):
                logger.info("유효한 응답 생성 성공")
                return response_text
            else:
                logger.warning(f"유효하지 않은 응답 (시도 {attempt + 1}/{max_retries})")
                # 마지막 시도에서는 유효하지 않은 응답이라도 반환
                if attempt == max_retries - 1:
                    logger.info("마지막 시도에서 유효하지 않은 응답 반환")
                    return response_text
                
        except Exception as e:
            logger.error(f"생성 중 예외 발생 (시도 {attempt + 1}/{max_retries}): {str(e)}")
            logger.exception("상세 오류 정보:")
            
    raise Exception("최대 재시도 횟수 초과")

# 시스템 프롬프트 설정
SYSTEM_PROMPT = load_system_prompt()

async def watch_for_changes(path: str):
    """awatch 기반 파일 변경 감지 코루틴"""
    async for changes in awatch(path):
        for _, changed_path in changes:
            if str(changed_path).endswith(('.py', '.env', '.txt')):
                logger.info(f"파일 변경 감지: {changed_path} → 서버 재시작")
                # 프로세스를 강제 종료해서, uvicorn --reload 가 재시작
                os._exit(0)
        await asyncio.sleep(0.1)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션의 시작과 종료를 관리합니다."""
    global model, file_watcher_task, should_reload, model_loaded, model_load_error, SYSTEM_PROMPT

    # 시작 시
    try:
        logger.info("서버 시작 중...")
        model = load_model()
        model_loaded = True
        logger.info("모델 로드 완료")

        if DEV_MODE:
            watch_path = os.path.dirname(os.path.abspath(__file__))
            file_watcher_task = asyncio.create_task(watch_for_changes(watch_path))
            logger.info(f"개발 모드: 파일 감시 태스크 시작 (디렉토리: {watch_path})")
    except Exception as e:
        model_load_error = str(e)
        logger.error(f"시작 중 오류 발생: {model_load_error}")
        raise

    yield

    # 종료 시
    try:
        logger.info("서버 종료 중...")
        if file_watcher_task:
            file_watcher_task.cancel()
            logger.info("파일 감시 태스크 취소")
        if model:
            del model
            torch.cuda.empty_cache()
            logger.info("모델 메모리 해제")
    except Exception as e:
        logger.error(f"종료 중 오류 발생: {str(e)}")

# FastAPI 앱 생성
app = FastAPI(lifespan=lifespan)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gzip 압축 미들웨어 추가
app.add_middleware(GZipMiddleware, minimum_size=1000)

# 요청/응답 모델 정의
class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    num_return_sequences: int = 1
    do_sample: bool = True
    num_beams: int = 1

class GenerationResponse(BaseModel):
    generated_text: str

# 모델 설정
MODEL_FILE = "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
MODEL_PATH = os.path.join("models", MODEL_FILE)
CONTEXT_LENGTH = 2048  # 컨텍스트 길이 증가
MAX_NEW_TOKENS = 1024  # 생성 토큰 수 증가
N_THREADS = 8
N_BATCH = 256

# Instruct 모델 프롬프트 템플릿
INSTRUCT_TEMPLATE = """
<|system|>
{system_prompt}
<|user|>
{passage}
<|assistant|>"""

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

def get_device_info():
    if IS_APPLE_SILICON:
        return {"type": "Apple Silicon", "device": DEVICE, "memory": "Metal Performance Shaders (MPS)"}
    elif torch.cuda.is_available():
        return {
            "type": "NVIDIA GPU",
            "device": DEVICE,
            "name": torch.cuda.get_device_name(0),
            "memory": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB"
        }
    else:
        return {"type": "CPU", "device": DEVICE, "memory": "System RAM"}

def get_offload_path():
    global OFFLOAD_INDEX
    path = os.path.join(OFFLOAD_FOLDER, f"offload_{OFFLOAD_INDEX}")
    OFFLOAD_INDEX += 1
    os.makedirs(path, exist_ok=True)
    return path

def get_memory_usage() -> Dict[str, float]:
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return {
        "rss": memory_info.rss / 1024 / 1024,
        "vms": memory_info.vms / 1024 / 1024,
        "percent": process.memory_percent()
    }

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mem = get_memory_usage()
    logger.info(f"[메모리 사용량] RSS: {mem['rss']:.1f}MB, VMS: {mem['vms']:.1f}MB, Percent: {mem['percent']:.1f}%")

def truncate_text(text: str, max_length: int = 1500) -> str:
    if len(text) <= max_length:
        return text
    sentences = text.split('. ')
    truncated = ''
    for s in sentences:
        if len(truncated + s + '. ') <= max_length:
            truncated += s + '. '
        else:
            break
    return truncated.strip()

def load_model():
    global model, model_load_error
    try:
        with Timer("모델 로드"):
            os.environ["OMP_NUM_THREADS"] = "8"
            os.environ["MKL_NUM_THREADS"] = "8"
            model = Llama(
                model_path=MODEL_PATH,
                n_ctx=CONTEXT_LENGTH,
                n_batch=N_BATCH,
                n_threads=N_THREADS,
                n_gpu_layers=0,
                offload_kqv=True,
                verbose=False,
                seed=42,
                f16_kv=True,
                embedding=False,
                rope_scaling=None,
            )
    except Exception as e:
        model_load_error = str(e)
        logger.error(f"모델 로드 실패: {model_load_error}")
        raise
    return model

@app.get("/health")
async def health_check():
    if not model_loaded:
        return {"status": "error", "message": f"모델이 로드되지 않았습니다. 오류: {model_load_error}"}
    return {
        "status": "healthy",
        "model": "Meta-Llama-3-8B-Instruct GGUF (llama.cpp)",
        "device": f"CPU (스레드: {N_THREADS})",
        "memory_usage": get_memory_usage()
    }

@app.post("/generate")
async def generate_mcq(request: Request) -> Dict:
    """MCQ를 생성합니다."""
    try:
        data = await request.json()
        input_text = data.get("text", "").strip()
        
        if not input_text:
            raise HTTPException(status_code=400, detail="입력 텍스트가 필요합니다")
            
        logger.info(f"입력 텍스트: {input_text}")
        
        # 프롬프트 생성
        prompt = format_prompt(input_text)
        logger.info(f"생성된 프롬프트: {prompt}")
        
        # 텍스트 생성
        response = generate_text(prompt)
        logger.info(f"생성된 응답: {response}")
        
        # JSON 파싱
        result = json.loads(response)
        
        return result
        
    except HTTPException:
        raise
    except json.JSONDecodeError as e:
        logger.error(f"JSON 파싱 실패: {str(e)}")
        raise HTTPException(status_code=500, detail="응답 형식이 올바르지 않습니다")
    except Exception as e:
        logger.error(f"MCQ 생성 중 예외 발생: {str(e)}")
        logger.exception("상세 오류 정보:")
        raise HTTPException(status_code=500, detail="MCQ 생성 중 오류가 발생했습니다")

@app.get("/reload")
async def reload_server():
    if not DEV_MODE:
        raise HTTPException(status_code=403, detail="개발 모드에서만 가능합니다.")
    global should_reload
    should_reload = True
    return {"status": "reloading", "message": "서버 재시작이 예약되었습니다."}

@app.get("/status")
async def server_status():
    return {
        "status": "running",
        "dev_mode": DEV_MODE,
        "model_loaded": model_loaded,
        "should_reload": should_reload,
        "memory_usage": get_memory_usage()
    }

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if DEV_MODE:
        print(f"개발 모드: 자동 리로드 활성화됨 (감시 디렉토리: {current_dir})")
        uvicorn.run(
            "api_server:app",
            host=HOST,
            port=PORT,
            reload=True,
            reload_dirs=[current_dir],
            reload_delay=1.0,
            log_level="info"
        )
    else:
        print("프로덕션 모드: 자동 리로드 비활성화됨")
        uvicorn.run(
            "api_server:app",
            host=HOST,
            port=PORT,
            reload=False
        )