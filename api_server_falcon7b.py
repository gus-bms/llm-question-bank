import os
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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

# 환경 변수 로드
load_dotenv()

# CPU 최적화
num_cpus = psutil.cpu_count(logical=True)
torch.set_num_threads(num_cpus)
os.environ["OMP_NUM_THREADS"] = str(num_cpus)
os.environ["MKL_NUM_THREADS"] = str(num_cpus)
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# 서버 설정
DEV_MODE = False
PORT = int(os.getenv("API_PORT", "8003"))  # Falcon용 포트
HOST = os.getenv("API_HOST", "0.0.0.0")

# Mac 환경 감지
IS_MAC = platform.system() == "Darwin"
IS_APPLE_SILICON = IS_MAC and platform.machine() == "arm64"

# 메모리 최적화 설정
MAX_MEMORY = 15 * 1024 * 1024 * 1024  # 15GB VRAM
TORCH_MEMORY_FRACTION = 0.95

# GPU 설정
USE_GPU = True
DEVICE = "cuda"
DEVICE_MAP = "auto"
TORCH_DTYPE = torch.float16

# 로깅 설정
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# watchfiles 로거 비활성화
for logger_name in ['watchfiles.main', 'watchfiles.watcher', 'watchfiles.filters']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)
    logging.getLogger(logger_name).propagate = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{log_dir}/api_server_falcon7b_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 전역 변수
model = None
tokenizer = None
model_lock = asyncio.Lock()
file_watcher_task: Optional[asyncio.Task] = None
model_load_error: Optional[str] = None
model_loaded = False
should_reload = False
last_reload_time = None

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
    return f"User: {system_prompt}\n\n{input_text}\n\nAssistant:"

def validate_json_response(response: str) -> bool:
    """JSON 응답의 유효성을 검증합니다."""
    try:
        # JSON 파싱 전에 응답 정리
        response = response.strip()
        if not response.startswith('{') or not response.endswith('}'):
            logger.warning("응답이 올바른 JSON 형식이 아님")
            logger.warning(f"원본 응답: {response}")
            return False

        # 잘못된 형식 검사
        if "_json_" in response or "__metadata" in response or "@odata" in response:
            logger.warning("잘못된 JSON 형식 (메타데이터 포함)")
            logger.warning(f"원본 응답: {response}")
            return False

        data = json.loads(response)
        required_keys = ["question", "passage", "options", "answer", "explanation"]
        
        # 필수 키 확인
        if not all(key in data for key in required_keys):
            missing_keys = [key for key in required_keys if key not in data]
            logger.warning(f"응답에 필수 키가 누락됨: {missing_keys}")
            logger.warning(f"원본 응답: {response}")
            return False

        # 추가 필드 확인
        extra_keys = [key for key in data.keys() if key not in required_keys]
        if extra_keys:
            logger.warning(f"불필요한 필드가 포함됨: {extra_keys}")
            logger.warning(f"원본 응답: {response}")
            return False
            
        # 빈칸 개수 확인
        blanks = ["(A){{|bold-underline|}}", "(B){{|bold-underline|}}", "(C){{|bold-underline|}}"]
        blank_count = sum(1 for blank in blanks if blank in data["passage"])
        if blank_count != 3:  # 정확히 3개의 빈칸이 있어야 함
            logger.warning(f"빈칸 개수가 정확하지 않음: {blank_count} (필요: 3)")
            logger.warning(f"원본 응답: {response}")
            return False
            
        # 옵션 형식 확인
        options = data["options"].split("//")
        if len(options) != 5:  # 정확히 5개의 옵션이 있어야 함
            logger.warning(f"옵션 개수가 정확하지 않음: {len(options)} (필요: 5)")
            logger.warning(f"원본 응답: {response}")
            return False
            
        # 각 옵션의 형식 확인
        for i, option in enumerate(options):
            phrases = option.split(" - ")
            if len(phrases) != 3:  # 각 옵션은 정확히 3개의 구문이 있어야 함
                logger.warning(f"옵션 {i+1}의 구문 개수가 정확하지 않음: {len(phrases)} (필요: 3)")
                logger.warning(f"원본 응답: {response}")
                return False
                
        return True
    except json.JSONDecodeError:
        logger.warning("JSON 파싱 실패")
        logger.warning(f"원본 응답: {response}")
        return False
    except Exception as e:
        logger.warning(f"응답 검증 중 예외 발생: {str(e)}")
        logger.warning(f"원본 응답: {response}")
        return False

def generate_text(prompt: str, max_retries: int = 3) -> str:
    """텍스트를 생성하고 유효성을 검증합니다."""
    global model, tokenizer
    
    # 모델과 토크나이저 상태 확인
    if model is None or tokenizer is None:
        logger.error("모델 또는 토크나이저가 초기화되지 않았습니다.")
        try:
            logger.info("모델을 다시 로드합니다...")
            model, tokenizer = load_model()
            if model is None or tokenizer is None:
                raise Exception("모델 로드 실패")
            
            # 토크나이저가 제대로 작동하는지 테스트
            test_text = "Hello, world!"
            test_tokens = tokenizer(test_text, return_tensors="pt")
            if test_tokens is None:
                raise Exception("토크나이저 테스트 실패")
            logger.info("토크나이저 테스트 성공")
            
        except Exception as e:
            logger.error(f"모델 재로드 실패: {str(e)}")
            logger.exception("상세 오류 정보:")
            raise Exception("모델 초기화 실패")
    
    for attempt in range(max_retries):
        try:
            logger.info(f"생성 시도 {attempt + 1}/{max_retries}")
            logger.info(f"프롬프트 길이  --: {len(prompt)}")
            
            # 토크나이저가 제대로 작동하는지 한 번 더 확인
            if not callable(tokenizer):
                raise Exception("토크나이저가 호출 가능한 객체가 아닙니다")
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}  # GPU로 입력 이동
            
            with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.3,
                    top_p=0.95,
                    repetition_penalty=1.1,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                    num_beams=5,
                    no_repeat_ngram_size=3
                )
            
            response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response_text = response_text[len(prompt):].strip()
            
            # 응답 검증 및 재시도 결정
            should_retry, error = should_retry_response(response_text, attempt, max_retries)
            if should_retry:
                logger.warning(error)
                continue
                
            # 원본 응답 로깅
            logger.info(f"모델 원본 출력: {response_text}")
            
            # JSON 추출
            json_text, error = extract_json_from_response(response_text)
            if error:
                logger.warning(error)
                if attempt == max_retries - 1:
                    return response_text
                continue
                
            logger.info(f"추출된 JSON: {json_text}")
            return json_text
                
        except Exception as e:
            logger.error(f"생성 중 예외 발생 (시도 {attempt + 1}/{max_retries}): {str(e)}")
            logger.exception("상세 오류 정보:")
            
    raise Exception("최대 재시도 횟수 초과")

# FastAPI 앱 생성
app = FastAPI()

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
MODEL_PATH = "/mnt/storage/models/falcon-7b-instruct"

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

def load_model():
    """모델을 로드합니다."""
    global model, tokenizer
    try:
        with Timer("모델 로드"):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.set_per_process_memory_fraction(TORCH_MEMORY_FRACTION)
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.enabled = True
            
            logger.info(f"토크나이저 로드 시작 (경로: {MODEL_PATH})")
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    MODEL_PATH,
                    trust_remote_code=True,
                    use_fast=True,
                    padding_side="left",
                    truncation_side="left",
                    local_files_only=True,
                    model_max_length=2048
                )
                if tokenizer is None:
                    raise Exception("토크나이저 로드 실패")
                logger.info("토크나이저 로드 완료")
                logger.info(f"토크나이저 클래스: {type(tokenizer).__name__}")
                logger.info(f"토크나이저 vocab 크기: {len(tokenizer)}")
                
                test_text = "Hello, world!"
                test_tokens = tokenizer(test_text, return_tensors="pt")
                if test_tokens is None:
                    raise Exception("토크나이저 테스트 실패")
                logger.info("토크나이저 테스트 성공")
                
            except Exception as e:
                logger.error(f"토크나이저 로드 중 오류 발생: {str(e)}")
                logger.exception("상세 오류 정보:")
                raise
            
            logger.info("모델 로드 시작...")
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_PATH,
                    torch_dtype=TORCH_DTYPE,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    max_memory={
                        0: f"{int(MAX_MEMORY * TORCH_MEMORY_FRACTION / 1024**3)}GB",
                        "cpu": f"{int(MAX_MEMORY * 0.8 / 1024**3)}GB"
                    },
                    offload_folder="offload",
                    offload_state_dict=True,
                    offload_buffers=True,
                    trust_remote_code=True,
                    use_cache=True,
                    attn_implementation="eager",
                    local_files_only=True,
                    use_safetensors=True
                )
                if model is None:
                    raise Exception("모델 로드 실패")
                logger.info("모델 로드 완료")
                
                if hasattr(model, 'device'):
                    logger.info(f"모델이 {model.device}에 로드되었습니다.")
                else:
                    logger.warning("모델 디바이스 정보를 확인할 수 없습니다.")
                
                with torch.no_grad():
                    test_tokens = {k: v.to(DEVICE) for k, v in test_tokens.items()}
                    test_output = model.generate(
                        **test_tokens,
                        max_new_tokens=10,
                        do_sample=False
                    )
                if test_output is None:
                    raise Exception("모델 테스트 실패")
                logger.info("모델 테스트 성공")
                
            except Exception as e:
                logger.error(f"모델 로드 중 오류 발생: {str(e)}")
                logger.exception("상세 오류 정보:")
                raise
            
            return model, tokenizer
    except Exception as e:
        logger.error(f"모델 로드 중 오류 발생: {str(e)}")
        logger.exception("상세 오류 정보:")
        raise

@app.on_event("startup")
async def startup_event():
    global model, tokenizer, file_watcher_task
    
    try:
        logger.info("서버 시작 중...")
        
        logger.info(f"PyTorch 버전: {torch.__version__}")
        logger.info(f"Transformers 버전: {transformers.__version__}")
        logger.info(f"CUDA 사용 가능: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA 버전: {torch.version.cuda}")
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        
        logger.info("Falcon-7B-Instruct 모델을 로드합니다...")
        model, tokenizer = load_model()
        if model is None or tokenizer is None:
            raise Exception("모델 또는 토크나이저 로드 실패")
            
        logger.info("Falcon-7B-Instruct 모델이 성공적으로 로드되었습니다.")
            
    except Exception as e:
        model_load_error = str(e)
        logger.error(f"서버 시작 실패: {str(e)}")
        logger.exception("상세 오류 정보:")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    global file_watcher_task
    if file_watcher_task:
        file_watcher_task.cancel()
        try:
            await file_watcher_task
        except asyncio.CancelledError:
            pass
    logger.info("서버가 종료되었습니다.")

@app.get("/health")
async def health_check():
    """서버 상태 확인"""
    global model, tokenizer, model_load_error
    
    if model is None or tokenizer is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "message": "모델이 로드되지 않았습니다.",
                "error": model_load_error
            }
        )
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_type": "falcon-7b-instruct",
        "device": str(model.device) if hasattr(model, 'device') else "unknown"
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
        
        prompt = format_prompt(input_text)
        logger.info(f"생성된 프롬프트: {prompt}")
        
        response = generate_text(prompt)
        logger.info(f"생성된 응답: {response}")
        
        # 응답이 올바른 MCQ 형식인지 검증
        if validate_json_response(response):
            try:
                result = json.loads(response)
                return {
                    "type": "mcq",
                    "data": result
                }
            except json.JSONDecodeError as e:
                logger.error(f"JSON 파싱 실패: {str(e)}")
                raise HTTPException(status_code=500, detail="응답 형식이 올바르지 않습니다")
        else:
            # 일반 대화형 응답인 경우
            return {
                "type": "chat",
                "data": {
                    "message": response
                }
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MCQ 생성 중 예외 발생: {str(e)}")
        logger.exception("상세 오류 정보:")
        raise HTTPException(status_code=500, detail="MCQ 생성 중 오류가 발생했습니다")

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
            "api_server_falcon7b:app",
            host=HOST,
            port=PORT,
            reload=True,
            reload_dirs=[current_dir],
            reload_delay=2.0,
            log_level="info",
            reload_excludes=["logs/*"],
            log_config=None
        )
    else:
        print("프로덕션 모드: 자동 리로드 비활성화됨")
        uvicorn.run(
            "api_server_falcon7b:app",
            host=HOST,
            port=PORT,
            reload=False
        ) 