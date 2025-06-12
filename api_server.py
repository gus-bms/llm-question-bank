import os
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request
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
from watchfiles import watch
from pathlib import Path
from fastapi.middleware.gzip import GZipMiddleware


# 환경 변수 로드
load_dotenv()


# 서버 설정
DEV_MODE = os.getenv("DEV_MODE", "false").lower() == "true"  # 개발 모드 기본값을 false로 설정
PORT = int(os.getenv("API_PORT", "8000"))  # 기본값 8000
HOST = os.getenv("API_HOST", "0.0.0.0")    # 기본값 0.0.0.0


# Mac 환경 감지
IS_MAC = platform.system() == "Darwin"
IS_APPLE_SILICON = IS_MAC and platform.machine() == "arm64"


# 메모리 최적화 설정
MAX_MEMORY = 4 * 1024 * 1024 * 1024  # 4GB
TORCH_MEMORY_FRACTION = 0.3


# 강제로 CPU 모드 사용
USE_GPU = False
DEVICE = "cpu"
DEVICE_MAP = "cpu"
TORCH_DTYPE = torch.float32


# 오프로딩 설정
OFFLOAD_FOLDER = "offload"
OFFLOAD_INDEX = 0


# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'api_server_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


# 전역 변수 선언
global server
server = None  # uvicorn 서버 인스턴스
model_loaded = False
model_load_error = None
model = None
tokenizer = None
should_reload = False
watched_files = set()


# 시그널 핸들러
def signal_handler(signum, frame):
    """시그널 핸들러"""
    logger.info(f"시그널 수신: {signum}")
    sys.exit(0)


# 시그널 핸들러 등록
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
if hasattr(signal, "SIGBREAK"):
    signal.signal(signal.SIGBREAK, signal_handler)  # Windows에서 사용


# FastAPI 앱 생성
app = FastAPI(
    title="LLaMA-3 API Server",
    description="Meta-Llama-3-8B-GGUF 모델을 사용하는 텍스트 생성 API 서버",
    version="1.0.0"
)


# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 운영 환경에서는 특정 도메인만 허용하도록 수정
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Gzip 압축 미들웨어 추가
app.add_middleware(GZipMiddleware, minimum_size=1000)


# 요청/응답 모델 정의
class GenerationRequest(BaseModel):
    """텍스트 생성 요청 모델"""
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
MODEL_NAME = "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF"
MODEL_FILE = "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
MODEL_PATH = os.path.join("models", MODEL_FILE)
CONTEXT_LENGTH = 2048  # 컨텍스트 길이 증가
MAX_NEW_TOKENS = 1024  # 생성 토큰 수 증가
N_THREADS = 4  # CPU 스레드 수
N_BATCH = 512  # 배치 크기

# 시스템 프롬프트 설정
SYSTEM_PROMPT = """<|system|>
당신은 전문적인 영어 선생님입니다. 아래 예시를 참고하여, 사용자 입력 문장 하나로 다지선다 문제(MCQ)를 생성하세요. 반드시 지정된 예약어 태그와 형식을 정확히 지켜야 합니다.

1. 질문: <|question|>…</|question|>  
   - 내용: “(A), (B), (C)에서 문맥에 맞는 낱말로 가장 적절한 것을 고르세요.”

2. 지문: <|passage|>…</|passage|>  
   - 원문 전체를 사용하되, 정확히 세 군데에만 (A), (B), (C) 빈칸을 삽입  
   - 각 빈칸은 반드시 다음 형식:  
     `(A) {{|<bold-underline>[옵션1 / 옵션2]</bold-underline>|/}}`  
     `(B) {{|<bold-underline>[옵션1 / 옵션2]</bold-underline>|/}}`  
     `(C) {{|<bold-underline>[옵션1 / 옵션2]</bold-underline>|/}}`

3. 선택지: <|options|>…</|options|>  
   - 총 5개 옵션  
   - 각 옵션은 `(A),(B),(C)`에 들어갈 단어 세 개를 쉼표로 구분해 나열  
   - 옵션 간 구분자는 `//`  
   - 예시: `symbolizes,requires,inspectors//synchronizes,requires,spectators//…`

4. 정답: <|answer|>…</|answer|>  
   - 1부터 5 사이의 정답 번호만 출력

5. 해설: <|explanation|>…</|explanation|>  
   - 한 줄로 요약 설명

예시 입력:  
You would be confused momentarily, but laugh when you learned that the term "robot" also means "traffic light" in South Africa.

예시 출력:
<|question|>  
(A), (B), (C)에서 문맥에 맞는 낱말로 가장 적절한 것을 고르세요.  
</|question|>

<|passage|>  
You would be confused momentarily, but laugh when you learned that the term "robot" also means "traffic light" in South Africa. It (A) {{|<bold-underline>[symbolizes / synchronizes]</bold-underline>|/}} a unique usage that (B) {{|<bold-underline>[inquires / requires]</bold-underline>|/}} further context, and many (C) {{|<bold-underline>[inspectors / spectators]</bold-underline>|/}} find it amusing.  
</|passage|>

<|options|>  
symbolizes,inquires,inspectors//synchronizes,inquires,spectators//symbolizes,requires,spectators//synchronizes,requires,spectators//symbolizes,requires,inspectors  
</|options|>

<|answer|>  
3  
</|answer|>

<|explanation|>  
‘symbolizes’는 “상징하다”로, ‘requires’는 “필요로 한다”로, ‘spectators’는 “구경꾼”으로 문맥에 가장 적절합니다.  
</|explanation|>
</|system|>
"""

# Instruct 모델 프롬프트 템플릿
INSTRUCT_TEMPLATE = """{system_prompt}

<|user|>
다음 지문을 바탕으로 영어 문제를 구성해주세요. 반드시 한국어로 작성해주세요.

지문:
{input}
<|assistant|>"""


class Timer:
    """작업 시간 측정을 위한 컨텍스트 매니저"""
    def __init__(self, name: str):
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"[{self.name}] 시작")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        logger.info(f"[{self.name}] 완료 - 소요시간: {duration:.2f}초")


def get_device_info():
    """디바이스 정보 반환"""
    if IS_APPLE_SILICON:
        return {
            "type": "Apple Silicon",
            "device": DEVICE,
            "memory": "Metal Performance Shaders (MPS)"
        }
    elif torch.cuda.is_available():
        return {
            "type": "NVIDIA GPU",
            "device": DEVICE,
            "name": torch.cuda.get_device_name(0),
            "memory": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB"
        }
    else:
        return {
            "type": "CPU",
            "device": DEVICE,
            "memory": "System RAM"
        }


def get_offload_path():
    """오프로딩 경로 생성"""
    global OFFLOAD_INDEX
    path = os.path.join(OFFLOAD_FOLDER, f"offload_{OFFLOAD_INDEX}")
    OFFLOAD_INDEX += 1
    os.makedirs(path, exist_ok=True)
    return path


def get_memory_usage() -> Dict[str, float]:
    """메모리 사용량 확인"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return {
        "rss": memory_info.rss / 1024 / 1024,  # MB
        "vms": memory_info.vms / 1024 / 1024,  # MB
        "percent": process.memory_percent()
    }


def clear_memory():
    """메모리 정리"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    memory_usage = get_memory_usage()
    logger.info(f"[메모리 사용량] RSS: {memory_usage['rss']:.1f}MB, VMS: {memory_usage['vms']:.1f}MB, Percent: {memory_usage['percent']:.1f}%")


def truncate_text(text: str, max_length: int = 1500) -> str:
    """텍스트를 최대 길이에 맞게 자르는 함수"""
    if len(text) <= max_length:
        return text
    
    # 문장 단위로 자르기
    sentences = text.split('. ')
    truncated_text = ''
    for sentence in sentences:
        if len(truncated_text + sentence + '. ') <= max_length:
            truncated_text += sentence + '. '
        else:
            break
    
    return truncated_text.strip()


def load_model():
    """모델 로드"""
    global model, model_loaded, model_load_error
    
    try:
        logger.info("모델 로드 시작...")
        with Timer("모델 로드"):
            # 메모리 정리
            clear_memory()
            
            # 모델 로드
            model = Llama(
                model_path=MODEL_PATH,
                n_ctx=CONTEXT_LENGTH,  # 컨텍스트 길이 설정
                n_batch=512,  # 배치 크기
                n_threads=4,  # CPU 스레드 수
                n_gpu_layers=0  # CPU 모드
            )
            
            model_loaded = True
            model_load_error = None
            logger.info(f"모델 로드 완료 (컨텍스트 길이: {CONTEXT_LENGTH})")
            
    except Exception as e:
        model_loaded = False
        model_load_error = str(e)
        logger.error(f"모델 로드 실패: {str(e)}")
        raise


@app.get("/health")
async def health_check():
    """서버 상태 확인"""
    if not model_loaded:
        return {
            "status": "error",
            "message": f"모델이 로드되지 않았습니다. 오류: {model_load_error}"
        }
    return {
        "status": "healthy",
        "model": "Meta-Llama-3-8B-Instruct GGUF (llama.cpp)",
        "device": f"CPU (스레드: {N_THREADS})",
        "memory_usage": get_memory_usage()
    }


@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    """텍스트 생성"""
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail=f"모델이 로드되지 않았습니다. 오류: {model_load_error}"
        )
    
    try:
        request_start_time = time.time()
        logger.info(f"[요청 시작] 프롬프트 길이: {len(request.prompt)}")
        
        with Timer("메모리 정리"):
            clear_memory()
        
        with Timer("프롬프트 처리"):
            # 입력 텍스트 길이 제한
            truncated_prompt = truncate_text(request.prompt)
            if len(truncated_prompt) < len(request.prompt):
                logger.warning(f"입력 텍스트가 {len(request.prompt)}에서 {len(truncated_prompt)}로 잘렸습니다.")
            
            # 시스템 프롬프트와 사용자 입력을 포함한 프롬프트 생성
            formatted_prompt = INSTRUCT_TEMPLATE.format(
                system_prompt=SYSTEM_PROMPT,
                input=f"다음 지문을 바탕으로 영어 문제를 구성해주세요. 반드시 한국어로 작성해주세요.\n\n지문: {truncated_prompt}"
            )
            logger.info(f"[프롬프트] 포맷팅된 프롬프트 길이: {len(formatted_prompt)}")
        
        with Timer("텍스트 생성"):
            # llama.cpp의 create_completion 메서드 사용
            response = model.create_completion(
                prompt=formatted_prompt,
                max_tokens=min(request.max_length, MAX_NEW_TOKENS),  # 최대 토큰 수 제한
                temperature=0.7,
                top_p=0.9,
                stop=["<|user|>", "<|system|>"],  # 영어 시작 단어 제거
                echo=False,
                stream=False,
                repeat_penalty=1.2,
                presence_penalty=0.1
            )
            generated_text = response["choices"][0]["text"]
            logger.info(f"[텍스트 생성] 생성된 텍스트 길이: {len(generated_text)}")
        
        with Timer("응답 처리"):
            # Instruct 모델 응답에서 assistant 부분만 추출
            if "<|assistant|>" in generated_text:
                generated_text = generated_text.split("<|assistant|>")[-1].strip()
            
            # 빈 응답 처리
            if not generated_text:
                generated_text = "죄송합니다. 응답을 생성하지 못했습니다. 다시 시도해주세요."
                logger.warning("빈 응답이 생성되었습니다.")
            
            logger.info(f"[응답 처리] 최종 텍스트 길이: {len(generated_text)}")
        
        with Timer("최종 메모리 정리"):
            clear_memory()
        
        total_time = time.time() - request_start_time
        logger.info(f"[요청 완료] 전체 응답 시간: {total_time:.2f}초")
        return GenerationResponse(generated_text=generated_text)
        
    except Exception as e:
        logger.error(f"[오류 발생] 텍스트 생성 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 초기화"""
    try:
        logger.info("서버 시작 중...")
        load_model()
        if DEV_MODE:
            # 개발 모드에서 파일 변경 감지
            async def file_change_handler(changes: List[Path]):
                global should_reload
                for change in changes:
                    if change.suffix in ['.py', '.env']:
                        logger.info(f"파일 변경 감지: {change}")
                        should_reload = True
                        break
            
            asyncio.create_task(watch("api_server.py", file_change_handler))
        logger.info("서버 시작 완료")
    except Exception as e:
        logger.error(f"서버 시작 중 오류 발생: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 정리"""
    logger.info("서버 종료 중...")
    if model_loaded:
        try:
            del model
            gc.collect()
            logger.info("모델 메모리 해제 완료")
        except Exception as e:
            logger.error(f"모델 메모리 해제 중 오류: {str(e)}")
    logger.info("서버 종료 완료")


@app.get("/reload")
async def reload_server():
    """서버 수동 재시작"""
    if not DEV_MODE:
        raise HTTPException(
            status_code=403,
            detail="개발 모드에서만 서버 재시작이 가능합니다."
        )
    
    global should_reload
    should_reload = True
    return {"status": "reloading", "message": "서버 재시작이 예약되었습니다."}


@app.get("/status")
async def server_status():
    """서버 상태 확인"""
    return {
        "status": "running",
        "dev_mode": DEV_MODE,
        "model_loaded": model_loaded,
        "watched_files": list(watched_files),
        "should_reload": should_reload,
        "memory_usage": get_memory_usage()
    }


if __name__ == "__main__":
    try:
        uvicorn.run(
            "api_server:app",
            host=HOST,
            port=PORT,
            reload=False  # 개발 모드 비활성화
        )
    except Exception as e:
        logger.error(f"서버 실행 중 오류 발생: {str(e)}")
        sys.exit(1) 