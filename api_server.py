import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import logging
from dotenv import load_dotenv
import uvicorn

# 환경 변수 로드
load_dotenv()

# 서버 설정
DEV_MODE = os.getenv("DEV_MODE", "false").lower() == "true"
PORT = int(os.getenv("API_PORT", "8000"))  # 기본값 8000
HOST = os.getenv("API_HOST", "0.0.0.0")    # 기본값 0.0.0.0

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="Llama-3 API",
    description="Meta-Llama-3-8B 모델을 사용한 텍스트 생성 API",
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

# 요청/응답 모델 정의
class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = 256  # 기본값을 256으로 줄임
    temperature: float = 0.7
    top_p: float = 0.9
    num_return_sequences: int = 1
    # 성능 최적화를 위한 추가 옵션
    do_sample: bool = True  # 샘플링 사용 여부
    num_beams: int = 1      # 빔 서치 크기 (1이면 그리디 디코딩)
    early_stopping: bool = True  # 조기 종료 사용

class GenerationResponse(BaseModel):
    generated_text: str

# 전역 변수로 모델과 토크나이저 저장
model = None
tokenizer = None

# 모델 설정
MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
FINETUNED_MODEL_PATH = "output/finetuned_llama3"

# 모델 로드 상태 추적
model_loaded = False
model_load_error = None

def load_model():
    """모델과 토크나이저를 로드하는 함수"""
    global model, tokenizer, model_loaded, model_load_error
    
    try:
        # 이미 로드된 모델이 있다면 해제
        if model is not None:
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # CPU 설정
        device = "cpu"
        logger.info(f"Using device: {device}")
        
        # 토크나이저 로드 (최적화)
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            use_fast=True  # 빠른 토크나이저 사용
        )
        
        # 패딩 토큰 설정
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("패딩 토큰을 EOS 토큰으로 설정했습니다.")
        
        # 모델 로드 (CPU 설정, 최적화)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,  # CPU 메모리 사용량 최적화
            offload_folder="offload"  # 메모리 부족 시 오프로드 폴더 사용
        )
        
        # 모델의 패딩 토큰 ID 설정
        model.config.pad_token_id = tokenizer.pad_token_id
        
        # 파인튜닝된 모델이 있다면 로드
        if os.path.exists(FINETUNED_MODEL_PATH):
            logger.info("파인튜닝된 모델을 로드합니다...")
            model = AutoModelForCausalLM.from_pretrained(
                FINETUNED_MODEL_PATH,
                torch_dtype=torch.float32,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                offload_folder="offload"
            )
            model.config.pad_token_id = tokenizer.pad_token_id
        
        # 모델을 평가 모드로 설정
        model.eval()
        
        model_loaded = True
        model_load_error = None
        logger.info("모델 로드 완료")
        
    except Exception as e:
        model_loaded = False
        model_load_error = str(e)
        logger.error(f"모델 로드 중 오류 발생: {model_load_error}")
        raise

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 로드"""
    load_model()

@app.get("/health")
async def health_check():
    """서버 상태 확인 엔드포인트"""
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "model_name": MODEL_NAME,
        "dev_mode": DEV_MODE,
        "error": model_load_error if not model_loaded else None
    }

@app.post("/reload")
async def reload_model():
    """모델 재로드"""
    if not DEV_MODE:
        raise HTTPException(
            status_code=403,
            detail="개발 모드에서만 모델 재로드가 가능합니다."
        )
    
    try:
        load_model()
        return {
            "status": "success",
            "message": "모델이 성공적으로 재로드되었습니다."
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"모델 재로드 중 오류 발생: {str(e)}"
        )

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    """텍스트 생성"""
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail=f"모델이 로드되지 않았습니다. 오류: {model_load_error}"
        )
    
    try:
        # 입력 텍스트 토크나이징 (최적화)
        inputs = tokenizer(
            request.prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=min(request.max_length, 512)  # 최대 512로 제한
        )
        
        # 텍스트 생성 (최적화된 설정)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p,
                num_return_sequences=request.num_return_sequences,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2,
                do_sample=request.do_sample,
                num_beams=request.num_beams,
                early_stopping=request.early_stopping,
                # 추가 최적화 옵션
                use_cache=True,  # 캐시 사용으로 속도 향상
                no_repeat_ngram_size=3,  # 반복 방지
                length_penalty=1.0,  # 길이 페널티
                min_length=10  # 최소 생성 길이
            )
        
        # 생성된 텍스트 디코딩
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return GenerationResponse(generated_text=generated_text)
        
    except Exception as e:
        logger.error(f"텍스트 생성 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    try:
        # 개발 모드 설정에 따라 서버 실행
        logger.info(f"서버를 시작합니다. (호스트: {HOST}, 포트: {PORT}, 개발 모드: {DEV_MODE})")
        uvicorn.run(
            "api_server:app",
            host=HOST,
            port=PORT,
            reload=DEV_MODE,  # 개발 모드에서만 자동 리로드 활성화
            reload_dirs=["python"],  # 감시할 디렉토리
            workers=1  # 단일 워커 사용
        )
    except OSError as e:
        if e.winerror == 10013:  # Windows 포트 접근 권한 오류
            logger.error(f"포트 {PORT}에 접근할 수 없습니다. 다음 중 하나를 시도해보세요:")
            logger.error("1. 다른 포트 번호를 사용하세요 (예: API_PORT=8001)")
            logger.error("2. 이미 실행 중인 서버를 종료하세요")
            logger.error("3. 관리자 권한으로 실행하세요")
        else:
            logger.error(f"서버 시작 중 오류 발생: {str(e)}")
        raise 