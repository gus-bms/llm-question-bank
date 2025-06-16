import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import logging
from dotenv import load_dotenv

# .env 파일에서 환경변수 로딩
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_model():
    """HyperCLOVAX-SEED-Vision-Instruct-1.5B 모델을 다운로드합니다."""
    model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
    save_dir = "models/hyperclovax-1_5b"
    
    # Hugging Face Hub 토큰 가져오기
    token = os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not token:
        logger.error("❌ 환경 변수 'HUGGING_FACE_HUB_TOKEN'를 찾을 수 없습니다.")
        return None
    
    try:
        # 저장 디렉토리 생성
        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"모델 저장 경로: {save_dir}")
        
        # 토크나이저 다운로드
        logger.info("토크나이저 다운로드 중...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        tokenizer.save_pretrained(save_dir)
        logger.info("토크나이저 다운로드 완료")
        
        # 모델 다운로드
        logger.info("모델 다운로드 중...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=token,
            torch_dtype=torch.float16,  # FP16 사용
            device_map="auto",  # 자동으로 적절한 디바이스에 할당
            low_cpu_mem_usage=True
        )
        
        # 모델 저장
        logger.info("모델 저장 중...")
        model.save_pretrained(save_dir)
        logger.info("모델 저장 완료")
        
        # 메모리 정리
        del model
        torch.cuda.empty_cache()
        
        logger.info("모델 다운로드 및 저장이 완료되었습니다.")
        logger.info(f"모델 경로: {os.path.abspath(save_dir)}")
        
    except Exception as e:
        logger.error(f"모델 다운로드 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    download_model() 