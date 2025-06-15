from huggingface_hub import snapshot_download
from dotenv import load_dotenv
import os

def download_model():
    """Falcon-7B-Instruct 모델 다운로드"""
    print("📦 모델 다운로드를 시작합니다...")

    # .env 파일에서 환경변수 로딩
    load_dotenv()
    token = os.getenv("HUGGING_FACE_HUB_TOKEN")

    if not token:
        print("❌ 환경 변수 'HUGGING_FACE_HUB_TOKEN'를 찾을 수 없습니다.")
        return None

    repo_id = "tiiuae/falcon-7b-instruct"
    local_dir = "/mnt/storage/models/falcon-7b-instruct"

    try:
        model_dir = snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            token=token
        )
        print(f"✅ 모델이 성공적으로 다운로드되었습니다: {model_dir}")
        return model_dir
    except Exception as e:
        print(f"❌ 모델 다운로드 중 오류 발생: {str(e)}")
        return None

if __name__ == "__main__":
    download_model()

# from huggingface_hub import hf_hub_download
# import os

# def download_model():
#     """Meta-Llama-3-8B-Instruct GGUF 모델 다운로드"""
#     print("모델 다운로드를 시작합니다...")
    
#     # 모델 정보
#     repo_id = "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF"
#     filename = "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
#     local_dir = "models"
    
#     try:
#         # 모델 다운로드
#         model_path = hf_hub_download(
#             repo_id=repo_id,
#             filename=filename,
#             local_dir=local_dir,
#             local_dir_use_symlinks=False
#         )
#         print(f"모델이 성공적으로 다운로드되었습니다: {model_path}")
#         return model_path
#     except Exception as e:
#         print(f"모델 다운로드 중 오류 발생: {str(e)}")
#         return None

# if __name__ == "__main__":
#     download_model() 