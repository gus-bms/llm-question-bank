from huggingface_hub import hf_hub_download
import os

def download_model():
    """Meta-Llama-3-8B-Instruct GGUF 모델 다운로드"""
    print("모델 다운로드를 시작합니다...")
    
    # 모델 정보
    repo_id = "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF"
    filename = "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
    local_dir = "models"
    
    try:
        # 모델 다운로드
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        print(f"모델이 성공적으로 다운로드되었습니다: {model_path}")
        return model_path
    except Exception as e:
        print(f"모델 다운로드 중 오류 발생: {str(e)}")
        return None

if __name__ == "__main__":
    download_model() 