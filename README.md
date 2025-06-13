# Llama 모델 파인튜닝 프로젝트

이 프로젝트는 Meta의 Llama 모델을 파인튜닝하고 API 서버로 제공하기 위한 코드를 포함하고 있습니다.

## 설치 및 설정 방법

### 1. Python 3.11 설치

- [Python 3.11.0 다운로드 페이지](https://www.python.org/downloads/release/python-3110/)에서 Windows installer를 다운로드
- 설치 시 "Add Python 3.11 to PATH" 옵션을 반드시 체크
- 설치 완료 후 PowerShell 재시작

### 2. 가상환경 설정

```powershell
# 가상환경 생성
python -m venv .venv-py311

# 가상환경 활성화 (Windows PowerShell)
.\.venv-py311\Scripts\Activate.ps1
```

### 3. 필요한 패키지 설치

```powershell
# pip 업그레이드
python -m pip install --upgrade pip

# 패키지 설치
pip install -r requirements.txt
```

### 4. 모델 다운로드

```powershell
# 모델 다운로드 스크립트 실행
python download_model.py
```

### 5. 환경 변수 설정

`.env` 파일을 프로젝트 루트 디렉토리에 생성하고 다음 내용을 추가:

```env
DEV_MODE=false
API_PORT=8000
API_HOST=0.0.0.0
```

### 6. 서버 실행

```powershell
# API 서버 실행
python api_server.py
```

## 시스템 요구사항

- Python 3.11
- 최소 8GB RAM
- Windows 10/11
- 인터넷 연결 (모델 다운로드용)

## 주의사항

- 모델 파일 크기가 약 4GB이므로 충분한 디스크 공간 필요
- CPU 모드로 실행되며, GPU는 현재 지원되지 않음
- 서버 실행 시 기본적으로 8000번 포트 사용

## 환경 설정

1. 가상환경 생성 및 활성화:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
```

2. 필요한 패키지 설치:

```bash
pip install -r requirements.txt
```

3. Hugging Face 토큰 설정:

- Hugging Face 계정 생성
- https://huggingface.co/settings/tokens 에서 토큰 생성
- `.env` 파일에 토큰 설정:

```bash
HUGGING_FACE_HUB_TOKEN=your_token_here
```

4. Weights & Biases 설정:

- https://wandb.ai 에서 계정 생성
- 로그인:

```bash
wandb login
```

## 데이터셋 준비

1. 데이터셋은 다음 형식을 따라야 합니다:

```json
{
  "text": "학습할 텍스트 데이터"
}
```

2. `finetune_llama.py` 파일에서 `DATASET_NAME` 변수를 실제 데이터셋 이름으로 변경하세요.

## 파인튜닝 실행

```bash
python finetune_llama.py
```

## API 서버 실행

1. 서버 시작:

```bash
python api_server.py
```

2. API 엔드포인트:

- 텍스트 생성: `POST /generate`
  ```json
  {
    "prompt": "생성할 텍스트의 프롬프트",
    "max_length": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "num_return_sequences": 1
  }
  ```
- 서버 상태 확인: `GET /health`

3. API 문서:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 주요 설정 매개변수

### 파인튜닝 설정

- `LORA_R`: LoRA 랭크 (기본값: 16)
- `LORA_ALPHA`: LoRA 알파 값 (기본값: 32)
- `LORA_DROPOUT`: LoRA 드롭아웃 비율 (기본값: 0.05)
- `LEARNING_RATE`: 학습률 (기본값: 2e-4)
- `BATCH_SIZE`: 배치 크기 (기본값: 4)
- `NUM_EPOCHS`: 학습 에포크 수 (기본값: 3)

### API 서버 설정

- `HOST`: 서버 호스트 (기본값: 0.0.0.0)
- `PORT`: 서버 포트 (기본값: 8000)
- `MAX_LENGTH`: 최대 생성 길이 (기본값: 512)
- `TEMPERATURE`: 생성 다양성 (기본값: 0.7)
- `TOP_P`: 누적 확률 임계값 (기본값: 0.9)

## 주의사항

1. GPU 메모리 요구사항:

   - 최소 16GB VRAM 권장
   - 8비트 양자화를 사용하여 메모리 사용량 최적화

2. 데이터셋 크기:

   - 학습 데이터는 충분히 큰 크기여야 함 (최소 수천 개의 샘플)
   - 검증 데이터셋도 포함되어야 함

3. 모델 접근:

   - Meta Llama 모델 사용을 위해서는 Meta의 승인이 필요합니다
   - https://ai.meta.com/llama/ 에서 접근 권한을 요청하세요

4. API 서버 보안:
   - 실제 운영 환경에서는 CORS 설정을 적절히 조정하세요
   - API 인증 메커니즘을 추가하는 것을 고려하세요
   - HTTPS를 사용하여 통신을 암호화하세요
