# HyperCLOVAX API Server (`api_server_hyper.py`)

이 서버는 HyperCLOVAX 모델을 활용하여 텍스트 생성 및 JSON 기반 문제 생성을 위한 API를 제공합니다.

## 주요 기능

- HyperCLOVAX 모델 로드 및 GPU/CPU 자동 지원
- `/generate` 엔드포인트를 통한 텍스트/문제 생성
- 요청마다 커스텀 system prompt 지정 가능
- JSON 응답 자동 추출 및 재시도 로직 내장
- 로그 파일 자동 저장

## 환경 변수

`.env` 파일 또는 시스템 환경 변수로 아래 값을 설정할 수 있습니다.

| 변수명   | 설명        | 기본값  |
| -------- | ----------- | ------- |
| API_PORT | 서버 포트   | 8004    |
| API_HOST | 서버 호스트 | 0.0.0.0 |

## 실행 방법

```bash
pip install -r requirements.txt
python api_server_hyper.py
```

## 주요 엔드포인트

### 1. 헬스 체크

- **GET** `/health`
- **응답 예시**:
  ```json
  {
    "status": "healthy",
    "model_loaded": true
  }
  ```

### 2. 텍스트/문제 생성

- **POST** `/generate`
- **요청 예시**:

  ```json
  {
    "text": "다음 지문을 읽고 빈칸 추론 문제를 만들어줘...",
    "system_prompt": "문제 생성에 대한 커스텀 시스템 프롬프트 (선택)"
  }
  ```

  - `system_prompt`는 생략 가능하며, 생략 시 서버의 기본 프롬프트(`system_prompt.txt`)가 사용됩니다.

- **응답 예시**:
  ```json
  {
    "result": "{...}",
    "tokens": 123,
    "time_taken": 1.23
  }
  ```

## 커스텀 System Prompt 사용법

- 요청의 JSON에 `"system_prompt"` 필드를 추가하면 해당 프롬프트가 우선 적용됩니다.
- 예시:
  ```json
  {
    "text": "문제를 만들어줘.",
    "system_prompt": "너는 영어 문제 출제 전문가야. 반드시 JSON만 반환해."
  }
  ```

## 모니터링 도구: `monitor.py`

`monitor.py`는 서버 및 모델의 상태, 리소스 사용량(CPU, 메모리 등)을 모니터링하거나, API 응답 상태를 주기적으로 체크하는 스크립트입니다.

### 주요 기능

- `/health` 엔드포인트를 주기적으로 호출하여 서버 및 모델 상태 확인
- 시스템의 CPU, 메모리 사용률 모니터링 및 로그 저장
- 장애 발생 시 로그 기록

### 실행 방법

```bash
python monitor.py
```

### 로그

- `logs/monitor.log` 파일에 API 상태 및 시스템 리소스 사용량이 기록됩니다.

---

이 문서는 실제 `api_server_hyper.py`의 기능과 구조에 맞춰 작성되었습니다. 추가로 궁금한 점이 있으면 문의해 주세요.
