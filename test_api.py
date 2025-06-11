import requests
import json
from typing import Dict, Any

def test_health_check(base_url: str = "http://localhost:8000") -> bool:
    """서버 상태 확인"""
    try:
        response = requests.get(f"{base_url}/health")
        response.raise_for_status()
        status = response.json()
        print("\n=== 서버 상태 ===")
        print(f"상태: {status['status']}")
        print(f"모델 로드됨: {status['model_loaded']}")
        print(f"모델 이름: {status['model_name']}")
        print("================")
        return status["status"] == "healthy" and status["model_loaded"]
    except requests.exceptions.RequestException as e:
        print(f"서버 상태 확인 중 오류 발생: {e}")
        return False

def generate_text(
    prompt: str,
    base_url: str = "http://localhost:8000",
    max_length: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    num_return_sequences: int = 1
) -> Dict[str, Any]:
    """텍스트 생성 API 호출"""
    try:
        # API 요청 데이터
        data = {
            "prompt": prompt,
            "max_length": max_length,
            "temperature": temperature,
            "top_p": top_p,
            "num_return_sequences": num_return_sequences
        }
        
        # API 호출
        response = requests.post(f"{base_url}/generate", json=data)
        response.raise_for_status()
        
        return response.json()
        
    except requests.exceptions.RequestException as e:
        print(f"텍스트 생성 중 오류 발생: {e}")
        return {"error": str(e)}

def main():
    # 서버 상태 확인
    if not test_health_check():
        print("서버가 정상적으로 실행되지 않았습니다.")
        return
    
    print("\nLlama-3 API 테스트를 시작합니다.")
    print("종료하려면 'quit' 또는 'exit'를 입력하세요.")
    print("설정을 변경하려면 'settings'를 입력하세요.")
    
    # 기본 설정
    settings = {
        "max_length": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "num_return_sequences": 1
    }
    
    while True:
        # 사용자 입력 받기
        user_input = input("\n프롬프트를 입력하세요: ").strip()
        
        # 종료 명령 확인
        if user_input.lower() in ['quit', 'exit']:
            print("\n테스트를 종료합니다.")
            break
        
        # 설정 변경
        if user_input.lower() == 'settings':
            print("\n현재 설정:")
            print(f"1. 최대 길이: {settings['max_length']}")
            print(f"2. 다양성 (temperature): {settings['temperature']}")
            print(f"3. 안정성 (top_p): {settings['top_p']}")
            print(f"4. 생성할 시퀀스 수: {settings['num_return_sequences']}")
            
            try:
                choice = input("\n변경할 설정 번호를 입력하세요 (1-4, 취소: Enter): ").strip()
                if not choice:
                    continue
                    
                choice = int(choice)
                if choice == 1:
                    new_value = int(input("새로운 최대 길이 (100-1024): "))
                    settings['max_length'] = max(100, min(1024, new_value))
                elif choice == 2:
                    new_value = float(input("새로운 다양성 (0.1-1.0): "))
                    settings['temperature'] = max(0.1, min(1.0, new_value))
                elif choice == 3:
                    new_value = float(input("새로운 안정성 (0.1-1.0): "))
                    settings['top_p'] = max(0.1, min(1.0, new_value))
                elif choice == 4:
                    new_value = int(input("새로운 시퀀스 수 (1-5): "))
                    settings['num_return_sequences'] = max(1, min(5, new_value))
                else:
                    print("잘못된 선택입니다.")
            except ValueError:
                print("올바른 값을 입력해주세요.")
            continue
        
        # 빈 입력 무시
        if not user_input:
            continue
        
        # 텍스트 생성
        print("\n생성 중...")
        result = generate_text(
            user_input,
            max_length=settings['max_length'],
            temperature=settings['temperature'],
            top_p=settings['top_p'],
            num_return_sequences=settings['num_return_sequences']
        )
        
        # 결과 출력
        if "error" in result:
            print(f"오류: {result['error']}")
        else:
            print("\n=== 생성된 텍스트 ===")
            print(result["generated_text"])
            print("===================")

if __name__ == "__main__":
    main()