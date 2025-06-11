import requests
import json
from typing import Optional

class Llama3Chat:
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.conversation_history = []
        
    def check_server(self) -> bool:
        """서버 상태 확인"""
        try:
            response = requests.get(f"{self.api_url}/health")
            response.raise_for_status()
            status = response.json()
            return status.get("status") == "healthy" and status.get("model_loaded", False)
        except requests.exceptions.RequestException:
            return False
    
    def generate_response(self, 
                         user_input: str, 
                         max_length: int = 512,
                         temperature: float = 0.7,
                         top_p: float = 0.9) -> Optional[str]:
        """Llama-3 모델로 응답 생성"""
        try:
            # 대화 기록에 사용자 입력 추가
            self.conversation_history.append(f"사용자: {user_input}")
            
            # API 요청 데이터 준비
            data = {
                "prompt": user_input,
                "max_length": max_length,
                "temperature": temperature,
                "top_p": top_p,
                "num_return_sequences": 1
            }
            
            # API 호출
            response = requests.post(f"{self.api_url}/generate", json=data)
            response.raise_for_status()
            
            # 응답 처리
            result = response.json()
            generated_text = result["generated_text"]
            
            # 대화 기록에 모델 응답 추가
            self.conversation_history.append(f"Llama-3: {generated_text}")
            
            return generated_text
            
        except requests.exceptions.RequestException as e:
            print(f"API 호출 중 오류 발생: {e}")
            return None
    
    def print_conversation_history(self):
        """대화 기록 출력"""
        print("\n=== 대화 기록 ===")
        for message in self.conversation_history:
            print(message)
        print("================\n")

def main():
    # Llama-3 채팅 인스턴스 생성
    chat = Llama3Chat()
    
    # 서버 상태 확인
    if not chat.check_server():
        print("서버에 연결할 수 없습니다. API 서버가 실행 중인지 확인해주세요.")
        return
    
    print("Llama-3 채팅을 시작합니다. (종료하려면 'quit' 또는 'exit'를 입력하세요)")
    print("생성 파라미터를 조정하려면 'settings'를 입력하세요.")
    
    # 기본 설정
    settings = {
        "max_length": 512,  # Llama-3는 더 긴 시퀀스 지원
        "temperature": 0.7,
        "top_p": 0.9
    }
    
    while True:
        # 사용자 입력 받기
        user_input = input("\n사용자: ").strip()
        
        # 종료 명령 확인
        if user_input.lower() in ['quit', 'exit']:
            print("\n대화를 종료합니다.")
            chat.print_conversation_history()
            break
        
        # 설정 변경
        if user_input.lower() == 'settings':
            print("\n현재 설정:")
            print(f"1. 최대 길이: {settings['max_length']}")
            print(f"2. 다양성 (temperature): {settings['temperature']}")
            print(f"3. 안정성 (top_p): {settings['top_p']}")
            
            try:
                choice = input("\n변경할 설정 번호를 입력하세요 (1-3, 취소: Enter): ").strip()
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
                else:
                    print("잘못된 선택입니다.")
            except ValueError:
                print("올바른 값을 입력해주세요.")
            continue
        
        # 빈 입력 무시
        if not user_input:
            continue
        
        # 응답 생성
        response = chat.generate_response(
            user_input,
            max_length=settings['max_length'],
            temperature=settings['temperature'],
            top_p=settings['top_p']
        )
        
        if response:
            print(f"\nLlama-3: {response}")
        else:
            print("\n응답을 생성하는 중 오류가 발생했습니다.")

if __name__ == "__main__":
    main() 